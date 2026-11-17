#!/usr/bin/env python3
"""V24.2b — Fix LightGBM (inner val ES) + non-circular nested gate vs ATH."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import time
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

REPO = Path(__file__).resolve().parents[1]
DATA, SUB = REPO / "data", REPO / "submissions"
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
OUT = Path("/tmp/s6e11_v16/out")
LOG = REPO / "logs/overnight"
ATH = SUB / "sub_v24_SRI440_pub.csv"
PY = "/tmp/s6e11_v23_venv/bin/python"
KAG = [PY, "-m", "kaggle"]
N_TR, N_TE = 699960, 299984
T0 = time.time()
LINES: list[str] = []


def log(msg: str) -> None:
    s = f"[{time.time()-T0:7.1f}s] {msg}"
    print(s, flush=True)
    LINES.append(s)


def L(v):
    v = np.asarray(v, float).ravel()
    return logit(np.clip((rankdata(v, method="average") - 0.5) / len(v), 1e-6, 1 - 1e-6))


def rf(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def omr(a, b):
    return (1.0 - float(spearmanr(a, b).correlation)) * 1e4


def auc(y, p):
    return float(roc_auc_score(y, p))


def rblend(parts, w):
    w = np.asarray(w, float)
    w = w / w.sum()
    return rf(sum(wi * L(p) for wi, p in zip(w, parts)))


def h2h(a, b, y, folds):
    ds = np.array([(auc(y[te], a[te]) - auc(y[te], b[te])) * 1e5 for _, te in folds])
    return dict(
        mean=float(ds.mean()),
        min=float(ds.min()),
        pos=int((ds > 0).sum()),
        t=float(ds.mean() / (ds.std(ddof=1) + 1e-12) * np.sqrt(len(ds))),
    )


def submit_and_poll(path: Path, msg: str):
    try:
        r = subprocess.run(
            KAG + ["competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg],
            capture_output=True,
            text=True,
            timeout=180,
        )
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        log(r.stdout.strip()[:350])
        log(r.stderr.strip()[:250])
        if "successfully submitted" not in out.lower():
            return False, None, "quota_or_error"
        for i in range(40):
            time.sleep(18 if i else 8)
            p = subprocess.run(
                KAG + ["competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
                capture_output=True,
                text=True,
                timeout=90,
            )
            for ln in p.stdout.splitlines()[1:15]:
                if path.name not in ln:
                    continue
                log(f"poll {ln[:220]}")
                if "COMPLETE" in ln.upper():
                    for tok in reversed([c.strip() for c in ln.split(",")]):
                        try:
                            return True, float(tok), "ok"
                        except ValueError:
                            continue
        return True, None, "timeout"
    except Exception as e:
        log(f"submit err {e}")
        traceback.print_exc()
        return False, None, "exception"


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    log("=== V24.2b fix LGB + gates ===")
    spec = importlib.util.spec_from_file_location("v242", REPO / "scripts" / "gm_v24_2_native_rf_gbm.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["is_fraud"].to_numpy(float)
    ids = test["id"].to_numpy()
    fold_id = pd.read_csv(BANK / "folds.csv")["fold"].to_numpy()
    folds = [(np.where(fold_id != i)[0], np.where(fold_id == i)[0]) for i in range(5)]

    tr_f = mod.build_base_frame(train)
    te_f = mod.build_base_frame(test)
    for c in mod.CATS:
        oof_e, te_e = mod.lofo_target_encode(tr_f[c], y, folds, te_f[c], prior=30.0)
        tr_f[f"te_{c}"] = oof_e
        te_f[f"te_{c}"] = te_e
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr_f[[f"ord_{c}" for c in mod.CATS]] = ord_enc.fit_transform(tr_f[mod.CATS])
    te_f[[f"ord_{c}" for c in mod.CATS]] = ord_enc.transform(te_f[mod.CATS])
    cols = [c for c in tr_f.columns if c not in mod.CATS]
    Xtr = np.nan_to_num(tr_f[cols].to_numpy(float))
    Xte = np.nan_to_num(te_f[cols].to_numpy(float))

    z = np.load(OUT / "v24_2_native_rf_gbm.npz", allow_pickle=True)
    oof_rf, oof_hgb = z["oof_rf"], z["oof_hgb"]
    te_rf, te_hgb = z["test_rf"], z["test_hgb"]
    log(f"RF OOF={auc(y,oof_rf):.6f} HGB OOF={auc(y,oof_hgb):.6f}")

    log("Retrain LightGBM with inner-holdout early stopping…")
    oof_l = np.zeros(N_TR)
    te_l = np.zeros(N_TE)
    fa = []
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=0.5,
        is_unbalance=True,
        verbosity=-1,
        seed=42,
        n_jobs=-1,
    )
    for i, (tr, te) in enumerate(folds):
        rng = np.random.default_rng(42 + i)
        perm = rng.permutation(tr)
        n_val = max(8000, int(0.1 * len(tr)))
        va, tr2 = perm[:n_val], perm[n_val:]
        dtr = lgb.Dataset(Xtr[tr2], y[tr2])
        dva = lgb.Dataset(Xtr[va], y[va], reference=dtr)
        bst = lgb.train(
            params,
            dtr,
            num_boost_round=4000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof_l[te] = bst.predict(Xtr[te], num_iteration=bst.best_iteration)
        te_l += bst.predict(Xte, num_iteration=bst.best_iteration) / 5.0
        a = auc(y[te], oof_l[te])
        fa.append(a)
        log(f"  LGB fold{i} AUC={a:.6f} iter={bst.best_iteration}")
    log(f"LGB OOF={auc(y,oof_l):.6f} mean_fold={float(np.mean(fa)):.6f}")

    best_w, best_a = (0.25, 0.50, 0.25), -1.0
    for wr in np.linspace(0.10, 0.45, 8):
        for wh in np.linspace(0.25, 0.70, 10):
            wl = 1.0 - wr - wh
            if wl < 0.05 or wl > 0.50:
                continue
            a = auc(y, rblend([oof_rf, oof_hgb, oof_l], [wr, wh, wl]))
            if a > best_a:
                best_a, best_w = a, (float(wr), float(wh), float(wl))
    blend_o = rblend([oof_rf, oof_hgb, oof_l], best_w)
    blend_t = rblend([te_rf, te_hgb, te_l], best_w)
    log(f"blend OOF={auc(y,blend_o):.6f} w=RF/HGB/LGB={best_w}")

    ath = pd.read_csv(ATH).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    Bo, Bt = [], []
    for name in (
        "gb_bag5",
        "gb_bag10",
        "lgb_diglead_l3",
        "lgb_diglead_l3_bag",
        "realmlp_v1",
        "glm_dl",
        "ebm_all_bag",
        "gbn_bag5_w512",
    ):
        op, tp = BANK / f"oof_{name}.npy", BANK / f"test_{name}.npy"
        if op.exists() and tp.exists():
            Bo.append(unit(L(np.load(op).ravel()[:N_TR])))
            Bt.append(unit(L(np.load(tp).ravel()[:N_TE])))
    Bo_m, Bt_m = np.column_stack(Bo), np.column_stack(Bt)
    lr = Ridge(alpha=10.0).fit(Bt_m, L(ath))
    z_ath = Bo_m @ lr.coef_ + lr.intercept_
    log(f"ATH surrogate (bank-only) AUC={auc(y,z_ath):.6f}")

    gates = {}
    for name, oo, tt in (
        ("rf", oof_rf, te_rf),
        ("hgb", oof_hgb, te_hgb),
        ("lgb", oof_l, te_l),
        ("blend", blend_o, blend_t),
    ):
        d = unit(L(oo))
        d = unit(d - np.dot(d, unit(z_ath)) * unit(z_ath))
        te0 = folds[0][1]
        sgn = +1
        if auc(y[te0], z_ath[te0] + 0.01 * d[te0]) < auc(y[te0], z_ath[te0] - 0.01 * d[te0]):
            sgn = -1
        d = sgn * d
        per = h2h(z_ath + 0.01 * d, z_ath, y, folds)
        gates[name] = {**per, "sgn": sgn}
        log(
            f"H2H ATH+orth({name}): {per['pos']}/5 t={per['t']:.2f} m={per['mean']:+.2f} "
            f"min={per['min']:+.2f} solo={auc(y,oo):.5f} ρ(ATH)={omr(tt,ath):.1f}e-4"
        )

    gate = gates["blend"]["pos"] >= 4 and gates["blend"]["mean"] > 0 and gates["blend"]["t"] >= 2.0
    log(f"GATE blend-orth: {gate}")

    np.savez_compressed(
        OUT / "v24_2_native_rf_gbm.npz",
        oof_rf=oof_rf,
        oof_hgb=oof_hgb,
        oof_lgb=oof_l,
        oof_blend=blend_o,
        test_rf=te_rf,
        test_hgb=te_hgb,
        test_lgb=te_l,
        test_blend=blend_t,
        blend_w=np.array(best_w),
        fold_aucs_lgb=np.array(fa),
    )

    # stage
    cands = {
        "native_rf": rf(te_rf),
        "native_hgb": rf(te_hgb),
        "native_lgb": rf(te_l),
        "native_blend": rf(blend_t),
        "ATH_x_nativeBlend_0.05": rf(0.95 * L(ath) + 0.05 * L(blend_t)),
        "ATH_x_nativeBlend_0.10": rf(0.90 * L(ath) + 0.10 * L(blend_t)),
        "ATH_x_nativeHGB_0.08": rf(0.92 * L(ath) + 0.08 * L(te_hgb)),
    }
    # orth inject
    d_te = unit(L(blend_t))
    d_te = unit(d_te - np.dot(d_te, unit(L(ath))) * unit(L(ath))) * gates["blend"]["sgn"]
    a, b = 1e-6, 0.8
    best_c, best_err = ath.copy(), 1e9
    for _ in range(30):
        mid = 0.5 * (a + b)
        cand = rf(L(ath) + mid * d_te)
        err = abs(omr(cand, ath) - 0.35)
        if err < best_err:
            best_err, best_c = err, cand
        if omr(cand, ath) < 0.35:
            a = mid
        else:
            b = mid
    cands["ATH_orth_nativeBlend_0p35e4"] = best_c

    staged = []
    for tag, vec in cands.items():
        path = SUB / f"sub_v24_2_{tag}.csv"
        pd.DataFrame({"id": ids, "is_fraud": vec}).to_csv(path, index=False)
        staged.append({"tag": tag, "path": path, "rho_ath": omr(vec, ath)})
        log(f"STAGED {path.name} ρ(ATH)={staged[-1]['rho_ath']:.3f}e-4")

    # submit if quota: prefer HGB cross-family (strongest solo) then blend 5-10%
    order = ["ATH_x_nativeHGB_0.08", "ATH_x_nativeBlend_0.05", "ATH_x_nativeBlend_0.10"]
    if gate:
        order = ["ATH_orth_nativeBlend_0p35e4"] + order
    live = []
    by = {s["tag"]: s for s in staged}
    for tag in order:
        s = by[tag]
        msg = f"v24.2b {tag} | ρ(ATH)={s['rho_ath']:.3f}e-4 | RF={auc(y,oof_rf):.4f} HGB={auc(y,oof_hgb):.4f} LGB={auc(y,oof_l):.4f} blend={auc(y,blend_o):.4f}"
        log(f"SUBMIT {msg}")
        ok, pub, status = submit_and_poll(s["path"], msg)
        live.append({"tag": tag, "pub": pub, "status": status})
        log(f"DONE {tag} pub={pub} status={status}")
        if status == "quota_or_error":
            break
        time.sleep(5)

    report = LOG / "GM_V24_2_NATIVE_RF_GBM_REPORT.md"
    body = [
        "# GM V24.2 Native RF & Gradient Boosting (expert DS)",
        "",
        f"Elapsed repair+train LGB phase in log · ATH ref **0.95440**",
        "",
        "## OOF results",
        "",
        "| Model | OOF AUC | Notes |",
        "|---|---:|---|",
        f"| **RandomForest** (sklearn native) | **{auc(y,oof_rf):.6f}** | 350 trees, depth22, max_samples0.35, balanced_subsample |",
        f"| **HistGradientBoosting** (sklearn native) | **{auc(y,oof_hgb):.6f}** | lr0.05, early stop, balanced |",
        f"| **LightGBM** (fixed inner-val ES) | **{auc(y,oof_l):.6f}** | folds mean {float(np.mean(fa)):.6f} |",
        f"| Rank-blend {best_w} | **{auc(y,blend_o):.6f}** | |",
        "",
        "## Nested gate vs bank-only ATH surrogate",
        "",
        "| residual | pos | t | meanΔe5 |",
        "|---|---:|---:|---:|",
    ]
    for k, g in gates.items():
        body.append(f"| `{k}` | {g['pos']}/5 | {g['t']:.2f} | {g['mean']:+.2f} |")
    body += [
        "",
        f"- Blend orth gate: **{gate}**",
        "",
        "## Live",
        "",
        "| tag | Public | status |",
        "|---|---:|---|",
    ]
    for r in live:
        body.append(f"| `{r['tag']}` | **{r.get('pub')}** | {r.get('status')} |")
    body += ["", "## Log", "```", *LINES, "```", ""]
    report.write_text("\n".join(body) + "\n")
    OUT.joinpath("v24_2b_meta.json").write_text(
        json.dumps(
            {
                "oof": {"rf": auc(y, oof_rf), "hgb": auc(y, oof_hgb), "lgb": auc(y, oof_l), "blend": auc(y, blend_o)},
                "blend_w": best_w,
                "gates": gates,
                "gate": gate,
                "live": live,
            },
            indent=2,
            default=float,
        )
    )
    log(f"REPORT {report}")


if __name__ == "__main__":
    main()
