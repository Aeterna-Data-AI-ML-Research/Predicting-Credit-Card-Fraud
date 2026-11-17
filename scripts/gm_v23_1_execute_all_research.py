#!/usr/bin/env python3
"""V23.1b — Execute remaining research probes; submit until daily quota exhausted.

Probes (board):
  1. R2 90/10 × goodpjw 0.95421 (near-peer LM)
  2. Best alex-15 residual ORTH @0.30e-4 (screened live)
  3. SINGLE sameer @0.28e-4
  4. Soft anti-K2 R2+0.5*(R2-K2)
  5. R2 95/5 × S_95417 shallow (softer than P0b)
  6. GPW_comp_ORTH @0.45e-4
Also stages CV-max private lock (no submit preference).
"""
from __future__ import annotations

import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
OUT = Path("/tmp/s6e11_v16/out")
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
FAM = np.load(OUT / "v20_1_families.npz")
NB = Path("/tmp/s6e11_nb_research")
P0 = Path("/tmp/s6e11_p0_blend/in")
ALEX15 = NB / "alex_oof15"
ALEX_PEER = NB / "peers/credit-card-fraud-lb-0-9539-shallow-tree-stack"
R2P = SUB / "sub_v22_3_LIVE_ridge_continuation.csv"
K2P = SUB / "sub_v22_5_LIVE_crest_orth4.csv"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
REPORT = LOG / "GM_V23_1B_EXECUTE_ALL_RESEARCH_REPORT.md"
N_TR, N_TE = 699960, 299984
T0 = time.time()
LINES: list[str] = []


def log(msg: str) -> None:
    s = f"[{time.time()-T0:7.1f}s] {msg}"
    print(s, flush=True)
    LINES.append(s)


def L(v):
    v = np.asarray(v, float).ravel()
    return logit((rankdata(v, method="average") - 0.5) / len(v))


def rf(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def omr(a, b):
    return (1.0 - float(spearmanr(a, b).correlation)) * 1e4


def project_out(v, B):
    v = np.asarray(v, float).ravel()
    X = np.column_stack([np.ones(len(v)), np.asarray(B, float)])
    xtx = X.T @ X
    xtx.flat[:: xtx.shape[0] + 1] += 1e-8
    return v - X @ np.linalg.solve(xtx, X.T @ v)


def bisect_vs(ref, base_L, d, target, n=28):
    d = unit(d)
    a, b = 1e-6, 1.5
    best_w, best_c, best_err = 0.0, rf(base_L), 1e9
    for _ in range(n):
        mid = 0.5 * (a + b)
        cand = rf(base_L + mid * d)
        rho = omr(cand, ref)
        err = abs(rho - target)
        if err < best_err:
            best_err, best_w, best_c = err, mid, cand
        if rho < target:
            a = mid
        else:
            b = mid
    return best_w, best_c, omr(best_c, ref)


def h2h(a, b, y, folds):
    ds = []
    for _, te in folds:
        ds.append((roc_auc_score(y[te], a[te]) - roc_auc_score(y[te], b[te])) * 1e5)
    ds = np.asarray(ds, float)
    return dict(
        mean=float(ds.mean()),
        min=float(ds.min()),
        pos=int((ds > 0).sum()),
        t=float(ds.mean() / (ds.std() + 1e-12) * np.sqrt(len(ds))),
    )


def submit_and_poll(path: Path, msg: str):
    pub = None
    try:
        r = subprocess.run(
            [KAG, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg],
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = r.stdout + "\n" + r.stderr
        log(r.stdout.strip()[:350])
        log(r.stderr.strip()[:350])
        low = out.lower()
        if r.returncode != 0 or "limit" in low or "400" in out:
            return False, None, "quota_or_error"
        for i in range(40):
            time.sleep(18 if i else 8)
            p = subprocess.run(
                [KAG, "competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
                capture_output=True,
                text=True,
                timeout=90,
            )
            for ln in p.stdout.strip().splitlines()[1:20]:
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
        log(f"submit error: {e}")
        traceback.print_exc()
        return False, None, "exception"


def stage(path: Path, ids, vec):
    v = np.asarray(vec, float).ravel()
    assert len(v) == N_TE and np.isfinite(v).all()
    pd.DataFrame({"id": ids, "is_fraud": v}).to_csv(path, index=False)
    log(f"STAGED {path.name}")


def load_test_member(name: str):
    for base in (BANK,):
        p = base / f"test_{name}.npy"
        if p.exists():
            return np.load(p).ravel()[:N_TE]
    key = f"{name}_test"
    if key in FAM.files:
        return FAM[key].ravel()[:N_TE]
    return None


def load_oof_member(name: str):
    p = BANK / f"oof_{name}.npy"
    if p.exists():
        return np.load(p).ravel()[:N_TR]
    key = f"{name}_oof"
    if key in FAM.files:
        return FAM[key].ravel()[:N_TR]
    return None


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log("=== V23.1b EXECUTE ALL remaining research ===")

    sample = pd.read_csv(R2P)
    ids = sample["id"].to_numpy()
    assert len(ids) == N_TE

    def load_sub(path):
        return pd.read_csv(path).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    r2 = load_sub(R2P)
    k2 = load_sub(K2P)
    raw_gp = load_sub(NB / "sriram_ds/reference.csv")
    raw_new = load_sub(NB / "sriram_ds/new_neural.csv")
    raw_old = load_sub(NB / "sriram_ds/old_neural.csv")
    sri = rf(L(raw_gp) + 0.15 * (L(raw_new) - L(raw_old)))
    s95417 = load_sub(P0 / "alex/submission.csv")
    gpw21 = load_sub(P0 / "gpw21/submission.csv")
    gpw_cv = load_sub(P0 / "gpw21/submission_cv.csv")
    areeb21 = load_sub(P0 / "areeb/submission.csv")

    log(
        f"peers ρ(R2): S95417={omr(s95417,r2):.3f}e-4 gpw21={omr(gpw21,r2):.3f}e-4 "
        f"areeb21={omr(areeb21,r2):.3f}e-4 gpw_cv={omr(gpw_cv,r2):.3f}e-4"
    )
    log(
        f"peer ρ(logit): R2×S95417={float(np.corrcoef(L(r2),L(s95417))[0,1]):.4f} "
        f"R2×gpw21={float(np.corrcoef(L(r2),L(gpw21))[0,1]):.4f} "
        f"gpw21×S95417={float(np.corrcoef(L(gpw21),L(s95417))[0,1]):.4f}"
    )

    y = pd.read_csv(REPO / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy()
    fold_id = pd.read_csv(BANK / "folds.csv")["fold"].to_numpy()
    folds = [(np.where(fold_id != i)[0], np.where(fold_id == i)[0]) for i in range(5)]

    # R2 OOF surrogate from available members
    mem_names = [
        "Q17_megayak_shallow_tree",
        "Q17_sameer_glm10f",
        "Q17_daoviet_gpdvae",
        "ebm_all_bag",
        "ebm_dig_bag",
        "gb_bag5",
        "glm_ohk_c01",
    ]
    Bo_cols, Bt_cols = [], []
    for n in mem_names:
        oo, ot = load_oof_member(n), load_test_member(n)
        if oo is None or ot is None:
            if n.startswith("Q17_") and f"{n}_oof" in FAM.files:
                oo, ot = FAM[f"{n}_oof"].ravel()[:N_TR], FAM[f"{n}_test"].ravel()[:N_TE]
        if oo is not None and ot is not None:
            Bo_cols.append(unit(L(oo)))
            Bt_cols.append(unit(L(ot)))
    o_new = pd.read_csv(NB / "yekenot_pytabkit/oof_preds.csv")["is_fraud"].to_numpy()[:N_TR]
    o_old = pd.read_csv(NB / "yekenot_pytorch/oof_preds.csv")["is_fraud"].to_numpy()[:N_TR]
    Bo_cols += [unit(L(o_new)), unit(L(o_old))]
    Bt_cols += [unit(L(raw_new)), unit(L(raw_old))]
    Bo = np.column_stack(Bo_cols)
    Bt = np.column_stack(Bt_cols)
    lr = Ridge(alpha=10.0, fit_intercept=True).fit(Bt, L(r2))
    z_r2 = Bo @ lr.coef_ + lr.intercept_
    log(f"R2 OOF surrogate AUC={roc_auc_score(y, z_r2):.6f}")

    B_macro_t = np.column_stack(
        [
            unit(L(sri)),
            unit(L(raw_gp)),
            unit(L(raw_new)),
            unit(L(raw_old)),
            unit(L(FAM["Q17_megayak_shallow_tree_test"].ravel()[:N_TE])),
            unit(L(FAM["Q17_sameer_glm10f_test"].ravel()[:N_TE])),
        ]
    )
    B_macro_o = np.column_stack(
        [
            unit(z_r2),
            unit(L(o_new)),
            unit(L(o_old)),
            unit(L(FAM["Q17_megayak_shallow_tree_oof"].ravel()[:N_TR])),
            unit(L(FAM["Q17_sameer_glm10f_oof"].ravel()[:N_TR])),
        ]
    )

    # --- Screen alex-15 residuals ---
    alex_ids = []
    for p in list(ALEX_PEER.glob("test_*.npy")) + list(ALEX15.glob("test_*.npy")):
        aid = p.name.replace("test_", "").replace(".npy", "")
        if aid not in alex_ids:
            alex_ids.append(aid)
    # skip knn junk
    alex_ids = [a for a in alex_ids if a != "989e37e5bd1f1063"]
    log(f"alex members to screen: {len(alex_ids)}")

    def find_pair(aid):
        for root in (ALEX_PEER, ALEX15):
            ot_p, oo_p = root / f"test_{aid}.npy", root / f"oof_{aid}.npy"
            if ot_p.exists() and oo_p.exists():
                ot = np.load(ot_p).ravel()
                oo = np.load(oo_p).ravel()
                if len(ot) >= N_TE and len(oo) >= N_TR:
                    return oo[:N_TR], ot[:N_TE]
        # fam
        for pref in (f"alex_{aid[:8]}", aid):
            if f"{pref}_oof" in FAM.files and f"{pref}_test" in FAM.files:
                return FAM[f"{pref}_oof"].ravel()[:N_TR], FAM[f"{pref}_test"].ravel()[:N_TE]
        return None, None

    alex_rows = []
    for aid in alex_ids:
        oo, ot = find_pair(aid)
        if oo is None:
            continue
        oo_u = unit(project_out(unit(L(oo)), B_macro_o))
        ot_u = unit(project_out(unit(L(ot)), B_macro_t))
        # sign vs R2 surrogate
        if roc_auc_score(y[folds[0][1]], z_r2[folds[0][1]] + 0.006 * oo_u[folds[0][1]]) < roc_auc_score(
            y[folds[0][1]], z_r2[folds[0][1]] - 0.006 * oo_u[folds[0][1]]
        ):
            oo_u, ot_u = -oo_u, -ot_u
        per = h2h(z_r2 + 0.006 * oo_u, z_r2, y, folds)
        alex_rows.append({"id": aid, "per": per, "oo": oo_u, "ot": ot_u})
        if per["pos"] >= 4:
            log(f"  alex {aid[:8]} pos={per['pos']}/5 t={per['t']:.2f} m={per['mean']:+.2f}")

    alex_rows = sorted(alex_rows, key=lambda r: (-r["per"]["pos"], -r["per"]["t"], -r["per"]["mean"]))
    best_alex = alex_rows[0] if alex_rows else None
    if best_alex:
        log(
            f"BEST alex {best_alex['id'][:12]} pos={best_alex['per']['pos']}/5 "
            f"t={best_alex['per']['t']:.2f} m={best_alex['per']['mean']:+.2f}"
        )

    # --- GPW compressed orth dir ---
    gpw_w = {
        "Q17_megayak_shallow_tree": +0.2108,
        "Q17_sameer_glm10f": +0.0905,
        "Q17_daoviet_gpdvae": +0.0118,
        "Q17_alex_glm_gen": -0.0716,
        "ebm_all_bag": +0.1680,
        "ebm_dig_bag": +0.1393,
        "ebm_dig": -0.1256,
        "gb_bag5": -0.0826,
        "gbn_bag5_w512": -0.0290,
        "glm_ohk_c01": +0.1127,
        "glm_dl_s50_bag": +0.1079,
    }
    d_gpw = np.zeros(N_TE)
    for n, w in gpw_w.items():
        ot = load_test_member(n)
        if ot is None and n.startswith("Q17_") and f"{n}_test" in FAM.files:
            ot = FAM[f"{n}_test"].ravel()[:N_TE]
        if ot is not None:
            d_gpw = d_gpw + w * unit(L(ot))
    d_gpw_orth = unit(project_out(unit(d_gpw), B_macro_t))

    sameer_t = unit(L(FAM["Q17_sameer_glm10f_test"].ravel()[:N_TE]))
    sameer_o = unit(L(FAM["Q17_sameer_glm10f_oof"].ravel()[:N_TR]))
    if roc_auc_score(y[folds[0][1]], z_r2[folds[0][1]] + 0.006 * sameer_o[folds[0][1]]) < roc_auc_score(
        y[folds[0][1]], z_r2[folds[0][1]] - 0.006 * sameer_o[folds[0][1]]
    ):
        sameer_o, sameer_t = -sameer_o, -sameer_t
    sameer_t = unit(project_out(sameer_t, B_macro_t))

    # Build all candidates
    cands = []

    def add(tag, vec, note=""):
        cands.append({"tag": tag, "vec": np.asarray(vec, float).ravel(), "note": note})

    add(
        "R2_gpw21_90_10",
        rf(0.90 * L(r2) + 0.10 * L(gpw21)),
        f"ρ(R2)={omr(rf(0.90*L(r2)+0.10*L(gpw21)), r2):.3f}e-4",
    )
    add(
        "R2_S95417_95_5",
        rf(0.95 * L(r2) + 0.05 * L(s95417)),
        f"ρ(R2)={omr(rf(0.95*L(r2)+0.05*L(s95417)), r2):.3f}e-4",
    )
    add(
        "gpw21_S95417_90_10",
        rf(0.90 * L(gpw21) + 0.10 * L(s95417)),
        "Areeb recipe on available peers (gpw21≈0.95421 × S95417)",
    )
    w, c, rho = bisect_vs(r2, L(r2), sameer_t, 0.28)
    add("SAMEER_ORTH_0p28e4", c, f"ρ(R2)={rho:.3f}e-4")
    w, c, rho = bisect_vs(r2, L(r2), d_gpw_orth, 0.45)
    add("GPW_comp_ORTH_0p45e4", c, f"ρ(R2)={rho:.3f}e-4")
    add("SOFT_antiK2_0p5", rf(L(r2) + 0.5 * (L(r2) - L(k2))), f"ρ(R2)={omr(rf(L(r2)+0.5*(L(r2)-L(k2))), r2):.3f}e-4")

    if best_alex is not None:
        w, c, rho = bisect_vs(r2, L(r2), best_alex["ot"], 0.30)
        add(
            f"ALEX15_{best_alex['id'][:8]}_ORTH_0p30",
            c,
            f"ρ(R2)={rho:.3f} nested={best_alex['per']['pos']}/5 t={best_alex['per']['t']:.2f}",
        )

    # CV-max private lock: larger GPW orth + slight gpw_cv blend
    w, c_cv, rho = bisect_vs(r2, L(r2), d_gpw_orth, 0.55)
    c_cv = rf(0.97 * L(c_cv) + 0.03 * L(gpw_cv))
    path_cv = SUB / "sub_v23_1b_CVMAX_PRIVATE_LOCK_NOSUBMIT.csv"
    stage(path_cv, ids, c_cv)
    log(f"CV lock ρ(R2)={omr(c_cv,r2):.3f}e-4 (not submitted)")

    # Stage all
    staged = []
    for c in cands:
        path = SUB / f"sub_v23_1b_{c['tag']}.csv"
        stage(path, ids, c["vec"])
        staged.append({**c, "path": path, "rho_r2": omr(c["vec"], r2), "rho_sri": omr(c["vec"], sri)})
        log(f"  {c['tag']}: ρ(R2)={staged[-1]['rho_r2']:.3f}e-4 ρ(sri)={staged[-1]['rho_sri']:.3f}e-4 | {c['note']}")

    # Submit order: diversity first (user: execute all whether lift)
    submit_order = [
        "R2_gpw21_90_10",
        "ALEX15_",  # prefix match
        "SAMEER_ORTH_0p28e4",
        "gpw21_S95417_90_10",
        "R2_S95417_95_5",
        "SOFT_antiK2_0p5",
        "GPW_comp_ORTH_0p45e4",
    ]

    def pick(order_key):
        for s in staged:
            if order_key.endswith("_"):
                if s["tag"].startswith(order_key[:-1]) or order_key[:-1] in s["tag"]:
                    return s
            elif s["tag"] == order_key:
                return s
        return None

    results = []
    for key in submit_order:
        s = pick(key)
        if s is None:
            log(f"SKIP missing {key}")
            continue
        msg = f"v23.1b {s['tag']} | ρ(R2)={s['rho_r2']:.3f}e-4 ρ(sri)={s['rho_sri']:.3f}e-4 | {s['note']}"[:480]
        log(f"SUBMIT {msg}")
        ok, pub, status = submit_and_poll(s["path"], msg)
        results.append({"tag": s["tag"], "pub": pub, "ok": ok, "status": status, "rho_r2": s["rho_r2"]})
        log(f"DONE {s['tag']} pub={pub} status={status}")
        if status == "quota_or_error":
            log("STOP: quota exhausted or submit error")
            break
        time.sleep(8)

    best = max((r["pub"] for r in results if r.get("pub") is not None), default=None)
    body = [
        "# GM V23.1b Execute-All Research Report",
        "",
        f"ATH reference `0.95426` · elapsed {time.time()-T0:.0f}s · CV lock `{path_cv.name}`",
        "",
        "## Live submits",
        "| tag | Public | ρ(R2) e-4 | status |",
        "|---|---:|---:|---|",
    ]
    for r in results:
        body.append(f"| `{r['tag']}` | **{r.get('pub')}** | {r.get('rho_r2', float('nan')):.3f} | {r.get('status')} |")
    body += [
        "",
        f"- Best among this batch: **{best}**",
        "",
        "## Staged (all research CSVs)",
        *[f"- `{s['path'].name}` ρ(R2)={s['rho_r2']:.3f}e-4 — {s['note']}" for s in staged],
        "",
        "## Log",
        "```",
        *LINES,
        "```",
        "",
    ]
    REPORT.write_text("\n".join(body) + "\n")
    OUT.joinpath("v23_1b_execute_all_meta.json").write_text(
        json.dumps(
            {
                "results": results,
                "staged": [{k: v for k, v in s.items() if k != "vec"} for s in staged],
                "best_alex": None
                if not best_alex
                else {"id": best_alex["id"], **best_alex["per"]},
                "elapsed": time.time() - T0,
            },
            indent=2,
            default=float,
        )
    )
    log(f"REPORT {REPORT} best={best}")


if __name__ == "__main__":
    main()
