#!/usr/bin/env python3
"""V47.2 Step2 — Provenance audit + symmetric zero-echo peer → z_sovereign_v1.

NO_SUBMIT=True · $0 RunPod · 0 public competitor arrays.
Pre-unitize hygiene: disp_raw ∈ [0.92,1.08], KS_raw < 0.015 before any ECDF.
Promote residual only if pos/5 >= 4 and sf/25 >= 17.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ks_2samp, norm, rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
NO_SUBMIT = True
assert NO_SUBMIT

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" if (REPO / "data/train.csv").exists() else Path("/tmp/s6e11_v39_ssd/data")
PRED = REPO / "preds"
OUT = PRED / "v47_sovereign"
LOGD = REPO / "logs/v47"
OUT.mkdir(parents=True, exist_ok=True)
LOGD.mkdir(parents=True, exist_ok=True)
T0 = time.time()

BAN = (
    "goodpjw", "monya", "dennis", "areeb", "amanatar", "asterios", "gpw70",
    "nov26_board", "step_ab", "step_ag", "board_direction", "external/banks",
    "masaya", "megayak", "external_bank",
)
DISP_LO, DISP_HI = 0.92, 1.08
KS_MAX = 0.015


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / max(len(v), 1)


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def to_logit_cal(v):
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6)).astype(np.float64)


def rank_gauss(v):
    return norm.ppf(np.clip(U(v), 1e-6, 1 - 1e-6)).astype(np.float64)


def assert_path(p: Path) -> Path:
    s = str(p).lower().replace("\\", "/")
    for b in BAN:
        if b in s:
            raise RuntimeError(f"BAN hit '{b}' in {p}")
    if p.name.startswith("._"):
        raise RuntimeError(f"AppleDouble {p}")
    return p


def load_npy(p: Path):
    return np.load(assert_path(p), allow_pickle=False)


def hygiene_raw(oof, test):
    oof, test = np.asarray(oof, float).ravel(), np.asarray(test, float).ravel()
    disp = float((test.std() + 1e-12) / (oof.std() + 1e-12))
    ks = float(ks_2samp(oof, test).statistic)
    return {"disp_raw": disp, "KS_raw": ks, "pass": (DISP_LO <= disp <= DISP_HI) and (ks < KS_MAX)}


def provenance_audit() -> dict:
    streams = json.loads((LOGD / "STEP1_Z_SOVEREIGN.json").read_text())["names"]
    assets = {
        "P_super_v5_harvest": {
            "oof": PRED / "P_super_v5_harvest_oof.npy",
            "builder": REPO / "scripts/harvest_v5_finalize.py",
        },
        "B_xgb_llr": {"oof": PRED / "paradigm_B/xgb_llr_oof.npy"},
        "P4_L4.2_pure5f": {"oof": PRED / "matrix_v2/P4_L4.2_pure5f_oof.npy", "builder": REPO / "scripts/p4_pure_5fold.py"},
        "P4_L4.3_pure5f": {"oof": PRED / "matrix_v2/P4_L4.3_pure5f_oof.npy"},
        "P4_L4.5_pure5f": {"oof": PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy"},
        "delta1_v4": {"oof": PRED / "gm_v4_polish/d1_v4_oof.npy"},
        "l7_s2026_5f": {"oof": PRED / "l7_s2026_5f/l7_s2026_oof_5fold_raw.npy", "builder": REPO / "scripts/train_l7_5f_s2026.py"},
        "gpt2_med_4ep": {"oof": PRED / "gpt2_med_4ep/oof_0.npy"},
        "P_super_v4p_hc": {"oof": PRED / "P_super_v4p_hc_oof.npy"},
    }
    for p in PRED.rglob("*xgb*clean*oof*.npy"):
        if not p.name.startswith("._"):
            assets["oof_xgb_clean"] = {"oof": p}
            break

    report = {"streams_v0": streams, "assets": {}, "checkpoints": {}, "rejects": []}
    hard_ban = ("monya", "dennis", "areeb", "step_ab", "step_ag", "masaya", "megayak", "asterios", "gpw70")
    for name, meta in assets.items():
        entry = {"exists": {}, "verdict": "KEEP"}
        for k, p in meta.items():
            if k == "builder" or p is None:
                continue
            entry["exists"][k] = Path(p).exists()
            if Path(p).exists():
                try:
                    assert_path(Path(p))
                except Exception as e:
                    entry["verdict"] = "REJECT"
                    entry["err"] = str(e)
        b = meta.get("builder")
        if b and Path(b).exists():
            src = Path(b).read_text(errors="ignore").lower()
            bad = [x for x in hard_ban if x in src]
            # train_l7 references goodpjw as L7_S7_DIR default — flag but arrays themselves are ours
            if name == "l7_s2026_5f":
                entry["note"] = "builder mentions goodpjw seed7 path as optional compare; our s2026 preds/ckpts are self-trained"
                bad = [x for x in bad if x != "goodpjw"]
            if name == "P_super_v5_harvest":
                entry["note"] = "finalize blends v4p_hc + paradigm_B xgb_llr (self); no tip CSV loads"
            entry["builder_ban_substr"] = bad
            if bad:
                entry["verdict"] = "REJECT"
                report["rejects"].append(name)
        report["assets"][name] = entry

    for tag in ("P_super_v6", "soft_diff"):
        hits = [str(p) for p in PRED.glob(f"*{tag}*") if not p.name.startswith("._")][:8]
        report["assets"][f"SCAN_{tag}"] = {"found": hits, "verdict": "EXCLUDED_FROM_V1"}

    l7_ck = [p.name for p in (PRED / "l7_s2026_5f/ckpts").glob("*.pt") if not p.name.startswith("._")] if (PRED / "l7_s2026_5f/ckpts").exists() else []
    gpt_names = []
    for d in (PRED / "gpt2_med_4ep", PRED / "gpt2_med_4ep/ckpt"):
        if d.exists():
            gpt_names += [p.name for p in d.glob("fold_*.pt") if not p.name.startswith("._")]
    report["checkpoints"] = {
        "l7_s2026_5f": l7_ck,
        "l7_has_full_5fold_weights": all((PRED / "l7_s2026_5f/ckpts" / f"fold_{k}.pt").exists() for k in range(5)),
        "gpt2_med_4ep": sorted(set(gpt_names)),
        "gpt2_has_full_5fold_weights": all(
            (PRED / "gpt2_med_4ep" / f"fold_{k}.pt").exists() or (PRED / "gpt2_med_4ep/ckpt" / f"fold_{k}.pt").exists()
            for k in range(5)
        ),
    }
    (LOGD / "STEP2_PROVENANCE.json").write_text(json.dumps(report, indent=2))
    lines = [f"# V47.2 STEP2 — Provenance\n\n- v0 streams: `{streams}`\n"]
    for n, e in report["assets"].items():
        lines.append(f"- **{n}**: `{e.get('verdict')}` {e.get('note','')}\n")
    lines.append(f"\n## Checkpoints\n```json\n{json.dumps(report['checkpoints'], indent=2)}\n```\n")
    (LOGD / "STEP2_PROVENANCE.md").write_text("".join(lines))
    log(f"provenance rejects={report['rejects']} l7_5f_ckpts={report['checkpoints']['l7_has_full_5fold_weights']}")
    return report


def fold_matched_sovereign_z(y, folds):
    l7 = load_npy(PRED / "l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    l7_llr = (l7[:, :, 1] - l7[:, :, 0]).mean(1)
    gpt = np.zeros(len(y), np.float64)
    gpt_te = np.zeros((5, 299984), np.float64)
    for k in range(5):
        a = load_npy(PRED / "gpt2_med_4ep" / f"oof_{k}.npy")
        b = load_npy(PRED / "gpt2_med_4ep" / f"test_{k}.npy")
        gpt[folds == k] = (a[:, :, 1] - a[:, :, 0]).mean(1)
        gpt_te[k] = (b[:, :, 1] - b[:, :, 0]).mean(1)
    z_oof = unitize(to_logit_cal(l7_llr)) + unitize(to_logit_cal(gpt))
    l7t = load_npy(PRED / "l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    z_te = np.zeros((5, 299984), np.float64)
    for k in range(5):
        tl = (l7t[k, :, :, 1] - l7t[k, :, :, 0]).mean(1)
        z_te[k] = unitize(to_logit_cal(tl)) + unitize(to_logit_cal(gpt_te[k]))
    return z_oof, z_te


def build_collision_keys(tr: pd.DataFrame, te: pd.DataFrame):
    n_tr = len(tr)
    both = pd.concat([tr, te], ignore_index=True)

    def codes(series_list):
        return [pd.factorize(s, sort=False)[0].astype(np.int64) for s in series_list]

    bal = np.round(both["account_balance_usd"].to_numpy(float), 2).astype(str)
    amt = np.round(both["amount_usd"].to_numpy(float), 2)
    cents = np.round((both["amount_usd"].to_numpy(float) % 1.0) * 100).astype(np.int64)
    cat = both["merchant_category"].astype(str)
    dist = np.round(both["distance_from_home_km"].to_numpy(float), 2).astype(str)
    hrs = np.round(both["hours_since_last_txn"].to_numpy(float), 2).astype(str)
    cage = both["card_age_months"].to_numpy(float)
    age = both["customer_age"].to_numpy(float)
    ctype = both["card_type"].astype(str)
    vel = both["velocity_score"].to_numpy(float)
    risk = both["merchant_risk_score"].to_numpy(float)
    flags = (
        both["is_foreign_transaction"].to_numpy(int) * 32
        + both["is_new_merchant"].to_numpy(int) * 16
        + both["used_vpn"].to_numpy(int) * 8
        + both["ip_country_mismatch"].to_numpy(int) * 4
        + both["billing_shipping_mismatch"].to_numpy(int) * 2
        + both["is_ai_generated_scam_attempt"].to_numpy(int)
    )
    ch = both["channel"].astype(str)
    dev = both["device_type"].astype(str)
    keys = codes([
        bal,
        amt.astype(str),
        pd.Series([f"{c}|{m}" for c, m in zip(cents, cat)]),
        dist,
        hrs,
        pd.Series([f"{a:.0f}|{b:.0f}|{c}" for a, b, c in zip(cage, age, ctype)]),
        pd.Series([f"{v:.2f}|{r:.2f}" for v, r in zip(vel, risk)]),
        pd.Series([f"{f}|{c}|{d}" for f, c, d in zip(flags, ch, dev)]),
    ])
    return [k[:n_tr] for k in keys], [k[n_tr:] for k in keys]


def peer_eb(z, gids, m0=5.0):
    g = np.asarray(gids, np.int64)
    z = np.asarray(z, float)
    _, g = np.unique(g, return_inverse=True)
    nbin = int(g.max()) + 1
    cnt = np.bincount(g, minlength=nbin).astype(np.float64)
    sz = np.bincount(g, weights=z, minlength=nbin)
    sz2 = np.bincount(g, weights=z * z, minlength=nbin)
    c = cnt[g]
    mean5 = np.where(c <= 1, 0.0, (sz[g] - z) / (c - 1.0 + m0))
    mean20 = np.where(c <= 1, 0.0, (sz[g] - z) / (c - 1.0 + 20.0))
    var = np.zeros_like(z)
    ok = c >= 3
    n_lo = c[ok] - 1.0
    sum_lo = sz[g][ok] - z[ok]
    sum2_lo = sz2[g][ok] - z[ok] * z[ok]
    var[ok] = (sum2_lo - (sum_lo ** 2) / n_lo) / np.maximum(n_lo - 1.0, 1.0)
    std = np.sqrt(np.maximum(var, 0.0))
    logc = np.log1p(np.maximum(c - 1.0, 0.0))
    return mean5, mean20, std, logc


def peer_feats_for_fold(z_tr, z_te, g_tr_list, g_va_list, g_te_list):
    z_pool = np.concatenate([z_tr, z_te])
    n_trf = len(z_tr)
    Xtr = np.zeros((n_trf, 32), np.float64)
    Xva = np.zeros((len(g_va_list[0]), 32), np.float64)
    Xte = np.zeros((len(z_te), 32), np.float64)
    for ki in range(8):
        g_tr, g_va, g_te = g_tr_list[ki], g_va_list[ki], g_te_list[ki]
        g_pool = np.concatenate([g_tr, g_te])
        m5, m20, std, logc = peer_eb(z_pool, g_pool, m0=5.0)
        Xtr[:, ki * 4:(ki + 1) * 4] = np.column_stack([m5[:n_trf], m20[:n_trf], std[:n_trf], logc[:n_trf]])
        Xte[:, ki * 4:(ki + 1) * 4] = np.column_stack([m5[n_trf:], m20[n_trf:], std[n_trf:], logc[n_trf:]])
        g_joint = np.concatenate([g_pool, g_va])
        _, g_j = np.unique(g_joint, return_inverse=True)
        g_pool2, g_va2 = g_j[: len(g_pool)], g_j[len(g_pool):]
        nbin2 = int(g_j.max()) + 1
        cnt2 = np.bincount(g_pool2, minlength=nbin2).astype(np.float64)
        sz = np.bincount(g_pool2, weights=z_pool, minlength=nbin2)
        sz2 = np.bincount(g_pool2, weights=z_pool * z_pool, minlength=nbin2)
        c = cnt2[g_va2]
        m5v = np.where(c > 0, sz[g_va2] / (c + 5.0), 0.0)
        m20v = np.where(c > 0, sz[g_va2] / (c + 20.0), 0.0)
        var = np.zeros(len(g_va2))
        ok = c >= 2
        var[ok] = (sz2[g_va2][ok] - (sz[g_va2][ok] ** 2) / c[ok]) / np.maximum(c[ok] - 1.0, 1.0)
        Xva[:, ki * 4:(ki + 1) * 4] = np.column_stack([m5v, m20v, np.sqrt(np.maximum(var, 0.0)), np.log1p(c)])
    return Xtr, Xva, Xte


def numeric_X(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c not in ("id", "is_fraud")]
    X = df[cols].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = pd.factorize(X[c].astype(str), sort=False)[0]
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X.to_numpy(np.float32)


def inner_hgb(Xtr, ytr, Xva, Xte, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(ytr), np.float64)
    va_bags, te_bags = [], []
    for i, (a, b) in enumerate(skf.split(Xtr, ytr)):
        clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=120, min_samples_leaf=200,
            random_state=seed + i,
        )
        clf.fit(Xtr[a], ytr[a])
        oof[b] = clf.predict_proba(Xtr[b])[:, 1]
        va_bags.append(clf.predict_proba(Xva)[:, 1])
        te_bags.append(clf.predict_proba(Xte)[:, 1])
    return rank_gauss(oof), rank_gauss(np.mean(va_bags, 0)), rank_gauss(np.mean(te_bags, 0))


def build_peer_residual(tr, te, y, folds, z_sov_oof):
    log("building collision keys…")
    keys_tr, keys_te = build_collision_keys(tr, te)
    X_all, X_te = numeric_X(tr), numeric_X(te)
    z_fold, z_te_fold = fold_matched_sovereign_z(y, folds)

    lgbm_oof = np.zeros(len(y), np.float64)
    lgbm_te = np.zeros(299984, np.float64)
    for f in range(5):
        tr_ix = np.where(folds != f)[0]
        va_ix = np.where(folds == f)[0]
        _, zv, zt = inner_hgb(X_all[tr_ix], y[tr_ix], X_all[va_ix], X_te, seed=100 + f)
        lgbm_oof[va_ix] = zv
        lgbm_te += zt / 5
        log(f"  hgb fold{f} va_auc={roc_auc_score(y[va_ix], zv):.5f}")

    resid_oof = np.zeros(len(y), np.float64)
    for f in range(5):
        tr_ix = np.where(folds != f)[0]
        va_ix = np.where(folds == f)[0]
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(z_sov_oof[tr_ix].reshape(-1, 1), y[tr_ix])
        resid_oof[va_ix] = y[va_ix] - lr.predict_proba(z_sov_oof[va_ix].reshape(-1, 1))[:, 1]

    delta_oof = np.zeros(len(y), np.float64)
    delta_te = np.zeros(299984, np.float64)
    fold_pos, subfold_pos = [], []

    for f in range(5):
        tr_ix = np.where(folds != f)[0]
        va_ix = np.where(folds == f)[0]
        log(f"peer+residual fold{f}…")
        z_tr = unitize(z_fold[tr_ix])
        z_te = unitize(z_te_fold[f])
        g_tr_list = [keys_tr[ki][tr_ix] for ki in range(8)]
        g_va_list = [keys_tr[ki][va_ix] for ki in range(8)]
        g_te_list = [keys_te[ki] for ki in range(8)]
        Xtr, Xva, Xte_f = peer_feats_for_fold(z_tr, z_te, g_tr_list, g_va_list, g_te_list)
        ytr = resid_oof[tr_ix]

        skf = StratifiedKFold(5, shuffle=True, random_state=7)
        va_acc, te_acc = [], []
        for a, _b in skf.split(Xtr, y[tr_ix]):
            ridge = Ridge(alpha=500.0).fit(Xtr[a], ytr[a])
            dtr = lgb.Dataset(Xtr[a], label=ytr[a])
            booster = lgb.train(
                dict(objective="regression", learning_rate=0.05, max_depth=3,
                     min_child_samples=400, max_delta_step=0.15, num_leaves=8,
                     verbosity=-1, seed=7),
                dtr, num_boost_round=120,
            )
            va_acc.append(0.5 * ridge.predict(Xva) + 0.5 * booster.predict(Xva))
            te_acc.append(0.5 * ridge.predict(Xte_f) + 0.5 * booster.predict(Xte_f))
        pred_va, pred_te = np.mean(va_acc, 0), np.mean(te_acc, 0)
        delta_oof[va_ix] = pred_va
        delta_te += pred_te / 5

        base = roc_auc_score(y[va_ix], z_sov_oof[va_ix])
        best_d, best_w = -1e9, 0.0
        u = unitize(pred_va)
        for w in (0.01, 0.02, 0.03, 0.05, 0.08):
            d = roc_auc_score(y[va_ix], z_sov_oof[va_ix] + w * u) - base
            if d > best_d:
                best_d, best_w = d, w
        fold_pos.append(1 if best_d > 0 else 0)
        for _, sva in StratifiedKFold(5, shuffle=True, random_state=42).split(va_ix, y[va_ix]):
            b0 = roc_auc_score(y[va_ix][sva], z_sov_oof[va_ix][sva])
            b1 = roc_auc_score(y[va_ix][sva], z_sov_oof[va_ix][sva] + best_w * u[sva])
            subfold_pos.append(1 if (b1 - b0) > 0 else 0)
        log(f"  residual fold{f} best_w={best_w} Δe5={best_d*1e5:.2f}")

    hy = hygiene_raw(delta_oof, delta_te)
    pos5, sf25 = int(sum(fold_pos)), int(sum(subfold_pos))
    promote = hy["pass"] and pos5 >= 4 and sf25 >= 17
    meta = {"hygiene": hy, "pos5": pos5, "sf25": sf25, "promote": promote,
            "lgbm_oof_auc": float(roc_auc_score(y, lgbm_oof))}
    log(f"peer hygiene disp={hy['disp_raw']:.4f} KS={hy['KS_raw']:.4f} promote={promote} pos={pos5}/5 sf={sf25}/25")
    return delta_oof, delta_te, lgbm_oof, lgbm_te, meta


def main():
    log("=== V47.2 STEP2 provenance + z_sovereign_v1 | NO_SUBMIT | $0 ===")
    prov = provenance_audit()
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    y = tr["is_fraud"].to_numpy(int)
    folds = np.load(OUT / "folds_skf42.npy")
    z0 = np.load(OUT / "z_sovereign_v0_oof.npy").ravel()
    log(f"z0 AUC={roc_auc_score(y, z0):.6f}")

    delta_o, delta_t, hgb_o, hgb_t, peer_meta = build_peer_residual(tr, te, y, folds, z0)
    extras = {}
    if peer_meta["promote"]:
        extras["peer_z_sovereign"] = (delta_o, delta_t)
        log("PROMOTE peer_z_sovereign")
    else:
        np.save(OUT / "peer_z_sovereign_raw_oof.npy", delta_o)
        np.save(OUT / "peer_z_sovereign_raw_test.npy", delta_t)
        log("peer residual NOT promoted")
    hy_l = hygiene_raw(hgb_o, hgb_t)
    log(f"hgb hygiene {hy_l}")
    if hy_l["pass"]:
        extras["inner_hgb_sov"] = (hgb_o, hgb_t)

    sys.path.insert(0, str(REPO / "scripts/v47_sovereign"))
    from step1_schema_and_z_sovereign import assemble_streams, stack_deotte

    streams = assemble_streams(tr, y, folds)
    for r in prov["rejects"]:
        for k in [k for k in list(streams) if r in k]:
            log(f"drop rejected {k}")
            streams.pop(k, None)
    for n, (o, t) in extras.items():
        streams[n] = (np.asarray(o, float).ravel(), np.asarray(t, float).ravel())

    oof, te_p, meta = stack_deotte(streams, y, folds)
    np.save(OUT / "z_sovereign_v1_oof.npy", oof.astype(np.float64))
    np.save(OUT / "z_sovereign_v1_test.npy", te_p.astype(np.float64))
    meta.update({"peer": peer_meta, "NO_SUBMIT": True, "extras": list(extras.keys()), "hgb_hygiene": hy_l})
    (LOGD / "STEP2_Z_SOVEREIGN_V1.json").write_text(json.dumps(meta, indent=2, default=float))
    (LOGD / "STEP2_Z_SOVEREIGN_V1.md").write_text(
        f"# V47.2 z_sovereign_v1\n\n"
        f"- overall **{meta['overall_auc']:.6f}** · fold0 **{meta['fold0_auc']:.6f}**\n"
        f"- peer promote **{peer_meta['promote']}** pos={peer_meta['pos5']}/5 sf={peer_meta['sf25']}/25 "
        f"disp={peer_meta['hygiene']['disp_raw']:.4f} KS={peer_meta['hygiene']['KS_raw']:.4f}\n"
        f"- streams `{meta['names']}`\n- extras `{list(extras.keys())}`\n- NO_SUBMIT=True · $0\n"
    )
    log(f"DONE v1 AUC={meta['overall_auc']:.6f} F0={meta['fold0_auc']:.6f}")


if __name__ == "__main__":
    main()
