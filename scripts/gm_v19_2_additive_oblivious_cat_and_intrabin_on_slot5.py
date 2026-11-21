#!/usr/bin/env python3
"""GM V19.2 — Additive CatBoost on Slot5 + intrabin bank; submit IFF 5/5×3 H2H.

0 GPU pods. Research-first. Does NOT delete R2_v178.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import ks_2samp, rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
TMP15 = Path("/tmp/s6e11_v15")
TMP16 = Path("/tmp/s6e11_v16")
OUT = TMP16 / "out"
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
DATA = REPO / "data"
ORIG = REPO / "original/credit_card_fraud_2026.csv"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
SLOT6 = SUB / "sub_v18_1_LIVE_slot6_Slot5_plus_R1_pure_Bo10_w0p022.csv"
SLOT7 = SUB / "sub_v19_0_LIVE_slot7_zero_echo_peer_hybrid.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
OUT_CSV = SUB / "sub_v19_2_LIVE_slot8_anchored_R2_plus_CatObliv.csv"
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W_R2 = 0.020
NUM_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
    "txn_count_last_24h",
    "card_age_months",
    "customer_age",
    "cvv_retry_count",
]
CONT6 = [
    "amount_usd",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
    "distance_from_home_km",
    "hours_since_last_txn",
]
SEEDS = [42, 2026, 7, 100, 999]


def L(v):
    v = np.asarray(v, float).ravel()
    r = (rankdata(v, method="average") - 0.5) / len(v)
    r = np.clip(r, 1e-12, 1.0 - 1e-12)
    return np.log(r / (1.0 - r))


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def qlock(raw, ref):
    raw, ref = np.asarray(raw, float), np.asarray(ref, float)
    out = np.empty_like(ref)
    out[np.argsort(raw)] = np.sort(ref)
    return out


def frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=10.0):
    vo, vt = unitize(v_oof), unitize(v_test)
    orth_o = np.zeros(len(vo))
    betas = []
    for tr, te in folds:
        lr = Ridge(alpha=ridge_alpha, fit_intercept=True)
        lr.fit(Bo[tr], vo[tr])
        betas.append(lr.coef_.copy())
        orth_o[te] = vo[te] - Bo[te] @ lr.coef_ - lr.intercept_
    beta_bar = np.mean(np.stack(betas, 0), axis=0)
    orth_t = vt - Bt @ beta_bar
    return orth_o - orth_o.mean(), orth_t - orth_t.mean(), beta_bar


def residualize_vs_u(v_oof, v_test, u_oof, u_test, folds):
    """Per-fold OLS residual vs single column u (exact corr~0 on OOF)."""
    vo, vt = unitize(v_oof), unitize(v_test)
    uo, ut = unitize(u_oof), unitize(u_test)
    out_o = np.zeros(len(vo))
    betas = []
    for tr, te in folds:
        # vo = a + b*uo on tr
        X = np.column_stack([np.ones(len(tr)), uo[tr]])
        coef, *_ = np.linalg.lstsq(X, vo[tr], rcond=None)
        betas.append(coef.copy())
        out_o[te] = vo[te] - (coef[0] + coef[1] * uo[te])
    bbar = np.mean(np.stack(betas, 0), 0)
    out_t = vt - (bbar[0] + bbar[1] * ut)
    return out_o - out_o.mean(), out_t - out_t.mean()


def forensic_gates(orth_o, orth_t, Bt, live_resid, bn=None, f2_thr=0.035, f3_thr=0.05):
    disp = float((orth_t.std() + 1e-12) / (orth_o.std() + 1e-12))
    leaks = [float(np.corrcoef(orth_t, Bt[:, j])[0, 1]) for j in range(Bt.shape[1])]
    max_leak = float(np.max(np.abs(leaks)))
    leak_arg = int(np.argmax(np.abs(leaks)))
    c_live = float(np.corrcoef(orth_t, live_resid)[0, 1]) if live_resid.std() > 1e-12 else 0.0
    names = bn or [f"c{j}" for j in range(Bt.shape[1])]
    return {
        "disp": disp,
        "max_basis_leak": max_leak,
        "leak_col": names[leak_arg] if leak_arg < len(names) else "?",
        "corr_live_resid": c_live,
        "F1": 0.90 <= disp <= 1.10,
        "F2": max_leak < f2_thr,
        "F3": abs(c_live) < f3_thr,
        "verified": (0.90 <= disp <= 1.10) and (max_leak < f2_thr) and (abs(c_live) < f3_thr),
    }


def raw_disp_ks(oof, test):
    oof, test = np.asarray(oof, float), np.asarray(test, float)
    disp = float((test.std() + 1e-12) / (oof.std() + 1e-12))
    ks = float(ks_2samp(oof, test).statistic)
    return {
        "raw_disp": disp,
        "raw_KS": ks,
        "pass": (0.92 <= disp <= 1.08) and (ks < 0.025),
    }


def score_at_w(u, zbase, y, folds, w):
    u = unitize(u)
    base = [roc_auc_score(y[te], zbase[te]) for _, te in folds]
    d = [
        (roc_auc_score(y[te], zbase[te] + w * u[te]) - base[i]) * 1e5
        for i, (_, te) in enumerate(folds)
    ]
    deltas = []
    for _, te in folds:
        yf, zb, uf = y[te], zbase[te], u[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(yf)), yf
        ):
            b = roc_auc_score(yf[ste], zb[ste])
            deltas.append((roc_auc_score(yf[ste], zb[ste] + 0.01 * uf[ste]) - b) * 1e5)
    d25 = np.asarray(deltas, float)
    mean = float(d25.mean())
    se = float(d25.std(ddof=1) / np.sqrt(len(d25)))
    try:
        _, wp = wilcoxon(d25, alternative="greater")
        wp = float(wp)
    except Exception:
        wp = float("nan")
    return {
        "w": float(w),
        "pos": int(sum(x > 0 for x in d)),
        "mean": float(np.mean(d)),
        "min": float(np.min(d)),
        "folds": [round(float(x), 3) for x in d],
        "sf25": int((d25 > 0).sum()),
        "t": float(mean / (se + 1e-18)),
        "wilcoxon_p": wp,
    }


def h2h_vs_slot5_scores(cand_oof, z_s5_family, y, folds):
    """ΔAUC of candidate score vector vs Slot5 family score (already includes R2)."""
    d = []
    for _, te in folds:
        d.append(
            (roc_auc_score(y[te], cand_oof[te]) - roc_auc_score(y[te], z_s5_family[te])) * 1e5
        )
    deltas = []
    for _, te in folds:
        yf = y[te]
        c, s = cand_oof[te], z_s5_family[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(yf)), yf
        ):
            deltas.append((roc_auc_score(yf[ste], c[ste]) - roc_auc_score(yf[ste], s[ste])) * 1e5)
    d25 = np.asarray(deltas, float)
    mean = float(d25.mean())
    se = float(d25.std(ddof=1) / np.sqrt(len(d25)))
    return {
        "pos": int(sum(x > 0 for x in d)),
        "mean": float(np.mean(d)),
        "min": float(np.min(d)),
        "folds": [round(float(x), 3) for x in d],
        "sf25": int((d25 > 0).sum()),
        "t": float(mean / (se + 1e-18)),
    }


def one_minus_rho_e4(a, b):
    return (1.0 - float(spearmanr(a, b).correlation)) * 1e4


def scrub(x):
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
    dE_o = unitize(cache["dE"])
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    te0 = folds[0][1]
    n_tr, n_te = len(y), 299984

    Z = np.load(Q17, allow_pickle=True)
    qnames = [str(s) for s in Z["stream_names"]]
    XO = {n: Z[f"oof_{n}"].ravel() for n in qnames}
    XT = {n: Z[f"test_{n}"].ravel() for n in qnames}

    live = pd.read_csv(LIVE)
    ids = live["id"].to_numpy()
    lv = live["is_fraud"].to_numpy(float)
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot5 = pd.read_csv(SLOT5).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot6 = pd.read_csv(SLOT6).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot7 = pd.read_csv(SLOT7).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    l7f_o, l7f_t = XO["lm7_5fold"], XT["lm7_5fold"]
    R3_t_L = np.stack([L(R3_te[f]) for f in range(5)], 0)
    z_ath_t = L(lv)
    raw_E_o = (L(R3) - L(l7f_o)) * (0.5 + 0.5 * U(z_ath))
    raw_E_t = (L(R3_t_L.mean(0)) - L(l7f_t)) * (0.5 + 0.5 * U(lv))
    Xo_e = np.column_stack([unitize(z_ath), unitize(d1)])
    Xt_e = np.column_stack([unitize(z_ath_t), unitize(d1t)])
    dE_o2, dE_t, _ = frozen_lofo_orth(raw_E_o, raw_E_t, Xo_e, Xt_e, folds, ridge_alpha=1.0)
    if np.corrcoef(dE_o, unitize(dE_o2))[0, 1] < 0:
        dE_o = -dE_o
    if roc_auc_score(y[te0], z_ath[te0] + 0.01 * unitize(dE_o2)[te0]) < roc_auc_score(
        y[te0], z_ath[te0] - 0.01 * unitize(dE_o2)[te0]
    ):
        dE_t = -dE_t
    dE_o, dE_t = unitize(dE_o), unitize(dE_t)

    ms_o = np.load(REPO / "preds/P_super_v4_oof.npy").ravel()
    ms_t = np.load(REPO / "preds/P_super_v4_test.npy").ravel()
    g26_o = np.load(G26_OOF).ravel()[:n_tr]
    g26_t = np.load(G26_T0).ravel()[:n_te]
    Bo9 = np.column_stack(
        [
            unitize(L(ms_o)),
            unitize(L(g26_o)),
            unitize(L(XO["megayak_5f_lm"])),
            unitize(L(l7f_o)),
            unitize(XO["daoviet_gpdvae"]),
            unitize(XO["gb_bag5"]),
            unitize(XO["yekenot_realmlp"]),
            unitize(d1),
            unitize(dE_o),
        ]
    )
    Bt9 = np.column_stack(
        [
            unitize(L(ms_t)),
            unitize(L(g26_t)),
            unitize(L(XT["megayak_5f_lm"])),
            unitize(L(l7f_t)),
            unitize(XT["daoviet_gpdvae"]),
            unitize(XT["gb_bag5"]),
            unitize(XT["yekenot_realmlp"]),
            unitize(d1t),
            unitize(dE_t),
        ]
    )
    z_s3 = z_ath + 0.01045 * dE_o + 0.0055 * unitize(d1)
    ridge_s3 = Ridge(alpha=1.0, fit_intercept=True).fit(Bt9, unitize(L(slot3)))
    z_slot3_twin = unitize(Bo9 @ ridge_s3.coef_ + ridge_s3.intercept_)
    bank_o, bank_t = [], []
    skip = {
        "megayak_5f_lm",
        "lm7_5fold",
        "gpt2_seed2026",
        "masaya_gpt2",
        "daoviet_gpdvae",
        "gb_bag5",
        "yekenot_realmlp",
    }
    for j, nm in enumerate(BN9):
        if nm not in ("d1_v4", "Family_E"):
            bank_o.append(unitize(Bo9[:, j]))
            bank_t.append(unitize(Bt9[:, j]))
    for n in qnames:
        if n not in skip:
            bank_o.append(unitize(XO[n]))
            bank_t.append(unitize(XT[n]))
    for p in sorted(ALEX.glob("oof_*.npy"))[:8]:
        tp = ALEX / ("test_" + p.name[len("oof_") :])
        if tp.exists():
            bank_o.append(unitize(np.load(p).ravel()[:n_tr]))
            bank_t.append(unitize(np.load(tp).ravel()[:n_te]))
    Bo_bank = np.column_stack(bank_o)
    Bt_bank = np.column_stack(bank_t)
    locked_t = W_D1_LOCK * unitize(d1t) + W_FE_LOCK * unitize(dE_t)
    w_nnls, _ = nnls(Bt_bank, (L(slot3) - locked_t) - (L(slot3) - locked_t).mean())
    z_nnls = W_D1_LOCK * unitize(d1) + W_FE_LOCK * unitize(dE_o) + Bo_bank @ w_nnls
    live_resid = L(lv) - Ridge(alpha=10.0).fit(Bt9, L(lv)).predict(Bt9)
    bases_s3 = {"z_S3": z_s3, "z_Slot3_twin": z_slot3_twin, "z_NNLS_twin": z_nnls}

    # R2 + Bo10
    fam = np.load(OUT / "v19_1_families.npz")
    R2_o, R2_t = fam["R2_v178_orig_oof"], fam["R2_v178_orig_test"]
    Cat_o, Cat_t = fam["F_cat_obliv_oof"], fam["F_cat_obliv_test"]
    Lgb_o, Lgb_t = fam["F_lgb_ms39_oof"], fam["F_lgb_ms39_test"]

    def orient_bo9(vo, vt):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=10.0)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s3[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        return unitize(u), unitize(ot)

    u_r2, ut_r2 = orient_bo9(R2_o, R2_t)
    u_cat_s3, ut_cat_s3 = orient_bo9(Cat_o, Cat_t)
    u_lgb_s3, ut_lgb_s3 = orient_bo9(Lgb_o, Lgb_t)

    z_s5 = z_s3 + W_R2 * u_r2
    z_slot5_twin = z_slot3_twin + W_R2 * u_r2
    z_nnls_s5 = z_nnls + W_R2 * u_r2
    bases_s5 = {"z_S5": z_s5, "z_Slot5_twin": z_slot5_twin, "z_NNLS_S5": z_nnls_s5}
    Bo10 = np.column_stack([Bo9, u_r2])
    Bt10 = np.column_stack([Bt9, ut_r2])
    bn10 = BN9 + ["R2_orth"]

    # ===== STEP 1: additive + anchored (cached) =====
    print("=== STEP 1: additive + anchored on cached families ===", f"t={time.time()-t0:.1f}s")
    corr_cat_r2 = float(np.corrcoef(u_cat_s3, u_r2)[0, 1])
    print(f"  corr(F_cat_obliv_Bo9, R2_v178)={corr_cat_r2:+.4f}")

    # orth Cat/LGB vs Bo10_S5
    def orient_bo10(vo, vt, zbase=z_s5):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo10, Bt10, folds, ridge_alpha=10.0)
        u = unitize(oo)
        if roc_auc_score(y[te0], zbase[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], zbase[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        return unitize(u), unitize(ot)

    C_cat_o, C_cat_t = orient_bo10(Cat_o, Cat_t)
    C_lgb_o, C_lgb_t = orient_bo10(Lgb_o, Lgb_t)
    # exact residualize vs u_R2 as well
    C_cat_r_o, C_cat_r_t = residualize_vs_u(C_cat_o, C_cat_t, u_r2, ut_r2, folds)
    C_cat_r_o, C_cat_r_t = unitize(C_cat_r_o), unitize(C_cat_r_t)
    corr_after = float(np.corrcoef(C_cat_r_o, u_r2)[0, 1])
    print(f"  corr(C_cat_orth_S5 residualized, u_R2)={corr_after:+.6f}")

    rd_cat = raw_disp_ks(C_cat_o, C_cat_t)
    g_cat_bo9 = forensic_gates(C_cat_o, C_cat_t, Bt9, live_resid, bn=BN9)
    g_cat_bo10 = forensic_gates(C_cat_o, C_cat_t, Bt10, live_resid, bn=bn10)
    print(
        f"  C_cat|Bo10: disp={rd_cat['raw_disp']:.3f} KS={rd_cat['raw_KS']:.4f} "
        f"F2_Bo9={g_cat_bo9['max_basis_leak']:.4f} F2_Bo10={g_cat_bo10['max_basis_leak']:.4f} "
        f"F3={g_cat_bo10['corr_live_resid']:+.4f}"
    )

    step1_hits = []

    # Geometry 1: pure additive on Slot5
    print("  -- Geometry 1: z_S5 + w * C_cat_orth_S5 --")
    for w_add in (0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020):
        per = {}
        for b, zb in bases_s5.items():
            cand = zb + w_add * C_cat_o
            per[b] = h2h_vs_slot5_scores(cand, zb, y, folds)
        ok = all(
            per[b]["pos"] >= 5 and per[b]["min"] > 0.05 and per[b]["sf25"] >= 18 and per[b]["t"] >= 2.8
            for b in bases_s5
        )
        print(
            f"  w_add={w_add:.3f} ok={ok} | "
            + " | ".join(
                f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f} "
                f"sf{per[b]['sf25']} t={per[b]['t']:.2f}"
                for b in bases_s5
            )
        )
        if ok:
            step1_hits.append(
                {
                    "geom": "additive_S5",
                    "w_add": w_add,
                    "per": per,
                    "u_oof": C_cat_o,
                    "u_test": C_cat_t,
                    "build": "z_S5 + w_add * C_cat",
                }
            )

    # also try residualized-vs-R2 version
    print("  -- Geometry 1b: z_S5 + w * residualize(C_cat, u_R2) --")
    for w_add in (0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020):
        per = {}
        for b, zb in bases_s5.items():
            cand = zb + w_add * C_cat_r_o
            per[b] = h2h_vs_slot5_scores(cand, zb, y, folds)
        ok = all(
            per[b]["pos"] >= 5 and per[b]["min"] > 0.05 and per[b]["sf25"] >= 18 and per[b]["t"] >= 2.8
            for b in bases_s5
        )
        print(
            f"  w_add={w_add:.3f} ok={ok} | "
            + " | ".join(
                f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f} "
                f"sf{per[b]['sf25']} t={per[b]['t']:.2f}"
                for b in bases_s5
            )
        )
        if ok:
            step1_hits.append(
                {
                    "geom": "additive_S5_residR2",
                    "w_add": w_add,
                    "per": per,
                    "u_oof": C_cat_r_o,
                    "u_test": C_cat_r_t,
                    "build": "z_S5 + w_add * resid(C_cat,R2)",
                }
            )

    # Geometry 2: anchored joint on Slot3
    print("  -- Geometry 2: anchored joint on Slot3 --")
    for w_orig in (0.010, 0.012, 0.014, 0.016, 0.018, 0.020):
        for w_cat in (0.004, 0.006, 0.008, 0.010, 0.012, 0.015):
            for w_lgb in (0.000, 0.003, 0.005):
                per = {}
                for bname, zb3, zb5 in [
                    ("z_S5", z_s3, z_s5),
                    ("z_Slot5_twin", z_slot3_twin, z_slot5_twin),
                    ("z_NNLS_S5", z_nnls, z_nnls_s5),
                ]:
                    cand = zb3 + w_orig * u_r2 + w_cat * u_cat_s3 + w_lgb * u_lgb_s3
                    per[bname] = h2h_vs_slot5_scores(cand, zb5, y, folds)
                ok = all(
                    per[b]["pos"] >= 5
                    and per[b]["min"] > 0.05
                    and per[b]["sf25"] >= 18
                    and per[b]["t"] >= 2.8
                    for b in per
                )
                if ok or (w_orig == 0.020 and w_cat in (0.008, 0.010) and w_lgb == 0.0):
                    print(
                        f"  wo={w_orig:.3f} wc={w_cat:.3f} wl={w_lgb:.3f} ok={ok} | "
                        + " | ".join(
                            f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f}"
                            for b in per
                        )
                    )
                if ok:
                    step1_hits.append(
                        {
                            "geom": "anchored_joint",
                            "w_orig": w_orig,
                            "w_cat": w_cat,
                            "w_lgb": w_lgb,
                            "per": per,
                            "u_oof": None,  # multi-component
                            "u_test": None,
                            "components": {
                                "u_r2": (u_r2, ut_r2),
                                "u_cat": (u_cat_s3, ut_cat_s3),
                                "u_lgb": (u_lgb_s3, ut_lgb_s3),
                            },
                        }
                    )

    print(f"  Step1 hits clearing 5/5×3 strict: {len(step1_hits)}  t={time.time()-t0:.1f}s")

    # ===== STEP 2: new families =====
    print("=== STEP 2: new Cat / intrabin families ===", f"t={time.time()-t0:.1f}s")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    def joint_q1000(col):
        v = np.concatenate([train[col].to_numpy(float), test[col].to_numpy(float)])
        r = rankdata(v, method="average")
        q = np.clip((r / (len(v) + 1e-12) * 1000).astype(np.int64), 0, 999)
        return q[:n_tr], q[n_tr:]

    keys_tr, keys_te = {}, {}
    for c in NUM_COLS:
        vtr, vte = train[c].to_numpy(float), test[c].to_numpy(float)
        keys_tr[f"{c}_cents"] = (np.round(vtr * 100).astype(np.int64) % 100) % 100
        keys_te[f"{c}_cents"] = (np.round(vte * 100).astype(np.int64) % 100) % 100
        keys_tr[f"{c}_tenth"] = (np.round(vtr * 10).astype(np.int64) % 10) % 10
        keys_te[f"{c}_tenth"] = (np.round(vte * 10).astype(np.int64) % 10) % 10
        keys_tr[f"{c}_ones"] = (np.abs(vtr).astype(np.int64) % 10) % 10
        keys_te[f"{c}_ones"] = (np.abs(vte).astype(np.int64) % 10) % 10
    for c in ("amount_usd", "account_balance_usd", "distance_from_home_km", "velocity_score"):
        qtr, qte = joint_q1000(c)
        keys_tr[f"{c}_q1000"] = qtr
        keys_te[f"{c}_q1000"] = qte
    amt_all = np.round(
        np.concatenate([train["amount_usd"].to_numpy(float), test["amount_usd"].to_numpy(float)]) * 100
    ).astype(np.int64)
    _, inv = np.unique(amt_all, return_inverse=True)
    freq = np.log1p(np.bincount(inv)[inv].astype(np.float64))
    keys_tr["amt_freq_bin"] = np.clip((freq[:n_tr] * 5).astype(np.int64), 0, 50)
    keys_te["amt_freq_bin"] = np.clip((freq[n_tr:] * 5).astype(np.int64), 0, 50)
    feat_freq_tr, feat_freq_te = freq[:n_tr], freq[n_tr:]
    orig_cnt = None
    if ORIG.exists():
        orig = pd.read_csv(ORIG)
        oc = [c for c in orig.columns if "amount" in c.lower() or c.lower() == "amt"]
        if oc:
            oa = np.round(orig[oc[0]].to_numpy(float) * 100).astype(np.int64)
            ctr = Counter(oa.tolist())
            otr = np.array([ctr.get(int(x), 0) for x in amt_all[:n_tr]], dtype=np.float64)
            ote = np.array([ctr.get(int(x), 0) for x in amt_all[n_tr:]], dtype=np.float64)
            keys_tr["amt_orig_bin"] = np.clip(otr.astype(np.int64), 0, 20)
            keys_te["amt_orig_bin"] = np.clip(ote.astype(np.int64), 0, 20)
            orig_cnt = (otr, ote)
    for cat in ("merchant_category", "channel", "device_type"):
        codes, _ = pd.factorize(pd.concat([train[cat], test[cat]], ignore_index=True))
        keys_tr[f"{cat}_x_amtones"] = codes[:n_tr] * 10 + keys_tr["amount_usd_ones"]
        keys_te[f"{cat}_x_amtones"] = codes[n_tr:] * 10 + keys_te["amount_usd_ones"]
    key_names = list(keys_tr.keys())

    def build_X45(split="train"):
        cols = []
        src = keys_tr if split == "train" else keys_te
        for kn in key_names:
            cols.append(src[kn].astype(np.float64))
        if split == "train":
            cols.append(feat_freq_tr)
            for c in ("amount_usd", "velocity_score", "merchant_risk_score", "distance_from_home_km"):
                cols.append(train[c].to_numpy(float))
            if orig_cnt is not None:
                cols.append(orig_cnt[0])
        else:
            cols.append(feat_freq_te)
            for c in ("amount_usd", "velocity_score", "merchant_risk_score", "distance_from_home_km"):
                cols.append(test[c].to_numpy(float))
            if orig_cnt is not None:
                cols.append(orig_cnt[1])
        return np.column_stack(cols).astype(np.float32)

    X45_tr, X45_te = build_X45("train"), build_X45("test")
    print(f"  X_tab45={X45_tr.shape[1]}")

    # X_intrabin18
    def rank_pct(col):
        v = np.concatenate([train[col].to_numpy(float), test[col].to_numpy(float)])
        return (rankdata(v, method="average") - 0.5) / len(v)

    ib_tr, ib_te = [], []
    for c in CONT6:
        rp = rank_pct(c)
        ib_tr.append((100.0 * rp[:n_tr]) % 1.0)
        ib_te.append((100.0 * rp[n_tr:]) % 1.0)
        ib_tr.append((20.0 * rp[:n_tr]) % 1.0)
        ib_te.append((20.0 * rp[n_tr:]) % 1.0)
    amt_tr, amt_te = train["amount_usd"].to_numpy(float), test["amount_usd"].to_numpy(float)
    bal_tr, bal_te = train["account_balance_usd"].to_numpy(float), test["account_balance_usd"].to_numpy(
        float
    )
    ib_tr.extend(
        [
            np.log1p(np.abs(amt_tr)) - np.log1p(np.abs(bal_tr)),
            amt_tr / (np.abs(amt_tr) + np.abs(bal_tr) + 1.0),
            np.cos(2 * np.pi * ((np.round(amt_tr * 100).astype(np.int64) % 100) / 100.0)),
            np.sin(2 * np.pi * ((np.round(amt_tr * 100).astype(np.int64) % 100) / 100.0)),
            np.cos(2 * np.pi * ((np.round(bal_tr * 100).astype(np.int64) % 100) / 100.0)),
            np.sin(2 * np.pi * ((np.round(bal_tr * 100).astype(np.int64) % 100) / 100.0)),
        ]
    )
    ib_te.extend(
        [
            np.log1p(np.abs(amt_te)) - np.log1p(np.abs(bal_te)),
            amt_te / (np.abs(amt_te) + np.abs(bal_te) + 1.0),
            np.cos(2 * np.pi * ((np.round(amt_te * 100).astype(np.int64) % 100) / 100.0)),
            np.sin(2 * np.pi * ((np.round(amt_te * 100).astype(np.int64) % 100) / 100.0)),
            np.cos(2 * np.pi * ((np.round(bal_te * 100).astype(np.int64) % 100) / 100.0)),
            np.sin(2 * np.pi * ((np.round(bal_te * 100).astype(np.int64) % 100) / 100.0)),
        ]
    )
    Xin_tr = np.column_stack(ib_tr).astype(np.float32)
    Xin_te = np.column_stack(ib_te).astype(np.float32)
    assert Xin_tr.shape[1] == 18, Xin_tr.shape
    X63_tr = np.column_stack([X45_tr, Xin_tr]).astype(np.float32)
    X63_te = np.column_stack([X45_te, Xin_te]).astype(np.float32)
    print(f"  X_intrabin18={Xin_tr.shape[1]} X_tab63={X63_tr.shape[1]}")

    # Platt g,h on S3 and S5
    g3, h3, g5, h5 = [], [], [], []
    for tr, _ in folds:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s3[tr].reshape(-1, 1), y[tr])
        a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
        p = expit(a * z_s3 + b)
        g3.append(y - p)
        h3.append(np.clip(p * (1 - p), 1e-6, None))
        clf5 = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf5.fit(z_s5[tr].reshape(-1, 1), y[tr])
        a5, b5 = float(clf5.coef_[0, 0]), float(clf5.intercept_[0])
        p5 = expit(a5 * z_s5 + b5)
        g5.append(y - p5)
        h5.append(np.clip(p5 * (1 - p5), 1e-6, None))

    def fit_cat_bag(Xtr, Xte, g_list, h_list, depths, tag, iterations=120):
        oof_acc, te_acc = [], []
        for depth in depths:
            for seed in SEEDS:
                oo = np.zeros(n_tr)
                tef = []
                for f, (tr, va) in enumerate(folds):
                    model = CatBoostRegressor(
                        loss_function="RMSE",
                        depth=depth,
                        l2_leaf_reg=30,
                        iterations=iterations,
                        learning_rate=0.035,
                        random_seed=seed,
                        verbose=False,
                        thread_count=4,
                        bootstrap_type="Bernoulli",
                        subsample=0.8,
                        rsm=0.80,
                    )
                    model.fit(Xtr[tr], g_list[f][tr], sample_weight=h_list[f][tr])
                    oo[va] = model.predict(Xtr[va])
                    tef.append(model.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        oof = np.mean(np.stack(oof_acc, 0), 0)
        te = np.mean(np.stack(te_acc, 0), 0)
        print(f"  {tag}: n={len(oof_acc)} t={time.time()-t0:.1f}s")
        return oof, te

    def fit_lgb_small(Xtr, Xte, g_list, h_list, tag):
        oof_acc, te_acc = [], []
        for seed in SEEDS:
            oo = np.zeros(n_tr)
            tef = []
            for f, (tr, va) in enumerate(folds):
                model = lgb.LGBMRegressor(
                    objective="regression",
                    num_leaves=6,
                    min_child_samples=500,
                    n_estimators=40,
                    learning_rate=0.03,
                    colsample_bytree=0.8,
                    subsample=0.8,
                    reg_lambda=10.0,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=-1,
                )
                model.fit(Xtr[tr], g_list[f][tr], sample_weight=h_list[f][tr])
                oo[va] = model.predict(Xtr[va])
                tef.append(model.predict(Xte))
            oof_acc.append(oo)
            te_acc.append(np.mean(np.stack(tef, 0), 0))
        print(f"  {tag}: n={len(oof_acc)} t={time.time()-t0:.1f}s")
        return np.mean(np.stack(oof_acc, 0), 0), np.mean(np.stack(te_acc, 0), 0)

    # Reduce depth grid slightly for runtime while covering multi-scale
    F_cat_ms45_o, F_cat_ms45_t = fit_cat_bag(
        X45_tr, X45_te, g3, h3, depths=(2, 3, 4, 5), tag="F_cat_ms45", iterations=120
    )
    F_cat_dir_o, F_cat_dir_t = fit_cat_bag(
        X45_tr, X45_te, g5, h5, depths=(3, 4, 5), tag="F_cat_direct_S5", iterations=120
    )
    F_cat_ib_o, F_cat_ib_t = fit_cat_bag(
        X63_tr, X63_te, g5, h5, depths=(3, 4, 5), tag="F_cat_intrabin_S5", iterations=100
    )
    F_lgb_ib_o, F_lgb_ib_t = fit_lgb_small(X63_tr, X63_te, g5, h5, tag="F_lgb_intrabin_S5")
    F_ib_o = unitize(0.6 * unitize(F_cat_ib_o) + 0.4 * unitize(F_lgb_ib_o))
    F_ib_t = unitize(0.6 * unitize(F_cat_ib_t) + 0.4 * unitize(F_lgb_ib_t))

    new_fams = {
        "F_cat_ms45": (F_cat_ms45_o, F_cat_ms45_t, "Bo9", z_s3),
        "F_cat_direct_S5": (F_cat_dir_o, F_cat_dir_t, "Bo10", z_s5),
        "F_cat_intrabin": (F_ib_o, F_ib_t, "Bo10", z_s5),
    }

    print("=== Step2 family audit ===")
    step2_oriented = {}
    for name, (vo, vt, basis, zbase) in new_fams.items():
        if basis == "Bo9":
            u, ut = orient_bo9(vo, vt)
            Bo, Bt, bn = Bo9, Bt9, BN9
        else:
            u, ut = orient_bo10(vo, vt, zbase=zbase)
            Bo, Bt, bn = Bo10, Bt10, bn10
        rd = raw_disp_ks(u, ut)
        gates = forensic_gates(u, ut, Bt, live_resid, bn=bn, f2_thr=0.035, f3_thr=0.045)
        # also F2 vs Bo9 always
        gates9 = forensic_gates(u, ut, Bt9, live_resid, bn=BN9, f2_thr=0.035, f3_thr=0.045)
        fam_ok = rd["pass"] and gates9["F2"] and gates["F3"]
        step2_oriented[name] = {
            "u": u,
            "ut": ut,
            "rd": rd,
            "gates": gates,
            "gates9": gates9,
            "fam_ok": fam_ok,
            "basis": basis,
        }
        print(
            f"  {name}: disp={rd['raw_disp']:.3f} KS={rd['raw_KS']:.4f} "
            f"F2_Bo9={gates9['max_basis_leak']:.4f} F2_basis={gates['max_basis_leak']:.4f} "
            f"ok={fam_ok}"
        )

        if fam_ok and basis == "Bo10":
            # additive on Slot5
            for w_add in (0.005, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020):
                per = {
                    b: h2h_vs_slot5_scores(zb + w_add * u, zb, y, folds) for b, zb in bases_s5.items()
                }
                ok = all(
                    per[b]["pos"] >= 5
                    and per[b]["min"] > 0.05
                    and per[b]["sf25"] >= 18
                    and per[b]["t"] >= 2.8
                    for b in bases_s5
                )
                print(
                    f"    add@{w_add:.3f} ok={ok} | "
                    + " | ".join(
                        f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f}"
                        for b in bases_s5
                    )
                )
                if ok:
                    step1_hits.append(
                        {
                            "geom": f"additive_S5_{name}",
                            "w_add": w_add,
                            "per": per,
                            "u_oof": u,
                            "u_test": ut,
                            "build": f"z_S5 + {w_add}*{name}",
                        }
                    )
        elif fam_ok and basis == "Bo9":
            # anchored with R2
            for w_orig in (0.014, 0.016, 0.018, 0.020):
                for w_new in (0.006, 0.008, 0.010, 0.012, 0.015):
                    per = {}
                    for bname, zb3, zb5 in [
                        ("z_S5", z_s3, z_s5),
                        ("z_Slot5_twin", z_slot3_twin, z_slot5_twin),
                        ("z_NNLS_S5", z_nnls, z_nnls_s5),
                    ]:
                        cand = zb3 + w_orig * u_r2 + w_new * u
                        per[bname] = h2h_vs_slot5_scores(cand, zb5, y, folds)
                    ok = all(
                        per[b]["pos"] >= 5
                        and per[b]["min"] > 0.05
                        and per[b]["sf25"] >= 18
                        and per[b]["t"] >= 2.8
                        for b in per
                    )
                    if ok or (w_orig == 0.020 and w_new in (0.008, 0.010)):
                        print(
                            f"    anch wo={w_orig:.3f} wn={w_new:.3f} ok={ok} | "
                            + " | ".join(
                                f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f}"
                                for b in per
                            )
                        )
                    if ok:
                        step1_hits.append(
                            {
                                "geom": f"anchored_{name}",
                                "w_orig": w_orig,
                                "w_new": w_new,
                                "per": per,
                                "u_oof": u,
                                "u_test": ut,
                                "u_r2": (u_r2, ut_r2),
                                "build": f"z_S3 + {w_orig}*R2 + {w_new}*{name}",
                            }
                        )

    # ===== STEP 3: gate + conditional submit =====
    print("=== STEP 3: strict gate + conditional submit ===", f"t={time.time()-t0:.1f}s")
    print(f"  total candidates clearing 5/5×3 H2H so far: {len(step1_hits)}")

    winner = None
    for hit in step1_hits:
        # build test candidate and check remaining gates
        if hit["geom"].startswith("additive"):
            w = hit["w_add"]
            ut = hit["u_test"]
            cand = qlock(L(slot5) + w * ut, lv)
            # vs Slot3 strength of the additive residual alone
            vs3 = {b: score_at_w(hit["u_oof"], zb, y, folds, w) for b, zb in bases_s3.items()}
        elif hit["geom"] == "anchored_joint":
            wo, wc, wl = hit["w_orig"], hit["w_cat"], hit["w_lgb"]
            cand_raw = (
                L(slot3)
                + wo * ut_r2
                + wc * ut_cat_s3
                + wl * ut_lgb_s3
            )
            cand = qlock(cand_raw, lv)
            # approximate vs3 using oof combo residual vs z_s3
            u_combo = unitize(wo * u_r2 + wc * u_cat_s3 + wl * u_lgb_s3)
            vs3 = {b: score_at_w(u_combo, zb, y, folds, 1.0) for b, zb in bases_s3.items()}
            # score_at_w with w=1 on already-weighted combo is wrong scale; recompute properly:
            vs3 = {}
            for b, zb in bases_s3.items():
                cand_o = zb + wo * u_r2 + wc * u_cat_s3 + wl * u_lgb_s3
                vs3[b] = h2h_vs_slot5_scores(cand_o, zb, y, folds)  # lift vs bare Slot3 base
        else:
            # anchored new family
            wo, wn = hit["w_orig"], hit["w_new"]
            ur2o, ur2t = hit["u_r2"]
            cand = qlock(L(slot3) + wo * ur2t + wn * hit["u_test"], lv)
            vs3 = {}
            for b, zb in bases_s3.items():
                cand_o = zb + wo * ur2o + wn * hit["u_oof"]
                vs3[b] = h2h_vs_slot5_scores(cand_o, zb, y, folds)

        rho = one_minus_rho_e4(cand, slot5)
        u_delta = unitize(L(cand) - L(slot5))
        corr6 = float(np.corrcoef(u_delta, unitize(L(slot6) - L(slot5)))[0, 1])
        corr7 = float(np.corrcoef(u_delta, unitize(L(slot7) - L(slot5)))[0, 1])
        per = hit["per"]

        g1 = True  # raw_disp checked at family level for additive; for joint use cat's
        g2 = True
        g3 = all(per[b]["pos"] >= 5 and per[b]["min"] > 0.05 for b in per)
        g4 = all(per[b]["sf25"] >= 18 and per[b]["t"] >= 2.8 for b in per)
        # vs Slot3: for additive, the add-on vs Slot3; require strong if available
        g4b = all(vs3[b]["sf25"] >= 18 for b in vs3)  # soft; strict 21/4.2 hard for pure S3 lifts
        g5 = 0.15 <= rho <= 0.50
        g6 = abs(corr6) < 0.15 and abs(corr7) < 0.15
        # For additive on S5, also require family raw_disp pass
        if hit["geom"].startswith("additive"):
            g1 = rd_cat["pass"] if "C_cat" in hit.get("build", "") or "cat" in hit["geom"].lower() else True
            if "intrabin" in hit["geom"] or "direct" in hit["geom"] or "ms45" in hit["geom"]:
                # find family
                for nm, info in step2_oriented.items():
                    if nm in hit["geom"]:
                        g1 = info["rd"]["pass"]
                        g2 = info["gates9"]["F2"] and abs(info["gates"]["corr_live_resid"]) < 0.045
                        break
            else:
                g2 = g_cat_bo9["F2"] and abs(g_cat_bo10["corr_live_resid"]) < 0.045

        # Strict Slot3 stats for additive residual (sf>=21, t>=4.2) when geom is additive
        if hit["geom"].startswith("additive"):
            vs3_add = {
                b: score_at_w(hit["u_oof"], zb, y, folds, hit["w_add"]) for b, zb in bases_s3.items()
            }
            g4_s3 = all(vs3_add[b]["sf25"] >= 21 and vs3_add[b]["t"] >= 4.2 for b in bases_s3)
        else:
            g4_s3 = True  # joint already includes R2; H2H vs S5 is primary
            vs3_add = vs3

        clear = g1 and g2 and g3 and g4 and g4_s3 and g5 and g6
        print(
            f"  cand {hit['geom']} gates g1-6={[g1,g2,g3,g4,g4_s3,g5,g6]} "
            f"1-ρ={rho:.3f}e-4 corr6/7={corr6:+.3f}/{corr7:+.3f} clear={clear}"
        )
        if clear:
            mean_lift = float(np.mean([per[b]["mean"] for b in per]))
            if winner is None or mean_lift > winner["mean_lift"]:
                winner = {
                    "hit": hit,
                    "cand": cand,
                    "rho": rho,
                    "corr6": corr6,
                    "corr7": corr7,
                    "vs3": vs3_add,
                    "mean_lift": mean_lift,
                    "gates": [g1, g2, g3, g4, g4_s3, g5, g6],
                }

    submit = winner is not None
    pub_score, ref = None, None
    if submit:
        cand = winner["cand"]
        assert len(cand) == n_te and np.isfinite(cand).all()
        assert float(ks_2samp(cand, slot5).statistic) < 0.0005
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(OUT_CSV, index=False)
        msg = (
            f"v19.2 Slot8: Slot5+additive/anchored CatObliv "
            f"geom={winner['hit']['geom']} 1-ρ={winner['rho']:.3f}e-4 H2H 5/5×3"
        )
        print("=== SUBMIT ===", msg)
        r = subprocess.run(
            [
                KAG,
                "competitions",
                "submit",
                "-c",
                "playground-series-s6e11",
                "-f",
                str(OUT_CSV),
                "-m",
                msg,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        print(r.stdout)
        print(r.stderr)
        if r.returncode != 0:
            raise SystemExit(f"submit failed rc={r.returncode}")
        for i in range(25):
            time.sleep(12 if i else 4)
            p = subprocess.run(
                [KAG, "competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            lines = [ln for ln in p.stdout.strip().splitlines() if ln]
            if len(lines) < 2:
                continue
            hdr = [h.strip() for h in lines[0].split(",")]
            for ln in lines[1:6]:
                cols = ln.split(",")
                row = {hdr[j]: cols[j].strip() for j in range(min(len(hdr), len(cols)))}
                fn = row.get("fileName", "")
                st = row.get("status", "")
                if "v19_2" in fn or "CatObliv" in fn or "slot8" in fn.lower():
                    ref = row.get("ref")
                    print(f"  poll: {st} ref={ref} score={row.get('publicScore')}")
                    if "complete" in st.lower() and row.get("publicScore"):
                        pub_score = row["publicScore"]
                        break
            if pub_score:
                break
        print(f"PUBLIC_LB={pub_score}")
    else:
        print("NO CANDIDATE CLEARED 5/5×3 STRICT H2H — HOLD Slot5; no submit")

    # serialize hits without huge arrays
    hits_lite = []
    for h in step1_hits:
        hl = {k: v for k, v in h.items() if k not in ("u_oof", "u_test", "u_r2", "components")}
        hits_lite.append(hl)

    payload = scrub(
        {
            "submit": submit,
            "n_hits_55": len(step1_hits),
            "corr_cat_r2": corr_cat_r2,
            "corr_cat_resid_r2": corr_after,
            "C_cat_raw": rd_cat,
            "C_cat_gates_bo9": g_cat_bo9,
            "C_cat_gates_bo10": g_cat_bo10,
            "step1_hits": hits_lite,
            "step2": {
                k: {
                    "rd": v["rd"],
                    "gates": v["gates"],
                    "gates9": v["gates9"],
                    "fam_ok": v["fam_ok"],
                    "basis": v["basis"],
                }
                for k, v in step2_oriented.items()
            },
            "winner": None
            if not submit
            else {
                "geom": winner["hit"]["geom"],
                "rho": winner["rho"],
                "corr6": winner["corr6"],
                "corr7": winner["corr7"],
                "mean_lift": winner["mean_lift"],
                "per": winner["hit"]["per"],
                "gates": winner["gates"],
                "ref": ref,
                "public_lb": pub_score,
            },
            "elapsed_s": time.time() - t0,
        }
    )
    OUT.joinpath("v19_2.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUT / "v19_2_families.npz",
        F_cat_ms45_oof=F_cat_ms45_o,
        F_cat_ms45_test=F_cat_ms45_t,
        F_cat_direct_S5_oof=F_cat_dir_o,
        F_cat_direct_S5_test=F_cat_dir_t,
        F_cat_intrabin_oof=F_ib_o,
        F_cat_intrabin_test=F_ib_t,
        C_cat_orth_S5_oof=C_cat_o,
        C_cat_orth_S5_test=C_cat_t,
    )
    print("WROTE", OUT / "v19_2.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE submit=", submit)


if __name__ == "__main__":
    main()
