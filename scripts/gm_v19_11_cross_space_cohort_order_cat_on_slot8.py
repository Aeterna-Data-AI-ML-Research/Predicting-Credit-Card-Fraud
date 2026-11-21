#!/usr/bin/env python3
"""GM V19.11 — Cross-space (cohort × order × tab) Cat/ExtraTrees on Slot8 residual.

Non-peer X_cross88; pre-train fail-ray depoison; orth vs [Bo12, M5, R1].
Submit Slot9 IFF all gates clear vs ATH 0.95395. 0 GPU pods.
"""
from __future__ import annotations

import json
import subprocess
import time
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parents[1]
TMP15, TMP16 = Path("/tmp/s6e11_v15"), Path("/tmp/s6e11_v16")
OUT, SUB, DATA, LOG = TMP16 / "out", REPO / "submissions", REPO / "data", REPO / "logs/overnight"
PRED = REPO / "preds"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
ORIG = REPO / "data/creditcard.csv"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
SLOT6 = SUB / "sub_v18_1_LIVE_slot6_Slot5_plus_R1_pure_Bo10_w0p022.csv"
SLOT7 = SUB / "sub_v19_0_LIVE_slot7_zero_echo_peer_hybrid.csv"
SLOT8 = SUB / "sub_v19_4_LIVE_slot8_A_cat_wo0.016_wc0.012_wu0.012.csv"
G26_OOF, G26_T0 = TMP16 / "g26/gpt2_s2026_oof_5fold.npy", TMP16 / "g26/gpt2_s2026_test_f0.npy"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
NUM_COLS = [
    "amount_usd",
    "account_balance_usd",
    "distance_from_home_km",
    "velocity_score",
    "merchant_risk_score",
]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W8_O, W8_C, W8_U = 0.016, 0.012, 0.012
SEEDS = (11, 21, 31, 41)
MIN_COHORT = 2000


def L(v):
    v = np.asarray(v, float).ravel()
    r = np.clip((rankdata(v, method="average") - 0.5) / len(v), 1e-12, 1 - 1e-12)
    return np.log(r / (1 - r))


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def qlock(raw, ref):
    """Same-length: replace raw order with sorted(ref) (submission PCHIP)."""
    raw, ref = np.asarray(raw, float), np.asarray(ref, float)
    out = np.empty_like(ref)
    out[np.argsort(raw)] = np.sort(ref)
    return out


def qmap_to_ref(raw, ref):
    """Different-length quantile map: give raw the marginal of ref."""
    raw = np.asarray(raw, float).ravel()
    ref = np.asarray(ref, float).ravel()
    r = (rankdata(raw, method="average") - 0.5) / len(raw)
    xs = (np.arange(len(ref), dtype=np.float64) + 0.5) / len(ref)
    return np.interp(r, xs, np.sort(ref))


def L_fold(raw, fold_id):
    out = np.zeros_like(raw, dtype=np.float64)
    for f in range(5):
        m = fold_id == f
        out[m] = L(raw[m])
    return out


def col_z(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-12)


def gini10(Z):
    K = Z.shape[-1]
    s = np.zeros(Z.shape[:-1], dtype=np.float64)
    for i, j in combinations(range(K), 2):
        s += np.abs(Z[..., i] - Z[..., j])
    return s / (K * (K - 1) / 2.0)


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


def h2h(cand, base, y, folds):
    d, deltas = [], []
    for _, te in folds:
        d.append((roc_auc_score(y[te], cand[te]) - roc_auc_score(y[te], base[te])) * 1e5)
        yf, c, b = y[te], cand[te], base[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(yf)), yf
        ):
            deltas.append((roc_auc_score(yf[ste], c[ste]) - roc_auc_score(yf[ste], b[ste])) * 1e5)
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


def one_m_rho_e4(a, b):
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


def codes_from_str(arr):
    _, inv = np.unique(np.asarray(arr, dtype=object), return_inverse=True)
    return inv.astype(np.int64)


def within_group_rank(vals, groups):
    vals = np.asarray(vals, float)
    groups = np.asarray(groups, np.int64)
    out = np.zeros(len(vals), dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2:
            out[m] = 0.5
        else:
            out[m] = U(vals[m])
    return out


def within_group_z(vals, groups, min_n=MIN_COHORT):
    vals = np.asarray(vals, float)
    groups = np.asarray(groups, np.int64)
    out = np.zeros(len(vals), dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        n = int(m.sum())
        if n < min_n:
            out[m] = 0.0
            continue
        v = vals[m]
        out[m] = (v - v.mean()) / (v.std() + 1e-12)
    return out


def within_group_log_ratio_median(vals, groups, min_n=MIN_COHORT):
    vals = np.asarray(vals, float)
    groups = np.asarray(groups, np.int64)
    out = np.zeros(len(vals), dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        if int(m.sum()) < min_n:
            out[m] = 0.0
            continue
        med = np.median(vals[m])
        out[m] = np.log1p(np.abs(vals[m])) - np.log1p(np.abs(med) + 1e-12)
    return out


def leave_self_peer_mean(z_all, codes, m0=15.0):
    codes = np.asarray(codes, np.int64)
    z_all = np.asarray(z_all, float)
    nbin = int(codes.max()) + 1
    s = np.bincount(codes, weights=z_all, minlength=nbin)
    c = np.bincount(codes, minlength=nbin).astype(np.float64)
    peer = (s[codes] - z_all) / np.maximum(c[codes] - 1.0, m0)
    peer[c[codes] < m0] = 0.0
    return peer, c[codes]


def project_cols_out(X, poison):
    """OLS-project each column of X orthogonal to poison columns (n, k)."""
    X = np.asarray(X, float)
    P = np.column_stack([unitize(p) for p in poison])
    # ridge solve for coeffs
    XtX = P.T @ P + 1e-3 * np.eye(P.shape[1])
    XtY = P.T @ X
    B = np.linalg.solve(XtX, XtY)
    return X - P @ B


def forensic(oo, ot, Bt):
    disp = float(np.std(ot) / (np.std(oo) + 1e-12))
    ut = unitize(ot)
    leaks = [abs(float(np.corrcoef(ut, Bt[:, j])[0, 1])) for j in range(Bt.shape[1])]
    return {
        "raw_disp": disp,
        "raw_KS": float(ks_2samp(unitize(oo), ut).statistic),
        "F2": float(max(leaks)),
        "F1": 0.92 <= disp <= 1.08,
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    # ---------- Slot8 reconstruct ----------
    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
    dE_o = unitize(cache["dE"])
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    te0 = folds[0][1]
    n_tr, n_te = len(y), 299984
    n_all = n_tr + n_te

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
    slot8 = pd.read_csv(SLOT8).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    bad6 = unitize(L(slot6) - L(slot5))
    bad7 = unitize(L(slot7) - L(slot5))

    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    l7f_o, l7f_t = XO["lm7_5fold"], XT["lm7_5fold"]
    R3_t_L = np.stack([L(R3_te[f]) for f in range(5)], 0)
    raw_E_o = (L(R3) - L(l7f_o)) * (0.5 + 0.5 * U(z_ath))
    raw_E_t = (L(R3_t_L.mean(0)) - L(l7f_t)) * (0.5 + 0.5 * U(lv))
    Xo_e = np.column_stack([unitize(z_ath), unitize(d1)])
    Xt_e = np.column_stack([unitize(L(lv)), unitize(d1t)])
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
    z_twin = unitize(Bo9 @ ridge_s3.coef_ + ridge_s3.intercept_)
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

    fam1 = np.load(OUT / "v19_1_families.npz")
    peers = np.load(OUT / "v18_9_peers.npz")
    r178 = np.load(OUT / "v17_8_residuals.npz")

    def orient_bo9(vo, vt):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=10.0)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s3[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        return unitize(u), unitize(ot)

    u_r2, ut_r2 = orient_bo9(fam1["R2_v178_orig_oof"], fam1["R2_v178_orig_test"])
    u_cat, ut_cat = orient_bo9(fam1["F_cat_obliv_oof"], fam1["F_cat_obliv_test"])

    raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
    llr_o = raw_o[:, :, 1] - raw_o[:, :, 0]
    llr_t = raw_t[:, :, :, 1] - raw_t[:, :, :, 0]
    Z_o = col_z(llr_o)
    Z_tf = [col_z(llr_t[f]) for f in range(5)]
    # Exact V19.4 Slot8 U5: L(gini) per test fold, then mean (NOT mean-then-L)
    U5_o = L_fold(gini10(Z_o), fold_id)
    U5_t = np.mean(np.stack([L(gini10(Z_tf[f])) for f in range(5)], 0), 0)
    Bo10 = np.column_stack([Bo9, u_r2])
    Bt10 = np.column_stack([Bt9, ut_r2])
    oo5, ot5, _ = frozen_lofo_orth(U5_o, U5_t, Bo10, Bt10, folds, ridge_alpha=1.0)
    u_u5 = unitize(oo5)
    z_s5_orient = z_s3 + 0.020 * u_r2  # V19.4 orients U5 vs Slot5 base
    if roc_auc_score(y[te0], z_s5_orient[te0] + 0.01 * u_u5[te0]) < roc_auc_score(
        y[te0], z_s5_orient[te0] - 0.01 * u_u5[te0]
    ):
        u_u5, ot5 = -u_u5, -ot5
    ut_u5 = unitize(ot5)

    z_s8 = z_s3 + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_t8 = z_twin + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_n8 = z_nnls + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    bases = {"z_S8": z_s8, "twin": z_t8, "nnls": z_n8}

    rebuild = qlock(L(slot3) + W8_O * ut_r2 + W8_C * ut_cat + W8_U * ut_u5, lv)
    rho_rebuild = one_m_rho_e4(rebuild, slot8)
    print(f"=== V19.11 cross-space === Slot8 rebuild 1-ρ={rho_rebuild:.6e}e-4 t={time.time()-t0:.1f}s")
    assert rho_rebuild < 1e-5, rho_rebuild

    # M5 / R1 for orth basis
    u_m5, ut_m5 = orient_bo9(peers["M5_oof"], peers["M5_test"])
    u_r1, ut_r1 = orient_bo9(r178["R1_oof"], r178["R1_test"])
    Bo12 = np.column_stack([Bo9, u_r2, u_cat, u_u5, u_m5, u_r1])
    Bt12 = np.column_stack([Bt9, ut_r2, ut_cat, ut_u5, ut_m5, ut_r1])

    # ---------- Build X blocks ----------
    print("=== Step1 build X_cross88 ===", f"t={time.time()-t0:.1f}s")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    assert len(train) == n_tr and len(test) == n_te

    amt = np.concatenate([train["amount_usd"].to_numpy(float), test["amount_usd"].to_numpy(float)])
    bal = np.concatenate(
        [train["account_balance_usd"].to_numpy(float), test["account_balance_usd"].to_numpy(float)]
    )
    vel = np.concatenate(
        [train["velocity_score"].to_numpy(float), test["velocity_score"].to_numpy(float)]
    )
    mr = np.concatenate(
        [
            train["merchant_risk_score"].to_numpy(float),
            test["merchant_risk_score"].to_numpy(float),
        ]
    )
    dist = np.concatenate(
        [
            train["distance_from_home_km"].to_numpy(float),
            test["distance_from_home_km"].to_numpy(float),
        ]
    )
    hrs = np.concatenate(
        [
            train["hours_since_last_txn"].to_numpy(float),
            test["hours_since_last_txn"].to_numpy(float),
        ]
    )
    mer = np.concatenate(
        [
            train["merchant_category"].astype(str).to_numpy(),
            test["merchant_category"].astype(str).to_numpy(),
        ]
    )
    ch = np.concatenate(
        [train["channel"].astype(str).to_numpy(), test["channel"].astype(str).to_numpy()]
    )
    dev = np.concatenate(
        [train["device_type"].astype(str).to_numpy(), test["device_type"].astype(str).to_numpy()]
    )
    foreign = np.concatenate(
        [
            train["is_foreign_transaction"].astype(str).to_numpy(),
            test["is_foreign_transaction"].astype(str).to_numpy(),
        ]
    )
    auth = np.concatenate(
        [train["auth_method"].astype(str).to_numpy(), test["auth_method"].astype(str).to_numpy()]
    )
    card = np.concatenate(
        [train["card_type"].astype(str).to_numpy(), test["card_type"].astype(str).to_numpy()]
    )
    cvv = np.concatenate(
        [train["cvv_retry_count"].to_numpy(float), test["cvv_retry_count"].to_numpy(float)]
    )
    # has_cvv proxy + is_online-ish channel
    has_cvv = (cvv > 0).astype(str)
    is_online = np.isin(ch, ["Online", "In-App", "Mobile"]).astype(str)

    g_cat = codes_from_str(mer)
    g_chdev = codes_from_str((pd.Series(ch) + "|" + pd.Series(dev)).to_numpy())
    g_forauth = codes_from_str((pd.Series(foreign) + "|" + pd.Series(auth)).to_numpy())
    g_card = codes_from_str(card)
    g_cvvon = codes_from_str((pd.Series(has_cvv) + "|" + pd.Series(is_online)).to_numpy())

    cohort_cols = []
    cohort_names = []
    for cname, vals in [
        ("amt", amt),
        ("bal", bal),
        ("vel", vel),
        ("mrisk", mr),
        ("dist", dist),
        ("hrs", hrs),
    ]:
        for gname, g in [
            ("cat", g_cat),
            ("chdev", g_chdev),
            ("forauth", g_forauth),
            ("card", g_card),
            ("cvvon", g_cvvon),
        ]:
            # only keep if enough large cohorts overall
            cohort_cols.append(within_group_z(vals, g))
            cohort_names.append(f"z_{cname}_{gname}")
            cohort_cols.append(within_group_rank(vals, g))
            cohort_names.append(f"rk_{cname}_{gname}")
    # trim to ~32 by picking strongest support groups: cat, chdev, forauth for z+rk of 4 nums + 2 extras
    # rebuild more compact (~32)
    cohort_cols, cohort_names = [], []
    for cname, vals in [("amt", amt), ("bal", bal), ("vel", vel), ("mrisk", mr)]:
        for gname, g in [("cat", g_cat), ("chdev", g_chdev), ("forauth", g_forauth)]:
            cohort_cols.append(within_group_z(vals, g))
            cohort_names.append(f"z_{cname}_{gname}")
            cohort_cols.append(within_group_rank(vals, g))
            cohort_names.append(f"rk_{cname}_{gname}")
            cohort_cols.append(within_group_log_ratio_median(vals, g))
            cohort_names.append(f"lrm_{cname}_{gname}")
    # 4*3*3 = 36 → keep 32
    X_coh_all = np.column_stack(cohort_cols[:32]).astype(np.float64)
    cohort_names = cohort_names[:32]

    # order tensor from L7 raw + gpt2
    def order_feats_from_llr(llr_mat):
        """llr_mat: (n, K)"""
        Z = col_z(llr_mat)
        feats = [
            gini10(Z),
            Z.std(1),
            np.median(np.abs(Z - np.median(Z, axis=1, keepdims=True)), axis=1),
            Z.max(1) - Z.min(1),
            Z.max(1) - np.median(Z, 1),
            Z.min(1) - np.median(Z, 1),
            llr_mat.std(1),
            (llr_mat * (llr_mat > 0)).std(1),
            (llr_mat * (llr_mat < 0)).std(1),
        ]
        # fwd-rev gap: first half vs second half seeds
        K = llr_mat.shape[1]
        h = K // 2
        if h >= 2:
            feats.append(llr_mat[:, :h].mean(1) - llr_mat[:, h : 2 * h].mean(1))
        return np.column_stack(feats)

    ord_o = order_feats_from_llr(llr_o)
    ord_t = np.mean(np.stack([order_feats_from_llr(llr_t[f]) for f in range(5)], 0), 0)

    # gpt2_med stitch OOF (optional; PCHIP-lock Test→OOF)
    GDIR = PRED / "gpt2_med_4ep"
    gpt_ord_o = gpt_ord_t = None
    if (GDIR / "oof_0.npy").exists() and (GDIR / "test_0.npy").exists():
        gpt_oof = np.zeros((n_tr, 4), dtype=np.float64)
        gpt_te = []
        for f in range(5):
            o = np.load(GDIR / f"oof_{f}.npy")  # (n_fold, 8, 2)
            t = np.load(GDIR / f"test_{f}.npy")  # (nte, 4, 2)
            idx = np.where(fold_id == f)[0]
            llr = o[:, :4, 1] - o[:, :4, 0]
            assert len(idx) == len(llr), (f, len(idx), len(llr))
            gpt_oof[idx] = llr
            gpt_te.append(t[:, :, 1] - t[:, :, 0])
        gpt_te_m = np.mean(np.stack(gpt_te, 0), 0)
        gpt_ord_o = order_feats_from_llr(gpt_oof)[:, :5]
        gpt_ord_t = order_feats_from_llr(gpt_te_m)[:, :5]
        for j in range(gpt_ord_t.shape[1]):
            gpt_ord_t[:, j] = qmap_to_ref(gpt_ord_t[:, j], gpt_ord_o[:, j])

    if gpt_ord_o is not None:
        X_ord_o = np.column_stack([ord_o, gpt_ord_o])
        X_ord_t = np.column_stack([ord_t, gpt_ord_t])
    else:
        X_ord_o, X_ord_t = ord_o.copy(), ord_t.copy()
    # L_fold stabilize order oof
    for j in range(X_ord_o.shape[1]):
        X_ord_o[:, j] = L_fold(X_ord_o[:, j], fold_id)
        X_ord_t[:, j] = L(X_ord_t[:, j])

    # X_tab45-ish (V17.8 digit keys + raw)
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
    amt_all = np.round(amt * 100).astype(np.int64)
    uniq, inv = np.unique(amt_all, return_inverse=True)
    cnt = np.bincount(inv)
    freq = np.log1p(cnt[inv])
    keys_tr["amt_freq_bin"] = np.clip((freq[:n_tr] * 5).astype(np.int64), 0, 50)
    keys_te["amt_freq_bin"] = np.clip((freq[n_tr:] * 5).astype(np.int64), 0, 50)
    for cat in ("merchant_category", "channel", "device_type"):
        codes, _ = pd.factorize(pd.concat([train[cat], test[cat]], ignore_index=True))
        keys_tr[f"{cat}_x_amtones"] = codes[:n_tr] * 10 + keys_tr["amount_usd_ones"]
        keys_te[f"{cat}_x_amtones"] = codes[n_tr:] * 10 + keys_te["amount_usd_ones"]
    key_names = list(keys_tr.keys())
    tab_tr = np.column_stack([keys_tr[k].astype(np.float64) for k in key_names] + [freq[:n_tr]])
    tab_te = np.column_stack([keys_te[k].astype(np.float64) for k in key_names] + [freq[n_tr:]])
    for c in ("amount_usd", "velocity_score", "merchant_risk_score", "distance_from_home_km"):
        tab_tr = np.column_stack([tab_tr, train[c].to_numpy(float)])
        tab_te = np.column_stack([tab_te, test[c].to_numpy(float)])
    # ratios
    ratio_all = np.column_stack(
        [
            np.log1p(np.abs(amt)) - np.log1p(np.abs(bal)),
            amt / (np.abs(bal) + np.abs(amt) + 1.0),
            mr * np.log1p(np.abs(amt)),
            vel * np.log1p(np.abs(amt)),
            (100.0 * U(amt)) % 1.0,
            (100.0 * U(bal)) % 1.0,
            (100.0 * U(vel)) % 1.0,
            (100.0 * U(mr)) % 1.0,
        ]
    )
    X_ratio_o, X_ratio_t = ratio_all[:n_tr], ratio_all[n_tr:]
    X_tab_o = np.column_stack([tab_tr, X_ratio_o])
    X_tab_t = np.column_stack([tab_te, X_ratio_t])

    X_coh_o, X_coh_t = X_coh_all[:n_tr], X_coh_all[n_tr:]

    # Pre-training fail-ray depoison on continuous new blocks
    # Use matched OOF/Test poison bases: M5+R1 (aligned with Slot7/Slot6 fail rays)
    poison_o = [u_m5, u_r1]
    poison_t = [ut_m5, ut_r1]
    X_coh_o = project_cols_out(X_coh_o, poison_o)
    X_coh_t = project_cols_out(X_coh_t, poison_t)
    X_ord_o = project_cols_out(X_ord_o, poison_o)
    X_ord_t = project_cols_out(X_ord_t, poison_t)
    X_ratio_o = project_cols_out(X_ratio_o, poison_o)
    X_ratio_t = project_cols_out(X_ratio_t, poison_t)

    X_new_o = np.column_stack([X_coh_o, X_ord_o, X_ratio_o]).astype(np.float32)
    X_new_t = np.column_stack([X_coh_t, X_ord_t, X_ratio_t]).astype(np.float32)
    X_cross_o = np.column_stack([X_new_o, tab_tr.astype(np.float32)])
    X_cross_t = np.column_stack([X_new_t, tab_te.astype(np.float32)])

    def check_disp(Xo, Xt, name):
        ok = True
        for j in range(Xo.shape[1]):
            so, st = float(np.std(Xo[:, j])), float(np.std(Xt[:, j]))
            if so < 1e-12:
                continue
            d = st / (so + 1e-12)
            if not (0.90 <= d <= 1.12):
                # soft scale
                Xt[:, j] *= so / (st + 1e-12)
                d = 1.0
            if np.isnan(Xo[:, j]).any() or np.isnan(Xt[:, j]).any():
                ok = False
        print(f"  {name}: cols={Xo.shape[1]} nan_ok={ok}")
        return Xt

    X_new_t = check_disp(X_new_o, X_new_t, "X_new_only")
    X_cross_t = check_disp(X_cross_o, X_cross_t, "X_cross")
    print(
        f"  shapes new={X_new_o.shape} cross={X_cross_o.shape} t={time.time()-t0:.1f}s"
    )

    # ---------- Step2: fit families ----------
    print("=== Step2 fit C1..C4 on (g_S8,h_S8) ===", f"t={time.time()-t0:.1f}s")
    ab, g5, h5 = [], [], []
    for tr, _ in folds:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s8[tr].reshape(-1, 1), y[tr])
        a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
        ab.append((a, b))
        p = expit(a * z_s8 + b)
        g5.append(y - p)
        h5.append(np.clip(p * (1 - p), 1e-6, None))

    def fit_cat_bag(Xtr, Xte, depths, tag, iterations=120, lr=0.035, l2=30.0, rsm=0.8):
        oof_acc, te_acc = [], []
        for depth in depths:
            for seed in SEEDS:
                oo = np.zeros(n_tr)
                tef = []
                for f, (tr, va) in enumerate(folds):
                    m = CatBoostRegressor(
                        loss_function="RMSE",
                        depth=depth,
                        l2_leaf_reg=l2,
                        iterations=iterations,
                        learning_rate=lr,
                        random_seed=seed,
                        verbose=False,
                        thread_count=4,
                        bootstrap_type="Bernoulli",
                        subsample=0.8,
                        rsm=rsm,
                    )
                    m.fit(Xtr[tr], g5[f][tr], sample_weight=h5[f][tr])
                    oo[va] = m.predict(Xtr[va])
                    tef.append(m.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        oof = np.mean(np.stack(oof_acc, 0), 0)
        te = np.mean(np.stack(te_acc, 0), 0)
        print(f"  {tag}: n_models={len(oof_acc)} t={time.time()-t0:.1f}s")
        return oof, te

    def fit_et_lgb(Xtr, Xte, tag):
        oof_acc, te_acc = [], []
        for leaves in (5, 8, 12):
            for seed in (7, 17, 27, 37, 47):
                oo = np.zeros(n_tr)
                tef = []
                for f, (tr, va) in enumerate(folds):
                    m = lgb.LGBMRegressor(
                        objective="regression",
                        learning_rate=0.03,
                        num_leaves=leaves,
                        min_child_samples=600,
                        n_estimators=50,
                        subsample=0.80,
                        colsample_bytree=0.60,
                        reg_lambda=30.0,
                        extra_trees=True,
                        verbosity=-1,
                        n_jobs=4,
                        random_state=seed,
                    )
                    m.fit(Xtr[tr], g5[f][tr], sample_weight=h5[f][tr])
                    oo[va] = m.predict(Xtr[va])
                    tef.append(m.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        oof = np.mean(np.stack(oof_acc, 0), 0)
        te = np.mean(np.stack(te_acc, 0), 0)
        print(f"  {tag}: n_models={len(oof_acc)} t={time.time()-t0:.1f}s")
        return oof, te

    C1_o, C1_t = fit_cat_bag(
        X_new_o, X_new_t, depths=(3, 4, 5), tag="C1_cat_new_only", iterations=160, l2=30, rsm=0.80
    )
    C2_o, C2_t = fit_cat_bag(
        X_cross_o,
        X_cross_t,
        depths=(4, 5, 6),
        tag="C2_cat_cross88",
        iterations=160,
        l2=35,
        rsm=0.70,
    )
    C3a_o, C3a_t = fit_et_lgb(X_new_o, X_new_t, "C3a_et_new")
    C3b_o, C3b_t = fit_et_lgb(X_cross_o, X_cross_t, "C3b_et_cross")
    C3_o = 0.5 * C3a_o + 0.5 * C3b_o
    C3_t = 0.5 * C3a_t + 0.5 * C3b_t

    # C4: coarse-only peer keys (no fine k1..k8 collisions)
    print("  C4 SafePeer coarse keys…", f"t={time.time()-t0:.1f}s")
    z_sym = np.concatenate([unitize(z_s8), unitize(L(slot8))])
    amt_r10 = (np.round(amt / 10.0) * 10.0).astype(np.int64).astype(str)
    bal_r100 = (np.round(bal / 100.0) * 100.0).astype(np.int64).astype(str)
    mr_r1 = np.round(mr, 1).astype(str)
    vel_r0 = np.round(vel, 0).astype(np.int64).astype(str)
    amt_ones = (np.abs(amt).astype(np.int64) % 10).astype(str)
    bal_ones = (np.abs(bal).astype(np.int64) % 10).astype(str)
    k_new = {
        "k_new1": codes_from_str(
            (pd.Series(mer) + "|" + pd.Series(ch) + "|" + pd.Series(dev) + "|" + pd.Series(amt_r10)).to_numpy()
        ),
        "k_new2": codes_from_str(
            (pd.Series(mer) + "|" + pd.Series(foreign) + "|" + pd.Series(bal_r100)).to_numpy()
        ),
        "k_new3": codes_from_str(
            (
                pd.Series(ch)
                + "|"
                + pd.Series(dev)
                + "|"
                + pd.Series(mr_r1)
                + "|"
                + pd.Series(vel_r0)
            ).to_numpy()
        ),
        "k_new4": codes_from_str(
            (pd.Series(mer) + "|" + pd.Series(amt_ones) + "|" + pd.Series(bal_ones)).to_numpy()
        ),
    }
    peer_cols_o, peer_cols_t = [], []
    for kn, codes in k_new.items():
        peer, cnt = leave_self_peer_mean(z_sym, codes, m0=25.0)
        # keep only medium support buckets signal; zero tiny/huge extremes softly via clip
        ok = (cnt >= 50) & (cnt <= 800)
        peer = peer.copy()
        peer[~ok] = 0.0
        peer_cols_o.append(peer[:n_tr])
        peer_cols_t.append(peer[n_tr:])
        print(f"    {kn}: mean_cnt={cnt.mean():.1f} frac_mid={ok.mean():.3f}")
    X_c4_o = np.column_stack(peer_cols_o + [X_coh_o]).astype(np.float32)
    X_c4_t = np.column_stack(peer_cols_t + [X_coh_t]).astype(np.float32)
    X_c4_t = check_disp(X_c4_o, X_c4_t, "X_c4")
    C4_o, C4_t = fit_cat_bag(
        X_c4_o, X_c4_t, depths=(3, 4), tag="C4_SafePeer_CoarseOnly", iterations=80, l2=30, rsm=0.85
    )

    np.savez_compressed(
        OUT / "v19_11_families.npz",
        C1_oof=C1_o,
        C1_test=C1_t,
        C2_oof=C2_o,
        C2_test=C2_t,
        C3_oof=C3_o,
        C3_test=C3_t,
        C4_oof=C4_o,
        C4_test=C4_t,
    )

    # ---------- Step3: audit ----------
    print("=== Step3 audit vs Slot8 ===", f"t={time.time()-t0:.1f}s")
    families = {
        "C1_cat_new_only": (C1_o, C1_t),
        "C2_cat_cross88": (C2_o, C2_t),
        "C3_et_cross": (C3_o, C3_t),
        "C4_SafePeer_Coarse": (C4_o, C4_t),
    }
    # blends
    for a, b, na, nb in [
        (0.5, 0.5, "C1", "C2"),
        (0.5, 0.5, "C1", "C3"),
        (0.6, 0.4, "C2", "C3"),
        (0.4, 0.3, "C1", "C4"),
    ]:
        # handled below via explicit mix
        pass
    families["B12"] = (0.5 * C1_o + 0.5 * C2_o, 0.5 * C1_t + 0.5 * C2_t)
    families["B13"] = (0.5 * C1_o + 0.5 * C3_o, 0.5 * C1_t + 0.5 * C3_t)
    families["B23"] = (0.6 * C2_o + 0.4 * C3_o, 0.6 * C2_t + 0.4 * C3_t)
    families["B14"] = (0.7 * C1_o + 0.3 * C4_o, 0.7 * C1_t + 0.3 * C4_t)
    families["B123"] = (
        0.4 * C1_o + 0.35 * C2_o + 0.25 * C3_o,
        0.4 * C1_t + 0.35 * C2_t + 0.25 * C3_t,
    )

    winners = []
    rows = []

    def evaluate(name, vo, vt):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo12, Bt12, folds, ridge_alpha=10.0)
        # Pre-scale forensic (raw_disp gate), then equalize OOF/Test scale for scoring
        gates = forensic(oo, ot, Bt12)
        scale = float(np.std(oo) / (np.std(ot) + 1e-12))
        ot = ot * scale
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s8[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        # F3 vs live resid proxy
        live_resid = unitize(L(slot8) - L(slot5))
        f3 = abs(float(np.corrcoef(ut, live_resid)[0, 1]))
        print(
            f"  {name}: disp={gates['raw_disp']:.3f} KS={gates['raw_KS']:.3f} "
            f"F2={gates['F2']:.4f} F3={f3:.4f} corrR2={np.corrcoef(u,u_r2)[0,1]:+.3f} "
            f"corrCat={np.corrcoef(u,u_cat)[0,1]:+.3f}"
        )
        local = []
        for w in (0.008, 0.010, 0.012, 0.015, 0.018, 0.022):
            oof = {b: bases[b] + w * u for b in bases}
            per = {b: h2h(oof[b], bases[b], y, folds) for b in bases}
            cand = qlock(L(slot8) + w * ut, lv)
            rho = one_m_rho_e4(cand, slot8)
            delta = unitize(L(cand) - L(slot8))
            c6 = float(np.corrcoef(delta, bad6)[0, 1])
            c7 = float(np.corrcoef(delta, bad7)[0, 1])
            ok = (
                all(
                    per[b]["pos"] >= 5
                    and per[b]["min"] > 0.0
                    and per[b]["sf25"] >= 18
                    and per[b]["t"] >= 2.8
                    for b in per
                )
                and (0.30 <= rho <= 1.20)
                and abs(c6) < 0.20
                and abs(c7) < 0.20
                and gates["F1"]
                and gates["F2"] < 0.040
                and f3 < 0.045
                and gates["raw_KS"] < 0.025
            )
            mark = "★" if ok else ("~" if per["z_S8"]["pos"] >= 4 else " ")
            print(
                f"{mark} {name}_w{w:.3f}: 1-ρ={rho:.3f} c6={c6:+.3f} c7={c7:+.3f} ok={ok} | "
                + " | ".join(
                    f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f} "
                    f"sf{per[b]['sf25']} t={per[b]['t']:.2f}"
                    for b in bases
                )
            )
            row = {
                "name": f"{name}_w{w:.3f}",
                "family": name,
                "w": w,
                "per": per,
                "rho": rho,
                "c6": c6,
                "c7": c7,
                "gates": gates,
                "f3": f3,
                "ok": ok,
                "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                "min_lift": float(np.min([per[b]["min"] for b in per])),
                "cand": cand if ok else None,
            }
            local.append({k: v for k, v in row.items() if k != "cand"})
            if ok:
                winners.append(row)
        # also joint reopt small grid on Slot3
        for wo, wc, wu, wn in [
            (0.014, 0.010, 0.010, 0.010),
            (0.016, 0.010, 0.010, 0.012),
            (0.016, 0.012, 0.010, 0.010),
            (0.014, 0.012, 0.012, 0.012),
            (0.015, 0.011, 0.011, 0.014),
        ]:
            oof = {
                "z_S8": z_s3 + wo * u_r2 + wc * u_cat + wu * u_u5 + wn * u,
                "twin": z_twin + wo * u_r2 + wc * u_cat + wu * u_u5 + wn * u,
                "nnls": z_nnls + wo * u_r2 + wc * u_cat + wu * u_u5 + wn * u,
            }
            per = {b: h2h(oof[b], bases[b], y, folds) for b in bases}
            cand = qlock(
                L(slot3) + wo * ut_r2 + wc * ut_cat + wu * ut_u5 + wn * ut, lv
            )
            rho = one_m_rho_e4(cand, slot8)
            delta = unitize(L(cand) - L(slot8))
            c6 = float(np.corrcoef(delta, bad6)[0, 1])
            c7 = float(np.corrcoef(delta, bad7)[0, 1])
            ok = (
                all(
                    per[b]["pos"] >= 5
                    and per[b]["min"] > 0.0
                    and per[b]["sf25"] >= 18
                    and per[b]["t"] >= 2.8
                    for b in per
                )
                and (0.30 <= rho <= 1.20)
                and abs(c6) < 0.20
                and abs(c7) < 0.20
                and gates["F1"]
                and gates["F2"] < 0.040
                and f3 < 0.045
            )
            mark = "★" if ok else ("~" if per["z_S8"]["pos"] >= 4 else " ")
            jn = f"{name}_joint_wo{wo:.3f}_wc{wc:.3f}_wu{wu:.3f}_wn{wn:.3f}"
            print(
                f"{mark} {jn}: 1-ρ={rho:.3f} c7={c7:+.3f} ok={ok} | "
                + " | ".join(
                    f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f} "
                    f"t={per[b]['t']:.2f}"
                    for b in bases
                )
            )
            local.append(
                {
                    "name": jn,
                    "family": name,
                    "w": wn,
                    "per": per,
                    "rho": rho,
                    "c6": c6,
                    "c7": c7,
                    "gates": gates,
                    "f3": f3,
                    "ok": ok,
                    "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                    "min_lift": float(np.min([per[b]["min"] for b in per])),
                    "joint": {"wo": wo, "wc": wc, "wu": wu, "wn": wn},
                }
            )
            if ok:
                winners.append(
                    {
                        "name": jn,
                        "family": name,
                        "w": wn,
                        "per": per,
                        "rho": rho,
                        "c6": c6,
                        "c7": c7,
                        "gates": gates,
                        "f3": f3,
                        "ok": True,
                        "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                        "min_lift": float(np.min([per[b]["min"] for b in per])),
                        "cand": cand,
                        "joint": {"wo": wo, "wc": wc, "wu": wu, "wn": wn},
                        "ut": ut,
                        "u": u,
                    }
                )
        rows.extend(local)
        return u, ut, gates

    oriented = {}
    for name, (vo, vt) in families.items():
        oriented[name] = evaluate(name, vo, vt)

    print(f"\nWinners: {len(winners)}")
    submit = False
    pub, ref, chosen = None, None, None
    if winners:
        winners.sort(
            key=lambda r: (r["min_lift"], r["mean_lift"], -abs(r["rho"] - 0.50)), reverse=True
        )
        chosen = winners[0]
        print(
            f"CHOSEN {chosen['name']} mean={chosen['mean_lift']:+.2f} "
            f"min={chosen['min_lift']:+.2f} 1-ρ={chosen['rho']:.3f} c7={chosen['c7']:+.3f}"
        )
        out_csv = SUB / "sub_v19_11_LIVE_slot9_cross_cohort_order_cat.csv"
        pd.DataFrame({"id": ids, "is_fraud": chosen["cand"]}).to_csv(out_csv, index=False)
        msg = (
            f"v19.11 Slot9: {chosen['name']} cross-space vs Slot8 ATH 0.95395 | "
            f"H2H mean={chosen['mean_lift']:+.2f}e-5 min={chosen['min_lift']:+.2f} "
            f"1-ρ={chosen['rho']:.3f}e-4 c7={chosen['c7']:+.3f}"
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
                str(out_csv),
                "-m",
                msg,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        print(r.stdout)
        print(r.stderr)
        if r.returncode == 0:
            submit = True
            for i in range(30):
                time.sleep(12 if i else 4)
                p = subprocess.run(
                    [
                        KAG,
                        "competitions",
                        "submissions",
                        "-c",
                        "playground-series-s6e11",
                        "--csv",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                lines = [ln for ln in p.stdout.strip().splitlines() if ln]
                if len(lines) < 2:
                    continue
                hdr = [h.strip() for h in lines[0].split(",")]
                for ln in lines[1:5]:
                    cols = ln.split(",")
                    row = {hdr[j]: cols[j].strip() for j in range(min(len(hdr), len(cols)))}
                    if "v19_11" in row.get("fileName", ""):
                        ref = row.get("ref")
                        st = row.get("status", "")
                        print(f"  poll {st} score={row.get('publicScore')}")
                        if "complete" in st.lower() and row.get("publicScore"):
                            pub = row["publicScore"]
                            break
                if pub:
                    break
            print(f"PUBLIC_LB={pub} (ATH Slot8=0.95395)")
    else:
        print("NO WINNER — HOLD 0.95395 / 2 slots")

    near = sorted(
        rows,
        key=lambda c: (
            c["per"]["z_S8"]["pos"],
            c["per"]["twin"]["pos"],
            -abs(c.get("c7", 1)),
            c["mean_lift"],
        ),
        reverse=True,
    )[:20]
    payload = scrub(
        {
            "submit": submit,
            "n_winners": len(winners),
            "chosen": None
            if not chosen
            else {
                "name": chosen["name"],
                "mean_lift": chosen["mean_lift"],
                "min_lift": chosen["min_lift"],
                "rho": chosen["rho"],
                "c6": chosen["c6"],
                "c7": chosen["c7"],
                "per": chosen["per"],
            },
            "public_lb": pub,
            "ref": ref,
            "near_misses": near,
            "shapes": {
                "X_new": list(X_new_o.shape),
                "X_cross": list(X_cross_o.shape),
                "X_c4": list(X_c4_o.shape),
            },
            "elapsed_s": time.time() - t0,
        }
    )
    OUT.joinpath("v19_11.json").write_text(json.dumps(payload, indent=2))

    # report
    lines = [
        "# GM V19.11 — Cross-space cohort×order Cat/ET on Slot8 → "
        + ("**LIVE Slot9**" if submit else "**HOLD**"),
        "",
        f"**0 GPU pods · elapsed {payload['elapsed_s']:.0f}s**",
        "",
        f"| | |",
        f"|---|---|",
        f"| Slot8 ATH | 0.95395 |",
        f"| Submit | {submit} |",
        f"| Public LB | {pub} |",
        f"| Winners | {len(winners)} |",
        "",
        "## Shapes",
        f"- X_new_only: {X_new_o.shape}",
        f"- X_cross: {X_cross_o.shape}",
        f"- X_c4: {X_c4_o.shape}",
        "",
        "## Top near-misses",
    ]
    for c in near[:12]:
        lines.append(
            f"- `{c['name']}`: 1-ρ={c['rho']:.3f} c7={c['c7']:+.3f} "
            f"z_S8 {c['per']['z_S8']['pos']}/5 m={c['per']['z_S8']['mean']:+.2f} "
            f"twin {c['per']['twin']['pos']}/5 m={c['per']['twin']['mean']:+.2f} ok={c['ok']}"
        )
    if chosen:
        lines += ["", f"## Chosen `{chosen['name']}`", f"- mean={chosen['mean_lift']:+.2f} min={chosen['min_lift']:+.2f}"]
    else:
        lines += ["", "## Decision", "HOLD ATH `0.95395`. Keep remaining daily slots."]
    report = LOG / "GM_V19_11_SLOT9_CROSS_SPACE_REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print("WROTE", report)
    print("WROTE", OUT / "v19_11.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
