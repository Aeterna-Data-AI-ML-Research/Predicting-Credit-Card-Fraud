#!/usr/bin/env python3
"""GM V20.0 — Unbiased Wilcoxon ROC-AUC functional GB + Method-B Bo12 whitening + clean P4.

Fixes: no |g|*|g| weight explosion; Method-B F2_Bo12 lock; ban KMAT/quarantine L4.3;
train on 100% of tr. Submit Slot9 IFF all 6 gates. 0 GPU pods.
"""
from __future__ import annotations

import json
import subprocess
import time
import warnings
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from scipy.signal import fftconvolve
from scipy.special import expit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parents[1]
TMP15, TMP16 = Path("/tmp/s6e11_v15"), Path("/tmp/s6e11_v16")
OUT, SUB, DATA, LOG = TMP16 / "out", REPO / "submissions", REPO / "data", REPO / "logs/overnight"
P4DIR = REPO / "preds/matrix_v2"
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
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W8_O, W8_C, W8_U = 0.016, 0.012, 0.012
SEEDS = (11, 21, 31, 41)


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
    raw, ref = np.asarray(raw, float), np.asarray(ref, float)
    out = np.empty_like(ref)
    out[np.argsort(raw)] = np.sort(ref)
    return out


def qmap_to_ref(raw, ref):
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


def var_match_cols(cols_o, cols_t):
    """Variance-match each OOF/Test column pair then unitize."""
    Io, It = [], []
    for co, ct in zip(cols_o, cols_t):
        co = np.asarray(co, float).ravel()
        ct = np.asarray(ct, float).ravel()
        so, st = co.std() + 1e-12, ct.std() + 1e-12
        ct = (ct - ct.mean()) * (so / st) + co.mean()
        Io.append(unitize(co))
        It.append(unitize(ct))
    return np.column_stack(Io), np.column_stack(It)


def method_b_orth12(vo, vt, Bo9, Bt9, incr_o, incr_t, folds, ridge_alpha=10.0):
    """Bo9 frozen orth, then variance-matched orth vs Slot8-increment basis."""
    oo9, ot9, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=ridge_alpha)
    Io, It = var_match_cols(incr_o, incr_t)
    oo12, ot12, _ = frozen_lofo_orth(oo9, ot9, Io, It, folds, ridge_alpha=ridge_alpha)
    return oo12, ot12, oo9, ot9


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


def compute_unbiased_wilcoxon_grad(z_tr, y_tr, taus=(0.12, 0.25, 0.50), n_bins=4000):
    """Unweighted multi-bandwidth Wilcoxon ROC-AUC functional gradient on tr."""
    z = unitize(z_tr)
    y = np.asarray(y_tr, float).ravel()
    ybar = float(y.mean())
    zmin, zmax = float(z.min() - 2.0), float(z.max() + 2.0)
    edges = np.linspace(zmin, zmax, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dz = float(centers[1] - centers[0])
    pos, neg = z[y > 0.5], z[y < 0.5]
    hp, _ = np.histogram(pos, bins=edges, density=True)
    hn, _ = np.histogram(neg, bins=edges, density=True)
    hp = hp.astype(np.float64)
    hn = hn.astype(np.float64)

    grads = []
    for tau in taus:
        half = int(min(n_bins // 2 - 1, max(16, int(10.0 * tau / dz))))
        u = np.arange(-half, half + 1, dtype=np.float64) * dz
        s = expit(u / tau)
        K = (1.0 / tau) * s * (1.0 - s)
        # density convolution
        conv_n = fftconvolve(hn, K, mode="same") * dz
        conv_p = fftconvolve(hp, K, mode="same") * dz
        from_n = np.interp(z, centers, conv_n)
        from_p = np.interp(z, centers, conv_p)
        g = np.zeros_like(z)
        g[y > 0.5] = from_n[y > 0.5] / (ybar + 1e-12)
        g[y < 0.5] = -from_p[y < 0.5] / (1.0 - ybar + 1e-12)
        g = g - g.mean()
        g = g / (g.std() + 1e-12)
        grads.append(g)

    g_auc = unitize(0.35 * grads[0] + 0.40 * grads[1] + 0.25 * grads[2])
    return g_auc, grads


def clip_z(v, lo=-3.0, hi=3.0):
    return np.clip(unitize(v), lo, hi)


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

    def orient_bo9(vo, vt, base=None):
        base = z_s3 if base is None else base
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=10.0)
        u = unitize(oo)
        if roc_auc_score(y[te0], base[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], base[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        return unitize(u), unitize(ot)

    u_r2, ut_r2 = orient_bo9(fam1["R2_v178_orig_oof"], fam1["R2_v178_orig_test"])
    u_cat, ut_cat = orient_bo9(fam1["F_cat_obliv_oof"], fam1["F_cat_obliv_test"])
    u_m5, ut_m5 = orient_bo9(peers["M5_oof"], peers["M5_test"])
    u_r1, ut_r1 = orient_bo9(r178["R1_oof"], r178["R1_test"])

    raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
    llr_o = raw_o[:, :, 1] - raw_o[:, :, 0]
    Z_o = col_z(llr_o)
    U5_o = L_fold(gini10(Z_o), fold_id)
    U5_t = np.mean(
        np.stack([L(gini10(col_z(raw_t[f, :, :, 1] - raw_t[f, :, :, 0]))) for f in range(5)], 0),
        0,
    )
    Bo10 = np.column_stack([Bo9, u_r2])
    Bt10 = np.column_stack([Bt9, ut_r2])
    oo5, ot5, _ = frozen_lofo_orth(U5_o, U5_t, Bo10, Bt10, folds, ridge_alpha=1.0)
    u_u5 = unitize(oo5)
    z_s5_orient = z_s3 + 0.020 * u_r2
    if roc_auc_score(y[te0], z_s5_orient[te0] + 0.01 * u_u5[te0]) < roc_auc_score(
        y[te0], z_s5_orient[te0] - 0.01 * u_u5[te0]
    ):
        u_u5, ot5 = -u_u5, -ot5
    ut_u5 = unitize(ot5)

    # U6/U7/U8 for W3 / interactions
    def u_bank_from_llr(ll_pos, ll_neg):
        llr = ll_pos - ll_neg
        Z = col_z(llr)
        K = Z.shape[1]
        h = K // 2
        pair_means = np.column_stack([0.5 * (Z[:, m] + Z[:, m + h]) for m in range(h)])
        pair_gaps = np.column_stack([np.abs(Z[:, m] - Z[:, m + h]) for m in range(h)])
        Zp, Zn = col_z(ll_pos), col_z(ll_neg)
        return {
            "U5": gini10(Z),
            "U6": pair_means.std(1),
            "U7": pair_gaps.mean(1),
            "U8": Zp.std(1) - Zn.std(1),
        }

    bank_o = u_bank_from_llr(raw_o[:, :, 1], raw_o[:, :, 0])
    bank_t = {k: [] for k in bank_o}
    for f in range(5):
        bf = u_bank_from_llr(raw_t[f, :, :, 1], raw_t[f, :, :, 0])
        for k in bank_t:
            bank_t[k].append(bf[k])
    U_o, U_t = {}, {}
    for k in bank_o:
        U_o[k] = L_fold(bank_o[k], fold_id)
        U_t[k] = np.mean(np.stack([L(x) for x in bank_t[k]], 0), 0)

    z_s8 = z_s3 + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_t8 = z_twin + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_n8 = z_nnls + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    bases = {"z_S8": z_s8, "twin": z_t8, "nnls": z_n8}

    rebuild = qlock(L(slot3) + W8_O * ut_r2 + W8_C * ut_cat + W8_U * ut_u5, lv)
    rho_rebuild = one_m_rho_e4(rebuild, slot8)
    print(f"=== V20.0 === Slot8 rebuild 1-ρ={rho_rebuild:.6e}e-4 t={time.time()-t0:.1f}s")
    assert rho_rebuild < 1e-5, rho_rebuild

    Bo12 = np.column_stack([Bo9, u_r2, u_cat, u_u5])
    Bt12 = np.column_stack([Bt9, ut_r2, ut_cat, ut_u5])
    incr_o = [u_r2, u_cat, u_u5, u_m5, u_r1]
    incr_t = [ut_r2, ut_cat, ut_u5, ut_m5, ut_r1]

    # ---------- Wilcoxon gradients per fold (on full tr) ----------
    print("=== Step1 Wilcoxon g_AUC ===", f"t={time.time()-t0:.1f}s")
    g_auc_folds, g_hyb_folds, g_bce_folds, h_bce_folds = [], [], [], []
    twilight_share = []
    for f, (tr, _) in enumerate(folds):
        # BCE Newton on Slot8 Platt
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s8[tr].reshape(-1, 1), y[tr])
        a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
        p = expit(a * z_s8 + b)
        g_bce = y - p
        h_bce = np.clip(p * (1 - p), 1e-6, None)
        g_auc, _ = compute_unbiased_wilcoxon_grad(z_s8[tr], y[tr])
        # map back to full length (only tr defined; others 0 — train uses tr mask)
        g_auc_full = np.zeros(n_tr)
        g_auc_full[tr] = g_auc
        g_bce_u = unitize(g_bce[tr])
        g_hyb_full = np.zeros(n_tr)
        g_hyb_full[tr] = unitize(0.50 * g_auc + 0.50 * g_bce_u)
        g_auc_folds.append(g_auc_full)
        g_hyb_folds.append(g_hyb_full)
        g_bce_folds.append(g_bce)
        h_bce_folds.append(h_bce)
        corr = float(np.corrcoef(g_auc, g_bce_u)[0, 1])
        # twilight energy share
        r = U(z_s8[tr])
        tw = (r >= 0.75) & (r <= 0.995)
        e_auc = float((g_auc[tw] ** 2).sum() / ((g_auc**2).sum() + 1e-12))
        e_bce = float((g_bce_u[tw] ** 2).sum() / ((g_bce_u**2).sum() + 1e-12))
        twilight_share.append({"fold": f, "corr_auc_bce": corr, "e_auc_tw": e_auc, "e_bce_tw": e_bce})
        print(
            f"  fold{f}: corr(g_AUC,g_BCE)={corr:+.3f} twilight_E auc={e_auc:.3f} bce={e_bce:.3f} "
            f"n_tr={len(tr)}"
        )

    # ---------- X_tab45 ----------
    print("=== Step1 X_tab45 ===", f"t={time.time()-t0:.1f}s")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    assert len(train) == n_tr and len(test) == n_te

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
        np.concatenate([train["amount_usd"].to_numpy(float), test["amount_usd"].to_numpy(float)])
        * 100
    ).astype(np.int64)
    uniq, inv = np.unique(amt_all, return_inverse=True)
    cnt = np.bincount(inv)
    freq = np.log1p(cnt[inv].astype(np.float64))
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
            orig_cnt_tr = np.array([ctr.get(int(x), 0) for x in amt_all[:n_tr]], dtype=np.float64)
            orig_cnt_te = np.array([ctr.get(int(x), 0) for x in amt_all[n_tr:]], dtype=np.float64)
            keys_tr["amt_orig_bin"] = np.clip(orig_cnt_tr.astype(np.int64), 0, 20)
            keys_te["amt_orig_bin"] = np.clip(orig_cnt_te.astype(np.int64), 0, 20)
            orig_cnt = (orig_cnt_tr, orig_cnt_te)
    for cat in ("merchant_category", "channel", "device_type"):
        codes, _ = pd.factorize(pd.concat([train[cat], test[cat]], ignore_index=True))
        keys_tr[f"{cat}_x_amtones"] = codes[:n_tr] * 10 + keys_tr["amount_usd_ones"]
        keys_te[f"{cat}_x_amtones"] = codes[n_tr:] * 10 + keys_te["amount_usd_ones"]
    key_names = list(keys_tr.keys())

    def build_X45(split="train"):
        cols, names = [], []
        src = keys_tr if split == "train" else keys_te
        for kn in key_names:
            cols.append(src[kn].astype(np.float64))
            names.append(kn)
        if split == "train":
            cols.append(feat_freq_tr)
            names.append("amt_logfreq")
            for c in ("amount_usd", "velocity_score", "merchant_risk_score", "distance_from_home_km"):
                cols.append(train[c].to_numpy(float))
                names.append(c)
            if orig_cnt is not None:
                cols.append(orig_cnt[0])
                names.append("orig20k_hit_count")
        else:
            cols.append(feat_freq_te)
            names.append("amt_logfreq")
            for c in ("amount_usd", "velocity_score", "merchant_risk_score", "distance_from_home_km"):
                cols.append(test[c].to_numpy(float))
                names.append(c)
            if orig_cnt is not None:
                cols.append(orig_cnt[1])
                names.append("orig20k_hit_count")
        return np.column_stack(cols).astype(np.float32), names

    X45_tr, x45_names = build_X45("train")
    X45_te, _ = build_X45("test")
    print(f"  X_tab45 cols={X45_tr.shape[1]} t={time.time()-t0:.1f}s")

    # U58 matrix for W3
    Xu_tr = np.column_stack([U_o[k] for k in ("U5", "U6", "U7", "U8")]).astype(np.float32)
    Xu_te = np.column_stack([U_t[k] for k in ("U5", "U6", "U7", "U8")]).astype(np.float32)
    X45u_tr = np.column_stack([X45_tr, Xu_tr]).astype(np.float32)
    X45u_te = np.column_stack([X45_te, Xu_te]).astype(np.float32)

    # ---------- Fit W1..W3 ----------
    print("=== Step2 fit W1..W3 ===", f"t={time.time()-t0:.1f}s")

    def fit_cat_unweighted(Xtr, Xte, targets, tag, depths=(3, 4, 5), iterations=150, l2=35.0):
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
                        learning_rate=0.035,
                        random_seed=seed,
                        verbose=False,
                        thread_count=4,
                        bootstrap_type="Bernoulli",
                        subsample=0.8,
                        rsm=0.80,
                    )
                    m.fit(Xtr[tr], targets[f][tr])  # sample_weight=None
                    oo[va] = m.predict(Xtr[va])
                    tef.append(m.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        print(f"  {tag}: n={len(oof_acc)} t={time.time()-t0:.1f}s")
        return np.mean(np.stack(oof_acc, 0), 0), np.mean(np.stack(te_acc, 0), 0)

    def fit_lgb_unweighted(Xtr, Xte, targets, tag, leaves=(4, 6, 8)):
        oof_acc, te_acc = [], []
        for nl in leaves:
            for seed in SEEDS:
                oo = np.zeros(n_tr)
                tef = []
                for f, (tr, va) in enumerate(folds):
                    m = lgb.LGBMRegressor(
                        num_leaves=nl,
                        min_child_samples=500,
                        n_estimators=45,
                        learning_rate=0.03,
                        colsample_bytree=0.75,
                        subsample=0.80,
                        reg_lambda=25.0,
                        verbosity=-1,
                        n_jobs=4,
                        random_state=seed,
                    )
                    m.fit(Xtr[tr], targets[f][tr])
                    oo[va] = m.predict(Xtr[va])
                    tef.append(m.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        print(f"  {tag}: n={len(oof_acc)} t={time.time()-t0:.1f}s")
        return np.mean(np.stack(oof_acc, 0), 0), np.mean(np.stack(te_acc, 0), 0)

    def fit_hyb_weighted(Xtr, Xte, g_hyb, h_bce, tag):
        oof_acc, te_acc = [], []
        for depth in (3, 4):
            for seed in SEEDS[:4]:
                oo = np.zeros(n_tr)
                tef = []
                for f, (tr, va) in enumerate(folds):
                    m = CatBoostRegressor(
                        loss_function="RMSE",
                        depth=depth,
                        l2_leaf_reg=35,
                        iterations=120,
                        learning_rate=0.035,
                        random_seed=seed,
                        verbose=False,
                        thread_count=4,
                        bootstrap_type="Bernoulli",
                        subsample=0.8,
                    )
                    m.fit(Xtr[tr], g_hyb[f][tr], sample_weight=h_bce[f][tr])
                    oo[va] = m.predict(Xtr[va])
                    tef.append(m.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        for nl in (5, 7):
            for seed in SEEDS[:4]:
                oo = np.zeros(n_tr)
                tef = []
                for f, (tr, va) in enumerate(folds):
                    m = lgb.LGBMRegressor(
                        num_leaves=nl,
                        min_child_samples=500,
                        n_estimators=45,
                        learning_rate=0.03,
                        colsample_bytree=0.75,
                        subsample=0.8,
                        reg_lambda=25.0,
                        verbosity=-1,
                        n_jobs=4,
                        random_state=seed,
                    )
                    m.fit(Xtr[tr], g_hyb[f][tr], sample_weight=h_bce[f][tr])
                    oo[va] = m.predict(Xtr[va])
                    tef.append(m.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        print(f"  {tag}: n={len(oof_acc)} t={time.time()-t0:.1f}s")
        return np.mean(np.stack(oof_acc, 0), 0), np.mean(np.stack(te_acc, 0), 0)

    W1_o, W1_t = fit_cat_unweighted(X45_tr, X45_te, g_auc_folds, "W1_wilcoxon_cat_tab45")
    W2_o, W2_t = fit_lgb_unweighted(X45_tr, X45_te, g_auc_folds, "W2_wilcoxon_lgb_tab45")
    W3_o, W3_t = fit_hyb_weighted(
        X45u_tr, X45u_te, g_hyb_folds, h_bce_folds, "W3_hyb_cat_lgb_tab45_U58"
    )

    # ---------- W4 clean P4 ----------
    print("=== Step2 W4 clean P4 ===", f"t={time.time()-t0:.1f}s")
    p4_keep = []
    p4_raw = {}
    for tag in ("P4_L4.2_pure5f", "P4_L4.3_pure5f", "P4_L4.5_pure5f"):
        po, pt = P4DIR / f"{tag}_oof.npy", P4DIR / f"{tag}_test.npy"
        if not (po.exists() and pt.exists()):
            continue
        oo = L_fold(np.load(po).ravel()[:n_tr], fold_id)
        ot = L(np.load(pt).ravel()[:n_te])
        ks = float(ks_2samp(unitize(oo), unitize(ot)).statistic)
        disp = float(np.std(ot) / (np.std(oo) + 1e-12))
        print(f"  {tag}: pre-PCHIP KS={ks:.4f} disp={disp:.3f}")
        if tag == "P4_L4.3_pure5f" and ks >= 0.020:
            print(f"  QUARANTINE {tag} (KS>={0.020})")
            continue
        if ks >= 0.020:
            print(f"  SKIP {tag} KS too high")
            continue
        ot = qmap_to_ref(ot, oo)
        # orth vs Bo12+M5+R1
        Bo_p = np.column_stack([Bo12, u_m5, u_r1])
        Bt_p = np.column_stack([Bt12, ut_m5, ut_r1])
        oo2, ot2, _ = frozen_lofo_orth(oo, ot, Bo_p, Bt_p, folds, ridge_alpha=10.0)
        p4_raw[tag] = (unitize(oo2), unitize(ot2))
        p4_keep.append(tag)
        print(f"  KEEP {tag}")

    # interaction features
    name_to_idx = {n: i for i, n in enumerate(x45_names)}
    inter_names = [
        "amt_logfreq",
        "account_balance_usd_cents",
        "merchant_category_x_amtones",
        "velocity_score_q1000",
    ]
    tab_sig_o = [u_r2, u_cat, u_u5, unitize(U_o["U8"])]
    tab_sig_t = [ut_r2, ut_cat, ut_u5, unitize(U_t["U8"])]
    for nm in inter_names:
        if nm in name_to_idx:
            tab_sig_o.append(X45_tr[:, name_to_idx[nm]])
            tab_sig_t.append(X45_te[:, name_to_idx[nm]])

    Xp4_cols_o, Xp4_cols_t = [], []
    for tag in p4_keep:
        po, pt = p4_raw[tag]
        Xp4_cols_o.append(po)
        Xp4_cols_t.append(pt)
        for so, st in zip(tab_sig_o, tab_sig_t):
            Xp4_cols_o.append(clip_z(po) * clip_z(so))
            Xp4_cols_t.append(clip_z(pt) * clip_z(st))
    if not Xp4_cols_o:
        print("  WARNING: no clean P4 — W4 skipped")
        W4a_o = W4a_t = W4b_o = W4b_t = None
    else:
        Xp4_o = np.column_stack(Xp4_cols_o).astype(np.float64)
        Xp4_t = np.column_stack(Xp4_cols_t).astype(np.float64)
        print(f"  X_p4_clean cols={Xp4_o.shape[1]}")

        # Ridge on g_AUC and g_BCE
        def fit_ridge_target(targets, tag):
            oo = np.zeros(n_tr)
            te = []
            for f, (tr, va) in enumerate(folds):
                lr = Ridge(alpha=1000.0, fit_intercept=True)
                lr.fit(Xp4_o[tr], targets[f][tr])
                oo[va] = lr.predict(Xp4_o[va])
                te.append(lr.predict(Xp4_t))
            print(f"  {tag} ridge t={time.time()-t0:.1f}s")
            return oo, np.mean(np.stack(te, 0), 0)

        def fit_cat_p4_fixed(targets, weights, tag):
            oof_acc, te_acc = [], []
            for seed in (11, 21, 31, 41, 51, 61):
                oo = np.zeros(n_tr)
                tef = []
                for f, (tr, va) in enumerate(folds):
                    m = CatBoostRegressor(
                        loss_function="RMSE",
                        depth=3,
                        l2_leaf_reg=50,
                        iterations=100,
                        learning_rate=0.03,
                        random_seed=seed,
                        verbose=False,
                        thread_count=4,
                        bootstrap_type="Bernoulli",
                        subsample=0.8,
                    )
                    if weights is None:
                        m.fit(Xp4_o[tr].astype(np.float32), targets[f][tr])
                    else:
                        m.fit(
                            Xp4_o[tr].astype(np.float32),
                            targets[f][tr],
                            sample_weight=weights[f][tr],
                        )
                    oo[va] = m.predict(Xp4_o[va].astype(np.float32))
                    tef.append(m.predict(Xp4_t.astype(np.float32)))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
            print(f"  {tag} cat n={len(oof_acc)} t={time.time()-t0:.1f}s")
            return np.mean(np.stack(oof_acc, 0), 0), np.mean(np.stack(te_acc, 0), 0)

        R_auc_o, R_auc_t = fit_ridge_target(g_auc_folds, "W4_ridge_auc")
        R_bce_o, R_bce_t = fit_ridge_target(g_bce_folds, "W4_ridge_bce")
        C_auc_o, C_auc_t = fit_cat_p4_fixed(g_auc_folds, None, "W4_cat_auc")
        C_bce_o, C_bce_t = fit_cat_p4_fixed(g_bce_folds, h_bce_folds, "W4_cat_bce")
        W4a_o = 0.5 * R_auc_o + 0.5 * C_auc_o
        W4a_t = 0.5 * R_auc_t + 0.5 * C_auc_t
        W4b_o = 0.5 * R_bce_o + 0.5 * C_bce_o
        W4b_t = 0.5 * R_bce_t + 0.5 * C_bce_t

    # ---------- Dual-basis orth ----------
    print("=== Step2 dual-basis orth ===", f"t={time.time()-t0:.1f}s")
    raw_fams = {
        "W1": (W1_o, W1_t),
        "W2": (W2_o, W2_t),
        "W3": (W3_o, W3_t),
        "W12": (0.5 * W1_o + 0.5 * W2_o, 0.5 * W1_t + 0.5 * W2_t),
    }
    if W4a_o is not None:
        raw_fams["W4a_auc"] = (W4a_o, W4a_t)
        raw_fams["W4b_bce"] = (W4b_o, W4b_t)
        raw_fams["W4"] = (0.5 * W4a_o + 0.5 * W4b_o, 0.5 * W4a_t + 0.5 * W4b_t)
        raw_fams["W12_W4"] = (
            0.4 * W1_o + 0.3 * W2_o + 0.3 * (0.5 * W4a_o + 0.5 * W4b_o),
            0.4 * W1_t + 0.3 * W2_t + 0.3 * (0.5 * W4a_t + 0.5 * W4b_t),
        )

    oriented = {}
    for name, (vo, vt) in raw_fams.items():
        oo12, ot12, oo9, ot9 = method_b_orth12(
            vo, vt, Bo9, Bt9, incr_o, incr_t, folds, ridge_alpha=10.0
        )
        g9 = forensic(oo9, ot9, Bt9)
        g12 = forensic(oo12, ot12, Bt12)
        # also F2 vs full Bo12 columns for gate
        g12b = forensic(oo12, ot12, np.column_stack([Bt12, ut_m5, ut_r1]))
        u9, ut9 = unitize(oo9), unitize(ot9)
        u12, ut12 = unitize(oo12), unitize(ot12)
        # orient vs Slot8
        if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u12[te0]) < roc_auc_score(
            y[te0], z_s8[te0] - 0.01 * u12[te0]
        ):
            u12, ut12 = -u12, -ut12
            u9, ut9 = -u9, -ut9
        print(
            f"  {name}: F2_Bo9={g9['F2']:.4f} F2_Bo12={g12['F2']:.4f} "
            f"disp9={g9['raw_disp']:.3f} disp12={g12['raw_disp']:.3f} "
            f"KS12={g12['raw_KS']:.3f}"
        )
        oriented[name] = {
            "u9": u9,
            "ut9": ut9,
            "u12": u12,
            "ut12": ut12,
            "g9": g9,
            "g12": g12,
            "g12b": g12b,
        }

    np.savez_compressed(
        OUT / "v20_0_families.npz",
        **{f"{k}_orth9_oof": v["u9"] for k, v in oriented.items()},
        **{f"{k}_orth9_test": v["ut9"] for k, v in oriented.items()},
        **{f"{k}_orth12_oof": v["u12"] for k, v in oriented.items()},
        **{f"{k}_orth12_test": v["ut12"] for k, v in oriented.items()},
    )

    # ---------- Step3 audit ----------
    print("=== Step3 H2H vs Slot8 ===", f"t={time.time()-t0:.1f}s")
    candidates, winners = [], []

    def pass_gates(per, rho, c6, c7, g9, g12, f3):
        return (
            all(
                per[b]["pos"] >= 5
                and per[b]["min"] > 0.0
                and per[b]["sf25"] >= 18
                and per[b]["t"] >= 2.8
                for b in per
            )
            and (0.30 <= rho <= 1.10)
            and abs(c6) < 0.20
            and abs(c7) < 0.20
            and 0.92 <= g12["raw_disp"] <= 1.08
            and g12["raw_KS"] < 0.025
            and g9["F2"] < 0.035
            and g12["F2"] < 0.038
            and f3 < 0.045
        )

    def eval_additive(name, u, ut, g9, g12):
        f3 = abs(float(np.corrcoef(unitize(ut), unitize(L(slot8) - L(slot5)))[0, 1]))
        print(
            f"  {name}: F2_9={g9['F2']:.4f} F2_12={g12['F2']:.4f} "
            f"disp={g12['raw_disp']:.3f} KS={g12['raw_KS']:.3f} F3={f3:.4f}"
        )
        for w in (0.004, 0.006, 0.008, 0.010, 0.012, 0.015, 0.018):
            oof = {b: bases[b] + w * u for b in bases}
            per = {b: h2h(oof[b], bases[b], y, folds) for b in bases}
            cand = qlock(L(slot8) + w * ut, lv)
            rho = one_m_rho_e4(cand, slot8)
            delta = unitize(L(cand) - L(slot8))
            c6 = float(np.corrcoef(delta, bad6)[0, 1])
            c7 = float(np.corrcoef(delta, bad7)[0, 1])
            ok = pass_gates(per, rho, c6, c7, g9, g12, f3)
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
                "geom": "add12",
                "w": w,
                "per": per,
                "rho": rho,
                "c6": c6,
                "c7": c7,
                "f2_bo9": g9["F2"],
                "f2_bo12": g12["F2"],
                "f3": f3,
                "gates": g12,
                "ok": ok,
                "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                "min_lift": float(np.min([per[b]["min"] for b in per])),
            }
            candidates.append(row)
            if ok:
                winners.append({**row, "cand": cand})

    for name, od in oriented.items():
        eval_additive(f"{name}_orth12", od["u12"], od["ut12"], od["g9"], od["g12"])

    # Geometry 2: Bo9 triad upgrade
    print("=== Geometry2 Bo9 triad vs Slot8 ===", f"t={time.time()-t0:.1f}s")
    # Use forensic of orth9 for gates (disp/KS from orth9; F2_Bo12 from method-B orth12 of same family)
    wr2s = (0.012, 0.014, 0.016)
    wcats = (0.008, 0.010, 0.012)
    wu5s = (0.010, 0.012, 0.014)
    wws = (0.004, 0.006, 0.008, 0.010)
    # Prioritize families with clean F2
    geom2_names = [
        n
        for n, od in oriented.items()
        if od["g9"]["F2"] < 0.040 and od["g12"]["F2"] < 0.050
    ]
    if not geom2_names:
        geom2_names = list(oriented.keys())[:4]
    print(f"  geom2 families: {geom2_names}")

    for name in geom2_names:
        od = oriented[name]
        u, ut = od["u9"], od["ut9"]
        g9, g12 = od["g9"], od["g12"]
        f3 = abs(float(np.corrcoef(unitize(ut), unitize(L(slot8) - L(slot5)))[0, 1]))
        # denser grid only if additive showed promise; else coarse mid grid
        best_add = max(
            (c for c in candidates if c["family"] == f"{name}_orth12"),
            key=lambda c: (c["per"]["z_S8"]["pos"], c["mean_lift"]),
            default=None,
        )
        if best_add and best_add["per"]["z_S8"]["pos"] >= 4:
            grid = product(wr2s, wcats, wu5s, wws)
        else:
            grid = product((0.014, 0.016), (0.010, 0.012), (0.012,), (0.006, 0.008, 0.010))
        for wr2, wcat, wu5, ww in grid:
            oof = {
                "z_S8": z_s3 + wr2 * u_r2 + wcat * u_cat + wu5 * u_u5 + ww * u,
                "twin": z_twin + wr2 * u_r2 + wcat * u_cat + wu5 * u_u5 + ww * u,
                "nnls": z_nnls + wr2 * u_r2 + wcat * u_cat + wu5 * u_u5 + ww * u,
            }
            per = {b: h2h(oof[b], bases[b], y, folds) for b in bases}
            cand = qlock(
                L(slot3) + wr2 * ut_r2 + wcat * ut_cat + wu5 * ut_u5 + ww * ut, lv
            )
            rho = one_m_rho_e4(cand, slot8)
            delta = unitize(L(cand) - L(slot8))
            c6 = float(np.corrcoef(delta, bad6)[0, 1])
            c7 = float(np.corrcoef(delta, bad7)[0, 1])
            # For triad geom, F2 gates use orth9 vs Bo9 and orth12 F2
            ok = pass_gates(per, rho, c6, c7, g9, g12, f3)
            jn = f"G2_{name}_r{wr2:.3f}_c{wcat:.3f}_u{wu5:.3f}_w{ww:.3f}"
            mark = "★" if ok else ("~" if per["z_S8"]["pos"] >= 4 else " ")
            if ok or per["z_S8"]["pos"] >= 4:
                print(
                    f"{mark} {jn}: 1-ρ={rho:.3f} c7={c7:+.3f} ok={ok} | "
                    + " | ".join(
                        f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f} "
                        f"t={per[b]['t']:.2f}"
                        for b in bases
                    )
                )
            row = {
                "name": jn,
                "family": name,
                "geom": "triad9",
                "w": ww,
                "per": per,
                "rho": rho,
                "c6": c6,
                "c7": c7,
                "f2_bo9": g9["F2"],
                "f2_bo12": g12["F2"],
                "f3": f3,
                "gates": g12,
                "ok": ok,
                "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                "min_lift": float(np.min([per[b]["min"] for b in per])),
                "meta": {"wr2": wr2, "wcat": wcat, "wu5": wu5, "ww": ww},
            }
            candidates.append(row)
            if ok:
                winners.append({**row, "cand": cand})

    print(f"\nWinners: {len(winners)}")
    submit, pub, ref, chosen = False, None, None, None
    if winners:
        winners.sort(
            key=lambda r: (r["min_lift"], r["mean_lift"], -abs(r["rho"] - 0.50)), reverse=True
        )
        chosen = winners[0]
        print(
            f"CHOSEN {chosen['name']} mean={chosen['mean_lift']:+.2f} "
            f"min={chosen['min_lift']:+.2f} 1-ρ={chosen['rho']:.3f} c7={chosen['c7']:+.3f}"
        )
        out_csv = SUB / "sub_v20_0_LIVE_slot9_wilcoxon_auc_super.csv"
        arr = chosen["cand"]
        assert len(arr) == n_te and np.isfinite(arr).all()
        pd.DataFrame({"id": ids, "is_fraud": arr}).to_csv(out_csv, index=False)
        msg = (
            f"v20.0 Slot9: {chosen['name']} Wilcoxon-AUC vs Slot8 0.95395 | "
            f"H2H mean={chosen['mean_lift']:+.2f}e-5 min={chosen['min_lift']:+.2f} "
            f"1-ρ={chosen['rho']:.3f}e-4 c7={chosen['c7']:+.3f} F2={chosen['f2_bo12']:.3f}"
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
                    if "v20_0" in row.get("fileName", ""):
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
        candidates,
        key=lambda c: (
            c["per"]["z_S8"]["pos"],
            c["per"]["twin"]["pos"],
            c["mean_lift"],
            -abs(c.get("c7", 1)),
        ),
        reverse=True,
    )[:25]
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
                "f2_bo9": chosen["f2_bo9"],
                "f2_bo12": chosen["f2_bo12"],
                "per": chosen["per"],
            },
            "public_lb": pub,
            "ref": ref,
            "twilight": twilight_share,
            "p4_kept": p4_keep,
            "orth_summary": {
                k: {"F2_Bo9": v["g9"]["F2"], "F2_Bo12": v["g12"]["F2"], "KS": v["g12"]["raw_KS"]}
                for k, v in oriented.items()
            },
            "near_misses": near,
            "elapsed_s": time.time() - t0,
        }
    )
    OUT.joinpath("v20_0.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# GM V20.0 — Wilcoxon-AUC + Method-B Bo12 + clean P4 → "
        + ("**LIVE Slot9**" if submit else "**HOLD**"),
        "",
        f"**0 GPU pods · elapsed {payload['elapsed_s']:.0f}s**",
        "",
        "| | |",
        "|---|---|",
        "| Slot8 ATH | 0.95395 |",
        f"| Slot8 rebuild 1-ρ | {rho_rebuild:.3e}e-4 |",
        f"| Submit | {submit} |",
        f"| Public LB | {pub} |",
        f"| Winners | {len(winners)} |",
        f"| P4 kept | {p4_keep} |",
        "",
        "## Dual-basis orth summary",
    ]
    for k, v in oriented.items():
        lines.append(
            f"- `{k}`: F2_Bo9={v['g9']['F2']:.4f} F2_Bo12={v['g12']['F2']:.4f} "
            f"disp={v['g12']['raw_disp']:.3f} KS={v['g12']['raw_KS']:.3f}"
        )
    lines += ["", "## Twilight energy (g_AUC vs g_BCE)"]
    for trow in twilight_share:
        lines.append(
            f"- fold{trow['fold']}: corr={trow['corr_auc_bce']:+.3f} "
            f"E_tw auc={trow['e_auc_tw']:.3f} bce={trow['e_bce_tw']:.3f}"
        )
    lines += ["", "## Top near-misses"]
    for c in near[:15]:
        lines.append(
            f"- `{c['name']}`: 1-ρ={c['rho']:.3f} c7={c['c7']:+.3f} "
            f"F2_12={c['f2_bo12']:.3f} z_S8 {c['per']['z_S8']['pos']}/5 "
            f"m={c['per']['z_S8']['mean']:+.2f} twin {c['per']['twin']['pos']}/5 "
            f"m={c['per']['twin']['mean']:+.2f} ok={c['ok']}"
        )
    if chosen:
        lines += ["", f"## Chosen `{chosen['name']}`", f"- mean={chosen['mean_lift']:+.2f}"]
    else:
        lines += ["", "## Decision", "HOLD ATH `0.95395`. Keep remaining daily slots."]
    report = LOG / "GM_V20_0_WILCOXON_AUC_SUPER_REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print("WROTE", report)
    print("WROTE", OUT / "v20_0.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
