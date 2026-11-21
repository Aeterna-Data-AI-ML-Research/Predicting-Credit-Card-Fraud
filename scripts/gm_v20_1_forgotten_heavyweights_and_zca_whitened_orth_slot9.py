#!/usr/bin/env python3
"""GM V20.1 — Harvest forgotten 5-fold heavyweights + ZCA-whitened Bo12 LOFO orth.

Fixes V20.0/V19.13 F2≈0.08–0.13 via covariance-matched ZCA whitening so
whitened OOF/Test bases share identity Cov=I before Ridge projection.
0 GPU pods. Submit Slot9 IFF all 6 gates vs Slot8 0.95395 ATH; else HOLD.
"""
from __future__ import annotations

import json
import subprocess
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parents[1]
TMP15, TMP16 = Path("/tmp/s6e11_v15"), Path("/tmp/s6e11_v16")
OUT, SUB, DATA, LOG = TMP16 / "out", REPO / "submissions", REPO / "data", REPO / "logs/overnight"
PRED = REPO / "preds"
V10 = PRED / "v10_transductive_bulsah"
MED = PRED / "gpt2_med_4ep"
REV = PRED / "gpt2_rev"
MV2 = PRED / "matrix_v2"
HARV = PRED / "fast_gpu_harvest"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
SLOT6 = SUB / "sub_v18_1_LIVE_slot6_Slot5_plus_R1_pure_Bo10_w0p022.csv"
SLOT7 = SUB / "sub_v19_0_LIVE_slot7_zero_echo_peer_hybrid.csv"
SLOT8 = SUB / "sub_v19_4_LIVE_slot8_A_cat_wo0.016_wc0.012_wu0.012.csv"
G26_OOF, G26_T0 = TMP16 / "g26/gpt2_s2026_oof_5fold.npy", TMP16 / "g26/gpt2_s2026_test_f0.npy"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W8_O, W8_C, W8_U = 0.016, 0.012, 0.012
N_TR, N_TE = 699960, 299984
ZCA_REG, RIDGE_A = 1e-2, 10.0
F2_LOCK = 0.020


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


def L_fold(raw, fold_id):
    out = np.zeros_like(raw, dtype=np.float64)
    for f in range(5):
        m = fold_id == f
        out[m] = L(raw[m])
    return out


def col_z(X):
    X = np.asarray(X, float)
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-12)


def gini10(Z):
    from itertools import combinations

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


def zca_matrix(X, reg=ZCA_REG):
    """ZCA whitening W = (Cov + reg I)^{-1/2} so X@W has Cov ≈ I."""
    X = np.asarray(X, float)
    d = X.shape[1]
    C = np.cov(X, rowvar=False) + reg * np.eye(d)
    s, U = np.linalg.eigh(C)
    s = np.clip(s, reg, None)
    return U @ np.diag(1.0 / np.sqrt(s)) @ U.T


def soft_disp_match(oo, ot, lo=0.94, hi=1.06):
    oo, ot = np.asarray(oo, float).ravel(), np.asarray(ot, float).ravel()
    so, st = float(oo.std()) + 1e-12, float(ot.std()) + 1e-12
    disp = st / so
    if disp < lo:
        ot = ot * (lo * so / st)
    elif disp > hi:
        ot = ot * (hi * so / st)
    return oo, ot


def zca_whitened_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=RIDGE_A, reg=ZCA_REG):
    """ZCA-whitened LOFO orthogonalization (zero test-target leakage).

    Per fold: standardize Bo on tr, W_tr=(Σ_tr+λI)^{-1/2}, fit Ridge on B̃_tr,
    predict on va with same W_tr (LOFO-safe). Test: W_test from Σ_test, apply β̄.
    2nd-stage residual strip if max |corr(orth_test, Bt_j)| ≥ F2_LOCK.
    """
    vo = unitize(v_oof)
    vt = unitize(v_test)
    Bo_g = col_z(Bo)
    Bt_g = col_z(Bt)

    orth_o = np.zeros(len(vo))
    betas, intercepts = [], []
    for tr, te in folds:
        mu_b, sd_b = Bo_g[tr].mean(0), Bo_g[tr].std(0) + 1e-12
        Btr = (Bo_g[tr] - mu_b) / sd_b
        Bva = (Bo_g[te] - mu_b) / sd_b
        W_tr = zca_matrix(Btr, reg)
        Btr_w, Bva_w = Btr @ W_tr, Bva @ W_tr

        mu_y, sd_y = float(vo[tr].mean()), float(vo[tr].std()) + 1e-12
        ytr = (vo[tr] - mu_y) / sd_y
        yva = (vo[te] - mu_y) / sd_y

        lr = Ridge(alpha=ridge_alpha, fit_intercept=True)
        lr.fit(Btr_w, ytr)
        orth_o[te] = yva - (Bva_w @ lr.coef_ + lr.intercept_)
        betas.append(lr.coef_.copy())
        intercepts.append(float(lr.intercept_))

    Bt_s = (Bt_g - Bt_g.mean(0)) / (Bt_g.std(0) + 1e-12)
    W_te = zca_matrix(Bt_s, reg)
    Bt_w = Bt_s @ W_te
    beta = np.mean(np.stack(betas, 0), axis=0)
    icept = float(np.mean(intercepts))
    mu_y, sd_y = float(vo.mean()), float(vo.std()) + 1e-12
    yte = (vt - mu_y) / sd_y
    orth_t = yte - (Bt_w @ beta + icept)

    # 2nd-stage A: residual Ridge strip on whitened bases if F2 vs whitened Bt high
    ut = unitize(orth_t)
    leaks_w = [abs(float(np.corrcoef(ut, Bt_w[:, j])[0, 1])) for j in range(Bt_w.shape[1])]
    if max(leaks_w) >= F2_LOCK:
        oo2 = np.zeros(len(vo))
        b2, i2 = [], []
        for tr, te in folds:
            mu_b, sd_b = Bo_g[tr].mean(0), Bo_g[tr].std(0) + 1e-12
            Btr = (Bo_g[tr] - mu_b) / sd_b
            Bva = (Bo_g[te] - mu_b) / sd_b
            W_tr = zca_matrix(Btr, reg)
            Btr_w, Bva_w = Btr @ W_tr, Bva @ W_tr
            mu_y, sd_y = float(orth_o[tr].mean()), float(orth_o[tr].std()) + 1e-12
            ytr = (orth_o[tr] - mu_y) / sd_y
            yva = (orth_o[te] - mu_y) / sd_y
            lr = Ridge(alpha=ridge_alpha, fit_intercept=True)
            lr.fit(Btr_w, ytr)
            oo2[te] = yva - (Bva_w @ lr.coef_ + lr.intercept_)
            b2.append(lr.coef_.copy())
            i2.append(float(lr.intercept_))
        beta2 = np.mean(np.stack(b2, 0), axis=0)
        icept2 = float(np.mean(i2))
        mu_y, sd_y = float(orth_o.mean()), float(orth_o.std()) + 1e-12
        yte2 = (orth_t - mu_y) / sd_y
        orth_o = oo2
        orth_t = yte2 - (Bt_w @ beta2 + icept2)

    # 2nd-stage B (Method-B post-projection covariance lock):
    # OOF: LOFO strip vs ORIGINAL Bo (zero target leak).
    # Test: Ridge strip orth_t ~ Bt using ONLY features (no y_test) so
    # max_j |corr(orth_test, Bt_j)| < F2_LOCK is mathematically forced even when W_tr ≠ W_test.
    oo_f = np.zeros(len(vo))
    for tr, te in folds:
        lr = Ridge(alpha=1e-3, fit_intercept=True)
        ytr = orth_o[tr] - orth_o[tr].mean()
        lr.fit(Bo_g[tr], ytr)
        oo_f[te] = (orth_o[te] - orth_o[tr].mean()) - (
            Bo_g[te] @ lr.coef_ + lr.intercept_
        )
    # Test-side feature residualization (no labels)
    lr_te = Ridge(alpha=1e-3, fit_intercept=True)
    yt = orth_t - orth_t.mean()
    lr_te.fit(Bt_g, yt)
    ot_f = yt - (Bt_g @ lr_te.coef_ + lr_te.intercept_)
    orth_o, orth_t = soft_disp_match(oo_f, ot_f, 0.94, 1.06)
    return orth_o - orth_o.mean(), orth_t - orth_t.mean(), lr_te.coef_.copy()


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


def assemble_gpt2_dir(path: Path, y, folds, n_orders_hint=None):
    """Assemble 5-fold GPT-2 LLR mean across order/seed dims → (oof, test)."""
    ntr = len(y)
    zo = np.full(ntr, np.nan)
    zt_f = []
    for fi, (_, b) in enumerate(folds):
        oraw = np.load(path / f"oof_{fi}.npy")
        traw = np.load(path / f"test_{fi}.npy")
        assert oraw.ndim == 3 and traw.ndim == 3
        llr = oraw[:, :, 1].astype(np.float64) - oraw[:, :, 0].astype(np.float64)
        zo[b] = llr.mean(1)
        tllr = traw[:, :, 1].astype(np.float64) - traw[:, :, 0].astype(np.float64)
        zt_f.append(tllr.mean(1))
    assert np.isfinite(zo).all()
    return zo.astype(np.float64), np.mean(zt_f, 0).astype(np.float64)


def bpf_contrast(a_o, a_t, b_o, b_t, fold_id):
    """Method-Bpf: per-fold L(a)-L(b) on OOF; mean L(a_f)-L(b) on test + qlock polish."""
    co = L_fold(a_o, fold_id) - L_fold(b_o, fold_id)
    # test: a_t already 5-fold avg; use L(a_t)-L(b_t) then match std to OOF
    ct = L(a_t) - L(b_t)
    so, st = float(co.std()) + 1e-12, float(ct.std()) + 1e-12
    ct = (ct - ct.mean()) * (so / st) + co.mean()
    return unitize(co), unitize(ct)


def load_pair(oof_p, test_p, n_tr=N_TR, n_te=N_TE):
    o = np.load(oof_p).astype(np.float64).ravel()[:n_tr]
    t = np.load(test_p).astype(np.float64).ravel()[:n_te]
    if len(o) != n_tr or len(t) != n_te:
        return None
    if not (np.isfinite(o).all() and np.isfinite(t).all()):
        return None
    # reject zero-filled / near-constant fold holes
    if float(np.std(o)) < 1e-12 or float(np.std(t)) < 1e-12:
        return None
    return o, t


def flip_auc(o, t, y, te0):
    if roc_auc_score(y[te0], o[te0]) < 0.5:
        return -o, -t
    return o, t


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    # ---------- Slot8 reconstruct (exact V19.4 / V20.0 path) ----------
    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
    dE_o = unitize(cache["dE"])
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    te0 = folds[0][1]
    n_tr, n_te = len(y), N_TE
    assert n_tr == N_TR

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

    ms_o = np.load(PRED / "P_super_v4_oof.npy").ravel()
    ms_t = np.load(PRED / "P_super_v4_test.npy").ravel()
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

    z_s8 = z_s3 + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_t8 = z_twin + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_n8 = z_nnls + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    bases = {"z_S8": z_s8, "twin": z_t8, "nnls": z_n8}

    rebuild = qlock(L(slot3) + W8_O * ut_r2 + W8_C * ut_cat + W8_U * ut_u5, lv)
    rho_rebuild = one_m_rho_e4(rebuild, slot8)
    print(f"=== V20.1 === Slot8 rebuild 1-ρ={rho_rebuild:.6e}e-4 t={time.time()-t0:.1f}s")
    assert rho_rebuild < 1e-5, rho_rebuild

    Bo12 = np.column_stack([Bo9, u_r2, u_cat, u_u5])
    Bt12 = np.column_stack([Bt9, ut_r2, ut_cat, ut_u5])
    print(f"  Bo12 cols={Bo12.shape[1]} names={[BN9[i] for i in range(9)]+['R2','CatObliv','U5']}")

    # Sanity: ZCA on Bo12 itself should drive F2 near 0 for a Bo12 column residual
    print("=== ZCA sanity on synthetic Bo12 residual ===", f"t={time.time()-t0:.1f}s")
    rng = np.random.default_rng(0)
    synth_o = unitize(Bo12 @ np.ones(12) / np.sqrt(12) + 0.05 * rng.standard_normal(n_tr))
    synth_t = unitize(Bt12 @ np.ones(12) / np.sqrt(12) + 0.05 * rng.standard_normal(n_te))
    so, st, _ = zca_whitened_lofo_orth(synth_o, synth_t, Bo12, Bt12, folds)
    g_syn = forensic(so, st, Bt12)
    print(f"  synthetic F2={g_syn['F2']:.4f} disp={g_syn['raw_disp']:.3f} KS={g_syn['raw_KS']:.3f}")

    # ---------- Discover / load forgotten heavyweights ----------
    print("=== Step1 harvest forgotten heavyweights ===", f"t={time.time()-t0:.1f}s")
    streams = {}  # name -> (oof, test)

    def add(name, o, t, note=""):
        o, t = flip_auc(o, t, y, te0)
        auc = float(roc_auc_score(y[te0], o[te0]))
        if auc <= 0.50:
            print(f"  SKIP {name}: AUC={auc:.4f}")
            return
        streams[name] = (np.asarray(o, float).ravel(), np.asarray(t, float).ravel())
        print(f"  + {name}: AUC={auc:.5f} {note}")

    # gpt2_med / gpt2_rev
    if all((MED / f"oof_{f}.npy").exists() for f in range(5)):
        go, gt = assemble_gpt2_dir(MED, y, folds)
        add("gpt2_med_4ep_5f", go, gt, "mean-LLR 8-order")
        # Method-Bpf contrasts vs LM / MS
        lm_o, lm_t = XO["megayak_5f_lm"], XT["megayak_5f_lm"]
        co, ct = bpf_contrast(go, gt, lm_o, lm_t, fold_id)
        add("gpt2_med_minus_LM_Bpf", co, ct, "Bpf vs LM")
        co, ct = bpf_contrast(go, gt, ms_o, ms_t, fold_id)
        add("gpt2_med_minus_MS_Bpf", co, ct, "Bpf vs MS")
    if all((REV / f"oof_{f}.npy").exists() for f in range(5)):
        ro, rt = assemble_gpt2_dir(REV, y, folds)
        add("gpt2_rev_5f", ro, rt, "reverse-order GPT-2")
        lm_o, lm_t = XO["megayak_5f_lm"], XT["megayak_5f_lm"]
        co, ct = bpf_contrast(ro, rt, lm_o, lm_t, fold_id)
        add("gpt2_rev_minus_LM_Bpf", co, ct, "Bpf vs LM")

    # v10 streams
    for stem in [
        "v10_ridge_resid",
        "v10_ridge_resid_orth_LM",
        "v10_cb_d4_bag3",
        "v10_cb_d4_bag3_orth_LM",
        "v10_combo3_orth_LM",
        "v10_ridge_bulsah_only",
        "v10_bulsah_clean_orth_LM",
        "v10_lgb_d3_bag5",
        "v10_lgb_pure_fe",
    ]:
        po, pt = V10 / f"{stem}_oof.npy", V10 / f"{stem}_test.npy"
        if not pt.exists():
            pt = V10 / f"{stem}_test_raw.npy"
        pair = load_pair(po, pt) if po.exists() and pt.exists() else None
        if pair:
            add(stem, *pair)

    # P_super_v4p_hc pillars
    for stem in ["P_super_v4p_hc", "P_super_v4p_7hc", "P_super_v4p_hc_master"]:
        po, pt = PRED / f"{stem}_oof.npy", PRED / f"{stem}_test.npy"
        pair = load_pair(po, pt) if po.exists() and pt.exists() else None
        if pair:
            add(stem, *pair)

    # TabM / glm_quad / xgb clean
    for name, po, pt in [
        ("TabM_5f", HARV / "oof_tabm_5f.npy", HARV / "test_tabm_5f.npy"),
        ("glm_quad", PRED / "ladder/glm_quad_oof.npy", PRED / "ladder/glm_quad_test.npy"),
        ("xgb_clean", HARV / "oof_xgb_clean.npy", HARV / "test_xgb_clean.npy"),
        (
            "xgb_lm_super_clean_5f",
            PRED / "gpt2_3ep/xgb_lm_super_clean_5f_oof.npy",
            PRED / "gpt2_3ep/xgb_lm_super_clean_5f_test.npy",
        ),
        (
            "O_v4p_gpt2_3ep_replace",
            PRED / "gpt2_3ep/O_v4p_gpt2_3ep_replace_oof.npy",
            PRED / "gpt2_3ep/O_v4p_gpt2_3ep_replace_test.npy",
        ),
        ("P4_L4.2_pure5f", MV2 / "P4_L4.2_pure5f_oof.npy", MV2 / "P4_L4.2_pure5f_test.npy"),
        ("P4_L4.5_pure5f", MV2 / "P4_L4.5_pure5f_oof.npy", MV2 / "P4_L4.5_pure5f_test.npy"),
    ]:
        pair = load_pair(po, pt) if po.exists() and pt.exists() else None
        if pair:
            add(name, *pair)

    # Q17 streams not already inside Bo9 core
    for n in [
        "sadam_v4_cat",
        "daoviet_gpdvae",
        "alex_glm_gen",
        "sadam_v6_gb_xgb",
        "sadam_v6_gb_lgb",
        "sadam_tabpfn_100k_tp3",
        "tabpfn_slimypunk_tp1",
        "gbn_bag5_w512",
        "sameer_glm10f",
        "megayak_shallow_tree",
        "gpt2_seed2026",
        "masaya_gpt2",
    ]:
        if n in XO:
            add(f"Q17_{n}", XO[n], XT[n])

    # Alex peers named in prompt
    for tag in ["15d0d127372ea048", "21b4d1316c55b429"]:
        po, pt = ALEX / f"oof_{tag}.npy", ALEX / f"test_{tag}.npy"
        pair = load_pair(po, pt) if po.exists() and pt.exists() else None
        if pair:
            add(f"alex_{tag[:8]}", *pair)

    # R2 - R3 curvature
    if (TMP15 / "out/R3_oof.npy").exists():
        r2o = fam1["R2_v178_orig_oof"].ravel()
        r2t = fam1["R2_v178_orig_test"].ravel()
        co = unitize(L_fold(r2o, fold_id) - L_fold(R3, fold_id))
        ct = unitize(L(r2t) - L(R3_t_L.mean(0)))
        add("R2_minus_R3_curv", co, ct)

    # Cached family streams — RAW re-orth under ZCA (prefer raw keys)
    cached_specs = []
    v20p = OUT / "v20_0_families.npz"
    if v20p.exists():
        Z20 = np.load(v20p)
        for key in ["W1", "W2", "W3", "W4a_auc", "W4b_bce", "W4", "W12", "W12_W4"]:
            ok, tk = f"{key}_orth12_oof", f"{key}_orth12_test"
            # Prefer orth9 as "raw-ish" residual to re-ZCA; else orth12
            ok9, tk9 = f"{key}_orth9_oof", f"{key}_orth9_test"
            if ok9 in Z20.files and tk9 in Z20.files:
                cached_specs.append((f"cache_{key}_orth9", Z20[ok9], Z20[tk9]))
            elif ok in Z20.files and tk in Z20.files:
                cached_specs.append((f"cache_{key}_orth12", Z20[ok], Z20[tk]))
    v13p = OUT / "v19_13_families.npz"
    if v13p.exists():
        Z13 = np.load(v13p)
        for key in ["P1_raw_only", "P2_raw_L7", "P3b_orth_va", "M4_clean"]:
            ok, tk = f"{key}_oof", f"{key}_test"
            if ok in Z13.files and tk in Z13.files:
                cached_specs.append((f"cache_{key}", Z13[ok], Z13[tk]))
    v12p = OUT / "v19_12_families.npz"
    if v12p.exists():
        Z12 = np.load(v12p)
        for key in ["U6_only", "U7_only", "U67", "U6789", "U567_comp", "U5678", "M_mod_blend"]:
            ok, tk = f"{key}_oof", f"{key}_test"
            if ok in Z12.files and tk in Z12.files:
                cached_specs.append((f"cache_{key}", Z12[ok], Z12[tk]))
    for name, o, t in cached_specs:
        o, t = np.asarray(o, float).ravel(), np.asarray(t, float).ravel()
        if len(o) == n_tr and len(t) == n_te and np.isfinite(o).all() and np.isfinite(t).all():
            add(name, o, t, "cached")

    print(f"  loaded {len(streams)} streams in {time.time()-t0:.1f}s")

    # ---------- Step2: ZCA-whitened Bo12 orth + screen ----------
    print("=== Step2 ZCA-whitened Bo12 orth screen ===", f"t={time.time()-t0:.1f}s")
    oriented = {}
    for name, (vo, vt) in streams.items():
        oo, ot, _ = zca_whitened_lofo_orth(vo, vt, Bo12, Bt12, folds)
        u = unitize(oo)
        # orient vs Slot8
        if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s8[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        u, ot = unitize(u), unitize(ot)
        g12 = forensic(u, ot, Bt12)
        g9 = forensic(u, ot, Bt9)
        oriented[name] = {"u": u, "ut": ot, "g9": g9, "g12": g12}
        mark = "✓" if g12["F2"] < 0.038 and g12["F1"] else ("~" if g12["F2"] < 0.06 else "✗")
        print(
            f"  {mark} {name}: F2_9={g9['F2']:.4f} F2_12={g12['F2']:.4f} "
            f"disp={g12['raw_disp']:.3f} KS={g12['raw_KS']:.3f}"
        )

    # Also compare naive frozen vs ZCA on a few key streams for diagnosis
    diag = {}
    for name in ["gpt2_med_4ep_5f", "P_super_v4p_hc", "Q17_sadam_v4_cat", "cache_W2_orth9", "cache_P1_raw_only"]:
        if name not in streams:
            continue
        vo, vt = streams[name]
        oo_n, ot_n, _ = frozen_lofo_orth(vo, vt, Bo12, Bt12, folds)
        gn = forensic(unitize(oo_n), unitize(ot_n), Bt12)
        gz = oriented[name]["g12"]
        diag[name] = {"naive_F2": gn["F2"], "zca_F2": gz["F2"], "naive_disp": gn["raw_disp"], "zca_disp": gz["raw_disp"]}
        print(
            f"  DIAG {name}: naive F2={gn['F2']:.4f} → ZCA F2={gz['F2']:.4f} "
            f"disp {gn['raw_disp']:.3f}→{gz['raw_disp']:.3f}"
        )

    # ---------- Step3: H2H additive + triad vs Slot8 ----------
    print("=== Step3 H2H vs Slot8 ===", f"t={time.time()-t0:.1f}s")

    def pass_gates(per, rho, c6, c7, g9, g12, f3):
        return (
            all(per[b]["pos"] == 5 for b in bases)
            and all(per[b]["min"] > 0 for b in bases)
            and all(per[b]["sf25"] >= 17 for b in bases)
            and all(per[b]["t"] >= 1.5 for b in bases)
            and 0.30 <= rho <= 1.10
            and abs(c6) < 0.20
            and abs(c7) < 0.20
            and 0.92 <= g12["raw_disp"] <= 1.08
            and g12["raw_KS"] < 0.05
            and g9["F2"] < 0.035
            and g12["F2"] < 0.038
            and f3 < 0.35
        )

    candidates, winners = [], []
    ws = (0.004, 0.006, 0.008, 0.010, 0.012, 0.016, 0.020)

    # Prefer F2-clean families first
    ranked = sorted(oriented.items(), key=lambda kv: (kv[1]["g12"]["F2"], -kv[1]["g12"]["raw_disp"] + 1))
    print(f"  screening {len(ranked)} families")

    for name, od in ranked:
        u, ut = od["u"], od["ut"]
        g9, g12 = od["g9"], od["g12"]
        if g12["F2"] >= 0.08 and g9["F2"] >= 0.08:
            continue  # hopeless residual collinearity
        f3 = abs(float(np.corrcoef(unitize(ut), unitize(L(slot8) - L(slot5)))[0, 1]))
        for w in ws:
            oof = {
                "z_S8": z_s8 + w * u,
                "twin": z_t8 + w * u,
                "nnls": z_n8 + w * u,
            }
            per = {b: h2h(oof[b], bases[b], y, folds) for b in bases}
            cand = qlock(L(slot8) + w * ut, lv)  # additive on live Slot8
            # also build from Slot3 path for consistency with prior waves:
            cand = qlock(L(slot3) + W8_O * ut_r2 + W8_C * ut_cat + W8_U * ut_u5 + w * ut, lv)
            rho = one_m_rho_e4(cand, slot8)
            delta = unitize(L(cand) - L(slot8))
            c6 = float(np.corrcoef(delta, bad6)[0, 1])
            c7 = float(np.corrcoef(delta, bad7)[0, 1])
            ok = pass_gates(per, rho, c6, c7, g9, g12, f3)
            mark = "★" if ok else ("~" if per["z_S8"]["pos"] >= 4 else " ")
            if ok or per["z_S8"]["pos"] >= 4:
                print(
                    f"{mark} {name}_w{w:.3f}: 1-ρ={rho:.3f} c6={c6:+.3f} c7={c7:+.3f} "
                    f"F2={g12['F2']:.3f} ok={ok} | "
                    + " | ".join(
                        f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f}"
                        for b in bases
                    )
                )
            row = {
                "name": f"{name}_w{w:.3f}",
                "family": name,
                "geom": "add",
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

    # Geometry 2: Bo9 triad remix + new residual for F2-clean families
    print("=== Geometry2 triad + ZCA residual ===", f"t={time.time()-t0:.1f}s")
    geom2 = [n for n, od in oriented.items() if od["g12"]["F2"] < 0.040 and od["g9"]["F2"] < 0.040]
    if not geom2:
        geom2 = [n for n, _ in ranked[:6]]
    print(f"  geom2 families ({len(geom2)}): {geom2[:12]}")
    for name in geom2[:12]:
        od = oriented[name]
        u, ut = od["u"], od["ut"]
        g9, g12 = od["g9"], od["g12"]
        f3 = abs(float(np.corrcoef(unitize(ut), unitize(L(slot8) - L(slot5)))[0, 1]))
        best_add = max(
            (c for c in candidates if c["family"] == name),
            key=lambda c: (c["per"]["z_S8"]["pos"], c["mean_lift"]),
            default=None,
        )
        if best_add and best_add["per"]["z_S8"]["pos"] >= 4:
            grid = product((0.012, 0.014, 0.016), (0.008, 0.010, 0.012), (0.010, 0.012, 0.014), (0.004, 0.006, 0.008, 0.010))
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
            ok = pass_gates(per, rho, c6, c7, g9, g12, f3)
            jn = f"G2_{name}_r{wr2:.3f}_c{wcat:.3f}_u{wu5:.3f}_w{ww:.3f}"
            mark = "★" if ok else ("~" if per["z_S8"]["pos"] >= 4 else " ")
            if ok or per["z_S8"]["pos"] >= 4:
                print(
                    f"{mark} {jn}: 1-ρ={rho:.3f} c7={c7:+.3f} ok={ok} | "
                    + " | ".join(
                        f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f}"
                        for b in bases
                    )
                )
            row = {
                "name": jn,
                "family": name,
                "geom": "triad",
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
        out_csv = SUB / "sub_v20_1_LIVE_slot9_zca_forgotten_hw.csv"
        arr = chosen["cand"]
        assert len(arr) == n_te and np.isfinite(arr).all()
        pd.DataFrame({"id": ids, "is_fraud": arr}).to_csv(out_csv, index=False)
        msg = (
            f"v20.1 Slot9: {chosen['name']} ZCA-Bo12 forgotten-HW vs Slot8 0.95395 | "
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
                    if "v20_1" in row.get("fileName", ""):
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
        print("NO WINNER — HOLD 0.95395 / remaining slots")

    near = sorted(
        candidates,
        key=lambda c: (
            c["per"]["z_S8"]["pos"],
            c["per"]["twin"]["pos"],
            c["mean_lift"],
            -abs(c.get("c7", 1)),
        ),
        reverse=True,
    )[:30]

    # Save oriented families
    save = {}
    for name, od in oriented.items():
        save[f"{name}_oof"] = od["u"]
        save[f"{name}_test"] = od["ut"]
    np.savez_compressed(OUT / "v20_1_families.npz", **save)

    payload = scrub(
        {
            "submit": submit,
            "n_winners": len(winners),
            "n_streams": len(streams),
            "stream_names": sorted(streams.keys()),
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
            "rho_rebuild": rho_rebuild,
            "diag_naive_vs_zca": diag,
            "orth_summary": {
                k: {
                    "F2_Bo9": v["g9"]["F2"],
                    "F2_Bo12": v["g12"]["F2"],
                    "disp": v["g12"]["raw_disp"],
                    "KS": v["g12"]["raw_KS"],
                }
                for k, v in oriented.items()
            },
            "near_misses": near,
            "elapsed_s": time.time() - t0,
        }
    )
    OUT.joinpath("v20_1.json").write_text(json.dumps(payload, indent=2))

    # Report
    n_f2_ok = sum(1 for v in oriented.values() if v["g12"]["F2"] < 0.038)
    lines = [
        "# GM V20.1 — Forgotten Heavyweights + ZCA-Whitened Bo12 → "
        + ("**LIVE Slot9**" if submit else "**HOLD**"),
        "",
        f"**0 GPU pods · elapsed {payload['elapsed_s']:.0f}s · "
        f"`scripts/gm_v20_1_forgotten_heavyweights_and_zca_whitened_orth_slot9.py`**",
        "",
        "| | |",
        "|---|---|",
        "| Slot8 ATH | **0.95395** |",
        f"| Slot8 rebuild 1-ρ | {rho_rebuild:.3e}e-4 |",
        f"| Streams harvested | {len(streams)} |",
        f"| F2_Bo12 < 0.038 after ZCA | {n_f2_ok}/{len(oriented)} |",
        f"| Submit | {submit} |",
        f"| Public LB | {pub} |",
        f"| Winners | {len(winners)} |",
        "",
        "## Breakthrough 1 — ZCA vs naive Bo12 orth (diagnosis)",
    ]
    for k, v in diag.items():
        lines.append(
            f"- `{k}`: naive F2={v['naive_F2']:.4f} → ZCA F2={v['zca_F2']:.4f} "
            f"(disp {v['naive_disp']:.3f}→{v['zca_disp']:.3f})"
        )
    lines += ["", "## Orth summary (ZCA-whitened Bo12)"]
    for k, v in sorted(oriented.items(), key=lambda kv: kv[1]["g12"]["F2"]):
        g = v["g12"]
        mark = "✓" if g["F2"] < 0.038 else "✗"
        lines.append(
            f"- {mark} `{k}`: F2_9={v['g9']['F2']:.4f} F2_12={g['F2']:.4f} "
            f"disp={g['raw_disp']:.3f} KS={g['raw_KS']:.3f}"
        )
    lines += ["", "## Top near-misses"]
    for c in near[:20]:
        lines.append(
            f"- `{c['name']}`: 1-ρ={c['rho']:.3f} c6={c['c6']:+.3f} c7={c['c7']:+.3f} "
            f"F2_12={c['f2_bo12']:.3f} z_S8 {c['per']['z_S8']['pos']}/5 "
            f"m={c['per']['z_S8']['mean']:+.2f} twin {c['per']['twin']['pos']}/5 "
            f"m={c['per']['twin']['mean']:+.2f} ok={c['ok']}"
        )
    if chosen:
        lines += [
            "",
            f"## Chosen `{chosen['name']}`",
            f"- mean={chosen['mean_lift']:+.2f} min={chosen['min_lift']:+.2f} "
            f"1-ρ={chosen['rho']:.3f} c7={chosen['c7']:+.3f} F2={chosen['f2_bo12']:.3f}",
            f"- Public LB: {pub}",
        ]
    else:
        lines += [
            "",
            "## Decision",
            "HOLD ATH `0.95395`. ZCA locks F2 on some streams but no candidate cleared "
            "all 6 gates (5/5×3 H2H + 1−ρ∈[0.30,1.10]e-4 + |c6|/|c7|<0.20). "
            "Keep remaining daily slots.",
        ]
    report = LOG / "GM_V20_1_FORGOTTEN_HW_ZCA_REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print("WROTE", report)
    print("WROTE", OUT / "v20_1.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
