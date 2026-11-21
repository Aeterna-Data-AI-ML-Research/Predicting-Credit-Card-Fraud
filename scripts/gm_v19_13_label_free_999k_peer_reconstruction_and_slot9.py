#!/usr/bin/env python3
"""GM V19.13 — Ultra-clean 999k peer reconstruction on Slot8.

11 flaws locked out: leave-self-out only, pure numerical keys, freq de-corr,
Gram-Schmidt e_va_noise ⊥ M4, L7 coarse teacher, F2/anti-fail-ray gates.
Submit Slot9 IFF all 6 gates. 0 GPU pods.
"""
from __future__ import annotations

import json
import subprocess
import time
import warnings
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
FLAG_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
SEEDS = (11, 21, 31, 41, 51)
M0_RAW, M0_SYM = 10.0, 12.0


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


def codes_from_str(arr):
    c, _ = pd.factorize(np.asarray(arr, dtype=object), sort=False)
    return c.astype(np.int64)


def peer_mean_only(z, gids, m0):
    """Leave-self-out peer mean ONLY — zero self, returns (peer, cnt_per_row)."""
    z = np.asarray(z, float)
    g = np.asarray(gids, np.int64)
    _, g2 = np.unique(g, return_inverse=True)
    nbin = int(g2.max()) + 1
    cnt = np.bincount(g2, minlength=nbin).astype(np.float64)
    sz = np.bincount(g2, weights=z, minlength=nbin)
    c = cnt[g2]
    peer = (sz[g2] - z) / (c - 1.0 + m0)
    return np.where(c <= 1, 0.0, peer), c


def freq_decorr(peer, cnt, m0):
    """Regress out [log1p(cnt), (cnt-1)/(cnt-1+m0)] via OLS."""
    peer = np.asarray(peer, float)
    cnt = np.asarray(cnt, float)
    shrink = (cnt - 1.0) / (cnt - 1.0 + m0)
    Xf = np.column_stack([np.log1p(np.maximum(cnt, 0.0)), shrink])
    Xf = Xf - Xf.mean(0, keepdims=True)
    y = peer - peer.mean()
    XtX = Xf.T @ Xf + 1e-6 * np.eye(2)
    beta = np.linalg.solve(XtX, Xf.T @ y)
    return y - Xf @ beta


def ols_project_out(v, basis):
    """v ⊥ columns of basis (unitized)."""
    v = unitize(v)
    B = np.column_stack([unitize(b) for b in basis])
    XtX = B.T @ B + 1e-8 * np.eye(B.shape[1])
    beta = np.linalg.solve(XtX, B.T @ v)
    return unitize(v - B @ beta), beta


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
    r178 = np.load(OUT / "v17_8_residuals.npz")
    peers = np.load(OUT / "v18_9_peers.npz")

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

    z_s5 = z_s3 + 0.020 * u_r2
    z_s8 = z_s3 + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_t8 = z_twin + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_n8 = z_nnls + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    bases = {"z_S8": z_s8, "twin": z_t8, "nnls": z_n8}

    rebuild = qlock(L(slot3) + W8_O * ut_r2 + W8_C * ut_cat + W8_U * ut_u5, lv)
    rho_rebuild = one_m_rho_e4(rebuild, slot8)
    print(f"=== V19.13 === Slot8 rebuild 1-ρ={rho_rebuild:.6e}e-4 t={time.time()-t0:.1f}s")
    assert rho_rebuild < 1e-5, rho_rebuild

    Bo12 = np.column_stack([Bo9, u_r2, u_cat, u_u5])
    Bt12 = np.column_stack([Bt9, ut_r2, ut_cat, ut_u5])

    # ---------- Cached M4 / e_va_noise diagnostic ----------
    print("=== Step1 M4 / e_va_noise diagnostic ===", f"t={time.time()-t0:.1f}s")
    M1o, M1t = unitize(peers["M1_oof"]), unitize(peers["M1_test"])
    M2o, M2t = unitize(peers["M2_oof"]), unitize(peers["M2_test"])
    M4o, M4t = unitize(peers["M4_oof"]), unitize(peers["M4_test"])
    m12_o = unitize(0.5 * (M1o + M2o))
    m12_t = unitize(0.5 * (M1t + M2t))
    # Gram-Schmidt: e_va = m12 - proj_M4(m12)  ⇒  corr(e_va, M4) ≡ 0
    beta_va = float(np.dot(M4o, m12_o) / (np.dot(M4o, M4o) + 1e-12))
    e_va_o = unitize(m12_o - beta_va * M4o)
    e_va_t = unitize(m12_t - beta_va * M4t)
    corr_e_m4 = float(np.corrcoef(e_va_o, M4o)[0, 1])
    corr_m12_m4 = float(np.corrcoef(m12_o, M4o)[0, 1])
    print(
        f"  corr(0.5(M1+M2), M4)={corr_m12_m4:+.4f}  beta_OLS={beta_va:.4f}  "
        f"corr(e_va_noise, M4)={corr_e_m4:+.6f}"
    )

    Bo_m4 = np.column_stack([Bo12, e_va_o, u_r1])
    Bt_m4 = np.column_stack([Bt12, e_va_t, ut_r1])
    oo_m4, ot_m4, _ = frozen_lofo_orth(M4o, M4t, Bo_m4, Bt_m4, folds, ridge_alpha=10.0)
    u_m4c = unitize(oo_m4)
    if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u_m4c[te0]) < roc_auc_score(
        y[te0], z_s8[te0] - 0.01 * u_m4c[te0]
    ):
        u_m4c, ot_m4 = -u_m4c, -ot_m4
    ut_m4c = unitize(ot_m4)
    g_m4 = forensic(oo_m4, ot_m4, Bt12)
    prof_m4 = h2h(z_s8 + 0.012 * u_m4c, z_s8, y, folds)
    c7_m4 = float(np.corrcoef(ut_m4c, bad7)[0, 1])
    corr_m4c_eva = float(np.corrcoef(u_m4c, e_va_o)[0, 1])
    print(
        f"  M4_clean: F2_Bo12={g_m4['F2']:.4f} disp={g_m4['raw_disp']:.3f} "
        f"c7={c7_m4:+.3f} corr(e_va)={corr_m4c_eva:+.4f} "
        f"folds={prof_m4['folds']} m={prof_m4['mean']:+.2f} pos={prof_m4['pos']}/5"
    )

    # ---------- Build attributes & keys across 999944 ----------
    print("=== Step1 build 999k peer banks ===", f"t={time.time()-t0:.1f}s")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    assert len(train) == n_tr and len(test) == n_te

    amt = np.concatenate([train["amount_usd"].to_numpy(float), test["amount_usd"].to_numpy(float)])
    bal = np.concatenate(
        [train["account_balance_usd"].to_numpy(float), test["account_balance_usd"].to_numpy(float)]
    )
    dist = np.concatenate(
        [
            train["distance_from_home_km"].to_numpy(float),
            test["distance_from_home_km"].to_numpy(float),
        ]
    )
    mr = np.concatenate(
        [train["merchant_risk_score"].to_numpy(float), test["merchant_risk_score"].to_numpy(float)]
    )
    vel = np.concatenate(
        [train["velocity_score"].to_numpy(float), test["velocity_score"].to_numpy(float)]
    )
    cvv = np.concatenate(
        [train["cvv_retry_count"].to_numpy(float), test["cvv_retry_count"].to_numpy(float)]
    )
    cage = np.concatenate(
        [train["card_age_months"].to_numpy(float), test["card_age_months"].to_numpy(float)]
    )
    flags = np.concatenate(
        [
            train[FLAG_COLS].to_numpy(float).sum(1),
            test[FLAG_COLS].to_numpy(float).sum(1),
        ]
    )
    amt_cents = (np.round(np.abs(amt) * 100).astype(np.int64) % 100) % 100
    bal_cents = (np.round(np.abs(bal) * 100).astype(np.int64) % 100) % 100

    # Pure numerical collision keys k1..k6 (+ k7,k8 for symD)
    key_codes = {
        "k1": codes_from_str(np.round(amt, 2).astype(str)),
        "k2": codes_from_str(np.round(bal, 2).astype(str)),
        "k3": codes_from_str(np.round(dist, 2).astype(str)),
        "k4": codes_from_str(
            (pd.Series(amt_cents.astype(str)) + "|" + pd.Series(bal_cents.astype(str))).to_numpy()
        ),
        "k5": codes_from_str(
            (
                pd.Series(np.round(amt, 0).astype(np.int64).astype(str))
                + "|"
                + pd.Series((np.round(bal / 10.0) * 10.0).astype(np.int64).astype(str))
            ).to_numpy()
        ),
        "k6": codes_from_str(
            (
                pd.Series(amt_cents.astype(str))
                + "|"
                + pd.Series(np.round(dist, 0).astype(np.int64).astype(str))
            ).to_numpy()
        ),
        "k7": codes_from_str(
            (
                pd.Series(np.round(amt, 1).astype(str))
                + "|"
                + pd.Series(np.round(dist, 1).astype(str))
            ).to_numpy()
        ),
        "k8": codes_from_str(
            (
                pd.Series(bal_cents.astype(str))
                + "|"
                + pd.Series(np.round(dist, 0).astype(np.int64).astype(str))
            ).to_numpy()
        ),
    }
    keys6 = ["k1", "k2", "k3", "k4", "k5", "k6"]
    keys8 = keys6 + ["k7", "k8"]

    # Non-overlapping attributes (no amt/bal/dist)
    A1 = unitize(mr)
    A2 = unitize(vel)
    A3 = unitize(flags)
    A4 = unitize(unitize(cvv) - 0.5 * unitize(np.log1p(np.maximum(cage, 0.0))))
    A5 = unitize(0.35 * A1 + 0.25 * A2 + 0.25 * A3 + 0.15 * A4)

    # U5 / U8 raw on train∪test (test: mean of fold L(stat))
    U5_raw_o = gini10(Z_o)
    U8_raw_o = col_z(raw_o[:, :, 1]).std(1) - col_z(raw_o[:, :, 0]).std(1)
    U5_raw_t = np.mean(
        np.stack([gini10(col_z(raw_t[f, :, :, 1] - raw_t[f, :, :, 0])) for f in range(5)], 0), 0
    )
    U8_raw_t = np.mean(
        np.stack(
            [
                col_z(raw_t[f, :, :, 1]).std(1) - col_z(raw_t[f, :, :, 0]).std(1)
                for f in range(5)
            ],
            0,
        ),
        0,
    )
    A6 = unitize(
        np.concatenate(
            [
                unitize(0.60 * unitize(U5_raw_o) + 0.40 * unitize(U8_raw_o)),
                unitize(0.60 * unitize(U5_raw_t) + 0.40 * unitize(U8_raw_t)),
            ]
        )
    )

    # A7 L7 coarse: OOF mean-llr on train, mean test-fold mean-llr on test
    l7_o = unitize(llr_o.mean(1))
    l7_t = unitize(
        np.mean([(raw_t[f, :, :, 1] - raw_t[f, :, :, 0]).mean(1) for f in range(5)], 0)
    )
    A7 = unitize(np.concatenate([l7_o, l7_t]))

    attrs = {"A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5, "A6": A6, "A7": A7}
    attrs_p1 = ["A1", "A2", "A3", "A4", "A5", "A6"]  # no L7
    attrs_p2 = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]

    def build_peer_bank(key_list, attr_names, m0, pairwise_attrs):
        cols, names = [], []
        peer_cache = {}  # (attr, key) -> (peer_decorr, cnt)
        for an in attr_names:
            for kn in key_list:
                p, c = peer_mean_only(attrs[an], key_codes[kn], m0=m0)
                p = freq_decorr(p, c, m0)
                peer_cache[(an, kn)] = (p, c)
                cols.append(p)
                names.append(f"{an}_{kn}")
        for an in pairwise_attrs:
            for i, j in combinations(range(len(key_list)), 2):
                ki, kj = key_list[i], key_list[j]
                pi = peer_cache[(an, ki)][0]
                pj = peer_cache[(an, kj)][0]
                cols.append(pi - pj)
                names.append(f"d_{an}_{ki}_{kj}")
        X = np.column_stack(cols).astype(np.float64)
        return X, names

    # ~54–57 cols: 6 keys × 7 attrs means + C(6,2) A5 pairwise diffs
    Xraw_all, raw_names = build_peer_bank(keys6, attrs_p2, M0_RAW, pairwise_attrs=["A5"])
    print(f"  X_raw_u_peer999k cols={Xraw_all.shape[1]} (names={len(raw_names)})")

    # P1 subset: drop A7 columns
    p1_idx = [i for i, n in enumerate(raw_names) if not n.startswith("A7_")]
    Xraw_p1_all = Xraw_all[:, p1_idx]
    print(f"  X_raw P1 (no A7) cols={Xraw_p1_all.shape[1]}")

    Xraw_o, Xraw_t = Xraw_all[:n_tr], Xraw_all[n_tr:]
    Xraw_p1_o, Xraw_p1_t = Xraw_p1_all[:n_tr], Xraw_p1_all[n_tr:]

    # ---------- Bank 2: uniform D_f q20 teacher peers ----------
    print("=== Step1 X_symD q20 teacher ===", f"t={time.time()-t0:.1f}s")
    q_cols = [
        "amount_usd",
        "account_balance_usd",
        "distance_from_home_km",
        "velocity_score",
        "merchant_risk_score",
        "hours_since_last_txn",
        "cvv_retry_count",
        "card_age_months",
        "txn_count_last_24h",
        "customer_age",
    ]

    # joint q20 over train∪test
    Xq_all = []
    for c in q_cols:
        v = np.concatenate([train[c].to_numpy(float), test[c].to_numpy(float)])
        r = rankdata(v, method="average")
        q = np.clip((r / (len(v) + 1e-12) * 20).astype(np.int64), 0, 19).astype(np.float64)
        Xq_all.append(q)
    Xq_all = np.column_stack(Xq_all).astype(np.float32)
    Xq_o, Xq_t = Xq_all[:n_tr], Xq_all[n_tr:]

    # Uniform D_f: 3-subfold OOF on va; full-va fit → tr/test; unitize per block
    S_folds = np.zeros((5, n_all))
    for f, (tr, va) in enumerate(folds):
        skf = StratifiedKFold(3, shuffle=True, random_state=2026 + f)
        d_va = np.zeros(len(va))
        for tr2, te2 in skf.split(np.zeros(len(va)), y[va]):
            m = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.06,
                max_bin=63,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                verbosity=-1,
                n_jobs=-1,
                random_state=100 + f,
            )
            m.fit(Xq_o[va][tr2], z_s5[va][tr2])
            d_va[te2] = m.predict(Xq_o[va][te2])
        m_full = lgb.LGBMRegressor(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.06,
            max_bin=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            verbosity=-1,
            n_jobs=-1,
            random_state=200 + f,
        )
        m_full.fit(Xq_o[va], z_s5[va])
        d_tr = m_full.predict(Xq_o[tr])
        d_te = m_full.predict(Xq_t)
        S_folds[f, va] = unitize(d_va)
        S_folds[f, tr] = unitize(d_tr)
        S_folds[f, n_tr:] = unitize(d_te)
        print(f"  D_f fold{f} done t={time.time()-t0:.1f}s")
    # Peer attribute = mean of fold-wise unitized teachers (uniform over 999k)
    D_peer_attr = unitize(S_folds.mean(0))

    # X_symD: 8 means + C(8,2)=28 diffs = 36
    sym_cols, sym_names = [], []
    sym_cache = {}
    for kn in keys8:
        p, c = peer_mean_only(D_peer_attr, key_codes[kn], m0=M0_SYM)
        p = freq_decorr(p, c, M0_SYM)
        sym_cache[kn] = p
        sym_cols.append(p)
        sym_names.append(f"D_{kn}")
    for i, j in combinations(range(8), 2):
        ki, kj = keys8[i], keys8[j]
        sym_cols.append(sym_cache[ki] - sym_cache[kj])
        sym_names.append(f"dD_{ki}_{kj}")
    Xsym_all = np.column_stack(sym_cols).astype(np.float64)
    Xsym_o, Xsym_t = Xsym_all[:n_tr], Xsym_all[n_tr:]
    print(f"  X_symD cols={Xsym_all.shape[1]} t={time.time()-t0:.1f}s")

    # ---------- Platt residuals on Slot8 ----------
    g_s8, h_s8 = [], []
    for tr, _ in folds:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s8[tr].reshape(-1, 1), y[tr])
        a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
        p = expit(a * z_s8 + b)
        g_s8.append(y - p)
        h_s8.append(np.clip(p * (1 - p), 1e-6, None))

    # ---------- Fit ensembles ----------
    print("=== Step2 fit P1..P3 ===", f"t={time.time()-t0:.1f}s")

    def fit_ensemble(Xtr, Xte, tag):
        # Ridge
        oo_r = np.zeros(n_tr)
        te_r = []
        for f, (tr, va) in enumerate(folds):
            sw = np.sqrt(h_s8[f][tr])
            lr = Ridge(alpha=800.0, fit_intercept=True)
            lr.fit(Xtr[tr] * sw[:, None], g_s8[f][tr] * sw)
            oo_r[va] = lr.predict(Xtr[va])
            te_r.append(lr.predict(Xte))
        te_r = np.mean(np.stack(te_r, 0), 0)
        # Cat depth3
        oo_c = np.zeros(n_tr)
        te_c = []
        for seed in SEEDS:
            oo = np.zeros(n_tr)
            tef = []
            for f, (tr, va) in enumerate(folds):
                m = CatBoostRegressor(
                    loss_function="RMSE",
                    depth=3,
                    l2_leaf_reg=45,
                    iterations=110,
                    learning_rate=0.03,
                    random_seed=seed,
                    verbose=False,
                    thread_count=4,
                    bootstrap_type="Bernoulli",
                    subsample=0.8,
                )
                m.fit(Xtr[tr], g_s8[f][tr], sample_weight=h_s8[f][tr])
                oo[va] = m.predict(Xtr[va])
                tef.append(m.predict(Xte))
            oo_c += oo
            te_c.append(np.mean(np.stack(tef, 0), 0))
        oo_c /= len(SEEDS)
        te_c = np.mean(np.stack(te_c, 0), 0)
        # LGB tiny
        oo_l = np.zeros(n_tr)
        te_l = []
        for seed in SEEDS:
            oo = np.zeros(n_tr)
            tef = []
            for f, (tr, va) in enumerate(folds):
                m = lgb.LGBMRegressor(
                    num_leaves=4,
                    min_child_samples=900,
                    n_estimators=35,
                    learning_rate=0.03,
                    reg_lambda=50.0,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    verbosity=-1,
                    n_jobs=4,
                    random_state=seed,
                )
                m.fit(Xtr[tr], g_s8[f][tr], sample_weight=h_s8[f][tr])
                oo[va] = m.predict(Xtr[va])
                tef.append(m.predict(Xte))
            oo_l += oo
            te_l.append(np.mean(np.stack(tef, 0), 0))
        oo_l /= len(SEEDS)
        te_l = np.mean(np.stack(te_l, 0), 0)
        oof = (oo_r + oo_c + oo_l) / 3.0
        te = (te_r + te_c + te_l) / 3.0
        print(f"  {tag} done t={time.time()-t0:.1f}s")
        return oof, te

    P1_o, P1_t = fit_ensemble(
        Xraw_p1_o.astype(np.float32), Xraw_p1_t.astype(np.float32), "P1_raw_only"
    )
    P2_o, P2_t = fit_ensemble(Xraw_o.astype(np.float32), Xraw_t.astype(np.float32), "P2_raw_L7")
    P3_o, P3_t = fit_ensemble(Xsym_o.astype(np.float32), Xsym_t.astype(np.float32), "P3_symD")

    def orient_vs(vo, vt, Bo, Bt, tag):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo, Bt, folds, ridge_alpha=10.0)
        g12 = forensic(oo, ot, Bt12)
        g9 = forensic(oo, ot, Bt9)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s8[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        print(
            f"  {tag}: F2_Bo9={g9['F2']:.4f} F2_Bo12={g12['F2']:.4f} "
            f"disp={g12['raw_disp']:.3f} KS={g12['raw_KS']:.3f}"
        )
        return u, ut, g12, g9

    Bo_m5 = np.column_stack([Bo12, u_m5, u_r1])
    Bt_m5 = np.column_stack([Bt12, ut_m5, ut_r1])
    Bo_eva = np.column_stack([Bo12, e_va_o, u_r1])
    Bt_eva = np.column_stack([Bt12, e_va_t, ut_r1])

    fam = {}
    fam["P1_raw_only"] = orient_vs(P1_o, P1_t, Bo_m5, Bt_m5, "P1_orth_M5")
    fam["P2_raw_L7"] = orient_vs(P2_o, P2_t, Bo_m5, Bt_m5, "P2_orth_M5")
    fam["P3a_orth_M5"] = orient_vs(P3_o, P3_t, Bo_m5, Bt_m5, "P3a_orth_M5")
    fam["P3b_orth_va"] = orient_vs(P3_o, P3_t, Bo_eva, Bt_eva, "P3b_orth_va_noise")
    fam["M4_clean"] = (u_m4c, ut_m4c, g_m4, forensic(u_m4c, ut_m4c, Bt9))

    # combos of cleanest
    for a, b, wa, wb, nm in [
        ("P1_raw_only", "P3b_orth_va", 0.5, 0.5, "P4_P1_P3b"),
        ("P2_raw_L7", "P3b_orth_va", 0.45, 0.55, "P4_P2_P3b"),
        ("P1_raw_only", "P2_raw_L7", 0.5, 0.5, "P4_P1_P2"),
        ("P1_raw_only", "M4_clean", 0.4, 0.6, "P4_P1_M4c"),
        ("P3b_orth_va", "M4_clean", 0.5, 0.5, "P4_P3b_M4c"),
    ]:
        u = unitize(wa * fam[a][0] + wb * fam[b][0])
        ut = unitize(wa * fam[a][1] + wb * fam[b][1])
        g12 = forensic(u, ut, Bt12)
        g9 = forensic(u, ut, Bt9)
        fam[nm] = (u, ut, g12, g9)
        print(f"  {nm}: F2_Bo9={g9['F2']:.4f} F2_Bo12={g12['F2']:.4f} disp={g12['raw_disp']:.3f}")

    np.savez_compressed(
        OUT / "v19_13_families.npz",
        **{f"{k}_oof": v[0] for k, v in fam.items()},
        **{f"{k}_test": v[1] for k, v in fam.items()},
        e_va_oof=e_va_o,
        e_va_test=e_va_t,
    )

    # ---------- Step3 audit ----------
    print("=== Step3 H2H vs Slot8 ===", f"t={time.time()-t0:.1f}s")
    candidates, winners = [], []

    def eval_cand(name, u, ut, g12, g9):
        f2_bo9, f2_bo12 = g9["F2"], g12["F2"]
        f3 = abs(float(np.corrcoef(unitize(ut), unitize(L(slot8) - L(slot5)))[0, 1]))
        corr_eva = float(np.corrcoef(unitize(ut), e_va_t)[0, 1])
        print(
            f"  {name}: disp={g12['raw_disp']:.3f} KS={g12['raw_KS']:.3f} "
            f"F2_9={f2_bo9:.4f} F2_12={f2_bo12:.4f} F3={f3:.4f} corr_eva={corr_eva:+.3f}"
        )
        for w in (0.008, 0.010, 0.012, 0.015, 0.018, 0.022):
            oof = {b: bases[b] + w * u for b in bases}
            per = {b: h2h(oof[b], bases[b], y, folds) for b in bases}
            cand = qlock(L(slot8) + w * ut, lv)
            rho = one_m_rho_e4(cand, slot8)
            delta = unitize(L(cand) - L(slot8))
            c6 = float(np.corrcoef(delta, bad6)[0, 1])
            c7 = float(np.corrcoef(delta, bad7)[0, 1])
            corr_eva_d = float(np.corrcoef(delta, e_va_t)[0, 1])
            ok = (
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
                and corr_eva_d <= 0.05
                and 0.92 <= g12["raw_disp"] <= 1.08
                and g12["raw_KS"] < 0.025
                and f2_bo9 < 0.035
                and f2_bo12 < 0.038
                and f3 < 0.045
            )
            mark = "★" if ok else ("~" if per["z_S8"]["pos"] >= 4 else " ")
            print(
                f"{mark} {name}_w{w:.3f}: 1-ρ={rho:.3f} c6={c6:+.3f} c7={c7:+.3f} "
                f"eva={corr_eva_d:+.3f} ok={ok} | "
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
                "corr_eva": corr_eva_d,
                "f2_bo9": f2_bo9,
                "f2_bo12": f2_bo12,
                "f3": f3,
                "gates": g12,
                "ok": ok,
                "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                "min_lift": float(np.min([per[b]["min"] for b in per])),
            }
            candidates.append(row)
            if ok:
                winners.append({**row, "cand": cand})

    for name, (u, ut, g12, g9) in fam.items():
        eval_cand(name, u, ut, g12, g9)

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
        out_csv = SUB / "sub_v19_13_LIVE_slot9_ultra_clean_999k_peer.csv"
        arr = chosen["cand"]
        assert len(arr) == n_te and np.isfinite(arr).all()
        pd.DataFrame({"id": ids, "is_fraud": arr}).to_csv(out_csv, index=False)
        msg = (
            f"v19.13 Slot9: {chosen['name']} ultra-clean 999k peer vs Slot8 0.95395 | "
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
                    if "v19_13" in row.get("fileName", ""):
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
                "f2_bo12": chosen["f2_bo12"],
                "per": chosen["per"],
            },
            "public_lb": pub,
            "ref": ref,
            "m4_diag": {
                "corr_m12_m4": corr_m12_m4,
                "corr_e_va_m4": corr_e_m4,
                "beta_va": beta_va,
                "M4_clean_F2": g_m4["F2"],
                "M4_clean_c7": c7_m4,
                "M4_clean_folds": prof_m4["folds"],
                "M4_clean_mean": prof_m4["mean"],
                "corr_m4c_eva": corr_m4c_eva,
            },
            "shapes": {
                "X_raw": int(Xraw_all.shape[1]),
                "X_raw_p1": int(Xraw_p1_all.shape[1]),
                "X_symD": int(Xsym_all.shape[1]),
            },
            "near_misses": near,
            "elapsed_s": time.time() - t0,
        }
    )
    OUT.joinpath("v19_13.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# GM V19.13 — Ultra-clean 999k peer → "
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
        f"| corr(e_va, M4) | {corr_e_m4:+.6f} |",
        f"| X_raw / X_symD | {Xraw_all.shape[1]} / {Xsym_all.shape[1]} |",
        "",
        "## M4 / e_va_noise diagnostic",
        f"- corr(0.5(M1+M2), M4) = {corr_m12_m4:+.4f}",
        f"- corr(e_va_noise, M4) = {corr_e_m4:+.6f} (Gram-Schmidt)",
        f"- M4_clean F2_Bo12={g_m4['F2']:.4f} c7={c7_m4:+.3f} "
        f"folds={prof_m4['folds']} m={prof_m4['mean']:+.2f}",
        "",
        "## Top near-misses",
    ]
    for c in near[:12]:
        lines.append(
            f"- `{c['name']}`: 1-ρ={c['rho']:.3f} c7={c['c7']:+.3f} "
            f"F2={c['f2_bo12']:.3f} eva={c['corr_eva']:+.3f} "
            f"z_S8 {c['per']['z_S8']['pos']}/5 m={c['per']['z_S8']['mean']:+.2f} "
            f"twin {c['per']['twin']['pos']}/5 m={c['per']['twin']['mean']:+.2f} ok={c['ok']}"
        )
    if chosen:
        lines += ["", f"## Chosen `{chosen['name']}`", f"- mean={chosen['mean_lift']:+.2f}"]
    else:
        lines += ["", "## Decision", "HOLD ATH `0.95395`. Keep remaining daily slots."]
    report = LOG / "GM_V19_13_ULTRA_CLEAN_999K_PEER_REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print("WROTE", report)
    print("WROTE", OUT / "v19_13.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
