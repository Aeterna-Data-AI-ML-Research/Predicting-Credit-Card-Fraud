#!/usr/bin/env python3
"""GM V18.9 — Zero-R1, zero-logcnt strict peer + continuous teacher distill.

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import rankdata, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from catboost import CatBoostRegressor

    HAS_CAT = True
except Exception:
    HAS_CAT = False

REPO = Path(__file__).resolve().parents[1]
TMP15 = Path("/tmp/s6e11_v15")
TMP16 = Path("/tmp/s6e11_v16")
OUT = TMP16 / "out"
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
DATA = REPO / "data"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
NO_SUBMIT = True
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W_R2 = 0.020
SCALE = 299984 / 139992


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


def std10(Z):
    return np.std(Z, axis=-1)


def frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=1.0):
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


def forensic_gates(orth_o, orth_t, Bt, live_resid, bn=None):
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
        "F2": max_leak < 0.035,
        "F3": abs(c_live) < 0.05,
        "verified": (0.90 <= disp <= 1.10) and (max_leak < 0.035) and (abs(c_live) < 0.05),
    }


def score_fixed(u, zbase, y, folds, w_grid=(0.008, 0.01, 0.015, 0.02, 0.025, 0.03)):
    u = unitize(u)
    base = [roc_auc_score(y[te], zbase[te]) for _, te in folds]
    best = None
    for w in w_grid:
        d = [
            (roc_auc_score(y[te], zbase[te] + w * u[te]) - base[i]) * 1e5
            for i, (_, te) in enumerate(folds)
        ]
        row = {
            "w": w,
            "pos": int(sum(x > 0 for x in d)),
            "mean": float(np.mean(d)),
            "min": float(np.min(d)),
            "folds": [round(x, 3) for x in d],
        }
        if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
            best = row
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
    best.update({"sf25": int((d25 > 0).sum()), "t": float(mean / (se + 1e-18)), "wilcoxon_p": wp})
    return best


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
    c, _ = pd.factorize(arr, sort=False)
    return c.astype(np.int64)


def peer_mean_only(z, gids, m0):
    """Leave-self-out peer mean ONLY — zero logcnt."""
    z = np.asarray(z, float)
    g = np.asarray(gids, np.int64)
    _, g2 = np.unique(g, return_inverse=True)
    nbin = int(g2.max()) + 1
    cnt = np.bincount(g2, minlength=nbin).astype(np.float64)
    sz = np.bincount(g2, weights=z, minlength=nbin)
    c = cnt[g2]
    peer = (sz[g2] - z) / (c - 1.0 + m0)
    return np.where(c <= 1, 0.0, peer), c


def within_group_rank(values, group_ids):
    v = np.asarray(values, float)
    g = np.asarray(group_ids, np.int64)
    out = np.empty(len(v), dtype=np.float64)
    order = np.lexsort((v, g))
    g_s, v_s = g[order], v[order]
    breaks = np.flatnonzero(np.r_[True, g_s[1:] != g_s[:-1], True])
    ranks = np.empty(len(v), dtype=np.float64)
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = b - a
        ranks[a:b] = (rankdata(v_s[a:b], method="average") - 0.5) / n
    out[order] = ranks
    return out


def main():
    assert NO_SUBMIT
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
    dE_o2, dE_t = unitize(dE_o2), unitize(dE_t)
    if np.corrcoef(dE_o, dE_o2)[0, 1] < 0:
        dE_o = -dE_o
    if roc_auc_score(y[te0], z_ath[te0] + 0.01 * dE_o2[te0]) < roc_auc_score(
        y[te0], z_ath[te0] - 0.01 * dE_o2[te0]
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
    ridge_ath = Ridge(alpha=1.0, fit_intercept=True).fit(Bo9, z_ath)
    z_ath_t_sym = Bt9 @ ridge_ath.coef_ + ridge_ath.intercept_
    z_s3 = z_ath + 0.01045 * dE_o + 0.0055 * unitize(d1)
    z_s3_t = z_ath_t_sym + 0.01045 * dE_t + 0.0055 * unitize(d1t)

    y_s3t = unitize(L(slot3))
    ridge_s3 = Ridge(alpha=1.0, fit_intercept=True).fit(Bt9, y_s3t)
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

    res = np.load(OUT / "v17_8_residuals.npz")
    R2_oof, R2_test = res["R2_oof"], res["R2_test"]
    oo2, ot2, _ = frozen_lofo_orth(R2_oof, R2_test, Bo9, Bt9, folds, ridge_alpha=10.0)
    u2 = unitize(oo2)
    if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u2[te0]) < roc_auc_score(
        y[te0], z_s3[te0] - 0.01 * u2[te0]
    ):
        u2, ot2 = -u2, -ot2
    ut2 = unitize(ot2)
    z_s5 = z_s3 + W_R2 * u2
    z_s5_t_sym = z_s3_t + W_R2 * ut2
    z_slot5_twin = z_slot3_twin + W_R2 * u2
    z_nnls_s5 = z_nnls + W_R2 * u2
    Bo10 = np.column_stack([Bo9, u2])
    Bt10 = np.column_stack([Bt9, ut2])
    bn10 = BN9 + ["R2_orth"]
    bases_s5 = {"z_S5": z_s5, "z_Slot5_twin": z_slot5_twin, "z_NNLS_S5": z_nnls_s5}

    ab_s5 = []
    for tr, _ in folds:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s5[tr].reshape(-1, 1), y[tr])
        ab_s5.append((float(clf.coef_[0, 0]), float(clf.intercept_[0])))
    g5, h5 = [], []
    for f in range(5):
        a, b = ab_s5[f]
        p = expit(a * z_s5 + b)
        g5.append(y - p)
        h5.append(np.clip(p * (1 - p), 1e-6, None))

    # U5 / U6 from L7 raw
    raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
    llr_o = raw_o[:, :, 1] - raw_o[:, :, 0]
    llr_t = raw_t[:, :, :, 1] - raw_t[:, :, :, 0]
    llr_z_o = col_z(llr_o)
    U5_o = L_fold(gini10(llr_z_o), fold_id)
    P_o = 0.5 * (llr_z_o[:, :5] + llr_z_o[:, 5:])
    U6_o = L_fold(std10(P_o), fold_id)
    U5_t_folds, U6_t_folds = [], []
    for f in range(5):
        lz = col_z(llr_t[f])
        U5_t_folds.append(L(gini10(lz)))
        Pf = 0.5 * (lz[:, :5] + lz[:, 5:])
        U6_t_folds.append(L(std10(Pf)))
    U5_t = np.mean(np.stack(U5_t_folds, 0), 0)
    U6_t = np.mean(np.stack(U6_t_folds, 0), 0)

    def orient_orth(vo, vt, name=""):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo10, Bt10, folds, ridge_alpha=1.0)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s5[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        return unitize(u), unitize(ot)

    u_u5, ut_u5 = orient_orth(U5_o, U5_t)
    u_u6, ut_u6 = orient_orth(U6_o, U6_t)

    print("=== STEP 1: zero-logcnt banks ===", f"t={time.time()-t0:.1f}s")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
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
    cvv = np.concatenate(
        [train["cvv_retry_count"].to_numpy(float), test["cvv_retry_count"].to_numpy(float)]
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
    cat = mer  # no separate category

    # V18.6 exact 8 keys + 4 extras
    keys8 = [
        "k1_amt",
        "k2_bal",
        "k3_vel",
        "k4_mr",
        "k5_dist",
        "k6_mer_amt",
        "k7_mer_prof",
        "k8_cents_mer",
    ]
    keys12 = keys8 + ["k9_mer_bal10", "k10_cat_amt0", "k11_amtcents_cat", "k12_balcents_cat"]
    key_all = {
        "k1_amt": codes_from_str(np.round(amt, 8).astype(str)),
        "k2_bal": codes_from_str(np.round(bal, 8).astype(str)),
        "k3_vel": codes_from_str(np.round(vel, 8).astype(str)),
        "k4_mr": codes_from_str(np.round(mr, 8).astype(str)),
        "k5_dist": codes_from_str(np.round(dist, 8).astype(str)),
        "k6_mer_amt": codes_from_str(
            (pd.Series(mer) + "|" + pd.Series(np.round(amt).astype(np.int64).astype(str))).to_numpy()
        ),
        "k7_mer_prof": codes_from_str(
            (
                pd.Series(mer)
                + "|"
                + pd.Series(ch)
                + "|"
                + pd.Series(dev)
                + "|"
                + pd.Series(cvv.astype(np.int64).astype(str))
                + "|"
                + pd.Series(foreign)
                + "|"
                + pd.Series(auth)
            ).to_numpy()
        ),
        "k8_cents_mer": codes_from_str(
            (
                pd.Series(((np.round(np.abs(amt) * 100).astype(np.int64) % 100) % 100).astype(str))
                + "|"
                + pd.Series(((np.round(np.abs(bal) * 100).astype(np.int64) % 100) % 100).astype(str))
                + "|"
                + pd.Series(mer)
            ).to_numpy()
        ),
        "k9_mer_bal10": codes_from_str(
            (
                pd.Series(mer)
                + "|"
                + pd.Series((np.round(bal / 10.0) * 10.0).astype(np.int64).astype(str))
            ).to_numpy()
        ),
        "k10_cat_amt0": codes_from_str(
            (pd.Series(cat) + "|" + pd.Series(np.round(amt, 0).astype(np.int64).astype(str))).to_numpy()
        ),
        "k11_amtcents_cat": codes_from_str(
            (
                pd.Series(((np.round(np.abs(amt) * 100).astype(np.int64) % 100) % 100).astype(str))
                + "|"
                + pd.Series(cat)
            ).to_numpy()
        ),
        "k12_balcents_cat": codes_from_str(
            (
                pd.Series(((np.round(np.abs(bal) * 100).astype(np.int64) % 100) % 100).astype(str))
                + "|"
                + pd.Series(cat)
            ).to_numpy()
        ),
    }

    def build_va_peer_matrix(key_list, m0_list, m0_te_scale, pairwise_from_first_n=8, pre_orth=True):
        """Means only + pairwise diffs of first N keys at first m0. ZERO logcnt. NO qlock."""
        n_keys = len(key_list)
        n_m = len(m0_list)
        n_pair = len(list(combinations(range(pairwise_from_first_n), 2)))
        n_feat = n_keys * n_m + n_pair
        Xo = np.zeros((n_tr, n_feat), np.float32)
        Xt = np.zeros((n_te, n_feat), np.float32)
        means_o = {}
        means_t = {}
        col = 0
        z_te = unitize(z_s5_t_sym)
        for mi, m0 in enumerate(m0_list):
            for kn in key_list:
                o = np.zeros(n_tr)
                for _, va in folds:
                    z_va = unitize(z_s5[va])
                    p, _ = peer_mean_only(z_va, key_all[kn][va], m0=m0)
                    o[va] = p
                t, _ = peer_mean_only(z_te, key_all[kn][n_tr:], m0=m0 * m0_te_scale)
                if pre_orth:
                    oo, ot, _ = frozen_lofo_orth(o, t, Bo10, Bt10, folds, ridge_alpha=1.0)
                    Xo[:, col] = unitize(oo)
                    Xt[:, col] = unitize(ot)
                else:
                    Xo[:, col] = unitize(o)
                    Xt[:, col] = unitize(t)
                means_o[(kn, mi)] = Xo[:, col].copy()
                means_t[(kn, mi)] = Xt[:, col].copy()
                col += 1
        for a, b in combinations(range(pairwise_from_first_n), 2):
            ka, kb = key_list[a], key_list[b]
            Xo[:, col] = unitize(means_o[(ka, 0)] - means_o[(kb, 0)])
            Xt[:, col] = unitize(means_t[(ka, 0)] - means_t[(kb, 0)])
            col += 1
        assert col == n_feat
        return Xo, Xt

    # Bank A: exact V18.6 peer8 — m0_va=5, m0_test≈10.7 (=5*SCALE), ZERO logcnt, no qlock
    X_peer8_va, X_peer8_te = build_va_peer_matrix(
        keys8, [5.0], m0_te_scale=SCALE, pairwise_from_first_n=8, pre_orth=True
    )
    print(f"  X_peer8={X_peer8_va.shape[1]} (expect 36) m0_te={5.0*SCALE:.2f} t={time.time()-t0:.1f}s")

    X_peer12_va, X_peer12_te = build_va_peer_matrix(
        keys12, [5.0, 15.0], m0_te_scale=SCALE, pairwise_from_first_n=8, pre_orth=True
    )
    print(f"  X_peer12_va={X_peer12_va.shape[1]} (expect 52) t={time.time()-t0:.1f}s")

    # Cohort (for M2 = P4)
    g1 = codes_from_str(mer)
    g2 = codes_from_str(
        (pd.Series(mer) + "|" + pd.Series(ch) + "|" + pd.Series(dev)).to_numpy()
    )
    cohort_tr, cohort_te = [], []
    global_u = {}
    for cname, vals in [
        ("amount_usd", amt),
        ("account_balance_usd", bal),
        ("velocity_score", vel),
        ("merchant_risk_score", mr),
        ("distance_from_home_km", dist),
        ("hours_since_last_txn", hrs),
    ]:
        ug = U(vals)
        global_u[cname] = ug
        u1 = within_group_rank(vals, g1)
        u2 = within_group_rank(vals, g2)
        for arr in (u1, u2, u2 - ug):
            cohort_tr.append(arr[:n_tr])
            cohort_te.append(arr[n_tr:])
    for arr in (
        amt / (np.abs(bal) + 10.0),
        global_u["amount_usd"] - global_u["account_balance_usd"],
        global_u["velocity_score"] - global_u["hours_since_last_txn"],
        global_u["merchant_risk_score"] + global_u["velocity_score"],
    ):
        cohort_tr.append(arr[:n_tr])
        cohort_te.append(arr[n_tr:])
    Xc_o = np.column_stack(cohort_tr).astype(np.float32)
    Xc_t = np.column_stack(cohort_te).astype(np.float32)

    # X_rich for D_f — 24 raw + cohort ranks + coarse q20 (exclude exact k1..k12 collisions)
    def build_X_rich():
        cols_tr, cols_te = [], []
        for c in (
            "amount_usd",
            "account_balance_usd",
            "velocity_score",
            "merchant_risk_score",
            "distance_from_home_km",
            "hours_since_last_txn",
            "txn_count_last_24h",
            "card_age_months",
            "customer_age",
            "cvv_retry_count",
            "time_of_day_hour",
            "day_of_week",
            "prior_disputes",
        ):
            cols_tr.append(train[c].to_numpy(float))
            cols_te.append(test[c].to_numpy(float))
            v = np.concatenate([train[c].to_numpy(float), test[c].to_numpy(float)])
            q = np.clip((rankdata(v) / (len(v) + 1e-12) * 20).astype(np.int64), 0, 19)
            cols_tr.append(q[:n_tr].astype(float))
            cols_te.append(q[n_tr:].astype(float))
            # coarse log-frequency of q20 bins (NOT exact collision keys)
            _, inv, cnt = np.unique(q, return_inverse=True, return_counts=True)
            lf = np.log1p(cnt[inv].astype(np.float64))
            cols_tr.append(lf[:n_tr])
            cols_te.append(lf[n_tr:])
        for c in (
            "is_foreign_transaction",
            "is_new_merchant",
            "used_vpn",
            "ip_country_mismatch",
            "billing_shipping_mismatch",
            "is_ai_generated_scam_attempt",
        ):
            cols_tr.append(train[c].to_numpy(float))
            cols_te.append(test[c].to_numpy(float))
        for c in ("merchant_category", "channel", "device_type", "card_type", "auth_method"):
            codes, _ = pd.factorize(pd.concat([train[c], test[c]], ignore_index=True))
            cols_tr.append(codes[:n_tr].astype(float))
            cols_te.append(codes[n_tr:].astype(float))
        for arr in cohort_tr:
            cols_tr.append(arr)
        for arr in cohort_te:
            cols_te.append(arr)
        return np.column_stack(cols_tr).astype(np.float32), np.column_stack(cols_te).astype(
            np.float32
        )

    Xr_tr, Xr_te = build_X_rich()
    print(f"  X_rich={Xr_tr.shape[1]} t={time.time()-t0:.1f}s")

    # Continuous teacher D_f: train on (X_rich[va], z_S5[va]), predict tr+test
    distill_corr = {}
    # Bank B: for each fold, s_all with va=z_S5, tr/test=D_f
    n_feat_d = 12 + 28  # 12 means m0=15 + C(8,2) diffs
    Xd_o = np.zeros((n_tr, n_feat_d), np.float32)
    Xd_t = np.zeros((n_te, n_feat_d), np.float32)
    means_d_o, means_d_t = {}, {}

    for f, (tr, va) in enumerate(folds):
        y_va = unitize(z_s5[va])
        # 5-subfold OOF inside va (distillation quality on held-out va rows)
        oof_va = np.zeros(len(va))
        skf = StratifiedKFold(5, shuffle=True, random_state=42 + f)
        # stratify via soft bins of teacher
        bins = np.clip((rankdata(y_va) / len(y_va) * 10).astype(int), 0, 9)
        for str_, ste_ in skf.split(np.zeros(len(va)), bins):
            m = lgb.LGBMRegressor(
                objective="regression",
                num_leaves=31,
                n_estimators=150,
                learning_rate=0.05,
                min_child_samples=100,
                colsample_bytree=0.7,
                subsample=0.8,
                reg_lambda=1.0,
                random_state=42 + f,
                verbosity=-1,
                n_jobs=-1,
            )
            m.fit(Xr_tr[va][str_], y_va[str_])
            oof_va[ste_] = m.predict(Xr_tr[va][ste_])
        distill_corr[f"fold{f}_va_oof"] = float(np.corrcoef(oof_va, y_va)[0, 1])

        # full-va teacher for tr+test peers
        model = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=31,
            n_estimators=150,
            learning_rate=0.05,
            min_child_samples=100,
            colsample_bytree=0.7,
            subsample=0.8,
            reg_lambda=1.0,
            random_state=42 + f,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(Xr_tr[va], y_va)
        pred_tr = model.predict(Xr_tr[tr])
        pred_te = model.predict(Xr_te)
        pred_all_tr = np.zeros(n_tr)
        pred_all_tr[tr] = pred_tr
        pred_all_tr[va] = y_va  # exact continuous teacher on va — ZERO y[va]
        distill_corr[f"fold{f}_tr"] = float(np.corrcoef(pred_tr, z_s5[tr])[0, 1])
        distill_corr[f"fold{f}_te"] = float(np.corrcoef(pred_te, z_s5_t_sym)[0, 1])
        print(
            f"  D_f{f}: ρ(va_oof)={distill_corr[f'fold{f}_va_oof']:+.4f} "
            f"ρ(tr,zS5)={distill_corr[f'fold{f}_tr']:+.4f} "
            f"ρ(te,zS5sym)={distill_corr[f'fold{f}_te']:+.4f}"
        )
        s_all = np.concatenate([unitize(pred_all_tr), unitize(pred_te)])
        for ki, kn in enumerate(keys12):
            p, _ = peer_mean_only(s_all, key_all[kn], m0=15.0)
            Xd_o[va, ki] = p[va]
            if f == 0:
                means_d_t[kn] = []
            means_d_t[kn].append(p[n_tr:])

    for ki, kn in enumerate(keys12):
        Xd_t[:, ki] = unitize(np.mean(np.stack(means_d_t[kn], 0), 0))
        # unitize oof per-fold already mixed — global unitize
        Xd_o[:, ki] = unitize(Xd_o[:, ki])

    col = 12
    for a, b in combinations(range(8), 2):
        ka, kb = keys12[a], keys12[b]
        Xd_o[:, col] = unitize(Xd_o[:, a] - Xd_o[:, b])
        Xd_t[:, col] = unitize(Xd_t[:, a] - Xd_t[:, b])
        col += 1
    print(f"  X_peer12_distill={Xd_o.shape[1]} t={time.time()-t0:.1f}s")

    # ===== STEP 2: fit M1..M5 =====
    print("=== STEP 2: fit M1..M5 ===")
    seeds = [42, 2026, 7, 100, 999]
    lgb_p = dict(
        objective="regression",
        num_leaves=6,
        min_child_samples=500,
        n_estimators=40,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbosity=-1,
        n_jobs=-1,
    )

    def fit_lgb(Xo, Xt):
        oof = np.zeros(n_tr)
        te_acc = []
        for seed in seeds:
            oo = np.zeros(n_tr)
            tef = []
            for f, (tr, va) in enumerate(folds):
                model = lgb.LGBMRegressor(**lgb_p, random_state=seed)
                model.fit(Xo[tr], g5[f][tr], sample_weight=h5[f][tr])
                oo[va] = model.predict(Xo[va])
                tef.append(model.predict(Xt))
            oof += oo
            te_acc.append(np.mean(np.stack(tef, 0), 0))
        return oof / len(seeds), np.mean(np.stack(te_acc, 0), 0)

    def fit_ridge_lgb_cat(Xo, Xt, use_cat=False):
        # ridge on mean columns (first 24 for peer12 = 12*2)
        n_mean = min(24, Xo.shape[1])
        Ro, Rt = Xo[:, :n_mean], Xt[:, :n_mean]
        R = np.zeros(n_tr)
        Rte = []
        for f, (tr, va) in enumerate(folds):
            sw = np.sqrt(h5[f][tr])
            lr = Ridge(alpha=250.0, fit_intercept=True)
            lr.fit(Ro[tr] * sw[:, None], g5[f][tr] * sw)
            R[va] = lr.predict(Ro[va])
            Rte.append(lr.predict(Rt))
        R_t = np.mean(np.stack(Rte, 0), 0)
        L_o, L_t = fit_lgb(Xo, Xt)
        parts_o = [0.3 * unitize(R), 0.5 * unitize(L_o)]
        parts_t = [0.3 * unitize(R_t), 0.5 * unitize(L_t)]
        if use_cat and HAS_CAT:
            C = np.zeros(n_tr)
            Cte = []
            for f, (tr, va) in enumerate(folds):
                model = CatBoostRegressor(
                    loss_function="RMSE",
                    depth=3,
                    l2_leaf_reg=50,
                    iterations=60,
                    learning_rate=0.05,
                    random_seed=42,
                    verbose=False,
                    thread_count=4,
                    min_data_in_leaf=500,
                )
                model.fit(Xo[tr], g5[f][tr], sample_weight=h5[f][tr])
                C[va] = model.predict(Xo[va])
                Cte.append(model.predict(Xt))
            C_t = np.mean(np.stack(Cte, 0), 0)
            parts_o.append(0.2 * unitize(C))
            parts_t.append(0.2 * unitize(C_t))
            return unitize(sum(parts_o)), unitize(sum(parts_t))
        return unitize(0.35 * unitize(R) + 0.65 * unitize(L_o)), unitize(
            0.35 * unitize(R_t) + 0.65 * unitize(L_t)
        )

    M1, M1_t = fit_lgb(X_peer8_va, X_peer8_te)
    print(f"  M1 done t={time.time()-t0:.1f}s")
    M2, M2_t = fit_lgb(
        np.column_stack([X_peer8_va, Xc_o]).astype(np.float32),
        np.column_stack([X_peer8_te, Xc_t]).astype(np.float32),
    )
    print(f"  M2 done t={time.time()-t0:.1f}s")
    M3, M3_t = fit_ridge_lgb_cat(X_peer12_va, X_peer12_te, use_cat=True)
    print(f"  M3 done t={time.time()-t0:.1f}s")
    M4, M4_t = fit_ridge_lgb_cat(Xd_o, Xd_t, use_cat=False)
    print(f"  M4 done t={time.time()-t0:.1f}s")
    M5 = unitize(
        0.25 * unitize(M1) + 0.25 * unitize(M2) + 0.25 * unitize(M3) + 0.25 * unitize(M4)
    )
    M5_t = unitize(
        0.25 * unitize(M1_t)
        + 0.25 * unitize(M2_t)
        + 0.25 * unitize(M3_t)
        + 0.25 * unitize(M4_t)
    )

    # ===== STEP 3 =====
    print("=== STEP 3: R1-free scorecard ===")
    results = {}
    oriented = {}

    def eval_sig(name, vo, vt):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo10, Bt10, folds, ridge_alpha=1.0)
        gates = forensic_gates(oo, ot, Bt10, live_resid, bn=bn10)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s5[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        oriented[name] = (u, ut)
        per = {b: score_fixed(u, zb, y, folds) for b, zb in bases_s5.items()}
        ok = (
            gates["verified"]
            and all(per[b]["pos"] >= 5 for b in bases_s5)
            and all(per[b]["sf25"] >= 19 for b in bases_s5)
            and all(per[b]["t"] >= 3.2 for b in bases_s5)
            and "R1" not in name
        )
        results[name] = {
            "gates": gates,
            "per_base": per,
            "pre_corr_zS5": float(np.corrcoef(vo, z_s5)[0, 1]),
            "corr_R2": float(np.corrcoef(vo, R2_oof)[0, 1]),
            "stage_ok": ok,
            "fold2": per["z_S5"]["folds"][2],
        }
        print(
            f"  {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{gates['leak_col']}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) "
            f"F2Δ={results[name]['fold2']} ρR2={results[name]['corr_R2']:+.3f} "
            f"stage_ok={ok}"
        )
        for b, row in per.items():
            print(
                f"    {b}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
                f"min={row['min']:+.3f} folds={row['folds']} sf={row['sf25']} t={row['t']:.2f}"
            )

    for name, vo, vt in [
        ("M1_P2_v186", M1, M1_t),
        ("M2_P4_v186", M2, M2_t),
        ("M3_peer12_va", M3, M3_t),
        ("M4_peer12_distill", M4, M4_t),
        ("M5_peer_hybrid", M5, M5_t),
    ]:
        eval_sig(name, vo, vt)

    # R1-free combos with U5/U6 only
    for base in ("M1_P2_v186", "M2_P4_v186", "M3_peer12_va", "M4_peer12_distill", "M5_peer_hybrid"):
        ub, utb = oriented[base]
        eval_sig(f"{base}_plus_U5", unitize(ub + u_u5), unitize(utb + ut_u5))
        eval_sig(f"{base}_plus_U6", unitize(ub + u_u6), unitize(utb + ut_u6))
        eval_sig(
            f"{base}_plus_U5_U6",
            unitize(ub + u_u5 + u_u6),
            unitize(utb + ut_u5 + ut_u6),
        )

    staged = []
    for name, r in results.items():
        if r["stage_ok"]:
            u, ut = oriented[name]
            w = r["per_base"]["z_Slot5_twin"]["w"]
            path = SUB / f"sub_v18_9_{name}_w{str(w).replace('.','p')}_NOSUBMIT.csv"
            cand = qlock(L(slot5) + w * ut, lv)
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
            staged.append({"file": path.name, "signal": name, "w": w})
            print("STAGED", path.name)
    if not staged:
        print("NO STAGE")

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "zero_R1": True,
            "zero_logcnt": True,
            "distill_corr": distill_corr,
            "n_peer8": int(X_peer8_va.shape[1]),
            "n_peer12": int(X_peer12_va.shape[1]),
            "n_distill": int(Xd_o.shape[1]),
            "results": results,
            "staged": staged,
            "elapsed_s": time.time() - t0,
            "stage_rule": "R1-free ∧ F1∧F2∧F3 ∧ 5/5 ∧ sf>=19 ∧ t>=3.2 on all 3 Slot5 baselines",
        }
    )
    OUT.joinpath("v18_9.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUT / "v18_9_peers.npz",
        M1_oof=M1,
        M1_test=M1_t,
        M2_oof=M2,
        M2_test=M2_t,
        M3_oof=M3,
        M3_test=M3_t,
        M4_oof=M4,
        M4_test=M4_t,
        M5_oof=M5,
        M5_test=M5_t,
    )
    print("WROTE", OUT / "v18_9.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
