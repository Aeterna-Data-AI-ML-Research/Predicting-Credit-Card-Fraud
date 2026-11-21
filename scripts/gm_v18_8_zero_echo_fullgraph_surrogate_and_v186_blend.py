#!/usr/bin/env python3
"""GM V18.8 — Zero-echo L2FO full-graph peers + V18.6 merchant-key va-only bank.

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
        "F1": 0.94 <= disp <= 1.06,
        "F2": max_leak < 0.035,
        "F3": abs(c_live) < 0.05,
        "verified": (0.94 <= disp <= 1.06) and (max_leak < 0.035) and (abs(c_live) < 0.05),
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


def peer_mean_logcnt(z, gids, m0, n_ref=100000.0):
    z = np.asarray(z, float)
    g = np.asarray(gids, np.int64)
    _, g2 = np.unique(g, return_inverse=True)
    nbin = int(g2.max()) + 1
    cnt = np.bincount(g2, minlength=nbin).astype(np.float64)
    sz = np.bincount(g2, weights=z, minlength=nbin)
    c = cnt[g2]
    peer = (sz[g2] - z) / (c - 1.0 + m0)
    peer = np.where(c <= 1, 0.0, peer)
    logcnt = np.log1p(c / (len(z) / n_ref))
    return peer, logcnt, c


def fold_unitize(oof, test, folds):
    out_o = np.zeros_like(oof)
    for _, va in folds:
        out_o[va] = unitize(oof[va])
    return out_o, unitize(test)


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

    # U5 + R1
    raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
    llr_o = raw_o[:, :, 1] - raw_o[:, :, 0]
    llr_t = raw_t[:, :, :, 1] - raw_t[:, :, :, 0]
    U5_o = L_fold(gini10(col_z(llr_o)), fold_id)
    U5_t = L(np.stack([L(gini10(col_z(llr_t[f]))) for f in range(5)], 0).mean(0))
    oo_u5, ot_u5, _ = frozen_lofo_orth(U5_o, U5_t, Bo10, Bt10, folds, ridge_alpha=1.0)
    u_u5 = unitize(oo_u5)
    if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u_u5[te0]) < roc_auc_score(
        y[te0], z_s5[te0] - 0.01 * u_u5[te0]
    ):
        u_u5, ot_u5 = -u_u5, -ot_u5
    ut_u5 = unitize(ot_u5)

    v180 = np.load(OUT / "v18_0_residuals.npz")
    oo_r1, ot_r1, _ = frozen_lofo_orth(
        v180["R1p_oof"], v180["R1p_test"], Bo10, Bt10, folds, ridge_alpha=10.0
    )
    u_r1 = unitize(oo_r1)
    if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u_r1[te0]) < roc_auc_score(
        y[te0], z_s5[te0] - 0.01 * u_r1[te0]
    ):
        u_r1, ot_r1 = -u_r1, -ot_r1
    ut_r1 = unitize(ot_r1)

    print("=== STEP 1: banks A/B ===", f"t={time.time()-t0:.1f}s")
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
    # category alias = merchant_category (no separate category col)
    cat = mer

    key_order = [
        "k1_amt2",
        "k2_bal2",
        "k3_vel2",
        "k4_mr4",
        "k5_dist2",
        "k6_mer_amt0",
        "k7_mer_profile",
        "k8_cents_mer",
        "k9_amtcents_cat",
        "k10_balcents_cat",
    ]
    key_all = {
        "k1_amt2": codes_from_str(np.round(amt, 2).astype(str)),
        "k2_bal2": codes_from_str(np.round(bal, 2).astype(str)),
        "k3_vel2": codes_from_str(np.round(vel, 2).astype(str)),
        "k4_mr4": codes_from_str(np.round(mr, 4).astype(str)),
        "k5_dist2": codes_from_str(np.round(dist, 2).astype(str)),
        "k6_mer_amt0": codes_from_str(
            (pd.Series(mer) + "|" + pd.Series(np.round(amt, 0).astype(np.int64).astype(str))).to_numpy()
        ),
        "k7_mer_profile": codes_from_str(
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
        "k9_amtcents_cat": codes_from_str(
            (
                pd.Series(((np.round(np.abs(amt) * 100).astype(np.int64) % 100) % 100).astype(str))
                + "|"
                + pd.Series(cat)
            ).to_numpy()
        ),
        "k10_balcents_cat": codes_from_str(
            (
                pd.Series(((np.round(np.abs(bal) * 100).astype(np.int64) % 100) % 100).astype(str))
                + "|"
                + pd.Series(cat)
            ).to_numpy()
        ),
    }

    # X_broad for B_h (~24 cols): nums + ranks + bools + cat codes — NO fine peer keys
    def build_X_broad():
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
        ):
            cols_tr.append(train[c].to_numpy(float))
            cols_te.append(test[c].to_numpy(float))
        for c in (
            "is_foreign_transaction",
            "is_new_merchant",
            "used_vpn",
            "ip_country_mismatch",
            "billing_shipping_mismatch",
        ):
            cols_tr.append(train[c].to_numpy(float))
            cols_te.append(test[c].to_numpy(float))
        for c in ("merchant_category", "channel", "device_type", "card_type", "auth_method"):
            codes, _ = pd.factorize(pd.concat([train[c], test[c]], ignore_index=True))
            cols_tr.append(codes[:n_tr].astype(float))
            cols_te.append(codes[n_tr:].astype(float))
        # drain + global ranks
        atr, ate = train["amount_usd"].to_numpy(float), test["amount_usd"].to_numpy(float)
        btr, bte = train["account_balance_usd"].to_numpy(float), test["account_balance_usd"].to_numpy(
            float
        )
        cols_tr.append(atr / (np.abs(btr) + 10.0))
        cols_te.append(ate / (np.abs(bte) + 10.0))
        ug_a = U(amt)
        ug_b = U(bal)
        cols_tr.append(ug_a[:n_tr] - ug_b[:n_tr])
        cols_te.append(ug_a[n_tr:] - ug_b[n_tr:])
        ug_v = U(vel)
        ug_h = U(hrs)
        cols_tr.append(ug_v[:n_tr] - ug_h[:n_tr])
        cols_te.append(ug_v[n_tr:] - ug_h[n_tr:])
        return np.column_stack(cols_tr).astype(np.float32), np.column_stack(cols_te).astype(
            np.float32
        )

    Xb_tr, Xb_te = build_X_broad()
    Xb_all = np.vstack([Xb_tr, Xb_te])
    print(f"  X_broad={Xb_tr.shape[1]} cols t={time.time()-t0:.1f}s")

    # Train 5 single-fold classifiers B_h on (X[h], y[h]) only — ZERO other folds' y
    B_pred = np.zeros((5, n_all), dtype=np.float64)  # B_h predictions on all rows
    for h in range(5):
        mh = fold_id == h
        model = lgb.LGBMClassifier(
            objective="binary",
            num_leaves=31,
            n_estimators=120,
            learning_rate=0.05,
            min_child_samples=200,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42 + h,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(Xb_tr[mh], y[mh])
        # predict_proba fraud class on train+test
        B_pred[h, :n_tr] = model.predict_proba(Xb_tr)[:, 1]
        B_pred[h, n_tr:] = model.predict_proba(Xb_te)[:, 1]
        print(f"  B_{h} trained on fold{h} n={mh.sum()} t={time.time()-t0:.1f}s")

    # For each validation fold f, build L2FO score s_f_all (999944)
    # and Bank B peer features on va + test
    # Bank A: va-only peers
    # Feature layout: per key m5, m20, logcnt (10*3=30) + diffs among top6 m20 (15) = 45
    n_keys = 10
    n_feat = n_keys * 3 + 15
    XA_oof = np.zeros((n_tr, n_feat), np.float32)
    XA_te = np.zeros((n_te, n_feat), np.float32)
    XB_oof = np.zeros((n_tr, n_feat), np.float32)
    XB_te = np.zeros((n_te, n_feat), np.float32)

    cnt_table = {}
    for kn in key_order:
        # va-only mean cnt
        cnts_va, cnts_full = [], []
        for f, (_, va) in enumerate(folds):
            _, _, c = peer_mean_logcnt(unitize(z_s5[va]), key_all[kn][va], m0=5.0)
            cnts_va.append(c.mean())
        # fullgraph using s_0 as proxy for cnt
        s0 = np.zeros(n_all)
        # rough: use mean of B_h h!=0 for cnt structure
        for h in range(1, 5):
            s0 += B_pred[h]
        s0 = unitize(s0 / 4.0)
        _, _, cfull = peer_mean_logcnt(s0, key_all[kn], m0=5.0)
        cnt_table[kn] = {
            "mean_cnt_va": float(np.mean(cnts_va)),
            "mean_cnt_fullgraph": float(cfull.mean()),
            "ratio": float(cfull.mean() / (np.mean(cnts_va) + 1e-12)),
        }
        print(
            f"  {kn}: cnt_va={cnt_table[kn]['mean_cnt_va']:.1f} "
            f"cnt_full={cnt_table[kn]['mean_cnt_fullgraph']:.1f} "
            f"×{cnt_table[kn]['ratio']:.2f}"
        )

    top6 = key_order[:6]

    def fill_bank(Xo, Xt, score_fn_oof, score_fn_te, m0_va=(5.0, 20.0), scale_te=True):
        """score_fn_oof(f, va_idx) -> z vector length |va|; score_fn_te(f) -> z length n_te
        For fullgraph, score_fn returns scores for ALL rows and we peer on all.
        """
        m15_o, m15_t = {}, {}
        col = 0
        for kn in key_order:
            o5 = np.zeros(n_tr)
            o20 = np.zeros(n_tr)
            olog = np.zeros(n_tr)
            # accumulate test across folds then average
            t5_acc, t20_acc, tlog_acc = [], [], []
            for f, (tr, va) in enumerate(folds):
                if score_fn_oof.__name__ == "va_scores":
                    z_va = score_fn_oof(f, va)
                    p5, l5, _ = peer_mean_logcnt(z_va, key_all[kn][va], m0_va[0])
                    p20, _, _ = peer_mean_logcnt(z_va, key_all[kn][va], m0_va[1])
                    o5[va], o20[va], olog[va] = p5, p20, l5
                    z_te = score_fn_te(f)
                    sc = SCALE if scale_te else 1.0
                    p5t, l5t, _ = peer_mean_logcnt(
                        z_te, key_all[kn][n_tr:], m0=m0_va[0] * sc
                    )
                    p20t, _, _ = peer_mean_logcnt(
                        z_te, key_all[kn][n_tr:], m0=m0_va[1] * sc
                    )
                    t5_acc.append(p5t)
                    t20_acc.append(p20t)
                    tlog_acc.append(l5t)
                else:
                    # fullgraph: score all rows for fold f, peer on all, take va/test slices
                    z_all = score_fn_oof(f)  # length n_all
                    p5, l5, _ = peer_mean_logcnt(z_all, key_all[kn], m0=m0_va[0])
                    p20, _, _ = peer_mean_logcnt(z_all, key_all[kn], m0=m0_va[1])
                    o5[va] = p5[va]
                    o20[va] = p20[va]
                    olog[va] = l5[va]
                    t5_acc.append(p5[n_tr:])
                    t20_acc.append(p20[n_tr:])
                    tlog_acc.append(l5[n_tr:])
            t5 = np.mean(np.stack(t5_acc, 0), 0)
            t20 = np.mean(np.stack(t20_acc, 0), 0)
            tlog = np.mean(np.stack(tlog_acc, 0), 0)
            o5, t5 = fold_unitize(o5, t5, folds)
            o20, t20 = fold_unitize(o20, t20, folds)
            olog, tlog = fold_unitize(olog, tlog, folds)
            Xo[:, col] = o5
            Xt[:, col] = t5
            col += 1
            Xo[:, col] = o20
            Xt[:, col] = t20
            m15_o[kn] = o20
            m15_t[kn] = t20
            col += 1
            Xo[:, col] = olog
            Xt[:, col] = tlog
            col += 1
        for a, b in combinations(range(6), 2):
            ka, kb = top6[a], top6[b]
            d_o = m15_o[ka] - m15_o[kb]
            d_t = m15_t[ka] - m15_t[kb]
            d_o, d_t = fold_unitize(d_o, d_t, folds)
            Xo[:, col] = d_o
            Xt[:, col] = d_t
            col += 1
        assert col == n_feat

    def va_scores(f, va):
        return unitize(z_s5[va])

    def va_scores_te(f):
        return unitize(z_s5_t_sym)

    def l2fo_scores_all(f):
        """999944 scores for fold f: L2FO B_h averages + blend z_S5 on va."""
        s = np.zeros(n_all)
        # for each train row in fold g:
        for g in range(5):
            mg = np.where(fold_id == g)[0]
            if g == f:
                # va: mean B_h h!=f, blend with z_S5_oof
                hs = [h for h in range(5) if h != f]
                pred = np.mean(B_pred[hs][:, mg], axis=0)
                s[mg] = 0.5 * unitize(pred) + 0.5 * unitize(z_s5[mg])
            else:
                # tr fold g: mean B_h h not in {f,g}
                hs = [h for h in range(5) if h not in (f, g)]
                pred = np.mean(B_pred[hs][:, mg], axis=0)
                s[mg] = unitize(pred) if len(mg) else pred
        # test: mean B_h h!=f
        hs = [h for h in range(5) if h != f]
        s[n_tr:] = unitize(np.mean(B_pred[hs][:, n_tr:], axis=0))
        return unitize(s)

    # monkeypatch name for fill_bank branch
    va_scores.__name__ = "va_scores"
    l2fo_scores_all.__name__ = "l2fo_scores_all"

    fill_bank(XA_oof, XA_te, va_scores, va_scores_te, m0_va=(5.0, 20.0), scale_te=True)
    print(f"  Bank A filled t={time.time()-t0:.1f}s")
    fill_bank(XB_oof, XB_te, l2fo_scores_all, None, m0_va=(5.0, 20.0), scale_te=False)
    print(f"  Bank B filled t={time.time()-t0:.1f}s")

    # ===== STEP 2: fit V1/V2/V3 =====
    print("=== STEP 2: fit V1/V2/V3 ===")
    seeds = [42, 2026, 7, 100, 999]
    lgb_p = dict(
        objective="regression",
        num_leaves=5,
        min_child_samples=600,
        n_estimators=40,
        learning_rate=0.03,
        reg_lambda=20.0,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=-1,
        n_jobs=-1,
    )

    def fit_ridge_lgb(Xo, Xt):
        # ridge on mean cols (every 3rd starting 0,1 — m5,m20; skip logcnt idx%3==2 among first 30)
        ridge_idx = [i for i in range(30) if i % 3 != 2]
        Ro, Rt = Xo[:, ridge_idx], Xt[:, ridge_idx]
        R = np.zeros(n_tr)
        Rte = []
        for f, (tr, va) in enumerate(folds):
            sw = np.sqrt(h5[f][tr])
            lr = Ridge(alpha=200.0, fit_intercept=True)
            lr.fit(Ro[tr] * sw[:, None], g5[f][tr] * sw)
            R[va] = lr.predict(Ro[va])
            Rte.append(lr.predict(Rt))
        R_t = np.mean(np.stack(Rte, 0), 0)
        L_o = np.zeros(n_tr)
        Lte = []
        for seed in seeds:
            oo = np.zeros(n_tr)
            tef = []
            for f, (tr, va) in enumerate(folds):
                model = lgb.LGBMRegressor(**lgb_p, random_state=seed)
                model.fit(Xo[tr], g5[f][tr], sample_weight=h5[f][tr])
                oo[va] = model.predict(Xo[va])
                tef.append(model.predict(Xt))
            L_o += oo
            Lte.append(np.mean(np.stack(tef, 0), 0))
        L_o /= len(seeds)
        L_t = np.mean(np.stack(Lte, 0), 0)
        blend = unitize(0.35 * unitize(R) + 0.65 * unitize(L_o))
        blend_t = unitize(0.35 * unitize(R_t) + 0.65 * unitize(L_t))
        return blend, blend_t

    V1, V1_t = fit_ridge_lgb(XA_oof, XA_te)
    print(f"  V1 done t={time.time()-t0:.1f}s")
    V2, V2_t = fit_ridge_lgb(XB_oof, XB_te)
    print(f"  V2 done t={time.time()-t0:.1f}s")
    V3 = unitize(0.5 * unitize(V1) + 0.5 * unitize(V2))
    V3_t = unitize(0.5 * unitize(V1_t) + 0.5 * unitize(V2_t))

    # ===== STEP 3 =====
    print("=== STEP 3: scorecard ===")
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
        )
        results[name] = {
            "gates": gates,
            "per_base": per,
            "pre_corr_zS5": float(np.corrcoef(vo, z_s5)[0, 1]),
            "corr_R2": float(np.corrcoef(vo, R2_oof)[0, 1]),
            "stage_ok": ok,
            "fold2_zS5": per["z_S5"]["folds"][2] if len(per["z_S5"]["folds"]) > 2 else None,
        }
        print(
            f"  {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{gates['leak_col']}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) "
            f"F2Δ={results[name]['fold2_zS5']} stage_ok={ok}"
        )
        for b, row in per.items():
            print(
                f"    {b}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
                f"min={row['min']:+.3f} folds={row['folds']} sf={row['sf25']} t={row['t']:.2f}"
            )

    for name, vo, vt in [
        ("V1_va_strict_repro", V1, V1_t),
        ("V2_l2fo_fullgraph", V2, V2_t),
        ("V3_dual_peer_strict", V3, V3_t),
    ]:
        eval_sig(name, vo, vt)

    # combos
    for base_name in ("V1_va_strict_repro", "V2_l2fo_fullgraph", "V3_dual_peer_strict"):
        ub, utb = oriented[base_name]
        for wr in (0.15, 0.25):
            name = f"{base_name}_plus_R1_{str(wr).replace('.','p')}"
            eval_sig(name, unitize(ub + wr * u_r1), unitize(utb + wr * ut_r1))
        name = f"{base_name}_plus_U5"
        eval_sig(name, unitize(ub + u_u5), unitize(utb + ut_u5))
        name = f"{base_name}_plus_U5_R1_0p25"
        eval_sig(
            name,
            unitize(ub + u_u5 + 0.25 * u_r1),
            unitize(utb + ut_u5 + 0.25 * ut_r1),
        )

    staged = []
    for name, r in results.items():
        if r["stage_ok"]:
            u, ut = oriented[name]
            w = r["per_base"]["z_Slot5_twin"]["w"]
            path = SUB / f"sub_v18_8_{name}_w{str(w).replace('.','p')}_NOSUBMIT.csv"
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
            "cnt_table": cnt_table,
            "n_feat": n_feat,
            "zero_echo_claim": "B_h trained only on fold h; L2FO excludes f and peer fold g",
            "results": results,
            "staged": staged,
            "elapsed_s": time.time() - t0,
            "stage_rule": "F1∧F2∧F3 ∧ 5/5 ∧ sf>=19 ∧ t>=3.2 on all 3 Slot5 baselines",
        }
    )
    OUT.joinpath("v18_8.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUT / "v18_8_peers.npz",
        V1_oof=V1,
        V1_test=V1_t,
        V2_oof=V2,
        V2_test=V2_t,
        V3_oof=V3,
        V3_test=V3_t,
        B_pred=B_pred,
    )
    print("WROTE", OUT / "v18_8.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
