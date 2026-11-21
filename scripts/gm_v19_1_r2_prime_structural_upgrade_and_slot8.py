#!/usr/bin/env python3
"""GM V19.1 — Replace 1-seed R2_v178 with multi-arch R2_Prime; Slot8 only if confident.

0 GPU pods. Research-first: submit IFF head-to-head vs Slot5 clears hard gates.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import ks_2samp, rankdata, spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
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
OUT_CSV = SUB / "sub_v19_1_LIVE_slot8_R2_Prime_multi_arch_upgrade.csv"
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W_R2_SLOT5 = 0.020
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
SEEDS = [42, 2026, 7, 100, 999]
LGB_SCALES = [
    dict(num_leaves=4, min_child_samples=800, n_estimators=45, learning_rate=0.03,
         colsample_bytree=0.75, subsample=0.8, reg_lambda=20.0),
    dict(num_leaves=6, min_child_samples=500, n_estimators=45, learning_rate=0.03,
         colsample_bytree=0.80, subsample=0.8, reg_lambda=10.0),
    dict(num_leaves=8, min_child_samples=400, n_estimators=40, learning_rate=0.025,
         colsample_bytree=0.70, subsample=0.8, reg_lambda=25.0),
]


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
        "pass_disp": 0.90 <= disp <= 1.10,
        "pass_ks": ks <= 0.015,
        "pass": (0.90 <= disp <= 1.10) and (ks <= 0.015),
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


def h2h_vs_slot5(u_prime, u_r2, z_base, y, folds, w, w_r2=W_R2_SLOT5):
    """ΔAUC of (base + w*Prime) vs (base + 0.020*R2) — exact Slot5 replacement test."""
    up, ur = unitize(u_prime), unitize(u_r2)
    d = []
    for _, te in folds:
        s5 = z_base[te] + w_r2 * ur[te]
        sp = z_base[te] + w * up[te]
        d.append((roc_auc_score(y[te], sp) - roc_auc_score(y[te], s5)) * 1e5)
    deltas = []
    for _, te in folds:
        yf, zb = y[te], z_base[te]
        urf, upf = ur[te], up[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(yf)), yf
        ):
            s5 = zb[ste] + w_r2 * urf[ste]
            sp = zb[ste] + w * upf[ste]
            deltas.append((roc_auc_score(yf[ste], sp) - roc_auc_score(yf[ste], s5)) * 1e5)
    d25 = np.asarray(deltas, float)
    mean = float(d25.mean())
    se = float(d25.std(ddof=1) / np.sqrt(len(d25)))
    return {
        "w": float(w),
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


def codes_from_str(arr):
    c, _ = pd.factorize(arr, sort=False)
    return c.astype(np.int64)


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
    z_s3 = z_ath + 0.01045 * dE_o + 0.0055 * unitize(d1)
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
    bases_s3 = {"z_S3": z_s3, "z_Slot3_twin": z_slot3_twin, "z_NNLS_twin": z_nnls}

    # ===== STEP 1: reconstruct R2_v178 + X_tab =====
    print("=== STEP 1: R2_v178 anchor + X_tab39/51 ===", f"t={time.time()-t0:.1f}s")
    res = np.load(OUT / "v17_8_residuals.npz")
    R2_oof_raw, R2_test_raw = res["R2_oof"], res["R2_test"]
    oo_r2, ot_r2, _ = frozen_lofo_orth(R2_oof_raw, R2_test_raw, Bo9, Bt9, folds, ridge_alpha=10.0)
    u_r2 = unitize(oo_r2)
    if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u_r2[te0]) < roc_auc_score(
        y[te0], z_s3[te0] - 0.01 * u_r2[te0]
    ):
        u_r2, ot_r2 = -u_r2, -ot_r2
    ut_r2 = unitize(ot_r2)

    # verify Slot5 reconstruction
    cand_s5 = qlock(L(slot3) + W_R2_SLOT5 * ut_r2, lv)
    rho_s5 = one_minus_rho_e4(cand_s5, slot5)
    print(f"  Slot5 rebuild 1-ρ vs LIVE Slot5 CSV = {rho_s5:.6e} e-4 (need <1e-5 = 1e-9 abs)")
    # 1-ρ < 1e-9 means rho > 1 - 1e-9; in e-4 units that's < 1e-5
    assert rho_s5 < 1e-5, f"Slot5 rebuild mismatch: 1-ρ={rho_s5}e-4"

    z_s5 = z_s3 + W_R2_SLOT5 * u_r2
    z_slot5_twin = z_slot3_twin + W_R2_SLOT5 * u_r2
    z_nnls_s5 = z_nnls + W_R2_SLOT5 * u_r2
    bases_s5 = {"z_S5": z_s5, "z_Slot5_twin": z_slot5_twin, "z_NNLS_S5": z_nnls_s5}
    # Slot3 bases used for H2H replacement math
    bases_s3_for_h2h = bases_s3

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
            print(f"  orig20k hits mean_tr={orig_cnt_tr.mean():.3f}")

    for cat in ("merchant_category", "channel", "device_type"):
        codes, _ = pd.factorize(pd.concat([train[cat], test[cat]], ignore_index=True))
        keys_tr[f"{cat}_x_amtones"] = codes[:n_tr] * 10 + keys_tr["amount_usd_ones"]
        keys_te[f"{cat}_x_amtones"] = codes[n_tr:] * 10 + keys_te["amount_usd_ones"]

    key_names = list(keys_tr.keys())

    def build_X39(split="train"):
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

    X39_tr, X39_te = build_X39("train"), build_X39("test")
    print(f"  X_tab39/actual={X39_tr.shape[1]} keys={len(key_names)}")

    # L4.6C deterministic extras
    flag_cols = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    flags_tr = np.column_stack([train[c].to_numpy(float) for c in flag_cols])
    flags_te = np.column_stack([test[c].to_numpy(float) for c in flag_cols])
    flag_mask_tr = np.packbits(flags_tr.astype(np.uint8), axis=1, bitorder="little").ravel().astype(
        np.float64
    )
    # packbits on 6 bits -> need pad to 8; use manual bitmask instead
    flag_mask_tr = (flags_tr.astype(np.int64) * (2 ** np.arange(6))).sum(1).astype(np.float64)
    flag_mask_te = (flags_te.astype(np.int64) * (2 ** np.arange(6))).sum(1).astype(np.float64)
    flag_count_tr, flag_count_te = flags_tr.sum(1), flags_te.sum(1)
    ch_dev_tr = codes_from_str(
        (train["channel"].astype(str) + "|" + train["device_type"].astype(str)).to_numpy()
    ).astype(float)
    ch_dev_te = codes_from_str(
        (test["channel"].astype(str) + "|" + test["device_type"].astype(str)).to_numpy()
    )
    # joint factorize for train/test consistency
    ch_dev_all = codes_from_str(
        np.concatenate(
            [
                (train["channel"].astype(str) + "|" + train["device_type"].astype(str)).to_numpy(),
                (test["channel"].astype(str) + "|" + test["device_type"].astype(str)).to_numpy(),
            ]
        )
    ).astype(float)
    ch_dev_tr, ch_dev_te = ch_dev_all[:n_tr], ch_dev_all[n_tr:]

    ip_tr = train["ip_country_mismatch"].to_numpy(float)
    ip_te = test["ip_country_mismatch"].to_numpy(float)
    for_tr = train["is_foreign_transaction"].to_numpy(float)
    for_te = test["is_foreign_transaction"].to_numpy(float)
    bill_tr = train["billing_shipping_mismatch"].to_numpy(float)
    bill_te = test["billing_shipping_mismatch"].to_numpy(float)

    def joint_q(col, nbin=250):
        v = np.concatenate([train[col].to_numpy(float), test[col].to_numpy(float)])
        r = rankdata(v, method="average")
        q = np.clip((r / (len(v) + 1e-12) * nbin).astype(np.int64), 0, nbin - 1)
        return q[:n_tr].astype(float), q[n_tr:].astype(float)

    mrisk_q250_tr, mrisk_q250_te = joint_q("merchant_risk_score", 250)
    vel_q250_tr, vel_q250_te = joint_q("velocity_score", 250)
    mrisk_step10_tr = (np.round(train["merchant_risk_score"].to_numpy(float) * 10).astype(np.int64) % 10).astype(
        float
    )
    mrisk_step10_te = (np.round(test["merchant_risk_score"].to_numpy(float) * 10).astype(np.int64) % 10).astype(
        float
    )
    vel_step10_tr = (np.round(train["velocity_score"].to_numpy(float) * 10).astype(np.int64) % 10).astype(float)
    vel_step10_te = (np.round(test["velocity_score"].to_numpy(float) * 10).astype(np.int64) % 10).astype(float)
    dist_cents_tr = (np.round(train["distance_from_home_km"].to_numpy(float) * 100).astype(np.int64) % 100).astype(
        float
    )
    dist_cents_te = (np.round(test["distance_from_home_km"].to_numpy(float) * 100).astype(np.int64) % 100).astype(
        float
    )
    hours_tenths_tr = (np.round(train["hours_since_last_txn"].to_numpy(float) * 10).astype(np.int64) % 10).astype(
        float
    )
    hours_tenths_te = (np.round(test["hours_since_last_txn"].to_numpy(float) * 10).astype(np.int64) % 10).astype(
        float
    )

    Xl46_tr = np.column_stack(
        [
            flag_mask_tr,
            flag_count_tr,
            ch_dev_tr,
            flag_mask_tr * 2 + ip_tr,
            flag_mask_tr * 2 + for_tr,
            flag_mask_tr * 2 + bill_tr,
            mrisk_q250_tr,
            mrisk_step10_tr,
            vel_q250_tr,
            vel_step10_tr,
            dist_cents_tr,
            hours_tenths_tr,
        ]
    ).astype(np.float32)
    Xl46_te = np.column_stack(
        [
            flag_mask_te,
            flag_count_te,
            ch_dev_te,
            flag_mask_te * 2 + ip_te,
            flag_mask_te * 2 + for_te,
            flag_mask_te * 2 + bill_te,
            mrisk_q250_te,
            mrisk_step10_te,
            vel_q250_te,
            vel_step10_te,
            dist_cents_te,
            hours_tenths_te,
        ]
    ).astype(np.float32)
    X51_tr = np.column_stack([X39_tr, Xl46_tr]).astype(np.float32)
    X51_te = np.column_stack([X39_te, Xl46_te]).astype(np.float32)
    print(f"  X_tab51={X51_tr.shape[1]} (39+12) t={time.time()-t0:.1f}s")

    # Platt g,h per fold on Slot3
    ab, g5, h5 = [], [], []
    for tr, _ in folds:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s3[tr].reshape(-1, 1), y[tr])
        a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
        ab.append((a, b))
        p = expit(a * z_s3 + b)
        g5.append(y - p)
        h5.append(np.clip(p * (1 - p), 1e-6, None))

    # ===== STEP 2: fit families =====
    print("=== STEP 2: multi-arch bags ===", f"t={time.time()-t0:.1f}s")

    def fit_lgb_bag(Xtr, Xte, scales, seeds, tag):
        oof_acc, te_acc = [], []
        for si, scale in enumerate(scales):
            for seed in seeds:
                oo = np.zeros(n_tr)
                tef = []
                for f, (tr, va) in enumerate(folds):
                    model = lgb.LGBMRegressor(
                        objective="regression",
                        verbosity=-1,
                        n_jobs=-1,
                        random_state=seed + 17 * si,
                        **scale,
                    )
                    model.fit(Xtr[tr], g5[f][tr], sample_weight=h5[f][tr])
                    oo[va] = model.predict(Xtr[va])
                    tef.append(model.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        oof = np.mean(np.stack(oof_acc, 0), 0)
        te = np.mean(np.stack(te_acc, 0), 0)
        print(f"  {tag}: n_models={len(oof_acc)} done t={time.time()-t0:.1f}s")
        return oof, te

    def fit_cat_bag(Xtr, Xte, tag, depths=(3, 4)):
        oof_acc, te_acc = [], []
        for depth in depths:
            for seed in SEEDS:
                oo = np.zeros(n_tr)
                tef = []
                for f, (tr, va) in enumerate(folds):
                    model = CatBoostRegressor(
                        loss_function="RMSE",
                        depth=depth,
                        l2_leaf_reg=25,
                        iterations=100,
                        learning_rate=0.035,
                        random_seed=seed,
                        verbose=False,
                        thread_count=4,
                        bootstrap_type="Bernoulli",
                        subsample=0.8,
                    )
                    model.fit(Xtr[tr], g5[f][tr], sample_weight=h5[f][tr])
                    oo[va] = model.predict(Xtr[va])
                    tef.append(model.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        oof = np.mean(np.stack(oof_acc, 0), 0)
        te = np.mean(np.stack(te_acc, 0), 0)
        print(f"  {tag}: n_models={len(oof_acc)} done t={time.time()-t0:.1f}s")
        return oof, te

    def fit_hgb_bag(Xtr, Xte, tag):
        oof_acc, te_acc = [], []
        for max_leaf in (5, 7):
            for seed in SEEDS:
                oo = np.zeros(n_tr)
                tef = []
                for f, (tr, va) in enumerate(folds):
                    model = HistGradientBoostingRegressor(
                        max_leaf_nodes=max_leaf,
                        min_samples_leaf=600,
                        max_iter=45,
                        learning_rate=0.03,
                        l2_regularization=20.0,
                        random_state=seed,
                    )
                    model.fit(Xtr[tr], g5[f][tr], sample_weight=h5[f][tr])
                    oo[va] = model.predict(Xtr[va])
                    tef.append(model.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        oof = np.mean(np.stack(oof_acc, 0), 0)
        te = np.mean(np.stack(te_acc, 0), 0)
        print(f"  {tag}: n_models={len(oof_acc)} done t={time.time()-t0:.1f}s")
        return oof, te

    F_lgb39_o, F_lgb39_t = fit_lgb_bag(X39_tr, X39_te, LGB_SCALES, SEEDS, "F_lgb_ms39")
    F_lgb51_o, F_lgb51_t = fit_lgb_bag(X51_tr, X51_te, LGB_SCALES, SEEDS, "F_lgb_ms51")
    # Cat: 5 seeds × depth3 on X39 + 5 seeds × depth4 on X51 = 10
    cat39_o, cat39_t = fit_cat_bag(X39_tr, X39_te, "F_cat39", depths=(3,))
    cat51_o, cat51_t = fit_cat_bag(X51_tr, X51_te, "F_cat51", depths=(4,))
    F_cat_o = 0.5 * (cat39_o + cat51_o)
    F_cat_t = 0.5 * (cat39_t + cat51_t)
    F_hgb_o, F_hgb_t = fit_hgb_bag(X51_tr, X51_te, "F_hgb_bag")

    # G1e audit
    P_BPE_o = 0.75 * L(ms_o) + 0.25 * L(g26_o)
    P_QLM_o = 0.40 * L(XO["megayak_5f_lm"]) + 0.30 * L(l7f_o) + 0.30 * L(XO["megayak_5f_lm"])
    # use LM+L7F only for QLM side (no s2026 required for G1e V17.4 style)
    P_QLM_o = 0.5 * L(XO["megayak_5f_lm"]) + 0.5 * L(l7f_o)
    P_BPE_t = 0.75 * L(ms_t) + 0.25 * L(g26_t)
    P_QLM_t = 0.5 * L(XT["megayak_5f_lm"]) + 0.5 * L(l7f_t)
    d_all_o = unitize(L_fold(P_BPE_o - P_QLM_o, fold_id))
    d_all_t = unitize(L(P_BPE_t - P_QLM_t))
    G1e_o = d_all_o * unitize(z_s3)
    G1e_t = d_all_t * unitize(L(slot3))

    families_raw = {
        "R2_v178_orig": (R2_oof_raw, R2_test_raw),
        "F_lgb_ms39": (F_lgb39_o, F_lgb39_t),
        "F_lgb_ms51": (F_lgb51_o, F_lgb51_t),
        "F_cat_obliv": (F_cat_o, F_cat_t),
        "F_hgb_bag": (F_hgb_o, F_hgb_t),
        "G1e": (G1e_o, G1e_t),
    }

    print("=== Individual family audit ===")
    oriented = {}
    family_audit = {}
    clean = []
    for name, (vo, vt) in families_raw.items():
        # pre-orth raw_disp on unitized preds (pre-PCHIP)
        rd_raw = raw_disp_ks(vo, vt)
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=10.0)
        rd_orth = raw_disp_ks(oo, ot)
        gates = forensic_gates(oo, ot, Bt9, live_resid, bn=BN9, f2_thr=0.035, f3_thr=0.05)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s3[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        oriented[name] = (u, ut)
        per020 = {b: score_at_w(u, zb, y, folds, 0.020) for b, zb in bases_s3.items()}
        per025 = {b: score_at_w(u, zb, y, folds, 0.025) for b, zb in bases_s3.items()}
        # quarantine: use orth raw_disp (matches Slot5 R2 reporting) + allow KS soft if disp ok
        # R2_v178 had raw_disp=0.990; KS gate secondary if disp in [0.92,1.08]
        disp_ok = 0.90 <= rd_orth["raw_disp"] <= 1.10
        ks_hard = rd_orth["raw_KS"] <= 0.015
        # for deterministic tabular, KS can be slightly high pre-PCHIP; require disp + F2
        fam_ok = (
            disp_ok
            and gates["F2"]
            and (ks_hard or (0.92 <= rd_orth["raw_disp"] <= 1.08 and name != "G1e"))
            and per020["z_S3"]["pos"] >= 4
        )
        # G1e stricter: must pass KS and raw_disp [0.92,1.08]
        if name == "G1e":
            fam_ok = (
                0.92 <= rd_orth["raw_disp"] <= 1.08
                and ks_hard
                and gates["F2"]
                and all(per020[b]["pos"] >= 4 for b in bases_s3)
            )
        family_audit[name] = {
            "raw_pre": rd_raw,
            "raw_orth": rd_orth,
            "gates": gates,
            "per_w020": per020,
            "per_w025": per025,
            "fam_ok": fam_ok,
        }
        print(
            f"  {name}: disp={rd_orth['raw_disp']:.3f} KS={rd_orth['raw_KS']:.4f} "
            f"F2={gates['max_basis_leak']:.4f} F3={gates['corr_live_resid']:+.4f} "
            f"ok={fam_ok} | S3@{0.02} {per020['z_S3']['pos']}/5 "
            f"m={per020['z_S3']['mean']:+.2f} min={per020['z_S3']['min']:+.2f} "
            f"sf{per020['z_S3']['sf25']} t={per020['z_S3']['t']:.2f} | "
            f"twin m={per020['z_Slot3_twin']['mean']:+.2f} min={per020['z_Slot3_twin']['min']:+.2f}"
        )
        if fam_ok and name != "R2_v178_orig":
            clean.append(name)

    # ===== STEP 3: blend R2_Prime + H2H =====
    print("=== STEP 3: R2_Prime blend + H2H vs Slot5 ===", f"t={time.time()-t0:.1f}s")
    # prefer upgraded families; always allow F_lgb_ms39 as V17.8 multi-seed upgrade
    blend_grid = []
    # weight configs
    for w_cat in (0.0, 0.20, 0.35):
        for w_hgb in (0.0, 0.15, 0.25):
            for w_g1e in (0.0, 0.15):
                if w_g1e > 0 and "G1e" not in clean:
                    continue
                rem = 1.0 - w_cat - w_hgb - w_g1e
                if rem < 0.2:
                    continue
                # split rem between lgb39 and lgb51 if both clean else all to available
                has39 = "F_lgb_ms39" in clean
                has51 = "F_lgb_ms51" in clean
                hascat = "F_cat_obliv" in clean
                hashgb = "F_hgb_bag" in clean
                if w_cat > 0 and not hascat:
                    continue
                if w_hgb > 0 and not hashgb:
                    continue
                if not (has39 or has51):
                    continue
                if has39 and has51:
                    for w39_frac in (0.35, 0.50, 0.65):
                        w39 = rem * w39_frac
                        w51 = rem - w39
                        blend_grid.append(
                            {
                                "w_lgb39": w39,
                                "w_lgb51": w51,
                                "w_cat": w_cat if hascat else 0.0,
                                "w_hgb": w_hgb if hashgb else 0.0,
                                "w_g1e": w_g1e,
                            }
                        )
                elif has51:
                    blend_grid.append(
                        {
                            "w_lgb39": 0.0,
                            "w_lgb51": rem,
                            "w_cat": w_cat if hascat else 0.0,
                            "w_hgb": w_hgb if hashgb else 0.0,
                            "w_g1e": w_g1e,
                        }
                    )
                else:
                    blend_grid.append(
                        {
                            "w_lgb39": rem,
                            "w_lgb51": 0.0,
                            "w_cat": w_cat if hascat else 0.0,
                            "w_hgb": w_hgb if hashgb else 0.0,
                            "w_g1e": w_g1e,
                        }
                    )

    # dedupe
    seen = set()
    uniq_grid = []
    for w in blend_grid:
        key = tuple(round(w[k], 4) for k in ("w_lgb39", "w_lgb51", "w_cat", "w_hgb", "w_g1e"))
        if key not in seen:
            seen.add(key)
            uniq_grid.append(w)
    print(f"  blend grid size={len(uniq_grid)} clean={clean}")

    def make_prime(weights):
        parts_o, parts_t, ww = [], [], []
        mapping = {
            "w_lgb39": "F_lgb_ms39",
            "w_lgb51": "F_lgb_ms51",
            "w_cat": "F_cat_obliv",
            "w_hgb": "F_hgb_bag",
            "w_g1e": "G1e",
        }
        for wk, fam in mapping.items():
            w = weights.get(wk, 0.0)
            if w > 0 and fam in oriented:
                parts_o.append(w * oriented[fam][0])
                parts_t.append(w * oriented[fam][1])
                ww.append(w)
        if not parts_o:
            return None, None
        return unitize(sum(parts_o)), unitize(sum(parts_t))

    best_blend = None
    blend_rows = []
    for weights in uniq_grid:
        u_p, ut_p = make_prime(weights)
        if u_p is None:
            continue
        rd = raw_disp_ks(u_p, ut_p)
        # re-orth blend vs Bo9 for forensic
        oo, ot, _ = frozen_lofo_orth(u_p, ut_p, Bo9, Bt9, folds, ridge_alpha=10.0)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s3[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        rd_o = raw_disp_ks(u, ut)
        gates = forensic_gates(u, ut, Bt9, live_resid, bn=BN9, f2_thr=0.035, f3_thr=0.045)
        # score vs Slot3 at 0.020 and H2H vs Slot5 at several w
        vs3 = {b: score_at_w(u, zb, y, folds, 0.020) for b, zb in bases_s3.items()}
        h2h = {}
        for w in (0.020, 0.022, 0.024, 0.026, 0.028):
            h2h[f"{w:.3f}"] = {
                b: h2h_vs_slot5(u, u_r2, zb, y, folds, w)
                for b, zb in bases_s3_for_h2h.items()
            }
        # pick best w: require all 3 H2H mean>0 and min>=0
        best_w_local = None
        best_score = -1e18
        for wk, per in h2h.items():
            if all(per[b]["mean"] > 0 and per[b]["min"] >= 0 and per[b]["pos"] >= 4 for b in per):
                sc = float(np.mean([per[b]["mean"] for b in per])) + 0.5 * float(
                    np.mean([per[b]["min"] for b in per])
                )
                if sc > best_score:
                    best_score, best_w_local = sc, float(wk)
        row = {
            "weights": weights,
            "raw_disp": rd_o,
            "gates": gates,
            "vs_slot3_w020": vs3,
            "h2h": h2h,
            "best_w": best_w_local,
            "best_score": best_score if best_w_local else None,
        }
        blend_rows.append(row)
        if best_w_local is not None:
            if best_blend is None or best_score > best_blend["best_score"]:
                best_blend = {
                    **row,
                    "u": u,
                    "ut": ut,
                    "best_score": best_score,
                }
            hw = h2h[f"{best_w_local:.3f}"]
            print(
                f"  blend {weights} disp={rd_o['raw_disp']:.3f} F2={gates['max_basis_leak']:.4f} "
                f"best_w={best_w_local} score={best_score:+.3f} "
                f"H2H@{best_w_local} "
                + " ".join(
                    f"{b}:{hw[b]['pos']}/5 m={hw[b]['mean']:+.2f} min={hw[b]['min']:+.2f}"
                    for b in bases_s3
                )
            )
        else:
            w020 = h2h["0.020"]
            print(
                f"  blend {weights} NO H2H-win | "
                + " ".join(
                    f"{b}:{w020[b]['pos']}/5 m={w020[b]['mean']:+.2f} min={w020[b]['min']:+.2f}"
                    for b in bases_s3
                )
            )

    submit = False
    submit_meta = {}
    if best_blend is None:
        print("NO CONFIDENT R2_Prime BLEND — HOLD Slot5; no submit")
        reason = "no_blend_beats_slot5_on_all_3_bases"
    else:
        u_p, ut_p = best_blend["u"], best_blend["ut"]
        w_star = best_blend["best_w"]
        vs3 = {b: score_at_w(u_p, zb, y, folds, w_star) for b, zb in bases_s3.items()}
        h2h_star = {
            b: h2h_vs_slot5(u_p, u_r2, zb, y, folds, w_star) for b, zb in bases_s3.items()
        }
        cand = qlock(L(slot3) + w_star * ut_p, lv)
        rho_s5 = one_minus_rho_e4(cand, slot5)
        rho_s3 = one_minus_rho_e4(cand, slot3)
        u_bad6 = unitize(L(slot6) - L(slot5))
        u_bad7 = unitize(L(slot7) - L(slot5))
        u_delta = unitize(L(cand) - L(slot5))
        corr6 = float(np.corrcoef(u_delta, u_bad6)[0, 1])
        corr7 = float(np.corrcoef(u_delta, u_bad7)[0, 1])
        rd = best_blend["raw_disp"]
        gates = best_blend["gates"]

        # HARD confidence gates (research bar)
        gate_disp = 0.92 <= rd["raw_disp"] <= 1.08
        gate_f = gates["F2"] and gates["F3"] and gates["F1"]
        gate_s3 = (
            all(vs3[b]["pos"] >= 5 for b in bases_s3)
            and all(vs3[b]["sf25"] >= 20 for b in bases_s3)
            and all(vs3[b]["t"] >= 3.8 for b in bases_s3)
            and vs3["z_Slot3_twin"]["min"] >= 1.0  # headroom like R2's +1.80
        )
        gate_h2h = (
            all(h2h_star[b]["pos"] >= 4 for b in bases_s3)
            and all(h2h_star[b]["mean"] > 0 for b in bases_s3)
            and all(h2h_star[b]["min"] >= 0 for b in bases_s3)
            and all(h2h_star[b]["sf25"] >= 18 for b in bases_s3)
            and all(h2h_star[b]["t"] >= 2.8 for b in bases_s3)
        )
        # Prefer 5/5 H2H for confidence
        gate_h2h_strong = all(h2h_star[b]["pos"] >= 5 for b in bases_s3)
        gate_rho = 0.18 <= rho_s5 <= 0.50
        gate_ray = abs(corr6) < 0.15 and abs(corr7) < 0.15
        # Extra: Prime must beat R2_v178 on twin mean at same w=0.020
        r2_twin = family_audit["R2_v178_orig"]["per_w020"]["z_Slot3_twin"]["mean"]
        p_twin = score_at_w(u_p, z_slot3_twin, y, folds, 0.020)["mean"]
        gate_upgrade = p_twin >= r2_twin + 0.15  # at least +0.15e-5 twin mean

        submit = all(
            [
                gate_disp,
                gate_f,
                gate_s3,
                gate_h2h,
                gate_h2h_strong,
                gate_rho,
                gate_ray,
                gate_upgrade,
            ]
        )
        submit_meta = {
            "w_star": w_star,
            "weights": best_blend["weights"],
            "rho_s5_e4": rho_s5,
            "rho_s3_e4": rho_s3,
            "corr_slot6": corr6,
            "corr_slot7": corr7,
            "vs3": vs3,
            "h2h": h2h_star,
            "gates_bool": {
                "disp": gate_disp,
                "forensic": gate_f,
                "vs_slot3": gate_s3,
                "h2h": gate_h2h,
                "h2h_5of5": gate_h2h_strong,
                "rho_band": gate_rho,
                "neg_ray": gate_ray,
                "upgrade_vs_r2_twin": gate_upgrade,
            },
            "r2_twin_mean_020": r2_twin,
            "prime_twin_mean_020": p_twin,
            "submit": submit,
        }
        print("  CONFIDENCE GATES:", submit_meta["gates_bool"])
        print(
            f"  w★={w_star} 1-ρ(S5)={rho_s5:.3f}e-4 twin_upgrade "
            f"{r2_twin:+.2f}->{p_twin:+.2f} submit={submit}"
        )
        reason = "all_gates_pass" if submit else "failed_confidence_gates"

        if submit:
            assert len(cand) == n_te and np.isfinite(cand).all()
            assert float(ks_2samp(cand, slot5).statistic) < 0.0005
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(OUT_CSV, index=False)
            msg = (
                f"v19.1 Slot8: Slot3+R2_Prime multi-arch bag replace 1-seed R2 "
                f"w={w_star:.3f} 1-ρ(S5)={rho_s5:.3f}e-4 H2H 5/5×3"
            )
            print("=== SUBMIT ===")
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
            pub_score, ref = None, None
            for i in range(30):
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
                    if "v19_1" in fn or "slot8" in fn.lower() or "R2_Prime" in fn:
                        ref = row.get("ref")
                        print(f"  poll: {st} ref={ref} score={row.get('publicScore')}")
                        if "complete" in st.lower() and row.get("publicScore"):
                            pub_score = row["publicScore"]
                            break
                if pub_score:
                    break
            submit_meta["ref"] = ref
            submit_meta["public_lb"] = pub_score
            submit_meta["submit_stdout"] = r.stdout
            print(f"PUBLIC_LB={pub_score}")
        else:
            # stage NOSUBMIT for inspection if we have a best blend
            nosub = SUB / f"sub_v19_1_R2_Prime_w{str(w_star).replace('.','p')}_NOSUBMIT.csv"
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(nosub, index=False)
            submit_meta["nosubmit_file"] = nosub.name
            print(f"STAGED NOSUBMIT {nosub.name} (not submitted)")

    payload = scrub(
        {
            "reason": reason,
            "submit": submit,
            "clean_families": clean,
            "family_audit": {k: {kk: vv for kk, vv in v.items() if kk != "u"} for k, v in family_audit.items()},
            "n_blend_tried": len(blend_rows),
            "best_blend_weights": None if best_blend is None else best_blend["weights"],
            "submit_meta": submit_meta,
            "X39_cols": int(X39_tr.shape[1]),
            "X51_cols": int(X51_tr.shape[1]),
            "slot5_rebuild_1mrho_e4": rho_s5 if "rho_s5" in dir() else None,
            "elapsed_s": time.time() - t0,
        }
    )
    # fix slot5 rebuild metric
    payload["slot5_rebuild_1mrho_e4"] = one_minus_rho_e4(
        qlock(L(slot3) + W_R2_SLOT5 * ut_r2, lv), slot5
    )
    OUT.joinpath("v19_1.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUT / "v19_1_families.npz",
        **{f"{k}_oof": v[0] for k, v in families_raw.items()},
        **{f"{k}_test": v[1] for k, v in families_raw.items()},
    )
    print("WROTE", OUT / "v19_1.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE submit=", submit, "reason=", reason)


if __name__ == "__main__":
    main()
