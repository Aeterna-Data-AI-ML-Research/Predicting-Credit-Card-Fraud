#!/usr/bin/env python3
"""GM V17.8 — Newton residual TE + shallow residual booster on r=y-p_S3.

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
from scipy.stats import rankdata, t as tdist, wilcoxon
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
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
NO_SUBMIT = True
RIDGE_ALPHA = 10.0
LAMBDA0 = 20.0
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
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


def frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=RIDGE_ALPHA):
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


def forensic_gates(orth_o, orth_t, Bt, live_resid):
    disp = float((orth_t.std() + 1e-12) / (orth_o.std() + 1e-12))
    leaks = [float(np.corrcoef(orth_t, Bt[:, j])[0, 1]) for j in range(Bt.shape[1])]
    max_leak = float(np.max(np.abs(leaks)))
    c_live = float(np.corrcoef(orth_t, live_resid)[0, 1]) if live_resid.std() > 1e-12 else 0.0
    return {
        "disp": disp,
        "max_basis_leak": max_leak,
        "corr_live_resid": c_live,
        "F1": 0.94 <= disp <= 1.06,
        "F2": max_leak < 0.035,
        "F3": abs(c_live) < 0.05,
        "verified": (0.94 <= disp <= 1.06) and (max_leak < 0.035) and (abs(c_live) < 0.05),
    }


def score_fixed(u, zbase, y, folds, w_grid=(0.005, 0.01, 0.015, 0.02, 0.025, 0.03)):
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
    t = mean / (se + 1e-18)
    try:
        _, wp = wilcoxon(d25, alternative="greater")
        wp = float(wp)
    except Exception:
        wp = float("nan")
    best.update({"sf25": int((d25 > 0).sum()), "t": float(t), "wilcoxon_p": wp, "mean_sf": mean})
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


def newton_te_map(keys_tr, g_tr, h_tr, keys_te, lam=LAMBDA0):
    """Mean g/(h+lam) per key; map to te keys. keys are int codes >=0."""
    keys_tr = np.asarray(keys_tr, np.int64)
    keys_te = np.asarray(keys_te, np.int64)
    g_tr = np.asarray(g_tr, float)
    h_tr = np.asarray(h_tr, float)
    nbin = int(max(keys_tr.max(), keys_te.max()) + 1)
    sg = np.bincount(keys_tr, weights=g_tr, minlength=nbin)
    sh = np.bincount(keys_tr, weights=h_tr, minlength=nbin)
    cnt = np.bincount(keys_tr, minlength=nbin)
    delta = sg / (sh + lam)
    # unseen -> 0
    delta[cnt == 0] = 0.0
    return delta[keys_tr], delta[keys_te]


def make_digit_keys(df):
    """Return dict name -> int key array (nonneg)."""
    out = {}
    for c in NUM_COLS:
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=np.float64)
        out[f"{c}_cents"] = (np.nan_to_num(np.round(v * 100), nan=0).astype(np.int64) % 100).astype(
            np.int64
        )
        out[f"{c}_tenth"] = (np.nan_to_num(np.round(v * 10), nan=0).astype(np.int64) % 10).astype(
            np.int64
        )
        out[f"{c}_ones"] = (np.nan_to_num(np.abs(v), nan=0).astype(np.int64) % 10).astype(np.int64)
    # q1000 on main floats
    for c in ("amount_usd", "account_balance_usd", "distance_from_home_km", "velocity_score"):
        v = df[c].to_numpy(dtype=np.float64)
        # ranks -> 1000 bins via train+test handled outside; here per-frame provisional
        r = rankdata(v, method="average")
        out[f"{c}_q1000"] = np.clip((r / (len(v) + 1e-12) * 1000).astype(np.int64), 0, 999)
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

    Z = np.load(Q17, allow_pickle=True)
    qnames = [str(s) for s in Z["stream_names"]]
    XO = {n: Z[f"oof_{n}"].ravel() for n in qnames}
    XT = {n: Z[f"test_{n}"].ravel() for n in qnames}

    live = pd.read_csv(LIVE)
    ids = live["id"].to_numpy()
    lv = live["is_fraud"].to_numpy(float)
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

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

    z_s3 = z_ath + 0.01045 * dE_o + 0.0055 * unitize(d1)
    print(f"AUC z_S3={roc_auc_score(y,z_s3):.7f}")

    ms_o = np.load(REPO / "preds/P_super_v4_oof.npy").ravel()
    ms_t = np.load(REPO / "preds/P_super_v4_test.npy").ravel()
    g26_o = np.load(G26_OOF).ravel()[:n_tr]
    g26_t = np.load(G26_T0).ravel()[:n_te]
    lm_o, lm_t = XO["megayak_5f_lm"], XT["megayak_5f_lm"]
    gpd_o, gpd_t = XO["daoviet_gpdvae"], XT["daoviet_gpdvae"]
    gb_o, gb_t = XO["gb_bag5"], XT["gb_bag5"]
    mlp_o, mlp_t = XO["yekenot_realmlp"], XT["yekenot_realmlp"]

    Bo9 = np.column_stack(
        [
            unitize(L(ms_o)),
            unitize(L(g26_o)),
            unitize(L(lm_o)),
            unitize(L(l7f_o)),
            unitize(gpd_o),
            unitize(gb_o),
            unitize(mlp_o),
            unitize(d1),
            unitize(dE_o),
        ]
    )
    Bt9 = np.column_stack(
        [
            unitize(L(ms_t)),
            unitize(L(g26_t)),
            unitize(L(lm_t)),
            unitize(L(l7f_t)),
            unitize(gpd_t),
            unitize(gb_t),
            unitize(mlp_t),
            unitize(d1t),
            unitize(dE_t),
        ]
    )

    # twins
    y_s3t = unitize(L(slot3))
    ridge_s3 = Ridge(alpha=1.0, fit_intercept=True).fit(Bt9, y_s3t)
    z_slot3_twin = unitize(Bo9 @ ridge_s3.coef_ + ridge_s3.intercept_)

    bank_o, bank_t, bank_n = [], [], []

    def add(n, o, t):
        bank_n.append(n)
        bank_o.append(unitize(o))
        bank_t.append(unitize(t))

    for j, nm in enumerate(BN9):
        if nm in ("d1_v4", "Family_E"):
            continue
        add(f"Bo9_{nm}", Bo9[:, j], Bt9[:, j])
    skip = {
        "megayak_5f_lm",
        "lm7_5fold",
        "gpt2_seed2026",
        "masaya_gpt2",
        "daoviet_gpdvae",
        "gb_bag5",
        "yekenot_realmlp",
    }
    for n in qnames:
        if n not in skip:
            add(f"Q17_{n}", XO[n], XT[n])
    for p in sorted(ALEX.glob("oof_*.npy"))[:8]:
        tp = ALEX / ("test_" + p.name[len("oof_") :])
        if tp.exists():
            add(f"alex_{p.stem[-8:]}", np.load(p).ravel()[:n_tr], np.load(tp).ravel()[:n_te])
    Bo_bank = np.column_stack(bank_o)
    Bt_bank = np.column_stack(bank_t)
    locked_t = W_D1_LOCK * unitize(d1t) + W_FE_LOCK * unitize(dE_t)
    resid_t = L(slot3) - locked_t
    w_nnls, _ = nnls(Bt_bank, resid_t - resid_t.mean())
    z_nnls = W_D1_LOCK * unitize(d1) + W_FE_LOCK * unitize(dE_o) + Bo_bank @ w_nnls

    baselines = {"z_S3": z_s3, "z_Slot3_twin": z_slot3_twin, "z_NNLS_twin": z_nnls}
    live_L = L(lv)
    live_resid = live_L - Ridge(alpha=RIDGE_ALPHA).fit(Bt9, live_L).predict(Bt9)

    # ========== STEP 1: calibrated residual ==========
    print("=== STEP 1: calibrated r_oof ===")
    r_oof = np.zeros(n_tr)
    p_oof = np.zeros(n_tr)
    g_fold = np.zeros(n_tr)  # same as r on va; on tr used for TE
    h_fold = np.zeros(n_tr)
    ab = []
    for f, (tr, va) in enumerate(folds):
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s3[tr].reshape(-1, 1), y[tr])
        a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
        ab.append((a, b))
        # full-vector p under this fold's calib (for g,h on tr)
        p_all = expit(a * z_s3 + b)
        g_all = y - p_all
        h_all = p_all * (1.0 - p_all)
        p_oof[va] = p_all[va]
        r_oof[va] = g_all[va]
        # store fold-specific g,h only needed on tr when encoding — keep per-fold arrays
        g_fold[va] = g_all[va]  # placeholder
        h_fold[va] = h_all[va]
    corr_r_z = float(np.corrcoef(r_oof, z_s3)[0, 1])
    # weighted residual corr with design on each fold
    wcorr = []
    for f, (tr, va) in enumerate(folds):
        a, b = ab[f]
        p = expit(a * z_s3[tr] + b)
        g = y[tr] - p
        h = p * (1 - p)
        # weighted corr of g with z
        wz = z_s3[tr] - np.average(z_s3[tr], weights=h)
        wg = g - np.average(g, weights=h)
        wcorr.append(float(np.average(wz * wg, weights=h) / (np.sqrt(np.average(wz**2, weights=h) * np.average(wg**2, weights=h)) + 1e-18)))
    print(f"  corr(r_oof,z_S3)={corr_r_z:+.6f}  mean_weighted_corr(g,z)_tr={float(np.mean(wcorr)):+.6e}")
    print(f"  AUC p_oof={roc_auc_score(y,p_oof):.7f}")

    # also calib for twin (for reporting)
    r_twin = np.zeros(n_tr)
    for tr, va in folds:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_slot3_twin[tr].reshape(-1, 1), y[tr])
        r_twin[va] = y[va] - clf.predict_proba(z_slot3_twin[va].reshape(-1, 1))[:, 1]

    # ========== STEP 2: features ==========
    print("=== STEP 2: raw features + R1/R2 ===")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    # align order to fold_id / ids
    assert len(train) == n_tr and len(test) == n_te

    # joint q1000 using train∪test ranks for consistency
    def joint_q1000(col):
        v = np.concatenate([train[col].to_numpy(float), test[col].to_numpy(float)])
        r = rankdata(v, method="average")
        q = np.clip((r / (len(v) + 1e-12) * 1000).astype(np.int64), 0, 999)
        return q[:n_tr], q[n_tr:]

    keys_tr = {}
    keys_te = {}
    for c in NUM_COLS:
        vtr = train[c].to_numpy(float)
        vte = test[c].to_numpy(float)
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

    # exact amount frequency in train∪test
    amt_all = np.round(
        np.concatenate([train["amount_usd"].to_numpy(float), test["amount_usd"].to_numpy(float)])
        * 100
    ).astype(np.int64)
    # hash to dense codes via unique
    uniq, inv = np.unique(amt_all, return_inverse=True)
    cnt = np.bincount(inv)
    freq = np.log1p(cnt[inv])
    keys_tr["amt_freq_bin"] = np.clip((freq[:n_tr] * 5).astype(np.int64), 0, 50)  # coarse for TE
    keys_te["amt_freq_bin"] = np.clip((freq[n_tr:] * 5).astype(np.int64), 0, 50)
    feat_freq_tr = freq[:n_tr]
    feat_freq_te = freq[n_tr:]

    # orig20k amount presence
    orig_cnt = None
    if ORIG.exists():
        orig = pd.read_csv(ORIG, usecols=lambda c: c in ("amount_usd", "Amount", "transaction_amount", "amt"))
        oc = [c for c in orig.columns if "amount" in c.lower() or c == "amt"]
        if oc:
            oa = np.round(orig[oc[0]].to_numpy(float) * 100).astype(np.int64)
            from collections import Counter

            ctr = Counter(oa.tolist())
            orig_cnt_tr = np.array([ctr.get(int(x), 0) for x in amt_all[:n_tr]], dtype=np.float64)
            orig_cnt_te = np.array([ctr.get(int(x), 0) for x in amt_all[n_tr:]], dtype=np.float64)
            keys_tr["amt_orig_bin"] = np.clip(orig_cnt_tr.astype(np.int64), 0, 20)
            keys_te["amt_orig_bin"] = np.clip(orig_cnt_te.astype(np.int64), 0, 20)
            orig_cnt = (orig_cnt_tr, orig_cnt_te)
            print(f"  orig20k amount hits: mean_tr={orig_cnt_tr.mean():.3f}")

    # cat x amount_ones for a couple cats
    for cat in ("merchant_category", "channel", "device_type"):
        codes, _ = pd.factorize(pd.concat([train[cat], test[cat]], ignore_index=True))
        ctr_c, cte_c = codes[:n_tr], codes[n_tr:]
        ones_tr = keys_tr["amount_usd_ones"]
        ones_te = keys_te["amount_usd_ones"]
        keys_tr[f"{cat}_x_amtones"] = ctr_c * 10 + ones_tr
        keys_te[f"{cat}_x_amtones"] = cte_c * 10 + ones_te

    key_names = list(keys_tr.keys())
    print(f"  TE keys={len(key_names)} elapsed={time.time()-t0:.1f}s")

    # R1: per-fold Newton TE + Ridge combiner
    R1_oof = np.zeros(n_tr)
    R1_test_folds = []
    for f, (tr, va) in enumerate(folds):
        a, b = ab[f]
        p = expit(a * z_s3 + b)
        g = y - p
        h = np.clip(p * (1 - p), 1e-6, None)
        # per-key TE columns on tr/va/test
        Xtr_cols, Xva_cols, Xte_cols = [], [], []
        for kn in key_names:
            dtr, dte = newton_te_map(keys_tr[kn][tr], g[tr], h[tr], keys_te[kn], LAMBDA0)
            # map va via same mapping from tr
            # rebuild delta map
            ktr = keys_tr[kn][tr]
            nbin = int(max(ktr.max(), keys_tr[kn][va].max(), keys_te[kn].max()) + 1)
            sg = np.bincount(ktr, weights=g[tr], minlength=nbin)
            sh = np.bincount(ktr, weights=h[tr], minlength=nbin)
            cnt = np.bincount(ktr, minlength=nbin)
            delta = sg / (sh + LAMBDA0)
            delta[cnt == 0] = 0.0
            Xtr_cols.append(delta[keys_tr[kn][tr]])
            Xva_cols.append(delta[keys_tr[kn][va]])
            Xte_cols.append(delta[keys_te[kn]])
        Xtr = np.column_stack(Xtr_cols)
        Xva = np.column_stack(Xva_cols)
        Xte = np.column_stack(Xte_cols)
        # weighted ridge on g
        sw = np.sqrt(h[tr])
        lr = Ridge(alpha=100.0, fit_intercept=True)
        lr.fit(Xtr * sw[:, None], g[tr] * sw)
        R1_oof[va] = lr.predict(Xva)
        R1_test_folds.append(lr.predict(Xte))
        print(f"  R1 fold{f} done t={time.time()-t0:.1f}s")
    R1_test = np.mean(np.stack(R1_test_folds, 0), 0)

    # R2: LightGBM residual with init via fitting g with weight h
    # features: digit keys as numeric + freq + a few raw nums
    def build_X(split="train"):
        cols = []
        src_keys = keys_tr if split == "train" else keys_te
        for kn in key_names:
            cols.append(src_keys[kn].astype(np.float64))
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
        return np.column_stack(cols)

    X_all_tr = build_X("train")
    X_all_te = build_X("test")
    R2_oof = np.zeros(n_tr)
    R2_test_folds = []
    lgb_params = dict(
        objective="regression",
        learning_rate=0.03,
        num_leaves=6,  # ~ max_leaf_nodes
        min_data_in_leaf=500,
        n_estimators=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbosity=-1,
        n_jobs=4,
    )
    for f, (tr, va) in enumerate(folds):
        a, b = ab[f]
        p = expit(a * z_s3 + b)
        g = y - p
        h = np.clip(p * (1 - p), 1e-6, None)
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_all_tr[tr],
            g[tr],
            sample_weight=h[tr],
        )
        R2_oof[va] = model.predict(X_all_tr[va])
        R2_test_folds.append(model.predict(X_all_te))
        print(f"  R2 fold{f} done t={time.time()-t0:.1f}s")
    R2_test = np.mean(np.stack(R2_test_folds, 0), 0)

    pre_corr = {
        "R1_zS3": float(np.corrcoef(R1_oof, z_s3)[0, 1]),
        "R2_zS3": float(np.corrcoef(R2_oof, z_s3)[0, 1]),
        "R1_r": float(np.corrcoef(R1_oof, r_oof)[0, 1]),
        "R2_r": float(np.corrcoef(R2_oof, r_oof)[0, 1]),
    }
    print(f"  pre-orth corr: {pre_corr}")

    # U5 from V17.7
    raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
    llr_o = raw_o[:, :, 1] - raw_o[:, :, 0]
    llr_t = raw_t[:, :, :, 1] - raw_t[:, :, :, 0]
    U5_o = L_fold(gini10(col_z(llr_o)), fold_id)
    U5_t = L(np.stack([L(gini10(col_z(llr_t[f]))) for f in range(5)], 0).mean(0))

    # ========== STEP 3: orth + scorecard ==========
    print("=== STEP 3: scorecard ===")
    cands_raw = {
        "R1_newton_TE": (R1_oof, R1_test),
        "R2_residual_LGB": (R2_oof, R2_test),
        "U5_gini": (U5_o, U5_t),
    }
    oriented = {}
    results = {}

    def eval_signal(name, vo, vt):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds)
        gates = forensic_gates(oo, ot, Bt9, live_resid)
        # fixed + sign = helpful on z_S3 fold0; then SAME u on all bases
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s3[te0] - 0.01 * u[te0]
        ):
            u = -u
            ot = -ot
        ut = unitize(ot)
        oriented[name] = (u, ut)
        per = {}
        for bname, zbase in baselines.items():
            per[bname] = score_fixed(u, zbase, y, folds)
        dual = (
            per["z_S3"]["pos"] >= 4
            and per["z_S3"]["mean"] > 0
            and per["z_Slot3_twin"]["pos"] >= 4
            and per["z_Slot3_twin"]["mean"] > 0
            and per["z_S3"]["sf25"] >= 18
            and per["z_S3"]["t"] >= 2.20
            and per["z_Slot3_twin"]["sf25"] >= 18
            and per["z_Slot3_twin"]["t"] >= 2.20
        )
        results[name] = {
            "gates": gates,
            "pre_corr_zS3": float(np.corrcoef(vo, z_s3)[0, 1]),
            "per_base": per,
            "dual_pass": dual,
        }
        print(
            f"  {name}: pre|ρ|={abs(results[name]['pre_corr_zS3']):.4f} "
            f"F1={gates['F1']}({gates['disp']:.3f}) F2={gates['F2']}({gates['max_basis_leak']:.4f}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) ver={gates['verified']} dual={dual}"
        )
        for bname, row in per.items():
            print(
                f"    {bname}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
                f"min={row['min']:+.3f} sf={row['sf25']}/25 t={row['t']:.2f} folds={row['folds']}"
            )
        return u, ut, oo, ot, gates

    for name, (vo, vt) in cands_raw.items():
        eval_signal(name, vo, vt)

    # R3 blend
    eval_signal(
        "R3_blend_R1_R2_U5",
        unitize(R1_oof) + unitize(R2_oof) + unitize(U5_o),
        unitize(R1_test) + unitize(R2_test) + unitize(U5_t),
    )
    # also R1+R2 without U5
    eval_signal(
        "R12_blend",
        unitize(R1_oof) + unitize(R2_oof),
        unitize(R1_test) + unitize(R2_test),
    )

    winners = [
        n
        for n, r in results.items()
        if r["gates"]["verified"] and r["dual_pass"]
    ]
    # also require NNLS soft: mean>0 pos>=4
    winners = [
        n
        for n in winners
        if results[n]["per_base"]["z_NNLS_twin"]["pos"] >= 4
        and results[n]["per_base"]["z_NNLS_twin"]["mean"] > 0
    ]
    print(f"WINNERS={winners}")

    staged = []
    for n in winners:
        u_o, u_t = oriented[n]
        wstar = results[n]["per_base"]["z_S3"]["w"]
        # also check twin best w — use min of the two best or S3 w
        path = SUB / f"sub_v17_8_{n}_w{str(wstar).replace('.','p')}_NOSUBMIT.csv"
        cand = qlock(L(slot3) + wstar * u_t, slot3)
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
        staged.append({"file": path.name, "signal": n, "w": wstar})
        print("STAGED", path.name)

    if not winners:
        print("NO STAGE")

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "elapsed_s": time.time() - t0,
            "step1": {
                "corr_r_oof_z_S3": corr_r_z,
                "mean_weighted_corr_g_z_tr": float(np.mean(wcorr)),
                "auc_p_oof": float(roc_auc_score(y, p_oof)),
                "calib_ab": [{"a": a, "b": b} for a, b in ab],
            },
            "pre_corr": pre_corr,
            "results": results,
            "winners": winners,
            "staged": staged,
            "stage_rule": "F1∧F2∧F3 ∧ dual S3∩twin (pos>=4, mean>0, sf>=18, t>=2.20) ∧ NNLS soft",
        }
    )
    OUT.joinpath("v17_8.json").write_text(json.dumps(payload, indent=2))
    # persist oof for reuse
    np.savez_compressed(
        OUT / "v17_8_residuals.npz",
        R1_oof=R1_oof,
        R1_test=R1_test,
        R2_oof=R2_oof,
        R2_test=R2_test,
        r_oof=r_oof,
        p_oof=p_oof,
    )
    print("WROTE", OUT / "v17_8.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
