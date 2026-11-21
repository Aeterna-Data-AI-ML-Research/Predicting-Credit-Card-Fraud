#!/usr/bin/env python3
"""GM V18.2 — Autopsy V17.8 R2 + leak-free multi-arch R2_pro on Slot3.

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import rankdata, wilcoxon
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
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
NO_SUBMIT = True
RIDGE_ALPHA = 10.0
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W_R2_V178 = 0.020
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
CAT_COLS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
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
        "leak_argmax": leak_arg,
        "leak_col": names[leak_arg] if leak_arg < len(names) else "?",
        "corr_live_resid": c_live,
        "F1": 0.94 <= disp <= 1.06,
        "F2": max_leak < 0.035,
        "F3": abs(c_live) < 0.05,
        "verified": (0.94 <= disp <= 1.06) and (max_leak < 0.035) and (abs(c_live) < 0.05),
    }


def score_fixed(u, zbase, y, folds, w_grid=(0.01, 0.015, 0.02, 0.025, 0.03)):
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
    best.update({"sf25": int((d25 > 0).sum()), "t": float(t), "wilcoxon_p": wp})
    return best


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
    return {
        "w": w,
        "pos": int(sum(x > 0 for x in d)),
        "mean": float(np.mean(d)),
        "min": float(np.min(d)),
        "folds": [round(x, 3) for x in d],
        "sf25": int((d25 > 0).sum()),
        "t": float(mean / (se + 1e-18)),
    }


def cross_fold_r(test_folds):
    # test_folds: list of length 5 arrays
    rs = []
    for i in range(5):
        for j in range(i + 1, 5):
            rs.append(float(np.corrcoef(test_folds[i], test_folds[j])[0, 1]))
    return float(np.mean(rs))


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

    z_s3 = z_ath + 0.01045 * dE_o + 0.0055 * unitize(d1)

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

    live_resid = L(lv) - Ridge(alpha=RIDGE_ALPHA).fit(Bt9, L(lv)).predict(Bt9)
    bases_s3 = {"z_S3": z_s3, "z_Slot3_twin": z_slot3_twin, "z_NNLS_twin": z_nnls}

    # calib ab for z_S3
    ab = []
    for tr, va in folds:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s3[tr].reshape(-1, 1), y[tr])
        ab.append((float(clf.coef_[0, 0]), float(clf.intercept_[0])))
    g_folds, h_folds = [], []
    for f in range(5):
        a, b = ab[f]
        p = expit(a * z_s3 + b)
        g_folds.append(y - p)
        h_folds.append(np.clip(p * (1 - p), 1e-6, None))

    # ========== FEATURES (V17.8 exact + expanded raw) ==========
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
        keys_tr[f"{c}_mod5"] = (np.abs(vtr).astype(np.int64) % 5) % 5
        keys_te[f"{c}_mod5"] = (np.abs(vte).astype(np.int64) % 5) % 5
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
    freq = np.log1p(cnt[inv])
    keys_tr["amt_freq_bin"] = np.clip((freq[:n_tr] * 5).astype(np.int64), 0, 50)
    keys_te["amt_freq_bin"] = np.clip((freq[n_tr:] * 5).astype(np.int64), 0, 50)
    feat_freq_tr, feat_freq_te = freq[:n_tr], freq[n_tr:]

    orig_df = pd.read_csv(ORIG) if ORIG.exists() else None
    orig_cnt = None
    if orig_df is not None:
        oc = [c for c in orig_df.columns if "amount" in c.lower() or c.lower() == "amt"]
        if oc:
            oa = np.round(orig_df[oc[0]].to_numpy(float) * 100).astype(np.int64)
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

    # V17.8 exact column list (no mod5 in original — keep separate)
    v178_key_names = [
        f"{c}_{s}"
        for c in NUM_COLS
        for s in ("cents", "tenth", "ones")
    ] + [
        f"{c}_q1000"
        for c in ("amount_usd", "account_balance_usd", "distance_from_home_km", "velocity_score")
    ] + ["amt_freq_bin"]
    if orig_cnt is not None:
        v178_key_names.append("amt_orig_bin")
    v178_key_names += [
        "merchant_category_x_amtones",
        "channel_x_amtones",
        "device_type_x_amtones",
    ]
    v178_feat_names = (
        list(v178_key_names)
        + ["amt_logfreq"]
        + ["amount_usd", "velocity_score", "merchant_risk_score", "distance_from_home_km"]
        + (["amt_orig_cnt"] if orig_cnt is not None else [])
    )

    def build_X_v178(split="train"):
        cols = []
        src = keys_tr if split == "train" else keys_te
        for kn in v178_key_names:
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
        return np.column_stack(cols)

    X_v178_tr = build_X_v178("train")
    X_v178_te = build_X_v178("test")
    assert X_v178_tr.shape[1] == len(v178_feat_names)
    print(
        f"V17.8 X cols={X_v178_tr.shape[1]} names_ok "
        f"(keys={len(v178_key_names)} + cont) t={time.time()-t0:.1f}s"
    )
    print("  V17.8 column list:", v178_feat_names)

    # ===== STEP 1: autopsy =====
    print("=== STEP 1: V17.8 R2 feature importance autopsy ===")
    gain = np.zeros(len(v178_feat_names))
    split = np.zeros(len(v178_feat_names))
    R2_v178_oof = np.zeros(n_tr)
    R2_v178_te_folds = []
    lgb_v178 = dict(
        objective="regression",
        learning_rate=0.03,
        num_leaves=6,
        min_data_in_leaf=500,
        n_estimators=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbosity=-1,
        n_jobs=4,
        random_state=42,
    )
    for f, (tr, va) in enumerate(folds):
        model = lgb.LGBMRegressor(**lgb_v178)
        model.fit(X_v178_tr[tr], g_folds[f][tr], sample_weight=h_folds[f][tr])
        R2_v178_oof[va] = model.predict(X_v178_tr[va])
        R2_v178_te_folds.append(model.predict(X_v178_te))
        # gain / split
        booster = model.booster_
        gain += np.asarray(booster.feature_importance(importance_type="gain"), float)
        split += np.asarray(booster.feature_importance(importance_type="split"), float)
    R2_v178_test = np.mean(np.stack(R2_v178_te_folds, 0), 0)
    # also load cached if close
    cached = np.load(OUT / "v17_8_residuals.npz")
    print(
        f"  ρ(rebuilt R2, cached)={np.corrcoef(R2_v178_oof, cached['R2_oof'])[0,1]:.6f} "
        f"n_feats={len(v178_feat_names)} (NO TE columns)"
    )
    order_gain = np.argsort(-gain)
    order_split = np.argsort(-split)
    top20_gain = [
        {"feat": v178_feat_names[i], "gain": float(gain[i]), "split": float(split[i])}
        for i in order_gain[:20]
    ]
    top20_split = [
        {"feat": v178_feat_names[i], "gain": float(gain[i]), "split": float(split[i])}
        for i in order_split[:20]
    ]
    print("  TOP20 by GAIN:")
    for j, row in enumerate(top20_gain, 1):
        print(f"    {j:2d}. {row['feat']:40s} gain={row['gain']:.1f} split={row['split']:.0f}")
    print("  TOP20 by SPLIT:")
    for j, row in enumerate(top20_split, 1):
        print(f"    {j:2d}. {row['feat']:40s} split={row['split']:.0f} gain={row['gain']:.1f}")

    # ===== STEP 2: expanded raw X (no TE) =====
    print("=== STEP 2: build R2_pro bags ===", f"t={time.time()-t0:.1f}s")
    feat_names = []
    cols_tr, cols_te = [], []

    def add(name, trv, tev):
        feat_names.append(name)
        cols_tr.append(np.asarray(trv, float))
        cols_te.append(np.asarray(tev, float))

    # start from V17.8 matrix columns
    for j, nm in enumerate(v178_feat_names):
        add(nm, X_v178_tr[:, j], X_v178_te[:, j])
    # mod5 digits
    for c in NUM_COLS:
        add(f"{c}_mod5", keys_tr[f"{c}_mod5"], keys_te[f"{c}_mod5"])
    # all-column train∪test freq + orig hits
    row_hits_tr = np.zeros(n_tr)
    row_hits_te = np.zeros(n_te)
    row_log_tr = np.zeros(n_tr)
    row_log_te = np.zeros(n_te)
    n_oc = 0
    all_cols = [c for c in train.columns if c not in ("id", "is_fraud")]
    for c in all_cols:
        if c in NUM_COLS:
            # numeric exact via rounded string of full precision limited
            trs = np.round(train[c].to_numpy(float), 6).astype(str)
            tes = np.round(test[c].to_numpy(float), 6).astype(str)
        else:
            trs = train[c].astype(str).to_numpy()
            tes = test[c].astype(str).to_numpy()
        allv = np.concatenate([trs, tes])
        _, invc = np.unique(allv, return_inverse=True)
        cntc = np.bincount(invc)
        logf = np.log1p(cntc[invc])
        add(f"logfreq_{c}", logf[:n_tr], logf[n_tr:])
        if orig_df is not None and c in orig_df.columns:
            if c in NUM_COLS:
                oc_vals = np.round(orig_df[c].to_numpy(float), 6).astype(str)
            else:
                oc_vals = orig_df[c].astype(str).to_numpy()
            ctr = Counter(oc_vals.tolist())
            otr = np.array([ctr.get(x, 0) for x in trs], dtype=np.float64)
            ote = np.array([ctr.get(x, 0) for x in tes], dtype=np.float64)
            add(f"origcnt_{c}", np.log1p(otr), np.log1p(ote))
            row_hits_tr += (otr > 0).astype(float)
            row_hits_te += (ote > 0).astype(float)
            row_log_tr += np.log1p(otr)
            row_log_te += np.log1p(ote)
            n_oc += 1
    add("row_orig_hits", row_hits_tr, row_hits_te)
    if n_oc:
        add("row_orig_logmean", row_log_tr / n_oc, row_log_te / n_oc)

    # pairwise ratios/diffs of top continuous
    topc = [
        "amount_usd",
        "velocity_score",
        "merchant_risk_score",
        "distance_from_home_km",
        "account_balance_usd",
        "hours_since_last_txn",
    ]
    for i, a in enumerate(topc):
        for b in topc[i + 1 :]:
            atr, ate = train[a].to_numpy(float), test[a].to_numpy(float)
            btr, bte = train[b].to_numpy(float), test[b].to_numpy(float)
            add(f"diff_{a}_{b}", atr - btr, ate - bte)
            add(f"ratio_{a}_{b}", atr / (np.abs(btr) + 1e-6), ate / (np.abs(bte) + 1e-6))

    # cat pair frequency encodings (NO TE)
    pairs = [
        ("merchant_category", "channel"),
        ("merchant_category", "device_type"),
        ("channel", "device_type"),
        ("card_type", "auth_method"),
        ("merchant_category", "card_type"),
    ]
    for a, b in pairs:
        sa_tr = train[a].astype(str).to_numpy()
        sa_te = test[a].astype(str).to_numpy()
        sb_tr = train[b].astype(str).to_numpy()
        sb_te = test[b].astype(str).to_numpy()
        ptr = np.char.add(np.char.add(sa_tr, "|"), sb_tr)
        pte = np.char.add(np.char.add(sa_te, "|"), sb_te)
        codes, _ = pd.factorize(np.concatenate([ptr, pte]))
        cntp = np.bincount(codes)
        logf = np.log1p(cntp[codes])
        add(f"pairfreq_{a}_{b}", logf[:n_tr], logf[n_tr:])
        add(f"paircode_{a}_{b}", codes[:n_tr].astype(float), codes[n_tr:].astype(float))

    X_tr = np.column_stack(cols_tr).astype(np.float32)
    X_te = np.column_stack(cols_te).astype(np.float32)
    print(f"  X_raw shape={X_tr.shape} (ZERO TE cols) t={time.time()-t0:.1f}s")

    # Family A: LGB bag
    lgb_seeds = [42, 2026, 7, 100, 999]
    lgb_leaves = [6, 8]
    R2_lgb_oof = np.zeros(n_tr)
    R2_lgb_te_acc = []
    n_lgb = 0
    for seed in lgb_seeds:
        for leaves in lgb_leaves:
            oof = np.zeros(n_tr)
            tef = []
            for f, (tr, va) in enumerate(folds):
                model = lgb.LGBMRegressor(
                    objective="regression",
                    learning_rate=0.03,
                    num_leaves=leaves,
                    min_data_in_leaf=500,
                    n_estimators=55,
                    subsample=0.8,
                    colsample_bytree=0.75,
                    reg_lambda=1.0,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=4,
                )
                model.fit(X_tr[tr], g_folds[f][tr], sample_weight=h_folds[f][tr])
                oof[va] = model.predict(X_tr[va])
                tef.append(model.predict(X_te))
            R2_lgb_oof += oof
            R2_lgb_te_acc.append(np.mean(np.stack(tef, 0), 0))
            n_lgb += 1
    R2_lgb_oof /= n_lgb
    R2_lgb_test = np.mean(np.stack(R2_lgb_te_acc, 0), 0)
    # cross-fold from last seed's fold preds — rebuild one pass for cross_fold_r
    print(f"  LGB bag done n={n_lgb} t={time.time()-t0:.1f}s")

    # Family B: CatBoost
    cat_seeds = [42, 2026, 7]
    R2_cat_oof = np.zeros(n_tr)
    R2_cat_te_acc = []
    n_cat = 0
    for seed in cat_seeds:
        oof = np.zeros(n_tr)
        tef = []
        for f, (tr, va) in enumerate(folds):
            model = CatBoostRegressor(
                loss_function="RMSE",
                depth=3,
                l2_leaf_reg=50,
                iterations=120,
                learning_rate=0.04,
                random_seed=seed,
                verbose=False,
                thread_count=4,
                min_data_in_leaf=500,
            )
            model.fit(X_tr[tr], g_folds[f][tr], sample_weight=h_folds[f][tr])
            oof[va] = model.predict(X_tr[va])
            tef.append(model.predict(X_te))
        R2_cat_oof += oof
        R2_cat_te_acc.append(np.mean(np.stack(tef, 0), 0))
        n_cat += 1
    R2_cat_oof /= n_cat
    R2_cat_test = np.mean(np.stack(R2_cat_te_acc, 0), 0)
    print(f"  Cat bag done n={n_cat} t={time.time()-t0:.1f}s")

    # Family C: HistGB
    hgb_seeds = [42, 2026, 7]
    R2_hgb_oof = np.zeros(n_tr)
    R2_hgb_te_acc = []
    n_hgb = 0
    for seed in hgb_seeds:
        oof = np.zeros(n_tr)
        tef = []
        for f, (tr, va) in enumerate(folds):
            model = HistGradientBoostingRegressor(
                max_leaf_nodes=6,
                min_samples_leaf=500,
                l2_regularization=20.0,
                learning_rate=0.03,
                max_iter=50,
                random_state=seed,
            )
            model.fit(X_tr[tr], g_folds[f][tr], sample_weight=h_folds[f][tr])
            oof[va] = model.predict(X_tr[va])
            tef.append(model.predict(X_te))
        R2_hgb_oof += oof
        R2_hgb_te_acc.append(np.mean(np.stack(tef, 0), 0))
        n_hgb += 1
    R2_hgb_oof /= n_hgb
    R2_hgb_test = np.mean(np.stack(R2_hgb_te_acc, 0), 0)
    print(f"  HGB bag done n={n_hgb} t={time.time()-t0:.1f}s")

    # R2_pro ensemble
    R2_pro_oof = unitize(
        0.45 * unitize(R2_lgb_oof) + 0.35 * unitize(R2_cat_oof) + 0.20 * unitize(R2_hgb_oof)
    )
    R2_pro_test = unitize(
        0.45 * unitize(R2_lgb_test) + 0.35 * unitize(R2_cat_test) + 0.20 * unitize(R2_hgb_test)
    )

    # cross_fold_r: need per-fold test preds for each family — recompute lightly for v178 + one config each
    def cf_from_list(te_folds):
        return cross_fold_r(te_folds)

    # For bags, approximate with seed-averaged fold stacks already lost; rebuild cheap single-config cf
    def rebuild_cf_lgb(seed=42, leaves=6, n_est=40, Xtr=X_v178_tr, Xte=X_v178_te):
        tef = []
        for f, (tr, va) in enumerate(folds):
            model = lgb.LGBMRegressor(
                objective="regression",
                learning_rate=0.03,
                num_leaves=leaves,
                min_data_in_leaf=500,
                n_estimators=n_est,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_lambda=1.0,
                random_state=seed,
                verbosity=-1,
                n_jobs=4,
            )
            model.fit(Xtr[tr], g_folds[f][tr], sample_weight=h_folds[f][tr])
            tef.append(model.predict(Xte))
        return cf_from_list(tef)

    cf_v178 = cf_from_list(R2_v178_te_folds)
    # use stored fold preds from last seed of each bag — recompute 1 seed for cf
    print("  computing cross_fold_r proxies...")
    tef_lgb = []
    tef_cat = []
    tef_hgb = []
    for f, (tr, va) in enumerate(folds):
        m1 = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.03,
            num_leaves=6,
            min_data_in_leaf=500,
            n_estimators=55,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            n_jobs=4,
        )
        m1.fit(X_tr[tr], g_folds[f][tr], sample_weight=h_folds[f][tr])
        tef_lgb.append(m1.predict(X_te))
        m2 = CatBoostRegressor(
            loss_function="RMSE",
            depth=3,
            l2_leaf_reg=50,
            iterations=120,
            learning_rate=0.04,
            random_seed=42,
            verbose=False,
            thread_count=4,
            min_data_in_leaf=500,
        )
        m2.fit(X_tr[tr], g_folds[f][tr], sample_weight=h_folds[f][tr])
        tef_cat.append(m2.predict(X_te))
        m3 = HistGradientBoostingRegressor(
            max_leaf_nodes=6,
            min_samples_leaf=500,
            l2_regularization=20.0,
            learning_rate=0.03,
            max_iter=50,
            random_state=42,
        )
        m3.fit(X_tr[tr], g_folds[f][tr], sample_weight=h_folds[f][tr])
        tef_hgb.append(m3.predict(X_te))
    # pro fold preds
    tef_pro = [
        unitize(
            0.45 * unitize(tef_lgb[f])
            + 0.35 * unitize(tef_cat[f])
            + 0.20 * unitize(tef_hgb[f])
        )
        for f in range(5)
    ]
    cf = {
        "R2_v178": cf_v178,
        "R2_lgb_bag": cf_from_list(tef_lgb),
        "R2_cat_bag": cf_from_list(tef_cat),
        "R2_hgb_bag": cf_from_list(tef_hgb),
        "R2_pro": cf_from_list(tef_pro),
    }
    print(f"  cross_fold_r: {cf} t={time.time()-t0:.1f}s")

    # ===== STEP 3: orth + scorecard =====
    print("=== STEP 3: forensic + Slot3 scorecard + replace-Slot5 ===")
    signals = {
        "R2_v178": (R2_v178_oof, R2_v178_test),
        "R2_lgb_bag": (R2_lgb_oof, R2_lgb_test),
        "R2_cat_bag": (R2_cat_oof, R2_cat_test),
        "R2_hgb_bag": (R2_hgb_oof, R2_hgb_test),
        "R2_pro": (R2_pro_oof, R2_pro_test),
    }
    results = {}
    oriented = {}
    for name, (vo, vt) in signals.items():
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=10.0)
        gates = forensic_gates(oo, ot, Bt9, live_resid, bn=BN9)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s3[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        oriented[name] = (u, ut)
        per = {b: score_fixed(u, zb, y, folds) for b, zb in bases_s3.items()}
        results[name] = {
            "gates": gates,
            "per_base": per,
            "pre_corr_zS3": float(np.corrcoef(vo, z_s3)[0, 1]),
            "cross_fold_r": cf[name],
        }
        print(
            f"  {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{gates['leak_col']}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) "
            f"cf_r={cf[name]:.4f} preρ={results[name]['pre_corr_zS3']:+.4f}"
        )
        for b, row in per.items():
            print(
                f"    {b}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
                f"min={row['min']:+.3f} sf={row['sf25']} t={row['t']:.2f}"
            )

    u_v178, ut_v178 = oriented["R2_v178"]
    u_pro, ut_pro = oriented["R2_pro"]

    # replace Slot5: z_base + w*u_pro vs z_base + 0.020*u_v178
    print("  --- R2_pro REPLACE Slot5 (0.020·R2_v178) ---")
    replace = {}
    for bname, zb in bases_s3.items():
        best = None
        for w in (0.015, 0.018, 0.020, 0.022, 0.025, 0.030):
            d = []
            for _, te in folds:
                s5 = zb[te] + W_R2_V178 * u_v178[te]
                sp = zb[te] + w * u_pro[te]
                d.append((roc_auc_score(y[te], sp) - roc_auc_score(y[te], s5)) * 1e5)
            row = {
                "w": w,
                "pos": int(sum(x > 0 for x in d)),
                "mean": float(np.mean(d)),
                "min": float(np.min(d)),
                "folds": [round(x, 3) for x in d],
            }
            if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
                best = row
        # also absolute vs zb
        abs_pro = score_at_w(u_pro, zb, y, folds, best["w"])
        abs_v178 = score_at_w(u_v178, zb, y, folds, W_R2_V178)
        replace[bname] = {"vs_slot5": best, "abs_pro": abs_pro, "abs_v178": abs_v178}
        print(
            f"    {bname}: REPLACE {best['pos']}/5 @{best['w']} "
            f"Δ={best['mean']:+.3f} min={best['min']:+.3f} folds={best['folds']} "
            f"| abs_pro={abs_pro['mean']:+.2f} abs_v178@0.02={abs_v178['mean']:+.2f}"
        )

    # Slot5-constructed baselines for replace check
    z_s5 = z_s3 + W_R2_V178 * u_v178
    z_slot5_twin = z_slot3_twin + W_R2_V178 * u_v178
    z_nnls_s5 = z_nnls + W_R2_V178 * u_v178
    # "beat Slot5" as: z_S3+w*pro vs existing Slot5-equivalent on same base
    # already done above via REPLACE

    beat_all = all(replace[b]["vs_slot5"]["pos"] >= 5 and replace[b]["vs_slot5"]["mean"] > 0 for b in bases_s3)
    gates_pro = results["R2_pro"]["gates"]
    stage_ok = bool(gates_pro["verified"] and beat_all)
    # also require dual-elite absolute on S3
    abs_ok = all(
        results["R2_pro"]["per_base"][b]["pos"] >= 5
        and results["R2_pro"]["per_base"][b]["sf25"] >= 19
        and results["R2_pro"]["per_base"][b]["t"] >= 3.0
        for b in bases_s3
    )
    stage_ok = stage_ok and abs_ok

    staged = []
    if stage_ok:
        wstar = replace["z_S3"]["vs_slot5"]["w"]
        # prefer absolute best w from per_base if close
        wstar = results["R2_pro"]["per_base"]["z_Slot3_twin"]["w"]
        path = SUB / f"sub_v18_2_R2_pro_w{str(wstar).replace('.','p')}_NOSUBMIT.csv"
        cand = qlock(L(slot3) + wstar * ut_pro, lv)
        # also stage Slot5-replace variant on Slot3 + w*pro (as potential Slot5 replacement)
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
        staged.append({"file": path.name, "w": wstar, "role": "R2_pro_on_Slot3"})
        # replace Slot5 file: Slot3 + w_replace * pro (not Slot5+pro)
        wr = replace["z_S3"]["vs_slot5"]["w"]
        path2 = SUB / f"sub_v18_2_REPLACE_Slot5_with_R2_pro_w{str(wr).replace('.','p')}_NOSUBMIT.csv"
        cand2 = qlock(L(slot3) + wr * ut_pro, slot5)  # qlock to Slot5 scale for fair compare
        pd.DataFrame({"id": ids, "is_fraud": cand2}).to_csv(path2, index=False)
        staged.append({"file": path2.name, "w": wr, "role": "replace_Slot5"})
        print("STAGED", [s["file"] for s in staged])
    else:
        # stage best absolute if forensic pass even if replace fails — only if verified+5/5
        if gates_pro["verified"] and all(results["R2_pro"]["per_base"][b]["pos"] >= 5 for b in bases_s3):
            wstar = results["R2_pro"]["per_base"]["z_Slot3_twin"]["w"]
            path = SUB / f"sub_v18_2_R2_pro_w{str(wstar).replace('.','p')}_SOFT_NOSUBMIT.csv"
            cand = qlock(L(slot3) + wstar * ut_pro, lv)
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
            staged.append({"file": path.name, "w": wstar, "role": "soft_forensic_ok_replace_fail"})
            print("SOFT_STAGED", path.name)
        else:
            print("NO STAGE")

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "v178_feat_names": v178_feat_names,
            "n_v178_feats": len(v178_feat_names),
            "n_X_raw": int(X_tr.shape[1]),
            "top20_gain": top20_gain,
            "top20_split": top20_split,
            "rho_rebuilt_cached_R2": float(np.corrcoef(R2_v178_oof, cached["R2_oof"])[0, 1]),
            "results": results,
            "replace_vs_slot5": replace,
            "beat_slot5_all3": beat_all,
            "stage_ok": stage_ok,
            "staged": staged,
            "elapsed_s": time.time() - t0,
            "bag_counts": {"lgb": n_lgb, "cat": n_cat, "hgb": n_hgb},
        }
    )
    OUT.joinpath("v18_2.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUT / "v18_2_r2_pro.npz",
        R2_v178_oof=R2_v178_oof,
        R2_v178_test=R2_v178_test,
        R2_lgb_oof=R2_lgb_oof,
        R2_lgb_test=R2_lgb_test,
        R2_cat_oof=R2_cat_oof,
        R2_cat_test=R2_cat_test,
        R2_hgb_oof=R2_hgb_oof,
        R2_hgb_test=R2_hgb_test,
        R2_pro_oof=R2_pro_oof,
        R2_pro_test=R2_pro_test,
        u_v178_oof=u_v178,
        u_v178_test=ut_v178,
        u_pro_oof=u_pro,
        u_pro_test=ut_pro,
        feat_names=np.array(feat_names, dtype=object),
        v178_feat_names=np.array(v178_feat_names, dtype=object),
        gain=gain,
        split=split,
    )
    print("WROTE", OUT / "v18_2.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
