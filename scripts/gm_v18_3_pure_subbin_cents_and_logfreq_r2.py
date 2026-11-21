#!/usr/bin/env python3
"""GM V18.3 — Pure sub-bin cents/logfreq R2; exact45 seed bag.

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU <40s target.
"""
from __future__ import annotations

import json
import time
from collections import Counter
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
FLOAT_FREQ_COLS = [
    "amount_usd",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
    "distance_from_home_km",
    "hours_since_last_txn",
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


def replace_vs(u_new, u_old, zb, y, folds, w_new, w_old=W_R2):
    d = []
    for _, te in folds:
        sold = zb[te] + w_old * u_old[te]
        snew = zb[te] + w_new * u_new[te]
        d.append((roc_auc_score(y[te], snew) - roc_auc_score(y[te], sold)) * 1e5)
    return {
        "w_new": w_new,
        "pos": int(sum(x > 0 for x in d)),
        "mean": float(np.mean(d)),
        "min": float(np.min(d)),
        "folds": [round(x, 3) for x in d],
    }


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


def exact_logfreq(tr_vals, te_vals):
    allv = np.concatenate([tr_vals, te_vals])
    _, inv = np.unique(allv, return_inverse=True)
    cnt = np.bincount(inv)
    logf = np.log1p(cnt[inv])
    return logf[: len(tr_vals)], logf[len(tr_vals) :]


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

    # cached R2_v178 + orth
    res = np.load(OUT / "v17_8_residuals.npz")
    R2_v178_oof, R2_v178_test = res["R2_oof"], res["R2_test"]
    oo2, ot2, _ = frozen_lofo_orth(R2_v178_oof, R2_v178_test, Bo9, Bt9, folds, ridge_alpha=10.0)
    u_v178 = unitize(oo2)
    if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u_v178[te0]) < roc_auc_score(
        y[te0], z_s3[te0] - 0.01 * u_v178[te0]
    ):
        u_v178, ot2 = -u_v178, -ot2
    ut_v178 = unitize(ot2)

    z_s5 = z_s3 + W_R2 * u_v178
    z_slot5_twin = z_slot3_twin + W_R2 * u_v178
    z_nnls_s5 = z_nnls + W_R2 * u_v178
    Bo10 = np.column_stack([Bo9, u_v178])
    Bt10 = np.column_stack([Bt9, ut_v178])
    bn10 = BN9 + ["R2_orth"]
    bases_s5 = {"z_S5": z_s5, "z_Slot5_twin": z_slot5_twin, "z_NNLS_S5": z_nnls_s5}

    # calib g/h for S3 and S5
    ab_s3, ab_s5 = [], []
    for tr, _ in folds:
        c3 = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        c3.fit(z_s3[tr].reshape(-1, 1), y[tr])
        ab_s3.append((float(c3.coef_[0, 0]), float(c3.intercept_[0])))
        c5 = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        c5.fit(z_s5[tr].reshape(-1, 1), y[tr])
        ab_s5.append((float(c5.coef_[0, 0]), float(c5.intercept_[0])))
    g3, h3, g5, h5 = [], [], [], []
    for f in range(5):
        a, b = ab_s3[f]
        p = expit(a * z_s3 + b)
        g3.append(y - p)
        h3.append(np.clip(p * (1 - p), 1e-6, None))
        a, b = ab_s5[f]
        p = expit(a * z_s5 + b)
        g5.append(y - p)
        h5.append(np.clip(p * (1 - p), 1e-6, None))

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    def joint_q1000(col):
        v = np.concatenate([train[col].to_numpy(float), test[col].to_numpy(float)])
        r = rankdata(v, method="average")
        q = np.clip((r / (len(v) + 1e-12) * 1000).astype(np.int64), 0, 999)
        return q[:n_tr], q[n_tr:]

    # ===== V17.8 exact 45-col matrix =====
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
    _, inv = np.unique(amt_all, return_inverse=True)
    freq = np.log1p(np.bincount(inv)[inv])
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

    v178_key_names = [
        f"{c}_{s}" for c in NUM_COLS for s in ("cents", "tenth", "ones")
    ] + [
        f"{c}_q1000"
        for c in ("amount_usd", "account_balance_usd", "distance_from_home_km", "velocity_score")
    ] + ["amt_freq_bin", "amt_orig_bin"] + [
        "merchant_category_x_amtones",
        "channel_x_amtones",
        "device_type_x_amtones",
    ]

    def build_X45(split="train"):
        cols = []
        src = keys_tr if split == "train" else keys_te
        for kn in v178_key_names:
            cols.append(src[kn].astype(np.float64))
        if split == "train":
            cols.append(feat_freq_tr)
            for c in ("amount_usd", "velocity_score", "merchant_risk_score", "distance_from_home_km"):
                cols.append(train[c].to_numpy(float))
            cols.append(orig_cnt[0])
        else:
            cols.append(feat_freq_te)
            for c in ("amount_usd", "velocity_score", "merchant_risk_score", "distance_from_home_km"):
                cols.append(test[c].to_numpy(float))
            cols.append(orig_cnt[1])
        return np.column_stack(cols)

    X45_tr = build_X45("train").astype(np.float32)
    X45_te = build_X45("test").astype(np.float32)
    print(f"X45={X45_tr.shape} t={time.time()-t0:.1f}s")

    # ===== STEP 1: exact45 5-seed bag =====
    print("=== STEP 1: exact45 5-seed bag ===")
    seeds = [0, 42, 2026, 7, 99]
    bag_oof = np.zeros(n_tr)
    bag_te = []
    for seed in seeds:
        oof = np.zeros(n_tr)
        tef = []
        for f, (tr, va) in enumerate(folds):
            model = lgb.LGBMRegressor(
                objective="regression",
                learning_rate=0.03,
                num_leaves=6,
                min_data_in_leaf=500,
                n_estimators=40,
                subsample=1.0,
                colsample_bytree=1.0,
                colsample_bynode=0.85,
                reg_lambda=1.0,
                random_state=seed,
                verbosity=-1,
                n_jobs=-1,
            )
            model.fit(X45_tr[tr], g3[f][tr], sample_weight=h3[f][tr])
            oof[va] = model.predict(X45_tr[va])
            tef.append(model.predict(X45_te))
        bag_oof += oof
        bag_te.append(np.mean(np.stack(tef, 0), 0))
    bag_oof /= len(seeds)
    bag_te_m = np.mean(np.stack(bag_te, 0), 0)
    # average with cached V17.8
    R2_exact45_bag_oof = unitize(0.5 * unitize(bag_oof) + 0.5 * unitize(R2_v178_oof))
    R2_exact45_bag_te = unitize(0.5 * unitize(bag_te_m) + 0.5 * unitize(R2_v178_test))
    print(
        f"  ρ(bag,cached)={np.corrcoef(bag_oof,R2_v178_oof)[0,1]:.4f} "
        f"ρ(exact45_bag,cached)={np.corrcoef(R2_exact45_bag_oof,R2_v178_oof)[0,1]:.4f} "
        f"t={time.time()-t0:.1f}s"
    )

    # ===== STEP 2: X_subbin =====
    print("=== STEP 2: X_subbin + R2_subbin ===")
    feat_names = []
    cols_tr, cols_te = [], []

    def add(name, a, b):
        feat_names.append(name)
        cols_tr.append(np.asarray(a, float))
        cols_te.append(np.asarray(b, float))

    # exact logfreq for float cols
    cents = {}
    for c in FLOAT_FREQ_COLS:
        vtr = np.round(train[c].to_numpy(float), 8)
        vte = np.round(test[c].to_numpy(float), 8)
        lf_tr, lf_te = exact_logfreq(vtr.astype(str), vte.astype(str))
        add(f"logfreq_{c}", lf_tr, lf_te)
        # also cents-key frequency
        ctr = (np.round(np.abs(train[c].to_numpy(float)) * 100).astype(np.int64) % 100) % 100
        cte = (np.round(np.abs(test[c].to_numpy(float)) * 100).astype(np.int64) % 100) % 100
        cents[c] = (ctr, cte)
        lf_tr, lf_te = exact_logfreq(ctr.astype(str), cte.astype(str))
        add(f"logfreq_{c}_centsval", lf_tr, lf_te)

    # pair freq (amount_cents, balance_cents)
    ac_tr, ac_te = cents["amount_usd"]
    bc_tr, bc_te = cents["account_balance_usd"]
    pair_tr = ac_tr.astype(np.int64) * 100 + bc_tr.astype(np.int64)
    pair_te = ac_te.astype(np.int64) * 100 + bc_te.astype(np.int64)
    lf_tr, lf_te = exact_logfreq(pair_tr.astype(str), pair_te.astype(str))
    add("logfreq_amt_cents_x_bal_cents", lf_tr, lf_te)

    # (merchant_category, amount_usd) exact pair freq
    mer_tr = train["merchant_category"].astype(str).to_numpy()
    mer_te = test["merchant_category"].astype(str).to_numpy()
    amt_tr = np.round(train["amount_usd"].to_numpy(float), 8).astype(str)
    amt_te = np.round(test["amount_usd"].to_numpy(float), 8).astype(str)
    mp_tr = np.char.add(np.char.add(mer_tr, "|"), amt_tr)
    mp_te = np.char.add(np.char.add(mer_te, "|"), amt_te)
    lf_tr, lf_te = exact_logfreq(mp_tr, mp_te)
    add("logfreq_merchant_x_amount", lf_tr, lf_te)

    # digits / indicators for continuous floats (no raw float, no q1000)
    for c in FLOAT_FREQ_COLS:
        vtr, vte = train[c].to_numpy(float), test[c].to_numpy(float)
        ctr = (np.round(np.abs(vtr) * 100).astype(np.int64) % 100) % 100
        cte = (np.round(np.abs(vte) * 100).astype(np.int64) % 100) % 100
        ttr = (np.round(np.abs(vtr) * 10).astype(np.int64) % 10) % 10
        tte = (np.round(np.abs(vte) * 10).astype(np.int64) % 10) % 10
        otr = (np.abs(vtr).astype(np.int64) % 10) % 10
        ote = (np.abs(vte).astype(np.int64) % 10) % 10
        add(f"{c}_cents", ctr, cte)
        add(f"{c}_tenth", ttr, tte)
        add(f"{c}_ones", otr, ote)
        add(f"{c}_mod5", ctr % 5, cte % 5)
        add(f"{c}_mod25", ctr % 25, cte % 25)
        add(f"{c}_is_00", (ctr == 0).astype(float), (cte == 0).astype(float))
        add(f"{c}_is_99", (ctr == 99).astype(float), (cte == 99).astype(float))
        add(f"{c}_is_round10", ((ctr % 10) == 0).astype(float), ((cte % 10) == 0).astype(float))

    add("abs_amt_bal_cents", np.abs(ac_tr - bc_tr), np.abs(ac_te - bc_te))
    add("eq_amt_bal_cents", (ac_tr == bc_tr).astype(float), (ac_te == bc_te).astype(float))

    # integer digits for discrete-ish cols
    for c in ("cvv_retry_count", "card_age_months", "txn_count_last_24h", "customer_age"):
        vtr, vte = train[c].to_numpy(float), test[c].to_numpy(float)
        add(f"{c}_ones", (np.abs(vtr).astype(np.int64) % 10) % 10, (np.abs(vte).astype(np.int64) % 10) % 10)
        add(f"{c}_mod5", (np.abs(vtr).astype(np.int64) % 5) % 5, (np.abs(vte).astype(np.int64) % 5) % 5)

    # cat × cents / ones
    cat_codes = {}
    for cat in ("merchant_category", "channel", "device_type"):
        codes, _ = pd.factorize(pd.concat([train[cat], test[cat]], ignore_index=True))
        cat_codes[cat] = (codes[:n_tr], codes[n_tr:])

    cross_specs = [
        ("merchant_category", "account_balance_usd_cents", bc_tr, bc_te),
        ("merchant_category", "amount_usd_cents", ac_tr, ac_te),
        ("merchant_category", "amount_usd_ones", keys_tr["amount_usd_ones"], keys_te["amount_usd_ones"]),
        ("merchant_category", "account_balance_usd_ones", keys_tr["account_balance_usd_ones"], keys_te["account_balance_usd_ones"]),
        ("channel", "account_balance_usd_cents", bc_tr, bc_te),
        ("channel", "amount_usd_cents", ac_tr, ac_te),
        ("channel", "amount_usd_ones", keys_tr["amount_usd_ones"], keys_te["amount_usd_ones"]),
        ("device_type", "account_balance_usd_cents", bc_tr, bc_te),
        ("device_type", "amount_usd_cents", ac_tr, ac_te),
        ("device_type", "amount_usd_ones", keys_tr["amount_usd_ones"], keys_te["amount_usd_ones"]),
        ("merchant_category", "cvv_retry_count_ones", keys_tr["cvv_retry_count_ones"], keys_te["cvv_retry_count_ones"]),
        ("merchant_category", "card_age_months_ones", keys_tr["card_age_months_ones"], keys_te["card_age_months_ones"]),
        ("channel", "cvv_retry_count_ones", keys_tr["cvv_retry_count_ones"], keys_te["cvv_retry_count_ones"]),
        ("device_type", "card_age_months_ones", keys_tr["card_age_months_ones"], keys_te["card_age_months_ones"]),
    ]
    for cat, dname, dtr, dte in cross_specs:
        ctr, cte = cat_codes[cat]
        # pack: assume digit < 100
        add(f"{cat}_x_{dname}", ctr.astype(np.int64) * 100 + dtr.astype(np.int64), cte.astype(np.int64) * 100 + dte.astype(np.int64))

    Xsub_tr = np.column_stack(cols_tr).astype(np.float32)
    Xsub_te = np.column_stack(cols_te).astype(np.float32)
    print(f"  X_subbin={Xsub_tr.shape} cols={len(feat_names)} t={time.time()-t0:.1f}s")

    def fit_lgb_bag(Xtr, Xte, g_list, h_list, seeds=(42, 2026, 7), n_est=45):
        oof = np.zeros(n_tr)
        te_acc = []
        n = 0
        for seed in seeds:
            oo = np.zeros(n_tr)
            tef = []
            for f, (tr, va) in enumerate(folds):
                model = lgb.LGBMRegressor(
                    objective="regression",
                    learning_rate=0.03,
                    num_leaves=6,
                    min_data_in_leaf=500,
                    n_estimators=n_est,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    colsample_bynode=0.85,
                    reg_lambda=1.0,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=-1,
                )
                model.fit(Xtr[tr], g_list[f][tr], sample_weight=h_list[f][tr])
                oo[va] = model.predict(Xtr[va])
                tef.append(model.predict(Xte))
            oof += oo
            te_acc.append(np.mean(np.stack(tef, 0), 0))
            n += 1
        return oof / n, np.mean(np.stack(te_acc, 0), 0)

    R2_sub_S5_oof, R2_sub_S5_te = fit_lgb_bag(Xsub_tr, Xsub_te, g5, h5)
    R2_sub_S3_oof, R2_sub_S3_te = fit_lgb_bag(Xsub_tr, Xsub_te, g3, h3)
    print(f"  subbin fits done t={time.time()-t0:.1f}s")

    # blends on S3
    blends = {
        "R2_subbin_plus_v178_a0p3": (
            unitize(unitize(R2_v178_oof) + 0.3 * unitize(R2_sub_S3_oof)),
            unitize(unitize(R2_v178_test) + 0.3 * unitize(R2_sub_S3_te)),
        ),
        "R2_subbin_plus_v178_a0p5": (
            unitize(unitize(R2_v178_oof) + 0.5 * unitize(R2_sub_S3_oof)),
            unitize(unitize(R2_v178_test) + 0.5 * unitize(R2_sub_S3_te)),
        ),
    }

    # ===== STEP 3: scorecard =====
    print("=== STEP 3: scorecard ===")
    results = {}
    oriented = {}

    def eval_signal(name, vo, vt, Bo, Bt, bn, bases, mode="add", u_ref=None):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo, Bt, folds, ridge_alpha=10.0)
        gates = forensic_gates(oo, ot, Bt, live_resid, bn=bn)
        # orient vs primary base
        z0 = list(bases.values())[0]
        u = unitize(oo)
        if roc_auc_score(y[te0], z0[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z0[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        oriented[name] = (u, ut)
        per = {b: score_fixed(u, zb, y, folds) for b, zb in bases.items()}
        pre = float(np.corrcoef(vo, z_s3 if "S5" not in name else z_s5)[0, 1])
        # replace metrics
        replace = None
        if mode == "replace" and u_ref is not None:
            replace = {}
            for b, zb in bases.items():
                best = None
                for w in (0.015, 0.018, 0.020, 0.022, 0.025):
                    row = replace_vs(u, u_ref, zb, y, folds, w)
                    if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
                        best = row
                replace[b] = best
        elif mode == "addon":
            # already on S5 bases — absolute lift is the addon score
            replace = None
        results[name] = {
            "gates": gates,
            "per_base": per,
            "pre_corr": pre,
            "replace": replace,
            "mode": mode,
        }
        print(
            f"  {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{gates['leak_col']}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) preρ={pre:+.4f}"
        )
        for b, row in per.items():
            print(
                f"    {b}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
                f"min={row['min']:+.3f} sf={row['sf25']} t={row['t']:.2f}"
            )
        if replace:
            for b, row in replace.items():
                print(
                    f"    REPLACE[{b}]: {row['pos']}/5 @{row['w_new']} "
                    f"Δ={row['mean']:+.3f} min={row['min']:+.3f} folds={row['folds']}"
                )

    # R2_v178 baseline ref on S3
    eval_signal("R2_v178", R2_v178_oof, R2_v178_test, Bo9, Bt9, BN9, bases_s3, mode="add")
    # exact45 bag replace
    eval_signal(
        "R2_exact45_bag",
        R2_exact45_bag_oof,
        R2_exact45_bag_te,
        Bo9,
        Bt9,
        BN9,
        bases_s3,
        mode="replace",
        u_ref=u_v178,
    )
    # also raw 5-seed bag alone (no blend with cached)
    eval_signal(
        "R2_exact45_seeds_only",
        bag_oof,
        bag_te_m,
        Bo9,
        Bt9,
        BN9,
        bases_s3,
        mode="replace",
        u_ref=u_v178,
    )
    # subbin on S5 addon
    eval_signal(
        "R2_subbin_S5",
        R2_sub_S5_oof,
        R2_sub_S5_te,
        Bo10,
        Bt10,
        bn10,
        bases_s5,
        mode="addon",
    )
    # subbin S3 alone
    eval_signal(
        "R2_subbin_S3",
        R2_sub_S3_oof,
        R2_sub_S3_te,
        Bo9,
        Bt9,
        BN9,
        bases_s3,
        mode="replace",
        u_ref=u_v178,
    )
    for nm, (vo, vt) in blends.items():
        eval_signal(nm, vo, vt, Bo9, Bt9, BN9, bases_s3, mode="replace", u_ref=u_v178)

    # staging rule
    staged = []

    def passes_elite(per):
        return all(
            per[b]["pos"] >= 5 and per[b]["sf25"] >= 19 and per[b]["t"] >= 3.0 and per[b]["mean"] > 0
            for b in per
        )

    def beats_slot5(rep):
        if not rep:
            return False
        return all(rep[b]["pos"] >= 5 and rep[b]["mean"] > 0 for b in rep)

    for name, r in results.items():
        if name == "R2_v178":
            continue
        gates = r["gates"]
        if not gates["verified"]:
            continue
        if r["mode"] == "addon":
            # addon to Slot5: need elite on all S5 bases
            if passes_elite(r["per_base"]):
                u, ut = oriented[name]
                w = r["per_base"]["z_Slot5_twin"]["w"]
                path = SUB / f"sub_v18_3_{name}_w{str(w).replace('.','p')}_NOSUBMIT.csv"
                cand = qlock(L(slot5) + w * ut, lv)
                pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
                staged.append({"file": path.name, "signal": name, "w": w, "role": "addon_S5"})
                print("STAGED", path.name)
        elif r["mode"] == "replace":
            if passes_elite(r["per_base"]) and beats_slot5(r["replace"]):
                u, ut = oriented[name]
                # use replace w on S3 from z_S3
                wr = r["replace"]["z_S3"]["w_new"]
                path = SUB / f"sub_v18_3_{name}_REPLACE_w{str(wr).replace('.','p')}_NOSUBMIT.csv"
                cand = qlock(L(slot3) + wr * ut, slot5)
                pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
                staged.append({"file": path.name, "signal": name, "w": wr, "role": "replace_S5"})
                print("STAGED", path.name)

    if not staged:
        print("NO STAGE")

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "n_X45": int(X45_tr.shape[1]),
            "n_X_subbin": int(Xsub_tr.shape[1]),
            "subbin_feats": feat_names,
            "results": results,
            "staged": staged,
            "elapsed_s": time.time() - t0,
            "stage_rule": "F1∧F2∧F3 ∧ (addon: elite on S5 | replace: elite on S3 ∧ beat Slot5 all3)",
        }
    )
    OUT.joinpath("v18_3.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUT / "v18_3_residuals.npz",
        R2_exact45_bag_oof=R2_exact45_bag_oof,
        R2_exact45_bag_te=R2_exact45_bag_te,
        bag_oof=bag_oof,
        bag_te=bag_te_m,
        R2_sub_S5_oof=R2_sub_S5_oof,
        R2_sub_S5_te=R2_sub_S5_te,
        R2_sub_S3_oof=R2_sub_S3_oof,
        R2_sub_S3_te=R2_sub_S3_te,
        feat_names=np.array(feat_names, dtype=object),
    )
    print("WROTE", OUT / "v18_3.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
