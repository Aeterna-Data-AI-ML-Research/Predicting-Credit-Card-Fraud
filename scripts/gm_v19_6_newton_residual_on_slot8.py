#!/usr/bin/env python3
"""GM V19.6 — Newton residual booster on Slot8 ATH (same recipe that made Slot5).

Train shallow LGB + CatBoost on calibrated residual of Slot8 OOF,
orth to Bo11=[Bo9,R2,Cat_obliv,U5], H2H vs Slot8. Submit Slot9 if gates clear.
0 GPU pods.
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
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
TMP15, TMP16 = Path("/tmp/s6e11_v15"), Path("/tmp/s6e11_v16")
OUT, SUB, DATA = TMP16 / "out", REPO / "submissions", REPO / "data"
ORIG = REPO / "data/creditcard.csv"
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
NUM_COLS = [
    "amount_usd",
    "account_balance_usd",
    "distance_from_home_km",
    "velocity_score",
    "merchant_risk_score",
]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W8_O, W8_C, W8_U = 0.016, 0.012, 0.012


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
    # raw_disp: std(test)/std(oof)
    disp = float(np.std(ot) / (np.std(oo) + 1e-12))
    # F2: max |corr| with Bt columns
    ut = unitize(ot)
    leaks = [abs(float(np.corrcoef(ut, Bt[:, j])[0, 1])) for j in range(Bt.shape[1])]
    return {"raw_disp": disp, "F2": float(max(leaks)), "F1": 0.90 <= disp <= 1.10}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

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
    U5_o = L_fold(gini10(col_z(llr_o)), fold_id)
    U5_t = np.mean(np.stack([L(gini10(col_z(llr_t[f]))) for f in range(5)], 0), 0)
    Bo10 = np.column_stack([Bo9, u_r2])
    Bt10 = np.column_stack([Bt9, ut_r2])
    oo5, ot5, _ = frozen_lofo_orth(U5_o, U5_t, Bo10, Bt10, folds, ridge_alpha=1.0)
    u_u5 = unitize(oo5)
    if roc_auc_score(y[te0], (z_s3 + W8_O * u_r2)[te0] + 0.01 * u_u5[te0]) < roc_auc_score(
        y[te0], (z_s3 + W8_O * u_r2)[te0] - 0.01 * u_u5[te0]
    ):
        u_u5, ot5 = -u_u5, -ot5
    ut_u5 = unitize(ot5)

    z_s8 = z_s3 + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_t8 = z_twin + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_n8 = z_nnls + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    bases = {"z_S8": z_s8, "twin": z_t8, "nnls": z_n8}

    Bo11 = np.column_stack([Bo9, u_r2, u_cat, u_u5])
    Bt11 = np.column_stack([Bt9, ut_r2, ut_cat, ut_u5])

    print("=== V19.6 Newton residual on Slot8 ===", f"t={time.time()-t0:.1f}s")

    # ---- build tabular X (V17.8 recipe) ----
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
    freq = np.log1p(cnt[inv])
    keys_tr["amt_freq_bin"] = np.clip((freq[:n_tr] * 5).astype(np.int64), 0, 50)
    keys_te["amt_freq_bin"] = np.clip((freq[n_tr:] * 5).astype(np.int64), 0, 50)
    feat_freq_tr, feat_freq_te = freq[:n_tr], freq[n_tr:]

    orig_cnt = None
    if ORIG.exists():
        orig = pd.read_csv(ORIG, usecols=lambda c: c in ("amount_usd", "Amount", "transaction_amount", "amt"))
        oc = [c for c in orig.columns if "amount" in c.lower() or c == "amt"]
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

    def build_X(split="train"):
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
        return np.column_stack(cols)

    Xtr = build_X("train")
    Xte = build_X("test")
    print(f"  X shape={Xtr.shape} keys={len(key_names)} t={time.time()-t0:.1f}s")

    # ---- train R8_lgb + R8_cat on Slot8 residual ----
    lgb_params = dict(
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
    )

    R8_lgb_oof = np.zeros(n_tr)
    R8_lgb_te_folds = []
    for f, (tr, va) in enumerate(folds):
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s8[tr].reshape(-1, 1), y[tr])
        a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
        p = expit(a * z_s8 + b)
        g = y - p
        h = np.clip(p * (1 - p), 1e-6, None)
        bags = []
        for seed in (7, 17, 27):
            m = lgb.LGBMRegressor(**{**lgb_params, "random_state": seed})
            m.fit(Xtr[tr], g[tr], sample_weight=h[tr])
            bags.append(m)
        R8_lgb_oof[va] = np.mean([m.predict(Xtr[va]) for m in bags], 0)
        R8_lgb_te_folds.append(np.mean([m.predict(Xte) for m in bags], 0))
        print(f"  R8_lgb fold{f} t={time.time()-t0:.1f}s")
    R8_lgb_te = np.mean(np.stack(R8_lgb_te_folds, 0), 0)

    # CatBoost if available
    try:
        from catboost import CatBoostRegressor

        has_cat = True
    except ImportError:
        has_cat = False
        print("  CatBoost missing — LGB only")

    R8_cat_oof = np.zeros(n_tr)
    R8_cat_te = np.zeros(n_te)
    if has_cat:
        R8_cat_te_folds = []
        for f, (tr, va) in enumerate(folds):
            clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
            clf.fit(z_s8[tr].reshape(-1, 1), y[tr])
            a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
            p = expit(a * z_s8 + b)
            g = y - p
            h = np.clip(p * (1 - p), 1e-6, None)
            bags = []
            for seed in (11, 21, 31, 41, 51):
                m = CatBoostRegressor(
                    loss_function="RMSE",
                    learning_rate=0.05,
                    depth=3,
                    l2_leaf_reg=3.0,
                    iterations=60,
                    random_seed=seed,
                    verbose=False,
                    thread_count=4,
                    bootstrap_type="Bernoulli",
                    subsample=0.8,
                )
                m.fit(Xtr[tr], g[tr], sample_weight=h[tr])
                bags.append(m)
            R8_cat_oof[va] = np.mean([m.predict(Xtr[va]) for m in bags], 0)
            R8_cat_te_folds.append(np.mean([m.predict(Xte) for m in bags], 0))
            print(f"  R8_cat fold{f} t={time.time()-t0:.1f}s")
        R8_cat_te = np.mean(np.stack(R8_cat_te_folds, 0), 0)

    # blend 0.5/0.5 if cat
    if has_cat:
        R8_blend_oof = 0.5 * R8_lgb_oof + 0.5 * R8_cat_oof
        R8_blend_te = 0.5 * R8_lgb_te + 0.5 * R8_cat_te
    else:
        R8_blend_oof, R8_blend_te = R8_lgb_oof, R8_lgb_te

    np.savez_compressed(
        OUT / "v19_6_newton_s8.npz",
        R8_lgb_oof=R8_lgb_oof,
        R8_lgb_te=R8_lgb_te,
        R8_cat_oof=R8_cat_oof,
        R8_cat_te=R8_cat_te,
        R8_blend_oof=R8_blend_oof,
        R8_blend_te=R8_blend_te,
    )

    # orth to Bo11 and score
    families = {
        "R8_lgb": (R8_lgb_oof, R8_lgb_te),
        "R8_blend": (R8_blend_oof, R8_blend_te),
    }
    if has_cat:
        families["R8_cat"] = (R8_cat_oof, R8_cat_te)

    winners = []
    candidates = []

    def passes(per, rho, c6, c7, gates):
        return (
            all(
                per[b]["pos"] >= 5
                and per[b]["min"] > 0.05
                and per[b]["sf25"] >= 18
                and per[b]["t"] >= 2.8
                for b in per
            )
            and (0.30 <= rho <= 1.20)
            and abs(c6) < 0.20
            and abs(c7) < 0.20
            and gates["F1"]
            and gates["F2"] < 0.035
        )

    print("=== scorecard vs Slot8 ===")
    for fname, (vo, vt) in families.items():
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo11, Bt11, folds, ridge_alpha=10.0)
        gates = forensic(oo, ot, Bt11)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s8[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        print(
            f"  {fname}: corr(R2)={np.corrcoef(u,u_r2)[0,1]:+.3f} "
            f"corr(Cat)={np.corrcoef(u,u_cat)[0,1]:+.3f} "
            f"corr(U5)={np.corrcoef(u,u_u5)[0,1]:+.3f} "
            f"disp={gates['raw_disp']:.3f} F2={gates['F2']:.4f}"
        )
        for w in (0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.022, 0.025, 0.030):
            oof = {
                "z_S8": z_s8 + w * u,
                "twin": z_t8 + w * u,
                "nnls": z_n8 + w * u,
            }
            per = {b: h2h(oof[b], bases[b], y, folds) for b in bases}
            cand = qlock(L(slot8) + w * ut, lv)
            rho = one_m_rho_e4(cand, slot8)
            delta = unitize(L(cand) - L(slot8))
            c6 = float(np.corrcoef(delta, bad6)[0, 1])
            c7 = float(np.corrcoef(delta, bad7)[0, 1])
            ok = passes(per, rho, c6, c7, gates)
            row = {
                "name": f"{fname}_w{w:.3f}",
                "family": fname,
                "w": w,
                "per": per,
                "rho": rho,
                "corr6": c6,
                "corr7": c7,
                "gates": gates,
                "ok": ok,
                "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                "min_lift": float(np.min([per[b]["min"] for b in per])),
                "u": u if ok else None,
                "ut": ut if ok else None,
            }
            mark = "★" if ok else ("~" if per["z_S8"]["pos"] >= 4 else " ")
            print(
                f"{mark} {row['name']}: 1-ρ={rho:.3f}e-4 ok={ok} | "
                + " | ".join(
                    f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f} "
                    f"sf{per[b]['sf25']} t={per[b]['t']:.2f}"
                    for b in bases
                )
            )
            candidates.append({k: v for k, v in row.items() if k not in ("u", "ut")})
            if ok:
                winners.append(row)

    # also try anchored: keep Slot8 recipe but add R8 on Slot3 bases (like Cat)
    print("-- anchored Slot3 + Slot8 recipe + R8 --")
    for fname, (vo, vt) in families.items():
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=10.0)
        gates = forensic(oo, ot, Bt9)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s8[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        for wr in (0.006, 0.008, 0.010, 0.012, 0.015):
            oof = {
                "z_S8": z_s3 + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5 + wr * u,
                "twin": z_twin + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5 + wr * u,
                "nnls": z_nnls + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5 + wr * u,
            }
            per = {b: h2h(oof[b], bases[b], y, folds) for b in bases}
            cand = qlock(
                L(slot3) + W8_O * ut_r2 + W8_C * ut_cat + W8_U * ut_u5 + wr * ut, lv
            )
            rho = one_m_rho_e4(cand, slot8)
            delta = unitize(L(cand) - L(slot8))
            c6 = float(np.corrcoef(delta, bad6)[0, 1])
            c7 = float(np.corrcoef(delta, bad7)[0, 1])
            ok = passes(per, rho, c6, c7, gates)
            row = {
                "name": f"anch_{fname}_wr{wr:.3f}",
                "family": f"anch_{fname}",
                "w": wr,
                "per": per,
                "rho": rho,
                "corr6": c6,
                "corr7": c7,
                "gates": gates,
                "ok": ok,
                "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                "min_lift": float(np.min([per[b]["min"] for b in per])),
                "u": u if ok else None,
                "ut": ut if ok else None,
                "anchored": True,
            }
            mark = "★" if ok else ("~" if per["z_S8"]["pos"] >= 4 else " ")
            print(
                f"{mark} {row['name']}: 1-ρ={rho:.3f} F2={gates['F2']:.3f} ok={ok} | "
                + " | ".join(
                    f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f} "
                    f"t={per[b]['t']:.2f}"
                    for b in bases
                )
            )
            candidates.append({k: v for k, v in row.items() if k not in ("u", "ut")})
            if ok:
                winners.append(row)

    print(f"\nWinners: {len(winners)}")
    submit = False
    pub, ref, chosen = None, None, None
    if winners:
        winners.sort(key=lambda r: (r["min_lift"], r["mean_lift"], -abs(r["rho"] - 0.50)), reverse=True)
        chosen = winners[0]
        print(
            f"CHOSEN {chosen['name']} mean={chosen['mean_lift']:+.2f} "
            f"min={chosen['min_lift']:+.2f} 1-ρ={chosen['rho']:.3f}e-4"
        )
        if chosen.get("anchored"):
            cand = qlock(
                L(slot3)
                + W8_O * ut_r2
                + W8_C * ut_cat
                + W8_U * ut_u5
                + chosen["w"] * chosen["ut"],
                lv,
            )
        else:
            cand = qlock(L(slot8) + chosen["w"] * chosen["ut"], lv)
        out_csv = SUB / f"sub_v19_6_LIVE_slot9_{chosen['name']}.csv"
        assert len(cand) == n_te and np.isfinite(cand).all()
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(out_csv, index=False)
        msg = (
            f"v19.6 Slot9: {chosen['name']} Newton-on-S8 vs ATH 0.95395 | "
            f"H2H mean={chosen['mean_lift']:+.2f}e-5 min={chosen['min_lift']:+.2f} "
            f"1-ρ={chosen['rho']:.3f}e-4"
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
                    if "v19_6" in row.get("fileName", ""):
                        ref = row.get("ref")
                        st = row.get("status", "")
                        print(f"  poll {st} ref={ref} score={row.get('publicScore')}")
                        if "complete" in st.lower() and row.get("publicScore"):
                            pub = row["publicScore"]
                            break
                if pub:
                    break
            print(f"PUBLIC_LB={pub} (ATH Slot8=0.95395)")
    else:
        print("NO WINNER — holding Slot8")

    near = sorted(
        candidates,
        key=lambda c: (c["per"]["z_S8"]["pos"], c["mean_lift"], -abs(c["rho"] - 0.50)),
        reverse=True,
    )[:15]
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
                "gates": chosen["gates"],
                "per": chosen["per"],
            },
            "public_lb": pub,
            "ref": ref,
            "near_misses": near,
            "elapsed_s": time.time() - t0,
        }
    )
    OUT.joinpath("v19_6.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v19_6.json", f"elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
