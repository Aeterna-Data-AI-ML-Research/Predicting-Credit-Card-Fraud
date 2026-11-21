#!/usr/bin/env python3
"""GM V19.12 — Strict F2<0.035: X_tab45 ⊗ U567 cross-mod + U6/U7 fill + multi-arch tab45.

Engines A/B/C only on (X_tab45, U5..U10_llr). Submit Slot9 IFF all 6 hard gates.
0 GPU pods.
"""
from __future__ import annotations

import json
import subprocess
import time
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import lightgbm as lgb  # noqa: F401 — available; XGB/Cat used below
import numpy as np
import pandas as pd
import xgboost as xgb
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
PRED = REPO / "preds"
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
SEEDS = (11, 21, 31, 41, 51)


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


def clip_z(v, lo=-3.0, hi=3.0):
    return np.clip(unitize(v), lo, hi)


def u_bank_from_llr(ll_pos, ll_neg):
    """Return dict of raw (pre-L) U5..U10 from (n, K) pos/neg logits."""
    llr = ll_pos - ll_neg
    Z = col_z(llr)
    K = Z.shape[1]
    h = K // 2
    pair_means = np.column_stack([0.5 * (Z[:, m] + Z[:, m + h]) for m in range(h)])
    pair_gaps = np.column_stack([np.abs(Z[:, m] - Z[:, m + h]) for m in range(h)])
    Zp, Zn = col_z(ll_pos), col_z(ll_neg)
    # top2 / bottom2 mean gap
    part = np.partition(Z, [1, K - 2], axis=1)
    top2 = 0.5 * (part[:, -1] + part[:, -2])
    bot2 = 0.5 * (part[:, 0] + part[:, 1])
    return {
        "U5": gini10(Z),
        "U6": pair_means.std(1),
        "U7": pair_gaps.mean(1),
        "U8": Zp.std(1) - Zn.std(1),
        "U9": top2 - bot2,
        "U10": pair_gaps.max(1),
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    # ---------- Slot8 reconstruct (exact V19.4) ----------
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
    u_lgb, ut_lgb = orient_bo9(fam1["F_lgb_ms39_oof"], fam1["F_lgb_ms39_test"])

    raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
    ll_pos_o, ll_neg_o = raw_o[:, :, 1], raw_o[:, :, 0]
    # Exact V19.4 U5 for Slot8
    Z_o = col_z(ll_pos_o - ll_neg_o)
    U5_o_raw = gini10(Z_o)
    U5_o = L_fold(U5_o_raw, fold_id)
    U5_t = np.mean(
        np.stack(
            [L(gini10(col_z(raw_t[f, :, :, 1] - raw_t[f, :, :, 0]))) for f in range(5)], 0
        ),
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
    print(f"=== V19.12 === Slot8 rebuild 1-ρ={rho_rebuild:.6e}e-4 t={time.time()-t0:.1f}s")
    assert rho_rebuild < 1e-5, rho_rebuild

    # Bo12_S8 = [Bo9, R2, Cat, U5]
    Bo12 = np.column_stack([Bo9, u_r2, u_cat, u_u5])
    Bt12 = np.column_stack([Bt9, ut_r2, ut_cat, ut_u5])

    # ---------- U5..U10 bank ----------
    print("=== Step1 U5..U10 bank ===", f"t={time.time()-t0:.1f}s")
    bank_o_raw = u_bank_from_llr(ll_pos_o, ll_neg_o)
    bank_t_raw = {k: [] for k in bank_o_raw}
    for f in range(5):
        bf = u_bank_from_llr(raw_t[f, :, :, 1], raw_t[f, :, :, 0])
        for k, v in bf.items():
            bank_t_raw[k].append(v)

    U_oriented = {}
    u_audit = {}
    for name in ("U5", "U6", "U7", "U8", "U9", "U10"):
        vo = L_fold(bank_o_raw[name], fold_id)
        vt = np.mean(np.stack([L(x) for x in bank_t_raw[name]], 0), 0)
        # orth vs Bo9 for forensic, then vs Bo12 for residual use
        oo9, ot9, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=1.0)
        g9 = forensic(oo9, ot9, Bt9)
        oo12, ot12, _ = frozen_lofo_orth(vo, vt, Bo12, Bt12, folds, ridge_alpha=10.0)
        g12 = forensic(oo12, ot12, Bt12)
        u = unitize(oo12)
        if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s8[te0] - 0.01 * u[te0]
        ):
            u, ot12 = -u, -ot12
        ut = unitize(ot12)
        # fold profile vs Slot8 at tiny w
        prof = h2h(z_s8 + 0.010 * u, z_s8, y, folds)
        c6 = float(np.corrcoef(ut, bad6)[0, 1])
        c7 = float(np.corrcoef(ut, bad7)[0, 1])
        U_oriented[name] = (u, ut, vo, vt)
        u_audit[name] = {
            "disp_bo9": g9["raw_disp"],
            "F2_bo9": g9["F2"],
            "disp_bo12": g12["raw_disp"],
            "F2_bo12": g12["F2"],
            "KS": g12["raw_KS"],
            "c6": c6,
            "c7": c7,
            "folds": prof["folds"],
            "mean": prof["mean"],
            "pos": prof["pos"],
        }
        print(
            f"  {name}: F2_Bo9={g9['F2']:.4f} F2_Bo12={g12['F2']:.4f} "
            f"disp={g12['raw_disp']:.3f} c7={c7:+.3f} folds={prof['folds']} m={prof['mean']:+.2f}"
        )

    # pre-orth U raw unitized for cross-mod (main-effect free products)
    def unit_pair(vo, vt):
        return unitize(vo), unitize(vt)

    u5o_raw, u5t_raw = unit_pair(*U_oriented["U5"][2:])
    # re-get raw unitized before Bo12 orth for modulation bases
    u_u5_r, ut_u5_r = unitize(U_oriented["U5"][2]), unitize(U_oriented["U5"][3])
    u_u6_r, ut_u6_r = unitize(U_oriented["U6"][2]), unitize(U_oriented["U6"][3])
    u_u7_r, ut_u7_r = unitize(U_oriented["U7"][2]), unitize(U_oriented["U7"][3])
    u_u8_r, ut_u8_r = unitize(U_oriented["U8"][2]), unitize(U_oriented["U8"][3])
    u_u9_r, ut_u9_r = unitize(U_oriented["U9"][2]), unitize(U_oriented["U9"][3])

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
            print(f"  orig20k hits mean_tr={orig_cnt_tr.mean():.3f}")

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
    print(f"  X_tab45 cols={X45_tr.shape[1]} keys={len(key_names)} t={time.time()-t0:.1f}s")

    # ---------- Platt residuals ----------
    def platt_gh(z):
        g5, h5 = [], []
        for tr, _ in folds:
            clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
            clf.fit(z[tr].reshape(-1, 1), y[tr])
            a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
            p = expit(a * z + b)
            g5.append(y - p)
            h5.append(np.clip(p * (1 - p), 1e-6, None))
        return g5, h5

    g_s3, h_s3 = platt_gh(z_s3)
    g_s8, h_s8 = platt_gh(z_s8)

    # ---------- Engine A: U complementary ----------
    print("=== Engine A U_comp ===", f"t={time.time()-t0:.1f}s")
    # blend raw U then orth to Bo12
    blends_A = {}
    for tag, w in [
        ("U6_only", {"U6": 1.0}),
        ("U7_only", {"U7": 1.0}),
        ("U67", {"U6": 0.55, "U7": 0.45}),
        ("U6789", {"U6": 0.35, "U7": 0.30, "U8": 0.20, "U9": 0.15}),
        ("U567_comp", {"U5": 0.45, "U6": 0.25, "U7": 0.20, "U9": 0.10}),
        ("U5678", {"U5": 0.40, "U6": 0.25, "U7": 0.20, "U8": 0.15}),
    ]:
        vo = np.zeros(n_tr)
        vt = np.zeros(n_te)
        for nm, ww in w.items():
            vo = vo + ww * unitize(U_oriented[nm][2])
            vt = vt + ww * unitize(U_oriented[nm][3])
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo12, Bt12, folds, ridge_alpha=10.0)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s8[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        blends_A[tag] = (u, unitize(ot))
        g = forensic(oo, ot, Bt12)
        print(f"  {tag}: F2={g['F2']:.4f} disp={g['raw_disp']:.3f} KS={g['raw_KS']:.3f}")

    # replace U5 inside Slot8 recipe with U567_comp (Mode2-lite)
    u_comp, ut_comp = blends_A["U567_comp"]

    # ---------- Engine B: multi-arch on X_tab45 ----------
    print("=== Engine B tab45 multi-arch ===", f"t={time.time()-t0:.1f}s")

    def fit_xgb_bag(Xtr, Xte, g5, h5, tag, depths=(2, 3)):
        oof_acc, te_acc = [], []
        for depth in depths:
            for seed in SEEDS:
                oo = np.zeros(n_tr)
                tef = []
                for f, (tr, va) in enumerate(folds):
                    m = xgb.XGBRegressor(
                        max_depth=depth,
                        learning_rate=0.03,
                        n_estimators=55,
                        min_child_weight=500,
                        subsample=0.8,
                        colsample_bytree=0.75,
                        reg_lambda=40.0,
                        reg_alpha=10.0,
                        objective="reg:squarederror",
                        tree_method="hist",
                        n_jobs=4,
                        random_state=seed,
                        verbosity=0,
                    )
                    m.fit(Xtr[tr], g5[f][tr], sample_weight=h5[f][tr])
                    oo[va] = m.predict(Xtr[va])
                    tef.append(m.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        oof = np.mean(np.stack(oof_acc, 0), 0)
        te = np.mean(np.stack(te_acc, 0), 0)
        print(f"  {tag}: n={len(oof_acc)} t={time.time()-t0:.1f}s")
        return oof, te

    def fit_cat_bag(Xtr, Xte, g5, h5, tag, depths=(4, 5), iterations=140, l2=40.0, lr=0.035):
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
                        learning_rate=lr,
                        random_seed=seed,
                        verbose=False,
                        thread_count=4,
                        bootstrap_type="Bernoulli",
                        subsample=0.8,
                        rsm=0.75,
                    )
                    m.fit(Xtr[tr], g5[f][tr], sample_weight=h5[f][tr])
                    oo[va] = m.predict(Xtr[va])
                    tef.append(m.predict(Xte))
                oof_acc.append(oo)
                te_acc.append(np.mean(np.stack(tef, 0), 0))
        oof = np.mean(np.stack(oof_acc, 0), 0)
        te = np.mean(np.stack(te_acc, 0), 0)
        print(f"  {tag}: n={len(oof_acc)} t={time.time()-t0:.1f}s")
        return oof, te

    # Train on (g_S8,h_S8) primarily; also S3 for Mode2
    Fx_s8_o, Fx_s8_t = fit_xgb_bag(X45_tr, X45_te, g_s8, h_s8, "F_xgb_d3_45_S8")
    Fc_s8_o, Fc_s8_t = fit_cat_bag(X45_tr, X45_te, g_s8, h_s8, "F_cat_d5_45_S8")
    Fx_s3_o, Fx_s3_t = fit_xgb_bag(X45_tr, X45_te, g_s3, h_s3, "F_xgb_d3_45_S3")
    Fc_s3_o, Fc_s3_t = fit_cat_bag(
        X45_tr, X45_te, g_s3, h_s3, "F_cat_d5_45_S3", depths=(4, 5), iterations=120
    )

    def orient_bo12(vo, vt, tag):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo12, Bt12, folds, ridge_alpha=10.0)
        g = forensic(oo, ot, Bt12)
        g9 = forensic(*frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=10.0)[:2], Bt9)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s8[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        print(
            f"  {tag} orth: F2_Bo9={g9['F2']:.4f} F2_Bo12={g['F2']:.4f} "
            f"disp={g['raw_disp']:.3f} KS={g['raw_KS']:.3f}"
        )
        return u, unitize(ot), g, g9

    families_B = {}
    for tag, vo, vt in [
        ("F_xgb_S8", Fx_s8_o, Fx_s8_t),
        ("F_cat_S8", Fc_s8_o, Fc_s8_t),
        ("F_xgb_S3", Fx_s3_o, Fx_s3_t),
        ("F_cat_S3", Fc_s3_o, Fc_s3_t),
        ("F_xc_S8", 0.5 * Fx_s8_o + 0.5 * Fc_s8_o, 0.5 * Fx_s8_t + 0.5 * Fc_s8_t),
        ("F_xc_S3", 0.5 * Fx_s3_o + 0.5 * Fc_s3_o, 0.5 * Fx_s3_t + 0.5 * Fc_s3_t),
    ]:
        families_B[tag] = orient_bo12(vo, vt, tag)

    # ---------- Engine C: X_mod36 ----------
    print("=== Engine C X_mod36 ===", f"t={time.time()-t0:.1f}s")

    def mod_cols(tab_signals_o, tab_signals_t, u_list_o, u_list_t):
        cols_o, cols_t = [], []
        for to, tt in zip(tab_signals_o, tab_signals_t):
            zt_o, zt_t = clip_z(to), clip_z(tt)
            for uo, ut in zip(u_list_o, u_list_t):
                zu_o, zu_t = clip_z(uo), clip_z(ut)
                cols_o.append(zt_o * zu_o)
                cols_t.append(zt_t * zu_t)
        return cols_o, cols_t

    # model-level (12+): R2, Cat, LGB, R2-Cat × U5,U6,U7 + gated
    tab_sig_o = [u_r2, u_cat, u_lgb, unitize(u_r2 - u_cat)]
    tab_sig_t = [ut_r2, ut_cat, ut_lgb, unitize(ut_r2 - ut_cat)]
    u_list_o = [u_u5_r, u_u6_r, u_u7_r]
    u_list_t = [ut_u5_r, ut_u6_r, ut_u7_r]
    mo, mt = mod_cols(tab_sig_o, tab_sig_t, u_list_o, u_list_t)
    # gated extras
    mo.append(clip_z(u_cat) * np.tanh(clip_z(u_u5_r)))
    mt.append(clip_z(ut_cat) * np.tanh(clip_z(ut_u5_r)))
    mo.append(clip_z(u_r2) * np.tanh(clip_z(u_u6_r)))
    mt.append(clip_z(ut_r2) * np.tanh(clip_z(ut_u6_r)))

    # feature-level top-8 sub-bin × U567
    want = [
        "amt_logfreq",
        "account_balance_usd_cents",
        "amount_usd_cents",
        "merchant_category_x_amtones",
        "velocity_score_q1000",
        "merchant_risk_score_cents",
        "cvv_retry_count_ones",
        "orig20k_hit_count",
    ]
    name_to_idx = {n: i for i, n in enumerate(x45_names)}
    feat_o, feat_t = [], []
    for wn in want:
        if wn not in name_to_idx:
            # try aliases
            aliases = {
                "velocity_score_q1000": "velocity_score_q1000",
                "velocity_q1000": "velocity_score_q1000",
            }
            wn2 = aliases.get(wn, wn)
            if wn2 not in name_to_idx:
                print(f"  WARN missing col {wn}")
                continue
            wn = wn2
        j = name_to_idx[wn]
        feat_o.append(X45_tr[:, j])
        feat_t.append(X45_te[:, j])
    fo, ft = mod_cols(feat_o, feat_t, u_list_o, u_list_t)
    mo.extend(fo)
    mt.extend(ft)

    Xmod_o = np.column_stack(mo).astype(np.float64)
    Xmod_t = np.column_stack(mt).astype(np.float64)
    # drop any near-zero variance
    keep = Xmod_o.std(0) > 1e-8
    Xmod_o, Xmod_t = Xmod_o[:, keep], Xmod_t[:, keep]
    print(f"  X_mod cols={Xmod_o.shape[1]} (kept {keep.sum()})")

    # Ridge on X_mod → g_S8
    M_ridge_o = np.zeros(n_tr)
    M_ridge_t_folds = []
    for f, (tr, va) in enumerate(folds):
        sw = np.sqrt(h_s8[f][tr])
        lr = Ridge(alpha=1000.0, fit_intercept=True)
        lr.fit(Xmod_o[tr] * sw[:, None], g_s8[f][tr] * sw)
        M_ridge_o[va] = lr.predict(Xmod_o[va])
        M_ridge_t_folds.append(lr.predict(Xmod_t))
    M_ridge_t = np.mean(np.stack(M_ridge_t_folds, 0), 0)

    # shallow Cat on X_mod
    Mc_o, Mc_t = fit_cat_bag(
        Xmod_o.astype(np.float32),
        Xmod_t.astype(np.float32),
        g_s8,
        h_s8,
        "M_mod_cat",
        depths=(3,),
        iterations=100,
        l2=50.0,
        lr=0.03,
    )

    families_C = {}
    for tag, vo, vt in [
        ("M_mod_ridge", M_ridge_o, M_ridge_t),
        ("M_mod_cat", Mc_o, Mc_t),
        ("M_mod_blend", 0.55 * M_ridge_o + 0.45 * Mc_o, 0.55 * M_ridge_t + 0.45 * Mc_t),
    ]:
        families_C[tag] = orient_bo12(vo, vt, tag)

    np.savez_compressed(
        OUT / "v19_12_families.npz",
        **{f"{k}_oof": v[0] for k, v in {**blends_A, **{kk: vv[:2] for kk, vv in families_B.items()}, **{kk: vv[:2] for kk, vv in families_C.items()}}.items()},
        **{f"{k}_test": v[1] for k, v in {**blends_A, **{kk: vv[:2] for kk, vv in families_B.items()}, **{kk: vv[:2] for kk, vv in families_C.items()}}.items()},
    )

    # ---------- Step3 audit ----------
    print("=== Step3 H2H vs Slot8 ===", f"t={time.time()-t0:.1f}s")
    candidates = []
    winners = []

    def push_cand(name, u, ut, gates=None, g9=None, meta=None):
        if gates is None:
            oo, ot, _ = frozen_lofo_orth(u, ut, Bo12, Bt12, folds, ridge_alpha=10.0)
            # u/ut already orth unitized; recompute forensic from raw-ish
            gates = {
                "raw_disp": float(np.std(ut) / (np.std(u) + 1e-12)),
                "raw_KS": float(ks_2samp(unitize(u), unitize(ut)).statistic),
                "F2": forensic(u, ut, Bt12)["F2"],
                "F1": True,
            }
            # better: use provided
        f2_bo12 = gates["F2"]
        f2_bo9 = g9["F2"] if g9 else f2_bo12
        live_resid = unitize(L(slot8) - L(slot5))
        f3 = abs(float(np.corrcoef(unitize(ut), live_resid)[0, 1]))
        print(
            f"  {name}: disp={gates['raw_disp']:.3f} KS={gates.get('raw_KS',0):.3f} "
            f"F2_9={f2_bo9:.4f} F2_12={f2_bo12:.4f} F3={f3:.4f}"
        )
        local = []
        for w in (0.006, 0.008, 0.010, 0.012, 0.015, 0.018):
            oof = {b: bases[b] + w * u for b in bases}
            per = {b: h2h(oof[b], bases[b], y, folds) for b in bases}
            cand = qlock(L(slot8) + w * ut, lv)
            rho = one_m_rho_e4(cand, slot8)
            delta = unitize(L(cand) - L(slot8))
            c6 = float(np.corrcoef(delta, bad6)[0, 1])
            c7 = float(np.corrcoef(delta, bad7)[0, 1])
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
                and 0.92 <= gates["raw_disp"] <= 1.08
                and gates.get("raw_KS", 0) < 0.025
                and f2_bo9 < 0.035
                and f2_bo12 < 0.038
                and f3 < 0.045
            )
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
                "w": w,
                "per": per,
                "rho": rho,
                "c6": c6,
                "c7": c7,
                "f2_bo9": f2_bo9,
                "f2_bo12": f2_bo12,
                "f3": f3,
                "gates": gates,
                "ok": ok,
                "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                "min_lift": float(np.min([per[b]["min"] for b in per])),
                "meta": meta or {},
            }
            local.append(row)
            if ok:
                winners.append({**row, "cand": cand, "u": u, "ut": ut})
        candidates.extend(local)

    # Mode1 additive
    for name, (u, ut) in blends_A.items():
        g = forensic(u, ut, Bt12)
        g9 = forensic(u, ut, Bt9)
        push_cand(f"A_{name}", u, ut, gates=g, g9=g9)

    for name, (u, ut, g, g9) in families_B.items():
        push_cand(f"B_{name}", u, ut, gates=g, g9=g9)

    for name, (u, ut, g, g9) in families_C.items():
        push_cand(f"C_{name}", u, ut, gates=g, g9=g9)

    # combos
    for a, b, wa, wb, na, nb in [
        (blends_A["U567_comp"], families_C["M_mod_blend"][:2], 0.55, 0.45, "U567", "mod"),
        (blends_A["U67"], families_B["F_xc_S8"][:2], 0.5, 0.5, "U67", "xc"),
        (families_B["F_xc_S8"][:2], families_C["M_mod_blend"][:2], 0.5, 0.5, "xc", "mod"),
        (blends_A["U567_comp"], families_B["F_xc_S8"][:2], 0.4, 0.6, "U567", "xc8"),
    ]:
        u = unitize(wa * a[0] + wb * b[0])
        ut = unitize(wa * a[1] + wb * b[1])
        g = forensic(u, ut, Bt12)
        g9 = forensic(u, ut, Bt9)
        push_cand(f"X_{na}_{nb}", u, ut, gates=g, g9=g9)

    # Mode2: upgrade Slot8 formula on Slot3 vs Slot8 H2H
    print("=== Mode2 upgraded Slot8 recipes ===", f"t={time.time()-t0:.1f}s")
    u_xgb, ut_xgb = families_B["F_xgb_S3"][0], families_B["F_xgb_S3"][1]
    u_cat45, ut_cat45 = families_B["F_cat_S3"][0], families_B["F_cat_S3"][1]
    u_mod, ut_mod = families_C["M_mod_blend"][0], families_C["M_mod_blend"][1]
    # R2/LGB blend
    u_r2b = unitize(0.7 * u_r2 + 0.3 * u_lgb)
    ut_r2b = unitize(0.7 * ut_r2 + 0.3 * ut_lgb)
    u_catb = unitize(0.6 * u_cat + 0.4 * u_cat45)
    ut_catb = unitize(0.6 * ut_cat + 0.4 * ut_cat45)

    mode2_grid = [
        # (w_lgb, w_cat, w_xgb, w_u, w_mod)
        (0.016, 0.012, 0.000, 0.012, 0.000),  # Slot8 baseline sanity
        (0.016, 0.012, 0.000, 0.000, 0.000),  # no U
        (0.014, 0.010, 0.008, 0.010, 0.008),
        (0.016, 0.010, 0.008, 0.010, 0.008),
        (0.014, 0.012, 0.006, 0.012, 0.008),
        (0.015, 0.011, 0.008, 0.010, 0.010),
        (0.016, 0.012, 0.000, 0.010, 0.008),  # U567 replace + mod
        (0.016, 0.012, 0.000, 0.012, 0.008),
        (0.014, 0.010, 0.000, 0.012, 0.010),
        (0.015, 0.010, 0.010, 0.008, 0.008),
    ]
    for wl, wc, wx, wu, wm in mode2_grid:
        # U slot: use U567_comp when wu>0 and not pure Slot8 sanity
        use_comp = not (wl == 0.016 and wc == 0.012 and wx == 0 and wu == 0.012 and wm == 0)
        uu, uut = (u_comp, ut_comp) if use_comp else (u_u5, ut_u5)
        oof = {
            "z_S8": z_s3
            + wl * u_r2b
            + wc * u_catb
            + wx * u_xgb
            + wu * uu
            + wm * u_mod,
            "twin": z_twin
            + wl * u_r2b
            + wc * u_catb
            + wx * u_xgb
            + wu * uu
            + wm * u_mod,
            "nnls": z_nnls
            + wl * u_r2b
            + wc * u_catb
            + wx * u_xgb
            + wu * uu
            + wm * u_mod,
        }
        per = {b: h2h(oof[b], bases[b], y, folds) for b in bases}
        cand = qlock(
            L(slot3)
            + wl * ut_r2b
            + wc * ut_catb
            + wx * ut_xgb
            + wu * uut
            + wm * ut_mod,
            lv,
        )
        rho = one_m_rho_e4(cand, slot8)
        delta = unitize(L(cand) - L(slot8))
        c6 = float(np.corrcoef(delta, bad6)[0, 1])
        c7 = float(np.corrcoef(delta, bad7)[0, 1])
        # forensic on the residual delta vs Slot8 recipe
        # approximate F2 via orth of new parts
        new_o = wx * u_xgb + (wu * uu if use_comp else 0) + wm * u_mod
        new_t = wx * ut_xgb + (wu * uut if use_comp else 0) + wm * ut_mod
        if np.std(new_o) < 1e-12:
            f2_bo9 = f2_bo12 = f3 = 0.0
            gates = {"raw_disp": 1.0, "raw_KS": 0.0, "F2": 0.0, "F1": True}
        else:
            oo, ot, _ = frozen_lofo_orth(new_o, new_t, Bo12, Bt12, folds, ridge_alpha=10.0)
            gates = forensic(oo, ot, Bt12)
            g9 = forensic(oo, ot, Bt9)
            f2_bo9, f2_bo12 = g9["F2"], gates["F2"]
            f3 = abs(float(np.corrcoef(unitize(ot), unitize(L(slot8) - L(slot5)))[0, 1]))
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
            and 0.92 <= gates["raw_disp"] <= 1.08
            and gates.get("raw_KS", 0) < 0.025
            and f2_bo9 < 0.035
            and f2_bo12 < 0.038
            and f3 < 0.045
        )
        jn = f"M2_wl{wl:.3f}_wc{wc:.3f}_wx{wx:.3f}_wu{wu:.3f}_wm{wm:.3f}"
        mark = "★" if ok else ("~" if per["z_S8"]["pos"] >= 4 else " ")
        print(
            f"{mark} {jn}: 1-ρ={rho:.3f} c7={c7:+.3f} F2={f2_bo12:.3f} ok={ok} | "
            + " | ".join(
                f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f} "
                f"t={per[b]['t']:.2f}"
                for b in bases
            )
        )
        row = {
            "name": jn,
            "family": "Mode2",
            "w": wu + wm + wx,
            "per": per,
            "rho": rho,
            "c6": c6,
            "c7": c7,
            "f2_bo9": f2_bo9,
            "f2_bo12": f2_bo12,
            "f3": f3,
            "gates": gates,
            "ok": ok,
            "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
            "min_lift": float(np.min([per[b]["min"] for b in per])),
            "meta": {"wl": wl, "wc": wc, "wx": wx, "wu": wu, "wm": wm},
        }
        candidates.append(row)
        if ok:
            winners.append({**row, "cand": cand})

    print(f"\nWinners: {len(winners)}")
    submit = False
    pub, ref, chosen = None, None, None
    if winners:
        winners.sort(
            key=lambda r: (r["min_lift"], r["mean_lift"], -abs(r["rho"] - 0.50)), reverse=True
        )
        chosen = winners[0]
        print(
            f"CHOSEN {chosen['name']} mean={chosen['mean_lift']:+.2f} "
            f"min={chosen['min_lift']:+.2f} 1-ρ={chosen['rho']:.3f} c7={chosen['c7']:+.3f}"
        )
        out_csv = SUB / f"sub_v19_12_LIVE_slot9_{chosen['name'][:60]}.csv"
        arr = chosen["cand"]
        assert len(arr) == n_te and np.isfinite(arr).all()
        pd.DataFrame({"id": ids, "is_fraud": arr}).to_csv(out_csv, index=False)
        msg = (
            f"v19.12 Slot9: {chosen['name']} tab45⊗U567 vs Slot8 ATH 0.95395 | "
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
                    if "v19_12" in row.get("fileName", ""):
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
            "u_audit": u_audit,
            "near_misses": near,
            "X_tab45_cols": int(X45_tr.shape[1]),
            "X_mod_cols": int(Xmod_o.shape[1]),
            "elapsed_s": time.time() - t0,
        }
    )
    OUT.joinpath("v19_12.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# GM V19.12 — Strict F2<0.035 tab45⊗U567 → "
        + ("**LIVE Slot9**" if submit else "**HOLD**"),
        "",
        f"**0 GPU pods · elapsed {payload['elapsed_s']:.0f}s**",
        "",
        f"| | |",
        f"|---|---|",
        f"| Slot8 ATH | 0.95395 |",
        f"| Slot8 rebuild 1-ρ | {rho_rebuild:.3e}e-4 |",
        f"| Submit | {submit} |",
        f"| Public LB | {pub} |",
        f"| Winners | {len(winners)} |",
        f"| X_tab45 | {X45_tr.shape[1]} cols |",
        f"| X_mod | {Xmod_o.shape[1]} cols |",
        "",
        "## U5..U10 audit (orth Bo12, w=0.010 vs Slot8)",
    ]
    for nm, a in u_audit.items():
        lines.append(
            f"- `{nm}`: F2_Bo9={a['F2_bo9']:.4f} F2_Bo12={a['F2_bo12']:.4f} "
            f"disp={a['disp_bo12']:.3f} c7={a['c7']:+.3f} folds={a['folds']} m={a['mean']:+.2f}"
        )
    lines += ["", "## Top near-misses"]
    for c in near[:15]:
        lines.append(
            f"- `{c['name']}`: 1-ρ={c['rho']:.3f} c7={c['c7']:+.3f} "
            f"F2={c['f2_bo12']:.3f} z_S8 {c['per']['z_S8']['pos']}/5 "
            f"m={c['per']['z_S8']['mean']:+.2f} min={c['per']['z_S8']['min']:+.2f} "
            f"twin {c['per']['twin']['pos']}/5 m={c['per']['twin']['mean']:+.2f} ok={c['ok']}"
        )
    if chosen:
        lines += [
            "",
            f"## Chosen `{chosen['name']}`",
            f"- mean={chosen['mean_lift']:+.2f} min={chosen['min_lift']:+.2f} "
            f"1-ρ={chosen['rho']:.3f} F2={chosen['f2_bo12']:.3f}",
        ]
    else:
        lines += ["", "## Decision", "HOLD ATH `0.95395`. Keep remaining daily slots."]
    report = LOG / "GM_V19_12_TAB45_U567_TENSOR_REPORT.md"
    report.write_text("\n".join(lines) + "\n")
    print("WROTE", report)
    print("WROTE", OUT / "v19_12.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
