#!/usr/bin/env python3
"""GM V18.7 — Strict zero-echo peer v2 + pool-size matching + ortho combos.

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU <75s target.
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
ORIG = REPO / "original/credit_card_fraud_2026.csv"
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
N_VA = 139992  # expected
SCALE = 299984 / 139992  # ~2.1429


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


def peer_stats_pool(z, gids, m0_mean, m0_pos=10.0, n_ref=100000.0):
    """Leave-self-out peer mean (two m0), pos-rate, logcnt rate-normalized."""
    z = np.asarray(z, float)
    g = np.asarray(gids, np.int64)
    _, g2 = np.unique(g, return_inverse=True)
    nbin = int(g2.max()) + 1
    cnt = np.bincount(g2, minlength=nbin).astype(np.float64)
    sz = np.bincount(g2, weights=z, minlength=nbin)
    pos = (z > 1.0).astype(np.float64)
    spos = np.bincount(g2, weights=pos, minlength=nbin)
    c = cnt[g2]
    n = len(z)

    def mean_m(m0):
        peer = (sz[g2] - z) / (c - 1.0 + m0)
        return np.where(c <= 1, 0.0, peer)

    peer_m = mean_m(m0_mean)
    # pos rate leave-self-out
    peer_pos = (spos[g2] - pos) / (c - 1.0 + m0_pos)
    peer_pos = np.where(c <= 1, 0.0, peer_pos)
    logcnt = np.log1p(c / (n / n_ref))
    return peer_m, peer_pos, logcnt, c


def fold_standardize_oof_test(oof, test, folds):
    """Per-fold unitize on va; unitize test independently (equal length qlock N/A)."""
    out_o = np.zeros_like(oof)
    for _, va in folds:
        out_o[va] = unitize(oof[va])
    out_t = unitize(test)
    return out_o, out_t


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

    # U5
    raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
    llr_o = raw_o[:, :, 1] - raw_o[:, :, 0]
    llr_t = raw_t[:, :, :, 1] - raw_t[:, :, :, 0]
    U5_o = L_fold(gini10(col_z(llr_o)), fold_id)
    U5_t = L(np.stack([L(gini10(col_z(llr_t[f]))) for f in range(5)], 0).mean(0))

    # R1_pure orth Bo10
    v180 = np.load(OUT / "v18_0_residuals.npz")
    R1p_o, R1p_t = v180["R1p_oof"], v180["R1p_test"]
    oo_r1, ot_r1, _ = frozen_lofo_orth(R1p_o, R1p_t, Bo10, Bt10, folds, ridge_alpha=10.0)
    u_r1 = unitize(oo_r1)
    if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u_r1[te0]) < roc_auc_score(
        y[te0], z_s5[te0] - 0.01 * u_r1[te0]
    ):
        u_r1, ot_r1 = -u_r1, -ot_r1
    ut_r1 = unitize(ot_r1)

    print("=== STEP 1: strict peer bank v2 ===", f"t={time.time()-t0:.1f}s")
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

    # k14: amt_q500 × bal_q500
    aq = np.clip((rankdata(amt) / (len(amt) + 1e-12) * 500).astype(np.int64), 0, 499)
    bq = np.clip((rankdata(bal) / (len(bal) + 1e-12) * 500).astype(np.int64), 0, 499)

    key_specs = {
        "k1_amt2": codes_from_str(np.round(amt, 2).astype(str)),
        "k2_bal2": codes_from_str(np.round(bal, 2).astype(str)),
        "k3_vel2": codes_from_str(np.round(vel, 2).astype(str)),
        "k4_mr4": codes_from_str(np.round(mr, 4).astype(str)),
        "k5_dist2": codes_from_str(np.round(dist, 2).astype(str)),
        "k6_cat_amt0": codes_from_str(
            (pd.Series(mer) + "|" + pd.Series(np.round(amt, 0).astype(np.int64).astype(str))).to_numpy()
        ),
        "k7_cat_bal10": codes_from_str(
            (
                pd.Series(mer)
                + "|"
                + pd.Series((np.round(bal / 10.0) * 10.0).astype(np.int64).astype(str))
            ).to_numpy()
        ),
        "k8_amtcents_cat": codes_from_str(
            (
                pd.Series(((np.round(np.abs(amt) * 100).astype(np.int64) % 100) % 100).astype(str))
                + "|"
                + pd.Series(mer)
            ).to_numpy()
        ),
        "k9_balcents_cat": codes_from_str(
            (
                pd.Series(((np.round(np.abs(bal) * 100).astype(np.int64) % 100) % 100).astype(str))
                + "|"
                + pd.Series(mer)
            ).to_numpy()
        ),
        "k10_amt1_cat": codes_from_str(
            (pd.Series(np.round(amt, 1).astype(str)) + "|" + pd.Series(mer)).to_numpy()
        ),
        "k11_mr2_cat": codes_from_str(
            (pd.Series(np.round(mr, 2).astype(str)) + "|" + pd.Series(mer)).to_numpy()
        ),
        "k12_hrs1_cat": codes_from_str(
            (pd.Series(np.round(hrs, 1).astype(str)) + "|" + pd.Series(mer)).to_numpy()
        ),
        "k13_vel1_cat": codes_from_str(
            (pd.Series(np.round(vel, 1).astype(str)) + "|" + pd.Series(mer)).to_numpy()
        ),
        "k14_aq500_bq500": aq * 500 + bq,
    }
    key_order = list(key_specs.keys())

    # Build OOF features fold-isolated; test with scaled m0
    # Feature layout per key: m3, m15, pos_m10, logcnt  → 14*4 = 56
    # + pairwise diffs among k1..k6 at m15 → C(6,2)=15
    n_feat_peer = 14 * 4 + 15
    Xp_oof = np.zeros((n_tr, n_feat_peer), dtype=np.float32)
    Xp_te = np.zeros((n_te, n_feat_peer), dtype=np.float32)
    feat_names = []
    for kn in key_order:
        feat_names += [f"{kn}_m3", f"{kn}_m15", f"{kn}_pos10", f"{kn}_logcnt"]
    top6 = key_order[:6]
    for a, b in combinations(range(6), 2):
        feat_names.append(f"diff_m15_{top6[a]}_minus_{top6[b]}")

    mean_cnt_va = {}
    for kn, gid_all in key_specs.items():
        cnts = []
        for f, (tr, va) in enumerate(folds):
            z_va = unitize(z_s5[va])
            g_va = gid_all[va]
            m3, pos, logc, c = peer_stats_pool(z_va, g_va, m0_mean=3.0, m0_pos=10.0)
            m15, _, _, _ = peer_stats_pool(z_va, g_va, m0_mean=15.0, m0_pos=10.0)
            # place into columns — fill after loop structure differently
            cnts.append(c.mean())
        mean_cnt_va[kn] = float(np.mean(cnts))
        print(f"  {kn}: mean_cnt_va≈{mean_cnt_va[kn]:.1f}")

    # Actually fill matrices
    col = 0
    m15_by_key_oof = {}
    m15_by_key_te = {}
    for kn in key_order:
        gid_all = key_specs[kn]
        o_m3 = np.zeros(n_tr)
        o_m15 = np.zeros(n_tr)
        o_pos = np.zeros(n_tr)
        o_log = np.zeros(n_tr)
        for f, (tr, va) in enumerate(folds):
            z_va = unitize(z_s5[va])
            g_va = gid_all[va]
            m3, pos, logc, _ = peer_stats_pool(z_va, g_va, m0_mean=3.0, m0_pos=10.0)
            m15, _, _, _ = peer_stats_pool(z_va, g_va, m0_mean=15.0, m0_pos=10.0)
            o_m3[va], o_m15[va], o_pos[va], o_log[va] = m3, m15, pos, logc
        # test with scaled m0
        z_te = unitize(z_s5_t_sym)
        g_te = gid_all[n_tr:]
        t_m3, t_pos, t_log, _ = peer_stats_pool(
            z_te, g_te, m0_mean=3.0 * SCALE, m0_pos=10.0 * SCALE
        )
        t_m15, _, _, _ = peer_stats_pool(z_te, g_te, m0_mean=15.0 * SCALE, m0_pos=10.0 * SCALE)
        # fold-standardize / qlock
        o_m3, t_m3 = fold_standardize_oof_test(o_m3, t_m3, folds)
        o_m15, t_m15 = fold_standardize_oof_test(o_m15, t_m15, folds)
        o_pos, t_pos = fold_standardize_oof_test(o_pos, t_pos, folds)
        o_log, t_log = fold_standardize_oof_test(o_log, t_log, folds)
        Xp_oof[:, col] = o_m3
        Xp_te[:, col] = t_m3
        col += 1
        Xp_oof[:, col] = o_m15
        Xp_te[:, col] = t_m15
        m15_by_key_oof[kn] = o_m15
        m15_by_key_te[kn] = t_m15
        col += 1
        Xp_oof[:, col] = o_pos
        Xp_te[:, col] = t_pos
        col += 1
        Xp_oof[:, col] = o_log
        Xp_te[:, col] = t_log
        col += 1

    for a, b in combinations(range(6), 2):
        ka, kb = top6[a], top6[b]
        d_o = m15_by_key_oof[ka] - m15_by_key_oof[kb]
        d_t = m15_by_key_te[ka] - m15_by_key_te[kb]
        d_o, d_t = fold_standardize_oof_test(d_o, d_t, folds)
        Xp_oof[:, col] = d_o
        Xp_te[:, col] = d_t
        col += 1
    assert col == n_feat_peer

    # half-subsample disp check on k2_m15
    rng = np.random.default_rng(42)
    half_preds = []
    g_te = key_specs["k2_bal2"][n_tr:]
    z_te = unitize(z_s5_t_sym)
    for b in range(5):
        idx = rng.choice(n_te, size=N_VA, replace=False)
        m15, _, _, _ = peer_stats_pool(
            z_te[idx], g_te[idx], m0_mean=15.0, m0_pos=10.0
        )
        half_preds.append(m15.std())
    raw_disp = float(np.std(half_preds) / (np.mean(half_preds) + 1e-12))
    print(
        f"  peer feats={n_feat_peer} half-subsample std CV(k2_m15)={raw_disp:.4f} "
        f"t={time.time()-t0:.1f}s"
    )

    # Cohort ranks (zero z)
    g1 = codes_from_str(mer)
    g2 = codes_from_str(
        (
            pd.Series(mer)
            + "|"
            + pd.Series(
                np.concatenate(
                    [train["channel"].astype(str), test["channel"].astype(str)]
                )
            )
            + "|"
            + pd.Series(
                np.concatenate(
                    [train["device_type"].astype(str), test["device_type"].astype(str)]
                )
            )
        ).to_numpy()
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
    Xc_oof = np.column_stack(cohort_tr).astype(np.float32)
    Xc_te = np.column_stack(cohort_te).astype(np.float32)

    # Ridge columns: m3, m15, pos for each key = 14*3 = 42 (skip logcnt for ridge)
    ridge_idx = []
    for i in range(14):
        ridge_idx += [i * 4, i * 4 + 1, i * 4 + 2]  # m3, m15, pos
    Xr_oof = Xp_oof[:, ridge_idx]
    Xr_te = Xp_te[:, ridge_idx]

    print("=== STEP 2: fit S1..S4 ===", f"t={time.time()-t0:.1f}s")
    # S1 ridge
    S1 = np.zeros(n_tr)
    S1_te = []
    for f, (tr, va) in enumerate(folds):
        sw = np.sqrt(h5[f][tr])
        lr = Ridge(alpha=300.0, fit_intercept=True)
        lr.fit(Xr_oof[tr] * sw[:, None], g5[f][tr] * sw)
        S1[va] = lr.predict(Xr_oof[va])
        S1_te.append(lr.predict(Xr_te))
    S1_t = np.mean(np.stack(S1_te, 0), 0)

    def fit_lgb(Xtr, Xte, params, seeds):
        oof = np.zeros(n_tr)
        te_acc = []
        for seed in seeds:
            oo = np.zeros(n_tr)
            tef = []
            for f, (tr, va) in enumerate(folds):
                model = lgb.LGBMRegressor(**params, random_state=seed)
                model.fit(Xtr[tr], g5[f][tr], sample_weight=h5[f][tr])
                oo[va] = model.predict(Xtr[va])
                tef.append(model.predict(Xte))
            oof += oo
            te_acc.append(np.mean(np.stack(tef, 0), 0))
        return oof / len(seeds), np.mean(np.stack(te_acc, 0), 0)

    seeds5 = [42, 2026, 7, 100, 999]
    params_reg = dict(
        objective="regression",
        num_leaves=4,
        min_child_samples=1200,
        n_estimators=35,
        learning_rate=0.04,
        colsample_bytree=0.65,
        subsample=0.8,
        reg_lambda=50.0,
        verbosity=-1,
        n_jobs=-1,
    )
    params_med = dict(
        objective="regression",
        num_leaves=6,
        min_child_samples=800,
        n_estimators=40,
        learning_rate=0.04,
        colsample_bytree=0.65,
        subsample=0.8,
        reg_lambda=30.0,
        verbosity=-1,
        n_jobs=-1,
    )
    S2, S2_t = fit_lgb(Xp_oof, Xp_te, params_reg, seeds5)
    Xmed_o = np.column_stack([Xp_oof, Xc_oof]).astype(np.float32)
    Xmed_t = np.column_stack([Xp_te, Xc_te]).astype(np.float32)
    S3, S3_t = fit_lgb(Xmed_o, Xmed_t, params_med, seeds5)
    S4 = unitize(0.25 * unitize(S1) + 0.40 * unitize(S2) + 0.35 * unitize(S3))
    S4_t = unitize(0.25 * unitize(S1_t) + 0.40 * unitize(S2_t) + 0.35 * unitize(S3_t))
    print(f"  fits done t={time.time()-t0:.1f}s")

    # orth U5 vs Bo10 for combo
    oo_u5, ot_u5, _ = frozen_lofo_orth(U5_o, U5_t, Bo10, Bt10, folds, ridge_alpha=1.0)
    u_u5 = unitize(oo_u5)
    if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u_u5[te0]) < roc_auc_score(
        y[te0], z_s5[te0] - 0.01 * u_u5[te0]
    ):
        u_u5, ot_u5 = -u_u5, -ot_u5
    ut_u5 = unitize(ot_u5)

    print("=== STEP 3: scorecard ===")
    signals_raw = {
        "S1_peer_ridge_strict": (S1, S1_t),
        "S2_peer_lgb_reg_strict": (S2, S2_t),
        "S3_peer_lgb_med_strict": (S3, S3_t),
        "S4_peer_ens_strict": (S4, S4_t),
    }
    results = {}
    oriented = {}

    def eval_raw(name, vo, vt):
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
            and all(per[b]["mean"] > 0 for b in bases_s5)
        )
        results[name] = {
            "gates": gates,
            "per_base": per,
            "pre_corr_zS5": float(np.corrcoef(vo, z_s5)[0, 1]),
            "corr_R2": float(np.corrcoef(vo, R2_oof)[0, 1]),
            "stage_ok": ok,
        }
        print(
            f"  {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{gates['leak_col']}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) stage_ok={ok}"
        )
        for b, row in per.items():
            print(
                f"    {b}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
                f"min={row['min']:+.3f} folds={row['folds']} sf={row['sf25']} t={row['t']:.2f}"
            )
        return u, ut

    for name, (vo, vt) in signals_raw.items():
        eval_raw(name, vo, vt)

    # Combos on already-oriented units
    u_s4, ut_s4 = oriented["S4_peer_ens_strict"]
    combos = {
        "S4_plus_U5": (unitize(u_s4 + u_u5), unitize(ut_s4 + ut_u5)),
        "S4_plus_0p30_R1": (unitize(u_s4 + 0.30 * u_r1), unitize(ut_s4 + 0.30 * ut_r1)),
        "S4_plus_U5_plus_0p30_R1": (
            unitize(u_s4 + u_u5 + 0.30 * u_r1),
            unitize(ut_s4 + ut_u5 + 0.30 * ut_r1),
        ),
    }
    for name, (u, ut) in combos.items():
        # already orth components — score u directly but still pass through orth for F2
        oo, ot, _ = frozen_lofo_orth(u, ut, Bo10, Bt10, folds, ridge_alpha=1.0)
        gates = forensic_gates(oo, ot, Bt10, live_resid, bn=bn10)
        uu = unitize(oo)
        if roc_auc_score(y[te0], z_s5[te0] + 0.01 * uu[te0]) < roc_auc_score(
            y[te0], z_s5[te0] - 0.01 * uu[te0]
        ):
            uu, ot = -uu, -ot
        utt = unitize(ot)
        oriented[name] = (uu, utt)
        per = {b: score_fixed(uu, zb, y, folds) for b, zb in bases_s5.items()}
        ok = (
            gates["verified"]
            and all(per[b]["pos"] >= 5 for b in bases_s5)
            and all(per[b]["sf25"] >= 19 for b in bases_s5)
            and all(per[b]["t"] >= 3.2 for b in bases_s5)
        )
        results[name] = {
            "gates": gates,
            "per_base": per,
            "pre_corr_zS5": float(np.corrcoef(uu, z_s5)[0, 1]),
            "corr_R2": float(np.corrcoef(uu, R2_oof)[0, 1]),
            "stage_ok": ok,
        }
        print(
            f"  {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{gates['leak_col']}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) stage_ok={ok}"
        )
        for b, row in per.items():
            print(
                f"    {b}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
                f"min={row['min']:+.3f} folds={row['folds']} sf={row['sf25']} t={row['t']:.2f}"
            )

    staged = []
    for name, r in results.items():
        if r["stage_ok"]:
            u, ut = oriented[name]
            w = r["per_base"]["z_Slot5_twin"]["w"]
            path = SUB / f"sub_v18_7_{name}_w{str(w).replace('.','p')}_NOSUBMIT.csv"
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
            "n_peer_feats": n_feat_peer,
            "mean_cnt_va": mean_cnt_va,
            "half_subsample_cv": raw_disp,
            "m0_scale_test": SCALE,
            "results": results,
            "staged": staged,
            "elapsed_s": time.time() - t0,
            "stage_rule": "F1∧F2∧F3 ∧ 5/5 ∧ sf>=19 ∧ t>=3.2 on all 3 Slot5 baselines",
        }
    )
    OUT.joinpath("v18_7.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUT / "v18_7_strict_peer.npz",
        S1_oof=S1,
        S1_test=S1_t,
        S2_oof=S2,
        S2_test=S2_t,
        S3_oof=S3,
        S3_test=S3_t,
        S4_oof=S4,
        S4_test=S4_t,
        feat_names=np.array(feat_names, dtype=object),
    )
    print("WROTE", OUT / "v18_7.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
