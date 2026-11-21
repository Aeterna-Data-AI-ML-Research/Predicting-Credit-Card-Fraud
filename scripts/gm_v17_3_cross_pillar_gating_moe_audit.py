#!/usr/bin/env python3
"""GM V17.3 — Cross-pillar non-linear gating & MoE audit on z_S3.

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU only.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
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
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
NO_SUBMIT = True
RIDGE_ALPHA = 10.0
STAGE_MIN_SF = 18
STAGE_MIN_T = 2.20


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
    raw = np.asarray(raw, float)
    ref = np.asarray(ref, float)
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
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-12
    return (X - mu) / sd


def gini10(Z):
    K = Z.shape[-1]
    s = np.zeros(Z.shape[:-1], dtype=np.float64)
    for i, j in combinations(range(K), 2):
        s += np.abs(Z[..., i] - Z[..., j])
    return s / (K * (K - 1) / 2.0)


def frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=RIDGE_ALPHA):
    vo = unitize(v_oof)
    vt = unitize(v_test)
    orth_o = np.zeros(len(vo))
    betas = []
    for tr, te in folds:
        lr = Ridge(alpha=ridge_alpha, fit_intercept=True)
        lr.fit(Bo[tr], vo[tr])
        betas.append(lr.coef_.copy())
        orth_o[te] = vo[te] - Bo[te] @ lr.coef_ - lr.intercept_
    beta_bar = np.mean(np.stack(betas, 0), axis=0)
    orth_t = vt - Bt @ beta_bar
    orth_t = orth_t - orth_t.mean()
    orth_o = orth_o - orth_o.mean()
    return orth_o, orth_t, beta_bar


def forensic_gates(orth_o, orth_t, Bt, live_resid):
    disp = float((orth_t.std() + 1e-12) / (orth_o.std() + 1e-12))
    max_leak = float(np.max(np.abs([np.corrcoef(orth_t, Bt[:, j])[0, 1] for j in range(Bt.shape[1])])))
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


def screen(name, u, zbase, y, folds, te0):
    u = unitize(u)
    if roc_auc_score(y[te0], zbase[te0] + 0.01 * u[te0]) < roc_auc_score(
        y[te0], zbase[te0] - 0.01 * u[te0]
    ):
        u = -u
        flipped = True
    else:
        flipped = False
    base = [roc_auc_score(y[te], zbase[te]) for _, te in folds]
    best = None
    for w in (0.006, 0.009, 0.012, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04):
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
    tcrit = float(tdist.ppf(0.95, len(d25) - 1))
    ci90 = [mean - tcrit * se, mean + tcrit * se]
    out = {
        "name": name,
        "flipped_to_plus_AUC": flipped,
        "main": best,
        "sf25": int((d25 > 0).sum()),
        "t": float(t),
        "mean_sf": mean,
        "wilcoxon_p": wp,
        "ci90": [float(ci90[0]), float(ci90[1])],
    }
    print(
        f"  SCREEN {name}: {best['pos']}/5 @{best['w']} mean={best['mean']:.3f} "
        f"min={best['min']:.3f} sf={out['sf25']}/25 t={t:.2f} "
        f"W_p={wp:.3g} CI90=[{ci90[0]:.3f},{ci90[1]:.3f}] folds={best['folds']}"
    )
    return out, u


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
    if isinstance(x, np.ndarray):
        return scrub(x.tolist())
    return x


def pairwise_feats(cols):
    """15 feats: C(6,2)=15 products? User: {Pa·Pb, |Pa-Pb|} across 6 → 15+15=30.
    Spec says '15 pairwise products {Pa·Pb, |Pa-Pb|}' — interpret as 15 of each family
    truncated, or C(6,2)=15 pairs with ONE of {prod, absdiff}. Use both for all pairs (=30)
    then Ridge α=5000; also try products-only (=15) as primary G8.
    """
    names = list(cols.keys())
    feats = []
    labels = []
    for i, j in combinations(range(len(names)), 2):
        a, b = names[i], names[j]
        feats.append(cols[a] * cols[b])
        labels.append(f"{a}*{b}")
        feats.append(np.abs(cols[a] - cols[b]))
        labels.append(f"|{a}-{b}|")
    return np.column_stack(feats), labels


def oof_ridge_meta(Xo, Xt, y, folds, alpha=5000.0):
    """5-fold OOF Ridge on features; test = mean of fold models."""
    oof = np.zeros(len(y))
    preds_t = []
    for tr, te in folds:
        lr = Ridge(alpha=alpha, fit_intercept=True)
        lr.fit(Xo[tr], y[tr].astype(float))
        oof[te] = lr.predict(Xo[te])
        preds_t.append(lr.predict(Xt))
    return oof, np.mean(np.stack(preds_t, 0), 0)


def oof_logit_meta(Xo, Xt, y, folds, C=1.0 / 5000.0):
    oof = np.zeros(len(y))
    preds_t = []
    for tr, te in folds:
        clf = LogisticRegression(
            penalty="l2", C=C, solver="lbfgs", max_iter=400, fit_intercept=True
        )
        clf.fit(Xo[tr], y[tr])
        oof[te] = clf.decision_function(Xo[te])
        preds_t.append(clf.decision_function(Xt))
    return oof, np.mean(np.stack(preds_t, 0), 0)


def main():
    assert NO_SUBMIT
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
    dE_o = cache["dE"] if "dE" in cache.files else None
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    te0 = folds[0][1]

    Z = np.load(Q17, allow_pickle=True)
    names = [str(s) for s in Z["stream_names"]]
    XO = {n: Z[f"oof_{n}"].ravel() for n in names}
    XT = {n: Z[f"test_{n}"].ravel() for n in names}

    live = pd.read_csv(LIVE)
    ids = live["id"].to_numpy()
    lv = live["is_fraud"].to_numpy(float)
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    l7f_o = XO["lm7_5fold"]
    l7f_t = XT["lm7_5fold"]

    # Family_E_fixed (same as V16.8/9)
    z_s2026_o = L(R3)
    z_l7f_o = L(l7f_o)
    raw_E_o = (z_s2026_o - z_l7f_o) * (0.5 + 0.5 * U(z_ath))
    z_ath_t = L(lv)
    R3_t_Lfolds = np.stack([L(R3_te[f]) for f in range(5)], 0)
    z_s2026_t = L(R3_t_Lfolds.mean(0))
    z_l7f_t = L(l7f_t)
    raw_E_t = (z_s2026_t - z_l7f_t) * (0.5 + 0.5 * U(lv))
    Xo_e = np.column_stack([unitize(z_ath), unitize(d1)])
    Xt_e = np.column_stack([unitize(z_ath_t), unitize(d1t)])
    dE_o2, dE_t, _ = frozen_lofo_orth(raw_E_o, raw_E_t, Xo_e, Xt_e, folds, ridge_alpha=1.0)
    dE_o2, dE_t = unitize(dE_o2), unitize(dE_t)
    if roc_auc_score(y[te0], z_ath[te0] + 0.01 * dE_o2[te0]) < roc_auc_score(
        y[te0], z_ath[te0] - 0.01 * dE_o2[te0]
    ):
        dE_o2, dE_t = -dE_o2, -dE_t
    if dE_o is None:
        dE_o = dE_o2
    else:
        dE_o = unitize(dE_o)
        if np.corrcoef(dE_o, dE_o2)[0, 1] < 0:
            dE_o = -dE_o

    z_s1 = z_ath + 0.01045 * dE_o
    z_s3 = z_s1 + 0.0055 * unitize(d1)
    auc_s3 = float(roc_auc_score(y, z_s3))
    print(f"AUC z_S3={auc_s3:.7f}")

    # ----- Stream logits: L_fold OOF / L Test -----
    ms_raw_o = np.load(REPO / "preds/P_super_v4_oof.npy").ravel()
    ms_raw_t = np.load(REPO / "preds/P_super_v4_test.npy").ravel()
    g26_raw_o = np.load(G26_OOF).ravel()[: len(y)]
    g26_raw_t = np.load(G26_T0).ravel()[:299984]
    lm_raw_o = XO["megayak_5f_lm"]
    lm_raw_t = XT["megayak_5f_lm"]
    gpd_raw_o = XO["daoviet_gpdvae"]
    gpd_raw_t = XT["daoviet_gpdvae"]
    gb_raw_o = XO["gb_bag5"]
    gb_raw_t = XT["gb_bag5"]
    mlp_raw_o = XO["yekenot_realmlp"]
    mlp_raw_t = XT["yekenot_realmlp"]

    z_MS_o = L_fold(ms_raw_o, fold_id)
    z_MS_t = L(ms_raw_t)
    z_G26_o = L_fold(g26_raw_o, fold_id)
    z_G26_t = L(g26_raw_t)
    z_LM_o = L_fold(lm_raw_o, fold_id)
    z_LM_t = L(lm_raw_t)
    z_L7F_o = L_fold(l7f_o, fold_id)
    z_L7F_t = L(l7f_t)
    z_R3_o = L_fold(R3, fold_id)
    z_R3_t = L(R3_t_Lfolds.mean(0))
    z_GB_o = L_fold(gb_raw_o, fold_id)
    z_GB_t = L(gb_raw_t)
    z_MLP_o = L_fold(mlp_raw_o, fold_id)
    z_MLP_t = L(mlp_raw_t)
    z_GPD_o = L_fold(gpd_raw_o, fold_id)
    z_GPD_t = L(gpd_raw_t)
    z_d1_o = L_fold(d1, fold_id)
    z_d1_t = L(d1t)

    print("=== STEP 1: pillars ===")
    P_BPE_o = unitize(0.55 * z_MS_o + 0.45 * z_G26_o)
    P_BPE_t = unitize(0.55 * z_MS_t + 0.45 * z_G26_t)
    P_QLM_o = unitize(0.45 * z_LM_o + 0.30 * z_L7F_o + 0.25 * z_R3_o)
    P_QLM_t = unitize(0.45 * z_LM_t + 0.30 * z_L7F_t + 0.25 * z_R3_t)
    P_TAB_o = unitize(0.50 * z_GB_o + 0.35 * z_MLP_o + 0.15 * z_GPD_o)
    P_TAB_t = unitize(0.50 * z_GB_t + 0.35 * z_MLP_t + 0.15 * z_GPD_t)

    raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
    ll_neg_o = raw_o[:, :, 0]
    ll_neg_t = raw_t[:, :, :, 0]  # (5,N,10)
    mean_k_neg_o = ll_neg_o.mean(1)
    mean_k_neg_t = ll_neg_t.mean(2)  # (5,N)
    LL_o = unitize(L_fold(mean_k_neg_o, fold_id))
    LL_t_folds = np.stack([L(mean_k_neg_t[f]) for f in range(5)], 0)
    LL_t = unitize(L(LL_t_folds.mean(0)))

    llr_o = raw_o[:, :, 1] - raw_o[:, :, 0]
    llr_t = raw_t[:, :, :, 1] - raw_t[:, :, :, 0]
    U5_o_raw = gini10(col_z(llr_o))
    U5_t_raw = np.stack([gini10(col_z(llr_t[f])) for f in range(5)], 0)
    U5_o = unitize(L_fold(U5_o_raw, fold_id))
    U5_t = unitize(L(np.stack([L(U5_t_raw[f]) for f in range(5)], 0).mean(0)))

    d1u_o = unitize(z_d1_o)
    d1u_t = unitize(z_d1_t)

    pillar_corr = {
        "BPE_QLM": float(np.corrcoef(P_BPE_o, P_QLM_o)[0, 1]),
        "BPE_TAB": float(np.corrcoef(P_BPE_o, P_TAB_o)[0, 1]),
        "QLM_TAB": float(np.corrcoef(P_QLM_o, P_TAB_o)[0, 1]),
        "BPE_LL": float(np.corrcoef(P_BPE_o, LL_o)[0, 1]),
        "QLM_U5": float(np.corrcoef(P_QLM_o, U5_o)[0, 1]),
    }
    print("  pillar OOF corr:", pillar_corr)

    # Bo9 / Bt9 (same recipe as V16.9)
    Bo9 = np.column_stack(
        [
            unitize(L(ms_raw_o)),
            unitize(L(g26_raw_o)),
            unitize(L(lm_raw_o)),
            unitize(L(l7f_o)),
            unitize(gpd_raw_o),
            unitize(gb_raw_o),
            unitize(mlp_raw_o),
            unitize(d1),
            unitize(dE_o),
        ]
    )
    Bt9 = np.column_stack(
        [
            unitize(L(ms_raw_t)),
            unitize(L(g26_raw_t)),
            unitize(L(lm_raw_t)),
            unitize(L(l7f_t)),
            unitize(gpd_raw_t),
            unitize(gb_raw_t),
            unitize(mlp_raw_t),
            unitize(d1t),
            unitize(dE_t),
        ]
    )
    live_L = L(lv)
    live_hat = Ridge(alpha=RIDGE_ALPHA).fit(Bt9, live_L).predict(Bt9)
    live_resid = live_L - live_hat

    print("=== STEP 2: gating features ===")
    # G1
    G1_o = np.maximum(0, P_BPE_o - P_QLM_o) ** 2 - np.maximum(0, P_QLM_o - P_BPE_o) ** 2
    G1_t = np.maximum(0, P_BPE_t - P_QLM_t) ** 2 - np.maximum(0, P_QLM_t - P_BPE_t) ** 2
    # G2
    G2_o = (0.5 * P_BPE_o + 0.5 * P_QLM_o - P_TAB_o) ** 2
    G2_t = (0.5 * P_BPE_t + 0.5 * P_QLM_t - P_TAB_t) ** 2
    # G3 / G3b
    G3_o = P_BPE_o * P_TAB_o
    G3_t = P_BPE_t * P_TAB_t
    stack_o = np.stack([P_BPE_o, P_QLM_o, P_TAB_o], 0)
    stack_t = np.stack([P_BPE_t, P_QLM_t, P_TAB_t], 0)
    G3b_o = stack_o.min(0) - stack_o.max(0)
    G3b_t = stack_t.min(0) - stack_t.max(0)
    # G4
    G4_o = LL_o * (P_BPE_o - P_TAB_o)
    G4_t = LL_t * (P_BPE_t - P_TAB_t)
    # G5
    G5_o = U5_o * (P_QLM_o - P_BPE_o)
    G5_t = U5_t * (P_QLM_t - P_BPE_t)
    # G6 / G6b
    G6_o = z_s3 * np.abs(z_s3)
    # test: use L(LIVE)~z_ath_t proxy for z_S3 scale on test (Slot3 closer to S3)
    z_s3_t_proxy = L(slot3)
    G6_t = z_s3_t_proxy * np.abs(z_s3_t_proxy)
    G6b_o = np.sign(P_BPE_o - P_TAB_o) * (P_BPE_o - P_TAB_o) ** 2 * U(z_s3)
    G6b_t = np.sign(P_BPE_t - P_TAB_t) * (P_BPE_t - P_TAB_t) ** 2 * U(z_s3_t_proxy)
    # G7
    G7_o = d1u_o * P_BPE_o
    G7_t = d1u_t * P_BPE_t

    # G8: pairwise products + abs diffs across 6 channels, Ridge α=5000 OOF
    cols_o = {
        "P_BPE": P_BPE_o,
        "P_QLM": P_QLM_o,
        "P_TAB": P_TAB_o,
        "d1": d1u_o,
        "LL": LL_o,
        "U5": U5_o,
    }
    cols_t = {
        "P_BPE": P_BPE_t,
        "P_QLM": P_QLM_t,
        "P_TAB": P_TAB_t,
        "d1": d1u_t,
        "LL": LL_t,
        "U5": U5_t,
    }
    Xo_raw, feat_labels = pairwise_feats(cols_o)
    Xt_raw, _ = pairwise_feats(cols_t)
    # project each raw feature via frozen_lofo_orth first, then meta-learn
    Xo_orth = np.zeros_like(Xo_raw)
    Xt_orth = np.zeros_like(Xt_raw)
    for j in range(Xo_raw.shape[1]):
        oo, ot, _ = frozen_lofo_orth(Xo_raw[:, j], Xt_raw[:, j], Bo9, Bt9, folds)
        Xo_orth[:, j] = unitize(oo)
        Xt_orth[:, j] = unitize(ot)

    # Primary G8: first 15 = products only (C(6,2)=15)
    prod_idx = [i for i, lab in enumerate(feat_labels) if "*" in lab and not lab.startswith("|")]
    # labels are "a*b" and "|a-b|" — products have '*' and don't start with |
    prod_idx = [i for i, lab in enumerate(feat_labels) if lab.count("*") == 1 and not lab.startswith("|")]
    assert len(prod_idx) == 15, len(prod_idx)
    G8_o, G8_t = oof_ridge_meta(Xo_orth[:, prod_idx], Xt_orth[:, prod_idx], y, folds, alpha=5000.0)
    G8log_o, G8log_t = oof_logit_meta(
        Xo_orth[:, prod_idx], Xt_orth[:, prod_idx], y, folds, C=1.0 / 5000.0
    )
    # Full 30-feat Ridge for secondary
    G8full_o, G8full_t = oof_ridge_meta(Xo_orth, Xt_orth, y, folds, alpha=5000.0)

    # Also U5 and q80 alone as reference controls (already orth'd in V16.9; re-eval here)
    q80_o_raw = np.quantile(np.stack([L(llr_o[:, k]) for k in range(10)], 1), 0.8, axis=1)
    q20_o_raw = np.quantile(np.stack([L(llr_o[:, k]) for k in range(10)], 1), 0.2, axis=1)
    U1_o = L_fold(q80_o_raw - q20_o_raw, fold_id)
    z_llr_t = np.stack(
        [np.stack([L(llr_t[f, :, k]) for k in range(10)], 1) for f in range(5)], 0
    )
    U1_t_folds = np.quantile(z_llr_t, 0.8, axis=2) - np.quantile(z_llr_t, 0.2, axis=2)
    U1_t = L(np.stack([L(U1_t_folds[f]) for f in range(5)], 0).mean(0))

    candidates = {
        "G1_BPE_vs_QLM_asym": (G1_o, G1_t),
        "G2_Gen_vs_Tab_disagreement": (G2_o, G2_t),
        "G3_Cross_Family_Synergy": (G3_o, G3_t),
        "G3b_min_minus_max": (G3b_o, G3b_t),
        "G4_Rarity_Gated_LM_vs_Tab": (G4_o, G4_t),
        "G5_Uncertainty_Gated_QLM": (G5_o, G5_t),
        "G6_Tail_Quadratic_Curvature": (G6_o, G6_t),
        "G6b_signed_diff_sq_U": (G6b_o, G6b_t),
        "G7_d1_x_LM": (G7_o, G7_t),
        "G8_PolyRidge_15prod": (G8_o, G8_t),
        "G8b_LogitL2_15prod": (G8log_o, G8log_t),
        "G8c_PolyRidge_30": (G8full_o, G8full_t),
        "U5_gini_ctrl": (U5_o, U5_t),
        "U1_q80_q20_ctrl": (U1_o, U1_t),
    }

    print("=== STEP 3: scorecard ===")
    results = {}
    oriented = {}
    for name, (vo, vt) in candidates.items():
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds)
        gates = forensic_gates(oo, ot, Bt9, live_resid)
        scr, u = screen(name, oo, z_s3, y, folds, te0)
        ut = unitize(ot if np.corrcoef(u, oo)[0, 1] > 0 else -ot)
        oriented[name] = (u, ut)
        results[name] = {**gates, **{k: v for k, v in scr.items() if k != "name"}}
        print(
            f"  META {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:.4f}) ver={gates['verified']}"
        )

    # Light combo: best single gate × U5 if both exist
    winners = []
    for n, r in results.items():
        if not r["verified"]:
            continue
        if r["main"]["pos"] < 5:
            continue
        if r["sf25"] < STAGE_MIN_SF or r["t"] < STAGE_MIN_T:
            continue
        winners.append(n)

    # Try top-2 verified orth signals mean if individually weak
    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1]["main"]["pos"], kv[1]["sf25"], kv[1]["main"]["mean"]),
        reverse=True,
    )
    combo_tried = []
    if len(ranked) >= 2:
        for a, b in combinations([n for n, _ in ranked[:6]], 2):
            uo = unitize(oriented[a][0] + oriented[b][0])
            ut = unitize(oriented[a][1] + oriented[b][1])
            # already orth'd components — re-orth the sum for fairness
            oo, ot, _ = frozen_lofo_orth(uo, ut, Bo9, Bt9, folds)
            gates = forensic_gates(oo, ot, Bt9, live_resid)
            cname = f"combo_{a}+{b}"
            scr, u = screen(cname, oo, z_s3, y, folds, te0)
            ut2 = unitize(ot if np.corrcoef(u, oo)[0, 1] > 0 else -ot)
            oriented[cname] = (u, ut2)
            results[cname] = {**gates, **{k: v for k, v in scr.items() if k != "name"}}
            combo_tried.append(cname)
            if (
                gates["verified"]
                and scr["main"]["pos"] >= 5
                and scr["sf25"] >= STAGE_MIN_SF
                and scr["t"] >= STAGE_MIN_T
            ):
                winners.append(cname)

    print(f"STAGE winners={winners}")
    staged = []
    for n in winners:
        u_o, u_t = oriented[n]
        wstar = results[n]["main"]["w"]
        cand = qlock(L(slot3) + wstar * u_t, slot3)
        path = SUB / f"sub_v17_3_{n}_NOSUBMIT.csv"
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
        staged.append(
            {
                "file": path.name,
                "signal": n,
                "w": wstar,
                "pos": results[n]["main"]["pos"],
                "sf25": results[n]["sf25"],
                "t": results[n]["t"],
            }
        )
        print("STAGED", path.name)

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "auc_z_s3": auc_s3,
            "pillar_corr_oof": pillar_corr,
            "feat_labels_G8": feat_labels,
            "results": results,
            "winners": winners,
            "combo_tried": combo_tried,
            "staged": staged,
            "stage_rule": "F1&F2&F3 AND 5/5 AND sf>=18 AND t>=2.20",
        }
    )
    OUT.joinpath("v17_3.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v17_3.json")
    print("DONE")
    return payload


if __name__ == "__main__":
    main()
