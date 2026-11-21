#!/usr/bin/env python3
"""GM V16.9 — Slot1 sign audit + GPU order-dispersion estimators + z_S3 scorecard.

STRICT NO_SUBMIT=True. 0 GPU pods.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import Ridge
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
SLOT1 = SUB / "sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
NO_SUBMIT = True
RIDGE_ALPHA = 10.0


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


def col_z(X):
    """Column-standardize over rows. X:(N,K) -> same shape."""
    if X.ndim != 2:
        raise ValueError(X.shape)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-12
    return (X - mu) / sd


def gini10(Z):
    """Gini mean difference across last axis K=10. Z:(...,10)"""
    # (1/45) sum_{i<j} |zi-zj|
    K = Z.shape[-1]
    s = np.zeros(Z.shape[:-1], dtype=np.float64)
    for i, j in combinations(range(K), 2):
        s += np.abs(Z[..., i] - Z[..., j])
    return s / (K * (K - 1) / 2.0)


def mad10(Z):
    med = np.median(Z, axis=-1, keepdims=True)
    return np.mean(np.abs(Z - med), axis=-1)


def std10(Z):
    return np.std(Z, axis=-1)


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
    # Wilcoxon signed-rank vs 0
    try:
        wstat, wp = wilcoxon(d25, alternative="greater")
        wp = float(wp)
    except Exception:
        wp = float("nan")
    # 90% CI for mean (t approx)
    from scipy.stats import t as tdist

    tcrit = float(tdist.ppf(0.95, len(d25) - 1))
    ci90 = [mean - tcrit * se, mean + tcrit * se]
    out = {
        "name": name,
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


def L_fold(raw, fold_id):
    out = np.zeros_like(raw, dtype=np.float64)
    for f in range(5):
        m = fold_id == f
        out[m] = L(raw[m])
    return out


def cross_fold_spearman(test_folds):
    """test_folds: (5, N) -> mean pairwise spearman."""
    rs = []
    for f, g in combinations(range(5), 2):
        rs.append(float(spearmanr(test_folds[f], test_folds[g]).statistic))
    return float(np.mean(rs)), rs


def build_estimators_oof(raw_oof):
    """raw_oof (N,10,2) -> dict name -> (N,) raw scores (pre L)."""
    llr = raw_oof[:, :, 1] - raw_oof[:, :, 0]  # (N,10)
    ll_neg = raw_oof[:, :, 0]
    ll_pos = raw_oof[:, :, 1]
    llr_z = col_z(llr)
    z = np.stack([L(llr[:, k]) for k in range(10)], axis=1)
    ll_pos_z = col_z(ll_pos)
    ll_neg_z = col_z(ll_neg)

    q80 = np.quantile(z, 0.8, axis=1)
    q20 = np.quantile(z, 0.2, axis=1)
    P = 0.5 * (llr_z[:, :5] + llr_z[:, 5:])
    gap = np.mean(np.abs(llr_z[:, :5] - llr_z[:, 5:]), axis=1)

    return {
        "U1_q80_q20_z": q80 - q20,
        "U2_gini10_z": gini10(z),
        "U3_mad10_z": mad10(z),
        "U4_std10_z": std10(z),
        "U5_gini10_llr": gini10(llr_z),
        "U6_pair_std5_llr": std10(P),  # std over 5 pairs; reuse std10 on last axis
        "U7_fwd_rev_abs_gap5": gap,
        "U8_pos_minus_neg_std": std10(ll_pos_z) - std10(ll_neg_z),
    }


def build_estimators_test(raw_test):
    """raw_test (5,N,10,2) -> dict name -> (5,N) per-fold scores."""
    F, N, K, _ = raw_test.shape
    llr = raw_test[:, :, :, 1] - raw_test[:, :, :, 0]  # (5,N,10)
    ll_neg = raw_test[:, :, :, 0]
    ll_pos = raw_test[:, :, :, 1]
    # col_z per fold
    llr_z = np.stack([col_z(llr[f]) for f in range(F)], 0)
    z = np.stack(
        [np.stack([L(llr[f, :, k]) for k in range(K)], axis=1) for f in range(F)], 0
    )
    ll_pos_z = np.stack([col_z(ll_pos[f]) for f in range(F)], 0)
    ll_neg_z = np.stack([col_z(ll_neg[f]) for f in range(F)], 0)

    out = {}
    out["U1_q80_q20_z"] = np.quantile(z, 0.8, axis=2) - np.quantile(z, 0.2, axis=2)
    out["U2_gini10_z"] = gini10(z)
    out["U3_mad10_z"] = mad10(z)
    out["U4_std10_z"] = std10(z)
    out["U5_gini10_llr"] = gini10(llr_z)
    P = 0.5 * (llr_z[:, :, :5] + llr_z[:, :, 5:])
    out["U6_pair_std5_llr"] = std10(P)
    out["U7_fwd_rev_abs_gap5"] = np.mean(np.abs(llr_z[:, :, :5] - llr_z[:, :, 5:]), axis=2)
    out["U8_pos_minus_neg_std"] = std10(ll_pos_z) - std10(ll_neg_z)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert NO_SUBMIT

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    te0 = folds[0][1]

    Z = np.load(Q17, allow_pickle=True)
    names = [str(s) for s in Z["stream_names"]]
    XO = {n: Z[f"oof_{n}"].ravel() for n in names}
    XT = {n: Z[f"test_{n}"].ravel() for n in names}

    live = pd.read_csv(LIVE)
    ids = live["id"].to_numpy()
    lv = live["is_fraud"].to_numpy(float)
    slot1 = pd.read_csv(SLOT1).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    l7f_o = XO["lm7_5fold"]
    l7f_t = XT["lm7_5fold"]

    # ========== STEP 1: Slot1 sign ==========
    print("=== STEP 1: Slot1 sign audit ===")
    delta_s1 = L(slot1) - L(lv)
    R3_t_Lfolds = np.stack([L(R3_te[f]) for f in range(5)], 0)
    z_s2026_t = L(R3_t_Lfolds.mean(0))
    z_l7f_t = L(l7f_t)
    raw_diff_t = z_s2026_t - z_l7f_t
    # also mean of L(R3_te) without outer L for "L(R3_test)-L(L7F)" as user wrote
    raw_diff_t_alt = L(R3_te.mean(0)) - z_l7f_t
    w_tail_t = 0.5 + 0.5 * U(lv)
    raw_E_t = raw_diff_t * w_tail_t

    # rebuild FE_fixed (V16.8)
    z_s2026_o = L(R3)
    z_l7f_o = L(l7f_o)
    raw_E_o = (z_s2026_o - z_l7f_o) * (0.5 + 0.5 * U(z_ath))
    z_ath_t = L(lv)
    Xo_e = np.column_stack([unitize(z_ath), unitize(d1)])
    Xt_e = np.column_stack([unitize(z_ath_t), unitize(d1t)])
    dE_o, dE_t, _ = frozen_lofo_orth(raw_E_o, raw_E_t, Xo_e, Xt_e, folds, ridge_alpha=1.0)
    dE_o, dE_t = unitize(dE_o), unitize(dE_t)
    # V16.8 oriented on fold0 AUC(+):
    if roc_auc_score(y[te0], z_ath[te0] + 0.01 * dE_o[te0]) < roc_auc_score(
        y[te0], z_ath[te0] - 0.01 * dE_o[te0]
    ):
        dE_o, dE_t = -dE_o, -dE_t
        fe_orient = "flipped_to_plus_AUC"
    else:
        fe_orient = "kept_plus_AUC"

    # OLD Family_E test (V16.7-style separate OLS)
    from scipy.special import expit
    from sklearn.linear_model import LinearRegression

    U_sig = expit((z_ath - np.median(z_ath)) / (np.std(z_ath) + 1e-12))
    raw_old = (L(R3) - L(l7f_o)) * (0.5 + 0.5 * U_sig)
    B2 = np.column_stack([unitize(z_ath), unitize(d1)])
    dE_old = unitize(raw_old - LinearRegression().fit(B2, raw_old).predict(B2))
    w_old_t = 0.5 + 0.5 * expit((z_ath_t - np.median(z_ath_t)) / (np.std(z_ath_t) + 1e-12))
    raw_old_t = (L(R3_te.mean(0)) - L(l7f_t)) * w_old_t
    dE_old_t = unitize(raw_old_t - LinearRegression().fit(Xt_e, raw_old_t).predict(Xt_e))
    # orient old on OOF fold0 same way
    if roc_auc_score(y[te0], z_ath[te0] + 0.01 * dE_old[te0]) < roc_auc_score(
        y[te0], z_ath[te0] - 0.01 * dE_old[te0]
    ):
        dE_old, dE_old_t = -dE_old, -dE_old_t

    pear = {
        "raw_diff_L_meanL": float(np.corrcoef(delta_s1, raw_diff_t)[0, 1]),
        "raw_diff_L_mean_raw": float(np.corrcoef(delta_s1, raw_diff_t_alt)[0, 1]),
        "raw_E": float(np.corrcoef(delta_s1, raw_E_t)[0, 1]),
        "FE_fixed": float(np.corrcoef(delta_s1, dE_t)[0, 1]),
        "FE_old": float(np.corrcoef(delta_s1, dE_old_t)[0, 1]),
    }
    spear = {
        "raw_diff_L_meanL": float(spearmanr(delta_s1, raw_diff_t).statistic),
        "raw_diff_L_mean_raw": float(spearmanr(delta_s1, raw_diff_t_alt).statistic),
        "raw_E": float(spearmanr(delta_s1, raw_E_t).statistic),
        "FE_fixed": float(spearmanr(delta_s1, dE_t).statistic),
        "FE_old": float(spearmanr(delta_s1, dE_old_t).statistic),
    }
    print("  delta_Slot1 = L(Slot1)-L(LIVE)")
    for k in pear:
        print(f"  vs {k}: pearson={pear[k]:+.4f} spearman={spear[k]:+.4f}")
    sp_old_fix = float(spearmanr(dE_old_t, dE_t).statistic)
    print(f"  spearman(FE_old_test, FE_fixed_test)={sp_old_fix:+.4f}  → 1-ρ={1-sp_old_fix:.4f}")
    print(f"  FE_fixed orient: {fe_orient}")

    # OOF both signs of FE_fixed
    print("  OOF ±FE_fixed on z_ATH:")
    fe_pm = {}
    for sign, tag in ((+1, "plus"), (-1, "minus")):
        fe_pm[tag] = []
        for w in (0.006, 0.01045, 0.015):
            d = [
                (
                    roc_auc_score(y[te], z_ath[te] + sign * w * dE_o[te])
                    - roc_auc_score(y[te], z_ath[te])
                )
                * 1e5
                for _, te in folds
            ]
            row = {"w": w, "pos": int(sum(x > 0 for x in d)), "mean": float(np.mean(d)), "folds": [round(x, 3) for x in d]}
            fe_pm[tag].append(row)
            print(f"    {tag} w={w}: pos={row['pos']}/5 mean={row['mean']:.3f} folds={row['folds']}")

    slot1_dir = "PLUS_(R3-L7F)" if pear["raw_diff_L_meanL"] > 0 else "MINUS_(R3-L7F)"
    print(f"  VERDICT: Slot1 moved in {slot1_dir} direction (pearson raw_diff={pear['raw_diff_L_meanL']:+.4f})")

    step1 = {
        "pearson": pear,
        "spearman": spear,
        "spearman_FE_old_vs_fixed_test": sp_old_fix,
        "one_minus_spearman_old_fixed": float(1 - sp_old_fix),
        "FE_fixed_orient": fe_orient,
        "slot1_direction": slot1_dir,
        "fe_pm_oof": fe_pm,
        "why_1mrho_was_2": (
            "V16.8 stored 1-spearman without noticing anti-correlation: "
            f"spearman(old,fixed)={sp_old_fix:.4f} so 1-ρ≈{1-sp_old_fix:.4f}. "
            "Caused by different raw construction (mean L vs L(mean), sigmoid vs Uniform w_tail, "
            "separate Test OLS vs frozen β) plus independent OOF fold0 orientation of each."
        ),
    }

    # z_S3 with FE_fixed (same as V16.8)
    z_s1 = z_ath + 0.01045 * dE_o
    z_s3 = z_s1 + 0.0055 * unitize(d1)
    print(f"AUC z_S3={roc_auc_score(y,z_s3):.7f}")

    # ========== STEP 2: estimators ==========
    print("=== STEP 2: GPU order estimators ===")
    raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
    print(f"  raw oof={raw_o.shape} test={raw_t.shape}")

    est_o = build_estimators_oof(raw_o)
    est_t = build_estimators_test(raw_t)  # (5,N) each

    # Bo9 / Bt9
    ms_o = L(np.load(REPO / "preds/P_super_v4_oof.npy").ravel())
    ms_t = L(np.load(REPO / "preds/P_super_v4_test.npy").ravel())
    g26_o = unitize(L(np.load(G26_OOF).ravel()[: len(y)]))
    g26_t = unitize(L(np.load(G26_T0).ravel()[:299984]))
    Bo9 = np.column_stack(
        [
            unitize(ms_o),
            g26_o,
            unitize(L(XO["megayak_5f_lm"])),
            unitize(z_l7f_o),
            unitize(XO["daoviet_gpdvae"]),
            unitize(XO["gb_bag5"]),
            unitize(XO["yekenot_realmlp"]),
            unitize(d1),
            unitize(dE_o),
        ]
    )
    Bt9 = np.column_stack(
        [
            unitize(ms_t),
            g26_t,
            unitize(L(XT["megayak_5f_lm"])),
            unitize(z_l7f_t),
            unitize(XT["daoviet_gpdvae"]),
            unitize(XT["gb_bag5"]),
            unitize(XT["yekenot_realmlp"]),
            unitize(d1t),
            unitize(dE_t),
        ]
    )
    live_L = L(lv)
    live_hat = Ridge(alpha=RIDGE_ALPHA).fit(Bt9, live_L).predict(Bt9)
    live_resid = live_L - live_hat

    # U1 reference for sign
    u1_o_ref = L_fold(est_o["U1_q80_q20_z"], fold_id)
    u1_t_folds = np.stack([L(est_t["U1_q80_q20_z"][f]) for f in range(5)], 0)
    u1_t_ref = L(u1_t_folds.mean(0))

    results = {}
    oriented = {}
    names_u = list(est_o.keys())

    def prep_signal(name, o_raw, t_folds):
        # cross-fold reliability on raw per-fold scores
        cfr, cfr_list = cross_fold_spearman(t_folds)
        # pre-unitize: L_fold(oof) vs mean_f(L(test_f))
        o_Lf = L_fold(o_raw, fold_id)
        t_Lfolds = np.stack([L(t_folds[f]) for f in range(5)], 0)
        t_meanL = t_Lfolds.mean(0)
        pre_ratio = float((t_meanL.std() + 1e-12) / (o_Lf.std() + 1e-12))
        # final v: L of mean of L(folds) for test; L_fold for oof — then frozen orth unitizes
        v_o = o_Lf
        v_t = L(t_meanL)  # outer L to lock scale like V16.8 Family_E
        # align sign to U1 on OOF
        if name != "U1_q80_q20_z":
            if np.corrcoef(v_o, u1_o_ref)[0, 1] < 0:
                v_o, v_t = -v_o, -v_t
                t_folds = -t_folds
        return v_o, v_t, cfr, cfr_list, pre_ratio, t_folds

    print("=== STEP 3: scorecard ===")
    for name in names_u:
        v_o, v_t, cfr, cfr_list, pre_ratio, _ = prep_signal(name, est_o[name], est_t[name])
        oo, ot, _ = frozen_lofo_orth(v_o, v_t, Bo9, Bt9, folds)
        gates = forensic_gates(oo, ot, Bt9, live_resid)
        scr, u = screen(name, oo, z_s3, y, folds, te0)
        ut = ot if np.corrcoef(u, oo)[0, 1] > 0 else -ot
        oriented[name] = (u, unitize(ut))
        results[name] = {
            **gates,
            **{k: v for k, v in scr.items()},
            "cross_fold_r": cfr,
            "cross_fold_r_pairs": cfr_list,
            "raw_pre_unitize_std_ratio": pre_ratio,
        }
        print(
            f"  META {name}: cross_fold_r={cfr:.4f} pre_std_ratio={pre_ratio:.4f} "
            f"F1={gates['F1']}({gates['disp']:.3f}) F2={gates['F2']}({gates['max_basis_leak']:.3f}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:.3f}) ver={gates['verified']}"
        )

    # Ensemble U_ens = unitize(U2+U5+U6) in raw L_fold space before orth
    e2 = L_fold(est_o["U2_gini10_z"], fold_id)
    e5 = L_fold(est_o["U5_gini10_llr"], fold_id)
    e6 = L_fold(est_o["U6_pair_std5_llr"], fold_id)
    if np.corrcoef(e2, u1_o_ref)[0, 1] < 0:
        e2 = -e2
    if np.corrcoef(e5, u1_o_ref)[0, 1] < 0:
        e5 = -e5
    if np.corrcoef(e6, u1_o_ref)[0, 1] < 0:
        e6 = -e6
    ens_o = unitize(e2 + e5 + e6)
    t2 = np.stack([L(est_t["U2_gini10_z"][f]) for f in range(5)], 0)
    t5 = np.stack([L(est_t["U5_gini10_llr"][f]) for f in range(5)], 0)
    t6 = np.stack([L(est_t["U6_pair_std5_llr"][f]) for f in range(5)], 0)
    if np.corrcoef(t2.mean(0), u1_t_ref)[0, 1] < 0:
        t2 = -t2
    if np.corrcoef(t5.mean(0), u1_t_ref)[0, 1] < 0:
        t5 = -t5
    if np.corrcoef(t6.mean(0), u1_t_ref)[0, 1] < 0:
        t6 = -t6
    ens_t_folds = t2 + t5 + t6
    cfr_ens, cfr_ens_list = cross_fold_spearman(ens_t_folds)
    ens_t = L(ens_t_folds.mean(0))
    pre_ens = float((ens_t_folds.mean(0).std() + 1e-12) / (ens_o.std() + 1e-12))
    oo, ot, _ = frozen_lofo_orth(ens_o, ens_t, Bo9, Bt9, folds)
    gates = forensic_gates(oo, ot, Bt9, live_resid)
    scr, u = screen("U_ens", oo, z_s3, y, folds, te0)
    oriented["U_ens"] = (u, unitize(ot if np.corrcoef(u, oo)[0, 1] > 0 else -ot))
    results["U_ens"] = {
        **gates,
        **{k: v for k, v in scr.items()},
        "cross_fold_r": cfr_ens,
        "cross_fold_r_pairs": cfr_ens_list,
        "raw_pre_unitize_std_ratio": pre_ens,
    }
    print(
        f"  META U_ens: cross_fold_r={cfr_ens:.4f} pre_std_ratio={pre_ens:.4f} "
        f"verified={gates['verified']}"
    )

    u1_cfr = results["U1_q80_q20_z"]["cross_fold_r"]
    winners = []
    for n, r in results.items():
        if not r["verified"]:
            continue
        if r["main"]["pos"] < 5:
            continue
        if r["sf25"] < 18 or r["t"] < 2.20:
            continue
        if r["cross_fold_r"] <= u1_cfr:
            continue
        winners.append(n)
    print(f"U1 cross_fold_r={u1_cfr:.4f}")
    print(f"STAGE winners={winners}")

    staged = []
    for n in winners:
        u_o, u_t = oriented[n]
        wstar = results[n]["main"]["w"]
        cand = qlock(L(slot3) + wstar * u_t, slot3)
        path = SUB / f"sub_v16_9_{n}_NOSUBMIT.csv"
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
        staged.append(
            {
                "file": path.name,
                "signal": n,
                "w": wstar,
                "pos": results[n]["main"]["pos"],
                "sf25": results[n]["sf25"],
                "t": results[n]["t"],
                "cross_fold_r": results[n]["cross_fold_r"],
            }
        )
        print("STAGED", path.name)

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

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "step1": step1,
            "u1_cross_fold_r": u1_cfr,
            "results": results,
            "winners": winners,
            "staged": staged,
            "auc_z_s3": float(roc_auc_score(y, z_s3)),
        }
    )
    OUT.joinpath("v16_9.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v16_9.json")
    print("DONE")


if __name__ == "__main__":
    main()
