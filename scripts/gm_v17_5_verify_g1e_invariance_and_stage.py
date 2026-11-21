#!/usr/bin/env python3
"""GM V17.5 — Grandmaster verification of G1e invariance + calibrated staging.

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t as tdist, wilcoxon
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
SLOT1 = SUB / "sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
G1E_CSV = SUB / "sub_v17_4_d_all_G1e_risk_gated_d__Bo9_NOSUBMIT.csv"
NO_SUBMIT = True
RIDGE_ALPHA = 10.0
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_GRID = (0.010, 0.015, 0.020, 0.025, 0.030, 0.035)


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


def orient_plus(u, zbase, y, te0):
    u = unitize(u)
    flipped = False
    if roc_auc_score(y[te0], zbase[te0] + 0.01 * u[te0]) < roc_auc_score(
        y[te0], zbase[te0] - 0.01 * u[te0]
    ):
        u = -u
        flipped = True
    return u, flipped


def score_at_w(u, zbase, y, folds, w):
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
    t = mean / (se + 1e-18)
    try:
        _, wp = wilcoxon(d25, alternative="greater")
        wp = float(wp)
    except Exception:
        wp = float("nan")
    tcrit = float(tdist.ppf(0.95, len(d25) - 1))
    return {
        "w": w,
        "pos": int(sum(x > 0 for x in d)),
        "mean": float(np.mean(d)),
        "min": float(np.min(d)),
        "folds": [round(x, 3) for x in d],
        "sf25": int((d25 > 0).sum()),
        "t": float(t),
        "mean_sf": mean,
        "wilcoxon_p": wp,
        "ci90": [float(mean - tcrit * se), float(mean + tcrit * se)],
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


def main():
    assert NO_SUBMIT
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
    dE_o = unitize(cache["dE"])
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
    g1e_csv = pd.read_csv(G1E_CSV).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    l7f_o, l7f_t = XO["lm7_5fold"], XT["lm7_5fold"]
    R3_t_Lfolds = np.stack([L(R3_te[f]) for f in range(5)], 0)

    z_ath_t = L(lv)
    raw_E_o = (L(R3) - L(l7f_o)) * (0.5 + 0.5 * U(z_ath))
    raw_E_t = (L(R3_t_Lfolds.mean(0)) - L(l7f_t)) * (0.5 + 0.5 * U(lv))
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
    g26_o = np.load(G26_OOF).ravel()[: len(y)]
    g26_t = np.load(G26_T0).ravel()[:299984]
    lm_o, lm_t = XO["megayak_5f_lm"], XT["megayak_5f_lm"]
    gpd_o, gpd_t = XO["daoviet_gpdvae"], XT["daoviet_gpdvae"]
    gb_o, gb_t = XO["gb_bag5"], XT["gb_bag5"]
    mlp_o, mlp_t = XO["yekenot_realmlp"], XT["yekenot_realmlp"]

    z_MS_o, z_MS_t = L_fold(ms_o, fold_id), L(ms_t)
    z_G26_o, z_G26_t = L_fold(g26_o, fold_id), L(g26_t)
    z_LM_o, z_LM_t = L_fold(lm_o, fold_id), L(lm_t)
    z_L7F_o, z_L7F_t = L_fold(l7f_o, fold_id), L(l7f_t)
    z_R3_o, z_R3_t = L_fold(R3, fold_id), L(R3_t_Lfolds.mean(0))

    P_BPE_o = unitize(0.55 * z_MS_o + 0.45 * z_G26_o)
    P_BPE_t = unitize(0.55 * z_MS_t + 0.45 * z_G26_t)
    P_QLM_o = unitize(0.45 * z_LM_o + 0.30 * z_L7F_o + 0.25 * z_R3_o)
    P_QLM_t = unitize(0.45 * z_LM_t + 0.30 * z_L7F_t + 0.25 * z_R3_t)

    d_all_o = unitize(L_fold(P_BPE_o - P_QLM_o, fold_id))
    d_all_t = unitize(L(P_BPE_t - P_QLM_t))
    raw_g1e_o = d_all_o * unitize(z_s3)
    raw_g1e_t = d_all_t * unitize(L(slot3))

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
    live_L = L(lv)
    live_hat = Ridge(alpha=RIDGE_ALPHA).fit(Bt9, live_L).predict(Bt9)
    live_resid = live_L - live_hat

    oo, ot, _ = frozen_lofo_orth(raw_g1e_o, raw_g1e_t, Bo9, Bt9, folds)
    gates = forensic_gates(oo, ot, Bt9, live_resid)
    u_o, flipped = orient_plus(oo, z_s3, y, te0)
    # match test orientation to OOF
    if np.corrcoef(u_o, oo)[0, 1] < 0:
        u_t = unitize(-ot)
    else:
        u_t = unitize(ot)

    # ----- Check 1: sign & pair decomposition -----
    print("=== CHECK 1: sign & pair drivers ===")
    # In high-risk tail z_S3>0: winning G1e = flipped? * orth(d_all * U(z_S3))
    # Before orth, sign(d_all)>0 means P_BPE > P_QLM.
    # After orientation, corr(u, d_all * U(z_s3)_pos) tells direction.
    tail = z_s3 > 0
    raw_pre = d_all_o * unitize(z_s3)
    # u aligned: positive add helps AUC. On tail, mean(u * sign(d_all)) quantifies BPE vs QLM
    corr_u_d = float(np.corrcoef(u_o, d_all_o)[0, 1])
    corr_u_raw = float(np.corrcoef(u_o, raw_pre)[0, 1])
    # On tail rows: when d_all>0 (BPE>QLM), is u positive on average?
    d_pos_tail = tail & (d_all_o > 0)
    d_neg_tail = tail & (d_all_o < 0)
    mean_u_when_bpe_high = float(u_o[d_pos_tail].mean()) if d_pos_tail.any() else float("nan")
    mean_u_when_qlm_high = float(u_o[d_neg_tail].mean()) if d_neg_tail.any() else float("nan")
    # Prefer BPE if mean_u_when_bpe_high > 0 under winning orientation
    prefers = "P_BPE (MS+G26)" if mean_u_when_bpe_high > 0 else "P_QLM (LM+L7F+R3)"

    pairs = {
        "MS-LM": (z_MS_o - z_LM_o, z_MS_t - z_LM_t),
        "MS-L7F": (z_MS_o - z_L7F_o, z_MS_t - z_L7F_t),
        "G26-LM": (z_G26_o - z_LM_o, z_G26_t - z_LM_t),
        "G26-L7F": (z_G26_o - z_L7F_o, z_G26_t - z_L7F_t),
        "MS-R3": (z_MS_o - z_R3_o, z_MS_t - z_R3_t),
    }
    pair_rows = {}
    for pname, (diff_o, diff_t) in pairs.items():
        dpo = unitize(L_fold(diff_o, fold_id))
        dpt = unitize(L(diff_t))
        vo = dpo * unitize(z_s3)
        vt = dpt * unitize(L(slot3))
        po, pt, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds)
        pu, pflip = orient_plus(po, z_s3, y, te0)
        # align pair orientation to G1e u_o
        if np.corrcoef(pu, u_o)[0, 1] < 0:
            pu = -pu
            pflip = not pflip
        row = score_at_w(pu, z_s3, y, folds, 0.035)
        # also per-fold contribution correlation with G1e fold deltas
        pair_rows[pname] = {
            "flipped_vs_raw": pflip,
            "corr_with_G1e_u": float(np.corrcoef(pu, u_o)[0, 1]),
            **row,
        }
        print(
            f"  PAIR {pname}: {row['pos']}/5 @{0.035} mean={row['mean']:.3f} "
            f"min={row['min']:.3f} sf={row['sf25']}/25 t={row['t']:.2f} "
            f"folds={row['folds']} corr_G1e={pair_rows[pname]['corr_with_G1e_u']:+.3f}"
        )

    # fold-wise std of d_all and of G1e within each main fold
    fold_diag = []
    for f, (_, te) in enumerate(folds):
        fold_diag.append(
            {
                "fold": f,
                "std_d_all": float(d_all_o[te].std()),
                "std_g1e_raw": float(raw_pre[te].std()),
                "mean_abs_d_all": float(np.abs(d_all_o[te]).mean()),
                "frac_bpe_gt_qlm": float((d_all_o[te] > 0).mean()),
                "corr_d_zS3": float(np.corrcoef(d_all_o[te], z_s3[te])[0, 1]),
            }
        )
    check1 = {
        "flipped": flipped,
        "corr_u_vs_d_all": corr_u_d,
        "corr_u_vs_raw_g1e": corr_u_raw,
        "mean_u_tail_when_BPE_gt_QLM": mean_u_when_bpe_high,
        "mean_u_tail_when_QLM_gt_BPE": mean_u_when_qlm_high,
        "high_risk_tail_prefers": prefers,
        "interpretation": (
            f"flipped={flipped}: winning add increases score where oriented G1e>0. "
            f"On z_S3>0 tail, mean(u|BPE>QLM)={mean_u_when_bpe_high:+.4f} vs "
            f"mean(u|QLM>BPE)={mean_u_when_qlm_high:+.4f} → prefers {prefers}."
        ),
        "fold_diag": fold_diag,
        "pairs": pair_rows,
        "gates": gates,
    }
    print(f"  flipped={flipped} prefers_tail={prefers}")
    print(f"  corr(u,d_all)={corr_u_d:+.4f} corr(u,raw_g1e)={corr_u_raw:+.4f}")

    # ----- Check 2: weight invariance -----
    print("=== CHECK 2: basis-weight invariance ===")
    # Slot3 twin: Ridge α=1 of unitize(L(Slot3)) on Bt9
    y_s3t = unitize(L(slot3))
    ridge_s3 = Ridge(alpha=1.0, fit_intercept=True).fit(Bt9, y_s3t)
    r2 = float(ridge_s3.score(Bt9, y_s3t))
    beta_s3 = ridge_s3.coef_.copy()
    z_slot3_twin = Bo9 @ beta_s3  # ignore intercept for relative; add intercept for level
    z_slot3_twin = z_slot3_twin + ridge_s3.intercept_
    # Logit OOF stack
    z_logit = np.zeros(len(y))
    for tr, te in folds:
        clf = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=500, fit_intercept=True
        )
        # sklearn 1.8: use C large / penalty none
        try:
            clf.fit(Bo9[tr], y[tr])
        except TypeError:
            clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
            clf.fit(Bo9[tr], y[tr])
        z_logit[te] = clf.decision_function(Bo9[te])

    baselines = {
        "z_S3": z_s3,
        "z_Slot3_twin": unitize(z_slot3_twin),
        "z_Logit_oof": unitize(z_logit),
    }
    print(f"  Slot3 Ridge(α=1) R²={r2:.4f}")
    print(f"  beta_Slot3: { {BN9[j]: round(float(beta_s3[j]),4) for j in range(9)} }")
    print(
        f"  AUC twin={roc_auc_score(y,z_slot3_twin):.7f} "
        f"logit={roc_auc_score(y,z_logit):.7f} S3={roc_auc_score(y,z_s3):.7f}"
    )

    inv = {}
    for bname, zbase in baselines.items():
        # re-orient u on this baseline's fold0 (keep physical u from S3 if corr high)
        u_b, flip_b = orient_plus(oo, zbase, y, te0)
        inv[bname] = {
            "flipped_vs_S3_orient": bool(np.corrcoef(u_b, u_o)[0, 1] < 0) or flip_b != flipped,
            "orient_flipped": flip_b,
            "auc": float(roc_auc_score(y, zbase)),
            "by_w": {},
        }
        print(f"  -- baseline {bname} AUC={inv[bname]['auc']:.7f} flipped={flip_b}")
        for w in W_GRID:
            row = score_at_w(u_b, zbase, y, folds, w)
            inv[bname]["by_w"][str(w)] = row
            print(
                f"    w={w:.3f}: {row['pos']}/5 mean={row['mean']:.3f} min={row['min']:.3f} "
                f"sf={row['sf25']}/25 t={row['t']:.2f} folds={row['folds']}"
            )

    # ----- Check 3: spearman steps + stage -----
    print("=== CHECK 3: spearman steps + stage ===")
    # Rebuild cand at w=0.035 from u_t (should match v17.4 csv closely)
    cand035 = qlock(L(slot3) + 0.035 * u_t, slot3)
    rho_s3 = float(spearmanr(slot3, g1e_csv).statistic)
    rho_s1 = float(spearmanr(slot1, g1e_csv).statistic)
    rho_lv = float(spearmanr(lv, g1e_csv).statistic)
    rho_s3_rebuild = float(spearmanr(slot3, cand035).statistic)
    steps = {
        "v17_4_csv": {
            "1mrho_Slot3_e4": (1.0 - rho_s3) * 1e4,
            "1mrho_Slot1_e4": (1.0 - rho_s1) * 1e4,
            "1mrho_LIVE_e4": (1.0 - rho_lv) * 1e4,
        },
        "rebuild_w035": {"1mrho_Slot3_e4": (1.0 - rho_s3_rebuild) * 1e4},
        "spearman_v17_4_vs_rebuild": float(spearmanr(g1e_csv, cand035).statistic),
    }
    print(
        f"  v17.4 csv 1-ρ Slot3/Slot1/LIVE (e-4) = "
        f"{steps['v17_4_csv']['1mrho_Slot3_e4']:.4f} / "
        f"{steps['v17_4_csv']['1mrho_Slot1_e4']:.4f} / "
        f"{steps['v17_4_csv']['1mrho_LIVE_e4']:.4f}"
    )

    staged = []
    for w, tag in ((0.018, "w0p018"), (0.025, "w0p025")):
        # OOF metrics on z_S3 at this w
        oof = score_at_w(u_o, z_s3, y, folds, w)
        # also on Slot3 twin
        oof_twin = score_at_w(
            orient_plus(oo, baselines["z_Slot3_twin"], y, te0)[0],
            baselines["z_Slot3_twin"],
            y,
            folds,
            w,
        )
        cand = qlock(L(slot3) + w * u_t, slot3)
        path = SUB / f"sub_v17_5_Slot3_plus_G1e_{tag}_NOSUBMIT.csv"
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
        # also Slot1-based variant
        cand1 = qlock(L(slot1) + w * u_t, slot1)
        path1 = SUB / f"sub_v17_5_Slot1_plus_G1e_{tag}_NOSUBMIT.csv"
        pd.DataFrame({"id": ids, "is_fraud": cand1}).to_csv(path1, index=False)
        one_m_s3 = (1.0 - float(spearmanr(slot3, cand).statistic)) * 1e4
        one_m_s1 = (1.0 - float(spearmanr(slot1, cand1).statistic)) * 1e4
        entry = {
            "file_slot3": path.name,
            "file_slot1": path1.name,
            "w": w,
            "oof_z_S3": oof,
            "oof_z_Slot3_twin": oof_twin,
            "1mrho_Slot3_e4": one_m_s3,
            "1mrho_Slot1_base_e4": one_m_s1,
        }
        staged.append(entry)
        print(
            f"  STAGED {path.name}: OOF {oof['pos']}/5 mean={oof['mean']:.3f} "
            f"sf={oof['sf25']} t={oof['t']:.2f} | 1-ρ(Slot3)={one_m_s3:.4f}e-4"
        )
        print(f"  STAGED {path1.name}: 1-ρ(Slot1)={one_m_s1:.4f}e-4")

    # invariance summary flag
    def ok5(bname, w=0.035):
        r = inv[bname]["by_w"][str(w)]
        return r["pos"] >= 5 and r["min"] > 0

    invariance_pass = all(ok5(b) for b in baselines)
    print(f"INVARIANCE @w=0.035 all-baselines 5/5+: {invariance_pass}")

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "check1": check1,
            "check2": {
                "slot3_ridge_r2": r2,
                "beta_Slot3_test": {BN9[j]: float(beta_s3[j]) for j in range(9)},
                "intercept": float(ridge_s3.intercept_),
                "invariance": inv,
                "invariance_pass_w035": invariance_pass,
            },
            "check3": {"steps": steps, "staged": staged},
        }
    )
    OUT.joinpath("v17_5.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v17_5.json")
    print("DONE")


if __name__ == "__main__":
    main()
