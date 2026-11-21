#!/usr/bin/env python3
"""GM V17.4 — Cure G1 F2/F3 leak via pre-nonlinearity distribution locking.

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, rankdata, skew, t as tdist, wilcoxon
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
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
NO_SUBMIT = True
RIDGE_ALPHA = 10.0
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
STAGE_MIN_SF = 18
STAGE_MIN_T = 2.50


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
    orth_t = orth_t - orth_t.mean()
    orth_o = orth_o - orth_o.mean()
    return orth_o, orth_t, beta_bar


def forensic_gates(orth_o, orth_t, Bt, live_resid):
    disp = float((orth_t.std() + 1e-12) / (orth_o.std() + 1e-12))
    leaks = [float(np.corrcoef(orth_t, Bt[:, j])[0, 1]) for j in range(Bt.shape[1])]
    max_leak = float(np.max(np.abs(leaks)))
    leak_argmax = int(np.argmax(np.abs(leaks)))
    c_live = float(np.corrcoef(orth_t, live_resid)[0, 1]) if live_resid.std() > 1e-12 else 0.0
    return {
        "disp": disp,
        "max_basis_leak": max_leak,
        "leak_argmax": leak_argmax,
        "leak_by_col": leaks,
        "corr_live_resid": c_live,
        "F1": 0.94 <= disp <= 1.06,
        "F2": max_leak < 0.035,
        "F3": abs(c_live) < 0.05,
        "verified": (0.94 <= disp <= 1.06) and (max_leak < 0.035) and (abs(c_live) < 0.05),
    }


def screen(name, u, zbase, y, folds, te0):
    u = unitize(u)
    flipped = False
    if roc_auc_score(y[te0], zbase[te0] + 0.01 * u[te0]) < roc_auc_score(
        y[te0], zbase[te0] - 0.01 * u[te0]
    ):
        u = -u
        flipped = True
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


def moments(v, tag):
    v = np.asarray(v, float).ravel()
    return {
        "tag": tag,
        "n": int(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "skew": float(skew(v)),
        "kurtosis_excess": float(kurtosis(v)),
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
    SUB.mkdir(parents=True, exist_ok=True)

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
    dE_o = unitize(cache["dE"]) if "dE" in cache.files else None
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
    l7f_o, l7f_t = XO["lm7_5fold"], XT["lm7_5fold"]
    R3_t_Lfolds = np.stack([L(R3_te[f]) for f in range(5)], 0)

    # Family_E_fixed
    z_ath_t = L(lv)
    raw_E_o = (L(R3) - L(l7f_o)) * (0.5 + 0.5 * U(z_ath))
    raw_E_t = (L(R3_t_Lfolds.mean(0)) - L(l7f_t)) * (0.5 + 0.5 * U(lv))
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
    elif np.corrcoef(dE_o, dE_o2)[0, 1] < 0:
        dE_o = -dE_o

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
    z_GB_o, z_GB_t = L_fold(gb_o, fold_id), L(gb_t)
    z_MLP_o, z_MLP_t = L_fold(mlp_o, fold_id), L(mlp_t)
    z_GPD_o, z_GPD_t = L_fold(gpd_o, fold_id), L(gpd_t)

    P_BPE_o = unitize(0.55 * z_MS_o + 0.45 * z_G26_o)
    P_BPE_t = unitize(0.55 * z_MS_t + 0.45 * z_G26_t)
    P_QLM_o = unitize(0.45 * z_LM_o + 0.30 * z_L7F_o + 0.25 * z_R3_o)
    P_QLM_t = unitize(0.45 * z_LM_t + 0.30 * z_L7F_t + 0.25 * z_R3_t)
    P_TAB_o = unitize(0.50 * z_GB_o + 0.35 * z_MLP_o + 0.15 * z_GPD_o)
    P_TAB_t = unitize(0.50 * z_GB_t + 0.35 * z_MLP_t + 0.15 * z_GPD_t)

    # ----- Step 1: pre-squaring drift -----
    print("=== STEP 1: pre-squaring σ drift ===")
    raw_d_o = P_BPE_o - P_QLM_o
    raw_d_t = P_BPE_t - P_QLM_t
    # V17.3 G1 used raw_d without L-lock
    g1_v173_o = np.maximum(0, raw_d_o) ** 2 - np.maximum(0, -raw_d_o) ** 2
    g1_v173_t = np.maximum(0, raw_d_t) ** 2 - np.maximum(0, -raw_d_t) ** 2
    step1 = {
        "std_raw_d_oof": float(raw_d_o.std()),
        "std_raw_d_test": float(raw_d_t.std()),
        "ratio_test_over_oof": float(raw_d_t.std() / (raw_d_o.std() + 1e-12)),
        "std_G26_oof_unit": float(unitize(z_G26_o).std()),  # 1
        "std_contrib": {
            "std_MS_minus_QLM_o": float((unitize(z_MS_o) - P_QLM_o).std()),
            "std_MS_minus_QLM_t": float((unitize(z_MS_t) - P_QLM_t).std()),
            "std_G26_minus_QLM_o": float((unitize(z_G26_o) - P_QLM_o).std()),
            "std_G26_minus_QLM_t": float((unitize(z_G26_t) - P_QLM_t).std()),
        },
        "moments_raw_d_o": moments(raw_d_o, "raw_d_o"),
        "moments_raw_d_t": moments(raw_d_t, "raw_d_t"),
        "moments_g1_v173_o": moments(g1_v173_o, "g1_v173_o"),
        "moments_g1_v173_t": moments(g1_v173_t, "g1_v173_t"),
    }
    print(
        f"  std(P_BPE-P_QLM) OOF={step1['std_raw_d_oof']:.6f} "
        f"Test={step1['std_raw_d_test']:.6f} ratio={step1['ratio_test_over_oof']:.4f}"
    )
    print(
        f"  std(MS-QLM) o/t={step1['std_contrib']['std_MS_minus_QLM_o']:.4f}/"
        f"{step1['std_contrib']['std_MS_minus_QLM_t']:.4f}  "
        f"std(G26-QLM) o/t={step1['std_contrib']['std_G26_minus_QLM_o']:.4f}/"
        f"{step1['std_contrib']['std_G26_minus_QLM_t']:.4f}"
    )

    d_all_o = unitize(L_fold(raw_d_o, fold_id))
    d_all_t = unitize(L(raw_d_t))
    d_5f_o = unitize(L_fold(unitize(z_MS_o) - P_QLM_o, fold_id))
    d_5f_t = unitize(L(unitize(z_MS_t) - P_QLM_t))
    d_bt_o = unitize(L_fold(P_BPE_o - P_TAB_o, fold_id))
    d_bt_t = unitize(L(P_BPE_t - P_TAB_t))
    d_bt5_o = unitize(L_fold(unitize(z_MS_o) - P_TAB_o, fold_id))
    d_bt5_t = unitize(L(unitize(z_MS_t) - P_TAB_t))

    lock_check = {
        "d_all": {"o": moments(d_all_o, "d_all_o"), "t": moments(d_all_t, "d_all_t")},
        "d_5f": {"o": moments(d_5f_o, "d_5f_o"), "t": moments(d_5f_t, "d_5f_t")},
        "d_bt": {"o": moments(d_bt_o, "d_bt_o"), "t": moments(d_bt_t, "d_bt_t")},
    }
    for k, v in lock_check.items():
        print(
            f"  {k}: mean o/t={v['o']['mean']:.6f}/{v['t']['mean']:.6f} "
            f"std={v['o']['std']:.6f}/{v['t']['std']:.6f} "
            f"skew={v['o']['skew']:.6f}/{v['t']['skew']:.6f} "
            f"kurt={v['o']['kurtosis_excess']:.6f}/{v['t']['kurtosis_excess']:.6f}"
        )

    # Bo9 / Bt9
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

    # Diagnose V17.3 G1 F3: which LIVE residual / basis component correlates
    print("=== STEP 1b: V17.3 G1 F3 autopsy ===")
    oo173, ot173, _ = frozen_lofo_orth(g1_v173_o, g1_v173_t, Bo9, Bt9, folds)
    gates173 = forensic_gates(oo173, ot173, Bt9, live_resid)
    # LIVE residual attribution: corr of ot173 with each Bt9 col of live prediction pieces
    # live_resid = live_L - Bt9 @ β_live - intercept; correlate ot with each centered basis
    # and with partial residuals removing one column
    beta_live = Ridge(alpha=RIDGE_ALPHA).fit(Bt9, live_L)
    live_by_col = {
        BN9[j]: float(np.corrcoef(ot173, Bt9[:, j] * beta_live.coef_[j])[0, 1])
        for j in range(9)
    }
    live_corr_basis = {BN9[j]: float(np.corrcoef(ot173, Bt9[:, j])[0, 1]) for j in range(9)}
    # Also corr with raw G26-MS / BPE-QLM gap on test
    autopsy = {
        "gates_v173_G1": gates173,
        "corr_ot_live_resid": gates173["corr_live_resid"],
        "corr_ot_vs_Bt9": live_corr_basis,
        "corr_ot_vs_Bt9_weighted_by_live_beta": live_by_col,
        "corr_ot_raw_d_t": float(np.corrcoef(ot173, raw_d_t)[0, 1]),
        "corr_ot_unitize_G26": float(np.corrcoef(ot173, unitize(L(g26_t)))[0, 1]),
        "corr_ot_G26_minus_MS": float(
            np.corrcoef(ot173, unitize(L(g26_t)) - unitize(L(ms_t)))[0, 1]
        ),
        "corr_live_resid_vs_G26": float(np.corrcoef(live_resid, unitize(L(g26_t)))[0, 1]),
        "corr_live_resid_vs_raw_d": float(np.corrcoef(live_resid, raw_d_t)[0, 1]),
        "live_ridge_coefs": {BN9[j]: float(beta_live.coef_[j]) for j in range(9)},
    }
    # largest |corr| of ot with basis
    top = sorted(live_corr_basis.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print(f"  V17.3 G1 after orth: F2 leak={gates173['max_basis_leak']:.4f} "
          f"({BN9[gates173['leak_argmax']]}) F3={gates173['corr_live_resid']:+.4f}")
    print(f"  corr(orth_t, Bt9): { {k: round(v,4) for k,v in top[:5]} }")
    print(
        f"  corr(orth_t, G26)={autopsy['corr_ot_unitize_G26']:+.4f} "
        f"corr(live_resid, G26)={autopsy['corr_live_resid_vs_G26']:+.4f} "
        f"corr(live_resid, raw_d)={autopsy['corr_live_resid_vs_raw_d']:+.4f}"
    )

    # ----- Step 2: transforms -----
    print("=== STEP 2: locked G1 family ===")
    z_s3_t_proxy = L(slot3)

    def family(d_o, d_t, d_bt_o_, d_bt_t_, prefix):
        return {
            f"{prefix}_G1a_signed_sq": (d_o * np.abs(d_o), d_t * np.abs(d_t)),
            f"{prefix}_G1b_signed_p15": (
                np.sign(d_o) * (np.abs(d_o) ** 1.5),
                np.sign(d_t) * (np.abs(d_t) ** 1.5),
            ),
            f"{prefix}_G1c_pos_hinge_sq": (np.maximum(0.0, d_o) ** 2, np.maximum(0.0, d_t) ** 2),
            f"{prefix}_G1d_neg_hinge_sq": (np.maximum(0.0, -d_o) ** 2, np.maximum(0.0, -d_t) ** 2),
            f"{prefix}_G1e_risk_gated_d": (
                d_o * unitize(z_s3),
                d_t * unitize(z_s3_t_proxy),
            ),
            f"{prefix}_G1f_bpe_tab_signed_sq": (
                d_bt_o_ * np.abs(d_bt_o_),
                d_bt_t_ * np.abs(d_bt_t_),
            ),
        }

    cands = {}
    cands.update(family(d_all_o, d_all_t, d_bt_o, d_bt_t, "d_all"))
    cands.update(family(d_5f_o, d_5f_t, d_bt5_o, d_bt5_t, "d_5f"))
    # also V17.3 baseline for comparison
    cands["v173_G1_raw"] = (g1_v173_o, g1_v173_t)

    results = {}
    oriented = {}

    def eval_one(name, vo, vt, Bo, Bt, basis_tag, d_o=None, d_t=None):
        oo, ot, bbar = frozen_lofo_orth(vo, vt, Bo, Bt, folds)
        gates = forensic_gates(oo, ot, Bt, live_resid)
        # if Bo10, also report corr with d column
        corr_d = None
        if d_t is not None and Bt.shape[1] >= 10:
            corr_d = float(np.corrcoef(ot, Bt[:, -1])[0, 1])
        scr, u = screen(f"{name}[{basis_tag}]", oo, z_s3, y, folds, te0)
        ut = unitize(ot if np.corrcoef(u, oo)[0, 1] > 0 else -ot)
        key = f"{name}__{basis_tag}"
        oriented[key] = (u, ut)
        bn = BN9 + (["d"] if Bt.shape[1] == 10 else [])
        results[key] = {
            **gates,
            "basis": basis_tag,
            "corr_orth_t_vs_d": corr_d,
            "leak_col_name": bn[gates["leak_argmax"]] if gates["leak_argmax"] < len(bn) else "?",
            **{k: v for k, v in scr.items() if k != "name"},
        }
        print(
            f"  META {key}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{results[key]['leak_col_name']}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) "
            f"corr_d={corr_d if corr_d is not None else float('nan'):+.6f} ver={gates['verified']}"
        )
        return key

    print("=== STEP 3: Bo9 and Bo10 scorecards ===")
    for name, (vo, vt) in cands.items():
        # pick matching d for Bo10
        if name.startswith("d_all"):
            d_o, d_t = d_all_o, d_all_t
        elif name.startswith("d_5f"):
            d_o, d_t = d_5f_o, d_5f_t
        else:
            d_o, d_t = unitize(raw_d_o), unitize(raw_d_t)
        eval_one(name, vo, vt, Bo9, Bt9, "Bo9")
        Bo10 = np.column_stack([Bo9, unitize(d_o)])
        Bt10 = np.column_stack([Bt9, unitize(d_t)])
        eval_one(name, vo, vt, Bo10, Bt10, "Bo10", d_o=d_o, d_t=d_t)

    winners = []
    for k, r in results.items():
        if not r["verified"]:
            continue
        if r["main"]["pos"] < 5:
            continue
        if r["sf25"] < STAGE_MIN_SF or r["t"] < STAGE_MIN_T:
            continue
        winners.append(k)
    print(f"STAGE winners={winners}")

    staged = []
    for k in winners:
        u_o, u_t = oriented[k]
        wstar = results[k]["main"]["w"]
        safe = k.replace("/", "_")
        path = SUB / f"sub_v17_4_{safe}_NOSUBMIT.csv"
        cand = qlock(L(slot3) + wstar * u_t, slot3)
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
        staged.append(
            {
                "file": path.name,
                "signal": k,
                "w": wstar,
                "pos": results[k]["main"]["pos"],
                "sf25": results[k]["sf25"],
                "t": results[k]["t"],
            }
        )
        print("STAGED", path.name)

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "auc_z_s3": float(roc_auc_score(y, z_s3)),
            "step1": step1,
            "lock_check": lock_check,
            "autopsy_v173_G1": autopsy,
            "results": results,
            "winners": winners,
            "staged": staged,
            "stage_rule": "F1&F2&F3 AND 5/5 AND sf>=18 AND t>=2.50",
        }
    )
    OUT.joinpath("v17_4.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v17_4.json")
    print("DONE")


if __name__ == "__main__":
    main()
