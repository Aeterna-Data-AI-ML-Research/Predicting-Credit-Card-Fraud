#!/usr/bin/env python3
"""GM V16.8 — Fix Family_E OOF↔Test drift; frozen Ridge LOFO orth; re-score add-ons.

STRICT NO_SUBMIT=True. 0 GPU pods.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
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
ALEX = REPO / "external_bank/alex_oof"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
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
    """Exact Uniform(0,1) average-rank quantile."""
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


def frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=RIDGE_ALPHA):
    """LOFO Ridge on OOF; apply mean β_oof to Test. Returns orth_o, orth_t_raw, beta_bar, betas."""
    vo = unitize(v_oof)
    vt = unitize(v_test)
    Bo = np.asarray(Bo, float)
    Bt = np.asarray(Bt, float)
    # columns already expected unitized by caller
    orth_o = np.zeros(len(vo))
    betas = []
    for tr, te in folds:
        lr = Ridge(alpha=ridge_alpha, fit_intercept=True)
        lr.fit(Bo[tr], vo[tr])
        betas.append(lr.coef_.copy())
        orth_o[te] = vo[te] - Bo[te] @ lr.coef_ - lr.intercept_
    beta_bar = np.mean(np.stack(betas, 0), axis=0)
    # intercept: average of fold intercepts via residual mean alignment
    # recompute with beta_bar only (no intercept) after centering — match Ridge intercept via means
    # Apply: orth_t = vt - Bt @ beta_bar - b0, choose b0 so orth_t.mean()≈0
    orth_t = vt - Bt @ beta_bar
    orth_t = orth_t - orth_t.mean()
    orth_o = orth_o - orth_o.mean()
    return orth_o, orth_t, beta_bar, betas


def forensic_gates(orth_o, orth_t, Bt, live_resid):
    disp = float((orth_t.std() + 1e-12) / (orth_o.std() + 1e-12))
    max_leak = float(np.max(np.abs([np.corrcoef(orth_t, Bt[:, j])[0, 1] for j in range(Bt.shape[1])])))
    c_live = float(np.corrcoef(orth_t, live_resid)[0, 1]) if live_resid.std() > 1e-12 else 0.0
    f1 = 0.94 <= disp <= 1.06
    f2 = max_leak < 0.035
    f3 = abs(c_live) < 0.05
    return {
        "disp": disp,
        "max_basis_leak": max_leak,
        "corr_live_resid": c_live,
        "F1": f1,
        "F2": f2,
        "F3": f3,
        "verified": f1 and f2 and f3,
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
    out = {
        "name": name,
        "main": best,
        "sf25": int((d25 > 0).sum()),
        "t": float(t),
        "mean_sf": mean,
    }
    print(
        f"  SCREEN {name}: {best['pos']}/5 @{best['w']} mean={best['mean']:.3f} "
        f"min={best['min']:.3f} sf={out['sf25']}/25 t={t:.2f} folds={best['folds']}"
    )
    return out, u


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    assert NO_SUBMIT

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath = cache["z_ath"]
    d1 = cache["d1"]
    y = cache["y"]
    fold_id = cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()

    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    te0 = folds[0][1]

    Z = np.load(Q17, allow_pickle=True)
    names = [str(s) for s in Z["stream_names"]]
    XO = {n: Z[f"oof_{n}"].ravel() for n in names}
    XT = {n: Z[f"test_{n}"].ravel() for n in names}

    live_df = pd.read_csv(LIVE)
    ids = live_df["id"].to_numpy()
    lv = live_df["is_fraud"].to_numpy(float)
    z_ath_t = L(lv)  # LIVE-locked ATH proxy on test
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    # --- streams ---
    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
    R2 = np.load(TMP15 / "out/R2_oof.npy").ravel()
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")  # (5,N)
    R2_te = np.load(TMP15 / "out/R2_test_folds.npy")
    l7f_o = XO["lm7_5fold"]
    l7f_t = XT["lm7_5fold"]
    lm_o = XO["megayak_5f_lm"]
    lm_t = XT["megayak_5f_lm"]

    # Step 1: L7_s2026 with matched std via outer L on test mean
    z_s2026_o = L(R3)
    # Test: L(mean_f(test_f)) then outer L — user asks L(mean_f(L(test_f))) for std match
    # Apply L per fold then mean then L again for std lock to match L(L7F_test)
    R3_t_Lfolds = np.stack([L(R3_te[f]) for f in range(5)], 0)
    z_s2026_t = L(R3_t_Lfolds.mean(0))
    z_l7f_o = L(l7f_o)
    z_l7f_t = L(l7f_t)

    print("=== STEP 1: Family_E fix ===")
    print(
        f"  std z_s2026_o={z_s2026_o.std():.5f} z_l7f_o={z_l7f_o.std():.5f} "
        f"z_s2026_t={z_s2026_t.std():.5f} z_l7f_t={z_l7f_t.std():.5f}"
    )

    diff_o = z_s2026_o - z_l7f_o
    diff_t = z_s2026_t - z_l7f_t
    c_before_o = float(np.corrcoef(diff_o, z_l7f_o)[0, 1])
    c_before_t = float(np.corrcoef(diff_t, z_l7f_t)[0, 1])
    print(f"  corr(diff, L7F) BEFORE w_tail: oof={c_before_o:.4f} test={c_before_t:.4f} Δ={abs(c_before_o-c_before_t):.4f}")

    # w_tail from Uniform ranks of ATH
    u_ath_o = U(z_ath)
    u_ath_t = U(lv)  # LIVE ranks — Uniform(0,1)
    w_tail_o = 0.5 + 0.5 * u_ath_o
    w_tail_t = 0.5 + 0.5 * u_ath_t
    raw_e_o = diff_o * w_tail_o
    raw_e_t = diff_t * w_tail_t
    c_after_o = float(np.corrcoef(raw_e_o, z_l7f_o)[0, 1])
    c_after_t = float(np.corrcoef(raw_e_t, z_l7f_t)[0, 1])
    print(f"  corr(raw_E, L7F) AFTER w_tail: oof={c_after_o:.4f} test={c_after_t:.4f} Δ={abs(c_after_o-c_after_t):.4f}")

    # Frozen LOFO orth of raw_E vs [z_ATH, d1]
    Xo_e = np.column_stack([unitize(z_ath), unitize(d1)])
    Xt_e = np.column_stack([unitize(z_ath_t), unitize(d1t)])
    dE_o, dE_t, beta_e, _ = frozen_lofo_orth(raw_e_o, raw_e_t, Xo_e, Xt_e, folds, ridge_alpha=1.0)
    dE_o = unitize(dE_o)
    dE_t = unitize(dE_t)
    # orient on fold0
    if roc_auc_score(y[te0], z_ath[te0] + 0.01 * dE_o[te0]) < roc_auc_score(
        y[te0], z_ath[te0] - 0.01 * dE_o[te0]
    ):
        dE_o = -dE_o
        dE_t = -dE_t

    c_fe_l7_o = float(np.corrcoef(dE_o, unitize(z_l7f_o))[0, 1])
    c_fe_l7_t = float(np.corrcoef(dE_t, unitize(z_l7f_t))[0, 1])
    dcorr = abs(c_fe_l7_o - c_fe_l7_t)
    fam_disp = float((dE_t.std() + 1e-12) / (dE_o.std() + 1e-12))
    print(
        f"  Family_E_fixed vs L7F: corr_oof={c_fe_l7_o:.4f} corr_test={c_fe_l7_t:.4f} "
        f"|Δ|={dcorr:.4f} (need <0.015) disp={fam_disp:.4f}"
    )

    # Old Family_E (V16.7 style) for comparison
    from sklearn.linear_model import LinearRegression
    from scipy.special import expit

    U_sig = expit((z_ath - np.median(z_ath)) / (np.std(z_ath) + 1e-12))
    raw_old = (L(R3) - L(l7f_o)) * (0.5 + 0.5 * U_sig)
    B2 = np.column_stack([unitize(z_ath), unitize(d1)])
    dE_old = unitize(raw_old - LinearRegression().fit(B2, raw_old).predict(B2))
    R3t_mean = R3_te.mean(0)
    w_old_t = 0.5 + 0.5 * expit((z_ath_t - np.median(z_ath_t)) / (np.std(z_ath_t) + 1e-12))
    raw_old_t = (L(R3t_mean) - L(l7f_t)) * w_old_t
    dE_old_t = unitize(raw_old_t - LinearRegression().fit(Xt_e, raw_old_t).predict(Xt_e))
    c_old_o = float(np.corrcoef(dE_old, unitize(z_l7f_o))[0, 1])
    c_old_t = float(np.corrcoef(dE_old_t, unitize(z_l7f_t))[0, 1])
    print(
        f"  OLD Family_E vs L7F: corr_oof={c_old_o:.4f} corr_test={c_old_t:.4f} "
        f"|Δ|={abs(c_old_o-c_old_t):.4f}"
    )
    rho_fe_change = float(1 - spearmanr(dE_old_t, dE_t).statistic)
    # Family_E_fixed on z_ATH
    scr_fe = []
    for w in (0.006, 0.01045, 0.01383, 0.01655):
        d = [
            (roc_auc_score(y[te], z_ath[te] + w * dE_o[te]) - roc_auc_score(y[te], z_ath[te])) * 1e5
            for _, te in folds
        ]
        scr_fe.append({"w": w, "pos": int(sum(x > 0 for x in d)), "mean": float(np.mean(d)), "folds": [round(x, 3) for x in d]})
        print(f"  Family_E_fixed @w={w}: pos={scr_fe[-1]['pos']}/5 mean={scr_fe[-1]['mean']:.3f} folds={scr_fe[-1]['folds']}")

    step1 = {
        "stds": {
            "z_s2026_o": float(z_s2026_o.std()),
            "z_l7f_o": float(z_l7f_o.std()),
            "z_s2026_t": float(z_s2026_t.std()),
            "z_l7f_t": float(z_l7f_t.std()),
        },
        "corr_diff_L7F_before": {"oof": c_before_o, "test": c_before_t, "dabs": abs(c_before_o - c_before_t)},
        "corr_rawE_L7F_after": {"oof": c_after_o, "test": c_after_t, "dabs": abs(c_after_o - c_after_t)},
        "Family_E_fixed_vs_L7F": {
            "corr_oof": c_fe_l7_o,
            "corr_test": c_fe_l7_t,
            "dabs": dcorr,
            "pass_dcorr": dcorr < 0.015,
            "disp": fam_disp,
            "pass_disp": 0.97 <= fam_disp <= 1.03,
        },
        "OLD_Family_E_vs_L7F": {"corr_oof": c_old_o, "corr_test": c_old_t, "dabs": abs(c_old_o - c_old_t)},
        "1mrho_old_vs_fixed_test": rho_fe_change,
        "beta_E_bar": beta_e.tolist(),
        "screen_on_zATH": scr_fe,
    }

    # z_S1 / z_S3 with Family_E_fixed
    z_s1 = z_ath + 0.01045 * dE_o
    z_s3 = z_s1 + 0.0055 * unitize(d1)
    print(f"AUC z_ATH={roc_auc_score(y,z_ath):.7f} z_S1={roc_auc_score(y,z_s1):.7f} z_S3={roc_auc_score(y,z_s3):.7f}")

    # --- Step 2: 9-stream non-redundant basis (NO z_ATH) ---
    print("=== STEP 2: frozen_lofo_orth 9-basis ===")
    bn9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E_fixed"]
    ms_o = L(np.load(REPO / "preds/P_super_v4_oof.npy").ravel())
    ms_t = L(np.load(REPO / "preds/P_super_v4_test.npy").ravel())
    g26_o = L(np.load(G26_OOF).ravel()[: len(y)]) if G26_OOF.exists() else XO["gpt2_seed2026"]
    g26_t = L(np.load(G26_T0).ravel()[:299984]) if G26_T0.exists() else XT["gpt2_seed2026"]
    g26_o = unitize(np.asarray(g26_o, float).ravel()[: len(y)])
    g26_t = unitize(np.asarray(g26_t, float).ravel()[:299984])
    Bo9 = np.column_stack(
        [
            unitize(ms_o),
            g26_o,
            unitize(L(lm_o)),
            unitize(z_l7f_o),
            unitize(XO["daoviet_gpdvae"]),
            unitize(XO["gb_bag5"]),
            unitize(XO["yekenot_realmlp"]),
            unitize(d1),
            dE_o,
        ]
    )
    Bt9 = np.column_stack(
        [
            unitize(ms_t),
            g26_t,
            unitize(L(lm_t)),
            unitize(z_l7f_t),
            unitize(XT["daoviet_gpdvae"]),
            unitize(XT["gb_bag5"]),
            unitize(XT["yekenot_realmlp"]),
            unitize(d1t),
            dE_t,
        ]
    )
    print(f"  basis {bn9} cond(Bo9)={np.linalg.cond(Bo9):.1f} cond(Bt9)={np.linalg.cond(Bt9):.1f}")
    # corr Family_E_fixed vs L7F in 9-basis
    print(
        f"  in-basis corr(FE,L7F) oof={np.corrcoef(Bo9[:,8], Bo9[:,3])[0,1]:.4f} "
        f"test={np.corrcoef(Bt9[:,8], Bt9[:,3])[0,1]:.4f}"
    )

    # LIVE resid proxy: L(LIVE) - projected onto Bt9 (or vs z_ath_t which IS L(LIVE) → use bold-less:
    # user asks L(LIVE)-z_ATH_t_proxy; if z_ath_t=L(LIVE) resid=0. Use residual of LIVE after Bt9.
    live_L = L(lv)
    live_hat = Ridge(alpha=RIDGE_ALPHA).fit(Bt9, live_L).predict(Bt9)
    live_resid = live_L - live_hat

    # GPU candidates
    oof_raw = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    llr = oof_raw[:, :, 1] - oof_raw[:, :, 0]
    spread_o = L(np.quantile(llr, 0.8, axis=1) - np.quantile(llr, 0.2, axis=1))
    traw = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy")
    tmean = traw.mean(0)
    tllr = tmean[:, :, 1] - tmean[:, :, 0]
    spread_t = L(np.quantile(tllr, 0.8, axis=1) - np.quantile(tllr, 0.2, axis=1))

    R2_t_L = np.stack([L(R2_te[f]) for f in range(5)], 0)
    curv_o = unitize(L(R2) - z_s2026_o)
    curv_t = unitize(L(R2_t_L.mean(0)) - z_s2026_t)

    candidates = {
        "q80_q20": (spread_o, spread_t),
        "R2_R3_curv": (curv_o, curv_t),
        "sadam_v4_cat": (L(XO["sadam_v4_cat"]), L(XT["sadam_v4_cat"])),
        "sameer_glm10f": (L(XO["sameer_glm10f"]), L(XT["sameer_glm10f"])),
        "alex_372ea048": (
            L(np.load(ALEX / "oof_15d0d127372ea048.npy").ravel()),
            L(np.load(ALEX / "test_15d0d127372ea048.npy").ravel()),
        ),
        "alex_6c55b429": (
            L(np.load(ALEX / "oof_21b4d1316c55b429.npy").ravel()),
            L(np.load(ALEX / "test_21b4d1316c55b429.npy").ravel()),
        ),
        "megayak_shallow_tree": (L(XO["megayak_shallow_tree"]), L(XT["megayak_shallow_tree"])),
        "sadam_v6_gb_xgb": (L(XO["sadam_v6_gb_xgb"]), L(XT["sadam_v6_gb_xgb"])),
        "sadam_v6_gb_lgb": (L(XO["sadam_v6_gb_lgb"]), L(XT["sadam_v6_gb_lgb"])),
        "tabpfn_slimypunk_tp1": (L(XO["tabpfn_slimypunk_tp1"]), L(XT["tabpfn_slimypunk_tp1"])),
        "sadam_tabpfn_100k_tp3": (L(XO["sadam_tabpfn_100k_tp3"]), L(XT["sadam_tabpfn_100k_tp3"])),
        "gbn_bag5_w512": (L(XO["gbn_bag5_w512"]), L(XT["gbn_bag5_w512"])),
        "alex_glm_gen": (L(XO["alex_glm_gen"]), L(XT["alex_glm_gen"])),
    }

    print("=== STEP 3: frozen orth + screens on z_S3 ===")
    results = {}
    oriented = {}
    for name, (vo, vt) in candidates.items():
        oo, ot, bbar, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, RIDGE_ALPHA)
        gates = forensic_gates(oo, ot, Bt9, live_resid)
        # exclude self if candidate is a basis member — skip for Q17 extras
        scr, u = screen(f"orth({name})", oo, z_s3, y, folds, te0)
        # orient test
        ut = ot if np.corrcoef(u, oo)[0, 1] > 0 else -ot
        oriented[name] = (u, unitize(ut))
        row = {**gates, **scr, "beta_bar": bbar.tolist()}
        results[name] = {k: v for k, v in row.items() if k != "beta_bar"}
        print(
            f"  GATES {name}: F1_disp={gates['disp']:.4f}({gates['F1']}) "
            f"F2_leak={gates['max_basis_leak']:.4f}({gates['F2']}) "
            f"F3_live={gates['corr_live_resid']:.4f}({gates['F3']}) "
            f"verified={gates['verified']}"
        )

    verified = [
        n
        for n, r in results.items()
        if r["verified"] and r["main"]["pos"] >= 5 and r["sf25"] >= 18
    ]
    verified_gates = [n for n, r in results.items() if r["verified"]]
    print(f"verified_gates={verified_gates}")
    print(f"verified_gates_and_5of5_sf18={verified}")

    staged = []
    best_combo = None
    if verified:
        from itertools import product

        keys = verified
        mats = {k: oriented[k][0] for k in keys}
        grid = [0.0, 0.3, 0.5, 0.7, 1.0]
        best_u = None
        for ws in product(grid, repeat=len(keys)):
            if sum(ws) < 1e-9:
                continue
            u = unitize(sum(w * mats[k] for w, k in zip(ws, keys)))
            if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u[te0]) < roc_auc_score(
                y[te0], z_s3[te0] - 0.01 * u[te0]
            ):
                u = -u
            d = [
                (roc_auc_score(y[te], z_s3[te] + 0.015 * u[te]) - roc_auc_score(y[te], z_s3[te]))
                * 1e5
                for _, te in folds
            ]
            pos = int(sum(x > 0 for x in d))
            mean = float(np.mean(d))
            deltas = []
            for _, te in folds:
                yf, zb, uf = y[te], z_s3[te], u[te]
                for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(
                    np.zeros(len(yf)), yf
                ):
                    b = roc_auc_score(yf[ste], zb[ste])
                    deltas.append((roc_auc_score(yf[ste], zb[ste] + 0.01 * uf[ste]) - b) * 1e5)
            d25 = np.asarray(deltas, float)
            t = float(d25.mean() / (d25.std(ddof=1) / np.sqrt(25) + 1e-18))
            sf = int((d25 > 0).sum())
            row = {"w": dict(zip(keys, ws)), "pos": pos, "mean": mean, "sf": sf, "t": t}
            if best_combo is None or (pos, sf, t, mean) > (
                best_combo["pos"],
                best_combo["sf"],
                best_combo["t"],
                best_combo["mean"],
            ):
                best_combo = row
                best_u = u
        if best_u is not None:
            s_best, u_best = screen("best_composite", best_u, z_s3, y, folds, te0)
            results["best_composite"] = s_best
            ut = unitize(
                sum(best_combo["w"][k] * oriented[k][1] for k in keys if best_combo["w"][k] > 0)
            )
            if np.corrcoef(u_best, unitize(sum(best_combo["w"][k] * oriented[k][0] for k in keys)))[
                0, 1
            ] < 0:
                ut = -ut
            wstar = s_best["main"]["w"]
            cand = qlock(L(slot3) + wstar * ut, slot3)
            path = SUB / "sub_v16_8_composite_frozen_NOSUBMIT.csv"
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
            staged.append(
                {
                    "file": path.name,
                    "weights": best_combo["w"],
                    "w": wstar,
                    "pos": s_best["main"]["pos"],
                    "sf25": s_best["sf25"],
                    "t": s_best["t"],
                    "1mrho_e4": float((1 - spearmanr(slot3, cand).statistic) * 1e4),
                }
            )
            print("STAGED", path.name)

        for n in verified:
            s = results[n]
            u_o, u_t = oriented[n]
            wstar = s["main"]["w"]
            cand = qlock(L(slot3) + wstar * u_t, slot3)
            path = SUB / f"sub_v16_8_{n}_NOSUBMIT.csv"
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
            staged.append(
                {
                    "file": path.name,
                    "signal": n,
                    "w": wstar,
                    "pos": s["main"]["pos"],
                    "sf25": s["sf25"],
                    "t": s["t"],
                }
            )
            print("STAGED", path.name)

    payload = {
        "NO_SUBMIT": True,
        "step1": step1,
        "basis9": bn9,
        "cond_Bo9": float(np.linalg.cond(Bo9)),
        "cond_Bt9": float(np.linalg.cond(Bt9)),
        "results": results,
        "verified_gates": verified_gates,
        "verified_full": verified,
        "best_combo": best_combo,
        "staged": staged,
        "auc": {
            "z_ath": float(roc_auc_score(y, z_ath)),
            "z_s1": float(roc_auc_score(y, z_s1)),
            "z_s3": float(roc_auc_score(y, z_s3)),
        },
    }
    OUT.joinpath("v16_8.json").write_text(json.dumps(payload, indent=2, default=float))
    print("WROTE", OUT / "v16_8.json")
    print("DONE")
    return payload


if __name__ == "__main__":
    main()
