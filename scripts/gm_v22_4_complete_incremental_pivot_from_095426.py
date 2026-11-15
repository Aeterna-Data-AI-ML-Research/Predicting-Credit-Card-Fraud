#!/usr/bin/env python3
"""GM V22.4 COMPLETE — Incremental H2H vs R2_095426 + signed LOFO + HOLD gate.

All 13 DS upgrades: incremental Δ vs R2, GS basis (no U_C self-cancel), symmetric
signed LOFO, forgotten champions, quadratic apex, twilight/monotonicity, hedge, HOLD.
Local Mac CPU only. Submit ONLY if strict 5/5×3 vs R2 gates pass.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO = Path(__file__).resolve().parents[1]
TMP16 = Path("/tmp/s6e11_v16")
OUT, SUB, LOG = TMP16 / "out", REPO / "submissions", REPO / "logs/overnight"
NB = Path("/tmp/s6e11_nb_research")
SRIRAM_DS = NB / "sriram_ds"
PYTAB_OOF = NB / "yekenot_pytabkit" / "oof_preds.csv"
PYTORCH_OOF = NB / "yekenot_pytorch" / "oof_preds.csv"
FAM20 = OUT / "v20_1_families.npz"
FAM191 = OUT / "v19_1_families.npz"
FAM1913 = OUT / "v19_13_families.npz"
GPT2MED = REPO / "preds" / "gpt2_med_4ep"
PARADIGM_B = REPO / "preds" / "paradigm_B"
MATRIX = REPO / "preds" / "matrix_v2"
COMPACT = REPO / "external/banks/goodpjw2008_s6e11-compact-lm-seed7"
R2_ATH = SUB / "sub_v22_3_LIVE_ridge_continuation.csv"  # 0.95426
P2_ATH = SUB / "sub_v22_2_HARDENED_LIVE_microleap.csv"  # 0.95425
C1 = SUB / "sub_v22_1_LIVE_leap.csv"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
REPORT = LOG / "GM_V22_4_COMPLETE_INCREMENTAL_REPORT.md"
N_TR, N_TE = 699960, 299984
W_GRID = (-0.012, -0.008, -0.005, -0.003, 0.003, 0.005, 0.008, 0.012)
T0 = time.time()
LINES: list[str] = []


def log(msg: str) -> None:
    s = f"[{time.time()-T0:7.1f}s] {msg}"
    print(s, flush=True)
    LINES.append(s)


def load_v20():
    path = REPO / "scripts" / "gm_v20_overnight_cpu_grandmaster_factory.py"
    spec = importlib.util.spec_from_file_location("gm_v20_fac", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def rank_logit(v):
    v = np.asarray(v, float).ravel()
    return logit((rankdata(v, method="average") - 0.5) / len(v))


def rank_frac(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def one_m_rho_e4(a, b):
    return (1.0 - float(spearmanr(a, b).correlation)) * 1e4


def project_out(v, B):
    v = np.asarray(v, float).ravel()
    X = np.column_stack([np.ones(len(v)), np.asarray(B, float)])
    xtx = X.T @ X
    xtx.flat[:: xtx.shape[0] + 1] += 1e-8
    return v - X @ np.linalg.solve(xtx, X.T @ v)


def max_abs_corr(v, M):
    v = np.asarray(v, float).ravel()
    v = (v - v.mean()) / (v.std() + 1e-12)
    M = np.asarray(M, float)
    Mc = M - M.mean(0, keepdims=True)
    Ms = Mc / (Mc.std(0, keepdims=True) + 1e-12)
    return float(np.max(np.abs(Ms.T @ v / len(v)))) if M.size else 0.0


def ecdf_lock(test, oof):
    """Map test onto OOF marginal (quantile matching) → KS≈0 vs oof ranks."""
    test = np.asarray(test, float).ravel()
    oof = np.asarray(oof, float).ravel()
    q = (rankdata(test, method="average") - 0.5) / len(test)
    return np.quantile(oof, q, method="linear")


def bisect_rho(base, direction, target_e4, lo=1e-6, hi=0.4, n=28):
    base = np.asarray(base, float).ravel()
    d = np.asarray(direction, float).ravel()
    d = (d - d.mean()) / (d.std() + 1e-12)
    best_w, best_err, best_c = 0.0, 1e9, base.copy()
    a, b = lo, hi
    for _ in range(n):
        mid = 0.5 * (a + b)
        cand = rank_frac(rank_logit(base) + mid * d)
        rho = one_m_rho_e4(cand, base)
        err = abs(rho - target_e4)
        if err < best_err:
            best_err, best_w, best_c = err, mid, cand
        if rho < target_e4:
            a = mid
        else:
            b = mid
    return best_w, best_c, one_m_rho_e4(best_c, base)


def quadratic_apex(points):
    """points: list of (u, s) with u=sqrt((1-rho)*1e4). Fit s = S0 + a*u - 0.5*b*u^2."""
    U = np.array([p[0] for p in points], float)
    S = np.array([p[1] for p in points], float)
    # design: [1, u, -0.5 u^2]
    X = np.column_stack([np.ones(len(U)), U, -0.5 * U * U])
    coef, _, _, _ = np.linalg.lstsq(X, S, rcond=None)
    S0, alpha, beta = float(coef[0]), float(coef[1]), float(coef[2])
    u_star = alpha / beta if abs(beta) > 1e-12 else float("nan")
    s_star = S0 + alpha * u_star - 0.5 * beta * u_star * u_star if np.isfinite(u_star) else float("nan")
    return {"S0": S0, "alpha": alpha, "beta": beta, "u_star": u_star, "s_star": s_star}


def twilight_share(base, cand, y, q_lo=0.80, q_hi=0.985):
    """Fraction of |Δscore| mass in twilight band by base quantile."""
    base, cand = np.asarray(base, float), np.asarray(cand, float)
    r = (rankdata(base, method="average") - 0.5) / len(base)
    m = (r >= q_lo) & (r <= q_hi)
    d = np.abs(cand - base)
    return float(d[m].sum() / (d.sum() + 1e-18))


def mono_guard(base, cand):
    """Proxy monotonicity: corr(cand, base) and local slope via ranks."""
    return float(np.corrcoef(base, cand)[0, 1])


def signed_lofo(uo, base, y, folds, w_grid=W_GRID):
    """Strict LOFO: pick w on train folds by meanΔ, eval on held-out fold."""
    fold_deltas = []
    chosen = []
    for i, (tr, te) in enumerate(folds):
        best_w, best_m = 0.0, -1e18
        for w in w_grid:
            # mean Δ on other folds only
            ds = []
            for j, (tr2, te2) in enumerate(folds):
                if j == i:
                    continue
                ds.append(
                    (roc_auc_score(y[te2], base[te2] + w * uo[te2]) - roc_auc_score(y[te2], base[te2])) * 1e5
                )
            m = float(np.mean(ds))
            if m > best_m:
                best_m, best_w = m, w
        chosen.append(best_w)
        d = (roc_auc_score(y[te], base[te] + best_w * uo[te]) - roc_auc_score(y[te], base[te])) * 1e5
        fold_deltas.append(d)
    d = np.asarray(fold_deltas, float)
    # also sf25 with median |w|
    w_med = float(np.median(chosen))
    # reuse h2h-style subfolds via calling pattern externally
    sign = 1 if w_med >= 0 else -1
    return {
        "w_med": w_med,
        "w_folds": chosen,
        "sign": sign,
        "pos": int((d > 0).sum()),
        "mean": float(d.mean()),
        "min": float(d.min()),
        "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-18)),
        "folds": [round(float(x), 3) for x in d],
    }


def submit_and_poll(path: Path, msg: str):
    pub = None
    ok = False
    try:
        r = subprocess.run(
            [KAG, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg],
            capture_output=True,
            text=True,
            timeout=90,
        )
        log(r.stdout.strip())
        log(r.stderr.strip())
        blob = (r.stdout + r.stderr).lower()
        ok = r.returncode == 0 and "400" not in (r.stdout + r.stderr) and "limit" not in blob
        if ok:
            for i in range(36):
                time.sleep(18 if i else 8)
                p = subprocess.run(
                    [KAG, "competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                lines = p.stdout.strip().splitlines()
                if not lines:
                    continue
                hdr = [h.strip() for h in lines[0].split(",")]
                for ln in lines[1:12]:
                    if "v22_4" in ln or "v22.4" in ln or "incremental_over_095426" in ln:
                        log(f"poll {ln[:260]}")
                        if "COMPLETE" in ln.upper():
                            cols = [c.strip() for c in ln.split(",")]
                            try:
                                pub = float(cols[hdr.index("publicScore")])
                            except Exception:
                                for tok in reversed(cols):
                                    if not tok:
                                        continue
                                    try:
                                        pub = float(tok)
                                        break
                                    except ValueError:
                                        continue
                        break
                if pub is not None:
                    break
    except Exception as e:
        log(f"submit error: {e}")
        traceback.print_exc()
    return ok, pub


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    v20 = load_v20()
    unitize, h2h, frozen_lofo_orth, L_fold = v20.unitize, v20.h2h, v20.frozen_lofo_orth, v20.L_fold

    log("=== V22.4 START: rebuild + incremental pivot from R2_095426 ===")
    fac = v20.OvernightFactory()
    fac.rebuild()
    f = fac
    ids, y, folds, fold_id = f.ids, f.y, f.folds, f.fold_id

    def load_sub(path):
        return pd.read_csv(path).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    raw_gp = load_sub(SRIRAM_DS / "reference.csv")
    raw_new = load_sub(SRIRAM_DS / "new_neural.csv")
    raw_old = load_sub(SRIRAM_DS / "old_neural.csv")
    sriram = rank_frac(rank_logit(raw_gp) + 0.15 * (rank_logit(raw_new) - rank_logit(raw_old)))
    r2_live = load_sub(R2_ATH)
    p2_live = load_sub(P2_ATH)
    c1 = load_sub(C1)
    log(
        f"R2 vs sriram={one_m_rho_e4(r2_live, sriram):.3f}e-4 | "
        f"R2 vs P2={one_m_rho_e4(r2_live, p2_live):.3f}e-4"
    )

    # ---------- Step1: quadratic apex ----------
    log("=== Step1: interval-bounded quadratic apex on Ray 1 ===")
    # u = sqrt((1-rho)*1e4)  — Euclidean step distance in e-4 Spearman units
    pts_nom = [
        (np.sqrt(0.00), 0.95422),
        (np.sqrt(0.55), 0.95425),
        (np.sqrt(1.25), 0.95426),
    ]
    apex = quadratic_apex(pts_nom)
    # rounding bounds ±0.49e-5 on last two points
    bound_apices = []
    for d1 in (-0.49e-5, 0.0, 0.49e-5):
        for d2 in (-0.49e-5, 0.0, 0.49e-5):
            pts = [
                (np.sqrt(0.00), 0.95422),
                (np.sqrt(0.55), 0.95425 + d1),
                (np.sqrt(1.25), 0.95426 + d2),
            ]
            bound_apices.append(quadratic_apex(pts))
    u_stars = [a["u_star"] for a in bound_apices if np.isfinite(a["u_star"]) and a["u_star"] > 0]
    u_star_cons = float(np.median(u_stars)) if u_stars else apex["u_star"]
    # convert u* back to (1-rho)e-4: u^2
    rho_star = float(u_star_cons**2) if np.isfinite(u_star_cons) else float("nan")
    log(
        f"apex nominal u*={apex['u_star']:.4f} → (1-ρ)*≈{apex['u_star']**2:.3f}e-4 "
        f"s*≈{apex['s_star']:.6f} | conservative u*={u_star_cons:.4f} → (1-ρ)*≈{rho_star:.3f}e-4 "
        f"α={apex['alpha']:.3e} β={apex['beta']:.3e}"
    )

    # ---------- load streams / rebuild UB UD UE ----------
    fam = np.load(FAM20)
    fam1 = np.load(FAM191)
    o_new = pd.read_csv(PYTAB_OOF).is_fraud.to_numpy(float)
    o_old = pd.read_csv(PYTORCH_OOF).is_fraud.to_numpy(float)
    shallow_o, shallow_t = fam["Q17_megayak_shallow_tree_oof"].ravel()[:N_TR], fam["Q17_megayak_shallow_tree_test"].ravel()[:N_TE]
    masaya_o, masaya_t = fam["Q17_masaya_gpt2_oof"].ravel()[:N_TR], fam["Q17_masaya_gpt2_test"].ravel()[:N_TE]
    gpd_o, gpd_t = fam["Q17_daoviet_gpdvae_oof"].ravel()[:N_TR], fam["Q17_daoviet_gpdvae_test"].ravel()[:N_TE]
    if (COMPACT / "lm7_oof_f0.npy").exists():
        compact_o = np.load(COMPACT / "lm7_oof_f0.npy").ravel()[:N_TR]
        compact_t = np.load(COMPACT / "lm7_test_f0.npy").ravel()[:N_TE]
    else:
        compact_o, compact_t = masaya_o, masaya_t

    def uL(x):
        return unitize(rank_logit(x))

    B_macro_t = np.column_stack(
        [uL(raw_gp), uL(sriram), uL(raw_new), uL(raw_old), uL(shallow_t), uL(compact_t), uL(masaya_t), uL(gpd_t)]
    )
    Bo_map = np.column_stack(
        [
            unitize(L_fold(o_new, fold_id)),
            unitize(L_fold(o_old, fold_id)),
            unitize(L_fold(shallow_o, fold_id)),
            unitize(L_fold(compact_o, fold_id)),
            unitize(L_fold(masaya_o, fold_id)),
            unitize(L_fold(gpd_o, fold_id)),
            f.Bo9,
        ]
    )
    Bt_map = np.column_stack(
        [uL(raw_new), uL(raw_old), uL(shallow_t), uL(compact_t), uL(masaya_t), uL(gpd_t), f.Bt9]
    )
    lr_gp = Ridge(alpha=10.0, fit_intercept=True).fit(Bt_map, rank_logit(raw_gp))
    z_gp_oof_L = Bo_map @ lr_gp.coef_ + lr_gp.intercept_
    z_sr_oof_L = z_gp_oof_L + 0.15 * (L_fold(o_new, fold_id) - L_fold(o_old, fold_id))
    z_sriram_oof = unitize(z_sr_oof_L) * float(np.std(f.z_s8)) + float(np.mean(f.z_s8))

    vault = np.load(OUT / "gm_overnight_vault.npz")
    u_e2, ut_e2 = unitize(vault["E2_heavy_ridge_oof"]), unitize(vault["E2_heavy_ridge_test"])
    if roc_auc_score(y[f.te0], f.z_s8[f.te0] + 0.01 * u_e2[f.te0]) < roc_auc_score(
        y[f.te0], f.z_s8[f.te0] - 0.01 * u_e2[f.te0]
    ):
        u_e2, ut_e2 = -u_e2, -ut_e2
    go = sum(np.load(GPT2MED / f"oof_{i}.npy").ravel()[:N_TR] for i in range(5)) / 5.0
    gt = sum(np.load(GPT2MED / f"test_{i}.npy").ravel()[:N_TE] for i in range(5)) / 5.0
    Braw_o = 0.55 * unitize(L_fold(go, fold_id) - L_fold(masaya_o, fold_id)) + 0.45 * u_e2
    Braw_t = 0.55 * unitize(rank_logit(gt) - rank_logit(masaya_t)) + 0.45 * ut_e2
    Draw_o = 0.55 * unitize(fam["Q17_gbn_bag5_w512_oof"].ravel()[:N_TR]) + 0.45 * unitize(
        fam["Q17_gb_bag5_oof"].ravel()[:N_TR] if "Q17_gb_bag5_oof" in fam.files else np.load(
            REPO / "external/banks/goodpjw2008_s6e11-oof-bank/oof_gb_bag5.npy"
        ).ravel()[:N_TR]
    )
    # fix D if Q17 gb missing
    if "Q17_gbn_bag5_w512_oof" in fam.files:
        gbn_o, gbn_t = fam["Q17_gbn_bag5_w512_oof"].ravel()[:N_TR], fam["Q17_gbn_bag5_w512_test"].ravel()[:N_TE]
    else:
        gbn_o = np.load(REPO / "external/banks/goodpjw2008_s6e11-oof-bank/oof_gbn_bag5_w512.npy").ravel()[:N_TR]
        gbn_t = np.load(REPO / "external/banks/goodpjw2008_s6e11-oof-bank/test_gbn_bag5_w512.npy").ravel()[:N_TE]
    gb_o = np.load(REPO / "external/banks/goodpjw2008_s6e11-oof-bank/oof_gb_bag5.npy").ravel()[:N_TR]
    gb_t = np.load(REPO / "external/banks/goodpjw2008_s6e11-oof-bank/test_gb_bag5.npy").ravel()[:N_TE]
    Draw_o = 0.55 * unitize(gbn_o) + 0.45 * unitize(gb_o)
    Draw_t = 0.55 * unitize(gbn_t) + 0.45 * unitize(gb_t)
    dmlp_o = L_fold(o_new, fold_id) - L_fold(o_old, fold_id)
    dmlp_t = rank_logit(raw_new) - rank_logit(raw_old)
    Eraw_o = project_out(dmlp_o, np.column_stack([unitize(L_fold(o_old, fold_id)), unitize(z_gp_oof_L)]))
    Eraw_t = project_out(dmlp_t, np.column_stack([uL(raw_old), uL(raw_gp)]))

    bad6_o = unitize(Bo_map @ Ridge(alpha=10.0).fit(Bt_map, f.bad6).coef_)
    bad7_o = unitize(Bo_map @ Ridge(alpha=10.0).fit(Bt_map, f.bad7).coef_)
    B_macro_o = np.column_stack(
        [
            unitize(z_gp_oof_L),
            unitize(z_sr_oof_L),
            unitize(L_fold(o_new, fold_id)),
            unitize(L_fold(o_old, fold_id)),
            unitize(L_fold(shallow_o, fold_id)),
            unitize(L_fold(compact_o, fold_id)),
            unitize(L_fold(masaya_o, fold_id)),
            unitize(L_fold(gpd_o, fold_id)),
        ]
    )

    def orth_macro(vo, vt):
        oo, ot, _ = frozen_lofo_orth(vo, vt, B_macro_o, B_macro_t, folds, ridge_alpha=10.0)
        ot = project_out(ot, B_macro_t)
        return unitize(oo), unitize(ot)

    uB_o, uB_t = orth_macro(Braw_o, Braw_t)
    uD_o, uD_t = orth_macro(Draw_o, Draw_t)
    uE_o, uE_t = orth_macro(Eraw_o, Eraw_t)
    # orient vs sriram oof
    for name, pair in ("UB", (uB_o, uB_t)), ("UD", (uD_o, uD_t)), ("UE", (uE_o, uE_t)):
        oo, ot = pair
        if roc_auc_score(y[f.te0], z_sriram_oof[f.te0] + 0.01 * oo[f.te0]) < roc_auc_score(
            y[f.te0], z_sriram_oof[f.te0] - 0.01 * oo[f.te0]
        ):
            if name == "UB":
                uB_o, uB_t = -oo, -ot
            elif name == "UD":
                uD_o, uD_t = -oo, -ot
            else:
                uE_o, uE_t = -oo, -ot

    # GS basis: macro + UB + UD + UE + rays  (NO U_C, NO R2-sriram delta)
    B_gs_o = np.column_stack([B_macro_o, uB_o, uD_o, uE_o, bad6_o, bad7_o]).astype(np.float64)
    B_gs_t = np.column_stack([B_macro_t, uB_t, uD_t, uE_t, f.bad6, f.bad7]).astype(np.float64)

    # Reconstruct R2 OOF proxy via ridge from test
    z_s9 = f.z_s8 + 0.020 * u_e2
    z_s9_d = z_s9 + 0.10 * unitize(dmlp_o)
    Bt_r2 = np.column_stack([Bt_map, uB_t, uD_t, uE_t, uL(c1), uL(p2_live)])
    Bo_r2 = np.column_stack([Bo_map, uB_o, uD_o, uE_o, unitize(z_s9), unitize(z_sr_oof_L)])
    lr_r2 = Ridge(alpha=10.0, fit_intercept=True).fit(Bt_r2, rank_logit(r2_live))
    z_r2_oof = unitize(Bo_r2 @ lr_r2.coef_ + lr_r2.intercept_) * float(np.std(z_s9)) + float(np.mean(z_s9))
    r2_auc = float(roc_auc_score(y, z_r2_oof))
    r2_r2 = 1.0 - float(np.sum((rank_logit(r2_live) - lr_r2.predict(Bt_r2)) ** 2)) / (
        float(np.var(rank_logit(r2_live)) * N_TE) + 1e-18
    )
    log(f"z_R2_oof AUC={r2_auc:.6f} surrogate R²={r2_r2:.4f}")
    # three R2 anchors: transfer R2 lift onto each baseline
    # z_R2_sri ≈ z_sriram_oof + (z_r2_oof - z_sriram_oof) projected
    delta_r2 = z_r2_oof - z_sriram_oof
    bases_r2 = {
        "z_R2_sri": z_sriram_oof + delta_r2,
        "z_R2_z9": z_s9 + delta_r2,
        "z_R2_z9d": z_s9_d + delta_r2,
    }
    # projection weights of pillars inside R2 - sriram
    ray = unitize(rank_logit(r2_live) - rank_logit(sriram))
    for nm, ut in ("UB", uB_t), ("UC_raw", unitize(0.55 * f.ut_cat + 0.45 * f.ut_r2)), ("UD", uD_t), ("UE", uE_t), ("C1", unitize(rank_logit(c1) - rank_logit(sriram))):
        w = float(np.corrcoef(ray, ut)[0, 1]) * float(np.std(ray)) / (float(np.std(ut)) + 1e-12)
        # OLS weight on single column
        w_ols = float(np.linalg.lstsq(ut.reshape(-1, 1), ray, rcond=None)[0][0])
        log(f"  proj weight in R2-sriram: {nm} corr={w:+.4f} ols={w_ols:+.4f}")

    # U_C projection into R2-sriram (diagnostic — NOT in GS basis)
    Craw_o = 0.55 * f.u_cat + 0.45 * f.u_r2
    if "F_lgb_ms39_oof" in fam1.files:
        Craw_o = 0.45 * f.u_cat + 0.35 * f.u_r2 + 0.20 * unitize(fam1["F_lgb_ms39_oof"].ravel()[:N_TR])
        Craw_t = 0.45 * f.ut_cat + 0.35 * f.ut_r2 + 0.20 * unitize(fam1["F_lgb_ms39_test"].ravel()[:N_TE])
    else:
        Craw_t = 0.55 * f.ut_cat + 0.45 * f.ut_r2
    # orth UC against GS (without putting R2-sriram in basis)
    uC_o, uC_t_tmp, _ = frozen_lofo_orth(Craw_o, Craw_t, B_gs_o, B_gs_t, folds, ridge_alpha=10.0)
    uC_t = unitize(project_out(uC_t_tmp, B_gs_t))
    uC_o = unitize(uC_o)
    uc_in_ray = float(np.linalg.lstsq(uC_t.reshape(-1, 1), ray, rcond=None)[0][0])
    log(f"U_C_tab_gs existing projection weight inside R2-sriram: ols={uc_in_ray:+.4f}")

    # ---------- Step2: load 12 vectors ----------
    log("=== Step2: screen V1..V12 signed LOFO vs R2_095426 ===")

    def load_pair(oof_k, test_k, npz=fam, alt=None):
        if oof_k in npz.files and test_k in npz.files:
            return npz[oof_k].ravel()[:N_TR], npz[test_k].ravel()[:N_TE]
        if alt and alt[0].exists():
            return np.load(alt[0]).ravel()[:N_TR], np.load(alt[1]).ravel()[:N_TE]
        return None

    def gs_orth(vo, vt):
        oo, ot, beta = frozen_lofo_orth(vo, vt, B_gs_o, B_gs_t, folds, ridge_alpha=10.0)
        ot2 = project_out(ot, B_gs_t)
        raw_disp = float(np.std(ot2) / (np.std(oo) + 1e-12))
        raw_ks = float(ks_2samp(unitize(oo), unitize(ot2)).statistic)
        return unitize(oo), unitize(ot2), {"raw_disp": raw_disp, "raw_KS": raw_ks}

    vectors = {}

    # V1 UC
    vectors["V1_UC_tab_gs"] = (Craw_o, Craw_t)

    # V2 sameer
    p = load_pair("Q17_sameer_glm10f_oof", "Q17_sameer_glm10f_test")
    if p:
        vectors["V2_sameer_glm10f_gs"] = p

    # V3 sadam cat
    p = load_pair("Q17_sadam_v4_cat_oof", "Q17_sadam_v4_cat_test")
    if p:
        vectors["V3_sadam_cat_gs"] = p

    # V4 P4 L4.2
    p = load_pair(
        "P4_L4.2_pure5f_oof",
        "P4_L4.2_pure5f_test",
        alt=(MATRIX / "P4_L4.2_pure5f_oof.npy", MATRIX / "P4_L4.2_pure5f_test.npy"),
    )
    if p:
        vectors["V4_P4_L42_gs"] = p

    # V5 P4 L4.3 ECDF-locked
    oof_p, te_p = None, None
    if "P4_L4.3_pure5f_oof" in fam.files:
        oof_p, te_p = fam["P4_L4.3_pure5f_oof"].ravel()[:N_TR], fam["P4_L4.3_pure5f_test"].ravel()[:N_TE]
    elif (MATRIX / "P4_L4.3_pure5f_oof.npy").exists():
        oof_p = np.load(MATRIX / "P4_L4.3_pure5f_oof.npy").ravel()[:N_TR]
        te_p = np.load(MATRIX / "P4_L4.3_pure5f_test.npy").ravel()[:N_TE]
    if oof_p is not None:
        te_p_lock = ecdf_lock(te_p, oof_p)
        ks0 = float(ks_2samp(unitize(te_p), unitize(oof_p)).statistic)
        ks1 = float(ks_2samp(unitize(te_p_lock), unitize(oof_p)).statistic)
        log(f"  P4_L4.3 pure KS before={ks0:.4f} after ECDF-lock={ks1:.4f}")
        if (MATRIX / "P4_L4.3_ext_oof.npy").exists():
            oof_e = np.load(MATRIX / "P4_L4.3_ext_oof.npy").ravel()[:N_TR]
            te_e = ecdf_lock(np.load(MATRIX / "P4_L4.3_ext_test.npy").ravel()[:N_TE], oof_e)
            vo = 0.55 * unitize(oof_p) + 0.45 * unitize(oof_e)
            vt = 0.55 * unitize(te_p_lock) + 0.45 * unitize(te_e)
        else:
            vo, vt = oof_p, te_p_lock
        vectors["V5_P4_L43_rect_gs"] = (vo, vt)

    # V6 P4 L4.5
    p = load_pair(
        "P4_L4.5_pure5f_oof",
        "P4_L4.5_pure5f_test",
        alt=(MATRIX / "P4_L4.5_pure5f_oof.npy", MATRIX / "P4_L4.5_pure5f_test.npy"),
    )
    if p:
        vectors["V6_P4_L45_gs"] = p

    # V7 B xgb llr
    if (PARADIGM_B / "xgb_llr_oof.npy").exists():
        vectors["V7_B_xgb_llr_gs"] = (
            np.load(PARADIGM_B / "xgb_llr_oof.npy").ravel()[:N_TR],
            np.load(PARADIGM_B / "xgb_llr_test.npy").ravel()[:N_TE],
        )

    # V8 v10 ridge
    p = load_pair("v10_ridge_resid_orth_LM_oof", "v10_ridge_resid_orth_LM_test")
    if p:
        vectors["V8_v10_ridge_gs"] = p

    # V9 curv + U567 stand-in for antithetic
    p = load_pair("R2_minus_R3_curv_oof", "R2_minus_R3_curv_test")
    if p:
        if "cache_U567_comp_oof" in fam.files:
            vo = 0.60 * unitize(p[0]) + 0.40 * unitize(fam["cache_U567_comp_oof"].ravel()[:N_TR])
            vt = 0.60 * unitize(p[1]) + 0.40 * unitize(fam["cache_U567_comp_test"].ravel()[:N_TE])
            vectors["V9_L7_curv_and_antithetic_gs"] = (vo, vt)
        else:
            vectors["V9_L7_curv_and_antithetic_gs"] = p

    # V10 U5 U8
    u8_o, u8_t = unitize(vault["E2_U8_oof"]), unitize(vault["E2_U8_test"])
    vectors["V10_L7_U5_U8_gs"] = (0.55 * f.u_u5 + 0.45 * u8_o, 0.55 * f.ut_u5 + 0.45 * u8_t)

    # V11 P1 raw
    if "cache_P1_raw_only_oof" in fam.files:
        vectors["V11_P1_raw_peer_gs"] = (
            fam["cache_P1_raw_only_oof"].ravel()[:N_TR],
            fam["cache_P1_raw_only_test"].ravel()[:N_TE],
        )
    elif FAM1913.exists():
        a13 = np.load(FAM1913)
        if "P1_raw_only_oof" in a13.files:
            vectors["V11_P1_raw_peer_gs"] = (a13["P1_raw_only_oof"].ravel()[:N_TR], a13["P1_raw_only_test"].ravel()[:N_TE])

    # V12 C1 orth
    vectors["V12_C1_orth_gs"] = (unitize(z_s9 + 0.10 * unitize(dmlp_o) - z_sriram_oof), unitize(rank_logit(c1) - rank_logit(sriram)))

    log(f"loaded {len(vectors)}/{12} vectors: {list(vectors)}")

    results = []
    passing = []
    for name, (vo, vt) in vectors.items():
        try:
            oo, ot, pre = gs_orth(vo, vt)
        except Exception as e:
            log(f"  {name}: ORTH FAIL {e}")
            continue
        mc = max_abs_corr(ot, B_macro_t)
        c6 = float(np.corrcoef(ot, f.bad6)[0, 1])
        c7 = float(np.corrcoef(ot, f.bad7)[0, 1])
        # signed LOFO on primary R2 sri anchor; then evaluate locked w on all 3
        lofo = signed_lofo(oo, bases_r2["z_R2_sri"], y, folds)
        w = lofo["w_med"]
        # if median ~0, use sign of mean fold w
        if abs(w) < 1e-6:
            w = 0.005 * lofo["sign"]
        per3 = {b: h2h(bases_r2[b] + w * oo, bases_r2[b], y, folds) for b in bases_r2}
        # twilight on test
        te_base = r2_live
        te_cand = rank_frac(rank_logit(r2_live) + abs(w) * np.sign(w) * ot)
        tw = twilight_share(te_base, te_cand, None)
        mono = mono_guard(te_base, te_cand)
        row = {
            "name": name,
            "w": w,
            "pre": pre,
            "mc": mc,
            "c6": c6,
            "c7": c7,
            "lofo": lofo,
            "per3": per3,
            "twilight": tw,
            "mono": mono,
            "uo": oo,
            "ut": ot,
        }
        results.append(row)
        ok3 = all(
            per3[b]["pos"] == 5 and per3[b]["min"] > 0 and per3[b]["sf25"] >= 18 and per3[b]["t"] >= 2.80
            for b in per3
        )
        disp_ok = 0.92 <= pre["raw_disp"] <= 1.08
        log(
            f"  {name}: w={w:+.4f} disp={pre['raw_disp']:.3f} KS={pre['raw_KS']:.4f} "
            f"max|B|={mc:.4f} c6={c6:+.3f} c7={c7:+.3f} tw={tw:.3f} mono={mono:.4f} | "
            f"LOFO {lofo['pos']}/5 m={lofo['mean']:+.2f} min={lofo['min']:+.2f} t={lofo['t']:.2f} | "
            f"sri {per3['z_R2_sri']['pos']}/5 t={per3['z_R2_sri']['t']:.2f} min={per3['z_R2_sri']['min']:+.2f} | "
            f"z9 {per3['z_R2_z9']['pos']}/5 t={per3['z_R2_z9']['t']:.2f} | "
            f"z9d {per3['z_R2_z9d']['pos']}/5 t={per3['z_R2_z9d']['t']:.2f} | PASS3={ok3} dispOK={disp_ok}"
        )
        if ok3 and disp_ok and mc < 0.05 and abs(c6) < 0.05 and abs(c7) < 0.05:
            passing.append(row)
            log(f"  >>> PASS {name}")

    log(f"passing vectors: {[p['name'] for p in passing]}")

    # ---------- Step3: synthesize Q1/Q2 + hedge ----------
    log("=== Step3: synthesize Q1/Q2 + anti-shakeup hedge ===")
    submit_ok = False
    pub = None
    reason = "HOLD_0.95426"
    q1 = q2 = None

    if passing:
        u_syn_o = np.zeros(N_TR)
        u_syn_t = np.zeros(N_TE)
        for p in passing:
            # lock sign into weight
            u_syn_o += np.sign(p["w"]) * p["uo"]
            u_syn_t += np.sign(p["w"]) * p["ut"]
        u_syn_o = unitize(u_syn_o)
        u_syn_t = unitize(project_out(u_syn_t, B_gs_t))
        # orient
        if roc_auc_score(y[f.te0], bases_r2["z_R2_sri"][f.te0] + 0.01 * u_syn_o[f.te0]) < roc_auc_score(
            y[f.te0], bases_r2["z_R2_sri"][f.te0] - 0.01 * u_syn_o[f.te0]
        ):
            u_syn_o, u_syn_t = -u_syn_o, -u_syn_t

        w1, te_q1, rho_q1 = bisect_rho(r2_live, u_syn_t, 0.35)
        per_q1 = {b: h2h(bases_r2[b] + w1 * u_syn_o, bases_r2[b], y, folds) for b in bases_r2}
        q1 = {"tag": "Q1_Incremental_Orth_on_R2", "test": te_q1, "w": w1, "rho": rho_q1, "per": per_q1}
        log(
            f"Q1: w={w1:.5f} ρ(R2)={rho_q1:.3f}e-4 sri {per_q1['z_R2_sri']['pos']}/5 "
            f"t={per_q1['z_R2_sri']['t']:.2f} min={per_q1['z_R2_sri']['min']:+.2f}"
        )

        # Q2: move Ray1 to apex then add orth
        target_from_sri = float(np.clip(rho_star, 1.25, 2.5)) if np.isfinite(rho_star) else 1.60
        # direction from sriram along current R2 ray + synthesis
        dir_apex = (rank_logit(r2_live) - rank_logit(sriram)) + 0.5 * u_syn_t
        # build candidate near apex distance from sriram, then measure vs R2
        # simpler: start at R2, step synthesis toward rho in [0.25,0.50]
        tgt = float(np.clip(abs(target_from_sri - 1.25), 0.25, 0.50))
        w2, te_q2, rho_q2 = bisect_rho(r2_live, u_syn_t, tgt)
        per_q2 = {b: h2h(bases_r2[b] + w2 * u_syn_o, bases_r2[b], y, folds) for b in bases_r2}
        q2 = {"tag": "Q2_Apex_Adjusted_Ray1_plus_Orth", "test": te_q2, "w": w2, "rho": rho_q2, "per": per_q2}
        log(
            f"Q2: w={w2:.5f} ρ(R2)={rho_q2:.3f}e-4 (tgt≈{tgt:.2f}) sri {per_q2['z_R2_sri']['pos']}/5 "
            f"t={per_q2['z_R2_sri']['t']:.2f} min={per_q2['z_R2_sri']['min']:+.2f}"
        )

        def clears(q):
            if q is None:
                return False
            per = q["per"]
            delta = unitize(rank_logit(q["test"]) - rank_logit(r2_live))
            mc = max_abs_corr(delta, B_macro_t)
            c6 = float(np.corrcoef(delta, f.bad6)[0, 1])
            c7 = float(np.corrcoef(delta, f.bad7)[0, 1])
            q["mc"], q["c6"], q["c7"] = mc, c6, c7
            ok = (
                all(per[b]["pos"] == 5 and per[b]["min"] > 0.05 and per[b]["sf25"] >= 18 and per[b]["t"] >= 2.80 for b in per)
                and mc < 0.005
                and abs(c6) < 0.02
                and abs(c7) < 0.02
                and 0.25 <= q["rho"] <= 0.55
                and mono_guard(r2_live, q["test"]) > 0.90
            )
            log(
                f"  gate {q['tag']}: ok={ok} mc={mc:.4f} c6={c6:+.3f} c7={c7:+.3f} "
                f"mono={mono_guard(r2_live, q['test']):.4f}"
            )
            return ok

        candidates = []
        if clears(q1):
            candidates.append(q1)
        if clears(q2):
            candidates.append(q2)
        if candidates:
            submit_row = max(
                candidates,
                key=lambda q: (q["per"]["z_R2_sri"]["min"], q["per"]["z_R2_sri"]["t"]),
            )
            live = SUB / "sub_v22_4_LIVE_incremental_over_095426.csv"
            pd.DataFrame({"id": ids, "is_fraud": submit_row["test"]}).to_csv(live, index=False)
            msg = (
                f"v22.4 {submit_row['tag']} | ρ(R2)={submit_row['rho']:.3f}e-4 "
                f"OOF vsR2 {submit_row['per']['z_R2_sri']['pos']}/5 "
                f"t={submit_row['per']['z_R2_sri']['t']:.2f} min={submit_row['per']['z_R2_sri']['min']:+.2f}"
            )
            log(f"SUBMIT {msg}")
            submit_ok, pub = submit_and_poll(live, msg)
            reason = submit_row["tag"]
        else:
            log("NO candidate cleared strict 5/5×3 vs R2 — HOLD 0.95426")
            reason = "HOLD_0.95426_no_gate_clear"
    else:
        log("NO individual vector passed 5/5×3 vs R2 — HOLD 0.95426")
        reason = "HOLD_0.95426_no_passing_vectors"

    # Always stage hedge
    # 0.70 * L(R2) + 0.30 * L(C1 + w_clean*(UB+UC))
    uc_t_h = unitize(project_out(Craw_t, B_macro_t))
    hedge_dir = unitize(uB_t + 0.5 * uc_t_h)
    _, c1_boost, _ = bisect_rho(c1, hedge_dir, 0.40)
    hedge = rank_frac(0.70 * rank_logit(r2_live) + 0.30 * rank_logit(c1_boost))
    hedge_path = SUB / "sub_v22_4_PRIVATE_LB_HEDGE_NOSUBMIT.csv"
    pd.DataFrame({"id": ids, "is_fraud": hedge}).to_csv(hedge_path, index=False)
    log(f"STAGED hedge {hedge_path.name} ρ(R2)={one_m_rho_e4(hedge, r2_live):.3f}e-4 (NO SUBMIT)")

    # stage Q diagnostics if exist
    if q1 is not None:
        pd.DataFrame({"id": ids, "is_fraud": q1["test"]}).to_csv(SUB / "sub_v22_4_Q1_incremental_orth_on_R2.csv", index=False)
    if q2 is not None:
        pd.DataFrame({"id": ids, "is_fraud": q2["test"]}).to_csv(SUB / "sub_v22_4_Q2_apex_ray1_plus_orth.csv", index=False)

    meta = {
        "reason": reason,
        "apex": apex,
        "rho_star_e4": rho_star,
        "n_passing": len(passing),
        "passing": [p["name"] for p in passing],
        "submit_ok": submit_ok,
        "pub": pub,
        "vector_summary": [
            {
                "name": r["name"],
                "w": r["w"],
                "mc": r["mc"],
                "c6": r["c6"],
                "c7": r["c7"],
                "disp": r["pre"]["raw_disp"],
                "sri_pos": r["per3"]["z_R2_sri"]["pos"],
                "sri_t": r["per3"]["z_R2_sri"]["t"],
                "sri_min": r["per3"]["z_R2_sri"]["min"],
            }
            for r in results
        ],
    }
    OUT.joinpath("v22_4_meta.json").write_text(json.dumps(meta, indent=2, default=float))

    body = [
        "# GM V22.4 Complete Incremental Pivot Report",
        "",
        f"**elapsed** {time.time()-T0:.0f}s · local Mac CPU · ATH `0.95426` (R2)",
        "",
        f"## Decision: **{reason}**",
        f"- Submit OK: `{submit_ok}` · Public LB: **{pub}**",
        f"- Passing vectors: `{[p['name'] for p in passing]}`",
        "",
        "## Ray-1 quadratic apex",
        f"- nominal u*=`{apex['u_star']:.4f}` → (1−ρ)*≈`{apex['u_star']**2:.3f}e-4` s*≈`{apex['s_star']:.6f}`",
        f"- conservative (1−ρ)*≈`{rho_star:.3f}e-4`",
        f"- α=`{apex['alpha']:.3e}` β=`{apex['beta']:.3e}`",
        "",
        "## U_C projection into R2−sriram (not in GS basis)",
        f"- ols weight=`{uc_in_ray:+.4f}`",
        "",
        "## Vector scorecard (incremental vs R2)",
        "| vec | w | disp | max\\|B\\| | sri pos/t/min | PASS |",
        "|---|---:|---:|---:|---|:---:|",
    ]
    for r in results:
        p = r["per3"]["z_R2_sri"]
        ok = (
            all(
                r["per3"][b]["pos"] == 5 and r["per3"][b]["min"] > 0 and r["per3"][b]["t"] >= 2.80
                for b in r["per3"]
            )
            and 0.92 <= r["pre"]["raw_disp"] <= 1.08
        )
        body.append(
            f"| `{r['name']}` | {r['w']:+.4f} | {r['pre']['raw_disp']:.3f} | {r['mc']:.4f} | "
            f"{p['pos']}/5 t={p['t']:.2f} min={p['min']:+.2f} | {'Y' if ok else 'n'} |"
        )
    body += [
        "",
        "## Staged",
        f"- `{hedge_path.name}` (NOSUBMIT private hedge)",
        "",
        "## Log",
        "```",
        *LINES,
        "```",
        "",
    ]
    REPORT.write_text("\n".join(body) + "\n")
    log(f"REPORT {REPORT}")
    log(f"DONE reason={reason} pub={pub} submit_ok={submit_ok}")


if __name__ == "__main__":
    main()
