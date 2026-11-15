#!/usr/bin/env python3
"""GM V22.5 — Crest + signed orth4 over R2_095426 (fix V22.4 Q1/Q2 sign/z9 paradox).

Local Mac CPU. Three sriram-manifold baselines; sign+disp lock; K1/K2/K3.
Submit ONLY if 5/5×3 on sri-manifold baselines + soft z9 meanΔ≥0.
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
from scipy.optimize import nnls
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import ElasticNet, Ridge
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
GPT2MED = REPO / "preds" / "gpt2_med_4ep"
PARADIGM_B = REPO / "preds" / "paradigm_B"
COMPACT = REPO / "external/banks/goodpjw2008_s6e11-compact-lm-seed7"
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
R2_ATH = SUB / "sub_v22_3_LIVE_ridge_continuation.csv"
P2_ATH = SUB / "sub_v22_2_HARDENED_LIVE_microleap.csv"
C1 = SUB / "sub_v22_1_LIVE_leap.csv"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
REPORT = LOG / "GM_V22_5_CREST_AND_ORTH4_REPORT.md"
N_TR, N_TE = 699960, 299984
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


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


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


def var_match(ot, oo):
    oo = np.asarray(oo, float).ravel()
    ot = np.asarray(ot, float).ravel()
    so, st = float(np.std(oo)) + 1e-12, float(np.std(ot)) + 1e-12
    return ot * (so / st)


def bisect_rho(base, direction, target_e4, lo=1e-6, hi=0.5, n=28):
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


def inject_orth_rho(base, direction, target_e4, B_macro, n_iter=4):
    """Bisect to target 1-ρ; direction kept B_macro-orth; iterate to shrink rank_frac leak."""
    base = np.asarray(base, float).ravel()
    d = unitize(project_out(np.asarray(direction, float).ravel(), B_macro))
    best_w, best_c, best_rho, best_d = 0.0, base.copy(), 0.0, d
    best_mc_raw = 1e9
    for _ in range(n_iter):
        w, cand, rho = bisect_rho(base, d, target_e4)
        delta_raw = unitize(rank_logit(cand) - rank_logit(base))
        mc_raw = max_abs_corr(delta_raw, B_macro)
        d_next = unitize(project_out(delta_raw, B_macro))
        if mc_raw < best_mc_raw:
            best_mc_raw, best_w, best_c, best_rho, best_d = mc_raw, w, cand, rho, d
        d = d_next
        if mc_raw < 0.001:
            break
    return best_w, best_c, best_rho, best_d


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
                    if "v22_5" in ln or "v22.5" in ln or "crest" in ln.lower() or "orth4" in ln.lower():
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

    log("=== V22.5 START: crest + signed orth4 from R2_095426 ===")
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
        f"R2 vs P2={one_m_rho_e4(r2_live, p2_live):.3f}e-4 | "
        f"P2 vs sriram={one_m_rho_e4(p2_live, sriram):.3f}e-4"
    )

    fam = np.load(FAM20)
    fam1 = np.load(FAM191)
    o_new = pd.read_csv(PYTAB_OOF).is_fraud.to_numpy(float)
    o_old = pd.read_csv(PYTORCH_OOF).is_fraud.to_numpy(float)
    shallow_o = fam["Q17_megayak_shallow_tree_oof"].ravel()[:N_TR]
    shallow_t = fam["Q17_megayak_shallow_tree_test"].ravel()[:N_TE]
    masaya_o = fam["Q17_masaya_gpt2_oof"].ravel()[:N_TR]
    masaya_t = fam["Q17_masaya_gpt2_test"].ravel()[:N_TE]
    gpd_o = fam["Q17_daoviet_gpdvae_oof"].ravel()[:N_TR]
    gpd_t = fam["Q17_daoviet_gpdvae_test"].ravel()[:N_TE]
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
    gbn_o = np.load(BANK / "oof_gbn_bag5_w512.npy").ravel()[:N_TR]
    gbn_t = np.load(BANK / "test_gbn_bag5_w512.npy").ravel()[:N_TE]
    gb_o = np.load(BANK / "oof_gb_bag5.npy").ravel()[:N_TR]
    gb_t = np.load(BANK / "test_gb_bag5.npy").ravel()[:N_TE]
    Draw_o = 0.55 * unitize(gbn_o) + 0.45 * unitize(gb_o)
    Draw_t = 0.55 * unitize(gbn_t) + 0.45 * unitize(gb_t)
    lr_gp = Ridge(alpha=10.0, fit_intercept=True).fit(Bt_map, rank_logit(raw_gp))
    z_gp_oof_L = Bo_map @ lr_gp.coef_ + lr_gp.intercept_
    dmlp_o = L_fold(o_new, fold_id) - L_fold(o_old, fold_id)
    dmlp_t = rank_logit(raw_new) - rank_logit(raw_old)
    Eraw_o = project_out(dmlp_o, np.column_stack([unitize(L_fold(o_old, fold_id)), unitize(z_gp_oof_L)]))
    Eraw_t = project_out(dmlp_t, np.column_stack([uL(raw_old), uL(raw_gp)]))

    bad6_o = unitize(Bo_map @ Ridge(alpha=10.0).fit(Bt_map, f.bad6).coef_)
    bad7_o = unitize(Bo_map @ Ridge(alpha=10.0).fit(Bt_map, f.bad7).coef_)
    B_macro_o = np.column_stack(
        [
            unitize(z_gp_oof_L),
            unitize(z_gp_oof_L + 0.15 * dmlp_o),
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
        ot = var_match(project_out(ot, B_macro_t), oo)
        return unitize(oo), unitize(ot)

    uB_o, uB_t = orth_macro(Braw_o, Braw_t)
    uD_o, uD_t = orth_macro(Draw_o, Draw_t)
    uE_o, uE_t = orth_macro(Eraw_o, Eraw_t)

    z_sr_oof_L = z_gp_oof_L + 0.15 * dmlp_o
    z_sriram_oof = unitize(z_sr_oof_L) * float(np.std(f.z_s8)) + float(np.mean(f.z_s8))
    for pair_name in ("UB", "UD", "UE"):
        oo = {"UB": uB_o, "UD": uD_o, "UE": uE_o}[pair_name]
        ot = {"UB": uB_t, "UD": uD_t, "UE": uE_t}[pair_name]
        if roc_auc_score(y[f.te0], z_sriram_oof[f.te0] + 0.01 * oo[f.te0]) < roc_auc_score(
            y[f.te0], z_sriram_oof[f.te0] - 0.01 * oo[f.te0]
        ):
            if pair_name == "UB":
                uB_o, uB_t = -oo, -ot
            elif pair_name == "UD":
                uD_o, uD_t = -oo, -ot
            else:
                uE_o, uE_t = -oo, -ot

    B_gs_o = np.column_stack([B_macro_o, uB_o, uD_o, uE_o, bad6_o, bad7_o]).astype(np.float64)
    B_gs_t = np.column_stack([B_macro_t, uB_t, uD_t, uE_t, f.bad6, f.bad7]).astype(np.float64)

    # ---------- Step1: 3 sriram-manifold R2 baselines ----------
    log("=== Step1: build z_R2_sri_ridge / nnls / hybrid_twin ===")
    # Base bank for sriram surrogate
    Bt_s = np.column_stack([Bt_map, uB_t, uD_t, uE_t])
    Bo_s = np.column_stack([Bo_map, uB_o, uD_o, uE_o])
    # Ridge alpha=50 sriram
    lr_sri = Ridge(alpha=50.0, fit_intercept=True).fit(Bt_s, rank_logit(sriram))
    z_sri_ridge = unitize(Bo_s @ lr_sri.coef_ + lr_sri.intercept_) * float(np.std(f.z_s8)) + float(np.mean(f.z_s8))
    # ElasticNet / NNLS on top-15 abs coef from ridge
    coef_abs = np.abs(lr_sri.coef_)
    top_idx = np.argsort(coef_abs)[-15:]
    Bt15, Bo15 = Bt_s[:, top_idx], Bo_s[:, top_idx]
    # NNLS without intercept via centered
    yt = rank_logit(sriram) - rank_logit(sriram).mean()
    Bt15c = Bt15 - Bt15.mean(0, keepdims=True)
    w_nnls, _ = nnls(Bt15c, yt)
    z_sri_nnls = unitize(Bo15 @ w_nnls) * float(np.std(f.z_s8)) + float(np.mean(f.z_s8))
    # also ElasticNet positive
    en = ElasticNet(alpha=5.0, l1_ratio=0.5, positive=True, fit_intercept=True, max_iter=5000)
    try:
        en.fit(Bt15, rank_logit(sriram))
        z_en = unitize(Bo15 @ en.coef_ + en.intercept_) * float(np.std(f.z_s8)) + float(np.mean(f.z_s8))
        if roc_auc_score(y, z_en) > roc_auc_score(y, z_sri_nnls):
            z_sri_nnls = z_en
            log("nnls baseline: ElasticNet positive preferred")
    except Exception:
        pass

    # R2 increment on OOF: ridge map L(R2)-L(sriram) 
    Bt_r = np.column_stack([Bt_s, uL(c1), uL(p2_live)])
    Bo_r = np.column_stack([Bo_s, unitize(f.z_s8 + 0.020 * u_e2), unitize(z_sr_oof_L)])
    lr_inc = Ridge(alpha=10.0, fit_intercept=True).fit(Bt_r, rank_logit(r2_live) - rank_logit(sriram))
    delta_r2_o = Bo_r @ lr_inc.coef_ + lr_inc.intercept_
    # scale delta to match test magnitude roughly
    delta_r2_o = unitize(delta_r2_o) * float(np.std(rank_logit(r2_live) - rank_logit(sriram)))

    z_R2_sri_ridge = z_sri_ridge + delta_r2_o
    z_R2_sri_nnls = z_sri_nnls + delta_r2_o
    z_S8 = f.z_s8
    z_R2_hybrid = 0.75 * z_R2_sri_ridge + 0.25 * z_S8
    z_s9 = f.z_s8 + 0.020 * u_e2
    z_R2_z9 = z_s9 + delta_r2_o  # logged soft — meanΔ≥0 only

    bases = {
        "z_R2_sri_ridge": z_R2_sri_ridge,
        "z_R2_sri_nnls": z_R2_sri_nnls,
        "z_R2_hybrid_twin": z_R2_hybrid,
    }
    for nm, z in bases.items():
        log(f"  {nm} AUC={roc_auc_score(y, z):.6f}")
    log(f"  z_R2_z9 AUC={roc_auc_score(y, z_R2_z9):.6f} (soft meanΔ≥0 gate only)")

    # ---------- signed locked vectors ----------
    log("=== Step1b: GS-orth + sign + variance lock ===")

    def gs_lock(vo, vt, name, prefer_sign=+1, hard=True):
        oo, ot, _ = frozen_lofo_orth(vo, vt, B_gs_o, B_gs_t, folds, ridge_alpha=10.0)
        ot = project_out(ot, B_gs_t)
        ot = var_match(ot, oo)
        oo, ot = unitize(oo), unitize(ot)
        # choose sign that lifts ridge base at w=0.006 on 5/5
        best_sign, best_pos, best_mean = prefer_sign, -1, -1e9
        for sgn in (+1, -1):
            per = h2h(bases["z_R2_sri_ridge"] + 0.006 * sgn * oo, bases["z_R2_sri_ridge"], y, folds)
            if (per["pos"], per["mean"]) > (best_pos, best_mean):
                best_sign, best_pos, best_mean = sgn, per["pos"], per["mean"]
        oo, ot = best_sign * oo, best_sign * ot
        # assert 5/5 at 0.006
        per = h2h(bases["z_R2_sri_ridge"] + 0.006 * oo, bases["z_R2_sri_ridge"], y, folds)
        disp = float(np.std(ot) / (np.std(oo) + 1e-12))
        mc = max_abs_corr(ot, B_macro_t)
        c6 = float(np.corrcoef(ot, f.bad6)[0, 1])
        c7 = float(np.corrcoef(ot, f.bad7)[0, 1])
        ok = per["pos"] == 5
        log(
            f"  {name}: sign={best_sign:+d} assert5/5={ok} pos={per['pos']}/5 "
            f"m={per['mean']:+.2f} t={per['t']:.2f} disp={disp:.4f} max|B|={mc:.6f} "
            f"c6={c6:+.4f} c7={c7:+.4f}"
        )
        if hard:
            assert ok, f"{name} failed 5/5 sign lock"
            assert mc < 0.001, f"{name} max|B|={mc}"
        return oo, ot, per, {"disp": disp, "mc": mc, "c6": c6, "c7": c7, "sign": best_sign, "ok": ok}

    # raw vectors
    Craw_o = 0.55 * f.u_cat + 0.45 * f.u_r2
    Craw_t = 0.55 * f.ut_cat + 0.45 * f.ut_r2
    if "F_lgb_ms39_oof" in fam1.files:
        Craw_o = 0.45 * f.u_cat + 0.35 * f.u_r2 + 0.20 * unitize(fam1["F_lgb_ms39_oof"].ravel()[:N_TR])
        Craw_t = 0.45 * f.ut_cat + 0.35 * f.ut_r2 + 0.20 * unitize(fam1["F_lgb_ms39_test"].ravel()[:N_TE])

    S = {}
    v8_o, v8_t = fam["v10_ridge_resid_orth_LM_oof"].ravel()[:N_TR], fam["v10_ridge_resid_orth_LM_test"].ravel()[:N_TE]
    S["V8"] = gs_lock(v8_o, v8_t, "S_V8_v10_ridge", +1, hard=True)
    S["V1"] = gs_lock(Craw_o, Craw_t, "S_V1_UC_tab", +1, hard=True)
    x_o = np.load(PARADIGM_B / "xgb_llr_oof.npy").ravel()[:N_TR]
    x_t = np.load(PARADIGM_B / "xgb_llr_test.npy").ravel()[:N_TE]
    S["V7"] = gs_lock(x_o, x_t, "S_V7_B_xgb_llr", +1, hard=True)
    s_o = fam["Q17_sameer_glm10f_oof"].ravel()[:N_TR]
    s_t = fam["Q17_sameer_glm10f_test"].ravel()[:N_TE]
    S["V2"] = gs_lock(s_o, s_t, "S_V2_sameer_glm10f", +1, hard=True)
    # Ray1 continuation (used in K2 dir; soft assert)
    ray1_t = unitize(rank_logit(r2_live) - rank_logit(p2_live))
    ray1_o = unitize(delta_r2_o)
    S["V12"] = gs_lock(ray1_o, ray1_t, "S_V12_Ray1_cont", +1, hard=False)

    # ---------- Step2: U_orth4 ----------
    log("=== Step2: build U_orth4 + K1/K2/K3 ===")
    # weight by t-stat from ridge H2H at 0.006
    weights = {}
    for k in ("V8", "V1", "V7", "V2"):
        weights[k] = max(S[k][2]["t"], 0.5)
    wsum = sum(weights.values())
    u4_o = np.zeros(N_TR)
    u4_t = np.zeros(N_TE)
    for k in ("V8", "V1", "V7", "V2"):
        wk = weights[k] / wsum
        u4_o += wk * S[k][0]
        u4_t += wk * S[k][1]
        log(f"  orth4 weight {k}={wk:.3f} (t={S[k][2]['t']:.2f})")
    # re-project test
    u4_t = project_out(u4_t, B_gs_t)
    u4_t = var_match(u4_t, u4_o)
    u4_o, u4_t = unitize(u4_o), unitize(u4_t)
    # orient
    if roc_auc_score(y[f.te0], bases["z_R2_sri_ridge"][f.te0] + 0.006 * u4_o[f.te0]) < roc_auc_score(
        y[f.te0], bases["z_R2_sri_ridge"][f.te0] - 0.006 * u4_o[f.te0]
    ):
        u4_o, u4_t = -u4_o, -u4_t

    disp4 = float(np.std(u4_t) / (np.std(u4_o) + 1e-12))
    mc4 = max_abs_corr(u4_t, B_macro_t)
    c6_4 = float(np.corrcoef(u4_t, f.bad6)[0, 1])
    c7_4 = float(np.corrcoef(u4_t, f.bad7)[0, 1])
    log(f"U_orth4: disp={disp4:.4f} max|B|={mc4:.6f} c6={c6_4:+.4f} c7={c7_4:+.4f}")

    per4 = {b: h2h(bases[b] + 0.010 * u4_o, bases[b], y, folds) for b in bases}
    per4_z9 = h2h(z_R2_z9 + 0.010 * u4_o, z_R2_z9, y, folds)
    for b in bases:
        log(
            f"  U_orth4 vs {b}: {per4[b]['pos']}/5 m={per4[b]['mean']:+.2f} "
            f"min={per4[b]['min']:+.2f} t={per4[b]['t']:.2f} sf={per4[b]['sf25']}"
        )
    log(f"  U_orth4 vs z_R2_z9: mean={per4_z9['mean']:+.2f} pos={per4_z9['pos']}/5 (soft)")

    orth4_ok = (
        all(per4[b]["pos"] == 5 and per4[b]["min"] > 0 and per4[b]["t"] >= 5.0 for b in bases)
        and per4_z9["mean"] >= 0
        and mc4 < 0.0001
        and abs(c6_4) < 0.005
        and abs(c7_4) < 0.005
    )
    log(f"U_orth4 ASSERT ok={orth4_ok}")

    # K1: pure orth4 @ 0.35e-4 from R2 (iterative B_macro kill after rank_frac)
    w1, k1, rho1, d1 = inject_orth_rho(r2_live, u4_t, 0.35, B_macro_t)
    # K2: Ray1 crest step + orth4 → 0.45e-4 from R2
    ray_cont = unitize(project_out(rank_logit(r2_live) - rank_logit(p2_live), B_macro_t))
    dir_k2 = unitize(project_out(0.35 * ray_cont + 1.0 * u4_t, B_gs_t))
    w2, k2, rho2, d2 = inject_orth_rho(r2_live, dir_k2, 0.45, B_macro_t)
    # K3: only V8+V1 @ 0.30e-4
    u31_o = unitize(0.55 * S["V8"][0] + 0.45 * S["V1"][0])
    u31_t = unitize(var_match(project_out(0.55 * S["V8"][1] + 0.45 * S["V1"][1], B_gs_t), u31_o))
    if roc_auc_score(y[f.te0], bases["z_R2_sri_ridge"][f.te0] + 0.006 * u31_o[f.te0]) < roc_auc_score(
        y[f.te0], bases["z_R2_sri_ridge"][f.te0] - 0.006 * u31_o[f.te0]
    ):
        u31_o, u31_t = -u31_o, -u31_t
    w3, k3, rho3, d3 = inject_orth_rho(r2_live, u31_t, 0.30, B_macro_t)

    def eval_k(tag, te, uo, w_used, target, d_inj):
        # OOF: add matched weight
        oof_add = w_used * uo
        per = {b: h2h(bases[b] + oof_add, bases[b], y, folds) for b in bases}
        per_z9 = h2h(z_R2_z9 + oof_add, z_R2_z9, y, folds)
        delta = unitize(rank_logit(te) - rank_logit(r2_live))
        # Gate max|B| on injection direction (exact OLS orth); log rank_frac residual separately
        mc = max_abs_corr(d_inj, B_macro_t)
        mc_delta = max_abs_corr(delta, B_macro_t)
        c6 = float(np.corrcoef(d_inj, f.bad6)[0, 1])
        c7 = float(np.corrcoef(d_inj, f.bad7)[0, 1])
        disp = float(np.std(d_inj) / (np.std(uo) + 1e-12)) if np.std(uo) > 0 else 1.0
        row = {
            "tag": tag,
            "test": te,
            "per": per,
            "per_z9": per_z9,
            "rho_r2": one_m_rho_e4(te, r2_live),
            "rho_p2": one_m_rho_e4(te, p2_live),
            "rho_s": one_m_rho_e4(te, sriram),
            "mc": mc,
            "mc_delta": mc_delta,
            "c6": c6,
            "c7": c7,
            "disp": disp,
            "w": w_used,
            "target": target,
        }
        log(
            f"{tag}: ρ(R2)={row['rho_r2']:.3f} ρ(sri)={row['rho_s']:.3f} "
            f"max|B|={mc:.6f} (Δraw={mc_delta:.6f}) disp={disp:.4f} "
            f"c6={c6:+.4f} c7={c7:+.4f} | "
            + " | ".join(
                f"{b}:{per[b]['pos']}/5 t={per[b]['t']:.2f} min={per[b]['min']:+.2f}" for b in bases
            )
            + f" | z9 mean={per_z9['mean']:+.2f}"
        )
        return row

    k1c = eval_k("K1_Pure_Orth4_on_R2_0p35e4", k1, u4_o, w1, 0.35, d1)
    k2c = eval_k("K2_Ray1_Crest_2p05e4_plus_Orth4_0p45e4", k2, unitize(0.35 * S["V12"][0] + 1.0 * u4_o), w2, 0.45, d2)
    k3c = eval_k("K3_Pure_V8_v10_ridge_plus_V1_UC_0p30e4", k3, u31_o, w3, 0.30, d3)

    def stage(path, vec):
        v = np.asarray(vec, float)
        assert len(v) == N_TE and np.isfinite(v).all()
        pd.DataFrame({"id": ids, "is_fraud": v}).to_csv(path, index=False)
        log(f"STAGED {path.name}")

    p_k1 = SUB / "sub_v22_5_K1_pure_orth4_0p35e4.csv"
    p_k2 = SUB / "sub_v22_5_K2_crest_plus_orth4_0p45e4.csv"
    p_k3 = SUB / "sub_v22_5_K3_v8_v1_0p30e4.csv"
    stage(p_k1, k1c["test"])
    stage(p_k2, k2c["test"])
    stage(p_k3, k3c["test"])

    # ---------- Step3: gate + submit ----------
    log("=== Step3: gate check + conditional submit ===")

    def clears(q):
        per = q["per"]
        ok = (
            all(per[b]["pos"] == 5 and per[b]["min"] > 0.20 and per[b]["t"] >= 4.0 for b in bases)
            and q["per_z9"]["mean"] >= 0
            and q["mc"] < 0.001
            and abs(q["c6"]) < 0.01
            and abs(q["c7"]) < 0.01
            and 0.25 <= q["rho_r2"] <= 0.50
        )
        log(f"  gate {q['tag']}: ok={ok}")
        return ok

    pool = [k for k in (k2c, k1c, k3c) if clears(k)]
    # prefer K2 then K1 then K3 among clearers
    order = {"K2_Ray1_Crest_2p05e4_plus_Orth4_0p45e4": 0, "K1_Pure_Orth4_on_R2_0p35e4": 1, "K3_Pure_V8_v10_ridge_plus_V1_UC_0p30e4": 2}
    pool = sorted(pool, key=lambda q: (order.get(q["tag"], 9), -q["per"]["z_R2_sri_ridge"]["t"]))

    submit_ok, pub, reason = False, None, "HOLD_0.95426"
    if pool:
        submit_row = pool[0]
        live = SUB / "sub_v22_5_LIVE_crest_orth4.csv"
        stage(live, submit_row["test"])
        msg = (
            f"v22.5 {submit_row['tag']} | ρ(R2)={submit_row['rho_r2']:.3f}e-4 ρ(sri)={submit_row['rho_s']:.3f}e-4 "
            f"OOF ridge {submit_row['per']['z_R2_sri_ridge']['pos']}/5 "
            f"t={submit_row['per']['z_R2_sri_ridge']['t']:.2f} min={submit_row['per']['z_R2_sri_ridge']['min']:+.2f}"
        )
        log(f"SUBMIT {msg}")
        submit_ok, pub = submit_and_poll(live, msg)
        reason = submit_row["tag"]
    else:
        log("NO candidate cleared gates — HOLD 0.95426")
        reason = "HOLD_0.95426_no_gate_clear"

    meta = {
        "reason": reason,
        "submit_ok": submit_ok,
        "pub": pub,
        "orth4_ok": orth4_ok,
        "orth4": {"disp": disp4, "mc": mc4, "c6": c6_4, "c7": c7_4},
        "weights": weights,
        "K": {r["tag"]: {"rho_r2": r["rho_r2"], "rho_s": r["rho_s"], "mc": r["mc"], "c7": r["c7"]} for r in (k1c, k2c, k3c)},
    }
    OUT.joinpath("v22_5_meta.json").write_text(json.dumps(meta, indent=2, default=float))

    body = [
        "# GM V22.5 Crest + Orth4 Report",
        "",
        f"**elapsed** {time.time()-T0:.0f}s · local Mac CPU · ATH `0.95426`",
        "",
        f"## Decision: **{reason}**",
        f"- Submit OK: `{submit_ok}` · Public LB: **{pub}**",
        f"- U_orth4 assert: `{orth4_ok}` · max|B|=`{mc4:.6f}` disp=`{disp4:.4f}`",
        "",
        "## Sign-locked vectors (vs z_R2_sri_ridge @ w=0.006)",
    ]
    for k in ("V8", "V1", "V7", "V2", "V12"):
        per, meta_v = S[k][2], S[k][3]
        body.append(
            f"- `{k}` sign=`{meta_v['sign']:+d}` pos=`{per['pos']}/5` t=`{per['t']:.2f}` "
            f"disp=`{meta_v['disp']:.4f}` max|B|=`{meta_v['mc']:.6f}`"
        )
    body += [
        "",
        "## Candidates vs R2",
        "| cand | ρ(R2) | ρ(sri) | ridge pos/t/min | nnls | hybrid | z9 mean | max\\|B\\| dir | Δraw |",
        "|---|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for r in (k1c, k2c, k3c):
        pr, pn, ph = r["per"]["z_R2_sri_ridge"], r["per"]["z_R2_sri_nnls"], r["per"]["z_R2_hybrid_twin"]
        body.append(
            f"| `{r['tag']}` | {r['rho_r2']:.3f} | {r['rho_s']:.3f} | "
            f"{pr['pos']}/5 t={pr['t']:.2f} min={pr['min']:+.2f} | "
            f"{pn['pos']}/5 t={pn['t']:.2f} | {ph['pos']}/5 t={ph['t']:.2f} | "
            f"{r['per_z9']['mean']:+.2f} | {r['mc']:.6f} | {r['mc_delta']:.6f} |"
        )
    body += [
        "",
        "## Staged",
        f"- `{p_k1.name}`",
        f"- `{p_k2.name}`",
        f"- `{p_k3.name}`",
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
