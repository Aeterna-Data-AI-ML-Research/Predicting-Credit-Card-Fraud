#!/usr/bin/env python3
"""V23 — Pure Ray-1 crest @ 1.55e-4 / 1.75e-4 from sriram + one −U_orth twilight @ 0.35e-4 from R2.

Local Mac CPU. Three submits with exact-fname poll.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

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
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
REPORT = LOG / "GM_V23_RAY1_AND_NEG_TWILIGHT_REPORT.md"
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
    v = unitize(v)
    M = np.asarray(M, float)
    Mc = M - M.mean(0, keepdims=True)
    Ms = Mc / (Mc.std(0, keepdims=True) + 1e-12)
    return float(np.max(np.abs(Ms.T @ v / len(v)))) if M.size else 0.0


def var_match(ot, oo):
    return ot * (float(np.std(oo)) + 1e-12) / (float(np.std(ot)) + 1e-12)


def sigmoid(x):
    return expit(np.clip(x, -40, 40))


def twilight_share(base, cand, q_lo=0.78, q_hi=0.992):
    r = (rankdata(base, method="average") - 0.5) / len(base)
    d = np.abs(cand - base)
    m = (r >= q_lo) & (r <= q_hi)
    return float(d[m].sum() / (d.sum() + 1e-18))


def bisect_rho_vs(ref, base_L, direction, target_e4, lo=1e-6, hi=2.0, n=32):
    """1−ρ(cand, ref) ≈ target_e4 with cand = rank_frac(base_L + w*d)."""
    ref = np.asarray(ref, float).ravel()
    d = unitize(direction)
    base_L = np.asarray(base_L, float).ravel()
    best_w, best_err, best_c = 0.0, 1e9, rank_frac(base_L)
    a, b = lo, hi
    for _ in range(n):
        mid = 0.5 * (a + b)
        cand = rank_frac(base_L + mid * d)
        rho = one_m_rho_e4(cand, ref)
        err = abs(rho - target_e4)
        if err < best_err:
            best_err, best_w, best_c = err, mid, cand
        if rho < target_e4:
            a = mid
        else:
            b = mid
    return best_w, best_c, one_m_rho_e4(best_c, ref)


def inject_twilight_rho(base, direction, target_e4, B_macro, sign=-1.0, min_tw=0.55, n_sharpen=4):
    target_e4 = float(np.clip(target_e4, 0.28, 0.42))
    base = np.asarray(base, float).ravel()
    d0 = unitize(project_out(np.asarray(direction, float).ravel(), B_macro)) * float(sign)
    q = rank_frac(base)
    best = None
    scale_lo, scale_hi = 0.04, 0.004
    for k in range(n_sharpen):
        lo_s = scale_lo * (0.70 ** k)
        hi_s = scale_hi * (0.70 ** k)
        gate = sigmoid((q - 0.78) / lo_s) * sigmoid((0.992 - q) / hi_s)
        d_eff = unitize(d0 * gate)
        # bisect vs base (displacement from R2)
        a, b = 1e-6, 0.5
        best_w, best_c, best_rho = 0.0, base.copy(), 0.0
        best_err = 1e9
        for _ in range(28):
            mid = 0.5 * (a + b)
            cand = rank_frac(rank_logit(base) + mid * d_eff)
            rho = one_m_rho_e4(cand, base)
            err = abs(rho - target_e4)
            if err < best_err:
                best_err, best_w, best_c, best_rho = err, mid, cand, rho
            if rho < target_e4:
                a = mid
            else:
                b = mid
        tw = twilight_share(base, best_c)
        row = (tw, best_w, best_c, best_rho, d_eff)
        if best is None or tw > best[0]:
            best = row
        if tw >= min_tw:
            break
    return best[1], best[2], best[3], best[4], best[0]


def submit_and_poll(path: Path, msg: str):
    pub = None
    ok = False
    try:
        r = subprocess.run(
            [KAG, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg],
            capture_output=True,
            text=True,
            timeout=120,
        )
        log(r.stdout.strip()[:400])
        log(r.stderr.strip()[:400])
        blob = (r.stdout + r.stderr).lower()
        ok = r.returncode == 0 and "400" not in (r.stdout + r.stderr) and "limit" not in blob
        if not ok:
            return False, None
        for i in range(40):
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
            for ln in lines[1:15]:
                if path.name not in ln:
                    continue
                log(f"poll {ln[:240]}")
                if "COMPLETE" in ln.upper():
                    cols = [c.strip() for c in ln.split(",")]
                    try:
                        pub = float(cols[hdr.index("publicScore")])
                    except Exception:
                        for tok in reversed(cols):
                            try:
                                pub = float(tok)
                                break
                            except ValueError:
                                continue
                    return True, pub
    except Exception as e:
        log(f"submit error: {e}")
        traceback.print_exc()
    return ok, pub


def stage(path: Path, ids, vec):
    v = np.asarray(vec, float).ravel()
    assert len(v) == N_TE and np.isfinite(v).all()
    pd.DataFrame({"id": ids, "is_fraud": v}).to_csv(path, index=False)
    log(f"STAGED {path.name}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    log("=== V23 Ray-1 crest + −U_orth twilight ===")
    # Fast path: Ray-1 needs only CSVs
    sample = pd.read_csv(R2_ATH)
    ids = sample["id"].to_numpy()
    assert len(ids) == N_TE

    def load_sub(path):
        return pd.read_csv(path).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    raw_gp = load_sub(SRIRAM_DS / "reference.csv")
    raw_new = load_sub(SRIRAM_DS / "new_neural.csv")
    raw_old = load_sub(SRIRAM_DS / "old_neural.csv")
    sriram = rank_frac(rank_logit(raw_gp) + 0.15 * (rank_logit(raw_new) - rank_logit(raw_old)))
    r2 = load_sub(R2_ATH)
    p2 = load_sub(P2_ATH)
    log(
        f"anchors 1−ρ(sri): P2={one_m_rho_e4(p2, sriram):.3f}e-4 "
        f"R2={one_m_rho_e4(r2, sriram):.3f}e-4 R2−P2={one_m_rho_e4(r2, p2):.3f}e-4"
    )

    # ---------- Pure Ray-1 from sriram along R2−sriram (no orth) ----------
    ray1 = unitize(rank_logit(r2) - rank_logit(sriram))
    results = []

    for target, tag in ((1.55, "R1_1p55e4"), (1.75, "R1_1p75e4")):
        w, cand, rho_s = bisect_rho_vs(sriram, rank_logit(sriram), ray1, target)
        rho_r2 = one_m_rho_e4(cand, r2)
        rho_p2 = one_m_rho_e4(cand, p2)
        path = SUB / f"sub_v23_{tag}_pure_ray1.csv"
        stage(path, ids, cand)
        msg = (
            f"v23.{tag} pure Ray-1 no-orth | ρ(sri)={rho_s:.3f}e-4 "
            f"ρ(R2)={rho_r2:.3f}e-4 ρ(P2)={rho_p2:.3f}e-4 w={w:.5f}"
        )
        log(f"SUBMIT {msg}")
        ok, pub = submit_and_poll(path, msg)
        results.append(
            {"tag": tag, "path": path.name, "rho_s": rho_s, "rho_r2": rho_r2, "rho_p2": rho_p2, "ok": ok, "pub": pub, "w": w}
        )
        log(f"{tag} DONE pub={pub} ok={ok}")
        time.sleep(12)

    # ---------- −U_orth_super twilight @ 0.35e-4 from R2 ----------
    log("=== rebuild factory + U_orth4 (V1+V7+V8+V2) for −twilight ===")
    v20 = load_v20()
    h2h, frozen_lofo_orth, L_fold = v20.h2h, v20.frozen_lofo_orth, v20.L_fold
    fac = v20.OvernightFactory()
    fac.rebuild()
    f = fac
    ids, y, folds, fold_id = f.ids, f.y, f.folds, f.fold_id
    # realign vectors to factory id order
    sriram = pd.Series(sriram, index=sample["id"].to_numpy()).loc[ids].to_numpy(float)
    r2 = pd.Series(r2, index=sample["id"].to_numpy()).loc[ids].to_numpy(float)
    p2 = pd.Series(p2, index=sample["id"].to_numpy()).loc[ids].to_numpy(float)
    raw_gp = pd.Series(raw_gp, index=sample["id"].to_numpy()).loc[ids].to_numpy(float)
    raw_new = pd.Series(raw_new, index=sample["id"].to_numpy()).loc[ids].to_numpy(float)
    raw_old = pd.Series(raw_old, index=sample["id"].to_numpy()).loc[ids].to_numpy(float)

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
    B_gs_o = np.column_stack([B_macro_o, uB_o, uD_o, uE_o, bad6_o, bad7_o]).astype(np.float64)
    B_gs_t = np.column_stack([B_macro_t, uB_t, uD_t, uE_t, f.bad6, f.bad7]).astype(np.float64)

    # R2 sri ridge base for sign lock (same as V22.5 lite)
    Bt_s = np.column_stack([Bt_map, uB_t, uD_t, uE_t])
    Bo_s = np.column_stack([Bo_map, uB_o, uD_o, uE_o])
    lr_sri = Ridge(alpha=50.0, fit_intercept=True).fit(Bt_s, rank_logit(sriram))
    z_sri_ridge = unitize(Bo_s @ lr_sri.coef_ + lr_sri.intercept_) * float(np.std(f.z_s8)) + float(np.mean(f.z_s8))
    Bt_r = np.column_stack([Bt_s, uL(p2), uL(r2)])
    Bo_r = np.column_stack([Bo_s, unitize(f.z_s8 + 0.020 * u_e2), unitize(z_gp_oof_L + 0.15 * dmlp_o)])
    lr_inc = Ridge(alpha=10.0, fit_intercept=True).fit(Bt_r, rank_logit(r2) - rank_logit(sriram))
    delta_r2_o = unitize(Bo_r @ lr_inc.coef_ + lr_inc.intercept_) * float(
        np.std(rank_logit(r2) - rank_logit(sriram))
    )
    z_R2_sri_ridge = z_sri_ridge + delta_r2_o

    Craw_o = 0.45 * f.u_cat + 0.35 * f.u_r2
    Craw_t = 0.45 * f.ut_cat + 0.35 * f.ut_r2
    if "F_lgb_ms39_oof" in fam1.files:
        Craw_o = 0.45 * f.u_cat + 0.35 * f.u_r2 + 0.20 * unitize(fam1["F_lgb_ms39_oof"].ravel()[:N_TR])
        Craw_t = 0.45 * f.ut_cat + 0.35 * f.ut_r2 + 0.20 * unitize(fam1["F_lgb_ms39_test"].ravel()[:N_TE])

    raws = {
        "V1": (Craw_o, Craw_t),
        "V8": (
            fam["v10_ridge_resid_orth_LM_oof"].ravel()[:N_TR],
            fam["v10_ridge_resid_orth_LM_test"].ravel()[:N_TE],
        ),
        "V7": (
            np.load(PARADIGM_B / "xgb_llr_oof.npy").ravel()[:N_TR],
            np.load(PARADIGM_B / "xgb_llr_test.npy").ravel()[:N_TE],
        ),
        "V2": (
            fam["Q17_sameer_glm10f_oof"].ravel()[:N_TR],
            fam["Q17_sameer_glm10f_test"].ravel()[:N_TE],
        ),
    }
    locked = {}
    for name, (vo, vt) in raws.items():
        oo, ot, _ = frozen_lofo_orth(vo, vt, B_gs_o, B_gs_t, folds, ridge_alpha=10.0)
        ot = var_match(project_out(ot, B_gs_t), oo)
        oo, ot = unitize(oo), unitize(ot)
        # LOFO-agree sign vs ridge base
        signs = []
        for i, (_, te) in enumerate(folds):
            best_s, best_m = +1, -1e18
            for sgn in (+1, -1):
                ds = []
                for j, (_, te2) in enumerate(folds):
                    if j == i:
                        continue
                    ds.append(
                        (
                            roc_auc_score(y[te2], z_R2_sri_ridge[te2] + 0.006 * sgn * oo[te2])
                            - roc_auc_score(y[te2], z_R2_sri_ridge[te2])
                        )
                        * 1e5
                    )
                m = float(np.mean(ds))
                if m > best_m:
                    best_m, best_s = m, sgn
            signs.append(best_s)
        sgn = int(signs[0]) if len(set(signs)) == 1 else (
            +1
            if h2h(z_R2_sri_ridge + 0.006 * oo, z_R2_sri_ridge, y, folds)["mean"]
            >= h2h(z_R2_sri_ridge - 0.006 * oo, z_R2_sri_ridge, y, folds)["mean"]
            else -1
        )
        oo, ot = sgn * oo, sgn * ot
        per = h2h(z_R2_sri_ridge + 0.006 * oo, z_R2_sri_ridge, y, folds)
        locked[name] = (oo, ot, per)
        log(f"  {name}: sgn={sgn:+d} ridge {per['pos']}/5 t={per['t']:.2f} m={per['mean']:+.2f}")

    wsum = sum(max(locked[k][2]["t"], 0.5) for k in locked)
    u4_o = sum(max(locked[k][2]["t"], 0.5) / wsum * locked[k][0] for k in locked)
    u4_t = sum(max(locked[k][2]["t"], 0.5) / wsum * locked[k][1] for k in locked)
    u4_o, u4_t = unitize(u4_o), unitize(var_match(project_out(u4_t, B_gs_t), u4_o))
    if roc_auc_score(y[f.te0], z_R2_sri_ridge[f.te0] + 0.006 * u4_o[f.te0]) < roc_auc_score(
        y[f.te0], z_R2_sri_ridge[f.te0] - 0.006 * u4_o[f.te0]
    ):
        u4_o, u4_t = -u4_o, -u4_t
    log(f"U_orth4 max|B|={max_abs_corr(u4_t, B_macro_t):.6f}")

    # Force negative twilight (V22_6_TWI_SIGN=-1)
    w3, cand3, rho3, d3, tw3 = inject_twilight_rho(r2, u4_t, 0.35, B_macro_t, sign=-1.0, min_tw=0.55)
    path3 = SUB / "sub_v23_NEG_TWI_Uorth_0p35e4.csv"
    stage(path3, ids, cand3)
    msg3 = (
        f"v23.NEG_TWI −U_orth4 @0.35e-4 from R2 | ρ(R2)={rho3:.3f}e-4 "
        f"ρ(sri)={one_m_rho_e4(cand3, sriram):.3f}e-4 tw={tw3:.3f} sign=-1"
    )
    log(f"SUBMIT {msg3}")
    ok3, pub3 = submit_and_poll(path3, msg3)
    results.append(
        {
            "tag": "NEG_TWI_0p35e4",
            "path": path3.name,
            "rho_s": one_m_rho_e4(cand3, sriram),
            "rho_r2": rho3,
            "tw": tw3,
            "ok": ok3,
            "pub": pub3,
            "w": w3,
        }
    )
    log(f"NEG_TWI DONE pub={pub3} ok={ok3} tw={tw3:.3f}")

    # report
    body = [
        "# GM V23 Ray-1 Crest + −U_orth Twilight Report",
        "",
        f"**elapsed** {time.time()-T0:.0f}s · ATH was `0.95426`",
        "",
        "## Live results",
        "| tag | Public LB | ρ(sri) e-4 | ρ(R2) e-4 | notes |",
        "|---|---:|---:|---:|---|",
    ]
    for r in results:
        notes = f"tw={r['tw']:.3f}" if "tw" in r else "pure Ray-1"
        body.append(
            f"| `{r['tag']}` | **{r.get('pub')}** | {r.get('rho_s', float('nan')):.3f} | "
            f"{r.get('rho_r2', float('nan')):.3f} | {notes} |"
        )
    best = max((r.get("pub") for r in results if r.get("pub") is not None), default=None)
    body += [
        "",
        f"- Best among probes: **{best}** (ATH hold if `< 0.95426`)",
        "",
        "## Log",
        "```",
        *LINES,
        "```",
        "",
    ]
    REPORT.write_text("\n".join(body) + "\n")
    OUT.joinpath("v23_ray1_neg_twi_meta.json").write_text(
        json.dumps({"results": results, "elapsed": time.time() - T0}, indent=2, default=float)
    )
    log(f"REPORT {REPORT}")
    log(f"DONE best={best}")


if __name__ == "__main__":
    main()
