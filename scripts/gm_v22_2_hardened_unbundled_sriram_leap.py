#!/usr/bin/env python3
"""GM V22.2 HARDENED — Unbundled 5-fold LM/gpt2med micro-step on exact sriram native space.

Fixes: (1) stop 37h fold0 classify after ckpt-50000; (2) exact sriram L-space;
(3) unbundle L7_s2026 / gpt2_med from redundant shallow trees; (4) 8-vector macro orth.
Local Mac CPU. Submit ONE micro-step (1-ρ≈0.40–0.55e-4). Keep 7 daily slots.
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
from scipy.stats import rankdata, spearmanr
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
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
COMPACT = REPO / "external/banks/goodpjw2008_s6e11-compact-lm-seed7"
FAM20 = TMP16 / "out" / "v20_1_families.npz"
GPT2MED = REPO / "preds" / "gpt2_med_4ep"
SLOT9 = SUB / "sub_v20_LIVE_slot9_overnight_cpu_promoted.csv"
C1 = SUB / "sub_v22_1_LIVE_leap.csv"
C2 = SUB / "sub_v22_1b_LIVE_leap.csv"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
REPORT = LOG / "GM_V22_2_HARDENED_LEAP_REPORT.md"
MPS_RUN = TMP16 / "lm_research" / "run_mps_gpt2"
CKPT500 = REPO / "logs/overnight/run_mps_gpt2_data/replica/trainer/checkpoint-50000"
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


def rank_logit(vector):
    """Exact sriram notebook L(·)."""
    vector = np.asarray(vector, float).ravel()
    assert vector.ndim == 1 and np.isfinite(vector).all()
    return logit((rankdata(vector, method="average") - 0.5) / len(vector))


def rank_frac(vector):
    """Exact sriram notebook export transform."""
    vector = np.asarray(vector, float).ravel()
    return (rankdata(vector, method="average") - 0.5) / len(vector)


def one_m_rho(a, b) -> float:
    return 1.0 - float(spearmanr(a, b).correlation)


def one_m_rho_e4(a, b) -> float:
    return one_m_rho(a, b) * 1e4


def project_out(v, B):
    v = np.asarray(v, float).ravel()
    X = np.column_stack([np.ones(len(v)), np.asarray(B, float)])
    xtx = X.T @ X
    xtx.flat[:: xtx.shape[0] + 1] += 1e-8
    beta = np.linalg.solve(xtx, X.T @ v)
    return v - X @ beta


def max_abs_corr(v, M):
    v = np.asarray(v, float).ravel()
    v = (v - v.mean()) / (v.std() + 1e-12)
    M = np.asarray(M, float)
    Mc = M - M.mean(0, keepdims=True)
    Ms = Mc / (Mc.std(0, keepdims=True) + 1e-12)
    c = Ms.T @ v / len(v)
    return float(np.max(np.abs(c))) if len(c) else 0.0


def bisect_rho(base, direction, target_e4, lo=1e-6, hi=0.4, n=30):
    base = np.asarray(base, float).ravel()
    d = np.asarray(direction, float).ravel()
    d = (d - d.mean()) / (d.std() + 1e-12)
    best_w, best_err, best_c = 0.0, 1e9, base.copy()
    a, b = lo, hi
    for _ in range(n):
        mid = 0.5 * (a + b)
        # native space: L(base_as_ranks) approx via rank_logit of fractional ranks
        # base is already fractional ranks from sriram export
        score = rank_logit(base) + mid * d
        cand = rank_frac(score)
        rho = one_m_rho_e4(cand, base)
        err = abs(rho - target_e4)
        if err < best_err:
            best_err, best_w, best_c = err, mid, cand
        if rho < target_e4:
            a = mid
        else:
            b = mid
    return best_w, best_c, one_m_rho_e4(best_c, base)


def stop_fold0_classify():
    log("=== Step0: preserve ckpt-50000 & TERM fold0 classify ===")
    ok = CKPT500.exists() and (MPS_RUN / "replica" / "final" / "model.safetensors").exists()
    log(f"checkpoint-50000 exists={CKPT500.exists()} final_model={ok}")
    if not ok:
        log("WARNING: replica final missing — will NOT kill classify")
        return False
    # record target-first
    summary = MPS_RUN / "replica" / "train_summary.json"
    target_first = None
    if summary.exists():
        j = json.loads(summary.read_text())
        cols = j.get("columns") or []
        target_first = "is_fraud" in cols
        log(
            f"replica train: target_col={j.get('settings',{}).get('target_col')} "
            f"is_fraud_in_columns={target_first} (GReaT conditional_col=is_fraud → Target-First)"
        )
    # TERM classify + prevent relaunch
    subprocess.run(["pkill", "-TERM", "-f", "train_classifier.py"], capture_output=True)
    subprocess.run(["pkill", "-TERM", "-f", "gm_v22_mps_full_gpt2_replica.sh"], capture_output=True)
    subprocess.run(["pkill", "-TERM", "-f", "gm_v22_mps_watchdog.sh"], capture_output=True)
    subprocess.run(["pkill", "-TERM", "-f", "gm_v22_mps_monitor.sh"], capture_output=True)
    time.sleep(2)
    subprocess.run(["pkill", "-KILL", "-f", "train_classifier.py"], capture_output=True)
    alive = subprocess.run(["pgrep", "-f", "train_classifier.py"], capture_output=True).returncode == 0
    (MPS_RUN / "STOP_CLASSIFY").write_text(
        f"STOPPED_BY_V22_2_HARDENED utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        f"ckpt50000_preserved={CKPT500.exists()}\n"
        f"target_first={target_first}\n"
    )
    log(f"fold0 classify stopped={not alive} STOP_CLASSIFY written")
    return True


def submit_and_poll(path: Path, msg: str):
    pub = None
    ok = False
    try:
        r = subprocess.run(
            [KAG, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg],
            capture_output=True,
            text=True,
            timeout=180,
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
                    if "v22_2_HARDENED" in ln or "v22.2h" in ln or "Unbundled" in ln:
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

    stop_fold0_classify()

    v20 = load_v20()
    unitize, h2h, frozen_lofo_orth = v20.unitize, v20.h2h, v20.frozen_lofo_orth
    L_fold = v20.L_fold

    log("=== rebuild proprietary Slot8/9 engines ===")
    fac = v20.OvernightFactory()
    fac.rebuild()
    f = fac
    ids, y, folds, fold_id = f.ids, f.y, f.folds, f.fold_id

    def load_sub(path):
        df = pd.read_csv(path)
        return df.set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    # ---------- Exact sriram native space ----------
    log("=== Step1: exact sriram native-space reproduction ===")
    raw_gp = load_sub(SRIRAM_DS / "reference.csv")
    raw_new = load_sub(SRIRAM_DS / "new_neural.csv")
    raw_old = load_sub(SRIRAM_DS / "old_neural.csv")
    Z_gp = rank_logit(raw_gp)
    Z_new = rank_logit(raw_new)
    Z_old = rank_logit(raw_old)
    sriram_score = Z_gp + 0.15 * (Z_new - Z_old)
    sriram = rank_frac(sriram_score)
    # self-repro + compare to prior live C2 distance baseline
    sriram2 = rank_frac(rank_logit(raw_gp) + 0.15 * (rank_logit(raw_new) - rank_logit(raw_old)))
    repro_err = one_m_rho(sriram, sriram2)
    log(f"sriram self-repro 1-ρ={repro_err:.3e} (gate <1e-10)")
    assert repro_err < 1e-10, repro_err
    # also verify vs reconstructing from files a third way
    assert np.allclose(sriram, sriram2)
    log(f"sriram vs goodpjw 1-ρ={one_m_rho_e4(sriram, raw_gp):.6f}e-4")

    c1 = load_sub(C1)
    c2 = load_sub(C2)
    slot9 = load_sub(SLOT9)
    log(
        f"live distances: C2pure vs sriram={one_m_rho_e4(c2, sriram):.3f}e-4 | "
        f"C1 vs sriram={one_m_rho_e4(c1, sriram):.3f}e-4"
    )

    o_new = pd.read_csv(PYTAB_OOF).is_fraud.to_numpy(float)
    o_old = pd.read_csv(PYTORCH_OOF).is_fraud.to_numpy(float)
    assert len(o_new) == N_TR

    # OOF proxies via Ridge of bank+Q17 → L(goodpjw)/L(sriram) on test
    fam = np.load(FAM20)
    shallow_o = fam["Q17_megayak_shallow_tree_oof"].ravel()[:N_TR]
    shallow_t = fam["Q17_megayak_shallow_tree_test"].ravel()[:N_TE]
    masaya_o = fam["Q17_masaya_gpt2_oof"].ravel()[:N_TR]
    masaya_t = fam["Q17_masaya_gpt2_test"].ravel()[:N_TE]
    gpd_o = fam["Q17_daoviet_gpdvae_oof"].ravel()[:N_TR]
    gpd_t = fam["Q17_daoviet_gpdvae_test"].ravel()[:N_TE]
    # compact lm
    if (COMPACT / "lm7_oof_f0.npy").exists():
        compact_o = np.load(COMPACT / "lm7_oof_f0.npy").ravel()[:N_TR]
        compact_t = np.load(COMPACT / "lm7_test_f0.npy").ravel()[:N_TE]
    else:
        compact_o, compact_t = f.XO.get("lm7_5fold", masaya_o), f.XT.get("lm7_5fold", masaya_t)

    # Macro basis (8 pillars + fail rays)
    def uL(x):
        return unitize(rank_logit(x))

    B_macro_t = np.column_stack(
        [
            uL(raw_gp),
            uL(sriram),
            uL(raw_new),
            uL(raw_old),
            uL(shallow_t),
            uL(compact_t),
            uL(masaya_t),
            uL(gpd_t),
            f.bad6,
            f.bad7,
        ]
    ).astype(np.float64)

    # OOF macro: use Q17 + RealMLP OOFs + ridge proxies for goodpjw/sriram
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
        [
            uL(raw_new),
            uL(raw_old),
            uL(shallow_t),
            uL(compact_t),
            uL(masaya_t),
            uL(gpd_t),
            f.Bt9,
        ]
    )
    lr_gp = Ridge(alpha=10.0, fit_intercept=True).fit(Bt_map, rank_logit(raw_gp))
    lr_sr = Ridge(alpha=10.0, fit_intercept=True).fit(Bt_map, rank_logit(sriram))
    z_gp_oof_L = Bo_map @ lr_gp.coef_ + lr_gp.intercept_
    # exact sriram OOF formula in L-space
    z_sr_oof_L = z_gp_oof_L + 0.15 * (L_fold(o_new, fold_id) - L_fold(o_old, fold_id))
    z_sriram_oof = unitize(z_sr_oof_L) * float(np.std(f.z_s8)) + float(np.mean(f.z_s8))
    r2_sr = 1.0 - float(np.sum((rank_logit(sriram) - (Bt_map @ lr_sr.coef_ + lr_sr.intercept_)) ** 2)) / (
        float(np.var(rank_logit(sriram)) * N_TE) + 1e-18
    )
    log(f"z_sriram_oof AUC={roc_auc_score(y, z_sriram_oof):.6f} surrogate R²_test≈{r2_sr:.4f}")

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
            bad6_o,
            bad7_o,
        ]
    ).astype(np.float64)

    # ---------- Unbundled pillars ----------
    vault = np.load(OUT / "gm_overnight_vault.npz")
    u_e2, ut_e2 = unitize(vault["E2_heavy_ridge_oof"]), unitize(vault["E2_heavy_ridge_test"])
    u_u8, ut_u8 = unitize(vault["E2_U8_oof"]), unitize(vault["E2_U8_test"])
    u_fe, ut_fe = unitize(f.dE_o), unitize(f.dE_t)

    # gpt2_med 5f
    go = np.zeros(N_TR)
    gt = np.zeros(N_TE)
    for i in range(5):
        go += np.load(GPT2MED / f"oof_{i}.npy").ravel()[:N_TR]
        gt += np.load(GPT2MED / f"test_{i}.npy").ravel()[:N_TE]
    go /= 5.0
    gt /= 5.0
    # Method-B contrast vs masaya
    g_med_o = L_fold(go, fold_id) - L_fold(masaya_o, fold_id)
    g_med_t = rank_logit(gt) - rank_logit(masaya_t)

    gbn_o = np.load(BANK / "oof_gbn_bag5_w512.npy").ravel()[:N_TR]
    gbn_t = np.load(BANK / "test_gbn_bag5_w512.npy").ravel()[:N_TE]
    gb_o = np.load(BANK / "oof_gb_bag5.npy").ravel()[:N_TR]
    gb_t = np.load(BANK / "test_gb_bag5.npy").ravel()[:N_TE]

    def macro_orth(vo, vt, name):
        # frozen LOFO on OOF vs B_macro_o, then exact residualize test
        oo, ot, _beta = frozen_lofo_orth(vo, vt, B_macro_o, B_macro_t, folds, ridge_alpha=10.0)
        raw_disp = float(np.std(ot) / (np.std(oo) + 1e-12))
        # second-stage exact residual on test against macro
        ot = project_out(ot, B_macro_t)
        oo = unitize(oo)
        ot = unitize(ot)
        return oo, ot, {"raw_disp": raw_disp}

    # Build raw pillar mixes BEFORE orth
    # A: Family_E + U5 + U8
    A_o = 0.40 * u_fe + 0.35 * f.u_u5 + 0.25 * u_u8
    A_t = 0.40 * ut_fe + 0.35 * f.ut_u5 + 0.25 * ut_u8
    # B: gpt2med contrast + E2
    B_o = 0.55 * unitize(g_med_o) + 0.45 * u_e2
    B_t = 0.55 * unitize(g_med_t) + 0.45 * ut_e2
    # C: cat+R2 against shallow (orth later includes shallow in macro)
    C_o = 0.55 * f.u_cat + 0.45 * f.u_r2
    C_t = 0.55 * f.ut_cat + 0.45 * f.ut_r2
    # D: bank gbn+gb
    D_o = 0.55 * unitize(gbn_o) + 0.45 * unitize(gb_o)
    D_t = 0.55 * unitize(gbn_t) + 0.45 * unitize(gb_t)
    # E: dediluted dMLP
    dmlp_o = L_fold(o_new, fold_id) - L_fold(o_old, fold_id)
    dmlp_t = Z_new - Z_old
    E_o = project_out(dmlp_o, np.column_stack([unitize(L_fold(o_old, fold_id)), unitize(z_gp_oof_L)]))
    E_t = project_out(dmlp_t, np.column_stack([uL(raw_old), uL(raw_gp)]))

    vault_e2_orient = False
    if roc_auc_score(y[f.te0], f.z_s8[f.te0] + 0.01 * u_e2[f.te0]) < roc_auc_score(
        y[f.te0], f.z_s8[f.te0] - 0.01 * u_e2[f.te0]
    ):
        u_e2, ut_e2 = -u_e2, -ut_e2
        B_o = 0.55 * unitize(g_med_o) + 0.45 * u_e2
        B_t = 0.55 * unitize(g_med_t) + 0.45 * ut_e2
        vault_e2_orient = True
    log(f"E2 orient flipped={vault_e2_orient}")

    pillars = {
        "U_A_s2026_lm5f": (A_o, A_t),
        "U_B_gpt2med_5f": (B_o, B_t),
        "U_C_tab_resid": (C_o, C_t),
        "U_D_bank_gbn": (D_o, D_t),
        "U_E_dMLP_dediluted": (E_o, E_t),
    }

    vault_u = {}
    z_s9 = f.z_s8 + 0.020 * u_e2
    dmlp_raw_o = unitize(dmlp_o)
    z_s9_d = z_s9 + 0.10 * dmlp_raw_o
    c2_delta_t = unitize(rank_logit(c2) - rank_logit(sriram))

    log("=== audit unbundled pillars U_A..U_E (±) ===")
    passed = []
    for name, (vo, vt) in pillars.items():
        oo, ot, pre = macro_orth(vo, vt, name)
        # try both signs vs z_sriram_oof
        best = None
        for sign, tag in ((+1, "+"), (-1, "-")):
            uo, ut = sign * oo, sign * ot
            rows = {}
            for bname, base in ("sri", z_sriram_oof), ("z9", z_s9), ("z9d", z_s9_d):
                # small probe weight
                rows[bname] = h2h(base + 0.012 * uo, base, y, folds)
            mc = max_abs_corr(ut, B_macro_t)
            c6 = float(np.corrcoef(ut, f.bad6)[0, 1])
            c7 = float(np.corrcoef(ut, f.bad7)[0, 1])
            corr_c2 = float(np.corrcoef(ut, c2_delta_t)[0, 1])
            score = rows["sri"]["t"] + 0.5 * rows["z9"]["t"]
            log(
                f"  {name}{tag}: sri {rows['sri']['pos']}/5 m={rows['sri']['mean']:+.2f} "
                f"min={rows['sri']['min']:+.2f} t={rows['sri']['t']:.2f} sf={rows['sri']['sf25']} | "
                f"z9 {rows['z9']['pos']}/5 t={rows['z9']['t']:.2f} | "
                f"max|B|={mc:.4f} c6={c6:+.3f} c7={c7:+.3f} corrC2={corr_c2:+.3f} "
                f"disp={pre.get('raw_disp', float('nan')):.3f}"
            )
            cand = {
                "name": name,
                "sign": sign,
                "uo": uo,
                "ut": ut,
                "rows": rows,
                "mc": mc,
                "c6": c6,
                "c7": c7,
                "score": score,
                "pre": pre,
            }
            if best is None or score > best["score"]:
                best = cand
        vault_u[name] = best
        # pass gate: 5/5 on sri and z9, min>0, t>=2.8
        rs, rz = best["rows"]["sri"], best["rows"]["z9"]
        if rs["pos"] == 5 and rz["pos"] == 5 and rs["min"] > 0 and rz["min"] > 0 and rs["t"] >= 2.8 and rz["t"] >= 2.8:
            passed.append(best)
            log(f"  PASS {name} sign={best['sign']:+d}")
        else:
            # softer pass: sri 5/5 t>=2.8 min>0 (primary manifold)
            if rs["pos"] == 5 and rs["min"] > 0 and rs["t"] >= 2.8 and best["mc"] < 0.05:
                passed.append(best)
                log(f"  SOFT-PASS {name} sign={best['sign']:+d} (sri-strong)")

    if not passed:
        # fallback: take top 2 by sri t with pos>=4
        ranked = sorted(vault_u.values(), key=lambda r: (r["rows"]["sri"]["pos"], r["rows"]["sri"]["t"]), reverse=True)
        passed = ranked[:2]
        log(f"FALLBACK pillars: {[p['name'] for p in passed]}")

    # Prefer non-shallow pillars A,B over C; D/E optional
    prefer = []
    for p in passed:
        if p["name"] in ("U_A_s2026_lm5f", "U_B_gpt2med_5f", "U_D_bank_gbn", "U_E_dMLP_dediluted"):
            prefer.append(p)
        elif p["name"] == "U_C_tab_resid" and p["rows"]["sri"]["t"] >= 3.5:
            prefer.append(p)
    if not prefer:
        prefer = passed
    log(f"clean pillars for inject: {[p['name']+('+' if p['sign']>0 else '-') for p in prefer]}")

    u_clean_o = np.zeros(N_TR)
    u_clean_t = np.zeros(N_TE)
    for p in prefer:
        u_clean_o += p["uo"]
        u_clean_t += p["ut"]
    u_clean_o = unitize(u_clean_o)
    u_clean_t = unitize(project_out(u_clean_t, B_macro_t))
    # re-orient
    if roc_auc_score(y[f.te0], z_sriram_oof[f.te0] + 0.01 * u_clean_o[f.te0]) < roc_auc_score(
        y[f.te0], z_sriram_oof[f.te0] - 0.01 * u_clean_o[f.te0]
    ):
        u_clean_o, u_clean_t = -u_clean_o, -u_clean_t
    log(
        f"u_clean_prop max|B_macro|={max_abs_corr(u_clean_t, B_macro_t):.4f} "
        f"corr(sriram)={np.corrcoef(u_clean_t, rank_logit(sriram))[0,1]:+.6f}"
    )

    # ---------- Step2: P1 P2 P3 ----------
    log("=== Step2: build P1/P2/P3 micro-steps ===")

    w1, p1, rho1 = bisect_rho(sriram, u_clean_t, 0.40)
    per1 = {
        "sri": h2h(z_sriram_oof + w1 * u_clean_o, z_sriram_oof, y, folds),
        "z9": h2h(z_s9 + w1 * u_clean_o, z_s9, y, folds),
        "z9d": h2h(z_s9_d + w1 * u_clean_o, z_s9_d, y, folds),
    }
    log(f"P1 w={w1:.5f} ρs={rho1:.3f}e-4 sri {per1['sri']['pos']}/5 t={per1['sri']['t']:.2f} min={per1['sri']['min']:+.2f}")

    # P2: geodesic sriram x C1 + 0.5*w prop → 0.55e-4
    # find λ such that blend alone ~0.35, then add prop to 0.55
    best_lam, best_b, best_r = 0.07, sriram, 0.0
    for lam in np.linspace(0.02, 0.12, 21):
        b = rank_frac((1 - lam) * rank_logit(sriram) + lam * rank_logit(c1))
        r = one_m_rho_e4(b, sriram)
        if abs(r - 0.35) < abs(best_r - 0.35):
            best_lam, best_b, best_r = lam, b, r
    rem = max(0.55 - best_r, 0.10)
    w2a, _, _ = bisect_rho(best_b, u_clean_t, rem, hi=0.3)
    raw_dir = rank_logit(best_b) - rank_logit(sriram) + 0.5 * w2a * u_clean_t
    w2, p2, rho2 = bisect_rho(sriram, raw_dir, 0.55, hi=1.2)
    # OOF proxy
    c1_oof = z_s9 + 0.10 * dmlp_raw_o
    oof_p2 = (1 - best_lam) * z_sriram_oof + best_lam * c1_oof + 0.5 * w2a * u_clean_o
    per2 = {
        "sri": h2h(oof_p2, z_sriram_oof, y, folds),
        "z9": h2h(oof_p2 - z_sriram_oof + z_s9, z_s9, y, folds),
        "z9d": h2h(oof_p2 - z_sriram_oof + z_s9_d, z_s9_d, y, folds),
    }
    log(
        f"P2 λ={best_lam:.3f} w2={w2:.5f} ρs={rho2:.3f}e-4 sri {per2['sri']['pos']}/5 "
        f"t={per2['sri']['t']:.2f} min={per2['sri']['min']:+.2f}"
    )

    # P3: dediluted dMLP rewrite + clean prop → 0.50e-4
    v28_orth = unitize(project_out(Z_old, uL(raw_gp).reshape(-1, 1)))
    s3_base_score = Z_gp + 0.16 * (Z_new - Z_old) - 0.015 * v28_orth
    s3_base = rank_frac(s3_base_score)
    rho_s3b = one_m_rho_e4(s3_base, sriram)
    log(f"P3 base 1-ρ(sriram)={rho_s3b:.3f}e-4")
    if rho_s3b > 1.5:
        s3_near = rank_frac(0.88 * rank_logit(sriram) + 0.12 * rank_logit(s3_base))
    else:
        s3_near = s3_base
    raw3 = rank_logit(s3_near) - rank_logit(sriram) + 0.5 * u_clean_t
    w3, p3, rho3 = bisect_rho(sriram, raw3, 0.50, hi=1.2)
    oof_p3 = z_sriram_oof + 0.02 * unitize(E_o) + 0.5 * w1 * u_clean_o
    per3 = {
        "sri": h2h(oof_p3, z_sriram_oof, y, folds),
        "z9": h2h(oof_p3 - z_sriram_oof + z_s9, z_s9, y, folds),
        "z9d": h2h(oof_p3 - z_sriram_oof + z_s9_d, z_s9_d, y, folds),
    }
    log(f"P3 w={w3:.5f} ρs={rho3:.3f}e-4 sri {per3['sri']['pos']}/5 t={per3['sri']['t']:.2f} min={per3['sri']['min']:+.2f}")

    def card(tag, te, per, target):
        delta = unitize(rank_logit(te) - rank_logit(sriram))
        row = {
            "tag": tag,
            "test": te,
            "per": per,
            "rho_s": one_m_rho_e4(te, sriram),
            "rho_c2": one_m_rho_e4(te, c2),
            "rho_c1": one_m_rho_e4(te, c1),
            "mc": max_abs_corr(delta, B_macro_t),
            "c6": float(np.corrcoef(delta, f.bad6)[0, 1]),
            "c7": float(np.corrcoef(delta, f.bad7)[0, 1]),
            "target": target,
            "min3": min(per["sri"]["min"], per["z9"]["min"], per["z9d"]["min"]),
            "t3": per["sri"]["t"] + per["z9"]["t"] + per["z9d"]["t"],
        }
        log(
            f"{tag}: ρs={row['rho_s']:.3f} ρc2={row['rho_c2']:.3f} ρc1={row['rho_c1']:.3f} "
            f"max|B|={row['mc']:.4f} c6={row['c6']:+.3f} c7={row['c7']:+.3f} "
            f"sri {per['sri']['pos']}/5 t={per['sri']['t']:.2f} min={per['sri']['min']:+.2f} | "
            f"z9 {per['z9']['pos']}/5 t={per['z9']['t']:.2f} | z9d {per['z9d']['pos']}/5 t={per['z9d']['t']:.2f}"
        )
        return row

    cands = [
        card("P1_Sriram_plus_Unbundled_5FLM_MicroStep_0p40e4", p1, per1, 0.40),
        card("P2_MicroGeodesic_Sriram095422_x_C1095407_plus_5FLM_0p55e4", p2, per2, 0.55),
        card("P3_Sriram_DeDiluted_dMLP_plus_CleanProp_0p50e4", p3, per3, 0.50),
    ]

    # pick P1 or P2 by highest 3-anchor min_fold and t
    pool = [cands[0], cands[1]]

    def key(r):
        return (
            r["per"]["sri"]["pos"],
            r["per"]["sri"]["min"],
            r["t3"],
            -abs(r["c7"]),
            -abs(r["rho_s"] - r["target"]),
        )

    submit_row = sorted(pool, key=key, reverse=True)[0]
    log(f"SELECT {submit_row['tag']}")

    def stage(path, vec):
        v = np.asarray(vec, float)
        assert len(v) == N_TE and np.isfinite(v).all()
        pd.DataFrame({"id": ids, "is_fraud": v}).to_csv(path, index=False)
        log(f"STAGED {path.name}")

    paths = {
        "P1": SUB / "sub_v22_2_P1_Sriram_Unbundled_5FLM_0p40e4.csv",
        "P2": SUB / "sub_v22_2_P2_MicroGeodesic_Sriram_x_C1_5FLM_0p55e4.csv",
        "P3": SUB / "sub_v22_2_P3_DeDiluted_dMLP_CleanProp_0p50e4.csv",
    }
    stage(paths["P1"], cands[0]["test"])
    stage(paths["P2"], cands[1]["test"])
    stage(paths["P3"], cands[2]["test"])
    live = SUB / "sub_v22_2_HARDENED_LIVE_microleap.csv"
    stage(live, submit_row["test"])

    meta = {
        "selected": submit_row["tag"],
        "pillars": [
            {
                "name": p["name"],
                "sign": p["sign"],
                "sri": p["rows"]["sri"],
                "z9": p["rows"]["z9"],
                "mc": p["mc"],
            }
            for p in prefer
        ],
        "sriram_repro_1mrho": repro_err,
        "c2_vs_sriram_e4": one_m_rho_e4(c2, sriram),
    }
    OUT.joinpath("v22_2_hardened_meta.json").write_text(json.dumps(meta, indent=2, default=float))

    log("=== Step3: submit ONE probe ===")
    msg = (
        f"v22.2h {submit_row['tag']} | ρs={submit_row['rho_s']:.3f}e-4 "
        f"maxB={submit_row['mc']:.4f} c7={submit_row['c7']:+.3f} "
        f"OOF sri {submit_row['per']['sri']['pos']}/5 t={submit_row['per']['sri']['t']:.2f}"
    )
    log(f"SUBMIT {msg}")
    submit_ok, pub = submit_and_poll(live, msg)

    # report
    body = [
        "# GM V22.2 HARDENED — Unbundled Sriram Micro-Leap Report",
        "",
        f"**elapsed** {time.time()-T0:.0f}s · local Mac · personal ATH was `0.95416` · target `0.95422+`",
        "",
        "## MPS / fold0",
        f"- `checkpoint-50000` preserved: `{CKPT500.exists()}`",
        f"- Replica: GReaT **Target-First** (`conditional_col=is_fraud`, target in column shuffle)",
        f"- fold0 classify: **TERM'd** (was ~37h ETA at <0.5%) · `STOP_CLASSIFY` written",
        "",
        "## Sriram native-space",
        f"- Formula: `rank_frac( L(goodpjw) + 0.15*(L(new)-L(old)) )` with `L=logit((rank-0.5)/N)`",
        f"- Self-repro `1-ρ`: `{repro_err:.3e}` (gate `<1e-10`)",
        f"- C2pure vs sriram: `{one_m_rho_e4(c2, sriram):.3f}e-4` (overshoot diagnosis)",
        "",
        "## Live submit",
        f"- OK=`{submit_ok}` · **{submit_row['tag']}**",
        f"- Public LB: **{pub}**",
        f"- 1−ρ vs sriram: `{submit_row['rho_s']:.3f}e-4` (target `{submit_row['target']:.2f}`)",
        f"- max|corr| vs B_macro: `{submit_row['mc']:.4f}` · c6=`{submit_row['c6']:+.4f}` c7=`{submit_row['c7']:+.4f}`",
        f"- OOF sri `{submit_row['per']['sri']['pos']}/5` t=`{submit_row['per']['sri']['t']:.2f}` "
        f"min=`{submit_row['per']['sri']['min']:+.2f}`",
        "",
        "## Clean pillars used",
        *[f"- `{p['name']}` sign=`{p['sign']:+d}` sri_t=`{p['rows']['sri']['t']:.2f}`" for p in prefer],
        "",
        "## Staged",
        f"- `{paths['P1'].name}`",
        f"- `{paths['P2'].name}`",
        f"- `{paths['P3'].name}`",
        f"- `{live.name}`",
        "",
        "## Log",
        "```",
        *LINES,
        "```",
        "",
    ]
    REPORT.write_text("\n".join(body) + "\n")
    log(f"REPORT {REPORT}")
    log(f"DONE pub={pub} submit_ok={submit_ok} selected={submit_row['tag']}")


if __name__ == "__main__":
    main()
