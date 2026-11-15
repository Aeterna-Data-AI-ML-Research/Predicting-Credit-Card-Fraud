#!/usr/bin/env python3
"""GM V22.2 — Surpass 0.95422 via full-bank SVD-whitened micro-step + MPS canary handoff.

Fixes V22.1b C2pure bugs:
  (1) 4.37e-4 overshoot → bisect to 1-ρ ∈ {0.45, 0.50, 0.60}e-4 trust region
  (2) 3-vector under-orth → full 70-bank + top-30 SVD + anchors + fail-rays
Local Mac CPU only. Submits ONE micro-step probe. Monitors MPS fold0 canary.
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
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
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
SERIES = REPO / "external/banks/goodpjw2008_s6e11-fraud-series-npz"
COMPACT = REPO / "external/banks/goodpjw2008_s6e11-compact-lm-seed7"
SLOT9 = SUB / "sub_v20_LIVE_slot9_overnight_cpu_promoted.csv"
C1_LIVE = SUB / "sub_v22_1_LIVE_leap.csv"  # 0.95407
C2_LIVE = SUB / "sub_v22_1b_LIVE_leap.csv"  # 0.95416
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
REPORT = LOG / "GM_V22_2_FULLBANK_LEAP_AND_MPS_CANARY.md"
MPS_RUN = TMP16 / "lm_research" / "run_mps_gpt2"
N_TR, N_TE = 699960, 299984
W8_O, W8_C, W8_U, W_E2, W_FE = 0.016, 0.012, 0.012, 0.020, 0.01045
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


def one_m_rho_e4(a, b) -> float:
    return (1.0 - float(spearmanr(a, b).correlation)) * 1e4


def max_abs_corr(v: np.ndarray, M: np.ndarray) -> float:
    v = np.asarray(v, float).ravel()
    v = (v - v.mean()) / (v.std() + 1e-12)
    M = np.asarray(M, float)
    Mc = M - M.mean(0, keepdims=True)
    Ms = Mc / (Mc.std(0, keepdims=True) + 1e-12)
    c = Ms.T @ v / len(v)
    return float(np.max(np.abs(c))) if len(c) else 0.0


def project_out(v: np.ndarray, B: np.ndarray) -> np.ndarray:
    """OLS residual of v on columns of B (with intercept)."""
    v = np.asarray(v, float).ravel()
    X = np.column_stack([np.ones(len(v)), np.asarray(B, float)])
    # ridge for stability
    xtx = X.T @ X
    xtx.flat[:: xtx.shape[0] + 1] += 1e-6
    beta = np.linalg.solve(xtx, X.T @ v)
    return v - X @ beta


def unitize_local(v, unitize):
    return unitize(np.asarray(v, float).ravel())


def bisect_to_rho(base, direction, target_e4, U, L, unitize, lo=1e-5, hi=0.35, n=28):
    """Find w>=0 s.t. 1-ρ(U(L(base)+w*dir), base) ≈ target_e4 (e-4 units)."""
    base = np.asarray(base, float).ravel()
    d = unitize(np.asarray(direction, float).ravel())
    # orient: small positive w should increase distance from base
    best_w, best_err = 0.0, 1e9
    a, b = lo, hi
    for _ in range(n):
        mid = 0.5 * (a + b)
        cand = U(L(base) + mid * d)
        rho = one_m_rho_e4(cand, base)
        err = abs(rho - target_e4)
        if err < best_err:
            best_err, best_w = err, mid
        if rho < target_e4:
            a = mid
        else:
            b = mid
    cand = U(L(base) + best_w * d)
    return best_w, cand, one_m_rho_e4(cand, base)


def load_bank_pairs():
    names = []
    for p in sorted(BANK.glob("oof_*.npy")):
        name = p.name[len("oof_") : -len(".npy")]
        tp = BANK / f"test_{name}.npy"
        if tp.exists():
            names.append(name)
    return names


def mps_canary_status() -> dict:
    info = {
        "replica_ckpt": None,
        "replica_final": (MPS_RUN / "replica" / "final").exists(),
        "train_done": False,
        "fold0_alive": False,
        "fold0_step": None,
        "fold0_total": 139992,
        "fold0_eta_h": None,
        "fold0_rows_per_s": None,
        "fold0_auc": None,
        "note": "",
    }
    ckpts = sorted(
        (REPO / "logs/overnight/run_mps_gpt2_data/replica/trainer").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if ckpts:
        info["replica_ckpt"] = int(ckpts[-1].name.split("-")[-1])
    prog_rep = MPS_RUN / "replica" / "progress.jsonl"
    if prog_rep.exists():
        for ln in reversed(prog_rep.read_text().strip().splitlines()):
            try:
                j = json.loads(ln)
            except Exception:
                continue
            if j.get("status") == "done" or j.get("global_step") == 50000:
                info["train_done"] = True
                break
            if j.get("step") == 50000:
                info["train_done"] = True
                break
    # classify progress
    prog_f0 = MPS_RUN / "fits" / "f0" / "progress.jsonl"
    if prog_f0.exists():
        last = None
        for ln in prog_f0.read_text().strip().splitlines():
            try:
                last = json.loads(ln)
            except Exception:
                continue
        if last:
            info["fold0_step"] = last.get("step")
            info["fold0_total"] = last.get("total_steps", 139992)
            info["fold0_eta_h"] = last.get("eta_hours")
            info["fold0_rows_per_s"] = last.get("rows_per_second")
            if "auc" in last:
                info["fold0_auc"] = last.get("auc")
    # process alive?
    try:
        r = subprocess.run(["pgrep", "-f", "train_classifier.py.*fold 0"], capture_output=True, text=True)
        info["fold0_alive"] = bool(r.stdout.strip())
    except Exception:
        pass
    # early auc file
    early = MPS_RUN / "early_f0"
    if early.exists():
        info["note"] = f"early_f0 exists: {list(early.iterdir())[:5]}"
        for p in early.rglob("*"):
            if "auc" in p.name.lower() or p.suffix == ".json":
                try:
                    info["fold0_auc_file"] = p.read_text()[:200]
                except Exception:
                    pass
    info["pipeline_tail"] = ""
    plog = MPS_RUN / "pipeline.log"
    if plog.exists():
        info["pipeline_tail"] = "\n".join(plog.read_text().strip().splitlines()[-8:])
    return info


def submit_and_poll(path: Path, msg: str, match_token: str):
    pub = None
    submit_ok = False
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
        submit_ok = r.returncode == 0 and "400" not in (r.stdout + r.stderr) and "limit" not in blob
        if submit_ok:
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
                    if match_token in ln or "v22.2" in ln or "v22_2" in ln:
                        log(f"poll {ln[:240]}")
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
    return submit_ok, pub


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    v20 = load_v20()
    L, U, unitize, qlock = v20.L, v20.U, v20.unitize, v20.qlock
    h2h = v20.h2h

    log("=== V22.2 START: rebuild Slot8/9 proprietary anchors ===")
    fac = v20.OvernightFactory()
    fac.rebuild()
    f = fac
    ids, y, folds = f.ids, f.y, f.folds

    def load_sub(path):
        df = pd.read_csv(path)
        return df.set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    slot9 = load_sub(SLOT9)
    c1 = load_sub(C1_LIVE)
    c2 = load_sub(C2_LIVE)
    goodpjw = load_sub(SRIRAM_DS / "reference.csv")
    new_mlp_t = load_sub(SRIRAM_DS / "new_neural.csv")
    old_mlp_t = load_sub(SRIRAM_DS / "old_neural.csv")
    sriram = U(L(goodpjw) + 0.15 * (L(new_mlp_t) - L(old_mlp_t)))
    log(
        f"anchors: sriram vs goodpjw 1-ρ={one_m_rho_e4(sriram, goodpjw):.3f}e-4 | "
        f"C2 vs sriram={one_m_rho_e4(c2, sriram):.3f}e-4 | C1 vs sriram={one_m_rho_e4(c1, sriram):.3f}e-4"
    )

    o_new = pd.read_csv(PYTAB_OOF).is_fraud.to_numpy(float)
    o_old = pd.read_csv(PYTORCH_OOF).is_fraud.to_numpy(float)
    assert len(o_new) == N_TR and len(o_old) == N_TR

    # ---------- Step1: full bank + SVD ----------
    log("=== Step1: load 70-bank + build SVD-30 + sriram OOF proxy ===")
    names = load_bank_pairs()
    log(f"bank pairs={len(names)}")
    Bo_cols, Bt_cols, col_names = [], [], []
    for name in names:
        vo = np.load(BANK / f"oof_{name}.npy").ravel()[:N_TR]
        vt = np.load(BANK / f"test_{name}.npy").ravel()[:N_TE]
        if not (np.isfinite(vo).all() and np.isfinite(vt).all()):
            continue
        Bo_cols.append(unitize(L(vo)))
        Bt_cols.append(unitize(L(vt)))
        col_names.append(name)
    # extras: compact / series if present
    for label, path_o, path_t in [
        ("compact_lm7_f0", COMPACT / "lm7_oof_f0.npy", COMPACT / "lm7_test_f0.npy"),
    ]:
        if path_o.exists() and path_t.exists():
            vo, vt = np.load(path_o).ravel(), np.load(path_t).ravel()
            if len(vo) >= N_TR and len(vt) >= N_TE:
                Bo_cols.append(unitize(L(vo[:N_TR])))
                Bt_cols.append(unitize(L(vt[:N_TE])))
                col_names.append(label)
    # RealMLP + Bo9 + anchors
    Bo_cols += [unitize(L(o_new)), unitize(L(o_old))]
    Bt_cols += [unitize(L(new_mlp_t)), unitize(L(old_mlp_t))]
    col_names += ["RealMLP_v5", "RealMLP_v28"]
    for j in range(f.Bo9.shape[1]):
        Bo_cols.append(unitize(f.Bo9[:, j]))
        Bt_cols.append(unitize(f.Bt9[:, j]))
        col_names.append(f"Bo9_{j}")
    Bo_bank = np.column_stack(Bo_cols).astype(np.float64)
    Bt_bank = np.column_stack(Bt_cols).astype(np.float64)
    log(f"Bo_bank shape={Bo_bank.shape} Bt_bank={Bt_bank.shape}")

    # Ridge surrogate: predict L(goodpjw), L(sriram) on test → apply to OOF
    lr_gp = Ridge(alpha=10.0, fit_intercept=True).fit(Bt_bank, L(goodpjw))
    lr_sr = Ridge(alpha=10.0, fit_intercept=True).fit(Bt_bank, L(sriram))
    pred_gp_t = lr_gp.predict(Bt_bank)
    pred_sr_t = lr_sr.predict(Bt_bank)
    r2_gp = 1.0 - float(np.sum((L(goodpjw) - pred_gp_t) ** 2)) / (float(np.var(L(goodpjw)) * N_TE) + 1e-18)
    r2_sr = 1.0 - float(np.sum((L(sriram) - pred_sr_t) ** 2)) / (float(np.var(L(sriram)) * N_TE) + 1e-18)
    z_gp_oof = Bo_bank @ lr_gp.coef_ + lr_gp.intercept_
    z_sr_oof = Bo_bank @ lr_sr.coef_ + lr_sr.intercept_
    # calibrate to score-like scale for AUC / H2H
    z_sriram_oof = unitize(z_sr_oof) * float(np.std(f.z_s8)) + float(np.mean(f.z_s8))
    log(
        f"surrogate R²_test goodpjw={r2_gp:.4f} sriram={r2_sr:.4f} | "
        f"z_sriram_oof AUC={roc_auc_score(y, z_sriram_oof):.6f}"
    )

    # Top-30 SVD of centered bank (OOF), transfer loadings to test
    n_pc = 30
    Bo_c = Bo_bank - Bo_bank.mean(0, keepdims=True)
    Bt_c = Bt_bank - Bt_bank.mean(0, keepdims=True)
    svd = TruncatedSVD(n_components=n_pc, random_state=42)
    Uo30 = svd.fit_transform(Bo_c)  # (N_TR, 30)
    # project test with same components
    Ut30 = Bt_c @ svd.components_.T  # (N_TE, 30)
    # ZCA-ish unitize each PC
    Uo30 = np.column_stack([unitize(Uo30[:, j]) for j in range(n_pc)])
    Ut30 = np.column_stack([unitize(Ut30[:, j]) for j in range(n_pc)])
    log(f"SVD-30 explained={svd.explained_variance_ratio_.sum():.4f}")

    # Full orth basis: PCs + key anchors + fail rays
    Bo_orth = np.column_stack(
        [
            Uo30,
            unitize(L(o_new)),
            unitize(L(o_old)),
            unitize(z_gp_oof),
            unitize(z_sr_oof),
            f.bad6[:N_TR] * 0 + 0,  # placeholder replaced below for OOF rays
        ]
    )
    # fail rays are test-only in factory; build OOF proxies via ridge from Bo_bank
    lr6 = Ridge(alpha=10.0, fit_intercept=True).fit(Bt_bank, f.bad6)
    lr7 = Ridge(alpha=10.0, fit_intercept=True).fit(Bt_bank, f.bad7)
    bad6_o = unitize(Bo_bank @ lr6.coef_ + lr6.intercept_)
    bad7_o = unitize(Bo_bank @ lr7.coef_ + lr7.intercept_)
    Bo_full = np.column_stack(
        [
            Uo30,
            unitize(L(o_new)),
            unitize(L(o_old)),
            unitize(z_gp_oof),
            unitize(z_sr_oof),
            bad6_o,
            bad7_o,
            f.Bo9,
        ]
    ).astype(np.float64)
    Bt_full = np.column_stack(
        [
            Ut30,
            unitize(L(new_mlp_t)),
            unitize(L(old_mlp_t)),
            unitize(L(goodpjw)),
            unitize(L(sriram)),
            f.bad6,
            f.bad7,
            f.Bt9,
        ]
    ).astype(np.float64)
    log(f"full orth basis Bo={Bo_full.shape} Bt={Bt_full.shape}")

    # ---------- proprietary engines ----------
    vault = np.load(OUT / "gm_overnight_vault.npz")
    u_e2, ut_e2 = unitize(vault["E2_heavy_ridge_oof"]), unitize(vault["E2_heavy_ridge_test"])
    if roc_auc_score(y[f.te0], f.z_s8[f.te0] + 0.01 * u_e2[f.te0]) < roc_auc_score(
        y[f.te0], f.z_s8[f.te0] - 0.01 * u_e2[f.te0]
    ):
        u_e2, ut_e2 = -u_e2, -ut_e2
    u_fe, ut_fe = unitize(f.dE_o), unitize(f.dE_t)
    prop_o = W8_O * f.u_r2 + W8_C * f.u_cat + W8_U * f.u_u5 + W_E2 * u_e2 + W_FE * u_fe
    prop_t = W8_O * f.ut_r2 + W8_C * f.ut_cat + W8_U * f.ut_u5 + W_E2 * ut_e2 + W_FE * ut_fe

    # Full-bank orth of prop: SVD/anchors first, then raw 70-bank columns
    def full_orth(vo, vt):
        oo = project_out(vo, Bo_full)
        ot = project_out(vt, Bt_full)
        oo = project_out(oo, Bo_bank)
        ot = project_out(ot, Bt_bank)
        # also kill residual correlation with live anchors explicitly
        oo = project_out(oo, np.column_stack([unitize(z_sr_oof), unitize(z_gp_oof), unitize(L(o_new)), unitize(L(o_old))]))
        ot = project_out(
            ot,
            np.column_stack([unitize(L(sriram)), unitize(L(goodpjw)), unitize(L(new_mlp_t)), unitize(L(old_mlp_t))]),
        )
        oo, ot = unitize(oo), unitize(ot)
        if roc_auc_score(y[f.te0], z_sriram_oof[f.te0] + 0.01 * oo[f.te0]) < roc_auc_score(
            y[f.te0], z_sriram_oof[f.te0] - 0.01 * oo[f.te0]
        ):
            oo, ot = -oo, -ot
        return oo, ot

    u_prop_o, u_prop_t = full_orth(prop_o, prop_t)
    maxc = max_abs_corr(u_prop_t, Bt_bank)
    c_sr = float(np.corrcoef(u_prop_t, L(sriram))[0, 1])
    c6 = float(np.corrcoef(u_prop_t, f.bad6)[0, 1])
    c7 = float(np.corrcoef(u_prop_t, f.bad7)[0, 1])
    log(
        f"u_prop_fullorth: max|corr|vs_bank={maxc:.4f} corr(sriram)={c_sr:+.6f} "
        f"c6={c6:+.4f} c7={c7:+.4f}"
    )

    # OOF H2H of micro inject on sriram proxy + z_S9
    z_s9 = f.z_s8 + W_E2 * u_e2
    rows_w = []
    for w in (0.008, 0.012, 0.016, 0.020, 0.028):
        per_s = h2h(z_sriram_oof + w * u_prop_o, z_sriram_oof, y, folds)
        per_z = h2h(z_s9 + w * u_prop_o, z_s9, y, folds)
        rows_w.append((w, per_s, per_z))
        log(
            f"  prop_orth w={w:.3f}: vs_sriProx {per_s['pos']}/5 m={per_s['mean']:+.2f} "
            f"min={per_s['min']:+.2f} t={per_s['t']:.2f} | vs_z9 {per_z['pos']}/5 m={per_z['mean']:+.2f}"
        )

    # ---------- Step2: S1 / S2 / S3 ----------
    log("=== Step2: build S1/S2/S3 micro-step candidates ===")

    # S1: sriram + w*prop bisected to 0.45e-4
    w1, s1, rho1 = bisect_to_rho(sriram, u_prop_t, 0.45, U, L, unitize)
    per_s1 = h2h(z_sriram_oof + w1 * u_prop_o, z_sriram_oof, y, folds)
    log(f"S1: w={w1:.5f} 1-ρ(sriram)={rho1:.3f}e-4 vsProx {per_s1['pos']}/5 m={per_s1['mean']:+.2f} t={per_s1['t']:.2f}")

    # S2: 0.93*L(sriram)+0.07*L(C1) + w2*prop → 0.60e-4 vs sriram
    bridge = U(0.93 * L(sriram) + 0.07 * L(c1))
    rho_bridge = one_m_rho_e4(bridge, sriram)
    log(f"S2 bridge alone 1-ρ(sriram)={rho_bridge:.3f}e-4")
    # residual room to target 0.60
    # if bridge already near/over target, shrink λ
    if rho_bridge > 0.60:
        # re-bisect λ for geodesic only
        lo, hi = 0.0, 0.07
        best_lam, best_b = 0.07, bridge
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            b = U((1 - mid) * L(sriram) + mid * L(c1))
            r = one_m_rho_e4(b, sriram)
            if r < 0.35:
                lo = mid
            else:
                hi = mid
            if abs(r - 0.45) < abs(one_m_rho_e4(best_b, sriram) - 0.45):
                best_lam, best_b = mid, b
        bridge = best_b
        rho_bridge = one_m_rho_e4(bridge, sriram)
        log(f"S2 bridge retuned λ={best_lam:.4f} 1-ρ={rho_bridge:.3f}e-4")
    # add prop to reach ~0.60e-4
    # direction from bridge
    rem = max(0.60 - rho_bridge, 0.10)
    w2, s2_prop, rho2p = bisect_to_rho(bridge, u_prop_t, rem, U, L, unitize, hi=0.25)
    # final: recompose as blend+prop then re-bisect whole vs sriram to exact 0.60
    raw_s2_dir = L(bridge) - L(sriram) + w2 * u_prop_t
    w2f, s2, rho2 = bisect_to_rho(sriram, raw_s2_dir, 0.60, U, L, unitize, hi=1.5)
    # OOF proxy for S2
    oof_bridge = (0.93 * z_sriram_oof + 0.07 * (unitize(L(c1)) * 0 + z_s9))  # use z_s9 as C1 oof proxy
    # better C1 oof proxy: z_s9 + 0.10 * unitize(dmlp_o) raw
    dmlp_o = L(o_new) - L(o_old)
    c1_oof = z_s9 + 0.10 * unitize(dmlp_o)
    oof_s2 = 0.93 * z_sriram_oof + 0.07 * c1_oof + w2 * u_prop_o
    per_s2 = h2h(oof_s2, z_sriram_oof, y, folds)
    log(
        f"S2: w_prop≈{w2:.5f} w_final={w2f:.5f} 1-ρ(sriram)={rho2:.3f}e-4 "
        f"vsProx {per_s2['pos']}/5 m={per_s2['mean']:+.2f} t={per_s2['t']:.2f}"
    )

    # S3: de-diluted RealMLP + fullorth prop at ~0.50e-4
    dmlp_t = L(new_mlp_t) - L(old_mlp_t)
    # orth(Δ | [L(v28), L(goodpjw)])
    B_dd = np.column_stack([unitize(L(old_mlp_t)), unitize(L(goodpjw))])
    dmlp_clean_t = unitize(project_out(dmlp_t, B_dd))
    # OOF analog
    B_dd_o = np.column_stack([unitize(L(o_old)), unitize(z_gp_oof)])
    dmlp_clean_o = unitize(project_out(dmlp_o, B_dd_o))
    # base S3 without prop: goodpjw + 0.165*Δ - 0.015*orth(v28|gp)
    v28_orth_t = unitize(project_out(L(old_mlp_t), unitize(L(goodpjw)).reshape(-1, 1)))
    s3_base = U(L(goodpjw) + 0.165 * dmlp_t - 0.015 * v28_orth_t)
    rho_s3b = one_m_rho_e4(s3_base, sriram)
    log(f"S3 base (no prop) 1-ρ(sriram)={rho_s3b:.3f}e-4")
    # move toward sriram manifold if far, then add prop to 0.50
    if rho_s3b > 2.0:
        # blend s3_base toward sriram first
        s3_near = U(0.85 * L(sriram) + 0.15 * L(s3_base))
    else:
        s3_near = s3_base
    raw_s3_dir = L(s3_near) - L(sriram) + 0.02 * u_prop_t
    w3, s3, rho3 = bisect_to_rho(sriram, raw_s3_dir, 0.50, U, L, unitize, hi=1.5)
    oof_s3 = z_sriram_oof + 0.02 * u_prop_o + 0.02 * dmlp_clean_o
    per_s3 = h2h(oof_s3, z_sriram_oof, y, folds)
    log(
        f"S3: w={w3:.5f} 1-ρ(sriram)={rho3:.3f}e-4 vsProx {per_s3['pos']}/5 "
        f"m={per_s3['mean']:+.2f} t={per_s3['t']:.2f}"
    )

    def scorecard(tag, te, per, target_rho):
        rho_s = one_m_rho_e4(te, sriram)
        rho_c2 = one_m_rho_e4(te, c2)
        rho_c1 = one_m_rho_e4(te, c1)
        delta = unitize(L(te) - L(sriram))
        cc6 = float(np.corrcoef(delta, f.bad6)[0, 1])
        cc7 = float(np.corrcoef(delta, f.bad7)[0, 1])
        mc = max_abs_corr(delta, Bt_bank)
        row = {
            "tag": tag,
            "test": te,
            "per": per,
            "rho_s": rho_s,
            "rho_c2": rho_c2,
            "rho_c1": rho_c1,
            "c6": cc6,
            "c7": cc7,
            "maxc_bank": mc,
            "target": target_rho,
            "t": per["t"],
            "pos": per["pos"],
            "min": per["min"],
            "mean": per["mean"],
        }
        log(
            f"{tag}: ρs={rho_s:.3f} (tgt {target_rho:.2f}) ρc2={rho_c2:.3f} ρc1={rho_c1:.3f} "
            f"max|bank|={mc:.4f} c6={cc6:+.3f} c7={cc7:+.3f} "
            f"OOF {per['pos']}/5 m={per['mean']:+.2f} min={per['min']:+.2f} t={per['t']:.2f}"
        )
        return row

    cands = [
        scorecard("S1_FullBankOrth_Prop_0p45e4", s1, per_s1, 0.45),
        scorecard("S2_MicroGeodesic_Sriram_x_C1_plus_FullOrthProp_0p60e4", s2, per_s2, 0.60),
        scorecard("S3_DeDiluted_RealMLP_plus_FullOrthProp_0p50e4", s3, per_s3, 0.50),
    ]

    # pick submit: higher t among S1/S2 with pos==5 and |c7|<0.15; else best t
    def pick_key(r):
        return (r["pos"] == 5, r["t"], r["min"], -abs(r["c7"]), -abs(r["rho_s"] - r["target"]))

    s1r, s2r, s3r = cands
    # user prefers S2 or S1 by t_stat / 5/5
    pool = [s1r, s2r]
    submit_row = sorted(pool, key=pick_key, reverse=True)[0]
    # if both weak, still take best of S1/S2
    log(f"SELECT {submit_row['tag']} t={submit_row['t']:.2f} pos={submit_row['pos']}/5")

    # stage
    def stage(path, vec):
        pd.DataFrame({"id": ids, "is_fraud": np.asarray(vec, float)}).to_csv(path, index=False)
        log(f"STAGED {path.name} n={len(vec)} nan={np.isnan(vec).sum()}")

    p1 = SUB / "sub_v22_2_S1_FullBankOrth_Prop_0p45e4.csv"
    p2 = SUB / "sub_v22_2_S2_MicroGeodesic_Sriram_x_C1_FullOrthProp_0p60e4.csv"
    p3 = SUB / "sub_v22_2_S3_DeDiluted_RealMLP_FullOrthProp_0p50e4.csv"
    stage(p1, s1r["test"])
    stage(p2, s2r["test"])
    stage(p3, s3r["test"])
    out_live = SUB / "sub_v22_2_LIVE_microleap.csv"
    stage(out_live, submit_row["test"])

    meta = {
        "selected": submit_row["tag"],
        "prop_max_bank_corr": maxc,
        "prop_corr_sriram": c_sr,
        "prop_c6": c6,
        "prop_c7": c7,
        "surrogate_r2": {"goodpjw": r2_gp, "sriram": r2_sr},
        "candidates": [
            {k: (float(v) if isinstance(v, (float, np.floating, int)) else v) for k, v in r.items() if k != "test"}
            for r in cands
        ],
    }
    OUT.joinpath("v22_2_candidates_meta.json").write_text(json.dumps(meta, indent=2, default=float))

    # ---------- Step3: submit ONE ----------
    log("=== Step3: submit ONE micro-step probe ===")
    msg = (
        f"v22.2 {submit_row['tag']} | ρs={submit_row['rho_s']:.3f}e-4 "
        f"maxBank={submit_row['maxc_bank']:.4f} c7={submit_row['c7']:+.3f} "
        f"OOF {submit_row['pos']}/5 t={submit_row['t']:.2f}"
    )
    log(f"SUBMIT {msg}")
    submit_ok, pub = submit_and_poll(out_live, msg, "v22_2_LIVE_microleap")

    # ---------- MPS canary ----------
    log("=== MPS replica / fold0 canary status ===")
    mps = mps_canary_status()
    log(
        f"MPS train_done={mps['train_done']} final={mps['replica_final']} ckpt={mps['replica_ckpt']} | "
        f"fold0_alive={mps['fold0_alive']} step={mps['fold0_step']}/{mps['fold0_total']} "
        f"eta_h={mps['fold0_eta_h']} rps={mps['fold0_rows_per_s']}"
    )
    # ensure fold0 is running if train done
    if mps["train_done"] and not mps["fold0_alive"] and mps["fold0_step"] is None:
        log("fold0 not running — launching classify handoff")
        cmd = (
            f"nohup /tmp/s6e11_v15/venv/bin/python -u "
            f"/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/scripts/../ 2>/dev/null; "
            f"cd {MPS_RUN} && nohup bash {REPO}/scripts/gm_v22_mps_full_gpt2_replica.sh "
            f">> {MPS_RUN}/pipeline.log 2>&1 &"
        )
        # safer: invoke classify directly like the shell script
        cls = REPO / "scripts"
        # find train_classifier path from pipeline
        # Use the same command as pipeline
        launch = [
            "/tmp/s6e11_v15/venv/bin/python",
            "-u",
            str(MPS_RUN / "train_classifier.py")
            if (MPS_RUN / "train_classifier.py").exists()
            else str(REPO / "lm_research" / "train_classifier.py"),
        ]
        # discover classifier script
        candidates = list(Path("/tmp/s6e11_v16/lm_research").rglob("train_classifier.py"))
        if not candidates:
            candidates = list(REPO.rglob("train_classifier.py"))
        if candidates and mps["replica_final"]:
            py = candidates[0]
            args = [
                "/tmp/s6e11_v15/venv/bin/python",
                "-u",
                str(py),
                "--train-csv",
                str(REPO / "data" / "train.csv"),
                "--fold-ids",
                str(MPS_RUN / "fold_ids.npy"),
                "--fold",
                "0",
                "--id-col",
                "id",
                "--target-col",
                "is_fraud",
                "--replica",
                str(MPS_RUN / "replica" / "final"),
                "--out",
                str(MPS_RUN / "fits" / "f0"),
                "--epochs",
                "2",
                "--batch-size",
                "8",
                "--precision",
                "bf16",
                "--device",
                "mps",
                "--seed",
                "42",
                "--save-every-steps",
                "500",
                "--log-every-steps",
                "50",
            ]
            log(f"launch classify: {py}")
            subprocess.Popen(
                args,
                stdout=open(MPS_RUN / "pipeline.log", "a"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(3)
            mps = mps_canary_status()
            log(f"recheck fold0_alive={mps['fold0_alive']} step={mps['fold0_step']}")
        else:
            log("classify script or final replica missing — pipeline/watchdog should resume")
    elif mps["fold0_alive"]:
        log("fold0 canary already running — handoff OK (do not kill)")

    # Report
    body = [
        "# GM V22.2 Full-Bank Whitened Leap + MPS Canary",
        "",
        f"**elapsed** {time.time()-T0:.0f}s · local Mac · prior ATH `0.95416` · target beat `0.95422`",
        "",
        "## Live submit",
        f"- Submit OK: `{submit_ok}` · chosen: `{submit_row['tag']}`",
        f"- File: `{out_live.name}`",
        f"- Public LB: **{pub}**",
        f"- 1−ρ vs sriram: `{submit_row['rho_s']:.3f}e-4` (target `{submit_row['target']:.2f}`)",
        f"- max|corr| vs 70-bank (delta): `{submit_row['maxc_bank']:.4f}`",
        f"- Fail rays: c6=`{submit_row['c6']:+.4f}` c7=`{submit_row['c7']:+.4f}`",
        f"- OOF vs sriram proxy: `{submit_row['pos']}/5` mean=`{submit_row['mean']:+.2f}` "
        f"min=`{submit_row['min']:+.2f}` t=`{submit_row['t']:.2f}`",
        "",
        "## Prop full-bank orth diagnostics",
        f"- max|corr| vs Bt_bank: `{maxc:.4f}` (gate <0.005 ideal)",
        f"- corr vs L(sriram): `{c_sr:+.6f}`",
        f"- c6=`{c6:+.4f}` c7=`{c7:+.4f}`",
        f"- Surrogate R² test: goodpjw=`{r2_gp:.4f}` sriram=`{r2_sr:.4f}`",
        "",
        "## Staged",
        f"- `{p1.name}`",
        f"- `{p2.name}`",
        f"- `{p3.name}`",
        "",
        "## MPS canary",
        f"- train_done=`{mps['train_done']}` final=`{mps['replica_final']}` ckpt=`{mps['replica_ckpt']}`",
        f"- fold0_alive=`{mps['fold0_alive']}` step=`{mps['fold0_step']}/{mps['fold0_total']}` "
        f"eta_h=`{mps['fold0_eta_h']}` rps=`{mps['fold0_rows_per_s']}`",
        f"- gate: fold0 AUC ≥ `0.9505` vs masaya f0 `0.94888` (pending)",
        "",
        "```",
        mps.get("pipeline_tail", ""),
        "```",
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
