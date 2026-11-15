#!/usr/bin/env python3
"""GM V22.3 — Ride 0.95425 ridge: purify U_B (t=5.95), micro-continue, replica canary probe.

Local Mac only. Submit ONE continuation (R1 or R2). Keep 6 daily slots.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
P2_ATH = SUB / "sub_v22_2_HARDENED_LIVE_microleap.csv"
P1_CSV = SUB / "sub_v22_2_P1_Sriram_Unbundled_5FLM_0p40e4.csv"
P3_CSV = SUB / "sub_v22_2_P3_DeDiluted_dMLP_CleanProp_0p50e4.csv"
C1 = SUB / "sub_v22_1_LIVE_leap.csv"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
REPORT = LOG / "GM_V22_3_RIDGE_CONTINUATION_REPORT.md"
MPS_RUN = TMP16 / "lm_research" / "run_mps_gpt2"
MAT = TMP16 / "lm_research" / "matapanino"
PRED = TMP16 / "preds" if (TMP16 / "preds").exists() else REPO / "preds"
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


def bisect_rho(base, direction, target_e4, lo=1e-6, hi=0.5, n=30):
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


def bisect_rho_vs(ref, base_score_L, direction, target_e4_vs_ref, lo=1e-6, hi=1.5, n=32):
    """Find w s.t. 1-ρ(rank_frac(base_L + w*d), ref) ≈ target."""
    ref = np.asarray(ref, float).ravel()
    d = np.asarray(direction, float).ravel()
    d = (d - d.mean()) / (d.std() + 1e-12)
    base_L = np.asarray(base_score_L, float).ravel()
    best_w, best_err, best_c = 0.0, 1e9, rank_frac(base_L)
    a, b = lo, hi
    for _ in range(n):
        mid = 0.5 * (a + b)
        cand = rank_frac(base_L + mid * d)
        rho = one_m_rho_e4(cand, ref)
        err = abs(rho - target_e4_vs_ref)
        if err < best_err:
            best_err, best_w, best_c = err, mid, cand
        if rho < target_e4_vs_ref:
            a = mid
        else:
            b = mid
    return best_w, best_c, one_m_rho_e4(best_c, ref)


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
                    if "v22_3" in ln or "v22.3" in ln or "RIDGE" in ln or "purified_UB" in ln:
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


def replica_canary_probe(fold_id, y, timeout_s=60):
    """45–60s zero-shot ll_pos-ll_neg probe on replica/final (diagnostic only)."""
    info = {"ok": False, "auc_f0": None, "n": 0, "seconds": 0.0, "note": "", "corr_ms": None, "corr_g26": None}
    replica = MPS_RUN / "replica" / "final"
    if not (replica / "model.safetensors").exists():
        info["note"] = "replica/final missing"
        return info
    t_probe = time.time()
    try:
        sys.path.insert(0, str(MAT))
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tabtext import RowTokens, feature_columns, label_prefix_ids, read_csv_as_strings

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        use_bf16 = device == "mps"
        log(f"replica probe device={device}")

        # stratified 1000+1000 from fold0
        f0 = np.where(fold_id == 0)[0]
        y0 = y[f0]
        pos = f0[y0 == 1]
        neg = f0[y0 == 0]
        rng = np.random.default_rng(42)
        n_pos = min(1000, len(pos))
        n_neg = min(1000, len(neg))
        # if not enough positives in fold0 (~few thousand), take all pos + fill neg
        sel_pos = rng.choice(pos, size=n_pos, replace=False)
        sel_neg = rng.choice(neg, size=n_neg, replace=False)
        selected = np.concatenate([sel_pos, sel_neg])
        rng.shuffle(selected)
        info["n"] = int(len(selected))

        train_csv = REPO / "data" / "train.csv"
        df = pd.read_csv(train_csv)
        # write tiny csv for tokenization alignment
        probe_csv = OUT / "v22_3_replica_probe_sample.csv"
        df.iloc[selected].to_csv(probe_csv, index=False)

        # Load model
        tok = AutoTokenizer.from_pretrained(str(replica))
        model = AutoModelForCausalLM.from_pretrained(str(replica))
        model.to(device)
        model.eval()

        strings = read_csv_as_strings(probe_csv)
        cols = feature_columns(strings, id_col="id", target_col="is_fraud")
        tokens = RowTokens(strings, cols, tok)
        prefix_true, prefix_false = label_prefix_ids(tok, "is_fraud", "truefalse")

        # K=1 canonical order = sorted column index order
        n_cols = len(cols)
        canon = np.arange(n_cols, dtype=np.int64)
        orders = np.tile(canon, (len(selected), 1, 1))  # rows x 1 x cols — fix shape
        orders = np.broadcast_to(canon, (len(selected), 1, n_cols)).copy()

        from score_rows import score_chunk

        # score in batches with wall clock cap
        rows = np.arange(len(selected))
        # score_chunk expects rows as indices into tokens (=0..n-1 for our probe csv)
        result = None
        deadline = t_probe + timeout_s

        def should_stop():
            return time.time() > deadline

        result = score_chunk(
            model,
            tokens,
            rows,
            orders,
            prefix_true,
            prefix_false,
            batch_sequences=64,
            use_bf16=use_bf16,
            device=device,
            should_stop=should_stop,
            layout="first",
        )
        if result is None:
            info["note"] = "timeout/stop before complete"
            info["seconds"] = time.time() - t_probe
            return info
        log_p_true, log_p_false = result
        score = (log_p_true - log_p_false).ravel()
        y_sel = y[selected]
        auc = float(roc_auc_score(y_sel, score))
        info["auc_f0"] = auc
        info["ok"] = True
        info["seconds"] = time.time() - t_probe
        log(f"replica probe fold0 stratified n={info['n']} AUC={auc:.4f} in {info['seconds']:.1f}s")

        # test slice corr vs MS / G26 (2000 rows) — use random test ids, score subset if time left
        # Skip heavy rescoring; instead report that probe is train-fold diagnostic only unless time
        ms_path = None
        for cand in [PRED / "P_super_v4_test.npy", TMP16 / "preds" / "P_super_v4_test.npy"]:
            if cand.exists():
                ms_path = cand
                break
        g26_path = TMP16 / "g26" / "gpt2_s2026_test_f0.npy"
        info["note"] = (
            f"diagnostic only (replica saw train/orig20k in GReaT fit); "
            f"ms_exists={ms_path is not None} g26_exists={g26_path.exists()}; "
            f"NOT folded into live submit"
        )
        # quick: if leftover time > 20s, score 500 test rows for corr — optional light
        rem = timeout_s - (time.time() - t_probe)
        if rem > 25 and ms_path is not None:
            test_csv = REPO / "data" / "test.csv"
            tdf = pd.read_csv(test_csv)
            tsel = rng.choice(len(tdf), size=min(500, len(tdf)), replace=False)
            tprobe = OUT / "v22_3_replica_probe_test500.csv"
            tdf.iloc[tsel].to_csv(tprobe, index=False)
            tstrings = read_csv_as_strings(tprobe)
            tcols = feature_columns(tstrings, id_col="id", target_col="is_fraud")
            # test has no is_fraud — feature_columns may differ; if fails, skip
            try:
                ttokens = RowTokens(tstrings, tcols, tok)
                # For test without labels, score_rows layout first still prefixes True/False
                torders = np.broadcast_to(np.arange(len(tcols)), (len(tsel), 1, len(tcols))).copy()
                tres = score_chunk(
                    model,
                    ttokens,
                    np.arange(len(tsel)),
                    torders,
                    prefix_true,
                    prefix_false,
                    64,
                    use_bf16,
                    device,
                    should_stop=lambda: time.time() > t_probe + timeout_s,
                )
                if tres is not None:
                    ts = (tres[0] - tres[1]).ravel()
                    ms = np.load(ms_path).ravel()[:N_TE][tsel]
                    info["corr_ms"] = float(spearmanr(ts, ms).correlation)
                    if g26_path.exists():
                        g26 = np.load(g26_path).ravel()[:N_TE][tsel]
                        info["corr_g26"] = float(spearmanr(ts, g26).correlation)
                    log(f"replica test500 corr MS={info['corr_ms']:+.3f} G26={info['corr_g26']}")
            except Exception as e:
                info["note"] += f"; test_corr_skip={e}"
        info["seconds"] = time.time() - t_probe
    except Exception as e:
        info["note"] = f"probe_error: {e}"
        info["seconds"] = time.time() - t_probe
        log(f"replica probe failed: {e}")
        traceback.print_exc()
    return info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    v20 = load_v20()
    unitize, h2h, frozen_lofo_orth, L_fold = v20.unitize, v20.h2h, v20.frozen_lofo_orth, v20.L_fold

    log("=== V22.3 START: rebuild anchors + purify U_B ridge ===")
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
    p2 = load_sub(P2_ATH)
    c1 = load_sub(C1)
    p1 = load_sub(P1_CSV)
    p3 = load_sub(P3_CSV)
    log(
        f"ATH P2 vs sriram 1-ρ={one_m_rho_e4(p2, sriram):.3f}e-4 | "
        f"P1 vs P2={one_m_rho_e4(p1, p2):.3f}e-4 | P3 vs P2={one_m_rho_e4(p3, p2):.3f}e-4"
    )

    o_new = pd.read_csv(PYTAB_OOF).is_fraud.to_numpy(float)
    o_old = pd.read_csv(PYTORCH_OOF).is_fraud.to_numpy(float)
    fam = np.load(FAM20)
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

    vault = np.load(OUT / "gm_overnight_vault.npz")
    u_e2, ut_e2 = unitize(vault["E2_heavy_ridge_oof"]), unitize(vault["E2_heavy_ridge_test"])
    if roc_auc_score(y[f.te0], f.z_s8[f.te0] + 0.01 * u_e2[f.te0]) < roc_auc_score(
        y[f.te0], f.z_s8[f.te0] - 0.01 * u_e2[f.te0]
    ):
        u_e2, ut_e2 = -u_e2, -ut_e2

    go = sum(np.load(GPT2MED / f"oof_{i}.npy").ravel()[:N_TR] for i in range(5)) / 5.0
    gt = sum(np.load(GPT2MED / f"test_{i}.npy").ravel()[:N_TE] for i in range(5)) / 5.0
    g_med_o = L_fold(go, fold_id) - L_fold(masaya_o, fold_id)
    g_med_t = rank_logit(gt) - rank_logit(masaya_t)
    B_o = 0.55 * unitize(g_med_o) + 0.45 * u_e2
    B_t = 0.55 * unitize(g_med_t) + 0.45 * ut_e2

    gbn_o = np.load(BANK / "oof_gbn_bag5_w512.npy").ravel()[:N_TR]
    gbn_t = np.load(BANK / "test_gbn_bag5_w512.npy").ravel()[:N_TE]
    gb_o = np.load(BANK / "oof_gb_bag5.npy").ravel()[:N_TR]
    gb_t = np.load(BANK / "test_gb_bag5.npy").ravel()[:N_TE]
    D_o = 0.55 * unitize(gbn_o) + 0.45 * unitize(gb_o)
    D_t = 0.55 * unitize(gbn_t) + 0.45 * unitize(gb_t)

    dmlp_o = L_fold(o_new, fold_id) - L_fold(o_old, fold_id)
    dmlp_t = rank_logit(raw_new) - rank_logit(raw_old)
    E_o = project_out(dmlp_o, np.column_stack([unitize(L_fold(o_old, fold_id)), unitize(z_gp_oof_L)]))
    E_t = project_out(dmlp_t, np.column_stack([uL(raw_old), uL(raw_gp)]))

    def macro_orth(vo, vt):
        oo, ot, _ = frozen_lofo_orth(vo, vt, B_macro_o, B_macro_t, folds, ridge_alpha=10.0)
        ot = project_out(ot, B_macro_t)
        oo, ot = unitize(oo), unitize(ot)
        if roc_auc_score(y[f.te0], z_sriram_oof[f.te0] + 0.01 * oo[f.te0]) < roc_auc_score(
            y[f.te0], z_sriram_oof[f.te0] - 0.01 * oo[f.te0]
        ):
            oo, ot = -oo, -ot
        return oo, ot

    log("=== Step1: purify U_B vs U_B+U_D+U_E ===")
    uB_o, uB_t = macro_orth(B_o, B_t)
    uD_o, uD_t = macro_orth(D_o, D_t)
    uE_o, uE_t = macro_orth(E_o, E_t)
    # also U_C for comparison reprint
    C_o = 0.55 * f.u_cat + 0.45 * f.u_r2
    C_t = 0.55 * f.ut_cat + 0.45 * f.ut_r2
    uC_o, uC_t = macro_orth(C_o, C_t)

    z_s9 = f.z_s8 + 0.020 * u_e2
    z_s9_d = z_s9 + 0.10 * unitize(dmlp_o)

    for name, uo, ut in [
        ("U_B_pure", uB_o, uB_t),
        ("U_C_tab", uC_o, uC_t),
        ("U_D_gbn", uD_o, uD_t),
        ("U_E_dMLP", uE_o, uE_t),
    ]:
        for bname, base in ("sri", z_sriram_oof), ("z9", z_s9), ("z9d", z_s9_d):
            per = h2h(base + 0.012 * uo, base, y, folds)
            if bname == "sri":
                log(
                    f"  {name}: sri {per['pos']}/5 m={per['mean']:+.2f} min={per['min']:+.2f} "
                    f"t={per['t']:.2f} sf={per['sf25']} max|B|={max_abs_corr(ut, B_macro_t):.4f}"
                )

    u_pure_UB_o, u_pure_UB_t = uB_o, uB_t
    u_mix_o = unitize(uB_o + 0.35 * uD_o + 0.25 * uE_o)
    u_mix_t = unitize(project_out(uB_t + 0.35 * uD_t + 0.25 * uE_t, B_macro_t))
    if roc_auc_score(y[f.te0], z_sriram_oof[f.te0] + 0.01 * u_mix_o[f.te0]) < roc_auc_score(
        y[f.te0], z_sriram_oof[f.te0] - 0.01 * u_mix_o[f.te0]
    ):
        u_mix_o, u_mix_t = -u_mix_o, -u_mix_t
    per_pure = h2h(z_sriram_oof + 0.012 * u_pure_UB_o, z_sriram_oof, y, folds)
    per_mix = h2h(z_sriram_oof + 0.012 * u_mix_o, z_sriram_oof, y, folds)
    log(f"vector1 pure_UB sri t={per_pure['t']:.2f} | vector2 UB+UD+UE sri t={per_mix['t']:.2f}")

    # ---------- R1 / R2 ----------
    log("=== Step1b: build R1 / R2 from P2 ATH ===")
    # R1: C1 geodesic + pure U_B → 1.15e-4 from sriram
    best_lam, best_b, best_r = 0.11, sriram, 0.0
    for lam in np.linspace(0.04, 0.14, 21):
        b = rank_frac((1 - lam) * rank_logit(sriram) + lam * rank_logit(c1))
        r = one_m_rho_e4(b, sriram)
        if abs(r - 0.40) < abs(best_r - 0.40):
            best_lam, best_b, best_r = lam, b, r
    # compose direction from geodesic residual + pure UB
    dir1 = (rank_logit(best_b) - rank_logit(sriram)) + 1.0 * u_pure_UB_t
    w1, r1, rho1_s = bisect_rho_vs(sriram, rank_logit(sriram), dir1, 1.15, hi=2.0)
    rho1_p2 = one_m_rho_e4(r1, p2)
    # OOF for R1
    c1_oof = z_s9 + 0.10 * unitize(dmlp_o)
    oof_r1 = (1 - best_lam) * z_sriram_oof + best_lam * c1_oof + w1 * u_pure_UB_o
    per_r1 = {
        "sri": h2h(oof_r1, z_sriram_oof, y, folds),
        "z9": h2h(oof_r1 - z_sriram_oof + z_s9, z_s9, y, folds),
        "z9d": h2h(oof_r1 - z_sriram_oof + z_s9_d, z_s9_d, y, folds),
    }
    log(
        f"R1: λ={best_lam:.3f} w={w1:.5f} ρs={rho1_s:.3f}e-4 ρ(P2)={rho1_p2:.3f}e-4 "
        f"sri {per_r1['sri']['pos']}/5 t={per_r1['sri']['t']:.2f} min={per_r1['sri']['min']:+.2f}"
    )

    # R2: P2 continuation + U_E → 1.25e-4 from sriram
    w2, r2, rho2_s = bisect_rho_vs(sriram, rank_logit(p2), uE_t, 1.25, hi=1.0)
    # if P2 already at 0.55, need small step — if bisect fails high, step from P2 by target delta
    if abs(rho2_s - 1.25) > 0.15:
        # step from P2 by ~0.20e-4 in U_E direction then re-check vs sriram
        w2b, r2b, rho_from_p2 = bisect_rho(p2, uE_t, 0.20)
        r2, w2, rho2_s = r2b, w2b, one_m_rho_e4(r2b, sriram)
        log(f"R2 retuned via P2+0.20e-4: w={w2:.5f} ρs={rho2_s:.3f} ρ(P2)={rho_from_p2:.3f}")
    rho2_p2 = one_m_rho_e4(r2, p2)
    # also try mix direction if pure E weak
    w2m, r2m, rho2m_s = bisect_rho_vs(sriram, rank_logit(p2), u_mix_t, 1.25, hi=1.0)
    if abs(rho2m_s - 1.25) > 0.15:
        w2m, r2m, _ = bisect_rho(p2, u_mix_t, 0.20)
        rho2m_s = one_m_rho_e4(r2m, sriram)
    oof_r2 = z_sriram_oof + (rank_logit(p2).std() * 0 + 1)  # placeholder
    # proxy: move z_sriram toward P2 oof then add E
    # Use: z_p2_oof ≈ blend; simpler add E onto sri proxy scaled
    oof_r2 = z_sriram_oof + 0.55 / 1.25 * (oof_r1 - z_sriram_oof) + w2 * uE_o  # rough
    # better OOF: start from P2-like oof = oof used in v22.2 P2
    oof_p2 = (1 - 0.11) * z_sriram_oof + 0.11 * c1_oof + 0.02 * unitize(uB_o + uC_o + uD_o)
    oof_r2 = oof_p2 + w2 * uE_o
    per_r2 = {
        "sri": h2h(oof_r2, z_sriram_oof, y, folds),
        "z9": h2h(oof_r2 - z_sriram_oof + z_s9, z_s9, y, folds),
        "z9d": h2h(oof_r2 - z_sriram_oof + z_s9_d, z_s9_d, y, folds),
    }
    # Prefer R2 candidate with better OOF between E-only and mix
    oof_r2m = oof_p2 + w2m * u_mix_o
    per_r2m = {
        "sri": h2h(oof_r2m, z_sriram_oof, y, folds),
        "z9": h2h(oof_r2m - z_sriram_oof + z_s9, z_s9, y, folds),
        "z9d": h2h(oof_r2m - z_sriram_oof + z_s9_d, z_s9_d, y, folds),
    }
    if per_r2m["sri"]["t"] > per_r2["sri"]["t"] and per_r2m["sri"]["min"] > per_r2["sri"]["min"]:
        r2, w2, rho2_s, per_r2 = r2m, w2m, rho2m_s, per_r2m
        rho2_p2 = one_m_rho_e4(r2, p2)
        log("R2 selected mix direction UB+UD+UE")
    log(
        f"R2: w={w2:.5f} ρs={rho2_s:.3f}e-4 ρ(P2)={rho2_p2:.3f}e-4 "
        f"sri {per_r2['sri']['pos']}/5 t={per_r2['sri']['t']:.2f} min={per_r2['sri']['min']:+.2f}"
    )

    # R3 audit staged P1/P3
    log("=== R3 audit staged P1/P3 vs P2 ===")
    for tag, vec in ("P1", p1), ("P3", p3):
        log(
            f"  {tag}: ρ(P2)={one_m_rho_e4(vec, p2):.3f}e-4 ρ(sriram)={one_m_rho_e4(vec, sriram):.3f}e-4 "
            f"ρ(C1)={one_m_rho_e4(vec, c1):.3f}e-4"
        )

    def card(tag, te, per, target_s):
        delta = unitize(rank_logit(te) - rank_logit(p2))
        row = {
            "tag": tag,
            "test": te,
            "per": per,
            "rho_s": one_m_rho_e4(te, sriram),
            "rho_p2": one_m_rho_e4(te, p2),
            "mc": max_abs_corr(delta, B_macro_t),
            "c6": float(np.corrcoef(delta, f.bad6)[0, 1]),
            "c7": float(np.corrcoef(delta, f.bad7)[0, 1]),
            "target": target_s,
            "t3": per["sri"]["t"] + per["z9"]["t"] + per["z9d"]["t"],
            "min3": min(per["sri"]["min"], per["z9"]["min"], per["z9d"]["min"]),
        }
        log(
            f"{tag}: ρs={row['rho_s']:.3f} ρP2={row['rho_p2']:.3f} max|B|={row['mc']:.4f} "
            f"c6={row['c6']:+.3f} c7={row['c7']:+.3f} | "
            f"sri {per['sri']['pos']}/5 t={per['sri']['t']:.2f} min={per['sri']['min']:+.2f} | "
            f"z9 {per['z9']['pos']}/5 t={per['z9']['t']:.2f} | z9d {per['z9d']['pos']}/5 t={per['z9d']['t']:.2f}"
        )
        return row

    r1c = card("R1_P2_Purified_UB_Step_1p15e4", r1, per_r1, 1.15)
    r2c = card("R2_P2_Continuation_plus_DeDiluted_dMLP_1p25e4", r2, per_r2, 1.25)

    # ---------- Step2: replica probe (before submit; parallel-ish) ----------
    log("=== Step2: 45–60s replica/final canary probe ===")
    probe = replica_canary_probe(fold_id, y, timeout_s=60)
    log(f"probe summary: {json.dumps({k:v for k,v in probe.items() if k!='note'})} | {probe.get('note','')[:180]}")

    # pick R1 vs R2
    def key(r):
        return (r["per"]["sri"]["pos"], r["per"]["sri"]["min"], r["t3"], -abs(r["c7"]))

    submit_row = sorted([r1c, r2c], key=key, reverse=True)[0]
    # prefer R1 if both strong — user listed R1 first as purified UB
    if r1c["per"]["sri"]["pos"] == 5 and r1c["per"]["sri"]["min"] > 0 and r1c["t3"] >= r2c["t3"] - 1.0:
        submit_row = r1c
    log(f"SELECT {submit_row['tag']}")

    def stage(path, vec):
        v = np.asarray(vec, float)
        assert len(v) == N_TE and np.isfinite(v).all()
        pd.DataFrame({"id": ids, "is_fraud": v}).to_csv(path, index=False)
        log(f"STAGED {path.name}")

    p_r1 = SUB / "sub_v22_3_R1_purified_UB_step_1p15e4.csv"
    p_r2 = SUB / "sub_v22_3_R2_P2_cont_dediluted_1p25e4.csv"
    stage(p_r1, r1c["test"])
    stage(p_r2, r2c["test"])
    live = SUB / "sub_v22_3_LIVE_ridge_continuation.csv"
    stage(live, submit_row["test"])

    log("=== Step3: submit ONE continuation ===")
    msg = (
        f"v22.3 {submit_row['tag']} | ρs={submit_row['rho_s']:.3f}e-4 ρP2={submit_row['rho_p2']:.3f}e-4 "
        f"c7={submit_row['c7']:+.3f} OOF sri {submit_row['per']['sri']['pos']}/5 t={submit_row['per']['sri']['t']:.2f}"
    )
    log(f"SUBMIT {msg}")
    submit_ok, pub = submit_and_poll(live, msg)

    # 3-point curve
    curve = {
        "0.00e-4_sriram": 0.95422,
        "0.55e-4_P2": 0.95425,
        f"{submit_row['rho_s']:.2f}e-4_R": pub,
    }
    log(f"curve: {curve}")

    body = [
        "# GM V22.3 Ridge Continuation Report",
        "",
        f"**elapsed** {time.time()-T0:.0f}s · local Mac · ATH was `0.95425` (P2)",
        "",
        "## Live submit",
        f"- OK=`{submit_ok}` · **{submit_row['tag']}**",
        f"- Public LB: **{pub}**",
        f"- 1−ρ vs sriram: `{submit_row['rho_s']:.3f}e-4` · vs P2: `{submit_row['rho_p2']:.3f}e-4`",
        f"- max|B_macro|: `{submit_row['mc']:.4f}` · c6=`{submit_row['c6']:+.4f}` c7=`{submit_row['c7']:+.4f}`",
        f"- OOF sri `{submit_row['per']['sri']['pos']}/5` t=`{submit_row['per']['sri']['t']:.2f}` "
        f"min=`{submit_row['per']['sri']['min']:+.2f}`",
        "",
        "## 3-point live curve",
        f"- `0.00e-4` → sriram **0.95422**",
        f"- `0.55e-4` → P2 **0.95425** (+3e-5)",
        f"- `{submit_row['rho_s']:.2f}e-4` → this sub **{pub}**",
        "",
        "## Replica canary (NOT in submit)",
        f"- ok=`{probe.get('ok')}` n=`{probe.get('n')}` AUC_f0=`{probe.get('auc_f0')}` "
        f"sec=`{probe.get('seconds')}`",
        f"- corr_MS=`{probe.get('corr_ms')}` corr_G26=`{probe.get('corr_g26')}`",
        f"- note: {probe.get('note','')}",
        "",
        "## Staged",
        f"- `{p_r1.name}`",
        f"- `{p_r2.name}`",
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
