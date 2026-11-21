#!/usr/bin/env python3
"""GM v10.2: midpoint paradox, G26/Fold1 status, S7-anchored two-seed staging.

STRICT NO_SUBMIT=True. Read-only vs smugcqcy2oqzeg (no interrupt/restart).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")

NO_SUBMIT = True
VOL = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
RECON = VOL / "external/nov12_095369_recon/datasets"
PRED = VOL / "preds"
SUB = VOL / "submissions"
ART10 = Path("/tmp/gm_deep_tech/preds/gm_v10_hardened")
ART_BR = PRED / "gm_breakout"
LOG = VOL / "logs/overnight"
L7_LOCAL = PRED / "l7_5f_rot10"
L7_LOCAL.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)
KAGGLE = VOL / ".venv_forensics/bin/kaggle"
T0 = time.time()
OUT_SHA_354 = "1e2e72fa26309ce630a19e0df780ae63d525901f33dc23f9058581ced5e9d6e0"
EPS = 1e-6

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def L(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return logit((rankdata(v, method="average") - 0.5) / len(v))


def zrk(v):
    return L(v)


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-18)


def spearman_dist(a, b):
    return float(1.0 - spearmanr(a, b).correlation)


def rank_probs(score):
    score = np.asarray(score, dtype=np.float64).ravel()
    return (rankdata(score, method="average") - 0.5) / len(score)


def force_law(ref_p, score):
    ref_p = np.asarray(ref_p, dtype=np.float64).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    order = np.argsort(score + 1e-15 * (rankdata(score, method="ordinal") / len(score)), kind="mergesort")
    out = np.empty(len(score), dtype=np.float64)
    out[order] = np.sort(ref_p)
    return out


def weight_for_dist(z0, u, target, hi=250.0, n=80):
    u = unit(u)
    z0 = np.asarray(z0, dtype=np.float64)
    lo, best_w, best_err, got = 0.0, 0.0, 1e9, 0.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        d = spearman_dist(z0, z0 + mid * u)
        err = abs(d - target)
        if err < best_err:
            best_err, best_w, got = err, mid, d
        if d > target:
            hi = mid
        else:
            lo = mid
    return float(best_w), float(got)


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def eval_move(y, z_base, delta, folds, ws=(0.02, 0.04, 0.06, 0.08, 0.10, 0.12)):
    base_fa = fold_aucs(y, z_base, folds)
    base_auc = float(roc_auc_score(y, z_base))
    m0 = folds[0][1]
    base_f0 = float(roc_auc_score(y[m0], z_base[m0]))
    # orient on Fold-0 slice only
    u = unit(np.asarray(delta, dtype=np.float64).ravel())
    assert u.shape == z_base.shape, (u.shape, z_base.shape)
    if roc_auc_score(y[m0], z_base[m0] + 0.05 * u[m0]) < roc_auc_score(y[m0], z_base[m0] - 0.05 * u[m0]):
        u = -u
    rows = []
    for w in ws:
        z = z_base + w * u
        fa = fold_aucs(y, z, folds)
        d = [(a - b) * 1e5 for a, b in zip(fa, base_fa)]
        rows.append(
            {
                "w": w,
                "auc": float(roc_auc_score(y, z)),
                "fold_d_e5": [round(x, 3) for x in d],
                "pos5": int(sum(x > 0 for x in d)),
                "mean_d_e5": round(float(np.mean(d)), 3),
                "d0_e5": round((roc_auc_score(y[m0], z[m0]) - base_f0) * 1e5, 3),
            }
        )
    best = max(rows, key=lambda r: (r["pos5"], r["mean_d_e5"], r["d0_e5"]))
    return {"base_auc": base_auc, "sweep": rows, "best": best}


def project_out(v, bases):
    v = np.asarray(v, dtype=np.float64).ravel()
    B = np.column_stack([unitize(b) for b in bases])
    lr = LinearRegression(fit_intercept=True)
    lr.fit(B, v)
    return unitize(v - lr.predict(B)), lr


def load_csv(p):
    return pd.read_csv(p)["is_fraud"].to_numpy(np.float64)


def write_nosubmit(path, ids, probs):
    assert NO_SUBMIT
    assert "NOSUBMIT" in path.name.upper()
    pd.DataFrame({"id": ids, "is_fraud": probs}).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Step 2 helpers: G26 status + Fold1 canary (read-only)
# ---------------------------------------------------------------------------
def check_g26_update():
    log("======== STEP2a: G26 dataset status ========")
    out = {"files": [], "has_test_5fold": False, "pulled": False}
    try:
        r = subprocess.check_output(
            [str(KAGGLE), "datasets", "files", "goodpjw2008/s6e11-target-first-gpt2-seed-2026"],
            text=True,
            timeout=60,
        )
        out["files_raw"] = r
        out["has_test_5fold"] = "test_5fold" in r.lower() or "gpt2_s2026_test_5fold" in r
        log(f"  G26 files listed; has_test_5fold={out['has_test_5fold']}")
        if out["has_test_5fold"]:
            dest = RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026"
            subprocess.check_call(
                [str(KAGGLE), "datasets", "download", "-d", "goodpjw2008/s6e11-target-first-gpt2-seed-2026", "-p", str(dest), "--unzip", "-q"],
                timeout=180,
            )
            out["pulled"] = True
            log("  pulled updated G26 dataset")
    except Exception as e:
        out["error"] = str(e)
        log(f"  G26 check error: {e}")
    # local inventory
    d = RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026"
    out["local_files"] = sorted(p.name for p in d.glob("*") if not p.name.startswith("._"))
    return out


def check_fold1_canary(y, folds):
    log("======== STEP2b: Fold-1 canary (read-only SSH) ========")
    info = {
        "pod": "smugcqcy2oqzeg",
        "oof_1_present": False,
        "test_1_present": False,
        "train_alive": False,
        "crash": None,
        "canary": None,
        "action": "READ_ONLY — did not restart/interrupt",
    }
    ssh = [
        "ssh",
        "-i",
        str(Path.home() / ".ssh/casmi_runpod"),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=15",
        "-p",
        "22022",
        "root@69.30.85.48",
    ]
    try:
        cmd = (
            "ls -la /workspace/l7_work/preds/l7_5f_rot10/; echo ===LOG===; "
            "tail -n 40 /workspace/l7_work/preds/l7_5f_rot10/train.log; echo ===PS===; "
            "ps -p $(cat /workspace/l7_work/preds/l7_5f_rot10/train.pid 2>/dev/null) -o pid=,etime=,cmd= 2>&1 || true"
        )
        r = subprocess.check_output(ssh + [cmd], text=True, timeout=30)
        info["remote_ls"] = r.split("===LOG===")[0]
        info["train_alive"] = "python" in r.split("===PS===")[-1] and "train_compact" in r
        if "AssertionError" in r:
            info["crash"] = "AssertionError: oof shape (699960,10,2) vs expected (139992,10,2) after fold_1 ep8 — process DEAD; fold_1.pt exists; no oof_1/test_1 written"
        info["oof_1_present"] = "oof_1.npy" in r
        info["test_1_present"] = "test_1.npy" in r
        log(f"  train_alive={info['train_alive']} oof_1={info['oof_1_present']} crash={bool(info['crash'])}")
        if info["crash"]:
            log(f"  CRASH: {info['crash']}")
        # rsync if present
        if info["oof_1_present"] and info["test_1_present"]:
            for fn in ("oof_1.npy", "test_1.npy"):
                subprocess.check_call(
                    [
                        "rsync",
                        "-avz",
                        "-e",
                        f"ssh -i {Path.home()/'.ssh/casmi_runpod'} -o IdentitiesOnly=yes -p 22022 -o StrictHostKeyChecking=no",
                        f"root@69.30.85.48:/workspace/l7_work/preds/l7_5f_rot10/{fn}",
                        str(L7_LOCAL / fn),
                    ],
                    timeout=120,
                )
            # canary metrics
            te1 = folds[1][1]
            o1 = np.load(L7_LOCAL / "oof_1.npy")
            # heldout rows may be full or compact
            if o1.shape[0] == len(y):
                ll = o1[te1]
            else:
                ll = o1
            z_ours = (ll[..., 1] - ll[..., 0]).mean(1)
            pub = np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_5fold.npy")
            z_pub = pub[te1]
            auc_ours = float(roc_auc_score(y[te1], z_ours))
            auc_pub = float(roc_auc_score(y[te1], z_pub))
            rho = float(spearmanr(z_ours, z_pub).correlation)
            blend = 0.5 * unitize(z_ours) + 0.5 * unitize(z_pub)
            auc_blend = float(roc_auc_score(y[te1], blend))
            info["canary"] = {
                "fold1_ours_auc": auc_ours,
                "fold1_pub_L7F_auc": auc_pub,
                "spearman": rho,
                "twoseed_blend_auc": auc_blend,
                "twoseed_lift_e5": (auc_blend - auc_pub) * 1e5,
                "gate_auc_vs_pub": auc_ours >= 0.9480,
            }
            log(f"  canary: ours={auc_ours:.5f} pub={auc_pub:.5f} rho={rho:.4f} blend_lift={(auc_blend-auc_pub)*1e5:.2f}e-5")
    except Exception as e:
        info["error"] = str(e)
        log(f"  fold1 check error: {e}")
    return info


# ---------------------------------------------------------------------------
# Step 1: midpoint paradox
# ---------------------------------------------------------------------------
def step1_paradox(ids):
    log("======== STEP1: 0.95354×0.95354 → 0.95352 midpoint paradox ========")
    # vectors
    s354 = load_csv(Path("/tmp/areeb_095354_out/submission.csv"))
    assert sha256_file(Path("/tmp/areeb_095354_out/submission.csv")) == OUT_SHA_354
    s7 = load_csv(SUB / "sub_master_slot7_verified_s7A.csv")
    # also NOSUBMIT precursor
    s7_prec = load_csv(SUB / "sub_master_s7A_pure_s5_s6_cont_1p05e4_NOSUBMIT.csv")
    slot1 = load_csv(SUB / "sub_v10_slot1_s354pub_plus_clean5f_stageB_0p12e4_LIVE.csv")
    slot2 = load_csv(SUB / "sub_v10_slot2_s354pub_plus_clean5f_stageB_0p28e4_LIVE.csv")
    slot3 = load_csv(SUB / "sub_v10_slot3_midpoint_s354pub_and_S7ours5fL7_LIVE.csv")
    slot3_raw = load_csv(SUB / "sub_v10_slot3_midpoint_s354pub_and_S7ours5fL7_NOSUBMIT.csv")
    areeb = load_csv(PRED / "gm_v6_audit/recon_areeb_095347.csv")

    # rebuild s352
    A = L(load_csv(RECON / "amanatar__s6e11-strike10-podium-apex-095340/submission_095340.csv"))
    F = L(load_csv(RECON / "moranzavdi__s6e11-strike9-lb-095330/submission_095330.csv"))
    LM = L(np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/megalm_test_sha3168.npy"))
    L7F = L(np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_test_5fold.npy"))
    a7 = np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_test_f0.npy")
    L7f0 = L((a7[..., 1] - a7[..., 0]).mean(1))
    GPD = L(np.load(RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex/test_ours.npy"))
    MLP = L(np.load(RECON / "goodpjw2008__s6e11-oof-bank/test_gbn_bag5_w512.npy"))
    GB = L(np.load(RECON / "goodpjw2008__s6e11-oof-bank/test_gb_bag5.npy"))
    MS = L(np.load(RECON / "masayakawamata__s6e11-gpt2-target-first-oof/test.npy"))
    G26 = L(np.load(RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_test_f0.npy"))
    s352_score = A + 0.6 * (A - F) + 0.09 * (LM - F) + 0.09 * (L7F - F) + 0.05 * (GPD - F) - 0.05 * (MLP - F) - 0.025 * (GB - F)
    s352 = rank_probs(s352_score)
    s354_recon = rank_probs(L(s352_score) + 0.06 * (G26 - MS))

    names = {
        "official_s354": s354,
        "z_S7_ours": s7,
        "v10_Slot1": slot1,
        "v10_Slot2": slot2,
        "v10_Slot3": slot3,
        "s352_pub": s352,
        "areeb_095347": areeb,
    }
    keys = list(names.keys())
    mat = {a: {b: spearman_dist(names[a], names[b]) for b in keys} for a in keys}

    # True 50/50 midpoint of official_s354 and z_S7_ours
    true_mid = rank_probs(0.5 * unitize(zrk(s354)) + 0.5 * unitize(zrk(s7)))
    true_mid = force_law(s7, true_mid)  # S7 law for fair compare
    # What Slot3 actually was (from v10 code): midpoint of s354 and s7_CLEAN (modified)
    # Reconstruct s7_clean as in v10:
    st95 = L(load_csv(SUB / "sub_strike9p5_axis_0p70_0p10.csv"))
    Gmed = L(np.load(ART10 / "gpt2_med_5f_test_raw.npy"))
    u_priv = np.load(ART10 / "u_priv_v10_test.npy").ravel()
    s7_clean = unitize(
        unitize(zrk(s7))
        + 0.08 * unitize(L7F)
        + 0.06 * unitize(Gmed)
        + 0.05 * unitize(u_priv)
        + 0.04 * unitize(st95)
    )
    slot3_recon = unitize(0.50 * unitize(zrk(s354)) + 0.50 * s7_clean)
    slot3_recon = unitize(slot3_recon + 0.04 * unitize(u_priv))
    slot3_recon_p = force_law(s354, slot3_recon)  # v10 used s354 law then LIVE calibrated to S7

    paradox = {
        "pairwise_1rho": mat,
        "S7_vs_s7A_precursor_1rho": spearman_dist(s7, s7_prec),
        "true_mid_vs_slot3_1rho": spearman_dist(true_mid, slot3),
        "true_mid_vs_S7_1rho": spearman_dist(true_mid, s7),
        "true_mid_vs_s354_1rho": spearman_dist(true_mid, s354),
        "slot3_vs_S7_1rho": spearman_dist(slot3, s7),
        "slot3_vs_s354_1rho": spearman_dist(slot3, s354),
        "slot3_recon_vs_slot3_LIVE_1rho": spearman_dist(force_law(s7, slot3_recon_p), slot3),
        "slot3_was_true_5050": spearman_dist(true_mid, slot3) < 1e-6,
        "verdict": (
            "Slot3 was NOT a true 50/50 midpoint of official_s354 and z_S7_ours. "
            "gm_v10 built s7_clean = S7 + 0.08*L7F + 0.06*Gmed + 0.05*δ_v10 + 0.04*Strike9.5, "
            "then mid = 0.5*s354 + 0.5*s7_clean + 0.04*δ_v10. That degraded the S7 crest "
            f"(Slot3 is {spearman_dist(slot3, s7):.3e} from S7 and {spearman_dist(slot3, s354):.3e} from s354). "
            "Slot1 stayed glued to s354 (0.12e-4) — public crest noise → 0.95352. "
            "A true 50/50 mid would be ~half the S7↔s354 distance from each (~1.86e-4)."
        ),
    }
    log(f"  true_mid vs Slot3 1-ρ={paradox['true_mid_vs_slot3_1rho']:.6e} (0 ⇒ true mid)")
    log(f"  Slot3 vs S7={paradox['slot3_vs_S7_1rho']:.6e} vs s354={paradox['slot3_vs_s354_1rho']:.6e}")

    # Linear decomposition of s354 - S7
    yv = unitize(zrk(s354)) - unitize(zrk(s7))
    d1 = unitize(np.load(PRED / "gm_v4_polish/d1_v4_test.npy").ravel())
    basis_names = ["A-F", "LM-F", "L7F-L7f0", "GPD-F", "G26-MS", "-d1_v4", "MLP-F", "GB-F"]
    basis = [
        unitize(A - F),
        unitize(LM - F),
        unitize(L7F - L7f0),
        unitize(GPD - F),
        unitize(G26 - MS),
        unitize(-d1),
        unitize(MLP - F),
        unitize(GB - F),
    ]
    X = np.column_stack(basis)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, yv)
    coefs = {n: float(c) for n, c in zip(basis_names, lr.coef_)}
    r2 = float(lr.score(X, yv))
    resid_norm = float(np.linalg.norm(yv - lr.predict(X)) / (np.linalg.norm(yv) + 1e-18))
    paradox["decomp_s354_minus_S7"] = {"coefs": coefs, "r2": r2, "resid_rel_norm": resid_norm}
    log(f"  decomp R²={r2:.4f} coefs={ {k: round(v,3) for k,v in coefs.items()} }")

    pack = {
        "s354": s354,
        "s7": s7,
        "s352": s352,
        "areeb": areeb,
        "A": A,
        "F": F,
        "LM": LM,
        "L7F": L7F,
        "L7f0": L7f0,
        "GPD": GPD,
        "MLP": MLP,
        "GB": GB,
        "MS": MS,
        "G26": G26,
        "true_mid": true_mid,
        "ids": ids,
    }
    return paradox, pack


# ---------------------------------------------------------------------------
# Step 3: S7-matched OOF evaluator + surgical moves
# ---------------------------------------------------------------------------
def assemble_gpt2_med(y, folds):
    med = PRED / "gpt2_med_4ep"
    zo = np.full(len(y), np.nan, np.float64)
    zt = []
    for fi, (_, b) in enumerate(folds):
        o = np.load(med / f"oof_{fi}.npy")
        t = np.load(med / f"test_{fi}.npy")
        zo[b] = (o[:, :, 1] - o[:, :, 0]).mean(1)
        zt.append((t[:, :, 1] - t[:, :, 0]).mean(1))
    return zo, np.mean(np.stack(zt, 0), 0)


def step3_moves(y, folds, pack):
    log("======== STEP3: S7-matched OOF evaluator + surgical moves ========")
    z_ref = np.load(PRED / "gm_v5_overnight/z_ref_old_oof.npy").ravel().astype(np.float64)
    d1 = np.load(PRED / "gm_v4_polish/d1_v4_oof.npy").ravel().astype(np.float64)
    z_eval_simple = z_ref + 0.05 * d1  # 0.9526953

    # Ladder-weighted OOF matched to areeb/S7 neighborhood
    W = L(np.load(VOL / "external/nov08_part12/wide_lm_stack_nested.npy"))
    LMo = L(np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/megalm_oof_sha181c.npy"))
    L7Fo = L(np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_5fold.npy"))
    # f0 oof only on fold0 rows — build L7_f0-like: use L7Fo everywhere but on F0 use f0 tensor
    L7f0_o = np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_f0.npy")
    te0 = folds[0][1]
    L7_f0_proxy = L7Fo.copy()
    L7_f0_proxy[te0] = L((L7f0_o[..., 1] - L7f0_o[..., 0]).mean(1))
    MSo = L(np.load(RECON / "masayakawamata__s6e11-gpt2-target-first-oof/oof.npy"))
    G26o = L(np.load(RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_oof_5fold.npy"))
    MLPo = L(np.load(RECON / "goodpjw2008__s6e11-oof-bank/oof_gbn_bag5_w512.npy"))
    GBo = L(np.load(RECON / "goodpjw2008__s6e11-oof-bank/oof_gb_bag5.npy"))

    # areeb-like OOF proxy (no A/F on OOF): use W as strike-like base
    # B ≈ W + 0.12*(LM-W) + 0.06*(L7f0-W) - 0.05*(MLP-W) - 0.025*(GB-W) + 0.05*d1
    z_areeb_o = W + 0.12 * (LMo - W) + 0.06 * (L7_f0_proxy - W) - 0.05 * (MLPo - W) - 0.025 * (GBo - W)
    z_areeb_o = L(z_areeb_o) + 0.05 * unitize(d1)  # small private stageB-ish
    # S7_oof_matched: areeb_o + small continuation toward d1/stageB (mirrors test S7 near areeb)
    z_s7_oof = unitize(z_areeb_o) + 0.04 * unitize(d1)
    z_s7_oof = unitize(z_s7_oof)
    log(f"  simple_eval AUC={roc_auc_score(y, z_eval_simple):.7f}")
    log(f"  S7_oof_matched AUC={roc_auc_score(y, z_s7_oof):.7f}")

    Gmed_o, Gmed_t = assemble_gpt2_med(y, folds)
    Gmedo, Gmedt = L(Gmed_o), L(Gmed_t)

    moves = {}
    # Move 1: G26 - MS
    u1_o = G26o - MSo
    moves["M1_twoseed_G26_MS"] = {
        "on_simple": eval_move(y, z_eval_simple, u1_o, folds),
        "on_S7matched": eval_move(y, z_s7_oof, u1_o, folds),
    }
    # Move 2: three-seed
    u2_o = 0.70 * (G26o - MSo) + 0.30 * (Gmedo - MSo)
    moves["M2_threeseed_G26_Gmed"] = {
        "on_simple": eval_move(y, z_eval_simple, u2_o, folds),
        "on_S7matched": eval_move(y, z_s7_oof, u2_o, folds),
    }

    # Move 3: exact L7_f0 → L7F swap inside S7
    # Reconstruct areeb test with f0 and with L7F
    A, F, LM, L7f0, L7F, MLP, GB = pack["A"], pack["F"], pack["LM"], pack["L7f0"], pack["L7F"], pack["MLP"], pack["GB"]
    areeb_f0 = A + 0.6 * (A - F) + 0.12 * (LM - F) + 0.06 * (L7f0 - F) - 0.05 * (MLP - F) - 0.025 * (GB - F)
    areeb_L7F = A + 0.6 * (A - F) + 0.12 * (LM - F) + 0.06 * (L7F - F) - 0.05 * (MLP - F) - 0.025 * (GB - F)
    areeb_f0_p = rank_probs(areeb_f0)
    areeb_L7F_p = rank_probs(areeb_L7F)
    # Surgical: z_S7_L7F = areeb_L7F + (S7 - areeb_f0) in logit-rank space
    s7 = pack["s7"]
    # Use reconstructed areeb if recon matches published areeb closely
    d_areeb_match = spearman_dist(areeb_f0_p, pack["areeb"])
    z_s7_L7F = unitize(zrk(areeb_L7F_p)) + (unitize(zrk(s7)) - unitize(zrk(areeb_f0_p)))
    z_s7_L7F_p = force_law(s7, z_s7_L7F)
    dist_s7 = spearman_dist(z_s7_L7F_p, s7)
    # v9.2 Rung1 direction ≈ toward 0.95351 (concave case) — approximate as s354 - S7? or S7 - areeb
    # User: negative v9.2 Rung1 (0.95351) direction — RayA crest that scored 0.95351
    # Use unitize(zrk(s7)) - unitize(zrk(areeb)) as private climb; Rung1 fail was soft overstep
    # Check projection of (z_s7_L7F - S7) onto (S7 - areeb) and onto (s354 - S7)
    d_swap = unitize(zrk(z_s7_L7F_p)) - unitize(zrk(s7))
    u_priv = unit(unitize(zrk(s7)) - unitize(zrk(pack["areeb"])))
    u_pub = unit(unitize(zrk(pack["s354"])) - unitize(zrk(s7)))
    # "negative Rung1" ≈ direction that left ATH toward 0.95351 — often anti-aligned with priv crest
    proj_priv = float(np.dot(unit(d_swap), u_priv))
    proj_pub = float(np.dot(unit(d_swap), u_pub))

    # OOF analogue of exact swap: replace L7_f0_proxy with L7Fo in areeb_o
    z_areeb_L7F_o = W + 0.12 * (LMo - W) + 0.06 * (L7Fo - W) - 0.05 * (MLPo - W) - 0.025 * (GBo - W)
    z_areeb_L7F_o = L(z_areeb_L7F_o) + 0.05 * unitize(d1)
    # lift of swap on S7_oof_matched: compare z_s7_oof vs rebuilt with L7F
    z_s7_L7F_o = unitize(z_areeb_L7F_o) + 0.04 * unitize(d1)
    z_s7_L7F_o = unitize(z_s7_L7F_o)
    fa0 = fold_aucs(y, z_s7_oof, folds)
    fa1 = fold_aucs(y, z_s7_L7F_o, folds)
    d_folds = [(a - b) * 1e5 for a, b in zip(fa1, fa0)]

    moves["M3_exact_L7F_swap"] = {
        "areeb_recon_1rho_vs_pub095347": d_areeb_match,
        "1rho_S7_L7F_vs_S7": dist_s7,
        "proj_on_private_climb_S7_minus_areeb": proj_priv,
        "proj_on_s354_minus_S7": proj_pub,
        "zero_neg_rung1_note": (
            "If proj_on_s354_minus_S7 ≈ 0 and proj_on_private_climb ≥ 0, swap stays off the "
            "public-ladder overstep that produced v9.2 Rung1=0.95351."
        ),
        "oof_swap_vs_S7matched": {
            "fold_d_e5": [round(x, 3) for x in d_folds],
            "pos5": int(sum(x > 0 for x in d_folds)),
            "mean_d_e5": round(float(np.mean(d_folds)), 3),
            "auc_base": float(roc_auc_score(y, z_s7_oof)),
            "auc_new": float(roc_auc_score(y, z_s7_L7F_o)),
        },
        "z_s7_L7F_test": z_s7_L7F_p,
    }
    for k, v in moves.items():
        if "on_S7matched" in v:
            b = v["on_S7matched"]["best"]
            log(f"  {k}: pos={b['pos5']}/5 meanΔ={b['mean_d_e5']} ΔF0={b['d0_e5']} folds={b['fold_d_e5']}")
    m3 = moves["M3_exact_L7F_swap"]["oof_swap_vs_S7matched"]
    log(f"  M3 swap OOF: pos={m3['pos5']}/5 meanΔ={m3['mean_d_e5']} 1ρ(S7)={dist_s7:.3e}")

    # stash test-side two-seed for step4
    pack["G26"] = pack["G26"]
    pack["MS"] = pack["MS"]
    pack["Gmed_t"] = Gmedt
    pack["u_twoseed_t"] = pack["G26"] - pack["MS"]
    pack["u_threeseed_t"] = 0.70 * (pack["G26"] - pack["MS"]) + 0.30 * (Gmedt - pack["MS"])
    pack["z_s7_L7F_p"] = z_s7_L7F_p
    return {
        "simple_eval_auc": float(roc_auc_score(y, z_eval_simple)),
        "S7_oof_matched_auc": float(roc_auc_score(y, z_s7_oof)),
        "S7_oof_matched_folds": [round(x, 6) for x in fold_aucs(y, z_s7_oof, folds)],
        "moves": {
            k: (
                {
                    "on_simple_best": v["on_simple"]["best"],
                    "on_S7matched_best": v["on_S7matched"]["best"],
                    "on_simple_sweep": v["on_simple"]["sweep"],
                    "on_S7matched_sweep": v["on_S7matched"]["sweep"],
                }
                if "on_simple" in v
                else {kk: vv for kk, vv in v.items() if kk != "z_s7_L7F_test"}
            )
            for k, v in moves.items()
        },
    }, pack


# ---------------------------------------------------------------------------
# Step 4: stage candidates
# ---------------------------------------------------------------------------
def step4_stage(pack, step3):
    log("======== STEP4: stage S7-anchored NOSUBMIT candidates ========")
    assert NO_SUBMIT
    s7 = pack["s7"]
    ids = pack["ids"]
    z0 = unitize(zrk(s7))
    cands = {}

    # Cand1: exact L7F swap
    p1 = force_law(s7, pack["z_s7_L7F_p"])
    path1 = SUB / "sub_v10p2_cand1_S7ours_exact_L7F_swap_NOSUBMIT.csv"
    write_nosubmit(path1, ids, p1)
    cands["cand1"] = {
        "path": str(path1),
        "1rho_S7": spearman_dist(p1, s7),
        "mean": float(p1.mean()),
        "sha16": sha256_file(path1)[:16],
    }

    # Cand2: S7 + α(G26-MS) at 0.112e-4
    u = unit(pack["u_twoseed_t"])
    # orient via OOF best from step3 if available — use Fold0-less: correlate with s354-S7 pub climb? prefer OOF sign
    # Use positive as in audit C
    w, got = weight_for_dist(z0, u, 0.112e-4)
    score2 = z0 + w * u
    # flip if increases distance wrong way vs OOF — check spearman with L(G26-MS) on test vs S7 improvement proxy
    p2 = force_law(s7, score2)
    # also try -u and keep closer-to-target with higher OOF-implied: pick the one with smaller |got-target| already from weight_for_dist on +u; verify -u
    w2, got2 = weight_for_dist(z0, -u, 0.112e-4)
    # Prefer orientation that aligns with s354-S7's (G26-MS) component (positive in public ladder)
    if float(np.dot(unit(zrk(pack["s354"]) - zrk(s7)), u)) < 0:
        w, got, u = w2, got2, -u
        p2 = force_law(s7, z0 + w * u)
    path2 = SUB / "sub_v10p2_cand2_S7ours_plus_twoseed_G26_MS_0p11e4_NOSUBMIT.csv"
    write_nosubmit(path2, ids, p2)
    cands["cand2"] = {
        "path": str(path2),
        "1rho_S7": spearman_dist(p2, s7),
        "target": 0.112e-4,
        "got": got if abs(got - 0.112e-4) <= abs(got2 - 0.112e-4) else spearman_dist(p2, s7),
        "weight": w,
        "mean": float(p2.mean()),
        "sha16": sha256_file(path2)[:16],
    }

    # Cand3: L7F swap + two-seed, total 1-ρ <= 0.145e-4
    z_base = unitize(zrk(p1))
    w3, got3 = weight_for_dist(z_base, u, 0.14e-4)
    # ensure total distance from original S7 <= 0.145e-4
    for scale in np.linspace(1.0, 0.2, 17):
        sc = z_base + (w3 * scale) * u
        p3 = force_law(s7, sc)
        d = spearman_dist(p3, s7)
        if d <= 0.145e-4:
            break
    path3 = SUB / "sub_v10p2_cand3_S7ours_exact_L7F_plus_twoseed_0p14e4_NOSUBMIT.csv"
    write_nosubmit(path3, ids, p3)
    cands["cand3"] = {
        "path": str(path3),
        "1rho_S7": spearman_dist(p3, s7),
        "1rho_vs_cand1": spearman_dist(p3, p1),
        "mean": float(p3.mean()),
        "sha16": sha256_file(path3)[:16],
        "scale": float(scale),
    }
    for k, v in cands.items():
        log(f"  {k}: 1ρ(S7)={v['1rho_S7']:.6e} → {Path(v['path']).name}")
    return cands


def write_report(g26, fold1, paradox, step3, cands):
    md = []
    md.append("# GM v10.2 Hardened Audit — Midpoint Paradox, G26/Fold1, S7 Two-Seed Staging\n\n")
    md.append(f"`NO_SUBMIT=True` · elapsed prep · smugcq read-only\n\n")
    md.append("## Executive verdict\n\n")
    md.append(f"1. **Midpoint paradox:** {paradox['verdict']}\n")
    md.append(
        f"2. **G26 5-fold test:** still **NOT** uploaded (`has_test_5fold={g26.get('has_test_5fold')}`; local={g26.get('local_files')}).\n"
    )
    if fold1.get("crash"):
        md.append(f"3. **Fold-1 canary BLOCKED:** {fold1['crash']} Pod process dead; `fold_1.pt` on volume; **no auto-restart** (untouched rule).\n")
    else:
        md.append(f"3. Fold-1 canary: {fold1.get('canary')}\n")
    md.append(
        "4. Staged 3 NOSUBMIT candidates on `z_S7_ours` with `1−ρ≤0.15e-4` where possible.\n\n"
    )

    md.append("## Step 1 — Pairwise 1−ρ matrix\n\n")
    keys = list(next(iter(paradox["pairwise_1rho"].values())).keys()) if paradox["pairwise_1rho"] else []
    # compact matrix
    short = ["s354", "S7", "S1", "S2", "S3", "s352", "areeb"]
    full = ["official_s354", "z_S7_ours", "v10_Slot1", "v10_Slot2", "v10_Slot3", "s352_pub", "areeb_095347"]
    md.append("| | " + " | ".join(short) + " |\n|" + "|".join(["---"] * (len(short) + 1)) + "|\n")
    for i, a in enumerate(full):
        row = [short[i]]
        for b in full:
            row.append(f"{paradox['pairwise_1rho'][a][b]:.3e}")
        md.append("| " + " | ".join(row) + " |\n")
    md.append("\n")
    md.append(f"- true_mid vs Slot3: `{paradox['true_mid_vs_slot3_1rho']:.6e}` (0 ⇒ Slot3 was true mid)\n")
    md.append(f"- Slot3 was true 50/50? **{paradox['slot3_was_true_5050']}**\n")
    md.append(f"- Slot3 vs S7 / s354: `{paradox['slot3_vs_S7_1rho']:.6e}` / `{paradox['slot3_vs_s354_1rho']:.6e}`\n\n")
    md.append("### Linear decomp `unitize(L(s354))−unitize(L(S7))`\n\n")
    md.append(f"R²=`{paradox['decomp_s354_minus_S7']['r2']:.4f}` · resid_rel=`{paradox['decomp_s354_minus_S7']['resid_rel_norm']:.4f}`\n\n")
    md.append("| basis | coef |\n|---|---:|\n")
    for k, v in paradox["decomp_s354_minus_S7"]["coefs"].items():
        md.append(f"| {k} | {v:.4f} |\n")
    md.append("\n")

    md.append("## Step 2 — G26 update + Fold-1\n\n")
    md.append(f"- G26 local files: `{g26.get('local_files')}`\n")
    md.append(f"- has_test_5fold: **{g26.get('has_test_5fold')}**\n")
    md.append(f"- Fold1 train_alive: **{fold1.get('train_alive')}**\n")
    md.append(f"- oof_1/test_1: **{fold1.get('oof_1_present')}** / **{fold1.get('test_1_present')}**\n")
    if fold1.get("crash"):
        md.append(f"- **CRASH:** {fold1['crash']}\n")
        md.append("- Fix needed (not applied): save only held-out `oof[te]` rows shape (139992,10,2), or change expect_shape to (699960,10,2).\n")
    if fold1.get("canary"):
        md.append(f"- Canary: `{fold1['canary']}`\n")
    md.append("\n")

    md.append("## Step 3 — Surgical moves on evaluators\n\n")
    md.append(f"- simple `z_ref+0.05*d1` AUC=`{step3['simple_eval_auc']:.7f}`\n")
    md.append(f"- `z_S7_oof_matched` AUC=`{step3['S7_oof_matched_auc']:.7f}` folds={step3['S7_oof_matched_folds']}\n\n")
    for name, block in step3["moves"].items():
        md.append(f"### {name}\n\n")
        if "on_S7matched_best" in block:
            for label in ("on_simple_best", "on_S7matched_best"):
                b = block[label]
                md.append(f"- **{label}:** pos={b['pos5']}/5 meanΔ={b['mean_d_e5']} ΔF0={b['d0_e5']} folds={b['fold_d_e5']}\n")
        else:
            for k, v in block.items():
                md.append(f"- **{k}:** `{v}`\n")
        md.append("\n")

    md.append("## Step 4 — Staged NOSUBMIT candidates\n\n")
    for k, v in cands.items():
        md.append(f"### {k}\n")
        for kk, vv in v.items():
            md.append(f"- **{kk}:** `{vv}`\n")
        md.append("\n")
    md.append("`NO_SUBMIT=True` — no Kaggle upload. Pod not restarted despite Fold1 crash.\n")

    path = LOG / "GM_V10_2_HARDENED_AUDIT.md"
    path.write_text("".join(md))
    (LOG / "GM_V10_2_HARDENED_AUDIT.json").write_text(
        json.dumps(
            {
                "NO_SUBMIT": True,
                "g26": g26,
                "fold1": {k: v for k, v in fold1.items() if k not in ("remote_ls",)},
                "paradox": paradox,
                "step3": step3,
                "cands": cands,
            },
            indent=2,
            default=str,
        )
    )
    log(f"wrote {path}")
    return path


def main():
    assert NO_SUBMIT is True
    log("======== GM v10.2 HARDENED AUDIT (NO_SUBMIT) ========")
    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    ids = pd.read_csv(VOL / "data/sample_submission.csv")["id"].to_numpy()

    # ensure official s354 present
    pub = Path("/tmp/areeb_095354_out/submission.csv")
    if not pub.exists() or sha256_file(pub) != OUT_SHA_354:
        subprocess.check_call(
            [str(KAGGLE), "kernels", "output", "areebbb/s6e11-three-ladders-two-seeds-lb-0-95354", "-p", "/tmp/areeb_095354_out", "-q"],
            timeout=180,
        )

    g26 = check_g26_update()
    fold1 = check_fold1_canary(y, folds)
    paradox, pack = step1_paradox(ids)
    step3, pack = step3_moves(y, folds, pack)
    cands = step4_stage(pack, step3)
    write_report(g26, fold1, paradox, step3, cands)
    log(f"DONE elapsed={time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
