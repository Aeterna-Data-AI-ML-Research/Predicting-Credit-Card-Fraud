#!/usr/bin/env python3
"""GM v10 Hardened 0.95369 breaker — forensic audit + clean 5-fold L7F Stage-B fusion.

STRICT NO_SUBMIT=True. Local Mac CPU only. Never touches RunPod pods.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("LIGHTGBM_EXEC", "")
warnings.filterwarnings("ignore")

NO_SUBMIT = True  # HARD LOCK
VOL = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
WORK = Path(os.environ.get("GM_V10_ROOT", "/tmp/gm_deep_tech"))
RECON = VOL / "external/nov12_095369_recon/datasets"
PRED = VOL / "preds"
SUB = VOL / "submissions"
ART = WORK / "preds/gm_v10_hardened"
LOG = VOL / "logs/overnight"
for p in (ART, LOG, SUB, WORK / "logs/overnight"):
    p.mkdir(parents=True, exist_ok=True)

T0 = time.time()
EPS = 1e-6
REPORT: dict = {"NO_SUBMIT": True, "generated_at": None, "steps": {}}


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def zrk(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS)).astype(np.float32)


def unitize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return ((v - float(v.mean())) / (float(v.std()) + 1e-12)).astype(np.float32)


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v / (float(np.linalg.norm(v)) + 1e-18)).astype(np.float32)


def spearman_dist(a, b) -> float:
    return float(1.0 - spearmanr(a, b).correlation)


def cos_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


def load_f32(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32).ravel()


def load_csv_score(path: Path) -> np.ndarray:
    return pd.read_csv(path, usecols=["is_fraud"])["is_fraud"].to_numpy(np.float64)


def fold_aucs(y, z, folds) -> list[float]:
    z = np.asarray(z, dtype=np.float64)
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def calibrate_logit_to_prevalence(z, p=0.0167) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).ravel()
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        m = float(expit(z + mid).mean())
        if m > p:
            hi = mid
        else:
            lo = mid
    return (z + 0.5 * (lo + hi)).astype(np.float32)


def pchip_align_fold(oof, test, folds):
    oof = np.asarray(oof, dtype=np.float64).ravel()
    test = np.asarray(test, dtype=np.float64).ravel()
    o2 = np.zeros_like(oof)
    for _, b in folds:
        mu, sd = float(oof[b].mean()), float(oof[b].std() + 1e-12)
        o2[b] = (oof[b] - mu) / sd
    o2 = (o2 - o2.mean()) / (o2.std() + 1e-12)
    t2 = (test - test.mean()) / (test.std() + 1e-12)

    def jit(x, s=1e-11):
        return x + s * (rankdata(x, method="ordinal") / len(x) - 0.5)

    q = np.linspace(0, 1, 2001)
    pchip = PchipInterpolator(np.quantile(jit(t2), q), np.quantile(jit(o2), q), extrapolate=True)
    ta = np.asarray(pchip(jit(t2)), dtype=np.float64)
    ta = (ta - ta.mean()) / (ta.std() + 1e-12)
    return o2.astype(np.float32), ta.astype(np.float32)


def project_out(v, bases) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    B = np.column_stack([unitize(b).astype(np.float64) for b in bases])
    lr = LinearRegression(fit_intercept=True)
    lr.fit(B, v)
    return unitize(v - lr.predict(B))


def weight_for_dist(z0, u, target, hi=250.0, n=72):
    u = unit(u).astype(np.float64)
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


def force_exact_law_probs(ref_p, score) -> np.ndarray:
    ref_p = np.asarray(ref_p, dtype=np.float64).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    ref_s = np.sort(ref_p)
    order = np.argsort(score + 1e-12 * (rankdata(score, method="ordinal") / len(score)), kind="mergesort")
    out = np.empty(len(score), dtype=np.float64)
    out[order] = ref_s
    return out


def assemble_gpt2_med(y, folds):
    med = PRED / "gpt2_med_4ep"
    ntr = len(y)
    zo = np.full(ntr, np.nan, np.float64)
    zt_f = []
    for fi, (_, b) in enumerate(folds):
        oraw = np.load(med / f"oof_{fi}.npy")
        traw = np.load(med / f"test_{fi}.npy")
        llr = oraw[:, :, 1].astype(np.float64) - oraw[:, :, 0].astype(np.float64)
        zo[b] = llr.mean(1)
        tllr = traw[:, :, 1].astype(np.float64) - traw[:, :, 0].astype(np.float64)
        zt_f.append(tllr.mean(1))
        del oraw, traw, llr, tllr
    assert np.isfinite(zo).all()
    return zo.astype(np.float32), np.mean(np.stack(zt_f, 0), 0).astype(np.float32)


def quadratic_features(X: np.ndarray) -> np.ndarray:
    """2nd-order cross interactions (incl squares). X: (n,d) float32 unitized cols."""
    n, d = X.shape
    cols = [X]
    for i in range(d):
        for j in range(i, d):
            cols.append((X[:, i] * X[:, j])[:, None])
    return np.concatenate(cols, axis=1).astype(np.float32)


def write_nosubmit(path: Path, ids, probs):
    assert NO_SUBMIT
    path = Path(path)
    assert "NOSUBMIT" in path.name.upper()
    pd.DataFrame({"id": ids, "is_fraud": probs}).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Step 0 — pod termination status (read-only local note; MCP already verified)
# ---------------------------------------------------------------------------
def step0_pod_status():
    log("======== STEP0: pod termination status ========")
    note = {
        "requested_terminate": "rsoju37zhpzaae",
        "result": "ALREADY_GONE_404",
        "final_spend_est_usd": "~0.45–0.55 (pod died earlier; remount was smugcqcy2oqzeg)",
        "active_usd_per_hr_for_rsoju": 0.0,
        "untouched": ["smugcqcy2oqzeg", "lheroidmuab1j4", "fujymbm9nkwulv"],
        "rationale": "Public goodpjw compact-lm-seed7 now ships lm7_*_5fold.npy — redundant L7 retrain on rsoju not needed. User hard-rule: never touch smugcqcy2oqzeg.",
    }
    REPORT["steps"]["step0_pod"] = note
    log(f"rsoju37zhpzaae: {note['result']} · $0/hr · untouched={note['untouched']}")
    return note


# ---------------------------------------------------------------------------
# Step 1 — daoviet forensic + asset inventory
# ---------------------------------------------------------------------------
def step1_forensics():
    log("======== STEP1: daoviet / goodpjw forensics ========")
    nb = (
        VOL
        / "external/nov12_095369_recon/daoviet__s6e11-on-the-journey-to-your-first-top-100_code"
        / "s6e11-on-the-journey-to-your-first-top-100.ipynb"
    )
    daoviet = {
        "notebook": str(nb),
        "verified_public_header": 0.95351,
        "ablation_table_public_lb_stage10": 0.95368,
        "ablation_table_local_cv_stage10": 0.95378,
        "verdict": (
            "0.95368/0.95378 appear ONLY in a marketing ablation DataFrame (cell 8). "
            "Executable path prefers prebaked submission_095351.csv (Public LB 0.95351) "
            "OR reconstructs with 1-fold lm7_test_f0 + 0.06*(L7-F) + 0.05*(OUR-F). "
            "No SHA-locked 0.95368 file exists in daoviet/s6e11-gp-dvae-orthogonal-apex."
        ),
        "reconstruct_formula_if_no_prebake": (
            "B = A + 0.60*(A-F) + 0.12*(LM-F) + 0.06*(L7_f0-F); "
            "score = B - 0.05*(MLP-F) - 0.025*(GB-F) + 0.05*(OUR-F)"
        ),
        "dataset_files": sorted(
            p.name for p in (RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex").glob("*") if not p.name.startswith("._")
        ),
        "kernel_output_files": ["submission.csv (876B stub on Kaggle files API — not a full 299984-row artifact)"],
        "combines_with_goodpjw_095354": (
            "No. daoviet apex (0.95351) is an earlier ladder rung; goodpjw 0.95354 adds "
            "5-fold L7F @0.09 + both LMs + GPD@0.05 + 0.06*(G26-MS). "
            "daoviet's claimed 0.95368 is not a composable verified tensor."
        ),
    }
    goodpjw = {
        "s354_formula": (
            "s352 = A + 0.6*(A-F) + 0.09*(LM-F) + 0.09*(L7F-F) + 0.05*(GPD-F) "
            "- 0.05*(MLP-F) - 0.025*(GB-F); "
            "s354 = logit_rank(s352) + 0.06*(G26-MS)"
        ),
        "public_lb": 0.95354,
        "L7F": "lm7_test_5fold.npy (NOT lm7_test_f0)",
        "G26_test": "gpt2_s2026_test_f0.npy — Fold-0 ONLY (README: other folds still scoring)",
    }
    REPORT["steps"]["step1_forensics"] = {"daoviet": daoviet, "goodpjw": goodpjw}
    log(f"daoviet verdict: {daoviet['verdict'][:120]}...")
    return daoviet, goodpjw


# ---------------------------------------------------------------------------
# Step 2 — asset audit + s352/s354 reconstruct
# ---------------------------------------------------------------------------
def step2_audit(y, folds, fold_id):
    log("======== STEP2: asset & split-alignment audit ========")
    paths = {
        "L7F_oof": RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_5fold.npy",
        "L7F_te": RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_test_5fold.npy",
        "mega_oof": RECON / "goodpjw2008__s6e11-compact-lm-seed7/megalm_oof_sha181c.npy",
        "mega_te": RECON / "goodpjw2008__s6e11-compact-lm-seed7/megalm_test_sha3168.npy",
        "G26_oof": RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_oof_5fold.npy",
        "G26_te": RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_test_f0.npy",
        "MS_oof": RECON / "masayakawamata__s6e11-gpt2-target-first-oof/oof.npy",
        "MS_te": RECON / "masayakawamata__s6e11-gpt2-target-first-oof/test.npy",
        "GPD_oof": RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex/oof_ours.npy",
        "GPD_te": RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex/test_ours.npy",
        "MLP_oof": RECON / "goodpjw2008__s6e11-oof-bank/oof_gbn_bag5_w512.npy",
        "MLP_te": RECON / "goodpjw2008__s6e11-oof-bank/test_gbn_bag5_w512.npy",
        "GB_oof": RECON / "goodpjw2008__s6e11-oof-bank/oof_gb_bag5.npy",
        "GB_te": RECON / "goodpjw2008__s6e11-oof-bank/test_gb_bag5.npy",
        "A": RECON / "amanatar__s6e11-strike10-podium-apex-095340/submission_095340.csv",
        "F": RECON / "moranzavdi__s6e11-strike9-lb-095330/submission_095330.csv",
    }
    audit = {}
    vecs = {}
    for k, p in paths.items():
        assert p.exists(), p
        if p.suffix == ".csv":
            v = load_csv_score(p)
            meta = {"shape": [len(v)], "dtype": "csv", "sha16": sha256(p)[:16], "path": str(p)}
        else:
            a = np.load(p)
            v = np.asarray(a, dtype=np.float64).ravel()
            meta = {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "finite": float(np.isfinite(a).mean()),
                "nan": int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0,
                "sha16": sha256(p)[:16],
                "path": str(p),
            }
            if v.shape[0] == len(y):
                fa = fold_aucs(y, v, folds)
                meta["oof_auc"] = float(roc_auc_score(y, v))
                meta["fold_aucs"] = [round(x, 5) for x in fa]
                meta["fold_pattern_ok"] = bool(fa[2] - fa[0] > 0.004)  # F2 harder-easy ~500bps
        audit[k] = meta
        vecs[k] = v.astype(np.float32)
        log(f"  {k}: {meta.get('shape')} auc={meta.get('oof_auc', '—')}")

    # G26 classification
    g26_note = {
        "oof": "5-fold complete (699960) — folds.npy aligned",
        "test": "Fold-0 ONLY (gpt2_s2026_test_f0.npy) — √5× higher test fold noise vs 5-fold",
        "block_near_5f_gbdt": True,
        "reason": "Test vector is 1-fold; do not treat G26_te as 5-fold-complete feature",
    }
    gpd_note = {
        "has_5fold_oof": True,
        "oof_auc": audit["GPD_oof"]["oof_auc"],
        "fold_aucs": audit["GPD_oof"]["fold_aucs"],
        "fold_pattern_ok": audit["GPD_oof"]["fold_pattern_ok"],
        "aligned_with_folds_npy": True,  # SKF42 pattern matches; no fold_ids file but OOF length+pattern OK
        "caution": "Allow as ladder component; exclude from Stage-B hygiene if want pure SKF42 proven bank only — pattern OK so KEEP optional",
    }
    REPORT["steps"]["step2_audit"] = {"assets": audit, "G26": g26_note, "GPD": gpd_note}

    # Reconstruct s352 / s354 on TEST
    A, F = zrk(vecs["A"]), zrk(vecs["F"])
    LM, L7F = zrk(vecs["mega_te"]), zrk(vecs["L7F_te"])
    GPD, MLP, GB = zrk(vecs["GPD_te"]), zrk(vecs["MLP_te"]), zrk(vecs["GB_te"])
    G26, MS = zrk(vecs["G26_te"]), zrk(vecs["MS_te"])
    s352 = A + 0.6 * (A - F) + 0.09 * (LM - F) + 0.09 * (L7F - F) + 0.05 * (GPD - F) - 0.05 * (MLP - F) - 0.025 * (GB - F)
    s354 = zrk(s352) + 0.06 * (G26 - MS)
    s352_u, s354_u = unitize(s352), unitize(s354)

    # OOF counterpart (G26 OOF is 5f; G26 contrast uses 5f OOF)
    Ao = zrk(vecs["A"])  # no OOF for A/F — use test proxies only for test; for OOF use published members
    # Build OOF ladder using OOF members + Strike CSV as constant offsets is invalid.
    # Honest OOF proxy: replace A/F with wide_lm / areeb-style unavailable → use MLP/GB free form on LM members:
    # Use published OOF members relative to a fold-safe base = zrk(wide_lm) as F-proxy and Strike10 unavailable.
    # Practical OOF proxy matching notebook CV: W = wide LM nested; then apply LM/L7F/GPD/MLP/GB moves.
    W = zrk(load_f32(VOL / "external/nov08_part12/wide_lm_stack_nested.npy"))
    Wt = zrk(load_f32(VOL / "external/nov08_part12/wide_lm_stack_test.npy"))
    LMo, L7Fo = zrk(vecs["mega_oof"]), zrk(vecs["L7F_oof"])
    GPDo, MLPo, GBo = zrk(vecs["GPD_oof"]), zrk(vecs["MLP_oof"]), zrk(vecs["GB_oof"])
    G26o, MSo = zrk(vecs["G26_oof"]), zrk(vecs["MS_oof"])
    # Mirror coefficients onto wide base (goodpjw nested CV style)
    s352_o = W + 0.09 * (LMo - W) + 0.09 * (L7Fo - W) + 0.05 * (GPDo - W) - 0.05 * (MLPo - W) - 0.025 * (GBo - W)
    s354_o = zrk(s352_o) + 0.06 * (G26o - MSo)
    s352_o_u, s354_o_u = unitize(s352_o), unitize(s354_o)

    z_areeb = unitize(load_f32(PRED / "gm_breakout/z_areeb.npy"))
    s7_p = load_csv_score(SUB / "sub_master_slot7_verified_s7A.csv")
    z_s7 = unitize(zrk(s7_p))
    d_pub = unitize(s354_u.astype(np.float64) - z_areeb.astype(np.float64))
    d_priv = unitize(z_s7.astype(np.float64) - z_areeb.astype(np.float64))
    recon_stats = {
        "s354_test_mean": float(np.mean((rankdata(s354) - 0.5) / len(s354))),
        "spearman_dist_s354_vs_S7": spearman_dist(s354_u, z_s7),
        "cos_delta_pub_vs_priv": cos_sim(d_pub, d_priv),
        "spearman_dist_s354_vs_areeb": spearman_dist(s354_u, z_areeb),
        "s354_oof_auc": float(roc_auc_score(y, s354_o)),
        "s354_oof_fold_aucs": [round(x, 6) for x in fold_aucs(y, s354_o, folds)],
        "s352_oof_auc": float(roc_auc_score(y, s352_o)),
        "G26_is_test_fold0_only": True,
    }
    log(
        f"s354 reconstr: dist(S7)={recon_stats['spearman_dist_s354_vs_S7']:.6e} "
        f"cos(Δpub,Δpriv)={recon_stats['cos_delta_pub_vs_priv']:.4f} "
        f"OOF AUC={recon_stats['s354_oof_auc']:.6f}"
    )
    REPORT["steps"]["step2_recon"] = recon_stats

    pack = {
        "vecs": vecs,
        "A": A,
        "F": F,
        "LM": LM,
        "L7F": L7F,
        "GPD": GPD,
        "MLP": MLP,
        "GB": GB,
        "G26": G26,
        "MS": MS,
        "s352": s352.astype(np.float32),
        "s354": s354.astype(np.float32),
        "s354_u": s354_u,
        "s352_o": s352_o.astype(np.float32),
        "s354_o": s354_o.astype(np.float32),
        "s354_o_u": s354_o_u,
        "W": W,
        "Wt": Wt,
        "LMo": LMo,
        "L7Fo": L7Fo,
        "GPDo": GPDo,
        "MLPo": MLPo,
        "GBo": GBo,
        "G26o": G26o,
        "MSo": MSo,
        "z_areeb": z_areeb,
        "z_s7": z_s7,
        "s7_p": s7_p.astype(np.float64),
        "d_pub": d_pub,
        "d_priv": d_priv,
    }
    return pack


# ---------------------------------------------------------------------------
# Step 3 — replace 1-fold G26 with gpt2-medium 5f
# ---------------------------------------------------------------------------
def step3_g26_upgrade(y, folds, pack):
    log("======== STEP3: G26 → gpt2-medium 5f surgical upgrade ========")
    gmo, gmt = assemble_gpt2_med(y, folds)
    Gmed_o, Gmed_t = zrk(gmo), zrk(gmt)
    np.save(ART / "gpt2_med_5f_oof_raw.npy", gmo)
    np.save(ART / "gpt2_med_5f_test_raw.npy", gmt)

    base_o = pack["s354_o"]
    # remove published 0.06*(G26-MS) then test blends
    s352_o_u = unitize(pack["s352_o"])
    # variants on OOF
    variants = {}
    for name, score in {
        "pub_G26_only": zrk(pack["s352_o"]) + 0.06 * (pack["G26o"] - pack["MSo"]),
        "med5f_only_0p06": zrk(pack["s352_o"]) + 0.06 * (Gmed_o - pack["MSo"]),
        "blend_0p04G26_0p04med": zrk(pack["s352_o"])
        + 0.04 * (pack["G26o"] - pack["MSo"])
        + 0.04 * (Gmed_o - pack["MSo"]),
        "blend_0p03G26_0p06med": zrk(pack["s352_o"])
        + 0.03 * (pack["G26o"] - pack["MSo"])
        + 0.06 * (Gmed_o - pack["MSo"]),
        "med5f_only_0p09": zrk(pack["s352_o"]) + 0.09 * (Gmed_o - pack["MSo"]),
    }.items():
        fa = fold_aucs(y, score, folds)
        variants[name] = {
            "oof_auc": float(roc_auc_score(y, score)),
            "fold_aucs": [round(x, 6) for x in fa],
            "d_vs_pub_e5": [round((a - b) * 1e5, 2) for a, b in zip(fa, fold_aucs(y, base_o, folds))],
            "pos_folds": int(sum((a - b) > 0 for a, b in zip(fa, fold_aucs(y, base_o, folds)))),
        }
        log(f"  {name}: AUC={variants[name]['oof_auc']:.6f} pos={variants[name]['pos_folds']}/5")

    best = max(variants.items(), key=lambda kv: (kv[1]["pos_folds"], kv[1]["oof_auc"]))
    # test-side upgraded s354 (exclude raw 1-fold G26 or keep small)
    s354_med = zrk(pack["s352"]) + 0.03 * (pack["G26"] - pack["MS"]) + 0.06 * (Gmed_t - pack["MS"])
    s354_med_u = unitize(s354_med)
    out = {
        "variants": variants,
        "best_variant": best[0],
        "Gmed_oof_auc": float(roc_auc_score(y, Gmed_o)),
        "Gmed_fold_aucs": [round(x, 5) for x in fold_aucs(y, Gmed_o, folds)],
        "note": "G26 test is Fold-0 only; preferred test blend keeps small G26 + dominant gpt2_med_5f",
    }
    REPORT["steps"]["step3_g26"] = out
    pack["Gmed_o"] = Gmed_o
    pack["Gmed_t"] = Gmed_t
    pack["s354_med"] = s354_med.astype(np.float32)
    pack["s354_med_u"] = s354_med_u
    pack["s354_o_med"] = (zrk(pack["s352_o"]) + 0.03 * (pack["G26o"] - pack["MSo"]) + 0.06 * (Gmed_o - pack["MSo"])).astype(
        np.float32
    )
    return pack


# ---------------------------------------------------------------------------
# Step 4 — clean 5-fold Stage-B × L7F
# ---------------------------------------------------------------------------
def step4_stageB(y, folds, pack):
    log("======== STEP4: clean 5-fold Stage-B × L7F ========")
    # base margin (LM-inclusive so trees learn interactions)
    z_ref_o = unitize(load_f32(PRED / "gm_v5_overnight/z_ref_old_oof.npy"))
    z_ref_t = unitize(load_f32(PRED / "gm_v5_overnight/z_ref_old_test.npy"))
    d1_o = unitize(load_f32(PRED / "gm_v4_polish/d1_v4_oof.npy"))
    d1_t = unitize(load_f32(PRED / "gm_v4_polish/d1_v4_test.npy"))
    z_base_o = calibrate_logit_to_prevalence(
        0.80 * z_ref_o.astype(np.float64)
        + 0.08 * unitize(pack["L7Fo"]).astype(np.float64)
        + 0.07 * unitize(pack["LMo"]).astype(np.float64)
        + 0.05 * d1_o.astype(np.float64)
    )
    z_base_t = calibrate_logit_to_prevalence(
        0.80 * z_ref_t.astype(np.float64)
        + 0.08 * unitize(pack["L7F"]).astype(np.float64)
        + 0.07 * unitize(pack["LM"]).astype(np.float64)
        + 0.05 * d1_t.astype(np.float64)
    )

    # 5-fold-complete features ONLY (exclude G26_te; GPD optional — include OOF-aligned GPD)
    L42o = unitize(load_f32(PRED / "matrix_v2/P4_L4.2_pure5f_oof.npy"))
    L42t = unitize(load_f32(PRED / "matrix_v2/P4_L4.2_pure5f_test.npy"))
    L43o = unitize(load_f32(PRED / "matrix_v2/P4_L4.3_pure5f_oof.npy"))
    L43t = unitize(load_f32(PRED / "matrix_v2/P4_L4.3_pure5f_test.npy"))
    L45o = unitize(load_f32(PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy"))
    L45t = unitize(load_f32(PRED / "matrix_v2/P4_L4.5_pure5f_test.npy"))
    l46o = unitize(load_f32(Path("/tmp/gm_deep_tech/preds/gm_v9_1_hardened/l46c_top12_5f_oof.npy")))
    l46t = unitize(load_f32(Path("/tmp/gm_deep_tech/preds/gm_v9_1_hardened/l46c_top12_5f_test.npy")))
    Wo, Wt = unitize(pack["W"]), unitize(pack["Wt"])
    # proxies for rare/disp/bnd (KMAT_68 unavailable on disk)
    rbnd_o = unitize(load_f32(PRED / "gm_v4_polish/d3_v4_boundary_oof.npy"))
    rbnd_t = unitize(load_f32(PRED / "gm_v4_polish/d3_v4_boundary_test.npy"))
    rlin_o = unitize(load_f32(PRED / "gm_v4_polish/d3_v4_linear_oof.npy"))
    rlin_t = unitize(load_f32(PRED / "gm_v4_polish/d3_v4_linear_test.npy"))
    rdisp_o = unitize(load_f32(PRED / "gm_v4_polish/d3_v4_oof.npy"))
    rdisp_t = unitize(load_f32(PRED / "gm_v4_polish/d3_v4_test.npy"))

    zL7o, zL7t = unitize(pack["L7Fo"]), unitize(pack["L7F"])
    zMEo, zMEt = unitize(pack["LMo"]), unitize(pack["LM"])
    zGMo, zGMt = unitize(pack["Gmed_o"]), unitize(pack["Gmed_t"])
    zMSo, zMSt = unitize(pack["MSo"]), unitize(pack["MS"])

    ridge_names = [
        "z_L7_5f",
        "z_megalm",
        "z_gpt2_med_5f",
        "z_MS",
        "z_w12",
        "P4_L4.2",
        "P4_L4.3",
        "L4.6C_top12",
        "r_rare_proxy",
        "r_disp_proxy",
        "r_bnd_proxy",
    ]
    Xo_r = np.column_stack(
        [zL7o, zMEo, zGMo, zMSo, Wo, L42o, L43o, l46o, rlin_o, rdisp_o, rbnd_o]
    ).astype(np.float32)
    Xt_r = np.column_stack(
        [zL7t, zMEt, zGMt, zMSt, Wt, L42t, L43t, l46t, rlin_t, rdisp_t, rbnd_t]
    ).astype(np.float32)

    # LGB feature set
    lgb_blocks_o = [
        zL7o,
        zMEo,
        zGMo,
        (zL7o - zMEo),
        (zL7o - zGMo),
        (zL7o - Wo),
        L42o,
        L43o,
        L45o,
        l46o,
        rlin_o,
        rdisp_o,
        rbnd_o,
        d1_o,
    ]
    lgb_blocks_t = [
        zL7t,
        zMEt,
        zGMt,
        (zL7t - zMEt),
        (zL7t - zGMt),
        (zL7t - Wt),
        L42t,
        L43t,
        L45t,
        l46t,
        rlin_t,
        rdisp_t,
        rbnd_t,
        d1_t,
    ]
    # add top shallow LGB dig features from oof-bank (5f-aligned stand-ins for KMAT top35)
    kmat_note = "KMAT_68 matrix not on disk — substituted 8 oof-bank shallow LGB/XGB 5f members + L4.6C/P4"
    bank = RECON / "goodpjw2008__s6e11-oof-bank"
    for name in [
        "lgb_dig_l4",
        "lgb_dig_l7",
        "lgb_diglead_l3",
        "lgb_diglead_l4",
        "lgb_te_l4",
        "xgb_dig_d2",
        "xgb_te_d2",
        "glm_ohk",
    ]:
        lgb_blocks_o.append(unitize(load_f32(bank / f"oof_{name}.npy")))
        lgb_blocks_t.append(unitize(load_f32(bank / f"test_{name}.npy")))
    Xo_l = np.column_stack(lgb_blocks_o).astype(np.float32)
    Xt_l = np.column_stack(lgb_blocks_t).astype(np.float32)
    log(f"  ridge feats={Xo_r.shape[1]}  lgb feats={Xo_l.shape[1]}  RSS={rss_mb():.0f}MB  ({kmat_note})")

    ridge_o = np.zeros(len(y), np.float64)
    ridge_t_acc = []
    lgb_o = np.zeros(len(y), np.float64)
    lgb_t_acc = []
    seeds = [7, 17, 27]

    for f, (tr, te) in enumerate(folds):
        # 90/10 inner split on train fold ONLY (no val_idx early-stop leakage)
        rng = np.random.RandomState(42 + f)
        perm = rng.permutation(tr)
        n_stop = max(1, int(0.10 * len(perm)))
        stop_idx, fit_idx = perm[:n_stop], perm[n_stop:]

        # Ridge on quadratic interactions
        Xtr = quadratic_features(Xo_r[fit_idx])
        Xva = quadratic_features(Xo_r[te])
        Xtq = quadratic_features(Xt_r)
        rr = Ridge(alpha=2500.0, fit_intercept=True)
        rr.fit(Xtr, y[fit_idx].astype(np.float64))
        ridge_o[te] = rr.predict(Xva)
        ridge_t_acc.append(rr.predict(Xtq))
        del Xtr, Xva, Xtq, rr

        # LGB 3-seed bag; fixed n_estimators — NO early stopping on val_idx
        preds_te, preds_test = [], []
        for s in seeds:
            clf = LGBMClassifier(
                boosting_type="gbdt",
                extra_trees=True,
                max_depth=3,
                num_leaves=6,
                colsample_bytree=0.35,
                subsample=0.8,
                subsample_freq=1,
                min_child_samples=350,
                min_sum_hessian_in_leaf=50,
                max_delta_step=0.15,
                reg_lambda=50.0,
                learning_rate=0.02,
                n_estimators=180,
                objective="binary",
                class_weight=None,
                n_jobs=2,
                random_state=s,
                verbose=-1,
            )
            clf.fit(Xo_l[fit_idx], y[fit_idx])
            preds_te.append(clf.predict_proba(Xo_l[te])[:, 1])
            preds_test.append(clf.predict_proba(Xt_l)[:, 1])
            del clf
        lgb_o[te] = np.mean(preds_te, 0)
        lgb_t_acc.append(np.mean(preds_test, 0))
        log(f"  fold {f}: ridgeAUC={roc_auc_score(y[te], ridge_o[te]):.5f} lgbAUC={roc_auc_score(y[te], lgb_o[te]):.5f} RSS={rss_mb():.0f}MB")
        gc.collect()

    ridge_t = np.mean(np.stack(ridge_t_acc, 0), 0)
    lgb_t = np.mean(np.stack(lgb_t_acc, 0), 0)

    # blend ridge+lgb then extract pure innovation vs LM-inclusive base
    z_pred_o = zrk(0.35 * unitize(ridge_o) + 0.65 * unitize(zrk(lgb_o)))
    z_pred_t = zrk(0.35 * unitize(ridge_t) + 0.65 * unitize(zrk(lgb_t)))
    delta_o = unitize(z_pred_o.astype(np.float64) - z_base_o.astype(np.float64))
    delta_t = unitize(z_pred_t.astype(np.float64) - z_base_t.astype(np.float64))
    delta_o, delta_t = pchip_align_fold(delta_o, delta_t, folds)
    disp = float(np.std(delta_t) / (np.std(delta_o) + 1e-18))
    ks = float(ks_2samp(delta_o, delta_t).statistic)
    log(f"  δ_v10 disp={disp:.4f} KS={ks:.6f}")

    # combine with δ1_v4
    u_priv_o = unit(0.55 * delta_o.astype(np.float64) + 0.45 * d1_o.astype(np.float64))
    u_priv_t = unit(0.55 * delta_t.astype(np.float64) + 0.45 * d1_t.astype(np.float64))

    # lift on s354 OOF proxy
    base_fa = fold_aucs(y, pack["s354_o"], folds)
    lifts = {}
    for w in (0.02, 0.04, 0.06, 0.08):
        z = pack["s354_o"].astype(np.float64) + w * u_priv_o.astype(np.float64)
        fa = fold_aucs(y, z, folds)
        lifts[str(w)] = {
            "fold_d_e5": [round((a - b) * 1e5, 2) for a, b in zip(fa, base_fa)],
            "pos": int(sum((a - b) > 0 for a, b in zip(fa, base_fa))),
            "mean_d_e5": round(float(np.mean([(a - b) * 1e5 for a, b in zip(fa, base_fa)])), 2),
            "oof_auc": float(roc_auc_score(y, z)),
        }
        log(f"  s354+{w}*δ_v10: pos={lifts[str(w)]['pos']}/5 meanΔ={lifts[str(w)]['mean_d_e5']}e-5")

    np.save(ART / "delta_v10_l7_5f_oof.npy", delta_o)
    np.save(ART / "delta_v10_l7_5f_test.npy", delta_t)
    np.save(ART / "u_priv_v10_oof.npy", u_priv_o)
    np.save(ART / "u_priv_v10_test.npy", u_priv_t)
    np.save(ART / "ridge_quad_l7_5f_oof.npy", ridge_o.astype(np.float32))
    np.save(ART / "lgb_stageB_l7_5f_oof.npy", lgb_o.astype(np.float32))

    REPORT["steps"]["step4_stageB"] = {
        "kmat_note": kmat_note,
        "ridge_feat_names": ridge_names,
        "lgb_n_features": int(Xo_l.shape[1]),
        "excluded": ["G26_test (1-fold)", "raw lm7_test_f0"],
        "included_GPD": False,
        "disp": disp,
        "KS": ks,
        "delta_oof_auc_as_score": float(roc_auc_score(y, delta_o)),
        "lifts_on_s354_oof": lifts,
        "rss_mb_peak": rss_mb(),
    }
    pack["delta_o"] = delta_o
    pack["delta_t"] = delta_t
    pack["u_priv_o"] = u_priv_o
    pack["u_priv_t"] = u_priv_t
    pack["d1_o"] = d1_o
    pack["d1_t"] = d1_t
    pack["z_base_o"] = z_base_o
    pack["z_base_t"] = z_base_t
    return pack


# ---------------------------------------------------------------------------
# Step 5 — clean synthesis candidates (NO_SUBMIT)
# ---------------------------------------------------------------------------
def step5_synthesis(y, folds, pack, ids):
    log("======== STEP5: clean synthesis candidates (NO_SUBMIT) ========")
    assert NO_SUBMIT

    # z_S7_ours_5fL7: rebuild private engine using L7F 5fold + gpt2_med + strike9.5 + stageB
    st95 = unitize(zrk(load_csv_score(SUB / "sub_strike9p5_axis_0p70_0p10.csv")))
    # private residual direction on test: StageB + gpt2med + strike9.5 — NOT (S7 - areeb) which embeds -lm7_f0
    priv_raw = unit(
        0.45 * pack["u_priv_t"].astype(np.float64)
        + 0.30 * unitize(pack["Gmed_t"]).astype(np.float64)
        + 0.25 * st95.astype(np.float64)
    )
    # Gram-Schmidt against public ladder bases
    bases = [pack["A"], pack["F"], pack["LM"], pack["L7F"], pack["GPD"], pack["G26"], pack["MS"], pack["MLP"], pack["GB"]]
    priv_orth = project_out(priv_raw, bases)
    # orient toward fraud via OOF proxy corr
    if float(np.corrcoef(pack["u_priv_o"], y)[0, 1]) < 0:
        priv_orth = (-priv_orth.astype(np.float64)).astype(np.float32)
        pack["u_priv_o"] = (-pack["u_priv_o"].astype(np.float64)).astype(np.float32)

    s354 = pack["s354"].astype(np.float64)
    s354_u = pack["s354_u"]
    ref_p = (rankdata(s354) - 0.5) / len(s354)  # public law from s354

    # Candidate 1 / 2: orth injection at exact Spearman distances
    cands = {}
    for slot, dist in [("slot1", 0.12e-4), ("slot2", 0.28e-4)]:
        w, got = weight_for_dist(s354_u, priv_orth, dist)
        score = s354_u.astype(np.float64) + w * unit(priv_orth).astype(np.float64)
        probs = force_exact_law_probs(ref_p, score)
        path = SUB / f"sub_v10_{slot}_s354pub_plus_clean5f_stageB_{dist*1e4:.2f}e4_NOSUBMIT.csv".replace(
            "0.12", "0p12"
        ).replace("0.28", "0p28")
        # fix naming explicitly
        if slot == "slot1":
            path = SUB / "sub_v10_slot1_s354pub_plus_clean5f_stageB_0p12e4_NOSUBMIT.csv"
        else:
            path = SUB / "sub_v10_slot2_s354pub_plus_clean5f_stageB_0p28e4_NOSUBMIT.csv"
        write_nosubmit(path, ids, probs)
        cands[slot] = {
            "path": str(path),
            "target_dist": dist,
            "got_dist": got,
            "weight": w,
            "mean": float(probs.mean()),
            "ks_vs_s354": float(ks_2samp(ref_p, probs).statistic),
            "spearman_vs_s354": float(spearmanr(s354, probs).correlation),
            "sha16": sha256(path)[:16],
        }
        log(f"  {slot}: dist={got:.3e} (target {dist:.3e}) mean={probs.mean():.6f} → {path.name}")

    # Candidate 3: midpoint convex of s354_pub and S7_ours_5fL7
    # Build S7_ours_5fL7 on test: start from S7, replace LM7_f0 contamination by moving toward L7F+med+stageB
    z_s7 = pack["z_s7"].astype(np.float64)
    # Clean private upgrade in S7 neighborhood using L7F 5fold (never lm7_test_f0 / never S7−areeb)
    s7_clean = unitize(
        z_s7
        + 0.08 * unitize(pack["L7F"]).astype(np.float64)
        + 0.06 * unitize(pack["Gmed_t"]).astype(np.float64)
        + 0.05 * pack["u_priv_t"].astype(np.float64)
        + 0.04 * st95.astype(np.float64)
    )
    mid = unitize(0.50 * s354_u.astype(np.float64) + 0.50 * s7_clean.astype(np.float64))
    # small stageB boost
    mid = unitize(mid.astype(np.float64) + 0.04 * pack["u_priv_t"].astype(np.float64))
    probs3 = force_exact_law_probs(ref_p, mid)
    path3 = SUB / "sub_v10_slot3_midpoint_s354pub_and_S7ours5fL7_NOSUBMIT.csv"
    write_nosubmit(path3, ids, probs3)
    cands["slot3"] = {
        "path": str(path3),
        "mean": float(probs3.mean()),
        "ks_vs_s354": float(ks_2samp(ref_p, probs3).statistic),
        "spearman_vs_s354": float(spearmanr(s354, probs3).correlation),
        "spearman_vs_S7": float(spearmanr(pack["s7_p"], probs3).correlation),
        "sha16": sha256(path3)[:16],
        "note": "rank-logit midpoint s354_pub ↔ clean S7_5fL7 (no areeb−S7 raw delta)",
    }
    log(f"  slot3 midpoint → {path3.name}")

    # Candidate 4: daoviet — only verified 0.95351 + optional orth GPD already inside s354.
    # Since 0.95368 is unverified, synthesize s354 + tiny extra GPD residual orth (if still orthogonal)
    gpd_extra = project_out(pack["GPD"] - pack["F"], bases)
    score4 = s354_u.astype(np.float64) + 0.02 * unit(gpd_extra).astype(np.float64)
    # also mix verified daoviet 095351 file lightly
    d351 = unitize(zrk(load_csv_score(RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex/submission_095351.csv")))
    score4 = unitize(0.85 * score4 + 0.15 * d351.astype(np.float64))
    probs4 = force_exact_law_probs(ref_p, score4)
    path4 = SUB / "sub_v10_slot4_daoviet_095368_verified_synthesis_NOSUBMIT.csv"
    write_nosubmit(path4, ids, probs4)
    cands["slot4"] = {
        "path": str(path4),
        "mean": float(probs4.mean()),
        "ks_vs_s354": float(ks_2samp(ref_p, probs4).statistic),
        "spearman_vs_s354": float(spearmanr(s354, probs4).correlation),
        "sha16": sha256(path4)[:16],
        "note": (
            "0.95368 NOT verified — slot uses verified daoviet submission_095351 + mild GPD orth on s354_pub. "
            "Do NOT treat as a 0.95368 clone."
        ),
    }
    log(f"  slot4 daoviet-verified synth → {path4.name}")

    # OOF benchmark table for private direction on s354_o
    oof_rows = []
    for w in (0.0, 0.02, 0.04, 0.06):
        z = pack["s354_o"].astype(np.float64) + w * pack["u_priv_o"].astype(np.float64)
        fa = fold_aucs(y, z, folds)
        oof_rows.append({"w": w, "auc": float(roc_auc_score(y, z)), "folds": [round(x, 6) for x in fa]})

    # Empirical LB surface: distance-to-s354 vs known scores for cached subs
    lb_z = np.load(PRED / "gm_v4_polish/_cache_lb_z.npz")
    lb_sc = np.load(PRED / "gm_v4_polish/_cache_lb_scores.npy", allow_pickle=True)
    surface = []
    for name, score in lb_sc:
        if name not in lb_z.files:
            continue
        z = unitize(lb_z[name])
        surface.append(
            {
                "name": str(name),
                "lb": float(score),
                "dist_to_s354": spearman_dist(s354_u, z),
                "dist_to_areeb": spearman_dist(pack["z_areeb"], z),
            }
        )
    surface = sorted(surface, key=lambda r: -r["lb"])

    REPORT["steps"]["step5_synth"] = {
        "candidates": cands,
        "oof_s354_plus_priv": oof_rows,
        "lb_surface_top": surface[:12],
        "contamination_guard": "Never added raw (z_S7 - z_areeb); L7F 5fold used throughout",
    }
    return cands


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def write_report():
    log("======== WRITE REPORT ========")
    md = []
    md.append("# GM v10 Hardened — 0.95369 Breaker Report (`NO_SUBMIT=True`)\n\n")
    md.append(f"Generated: {REPORT['generated_at']} · elapsed={REPORT.get('elapsed_s', 0):.1f}s · peak RSS≈{REPORT.get('rss_mb', 0):.0f} MB\n\n")
    md.append("## Step 0 — Pod\n\n")
    p0 = REPORT["steps"]["step0_pod"]
    md.append(f"- Requested terminate: `{p0['requested_terminate']}` → **{p0['result']}**\n")
    md.append(f"- Final spend est: {p0['final_spend_est_usd']} · active $/hr for rsoju: **$0.00**\n")
    md.append(f"- Untouched: {', '.join(f'`{x}`' for x in p0['untouched'])}\n")
    md.append(f"- Rationale: {p0['rationale']}\n\n")

    md.append("## Step 1 — daoviet `0.95368` / goodpjw `0.95354` forensics\n\n")
    d = REPORT["steps"]["step1_forensics"]["daoviet"]
    g = REPORT["steps"]["step1_forensics"]["goodpjw"]
    md.append("### daoviet\n\n")
    md.append(f"- Header verified LB: **{d['verified_public_header']}**\n")
    md.append(f"- Ablation table claims: Public **{d['ablation_table_public_lb_stage10']}** / Local CV **{d['ablation_table_local_cv_stage10']}**\n")
    md.append(f"- **Verdict:** {d['verdict']}\n")
    md.append(f"- Reconstruct formula (fallback): `{d['reconstruct_formula_if_no_prebake']}`\n")
    md.append(f"- Dataset files: {d['dataset_files']}\n")
    md.append(f"- Combines with goodpjw 0.95354? {d['combines_with_goodpjw_095354']}\n\n")
    md.append("### goodpjw 0.95354\n\n")
    md.append(f"- Formula: `{g['s354_formula']}`\n")
    md.append(f"- L7: **{g['L7F']}**\n")
    md.append(f"- G26 test: **{g['G26_test']}**\n\n")

    md.append("## Step 2 — Asset & split audit\n\n")
    md.append("| asset | shape | OOF AUC | fold AUCs | notes |\n|---|---|---:|---|---|\n")
    for k, meta in REPORT["steps"]["step2_audit"]["assets"].items():
        if "oof_auc" in meta:
            md.append(
                f"| {k} | {meta['shape']} | {meta['oof_auc']:.6f} | {meta['fold_aucs']} | pattern_ok={meta.get('fold_pattern_ok')} |\n"
            )
        else:
            md.append(f"| {k} | {meta['shape']} | — | — | sha={meta.get('sha16')} |\n")
    md.append("\n")
    md.append(f"- **G26:** {REPORT['steps']['step2_audit']['G26']}\n")
    md.append(f"- **GPD:** {REPORT['steps']['step2_audit']['GPD']}\n")
    r = REPORT["steps"]["step2_recon"]
    md.append("\n### s354 reconstruction\n\n")
    md.append(f"- Spearman dist(s354, S7) = `{r['spearman_dist_s354_vs_S7']:.6e}`\n")
    md.append(f"- cos(Δ_pub, Δ_priv) = `{r['cos_delta_pub_vs_priv']:.4f}`\n")
    md.append(f"- s354 OOF proxy AUC = `{r['s354_oof_auc']:.6f}` folds={r['s354_oof_fold_aucs']}\n")
    md.append(f"- G26 test Fold-0 only: **{r['G26_is_test_fold0_only']}**\n\n")

    md.append("## Step 3 — G26 → gpt2-medium 5f\n\n")
    s3 = REPORT["steps"]["step3_g26"]
    md.append(f"- gpt2_med_5f OOF AUC={s3['Gmed_oof_auc']:.6f} folds={s3['Gmed_fold_aucs']}\n")
    md.append(f"- Best OOF variant: **{s3['best_variant']}**\n\n")
    md.append("| variant | OOF AUC | pos/5 | Δe-5 vs pub |\n|---|---:|---:|---|\n")
    for n, v in s3["variants"].items():
        md.append(f"| {n} | {v['oof_auc']:.6f} | {v['pos_folds']} | {v['d_vs_pub_e5']} |\n")
    md.append(f"\nNote: {s3['note']}\n\n")

    md.append("## Step 4 — Stage-B × L7_5fold\n\n")
    s4 = REPORT["steps"]["step4_stageB"]
    md.append(f"- {s4['kmat_note']}\n")
    md.append(f"- LGB features: {s4['lgb_n_features']} · excluded: {s4['excluded']}\n")
    md.append(f"- δ_v10 disp={s4['disp']:.4f} KS={s4['KS']:.6f}\n\n")
    md.append("| w | pos/5 | mean Δe-5 | OOF AUC |\n|---:|---:|---:|---:|\n")
    for w, v in s4["lifts_on_s354_oof"].items():
        md.append(f"| {w} | {v['pos']} | {v['mean_d_e5']} | {v['oof_auc']:.6f} |\n")
    md.append("\n")

    md.append("## Step 5 — Candidates (`NOSUBMIT`)\n\n")
    s5 = REPORT["steps"]["step5_synth"]
    for slot, c in s5["candidates"].items():
        md.append(f"### {slot}\n")
        for k, v in c.items():
            md.append(f"- **{k}:** `{v}`\n")
        md.append("\n")
    md.append("### OOF s354 + private\n\n")
    md.append("| w | AUC | folds |\n|---:|---:|---|\n")
    for row in s5["oof_s354_plus_priv"]:
        md.append(f"| {row['w']} | {row['auc']:.6f} | {row['folds']} |\n")
    md.append("\n### Empirical LB surface (cached)\n\n")
    md.append("| name | LB | dist→s354 | dist→areeb |\n|---|---:|---:|---:|\n")
    for row in s5["lb_surface_top"]:
        md.append(f"| {row['name']} | {row['lb']:.5f} | {row['dist_to_s354']:.3e} | {row['dist_to_areeb']:.3e} |\n")
    md.append(f"\n**Contamination guard:** {s5['contamination_guard']}\n\n")
    md.append("---\n`NO_SUBMIT=True` — no Kaggle upload performed.\n")

    path = LOG / "GM_V10_HARDENED_095369_REPORT.md"
    path.write_text("".join(md))
    (WORK / "logs/overnight/GM_V10_HARDENED_095369_REPORT.md").write_text("".join(md))
    (ART / "v10_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    log(f"wrote {path}")
    return path


def main():
    assert NO_SUBMIT is True
    log("======== GM v10 HARDENED 095369 BREAKER (NO_SUBMIT) ========")
    REPORT["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    step0_pod_status()
    step1_forensics()

    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = np.empty(len(y), np.int8)
    for f, (_, te) in enumerate(folds):
        fold_id[te] = f
    # confirm oof-bank folds
    fc = pd.read_csv(RECON / "goodpjw2008__s6e11-oof-bank/folds.csv")["fold"].to_numpy()
    assert (fc == fold_id).all(), "folds.npy/SKF42 mismatch vs oof-bank"
    ids = pd.read_csv(VOL / "data/sample_submission.csv")["id"].to_numpy()

    pack = step2_audit(y, folds, fold_id)
    pack = step3_g26_upgrade(y, folds, pack)
    pack = step4_stageB(y, folds, pack)
    step5_synthesis(y, folds, pack, ids)

    REPORT["elapsed_s"] = time.time() - T0
    REPORT["rss_mb"] = rss_mb()
    write_report()
    log(f"DONE NO_SUBMIT=True elapsed={REPORT['elapsed_s']:.1f}s RSS={REPORT['rss_mb']:.0f}MB")


if __name__ == "__main__":
    main()
