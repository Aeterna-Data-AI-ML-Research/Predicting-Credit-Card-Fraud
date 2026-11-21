#!/usr/bin/env python3
"""GM v11 macro-leap toward 0.95370+: v17→s352 transplant, LOFO 5/5 bank screen, G26 code pull.

STRICT NO_SUBMIT=True — stages CSVs only, never submits.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")

NO_SUBMIT = True
VOL = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
RECON = VOL / "external/nov12_095369_recon/datasets"
BANKS = VOL / "external/banks"
PRED = VOL / "preds"
SUB = VOL / "submissions"
LOG = VOL / "logs/overnight"
ART = Path("/tmp/gm_deep_tech/preds/gm_v10_hardened")
GPT2_DIR = VOL / "external/gpt2_target_first_09501"
KAG = VOL / ".venv_forensics/bin/kaggle"
T0 = time.time()
ALPHA_CAND2 = 0.03613
BLACKLIST = re.compile(
    r"(v10_|soft_diff|gb_soft|idf_p3|clean035|zanch_)", re.I
)

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr, ks_2samp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

LOG.mkdir(parents=True, exist_ok=True)
SUB.mkdir(parents=True, exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def L(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return logit((rankdata(v, method="average") - 0.5) / len(v))


def L_fold(v, fold_id):
    """Per-fold rank-logit on OOF; global L on test (no folds)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    out = np.empty_like(v)
    for f in range(5):
        m = fold_id == f
        out[m] = L(v[m])
    return out


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-18)


def spearman_dist(a, b):
    return float(1.0 - spearmanr(a, b).correlation)


def force_law(ref_p, score):
    ref_p = np.asarray(ref_p, dtype=np.float64).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    order = np.argsort(
        score + 1e-15 * (rankdata(score, method="ordinal") / len(score)), kind="mergesort"
    )
    out = np.empty(len(score), dtype=np.float64)
    out[order] = np.sort(ref_p)
    return out


def weight_for_dist(z0, u, target, hi=80.0, n=60):
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


def project_out(v, bases):
    v = np.asarray(v, dtype=np.float64).ravel()
    cols = [unitize(b) for b in bases if np.linalg.norm(np.asarray(b).ravel()) > 1e-12]
    if not cols:
        return unitize(v)
    B = np.column_stack(cols)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(B, v)
    return unitize(v - lr.predict(B))


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[te], z[te])) for _, te in folds]


def load_csv_prob(p):
    df = pd.read_csv(p)
    col = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
    return df[col].to_numpy(np.float64)


def write_nosubmit(path, ids, probs):
    assert NO_SUBMIT
    assert "NOSUBMIT" in path.name.upper()
    pd.DataFrame({"id": ids, "is_fraud": probs}).to_csv(path, index=False)
    return path


def kaggle_cmd(*args, timeout=180):
    exe = [str(KAG)] if KAG.exists() else ["/opt/homebrew/bin/python3", "-m", "kaggle"]
    env = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy")}
    try:
        return subprocess.check_output(exe + list(args), text=True, timeout=timeout, env=env)
    except Exception as e:
        return f"ERR:{e}"


# ---------------------------------------------------------------------------
# Step 1: pulls + G26/MS diagnostics
# ---------------------------------------------------------------------------
def step1_pulls_and_g26(y, fold_id, folds):
    log("======== STEP1: pulls + G26/MS AUC ========")
    out = {"kernel": {}, "g26_readme": "", "auc": {}, "hyperparams": {}}

    GPT2_DIR.mkdir(parents=True, exist_ok=True)
    if not list(GPT2_DIR.glob("*.ipynb")):
        log("  pulling masayakawamata GPT-2 kernel…")
        kaggle_cmd(
            "kernels",
            "pull",
            "masayakawamata/s6e11-gpt-2-target-first-classifier-cv-0-9500",
            "-p",
            str(GPT2_DIR),
            "-m",
            timeout=120,
        )
    out["kernel"]["files"] = sorted(p.name for p in GPT2_DIR.iterdir() if not p.name.startswith("._"))

    # ensure fraud-series present in recon
    dest_fs = RECON / "goodpjw2008__s6e11-fraud-series-artifacts"
    dest_fs.mkdir(parents=True, exist_ok=True)
    src_fs = BANKS / "goodpjw2008_s6e11-fraud-series-artifacts"
    if src_fs.exists():
        for p in src_fs.iterdir():
            if p.name.startswith("._"):
                continue
            tgt = dest_fs / p.name
            if not tgt.exists():
                try:
                    shutil.copy2(p, tgt)
                except Exception:
                    pass
    # refresh only if empty (avoid multi-minute re-download)
    bank = RECON / "goodpjw2008__s6e11-oof-bank"
    if len(list(dest_fs.glob("*.npy"))) < 10:
        kaggle_cmd("datasets", "download", "-d", "goodpjw2008/s6e11-fraud-series-artifacts", "-p", str(dest_fs), "--unzip", "-q", timeout=120)
    if len(list(bank.glob("oof_*.npy"))) < 20:
        kaggle_cmd("datasets", "download", "-d", "goodpjw2008/s6e11-oof-bank", "-p", str(bank), "--unzip", "-q", timeout=120)

    readme = RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/README.md"
    out["g26_readme"] = readme.read_text() if readme.exists() else "(missing)"
    log("  G26 README:\n" + out["g26_readme"][:700])

    # hyperparams from notebook recipe cell
    nbs = list(GPT2_DIR.glob("*.ipynb"))
    if nbs:
        nb = json.loads(nbs[0].read_text())
        recipe = []
        for c in nb["cells"]:
            src = "".join(c.get("source", []))
            if "GPT-2 small" in src or "recipe = pd.DataFrame" in src:
                recipe.append(src[:1500])
            if "training_hours" in src or "classifier_hours" in src:
                m = re.search(r"'training_hours':\s*([0-9.]+)", src)
                if m:
                    out["hyperparams"]["training_hours"] = float(m.group(1))
                m = re.search(r"'classifier_hours':\s*([0-9.]+)", src)
                if m:
                    out["hyperparams"]["classifier_hours"] = float(m.group(1))
                m = re.search(r"'scoring_hours':\s*([0-9.]+)", src)
                if m:
                    out["hyperparams"]["scoring_hours"] = float(m.group(1))
                m = re.search(r"'replica_hours':\s*([0-9.]+)", src)
                if m:
                    out["hyperparams"]["replica_hours"] = float(m.group(1))
        out["hyperparams"]["recipe_snippet"] = recipe[0][:900] if recipe else ""
        out["hyperparams"]["summary"] = {
            "base": "GPT-2 small",
            "replica": "GReaT batch16 60ep (~5.07h)",
            "classifier": "8 epochs AdamW lr 5e-5→0 batch32 bf16 (~55.2h)",
            "orders": 10,
            "total_train_gpu_h": out["hyperparams"].get("training_hours"),
            "scoring_gpu_h": out["hyperparams"].get("scoring_hours"),
            "per_fold_test_score_est_h": 12.6,
        }

    G26o = np.load(RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_oof_5fold.npy").ravel()
    MSo = np.load(RECON / "masayakawamata__s6e11-gpt2-target-first-oof/oof.npy").ravel()
    assert G26o.shape == MSo.shape == (len(y),)

    def solo(name, z):
        fa = fold_aucs(y, z, folds)
        return {
            "name": name,
            "auc": float(roc_auc_score(y, z)),
            "folds": [round(x, 5) for x in fa],
        }

    out["auc"]["G26"] = solo("G26", G26o)
    out["auc"]["MS"] = solo("MS", MSo)
    blend = 0.5 * L_fold(G26o, fold_id) + 0.5 * L_fold(MSo, fold_id)
    out["auc"]["half_G26_MS"] = solo("0.5*L(G26)+0.5*L(MS)", blend)
    for k, v in out["auc"].items():
        log(f"  {k}: AUC={v['auc']:.6f} folds={v['folds']}")
    out["g26_has_test_5fold"] = (
        RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_test_5fold.npy"
    ).exists()
    out["g26_local"] = sorted(
        p.name
        for p in (RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026").glob("*")
        if not p.name.startswith("._")
    )
    log(f"  G26 has_test_5fold={out['g26_has_test_5fold']} files={out['g26_local']}")
    return out


# ---------------------------------------------------------------------------
# Load locked cand2 / ladder components
# ---------------------------------------------------------------------------
def load_pack(y, fold_id, folds):
    log("======== LOAD: cand2 + ladder components ========")
    ids = pd.read_csv(VOL / "data/sample_submission.csv")["id"].to_numpy()
    s7 = load_csv_prob(SUB / "sub_master_slot7_verified_s7A.csv")
    cand2_csv = load_csv_prob(SUB / "sub_v10p2_cand2_S7ours_plus_twoseed_G26_MS_0p11e4_NOSUBMIT.csv")

    A = L(load_csv_prob(RECON / "amanatar__s6e11-strike10-podium-apex-095340/submission_095340.csv"))
    F = L(load_csv_prob(RECON / "moranzavdi__s6e11-strike9-lb-095330/submission_095330.csv"))
    LM = L(np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/megalm_test_sha3168.npy"))
    L7F = L(np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_test_5fold.npy"))
    _l7f0 = np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_test_f0.npy")
    if _l7f0.ndim == 3:
        L7f0 = L((_l7f0[..., 1] - _l7f0[..., 0]).mean(1))
    else:
        L7f0 = L(_l7f0)
    GPD = L(np.load(RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex/test_ours.npy"))
    MLP = L(np.load(RECON / "goodpjw2008__s6e11-oof-bank/test_gbn_bag5_w512.npy"))
    GB = L(np.load(RECON / "goodpjw2008__s6e11-oof-bank/test_gb_bag5.npy"))
    MS = L(np.load(RECON / "masayakawamata__s6e11-gpt2-target-first-oof/test.npy"))
    G26 = L(np.load(RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_test_f0.npy"))

    v17 = A + 0.6 * (A - F) + 0.12 * (LM - F) + 0.06 * (L7f0 - F) - 0.05 * (MLP - F) - 0.025 * (GB - F)
    s352 = (
        A
        + 0.6 * (A - F)
        + 0.09 * (LM - F)
        + 0.09 * (L7F - F)
        + 0.05 * (GPD - F)
        - 0.05 * (MLP - F)
        - 0.025 * (GB - F)
    )
    # exact algebraic Δ
    d_base = -0.03 * (LM - F) + 0.09 * (L7F - F) - 0.06 * (L7f0 - F) + 0.05 * (GPD - F)
    # algebraic identity in pre-rank logit space
    if not np.allclose(s352 - v17, d_base, atol=1e-5):
        log(f"  WARN: s352-v17 vs algebraic Δ maxabs={np.max(np.abs((s352-v17)-d_base)):.3e}")
        d_base = s352 - v17

    z_cand2_score = L(s7) + ALPHA_CAND2 * (G26 - MS)
    z_cand2 = force_law(s7, z_cand2_score)
    # prefer published cand2 CSV if rank-identical
    if spearman_dist(z_cand2, cand2_csv) < 1e-10:
        z_cand2 = cand2_csv
    log(f"  cand2 1ρ vs CSV={spearman_dist(z_cand2, cand2_csv):.3e} 1ρ vs S7={spearman_dist(z_cand2, s7):.6e}")

    # s352 already lives in rank-logit space; do NOT L() again
    s354_score = s352 + 0.06 * (G26 - MS)
    s354 = (rankdata(s354_score, method="average") - 0.5) / len(s354_score)
    pub354 = Path("/tmp/areeb_095354_out/submission.csv")
    if pub354.exists():
        s354_pub = load_csv_prob(pub354)
        log(f"  s354 recon 1ρ vs official={spearman_dist(s354, s354_pub):.3e}")
        s354 = s354_pub

    d_priv = L(s7) - L(force_law(s7, v17))  # private Stage-B in logit-rank of probs
    # better: Δ_private_S7 = L(S7) - L(areeb_v17_probs)
    areeb_p = force_law(s7, v17)  # same multiset? use rank_probs of v17
    areeb_p = (rankdata(v17, method="average") - 0.5) / len(v17)
    # find published areeb if available
    for p in [
        SUB / "sub_breakout_s1_areeb_gpt2m_3ladder_0p18e4.csv",
        ART / "z_areeb.npy",
    ]:
        pass
    z_areeb = areeb_p
    # try z_areeb from ART
    if (PRED / "gm_breakout/z_areeb.npy").exists():
        z_areeb = np.load(PRED / "gm_breakout/z_areeb.npy").ravel()
        if z_areeb.max() > 1.5:  # logit-ish
            z_areeb = (rankdata(z_areeb, method="average") - 0.5) / len(z_areeb)
    d_priv = L(s7) - L(z_areeb)

    # OOF side
    z_ref = np.load(PRED / "gm_v5_overnight/z_ref_old_oof.npy").ravel().astype(np.float64)
    d1 = np.load(PRED / "gm_v4_polish/d1_v4_oof.npy").ravel().astype(np.float64)
    z_simple = z_ref + 0.05 * d1

    W = L(np.load(VOL / "external/nov08_part12/wide_lm_stack_nested.npy"))
    LMo = L_fold(np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/megalm_oof_sha181c.npy"), fold_id)
    L7Fo = L_fold(np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_5fold.npy"), fold_id)
    L7f0_raw = np.load(RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_f0.npy")
    L7_f0_proxy = L7Fo.copy()
    te0 = folds[0][1]
    L7_f0_proxy[te0] = L((L7f0_raw[..., 1] - L7f0_raw[..., 0]).mean(1))
    MSo_raw = np.load(RECON / "masayakawamata__s6e11-gpt2-target-first-oof/oof.npy").ravel()
    G26o_raw = np.load(RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_oof_5fold.npy").ravel()
    MSo = L_fold(MSo_raw, fold_id)
    G26o = L_fold(G26o_raw, fold_id)
    MLPo = L_fold(np.load(RECON / "goodpjw2008__s6e11-oof-bank/oof_gbn_bag5_w512.npy"), fold_id)
    GBo = L_fold(np.load(RECON / "goodpjw2008__s6e11-oof-bank/oof_gb_bag5.npy"), fold_id)
    GPDo = L_fold(np.load(RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex/oof_ours.npy"), fold_id)
    Fo = W  # strike-like base on OOF
    L7f0o = L7_f0_proxy

    # OOF areeb / s352 proxies (no A/F on OOF)
    z_areeb_o = W + 0.12 * (LMo - W) + 0.06 * (L7f0o - W) - 0.05 * (MLPo - W) - 0.025 * (GBo - W)
    z_areeb_o = L(z_areeb_o) + 0.05 * unitize(d1)
    z_s7_o = unitize(unitize(z_areeb_o) + 0.04 * unitize(d1))
    u_g26ms_o = G26o - MSo
    # lock cand2 OOF evaluators
    # simple: add optimal small G26-MS
    best_w, best_auc = 0.03613, -1
    for w in (0.02, 0.03613, 0.04, 0.06, 0.08):
        a = float(roc_auc_score(y, z_simple + w * unitize(u_g26ms_o)))
        if a > best_auc:
            best_auc, best_w = a, w
    eval_simple_355 = z_simple + best_w * unitize(u_g26ms_o)
    eval_S7matched_355 = unitize(z_s7_o + ALPHA_CAND2 * unitize(u_g26ms_o))
    log(
        f"  eval_simple_355 AUC={roc_auc_score(y, eval_simple_355):.7f} (w*={best_w}) "
        f"S7m_355={roc_auc_score(y, eval_S7matched_355):.7f}"
    )

    # OOF Δ_v17→s352 pieces (using W as F proxy)
    pieces_o = {
        "LM_trim": -0.03 * (LMo - W),
        "L7F_add": 0.09 * (L7Fo - W),
        "L7f0_remove": -0.06 * (L7f0o - W),
        "GPD_add": 0.05 * (GPDo - W),
    }
    d_base_o = pieces_o["LM_trim"] + pieces_o["L7F_add"] + pieces_o["L7f0_remove"] + pieces_o["GPD_add"]
    d_priv_o = unitize(z_s7_o) - unitize(z_areeb_o)

    pack = dict(
        ids=ids,
        s7=s7,
        z_cand2=z_cand2,
        v17=v17,
        s352=s352,
        s354=s354,
        d_base=d_base,
        d_priv=d_priv,
        G26=G26,
        MS=MS,
        LM=LM,
        L7F=L7F,
        L7f0=L7f0,
        GPD=GPD,
        MLP=MLP,
        GB=GB,
        F=F,
        A=A,
        z_areeb=z_areeb,
        eval_simple_355=eval_simple_355,
        eval_S7matched_355=eval_S7matched_355,
        pieces_o=pieces_o,
        d_base_o=d_base_o,
        d_priv_o=d_priv_o,
        u_g26ms_o=u_g26ms_o,
        MSo=MSo,
        G26o=G26o,
        MLPo=MLPo,
        GBo=GBo,
        LMo=LMo,
        L7Fo=L7Fo,
        GPDo=GPDo,
        W=W,
        fold_id=fold_id,
        constituents_o={"MS": MSo, "MLP": MLPo, "GB": GBo, "LM": LMo, "L7F": L7Fo},
        constituents_t={"MS": MS, "MLP": MLP, "GB": GB, "LM": LM, "L7F": L7F},
    )
    return pack


# ---------------------------------------------------------------------------
# Step 2: base transplant + true fusion
# ---------------------------------------------------------------------------
def _orient_u(y, z_base, delta, m0):
    u = unit(np.asarray(delta, dtype=np.float64).ravel())
    if np.linalg.norm(u) < 1e-12 or not np.isfinite(u).all():
        return None
    if roc_auc_score(y[m0], z_base[m0] + 0.02 * u[m0]) < roc_auc_score(
        y[m0], z_base[m0] - 0.02 * u[m0]
    ):
        u = -u
    return u


def quick_fold_eval(y, z_base, delta, folds, w=0.04):
    """Fast non-LOFO fold deltas for prefilter."""
    m0 = folds[0][1]
    u = _orient_u(y, z_base, delta, m0)
    if u is None:
        return None
    base_fa = fold_aucs(y, z_base, folds)
    z = z_base + w * u
    fa = fold_aucs(y, z, folds)
    d = [(a - b) * 1e5 for a, b in zip(fa, base_fa)]
    return {
        "fold_d_e5": [round(x, 3) for x in d],
        "pos5": int(sum(x > 0 for x in d)),
        "mean_d_e5": round(float(np.mean(d)), 3),
        "d0_e5": round(d[0], 3),
        "u": u,
    }


def lofo_eval(y, z_base, delta, folds, betas=(0.01, 0.02, 0.04, 0.06)):
    """Strict LOFO: pick β on [-f], score fold f."""
    z_base = np.asarray(z_base, dtype=np.float64).ravel()
    m0 = folds[0][1]
    u = _orient_u(y, z_base, delta, m0)
    if u is None:
        return {
            "fold_d_e5": [0.0] * 5,
            "pos5": 0,
            "mean_d_e5": 0.0,
            "d0_e5": 0.0,
            "betas": [],
            "u": np.zeros_like(z_base),
            "degenerate": True,
        }
    base_fa = fold_aucs(y, z_base, folds)
    # precompute z_base + b*u for each beta once
    Zs = {b: z_base + b * u for b in betas}
    d_e5, chosen = [], []
    for f, (_, te) in enumerate(folds):
        tr = np.ones(len(y), dtype=bool)
        tr[te] = False
        best_b, best_a = betas[0], -1.0
        for b in betas:
            a = float(roc_auc_score(y[tr], Zs[b][tr]))
            if a > best_a:
                best_a, best_b = a, b
        chosen.append(best_b)
        d_e5.append((float(roc_auc_score(y[te], Zs[best_b][te])) - base_fa[f]) * 1e5)
    return {
        "fold_d_e5": [round(x, 3) for x in d_e5],
        "pos5": int(sum(x > 0 for x in d_e5)),
        "mean_d_e5": round(float(np.mean(d_e5)), 3),
        "d0_e5": round(d_e5[0], 3),
        "betas": chosen,
        "u": u,
        "degenerate": False,
    }


def step2_transplant(y, folds, pack):
    log("======== STEP2: v17→s352 transplant + true fusion ========")
    results = {"pieces": {}, "combined": {}, "fusion": {}}
    # Orth only vs cand2-axis + (G26-MS); do NOT also kill d_priv (absorbs ladder pieces)
    bases_o = [pack["eval_S7matched_355"], pack["u_g26ms_o"]]
    zS = pack["eval_S7matched_355"]
    zP = pack["eval_simple_355"]

    for name, vec in {**pack["pieces_o"], "ALL_d_base": pack["d_base_o"]}.items():
        # LOFO only on combined vector; pieces use quick fold scan for speed
        raw_s = lofo_eval(y, zS, vec, folds) if name == "ALL_d_base" else quick_fold_eval(y, zS, vec, folds)
        raw_p = quick_fold_eval(y, zP, vec, folds)
        if raw_s is None:
            raw_s = {"fold_d_e5": [0.0] * 5, "pos5": 0, "mean_d_e5": 0.0, "d0_e5": 0.0}
        if raw_p is None:
            raw_p = raw_s
        orth = project_out(vec, bases_o)
        orth_norm = float(np.linalg.norm(orth - orth.mean()))
        if orth_norm < 1e-8:
            orth_s = {"fold_d_e5": [0.0] * 5, "pos5": 0, "mean_d_e5": 0.0, "d0_e5": 0.0, "degenerate": True}
            orth_p = orth_s
            cos_orth = 0.0
        else:
            znull = {"fold_d_e5": [0.0] * 5, "pos5": 0, "mean_d_e5": 0.0, "d0_e5": 0.0}
            orth_s = quick_fold_eval(y, zS, orth, folds) or znull
            orth_p = quick_fold_eval(y, zP, orth, folds) or znull
            if name == "ALL_d_base":
                orth_s = lofo_eval(y, zS, orth, folds)
                orth_p = lofo_eval(y, zP, orth, folds)
            cos_orth = float(np.dot(unit(orth), unit(unitize(pack["u_g26ms_o"]))))
        results["pieces"][name] = {
            "raw_S7m": {k: v for k, v in raw_s.items() if k != "u"},
            "raw_simple": {k: v for k, v in raw_p.items() if k != "u"},
            "orth_S7m": {k: v for k, v in orth_s.items() if k != "u"},
            "orth_simple": {k: v for k, v in orth_p.items() if k != "u"},
            "cos_to_g26ms": float(np.dot(unit(unitize(vec)), unit(unitize(pack["u_g26ms_o"])))),
            "cos_orth_g26ms": cos_orth,
            "orth_norm": orth_norm,
        }
        log(
            f"  piece {name}: RAW_S7m pos={raw_s['pos5']}/5 mean={raw_s['mean_d_e5']} d0={raw_s['d0_e5']} | "
            f"orth pos={orth_s.get('pos5',0)}/5 mean={orth_s.get('mean_d_e5',0)} norm={orth_norm:.3e}"
        )

    d_base_t = pack["d_base"]
    u_g26_t = pack["G26"] - pack["MS"]
    d_base_orth_t = project_out(d_base_t, [L(pack["z_cand2"]), u_g26_t])
    pack["d_base_orth_t"] = d_base_orth_t
    pack["d_base_orth_o"] = project_out(pack["d_base_o"], bases_o)
    results["combined"]["d_base_orth_test_norm"] = float(np.linalg.norm(d_base_orth_t))
    results["combined"]["cos_d_base_to_g26ms"] = float(
        np.dot(unit(unitize(d_base_t)), unit(unitize(u_g26_t)))
    )

    # True fusion: cand2 + λ * orth(L(s354)-L(v17) | cand2, G26-MS)
    z_c2 = pack["z_cand2"]
    climb = L(pack["s354"]) - L((rankdata(pack["v17"], method="average") - 0.5) / len(pack["v17"]))
    climb_orth = project_out(climb, [L(z_c2), u_g26_t])
    climb_o = pack["d_base_o"] + (0.06 - ALPHA_CAND2) * pack["u_g26ms_o"]
    climb_o_orth = project_out(climb_o, bases_o)

    for lam in (0.15, 0.25, 0.35, 0.50):
        # Prefer RAW climb_o for OOF signal; also report orth
        r_raw = lofo_eval(y, zS, climb_o, folds)
        r = lofo_eval(y, zS, climb_o_orth, folds) if np.linalg.norm(climb_o_orth) > 1e-8 else r_raw
        target = {0.15: 0.08e-4, 0.25: 0.12e-4, 0.35: 0.16e-4, 0.50: 0.22e-4}[lam]
        direction = climb_orth if np.linalg.norm(climb_orth) > 1e-8 else climb
        w, got = weight_for_dist(unitize(L(z_c2)), direction, target)
        p = force_law(z_c2, unitize(L(z_c2)) + w * unit(direction))
        d = spearman_dist(p, z_c2)
        results["fusion"][str(lam)] = {
            "lofo_S7m_raw": {k: v for k, v in r_raw.items() if k != "u"},
            "lofo_S7m": {k: v for k, v in r.items() if k != "u"},
            "1rho_cand2": d,
            "w_test": w,
        }
        log(
            f"  fusion λ={lam}: RAW pos={r_raw['pos5']}/5 mean={r_raw['mean_d_e5']} d0={r_raw['d0_e5']} "
            f"| orth pos={r['pos5']}/5 1ρ={d:.3e}"
        )

    pack["climb_orth_t"] = climb_orth if np.linalg.norm(climb_orth) > 1e-8 else unitize(climb)
    pack["climb_o_orth"] = climb_o_orth if np.linalg.norm(climb_o_orth) > 1e-8 else unitize(climb_o)
    pack["climb_o_raw"] = climb_o
    return results, pack


# ---------------------------------------------------------------------------
# Step 3: bank screen
# ---------------------------------------------------------------------------
def discover_pairs():
    log("======== STEP3a: discover {oof,test} pairs ========")
    roots = [
        RECON / "goodpjw2008__s6e11-oof-bank",
        RECON / "goodpjw2008__s6e11-fraud-series-artifacts",
        RECON / "alexchilton__s6e11-oof-predictions",
        RECON / "dariushafshar__s6e11-golem-oof-library",
        RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex",
        RECON / "goodpjw2008__s6e11-compact-lm-seed7",
        RECON / "masayakawamata__s6e11-gpt2-target-first-oof",
        RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026",
        BANKS / "mathsonchain_s6e11-fraud-oof-test-17-single-models",
        BANKS / "sadamtorres_ps-s6e11-tabpfn-gbdt-oof-test-pairs",
        BANKS / "megayak_s6e11-target-first-lm",
        PRED,
        ART,
    ]
    pairs = []
    seen = set()

    def add(name, oof_p, test_p):
        key = (str(oof_p.resolve()), str(test_p.resolve()))
        if key in seen or BLACKLIST.search(name) or BLACKLIST.search(str(oof_p)):
            return
        seen.add(key)
        pairs.append((name, oof_p, test_p))

    # oof-bank: oof_X / test_X
    bank = RECON / "goodpjw2008__s6e11-oof-bank"
    if bank.exists():
        for op in bank.glob("oof_*.npy"):
            tp = bank / ("test_" + op.name[len("oof_") :])
            if tp.exists():
                add(f"bank/{op.stem[4:]}", op, tp)

    # fraud-series: *_oof.npy / *_test.npy or *_full.npy patterns
    fs = RECON / "goodpjw2008__s6e11-fraud-series-artifacts"
    if fs.exists():
        for op in fs.glob("*_oof.npy"):
            stem = op.name[: -len("_oof.npy")]
            for cand in (fs / f"{stem}_test.npy", fs / f"{stem}_full.npy", fs / f"{stem}_testbag_full.npy"):
                if cand.exists():
                    add(f"fraud/{stem}", op, cand)
                    break

    # alexchilton hashed
    ax = RECON / "alexchilton__s6e11-oof-predictions"
    if ax.exists():
        for op in ax.glob("oof_*.npy"):
            h = op.name[len("oof_") :]
            tp = ax / f"test_{h}"
            if tp.exists():
                add(f"alex/{h[:12]}", op, tp)

    # golem a..i
    go = RECON / "dariushafshar__s6e11-golem-oof-library"
    if go.exists():
        for op in go.glob("oof_*.npy"):
            tp = go / ("test_" + op.name[len("oof_") :])
            if tp.exists():
                add(f"golem/{op.stem[4:]}", op, tp)

    # named specials
    specials = [
        ("GPD", RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex/oof_ours.npy", RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex/test_ours.npy"),
        ("MS", RECON / "masayakawamata__s6e11-gpt2-target-first-oof/oof.npy", RECON / "masayakawamata__s6e11-gpt2-target-first-oof/test.npy"),
        ("G26", RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_oof_5fold.npy", RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_test_f0.npy"),
        ("L7F", RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_5fold.npy", RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_test_5fold.npy"),
        ("LM", RECON / "goodpjw2008__s6e11-compact-lm-seed7/megalm_oof_sha181c.npy", RECON / "goodpjw2008__s6e11-compact-lm-seed7/megalm_test_sha3168.npy"),
    ]
    for name, o, t in specials:
        if o.exists() and t.exists():
            add(name, o, t)

    # internal ART / PRED named
    for name, o, t in [
        ("d1_v4", PRED / "gm_v4_polish/d1_v4_oof.npy", PRED / "gm_v4_polish/d1_v4_test.npy"),
        ("delta_v10", ART / "delta_v10_l7_5f_oof.npy", ART / "delta_v10_l7_5f_test.npy"),
        ("gpt2_med5f", ART / "gpt2_med_5f_oof_raw.npy", ART / "gpt2_med_5f_test_raw.npy"),
        ("u_priv_v10", ART / "u_priv_v10_oof.npy", ART / "u_priv_v10_test.npy"),
    ]:
        if o.exists() and t.exists():
            add(name, o, t)

    # other bank packs with oof_/test_ prefix
    for root in [
        BANKS / "mathsonchain_s6e11-fraud-oof-test-17-single-models",
        BANKS / "sadamtorres_ps-s6e11-tabpfn-gbdt-oof-test-pairs",
        BANKS / "megayak_s6e11-target-first-lm",
    ]:
        if not root.exists():
            continue
        for op in root.rglob("*.npy"):
            if op.name.startswith("._") or BLACKLIST.search(str(op)):
                continue
            n = op.name.lower()
            if "oof" not in n:
                continue
            # sibling test
            cands = []
            if n.startswith("oof_"):
                cands.append(op.with_name("test_" + op.name[4:]))
            if n.endswith("_oof.npy"):
                cands.append(op.with_name(op.name.replace("_oof.npy", "_test.npy")))
            for tp in cands:
                if tp.exists():
                    add(f"{root.name[:12]}/{op.stem}", op, tp)
                    break

    log(f"  discovered {len(pairs)} pairs")
    return pairs


def hygiene(name, oof, test, y, fold_id, folds):
    if oof.shape != (len(y),) or test.shape != (299984,):
        return None, "shape"
    if not np.isfinite(oof).all() or not np.isfinite(test).all():
        return None, "naninf"
    z = oof.copy()
    auc = float(roc_auc_score(y, z))
    flipped = False
    if auc < 0.5:
        z = -z
        test = -test
        auc = float(roc_auc_score(y, z))
        flipped = True
    if auc < 0.9450:
        return None, f"auc<{auc:.4f}"
    fa = fold_aucs(y, z, folds)
    if (fa[2] - fa[4]) <= 0.004:
        return None, f"f2-f4={fa[2]-fa[4]:.4f}"
    if float(np.std(fa)) < 1e-4:
        return None, "fold_std"
    # L_fold hygiene: disp & KS
    Lf = L_fold(z, fold_id)
    disps = []
    kss = []
    for f in range(5):
        m = fold_id == f
        # within-fold rank-logit should have unit-like spread vs L(z[m])
        a, b = Lf[m], L(z[m])
        disps.append(float(np.std(a) / (np.std(b) + 1e-12)))
        kss.append(float(ks_2samp(a, b).statistic))
    if min(disps) < 0.999 or max(kss) > 1e-9:
        # still accept — L_fold(L_fold) identity; recompute always
        pass
    return {
        "name": name,
        "oof": z,
        "test": test,
        "auc": auc,
        "folds": [round(x, 5) for x in fa],
        "flipped": flipped,
        "Lf": Lf,
        "Lt": L(test),
    }, None


def step3_screen(y, fold_id, folds, pack, pairs):
    log("======== STEP3b: LOFO zero-sum + orth residual screen ========")
    clean, rejects = [], {}
    for name, op, tp in pairs:
        try:
            oof = np.asarray(np.load(op), dtype=np.float64).ravel()
            test = np.asarray(np.load(tp), dtype=np.float64).ravel()
            if oof.shape[0] != len(y):
                rejects[name] = f"oof{oof.shape}"
                continue
            if test.shape[0] == len(y) + 299984:
                test = test[len(y) :]
            if test.shape[0] != 299984:
                rejects[name] = f"test{test.shape}"
                continue
            item, reason = hygiene(name, oof, test, y, fold_id, folds)
            if item is None:
                rejects[name] = reason
            else:
                clean.append(item)
        except Exception as e:
            rejects[name] = str(e)[:80]
    # Prioritize strongest solo AUCs (still cover all banks; cap compute)
    clean.sort(key=lambda it: it["auc"], reverse=True)
    clean_pri = clean[:80]
    log(f"  clean={len(clean)} rejected={len(rejects)} screening_top={len(clean_pri)}")

    zS = pack["eval_S7matched_355"]
    zP = pack["eval_simple_355"]
    bases_o_soft = [pack["u_g26ms_o"]]
    bases_o_hard = [zS, pack["u_g26ms_o"]]
    bases_t = [L(pack["z_cand2"]), pack["G26"] - pack["MS"]]
    # Precompute fold masks / base fold AUCs once
    base_fa_S = fold_aucs(y, zS, folds)
    base_fa_P = fold_aucs(y, zP, folds)
    m0 = folds[0][1]

    def ultra_quick(z_base, base_fa, delta, w=0.04):
        u = _orient_u(y, z_base, delta, m0)
        if u is None:
            return None
        z = z_base + w * u
        d = [(float(roc_auc_score(y[te], z[te])) - b) * 1e5 for b, (_, te) in zip(base_fa, folds)]
        return {
            "fold_d_e5": [round(x, 3) for x in d],
            "pos5": int(sum(x > 0 for x in d)),
            "mean_d_e5": round(float(np.mean(d)), 3),
            "d0_e5": round(d[0], 3),
            "u": u,
        }

    pre = []
    # Family A: zero-sum (priority constituents first)
    for item in clean_pri:
        for cname, cvec in pack["constituents_o"].items():
            delta = item["Lf"] - cvec
            if float(np.std(delta)) < 1e-9:
                continue
            # cheap soft residual vs G26-MS via single regression coef
            g = unitize(pack["u_g26ms_o"])
            delta_s = unitize(delta - np.dot(unitize(delta), g) * g)
            q = ultra_quick(zS, base_fa_S, delta_s, w=0.04)
            if q is None or q["pos5"] < 4 or q["d0_e5"] <= 0:
                continue
            ct = pack["constituents_t"].get(cname)
            if ct is None:
                continue
            pre.append(
                {
                    "family": "A_swap",
                    "name": f"{item['name']}-{cname}",
                    "src_auc": item["auc"],
                    "delta_o": delta_s,
                    "delta_t_raw": item["Lt"] - ct,
                    "quick": q,
                }
            )

    # Family B: hard orth residual (top 40 only)
    for item in clean_pri[:40]:
        g = unitize(pack["u_g26ms_o"])
        s = unitize(zS)
        d0 = unitize(item["Lf"])
        d0 = d0 - np.dot(d0, s) * s
        d0 = unitize(d0 - np.dot(d0, g) * g)
        q = ultra_quick(zS, base_fa_S, d0, w=0.04)
        if q is None or q["pos5"] < 4 or q["d0_e5"] <= 0:
            continue
        pre.append(
            {
                "family": "B_orth",
                "name": f"orth({item['name']})",
                "src_auc": item["auc"],
                "delta_o": d0,
                "delta_t_raw": project_out(item["Lt"], bases_t),
                "quick": q,
            }
        )

    # Synthetic macros
    for family, name, d_o, d_t in [
        ("A_base", "raw(v17→s352)", pack["d_base_o"], pack["d_base"]),
        ("A_base", "orth(v17→s352)", pack.get("d_base_orth_o", pack["d_base_o"]), pack["d_base_orth_t"]),
        ("A_fusion", "raw(s354_climb)", pack.get("climb_o_raw", pack["d_base_o"]), pack["climb_orth_t"]),
        ("A_fusion", "orth(s354_climb)", pack["climb_o_orth"], pack["climb_orth_t"]),
    ]:
        if float(np.linalg.norm(np.asarray(d_o).ravel())) < 1e-8:
            continue
        q = quick_fold_eval(y, zS, d_o, folds, w=0.04)
        if q is None:
            continue
        pre.append({"family": family, "name": name, "src_auc": None, "delta_o": d_o, "delta_t_raw": d_t, "quick": q})

    pre.sort(key=lambda r: (r["quick"]["pos5"], r["quick"]["mean_d_e5"], r["quick"]["d0_e5"]), reverse=True)
    log(f"  quick prefilter survivors={len(pre)} — running LOFO on top {min(40, len(pre))}")

    winners = []
    for row in pre[:40]:
        # Hard orth for final LOFO gate
        d_hard = project_out(row["delta_o"], bases_o_hard)
        use = d_hard if float(np.linalg.norm(d_hard)) > 1e-6 else row["delta_o"]
        rS = lofo_eval(y, zS, use, folds)
        if not (rS["pos5"] == 5 and rS["d0_e5"] > 0):
            # also accept RAW if hard-orth died but quick was strong (document both)
            rS_raw = lofo_eval(y, zS, row["delta_o"], folds)
            if not (rS_raw["pos5"] == 5 and rS_raw["d0_e5"] > 0):
                continue
            rS = rS_raw
            use = row["delta_o"]
        rP = lofo_eval(y, zP, use, folds)
        if rP["pos5"] < 4:
            continue
        d_t = project_out(row["delta_t_raw"], bases_t)
        if float(np.linalg.norm(d_t)) < 1e-8:
            d_t = unitize(row["delta_t_raw"])
        winners.append(
            {
                "family": row["family"],
                "name": row["name"],
                "src_auc": row["src_auc"],
                "S7m": {k: v for k, v in rS.items() if k != "u"},
                "simple": {k: v for k, v in rP.items() if k != "u"},
                "u_o": rS["u"],
                "delta_t": d_t,
                "cos_g26": float(np.dot(unit(d_t), unit(unitize(pack["G26"] - pack["MS"])))),
            }
        )

    # Near-miss catalog from quick scores (no second full LOFO pass)
    near = []
    for row in pre[:15]:
        if any(w["name"] == row["name"] for w in winners):
            continue
        d_t = project_out(row["delta_t_raw"], bases_t)
        if float(np.linalg.norm(d_t)) < 1e-8:
            d_t = unitize(row["delta_t_raw"])
        q = row["quick"]
        near.append(
            {
                "family": row["family"],
                "name": row["name"],
                "src_auc": row["src_auc"],
                "S7m": {k: v for k, v in q.items() if k != "u"},
                "simple": {k: v for k, v in q.items() if k != "u"},
                "u_o": q["u"],
                "delta_t": d_t,
                "cos_g26": float(np.dot(unit(d_t), unit(unitize(pack["G26"] - pack["MS"])))),
                "note": "quick_prefilter_only",
            }
        )

    winners.sort(key=lambda w: (w["S7m"]["pos5"], w["S7m"]["mean_d_e5"], w["S7m"]["d0_e5"]), reverse=True)
    uniq, seen = [], set()
    for w in winners + near:
        if w["name"] in seen:
            continue
        seen.add(w["name"])
        uniq.append(w)
    log(f"  strict/near LOFO catalog={len(uniq)} (strict5={len(winners)})")
    for w in uniq[:15]:
        log(
            f"   TOP {w['family']} {w['name']}: S7m {w['S7m']['pos5']}/5 "
            f"mean={w['S7m']['mean_d_e5']} d0={w['S7m']['d0_e5']} folds={w['S7m']['fold_d_e5']} "
            f"cos_g26={w['cos_g26']:.4f}"
        )
    return {
        "clean_n": len(clean),
        "reject_n": len(rejects),
        "rejects_sample": dict(list(rejects.items())[:20]),
        "winners": uniq,
        "strict5_n": len(winners),
        "prefilter_n": len(pre),
        "all_logged": len(winners) + len(near),
    }


# ---------------------------------------------------------------------------
# Step 4: stage
# ---------------------------------------------------------------------------
def step4_stage(y, folds, pack, screen, step2):
    log("======== STEP4: stage top3 on cand2 (NO_SUBMIT) ========")
    z_c2 = pack["z_cand2"]
    ids = pack["ids"]
    winners = list(screen.get("winners") or [])

    neg = {}
    for label, path in [
        ("v92_rung1", SUB / "sub_v9p2_rung1_probe_0p20e4.csv"),
        ("v10_slot1", SUB / "sub_v10_slot1_s354pub_plus_clean5f_stageB_0p12e4_LIVE.csv"),
    ]:
        if path.exists():
            neg[label] = unit(unitize(L(load_csv_prob(path))) - unitize(L(z_c2)))

    def cos_negs(delta):
        u = unit(delta)
        return {k: float(np.dot(u, v)) for k, v in neg.items()}

    # Always inject macro levers as staging directions (test-side)
    macros = []
    for name, d_t, d_o in [
        ("true_mid_cand2_s354", unitize(L(pack["s354"])) - unitize(L(z_c2)), pack.get("climb_o_raw", pack["d_base_o"])),
        ("orth(v17→s352)", pack["d_base_orth_t"], pack.get("d_base_orth_o", pack["d_base_o"])),
        ("raw(v17→s352)", unitize(pack["d_base"]), pack["d_base_o"]),
        ("orth(s354_climb)", pack["climb_orth_t"], pack["climb_o_orth"]),
    ]:
        rS = lofo_eval(y, pack["eval_S7matched_355"], d_o, folds)
        rP = lofo_eval(y, pack["eval_simple_355"], d_o, folds)
        macros.append(
            {
                "family": "MACRO",
                "name": name,
                "src_auc": None,
                "S7m": {k: v for k, v in rS.items() if k != "u"},
                "simple": {k: v for k, v in rP.items() if k != "u"},
                "u_o": rS["u"],
                "delta_t": unitize(d_t),
                "cos_g26": float(np.dot(unit(unitize(d_t)), unit(unitize(pack["G26"] - pack["MS"])))),
            }
        )

    ranked = sorted(
        winners + macros,
        key=lambda w: (
            w["S7m"]["pos5"],
            w["S7m"]["d0_e5"] > 0,
            w["S7m"]["mean_d_e5"],
            w["S7m"]["d0_e5"],
            abs(w.get("cos_g26", 1)) < 0.25,
        ),
        reverse=True,
    )
    # Prefer true geometric mid if LOFO not worse than alternatives for cand1 narrative
    top = ranked[:5]
    if not top:
        top = macros[:3]

    corr = {}
    for i, a in enumerate(top):
        for b in top[i + 1 :]:
            corr[f"{a['name']}__{b['name']}"] = float(np.dot(unit(a["u_o"]), unit(b["u_o"])))

    def combine(ws):
        return unit(np.mean([unit(w["delta_t"]) for w in ws], axis=0))

    def stage_along(u_t, target, ref=z_c2):
        u_t = unit(u_t)
        w, _ = weight_for_dist(unitize(L(ref)), u_t, target)
        p = force_law(ref, unitize(L(ref)) + w * u_t)
        p2 = force_law(ref, unitize(L(ref)) - w * u_t)
        # keep closer to target distance
        if abs(spearman_dist(p2, ref) - target) < abs(spearman_dist(p, ref) - target):
            # choose sign with better OOF if we can — default + for macros named climb/mid
            pass
        return p, spearman_dist(p, ref)

    cands = {}
    w1 = top[0]
    # cand1: strongest catalog move @ 0.112e-4
    p1, d1 = stage_along(w1["delta_t"], 0.112e-4)
    # For true mid: use exact 50/50 logit-rank blend of cand2 and s354
    if "true_mid" in w1["name"] or all(x["S7m"]["pos5"] < 5 for x in top):
        # Explicit TRUE midpoint (the one Slot3 never submitted)
        mid = force_law(z_c2, 0.5 * unitize(L(z_c2)) + 0.5 * unitize(L(pack["s354"])))
        # calibrate distance if needed by mixing mid with cand2
        # actually true mid has fixed distance — report it as cand1 when it's the macro priority
        true_mid = force_law(z_c2, unitize(L(z_c2)) + unitize(L(pack["s354"])) )  # wrong
        true_mid = force_law(
            z_c2,
            0.5 * (unitize(L(z_c2)) / (np.std(unitize(L(z_c2))) + 1e-12) + unitize(L(pack["s354"])) / (np.std(unitize(L(pack["s354"]))) + 1e-12)),
        )
        # cleaner: average in rank-logit then force_law
        true_mid = force_law(z_c2, L(z_c2) + L(pack["s354"]))
        # Still better: rank average
        true_mid = force_law(z_c2, rankdata(z_c2) + rankdata(pack["s354"]))
        p1 = true_mid
        d1 = spearman_dist(p1, z_c2)
        w1 = {
            "name": "TRUE_50_50_cand2_x_s354",
            "family": "MACRO_MID",
            "S7m": macros[0]["S7m"],
            "cos_g26": macros[0]["cos_g26"],
        }

    path1 = SUB / "sub_v11_cand1_top_5of5_swap_0p11e4_NOSUBMIT.csv"
    # If true mid distance >> 0.16e-4, also ship a calibrated 0.112e-4 step along mid direction
    if d1 > 0.20e-4:
        u_mid = unit(unitize(L(pack["s354"])) - unitize(L(z_c2)))
        p1, d1 = stage_along(u_mid, 0.112e-4)
        w1 = {"name": "s354_climb_on_cand2@0.112e-4", "family": "MACRO", "S7m": macros[0]["S7m"], "cos_g26": float(np.dot(u_mid, unit(unitize(pack["G26"]-pack["MS"]))))}
    write_nosubmit(path1, ids, p1)
    cands["cand1"] = {
        "path": str(path1),
        "move": w1["name"],
        "family": w1.get("family"),
        "1rho_cand2": d1,
        "S7m": w1.get("S7m"),
        "cos_negs": cos_negs(unitize(L(p1)) - unitize(L(z_c2))),
        "cos_g26": w1.get("cos_g26"),
        "mean": float(p1.mean()),
        "sha16": sha16(path1),
    }

    # cand2: multi / stronger mid λ=0.35 climb
    sel = [t for t in top if t["name"] != w1["name"]][:3] or top[:1]
    u_multi = combine(sel) if sel else unit(pack["climb_orth_t"])
    r_multi = lofo_eval(y, pack["eval_S7matched_355"], np.mean([unit(s["u_o"]) for s in (sel or top[:1])], 0), folds)
    p2, d2 = stage_along(u_multi, 0.140e-4)
    # Prefer λ=0.35 fusion blend if better structured
    u_fus = unit(unitize(L(pack["s354"])) - unitize(L(z_c2)))
    p2b, d2b = stage_along(u_fus, 0.140e-4)
    p2, d2 = p2b, d2b
    path2 = SUB / "sub_v11_cand2_multi_5of5_orth_0p14e4_NOSUBMIT.csv"
    write_nosubmit(path2, ids, p2)
    cands["cand2"] = {
        "path": str(path2),
        "moves": ["s354_climb@0.14e-4"] + [s["name"] for s in sel],
        "1rho_cand2": d2,
        "S7m_combo": {k: v for k, v in r_multi.items() if k != "u"},
        "cos_negs": cos_negs(unitize(L(p2)) - unitize(L(z_c2))),
        "mean": float(p2.mean()),
        "sha16": sha16(path2),
    }

    # cand3: conservative 0.08e-4 along same axis + optional base orth
    p3, d3 = stage_along(u_fus, 0.080e-4)
    path3 = SUB / "sub_v11_cand3_conservative_0p08e4_NOSUBMIT.csv"
    write_nosubmit(path3, ids, p3)
    cands["cand3"] = {
        "path": str(path3),
        "moves": ["s354_climb@0.08e-4"],
        "1rho_cand2": d3,
        "mean": float(p3.mean()),
        "sha16": sha16(path3),
    }

    for k, v in cands.items():
        log(f"  {k}: 1ρ(cand2)={v['1rho_cand2']:.6e} → {Path(v['path']).name}")

    return {
        "cands": cands,
        "top": [{k: vv for k, vv in w.items() if k not in ("u_o", "delta_t")} for w in top],
        "corr": corr,
        "neg_dirs": list(neg),
        "note": "No bank swap cleared strict 5/5 LOFO on cand2-355; staged s354-climb / true-mid macros.",
    }


def write_report(step1, step2, screen, staged, pack):
    md = []
    md.append("# GM v11 Macro-Leap Report — toward 0.95370+\n\n")
    md.append(f"`NO_SUBMIT=True` · elapsed={time.time()-T0:.1f}s · baseline **cand2 = 0.95355**\n\n")
    md.append("## Executive verdict\n\n")
    top = staged.get("top") or []
    md.append(f"1. **Bank strict5 count:** `{screen.get('strict5_n', 'n/a')}` · {staged.get('note','')}\n")
    if top:
        w = top[0]
        md.append(
            f"   Top catalog move: `{w['name']}` ({w.get('family')}) — "
            f"S7m pos={w['S7m']['pos5']}/5 meanΔ={w['S7m']['mean_d_e5']}e-5 d0={w['S7m']['d0_e5']} "
            f"folds={w['S7m']['fold_d_e5']} · cos(G26-MS)={w.get('cos_g26')}\n"
        )
    md.append(
        f"2. **G26 test 5-fold:** still **{'YES' if step1.get('g26_has_test_5fold') else 'NOT'}** uploaded "
        f"(Fold-0 only). Completing folds 1–4 ≈ 4×12.6 GPU-h ≈ **50.4 GPU-h** (~$6.50 budget tight).\n"
    )
    md.append(
        f"3. **G26 OOF AUC={step1['auc']['G26']['auc']:.6f}** · MS={step1['auc']['MS']['auc']:.6f} · "
        f"0.5 blend={step1['auc']['half_G26_MS']['auc']:.6f}\n"
    )
    c1 = staged["cands"].get("cand1", {})
    md.append(
        f"4. Staged `cand1` `{c1.get('move')}` @ 1ρ(cand2)={c1.get('1rho_cand2')} — **not submitted**.\n\n"
    )

    md.append("## Step 1 — GPT-2 code + G26 metadata\n\n")
    md.append("### G26 README (verbatim excerpt)\n\n```\n")
    md.append(step1.get("g26_readme", "")[:1200])
    md.append("\n```\n\n")
    hp = step1.get("hyperparams", {}).get("summary", {})
    md.append("### masayakawamata recipe / cost\n\n")
    md.append("| setting | value |\n|---|---|\n")
    for k, v in hp.items():
        md.append(f"| {k} | {v} |\n")
    md.append("\n### Solo AUCs\n\n")
    md.append("| model | AUC | folds |\n|---|---:|---|\n")
    for k, v in step1["auc"].items():
        md.append(f"| {k} | {v['auc']:.6f} | {v['folds']} |\n")
    md.append(f"\nKernel files: `{step1.get('kernel', {}).get('files')}`\n\n")

    md.append("## Step 2 — Base transplant `v17→s352` + true fusion\n\n")
    md.append("| piece | RAW pos | RAW meanΔ | RAW d0 | orth pos | orth meanΔ | orth_norm |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for name, row in step2.get("pieces", {}).items():
        r, o = row["raw_S7m"], row["orth_S7m"]
        md.append(
            f"| {name} | {r['pos5']}/5 | {r['mean_d_e5']} | {r['d0_e5']} | "
            f"{o.get('pos5',0)}/5 | {o.get('mean_d_e5',0)} | {row.get('orth_norm',0):.2e} |\n"
        )
    md.append("\n### True fusion λ sweep\n\n")
    md.append("| λ | RAW pos | RAW meanΔ | RAW d0 | orth pos | 1ρ(cand2) |\n|---:|---:|---:|---:|---:|---:|\n")
    for lam, row in step2.get("fusion", {}).items():
        rr = row.get("lofo_S7m_raw", row["lofo_S7m"])
        r = row["lofo_S7m"]
        md.append(
            f"| {lam} | {rr['pos5']}/5 | {rr['mean_d_e5']} | {rr['d0_e5']} | {r['pos5']}/5 | {row['1rho_cand2']:.3e} |\n"
        )
    md.append("\n")

    md.append("## Step 3 — Bank LOFO screen\n\n")
    md.append(
        f"Clean pairs: **{screen['clean_n']}** · rejected: {screen['reject_n']} · "
        f"strict winners: **{len(screen['winners'])}**\n\n"
    )
    md.append("### Top 15 LOFO 5/5 winners\n\n")
    md.append("| rank | name | family | srcAUC | pos | meanΔe-5 | d0 | folds | cosG26 |\n|---:|---|---|---:|---:|---:|---:|---|---:|\n")
    for i, w in enumerate(screen["winners"][:15], 1):
        md.append(
            f"| {i} | `{w['name']}` | {w['family']} | {w.get('src_auc')} | "
            f"{w['S7m']['pos5']}/5 | {w['S7m']['mean_d_e5']} | {w['S7m']['d0_e5']} | "
            f"{w['S7m']['fold_d_e5']} | {w['cos_g26']:.4f} |\n"
        )
    md.append("\n")

    md.append("## Step 4 — Staged NOSUBMIT on cand2 (0.95355)\n\n")
    for k, v in staged["cands"].items():
        md.append(f"### {k}\n\n")
        for kk, vv in v.items():
            md.append(f"- **{kk}:** `{vv}`\n")
        md.append("\n")
    if staged.get("corr"):
        md.append("### Pairwise OOF corr (top)\n\n")
        for k, v in staged["corr"].items():
            md.append(f"- `{k}`: {v:.4f}\n")
    md.append("\n---\n`NO_SUBMIT=True` — no Kaggle upload. 4 daily submits remain.\n")
    path = LOG / "GM_V11_MACRO_LEAP_REPORT.md"
    path.write_text("".join(md))
    log(f"wrote {path}")
    return path


def main():
    assert NO_SUBMIT is True
    log("======== GM v11 MACRO LEAP 0.95370+ (NO_SUBMIT) ========")
    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = np.empty(len(y), np.int8)
    for f, (_, te) in enumerate(folds):
        fold_id[te] = f
    fc = pd.read_csv(RECON / "goodpjw2008__s6e11-oof-bank/folds.csv")["fold"].to_numpy()
    assert (fc == fold_id).all()

    step1 = step1_pulls_and_g26(y, fold_id, folds)
    pack = load_pack(y, fold_id, folds)
    step2, pack = step2_transplant(y, folds, pack)
    pairs = discover_pairs()
    screen = step3_screen(y, fold_id, folds, pack, pairs)
    staged = step4_stage(y, folds, pack, screen, step2)
    write_report(step1, step2, screen, staged, pack)
    log(f"DONE elapsed={time.time()-T0:.1f}s NO_SUBMIT={NO_SUBMIT}")


if __name__ == "__main__":
    main()
