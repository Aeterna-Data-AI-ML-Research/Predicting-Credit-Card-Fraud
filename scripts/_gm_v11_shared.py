#!/usr/bin/env python3
"""Shared GM v11.1 / teamlead primitives — NO_SUBMIT helpers only."""
from __future__ import annotations

import hashlib
import os
import re
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
LOG = VOL / "logs/overnight"
ART = Path("/tmp/gm_deep_tech/preds/gm_v10_hardened")
GPT2_DIR = VOL / "external/gpt2_target_first_09501"
KAG = VOL / ".venv_forensics/bin/kaggle"
ALPHA_CAND2 = 0.03613
BLACKLIST = re.compile(r"(v10_|soft_diff|gb_soft|idf_p3|clean035|zanch_)", re.I)
T0 = time.time()

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb
except Exception:
    lgb = None

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


def project_out(v, bases):
    v = np.asarray(v, dtype=np.float64).ravel()
    cols = [unitize(b) for b in bases if np.linalg.norm(np.asarray(b).ravel()) > 1e-12]
    if not cols:
        return unitize(v)
    B = np.column_stack(cols)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(B, v)
    return unitize(v - lr.predict(B))


def pchip_lock(src, ref):
    """Match src quantiles to ref distribution (disp≈1, KS≈0)."""
    src = np.asarray(src, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    n = len(src)
    qs = np.linspace(0, 1, 401)
    x = np.quantile(src, qs)
    y = np.quantile(ref, qs)
    # unique x for PCHIP
    _, ui = np.unique(x, return_index=True)
    x, y = x[ui], y[ui]
    if len(x) < 4:
        return unitize(src) * (ref.std() + 1e-12) + ref.mean()
    f = PchipInterpolator(x, y, extrapolate=True)
    out = f(src)
    return out


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[te], z[te])) for _, te in folds]


def load_prob(p):
    df = pd.read_csv(p)
    col = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
    return df[col].to_numpy(np.float64)


def write_nosubmit(path, ids, probs):
    assert NO_SUBMIT and "NOSUBMIT" in path.name.upper()
    pd.DataFrame({"id": ids, "is_fraud": probs}).to_csv(path, index=False)
    return path


def weight_force(cand2, u, target, hi=80.0, n=80):
    z0 = unitize(L(cand2))
    u = unit(u)
    lo, best_w, best_err, got, best_p = 0.0, 0.0, 1e9, 0.0, cand2
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        p = force_law(cand2, z0 + mid * u)
        d = spearman_dist(p, cand2)
        err = abs(d - target)
        if err < best_err:
            best_err, best_w, got, best_p = err, mid, d, p
        if d > target:
            hi = mid
        else:
            lo = mid
    return best_p, float(best_w), float(got)


def lofo_eval(y, z_base, delta, folds, betas=(0.01, 0.02, 0.04, 0.06, 0.08)):
    z_base = np.asarray(z_base, dtype=np.float64).ravel()
    u = unit(np.asarray(delta, dtype=np.float64).ravel())
    if not np.isfinite(u).all() or np.linalg.norm(u) < 1e-12:
        return {"fold_d_e5": [0.0] * 5, "pos5": 0, "mean_d_e5": 0.0, "d0_e5": 0.0, "u": u, "pass": False}
    m0 = folds[0][1]
    if roc_auc_score(y[m0], z_base[m0] + 0.02 * u[m0]) < roc_auc_score(y[m0], z_base[m0] - 0.02 * u[m0]):
        u = -u
    base_fa = fold_aucs(y, z_base, folds)
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
    pos = int(sum(x > 0 for x in d_e5))
    return {
        "fold_d_e5": [round(x, 3) for x in d_e5],
        "pos5": pos,
        "mean_d_e5": round(float(np.mean(d_e5)), 3),
        "d0_e5": round(d_e5[0], 3),
        "betas": chosen,
        "u": u,
        "pass": pos == 5 and d_e5[0] > 0,
    }


def gate_pass(rS, rP):
    return bool(rS.get("pass")) and rP["pos5"] >= 4 and rP["d0_e5"] > 0


def setup_folds(y):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = np.empty(len(y), np.int8)
    for f, (_, te) in enumerate(folds):
        fold_id[te] = f
    fc = pd.read_csv(RECON / "goodpjw2008__s6e11-oof-bank/folds.csv")["fold"].to_numpy()
    assert (fc == fold_id).all()
    return folds, fold_id


def load_anchor_and_bank(y, fold_id, folds):
    """Immutable cand2 + core members + evaluators."""
    ids = pd.read_csv(VOL / "data/sample_submission.csv")["id"].to_numpy()
    s7 = load_prob(SUB / "sub_master_slot7_verified_s7A.csv")
    cand2 = load_prob(SUB / "sub_v10p2_cand2_S7ours_plus_twoseed_G26_MS_0p11e4_NOSUBMIT.csv")

    def npy(p):
        return np.load(p).astype(np.float64).ravel()

    A = L(load_prob(RECON / "amanatar__s6e11-strike10-podium-apex-095340/submission_095340.csv"))
    F = L(load_prob(RECON / "moranzavdi__s6e11-strike9-lb-095330/submission_095330.csv"))
    LM = L(npy(RECON / "goodpjw2008__s6e11-compact-lm-seed7/megalm_test_sha3168.npy"))
    L7F = L(npy(RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_test_5fold.npy"))
    GPD = L(npy(RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex/test_ours.npy"))
    MLP = L(npy(RECON / "goodpjw2008__s6e11-oof-bank/test_gbn_bag5_w512.npy"))
    GB = L(npy(RECON / "goodpjw2008__s6e11-oof-bank/test_gb_bag5.npy"))
    MS = L(npy(RECON / "masayakawamata__s6e11-gpt2-target-first-oof/test.npy"))
    G26 = L(npy(RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_test_f0.npy"))
    d1t = unitize(npy(PRED / "gm_v4_polish/d1_v4_test.npy"))

    # verify cand2 identity
    recon = force_law(s7, L(s7) + ALPHA_CAND2 * (G26 - MS))
    assert spearman_dist(recon, cand2) < 1e-10, spearman_dist(recon, cand2)
    # byte-ish: force_law of cand2 alone reproduces
    assert spearman_dist(force_law(cand2, L(cand2)), cand2) < 1e-12

    MSo = L_fold(npy(RECON / "masayakawamata__s6e11-gpt2-target-first-oof/oof.npy"), fold_id)
    G26o = L_fold(npy(RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026/gpt2_s2026_oof_5fold.npy"), fold_id)
    LMo = L_fold(npy(RECON / "goodpjw2008__s6e11-compact-lm-seed7/megalm_oof_sha181c.npy"), fold_id)
    L7Fo = L_fold(npy(RECON / "goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_5fold.npy"), fold_id)
    GPDo = L_fold(npy(RECON / "daoviet__s6e11-gp-dvae-orthogonal-apex/oof_ours.npy"), fold_id)
    MLPo = L_fold(npy(RECON / "goodpjw2008__s6e11-oof-bank/oof_gbn_bag5_w512.npy"), fold_id)
    GBo = L_fold(npy(RECON / "goodpjw2008__s6e11-oof-bank/oof_gb_bag5.npy"), fold_id)
    d1o = unitize(npy(PRED / "gm_v4_polish/d1_v4_oof.npy"))
    W = L(npy(VOL / "external/nov08_part12/wide_lm_stack_nested.npy"))
    Fo = W  # OOF strike proxy

    # gpt2 med
    Gmedo = Gmedt = None
    if (ART / "gpt2_med_5f_oof_raw.npy").exists():
        Gmedo = L_fold(npy(ART / "gpt2_med_5f_oof_raw.npy"), fold_id)
        Gmedt = L(npy(ART / "gpt2_med_5f_test_raw.npy"))

    z_ref = npy(PRED / "gm_v5_overnight/z_ref_old_oof.npy")
    z_simple = z_ref + 0.05 * d1o
    z_areeb_o = W + 0.12 * (LMo - W) + 0.06 * (L7Fo - W) - 0.05 * (MLPo - W) - 0.025 * (GBo - W)
    z_areeb_o = L(z_areeb_o) + 0.05 * d1o
    z_s7_o = unitize(unitize(z_areeb_o) + 0.04 * d1o)
    u_g26ms_o = G26o - MSo
    eval_simple_355 = z_simple + 0.04 * unitize(u_g26ms_o)
    eval_S7matched_355 = unitize(z_s7_o + ALPHA_CAND2 * unitize(u_g26ms_o))

    bases_o = [unitize(L(s7)[:0]), unitize(u_g26ms_o)]  # placeholder fix below
    # OOF bases: z_s7_o and g26-ms (proxy for S7 + G26-MS double orth)
    bases_o = [z_s7_o, u_g26ms_o]
    bases_t = [L(s7), G26 - MS]

    # P4
    p4 = {}
    for tag in ("L4.2", "L4.3", "L4.5"):
        op = PRED / f"matrix_v2/P4_{tag}_pure5f_oof.npy"
        tp = PRED / f"matrix_v2/P4_{tag}_pure5f_test.npy"
        if op.exists() and tp.exists():
            p4[tag] = (L_fold(npy(op), fold_id), L(npy(tp)))

    # bank shallow members for KMAT proxy / denoiser features
    bank_o, bank_t = [], []
    bank = RECON / "goodpjw2008__s6e11-oof-bank"
    prefer = [
        "oof_lgb_dig_l4.npy",
        "oof_lgb_diglead_l3.npy",
        "oof_lgb_te_l4.npy",
        "oof_gb_bag5.npy",
        "oof_gbn_bag5_w512.npy",
        "oof_ebm_dig_bag.npy",
        "oof_glm_ohk2_bag.npy",
        "oof_lgb_raw_l4.npy",
        "oof_gbx_bag5.npy",
        "oof_lgb_dig_s100_bag.npy",
        "oof_ebm_all_bag.npy",
        "oof_lgb_diglead_bag5.npy",
        "oof_gbn_bag5_w128.npy",
        "oof_glm_dl_s50_bag.npy",
        "oof_lgb_te_l4_s100.npy",
    ]
    for name in prefer:
        op = bank / name
        tp = bank / ("test_" + name[len("oof_") :])
        if op.exists() and tp.exists():
            oo = npy(op)
            tt = npy(tp)
            if oo.shape == (len(y),) and tt.shape == (299984,):
                bank_o.append(L_fold(oo, fold_id))
                bank_t.append(L(tt))

    # P_super
    if (PRED / "P_super_v5_harvest_oof.npy").exists():
        Ps_o = L_fold(npy(PRED / "P_super_v5_harvest_oof.npy"), fold_id)
        Ps_t = L(npy(PRED / "P_super_v5_harvest_test.npy"))
    else:
        Ps_o = Ps_t = None

    pack = dict(
        ids=ids,
        s7=s7,
        cand2=cand2,
        A=A,
        F=F,
        LM=LM,
        L7F=L7F,
        GPD=GPD,
        MLP=MLP,
        GB=GB,
        MS=MS,
        G26=G26,
        d1t=d1t,
        d1o=d1o,
        MSo=MSo,
        G26o=G26o,
        LMo=LMo,
        L7Fo=L7Fo,
        GPDo=GPDo,
        MLPo=MLPo,
        GBo=GBo,
        W=W,
        Fo=Fo,
        Gmedo=Gmedo,
        Gmedt=Gmedt,
        eval_simple_355=eval_simple_355,
        eval_S7matched_355=eval_S7matched_355,
        bases_o=bases_o,
        bases_t=bases_t,
        u_g26ms_o=u_g26ms_o,
        u_g26ms_t=G26 - MS,
        p4=p4,
        bank_o=bank_o,
        bank_t=bank_t,
        Ps_o=Ps_o,
        Ps_t=Ps_t,
        fold_id=fold_id,
        folds=folds,
        y=y,
    )
    log(
        f"  cand2 locked 1ρ(S7)={spearman_dist(cand2,s7):.6e} "
        f"simple={roc_auc_score(y,eval_simple_355):.7f} S7m={roc_auc_score(y,eval_S7matched_355):.7f}"
    )
    return pack


def g26_status():
    out = {"has_test_5fold": False, "files": [], "status": None}
    d = RECON / "goodpjw2008__s6e11-target-first-gpt2-seed-2026"
    out["files"] = sorted(p.name for p in d.glob("*") if not p.name.startswith("._"))
    out["has_test_5fold"] = any("test_5fold" in f for f in out["files"])
    try:
        import subprocess

        exe = [str(KAG)] if KAG.exists() else ["/opt/homebrew/bin/python3", "-m", "kaggle"]
        env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
        r = subprocess.check_output(
            exe + ["datasets", "files", "goodpjw2008/s6e11-target-first-gpt2-seed-2026"],
            text=True,
            timeout=60,
            env=env,
        )
        out["files_remote"] = r
        out["has_test_5fold_remote"] = "test_5fold" in r.lower()
    except Exception as e:
        out["status"] = str(e)[:120]
    return out
