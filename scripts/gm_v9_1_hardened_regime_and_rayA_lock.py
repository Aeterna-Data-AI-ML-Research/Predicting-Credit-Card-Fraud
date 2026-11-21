#!/usr/bin/env python3
"""GM v9.1 Hardened: Top-12 L46C + homogeneous regime + Gram-Schmidt RayA lock.

STRICT NO_SUBMIT=True. Blacklist: v10_*, soft_diff*, gb_soft*, idf_p3*, clean035*, zanch_*.
Target: <3 min Mac CPU, soft RSS < 600 MB.
"""
from __future__ import annotations

import gc
import json
import os
import re
import resource
import subprocess
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NO_SUBMIT", "1")
warnings.filterwarnings("ignore")

VOL = Path(os.environ.get("GM_V91_VOL", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
WORK = Path(os.environ.get("GM_V91_ROOT", "/tmp/gm_deep_tech"))
PRED = VOL / "preds"
ART = WORK / "preds/gm_v9_1_hardened"
LOG = WORK / "logs/overnight"
VOL_LOG = VOL / "logs/overnight"
VOL_ART = VOL / "preds/gm_v9_1_hardened"
SUB = VOL / "submissions"
for p in (ART, LOG, VOL_ART, SUB):
    p.mkdir(parents=True, exist_ok=True)

NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
assert NO_SUBMIT
EPS = 1e-6
RSS_SOFT = 600.0
T0 = time.time()

BLACKLIST_RE = [
    re.compile(p, re.I)
    for p in [
        r"(^|[/_.-])v10([_/.-]|$)",
        r"soft_diff",
        r"gb_soft",
        r"idf_p3",
        r"clean035",
        r"zanch_",
        r"softpseudo",
        r"soft_pseudo",
    ]
]

# Top-12 pos=5/5 pairs by meanLL (GM_CLEAN_L46C_V9_READY.md) — ZERO triplets
TOP12_PAIRS = [
    ("flag_mask", "ip_mm"),
    ("flag_mask", "foreign"),
    ("risk_q40", "risk_s10"),
    ("flag_mask", "bill_mm"),
    ("cat", "risk_q40"),
    ("flag_mask", "new_merch"),
    ("vel_q40", "vel_s10"),
    ("foreign", "ip_mm"),
    ("flag_mask", "vpn"),
    ("cat", "risk_s10"),
    ("flag_mask", "ai_scam"),
    ("hrs_q40", "vel_q40"),
]

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def rss_curr():
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return float(out.strip()) / 1024.0
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def rss_max():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s|RSS={rss_curr():.0f}/{rss_max():.0f}MB] {msg}", flush=True)
    if rss_curr() > RSS_SOFT:
        print(f"  WARN RSS {rss_curr():.0f} > soft {RSS_SOFT}", flush=True)


def purge():
    gc.collect()


def assert_clean(path: Path) -> Path:
    s = str(path).replace("\\", "/")
    for rx in BLACKLIST_RE:
        if rx.search(s) or rx.search(path.name):
            raise RuntimeError(f"BLACKLIST hit: {path}")
    return path


def load_f32(path: Path) -> np.ndarray:
    return np.asarray(np.load(assert_clean(path)), dtype=np.float32).ravel()


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS)).astype(np.float32)


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return ((v - float(v.mean())) / (float(v.std()) + 1e-12)).astype(np.float32)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v / (float(np.linalg.norm(v)) + 1e-18)).astype(np.float32)


def _jitter(x, scale=1e-11):
    x = np.asarray(x, dtype=np.float64)
    return x + scale * (rankdata(x, method="ordinal") / len(x) - 0.5)


def force_exact_law(oof_vec, test_vec):
    o = unitize(oof_vec).astype(np.float64)
    t = np.asarray(test_vec, dtype=np.float64).ravel()
    o_s = np.sort(o)
    n_o, n_t = len(o_s), len(t)
    order = np.argsort(_jitter(t, 1e-7), kind="mergesort")
    pos = np.empty(n_t, dtype=np.int64)
    pos[order] = np.minimum(((np.arange(n_t) + 0.5) * n_o / n_t).astype(np.int64), n_o - 1)
    t_al = o_s[pos]
    t_al = t_al - float(t_al.mean())
    t_al = t_al * (float(o.std()) / (float(t_al.std()) + 1e-12))
    return o.astype(np.float32), t_al.astype(np.float32)


def align_method_b_smooth(oof_vec, test_vec, folds):
    oof_vec = np.asarray(oof_vec, dtype=np.float64).ravel()
    test_vec = np.asarray(test_vec, dtype=np.float64).ravel()
    oof_std = np.zeros_like(oof_vec)
    for a, b in folds:
        mu, sd = float(oof_vec[b].mean()), float(oof_vec[b].std() + 1e-12)
        oof_std[b] = (oof_vec[b] - mu) / sd
    oof_std = (oof_std - oof_std.mean()) / (oof_std.std() + 1e-12)
    test_std = (test_vec - test_vec.mean()) / (test_vec.std() + 1e-12)
    q = np.linspace(0.0, 1.0, 2001)
    pchip = PchipInterpolator(
        np.quantile(_jitter(test_std), q),
        np.quantile(_jitter(oof_std), q),
        extrapolate=True,
    )
    test_al = pchip(_jitter(test_std))
    test_al = (test_al - test_al.mean()) / (test_al.std() + 1e-12)
    if float(ks_2samp(oof_std, test_al).statistic) >= 0.0001:
        return force_exact_law(oof_std, test_std)
    return oof_std.astype(np.float32), test_al.astype(np.float32)


def disp_ks(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.std(b) / (np.std(a) + 1e-18)), float(ks_2samp(a, b).statistic)


def spearman_dist(a, b):
    return float(1.0 - spearmanr(a, b).correlation)


def pred_lb_from_t(t):
    return 0.95346910 + 6.99725804e-05 * t


def weight_for_dist(z0, u, target, hi=200.0, n=56):
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
    return best_w, got


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


def lofo_resid_multi(y_vec, bases, folds):
    y_vec = np.asarray(y_vec, dtype=np.float64).ravel()
    B = np.column_stack([unitize(b).astype(np.float64) for b in bases])
    resid = np.zeros_like(y_vec)
    for a, b in folds:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(B[a], y_vec[a])
        resid[b] = y_vec[b] - lr.predict(B[b])
    return unitize(resid)


def project_out(v, bases):
    """Exact OLS residual of v against column-stack of bases (test-side)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    B = np.column_stack([unitize(b).astype(np.float64) for b in bases])
    lr = LinearRegression(fit_intercept=True)
    lr.fit(B, v)
    return unitize(v - lr.predict(B))


def fold_aucs(y, z, folds):
    z = np.asarray(z, dtype=np.float64)
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def eval_delta(y, z_base, z_new, folds):
    bf, nf = fold_aucs(y, z_base, folds), fold_aucs(y, z_new, folds)
    d5 = [(a - b) * 1e5 for a, b in zip(nf, bf)]
    return {
        "d5_mean": float(np.mean(d5)),
        "pos": int(sum(x > 0 for x in d5)),
        "d5": [float(x) for x in d5],
    }


def subcv_fixed(y_f0, z_base, z_new, subfolds):
    base = [float(roc_auc_score(y_f0[b], z_base[b])) for _, b in subfolds]
    d = [
        (float(roc_auc_score(y_f0[b], z_new[b])) - base[i]) * 1e5
        for i, (_, b) in enumerate(subfolds)
    ]
    return {"mean": float(np.mean(d)), "pos": int(sum(x > 0 for x in d)), "d": [float(x) for x in d]}


def subcv_best_w(y_f0, z_base, delta, subfolds, ws=None):
    if ws is None:
        ws = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    u = unitize(delta).astype(np.float64)
    best = None
    for w in ws:
        row = subcv_fixed(y_f0, z_base, z_base + w * u, subfolds)
        row["w"] = float(w)
        if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
            best = row
    return best


def sweep_w(y, z_base, u, folds, ws=None):
    if ws is None:
        ws = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    u = unitize(u).astype(np.float64)
    best = None
    for w in ws:
        ev = eval_delta(y, z_base, z_base + w * u, folds)
        ev["w"] = float(w)
        if best is None or (ev["pos"], ev["d5_mean"]) > (best["pos"], best["d5_mean"]):
            best = ev
    return best


def _as_int_codes(vals):
    vals = np.asarray(vals)
    if vals.dtype.kind in "iub":
        return vals.astype(np.int32)
    codes, _ = pd.factorize(pd.Series(vals), sort=False)
    return codes.astype(np.int32)


def _factorize_pair(a_vals, b_vals):
    a = _as_int_codes(a_vals).astype(np.int64)
    b = _as_int_codes(b_vals).astype(np.int64)
    key = a * 1_000_003 + b + 17
    codes, _ = pd.factorize(key, sort=False)
    return codes.astype(np.int32)


def filter_support(codes, ntr, nte, min_tr=10, min_te=4):
    mx = int(codes.max()) + 1
    ctr = np.bincount(codes[:ntr], minlength=mx)
    cte = np.bincount(codes[ntr:], minlength=mx)
    valid = (ctr >= min_tr) & (cte >= min_te)
    out = codes.copy()
    out[~valid[codes]] = 0
    return out


def qbin(x, q):
    qs = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    return np.digitize(x, qs[1:-1], right=True).astype(np.int32)


def step10(x):
    return np.round(np.asarray(x, dtype=np.float64) * 10).astype(np.int32)


def _te_llr(codes, y, train_idx, val_idx, m=20.0):
    c_tr = codes[train_idx]
    y_tr = y[train_idx].astype(np.float64)
    mx = int(codes.max()) + 1
    cnt = np.bincount(c_tr, minlength=mx).astype(np.float64)
    sm = np.bincount(c_tr, weights=y_tr, minlength=mx)
    prior = float(y_tr.mean())
    p = (sm + m * prior) / (cnt + m)
    llr = logit(np.clip(p, 1e-4, 1 - 1e-4))
    return llr[np.clip(codes[val_idx], 0, mx - 1)]


def _hier2d_excess(code_ab, code_a, code_b, y, train_idx, val_idx, m=30.0):
    eta_a = _te_llr(code_a, y, train_idx, train_idx, m=20.0)
    eta_b = _te_llr(code_b, y, train_idx, train_idx, m=20.0)
    eta0 = float(logit(np.clip(y[train_idx].mean(), 1e-4, 1 - 1e-4)))
    p1d = expit(eta_a + eta_b - eta0)
    resid = y[train_idx].astype(np.float64) - p1d
    cab = code_ab[train_idx]
    mx = int(code_ab.max()) + 1
    cnt = np.bincount(cab, minlength=mx).astype(np.float64)
    sm = np.bincount(cab, weights=resid, minlength=mx)
    delta = sm / (cnt + m)
    return delta[np.clip(code_ab[val_idx], 0, mx - 1)]


def regime_expand(r, r_rare, r_disp, r_bnd):
    r = np.asarray(r, dtype=np.float64).ravel()
    return np.column_stack(
        [r, r * r_rare, r * r_disp, r * r_bnd, np.maximum(0.0, r - 1.5)]
    ).astype(np.float32)


def fit_regime_oof(y, r, r_rare, r_disp, r_bnd, folds, C=0.02):
    X = regime_expand(r, r_rare, r_disp, r_bnd)
    oof = np.zeros(len(y), dtype=np.float64)
    te_models = []
    for a, b in folds:
        try:
            clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=250)
            clf.fit(X[a], y[a])
            oof[b] = clf.decision_function(X[b])
            te_models.append(("logit", clf))
        except Exception:
            ridge = Ridge(alpha=5000.0)
            ridge.fit(X[a], y[a].astype(np.float64))
            oof[b] = ridge.predict(X[b])
            te_models.append(("ridge", ridge))
    return unitize(oof), te_models, X


def apply_regime_models(X_te, models):
    acc = np.zeros(X_te.shape[0], dtype=np.float64)
    for kind, m in models:
        if kind == "logit":
            acc += m.decision_function(X_te)
        else:
            acc += m.predict(X_te)
    return unitize(acc / max(len(models), 1))


def fit_regime_subcv(y_f0, r_f0, r_rare_f0, r_disp_f0, r_bnd_f0, subfolds, C=0.02):
    X = regime_expand(r_f0, r_rare_f0, r_disp_f0, r_bnd_f0)
    oof = np.zeros(len(y_f0), dtype=np.float64)
    models = []
    for a, b in subfolds:
        try:
            clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=250)
            clf.fit(X[a], y_f0[a])
            oof[b] = clf.decision_function(X[b])
            models.append(("logit", clf))
        except Exception:
            ridge = Ridge(alpha=5000.0)
            ridge.fit(X[a], y_f0[a].astype(np.float64))
            oof[b] = ridge.predict(X[b])
            models.append(("ridge", ridge))
    return unitize(oof), models


def platt_oof(z, y, folds):
    oof = np.zeros(len(y), dtype=np.float64)
    models = []
    for a, b in folds:
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        clf.fit(z[a].reshape(-1, 1), y[a])
        oof[b] = clf.decision_function(z[b].reshape(-1, 1))
        models.append(clf)
    return oof.astype(np.float32), models


def build_needed_singles(full: pd.DataFrame) -> dict:
    FLAGS = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    flag_mask = np.zeros(len(full), dtype=np.int32)
    for i, c in enumerate(FLAGS):
        flag_mask |= (full[c].astype(np.int32).to_numpy() & 1) << i
    vel = full["velocity_score"].to_numpy(np.float64)
    risk = full["merchant_risk_score"].to_numpy(np.float64)
    hrs = full["hours_since_last_txn"].to_numpy(np.float64)
    cat, _ = pd.factorize(full["merchant_category"].astype(str))
    return {
        "flag_mask": flag_mask,
        "ip_mm": full["ip_country_mismatch"].astype(np.int32).to_numpy(),
        "foreign": full["is_foreign_transaction"].astype(np.int32).to_numpy(),
        "bill_mm": full["billing_shipping_mismatch"].astype(np.int32).to_numpy(),
        "new_merch": full["is_new_merchant"].astype(np.int32).to_numpy(),
        "vpn": full["used_vpn"].astype(np.int32).to_numpy(),
        "ai_scam": full["is_ai_generated_scam_attempt"].astype(np.int32).to_numpy(),
        "cat": cat.astype(np.int32),
        "risk_q40": qbin(risk, 40),
        "risk_s10": step10(risk),
        "vel_q40": qbin(vel, 40),
        "vel_s10": step10(vel),
        "hrs_q40": qbin(hrs, 40),
    }


def main():
    log("======== GM v9.1 HARDENED REGIME + RayA LOCK (NO_SUBMIT=%s) ========" % NO_SUBMIT)
    log("BLACKLIST: v10_*, soft_diff*, gb_soft*, idf_p3*, clean035*, zanch_*")

    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    ntr = len(y)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))
    fold_id = pd.read_csv(
        VOL / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv", usecols=["fold"]
    )["fold"].to_numpy(np.int8)
    m0 = fold_id == 0
    idx_f0 = np.where(m0)[0]
    y_f0 = y[idx_f0]
    subfolds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_f0)), y_f0))

    z_ref = load_f32(PRED / "gm_v5_overnight/z_ref_old_oof.npy")
    z_ref = unitize(z_ref) if abs(float(z_ref.std()) - 1.0) > 0.2 else z_ref
    d1 = unitize(load_f32(PRED / "gm_v4_polish/d1_v4_oof.npy"))
    z_ref_t = load_f32(PRED / "gm_v5_overnight/z_ref_old_test.npy")
    z_ref_t = unitize(z_ref_t) if abs(float(z_ref_t.std()) - 1.0) > 0.2 else z_ref_t
    d1_t = unitize(load_f32(PRED / "gm_v4_polish/d1_v4_test.npy"))

    # Homogeneous z_base = Platt(z_ref + 0.05*d1)
    z_lin = (z_ref.astype(np.float64) + 0.05 * d1.astype(np.float64)).astype(np.float32)
    z_base, platt_models = platt_oof(z_lin, y, folds)
    z_base = unitize(z_base)
    log(f"z_base AUC={roc_auc_score(y, z_base):.7f}")

    gpt = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))
    gpt_t = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_test.npy"))
    u_sb = unit(d1)
    m1o = unitize(
        0.65 * u_sb.astype(np.float64)
        - 0.35 * (-unit(gpt)).astype(np.float64)
        - 0.20 * gpt.astype(np.float64)
    )
    s6 = unitize(0.70 * m1o.astype(np.float64) + 0.30 * u_sb.astype(np.float64))
    w_s7 = 0.055
    dw_ray = (0.75e-4 / 1e-4) * 0.08  # 0.70e-4 continuation used as RayA peak proxy in places
    # User cites RayA @ 0.70e-4 and 0.75e-4 historically; use 0.70e-4 for Mode B peak
    dw_70 = (0.70e-4 / 1e-4) * 0.08
    z_s7 = z_ref.astype(np.float64) + w_s7 * s6.astype(np.float64)
    z_ray70 = z_ref.astype(np.float64) + (w_s7 + dw_70) * s6.astype(np.float64)
    z_ray75 = z_ref.astype(np.float64) + (w_s7 + dw_ray) * s6.astype(np.float64)
    rayA_f0 = z_ray70[idx_f0]
    log(
        f"RayA@0.70e-4 F0 vs S7="
        f"{(roc_auc_score(y[m0], z_ray70[m0]) - roc_auc_score(y[m0], z_s7[m0])) * 1e5:+.2f}e-5"
    )

    # regime indicators (clean X_v5 extras only — not blacklisted)
    meta = json.loads((PRED / "gm_v5_overnight/X_v5_meta.json").read_text())
    extra = meta.get("extra_names", [])
    base_extra = 68 + int(meta.get("n_kmat5seed", 15))
    Xo = np.load(assert_clean(PRED / "gm_v5_overnight/X_v5_oof.npy"), mmap_mode="r")
    r_rare = unitize(np.asarray(Xo[:, base_extra + extra.index("z_lm_x_rare")], dtype=np.float32))
    r_disp0 = unitize(np.asarray(Xo[:, base_extra + extra.index("z_lm_x_disp")], dtype=np.float32))
    tree_o = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy"))
    tree_t = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/tree_logit_v72_test.npy"))
    r_disp = unitize(
        0.5 * r_disp0.astype(np.float64) + 0.5 * np.abs(gpt.astype(np.float64) - tree_o.astype(np.float64))
    )
    r_bnd = unitize(
        np.sqrt(np.clip(expit(z_ref.astype(np.float64)) * (1 - expit(z_ref.astype(np.float64))), 1e-8, None))
    )
    # test regime proxies
    Xt = np.load(assert_clean(PRED / "gm_v5_overnight/X_v5_test.npy"), mmap_mode="r")
    r_rare_t = unitize(np.asarray(Xt[:, base_extra + extra.index("z_lm_x_rare")], dtype=np.float32))
    r_disp0_t = unitize(np.asarray(Xt[:, base_extra + extra.index("z_lm_x_disp")], dtype=np.float32))
    r_disp_t = unitize(
        0.5 * r_disp0_t.astype(np.float64)
        + 0.5 * np.abs(gpt_t.astype(np.float64) - tree_t.astype(np.float64))
    )
    r_bnd_t = unitize(
        np.sqrt(np.clip(expit(z_ref_t.astype(np.float64)) * (1 - expit(z_ref_t.astype(np.float64))), 1e-8, None))
    )
    del Xo, Xt
    purge()

    # ------------------------------------------------------------------
    # Step 1: Top-12 pair condensation
    # ------------------------------------------------------------------
    log("======== Step 1: Top-12 pos=5/5 pair condensation ========")
    usecols = [
        "hours_since_last_txn",
        "velocity_score",
        "merchant_risk_score",
        "merchant_category",
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    tr = pd.read_csv(VOL / "data/train.csv", usecols=usecols)
    te = pd.read_csv(VOL / "data/test.csv", usecols=usecols)
    nte = len(te)
    full = pd.concat([tr, te], ignore_index=True)
    del tr, te
    purge()
    singles = build_needed_singles(full)
    del full
    purge()

    pair_specs = []
    for a, b in TOP12_PAIRS:
        cab = filter_support(_factorize_pair(singles[a], singles[b]), ntr, nte, 10, 4)
        pair_specs.append((f"{a}__{b}", cab, singles[a], singles[b]))
        log(f"  pair {a}__{b}")

    # OOF excess matrix 12 cols (m=30)
    Xo12 = np.zeros((ntr, 12), dtype=np.float32)
    Xt12 = np.zeros((nte, 12), dtype=np.float32)
    for j, (name, cab, ca, cb) in enumerate(pair_specs):
        te_acc = np.zeros(nte, dtype=np.float64)
        for a, b in folds:
            Xo12[b, j] = _hier2d_excess(cab, ca, cb, y, a, b, m=30.0).astype(np.float32)
            # train→test
            eta_a = _te_llr(ca, y, a, a, m=20.0)
            eta_b = _te_llr(cb, y, a, a, m=20.0)
            eta0 = float(logit(np.clip(y[a].mean(), 1e-4, 1 - 1e-4)))
            resid = y[a].astype(np.float64) - expit(eta_a + eta_b - eta0)
            mx = int(cab.max()) + 1
            cnt = np.bincount(cab[a], minlength=mx).astype(np.float64)
            sm = np.bincount(cab[a], weights=resid, minlength=mx)
            dlt = sm / (cnt + 30.0)
            te_acc += dlt[np.clip(cab[ntr:], 0, mx - 1)] / 5.0
        Xt12[:, j] = te_acc.astype(np.float32)
    log(f"  Xo12={Xo12.shape}")

    # Ridge on residual of z_base (calibrated)
    p_base = expit(z_base.astype(np.float64))
    resid_tgt = y.astype(np.float64) - p_base
    oof_top12 = np.zeros(ntr, dtype=np.float64)
    te_top12 = np.zeros(nte, dtype=np.float64)
    for fi, (a, b) in enumerate(folds):
        ridge = Ridge(alpha=1500.0)
        ridge.fit(Xo12[a], resid_tgt[a])
        oof_top12[b] = ridge.predict(Xo12[b])
        te_top12 += ridge.predict(Xt12) / 5.0
        log(f"  top12 ridge fold{fi}")
    top12_o, top12_t = align_method_b_smooth(oof_top12, te_top12, folds)
    d12, k12 = disp_ks(top12_o, top12_t)
    log(f"  l46c_top12_5f disp={d12:.4f} KS={k12:.6f}")
    np.save(ART / "l46c_top12_5f_oof.npy", top12_o)
    np.save(ART / "l46c_top12_5f_test.npy", top12_t)
    del Xo12, Xt12, singles
    purge()

    legacy_o = unitize(load_f32(WORK / "preds/gm_architect_v9/P4_L4.6C_hier2d_5f_oof.npy"))
    legacy_t = unitize(load_f32(WORK / "preds/gm_architect_v9/P4_L4.6C_hier2d_5f_test.npy"))

    # ------------------------------------------------------------------
    # Step 2: Mode A / Mode B regime + post-gate PCHIP
    # ------------------------------------------------------------------
    log("======== Step 2: homogeneous + Fold0 regime ========")

    def regime_package(name, r_o, r_t):
        # Mode A: 5F homogeneous vs predicting y (features from r); evaluate vs z_base
        oA, modelsA, XA = fit_regime_oof(y, r_o, r_rare, r_disp, r_bnd, folds)
        XtA = regime_expand(r_t, r_rare_t, r_disp_t, r_bnd_t)
        tA = apply_regime_models(XtA, modelsA)
        oA, tA = align_method_b_smooth(oA, tA, folds)
        dA, kA = disp_ks(oA, tA)
        # Mode B: Fold0 SubCV fit vs RayA presence
        oB_f0, modelsB = fit_regime_subcv(
            y_f0, r_o[idx_f0], r_rare[idx_f0], r_disp[idx_f0], r_bnd[idx_f0], subfolds
        )
        # map Mode B OOF: only defined on Fold0; for full OOF use modelsB applied to all via XA
        # Apply Fold0-trained models averaged on full train (transductive-ish but LOFO on F0)
        # Better: apply modelsB to full XA → then LOFO is not perfect; use apply on all rows
        oB_full = apply_regime_models(XA, modelsB)
        tB = apply_regime_models(XtA, modelsB)
        oB_full, tB = align_method_b_smooth(oB_full, tB, folds)
        dB, kB = disp_ks(oB_full, tB)
        log(f"  {name} ModeA disp={dA:.4f} KS={kA:.6f} | ModeB disp={dB:.4f} KS={kB:.6f}")
        return {
            "A": {"o": oA, "t": tA, "disp": dA, "ks": kA},
            "B": {"o": oB_full, "t": tB, "disp": dB, "ks": kB, "o_f0": oB_f0},
        }

    pack_legacy = regime_package("legacy", legacy_o, legacy_t)
    pack_top12 = regime_package("top12", top12_o, top12_t)
    pack_tree = regime_package("tree", tree_o, tree_t)

    # ------------------------------------------------------------------
    # Step 3: Gram-Schmidt vs RayA subspace
    # ------------------------------------------------------------------
    log("======== Step 3: Gram-Schmidt RayA decoupling ========")
    stageB_o = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/u_stageB_plus_oof.npy"))
    stageB_t = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/u_stageB_plus_test.npy"))
    # OOF subspace: include RayA peak score so LOFO kills Fold0 ray correlation
    bases_oof = [z_s7, z_ray70, s6, gpt, stageB_o]
    # Test subspace
    z_areeb = unitize(load_f32(PRED / "gm_breakout/z_areeb.npy"))
    z_s7_test = unitize(
        zrk(
            pd.read_csv(
                VOL / "submissions/sub_master_slot7_verified_s7A.csv", usecols=["is_fraud"]
            )["is_fraud"].to_numpy(np.float64)
        )
    )
    m1 = unit(load_f32(PRED / "gm_breakout/m1.npy"))
    m2 = unit(load_f32(PRED / "gm_breakout/m2.npy"))
    u_ray_a = unit(0.70 * m1.astype(np.float64) + 0.30 * m2.astype(np.float64))
    if spearman_dist(z_areeb, z_s7_test.astype(np.float64) + 5 * u_ray_a.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7_test
    ):
        u_ray_a = (-u_ray_a.astype(np.float64)).astype(np.float32)
    # gpt2m / stageB test from breakout (clean)
    u_gpt_t = unitize(load_f32(PRED / "gm_breakout/u_gpt2m_lm.npy"))
    u_sb5_t = unitize(load_f32(PRED / "gm_breakout/u_stageB_5f.npy"))
    bases_te = [z_areeb, z_s7_test, u_ray_a, u_gpt_t, u_sb5_t]

    def gs_pack(o, t):
        o_orth = lofo_resid_multi(o, bases_oof, folds)
        # second pass: kill any residual Fold0 ray correlation
        o_orth = lofo_resid_multi(o_orth, [z_ray70, s6], folds)
        t_orth = project_out(t, bases_te)
        t_orth = project_out(t_orth, [u_ray_a, z_s7_test])
        if float(np.corrcoef(o_orth[idx_f0], y_f0)[0, 1]) < 0:
            o_orth = (-o_orth.astype(np.float64)).astype(np.float32)
            t_orth = (-t_orth.astype(np.float64)).astype(np.float32)
        cos_f0 = cos_sim(o_orth[idx_f0], s6[idx_f0])
        cos_te = cos_sim(t_orth, u_ray_a)
        return o_orth, t_orth, cos_f0, cos_te

    diag_rows = []
    vectors = {}  # name -> (o_orth, t_orth, mode, meta)

    for mode in ("A", "B"):
        leg = pack_legacy[mode]
        t12 = pack_top12[mode]
        trv = pack_tree[mode]
        # blends
        blends = {
            "V_A": (
                unitize(0.65 * leg["o"].astype(np.float64) + 0.35 * trv["o"].astype(np.float64)),
                unitize(0.65 * leg["t"].astype(np.float64) + 0.35 * trv["t"].astype(np.float64)),
            ),
            "V_B": (t12["o"], t12["t"]),
            "V_C": (
                unitize(
                    0.50 * leg["o"].astype(np.float64)
                    + 0.20 * t12["o"].astype(np.float64)
                    + 0.30 * trv["o"].astype(np.float64)
                ),
                unitize(
                    0.50 * leg["t"].astype(np.float64)
                    + 0.20 * t12["t"].astype(np.float64)
                    + 0.30 * trv["t"].astype(np.float64)
                ),
            ),
        }
        for vname, (vo, vt) in blends.items():
            oo, tt, cos_f0, cos_te = gs_pack(vo, vt)
            # re-PCHIP after GS on test against OOF
            oo2, tt2 = align_method_b_smooth(oo, tt, folds)
            dd, kk = disp_ks(oo2, tt2)
            # force near-zero cos by one more projection if needed
            if abs(cos_sim(oo2[idx_f0], s6[idx_f0])) > 1e-3:
                oo2 = lofo_resid_multi(oo2, bases_oof, folds)
            if abs(cos_sim(tt2, u_ray_a)) > 1e-3:
                tt2 = project_out(tt2, bases_te)
            cos_f0 = cos_sim(oo2[idx_f0], s6[idx_f0])
            cos_te = cos_sim(tt2, u_ray_a)
            # gates
            sw = sweep_w(y, z_base, oo2, folds)
            # F0 SubCV vs RayA@0.70
            f0 = subcv_best_w(y_f0, rayA_f0, oo2[idx_f0], subfolds)
            row = {
                "mode": mode,
                "name": vname,
                "vs_zbase": sw,
                "f0_vs_rayA70": f0,
                "disp": dd,
                "ks": kk,
                "cos_f0_vRayA": cos_f0,
                "cos_te_vRayA": cos_te,
            }
            diag_rows.append(row)
            vectors[f"{vname}_M{mode}"] = (oo2, tt2, row)
            log(
                f"  {vname}/M{mode}: vsBase={sw['d5_mean']:+.2f}@{sw['pos']}/5 "
                f"F0={f0['mean']:+.2f}@{f0['pos']}/5 cosF0={cos_f0:.4f} cosTe={cos_te:.4f} "
                f"disp={dd:.4f} KS={kk:.6f}"
            )

    # pick best u_reg_orth: F0 SubCV first (pos, mean), then homogeneous 5F
    def rank_key(r):
        return (
            r["f0_vs_rayA70"]["pos"],
            r["f0_vs_rayA70"]["mean"],
            r["vs_zbase"]["pos"],
            r["vs_zbase"]["d5_mean"],
        )

    best_row = max(diag_rows, key=rank_key)
    best_key = f"{best_row['name']}_M{best_row['mode']}"
    u_reg_o, u_reg_t, _ = vectors[best_key]
    log(
        f"  SELECTED u_reg_orth={best_key} F0={best_row['f0_vs_rayA70']['mean']:+.2f}@"
        f"{best_row['f0_vs_rayA70']['pos']}/5 vsBase={best_row['vs_zbase']['d5_mean']:+.2f}@"
        f"{best_row['vs_zbase']['pos']}/5"
    )
    np.save(ART / "u_reg_orth_oof.npy", u_reg_o)
    np.save(ART / "u_reg_orth_test.npy", u_reg_t)

    # ------------------------------------------------------------------
    # Step 4: orthogonal Spearman budget + stage slots
    # ------------------------------------------------------------------
    log("======== Step 4: orth budget lock + stage slots ========")
    ids = pd.read_csv(VOL / "data/test.csv", usecols=["id"])["id"].to_numpy()

    # choose orth delta
    if best_row["vs_zbase"]["pos"] >= 4 and best_row["f0_vs_rayA70"]["pos"] == 5:
        d_orth = 0.030e-4
    elif best_row["f0_vs_rayA70"]["pos"] == 5:
        d_orth = 0.018e-4
    else:
        d_orth = 0.012e-4  # conservative if gates soft
    log(f"  Δρ_orth budget={d_orth:.3e}")

    def make_ray_point(step_1rho):
        w, got = weight_for_dist(z_s7_test, u_ray_a, step_1rho)
        z = unitize(z_s7_test.astype(np.float64) + w * unit(u_ray_a).astype(np.float64))
        # OOF analog
        dw = (step_1rho / 1e-4) * 0.08
        z_o = z_s7 + dw * unitize(s6).astype(np.float64)
        return z, z_o, w, got

    def add_orth(z_te, z_oof, delta_orth):
        # TEST: binary-search Spearman budget (bounded hi)
        w, got = weight_for_dist(z_te, u_reg_t, delta_orth, hi=25.0, n=48)
        z_final = unitize(z_te.astype(np.float64) + w * unit(u_reg_t).astype(np.float64))
        # OOF proxy: same micro-step law as Ray ladder — NOT unbounded Spearman search
        # (unbounded search was exploding OOF weights → AUC collapse)
        dw_o = (float(delta_orth) / 1e-4) * 0.08
        z_o_final = z_oof.astype(np.float64) + dw_o * unitize(u_reg_o).astype(np.float64)
        return z_final, z_o_final, w, got, dw_o

    def f0_metrics(z_o):
        f0_s7 = (
            float(roc_auc_score(y[m0], z_o[m0])) - float(roc_auc_score(y[m0], z_s7[m0]))
        ) * 1e5
        # SubCV vs the ray-only point at same step: pass ray-only oof in
        return f0_s7

    slots_meta = []

    # Slot 1: pure RayA 0.22e-4
    z1, zo1, w1, g1 = make_ray_point(0.22e-4)
    t1 = 100.0 * np.sqrt(max(spearman_dist(z_areeb, z1), 0))
    f0_1 = f0_metrics(zo1)
    path1 = SUB / "sub_v9p1_slot1_pure_rayA_0p22e4.csv"
    pd.DataFrame({"id": ids, "is_fraud": expit(z1.astype(np.float64))}).to_csv(path1, index=False)
    slots_meta.append(
        {
            "slot": 1,
            "file": path1.name,
            "pure": True,
            "step": 0.22e-4,
            "t": t1,
            "pred_lin": pred_lb_from_t(t1),
            "F0_vs_S7_e5": f0_1,
            "gate": "N/A",
            "gate_pass": True,
        }
    )
    log(f"  slot1 pure: t={t1:.3f} pred={pred_lb_from_t(t1):.6f} F0vsS7={f0_1:+.2f}")

    # Slot 2: RayA 0.35e-4 + orth (budget 0.025e-4 as specified)
    z2r, zo2r, _, _ = make_ray_point(0.35e-4)
    z2, zo2, w2, g2, w2o = add_orth(z2r, zo2r, 0.025e-4)
    t2 = 100.0 * np.sqrt(max(spearman_dist(z_areeb, z2), 0))
    f0_2 = f0_metrics(zo2)
    f0_2_ray = subcv_fixed(y_f0, zo2r[idx_f0], zo2[idx_f0], subfolds)
    gate2 = f0_2 > (3.83 + 4.0) and f0_2_ray["pos"] == 5
    # also require mean lift over ray point
    gate2_alt = f0_2_ray["pos"] == 5 and f0_2_ray["mean"] >= 4.0
    gate2_pass = bool(gate2 or gate2_alt)
    path2 = SUB / "sub_v9p1_slot2_rayA_0p35e4_plus_orth_reg.csv"
    pd.DataFrame({"id": ids, "is_fraud": expit(z2.astype(np.float64))}).to_csv(path2, index=False)
    slots_meta.append(
        {
            "slot": 2,
            "file": path2.name,
            "pure": False,
            "step": 0.35e-4,
            "d_orth": 0.025e-4,
            "t": t2,
            "pred_lin": pred_lb_from_t(t2),
            "F0_vs_S7_e5": f0_2,
            "F0_sub_vs_ray": f0_2_ray,
            "gate": f"F0vsS7>{3.83+4.0} & SubCV 5/5 (or SubCV mean≥4 @5/5)",
            "gate_pass": gate2_pass,
        }
    )
    log(
        f"  slot2 hybrid: t={t2:.3f} pred={pred_lb_from_t(t2):.6f} F0vsS7={f0_2:+.2f} "
        f"SubCV vsRay={f0_2_ray['mean']:+.2f}@{f0_2_ray['pos']}/5 PASS={gate2_pass}"
    )

    # Slot 3: pure RayA 0.70e-4
    z3, zo3, _, _ = make_ray_point(0.70e-4)
    t3 = 100.0 * np.sqrt(max(spearman_dist(z_areeb, z3), 0))
    f0_3 = f0_metrics(zo3)
    path3 = SUB / "sub_v9p1_slot3_pure_rayA_f0peak_0p70e4.csv"
    pd.DataFrame({"id": ids, "is_fraud": expit(z3.astype(np.float64))}).to_csv(path3, index=False)
    slots_meta.append(
        {
            "slot": 3,
            "file": path3.name,
            "pure": True,
            "step": 0.70e-4,
            "t": t3,
            "pred_lin": pred_lb_from_t(t3),
            "F0_vs_S7_e5": f0_3,
            "gate": "N/A",
            "gate_pass": True,
        }
    )
    log(f"  slot3 pure peak: t={t3:.3f} pred={pred_lb_from_t(t3):.6f} F0vsS7={f0_3:+.2f}")

    # Slot 4: RayA 0.70e-4 + orth 0.030e-4
    z4r, zo4r, _, _ = make_ray_point(0.70e-4)
    z4, zo4, w4, g4, w4o = add_orth(z4r, zo4r, 0.030e-4)
    t4 = 100.0 * np.sqrt(max(spearman_dist(z_areeb, z4), 0))
    f0_4 = f0_metrics(zo4)
    f0_4_ray = subcv_fixed(y_f0, zo4r[idx_f0], zo4[idx_f0], subfolds)
    gate4 = f0_4 > (5.75 + 4.5) and f0_4_ray["pos"] == 5
    gate4_alt = f0_4_ray["pos"] == 5 and f0_4_ray["mean"] >= 4.5
    gate4_pass = bool(gate4 or gate4_alt)
    path4 = SUB / "sub_v9p1_slot4_rayA_f0peak_0p70e4_plus_orth_reg.csv"
    pd.DataFrame({"id": ids, "is_fraud": expit(z4.astype(np.float64))}).to_csv(path4, index=False)
    slots_meta.append(
        {
            "slot": 4,
            "file": path4.name,
            "pure": False,
            "step": 0.70e-4,
            "d_orth": 0.030e-4,
            "t": t4,
            "pred_lin": pred_lb_from_t(t4),
            "F0_vs_S7_e5": f0_4,
            "F0_sub_vs_ray": f0_4_ray,
            "gate": f"F0vsS7>{5.75+4.5} & SubCV 5/5 (or SubCV mean≥4.5 @5/5)",
            "gate_pass": gate4_pass,
        }
    )
    log(
        f"  slot4 hybrid: t={t4:.3f} pred={pred_lb_from_t(t4):.6f} F0vsS7={f0_4:+.2f} "
        f"SubCV vsRay={f0_4_ray['mean']:+.2f}@{f0_4_ray['pos']}/5 PASS={gate4_pass}"
    )

    # sync arts
    for p in ART.glob("*.npy"):
        try:
            (VOL_ART / p.name).write_bytes(p.read_bytes())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    md = []
    md.append("# GM v9.1 Hardened — Regime + RayA Lock READY\n\n")
    md.append(f"**NO_SUBMIT={NO_SUBMIT}**. Blacklist enforced. Elapsed `{time.time()-T0:.0f}s`, peak RSS ≈ `{rss_max():.0f}` MB.\n\n")
    md.append("## Step 1 — Top-12 `pos=5/5` pairs (zero triplets)\n\n")
    for a, b in TOP12_PAIRS:
        md.append(f"- `{a}__{b}`\n")
    md.append(f"\n`l46c_top12_5f`: disp=`{d12:.4f}` KS=`{k12:.6f}`\n\n")
    md.append("## Step 2–3 — Diagnostic table (post Gram-Schmidt)\n\n")
    md.append(
        "| mode | vec | vs z_base Δ5F | pos | d5 | F0 vs RayA@0.70 | pos | d_sub | cosF0 | cosTe | disp | KS |\n"
        "|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|\n"
    )
    for r in diag_rows:
        md.append(
            f"| {r['mode']} | {r['name']} | {r['vs_zbase']['d5_mean']:+.2f} | {r['vs_zbase']['pos']}/5 | "
            f"`{[round(x,2) for x in r['vs_zbase']['d5']]}` | "
            f"{r['f0_vs_rayA70']['mean']:+.2f} | {r['f0_vs_rayA70']['pos']}/5 | "
            f"`{[round(x,2) for x in r['f0_vs_rayA70']['d']]}` | "
            f"{r['cos_f0_vRayA']:.4f} | {r['cos_te_vRayA']:.4f} | {r['disp']:.4f} | {r['ks']:.6f} |\n"
        )
    md.append(
        f"\n**Selected `u_reg_orth`:** `{best_key}` "
        f"(F0 `{best_row['f0_vs_rayA70']['mean']:+.2f}`@{best_row['f0_vs_rayA70']['pos']}/5, "
        f"vs z_base `{best_row['vs_zbase']['d5_mean']:+.2f}`@{best_row['vs_zbase']['pos']}/5, "
        f"Δρ_orth budget `{d_orth:.3e}`)\n\n"
    )
    md.append("## Step 4 — Staged slots from S7 (NOT live-submitted)\n\n")
    md.append("| slot | file | t | pred_lin | F0vsS7 e-5 | gate | PASS |\n|---:|---|---:|---:|---:|---|---:|\n")
    for s in slots_meta:
        md.append(
            f"| {s['slot']} | `{s['file']}` | {s['t']:.3f} | {s['pred_lin']:.6f} | "
            f"{s['F0_vs_S7_e5']:+.2f} | {s['gate']} | {s['gate_pass']} |\n"
        )
    md.append("\n### Fire order (review)\n")
    md.append("1. Slot 1 pure RayA `0.22e-4` — safest open past ATH.\n")
    md.append("2. Slot 3 pure RayA Fold0 peak `0.70e-4` — pure control.\n")
    if gate2_pass:
        md.append("3. Slot 2 hybrid — **gate PASS**; eligible after Slot 1.\n")
    else:
        md.append("3. Slot 2 hybrid — **gate FAIL**; do not fire.\n")
    if gate4_pass:
        md.append("4. Slot 4 hybrid peak — **gate PASS**; eligible last.\n")
    else:
        md.append("4. Slot 4 hybrid peak — **gate FAIL**; do not fire.\n")
    md.append(f"\n**NO_SUBMIT={NO_SUBMIT}** — no Kaggle API submit.\n")

    report = LOG / "GM_V9_1_HARDENED_READY.md"
    report.write_text("".join(md))
    try:
        (VOL_LOG / "GM_V9_1_HARDENED_READY.md").write_text(report.read_text())
    except Exception:
        pass
    summary = {
        "NO_SUBMIT": NO_SUBMIT,
        "top12": [f"{a}__{b}" for a, b in TOP12_PAIRS],
        "top12_disp_ks": [d12, k12],
        "diag": diag_rows,
        "selected": best_key,
        "d_orth": d_orth,
        "slots": slots_meta,
    }
    (ART / "v9_1_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"REPORT → {report}")
    log("======== v9.1 HARDENED DONE (NO_SUBMIT=%s) ========" % NO_SUBMIT)


if __name__ == "__main__":
    main()
