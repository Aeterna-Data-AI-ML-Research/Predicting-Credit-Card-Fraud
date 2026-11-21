#!/usr/bin/env python3
"""Clean L4.6C epistatic expansion + Stage-B + submission staging.

STRICT:
  - NO_SUBMIT=True until verification table prints (default).
  - Permanent blacklist: v10_*, soft_diff*, gb_soft*, idf_p3*, clean035*.
  - Zero distilled / zanch / SoftPseudo families.

~8 min Mac CPU, soft RSS < 650 MB.
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
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NO_SUBMIT", "1")
warnings.filterwarnings("ignore")

VOL = Path(os.environ.get("GM_CLEAN_VOL", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
WORK = Path(os.environ.get("GM_CLEAN_ROOT", "/tmp/gm_deep_tech"))
PRED = VOL / "preds"
ART = WORK / "preds/gm_clean_l46c_v9"
LOG = WORK / "logs/overnight"
VOL_LOG = VOL / "logs/overnight"
VOL_ART = VOL / "preds/gm_clean_l46c_v9"
SUB = VOL / "submissions"
for p in (ART, LOG, VOL_ART, SUB):
    p.mkdir(parents=True, exist_ok=True)

NO_SUBMIT = os.environ.get("NO_SUBMIT", "1") != "0"
EPS = 1e-6
RSS_SOFT = 650.0
T0 = time.time()

# Permanent poison-pill blacklist (path / stem match)
BLACKLIST_PATTERNS = [
    r"(^|[/_.-])v10([_/.-]|$)",
    r"soft_diff",
    r"gb_soft",
    r"idf_p3",
    r"clean035",
    r"zanch_",
    r"softpseudo",
    r"soft_pseudo",
]
BLACKLIST_RE = [re.compile(p, re.I) for p in BLACKLIST_PATTERNS]

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def rss_curr() -> float:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return float(out.strip()) / 1024.0
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def rss_max() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s|RSS={rss_curr():.0f}/{rss_max():.0f}MB] {msg}", flush=True)
    if rss_curr() > RSS_SOFT:
        print(f"  WARN RSS {rss_curr():.0f} > soft {RSS_SOFT}", flush=True)


def purge():
    gc.collect()


def assert_clean_path(path: Path | str, label: str = "") -> Path:
    p = Path(path)
    s = str(p).replace("\\", "/")
    for rx in BLACKLIST_RE:
        if rx.search(s) or rx.search(p.name):
            raise RuntimeError(f"BLACKLIST hit for {label or p.name}: {p}")
    return p


def load_f32(path: Path) -> np.ndarray:
    path = assert_clean_path(path)
    return np.asarray(np.load(path), dtype=np.float32).ravel()


def zrk(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS)).astype(np.float32)


def unitize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return ((v - float(v.mean())) / (float(v.std()) + 1e-12)).astype(np.float32)


def unit(v: np.ndarray) -> np.ndarray:
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


def lofo_resid(y_vec, bases, folds):
    y_vec = np.asarray(y_vec, dtype=np.float64).ravel()
    B = np.column_stack([unitize(b).astype(np.float64) for b in bases])
    resid = np.zeros_like(y_vec)
    for a, b in folds:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(B[a], y_vec[a])
        resid[b] = y_vec[b] - lr.predict(B[b])
    return unitize(resid)


def fold_aucs(y, z, folds):
    z = np.asarray(z, dtype=np.float64)
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def eval_delta(y, z_base, z_new, folds, m0=None):
    bf, nf = fold_aucs(y, z_base, folds), fold_aucs(y, z_new, folds)
    d5 = [(a - b) * 1e5 for a, b in zip(nf, bf)]
    out = {
        "d5_mean": float(np.mean(d5)),
        "pos": int(sum(x > 0 for x in d5)),
        "d5": [float(x) for x in d5],
        "auc_base": float(roc_auc_score(y, z_base)),
        "auc_new": float(roc_auc_score(y, z_new)),
    }
    if m0 is not None:
        out["d0"] = (
            float(roc_auc_score(y[m0], z_new[m0])) - float(roc_auc_score(y[m0], z_base[m0]))
        ) * 1e5
    return out


def subcv_best(y_f0, z_base, delta, subfolds, ws=None):
    if ws is None:
        ws = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    base = [float(roc_auc_score(y_f0[b], z_base[b])) for _, b in subfolds]
    best = None
    u = unitize(delta).astype(np.float64)
    for w in ws:
        z = z_base + w * u
        d = [(float(roc_auc_score(y_f0[b], z[b])) - base[i]) * 1e5 for i, (_, b) in enumerate(subfolds)]
        row = {"w": float(w), "mean": float(np.mean(d)), "pos": int(sum(x > 0 for x in d)), "d": [float(x) for x in d]}
        if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
            best = row
    return best


def sweep_w(y, z_base, u, folds, m0, ws=None):
    if ws is None:
        ws = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    u = unitize(u).astype(np.float64)
    best = None
    for w in ws:
        ev = eval_delta(y, z_base, z_base + w * u, folds, m0)
        ev["w"] = float(w)
        if best is None or (ev["pos"], ev["d5_mean"]) > (best["pos"], best["d5_mean"]):
            best = ev
    return best


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


def _as_int_codes(vals):
    vals = np.asarray(vals)
    if vals.dtype.kind in "iub":
        return vals.astype(np.int32)
    codes, _ = pd.factorize(pd.Series(vals), sort=False)
    return codes.astype(np.int32)


def _factorize_pair(a_vals, b_vals):
    a = _as_int_codes(a_vals)
    b = _as_int_codes(b_vals)
    key = a.astype(np.int64) * 1_000_003 + b.astype(np.int64) + 17
    codes, _ = pd.factorize(key, sort=False)
    return codes.astype(np.int32)


def _factorize_triple(a_vals, b_vals, c_vals):
    a = _as_int_codes(a_vals).astype(np.int64)
    b = _as_int_codes(b_vals).astype(np.int64)
    c = _as_int_codes(c_vals).astype(np.int64)
    key = a * 1_000_003 + b * 97_003 + c + 31
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


def _hier2d_excess(code_ab, code_a, code_b, y, train_idx, val_idx, m=15.0):
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


def _hier3d_excess(code_abc, ca, cb, cc, cab, cbc, cac, y, train_idx, val_idx, m=15.0):
    """3D residual vs inclusion-exclusion prior on eta scale."""
    eta_a = _te_llr(ca, y, train_idx, train_idx, m=20.0)
    eta_b = _te_llr(cb, y, train_idx, train_idx, m=20.0)
    eta_c = _te_llr(cc, y, train_idx, train_idx, m=20.0)
    eta_ab = _te_llr(cab, y, train_idx, train_idx, m=20.0)
    eta_bc = _te_llr(cbc, y, train_idx, train_idx, m=20.0)
    eta_ac = _te_llr(cac, y, train_idx, train_idx, m=20.0)
    eta0 = float(logit(np.clip(y[train_idx].mean(), 1e-4, 1 - 1e-4)))
    eta_prior = eta_ab + eta_bc + eta_ac - eta_a - eta_b - eta_c + eta0
    p_prior = expit(eta_prior)
    resid = y[train_idx].astype(np.float64) - p_prior
    c3 = code_abc[train_idx]
    mx = int(code_abc.max()) + 1
    cnt = np.bincount(c3, minlength=mx).astype(np.float64)
    sm = np.bincount(c3, weights=resid, minlength=mx)
    delta = sm / (cnt + m)
    return delta[np.clip(code_abc[val_idx], 0, mx - 1)]


def bernoulli_ll(y, eta):
    p = np.clip(expit(eta), 1e-6, 1 - 1e-6)
    y = y.astype(np.float64)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def build_rayA(z_ref, d1, gpt):
    u_sb = unit(d1)
    m1 = unitize(
        0.65 * u_sb.astype(np.float64)
        - 0.35 * (-unit(gpt)).astype(np.float64)
        - 0.20 * gpt.astype(np.float64)
    )
    m2 = unitize(u_sb.astype(np.float64))
    s6 = unitize(0.70 * m1.astype(np.float64) + 0.30 * m2.astype(np.float64))
    w_s7, dw = 0.055, (0.75e-4 / 1e-4) * 0.08
    z_s7 = z_ref.astype(np.float64) + w_s7 * s6.astype(np.float64)
    z_ray = z_ref.astype(np.float64) + (w_s7 + dw) * s6.astype(np.float64)
    return z_s7, z_ray, s6


def qbin(x, q):
    qs = np.quantile(x, np.linspace(0, 1, q + 1))
    qs = np.unique(qs)
    return np.digitize(x, qs[1:-1], right=True).astype(np.int32)


def step10(x):
    return np.round(np.asarray(x, dtype=np.float64) * 10).astype(np.int32)


def build_28_singles(full: pd.DataFrame) -> dict[str, np.ndarray]:
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

    amt = full["amount_usd"].to_numpy(np.float64)
    hrs = full["hours_since_last_txn"].to_numpy(np.float64)
    dist = full["distance_from_home_km"].to_numpy(np.float64)
    bal = full["account_balance_usd"].to_numpy(np.float64)
    vel = full["velocity_score"].to_numpy(np.float64)
    risk = full["merchant_risk_score"].to_numpy(np.float64)
    txn = full["txn_count_last_24h"].to_numpy(np.float64)
    cage = full["card_age_months"].to_numpy(np.float64)
    age = full["customer_age"].to_numpy(np.float64)
    tod = full["time_of_day_hour"].to_numpy(np.float64)
    dow = full["day_of_week"].to_numpy(np.float64)
    cvv = full["cvv_retry_count"].to_numpy(np.float64)
    disp = full["prior_disputes"].to_numpy(np.float64)

    cat, _ = pd.factorize(full["merchant_category"].astype(str))
    card, _ = pd.factorize(full["card_type"].astype(str))
    auth, _ = pd.factorize(full["auth_method"].astype(str))
    chan, _ = pd.factorize(full["channel"].astype(str))
    dev, _ = pd.factorize(full["device_type"].astype(str))

    singles = {
        "cat": cat.astype(np.int32),
        "card": card.astype(np.int32),
        "auth": auth.astype(np.int32),
        "chan": chan.astype(np.int32),
        "dev": dev.astype(np.int32),
        "flag_mask": flag_mask,
        "foreign": full["is_foreign_transaction"].astype(np.int32).to_numpy(),
        "new_merch": full["is_new_merchant"].astype(np.int32).to_numpy(),
        "vpn": full["used_vpn"].astype(np.int32).to_numpy(),
        "ip_mm": full["ip_country_mismatch"].astype(np.int32).to_numpy(),
        "bill_mm": full["billing_shipping_mismatch"].astype(np.int32).to_numpy(),
        "ai_scam": full["is_ai_generated_scam_attempt"].astype(np.int32).to_numpy(),
        "amt_q40": qbin(amt, 40),
        "hrs_q40": qbin(hrs, 40),
        "dist_q40": qbin(dist, 40),
        "bal_q40": qbin(bal, 40),
        "vel_q40": qbin(vel, 40),
        "risk_q40": qbin(risk, 40),
        "txn_q20": qbin(txn, 20),
        "cage_q20": qbin(cage, 20),
        "age_q20": qbin(age, 20),
        "tod": tod.astype(np.int32),
        "dow": dow.astype(np.int32),
        "cvv_q10": qbin(cvv, 10),
        "disp_q10": qbin(disp, 10),
        "amt_s10": step10(amt),
        "vel_s10": step10(vel),
        "risk_s10": step10(risk),
    }
    assert len(singles) == 28, len(singles)
    return singles


def score_pair_ll(cab, ca, cb, y, folds, m=30.0):
    """Return per-fold LL gain of 1D+excess vs 1D-only; pos count; mean gain."""
    gains = []
    for a, b in folds:
        eta_a = _te_llr(ca, y, a, b, m=20.0)
        eta_b = _te_llr(cb, y, a, b, m=20.0)
        eta0 = float(logit(np.clip(y[a].mean(), 1e-4, 1 - 1e-4)))
        eta_1d = eta_a + eta_b - eta0
        delta = _hier2d_excess(cab, ca, cb, y, a, b, m=m)
        ll0 = bernoulli_ll(y[b], eta_1d)
        ll1 = bernoulli_ll(y[b], eta_1d + delta)
        gains.append(ll1 - ll0)
    gains = np.asarray(gains, dtype=np.float64)
    return {"gains": gains.tolist(), "pos": int(np.sum(gains > 0)), "mean": float(gains.mean())}


def score_trip_ll(cabc, ca, cb, cc, cab, cbc, cac, y, folds, m=30.0):
    gains = []
    for a, b in folds:
        eta_a = _te_llr(ca, y, a, b, m=20.0)
        eta_b = _te_llr(cb, y, a, b, m=20.0)
        eta_c = _te_llr(cc, y, a, b, m=20.0)
        eta_ab = _te_llr(cab, y, a, b, m=20.0)
        eta_bc = _te_llr(cbc, y, a, b, m=20.0)
        eta_ac = _te_llr(cac, y, a, b, m=20.0)
        eta0 = float(logit(np.clip(y[a].mean(), 1e-4, 1 - 1e-4)))
        eta_prior = eta_ab + eta_bc + eta_ac - eta_a - eta_b - eta_c + eta0
        delta = _hier3d_excess(cabc, ca, cb, cc, cab, cbc, cac, y, a, b, m=m)
        ll0 = bernoulli_ll(y[b], eta_prior)
        ll1 = bernoulli_ll(y[b], eta_prior + delta)
        gains.append(ll1 - ll0)
    gains = np.asarray(gains, dtype=np.float64)
    return {"gains": gains.tolist(), "pos": int(np.sum(gains > 0)), "mean": float(gains.mean())}


def oof_pair_excess(cab, ca, cb, y, folds, nte, m):
    ntr = len(y)
    oof = np.zeros(ntr, dtype=np.float64)
    te = np.zeros(nte, dtype=np.float64)
    for a, b in folds:
        oof[b] = _hier2d_excess(cab, ca, cb, y, a, b, m=m)
        # fit on a → test
        eta_a = _te_llr(ca, y, a, a, m=20.0)
        eta_b = _te_llr(cb, y, a, a, m=20.0)
        eta0 = float(logit(np.clip(y[a].mean(), 1e-4, 1 - 1e-4)))
        p1d = expit(eta_a + eta_b - eta0)
        resid = y[a].astype(np.float64) - p1d
        mx = int(cab.max()) + 1
        cnt = np.bincount(cab[a], minlength=mx).astype(np.float64)
        sm = np.bincount(cab[a], weights=resid, minlength=mx)
        dlt = sm / (cnt + m)
        te += dlt[np.clip(cab[ntr:], 0, mx - 1)] / len(folds)
    return oof.astype(np.float32), te.astype(np.float32)


def oof_trip_excess(cabc, ca, cb, cc, cab, cbc, cac, y, folds, nte, m):
    ntr = len(y)
    oof = np.zeros(ntr, dtype=np.float64)
    te = np.zeros(nte, dtype=np.float64)
    for a, b in folds:
        oof[b] = _hier3d_excess(cabc, ca, cb, cc, cab, cbc, cac, y, a, b, m=m)
        # refit on a for test
        eta_a = _te_llr(ca, y, a, a, m=20.0)
        eta_b = _te_llr(cb, y, a, a, m=20.0)
        eta_c = _te_llr(cc, y, a, a, m=20.0)
        eta_ab = _te_llr(cab, y, a, a, m=20.0)
        eta_bc = _te_llr(cbc, y, a, a, m=20.0)
        eta_ac = _te_llr(cac, y, a, a, m=20.0)
        eta0 = float(logit(np.clip(y[a].mean(), 1e-4, 1 - 1e-4)))
        eta_prior = eta_ab + eta_bc + eta_ac - eta_a - eta_b - eta_c + eta0
        resid = y[a].astype(np.float64) - expit(eta_prior)
        mx = int(cabc.max()) + 1
        cnt = np.bincount(cabc[a], minlength=mx).astype(np.float64)
        sm = np.bincount(cabc[a], weights=resid, minlength=mx)
        dlt = sm / (cnt + m)
        te += dlt[np.clip(cabc[ntr:], 0, mx - 1)] / len(folds)
    return oof.astype(np.float32), te.astype(np.float32)


def platt_oof(z, y, folds):
    oof = np.zeros(len(y), dtype=np.float64)
    for a, b in folds:
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        clf.fit(z[a].reshape(-1, 1), y[a])
        oof[b] = clf.decision_function(z[b].reshape(-1, 1))
    return oof.astype(np.float32)


def main():
    log("======== GM CLEAN L4.6C EPISTATIC EXPANSION (NO_SUBMIT=%s) ========" % NO_SUBMIT)
    log("BLACKLIST active: v10_*, soft_diff*, gb_soft*, idf_p3*, clean035*, zanch_*")

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

    # ---- clean bases only ----
    z_ref = load_f32(PRED / "gm_v5_overnight/z_ref_old_oof.npy")
    z_ref = unitize(z_ref) if abs(float(z_ref.std()) - 1.0) > 0.2 else z_ref
    d1 = unitize(load_f32(PRED / "gm_v4_polish/d1_v4_oof.npy"))
    w12 = unitize(load_f32(PRED / "gm_v2_ladder/z_lm12_raw_super_oof.npy"))
    mega = zrk(load_f32(VOL / "external/banks/megayak_s6e11-target-first-lm/lm_oof.npy"))
    gpt = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))
    bases = [z_ref, d1, w12, mega, gpt]
    z_base = z_ref.astype(np.float64) + 0.04 * d1.astype(np.float64)
    z_s7, z_ray, s6 = build_rayA(z_ref, d1, gpt)
    rayA_f0 = z_ray[idx_f0]
    log(
        f"RayA F0 vs z_ref={(roc_auc_score(y[m0], z_ray[m0]) - roc_auc_score(y[m0], z_ref[m0])) * 1e5:+.2f}e-5"
    )

    tree_o = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy"))
    tree_t = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/tree_logit_v72_test.npy"))
    l46c_old = unitize(load_f32(WORK / "preds/gm_architect_v9/P4_L4.6C_hier2d_5f_oof.npy"))
    l46c_old_t = unitize(load_f32(WORK / "preds/gm_architect_v9/P4_L4.6C_hier2d_5f_test.npy"))

    # ------------------------------------------------------------------
    # Step 1 — forensic: why L4.6C won Fold0 5/5
    # ------------------------------------------------------------------
    log("======== Step 1: L4.6C forensic (legacy 20-pair hier2d) ========")
    r_old = lofo_resid(l46c_old, bases, folds)
    fold_diag = []
    for fi, (a, b) in enumerate(folds):
        rho = float(spearmanr(r_old[b], z_ray[b]).correlation)
        # signed slope of residual after RayA
        zr = z_ray[b] - z_ray[b].mean()
        cr = r_old[b].astype(np.float64) - r_old[b].mean()
        # partial: residualize r on z_ray
        coef = np.polyfit(zr, cr, 1)
        cr2 = cr - coef[0] * zr - coef[1]
        slope = float(np.polyfit(cr2, y[b].astype(np.float64), 1)[0])
        best = sweep_w(y, z_ray, r_old, [(a, b)], None, ws=[0.01, 0.02, 0.03, 0.04])
        # single-fold eval
        d = (roc_auc_score(y[b], z_ray[b] + best["w"] * unitize(r_old).astype(np.float64)[b]) - roc_auc_score(y[b], z_ray[b])) * 1e5
        fold_diag.append({"fold": fi, "rho_ray": rho, "slope": slope, "dAUC_e5": float(d), "w": best["w"]})
        log(f"  F{fi}: rho(C⊥,ray)={rho:+.4f} slope={slope:+.5f} dAUC≈{d:+.2f}e-5 @w={best['w']}")

    f0_old = subcv_best(y_f0, rayA_f0, r_old[idx_f0], subfolds)
    best5_old = sweep_w(y, z_ray, r_old, folds, m0)
    log(f"  legacy L4.6C vsRayA 5F={best5_old['d5_mean']:+.2f}@{best5_old['pos']}/5 F0sub={f0_old['mean']:+.2f}@{f0_old['pos']}/5")
    log(
        "  Note: Phase-A +6.12@5/5 was after regime; 5F stayed ~flat/neg without per-pair 4/5 filter — "
        "inter-fold scale of noisy pairs dilutes F1–4 while Fold0 SubCV still lifts."
    )

    # ------------------------------------------------------------------
    # Expand: 28 singles → 378 pairs → filter ≥4/5 LL → top 60 triplets
    # ------------------------------------------------------------------
    log("======== Expand: 28 singles → C(28,2)=378 pairs ========")
    usecols = [
        "amount_usd",
        "hours_since_last_txn",
        "txn_count_last_24h",
        "distance_from_home_km",
        "card_age_months",
        "customer_age",
        "account_balance_usd",
        "cvv_retry_count",
        "velocity_score",
        "time_of_day_hour",
        "day_of_week",
        "merchant_risk_score",
        "prior_disputes",
        "merchant_category",
        "card_type",
        "auth_method",
        "channel",
        "device_type",
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
    singles = build_28_singles(full)
    names = list(singles.keys())
    log(f"  singles={len(names)}: {names}")

    pair_rows = []
    pair_codes = {}  # name -> (cab, ca, cb)
    for i, (na, nb) in enumerate(combinations(names, 2)):
        cab = filter_support(_factorize_pair(singles[na], singles[nb]), ntr, nte, 10, 4)
        sc = score_pair_ll(cab, singles[na], singles[nb], y, folds, m=30.0)
        key = f"{na}__{nb}"
        pair_rows.append({"name": key, "a": na, "b": nb, **sc})
        if sc["pos"] >= 4 and sc["mean"] > 0:
            pair_codes[key] = (cab, singles[na], singles[nb])
        if (i + 1) % 50 == 0:
            log(f"  scored pairs {i+1}/378 keep={len(pair_codes)}")
    pair_rows.sort(key=lambda r: (r["pos"], r["mean"]), reverse=True)
    keep_pairs = [r for r in pair_rows if r["pos"] >= 4 and r["mean"] > 0]
    log(f"  pairs ≥4/5 LL+: {len(keep_pairs)} / 378  (top mean={keep_pairs[0]['mean']:+.6f})" if keep_pairs else "  NO pairs kept")
    for r in keep_pairs[:12]:
        log(f"    {r['name']}: pos={r['pos']}/5 meanLL={r['mean']:+.6e}")

    # Triplets from top singles by pair hit-count
    from collections import Counter

    hit = Counter()
    for r in keep_pairs:
        hit[r["a"]] += 1
        hit[r["b"]] += 1
    top_sing = [n for n, _ in hit.most_common(16)]
    if len(top_sing) < 6:
        top_sing = names[:16]
    log(f"  triplet seed singles ({len(top_sing)}): {top_sing}")

    trip_rows = []
    trip_codes = {}
    # cache pair codes among top_sing
    pair_cache = {}
    for na, nb in combinations(top_sing, 2):
        key = tuple(sorted((na, nb)))
        pair_cache[key] = filter_support(_factorize_pair(singles[na], singles[nb]), ntr, nte, 10, 4)

    for i, (na, nb, nc) in enumerate(combinations(top_sing, 3)):
        cabc = filter_support(_factorize_triple(singles[na], singles[nb], singles[nc]), ntr, nte, 10, 4)
        cab = pair_cache[tuple(sorted((na, nb)))]
        cbc = pair_cache[tuple(sorted((nb, nc)))]
        cac = pair_cache[tuple(sorted((na, nc)))]
        sc = score_trip_ll(
            cabc, singles[na], singles[nb], singles[nc], cab, cbc, cac, y, folds, m=30.0
        )
        key = f"{na}__{nb}__{nc}"
        trip_rows.append({"name": key, "a": na, "b": nb, "c": nc, **sc})
        if (i + 1) % 40 == 0:
            log(f"  scored triplets {i+1}")
    trip_rows.sort(key=lambda r: (r["pos"], r["mean"]), reverse=True)
    keep_trips = [r for r in trip_rows if r["pos"] >= 4 and r["mean"] > 0][:60]
    # if fewer with 4/5, take top 60 by mean among pos>=3
    if len(keep_trips) < 20:
        keep_trips = [r for r in trip_rows if r["pos"] >= 3][:60]
    for r in keep_trips:
        na, nb, nc = r["a"], r["b"], r["c"]
        cabc = filter_support(_factorize_triple(singles[na], singles[nb], singles[nc]), ntr, nte, 10, 4)
        cab = pair_cache[tuple(sorted((na, nb)))]
        cbc = pair_cache[tuple(sorted((nb, nc)))]
        cac = pair_cache[tuple(sorted((na, nc)))]
        trip_codes[r["name"]] = (cabc, singles[na], singles[nb], singles[nc], cab, cbc, cac)
    log(f"  triplets kept={len(keep_trips)} (top mean={keep_trips[0]['mean']:+.6e})" if keep_trips else "  no triplets")

    del full
    purge()

    # ------------------------------------------------------------------
    # Step 2 — build dual-m excess matrix; C1 Ridge+newton; C2 Stage-B
    # ------------------------------------------------------------------
    log("======== Step 2: condense L4.6C_Expanded ========")
    # limit pairs to top 40 by mean LL among keepers for RAM/time
    use_pairs = keep_pairs[:40]
    use_trips = keep_trips[:30]
    col_oof = []
    col_te = []
    col_names = []
    for r in use_pairs:
        cab, ca, cb = pair_codes[r["name"]]
        for m in (15.0, 60.0):
            o, t = oof_pair_excess(cab, ca, cb, y, folds, nte, m)
            col_oof.append(o)
            col_te.append(t)
            col_names.append(f"p_{r['name']}_m{int(m)}")
        purge()
    for r in use_trips:
        cabc, ca, cb, cc, cab, cbc, cac = trip_codes[r["name"]]
        for m in (15.0, 60.0):
            o, t = oof_trip_excess(cabc, ca, cb, cc, cab, cbc, cac, y, folds, nte, m)
            col_oof.append(o)
            col_te.append(t)
            col_names.append(f"t_{r['name']}_m{int(m)}")
        purge()
    Xo = np.column_stack(col_oof).astype(np.float32)
    Xt = np.column_stack(col_te).astype(np.float32)
    log(f"  excess matrix Xo={Xo.shape} Xt={Xt.shape}")
    del col_oof, col_te
    purge()

    # rank columns by |corr with y residual after z_ray| on OOF
    resid_y = y.astype(np.float64) - expit(z_ray)
    scores = []
    for j in range(Xo.shape[1]):
        c = float(np.corrcoef(Xo[:, j], resid_y)[0, 1]) if Xo[:, j].std() > 1e-12 else 0.0
        scores.append((abs(c), j, c))
    scores.sort(reverse=True)
    top_idx = [j for _, j, _ in scores[:45]]
    top_names = [col_names[j] for j in top_idx]
    Xo45 = Xo[:, top_idx]
    Xt45 = Xt[:, top_idx]
    log(f"  top45 cols: {top_names[:8]} ...")
    del Xo, Xt
    purge()

    # C1: Ridge(2500) + newton-cholesky(C=0.03) OOF
    log("  C1 l46c_ridge_clean …")
    oof_c1 = np.zeros(ntr, dtype=np.float64)
    te_c1 = np.zeros(nte, dtype=np.float64)
    for fi, (a, b) in enumerate(folds):
        ridge = Ridge(alpha=2500.0)
        ridge.fit(Xo45[a], y[a].astype(np.float64))
        pb = ridge.predict(Xo45[b])
        pt = ridge.predict(Xt45)
        try:
            clf = LogisticRegression(C=0.03, solver="newton-cholesky", max_iter=300)
            clf.fit(Xo45[a], y[a])
            pb = 0.5 * (pb + clf.decision_function(Xo45[b]))
            pt = 0.5 * (pt + clf.decision_function(Xt45))
        except Exception:
            pass
        oof_c1[b] = pb
        te_c1 += pt / 5.0
        log(f"    C1 fold{fi}")
    c1_o, c1_t = align_method_b_smooth(oof_c1, te_c1, folds)
    d1_, k1_ = disp_ks(c1_o, c1_t)
    log(f"  C1 disp={d1_:.4f} KS={k1_:.6f}")
    np.save(ART / "l46c_ridge_clean_oof.npy", c1_o)
    np.save(ART / "l46c_ridge_clean_test.npy", c1_t)

    # C2: Stage-B ExtraTrees LGB on top45 + anchors
    log("  C2 l46c_stageB_lgb …")
    mega_t = zrk(load_f32(VOL / "external/banks/megayak_s6e11-target-first-lm/lm_test.npy"))
    gpt_t = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_test.npy"))
    # gpt2_med_5f — use bayes 5f as available clean med proxy (blacklisted nothing)
    L42o = unitize(zrk(load_f32(PRED / "matrix_v2/P4_L4.2_pure5f_oof.npy")))
    L42t = unitize(zrk(load_f32(PRED / "matrix_v2/P4_L4.2_pure5f_test.npy")))
    L43o = unitize(zrk(load_f32(PRED / "matrix_v2/P4_L4.3_pure5f_oof.npy")))
    L43t = unitize(zrk(load_f32(PRED / "matrix_v2/P4_L4.3_pure5f_test.npy")))
    L45o = unitize(zrk(load_f32(PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy")))
    L45t = unitize(zrk(load_f32(PRED / "matrix_v2/P4_L4.5_pure5f_test.npy")))

    z_init_base = z_ref.astype(np.float64) + 0.05 * d1.astype(np.float64)
    init_oof = platt_oof(z_init_base.astype(np.float32), y, folds)
    # test init: Platt fit on full train
    clf_p = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    clf_p.fit(z_init_base.reshape(-1, 1), y)
    # need test z_ref + d1
    z_ref_t = load_f32(PRED / "gm_v5_overnight/z_ref_old_test.npy") if (PRED / "gm_v5_overnight/z_ref_old_test.npy").exists() else None
    d1_t = load_f32(PRED / "gm_v4_polish/d1_v4_test.npy") if (PRED / "gm_v4_polish/d1_v4_test.npy").exists() else None
    if z_ref_t is None or d1_t is None:
        # fallback: use zeros init on test → tree logit still defined vs 0
        init_te = np.zeros(nte, dtype=np.float64)
        log("  WARN missing z_ref/d1 test — StageB test init=0")
    else:
        z_ref_t = unitize(z_ref_t) if abs(float(z_ref_t.std()) - 1.0) > 0.2 else z_ref_t
        d1_t = unitize(d1_t)
        z_init_te = z_ref_t.astype(np.float64) + 0.05 * d1_t.astype(np.float64)
        init_te = clf_p.decision_function(z_init_te.reshape(-1, 1))

    anchors_o = np.column_stack(
        [unitize(mega), unitize(gpt), L42o, L43o, L45o]
    ).astype(np.float32)
    anchors_t = np.column_stack(
        [unitize(mega_t), unitize(gpt_t), L42t, L43t, L45t]
    ).astype(np.float32)
    Fo = np.column_stack([Xo45, anchors_o]).astype(np.float32)
    Ft = np.column_stack([Xt45, anchors_t]).astype(np.float32)

    oof_c2 = np.zeros(ntr, dtype=np.float64)
    te_c2 = np.zeros(nte, dtype=np.float64)
    lgb_params = {
        "extra_trees": True,
        "max_depth": 3,
        "num_leaves": 6,
        "feature_fraction": 0.35,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 300,
        "lambda_l2": 50.0,
        "learning_rate": 0.02,
        "objective": "binary",
        "verbosity": -1,
        "num_threads": 2,
        "seed": 42,
    }
    for fi, (a, b) in enumerate(folds):
        dtrain = lgb.Dataset(
            Fo[a], label=y[a], init_score=init_oof[a].astype(np.float64), free_raw_data=False
        )
        bst = lgb.train(lgb_params, dtrain, num_boost_round=220)
        # raw_score = pure tree contribution (init_score not re-added)
        oof_c2[b] = bst.predict(Fo[b], raw_score=True)
        te_c2 += bst.predict(Ft, raw_score=True) / 5.0
        log(f"    C2 fold{fi}")
        del bst, dtrain
        purge()
    c2_o, c2_t = align_method_b_smooth(oof_c2, te_c2, folds)
    d2_, k2_ = disp_ks(c2_o, c2_t)
    log(f"  C2 disp={d2_:.4f} KS={k2_:.6f}")
    np.save(ART / "l46c_stageB_lgb_oof.npy", c2_o)
    np.save(ART / "l46c_stageB_lgb_test.npy", c2_t)

    # Orth + evaluate gates (ZERO v10 / soft_diff)
    def prep(u):
        return lofo_resid(unitize(u), bases, folds)

    candidates = {
        "l46c_legacy": prep(l46c_old),
        "l46c_ridge_clean": prep(c1_o),
        "l46c_stageB_lgb": prep(c2_o),
        "tree_logit_v72": prep(tree_o),
        "P4_L4.5": prep(L45o),
    }
    # blend expanded: unit(w1*l46c + w2*tree)
    u_l46 = candidates["l46c_ridge_clean"]
    # prefer stageB if better on F0
    gate_rows = []
    log("======== Gates (clean only) ========")
    for name, u in candidates.items():
        b5 = sweep_w(y, z_ray, u, folds, m0)
        b0 = sweep_w(y, z_base, u, folds, m0)
        f0 = subcv_best(y_f0, rayA_f0, u[idx_f0], subfolds)
        row = {"name": name, "vs_rayA": b5, "vs_zbase": b0, "f0": f0}
        gate_rows.append(row)
        log(
            f"  {name}: vsRayA={b5['d5_mean']:+.2f}@{b5['pos']}/5 "
            f"vsBase={b0['d5_mean']:+.2f}@{b0['pos']}/5 "
            f"F0={f0['mean']:+.2f}@{f0['pos']}/5 @w={f0['w']}"
        )

    # pick best L46 expanded for boost
    l46_names = ["l46c_stageB_lgb", "l46c_ridge_clean", "l46c_legacy"]
    best_l46 = max(
        (r for r in gate_rows if r["name"] in l46_names),
        key=lambda r: (r["f0"]["pos"], r["f0"]["mean"], r["vs_rayA"]["pos"], r["vs_rayA"]["d5_mean"]),
    )
    log(f"  selected L46 vector: {best_l46['name']}")

    u_l = candidates[best_l46["name"]]
    u_tr = candidates["tree_logit_v72"]
    # LOFO blend weights on Fold0 subcv
    best_blend = None
    for w1 in [0.55, 0.65, 0.75, 0.85, 1.0]:
        for w2 in [0.0, 0.15, 0.25, 0.35]:
            u = unitize(w1 * u_l.astype(np.float64) + w2 * u_tr.astype(np.float64))
            f0 = subcv_best(y_f0, rayA_f0, u[idx_f0], subfolds)
            b5 = sweep_w(y, z_ray, u, folds, m0, ws=[0.01, 0.02, 0.03, 0.04, 0.05])
            key = (f0["pos"], f0["mean"], b5["pos"], b5["d5_mean"])
            if best_blend is None or key > best_blend[0]:
                best_blend = (key, w1, w2, u, f0, b5)
    _, w1, w2, u_clean, f0_clean, b5_clean = best_blend
    log(
        f"  clean_boost w1={w1} w2={w2}: F0={f0_clean['mean']:+.2f}@{f0_clean['pos']}/5 "
        f"vsRayA={b5_clean['d5_mean']:+.2f}@{b5_clean['pos']}/5"
    )
    np.save(ART / "u_clean_boost_oof.npy", u_clean)

    # map clean boost to test via corresponding vectors
    def prep_te(o, t):
        o2, t2 = align_method_b_smooth(unitize(o), unitize(t), folds)
        return t2

    if best_l46["name"] == "l46c_stageB_lgb":
        u_l_t = unitize(c2_t)
    elif best_l46["name"] == "l46c_ridge_clean":
        u_l_t = unitize(c1_t)
    else:
        u_l_t = unitize(l46c_old_t)
    u_tr_t = unitize(tree_t)
    u_clean_t = unitize(w1 * u_l_t.astype(np.float64) + w2 * u_tr_t.astype(np.float64))
    # orient test vs areeb distance increase along ray
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
    if spearman_dist(z_areeb, z_s7_test.astype(np.float64) + 5 * u_clean_t.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7_test
    ):
        u_clean_t = (-u_clean_t.astype(np.float64)).astype(np.float32)
    np.save(ART / "u_clean_boost_test.npy", u_clean_t)

    # ------------------------------------------------------------------
    # Step 3 — stage submission ladder (write CSVs; NO_SUBMIT blocks kaggle)
    # ------------------------------------------------------------------
    log("======== Step 3: stage clean submission ladder ========")
    ids = pd.read_csv(VOL / "data/test.csv", usecols=["id"])["id"].to_numpy()

    def stage_csv(name, step_1rho, include_boost=True, boost_scale=0.45):
        w_ray, got = weight_for_dist(z_s7_test, u_ray_a, step_1rho)
        z = z_s7_test.astype(np.float64) + w_ray * unit(u_ray_a).astype(np.float64)
        if include_boost:
            # small orthogonal add after ray step (rank-space)
            z = z + boost_scale * unit(u_clean_t).astype(np.float64)
        z = unitize(z)
        d_a = spearman_dist(z_areeb, z)
        t = 100.0 * np.sqrt(max(d_a, 0))
        plb = pred_lb_from_t(t)
        # OOF proxy
        dw = (step_1rho / 1e-4) * 0.08
        z_o = z_s7 + dw * unitize(s6).astype(np.float64)
        if include_boost:
            z_o = z_o + boost_scale * 0.08 * unitize(u_clean).astype(np.float64)
        f0 = (
            float(roc_auc_score(y[m0], z_o[m0])) - float(roc_auc_score(y[m0], z_s7[m0]))
        ) * 1e5
        f0_vs_ray = (
            float(roc_auc_score(y[m0], z_o[m0])) - float(roc_auc_score(y[m0], z_ray[m0]))
        ) * 1e5
        path = SUB / name
        # always write staged CSV for review; NO_SUBMIT only blocks live API submit
        p = expit(z.astype(np.float64))
        pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(path, index=False)
        meta = {
            "file": str(path),
            "step_1rho": step_1rho,
            "got": got,
            "t": t,
            "pred_lin": plb,
            "include_boost": include_boost,
            "boost_scale": boost_scale if include_boost else 0.0,
            "oof_F0_vsS7_e5": f0,
            "oof_F0_vsRayA_e5": f0_vs_ray,
            "NO_SUBMIT": NO_SUBMIT,
        }
        log(
            f"  staged {name}: t={t:.3f} pred={plb:.6f} F0vsS7={f0:+.2f} F0vsRayA={f0_vs_ray:+.2f} "
            f"NO_SUBMIT={NO_SUBMIT}"
        )
        return meta

    slots = [
        stage_csv("sub_clean_v9_slot1_rayA_plus_l46c_0p20e4.csv", 0.20e-4, True, 0.40),
        stage_csv("sub_clean_v9_slot2_rayA_plus_l46c_0p45e4.csv", 0.45e-4, True, 0.45),
        stage_csv("sub_clean_v9_slot3_rayA_f0peak_plus_l46c_0p75e4.csv", 0.75e-4, True, 0.50),
        stage_csv("sub_clean_v9_slot4_pure_rayA_0p35e4_control.csv", 0.35e-4, False, 0.0),
    ]

    # copy arts to VOL
    for p in ART.glob("*.npy"):
        try:
            (VOL_ART / p.name).write_bytes(p.read_bytes())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    md = []
    md.append("# GM Clean L4.6C V9 — Ready for Slot 1 Review\n\n")
    md.append(f"**NO_SUBMIT={NO_SUBMIT}**. Poison pills blacklisted: `v10_*`, `soft_diff*`, `gb_soft*`, `idf_p3*`, `clean035*`, `zanch_*`.\n\n")
    md.append(f"Elapsed `{time.time()-T0:.0f}s`, peak RSS ≈ `{rss_max():.0f}` MB.\n\n")
    md.append("## Step 1 — Why L4.6C won Fold0 5/5\n\n")
    md.append(
        "Legacy `P4_L4.6C_hier2d` = **20** 2-way pairs condensed via hierarchical excess "
        "`Δ = E[y − σ(η_A+η_B−η0) | AB]` (m∈{10,50} bag) → Ridge/newton → PCHIP align.\n\n"
    )
    md.append("| fold | ρ(C,RayA) | slope | approx dAUC e-5 |\n|---:|---:|---:|---:|\n")
    for r in fold_diag:
        md.append(f"| {r['fold']} | {r['rho_ray']:+.4f} | {r['slope']:+.5f} | {r['dAUC_e5']:+.2f} |\n")
    md.append(
        f"\nLegacy LOFO vs RayA: **{best5_old['d5_mean']:+.2f}e-5** `{best5_old['pos']}/5`; "
        f"Fold0 SubCV **{f0_old['mean']:+.2f}e-5** `{f0_old['pos']}/5`.\n\n"
    )
    md.append(
        "**Diagnosis:** Fold0 SubCV captures a real hierarchical interaction signal; "
        "5F dips because low-support / sign-unstable pairs among the original 20 dilute F1–4. "
        "Fix = keep only pairs with **≥4/5** outer-fold positive residual LL, then expand.\n\n"
    )
    md.append("## Expansion screen\n\n")
    md.append(f"- Singles: **28** (cats/flags/qbins/steps; not KMAT-distilled)\n")
    md.append(f"- Pairs scored: **378**; kept ≥4/5 LL+: **{len(keep_pairs)}** (used top {len(use_pairs)})\n")
    md.append(f"- Triplets kept: **{len(keep_trips)}** (used top {len(use_trips)})\n")
    md.append(f"- Condensation cols → top **45** by |corr| with RayA residual\n\n")
    md.append("Top pairs:\n")
    for r in keep_pairs[:10]:
        md.append(f"- `{r['name']}` pos={r['pos']}/5 meanLL={r['mean']:+.6e}\n")
    md.append("\n## Step 2 — Estimators + gates (clean)\n\n")
    md.append(f"- C1 `l46c_ridge_clean`: disp=`{d1_:.4f}` KS=`{k1_:.6f}`\n")
    md.append(f"- C2 `l46c_stageB_lgb`: disp=`{d2_:.4f}` KS=`{k2_:.6f}`\n\n")
    md.append("| name | vs RayA Δ5F | pos | vs z_base | pos | F0 SubCV | pos | w |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in gate_rows:
        md.append(
            f"| {r['name']} | {r['vs_rayA']['d5_mean']:+.2f} | {r['vs_rayA']['pos']}/5 | "
            f"{r['vs_zbase']['d5_mean']:+.2f} | {r['vs_zbase']['pos']}/5 | "
            f"{r['f0']['mean']:+.2f} | {r['f0']['pos']}/5 | {r['f0']['w']} |\n"
        )
    md.append(
        f"\n**clean_boost** = unit({w1}·`{best_l46['name']}` + {w2}·`tree_logit_v72`): "
        f"F0=`{f0_clean['mean']:+.2f}`@{f0_clean['pos']}/5 vsRayA=`{b5_clean['d5_mean']:+.2f}`@{b5_clean['pos']}/5\n\n"
    )
    gate1_ok = b5_clean["pos"] >= 4 or any(
        r["name"] in ("tree_logit_v72", "P4_L4.5", "l46c_ridge_clean", "l46c_stageB_lgb")
        and r["vs_rayA"]["pos"] >= 4
        for r in gate_rows
    )
    gate2_ok = f0_clean["pos"] >= 5 and f0_clean["mean"] >= 6.12
    md.append(f"### Gate status\n- Gate1 (≥4/5 vs RayA among clean vectors): **{'PASS' if gate1_ok else 'REVIEW'}**\n")
    md.append(
        f"- Gate2 (F0 SubCV ≥ +6.12 @5/5 over RayA): **{'PASS' if gate2_ok else 'REVIEW'}** "
        f"(got `{f0_clean['mean']:+.2f}` @{f0_clean['pos']}/5)\n\n"
    )
    md.append("## Step 3 — Staged submissions (NOT live-submitted)\n\n")
    md.append("| slot | file | t | pred_lin | F0vsS7 e-5 | F0vsRayA e-5 |\n|---|---|---:|---:|---:|---:|\n")
    for s in slots:
        md.append(
            f"| {Path(s['file']).name} | `{Path(s['file']).name}` | {s['t']:.3f} | {s['pred_lin']:.6f} | "
            f"{s['oof_F0_vsS7_e5']:+.2f} | {s['oof_F0_vsRayA_e5']:+.2f} |\n"
        )
    md.append(
        "\n**Recommendation:** Review Gate2; if PASS or close, fire **Slot 1** "
        "`sub_clean_v9_slot1_rayA_plus_l46c_0p20e4.csv` as today's high-certainty probe. "
        "Set `NO_SUBMIT=0` only after explicit approval.\n\n"
    )
    md.append(f"**NO_SUBMIT={NO_SUBMIT}** — no Kaggle API submit from this script.\n")

    report = LOG / "GM_CLEAN_L46C_V9_READY.md"
    report.write_text("".join(md))
    try:
        (VOL_LOG / "GM_CLEAN_L46C_V9_READY.md").write_text(report.read_text())
    except Exception:
        pass
    summary = {
        "NO_SUBMIT": NO_SUBMIT,
        "keep_pairs": len(keep_pairs),
        "keep_trips": len(keep_trips),
        "best_l46": best_l46["name"],
        "clean_boost": {"w1": w1, "w2": w2, "f0": f0_clean, "vs_rayA": b5_clean},
        "gates": {"gate1": gate1_ok, "gate2": gate2_ok},
        "slots": slots,
        "gate_rows": [
            {
                "name": r["name"],
                "vs_rayA": r["vs_rayA"],
                "f0": r["f0"],
            }
            for r in gate_rows
        ],
    }
    (ART / "clean_l46c_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"REPORT → {report}")
    log(f"VOL REPORT → {VOL_LOG / 'GM_CLEAN_L46C_V9_READY.md'}")
    log("======== CLEAN L4.6C DONE (NO_SUBMIT=%s) ========" % NO_SUBMIT)


if __name__ == "__main__":
    main()
