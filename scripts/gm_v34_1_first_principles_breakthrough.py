#!/usr/bin/env python3
"""GM V34.1 — First-principles breakthrough engine.
STRICT NO_SUBMIT=True · 0 GPU pods · 0 Kaggle CLI · Mac CPU <3.5m · RSS <2.8GB · n_jobs=4
"""
from __future__ import annotations

import gc
import json
import os
import resource
import threading
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

NO_SUBMIT = True
N_JOBS = 4
RSS_CAP_GB = 2.8
T0 = time.time()
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "preds" / "v34_1_first_principles"
REPORT = REPO / "logs" / "overnight" / "GM_V34_1_FIRST_PRINCIPLES_BREAKTHROUGH_REPORT.md"
SUB_DIR = REPO / "submissions"

FEAT_COLS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
    "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
FLOAT_COLS = [
    "amount_usd", "account_balance_usd", "velocity_score", "merchant_risk_score",
    "distance_from_home_km", "hours_since_last_txn",
]
BOOL_COLS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
CAT_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]

# RSS / CPU monitor
_mon = {"peak_rss_gb": 0.0, "samples": [], "stop": False}


def rss_gb() -> float:
    # macOS: ru_maxrss is bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def cur_rss_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except Exception:
        # fallback: parse vmmap-ish via resource current is max only
        return rss_gb()


def _monitor_loop():
    while not _mon["stop"]:
        try:
            r = cur_rss_gb()
            _mon["peak_rss_gb"] = max(_mon["peak_rss_gb"], r)
            _mon["samples"].append((time.time() - T0, r))
            if r > RSS_CAP_GB:
                log(f"!! RSS {r:.2f}GB > {RSS_CAP_GB}GB cap")
        except Exception:
            pass
        time.sleep(2.0)


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s|rss={cur_rss_gb():.2f}G] {m}", flush=True)


def L(x: np.ndarray) -> np.ndarray:
    r = pd.Series(np.asarray(x, float).ravel()).rank(method="average").to_numpy()
    u = (r - 0.5) / max(len(r), 1)
    return logit(np.clip(u, 1e-6, 1 - 1e-6)).astype(np.float64)


def L_fold(x: np.ndarray, fold: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x), np.float64)
    for f in np.unique(fold):
        m = fold == f
        out[m] = L(x[m])
    return out


def unitize(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float).ravel()
    s = z.std()
    return (z - z.mean()) / (s if s > 1e-12 else 1.0)


def auc(y, s) -> float:
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    n1 = float(y.sum())
    n0 = len(y) - n1
    if n1 == 0.0 or n0 == 0.0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    n = len(s)
    while i < n:
        j = i + 1
        while j < n and s[order[j]] == s[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pchip_align(oof: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Quantile-align test onto OOF empirical distribution via PCHIP."""
    o = np.sort(np.asarray(oof, float).ravel())
    t = np.asarray(test, float).ravel()
    # empirical CDF of test
    rt = (pd.Series(t).rank(method="average").to_numpy() - 0.5) / len(t)
    q = np.linspace(0.5 / len(o), 1.0 - 0.5 / len(o), len(o))
    # subsample for speed if huge
    if len(o) > 50000:
        idx = np.linspace(0, len(o) - 1, 50000).astype(int)
        o, q = o[idx], q[idx]
    f = PchipInterpolator(q, o, extrapolate=True)
    return f(np.clip(rt, q[0], q[-1])).astype(np.float64)


def ks_stat(a, b) -> float:
    try:
        return float(stats.ks_2samp(a, b).statistic)
    except Exception:
        return 1.0


def raw_disp(oof, test) -> float:
    return float(np.std(test) / (np.std(oof) + 1e-12))


def platt(z_tr, y_tr, z_va):
    lr = LogisticRegression(C=1e6, max_iter=400, solver="lbfgs")
    lr.fit(z_tr.reshape(-1, 1), y_tr.astype(int))
    return lr.predict_proba(z_va.reshape(-1, 1))[:, 1], lr


def orth_resid(cand, base):
    """r = cand - beta * base, beta from OLS."""
    b = np.asarray(base, float).ravel()
    c = np.asarray(cand, float).ravel()
    beta = float(np.dot(c, b) / (np.dot(b, b) + 1e-12))
    return c - beta * b, beta


def _strat_sub(y, idx, n_max, seed):
    if len(idx) <= n_max:
        return idx
    rng = np.random.default_rng(seed)
    ypos, yneg = idx[y[idx] == 1], idx[y[idx] == 0]
    n_pos = min(len(ypos), max(1, int(round(n_max * len(ypos) / len(idx)))))
    n_neg = min(len(yneg), n_max - n_pos)
    return np.concatenate([rng.choice(ypos, n_pos, replace=False), rng.choice(yneg, n_neg, replace=False)])


def gate_25(y, base, cand, fold, name: str, test_base=None, test_cand=None) -> dict:
    """25-subfold gate + optional pre-unitize raw_disp / KS on test."""
    lifts, sub_pos, outer_pos = [], 0, 0
    ws = np.linspace(0.0, 0.35, 8)
    for f in range(5):
        va, tr = fold == f, fold != f
        idx_s = _strat_sub(y, np.where(tr)[0], 35000, 42 + f)
        y_s, b_s, c_s = y[idx_s], base[idx_s], cand[idx_s]
        base_a = auc(y_s, b_s)
        best_w, best = 0.0, -1e9
        for w in ws:
            d = auc(y_s, (1 - w) * b_s + w * c_s) - base_a
            if d > best:
                best, best_w = d, float(w)
        yva, bva, cva = y[va], base[va], cand[va]
        blend = (1 - best_w) * bva + best_w * cva
        d_va = auc(yva, blend) - auc(yva, bva)
        lifts.append(d_va)
        outer_pos += int(d_va > 0)
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for _, te in skf.split(np.zeros(len(yva)), yva.astype(int)):
            sub_pos += int(auc(yva[te], blend[te]) - auc(yva[te], bva[te]) > 0)
    lifts = np.asarray(lifts, float)
    t_stat = float(lifts.mean() / (lifts.std(ddof=1) / np.sqrt(5) + 1e-12)) if lifts.std() > 1e-15 else 0.0
    try:
        w_p = float(stats.wilcoxon(lifts, alternative="greater").pvalue)
    except Exception:
        w_p = 1.0
    # pre-unitize raw_disp / KS if test provided
    if test_base is not None and test_cand is not None:
        r_oof, _ = orth_resid(cand, base)
        r_te, _ = orth_resid(test_cand, test_base)
        disp = raw_disp(r_oof, r_te)
        ks = ks_stat(r_oof, r_te)
    else:
        disp = float(np.mean([np.std(cand[fold == f]) / (np.std(base[fold == f]) + 1e-12) for f in range(5)]))
        ks = 0.0
    passed = (
        outer_pos == 5 and sub_pos >= 18 and t_stat >= 2.80 and w_p < 0.005
        and 0.92 <= disp <= 1.08 and ks < 0.015
    )
    return {
        "name": name, "outer_pos": f"{outer_pos}/5", "sub_pos": f"{sub_pos}/25",
        "mean_lift_e4": float(lifts.mean() * 1e4), "t_stat": t_stat, "wilcoxon_p": w_p,
        "raw_disp": disp, "KS": ks, "pass": bool(passed),
        "lifts_e4": [float(x * 1e4) for x in lifts],
        "solo_auc": float(auc(y, cand)),
    }


# ─── DATA ───────────────────────────────────────────────────────────
def load_core():
    log("load core tensors")
    fold = pd.read_csv(REPO / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv")["fold"].to_numpy().astype(np.int8)
    y = pd.read_csv(REPO / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.float32)
    wide_oof = np.load(REPO / "external/nov08_part12/wide_lm_stack_nested.npy").ravel()[: len(y)].astype(np.float64)
    wide_te = np.load(REPO / "external/nov08_part12/wide_lm_stack_test.npy").ravel().astype(np.float64)
    # FREEZE base anchor once
    z_wide_oof = unitize(L_fold(wide_oof, fold))
    z_wide_test = unitize(L(wide_te))
    log(f"z_wide_oof AUC={auc(y, z_wide_oof):.6f} (target≈0.952565)")
    return dict(fold=fold, y=y, wide_oof=wide_oof, wide_te=wide_te,
                z_wide_oof=z_wide_oof, z_wide_test=z_wide_test)


def audit_delta_cat(core):
    log("audit delta_cat")
    y, fold, zw = core["y"], core["fold"], core["z_wide_oof"]
    op = REPO / "preds/macro_v11/delta_cat_oof.npy"
    tp = REPO / "preds/macro_v11/delta_cat_test.npy"
    if not op.exists():
        return {"exists": False, "keep": False}
    oof = np.load(op).ravel()[: len(y)].astype(np.float64)
    te = np.load(tp).ravel().astype(np.float64) if tp.exists() else None
    Ld = L_fold(oof, fold)
    r, beta = orth_resid(Ld, zw)
    # quick 5-fold Δ at fixed w=0.1 (audit only)
    lifts = []
    for f in range(5):
        va = fold == f
        lifts.append(auc(y[va], 0.9 * zw[va] + 0.1 * Ld[va]) - auc(y[va], zw[va]))
    disp = raw_disp(r, orth_resid(L(te), core["z_wide_test"])[0]) if te is not None else float("nan")
    r2 = float(np.corrcoef(Ld, zw)[0, 1] ** 2)
    keep = bool(0.92 <= disp <= 1.08) and r2 < 0.98  # drop pre-blended wrapper
    out = {
        "exists": True, "r2_vs_z_wide": r2, "beta": beta,
        "nested_delta_e4": float(np.mean(lifts) * 1e4),
        "raw_disp": disp, "keep": keep,
        "reason": "keep" if keep else ("drop_preblend_r2" if r2 >= 0.98 else "drop_disp"),
    }
    log(f"  delta_cat R²={r2:.4f} raw_disp={disp:.4f} Δ={out['nested_delta_e4']:.2f}e-4 keep={keep}")
    return out


def build_z_anchor(core, delta_keep: bool):
    log("build z_anchor (nested LOFO)")
    y, fold, zw = core["y"], core["fold"], core["z_wide_oof"]
    n, nt = len(y), len(core["z_wide_test"])
    candidates = [
        ("v10_ridge_orth", "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy",
         "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"),
        ("P4_L4.2", "preds/matrix_v2/P4_L4.2_pure5f_oof.npy", "preds/matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("P4_L4.3", "preds/matrix_v2/P4_L4.3_pure5f_oof.npy", "preds/matrix_v2/P4_L4.3_pure5f_test.npy"),
        ("P4_L4.3_ext", "preds/matrix_v2/P4_L4.3_ext_oof.npy", "preds/matrix_v2/P4_L4.3_ext_test.npy"),
        ("P4_L4.5", "preds/matrix_v2/P4_L4.5_pure5f_oof.npy", "preds/matrix_v2/P4_L4.5_pure5f_test.npy"),
        ("B_xgb_llr", "preds/paradigm_B/xgb_llr_oof.npy", "preds/paradigm_B/xgb_llr_test.npy"),
        ("gpt2_3ep", "preds/gpt2_3ep/gpt2_3ep_oof.npy", "preds/gpt2_3ep/gpt2_3ep_test.npy"),
    ]
    # optional missing
    for name, op, tp in [
        ("F_cat_obliv", "preds/F_cat_obliv_oof.npy", "preds/F_cat_obliv_test.npy"),
        ("R2_v178", "preds/R2_v178_oof.npy", "preds/R2_v178_test.npy"),
    ]:
        if (REPO / op).exists():
            candidates.append((name, op, tp))

    prop_oof, prop_te, meta = [], [], []
    # assemble gpt2_med mean LLR from per-fold if present
    gm_dir = REPO / "preds/gpt2_med_4ep"
    if (gm_dir / "oof_0.npy").exists():
        g_oof = np.zeros(n)
        for f in range(5):
            a = np.load(gm_dir / f"oof_{f}.npy")
            va = fold == f
            llr = a[:, :4, 1] - a[:, :4, 0]
            if llr.shape[0] == va.sum():
                g_oof[va] = llr.mean(1)
        g_te = np.mean([
            (np.load(gm_dir / f"test_{f}.npy")[:, :4, 1] - np.load(gm_dir / f"test_{f}.npy")[:, :4, 0]).mean(1)
            for f in range(5)
        ], 0)
        candidates.append(("gpt2_med_4ep", None, None))
        # inject via special-case below

    for name, op, tp in candidates:
        if name == "gpt2_med_4ep" and op is None:
            o, t = g_oof, g_te
            tpp_exists = True
        else:
            opp, tpp = REPO / op, REPO / tp
            if not opp.exists():
                meta.append({"name": name, "exists": False})
                continue
            o = np.load(opp).ravel().astype(np.float64)
            if len(o) != n:
                meta.append({"name": name, "exists": True, "skip": f"len={len(o)}!={n}"})
                log(f"  prop {name}: SKIP len={len(o)}")
                continue
            tpp_exists = tpp.exists()
            t = np.load(tpp).ravel()[:nt].astype(np.float64) if tpp_exists else None
        Lo = L_fold(o, fold)
        if t is not None and len(t) == nt:
            t_al = pchip_align(o, t)
            ks = ks_stat(o, t)
            disp = raw_disp(Lo, L(t_al))
        else:
            t_al = np.zeros(nt)
            ks, disp = 1.0, float("nan")
            tpp_exists = False
        r2w = float(np.corrcoef(Lo, zw)[0, 1] ** 2)
        meta.append({"name": name, "exists": True, "KS_pre": ks, "disp": disp, "r2_wide": r2w})
        # keep near-dupes out of anchor design (still logged); allow mild R²
        if r2w < 0.85:
            prop_oof.append(Lo)
            prop_te.append(L(t_al) if tpp_exists else np.zeros(nt))
            used = True
        else:
            used = False
        log(f"  prop {name}: KS_pre={ks:.4f} disp={disp:.3f} R²w={r2w:.4f} used={used}")

    if not prop_oof:
        z_a = zw.copy()
        z_at = core["z_wide_test"].copy()
        return z_a, z_at, meta

    Xo = np.column_stack([zw] + prop_oof)
    Xt = np.column_stack([core["z_wide_test"]] + prop_te)
    z_a = np.zeros(n)
    z_at_folds = []
    for f in range(5):
        tr, va = fold != f, fold == f
        # EB-style logistic (negative weights via coefficients OK)
        lr = LogisticRegression(C=0.3, max_iter=250, solver="lbfgs")
        lr.fit(Xo[tr], y[tr].astype(int))
        z_a[va] = logit(np.clip(lr.predict_proba(Xo[va])[:, 1], 1e-6, 1 - 1e-6))
        z_at_folds.append(logit(np.clip(lr.predict_proba(Xt)[:, 1], 1e-6, 1 - 1e-6)))
    z_at = np.mean(z_at_folds, axis=0)
    z_a = unitize(z_a)
    z_at = unitize(z_at)
    log(f"  z_anchor AUC={auc(y, z_a):.6f} Δwide={(auc(y, z_a)-auc(y, zw))*1e4:.2f}e-4 n_prop={len(prop_oof)}")
    return z_a, z_at, meta


# ─── E1 ─────────────────────────────────────────────────────────────
def engine1(core, z_anchor):
    log("E1 clone / id forensics")
    y, fold = core["y"], core["fold"]
    n_tr = len(y)
    # load strings — only FEAT_COLS as str, chunked
    usecols = ["id"] + FEAT_COLS + ["is_fraud"]
    tr = pd.read_csv(REPO / "data/train.csv", usecols=usecols)
    te = pd.read_csv(REPO / "data/test.csv", usecols=["id"] + FEAT_COLS)
    orig = pd.read_csv(REPO / "original/credit_card_fraud_2026.csv")
    # align orig columns
    for c in FEAT_COLS:
        if c not in orig.columns:
            raise RuntimeError(f"orig missing {c}")
    y_orig = orig["is_fraud"].to_numpy(np.float32)

    # exact string matrix for all rows
    parts = []
    for df in (tr, te, orig):
        parts.append(df[FEAT_COLS].astype(str))
    S = pd.concat(parts, ignore_index=True)
    N = len(S)
    n_te = len(te)
    n_or = len(orig)
    assert N == n_tr + n_te + n_or
    log(f"  universe N={N} (tr={n_tr} te={n_te} orig={n_or})")

    # factorize each col → int32 codes + counts
    codes = np.empty((N, len(FEAT_COLS)), dtype=np.int32)
    col_counts = []
    for j, c in enumerate(FEAT_COLS):
        codes[:, j], uniques = pd.factorize(S[c].to_numpy(), sort=False)
        cnt = np.bincount(codes[:, j])
        col_counts.append(cnt)

    # Tier A: rare exact-value posting lists (2..200)
    # accumulate top neighbor score per row (train+test only for features; labels from tr+orig)
    top_score = np.zeros(N, np.float32)
    top_nbr = np.full(N, -1, np.int32)
    # also keep second best for clone features
    top2_score = np.zeros(N, np.float32)
    top2_nbr = np.full(N, -1, np.int32)
    pair_touches = 0
    for j in range(len(FEAT_COLS)):
        cnt = col_counts[j]
        rare_vals = np.where((cnt >= 2) & (cnt <= 200))[0]
        for v in rare_vals:
            idx = np.where(codes[:, j] == v)[0]
            k = len(idx)
            idf = np.log(N / float(k))
            # all pairs
            for a in range(k):
                ia = idx[a]
                for b in range(a + 1, k):
                    ib = idx[b]
                    pair_touches += 1
                    # update both directions
                    for i, oth, sc in ((ia, ib, idf), (ib, ia, idf)):
                        # accumulate into best
                        if sc > top_score[i]:
                            top2_score[i], top2_nbr[i] = top_score[i], top_nbr[i]
                            top_score[i], top_nbr[i] = sc, oth
                        elif sc > top2_score[i]:
                            top2_score[i], top2_nbr[i] = sc, oth
            if pair_touches > 8_000_000:
                break
        if pair_touches > 8_000_000:
            break
    log(f"  TierA pair_touches={pair_touches}")

    # Tier B: 16 deterministic 8-col subspace hashes
    rng = np.random.default_rng(2026)
    subsets = []
    # fixed covering subsets
    base_sets = [
        [0, 1, 2, 7, 8, 15, 16, 17],   # floats+auth+ch+dev
        [13, 14, 15, 16, 17, 18, 19, 20],  # cats+bools
        [3, 4, 5, 6, 9, 10, 11, 12],
        [7, 8, 15, 18, 19, 20, 21, 22],
        [0, 6, 8, 11, 15, 16, 17, 23],
        [1, 2, 7, 13, 14, 15, 16, 17],
        [0, 3, 8, 11, 18, 19, 20, 21],
        [4, 5, 9, 10, 12, 14, 15, 22],
        [0, 1, 7, 8, 13, 15, 16, 23],
        [2, 6, 11, 14, 17, 18, 19, 20],
        [0, 5, 8, 12, 15, 16, 21, 22],
        [1, 3, 7, 9, 13, 17, 18, 23],
        [2, 4, 6, 10, 14, 15, 19, 20],
        [0, 8, 11, 13, 16, 17, 22, 23],
        [1, 2, 5, 7, 15, 18, 20, 21],
        [3, 6, 8, 12, 14, 16, 19, 23],
    ]
    for s in base_sets:
        subsets.append([i for i in s if i < len(FEAT_COLS)])

    # match counts via hash collisions then verify
    match_ge = {12: 0, 15: 0, 18: 0}
    # store best verified match count partner for each train row (within universe)
    best_match = np.zeros(n_tr, np.int16)
    best_match_j = np.full(n_tr, -1, np.int32)

    for si, cols in enumerate(subsets):
        # hash
        h = np.zeros(N, np.int64)
        for c in cols:
            h = h * 1000003 + (codes[:, c].astype(np.int64) + 1)
        # group
        order = np.argsort(h, kind="mergesort")
        i = 0
        while i < N:
            j = i + 1
            while j < N and h[order[j]] == h[order[i]]:
                j += 1
            if j - i >= 2 and j - i <= 500:
                grp = order[i:j]
                # verify exact col matches for pairs (cap)
                # sample pairs inside collision group (speed)
                g = grp
                lim = min(len(g), 24)
                for a in range(lim):
                    ia = int(g[a])
                    for b in range(a + 1, lim):
                        ib = int(g[b])
                        m = int((codes[ia] == codes[ib]).sum())
                        for thr in (12, 15, 18):
                            if m >= thr:
                                match_ge[thr] += 1
                        for src, oth in ((ia, ib), (ib, ia)):
                            if src < n_tr and m > best_match[src]:
                                best_match[src] = m
                                best_match_j[src] = oth
            i = j
    log(f"  TierB collision counts (>=pairs counted w/ double): {match_ge}")

    # Forensic: P(y_i=1 | neighbor label) among train-train matches
    # use top_nbr within train
    same_pos = same_neg = n_pos = n_neg = 0
    for i in range(n_tr):
        j = int(top_nbr[i])
        if j < 0 or j >= n_tr:
            continue
        if y[j] == 1:
            n_pos += 1
            same_pos += int(y[i] == 1)
        else:
            n_neg += 1
            same_neg += int(y[i] == 1)
    p_given_pos = same_pos / max(n_pos, 1)
    p_given_neg = same_neg / max(n_neg, 1)
    log(f"  P(yi=1|yj=1)={p_given_pos:.4f} P(yi=1|yj=0)={p_given_neg:.4f} base={float(y.mean()):.5f}")

    # Leak-free OOF clone features — vectorized over top/top2/best_match nbrs
    y_uni = np.full(N, np.nan, np.float64)
    y_uni[:n_tr] = y
    y_uni[n_tr + n_te:] = y_orig
    s_uni = np.zeros(N, np.float64)
    s_uni[:n_tr] = core["z_wide_oof"]
    s_uni[n_tr:n_tr + n_te] = core["z_wide_test"]
    y_base = float(y.mean())
    y_orig_base = float(y_orig.mean())

    def _nbr_stack(n0):
        """Return (nbr_idx[n0,3], nbr_sc[n0,3]) for rows [0,n0)."""
        nbr = np.column_stack([top_nbr[:n0], top2_nbr[:n0], best_match_j[:n0] if n0 == n_tr else np.full(n0, -1)])
        sc = np.column_stack([
            top_score[:n0], top2_score[:n0],
            best_match[:n0].astype(np.float32) if n0 == n_tr else np.zeros(n0, np.float32),
        ])
        # for test rows use top/top2 only (+ zeros col)
        if n0 != n_tr:
            nbr = np.column_stack([top_nbr[n_tr:n_tr + n0], top2_nbr[n_tr:n_tr + n0], np.full(n0, -1)])
            sc = np.column_stack([top_score[n_tr:n_tr + n0], top2_score[n_tr:n_tr + n0], np.zeros(n0, np.float32)])
        return nbr.astype(np.int32), sc.astype(np.float64)

    f1 = np.zeros(n_tr, np.float64)
    f2 = np.zeros(n_tr, np.float64)
    nbr_tr, sc_tr = _nbr_stack(n_tr)
    for f in range(5):
        va = fold == f
        tr_m = fold != f
        y_tr_base = float(y[tr_m].mean())
        # labeled residual of neighbor
        num = np.zeros(va.sum())
        den = np.zeros(va.sum())
        ssum = np.zeros(va.sum())
        scnt = np.zeros(va.sum())
        ii = np.where(va)[0]
        for k in range(3):
            j = nbr_tr[ii, k]
            scv = sc_tr[ii, k]
            valid = j >= 0
            # labeled if in tr_m or orig
            in_tr = valid & (j < n_tr)
            in_tr_ok = np.zeros(len(ii), bool)
            in_tr_ok[in_tr] = tr_m[j[in_tr]]
            in_or = valid & (j >= n_tr + n_te)
            use = in_tr_ok | in_or
            yj = np.zeros(len(ii))
            yj[in_tr_ok] = y[j[in_tr_ok]] - y_tr_base
            yj[in_or] = y_uni[j[in_or]] - y_orig_base
            num[use] += scv[use] * yj[use]
            den[use] += scv[use]
            # surrogate among train∪test
            in_st = valid & (j < n_tr + n_te) & (j != ii)
            ssum[in_st] += scv[in_st] * s_uni[j[in_st]]
            scnt[in_st] += scv[in_st]
        f1[ii] = num / (den + 1e-6)
        f2[ii] = ssum / (scnt + 1e-6)

    nbr_te, sc_te = _nbr_stack(n_te)
    f1_te_folds, f2_te_folds = [], []
    for f in range(5):
        tr_m = fold != f
        y_tr_base = float(y[tr_m].mean())
        num = np.zeros(n_te)
        den = np.zeros(n_te)
        ssum = np.zeros(n_te)
        scnt = np.zeros(n_te)
        ii = np.arange(n_te)
        glob = n_tr + ii
        for k in range(3):
            j = nbr_te[:, k]
            scv = sc_te[:, k]
            valid = j >= 0
            in_tr = valid & (j < n_tr)
            in_tr_ok = np.zeros(n_te, bool)
            in_tr_ok[in_tr] = tr_m[j[in_tr]]
            in_or = valid & (j >= n_tr + n_te)
            use = in_tr_ok | in_or
            yj = np.zeros(n_te)
            yj[in_tr_ok] = y[j[in_tr_ok]] - y_tr_base
            yj[in_or] = y_uni[j[in_or]] - y_orig_base
            num[use] += scv[use] * yj[use]
            den[use] += scv[use]
            in_st = valid & (j < n_tr + n_te) & (j != glob)
            ssum[in_st] += scv[in_st] * s_uni[j[in_st]]
            scnt[in_st] += scv[in_st]
        f1_te_folds.append(num / (den + 1e-6))
        f2_te_folds.append(ssum / (scnt + 1e-6))
    f1_te = np.mean(f1_te_folds, 0)
    f2_te = np.mean(f2_te_folds, 0)

    # combine E1 score via nested LOFO ridge on [f1,f2]
    Xo = np.column_stack([L_fold(f1, fold), L_fold(f2, fold)])
    Xt = np.column_stack([L(f1_te), L(f2_te)])
    e1_oof = np.zeros(n_tr)
    e1_te_folds = []
    for f in range(5):
        tr_m, va = fold != f, fold == f
        lr = LogisticRegression(C=0.5, max_iter=300)
        lr.fit(Xo[tr_m], y[tr_m].astype(int))
        e1_oof[va] = lr.predict_proba(Xo[va])[:, 1]
        e1_te_folds.append(lr.predict_proba(Xt)[:, 1])
    e1_te = np.mean(e1_te_folds, 0)
    e1_oof_z = unitize(L_fold(e1_oof, fold))
    e1_te_z = unitize(L(e1_te))

    # id / batch-order audits
    id_tr = tr["id"].to_numpy()
    # lag autocorr of y
    y_ord = y  # assume csv order ≈ id order
    ac = []
    for lag in range(1, 31):
        ac.append(float(np.corrcoef(y_ord[:-lag], y_ord[lag:])[0, 1]))
    # residual autocorr
    p_cal, _ = platt(core["z_wide_oof"], y, core["z_wide_oof"])
    r = y - p_cal
    ac_r = [float(np.corrcoef(r[:-lag], r[lag:])[0, 1]) for lag in range(1, 31)]
    # FFT peak
    fft = np.abs(np.fft.rfft(r - r.mean()))
    fft[0] = 0
    peak = float(fft.max())
    mu, sig = float(fft.mean()), float(fft.std() + 1e-12)
    fft_z = (peak - mu) / sig
    # IEEE decimal string lengths
    def dec_stats(series):
        s = series.astype(str)
        lens = s.str.len()
        trail = s.str.extract(r"\.(\d+)$", expand=False).fillna("").str.len()
        return {"mean_len": float(lens.mean()), "mean_frac": float(trail.mean())}
    ieee = {c: {"train": dec_stats(tr[c]), "test": dec_stats(te[c])} for c in FLOAT_COLS[:3]}
    # id_norm adversarial AUC
    idn = (id_tr - id_tr.min()) / (id_tr.max() - id_tr.min() + 1e-12)
    id_auc = auc(y, idn)

    g_w = gate_25(y, core["z_wide_oof"], e1_oof_z, fold, "E1_vs_wide",
                  core["z_wide_test"], e1_te_z)
    g_a = gate_25(y, z_anchor, e1_oof_z, fold, "E1_vs_anchor") if g_w["mean_lift_e4"] > 0.05 else None
    log(f"  E1 vs_wide {g_w['outer_pos']} Δ={g_w['mean_lift_e4']:.2f}e-4 pass={g_w['pass']}")

    # free heavy
    del S, codes, tr, te, orig
    gc.collect()

    return {
        "oof": e1_oof_z, "test": e1_te_z,
        "gate_wide": g_w, "gate_anchor": g_a,
        "match_ge": match_ge, "pair_touches": pair_touches,
        "p_given_pos": p_given_pos, "p_given_neg": p_given_neg,
        "ac_y_max": float(np.max(np.abs(ac))), "ac_r_max": float(np.max(np.abs(ac_r))),
        "fft_z": fft_z, "ieee": ieee, "id_auc": id_auc,
        "f1_oof": f1, "f2_oof": f2,
    }


# ─── E2 ─────────────────────────────────────────────────────────────
def _tok_ascii(s: str):
    """Deterministic multi-scale ASCII chunks (GPT2TokenizerFast fallback)."""
    s = str(s)
    toks = []
    if len(s) >= 2:
        toks += [f"p2:{s[:2]}", f"s2:{s[-2:]}"]
    if len(s) >= 3:
        toks += [f"p3:{s[:3]}", f"s3:{s[-3:]}"]
    if "." in s:
        a, _, b = s.partition(".")
        toks += [f"int:{a[-4:]}", f"frac:{b[:4]}"]
    else:
        toks.append(f"int:{s[-4:]}")
    toks.append(f"L{min(len(s), 32)}")
    # byte bigrams
    bs = s.encode("utf-8", "ignore")[:24]
    for i in range(len(bs) - 1):
        toks.append(f"b:{bs[i]}-{bs[i+1]}")
    return toks


def engine2(core, z_anchor):
    log("E2 GPT2-BPE/ASCII GLM on raw y")
    y, fold = core["y"], core["fold"]
    n_tr = len(y)
    tr = pd.read_csv(REPO / "data/train.csv", usecols=FEAT_COLS)
    te = pd.read_csv(REPO / "data/test.csv", usecols=FEAT_COLS)
    orig = pd.read_csv(REPO / "original/credit_card_fraud_2026.csv", usecols=FEAT_COLS)
    n_te = len(te)
    N = n_tr + n_te
    DIM = 2 ** 16
    log(f"  unique-value ASCII-BPE hash DIM={DIM} (vectorized expand)")

    def hfeat(*parts):
        return hash(parts) % DIM

    # Build CSR column-blocks via unique→token feature lists expanded by inv_idx
    blocks = []
    tr_cnt = np.zeros(DIM, np.int32)
    te_cnt = np.zeros(DIM, np.int32)

    # regime keys for cross (factorized to small ints)
    auth_all = pd.concat([tr["auth_method"], te["auth_method"]]).astype(str)
    auth_codes = pd.factorize(auth_all)[0]
    auth_tr, auth_te = auth_codes[:n_tr], auth_codes[n_tr:]
    cvv_tr = tr["cvv_retry_count"].astype(int).to_numpy()
    cvv_te = te["cvv_retry_count"].astype(int).to_numpy()
    fbm_tr = np.zeros(n_tr, np.int32)
    fbm_te = np.zeros(n_te, np.int32)
    for i, c in enumerate(BOOL_COLS):
        fbm_tr |= tr[c].astype(int).to_numpy() << i
        fbm_te |= te[c].astype(int).to_numpy() << i
    cd_all = (pd.concat([tr["channel"], te["channel"]]).astype(str) + "|" +
              pd.concat([tr["device_type"], te["device_type"]]).astype(str))
    cd_codes = pd.factorize(cd_all)[0]
    cd_tr, cd_te = cd_codes[:n_tr], cd_codes[n_tr:]

    # RAM budget: floats + cats only, ≤3 toks/value, no bool/int spam
    E2_COLS = FLOAT_COLS + CAT_COLS + ["cvv_retry_count", "prior_disputes"]
    for j, c in enumerate(E2_COLS):
        col_all = pd.concat([tr[c].astype(str), te[c].astype(str), orig[c].astype(str)], ignore_index=True)
        uniq, inv = np.unique(col_all.to_numpy(), return_inverse=True)
        inv_tr, inv_te = inv[:n_tr], inv[n_tr:n_tr + n_te]
        max_f = 3
        feat_mat = np.full((len(uniq), max_f), -1, np.int32)
        for ui, u in enumerate(uniq):
            toks = _tok_ascii(u)
            feats = [hfeat("u", j, p, tok) for p, tok in enumerate(toks[:3])]
            if c in FLOAT_COLS and len(toks) >= 2:
                feats.append(hfeat("bi", j, toks[0], toks[1]))
            feat_mat[ui, : min(len(feats), max_f)] = feats[:max_f]
        for inv_s, row0, is_tr in ((inv_tr, 0, True), (inv_te, n_tr, False)):
            fm = feat_mat[inv_s]
            rows = np.repeat(np.arange(len(inv_s), dtype=np.int32) + row0, max_f)
            cols = fm.ravel()
            m = cols >= 0
            rows, cols = rows[m], cols[m]
            if is_tr:
                np.add.at(tr_cnt, cols, 1)
            else:
                np.add.at(te_cnt, cols, 1)
            blocks.append((rows, cols))
        # one cross: amount/velocity × auth only
        if c in ("amount_usd", "velocity_score"):
            tok0 = feat_mat[:, 0]
            for inv_s, reg, row0, is_tr in (
                (inv_tr, auth_tr, 0, True), (inv_te, auth_te, n_tr, False),
            ):
                t0 = tok0[inv_s]
                cols = ((t0.astype(np.int64) * 1000003 + reg.astype(np.int64) * 9176 + j) % DIM).astype(np.int32)
                rows = np.arange(len(inv_s), dtype=np.int32) + row0
                if is_tr:
                    np.add.at(tr_cnt, cols, 1)
                else:
                    np.add.at(te_cnt, cols, 1)
                blocks.append((rows, cols))
        log(f"  col {c}: uniq={len(uniq)} blocks={len(blocks)} rss={cur_rss_gb():.2f}G")

    active = (tr_cnt >= 30) & (te_cnt >= 8)
    active_idx = np.where(active)[0]
    if len(active_idx) > 8000:
        scores = tr_cnt[active_idx]
        keep = np.argpartition(-scores, 8000)[:8000]
        active_idx = active_idx[keep]
    remap = -np.ones(DIM, np.int32)
    remap[active_idx] = np.arange(len(active_idx), dtype=np.int32)
    log(f"  active={len(active_idx)}")

    # stream-filter into single COO without duplicating all blocks in RAM
    n_est = min(sum(len(r) for r, _ in blocks), N * 8)
    ri = np.empty(n_est, np.int32)
    ci = np.empty(n_est, np.int32)
    ptr = 0
    for rows, cols in blocks:
        cc = remap[cols]
        m = cc >= 0
        rr, cc = rows[m], cc[m]
        need = ptr + len(rr)
        if need > len(ri):
            new = int(need * 1.2)
            ri2, ci2 = np.empty(new, np.int32), np.empty(new, np.int32)
            ri2[:ptr], ci2[:ptr] = ri[:ptr], ci[:ptr]
            ri, ci = ri2, ci2
        ri[ptr:ptr + len(rr)] = rr
        ci[ptr:ptr + len(cc)] = cc
        ptr += len(rr)
    del blocks
    gc.collect()
    X = sparse.csr_matrix(
        (np.ones(ptr, np.float32), (ri[:ptr], ci[:ptr])),
        shape=(N, max(len(active_idx), 1)), dtype=np.float32,
    )
    del ri, ci, tr_cnt, te_cnt
    gc.collect()
    log(f"  CSR {X.shape} nnz={X.nnz} rss={cur_rss_gb():.2f}G")

    Xtr, Xte = X[:n_tr], X[n_tr:]
    num = ["amount_usd", "velocity_score", "merchant_risk_score", "cvv_retry_count",
           "customer_age", "hours_since_last_txn", "account_balance_usd", "prior_disputes"]
    Xn = tr[num].to_numpy(np.float64)
    Xn_te = te[num].to_numpy(np.float64)
    e2_oof = np.zeros(n_tr)
    e2_te_folds = []
    for f in range(5):
        tr_m, va = fold != f, fold == f
        sc = Xn[tr_m].std(0) + 1e-6
        mu = Xn[tr_m].mean(0)
        Zn_tr = (Xn[tr_m] - mu) / sc
        Zn_va = (Xn[va] - mu) / sc
        Zn_te = (Xn_te - mu) / sc
        lr0 = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
        lr0.fit(Zn_tr, y[tr_m].astype(int))
        off_tr = logit(np.clip(lr0.predict_proba(Zn_tr)[:, 1], 1e-6, 1 - 1e-6))
        off_va = logit(np.clip(lr0.predict_proba(Zn_va)[:, 1], 1e-6, 1 - 1e-6))
        off_te = logit(np.clip(lr0.predict_proba(Zn_te)[:, 1], 1e-6, 1 - 1e-6))
        Xtr_f = sparse.hstack([Xtr[tr_m], sparse.csr_matrix(off_tr.reshape(-1, 1))], format="csr")
        Xva_f = sparse.hstack([Xtr[va], sparse.csr_matrix(off_va.reshape(-1, 1))], format="csr")
        Xte_f = sparse.hstack([Xte, sparse.csr_matrix(off_te.reshape(-1, 1))], format="csr")
        lr = LogisticRegression(C=0.1, max_iter=25, solver="sag", n_jobs=N_JOBS)
        lr.fit(Xtr_f, y[tr_m].astype(int))
        e2_oof[va] = lr.predict_proba(Xva_f)[:, 1]
        e2_te_folds.append(lr.predict_proba(Xte_f)[:, 1])
        del Xtr_f, Xva_f, Xte_f
        log(f"  E2 fold{f} va_auc={auc(y[va], e2_oof[va]):.5f}")
    e2_te = np.mean(e2_te_folds, 0)
    e2_z = unitize(L_fold(e2_oof, fold))
    e2_tz = unitize(L(e2_te))
    r_oof, beta = orth_resid(e2_z, core["z_wide_oof"])
    r_te, _ = orth_resid(e2_tz, core["z_wide_test"])
    disp, ks = raw_disp(r_oof, r_te), ks_stat(r_oof, r_te)
    g_w = gate_25(y, core["z_wide_oof"], e2_z, fold, "E2_vs_wide", core["z_wide_test"], e2_tz)
    g_a = gate_25(y, z_anchor, e2_z, fold, "E2_vs_anchor") if g_w["mean_lift_e4"] > 0.05 else None
    log(f"  E2 vs_wide {g_w['outer_pos']} Δ={g_w['mean_lift_e4']:.2f}e-4 disp={disp:.3f} KS={ks:.4f} pass={g_w['pass']}")
    del X, Xtr, Xte, tr, te, orig
    gc.collect()
    return {"oof": e2_z, "test": e2_tz, "gate_wide": g_w, "gate_anchor": g_a,
            "raw_disp": disp, "KS": ks, "beta": beta, "n_active": int(len(active_idx))}


# ─── E3 ─────────────────────────────────────────────────────────────
def engine3(core, z_anchor, z_anchor_te, e1, e2, prop_meta):
    log("E3 antithetic Stage-B")
    y, fold = core["y"], core["fold"]
    n_tr, n_te = len(y), len(core["z_wide_test"])
    import lightgbm as lgb

    # L7 antithetic
    l7o = np.load(REPO / "preds/l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")  # N,10,2
    l7t = np.load(REPO / "preds/l7_s2026_5f/l7_s2026_test_5fold_raw.npy")  # 5,Nte,10,2
    llr_o = l7o[:, :, 1] - l7o[:, :, 0]  # N,10
    # antithetic pairs m & m+5
    P = 0.5 * (llr_o[:, :5] + llr_o[:, 5:10])  # N,5
    # lme_llr tau=0.5 ≈ median across pairs; pair_std
    lme_o = np.median(P, axis=1)
    pstd_o = P.std(axis=1)
    # test: per-fold then average after PCHIP
    lme_te_f, pstd_te_f = [], []
    for f in range(5):
        llr_t = l7t[f, :, :, 1] - l7t[f, :, :, 0]  # Nte,10
        Pf = 0.5 * (llr_t[:, :5] + llr_t[:, 5:10])
        lme_f = np.median(Pf, axis=1)
        pst_f = Pf.std(axis=1)
        # PCHIP align pair_std to oof fold distribution
        va = fold == f
        pst_f = pchip_align(pstd_o[va], pst_f)
        lme_f = pchip_align(lme_o[va], lme_f)
        lme_te_f.append(lme_f)
        pstd_te_f.append(pst_f)
    lme_te = np.mean(lme_te_f, 0)
    pstd_te = np.mean(pstd_te_f, 0)

    # gpt2_med: match K=4
    g_llr_o = np.zeros((n_tr, 4))
    g_llr_t = np.zeros((5, n_te, 4))
    for f in range(5):
        o = np.load(REPO / f"preds/gpt2_med_4ep/oof_{f}.npy")  # nva,8,2
        t = np.load(REPO / f"preds/gpt2_med_4ep/test_{f}.npy")  # Nte,4,2
        va = fold == f
        llr = o[:, :4, 1] - o[:, :4, 0]  # match first 4 perms
        g_llr_o[va] = llr
        g_llr_t[f] = t[:, :4, 1] - t[:, :4, 0]
    g_mean_o = g_llr_o.mean(1)
    g_std_o = g_llr_o.std(1)
    g_mean_te = np.mean([g_llr_t[f].mean(1) for f in range(5)], 0)
    g_std_te = np.mean([pchip_align(g_std_o[fold == f], g_llr_t[f].std(1)) for f in range(5)], 0)

    # signed contrasts
    z_lme = unitize(L_fold(lme_o, fold))
    z_g = unitize(L_fold(g_mean_o, fold))
    z_tree = unitize(L_fold(np.load(REPO / "preds/paradigm_B/xgb_llr_oof.npy").ravel()[:n_tr], fold))
    dis_BPE_QLM = z_g - z_lme
    dis_LM_TAB = z_lme - z_tree
    # test contrasts
    z_lme_t = unitize(L(lme_te))
    z_g_t = unitize(L(g_mean_te))
    z_tree_t = unitize(L(pchip_align(
        np.load(REPO / "preds/paradigm_B/xgb_llr_oof.npy").ravel()[:n_tr],
        np.load(REPO / "preds/paradigm_B/xgb_llr_test.npy").ravel()[:n_te])))
    dis_BPE_QLM_t = z_g_t - z_lme_t
    dis_LM_TAB_t = z_lme_t - z_tree_t

    # structural keys
    tr = pd.read_csv(REPO / "data/train.csv", usecols=["channel", "device_type", "cvv_retry_count"] + BOOL_COLS)
    te = pd.read_csv(REPO / "data/test.csv", usecols=["channel", "device_type", "cvv_retry_count"] + BOOL_COLS)
    chdev = pd.factorize(tr["channel"].astype(str) + "|" + tr["device_type"].astype(str))[0].astype(np.float64)
    chdev_t = pd.factorize(
        pd.concat([tr["channel"].astype(str) + "|" + tr["device_type"].astype(str),
                   te["channel"].astype(str) + "|" + te["device_type"].astype(str)])
    )[0][n_tr:].astype(np.float64)
    fbm = np.zeros(n_tr, np.float64)
    fbm_t = np.zeros(n_te, np.float64)
    for i, c in enumerate(BOOL_COLS):
        fbm += tr[c].astype(float).to_numpy() * (2 ** i)
        fbm_t += te[c].astype(float).to_numpy() * (2 ** i)
    cvv = tr["cvv_retry_count"].to_numpy(float)
    cvv_t = te["cvv_retry_count"].to_numpy(float)

    def pack(dis, chd, fb, cv, e1v, e2v, pstd, gstd):
        return np.column_stack([
            dis, dis * (chd - chd.mean()), dis * (fb - fb.mean()),
            dis * (cv - cv.mean()), e1v, e2v, pstd, gstd,
        ])

    e3_oof = np.zeros(n_tr)
    e3_te_folds = []
    # precompute unitized dispersion once
    upstd_o = unitize(L_fold(pstd_o, fold))
    ugstd_o = unitize(L_fold(g_std_o, fold))
    upstd_t = unitize(L(pstd_te))
    ugstd_t = unitize(L(g_std_te))
    seeds = [42, 7]  # 2-seed bag for budget
    for f in range(5):
        tr_m, va = fold != f, fold == f
        p_tr, lr_p = platt(z_anchor[tr_m], y[tr_m], z_anchor[tr_m])
        p_va = lr_p.predict_proba(z_anchor[va].reshape(-1, 1))[:, 1]
        p_te = lr_p.predict_proba(z_anchor_te.reshape(-1, 1))[:, 1]
        init_tr = logit(np.clip(p_tr, 1e-6, 1 - 1e-6))
        init_va = logit(np.clip(p_va, 1e-6, 1 - 1e-6))
        init_te = logit(np.clip(p_te, 1e-6, 1 - 1e-6))

        Xtr = pack(dis_LM_TAB[tr_m], chdev[tr_m], fbm[tr_m], cvv[tr_m],
                   e1["oof"][tr_m], e2["oof"][tr_m], upstd_o[tr_m], ugstd_o[tr_m])
        Xva = pack(dis_LM_TAB[va], chdev[va], fbm[va], cvv[va],
                   e1["oof"][va], e2["oof"][va], upstd_o[va], ugstd_o[va])
        Xte = pack(dis_LM_TAB_t, chdev_t, fbm_t, cvv_t,
                   e1["test"], e2["test"], upstd_t, ugstd_t)

        preds_va, preds_te = [], []
        for sd in seeds:
            dtr = lgb.Dataset(Xtr, label=y[tr_m], init_score=init_tr, free_raw_data=False)
            params = dict(
                objective="binary", learning_rate=0.02, num_leaves=6, max_depth=3,
                feature_fraction=0.35, bagging_fraction=0.80, bagging_freq=1,
                lambda_l1=15.0, lambda_l2=80.0, max_delta_step=0.15,
                min_sum_hessian_in_leaf=50.0, max_bin=63, n_jobs=N_JOBS,
                verbosity=-1, seed=sd,
            )
            booster = lgb.train(params, dtr, num_boost_round=60)
            preds_va.append(expit(booster.predict(Xva, raw_score=True) + init_va))
            preds_te.append(expit(booster.predict(Xte, raw_score=True) + init_te))
        e3_oof[va] = np.mean(preds_va, 0)
        e3_te_folds.append(np.mean(preds_te, 0))
        log(f"  E3 fold{f} va_auc={auc(y[va], e3_oof[va]):.5f}")
    e3_te = np.mean(e3_te_folds, 0)
    e3_z = unitize(L_fold(e3_oof, fold))
    e3_tz = unitize(L(e3_te))
    g_w = gate_25(y, core["z_wide_oof"], e3_z, fold, "E3_vs_wide", core["z_wide_test"], e3_tz)
    g_a = gate_25(y, z_anchor, e3_z, fold, "E3_vs_anchor") if g_w["mean_lift_e4"] > 0.05 else None
    log(f"  E3 vs_wide {g_w['outer_pos']} Δ={g_w['mean_lift_e4']:.2f}e-4 pass={g_w['pass']}")
    return {
        "oof": e3_z, "test": e3_tz, "gate_wide": g_w, "gate_anchor": g_a,
        "dis_LM_TAB": dis_LM_TAB, "dis_LM_TAB_t": dis_LM_TAB_t,
        "dis_BPE_QLM": dis_BPE_QLM, "lme_o": lme_o, "pstd_o": pstd_o,
    }


# ─── E4 ─────────────────────────────────────────────────────────────
def engine4(core, z_anchor, z_anchor_te, e1, e2, e3):
    log("E4 halluc + twilight opt")
    y, fold = core["y"], core["fold"]
    n_tr, n_te = len(y), len(z_anchor_te)
    tr = pd.read_csv(REPO / "data/train.csv", usecols=["channel", "device_type", "auth_method"] + BOOL_COLS)
    te = pd.read_csv(REPO / "data/test.csv", usecols=["channel", "device_type", "auth_method"] + BOOL_COLS)
    orig = pd.read_csv(REPO / "original/credit_card_fraud_2026.csv",
                       usecols=["channel", "device_type", "auth_method"] + BOOL_COLS)

    def keys(df):
        k1 = (df["channel"].astype(str) + "|" + df["device_type"].astype(str) + "|" + df["auth_method"].astype(str)).to_numpy()
        fbm = np.zeros(len(df), np.int32)
        for i, c in enumerate(BOOL_COLS):
            fbm |= df[c].astype(int).to_numpy() << i
        return k1, fbm.astype(str)

    k_tr, f_tr = keys(tr)
    k_te, f_te = keys(te)
    k_or, f_or = keys(orig)
    # universe train∪test counts vs orig
    def halluc(k_all, k_o):
        c_all = pd.Series(k_all).value_counts()
        c_o = pd.Series(k_o).value_counts()
        # score per key
        sc = {}
        for k, cnt in c_all.items():
            sc[k] = np.log((cnt + 1) / (50 * c_o.get(k, 0) + 1))
        return sc

    kall = np.concatenate([k_tr, k_te])
    fall = np.concatenate([f_tr, f_te])
    sc_k = halluc(kall, k_or)
    sc_f = halluc(fall, f_or)
    h_tr = np.array([sc_k.get(k, 0.0) + sc_f.get(f, 0.0) for k, f in zip(k_tr, f_tr)])
    h_te = np.array([sc_k.get(k, 0.0) + sc_f.get(f, 0.0) for k, f in zip(k_te, f_te)])
    # interaction with dis
    inter = h_tr * e3["dis_LM_TAB"]
    inter_t = h_te * e3["dis_LM_TAB_t"]

    # Twilight-band smooth pairwise optimizer
    passing = []
    for name, oof, tev in [
        ("E1", e1["oof"], e1["test"]),
        ("E2", e2["oof"], e2["test"]),
        ("E3", e3["oof"], e3["test"]),
        ("halluc_x_dis", unitize(L_fold(inter, fold)), unitize(L(inter_t))),
    ]:
        g = e1["gate_wide"] if name == "E1" else e2["gate_wide"] if name == "E2" else e3["gate_wide"] if name == "E3" else None
        # include all as candidates; twilight will shrink
        passing.append((name, oof, tev))

    e4_oof = np.zeros(n_tr)
    e4_te_folds = []
    twilight_attr = {}
    for f in range(5):
        tr_m, va = fold != f, fold == f
        # active band on z_anchor
        q = pd.Series(z_anchor[tr_m]).rank(pct=True).to_numpy()
        band = (q >= 0.80) & (q <= 0.985)
        # design: z_anchor + candidates
        cols = [z_anchor] + [p[1] for p in passing]
        X = np.column_stack(cols)
        Xtr, Xva = X[tr_m], X[va]
        Xt = np.column_stack([z_anchor_te] + [p[2] for p in passing])
        # smooth pairwise logistic loss on twilight band subsample
        idx = np.where(tr_m)[0][band]
        if len(idx) > 25000:
            idx = _strat_sub(y, idx, 25000, 42 + f)
        yb = y[idx]
        Xb = X[idx]
        pos = Xb[yb == 1]
        neg = Xb[yb == 0]
        rng = np.random.default_rng(42 + f)
        # sample pairs
        n_pairs = min(8000, len(pos) * 3)
        pi = rng.integers(0, len(pos), n_pairs)
        ni = rng.integers(0, len(neg), n_pairs)
        Dp = pos[pi] - neg[ni]  # (n_pairs, d)

        def loss(w):
            s = Dp @ w
            # softplus(-s) ≈ -log sigmoid(s)
            return float(np.mean(np.logaddexp(0, -s)) + 0.5 * 2.0 * np.dot(w, w))

        def grad(w):
            s = Dp @ w
            # d/dw softplus(-s) = -sigmoid(-s) * Dp
            g = -(expit(-s)[:, None] * Dp).mean(0) + 2.0 * w
            return g

        w0 = np.zeros(X.shape[1])
        w0[0] = 1.0
        res = minimize(loss, w0, jac=grad, method="L-BFGS-B", options=dict(maxiter=40))
        w = res.x
        e4_oof[va] = Xva @ w
        e4_te_folds.append(Xt @ w)
        # attribution: twilight vs high-tail ΔAUC on va
        qva = pd.Series(z_anchor[va]).rank(pct=True).to_numpy()
        tw = (qva >= 0.80) & (qva <= 0.985)
        ht = qva > 0.985
        pred = e4_oof[va]
        base = z_anchor[va]
        def safe_d(mask):
            if mask.sum() < 50 or y[va][mask].sum() < 5:
                return 0.0
            return auc(y[va][mask], pred[mask]) - auc(y[va][mask], base[mask])
        twilight_attr[f] = {"twilight": safe_d(tw), "high_tail": safe_d(ht)}
        log(f"  E4 fold{f} twΔ={twilight_attr[f]['twilight']*1e4:.2f}e-4 htΔ={twilight_attr[f]['high_tail']*1e4:.2f}e-4")

    e4_te = np.mean(e4_te_folds, 0)
    e4_z = unitize(L_fold(e4_oof, fold))
    e4_tz = unitize(L(e4_te))
    g_w = gate_25(y, core["z_wide_oof"], e4_z, fold, "E4_vs_wide", core["z_wide_test"], e4_tz)
    g_a = gate_25(y, z_anchor, e4_z, fold, "E4_vs_anchor") if g_w["mean_lift_e4"] > 0.05 else None
    log(f"  E4 vs_wide {g_w['outer_pos']} Δ={g_w['mean_lift_e4']:.2f}e-4 pass={g_w['pass']}")
    return {"oof": e4_z, "test": e4_tz, "gate_wide": g_w, "gate_anchor": g_a,
            "twilight_attr": twilight_attr, "halluc_oof": unitize(L_fold(inter, fold))}


# ─── FINAL ──────────────────────────────────────────────────────────
def loser_cone(u, losers):
    u = unitize(u)
    for d in losers:
        d = unitize(d)
        c = float(np.dot(u, d) / len(u))
        # actually cosine
        cos = float(np.dot(u, d) / (np.linalg.norm(u) * np.linalg.norm(d) + 1e-12))
        if cos > 0:
            u = u - cos * d * (np.linalg.norm(u) / (np.linalg.norm(d) + 1e-12))
            # one-sided: u <- u - max(0,<u,dhat>)*dhat
            # redo properly:
    u = unitize(u.copy())
    for d in losers:
        dhat = unitize(d)
        inp = float(np.dot(u, dhat))
        if inp > 0:
            u = u - inp * dhat
    return unitize(u)


def finalize(core, z_anchor, z_anchor_te, engines, prop_meta, delta_meta):
    log("FINAL audit + staging")
    y, fold = core["y"], core["fold"]
    n_te = len(core["z_wide_test"])
    rows = []
    modules = {
        "z_wide": (core["z_wide_oof"], core["z_wide_test"]),
        "z_anchor": (z_anchor, z_anchor_te),
        "E1": (engines["E1"]["oof"], engines["E1"]["test"]),
        "E2": (engines["E2"]["oof"], engines["E2"]["test"]),
        "E3": (engines["E3"]["oof"], engines["E3"]["test"]),
        "E4": (engines["E4"]["oof"], engines["E4"]["test"]),
    }
    # nested LOFO combo of passing
    passing = []
    for name in ("E1", "E2", "E3", "E4"):
        gw = engines[name]["gate_wide"]
        ga = engines[name].get("gate_anchor") or {}
        rows.append({
            "name": name,
            "solo_auc": gw.get("solo_auc"),
            "d_wide_e4": gw["mean_lift_e4"],
            "d_anchor_e4": ga.get("mean_lift_e4"),
            "outer": gw["outer_pos"], "sub": gw["sub_pos"],
            "t": gw["t_stat"], "p": gw["wilcoxon_p"],
            "disp": gw["raw_disp"], "KS": gw["KS"], "pass": gw["pass"],
        })
        if gw["pass"] or ga.get("pass"):
            passing.append(name)

    # always try combo of all engines + anchor via nested LOFO
    cols_o = [z_anchor] + [engines[n]["oof"] for n in ("E1", "E2", "E3", "E4")]
    cols_t = [z_anchor_te] + [engines[n]["test"] for n in ("E1", "E2", "E3", "E4")]
    Xo = np.column_stack(cols_o)
    Xt = np.column_stack(cols_t)
    combo = np.zeros(len(y))
    combo_te_f = []
    for f in range(5):
        tr_m, va = fold != f, fold == f
        ridge = Ridge(alpha=10.0)
        ridge.fit(Xo[tr_m], y[tr_m])
        combo[va] = ridge.predict(Xo[va])
        combo_te_f.append(ridge.predict(Xt))
    combo_te = np.mean(combo_te_f, 0)
    combo_z = unitize(L_fold(combo, fold))
    combo_tz = unitize(L(combo_te))
    g_w = gate_25(y, core["z_wide_oof"], combo_z, fold, "COMBO_vs_wide", core["z_wide_test"], combo_tz)
    g_a = gate_25(y, z_anchor, combo_z, fold, "COMBO_vs_anchor")
    rows.append({
        "name": "COMBO", "solo_auc": g_w["solo_auc"], "d_wide_e4": g_w["mean_lift_e4"],
        "d_anchor_e4": g_a["mean_lift_e4"], "outer": g_w["outer_pos"], "sub": g_w["sub_pos"],
        "t": g_w["t_stat"], "p": g_w["wilcoxon_p"], "disp": g_w["raw_disp"], "KS": g_w["KS"],
        "pass": g_w["pass"],
    })
    if g_w["pass"]:
        passing.append("COMBO")

    # champion / loser cone
    ids = pd.read_csv(REPO / "data/test.csv", usecols=["id"])["id"].to_numpy()
    champ_path = REPO / "submissions/sub_v31_2_SLOT1_C2_GateA_native_momentum.csv"
    champ = pd.read_csv(champ_path).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    champ_z = unitize(L(champ))
    losers = []
    for p in [
        REPO / "preds/v10_transductive_bulsah/soft_diff_orth_LM_test.npy",
        REPO / "preds/v10_transductive_bulsah/soft_diff_orth_LM_test_raw.npy",
    ]:
        if p.exists():
            losers.append(np.load(p).ravel()[:n_te].astype(float))

    staged = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    any_pass = bool(passing)
    # stage tensors always for audit
    np.save(OUT_DIR / "z_wide_oof.npy", core["z_wide_oof"])
    np.save(OUT_DIR / "z_anchor_oof.npy", z_anchor)
    np.save(OUT_DIR / "z_anchor_test.npy", z_anchor_te)
    for n in ("E1", "E2", "E3", "E4"):
        np.save(OUT_DIR / f"{n}_oof.npy", engines[n]["oof"])
        np.save(OUT_DIR / f"{n}_test.npy", engines[n]["test"])
    np.save(OUT_DIR / "COMBO_oof.npy", combo_z)
    np.save(OUT_DIR / "COMBO_test.npy", combo_tz)

    spear_te = combo_tz.copy()
    hedge_te = combo_tz.copy()
    if losers:
        spear_te = loser_cone(spear_te, losers)
        # hedge: 0.7 combo + 0.3 champ
        hedge_te = unitize(0.7 * unitize(combo_tz) + 0.3 * champ_z)

    cos_champ = float(np.dot(unitize(combo_tz), champ_z) / (
        np.linalg.norm(unitize(combo_tz)) * np.linalg.norm(champ_z) + 1e-12))

    csvs = []
    if any_pass or g_w["mean_lift_e4"] > 0.5:
        # stage NOSUBMIT CSVs if any engine clears OR combo shows meaningful lift
        # Strict rule: only if at least one clears 5/5 + 18/25
        strict_pass = any(
            engines[n]["gate_wide"]["pass"] for n in ("E1", "E2", "E3", "E4")
        ) or g_w["pass"]
        if strict_pass:
            for tag, vec in [("SPEARHEAD", spear_te), ("PRIVATE_HEDGE", hedge_te)]:
                # convert to probability-like via rank
                p = expit(unitize(vec))
                path = SUB_DIR / f"sub_v34_1_{tag}_NOSUBMIT.csv"
                pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(path, index=False)
                csvs.append(str(path))
                staged.append(tag)
            log(f"  staged NOSUBMIT CSVs: {csvs}")
        else:
            log("  no engine cleared strict 25-subfold gate — skip NOSUBMIT CSVs")
    else:
        log("  no positive lift — skip staging CSVs")

    return {
        "rows": rows, "passing": passing, "csvs": csvs, "cos_champ": cos_champ,
        "combo_gate": g_w, "combo_gate_anchor": g_a,
        "prop_meta": prop_meta, "delta_meta": delta_meta,
        "peak_rss_gb": _mon["peak_rss_gb"],
    }


def write_report(core, delta_meta, prop_meta, engines, final):
    lines = [
        "# GM V34.1 FIRST-PRINCIPLES BREAKTHROUGH REPORT", "",
        f"- NO_SUBMIT: `{NO_SUBMIT}`",
        f"- elapsed_s: {time.time()-T0:.1f}",
        f"- peak_rss_gb: {_mon['peak_rss_gb']:.3f} (cap {RSS_CAP_GB})",
        f"- z_wide_oof AUC: {auc(core['y'], core['z_wide_oof']):.6f}",
        "",
        "## Anchor Hygiene", "",
        f"- delta_cat: {json.dumps(delta_meta)}",
        "- proprietary:",
    ]
    for m in prop_meta:
        lines.append(f"  - {m}")
    lines += ["", "## Engine Gates", "",
              "| module | solo_auc | Δwide e-4 | Δanchor e-4 | outer | sub | t | p | disp | KS | pass |",
              "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|"]
    for r in final["rows"]:
        lines.append(
            f"| {r['name']} | {r['solo_auc']:.6f} | {r['d_wide_e4']:.2f} | "
            f"{r['d_anchor_e4'] if r['d_anchor_e4'] is not None else '—'} | "
            f"{r['outer']} | {r['sub']} | {r['t']:.2f} | {r['p']:.4g} | "
            f"{r['disp']:.3f} | {r['KS']:.4f} | {r['pass']} |"
        )
    e1 = engines["E1"]
    lines += ["", "## E1 Forensics", "",
              f"- TierB match_ge: {e1['match_ge']}",
              f"- pair_touches: {e1['pair_touches']}",
              f"- P(yi=1|yj=1)={e1['p_given_pos']:.4f} P(yi=1|yj=0)={e1['p_given_neg']:.4f}",
              f"- max|ac(y)|={e1['ac_y_max']:.4f} max|ac(r)|={e1['ac_r_max']:.4f} fft_z={e1['fft_z']:.2f}",
              f"- id_norm adversarial AUC={e1['id_auc']:.4f}",
              f"- ieee: {json.dumps(e1['ieee'])}", ""]
    lines += ["## E2", "", f"- n_active={engines['E2'].get('n_active')} disp={engines['E2'].get('raw_disp')} KS={engines['E2'].get('KS')}", ""]
    lines += ["## E4 Twilight Attribution", "", json.dumps(engines["E4"].get("twilight_attr"), indent=2), ""]
    lines += ["## Staging", "",
              f"- passing: {final['passing']}",
              f"- cos(combo, SLOT1/0.95537)={final['cos_champ']:.4f}",
              f"- NOSUBMIT CSVs: {final['csvs']}", "",
              "## Verdict", ""]
    if final["passing"]:
        lines.append(f"- Engines clearing gate: {final['passing']}")
    else:
        lines.append("- **No engine cleared the 25-subfold gate.** First-principles signals did not beat frozen `z_wide_oof` under leakage-free validation.")
    lines += ["", f"**STRICT NO_SUBMIT={NO_SUBMIT} — 0 submits, 0 pods, 0 Kaggle pulls.**", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    log(f"wrote {REPORT}")


def main():
    assert NO_SUBMIT is True
    os.environ.setdefault("OMP_NUM_THREADS", str(N_JOBS))
    os.environ.setdefault("MKL_NUM_THREADS", str(N_JOBS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_JOBS))
    th = threading.Thread(target=_monitor_loop, daemon=True)
    th.start()
    log("START GM V34.1 NO_SUBMIT")
    core = load_core()
    delta_meta = audit_delta_cat(core)
    z_anchor, z_anchor_te, prop_meta = build_z_anchor(core, delta_meta.get("keep", False))
    e1 = engine1(core, z_anchor)
    e2 = engine2(core, z_anchor)
    e3 = engine3(core, z_anchor, z_anchor_te, e1, e2, prop_meta)
    e4 = engine4(core, z_anchor, z_anchor_te, e1, e2, e3)
    engines = {"E1": e1, "E2": e2, "E3": e3, "E4": e4}
    final = finalize(core, z_anchor, z_anchor_te, engines, prop_meta, delta_meta)
    write_report(core, delta_meta, prop_meta, engines, final)
    (REPO / "logs/overnight/GM_V34_1_FIRST_PRINCIPLES.json").write_text(
        json.dumps({"rows": final["rows"], "passing": final["passing"], "delta": delta_meta,
                    "prop": prop_meta, "peak_rss": _mon["peak_rss_gb"],
                    "elapsed": time.time() - T0}, indent=2, default=str)[:400000]
    )
    _mon["stop"] = True
    log(f"DONE in {time.time()-T0:.1f}s peak_rss={_mon['peak_rss_gb']:.3f}GB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        _mon["stop"] = True
        traceback.print_exc()
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("# GM V34.1 PARTIAL FAILURE\n\n```\n" + traceback.format_exc() + "\n```\n")
        raise
