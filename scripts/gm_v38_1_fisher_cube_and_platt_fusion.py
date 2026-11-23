#!/usr/bin/env python3
"""GM V38.1 — Fisher Residual-TE cube + Platt family fusion + 60k/240k #1 sim.

STRICT: $0 cloud GPU · ZERO outer-fold echo · ≤1 Kaggle submit iff all 6 gates pass.
Budget: <10 min Mac CPU · <800 MB RSS (int32 hashing + gc).
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import time
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "logs" / "overnight"
SUB = REPO / "submissions"
PRED = REPO / "preds"
V36 = PRED / "v36_1"
M2 = PRED / "matrix_v2"
ATH_CSV = SUB / "sub_v35_S4_v10gate_StepO_0.30e4_on_areeb.csv"
FOLDS_CSV = REPO / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv"
AMAN = (
    REPO
    / "external/v37_twin_check/vaults/amanatar_s6e11-sovereign-095423-simplex-and-oof-vault"
    / "s6e11_sovereign_095423_vault.npz"
)
BUSY = REPO / "external/v37_busyaprime_own"
ORIG20K = REPO / "original/credit_card_fraud_2026.csv"
if not ORIG20K.exists():
    ORIG20K = REPO / "external/v37_twin_check/orig20k_uditjain.csv"

N_TRAIN, N_TEST = 699_960, 299_984
PREV = 0.0167
ATH_LB = 0.95548
TARGET_LB = 0.96242
T0 = time.time()

# refined signed weight grid (incl 0)
_WPOS = [0.0002, 0.0005, 0.001, 0.002, 0.0035, 0.005, 0.008, 0.012, 0.018, 0.025, 0.035, 0.05]
W_GRID = np.array([-w for w in reversed(_WPOS)] + [0.0] + _WPOS, float)

CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
FLOATS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]
INTS = [
    "txn_count_last_24h",
    "card_age_months",
    "customer_age",
    "cvv_retry_count",
    "time_of_day_hour",
    "day_of_week",
    "prior_disputes",
]


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / max(len(v), 1)


def L(v):
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6))


def L_fold(v, folds):
    v = np.asarray(v, float).ravel()
    out = np.zeros_like(v, dtype=np.float64)
    for k in np.unique(folds):
        m = folds == k
        out[m] = L(v[m])
    return out


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def winsor(v, k=4.0):
    return np.clip(unit(v), -k, k)


def auc(y, p):
    return float(roc_auc_score(y, np.asarray(p, float).ravel()))


def pchip_align(oof, test):
    o = np.sort(np.asarray(oof, float).ravel())
    t = np.asarray(test, float).ravel()
    rt = (rankdata(t, method="average") - 0.5) / max(len(t), 1)
    q = np.linspace(0.5 / len(o), 1.0 - 0.5 / len(o), len(o))
    if len(o) > 40000:
        idx = np.linspace(0, len(o) - 1, 40000).astype(int)
        o, q = o[idx], q[idx]
    f = PchipInterpolator(q, o, extrapolate=True)
    return f(np.clip(rt, q[0], q[-1])).astype(np.float64)


def svd_basis(X, k):
    Xc = X - X.mean(0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return Xc[:, : min(k, X.shape[1])]
    return Xc @ vt[:k].T


def gs_strip(u, cols):
    out = unit(u)
    for c in cols:
        c = np.asarray(c, float)
        seq = [c[:, j] for j in range(c.shape[1])] if c.ndim == 2 else [c.ravel()]
        for col in seq:
            bb = unit(col)
            out = unit(out - np.dot(out, bb) / (np.dot(bb, bb) + 1e-12) * bb)
    return winsor(out)


def hash_pair(a, b, salt=0):
    """int32 streaming hash for 2-way / 3-way tokens."""
    x = (a.astype(np.int64) * 1_000_003 + b.astype(np.int64) + salt) & 0x7FFFFFFF
    return x.astype(np.int32)


def hash3(a, b, c, salt=0):
    x = (a.astype(np.int64) * 1_000_003 + b.astype(np.int64) * 9173 + c.astype(np.int64) + salt) & 0x7FFFFFFF
    return x.astype(np.int32)


# -------------------- Stage 1: 60k/240k MC gap proof --------------------
def stage1_mc_gap(y, B):
    """30 stratified 60k-public / 240k-private probes of bank ceiling (fast)."""
    log("[S1] 60k/240k Monte-Carlo bank ceiling vs 0.96242")
    rng = np.random.default_rng(2026)
    # SVD compress bank → 64 comps (preserves span; <<800MB; fast probes)
    n_col = B.shape[1]
    B32 = np.nan_to_num(B.astype(np.float32), nan=0.0)
    B32 -= B32.mean(0, keepdims=True)
    # economy SVD on subsample for basis, then project all
    idx = rng.choice(len(B32), size=min(80000, len(B32)), replace=False)
    _, _, vt = np.linalg.svd(B32[idx], full_matrices=False)
    k_svd = min(64, vt.shape[0])
    Bc = (B32 @ vt[:k_svd].T).astype(np.float32)
    Bc = np.column_stack([unit(Bc[:, j]) for j in range(Bc.shape[1])]).astype(np.float32)
    # also keep top-|corr with y| raw cols for K=114/210 probes on original space (subsample cols)
    # pick up to 210 columns by |corr| on 40k subsample
    sub = rng.choice(len(y), 40000, replace=False)
    corrs = np.array([abs(float(np.corrcoef(B32[sub, j], y[sub])[0, 1])) for j in range(n_col)])
    order = np.argsort(-corrs)
    Ks = [min(114, n_col), min(210, n_col)]
    Ks = sorted(set(Ks))
    results = {"n_bank": n_col, "target": TARGET_LB, "ath": ATH_LB, "trials": []}

    def stratified_pub_priv(y_all, n_pub=60_000, seed=0):
        r = np.random.default_rng(seed)
        pos = np.where(y_all == 1)[0]
        neg = np.where(y_all == 0)[0]
        n_pos = int(round(n_pub * (len(pos) / len(y_all))))
        n_neg = n_pub - n_pos
        pub = np.concatenate([r.choice(pos, n_pos, replace=False), r.choice(neg, n_neg, replace=False)])
        mask = np.zeros(len(y_all), bool)
        mask[pub] = True
        return mask

    max_pub_coord = -1.0
    max_priv_at_coord = -1.0
    max_pub_oracle = -1.0
    max_priv_at_oracle = -1.0
    # pre-unitize selected raw columns once
    raw_by_K = {}
    for K in Ks:
        cols = order[:K]
        X = np.column_stack([unit(B32[:, j]) for j in cols]).astype(np.float32)
        raw_by_K[K] = X

    for t in range(30):
        pub = stratified_pub_priv(y, 60_000, seed=1000 + t)
        priv = ~pub
        y_pub, y_priv = y[pub], y[priv]

        # (b) cheating ridge oracle on SVD-64 bank @ 60k public
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(Bc[pub], y_pub.astype(float))
        a_pub_o = auc(y_pub, ridge.predict(Bc[pub]))
        a_priv_o = auc(y_priv, ridge.predict(Bc[priv]))
        if a_pub_o > max_pub_oracle:
            max_pub_oracle, max_priv_at_oracle = a_pub_o, a_priv_o

        # (a) K∈{114,210} CMA-lite / coord probes (5-decimal quantized)
        best_pub_t, best_priv_t = -1.0, -1.0
        for K in Ks:
            Xk = raw_by_K[K][pub]
            Xp = raw_by_K[K][priv]
            w = np.ones(K, float) / K
            best = round(auc(y_pub, Xk @ w), 5)
            best_w = w.copy()
            # sparse coordinate: only top-16 |w| dims × 2 rounds × 2 deltas
            for _ in range(2):
                js = np.argsort(-np.abs(best_w))[:16]
                for j in js:
                    for delta in (+0.35, -0.35):
                        wt = best_w.copy()
                        wt[j] += delta
                        a = round(auc(y_pub, Xk @ wt), 5)
                        if a > best:
                            best, best_w = a, wt
            # CMA-lite isotropic probes (allows 1-ρ up to ~200e-4 in score space)
            for _ in range(24):
                wt = best_w + rng.normal(0, 0.5, size=K)
                a = round(auc(y_pub, Xk @ wt), 5)
                if a > best:
                    best, best_w = a, wt
            priv_a = auc(y_priv, Xp @ best_w)
            if best > best_pub_t:
                best_pub_t, best_priv_t = best, priv_a

        if best_pub_t > max_pub_coord:
            max_pub_coord, max_priv_at_coord = best_pub_t, best_priv_t

        if t < 5 or t == 29:
            results["trials"].append(
                {
                    "t": t,
                    "pub_coord": best_pub_t,
                    "priv_coord": best_priv_t,
                    "pub_oracle": a_pub_o,
                    "priv_oracle": a_priv_o,
                }
            )
        if t % 10 == 9:
            log(f"[S1] trial {t+1}/30 max_pub_coord={max_pub_coord:.5f} oracle={max_pub_oracle:.5f}")

    gap_bank, gap_oracle = max_pub_coord, max_pub_oracle
    reachable_coord = gap_bank >= 0.9575
    reachable_962 = gap_bank >= TARGET_LB - 1e-5
    oracle_962 = gap_oracle >= TARGET_LB - 1e-5
    verdict = {
        "max_60k_pub_coord_CMA": float(gap_bank),
        "priv_240k_at_best_coord": float(max_priv_at_coord),
        "max_60k_pub_ridge_oracle": float(gap_oracle),
        "priv_240k_at_best_oracle": float(max_priv_at_oracle),
        "reachable_0.9575_via_bank_probe": bool(reachable_coord),
        "reachable_0.96242_via_bank_probe": bool(reachable_962),
        "reachable_0.96242_via_cheating_oracle": bool(oracle_962),
        "math_proof": (
            "EXTERNAL/NON-BANK paradigm required to explain live 0.96242"
            if (not oracle_962 and not reachable_962)
            else (
                f"Cheating 60k ridge-oracle max={gap_oracle:.5f}; "
                f"unconstrained K-probe max={gap_bank:.5f}. "
                + (
                    "0.96242 is within lucky-Public / oracle reach on this bank."
                    if oracle_962 or reachable_962
                    else "0.9575–0.9624 not reached by bank probe → external paradigm."
                )
            )
        ),
        "trials_head": results["trials"],
        "n_bank_cols": n_col,
        "Ks": Ks,
    }
    log(
        f"[S1] max_pub_coord={gap_bank:.5f} priv@coord={max_priv_at_coord:.5f} "
        f"oracle={gap_oracle:.5f} → {verdict['math_proof'][:100]}"
    )
    del Bc, B32, raw_by_K
    gc.collect()
    return verdict


# -------------------- Token dictionary (int32 codes) --------------------
def build_token_dict(tr_df, te_df):
    """~72 discrete token columns as int32 codes; support-filter later."""
    log("[S2A] building discrete token dictionary")
    tokens_tr, tokens_te, names = [], [], []

    def add(name, a, b):
        # factorize on union
        lab = pd.factorize(pd.concat([pd.Series(a), pd.Series(b)], ignore_index=True))[0]
        tokens_tr.append(lab[: len(a)].astype(np.int32))
        tokens_te.append(lab[len(a) :].astype(np.int32))
        names.append(name)

    # raw cats + ints + flags
    for c in CATS + INTS:
        add(c, tr_df[c].astype(str).to_numpy(), te_df[c].astype(str).to_numpy())
    for c in FLAGS:
        add(c, tr_df[c].astype(int).to_numpy(), te_df[c].astype(int).to_numpy())

    # flag bitmask 64
    bm_tr = np.zeros(len(tr_df), np.int32)
    bm_te = np.zeros(len(te_df), np.int32)
    for i, c in enumerate(FLAGS):
        bm_tr |= (tr_df[c].astype(int).to_numpy() << i).astype(np.int32)
        bm_te |= (te_df[c].astype(int).to_numpy() << i).astype(np.int32)
    add("flag_bitmask_64", bm_tr, bm_te)

    # Alibay channel×device×merchant
    add(
        "ch_dev_mer",
        (
            tr_df["channel"].astype(str)
            + "|"
            + tr_df["device_type"].astype(str)
            + "|"
            + tr_df["merchant_category"].astype(str)
        ).to_numpy(),
        (
            te_df["channel"].astype(str)
            + "|"
            + te_df["device_type"].astype(str)
            + "|"
            + te_df["merchant_category"].astype(str)
        ).to_numpy(),
    )

    # sub-float tokens on continuous
    for c in FLOATS:
        xtr = tr_df[c].to_numpy(float)
        xte = te_df[c].to_numpy(float)
        add(f"{c}__int", np.floor(xtr).astype(np.int32), np.floor(xte).astype(np.int32))
        add(
            f"{c}__cents",
            (np.floor(np.abs(xtr) * 100) % 100).astype(np.int32),
            (np.floor(np.abs(xte) * 100) % 100).astype(np.int32),
        )
        # 3rd–4th decimals
        add(
            f"{c}__d34",
            (np.floor(np.abs(xtr) * 10000) % 100).astype(np.int32),
            (np.floor(np.abs(xte) * 10000) % 100).astype(np.int32),
        )
        # ASCII length proxy: digits before/after decimal via log10 (fast)
        def strlen_proxy(arr):
            a = np.abs(arr) + 1e-12
            return (np.floor(np.log10(a)).astype(np.int32) + 6).clip(1, 20)

        add(f"{c}__strlen", strlen_proxy(xtr), strlen_proxy(xte))
        # last non-zero digit proxy from scaled integer
        def last_nz(arr):
            scaled = np.floor(np.abs(arr) * 1e6).astype(np.int64)
            out = np.zeros(len(arr), np.int32)
            m = scaled > 0
            s = scaled[m]
            # strip trailing zeros
            for _ in range(8):
                z = s % 10 == 0
                if not z.any():
                    break
                s = np.where(z, s // 10, s)
            out[m] = (s % 10).astype(np.int32)
            return out

        add(f"{c}__lastnz", last_nz(xtr), last_nz(xte))
        # 250-quantile bin (fit on train∪test ranks via train qcut edges)
        try:
            _, bins = pd.qcut(xtr, 250, retbins=True, duplicates="drop")
            bins[0], bins[-1] = -np.inf, np.inf
            add(
                f"{c}__q250",
                pd.cut(xtr, bins=bins, labels=False).fillna(0).astype(np.int32).to_numpy(),
                pd.cut(xte, bins=bins, labels=False).fillna(0).astype(np.int32).to_numpy(),
            )
        except Exception:
            pass

    # SCAN6 stealth-FN: age_bin × balance_decile × auth × card × cvv
    age_tr = pd.qcut(tr_df["customer_age"].rank(method="first"), 10, labels=False, duplicates="drop").astype(int)
    age_te = pd.qcut(te_df["customer_age"].rank(method="first"), 10, labels=False, duplicates="drop").astype(int)
    bal_tr = pd.qcut(tr_df["account_balance_usd"].rank(method="first"), 10, labels=False, duplicates="drop").astype(int)
    bal_te = pd.qcut(te_df["account_balance_usd"].rank(method="first"), 10, labels=False, duplicates="drop").astype(int)
    add(
        "SCAN6",
        (
            age_tr.astype(str)
            + "|"
            + bal_tr.astype(str)
            + "|"
            + tr_df["auth_method"].astype(str)
            + "|"
            + tr_df["card_type"].astype(str)
            + "|"
            + tr_df["cvv_retry_count"].astype(str)
        ).to_numpy(),
        (
            age_te.astype(str)
            + "|"
            + bal_te.astype(str)
            + "|"
            + te_df["auth_method"].astype(str)
            + "|"
            + te_df["card_type"].astype(str)
            + "|"
            + te_df["cvv_retry_count"].astype(str)
        ).to_numpy(),
    )

    Tok_tr = np.column_stack(tokens_tr)
    Tok_te = np.column_stack(tokens_te)
    # support filter: n_train>=8 and n_test>=3 for levels — keep columns; filter cells in TE
    log(f"[S2A] tokens={Tok_tr.shape[1]} names_sample={names[:8]}…")
    return Tok_tr, Tok_te, names


def support_mask_levels(codes_tr, codes_te, n_tr_min=8, n_te_min=3):
    """Return boolean mask over rows where level passes support (train∪test)."""
    # map code -> counts
    mx = int(max(codes_tr.max(), codes_te.max())) + 1
    ctr = np.bincount(codes_tr, minlength=mx)
    cte = np.bincount(codes_te, minlength=mx)
    ok = (ctr >= n_tr_min) & (cte >= n_te_min)
    return ok[codes_tr], ok[codes_te], ok


# -------------------- zero-echo p_clean inside tr_f --------------------
def fit_p_clean_outer(Xnum, y, folds):
    """For each outer fold: 5-inner SKF × 3 seeds strictly inside tr_f → p_clean OOF + test-ready models."""
    log("[S2A] zero-echo inner-fold p_clean (NO z_fullbank echo)")
    p_clean = np.zeros(len(y), float)
    models = []  # per outer fold: list of (seed_models) for test
    for f in range(5):
        tr = folds != f
        va = folds == f
        idx = np.where(tr)[0]
        ytr = y[tr]
        Xtr = Xnum[tr]
        acc = np.zeros(tr.sum(), float)
        nacc = np.zeros(tr.sum(), float)
        fold_models = []
        for seed in (11, 22, 33):
            skf = StratifiedKFold(5, shuffle=True, random_state=seed)
            seed_clfs = []
            for itr, iva in skf.split(np.zeros(len(ytr)), ytr):
                clf = LogisticRegression(C=0.5, max_iter=120, solver="lbfgs")
                clf.fit(Xtr[itr], ytr[itr])
                pred = clf.predict_proba(Xtr[iva])[:, 1]
                acc[iva] += pred
                nacc[iva] += 1.0
                seed_clfs.append(clf)
            # also fit on all tr for test
            clf_all = LogisticRegression(C=0.5, max_iter=120, solver="lbfgs")
            clf_all.fit(Xtr, ytr)
            fold_models.append(clf_all)
        p_clean[tr] = acc / np.maximum(nacc, 1.0)
        # for va rows: predict with mean of fold_models (fit only on tr — zero echo)
        preds = [m.predict_proba(Xnum[va])[:, 1] for m in fold_models]
        p_clean[va] = np.mean(preds, 0)
        models.append(fold_models)
        log(f"[S2A] p_clean fold{f} tr_auc={auc(ytr, p_clean[tr]):.5f} va_auc={auc(y[va], p_clean[va]):.5f}")
    return p_clean, models


def chi2_resid_score(codes, r, v, m0=15):
    """χ²_resid = Σ_g (Σ r)² / (Σ v + m0) over groups."""
    mx = int(codes.max()) + 1
    sr = np.bincount(codes, weights=r, minlength=mx)
    sv = np.bincount(codes, weights=v, minlength=mx)
    return float(np.sum((sr ** 2) / (sv + m0)))


def fisher_delta(codes, r, v, m0=15):
    mx = int(codes.max()) + 1
    n_g = np.bincount(codes, minlength=mx).astype(np.float64)
    sr = np.bincount(codes, weights=r, minlength=mx)
    sv = np.bincount(codes, weights=v, minlength=mx)
    mv = sv / np.maximum(n_g, 1.0)
    return sr / (n_g + m0 / (mv + 1e-9))


def apply_delta(codes, delta):
    mx = len(delta)
    codes = np.clip(codes, 0, mx - 1)
    return delta[codes]


def pillar_a_fisher(Tok_tr, Tok_te, names, y, folds, p_clean, Xnum_te, models):
    """Exhaustive 2D/3D Fisher Residual-TE → Ridge cube (zero echo)."""
    log("[S2A] Fisher Residual-TE cube ranking")
    n_tok = Tok_tr.shape[1]
    # precompute p_clean test via models
    p_te = np.zeros(N_TEST, float)
    for fmods in models:
        p_te += np.mean([m.predict_proba(Xnum_te)[:, 1] for m in fmods], 0)
    p_te /= max(len(models), 1)

    # residual + variance from p_clean (NOT z_fullbank)
    r_all = y - p_clean
    v_all = np.clip(p_clean * (1.0 - p_clean), 1e-6, None)

    # candidate keys: all 2-way among first 36 tokens + 250 domain 3-ways
    base_idx = list(range(min(36, n_tok)))
    pair_list = list(combinations(base_idx, 2))  # ≤630
    # domain 3-ways: SCAN6-ish + Alibay + flags combos
    name_to_i = {n: i for i, n in enumerate(names)}
    triples = []
    prefer = [
        ("channel", "device_type", "merchant_category"),
        ("auth_method", "card_type", "cvv_retry_count"),
        ("customer_age", "account_balance_usd__q250", "auth_method"),
        ("flag_bitmask_64", "auth_method", "channel"),
        ("SCAN6", "channel", "merchant_risk_score__q250"),
        ("ch_dev_mer", "auth_method", "cvv_retry_count"),
        ("amount_usd__cents", "auth_method", "channel"),
        ("velocity_score__q250", "channel", "device_type"),
    ]
    for a, b, c in prefer:
        if a in name_to_i and b in name_to_i and c in name_to_i:
            triples.append((name_to_i[a], name_to_i[b], name_to_i[c]))
    # fill to ~250 with hashed random triples from base
    rng = np.random.default_rng(7)
    while len(triples) < 250 and len(base_idx) >= 3:
        t = tuple(sorted(rng.choice(base_idx, 3, replace=False).tolist()))
        if t not in triples:
            triples.append(t)

    # score keys inside fold0 tr only for speed ranking (then refine top on all folds)
    tr0 = folds != 0
    r0, v0 = r_all[tr0], v_all[tr0]
    scored = []
    for i, j in pair_list:
        codes = hash_pair(Tok_tr[tr0, i], Tok_tr[tr0, j])
        # reindex to dense
        codes = pd.factorize(codes)[0].astype(np.int32)
        for m0 in (15, 60):
            sc = chi2_resid_score(codes, r0, v0, m0)
            scored.append(("2", (i, j), m0, sc))
    for i, j, k in triples:
        codes = hash3(Tok_tr[tr0, i], Tok_tr[tr0, j], Tok_tr[tr0, k])
        codes = pd.factorize(codes)[0].astype(np.int32)
        for m0 in (15, 60):
            sc = chi2_resid_score(codes, r0, v0, m0)
            scored.append(("3", (i, j, k), m0, sc))
    scored.sort(key=lambda z: -z[3])
    top = scored[:80]
    log(f"[S2A] top1 chi2={top[0][3]:.2f} key={top[0][:3]}")

    # build OOF + test TE matrix for top-80 with ZERO-ECHO: estimate delta on tr_f only
    n_keys = len(top)
    TE_o = np.zeros((N_TRAIN, n_keys), np.float32)
    TE_t = np.zeros((N_TEST, n_keys), np.float32)
    key_meta = []
    for ki, (kind, idxs, m0, sc) in enumerate(top):
        if kind == "2":
            i, j = idxs
            ctr = hash_pair(Tok_tr[:, i], Tok_tr[:, j])
            cte = hash_pair(Tok_te[:, i], Tok_te[:, j])
            label = f"2:{names[i]}×{names[j]}|m{m0}"
        else:
            i, j, k = idxs
            ctr = hash3(Tok_tr[:, i], Tok_tr[:, j], Tok_tr[:, k])
            cte = hash3(Tok_te[:, i], Tok_te[:, j], Tok_te[:, k])
            label = f"3:{names[i]}×{names[j]}×{names[k]}|m{m0}"
        # global factorize train∪test for stable codes
        lab = pd.factorize(np.concatenate([ctr, cte]))[0].astype(np.int32)
        ctr_c, cte_c = lab[:N_TRAIN], lab[N_TRAIN:]
        col_o = np.zeros(N_TRAIN, float)
        te_acc = []
        for f in range(5):
            tr, va = folds != f, folds == f
            # support filter on tr∪te levels
            mx = int(max(ctr_c.max(), cte_c.max())) + 1
            ok = (np.bincount(ctr_c[tr], minlength=mx) >= 8) & (np.bincount(cte_c, minlength=mx) >= 3)
            codes_tr = ctr_c[tr].copy()
            # map unsupported to 0-bucket with r~0
            delta = fisher_delta(codes_tr, r_all[tr], v_all[tr], m0)
            delta = np.where(ok[: len(delta)], delta, 0.0)
            if len(delta) < mx:
                delta = np.pad(delta, (0, mx - len(delta)))
            col_o[va] = apply_delta(ctr_c[va], delta)
            te_acc.append(apply_delta(cte_c, delta))
        TE_o[:, ki] = col_o.astype(np.float32)
        te_raw = np.mean(te_acc, 0)
        # KS screen vs OOF ECDF
        ks = float(ks_2samp(col_o, te_raw).statistic)
        if ks >= 0.012:
            te_raw = pchip_align(col_o, te_raw)
        TE_t[:, ki] = te_raw.astype(np.float32)
        key_meta.append({"key": label, "chi2": float(sc), "ks": ks})

    # Ridge condense alpha sweep — LOFO
    best_alpha, best_oof, best_te, best_solo = None, None, None, -1.0
    for alpha in (500.0, 2000.0, 5000.0):
        oof = np.zeros(N_TRAIN)
        te_parts = []
        for f in range(5):
            tr, va = folds != f, folds == f
            ridge = Ridge(alpha=alpha, fit_intercept=False)
            # predict residual → then use as score additive in L-space later
            ridge.fit(TE_o[tr], r_all[tr])
            oof[va] = ridge.predict(TE_o[va])
            te_parts.append(ridge.predict(TE_t))
        te = np.mean(te_parts, 0)
        # score as residual AUC proxy vs y via p_clean + oof
        solo = auc(y, p_clean + oof)
        if solo > best_solo:
            best_alpha, best_oof, best_te, best_solo = alpha, oof, te, solo
    log(f"[S2A] Ridge α={best_alpha} solo(p_clean+u)={best_solo:.5f}")
    del TE_o, TE_t
    gc.collect()
    return best_oof, best_te, key_meta[:20], {"alpha": best_alpha, "solo": best_solo}


def delta_gen_top(Tok_tr, names, y, folds, orig_df, tr_df):
    """Discrete generator divergence cells (shrunk by support)."""
    log("[S2A] Δ_gen discrete cells vs orig20k")
    if orig_df is None or len(orig_df) < 100:
        return []
    # map a few shared cats
    cells = []
    for c in CATS + ["cvv_retry_count"]:
        if c not in orig_df.columns or c not in tr_df.columns:
            continue
        # P(Y=1|v) on full train vs orig if is_fraud present
        y_orig = None
        if "is_fraud" in orig_df.columns:
            y_orig = orig_df["is_fraud"].to_numpy(float)
        tr_v = tr_df[c].astype(str)
        # fold0 tr only for speed
        tr = folds != 0
        sub = tr_df.loc[tr, [c]].copy()
        sub["_y"] = y[tr]
        for v, g in sub.groupby(c):
            n = len(g)
            if n < 30:
                continue
            p_tr = float(g["_y"].mean())
            if y_orig is not None:
                m = orig_df[c].astype(str) == str(v)
                n_o = int(m.sum())
                if n_o < 5:
                    continue
                p_o = float(y_orig[m].mean())
            else:
                continue
            w = n / (n + 50.0)
            d = w * (p_tr - p_o)
            cells.append(
                {"col": c, "v": str(v), "n_tr": int(n), "n_orig": n_o, "delta_gen": float(d), "p_tr": p_tr, "p_orig": p_o}
            )
    cells.sort(key=lambda z: -abs(z["delta_gen"]))
    return cells[:20]


# -------------------- Pillar B: Platt family fusion --------------------
def load_families(y, folds):
    """Re-admit vaults with L_fold / L hygiene; collapse to 6 macro families."""
    log("[S2B] prevalence-calibrated family fusion")
    fam_members = {f: [] for f in [
        "F1_Tree_GBDT",
        "F2_GPT2_BPE_LM",
        "F3_Compact_LM",
        "F4_EBM_GAM_Spline",
        "F5_TabM_MLP",
        "F6_TriTabPFN",
    ]}
    scorecard = []

    def try_add(family, name, po, pt):
        po, pt = np.asarray(po, float).ravel(), np.asarray(pt, float).ravel()
        if len(po) != N_TRAIN or len(pt) != N_TEST:
            return
        # L_fold / L FIRST — fixes raw-proba disp artifact
        po_L = expit(L_fold(po, folds))
        pt_L = expit(L(pt))
        solo = auc(y, po_L)
        # fold-diff vs fullbank
        zref = np.load(V36 / "z_fullbank_oof.npy").ravel()
        a = [auc(y[folds == k], po_L[folds == k]) for k in range(5)]
        b = [auc(y[folds == k], zref[folds == k]) for k in range(5)]
        fdc = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 else 0.0
        disp = float((L(pt_L) - L(np.load(V36 / "z_fullbank_test.npy").ravel())).std() / (
            (L(po_L) - L(zref)).std() + 1e-12
        ))
        ok = solo >= 0.9480 and fdc > 0.85
        scorecard.append({"name": name, "family": family, "solo": solo, "disp_after_L": disp, "fdc": fdc, "pass": ok})
        if ok:
            fam_members[family].append((po_L, pt_L))

    if AMAN.exists():
        npz = np.load(AMAN, allow_pickle=True)
        mapping = [
            ("F6_TriTabPFN", "U_TriTabPFN", "oof_P2_tri_tabpfn", "test_P2_tri_tabpfn"),
            ("F4_EBM_GAM_Spline", "U_GAM_P3", "oof_P3_gpdvae_gen", "test_P3_gpdvae_gen"),
            ("F5_TabM_MLP", "U_Deep_P4", "oof_P4_deep_neural_gam", "test_P4_deep_neural_gam"),
            ("F1_Tree_GBDT", "U_BPE_P5", "oof_P5_boosted_gbdt_v6", "test_P5_boosted_gbdt_v6"),
            ("F2_GPT2_BPE_LM", "U_GPT2_P1", "oof_P1_gpt2_5fold", "test_P1_gpt2_5fold"),
        ]
        for fam, nm, ok, tk in mapping:
            if ok in npz and tk in npz:
                try_add(fam, nm, npz[ok], npz[tk])

    if (BUSY / "oof.npy").exists():
        oof = np.load(BUSY / "oof.npy")
        te = np.load(BUSY / "test.npy")
        man = pd.read_csv(BUSY / "manifest.csv")
        names = man["name"].astype(str).tolist()
        fams = man["family"].astype(str).tolist() if "family" in man.columns else [""] * len(names)
        buckets = {
            "F1_Tree_GBDT": ["lgb", "xgb", "cat", "gbdt", "booster"],
            "F4_EBM_GAM_Spline": ["ebm", "gam", "glm", "probit"],
            "F5_TabM_MLP": ["tabm", "mlp", "nn"],
        }
        for fam, keys in buckets.items():
            ix = [i for i, (n, f) in enumerate(zip(names, fams)) if any(k in (n + f).lower() for k in keys)]
            if ix:
                Lm = np.column_stack([L(oof[:, i]) for i in ix])
                Lt = np.column_stack([L(te[:, i]) for i in ix])
                try_add(fam, f"busy_{fam}", expit(Lm.mean(1)), expit(Lt.mean(1)))
        if (BUSY / "tabm_oof.npy").exists():
            try_add("F5_TabM_MLP", "busy_tabm", np.load(BUSY / "tabm_oof.npy"), np.load(BUSY / "tabm_test.npy"))

    # L7 + gpt2
    L7 = PRED / "l7_s2026_5f"
    if (L7 / "l7_s2026_oof_5fold_raw.npy").exists():
        ll = np.load(L7 / "l7_s2026_oof_5fold_raw.npy")
        lt = np.load(L7 / "l7_s2026_test_5fold_raw.npy").mean(0)
        llr_o = (ll[:, :, 1] - ll[:, :, 0]).mean(1)
        llr_t = (lt[:, :, 1] - lt[:, :, 0]).mean(1)
        try_add("F3_Compact_LM", "L7_mean", U(llr_o), U(llr_t))

    GDIR = PRED / "gpt2_med_4ep"
    if all((GDIR / f"oof_{k}.npy").exists() for k in range(5)):
        po = np.zeros(N_TRAIN)
        tes = []
        for k in range(5):
            o = np.load(GDIR / f"oof_{k}.npy")
            t = np.load(GDIR / f"test_{k}.npy")
            # matched first-4 LLR mean
            po[folds == k] = (o[:, :4, 1] - o[:, :4, 0]).mean(1)
            tes.append((t[:, :4, 1] - t[:, :4, 0]).mean(1))
        try_add("F2_GPT2_BPE_LM", "gpt2_med_4ep", U(po), U(np.mean(tes, 0)))

    # B_clean columns → Tree family
    B_o = np.load(V36 / "B_clean_oof.npy")
    B_t = np.load(V36 / "B_clean_test.npy")
    for j in range(B_o.shape[1]):
        try_add("F1_Tree_GBDT", f"B{j}", B_o[:, j], B_t[:, j])

    # collapse families to mean L
    fam_o, fam_t = {}, {}
    for fam, mems in fam_members.items():
        if not mems:
            continue
        Lo = np.column_stack([L(m[0]) for m in mems])
        Lt = np.column_stack([L(m[1]) for m in mems])
        fam_o[fam] = expit(Lo.mean(1))
        fam_t[fam] = expit(Lt.mean(1))
        log(f"[S2B] {fam} n={len(mems)} solo={auc(y, fam_o[fam]):.5f}")
    return fam_o, fam_t, scorecard


def platt_family_fusion(fam_o, fam_t, y, folds, z_fb):
    """Non-negative simplex Logistic fusion + Jacobian contrast vs fullbank."""
    names = [k for k in sorted(fam_o.keys())]
    if len(names) < 2:
        return None, None, {"error": "need≥2 families"}
    oof = np.zeros(N_TRAIN)
    te_acc = []
    for f in range(5):
        tr, va = folds != f, folds == f
        # Platt each family on tr to prevalence 0.0167
        Xtr, Xva, Xte = [], [], []
        for nm in names:
            z = L(fam_o[nm]).reshape(-1, 1)
            zt = L(fam_t[nm]).reshape(-1, 1)
            lr = LogisticRegression(C=1e8, max_iter=300)
            lr.fit(z[tr], y[tr])
            # soft prevalence nudge via intercept shift
            p_tr = lr.predict_proba(z[tr])[:, 1]
            # shift logit so mean≈PREV on tr
            lo = logit(np.clip(p_tr, 1e-6, 1 - 1e-6))
            shift = logit(PREV) - lo.mean()
            Xtr.append(lo + shift)
            Xva.append(logit(np.clip(lr.predict_proba(z[va])[:, 1], 1e-6, 1 - 1e-6)) + shift)
            Xte.append(logit(np.clip(lr.predict_proba(zt)[:, 1], 1e-6, 1 - 1e-6)) + shift)
        Xtr = np.column_stack(Xtr)
        Xva = np.column_stack(Xva)
        Xte = np.column_stack(Xte)
        # non-negative logistic via Ridge on features + softmax-like positive weights
        # sklearn LogisticRegression positive=True (1.6+)
        try:
            clf = LogisticRegression(C=0.03, max_iter=500, positive=True, solver="lbfgs")
            clf.fit(Xtr, y[tr])
        except TypeError:
            clf = LogisticRegression(C=0.03, max_iter=500)
            clf.fit(Xtr, y[tr])
            # project coefs to non-neg + renormalize
            clf.coef_ = np.maximum(clf.coef_, 0.0)
        oof[va] = clf.predict_proba(Xva)[:, 1]
        te_acc.append(clf.predict_proba(Xte)[:, 1])
        # 2D EB disagreement surface Tree vs LM (optional additive)
        tree = [n for n in names if "Tree" in n or "TabM" in n or "EBM" in n or "Tri" in n]
        lm = [n for n in names if "LM" in n or "GPT" in n or "Compact" in n]
        # already in fusion; Jacobian extracted below
    te = np.mean(te_acc, 0)
    # Jacobian contrast
    u_o = unit(L_fold(oof, folds) - L_fold(z_fb, folds))
    u_t = unit(L(te) - L(np.load(V36 / "z_fullbank_test.npy").ravel()))
    log(f"[S2B] z_platt solo={auc(y, oof):.5f} families={names}")
    return u_o, u_t, {"families": names, "solo": auc(y, oof), "oof": oof, "te": te}


# -------------------- Pillar C: winsor residuals --------------------
def pillar_c_winsor(y, folds, z_fb, z_tw, z_fb_te, z_tw_te, B_o, B_t, ath_te):
    log("[S2C] winsor pure-residual preservation")
    out = {}
    paths = [
        ("u_p4_winsor", M2 / "harvest_p4_cuda_xgb_oof.npy", M2 / "harvest_p4_cuda_xgb_test.npy"),
        (
            "u_v10_winsor",
            PRED / "v10_transductive_bulsah/v10_ridge_resid_oof.npy",
            PRED / "v10_transductive_bulsah/v10_ridge_resid_test.npy",
        ),
    ]
    # ladder span on test
    ladder = [svd_basis(np.column_stack([unit(B_t[:, j]) for j in range(B_t.shape[1])]), min(25, B_t.shape[1]))]
    for p in [
        SUB / "sub_v35_S1_PUB_areeb_asterios_95547.csv",
        ATH_CSV,
    ]:
        if p.exists():
            s = pd.read_csv(p).iloc[:, 1].to_numpy(float)
            if len(s) == N_TEST:
                ladder.append(L(U(s)))
    ladder.append(L(U(ath_te)))

    for nm, o, t in paths:
        if not (o.exists() and t.exists()):
            continue
        uo = np.load(o).ravel().astype(float)
        ut = np.load(t).ravel().astype(float)
        # winsor ±4σ per fold BEFORE orth
        uo_w = np.zeros_like(uo)
        for f in range(5):
            m = folds == f
            uo_w[m] = winsor(uo[m], 4.0)
        ut_w = winsor(pchip_align(uo_w, ut), 4.0)
        # frozen LOFO orth vs [z_fb, z_tw, SVD_15(B)]
        u_o = np.zeros_like(uo_w)
        S15_o = svd_basis(np.column_stack([unit(B_o[:, j]) for j in range(B_o.shape[1])]), min(15, B_o.shape[1]))
        for f in range(5):
            tr, va = folds != f, folds == f
            Xtr = np.column_stack([unit(z_fb[tr]), unit(z_tw[tr]), S15_o[tr]])
            Xva = np.column_stack([unit(z_fb[va]), unit(z_tw[va]), S15_o[va]])
            ridge = Ridge(alpha=50.0, fit_intercept=False)
            ridge.fit(Xtr, unit(uo_w[tr]))
            u_o[va] = unit(uo_w[va]) - ridge.predict(Xva)
        u_o = winsor(u_o)
        Xo = np.column_stack([unit(z_fb), unit(z_tw), S15_o])
        S15_t = svd_basis(np.column_stack([unit(B_t[:, j]) for j in range(B_t.shape[1])]), min(15, B_t.shape[1]))
        Xt = np.column_stack([unit(z_fb_te), unit(z_tw_te), S15_t])
        ridge = Ridge(alpha=50.0, fit_intercept=False)
        ridge.fit(Xo, unit(uo_w))
        u_t = unit(ut_w) - ridge.predict(Xt)
        # project out Test Ladder Span
        u_t = gs_strip(u_t, ladder)
        u_t = winsor(unit(pchip_align(u_o, u_t)))
        # Δβ test-oof: compare |corr| with B before/after — require test not more aligned
        beta_o = max(abs(float(np.corrcoef(u_o, unit(B_o[:, j]))[0, 1])) for j in range(B_o.shape[1]))
        beta_t = max(abs(float(np.corrcoef(u_t, unit(B_t[:, j]))[0, 1])) for j in range(B_t.shape[1]))
        dbeta = beta_t - beta_o
        f2 = beta_t
        disp = float(u_t.std() / (u_o.std() + 1e-12))
        ks = float(ks_2samp(u_o, u_t).statistic)
        out[nm] = {
            "u_o": u_o,
            "u_t": u_t,
            "F2": f2,
            "disp": disp,
            "KS": ks,
            "dbeta": dbeta,
        }
        log(f"[S2C] {nm} F2={f2:.4f} disp={disp:.3f} KS={ks:.4f} dβ={dbeta:+.4f}")
    return out


# -------------------- LOFO + gates --------------------
def lofo_signed(u, y, folds, z_base):
    d_folds, w_folds = [], []
    for k in range(5):
        tr, va = folds != k, folds == k
        best_w, best_a = 0.0, -1.0
        zb_tr = L(z_base[tr])
        for w in W_GRID:
            a = auc(y[tr], zb_tr + w * u[tr])
            if a > best_a:
                best_a, best_w = a, float(w)
        d = auc(y[va], L(z_base[va]) + best_w * u[va]) - auc(y[va], z_base[va])
        d_folds.append(float(d))
        w_folds.append(best_w)
    d = np.asarray(d_folds)
    w_bar = float(np.mean(w_folds))
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    sub = []
    for k in range(5):
        m = folds == k
        yk, uk, zk = y[m], u[m], z_base[m]
        for _, va_sf in skf.split(np.zeros(len(yk)), yk):
            sub.append(auc(yk[va_sf], L(zk[va_sf]) + w_bar * uk[va_sf]) - auc(yk[va_sf], zk[va_sf]))
    sub = np.asarray(sub)
    t25 = float(sub.mean() / (sub.std() + 1e-12) * np.sqrt(len(sub))) if len(sub) else 0.0
    try:
        wp = float(wilcoxon(d, alternative="greater").pvalue)
    except Exception:
        wp = 1.0
    return {
        "mean_d": float(d.mean()),
        "d_folds": d_folds,
        "w_mean": w_bar,
        "n_pos_folds": int((d > 0).sum()),
        "min_fold": float(d.min()),
        "n_pos_sub": int((sub > 0).sum()),
        "t25": t25,
        "wilcoxon_p": wp,
        "sign": int(np.sign(w_bar)) if abs(w_bar) > 1e-12 else 0,
    }


def count_today_submits():
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / "Library/Python/3.14/bin") + ":" + env.get("PATH", "")
    r = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", "playground-series-s6e11", "-v"],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
    )
    lines = [ln for ln in (r.stdout or "").splitlines() if not ln.startswith("Warning:")]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n = sum(1 for ln in lines[1:] if today in ln and "Error" not in ln)
    return n, "\n".join(lines[:8])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("=== GM V38.1 START ===")
    y = pd.read_csv(REPO / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS_CSV)["fold"].to_numpy(int)
    assert len(y) == N_TRAIN

    z_fb = np.load(V36 / "z_fullbank_oof.npy").ravel()
    z_tw = np.load(V36 / "z_ATH_twin_oof.npy").ravel()
    z_fb_te = np.load(V36 / "z_fullbank_test.npy").ravel()
    z_tw_te = np.load(V36 / "z_ATH_twin_test.npy").ravel()
    B_o = np.load(V36 / "B_clean_oof.npy")
    B_t = np.load(V36 / "B_clean_test.npy")
    ath = pd.read_csv(ATH_CSV)
    ath_te = ath.iloc[:, 1].to_numpy(float)
    ids = ath["id"].to_numpy()

    # bank for Stage1: prefer larger matrix
    bank_path = PRED / "gm_v5_overnight/X_v5_oof.npy"
    if bank_path.exists():
        Bank = np.load(bank_path)
        log(f"[S1] using bank {bank_path.name} shape={Bank.shape}")
    else:
        # stitch busy + B_clean + aman if present
        cols = [B_o]
        if (BUSY / "oof.npy").exists():
            cols.append(np.load(BUSY / "oof.npy"))
        Bank = np.column_stack(cols)
        log(f"[S1] stitched bank shape={Bank.shape}")

    s1 = stage1_mc_gap(y, Bank)
    del Bank
    gc.collect()

    # numeric design for p_clean
    tr_df = pd.read_csv(REPO / "data/train.csv")
    te_df = pd.read_csv(REPO / "data/test.csv")
    # ensure is_fraud on tr
    num_cols = FLOATS + INTS
    Xnum = np.column_stack([unit(tr_df[c].to_numpy(float)) for c in num_cols]).astype(np.float32)
    Xnum_te = np.column_stack([unit(te_df[c].to_numpy(float)) for c in num_cols]).astype(np.float32)
    # add cat codes
    for c in CATS:
        lab = pd.factorize(pd.concat([tr_df[c].astype(str), te_df[c].astype(str)]))[0]
        Xnum = np.column_stack([Xnum, unit(lab[:N_TRAIN].astype(float))])
        Xnum_te = np.column_stack([Xnum_te, unit(lab[N_TRAIN:].astype(float))])

    Tok_tr, Tok_te, tok_names = build_token_dict(tr_df, te_df)
    p_clean, models = fit_p_clean_outer(Xnum, y, folds)

    u_f_o, u_f_t, top20, fmeta = pillar_a_fisher(Tok_tr, Tok_te, tok_names, y, folds, p_clean, Xnum_te, models)
    # treat fisher as residual
    u_fisher_o = winsor(u_f_o)
    u_fisher_t = winsor(unit(pchip_align(u_fisher_o, u_f_t)))

    orig = pd.read_csv(ORIG20K) if ORIG20K.exists() else None
    dgen = delta_gen_top(Tok_tr, tok_names, y, folds, orig, tr_df)

    fam_o, fam_t, fam_sc = load_families(y, folds)
    u_pl_o, u_pl_t, pl_meta = platt_family_fusion(fam_o, fam_t, y, folds, z_fb)
    if u_pl_o is None:
        u_pl_o = np.zeros(N_TRAIN)
        u_pl_t = np.zeros(N_TEST)

    c_out = pillar_c_winsor(y, folds, z_fb, z_tw, z_fb_te, z_tw_te, B_o, B_t, ath_te)

    # Stage 3: score pillars
    log("[S3] fine-grid LOFO synthesis")
    pillars = {
        "u_fisher_cube": (u_fisher_o, u_fisher_t, {"note": fmeta}),
    }
    if u_pl_o is not None and np.std(u_pl_o) > 1e-8:
        pillars["u_platt_jacobian"] = (u_pl_o, u_pl_t, pl_meta)
    if "u_p4_winsor" in c_out:
        pillars["u_p4_winsor"] = (c_out["u_p4_winsor"]["u_o"], c_out["u_p4_winsor"]["u_t"], c_out["u_p4_winsor"])
    if "u_v10_winsor" in c_out:
        pillars["u_v10_winsor"] = (c_out["u_v10_winsor"]["u_o"], c_out["u_v10_winsor"]["u_t"], c_out["u_v10_winsor"])

    table = []
    survivors = []
    for nm, (uo, ut, meta) in pillars.items():
        g_fb = lofo_signed(unit(uo), y, folds, z_fb)
        g_tw = lofo_signed(unit(uo), y, folds, z_tw)
        same = g_fb["sign"] != 0 and g_fb["sign"] == g_tw["sign"] and g_fb["mean_d"] > 0 and g_tw["mean_d"] > 0
        rec = {"name": nm, "g_fb": g_fb, "g_tw": g_tw, "same_sign": same, "meta": {k: v for k, v in (meta or {}).items() if k not in ("u_o", "u_t", "oof", "te")}}
        table.append(rec)
        log(
            f"[S3] {nm} same={same} fb={g_fb['mean_d']:+.2e} ({g_fb['n_pos_folds']}/5 t25={g_fb['t25']:.2f} p={g_fb['wilcoxon_p']:.4f}) "
            f"tw={g_tw['mean_d']:+.2e} ({g_tw['n_pos_folds']}/5 t25={g_tw['t25']:.2f})"
        )
        if same:
            survivors.append((nm, unit(uo), unit(ut), g_fb["w_mean"]))

    # collinearity filter
    kept = []
    for nm, uo, ut, w in survivors:
        if all(abs(float(np.corrcoef(uo, u2)[0, 1])) < 0.30 for _, u2, _, _ in kept):
            kept.append((nm, uo, ut, w))
    log(f"[S3] survivors={ [k[0] for k in kept] }")

    synth = None
    if kept:
        names = [k[0] for k in kept]
        Uo = np.column_stack([k[1] for k in kept])
        Ut = np.column_stack([k[2] for k in kept])
        w = np.array([k[3] for k in kept], float)
        # light coordinate ascent
        def score_w(wv, z):
            combo = Uo @ wv
            return float(np.mean([
                auc(y[folds == k], L(z[folds == k]) + combo[folds == k]) - auc(y[folds == k], z[folds == k])
                for k in range(5)
            ]))

        if len(w) >= 2:
            for _ in range(4):
                for j in range(len(w)):
                    best, best_val = w[j], score_w(w, z_fb)
                    for cand in W_GRID:
                        wt = w.copy()
                        wt[j] = cand
                        val = score_w(wt, z_fb)
                        if val > best_val:
                            best_val, best = val, cand
                    w[j] = best
        u_best_o = unit(Uo @ w) if len(w) > 1 else unit(Uo[:, 0])
        u_best_t = unit(Ut @ w) if len(w) > 1 else unit(Ut[:, 0])
        g_opt = lofo_signed(u_best_o, y, folds, z_fb)
        g_opt_tw = lofo_signed(u_best_o, y, folds, z_tw)
        # twilight gate check (smooth) — only if beats linear on ≥4/5 both
        q = U(z_fb)
        # linear already chosen; optional twilight soft scale
        chosen = "A_linear"
        synth = {
            "chosen": chosen,
            "names": names,
            "w_vec": w.tolist(),
            "w_opt_lofo": g_opt["w_mean"],
            "g_fb": g_opt,
            "g_tw": g_opt_tw,
            "u_o": u_best_o,
            "u_t": u_best_t,
        }
        log(f"[S3] Synth {chosen} w_opt={g_opt['w_mean']:+.5f} mean_d={g_opt['mean_d']:+.2e}")

    # Gates
    report = {"s1": s1, "top20_fisher": top20, "delta_gen": dgen, "family_scorecard": fam_sc, "lofo_table": table}
    gates = {}
    submitted = False
    pub = None
    cand_path = SUB / "sub_v38_1_BEST_0p85x.csv"

    if synth is None:
        gates["fail"] = "no_synth"
        report["gates"] = gates
        report["submitted"] = False
    else:
        u_o, u_t, w_opt = synth["u_o"], synth["u_t"], synth["w_opt_lofo"]
        g_fb, g_tw = synth["g_fb"], synth["g_tw"]
        # hygiene on u_t
        f2 = max(abs(float(np.corrcoef(u_t, unit(B_t[:, j]))[0, 1])) for j in range(B_t.shape[1]))
        disp = float(u_t.std() / (u_o.std() + 1e-12))
        ks = float(ks_2samp(u_o, u_t).statistic)
        dbeta = f2 - max(abs(float(np.corrcoef(u_o, unit(B_o[:, j]))[0, 1])) for j in range(B_o.shape[1]))

        g1 = (
            g_fb["n_pos_folds"] == 5
            and g_fb["min_fold"] > 0
            and g_fb["n_pos_sub"] >= 19
            and g_fb["t25"] >= 3.0
            and g_fb["wilcoxon_p"] < 0.005
        )
        g2 = g_tw["n_pos_folds"] >= 4 and g_tw["mean_d"] > 0 and g_tw["t25"] >= 2.2
        g3 = 0.97 <= disp <= 1.03 and ks < 0.008 and f2 < 0.015 and dbeta <= 0.01
        g4 = True  # anti-loser soft
        c = 0.85
        cand = U(L(ath_te) + c * w_opt * unit(u_t))
        step = 1.0 - float(spearmanr(cand, ath_te).correlation)
        min_hist = 1.0
        for p in sorted(SUB.glob("sub_v*.csv"), key=lambda x: -x.stat().st_mtime)[:60]:
            if p.name.startswith("._"):
                continue
            try:
                old = pd.read_csv(p)
                col = [c for c in old.columns if c != "id"][0]
                if len(old) != N_TEST:
                    continue
                min_hist = min(min_hist, abs(1.0 - float(spearmanr(cand, old[col].to_numpy(float)).correlation)))
            except Exception:
                continue
        g5 = 0.12e-4 <= step <= 15.0e-4 and min_hist > 0.05e-4 and np.isfinite(cand).all()
        n_today, _ = count_today_submits()
        g6 = n_today < 5

        gates = {
            "G1": {"pass": g1, **{k: g_fb[k] for k in ("mean_d", "n_pos_folds", "min_fold", "n_pos_sub", "t25", "wilcoxon_p")}},
            "G2": {"pass": g2, **{k: g_tw[k] for k in ("mean_d", "n_pos_folds", "t25")}},
            "G3": {"pass": g3, "disp": disp, "KS": ks, "F2": f2, "dbeta": dbeta},
            "G4": {"pass": g4},
            "G5": {"pass": g5, "step": step, "min_hist": min_hist},
            "G6": {"pass": g6, "n_today": n_today},
        }
        all_pass = all(gates[k]["pass"] for k in gates)
        SUB.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(cand_path, index=False)
        sha = hashlib.sha256(cand_path.read_bytes()).hexdigest()
        report["candidate"] = {"path": str(cand_path), "sha256": sha, "step": step, "w_opt": w_opt, "c": c}
        report["gates"] = gates
        report["synth"] = {k: v for k, v in synth.items() if k not in ("u_o", "u_t")}

        if not all_pass:
            report["submitted"] = False
            report["submit_block"] = {k: v for k, v in gates.items() if not v["pass"]}
            log(f"[S4] GATES FAILED — 0 submits. failed={list(report['submit_block'].keys())}")
        else:
            env = os.environ.copy()
            env["PATH"] = str(Path.home() / "Library/Python/3.14/bin") + ":" + env.get("PATH", "")
            cmd = [
                "kaggle",
                "competitions",
                "submit",
                "-c",
                "playground-series-s6e11",
                "-f",
                str(cand_path),
                "-m",
                "v38.1 fisher+platt 0.85x dual-baseline LOFO",
            ]
            log(f"[S4] SUBMITTING {cand_path}")
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=90)
            out = "\n".join(
                ln for ln in ((r.stdout or "") + (r.stderr or "")).splitlines() if not ln.startswith("Warning:")
            )
            submitted = r.returncode == 0
            report["submit_stdout"] = out[-1500:]
            report["submitted"] = submitted
            if submitted:
                t_end = time.time() + 12 * 60
                while time.time() < t_end:
                    time.sleep(25)
                    rr = subprocess.run(
                        ["kaggle", "competitions", "submissions", "-c", "playground-series-s6e11", "-v"],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=90,
                    )
                    lines = [ln for ln in (rr.stdout or "").splitlines() if not ln.startswith("Warning:")]
                    for ln in lines[1:6]:
                        if "v38_1" in ln or "v38.1" in ln or "0p85x" in ln or "COMPLETE" in ln.upper():
                            if "COMPLETE" in ln.upper():
                                report["live_line"] = ln
                                for tok in ln.replace(",", " ").split():
                                    try:
                                        val = float(tok)
                                        if 0.9 < val < 1.0:
                                            pub = val
                                    except Exception:
                                        pass
                    if pub is not None:
                        break
                report["publicScore"] = pub
                report["delta_vs_ATH"] = (pub - ATH_LB) if pub is not None else None
            log(f"[S4] submitted={submitted} publicScore={pub}")

    # Report
    lines = [
        "# GM V38.1 Report",
        "",
        f"**UTC:** {datetime.now(timezone.utc).isoformat()}",
        f"**Elapsed:** {time.time() - T0:.1f}s",
        f"**Submitted:** {report.get('submitted')} · publicScore={report.get('publicScore')}",
        "",
        "## 1. 60k/240k #1 Monte-Carlo Verdict",
        "",
        f"```json\n{json.dumps(s1, indent=2)}\n```",
        "",
        f"**Math proof:** {s1.get('math_proof')}",
        "",
        "## 2. Top-20 Fisher Residual-TE & Δ_gen",
        "",
        "### Fisher keys",
        "",
    ]
    for row in top20:
        lines.append(f"- `{row['key']}` chi2={row['chi2']:.2f} ks={row['ks']:.4f}")
    lines += ["", "### Δ_gen cells", ""]
    for row in dgen:
        lines.append(
            f"- `{row['col']}={row['v']}` Δ={row['delta_gen']:+.5f} (p_tr={row['p_tr']:.4f} p_orig={row['p_orig']:.4f} n={row['n_tr']}/{row['n_orig']})"
        )
    lines += ["", "## 3. Pillar LOFO Table", "", "| name | g_fb Δ | folds | sub | t25 | p | g_tw Δ | folds_tw | t25_tw | same |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in table:
        g, t = r["g_fb"], r["g_tw"]
        lines.append(
            f"| {r['name']} | {g['mean_d']:+.2e} | {g['n_pos_folds']}/5 | {g['n_pos_sub']}/25 | {g['t25']:.2f} | {g['wilcoxon_p']:.4f} | "
            f"{t['mean_d']:+.2e} | {t['n_pos_folds']}/5 | {t['t25']:.2f} | {r['same_sign']} |"
        )
    lines += ["", "## 4. Live 1-Submit / Hold Telemetry", "", f"```json\n{json.dumps({k: report.get(k) for k in ('gates', 'candidate', 'submitted', 'publicScore', 'delta_vs_ATH', 'submit_block')}, indent=2, default=str)}\n```", ""]
    if report.get("submitted") and report.get("publicScore") is not None:
        d = report["delta_vs_ATH"]
        lines.append(f"- Live {report['publicScore']} vs ATH 0.95548 → Δ={d:+.5f}")
    else:
        lines.append("- **0 submits.** Hold all 5 slots.")
        for k, v in (report.get("submit_block") or report.get("gates") or {}).items():
            if isinstance(v, dict) and not v.get("pass", True):
                lines.append(f"  - `{k}`: {v}")

    path = OUT / "GM_V38_1_REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    (OUT / "GM_V38_1_REPORT.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"[S5] wrote {path}")
    print("\n".join(lines[:60]), flush=True)
    log("=== GM V38.1 DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
