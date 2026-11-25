#!/usr/bin/env python3
"""V43.2 — Local parallel 4-engine discovery suite.

$0 Mac CPU/MPS only. NO GPU pods. STRICT NO_SUBMIT=True.
float32 · n_jobs=4 · baseline z_modern_pub+Zr = 0.952652
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")

NO_SUBMIT = True
assert NO_SUBMIT
N_JOBS = 4
T0 = time.time()
N_TR, N_TE = 699_960, 299_984
ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
SSD = Path("/tmp/s6e11_v39_ssd")
DATA = ROOT / "data" if (ROOT / "data/train.csv").exists() else SSD / "data"
ORIG_CANDS = [
    DATA / "orig20k.csv",
    SSD / "data/orig20k.csv",
    ROOT / "external/v37_twin_check/orig20k_uditjain.csv",
]
LOG = ROOT / "logs/v43"
OUT = ROOT / "out/v43_2_local_engines"
PRED = ROOT / "preds"
MOD = ROOT / "out/v42_1_modern"
FINAL = ROOT / "out/v41_final_10slots"
GPW = ROOT / "external/banks/goodpjw2008_s6e11-oof-bank"
AMAN = ROOT / "external/banks_v31/s6e11-rank1-q17-meta-bundle/rank1_q17_streams.npz"
V43T = ROOT / "preds/v43_order_tensor"

BLACKLIST = (
    "uid_catboost", "v41_d27_eb_zone", "fintfm", "fin_tfm",
    "zanch", "v10_", "gb_soft_bag3", "idf_p3", "clean035",
)

FEATS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
DISC = {
    "txn_count_last_24h", "card_age_months", "customer_age", "cvv_retry_count",
    "time_of_day_hour", "day_of_week", "prior_disputes",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
}
CONT = [c for c in FEATS if c not in DISC]
FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]


def log(m: str):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def U(v):
    v = np.asarray(v, np.float64).ravel()
    return ((rankdata(v, method="average") - 0.5) / len(v)).astype(np.float64)


def L(v):
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12)).astype(np.float64)


def L_fold(v, folds):
    v = np.asarray(v, np.float64).ravel()
    out = np.zeros_like(v)
    for k in np.unique(folds):
        m = folds == k
        out[m] = L(v[m])
    return out


def unit(v):
    v = np.asarray(v, np.float64).ravel()
    return ((v - v.mean()) / (v.std() + 1e-12)).astype(np.float64)


def is_black(name: str) -> bool:
    s = name.lower()
    return any(b in s for b in BLACKLIST)


def z_goodpjw(df: pd.DataFrame) -> np.ndarray:
    clean = (df["cvv_retry_count"].to_numpy() == 0) & (df["prior_disputes"].to_numpy() == 0)
    for f in FLAGS:
        clean &= df[f].to_numpy().astype(int) == 0
    risk, vel = df["merchant_risk_score"].to_numpy(), df["velocity_score"].to_numpy()
    return (
        (risk < 13.1)
        | ((vel <= 4.1) & (risk <= 20.0))
        | (clean & (((vel <= 4.1) & (risk <= 33.8)) | ((vel <= 6.0) & (risk <= 25.0))))
    )


def apply_zr(scores, zones, u_ref):
    out = L(scores).copy()
    uref = U(u_ref)
    if zones.any() and (~zones).any():
        floor = float(np.min(out[~zones])) - 1.0
        out[zones] = floor + uref[zones]
    return U(out)


def gs_orth(v, bases):
    v = unit(v)
    for b in bases:
        b = unit(b)
        v = unit(v - np.dot(v, b) * b)
    return v


def fold_metrics(y, p_new, p_base, folds):
    d = []
    for k in range(5):
        m = folds == k
        d.append((roc_auc_score(y[m], p_new[m]) - roc_auc_score(y[m], p_base[m])) * 1e5)
    d = np.asarray(d, float)
    return {
        "mean_e5": float(d.mean()),
        "pos": int((d > 0).sum()),
        "min_e5": float(d.min()),
        "folds": d.tolist(),
        "t25": float(d.mean() / (d.std(ddof=1) / np.sqrt(5) + 1e-12)),
    }


def sf25(y, a, b, folds):
    wins = 0
    deltas = []
    for k in range(5):
        m = folds == k
        yk, ak, bk = y[m], a[m], b[m]
        skf = StratifiedKFold(5, shuffle=True, random_state=42 + k)
        for _, va in skf.split(np.arange(len(yk)), yk):
            da = (roc_auc_score(yk[va], ak[va]) - roc_auc_score(yk[va], bk[va])) * 1e5
            deltas.append(da)
            wins += int(da > 0)
    d25 = np.asarray(deltas, float)
    try:
        _, wp = wilcoxon(d25, alternative="greater")
        wp = float(wp)
    except Exception:
        wp = float("nan")
    return wins, float(d25.mean() / (d25.std(ddof=1) / np.sqrt(25) + 1e-12)), wp


def cos_full(a, b):
    a, b = unit(a), unit(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def flag_bitmask64(df: pd.DataFrame) -> np.ndarray:
    bits = np.zeros(len(df), dtype=np.int64)
    for i, f in enumerate(FLAGS):
        bits |= (df[f].to_numpy().astype(np.int64) & 1) << i
    # pack a few more discrete into higher bits
    bits |= (df["cvv_retry_count"].to_numpy().astype(np.int64).clip(0, 7)) << 8
    bits |= (df["prior_disputes"].to_numpy().astype(np.int64).clip(0, 7)) << 11
    return bits


# ── tokenization ──────────────────────────────────────────────
def build_tokens(tr, te, orig):
    """Return int token matrices (n, n_tok) for tr/te/orig + names."""
    frames = {"tr": tr, "te": te, "orig": orig}
    toks = {k: [] for k in frames}
    names = []

    # discrete exact
    for c in FEATS:
        if c in DISC or tr[c].dtype == object:
            cats = pd.Categorical(pd.concat([tr[c].astype(str), te[c].astype(str), orig[c].astype(str)]))
            n_tr, n_te = len(tr), len(te)
            codes = cats.codes.astype(np.int32)
            toks["tr"].append(codes[:n_tr])
            toks["te"].append(codes[n_tr:n_tr + n_te])
            toks["orig"].append(codes[n_tr + n_te:])
            names.append(f"disc:{c}")

    # continuous sub-float
    for c in CONT:
        for key, df in frames.items():
            x = df[c].to_numpy(np.float64)
            if key == "tr":
                pass
        x_all = pd.concat([tr[c], te[c], orig[c]]).to_numpy(np.float64)
        n_tr, n_te = len(tr), len(te)
        ip = np.floor(x_all).astype(np.int32)
        cents = np.round((x_all % 1.0) * 100).astype(np.int32).clip(0, 100)
        # 2-sig mantissa bucket
        ax = np.abs(x_all) + 1e-12
        mant = (ax / (10.0 ** np.floor(np.log10(ax))) * 10).astype(np.int32).clip(0, 99)
        slen = np.array([len(f"{v:.8g}") for v in x_all], dtype=np.int32)
        # 200-quantile on train∪test only
        qt = QuantileTransformer(n_quantiles=200, output_distribution="uniform", subsample=200_000, random_state=42)
        qt.fit(np.concatenate([tr[c].to_numpy(), te[c].to_numpy()]).reshape(-1, 1))
        qbin = (qt.transform(x_all.reshape(-1, 1)).ravel() * 199).astype(np.int32).clip(0, 199)
        for arr, nm in ((ip, f"int:{c}"), (cents, f"cents:{c}"), (mant, f"mant:{c}"), (slen, f"slen:{c}"), (qbin, f"q200:{c}")):
            toks["tr"].append(arr[:n_tr])
            toks["te"].append(arr[n_tr:n_tr + n_te])
            toks["orig"].append(arr[n_tr + n_te:])
            names.append(nm)

    # flag bitmask as token
    for key, df in frames.items():
        toks[key].append(flag_bitmask64(df).astype(np.int32))
    names.append("flag_bitmask64")

    out = {k: np.column_stack(v).astype(np.int32) for k, v in toks.items()}
    return out, names


def overlap_feats(X, seeds, seed_y, names):
    """Overlap vs Pos_339 and Neg contrast IDF scores. X:(N,C) seeds:(S,C)."""
    pos = seeds[seed_y == 1]
    neg = seeds[seed_y == 0]
    N, C = X.shape
    Spos = len(pos)
    # column frequencies on seeds for IDF
    # for each col, value freq among pos/neg
    max_match = np.zeros(N, np.float32)
    top3 = np.zeros((N, 3), np.float32)
    # process seeds in chunks
    match_ge = {6: np.zeros(N, np.float32), 8: np.zeros(N, np.float32), 10: np.zeros(N, np.float32)}
    idf_pos = np.zeros(N, np.float64)
    idf_neg = np.zeros(N, np.float64)

    # Precompute IDF weight per (col, value) from full seed bank
    # weight = log(1 + S / freq) for matches against pos vs neg separately
    def idf_table(Smat):
        tables = []
        S = len(Smat)
        for c in range(C):
            vals, cnts = np.unique(Smat[:, c], return_counts=True)
            # map val->log(1+S/cnt)
            w = np.log1p(S / cnts.astype(np.float64))
            tables.append((vals, w))
        return tables

    pos_idf = idf_table(pos)
    neg_idf = idf_table(neg)

    # For each row, compute matches to all pos seeds — chunked
    chunk = 50
    running_top = np.full((N, 3), -1.0, np.float32)
    for s0 in range(0, Spos, chunk):
        block = pos[s0:s0 + chunk]  # (b, C)
        b = len(block)
        # matches (N, b): sum over C of equality
        m = np.zeros((N, b), np.float32)
        for c in range(C):
            m += (X[:, c:c + 1] == block[:, c].reshape(1, -1)).astype(np.float32)
        max_match = np.maximum(max_match, m.max(1))
        for t in (6, 8, 10):
            match_ge[t] += (m >= t).sum(1).astype(np.float32)
        # update top3
        stacked = np.concatenate([running_top, m], 1)
        running_top = np.partition(stacked, -3, axis=1)[:, -3:]
        # IDF-weighted collision vs this block
        for bi in range(b):
            wsum = np.zeros(N, np.float64)
            for c in range(C):
                vals, w = pos_idf[c]
                # match mask
                eq = X[:, c] == block[bi, c]
                # weight for this value
                ix = np.searchsorted(vals, block[bi, c])
                if ix < len(vals) and vals[ix] == block[bi, c]:
                    wsum[eq] += w[ix]
            idf_pos += wsum

    top3_mean = running_top.mean(1).astype(np.float32)

    # Neg contrast: sample 800 negs for speed
    rng = np.random.default_rng(42)
    neg_s = neg[rng.choice(len(neg), size=min(800, len(neg)), replace=False)]
    for s0 in range(0, len(neg_s), chunk):
        block = neg_s[s0:s0 + chunk]
        b = len(block)
        for bi in range(b):
            wsum = np.zeros(N, np.float64)
            for c in range(C):
                vals, w = neg_idf[c]
                eq = X[:, c] == block[bi, c]
                ix = np.searchsorted(vals, block[bi, c])
                if ix < len(vals) and vals[ix] == block[bi, c]:
                    wsum[eq] += w[ix]
            idf_neg += wsum

    feats = np.column_stack([
        max_match,
        top3_mean,
        match_ge[6], match_ge[8], match_ge[10],
        idf_pos.astype(np.float32),
        (idf_pos - idf_neg).astype(np.float32),
        (idf_pos / (idf_neg + 1.0)).astype(np.float32),
    ])
    return feats


def gen_shift_feats(tr, te, orig, y_orig, folds, y_tr):
    """1D + cube Δ_gen log-odds shifts, LOFO on train."""
    cols_1d = [
        "cvv_retry_count", "prior_disputes", "merchant_category", "card_type",
        "auth_method", "channel", "device_type", "time_of_day_hour",
        "is_foreign_transaction", "is_new_merchant", "used_vpn",
        "ip_country_mismatch", "billing_shipping_mismatch",
    ]
    # q40 bins for risk/vel
    for c, name in (("merchant_risk_score", "risk_q40"), ("velocity_score", "vel_q40")):
        edges = np.quantile(tr[c].to_numpy(), np.linspace(0, 1, 41))
        edges[0], edges[-1] = -np.inf, np.inf
        tr[name] = np.digitize(tr[c], edges[1:-1])
        te[name] = np.digitize(te[c], edges[1:-1])
        orig[name] = np.digitize(orig[c], edges[1:-1])
        cols_1d.append(name)

    def smoothed_logit(y, key, m=20.0):
        # map value -> logit
        s = pd.Series(y).groupby(key).agg(["sum", "count"])
        p = (s["sum"] + m * 0.0167) / (s["count"] + m)
        return logit(np.clip(p.to_numpy(), 1e-6, 1 - 1e-6)), s.index.to_numpy()

    n_tr, n_te = len(tr), len(te)
    n_f = len(cols_1d) + 2  # +2 cubes
    Fo = np.zeros((n_tr, n_f), np.float32)
    Ft = np.zeros((n_te, n_f), np.float32)

    for j, c in enumerate(cols_1d):
        # orig fixed
        lo, keys = smoothed_logit(y_orig, orig[c])
        orig_map = dict(zip(keys.tolist(), lo.tolist()))
        for f in range(5):
            tr_m = folds != f
            lo_tr, keys_tr = smoothed_logit(y_tr[tr_m], tr.loc[tr_m, c])
            tr_map = dict(zip(keys_tr.tolist(), lo_tr.tolist()))
            va = folds == f
            keys_va = tr.loc[va, c]
            Fo[va, j] = np.array([tr_map.get(k, 0.0) - orig_map.get(k, 0.0) for k in keys_va], np.float32)
        # test: mean over folds' maps
        acc = np.zeros(n_te, np.float64)
        for f in range(5):
            tr_m = folds != f
            lo_tr, keys_tr = smoothed_logit(y_tr[tr_m], tr.loc[tr_m, c])
            tr_map = dict(zip(keys_tr.tolist(), lo_tr.tolist()))
            acc += np.array([tr_map.get(k, 0.0) - orig_map.get(k, 0.0) for k in te[c]], np.float64)
        Ft[:, j] = (acc / 5).astype(np.float32)

    # cubes
    tr["_flag"] = flag_bitmask64(tr)
    te["_flag"] = flag_bitmask64(te)
    orig["_flag"] = flag_bitmask64(orig)
    cubes = [
        ("channel", "device_type", "merchant_category"),
        ("auth_method", "channel", "_flag"),
    ]
    for ci, cols in enumerate(cubes):
        j = len(cols_1d) + ci
        key_o = orig[list(cols)].astype(str).agg("|".join, axis=1)
        lo, keys = smoothed_logit(y_orig, key_o)
        orig_map = dict(zip(keys.tolist(), lo.tolist()))
        for f in range(5):
            tr_m = folds != f
            key_tr = tr.loc[tr_m, list(cols)].astype(str).agg("|".join, axis=1)
            lo_tr, keys_tr = smoothed_logit(y_tr[tr_m], key_tr)
            tr_map = dict(zip(keys_tr.tolist(), lo_tr.tolist()))
            va = folds == f
            key_va = tr.loc[va, list(cols)].astype(str).agg("|".join, axis=1)
            Fo[va, j] = np.array([tr_map.get(k, 0.0) - orig_map.get(k, 0.0) for k in key_va], np.float32)
        acc = np.zeros(n_te, np.float64)
        key_te = te[list(cols)].astype(str).agg("|".join, axis=1)
        for f in range(5):
            tr_m = folds != f
            key_tr = tr.loc[tr_m, list(cols)].astype(str).agg("|".join, axis=1)
            lo_tr, keys_tr = smoothed_logit(y_tr[tr_m], key_tr)
            tr_map = dict(zip(keys_tr.tolist(), lo_tr.tolist()))
            acc += np.array([tr_map.get(k, 0.0) - orig_map.get(k, 0.0) for k in key_te], np.float64)
        Ft[:, j] = (acc / 5).astype(np.float32)

    return Fo, Ft


def newton_booster(y, folds, z_base, Xo, Xt, name):
    """Newton residual Ridge+LGB on features; return oof/test scores."""
    oof = np.zeros(N_TR, np.float64)
    te = np.zeros(N_TE, np.float64)
    for f in range(5):
        tr, va = folds != f, folds == f
        z_tr = L(z_base[tr])
        # calibrate a,b: p = σ(a*z + b)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        lr.fit(z_tr.reshape(-1, 1), y[tr])
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        p_tr = expit(a * z_tr + b)
        g = y[tr] - p_tr
        h = p_tr * (1 - p_tr)
        y_tgt = g / (h + 1e-4)
        # Ridge
        ridge = Ridge(alpha=1000.0)
        ridge.fit(Xo[tr], y_tgt, sample_weight=h)
        pred_r = ridge.predict(Xo[va])
        pred_t = ridge.predict(Xt)
        # LGB 3 seeds
        preds_va, preds_te = [], []
        for seed in (0, 1, 2):
            dtr = lgb.Dataset(Xo[tr], label=y_tgt, weight=h)
            params = dict(
                objective="regression", metric="l2", learning_rate=0.05,
                max_depth=4, num_leaves=11, min_child_samples=300,
                feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                verbosity=-1, seed=42 + seed + f,
            )
            bst = lgb.train(params, dtr, num_boost_round=200)
            preds_va.append(bst.predict(Xo[va]))
            preds_te.append(bst.predict(Xt))
        pred_l = np.mean(preds_va, 0)
        pred_lt = np.mean(preds_te, 0)
        oof[va] = 0.4 * pred_r + 0.6 * pred_l
        te += (0.4 * pred_t + 0.6 * pred_lt) / 5
        log(f"  {name} fold {f} done")
    return oof.astype(np.float32), te.astype(np.float32)


# ── Engine 2 helpers ─────────────────────────────────────────
def epistatic_block(y, folds, z_base, tr, te, pair_specs, m_list=(15, 60)):
    """Build Δη features for listed (colA, colB) pairs with support gate."""
    n_tr, n_te = len(tr), len(te)
    cols_out = []
    Fo_list, Ft_list = [], []

    # prep encodings
    def enc(df, c):
        if c == "_flag":
            return flag_bitmask64(df).astype(object)
        if c.endswith("_q40"):
            return df[c].to_numpy()
        return df[c].astype(str).to_numpy()

    # ensure q40
    for c, name in (("merchant_risk_score", "risk_q40"), ("velocity_score", "vel_q40")):
        if name not in tr.columns:
            edges = np.quantile(tr[c].to_numpy(), np.linspace(0, 1, 41))
            tr[name] = np.digitize(tr[c], edges[1:-1])
            te[name] = np.digitize(te[c], edges[1:-1])

    for (ca, cb) in pair_specs:
        for m in m_list:
            fo = np.zeros(n_tr, np.float32)
            ft = np.zeros(n_te, np.float32)
            for f in range(5):
                tr_m, va = folds != f, folds == f
                # 1D logits on tr
                def logit_map(keys, yy, m_):
                    s = pd.Series(yy).groupby(keys).agg(["sum", "count"])
                    p = (s["sum"] + m_ * 0.0167) / (s["count"] + m_)
                    return dict(zip(s.index.tolist(), logit(np.clip(p, 1e-6, 1 - 1e-6)).tolist())), s["count"].to_dict()

                Ka = enc(tr.loc[tr_m], ca)
                Kb = enc(tr.loc[tr_m], cb)
                map_a, cnt_a = logit_map(Ka, y[tr_m], m)
                map_b, cnt_b = logit_map(Kb, y[tr_m], m)
                p0 = float(y[tr_m].mean())
                l0 = logit(np.clip(p0, 1e-6, 1 - 1e-6))
                # pair keys
                Ka_s = pd.Series(Ka).astype(str)
                Kb_s = pd.Series(Kb).astype(str)
                pair = Ka_s + "|" + Kb_s
                # Δη
                eta = np.array([map_a.get(a, l0) + map_b.get(b, l0) - l0 for a, b in zip(Ka, Kb)])
                resid = y[tr_m] - expit(eta)
                s = pd.Series(resid).groupby(pair.to_numpy()).agg(["sum", "count"])
                cnt_tr = s["count"].to_dict()
                # test counts
                Ka_te = pd.Series(enc(te, ca)).astype(str)
                Kb_te = pd.Series(enc(te, cb)).astype(str)
                pair_te = (Ka_te + "|" + Kb_te).to_numpy()
                cnt_te = pd.Series(pair_te).value_counts().to_dict()
                d_eta = {}
                for k, row in s.iterrows():
                    if cnt_tr.get(k, 0) >= 10 and cnt_te.get(k, 0) >= 4:
                        d_eta[k] = float(row["sum"] / (row["count"] + m))
                    else:
                        d_eta[k] = 0.0
                # apply to va
                Ka_va = pd.Series(enc(tr.loc[va], ca)).astype(str)
                Kb_va = pd.Series(enc(tr.loc[va], cb)).astype(str)
                pair_va = (Ka_va + "|" + Kb_va).to_numpy()
                fo[va] = np.array([d_eta.get(k, 0.0) for k in pair_va], np.float32)
                ft += np.array([d_eta.get(k, 0.0) for k in pair_te], np.float32) / 5
            Fo_list.append(fo)
            Ft_list.append(ft)
            cols_out.append(f"{ca}x{cb}_m{m}")
    if not Fo_list:
        return np.zeros((n_tr, 1), np.float32), np.zeros((n_te, 1), np.float32)
    return np.column_stack(Fo_list), np.column_stack(Ft_list)


# ── Engine 3 AUC gradient ────────────────────────────────────
def auc_gradient_binned(z, y, tau, n_bins=512):
    """O(N log N) approx Wilcoxon gradient via 1D bins."""
    z = np.asarray(z, np.float64)
    y = np.asarray(y, int)
    # rank-based bins
    r = rankdata(z, method="average")
    bins = np.clip((r / len(z) * n_bins).astype(int), 0, n_bins - 1)
    pos_w = np.zeros(n_bins)
    neg_w = np.zeros(n_bins)
    for b in range(n_bins):
        m = bins == b
        pos_w[b] = (y[m] == 1).sum()
        neg_w[b] = (y[m] == 0).sum()
    # center z per bin
    zc = np.zeros(n_bins)
    for b in range(n_bins):
        m = bins == b
        if m.any():
            zc[b] = z[m].mean()
    g = np.zeros(len(z))
    # for each bin, compute mean σ((zj-zi)/τ) over opposite class bins
    for b in range(n_bins):
        m = bins == b
        if not m.any():
            continue
        # positives in b: +mean_neg σ((z_neg - z_b)/τ)
        if pos_w[b] > 0:
            # weighted over neg bins
            s = 0.0
            wtot = neg_w.sum() + 1e-12
            for bn in range(n_bins):
                if neg_w[bn] == 0:
                    continue
                s += neg_w[bn] * expit((zc[bn] - zc[b]) / tau)
            g[m & (y == 1)] = s / wtot
        if neg_w[b] > 0:
            s = 0.0
            wtot = pos_w.sum() + 1e-12
            for bp in range(n_bins):
                if pos_w[bp] == 0:
                    continue
                s += pos_w[bp] * expit((zc[b] - zc[bp]) / tau)
            g[m & (y == 0)] = -s / wtot
    return g


def build_x_auc(tr, te, folds):
    """~55 cols for Engine 3."""
    cols = []
    # raw numerics
    for c in FEATS:
        if c in DISC or tr[c].dtype == object:
            cats = pd.Categorical(pd.concat([tr[c].astype(str), te[c].astype(str)]))
            codes = cats.codes.astype(np.float32)
            cols.append(("tr", codes[:len(tr)]))
            cols.append(("te", codes[len(tr):]))
        else:
            cols.append(("tr", tr[c].to_numpy(np.float32)))
            cols.append(("te", te[c].to_numpy(np.float32)))
    # regroup
    # fix: rebuild properly
    Xo_list, Xt_list = [], []
    for c in FEATS:
        if c in DISC or tr[c].dtype == object:
            cats = pd.Categorical(pd.concat([tr[c].astype(str), te[c].astype(str)]))
            codes = cats.codes.astype(np.float32)
            Xo_list.append(codes[:len(tr)])
            Xt_list.append(codes[len(tr):])
        else:
            x = tr[c].to_numpy(np.float64)
            Xo_list.append(x.astype(np.float32))
            Xt_list.append(te[c].to_numpy(np.float32))
            # cents
            Xo_list.append(np.round((x % 1) * 100).astype(np.float32))
            Xt_list.append(np.round((te[c].to_numpy(np.float64) % 1) * 100).astype(np.float32))

    # recovered
    for nm in ("l7_R5_antithetic_absdiff", "U5_gini10_llr"):
        po, pt = V43T / f"{nm}_oof.npy", V43T / f"{nm}_test.npy"
        if not po.exists():
            po, pt = V43T / f"l7_{nm}_oof.npy", V43T / f"l7_{nm}_test.npy"
        if po.exists():
            Xo_list.append(np.load(po).ravel().astype(np.float32))
            Xt_list.append(np.load(pt).ravel().astype(np.float32))
    for nm in ("P4_L4.3_pure5f", "P4_L4.5_pure5f"):
        po, pt = PRED / f"matrix_v2/{nm}_oof.npy", PRED / f"matrix_v2/{nm}_test.npy"
        if po.exists():
            Xo_list.append(np.load(po).ravel().astype(np.float32))
            Xt_list.append(np.load(pt).ravel().astype(np.float32))

    # pillar contrasts from gpt2 / l7 / realmlp proxies
    def load_gpt2(folds_):
        gdir = PRED / "gpt2_med_4ep"
        go = np.zeros(N_TR)
        gt = []
        for k in range(5):
            a = np.load(gdir / f"oof_{k}.npy")
            b = np.load(gdir / f"test_{k}.npy")
            go[folds_ == k] = (a[:, :, 1] - a[:, :, 0]).mean(1)
            gt.append((b[:, :, 1] - b[:, :, 0]).mean(1))
        return go, np.mean(gt, 0)

    lo = np.load(PRED / "l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    lt = np.load(PRED / "l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    l7o = (lo[:, :, 1] - lo[:, :, 0]).mean(1)
    l7t = np.mean([(lt[f, :, :, 1] - lt[f, :, :, 0]).mean(1) for f in range(5)], 0)
    gpt_o, gpt_t = load_gpt2(folds)
    Xo_list.append((L_fold(gpt_o, folds) - L_fold(l7o, folds)).astype(np.float32))
    Xt_list.append((L(gpt_t) - L(l7t)).astype(np.float32))
    Xo_list.append(np.abs(L_fold(gpt_o, folds) - L_fold(l7o, folds)).astype(np.float32))
    Xt_list.append(np.abs(L(gpt_t) - L(l7t)).astype(np.float32))

    # RealMLP / GBDT if available
    if AMAN.exists():
        Z = np.load(AMAN)
        if "oof_yekenot_realmlp" in Z.files:
            r_o, r_t = Z["oof_yekenot_realmlp"].ravel(), Z["test_yekenot_realmlp"].ravel()
            Xo_list.append((L_fold(gpt_o, folds) - L_fold(r_o, folds)).astype(np.float32))
            Xt_list.append((L(gpt_t) - L(r_t)).astype(np.float32))
        if "oof_gb_bag5" in Z.files:
            g_o, g_t = Z["oof_gb_bag5"].ravel(), Z["test_gb_bag5"].ravel()
            Xo_list.append((L_fold(l7o, folds) - L_fold(g_o, folds)).astype(np.float32))
            Xt_list.append((L(l7t) - L(g_t)).astype(np.float32))

    Xo = np.column_stack(Xo_list).astype(np.float32)
    Xt = np.column_stack(Xt_list).astype(np.float32)
    # sanitize
    Xo = np.nan_to_num(Xo, nan=0.0, posinf=0.0, neginf=0.0)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    return Xo, Xt


def engine3_wilcoxon(y, folds, z_base, Xo, Xt):
    oof = np.zeros(N_TR, np.float64)
    te = np.zeros(N_TE, np.float64)
    try:
        from catboost import CatBoostRegressor
        have_cb = True
    except Exception:
        have_cb = False
    for f in range(5):
        tr, va = folds != f, folds == f
        z_tr = L(z_base[tr])
        g08 = auc_gradient_binned(z_tr, y[tr], 0.08)
        g25 = auc_gradient_binned(z_tr, y[tr], 0.25)
        g70 = auc_gradient_binned(z_tr, y[tr], 0.70)
        g = 0.45 * unit(g08) + 0.35 * unit(g25) + 0.20 * unit(g70)
        preds_va, preds_te = [], []
        # LGB
        dtr = lgb.Dataset(Xo[tr], label=g)
        params = dict(
            objective="regression", metric="l2", learning_rate=0.025,
            max_depth=4, num_leaves=11, min_child_samples=400,
            feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
            verbosity=-1, seed=42 + f, n_jobs=N_JOBS,
        )
        bst = lgb.train(params, dtr, num_boost_round=700)
        preds_va.append(bst.predict(Xo[va]))
        preds_te.append(bst.predict(Xt))
        if have_cb:
            try:
                cb = CatBoostRegressor(
                    depth=5, l2_leaf_reg=40, iterations=400, learning_rate=0.03,
                    random_seed=42 + f, verbose=False, thread_count=N_JOBS,
                )
                cb.fit(Xo[tr], g)
                preds_va.append(cb.predict(Xo[va]))
                preds_te.append(cb.predict(Xt))
            except Exception as e:
                log(f"  E3 CB fold {f} skip: {e}")
        oof[va] = np.mean(preds_va, 0)
        te += np.mean(preds_te, 0) / 5
        log(f"  E3 fold {f} done")
    # Method-B lock
    o = unit(L_fold(oof, folds))
    t = unit(L(te))
    t = t * (o.std() / (t.std() + 1e-12))
    return o.astype(np.float32), t.astype(np.float32)


# ── Engine 4 ─────────────────────────────────────────────────
def load_pillar_streams(folds):
    pillars = {f"P{i}": [] for i in range(1, 6)}
    # P1 BPE LM
    gdir = PRED / "gpt2_med_4ep"
    if gdir.exists():
        go = np.zeros(N_TR)
        gt = []
        for k in range(5):
            a = np.load(gdir / f"oof_{k}.npy")
            b = np.load(gdir / f"test_{k}.npy")
            go[folds == k] = (a[:, :, 1] - a[:, :, 0]).mean(1)
            gt.append((b[:, :, 1] - b[:, :, 0]).mean(1))
        pillars["P1"].append((go, np.mean(gt, 0), "gpt2_med_4ep"))
    if AMAN.exists():
        Z = np.load(AMAN)
        mapping = {
            "P1": ["masaya_gpt2", "gpt2_seed2026"],
            "P2": ["megayak_5f_lm", "lm7_5fold"],
            "P3": ["yekenot_realmlp", "daoviet_gpdvae"],
            "P4": ["tabpfn_slimypunk_tp1", "sadam_tabpfn_100k_tp3", "sameer_glm10f", "alex_glm_gen"],
            "P5": ["gbn_bag5_w512", "gb_bag5", "megayak_shallow_tree", "sadam_v6_gb_xgb", "sadam_v6_gb_lgb"],
        }
        for pk, names in mapping.items():
            for n in names:
                if is_black(n):
                    continue
                ok, tk = f"oof_{n}", f"test_{n}"
                if ok in Z.files and tk in Z.files:
                    pillars[pk].append((Z[ok].ravel(), Z[tk].ravel(), n))
    # l7 -> P2
    lo = np.load(PRED / "l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    lt = np.load(PRED / "l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    l7o = (lo[:, :, 1] - lo[:, :, 0]).mean(1)
    l7t = np.mean([(lt[f, :, :, 1] - lt[f, :, :, 0]).mean(1) for f in range(5)], 0)
    pillars["P2"].append((l7o, l7t, "l7_s2026"))
    # xgb -> P5
    xo = np.load(PRED / "paradigm_B/xgb_llr_oof.npy").ravel()
    xt = np.load(PRED / "paradigm_B/xgb_llr_test.npy").ravel()
    pillars["P5"].append((xo, xt, "B_xgb_llr"))
    # gpw realmlp -> P3
    if GPW.exists():
        for p in sorted(GPW.glob("oof_realmlp*.npy"))[:6]:
            n = p.name[4:-4]
            if is_black(n):
                continue
            tp = GPW / f"test_{n}.npy"
            if tp.exists():
                pillars["P3"].append((np.load(p).ravel(), np.load(tp).ravel(), n))
    return pillars


def pillar_mean(streams):
    if not streams:
        return None, None
    o = np.mean([U(s[0]) for s in streams], 0)
    t = np.mean([U(s[1]) for s in streams], 0)
    return o, t


def engine4(y, folds, z_base, pillars):
    P = {}
    for k, streams in pillars.items():
        o, t = pillar_mean(streams)
        if o is not None:
            P[k] = (o, t)
            log(f"  pillar {k}: {len(streams)} streams")
    keys = sorted(P.keys())
    assert len(keys) >= 3, keys

    # 4A Platt fusion
    oof_a = np.zeros(N_TR)
    te_a = np.zeros(N_TE)
    for f in range(5):
        tr, va = folds != f, folds == f
        probs_tr, probs_va, probs_te = [], [], []
        for k in keys:
            o, t = P[k]
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            lr.fit(L(o[tr]).reshape(-1, 1), y[tr])
            # rescale so mean ≈ 0.0167 on tr
            raw = lr.decision_function(L(o[tr]).reshape(-1, 1))
            # simple temperature
            p = expit(raw)
            # shift intercept to match prevalence
            # binary search bias
            lo, hi = -5.0, 5.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if expit(raw + mid).mean() < 0.0167:
                    lo = mid
                else:
                    hi = mid
            bias = 0.5 * (lo + hi)
            probs_tr.append(expit(lr.decision_function(L(o[tr]).reshape(-1, 1)) + bias))
            probs_va.append(expit(lr.decision_function(L(o[va]).reshape(-1, 1)) + bias))
            probs_te.append(expit(lr.decision_function(L(t).reshape(-1, 1)) + bias))
        # fuse in log-odds
        fuse_va = np.mean([logit(np.clip(p, 1e-6, 1 - 1e-6)) for p in probs_va], 0)
        fuse_te = np.mean([logit(np.clip(p, 1e-6, 1 - 1e-6)) for p in probs_te], 0)
        oof_a[va] = fuse_va
        te_a += fuse_te / 5

    # 4B disagreement router
    Po = np.column_stack([L_fold(P[k][0], folds) for k in keys])
    Pt = np.column_stack([L(P[k][1]) for k in keys])
    zL = L_fold(z_base, folds)
    zLt = L(z_base)  # placeholder; use mean L of z for test via U then L of test z — caller passes z_test
    # Actually need z_test — use from MOD
    z_te = np.load(MOD / "z_modern_pub_test.npy").ravel()
    zLt = L(z_te)

    twilight = 4 * expit(zL) * (1 - expit(zL))
    twilight_t = 4 * expit(zLt) * (1 - expit(zLt))
    p1 = Po[:, keys.index("P1")] if "P1" in keys else Po[:, 0]
    p2 = Po[:, keys.index("P2")] if "P2" in keys else Po[:, min(1, Po.shape[1] - 1)]
    p5 = Po[:, keys.index("P5")] if "P5" in keys else Po[:, -1]
    p1t = Pt[:, keys.index("P1")] if "P1" in keys else Pt[:, 0]
    p2t = Pt[:, keys.index("P2")] if "P2" in keys else Pt[:, min(1, Pt.shape[1] - 1)]
    p5t = Pt[:, keys.index("P5")] if "P5" in keys else Pt[:, -1]
    std_p = Po.std(1)
    std_pt = Pt.std(1)
    mx, mn = Po.max(1), Po.min(1)
    mxt, mnt = Pt.max(1), Pt.min(1)

    u5o = np.load(V43T / "U5_gini10_llr_oof.npy").ravel() if (V43T / "U5_gini10_llr_oof.npy").exists() else np.zeros(N_TR)
    u5t = np.load(V43T / "U5_gini10_llr_test.npy").ravel() if (V43T / "U5_gini10_llr_test.npy").exists() else np.zeros(N_TE)

    Xo = np.column_stack([
        (p1 - p2) * twilight,
        (mx - mn) * np.sign(p1 + p2 - 2 * p5),
        std_p * p1,
        unit(u5o) * (p1 - p5),
        zL,
    ]).astype(np.float32)
    Xt = np.column_stack([
        (p1t - p2t) * twilight_t,
        (mxt - mnt) * np.sign(p1t + p2t - 2 * p5t),
        std_pt * p1t,
        unit(u5t) * (p1t - p5t),
        zLt,
    ]).astype(np.float32)

    oof_b = np.zeros(N_TR)
    te_b = np.zeros(N_TE)
    for f in range(5):
        tr, va = folds != f, folds == f
        clf = LogisticRegression(C=0.05, solver="lbfgs", max_iter=300)
        clf.fit(Xo[tr], y[tr])
        # enforce soft monotonicity via residual orth to z then add back 0.85*z
        raw_va = clf.decision_function(Xo[va])
        raw_te = clf.decision_function(Xt)
        # blend
        oof_b[va] = 0.85 * zL[va] + 0.15 * unit(raw_va) * zL[va].std()
        te_b += (0.85 * zLt + 0.15 * unit(raw_te) * zLt.std()) / 5

    # combine 4A and 4B
    oof = 0.5 * unit(oof_a) + 0.5 * unit(oof_b)
    te = 0.5 * unit(te_a) + 0.5 * unit(te_b)
    return oof.astype(np.float32), te.astype(np.float32)


# ── gating ───────────────────────────────────────────────────
def evaluate_engine(name, oof, te, y, folds, z_mod, z_mod_t, z_pub, z_pub_t, Zg, ray_s02, ray_s06, B147_t=None):
    # pre-unitize raw disp
    disp_raw = float(np.std(te) / (np.std(oof) + 1e-12))
    ks_raw = float(ks_2samp(unit(oof), unit(te)).statistic)
    # GS orth vs modern + loser rays (on test for tip; OOF vs modern only LOFO-ish)
    Bo = unit(L_fold(z_mod, folds))
    Bt = unit(L(z_mod_t))
    oo = gs_orth(oof, [Bo])
    tt = gs_orth(te, [Bt, unit(ray_s02), unit(ray_s06)])
    tt = tt * (oo.std() / (tt.std() + 1e-12))

    results = {"name": name, "disp_raw": disp_raw, "KS_raw": ks_raw}
    for tag, base_o, base_t, base_label in (
        ("z_modern", z_mod, z_mod_t, "z_modern_pub"),
        ("z_pub", z_pub, z_pub_t, "z_pub_only"),
    ):
        base_zr = apply_zr(base_o, Zg, base_o)
        best = {"mean_e5": -1e9}
        for w in np.linspace(-0.08, 0.08, 161):
            cand = apply_zr(L(base_o) + float(w) * oo, Zg, base_o)
            fm = fold_metrics(y, cand, base_zr, folds)
            if fm["mean_e5"] > best["mean_e5"]:
                best = {**fm, "w": float(w), "cand": cand}
        sf, t25b, wp = sf25(y, best["cand"], base_zr, folds)
        tip = unit(tt) * np.sign(best["w"] if abs(best["w"]) > 1e-12 else 1.0)
        r2 = float("nan")
        if B147_t is not None:
            # novelty vs tip direction correl
            r2 = float(spearmanr(tip, B147_t).statistic) ** 2
        results[tag] = {
            "base": base_label,
            "mean_e5": best["mean_e5"],
            "pos": best["pos"],
            "min_e5": best["min_e5"],
            "folds": best["folds"],
            "w": best["w"],
            "sf25": sf,
            "t25": best["t25"],
            "t25_sf": t25b,
            "wilcoxon_p25": wp,
            "cos_S02": cos_full(tip, ray_s02),
            "cos_S06": cos_full(tip, ray_s06),
            "R2_vs_B147": r2,
            "disp_post": float(np.std(tt) / (np.std(oo) + 1e-12)),
            "KS_post": float(ks_2samp(oo, tt).statistic),
        }
        log(
            f"GATE {name}/{tag}: Δe5={best['mean_e5']:+.2f} pos={best['pos']}/5 "
            f"min={best['min_e5']:+.2f} sf={sf}/25 t={best['t25']:.2f} w={best['w']:+.4f}"
        )
    return results, oo, tt


def main():
    log("=== V43.2 local 4-engine suite | NO_SUBMIT | 0 pods ===")
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    y = tr["is_fraud"].to_numpy(int)
    ids = te["id"].to_numpy()
    Zg = z_goodpjw(tr)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = np.zeros(N_TR, int)
    for f, (_, va) in enumerate(skf.split(np.zeros(N_TR), y)):
        folds[va] = f

    z_mod = np.load(MOD / "z_modern_pub_oof.npy").ravel()
    z_mod_t = np.load(MOD / "z_modern_pub_test.npy").ravel()
    z_pub = np.load(SSD / "preds/z_pub_only_oof.npy").ravel()
    z_pub_t = np.load(SSD / "preds/z_pub_only_test.npy").ravel()
    z_mod_zr = apply_zr(z_mod, Zg, z_mod)
    assert abs(float(roc_auc_score(y, z_mod_zr)) - 0.952652) < 5e-5

    ab = pd.read_csv(FINAL / "SLOT01_pub_Step_AB_095569.csv")["is_fraud"].to_numpy(float)
    s02 = pd.read_csv(FINAL / "SLOT02_AB_clip985_areeb_a0p32.csv")["is_fraud"].to_numpy(float)
    s06 = pd.read_csv(FINAL / "SLOT06_AB_C3_harvest_p4_clip985_0p28e4.csv")["is_fraud"].to_numpy(float)
    ray_s02, ray_s06 = L(s02) - L(ab), L(s06) - L(ab)

    orig_path = next(p for p in ORIG_CANDS if p.exists())
    orig = pd.read_csv(orig_path)
    y_orig = orig["is_fraud"].to_numpy(int)
    log(f"orig20k={orig_path} n={len(orig)} fraud={(y_orig==1).sum()}")

    # ── shared tokens ──
    log("tokenizeize...")
    toks, tok_names = build_tokens(tr, te, orig)
    log(f"tokens C={toks['tr'].shape[1]}")

    # ════════ ENGINE 1 ════════
    log("=== ENGINE 1 orig339 Newton ===")
    ov_tr = overlap_feats(toks["tr"], toks["orig"], y_orig, tok_names)
    ov_te = overlap_feats(toks["te"], toks["orig"], y_orig, tok_names)
    log("overlap done; gen-shift...")
    gs_tr, gs_te = gen_shift_feats(tr.copy(), te.copy(), orig.copy(), y_orig, folds, y)
    X1o = np.concatenate([ov_tr, gs_tr], 1).astype(np.float32)
    X1t = np.concatenate([ov_te, gs_te], 1).astype(np.float32)
    X1o = np.nan_to_num(X1o)
    X1t = np.nan_to_num(X1t)
    e1o, e1t = newton_booster(y, folds, z_mod, X1o, X1t, "E1")
    np.save(OUT / "e1_orig339_oof.npy", e1o)
    np.save(OUT / "e1_orig339_test.npy", e1t)

    # ════════ ENGINE 2 ════════
    log("=== ENGINE 2 epistatic ===")
    # Block 2A
    cont_sf = []
    for c in ("amount_usd", "account_balance_usd", "distance_from_home_km", "hours_since_last_txn"):
        tr[f"{c}_cents"] = np.round((tr[c].to_numpy() % 1) * 100).astype(int)
        te[f"{c}_cents"] = np.round((te[c].to_numpy() % 1) * 100).astype(int)
        cont_sf.append(f"{c}_cents")
    pairs_2a = [(a, b) for a in cont_sf for b in ("merchant_category", "card_type", "time_of_day_hour")]
    pairs_2a = pairs_2a[:12]  # cap
    Fo_a, Ft_a = epistatic_block(y, folds, z_mod, tr, te, pairs_2a)
    log(f"  2A cols={Fo_a.shape[1]}")
    # Block 2B
    for c, name in (("merchant_risk_score", "risk_q40"), ("velocity_score", "vel_q40")):
        edges = np.quantile(tr[c].to_numpy(), np.linspace(0, 1, 41))
        tr[name] = np.digitize(tr[c], edges[1:-1])
        te[name] = np.digitize(te[c], edges[1:-1])
    tr["_flag"] = flag_bitmask64(tr)
    te["_flag"] = flag_bitmask64(te)
    pairs_2b = [
        ("vel_q40", "_flag"), ("risk_q40", "_flag"),
        ("cvv_retry_count", "_flag"), ("prior_disputes", "_flag"),
        ("auth_method", "channel"), ("channel", "device_type"),
    ]
    Fo_b, Ft_b = epistatic_block(y, folds, z_mod, tr, te, pairs_2b)
    log(f"  2B cols={Fo_b.shape[1]}")
    # Block 2C twilight-gated 3-way proxies as pairwise
    pairs_2c = [
        ("_flag", "is_foreign_transaction"),
        ("_flag", "is_new_merchant"),
        ("merchant_category", "risk_q40"),
        ("merchant_category", "vel_q40"),
    ]
    Fo_c, Ft_c = epistatic_block(y, folds, z_mod, tr, te, pairs_2c)
    # twilight gate
    pcal = expit(L_fold(z_mod, folds))
    tw = (4 * pcal * (1 - pcal)).astype(np.float32)
    Fo_c = Fo_c * tw.reshape(-1, 1)
    Ft_c = Ft_c * (4 * expit(L(z_mod_t)) * (1 - expit(L(z_mod_t)))).reshape(-1, 1).astype(np.float32)
    log(f"  2C cols={Fo_c.shape[1]}")

    # condense blocks via Ridge vs residual then nest
    def condense(Fo, Ft, tag):
        oof = np.zeros(N_TR)
        te_ = np.zeros(N_TE)
        for f in range(5):
            tr_m, va = folds != f, folds == f
            z_tr = L(z_mod[tr_m])
            lr = LogisticRegression(C=1e6, max_iter=200)
            lr.fit(z_tr.reshape(-1, 1), y[tr_m])
            p = expit(lr.decision_function(z_tr.reshape(-1, 1)))
            resid = y[tr_m] - p
            ridge = Ridge(alpha=2500.0)
            ridge.fit(Fo[tr_m], resid)
            oof[va] = ridge.predict(Fo[va])
            te_ += ridge.predict(Ft) / 5
        return oof.astype(np.float32), te_.astype(np.float32)

    b2ao, b2at = condense(Fo_a, Ft_a, "2A")
    b2bo, b2bt = condense(Fo_b, Ft_b, "2B")
    b2co, b2ct = condense(Fo_c, Ft_c, "2C")
    X2o = np.column_stack([b2ao, b2bo, b2co]).astype(np.float32)
    X2t = np.column_stack([b2at, b2bt, b2ct]).astype(np.float32)
    e2o, e2t = newton_booster(y, folds, z_mod, X2o, X2t, "E2")
    np.save(OUT / "e2_epistatic_oof.npy", e2o)
    np.save(OUT / "e2_epistatic_test.npy", e2t)

    # ════════ ENGINE 3 ════════
    log("=== ENGINE 3 Wilcoxon trees ===")
    X3o, X3t = build_x_auc(tr, te, folds)
    log(f"  X_auc cols={X3o.shape[1]}")
    e3o, e3t = engine3_wilcoxon(y, folds, z_mod, X3o, X3t)
    np.save(OUT / "e3_wilcoxon_tree_oof.npy", e3o)
    np.save(OUT / "e3_wilcoxon_tree_test.npy", e3t)

    # ════════ ENGINE 4 ════════
    log("=== ENGINE 4 pillar MoE ===")
    pillars = load_pillar_streams(folds)
    e4o, e4t = engine4(y, folds, z_mod, pillars)
    np.save(OUT / "e4_pillar_moe_oof.npy", e4o)
    np.save(OUT / "e4_pillar_moe_test.npy", e4t)

    # combo
    log("=== E_combo ===")
    Xco = np.column_stack([e1o, e2o, e3o, e4o]).astype(np.float32)
    Xct = np.column_stack([e1t, e2t, e3t, e4t]).astype(np.float32)
    eco, ect = newton_booster(y, folds, z_mod, Xco, Xct, "Ecombo")
    np.save(OUT / "e_combo_oof.npy", eco)
    np.save(OUT / "e_combo_test.npy", ect)

    # ── evaluate all ──
    engines = {
        "E1_orig339": (e1o, e1t),
        "E2_epistatic": (e2o, e2t),
        "E3_wilcoxon": (e3o, e3t),
        "E4_pillar_moe": (e4o, e4t),
        "E_combo": (eco, ect),
    }
    B147_t = unit(L(z_mod_t))  # proxy novelty vs modern tip
    all_res = {}
    staged = []
    for name, (oo, tt) in engines.items():
        res, oo_orth, tt_orth = evaluate_engine(
            name, oo, tt, y, folds, z_mod, z_mod_t, z_pub, z_pub_t, Zg, ray_s02, ray_s06, B147_t
        )
        all_res[name] = res
        # stage if pos>=4 on z_modern
        zm = res["z_modern"]
        if zm["pos"] >= 4:
            w = zm["w"]
            tip = unit(tt_orth) * np.sign(w if abs(w) > 1e-12 else 1.0)
            Zg_te = z_goodpjw(te)
            lo, hi = 1e-6, 20.0
            target = 0.10e-4
            best_p = ab
            for _ in range(28):
                mid = 0.5 * (lo + hi)
                p = apply_zr(L(ab) + mid * tip, Zg_te, ab)
                d = 1.0 - float(spearmanr(p, ab).statistic)
                if d < target:
                    lo = mid
                else:
                    hi = mid
                best_p = p
            path = OUT / f"NOSUBMIT_{name}_w{w:+.4f}_tip0p10e4.csv"
            text = pd.DataFrame({"id": ids, "is_fraud": U(best_p)}).to_csv(index=False, lineterminator="\n")
            path.write_bytes(text.encode())
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            staged.append({"name": name, "file": path.name, "sha": sha, "metrics": zm})
            log(f"STAGED {path.name} sha={sha[:12]}")

    report = {
        "NO_SUBMIT": True,
        "pods_touched": 0,
        "baseline_z_modern_pub_zr": float(roc_auc_score(y, z_mod_zr)),
        "orig20k": str(orig_path),
        "engines": all_res,
        "staged": staged,
    }
    (LOG / "V43_2_LOCAL_4ENGINES_REPORT.json").write_text(json.dumps(report, indent=2, default=str))

    # markdown
    lines = [
        "# V43.2 — Local Parallel 4-Engine Discovery",
        "",
        "**NO_SUBMIT=True · 0 pods · $0 local CPU/MPS**",
        "",
        f"Baseline `z_modern_pub+Zr` = **{report['baseline_z_modern_pub_zr']:.6f}** · orig20k = `{orig_path.name}` (339 fraud)",
        "",
        "## Gates vs z_modern_pub",
        "",
        "| engine | Δe5 | pos/5 | min | sf/25 | t25 | w | cosS02 | cosS06 | disp_raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, res in all_res.items():
        zm = res["z_modern"]
        lines.append(
            f"| {name} | {zm['mean_e5']:+.2f} | {zm['pos']}/5 | {zm['min_e5']:+.2f} | "
            f"{zm['sf25']}/25 | {zm['t25']:.2f} | {zm['w']:+.4f} | {zm['cos_S02']:+.3f} | "
            f"{zm['cos_S06']:+.3f} | {res['disp_raw']:.3f} |"
        )
    lines += ["", "## Gates vs z_pub_only", "",
              "| engine | Δe5 | pos/5 | sf/25 | t25 |",
              "|---|---:|---:|---:|---:|"]
    for name, res in all_res.items():
        zp = res["z_pub"]
        lines.append(f"| {name} | {zp['mean_e5']:+.2f} | {zp['pos']}/5 | {zp['sf25']}/25 | {zp['t25']:.2f} |")
    lines += ["", f"## Staged NOSUBMIT ({len(staged)})"]
    for s in staged:
        lines.append(f"- `{s['file']}` sha=`{s['sha'][:16]}…` Δe5={s['metrics']['mean_e5']:+.2f} pos={s['metrics']['pos']}/5")
    lines += ["", "## STOP", "NO_SUBMIT=True. Zero submissions. Zero GPU pods touched."]
    (LOG / "V43_2_LOCAL_4ENGINES_REPORT.md").write_text("\n".join(lines) + "\n")
    log(f"WROTE {LOG / 'V43_2_LOCAL_4ENGINES_REPORT.md'}")
    print("\n".join(lines[lines.index("## Gates vs z_modern_pub"):]), flush=True)
    print("STOP. NO_SUBMIT=True. 0 pods.", flush=True)


if __name__ == "__main__":
    main()
