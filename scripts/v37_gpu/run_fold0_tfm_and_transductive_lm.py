#!/usr/bin/env python3
"""V37 production GPU worker — TabPFN + TabICLv2 + GPT-2 Storyteller.

NO_SUBMIT=True. Never calls runpodctl remove (Mac controller owns lifecycle).
Usage:
  python -u scripts/v37_gpu/run_fold0_tfm_and_transductive_lm.py --folds 0
  python -u scripts/v37_gpu/run_fold0_tfm_and_transductive_lm.py --folds 1,2,3,4 --canaries tabicl,gpt2
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit, logsumexp
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(os.environ.get("V37_REPO", "/workspace/repo")).resolve()
if not (REPO / "data" / "train.csv").exists():
    REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("V37_OUT", "/workspace/out")).resolve()
PRED_LOCAL = REPO / "preds" / "v36_1"
FOLDS_CSV = REPO / "external" / "banks" / "goodpjw2008_s6e11-oof-bank" / "folds.csv"
ORIG_CANDIDATES = [
    REPO / "data" / "original.csv",
    REPO / "data" / "orig20k.csv",
    REPO / "external" / "v37_twin_check" / "candidates" / "uditjain_orig" / "credit_card_fraud_2026.csv",
    REPO / "external" / "v37_twin_check" / "orig20k_uditjain.csv",
]

PREV = 0.0167
N_TEST = 299_984
MAX_CAT_LEVELS = 100
USE_COMPILE = os.environ.get("USE_COMPILE", "0") == "1"

CONT13 = [
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
]
CAT11 = [
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
RAW_24 = CONT13 + CAT11
BOOL_FLAG = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def U(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / max(len(v), 1)


def L(v: np.ndarray) -> np.ndarray:
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def auc(y, p) -> float:
    return float(roc_auc_score(y, np.asarray(p, float).ravel()))


def cuda_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prior_shift(p: np.ndarray, target: float = PREV) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    lo, hi = -12.0, 12.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        m = float(expit(logit(p) + mid).mean())
        if m > target:
            hi = mid
        else:
            lo = mid
    return expit(logit(p) + 0.5 * (lo + hi))


def svd40(B: np.ndarray) -> np.ndarray:
    X = np.column_stack([unit(B[:, j]) for j in range(B.shape[1])])
    Xc = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(40, X.shape[1])
    return Xc @ vt[:k].T


def r2_span(u: np.ndarray, B40: np.ndarray) -> float:
    u = unit(u)
    ridge = Ridge(alpha=50.0, fit_intercept=False)
    ridge.fit(B40, u)
    pred = ridge.predict(B40)
    ss = np.sum((u - u.mean()) ** 2) + 1e-12
    return float(1.0 - np.sum((u - pred) ** 2) / ss)


def save_finite(path: Path, arr: np.ndarray, expect_shape: tuple) -> None:
    arr = np.asarray(arr, np.float64).ravel()
    assert arr.shape == expect_shape, f"{path}: shape {arr.shape} != {expect_shape}"
    assert np.isfinite(arr).all(), f"{path}: NaN/Inf"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))
    log(f"[save] {path} shape={arr.shape} finite=OK")


def load_data():
    tr = pd.read_csv(REPO / "data" / "train.csv")
    te = pd.read_csv(REPO / "data" / "test.csv")
    folds = pd.read_csv(FOLDS_CSV)["fold"].to_numpy(int)
    assert len(folds) == len(tr)
    y = tr["is_fraud"].to_numpy(int)
    assert len(te) == N_TEST
    orig = None
    for p in ORIG_CANDIDATES:
        if p.exists():
            orig = pd.read_csv(p)
            log(f"[data] orig20k={p} rows={len(orig)}")
            break
    if orig is None:
        raise FileNotFoundError("orig20k not found")
    return tr, te, y, folds, orig


def factorize_bounded(series_list, max_levels=MAX_CAT_LEVELS):
    """Fit codes on concat; rare levels -> 0 (unk). Returns list of int arrays."""
    all_s = pd.concat([s.astype(str) for s in series_list], ignore_index=True)
    vc = all_s.value_counts()
    keep = set(vc.head(max_levels - 1).index)  # reserve 0=unk
    mapping = {v: i + 1 for i, v in enumerate(sorted(keep))}
    outs = []
    offset = 0
    for s in series_list:
        n = len(s)
        codes = s.astype(str).map(lambda x: mapping.get(x, 0)).to_numpy(np.int32)
        outs.append(codes)
        offset += n
    return outs


def mantissa(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), 1e-12, None)
    return np.log10(x) - np.floor(np.log10(x))


def build_tfm_features(tr, te, folds, fold_id: int):
    """Label-free features; fold-clean freqs on tr_m only; cats bounded."""
    f_mask = folds == fold_id
    tr_m = ~f_mask
    feats_tr, feats_te = [], []
    names, cat_idx = [], []

    def add(name, a, b, categorical=False):
        feats_tr.append(np.asarray(a, np.float32))
        feats_te.append(np.asarray(b, np.float32))
        names.append(name)
        if categorical:
            cat_idx.append(len(names) - 1)

    for c in CONT13:
        add(c, tr[c], te[c])
    amt_tr, bal_tr = tr["amount_usd"].to_numpy(float), tr["account_balance_usd"].to_numpy(float)
    amt_te, bal_te = te["amount_usd"].to_numpy(float), te["account_balance_usd"].to_numpy(float)
    add("log1p_amt", np.log1p(amt_tr), np.log1p(amt_te))
    add("log1p_bal", np.log1p(bal_tr), np.log1p(bal_te))
    add("amt_over_bal", amt_tr / (bal_tr + 1.0), amt_te / (bal_te + 1.0))
    add("amt_mantissa", mantissa(amt_tr), mantissa(amt_te))
    add("bal_mantissa", mantissa(bal_tr), mantissa(bal_te))

    # fold-clean freqs
    g_amt = tr.loc[tr_m, "amount_usd"].astype(str).value_counts()
    g_bal = tr.loc[tr_m, "account_balance_usd"].astype(str).value_counts()
    add(
        "log1p_amt_freq_tr",
        np.log1p(tr["amount_usd"].astype(str).map(g_amt).fillna(0).to_numpy(float)),
        np.log1p(te["amount_usd"].astype(str).map(g_amt).fillna(0).to_numpy(float)),
    )
    add(
        "log1p_bal_freq_tr",
        np.log1p(tr["account_balance_usd"].astype(str).map(g_bal).fillna(0).to_numpy(float)),
        np.log1p(te["account_balance_usd"].astype(str).map(g_bal).fillna(0).to_numpy(float)),
    )
    # transductive freqs train∪test
    for c, name in (("amount_usd", "amt"), ("account_balance_usd", "bal")):
        allv = pd.concat([tr[c].astype(str), te[c].astype(str)], ignore_index=True)
        freq = allv.value_counts()
        fr = pd.concat([tr[c].astype(str), te[c].astype(str)], ignore_index=True).map(freq).fillna(0).to_numpy(float)
        add(f"log1p_{name}_freq_tt", np.log1p(fr[: len(tr)]), np.log1p(fr[len(tr) :]))

    # categoricals bounded
    for c in CAT11:
        ctr, cte = factorize_bounded([tr[c], te[c]])
        add(c, ctr, cte, categorical=True)

    cents_tr = (np.round(amt_tr * 100) % 100).astype(np.int32)
    cents_te = (np.round(amt_te * 100) % 100).astype(np.int32)
    add("amt_cents", cents_tr, cents_te, categorical=True)
    bc_tr = (np.round(bal_tr * 100) % 100).astype(np.int32)
    bc_te = (np.round(bal_te * 100) % 100).astype(np.int32)
    add("bal_cents", bc_tr, bc_te, categorical=True)

    for c, name in (("amount_usd", "amt"), ("account_balance_usd", "bal")):
        allv = pd.concat([tr[c].astype(str), te[c].astype(str)], ignore_index=True)
        freq = allv.value_counts()
        rank = pd.concat([tr[c].astype(str), te[c].astype(str)], ignore_index=True).map(freq).fillna(0).to_numpy(float)
        edges = np.quantile(rank, np.linspace(0, 1, 33)[1:-1])
        bins = np.digitize(rank, edges).astype(np.int32)  # 0..31
        bins = np.clip(bins, 0, 31)
        add(f"{name}_freq_bin32", bins[: len(tr)], bins[len(tr) :], categorical=True)

    def flag64(df):
        v = np.zeros(len(df), np.int32)
        for i, c in enumerate(BOOL_FLAG[:6]):
            v |= (df[c].astype(int).to_numpy() & 1) << i
        return np.clip(v, 0, 63)

    add("flag64", flag64(tr), flag64(te), categorical=True)

    X = np.column_stack(feats_tr).astype(np.float32)
    Xt = np.column_stack(feats_te).astype(np.float32)
    # assert cat cardinality
    for i in cat_idx:
        assert X[:, i].max() < MAX_CAT_LEVELS + 5, names[i]
    log(f"[feat] fold={fold_id} X={X.shape} cats={len(cat_idx)} names={len(names)}")
    return X, Xt, names, cat_idx, f_mask, tr_m


def disjoint_pos_neg_bags(y_pool: np.ndarray, n_bags: int, bag_n: int, n_pos_target: int, seed: int):
    """Disjoint neg bags; positives may recycle if needed but prefer all pos."""
    rng = np.random.default_rng(seed)
    pos = np.where(y_pool == 1)[0]
    neg = np.where(y_pool == 0)[0]
    rng.shuffle(neg)
    n_pos = min(n_pos_target, len(pos), bag_n - 1)
    n_neg = bag_n - n_pos
    bags = []
    # split negs into disjoint chunks
    chunks = []
    for b in range(n_bags):
        start = b * n_neg
        chunk = neg[start : start + n_neg]
        if len(chunk) < n_neg:
            # wrap remaining with replace for leftover bags
            need = n_neg - len(chunk)
            extra = rng.choice(neg, size=need, replace=True)
            chunk = np.concatenate([chunk, extra])
        chunks.append(chunk)
    for b in range(n_bags):
        ip = rng.choice(pos, size=n_pos, replace=len(pos) < n_pos)
        bags.append(np.concatenate([ip, chunks[b]]))
    return bags


def predict_chunks(clf, Xq, chunk: int):
    outs = []
    for s in range(0, len(Xq), chunk):
        outs.append(np.asarray(clf.predict_proba(Xq[s : s + chunk]))[:, 1])
    return np.concatenate(outs)


def run_tabpfn(X, y, tr_m, Xq, cat_idx, vram_gb: float):
    from tabpfn import TabPFNClassifier

    y_pool = y[tr_m]
    X_pool = X[tr_m]
    n_pos = int(y_pool.sum())
    bags = disjoint_pos_neg_bags(y_pool, n_bags=8, bag_n=20_000, n_pos_target=min(9351, n_pos), seed=2026)
    acc = np.zeros(len(Xq), float)
    for bi, idx in enumerate(bags):
        kw = dict(
            device="cuda",
            ignore_pretraining_limits=True,
            random_state=42 + bi,
            n_estimators=4,
            categorical_features_indices=cat_idx or None,
        )
        try:
            clf = TabPFNClassifier(**kw, fit_mode="fit_with_cache")
        except TypeError:
            clf = TabPFNClassifier(**kw)
        clf.fit(X_pool[idx], y_pool[idx])
        p = prior_shift(predict_chunks(clf, Xq, 20_000))
        acc += p
        del clf
        cuda_free()
        log(f"[tabpfn] bag {bi+1}/8 done")
    return acc / 8.0


def run_tabicl(X, y, tr_m, Xq, cat_idx, vram_gb: float, n_va: int):
    """A40 timing: ~5ms/row/est without useful speedup → 440k×4×4 ≫40min.

    Fold-0 canary (still Deotte-faithful bags, wall-safe):
      - 4 disjoint 20k bags (15k if VRAM<30GB), n_estimators=2, fp32, kv_cache=True
      - Each bag scores Fold-va only (ensemble)
      - Bags 0–1 also score full test (2-bag test mean) so raw_disp is defined
    """
    from tabicl import TabICLClassifier

    bag_n = 15_000 if vram_gb < 30 else 20_000
    qchunk = 2_048 if vram_gb < 30 else 4_096
    n_bags = 4
    n_est = 2
    y_pool = y[tr_m]
    X_pool = X[tr_m]
    n_pos = int(y_pool.sum())
    bags = disjoint_pos_neg_bags(y_pool, n_bags=n_bags, bag_n=bag_n, n_pos_target=min(9351, n_pos), seed=4242)
    X_va, X_te = Xq[:n_va], Xq[n_va:]
    acc_va = np.zeros(n_va, float)
    acc_te = np.zeros(len(X_te), float)
    n_te_bags = 0
    for bi, idx in enumerate(bags):
        t_bag = time.time()
        clf = TabICLClassifier(
            n_estimators=n_est, batch_size=1, kv_cache=True, device="cuda", random_state=7 + bi
        )
        with torch.amp.autocast("cuda", enabled=False):
            clf.fit(X_pool[idx], y_pool[idx])
            log(f"[tabicl] bag {bi+1}/{n_bags} fit {time.time()-t_bag:.1f}s — score va={n_va}")
            p_va = prior_shift(predict_chunks(clf, X_va, qchunk))
            acc_va += p_va
            if bi < 2:  # two bags cover test for disp/r²
                log(f"[tabicl] bag {bi+1} scoring test={len(X_te)}…")
                p_te = prior_shift(predict_chunks(clf, X_te, qchunk))
                acc_te += p_te
                n_te_bags += 1
        del clf
        cuda_free()
        log(f"[tabicl] bag {bi+1}/{n_bags} done wall={time.time()-t_bag:.1f}s")
    p_va = acc_va / n_bags
    p_te = acc_te / max(n_te_bags, 1)
    return np.concatenate([p_va, p_te])


def gate_canary(
    name: str,
    p_va: np.ndarray,
    p_te: np.ndarray,
    y_va: np.ndarray,
    z_fb_va,
    z_tw_va,
    z_fb_te,
    B_oof_va,
    B_te,
) -> dict:
    """Out-of-subfold blend-lift gate — zero in-sample leakage on w* / orth."""
    solo = auc(y_va, p_va)
    delta_va = L(p_va) - L(z_fb_va)
    delta_te = L(p_te) - L(z_fb_te)
    raw_disp = float(delta_te.std() / (delta_va.std() + 1e-12))
    B40 = svd40(B_te)
    r2 = r2_span(delta_te, B40)

    def subfold_stats(z_base_va):
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        d_list = []
        for tr_sf, va_sf in skf.split(np.zeros(len(y_va)), y_va):
            # orth vs B on tr_sf only
            u = unit(L(p_va) - L(z_base_va))
            Btr = np.column_stack([unit(B_oof_va[tr_sf, j]) for j in range(B_oof_va.shape[1])])
            ridge = Ridge(alpha=50.0, fit_intercept=False)
            ridge.fit(Btr, u[tr_sf])
            Bva = np.column_stack([unit(B_oof_va[va_sf, j]) for j in range(B_oof_va.shape[1])])
            u_orth = u.copy()
            u_orth[va_sf] = u[va_sf] - ridge.predict(Bva)
            u_orth[tr_sf] = u[tr_sf] - ridge.predict(Btr)
            # grid w* on tr_sf only
            best_w, best_a = 0.0, -1.0
            base_tr = auc(y_va[tr_sf], z_base_va[tr_sf])
            for w in np.linspace(-0.08, 0.08, 81):
                a = auc(y_va[tr_sf], L(z_base_va[tr_sf]) + w * u_orth[tr_sf])
                if a > best_a:
                    best_a, best_w = a, w
            d = auc(y_va[va_sf], L(z_base_va[va_sf]) + best_w * u_orth[va_sf]) - auc(
                y_va[va_sf], z_base_va[va_sf]
            )
            d_list.append(float(d))
        d_arr = np.asarray(d_list)
        return int((d_arr >= 0.40e-4).sum()), float(d_arr.mean()), d_arr.tolist()

    pos_fb, mean_fb, d_fb = subfold_stats(z_fb_va)
    pos_tw, mean_tw, d_tw = subfold_stats(z_tw_va)
    promote = (
        solo >= 0.9465
        and 0.86 <= raw_disp <= 1.14
        and r2 < 0.80
        and pos_fb >= 4
        and pos_tw >= 4
        and mean_fb >= 0.40e-4
        and mean_tw >= 0.40e-4
        and np.sign(mean_fb) == np.sign(mean_tw)
    )
    rec = {
        "name": name,
        "solo_f0": solo,
        "raw_disp": raw_disp,
        "r2_delta_B40": r2,
        "pos_sf_fb": pos_fb,
        "pos_sf_tw": pos_tw,
        "mean_d_fb": mean_fb,
        "mean_d_tw": mean_tw,
        "d_fb": d_fb,
        "d_tw": d_tw,
        "promote": bool(promote),
    }
    log(f"[GATE] {json.dumps({k: v for k, v in rec.items() if k not in ('d_fb', 'd_tw')})}")
    return rec


# -------------------- GPT-2 Storyteller --------------------
def verify_true_false_tokens(tok):
    t_true = tok.encode(" True", add_special_tokens=False)
    t_false = tok.encode(" False", add_special_tokens=False)
    assert len(t_true) == 1 and len(t_false) == 1, (t_true, t_false)
    return t_true[0], t_false[0]


def build_storyteller_tables(tr, te, orig, folds, fold_id):
    """Intersect RAW_24; add af/bf/ac/od; map orig onto train∪test edges."""
    cols = [c for c in RAW_24 if c in tr.columns and c in te.columns and c in orig.columns]
    assert len(cols) >= 20, cols
    # bin edges on train∪test
    amt_all = pd.concat([tr["amount_usd"], te["amount_usd"]], ignore_index=True)
    bal_all = pd.concat([tr["account_balance_usd"], te["account_balance_usd"]], ignore_index=True)
    afreq = pd.concat([tr["amount_usd"].astype(str), te["amount_usd"].astype(str)], ignore_index=True).value_counts()
    bfreq = pd.concat(
        [tr["account_balance_usd"].astype(str), te["account_balance_usd"].astype(str)], ignore_index=True
    ).value_counts()

    def freq_bin(series, freq_map):
        rank = series.astype(str).map(freq_map).fillna(0).to_numpy(float)
        # use global edges from train∪test ranks
        return rank

    # compute edges from concat ranks
    rank_amt = pd.concat([tr["amount_usd"].astype(str), te["amount_usd"].astype(str)], ignore_index=True).map(afreq).fillna(0).to_numpy(float)
    rank_bal = pd.concat(
        [tr["account_balance_usd"].astype(str), te["account_balance_usd"].astype(str)], ignore_index=True
    ).map(bfreq).fillna(0).to_numpy(float)
    edges_a = np.quantile(rank_amt, np.linspace(0, 1, 33)[1:-1])
    edges_b = np.quantile(rank_bal, np.linspace(0, 1, 33)[1:-1])

    def attach(df):
        out = df[cols].copy()
        ra = df["amount_usd"].astype(str).map(afreq).fillna(0).to_numpy(float)
        rb = df["account_balance_usd"].astype(str).map(bfreq).fillna(0).to_numpy(float)
        out["af"] = np.clip(np.digitize(ra, edges_a), 0, 31)
        out["bf"] = np.clip(np.digitize(rb, edges_b), 0, 31)
        out["ac"] = (np.round(df["amount_usd"].to_numpy(float) * 100) % 100).astype(int)
        okey = set(
            (orig["amount_usd"].astype(str) + "|" + orig["account_balance_usd"].astype(str)).tolist()
        )
        k = df["amount_usd"].astype(str) + "|" + df["account_balance_usd"].astype(str)
        out["od"] = k.map(lambda x: 0 if x in okey else 1).astype(int)
        return out

    tr_t = attach(tr)
    te_t = attach(te)
    # orig mapped onto same edges; fillna(0)
    orig_t = attach(orig)
    for c in ["af", "bf", "ac", "od"]:
        orig_t[c] = orig_t[c].fillna(0).astype(int)
    text_cols = cols + ["af", "bf", "ac", "od"]
    return tr_t, te_t, orig_t, text_cols


def pretok_column_fragments(tok, frames: list[pd.DataFrame], cols: list[str]):
    """Pre-tokenize all unique ' k=val' strings per column once (vectorized)."""
    frag = {}  # (col, strval) -> list[int]
    for c in cols:
        vals = set()
        for df in frames:
            vals.update(df[c].astype(str).unique().tolist())
        # batch encode
        texts = [f" {c}={v}" for v in vals]
        # map back
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for v, ids in zip(vals, enc):
            frag[(c, v)] = ids
    return frag


def build_orders(n_fields: int, seed: int = 2026):
    rng = np.random.default_rng(seed)
    pi0 = rng.permutation(n_fields)
    pi1 = rng.permutation(n_fields)
    return [pi0, pi0[::-1].copy(), pi1, pi1[::-1].copy()]


def assemble_ids(row_vals, cols, order, frag, bos_id, y_id, max_len=128):
    ids = [bos_id, y_id]
    for j in order:
        c = cols[j]
        ids.extend(frag[(c, str(row_vals[j]))])
        if len(ids) >= max_len:
            break
    return ids[:max_len]


def pad_batch_ids(seqs, pad_id, multiple=16):
    m = max(len(s) for s in seqs)
    m = ((m + multiple - 1) // multiple) * multiple
    m = min(m, 128)
    B = len(seqs)
    out = np.full((B, m), pad_id, np.int64)
    attn = np.zeros((B, m), np.int64)
    for i, s in enumerate(seqs):
        n = min(len(s), m)
        out[i, :n] = s[:n]
        attn[i, :n] = 1
    return out, attn


@torch.no_grad()
def score_ll_masked(model, ids, attn, field_start=1):
    """Sum logp of field tokens: targets at positions >= field_start (skip bos->y)."""
    ids_t = torch.as_tensor(ids, device="cuda")
    attn_t = torch.as_tensor(attn, device="cuda")
    out = model(input_ids=ids_t, attention_mask=attn_t)
    logits = out.logits.float()  # fp32 for LL precision
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids_t[:, 1:]
    # gather
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    # mask: valid attn on target AND position index+1 >= field_start+1 → pos>=field_start for predicting next
    # ids positions: 0=bos, 1=y, 2..=fields. We want loss on predicting tokens at pos>=2 from pos>=1,
    # i.e. target indices j where j >= 1 (0-based in tgt which aligns to ids[1:]).
    # Skip tgt index 0 (bos->y). Keep tgt index >= 1 and attn on that target token.
    pos = torch.arange(tgt.shape[1], device="cuda").unsqueeze(0).expand_as(tgt)
    mask = (attn_t[:, 1:] == 1) & (pos >= 1) & (tgt != model.config.eos_token_id if False else True)
    # also ignore pad
    pad_id = 50256  # gpt2 eos used as pad often; we use eos as pad
    mask = mask & (tgt != pad_id)
    tok_lp = tok_lp.masked_fill(~mask, 0.0)
    return tok_lp.sum(dim=1).detach().cpu().numpy()


def run_gpt2_storyteller(tr, te, y, folds, fold_id, orig, vram_gb: float):
    from transformers import AutoTokenizer, GPT2LMHeadModel

    os.environ["USE_COMPILE"] = "0"
    f_mask = folds == fold_id
    tr_m = ~f_mask
    tr_t, te_t, orig_t, text_cols = build_storyteller_tables(tr, te, orig, folds, fold_id)
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    true_id, false_id = verify_true_false_tokens(tok)
    bos_id = tok.bos_token_id or tok.eos_token_id
    pad_id = tok.pad_token_id
    log(f"[gpt2] True={true_id} False={false_id} bos={bos_id} pad={pad_id}")

    frag = pretok_column_fragments(tok, [tr_t, te_t, orig_t], text_cols)
    orders = build_orders(len(text_cols), seed=2026)
    log(f"[gpt2] cols={len(text_cols)} frag_keys={len(frag)} orders={len(orders)}")

    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", attn_implementation="sdpa", torch_dtype=torch.bfloat16
    ).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    def make_train_seqs(df, yarr, n_max=None, lr_scale=1.0):
        idx = np.arange(len(df))
        if n_max and len(idx) > n_max:
            idx = np.random.default_rng(0).choice(idx, size=n_max, replace=False)
        return idx

    def train_epoch(df, yarr, indices, lr, bs=128, steps_log=50):
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        rng = np.random.default_rng(int(time.time()) % 10_000)
        n = len(indices)
        order_i = rng.permutation(n)
        step = 0
        for s0 in range(0, n, bs):
            batch_idx = indices[order_i[s0 : s0 + bs]]
            seqs = []
            for ii in batch_idx:
                row = df.iloc[int(ii)]
                vals = [row[c] for c in text_cols]
                od = rng.integers(0, len(orders))
                y_id = true_id if int(yarr[int(ii)]) == 1 else false_id
                seqs.append(assemble_ids(vals, text_cols, orders[od], frag, bos_id, y_id))
            ids, attn = pad_batch_ids(seqs, pad_id)
            ids_t = torch.as_tensor(ids, device="cuda")
            attn_t = torch.as_tensor(attn, device="cuda")
            out = model(input_ids=ids_t, attention_mask=attn_t)
            logits = out.logits
            # CE on field tokens only (tgt pos >= 1)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = ids_t[:, 1:].contiguous()
            pos = torch.arange(shift_labels.shape[1], device="cuda").unsqueeze(0)
            mask = (attn_t[:, 1:] == 1) & (pos >= 1)
            shift_labels = shift_labels.masked_fill(~mask, -100)
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % steps_log == 0:
                log(f"[gpt2] train step={step} loss={float(loss):.4f}")
        return step

    # 10 ep orig20k @ 1e-4
    orig_y = orig["is_fraud"].to_numpy(int)
    assert len(orig_y) == len(orig_t)
    oi = np.arange(len(orig_t))
    log("[gpt2] phase A: 10 ep orig20k")
    for ep in range(10):
        train_epoch(orig_t, orig_y, oi, lr=1e-4, bs=128)
        log(f"[gpt2] orig ep {ep+1}/10")

    # 1 full ep fold train
    tr_idx = np.where(tr_m)[0]
    log(f"[gpt2] phase B: 1 ep fold train n={len(tr_idx)}")
    # cosine via single epoch mid lr 8e-5
    train_epoch(tr_t, y, tr_idx, lr=8e-5, bs=128, steps_log=200)

    # rolling fp16 model-only ckpt
    ckpt = OUT / "model_rolling.pt"
    torch.save({k: v.detach().half().cpu() for k, v in model.state_dict().items()}, ckpt)
    log(f"[gpt2] wrote {ckpt}")

    model.eval()
    bs_score = 512 if vram_gb >= 40 else 256
    # queries: va + test
    va_idx = np.where(f_mask)[0]
    q_dfs = [("va", tr_t.iloc[va_idx].reset_index(drop=True)), ("te", te_t.reset_index(drop=True))]
    ll_parts = []
    for tag, qdf in q_dfs:
        N = len(qdf)
        ll = np.zeros((N, 4, 2), np.float64)
        log(f"[gpt2] score {tag} N={N} bs={bs_score}")
        for lab, y_id in enumerate([false_id, true_id]):
            for oi, order in enumerate(orders):
                for s0 in range(0, N, bs_score):
                    sl = slice(s0, min(s0 + bs_score, N))
                    seqs = []
                    for i in range(sl.start, sl.stop):
                        row = qdf.iloc[i]
                        vals = [row[c] for c in text_cols]
                        seqs.append(assemble_ids(vals, text_cols, order, frag, bos_id, y_id))
                    ids, attn = pad_batch_ids(seqs, pad_id)
                    ll[sl, oi, lab] = score_ll_masked(model, ids, attn, field_start=1)
                log(f"[gpt2] {tag} lab={lab} order={oi+1}/4")
        ll_parts.append(ll)

    ll_va, ll_te = ll_parts
    np.save(OUT / f"ll_raw_f{fold_id}.npy", np.concatenate([ll_va, ll_te], axis=0))

    def reducers(ll):
        # ll: (N,4,2)
        ll0, ll1 = ll[:, :, 0], ll[:, :, 1]
        r_mean = (ll1 - ll0).mean(axis=1)
        r_lse = logsumexp(ll1, axis=1) - logsumexp(ll0, axis=1)
        r_rank = np.mean([L(ll1[:, k] - ll0[:, k]) for k in range(ll.shape[1])], axis=0)
        return {
            "R_mean_diff": prior_shift(expit(unit(r_mean))),
            "R_deotte_lse": prior_shift(expit(unit(r_lse))),
            "R_ranklogit": prior_shift(expit(unit(r_rank))),
        }

    return reducers(ll_va), reducers(ll_te), ll_va, ll_te


def cuda_preflight(vram_gb: float) -> dict:
    """200-row CUDA smoke < 60s for TabPFN, TabICL, GPT2."""
    assert torch.cuda.is_available(), "CUDA required — no CPU stubs"
    t0 = time.time()
    status = {"tabpfn": None, "tabicl": None, "gpt2": None}
    X = np.random.randn(200, 8).astype(np.float32)
    y = np.random.randint(0, 2, 200)

    # TabPFN
    try:
        from tabpfn import TabPFNClassifier

        clf = TabPFNClassifier(device="cuda", ignore_pretraining_limits=True, n_estimators=1, random_state=0)
        clf.fit(X[:150], y[:150])
        _ = clf.predict_proba(X[150:])
        del clf
        cuda_free()
        status["tabpfn"] = "ok"
    except Exception as e:
        msg = str(e)
        ename = type(e).__name__
        if (
            "401" in msg
            or "Unauthorized" in msg
            or "gated" in msg.lower()
            or "huggingface" in msg.lower()
            or "License" in ename
            or "TABPFN_TOKEN" in msg
            or "license acceptance" in msg.lower()
        ):
            log(f"[preflight] TABPFN_AUTH_SKIP: {ename}: {e}")
            status["tabpfn"] = "TABPFN_AUTH_SKIP"
        else:
            log(f"[preflight] TabPFN fail: {traceback.format_exc()}")
            status["tabpfn"] = f"fail:{ename}"

    # TabICL
    try:
        from tabicl import TabICLClassifier

        clf = TabICLClassifier(n_estimators=1, batch_size=1, device="cuda", random_state=0)
        with torch.amp.autocast("cuda", enabled=False):
            clf.fit(X[:150], y[:150])
            _ = clf.predict_proba(X[150:])
        del clf
        cuda_free()
        status["tabicl"] = "ok"
    except Exception as e:
        log(f"[preflight] TabICL fail: {traceback.format_exc()}")
        status["tabicl"] = f"fail:{type(e).__name__}"

    # GPT2
    try:
        from transformers import GPT2LMHeadModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        true_id, false_id = verify_true_false_tokens(tok)
        model = GPT2LMHeadModel.from_pretrained(
            "gpt2", attn_implementation="sdpa", torch_dtype=torch.bfloat16
        ).cuda()
        ids = torch.randint(0, 100, (4, 32), device="cuda")
        _ = model(input_ids=ids)
        del model
        cuda_free()
        status["gpt2"] = f"ok True={true_id} False={false_id}"
    except Exception as e:
        log(f"[preflight] GPT2 fail: {traceback.format_exc()}")
        status["gpt2"] = f"fail:{type(e).__name__}"

    elapsed = time.time() - t0
    log(f"[preflight] {status} elapsed={elapsed:.1f}s vram={vram_gb:.1f}GB")
    assert elapsed < 90, "preflight too slow"
    tfm_ok = status["tabicl"] == "ok" or status["tabpfn"] == "ok"
    gpt_ok = bool(status["gpt2"]) and str(status["gpt2"]).startswith("ok")
    assert tfm_ok or gpt_ok, f"need TFM or GPT2 preflight ok; got {status}"
    if not gpt_ok:
        log(f"[preflight] GPT2 unavailable — continuing with TFM-only canaries: {status['gpt2']}")
    if status.get("tabpfn") != "ok":
        log(f"[preflight] TabPFN skipped/failed: {status.get('tabpfn')}")
    return status


def run_fold(fold_id: int, canaries: set, pre: dict, tr, te, y, folds, orig, baselines):
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    X, Xt, names, cat_idx, f_mask, tr_m = build_tfm_features(tr, te, folds, fold_id)
    n_va = int(f_mask.sum())
    Xq = np.concatenate([X[f_mask], Xt], axis=0)
    y_va = y[f_mask]
    z_fb, z_tw, z_fb_te, B_oof, B_te = baselines
    z_fb_va, z_tw_va = z_fb[f_mask], z_tw[f_mask]
    B_oof_va = B_oof[f_mask]

    fold_out = OUT / f"fold_{fold_id}"
    fold_out.mkdir(parents=True, exist_ok=True)
    gates = {}

    p_pfn = p_icl = None
    if "tabpfn" in canaries and pre.get("tabpfn") == "ok":
        try:
            p_all = run_tabpfn(X, y, tr_m, Xq, cat_idx, vram_gb)
            p_pfn_va, p_pfn_te = p_all[:n_va], p_all[n_va:]
            save_finite(fold_out / "tabpfn_oof.npy", p_pfn_va, (n_va,))
            save_finite(fold_out / "tabpfn_test.npy", p_pfn_te, (N_TEST,))
            p_pfn = (p_pfn_va, p_pfn_te)
            if fold_id == 0:
                gates["tabpfn"] = gate_canary(
                    "tabpfn", p_pfn_va, p_pfn_te, y_va, z_fb_va, z_tw_va, z_fb_te, B_oof_va, B_te
                )
        except Exception as e:
            log(f"[tabpfn] TABPFN_AUTH_SKIP or fail: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                log("[tabpfn] TABPFN_AUTH_SKIP")
    elif "tabpfn" in canaries:
        log(f"[tabpfn] skipped preflight={pre.get('tabpfn')}")

    if "tabicl" in canaries and pre.get("tabicl") == "ok":
        p_all = run_tabicl(X, y, tr_m, Xq, cat_idx, vram_gb, n_va)
        p_icl_va, p_icl_te = p_all[:n_va], p_all[n_va:]
        save_finite(fold_out / "tabicl_oof.npy", p_icl_va, (n_va,))
        save_finite(fold_out / "tabicl_test.npy", p_icl_te, (N_TEST,))
        p_icl = (p_icl_va, p_icl_te)
        if fold_id == 0:
            gates["tabicl"] = gate_canary(
                "tabicl", p_icl_va, p_icl_te, y_va, z_fb_va, z_tw_va, z_fb_te, B_oof_va, B_te
            )

    if p_pfn is not None and p_icl is not None and fold_id == 0:
        blend_va = expit(0.5 * L(p_pfn[0]) + 0.5 * L(p_icl[0]))
        blend_te = expit(0.5 * L(p_pfn[1]) + 0.5 * L(p_icl[1]))
        save_finite(fold_out / "tfm_blend_oof.npy", blend_va, (n_va,))
        save_finite(fold_out / "tfm_blend_test.npy", blend_te, (N_TEST,))
        gates["tfm_blend"] = gate_canary(
            "tfm_blend", blend_va, blend_te, y_va, z_fb_va, z_tw_va, z_fb_te, B_oof_va, B_te
        )

    if "gpt2" in canaries:
        red_va, red_te, _, _ = run_gpt2_storyteller(tr, te, y, folds, fold_id, orig, vram_gb)
        for rname in red_va:
            save_finite(fold_out / f"gpt2_{rname}_oof.npy", red_va[rname], (n_va,))
            save_finite(fold_out / f"gpt2_{rname}_test.npy", red_te[rname], (N_TEST,))
            if fold_id == 0:
                gates[f"gpt2_{rname}"] = gate_canary(
                    f"gpt2_{rname}",
                    red_va[rname],
                    red_te[rname],
                    y_va,
                    z_fb_va,
                    z_tw_va,
                    z_fb_te,
                    B_oof_va,
                    B_te,
                )

    (fold_out / "gates.json").write_text(json.dumps(gates, indent=2))
    # promote list
    promote = [k for k, v in gates.items() if v.get("promote")]
    (fold_out / "PROMOTE.json").write_text(json.dumps({"promote": promote, "gates": gates}, indent=2))
    log(f"[fold {fold_id}] promote={promote}")
    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    ap.add_argument("--canaries", default="tabpfn,tabicl,gpt2")
    args = ap.parse_args()
    os.environ.setdefault("NO_SUBMIT", "True")
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"[v37] NO_SUBMIT={os.environ.get('NO_SUBMIT')} OUT={OUT} REPO={REPO}")
    assert torch.cuda.is_available(), "CUDA required"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"[v37] GPU={torch.cuda.get_device_name(0)} VRAM={vram_gb:.1f}GB")
    assert vram_gb >= 22.0, f"VRAM {vram_gb}<24GB rejected"

    pre = cuda_preflight(vram_gb)
    (OUT / "preflight.json").write_text(json.dumps(pre, indent=2))

    tr, te, y, folds, orig = load_data()
    baselines = (
        np.load(PRED_LOCAL / "z_fullbank_oof.npy").ravel()[: len(y)],
        np.load(PRED_LOCAL / "z_ATH_twin_oof.npy").ravel()[: len(y)],
        np.load(PRED_LOCAL / "z_fullbank_test.npy").ravel()[:N_TEST],
        np.load(PRED_LOCAL / "B_clean_oof.npy"),
        np.load(PRED_LOCAL / "B_clean_test.npy"),
    )
    canaries = set(c.strip() for c in args.canaries.split(",") if c.strip())
    fold_ids = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    all_gates = {}
    for fid in fold_ids:
        log(f"========== FOLD {fid} ==========")
        all_gates[str(fid)] = run_fold(fid, canaries, pre, tr, te, y, folds, orig, baselines)
    (OUT / "all_gates.json").write_text(json.dumps(all_gates, indent=2))
    (OUT / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    log("[v37] DONE — controller may rsync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
