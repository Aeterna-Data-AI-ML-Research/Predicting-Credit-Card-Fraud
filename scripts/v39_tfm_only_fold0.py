#!/usr/bin/env python3
"""V39 Lever#2 — Fold-0 TabPFN + TabICLv2 ONLY (no LLM). NO_SUBMIT=True.

Writes:
  tabpfn_f0_va_test.npy  (n_va+n_te,)
  tabicl_f0_va_test.npy
  tfm_blend_f0_va_test.npy
  fold0_tfm_result.json
"""
from __future__ import annotations

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
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
DATA = Path(os.environ.get("V39_DATA", "/workspace/data"))
FOLDS = Path(os.environ.get("V39_FOLDS", "/workspace/folds/folds.csv"))
PREDS = Path(os.environ.get("V39_PREDS", "/workspace/preds"))
OUT = Path(os.environ.get("V39_OUT", "/workspace/out"))
PREV = 0.0167
N_TEST = 299_984
MAX_CAT = 100
SOLO_MIN = 0.9465
DISP_LO, DISP_HI = 0.88, 1.12
R2_MAX = 0.80
LIFT_MIN = 0.40e-4

CONT13 = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
]
CAT11 = [
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
    "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
FLAGS = CAT11[5:]


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / max(len(v), 1)


def L(v):
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6))


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def auc(y, p):
    return float(roc_auc_score(y, np.asarray(p, float).ravel()))


def cuda_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prior_shift(p, target=PREV):
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


def mantissa(x):
    x = np.clip(np.asarray(x, float), 1e-12, None)
    return np.log10(x) - np.floor(np.log10(x))


def factorize_bounded(tr_s, te_s, max_levels=MAX_CAT):
    all_s = pd.concat([tr_s.astype(str), te_s.astype(str)], ignore_index=True)
    keep = set(all_s.value_counts().head(max_levels - 1).index)
    mapping = {v: i + 1 for i, v in enumerate(sorted(keep))}
    return (
        tr_s.astype(str).map(lambda x: mapping.get(x, 0)).to_numpy(np.int32),
        te_s.astype(str).map(lambda x: mapping.get(x, 0)).to_numpy(np.int32),
    )


def build_features(tr, te, folds, fold_id=0):
    f_mask = folds == fold_id
    tr_m = ~f_mask
    cols_tr, cols_te, names, cat_idx = [], [], [], []

    def add(name, a, b, cat=False):
        cols_tr.append(np.asarray(a, np.float32))
        cols_te.append(np.asarray(b, np.float32))
        names.append(name)
        if cat:
            cat_idx.append(len(names) - 1)

    for c in CONT13:
        add(c, tr[c], te[c])
    amt_tr, bal_tr = tr["amount_usd"].to_numpy(float), tr["account_balance_usd"].to_numpy(float)
    amt_te, bal_te = te["amount_usd"].to_numpy(float), te["account_balance_usd"].to_numpy(float)
    add("log1p_amt", np.log1p(amt_tr), np.log1p(amt_te))
    add("log1p_bal", np.log1p(bal_tr), np.log1p(bal_te))
    add("amt_over_bal", amt_tr / (bal_tr + 1), amt_te / (bal_te + 1))
    add("amt_mantissa", mantissa(amt_tr), mantissa(amt_te))
    add("bal_mantissa", mantissa(bal_tr), mantissa(bal_te))

    g_amt = tr.loc[tr_m, "amount_usd"].astype(str).value_counts()
    g_bal = tr.loc[tr_m, "account_balance_usd"].astype(str).value_counts()
    add("log1p_amt_freq_tr", np.log1p(tr["amount_usd"].astype(str).map(g_amt).fillna(0).astype(float)),
        np.log1p(te["amount_usd"].astype(str).map(g_amt).fillna(0).astype(float)))
    add("log1p_bal_freq_tr", np.log1p(tr["account_balance_usd"].astype(str).map(g_bal).fillna(0).astype(float)),
        np.log1p(te["account_balance_usd"].astype(str).map(g_bal).fillna(0).astype(float)))
    for c, nm in (("amount_usd", "amt"), ("account_balance_usd", "bal")):
        freq = pd.concat([tr[c].astype(str), te[c].astype(str)]).value_counts()
        fr = pd.concat([tr[c].astype(str), te[c].astype(str)]).map(freq).fillna(0).to_numpy(float)
        add(f"log1p_{nm}_freq_tt", np.log1p(fr[: len(tr)]), np.log1p(fr[len(tr) :]))

    for c in CAT11:
        a, b = factorize_bounded(tr[c], te[c])
        add(c, a, b, cat=True)
    add("amt_cents", (np.round(amt_tr * 100) % 100).astype(np.int32),
        (np.round(amt_te * 100) % 100).astype(np.int32), cat=True)
    add("bal_cents", (np.round(bal_tr * 100) % 100).astype(np.int32),
        (np.round(bal_te * 100) % 100).astype(np.int32), cat=True)
    for c, nm in (("amount_usd", "amt"), ("account_balance_usd", "bal")):
        freq = pd.concat([tr[c].astype(str), te[c].astype(str)]).value_counts()
        rank = pd.concat([tr[c].astype(str), te[c].astype(str)]).map(freq).fillna(0).to_numpy(float)
        edges = np.quantile(rank, np.linspace(0, 1, 33)[1:-1])
        bins = np.clip(np.digitize(rank, edges), 0, 31).astype(np.int32)
        add(f"{nm}_freq_bin32", bins[: len(tr)], bins[len(tr) :], cat=True)

    def flag64(df):
        v = np.zeros(len(df), np.int32)
        for i, c in enumerate(FLAGS[:6]):
            v |= (df[c].astype(int).to_numpy() & 1) << i
        return np.clip(v, 0, 63)

    add("flag64", flag64(tr), flag64(te), cat=True)
    X = np.column_stack(cols_tr).astype(np.float32)
    Xt = np.column_stack(cols_te).astype(np.float32)
    log(f"feat X={X.shape} Xt={Xt.shape} n_cat={len(cat_idx)}")
    return X, Xt, cat_idx, f_mask, tr_m


def bags_pos_neg(y_pool, n_bags, bag_n, n_pos_target, seed):
    rng = np.random.default_rng(seed)
    pos = np.where(y_pool == 1)[0]
    neg = np.where(y_pool == 0)[0]
    rng.shuffle(neg)
    n_pos = min(n_pos_target, len(pos), bag_n - 1)
    n_neg = bag_n - n_pos
    out = []
    for b in range(n_bags):
        chunk = neg[b * n_neg : (b + 1) * n_neg]
        if len(chunk) < n_neg:
            chunk = np.concatenate([chunk, rng.choice(neg, n_neg - len(chunk), replace=True)])
        ip = rng.choice(pos, n_pos, replace=len(pos) < n_pos)
        out.append(np.concatenate([ip, chunk]))
    return out


def predict_chunks(clf, Xq, chunk):
    outs = []
    for s in range(0, len(Xq), chunk):
        outs.append(np.asarray(clf.predict_proba(Xq[s : s + chunk]))[:, 1])
    return np.concatenate(outs)


def run_tabpfn(X, y, tr_m, Xq, cat_idx):
    from tabpfn import TabPFNClassifier
    import tabpfn as _tp

    y_pool, X_pool = y[tr_m], X[tr_m]
    bags = bags_pos_neg(y_pool, 8, 20_000, min(9351, int(y_pool.sum())), 2026)
    acc = np.zeros(len(Xq), float)
    ver = getattr(_tp, "__version__", "?")
    log(f"tabpfn version={ver}")
    for bi, idx in enumerate(bags):
        # Prefer v2-style API (2.0.x, no Prior Labs license gate); fall back to v3+.
        clf = None
        for kw in (
            dict(device="cuda", N_ensemble_configurations=4),
            dict(device="cuda", ignore_pretraining_limits=True, random_state=42 + bi,
                 n_estimators=4, categorical_features_indices=cat_idx or None),
        ):
            try:
                clf = TabPFNClassifier(**kw)
                break
            except TypeError:
                continue
        if clf is None:
            raise RuntimeError("TabPFNClassifier kwargs unsupported")
        clf.fit(X_pool[idx], y_pool[idx])
        acc += prior_shift(predict_chunks(clf, Xq, 20_000))
        del clf
        cuda_free()
        log(f"tabpfn bag {bi+1}/8 done")
    return acc / 8.0


def run_tabicl(X, y, tr_m, Xq, n_va):
    from tabicl import TabICLClassifier

    y_pool, X_pool = y[tr_m], X[tr_m]
    bags = bags_pos_neg(y_pool, 4, 20_000, min(9351, int(y_pool.sum())), 4242)
    X_va, X_te = Xq[:n_va], Xq[n_va:]
    acc_va = np.zeros(n_va, float)
    acc_te = np.zeros(len(X_te), float)
    for bi, idx in enumerate(bags):
        t0 = time.time()
        clf = TabICLClassifier(n_estimators=4, batch_size=1, kv_cache=True, device="cuda", random_state=7 + bi)
        with torch.amp.autocast("cuda", enabled=False):
            clf.fit(X_pool[idx], y_pool[idx])
            log(f"tabicl bag {bi+1}/4 fit {time.time()-t0:.1f}s")
            acc_va += prior_shift(predict_chunks(clf, X_va, 4096))
            acc_te += prior_shift(predict_chunks(clf, X_te, 4096))
        del clf
        cuda_free()
        log(f"tabicl bag {bi+1}/4 done wall={time.time()-t0:.1f}s")
    return np.concatenate([acc_va / 4.0, acc_te / 4.0])


def svd40(B):
    X = np.column_stack([unit(B[:, j]) for j in range(B.shape[1])])
    Xc = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[: min(40, X.shape[1])].T


def r2_span(u, B40):
    u = unit(u)
    r = Ridge(alpha=50.0, fit_intercept=False).fit(B40, u)
    pred = r.predict(B40)
    return float(1.0 - np.sum((u - pred) ** 2) / (np.sum((u - u.mean()) ** 2) + 1e-12))


def gate(name, p_va, p_te, y_va, z_fb, z_tw, z_fb_te, B_va):
    solo = auc(y_va, p_va)
    d_va = L(p_va) - L(z_fb)
    d_te = L(p_te) - L(z_fb_te)
    raw_disp = float(d_te.std() / (d_va.std() + 1e-12))
    # out-of-subfold lifts + r2 on Fold-0 VA residuals vs B_clean span
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    B40 = svd40(B_va)
    r2 = r2_span(d_va, B40)

    def lifts(z):
        # orth residual of L(p)-L(z) vs B40
        u = unit(L(p_va) - L(z))
        rr = Ridge(alpha=50.0, fit_intercept=False).fit(B40, u)
        feat = u - rr.predict(B40)
        d = []
        for tr, va in skf.split(np.zeros(len(y_va)), y_va):
            bw, ba = 0.0, -1.0
            for w in np.linspace(-0.08, 0.08, 33):
                a = auc(y_va[tr], L(z[tr]) + w * feat[tr])
                if a > ba:
                    ba, bw = a, float(w)
            d.append(auc(y_va[va], L(z[va]) + bw * feat[va]) - auc(y_va[va], z[va]))
        d = np.asarray(d)
        return {"mean_d": float(d.mean()), "n_pos": int((d > 0).sum()), "d": d.tolist()}

    g_fb, g_tw = lifts(z_fb), lifts(z_tw)
    promote = (
        solo >= SOLO_MIN
        and DISP_LO <= raw_disp <= DISP_HI
        and r2 < R2_MAX
        and g_fb["n_pos"] >= 4
        and g_tw["n_pos"] >= 4
        and g_fb["mean_d"] >= LIFT_MIN
        and g_tw["mean_d"] >= LIFT_MIN
    )
    out = {
        "name": name, "solo_f0": solo, "raw_disp": raw_disp, "r2_B40": r2,
        "g_fb": g_fb, "g_tw": g_tw, "promote": promote,
    }
    log(f"GATE {name}: solo={solo:.5f} disp={raw_disp:.3f} r2={r2:.3f} "
        f"fbΔ={g_fb['mean_d']:+.2e}({g_fb['n_pos']}/5) twΔ={g_tw['mean_d']:+.2e}({g_tw['n_pos']}/5) promote={promote}")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("=== V39 TFM-only Fold-0 | NO_SUBMIT | skip LLM ===")
    log(f"cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    y = tr["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS)["fold"].to_numpy(int)
    z_fb = np.load(PREDS / "z_fullbank_oof.npy").ravel()
    z_tw = np.load(PREDS / "z_ATH_twin_oof.npy").ravel()
    z_fb_te = np.load(PREDS / "z_fullbank_test.npy").ravel()
    B = np.load(PREDS / "B_clean_oof.npy")
    X, Xt, cat_idx, f_mask, tr_m = build_features(tr, te, folds, 0)
    n_va = int(f_mask.sum())
    Xq = np.vstack([X[f_mask], Xt])
    y_va = y[f_mask]
    z_fb_va, z_tw_va = z_fb[f_mask], z_tw[f_mask]
    B_va = B[f_mask]
    skip_icl = os.environ.get("V39_SKIP_TABICL", "0") == "1"
    skip_pfn = os.environ.get("V39_SKIP_TABPFN", "0") == "1"

    result = {"NO_SUBMIT": True, "gates": {}, "errors": {}}

    # TabPFN
    p_pfn = None
    if not skip_pfn:
        try:
            p_pfn = run_tabpfn(X, y, tr_m, Xq, cat_idx)
            np.save(OUT / "tabpfn_f0_va_test.npy", p_pfn.astype(np.float32))
            result["gates"]["tabpfn"] = gate(
                "tabpfn", p_pfn[:n_va], p_pfn[n_va:], y_va, z_fb_va, z_tw_va, z_fb_te, B_va
            )
        except Exception as e:
            log(f"tabpfn FAIL: {traceback.format_exc()}")
            result["errors"]["tabpfn"] = str(e)
            p_pfn = None
    else:
        pfn_path = OUT / "tabpfn_f0_va_test.npy"
        if pfn_path.exists():
            p_pfn = np.load(pfn_path).ravel()

    # TabICL
    p_icl = None
    if not skip_icl:
        try:
            p_icl = run_tabicl(X, y, tr_m, Xq, n_va)
            np.save(OUT / "tabicl_f0_va_test.npy", p_icl.astype(np.float32))
            result["gates"]["tabicl"] = gate(
                "tabicl", p_icl[:n_va], p_icl[n_va:], y_va, z_fb_va, z_tw_va, z_fb_te, B_va
            )
        except Exception as e:
            log(f"tabicl FAIL: {traceback.format_exc()}")
            result["errors"]["tabicl"] = str(e)
            p_icl = None
    else:
        icl_path = OUT / "tabicl_f0_va_test.npy"
        if icl_path.exists():
            p_icl = np.load(icl_path).ravel()
            result["gates"]["tabicl"] = gate(
                "tabicl", p_icl[:n_va], p_icl[n_va:], y_va, z_fb_va, z_tw_va, z_fb_te, B_va
            )

    if p_pfn is not None and p_icl is not None:
        blend = 0.5 * (p_pfn + p_icl)
        np.save(OUT / "tfm_blend_f0_va_test.npy", blend.astype(np.float32))
        result["gates"]["tfm_blend"] = gate(
            "tfm_blend", blend[:n_va], blend[n_va:], y_va, z_fb_va, z_tw_va, z_fb_te, B_va
        )

    result["promote_any"] = any(g.get("promote") for g in result["gates"].values())
    result["elapsed_s"] = time.time() - T0
    (OUT / "fold0_tfm_result.json").write_text(json.dumps(result, indent=2))
    log(f"DONE promote_any={result['promote_any']} elapsed={result['elapsed_s']:.1f}s")
    print(json.dumps({"promote_any": result["promote_any"], "gates": {k: v.get("promote") for k, v in result["gates"].items()}}, indent=2), flush=True)
    return 0 if result.get("promote_any") else 1


if __name__ == "__main__":
    raise SystemExit(main())
