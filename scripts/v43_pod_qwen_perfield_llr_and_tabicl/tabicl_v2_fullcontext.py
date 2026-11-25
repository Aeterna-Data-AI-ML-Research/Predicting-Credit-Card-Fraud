#!/usr/bin/env python3
"""Module B — TabICLv2 zero-shot Fold-0 canary (full-context chunks).

Uses `tabicl.TabICLClassifier` with stratified 100k–200k context chunks +
feature-permutation views in bf16. ~5 min on GPU.

Set V43_LAUNCH=1 only after explicit authorization. NO_SUBMIT forever.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
assert NO_SUBMIT
T0 = time.time()
FEATS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
CTX = int(os.environ.get("V43_TABICL_CTX", "150000"))
N_EST = int(os.environ.get("V43_TABICL_EST", "4"))
N_FEAT_PERM = int(os.environ.get("V43_TABICL_FEAT_PERM", "2"))


def log(m: str) -> None:
    print(f"[{time.time()-T0:7.1f}s][B] {m}", flush=True)


def encode_frame(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATS].copy()
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = pd.Categorical(X[c]).codes.astype(np.float32)
        else:
            X[c] = X[c].astype(np.float32)
    return X.to_numpy(np.float32)


def main():
    if os.environ.get("V43_LAUNCH", "0") != "1":
        log("DRY-RUN. Set V43_LAUNCH=1 to run TabICL on GPU. Exit.")
        print(json.dumps({"status": "dry_run", "module": "tabicl", "NO_SUBMIT": True}))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        from tabicl import TabICLClassifier
    except ImportError as e:
        raise SystemExit(f"tabicl not installed on pod: {e}")

    tr = pd.read_csv(args.data / "train.csv")
    te = pd.read_csv(args.data / "test.csv")
    y = tr["is_fraud"].to_numpy(int)
    X = encode_frame(tr)
    Xt = encode_frame(te)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    tr_ix, va_ix = list(skf.split(np.zeros(len(y)), y))[args.fold]
    log(f"fold={args.fold} ctx_pool={len(tr_ix)} va={len(va_ix)} test={len(Xt)}")

    rng = np.random.default_rng(42 + args.fold)
    preds_va = []
    preds_te = []
    for e in range(N_EST):
        # stratified context subsample
        pos = tr_ix[y[tr_ix] == 1]
        neg = tr_ix[y[tr_ix] == 0]
        n_ctx = min(CTX, len(tr_ix))
        n_pos = min(len(pos), max(int(n_ctx * y[tr_ix].mean()), 1))
        n_neg = n_ctx - n_pos
        ctx = np.concatenate([
            rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
            rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
        ])
        for fp in range(N_FEAT_PERM):
            feat_order = rng.permutation(X.shape[1]) if fp > 0 else np.arange(X.shape[1])
            clf = TabICLClassifier(
                n_estimators=1,
                random_state=1000 * e + fp,
            )
            log(f"fit est={e} feat_perm={fp} ctx={len(ctx)}")
            clf.fit(X[ctx][:, feat_order], y[ctx])
            preds_va.append(clf.predict_proba(X[va_ix][:, feat_order])[:, 1])
            preds_te.append(clf.predict_proba(Xt[:, feat_order])[:, 1])

    va = np.mean(preds_va, 0)
    te_p = np.mean(preds_te, 0)
    np.save(args.out / f"tabicl_f{args.fold}_oof.npy", va.astype(np.float32))
    np.save(args.out / f"tabicl_f{args.fold}_test.npy", te_p.astype(np.float32))
    meta = {
        "fold": args.fold, "ctx": CTX, "n_est": N_EST, "n_feat_perm": N_FEAT_PERM,
        "NO_SUBMIT": True, "shapes": {"oof": list(va.shape), "test": list(te_p.shape)},
    }
    (args.out / f"tabicl_f{args.fold}_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"done → {args.out}")


if __name__ == "__main__":
    main()
