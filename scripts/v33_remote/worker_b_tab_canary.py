#!/usr/bin/env python3
"""Worker B — TabICL + TabPFN Fold-0 canary (va0 only). CUDA device via CUDA_VISIBLE_DEVICES."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

T0 = time.time()
BUNDLE = Path(os.environ.get("V33_BUNDLE", "/workspace/bundle"))
OUT = Path(os.environ.get("V33_OUT", "/workspace/canary"))
OUT.mkdir(parents=True, exist_ok=True)
N_BAGS_ICL = int(os.environ.get("V33_ICL_BAGS", "16"))
N_BAGS_PFN = int(os.environ.get("V33_PFN_BAGS", "16"))
CHUNK = 4096
PI0 = 0.016699525687182123


def log(m):
    print(f"[{time.time()-T0:7.1f}s][B] {m}", flush=True)


def build_X(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    num = [
        "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
        "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
        "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
    ]
    cats = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
    bools = [
        "is_foreign_transaction", "is_new_merchant", "used_vpn",
        "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
    ]
    Xn = df[num].to_numpy(np.float32)
    # engineered
    bal_cents = np.round(df["account_balance_usd"].to_numpy(float) * 100).astype(np.float32)
    amt_cents = np.round(df["amount_usd"].to_numpy(float) * 100).astype(np.float32)
    hours_tenths = np.round(df["hours_since_last_txn"].to_numpy(float) * 10).astype(np.float32)
    vel_tenths = np.round(df["velocity_score"].to_numpy(float) * 10).astype(np.float32)
    # count of exact balance in all rows passed later — placeholder 0 here, filled outside
    cnt_bal = np.zeros(len(df), np.float32)
    bits = np.zeros(len(df), np.int64)
    for i, b in enumerate(bools):
        bits |= df[b].astype(int).to_numpy(np.int64) << i
    bits |= df["cvv_retry_count"].clip(0, 7).to_numpy(np.int64) << 6
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xc = enc.fit_transform(df[cats].astype(str)).astype(np.float32)
    X = np.column_stack([Xn, bal_cents, amt_cents, hours_tenths, vel_tenths, cnt_bal, bits.astype(np.float32), Xc])
    names = num + ["bal_cents", "amt_cents", "hours_tenths", "vel_tenths", "log1p_cnt_bal", "flag_bitmask", *cats]
    return X, names, enc, bal_cents


def prior_correct(p, pi_bag, pi0=PI0):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return expit(logit(p) - np.log((pi_bag * (1 - pi0)) / (pi0 * (1 - pi_bag) + 1e-12)))


def score_chunks(clf, Xq, predict_proba):
    outs = []
    for i in range(0, len(Xq), CHUNK):
        outs.append(predict_proba(Xq[i : i + CHUNK])[:, 1])
    return np.concatenate(outs)


def main():
    log("load data")
    tr = pd.read_csv(BUNDLE / "train.csv.gz")
    fold = np.load(BUNDLE / "fold_id.npy")
    y = tr["is_fraud"].to_numpy(float)
    wide = np.load(BUNDLE / "wide_lm_oof.npy")
    assert len(fold) == len(tr) == len(wide)
    va = fold == 0
    trm = ~va
    log(f"tr0={trm.sum()} va0={va.sum()} wide_va_auc={roc_auc_score(y[va], wide[va]):.6f}")

    X_all, names, enc, bal_cents = build_X(tr)
    # fill log1p cnt balance using train∪ later — use full-train counts
    from collections import Counter
    cnt = Counter(np.round(tr["account_balance_usd"].to_numpy(float), 6).tolist())
    key = np.round(tr["account_balance_usd"].to_numpy(float), 6)
    X_all[:, names.index("log1p_cnt_bal")] = np.log1p(np.array([cnt[k] for k in key], float)).astype(np.float32)

    Xtr, ytr = X_all[trm], y[trm]
    Xva, yva = X_all[va], y[va]
    pos_idx = np.where(ytr == 1)[0]
    neg_idx = np.where(ytr == 0)[0]
    rng = np.random.default_rng(42)

    # ---- TabICL ----
    tabicl_ok = False
    icl_preds = []
    try:
        from tabicl import TabICLClassifier
        tabicl_ok = True
    except Exception as e:
        log(f"TabICL import failed: {e}")

    if tabicl_ok:
        log(f"TabICL {N_BAGS_ICL} bags…")
        for b in range(N_BAGS_ICL):
            n_pos = min(7013, len(pos_idx))
            n_neg = 30000 - n_pos
            p = rng.choice(pos_idx, size=n_pos, replace=False)
            # rotated negatives
            start = (b * n_neg) % max(1, len(neg_idx) - n_neg)
            n = neg_idx[start : start + n_neg]
            if len(n) < n_neg:
                n = rng.choice(neg_idx, size=n_neg, replace=False)
            ctx = np.concatenate([p, n])
            rng.shuffle(ctx)
            pi_bag = float(ytr[ctx].mean())
            clf = TabICLClassifier()
            clf.fit(Xtr[ctx], ytr[ctx])
            raw = score_chunks(clf, Xva, clf.predict_proba)
            icl_preds.append(prior_correct(raw, pi_bag))
            log(f"  ICL bag {b+1}/{N_BAGS_ICL} pi={pi_bag:.4f} auc={roc_auc_score(yva, icl_preds[-1]):.5f}")
        tabicl = np.mean(icl_preds, axis=0)
        np.save(OUT / "tabicl_va0.npy", tabicl.astype(np.float32))
        log(f"TabICL mean AUC={roc_auc_score(yva, tabicl):.6f}")
    else:
        tabicl = None

    # ---- TabPFN ----
    pfn_ok = False
    pfn_preds = []
    try:
        from tabpfn import TabPFNClassifier
        pfn_ok = True
    except Exception as e:
        log(f"TabPFN import failed: {e}")

    if pfn_ok:
        log(f"TabPFN {N_BAGS_PFN} bags…")
        try:
            for b in range(N_BAGS_PFN):
                n_pos = min(2000, len(pos_idx))
                n_neg = 10000 - n_pos
                p = rng.choice(pos_idx, size=n_pos, replace=False)
                start = (b * n_neg) % max(1, len(neg_idx) - n_neg)
                n = neg_idx[start : start + n_neg]
                if len(n) < n_neg:
                    n = rng.choice(neg_idx, size=n_neg, replace=False)
                ctx = np.concatenate([p, n])
                rng.shuffle(ctx)
                pi_bag = float(ytr[ctx].mean())
                try:
                    clf = TabPFNClassifier(ignore_pretraining_limits=True, device="cuda")
                except TypeError:
                    clf = TabPFNClassifier(device="cuda")
                clf.fit(Xtr[ctx], ytr[ctx])
                raw = score_chunks(clf, Xva, clf.predict_proba)
                pfn_preds.append(prior_correct(raw, pi_bag))
                log(f"  PFN bag {b+1}/{N_BAGS_PFN} pi={pi_bag:.4f} auc={roc_auc_score(yva, pfn_preds[-1]):.5f}")
            tabpfn = np.mean(pfn_preds, axis=0)
            np.save(OUT / "tabpfn_va0.npy", tabpfn.astype(np.float32))
            log(f"TabPFN mean AUC={roc_auc_score(yva, tabpfn):.6f}")
        except Exception as e:
            log(f"TabPFN failed (continuing with TabICL only): {e}")
            tabpfn = None
            pfn_preds = []
    else:
        tabpfn = None

    meta = {
        "tabicl_auc": float(roc_auc_score(yva, tabicl)) if tabicl is not None else None,
        "tabpfn_auc": float(roc_auc_score(yva, tabpfn)) if tabpfn is not None else None,
        "wide_va_auc": float(roc_auc_score(yva, wide[va])),
        "n_bags_icl": N_BAGS_ICL if tabicl is not None else 0,
        "n_bags_pfn": N_BAGS_PFN if tabpfn is not None else 0,
    }
    (OUT / "worker_b_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"DONE meta={meta}")


if __name__ == "__main__":
    main()
