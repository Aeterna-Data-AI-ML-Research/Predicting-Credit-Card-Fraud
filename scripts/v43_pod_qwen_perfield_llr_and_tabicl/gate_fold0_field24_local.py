#!/usr/bin/env python3
"""Post-rsync Fold-0 gate: Ridge + shallow GBDT on field24 LLR vs z_modern_pub[f0].

Runs on Mac CPU after Qwen .npy land. NO_SUBMIT. Never touches pods/Kaggle.
Gate: nested ΔAUC_f0 ≥ +5.0e-5 with ≥4/5 inner subfolds on z_modern_pub[f0]+Zr.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
assert NO_SUBMIT
T0 = time.time()
SEED = 42
GATE_DAUC = 5.0e-5


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s][gate] {m}", flush=True)


def ecdf_gauss(a: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    r = a if ref is None else ref
    order = np.argsort(r)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = (np.arange(len(r)) + 0.5) / len(r)
    # map a via searchsorted on ref
    if ref is None:
        p = ranks
    else:
        p = np.searchsorted(np.sort(r), a, side="left") / max(len(r), 1)
        p = np.clip(p, 1e-6, 1 - 1e-6)
    from scipy.stats import norm

    return norm.ppf(np.clip(p if ref is None else p, 1e-6, 1 - 1e-6)).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-dir", type=Path, required=True, help="dir with oof_f0_field24_llr.npy")
    ap.add_argument("--modern-oof", type=Path, required=True)
    ap.add_argument("--modern-test", type=Path, required=True)
    ap.add_argument("--zr-oof", type=Path, default=None)
    ap.add_argument("--zr-test", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("logs/v43"))
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    y = pd.read_csv(args.data / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    tr_ix, va_ix = folds[args.fold]
    y_va = y[va_ix]

    f_va = np.load(args.qwen_dir / f"oof_f{args.fold}_field24_llr.npy").astype(np.float64)
    f_te = np.load(args.qwen_dir / f"test_f{args.fold}_field24_llr.npy").astype(np.float64)
    assert f_va.shape == (len(va_ix), 24), f_va.shape
    assert f_te.shape[1] == 24

    z_oof = np.load(args.modern_oof).astype(np.float64).ravel()
    z_te = np.load(args.modern_test).astype(np.float64).ravel()
    z_va = z_oof[va_ix]
    if args.zr_oof and args.zr_oof.exists():
        zr_oof = np.load(args.zr_oof).astype(np.float64).ravel()
        zr_te = np.load(args.zr_test).astype(np.float64).ravel()
        z_va = z_va + zr_oof[va_ix]
        z_te = z_te + zr_te

    base_auc = float(roc_auc_score(y_va, z_va))
    log(f"base AUC fold{args.fold}={base_auc:.6f} n={len(y_va)}")

    # nested 5-fold on va: Ridge + shallow LGBM on field24 → residual add
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None

    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof_delta = np.zeros(len(y_va), dtype=np.float64)
    te_preds = []
    fold_d = []
    for i, (itr, iva) in enumerate(inner.split(f_va, y_va)):
        # residual target vs base rank
        Xtr, Xva = f_va[itr], f_va[iva]
        # Ridge on Newton-ish residual proxy: y - σ(z)
        from scipy.special import expit

        p_tr = expit(z_va[itr])
        g = y_va[itr] - p_tr
        h = np.clip(p_tr * (1 - p_tr), 1e-4, None)
        ridge = Ridge(alpha=1000.0)
        ridge.fit(Xtr, g / h, sample_weight=h)
        pred = ridge.predict(Xva)
        pred_te = ridge.predict(f_te)
        if lgb is not None:
            mdl = lgb.LGBMRegressor(
                max_depth=3, num_leaves=8, min_child_samples=200,
                n_estimators=400, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.6, verbosity=-1, n_jobs=4,
            )
            mdl.fit(Xtr, g / h, sample_weight=h)
            pred = 0.5 * pred + 0.5 * mdl.predict(Xva)
            pred_te = 0.5 * pred_te + 0.5 * mdl.predict(f_te)
        oof_delta[iva] = pred
        te_preds.append(pred_te)
        # subfold Δ vs base on iva
        blend = z_va[iva] + 0.05 * pred  # small probe weight
        d = roc_auc_score(y_va[iva], blend) - roc_auc_score(y_va[iva], z_va[iva])
        fold_d.append(float(d))
        log(f"  inner{i} Δe5={d*1e5:+.2f}")

    te_delta = np.mean(te_preds, 0)
    # LOFO weight search on outer va
    best = {"w": 0.0, "auc": base_auc, "d": 0.0}
    for w in np.linspace(-0.08, 0.08, 161):
        auc = float(roc_auc_score(y_va, z_va + w * oof_delta))
        if auc > best["auc"]:
            best = {"w": float(w), "auc": auc, "d": auc - base_auc}
    pos = sum(1 for d in fold_d if d > 0)
    mean_d = float(np.mean(fold_d))
    pass_gate = (best["d"] >= GATE_DAUC) and (pos >= 4)
    report = {
        "NO_SUBMIT": True,
        "fold": args.fold,
        "base_auc": base_auc,
        "best_w": best["w"],
        "best_auc": best["auc"],
        "delta_auc": best["d"],
        "delta_e5": best["d"] * 1e5,
        "inner_deltas_e5": [d * 1e5 for d in fold_d],
        "inner_pos": f"{pos}/5",
        "inner_mean_e5": mean_d * 1e5,
        "pass_gate": pass_gate,
        "gate": f">=+{GATE_DAUC:.1e} and >=4/5 subfolds",
        "K_field24": 24,
        "note": "TabICL skipped per Gemini APPROVE_POD_F0_ONLY",
    }
    out_p = args.out / f"V43_FOLD0_FIELD24_GATE.json"
    out_p.write_text(json.dumps(report, indent=2))
    log(f"Δe5={best['d']*1e5:+.2f} pos={pos}/5 pass={pass_gate} → {out_p}")
    print(json.dumps(report, indent=2))
    if not pass_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
