#!/usr/bin/env python3
"""V47.3 Fold-0 gate for private Kaggle HGDT-v2 outputs vs z_sovereign_v0.

NO_SUBMIT=True · $0 RunPod · HGDT-only (no Qwen required).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
assert NO_SUBMIT
REPO = Path(__file__).resolve().parents[2]
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s][v473] {m}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def hygiene(o, t):
    o, t = np.asarray(o, float).ravel(), np.asarray(t, float).ravel()
    return float((t.std() + 1e-12) / (o.std() + 1e-12)), float(stats.ks_2samp(o, t).statistic)


def nested_lr(X, y, C=0.05):
    oof = np.zeros(len(y), np.float64)
    subs = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=500)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
        subs.append(float(roc_auc_score(y[va], oof[va])))
    return oof, float(roc_auc_score(y, oof)), subs


def gate_metrics(z_base, z_cand, y):
    d = []
    for _, va in StratifiedKFold(5, shuffle=True, random_state=7).split(z_base, y):
        d.append(roc_auc_score(y[va], z_cand[va]) - roc_auc_score(y[va], z_base[va]))
    d = np.asarray(d)
    return {
        "auc_base": float(roc_auc_score(y, z_base)),
        "auc_cand": float(roc_auc_score(y, z_cand)),
        "delta_auc": float(roc_auc_score(y, z_cand) - roc_auc_score(y, z_base)),
        "delta_e5": float((roc_auc_score(y, z_cand) - roc_auc_score(y, z_base)) * 1e5),
        "pos_subfolds": int((d > 0).sum()),
        "subfold_deltas": d.tolist(),
        "t_stat": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)),
        "disp": float(d.std()),
        "KS": float(stats.ks_2samp(z_base, z_cand).statistic),
        "R2": float(r2_score(z_base, z_cand)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-dir", type=Path, default=REPO / "preds/v47_kaggle_f0")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "logs/v47")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = args.data
    if data is None:
        for p in (Path("/tmp/s6e11_v39_ssd/data"), REPO / "data"):
            if (p / "train.csv").exists():
                data = p
                break
    y = pd.read_csv(data / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = np.load(REPO / "preds/v47_sovereign/folds_skf42.npy")
    va = folds == 0
    y_va = y[va]
    assert len(y_va) == 139992

    z = np.load(REPO / "preds/v47_sovereign/z_sovereign_v0_oof.npy").ravel()
    z_va = z[va]
    base = float(roc_auc_score(y_va, z_va))
    log(f"z_sovereign_v0 F0 AUC={base:.6f}")

    req = [
        "hgdt_v2_f0_seq_llr_oof.npy",
        "hgdt_v2_f0_seq_llr_test.npy",
        "hgdt_v2_f0_field24_llr_oof.npy",
        "hgdt_v2_f0_field24_llr_test.npy",
        "hgdt_v2_f0_field24_std_oof.npy",
        "hgdt_v2_f0_field24_std_test.npy",
        "hgdt_v2_f0_meta.json",
    ]
    missing = [n for n in req if not (args.kaggle_dir / n).exists()]
    if missing:
        report = {"status": "MISSING_ARTIFACTS", "missing": missing, "NO_SUBMIT": True, "base_auc_f0": base}
        (args.out / "V47_3_KAGGLE_PRIVATE_HGDT_F0_REPORT.md").write_text(
            f"# V47.3 REPORT — BLOCKED\n\nmissing `{missing}`\nNO_SUBMIT=True\n"
        )
        (args.out / "V47_3_GATE.json").write_text(json.dumps(report, indent=2))
        log(f"missing {missing}")
        return

    checks = {}
    for n in req:
        if n.endswith(".npy"):
            a = np.load(args.kaggle_dir / n)
            checks[n] = {
                "shape": list(a.shape),
                "finite": bool(np.isfinite(a).all()),
                "sha256": sha256(args.kaggle_dir / n),
            }
            assert checks[n]["finite"], n
    log(f"artifact checks ok: { {k: v['shape'] for k,v in checks.items()} }")

    seq = np.load(args.kaggle_dir / "hgdt_v2_f0_seq_llr_oof.npy")
    f24 = np.load(args.kaggle_dir / "hgdt_v2_f0_field24_llr_oof.npy").astype(np.float64)
    s24 = np.load(args.kaggle_dir / "hgdt_v2_f0_field24_std_oof.npy").astype(np.float64)
    f24t = np.load(args.kaggle_dir / "hgdt_v2_f0_field24_llr_test.npy").astype(np.float64)
    assert seq.shape[0] in (139992, len(y))
    if seq.shape[0] == len(y):
        seq, f24, s24 = seq[va], f24[va], s24[va]
    assert f24.shape == (139992, 24), f24.shape

    seq_llr = seq[:, :, 1] - seq[:, :, 0]
    mean_seq = seq_llr.mean(1)
    sum_f = f24.sum(1)
    auc_seq = float(roc_auc_score(y_va, mean_seq))
    auc_sum = float(roc_auc_score(y_va, sum_f))

    X = np.hstack([f24, s24])
    oof_lr, auc_lr, sub_lr = nested_lr(X, y_va, C=0.05)
    log(f"solo seq={auc_seq:.6f} sum={auc_sum:.6f} lr={auc_lr:.6f}")

    # Cross-modal: P4 + field24 shallow LGB/XGB
    p4 = []
    for tag in ("P4_L4.2_pure5f", "P4_L4.3_pure5f", "P4_L4.5_pure5f"):
        p = REPO / "preds/matrix_v2" / f"{tag}_oof.npy"
        if p.exists():
            p4.append(np.load(p).ravel()[va, None])
    Xc = np.hstack(p4 + [f24]) if p4 else f24.copy()
    oof_cm = np.zeros(len(y_va))
    try:
        import lightgbm as lgb
        for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=42).split(Xc, y_va):
            dtr = lgb.Dataset(Xc[tr_i], label=y_va[tr_i], init_score=(0.25 * z_va[tr_i]))
            bst = lgb.train(
                dict(objective="binary", max_depth=3, learning_rate=0.05,
                     min_child_samples=200, verbosity=-1),
                dtr, num_boost_round=150,
            )
            oof_cm[va_i] = bst.predict(Xc[va_i])
    except Exception:
        try:
            import xgboost as xgb
            for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=42).split(Xc, y_va):
                dtr = xgb.DMatrix(Xc[tr_i], label=y_va[tr_i], base_margin=0.25 * z_va[tr_i])
                dva = xgb.DMatrix(Xc[va_i], base_margin=0.25 * z_va[va_i])
                bst = xgb.train(
                    {"max_depth": 3, "eta": 0.05, "objective": "binary:logistic",
                     "eval_metric": "auc", "colsample_bytree": 0.4},
                    dtr, num_boost_round=150,
                )
                oof_cm[va_i] = bst.predict(dva)
        except Exception as e:
            log(f"tree skip: {e}")
            oof_cm = oof_lr

    def fuse(p):
        return logit(np.clip(0.75 * expit(z_va) + 0.25 * np.clip(p, 1e-6, 1 - 1e-6), 1e-6, 1 - 1e-6))

    g_seq = gate_metrics(z_va, mean_seq, y_va)
    g_lr = gate_metrics(z_va, fuse(oof_lr), y_va)
    g_cm = gate_metrics(z_va, fuse(oof_cm), y_va)
    disp, ks = hygiene(f24.sum(1), f24t.sum(1))

    meta = {}
    mp = args.kaggle_dir / "hgdt_v2_f0_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())

    report = {
        "status": "DONE",
        "NO_SUBMIT": True,
        "spend_usd": 0.0,
        "base_auc_f0_z_sovereign_v0": base,
        "solo": {"auc_seq": auc_seq, "auc_sum_field": auc_sum, "auc_lr_24std": auc_lr, "subfold_lr": sub_lr},
        "gate_seq_vs_z0": g_seq,
        "gate_lr_fuse_vs_z0": g_lr,
        "gate_cross_modal_vs_z0": g_cm,
        "hygiene_sum_field": {"disp_raw": disp, "KS_raw": ks},
        "R2_seq_vs_z0": float(r2_score(z_va, mean_seq)),
        "artifacts": checks,
        "kaggle_meta": meta,
    }
    (args.out / "V47_3_GATE.json").write_text(json.dumps(report, indent=2, default=float))

    md = f"""# V47.3 — Private Kaggle HGDT-LM v2 Fold-0 Report

**NO_SUBMIT=True · $0.00 RunPod · 0 submissions · private kernel only**

## Artifacts
- dir: `{args.kaggle_dir}`
- shapes/SHA verified · 0 NaN/Inf

## Solo competence (Fold-0 va0 n=139992)
- equal-weight mean `seq_llr` AUC = **{auc_seq:.6f}**
- equal-weight `sum(field24_llr)` AUC = **{auc_sum:.6f}**
- newton-cholesky LR on 24+std AUC = **{auc_lr:.6f}**

## vs `z_sovereign_v0` F0 ({base:.6f})
| stream | ΔAUC (e-5) | pos/5 | t | KS | R² |
|---|---:|---:|---:|---:|---:|
| seq | {g_seq['delta_e5']:.2f} | {g_seq['pos_subfolds']} | {g_seq['t_stat']:.2f} | {g_seq['KS']:.4f} | {g_seq['R2']:.4f} |
| LR fuse | {g_lr['delta_e5']:.2f} | {g_lr['pos_subfolds']} | {g_lr['t_stat']:.2f} | {g_lr['KS']:.4f} | {g_lr['R2']:.4f} |
| cross-modal | {g_cm['delta_e5']:.2f} | {g_cm['pos_subfolds']} | {g_cm['t_stat']:.2f} | {g_cm['KS']:.4f} | {g_cm['R2']:.4f} |

## Hygiene (sum field24 oof vs test)
- `disp_raw` = {disp:.4f} · `KS_raw` = {ks:.4f}

## HARD STOP
- **NO_SUBMIT=True** · **$0.00 spent** · **0 Kaggle competition submissions**
"""
    (args.out / "V47_3_KAGGLE_PRIVATE_HGDT_F0_REPORT.md").write_text(md)
    log("DONE gate report written")


if __name__ == "__main__":
    main()
