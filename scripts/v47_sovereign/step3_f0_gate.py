#!/usr/bin/env python3
"""V47 Step3 — Fold-0 promotion gate on HGDT + Qwen field24 LLRs.

Tests:
  1) Solo: sum(field24) vs OOF LR(newton-cholesky, C=0.05) on 24 LLR + 24 std
  2) Cross-modal: shallow XGB/CatBoost on [P4 + field24s] with init_score=0.25*z_sov
  3) Sovereign stack gate: ΔAUC_f0, pos_subfolds/5, t_stat, disp, KS vs z_sovereign_v0[f0]

HARD STOP: NO_SUBMIT=True. Scale to 5-fold only if ΔAUC_f0 >= +8.0e-5 and >=4/5 subfolds.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats

NO_SUBMIT = True
assert NO_SUBMIT
T0 = time.time()
GATE = 8.0e-5
REPO = Path(__file__).resolve().parents[2]


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s][gate] {m}", flush=True)


def load_va_field(path: Path, va_ix: np.ndarray) -> np.ndarray:
    a = np.load(path).astype(np.float64)
    if a.shape[0] == len(va_ix):
        return a
    return a[va_ix]


def nested_lr_auc(X: np.ndarray, y: np.ndarray, C: float = 0.05) -> tuple[float, np.ndarray, list[float]]:
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), np.float64)
    sub = []
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=500)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
        sub.append(float(roc_auc_score(y[va], oof[va])))
    return float(roc_auc_score(y, oof)), oof, sub


def solo_tests(name: str, f24: np.ndarray, s24: np.ndarray | None, y: np.ndarray) -> dict:
    sum_llr = f24.sum(1)
    auc_sum = float(roc_auc_score(y, sum_llr))
    X = f24 if s24 is None else np.hstack([f24, s24])
    auc_lr, oof, sub = nested_lr_auc(X, y, C=0.05)
    return {
        "model": name,
        "auc_sum_equal_weight": auc_sum,
        "auc_lr_24plus_std": auc_lr,
        "delta_bps_vs_sum": (auc_lr - auc_sum) * 1e4,
        "subfold_aucs": sub,
        "oof": oof,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-dir", type=Path, default=REPO / "preds/v47_pod_f0")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "logs/v47")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    required = [
        "hgdt_f0_field24_llr_oof.npy",
        "qwen_f0_field24_llr_oof.npy",
    ]
    missing = [n for n in required if not (args.pod_dir / n).exists()]
    meta_z = {}
    mz = REPO / "logs/v47/STEP1_Z_SOVEREIGN.json"
    if mz.exists():
        meta_z = json.loads(mz.read_text())
    base_meta = float(
        meta_z.get("fold0_auc")
        or meta_z.get("auc_fold0")
        or meta_z.get("auc_f0")
        or (meta_z.get("fold_aucs") or [0.951251])[0]
    )
    if missing:
        report = {
            "status": "BLOCKED_NO_POD_ARTIFACTS",
            "missing": missing,
            "base_auc_f0": base_meta,
            "NO_SUBMIT": True,
            "gate": GATE,
            "note": "Step2 pod artifacts absent — cannot run promotion gate.",
        }
        (args.out / "STEP3_GATE.json").write_text(json.dumps(report, indent=2))
        log(f"HARD STOP — missing {missing}")
        return

    data = args.data
    if data is None:
        for p in [Path("/tmp/s6e11_v39_ssd/data"), REPO / "data"]:
            if (p / "train.csv").exists():
                data = p
                break
    assert data is not None and (data / "train.csv").exists()

    y = pd.read_csv(data / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y))
    tr_ix, va_ix = folds[0]
    y_va = y[va_ix]
    assert len(va_ix) == 139992, len(va_ix)

    z = np.load(REPO / "preds/v47_sovereign/z_sovereign_v0_oof.npy").astype(np.float64).ravel()
    z_va = z[va_ix]
    base = float(roc_auc_score(y_va, z_va))
    log(f"z_sovereign_v0 F0 AUC={base:.6f}")

    hg = load_va_field(args.pod_dir / "hgdt_f0_field24_llr_oof.npy", va_ix)
    qw = load_va_field(args.pod_dir / "qwen_f0_field24_llr_oof.npy", va_ix)
    hs = load_va_field(args.pod_dir / "hgdt_f0_field24_std_oof.npy", va_ix) if (args.pod_dir / "hgdt_f0_field24_std_oof.npy").exists() else None
    qs = load_va_field(args.pod_dir / "qwen_f0_field24_std_oof.npy", va_ix) if (args.pod_dir / "qwen_f0_field24_std_oof.npy").exists() else None

    t1_h = solo_tests("HGDT-LM", hg, hs, y_va)
    t1_q = solo_tests("Qwen2.5-0.5B", qw, qs, y_va)
    log(f"T1 HGDT sum={t1_h['auc_sum_equal_weight']:.6f} lr={t1_h['auc_lr_24plus_std']:.6f}")
    log(f"T1 Qwen sum={t1_q['auc_sum_equal_weight']:.6f} lr={t1_q['auc_lr_24plus_std']:.6f}")

    # Test 2 — P4 features + field24s
    p4 = []
    for tag in ("P4_L4.2_pure5f", "P4_L4.3_pure5f", "P4_L4.5_pure5f"):
        p = REPO / "preds/matrix_v2" / f"{tag}_oof.npy"
        if p.exists():
            p4.append(np.load(p).astype(np.float64).ravel()[va_ix, None])
    X2 = np.hstack(p4 + [hg, qw]) if p4 else np.hstack([hg, qw])
    init = 0.25 * z_va

    oof2 = np.zeros(len(y_va))
    sub2 = []
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, va in skf.split(X2, y_va):
        pred = None
        try:
            import xgboost as xgb

            dtr = xgb.DMatrix(X2[tr], label=y_va[tr], base_margin=init[tr])
            dva = xgb.DMatrix(X2[va], base_margin=init[va])
            bst = xgb.train(
                {"max_depth": 3, "eta": 0.05, "objective": "binary:logistic",
                 "eval_metric": "auc", "colsample_bytree": 0.4, "subsample": 0.8,
                 "tree_method": "hist"},
                dtr, num_boost_round=200,
            )
            pred = bst.predict(dva)
        except Exception as e:
            log(f"xgb skip: {e}")
        try:
            from catboost import CatBoostClassifier

            cb = CatBoostClassifier(
                depth=4, iterations=200, learning_rate=0.05, verbose=False,
                loss_function="Logloss",
            )
            cb.fit(X2[tr], y_va[tr], init_model=None)
            # soften via blend with init
            from scipy.special import expit, logit

            p_cb = cb.predict_proba(X2[va])[:, 1]
            p_init = expit(init[va])
            p_soft = 0.7 * p_cb + 0.3 * p_init
            pred = p_soft if pred is None else 0.5 * pred + 0.5 * p_soft
        except Exception as e:
            log(f"catboost skip: {e}")
        if pred is None:
            clf = LogisticRegression(C=0.05, solver="lbfgs", max_iter=500)
            clf.fit(X2[tr], y_va[tr])
            pred = clf.predict_proba(X2[va])[:, 1]
        oof2[va] = pred
        sub2.append(float(roc_auc_score(y_va[va], pred)))
    auc2 = float(roc_auc_score(y_va, oof2))
    log(f"T2 cross-modal AUC={auc2:.6f}")

    # Test 3 — stack vs sovereign
    # fuse: z + calibrated residual from best field stream
    best_oof = t1_h["oof"] if t1_h["auc_lr_24plus_std"] >= t1_q["auc_lr_24plus_std"] else t1_q["oof"]
    from scipy.special import logit, expit

    def safe_logit(p):
        return logit(np.clip(p, 1e-6, 1 - 1e-6))

    z_new = safe_logit(0.75 * expit(z_va) + 0.25 * best_oof)
    # also try oof2
    z_new2 = safe_logit(0.5 * expit(z_va) + 0.5 * oof2)

    def gate_metrics(z_base, z_cand, y):
        skf = StratifiedKFold(5, shuffle=True, random_state=7)
        d = []
        for tr, va in skf.split(z_base, y):
            a0 = roc_auc_score(y[va], z_base[va])
            a1 = roc_auc_score(y[va], z_cand[va])
            d.append(a1 - a0)
        d = np.asarray(d)
        auc0 = roc_auc_score(y, z_base)
        auc1 = roc_auc_score(y, z_cand)
        ks = float(stats.ks_2samp(z_base, z_cand).statistic)
        t_stat = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12))
        return {
            "auc_base": float(auc0),
            "auc_cand": float(auc1),
            "delta_auc": float(auc1 - auc0),
            "delta_e5": float((auc1 - auc0) * 1e5),
            "subfold_deltas": d.tolist(),
            "pos_subfolds": int((d > 0).sum()),
            "t_stat": t_stat,
            "disp": float(d.std()),
            "KS": ks,
        }

    g_h = gate_metrics(z_va, safe_logit(0.75 * expit(z_va) + 0.25 * t1_h["oof"]), y_va)
    g_q = gate_metrics(z_va, safe_logit(0.75 * expit(z_va) + 0.25 * t1_q["oof"]), y_va)
    g_x = gate_metrics(z_va, z_new2, y_va)

    def clears(g):
        return g["delta_auc"] >= GATE and g["pos_subfolds"] >= 4

    winners = []
    if clears(g_h):
        winners.append("HGDT-LM")
    if clears(g_q):
        winners.append("Qwen2.5-0.5B")
    if clears(g_x):
        winners.append("cross_modal")

    # Folds 1–4 projection (~same wall as F0 ×4)
    proj = {
        "f0_wall_hr_est": 3.5,
        "f1_4_wall_hr_est": 14.0,
        "f1_4_cost_usd_est_at_0.59": 8.26,
        "note": "Only scale if winners non-empty; else abort.",
    }

    report = {
        "status": "PASS_SCALE_5FOLD" if winners else "FAIL_HARD_STOP",
        "winners": winners,
        "base_auc_f0": base,
        "test1": {"hgdt": {k: v for k, v in t1_h.items() if k != "oof"},
                  "qwen": {k: v for k, v in t1_q.items() if k != "oof"}},
        "test2": {"auc": auc2, "subfolds": sub2},
        "test3": {"hgdt": g_h, "qwen": g_q, "cross_modal": g_x},
        "projection": proj,
        "NO_SUBMIT": True,
        "gate": GATE,
    }
    (args.out / "STEP3_GATE.json").write_text(json.dumps(report, indent=2))

    md = [
        "# V47 SOVEREIGN F0 REPORT",
        "",
        f"- status: **{report['status']}**",
        f"- winners: `{winners}`",
        f"- z_sovereign_v0 F0 AUC: **{base:.6f}**",
        f"- gate: ΔAUC_f0 >= +{GATE:.1e}, >=4/5 subfolds",
        "",
        "## Test 1 — Solo competence",
        f"- HGDT sum={t1_h['auc_sum_equal_weight']:.6f} · LR={t1_h['auc_lr_24plus_std']:.6f} · Δbps={t1_h['delta_bps_vs_sum']:.2f}",
        f"- Qwen sum={t1_q['auc_sum_equal_weight']:.6f} · LR={t1_q['auc_lr_24plus_std']:.6f} · Δbps={t1_q['delta_bps_vs_sum']:.2f}",
        "",
        "## Test 2 — Cross-modal tree",
        f"- AUC={auc2:.6f} · subfolds={sub2}",
        "",
        "## Test 3 — Sovereign stack gate",
        f"- HGDT: Δe5={g_h['delta_e5']:.2f} pos={g_h['pos_subfolds']}/5 t={g_h['t_stat']:.2f} KS={g_h['KS']:.4f}",
        f"- Qwen: Δe5={g_q['delta_e5']:.2f} pos={g_q['pos_subfolds']}/5 t={g_q['t_stat']:.2f} KS={g_q['KS']:.4f}",
        f"- Cross: Δe5={g_x['delta_e5']:.2f} pos={g_x['pos_subfolds']}/5 t={g_x['t_stat']:.2f} KS={g_x['KS']:.4f}",
        "",
        "## Cost / ETA projection (Folds 1–4)",
        f"- ~{proj['f1_4_wall_hr_est']} h · ~${proj['f1_4_cost_usd_est_at_0.59']:.2f} @ $0.59/hr",
        "",
        "## HARD STOP",
        "- **NO_SUBMIT=True** · 0 submissions · 0 public predictions admitted",
    ]
    (args.out / "V47_SOVEREIGN_F0_REPORT.md").write_text("\n".join(md) + "\n")
    log(f"DONE status={report['status']} winners={winners}")


if __name__ == "__main__":
    main()
