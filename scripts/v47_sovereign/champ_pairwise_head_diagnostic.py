#!/usr/bin/env python3
"""Champion-compatible pairwise meta swap on z_sovereign_v0's 14 streams.

Reproduce row-wise Deotte stack, then swap only the meta head to frozen
pairwise recipe (seed=2035, pairs_per_fraud=64, C=0.03). Screen on folds 1–4.
NO_SUBMIT / NO_GPU. Does not retune B2037.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJ = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
LOGS = PROJ / "logs/v47/codex_loop"
PRED = PROJ / "preds/v47_sovereign"
REPORT = LOGS / "CHAMP_PAIRWISE_HEAD_DIAGNOSTIC.json"
SCORES = PRED / "champ_pairwise_head_diagnostic.npz"
PAIR_SEED = 2035
PAIRS_PER_FRAUD = 64
C = 0.03


def load_step1():
    path = PROJ / "scripts/v47_sovereign/step1_schema_and_z_sovereign.py"
    spec = importlib.util.spec_from_file_location("step1_z", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def pairwise_head(Xtr, ytr, Xva, Xte):
    fraud = np.flatnonzero(ytr == 1)
    legitimate = np.flatnonzero(ytr == 0)
    rng = np.random.default_rng(PAIR_SEED)
    fraud_rows = np.repeat(fraud, PAIRS_PER_FRAUD)
    legitimate_rows = rng.choice(legitimate, size=len(fraud_rows), replace=True)
    pair_design = Xtr[fraud_rows] - Xtr[legitimate_rows]
    pair_labels = rng.integers(0, 2, size=len(pair_design))
    pair_design = pair_design * (2 * pair_labels - 1)[:, None]
    head = LogisticRegression(
        C=C, solver="lbfgs", fit_intercept=False, max_iter=2000,
    ).fit(pair_design, pair_labels)
    assert np.max(head.n_iter_) < head.max_iter
    return head.decision_function(Xva), head.decision_function(Xte), head.coef_.ravel().tolist()


def main() -> None:
    assert not REPORT.exists(), "diagnostic already exists"
    step1 = load_step1()
    print("loading train + assembling 14 streams…", flush=True)
    tr = pd.read_csv(step1.DATA / "train.csv")
    y = tr["is_fraud"].to_numpy(int)
    folds = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        folds[va] = k
    saved = np.load(PRED / "folds_skf42.npy")
    np.testing.assert_array_equal(folds, saved)

    streams = step1.assemble_streams(tr, y, folds)
    names = sorted(streams.keys())
    assert len(names) == 14, names

    oof_z = np.zeros((len(y), len(names)), np.float64)
    te_z = np.zeros((299984, len(names)), np.float64)
    for j, n in enumerate(names):
        o, t = streams[n]
        oof_z[:, j] = step1.to_logit_cal(o)
        te_z[:, j] = step1.to_logit_cal(t)

    champ_saved = np.load(PRED / "z_sovereign_v0_oof.npy").ravel()
    control = np.zeros(len(y), np.float64)
    candidate = np.zeros(len(y), np.float64)
    te_control = np.zeros(299984, np.float64)
    te_candidate = np.zeros(299984, np.float64)
    fold_rows = []
    coefs = []

    for k in range(5):
        trm, va = folds != k, folds == k
        mu = oof_z[trm].mean(0)
        sd = oof_z[trm].std(0) + 1e-8
        Xtr = (oof_z[trm] - mu) / sd
        Xva = (oof_z[va] - mu) / sd
        Xte = (te_z - mu) / sd
        lr = LogisticRegression(C=0.03, solver="lbfgs", max_iter=2000)
        lr.fit(Xtr, y[trm])
        control[va] = lr.predict_proba(Xva)[:, 1]
        te_control += lr.predict_proba(Xte)[:, 1] / 5
        va_scores, te_scores, coef = pairwise_head(Xtr, y[trm], Xva, Xte)
        candidate[va] = va_scores
        te_candidate += te_scores / 5
        coefs.append(coef)
        base = roc_auc_score(y[va], control[va])
        cand = roc_auc_score(y[va], candidate[va])
        fold_rows.append(
            {
                "fold": k,
                "n": int(va.sum()),
                "base_auc": float(base),
                "candidate_auc": float(cand),
                "delta_e5": float((cand - base) * 1e5),
                "control_coef": lr.coef_.ravel().tolist(),
                "pairwise_coef": coef,
            }
        )
        print(
            f"fold{k}: base={base:.6f} cand={cand:.6f} Δe-5={(cand-base)*1e5:+.3f}",
            flush=True,
        )

    # Reproduce saved champion control on all rows
    repro_delta = float(np.max(np.abs(control - champ_saved)))
    print(f"control vs saved z0 max|Δ|={repro_delta:.3e}", flush=True)

    # Screen on historical folds 1–4 (leave fold0 as previously heavily examined)
    eval_folds = [r for r in fold_rows if r["fold"] in (1, 2, 3, 4)]
    weights = np.array([r["n"] for r in eval_folds], dtype=float)
    deltas = np.array([r["delta_e5"] for r in eval_folds])
    weighted = float(np.average(deltas, weights=weights))
    wins = int(np.count_nonzero(deltas > 0))
    loo = [
        float(np.average(np.delete(deltas, i), weights=np.delete(weights, i)))
        for i in range(len(deltas))
    ]
    passed = weighted >= 8 and wins >= 3 and min(loo) > 0
    verdict = "DEV_ADVANCE_CHAMP_PAIRWISE" if passed else "NO-GO"

    overall_base = float(roc_auc_score(y, control))
    overall_cand = float(roc_auc_score(y, candidate))
    result = {
        "scope": "champion_14stream_pairwise_head_swap_matched",
        "streams": names,
        "pairwise_head": {"seed": PAIR_SEED, "pairs_per_fraud": PAIRS_PER_FRAUD, "C": C},
        "control_repro_max_abs_delta_vs_saved_z0": repro_delta,
        "per_fold": fold_rows,
        "eval_folds": [1, 2, 3, 4],
        "weighted_delta_e5_folds1_4": weighted,
        "positive_folds_1_4": wins,
        "loo_delta_e5": loo,
        "overall_base_auc": overall_base,
        "overall_candidate_auc": overall_cand,
        "verdict": verdict,
        "gate": {"min_delta_e5": 8.0, "min_positive_folds": 3, "min_loo": 0},
        "NO_SUBMIT": True,
        "NO_GPU": True,
        "note": "Char-ngram excluded by design. B2037 not retuned.",
    }
    np.savez(
        SCORES,
        control_oof=control,
        candidate_oof=candidate,
        control_test=te_control,
        candidate_test=te_candidate,
        folds=folds,
        stream_names=np.array(names),
    )
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    with (PROJ / "logs/v47/CODEX_DS_REVIEW_NEXT.md").open("a") as handle:
        handle.write(
            "\n## Champion 14-stream pairwise head diagnostic\n\n"
            f"{verdict}. Folds1–4 weighted Δe-5 {weighted:+.8f}, {wins}/4 positive, "
            f"LOO {loo}. Control repro max|Δ| vs saved z0 = {repro_delta:.3e}. "
            "NO_SUBMIT.\n"
        )
    print(json.dumps({k: result[k] for k in (
        "verdict", "weighted_delta_e5_folds1_4", "positive_folds_1_4",
        "loo_delta_e5", "overall_base_auc", "overall_candidate_auc",
        "control_repro_max_abs_delta_vs_saved_z0",
    )}, indent=2))


if __name__ == "__main__":
    main()
