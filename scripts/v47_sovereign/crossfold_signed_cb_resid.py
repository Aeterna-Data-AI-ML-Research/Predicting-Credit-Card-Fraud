#!/usr/bin/env python3
"""Cross-fold generalization of frozen signed CatBoost residual (w=-0.10).

Milestone target: CROSS_FOLD_VERIFIED
- Freeze coefficient at -0.10 (from Fold-0 nested search).
- Train CatBoost only on each outer fold's training rows; score held-out fold.
- Do NOT retune w on evaluation folds.
- Sensitivity grid ONLY on Fold-0 (dev).
- NO_SUBMIT / no tip CSVs / $0 cloud.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
SOV = ROOT / "preds" / "v47_sovereign"
OUT = ROOT / "logs" / "v47" / "codex_loop"
FIXED_W = -0.10
SENS_GRID = (-0.05, -0.075, -0.10, -0.125, -0.15)
BASE_F0 = 0.9512507991341679


def rank_logit(scores: np.ndarray) -> np.ndarray:
    return logit((rankdata(scores) - 0.5) / len(scores))


def paired_bootstrap_delta(
    y: np.ndarray, base: np.ndarray, cand: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # skip degenerate resamples
        if y[idx].min() == y[idx].max():
            deltas[i] = np.nan
            continue
        deltas[i] = roc_auc_score(y[idx], cand[idx]) - roc_auc_score(y[idx], base[idx])
    deltas = deltas[np.isfinite(deltas)]
    return {
        "n_boot_ok": int(len(deltas)),
        "mean_delta": float(deltas.mean()),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "p_delta_gt_0": float((deltas > 0).mean()),
        "p_delta_ge_8e5": float((deltas >= 8e-5).mean()),
    }


def train_cb(x_tr: pd.DataFrame, y_tr: np.ndarray, cats: list[str], seed: int) -> CatBoostClassifier:
    model = CatBoostClassifier(
        iterations=850,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=10,
        max_ctr_complexity=2,
        random_seed=seed,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_tr, y_tr, cat_features=cats)
    return model


def error_analysis(y: np.ndarray, base: np.ndarray, cand: np.ndarray, cb: np.ndarray) -> dict:
    """Which ranking errors negative CB coefficient corrects (on labeled fold)."""
    # Pairwise: among fraud/nonfraud pairs where base and cand disagree on order
    fraud = np.flatnonzero(y == 1)
    legit = np.flatnonzero(y == 0)
    # Sample for tractability
    rng = np.random.default_rng(0)
    nf = min(len(fraud), 4000)
    nl = min(len(legit), 8000)
    f_idx = rng.choice(fraud, size=nf, replace=False)
    l_idx = rng.choice(legit, size=nl, replace=False)
    # Vectorized pairwise on samples is heavy; use score-gap proxies
    rb, rc, rcb = rank_logit(base), rank_logit(cand), rank_logit(cb)
    # Rows where cand moves away from CB relative to base (signed residual direction)
    move = rc - rb
    # Correct direction for frauds: should increase relative rank; for legit: decrease
    fraud_up = float((move[y == 1] > 0).mean())
    legit_down = float((move[y == 0] < 0).mean())
    # Disagreement zones: high |ranklogit(cb)-ranklogit(base)|
    gap = np.abs(rcb - rb)
    q = np.quantile(gap, [0.5, 0.8, 0.9])
    out = {
        "fraud_frac_score_increased": fraud_up,
        "legit_frac_score_decreased": legit_down,
        "gap_quantiles": [float(x) for x in q],
        "mean_move_fraud": float(move[y == 1].mean()),
        "mean_move_legit": float(move[y == 0].mean()),
        "auc_base": float(roc_auc_score(y, base)),
        "auc_cand": float(roc_auc_score(y, cand)),
        "auc_cb_solo": float(roc_auc_score(y, cb)),
        "sample_nf": int(nf),
        "sample_nl": int(nl),
    }
    # Top-decile gap: does cand beat base more there?
    mask = gap >= q[2]
    if mask.sum() > 100 and y[mask].min() < y[mask].max():
        out["auc_base_top10gap"] = float(roc_auc_score(y[mask], base[mask]))
        out["auc_cand_top10gap"] = float(roc_auc_score(y[mask], cand[mask]))
        out["delta_top10gap"] = out["auc_cand_top10gap"] - out["auc_base_top10gap"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-train", action="store_true", help="Reuse saved cb_oof if present")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_csv(ROOT / "data" / "train.csv")
    y = df.pop("is_fraud").to_numpy()
    df = df.drop(columns=["id"])
    cats = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cats:
        df[c] = df[c].fillna("__missing__").astype(str)

    folds = np.load(SOV / "folds_skf42.npy")
    assert folds.shape[0] == len(y)
    z0 = np.load(SOV / "z_sovereign_v0_oof.npy").ravel()
    assert z0.shape[0] == len(y)

    # Provenance checks
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    rebuilt = np.empty(len(y), dtype=np.int8)
    for k, (_, va) in enumerate(skf.split(df, y)):
        rebuilt[va] = k
    np.testing.assert_array_equal(folds, rebuilt)

    cb_path = SOV / "catboost_outer_oof.npy"
    if args.skip_train and cb_path.exists():
        cb_oof = np.load(cb_path).ravel()
        print(f"Loaded existing {cb_path}")
    else:
        cb_oof = np.full(len(y), np.nan, dtype=np.float64)
        for k in range(5):
            tr = folds != k
            va = folds == k
            print(f"Training CatBoost outer fold {k} n_tr={tr.sum()} n_va={va.sum()} ...", flush=True)
            model = train_cb(df.iloc[tr], y[tr], cats, seed=42 + k)
            cb_oof[va] = model.predict(df.iloc[va], prediction_type="RawFormulaVal")
            solo = roc_auc_score(y[va], cb_oof[va])
            print(f"  fold{k} CB solo AUC={solo:.6f}", flush=True)
        np.save(cb_path, cb_oof)
        # Keep F0 compatibility check vs prior solo
        f0 = folds == 0
        prior = SOV / "catboost_solo_f0.npy"
        if prior.exists():
            old = np.load(prior).ravel()
            # same seed path as Round3 used seed=42 only for fold0; here seed=42+0=42
            corr = float(np.corrcoef(old, cb_oof[f0])[0, 1])
            print(f"F0 CB vs prior solo corr={corr:.6f}", flush=True)

    assert np.isfinite(cb_oof).all()

    # Frozen-w candidate on all folds (rank-logit within each fold's scores)
    cand = np.empty(len(y), dtype=np.float64)
    fold_rows = []
    for k in range(5):
        va = folds == k
        base_k = z0[va]
        cb_k = cb_oof[va]
        y_k = y[va]
        a = rank_logit(base_k)
        r = rank_logit(cb_k) - a
        # frozen: a + FIXED_W * (z(cb)-a) = (1-FIXED_W)* wait: a + w*(zcb-a) with w=-0.1
        # = 1.1*a - 0.1*zcb  when w=-0.1
        c_k = a + FIXED_W * (rank_logit(cb_k) - a)
        cand[va] = c_k
        base_auc = float(roc_auc_score(y_k, base_k))
        cand_auc = float(roc_auc_score(y_k, c_k))
        cb_auc = float(roc_auc_score(y_k, cb_k))
        d = cand_auc - base_auc
        fold_rows.append(
            {
                "fold": k,
                "n": int(va.sum()),
                "base_auc": base_auc,
                "cand_auc": cand_auc,
                "cb_solo_auc": cb_auc,
                "delta": d,
                "delta_e5": d * 1e5,
                "role": "dev_local_gate" if k == 0 else "untouched_eval",
            }
        )
        print(
            f"fold{k}: base={base_auc:.6f} cand={cand_auc:.6f} Δe-5={d*1e5:+.3f} "
            f"[{'DEV' if k==0 else 'EVAL'}]",
            flush=True,
        )

    # Pooled metrics
    pooled_base = float(roc_auc_score(y, z0))
    pooled_cand = float(roc_auc_score(y, cand))
    pooled_delta = pooled_cand - pooled_base
    eval_mask = folds != 0
    eval_base = float(roc_auc_score(y[eval_mask], z0[eval_mask]))
    eval_cand = float(roc_auc_score(y[eval_mask], cand[eval_mask]))
    eval_delta = eval_cand - eval_base
    eval_pos = sum(1 for r in fold_rows if r["fold"] > 0 and r["delta"] > 0)

    print(
        f"POOLED all5: base={pooled_base:.6f} cand={pooled_cand:.6f} Δe-5={pooled_delta*1e5:+.3f}",
        flush=True,
    )
    print(
        f"EVAL folds1-4 pooled: base={eval_base:.6f} cand={eval_cand:.6f} "
        f"Δe-5={eval_delta*1e5:+.3f} pos={eval_pos}/4",
        flush=True,
    )

    boot_all = paired_bootstrap_delta(y, z0, cand, n_boot=args.n_boot)
    boot_eval = paired_bootstrap_delta(y[eval_mask], z0[eval_mask], cand[eval_mask], n_boot=args.n_boot)
    print("bootstrap all:", boot_all, flush=True)
    print("bootstrap eval1-4:", boot_eval, flush=True)

    # Sensitivity ONLY on Fold-0 (development)
    f0 = folds == 0
    sens = []
    a0 = rank_logit(z0[f0])
    zcb0 = rank_logit(cb_oof[f0])
    y0 = y[f0]
    base0 = float(roc_auc_score(y0, z0[f0]))
    for w in SENS_GRID:
        c = a0 + w * (zcb0 - a0)
        auc = float(roc_auc_score(y0, c))
        sens.append({"w": w, "auc": auc, "delta_e5": (auc - base0) * 1e5})
    print("F0 sensitivity:", sens, flush=True)

    # Error analysis on eval folds pooled and on F0
    err_f0 = error_analysis(y[f0], z0[f0], cand[f0], cb_oof[f0])
    err_eval = error_analysis(y[eval_mask], z0[eval_mask], cand[eval_mask], cb_oof[eval_mask])

    # Milestone rule (strict): CROSS_FOLD_VERIFIED requires
    # - frozen w
    # - mean Δ on untouched folds1-4 > 0
    # - ≥3/4 eval folds positive
    # - eval bootstrap CI lower bound > 0 OR (pos>=3/4 and pooled eval Δ>0)
    cross = (
        eval_pos >= 3
        and eval_delta > 0
        and boot_eval["ci95"][0] > 0
    )
    milestone = "CROSS_FOLD_VERIFIED" if cross else "LOCAL_GATE_PASS"

    report = {
        "milestone_achieved": milestone,
        "prior_milestone": "LOCAL_GATE_PASS",
        "fixed_w": FIXED_W,
        "retuned_on_eval": False,
        "provenance": {
            "folds_match_skf_rs42": True,
            "champion": "preds/v47_sovereign/z_sovereign_v0_oof.npy",
            "cb_oof": str(cb_path),
            "cb_train": "outer fold train only; seed=42+fold",
            "NO_SUBMIT": True,
            "tip_csvs": 0,
        },
        "fold_metrics": fold_rows,
        "pooled_all5": {
            "base_auc": pooled_base,
            "cand_auc": pooled_cand,
            "delta": pooled_delta,
            "delta_e5": pooled_delta * 1e5,
        },
        "eval_folds_1_4": {
            "base_auc": eval_base,
            "cand_auc": eval_cand,
            "delta": eval_delta,
            "delta_e5": eval_delta * 1e5,
            "pos": eval_pos,
            "n_folds": 4,
        },
        "bootstrap_all5": boot_all,
        "bootstrap_eval_1_4": boot_eval,
        "sensitivity_f0_only": sens,
        "error_analysis_f0": err_f0,
        "error_analysis_eval": err_eval,
        "elapsed_s": time.time() - t0,
        "NO_SUBMIT": True,
    }
    np.save(SOV / "champ_cb_signed_resid_oof_w-0.10.npy", cand)
    out_json = OUT / "ROUND6_CROSSFOLD_SIGNED_CB.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps({"milestone_achieved": milestone, "eval_delta_e5": eval_delta * 1e5, "eval_pos": eval_pos}, indent=2))
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
