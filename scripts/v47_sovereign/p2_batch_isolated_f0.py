#!/usr/bin/env python3
"""P2 batch vs isolated F0 anchor: interaction LR + histogram XGB (fixed recipes).

Uses /tmp staged train if present. NO signed-CB. NO public arrays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

PROJ = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
TMP = Path("/tmp/s6e11_isol")
TRAIN = TMP / "data/train.csv" if (TMP / "data/train.csv").exists() else PROJ / "data/train.csv"
PRED = PROJ / "preds/v47_sovereign"
OUT = PROJ / "logs/v47/codex_loop"
REVIEW = PROJ / "logs/v47/CODEX_DS_REVIEW_NEXT.md"


def rank_logit(scores: np.ndarray) -> np.ndarray:
    return logit((rankdata(scores) - 0.5) / len(scores))


def eval_blend(anchor, stream, y, weight: float) -> dict:
    cand = (1 - weight) * rank_logit(anchor) + weight * rank_logit(stream)
    base = roc_auc_score(y, anchor)
    solo = roc_auc_score(y, stream)
    auc = roc_auc_score(y, cand)
    d = auc - base
    subs = [
        (roc_auc_score(y[i], cand[i]) - roc_auc_score(y[i], anchor[i])) * 1e5
        for _, i in StratifiedKFold(5, shuffle=True, random_state=7).split(anchor, y)
    ]
    pos = int(sum(s > 0 for s in subs))
    return {
        "base_auc": float(base),
        "solo_auc": float(solo),
        "candidate_auc": float(auc),
        "delta_e5": float(d * 1e5),
        "positive_subfolds": pos,
        "subfold_deltas_e5": [float(s) for s in subs],
        "fixed_weight": weight,
        "verdict": "DEV_ADVANCE" if d >= 8e-5 and pos >= 4 else "NO-GO",
    }


def main() -> None:
    print("loading", TRAIN, flush=True)
    frame = pd.read_csv(TRAIN, low_memory=False)
    labels = frame.pop("is_fraud").to_numpy()
    row_ids = frame.pop("id").to_numpy()
    folds = np.load(PRED / "folds_skf42.npy")
    train, valid = next(StratifiedKFold(5, shuffle=True, random_state=42).split(frame, labels))
    np.testing.assert_array_equal(valid, np.flatnonzero(folds == 0))

    with np.load(PRED / "isolated_champion_f0_diagnostic.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["row_indices"], valid)
        anchor = saved["stack"]
    manifest = json.loads((OUT / "ISOLATED_CHAMPION_F0_DIAGNOSTIC.json").read_text())
    assert hashlib.sha256(np.ascontiguousarray(anchor).tobytes()).hexdigest() == manifest["candidate_sha256"]
    y0 = labels[valid]
    assert abs(roc_auc_score(y0, anchor) - manifest["isolated_stack_auc"]) < 1e-12

    fit = frame.iloc[train]
    score = frame.iloc[valid]
    cats = fit.select_dtypes(include=["object", "str", "category"]).columns.tolist()

    # --- Exp2: smoothed interactions ---
    print("Exp2: interaction LR", flush=True)
    x_fit = fit.copy()
    x_score = score.copy()
    pairs = [
        ("auth_method", "channel"),
        ("merchant_category", "card_type"),
    ]
    inter_cols = []
    for a, b in pairs:
        name = f"{a}__x__{b}"
        x_fit[name] = x_fit[a].astype(str) + "|" + x_fit[b].astype(str)
        x_score[name] = x_score[a].astype(str) + "|" + x_score[b].astype(str)
        inter_cols.append(name)
    # coarse velocity × merchant_risk grid
    def coarse(s: pd.Series, q=(0.25, 0.5, 0.75)) -> pd.Series:
        edges = np.unique(np.quantile(s.to_numpy(float), q))
        # digitize into bins using train edges
        return pd.Series(np.digitize(s.to_numpy(float), edges), index=s.index).astype(str)

    vbin = coarse(x_fit["velocity_score"])
    rbin = coarse(x_fit["merchant_risk_score"])
    edges_v = np.unique(np.quantile(x_fit["velocity_score"].to_numpy(float), (0.25, 0.5, 0.75)))
    edges_r = np.unique(np.quantile(x_fit["merchant_risk_score"].to_numpy(float), (0.25, 0.5, 0.75)))
    x_fit["vel__x__risk"] = vbin + "|" + rbin
    x_score["vel__x__risk"] = (
        pd.Series(np.digitize(x_score["velocity_score"].to_numpy(float), edges_v), index=x_score.index).astype(str)
        + "|"
        + pd.Series(np.digitize(x_score["merchant_risk_score"].to_numpy(float), edges_r), index=x_score.index).astype(str)
    )
    inter_cols.append("vel__x__risk")
    cat_all = cats + inter_cols
    nums = [c for c in fit.columns if c not in cats]
    pipe = make_pipeline(
        ColumnTransformer(
            [
                ("num", StandardScaler(), nums),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat_all),
            ]
        ),
        LogisticRegression(C=0.05, max_iter=1000),
    )
    pipe.fit(x_fit[nums + cat_all], labels[train])
    stream2 = pipe.decision_function(x_score[nums + cat_all])
    r2 = eval_blend(anchor, stream2, y0, weight=0.15)
    r2["experiment"] = "interaction_lr_fixed15"
    print(r2["verdict"], r2["delta_e5"], r2["positive_subfolds"], flush=True)

    # --- Exp3: histogram XGB ---
    print("Exp3: histogram XGB", flush=True)
    if XGBClassifier is None:
        r3 = {"experiment": "hist_xgb_fixed15", "verdict": "SKIP", "reason": "xgboost missing"}
    else:
        x_fit2 = fit.copy()
        x_score2 = score.copy()
        for c in cats:
            x_fit2[c] = x_fit2[c].astype("category")
            x_score2[c] = pd.Categorical(x_score2[c], categories=x_fit2[c].cat.categories)
        # train-fitted frequency features
        for c in cats:
            freq = x_fit2[c].value_counts(dropna=False)
            x_fit2[f"{c}_freq"] = x_fit2[c].map(freq).astype(float)
            x_score2[f"{c}_freq"] = x_score2[c].astype(object).map(freq).fillna(0).astype(float)
        model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            tree_method="hist",
            enable_categorical=True,
            n_jobs=4,
            random_state=42,
            eval_metric="auc",
        )
        model.fit(x_fit2, labels[train])
        stream3 = model.predict(x_score2, output_margin=True)
        r3 = eval_blend(anchor, stream3, y0, weight=0.15)
        r3["experiment"] = "hist_xgb_fixed15"
        print(r3["verdict"], r3["delta_e5"], r3["positive_subfolds"], flush=True)

    out = {
        "anchor_auc": float(manifest["isolated_stack_auc"]),
        "experiments": [r2, r3],
        "NO_SUBMIT": True,
        "milestone": "LOCAL_GATE_PASS",
    }
    (OUT / "P2_BATCH_F0_DEV.json").write_text(json.dumps(out, indent=2) + "\n")
    with REVIEW.open("a") as f:
        f.write("\n## P2 batch vs isolated anchor (F0 DEV)\n\n")
        for r in (r2, r3):
            f.write(
                f"- {r.get('experiment')}: Δe-5={r.get('delta_e5')} pos={r.get('positive_subfolds')}/5 "
                f"{r.get('verdict')} solo={r.get('solo_auc')}\n"
            )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
