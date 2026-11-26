#!/usr/bin/env python3
"""Sealed-B transport: uniform pairwise vs hard-negative pairwise on frozen 5-stream (SKF rs=2039).

Freeze-before-touch. Uniform vs hard-neg heads. Previously examined rows. NO_SUBMIT/NO_GPU.
Streams: raw one-hot LR + native-cat LGBM + hist-XGB + decimal-token XGB + char n-gram LR.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import hstack
from scipy.special import logit
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJ = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
TRAIN = Path("/tmp/s6e11_isol/data/train.csv")
PRED = PROJ / "preds/v47_sovereign"
LOGS = PROJ / "logs/v47/codex_loop"
LOCK = PROJ / "artifacts/research_loop/SEALED_LOCK_FIVE_STREAM_HARD_NEGATIVE_B2039.json"
REPORT = LOGS / "SEALED_B_FIVE_STREAM_HARD_NEGATIVE_2039_TRANSPORT.json"
SCORES = PRED / "sealed_b_five_stream_hard_negative_2039_transport.npz"
SEED = 2039
PAIR_SEED = 2035
PAIRS_PER_FRAUD = 64
LEXICAL_CACHE = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step_decimal_tables(frame: pd.DataFrame, ids: np.ndarray):
    decimal_groups = [
        ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd"],
        ["velocity_score", "merchant_risk_score"],
    ]
    decimal_fields = [c for g in decimal_groups for c in g]
    raw = pd.read_csv(
        TRAIN,
        usecols=["id", *decimal_fields],
        dtype={c: "string" for c in decimal_fields},
    )
    np.testing.assert_array_equal(raw.pop("id").to_numpy(), ids)
    suffix, joint = {}, {}
    for column in decimal_fields:
        parts = raw[column].str.partition(".")
        suffix[column] = parts[2].where(parts[1].eq("."), "<integer>").fillna("<missing>")
        joint[column] = parts[0].fillna("<missing>").str[:2].str.cat(suffix[column], sep="|")
    return decimal_groups, suffix, joint


def make_fit_score(categories, numeric, decimal_groups, suffix, joint, labels):
    def rate_table(keys, target):
        return pd.DataFrame({"key": keys.to_numpy(), "label": target}).groupby("key")["label"].agg(["sum", "size"])

    def decimal_rates(reference, query):
        target = labels[reference]
        prior = float(target.mean())
        features = {}
        for group in decimal_groups:
            field_tables = {column: rate_table(suffix[column].iloc[reference], target) for column in group}
            pooled = pd.concat(field_tables.values()).groupby(level=0).sum()
            pool_rate = (pooled["sum"] + 50 * prior) / (pooled["size"] + 50)
            for column in group:
                table = field_tables[column]
                local_prior = pool_rate.reindex(table.index).fillna(prior)
                local_rate = (table["sum"] + 2000 * local_prior) / (table["size"] + 2000)
                query_suffix = suffix[column].iloc[query]
                coarse = query_suffix.map(local_rate).fillna(query_suffix.map(pool_rate)).fillna(prior)
                detail = pd.DataFrame(
                    {
                        "key": joint[column].iloc[reference].to_numpy(),
                        "suffix": suffix[column].iloc[reference].to_numpy(),
                        "label": target,
                    }
                ).groupby("key").agg(total=("label", "sum"), count=("label", "size"), suffix=("suffix", "first"))
                detail_prior = detail["suffix"].map(local_rate).fillna(prior)
                detail_rate = (detail["total"] + 200 * detail_prior) / (detail["count"] + 200)
                features[f"dec_rate__{column}"] = coarse.to_numpy(dtype=float)
                features[f"dec_joint_rate__{column}"] = (
                    joint[column].iloc[query].map(detail_rate).fillna(coarse).to_numpy(dtype=float)
                )
        return pd.DataFrame(features, index=query)

    def fit_score(fit_frame, fit_labels, score_frame, fit_rows):
        linear = make_pipeline(
            ColumnTransformer(
                [
                    ("numeric", StandardScaler(), numeric),
                    ("categorical", OneHotEncoder(handle_unknown="ignore"), categories),
                ]
            ),
            LogisticRegression(C=0.1, max_iter=1000),
        ).fit(fit_frame, fit_labels)
        margins = [linear.decision_function(score_frame)]

        tree_fit, tree_score = fit_frame.copy(), score_frame.copy()
        for column in categories:
            tree_fit[column] = tree_fit[column].astype("category")
            tree_score[column] = pd.Categorical(tree_score[column], categories=tree_fit[column].cat.categories)
        lgbm = LGBMClassifier(
            n_estimators=450,
            learning_rate=0.045,
            num_leaves=31,
            min_child_samples=120,
            reg_lambda=5,
            n_jobs=4,
            verbosity=-1,
            random_state=42,
        ).fit(tree_fit, fit_labels)
        margins.append(lgbm.predict(tree_score, raw_score=True))

        for column in categories:
            freq = tree_fit[column].value_counts(dropna=False)
            tree_fit[f"{column}_freq"] = tree_fit[column].map(freq).astype(float)
            tree_score[f"{column}_freq"] = tree_score[column].astype(object).map(freq).fillna(0).astype(float)
        xgb = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            tree_method="hist",
            enable_categorical=True,
            device="cpu",
            n_jobs=4,
            random_state=42,
            eval_metric="auc",
        ).fit(tree_fit, fit_labels)
        margins.append(xgb.predict(tree_score, output_margin=True))

        rate_parts = []
        encoder_cv = StratifiedKFold(3, shuffle=True, random_state=2030)
        for encoder_fit, encoder_held in encoder_cv.split(fit_frame, fit_labels):
            rate_parts.append(decimal_rates(fit_rows[encoder_fit], fit_rows[encoder_held]))
        training_rates = pd.concat(rate_parts).reindex(tree_fit.index)
        inference_rates = decimal_rates(fit_rows, score_frame.index.to_numpy())
        assert np.isfinite(training_rates.to_numpy()).all()
        assert np.isfinite(inference_rates.to_numpy()).all()
        decimal_model = clone(xgb).set_params(device="cpu")
        decimal_model.fit(tree_fit.join(training_rates), fit_labels)
        margins.append(decimal_model.predict(tree_score.join(inference_rates), output_margin=True))

        decimal_fields = [c for g in decimal_groups for c in g]
        lexical = LEXICAL_CACHE
        hasher = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=2**14,
            alternate_sign=False,
            binary=True,
            norm=None,
            lowercase=False,
            dtype=np.float64,
        )

        def lexical_matrix(row_indices):
            return hstack(
                [hasher.transform(lexical[column].iloc[row_indices]) for column in decimal_fields],
                format="csr",
            )

        lexical_model = LogisticRegression(C=0.1, solver="liblinear", max_iter=200, random_state=42)
        lexical_model.fit(lexical_matrix(fit_rows), fit_labels)
        margins.append(lexical_model.decision_function(lexical_matrix(score_frame.index.to_numpy())))
        return np.column_stack(margins)

    return fit_score


def rank_design(reference_scores, query_scores):
    columns = []
    for reference, values in zip(reference_scores.T, query_scores.T):
        ordered = np.sort(reference)
        left = np.searchsorted(ordered, values, side="left")
        right = np.searchsorted(ordered, values, side="right")
        eps = 0.5 / len(ordered)
        columns.append(logit(np.clip((left + right) / (2 * len(ordered)), eps, 1 - eps)))
    return np.column_stack(columns)


def stack_rowwise(inner_oof, outer_scores, y_train):
    inner_design = rank_design(inner_oof, inner_oof)
    outer_designs = [rank_design(inner_oof, scores) for scores in outer_scores]
    scaler = StandardScaler().fit(inner_design)
    model = LogisticRegression(C=0.03, max_iter=2000).fit(scaler.transform(inner_design), y_train)
    scores = [model.decision_function(scaler.transform(d)) for d in outer_designs]
    return np.mean(scores, axis=0), model.coef_.ravel().tolist()


def _fit_pairwise_head(design, y_train, negatives):
    fraud = np.flatnonzero(y_train == 1)
    rng = np.random.default_rng(PAIR_SEED)
    positives = np.repeat(fraud, PAIRS_PER_FRAUD)
    assert len(negatives) == len(positives)
    pair_design = design[positives] - design[negatives]
    pair_labels = rng.integers(0, 2, size=len(pair_design))
    pair_design *= (2 * pair_labels - 1)[:, None]
    head = LogisticRegression(
        C=0.03, solver="lbfgs", fit_intercept=False, max_iter=2000,
    ).fit(pair_design, pair_labels)
    assert np.max(head.n_iter_) < head.max_iter
    return head


def stack_uniform_pairwise(inner_oof, outer_scores, y_train):
    inner_design = rank_design(inner_oof, inner_oof)
    outer_designs = [rank_design(inner_oof, scores) for scores in outer_scores]
    scaler = StandardScaler().fit(inner_design)
    design = scaler.transform(inner_design)
    fraud = np.flatnonzero(y_train == 1)
    legitimate = np.flatnonzero(y_train == 0)
    rng = np.random.default_rng(PAIR_SEED)
    negatives = rng.choice(legitimate, size=len(fraud) * PAIRS_PER_FRAUD, replace=True)
    head = _fit_pairwise_head(design, y_train, negatives)
    scores = np.mean(
        [head.decision_function(scaler.transform(outer_design)) for outer_design in outer_designs],
        axis=0,
    )
    assert np.isfinite(scores).all()
    return scores, head.coef_.ravel().tolist()


def stack_hard_negative_pairwise(inner_oof, outer_scores, y_train):
    inner_design = rank_design(inner_oof, inner_oof)
    outer_designs = [rank_design(inner_oof, scores) for scores in outer_scores]
    scaler = StandardScaler().fit(inner_design)
    design = scaler.transform(inner_design)
    mining_score = np.full(len(y_train), np.nan)
    for fit, held in StratifiedKFold(3, shuffle=True, random_state=2036).split(design, y_train):
        miner = LogisticRegression(C=0.03, solver="lbfgs", max_iter=2000)
        miner.fit(design[fit], y_train[fit])
        assert miner.n_iter_.max() < miner.max_iter
        mining_score[held] = miner.decision_function(design[held])
    assert np.isfinite(mining_score).all()
    fraud = np.flatnonzero(y_train == 1)
    legitimate = np.flatnonzero(y_train == 0)
    ordered = legitimate[np.argsort(mining_score[legitimate], kind="stable")]
    location = np.searchsorted(mining_score[ordered], mining_score[fraud])
    radius = max(1, int(0.01 * len(ordered)))
    lower = np.maximum(0, location - radius)
    upper = np.minimum(len(ordered), location + radius + 1)
    rng = np.random.default_rng(PAIR_SEED)
    half = PAIRS_PER_FRAUD // 2
    uniform = rng.choice(legitimate, size=(len(fraud), half), replace=True)
    near_rank = rng.integers(lower[:, None], upper[:, None], size=(len(fraud), half))
    negatives = np.column_stack([uniform, ordered[near_rank]]).ravel()
    head = _fit_pairwise_head(design, y_train, negatives)
    scores = np.mean(
        [head.decision_function(scaler.transform(outer_design)) for outer_design in outer_designs],
        axis=0,
    )
    assert np.isfinite(scores).all()
    return scores, head.coef_.ravel().tolist()


def main() -> None:
    assert not REPORT.exists() and not SCORES.exists() and not LOCK.exists(), "B2039 artifacts exist"
    print("loading", TRAIN, flush=True)
    frame = pd.read_csv(TRAIN, low_memory=False)
    labels = frame.pop("is_fraud").to_numpy()
    ids = frame.pop("id").to_numpy()
    categories = frame.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    numeric = [c for c in frame.columns if c not in categories]
    decimal_groups, suffix, joint = step_decimal_tables(frame, ids)

    decimal_fields = [c for g in decimal_groups for c in g]
    raw_strings = pd.read_csv(
        TRAIN, usecols=["id", *decimal_fields], dtype={c: "string" for c in decimal_fields}
    )
    np.testing.assert_array_equal(raw_strings.pop("id").to_numpy(), ids)
    global LEXICAL_CACHE
    LEXICAL_CACHE = {
        column: ("^" + raw_strings[column].fillna("<missing>") + "$") for column in decimal_fields
    }

    fit_score = make_fit_score(categories, numeric, decimal_groups, suffix, joint, labels)

    study = {
        "scope": "B_hard_negative_vs_uniform_previously_examined_rows",
        "seed": SEED,
        "folds_used": [1, 2, 3, 4],
        "baseline": "Frozen five streams + 64 uniform negatives per fraud",
        "candidate": "Same five streams + frozen Round28 hard-negative sampler",
        "sampling": {"seed": PAIR_SEED, "uniform": 32, "near_rank": 32, "radius": 0.01, "mining_folds": 3, "mining_seed": 2036, "C": 0.03},
        "train_sha256": sha256(TRAIN),
        "script_sha256": sha256(Path(__file__)),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost")
        },
        "gate": {"min_delta_e5": 8.0, "min_positive_folds": 3, "min_loo": 0, "min_ci95_lower": 0},
        "NO_SUBMIT": True,
        "NO_GPU": True,
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x") as handle:
        handle.write(json.dumps(study, indent=2) + "\n")

    panels = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(frame, labels))
    results, paired = [], []
    for fold in (1, 2, 3, 4):
        print(f"Frozen B fold {fold}/4", flush=True)
        train, valid = panels[fold]
        assert np.intersect1d(train, valid).size == 0
        x_train, x_valid = frame.iloc[train], frame.iloc[valid]
        y_train = labels[train]
        inner_oof = np.full((len(train), 5), np.nan)
        outer_scores = []
        for k, (fit, held) in enumerate(
            StratifiedKFold(3, shuffle=True, random_state=2026).split(x_train, y_train)
        ):
            print(f"  Inner fit {k + 1}/3", flush=True)
            fit_rows, held_rows = train[fit], train[held]
            batch = pd.concat([x_train.iloc[held], x_valid])
            scores = fit_score(x_train.iloc[fit], y_train[fit], batch, fit_rows)
            inner_oof[held] = scores[: len(held)]
            outer_scores.append(scores[len(held) :])
        assert np.isfinite(inner_oof).all()
        control, control_coefs = stack_uniform_pairwise(inner_oof, outer_scores, y_train)
        candidate, pairwise_coefs = stack_hard_negative_pairwise(inner_oof, outer_scores, y_train)
        observed = labels[valid]
        base_auc = roc_auc_score(observed, control)
        cand_auc = roc_auc_score(observed, candidate)
        results.append(
            {
                "fold": fold,
                "n": int(len(valid)),
                "base_auc": float(base_auc),
                "candidate_auc": float(cand_auc),
                "delta_e5": float((cand_auc - base_auc) * 1e5),
                "control_coefficients": control_coefs,
                "pairwise_coefficients": pairwise_coefs,
            }
        )
        paired.append((valid, observed, control, candidate))
        print(f"  Δe-5={results[-1]['delta_e5']:+.3f} base={base_auc:.6f} cand={cand_auc:.6f}", flush=True)

    weights = np.array([item["n"] for item in results], dtype=float)
    deltas = np.array([item["delta_e5"] for item in results])
    weighted = float(np.average(deltas, weights=weights))
    loo = [
        float(np.average(np.delete(deltas, i), weights=np.delete(weights, i))) for i in range(len(deltas))
    ]
    rng = np.random.default_rng(SEED)
    bootstrap = np.zeros(1000)
    for weight, (_, observed, control, candidate) in zip(weights / weights.sum(), paired):
        negative = np.flatnonzero(observed == 0)
        positive = np.flatnonzero(observed == 1)
        for draw in range(len(bootstrap)):
            sample = np.concatenate(
                [rng.choice(negative, len(negative)), rng.choice(positive, len(positive))]
            )
            bootstrap[draw] += weight * (
                roc_auc_score(observed[sample], candidate[sample])
                - roc_auc_score(observed[sample], control[sample])
            ) * 1e5
    ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    wins = int(np.count_nonzero(deltas > 0))
    passed = weighted >= 8 and wins >= 3 and min(loo) > 0 and ci[0] > 0
    result = {
        "scope": study["scope"],
        "lock_sha256": sha256(LOCK),
        "per_fold": results,
        "weighted_delta_e5": weighted,
        "positive_folds": wins,
        "loo_delta_e5": loo,
        "paired_ci95_e5": [float(ci[0]), float(ci[1])],
        "verdict": "B_TRANSPORT_PASS" if passed else "B_TRANSPORT_FAIL",
        "milestone": "LOCAL_GATE_PASS",
        "NO_SUBMIT": True,
        "NO_GPU": True,
    }
    np.savez(
        SCORES,
        row_indices=np.concatenate([p[0] for p in paired]),
        ids=np.concatenate([ids[p[0]] for p in paired]),
        fold=np.concatenate([np.full(len(p[0]), item["fold"]) for item, p in zip(results, paired)]),
        control=np.concatenate([p[2] for p in paired]),
        candidate=np.concatenate([p[3] for p in paired]),
    )
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    with (PROJ / "logs/v47/CODEX_DS_REVIEW_NEXT.md").open("a") as handle:
        handle.write(
            "\n## Frozen five-stream pairwise B transport diagnostic\n\n"
            f"{result['verdict']}. Weighted within-fold Δe-5 {weighted:+.8f}, "
            f"{wins}/4 positive, leave-one-fold-out Δe-5 {loo}, paired 95% interval {ci}. "
            "Historical rows under SKF seed 2039. Independent holdout verification remains pending.\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
