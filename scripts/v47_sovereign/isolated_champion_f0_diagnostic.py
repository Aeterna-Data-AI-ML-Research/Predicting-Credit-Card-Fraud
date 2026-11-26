import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("START isolated champion diagnostic", flush=True)
pred = Path("preds/v47_sovereign")
log = Path("logs/v47")
print("loading train.csv...", flush=True)
df = pd.read_csv("data/train.csv", low_memory=False)
print("train loaded", df.shape, flush=True)
y = df.pop("is_fraud").to_numpy()
ids = df.pop("id").to_numpy()
folds = np.load(pred / "folds_skf42.npy")
train, valid = next(StratifiedKFold(5, shuffle=True, random_state=42).split(df, y))
np.testing.assert_array_equal(valid, np.flatnonzero(folds == 0))
assert len(valid) == 139992

# Keep scores and category vocabularies local to each fit.
cats = df.select_dtypes(include=["object", "str", "category"]).columns.tolist()
nums = [col for col in df.columns if col not in cats]
x_train, x_valid = df.iloc[train], df.iloc[valid]
y_train, y_valid = y[train], y[valid]

def fit_score(fit, labels, score):
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), nums),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cats),
    ])
    linear = make_pipeline(preprocessing, LogisticRegression(C=0.1, max_iter=1000))
    linear.fit(fit, labels)

    tree_fit, tree_score = fit.copy(), score.copy()
    for col in cats:
        tree_fit[col] = tree_fit[col].astype("category")
        tree_score[col] = pd.Categorical(tree_score[col], categories=tree_fit[col].cat.categories)
    tree = LGBMClassifier(
        n_estimators=450, learning_rate=0.045, num_leaves=31,
        min_child_samples=120, reg_lambda=5, n_jobs=4,
        verbosity=-1, random_state=42,
    )
    tree.fit(tree_fit, labels)
    return np.column_stack([
        linear.decision_function(score),
        tree.predict(tree_score, raw_score=True),
    ])

def rank_logit_from_reference(reference, scores):
    columns = []
    for j in range(reference.shape[1]):
        ordered = np.sort(reference[:, j])
        values = scores[:, j]
        left = np.searchsorted(ordered, values, side="left")
        right = np.searchsorted(ordered, values, side="right")
        q = (left + right) / (2 * len(ordered))
        eps = 0.5 / len(ordered)
        columns.append(logit(np.clip(q, eps, 1 - eps)))
    return np.column_stack(columns)

inner_oof = np.full((len(train), 2), np.nan)
outer_by_inner_model = []
for k, (fit, score) in enumerate(
    StratifiedKFold(3, shuffle=True, random_state=2026).split(x_train, y_train)
):
    fit_rows = train[fit]
    score_rows = train[score]
    assert np.all(folds[fit_rows] != 0) and np.all(folds[score_rows] != 0)
    print(f"Inner fold {k + 1}/3, fitting on {len(fit_rows)} rows", flush=True)
    batch = pd.concat([x_train.iloc[score], x_valid])
    scores = fit_score(x_train.iloc[fit], y_train[fit], batch)
    inner_oof[score] = scores[:len(score)]
    outer_by_inner_model.append(scores[len(score):])

assert np.isfinite(inner_oof).all()
scaler = StandardScaler()
meta_train = scaler.fit_transform(rank_logit_from_reference(inner_oof, inner_oof))
meta = LogisticRegression(C=0.03, max_iter=2000).fit(meta_train, y_train)

# Apply the same inner-OOF rank references to each component model, then average logits.
outer_design = np.stack([
    rank_logit_from_reference(inner_oof, scores) for scores in outer_by_inner_model
])
assert np.isfinite(outer_design).all()
outer_streams = outer_design.mean(axis=0)
candidate = np.mean([
    meta.decision_function(scaler.transform(design)) for design in outer_design
], axis=0)

def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

champion = np.load(pred / "z_sovereign_v0_oof.npy").ravel()[valid]
base_auc = roc_auc_score(y_valid, champion)
assert abs(base_auc - 0.9512507991341679) < 1e-12
auc = roc_auc_score(y_valid, candidate)
result = {
    "scope": "historical_F0_DEV_ONLY_isolation_diagnostic",
    "streams": ["raw_onehot_LR", "raw_native_categorical_LGBM"],
    "n_outer_train": len(train),
    "n_outer_score": len(valid),
    "inner_folds": 3,
    "inner_split_seed": 2026,
    "saved_champion_auc": base_auc,
    "isolated_stack_auc": auc,
    "delta_e5_vs_saved_champion": (auc - base_auc) * 1e5,
    "isolated_lr_auc": roc_auc_score(y_valid, outer_streams[:, 0]),
    "isolated_lgbm_auc": roc_auc_score(y_valid, outer_streams[:, 1]),
    "outer_train_index_sha256": digest(train),
    "outer_score_index_sha256": digest(valid),
    "candidate_sha256": digest(candidate),
    "milestone": "LOCAL_GATE_PASS",
    "NO_SUBMIT": True,
    "SUBMIT_ARMED": False,
}
payload = json.dumps(result, indent=2)
(log / "codex_loop/ISOLATED_CHAMPION_F0_DIAGNOSTIC.json").write_text(payload + "\n")
np.savez(
    pred / "isolated_champion_f0_diagnostic.npz",
    ids=ids[valid], row_indices=valid, stack=candidate, streams=outer_streams,
)
with (log / "CODEX_DS_REVIEW_NEXT.md").open("a") as f:
    f.write(
        "\n## Isolated champion Fold-0 diagnostic\n\n"
        "Corrected Round7b reproduced solo AUC 0.9368051080303192. "
        "The signed blend scored 0.9512048308028603, Δe-5 −4.5968, 0/5, NO-GO.\n\n"
        f"The isolated two-stream stack scored {auc:.16f} against saved champion "
        f"{base_auc:.16f}, Δe-5 {result['delta_e5_vs_saved_champion']:+.8f}. "
        f"Isolated LR {result['isolated_lr_auc']:.16f}, "
        f"isolated LightGBM {result['isolated_lgbm_auc']:.16f}. "
        "All fitting used the outer training partition. Only historical Fold-0 was scored. "
        "This is a smaller-stack diagnostic. Milestone remains LOCAL_GATE_PASS. "
        "NO_SUBMIT=True, SUBMIT_ARMED=False, RunPod $0, tip CSVs read 0.\n"
    )
print(payload)
