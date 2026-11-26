#!/usr/bin/env python3
"""CPU-only conditional permutation pooling on sealed Fold-0 predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
NO_SUBMIT = True
SUBMIT_ARMED = False
FLAGS = [
    'is_foreign_transaction', 'is_new_merchant', 'used_vpn',
    'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt',
]
RAW = ['merchant_risk_score', 'velocity_score', 'cvv_retry_count', 'prior_disputes', *FLAGS]


def design(parts, raw, train, residual, tabular):
    columns = []
    pooled = []
    for orders in parts:
        mean = orders.mean(axis=1)
        std = orders.std(axis=1)
        quantile = (rankdata(mean) - 0.5) / len(mean)
        score = logit(quantile)
        pooled.append(score)
        contrasts = np.column_stack([
            orders - mean[:, None], np.log1p(std), orders.max(axis=1) - mean,
            mean / np.sqrt(1 + np.pi / 8 * orders.shape[1] * std**2),
        ])
        low, high = np.quantile(contrasts[train], [0.005, 0.995], axis=0)
        contrasts = contrasts.clip(low, high)
        columns.extend([
            score[:, None], np.maximum(score[:, None] - [-3, -1, 1, 3], 0),
            contrasts, contrasts * np.tanh(score[:, None] / 2),
        ])
    consensus = np.mean(pooled, axis=0)
    if tabular is not None:
        columns.extend([
            tabular[:, None],
            np.maximum(tabular[:, None] - [-3, -1, 1, 3], 0),
            (tabular * np.tanh(consensus / 2))[:, None],
        ])
    risk, velocity = raw[:, :2].T
    clean = (raw[:, 2:] == 0).all(axis=1)
    channels = np.column_stack([clean, raw[:, 2:], risk, velocity, risk * velocity])
    columns.extend([channels, channels * np.tanh(consensus[:, None] / 2)])
    if residual:
        columns.extend([
            np.maximum(risk[:, None] - [13, 20, 27, 34, 45, 60], 0),
            np.maximum(velocity[:, None] - [4, 6, 10, 20, 40], 0),
            channels * np.tanh(consensus[:, None] / 2 - 1),
        ])
    x = np.column_stack(columns)
    return (x - x[train].mean(axis=0)) / np.maximum(x[train].std(axis=0), 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regime-splines', action='store_true')
    parser.add_argument('--tabular', action='store_true')
    args = parser.parse_args()
    pred = ROOT / 'preds'
    sovereign = pred / 'v47_sovereign'
    frame = pd.read_csv(ROOT / 'data/train.csv', usecols=['is_fraud', *RAW])
    y_full = frame['is_fraud'].to_numpy()
    f0 = np.load(sovereign / 'folds_skf42.npy') == 0
    expected = next(StratifiedKFold(5, shuffle=True, random_state=42).split(y_full, y_full))[1]
    np.testing.assert_array_equal(np.flatnonzero(f0), expected)
    assert f0.sum() == 139992
    y = y_full[f0]
    baseline = np.load(sovereign / 'z_sovereign_v0_oof.npy').ravel()[f0]
    base_auc = roc_auc_score(y, baseline)
    assert abs(base_auc - 0.9512507991341679) < 1e-12

    l7 = np.load(pred / 'l7_s2026_5f/l7_s2026_oof_5fold_raw.npy', mmap_mode='r')[f0]
    gpt2 = np.load(pred / 'gpt2_med_4ep/oof_0.npy')
    assert l7.shape == (len(y), 10, 2) and gpt2.shape == (len(y), 8, 2)
    parts = [tensor[:, :, 1].astype(float) - tensor[:, :, 0] for tensor in (l7, gpt2)]
    raw = frame.loc[f0, RAW].to_numpy(dtype=float)
    assert all(np.isfinite(array).all() for array in [raw, baseline, *parts])

    tabular = None
    if args.tabular:
        from lightgbm import LGBMClassifier

        features = pd.read_csv(ROOT / 'data/train.csv').drop(columns=['id', 'is_fraud'])
        for name in features.select_dtypes(include=['object', 'str']).columns:
            features[name] = features[name].astype('category')
        tree = LGBMClassifier(
            n_estimators=450, learning_rate=0.045, num_leaves=31,
            min_child_samples=120, reg_lambda=5, n_jobs=4, verbosity=-1,
            random_state=42,
        )
        print('Fitting tabular anchor on the outer training split', flush=True)
        tree.fit(features.loc[~f0], y_full[~f0])
        score = tree.predict(features.loc[f0], raw_score=True)
        tabular = logit((rankdata(score) - 0.5) / len(score))
        assert np.isfinite(tabular).all()
        print(f'Tabular anchor AUC {roc_auc_score(y, score):.12f}', flush=True)

    oof = np.full(len(y), np.nan)
    deltas = []
    for fold, (train, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(raw, y)):
        x = design(parts, raw, train, args.regime_splines, tabular)
        model = LogisticRegression(C=0.03, solver='lbfgs', max_iter=2000)
        model.fit(x[train], y[train])
        assert model.n_iter_[0] < model.max_iter
        oof[valid] = model.decision_function(x[valid])
        delta = roc_auc_score(y[valid], oof[valid]) - roc_auc_score(y[valid], baseline[valid])
        deltas.append(delta)
        print(f'Subfold {fold}: {delta * 1e5:+.8f} e-5', flush=True)
    assert np.isfinite(oof).all()
    auc = roc_auc_score(y, oof)
    delta = auc - base_auc
    positive = sum(d > 0 for d in deltas)
    result = {
        'n': len(y), 'base_auc': base_auc, 'candidate_auc': auc,
        'delta_e5': delta * 1e5, 'positive_subfolds': positive,
        'subfold_deltas_e5': [d * 1e5 for d in deltas],
        'verdict': 'GO' if delta >= 8e-5 and positive >= 4 else 'NO-GO',
        'NO_SUBMIT': NO_SUBMIT, 'SUBMIT_ARMED': SUBMIT_ARMED,
    }
    name = 'conditional_f0'
    if args.tabular:
        name += '_tabular'
    if args.regime_splines:
        name += '_splines'
    np.save(sovereign / f'{name}_oof.npy', oof)
    payload = json.dumps(result, indent=2)
    (ROOT / 'logs/v47' / f'{name}.json').write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
