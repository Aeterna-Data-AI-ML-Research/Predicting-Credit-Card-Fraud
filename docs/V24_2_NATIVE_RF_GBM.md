# V24.2 Native Random Forest & Gradient Boosting

Expert-DS from-scratch models on playground S6E11 (same `folds.csv` as the public OOF bank).

## Protocol

- **62 features:** numerics, bools, cyclic hour/DoW, interactions (`amt/bal`, `vel×txn24`, …), digit + leading-digit family, LOFO target encoding + ordinal cats.
- **Imbalance:** RF `balanced_subsample`, HGB `class_weight=balanced`, LGB `is_unbalance=True`.
- **CV:** 5-fold OOF; LGB early-stops on an **inner 10% holdout** of the train fold (not the OOF fold).

## Results (OOF AUC)

| Model | OOF AUC | Role |
|-------|--------:|------|
| sklearn **RandomForest** | **0.93475** | Native RF baseline |
| sklearn **HistGradientBoosting** | **0.94457** | Native GBM (strongest sklearn) |
| **LightGBM** (fixed) | **0.94036** | Expert GBDT reference |
| Rank-blend (0.10 / 0.70 / 0.20) | **0.94472** | ≈ HGB-dominated |

Public bank anchors for context: `gb_bag5` OOF ≈ 0.9515 — natives are **~70–170e-4 AUC behind** strong bank members.

## Interpretation

1. Native RF/GBM are **competent but not ATH-competitive alone** (public ATH 0.95440 lives on LM-ladder + Δmlp_v15 dual-base, not from-scratch trees).
2. Best use is **small cross-family mass** into ATH (`w≈0.05–0.08` HGB/blend), not orth micro-inject (no nested lift vs bank ATH surrogate at probe scale).
3. Quota was **0** when staging finished — CSVs ready for next reset:
   - `sub_v24_2_native_{rf,hgb,lgb,blend}.csv`
   - `sub_v24_2_ATH_x_nativeHGB_0.08.csv` (priority)
   - `sub_v24_2_ATH_x_nativeBlend_0.05.csv`

## Artifacts

- Predictions: `/tmp/s6e11_v16/out/v24_2_native_rf_gbm.npz`
- Report: `logs/overnight/GM_V24_2_NATIVE_RF_GBM_REPORT.md`
- Scripts: `scripts/gm_v24_2_native_rf_gbm.py`, `scripts/gm_v24_2b_fix_lgb_and_gate.py`
