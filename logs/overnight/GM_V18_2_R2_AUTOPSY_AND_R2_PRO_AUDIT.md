# GM V18.2 — R2 Autopsy & Leak-Free `R2_pro` Upgrade Audit

**STRICT `NO_SUBMIT=True` · 0 GPU pods · local CPU (~376s; over 50s budget due to 10×LGB + 3×Cat + 3×HGB bags)**  
Script: `scripts/gm_v18_2_upgrade_r2_newton_ensemble_on_slot3.py`  
Artifacts: `/tmp/s6e11_v16/out/v18_2.json`, `v18_2_r2_pro.npz`

---

## Verdict

**Do not replace Slot5’s 1-seed `R2_v178`.** Expanded multi-arch `R2_pro` is **weaker and leakier** (F2 fail) than the live `+5e-5` winner. Autopsy shows Slot5 was driven by **amount rarity + balance cents + cat×amount_ones + q1000 sub-bins**, not TE.

| Candidate | F1 | F2 | vs Slot5 replace | Stage |
|---|---|---|---|---|
| **R2_v178** (live) | ✓ 0.991 | ✓ 0.033 | baseline | hold ATH |
| R2_lgb_bag | ✓ | ✗ 0.059 | — | no |
| R2_cat_bag | ✓ | ✗ 0.045 | — | no |
| R2_hgb_bag | ✓ | ✗ 0.094 | — | no |
| **R2_pro** | ✓ 0.982 | ✗ **0.058** | **1/5 all bases, Δ≈−2e-5** | **NO** |

---

## Step 1 — V17.8 R2 feature autopsy (what made `0.95389→0.95394`)

**Confirmed:** R2’s `X` has **45 columns = 39 raw key integers + `amt_logfreq` + 4 continuous nums + `amt_orig_cnt`**. **Zero in-sample TE columns** (TE was R1-only).

`ρ(rebuilt seed=42 R2, cached V17.8 R2) = 0.902` — original V17.8 omitted `random_state`; importances below are from the exact recipe with `seed=42`.

### Top 20 by total split **gain** (5 folds summed)

| # | Feature | Gain | Splits | Family |
|---|---|---|---|---|
| 1 | **`amt_logfreq`** | 13.7 | 82 | exact amount rarity in train∪test |
| 2 | **`account_balance_usd_cents`** | 13.4 | 95 | decimal mantissa |
| 3 | **`merchant_category_x_amtones`** | 10.5 | 58 | cat×digit cross |
| 4 | **`velocity_score_q1000`** | 10.0 | 57 | coarse quantile bin |
| 5 | `cvv_retry_count_ones` | 9.0 | 39 | ones digit |
| 6 | `account_balance_usd_q1000` | 7.2 | 50 | q1000 |
| 7 | `card_age_months_ones` | 6.3 | 47 | ones digit |
| 8 | `merchant_risk_score` | 6.3 | 29 | raw continuous |
| 9 | `merchant_risk_score_cents` | 6.3 | 35 | cents |
| 10 | `device_type_x_amtones` | 6.1 | 35 | cat×digit |
| 11–20 | distance/hours/amount q1000 & cents, raw amount/velocity, `amt_freq_bin` | … | … | sub-bin / raw |

**Split-count leaders** mirror gain: `account_balance_usd_cents` (95), `amt_logfreq` (82), `merchant_category_x_amtones` (58).

**Not in Top 20:** `amt_orig_bin` / `amt_orig_cnt` — orig20k amount hits were secondary; the live jump is mostly **synthetic-generator frequency + fine decimal structure + merchant×amount digit**.

---

## Step 2 — Leak-free `R2_pro` (no TE)

| Bank | Detail |
|---|---|
| `X_raw` | **145** cols: V17.8 45 + mod5 + all-col `logfreq`/`origcnt` + row orig hits + pairwise ratio/diff + cat-pair freqs (**zero TE**) |
| Family A | 5 seeds × 2 leaves → **10** LGB models |
| Family B | 3-seed CatBoost depth=3, `l2=50`, 120 iter |
| Family C | 3-seed HistGB `max_leaf=6`, `min_leaf=500` |
| `R2_pro` | `unitize(0.45·lgb + 0.35·cat + 0.20·hgb)` |

### `cross_fold_r` (test fold agreement)

| Signal | cross_fold_r |
|---|---|
| R2_v178 | 0.283 |
| R2_lgb_bag | 0.229 |
| R2_cat_bag | **0.315** |
| R2_hgb_bag | 0.182 |
| R2_pro | **0.336** (most stable fold-to-fold) |

Higher stability ≠ higher lift: bags diluted the sharp residual direction that 1-seed shallow LGB locked onto.

---

## Step 3 — Forensic + Slot3 scorecard + replace Slot5

### Absolute lift on `z_S3` / twin / NNLS (orth vs Bo9)

| Signal | F2 | z_S3 | Slot3_twin | NNLS |
|---|---|---|---|---|
| **R2_v178** | ✓ 0.033 | **5/5 +3.72** sf19 t3.81 | **5/5 +4.56** sf20 t3.17 | **5/5 +3.74** sf19 |
| R2_lgb_bag | ✗ 0.059 | 3/5 +0.28 | 3/5 | 4/5 |
| R2_cat_bag | ✗ 0.045 | 4/5 +2.25 | 4/5 | 4/5 |
| R2_hgb_bag | ✗ 0.094 | 3/5 | 3/5 | 4/5 |
| **R2_pro** | ✗ 0.058 | 5/5 +0.83 sf**13** t1.47 | 5/5 +1.76 sf16 | 5/5 +0.97 sf14 |

V17.8 reference (cached, prior audit): S3 +4.02 / twin +4.99 / NNLS +4.30, sf20, t≈3.7 — rebuilt seed=42 is slightly softer but same regime.

### Replace `0.020·R2_v178` with `w·R2_pro`

| Base | Best replace | pos | meanΔ×1e5 | min |
|---|---|---|---|---|
| z_S3 | w=0.015 | **1/5** | **−2.32** | −5.01 |
| Slot3_twin | w=0.015 | **1/5** | **−2.95** | −7.67 |
| NNLS_twin | w=0.020 | **1/5** | **−1.83** | −4.60 |

`R2_pro` **loses to Slot5 on every baseline.** Expanded features + multi-seed averaging pushed preρ(z_S3) from 0.060 → 0.124 and **F2 leak to MS/LM/GPD** — classic “more capacity → more Bo9-collinear residual.”

---

## Staged NOSUBMIT

**None.** No candidate passed `F1∧F2∧F3` and beat Slot5.

---

## Guardrails

| Check | Status |
|---|---|
| Kaggle submit | **NOT done** |
| Pods | **0** |
| Hold ATH | **Slot5 = 0.95394** |

---

## Bottom line

1. Slot5’s `+5e-5` came from **raw** `amt_logfreq`, **balance cents**, **merchant×amount_ones**, and **q1000** — leak-free shallow LGB on Newton `g`.  
2. Bigger bags / more features **hurt**: F2 breaks and replace-vs-Slot5 is deeply negative.  
3. Next residual work should **keep the V17.8 recipe tiny** (or bag *only* seeds on the **same 45 cols / 6 leaves / 40 trees**) — not orth-out R2 and not grow the feature bank.

---

*End GM V18.2*
