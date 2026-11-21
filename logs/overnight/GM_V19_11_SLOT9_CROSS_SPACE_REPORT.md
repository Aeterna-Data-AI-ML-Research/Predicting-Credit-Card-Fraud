# GM V19.11 — Cross-space cohort×order Cat/ET on Slot8 → **HOLD**

**0 GPU pods · elapsed 1671s · script `scripts/gm_v19_11_cross_space_cohort_order_cat_on_slot8.py`**

| | |
|---|---|
| Slot8 ATH | **0.95395** (hold) |
| Slot8 rebuild `1-ρ` | **0** (exact match) |
| Submit | **False** — 0 winners |
| Public LB | — |
| Daily slots used | **0** (2 remain) |

## Shapes
- `X_new_only` (cohort+order+ratios, no tab45 / no fine peers): `(699960, 55)`
- `X_cross` (`X_new` + tab keys): `(699960, 83)`
- `X_c4` (coarse SafePeer + cohort): `(699960, 36)`

## Step1 — Slot8 base & fail rays
- Rebuilt `z_S8 = z_S3 + 0.016·R2 + 0.012·CatObliv + 0.012·U5` with **exact V19.4 U5** (`L(gini)` per test fold → mean).
- `qlock(L(Slot3)+…, LIVE)` vs `sub_v19_4_LIVE_slot8_*`: **`1-ρ = 0`**.
- Pre-train depoison: project `[u_M5, u_R1]` out of cohort/order/ratio cols before fitting.
- Orth basis for residuals: `[Bo9, R2, Cat, U5, M5, R1]` (`alpha=10`).

## Step2 — Family forensics (post-orth)

| Family | raw_disp | raw_KS | F2 vs Bo12 | F3 vs S8−S5 | corr(R2) | corr(Cat) | \|c6\| | \|c7\| |
|--------|----------|--------|-----------|-------------|----------|-----------|--------|--------|
| C1_cat_new_only | 0.982 | 0.017 | **0.053** ❌ | 0.021 | −0.003 | −0.002 | 0.021 | 0.058 |
| C2_cat_cross88 | 0.963 | 0.013 | **0.058** ❌ | 0.002 | +0.002 | +0.003 | 0.009 | 0.058 |
| C3_et_cross | 0.966 | 0.005 | **0.047** ❌ | 0.009 | +0.004 | +0.002 | 0.030 | 0.030 |
| C4_SafePeer_Coarse | 0.979 | 0.017 | **0.044** ❌ | 0.021 | −0.007 | −0.005 | 0.024 | 0.055 |
| B12 / B13 / B123 | ~0.98 | ~0.016 | **0.049–0.055** ❌ | ≤0.022 | ≈0 | ≈0 | ≤0.025 | ≤0.060 |

**Anti-fail-ray success:** every family has `|c6|<0.20` and `|c7|<0.20` (typically `|c7|≈0.03–0.06`). Excluding fine `k1..k8` + orth vs M5/R1 killed the V19.9/V19.10 Slot7 poison.

**Gate failures (common):**
1. **F2 ≥ 0.040** on all families (Bo12 residual leakage ~0.044–0.058).
2. **Never 5/5×3** vs Slot8 — best is **4/5** with `min_fold < 0` and `t ≪ 2.8`.
3. At weights where `1-ρ ∈ [0.30, 1.20]e-4`, twin/nnls often go negative; at tiny `w` where OOF is least bad, `1-ρ` is too small (~0.10–0.17e-4).

## C4 SafePeer coarse support
| key | mean_cnt | frac in [50,800] |
|-----|----------|------------------|
| k_new1 `(cat,ch×dev,amt_r10)` | 1645 | 0.355 |
| k_new2 `(cat,foreign,bal_r100)` | 4832 | 0.138 |
| k_new3 `(ch×dev,mr_r1,vel_r0)` | **5.4** | **0.000** (dead) |
| k_new4 `(cat,amt_ones,bal_ones)` | 1416 | 0.311 |

Coarse peers did **not** recover V19.9 `P4` OOF lift once fine `X_peer8` was removed and M5 was in the orth basis.

## Best additive H2H (not gate-clear)

| Candidate | 1-ρ e-4 | c7 | z_S8 | twin | nnls |
|-----------|---------|----|------|------|------|
| C1_w0.010 | 0.158 | +0.058 | 4/5 m=+0.46 min=−1.93 | 4/5 m=+0.28 min=−4.01 | 4/5 m=+0.52 min=−1.99 |
| C2_w0.015 | 0.359 | +0.058 | 3/5 m=+0.68 min=−1.45 | 2/5 m=−0.12 min=−4.50 | 3/5 m=+0.74 min=−1.42 |
| B12_w0.010 | 0.158 | +0.059 | 4/5 m=+0.56 min=−1.42 | 3/5 m=+0.46 min=−3.18 | 4/5 m=+0.63 min=−1.44 |
| B12_joint wo0.016/wc0.012/wu0.010/wn0.010 | 0.168 | +0.041 | 4/5 m=+0.47 min=−1.55 | 4/5 m=+0.52 min=−3.21 | 4/5 m=+0.52 min=−1.58 |

Mean OOF lifts stay **≪ +2e-5** and fold minima are negative — noise, not a Slot9 engine.

## Diagnosis
1. **Fail-ray depoison worked** (`|c7|~0.05` vs V19.9 `P4` `c7=+0.51`).
2. **Cross-space `X_cohort×X_order×X_tab` has no stable residual vs saturated Slot8** after orth to Bo12 — trees rediscover Bo12-correlated margins (`F2>0.04`) with weak, fold-fragile AUC.
3. **C4 without fine peers ≠ P4** — the V18.9/V19.9 jump needed `X_peer8` collision buckets that are structurally Slot7-poisoned.

## Decision
**HOLD** ATH `0.95395`. Do **not** submit. Keep remaining daily slots.

Artifacts: `/tmp/s6e11_v16/out/v19_11.json`, `v19_11_families.npz`, `logs/overnight/GM_V19_11_CROSS_SPACE.log`.
