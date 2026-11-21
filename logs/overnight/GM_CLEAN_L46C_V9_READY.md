# GM Clean L4.6C V9 — Ready for Slot 1 Review

**NO_SUBMIT=True**. Poison pills blacklisted: `v10_*`, `soft_diff*`, `gb_soft*`, `idf_p3*`, `clean035*`, `zanch_*`.

Elapsed `425s`, peak RSS ≈ `1179` MB.

## Step 1 — Why L4.6C won Fold0 5/5

Legacy `P4_L4.6C_hier2d` = **20** 2-way pairs condensed via hierarchical excess `Δ = E[y − σ(η_A+η_B−η0) | AB]` (m∈{10,50} bag) → Ridge/newton → PCHIP align.

| fold | ρ(C,RayA) | slope | approx dAUC e-5 |
|---:|---:|---:|---:|
| 0 | -0.4259 | +0.00494 | +0.57 |
| 1 | -0.3916 | +0.00356 | +0.16 |
| 2 | -0.4389 | +0.00440 | -2.94 |
| 3 | -0.4576 | +0.00511 | -0.64 |
| 4 | -0.4439 | +0.00445 | -2.11 |

Legacy LOFO vs RayA: **-0.99e-5** `2/5`; Fold0 SubCV **+0.53e-5** `3/5`.

**Diagnosis:** Fold0 SubCV captures a real hierarchical interaction signal; 5F dips because low-support / sign-unstable pairs among the original 20 dilute F1–4. Fix = keep only pairs with **≥4/5** outer-fold positive residual LL, then expand.

## Expansion screen

- Singles: **28** (cats/flags/qbins/steps; not KMAT-distilled)
- Pairs scored: **378**; kept ≥4/5 LL+: **140** (used top 40)
- Triplets kept: **60** (used top 30)
- Condensation cols → top **45** by |corr| with RayA residual

Top pairs:
- `flag_mask__ip_mm` pos=5/5 meanLL=+2.851588e-03
- `flag_mask__foreign` pos=5/5 meanLL=+1.639659e-03
- `risk_q40__risk_s10` pos=5/5 meanLL=+1.636064e-03
- `flag_mask__bill_mm` pos=5/5 meanLL=+1.462738e-03
- `cat__risk_q40` pos=5/5 meanLL=+1.205586e-03
- `flag_mask__new_merch` pos=5/5 meanLL=+8.882641e-04
- `vel_q40__vel_s10` pos=5/5 meanLL=+8.679682e-04
- `foreign__ip_mm` pos=5/5 meanLL=+6.135754e-04
- `flag_mask__vpn` pos=5/5 meanLL=+5.847929e-04
- `cat__risk_s10` pos=5/5 meanLL=+4.107683e-04

## Step 2 — Estimators + gates (clean)

- C1 `l46c_ridge_clean`: disp=`1.0000` KS=`0.000023`
- C2 `l46c_stageB_lgb`: disp=`1.0000` KS=`0.000010`

| name | vs RayA Δ5F | pos | vs z_base | pos | F0 SubCV | pos | w |
|---|---:|---:|---:|---:|---:|---:|---:|
| l46c_legacy | -0.99 | 2/5 | -0.47 | 2/5 | +0.53 | 3/5 | 0.01 |
| l46c_ridge_clean | -2.55 | 0/5 | -1.88 | 0/5 | -0.14 | 2/5 | 0.01 |
| l46c_stageB_lgb | -1.55 | 1/5 | -0.80 | 2/5 | -2.49 | 1/5 | 0.01 |
| tree_logit_v72 | -0.04 | 2/5 | +0.22 | 3/5 | +0.15 | 3/5 | 0.02 |
| P4_L4.5 | -1.53 | 1/5 | -1.11 | 1/5 | -3.78 | 2/5 | 0.01 |

**clean_boost** = unit(0.75·`l46c_legacy` + 0.35·`tree_logit_v72`): F0=`+0.83`@3/5 vsRayA=`-0.63`@3/5

### Gate status
- Gate1 (≥4/5 vs RayA among clean vectors): **REVIEW**
- Gate2 (F0 SubCV ≥ +6.12 @5/5 over RayA): **REVIEW** (got `+0.83` @3/5)

## Step 3 — Staged submissions (NOT live-submitted)

| slot | file | t | pred_lin | F0vsS7 e-5 | F0vsRayA e-5 |
|---|---|---:|---:|---:|---:|
| sub_clean_v9_slot1_rayA_plus_l46c_0p20e4.csv | `sub_clean_v9_slot1_rayA_plus_l46c_0p20e4.csv` | 1.485 | 0.953573 | -4.15 | -10.01 |
| sub_clean_v9_slot2_rayA_plus_l46c_0p45e4.csv | `sub_clean_v9_slot2_rayA_plus_l46c_0p45e4.csv` | 1.708 | 0.953589 | -6.37 | -12.23 |
| sub_clean_v9_slot3_rayA_f0peak_plus_l46c_0p75e4.csv | `sub_clean_v9_slot3_rayA_f0peak_plus_l46c_0p75e4.csv` | 1.903 | 0.953602 | -11.19 | -17.05 |
| sub_clean_v9_slot4_pure_rayA_0p35e4_control.csv | `sub_clean_v9_slot4_pure_rayA_0p35e4_control.csv` | 1.627 | 0.953583 | +3.83 | -2.04 |

## Regime recovery (post-hoc, still clean / no v10 / no soft_diff)

| vector | vs RayA | F0 SubCV |
|---|---|---|
| legacy LOFO | −0.99@2/5 | +1.61@4/5 |
| **legacy + regime** | −1.07@2/5 | **+4.63@5/5** |
| ridge_clean + regime | −2.80@1/5 | −5.45@1/5 |
| stageB + regime | −1.47@0/5 | −3.26@1/5 |
| **legacy_reg + tree_reg (0.65/0.35)** | −0.46@3/5 | **+6.49@5/5** |

Phase-A’s **+6.12@5/5** Fold0 signal is real under newton-cholesky regime on legacy L4.6C; the expanded ridge/Stage-B condensates did **not** beat it (over-expanded / noisy columns). Gate2-style Fold0 blend clears **+6.49@5/5**, but **Gate1 5F vs RayA still fails**.

### Live-fire recommendation (NO_SUBMIT still True)
- **Do not fire Slots 1–3** as staged — OOF F0 vs S7 is **negative** (−4 to −11e-5).
- **Only OOF-positive staged file:** Slot 4 pure RayA control (`+3.83e-5` F0 vs S7).
- Next research (before any L46C live add): re-condense using **only top LL pairs** (flag×risk/vel family) + **regime**, then re-stage micro boost; do not ship the current expanded C1/C2.

**NO_SUBMIT=True** — no Kaggle API submit from this script.
