# GM v9.1 Hardened — Regime + RayA Lock READY

**NO_SUBMIT=True**. Blacklist enforced. Elapsed `99s`, peak RSS ≈ `845` MB.

## Step 1 — Top-12 `pos=5/5` pairs (zero triplets)

- `flag_mask__ip_mm`
- `flag_mask__foreign`
- `risk_q40__risk_s10`
- `flag_mask__bill_mm`
- `cat__risk_q40`
- `flag_mask__new_merch`
- `vel_q40__vel_s10`
- `foreign__ip_mm`
- `flag_mask__vpn`
- `cat__risk_s10`
- `flag_mask__ai_scam`
- `hrs_q40__vel_q40`

`l46c_top12_5f`: disp=`1.0000` KS=`0.000018`

## Step 2–3 — Diagnostic table (post Gram-Schmidt)

| mode | vec | vs z_base Δ5F | pos | d5 | F0 vs RayA@0.70 | pos | d_sub | cosF0 | cosTe | disp | KS |
|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| A | V_A | -1.22 | 2/5 | `[-3.74, 0.18, -0.65, 1.76, -3.66]` | -4.54 | 1/5 | `[-4.79, -7.51, -1.99, -10.71, 2.3]` | -0.1654 | -0.0000 | 1.0000 | 0.000071 |
| A | V_B | -0.94 | 1/5 | `[-2.07, -2.04, -0.9, -0.71, 1.04]` | -2.85 | 1/5 | `[-3.23, -0.18, -9.96, -4.52, 3.63]` | 0.0238 | 0.0000 | 1.0000 | 0.000048 |
| A | V_C | -1.00 | 1/5 | `[-3.32, -0.14, -0.73, 1.77, -2.58]` | -4.44 | 1/5 | `[-4.7, -5.97, -4.76, -9.83, 3.04]` | -0.1481 | -0.0000 | 1.0000 | 0.000050 |
| B | V_A | -1.06 | 1/5 | `[-2.77, -0.29, -1.07, 1.72, -2.89]` | -3.40 | 1/5 | `[-4.15, -5.63, -3.07, -9.61, 5.46]` | -0.1838 | -0.0000 | 1.0000 | 0.000074 |
| B | V_B | -0.94 | 1/5 | `[-1.96, -2.13, -0.97, -0.62, 0.96]` | -2.85 | 1/5 | `[-3.47, -0.11, -10.1, -4.37, 3.8]` | 0.0275 | -0.0000 | 1.0000 | 0.000056 |
| B | V_C | -0.86 | 1/5 | `[-2.5, -0.58, -0.97, 1.66, -1.93]` | -3.39 | 1/5 | `[-4.12, -4.38, -5.1, -8.94, 5.61]` | -0.1624 | -0.0000 | 1.0000 | 0.000081 |

**Selected `u_reg_orth`:** `V_B_MB` (F0 `-2.85`@1/5, vs z_base `-0.94`@1/5, Δρ_orth budget `1.200e-06`)

## Step 4 — Staged slots from S7 (NOT live-submitted)

| slot | file | t | pred_lin | F0vsS7 e-5 | gate | PASS |
|---:|---|---:|---:|---:|---|---:|
| 1 | `sub_v9p1_slot1_pure_rayA_0p22e4.csv` | 1.505 | 0.953574 | +2.28 | N/A | True |
| 2 | `sub_v9p1_slot2_rayA_0p35e4_plus_orth_reg.csv` | 1.646 | 0.953584 | +3.39 | F0vsS7>7.83 & SubCV 5/5 (or SubCV mean≥4 @5/5) | False |
| 3 | `sub_v9p1_slot3_pure_rayA_f0peak_0p70e4.csv` | 1.872 | 0.953600 | +6.38 | N/A | True |
| 4 | `sub_v9p1_slot4_rayA_f0peak_0p70e4_plus_orth_reg.csv` | 1.893 | 0.953602 | +5.98 | F0vsS7>10.25 & SubCV 5/5 (or SubCV mean≥4.5 @5/5) | False |

### Fire order (review)
1. Slot 1 pure RayA `0.22e-4` — safest open past ATH (`F0vsS7=+2.28e-5`, pred≈0.953574).
2. Slot 3 pure RayA Fold0 peak `0.70e-4` — pure control (`F0vsS7=+6.38e-5`, pred≈0.953600).
3. Slot 2 hybrid — **gate FAIL**; do not fire.
4. Slot 4 hybrid peak — **gate FAIL**; do not fire.

### Hardening verdict
- Top-12 condensation OK (`disp=1.0000`, `KS=1.8e-5`).
- Test-side Gram-Schmidt succeeds (`cos(u,v_RayA)≈0` on test).
- Fold-0 LOFO cos is **not** exactly 0 (≈±0.02…0.18) — expected under LOFO vs exact test projection.
- After full RayA-subspace decoupling, **all** `V_A/B/C` lose the prior `+6.49e-5` Fold0 regime lift (that lift was RayA-entangled). Hybrids therefore fail hard gates; **only pure RayA slots are live-eligible today**.

**NO_SUBMIT=True** — no Kaggle API submit.
