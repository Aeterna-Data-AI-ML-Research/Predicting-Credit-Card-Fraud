# GM V22.4 Complete Incremental Pivot Report

**elapsed** 159s · local Mac CPU · ATH `0.95426` (R2)

## Decision: **HOLD_0.95426_no_gate_clear**
- Submit OK: `False` · Public LB: **None**
- Passing vectors: `['V8_v10_ridge_gs']`

## Ray-1 quadratic apex
- nominal u*=`1.9994` → (1−ρ)*≈`3.997e-4` s*≈`0.954270`
- conservative (1−ρ)*≈`2.737e-4`
- α=`4.966e-05` β=`2.484e-05`

## U_C projection into R2−sriram (not in GS basis)
- ols weight=`+0.2540`

## Vector scorecard (incremental vs R2)
| vec | w | disp | max\|B\| | sri pos/t/min | PASS |
|---|---:|---:|---:|---|:---:|
| `V1_UC_tab_gs` | +0.0120 | 0.961 | 0.0000 | 5/5 t=6.15 min=+2.91 | n |
| `V2_sameer_glm10f_gs` | +0.0120 | 0.932 | 0.0000 | 5/5 t=4.35 min=+2.14 | n |
| `V3_sadam_cat_gs` | -0.0120 | 1.057 | 0.0000 | 2/5 t=0.35 min=-1.63 | n |
| `V4_P4_L42_gs` | +0.0120 | 0.933 | 0.0000 | 3/5 t=0.99 min=-1.84 | n |
| `V5_P4_L43_rect_gs` | -0.0030 | 0.996 | 0.0000 | 2/5 t=0.18 min=-0.71 | n |
| `V6_P4_L45_gs` | -0.0120 | 0.976 | 0.0000 | 3/5 t=0.52 min=-4.32 | n |
| `V7_B_xgb_llr_gs` | +0.0120 | 1.112 | 0.0000 | 5/5 t=6.83 min=+3.53 | n |
| `V8_v10_ridge_gs` | +0.0120 | 0.952 | 0.0000 | 5/5 t=4.35 min=+0.16 | Y |
| `V9_L7_curv_and_antithetic_gs` | -0.0120 | 0.726 | 0.0000 | 4/5 t=1.08 min=-0.69 | n |
| `V10_L7_U5_U8_gs` | +0.0120 | 0.985 | 0.0000 | 4/5 t=1.47 min=-1.73 | n |
| `V11_P1_raw_peer_gs` | +0.0120 | 0.981 | 0.0000 | 4/5 t=1.28 min=-1.80 | n |
| `V12_C1_orth_gs` | +0.0120 | 0.840 | 0.0000 | 5/5 t=13.13 min=+6.67 | n |

## Staged
- `sub_v22_4_PRIVATE_LB_HEDGE_NOSUBMIT.csv` (NOSUBMIT private hedge)

## Log
```
[    0.2s] === V22.4 START: rebuild + incremental pivot from R2_095426 ===
[   30.7s] R2 vs sriram=1.250e-4 | R2 vs P2=0.400e-4
[   30.7s] === Step1: interval-bounded quadratic apex on Ray 1 ===
[   30.7s] apex nominal u*=1.9994 → (1-ρ)*≈3.997e-4 s*≈0.954270 | conservative u*=1.6545 → (1-ρ)*≈2.737e-4 α=4.966e-05 β=2.484e-05
[   39.0s] z_R2_oof AUC=0.951694 surrogate R²=1.0000
[   39.1s]   proj weight in R2-sriram: UB corr=+0.7157 ols=+0.7157
[   39.1s]   proj weight in R2-sriram: UC_raw corr=+0.2560 ols=+0.2560
[   39.1s]   proj weight in R2-sriram: UD corr=-0.0143 ols=-0.0143
[   39.1s]   proj weight in R2-sriram: UE corr=-0.0210 ols=-0.0210
[   39.1s]   proj weight in R2-sriram: C1 corr=+0.6793 ols=+0.6793
[   39.3s] U_C_tab_gs existing projection weight inside R2-sriram: ols=+0.2540
[   39.3s] === Step2: screen V1..V12 signed LOFO vs R2_095426 ===
[   40.1s]   P4_L4.3 pure KS before=0.0854 after ECDF-lock=0.0380
[   41.2s] loaded 12/12 vectors: ['V1_UC_tab_gs', 'V2_sameer_glm10f_gs', 'V3_sadam_cat_gs', 'V4_P4_L42_gs', 'V5_P4_L43_rect_gs', 'V6_P4_L45_gs', 'V7_B_xgb_llr_gs', 'V8_v10_ridge_gs', 'V9_L7_curv_and_antithetic_gs', 'V10_L7_U5_U8_gs', 'V11_P1_raw_peer_gs', 'V12_C1_orth_gs']
[   50.0s]   V1_UC_tab_gs: w=+0.0120 disp=0.961 KS=0.0094 max|B|=0.0000 c6=+0.000 c7=-0.000 tw=0.117 mono=1.0000 | LOFO 5/5 m=+3.61 min=+2.91 t=11.21 | sri 5/5 t=6.15 min=+2.91 | z9 5/5 t=2.34 | z9d 5/5 t=2.19 | PASS3=False dispOK=True
[   58.2s]   V2_sameer_glm10f_gs: w=+0.0120 disp=0.932 KS=0.0073 max|B|=0.0000 c6=+0.000 c7=-0.000 tw=0.101 mono=1.0000 | LOFO 5/5 m=+3.22 min=+2.14 t=8.05 | sri 5/5 t=4.35 min=+2.14 | z9 4/5 t=1.80 | z9d 4/5 t=2.16 | PASS3=False dispOK=True
[   66.4s]   V3_sadam_cat_gs: w=-0.0120 disp=1.057 KS=0.0095 max|B|=0.0000 c6=+0.000 c7=-0.000 tw=0.098 mono=1.0000 | LOFO 2/5 m=-0.13 min=-1.63 t=-0.30 | sri 2/5 t=0.35 min=-1.63 | z9 0/5 t=-2.82 | z9d 0/5 t=-2.69 | PASS3=False dispOK=True
[   74.3s]   V4_P4_L42_gs: w=+0.0120 disp=0.933 KS=0.1209 max|B|=0.0000 c6=+0.000 c7=-0.000 tw=0.107 mono=1.0000 | LOFO 3/5 m=+0.03 min=-1.84 t=0.05 | sri 3/5 t=0.99 min=-1.84 | z9 2/5 t=-0.44 | z9d 2/5 t=-0.39 | PASS3=False dispOK=True
[   82.5s]   V5_P4_L43_rect_gs: w=-0.0030 disp=0.996 KS=0.0109 max|B|=0.0000 c6=-0.000 c7=+0.000 tw=0.118 mono=1.0000 | LOFO 0/5 m=-1.37 min=-3.03 t=-2.53 | sri 2/5 t=0.18 min=-0.71 | z9 3/5 t=1.13 | z9d 3/5 t=1.37 | PASS3=False dispOK=True
[   90.6s]   V6_P4_L45_gs: w=-0.0120 disp=0.976 KS=0.0772 max|B|=0.0000 c6=+0.000 c7=-0.000 tw=0.105 mono=1.0000 | LOFO 2/5 m=-2.16 min=-6.27 t=-1.39 | sri 3/5 t=0.52 min=-4.32 | z9 2/5 t=0.39 | z9d 2/5 t=0.33 | PASS3=False dispOK=True
[   98.3s]   V7_B_xgb_llr_gs: w=+0.0120 disp=1.112 KS=0.0150 max|B|=0.0000 c6=-0.000 c7=+0.000 tw=0.098 mono=1.0000 | LOFO 5/5 m=+5.21 min=+3.53 t=8.55 | sri 5/5 t=6.83 min=+3.53 | z9 4/5 t=0.97 | z9d 4/5 t=1.18 | PASS3=False dispOK=False
[  107.6s]   V8_v10_ridge_gs: w=+0.0120 disp=0.952 KS=0.0128 max|B|=0.0000 c6=+0.000 c7=-0.000 tw=0.145 mono=1.0000 | LOFO 5/5 m=+2.84 min=+0.16 t=2.89 | sri 5/5 t=4.35 min=+0.16 | z9 5/5 t=3.57 | z9d 5/5 t=3.52 | PASS3=True dispOK=True
[  107.6s]   >>> PASS V8_v10_ridge_gs
[  116.7s]   V9_L7_curv_and_antithetic_gs: w=-0.0120 disp=0.726 KS=0.0648 max|B|=0.0000 c6=+0.000 c7=-0.000 tw=0.104 mono=1.0000 | LOFO 4/5 m=+0.61 min=-0.69 t=1.23 | sri 4/5 t=1.08 min=-0.69 | z9 4/5 t=0.64 | z9d 4/5 t=0.73 | PASS3=False dispOK=False
[  126.0s]   V10_L7_U5_U8_gs: w=+0.0120 disp=0.985 KS=0.0144 max|B|=0.0000 c6=-0.000 c7=+0.000 tw=0.115 mono=1.0000 | LOFO 4/5 m=+1.07 min=-1.73 t=1.46 | sri 4/5 t=1.47 min=-1.73 | z9 3/5 t=0.52 | z9d 4/5 t=0.64 | PASS3=False dispOK=True
[  134.7s]   V11_P1_raw_peer_gs: w=+0.0120 disp=0.981 KS=0.0182 max|B|=0.0000 c6=-0.000 c7=+0.000 tw=0.106 mono=1.0000 | LOFO 4/5 m=+1.11 min=-1.80 t=1.36 | sri 4/5 t=1.28 min=-1.80 | z9 4/5 t=1.08 | z9d 4/5 t=1.16 | PASS3=False dispOK=True
[  143.6s]   V12_C1_orth_gs: w=+0.0120 disp=0.840 KS=0.0233 max|B|=0.0000 c6=+0.000 c7=-0.000 tw=0.098 mono=1.0000 | LOFO 5/5 m=+8.64 min=+6.67 t=8.01 | sri 5/5 t=13.13 min=+6.67 | z9 3/5 t=-0.45 | z9d 3/5 t=0.57 | PASS3=False dispOK=False
[  143.6s] passing vectors: ['V8_v10_ridge_gs']
[  143.6s] === Step3: synthesize Q1/Q2 + anti-shakeup hedge ===
[  148.3s] Q1: w=0.02239 ρ(R2)=0.350e-4 sri 3/5 t=3.20 min=-0.81
[  153.0s] Q2: w=0.02675 ρ(R2)=0.500e-4 (tgt≈0.50) sri 3/5 t=2.64 min=-2.18
[  153.2s]   gate Q1_Incremental_Orth_on_R2: ok=False mc=0.0063 c6=+0.000 c7=-0.000 mono=1.0000
[  153.2s]   gate Q2_Apex_Adjusted_Ray1_plus_Orth: ok=False mc=0.0075 c6=+0.000 c7=-0.000 mono=0.9999
[  153.2s] NO candidate cleared strict 5/5×3 vs R2 — HOLD 0.95426
[  157.7s] STAGED hedge sub_v22_4_PRIVATE_LB_HEDGE_NOSUBMIT.csv ρ(R2)=1.957e-4 (NO SUBMIT)
```

