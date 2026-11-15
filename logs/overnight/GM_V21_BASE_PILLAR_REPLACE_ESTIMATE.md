# GM V21 — Base Pillar Replace Estimate (local CPU)

**elapsed** 1560s · bank=102 · HC members=6

## Score estimate
- Nested CV AUC: **0.952426**
- Full HC OOF AUC: **0.952501**
- LOFO-NNLS AUC: **0.952428**
- Slot8-twin OOF: None
- Bias: n/a
- **Estimated LB (nested): 0.952426  (Δ vs Slot9 -156.44e-5)**
- **Estimated LB (nnls): 0.952428  (Δ vs Slot9 -156.23e-5)**
- Slot8 LB=0.95395 · Slot9 ATH=0.95399 · target 0.95784 gap still 541e-5 if est holds

## New bases (OOF AUC)
```
v21_boost_j: 0.950173570948985
v21_glm_te: 0.9367863853275212
v21_lgb_te: 0.9497755211662757
v21_orig_lgb: n/a
v21_orig_ridge: n/a
masaya_gpt2: 0.9500272037932737
gpt2_s2026: 0.9501396990738895
math:boost_j: 0.9510261328790263
math:glm_j: 0.9507740148371077
```

## Selected HC members
```
pred:P_restack_pub
pred:P_super_v6_grandmaster
pred:P_super_v4
q17:sameer_glm10f
q17:megayak_5f_lm
pred:P_super_v6_clean
```

Staged (no submit): `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v21_NOSUBMIT_wide_hc_estimate.csv`

## Log
```
[    0.0s] GM V21 base-pillar replace estimate — local CPU
[    5.1s] n_tr=699960 n_te=299984 n_jobs=10
[    5.1s] === A/C) Assemble wide OOF bank ===
[    7.4s]   Q17 streams loaded: 16
[   44.7s]   bank size=99
[   44.7s]     0.952528  pred:P_super_v6_grandmaster
[   44.7s]     0.952461  pred:P_restack_pub
[   44.7s]     0.952257  pred:P_super_v4
[   44.7s]     0.952253  pred:P_super_v5_full
[   44.7s]     0.952202  pred:P_super_v6_clean
[   44.7s]     0.952172  pred:P_super_v5_pure
[   44.7s]     0.952142  pred:P_super_champ
[   44.7s]     0.952138  pred:P_champ
[   44.7s]     0.951595  q17:sameer_glm10f
[   44.7s]     0.951506  q17:gb_bag5
[   44.7s]     0.951470  q17:megayak_shallow_tree
[   44.7s]     0.951223  q17:gbn_bag5_w512
[   44.7s]     0.951144  harv:xgb
[   44.7s]     0.951071  harv:xgb_clean
[   44.7s]     0.951026  math:boost_j
[   44.7s] === B) Digit-key TE + GLM + 4-leaf LGB offset ===
[   48.8s] keys=73 K=(999944, 73)
[   95.0s]   fold0 glm=0.93598 lgb=0.94865 boost=0.94877 trees=1571/1187
[  135.2s]   fold1 glm=0.93640 lgb=0.94993 boost=0.95058 trees=1507/883
[  172.1s]   fold2 glm=0.94455 lgb=0.95511 boost=0.95563 trees=1273/1025
[  219.3s]   fold3 glm=0.93619 lgb=0.95065 boost=0.95098 trees=1484/1287
[  261.6s]   fold4 glm=0.93106 lgb=0.94471 boost=0.94513 trees=1283/1218
[  261.7s]   v21_glm_te OOF AUC=0.936786
[  261.8s]   v21_lgb_te OOF AUC=0.949776
[  261.9s]   v21_boost_j OOF AUC=0.950174
[  262.2s] === D) orig20k collisions + knn fraud fingerprints ===
[  262.4s] orig rows=20000 shared_cols=24
[  262.4s]   computing collision feats…
[  266.8s]   computing knn feats (k=32)…
[  331.1s]   mantissa knn…
[  342.2s]   fold0 orig_lgb=0.89369 trees=341
[  351.6s]   fold1 orig_lgb=0.89498 trees=371
[  359.3s]   fold2 orig_lgb=0.89550 trees=293
[  369.1s]   fold3 orig_lgb=0.89627 trees=418
[  379.1s]   fold4 orig_lgb=0.88989 trees=423
[  379.7s]   v21_orig_lgb OOF=0.894026
[  379.8s]   v21_orig_ridge OOF=0.822119
[  380.0s] === Rebuild Slot8 twin OOF (baseline) ===
[  380.6s] twin rebuild failed: [Errno 2] No such file or directory: '/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v16_3_LIVE_slot3_FamilyE_w0.01045.csv'
[  381.3s] === C) Nested CV forward hill-climb ===
[  409.4s]   unique members=100 (from 102)
[  638.3s]   outer0 members=5 trainAUC=0.952760 hold=0.951468
[  832.2s]   outer1 members=3 trainAUC=0.952517 hold=0.952549
[ 1023.8s]   outer2 members=3 trainAUC=0.951363 hold=0.957186
[ 1310.6s]   outer3 members=3 trainAUC=0.952450 hold=0.952819
[ 1504.4s]   outer4 members=3 trainAUC=0.953626 hold=0.948124
[ 1504.5s]   NESTED AUC=0.952426
[ 1504.5s]   frequent members:
[ 1504.5s]     5/5  auc=0.95246  pred:P_restack_pub
[ 1504.5s]     5/5  auc=0.95253  pred:P_super_v6_grandmaster
[ 1504.5s]     3/5  auc=0.95226  pred:P_super_v4
[ 1504.5s]     2/5  auc=0.95159  q17:sameer_glm10f
[ 1504.5s]     1/5  auc=0.95043  q17:megayak_5f_lm
[ 1504.5s]     1/5  auc=0.95220  pred:P_super_v6_clean
[ 1558.0s]   full HC members=6 OOF=0.952501
[ 1558.5s]   LOFO-NNLS on HC members AUC=0.952428
[ 1558.6s]   test 1-ρ vs Slot8=58.870e-4
```

## Follow-up: calibrated pillar REPLACE/ADD vs Slot8-twin
- Slot8-twin OOF **0.951859** · bias(LB9−twin)=**+0.002131**
- V21 nested raw 0.952426 → calibrated est **0.954557** (Δ vs Slot9 **+56.7e-5**)
- **Best follow-up: replace `swap_ms_with_Pv6` estLB=0.954602 (Δ +61.22e-5)**

| kind | name | OOF | estLB | Δe-5 |
|---|---|---:|---:|---:|
| replace | swap_ms_with_Pv6 | 0.952471 | 0.954602 | +61.22 |
| add_pillar | P_v6 | 0.952401 | 0.954533 | +54.27 |
| rank_blend | twin+Pv6+restack+math+masaya | 0.952209 | 0.954341 | +35.07 |
| add_pillar | P_restack | 0.952206 | 0.954337 | +34.73 |
| replace | swap_ms_with_restack | 0.952205 | 0.954337 | +34.66 |
| multi_orth_logit | math_boost_j+P_v6+P_restack+masaya+math_glm_j | 0.952097 | 0.954228 | +23.85 |
| orth_add | P_v6 | 0.952000 | 0.954131 | +14.10 |
| orth_add | P_restack | 0.951985 | 0.954117 | +12.68 |
| orth_add | g26 | 0.951905 | 0.954037 | +4.65 |
| replace | swap_gb_with_math_glm | 0.951880 | 0.954012 | +2.19 |
| add_pillar | math_glm_j | 0.951871 | 0.954002 | +1.24 |
| add_pillar | masaya | 0.951871 | 0.954002 | +1.24 |

### Verdict
Wide-bank HC did **not** select new TE/LM/orig pillars (supers dominate).
Calibrated pillar replace/add lifts are small vs Slot9; **not a path to 0.95784**.
Gap to 0.95784 remains ~324e-5 on best estimate.

## Honest LOFO (no Slot8-test fit)

- Bo9 baseline LOFO-LogReg: **0.952073**
- Best: **ADD_Pv6** = 0.952288 (**+21.47e-5** vs Bo9)
- Projected LB if deltas transfer: **0.954205**
- Gap to 0.95784 after that: **364e-5**

| config | LOFO | Δe-5 |
|---|---:|---:|
| ADD_Pv6 | 0.952288 | +21.47 |
| ADD_Prest | 0.952266 | +19.26 |
| wide_Pv6_Prest_math_masaya | 0.952261 | +18.74 |
| REPLACE_ms→Pv6 + math_boost | 0.952241 | +16.82 |
| REPLACE_ms→Pv6 | 0.952240 | +16.66 |
| REPLACE_ms→Prest | 0.952133 | +6.02 |
| Bo9_baseline_Psv4 | 0.952073 | +0.00 |
| ADD_math_boost | 0.952050 | -2.28 |
| ADD_math_glm | 0.952034 | -3.95 |
| ADD_masaya | 0.951987 | -8.60 |
