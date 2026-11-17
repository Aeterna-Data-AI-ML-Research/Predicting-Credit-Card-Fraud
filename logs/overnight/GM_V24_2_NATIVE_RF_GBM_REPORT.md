# GM V24.2 Native RF & Gradient Boosting (expert DS)

Elapsed repair+train LGB phase in log · ATH ref **0.95440**

## OOF results

| Model | OOF AUC | Notes |
|---|---:|---|
| **RandomForest** (sklearn native) | **0.934746** | 350 trees, depth22, max_samples0.35, balanced_subsample |
| **HistGradientBoosting** (sklearn native) | **0.944568** | lr0.05, early stop, balanced |
| **LightGBM** (fixed inner-val ES) | **0.940356** | folds mean 0.941679 |
| Rank-blend (0.1, 0.7, 0.20000000000000007) | **0.944720** | |

## Nested gate vs bank-only ATH surrogate

| residual | pos | t | meanΔe5 |
|---|---:|---:|---:|
| `rf` | 0/5 | 0.00 | +0.00 |
| `hgb` | 0/5 | 0.00 | +0.00 |
| `lgb` | 0/5 | 0.00 | +0.00 |
| `blend` | 0/5 | 0.00 | +0.00 |

- Blend orth gate: **False**

## Live

| tag | Public | status |
|---|---:|---|
| `ATH_x_nativeHGB_0.08` | **None** | quota_or_error |

## Log
```
[    0.0s] === V24.2b fix LGB + gates ===
[    9.0s] RF OOF=0.934746 HGB OOF=0.944568
[    9.0s] Retrain LightGBM with inner-holdout early stopping…
[   27.1s]   LGB fold0 AUC=0.938730 iter=218
[   40.3s]   LGB fold1 AUC=0.943136 iter=121
[   54.7s]   LGB fold2 AUC=0.948016 iter=138
[   67.4s]   LGB fold3 AUC=0.941529 iter=159
[   83.8s]   LGB fold4 AUC=0.936986 iter=263
[   83.9s] LGB OOF=0.940356 mean_fold=0.941679
[  113.6s] blend OOF=0.944720 w=RF/HGB/LGB=(0.1, 0.7, 0.20000000000000007)
[  116.5s] ATH surrogate (bank-only) AUC=0.951382
[  117.1s] H2H ATH+orth(rf): 0/5 t=0.00 m=+0.00 min=+0.00 solo=0.93475 ρ(ATH)=1531.7e-4
[  117.7s] H2H ATH+orth(hgb): 0/5 t=0.00 m=+0.00 min=+0.00 solo=0.94457 ρ(ATH)=588.2e-4
[  118.3s] H2H ATH+orth(lgb): 0/5 t=0.00 m=+0.00 min=+0.00 solo=0.94036 ρ(ATH)=663.1e-4
[  119.2s] H2H ATH+orth(blend): 0/5 t=0.00 m=+0.00 min=+0.00 solo=0.94472 ρ(ATH)=588.5e-4
[  119.2s] GATE blend-orth: False
[  128.0s] STAGED sub_v24_2_native_rf.csv ρ(ATH)=1531.708e-4
[  128.9s] STAGED sub_v24_2_native_hgb.csv ρ(ATH)=588.207e-4
[  129.8s] STAGED sub_v24_2_native_lgb.csv ρ(ATH)=663.068e-4
[  130.5s] STAGED sub_v24_2_native_blend.csv ρ(ATH)=588.453e-4
[  131.3s] STAGED sub_v24_2_ATH_x_nativeBlend_0.05.csv ρ(ATH)=1.438e-4
[  132.2s] STAGED sub_v24_2_ATH_x_nativeBlend_0.10.csv ρ(ATH)=5.782e-4
[  133.5s] STAGED sub_v24_2_ATH_x_nativeHGB_0.08.csv ρ(ATH)=3.708e-4
[  134.3s] STAGED sub_v24_2_ATH_orth_nativeBlend_0p35e4.csv ρ(ATH)=0.000e-4
[  134.8s] SUBMIT v24.2b ATH_x_nativeHGB_0.08 | ρ(ATH)=3.708e-4 | RF=0.9347 HGB=0.9446 LGB=0.9404 blend=0.9447
[  138.0s] 
[  138.0s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<00:55, 143kB/s]
  7%|▋         | 528k/7.52M [00:00<00:07, 938kB/s] 
  8%|▊         | 640k/7.52M [00:00<00:08, 902kB/s]
 11%|█         | 816k/7.52M [00:00<00:06, 1.08MB/s]
 
[  138.0s] DONE ATH_x_nativeHGB_0.08 pub=None status=quota_or_error
```

