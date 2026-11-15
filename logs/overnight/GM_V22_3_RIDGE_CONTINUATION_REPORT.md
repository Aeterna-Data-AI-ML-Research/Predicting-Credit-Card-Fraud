# GM V22.3 Ridge Continuation Report

**elapsed** 108s · local Mac · ATH was `0.95425` (P2)

## Live submit
- OK=`True` · **R2_P2_Continuation_plus_DeDiluted_dMLP_1p25e4**
- Public LB: **0.95426**
- 1−ρ vs sriram: `1.250e-4` · vs P2: `0.400e-4`
- max|B_macro|: `0.0107` · c6=`+0.0006` c7=`+0.0007`
- OOF sri `5/5` t=`14.74` min=`+30.61`

## 3-point live curve
- `0.00e-4` → sriram **0.95422**
- `0.55e-4` → P2 **0.95425** (+3e-5)
- `1.25e-4` → this sub **0.95426**

## Replica canary (NOT in submit)
- n=2000 @55s: **timeout** (too slow for full stratified 2k on MPS)
- n=200 quick canary: **AUC=`0.9164`** in `16.9`s (ll_pos−ll_neg, K=1, Target-First replica/final)
- NOT folded into live submit (train-overlap / diagnostic only)

## Staged
- `sub_v22_3_R1_purified_UB_step_1p15e4.csv`
- `sub_v22_3_R2_P2_cont_dediluted_1p25e4.csv`
- `sub_v22_3_LIVE_ridge_continuation.csv`

## Log
```
[    0.6s] === V22.3 START: rebuild anchors + purify U_B ridge ===
[   40.6s] ATH P2 vs sriram 1-ρ=0.550e-4 | P1 vs P2=0.435e-4 | P3 vs P2=0.482e-4
[   49.1s] === Step1: purify U_B vs U_B+U_D+U_E ===
[   50.2s]   U_B_pure: sri 5/5 m=+4.90 min=+2.37 t=5.95 sf=21 max|B|=0.0000
[   51.4s]   U_C_tab: sri 5/5 m=+3.66 min=+3.28 t=5.77 sf=22 max|B|=0.0000
[   52.7s]   U_D_gbn: sri 5/5 m=+1.69 min=+1.02 t=5.09 sf=20 max|B|=0.0000
[   54.1s]   U_E_dMLP: sri 5/5 m=+1.68 min=+0.06 t=2.24 sf=18 max|B|=0.0024
[   55.8s] vector1 pure_UB sri t=5.95 | vector2 UB+UD+UE sri t=7.55
[   55.9s] === Step1b: build R1 / R2 from P2 ATH ===
[   63.0s] R1: λ=0.120 w=0.02630 ρs=1.150e-4 ρ(P2)=1.263e-4 sri 5/5 t=13.65 min=+26.85
[   71.3s] R2 selected mix direction UB+UD+UE
[   71.4s] R2: w=0.01542 ρs=1.250e-4 ρ(P2)=0.400e-4 sri 5/5 t=14.74 min=+30.61
[   71.4s] === R3 audit staged P1/P3 vs P2 ===
[   71.5s]   P1: ρ(P2)=0.435e-4 ρ(sriram)=0.400e-4 ρ(C1)=26.964e-4
[   71.7s]   P3: ρ(P2)=0.482e-4 ρ(sriram)=0.500e-4 ρ(C1)=26.937e-4
[   71.9s] R1_P2_Purified_UB_Step_1p15e4: ρs=1.150 ρP2=1.263 max|B|=0.1052 c6=-0.062 c7=+0.026 | sri 5/5 t=13.65 min=+26.85 | z9 5/5 t=3.95 | z9d 5/5 t=3.23
[   72.1s] R2_P2_Continuation_plus_DeDiluted_dMLP_1p25e4: ρs=1.250 ρP2=0.400 max|B|=0.0107 c6=+0.001 c7=+0.001 | sri 5/5 t=14.74 min=+30.61 | z9 5/5 t=4.78 | z9d 5/5 t=3.73
[   72.1s] === Step2: 45–60s replica/final canary probe ===
[   72.1s] replica probe failed: No module named 'torch'
[   72.1s] probe summary: {"ok": false, "auc_f0": null, "n": 0, "seconds": 0.000698089599609375, "corr_ms": null, "corr_g26": null} | probe_error: No module named 'torch'
[   72.1s] SELECT R2_P2_Continuation_plus_DeDiluted_dMLP_1p25e4
[   73.1s] STAGED sub_v22_3_R1_purified_UB_step_1p15e4.csv
[   73.7s] STAGED sub_v22_3_R2_P2_cont_dediluted_1p25e4.csv
[   74.5s] STAGED sub_v22_3_LIVE_ridge_continuation.csv
[   74.5s] === Step3: submit ONE continuation ===
[   74.5s] SUBMIT v22.3 R2_P2_Continuation_plus_DeDiluted_dMLP_1p25e4 | ρs=1.250e-4 ρP2=0.400e-4 c7=+0.001 OOF sri 5/5 t=14.74
[   80.0s] Submission ref: 58388832
6 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[   80.1s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  2%|▏         | 144k/7.52M [00:00<00:05, 1.31MB/s]
  7%|▋         | 528k/7.52M [00:00<00:08, 896kB/s] 
  8%|▊         | 640k/7.52M [00:00<00:08, 850kB/s]
 11%|█         | 832k/7.52M [00:00<00:07, 891kB/s]
 32%|███▏      | 2.44M/7.52M [00:01<00:01, 4.07MB/s]
 80%|████████  | 6.02M/7.52M [00:01<00:00, 11.4MB/s]
100%|██████████| 7.52M/7.52M [00:01<00:00, 3.96MB/s]
[   88.9s] poll 58388832,sub_v22_3_LIVE_ridge_continuation.csv,2026-11-15 14:41:05.310000,v22.3 R2_P2_Continuation_plus_DeDiluted_dMLP_1p25e4 | ρs=1.250e-4 ρP2=0.400e-4 c7=+0.001 OOF sri 5/5 t=14.74,SubmissionStatus.PENDING,,
[  107.8s] poll 58388832,sub_v22_3_LIVE_ridge_continuation.csv,2026-11-15 14:41:05.310000,v22.3 R2_P2_Continuation_plus_DeDiluted_dMLP_1p25e4 | ρs=1.250e-4 ρP2=0.400e-4 c7=+0.001 OOF sri 5/5 t=14.74,SubmissionStatus.COMPLETE,0.95426,
[  107.8s] curve: {'0.00e-4_sriram': 0.95422, '0.55e-4_P2': 0.95425, '1.25e-4_R': 0.95426}
```

