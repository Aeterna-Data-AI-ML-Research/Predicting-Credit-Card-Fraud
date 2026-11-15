# GM V22.5 Crest + Orth4 Report

**elapsed** 136s · local Mac CPU · ATH `0.95426`

## Decision: **K2_Ray1_Crest_2p05e4_plus_Orth4_0p45e4**
- Submit OK: `True` · Public LB: **0.95426** (tie with R2 ATH — no lift)
- U_orth4 assert: `True` · max|B|=`0.000000` disp=`1.0000`
- Live: `1−ρ(sri)=2.054e-4` (inside cons. crest `[2.0, 2.74]e-4`) · `1−ρ(R2)=0.450e-4`
- Slots remaining after submit: **5**

## Sign-locked vectors (vs z_R2_sri_ridge @ w=0.006)
- `V8` sign=`+1` pos=`5/5` t=`4.86` disp=`1.0000` max|B|=`0.000000`
- `V1` sign=`+1` pos=`5/5` t=`6.30` disp=`1.0000` max|B|=`0.000000`
- `V7` sign=`+1` pos=`5/5` t=`5.62` disp=`1.0000` max|B|=`0.000000`
- `V2` sign=`+1` pos=`5/5` t=`3.93` disp=`1.0000` max|B|=`0.000000`
- `V12` sign=`+1` pos=`5/5` t=`11.38` disp=`1.0000` max|B|=`0.000000`

## Candidates vs R2
| cand | ρ(R2) | ρ(sri) | ridge pos/t/min | nnls | hybrid | z9 mean | max\|B\| dir | Δraw |
|---|---:|---:|---|---|---|---:|---:|---:|
| `K1_Pure_Orth4_on_R2_0p35e4` | 0.350 | 1.900 | 5/5 t=12.03 min=+6.79 | 5/5 t=7.29 | 5/5 t=10.08 | +3.67 | 0.000000 | 0.008223 |
| `K2_Ray1_Crest_2p05e4_plus_Orth4_0p45e4` | 0.450 | 2.054 | 5/5 t=13.79 min=+9.15 | 5/5 t=8.01 | 5/5 t=11.27 | +3.40 | 0.000000 | 0.008888 |
| `K3_Pure_V8_v10_ridge_plus_V1_UC_0p30e4` | 0.300 | 1.636 | 5/5 t=7.10 min=+3.55 | 5/5 t=6.34 | 5/5 t=6.92 | +3.53 | 0.000000 | 0.006590 |

## Staged
- `sub_v22_5_K1_pure_orth4_0p35e4.csv`
- `sub_v22_5_K2_crest_plus_orth4_0p45e4.csv`
- `sub_v22_5_K3_v8_v1_0p30e4.csv`

## Log
```
[    0.2s] === V22.5 START: crest + signed orth4 from R2_095426 ===
[   28.3s] R2 vs sriram=1.250e-4 | R2 vs P2=0.400e-4 | P2 vs sriram=0.550e-4
[   36.1s] === Step1: build z_R2_sri_ridge / nnls / hybrid_twin ===
[   37.0s]   z_R2_sri_ridge AUC=0.951846
[   37.1s]   z_R2_sri_nnls AUC=0.952206
[   37.2s]   z_R2_hybrid_twin AUC=0.952177
[   37.3s]   z_R2_z9 AUC=0.952643 (soft meanΔ≥0 gate only)
[   37.3s] === Step1b: GS-orth + sign + variance lock ===
[   39.2s]   S_V8_v10_ridge: sign=+1 assert5/5=True pos=5/5 m=+1.62 t=4.86 disp=1.0000 max|B|=0.000000 c6=+0.0000 c7=+0.0000
[   40.7s]   S_V1_UC_tab: sign=+1 assert5/5=True pos=5/5 m=+1.98 t=6.30 disp=1.0000 max|B|=0.000000 c6=+0.0000 c7=-0.0000
[   43.2s]   S_V7_B_xgb_llr: sign=+1 assert5/5=True pos=5/5 m=+2.14 t=5.62 disp=1.0000 max|B|=0.000000 c6=-0.0000 c7=+0.0000
[   45.2s]   S_V2_sameer_glm10f: sign=+1 assert5/5=True pos=5/5 m=+1.61 t=3.93 disp=1.0000 max|B|=0.000000 c6=+0.0000 c7=-0.0000
[   47.0s]   S_V12_Ray1_cont: sign=+1 assert5/5=True pos=5/5 m=+4.23 t=11.38 disp=1.0000 max|B|=0.000000 c6=-0.0000 c7=-0.0000
[   47.0s] === Step2: build U_orth4 + K1/K2/K3 ===
[   47.0s]   orth4 weight V8=0.235 (t=4.86)
[   47.0s]   orth4 weight V1=0.304 (t=6.30)
[   47.0s]   orth4 weight V7=0.271 (t=5.62)
[   47.0s]   orth4 weight V2=0.190 (t=3.93)
[   47.0s] U_orth4: disp=1.0000 max|B|=0.000000 c6=-0.0000 c7=+0.0000
[   48.8s]   U_orth4 vs z_R2_sri_ridge: 5/5 m=+6.36 min=+4.57 t=12.82 sf=24
[   48.8s]   U_orth4 vs z_R2_sri_nnls: 5/5 m=+4.18 min=+3.07 t=7.85 sf=24
[   48.8s]   U_orth4 vs z_R2_hybrid_twin: 5/5 m=+5.39 min=+3.91 t=10.26 sf=24
[   48.8s]   U_orth4 vs z_R2_z9: mean=+2.54 pos=5/5 (soft)
[   48.8s] U_orth4 ASSERT ok=True
[   95.3s] K1_Pure_Orth4_on_R2_0p35e4: ρ(R2)=0.350 ρ(sri)=1.900 max|B|=0.000000 (Δraw=0.008223) disp=1.0000 c6=-0.0006 c7=-0.0006 | z_R2_sri_ridge:5/5 t=12.03 min=+6.79 | z_R2_sri_nnls:5/5 t=7.29 min=+4.58 | z_R2_hybrid_twin:5/5 t=10.08 min=+5.81 | z9 mean=+3.67
[   97.8s] K2_Ray1_Crest_2p05e4_plus_Orth4_0p45e4: ρ(R2)=0.450 ρ(sri)=2.054 max|B|=0.000000 (Δraw=0.008888) disp=1.0000 c6=-0.0007 c7=-0.0006 | z_R2_sri_ridge:5/5 t=13.79 min=+9.15 | z_R2_sri_nnls:5/5 t=8.01 min=+5.30 | z_R2_hybrid_twin:5/5 t=11.27 min=+7.35 | z9 mean=+3.40
[   99.9s] K3_Pure_V8_v10_ridge_plus_V1_UC_0p30e4: ρ(R2)=0.300 ρ(sri)=1.636 max|B|=0.000000 (Δraw=0.006590) disp=1.0000 c6=+0.0000 c7=+0.0001 | z_R2_sri_ridge:5/5 t=7.10 min=+3.55 | z_R2_sri_nnls:5/5 t=6.34 min=+3.29 | z_R2_hybrid_twin:5/5 t=6.92 min=+3.19 | z9 mean=+3.53
[  100.6s] STAGED sub_v22_5_K1_pure_orth4_0p35e4.csv
[  101.3s] STAGED sub_v22_5_K2_crest_plus_orth4_0p45e4.csv
[  101.9s] STAGED sub_v22_5_K3_v8_v1_0p30e4.csv
[  101.9s] === Step3: gate check + conditional submit ===
[  101.9s]   gate K2_Ray1_Crest_2p05e4_plus_Orth4_0p45e4: ok=True
[  101.9s]   gate K1_Pure_Orth4_on_R2_0p35e4: ok=True
[  101.9s]   gate K3_Pure_V8_v10_ridge_plus_V1_UC_0p30e4: ok=True
[  102.5s] STAGED sub_v22_5_LIVE_crest_orth4.csv
[  102.5s] SUBMIT v22.5 K2_Ray1_Crest_2p05e4_plus_Orth4_0p45e4 | ρ(R2)=0.450e-4 ρ(sri)=2.054e-4 OOF ridge 5/5 t=13.79 min=+9.15
[  108.4s] Submission ref: 58392107
5 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[  108.4s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<00:48, 162kB/s]
 17%|█▋        | 1.28M/7.52M [00:00<00:00, 7.82MB/s]
 27%|██▋       | 2.03M/7.52M [00:00<00:02, 1.98MB/s]
 33%|███▎      | 2.48M/7.52M [00:01<00:02, 2.11MB/s]
 40%|████      | 3.02M/7.52M [00:01<00:01, 2.63MB/s]
 46%|████▌     | 3.47M/7.52M [00:01<00:01, 2.98MB/s]
 86%|████████▌ | 6.44M/7.52M [00:01<00:00, 8.60MB/s]
100%|██████████| 7.52M/7.52M [00:02<00:00, 3.72MB/s]
[  117.2s] poll 58392107,sub_v22_5_LIVE_crest_orth4.csv,2026-11-15 16:13:23.997000,v22.5 K2_Ray1_Crest_2p05e4_plus_Orth4_0p45e4 | ρ(R2)=0.450e-4 ρ(sri)=2.054e-4 OOF ridge 5/5 t=13.79 min=+9.15,SubmissionStatus.PENDING,,
[  136.4s] poll 58392107,sub_v22_5_LIVE_crest_orth4.csv,2026-11-15 16:13:23.997000,v22.5 K2_Ray1_Crest_2p05e4_plus_Orth4_0p45e4 | ρ(R2)=0.450e-4 ρ(sri)=2.054e-4 OOF ridge 5/5 t=13.79 min=+9.15,SubmissionStatus.COMPLETE,0.95426,
```

