# GM V22.2 HARDENED — Unbundled Sriram Micro-Leap Report

**elapsed** 219s · local Mac · personal ATH was `0.95416` · target `0.95422+`

## MPS / fold0
- `checkpoint-50000` preserved: `True`
- Replica: GReaT **Target-First** (`conditional_col=is_fraud`, target in column shuffle)
- fold0 classify: **TERM'd** (was ~37h ETA at <0.5%) · `STOP_CLASSIFY` written

## Sriram native-space
- Formula: `rank_frac( L(goodpjw) + 0.15*(L(new)-L(old)) )` with `L=logit((rank-0.5)/N)`
- Self-repro `1-ρ`: `0.000e+00` (gate `<1e-10`)
- C2pure vs sriram: `4.371e-4` (overshoot diagnosis)

## Live submit
- OK=`True` · **P2_MicroGeodesic_Sriram095422_x_C1095407_plus_5FLM_0p55e4**
- Public LB: **0.95425**
- 1−ρ vs sriram: `0.550e-4` (target `0.55`)
- max|corr| vs B_macro: `0.1613` · c6=`+0.0942` c7=`-0.0390`
- OOF sri `5/5` t=`15.83` min=`+19.44`

## Clean pillars used
- `U_B_gpt2med_5f` sign=`+1` sri_t=`5.95`
- `U_C_tab_resid` sign=`+1` sri_t=`5.77`
- `U_D_bank_gbn` sign=`+1` sri_t=`5.09`

## Staged
- `sub_v22_2_P1_Sriram_Unbundled_5FLM_0p40e4.csv`
- `sub_v22_2_P2_MicroGeodesic_Sriram_x_C1_5FLM_0p55e4.csv`
- `sub_v22_2_P3_DeDiluted_dMLP_CleanProp_0p50e4.csv`
- `sub_v22_2_HARDENED_LIVE_microleap.csv`

## Log
```
[    0.0s] === Step0: preserve ckpt-50000 & TERM fold0 classify ===
[    0.0s] checkpoint-50000 exists=True final_model=True
[    0.0s] replica train: target_col=is_fraud is_fraud_in_columns=True (GReaT conditional_col=is_fraud → Target-First)
[    2.1s] fold0 classify stopped=True STOP_CLASSIFY written
[    2.4s] === rebuild proprietary Slot8/9 engines ===
[  110.6s] === Step1: exact sriram native-space reproduction ===
[  111.0s] sriram self-repro 1-ρ=0.000e+00 (gate <1e-10)
[  111.1s] sriram vs goodpjw 1-ρ=2.258124e-4
[  120.8s] live distances: C2pure vs sriram=4.371e-4 | C1 vs sriram=28.241e-4
[  127.5s] z_sriram_oof AUC=0.951612 surrogate R²_test≈0.9765
[  140.5s] E2 orient flipped=False
[  140.7s] === audit unbundled pillars U_A..U_E (±) ===
[  142.4s]   U_A_s2026_lm5f+: sri 4/5 m=+1.28 min=-1.28 t=1.42 sf=16 | z9 4/5 t=1.06 | max|B|=0.0000 c6=+0.000 c7=-0.000 corrC2=-0.036 disp=1.580
[  143.9s]   U_A_s2026_lm5f-: sri 1/5 m=-2.39 min=-3.44 t=-3.07 sf=9 | z9 0/5 t=-1.91 | max|B|=0.0000 c6=-0.000 c7=+0.000 corrC2=+0.036 disp=1.580
[  145.6s]   U_B_gpt2med_5f+: sri 5/5 m=+4.90 min=+2.37 t=5.95 sf=21 | z9 5/5 t=5.11 | max|B|=0.0000 c6=+0.000 c7=+0.000 corrC2=+0.024 disp=0.877
[  147.0s]   U_B_gpt2med_5f-: sri 0/5 m=-5.49 min=-8.63 t=-6.11 sf=4 | z9 0/5 t=-5.24 | max|B|=0.0000 c6=-0.000 c7=-0.000 corrC2=-0.024 disp=0.877
[  147.0s]   PASS U_B_gpt2med_5f sign=+1
[  148.5s]   U_C_tab_resid+: sri 5/5 m=+3.66 min=+3.28 t=5.77 sf=22 | z9 5/5 t=2.21 | max|B|=0.0000 c6=+0.000 c7=+0.000 corrC2=-0.038 disp=1.016
[  149.9s]   U_C_tab_resid-: sri 0/5 m=-4.58 min=-5.79 t=-7.33 sf=1 | z9 0/5 t=-3.42 | max|B|=0.0000 c6=-0.000 c7=-0.000 corrC2=+0.038 disp=1.016
[  149.9s]   SOFT-PASS U_C_tab_resid sign=+1 (sri-strong)
[  151.5s]   U_D_bank_gbn+: sri 5/5 m=+1.69 min=+1.02 t=5.09 sf=20 | z9 5/5 t=2.08 | max|B|=0.0000 c6=-0.000 c7=+0.000 corrC2=+0.028 disp=1.156
[  153.1s]   U_D_bank_gbn-: sri 0/5 m=-1.81 min=-3.16 t=-4.65 sf=7 | z9 0/5 t=-2.24 | max|B|=0.0000 c6=+0.000 c7=-0.000 corrC2=-0.028 disp=1.156
[  153.1s]   SOFT-PASS U_D_bank_gbn sign=+1 (sri-strong)
[  154.7s]   U_E_dMLP_dediluted+: sri 5/5 m=+1.68 min=+0.06 t=2.24 sf=18 | z9 4/5 t=1.54 | max|B|=0.0024 c6=+0.000 c7=+0.000 corrC2=-0.006 disp=261.399
[  156.2s]   U_E_dMLP_dediluted-: sri 0/5 m=-2.17 min=-5.14 t=-2.89 sf=6 | z9 1/5 t=-1.98 | max|B|=0.0024 c6=-0.000 c7=-0.000 corrC2=+0.006 disp=261.399
[  156.4s] clean pillars for inject: ['U_B_gpt2med_5f+', 'U_C_tab_resid+', 'U_D_bank_gbn+']
[  156.5s] u_clean_prop max|B_macro|=0.0000 corr(sriram)=+0.000000
[  156.5s] === Step2: build P1/P2/P3 micro-steps ===
[  162.0s] P1 w=0.01656 ρs=0.400e-4 sri 5/5 t=8.95 min=+6.56
[  174.8s] P2 λ=0.110 w2=0.01821 ρs=0.550e-4 sri 5/5 t=15.83 min=+19.44
[  174.9s] P3 base 1-ρ(sriram)=0.330e-4
[  180.5s] P3 w=0.01858 ρs=0.500e-4 sri 5/5 t=5.65 min=+2.94
[  180.8s] P1_Sriram_plus_Unbundled_5FLM_MicroStep_0p40e4: ρs=0.400 ρc2=4.777 ρc1=26.964 max|B|=0.0110 c6=+0.002 c7=+0.001 sri 5/5 t=8.95 min=+6.56 | z9 5/5 t=5.85 | z9d 5/5 t=5.35
[  181.1s] P2_MicroGeodesic_Sriram095422_x_C1095407_plus_5FLM_0p55e4: ρs=0.550 ρc2=5.634 ρc1=21.347 max|B|=0.1613 c6=+0.094 c7=-0.039 sri 5/5 t=15.83 min=+19.44 | z9 3/5 t=1.75 | z9d 3/5 t=0.75
[  181.5s] P3_Sriram_DeDiluted_dMLP_plus_CleanProp_0p50e4: ρs=0.500 ρc2=4.876 ρc1=26.937 max|B|=0.0200 c6=+0.001 c7=+0.002 sri 5/5 t=5.65 min=+2.94 | z9 5/5 t=5.89 | z9d 3/5 t=0.32
[  181.6s] SELECT P2_MicroGeodesic_Sriram095422_x_C1095407_plus_5FLM_0p55e4
[  182.7s] STAGED sub_v22_2_P1_Sriram_Unbundled_5FLM_0p40e4.csv
[  183.3s] STAGED sub_v22_2_P2_MicroGeodesic_Sriram_x_C1_5FLM_0p55e4.csv
[  184.2s] STAGED sub_v22_2_P3_DeDiluted_dMLP_CleanProp_0p50e4.csv
[  185.1s] STAGED sub_v22_2_HARDENED_LIVE_microleap.csv
[  185.2s] === Step3: submit ONE probe ===
[  185.2s] SUBMIT v22.2h P2_MicroGeodesic_Sriram095422_x_C1095407_plus_5FLM_0p55e4 | ρs=0.550e-4 maxB=0.1613 c7=-0.039 OOF sri 5/5 t=15.83
[  190.2s] Submission ref: 58388361
7 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[  190.2s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  2%|▏         | 144k/7.52M [00:00<00:05, 1.34MB/s]
  9%|▊         | 656k/7.52M [00:00<00:06, 1.15MB/s]
 10%|█         | 784k/7.52M [00:00<00:06, 1.06MB/s]
 12%|█▏        | 960k/7.52M [00:00<00:06, 997kB/s] 
 18%|█▊        | 1.34M/7.52M [00:01<00:03, 1.63MB/s]
 21%|██        | 1.55M/7.52M [00:01<00:03, 1.70MB/s]
 44%|████▍     | 3.31M/7.52M [00:01<00:00, 5.63MB/s]
 77%|███████▋  | 5.80M/7.52M [00:01<00:00, 10.7MB/s]
 94%|█████████▍| 7.06M/7.52M [00:01<00:00, 6.25MB/s]
100%|██████████| 7.52M/7.52M [00:02<00:00, 3.49MB/s]
[  199.0s] poll 58388361,sub_v22_2_HARDENED_LIVE_microleap.csv,2026-11-15 14:29:16.287000,v22.2h P2_MicroGeodesic_Sriram095422_x_C1095407_plus_5FLM_0p55e4 | ρs=0.550e-4 maxB=0.1613 c7=-0.039 OOF sri 5/5 t=15.83,SubmissionStatus.PENDING,,
[  217.9s] poll 58388361,sub_v22_2_HARDENED_LIVE_microleap.csv,2026-11-15 14:29:16.287000,v22.2h P2_MicroGeodesic_Sriram095422_x_C1095407_plus_5FLM_0p55e4 | ρs=0.550e-4 maxB=0.1613 c7=-0.039 OOF sri 5/5 t=15.83,SubmissionStatus.COMPLETE,0.95425,
```

