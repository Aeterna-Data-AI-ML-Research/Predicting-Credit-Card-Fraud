# GM V25 New Score Channels

ATH **0.95440** · elapsed 402s

## Channel map

| ID | Channel | Idea |
|----|---------|------|
| C1 | Alex-fresh 40-model | New public OOF bank (Nov17); family rank-means + best singles |
| C2 | Mathsonchain 17 | Alternate external singles bank |
| C3 | Peer disagreement / triple | Pull/push vs gpw/bhuvan/step11; dual without v15; digitGPT2 neural swap |
| C4 | ATH × channel | Small-w / micro-orth injects of gated channels |

## C1 Alex top singles (raw5 gate)

| solo | raw5 | gb8 | ath-orth | ρATH e-4 | model |
|-----:|------|-----|----------|--------:|-------|
| 0.95054 | 5/5 +2.37 | 3/5 +2.42 | 0/5 | 205.0 | alex:LightGBM 4 leaves boosted from the joint GLM, full refit|3d3b857b |
| 0.95011 | 4/5 +1.81 | 2/5 -0.62 | 0/5 | 260.9 | alex:Joint GLM, full refit|29b8383a |
| 0.95039 | 4/5 +1.77 | 4/5 +4.24 | 0/5 | 256.0 | alex:LightGBM 4 leaves, Part 3 recipe|ff715163 |
| 0.95066 | 4/5 +1.65 | 4/5 +2.96 | 0/5 | 237.8 | alex:LightGBM 3 leaves, Part 5 recipe, 5 encoder seeds|4188470e |
| 0.94916 | 4/5 +1.47 | 2/5 -1.83 | 0/5 | 376.5 | alex:GLM on the encodings|e0f351ee |
| 0.95052 | 4/5 +1.46 | 4/5 +3.45 | 0/5 | 238.5 | alex:LightGBM 3 leaves, Part 5 recipe|21b4d131 |
| 0.95054 | 4/5 +1.40 | 4/5 +2.83 | 0/5 | 222.0 | alex:XGBoost depth 2, Part 3 recipe, 5 encoder seeds, full refit|82e9a796 |
| 0.94868 | 4/5 +1.36 | 4/5 +3.37 | 0/5 | 344.0 | alex:LightGBM 4 leaves plus|d5bd0671 |
| 0.95036 | 4/5 +1.11 | 4/5 +3.03 | 0/5 | 240.6 | alex:XGBoost depth 2, Part 3 recipe|55b2e2d2 |
| 0.94959 | 4/5 +0.86 | 4/5 +2.25 | 0/5 | 258.2 | alex:LightGBM 4 leaves digits + GPT-2 tokens|f9c597c5 |
| 0.94763 | 3/5 +1.80 | 4/5 +5.57 | 0/5 | 460.9 | alex:LightGBM 4 leaves raw|750a3a13 |
| 0.95042 | 3/5 +1.74 | 3/5 +1.09 | 0/5 | 213.3 | alex:XGBoost depth 2 boosted from the joint GLM, full refit|00107284 |

## Family / bank composites

| name | solo | raw5 | gb8 | ρATH |
|------|-----:|------|-----|-----:|
| `alex_joint_glm` | 0.95044 | 4/5 +2.01 | 3/5 +1.07 | 218.0 |
| `math_rankmean17` | 0.95017 | 4/5 +1.94 | 4/5 +3.27 | 294.7 |
| `alex_top5_solo` | 0.95085 | 4/5 +1.79 | 4/5 +2.75 | 200.2 |
| `alex_part5_lgb` | 0.95064 | 4/5 +1.56 | 4/5 +3.25 | 237.3 |
| `alex_digit_gpt2` | 0.94959 | 4/5 +0.86 | 4/5 +2.25 | 258.2 |
| `alex_digit` | 0.94980 | 3/5 +0.64 | 3/5 +2.23 | 259.0 |
| `alex_cat` | 0.94766 | 2/5 -0.84 | 2/5 -0.72 | 532.6 |
| `alex_depth1` | 0.94840 | 2/5 -1.10 | 2/5 -1.39 | 399.0 |
| `alex_nn` | 0.94824 | 2/5 -2.02 | 2/5 -2.39 | 420.4 |

## Submit priority (staged)

1. `sub_v25_ATH_x_alex_joint_glm_05.csv` — ρATH=0.539e-4
2. `sub_v25_ATH_x_alex_joint_glm_08.csv` — ρATH=1.380e-4
3. `sub_v25_ATH_x_math_rankmean17_05.csv` — ρATH=0.715e-4
4. `sub_v25_ATH_x_math_rankmean17_08.csv` — ρATH=1.833e-4
5. `sub_v25_ATH_x_alex_top5_solo_05.csv` — ρATH=0.496e-4
6. `sub_v25_ATH_x_alex_top5_solo_08.csv` — ρATH=1.270e-4
7. `sub_v25_ATH_x_alex_part5_lgb_05.csv` — ρATH=0.584e-4
8. `sub_v25_ATH_x_alex_part5_lgb_08.csv` — ρATH=1.497e-4
9. `sub_v25_ATH_x_alex_digit_gpt2_05.csv` — ρATH=0.641e-4
10. `sub_v25_ATH_x_alex_digit_gpt2_08.csv` — ρATH=1.641e-4
11. `sub_v25_DIS_gpw32_015.csv` — ρATH=30.380e-4
12. `sub_v25_DIS_gpw32_030.csv` — ρATH=121.577e-4

## Live

| tag | status |
|-----|--------|
| `ATH_x_alex_joint_glm_05` | quota_or_error |

## Verdict rules

- **VALID channel candidate:** family/single raw5 ≥4/5 and meanΔ>0, or structural dual/disagree with ρATH∈[0.15, 3.0]e-4 near ATH manifold.
- **INVALID:** ath-orth 0/5 with raw5 fail; solo≪0.94; ρATH≫10e-4 unless intentional peer blend.
- Exhausted (not retested as primary): Ray-1 crest, ±twilight, bank micro-GS on R2, CSV roundtrip.

## Log
```
[    0.0s] === V25 NEW SCORE CHANNELS ===
[    6.6s] y fraud=0.01670 ATH loaded ρ self=0
[    9.8s] ATH surrogate AUC=0.951428 gb_bag5=0.951506
[    9.8s] C1 loading alex_fresh 40-model bank…
[   10.1s] C1 pairs=40
[  169.1s] C1 top-12 by raw5/gb8:
[  169.1s]   0.95054 raw5 5/5 t=1.84 m=+2.37 | gb8 3/5 m=+2.42 | ath-orth 0/5 | ρATH=205.0 | alex:LightGBM 4 leaves boosted from the joint GLM, full refit|3d3b857b
[  169.1s]   0.95011 raw5 4/5 t=1.89 m=+1.81 | gb8 2/5 m=-0.62 | ath-orth 0/5 | ρATH=260.9 | alex:Joint GLM, full refit|29b8383a
[  169.1s]   0.95039 raw5 4/5 t=1.01 m=+1.77 | gb8 4/5 m=+4.24 | ath-orth 0/5 | ρATH=256.0 | alex:LightGBM 4 leaves, Part 3 recipe|ff715163
[  169.1s]   0.95066 raw5 4/5 t=1.56 m=+1.65 | gb8 4/5 m=+2.96 | ath-orth 0/5 | ρATH=237.8 | alex:LightGBM 3 leaves, Part 5 recipe, 5 encoder seeds|4188470e
[  169.1s]   0.94916 raw5 4/5 t=1.02 m=+1.47 | gb8 2/5 m=-1.83 | ath-orth 0/5 | ρATH=376.5 | alex:GLM on the encodings|e0f351ee
[  169.1s]   0.95052 raw5 4/5 t=1.17 m=+1.46 | gb8 4/5 m=+3.45 | ath-orth 0/5 | ρATH=238.5 | alex:LightGBM 3 leaves, Part 5 recipe|21b4d131
[  169.1s]   0.95054 raw5 4/5 t=0.99 m=+1.40 | gb8 4/5 m=+2.83 | ath-orth 0/5 | ρATH=222.0 | alex:XGBoost depth 2, Part 3 recipe, 5 encoder seeds, full refit|82e9a796
[  169.1s]   0.94868 raw5 4/5 t=0.80 m=+1.36 | gb8 4/5 m=+3.37 | ath-orth 0/5 | ρATH=344.0 | alex:LightGBM 4 leaves plus|d5bd0671
[  169.1s]   0.95036 raw5 4/5 t=0.69 m=+1.11 | gb8 4/5 m=+3.03 | ath-orth 0/5 | ρATH=240.6 | alex:XGBoost depth 2, Part 3 recipe|55b2e2d2
[  169.1s]   0.94959 raw5 4/5 t=0.85 m=+0.86 | gb8 4/5 m=+2.25 | ath-orth 0/5 | ρATH=258.2 | alex:LightGBM 4 leaves digits + GPT-2 tokens|f9c597c5
[  169.1s]   0.94763 raw5 3/5 t=1.00 m=+1.80 | gb8 4/5 m=+5.57 | ath-orth 0/5 | ρATH=460.9 | alex:LightGBM 4 leaves raw|750a3a13
[  169.1s]   0.95042 raw5 3/5 t=1.58 m=+1.74 | gb8 3/5 m=+1.09 | ath-orth 0/5 | ρATH=213.3 | alex:XGBoost depth 2 boosted from the joint GLM, full refit|00107284
[  173.6s] C1 FAM alex_part5_lgb: n=2 solo=0.95064 raw5 4/5 m=+1.56 gb8 4/5 m=+3.25 ρATH=237.3
[  179.2s] C1 FAM alex_digit: n=7 solo=0.94980 raw5 3/5 m=+0.64 gb8 3/5 m=+2.23 ρATH=259.0
[  183.5s] C1 FAM alex_nn: n=3 solo=0.94824 raw5 2/5 m=-2.02 gb8 2/5 m=-2.39 ρATH=420.4
[  188.0s] C1 FAM alex_joint_glm: n=3 solo=0.95044 raw5 4/5 m=+2.01 gb8 3/5 m=+1.07 ρATH=218.0
[  192.4s] C1 FAM alex_depth1: n=2 solo=0.94840 raw5 2/5 m=-1.10 gb8 2/5 m=-1.39 ρATH=399.0
[  197.3s] C1 FAM alex_cat: n=3 solo=0.94766 raw5 2/5 m=-0.84 gb8 2/5 m=-0.72 ρATH=532.6
[  202.1s] C1 FAM alex_top5_solo: n=5 solo=0.95085 raw5 4/5 m=+1.79 gb8 4/5 m=+2.75 ρATH=200.2
[  206.3s] C1 FAM alex_digit_gpt2: n=1 solo=0.94959 raw5 4/5 m=+0.86 gb8 4/5 m=+2.25 ρATH=258.2
[  206.3s] C2 mathsonchain 17-single…
[  210.3s] C2 pairs=17
[  268.2s]   0.95069 raw5 5/5 m=+3.32 | gb8 3/5 | math:lgb_glmboost
[  268.2s]   0.95103 raw5 5/5 m=+2.66 | gb8 3/5 | math:boost_j
[  268.2s]   0.95011 raw5 5/5 m=+2.58 | gb8 2/5 | math:glm_te_nc
[  268.2s]   0.95051 raw5 4/5 m=+3.09 | gb8 4/5 | math:xgb_te
[  268.2s]   0.95077 raw5 4/5 m=+2.98 | gb8 3/5 | math:glm_j
[  268.2s]   0.95028 raw5 4/5 m=+2.63 | gb8 2/5 | math:glm_te
[  268.2s]   0.95038 raw5 4/5 m=+2.62 | gb8 4/5 | math:lgb_te
[  268.2s]   0.95060 raw5 4/5 m=+2.16 | gb8 4/5 | math:lgb_c
[  276.0s] C2 FAM math_rankmean17 solo=0.95017 raw5 4/5 m=+1.94 gb8 4/5 ρATH=294.7
[  276.0s] C3 peer disagreement + triple…
[  277.1s]   peer sri440 ρATH=0.000e-4
[  277.2s]   peer gpw32 ρATH=5.599e-4
[  277.2s]   peer bhuvan ρATH=7.386e-4
[  277.9s]   peer dual_r2_gpw ρATH=1.969e-4
[  278.3s]   peer r2 ρATH=6.586e-4
[  278.4s]   peer step11 ρATH=6.593e-4
[  278.5s]   peer bhuvan_pin ρATH=7.386e-4
[  278.6s]   peer n15 ρATH=251.765e-4
[  278.7s]   peer n_old ρATH=284.349e-4
[  279.1s]   DIS_gpw32_015 ρATH=30.380e-4
[  279.4s]   DIS_gpw32_030 ρATH=121.577e-4
[  279.7s]   DIS_gpw32_050 ρATH=336.501e-4
[  280.2s]   DIS_bhuvan_015 ρATH=29.730e-4
[  280.6s]   DIS_bhuvan_030 ρATH=118.911e-4
[  280.9s]   DIS_bhuvan_050 ρATH=329.626e-4
[  281.3s]   DIS_dual_r2_gpw_015 ρATH=31.873e-4
[  281.5s]   DIS_dual_r2_gpw_030 ρATH=126.856e-4
[  281.8s]   DIS_dual_r2_gpw_050 ρATH=348.692e-4
[  282.2s]   DIS_step11_015 ρATH=30.961e-4
[  282.5s]   DIS_step11_030 ρATH=124.040e-4
[  282.8s]   DIS_step11_050 ρATH=342.824e-4
[  283.2s]   DIS_n15_015 ρATH=30.474e-4
[  283.4s]   DIS_n15_030 ρATH=122.868e-4
[  283.7s]   DIS_n15_050 ρATH=342.887e-4
[  284.1s]   ANTI_r2_030 ρATH=128.184e-4
[  284.5s]   ANTI_gpw32_030 ρATH=120.345e-4
[  284.8s]   TRIPLE_near n=3 ρATH=5.891e-4
[  285.1s]   DUAL_step11_gpw32 ρATH=5.918e-4
[  285.5s]   DUAL_GB_x_digitGPT2_w0.22 ρATH=70.887e-4
[  285.9s]   DUAL_GB_x_digitGPT2_w0.30 ρATH=131.901e-4
[  286.5s]   DUAL_GB_n15mixDigit_w0.30 ρATH=89.143e-4
[  286.8s]   DUAL_GB_n15mixDigit_w0.35 ρATH=127.485e-4
[  286.8s] C4 building ATH×channel injects…
[  298.7s]   orth ATH_orth_alex_3d3b857b_0p30 ρATH=0.000e-4 (target 0.30)
[  305.9s]   orth ATH_orth_alex_29b8383a_0p30 ρATH=0.000e-4 (target 0.30)
[  313.0s]   orth ATH_orth_alex_ff715163_0p30 ρATH=0.000e-4 (target 0.30)
[  320.3s]   orth ATH_orth_alex_4188470e_0p30 ρATH=0.000e-4 (target 0.30)
[  327.6s]   orth ATH_orth_alex_e0f351ee_0p30 ρATH=0.000e-4 (target 0.30)
[  398.6s] STAGED 65 CSVs
[  398.6s]   ATH_orth_alex_3d3b857b_0p30              ρATH=0.000e-4
[  398.6s]   ATH_orth_alex_29b8383a_0p30              ρATH=0.000e-4
[  398.6s]   ATH_orth_alex_ff715163_0p30              ρATH=0.000e-4
[  398.6s]   ATH_orth_alex_4188470e_0p30              ρATH=0.000e-4
[  398.6s]   ATH_orth_alex_e0f351ee_0p30              ρATH=0.000e-4
[  398.6s]   ATH_x_alex_top5_solo_05                  ρATH=0.496e-4
[  398.6s]   ATH_x_alex_3d3b857b_05                   ρATH=0.510e-4
[  398.6s]   ATH_x_alex_joint_glm_05                  ρATH=0.539e-4
[  398.6s]   ATH_x_alex_part5_lgb_05                  ρATH=0.584e-4
[  398.6s]   ATH_x_alex_4188470e_05                   ρATH=0.585e-4
[  398.6s]   ATH_x_alex_ff715163_05                   ρATH=0.628e-4
[  398.6s]   ATH_x_alex_29b8383a_05                   ρATH=0.639e-4
[  398.6s]   ATH_x_alex_digit_05                      ρATH=0.640e-4
[  398.6s]   ATH_x_alex_digit_gpt2_05                 ρATH=0.641e-4
[  398.6s]   ATH_x_math_rankmean17_05                 ρATH=0.715e-4
[  398.6s] SUBMIT try v25.ATH_x_alex_joint_glm_05 | ρ(ATH)=0.539e-4 | new-channel research
[  402.5s] 08, 883kB/s]  |  12%|█▏        | 928k/7.52M [00:00<00:07, 899kB/s] |  22%|██▏       | 1.64M/7.52M [00:01<00:02, 2.07MB/s] |  44%|████▍     | 3.34M/7.52M [00:01<00:00, 5.17MB/s] |  55%|█████▌    | 4.14M/7.52M [00:01<00:00, 5.79MB/s] |  70%|███████   | 5.28M/7.52M [00:01<00:00, 7.24MB/s] |  92%|█████████▏| 6.92M/7.52M [00:01<00:00, 9.74MB/s] | 100%|██████████| 7.52M/7.52M [00:02<00:00, 3.91MB/s] | 400 Client Error: Bad Request for url: https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateSubmission | 
[  402.5s] STOP quota — remaining staged
```
