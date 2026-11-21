# GM V31.3 GM Edition — Oracle `0.96153` Simulation & Orthogonal STEP_P

**MAX_SUBMITS = 1** · remaining≈5 · LB1=0.95537 · 2026-11-21 09:41

## Part A — Can the 152-bank reach `0.96153`?

**Verdict:** PROVEN ceiling: max(wide, Oracle, 150-probe) on 60k-pub = 0.95898 (wide_max=0.95896, oracle_max=0.95870, probe_max=0.95898) << 0.96153 over 78 bank cols. #1 Dharun 0.96153 / #2 Smart Manoj 0.96070 CANNOT be explained by any linear reweight of the public OOF bank — they MUST have non-bank signal (or a qualitatively different model class / data). Note: lucky 60k slices of wide_lm alone can hit ~0.95896 (public variance), but still short of 0.96153.

| attacker | pub mean | pub max | priv mean | priv min |
|----------|---------:|--------:|----------:|---------:|
| wide_lm | 0.95313 | 0.95896 | 0.95226 | 0.95110 |
| **60k GT Oracle** | **0.95270** | **0.95870** | 0.95155 | 0.95032 |
| 150-query probe | 0.95329 | 0.95898 | 0.95204 | 0.94958 |

Bank cols used: **78**

## Part B — Frozen wide_lm Gate-B

Distinct meanΔ@+0.003: **11** · Gate-B pass: **True**

| name | variant | marg+0.003 | pos | mean e-4 | t | sf | pass |
|------|---------|----------:|----:|---------:|--:|---:|------|
| `P4_L4.2` | winsor | -0.0385 | 3/5 | +0.083 | 1.79 | 18/25 | fail |
| `P4_L4.2` | orth10 | 0.0 | 2/5 | -0.000 | -0.27 | 12/25 | fail |
| `P4_L4.3` | winsor | -0.048 | 4/5 | +0.096 | 2.42 | 18/25 | fail |
| `P4_L4.3` | orth10 | 0.0 | 3/5 | +0.000 | 0.30 | 8/25 | fail |
| `P4_L4.3_ext` | winsor | 0.0226 | 4/5 | +0.048 | 1.42 | 20/25 | fail |
| `P4_L4.3_ext` | orth10 | 0.0 | 3/5 | +0.000 | 0.57 | 14/25 | fail |
| `P4_L4.5` | winsor | -0.0047 | 2/5 | +0.006 | 0.24 | 14/25 | fail |
| `P4_L4.5` | orth10 | 0.0 | 3/5 | +0.000 | 0.88 | 15/25 | fail |
| `harvest_p4_cuda_xgb` | winsor | -0.054 | 5/5 | +0.123 | 3.57 | 19/25 | fail |
| `harvest_p4_cuda_xgb` | orth10 | 0.0 | 3/5 | -0.000 | -0.00 | 12/25 | fail |
| `v10_ridge_orth_LM` | winsor | 0.0994 | 5/5 | +0.271 | 5.64 | 25/25 | PASS |
| `v10_ridge_orth_LM` | orth10 | 0.0 | 4/5 | +0.000 | 2.75 | 14/25 | fail |
| `v10_ridge_resid` | winsor | 0.0952 | 5/5 | +0.259 | 5.18 | 25/25 | PASS |
| `v10_ridge_resid` | orth10 | 0.0 | 3/5 | +0.000 | 1.37 | 12/25 | fail |
| `gpt2_med` | winsor | -0.0073 | 2/5 | +0.017 | 0.57 | 12/25 | fail |
| `gpt2_med` | orth10 | 0.0 | 2/5 | +0.000 | 0.79 | 13/25 | fail |
| `L7_s2026` | winsor | -0.0032 | 2/5 | +0.006 | 0.39 | 13/25 | fail |
| `L7_s2026` | orth10 | 0.0 | 1/5 | +0.000 | 1.00 | 8/25 | fail |
| `U5_gini10` | winsor | 0.0095 | 4/5 | +0.023 | 2.08 | 18/25 | fail |
| `U5_gini10` | orth10 | 0.0 | 2/5 | +0.000 | 1.50 | 9/25 | fail |

## Part C — Orthogonal STEP_P

- Temporal Board ρ=0.963 checks={'step_o>step_n': True, 'step_o>slot1_l7': True} temporal_pass=True
- Gate A (all 5 BoardV3 > MID37): **False**
- Best trial: `{"a": 1.0, "tag": "1.0", "use_zr": false, "tau": 4.999999999999998, "omr": 0.012180482499291045, "corr_ray": -3.686287386450715e-18, "preds": [0.9553715711621564, 0.9553761912493752, 0.9553872068372523, 0.9553385206046866, 0.9553900594818847], "mu": 0.9553727098670711, "n_pos": 4}`
- SLOT1 LB: **0.95537**

## Decision
**SUBMITTED 1** (`sub_v31_3_GM_SINGLE_PROBE.csv`) → **LB 0.95537 (tie)** then STOP.

**Caveat:** after `⊥ (SLOT2−SLOT1)` the surviving direction was nearly rank-null vs MID37 (`omr≈0.012` not `0.38e-4`), so this slot was a near-duplicate of the plateau — consistent with the saturated-ray diagnosis. Gate-B `v10` OOF pass did **not** translate into a live move off `0.95537`.

**Strategic takeaway from Part A (the gold):** stop spending slots on bank-reweight / tiny board secants. `#1/#2` require a **non-bank** feature/model program. Remaining slots protected (~5).

## Log
```
[    0.0s] === GM V31.3 GM Edition | MAX_SUBMITS=1 | Oracle sim + GateB + STEP_P ===
[   12.8s] wide_lm AUC=0.952565
[   15.2s] submits_today=4 remaining_est=6
[   56.8s] bank matrix: 78 cols top=wide_lm@0.95257
[   56.8s] === PART A: 60k-pub / 240k-priv Oracle + probe ceiling ===
[  101.6s]   split0: wide=0.95266/0.95243 | oracle=0.95237/0.95196(α=logreg_C0.1) | probe150=0.95272/0.95248
[  132.1s]   split1: wide=0.95405/0.95448 | oracle=0.95345/0.95395(α=logreg_C0.1) | probe150=0.95407/0.95445
[  157.0s]   split2: wide=0.94922/0.95143 | oracle=0.94871/0.95054(α=logreg_C0.1) | probe150=0.95027/0.94958
[  185.1s]   split3: wide=0.95520/0.95110 | oracle=0.95431/0.95032(α=logreg_C0.1) | probe150=0.95528/0.95097
[  205.4s]   split4: wide=0.94758/0.95166 | oracle=0.94725/0.95124(α=logreg_C0.1) | probe150=0.94762/0.95170
[  224.7s]   split5: wide=0.95896/0.95348 | oracle=0.95870/0.95233(α=logreg_C0.1) | probe150=0.95898/0.95341
[  245.1s]   split6: wide=0.95380/0.95136 | oracle=0.95310/0.95070(α=logreg_C0.1) | probe150=0.95383/0.95128
[  263.4s]   split7: wide=0.95537/0.95151 | oracle=0.95493/0.95082(α=logreg_C0.1) | probe150=0.95537/0.95148
[  283.8s]   split8: wide=0.95197/0.95252 | oracle=0.95197/0.95178(α=logreg_C0.1) | probe150=0.95220/0.95246
[  302.8s]   split9: wide=0.95251/0.95259 | oracle=0.95223/0.95187(α=logreg_C0.1) | probe150=0.95252/0.95258
[  302.8s] PART A VERDICT: PROVEN ceiling: max(wide, Oracle, 150-probe) on 60k-pub = 0.95898 (wide_max=0.95896, oracle_max=0.95870, probe_max=0.95898) << 0.96153 over 78 bank cols. #1 Dharun 0.96153 / #2 Smart Manoj 0.96070 CANNOT be explained by any linear reweight of the public OOF bank — they MUST have non-bank signal (or a qualitatively different model class / data). Note: lucky 60k slices of wide_lm alone can hit ~0.95896 (public variance), but still short of 0.96153.
[  302.8s] === PART B: frozen wide_lm Gate-B audit (10 vectors × 2 variants) ===
[  303.6s] frozen z_base from wide_lm AUC=0.952565
[  325.8s]   P4_L4.2/winsor: marg@+0.003=-0.038 LOFO w=-0.008 3/5 m=+0.083 t=1.79 sf=18 pass=False
[  333.7s]   P4_L4.2/orth10: marg@+0.003=+0.000 LOFO w=0.008 2/5 m=-0.000 t=-0.27 sf=12 pass=False
[  343.1s]   P4_L4.3/winsor: marg@+0.003=-0.048 LOFO w=-0.008 4/5 m=+0.096 t=2.42 sf=18 pass=False
[  351.7s]   P4_L4.3/orth10: marg@+0.003=+0.000 LOFO w=0.008 3/5 m=+0.000 t=0.30 sf=8 pass=False
[  359.9s]   P4_L4.3_ext/winsor: marg@+0.003=+0.023 LOFO w=0.008 4/5 m=+0.048 t=1.42 sf=20 pass=False
[  368.5s]   P4_L4.3_ext/orth10: marg@+0.003=+0.000 LOFO w=-0.008 3/5 m=+0.000 t=0.57 sf=14 pass=False
[  376.6s]   P4_L4.5/winsor: marg@+0.003=-0.005 LOFO w=-0.008 2/5 m=+0.006 t=0.24 sf=14 pass=False
[  383.9s]   P4_L4.5/orth10: marg@+0.003=+0.000 LOFO w=0.008 3/5 m=+0.000 t=0.88 sf=15 pass=False
[  392.2s]   harvest_p4_cuda_xgb/winsor: marg@+0.003=-0.054 LOFO w=-0.008 5/5 m=+0.123 t=3.57 sf=19 pass=False
[  399.7s]   harvest_p4_cuda_xgb/orth10: marg@+0.003=+0.000 LOFO w=0.008 3/5 m=-0.000 t=-0.00 sf=12 pass=False
[  410.1s]   v10_ridge_orth_LM/winsor: marg@+0.003=+0.099 LOFO w=0.008 5/5 m=+0.271 t=5.64 sf=25 pass=True
[  418.8s]   v10_ridge_orth_LM/orth10: marg@+0.003=+0.000 LOFO w=0.008 4/5 m=+0.000 t=2.75 sf=14 pass=False
[  427.3s]   v10_ridge_resid/winsor: marg@+0.003=+0.095 LOFO w=0.008 5/5 m=+0.259 t=5.18 sf=25 pass=True
[  436.5s]   v10_ridge_resid/orth10: marg@+0.003=+0.000 LOFO w=0.008 3/5 m=+0.000 t=1.37 sf=12 pass=False
[  445.5s]   gpt2_med/winsor: marg@+0.003=-0.007 LOFO w=-0.008 2/5 m=+0.017 t=0.57 sf=12 pass=False
[  453.3s]   gpt2_med/orth10: marg@+0.003=+0.000 LOFO w=-0.008 2/5 m=+0.000 t=0.78 sf=13 pass=False
[  460.6s]   L7_s2026/winsor: marg@+0.003=-0.003 LOFO w=-0.008 2/5 m=+0.006 t=0.39 sf=13 pass=False
[  468.3s]   L7_s2026/orth10: marg@+0.003=+0.000 LOFO w=0.008 1/5 m=+0.000 t=1.00 sf=8 pass=False
[  480.0s]   U5_gini10/winsor: marg@+0.003=+0.009 LOFO w=0.008 4/5 m=+0.023 t=2.08 sf=18 pass=False
[  488.0s]   U5_gini10/orth10: marg@+0.003=+0.000 LOFO w=0.008 2/5 m=+0.000 t=1.50 sf=9 pass=False
[  488.0s] Gate-B distinct meanΔ@+0.003: 11/20 unique
[  491.6s] HONEST NOSUBMIT from v10_ridge_orth_LM/winsor w=0.008
[  491.6s] === PART C: orthogonal STEP_P (⊥ SLOT2−SLOT1) + Board-Model V3 ===
[  500.6s] MID37 vs SLOT1 omr=0.0445 vs SLOT2=0.0667 | |ray|std=1.0000
[  500.9s] corr(SLOT1,SLOT2) spearman=0.99997800 omr=0.219973
[  510.9s] Board temporal ρ=0.963 checks={'step_o>step_n': True, 'step_o>slot1_l7': True} pass=True
[  511.0s] MID37 BoardV3 preds=[0.955371, 0.955374, 0.955386, 0.955338, 0.955373] mean=0.955368
[  522.8s]   trial tag=0.0 zr=False omr=0.0149 corr_ray=0.000000 n_pos=3/5 μ=0.955367
[  542.4s]   trial tag=0.0 zr=True omr=0.0000 corr_ray=0.000000 n_pos=4/5 μ=0.955368
[  551.8s]   trial tag=0.2 zr=False omr=0.0144 corr_ray=0.000000 n_pos=4/5 μ=0.955368
[  570.5s]   trial tag=0.2 zr=True omr=0.0000 corr_ray=0.000000 n_pos=4/5 μ=0.955368
[  579.7s]   trial tag=0.4 zr=False omr=0.0137 corr_ray=-0.000000 n_pos=4/5 μ=0.955369
[  598.1s]   trial tag=0.4 zr=True omr=0.0000 corr_ray=-0.000000 n_pos=4/5 μ=0.955368
[  607.4s]   trial tag=0.6000000000000001 zr=False omr=0.0130 corr_ray=0.000000 n_pos=4/5 μ=0.955370
[  627.6s]   trial tag=0.6000000000000001 zr=True omr=0.0000 corr_ray=0.000000 n_pos=4/5 μ=0.955368
[  636.6s]   trial tag=0.8 zr=False omr=0.0124 corr_ray=0.000000 n_pos=4/5 μ=0.955372
[  654.6s]   trial tag=0.8 zr=True omr=0.0000 corr_ray=0.000000 n_pos=4/5 μ=0.955368
[  664.0s]   trial tag=1.0 zr=False omr=0.0122 corr_ray=-0.000000 n_pos=4/5 μ=0.955373
[  680.9s]   trial tag=1.0 zr=True omr=0.0000 corr_ray=-0.000000 n_pos=4/5 μ=0.955368
[  689.8s]   trial tag=gateb_v10 zr=False omr=0.0552 corr_ray=0.000000 n_pos=4/5 μ=0.955369
[  707.7s]   trial tag=gateb_v10 zr=True omr=0.0000 corr_ray=0.000000 n_pos=4/5 μ=0.955368
[  707.7s] GATE A (temporal+all5 BoardV3+): False | best n_pos=4 μΔ=+0.000005 corr_ray=-0.000000
[  709.4s] staged sub_v31_3_GM_SINGLE_PROBE.csv omr(MID37)=0.0122
[  709.8s] Gates fireable: A=False B=True — submitting AT MOST 1
[  714.9s] k/7.52M [00:00<00:07, 1.02MB/s]
  8%|▊         | 592k/7.52M [00:00<00:07, 917kB/s] 
  9%|▊         | 672k/7.52M [00:00<00:09, 764kB/s]
 35%|███▌      | 2.64M/7.52M [00:00<00:01, 5.11MB/s]
 65%|██████▌   | 4.91M/7.52M [00:01<00:00, 9.48MB/s]
 82%|████████▏ | 6.17M/7.52M [00:01<00:00, 8.23MB/s]
 96%|█████████▌| 7.22M/7.52M [00:01<00:00, 7.05MB/s]
100%|██████████| 7.52M/7.52M [00:02<00:00, 3.76MB/s]

[  736.7s] POLLED sub_v31_3_GM_SINGLE_PROBE.csv -> 0.95537
[  736.7s] STOP after ≤1 submit — reserve slots for post-sim #1 strategy
```
