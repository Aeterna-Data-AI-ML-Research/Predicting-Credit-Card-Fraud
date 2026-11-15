# GM V21 Local-Only 4-Leverage Upgrade

**Local CPU only · n_jobs=4 · elapsed 2939s · peak RSS≈2827MB**

- Live ATH: `0.95399` (`sub_v20_LIVE_slot9_overnight_cpu_promoted.csv`)
- Promoted (6-gate): 0
- Submit: False
- Public LB: None

## Scorecard (best w per family)

| family | w | meanΔe-5 | minΔe-5 | 1-ρ e-4 | c7 | F2 | gates |
|---|---:|---:|---:|---:|---:|---:|:---:|
| M_A_bayes_lm | 0.020 | +3.64 | -4.35 | 0.541 | +0.000 | 0.0000 | False |
| M_B_vault_booster | 0.006 | -0.39 | -2.29 | 0.040 | +0.000 | 0.0000 | False |
| M_C_ordered_ctr | 0.012 | +0.94 | -3.47 | 0.034 | -0.002 | 0.0000 | False |
| M_D_minimax_portfolio | 0.020 | +6.15 | -3.23 | 0.589 | -0.001 | 0.0000 | False |

## Log
```
[    0.0s|rss=175MB] V21 local 4-leverage START pid=90780 n_jobs=4 monitor=90791
[    0.0s|rss=177MB] === Step1: rebuild ATH anchors (Slot8 stack + Slot9 live) ===
[   17.8s|rss=1979MB] anchors ready Slot9 1-ρ vs Slot8=0.658e-4 z_S8 AUC f0=0.951516 rss=1979MB
[   17.9s|rss=1979MB] === Leverage A: per-fold test_f-matched GPT2-med + L7 Bayes ===
[   45.5s|rss=2421MB] A done oof_std=0.0050 rss=2421MB
[   45.5s|rss=2421MB] === Leverage B: vault fusion booster ===
[   55.4s|rss=2444MB] B feature matrix (699960, 32) rss=2444MB
[   92.8s|rss=2827MB] B done oof_std=0.0035 rss=2827MB
[   92.9s|rss=2827MB] === Leverage C: ordered-CTR CatBoost on raw cat keys ===
[   93.7s|rss=2827MB] C keys=20 X=(699960, 20) rss=2827MB
[  792.6s|rss=2827MB] C fold0 AUC=0.90889 rss=2827MB
[ 1547.8s|rss=2827MB] C fold1 AUC=0.91560 rss=2827MB
[ 1886.6s|rss=2827MB] C fold2 AUC=0.91409 rss=2827MB
[ 2307.7s|rss=2827MB] C fold3 AUC=0.91301 rss=2827MB
[ 2745.9s|rss=2827MB] C fold4 AUC=0.90402 rss=2827MB
[ 2746.0s|rss=2827MB] === Leverage D: minimax portfolio synthesis ===
[ 2748.4s|rss=2827MB]   projected M_A_bayes_lm raw_disp=1.164 raw_KS=0.0412
[ 2749.6s|rss=2827MB]   projected M_B_vault_booster raw_disp=0.962 raw_KS=0.0367
[ 2750.6s|rss=2827MB]   projected M_C_ordered_ctr raw_disp=1.022 raw_KS=0.0177
[ 2751.5s|rss=2827MB]   projected cache P1_raw_only raw_disp=0.979
[ 2752.3s|rss=2827MB]   projected cache E2_heavy_ridge raw_disp=0.881
[ 2888.5s|rss=2827MB] D alpha={'M_A_bayes_lm': 0.19282988167117743, 'M_B_vault_booster': 0.011572764741442356, 'M_C_ordered_ctr': 0.21542341359997172, 'P1_raw_only': 0.27429983491787996, 'E2_heavy_ridge': 0.3058741050695286} score=13.51
[ 2888.6s|rss=2827MB] === Step3: 6-gate scorecard vs ATH 0.95399 ===
[ 2891.4s|rss=2827MB] ~ M_A_bayes_lm_w0.006: disp=1.164 KS=0.0412 F2a=0.0000 ρ=0.049 c7=-0.000 z=5/5 m=+0.97 t=2.83 sf=18 gates=False
[ 2893.0s|rss=2827MB] ~ M_A_bayes_lm_w0.008: disp=1.164 KS=0.0412 F2a=0.0000 ρ=0.086 c7=-0.000 z=5/5 m=+1.24 t=2.82 sf=17 gates=False
[ 2894.5s|rss=2827MB] ~ M_A_bayes_lm_w0.010: disp=1.164 KS=0.0412 F2a=0.0000 ρ=0.135 c7=-0.000 z=5/5 m=+1.54 t=2.83 sf=18 gates=False
[ 2896.1s|rss=2827MB] ~ M_A_bayes_lm_w0.012: disp=1.164 KS=0.0412 F2a=0.0000 ρ=0.194 c7=-0.000 z=5/5 m=+1.81 t=2.81 sf=18 gates=False
[ 2897.7s|rss=2827MB] ~ M_A_bayes_lm_w0.016: disp=1.164 KS=0.0412 F2a=0.0000 ρ=0.346 c7=-0.000 z=4/5 m=+2.29 t=2.67 sf=18 gates=False
[ 2899.3s|rss=2827MB] ~ M_A_bayes_lm_w0.020: disp=1.164 KS=0.0412 F2a=0.0000 ρ=0.541 c7=+0.000 z=3/5 m=+2.79 t=2.64 sf=19 gates=False
[ 2901.3s|rss=2827MB] STAGED sub_v21_local_4leverage_M_A_bayes_lm_NOSUBMIT.csv
[ 2903.5s|rss=2827MB] ~ M_B_vault_booster_w0.006: disp=0.962 KS=0.0367 F2a=0.0000 ρ=0.040 c7=+0.000 z=5/5 m=+0.60 t=2.94 sf=16 gates=False
[ 2905.0s|rss=2827MB] ~ M_B_vault_booster_w0.008: disp=0.962 KS=0.0367 F2a=0.0000 ρ=0.071 c7=+0.000 z=5/5 m=+0.76 t=2.83 sf=15 gates=False
[ 2906.6s|rss=2827MB] ~ M_B_vault_booster_w0.010: disp=0.962 KS=0.0367 F2a=0.0000 ρ=0.111 c7=+0.001 z=5/5 m=+0.94 t=2.86 sf=15 gates=False
[ 2908.1s|rss=2827MB] ~ M_B_vault_booster_w0.012: disp=0.962 KS=0.0367 F2a=0.0000 ρ=0.160 c7=+0.001 z=5/5 m=+1.10 t=2.76 sf=15 gates=False
[ 2909.6s|rss=2827MB] ~ M_B_vault_booster_w0.016: disp=0.962 KS=0.0367 F2a=0.0000 ρ=0.284 c7=+0.001 z=5/5 m=+1.44 t=2.74 sf=16 gates=False
[ 2911.2s|rss=2827MB] ~ M_B_vault_booster_w0.020: disp=0.962 KS=0.0367 F2a=0.0000 ρ=0.444 c7=+0.001 z=5/5 m=+1.76 t=2.52 sf=16 gates=False
[ 2911.7s|rss=2827MB] STAGED sub_v21_local_4leverage_M_B_vault_booster_NOSUBMIT.csv
[ 2914.2s|rss=2827MB] ~ M_C_ordered_ctr_w0.006: disp=1.022 KS=0.0177 F2a=0.0000 ρ=0.008 c7=-0.002 z=3/5 m=+0.17 t=0.45 sf=11 gates=False
[ 2915.9s|rss=2827MB] ~ M_C_ordered_ctr_w0.008: disp=1.022 KS=0.0177 F2a=0.0000 ρ=0.015 c7=-0.002 z=3/5 m=+0.15 t=0.37 sf=12 gates=False
[ 2917.7s|rss=2827MB] ~ M_C_ordered_ctr_w0.010: disp=1.022 KS=0.0177 F2a=0.0000 ρ=0.023 c7=-0.002 z=2/5 m=+0.07 t=0.17 sf=12 gates=False
[ 2919.4s|rss=2827MB] ~ M_C_ordered_ctr_w0.012: disp=1.022 KS=0.0177 F2a=0.0000 ρ=0.034 c7=-0.002 z=2/5 m=+0.03 t=-0.02 sf=11 gates=False
[ 2921.0s|rss=2827MB] ~ M_C_ordered_ctr_w0.016: disp=1.022 KS=0.0177 F2a=0.0000 ρ=0.060 c7=-0.002 z=2/5 m=-0.20 t=-0.31 sf=11 gates=False
[ 2922.8s|rss=2827MB] ~ M_C_ordered_ctr_w0.020: disp=1.022 KS=0.0177 F2a=0.0000 ρ=0.094 c7=-0.002 z=2/5 m=-0.59 t=-0.51 sf=10 gates=False
[ 2923.4s|rss=2827MB] STAGED sub_v21_local_4leverage_M_C_ordered_ctr_NOSUBMIT.csv
[ 2926.6s|rss=2827MB] ~ M_D_minimax_portfolio_w0.006: disp=0.803 KS=0.0396 F2a=0.0000 ρ=0.053 c7=-0.001 z=4/5 m=+1.50 t=3.60 sf=19 gates=False
[ 2928.5s|rss=2827MB] ~ M_D_minimax_portfolio_w0.008: disp=0.803 KS=0.0396 F2a=0.0000 ρ=0.094 c7=-0.001 z=3/5 m=+2.00 t=3.49 sf=19 gates=False
[ 2930.7s|rss=2827MB] ~ M_D_minimax_portfolio_w0.010: disp=0.803 KS=0.0396 F2a=0.0000 ρ=0.147 c7=-0.001 z=3/5 m=+2.47 t=3.42 sf=19 gates=False
[ 2932.3s|rss=2827MB] ~ M_D_minimax_portfolio_w0.012: disp=0.803 KS=0.0396 F2a=0.0000 ρ=0.212 c7=-0.001 z=3/5 m=+2.91 t=3.33 sf=17 gates=False
[ 2934.1s|rss=2827MB] ~ M_D_minimax_portfolio_w0.016: disp=0.803 KS=0.0396 F2a=0.0000 ρ=0.377 c7=-0.001 z=3/5 m=+3.80 t=3.20 sf=17 gates=False
[ 2935.9s|rss=2827MB] ~ M_D_minimax_portfolio_w0.020: disp=0.803 KS=0.0396 F2a=0.0000 ρ=0.589 c7=-0.001 z=3/5 m=+4.64 t=3.15 sf=17 gates=False
[ 2937.6s|rss=2827MB] STAGED sub_v21_local_4leverage_M_D_minimax_portfolio_NOSUBMIT.csv
[ 2938.9s|rss=2827MB] saved /tmp/s6e11_v16/out/v21_local_4leverages.npz
[ 2938.9s|rss=2827MB] NO 6-gate clear — HOLD Slot10
```

