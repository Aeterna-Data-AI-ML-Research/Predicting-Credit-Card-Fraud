# V43.2 — Local Parallel 4-Engine Discovery

**NO_SUBMIT=True · 0 pods · $0 local CPU/MPS**

Baseline `z_modern_pub+Zr` = **0.952652** · orig20k = `orig20k.csv` (339 fraud)

## Gates vs z_modern_pub

| engine | Δe5 | pos/5 | min | sf/25 | t25 | w | cosS02 | cosS06 | disp_raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1_orig339 | +0.00 | 3/5 | -0.00 | 13/25 | 0.40 | -0.0800 | +0.108 | +1.000 | 0.750 |
| E2_epistatic | +0.00 | 3/5 | -0.00 | 11/25 | 1.52 | +0.0710 | +0.108 | +1.000 | 0.750 |
| E3_wilcoxon | +0.00 | 3/5 | -0.00 | 15/25 | 0.85 | +0.0010 | -0.108 | -1.000 | 1.000 |
| E4_pillar_moe | +0.00 | 4/5 | -0.00 | 12/25 | 1.42 | -0.0800 | -0.108 | -1.000 | 1.000 |
| E_combo | +0.00 | 3/5 | -0.00 | 7/25 | 0.33 | +0.0010 | -0.108 | -1.000 | 0.719 |

## Gates vs z_pub_only

| engine | Δe5 | pos/5 | sf/25 | t25 |
|---|---:|---:|---:|---:|
| E1_orig339 | +0.35 | 4/5 | 16/25 | 2.26 |
| E2_epistatic | +0.34 | 4/5 | 16/25 | 2.26 |
| E3_wilcoxon | +0.35 | 4/5 | 16/25 | 2.27 |
| E4_pillar_moe | +0.35 | 4/5 | 16/25 | 2.27 |
| E_combo | +0.35 | 4/5 | 16/25 | 2.26 |

## Staged NOSUBMIT (1)
- `NOSUBMIT_E4_pillar_moe_w-0.0800_tip0p10e4.csv` sha=`9623b8b6983a7f8d…` Δe5=+0.00 pos=4/5

## STOP
NO_SUBMIT=True. Zero submissions. Zero GPU pods touched.
