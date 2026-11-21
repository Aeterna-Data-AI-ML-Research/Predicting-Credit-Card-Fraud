# GM V31.1 — Step O × Proprietary 5-Fold LM Leap

**MAX_SUBMITS=2** · remaining≈8

## Design fix vs first abort
- OOF competence LOFO vs `z_honest` **without** annihilating Top-40 SVD (V31.0 protocol).
- Test inject GS-orth to **ladder** `[M,N,Areeb,O]` (+ light SVD-10) so Step-O star stays clean.
- Full Top-40 SVD orth removed all LM residual (max|corr|=1 / LOFO collapse) — documented in V31.0→V31.1 lesson.

## z_honest
- Rank-average top bank members · AUC=**0.951419** · n_bank_pool=80

## Pillars
| Pillar | kind | sign | pos | meanΔ(e-5) | min(e-5) | sf25 | t | disp | max\|corr\| ladder | gate |
|--------|------|------|-----|------------|----------|------|---|------|-------------------|------|
| `P_L7_s2026_5f` | prob | -1 | 5/5 | +9.14 | +2.40 | 23/25 | 3.33 | 1.000 | 0.9999 | PASS |
| `P_gpt2med_5f` | prob | -1 | 4/5 | +6.51 | -2.55 | 19/25 | 2.00 | 1.000 | 0.9999 | FAIL |
| `P_L7_disp` | resid | -1 | 1/5 | -5.87 | -12.79 | 6/25 | -2.12 | 1.000 | 0.9999 | FAIL |
| `P4_L4.2` | resid | 1 | 0/5 | -35.44 | -49.22 | 0/25 | -8.55 | 1.000 | 0.9999 | FAIL |
| `P4_L4.5` | resid | 1 | 0/5 | -23.49 | -37.82 | 2/25 | -3.49 | 1.000 | 0.9999 | FAIL |
| `v10_ridge_orth_LM` | resid | -1 | 1/5 | -12.77 | -37.44 | 5/25 | -1.58 | 1.000 | 0.9999 | FAIL |

## Composites
- u_LM_prop: `[('P_L7_s2026_5f', 1.0)]`
- u_Full_prop: `[('P_L7_s2026_5f', 1.0)]`
- offline proxy: `{'StepO': 0.95536, 'C1_proxy_e4': 0.6337951412123122}`

## Live
| Slot | LB |
|------|----|
| SLOT1 StepO+5fLM @0.35e-4 | 0.95534 |
| SLOT2 adaptive | 0.95536 |

## Honest NOSUBMIT
- w=0.12 LOFO={'mean': 0.8328309648790722, 'pos': 5, 't': 4.631448247943418, 'min_fold': 0.41619145947136005, 'd': [0.4876, 0.8131, 1.0685, 0.4162, 1.3788]}
- `sub_v31_1_HONEST_700K_CV_SUPERSTACK_NOSUBMIT.csv`

## Log
```
[    0.0s] === GM V31.1b Step-O + 5f LM (fixed gates) | MAX_SUBMITS=2 ===
[    7.5s] submits_today=0 remaining_est=10
[    8.8s] ladder omr O↔N=1.0266 O↔Areeb=0.6170
[   32.7s] bank members solo>=0.94: 80 top=wide_lm@0.95257
[   38.2s] z_honest_152_oof (rankavg top40) AUC=0.951419
[   46.1s] light SVD-10 explained=0.9989
[   60.0s] PILLAR P_L7_s2026_5f: pos=5/5 mean=+0.914e-4 min=+0.240 sf=23/25 t=3.33 max|corr_ladder|=0.9999 pass=True
[   61.9s] PILLAR P_gpt2med_5f: pos=4/5 mean=+0.651e-4 min=-0.255 sf=19/25 t=2.00 max|corr_ladder|=0.9999 pass=False
[   63.7s] PILLAR P_L7_disp: pos=1/5 mean=-0.587e-4 min=-1.279 sf=6/25 t=-2.12 max|corr_ladder|=0.9999 pass=False
[   65.5s] PILLAR P4_L4.2: pos=0/5 mean=-3.544e-4 min=-4.922 sf=0/25 t=-8.55 max|corr_ladder|=0.9999 pass=False
[   67.6s] PILLAR P4_L4.5: pos=0/5 mean=-2.349e-4 min=-3.782 sf=2/25 t=-3.49 max|corr_ladder|=0.9999 pass=False
[   69.4s] PILLAR v10_ridge_orth_LM: pos=1/5 mean=-1.277e-4 min=-3.744 sf=5/25 t=-1.58 max|corr_ladder|=0.9999 pass=False
[   69.4s] u_LM_prop [('P_L7_s2026_5f', 1.0)]
[   69.4s] u_Full_prop [('P_L7_s2026_5f', 1.0)]
[   69.8s] COMPOSITE u_LM: pos=5/5 mean=+0.634e-4 t=4.77
[   70.3s] COMPOSITE u_Full: pos=5/5 mean=+0.634e-4 t=4.77
[   73.6s] HONEST SUPERSTACK w=0.12 OOF AUC=0.951505 LOFO 5/5 m=+0.833
[   73.6s] Board-model not shipped; offline proxy = LOFO mean on z_honest
[   77.4s] SLOT1 omr(StepO)=0.3501
[   83.6s] 6.0k/7.52M [00:00<00:52, 151kB/s]
  6%|▋         | 496k/7.52M [00:00<00:07, 999kB/s] 
  8%|▊         | 608k/7.52M [00:00<00:08, 853kB/s]
 10%|█         | 800k/7.52M [00:00<00:08, 865kB/s]
 57%|█████▋    | 4.30M/7.52M [00:01<00:00, 7.73MB/s]
 75%|███████▍  | 5.61M/7.52M [00:01<00:00, 7.28MB/s]
 94%|█████████▍| 7.05M/7.52M [00:01<00:00, 8.80MB/s]
100%|██████████| 7.52M/7.52M [00:02<00:00, 3.93MB/s]

[  102.7s] POLLED sub_v31_1_SLOT1_stepO_5flm.csv -> 0.95534
[  102.7s] CASE C: Regress — lock StepO
[  107.8s] .0k/7.52M [00:00<00:48, 162kB/s]
  7%|▋         | 576k/7.52M [00:00<00:06, 1.15MB/s]
  9%|▉         | 688k/7.52M [00:00<00:07, 958kB/s] 
 11%|█▏        | 880k/7.52M [00:00<00:07, 935kB/s]
 33%|███▎      | 2.52M/7.52M [00:01<00:01, 4.02MB/s]
 67%|██████▋   | 5.05M/7.52M [00:01<00:00, 8.78MB/s]
 84%|████████▍ | 6.30M/7.52M [00:01<00:00, 8.15MB/s]
100%|██████████| 7.52M/7.52M [00:01<00:00, 3.97MB/s]

[  126.6s] POLLED sub_v31_1_SLOT2_anchor_lock_095536.csv -> 0.95536
```
