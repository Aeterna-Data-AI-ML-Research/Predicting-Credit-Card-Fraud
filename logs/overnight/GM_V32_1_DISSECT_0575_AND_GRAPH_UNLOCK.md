# GM V32.1 — Dissect SCAN4 `0.57489` Local-Rank + Graph Unlock

**NO_SUBMIT = True** · wide_lm baseline AUC = **0.952565** · generated 2026-11-21 09:08

## Test 1 — Integer Rank vs Group Size vs Uniform `u_id_grp`

### SCAN4 G1 conclusion
**SIZE LEAK (A):** integer-rank AUC≈0.57522 is largely **group_size** (AUC=0.60136); uniform `u` drops to 0.50112 (corr(u,size)=0.0). SCAN4 0.57489 was **not** pure within-group order.

### All cohorts
| group | n_groups | corr(u,size) | AUC rank | AUC size | AUC u | verdict |
|-------|---------:|-------------:|---------:|---------:|------:|---------|
| `G1_card_merch_dow_hour` | 9968 | -0.007827 | 0.57522 | 0.60323 | 0.50327 | SIZE_LEAK |
| `G2_exact_balance` | 49123 | -0.017713 | 0.50888 | 0.50313 | 0.50151 | WEAK/NULL |
| `G3_exact_amount` | 14800 | -0.007503 | 0.53975 | 0.5492 | 0.507 | SIZE_LEAK |
| `G4_merch_chan_dev_auth` | 2395 | -0.001383 | 0.5978 | 0.62666 | 0.5015 | SIZE_LEAK |
| `G5_flag_bitmask` | 145 | -0.000726 | 0.81937 | 0.83896 | 0.50908 | SIZE_LEAK |
| `G6_global_id_quantile_in_all` | | | | | | `{'auc_u': 0.50217, 'auc_raw_id': 0.50225, 'auc_u_train_only_denom': 0.50225}` |
| `G1_train_only_SCAN4_repro` |  |  | 0.57522 | 0.60136 | 0.50112 | {'auc_integer_rank': 0.57522, 'auc_group_size': 0.60136, 'auc_u_id_grp': 0.50112, 'corr(u,size)': 0.0} |

### Train/test split structure
```json
{
  "auc_id_predicts_is_test": 1.0,
  "auc_id_diff_to_nearest_test_vs_y": 0.50217,
  "auc_local_test_frac_k20_vs_y": 0.49997,
  "auc_local_test_frac_k100_vs_y": 0.49997,
  "mean_d_near": 349980.5
}
```

### Neighbor lags (id-sorted within group)
```json
{
  "G1_card_merch_dow_hour": {
    "-1": {
      "n": 690074,
      "auc_y_neighbor": 0.50454,
      "auc_z_neighbor": 0.55416,
      "auc_id_diff": 0.55699,
      "mean_y_neighbor_when_y1": 0.02547,
      "base": 0.01655
    },
    "+1": {
      "n": 690074,
      "auc_y_neighbor": 0.50455,
      "auc_z_neighbor": 0.55791,
      "auc_id_diff": 0.56454,
      "mean_y_neighbor_when_y1": 0.0255,
      "base": 0.01654
    },
    "-2": {
      "n": 680495,
      "auc_y_neighbor": 0.50422,
      "auc_z_neighbor": 0.5505,
      "auc_id_diff": 0.57381,
      "mean_y_neighbor_when_y1": 0.02471,
      "base": 0.01641
    },
    "+2": {
      "n": 680495,
      "auc_y_neighbor": 0.50422,
      "auc_z_neighbor": 0.55731,
      "auc_id_diff": 0.57768,
      "mean_y_neighbor_when_y1": 0.02471,
      "base": 0.01641
    }
  },
  "G2_exact_balance": {
    "-1": {
      "n": 656643,
      "auc_y_neighbor": 0.50155,
      "auc_z_neighbor": 0.51653,
      "auc_id_diff": 0.50468,
      "mean_y_neighbor_when_y1": 0.0197,
      "base": 0.01662
    },
    "+1": {
      "n": 656643,
      "auc_y_neighbor": 0.50155,
      "auc_z_neighbor": 0.51906,
      "auc_id_diff": 0.49917,
      "mean_y_neighbor_when_y1": 0.01967,
      "base": 0.01665
    },
    "-2": {
      "n": 628605,
      "auc_y_neighbor": 0.50091,
      "auc_z_neighbor": 0.51577,
      "auc_id_diff": 0.50275,
      "mean_y_neighbor_when_y1": 0.01842,
      "base": 0.01658
    },
    "+2": {
      "n": 628605,
      "auc_y_neighbor": 0.50091,
      "auc_z_neighbor": 0.51848,
      "auc_id_diff": 0.50044,
      "mean_y_neighbor_when_y1": 0.01837,
      "base": 0.01663
    }
  },
  "G3_exact_amount": {
    "-1": {
      "n": 686573,
      "auc_y_neighbor": 0.50276,
      "auc_z_neighbor": 0.52729,
      "auc_id_diff": 0.53149,
      "mean_y_neighbor_when_y1": 0.02189,
      "base": 0.01643
    },
    "+1": {
      "n": 686573,
      "auc_y_neighbor": 0.50275,
      "auc_z_neighbor": 0.52708,
      "auc_id_diff": 0.53235,
      "mean_y_neighbor_when_y1": 0.02185,
      "base": 0.01647
    },
    "-2": {
      "n": 676966,
      "auc_y_neighbor": 0.50284,
      "auc_z_neighbor": 0.52595,
      "auc_id_diff": 0.53446,
      "mean_y_neighbor_when_y1": 0.02191,
      "base": 0.01632
    },
    "+2": {
      "n": 676966,
      "auc_y_neighbor": 0.50284,
      "auc_z_neighbor": 0.52904,
      "auc_id_diff": 0.54,
      "mean_y_neighbor_when_y1": 0.02189,
      "base": 0.01633
    }
  },
  "G4_merch_chan_dev_auth": {
    "-1": {
      "n": 697569,
      "auc_y_neighbor": 0.51631,
      "auc_z_neighbor": 0.62818,
      "auc_id_diff": 0.5844,
      "mean_y_neighbor_when_y1": 0.04872,
      "base": 0.01662
    },
    "+1": {
      "n": 697569,
      "auc_y_neighbor": 0.5163,
      "auc_z_neighbor": 0.6302,
      "auc_id_diff": 0.58735,
      "mean_y_neighbor_when_y1": 0.04869,
      "base": 0.01664
    },
    "-2": {
      "n": 695193,
      "auc_y_neighbor": 0.5156,
      "auc_z_neighbor": 0.62779,
      "auc_id_diff": 0.59973,
      "mean_y_neighbor_when_y1": 0.04727,
      "base": 0.01659
    },
    "+2": {
      "n": 695193,
      "auc_y_neighbor": 0.5156,
      "auc_z_neighbor": 0.63155,
      "auc_id_diff": 0.60295,
      "mean_y_neighbor_when_y1": 0.04728,
      "base": 0.01658
    }
  }
}
```

## Test 2 — Entity-Graph Peer Propagation on `wide_lm`

Top lifts (Δ in e-4 = bps×0.1? **ΔAUC × 1e4**):

| feature | solo AUC | stack AUC | Δe-4 | R² vs bank40 |
|---------|---------:|----------:|-----:|-------------:|
| `k1_balance_peer_std` | 0.49893 | 0.952458 | -1.07 | 0.0026 |
| `k2_amount_peer_std` | 0.49560 | 0.952418 | -1.48 | 0.0042 |
| `k6_hours_peer_std` | 0.46604 | 0.952294 | -2.71 | 0.0105 |
| `resid_balance_m30` | 0.48515 | 0.951961 | -6.04 | 0.0161 |
| `k3_distance_peer_std` | 0.51068 | 0.951890 | -6.75 | 0.0017 |
| `resid_amount_m10` | 0.46834 | 0.951772 | -7.93 | 0.0106 |
| `resid_amount_m3` | 0.43220 | 0.951641 | -9.24 | 0.0127 |
| `resid_balance_m10` | 0.46475 | 0.951426 | -11.39 | 0.0135 |
| `resid_balance_m3` | 0.44175 | 0.950854 | -17.11 | 0.0101 |
| `hop2_bal_to_amt` | 0.53772 | 0.950733 | -18.32 | 0.0055 |
| `k4_velocity_peer_mean` | 0.71364 | 0.950592 | -19.73 | 0.3129 |
| `k7_bal_merch_peer_std` | 0.45760 | 0.950057 | -25.09 | 0.0051 |
| `k8_amt_merch_peer_std` | 0.46572 | 0.949987 | -25.78 | 0.007 |
| `resid_amount_m30` | 0.50272 | 0.949951 | -26.14 | 0.013 |
| `k3_distance_peer_mean` | 0.54172 | 0.949684 | -28.81 | 0.0054 |
| `k1_balance_peer_mean` | 0.58630 | 0.948563 | -40.02 | 0.0113 |
| `k6_hours_peer_mean` | 0.61266 | 0.946614 | -59.51 | 0.0625 |
| `k4_velocity_peer_max` | 0.61176 | 0.944845 | -77.21 | 0.1005 |
| `k5_merch_risk_peer_max` | 0.64870 | 0.944387 | -81.78 | 0.1904 |
| `k5_merch_risk_peer_mean` | 0.75651 | 0.943499 | -90.66 | 0.3041 |

## Test 3 — Raw Float-String Forensics

| feature | solo TE/tree AUC | Δ residTE e-4 | Δ raw e-4 | R² residTE |
|---------|-----------------:|--------------:|----------:|-----------:|
| `velocity_score.trail_pat` | 0.50546 | -1.19 | -57.30 | 0.021 |
| `velocity_score.last_digit` | 0.49599 | -1.11 | -54.26 | 0.004 |
| `velocity_score.str_len` | 0.58327 | -0.32 | -27.33 | 0.011 |
| `amount_usd.str_len` | 0.53499 | -0.63 | -10.12 | 0.004 |
| `merchant_risk_score.str_len` | 0.50317 | -0.94 | -4.38 | 0.011 |
| `hours_since_last_txn.n_dec` | 0.50365 | -0.23 | -2.91 | 0.001 |
| `merchant_risk_score.trail_pat` | 0.50140 | -0.23 | -2.82 | 0.007 |
| `distance_from_home_km.last_digit` | 0.50514 | -0.48 | -2.46 | 0.001 |
| `account_balance_usd.n_dec` | 0.49840 | -0.25 | -2.45 | 0.002 |
| `distance_from_home_km.n_dec` | 0.49771 | -0.24 | -1.75 | 0.001 |
| `amount_usd.last_digit` | 0.50028 | -0.28 | -1.48 | 0.000 |
| `hours_since_last_txn.str_len` | 0.51325 | -0.23 | -1.32 | 0.003 |
| `amount_usd.trail_pat` | 0.49886 | -0.26 | -1.26 | 0.002 |
| `account_balance_usd.trail_pat` | 0.49995 | -0.27 | -1.22 | 0.001 |
| `account_balance_usd.last_digit` | 0.50010 | -0.26 | -1.17 | 0.000 |

## Test 4 — Non-Linear L2 LightGBM Meta-Stacker

| metric | value |
|--------|------:|
| wide_lm AUC | 0.952565 |
| LGB L2 OOF AUC | 0.951676 |
| Δ (e-4) | -8.893 |
| n_features | 46 |

Peer feats used: `['k1_balance_peer_std', 'k2_amount_peer_std', 'k6_hours_peer_std', 'k3_distance_peer_std']`

## Executive Verdict

1. **SCAN4 `0.57489` = (A) GROUP-SIZE LEAK, not (B) true id order.**  
   G1 train-only repro: integer-rank **0.57522** ≈ SCAN4; **group_size alone 0.60136**; uniform `u_id_grp` collapses to **0.50112** with `corr(u,size)=0`. Same pattern on G3/G4/G5. **Within-group fraud ordering by id is random.**
2. **Neighbor y-lags ~0.50–0.52** (null). Apparent `id_diff` AUCs ~0.56 inside G1 are the same size-proxy artifact (gap distribution tracks cohort size), not label autocorrelation.
3. **`id → is_test` AUC = 1.0** because train/test are contiguous id blocks (`mean d_near ≈ 350k`); local test-fraction vs `y` is **0.50** — no useful split geometry for fraud.
4. **V18.4 peer-graph on `wide_lm` does NOT unlock:** all peer_mean / hop2 / residual-TE stacks are **negative** (best “lift” −1.07e-4). The old +23.4e-5 was vs a weaker base, not vs 0.95257.
5. **Float-string / LGB L2:** no positive incremental path; LGB meta **−8.89e-4** vs linear wide_lm. Does **not** explain 0.956–0.961.
6. **NO_SUBMIT = True** — zero files submitted; ~6 slots kept.

## Log
```
[   0.0s] === GM V32.1 dissect 0.57489 local-rank + graph unlock | NO_SUBMIT ===
[  14.8s] wide_lm AUC=0.952565 | train=699960 test=299984
[  16.0s] TEST1: dissect integer-rank vs group_size vs uniform u_id_grp…
[  26.5s]   G1_card_merch_dow_hour: n_g=9968 corr(u,size)=-0.0078 | AUC rank=0.5752 size=0.6032 u=0.5033 → SIZE_LEAK
[  33.5s]   G2_exact_balance: n_g=49123 corr(u,size)=-0.0177 | AUC rank=0.5089 size=0.5031 u=0.5015 → WEAK/NULL
[  41.9s]   G3_exact_amount: n_g=14800 corr(u,size)=-0.0075 | AUC rank=0.5398 size=0.5492 u=0.5070 → SIZE_LEAK
[  56.0s]   G4_merch_chan_dev_auth: n_g=2395 corr(u,size)=-0.0014 | AUC rank=0.5978 size=0.6267 u=0.5015 → SIZE_LEAK
[  65.4s]   G5_flag_bitmask: n_g=145 corr(u,size)=-0.0007 | AUC rank=0.8194 size=0.8390 u=0.5091 → SIZE_LEAK
[  73.9s]   G6 global id AUC u_all=0.5022 raw=0.5022
[  80.3s]   G1 train-only SCAN4 repro: rank=0.5752 size=0.6014 u=0.5011
[  80.3s] TEST1b: within-cohort id-sorted neighbor lags…
[  93.9s]   lag G1_card_merch_dow_hour: best y-neighbor AUC=('+1', 0.50455)
[ 104.8s]   lag G2_exact_balance: best y-neighbor AUC=('-1', 0.50155)
[ 113.6s]   lag G3_exact_amount: best y-neighbor AUC=('-2', 0.50284)
[ 121.7s]   lag G4_merch_chan_dev_auth: best y-neighbor AUC=('-1', 0.51631)
[ 121.7s] TEST1c: is_test vs id structure…
[ 125.0s]   is_test: AUC(id→is_test)=1.0000 | d_near vs y=0.5022 | f20=0.5000
[ 125.0s] TEST2: entity-graph peer props on wide_lm…
[ 135.4s]   bank cols for R²: 70
[ 141.5s]   k1_balance peer_mean Δ=-40.02e-4 stack=0.948563
[ 144.3s]   k2_amount peer_mean Δ=-111.48e-4 stack=0.941417
[ 147.0s]   k3_distance peer_mean Δ=-28.81e-4 stack=0.949684
[ 149.4s]   k4_velocity peer_mean Δ=-19.73e-4 stack=0.950592
[ 152.1s]   k5_merch_risk peer_mean Δ=-90.66e-4 stack=0.943499
[ 154.8s]   k6_hours peer_mean Δ=-59.51e-4 stack=0.946614
[ 159.2s]   k7_bal_merch peer_mean Δ=-106.29e-4 stack=0.941936
[ 162.7s]   k8_amt_merch peer_mean Δ=-189.38e-4 stack=0.933627
[ 162.7s]   2-hop bipartite balance↔amount…
[ 165.3s]   hop2 Δ=-18.32e-4
[ 165.3s]   fold-safe residual TE on balance/amount…
[ 166.1s]   resid balance m=3: Δ=-17.11e-4 stack=0.950854 R²=0.010
[ 166.8s]   resid balance m=10: Δ=-11.39e-4 stack=0.951426 R²=0.013
[ 167.4s]   resid balance m=30: Δ=-6.04e-4 stack=0.951961 R²=0.016
[ 168.0s]   resid amount m=3: Δ=-9.24e-4 stack=0.951641 R²=0.013
[ 168.7s]   resid amount m=10: Δ=-7.93e-4 stack=0.951772 R²=0.011
[ 169.2s]   resid amount m=30: Δ=-26.14e-4 stack=0.949951 R²=0.013
[ 169.2s]   TEST2 best: k1_balance_peer_std Δ=-1.07e-4
[ 169.2s] TEST3: raw CSV float-string forensics…
[ 180.0s]   amount_usd: str_len solo≈TE done; best Δ raw/resid logged
[ 185.0s]   account_balance_usd: str_len solo≈TE done; best Δ raw/resid logged
[ 189.9s]   distance_from_home_km: str_len solo≈TE done; best Δ raw/resid logged
[ 202.6s]   hours_since_last_txn: str_len solo≈TE done; best Δ raw/resid logged
[ 210.1s]   velocity_score: str_len solo≈TE done; best Δ raw/resid logged
[ 216.6s]   merchant_risk_score: str_len solo≈TE done; best Δ raw/resid logged
[ 216.6s]   TEST3 best: {'name': 'velocity_score.trail_pat', 'auc_solo_te': 0.50546, 'delta_residTE_e4': -1.186, 'delta_raw_e4': -57.302, 'R2_residTE': 0.021, 'R2_raw': 0.1338, 'residue_maxabs': 0.0}
[ 216.6s] TEST4: LightGBM L2 over bank+peers+raw…
[ 261.3s]   selected 27 bank OOFs
[ 262.9s]   meta X shape=(699960, 46)
[ 322.5s]   LGB L2 AUC=0.951676 vs wide_lm=0.952565 Δ=-8.89e-4
```
