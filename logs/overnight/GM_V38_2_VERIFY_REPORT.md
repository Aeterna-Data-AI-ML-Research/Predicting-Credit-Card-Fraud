# GM V38.2 Verify Report

**UTC:** 2026-11-22T13:42:09.556061+00:00
**Elapsed:** 545.4s
**Submitted:** False · publicScore=None

## 0. Bug Fixes Applied

1. **Wilcoxon on 25 subfolds** (`wilcoxon_p_25`) — N=5 outer-fold p has mathematical floor `(1/2)^5=0.03125`.
2. **ZCA Σ^{-1/2} whitening + Exact Test GS** vs `[SVD_30(B), Areeb, ATH, ATH−Areeb, z_fb, z_tw]`.
3. **S4 double-count strip:** `|corr(u_test, L(ATH)−L(Areeb))| < 1e-5` required.
4. **60k/240k sim re-anchored** at `z_fullbank` (~0.9526).

## 1. Anchored 60k/240k Simulation

```json
{
  "anchor_fullbank_oof_auc": 0.9526207252382072,
  "max_60k_pub_anchored_probe": 0.96061,
  "priv_240k_at_best_probe": 0.9518705284309594,
  "max_60k_pub_ridge_oracle": 0.9606082083096427,
  "priv_240k_at_oracle": 0.9518705239700197,
  "gap_to_ATH_0.95548": -0.005129999999999968,
  "gap_to_0.96242": 0.0018100000000000893,
  "reachable_0.9575": true,
  "reachable_0.96242": false,
  "trials_head": [
    {
      "t": 0,
      "base_pub": 0.95725,
      "pub_probe": 0.95725,
      "priv": 0.9521954644641965,
      "oracle": 0.9572516548651231
    },
    {
      "t": 1,
      "base_pub": 0.95108,
      "pub_probe": 0.95108,
      "priv": 0.952766422508372,
      "oracle": 0.9510848129836129
    },
    {
      "t": 2,
      "base_pub": 0.9479,
      "pub_probe": 0.9479,
      "priv": 0.9530633490450755,
      "oracle": 0.9478952870894706
    },
    {
      "t": 29,
      "base_pub": 0.95249,
      "pub_probe": 0.95249,
      "priv": 0.9526356470887546,
      "oracle": 0.9524929259417367
    }
  ],
  "note": "Anchored at z_fullbank (~0.9526); V38.1 0.928 was unoriented raw-bank mean."
}
```

## 2. Pillar LOFO (25-subfold Wilcoxon)

| name | F2 | corr_S4 | g_fb Δ | folds | sub | t25 | **p25** | p5(floor) | g_tw Δ | folds_tw | t25_tw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| u_p4_winsor | 0.000334 | -2.32e-05 | +1.32e-05 | 4/5 | 13/25 | 0.76 | **0.2289** | 0.2188 | +5.47e-07 | 4/5 | 0.58 |
| u_v10_winsor | 0.006033 | -4.32e-02 | +1.55e-05 | 4/5 | 16/25 | 1.50 | **0.1152** | 0.0625 | +6.27e-05 | 5/5 | 3.17 |
| u_g26_5f_clean | 0.091537 | +6.64e-04 | -9.45e-06 | 3/5 | 12/25 | 0.13 | **0.6144** | 0.5938 | -2.02e-07 | 3/5 | 0.30 |

## 3. Gates & Live Verdict

```json
{
  "gates": {
    "G1": {
      "pass": false,
      "mean_d": 1.316395095614098e-05,
      "n_pos_folds": 4,
      "min_fold": -2.5335959987438095e-05,
      "n_pos_sub": 13,
      "t25": 0.7622874132533557,
      "wilcoxon_p_25": 0.2289232313632965,
      "wilcoxon_p_5_IMPOSSIBLE_FLOOR": 0.21875
    },
    "G2": {
      "pass": false,
      "mean_d": 5.46695974068534e-07,
      "n_pos_folds": 4,
      "t25": 0.5827878242615574
    },
    "G3": {
      "pass": true,
      "disp": 0.9999979422607549,
      "KS": 0.0002200125721469798,
      "F2": 8.263351912163903e-06,
      "corr_S4": 5.393395171392114e-06
    },
    "G4": {
      "pass": true
    },
    "G5": {
      "pass": false,
      "step": 4.960238488860291e-05,
      "c": 0.7423076923076923,
      "min_hist": 1.2383127856452347e-10
    },
    "G6": {
      "pass": true,
      "n_today": 0
    }
  },
  "candidate": {
    "path": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v38_2_ZCA_ORTH_VERIFY.csv",
    "sha256": "2c04ea9a948c9ec5613b8d2052582078e5999e24b96abf55caec38b3f4f9f9d8",
    "step": 4.960238488860291e-05,
    "c": 0.7423076923076923,
    "w_opt": -0.0222
  },
  "submitted": false,
  "publicScore": null,
  "delta_vs_ATH": null,
  "submit_block": {
    "G1": {
      "pass": false,
      "mean_d": 1.316395095614098e-05,
      "n_pos_folds": 4,
      "min_fold": -2.5335959987438095e-05,
      "n_pos_sub": 13,
      "t25": 0.7622874132533557,
      "wilcoxon_p_25": 0.2289232313632965,
      "wilcoxon_p_5_IMPOSSIBLE_FLOOR": 0.21875
    },
    "G2": {
      "pass": false,
      "mean_d": 5.46695974068534e-07,
      "n_pos_folds": 4,
      "t25": 0.5827878242615574
    },
    "G5": {
      "pass": false,
      "step": 4.960238488860291e-05,
      "c": 0.7423076923076923,
      "min_hist": 1.2383127856452347e-10
    }
  },
  "s4_v10_saturated": false,
  "hold_reason": null
}
```

- **0 submits.** Hold remaining slots.
  - `G1`: 4/5 · 13/25 · t25=0.76 · **p25=0.23** (Wilcoxon-on-25 is valid; signal too weak)
  - `G2`: 4/5 · t25=0.58
  - `G5`: min_hist≈0 vs prior `sub_v38_2_ZCA_ORTH_VERIFY.csv` (duplicate of last hold candidate)
  - **v10:** twin still 5/5 t25=3.17, but after S4-strip `|corr(ATH−Areeb)|≈0.043`, KS/disp fail → **not safely separable from S4; ATH has largely saturated v10**. Only weak `u_p4` remains for synth.
  - `G5`: {'pass': False, 'step': 4.960238488860291e-05, 'c': 0.7423076923076923, 'min_hist': 1.2383127856452347e-10}
