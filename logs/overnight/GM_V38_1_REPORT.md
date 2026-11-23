# GM V38.1 Report

**UTC:** 2026-11-22T11:43:58.527827+00:00
**Elapsed:** 730.1s
**Submitted:** False · publicScore=None

## 1. 60k/240k #1 Monte-Carlo Verdict

```json
{
  "max_60k_pub_coord_CMA": 0.93997,
  "priv_240k_at_best_coord": 0.9268003665900869,
  "max_60k_pub_ridge_oracle": 0.9492881757418076,
  "priv_240k_at_best_oracle": 0.9363888728616828,
  "reachable_0.9575_via_bank_probe": false,
  "reachable_0.96242_via_bank_probe": false,
  "reachable_0.96242_via_cheating_oracle": false,
  "math_proof": "EXTERNAL/NON-BANK paradigm required to explain live 0.96242",
  "trials_head": [
    {
      "t": 0,
      "pub_coord": 0.92803,
      "priv_coord": 0.9279231611580507,
      "pub_oracle": 0.9386500990358007,
      "priv_oracle": 0.9383828224740767
    },
    {
      "t": 1,
      "pub_coord": 0.92784,
      "priv_coord": 0.9279418627553193,
      "pub_oracle": 0.9392171959684145,
      "priv_oracle": 0.9367606294304411
    },
    {
      "t": 2,
      "pub_coord": 0.9241,
      "priv_coord": 0.9282934152823371,
      "pub_oracle": 0.9383462218922946,
      "priv_oracle": 0.9388209616908822
    },
    {
      "t": 3,
      "pub_coord": 0.93573,
      "priv_coord": 0.9272038159045445,
      "pub_oracle": 0.9433936628590339,
      "priv_oracle": 0.9377673864081358
    },
    {
      "t": 4,
      "pub_coord": 0.9255,
      "priv_coord": 0.9281603389100513,
      "pub_oracle": 0.9346733412053143,
      "priv_oracle": 0.9388198954519583
    },
    {
      "t": 29,
      "pub_coord": 0.92429,
      "priv_coord": 0.9282752854288326,
      "pub_oracle": 0.936342407222573,
      "priv_oracle": 0.9381910564928577
    }
  ],
  "n_bank_cols": 138,
  "Ks": [
    114,
    138
  ]
}
```

**Math proof:** EXTERNAL/NON-BANK paradigm required to explain live 0.96242

## 2. Top-20 Fisher Residual-TE & Δ_gen

### Fisher keys

- `2:is_foreign_transaction×flag_bitmask_64|m15` chi2=10703.09 ks=0.4710
- `2:is_new_merchant×flag_bitmask_64|m15` chi2=10703.09 ks=0.4710
- `2:used_vpn×flag_bitmask_64|m15` chi2=10703.09 ks=0.4710
- `2:ip_country_mismatch×flag_bitmask_64|m15` chi2=10703.09 ks=0.4710
- `2:billing_shipping_mismatch×flag_bitmask_64|m15` chi2=10703.09 ks=0.4710
- `2:is_ai_generated_scam_attempt×flag_bitmask_64|m15` chi2=10703.09 ks=0.4710
- `2:flag_bitmask_64×hours_since_last_txn__d34|m15` chi2=10264.75 ks=0.4450
- `2:flag_bitmask_64×distance_from_home_km__d34|m15` chi2=10249.07 ks=0.4411
- `2:flag_bitmask_64×amount_usd__d34|m15` chi2=10183.68 ks=0.4418
- `2:prior_disputes×flag_bitmask_64|m15` chi2=9520.57 ks=0.4025
- `2:cvv_retry_count×flag_bitmask_64|m15` chi2=9409.46 ks=0.4341
- `3:cvv_retry_count×is_new_merchant×ip_country_mismatch|m15` chi2=9213.04 ks=0.4611
- `2:auth_method×flag_bitmask_64|m15` chi2=9157.00 ks=0.1416
- `3:auth_method×used_vpn×flag_bitmask_64|m15` chi2=9157.00 ks=0.1416
- `2:is_new_merchant×ip_country_mismatch|m15` chi2=9060.09 ks=0.5007
- `2:flag_bitmask_64×amount_usd__strlen|m15` chi2=8782.88 ks=0.3710
- `2:flag_bitmask_64×hours_since_last_txn__strlen|m15` chi2=8650.59 ks=0.3130
- `2:flag_bitmask_64×distance_from_home_km__strlen|m15` chi2=8550.65 ks=0.3743
- `3:is_ai_generated_scam_attempt×flag_bitmask_64×distance_from_home_km__strlen|m15` chi2=8550.65 ks=0.3743
- `3:auth_method×prior_disputes×flag_bitmask_64|m15` chi2=7983.95 ks=0.1213

### Δ_gen cells

- `cvv_retry_count=2` Δ=+0.24480 (p_tr=0.4142 p_orig=0.1614 n=1538/254)
- `cvv_retry_count=1` Δ=+0.04121 (p_tr=0.0878 p_orig=0.0465 n=44768/3073)
- `merchant_category=Gift Cards` Δ=+0.02744 (p_tr=0.0714 p_orig=0.0438 n=7607/617)
- `auth_method=No Authentication` Δ=+0.02003 (p_tr=0.0644 p_orig=0.0444 n=53757/2388)
- `merchant_category=Gaming` Δ=+0.01967 (p_tr=0.0564 p_orig=0.0367 n=18988/1036)
- `merchant_category=Crypto Exchange` Δ=+0.01425 (p_tr=0.0643 p_orig=0.0499 n=4244/561)
- `merchant_category=Electronics` Δ=+0.00743 (p_tr=0.0311 p_orig=0.0236 n=38956/2032)
- `merchant_category=Travel` Δ=+0.00732 (p_tr=0.0186 p_orig=0.0113 n=22897/1420)
- `device_type=Windows PC` Δ=+0.00519 (p_tr=0.0162 p_orig=0.0111 n=64619/2623)
- `card_type=Discover` Δ=-0.00501 (p_tr=0.0145 p_orig=0.0196 n=13776/1023)
- `merchant_category=Utilities` Δ=-0.00488 (p_tr=0.0139 p_orig=0.0188 n=33724/1330)
- `device_type=Android Phone` Δ=-0.00290 (p_tr=0.0160 p_orig=0.0189 n=190206/5503)
- `card_type=RuPay` Δ=+0.00290 (p_tr=0.0163 p_orig=0.0134 n=29635/1796)
- `auth_method=Biometric` Δ=-0.00283 (p_tr=0.0049 p_orig=0.0078 n=75038/3093)
- `auth_method=PIN` Δ=-0.00274 (p_tr=0.0088 p_orig=0.0115 n=123198/4173)
- `channel=ATM` Δ=-0.00272 (p_tr=0.0155 p_orig=0.0182 n=24312/1374)
- `channel=Online` Δ=-0.00218 (p_tr=0.0176 p_orig=0.0198 n=219120/6810)
- `merchant_category=Online Retail` Δ=+0.00182 (p_tr=0.0112 p_orig=0.0094 n=124092/3416)
- `device_type=Tablet` Δ=-0.00168 (p_tr=0.0173 p_orig=0.0189 n=22127/1161)
- `auth_method=OTP` Δ=-0.00167 (p_tr=0.0140 p_orig=0.0157 n=140039/4792)

## 3. Pillar LOFO Table

| name | g_fb Δ | folds | sub | t25 | p | g_tw Δ | folds_tw | t25_tw | same |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| u_fisher_cube | +1.01e-06 | 3/5 | 13/25 | 0.94 | 0.4062 | +7.31e-07 | 3/5 | 0.81 | True |
| u_platt_jacobian | -2.65e-05 | 0/5 | 14/25 | 2.13 | 1.0000 | +2.53e-05 | 3/5 | 1.65 | False |
| u_p4_winsor | +1.59e-05 | 4/5 | 14/25 | 0.87 | 0.2188 | +1.01e-06 | 4/5 | 0.70 | True |
| u_v10_winsor | +3.88e-05 | 3/5 | 16/25 | 2.50 | 0.1562 | +1.31e-04 | 5/5 | 4.08 | True |

## 4. Live 1-Submit / Hold Telemetry

```json
{
  "gates": {
    "G1": {
      "pass": false,
      "mean_d": 7.027277967976974e-05,
      "n_pos_folds": 5,
      "min_fold": 3.1175033451624756e-05,
      "n_pos_sub": 18,
      "t25": 2.8472861527693745,
      "wilcoxon_p": 0.03125
    },
    "G2": {
      "pass": true,
      "mean_d": 0.00014204862126938612,
      "n_pos_folds": 5,
      "t25": 4.7445177423484814
    },
    "G3": {
      "pass": false,
      "disp": 0.9999999999985161,
      "KS": 0.0028868553390298257,
      "F2": 0.1131267015110242,
      "dbeta": -0.003000238438679567
    },
    "G4": {
      "pass": true
    },
    "G5": {
      "pass": "True",
      "step": 0.00020779766674028366,
      "min_hist": 0.00012298907999119812
    },
    "G6": {
      "pass": true,
      "n_today": 0
    }
  },
  "candidate": {
    "path": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v38_1_BEST_0p85x.csv",
    "sha256": "589be812043a9ed350cedc3bd7ec17d81ab7b56b638204e69b0a19d7010da1a3",
    "step": 0.00020779766674028366,
    "w_opt": 0.047,
    "c": 0.85
  },
  "submitted": false,
  "publicScore": null,
  "delta_vs_ATH": null,
  "submit_block": {
    "G1": {
      "pass": false,
      "mean_d": 7.027277967976974e-05,
      "n_pos_folds": 5,
      "min_fold": 3.1175033451624756e-05,
      "n_pos_sub": 18,
      "t25": 2.8472861527693745,
      "wilcoxon_p": 0.03125
    },
    "G3": {
      "pass": false,
      "disp": 0.9999999999985161,
      "KS": 0.0028868553390298257,
      "F2": 0.1131267015110242,
      "dbeta": -0.003000238438679567
    }
  }
}
```

- **0 submits.** Hold all 5 slots.
  - `G1`: {'pass': False, 'mean_d': 7.027277967976974e-05, 'n_pos_folds': 5, 'min_fold': 3.1175033451624756e-05, 'n_pos_sub': 18, 't25': 2.8472861527693745, 'wilcoxon_p': 0.03125}
  - `G3`: {'pass': False, 'disp': 0.9999999999985161, 'KS': 0.0028868553390298257, 'F2': 0.1131267015110242, 'dbeta': -0.003000238438679567}
