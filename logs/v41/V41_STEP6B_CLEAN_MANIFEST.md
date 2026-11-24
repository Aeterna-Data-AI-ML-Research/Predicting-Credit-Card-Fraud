# V41 STEP 6b — Clean Manifest (v10/zanch PURGED)

**NO_SUBMIT=True** · **LB_01 (live SLOT01) = 0.95569** · remaining daily ≈ 9

## Executive

- Quarantined: `v10_ridge*`, `v10_cb_d4*`, `zanch_*` (+ prior TE-bleed list)
- `z_pub_only` (part12) raw=0.952565 +Zr=**0.952614** (≥ Part12+Zr 0.952614) — zero private files
- `Fullbank_clean` (wide+0.02orth_nest, w_v10=0) raw=0.952566 +Zr=**0.952614**
- Private lift `Fullbank_clean - z_pub_only` fold wins: **1/5**
- Passing clean dirs: **['C3']**
- Slots 04–10 refreshed under `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/out/v41_final_10slots/` (01–03 intact)

## Scorecard

```json
{
  "C1_seq_llr": {
    "sign": -1,
    "pos": 1,
    "mean_e5": -0.6717726942984115,
    "sf25": 7,
    "folds": [
      0.0,
      -2.7333877852586674,
      -1.0868921760498473,
      -0.15287334208524683,
      0.6142898319017043
    ],
    "pass": false,
    "disp": 0.9999999999990001,
    "KS": 0.02145365953890932,
    "weights": {
      "gpt2_med": 0.33333333333299997,
      "l7_s2026": 0.33333333333299997,
      "xgb_llr": 0.33333333333299997
    },
    "P_perp": {
      "cos": 35650.34766444926,
      "projected": false
    }
  },
  "C2_famE_l7swap": {
    "sign": 1,
    "pos": 0,
    "mean_e5": 0.0,
    "sf25": 0,
    "folds": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "pass": false,
    "disp": 0.999999999999,
    "KS": 0.003151744449781557,
    "best_w_sweep": "-0.005",
    "P_perp": {
      "cos": 18037.19199153736,
      "projected": false
    }
  },
  "C3_harvest_p4": {
    "sign": -1,
    "pos": 4,
    "mean_e5": 1.1196471772789707,
    "sf25": 18,
    "folds": [
      5.975137032654398,
      1.211490164221285,
      3.19325068420806,
      -7.481161498845701,
      2.699519504156811
    ],
    "pass": true,
    "disp": 0.9999999999990002,
    "KS": 0.0008755449527741987,
    "P_perp": {
      "cos": 37861.88776143899,
      "projected": false
    },
    "hygiene": {
      "disp": 0.9999999999920395,
      "KS": 4.6956425518995815e-06
    }
  },
  "C4_fullbank_priv_diff": {
    "sign": -1,
    "pos": 0,
    "mean_e5": 0.0,
    "sf25": 0,
    "folds": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "pass": false,
    "disp": 1.0000000009622547,
    "KS": 0.312591969512542,
    "P_perp": {
      "cos": -2788.865520324956,
      "projected": true
    }
  }
}
```

## 10-Slot Master Table

| Slot | File | 1-ρ vs AB e4 | minDiv e4 | div_ok |
|---|---|---:|---:|---|
| SLOT01 | `SLOT01_pub_Step_AB_095569.csv` | 0.0000 |  | True |
| SLOT02 | `SLOT02_AB_clip985_areeb_a0p32.csv` | 0.0275 | 0.0275 | True |
| SLOT03 | `SLOT03_AB_clip985_quarter_a0p28.csv` | 0.1200 | 0.0392 | True |
| SLOT04 | `SLOT04_AB_C1_seq_llr_clip985_0p25e4.csv` | 0.2500 | 0.0432 | True |
| SLOT05 | `SLOT05_AB_C1_seq_llr_clip985_0p55e4.csv` | 0.5500 | 0.0585 | True |
| SLOT06 | `SLOT06_AB_C3_harvest_p4_clip985_0p28e4.csv` | 0.2800 | 0.2800 | True |
| SLOT07 | `SLOT07_AB_C4_fullbank_clean_priv_diff_clip985_0p40e4.csv` | 0.4000 | 0.0120 | False |
| SLOT08 | `SLOT08_AB_master_clean_combo_0p45e4.csv` | 0.4500 | 0.0311 | True |
| SLOT09 | `SLOT09_pure_nested_cv_champion_zr.csv` | 171.8140 | 171.8140 | True |
| SLOT10 | `SLOT10_private_hedge_55BestTip_45CVChamp_zr.csv` | 35.0211 | 35.0211 | True |

## 10×10 1-ρ (×1e-4)

| |SLOT01|SLOT02|SLOT03|SLOT04|SLOT05|SLOT06|SLOT07|SLOT08|SLOT09|SLOT10|
|---|---|---|---|---|---|---|---|---|---|---|
| SLOT01 | 0.000 | 0.028 | 0.120 | 0.250 | 0.550 | 0.280 | 0.400 | 0.450 | 171.814 | 35.021 |
| SLOT02 | 0.028 | 0.000 | 0.039 | 0.112 | 0.332 | 0.282 | 0.218 | 0.408 | 173.777 | 35.929 |
| SLOT03 | 0.120 | 0.039 | 0.000 | 0.043 | 0.185 | 0.335 | 0.107 | 0.414 | 174.888 | 36.485 |
| SLOT04 | 0.250 | 0.112 | 0.043 | 0.000 | 0.058 | 0.454 | 0.018 | 0.491 | 177.903 | 37.929 |
| SLOT05 | 0.550 | 0.332 | 0.185 | 0.058 | 0.000 | 0.718 | 0.012 | 0.689 | 181.028 | 39.515 |
| SLOT06 | 0.280 | 0.282 | 0.335 | 0.454 | 0.718 | 0.000 | 0.584 | 0.031 | 173.965 | 36.154 |
| SLOT07 | 0.400 | 0.218 | 0.107 | 0.018 | 0.012 | 0.584 | 0.000 | 0.585 | 179.601 | 38.783 |
| SLOT08 | 0.450 | 0.408 | 0.414 | 0.491 | 0.689 | 0.031 | 0.585 | 0.000 | 175.909 | 37.133 |
| SLOT09 | 171.814 | 173.777 | 174.888 | 177.903 | 181.028 | 173.965 | 179.601 | 175.909 | 0.000 | 52.169 |
| SLOT10 | 35.021 | 35.929 | 36.485 | 37.929 | 39.515 | 36.154 | 38.783 | 37.133 | 52.169 | 0.000 |

## STOP

LB_01=0.95569. Step 6b complete. **NO_SUBMIT for slots 02–10.** Awaiting explicit `SLOT 02` command before any further live burn.
