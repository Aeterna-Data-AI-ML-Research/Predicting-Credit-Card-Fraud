# V41 STEP4 AUDIT — kabartay D27 Zero-Fraud Zones vs Step_AB Zr

**NO_SUBMIT=True** · Hard stop. Do not run Step 5 without `PROCEED TO STEP 5`.

## Executive findings

- Kabartay §8 hand zones **≡** `Z_goodpjw` on test: `|∩|=16730`, `|D27\\Zr|=0`, identical=True.
- Train `Z_goodpjw`: n=39149, **frauds=0** (rate 0 vs baseline 0.01670).
- Kabartay OOF edge-refit: held-out zone frauds = **0**; incremental vs Zr = **0**.
- Cell-miner N_min=80: test incremental vs Zr = **6858**; OOF inc frauds = 17.
- **LIVE_STAGE = `SKIP (keep for Step 5 own-stack only)`**
- Reason: Z_D27_kabartay ≡ Z_goodpjw on test (inc=0). Mined N_min=80 test_inc=6858 OOF marginal pos5=0 mean_e5=-66.760 — not clearly positive; LIVE_STAGE=SKIP.
- EB feature: disp=0.9957 KS=0.1119 (FAIL hygiene) → `v41_d27_eb_zone_oof.npy`.
- Step5 note: **exclude TE-leaked `uid_catboost`** from nested CV.

## 1. Lineage & overlap

```json
{
  "lineage": {
    "notebook": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/v41_nov23_frontier/kabartay/s6e11-own-models-only-honest-5-fold-cv.ipynb",
    "section": "8 \u00b7 Zero-fraud zones",
    "discovery": "Hand rules published by daoviet/goodpjw/Weni Mohamed; found on FULL train (potential selection bias). kabartay mitigates by refitting each merchant-risk edge inside train_idx (just below lowest fraud in the zone-condition, never above published edge). Velocity edges stay published (residual optimism).",
    "N_min": "NOT used in kabartay hand rules; applied in our supplemental cell-miner.",
    "zones": [
      "Z1: merchant_risk_score < 13.1",
      "Z2: velocity_score <= 4.1 AND merchant_risk_score <= 20",
      "Z3: clean(8 flags) AND velocity <= 4.1 AND risk <= 33.8",
      "Z4: clean(8 flags) AND velocity <= 6.0 AND risk <= 25"
    ],
    "postprocess": "OOF: zoned[val]=stack_oof.min()-1 for zone rows. Test: p[z_test] *= 1e-6 (soft bottom).",
    "reported": "Markdown: small gain every fold; 0 held-out frauds in refitted zones."
  },
  "overlap": {
    "test": {
      "n_Z_goodpjw": 16730,
      "n_Z_D27_pub": 16730,
      "intersection": 16730,
      "D27_minus_goodpjw": 0,
      "goodpjw_minus_D27": 0,
      "identical": true
    },
    "train": {
      "n_Z_goodpjw": 39149,
      "n_Z_D27_pub": 39149,
      "intersection": 39149,
      "D27_minus_goodpjw": 0,
      "goodpjw_minus_D27": 0,
      "fraud_in_Z_goodpjw": 0,
      "fraud_rate_Z_goodpjw": 0.0,
      "baseline_fraud_rate": 0.016699525687182123
    }
  }
}
```

## 2. Strict OOF leakage

```json
{
  "kabartay_edge_refit": [
    {
      "fold": 0,
      "n_val_zone": 7946,
      "n_val_fraud": 0,
      "val_fraud_rate": 0.0,
      "n_inc_vs_Zr": 0,
      "n_inc_fraud": 0
    },
    {
      "fold": 1,
      "n_val_zone": 7754,
      "n_val_fraud": 0,
      "val_fraud_rate": 0.0,
      "n_inc_vs_Zr": 0,
      "n_inc_fraud": 0
    },
    {
      "fold": 2,
      "n_val_zone": 7947,
      "n_val_fraud": 0,
      "val_fraud_rate": 0.0,
      "n_inc_vs_Zr": 0,
      "n_inc_fraud": 0
    },
    {
      "fold": 3,
      "n_val_zone": 7586,
      "n_val_fraud": 0,
      "val_fraud_rate": 0.0,
      "n_inc_vs_Zr": 0,
      "n_inc_fraud": 0
    },
    {
      "fold": 4,
      "n_val_zone": 7916,
      "n_val_fraud": 0,
      "val_fraud_rate": 0.0,
      "n_inc_vs_Zr": 0,
      "n_inc_fraud": 0
    }
  ],
  "cell_miner": {
    "20": {
      "folds": [
        {
          "fold": 0,
          "n_val_zone": 10439,
          "n_val_fraud": 1,
          "n_inc": 3387,
          "n_inc_fraud": 1,
          "n_test_zone": 21861
        },
        {
          "fold": 1,
          "n_val_zone": 10452,
          "n_val_fraud": 12,
          "n_inc": 3610,
          "n_inc_fraud": 12,
          "n_test_zone": 22515
        },
        {
          "fold": 2,
          "n_val_zone": 11202,
          "n_val_fraud": 3,
          "n_inc": 4217,
          "n_inc_fraud": 3,
          "n_test_zone": 23574
        },
        {
          "fold": 3,
          "n_val_zone": 11072,
          "n_val_fraud": 5,
          "n_inc": 4401,
          "n_inc_fraud": 5,
          "n_test_zone": 23801
        },
        {
          "fold": 4,
          "n_val_zone": 12034,
          "n_val_fraud": 7,
          "n_inc": 4241,
          "n_inc_fraud": 7,
          "n_test_zone": 25163
        }
      ],
      "sum_n_val_zone": 55199,
      "sum_n_val_fraud": 28,
      "sum_n_inc_vs_Zr": 19856,
      "sum_n_inc_fraud": 28,
      "inc_fraud_rate": 0.0014101531023368251,
      "test_majority_n": 21731,
      "test_inc_vs_Zr": 7034
    },
    "35": {
      "folds": [
        {
          "fold": 0,
          "n_val_zone": 10292,
          "n_val_fraud": 1,
          "n_inc": 3310,
          "n_inc_fraud": 1,
          "n_test_zone": 21547
        },
        {
          "fold": 1,
          "n_val_zone": 10333,
          "n_val_fraud": 12,
          "n_inc": 3560,
          "n_inc_fraud": 12,
          "n_test_zone": 22225
        },
        {
          "fold": 2,
          "n_val_zone": 11023,
          "n_val_fraud": 3,
          "n_inc": 4136,
          "n_inc_fraud": 3,
          "n_test_zone": 23216
        },
        {
          "fold": 3,
          "n_val_zone": 10941,
          "n_val_fraud": 4,
          "n_inc": 4336,
          "n_inc_fraud": 4,
          "n_test_zone": 23502
        },
        {
          "fold": 4,
          "n_val_zone": 11888,
          "n_val_fraud": 7,
          "n_inc": 4177,
          "n_inc_fraud": 7,
          "n_test_zone": 24888
        }
      ],
      "sum_n_val_zone": 54477,
      "sum_n_val_fraud": 27,
      "sum_n_inc_vs_Zr": 19519,
      "sum_n_inc_fraud": 27,
      "inc_fraud_rate": 0.0013832675854295814,
      "test_majority_n": 21409,
      "test_inc_vs_Zr": 6890
    },
    "50": {
      "folds": [
        {
          "fold": 0,
          "n_val_zone": 10242,
          "n_val_fraud": 1,
          "n_inc": 3292,
          "n_inc_fraud": 1,
          "n_test_zone": 21450
        },
        {
          "fold": 1,
          "n_val_zone": 10256,
          "n_val_fraud": 9,
          "n_inc": 3518,
          "n_inc_fraud": 9,
          "n_test_zone": 22060
        },
        {
          "fold": 2,
          "n_val_zone": 11001,
          "n_val_fraud": 3,
          "n_inc": 4128,
          "n_inc_fraud": 3,
          "n_test_zone": 23189
        },
        {
          "fold": 3,
          "n_val_zone": 10869,
          "n_val_fraud": 3,
          "n_inc": 4308,
          "n_inc_fraud": 3,
          "n_test_zone": 23315
        },
        {
          "fold": 4,
          "n_val_zone": 11826,
          "n_val_fraud": 6,
          "n_inc": 4150,
          "n_inc_fraud": 6,
          "n_test_zone": 24725
        }
      ],
      "sum_n_val_zone": 54194,
      "sum_n_val_fraud": 22,
      "sum_n_inc_vs_Zr": 19396,
      "sum_n_inc_fraud": 22,
      "inc_fraud_rate": 0.0011342544854609198,
      "test_majority_n": 21312,
      "test_inc_vs_Zr": 6862
    },
    "80": {
      "folds": [
        {
          "fold": 0,
          "n_val_zone": 10119,
          "n_val_fraud": 1,
          "n_inc": 3289,
          "n_inc_fraud": 1,
          "n_test_zone": 21206
        },
        {
          "fold": 1,
          "n_val_zone": 10105,
          "n_val_fraud": 5,
          "n_inc": 3478,
          "n_inc_fraud": 5,
          "n_test_zone": 21719
        },
        {
          "fold": 2,
          "n_val_zone": 10862,
          "n_val_fraud": 3,
          "n_inc": 4095,
          "n_inc_fraud": 3,
          "n_test_zone": 22871
        },
        {
          "fold": 3,
          "n_val_zone": 10785,
          "n_val_fraud": 3,
          "n_inc": 4308,
          "n_inc_fraud": 3,
          "n_test_zone": 23096
        },
        {
          "fold": 4,
          "n_val_zone": 11712,
          "n_val_fraud": 5,
          "n_inc": 4136,
          "n_inc_fraud": 5,
          "n_test_zone": 24453
        }
      ],
      "sum_n_val_zone": 53583,
      "sum_n_val_fraud": 17,
      "sum_n_inc_vs_Zr": 19306,
      "sum_n_inc_fraud": 17,
      "inc_fraud_rate": 0.0008805552677923961,
      "test_majority_n": 21068,
      "test_inc_vs_Zr": 6858
    },
    "120": {
      "folds": [
        {
          "fold": 0,
          "n_val_zone": 9945,
          "n_val_fraud": 1,
          "n_inc": 3187,
          "n_inc_fraud": 1,
          "n_test_zone": 20816
        },
        {
          "fold": 1,
          "n_val_zone": 9908,
          "n_val_fraud": 5,
          "n_inc": 3352,
          "n_inc_fraud": 5,
          "n_test_zone": 21329
        },
        {
          "fold": 2,
          "n_val_zone": 10664,
          "n_val_fraud": 3,
          "n_inc": 3986,
          "n_inc_fraud": 3,
          "n_test_zone": 22525
        },
        {
          "fold": 3,
          "n_val_zone": 10618,
          "n_val_fraud": 3,
          "n_inc": 4202,
          "n_inc_fraud": 3,
          "n_test_zone": 22706
        },
        {
          "fold": 4,
          "n_val_zone": 11532,
          "n_val_fraud": 5,
          "n_inc": 4022,
          "n_inc_fraud": 5,
          "n_test_zone": 24063
        }
      ],
      "sum_n_val_zone": 52667,
      "sum_n_val_fraud": 17,
      "sum_n_inc_vs_Zr": 18749,
      "sum_n_inc_fraud": 17,
      "inc_fraud_rate": 0.0009067150248013228,
      "test_majority_n": 20678,
      "test_inc_vs_Zr": 6619
    }
  },
  "kabartay_edge_refit_sum": {
    "n_val_zone": 39149,
    "n_val_fraud": 0,
    "n_inc_vs_Zr": 0,
    "n_inc_fraud": 0,
    "baseline": 0.016699525687182123
  }
}
```

## 3. Mechanisms (top 20 by pos5/meanΔ)

```json
[
  {
    "tag": "A1_Zr_goodpjw@fullbank",
    "pos5": 5,
    "mean_e5": 4.944202590735358,
    "min_e5": 3.395528216110222,
    "folds_e5": [
      5.087974349760049,
      4.812092375838795,
      4.588064571675599,
      3.395528216110222,
      6.8373534402921265
    ],
    "tstat5": 8.917981618127879,
    "sf25_pos": 25,
    "sf25_n": 25,
    "t25": 10.589670227749574,
    "wilcoxon_p_25": 2.9802322387695312e-08,
    "sf25_mean_e5": 4.99418056064771,
    "base": "fullbank"
  },
  {
    "tag": "A1_kabartay_oof@fullbank",
    "pos5": 5,
    "mean_e5": 4.944202590735358,
    "min_e5": 3.395528216110222,
    "folds_e5": [
      5.087974349760049,
      4.812092375838795,
      4.588064571675599,
      3.395528216110222,
      6.8373534402921265
    ],
    "tstat5": 8.917981618127879,
    "sf25_pos": 25,
    "sf25_n": 25,
    "t25": 10.589670227749574,
    "wilcoxon_p_25": 2.9802322387695312e-08,
    "sf25_mean_e5": 4.99418056064771,
    "base": "fullbank"
  },
  {
    "tag": "A1_union_Zr_U_kab@fullbank",
    "pos5": 5,
    "mean_e5": 4.944202590735358,
    "min_e5": 3.395528216110222,
    "folds_e5": [
      5.087974349760049,
      4.812092375838795,
      4.588064571675599,
      3.395528216110222,
      6.8373534402921265
    ],
    "tstat5": 8.917981618127879,
    "sf25_pos": 25,
    "sf25_n": 25,
    "t25": 10.589670227749574,
    "wilcoxon_p_25": 2.9802322387695312e-08,
    "sf25_mean_e5": 4.99418056064771,
    "base": "fullbank"
  },
  {
    "tag": "A1_Zr_goodpjw@part12",
    "pos5": 5,
    "mean_e5": 4.888140249077466,
    "min_e5": 3.386517389036836,
    "folds_e5": [
      4.983529128521624,
      4.726334159532719,
      4.569421481137592,
      3.386517389036836,
      6.774899087158559
    ],
    "tstat5": 8.960702921250427,
    "sf25_pos": 25,
    "sf25_n": 25,
    "t25": 10.716218313928136,
    "wilcoxon_p_25": 2.9802322387695312e-08,
    "sf25_mean_e5": 4.946897044707299,
    "base": "part12"
  },
  {
    "tag": "A1_kabartay_oof@part12",
    "pos5": 5,
    "mean_e5": 4.888140249077466,
    "min_e5": 3.386517389036836,
    "folds_e5": [
      4.983529128521624,
      4.726334159532719,
      4.569421481137592,
      3.386517389036836,
      6.774899087158559
    ],
    "tstat5": 8.960702921250427,
    "sf25_pos": 25,
    "sf25_n": 25,
    "t25": 10.716218313928136,
    "wilcoxon_p_25": 2.9802322387695312e-08,
    "sf25_mean_e5": 4.946897044707299,
    "base": "part12"
  },
  {
    "tag": "A1_union_Zr_U_kab@part12",
    "pos5": 5,
    "mean_e5": 4.888140249077466,
    "min_e5": 3.386517389036836,
    "folds_e5": [
      4.983529128521624,
      4.726334159532719,
      4.569421481137592,
      3.386517389036836,
      6.774899087158559
    ],
    "tstat5": 8.960702921250427,
    "sf25_pos": 25,
    "sf25_n": 25,
    "t25": 10.716218313928136,
    "wilcoxon_p_25": 2.9802322387695312e-08,
    "sf25_mean_e5": 4.946897044707299,
    "base": "part12"
  },
  {
    "tag": "B_EB_lofo_s-1_a0.02@part12",
    "pos5": 3,
    "mean_e5": -0.30456586044724787,
    "min_e5": -3.204747256690954,
    "folds_e5": [
      0.7401311661214827,
      -3.204747256690954,
      2.8756967093479524,
      -2.2001953969952304,
      0.26628547598051
    ],
    "tstat5": -0.280757759003291,
    "sf25_pos": 13,
    "sf25_n": 25,
    "t25": -0.2795999880587737,
    "wilcoxon_p_25": 0.5940348207950592,
    "sf25_mean_e5": -0.2718291416901053,
    "base": "part12",
    "sign": -1,
    "alpha": 0.02
  },
  {
    "tag": "B_EB_lofo_s-1_a0.02@fullbank",
    "pos5": 3,
    "mean_e5": -0.5180389444259248,
    "min_e5": -3.626391820099606,
    "folds_e5": [
      0.6248062343061633,
      -3.626391820099606,
      2.680565695512005,
      -2.2710391408997666,
      0.001864309051580193
    ],
    "tstat5": -0.4676484309932176,
    "sf25_pos": 11,
    "sf25_n": 25,
    "t25": -0.5483614276715923,
    "wilcoxon_p_25": 0.6542405784130096,
    "sf25_mean_e5": -0.5097983872976997,
    "base": "fullbank",
    "sign": -1,
    "alpha": 0.02
  },
  {
    "tag": "A2_soft_lam0.25_inc_on_Zr@fullbank",
    "pos5": 2,
    "mean_e5": -0.1636516968628321,
    "min_e5": -3.5567909489331306,
    "folds_e5": [
      4.10693530722428,
      -2.3714011114006617,
      -1.3447882613037976,
      2.3477865300991496,
      -3.5567909489331306
    ],
    "tstat5": -0.1124921717693822,
    "sf25_pos": 16,
    "sf25_n": 25,
    "t25": -0.1577199453730966,
    "wilcoxon_p_25": 0.2814549358317319,
    "sf25_mean_e5": -0.23492834195115722,
    "base": "fullbank",
    "lambda": 0.25,
    "n_inc_oof": 19306
  },
  {
    "tag": "A2_soft_lam0.25_inc_on_Zr@part12",
    "pos5": 2,
    "mean_e5": -0.18229985843998264,
    "min_e5": -3.519815486130451,
    "folds_e5": [
      4.04663062589794,
      -2.3117432218056067,
      -1.2913447351881047,
      2.164773525026309,
      -3.519815486130451
    ],
    "tstat5": -0.12843094491382573,
    "sf25_pos": 16,
    "sf25_n": 25,
    "t25": -0.17527504370940608,
    "wilcoxon_p_25": 0.3169134259223938,
    "sf25_mean_e5": -0.2614386584292028,
    "base": "part12",
    "lambda": 0.25,
    "n_inc_oof": 19306
  },
  {
    "tag": "B_EB_lofo_s1_a0.02@fullbank",
    "pos5": 2,
    "mean_e5": -1.3990859965407054,
    "min_e5": -4.453523601910536,
    "folds_e5": [
      -2.7681092121478734,
      1.4215356505031806,
      -4.453523601910536,
      0.5916074051315334,
      -1.7869402242798316
    ],
    "tstat5": -1.2969795184920927,
    "sf25_pos": 9,
    "sf25_n": 25,
    "t25": -1.3412282326654044,
    "wilcoxon_p_25": 0.890006810426712,
    "sf25_mean_e5": -1.3425121509573756,
    "base": "fullbank",
    "sign": 1,
    "alpha": 0.02
  },
  {
    "tag": "B_EB_lofo_s1_a0.02@part12",
    "pos5": 2,
    "mean_e5": -1.5191524919999644,
    "min_e5": -4.452280729205782,
    "folds_e5": [
      -2.8274813468454063,
      1.4075533326218803,
      -4.452280729205782,
      0.2836856937471488,
      -2.0072394103176627
    ],
    "tstat5": -1.4364576329712992,
    "sf25_pos": 9,
    "sf25_n": 25,
    "t25": -1.4875811813086735,
    "wilcoxon_p_25": 0.9400331079959869,
    "sf25_mean_e5": -1.495074164202137,
    "base": "part12",
    "sign": 1,
    "alpha": 0.02
  },
  {
    "tag": "A2_soft_lam0.5_inc_on_Zr@fullbank",
    "pos5": 2,
    "mean_e5": -1.7053201299610519,
    "min_e5": -8.439105632285226,
    "folds_e5": [
      6.954310981943657,
      -6.79882438661128,
      -2.777199047920842,
      2.534217435068431,
      -8.439105632285226
    ],
    "tstat5": -0.5929988670149805,
    "sf25_pos": 15,
    "sf25_n": 25,
    "t25": -0.5846409055371034,
    "wilcoxon_p_25": 0.43699392676353455,
    "sf25_mean_e5": -1.7518109501266643,
    "base": "fullbank",
    "lambda": 0.5,
    "n_inc_oof": 19306
  },
  {
    "tag": "A2_soft_lam0.5_inc_on_Zr@part12",
    "pos5": 2,
    "mean_e5": -1.722728189625933,
    "min_e5": -8.250188981917539,
    "folds_e5": [
      6.86105632008438,
      -6.764645387358215,
      -2.6581939868930426,
      2.1983310879547524,
      -8.250188981917539
    ],
    "tstat5": -0.6129971092248929,
    "sf25_pos": 15,
    "sf25_n": 25,
    "t25": -0.5969231829895127,
    "wilcoxon_p_25": 0.44743022322654724,
    "sf25_mean_e5": -1.7739602238830976,
    "base": "part12",
    "lambda": 0.5,
    "n_inc_oof": 19306
  },
  {
    "tag": "B_EB_lofo_s-1_a0.05@part12",
    "pos5": 1,
    "mean_e5": -4.280029793215512,
    "min_e5": -12.003353817402473,
    "folds_e5": [
      -1.6114405565037515,
      -12.003353817402473,
      3.867198405693717,
      -8.695448126649907,
      -2.9571048712151438
    ],
    "tstat5": -1.5404439699337467,
    "sf25_pos": 10,
    "sf25_n": 25,
    "t25": -1.7609086218511312,
    "wilcoxon_p_25": 0.9331799745559692,
    "sf25_mean_e5": -4.2397589790392765,
    "base": "part12",
    "sign": -1,
    "alpha": 0.05
  },
  {
    "tag": "B_EB_lofo_s-1_a0.05@fullbank",
    "pos5": 1,
    "mean_e5": -4.631362658145122,
    "min_e5": -12.454516607451893,
    "folds_e5": [
      -2.0264238016709513,
      -12.454516607451893,
      3.405471197726051,
      -8.716576962541911,
      -3.364767116786904
    ],
    "tstat5": -1.6862437467295572,
    "sf25_pos": 10,
    "sf25_n": 25,
    "t25": -1.9449532038668393,
    "wilcoxon_p_25": 0.9574304819107056,
    "sf25_mean_e5": -4.569739126053296,
    "base": "fullbank",
    "sign": -1,
    "alpha": 0.05
  },
  {
    "tag": "A2_soft_lam1.0_inc_on_Zr@fullbank",
    "pos5": 1,
    "mean_e5": -7.675219862348647,
    "min_e5": -20.65530139956362,
    "folds_e5": [
      9.799821563416522,
      -20.65530139956362,
      -5.4201678442500345,
      -1.531219166273079,
      -20.56923246507303
    ],
    "tstat5": -1.3134442298301536,
    "sf25_pos": 15,
    "sf25_n": 25,
    "t25": -1.300227680286884,
    "wilcoxon_p_25": 0.7199142575263977,
    "sf25_mean_e5": -7.813738606087473,
    "base": "fullbank",
    "lambda": 1.0,
    "n_inc_oof": 19306
  },
  {
    "tag": "A2_soft_lam1.0_inc_on_Zr@part12",
    "pos5": 1,
    "mean_e5": -7.707358786579466,
    "min_e5": -20.60185787347013,
    "folds_e5": [
      9.67299522331988,
      -20.60185787347013,
      -5.152950213760388,
      -2.2489781504542172,
      -20.206002918532473
    ],
    "tstat5": -1.3407855546367058,
    "sf25_pos": 15,
    "sf25_n": 25,
    "t25": -1.3169567839270249,
    "wilcoxon_p_25": 0.7199142575263977,
    "sf25_mean_e5": -7.851825251273592,
    "base": "part12",
    "lambda": 1.0,
    "n_inc_oof": 19306
  },
  {
    "tag": "A2_soft_lam2.0_inc_on_Zr@fullbank",
    "pos5": 1,
    "mean_e5": -25.676938055341214,
    "min_e5": -58.597097745605176,
    "folds_e5": [
      7.438302676854303,
      -58.597097745605176,
      -9.312845140307946,
      -20.36633349682404,
      -47.54671657082321
    ],
    "tstat5": -2.112548689270269,
    "sf25_pos": 13,
    "sf25_n": 25,
    "t25": -2.1523191114004305,
    "wilcoxon_p_25": 0.890006810426712,
    "sf25_mean_e5": -25.900143941964604,
    "base": "fullbank",
    "lambda": 2.0,
    "n_inc_oof": 19306
  },
  {
    "tag": "A2_soft_lam2.0_inc_on_Zr@part12",
    "pos5": 1,
    "mean_e5": -25.76501103461748,
    "min_e5": -58.46721754845241,
    "folds_e5": [
      7.25459099302217,
      -58.46721754845241,
      -8.852050086827212,
      -21.991078833771827,
      -46.769299697058116
    ],
    "tstat5": -2.139225379592414,
    "sf25_pos": 13,
    "sf25_n": 25,
    "t25": -2.1768508338099237,
    "wilcoxon_p_25": 0.9218308925628662,
    "sf25_mean_e5": -25.970735570281622,
    "base": "part12",
    "lambda": 2.0,
    "n_inc_oof": 19306
  }
]
```

### Marginal gates

```json
{
  "kabartay_on_Zr_fullbank": {
    "tag": "A1_marginal_kab_on_Zr@fullbank",
    "pos5": 0,
    "mean_e5": 0.0,
    "min_e5": 0.0,
    "folds_e5": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "tstat5": 0.0,
    "sf25_pos": 0,
    "sf25_n": 25,
    "t25": 0.0,
    "wilcoxon_p_25": 1.0,
    "sf25_mean_e5": 0.0,
    "base": "fullbank",
    "note": "\u0394 vs Zr-already-applied"
  },
  "mine_on_Zr_fullbank": {
    "tag": "A1_marginal_mine80_on_Zr@fullbank",
    "pos5": 0,
    "mean_e5": -66.76022439406904,
    "min_e5": -138.66854999996514,
    "folds_e5": [
      -20.70284577636583,
      -138.66854999996514,
      -6.746934451384856,
      -83.03011071646438,
      -84.65268102616497
    ],
    "tstat5": -2.786695914130441,
    "sf25_pos": 13,
    "sf25_n": 25,
    "t25": -2.772167334188475,
    "wilcoxon_p_25": 0.9331799745559692,
    "sf25_mean_e5": -66.92441570575579,
    "base": "fullbank"
  },
  "mine_on_Zr_part12": {
    "tag": "A1_marginal_mine80_on_Zr@part12",
    "pos5": 0,
    "mean_e5": -66.61675319540316,
    "min_e5": -138.94415702117024,
    "folds_e5": [
      -20.92230508056181,
      -138.94415702117024,
      -6.092561974868627,
      -82.6522774157068,
      -84.47246448470835
    ],
    "tstat5": -2.771086042804279,
    "sf25_pos": 13,
    "sf25_n": 25,
    "t25": -2.7668645970168395,
    "wilcoxon_p_25": 0.9331799745559692,
    "sf25_mean_e5": -66.79970918395783,
    "base": "part12"
  }
}
```

## 4. Disagreement vs Step_AB + staging

```json
{
  "disagree": {
    "kabartay_pub_minus_Zr": {
      "n_rows": 0
    },
    "mined_minus_Zr": {
      "n_rows": 6858,
      "min": 0.05582464398101232,
      "median": 0.17409595178409515,
      "p90": 0.5192676942770281,
      "p99": 0.8475415688836744,
      "max": 0.983284108485786,
      "n_gt_010": 5287,
      "n_gt_050": 742,
      "n_gt_080": 125
    },
    "n_min_selected": 80
  },
  "LIVE_STAGE": "SKIP (keep for Step 5 own-stack only)",
  "LIVE_REASON": "Z_D27_kabartay \u2261 Z_goodpjw on test (inc=0). Mined N_min=80 test_inc=6858 OOF marginal pos5=0 mean_e5=-66.760 \u2014 not clearly positive; LIVE_STAGE=SKIP.",
  "staged": {},
  "eb_hygiene": {
    "oof_path": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/v41_d27_eb_zone_oof.npy",
    "test_path": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/v41_d27_eb_zone_test.npy",
    "nan_inf": 0,
    "raw_disp": 0.9957408180910481,
    "KS_raw": 0.11190490942037823,
    "pass_disp": true,
    "pass_ks": false,
    "oof_std": 1.9043489615648297,
    "te_std": 1.8962379929203972
  }
}
```

## 5. STOP

Step 4 complete. **Awaiting human audit.** Reply `PROCEED TO STEP 5` to continue.
