# V41 STEP5 AUDIT — Nested Dual Stack × Part12 + Zr

**NO_SUBMIT=True** · Hard stop. Do not run Step 6 without `PROCEED TO STEP 6`.

## Executive findings

- Kept columns after hygiene+dedup: **40**
- Inventory by family: `{"LM_GPT2": {"before_dedup": 30, "after_dedup": 7}, "GBDT": {"before_dedup": 42, "after_dedup": 11}, "TabM_MLP": {"before_dedup": 16, "after_dedup": 15}, "EBM_GLM": {"before_dedup": 36, "after_dedup": 5}, "Proprietary": {"before_dedup": 2, "after_dedup": 2}, "Other": {"before_dedup": 0, "after_dedup": 0}}`
- Nested winner: **Meta_Ens_L2_Cal** OOF=0.952011 → +Zr **0.952059**
- Part12 OOF=0.952565 → +Zr **0.952614**
- Divergence `1−ρ×1e−4`: OOF=49.275, test=102.099, ratio=2.07
- Nested `w*` mean Part12 weight=1.000; dual+Zr OOF=**0.952614**
- Staged: N1/N2/N3 + H1/H2 under `out/v41_staged/`

## 1. Inventory & quarantine

```json
{
  "by_family": {
    "LM_GPT2": {
      "before_dedup": 30,
      "after_dedup": 7
    },
    "GBDT": {
      "before_dedup": 42,
      "after_dedup": 11
    },
    "TabM_MLP": {
      "before_dedup": 16,
      "after_dedup": 15
    },
    "EBM_GLM": {
      "before_dedup": 36,
      "after_dedup": 5
    },
    "Proprietary": {
      "before_dedup": 2,
      "after_dedup": 2
    },
    "Other": {
      "before_dedup": 0,
      "after_dedup": 0
    }
  },
  "dedup_dropped_n": 100,
  "quarantine_or_fail_n": 7
}
```

### Kept members (name / family / auc / KS / disp)

| name | family | auc | KS | disp |
|---|---|---:|---:|---:|
| `gpw70:gb_bag5` | EBM_GLM | 0.95151 | 0.0004 | 1.000 |
| `gpw70:ebm_i25_bag` | EBM_GLM | 0.95026 | 0.0098 | 1.005 |
| `gpw70:lgb_dig_l7` | GBDT | 0.95016 | 0.0190 | 1.006 |
| `gpw70:glm_ohk` | LM_GPT2 | 0.94924 | 0.0037 | 1.001 |
| `gpw70:lgb_dk_l4` | GBDT | 0.94901 | 0.0048 | 0.992 |
| `gpw70:lgb_te_l4_s100` | GBDT | 0.94885 | 0.0048 | 0.993 |
| `gpw70:xgb_te_d2` | GBDT | 0.94881 | 0.0045 | 0.993 |
| `gpw70:realmlp_dig_s2` | TabM_MLP | 0.94867 | 0.0093 | 0.991 |
| `gpw70:realmlp_dig_s100` | TabM_MLP | 0.94858 | 0.0126 | 0.992 |
| `gpw70:realmlp_dig` | TabM_MLP | 0.94853 | 0.0102 | 0.991 |
| `gpw70:lgb_rawdig_l4` | GBDT | 0.94837 | 0.0131 | 1.005 |
| `gpw70:lgb_raw_l4_cnt` | GBDT | 0.94778 | 0.0023 | 0.998 |
| `gpw70:lgb_raw_l4` | GBDT | 0.94777 | 0.0027 | 0.997 |
| `gpw70:realmlp_te` | TabM_MLP | 0.94771 | 0.0065 | 0.993 |
| `gpw70:realmlp_te_s2` | TabM_MLP | 0.94762 | 0.0066 | 0.994 |
| `gpw70:realmlp_te_cnt` | TabM_MLP | 0.94760 | 0.0072 | 0.994 |
| `gpw70:xgb_raw_d3` | GBDT | 0.94749 | 0.0031 | 0.996 |
| `gpw70:lgb_raw_l31` | GBDT | 0.94706 | 0.0040 | 0.995 |
| `gpw70:glm_te` | LM_GPT2 | 0.94706 | 0.0040 | 0.996 |
| `gpw70:realmlp_cnt` | TabM_MLP | 0.94664 | 0.0039 | 0.996 |
| `gpw70:realmlp_allcat` | TabM_MLP | 0.94640 | 0.0093 | 0.994 |
| `gpw70:realmlp_rawdig` | TabM_MLP | 0.94615 | 0.0037 | 0.995 |
| `gpw70:glm_sb` | LM_GPT2 | 0.94586 | 0.0018 | 0.999 |
| `gpw70:realmlp_v1` | TabM_MLP | 0.94584 | 0.0031 | 0.996 |
| `busy52:lgb_g2_plaininit_l4` | EBM_GLM | 0.95109 | 0.0031 | 0.999 |
| `busy52:lgb_dl3_g2c3` | GBDT | 0.95072 | 0.0141 | 1.006 |
| `busy52:glm_plain_g2_s50` | LM_GPT2 | 0.95062 | 0.0033 | 1.000 |
| `busy52:ebm_g2_glminit` | EBM_GLM | 0.95059 | 0.0045 | 0.999 |
| `busy52:glm_ohk_g2_c0003_s100` | LM_GPT2 | 0.95032 | 0.0083 | 1.002 |
| `busy52:lgb_dl2_hier` | GBDT | 0.95023 | 0.0116 | 1.002 |
| `busy52:tabm_dl` | TabM_MLP | 0.94970 | 0.0087 | 0.992 |
| `busy52:rmlp2_dl_hier` | TabM_MLP | 0.94967 | 0.0286 | 0.984 |
| `busy52:glm_dl` | LM_GPT2 | 0.94887 | 0.0025 | 1.000 |
| `busy52:rmlp_dl` | TabM_MLP | 0.94856 | 0.0120 | 0.990 |
| `busy52:slr_c003` | EBM_GLM | 0.94853 | 0.0069 | 1.003 |
| `busy52:rmlp_dl_g2` | TabM_MLP | 0.94839 | 0.0204 | 0.999 |
| `busy52:tabm_standalone` | TabM_MLP | 0.93731 | 0.0326 | 0.980 |
| `prop:uid_v2` | Proprietary | 0.93485 | 0.0219 | 0.997 |
| `prop:xgb_llr` | Proprietary | 0.95195 | 0.0011 | 1.002 |
| `prop:l7_s2026` | LM_GPT2 | 0.94927 | 0.0031 | 0.996 |

## 2. Nested meta-stackers

```json
{
  "Meta_L2_LogReg": {
    "oof_auc": 0.9519692849777965,
    "oof_auc_zr": 0.9520158553567911,
    "folds": [
      0.9508618992179152,
      0.9522739368985825,
      0.9569400849476147,
      0.9525068388138157,
      0.9474487508588717
    ],
    "folds_zr": [
      0.9509078799831775,
      0.9523171640110849,
      0.9569800992341879,
      0.9525409960627905,
      0.9475211357649135
    ],
    "sf25_vs_part12": 3,
    "sf25_vs_fullbank": 4
  },
  "Meta_Cal_LogOdds": {
    "oof_auc": 0.9519828832729068,
    "oof_auc_zr": 0.952032128041693,
    "folds": [
      0.950858601111375,
      0.9522671166346417,
      0.956977253055705,
      0.9523682709365044,
      0.9476644762733923
    ],
    "folds_zr": [
      0.9509083400395092,
      0.9523104493913236,
      0.9570219343292663,
      0.9524059796942194,
      0.9477368301076167
    ],
    "sf25_vs_part12": 3,
    "sf25_vs_fullbank": 4
  },
  "Meta_Simplex_Ridge": {
    "oof_auc": 0.9508955487994892,
    "oof_auc_zr": 0.9509611593521711,
    "folds": [
      0.9497938349249805,
      0.9517979663694306,
      0.9558858647876552,
      0.9513442960836971,
      0.9456691187260732
    ],
    "folds_zr": [
      0.9498605617440127,
      0.951857335291123,
      0.9559505065967767,
      0.9513927587974477,
      0.9457604543336068
    ],
    "sf25_vs_part12": 2,
    "sf25_vs_fullbank": 2
  },
  "Meta_Ens_L2_Cal": {
    "oof_auc": 0.9520113324112743,
    "oof_auc_zr": 0.9520590674599261,
    "folds": [
      0.9508927261006291,
      0.9523012925267073,
      0.9569902970046905,
      0.9524760000349497,
      0.9475927096965185
    ],
    "folds_zr": [
      0.9509403170630525,
      0.9523445444966635,
      0.9570323496024914,
      0.9525118475907963,
      0.9476647652412951
    ],
    "sf25_vs_part12": 3,
    "sf25_vs_fullbank": 4
  }
}
```

### Weight attribution (per outer fold, top coefs)

```json
[
  {
    "fold": 0,
    "C_L2": 0.03,
    "inner_L2": 0.9522978878323048,
    "C_Cal": 0.01,
    "inner_Cal": 0.9522958168956687,
    "alpha_Sim": 1000.0,
    "inner_Sim": 0.9497479441735885,
    "family_abs_weight": {
      "EBM_GLM": 0.6823173853268599,
      "GBDT": 0.7432170381340404,
      "LM_GPT2": 0.7436322467881847,
      "Proprietary": 0.895945417453548,
      "TabM_MLP": 0.8299116481580662
    },
    "named_weights": {
      "tabm": {
        "busy52:tabm_dl": -0.13094795779670804,
        "busy52:tabm_standalone": 0.09857440023135881
      },
      "uid_v2": {
        "prop:uid_v2": 0.015290388791965866
      },
      "realmlp": {
        "gpw70:realmlp_dig_s2": -0.026550966197704022,
        "gpw70:realmlp_dig_s100": -0.011342747978862342,
        "gpw70:realmlp_dig": -0.047569938376399126,
        "gpw70:realmlp_te": 0.039841398739645144,
        "gpw70:realmlp_te_s2": -0.08495316480982604,
        "gpw70:realmlp_te_cnt": -0.0490282319170017,
        "gpw70:realmlp_cnt": 0.045864781147667,
        "gpw70:realmlp_allcat": 0.04554070238753828,
        "gpw70:realmlp_rawdig": 0.017573030844937148,
        "gpw70:realmlp_v1": -0.07031013692022446
      },
      "rmlp": {
        "busy52:rmlp2_dl_hier": 0.142192913830946,
        "busy52:rmlp_dl": 0.019179829644722927,
        "busy52:rmlp_dl_g2": -0.0004414473345251158
      },
      "ebm": {
        "gpw70:ebm_i25_bag": 0.12687310550535894,
        "busy52:ebm_g2_glminit": 0.05532023653823274
      },
      "xgb_llr": {
        "prop:xgb_llr": 0.8806550286615822
      },
      "l7": {
        "gpw70:lgb_dig_l7": 0.051309216915038834,
        "prop:l7_s2026": 0.27551860468420003
      },
      "gpt2": {}
    },
    "top_abs_coefs": [
      {
        "name": "prop:xgb_llr",
        "family": "Proprietary",
        "coef": 0.8806550286615822
      },
      {
        "name": "gpw70:gb_bag5",
        "family": "EBM_GLM",
        "coef": 0.3148635217399572
      },
      {
        "name": "prop:l7_s2026",
        "family": "LM_GPT2",
        "coef": 0.27551860468420003
      },
      {
        "name": "busy52:lgb_dl2_hier",
        "family": "GBDT",
        "coef": -0.23349844517945675
      },
      {
        "name": "busy52:lgb_g2_plaininit_l4",
        "family": "EBM_GLM",
        "coef": 0.15686861813838016
      },
      {
        "name": "busy52:rmlp2_dl_hier",
        "family": "TabM_MLP",
        "coef": 0.142192913830946
      },
      {
        "name": "gpw70:lgb_dk_l4",
        "family": "GBDT",
        "coef": -0.1387478231343368
      },
      {
        "name": "busy52:tabm_dl",
        "family": "TabM_MLP",
        "coef": -0.13094795779670804
      },
      {
        "name": "gpw70:ebm_i25_bag",
        "family": "EBM_GLM",
        "coef": 0.12687310550535894
      },
      {
        "name": "busy52:glm_dl",
        "family": "LM_GPT2",
        "coef": -0.12147822564346598
      },
      {
        "name": "gpw70:glm_ohk",
        "family": "LM_GPT2",
        "coef": -0.10550065779615127
      },
      {
        "name": "busy52:glm_ohk_g2_c0003_s100",
        "family": "LM_GPT2",
        "coef": -0.10466150236352252
      },
      {
        "name": "busy52:tabm_standalone",
        "family": "TabM_MLP",
        "coef": 0.09857440023135881
      },
      {
        "name": "gpw70:xgb_te_d2",
        "family": "GBDT",
        "coef": 0.09167139147731788
      },
      {
        "name": "gpw70:lgb_rawdig_l4",
        "family": "GBDT",
        "coef": 0.08850023242062553
      }
    ]
  },
  {
    "fold": 1,
    "C_L2": 0.03,
    "inner_L2": 0.9519481714179044,
    "C_Cal": 0.01,
    "inner_Cal": 0.9519519677294486,
    "alpha_Sim": 1000.0,
    "inner_Sim": 0.9493104467637781,
    "family_abs_weight": {
      "EBM_GLM": 0.6451508101788853,
      "GBDT": 0.6078606226781822,
      "LM_GPT2": 0.7099653442456184,
      "Proprietary": 0.713613957669962,
      "TabM_MLP": 0.6430560802957691
    },
    "named_weights": {
      "tabm": {
        "busy52:tabm_dl": -0.038511226792940716,
        "busy52:tabm_standalone": 0.08880628281342436
      },
      "uid_v2": {
        "prop:uid_v2": -0.012445865162407293
      },
      "realmlp": {
        "gpw70:realmlp_dig_s2": -0.06440210670315512,
        "gpw70:realmlp_dig_s100": -0.02709481138376193,
        "gpw70:realmlp_dig": -0.06239684730803988,
        "gpw70:realmlp_te": 0.027991473078229114,
        "gpw70:realmlp_te_s2": -0.01861981879194085,
        "gpw70:realmlp_te_cnt": 0.038914737471386705,
        "gpw70:realmlp_cnt": 0.036569699018287785,
        "gpw70:realmlp_allcat": -0.04590143577386632,
        "gpw70:realmlp_rawdig": 0.015928940688398206,
        "gpw70:realmlp_v1": -0.05588941144785545
      },
      "rmlp": {
        "busy52:rmlp2_dl_hier": 0.04221307060336925,
        "busy52:rmlp_dl": -0.03273261964665669,
        "busy52:rmlp_dl_g2": -0.0470835987744567
      },
      "ebm": {
        "gpw70:ebm_i25_bag": 0.09380163804874794,
        "busy52:ebm_g2_glminit": 0.07526493931626861
      },
      "xgb_llr": {
        "prop:xgb_llr": 0.7011680925075547
      },
      "l7": {
        "gpw70:lgb_dig_l7": 0.07501811568059637,
        "prop:l7_s2026": 0.36115692779774405
      },
      "gpt2": {}
    },
    "top_abs_coefs": [
      {
        "name": "prop:xgb_llr",
        "family": "Proprietary",
        "coef": 0.7011680925075547
      },
      {
        "name": "prop:l7_s2026",
        "family": "LM_GPT2",
        "coef": 0.36115692779774405
      },
      {
        "name": "gpw70:gb_bag5",
        "family": "EBM_GLM",
        "coef": 0.30922215473297665
      },
      {
        "name": "busy52:lgb_g2_plaininit_l4",
        "family": "EBM_GLM",
        "coef": 0.14210633604972484
      },
      {
        "name": "busy52:lgb_dl2_hier",
        "family": "GBDT",
        "coef": -0.11555876977575374
      },
      {
        "name": "busy52:glm_dl",
        "family": "LM_GPT2",
        "coef": -0.09557161072875114
      },
      {
        "name": "gpw70:glm_sb",
        "family": "LM_GPT2",
        "coef": -0.09416952892826949
      },
      {
        "name": "gpw70:ebm_i25_bag",
        "family": "EBM_GLM",
        "coef": 0.09380163804874794
      },
      {
        "name": "busy52:tabm_standalone",
        "family": "TabM_MLP",
        "coef": 0.08880628281342436
      },
      {
        "name": "gpw70:lgb_raw_l4_cnt",
        "family": "GBDT",
        "coef": 0.08486656786439013
      },
      {
        "name": "busy52:ebm_g2_glminit",
        "family": "EBM_GLM",
        "coef": 0.07526493931626861
      },
      {
        "name": "gpw70:lgb_dig_l7",
        "family": "GBDT",
        "coef": 0.07501811568059637
      },
      {
        "name": "gpw70:lgb_rawdig_l4",
        "family": "GBDT",
        "coef": 0.07272236612121531
      },
      {
        "name": "gpw70:realmlp_dig_s2",
        "family": "TabM_MLP",
        "coef": -0.06440210670315512
      },
      {
        "name": "gpw70:realmlp_dig",
        "family": "TabM_MLP",
        "coef": -0.06239684730803988
      }
    ]
  },
  {
    "fold": 2,
    "C_L2": 0.03,
    "inner_L2": 0.9507871970152028,
    "C_Cal": 0.01,
    "inner_Cal": 0.9508051654730411,
    "alpha_Sim": 1000.0,
    "inner_Sim": 0.948058750407403,
    "family_abs_weight": {
      "EBM_GLM": 0.6711390999642687,
      "GBDT": 0.6048876859271519,
      "LM_GPT2": 0.6968100982318902,
      "Proprietary": 0.7526422026124127,
      "TabM_MLP": 0.6972815442777776
    },
    "named_weights": {
      "tabm": {
        "busy52:tabm_dl": -0.043967873933900795,
        "busy52:tabm_standalone": 0.09341352773102532
      },
      "uid_v2": {
        "prop:uid_v2": -0.0144934689821712
      },
      "realmlp": {
        "gpw70:realmlp_dig_s2": -0.06283450739075297,
        "gpw70:realmlp_dig_s100": -0.049214366429593454,
        "gpw70:realmlp_dig": -0.08252653795062245,
        "gpw70:realmlp_te": 0.031071093089574016,
        "gpw70:realmlp_te_s2": -0.04257066425820773,
        "gpw70:realmlp_te_cnt": -0.00948454417088279,
        "gpw70:realmlp_cnt": 0.10143357593022315,
        "gpw70:realmlp_allcat": -0.009574518686348833,
        "gpw70:realmlp_rawdig": 0.0037979003340762865,
        "gpw70:realmlp_v1": -0.030369255773685933
      },
      "rmlp": {
        "busy52:rmlp2_dl_hier": 0.06113466177326172,
        "busy52:rmlp_dl": -0.0644138339952612,
        "busy52:rmlp_dl_g2": -0.011474682830360884
      },
      "ebm": {
        "gpw70:ebm_i25_bag": 0.11369330606785812,
        "busy52:ebm_g2_glminit": 0.06431996420824272
      },
      "xgb_llr": {
        "prop:xgb_llr": 0.7381487336302415
      },
      "l7": {
        "gpw70:lgb_dig_l7": 0.04783634976676735,
        "prop:l7_s2026": 0.3353749311079268
      },
      "gpt2": {}
    },
    "top_abs_coefs": [
      {
        "name": "prop:xgb_llr",
        "family": "Proprietary",
        "coef": 0.7381487336302415
      },
      {
        "name": "gpw70:gb_bag5",
        "family": "EBM_GLM",
        "coef": 0.3421525056070076
      },
      {
        "name": "prop:l7_s2026",
        "family": "LM_GPT2",
        "coef": 0.3353749311079268
      },
      {
        "name": "busy52:lgb_g2_plaininit_l4",
        "family": "EBM_GLM",
        "coef": 0.12226678372606431
      },
      {
        "name": "busy52:lgb_dl2_hier",
        "family": "GBDT",
        "coef": -0.11838726675375388
      },
      {
        "name": "gpw70:ebm_i25_bag",
        "family": "EBM_GLM",
        "coef": 0.11369330606785812
      },
      {
        "name": "gpw70:realmlp_cnt",
        "family": "TabM_MLP",
        "coef": 0.10143357593022315
      },
      {
        "name": "busy52:glm_dl",
        "family": "LM_GPT2",
        "coef": -0.09728840110553374
      },
      {
        "name": "gpw70:lgb_dk_l4",
        "family": "GBDT",
        "coef": -0.09568241526676347
      },
      {
        "name": "busy52:tabm_standalone",
        "family": "TabM_MLP",
        "coef": 0.09341352773102532
      },
      {
        "name": "gpw70:realmlp_dig",
        "family": "TabM_MLP",
        "coef": -0.08252653795062245
      },
      {
        "name": "busy52:glm_ohk_g2_c0003_s100",
        "family": "LM_GPT2",
        "coef": -0.07355591791041387
      },
      {
        "name": "gpw70:glm_sb",
        "family": "LM_GPT2",
        "coef": -0.07071862803396807
      },
      {
        "name": "busy52:lgb_dl3_g2c3",
        "family": "GBDT",
        "coef": 0.06930770233524056
      },
      {
        "name": "gpw70:xgb_te_d2",
        "family": "GBDT",
        "coef": 0.064909073745136
      }
    ]
  },
  {
    "fold": 3,
    "C_L2": 0.03,
    "inner_L2": 0.9518765752316316,
    "C_Cal": 0.03,
    "inner_Cal": 0.9519171224045634,
    "alpha_Sim": 1000.0,
    "inner_Sim": 0.9493266786749791,
    "family_abs_weight": {
      "EBM_GLM": 0.5897625959197087,
      "GBDT": 0.6905439021819244,
      "LM_GPT2": 0.7160665488114349,
      "Proprietary": 0.7298111835002784,
      "TabM_MLP": 0.6269565989221504
    },
    "named_weights": {
      "tabm": {
        "busy52:tabm_dl": -0.054165082539157396,
        "busy52:tabm_standalone": 0.09011790943638472
      },
      "uid_v2": {
        "prop:uid_v2": -0.02153804336400105
      },
      "realmlp": {
        "gpw70:realmlp_dig_s2": -0.05431507744714725,
        "gpw70:realmlp_dig_s100": -0.04377926825278645,
        "gpw70:realmlp_dig": -0.0645547070905269,
        "gpw70:realmlp_te": 0.04077594124964691,
        "gpw70:realmlp_te_s2": -0.014567679104667039,
        "gpw70:realmlp_te_cnt": 0.021348336221209966,
        "gpw70:realmlp_cnt": 0.03565329200434767,
        "gpw70:realmlp_allcat": -0.06890692887078582,
        "gpw70:realmlp_rawdig": 0.021709827758510852,
        "gpw70:realmlp_v1": -0.04728627968071019
      },
      "rmlp": {
        "busy52:rmlp2_dl_hier": 0.05188416186649858,
        "busy52:rmlp_dl": -0.01540819576575837,
        "busy52:rmlp_dl_g2": 0.0024839116340121905
      },
      "ebm": {
        "gpw70:ebm_i25_bag": 0.10426318401010569,
        "busy52:ebm_g2_glminit": 0.048604822460544136
      },
      "xgb_llr": {
        "prop:xgb_llr": 0.7082731401362774
      },
      "l7": {
        "gpw70:lgb_dig_l7": 0.024016451956773602,
        "prop:l7_s2026": 0.38548820725270583
      },
      "gpt2": {}
    },
    "top_abs_coefs": [
      {
        "name": "prop:xgb_llr",
        "family": "Proprietary",
        "coef": 0.7082731401362774
      },
      {
        "name": "prop:l7_s2026",
        "family": "LM_GPT2",
        "coef": 0.38548820725270583
      },
      {
        "name": "gpw70:gb_bag5",
        "family": "EBM_GLM",
        "coef": 0.2924948382105839
      },
      {
        "name": "gpw70:lgb_dk_l4",
        "family": "GBDT",
        "coef": -0.11474636091083854
      },
      {
        "name": "busy52:lgb_g2_plaininit_l4",
        "family": "EBM_GLM",
        "coef": 0.11311619284770941
      },
      {
        "name": "gpw70:xgb_te_d2",
        "family": "GBDT",
        "coef": 0.1122820527903089
      },
      {
        "name": "busy52:glm_dl",
        "family": "LM_GPT2",
        "coef": -0.11069761870539621
      },
      {
        "name": "gpw70:ebm_i25_bag",
        "family": "EBM_GLM",
        "coef": 0.10426318401010569
      },
      {
        "name": "busy52:lgb_dl2_hier",
        "family": "GBDT",
        "coef": -0.09855933246121346
      },
      {
        "name": "gpw70:lgb_rawdig_l4",
        "family": "GBDT",
        "coef": 0.09654613635296645
      },
      {
        "name": "busy52:tabm_standalone",
        "family": "TabM_MLP",
        "coef": 0.09011790943638472
      },
      {
        "name": "gpw70:lgb_raw_l4_cnt",
        "family": "GBDT",
        "coef": 0.08412693596229268
      },
      {
        "name": "gpw70:realmlp_allcat",
        "family": "TabM_MLP",
        "coef": -0.06890692887078582
      },
      {
        "name": "gpw70:glm_te",
        "family": "LM_GPT2",
        "coef": -0.06890135776654498
      },
      {
        "name": "gpw70:realmlp_dig",
        "family": "TabM_MLP",
        "coef": -0.0645547070905269
      }
    ]
  },
  {
    "fold": 4,
    "C_L2": 0.01,
    "inner_L2": 0.9531704214459067,
    "C_Cal": 0.01,
    "inner_Cal": 0.9531281269059686,
    "alpha_Sim": 1000.0,
    "inner_Sim": 0.9507932699141071,
    "family_abs_weight": {
      "EBM_GLM": 0.5606375531761745,
      "GBDT": 0.6262113558297417,
      "LM_GPT2": 0.6360764112418009,
      "Proprietary": 0.6570410500323108,
      "TabM_MLP": 0.6062318443502033
    },
    "named_weights": {
      "tabm": {
        "busy52:tabm_dl": -0.054479682049714456,
        "busy52:tabm_standalone": 0.09036984527775116
      },
      "uid_v2": {
        "prop:uid_v2": 0.0026532451013720707
      },
      "realmlp": {
        "gpw70:realmlp_dig_s2": 0.019842203165079314,
        "gpw70:realmlp_dig_s100": 0.0009493693508113386,
        "gpw70:realmlp_dig": -0.04502783998492448,
        "gpw70:realmlp_te": 0.042013978944454224,
        "gpw70:realmlp_te_s2": -0.042744396785919146,
        "gpw70:realmlp_te_cnt": 0.027241543088189865,
        "gpw70:realmlp_cnt": 0.05077913214222566,
        "gpw70:realmlp_allcat": 0.007721621793754147,
        "gpw70:realmlp_rawdig": -0.010507644589383445,
        "gpw70:realmlp_v1": -0.06516178741637893
      },
      "rmlp": {
        "busy52:rmlp2_dl_hier": 0.09272934920927692,
        "busy52:rmlp_dl": -0.04422295661153455,
        "busy52:rmlp_dl_g2": -0.012440493940805663
      },
      "ebm": {
        "gpw70:ebm_i25_bag": 0.10402234170321537,
        "busy52:ebm_g2_glminit": 0.034455913249575
      },
      "xgb_llr": {
        "prop:xgb_llr": 0.6543878049309387
      },
      "l7": {
        "gpw70:lgb_dig_l7": 0.11741330907085847,
        "prop:l7_s2026": 0.30830330621759333
      },
      "gpt2": {}
    },
    "top_abs_coefs": [
      {
        "name": "prop:xgb_llr",
        "family": "Proprietary",
        "coef": 0.6543878049309387
      },
      {
        "name": "prop:l7_s2026",
        "family": "LM_GPT2",
        "coef": 0.30830330621759333
      },
      {
        "name": "gpw70:gb_bag5",
        "family": "EBM_GLM",
        "coef": 0.2571596481030893
      },
      {
        "name": "busy52:lgb_g2_plaininit_l4",
        "family": "EBM_GLM",
        "coef": 0.11883683571271994
      },
      {
        "name": "gpw70:lgb_dig_l7",
        "family": "GBDT",
        "coef": 0.11741330907085847
      },
      {
        "name": "gpw70:ebm_i25_bag",
        "family": "EBM_GLM",
        "coef": 0.10402234170321537
      },
      {
        "name": "busy52:lgb_dl2_hier",
        "family": "GBDT",
        "coef": -0.09438238777917164
      },
      {
        "name": "busy52:rmlp2_dl_hier",
        "family": "TabM_MLP",
        "coef": 0.09272934920927692
      },
      {
        "name": "busy52:tabm_standalone",
        "family": "TabM_MLP",
        "coef": 0.09036984527775116
      },
      {
        "name": "busy52:glm_dl",
        "family": "LM_GPT2",
        "coef": -0.07614922496528859
      },
      {
        "name": "gpw70:glm_sb",
        "family": "LM_GPT2",
        "coef": -0.0731438416053769
      },
      {
        "name": "gpw70:xgb_te_d2",
        "family": "GBDT",
        "coef": 0.06872552120847684
      },
      {
        "name": "gpw70:lgb_rawdig_l4",
        "family": "GBDT",
        "coef": 0.06664683212199218
      },
      {
        "name": "gpw70:realmlp_v1",
        "family": "TabM_MLP",
        "coef": -0.06516178741637893
      },
      {
        "name": "gpw70:lgb_raw_l4_cnt",
        "family": "GBDT",
        "coef": 0.05733561629379629
      }
    ]
  }
]
```

## 3. Yash divergence & w*

```json
{
  "divergence": {
    "1mrho_oof_e4": 49.274843362342445,
    "1mrho_oof_zr_e4": 46.80475005017337,
    "1mrho_test_e4": 102.09863047306311,
    "1mrho_test_zr_e4": 97.77920248075111,
    "1mrho_test_nonzone_e4": 116.10319269325764,
    "ratio_test_over_oof": 2.0720234404862756,
    "ratio_test_over_oof_zr": 2.0890871626476923
  },
  "wstar": {
    "per_fold": [
      {
        "fold": 0,
        "w_star": 1.0,
        "inner_auc": 0.9527629342872199
      },
      {
        "fold": 1,
        "w_star": 1.0,
        "inner_auc": 0.9525736316850082
      },
      {
        "fold": 2,
        "w_star": 1.0,
        "inner_auc": 0.9513632321415799
      },
      {
        "fold": 3,
        "w_star": 1.0,
        "inner_auc": 0.9525321174064454
      },
      {
        "fold": 4,
        "w_star": 1.0,
        "inner_auc": 0.9535934250479761
      }
    ],
    "mean_w": 1.0,
    "oof_auc_zr": 0.9526136241093358,
    "folds_zr": [
      0.9518234387106351,
      0.952578077169792,
      0.9574181062167212,
      0.9527307361163383,
      0.9485193893671968
    ],
    "sf25_vs_own": 22,
    "sf25_vs_part12": 0,
    "fixed_grid_zr": {
      "0.0": 0.9520983161437412,
      "0.25": 0.9523271678894165,
      "0.35": 0.9523999453244649,
      "0.5": 0.9524892035188021,
      "0.65": 0.9525547207238945,
      "0.75": 0.9525848659046822,
      "1.0": 0.9526136241093358
    }
  }
}
```

## 4. Staged files

```json
{
  "N1_own_nested_stack_zr.csv": {
    "sha": "ed60b410fb52a86393a978d1b2879d7af7fc94de2283d4ed67367cd6e1ebd134",
    "1mrho_vs_AB_e4": 191.3181625297422,
    "1mrho_vs_part12_e4": 97.77920248075111,
    "meta": "Meta_Ens_L2_Cal"
  },
  "N2_dual_nested_x_part12_cv_zr.csv": {
    "sha": "adbdbd84c0c08ff068d98affe20f0afc246a8fa21d7c74be4b3d36d82a484553",
    "mean_w_part12": 1.0,
    "1mrho_vs_AB_e4": 172.25933562884154,
    "1mrho_vs_N1_e4": 97.77920248075111
  },
  "N3_dual_nested_x_part12_50_50_zr.csv": {
    "sha": "fb4b2ba59dad27da61f494a04051aebc12ab32bda483efc343ad332856b27437",
    "1mrho_vs_AB_e4": 157.81279766681422,
    "1mrho_vs_N2_e4": 24.461366470625023
  },
  "H1_private_hedge_50AB_50N2.csv": {
    "sha": "6e17e0225a534d71cd342c45084864b48eed3f84ae2739bf861ed75af0c5dae3",
    "w_AB": 0.5,
    "1mrho_vs_AB_e4": 43.26066450095145,
    "1mrho_vs_N2_e4": 43.5026634396396
  },
  "H2_private_hedge_70AB_30N2.csv": {
    "sha": "1a49e674aad5d247da3162e688bf225d9d224f262a3f586ff9db8b4885b34a78",
    "w_AB": 0.7,
    "1mrho_vs_AB_e4": 15.542560074396361,
    "1mrho_vs_N2_e4": 84.94689718188786
  }
}
```

## 5. STOP

Step 5 complete. **Awaiting human audit.** Reply `PROCEED TO STEP 6` to continue.
