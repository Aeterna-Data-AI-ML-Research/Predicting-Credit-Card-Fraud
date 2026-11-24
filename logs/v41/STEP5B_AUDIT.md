# V41 STEP5b AUDIT — Surgical Nested Stack Fix

**NO_SUBMIT=True** · Hard stop after 5b. Do **not** run Step 6 without `PROCEED TO STEP 6`.

## Executive findings

- Step5 high-AUC drops audited: **87** (see §1)
- Hygiene_v2 candidates → kept after ρ>0.9992 dedup: **52**
- By family: `{"LM_GPT2": {"before_dedup": 25, "after_dedup": 12}, "GBDT": {"before_dedup": 22, "after_dedup": 12}, "TabM_MLP": {"before_dedup": 16, "after_dedup": 16}, "EBM_GLM": {"before_dedup": 12, "after_dedup": 8}, "Proprietary": {"before_dedup": 4, "after_dedup": 4}, "Other": {"before_dedup": 0, "after_dedup": 0}}`
- fullbank recipe: pillars=['wide_lm', 'lgb_subodh', 'nov08_lm', 'gpt2_rev', 'gpt2_rev_raw', 'P_super_v5p5_clean', 'P_super_v6_clean', 'P_champ'] → AUC **0.952621** (`scripts/v36_1_lib.py::build_z_fullbank → scripts/v36_1_workers/w0_repair_baselines_and_intel.py (+v10orth grid)`)
- Nested winner: **Meta_Step3_FullbankArtifact** OOF=**0.952621** → +Zr **0.952670** (ASSERT ≥0.95262: PASS)
- Part12 0.952565/0.952614 · fullbank 0.952621/0.952670
- Divergence `1−ρ×1e−4`: OOF=0.642, test=0.615, ratio_test_over_oof=**0.96**
- Nested w* mean Part12=0.050; dual+Zr OOF=**0.952664**
- Tip residual: AB vs N1v2 e4=170.964, vs dual=171.001
- Staged: N1v2/N2v2/N3v2 + H1v2/H2v2

## 1. Step5 drop autopsy (solo OOF AUC ≥ 0.945)

| name | reason | auc | raw_disp | raw_KS | cal_fold_KS |
|---|---|---:|---:|---:|---:|
| `gpw70:gb_bag10` | dedup_rho=0.999861_kept=gpw70:gb_bag5 | 0.95150 | 1.000 | 0.0006 | 0.0000 |
| `gpw70:gbx_bag5` | dedup_rho=0.999725_kept=gpw70:gb_bag5 | 0.95146 | 1.000 | 0.0006 | 0.0000 |
| `gpw70:gbx_bag3` | dedup_rho=0.999546_kept=gpw70:gb_bag5 | 0.95142 | 1.000 | 0.0005 | 0.0000 |
| `gpw70:gb_l3` | dedup_rho=0.998995_kept=gpw70:gb_bag5 | 0.95132 | 1.013 | 0.0064 | 0.0000 |
| `gpw70:gbc_bag3` | dedup_rho=0.998213_kept=gpw70:gb_bag5 | 0.95126 | 1.000 | 0.0004 | 0.0000 |
| `gpw70:gbn_bag5_w512` | dedup_rho=0.996175_kept=gpw70:gb_bag5 | 0.95122 | 1.000 | 0.0005 | 0.0000 |
| `gpw70:gbn_bag5_w128` | dedup_rho=0.995669_kept=gpw70:gb_bag5 | 0.95122 | 1.000 | 0.0005 | 0.0000 |
| `gpw70:gb_ohk` | dedup_rho=0.996984_kept=gpw70:gb_bag5 | 0.95119 | 1.017 | 0.0079 | 0.0000 |
| `gpw70:gbe_bag3` | dedup_rho=0.996578_kept=gpw70:gb_bag5 | 0.95115 | 1.000 | 0.0006 | 0.0000 |
| `gpw70:gbr_bag3` | dedup_rho=0.997164_kept=gpw70:gb_bag5 | 0.95107 | 1.000 | 0.0004 | 0.0000 |
| `busy52:lgb_g2_plaininit_l3` | dedup_rho=0.999619_kept=busy52:lgb_g2_plaininit_l4 | 0.95102 | 0.997 | 0.0024 | 0.0000 |
| `busy52:lgb_g2_glminit_l4` | dedup_rho=0.996280_kept=busy52:lgb_g2_plaininit_l4 | 0.95089 | 1.003 | 0.0069 | 0.0000 |
| `busy52:lgb_g2_glminit_l5` | dedup_rho=0.996231_kept=busy52:lgb_g2_plaininit_l4 | 0.95088 | 1.003 | 0.0064 | 0.0000 |
| `busy52:lgb_g2_glminit_s50` | dedup_rho=0.996527_kept=busy52:lgb_g2_plaininit_l4 | 0.95083 | 1.002 | 0.0073 | 0.0000 |
| `busy52:lgb_g2_glminit` | dedup_rho=0.996156_kept=busy52:lgb_g2_plaininit_l4 | 0.95082 | 1.002 | 0.0074 | 0.0000 |
| `busy52:lgb_g2_glminit_l7` | dedup_rho=0.995911_kept=busy52:lgb_g2_plaininit_l4 | 0.95080 | 1.002 | 0.0069 | 0.0000 |
| `busy52:lgb_g2_glminit_l4_hier` | dedup_rho=0.995805_kept=busy52:lgb_g2_plaininit_l4 | 0.95079 | 1.003 | 0.0066 | 0.0000 |
| `busy52:xgb_g2_glminit` | dedup_rho=0.995956_kept=busy52:lgb_g2_plaininit_l4 | 0.95078 | 1.003 | 0.0080 | 0.0000 |
| `busy52:lgb_dl3` | dedup_rho=0.998127_kept=busy52:lgb_dl3_g2c3 | 0.95067 | 1.005 | 0.0140 | 0.0000 |
| `busy52:lgb_dl4_g2cc` | dedup_rho=0.997900_kept=busy52:lgb_dl3_g2c3 | 0.95067 | 1.006 | 0.0158 | 0.0000 |
| `gpw70:lgb_diglead_l3` | dedup_rho=0.997476_kept=busy52:lgb_dl3 | 0.95064 | 1.008 | 0.0149 | 0.0000 |
| `busy52:lgb_dl4_g2` | dedup_rho=0.998439_kept=busy52:lgb_dl3_g2c3 | 0.95064 | 1.006 | 0.0147 | 0.0000 |
| `gpw70:lgb_diglead_l3_bag` | dedup_rho=0.998171_kept=gpw70:lgb_diglead_l3 | 0.95063 | 1.008 | 0.0182 | 0.0000 |
| `busy52:lgb_dl3_cc` | dedup_rho=0.997418_kept=gpw70:lgb_diglead_l3 | 0.95062 | 1.005 | 0.0158 | 0.0000 |
| `busy52:lgb_dl3_hier` | dedup_rho=0.997547_kept=gpw70:lgb_diglead_l3 | 0.95060 | 1.004 | 0.0166 | 0.0000 |
| `busy52:lgb_dl3_trie` | dedup_rho=0.997733_kept=gpw70:lgb_diglead_l3 | 0.95059 | 1.003 | 0.0128 | 0.0000 |
| `busy52:lgb_g2_glminit_l2` | dedup_rho=0.995654_kept=busy52:lgb_g2_plaininit_l4 | 0.95058 | 1.000 | 0.0057 | 0.0000 |
| `busy52:lgb_dl3_s10` | dedup_rho=0.997795_kept=gpw70:lgb_diglead_l3 | 0.95056 | 1.005 | 0.0139 | 0.0000 |
| `gpw70:lgb_diglead_bag5` | dedup_rho=0.996728_kept=gpw70:lgb_diglead_l3 | 0.95054 | 1.007 | 0.0208 | 0.0000 |
| `busy52:lgb_dl4_hier` | dedup_rho=0.996276_kept=gpw70:lgb_diglead_l3 | 0.95053 | 1.004 | 0.0181 | 0.0000 |
| `busy52:xgb_dl2` | dedup_rho=0.995507_kept=gpw70:lgb_diglead_l3 | 0.95051 | 1.003 | 0.0155 | 0.0000 |
| `gpw70:lgb_diglead_bag` | dedup_rho=0.996779_kept=gpw70:lgb_diglead_l3 | 0.95051 | 1.006 | 0.0211 | 0.0000 |
| `busy52:glm_plain_s50` | dedup_rho=0.998563_kept=busy52:glm_plain_g2_s50 | 0.95051 | 1.000 | 0.0035 | 0.0000 |
| `gpw70:glm_dl_s50_bag` | dedup_rho=1.000000_kept=busy52:glm_plain_s50 | 0.95051 | 1.000 | 0.0036 | 0.0000 |
| `gpw70:lgb_dig_s100_l3_bag` | dedup_rho=0.995053_kept=gpw70:lgb_diglead_l3 | 0.95050 | 1.007 | 0.0208 | 0.0000 |
| `gpw70:glm_dl_s50` | dedup_rho=0.999885_kept=gpw70:glm_dl_s50_bag | 0.95049 | 1.000 | 0.0036 | 0.0000 |
| `gpw70:lgb_dig_s100_bag5` | dedup_rho=0.996224_kept=busy52:lgb_dl4_g2 | 0.95049 | 1.007 | 0.0221 | 0.0000 |
| `busy52:lgb_dl4` | dedup_rho=0.996822_kept=gpw70:lgb_diglead_l3 | 0.95048 | 1.004 | 0.0164 | 0.0000 |
| `busy52:lgb_dl4_c3` | dedup_rho=0.996281_kept=gpw70:lgb_diglead_l3 | 0.95047 | 1.005 | 0.0156 | 0.0000 |
| `busy52:lgb_dl3_s100` | dedup_rho=0.995678_kept=gpw70:lgb_diglead_l3 | 0.95044 | 1.005 | 0.0160 | 0.0000 |
| `gpw70:lgb_dig_s100_l3` | dedup_rho=0.995146_kept=gpw70:lgb_diglead_l3 | 0.95043 | 1.009 | 0.0174 | 0.0000 |
| `gpw70:glm_dl_bag` | dedup_rho=0.999326_kept=gpw70:glm_dl_s50_bag | 0.95043 | 1.000 | 0.0032 | 0.0000 |
| `gpw70:lgb_dig_s100_bag` | dedup_rho=0.999633_kept=gpw70:lgb_dig_s100_bag5 | 0.95043 | 1.005 | 0.0238 | 0.0000 |
| `busy52:lgb_dl4_s100` | dedup_rho=0.998987_kept=gpw70:lgb_dig_s100_bag5 | 0.95041 | 1.005 | 0.0181 | 0.0000 |
| `gpw70:glm_dl` | dedup_rho=0.999211_kept=gpw70:glm_dl_s50_bag | 0.95041 | 1.000 | 0.0031 | 0.0000 |
| `gpw70:lgb_diglead_l4` | dedup_rho=0.997469_kept=gpw70:lgb_diglead_l3 | 0.95040 | 1.008 | 0.0181 | 0.0000 |
| `busy52:lgb_dl4_s10` | dedup_rho=0.996056_kept=gpw70:lgb_diglead_l3 | 0.95039 | 1.004 | 0.0193 | 0.0000 |
| `gpw70:lgb_dig_l4` | dedup_rho=0.996284_kept=gpw70:lgb_diglead_l3 | 0.95036 | 0.994 | 0.0055 | 0.0000 |
| `busy52:glm_ohk_g2_c0003_s50` | dedup_rho=0.995337_kept=busy52:glm_plain_g2_s50 | 0.95034 | 1.003 | 0.0083 | 0.0000 |
| `busy52:glm_ohk_g2_c0005_s50` | dedup_rho=0.999219_kept=busy52:glm_ohk_g2_c0003_s100 | 0.95031 | 1.002 | 0.0066 | 0.0000 |
| `busy52:glm_ohk_g2_c0005_s100` | dedup_rho=0.999665_kept=busy52:glm_ohk_g2_c0003_s100 | 0.95030 | 1.001 | 0.0068 | 0.0000 |
| `busy52:glm_ohk_g2_c0002_s50` | dedup_rho=0.999456_kept=busy52:glm_ohk_g2_c0003_s100 | 0.95029 | 1.004 | 0.0097 | 0.0000 |
| `busy52:glm_ohk_g2_c0003` | dedup_rho=0.995221_kept=busy52:glm_plain_g2_s50 | 0.95029 | 1.003 | 0.0078 | 0.0000 |
| `gpw70:lgb_dig_s100` | dedup_rho=0.997768_kept=gpw70:lgb_dig_s100_bag5 | 0.95029 | 1.005 | 0.0192 | 0.0000 |
| `busy52:glm_ohk_g2_c0005_m2` | dedup_rho=0.997786_kept=busy52:glm_ohk_g2_c0003_s100 | 0.95027 | 1.002 | 0.0062 | 0.0000 |
| `gpw70:ebm_i25_bag4` | dedup_rho=0.999658_kept=gpw70:ebm_i25_bag | 0.95026 | 1.005 | 0.0109 | 0.0000 |
| `busy52:glm_ohk_g2_c0005` | dedup_rho=0.997778_kept=busy52:glm_ohk_g2_c0003_s100 | 0.95025 | 1.002 | 0.0060 | 0.0000 |
| `gpw70:xgb_dig_d2` | dedup_rho=0.995166_kept=gpw70:lgb_diglead_l3 | 0.95025 | 0.994 | 0.0054 | 0.0000 |
| `gpw70:ebm_lead4_bag` | dedup_rho=0.999523_kept=gpw70:ebm_i25_bag | 0.95024 | 1.006 | 0.0104 | 0.0000 |
| `gpw70:ebm_diglead_i25` | dedup_rho=0.999563_kept=gpw70:ebm_i25_bag | 0.95022 | 1.005 | 0.0092 | 0.0000 |
| `gpw70:lgb_diglead_l2` | dedup_rho=0.998903_kept=busy52:lgb_dl2_hier | 0.95020 | 0.995 | 0.0045 | 0.0000 |
| `busy52:glm_ohk_g2x_c0005` | dedup_rho=0.997178_kept=busy52:glm_ohk_g2_c0003_s100 | 0.95019 | 1.001 | 0.0063 | 0.0000 |
| `busy52:glm_ohk_c0005` | dedup_rho=0.995065_kept=gpw70:glm_dl_s50_bag | 0.95017 | 1.001 | 0.0066 | 0.0000 |
| `gpw70:ebm_all_bag` | dedup_rho=0.998489_kept=gpw70:ebm_i25_bag | 0.95013 | 1.006 | 0.0122 | 0.0000 |
| `gpw70:ebm_diglead_bag` | dedup_rho=0.998229_kept=gpw70:ebm_i25_bag | 0.95013 | 1.006 | 0.0108 | 0.0000 |
| `gpw70:glm_ohk2` | dedup_rho=0.996804_kept=busy52:glm_ohk_g2x_c0005 | 0.95012 | 1.001 | 0.0071 | 0.0000 |
| `gpw70:glm_ohk2_bag` | dedup_rho=0.999937_kept=gpw70:glm_ohk2 | 0.95012 | 1.002 | 0.0071 | 0.0000 |
| `gpw70:ebm_diglead_r11` | dedup_rho=0.998126_kept=gpw70:ebm_i25_bag | 0.95009 | 0.998 | 0.0120 | 0.0000 |
| `busy52:ebm_dl` | dedup_rho=0.996427_kept=gpw70:ebm_i25_bag | 0.95008 | 0.990 | 0.0059 | 0.0003 |
| `busy52:ebm_dl_g2` | dedup_rho=0.996032_kept=gpw70:ebm_i25_bag | 0.95006 | 1.007 | 0.0167 | 0.0000 |
| `busy52:glm_ohk_c0002` | dedup_rho=0.997232_kept=gpw70:glm_ohk2 | 0.95006 | 1.004 | 0.0106 | 0.0000 |
| `busy52:glm_ohk_g2_c001` | dedup_rho=0.996351_kept=gpw70:glm_ohk2 | 0.95005 | 1.001 | 0.0045 | 0.0000 |
| `gpw70:ebm_diglead_r13` | dedup_rho=0.997131_kept=gpw70:ebm_i25_bag | 0.95005 | 1.014 | 0.0112 | 0.0000 |
| `gpw70:glm_ohkl_bag` | dedup_rho=0.998060_kept=gpw70:glm_ohk2 | 0.95003 | 1.001 | 0.0051 | 0.0000 |
| `gpw70:glm_ohkl` | dedup_rho=0.998082_kept=gpw70:glm_ohk2 | 0.95002 | 1.000 | 0.0050 | 0.0000 |
| `busy52:glm_ohk_c001` | dedup_rho=0.997958_kept=gpw70:glm_ohk2 | 0.95002 | 1.000 | 0.0049 | 0.0000 |
| `gpw70:ebm_dig_r7` | dedup_rho=0.996520_kept=gpw70:ebm_i25_bag | 0.94996 | 1.013 | 0.0131 | 0.0000 |
| `gpw70:ebm_dig_bag` | dedup_rho=0.997451_kept=gpw70:ebm_i25_bag | 0.94996 | 1.005 | 0.0137 | 0.0000 |
| `gpw70:glm_ohk_c01` | dedup_rho=0.997660_kept=gpw70:glm_ohk2 | 0.94991 | 1.001 | 0.0054 | 0.0000 |
| `gpw70:ebm_dig` | dedup_rho=0.997005_kept=gpw70:ebm_i25_bag | 0.94981 | 0.998 | 0.0156 | 0.0000 |
| `busy52:rmlp2_dl` | dedup_rho=0.996231_kept=busy52:rmlp2_dl_hier | 0.94959 | 0.993 | 0.0086 | 0.0000 |
| `gpw70:lgb_te_l4` | dedup_rho=0.996272_kept=gpw70:lgb_te_l4_s100 | 0.94885 | 0.992 | 0.0049 | 0.0000 |
| `gpw70:glm_dig` | dedup_rho=0.998886_kept=busy52:glm_dl | 0.94879 | 0.996 | 0.0049 | 0.0000 |
| `gpw70:lgb_te_l4_s5` | dedup_rho=0.995265_kept=gpw70:lgb_te_l4_s100 | 0.94878 | 0.993 | 0.0045 | 0.0000 |
| `gpw70:lgb_te_l4_s300` | dedup_rho=0.995933_kept=gpw70:lgb_te_l4_s100 | 0.94873 | 0.993 | 0.0049 | 0.0000 |
| `busy52:lgb_rawdig` | dedup_rho=0.997563_kept=gpw70:lgb_rawdig_l4 | 0.94835 | 1.003 | 0.0054 | 0.0000 |
| `prop:gpt2_med_4ep` | hygiene_raw_disp=0.9532356774035324 raw_KS=0.2488374394920964 | 0.94824 | 0.953 | 0.2488 | 0.0000 |

## 2. fullbank recipe

```json
{
  "script": "scripts/v36_1_lib.py::build_z_fullbank \u2192 scripts/v36_1_workers/w0_repair_baselines_and_intel.py (+v10orth grid)",
  "method": "B_clean 8-pillar nest LogReg(C=0.5) \u2192 blend with wide (best of nest/wide/0.85w+0.15n/\u2026/wide+0.02orth) \u2192 then micro-orth of v10_ridge_resid_orth_LM onto wide (c grid \u2208[-0.04,0.04]) \u2192 AUC 0.9526207. GM_V36_1_RUN.log: selected wide+orth 0.952565 then +v10orth \u2192 0.952621.",
  "pillars": [
    "wide_lm",
    "lgb_subodh",
    "nov08_lm",
    "gpt2_rev",
    "gpt2_rev_raw",
    "P_super_v5p5_clean",
    "P_super_v6_clean",
    "P_champ"
  ],
  "n_pillars": 8,
  "fb_auc": 0.9526207252382072,
  "v10_w_fixed": 0.006,
  "note": "Artifact at SSD/preds/z_fullbank_{oof,test}.npy"
}
```

## 3. Kept bank (hygiene_v2 + dedup)

n_kept=52 dedup_dropped=27

| name | family | auc | cal_fold_KS | raw_disp |
|---|---|---:|---:|---:|
| `gpw70:gb_bag5` | GBDT | 0.95151 | 0.0000 | 1.000 |
| `gpw70:gb_l3` | GBDT | 0.95132 | 0.0000 | 1.013 |
| `gpw70:gbc_bag3` | GBDT | 0.95126 | 0.0000 | 1.000 |
| `gpw70:gbn_bag5_w512` | GBDT | 0.95122 | 0.0000 | 1.000 |
| `gpw70:gb_ohk` | GBDT | 0.95119 | 0.0000 | 1.017 |
| `gpw70:gbe_bag3` | GBDT | 0.95115 | 0.0000 | 1.000 |
| `gpw70:gbr_bag3` | GBDT | 0.95107 | 0.0000 | 1.000 |
| `gpw70:ebm_i25_bag` | EBM_GLM | 0.95026 | 0.0000 | 1.005 |
| `gpw70:ebm_all_bag` | EBM_GLM | 0.95013 | 0.0000 | 1.006 |
| `gpw70:glm_ohk2` | LM_GPT2 | 0.95012 | 0.0000 | 1.001 |
| `gpw70:ebm_diglead_r11` | EBM_GLM | 0.95009 | 0.0000 | 0.998 |
| `gpw70:ebm_diglead_r13` | EBM_GLM | 0.95005 | 0.0000 | 1.014 |
| `gpw70:ebm_dig_r7` | EBM_GLM | 0.94996 | 0.0000 | 1.013 |
| `gpw70:realmlp_dig_s2` | TabM_MLP | 0.94867 | 0.0000 | 0.991 |
| `gpw70:realmlp_dig_s100` | TabM_MLP | 0.94858 | 0.0000 | 0.992 |
| `gpw70:realmlp_dig` | TabM_MLP | 0.94853 | 0.0000 | 0.991 |
| `gpw70:realmlp_te` | TabM_MLP | 0.94771 | 0.0000 | 0.993 |
| `gpw70:realmlp_te_s2` | TabM_MLP | 0.94762 | 0.0000 | 0.994 |
| `gpw70:realmlp_te_cnt` | TabM_MLP | 0.94760 | 0.0000 | 0.994 |
| `gpw70:realmlp_cnt` | TabM_MLP | 0.94664 | 0.0000 | 0.996 |
| `gpw70:realmlp_allcat` | TabM_MLP | 0.94640 | 0.0000 | 0.994 |
| `gpw70:realmlp_rawdig` | TabM_MLP | 0.94615 | 0.0000 | 0.995 |
| `gpw70:realmlp_v1` | TabM_MLP | 0.94584 | 0.0000 | 0.996 |
| `busy52:lgb_g2_plaininit_l4` | GBDT | 0.95109 | 0.0000 | 0.999 |
| `busy52:lgb_g2_glminit_l4` | GBDT | 0.95089 | 0.0000 | 1.003 |
| `busy52:lgb_g2_glminit_s50` | GBDT | 0.95083 | 0.0000 | 1.002 |
| `busy52:glm_plain_g2_s50` | LM_GPT2 | 0.95062 | 0.0000 | 1.000 |
| `busy52:ebm_g2_glminit` | EBM_GLM | 0.95059 | 0.0000 | 0.999 |
| `busy52:glm_plain_s50` | LM_GPT2 | 0.95051 | 0.0000 | 1.000 |
| `busy52:glm_ohk_g2_c0003_s50` | LM_GPT2 | 0.95034 | 0.0000 | 1.003 |
| `busy52:glm_ohk_g2_c0005_m2` | LM_GPT2 | 0.95027 | 0.0000 | 1.002 |
| `busy52:glm_ohk_c0005` | LM_GPT2 | 0.95017 | 0.0000 | 1.001 |
| `busy52:ebm_dl` | EBM_GLM | 0.95008 | 0.0003 | 0.990 |
| `busy52:ebm_dl_g2` | EBM_GLM | 0.95006 | 0.0000 | 1.007 |
| `busy52:glm_ohk_c0002` | LM_GPT2 | 0.95006 | 0.0000 | 1.004 |
| `busy52:tabm_dl` | TabM_MLP | 0.94970 | 0.0000 | 0.992 |
| `busy52:rmlp2_dl_hier` | TabM_MLP | 0.94967 | 0.0000 | 0.984 |
| `busy52:rmlp2_dl` | TabM_MLP | 0.94959 | 0.0000 | 0.993 |
| `busy52:rmlp_dl` | TabM_MLP | 0.94856 | 0.0000 | 0.990 |
| `busy52:rmlp_dl_g2` | TabM_MLP | 0.94839 | 0.0000 | 0.999 |
| `busy52:tabm_standalone` | TabM_MLP | 0.93731 | 0.0000 | 0.980 |
| `prop:uid_v2` | Proprietary | 0.93485 | 0.0000 | 0.997 |
| `prop:xgb_llr` | GBDT | 0.95195 | 0.0000 | 1.002 |
| `prop:l7_s2026` | LM_GPT2 | 0.94927 | 0.0000 | 0.996 |
| `prop:gpt2_med_4ep` | LM_GPT2 | 0.94824 | 0.0000 | 0.953 |
| `pillar:wide_lm_part12` | LM_GPT2 | 0.95257 | 0.0000 | 1.000 |
| `pillar:nov08_lm` | LM_GPT2 | 0.95043 | 0.0000 | 0.128 |
| `pillar:lgb_subodh` | GBDT | 0.94679 | 0.0000 | 0.995 |
| `pillar:gpt2_rev` | LM_GPT2 | 0.94818 | 0.0000 | 0.997 |
| `pillar:P_super_v5p5_clean` | Proprietary | 0.95234 | 0.0000 | 1.000 |
| `pillar:P_super_v6_clean` | Proprietary | 0.95220 | 0.0000 | 1.000 |
| `pillar:P_champ` | Proprietary | 0.95214 | 0.0000 | 1.000 |

## 4. Nested meta-stackers

```json
{
  "Meta_L2_LogReg": {
    "oof_auc": 0.9523266234032444,
    "oof_auc_zr": 0.9523747248815811,
    "folds": [
      0.951243040346303,
      0.9525883650485654,
      0.9572020638696621,
      0.9527869450341911,
      0.9479362241748609
    ],
    "folds_zr": [
      0.9512950733391187,
      0.9526330121431273,
      0.9572450361932608,
      0.952819598407199,
      0.9480069747033024
    ],
    "sf25_vs_part12": 9,
    "sf25_vs_fullbank": 8
  },
  "Meta_Ridge_Logit": {
    "oof_auc": 0.9500654103028363,
    "oof_auc_zr": 0.950141109728739,
    "folds": [
      0.9480791334814124,
      0.9511439325757469,
      0.9556803775369999,
      0.9503193020752756,
      0.9451200517462591
    ],
    "folds_zr": [
      0.9481687325605039,
      0.9512145929959177,
      0.9557499939441029,
      0.950372285738472,
      0.9452180398299188
    ],
    "sf25_vs_part12": 2,
    "sf25_vs_fullbank": 2
  },
  "Meta_CoordAscent_Polish": {
    "oof_auc": 0.9488125409705568,
    "oof_auc_zr": 0.9488616062534321,
    "folds": [
      0.9515908740179722,
      0.9524850761128406,
      0.9573921053198395,
      0.9528438251033017,
      0.948165167540545
    ],
    "folds_zr": [
      0.9516441193213913,
      0.9525300028537601,
      0.957434325705455,
      0.9528758352896877,
      0.9482305271086505
    ],
    "sf25_vs_part12": 11,
    "sf25_vs_fullbank": 7
  },
  "Meta_Ens_L2_Polish": {
    "oof_auc": 0.9524774370613731,
    "oof_auc_zr": 0.9525246324093151,
    "folds": [
      0.9515455242759214,
      0.9525142618710158,
      0.9573447953705179,
      0.9528539234439882,
      0.9481354846332899
    ],
    "folds_zr": [
      0.9515988504000477,
      0.9525593346494776,
      0.9573869287550445,
      0.9528859336303742,
      0.9482010896687537
    ],
    "sf25_vs_part12": 9,
    "sf25_vs_fullbank": 9
  },
  "Meta_FullbankStyle_Blend": {
    "oof_auc": 0.9525651611110737,
    "oof_auc_zr": 0.9526136508954947,
    "folds": [
      0.9517736034193499,
      0.9525308138281967,
      0.9573724120019098,
      0.9526968709424479,
      0.9484516403763252
    ],
    "folds_zr": [
      0.9518234387106351,
      0.952578077169792,
      0.9574181062167212,
      0.9527307361163383,
      0.9485193893671968
    ],
    "sf25_vs_part12": 0,
    "sf25_vs_fullbank": 6
  },
  "Meta_PillarC05_FullbankBlend": {
    "oof_auc": 0.9525651699983607,
    "oof_auc_zr": 0.9526136597827819,
    "folds": [
      0.9517736034193499,
      0.9525308138281967,
      0.9573724120019098,
      0.9526968709424479,
      0.9484516403763252
    ],
    "folds_zr": [
      0.9518234387106351,
      0.952578077169792,
      0.9574181062167212,
      0.9527307361163383,
      0.9485193893671968
    ],
    "sf25_vs_part12": 0,
    "sf25_vs_fullbank": 6
  },
  "Meta_AllC05_FullbankBlend": {
    "oof_auc": 0.9525664885485983,
    "oof_auc_zr": 0.9526144583956474,
    "folds": [
      0.9517414398864829,
      0.9525668571364937,
      0.9573753451814814,
      0.9527346418437976,
      0.9484361355393943
    ],
    "folds_zr": [
      0.9517910886684441,
      0.952613424469377,
      0.9574203154229451,
      0.9527677737227952,
      0.9485037229568145
    ],
    "sf25_vs_part12": 13,
    "sf25_vs_fullbank": 8
  },
  "Meta_Wide_v10orth": {
    "oof_auc": 0.9525652021914708,
    "oof_auc_zr": 0.9526136932810182,
    "folds": [
      0.9517736189617936,
      0.9525308728646499,
      0.9573724834670899,
      0.9526969734794456,
      0.9484517304845963
    ],
    "folds_zr": [
      0.9518234573615674,
      0.9525781393134268,
      0.9574181745747197,
      0.9527308386533361,
      0.9485194825826491
    ],
    "sf25_vs_part12": 12,
    "sf25_vs_fullbank": 6
  },
  "Meta_Wide_xgbllr_orth": {
    "oof_auc": 0.9525651344492125,
    "oof_auc_zr": 0.9526136241093358,
    "folds": [
      0.9517736034193499,
      0.9525308138281967,
      0.9573724120019098,
      0.9526968709424479,
      0.9484516403763252
    ],
    "folds_zr": [
      0.9518234387106351,
      0.952578077169792,
      0.9574181062167212,
      0.9527307361163383,
      0.9485193893671968
    ],
    "sf25_vs_part12": 0,
    "sf25_vs_fullbank": 6
  },
  "Meta_Step3_FullbankArtifact": {
    "oof_auc": 0.9526207252382072,
    "oof_auc_zr": 0.9526697935042275,
    "folds": [
      0.95177377438623,
      0.9525366366867956,
      0.9574441847931466,
      0.9527735655095767,
      0.948568812200108
    ],
    "folds_zr": [
      0.9518246541297276,
      0.952584757610554,
      0.9574900654388634,
      0.9528075207917378,
      0.948637185734511
    ],
    "sf25_vs_part12": 19,
    "sf25_vs_fullbank": 0
  }
}
```

### Weight logs (truncated)

```json
[
  {
    "fold": 0,
    "C_L2": 0.01,
    "inner_L2": 0.9526299663282172,
    "alpha_Ridge": 10000.0,
    "inner_Ridge": 0.9505724698066761,
    "inner_Polish": 0.9527183213716571,
    "family_abs_weight": {
      "EBM_GLM": 0.11986530185363073,
      "GBDT": 0.37296547233697674,
      "LM_GPT2": 1.0372428857443836,
      "Proprietary": 0.5805249266973143,
      "TabM_MLP": 0.41214585694489286
    },
    "named_weights": {
      "tabm": {
        "busy52:tabm_dl": -0.05181597240158093,
        "busy52:tabm_standalone": 0.10375175646282458
      },
      "uid_v2": {
        "prop:uid_v2": 0.0056538524001111395
      },
      "realmlp": {
        "gpw70:realmlp_dig_s2": -0.022507234062414287,
        "gpw70:realmlp_dig_s100": -0.02134547499175293,
        "gpw70:realmlp_dig": -0.026554931052107614,
        "gpw70:realmlp_te": 0.02764406111383897,
        "gpw70:realmlp_te_s2": -0.013007771251256439,
        "gpw70:realmlp_te_cnt": 0.002249671718236733,
        "gpw70:realmlp_cnt": 0.028150456018579018,
        "gpw70:realmlp_allcat": 0.016448415109489804,
        "gpw70:realmlp_rawdig": 0.0038009830735889465,
        "gpw70:realmlp_v1": -0.003895347331061176
      },
      "rmlp": {
        "busy52:rmlp2_dl_hier": 0.04504845270513446,
        "busy52:rmlp2_dl": 0.0379691754349872,
        "busy52:rmlp_dl": -0.001654662530512369,
        "busy52:rmlp_dl_g2": -0.006301491687527391
      },
      "ebm": {
        "gpw70:ebm_i25_bag": 0.02959154251885008,
        "gpw70:ebm_all_bag": -0.003071961277352544,
        "gpw70:ebm_diglead_r11": 0.016366431689909398,
        "gpw70:ebm_diglead_r13": 0.011910558270430749,
        "gpw70:ebm_dig_r7": -0.0252041133447587,
        "busy52:ebm_g2_glminit": -0.0051517352684858535,
        "busy52:ebm_dl": -0.0034352646633699856,
        "busy52:ebm_dl_g2": 0.025133694820473416
      },
      "xgb_llr": {
        "prop:xgb_llr": 0.15475358557847904
      },
      "l7": {
        "prop:l7_s2026": 0.033659754271569406
      },
      "gpt2": {
        "prop:gpt2_med_4ep": 0.03315380952779869,
        "pillar:gpt2_rev": 0.05180294608405696
      },
      "compact": {}
    },
    "top_abs_coefs": [
      {
        "name": "pillar:wide_lm_part12",
        "family": "LM_GPT2",
        "coef": 0.30123966798224683
      },
      {
        "name": "pillar:P_super_v5p5_clean",
        "family": "Proprietary",
        "coef": 0.24054667884330405
      },
      {
        "name": "pillar:nov08_lm",
        "family": "LM_GPT2",
        "coef": 0.23332280474197772
      },
      {
        "name": "pillar:P_super_v6_clean",
        "family": "Proprietary",
        "coef": 0.19476184237780517
      },
      {
        "name": "prop:xgb_llr",
        "family": "GBDT",
        "coef": 0.15475358557847904
      },
      {
        "name": "pillar:P_champ",
        "family": "Proprietary",
        "coef": 0.13956255307609386
      },
      {
        "name": "busy52:tabm_standalone",
        "family": "TabM_MLP",
        "coef": 0.10375175646282458
      },
      {
        "name": "busy52:glm_ohk_c0002",
        "family": "LM_GPT2",
        "coef": -0.07311761850618494
      },
      {
        "name": "busy52:glm_ohk_c0005",
        "family": "LM_GPT2",
        "coef": -0.07161572150793234
      },
      {
        "name": "gpw70:glm_ohk2",
        "family": "LM_GPT2",
        "coef": -0.05850494700816533
      },
      {
        "name": "busy52:glm_ohk_g2_c0005_m2",
        "family": "LM_GPT2",
        "coef": -0.05616182284850981
      },
      {
        "name": "busy52:glm_ohk_g2_c0003_s50",
        "family": "LM_GPT2",
        "coef": -0.0547030224280372
      },
      {
        "name": "busy52:tabm_dl",
        "family": "TabM_MLP",
        "coef": -0.05181597240158093
      },
      {
        "name": "pillar:gpt2_rev",
        "family": "LM_GPT2",
        "coef": 0.05180294608405696
      },
      {
        "name": "busy52:rmlp2_dl_hier",
        "family": "TabM_MLP",
        "coef": 0.04504845270513446
      },
      {
        "name": "busy52:glm_plain_s50",
        "family": "LM_GPT2",
        "coef": -0.04330755520435229
      },
      {
        "name": "gpw70:gb_ohk",
        "family": "GBDT",
        "coef": 0.04271940164880687
      },
      {
        "name": "gpw70:gb_bag5",
        "family": "GBDT",
        "coef": 0.04185838193266805
      },
      {
        "name": "gpw70:gbc_bag3",
        "family": "GBDT",
        "coef": 0.03824858798019784
      },
      {
        "name": "busy52:rmlp2_dl",
        "family": "TabM_MLP",
        "coef": 0.0379691754349872
      }
    ],
    "top_polish": [
      {
        "name": "pillar:wide_lm_part12",
        "family": "LM_GPT2",
        "coef": 0.46974003854501367
      },
      {
        "name": "pillar:P_super_v5p5_clean",
        "family": "Proprietary",
        "coef": 0.091942998527037
      },
      {
        "name": "pillar:nov08_lm",
        "family": "LM_GPT2",
        "coef": 0.08918185192109941
      },
      {
        "name": "busy52:tabm_standalone",
        "family": "TabM_MLP",
        "coef": 0.03965653418084812
      },
      {
        "name": "busy52:glm_ohk_c0002",
        "family": "LM_GPT2",
        "coef": -0.027947395170622405
      },
      {
        "name": "busy52:glm_ohk_c0005",
        "family": "LM_GPT2",
        "coef": -0.027373332314456128
      },
      {
        "name": "gpw70:glm_ohk2",
        "family": "LM_GPT2",
        "coef": -0.02236206412186703
      },
      {
        "name": "busy52:glm_ohk_g2_c0005_m2",
        "family": "LM_GPT2",
        "coef": -0.02146646305933809
      },
      {
        "name": "busy52:glm_ohk_g2_c0003_s50",
        "family": "LM_GPT2",
        "coef": -0.020908872800533777
      },
      {
        "name": "busy52:tabm_dl",
        "family": "TabM_MLP",
        "coef": -0.01980536957360765
      },
      {
        "name": "pillar:gpt2_rev",
        "family": "LM_GPT2",
        "coef": 0.01980039058699816
      },
      {
        "name": "busy52:rmlp2_dl_hier",
        "family": "TabM_MLP",
        "coef": 0.017218653113941208
      },
      {
        "name": "gpw70:gb_ohk",
        "family": "GBDT",
        "coef": 0.01632843114591805
      },
      {
        "name": "gpw70:gb_bag5",
        "family": "GBDT",
        "coef": 0.0159993277266841
      },
      {
        "name": "gpw70:gbc_bag3",
        "family": "GBDT",
        "coef": 0.014619573569815953
      }
    ]
  },
  {
    "fold": 1,
    "C_L2": 0.1,
    "inner_L2": 0.9523106779034909,
    "alpha_Ridge": 10000.0,
    "inner_Ridge": 0.9499663169958595,
    "inner_Polish": 0.9525035172624319,
    "family_abs_weight": {
      "EBM_GLM": 0.1317138572487544,
      "GBDT": 0.4325709631226412,
      "LM_GPT2": 0.9962708275754983,
      "Proprietary": 0.694190956539846,
      "TabM_MLP": 0.46277154401798476
    },
    "named_weights": {
      "tabm": {
        "busy52:tabm_dl": -0.028514189284646062,
        "busy52:tabm_standalone": 0.10924031435660923
      },
      "uid_v2": {
        "prop:uid_v2": 0.0008144187422453476
      },
      "realmlp": {
        "gpw70:realmlp_dig_s2": -0.0335999415799079,
        "gpw70:realmlp_dig_s100": -0.013916290477061114,
        "gpw70:realmlp_dig": -0.03039613900246066,
        "gpw70:realmlp_te": 0.032975700256537455,
        "gpw70:realmlp_te_s2": -0.00436981185792706,
        "gpw70:realmlp_te_cnt": 0.019837016242754118,
        "gpw70:realmlp_cnt": 0.0040302643569604,
        "gpw70:realmlp_allcat": -0.033607567323630204,
        "gpw70:realmlp_rawdig": 0.012441749193885392,
        "gpw70:realmlp_v1": -0.01615689181865515
      },
      "rmlp": {
        "busy52:rmlp2_dl_hier": 0.04350245196503264,
        "busy52:rmlp2_dl": 0.04346492862558891,
        "busy52:rmlp_dl": -0.006650997089097647,
        "busy52:rmlp_dl_g2": -0.03006729058723089
      },
      "ebm": {
        "gpw70:ebm_i25_bag": 0.026739618966404238,
        "gpw70:ebm_all_bag": -0.009477570274222517,
        "gpw70:ebm_diglead_r11": 0.01496083517018842,
        "gpw70:ebm_diglead_r13": 0.016282776357293773,
        "gpw70:ebm_dig_r7": -0.032509391604235986,
        "busy52:ebm_g2_glminit": -0.01105677933750655,
        "busy52:ebm_dl": -0.014094018335548696,
        "busy52:ebm_dl_g2": 0.0065928672033542055
      },
      "xgb_llr": {
        "prop:xgb_llr": 0.1874378433062312
      },
      "l7": {
        "prop:l7_s2026": 0.0018843313914466342
      },
      "gpt2": {
        "prop:gpt2_med_4ep": 0.006948840623019559,
        "pillar:gpt2_rev": 0.031383480740594334
      },
      "compact": {}
    },
    "top_abs_coefs": [
      {
        "name": "pillar:wide_lm_part12",
        "family": "LM_GPT2",
        "coef": 0.363588111398096
      },
      {
        "name": "pillar:P_super_v5p5_clean",
        "family": "Proprietary",
        "coef": 0.29294220191878995
      },
      {
        "name": "pillar:P_super_v6_clean",
        "family": "Proprietary",
        "coef": 0.23680786559457365
      },
      {
        "name": "pillar:nov08_lm",
        "family": "LM_GPT2",
        "coef": 0.22409612493309522
      },
      {
        "name": "prop:xgb_llr",
        "family": "GBDT",
        "coef": 0.1874378433062312
      },
      {
        "name": "pillar:P_champ",
        "family": "Proprietary",
        "coef": 0.16362647028423702
      },
      {
        "name": "busy52:tabm_standalone",
        "family": "TabM_MLP",
        "coef": 0.10924031435660923
      },
      {
        "name": "busy52:glm_ohk_c0005",
        "family": "LM_GPT2",
        "coef": -0.06746127678817221
      },
      {
        "name": "busy52:glm_ohk_g2_c0005_m2",
        "family": "LM_GPT2",
        "coef": -0.06256612831680974
      },
      {
        "name": "busy52:glm_ohk_c0002",
        "family": "LM_GPT2",
        "coef": -0.062394126701655325
      },
      {
        "name": "busy52:glm_ohk_g2_c0003_s50",
        "family": "LM_GPT2",
        "coef": -0.05953203205593722
      },
      {
        "name": "gpw70:gb_ohk",
        "family": "GBDT",
        "coef": 0.04505424035041867
      },
      {
        "name": "busy52:rmlp2_dl_hier",
        "family": "TabM_MLP",
        "coef": 0.04350245196503264
      },
      {
        "name": "busy52:rmlp2_dl",
        "family": "TabM_MLP",
        "coef": 0.04346492862558891
      },
      {
        "name": "gpw70:glm_ohk2",
        "family": "LM_GPT2",
        "coef": -0.04345533366243467
      },
      {
        "name": "busy52:glm_plain_s50",
        "family": "LM_GPT2",
        "coef": -0.04056116540147708
      },
      {
        "name": "busy52:lgb_g2_plaininit_l4",
        "family": "GBDT",
        "coef": 0.036106320355832965
      },
      {
        "name": "gpw70:gbc_bag3",
        "family": "GBDT",
        "coef": 0.035983071615732096
      },
      {
        "name": "gpw70:realmlp_allcat",
        "family": "TabM_MLP",
        "coef": -0.033607567323630204
      },
      {
        "name": "gpw70:realmlp_dig_s2",
        "family": "TabM_MLP",
        "coef": -0.0335999415799079
      }
    ],
    "top_polish": [
      {
        "name": "pillar:wide_lm_part12",
        "family": "LM_GPT2",
        "coef": 0.6052308866924538
      },
      {
        "name": "pillar:P_super_v6_clean",
        "family": "Proprietary",
        "coef": 0.10795926942482756
      },
      {
        "name": "busy52:tabm_standalone",
        "family": "TabM_MLP",
        "coef": 0.04980199665271708
      },
      {
        "name": "busy52:glm_ohk_c0005",
        "family": "LM_GPT2",
        "coef": -0.030755186861006176
      },
      {
        "name": "busy52:glm_ohk_g2_c0005_m2",
        "family": "LM_GPT2",
        "coef": -0.02852351836736276
      },
      {
        "name": "busy52:glm_ohk_c0002",
        "family": "LM_GPT2",
        "coef": -0.02844510387439253
      },
      {
        "name": "busy52:glm_ohk_g2_c0003_s50",
        "family": "LM_GPT2",
        "coef": -0.027140292287156493
      },
      {
        "name": "gpw70:gb_ohk",
        "family": "GBDT",
        "coef": 0.020539954872315035
      },
      {
        "name": "busy52:lgb_g2_plaininit_l4",
        "family": "GBDT",
        "coef": 0.016460652425743696
      },
      {
        "name": "gpw70:realmlp_allcat",
        "family": "TabM_MLP",
        "coef": -0.015321486076043426
      },
      {
        "name": "gpw70:realmlp_dig",
        "family": "TabM_MLP",
        "coef": -0.0138574153852614
      },
      {
        "name": "busy52:tabm_dl",
        "family": "TabM_MLP",
        "coef": -0.012999445925001291
      },
      {
        "name": "busy52:lgb_g2_glminit_s50",
        "family": "GBDT",
        "coef": 0.009030037873555032
      },
      {
        "name": "gpw70:ebm_diglead_r13",
        "family": "EBM_GLM",
        "coef": 0.007423218968371849
      },
      {
        "name": "gpw70:realmlp_v1",
        "family": "TabM_MLP",
        "coef": -0.007365828970834458
      }
    ]
  },
  {
    "fold": 2,
    "C_L2": 0.01,
    "inner_L2": 0.9511775663546643,
    "alpha_Ridge": 10000.0,
    "inner_Ridge": 0.9487745909922639,
    "inner_Polish": 0.951245664544325,
    "family_abs_weight": {
      "EBM_GLM": 0.08118815067763802,
      "GBDT": 0.4669119020660995,
      "LM_GPT2": 0.9332174521469798,
      "Proprietary": 0.6169991102983201,
      "TabM_MLP": 0.47199794829687935
    },
    "named_weights": {
      "tabm": {
        "busy52:tabm_dl": -0.0217343952172044,
        "busy52:tabm_standalone": 0.10031910881649887
      },
      "uid_v2": {
        "prop:uid_v2": 0.0011666029993030493
      },
      "realmlp": {
        "gpw70:realmlp_dig_s2": -0.03655927362676882,
        "gpw70:realmlp_dig_s100": -0.03405014549362486,
        "gpw70:realmlp_dig": -0.045794640291796106,
        "gpw70:realmlp_te": 0.0209139388632551,
        "gpw70:realmlp_te_s2": -0.013118066989920229,
        "gpw70:realmlp_te_cnt": -0.007279027493006743,
        "gpw70:realmlp_cnt": 0.053179167485691556,
        "gpw70:realmlp_allcat": 0.0037819602306996443,
        "gpw70:realmlp_rawdig": 0.010635389789238699,
        "gpw70:realmlp_v1": 0.01769961952032032
      },
      "rmlp": {
        "busy52:rmlp2_dl_hier": 0.030210360510063476,
        "busy52:rmlp2_dl": 0.0292160687032763,
        "busy52:rmlp_dl": -0.03492226026680894,
        "busy52:rmlp_dl_g2": -0.01258452499870527
      },
      "ebm": {
        "gpw70:ebm_i25_bag": 0.030208929928695394,
        "gpw70:ebm_all_bag": -0.0035487626995983914,
        "gpw70:ebm_diglead_r11": 0.01063918397140275,
        "gpw70:ebm_diglead_r13": 0.010490110152777838,
        "gpw70:ebm_dig_r7": -0.010977142053340595,
        "busy52:ebm_g2_glminit": -0.004775692906220572,
        "busy52:ebm_dl": -0.0080599653331987,
        "busy52:ebm_dl_g2": -0.0024883636324037673
      },
      "xgb_llr": {
        "prop:xgb_llr": 0.16125708911902426
      },
      "l7": {
        "prop:l7_s2026": 0.025925765565399987
      },
      "gpt2": {
        "prop:gpt2_med_4ep": 0.016312068576457152,
        "pillar:gpt2_rev": 0.04112011667297319
      },
      "compact": {}
    },
    "top_abs_coefs": [
      {
        "name": "pillar:wide_lm_part12",
        "family": "LM_GPT2",
        "coef": 0.29257033557480217
      },
      {
        "name": "pillar:P_super_v5p5_clean",
        "family": "Proprietary",
        "coef": 0.24867824836862723
      },
      {
        "name": "pillar:P_super_v6_clean",
        "family": "Proprietary",
        "coef": 0.20975879909289002
      },
      {
        "name": "pillar:nov08_lm",
        "family": "LM_GPT2",
        "coef": 0.19300723553504964
      },
      {
        "name": "prop:xgb_llr",
        "family": "GBDT",
        "coef": 0.16125708911902426
      },
      {
        "name": "pillar:P_champ",
        "family": "Proprietary",
        "coef": 0.15739545983749983
      },
      {
        "name": "busy52:tabm_standalone",
        "family": "TabM_MLP",
        "coef": 0.10031910881649887
      },
      {
        "name": "busy52:glm_ohk_c0002",
        "family": "LM_GPT2",
        "coef": -0.07013632242831927
      },
      {
        "name": "busy52:glm_ohk_c0005",
        "family": "LM_GPT2",
        "coef": -0.061742475179882794
      },
      {
        "name": "busy52:glm_ohk_g2_c0003_s50",
        "family": "LM_GPT2",
        "coef": -0.05900008299713774
      },
      {
        "name": "gpw70:glm_ohk2",
        "family": "LM_GPT2",
        "coef": -0.056964763182596564
      },
      {
        "name": "gpw70:realmlp_cnt",
        "family": "TabM_MLP",
        "coef": 0.053179167485691556
      },
      {
        "name": "busy52:glm_ohk_g2_c0005_m2",
        "family": "LM_GPT2",
        "coef": -0.050893717140757314
      },
      {
        "name": "gpw70:gb_ohk",
        "family": "GBDT",
        "coef": 0.04993205039344974
      },
      {
        "name": "gpw70:gb_bag5",
        "family": "GBDT",
        "coef": 0.04913116963491842
      },
      {
        "name": "gpw70:gbc_bag3",
        "family": "GBDT",
        "coef": 0.047160459984912126
      },
      {
        "name": "gpw70:realmlp_dig",
        "family": "TabM_MLP",
        "coef": -0.045794640291796106
      },
      {
        "name": "pillar:gpt2_rev",
        "family": "LM_GPT2",
        "coef": 0.04112011667297319
      },
      {
        "name": "gpw70:gb_l3",
        "family": "GBDT",
        "coef": 0.04030085260626056
      },
      {
        "name": "busy52:glm_plain_s50",
        "family": "LM_GPT2",
        "coef": -0.038517548391002174
      }
    ],
    "top_polish": [
      {
        "name": "pillar:wide_lm_part12",
        "family": "LM_GPT2",
        "coef": 0.3877784951204126
      },
      {
        "name": "pillar:P_super_v5p5_clean",
        "family": "Proprietary",
        "coef": 0.17715445703205152
      },
      {
        "name": "prop:xgb_llr",
        "family": "GBDT",
        "coef": -0.14762871419337606
      },
      {
        "name": "busy52:lgb_g2_plaininit_l4",
        "family": "GBDT",
        "coef": 0.12106954697216349
      },
      {
        "name": "busy52:tabm_standalone",
        "family": "TabM_MLP",
        "coef": 0.025385230670035355
      },
      {
        "name": "busy52:glm_ohk_c0002",
        "family": "LM_GPT2",
        "coef": -0.01774763297038025
      },
      {
        "name": "busy52:glm_ohk_c0005",
        "family": "LM_GPT2",
        "coef": -0.015623613417930251
      },
      {
        "name": "busy52:glm_ohk_g2_c0003_s50",
        "family": "LM_GPT2",
        "coef": -0.014929665286133894
      },
      {
        "name": "gpw70:glm_ohk2",
        "family": "LM_GPT2",
        "coef": -0.014414638153327825
      },
      {
        "name": "busy52:glm_ohk_g2_c0005_m2",
        "family": "LM_GPT2",
        "coef": -0.012878391410322989
      },
      {
        "name": "gpw70:gbc_bag3",
        "family": "GBDT",
        "coef": 0.011933710031374136
      },
      {
        "name": "busy52:rmlp_dl",
        "family": "TabM_MLP",
        "coef": -0.008836897006466988
      },
      {
        "name": "gpw70:ebm_i25_bag",
        "family": "EBM_GLM",
        "coef": 0.007644213187116611
      },
      {
        "name": "busy52:glm_plain_g2_s50",
        "family": "LM_GPT2",
        "coef": -0.006839047595522258
      },
      {
        "name": "pillar:lgb_subodh",
        "family": "GBDT",
        "coef": 0.005739089659001438
      }
    ]
  },
  {
    "fold": 3,
    "C_L2": 0.01,
    "inner_L2": 0.9522496333422178,
    "alpha_Ridge": 10000.0,
    "inner_Ridge": 0.9500304451773087,
    "inner_Polish": 0.9523765453297635,
    "family_abs_weight": {
      "EBM_GLM": 0.09958021518933216,
      "GBDT": 0.37541293705005474,
      "LM_GPT2": 0.9871388349091218,
      "Proprietary": 0.6187275067697298,
      "TabM_MLP": 0.4269555294172117
    },
    "named_weights": {
      "tabm": {
        "busy52:tabm_dl": -0.0348463290382035,
        "busy52:tabm_standalone": 0.11394537463760289
      },
      "uid_v2": {
        "prop:uid_v2": -0.007526609850183808
      },
      "realmlp": {
      
```

## 5. Yash divergence, w*, tip residual

```json
{
  "divergence": {
    "1mrho_oof_e4": 0.6422411042628084,
    "1mrho_oof_zr_e4": 0.6050024648862617,
    "1mrho_test_e4": 0.6147094770148076,
    "1mrho_test_zr_e4": 0.5824733671311311,
    "1mrho_test_nonzone_e4": 0.691738385287799,
    "ratio_test_over_oof": 0.9571319445839537,
    "ratio_test_over_oof_zr": 0.9627619736072215
  },
  "wstar": {
    "per_fold": [
      {
        "fold": 0,
        "w_star": 0.0,
        "inner_auc": 0.9528307997974067
      },
      {
        "fold": 1,
        "w_star": 0.0,
        "inner_auc": 0.9526400842222653
      },
      {
        "fold": 2,
        "w_star": 0.0,
        "inner_auc": 0.9514131971956775
      },
      {
        "fold": 3,
        "w_star": 0.0,
        "inner_auc": 0.9525808520165701
      },
      {
        "fold": 4,
        "w_star": 0.25,
        "inner_auc": 0.9536326959980456
      }
    ],
    "mean_w": 0.05,
    "oof_auc": 0.9526153612943017,
    "oof_auc_zr": 0.952664389909349,
    "fixed_grid_zr": {
      "0.0": 0.9526685339332593,
      "0.25": 0.9526649815665067,
      "0.35": 0.9526615500792889,
      "0.5": 0.9526544119718452,
      "0.65": 0.9526450301662175,
      "0.75": 0.9526372998422763,
      "1.0": 0.9526136241093358
    }
  },
  "tip_residual": {
    "1mrho_AB_vs_N1v2_e4": 170.9636889056154,
    "1mrho_AB_vs_dual_wstar_e4": 171.00089408546015,
    "1mrho_AB_vs_Part12Zr_e4": 172.25933562884154,
    "1mrho_AB_vs_fullbankTe_e4": 170.9636889056154,
    "tip_minus_proj_dual_std": 0.35826833323024143,
    "tip_proj_coef": 973.8493131949168,
    "own_minus_proj_part12_oof_std": 0.029703905744397904,
    "note": "AB tip is test-only; OOF residual uses Part12 as public tip-proxy. Large test 1-\u03c1 vs dual with ratio_test_over_oof>1.5 \u21d2 Yash private gap."
  }
}
```

## 6. Staged files

```json
{
  "N1v2_own_nested_stack_zr.csv": {
    "sha": "2d00168ebabb4ffb8fcb3edf56de72b49f495e29d7c22f2bca43de8f60412660",
    "meta": "Meta_Step3_FullbankArtifact",
    "oof_auc": 0.9526207252382072,
    "oof_auc_zr": 0.9526697935042275,
    "1mrho_vs_AB_e4": 170.9636889056154,
    "1mrho_vs_part12_e4": 0.5824733671311311
  },
  "N2v2_dual_nested_x_part12_cv_zr.csv": {
    "sha": "35272473148698ddb678fe53b430e2ff8e96b7fbe02adf2823dc103e7bf5e2e7",
    "mean_w_part12": 0.05,
    "oof_auc": 0.9526153612943017,
    "oof_auc_zr": 0.952664389909349,
    "1mrho_vs_AB_e4": 171.00089408546015,
    "1mrho_vs_N1v2_e4": 0.0014722685326429286
  },
  "N3v2_dual_nested_x_part12_50_50_zr.csv": {
    "sha": "8302170d44427609643cebe9d9bb027c3e9ae001d0f60ce36669f26aeb141bab",
    "1mrho_vs_AB_e4": 171.46783089298845,
    "1mrho_vs_N2v2_e4": 0.11861772732491005
  },
  "H1v2_private_hedge_50AB_50N2.csv": {
    "sha": "f770efa1fc746c69f03151305fb9c204e3ed3e68fcdf7880f33d2bb1669d88cd",
    "w_AB": 0.5,
    "1mrho_vs_AB_e4": 42.944049205509984,
    "1mrho_vs_N2v2_e4": 43.183449539262455
  },
  "H2v2_private_hedge_70AB_30N2.csv": {
    "sha": "a4cb983c497e997eb2b53517ba8eddff9cac39a292309cdcf0f7c1acd6e8e15c",
    "w_AB": 0.7,
    "1mrho_vs_AB_e4": 15.429625128106084,
    "1mrho_vs_N2v2_e4": 84.32209472071573
  }
}
```

## 7. STOP

Step 5b complete. **Awaiting human audit.** Reply `PROCEED TO STEP 6` to continue.
