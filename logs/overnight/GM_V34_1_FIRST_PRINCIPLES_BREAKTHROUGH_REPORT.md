# GM V34.1 FIRST-PRINCIPLES BREAKTHROUGH REPORT

- NO_SUBMIT: `True`
- elapsed_s: 166.9
- peak_rss_gb: 2.925 (cap 2.8)
- z_wide_oof AUC: 0.952565

## Anchor Hygiene

- delta_cat: {"exists": true, "r2_vs_z_wide": 0.9982031617477887, "beta": -1.8121308038809136, "nested_delta_e4": -0.6072040524205402, "raw_disp": 2.13096959426338, "keep": false, "reason": "drop_preblend_r2"}
- proprietary:
  - {'name': 'v10_ridge_orth', 'exists': True, 'KS_pre': 5.497102520279906e-05, 'disp': 1.0000107298065877, 'r2_wide': 0.0009197086890787562}
  - {'name': 'P4_L4.2', 'exists': True, 'KS_pre': 0.12447094467740538, 'disp': 1.0000492846473163, 'r2_wide': 0.000244247545044476}
  - {'name': 'P4_L4.3', 'exists': True, 'KS_pre': 0.09191723831000842, 'disp': 0.9949356022509231, 'r2_wide': 0.003531963617407868}
  - {'name': 'P4_L4.3_ext', 'exists': True, 'KS_pre': 0.06207821732815216, 'disp': 1.0002441271089704, 'r2_wide': 0.017364626362671116}
  - {'name': 'P4_L4.5', 'exists': True, 'KS_pre': 0.13636155674688155, 'disp': 1.0001023898495451, 'r2_wide': 0.002525337126343925}
  - {'name': 'B_xgb_llr', 'exists': True, 'KS_pre': 0.0010571961441616146, 'disp': 1.0000107298043452, 'r2_wide': 0.9903329558709525}
  - {'name': 'gpt2_3ep', 'exists': True, 'KS_pre': 9.206156080621142e-05, 'disp': 1.0000104921395305, 'r2_wide': 0.9374178275088191}
  - {'name': 'gpt2_med_4ep', 'exists': True, 'KS_pre': 0.24871029580803877, 'disp': 1.000010729823268, 'r2_wide': 0.9274195760281391}

## Engine Gates

| module | solo_auc | Δwide e-4 | Δanchor e-4 | outer | sub | t | p | disp | KS | pass |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| E1 | 0.737067 | -2.01 | — | 0/5 | 1/25 | -1.47 | 1 | 1.073 | 0.4174 | False |
| E2 | 0.913635 | 0.00 | — | 0/5 | 0/25 | 0.00 | 1 | 0.998 | 0.0022 | False |
| E3 | 0.952826 | 1.19 | -0.0068370545482920875 | 5/5 | 24/25 | 8.58 | 0.03125 | 0.968 | 0.0190 | False |
| E4 | 0.948147 | -0.94 | — | 0/5 | 3/25 | -1.36 | 1 | 1.119 | 0.0628 | False |
| COMBO | 0.949287 | -0.44 | -0.8179096663467611 | 2/5 | 8/25 | -0.74 | 0.625 | 1.109 | 0.0263 | False |

## E1 Forensics

- TierB match_ge: {12: 3426018, 15: 238540, 18: 209}
- pair_touches: 8000229
- P(yi=1|yj=1)=0.0258 P(yi=1|yj=0)=0.0198
- max|ac(y)|=0.0034 max|ac(r)|=0.0029 fft_z=6.76
- id_norm adversarial AUC=0.4985
- ieee: {"amount_usd": {"train": {"mean_len": 4.906917538145037, "mean_frac": 1.9187982170409739}, "test": {"mean_len": 4.907848418582325, "mean_frac": 1.9189023414582111}}, "account_balance_usd": {"train": {"mean_len": 6.546896965540888, "mean_frac": 1.886954968855363}, "test": {"mean_len": 6.547392527601472, "mean_frac": 1.8871839831457677}}, "velocity_score": {"train": {"mean_len": 3.790742328133036, "mean_frac": 1.0}, "test": {"mean_len": 3.7910555229612246, "mean_frac": 1.0000033335111207}}}

## E2

- n_active=3525 disp=0.9980670324701362 KS=0.0022260328501784166

## E4 Twilight Attribution

{
  "0": {
    "twilight": -0.026736704208724205,
    "high_tail": -0.06523831152630322
  },
  "1": {
    "twilight": -0.02091500121927503,
    "high_tail": -0.05169174569670987
  },
  "2": {
    "twilight": -0.03704135309694123,
    "high_tail": -0.08156637925056054
  },
  "3": {
    "twilight": -0.024900535114798727,
    "high_tail": -0.07244878270350519
  },
  "4": {
    "twilight": -0.0289774918548088,
    "high_tail": -0.04564205801788723
  }
}

## Staging

- passing: []
- cos(combo, SLOT1/0.95537)=0.9070
- NOSUBMIT CSVs: []

## Verdict

- **No engine cleared the strict 25-subfold gate** → no NOSUBMIT CSVs staged.
- **E3 near-miss (closest):** 5/5 outer, **24/25** subfolds, Δ=+1.19e-4, t=8.58 — blocked by Wilcoxon `p=0.031` (need `<0.005`) and **KS=0.019** (need `<0.015`); `raw_disp=0.968` OK. Antithetic Stage-B is the only positive direction; needs tighter test-PCHIP / more seeds before promotion.
- `delta_cat` **DROPPED** (R²=0.9982, raw_disp=2.13) — pre-blended wide wrapper.
- `F_cat_obliv` / `R2_v178` **missing on disk**; high-R² LM twins excluded from `z_anchor` design.
- Peak RSS **2.925 GB** (brief E2 SAG spike over 2.8 GB cap); wall **167.7s** (<3.5m). Live CPU/RSS logged every line.

**STRICT NO_SUBMIT=True — 0 submits, 0 pods, 0 Kaggle pulls.**
