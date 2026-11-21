# GM V34.3 PRINCIPAL-DS BREAKTHROUGH REPORT

- NO_SUBMIT: `True`
- elapsed_s: 197.3
- peak_rss_gb: 2.122 (cap 2.4)
- z_wide_oof AUC: 0.952565

## Part 1 — Twin Recon + FFT

- schema_verdict: Only uditjain13-style orig20k locally shares the 24-col schema; public searches returned mirrors/EDA of the same 20k — no larger twin schema detected in this recon.
- local_twins: ['/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/original/credit_card_fraud_2026.csv', '/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/data/test.csv', '/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/data/train.csv']
- fft: {"k_star": 130522, "T": 5.362774091724001, "fft_z": 0.1954535875365458, "perm_p": 0.3225806451612903, "trig_delta_e4": -0.32449909153076106}

### Search snippets

- `kaggle` q=`credit card fraud 2026` ok=True: `Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5) ref,title,size,lastUpd`
- `kaggle` q=`fraud 2026` ok=True: `Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5) ref,title,size,lastUpd`
- `kaggle` q=`scam attempt` ok=True: `Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5) No datasets found `
- `kaggle` q=`cvv retry` ok=True: `Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5) No datasets found `
- `kaggle` q=`merchant risk` ok=True: `Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5) ref,title,size,lastUpd`
- `kaggle` q=`udit` ok=True: `Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5) ref,title,size,lastUpd`
- `kaggle` q=`playground s6e11` ok=True: `Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5) ref,title,size,lastUpd`
- `hf` q=`credit card fraud 2026` ok=True: `[]`
- `hf` q=`is_ai_generated_scam_attempt` ok=True: `[]`
- `hf` q=`cvv_retry_count` ok=True: `[]`
- `hf` q=`uditjain` ok=True: `[{"_id":"6a63f0dab3dc5ef0b25790ec","id":"uditjain/adaption-financial-math-qa","author":"uditjain","cardData":{"annotations_creators":[],"language":["en"],"langu`
- `hf` q=`s6e11` ok=True: `[]`

## Part 2 — z_prop_nested

- clean pillars: ['v10_ridge_resid_oof', 'v10_ridge_pure_fe_oof', 'v10_ridge_bulsah_only_oof', 'P4_L4.2_pure5f_oof', 'P4_L4.3_pure5f_oof', 'P4_L4.5_pure5f_oof', 'P4_L4.2_ext_oof', 'P4_L4.3_ext_oof', 'P4_L4.1_oof', 'P4_L4.3_oof', 'P4_L4.4_oof', 'harvest_p4_cuda_xgb_oof', 'd1_v4_oof']
- gate: {'name': 'z_prop_nested', 'outer': '3/5', 'sub': '11/25', 'mean_e4': 0.663362226026587, 't25': 1.6510471862741702, 'p25': 0.01545400682621478, 'disp': 0.9407885783941311, 'F2': 0.03016781806324583, 'KS': 0.02787483698275295, 'pass': False, 'solo': 0.942879640897576, 'sub_lifts_e4': [4.3175882230039875, -0.16609093649422313, 4.595170388809189, -0.40133851059942494, -0.04744505648468511, 0.613139438463417, 1.5716548897326543, 5.680184385490339, -5.135732917621594, 2.296639873827777, 1.0850239683191099, 1.0718298285050665, 0.5771185559400038, 0.616007946502295, 0.055106938506854775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

## Engine Gates

| module | solo | Δwide e-4 | outer | sub | t25 | p25 | disp | F2 | pass |
|---|---:|---:|---|---|---:|---:|---:|---:|---|
| E1 | 0.377331 | 0.00 | 0/5 | 0/25 | 0.00 | nan | 1.000 | 0.0000 | False |
| E2 | 0.494036 | -1.57 | 0/5 | 1/25 | -2.70 | 0.9891 | 1.000 | 0.0012 | False |
| E3A | 0.548973 | 0.00 | 0/5 | 0/25 | 0.00 | nan | 1.000 | 0.0000 | False |
| E3B | 0.529087 | 0.00 | 0/5 | 0/25 | 0.00 | nan | 1.000 | 0.0000 | False |
| z_final | 0.952209 | 0.60 | 3/5 | 13/25 | 1.47 | 0.006227 | 0.918 | 0.0114 | False |

## Part 5 Staging

- Δβ(u_win in MID37 | z_wide)=0.374198
- cos(u_win, loser_ray)=-0.1812 surv_norm=1.000
- staging_ok=False csvs=[]

## Verdict

- **No module cleared the strict V34.3 gate** (p25<0.002, t25≥3, ≥20/25, F2/disp). No CSVs staged.

**STRICT NO_SUBMIT=True — 0 submits, 0 pods.**
