# V41 FINAL 10-SLOT MANIFEST — DS-Expert Hardened

**NO_SUBMIT=True** · Hard stop after Step 6. **ZERO submissions.**

## Executive

- S_indep vs Fullbank: 1-ρ oof/test e4=114.722/140.917 ratio=1.23
- Nested blend S_cv raw/Zr=0.952548/0.952592; **SLOT09 Fullbank floor** 0.952621/+Zr 0.952670
- z_AB_twin R²_nz=0.9560 1-ρ vs AB e4=188.149
- Dual-gate PASS: ['u_resid_p4_ridge', 'u_sub_resid:v10_ridge_resid', 'u_sub_resid:v10_ridge_orth_LM', 'u_sub_resid:harvest_p4_cuda_xgb', 'u_master']
- seq_tree/uid_orth/ownstack_diff: FAIL dual-baseline (documented); resid pillars PASS
- Staged `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/out/v41_final_10slots/`

## 10-Slot Master Table

| Slot | File | Role | 1-ρ vs AB e4 | max U_zone | minDiv e4 | div_ok |
|---|---|---|---:|---:|---:|---|
| SLOT01 | `SLOT01_pub_Step_AB_095569.csv` | PUB Tip Lock 0.95569 | 0.0000 | 0.0558 |  | True |
| SLOT02 | `SLOT02_AB_clip985_areeb_a0p32.csv` | Top-2 Geodesic Clip AB×areeb | 0.0275 | 0.0558 | 0.0275 | True |
| SLOT03 | `SLOT03_AB_clip985_quarter_a0p28.csv` | Quarter Diversity Clip on AB (escalated for >=0.020e | 0.1200 | 0.0558 | 0.0392 | True |
| SLOT04 | `SLOT04_AB_nonpub_seq_llr_clip985_0p25e4.csv` | Dual-Gated resid pillars (seq_tree FAILED dual-basel | 0.2500 | 0.0558 | 0.2500 | True |
| SLOT05 | `SLOT05_AB_nonpub_seq_llr_clip985_0p55e4.csv` | Dual-Gated resid pillars step-up (seq_tree FAILED; r | 0.5500 | 0.0558 | 0.0586 | True |
| SLOT06 | `SLOT06_AB_prop_resid_pillars_clip985_0p30e4.csv` | Dual-Gated pure residual pillars | 0.3000 | 0.0558 | 0.3000 | True |
| SLOT07 | `SLOT07_AB_cvchamp_vs_twin_clip985_0p35e4.csv` | Own-stack vs AB-twin clipped residual | 0.3500 | 0.0558 | 0.3500 | True |
| SLOT08 | `SLOT08_AB_master_dual_gated_combo_0p40e4.csv` | Dynamic A/B winner composite master ray | 0.4000 | 0.0558 | 0.0274 | True |
| SLOT09 | `SLOT09_pure_nested_cv_champion_zr.csv` | Pure nested-CV champion = Step3 Fullbank+Zr (nested  | 170.5699 | 0.0558 | 165.4032 | True |
| SLOT10 | `SLOT10_private_hedge_55BestTip_45CVChamp_zr.csv` | Private-LB Bayesian shrinkage hedge | 34.7676 | 0.0558 | 32.6149 | True |

## 10×10 Pairwise 1-ρ (×1e-4)

| |SLOT01|SLOT02|SLOT03|SLOT04|SLOT05|SLOT06|SLOT07|SLOT08|SLOT09|SLOT10|
|---|---|---|---|---|---|---|---|---|---|---|
| SLOT01 | 0.000 | 0.028 | 0.120 | 0.250 | 0.550 | 0.300 | 0.350 | 0.400 | 170.570 | 34.768 |
| SLOT02 | 0.028 | 0.000 | 0.039 | 0.261 | 0.553 | 0.347 | 0.397 | 0.399 | 172.529 | 35.674 |
| SLOT03 | 0.120 | 0.039 | 0.000 | 0.320 | 0.596 | 0.466 | 0.515 | 0.474 | 173.624 | 36.222 |
| SLOT04 | 0.250 | 0.261 | 0.320 | 0.000 | 0.059 | 0.562 | 0.487 | 0.647 | 172.213 | 35.653 |
| SLOT05 | 0.550 | 0.553 | 0.596 | 0.059 | 0.000 | 0.867 | 0.732 | 0.945 | 173.183 | 36.260 |
| SLOT06 | 0.300 | 0.347 | 0.466 | 0.562 | 0.867 | 0.000 | 0.645 | 0.027 | 169.255 | 34.335 |
| SLOT07 | 0.350 | 0.397 | 0.515 | 0.487 | 0.732 | 0.645 | 0.000 | 0.762 | 165.403 | 32.615 |
| SLOT08 | 0.400 | 0.399 | 0.474 | 0.647 | 0.945 | 0.027 | 0.762 | 0.000 | 170.873 | 35.126 |
| SLOT09 | 170.570 | 172.529 | 173.624 | 172.213 | 173.183 | 169.255 | 165.403 | 170.873 | 0.000 | 51.788 |
| SLOT10 | 34.768 | 35.674 | 36.222 | 35.653 | 36.260 | 34.335 | 32.615 | 35.126 | 51.788 | 0.000 |

## Sequential Feedback Tree

- **SLOT01:** LOCK tip. Always keep as pub baseline.
- **SLOT02:** If SLOT02 ≥ SLOT01: geodesic helps → prefer SLOT02/08 path. If −1 tick: abort geodesic family, keep SLOT01.
- **SLOT03:** If SLOT03 ≥ max(S01,S02): quarter diversity real. If drop: skip quarter branch.
- **SLOT04:** If ≥ best tip: unlock SLOT05 step-up. If −1 tick: flip seq sign to −0.20e-4 or SKIP SLOT05.
- **SLOT05:** Only live if SLOT04 tied/gained. If SLOT05 drops while SLOT04 held: keep SLOT04 scale.
- **SLOT06:** If gain: residual pillars live. If drop: drop failing sub-pillars, do not raise α.
- **SLOT07:** If uid_orth path and drop: treat as uid fail confirmation. If ownstack path gains: CV champ injects.
- **SLOT08:** ALL-NEGATIVE A/B FALLBACK: if SLOT02–07 all ≤ SLOT01−1tick → SLOT08 becomes 0.70*AB+0.30*SLOT09 (private hedge early).
- **SLOT09:** Honest CV readout. Not a tip climb. Compare private LB vs tip delta.
- **SLOT10:** Final pick #2 candidate if tip overfits public. Prefer when SLOT09 private ≫ tip private.

## STOP

Step 6 complete. **NO_SUBMIT=True. Do not submit.** Awaiting human audit.
