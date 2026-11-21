# GM v10 Hardened — 0.95369 Breaker Report (`NO_SUBMIT=True`)

Generated: 2026-11-12T11:40:56Z · elapsed=83.9s · peak RSS≈1094 MB (over 700 MB budget during Ridge quadratic materialization; still finished <2 min)

**Live note:** `smugcqcy2oqzeg` left untouched per hard-rule (replica ~53/60 at report time). Only `rsoju37zhpzaae` was in-scope for terminate (already 404).

## Step 0 — Pod

- Requested terminate: `rsoju37zhpzaae` → **ALREADY_GONE_404**
- Final spend est: ~0.45–0.55 (pod died earlier; remount was smugcqcy2oqzeg) · active $/hr for rsoju: **$0.00**
- Untouched: `smugcqcy2oqzeg`, `lheroidmuab1j4`, `fujymbm9nkwulv`
- Rationale: Public goodpjw compact-lm-seed7 now ships lm7_*_5fold.npy — redundant L7 retrain on rsoju not needed. User hard-rule: never touch smugcqcy2oqzeg.

## Step 1 — daoviet `0.95368` / goodpjw `0.95354` forensics

### daoviet

- Header verified LB: **0.95351**
- Ablation table claims: Public **0.95368** / Local CV **0.95378**
- **Verdict:** 0.95368/0.95378 appear ONLY in a marketing ablation DataFrame (cell 8). Executable path prefers prebaked submission_095351.csv (Public LB 0.95351) OR reconstructs with 1-fold lm7_test_f0 + 0.06*(L7-F) + 0.05*(OUR-F). No SHA-locked 0.95368 file exists in daoviet/s6e11-gp-dvae-orthogonal-apex.
- Reconstruct formula (fallback): `B = A + 0.60*(A-F) + 0.12*(LM-F) + 0.06*(L7_f0-F); score = B - 0.05*(MLP-F) - 0.025*(GB-F) + 0.05*(OUR-F)`
- Dataset files: ['oof_ours.npy', 'submission_095351.csv', 'test_ours.npy']
- Combines with goodpjw 0.95354? No. daoviet apex (0.95351) is an earlier ladder rung; goodpjw 0.95354 adds 5-fold L7F @0.09 + both LMs + GPD@0.05 + 0.06*(G26-MS). daoviet's claimed 0.95368 is not a composable verified tensor.

### goodpjw 0.95354

- Formula: `s352 = A + 0.6*(A-F) + 0.09*(LM-F) + 0.09*(L7F-F) + 0.05*(GPD-F) - 0.05*(MLP-F) - 0.025*(GB-F); s354 = logit_rank(s352) + 0.06*(G26-MS)`
- L7: **lm7_test_5fold.npy (NOT lm7_test_f0)**
- G26 test: **gpt2_s2026_test_f0.npy — Fold-0 ONLY (README: other folds still scoring)**

## Step 2 — Asset & split audit

| asset | shape | OOF AUC | fold AUCs | notes |
|---|---|---:|---|---|
| L7F_oof | [699960] | 0.949448 | [0.94754, 0.94977, 0.95517, 0.94907, 0.94568] | pattern_ok=True |
| L7F_te | [299984] | — | — | sha=68639fec79f081e9 |
| mega_oof | [699960] | 0.950428 | [0.94851, 0.95068, 0.95608, 0.9506, 0.94638] | pattern_ok=True |
| mega_te | [299984] | — | — | sha=3168779be644a530 |
| G26_oof | [699960] | 0.950140 | [0.94919, 0.95037, 0.95524, 0.95021, 0.94586] | pattern_ok=True |
| G26_te | [299984] | — | — | sha=40e08058ed7632ee |
| MS_oof | [699960] | 0.950027 | [0.94888, 0.95032, 0.95531, 0.94987, 0.94585] | pattern_ok=True |
| MS_te | [299984] | — | — | sha=6c27aff7be744f7a |
| GPD_oof | [699960] | 0.946489 | [0.94473, 0.94769, 0.95146, 0.94666, 0.9421] | pattern_ok=True |
| GPD_te | [299984] | — | — | sha=96f3bdde816e6044 |
| MLP_oof | [699960] | 0.951223 | [0.94992, 0.95161, 0.95608, 0.95186, 0.94676] | pattern_ok=True |
| MLP_te | [299984] | — | — | sha=035d6544d7c03108 |
| GB_oof | [699960] | 0.951506 | [0.95043, 0.95159, 0.95627, 0.95239, 0.94699] | pattern_ok=True |
| GB_te | [299984] | — | — | sha=7d54e5f17e0707d8 |
| A | [299984] | — | — | sha=12b525b84f29e32d |
| F | [299984] | — | — | sha=f4a7a4f8636c99fa |

- **G26:** {'oof': '5-fold complete (699960) — folds.npy aligned', 'test': 'Fold-0 ONLY (gpt2_s2026_test_f0.npy) — √5× higher test fold noise vs 5-fold', 'block_near_5f_gbdt': True, 'reason': 'Test vector is 1-fold; do not treat G26_te as 5-fold-complete feature'}
- **GPD:** {'has_5fold_oof': True, 'oof_auc': 0.946489033767143, 'fold_aucs': [0.94473, 0.94769, 0.95146, 0.94666, 0.9421], 'fold_pattern_ok': True, 'aligned_with_folds_npy': True, 'caution': 'Allow as ladder component; exclude from Stage-B hygiene if want pure SKF42 proven bank only — pattern OK so KEEP optional'}

### s354 reconstruction

- Spearman dist(s354, S7) = `3.722933e-04`
- cos(Δ_pub, Δ_priv) = `-0.0949`
- s354 OOF proxy AUC = `0.952482` folds=[0.951499, 0.952588, 0.957364, 0.952539, 0.948426]
- G26 test Fold-0 only: **True**

## Step 3 — G26 → gpt2-medium 5f

- gpt2_med_5f OOF AUC=0.948238 folds=[0.94583, 0.94877, 0.95374, 0.94867, 0.94421]
- Best OOF variant: **pub_G26_only**

| variant | OOF AUC | pos/5 | Δe-5 vs pub |
|---|---:|---:|---|
| pub_G26_only | 0.952482 | 0 | [0.0, 0.0, 0.0, 0.0, 0.0] |
| med5f_only_0p06 | 0.952442 | 0 | [-11.42, -2.3, -2.01, -2.38, -2.6] |
| blend_0p04G26_0p04med | 0.952463 | 0 | [-6.66, -0.7, -1.3, -0.31, -1.28] |
| blend_0p03G26_0p06med | 0.952448 | 0 | [-10.33, -1.58, -2.36, -0.9, -2.2] |
| med5f_only_0p09 | 0.952419 | 0 | [-17.0, -3.69, -4.17, -3.02, -4.55] |

Note: G26 test is Fold-0 only; preferred test blend keeps small G26 + dominant gpt2_med_5f

## Step 4 — Stage-B × L7_5fold

- KMAT_68 matrix not on disk — substituted 8 oof-bank shallow LGB/XGB 5f members + L4.6C/P4
- LGB features: 22 · excluded: ['G26_test (1-fold)', 'raw lm7_test_f0']
- δ_v10 disp=1.0000 KS=0.000182

| w | pos/5 | mean Δe-5 | OOF AUC |
|---:|---:|---:|---:|
| 0.02 | 4 | 0.0 | 0.952482 |
| 0.04 | 4 | 0.01 | 0.952482 |
| 0.06 | 4 | 0.01 | 0.952482 |
| 0.08 | 4 | 0.01 | 0.952482 |

## Step 5 — Candidates (`NOSUBMIT`)

### slot1
- **path:** `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v10_slot1_s354pub_plus_clean5f_stageB_0p12e4_NOSUBMIT.csv`
- **target_dist:** `1.2e-05`
- **got_dist:** `1.1999999957823526e-05`
- **weight:** `2.550652385544888`
- **mean:** `0.5`
- **ks_vs_s354:** `0.0`
- **spearman_vs_s354:** `0.9999880000003509`
- **sha16:** `496496b26d7551c3`

### slot2
- **path:** `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v10_slot2_s354pub_plus_clean5f_stageB_0p28e4_NOSUBMIT.csv`
- **target_dist:** `2.8e-05`
- **got_dist:** `2.8000000145245174e-05`
- **weight:** `3.898101064459772`
- **mean:** `0.5`
- **ks_vs_s354:** `0.0`
- **spearman_vs_s354:** `0.9999720000013361`
- **sha16:** `eebab25d9cc0933f`

### slot3
- **path:** `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v10_slot3_midpoint_s354pub_and_S7ours5fL7_NOSUBMIT.csv`
- **mean:** `0.5`
- **ks_vs_s354:** `0.0`
- **spearman_vs_s354:** `0.9999081246470461`
- **spearman_vs_S7:** `0.9996873535834273`
- **sha16:** `5c46aa432c617c36`
- **note:** `rank-logit midpoint s354_pub ↔ clean S7_5fL7 (no areeb−S7 raw delta)`

### slot4
- **path:** `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v10_slot4_daoviet_095368_verified_synthesis_NOSUBMIT.csv`
- **mean:** `0.5`
- **ks_vs_s354:** `0.0`
- **spearman_vs_s354:** `0.9999985241134468`
- **sha16:** `b8ab1c0df84ad21f`
- **note:** `0.95368 NOT verified — slot uses verified daoviet submission_095351 + mild GPD orth on s354_pub. Do NOT treat as a 0.95368 clone.`

### OOF s354 + private

| w | AUC | folds |
|---:|---:|---|
| 0.0 | 0.952482 | [0.951499, 0.952588, 0.957364, 0.952539, 0.948426] |
| 0.02 | 0.952482 | [0.951499, 0.952588, 0.957364, 0.952539, 0.948426] |
| 0.04 | 0.952482 | [0.951499, 0.952588, 0.957364, 0.952539, 0.948426] |
| 0.06 | 0.952482 | [0.951499, 0.952588, 0.957364, 0.952539, 0.948426] |

### Empirical LB surface (cached)

| name | LB | dist→s354 | dist→areeb |
|---|---:|---:|---:|
| Today_S5 | 0.95335 | 1.874e-03 | 1.541e-03 |
| Today_S4 | 0.95334 | 1.634e-03 | 1.318e-03 |
| Today_S1 | 0.95333 | 1.766e-03 | 1.389e-03 |
| Today_S2 | 0.95333 | 1.762e-03 | 1.386e-03 |
| St9.5 | 0.95332 | 1.668e-03 | 1.312e-03 |
| St9 | 0.95330 | 1.572e-03 | 1.242e-03 |
| St8 | 0.95329 | 1.480e-03 | 1.179e-03 |
| ANTI | 0.95329 | 1.963e-03 | 1.587e-03 |
| Today_S3 | 0.95329 | 1.654e-03 | 1.332e-03 |
| soft | 0.95328 | 1.570e-03 | 1.259e-03 |
| St7 | 0.95327 | 1.401e-03 | 1.128e-03 |
| St6 | 0.95325 | 1.338e-03 | 1.095e-03 |

**Contamination guard:** Never added raw (z_S7 - z_areeb); L7F 5fold used throughout

---
`NO_SUBMIT=True` — no Kaggle upload performed.
