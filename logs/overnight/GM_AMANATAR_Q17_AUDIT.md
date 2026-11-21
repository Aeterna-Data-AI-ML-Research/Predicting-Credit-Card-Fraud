# GM Track-2 Audit: amanatar Q17 + detective + G26
**UTC:** 2026-11-12T16:32:36.735189+00:00  
**`NO_SUBMIT=True`** — 0 uploads  
**Elapsed:** 516.57s

## Executive verdict
- **`anchor_095370` cleared the dual LOFO gate** (S7m **5/5**, simple **4/5**, `d0>0`). Orth residual meanΔ ≈ **+0.035e-5** on S7m. **All 16 Q17 member streams failed** the dual gate (several hit S7m 5/5 alone — `gbn_bag5_w512`, `gb_bag5` — but simple ≤3/5).
- **HOLD all 4 submits** (`NO_SUBMIT=True`): anchor is already the public 0.95370 apex; do **not** burn a cand2↔anchor blend without a dedicated promotion plan.
- **F2−F4 > 0.004 on every stream** including known-clean public OOFs → structural fold difficulty, **not** cross-fold leak.
- **G26 live `test_5fold`:** **ABSENT** (still oof_5fold + test_f0 only).
- **Q17 `test_gpt2_seed2026` ≡ G26 `test_f0`** (`1−ρ = 0`) — not a full 5-fold test tensor.
- **Detective:** `model.pt` (~475MB) + `config.json`; data = orig20k CSV only — no OOF/test bank.

## Live dataset file lists
### amanatar/s6e11-rank1-q17-meta-bundle
```
name                       size  creationDate                
---------------------  --------  --------------------------  
rank1_q17_streams.npz  64574677  2026-11-12 14:04:47.476000
```
### aphoristicartist/s6e11-detective-weights
```
name              size  creationDate                
-----------  ---------  --------------------------  
config.json        961  2026-11-10 23:11:40.103000  
model.pt     497805283  2026-11-10 23:11:51.122000
```
### aphoristicartist/s6e11-detective-data
```
name                           size  creationDate                
--------------------------  -------  --------------------------  
credit_card_fraud_2026.csv  2754040  2026-11-10 15:55:58.549000
```
### goodpjw2008/s6e11-target-first-gpt2-seed-2026
```
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000
```

## Bundle contents (`rank1_q17_streams.npz`)
- Keys (38): `['y_train', 'folds', 'oof_anchor_095370', 'test_anchor_095370', 'test_anchor_prob', 'stream_names', 'oof_megayak_5f_lm', 'test_megayak_5f_lm', 'oof_masaya_gpt2', 'test_masaya_gpt2', 'oof_lm7_5fold', 'test_lm7_5fold', 'oof_gpt2_seed2026', 'test_gpt2_seed2026', 'oof_tabpfn_slimypunk_tp1', 'test_tabpfn_slimypunk_tp1', 'oof_sadam_tabpfn_100k_tp3', 'test_sadam_tabpfn_100k_tp3', 'oof_yekenot_realmlp', 'test_yekenot_realmlp', 'oof_alex_glm_gen', 'test_alex_glm_gen', 'oof_daoviet_gpdvae', 'test_daoviet_gpdvae', 'oof_gbn_bag5_w512', 'test_gbn_bag5_w512', 'oof_sameer_glm10f', 'test_sameer_glm10f', 'oof_sadam_v6_gb_xgb', 'test_sadam_v6_gb_xgb', 'oof_sadam_v6_gb_lgb', 'test_sadam_v6_gb_lgb', 'oof_sadam_v4_cat', 'test_sadam_v4_cat', 'oof_gb_bag5', 'test_gb_bag5', 'oof_megayak_shallow_tree', 'test_megayak_shallow_tree']`
- `stream_names`: `['megayak_5f_lm', 'masaya_gpt2', 'lm7_5fold', 'gpt2_seed2026', 'tabpfn_slimypunk_tp1', 'sadam_tabpfn_100k_tp3', 'yekenot_realmlp', 'alex_glm_gen', 'daoviet_gpdvae', 'gbn_bag5_w512', 'sameer_glm10f', 'sadam_v6_gb_xgb', 'sadam_v6_gb_lgb', 'sadam_v4_cat', 'gb_bag5', 'megayak_shallow_tree']`
- Folds match `StratifiedKFold(5, shuffle=True, random_state=42)` ✓
- **No README/md/script** inside the zip — only the NPZ.

## Test-vector Spearman distances (`1−ρ`) vs ATH refs
| stream | A_095340 | F_095330 | z_S7_ours | cand2 | areeb_095347 | s354 |
|---|---:|---:|---:|---:|---:|---:|
| anchor_095370 | 0.001659 | 0.001798 | 0.001046 | 0.00099 | 0.001158 | 0.001072 |
| megayak_5f_lm | 0.028099 | 0.026748 | 0.021591 | 0.02159 | 0.021209 | 0.021754 |
| masaya_gpt2 | 0.036373 | 0.035114 | 0.030145 | 0.030385 | 0.029936 | 0.030109 |
| lm7_5fold | 0.040173 | 0.03896 | 0.03298 | 0.032934 | 0.032578 | 0.033135 |
| gpt2_seed2026 | 0.038148 | 0.036865 | 0.032164 | 0.031795 | 0.031855 | 0.032222 |
| tabpfn_slimypunk_tp1 | 0.0298 | 0.027947 | 0.029442 | 0.029298 | 0.029536 | 0.029458 |
| sadam_tabpfn_100k_tp3 | 0.034597 | 0.032734 | 0.03446 | 0.034276 | 0.034596 | 0.034496 |
| yekenot_realmlp | 0.025211 | 0.023314 | 0.024371 | 0.024268 | 0.024597 | 0.02429 |
| alex_glm_gen | 0.01915 | 0.016838 | 0.019959 | 0.019879 | 0.020191 | 0.019789 |
| daoviet_gpdvae | 0.05827 | 0.056288 | 0.055484 | 0.055401 | 0.055051 | 0.055707 |
| gbn_bag5_w512 | 0.014765 | 0.012865 | 0.015987 | 0.015904 | 0.016437 | 0.015817 |
| sameer_glm10f | 0.014928 | 0.013021 | 0.016058 | 0.015939 | 0.0163 | 0.016001 |
| sadam_v6_gb_xgb | 0.016515 | 0.015259 | 0.017046 | 0.01694 | 0.01762 | 0.016786 |
| sadam_v6_gb_lgb | 0.016868 | 0.015395 | 0.017308 | 0.017209 | 0.017771 | 0.017108 |
| sadam_v4_cat | 0.026921 | 0.025011 | 0.025522 | 0.025413 | 0.025351 | 0.025612 |
| gb_bag5 | 0.011974 | 0.010236 | 0.01323 | 0.013129 | 0.013706 | 0.013042 |
| megayak_shallow_tree | 0.012311 | 0.010711 | 0.010149 | 0.010074 | 0.010162 | 0.010171 |
| anchor_095370_TEST_ONLY | 0.001659 | 0.001798 | 0.001046 | 0.00099 | 0.001158 | 0.001072 |

## OOF/Test pairs — solo AUC & fold profile
| stream | solo | F0 | F1 | F2 | F3 | F4 | F2−F4 | leak?>0.004 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| anchor_095370 | 0.9525514 | 0.9517749 | 0.9525318 | 0.9572653 | 0.9528037 | 0.948381 | 0.0088843 | True |
| megayak_5f_lm | 0.9504517 | 0.9485056 | 0.9506818 | 0.956085 | 0.9506043 | 0.9463809 | 0.0097041 | True |
| masaya_gpt2 | 0.950048 | 0.9488823 | 0.9503172 | 0.955312 | 0.9498742 | 0.945854 | 0.009458 | True |
| lm7_5fold | 0.949444 | 0.9475445 | 0.9497671 | 0.9551662 | 0.9490652 | 0.945676 | 0.0094901 | True |
| gpt2_seed2026 | 0.9501731 | 0.9491911 | 0.9503701 | 0.9552357 | 0.9502097 | 0.9458585 | 0.0093773 | True |
| tabpfn_slimypunk_tp1 | 0.9493429 | 0.9481552 | 0.9503329 | 0.9547382 | 0.9491845 | 0.9443034 | 0.0104348 | True |
| sadam_tabpfn_100k_tp3 | 0.9488896 | 0.9475724 | 0.9508788 | 0.9529789 | 0.9490339 | 0.9439834 | 0.0089955 | True |
| yekenot_realmlp | 0.9501145 | 0.9500496 | 0.9510404 | 0.9549457 | 0.9503984 | 0.9441386 | 0.0108072 | True |
| alex_glm_gen | 0.9505035 | 0.9494181 | 0.9506593 | 0.9555146 | 0.9514767 | 0.9454483 | 0.0100663 | True |
| daoviet_gpdvae | 0.9465294 | 0.9447321 | 0.9476896 | 0.951463 | 0.9466615 | 0.9420998 | 0.0093632 | True |
| gbn_bag5_w512 | 0.9512429 | 0.9499157 | 0.9516058 | 0.9560761 | 0.9518553 | 0.9467612 | 0.0093149 | True |
| sameer_glm10f | 0.9516046 | 0.9508096 | 0.9520959 | 0.9563718 | 0.9520656 | 0.9466798 | 0.0096919 | True |
| sadam_v6_gb_xgb | 0.9507703 | 0.9506578 | 0.950468 | 0.9558513 | 0.9510173 | 0.9458569 | 0.0099944 | True |
| sadam_v6_gb_lgb | 0.9509402 | 0.9506461 | 0.9506616 | 0.9560987 | 0.9513073 | 0.9459875 | 0.0101113 | True |
| sadam_v4_cat | 0.9501135 | 0.9491888 | 0.9505731 | 0.9550723 | 0.9510023 | 0.9447305 | 0.0103418 | True |
| gb_bag5 | 0.9515369 | 0.9504322 | 0.951589 | 0.9562735 | 0.9523948 | 0.9469945 | 0.009279 | True |
| megayak_shallow_tree | 0.9515043 | 0.9503825 | 0.9518021 | 0.9562093 | 0.9525071 | 0.9466198 | 0.0095895 | True |
| anchor_095370 | 0.9525514 | 0.9517749 | 0.9525318 | 0.9572653 | 0.9528037 | 0.948381 | 0.0088843 | True |

## Strict LOFO gate (orth residual vs `[S7, G26−MS]`)
| stream | S7m pos5 | S7m meanΔe5 | S7m d0 | simple pos5 | gate |
|---|---:|---:|---:|---:|:---:|
| anchor_095370 | 5/5 | 0.035 | 0.037 | 4/5 | True |
| megayak_5f_lm | 3/5 | 0.004 | 0.015 | 2/5 | False |
| masaya_gpt2 | 1/5 | -0.003 | 0.001 | 3/5 | False |
| lm7_5fold | 0/5 | -0.015 | -0.053 | 1/5 | False |
| gpt2_seed2026 | 1/5 | -0.003 | 0.001 | 3/5 | False |
| tabpfn_slimypunk_tp1 | 3/5 | 0.011 | 0.013 | 1/5 | False |
| sadam_tabpfn_100k_tp3 | 3/5 | -0.003 | 0.013 | 3/5 | False |
| yekenot_realmlp | 4/5 | 0.014 | 0.052 | 2/5 | False |
| alex_glm_gen | 4/5 | 0.021 | 0.012 | 2/5 | False |
| daoviet_gpdvae | 3/5 | 0.003 | -0.01 | 0/5 | False |
| gbn_bag5_w512 | 5/5 | 0.03 | 0.005 | 2/5 | False |
| sameer_glm10f | 4/5 | 0.035 | 0.035 | 3/5 | False |
| sadam_v6_gb_xgb | 3/5 | 0.011 | 0.049 | 0/5 | False |
| sadam_v6_gb_lgb | 4/5 | 0.017 | 0.053 | 2/5 | False |
| sadam_v4_cat | 3/5 | 0.004 | 0.005 | 3/5 | False |
| gb_bag5 | 5/5 | 0.024 | 0.0 | 2/5 | False |
| megayak_shallow_tree | 3/5 | 0.02 | -0.011 | 2/5 | False |
| anchor_095370 | 5/5 | 0.035 | 0.037 | 4/5 | True |

### Swap residuals (−MS/−MLP/−GB/−LM) then orth — any 5/5?
- None of the −MS/−MLP/−GB/−LM swap+orth residuals cleared the dual gate.

## Implications for Track Compact-LM seed2026
1. Q17 is a **frozen stream bank** (16 members + 0.95370 anchor), not a new training recipe.
2. Member streams fail dual LOFO → **no cand2+Q17-member submit**. Anchor LOFO-clears but is already LB 0.95370 — hold submits.
3. G26 5-fold test still missing → Compact-LM `SEED=2026` Fold-0 canary remains the budget GPU path.
4. Detective `model.pt` is out of scope for OOF stacking this turn (no matching OOF/test vectors in detective-data).

---
*Generated by `scripts/gm_track2_amanatar_q17_audit.py`*
