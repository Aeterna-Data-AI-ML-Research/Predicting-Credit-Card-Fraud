# V47.3 — Private Kaggle HGDT-LM v2 Fold-0 Report

**NO_SUBMIT=True · $0.00 RunPod · 0 competition submissions · private kernel**

- Kernel: https://www.kaggle.com/code/moranzavdi/v47-3-private-hgdt-v2-fold0 (`is_private=true`, GPU Tesla T4, internet off)
- Wall: **1169 s** (~19.5 min) · assert `max|sum(field24)−seq_llr|` = **4.0e-5** (≪ 1e-3)
- Artifacts: `preds/v47_kaggle_f0/` · shapes OK · 0 NaN/Inf · SHA-256 recorded in `hgdt_v2_f0_meta.json`

## Solo competence (Fold-0 va0 = 139,992)
| signal | AUC |
|---|---:|
| mean `seq_llr` (K=8) | **0.904605** |
| `sum(field24_llr)` | **0.904605** |
| newton-cholesky LR on 24 LLR + 24 std | **0.934201** |

Per-order AUCs: `[0.855, 0.813, 0.867, 0.826, 0.853, 0.814, 0.784, 0.842]`

## vs `z_sovereign_v0` Fold-0 (**0.951251**)
| stream | ΔAUC (e-5) | pos/5 | t | KS | R² |
|---|---:|---:|---:|---:|---:|
| seq | **−4664.6** | 0/5 | −28.9 | 0.850 | ≪0 |
| LR fuse (0.75·z0 + 0.25·LR) | **−583.1** | 0/5 | −9.1 | 0.988 | ≪0 |
| cross-modal P4+field24 LGB | **−633.7** | 0/5 | −7.2 | 0.990 | ≪0 |

## Hygiene (pre-unitize sum-field oof vs test)
- `disp_raw` = **0.9963** ∈ [0.92, 1.08] · `KS_raw` = **0.0026** < 0.015 → PASS

## Verdict
Fold-0 canary **does not clear** promotion vs sovereign ruler (0/5 subfolds; solo ≪ 0.951). Field24 LR helps vs equal-weight (+~2960 e-5 solo) but still far below `z_sovereign_v0`. **Do not scale to 5-fold.**

## HARD STOP
- **NO_SUBMIT=True** · **$0.00 spent** · **0 Kaggle competition submissions** · **0 RunPod pods**
