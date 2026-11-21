# GM V33.0 RUNPOD CANARY REPORT

- written: 2026-11-21 15:11 IST
- pod: `363jat229iyz7p` (2×RTX4090 SECURE @ $1.78/hr) — **TERMINATED** after va0 pull
- topology: Option1 2×4090 shared `/workspace`
- truncation fix: compact markers, max_len=88, **0 truncated**
- Epoch1: 606728 shuffled rows (9.25% pos), CE loss mean≈1.05, |V_tab|≈89

## Fold-0 Gate (Mac)

| worker | solo_AUC | R²_bank | opt_w | ΔAUC e-4 | subfolds | PASS |
|---|---:|---:|---:|---:|---|---|
| tabicl | 0.94257 | 0.927 | 0.0 | 0.0 | 0/5 | false |
| qwen15b_ep1 | 0.94692 | 0.950 | 0.0 | 0.0 | 0/5 | false |

**Decision:** FAIL — no lift on `wide_lm` (opt_w=0), solo < 0.9508, not promising for ep2 (≥0.9485). **0 Kaggle submits.**

## Artifacts
- `preds/v33_canary/qwen15b_ep1_va0.npy`
- `preds/v33_canary/ep1_ckpt/` (LoRA + sub head backed up mid-flight)
- `preds/v33_canary/tabicl_va0.npy`

## Cost
- See RunPod balance delta vs start ~$8.25; pod terminated immediately post-pull.
