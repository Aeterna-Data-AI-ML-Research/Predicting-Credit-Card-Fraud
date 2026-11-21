# GM L7 5F Rotated-10 — Phase 0 Pre-Flight ($0.00)

## Architecture (exact match to `lm7_*`)

- Backbone: **GPT-2 small (12 layers, 12 heads, d=768) — megayak compact lm_member.py / goodpjw seed7**
- Scoring: **full-sequence NLL under label False then True for each column order; LLR = mean(ll[...,1]-ll[...,0]) or Bayes logsumexp**
- Explicitly **distilgpt2; NOT final-token logits[:, -1, [tok0,tok1]]**
- Shapes: `{'oof': '(139992, K_oof, 2)', 'test': '(299984, 4, 2)', 'dtype': 'float32'}`
- Fold0 README AUC: `0.947545`
- T=72 V=1372 NTOK=1401 NC=24

## Rotated 4-of-10 cover

Every master order 0..9 appears exactly **2×** across folds 0..4 test tensors.

## A40 wall-clock estimate (Folds 1–4 only)

| metric | value |
|---|---:|
| anchor | A40 monitor ~970 rows/s @bs32; score≈20.5m/fold; BATCH=64 → ~1.7× train |
| batch | 64 (alt 64) |
| train min/fold | 45.3 |
| score min/fold | 20.5 |
| replica min (once) | 35.0 |
| **total hours** | **4.97** |
| **$ @ $0.59/hr** | **$2.93** |
| alt BATCH=96 hours/cost | 3.88h / $2.29 |
| ≤6.5h / ≤$3.85 gate | `PASS` |

Watchdog cap: 510 min ($5.01 @ $0.59) << $10 hard cap.

PHASE0_OK elapsed=1.5s
