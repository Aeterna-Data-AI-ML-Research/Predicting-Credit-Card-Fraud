# GM V17.0 — Fold-0 Debias Ground Truth + Weight Audit + G26 Architecture

**STRICT `NO_SUBMIT=True` · 0 GPU pods · no Kaggle submit**

Script: `scripts/gm_v17_0_fold0_debias_and_g26_audit.py`  
JSON: `/tmp/s6e11_v16/out/v17_0.json`

---

## Step 1 — Does Seed-7 Fold-0 error predict Seed-2026 Fold-0 error on Test?

Definitions on 299,984 test rows:
- `err_f0_s7 = L(mean_k llr(lm7_test_f0)) − L(lm7_test_5fold)`  
  (`lm7_test_f0` is `(N,4,2)` Fold-0 × 4 orders)
- `err_f0_s2026 = L(R3_test_0) − L(mean_f L(R3_test_f))`

### Cross-seed transfer

| Metric | Value |
|---|---:|
| Pearson(`err_f0_s7`, `err_f0_s2026`) | **+0.3305** |
| Spearman | **+0.3328** |
| `γ̂` (OLS) | 0.239 |
| **R²** / MSE reduction recovering `R3_0→R3_5` via `γ·err_s7` | **0.109** |

### Signature of shared Fold-0 split bias

`corr(err_f0_s7, L(R3_f) − L(R3_5))`:

| f | Pearson | Expected if pure fold-bias |
|---:|---:|---|
| **0** | **+0.3305** | strongly **+** |
| 1 | −0.0565 | ≈ −r/4 ≈ −0.083 |
| 2 | −0.0647 | ≈ −0.083 |
| 3 | −0.0935 | ≈ −0.083 |
| 4 | −0.1050 | ≈ −0.083 |

**Hypothesis: PASS** — f=0 strongly positive; f=1..4 all negative and near −r/4.

Within-seed fold-error matrix `C[f,g]=corr(R3_f−R3_5, R3_g−R3_5)` has off-diagonals ≈ **−0.22…−0.27** (zero-sum structure of fold deviations).

### G26_f0 − MS vs fold-0 errors (control seed contrast)

| | raw Pearson | partial \| `(R3_5 − L7F_5)` |
|---|---:|---:|
| `err_f0_s7` vs `L(G26_f0)−L(MS)` | 0.096 | 0.096 |
| `err_f0_s2026` vs same | 0.099 | 0.099 |

**Weak** (~0.10): Fold-0 Compact-LM bias only mildly aligns with the GPT-2 f0−MS gap. Debiasing `G26_test_f0` with `err_f0_s7` would recover little of a true 5-fold G26 (R²~0.11 even on Compact-LM).

---

## Step 2 — Simplex / Ridge LOFO weights (NO `flip_to_plus_AUC`)

### Bold recipe (`build_ath_proxy` RECIPE_JSON) — core streams

| Stream | Bold `w` |
|---|---:|
| MS (`s:masaya_gpt2`) | −0.070 |
| G26 (`s:gpt2_seed2026`) | −0.013 |
| LM (`s:megayak_5f_lm`) | −0.122 |
| L7F (`s:lm7_5fold`) | −0.041 |
| GPD | **+0.136** |
| GB (`s:gb_bag5`) | +0.009 |
| MLP | −0.015 |
| wide | −0.035 |

(LIVE OLS β from V16.5 is a different object: MS≈+0.63, G26≈−0.23 — reconstruction vs recipe.)

### Solo 5-fold OOF AUC

| Stream | mean | [F0..F4] |
|---|---:|---|
| **GB** | **0.95154** | [0.9504, 0.9516, 0.9563, 0.9524, 0.9470] |
| **LM** | **0.95045** | [0.9485, 0.9507, 0.9561, 0.9506, 0.9464] |
| G26 | 0.95017 | [0.9492, 0.9504, 0.9552, 0.9502, 0.9459] |
| MLP | 0.95012 | … |
| MS | 0.95005 | … |
| L7F | 0.94944 | … |
| R3_s2026 | 0.94930 | … |
| GPD | 0.94653 | … |

### Replace grid on `z_ATH`: `+α·(L(R3)−L(j))` **without sign flip**

| Contrast | pos/5 | best α | meanΔ e-5 | sf25 | t |
|---|---:|---:|---:|---:|---:|
| **`L7_2seed − LM`** | **5/5** | 0.005 | **+0.54** | 13 | 1.12 |
| `R3 − LM` | 4/5 | 0.01 | +0.52 | 11 | 0.52 |
| `R3 − L7F` | **1/5** | 0.005 | −0.40 | 7 | −1.56 |
| `R3 − MS/G26/GB/...` | ≤2/5 | — | negative | — | — |

**Why `0.5·R3+0.5·L7F` beats L7F alone (5/5) but `±(R3−L7F)` on `z_ATH` is 1/5:**  
ATH already holds **LM** (stronger than R3) with bold/LIVE mass. The useful move is **partially replacing LM with the 2-seed L7 blend**, not adding raw `R3−L7F`. Without `flip_to_plus_AUC`, `R3−L7F` is simply the wrong axis relative to the stack.

### LOFO non-negative weights on 9 streams

Mean **Logistic L2 → clip≥0 → renormalize**:

| MS | G26 | LM | L7F | R3 | GPD | GB | MLP | d1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.164 | 0.157 | **0.217** | 0.052 | **0.000** | 0.000 | **0.287** | 0.107 | 0.017 |

OOF AUC **0.95190** (below `z_ATH` 0.95243 — simplex on these 9 alone cannot match the full bold MIX).  
**R3_s2026 gets zero** mean logistic weight → second seed is absorbed via LM/L7F/GB, not as a free positive member.

Ridge(`α=10`, `positive=True`) spreads mass onto **d1 (0.32)** and gives R3 a small **0.049**; OOF AUC 0.9450 (worse — wrong loss for AUC).

---

## Step 3 — Matapanino / G26 architecture vs Compact-LM L7

From `external/matapanino_gpt2_repo/{tabtext,train_replica,train_classifier,score_rows,make_queue}.py` + GPT-2 tokenizer on `train.csv`:

| | **G26 (Matapanino/Masaya)** | **L7 Compact-LM (ours)** |
|---|---|---|
| Tokenization | **GPT-2 BPE text**: `"col is value"` × 24 cols + `", "` + target-first prefix + eos | **1 compact multi-slot token block per column** (V=1372) |
| **L_seq** | **median 224** (mean 224.5, p95 227, max 229); prefix 7 toks | **T = 72** |
| Backbone | HF **`gpt2`**: 12-layer, n_embd=768, n_head=12, ~124M | Same **12×768** block stack (init from gpt2); **not** 4-layer |
| Epochs | replica **60** + classifier **8** / fold | replica 60 + FT 8 (same recipe class) |
| Orders K | score default **30**; G26 test used **10** | **10** antithetic (fold0 test); folds1–4 test use 6 |

User’s “4-layer / L_seq=25” does **not** match this repo: L7 is **12×768 @ T=72**.

### FLOP & wall for G26 test folds 1–4

Forward FLOP ≈ `n_layer·(12 L d² + 2 L² d)`:
- Ratio G26/L7 ≈ **3.21×** per forward (L=224 vs 72; same n_layer, d, K=10).

| GPU | Est. score wall / fold | ×4 folds (1–4) |
|---|---:|---:|
| P100 (reported) | **12.6 h** | 50.4 h |
| RTX 4080 SUPER | ~2.3 h | ~9.2 h |
| **RTX 4090** | **~1.7 h** | **~6.7 h** |
| **A100 80GB** | **~1.4 h** | **~5.6 h** |

FLOP-scale from L7’s 75 min/fold @4080 → ~241 min (~4.0 h) G26 score/fold @4080 (same order as P100-scaled 2.3 h; remaining gap = IO/batching/bf16 vs P100 fp16).

**Bottleneck vs L7:** mostly **L_seq 224 vs 72 (≈3× FLOP)** and **hardware** (P100 vs 4080), not layer count.

---

## Verdicts

1. **Fold-0 split bias is real and cross-seed transferable** (ρ≈0.33; f=0 +, f≠0 −), but only **~11% R²** — weak for G26_f0 debias (partial corr with G26−MS ≈0.10).
2. **Do not add raw `±(R3−L7F)` to z_ATH** (1/5 without flip). Best constrained move: **tiny replace of LM by `0.5·L7F+0.5·R3`** (5/5, +0.54e-5, still soft). LOFO simplex gives **R3 weight ≈ 0**.
3. **G26 is BPE GPT-2 @ L≈224, 12×768, K=10**; regenerating folds 1–4 ≈ **1.4–1.7 h/fold** on A100/4090 (~6–7 h total), not a 4-layer/L=25 job.

`NO_SUBMIT=True` — nothing uploaded to Kaggle in V17.0.
