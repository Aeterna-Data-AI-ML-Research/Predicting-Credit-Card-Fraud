# GM V16.5 — Full GPU L7 Audit (Diagnostic, NO_SUBMIT)

**STRICT diagnostic · 0 pods · Why Slot1 only +1 tick**

---

## Step 1 — How much `lm7_test_f0` sits inside LIVE 0.95388?

OLS of `L(LIVE)` / `L(bold)` on test basis  
`[MS, G26_f0, LM_5f, L7F_5f, lm7_test_f0, GPD, MLP, GB, d1_v4]` (R²≈0.980):

| stream | β_LIVE | β_bold |
|--------|-------:|-------:|
| MS | +0.627 | +0.626 |
| G26_f0 | −0.226 | −0.227 |
| LM_5f / L7F_5f | **+0.068** | **+0.068** |
| **lm7_test_f0** | **+0.00032** | **+0.00032** |
| GPD | +0.034 | +0.034 |
| MLP | +0.181 | +0.182 |
| GB | +0.243 | +0.243 |
| d1_v4 | −0.103 | −0.133 |

**Verdict:** LIVE is **not** carrying meaningful 1-fold `lm7_test_f0` weight (β≈0). The 5-fold `L7F_5f` term is ~200× larger. Hypothesis “LIVE still suffers Fold-0 variance via f0” is **rejected**.

### Fold-0 subfolds: 2-seed GPU blend vs L7F
- Solo `0.5·L(R3)+0.5·L(L7F)` vs `L(L7F)` on Fold0’s 5 subfolds: **5/5**, meanΔ≈+52e-5 (noisy t=1.87) — 2-seed beats 1-model L7F locally.
- Adding `0.01·unitize(0.5(R3−L7F))` on top of **true z_ATH** on Fold0 subfolds: **2/5**, meanΔ≈−0.4e-5 — already absorbed by ATH.

---

## Step 2 — Full `(N,10,2)` tensor beyond R3 (on true z_S1)

All directions LOFO-orth vs `[z_ATH, d1, Family_E]`:

| extraction | pos/5 | sf25 | t |
|------------|------:|-----:|--:|
| α∈{0.85…1.15} asymmetric ll_pos−α·ll_neg | ≤2/5 | ≤13 | ≤0.46 |
| q80−q20 inter-order spread | **5/5** @0.012 | 16/25 | 1.91 |
| max−min order | 4/5 | 14 | 1.49 |
| fwd−rev antithetic gap | 3/5 | 12 | 0.29 |
| joint LOFO [R1,R2,R4,llp,lln,d1] | 2/5 | 12 | 0.14 |

**No extraction clears ≥19/25 & t≥2.8 on true z_S1.** Best near-miss: order-spread (5/5, 16/25, t=1.91).

---

## Step 3 — Unified candidate table (OOF on z_S1)

| candidate | pos/5 | meanΔ e-5 | min | sf25 | t | 1−ρ(Slot1) e-4 | live? |
|-----------|------:|----------:|----:|-----:|--:|---------------:|:-----:|
| Slot1 + d1 +0.0030 | **5** | 0.345 | +0.047 | **18** | **3.31** | 0.013 | NOSUBMIT P1 |
| Slot1 + d1 +0.0040 | 5 | 0.435 | +0.061 | 17 | 2.99 | 0.023 | — |
| Slot1 + d1 +0.0050 | 4 | 0.503 | −0.008 | 17 | 2.71 | 0.037 | — |
| Slot1 + d1 +0.0055 | 5 | 0.551 | +0.008 | 17 | 2.91 | 0.044 | **Slot3 submitted** |
| Slot1 + d1 +0.0060 | 5 | 0.587 | +0.002 | 17 | 2.94 | 0.053 | — |
| f0→5f replacement | — | — | — | — | — | — | **N/A** (β_f0≈0) |
| multi-stat GPU | best 5/5 | — | — | 16 | 1.91 | — | fail gate |

**Only full hard gate (5/5, ≥18/25, t≥3.0): `Δw_d1=+0.003`.**

---

## Why only +1 tick so far?
1. LIVE already uses **5-fold L7F**, not 1-fold f0.
2. True ATH already absorbs most L7_s2026 residual (Family E weak on 0.95243).
3. Extra tensor stats don’t clear orthogonal gates on z_S1.
