# GM V18.8 — L2FO Full-Graph Surrogate + V18.6 Merchant-Key Restore

**STRICT `NO_SUBMIT=True` · 0 GPU pods · ~152s**  
Script: `scripts/gm_v18_8_zero_echo_fullgraph_surrogate_and_v186_blend.py`  
Artifacts: `/tmp/s6e11_v16/out/v18_8.json`, `v18_8_peers.npz`

---

## Verdict

**One NOSUBMIT staged:** `V1_va_strict + U5 + 0.25·R1` clears F1/F2/F3, 5/5×3, sf≥19, t≥3.2.  
**L2FO full-graph (Bank B) did not unlock Fold2** — ~7× more peers, but single-fold `B_h` scores are too weak → V2 stays ~+1.5e-5.

| Candidate | F2 | Fold2 Δ | sf (S5/twin/NNLS) | Stage |
|---|---|---|---|---|
| V1 va-strict (merchant keys) | ✗ 0.047 | **+0.03** | 18/19/18 | ✗ |
| V2 L2FO fullgraph | ✗ 0.036 | +0.72 | 14/14/14 | ✗ |
| V3 dual | ✗ 0.041 | +1.06 | 14/15/16 | ✗ |
| **V1+U5+0.25·R1** | ✓ **0.026** | **+2.42** | **20/19/21** | **✓** |
| V1+R1 only / V3+R1 | soft F2 or sf | — | — | ✗ |

---

## Step 1 — Peer counts (va-only vs full graph)

| Key | cnt_va | cnt_full | × |
|---|---|---|---|
| k2 balance | 24.7 | **170.1** | **6.9** |
| k6 mer×amt | 405 | 2885 | 7.1 |
| k7 mer profile | 220 | 1563 | 7.1 |
| k1–k5, k9–k10 | … | … | ~7.1 |
| k8 cents×mer | 3.1 | 16.3 | 5.2 |

**Zero-echo construction:** `B_h` trained **only** on `(X_broad[fold==h], y[fold==h])`.  
For val fold `f`, peer score on fold `g≠f` = mean `B_h` for `h∉{f,g}` (L2FO); on `va`/`test` = mean `B_h` for `h≠f` (+ blend `z_S5_oof[va]` on va). **Never uses `y[va]`.**

---

## Step 2 — Learners

- **V1:** Bank A (10 merchant/numeric keys, va-only / test-only, m0∈{5,20}, scaled test) · Ridge+5-seed LGB · **no cohort trees**
- **V2:** Bank B (same keys, L2FO full-graph peers)
- **V3:** 0.5·V1 + 0.5·V2

---

## Step 3 — Scorecard highlights

### Fold2 recovery check

| Signal | Fold2 on z_S5 |
|---|---|
| V18.7 S4 | +0.30 |
| V1 alone | +0.03 (still dead) |
| V2 L2FO | +0.72 |
| **V1+U5+R1@0.25** | **+2.42** ✓ |

Merchant-key restore alone did **not** fix Fold2; the staged winner rides **U5+R1** to clear sf/t (R1 previously **failed live Slot6 −2e-5** — treat as NOSUBMIT ammo only).

### Staged winner

`sub_v18_8_V1_va_strict_repro_plus_U5_R1_0p25_w0p025_NOSUBMIT.csv`

| Base | pos | w | meanΔ | min | folds | sf | t |
|---|---|---|---|---|---|---|---|
| z_S5 | 5/5 | 0.03 | +6.87 | +2.42 | [9.64, 7.46, 2.42, 10.88, 3.96] | 20 | 4.69 |
| twin | 5/5 | 0.025 | +8.86 | +0.87 | [14.97, 7.65, 0.87, 14.51, 6.32] | 19 | 4.00 |
| NNLS | 5/5 | 0.03 | +7.70 | +3.21 | … | 21 | 4.43 |

---

## Guardrails

| Check | Status |
|---|---|
| Submit | **NOT done** |
| Pods | **0** |
| ATH | **0.95394** |

---

## Bottom line

1. **L2FO math works** (7× peer support, zero `y[va]`), but **single-fold `B_h` is too weak** to replace `z_S5` peer quality.  
2. Restoring V18.6 merchant keys helps V1 vs V18.7 cohort poison, yet Fold2 remains soft without R1/U5.  
3. Staged combo **depends on R1** — do **not** auto-submit; Slot6 already showed R1 can hurt LB.

---

*End GM V18.8*
