# GM V16.8 — Family_E Drift Fix + Frozen-β `lofo_orth` Re-Score

**STRICT `NO_SUBMIT=True` · 0 GPU pods · no Kaggle submit**

Script: `scripts/gm_v16_8_frozen_beta_orth_and_fix_famE_drift.py`  
JSON: `/tmp/s6e11_v16/out/v16_8.json`

---

## Step 1 — Fix `Family_E` OOF↔Test symmetry

### Construction changes
1. **Matched scale:** `z_s2026 = L(R3)` OOF; Test = `L(mean_f(L(R3_test_f)))` so  
   `std(z_s2026_*) = std(z_l7f_*) = **1.81378**` on both sides.
2. **`w_tail`:** `0.5 + 0.5·U(·)` with exact Uniform ranks — `U(z_ATH)` OOF, `U(LIVE)` Test.
3. **Frozen LOFO orth** of `raw_E` vs `[unitize(z_ATH), unitize(d1)]` (Ridge α=1):  
   `orth_test = raw_E_test − X_test @ mean_f(β_(−f))` (no separate Test OLS).

### Correlation vs L7F

| Version | corr_oof(FE, L7F) | corr_test(FE, L7F) | \|Δ\| | disp |
|---|---:|---:|---:|---:|
| **V16.7 broken** (reported) | +0.0783 | −0.0666 | **0.1449** | — |
| OLD recipe (this script) | −0.0783 | −0.0354 | 0.0429 | — |
| **`Family_E_fixed`** | **+0.0778** | **+0.0634** | **0.0144** ✓ (<0.015) | **1.0000** ✓ ∈[0.97,1.03] |

Before `w_tail`: corr(diff, L7F) oof/test = −0.0834 / −0.0692 (Δ=0.0142).  
After `w_tail`: −0.0811 / −0.0664 (Δ=0.0147). Frozen orth preserves the tight OOF↔Test match.

`1 − spearman(FE_old_test, FE_fixed_test) ≈ 2.00` → anti-aligned on Test (sign/construction flip vs old separate-OLS path).

### `Family_E_fixed` on true `z_ATH` (0.9524301)

| w | pos/5 | meanΔ (e-5) | folds |
|---:|---:|---:|---|
| 0.006 | 3/5 | +0.41 | [−0.03, 0.54, 0.43, −0.31, 1.43] |
| 0.01045 | 3/5 | +0.47 | [−0.27, 0.68, 0.56, −0.83, 2.20] |
| 0.01383 | 3/5 | +0.47 | [−0.38, 0.77, 0.45, −1.16, 2.69] |

Standalone FE_fixed is only **3/5** on `z_ATH` (weaker as a solo Slot-1 driver than the historical Family_E path). Used here as a **basis column**, not a new Slot-1 replacement.  
`AUC z_S1=0.9524347`, `z_S3=0.9524402` (FE_fixed @0.01045 + d1 @0.0055).

---

## Step 2 — `frozen_lofo_orth` on non-redundant 9-basis

**Basis (no `z_ATH`):**  
`[MS, G26, LM, L7F, GPD, GB, MLP, d1_v4, Family_E_fixed]`  
`cond(Bo9)=29.7`, `cond(Bt9)=32.3`.  
In-basis corr(FE, L7F): oof **0.0778** / test **0.0634**.

```text
orth_oof[va] = unitize(v)[va] − Bo9[va] @ β_(−f)          # Ridge α=10 per fold
orth_test    = unitize(v_test) − Bt9 @ mean_f(β_(−f))     # FROZEN — no Test OLS
```

### Gates
| Gate | Rule |
|---|---|
| **F1** Dispersion | `std(orth_test)/std(orth_oof) ∈ [0.94, 1.06]` |
| **F2** Basis leak | `max_j \|corr(orth_test, Bt9[:,j])\| < 0.035` |
| **F3** LIVE DC | `\|corr(orth_test, LIVE − Ridge(Bt9))\| < 0.05` |

---

## Step 3 — Full candidate table on `z_S3`

| Signal | disp (F1) | leak (F2) | LIVE (F3) | Verified | pos/5 | w* | meanΔ | min | [F0..F4] | sf25 | t |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|
| **`q80_q20`** | **0.997 ✓** | **0.027 ✓** | **−0.024 ✓** | **YES** | **5/5** | 0.015 | +1.62 | +0.05 | [2.45, 2.31, 0.05, 1.50, 1.78] | **17** | 1.72 |
| `R2_R3_curv` | 0.884 ✗ | 0.011 ✓ | −0.017 ✓ | no | **5/5** | 0.006 | +0.33 | +0.01 | [0.01, 0.74, 0.38, 0.43, 0.08] | 16 | 1.52 |
| `sadam_v4_cat` | 0.842 ✗ | 0.030 ✓ | −0.111 ✗ | no | 4/5 | 0.04 | +3.01 | −4.64 | [7.27, 1.85, 1.35, 9.23, −4.64] | 19 | 3.16 |
| `sameer_glm10f` | 0.730 ✗ | 0.040 ✗ | −0.003 ✓ | no | 4/5 | 0.006 | +0.80 | −0.22 | [1.59, 1.36, 1.16, −0.22, 0.10] | 20 | 2.04 |
| `alex_372ea048` | 0.534 ✗ | 0.137 ✗ | −0.136 ✗ | no | 4/5 | 0.04 | +1.94 | −3.49 | [2.02, 0.67, −3.49, 4.38, 6.13] | 17 | 1.55 |
| `alex_6c55b429` | 0.833 ✗ | 0.045 ✗ | +0.049 ✓ | no | 4/5 | 0.03 | +1.19 | −8.65 | [5.70, 2.72, 0.44, 5.72, −8.65] | 17 | 1.47 |
| `sadam_tabpfn_100k_tp3` | 0.948 ✓ | 0.087 ✗ | +0.087 ✗ | no | 4/5 | 0.025 | +0.92 | −10.0 | [2.91, −10.0, 6.32, 4.19, 1.23] | 15 | 1.13 |
| `alex_glm_gen` | 1.026 ✓ | 0.100 ✗ | −0.067 ✗ | no | 3/5 | 0.006 | −0.32 | −2.05 | [0.35, −0.41, 0.07, 0.44, −2.05] | 11 | −1.01 |
| `gbn_bag5_w512` | 1.095 ✗ | 0.035 ✓ | −0.039 ✓ | no | 2/5 | 0.006 | −0.11 | −1.22 | [0.45, −1.22, −0.16, 0.46, −0.09] | 14 | −0.28 |
| `megayak_shallow_tree` | 1.832 ✗ | 0.234 ✗ | −0.012 ✓ | no | 3/5 | 0.015 | +0.69 | −3.28 | … | 14 | 1.01 |
| `tabpfn_slimypunk_tp1` | 0.868 ✗ | 0.038 ✗ | +0.047 ✓ | no | 3/5 | 0.02 | +0.83 | −2.37 | … | 15 | 1.08 |
| `sadam_v6_gb_*` | fail | fail | fail | no | ≤2/5 | — | neg | — | — | ≤8 | <0 |

### Staging rule
Need **F1∧F2∧F3 ∧ 5/5 ∧ sf≥18**.

- **`q80_q20`:** gates **PASS**, **5/5**, but **sf=17/25** (one short of 18) → **not staged**.
- **`verified_full = []`** → **no** `sub_v16_8_*_NOSUBMIT.csv`.

---

## Verdicts

1. **Family_E flip fixed:** `|Δcorr(FE, L7F)|` **0.1449 → 0.0144** (<0.015); `disp=1.000`. Root fix = matched `L()` scales + Uniform `w_tail` + **frozen** orth vs `[z_ATH, d1]`.
2. **Frozen Ridge LOFO** (α=10) on 9-col basis **without `z_ATH`** makes OOF/Test orth the **same linear operator**. Under this definition, **`orth(q80−q20)` is the first fully forensic-clean add-on** (F1/F2/F3 all true) with **5/5** main folds.
3. Soft bar still blocks submit/stage: **17/25** sf (`t=1.72`). `R2−R3` is also 5/5 but fails F1 (disp 0.884). `sameer`/`sadam` still fail dispersion / LIVE leak under frozen β.
4. Next lever (if authorized): soft-grid `q80` weight / composite with other near-clean dirs, or accept 17/25 with explicit risk — **not done here** (`NO_SUBMIT`).

`NO_SUBMIT=True` — nothing uploaded to Kaggle in V16.8.
