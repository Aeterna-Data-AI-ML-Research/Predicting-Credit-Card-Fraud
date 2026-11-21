# GM V16.9 — Slot1 Sign Audit + GPU Order-Dispersion Scorecard

**STRICT `NO_SUBMIT=True` · 0 GPU pods · no Kaggle submit**

Script: `scripts/gm_v16_9_gpu_order_dispersion_family_audit.py`  
JSON: `/tmp/s6e11_v16/out/v16_9.json`

---

## Step 1 — Exact sign of live Slot 1 (`0.95389`) vs `(R3−L7F)` / `FE_fixed`

Files: `LIVE=sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv` (`0.95388`),  
`Slot1=sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv` (`0.95389`).

`delta_Slot1_test = L(Slot1) − L(LIVE)`:

| Contrast | Pearson | Spearman |
|---|---:|---:|
| `raw_diff = L(mean_f L(R3_test_f)) − L(L7F_test)` | **+0.9826** | **+0.9933** |
| `raw_diff` via `L(mean R3) − L(L7F)` | +0.9826 | +0.9933 |
| `raw_E = raw_diff · w_tail(U(LIVE))` | **+0.9938** | **+0.9958** |
| `FE_fixed` (V16.8, OOF-AUC-oriented) | **−0.9936** | **−0.9955** |
| `FE_old` (oriented same way) | −0.9963 | −0.9969 |

**Verdict:** Slot 1 moved in the **`+(R3 − L7F)` / `+raw_E`** direction on Test (ρ≈+0.99).  
It is **anti-aligned** with OOF-AUC-oriented `FE_fixed` (ρ≈−0.996).

### Why V16.8 reported `1 − spearman(FE_old, FE_fixed) ≈ 2.00`

That number is what you get from **anti-correlated** vectors: `1 − (−0.998) ≈ 1.998`.  
In this audit, after **both** are fold0-AUC-oriented the same way:

- `spearman(FE_old_test, FE_fixed_test) = **+0.9984**`
- `1 − ρ = **0.0016**` (nearly identical)

So V16.8’s “≈2.00” was a **sign-mismatch artifact** (comparing opposite orientations / constructions), not a structural disagreement between old and fixed Family_E once signs are aligned.  
`FE_fixed` itself was `flipped_to_plus_AUC` on fold0 — that flip makes it **opposite** to the live Slot1 `+raw_E` injection.

### OOF `± FE_fixed` on true `z_ATH` (0.9524301)

| Sign | w | pos/5 | meanΔ (e-5) | folds |
|---|---:|---:|---:|---|
| **+FE_fixed** | 0.006 | 3/5 | +0.41 | [−0.03, 0.54, 0.43, −0.31, 1.43] |
| **+FE_fixed** | 0.01045 | 3/5 | +0.47 | [−0.27, 0.68, 0.56, −0.83, 2.20] |
| **+FE_fixed** | 0.015 | 3/5 | +0.47 | [−0.41, 0.82, 0.42, −1.30, 2.81] |
| −FE_fixed | 0.006 | 1/5 | −0.56 | … |
| −FE_fixed | 0.01045 | 1/5 | −1.11 | … |
| −FE_fixed | 0.015 | 0/5 | −1.80 | … |

OOF prefers **+FE_fixed** (still only 3/5). Live Slot1 used **−FE_fixed ≡ +raw_E**. LB +1 tick and OOF fold screen disagree on Family_E sign.

---

## Step 2 — Physically grounded 10-order estimators

Tensors: OOF `(699960,10,2)`, Test `(5,299984,10,2)`; orders `k` / `k+5` are exact reverses.

Per fold/order:
- **`llr_z`:** column-zscore of `llr = ll_pos − ll_neg` (no rank-logit Jacobian)
- **`z`:** `L(llr)` per order (tail-inflating)

Estimators: **U1** q80−q20 on `z`; **U2** Gini10 on `z`; **U3** MAD10 on `z`; **U4** std10 on `z`;  
**U5** Gini10 on `llr_z`; **U6** std of 5 antithetic pair avgs; **U7** mean |fwd−rev|;  
**U8** `std(ll_pos_z) − std(ll_neg_z)`; **U_ens** = unitize(U2+U5+U6) sign-aligned to U1.

Prep for orth: `L_fold(oof)` vs `L(mean_f L(test_f))`; report  
`raw_pre_unitize_std_ratio = std(mean_f L(test_f)) / std(L_fold(oof))`  
(theory ≈ `√((1+4ρ̄)/5)` before outer rescaling). Then `frozen_lofo_orth` on Bo9/Bt9 (V16.8 9-basis).

---

## Step 3 — Unified scorecard on `z_S3` (OOF 0.9524402)

Gates: F1 disp∈[0.94,1.06], F2 leak<0.035, F3 |LIVE|<0.05.  
Stage only if F1∧F2∧F3 ∧ 5/5 ∧ sf≥18 ∧ t≥2.20 ∧ `cross_fold_r > U1`.

| Est | cross_fold_r | pre_std_ratio | F1 disp | F2 leak | F3 LIVE | Ver | pos/5 | w* | meanΔ | min | [F0..F4] | sf25 | t | W_p | 90% CI |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| **U1** q80−q20_z | 0.6065 | 0.834 | 0.990 ✓ | **0.043 ✗** | −0.026 ✓ | no | 5/5 | 0.009 | +0.82 | +0.12 | [1.08,1.28,0.12,0.62,1.02] | 13 | 1.21 | 0.183 | [−0.29,1.68] |
| U2 gini10_z | **0.7200** | 0.884 | 0.993 ✓ | 0.050 ✗ | −0.027 ✓ | no | 5/5 | 0.006 | +0.48 | +0.11 | [0.76,0.41,0.39,0.70,0.11] | 13 | 0.98 | 0.237 | [−0.37,1.37] |
| U3 mad10_z | 0.6994 | 0.875 | 0.993 ✓ | 0.049 ✗ | −0.027 ✓ | no | 5/5 | 0.02 | +1.13 | +0.15 | [2.54,0.96,0.15,1.62,0.37] | 15 | 1.30 | 0.138 | [−0.22,1.59] |
| U4 std10_z | **0.7248** | 0.886 | 0.994 ✓ | 0.050 ✗ | −0.027 ✓ | no | 5/5 | 0.006 | +0.44 | +0.04 | [0.75,0.37,0.33,0.72,0.04] | 13 | 0.85 | 0.271 | [−0.43,1.28] |
| **U5** gini10_llr | 0.6846 | 0.869 | 0.998 ✓ | **0.025 ✓** | −0.046 ✓ | **YES** | 5/5 | 0.009 | +0.72 | +0.09 | [1.41,0.73,0.35,1.04,0.09] | 15 | 1.10 | 0.213 | [−0.32,1.48] |
| **U6** pair_std5_llr | 0.5884 | 0.826 | 0.998 ✓ | **0.024 ✓** | −0.046 ✓ | **YES** | 5/5 | 0.009 | +0.66 | +0.11 | [0.83,0.24,1.11,1.01,0.11] | 15 | 1.06 | 0.105 | [−0.39,1.66] |
| **U7** fwd_rev_gap | 0.6317 | 0.845 | 0.997 ✓ | **0.022 ✓** | −0.040 ✓ | **YES** | 5/5 | 0.009 | +0.63 | +0.09 | [0.77,1.12,0.09,0.67,0.48] | 11 | 0.93 | 0.298 | [−0.44,1.49] |
| **U8** pos−neg_std | 0.6225 | 0.853 | 0.995 ✓ | **0.025 ✓** | −0.019 ✓ | **YES** | 4/5 | 0.006 | +0.51 | −0.02 | [−0.02,1.36,0.47,0.71,0.01] | 15 | 1.18 | 0.205 | [−0.31,1.71] |
| **U_ens** | **0.7090** | 4.12* | 0.998 ✓ | **0.034 ✓** | −0.044 ✓ | **YES** | 5/5 | 0.009 | +0.75 | +0.08 | [1.19,0.65,0.65,1.20,0.08] | 14 | 1.31 | 0.126 | [−0.20,1.50] |

\*U_ens `pre_std_ratio` is inflated because OOF side was pre-`unitize`d before the ratio.

### Staging
Need sf≥18 and t≥2.20 and `cross_fold_r > U1(0.6065)`.

- Best forensic-clean: **U5 / U6 / U7 / U_ens** — all **fail soft bar** (sf≤15, t≤1.31).
- Rank-logit family **U1–U4** have higher fold agreement (U4 `cross_fold_r=0.725`) but **fail F2** (basis leak ~0.05) under `L_fold` prep.
- **`winners=[]` → no `sub_v16_9_*_NOSUBMIT.csv`.**

Note: U1 here is weaker than V16.8’s `q80_q20` (17/25, F2 pass) because this step forces **`L_fold` + outer `L(mean L)`** prep to expose pre-unitize dispersion; that changes the orth and softens the screen.

---

## Verdicts

1. **Slot1 sign is `+(R3−L7F)` / `+raw_E`** (ρ≈+0.99 vs LIVE delta). OOF-oriented `FE_fixed` is the **opposite** sign of what went live.
2. **V16.8 `1−ρ≈2`** = anti-correlation artifact; with matched orientation, old≈fixed (`ρ=+0.998`).
3. **`llr_z` estimators (U5–U7, U_ens) clear F1–F3**; none clear **≥18/25 / t≥2.20**. Higher cross-fold reliability (U2/U4/U_ens) does not buy a stageable lift under current gates.
4. Nothing staged. `NO_SUBMIT=True`.
