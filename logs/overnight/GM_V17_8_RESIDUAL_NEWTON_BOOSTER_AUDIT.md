# GM V17.8 — Newton Residual TE & Shallow Residual Booster on `r = y − p_S3`

**STRICT `NO_SUBMIT=True` · 0 GPU pods · local CPU (~67s)**

Script: `scripts/gm_v17_8_residual_target_encoding_and_newton_step.py`  
JSON / tensors: `/tmp/s6e11_v16/out/v17_8.json`, `v17_8_residuals.npz`

---

## Verdict

**Residual targeting cured V16.2’s collinearity.** Fitting to Newton residuals of calibrated `z_S3` yields pre-orth `|ρ(feat, z_S3)| ∈ {0.023, 0.062}` vs digit4leaf’s **0.326**.

| Signal | Forensic | Dual S3∩twin (sf≥18, t≥2.2) | NNLS | Stage |
|---|---|---|---|---|
| **R2 residual LGB** | **✓** | **✓** (5/5 both, sf20, t≈3.7) | **✓** 5/5 | **YES** |
| R1 Newton TE | F2 0.052 ✗ | ✓ (huge +8…+14 e-5) | ✓ | no |
| R12 / R3 blends | F2 ✗ | ✓ | ✓ | no |
| U5 alone | ✓ | ✗ (weak sf/t) | soft | no |

**Staged (NOSUBMIT only):**  
`submissions/sub_v17_8_R2_residual_LGB_w0p025_NOSUBMIT.csv` @ **w\*=0.025**

---

## Step 1 — Calibrated OOF residual

Per fold: `LogisticRegression` of `y ~ z_S3` on `tr` → `p = σ(a_f z + b_f)`, `r = y − p`, `h = p(1−p)`.

| Check | Value |
|---|---|
| `corr(r_oof, z_S3)` | **−0.000026** ≈ 0 ✓ |
| mean weighted corr(g, z) on tr | +1.56e-2 (score orthogonality soft) |
| AUC(`p_oof`) | 0.952399 |

---

## Step 2 — Raw residual signals

**Keys (39):** per numeric col → cents / tenth / ones; q1000 on 4 floats (train∪test ranks); `log1p` amount frequency bins; orig20k amount hit bins (mean hits 3.06); cat×amount_ones for merchant/channel/device.

**R1:** fold-wise Newton TE `Δ = Σg / (Σh + λ₀)`, `λ₀=20`, Ridge(α=100) combiner → OOF/test.

**R2:** LightGBM `num_leaves=6`, `min_data_in_leaf=500`, `lr=0.03`, `n_estimators=40`, target=`g`, `sample_weight=h` (init at calibrated `z_S3`).

**R3:** `unitize(R1)+unitize(R2)+unitize(U5)`.

### Pre-orth collinearity (the V16.2 fix)

| Feature | `ρ(·, z_S3)` | vs V16.2 digit4leaf 0.326 |
|---|---|---|
| R1 | **0.0235** | ~14× lower |
| R2 | **0.0622** | < 0.08 gate ✓ |
| R3 blend | **0.0028** | |

---

## Step 3 — Forensic & dual-baseline scorecard

Fixed sign = S3-helpful on fold0; **same u** on all three baselines.

| Signal | pre \|ρ\| | F1 disp | F2 leak | F3 LIVE | ver | z_S3 | Slot3_twin | NNLS_twin |
|---|---|---|---|---|---|---|---|---|
| **R2** | 0.062 | 0.990 ✓ | **0.033** ✓ | −0.002 ✓ | **✓** | **5/5** @0.025 **+4.02** min+0.31 sf**20** t=**3.71** | **5/5** @0.020 **+4.99** min+1.80 sf**20** t=**3.72** | **5/5** @0.030 **+4.30** sf21 t=3.92 |
| R1 | 0.023 | 1.001 ✓ | **0.052** ✗ | +0.004 ✓ | ✗ | 5/5 @0.03 **+8.34** sf20 t=4.42 | 5/5 @0.03 **+14.5** sf18 t=4.61 | 5/5 +10.1 |
| R12 | 0.059 | 0.995 ✓ | 0.055 ✗ | +0.002 ✓ | ✗ | 5/5 +9.69 sf22 t=5.41 | 5/5 +16.1 sf22 t=5.77 | 5/5 +10.8 |
| R3 | 0.003 | 0.998 ✓ | 0.045 ✗ | −0.024 ✓ | ✗ | 5/5 +5.72 sf19 t=3.84 | 5/5 +8.71 sf19 t=3.88 | 5/5 +6.43 |
| U5 | 0.081 | 0.997 ✓ | 0.025 ✓ | −0.046 ✓ | ✓ | 5/5 +1.00 sf13 t=1.61 | 4/5 +0.58 sf11 t=0.78 | 5/5 +1.15 sf15 |

### R2 fold vector (the staged signal)

| Base | folds [F0..F4] e-5 |
|---|---|
| z_S3 @0.025 | [+3.17, +6.32, +6.78, +3.50, +0.31] |
| Slot3_twin @0.020 | [+2.21, +9.02, +8.19, +3.74, +1.80] |
| NNLS @0.030 | [+2.61, +6.95, +7.95, +3.88, +0.11] |

**First signal since Slot4 that clears the post-mortem dual gate with forensic + strength.**

R1 is the near-miss: strongest lift, dual-clean, but **F2=0.052** (needs <0.035). Likely residual TE still weakly couples to Bo9 tabular members (GB/MLP).

---

## Staging

| Field | Value |
|---|---|
| File | `submissions/sub_v17_8_R2_residual_LGB_w0p025_NOSUBMIT.csv` |
| w* | **0.025** (from z_S3 grid; twin prefers 0.020) |
| Kaggle submit | **NOT done** |
| Pods | **0** |

Slot 5 still requires explicit authorization — R2 is the first dual+forensic candidate that meets the V17.7/V17.8 bar on paper.

---

## Implications

1. **Fit residuals of `p_S3`, not raw `y`** — the V16.2 failure mode is fixed (`|ρ|<0.08`).
2. **Raw sub-bin / frequency features still have juice** that Bo9 remixing cannot create (R1/R2 vs V17.7 saturation).
3. Next hygiene for R1: stronger orth (Bo10+GB/MLP products) or TE only on pure digit keys to clear F2=0.052 → potentially larger dual lift than R2.

---

*End GM V17.8*
