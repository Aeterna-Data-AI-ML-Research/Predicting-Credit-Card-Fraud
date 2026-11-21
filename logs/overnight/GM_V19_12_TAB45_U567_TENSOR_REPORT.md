# GM V19.12 — Strict F2<0.035 tab45⊗U567 → **HOLD**

**0 GPU pods · elapsed 2186s · script `scripts/gm_v19_12_strict_tab45_x_u567_tensor_and_slot9.py`**

| | |
|---|---|
| Slot8 ATH | **0.95395** (hold) |
| Slot8 rebuild `1-ρ` | **0** (exact) |
| Submit | **False** — 0 winners |
| Daily slots used | **0** (2 remain) |
| X_tab45 | 43 cols |
| X_mod | 35 cols (cross-products only) |

---

## Step1 — U5..U10 forensic bank (raw-LLR order stats)

| U | F2 vs Bo9 | F2 vs Bo12_S8 | raw_disp | c7 | folds @ w=0.010 vs S8 (e-5) | mean |
|---|-----------|--------------|----------|----|------------------------------|------|
| U5 | **0.024** ✓ | 0.557† | 0.126† | +0.035 | [+2.07, −0.14, +0.21, +0.58, +0.34] | +0.61 |
| U6 | **0.026** ✓ | 0.110 | 0.946 | +0.054 | [+0.13, −0.64, +1.05, +0.64, −0.06] | +0.22 |
| U7 | **0.022** ✓ | 0.061 | 0.791 | +0.031 | [+2.01, −2.26, +0.96, +0.83, −1.78] | −0.05 |
| U8 | **0.033** ✓ | **0.035** ✓ | **0.998** | +0.003 | [+0.52, **+2.31**, +0.91, +1.37, −0.11] | +1.00 |
| U9 | **0.024** ✓ | 0.036 | 0.652 | −0.021 | [−0.28, +0.68, −0.07, +0.28, +2.23] | +0.57 |
| U10 | **0.021** ✓ | 0.109 | 0.776 | +0.006 | [−0.32, +0.29, +1.97, −0.57, −0.39] | +0.20 |

†U5 is **inside** Bo12_S8 → post-Bo12 orth of U5 alone is near-null (expected).

**Synergy-1 check:** U6 does peak on Fold2 (+1.05) vs U5 soft Fold2 (−0.14 after Bo12); U8 peaks Fold1–3. Blending them (`U567_comp`, `U6789`) keeps `|c7|<0.05` but **does not** clear 5/5×3 vs saturated Slot8 (U5 already in the recipe).

---

## Engine forensics (F2 lock)

| Engine | Candidate | F2_Bo9 | F2_Bo12 | disp | KS | H2H verdict |
|--------|-----------|--------|---------|------|-----|-------------|
| A | U567_comp | 0.036 | 0.036 | 1.00 | 0.027 | negative / ≤1/5 |
| A | U6789 | 0.049 | 0.049 | 1.00 | 0.019 | ≤3/5, twin collapses |
| A | U8-driven mixes | ~0.05 | ~0.05 | ~1 | ok | soft positive mean, not 5/5 |
| B | F_xgb_S8 | **0.030** ✓ | **0.166** ❌ | 0.875 | 0.010 | negative (already in R2/Cat span) |
| B | F_cat_S8 | **0.014** ✓ | **0.255** ❌ | 0.629 | 0.046 | dead after Bo12 orth |
| B | F_xc_S3 | **0.018** ✓ | 0.102 ❌ | 0.734 | 0.017 | no lift |
| C | **M_mod_ridge** | **0.009** ✓ | **0.027** ✓ | **0.998** | 0.014 | **F2 lock OK**, H2H **negative** |
| C | **M_mod_cat** | **0.003** ✓ | **0.027** ✓ | **0.989** | 0.021 | F2 lock OK, H2H negative |
| C | **M_mod_blend** | **0.008** ✓ | **0.021** ✓ | **0.997** | 0.013 | cleanest F2; still −0.7e-5 mean |

**Synergy-2 confirmed:** pure `X_tab45⊗U567` cross-products achieve **`F2_Bo12 ≈ 0.021 < 0.035`** with `raw_disp≈1.00` and `|c7|≈0`. Signal after orth is fold-fragile noise vs Slot8 — multiplicative interactions do not add AUC once R2+Cat+U5 are locked.

**Synergy-3:** new XGB/Cat on pure X_tab45 are F2-clean vs Bo9 but **highly collinear with Bo12** (F2_Bo12 0.09–0.25) → orth cancels them.

---

## Mode2 (upgraded Slot8 formula vs ATH)

All Mode2 grids (`w_xgb` / `w_mod` / `U567` replace) produced **0/5 or 1/5** vs Slot8 with large negative mean lifts (−2…−6e-5). Replacing or diluting the exact `0.016/0.012/0.012` Slot8 recipe loses the live optimum.

---

## Gate scorecard (all 6 required)

| Gate | Best Engine C | Best Engine A/B | Pass? |
|------|---------------|-----------------|-------|
| 1. disp∈[0.92,1.08], KS<0.025 | ✓ blend | mixed | partial |
| 2. F2_Bo9<0.035, F2_Bo12<0.038 | ✓ **0.021** | often fail Bo12 | C only |
| 3. 5/5×3 min_fold>0 vs Slot8 | ✗ ≤2/5 | ✗ ≤4/5 | **FAIL** |
| 4. sf≥18 & t≥2.8 ×3 | ✗ | ✗ | **FAIL** |
| 5. 1−ρ ∈ [0.30, 1.10]e-4 | at lift-free w only | — | **FAIL** with H2H |
| 6. \|c6\|,\|c7\|<0.20 | ✓ ~0.00–0.05 | ✓ | pass |

**0 candidates cleared all 6.**

---

## Diagnosis
1. **F2 lock works** for zero-main-effect cross-mod (Engine C) — V19.11 cohort leak avoided.
2. **Slot8 already extracted the usable tab45×U5 additive margin**; orthogonal complement of `(X_tab45 ⊗ U567)` is near-null for AUC.
3. **U6/U7 fold complementarity is real on raw forensics but not additive after Bo12** — Slot8’s U5 term already owns that residual subspace.
4. Do **not** spend a daily slot on Engine C / Mode2 remixes.

## Decision
**HOLD** ATH `0.95395`. Keep remaining daily slots.

Artifacts: `/tmp/s6e11_v16/out/v19_12.json`, `v19_12_families.npz`, `logs/overnight/GM_V19_12_TAB45_U567.log`.
