# GM V20.0 — Wilcoxon-AUC + Method-B Bo12 + clean P4 → **HOLD**

**0 GPU pods · elapsed 824s · script `scripts/gm_v20_0_wilcoxon_auc_gradient_and_p4_super_on_slot8.py`**

| | |
|---|---|
| Slot8 ATH | **0.95395** (hold) |
| Slot8 rebuild `1-ρ` | **0** (exact) |
| Submit | **False** — 0 winners |
| Daily slots used | **0** (2 remain) |
| P4 kept | `P4_L4.2_pure5f`, `P4_L4.5_pure5f` |
| P4 quarantined | `P4_L4.3_pure5f` (KS=0.035 ≥ 0.020) |

---

## Flaw fixes verified
1. **Unweighted `g_AUC`** (`sample_weight=None`) for W1/W2 — no `|g|·|g|` positive explosion.
2. **Method-B**: Bo9 frozen LOFO → variance-matched orth vs `[R2, Cat, U5, M5, R1]`.
3. **No KMAT_68**; L4.3 quarantined; L4.2 KS=0.0014 / L4.5 KS=0.0061 kept.
4. **Full `tr`** (559,968 rows/fold) — no `q>0.50` truncation.

## Wilcoxon diagnostics
| Fold | corr(g_AUC, g_BCE) | Twilight E `g_AUC²` | Twilight E `g_BCE²` |
|------|--------------------|---------------------|---------------------|
| 0..4 | ≈ **+0.75** | ≈ **0.56** | ≈ **0.86** |

`g_AUC` shifts energy **out of** the contested twilight band relative to BCE — expected for rank-AUC focus, but after Slot8 saturation that residual is weak/noisy.

## Dual-basis orth (Method-B)

| Family | F2_Bo9 | F2_Bo12 | disp12 | KS12 | Gate2? |
|--------|--------|---------|--------|------|--------|
| W1 cat g_AUC | 0.120 ❌ | 0.119 ❌ | 0.993 | 0.011 | FAIL |
| W2 lgb g_AUC | 0.125 ❌ | 0.124 ❌ | 0.995 | 0.023 | FAIL |
| W3 hyb+U58 | 0.035 ~ | 0.112 ❌ | 0.858 | 0.042 | FAIL |
| W12 | 0.125 ❌ | 0.123 ❌ | 0.994 | 0.016 | FAIL |
| W4a P4×auc | 0.100 ❌ | 0.100 ❌ | 0.995 | 0.043 | FAIL |
| W4b P4×bce | 0.133 ❌ | 0.132 ❌ | 0.983 | 0.042 | FAIL |
| W4 blend | 0.084 ❌ | 0.083 ❌ | 0.992 | 0.043 | FAIL |

**Method-B did not lock `F2_Bo12 < 0.035`.** New tab45/P4 trees remain correlated with the Slot8 increment basis on Test (~0.08–0.13) — same OOF/Test asymmetry as V19.12 Flaw #2, only partially mitigated.

## Best near-misses (not gate-clear)
| Candidate | 1-ρ e-4 | c7 | z_S8 | twin | nnls | Blocker |
|-----------|---------|----|------|------|------|---------|
| W2_orth12_w0.008 | 0.049 | +0.033 | 4/5 m=+0.78 | **5/5** m=+2.35 | 4/5 | F2 + ρ too small + not 5/5×3 |
| W4b_bce_w0.012 | 0.225 | −0.030 | 4/5 m=+1.17 | 3/5 | **5/5** | F2 + twin |
| G2_W2 r0.016/c0.012/u0.014/w0.006 | 0.032 | +0.049 | 4/5 | 5/5 | 4/5 | F2 + ρ + min_fold |

No candidate cleared **all 6 gates** (F2_Bo12 never < 0.038 on Method-B residuals; H2H never 5/5×3 with `1−ρ∈[0.30,1.10]`).

## Diagnosis
1. Unweighted Wilcoxon targeting is correctly implemented and shifts residual mass vs BCE.
2. That residual, after Method-B whitening, is **too collinear with Slot8’s R2/Cat/U5** for a clean additive or triad upgrade.
3. Clean P4_L4.2/4.5 interactions are hygiene-OK but do not unlock Slot9 vs 0.95395.

## Decision
**HOLD** ATH `0.95395`. Keep remaining daily slots.

Artifacts: `/tmp/s6e11_v16/out/v20_0.json`, `v20_0_families.npz`, `logs/overnight/GM_V20_0_WILCOXON.log`.
