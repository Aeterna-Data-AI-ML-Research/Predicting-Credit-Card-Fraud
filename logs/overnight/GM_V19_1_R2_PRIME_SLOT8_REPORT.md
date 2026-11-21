# GM V19.1 — R2_Prime Structural Upgrade Research → **HOLD Slot5 (NO Slot8 Submit)**

**0 GPU pods · research-first · submit withheld**  
Script: `scripts/gm_v19_1_r2_prime_structural_upgrade_and_slot8.py`  
Artifacts: `/tmp/s6e11_v16/out/v19_1.json`, `v19_1_families.npz`  
Runtime: ~522s (+62s extended H2H)

---

## Verdict

**Do not submit Slot8.** Multi-arch bags **improve Slot3 OOF** over 1-seed `R2_v178`, but **no family or blend replaces Slot5** with `5/5×3` head-to-head (`min_fold ≥ 0` vs `0.020·R2` on all three Slot3 bases).

| Decision | Detail |
|---|---|
| Live action | **HOLD ATH `0.95394`** |
| Slot8 CSV | **not written / not submitted** |
| Why | H2H replacement gate failed for every candidate (always ≥1 fold prefers original R2) |

This matches the research bar: *submit only if sure of a higher LB score*. Uniform OOF dominance over Slot5 was not achieved.

---

## Step 1 — Slot5 anchor + deterministic matrices

| Check | Result |
|---|---|
| Rebuild `qlock(L(Slot3)+0.020·orth_R2, LIVE)` vs LIVE Slot5 | **`1−ρ = 0`** (exact match) |
| `X_tab` (V17.8 `build_X`) | **45** cols (39 TE-keys as numeric + freq + 4 raw + orig20k) |
| `X_l46c12` | 12 deterministic flag/channel/q250/step cols |
| `X_tab51` | **57** cols |

---

## Step 2 — Individual family audit (orth vs `Bo9`, α=10)

| Family | `raw_disp` | `raw_KS` | F2 | Slot3@0.020 | Twin@0.020 | Clean? |
|---|---|---|---|---|---|---|
| **R2_v178_orig** | **0.990** | 0.052 | **0.033** ✓ | 5/5 +3.48 min+0.72 sf20 t=3.71 | **+4.99 min+1.80** | ✓ (anchor) |
| **F_lgb_ms39** (15×) | 0.991 | 0.030 | **0.033** ✓ | 5/5 +3.83 min+2.53 sf21 t=3.96 | **+5.65 min+2.75** | ✓ |
| F_lgb_ms51 (+L4.6C) | 0.992 | 0.035 | **0.045** ✗ | 5/5 +3.02 | +4.15 **min−1.32** | ✗ F2/twin |
| **F_cat_obliv** (10×) | 0.993 | **0.008** | **0.022** ✓ | 5/5 **+5.14** min+2.73 sf21 t=5.14 | **+8.00 min+3.68** | ✓ |
| F_hgb_bag | 0.979 | 0.071 | **0.068** ✗ | 4/5 +1.76 | min−11.6 | ✗ |
| G1e | 1.019 | 0.017 | **0.208** ✗ | 4/5 | twin **−3.20** | ✗ (Slot4 loser) |

**Clean upgrade parts:** `F_lgb_ms39`, `F_cat_obliv` only.  
**L4.6C / HistGB / G1e:** quarantined (F2 and/or twin collapse).

---

## Step 3 — Head-to-head vs Slot5 (`0.95394`)

Definition (exact replacement test):

\[
\Delta = \mathrm{AUC}(z_{S3}+w\cdot u_{\mathrm{Prime}}) - \mathrm{AUC}(z_{S3}+0.020\cdot u_{R2})
\]

on `{z_S3, twin, NNLS}` — require **5/5×3**, `meanΔ>0`, `min_fold≥0`, plus `1−ρ(Slot5)∈[0.18,0.50]e-4`.

### Auto blend grid (clean families only)

All `w_cat∈{0,0.20,0.35}` blends with LGB39: **NO H2H win** (4/5, negative min).

### Extended pure / high-Cat sweep (post-hoc)

| Candidate | Best H2H pattern | 5/5×3? |
|---|---|---|
| LGB39 alone | 4/5 means; twin often 3/5; min always &lt;0 on ≥1 base | **No** |
| CAT alone | Strong meanΔ on S3/NNLS but twin **min ≈ −5.5e-5** (3/5) | **No** |
| 0.65/0.50/0.35/0.20 LGB+CAT | MeanΔ often +1–3e-5 but **twin min always negative** | **No** |

**BEST 5/5 H2H: none.**

Interpretation: bagging / CatBoost **raises average Slot3 lift** and shrinks some fold mins vs 1-seed R2, but the **fold geometry that Slot5 already monetized** is not Pareto-dominated. Replacing R2 trades away at least one fold’s contribution — exactly the failure mode that makes live ATH replacement unsafe after Slot6/7.

---

## Confidence gates (submit bar)

| Gate | Result |
|---|---|
| `raw_disp∈[0.92,1.08]` on clean families | ✓ |
| Forensic F2/F3 on clean families | ✓ |
| Beat Slot3 harder than R2 (twin mean/min) | ✓ (CAT/LGB39) |
| **H2H 5/5×3 vs Slot5** | **✗** |
| `1−ρ` band with H2H pass | n/a |
| Neg-ray (would apply only if H2H passed) | n/a |
| **Submit** | **False — HOLD** |

---

## Implications

1. **Slot5’s 1-seed R2 is fold-efficient, not merely under-bagged.** Variance reduction helps OOF means but does not unlock a uniform replacement.
2. **Do not add L4.6C into the R2 matrix** — F2 jumps to 0.045 and twin min goes negative (same class of hygiene failure as V18.7 logcnt).
3. **CatBoost oblivious bag is the strongest *Slot3-side* residual seen** (+8e-5 twin mean) — keep as ammo for a *different* stacking geometry (e.g. small additive orth to Bo10_S5), **not** as a drop-in R2 replacement without H2H 5/5.
4. **Next live slot** needs a signal that is (a) deterministic-tabular or otherwise `raw_disp≈1`, and (b) **5/5 positive as Δ vs Slot5**, not merely strong vs Slot3.

---

*End GM V19.1 — no Kaggle submit*
