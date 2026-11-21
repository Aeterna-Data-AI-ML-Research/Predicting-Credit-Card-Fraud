# GM V18.3 — Pure Sub-Bin Cents / LogFreq R2 Audit

**STRICT `NO_SUBMIT=True` · 0 GPU pods · local CPU (~111s; over 40s due to 5+3+3 LGB bags)**  
Script: `scripts/gm_v18_3_pure_subbin_cents_and_logfreq_r2.py`  
Artifacts: `/tmp/s6e11_v16/out/v18_3.json`, `v18_3_residuals.npz`

---

## Verdict

**No stage.** Exact45 seed-bag is forensic-clean and slightly stronger OOF than cached `R2_v178`, but **does not beat Slot5 on twin/NNLS**. Pure sub-bin on `r_S5` **collapses into `R2_orth`** (F2=0.215). Hold ATH **0.95394**.

| Candidate | F1 | F2 | Beat Slot5? | Stage |
|---|---|---|---|---|
| R2_v178 (ref) | ✓ | ✓ 0.033 | — | hold |
| **R2_exact45_bag** (5seed⊕cached) | ✓ | ✓ 0.031 | S3 5/5; **twin/NNLS 3/5** | ✗ |
| R2_exact45_seeds_only | ✓ | ✓ 0.030 | 3/5 all | ✗ |
| R2_subbin_S5 (addon Bo10) | ✗ 0.855 | ✗ **0.215→R2** | negative | ✗ |
| R2_subbin_S3 | ✓ | ✗ 0.036 | 1/5 | ✗ |
| subbin⊕v178 α=0.3 | ✓ | ✓ 0.035 | 3–4/5 | ✗ |
| subbin⊕v178 α=0.5 | ✓ | ✗ 0.035 | 2–4/5 | ✗ |

---

## Step 1 — Exact V17.8 45-col 5-seed bag

Recipe: `num_leaves=6`, `n_estimators=40`, `min_data=500`, `colsample_bytree=1.0`, `subsample=1.0`, `colsample_bynode=0.85`, seeds `[0,42,2026,7,99]`, then `0.5·bag ⊕ 0.5·cached_R2`.

| Check | Value |
|---|---|
| ρ(bag, cached) | 0.947 |
| ρ(exact45_bag, cached) | **0.987** |
| F2 | **0.0314** ✓ (better than 0.0332) |

| Base | Absolute | REPLACE vs `0.020·R2_v178` |
|---|---|---|
| z_S3 | 5/5 @0.03 **+4.64** sf19 t3.86 | **5/5 @0.025 Δ=+0.74** |
| Slot3_twin | 5/5 @0.025 +5.47 sf19 t3.55 | **3/5 Δ=+0.49** |
| NNLS | 5/5 @0.03 +4.50 sf21 t3.76 | **3/5 Δ=+0.61** |

Seeds-only (no cached blend): absolute strong, replace still 3/5. **Seed bag smooths noise but does not unlock a Slot5 replacement.**

---

## Step 2 — `X_subbin` (86 cols; zero raw floats, zero q1000)

Families only: exact float `logfreq`, cents/tenths/ones/mod/round indicators, `(amt_cents, bal_cents)` & merchant×amount pair freqs, cat×cents/ones crosses.

| Model | Target | Basis |
|---|---|---|
| `R2_subbin_S5` | `g_S5`, wt `h_S5` | orth vs **Bo10_S5** |
| `R2_subbin_S3` | `g_S3` | orth vs Bo9 |
| `subbin⊕v178` | blend α∈{0.3,0.5} on S3 | orth vs Bo9 |

**`R2_subbin_S5` autopsy:** F1=0.855, F2=**0.215 leak into `R2_orth`** — fitting the same sub-bin families on `r_S5` rediscovers Slot5’s residual; orth against Bo10 kills the signal (3/5 / 2/5 negative).

---

## Step 3 — Unified scorecard

### Absolute (e-5)

| Signal | preρ | F2 | z_S3 / S5 | twin | NNLS |
|---|---|---|---|---|---|
| R2_v178 | 0.062 | ✓0.033 | **+4.02** sf20 t3.71 | **+4.99** sf20 | **+4.30** |
| exact45_bag | 0.060 | ✓0.031 | +4.64 sf19 | +5.47 | +4.50 |
| subbin_S5 | 0.068 | ✗0.215 | −0.19 on S5 | −1.12 | −0.38 |
| subbin_S3 | 0.072 | ✗0.036 | +1.44 | +1.14 | +1.12 |
| ⊕v178 α0.3 | 0.072 | ✓0.035 | +3.48 sf19 | +4.83 | +3.42 |
| ⊕v178 α0.5 | 0.075 | ✗0.035 | +2.79 | +4.24 | +1.98 |

### Direct Δ vs Slot5 (`z + w·new − z + 0.020·R2_v178`)

| Signal | S3 | twin | NNLS |
|---|---|---|---|
| exact45_bag | +0.74 (5/5) | +0.49 (**3/5**) | +0.61 (**3/5**) |
| seeds_only | +0.79 (3/5) | +0.62 (3/5) | +0.63 (3/5) |
| subbin_S3 | −2.03 (1/5) | −4.09 | −2.37 |
| ⊕ α0.3 | +0.51 (4/5) | −0.30 (3/5) | +0.44 (4/5) |
| ⊕ α0.5 | +0.42 (4/5) | −0.47 (2/5) | +0.24 (3/5) |

---

## Staged NOSUBMIT

**None** — no candidate cleared `F1∧F2∧F3` **and** beat Slot5 on all 3 baselines with elite sf/t.

---

## Guardrails

| Check | Status |
|---|---|
| Submit | **NOT done** |
| Pods | **0** |
| ATH | **Slot5 0.95394** |

---

## Bottom line

1. **Dropping coarse floats/q1000 into a pure sub-bin bank does not create a Slot5 add-on** — on `r_S5` it is mostly the same residual already harvested.  
2. **Exact45 seed bag is the only forensic upgrade** (F2 0.031, S3 replace +0.74) but twin rejects a full Slot5 swap.  
3. Next residual must be **orthogonal to both Bo9 and R2** with a *new* physical family — not denser cents/logfreq of the same columns.

---

*End GM V18.3*
