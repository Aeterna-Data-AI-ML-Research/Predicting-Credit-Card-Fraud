# GM V18.0 — Slot 5 Upgrade Audit (`0.95394` base)

**STRICT `NO_SUBMIT=True` · 0 GPU pods · local CPU**  
Script: `scripts/gm_v18_0_slot5_base_r1_orth_and_r2_v2.py`  
Artifacts: `/tmp/s6e11_v16/out/v18_0.json`, `v18_0_residuals.npz`  
Elapsed: **~177s** (Step1~60s + expanded bank + 6× LGB; over 45s budget due to multi-seed R2_v2)

---

## V17.9 `R1_pure` reference (pre–Slot 5)

From `GM_V17_9_R2_SUBMIT_AND_R12_CLEAN.md`:

| Item | Value |
|---|---|
| F2 | **0.0343** ✓ (cured from 0.052 by dropping q1000) |
| Dual/triple | **5/5** all bases @ w=0.03 |
| z_S3 | +7.10 · sf19 · t=4.34 |
| Slot3_twin | +10.78 · sf18 · t=4.66 |
| NNLS | +7.82 |

---

## Step 1 — Slot 5 baselines + `orth(R1_pure \| Bo10_S5)` + `Δw_R2`

### Bases

| Base | Construction | OOF AUC |
|---|---|---|
| `z_S5` | `z_S3 + 0.020 · unitize(orth_R2)` | 0.9524595 |
| `z_Slot5_twin` | `z_Slot3_twin + 0.020 · u_R2` | 0.9518462 |
| `z_NNLS_S5` | `z_NNLS_twin + 0.020 · u_R2` | 0.9522822 |
| `Bo10_S5` | `[Bo9, unitize(orth_R2)]` | — |

### `Δw_R2 ∈ {+0.005, +0.008, +0.010}` on Slot 5 (extra R2 alone)

| Δw | z_S5 | Slot5_twin | NNLS_S5 | Verdict |
|---|---|---|---|---|
| +0.005 | 4/5 +0.53 sf16 t=1.66 | **2/5 −0.06 sf10 t=−0.62** | 4/5 +0.48 sf18 t=1.61 | **FAIL** |
| +0.008 | 4/5 +0.82 sf16 t=1.66 | **2/5 −0.38 sf10 t=−0.62** | 4/5 +0.69 sf18 t=1.61 | **FAIL** |
| +0.010 | 4/5 +0.93 sf16 t=1.66 | **2/5 −0.73 sf10 t=−0.62** | 4/5 +0.81 sf18 t=1.61 | **FAIL** |

**`w_R2 = 0.020` is saturated on Slot 5.** Twin rejects any further R2 weight.

### `orth(R1_pure \| Bo10_S5)` — **STAGE ✓**

| Gate | Value |
|---|---|
| F1 disp | **0.997** ✓ |
| F2 max\|ρ\| | **0.0316** (Family_E) ✓ |
| F3 corr live resid | **−0.0056** ✓ |
| preρ vs z_S5 | +0.040 |

| Base | pos | w★ | meanΔ×1e5 | min_fold | sf25 | t |
|---|---|---|---|---|---|---|
| z_S5 | **5/5** | 0.03 | **+6.41** | +4.54 | **19** | **4.53** |
| Slot5_twin | **5/5** | 0.03 | **+9.46** | +5.29 | **19** | **4.13** |
| NNLS_S5 | **5/5** | 0.03 | **+7.21** | +4.27 | **20** | **4.40** |

R1 signal remains **orthogonal to Slot5’s R2** and still dual/triple-elite. Slight OOF attenuation vs V17.9 on S3 (+7.10→+6.41) is expected after projecting out Bo10’s R2 column.

### `orth(R1\|Bo10) + Δw_R2`

| Combo | z_S5 | twin | NNLS | Stage? |
|---|---|---|---|---|
| R1@0.03 + Δw=0.005 | 5/5 +6.92 sf**18** | 5/5 +9.33 sf**17** | 5/5 +7.69 sf**18** | **FAIL sf&lt;19** |
| R1@0.03 + Δw=0.008 | 5/5 +7.20 sf**18** | 5/5 +8.94 sf**17** | 5/5 +7.92 sf**18** | **FAIL sf&lt;19** |

Mean lifts a bit on S5/NNLS but **subfold reliability drops** and twin mean softens — do **not** co-add Δw_R2.

---

## Step 2 — Expanded Newton bank `R_v2` on `r_S5 = y − p_S5`

| Bank | Count |
|---|---|
| LGB continuous features | **94** (all-col `log1p(freq)`, `log1p(orig20k)`, row orig hits/mean, pair freqs, nums, digit keys) |
| Newton TE keys | **82** (freq bins, orig bins, cat×cat / cat×digit pairs, pure digit keys — **no q1000**) |
| R2_v2 | 3 seeds × 2 `num_leaves` ∈ {6,10}, `n_estimators=50`, `min_child=500`, `colsample=0.7` → **6 models avg** |

preρ vs z_S5: R1_v2=+0.040 · R2_v2=+0.129

---

## Step 3 — Unified Slot 5 scorecard

| Candidate | F1 | F2 | F3 | z_S5 | twin | NNLS | Stage |
|---|---|---|---|---|---|---|---|
| **Δw_R2 alone** | — | — | — | 4/5 | **2/5** | 4/5 | ✗ |
| **`orth(R1_pure\|Bo10)`** | ✓ 0.997 | ✓ 0.032 | ✓ | **5/5 +6.41 sf19 t4.53** | **5/5 +9.46 sf19 t4.13** | **5/5 +7.21 sf20 t4.40** | **✓** |
| R1\|Bo10 + Δw_R2 | — | — | — | 5/5 sf18 | 5/5 sf17 | 5/5 sf18 | ✗ (sf) |
| `orth(R1_v2\|Bo10)` | ✓ 0.996 | ✓ 0.032 (R2_orth) | ✓ | 5/5 +1.68 sf18 | **4/5** | 5/5 +2.49 | ✗ |
| `orth(R2_v2\|Bo10)` | ✗ 0.858 | ✗ **0.185→R2_orth** | ✓ | 3/5 | 3/5 | 4/5 | ✗ |
| `orth(R1_v2+R2_v2\|Bo10)` | ✗ 0.930 | ✗ 0.123→R2_orth | ✓ | 4/5 | 4/5 | 4/5 | ✗ |

**R2_v2 collapses into Slot5’s existing R2 column** (F2 leak 0.185) — expanded features largely rediscover the same residual already harvested at w=0.020. R1_v2 is forensic-clean but weaker / twin-unstable vs the V17.9 pure-digit R1.

---

## Staged NOSUBMIT

| File | Role |
|---|---|
| `submissions/sub_v18_0_R1_pure_Bo10_w0p03_NOSUBMIT.csv` | Slot5 + 0.03 · `orth(R1_pure\|Bo10)` — **next-slot ammo** |

No other candidate met `F1∧F2∧F3 ∧ 5/5 ∧ sf≥19 ∧ t≥3.0` on all three Slot5 baselines.

---

## Guardrails

| Check | Status |
|---|---|
| Kaggle submit | **NOT done** (`NO_SUBMIT=True`) |
| GPU pods | **0** |
| Daily quota | untouched |

---

## Bottom line

1. **Do not raise `w_R2` past 0.020** on Slot 5 — twin kills it.  
2. **`orth(R1_pure | Bo9+R2)` is the clean Slot-6 candidate** — still 5/5×3 with F2=0.032.  
3. Expanded `R_v2` on `r_S5` does **not** beat V17.8/9 residual harvest under Bo10 orth; next residual work should seek features **orthogonal to both Bo9 and R2**, not denser rediscovery of the same bank.

---

*End GM V18.0*
