# GM V17.4 — Clean `G1` (BPE vs QLM Asym) via Pre-Nonlinearity Distribution Lock

**STRICT `NO_SUBMIT=True` · 0 GPU pods · local CPU**

Script: `scripts/gm_v17_4_clean_g1_bpe_qlm_asym.py`  
JSON: `/tmp/s6e11_v16/out/v17_4.json`  
Base: `z_S3` AUC **0.9524402** · stage rule: **F1∧F2∧F3 ∧ 5/5 ∧ sf≥18 ∧ t≥2.50**

---

## Verdict

1. **Pre-squaring σ drift is real but modest:** `std(P_BPE−P_QLM)` OOF **0.1692** vs Test **0.1607** (ratio **0.950**). MS−QLM and G26−QLM both slightly tighter on Test; G26_f0 is not a huge σ shock alone.
2. **`unitize(L_fold / L)` locks moments:** mean **0**, std **1**, skew **0** on OOF and Test; excess kurtosis **1.195 vs 1.197** (tiny N-driven gap only).
3. **Distribution lock does NOT cure `G1a = d·|d|`.** Locked `d_all_G1a` still fails F2/F3 (Family_E leak **0.107**, LIVE_DC **−0.055**). The F2/F3 problem is **nonlinear covariance with `Family_E` / LIVE residual**, not primarily σ_d mismatch.
4. **One variant staged:** `d_all_G1e_risk_gated_d` on **Bo9** — `d·unitize(z_S3)` — passes all forensic + strength bars.

Staged (NOSUBMIT only):  
`submissions/sub_v17_4_d_all_G1e_risk_gated_d__Bo9_NOSUBMIT.csv` @ **w\*=0.035** (5/5, meanΔ **+7.12 e-5**, sf **20/25**, t=**3.36**).

---

## Step 1 — Pre-squaring drift & distribution lock

| Quantity | OOF | Test | Notes |
|---|---|---|---|
| `std(P_BPE − P_QLM)` | **0.16917** | **0.16066** | ratio 0.950 |
| `std(unitize(MS) − P_QLM)` | 0.2243 | 0.2166 | 5-fold pure side |
| `std(unitize(G26) − P_QLM)` | 0.1753 | 0.1695 | G26 5f OOF / 1f Test |
| `std(G1_v173 = d\|d\|)` | (see JSON) | | scale ∝ σ_d³ theoretically |

**Locked `d` moments** (`d_all`, `d_5f`, `d_bt`):

| | mean | std | skew | excess kurtosis |
|---|---|---|---|---|
| OOF | 0.000000 | 1.000000 | 0.000000 | 1.195241 |
| Test | 0.000000 | 1.000000 | 0.000000 | 1.197381 |

Rank-logit `L` + `unitize` forces location/scale/skew match; kurtosis differs only by sample size (699960 vs 299984).

### V17.3 `G1` F2/F3 autopsy

After `frozen_lofo_orth` vs Bo9:

| Gate | Value | Driver |
|---|---|---|
| F2 max \|corr(orth_t, Bt9)\| | **0.0674** | **`Family_E`** (then d1 −0.036, MS +0.027, GPD +0.026, G26 +0.024) |
| F3 corr(orth_t, live_resid) | **−0.0891** | same Family_E-aligned residual direction |

`corr(live_resid, G26) ≈ 0` and `corr(live_resid, raw_d) ≈ +0.004` → **F3 is not “LIVE still wants G26_f0 noise.”**  
It is: **nonlinear BPE−QLM residual co-moves with the LIVE−Ridge(Bt9) residual along `Family_E`**, which already encodes (R3−L7F) structure overlapping QLM/BPE disagreement.

---

## Step 2–3 — Locked G1 family scorecard

Transforms on `d ∈ {d_all, d_5f}` (and `d_bt` for G1f), each under **Bo9** and **Bo10=[Bo9, d]** (`ridge_α=10`).

> **Bo10 note:** Ridge does **not** force `corr(orth_t, d)=0`. Observed `|corr|` with the `d` column after Bo10 orth is typically **0.06–0.27** (signed_sq). Exact OLS residualization would zero it; α=10 leaves a soft projection — reported as `corr_d` below.

### Headline rows

| Signal | basis | F1 disp | F2 leak | F3 LIVE_DC | ver | pos/5 | w* | meanΔ | min | folds | sf | t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **d_all G1e risk_gated** `d·U(z_S3)` | **Bo9** | 1.002 ✓ | **0.023** ✓ (FE) | **+0.008** ✓ | **✓** | **5/5** | **0.035** | **+7.12** | +0.21 | [18.5, 4.07, 0.21, 8.86, 3.96] | **20** | **3.36** |
| d_all G1e | Bo10 | 1.009 ✓ | 0.134 ✗ (d) | −0.012 ✓ | ✗ | 5/5 | 0.040 | +7.14 | +0.30 | [19.2, 4.20, 0.30, 9.13, 2.84] | 21 | 3.38 |
| d_all G1a signed_sq | Bo9 | 0.966 ✓ | **0.107** ✗ (FE) | **−0.055** ✗ | ✗ | 5/5 | 0.025 | +2.33 | +0.11 | [7.45, 0.11, 1.08, 0.43, 2.60] | 15 | 2.60 |
| d_all G1a | Bo10 | 0.953 ✓ | 0.266 ✗ (d) | −0.033 ✓ | ✗ | 5/5 | 0.015 | +1.23 | +0.10 | [3.77, 0.37, 0.84, 0.10, 1.08] | 15 | 2.28 |
| d_all G1b p=1.5 | Bo9 | 0.964 ✓ | 0.210 ✗ (FE) | −0.012 ✓ | ✗ | 5/5 | 0.015 | +1.14 | +0.03 | [3.92, 0.03, 0.59, 0.10, 1.04] | 15 | 1.92 |
| d_all G1c pos hinge² | Bo9 | 1.004 ✓ | 0.040 ✗ (FE) | −0.046 ✓ | ✗ | 3/5 | 0.009 | +0.19 | −0.30 | … | 13 | 0.49 |
| d_all G1d neg hinge² | Bo9 | 0.975 ✓ | 0.052 ✗ (FE) | −0.003 ✓ | ✗ | 4/5 | 0.012 | +0.91 | −0.21 | … | 15 | 1.81 |
| d_all G1f BPE−TAB sq | Bo9 | **0.860** ✗ | 0.041 ✗ | **−0.188** ✗ | ✗ | 3/5 | 0.006 | −0.06 | −0.62 | … | 13 | −0.23 |
| d_5f G1a signed_sq | Bo9 | 0.973 ✓ | 0.080 ✗ (FE) | −0.033 ✓ | ✗ | 3/5 | 0.025 | +0.87 | −0.50 | … | 16 | 1.45 |
| d_5f G1e risk_gated | Bo9 | 0.995 ✓ | **0.0354** ✗ (d1) | −0.057 ✗ | ✗ | 4/5 | 0.035 | +4.28 | −3.64 | … | 18 | 2.87 |
| d_5f G1c pos hinge² | Bo9 | 1.007 ✓ | 0.027 ✓ | **−0.053** ✗ | ✗ | 5/5 | 0.009 | +0.38 | +0.01 | … | 17 | 1.33 |
| **v173 G1 raw** | Bo9 | 1.028 ✓ | 0.067 ✗ (FE) | −0.089 ✗ | ✗ | 5/5 | 0.035 | +3.08 | +0.08 | [8.88, 0.08, 0.53, 2.18, 3.71] | 18 | 2.87 |

Full rows (Wilcoxon / CI / Bo10) in `/tmp/s6e11_v16/out/v17_4.json`.

### What the lock fixed vs what it didn’t

| Hypothesis | Result |
|---|---|
| σ_d OOF≠Test → F2/F3 via Cov(d\|d\|, basis) | σ ratio 0.95 confirmed; **locking d does not clear G1a F2** (leak **worsens** 0.067→0.107 on Family_E) |
| Drop G26 (d_5f) to remove 1-fold Test noise | G1a **weakens** (3/5); G1e **loses** F2/F3 — MS-only gap is not the clean winner |
| Project out linear `d` via Bo10 | Soft Ridge leaves `corr_d≠0`; for G1a, F2 migrates onto the `d` column (0.27) |

**Physical read:** V17.3’s OOF strength in `d·|d|` was partly **Family_E-collinear nonlinear mass**. Rank-locking equalizes margins but does not orthogonalize that direction. The clean survivor is the **risk-gated linear gap** `G1e = d · U(z_S3)`, which keeps F2/F3 inside bounds and boosts tail sensitivity without the cubic moment that couples to Family_E.

---

## Staging

| Field | Value |
|---|---|
| Winner | `d_all_G1e_risk_gated_d__Bo9` |
| File | `submissions/sub_v17_4_d_all_G1e_risk_gated_d__Bo9_NOSUBMIT.csv` |
| w* | **0.035** |
| pos / sf / t | **5/5**, **20/25**, **3.36** |
| Forensic | F1=1.002, F2=0.0226, F3=+0.0081 |
| Kaggle submit | **NOT done** (`NO_SUBMIT=True`) |
| Pods | **0** |

No other variant cleared `t≥2.50` + forensic. Locked `G1a` signed-square remains a **false friend**.

---

## Implications

- Curing σ_d drift is **necessary hygiene**, not sufficient for `G1 = d|d|`.
- Next submit-worthy probe (when authorized): **G1e @ 0.035** on Slot3 — not raw/locked signed-square G1.
- Bo10 exact-zero residualization (OLS / explicit `d`-partial) is optional follow-up; it would not revive G1a on Bo9 forensics already failing before the `d` column.

---

*End GM V17.4*
