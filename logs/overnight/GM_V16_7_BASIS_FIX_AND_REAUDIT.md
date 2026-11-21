# GM V16.7 — Critical Basis Fix (`LM_test ≠ L7F_test`) & Re-Audit

**STRICT `NO_SUBMIT=True` · 0 GPU pods · no Kaggle submit**

Script: `scripts/gm_v16_7_fix_megayak_lm_test_basis.py`  
JSON: `/tmp/s6e11_v16/out/v16_7.json`

---

## Step 1 — Locate true `megayak_5f_lm` test in Q17

Q17 (`external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz`) already has **distinct** columns:

| Stream | OOF key | Test key | Solo OOF AUC |
|---|---|---|---:|
| **LM** = `megayak_5f_lm` | `oof_megayak_5f_lm` | **`test_megayak_5f_lm`** | **0.950428** |
| **L7F** = `lm7_5fold` | `oof_lm7_5fold` | `test_lm7_5fold` | 0.949448 |

| Check | Value |
|---|---:|
| Spearman(LM, L7F) **OOF** | 0.9778 |
| Spearman(LM, L7F) **Test** | **0.9836** (< 0.999 ✓ distinct) |
| Identical LM_test == L7F_test? | **False** |
| Spearman(Q17 `lm7_test`, banks `lm7_test_5fold`) | 1.000 |
| Spearman(Q17 `megayak_test`, banks `lm7_test_5fold`) | 0.9836 |
| Spearman(Q17 megayak OOF, `/tmp/.../lm_oof.npy`) | 1.000 |

**Root cause of V16.4/V16.6 bug:** scripts hard-coded `add_t("LM", L(l7f_te)); add_t("L7F", L(l7f_te))` instead of `XT['megayak_5f_lm']`. Q17 was never missing the column — the loader was.

### Fixed 10-stream basis

`['z_ATH', 'd1_v4', 'Family_E', 'MS', 'G26', 'LM', 'L7F', 'GPD', 'GB', 'MLP']`  
with **`LM_test = L(Q17 test_megayak_5f_lm)`**, **`L7F_test = L(Q17 test_lm7_5fold)`**.

| Metric | Broken (LM←L7F) | **Fixed** |
|---|---:|---:|
| corr(LM, L7F) Test | **1.0000** | **0.9856** |
| corr(LM, L7F) OOF | 0.9804 | 0.9804 |
| `cond(Bt)` | **~4.8×10¹⁵** (singular) | **35.4** |
| `cond(Bo)` | — | 38.9 |

### OOF vs Test correlation mismatch

`max |corr_oof(i,j) − corr_test(i,j)|` = **0.1449** on pair **`(Family_E, L7F)`**  
(OOF 0.0783 → Test −0.0666). This is **not** cured by the LM fix (same max on broken basis) — Family_E is built as orth(`(L(R3)−L(L7F))·w_tail`), so OOF/Test Family_E construction still drifts vs L7F.

Other key pairs (fixed basis):

| Pair | corr OOF | corr Test | \|Δ\| |
|---|---:|---:|---:|
| LM–L7F | 0.9804 | 0.9856 | 0.005 |
| MS–LM | 0.9741 | 0.9780 | 0.004 |
| MS–G26 | 0.9733 | 0.9736 | 0.000 |
| G26–Family_E | −0.0254 | 0.0163 | 0.042 |

Full 10×10 matrices are in `v16_7.json` (`corr_oof_matrix`, `corr_test_matrix`).

---

## Step 2 — Forensic re-run on fixed basis (on top of `z_S3`)

Gates: `disp ∈ [0.93, 1.07]`, `max|Δβ| < 0.08`, `|corr LIVE resid| < 0.08`.

| Signal | `raw_disp` | max\|Δβ\| | \|corr LIVE\| | Verdict | Top Δβ drivers |
|---|---:|---:|---:|---|---|
| `q80−q20` | **0.9958** ✓ | **0.451** ✗ | 0.005 | **FAIL sym** | MS (−0.39→−0.84), GB, G26 |
| `R2−R3` curv | 0.8806 ✗ | **0.720** ✗ | 0.001 | **FAIL** | LM (0.64→−0.09), L7F |
| `sadam_v4_cat` | 0.8778 ✗ | **0.818** ✗ | 0.001 | **FAIL** | z_ATH (−1.01→−0.19), MS |
| `sameer_glm10f` | 0.7438 ✗ | **0.551** ✗ | 0.002 | **FAIL** | z_ATH (0.55→0.00) |
| `daoviet_gpdvae` | ~0 ✗ | ~0 | ~0 | **N/A** (member of basis = GPD; orth≈0) |
| `alex_372ea048` | 0.5089 ✗ | 0.514 ✗ | 0.004 | **FAIL** | GB, z_ATH, LM |
| `alex_6c55b429` | 0.8271 ✗ | 0.459 ✗ | 0.004 | **FAIL** | z_ATH, MS |

Broken-basis comparison (sadam / q80 / curv): **disp and Δβ essentially unchanged** — the LM duplicate inflated `cond(Bt)` but was **not** the main cause of β asymmetry. Fixing it restores a well-conditioned test basis; it does **not** mint forensic-clean add-ons.

### Megayak inject: does LIVE already have true LM?

Solo OOF gap LM − L7F = **+9.8 bps** (0.95043 − 0.94945), **not** ~101 bps.

| Probe | Result |
|---|---|
| ρ(LM_test, LIVE) | **0.9737** |
| ρ(L7F_test, LIVE) | 0.9653 |
| ρ(LM_test, Slot3) | 0.9740 |
| corr(orth(LM)\|basis∖LM, LIVE resid) | **0.0005** |
| Screen `L(LM)−L(L7F)` on `z_S3` | **4/5**, mean +0.88e-5, 15/25, t=1.05 |
| Screen `orth(LM)` excl LM col on `z_S3` | **4/5**, mean +3.12e-5, 16/25, t=2.39 |

**Conclusion:** LIVE/Slot3 already track megayak more tightly than L7F. The LM−L7F contrast is **not** a massive unused +101 bps lever; OOF screens stay **4/5**. No inject submit candidate.

---

## Step 3 — Screens & composites on `z_S3` (`NO_SUBMIT`)

| Signal | pos/5 | w* | meanΔ (e-5) | min | [F0..F4] | sf25 | t | disp | max\|Δβ\| | Pass? |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| `orth(q80−q20)` | **5/5** | 0.015 | +1.550 | +0.017 | [2.38, 2.23, 0.02, 1.36, 1.77] | 16 | 1.72 | 0.996 | 0.451 | forensic **FAIL** |
| `orth(sameer_glm10f)` | **5/5** | 0.006 | +1.106 | +0.004 | [1.73, 1.99, 1.34, 0.47, 0.00] | **20** | 2.89 | 0.744 | 0.551 | forensic **FAIL** |
| `orth(sadam_v4_cat)` | 4/5 | 0.015 | +1.228 | −0.607 | [3.60, 0.13, 0.86, 2.16, −0.61] | 17 | 2.00 | 0.878 | 0.818 | FAIL |
| `orth(R2−R3)` | 4/5 | 0.015 | +0.860 | −0.155 | [−0.16, 2.10, 1.00, 1.27, 0.08] | 15 | 1.69 | 0.881 | 0.720 | FAIL |
| `orth(alex_372ea048)` | 4/5 | 0.04 | +2.604 | −3.321 | [2.04, 2.06, −3.32, 6.29, 5.95] | 16 | 1.95 | 0.509 | 0.514 | FAIL |
| `orth(alex_6c55b429)` | 4/5 | 0.02 | +0.770 | −5.315 | [4.12, 1.60, 0.36, 3.09, −5.32] | 16 | 1.07 | 0.827 | 0.459 | FAIL |
| `orth(daoviet)` | 3/5 | — | ~0 | — | — | 6 | −2.24 | ~0 | ~0 | N/A (in basis) |

**Verified forensic-clean signals: `[]`.**  
**Verified ∩ 5/5: `[]`.**  
**Near-pass (disp+dc, Δβ<0.15): `[]`.**  
**Composites: not formed** (empty pool).  
**Staged `sub_v16_7_*_NOSUBMIT.csv`: none** (no verified winners).

---

## Verdicts

1. **Bug confirmed & fixed in script:** Q17 has true `test_megayak_5f_lm`; V16.4–V16.6 wrongly aliased it to `lm7_test_5fold`, making `Bt` singular (`cond~1e15`). Fixed basis: `corr(LM,L7F)_test=0.986`, `cond(Bt)=35`.
2. **Forensic outcome unchanged in substance:** no add-on clears `max|Δβ|<0.08`. Best remaining 5/5 screens (`q80−q20`, `sameer`) still fail symmetry / disp.
3. **Megayak inject is not a hidden +101 bps LIVE upgrade** — gap is **+9.8 bps** solo; LIVE already correlates 0.974 with LM; OOF inject screens are 4/5.
4. Residual OOF↔Test structure issue to watch next: **`Family_E`↔`L7F` corr flip** (Δ=0.145), independent of the LM fix.

`NO_SUBMIT=True` — nothing uploaded to Kaggle in V16.7.
