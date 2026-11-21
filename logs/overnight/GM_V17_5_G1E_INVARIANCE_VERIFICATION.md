# GM V17.5 — G1e Grandmaster Invariance Verification

**STRICT `NO_SUBMIT=True` · 0 GPU pods · local CPU**

Script: `scripts/gm_v17_5_verify_g1e_invariance_and_stage.py`  
JSON: `/tmp/s6e11_v16/out/v17_5.json`

---

## Verdict — **DO NOT authorize Slot 4 on G1e yet**

`G1e` remains **strong and forensic-clean on `z_S3`**, but **fails the basis-weight invariance test**:

| OOF baseline | @ w=0.035 | Verdict |
|---|---|---|
| `z_S3` (0.95242) | **5/5**, mean **+6.91 e-5**, min +0.19, sf21, t=3.38 | Pass |
| `z_Slot3_twin = Bo9 @ β̂_Slot3` (R²=0.981) | **2/5**, mean **−4.05 e-5**, min **−17.0** | **FAIL** |
| `z_Logit_oof` (Bo9 L2/unpenalized) | **2/5**, mean **−4.75 e-5**, min **−24.1** | **FAIL** |

`G1e` is **not stack-invariant**. It exploits geometry specific to the `z_S3 = z_ATH + 0.01045·FE + 0.0055·d1` recipe. Slot3’s empirical Test weights (**G26 β̂ = −0.168**) make the same oriented residual **hurt** folds 1/2/4.

Calibrated NOSUBMIT CSVs were still written for step-size logging only — **not submit-ready**.

---

## Check 1 — Physical sign & pair drivers

| Item | Result |
|---|---|
| `screen()` `flipped` | **`False`** |
| High-risk tail (`z_S3>0`) | Winning direction prefers **`P_BPE (MS+G26)`** over `P_QLM` |
| `corr(u, d_all)` | +0.041 (weak after orth) |
| `corr(u, raw d·U(z_S3))` | **+0.897** |

**Interpretation:** Without flip, adding `+w·G1e` raises the score where oriented `d_all·U(z_S3)` is high → on the fraud tail, that is when **BPE > QLM**. Physically: up-weight exact-digit GPT-2 relative to quantile-LM in high-risk rows.

### Pair decomposition @ w=0.035 on `z_S3` (each `orth(unitize(L_fold(za−zb))·U(z_S3))`, aligned to G1e)

| Pair | pos/5 | meanΔ | min | folds [F0..F4] | sf25 | t | corr vs G1e |
|---|---|---|---|---|---|---|---|
| **G26−LM** | **5/5** | **+7.40** | **+2.51** | [16.04, 9.03, 2.51, 6.59, 2.83] | **23** | **3.79** | +0.67 |
| G26−L7F | 5/5 | +2.69 | +0.16 | [10.26, 0.24, 0.64, 2.17, 0.16] | 16 | 2.20 | +0.74 |
| MS−LM | 4/5 | +5.30 | −3.86 | [14.92, 3.87, **−3.86**, 8.16, 3.42] | 20 | 3.39 | +0.79 |
| MS−R3 | 3/5 | +3.40 | −1.92 | [11.99, −1.69, −1.92, 6.30, 2.31] | 17 | 2.47 | +0.79 |
| MS−L7F | 3/5 | +1.23 | −4.37 | [9.47, −4.07, −4.37, 3.53, 1.58] | 15 | 1.54 | +0.84 |

**Fold 0 (+18e-5) vs Fold 2 (+0.2e-5):** not a `σ(d_all)` effect (global unitize → per-fold std≈1 for all folds). The lift is carried by **`G26−LM`** (only pair that is 5/5 with large min); Fold 2 is the soft fold for MS−* pairs (negative) while G26−LM still clears +2.5e-5. Fold 0 is where every pair spikes — same fold-0 hypersensitivity seen in V17.0, not a separate `d` variance regime.

---

## Check 2 — Basis-weight invariance (the kill shot)

Slot3 Test Ridge(α=1) on `Bt9` → `unitize(L(Slot3))`:

| | MS | G26 | LM | L7F | GPD | GB | MLP | d1 | Family_E |
|---|---|---|---|---|---|---|---|---|---|
| **β̂** | **+0.592** | **−0.168** | +0.066 | +0.045 | +0.091 | +0.268 | +0.104 | −0.057 | +0.014 |

R² ≈ **0.9808**. OOF twin: `z_Slot3_twin = Bo9 @ β̂ + intercept` (AUC 0.95179).

### Grid `w ∈ {0.010…0.035}` — oriented `orth(G1e)`

**Baseline `z_S3`:** all w → **5/5**, min>0, sf21, t=3.38. @0.035: folds `[18.13, 3.78, 0.19, 8.52, 3.90]`.

**Baseline `z_Slot3_twin`:** all w → **2/5** (only F0 & F3 positive). @0.035: `[+21.0, −16.0, −17.0, +7.8, −16.0]`.

**Baseline `z_Logit_oof`:** ≤3/5; @0.035: `[+20.0, −8.4, −14.5, +3.3, −24.1]`.

**Why:** Slot3 already **shorts G26** (β=−0.17). G1e’s winning sign **loads BPE/G26 in the high-risk tail**. On `z_S3` (different FE/d1 geometry, not the Slot3 β), that residual helps; on the Slot3-weight twin it **double-counts / fights the short**, destroying F1/F2/F4.

→ **Invariance pass @ w=0.035: `False`.**

---

## Check 3 — Spearman step sizes & staged rungs (`NO_SUBMIT`)

### `sub_v17_4_d_all_G1e_risk_gated_d__Bo9_NOSUBMIT.csv` (w=0.035)

| Contrast | `1 − ρ` (×1e-4) |
|---|---|
| vs Slot3 (0.95389) | **0.657** |
| vs Slot1 (0.95389) | **0.711** |
| vs LIVE (0.95388) | **0.867** |

Rebuild @0.035 matches v17.4 csv (spearman **0.999985**); rebuild `1−ρ(Slot3)` ≈ 0.732 e-4 (tiny qlock / u_t rebuild delta).

### Newly staged (logging only)

| File | w | OOF on `z_S3` | `1−ρ` vs base (e-4) |
|---|---|---|---|
| `sub_v17_5_Slot3_plus_G1e_w0p018_NOSUBMIT.csv` | 0.018 | 5/5, mean +4.56, sf21, t=3.38 | **0.194** |
| `sub_v17_5_Slot3_plus_G1e_w0p025_NOSUBMIT.csv` | 0.025 | 5/5, mean +5.73, sf21, t=3.38 | **0.373** |
| `sub_v17_5_Slot1_plus_G1e_w0p018_NOSUBMIT.csv` | 0.018 | (same OOF u) | 0.194 vs Slot1 |
| `sub_v17_5_Slot1_plus_G1e_w0p025_NOSUBMIT.csv` | 0.025 | (same OOF u) | 0.374 vs Slot1 |

**OOF on `z_Slot3_twin` at these w remains 2/5** (same failure mode as Check 2).

---

## Guardrails

| Check | Status |
|---|---|
| Kaggle submit | **None** |
| GPU pods | **0** |
| Slot 4 authorization | **Blocked** until a G1e-like residual is 5/5 on `z_Slot3_twin` (or an OOF proxy built with Slot3’s β̂), not only on `z_S3` |

---

## Next (when continuing)

1. Re-screen G1e after orth against **`[Bo9, z_Slot3_twin]`** or rebuild `d` gates on the Slot3-twin residual.  
2. Or isolate **`G26−LM` risk-gated** alone (Check 1 winner) under Slot3-twin invariance.  
3. Do **not** burn a submit on v17.4/v17.5 G1e CSVs despite pretty `z_S3` bars.

---

*End GM V17.5*
