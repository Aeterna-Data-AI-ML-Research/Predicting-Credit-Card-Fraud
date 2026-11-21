# GM V17.7 — Dual-Baseline Invariance Audit (Post Slot-4 Feedback)

**STRICT `NO_SUBMIT=True` · 0 GPU pods · 1 Slot left (Slot 5 held)**

Script: `scripts/gm_v17_7_dual_baseline_invariance_audit.py`  
JSON: `/tmp/s6e11_v16/out/v17_7.json`

**Live feedback:** Slot4 `+0.018·G1e` → Public **0.95387** (−2e-5 vs Slot3 **0.95389**), matching the pessimistic `z_Slot3_twin` prediction.

**New hard rule:** stage only if a **fixed sign** improves **both** `z_S3` and `z_Slot3_twin` (`pos≥4/5` and `meanΔ>0`), preferably also `z_NNLS_twin`. **No per-baseline sign flip.**

---

## Verdict

| Outcome | Detail |
|---|---|
| Dual winners (forensic ∧ S3∩twin) | **`+U5_gini10_llr` only** |
| Slot-5 ready? | **No** — dual-pass but **weak** (sf13/11, t=1.61/0.78); far below historical Slot bars |
| `−G1e` rescue? | **No** — hurts `z_S3` (0/5) and twin (1/5) |
| Staged CSVs | **none** |

---

## Baselines

| Baseline | AUC | Role |
|---|---|---|
| `z_S3` | 0.95242 | optimistic OOF proxy (overfit G1e) |
| `z_Slot3_twin` = Bo9@β̂_Slot3 (Ridge α=1, R²=0.981) | 0.95179 | **live-calibrated** (predicted Slot4 drop); β(G26)=−0.168, β(d1)=−0.057 |
| `z_NNLS_twin` (locked +d1/+FE, NNLS bank) | 0.95225 | non-negative / omitted-var fixed twin |

Grid: `w ∈ {0.005, 0.010, 0.015, 0.020}`; best-w per baseline under **fixed** `±u`.

---

## Step 1–2 — Dual-baseline table (best w per base)

Dual Pass = `pos≥4` ∧ `meanΔ>0` on **both** `z_S3` and `z_Slot3_twin`, **same sign**.

| Signal / sign | F1 | F2 | F3 | ver | z_S3 pos/mean/sf/t @w | z_Slot3_twin pos/mean/sf/t @w | Dual | All3 |
|---|---|---|---|---|---|---|---|---|
| **+G1e** (Slot4 dir) | 1.00 | **0.233** ✗ | +0.008 | ✗ | **5/5** +4.94 sf21 t=3.38 @0.020 | **2/5** +2.01 sf17 t=1.09 @0.015 | ✗ | ✗ |
| **−G1e** | same | same | same | ✗ | **0/5** −1.62 sf5 t=−3.66 @0.005 | **1/5** −1.87 sf6 t=−2.69 @0.005 | ✗ | ✗ |
| +G1f BPE−TAB | 0.98 | 0.057 ✗ | +0.231 ✗ | ✗ | 3/5 +1.11 @0.015 | **5/5** +16.1 @0.020 | ✗ | ✗ |
| −G1f BPE−TAB | | | | ✗ | 1/5 −0.75 | 0/5 −6.08 | ✗ | ✗ |
| +G1f QLM−TAB | 1.06 | 0.307 ✗ | +0.198 ✗ | ✗ | 3/5 +1.28 | 1/5 −3.94 | ✗ | ✗ |
| −G1f QLM−TAB | | | | ✗ | 2/5 −0.79 | 4/5 +7.33 | ✗ | ✗ |
| **+U5_gini10_llr** | 0.997 ✓ | 0.025 ✓ | −0.046 ✓ | **✓** | **5/5** +1.00 sf13 t=1.61 @0.015 | **4/5** +0.58 sf11 t=0.78 @0.005 | **✓** | **✓** |
| −U5 | | | | ✓ | 0/5 −0.60 | 0/5 −1.12 | ✗ | ✗ |
| +q80_q20 | 0.993 ✓ | **0.074** ✗ | −0.026 ✓ | ✗ | 5/5 +0.90 sf16 t=1.67 @0.010 | 4/5 +1.00 sf13 t=0.83 @0.010 | **✓*** | ✓* |
| −q80 | | | | ✗ | 0/5 | 0/5 | ✗ | ✗ |
| +d1 unitize | 0.98 | **0.985** ✗ | ~0 | ✗ | 3/5 +0.15 | 5/5 +12.0 | ✗ | ✗ |
| −d1 unitize | | | | ✗ | 2/5 −0.16 | 0/5 −3.76 | ✗ | ✗ |

\*q80 dual-passes strength but **fails F2** → not stageable under forensic rule.

### −G1e detail (Slot4 opposite)

| Base | best | folds [F0..F4] e-5 |
|---|---|---|
| z_S3 | 0/5 @0.005 mean −1.62 | [−3.19, −1.15, −0.67, −1.95, −1.16] |
| Slot3_twin | 1/5 @0.005 mean −1.87 | [−5.61, +0.06, −0.07, −3.43, −0.29] |
| NNLS_twin | 3/5 @0.005 mean −0.21 | [−1.63, +0.42, +0.57, −0.57, +0.18] |

**Flipping G1e does not monetize the Slot4 miss.** The twin’s +G1e damage on F1/F2/F4 does not become a clean −G1e win — F0/F3 flip negative and net mean stays ≤0.

*(Note: this rebuild’s G1e F2 leak=0.23 is worse than V17.4’s 0.023 under a slightly different `z_S3`/Bo9 stamp; dual failure vs twin is unchanged and matches live.)*

### G1f (5-fold QLM vs TAB)

Neither sign is forensic-clean. Anti-correlated across bases: **+BPE−TAB** loves the twin (5/5 +16e-5) but only 3/5 on `z_S3`; **−QLM−TAB** helps twin (4/5) but not `z_S3`. No fixed-sign dual pass.

---

## d1 micro-adjustment

### Absolute stacks `z_s1 + w·u(d1)` (AUC)

| Stack | AUC | vs z_S3 folds | vs Slot3_twin folds |
|---|---|---|---|
| Slot1 w=0.030 | 0.952436 | 4/5 +1.07 | 5/5 (twin is weak — large Δ) |
| P1 w=0.033 | 0.952437 | 4/5 +1.09 | 5/5 |
| Slot3 w=0.0355 | 0.952437 | 4/5 +1.11 | 5/5 |

Stack AUCs are essentially tied; twin Δ is inflated by twin’s low absolute AUC, not a free LB tick.

### Δw·u(d1) add-on (fixed sign) on baselines

| Move | Dual S3∩twin | z_S3 | twin | NNLS |
|---|---|---|---|---|
| −0.0055 (toward Slot1) | ✗ | 1/5 −0.46 | 0/5 −4.03 | 4/5 +0.20 |
| −0.0025 (toward P1) | ✗ | 1/5 −0.21 | 0/5 −1.79 | 4/5 +0.11 |
| **+0.0030** | **✓** (not forensic; d1∈Bo9) | 5/5 +0.22 | 5/5 +2.05 | **1/5 −0.12** |

`+0.003` d1 dual-passes the weak twin (which shorts d1) but **fails NNLS twin** and is not a true residual (d1 already in the 9-basis).

---

## Dual-pass summary

| Candidate | Forensic | Dual S3∩twin | All3 | Slot-5? |
|---|---|---|---|---|
| **+U5_gini10_llr** | ✓ | ✓ | ✓ | **No** (t≪2.2, sf≪18) |
| +q80_q20 | ✗ F2 | ✓ | ✓ | No |
| +0.003 d1 | ✗ (in-basis) | ✓ | ✗ NNLS | No |
| ±G1e / G1f / −U5 | — | ✗ | ✗ | No |

---

## Guardrails / Slot 5

| Check | Status |
|---|---|
| Kaggle submit | **none** |
| Pods | **0** |
| Stage NOSUBMIT | **none** — no dual+forensic+strong candidate |
| Slot 5 | **still reserved** |

---

*End GM V17.7*
