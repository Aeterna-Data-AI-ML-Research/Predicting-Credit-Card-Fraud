# GM V17.9 — Slot5 R2 Submit + R1 F2 Clean → R12

Script: `scripts/gm_v17_9_submit_r2_and_clean_r12.py`  
JSON: `/tmp/s6e11_v16/out/v17_9.json`

---

## Step 1 — LIVE Slot 5: **NEW PUBLIC ATH `0.95394`**

| Field | Value |
|---|---|
| File | `submissions/sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv` |
| Construction | `qlock( L(Slot3) + 0.020·unitize(orth_t_R2) , sort(LIVE) )` |
| Ref | **58339747** |
| Status | Complete |
| **Public LB** | **`0.95394`** |
| vs Slot3 `0.95389` | **`+5e-5`** |
| vs Slot4 G1e `0.95387` | **`+7e-5`** |
| Verify | 299,984 rows, 0 NaNs |
| `1−ρ` (e-4) | Slot3 **0.763** / Slot1 **0.808** / LIVE **1.028** |
| v17.8 w=0.025 `1−ρ(Slot3)` | 1.192 e-4 |

OOF @ w=0.020 (twin peak): **5/5** on S3 / Slot3_twin / NNLS, twin mean **+4.99 e-5**, min **+1.80**, sf20, t=3.72; forensic F1/F2/F3 pass (F2=0.033).

**Dual-baseline gate predicted the live lift.** First residual raw-data add-on to clear public ATH since the Slot1–3 plateau.

---

## Step 2 — Clean R1 F2 leak

### Per-key leak diagnosis (α=10 orth)

Worst single-key leaks were **degenerate integer “cents/tenth”** encodings (`txn_count_*`, `card_age_*`, `customer_age_*`, `cvv_retry_*`) with F2≈**0.357** onto Family_E (not the q1000 keys). q1000 still removed as planned; those int-digit aliases are noise in the combiner.

### After dropping 4× q1000 keys → `R1_pure` (35 keys)

| Variant | F1 | F2 | F3 | ver | z_S3 | Slot3_twin | NNLS |
|---|---|---|---|---|---|---|---|
| **R1_pure α=10** | 0.999 ✓ | **0.0343** ✓ | −0.006 ✓ | **✓** | 5/5 @0.03 **+7.10** sf19 t=4.34 | 5/5 @0.03 **+10.78** sf18 t=4.66 | 5/5 +7.82 |
| R1_pure α=0.1 | same | 0.0343 ✓ | ✓ | ✓ | same | same | same |
| **R1_residBo9 α=10** | 1.000 ✓ | **0.0290** ✓ | −0.006 ✓ | **✓** | 5/5 +6.93 sf19 t=4.24 | 5/5 +10.30 sf18 t=4.61 | 5/5 +7.63 |
| R12_clean α=0.1/10 | 0.994 ✓ | **0.042** ✗ | −0.005 ✓ | ✗ | 5/5 +8.65 sf21 t=5.38 | 5/5 +13.27 sf20 t=5.28 | 5/5 +9.08 |
| R12_residR1+R2 α=0.1 | 0.995 ✓ | **0.0385** ✗ | ✓ | ✗ | 5/5 +8.52 | 5/5 +12.93 | 5/5 +8.94 |

**F2 cured for R1:** 0.052 → **0.034** (pure) / **0.029** (per-key Bo9 residualize).  
**R12 blend** still slightly over F2 (0.038–0.042) despite dual-elite strength — next slot should prefer **R1_pure / R1_residBo9 alone** or a softer blend weight, not equal `unitize(R1)+unitize(R2)`.

### Staged NOSUBMIT (next-slot ammo)

| File | Role |
|---|---|
| `sub_v17_9_R1_pure_a10_NOSUBMIT.csv` | forensic+dual verified R1_pure @ w=0.03 |
| `sub_v17_9_R12_clean_NOSUBMIT.csv` | dual-strong blend (F2 soft 0.042) — diagnostic / future α-tune |

---

## Guardrails

| Check | Status |
|---|---|
| Slot5 submit | **Done** — ATH **0.95394** |
| Daily remaining (post-submit msg) | 5 |
| Pods | **0** |
| R12 / R1_pure | NOSUBMIT only |

---

*End GM V17.9*
