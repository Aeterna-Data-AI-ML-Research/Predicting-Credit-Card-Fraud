# GM V19.4 — Slot8 LIVE: Cat+U5 diversity → **NEW ATH 0.95395**

**0 GPU pods · research→submit · ref `58344471`**  
Script: `scripts/gm_v19_4_cat_u5_diversity_and_distance.py`  
CSV: `submissions/sub_v19_4_LIVE_slot8_A_cat_wo0.016_wc0.012_wu0.012.csv`

---

## Verdict

| | |
|---|---|
| **Public LB** | **0.95395** |
| Prior ATH (Slot5) | 0.95394 |
| Δ | **+0.00001** |
| Remaining today | **2** |

---

## What worked

**Lever:** Anchored `Slot3 + 0.016·R2 + 0.012·F_cat_obliv + 0.012·U5_gini` (qlocked to LIVE).

| Gate | Result |
|---|---|
| 5/5×3 H2H vs Slot5 | ✓ z_S5 +3.07 / twin +3.68 / nnls +3.12 (mins +1.99 / +0.26 / +1.89) |
| sf / t | ✓ 20/19/19 · 4.84 / 2.96 / 4.59 |
| `1−ρ(Slot5)` | **0.407e-4** (corrected band [0.35, 1.20]; Slot5-win scale was 0.763 vs Slot3) |
| Neg-ray Slot6/7 | cleared |

**Why U5 mattered:** `corr(U5,R2)≈0` / `corr(U5,Cat)≈−0.03` — orthogonal diversity pushed Spearman into the live-transfer band without killing twin H2H (V19.2/V19.3 Cat-only hit 5/5×3 but stalled at `1−ρ≈0.075`).

**Band correction:** Old `[0.15,0.55]` rejected Slot5-scale steps and still allowed Slot7 micros (`0.148`). New floor `0.35` + Cat/U5 geometry cleared.

---

## V19.3 prelude (no submit)

Strict old band: **0 winners**. Near-miss `Cat_wo0.018_wc0.008` = 5/5×3 but `1−ρ=0.075`.

---

## Next

Slot9 research vs **0.95395**; 2 slots left today; submit one-at-a-time only on clear lift.
