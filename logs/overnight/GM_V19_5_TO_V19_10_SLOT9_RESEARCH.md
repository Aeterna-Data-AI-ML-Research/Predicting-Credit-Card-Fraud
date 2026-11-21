# GM V19.5–V19.10 — Slot9 research vs ATH Slot8 **0.95395** → **HOLD (2 slots left)**

**0 GPU pods · no Slot9 LIVE submit**

---

## Live status (2026-11-14)

| Slot | Score | Notes |
|---|---|---|
| **Slot8 ATH** | **0.95395** | V19.4 Cat+U5 (`wo=0.016,wc=0.012,wu=0.012`) |
| Slot5 prior | 0.95394 | R2@0.020 |
| Remaining today | **2** | |

---

## What was researched (no gate clear)

| Wave | Lever | Result |
|---|---|---|
| V19.5 | Slot8 weight neighborhood, ms45 swap, +lgb39, +U6, mixes | 0 winners (micro 1−ρ / twin collapse) |
| V19.6 | Newton residual LGB+Cat on Slot8 tabular | Empty / negative H2H (signal already extracted) |
| V19.7 | Alt V19.4 recipes, R4, externals, denser Cat/U5 | 0; R4 F2≈1 (already in stack) |
| V19.8/b | New L7 families (ent/std/mad/top2/MS) | U_ent F2-clean but weak twin; disp-fix didn't unlock |
| V19.9 | Peer M*/P4 orth Bo11 | **P4 huge OOF** (+8…+15e-5 5/5×3) but **c7=+0.51** = Slot7 fail ray |
| V19.9b | P4 weight for 1−ρ band | Clears ρ at w≥0.013 but anti-Slot7 gate fails |
| V19.10 | P4 depoison vs M5/FE/S6/S7 | Removes fail-ray **and** the lift (F2→1.0, H2H dies) |

---

## Why Slot9 is held

1. **Remixes of Slot8 recipe** are saturated — only micro Spearman moves vs ATH.
2. **Newton-on-S8** with the same digit/TE X as R2 has nothing left.
3. **P4 peer** is the only large OOF candidate, but it is **geometrically aligned with Slot7’s losing delta** (`corr≈+0.51` with S7−S5). Slot7 already failed live (−3e-5). Depoisoning kills the OOF edge.
4. Gates that produced Slot8 (strict 5/5×3 + 1−ρ∈[0.35,1.20] + anti-fail-ray) are not cleared by any safe family.

---

## Ammo for later (NOSUBMIT)

- P4@0.015 on Slot8: elite OOF, **do not auto-submit** until fail-ray geometry is solved without destroying signal.
- Path: new non-peer residual family (different X than R2/Cat), or peer rebuild that is FE-free **and** S7-orthogonal by construction.

---

## Decision

**HOLD ATH `0.95395`. Keep 2 daily slots.** Next live only on a candidate that clears Slot8 H2H **and** `|corr(δ, Slot7−Slot5)| < 0.20`.
