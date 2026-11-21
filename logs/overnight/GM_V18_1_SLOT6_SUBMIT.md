# GM V18.1 — Slot 6 Submit: Slot5 + `orth(R1_pure | Bo10_S5)` @ w=0.022

**0 GPU pods · submitted**  
Script: `scripts/gm_v18_1_submit_slot6_r1_pure_bo10.py`  
File: `submissions/sub_v18_1_LIVE_slot6_Slot5_plus_R1_pure_Bo10_w0p022.csv`  
Ref: **58340240**

---

## Step 1 — w-grid on Slot 5 baselines

| w | 1−ρ(Slot5)×1e4 | z_S5 | Slot5_twin | NNLS_S5 |
|---|---|---|---|---|
| 0.020 | 0.795 | 5/5 +4.93 min+3.85 sf19 t4.53 | 5/5 +8.20 min+5.05 sf19 t4.13 | 5/5 +5.31 min+3.35 sf20 t4.40 |
| **0.022** | **0.962** | **5/5 +5.28 min+4.12 sf19 t4.53** | **5/5 +8.61 min+5.33 sf19 t4.13** | **5/5 +5.78 min+3.62 sf20 t4.40** |
| 0.025 | 1.242 | 5/5 +5.75 min+4.21 sf19 t4.53 | 5/5 +9.04 min+5.48 sf19 t4.13 | 5/5 +6.34 min+3.88 sf20 t4.40 |
| 0.030 | 1.788 | 5/5 +6.41 min+4.54 sf19 t4.53 | 5/5 +9.46 min+5.29 sf19 t4.13 | 5/5 +7.21 min+4.27 sf20 t4.40 |

Submit choice **w=0.022** (~0.96e-4 step, Slot5-scale). Bracketing ammo remains: `sub_v18_0_R1_pure_Bo10_w0p03_NOSUBMIT.csv`.

CSV: **299,984** rows · **0** NaNs · qlock to LIVE 0.95388.

---

## Step 2 — Kaggle

| Item | Value |
|---|---|
| Submit | Success · ref **58340240** |
| Remaining today (post) | 4 |
| Public LB | **0.95392** (**−2e-5** vs Slot5 0.95394) |

**Result:** Slot6 did **not** climb. Strong OOF dual/triple gates on `orth(R1_pure|Bo10)` failed to translate — same pattern class as Slot4 G1e (−2e-5). Hold Slot5 ATH **0.95394**; do not chase w=0.03 bracketing on this axis without a new residual that is LB-aligned.

---

*End GM V18.1*
