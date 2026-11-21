# GM Leap Research — Path to 0.95369+

**NO_SUBMIT=True**. No submission CSVs. ATH `0.95354` → target `0.95369`.
Elapsed `178.5s`, peak RSS ≈ `678` MB.

Linear LB law needs **t≈3.157** for 0.95369 (S7 is t≈1.037, pred `0.953542`).

## Phase A — Disk / L4.6 verification (vs z_base & RayA)

| name | 1D Δ5F | pos | REG Δ5F | pos | vsRayA Δ5F | pos | F0 REG vsRayA | pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v10_lgb_orth | +66.05 | 5/5 | +23.93 | 5/5 | +23.59 | 5/5 | +24.65 | 5/5 |
| tree_logit_v72 | +0.22 | 3/5 | +0.89 | 3/5 | +1.03 | 4/5 | +2.01 | 3/5 |
| P4_L4.5 | +0.35 | 2/5 | +0.03 | 2/5 | +0.10 | 3/5 | +0.39 | 2/5 |
| P4_L4.3_ext | -0.26 | 3/5 | -0.46 | 2/5 | -0.38 | 2/5 | -2.34 | 1/5 |
| L4.6C | -0.47 | 2/5 | -0.64 | 2/5 | -0.82 | 2/5 | +6.12 | 5/5 |
| L4.6A | -1.44 | 1/5 | -1.68 | 1/5 | -1.80 | 1/5 | -0.82 | 0/5 |
| L4.6B | -1.13 | 1/5 | -2.33 | 1/5 | -2.75 | 2/5 | -3.99 | 1/5 |
| v10_ridge_orth | +5.01 | 5/5 | -4.88 | 0/5 | -5.42 | 0/5 | -6.81 | 0/5 |
| v10_combo3_orth | +19.78 | 5/5 | -6.43 | 0/5 | -7.06 | 0/5 | -6.54 | 0/5 |

## Phase B — LOFO stack

Winners: `['v10_lgb_orth', 'tree_logit_v72', 'L4.6C', 'L4.6A', 'L4.6B']`
- vs RayA: **+60.39e-5** pos `5/5` d5=`[37.04, 26.37, 72.96, 96.2, 69.37]`
- vs z_base: **+60.96e-5** pos `5/5`
- Fold0 SubCV vs RayA: **+18.00e-5** pos `5/5` @w=0.02

## Phase C — Leap geometry (RayA + v10_lgb direction)

| 1−ρ(S7) | t | pred_lin | OOF5F vsS7 | pos | F0 vsS7 | F0 vsRayA | ≥0.95369? |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00e-05 | 1.127 | 0.953548 | +6.79 | 5/5 | +6.35 | +0.48 | no |
| 3.50e-05 | 1.267 | 0.953558 | +22.07 | 5/5 | +20.64 | +14.78 | no |
| 7.50e-05 | 1.445 | 0.953570 | +39.58 | 5/5 | +36.43 | +30.57 | no |
| 1.20e-04 | 1.615 | 0.953582 | +48.73 | 5/5 | +44.33 | +38.47 | no |
| 2.00e-04 | 1.872 | 0.953600 | +31.92 | 4/5 | +26.51 | +20.65 | no |
| 3.00e-04 | 2.145 | 0.953619 | -65.20 | 1/5 | -71.12 | -76.98 | no |
| 4.50e-04 | 2.494 | 0.953644 | -434.64 | 0/5 | -431.57 | -437.43 | no |
| 6.00e-04 | 2.796 | 0.953665 | -1209.65 | 0/5 | -1165.36 | -1171.22 | no |

**OOF-safe frontier:** step=`2.00e-04` pred_lin=`0.953600` (F0vsRayA=`+20.65`, pos=`4/5`).
**Fold-0 peak along leap:** step=`1.20e-04` F0vsS7=`+44.33` pred_lin=`0.953582`.

## Pure v10_lgb_orth on RayA (no macro ray push)
- 5F: **+64.27e-5** `5/5` @w=0.08 d5=`[38.32, 34.31, 80.99, 96.78, 70.98]`
- Fold0 SubCV: **+19.60e-5** `5/5` @w=0.02

## Verdict

- **Primary unlock:** `v10_lgb_d3_bag5_orth_LM` lifts RayA by **+64.27e-5 (5/5)** — strongest post-S7 signal found.
- OOF-safe frontier pred_lin `0.953600` is still **below** `0.95369`; reaching 0.95369 on the linear law requires larger Spearman steps where Fold-0/5F currently degrade — need more orthogonal lift (v10-scale) before macro geometry.

**NO_SUBMIT=True** — research only.

---

# Phase D — Multi-orth amplify (partial; D5 crashed then resumed in D5+E)

## D1 Screen highlights
| name | RayA 1D | pos | Ray REG | F0 |
|---|---:|---:|---:|---:|
| v10_lgb | +64.27 | 5/5 | +23.59 | +19.60 |
| soft_diff | +17.20 | 5/5 | +9.81 | +3.75 |
| v10_combo3 | +20.01 | 5/5 | -7.06 | +8.74 |
| strike10 | +15.96 | 5/5 | -2.21 | +2.92 |
| gb_soft | +3.68 | 5/5 | +0.30 | +0.15 |

## D2 Greedy blend
Selected: `v10_lgb` + `soft_diff` → **+73.87e-5 @5/5** vs RayA, F0 SubCV **+37.68e-5 @5/5**

## D3 Fusion
vs RayA **+71.15e-5 @5/5**, F0 **+35.15e-5 @5/5**

## D4 Geometry (before D5 crash)
- `ray_v10` OOF-safe-ish at step `2.00e-4`: pred_lin=`0.953628` OOF=+31.9@4/5 F0ray=+20.6
- At target t≈3.16 (`step≈5.5e-4`) OOF collapses (~−900e-5)


---

# Phase D5+E — Dense 1D (partial; E2 killed as too slow)

## D5 F0-gated
Frontier pred_lin=`0.953585` @w=0.05 (OOF +59); larger w loses F0 pos.

## E1 dense 1D OOF-safe maxima
| dir | pred_lin | t | step | OOF | F0ray |
|---|---:|---:|---:|---:|---:|
| ray_v10 | **0.953637** | 2.397 | 2.40e-4 | +17.9@4/5 | +5.2 |
| work | 0.953626 | 2.245 | 2.10e-4 | +40.5@4/5 | +1.1 |
| ray_sel | 0.953624 | 2.213 | 2.00e-4 | +46.6@4/5 | +5.0 |
| ray35 | 0.953607 | 1.977 | 1.70e-4 | +55.0@4/5 | +1.8 |
| v10 | 0.953594 | 1.779 | 1.70e-4 | +49.1@4/5 | +1.7 |

Best OOF-safe frontier moved from 0.953600 → **0.953637** (still −5.3e-5 shy of 0.95369).

---

# Phase E-fast — 2D/3D calibrated geometry
**Best OOF-safe pred_lin:** `0.953658` (t=`2.700`, 2d a=0.7 b=0.25 c=0.0, step=3.00e-04, OOF=+4.6@4/5, F0ray=+26.6)

Gap to 0.95369: **3.2e-5**

**No OOF-safe geometry clears 0.95369.** Best safe frontier still below target; paths at t≈3.16 remain OOF-negative.

Soft-safe (≥3/5, F0ray≥−20) near-target best: pred=`0.953651` OOF=`+19.4`@4/5

**NO_SUBMIT=True**

---

# Phase F — Close gap to 0.95369
**Best OOF-safe:** pred=`0.953665` t=`2.797` a=0.65 b=0.175 c=0.1 step=3.30e-04 OOF=+0.1@4/5 F0ray=+16.1 gap=`2.52e-5`

**Best soft-safe (pos≥3, F0ray≥−5):** pred=`0.953667` t=`2.824` a=0.95 b=0.225 c=0.1 OOF=-4.9@3/5 F0ray=+21.5

**No OOF-safe hit of 0.95369 in Phase F fine grid.**

**NO_SUBMIT=True**

---

## Executive verdict (NO_SUBMIT)

| Milestone | pred_lin | note |
|---|---:|---|
| ATH | 0.95354 | public LB |
| Target | 0.95369 | +15e-5 vs ATH |
| Phase C OOF-safe | 0.953600 | ray+v10 |
| Phase E OOF-safe | 0.953658 | 2d a=0.7 b=0.25 |
| **Phase F OOF-safe** | **0.953665** | a=0.65 b=0.175 c=0.1 step=3.3e-4; OOF≈+0.1@4/5 |
| Phase F soft (pos≥3) | 0.953667 | still short; OOF≈−4.9 |

**Primary unlock:** `v10_lgb` + `soft_diff` blend vs RayA **+73.9e-5 @5/5** (F0 +37.7).

**Blocker:** linear-law t≈3.16 for 0.95369 needs Spearman steps where 5F/F0 collapse. Best OOF-safe is **2.5e-5** short; even soft gate tops at **0.953667**. Need a new orth signal beyond v10+soft (or a better test↔OOF step map) before a NO_SUBMIT-validated 0.95369+ path exists.

**NO_SUBMIT throughout — no CSVs staged.**
