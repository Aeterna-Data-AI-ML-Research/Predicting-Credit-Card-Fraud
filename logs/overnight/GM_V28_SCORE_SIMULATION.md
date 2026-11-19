# V28 Score Simulation (nested gates, no submit)

ATH lock **0.95511**. Simulation = multi-base nested 5-fold ΔAUC (e-4) + board surrogates.

## L1 zone soft-floor — PASS/FAIL

| zone | s | macroΔ e-4 | bases≥4/5 | ATH_surr | SUB_surr | E[Δpub] | verdict |
|------|--:|----------:|----------:|----------|----------|---------|---------|
| Z1 | 0.15 | +0.287 | 15/15 | 5/5 +0.221 | 5/5 +0.221 | +0.57–+1.44e-5 | **PASS** |
| Z1 | 0.30 | +0.380 | 15/15 | 5/5 +0.285 | 5/5 +0.285 | +0.76–+1.90e-5 | **PASS** |
| Z12 | 0.15 | +0.344 | 15/15 | 5/5 +0.273 | 5/5 +0.273 | +0.69–+1.72e-5 | **PASS** |
| Z12 | 0.25 | +0.430 | 15/15 | 5/5 +0.337 | 5/5 +0.337 | +0.86–+2.15e-5 | **PASS** |
| Z1234 | 0.10 | +0.355 | 15/15 | 5/5 +0.293 | 5/5 +0.293 | +0.71–+1.78e-5 | **PASS** |
| Z1234 | 0.20 | +0.510 | 15/15 | 5/5 +0.413 | 5/5 +0.413 | +1.02–+2.55e-5 | **PASS** |
| Z1234 | 0.35 | +0.592 | 15/15 | 5/5 +0.472 | 5/5 +0.472 | +1.18–+2.96e-5 | **PASS** |
| Z1234 | 0.50 | +0.613 | 15/15 | 5/5 +0.485 | 5/5 +0.485 | +1.23–+3.07e-5 | **PASS** |
| Z1234 | 0.75 | +0.619 | 15/15 | 5/5 +0.489 | 5/5 +0.489 | +1.24–+3.09e-5 | **PASS** |

## Simulated public score if PASS transfers

- Best: **Z1234@s0.75** → simulated public **0.95512 – 0.95514** (vs ATH 0.95511).
- Transfer assumption: 20–50% of nested macro ΔAUC lands on public (conservative vs board micro-steps).
- Rank moves on ATH test: **6915** rows; ρ(ATH)=0.0007e-4 (micro geometry, same family as prior +1e-5 BOLDs).

## Replay on scored parents

- floor(ATH511@0.95511 | Z1234@s0.20): 5/5 m=+0.413 t=9.53
- floor(ATH511@0.95511 | Z1234@s0.35): 5/5 m=+0.472 t=9.12
- floor(ATH511@0.95511 | Z12@s0.25): 5/5 m=+0.337 t=9.55
- floor(gpw_sub@0.95508 | Z1234@s0.20): 5/5 m=+0.413 t=9.54
- floor(gpw_sub@0.95508 | Z1234@s0.35): 5/5 m=+0.472 t=9.12
- floor(gpw_sub@0.95508 | Z12@s0.25): 5/5 m=+0.337 t=9.55
- floor(stepI@0.95504 | Z1234@s0.20): 5/5 m=+0.413 t=9.54
- floor(stepI@0.95504 | Z1234@s0.35): 5/5 m=+0.472 t=9.12
- floor(stepI@0.95504 | Z12@s0.25): 5/5 m=+0.337 t=9.55
- floor(stepH@0.95504 | Z1234@s0.20): 5/5 m=+0.413 t=9.53
- floor(stepH@0.95504 | Z1234@s0.35): 5/5 m=+0.472 t=9.12
- floor(stepH@0.95504 | Z12@s0.25): 5/5 m=+0.337 t=9.55
- floor(stepG@0.95498 | Z1234@s0.20): 5/5 m=+0.416 t=9.56
- floor(stepG@0.95498 | Z1234@s0.35): 5/5 m=+0.476 t=9.14
- floor(stepG@0.95498 | Z12@s0.25): 5/5 m=+0.340 t=9.59

## Notes

- This is **not** a Kaggle public score; it is a DS nested/surrogate simulation.
- L2 HGB already failed gb gate in V28 screen — not re-simulated as lift.
- L3 only counts if ATH_surr **and** SUB_surr clear (board geometry).

