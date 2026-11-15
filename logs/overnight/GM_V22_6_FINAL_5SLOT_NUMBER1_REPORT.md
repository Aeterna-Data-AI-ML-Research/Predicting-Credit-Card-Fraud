# GM V22.6 Final 5-Slot #1 Ladder Report

**elapsed** ~4+4 min · local Mac CPU · ATH held `0.95426` · **0 slots remaining today**

## Live Slot Leaderboard

| Slot | Tag | Public LB | ρ(R2) e-4 | tw_share | vs ATH |
|---:|---|---:|---:|---:|---|
| 1 | Pure `U_orth_super` @0.32e-4 (pre-fix) | **0.95426** | 0.320 | 0.141 | tie |
| 2 | 2B tri-centroid (pre-fix) | 0.95421 | 0.618 | 0.262 | −5e-5 |
| 3 | `U_3zone_moe` (pre-fix) | 0.95425 | 0.819 | 0.162 | −1e-5 |
| 4 | **FIXED** true-twilight `gLB*` @0.40e-4 | 0.95416 | 0.707 | **0.785** | −10e-5 |
| 5 | **FIXED** geodesic twilight @0.38e-4 | 0.95412 | 2.368 | **0.780** | −14e-5 |

Best live among V22.6 slots: **0.95426** (SLOT1 only). Personal ATH unchanged (`R2`/`K2`/`SLOT1`).

## Why Slots 1–3 Tied / Barely Moved (the bug)

1. **Twilight gate destroyed by post-gate `B_gs` projection (Fix #3 violation):**
   - Code did `project_out(G_twi(q) ⊙ U, B_gs)`.
   - `B_gs` is highly collinear with score rank → projection wiped quantile localization.
   - Measured `tw_share` ≈ **0.14–0.26** (need ≥0.55–0.80).
2. **Uniform orth residuals at `0.22–0.32e-4` mostly cancel on the ~60k public slice** → 5th-decimal rounding blindness (Fix #7) → SLOT1 identical `0.95426`.
3. **V7 excluded** by strict nnls `2/5` gate (softened later; V7 now passes).

## Fix Applied Mid-Ladder (Slots 4–5)

- `inject_twilight_rho`: **orth first, gate at inject time, never GS-project after gating**.
- Probe `tw_share` rose to **0.777–0.785** (Fix #3 satisfied).
- Soft nnls so `V7_B_xgb_llr` re-enters `U_orth_super`.
- Exact-fname poll (`sub_v22_6_SLOT{k}.csv`) — Fix #2 confirmed working.

## Critical Live Lesson (for next daily reset)

True twilight concentration **worked mechanically** but **`+U_orth_super` in the contested band is anti-aligned with Public LB**:

| tw_share | direction | LB |
|---:|---|---:|
| 0.14 | `+U_orth_super` uniform | 0.95426 (noop) |
| 0.78 | `+U_orth_super` twilight | **0.95416 / 0.95412** (hurt) |

Next ladder must **flip the twilight orth sign** (`−U_orth_super` in `q∈[0.78,0.992]`) and/or prefer **pure Ray-1 crest** steps on `[1.45e-4, 1.85e-4]` from `sriram` (between R2 `1.25e-4` and K2 `2.05e-4`) without orth residual.

## Artifacts

- `scripts/gm_v22_6_unattended_5slot_number1_ladder.py` (twilight inject fix locked)
- `submissions/sub_v22_6_SLOT{1..5}.csv`
- `submissions/sub_v22_6_FINAL_PRIVATE_LB_LOCK_NOSUBMIT.csv`
- Resume log: `logs/overnight/GM_V22_6_SLOTS45_TWILIGHT_FIX.log`
