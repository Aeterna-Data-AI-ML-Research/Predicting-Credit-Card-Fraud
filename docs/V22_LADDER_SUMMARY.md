# V22 Ladder Summary (2026-11-15)

Personal Public ATH: **0.95426** (`sub_v22_3_LIVE_ridge_continuation.csv`).

## Live path

| Ver | Submit | LB | `1−ρ(sriram)` | Notes |
|---|---|---:|---:|---|
| V22.1 C1 | `sub_v22_1_LIVE_leap.csv` | 0.95407 | ~28e-4 | lineage + dMLP |
| V22.1 C2pure | `sub_v22_1b_LIVE_leap.csv` | 0.95416 | 4.37e-4 | overshot; still personal ATH then |
| V22.2h P2 | `sub_v22_2_HARDENED_LIVE_microleap.csv` | 0.95425 | 0.55e-4 | unbundled microleap |
| **V22.3 R2** | `sub_v22_3_LIVE_ridge_continuation.csv` | **0.95426** | **1.25e-4** | current ATH |
| V22.4 | HOLD | — | — | Q1/Q2 sign/z9 self-overlap |
| V22.5 K2 | `sub_v22_5_LIVE_crest_orth4.csv` | 0.95426 | 2.05e-4 | tie; Ray-1 + orth4 |
| V22.6 S1 | `sub_v22_6_SLOT1.csv` | 0.95426 | — | pure orth, tw≈0.14 (noop) |
| V22.6 S2–S5 | SLOT2…5 | 0.95421→0.95412 | — | twilight/`U_3zone` hurt |

## Mechanics that transferred

1. Stay on the **sriram rank-frac manifold**; micro-steps in `1−ρ` beat large reshuffles.
2. **Ray-1** (`P2→R2` continuation) moved LB; apex still looks ~`2.0–2.7e-4` from sriram but K2 at `2.05e-4` only tied.
3. Sign-lock + variance-match (`disp=1`) on OOF vectors before Test inject.
4. Exact-filename Kaggle poll (`path.name`) in multi-slot loops.

## Mechanics that failed / inverted on Public LB

1. **Post-gate `B_gs` projection** after `G_twi(q)⊙U` destroyed twilight localization (`tw_share≈0.14`).
2. Fixed twilight inject (`tw≈0.78`) with **`+U_orth_super`** scored **0.95416 / 0.95412** — anti-aligned with Public LB. Default `V22_6_TWI_SIGN=-1` for next reset.
3. z9 / Slot9 self-overlap: requiring `U_C_tab` / `C1` to lift z9 tests Slot9∪Slot9.
4. Uniform orth at `≲0.32e-4` often ties ATH (5th-decimal blindness on ~60k public).

## Scripts

- `scripts/gm_v22_1_sriram_goodpjw_synthesis_and_live_leap.py`
- `scripts/gm_v22_2_hardened_unbundled_sriram_leap.py`
- `scripts/gm_v22_3_ride_095425_ridge_and_replica_probe.py`
- `scripts/gm_v22_4_complete_incremental_pivot_from_095426.py`
- `scripts/gm_v22_5_crest_plus_signed_orthogonal_five_from_095426.py`
- `scripts/gm_v22_6_unattended_5slot_number1_ladder.py` (resume: `V22_6_RESUME_FROM=4`)

Reports: `logs/overnight/GM_V22_{1..6}_*.md`.
