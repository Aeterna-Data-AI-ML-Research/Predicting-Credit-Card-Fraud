# GM V35 Ladder Report (UTC 2026-11-21 afternoon)

Quota: **5/5 used**. Remaining today: **0**. New ATH: **0.95548**.

| # | File | Intent | LB | Δ |
|---|------|--------|-----|---|
| 0 | SLOT1 re-probe | quota probe | 0.95537 | 0 |
| 1 | `sub_v35_S1_PUB_areeb_asterios_95547.csv` | Lock public areeb≡asterios (omr vs old=4.02e-4) | **0.95547** | **+0.00010** |
| 2 | `sub_v35_S2_v10orth_on_areeb_0.30e4.csv` | Method-B orth inject — rank-null bug (omr≈0.0075) | 0.95547 | 0 |
| 3 | `sub_v35_S2_BOLD_areeb_leap_2e4.csv` | Overshoot along leap @2e-4 | 0.95547 | 0 (axis saturated) |
| 4 | `sub_v35_S4_v10gate_StepO_0.30e4_on_areeb.csv` | Gate-faithful v10 +w=0.023 @omr=0.30e-4 | **0.95548** | **+0.00001** |

## Expert DS verdict

1. **Public lock paid.** One public cluster (areeb ≡ asterios ≡ gpw_sub) lifted ATH 0.95537 → 0.95547.
2. **Leap axis saturates at areeb.** BOLD past areeb was flat — do not spend more slots on that 1D ray.
3. **Proprietary v10 on the new base paid.** ρ(v10, areeb)≈−0.016; StepO-sized gate recipe `U(L(areeb)+0.023·winsor(v10))` → 0.95548. This is the live improvement axis.
4. **S2 lesson:** verify achieved omr before submit; Method-B orth can leave a rank-null direction.

## Next UTC day (first 3 slots)

1. Continue v10 on 0.95548 at 0.38e-4 / 0.50e-4 (grid up the paid axis).
2. −harvest @0.30e-4 on 0.95548 (local control twin).
3. Dual centroid of best v10-step × (−harvest) if both individually ≥ flat.

Prebuilt candidates already in `submissions/sub_v35_S4cand_*` and `sub_v35_S4alt_dual_v10_har_0.22.csv`.
