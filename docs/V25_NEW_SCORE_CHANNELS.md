# V25 New Score Channels

Research for lift past ATH **0.95440** via *new channels* (not Ray-1 / twilight / native solo).  
Full gates: `logs/overnight/GM_V25_NEW_SCORE_CHANNELS_REPORT.md` · meta `/tmp/s6e11_v16/out/v25_new_channels_meta.json`.

## Screened

| ID | Source | What |
|----|--------|------|
| C1 | Alex OOF bank (updated 2026-11-17) | **40** OOF+test models + family rank-means |
| C2 | Mathsonchain 17 | Alternate external singles |
| C3 | Peer files (gpw/bhuvan/step11/n15) | Disagreement / triple / dual_GB×digitGPT2 |
| C4 | ATH × gated channel | Small-w injects (board geometry) |

## VALID new channels (stage for tomorrow)

Small-w into ATH only (`w∈{0.05,0.08}`); raw5≥4/5. Kill if public ≤0.95439.

| Priority | File | Channel | ρ(ATH) e-4 | Gate note |
|---------:|------|---------|----------:|-----------|
| 1 | `sub_v25_ATH_x_alex_joint_glm_05.csv` | Joint-GLM / GLM-boosted LGB family | 0.54 | raw5 4/5 +2.01 |
| 2 | `sub_v25_ATH_x_alex_top5_solo_05.csv` | Rank-mean of top-5 alex solos | 0.50 | raw5 4/5 +1.79 · gb8 4/5 |
| 3 | `sub_v25_ATH_x_alex_part5_lgb_05.csv` | Part-5 LGB recipe family | 0.58 | raw5 4/5 · gb8 4/5 |
| 4 | `sub_v25_ATH_x_math_rankmean17_05.csv` | Mathson 17 rank-mean | 0.72 | raw5 4/5 · gb8 4/5 |
| 5 | `sub_v25_ATH_x_alex_digit_gpt2_05.csv` | Digits + GPT-2 tokens (unique FE) | 0.64 | raw5 4/5 · gb8 4/5 |
| 6–10 | same tags `_08` | Higher mass | 1.3–1.8 | Only if `_05` ties/lifts |

Best single under the hood: alex `3d3b857b` (LGB4 + joint GLM boost) raw5 **5/5**.

## INVALID / do not burn slots

| Channel | Result | Why |
|---------|--------|-----|
| Alex **NN** (TabM/RealMLP family) | raw5 2/5 −2.0 | Hurts ATH surrogate |
| Alex CatBoost / depth-1 families | raw5 2/5 | Negative |
| GS-orth alex → ATH @0.30e-4 | ρ≈0 noop / ath-orth **0/5** all | Same as V23 orth death |
| Peer DIS/ANTI @ α≥0.15 | ρATH **30–340** e-4 | Leaves ATH manifold |
| `DUAL_GB_x_digitGPT2` / n15mixDigit | ρ **70–130** e-4 | Wrong neural axis vs locked v15 |
| TRIPLE_near / DUAL_step11_gpw32 | ρ≈5.9 e-4 | Farther than micro-w; no new neural |
| Solo natives / KNN (V24.3) | already closed | Diversity only if at all |

## Law

New **score channels** that clear nested raw5 are still **cross-family mass into the locked dual_GB+v15 ATH**, not replacements for it. Replacing v15 with digitGPT2 or large peer-disagreement destroys geometry.

Merge with existing tomorrow queue: after V24.1 H1/H3/H5, prefer V25 alex_joint_glm / top5 / part5 before native HGB/MLP.
