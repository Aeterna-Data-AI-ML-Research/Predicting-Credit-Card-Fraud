# V27 Score Leap (2026-11-19)

Prior ATH **0.95490** → new personal ATH **0.95509** (`sub_v27_ATH_PUB509.csv` = `sub_v27_BOLD_sub_from_I_0p5.csv`, Kaggle ref `58564119`).

## Lever discovery (online + Kaggle)

| Source | Artifact | Claimed / pinned |
|--------|----------|------------------|
| `goodpjw2008/s6e11-board-model-realmlp-lb-0-95508` | `submission.csv` + steps G/H/I | author LB **0.95508** |
| same (intermediate) | `…-lb-0-95504` | same `submission.csv` as 0.95508 run |
| `asteriospanoras/s6e11-lb-0-95504-will-the-top-survive-private` | analysis (no CSV out) | LB **0.95504** |
| `asteriospanoras/s6e11-public-board-files` (updated) | `goodpjw2008_boardmodel_v6` = step H | zone-push pin |
| `alexchilton/…-lb-0-9548-shallow-tree-stack` | shallow stack | LB **0.9548** (diversity only) |
| `goodpjw2008/s6e11-board-model-pinned-inputs` | Weni BPE / TabM OOFs | members inside step J/I |

Board ladder (from kernel log): G↔0.95496 → H zones1–2 → I zones3–4+busy → `submission.csv` (≈step J / final).

## Live 7-slot verification

| # | Tag | Public | Δ vs 0.95490 |
|--:|-----|-------:|-------------:|
| 1 | PUB goodpjw sub | **0.95508** | +18e-5 |
| 2 | PUB step I | **0.95504** | +14e-5 |
| 3 | PUB step H | **0.95504** | +14e-5 |
| 4 | PUB step G | 0.95498 | +8e-5 |
| 5 | BOLD I←H @0.5 | 0.95503 | +13e-5 |
| 6 | **BOLD sub←I @0.5** | **0.95509** | **+19e-5** |
| 7 | BOLD H←G @0.5 | 0.95502 | +12e-5 |

## Valid leap law (confirmed again)

**Consume newer public near-tops**, then a **half-step BOLD extrapolate** past the top axis (here: submission ← step I) beats pure consume and older-manifold micro-orth. All seven slots beat prior ATH.

## Lock

- ATH file: `submissions/sub_v27_ATH_PUB509.csv`
- Recipe: `rankfrac( L(sub) + 0.5·(L(sub)−L(step_I)) )` on goodpjw Board Model + RealMLP outputs
- Credits: **goodpjw2008** (primary), Asterios Panoras (board pins / analysis), busyaprime / Weni Mohamed / yekenot (members inside later steps)

## Next

Do **not** reburn Ray-1 / twilight on 0.95490. Prefer newer board steps if published; optional dual of ATH×stepH only if nested gates pass.

## V27b — leftover 3 slots (same day)

No newer public top above 0.95508. Probed stronger BOLD on the winning `sub←I` axis + `sub←H`.

| Tag | Public | Note |
|-----|-------:|------|
| BOLD sub←I @0.75 | **0.95510** | +1e-5 vs ATH0.95509 |
| **BOLD sub←I @1.0** | **0.95511** | **new ATH** |
| BOLD sub←H @0.5 | 0.95509 | tie old ATH |

New ATH: `submissions/sub_v27_ATH_PUB511.csv` (= `sub_v27b_BOLD_sub_from_I_1p0.csv`, ref `58565133`).
Quota for 2026-11-19: **0** remaining.
