# V26 Score Leap (2026-11-18)

Prior ATH **0.95440** → new personal ATH **0.95487** (`sub_v26_PUB_boardmodel_v4.csv`).

## Lever discovery (inline search)

| Source | Artifact | Claimed / pinned |
|--------|----------|------------------|
| `sriram00777/s6e11-pinned-inputs-095422` (18 Nov) | `good_step_d_095482.csv` | **0.95482** |
| same | `busy_tabm_095465.csv` | **0.95465** |
| same | parents 0.95447 / 0.95442 | Asterios formula recon |
| `asteriospanoras/s6e11-public-board-files` | `goodpjw2008_boardmodel_v4_submission.csv` | newer board-model pin |
| goodpjw notebook | Board Model + RealMLP | LB **0.95489** (author) |

## Live 7-slot verification

| # | Tag | Public | Δ vs 0.95440 |
|--:|-----|-------:|-------------:|
| 1 | PUB good stepD | **0.95482** | +42e-5 |
| 2 | PUB busy TabM | **0.95465** | +25e-5 |
| 3 | DUAL stepD×busy | **0.95476** | +36e-5 |
| 4 | **PUB boardmodel v4** | **0.95487** | **+47e-5** |
| 5 | PUB parent 0.95447 | **0.95447** | +7e-5 |
| 6 | PUB stepC 0.95450 | **0.95450** | +10e-5 |
| 7 | ATH→stepD @6e-4 | **0.95469** | +29e-5 |

## Valid leap law (confirmed again)

**Consume newer public near-tops** (and light duals of them) dominates micro-orth / w-grid on the old 0.95440 manifold. Boardmodel v4 beat stepD by +5e-5.

## Next

Lock `sub_v26_PUB_boardmodel_v4.csv` as ATH. Remaining quota: dual(boardmodel_v4, stepD) / pull goodpjw 0.95489 recipe if published; do **not** reburn Ray-1/twilight.

## V26b — last 3 slots (same day)

Pulled goodpjw kernel `…-lb-0-95489`: **step E** = our prior ATH 0.95487; **step F** = author board file.

| Tag | Public | Note |
|-----|-------:|------|
| PUB stepF | **0.95489** | consume author board |
| DUAL F×stepD | 0.95486 | −1e-5 vs prior ATH |
| **BOLD F←E @0.5** | **0.95490** | new ATH |

New ATH file: `submissions/sub_v26b_ATH_PUB490.csv` (= `sub_v26b_BOLD_F_from_E_0p5.csv`).
