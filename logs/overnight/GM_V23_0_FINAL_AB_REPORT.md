# GM V23.0 FINAL A/B Report
Generated: 2026-11-20T16:54:55.454102+00:00

## Verdict
**Live ATH locked at `0.95534`** via public integrate (Areeb Ali). No public notebook above that score was found today. Local Rule-2 GS orthogonal levers all failed post-ATH gates → classical 5-slot star A/B aborted. Remaining quota spent on public-board / residual star probes on the new anchor.

## Rule 1 — Anchor
| Source | LB | Role |
|--------|-----|------|
| `sub_v30_PUB_areeb_95534.csv` (areebbb/…-lb-0-95534) | **0.95534** | `anchor_vec` |
| `sub_v30_PUB_gpw_95533.csv` (goodpjw2008/…-lb-0-95533) | 0.95533 | near-top probe |
| prior `sub_v29_ATH_PUB531.csv` | 0.95531 | superseded |

Obsolete V22 R2/`0.95426` origin intentionally **not** used.

## Public integration (newer notebooks vs prior ATH 0.95531)
| File | LB | Δ vs 0.95531 | Notes |
|------|-----|--------------|-------|
| PUB_areeb_95534 | **0.95534** | +3e-5 | new ATH |
| PUB_gpw_95533 | 0.95533 | +2e-5 | confirmed |
| BOLD_areeb_from_gpw_0p5 | 0.95534 | +3e-5 | tied ATH |
| PUB_malyshev (residual100 Areeb×gpw) | 0.95534 | +3e-5 | ≈ BOLD_gpw_1p0 (omr≈6e-8) |
| DUAL_areeb_gpw | 0.95533 | +2e-5 | centroid |
| STAR_areeb_along_mal_0.26e4 | 0.95534 | +3e-5 | tied |
| STAR_areeb_along_gpw_0.26e4 | 0.95532 | +1e-5 | slight regression |
| SLOT5_cent3_areeb_gpw_mal | 0.95534 | +3e-5 | Case2 geodesic, tied |
| BOLD_maths_from_areeb_0p5 | 0.95532 | +1e-5 | MathsOnChain diversity — slight regression |

Scanned / pulled (no >ATH claim): Malyshev residual100, MathsOnChain, Mayank (own 0.95044 / mix 0.95526), Georgy (no sub), Asterios board zip (steps I/J/K older / farther from ATH), Zohrab/Davut (≤0.952x).

## Local orth A/B (engine abort)
`scripts/gm_v23_0_final_5slot_ab_engine.py` admitted 5 local pairs (wide_lm, P_super, P_wide, soft diglead/weni) then **all u-vectors dropped** on LOFO vs `B_macro` after re-anchor to 0.95534 (`pos≤1/5`). Pre-flight: no mutually orthogonal live-safe levers → Slots 1–4 classical inject skipped.

## Synthesis takeaway
Public consume/BOLD on the **0.9553x board manifold** remains the only source of ATH gains. Micro-orth along Areeb↔goodpjw↔Malyshev saturates at **0.95534**; overshoot past gpw at 0.26e-4 already costs ~2e-5. Next leap requires a **new public ≥0.95535** or a diversity axis that clears nested gates against the Areeb surrogate — not V22-era stump/Orig20k alone.

## Artifacts
- Engine: `scripts/gm_v23_0_final_5slot_ab_engine.py`
- ATH lock: `submissions/sub_v30_ATH_PUB534.csv`
- Run log: `/tmp/s6e11_v30/v23_0_run.log`
