# S6E11 DS Audit — All Submissions 2026-11-20

**Competition:** `playground-series-s6e11`  
**Source of truth:** Kaggle `competitions submissions --csv`  
**Day-start ATH (before any Nov 20 submit):** `0.95511` (`sub_v27b_BOLD_sub_from_I_1p0`)  
**End-of-day ATH:** `0.95534`  
**Quota used:** 10 / 10 · **Remaining:** 0  
**Net day gain:** `+2.3e-4` (0.95511 → 0.95534)

Interactive canvas: `s6e11-nov20-submission-audit.canvas.tsx`

---

## Chronological scoreboard

| # | UTC | File | Public LB | Δ vs 0.95511 | Δ vs peak 0.95534 | Family | Outcome |
|---|-----|------|-----------|--------------|-------------------|--------|---------|
| 1 | 10:45:22 | `sub_v29_PUB_gpw_sub_95531.csv` | **0.95531** | +2.0e-4 | −3e-5 | V29 consume | Day gain |
| 2 | 16:46:45 | `sub_v30_PUB_areeb_95534.csv` | **0.95534** | +2.3e-4 | 0 | V30 public | **NEW ATH** |
| 3 | 16:46:51 | `sub_v30_PUB_gpw_95533.csv` | 0.95533 | +2.2e-4 | −1e-5 | V30 public | Near-ATH |
| 4 | 16:46:58 | `sub_v30_BOLD_areeb_from_gpw_0p5.csv` | **0.95534** | +2.3e-4 | 0 | V30 BOLD | Tie ATH |
| 5 | 16:52:24 | `sub_v30_PUB_malyshev.csv` | **0.95534** | +2.3e-4 | 0 | V30 public | Tie ATH |
| 6 | 16:52:29 | `sub_v30_DUAL_areeb_gpw.csv` | 0.95533 | +2.2e-4 | −1e-5 | V30 dual | Near-ATH |
| 7 | 16:52:35 | `sub_v30_STAR_areeb_along_mal_0.26e4.csv` | **0.95534** | +2.3e-4 | 0 | V23 star | Tie ATH |
| 8 | 16:52:40 | `sub_v30_STAR_areeb_along_gpw_0.26e4.csv` | 0.95532 | +2.1e-4 | −2e-5 | V23 star | Regress |
| 9 | 16:52:45 | `sub_v30_SLOT5_cent3_areeb_gpw_mal.csv` | **0.95534** | +2.3e-4 | 0 | V23 Slot-5 | Tie ATH |
| 10 | 16:54:32 | `sub_v30_BOLD_maths_from_areeb_0p5.csv` | 0.95532 | +2.1e-4 | −2e-5 | V30 diversity | Regress |

Kaggle refs: `58609056`, `58622788`, `58622796`, `58622803`, `58623004`, `58623012`, `58623016`, `58623019`, `58623021`, `58623090`.

---

## Per-submission detail

### #1 — `sub_v29_PUB_gpw_sub_95531.csv` → **0.95531**
- **Intent:** Leap off ATH 0.95511 with one public consume.
- **Method:** Direct consume of goodpjw Board Model submission claimed LB 0.95531.
- **Source:** `goodpjw2008` board-model RealMLP (board sub @ 0.95531).
- **Axis / distance:** Raw public vector.
- **Result:** Confirmed +2.0e-4; became morning ATH.

### #2 — `sub_v30_PUB_areeb_95534.csv` → **0.95534** (NEW ATH)
- **Intent:** Integrate public notebook scored above live ATH 0.95531.
- **Method:** Direct consume of kernel `submission.csv`.
- **Source:** `areebbb/s6e11-which-signals-rank-fraud-lb-0-95534`.
- **Axis / distance:** Locked as `anchor_vec` for all later star/BOLD probes.
- **Result:** +3e-5 over morning ATH; end-of-day peak.

### #3 — `sub_v30_PUB_gpw_95533.csv` → **0.95533**
- **Intent:** Confirm near-top public sibling for dual/BOLD geometry.
- **Method:** Direct consume.
- **Source:** `goodpjw2008/s6e11-board-model-realmlp-lb-0-95533`.
- **Axis / distance:** `1−ρ` vs Areeb ≈ **0.061×10⁻⁴** (extremely close).
- **Result:** −1e-5 vs Areeb; useful probe partner, not a new peak.

### #4 — `sub_v30_BOLD_areeb_from_gpw_0p5.csv` → **0.95534** (tie)
- **Intent:** Probe past 0.95534 on the Areeb↔gpw board manifold.
- **Method:** Rank-logit extrapolate from Areeb along (Areeb−gpw) residual (~0.5× step).
- **Source:** Local blend of Areeb × goodpjw.
- **Axis / distance:** `1−ρ` vs Areeb ≈ **0.137×10⁻⁴**.
- **Result:** Tied ATH; no breakout.

### #5 — `sub_v30_PUB_malyshev.csv` → **0.95534** (tie)
- **Intent:** Integrate newest hot public residual notebook.
- **Method:** Direct consume; recipe `residual100` step=1.0 with LEADER=Areeb, PROBE=gpw.
- **Source:** `malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion`.
- **Axis / distance:** Essentially identical to staged `BOLD_areeb_from_gpw_1p0` (`1−ρ` ≈ 6×10⁻⁸).
- **Result:** Tied ATH; confirms Malyshev is on the same residual axis we already probed.

### #6 — `sub_v30_DUAL_areeb_gpw.csv` → **0.95533**
- **Intent:** Variance-cancel geodesic of the two near-tops.
- **Method:** 0.5/0.5 rank-logit average → rank-frac.
- **Source:** Areeb 0.95534 + goodpjw 0.95533.
- **Axis / distance:** `1−ρ` vs Areeb ≈ **0.015×10⁻⁴**.
- **Result:** −1e-5 vs peak (matches gpw-side pull).

### #7 — `sub_v30_STAR_areeb_along_mal_0.26e4.csv` → **0.95534** (tie)
- **Intent:** Star-topology A/B attribution on Malyshev residual axis (fixed anchor).
- **Method:** Inject from Areeb along unitized `(mal−areeb)` to target `1−ρ = 0.26×10⁻⁴`.
- **Source:** Areeb + Malyshev direction.
- **Result:** Tied ATH; axis non-destructive at this step size.

### #8 — `sub_v30_STAR_areeb_along_gpw_0.26e4.csv` → **0.95532**
- **Intent:** Overshoot past gpw (areeb↔gpw only ~0.06e-4 apart, so 0.26e-4 is past gpw).
- **Method:** Star inject along unitized `(gpw−areeb)` to `0.26×10⁻⁴`.
- **Source:** Areeb + goodpjw direction.
- **Result:** **−2e-5** vs peak — overshoot regresses.

### #9 — `sub_v30_SLOT5_cent3_areeb_gpw_mal.csv` → **0.95534** (tie)
- **Intent:** V23 Slot-5 Case-2 synthesis when no single probe beat ATH.
- **Method:** Rank-logit mean of Areeb + gpw + Malyshev → rank-frac.
- **Source:** Three near-top vectors.
- **Axis / distance:** `1−ρ` vs Areeb ≈ **0.000013×10⁻⁴** (near-duplicate of Areeb).
- **Result:** Tied ATH (expected under Case-2 when centroid collapses to leader).

### #10 — `sub_v30_BOLD_maths_from_areeb_0p5.csv` → **0.95532**
- **Intent:** Last daily slot — off-manifold diversity (MathsOnChain alone is ~334e-4 from ATH).
- **Method:** BOLD from Areeb along Maths residual to `1−ρ ≈ 0.50×10⁻⁴`.
- **Source:** `areeb` + `mathsonchain/s6e11-fraud-what-worked-what-didn-t`.
- **Result:** **−2e-5** vs peak — small diversity step not helpful.

---

## Outcome tally

| Bucket | Count | Files |
|--------|-------|-------|
| NEW ATH / TIE @ 0.95534 | 5 | areeb, BOLD×gpw 0.5, malyshev, STAR_mal 0.26, cent3 |
| Near-ATH (−1e-5) | 2 | gpw 0.95533, dual |
| Day gain then superseded | 1 | V29 0.95531 |
| Regress (−2e-5) | 2 | STAR_gpw 0.26, BOLD_maths 0.5 |

---

## Not submitted (staged only)

STAR gpw 0.14/0.34/0.48/0.70/1.00 · STAR mal 0.48 · BOLD_gpw_1p0 · BOLD_areeb_from_ATH · BOARD step I/J/K · Mayank · Maths raw · LAST_fwd/opp variants.

## Engine note

`scripts/gm_v23_0_final_5slot_ab_engine.py` admitted local Rule-2 pairs but **all GS `u_k` failed LOFO** after re-anchor to 0.95534 → classical 5-slot orth inject aborted; evening slots were public-manifold star/BOLD/synthesis instead.
