# GM V19.0 — Slot 7 LIVE: Zero-Echo `M5_peer_hybrid` (α★=0)

**0 GPU pods · submitted · Complete**  
Script: `scripts/gm_v19_0_break_loop_synthesis_and_slot7_submit.py`  
File: `submissions/sub_v19_0_LIVE_slot7_zero_echo_peer_hybrid.csv`  
Ref: **58342549** · Artifacts: `/tmp/s6e11_v16/out/v19_0.json`

---

## Verdict

| Item | Value |
|---|---|
| Public LB | **0.95391** (**−3e-5** vs Slot5 ATH **0.95394**) |
| Winner | **α★ = 0.00** — pure cached `M5_peer_hybrid` (R1-free) |
| `G_regime` | **QUARANTINED** — both `G_ridge` / `G_lgb` failed pre-PCHIP `raw_disp` |
| vs Slot6 | Better than Slot6 R1 (**0.95392** was −2e-5); still below Slot5 |

**Hold Slot5 ATH 0.95394.** Zero-echo peer hybrid is gate-clean OOF but does not lift public LB at the twin-safe micro-step (`w★=0.009`, `1−ρ≈0.15e-4`).

---

## Step 1 — Cached M5 + joint 3-base

Loaded `M1..M5` from `/tmp/s6e11_v16/out/v18_9_peers.npz` (**no D_f retrain**).

| Check | Result |
|---|---|
| pre-PCHIP `raw_disp` | **1.000** ✓ |
| pre-PCHIP `raw_KS` | 0.031 (logged; repaired by PCHIP) |
| post-PCHIP `raw_KS` | **0.00005** |
| F1 / F2 / F3 | ✓ 1.000 / **0.0165** / 0.044 |

Unified `w` sweep (same `w` on all 3 Slot5 bases):

| w | 1−ρ×1e4 | z_S5 | twin | NNLS | Notes |
|---|---|---|---|---|---|
| **0.008** | 0.117 | **5/5** | **5/5** | **5/5** | only full 5/5×3 |
| 0.010–0.020 | 0.18–0.73 | 5/5 | **4/5** | 5/5 | twin Fold2 soft |
| 0.022+ | ≥0.88 | 4/5 | 4/5 | 5/5 | — |

`w_M5★=0.008` (strict `1−ρ∈[0.22,0.55]` incompatible with twin 5/5). Wilcoxon `p<0.005` and `sf≥19` hold at 0.008 on all 3 bases.

---

## Step 2 — `G_regime` (hygiene-locked)

Built `X_regime` (34 cols): Method-B PCHIP cross-pillar contrasts (`c1..c4`) + U5 + nested P4 disk assets (`L4.2/4.3/4.5 pure5f`, `L4.3_ext`) + hierarchical EB (`ch×dev`, `ch×dev×cat`, `cat×amtones`, `dev×amtones`) + smooth cents — **zero logcnt, zero R1**.

| Learner | pre-PCHIP `raw_disp` | `raw_KS` | Gate |
|---|---|---|---|
| `G_ridge` | **0.840** (<0.85) | 0.070 | **QUARANTINE** |
| `G_lgb` | **0.621** | 0.144 | **QUARANTINE** |

Individual-family gate blocked blend → **`G_regime` unused**. Law 3 (`v10_lgb_d3_bag5` trap) fired correctly.

---

## Step 3 — Promotion, negative-ray, submit

| α | Improves meanΔ+min on all 3 vs M5? | Hard promote |
|---|---|---|
| 0.00 / 0.15 / 0.25 / 0.35 | No (`G` empty) | No |

**α★ = 0.00.** Final `w★=0.009` (max `1−ρ` still with **5/5×3** + sf≥19).

| Guard | Value | Pass |
|---|---|---|
| corr vs Slot6 Δ | −0.020 | ✓ &lt;0.15 |
| corr vs `R1_pure` test | +0.020 | ✓ |
| corr vs `d1_v15` | +0.016 | ✓ |
| mono Spearman(z, z+wu) | 1.000 | ✓ |
| KS(cand, Slot5) | 0 | ✓ |
| rows / NaNs | 299,984 / 0 | ✓ |

OOF @ `w★=0.009`:

| Base | folds | mean Δe-5 | min | sf | t |
|---|---|---|---|---|---|
| z_S5 | 5/5 | +2.22 | +0.43 | 19 | 3.50 |
| twin | 5/5 | +3.99 | +0.02 | 19 | 3.31 |
| NNLS | 5/5 | +2.56 | +0.88 | 20 | 4.06 |

### Kaggle

| Item | Value |
|---|---|
| Submit | Success · ref **58342549** |
| Remaining today (post) | 3 |
| Public LB | **0.95391** (**−3e-5** vs 0.95394) |

---

## Architectural laws — outcome

1. Cached V18.9 peers — reused (no 277s distill).  
2. `L_fold` / PCHIP — applied on contrasts; M5 kept V18.9 unitization (per-fold re-z destroyed twin 5/5 in a dry run).  
3. Pre-PCHIP `raw_disp`/`raw_KS` — **quarantined both G learners**.  
4. Individual G audit before blend — enforced.  
5. Nested P4 disk + hierarchical EB — loaded; unused after quarantine.  
6. Smooth cents + Spearman mono guard — pass.  
7. Negative-ray vs Slot6/R1/`d1_v15` — all `|corr|<0.03`.

---

## Implications

1. **OOF-clean ≠ LB lift** at micro-steps (`1−ρ~0.15e-4`): Slot7 −3e-5 mirrors Slot6 R1 failure class (strong OOF, flat/down LB).  
2. **`G_regime` raw_disp collapse** (0.62–0.84) confirms Law 3 — do not PCHIP-wash a broken test scale.  
3. **Hold Slot5 0.95394.** Next LB attempt needs a residual with (a) `raw_disp∈[0.90,1.10]` pre-PCHIP, (b) twin-safe headroom above `w=0.01`, (c) negative-ray clean — not another sub-0.2e-4 polish.

---

*End GM V19.0*
