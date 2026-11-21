# GM V18.5 — Clean Peer Homophily (No Self-`z`, Symmetric `z_all`)

**STRICT `NO_SUBMIT=True` · 0 GPU pods · local CPU (~110s)**  
Script: `scripts/gm_v18_5_clean_peer_homophily_no_self_z.py`  
Artifacts: `/tmp/s6e11_v16/out/v18_5.json`, `v18_5_clean_peer.npz`

---

## Verdict

**F2 cured.** Dropping self-`z` + symmetrizing `z_S5` test twin moves peer LGB from F2=0.048 → **0.030–0.035**, with **5/5×3**, sf24–25, t≈9–10.

| Candidate | F1 | F2 | F3 | ρ(R2) | Elite S5? | Stage |
|---|---|---|---|---|---|---|
| P1_peer_linear | ✓ | ✗ 0.074 FE | ✗ +0.076 | +0.004 | 5/5×3 | ✗ |
| **P2_peer_only_lgb** | ✓ | ✓ **0.0347 FE** | ✓ +0.037 | +0.023 | **YES** | **✓** |
| P3_cohort_only_lgb | ✓ | ✗ 0.035 R2 | ✓ | +0.161 | 1–2/5 | ✗ |
| **P4_peer_clean_lgb** | ✓ | ✓ **0.0301 FE** | ✓ +0.037 | +0.025 | **YES** | **✓** |

---

## Step 1 — Symmetric `z` + zero self-`z` matrix

### Fix 1 — symmetric Slot5

`z_S5_test_sym = z_ATH_t_sym + 0.01045·FE_t + 0.0055·d1_t + 0.020·u_R2_t` with `z_ATH_t_sym = Bt9 @ Ridge(Bo9→z_ATH)` (**not** LIVE logits).

| Check | Value |
|---|---|
| corr(z_S5, FE) OOF | +0.0058 |
| corr(z_S5_sym, FE) test | +0.0003 |
| \|Δ\| | **0.0055** (warn vs &lt;0.005; LIVE-vs-FE was 0.022) |

### Fix 2 — no self-`z`

Dropped `abs_z_S5`, `z_S5_rank`, `peer_diff = z − peer_z`. Kept leave-self-out `peer_z` / `peer_logcnt` / **peer−peer** diffs only.

| Bank | Cols |
|---|---|
| `X_peer_only` | **24** (8 orth peer_z + 8 logcnt + 8 peer−peer diffs) |
| `X_cohort_only` | **22** (ranks + financial ratios; **zero z**) |
| `X_peer_clean` | **46** |

**Pre-orth note:** linear `frozen_lofo_orth(peer_z | Bo10, α=1)` does **not** drive raw peer leak &lt;0.015 (nonlinear Family_E remains in collision means; post-check still ~0.04–0.07). Curing happens at the **model residual + Bo10 orth** stage (self-`z` removal was the decisive F2 fix).

---

## Step 2 — Fits on `g_S5` / `h_S5`

P1 Ridge(α=100) on 8 peer_z · P2/P3/P4: 3-seed LGB (`leaves=6`, `n_est=40`, `min_data=500`) · final orth `α=1.0` vs Bo10.

### Top 15 — `P4_peer_clean`

| # | Feature | Gain |
|---|---|---|
| 1 | `peer_z_orth_k2_balance` | 88.9 |
| 2 | `peer_diff_k1_amount − k6_mer_amt` | 33.0 |
| 3 | `peer_diff_k2_balance − k3_velocity` | 20.3 |
| 4–8 | peer_z mrisk / amount / distance / velocity / cents diffs | … |
| 9+ | cohort ranks secondary | … |

### Top 15 — `P3_cohort_only`

Leaders: `rankdiff_hours_G2−global`, `rank_mrisk_G1`, `drain_ratio`, velocity/distance cohort ranks — alone they **fail** (couple to R2 / weak lift).

---

## Step 3 — Slot5 scorecard

| Signal | z_S5 | Slot5_twin | NNLS_S5 |
|---|---|---|---|
| **P2 @0.03** | **5/5 +19.0** min+12.0 sf**24** t=9.6 | **+33.5** sf25 t=10.3 | **+20.5** sf24 |
| **P4 @0.03** | **5/5 +17.9** min+11.3 sf**24** t=9.3 | **+31.6** sf25 t=10.1 | **+19.6** sf25 |
| P1 @0.03 | +20.7 (F2/F3 fail) | +37.8 | +22.5 |
| P3 | 2/5 −0.8 | 1/5 | 2/5 |

P2 folds on z_S5: `[+12.0, +18.7, +22.7, +22.9, +18.6]`.

---

## Staged NOSUBMIT

| File | Role |
|---|---|
| `sub_v18_5_P2_peer_only_lgb_w0p03_NOSUBMIT.csv` | peer-only (tightest physics; F2=0.0347) |
| `sub_v18_5_P4_peer_clean_lgb_w0p03_NOSUBMIT.csv` | peer+cohort (F2=0.0301; prefer if submitting) |

**Prefer P4** for next live slot (better F2 margin). P2 is the ablation confirming peer alone carries the lift.

---

## Guardrails

| Check | Status |
|---|---|
| Submit | **NOT done** |
| Pods | **0** |
| ATH | **0.95394** (unchanged) |

---

## Bottom line

1. **Self-`z` was the F2 bug** — not peer collisions per se.  
2. **P2/P4 are first forensic-clean Slot6 ammo since Slot5**, with ρ(R2)≈0.02–0.025 and dual/triple elite OOF.  
3. Cohort-only is dead; linear peer still leaks FE/LIVE_DC.

---

*End GM V18.5*
