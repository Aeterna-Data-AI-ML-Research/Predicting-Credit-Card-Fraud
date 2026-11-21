# GM V18.7 — Strict Peer V2 Upgrade + Ortho Combos

**STRICT `NO_SUBMIT=True` · 0 GPU pods · ~141s**  
Script: `scripts/gm_v18_7_strict_peer_upgrade_and_ortho_combo.py`  
Artifacts: `/tmp/s6e11_v16/out/v18_7.json`, `v18_7_strict_peer.npz`

---

## Verdict

**No stage.** Pool-size matching + multi-shrinkage + heavier LGB regularization **did not clear `sf≥19` on all 3 Slot5 baselines**. Best near-miss: `S4+0.30·R1` (sf **19/18/19**, twin short by 1).

| Candidate | F2 | z_S5 | twin | NNLS | sf | Stage |
|---|---|---|---|---|---|---|
| S1 ridge | ✓ 0.031 | +4.88 sf17 | +4.99 sf17 | +5.29 sf18 | ✗ | ✗ |
| S2 lgb_reg | ✗ 0.050 | +3.75 sf19 | +4.40 sf16 | +4.09 sf18 | ✗ | ✗ |
| S3 lgb_med | ✗ 0.049 | +3.13 sf18 | +4.14 sf16 | +4.09 sf17 | ✗ | ✗ |
| S4 ens | ✓ 0.031 | +4.60 sf**19** | +5.46 sf**18** | +4.99 sf**18** | soft | ✗ |
| S4+U5 | ✓ 0.022 | +4.90 sf16 | +4.80 sf16 | +5.24 sf18 | ✗ | ✗ |
| **S4+0.30·R1** | ✓ 0.024 | **+6.96 sf19** t3.50 | **+9.20 sf18** | **+7.50 sf19** | twin−1 | ✗ |
| S4+U5+0.30·R1 | ✓ 0.017 | +6.55 sf19 | +8.17 sf18 | +7.07 sf20 | twin−1 | ✗ |

---

## Step 1 — Strict peer bank v2

- **14 keys**, va-only leave-self-out; test `m0_test = m0_va × 2.143`
- Per key: `m3`, `m15`, `pos_rate_m10`, rate-normalized `logcnt` + m15 diffs on k1..k6 → **71 feats**
- Half-subsample CV on test `k2_m15` std: **0.013** (stable)
- Support: most keys `mean_cnt_va ∈ [25, 1000]`; **`k14_aq500_bq500` too sparse (1.6)** — noise

---

## Step 2 — Learners (zero echo)

| Model | Recipe |
|---|---|
| S1 | Ridge α=300 on 42 mean/pos cols |
| S2 | 5-seed LGB leaves=4, min_child=1200, λ=50 |
| S3 | 5-seed LGB leaves=6, min_child=800 + cohort |
| S4 | 0.25·S1 + 0.40·S2 + 0.35·S3 |

---

## Step 3 — Scorecard notes

S4 folds on z_S5: `[+5.87, +6.77, +0.30, +4.69, +5.38]` — Fold2 still the weak link (same pattern as V18.6).  
R1 blend lifts mean and min_fold but **does not fix twin sf=18**.  
U5 improves F2, hurts sf.

---

## Staged NOSUBMIT

**None** (gate: `sf≥19` ∧ `t≥3.2` on **all three** bases).

---

## Guardrails

| Check | Status |
|---|---|
| Submit | **NOT done** |
| Pods | **0** |
| ATH | **0.95394** |

---

## Bottom line

1. Strict peer physics remains real (~+4.5–5e-5) and forensically clean under S1/S4.  
2. **sf consistency is the bottleneck**, not F2 — Fold2 stays soft even with multi-m0 / reg LGB.  
3. Do not submit; next peer work needs a Fold2-specific robustness fix or a different orthogonal family, not more of the same bank.

---

*End GM V18.7*
