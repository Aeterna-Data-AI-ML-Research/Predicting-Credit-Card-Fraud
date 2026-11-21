# GM V18.9 — 100% R1-Free, Zero-Logcnt Strict Peer + Continuous Teacher Distill

**STRICT `NO_SUBMIT=True` · 0 GPU pods · ~277s CPU**  
Script: `scripts/gm_v18_9_zero_r1_zero_logcnt_peer_and_teacher_distill.py`  
Artifacts: `/tmp/s6e11_v16/out/v18_9.json`, `v18_9_peers.npz`

---

## Verdict

**Zero-logcnt restored F2.** Dropping `peer_logcnt` / amt-bal frequency columns brings max `|corr(Bo10)|` back to **0.017–0.032** (V18.6 regime), vs **0.047–0.050** in V18.7/V18.8.

**Continuous teacher `D_f` works.** Distilling `z_S5_oof[va]` → `ρ(D_f, z_S5) ≈ 0.965–0.968` on tr/test (vs V18.8 binary `B_h` ~0.915 AUC). **Zero `y[va]`.**

**One NOSUBMIT staged (100% R1-free):** `M5_peer_hybrid` @ `w=0.008` clears F1/F2/F3, 5/5×3, sf≥19, t≥3.2.

| Candidate | F2 | Fold2 Δ (e-5) | sf (S5/twin/NNLS) | Stage |
|---|---|---|---|---|
| M1_P2_v186 | ✓ 0.023 | **−4.80** | 18/16/19 | ✗ Fold2 |
| M2_P4_v186 | ✓ 0.028 | **−4.15** | 20/19/21 | ✗ Fold2 |
| M3_peer12_va | ✓ 0.019 | **0.00** | 17/19/18 | ✗ sf/twin |
| **M4_peer12_distill** | ✓ 0.032 | **+4.56** | 17/16/16 | ✗ sf/t |
| **M5_peer_hybrid** | ✓ **0.017** | **+0.04** | **19/19/20** | **✓** |
| M* + U5/U6 | mostly ✓ F2 | mixed | sf often <19 on twin | ✗ |

---

## Forensic confirmation (V18.6 vs V18.7/18.8)

1. **F2 spike was logcnt / frequency rediscovery of `R2_orth`.**  
   V18.9 Bank A = peer_z means + pairwise diffs only (**zero `logcnt`**, **no input qlock**) → F2 returns to **0.017–0.028**, leak col often `Family_E`/`d1_v4` not `R2_orth` domination.
2. **Fold2 collapse when Bo10 projects `R2_orth`:** M1/M2 (va-only, like V18.6) still lose Fold2 (−4e-5).  
   **M4 distill full-graph flips Fold2 to +4.56e-5** — continuous `D_f` peers recover the fold that binary `B_h` (V18.8) could not.
3. **M5 hybrid** blends va-strict stability (sf) with distill Fold2 lift → only candidate clearing all gates **without R1**.

---

## Step 1 — Banks

### Bank A — Exact V18.6 va-only (zero logcnt)

| Matrix | Cols | Recipe |
|---|---|---|
| `X_peer8_va` | **36** | 8 means @ `m0=5` + C(8,2)=28 diffs; `m0_test≈10.71` (=5×SCALE); per-col frozen-β orth; **no qlock, no logcnt** |
| `X_peer12_va` | **52** | 12 means @ m0∈{5,15} + 28 top diffs; same orth/qlock/logcnt rules |

Extra keys: `k9=(mer, round(bal,-1))`, `k10=(cat, round(amt,0))`, `k11=(amt_cents,cat)`, `k12=(bal_cents,cat)`.

### Bank B — Continuous teacher distill (full graph, zero y[va])

`X_rich` = 24 raw + q20 + coarse q20-logfreq (not exact k1..k12) + cohort ranks (72 cols).

For each outer fold `f`:
- 5-subfold OOF `D_f` inside `va` → `ρ(va_oof, z_S5[va]) ≈ 0.964–0.968`
- Full-va `D_f` predicts `tr` + `test`; **va peers use `z_S5_oof[va]` directly**
- Leave-self-out peer means on `k1..k12` @ `m0=15` over all **999,944** rows → `X_peer12_distill` (40 cols: 12 means + 28 diffs)

| Fold | ρ(va OOF) | ρ(tr, z_S5) | ρ(te, z_S5_sym) |
|---|---|---|---|
| 0 | +0.966 | +0.965 | +0.968 |
| 1 | +0.964 | +0.966 | +0.968 |
| 2 | +0.968 | +0.965 | +0.968 |
| 3 | +0.966 | +0.965 | +0.967 |
| 4 | +0.966 | +0.965 | +0.968 |

---

## Step 2 — Learners (LOFO on `(g_S5, h_S5)`, then orth vs `Bo10_S5`)

| Model | Features | Fit |
|---|---|---|
| M1_P2_v186 | `X_peer8_va` | 5-seed LGB (leaves=6, mcs=500, n=40, lr=0.03) |
| M2_P4_v186 | `[X_peer8_va, X_cohort]` | same LGB |
| M3_peer12_va | `X_peer12_va` | Ridge(α=250) + 5-seed LGB + CatBoost(depth=3) |
| M4_peer12_distill | `X_peer12_distill` | Ridge(α=250) + 5-seed LGB |
| M5_peer_hybrid | 0.25·(M1+M2+M3+M4) | clean blend |

---

## Step 3 — R1-free scorecard (`w_R1 = 0`)

Gates: F1∈[0.90,1.10], F2&lt;0.035, F3&lt;0.05, 5/5×3 bases, sf≥19×3, t≥3.2×3.

### Solo peers

| | F1 disp | F2 | F3 | Fold2 | 5/5×3 | sf | t (S5) |
|---|---|---|---|---|---|---|---|
| M1 | 0.996 | 0.023 ✓ | 0.026 ✓ | −4.80 | ✗ | 18/16/19 | 2.25 |
| M2 | 0.995 | 0.028 ✓ | 0.032 ✓ | −4.15 | ✗ | 20/19/21 | 3.17 |
| M3 | 0.998 | 0.019 ✓ | 0.040 ✓ | 0.00 | ✗ twin | 17/19/18 | 3.51 |
| M4 | 0.990 | 0.032 ✓ | 0.045 ✓ | **+4.56** | ✓ | 17/16/16 | 1.97 |
| **M5** | 0.995 | **0.017 ✓** | 0.043 ✓ | **+0.04** | **✓** | **19/19/20** | **3.54** |

M5 per-base @ best w:
- `z_S5` @0.02: folds `[+3.40, +4.61, +0.04, +7.51, +5.64]` sf=19 t=3.54
- `z_Slot5_twin` @0.008: folds `[+3.32, +6.10, +0.08, +4.69, +4.29]` sf=19 t=3.34
- `z_NNLS_S5` @0.03: folds `[+5.21, +7.76, +0.80, +11.47, +8.06]` sf=20 t=4.11

### + U5 / U6 (still `w_R1=0`)

U5/U6 lift mean Δ and often Fold2, but **twin sf drops below 19** for nearly all M1–M5 combos. Closest: `M5+U6` (sf 20/18/21 — twin short), `M2+U6` (sf 19/18/19). **None staged.**

---

## Staged (NOSUBMIT only)

| File | Signal | w | Notes |
|---|---|---|---|
| `submissions/sub_v18_9_M5_peer_hybrid_w0p008_NOSUBMIT.csv` | M5_peer_hybrid | 0.008 | qlocked to LIVE; **no R1**; do not auto-submit |

---

## Laws held

- **Zero `y[va]` echo:** va peers from `z_S5_oof[va]` / va-pool leave-self-out only; `D_f` trained on `(X_rich[va], z_S5[va])` never sees `y[va]`.
- **Zero `R1`:** no R1 feature, weight, or staged combo.
- **Zero logcnt in peer banks:** F2 restored.
- **NO_SUBMIT / 0 pods.**

---

## Implications

1. **Do not reintroduce `peer_logcnt` / amt-bal frequency into peer matrices** — it reopens the R2 rediscovery path.
2. **Distill peers unlock Fold2** (+4.6e-5 solo) but need va-strict blend (M5) for sf/t.
3. **R1 is not required** for a gate-passing Slot5 residual — M5 clears alone. Prior V18.8 stage that needed `0.25·R1` is superseded for R1-free policy.
4. Runtime ~277s (dominated by 5×5-subfold `D_f`); functional budget exceeded the &lt;110s target — distillation quality confirmed; future runs can drop va-OOF subfolds after first verification.
