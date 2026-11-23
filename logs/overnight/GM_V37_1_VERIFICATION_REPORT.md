# GM V37.1 Verification Report

**UTC:** 2026-11-22T09:55:35+00:00  
**Verdict: 0 SUBMITS** — Gates G1 + G2 failed. All 5 daily slots held. Active cloud spend **$0.00/hr**.

ATH anchor: `submissions/sub_v35_S4_v10gate_StepO_0.30e4_on_areeb.csv` @ **0.95548** Public LB.

---

## 1. Cloud Hygiene Summary

| Item | Result |
|---|---|
| Pod `khx6qxz80yfenn` | **Already absent** (watchdog/prior terminate) |
| Sync | N/A (nothing to pull) |
| Remove verified | Yes — not in `runpodctl get pod` |
| Active USD/hr | **0.00** |
| Foreign pods | Untouched |

### Fold-0 GPU diagnostic (NEVER injected into live Test)

From `logs/overnight/v37_fold0/fold0_canary_result.json`:

| Canary | solo_f0 | ΔAUC_fb | R²_B40 | sub_pos | promote? |
|---|---:|---:|---:|---:|---|
| `tabpfn_bags` | 0.71246 | **−0.239** | 0.291 | 0/… | **NO** |

- Artifact `canary1_fold0_va_test.npy` shape `[439976]` (concat/partial — **not** a clean 299,984 Test vector).
- **Earn future 5-fold?** **No** for this TabPFN-bags canary (catastrophic solo + negative blend lift). Re-gate only after matched-bag TabICL/GPT-2 Fold-0 artifacts exist with `raw_disp≈1` and low `R²_B40`.
- **Strict 1-fold rule enforced:** zero Fold-0-only vectors entered the live Test blend.

---

## 2. Baseline & Pillar Scorecard

Baselines: `z_fullbank` OOF AUC **0.95262** · `z_ATH_twin` OOF AUC **0.95258**.

### Stage-2 hygiene (competence / leak gate)

| name | solo_oof | raw_disp | fold_diff_corr | R²_B40 | pass? |
|---|---:|---:|---:|---:|---|
| U_TriTabPFN | 0.94968 | 1.010 | 0.964 | 0.320 | ✅ |
| U_GAM_Probit_P3 | 0.94865 | 0.990 | 0.976 | 0.339 | ✅ |
| U_DeepNeural_P4 | 0.95150 | **1.120** | 0.990 | 0.479 | ❌ disp |
| U_BPE_GBDT_P5 | 0.95112 | **1.143** | 0.996 | 0.333 | ❌ disp |
| U_ShallowMarkov_P6 | 0.95124 | 1.017 | 0.987 | 0.571 | ✅ |
| U_GPT2_P1 | 0.95099 | **1.104** | 0.984 | 0.650 | ❌ disp |
| U_EBM_fam | 0.95048 | **1.076** | 0.994 | 0.282 | ❌ disp |
| U_TabM_fam | 0.94970 | 0.951 | 0.992 | 0.363 | ✅ |
| U_BPE_GBDT_fam | 0.95101 | 1.039 | 0.984 | 0.405 | ✅ |
| U_GAM_Probit_fam | 0.95043 | **1.079** | 0.993 | 0.415 | ❌ disp |
| L7_EB | 0.94927 | 1.032 | 0.983 | 0.566 | ✅ |
| L7_mean_diff | 0.94927 | 1.033 | 0.983 | 0.569 | ✅ |
| gpt2_med_4ep_5f | **0.94745** | 1.013 | 0.943 | 0.473 | ❌ solo&lt;0.9485 |
| gpt2_med_EB_plain | **0.94814** | 1.006 | 0.953 | 0.495 | ❌ solo&lt;0.9485 |
| z_deotte_lr_C0.03 | 0.95232 | **1.201** | 0.998 | 0.587 | ❌ disp (kept as residual contrast) |

Signed residuals admitted without solo hygiene: P4_harvest / L4.2 / L4.3(PCHIP) / L4.5 / v10_cb_d4 / v10_ridge / B_xgb_llr / JEWEL_u_* / `u_deotte_vs_fb` / `u_g26_5f_clean` (corr(tgt,û)=0.254).

### Stage-3 LOFO orth scorecard (Method-B GS + PCHIP margin lock)

| name | F2 | g_fb mean Δ | folds | g_tw mean Δ | folds_tw | same_sign | survivor? |
|---|---:|---:|---:|---:|---:|---|---|
| U_TriTabPFN | 0.0029 | −1.08e-06 | 2/5 | −7.89e-07 | — | False | |
| U_GAM_Probit_P3 | 0.0036 | −9.27e-07 | 2/5 | −6.20e-07 | — | True⁻ | |
| U_ShallowMarkov_P6 | 0.0119 | −9.24e-07 | 1/5 | −1.93e-06 | — | False | |
| U_TabM_fam | 0.0034 | −5.12e-07 | 2/5 | +3.58e-07 | — | True⁻ | |
| U_BPE_GBDT_fam | 0.0038 | −8.98e-07 | 1/5 | −8.31e-07 | — | False | |
| L7_EB | 0.0167 | −5.33e-07 | 2/5 | −3.62e-07 | — | True⁻ | |
| L7_mean_diff | 0.0172 | −4.65e-07 | 2/5 | −3.42e-07 | — | True⁻ | |
| **P4_harvest_xgb** | **0.0111** | **+1.48e-05** | **5/5** | **+6.59e-06** | **4/5** | **True** | **YES** |
| P4_L4.2_pure5f | 0.0067 | +1.83e-05 | 4/5 | +6.45e-06 | — | True | collinear |
| P4_L4.3_pure5f | 0.0054 | +6.46e-06 | 3/5 | +3.53e-06 | — | True | |
| P4_L4.5_pure5f | 0.0137 | −4.33e-06 | 1/5 | +1.46e-07 | — | True⁻ | |
| v10_cb_d4 | 0.0409 | −2.40e-07 | 1/5 | +3.18e-07 | — | False | |
| v10_ridge_resid | 0.0342 | −4.37e-07 | 2/5 | +6.71e-08 | — | True⁻ | |
| B_xgb_llr | 0.0327 | +1.95e-06 | 2/5 | +5.04e-06 | — | True | F2 |
| JEWEL_u_P4_L4.2 | 0.0081 | +1.50e-05 | 4/5 | +4.70e-06 | — | True | collinear w/ harvest |
| JEWEL_u_P4_L4.3 | 0.0073 | +2.52e-06 | 4/5 | +1.91e-06 | — | True | |
| JEWEL_u_P4_L4.5 | 0.0037 | −2.17e-06 | 0/5 | −5.66e-07 | — | True⁻ | |
| JEWEL_u_v10_cb_d4 | 0.0409 | −2.40e-07 | 1/5 | +3.18e-07 | — | False | |
| JEWEL_u_B_xgb_llr | 0.0358 | −3.43e-07 | 1/5 | +5.10e-07 | — | True⁻ | |
| u_deotte_vs_fb | 0.0065 | −2.73e-07 | 1/5 | +6.23e-07 | — | False | |
| u_g26_5f_clean | 0.0309 | +6.21e-06 | 4/5 | +4.06e-06 | — | True | F2 |

True⁻ = same sign but mean Δ ≤ 0 on at least one baseline (excluded).

**Sole collinearity survivor:** `P4_harvest_xgb` (`w* = −0.0216`).

---

## 3. Synthesizer & Twilight Attribution

- **Synth A (locked):** linear residual on `P4_harvest_xgb` only.  
  - `w_opt_lofo = −0.0216` · mean Δ_fb = **+1.44e-05** · mean Δ_tw = **+9.59e-06**
- **Synth B:** not selected (single low-DoF survivor; B requires ≥4/5 fold beat of A on both baselines — not applicable / not superior).
- **Twilight-decile ΔAUC (on `z_fullbank` ranks):**

| Band | q range | ΔAUC |
|---|---|---:|
| Bulk | q &lt; 0.80 | +5.15e-05 |
| Contested Twilight | [0.80, 0.985] | +7.63e-06 |
| Fraud Tail | q &gt; 0.985 | +6.41e-05 |

---

## 4. Live Verification Verdict (MAX 1 Submit Calibration)

Candidate built (NOSUBMIT): `submissions/sub_v37_1_SINGLE_BEST_0p85x.csv`

| Check | Value |
|---|---|
| SHA-256 | `cf78022a7a86135e0132f909235c5b0c69449b178165b6f64fd656f3906e9caf` |
| `1 − ρ(ATH)` | **5.94e-05** (within [0.20e-4, 15e-4]) |
| Predicted OOF ΔAUC @ c=0.85 | ≈ **+1.26e-05** |
| Observed Live Public LB | **N/A — not submitted** |
| New ATH? | **No** |
| Hedge (NOSUBMIT) | `submissions/sub_v37_1_PURE_5F_DEOTTE_CV_HEDGE_NOSUBMIT.csv` |

### Hard Gate Results

| Gate | Pass? | Key metrics |
|---|---|---|
| G1 fullbank | **FAIL** | 5/5 folds ✅ · min_fold&gt;0 ✅ · 19/25 sub ✅ · **t25=2.82 &lt; 3.0** · **Wilcoxon p=0.031 &gt; 0.005** |
| G2 ATH twin | **FAIL** | 4/5 ✅ · meanΔ&gt;0 ✅ · **t25=1.61 &lt; 2.2** |
| G3 dist/orth | PASS | disp=1.001 · KS=0.00022 · F2=0.013 |
| G4 anti-loser | PASS | |
| G5 step/dedup | PASS | step=5.94e-05 · min_hist=4.49e-05 |
| G6 quota | PASS | n_today post-00:00 UTC = **0** |

### Next action (remaining 4+ slots untouched — actually all 5 held)

1. **Do not submit** the 0.85× P4_harvest candidate — signal is real but under-powered vs hard Gate 1/2 (Wilcoxon + twin t25).
2. **Highest-leverage next spend (local $0 first):** matched-bag 5-fold TabICL / GPT-2 only after Fold-0 canary shows `promote=true` (current TabPFN-bags canary is a hard reject).
3. Re-fit Deotte stack after gpt2_med competence recovery (solo currently 0.947–0.948 &lt; 0.9485 floor) and/or lower competence floor only with vault R² evidence.
4. Keep `u_g26_5f_clean` as research contrast (F2=0.031 too high for Method-B admit).
5. Daily quota: **0 used today** — preserve for a gate-passing 5-fold TFM/LLM step only.
