# GM V17.3 — Cross-Pillar Non-Linear Gating & MoE Audit

**STRICT `NO_SUBMIT=True` · 0 GPU pods · local CPU**

Script: `scripts/gm_v17_3_cross_pillar_gating_moe_audit.py`  
JSON: `/tmp/s6e11_v16/out/v17_3.json`  
Base: `z_S3` AUC **0.9524402** · `frozen_lofo_orth(..., Bo9/Bt9, ridge_α=10)` · stage rule: **F1∧F2∧F3 ∧ 5/5 ∧ sf≥18 ∧ t≥2.20**

---

## Verdict

**No candidate staged.** Every gate that clears the OOF strength bar (`5/5`, `sf≥18`, `t≥2.2`) **fails forensic F2 (basis leak) and/or F3 (LIVE residual corr)**. Every gate that is forensically clean is **too weak** on subfolds (`sf≤17`, `t≤1.2` for singles).

Cross-pillar MoE / second-order interactions do **not** unlock a clean lift on `z_S3` under the V16.8 forensic contract. Pillars are already **ρ≈0.98** with each other — nonlinear functions of near-collinear streams mostly re-express linear basis mass that `frozen_lofo_orth` either removes (→ weak residual) or fails to remove evenly OOF↔Test (→ F2/F3 fail).

---

## Step 1 — Pillars

| Pillar | Definition (after `L_fold` OOF / `L` Test, then `unitize`) |
|---|---|
| **P_BPE** | `unitize(0.55·z_MS + 0.45·z_G26)` — BPE GPT-2, L≈224 |
| **P_QLM** | `unitize(0.45·z_LM + 0.30·z_L7F + 0.25·z_R3_s2026)` — quantile LM, T=72 |
| **P_TAB** | `unitize(0.50·z_GB + 0.35·z_MLP + 0.15·z_GPD)` |
| **LL_marginal** | `unitize(L_fold(mean_k(ll_neg)))` OOF; `unitize(L(mean_f L(mean_k ll_neg_test_f)))` Test |
| **U5_gini** | `unitize(L_fold(U5_gini10_llr))` / Test analogue (V16.9 forensic-clean family) |

**OOF pillar correlations:**

| Pair | ρ |
|---|---|
| BPE–QLM | **0.986** |
| BPE–TAB | **0.987** |
| QLM–TAB | **0.977** |
| BPE–LL_marginal | −0.373 |
| QLM–U5 | −0.071 |

Near-perfect pillar agreement explains why asymmetric / product gates either (a) collapse to near-null after orth, or (b) retain basis leakage.

---

## Step 2–3 — Scorecard (singles + controls)

Gates after `frozen_lofo_orth` vs `Bo9/Bt9`; screen orients to `+` AUC on fold0; `w*` from `{0.006…0.04}`.

| Signal | F1 disp | F2 leak | F3 LIVE_DC | ver | pos/5 | w* | meanΔ e-5 | min | folds [F0..F4] | sf25 | t | W_p | 90% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **G1** BPE−QLM asym | 1.028 ✓ | **0.067** ✗ | **−0.089** ✗ | ✗ | **5/5** | 0.035 | **+3.08** | +0.08 | [8.88, 0.08, 0.53, 2.18, 3.71] | **18** | **2.87** | 0.003 | [0.46, 1.81] |
| **G2** Gen vs Tab² | 0.985 ✓ | 0.018 ✓ | +0.034 ✓ | ✓ | 3/5 | 0.035 | +0.48 | −2.13 | [5.15, −2.13, −2.06, 1.06, 0.35] | 14 | 0.95 | 0.19 | [−0.21, 0.73] |
| **G3** BPE·TAB | 1.049 ✓ | **0.220** ✗ | **+0.096** ✗ | ✗ | **5/5** | 0.040 | +1.26 | +0.79 | [1.70, 1.07, 0.79, 1.67, 1.07] | **18** | **2.48** | 0.007 | [0.10, 0.56] |
| **G3b** min−max | 0.993 ✓ | **0.044** ✗ | +0.028 ✓ | ✗ | 3/5 | 0.025 | +1.10 | −2.02 | [5.86, −2.02, −1.67, 1.90, 1.41] | 15 | 1.19 | 0.12 | [−0.26, 1.42] |
| **G4** LL·(BPE−TAB) | 1.001 ✓ | 0.007 ✓ | **−0.074** ✗ | ✗ | 4/5 | 0.030 | +1.79 | −3.89 | [1.90, 0.11, 2.53, −3.89, 8.28] | 16 | 1.81 | 0.063 | [0.05, 1.67] |
| **G5** U5·(QLM−BPE) | 0.990 ✓ | 0.021 ✓ | +0.027 ✓ | ✓ | **5/5** | 0.015 | +0.40 | +0.02 | [0.20, 0.19, 1.34, 0.26, 0.02] | 17 | 0.83 | 0.19 | [−0.41, 1.17] |
| **G6** z·\|z\| | **1.214** ✗ | **0.374** ✗ | **+0.393** ✗ | ✗ | 5/5 | 0.012 | +0.59 | +0.02 | [0.50, 0.41, 0.41, 1.59, 0.02] | 17 | 2.83 | 0.009 | [0.22, 0.88] |
| **G6b** signed Δ²·U(z) | 0.978 ✓ | 0.035 ✓ | **+0.134** ✗ | ✗ | 5/5 | 0.012 | +1.30 | +0.08 | [0.74, 0.08, 1.56, 1.09, 3.01] | 17 | 1.58 | 0.057 | [−0.09, 2.10] |
| **G7** d1·BPE | **1.122** ✗ | **0.313** ✗ | **−0.139** ✗ | ✗ | 2/5 | 0.006 | +0.06 | −0.48 | [0.31, −0.13, −0.09, 0.67, −0.48] | 12 | 0.27 | 0.58 | [−0.49, 0.68] |
| **G8** Ridge α=5k ×15 prod | 0.981 ✓ | **0.280** ✗ | **+0.060** ✗ | ✗ | 4/5 | 0.035 | +0.86 | −0.48 | [2.20, 0.06, 0.29, −0.48, 2.23] | 17 | 1.64 | 0.060 | [−0.01, 0.51] |
| G8b Logit L2 ×15 | 0.981 ✓ | **0.270** ✗ | **+0.071** ✗ | ✗ | 3/5 | 0.040 | +0.87 | −0.61 | [2.23, −0.61, 0.34, −0.20, 2.59] | 15 | 1.47 | 0.11 | [−0.03, 0.45] |
| G8c Ridge ×30 | 0.984 ✓ | **0.252** ✗ | **+0.063** ✗ | ✗ | 4/5 | 0.012 | +0.37 | −0.11 | [0.60, 0.30, 0.01, −0.11, 1.06] | 15 | 1.14 | 0.23 | [−0.11, 0.52] |
| **U5_gini** ctrl | 0.998 ✓ | 0.025 ✓ | −0.046 ✓ | ✓ | **5/5** | 0.009 | +0.73 | +0.08 | [1.43, 0.75, 0.41, 0.98, 0.08] | 15 | 1.16 | 0.18 | [−0.29, 1.50] |
| **U1 q80−q20** ctrl | 0.990 ✓ | **0.043** ✗ | −0.026 ✓ | ✗ | 5/5 | 0.009 | +0.82 | +0.14 | [1.05, 1.31, 0.14, 0.59, 1.00] | 15 | 1.31 | 0.15 | [−0.23, 1.71] |

### Forensic-clean set (only 3)

| Signal | Strength | Why not staged |
|---|---|---|
| G2 | 3/5, sf14, t=0.95 | Fails pos/5 |
| **G5** | 5/5 but mean only +0.40 e-5 | sf17 / t=0.83 — soft |
| **U5 ctrl** | 5/5, +0.73 e-5 | sf15 / t=1.16 — soft (matches V16.9 saturation) |

### OOF-strong but forensic-fail (false friends)

| Signal | OOF | Forensic death |
|---|---|---|
| **G1** (digit-spike asym) | Best raw lift: +3.1 e-5, 5/5, sf18, t=2.87 | leak 0.067, LIVE_DC −0.089 |
| **G3** (BPE·TAB) | 5/5, sf18, t=2.48 | leak **0.220**, LIVE_DC +0.096 |
| G6 / G6b / G7 / G8* | mixed 4–5/5 | systematic F2/F3 (G6 = monotonic of z_S3 → extreme leak) |

**G1 is the headline near-miss:** physically motivated (BPE exact-digit spike vs quantile-blind QLM), strongest OOF screen, but **not transportable** under F2/F3 — residual still correlates with Bo9 columns and LIVE residual.

---

## Combos (top-6 pairwise, re-orth)

No combo cleared **all** gates. Closest:

| Combo | ver | pos | mean | sf | t | blocker |
|---|---|---|---|---|---|---|
| G5 + U1 | ✓ | 5/5 | +1.83 | **16** | **1.96** | sf/t (need 18 / 2.20) |
| G1 + U1 | ✗ | 5/5 | +3.05 | 18 | 2.84 | F2 leak 0.044 (F3 ok) |
| G1 + G3 | ✗ | 5/5 | +3.56 | 18 | 3.20 | F2/F3 |
| G1 + G6 | ✗ | 5/5 | +3.58 | **20** | 3.38 | F2/F3 (G6 poison) |

---

## Physical interpretation

1. **Pillars are not independent experts** (ρ≳0.98). A mixture-of-experts story needs disagreement; here BPE/QLM/TAB already agree almost everywhere, so MoE gates have little regime to exploit.
2. **G1’s OOF signal is real but contaminated** — asymmetric BPE−QLM² still leaks into the 9-basis / LIVE residual after LOFO Ridge orth → fails the same forensic filters that killed V16.6–V16.9 soft add-ons.
3. **Rarity / uncertainty gating (G4, G5)** is forensically safer (G5 passes F1–F3) but effect size is **≪ 1 e-5 mean** at `w*` — consistent with U5 being a saturated, clean, weak control.
4. **Degree-2 poly Ridge (G8)** overfits main-effect products of near-collinear pillars → huge F2 leak (0.25–0.28) even at α=5000.
5. **Tail curvature G6** is essentially a nonlinear transform of `z_S3` itself → expected forensic disaster (disp 1.21, leak 0.37).

---

## Staging

| Check | Result |
|---|---|
| Winners | **[]** |
| `submissions/sub_v17_3_*_NOSUBMIT.csv` | **none written** |
| Kaggle submit | **none** |
| GPU pods | **0** |

---

## Implication for the overnight stack

Linear exhaustion (V16.8–V17.2) **extends to these cross-pillar second-order gates** under the forensic contract. Remaining leverage is **not** another orth’d interaction of `{MS,G26,LM,L7,TAB}` on `z_S3`, but:

1. Completing **true 5-fold G26 test** (V17.2: sub-tick expected, fidelity), and/or  
2. **New information** outside the 9-basis / 3-pillar span (new model family, new features, private residual DNA — not MoE of the same three pillars).

---

*End GM V17.3*
