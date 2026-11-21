# GM V17.1 — Verify `L7_2seed − LM` + Combined Local Upgrade

**STRICT `NO_SUBMIT=True` · 0 GPU pods · no Kaggle submit**

JSON: `/tmp/s6e11_v16/out/v17_1.json`  
Prior report (printed in full in chat): `logs/overnight/GM_V17_0_FOLD0_DEBIAS_AND_G26_AUDIT.md`

Direction (NO `flip_to_plus_AUC`):
`u = unitize( 0.5·L(L7F) + 0.5·L(R3_s2026) − L(LM) )`

---

## Forensic symmetry (OOF vs Test)

| Metric | Value |
|---|---:|
| `corr_oof(u, z_ATH)` | −0.0697 |
| `corr_test(u, LIVE)` | −0.0575 |
| **\|Δcorr\|** | **0.0121** ✓ tight |
| `raw_disp = std(u_test)/std(u_oof)` | **0.8854** (below [0.93,1.07] — mild under-dispersion on Test) |

Symmetry of *correlation with the stack* is good; raw scale is slightly compressed on Test.

---

## `L7_2seed − LM` alone

### On `z_ATH` (0.9524301)

| w* | pos/5 | meanΔ (e-5) | min | [F0..F4] | sf25 | t | Wilcoxon p | 90% CI |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| **0.007** | **5/5** | **+0.694** | +0.037 | [1.13, 1.25, 0.57, 0.04, 0.49] | **13/25** | 1.12 | 0.237 | [−0.37, 1.77] |

### On `z_S1` (OOF Slot1 proxy = ATH + 0.01045·FE_fixed)

| w* | pos/5 | meanΔ | min | [F0..F4] | sf25 | t | W_p | 90% CI |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| **0.003** | **5/5** | +0.301 | +0.025 | [0.50, 0.51, 0.15, 0.03, 0.32] | **14/25** | 1.14 | 0.19 | [−0.37, 1.83] |

**Verdict alone:** confirms V17.0 **5/5** main-fold signal; soft bar still fails (sf≤14, t≈1.1, CI crosses 0).

---

## Combined: `z + w1·u + 0.0035·unitize(d1)`

### On `z_ATH`

| w1 | pos/5 | meanΔ | min | [F0..F4] | sf25 | t | W_p |
|---:|---:|---:|---:|---|---:|---:|---:|
| **0.003** | **5/5** | +0.677 | +0.220 | [1.21, 0.92, 0.45, 0.58, 0.22] | **14** | **2.10** | 0.095 |
| 0.005 | 5/5 | +0.850 | +0.461 | [1.46, 1.19, 0.60, 0.54, 0.46] | 14 | 1.83 | 0.115 |
| 0.007 | 5/5 | +1.012 | +0.531 | [1.76, 1.48, 0.64, 0.53, 0.66] | 14 | 1.81 | 0.121 |
| 0.010 | 5/5 | +1.165 | +0.442 | [1.99, 1.85, 0.75, 0.44, 0.79] | 13 | 1.46 | 0.163 |
| 0.015 | 5/5 | +1.436 | +0.118 | [2.51, 2.47, 1.00, 0.12, 1.09] | 13 | 1.30 | 0.205 |

Best by (pos, sf, t): **w1=0.003** — t improves to **2.10** (90% CI [0.12, 1.14] for mean_sf) but sf still **14/25**.

### On `z_S1` (stronger soft stats)

| w1 | pos/5 | meanΔ | min | folds | sf25 | t |
|---:|---:|---:|---:|---|---:|---:|
| **0.003** | **5/5** | +0.616 | +0.302 | [1.06, 0.78, 0.30, 0.61, 0.33] | **18/25** | **2.54** |
| 0.005 | 5/5 | +0.799 | +0.490 | [1.39, 1.00, 0.49, 0.57, 0.55] | 16 | 2.18 |
| 0.007 | 5/5 | +0.947 | +0.469 | … | 16 | 1.85 |

On the Slot1 OOF proxy, **w1=0.003 + d1@0.0035 clears sf≥18 and t≥2.20** with 5/5. (Still NOSUBMIT — live stack already has Slot1/d1 history.)

---

## Staged (`NO_SUBMIT=True`, on Slot3 LIVE CSV)

| File | Contents | 1−ρ(Slot3)×1e4 |
|---|---|---:|
| `sub_v17_1_L7_2seed_minus_LM_NOSUBMIT.csv` | Slot3 + 0.007·u | 0.085 |
| `sub_v17_1_L7_2seed_minus_LM_plus_d1_NOSUBMIT.csv` | Slot3 + 0.003·u + 0.0035·d1 | 0.039 |
| `sub_v17_1_comb_plus_G26f0_debias_NOSUBMIT.csv` | comb − 0.005·err_f0_consensus | 0.084 |

G26 f0 debias is **test-only** (consensus of unitized seed7/s2026 fold0 errors); **not OOF-gated** — diagnostic only.

---

## Verdicts

1. `L7_2seed−LM` is a real **5/5** direction with **tight OOF↔Test corr symmetry** (|Δ|=0.012); raw_disp 0.885 is mild soft fail.
2. Alone: sf/t too weak. With **+0.0035 d1**, on **z_S1** reaches **5/5 · 18/25 · t=2.54** at w1=0.003.
3. Three NOSUBMIT CSVs staged; **nothing submitted to Kaggle**.
