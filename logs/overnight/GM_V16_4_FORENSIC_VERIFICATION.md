# GM V16.4 — Forensic: OOF Indexing, sameer_glm10f, d1 Polish

**STRICT `NO_SUBMIT=True` · 0 pods**

---

## Step 1 — Fold-index / Family E paradox

- Q17 folds ≡ StratifiedKFold(42): **1.000000**
- global `spearmanr(L7_s2026_R3, L7F)` = **0.984251** (>0.99? **False**)
- raw `oof_f` → `oof_raw[fold_id==f]` scramble? **False**

| fold | R3 AUC | L7F AUC | Δ bps | ρ(R3,L7F) |
|-----:|-------:|--------:|------:|----------:|
| 0 | 0.947517 | 0.947545 | -0.27 | 0.984322 |
| 1 | 0.949662 | 0.949767 | -1.05 | 0.983919 |
| 2 | 0.954906 | 0.955166 | -2.61 | 0.984015 |
| 3 | 0.949395 | 0.949065 | +3.30 | 0.984221 |
| 4 | 0.945015 | 0.945676 | -6.61 | 0.984898 |

**Paradox resolved:** No row scrambling. Fold3 R3 **beats** L7F (+3.3 bps); Fold4 R3 **loses** (−6.6 bps). V16.3 Family E deltas vs *true* `z_ATH` (which already embeds L7F/LM/MS/G26) are **not** the same as R3−L7F solo gains — residual after orth can flip fold signs.

Bold recipe weight mass (approx sums): `{'L7F': -0.040956430057266076, 'LM': 0.12438825448094774, 'MS': 0.01824868469827448, 'G26': -0.012508511564926699}`

### Family E raw vs orth (true z_ATH @ w=0.01045)

| definition | pos/5 | meanΔ e-5 | folds e-5 |
|------------|------:|----------:|-----------|
| `raw_tail` | 3/5 | 0.447 | `[-0.318, 0.669, 0.547, -0.855, 2.19]` |
| `global_orth_V163` | 3/5 | 0.456 | `[-0.297, 0.669, 0.551, -0.82, 2.174]` |
| `lofo_orth_V15` | 3/5 | 0.459 | `[-0.301, 0.666, 0.56, -0.814, 2.183]` |

V15 used **LOFO** orth against `[z_ATH,d1]`; V16.3 used **global** fit-all orth then unitize — both shown above.

---

## Step 2 — `Q17_sameer_glm10f` forensic

- Solo OOF AUC folds: `[0.95081, 0.952096, 0.956372, 0.952066, 0.94668]` mean=0.951605
- `raw_disp_unlocked` = **0.9162**
- max\|β_oof−β_test\| = **0.6457** (need <0.05 → **False**)
- corr(orth_test, LIVE−zATH_t) = **0.0019** (need \|·\|<0.05 → **True**)
- corr(orth_test, bold−zATH_t) = **-0.0019**
- **sameer_verified = False**

| basis | β_oof | β_test | \|Δ\| |
|-------|------:|-------:|----:|
| `z_ATH` | 0.9081 | 0.2624 | 0.6457 |
| `d1` | -0.0062 | 0.0324 | 0.0386 |
| `Family_E` | -0.0227 | 0.0141 | 0.0368 |
| `MS` | 0.5980 | 0.8731 | 0.2751 |
| `G26` | -0.0554 | -0.1906 | 0.1351 |
| `LM` | -0.4212 | 0.0351 | 0.4563 |
| `L7F` | 0.2086 | 0.0351 | 0.1735 |
| `GPD` | 0.1098 | 0.1037 | 0.0062 |
| `GB` | 0.0802 | 0.0656 | 0.0146 |
| `MLP` | 0.3833 | 0.6274 | 0.2441 |

> **Do not live-submit sameer.** Double-count and/or β asymmetry failed — OOF gate was optimistic.

---

## Step 3 — Moves vs Slot1 OOF + staged NOSUBMIT

| move | pos/5 | meanΔ e-5 | min | folds | sf25 | t | CI90 | 1−ρ(Slot1) e-4 |
|------|------:|----------:|----:|-------|-----:|--:|------|---------------:|
| P1 d1+0.003 | 5/5 | 0.345 | 0.047 | `[0.591, 0.344, 0.164, 0.576, 0.047]` | 18/25 | 3.31 | [0.22,0.60] | 0.0132 |
| P2 E0.0125+d1+0.0035 | 5/5 | 0.377 | 0.109 | `[0.615, 0.419, 0.109, 0.377, 0.367]` | 16/25 | 3.10 | [0.27,0.83] | 0.0253 |

### Staged files

- `sub_v16_4_P1_Slot1_d1plus0p003_NOSUBMIT.csv` · move=P1 · 1−ρ(Slot1)=0.0132e-4 · pos=5/5 sf=18/25 t=3.31
- `sub_v16_4_P2_FamE0p0125_d1plus0p0035_NOSUBMIT.csv` · move=P2 · 1−ρ(Slot1)=0.0253e-4 · pos=5/5 sf=16/25 t=3.10

**NO submissions.** elapsed=181.3s