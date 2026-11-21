# GM V17.6 — `G26−LM` Risk-Gated Audit + Fixed Non-Negative Slot3 Twin

**STRICT `NO_SUBMIT=True` · 0 GPU pods · local CPU**

Script: `scripts/gm_v17_6_verify_g26_lm_risk_gated.py`  
JSON: `/tmp/s6e11_v16/out/v17_6.json`

---

## Verdict — **Still no stage**

1. **Omitted-variable bug is real and fixed for Baseline B:** unconstrained Ridge on Bt9 alone flipped G26/d1 negative and dropped twin AUC to 0.95179. **NNLS + locked `+0.0355·d1 + 0.01045·FE`** recovers **OOF AUC 0.95227**, full R² vs `L(Slot3)` **0.976**, all free weights **≥ 0**.
2. **`R_G26_LM` / `R_G26_LM_z` stay forensic-clean and elite on `z_S3`** (5/5, ~+7.7 e-5, sf21, t≈3.3–3.9).
3. **Both still fail the corrected Slot3 twin (1/5)** and fail 1-fold G26 MC. The V17.5 invariance failure was **not** an artifact of the bad twin — it survives the fix.
4. **Nothing staged.**

---

## Step 1 — Baselines (fixed twin)

| Baseline | Construction | OOF AUC | Notes |
|---|---|---|---|
| **A `z_S3`** | `z_ATH + 0.01045·FE + 0.0055·d1` | **0.9524402** | reference |
| **B `z_NNLS_Slot3_twin`** | lock `0.0355·u(d1)+0.01045·u(FE)` + NNLS of residual on bank (Bo9\{d1,FE} ∪ Q17 extras ∪ 12 alex) | **0.952274** | R²_full **0.9758**; nnz 10/28 |
| **C `z_Simplex_LOFO`** | LOFO NNLS simplex on Bo9 vs binary y | **0.94500** | all w≥0 but **weak** (label-NNLS); diagnostic only |

### Baseline B weights (top / all nonzero)

| Stream | w |
|---|---|
| Bo9_MS | **0.840** |
| Q17_sadam_v6_gb_xgb | 0.404 |
| Bo9_LM | 0.179 |
| Q17_sadam_tabpfn_100k_tp3 | 0.154 |
| Bo9_GB | 0.146 |
| Q17_tabpfn_slimypunk_tp1 | 0.037 |
| Bo9_MLP / L7F | ~0.020 |
| Bo9_GPD / alex_* | ~0.003 |
| **Bo9_G26** | **0.000** (NNLS zeros it) |

**Read:** After locking the known **positive** d1/FE layers, Slot3’s residual is almost entirely **MS + tabular/Q17**, not an extra G26 load. V17.5’s `β̂(G26)=−0.168` was omitted-variable garbage; the honest twin simply **doesn’t need free G26** (MS already carries BPE mass).

Baseline C has positive G26 (0.006) and d1 (0.324) but AUC 0.945 — too weak to be a primary gate; reported for completeness. **Primary invariance bar = Baseline B.**

---

## Step 2 — Candidates @ w=0.035

### Forensic (vs Bo9/Bt9)

| Candidate | F1 disp | F2 leak | F3 LIVE_DC | ver |
|---|---|---|---|---|
| **R_G26_LM** `d·U(z_S3)` | 1.003 ✓ | 0.022 ✓ (d1) | +0.012 ✓ | **✓** |
| **R_G26_LM_z** `d·unitize(z_S3)` | 0.997 ✓ | 0.017 ✓ (d1) | +0.005 ✓ | **✓** |
| R_G26_LM_Uc `d·unitize(U−0.5)` | 0.994 ✓ | **0.046** ✗ | +0.010 ✓ | ✗ |
| R_70G26_30MS_LM | 1.010 ✓ | 0.017 ✓ | **−0.052** ✗ | ✗ |
| R_G1e_all | 1.006 ✓ | **0.053** ✗ (FE) | −0.034 ✓ | ✗ |

`U` vs `unitize` (R_G26_LM vs R_G26_LM_z): nearly identical OOF strength on `z_S3` → the lift is **not** “0.5·d linear smuggled via U”; both the level gate and zero-mean gate work. Pure centered interaction `Uc` **fails F2**.

### Multi-baseline scorecard

| Candidate | Base A `z_S3` | Base B `NNLS twin` | Base C `Simplex` |
|---|---|---|---|
| **R_G26_LM** | **5/5** +7.70 min+0.94 sf21 t=3.31 | **1/5** −5.10 min−10.5 sf10 | 2/5 −1.02 |
| **R_G26_LM_z** | **5/5** +7.68 min**+2.96** sf21 t=**3.85** | **1/5** −3.97 min−7.2 sf10 | 2/5 +2.15 |
| R_70G26_30MS_LM | 5/5 +8.77 min+0.35 sf20 t=3.47 | 1/5 −6.32 | 1/5 |
| R_G1e_all | 4/5 +7.14 | 1/5 −4.57 | 1/5 |

**Invariant on corrected Slot3 twin? No.** Same qualitative kill as V17.5: huge F0 help, deep holes on other folds once the stack matches Slot3’s true non-negative geometry (MS-heavy, G26 free-weight ≈0).

### 1-fold G26 MC (`ε ~ N(0, 0.0727²)` on OOF G26; 8 draws)

| Signal | frac S3 (5/5 ∧ sf≥20) | frac all 3 bases |
|---|---|---|
| R_G26_LM | **0.00** | **0.00** |
| R_70G26_30MS_LM | 0.25 | **0.00** |

Even on `z_S3` alone, injecting V17.2-calibrated 1-fold noise **destroys** the 5/5∧sf20 bar (typical 3–4/5). Blending 30% MS does **not** rescue all-base MC.

---

## Step 3 — Staging

| Rule | Result |
|---|---|
| F1∧F2∧F3 ∧ 5/5∧sf≥20∧t≥2.5∧min>0 on **A,B,C** ∧ MC all-bases ≥75% | **No winners** |
| `submissions/sub_v17_6_*` | **none** |
| Kaggle submit / pods | **none / 0** |

---

## Implications

- V17.5’s bad twin **was** biased — but fixing it **does not unlock** `G26−LM` risk-gating for Slot 4.
- Elite `z_S3`-only bars (23/25 in V17.5, 21/25 here) are **recipe-specific**, fragile to (i) Slot3-matched weights and (ii) realistic G26_f0 noise.
- Next leverage is still **outside** this residual family (true 5-fold G26 test, or a new stream), not another risk-gate of `G26−LM` on `z_S3`.

---

*End GM V17.6*
