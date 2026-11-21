# GM V18.6 — Strict Fold-Isolated Peer Verify (Zero Cross-Fold Echo?)

**STRICT `NO_SUBMIT=True` · 0 GPU pods · ~91s**  
Script: `scripts/gm_v18_6_strict_fold_isolated_peer_check.py`  
Artifact: `/tmp/s6e11_v16/out/v18_6.json`

---

## Verdict

**DO NOT SUBMIT V18.5 P2/P4.**  
~**74% of the V18.5 OOF lift was cross-fold peer echo**. Strict `va`-only peers keep a real but smaller signal (~+5e-5, 5/5) that **fails sf≥19**.

| Check | Result |
|---|---|
| Sign | **Positive homophily** (`ρ(P2, peer_z_k2)=+0.75`) |
| Strict lift vs V18.5 | **P2 0.262×** · **P4 0.255×** (+19.0 → **+5.0**; +17.9 → **+4.6**) |
| Strict F2 | **Better** (0.014 / 0.013) |
| Strict elite sf | **FAIL** (sf15–18) |
| Submit? | **NO** |

---

## Step 1 — Leak-impossible peer pool

For fold `f`, `peer_z_strict[i∈va]` = leave-self-out mean of `unitize(z_S5_oof)` **only among `va`** (`m0=5`).  
Test peers = leave-self-out within **test only**.  
No `tr` peer ever enters a `va` feature → peers’ `z_S5_oof` were produced by models that **never trained on `y[va]`**.

| Key | mean cnt (va) | mean cnt (test) | ρ(strict, global) |
|---|---|---|---|
| k1 amount | 58 | 123 | +0.69 |
| k2 balance | 25 | 52 | +0.54 |
| k3 velocity | 1032 | 2229 | +0.99 |
| k8 cents×mer | 3.1 | 5.6 | +0.42 |

### Sign check (V18.5 global `peer_z_k2`)

| Model | ρ(model, peer_z_k2) |
|---|---|
| P1_peer_linear | **+0.792** |
| P2_peer_only | **+0.752** |
| P4_peer_clean | **+0.759** |

⇒ Trees add **`+peer_z`** (high-risk peers ⇒ higher residual risk), not a negative cancellation artifact.

---

## Step 2/3 — Strict scorecard vs V18.5

| Signal | F1 | F2 | F3 | ρ(V18.5) | z_S5 | twin | NNLS | sf | Stage |
|---|---|---|---|---|---|---|---|---|---|
| P1_strict | ✓ | ✗ 0.049 | ✗ | +0.53 | +4.7 | 4/5 | +7.9 | 18–21 | ✗ |
| **P2_strict** | ✓ | ✓ **0.014** | ✓ | +0.35 | **5/5 +4.98** | **+6.11** | **+6.18** | **15–18** | ✗ |
| **P4_strict** | ✓ | ✓ **0.013** | ✓ | +0.34 | **5/5 +4.56** | **+5.22** | **+5.41** | **18** | ✗ |

V18.5 reference: P2 +18.97 / +33.5 / +20.5 (sf24–25).

P2_strict folds on z_S5: `[+7.15, +7.19, +1.41, +7.26, +1.88]` — real, uneven, **not** the uniform +17e-5 wall from V18.5.

---

## Interpretation

1. **Echo mechanism confirmed:** Global `train∪test` peer pool let `va` rows borrow `z_S5_oof[tr]`, and those `tr` scores came from models trained **with** `y[va]` → label echo into peer means.  
2. **Physics is not zero:** Isolated peers still give ~+5e-5 / 5/5 / F2≪0.035 — positive homophily is real.  
3. **V18.5 NOSUBMIT files are unsafe for live Slot6** until rebuilt with fold-isolated (or nested) peer pools and re-cleared sf/t gates.

---

## Staged NOSUBMIT

**None.**

---

## Guardrails

| Check | Status |
|---|---|
| Kaggle submit | **NOT done** |
| Pods | **0** |
| ATH | **0.95394** |

---

## Bottom line

**Block submit.** V18.5 P2/P4 OOF was ~4× echo-inflated. Next peer work must use **fold-isolated / nested** peer statistics from the start.

---

*End GM V18.6*
