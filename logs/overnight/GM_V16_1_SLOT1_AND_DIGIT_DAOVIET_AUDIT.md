# GM V16.1 — Slot1 Family E Submit + Digits / daoviet Audit

**Public LB Slot1: `0.95389` (+1.0 bps vs LIVE `0.95388`)** · 0 pods · 2026-11-14

---

## Step 1 — Live Slot 1 (`Family E` @ `0.20e-4`)

| Item | Value |
|------|------:|
| Source | `sub_v16_0_FamilyE_0p20e4_NOSUBMIT.csv` |
| Live file | `submissions/sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv` |
| Rows / NaNs | 299984 / 0 |
| `1−ρ(LIVE_095388)` | **2.000000e-05** (target 0.20e-4) |
| Gate0 | 5/5 main · **19/25** sf · **t=3.14** |
| Kaggle ref | `58331440` · Complete @ 2026-11-14 04:34:04Z |
| **Public score** | **`0.95389`** |
| Δ vs LIVE | **+1.0 bps** |

Message: `v16.1 Slot1: ATH 0.95388 + 5-fold L7_s2026 (K=10) Family E tail-weighted 2-seed swap (1-rho=0.20e-4, 5/5 main, 19/25 sf, t=3.14)`.

**Read:** Genuine micro-lift (not the V15 ≈LIVE bug). Small public gain; private survival TBD. Next probes should stay ≤0.35e-4 with stop-loss if ≤0.95388.

---

## Step 2 — daoviet 16:52 CSV decomposition on `Bt_full`

Basis: **28 streams** including `LIVE_095388`, `bold_095386`, `anchor_095370`, `cand2`, `L7F`, `L7_s2026_5f`, `G26_f0`, `d1_v4`, `MS`/`Psuper_v3`, alexchilton×12, Q17 amanatar CSVs.

### `submission_exact_095388.csv`

| metric | value |
|--------|------:|
| `1−ρ(LIVE)` | **1.748e-4** |
| OLS R² | **0.999947** |
| Ridge(α=1) R² | **0.999947** |
| resid std | ≈0 (essentially spanned) |

**Top-8 OLS weights:**

| stream | weight |
|--------|-------:|
| `bold_095386` | **+1.017** |
| `LIVE_095388` | +0.031 |
| `Q17_sub_unbundled_newton` | +0.025 |
| `Psuper_v3` | −0.020 |
| `cand2` | −0.018 |
| `alex_da228571` | −0.016 |
| `anchor_095370` | −0.016 |
| `alex_e20ba93c` | −0.015 |

**Why 1.75e-4 from LIVE?** It is almost a **bold_095386-shelf** file (`β≈1.02` on bold) with tiny LIVE/Q17/alex corrections — not a new orthogonal GP/DVAE direction. Distance from LIVE matches a small rank-step off the 0.95386→0.95388 family, not a leak residual.

### `submission_orthogonal_apex_095370plus.csv`

| metric | value |
|--------|------:|
| `1−ρ(LIVE)` | **8.725e-4** |
| OLS R² | **0.998646** |
| Ridge R² | **0.998646** |

**Top-8:** `cand2` (+0.649), `bold` (+0.441), `LIVE` (−0.297), `Psuper_v3` (+0.226), Q17 unbundled (−0.155), `MS` (−0.152), `anchor_095370` (+0.106), alex…

R² < 0.999 but still ~99.86% spanned — residual is small public-blend noise, not a mystery bank.

### OOF reconstruction (`0.5·z_ATH + 0.5·dao_exact_recon`)

OLS on matched OOF/TEST pairs (bold+LIVE OOF proxied by `z_ATH` — no public bold OOF):

| metric | value |
|--------|------:|
| R² (matched streams) | 0.99993 |
| 5-fold mean AUC | 0.94944 |
| lift vs `z_ATH` | **−1.17 bps** |
| sf25 vs `z_ATH` | **0/25** |

**Do not blend daoviet_exact into ATH** — it is a near-duplicate of the bold/LIVE shelf and hurts OOF under this reconstruction.

---

## Step 3 — Digits / exact-value TE vs quantile-binned `z_ATH`

### Kernel FE (downloaded under `external/nov14_recon/kernels/`)

**digits-leak (`dhananjayk2609/…part-2-digits-leak…`):**

- Keys: `v_<col>` exact value, `cents_` (`round(x*100)%100`), `int_`, `ten_`, `dec_`, leading `l1_`/`l2_`.
- OOF TE via bincount + smoothing `m≈20`; model matrix = raw + `te_*` + `cnt_*`.
- Shallow **4-leaf LightGBM** / depth-2 XGB; claims classic plateau ~0.947 → digit TE ~0.950.
- Cents χ² plots on `amount_usd`, `account_balance_usd`, `distance_from_home_km`, `hours_since_last_txn`.

**daoviet journey:** public ladder / file-discovery notebook paired with the 16:52 dataset (exact_095388 + orthogonal_apex) — blend narrative, not a new digit recipe.

**asterios:** public = 20% noise / winner’s curse / nested CV — no digit FE.

### Local fast audit (5-fold, train-only TE; 7 continuous cols → 48 keys)

`z_ATH` OOF proxy = `L(L7F)+0.03·d1` → AUC **≈0.9495** (fold mean).

| smoothing | TE OOF AUC | ρ(TE, z_ATH) | resid AUC (⊥ z_ATH,d1) | blend lift (bps) | all-5 lift? |
|-----------|----------:|-------------:|-----------------------:|-----------------:|:-----------:|
| m=10 | 0.80846 | 0.606 | (in JSON) | **−5.17** | **False** |
| m=30 | 0.81554 | 0.631 | (in JSON) | **−5.32** | **False** |

Top solo keys (m=30): `int_merchant_risk_score` (0.755), `lead2_merchant_risk_score` (0.752), `v_merchant_risk_score` (0.749), `int_velocity_score` (0.713), `v_velocity_score` (0.706).

**Interpretation:** Exact/digit TE alone carries real univariate signal (especially risk/velocity integers), but after LOFO-orth to `[z_ATH, d1]` the residual **does not** lift 5/5 folds — quantile-binned ATH stack already absorbs most usable label leakage from these keys under this aggregation. A fuller dhananjay-style shallow tree on all keys (with orig20k in TE stats) may still add edge; the mean-TE probe here is **negative** for a quick Stage-B inject.

---

## Verdict

1. **Slot1 Family E @ 0.20e-4 → `0.95389` (+1 bps)** — first verified post-`unitize` live lift.
2. **daoviet exact_095388** ≈ **bold-dominated** public file (R²≈1.000); not new OOF bank.
3. **Digit/exact TE** signal exists but **not orthogonal enough** to `z_ATH` in this fast mean-TE test to green-light a live digit inject.

## Artifacts

```
submissions/sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv
scripts/gm_v16_1_submit_famE_and_audit_digits_daoviet.py
external/nov14_recon/daoviet_apex_1652/
external/nov14_recon/kernels/{digits,daoviet,asterios}/
logs/overnight/v16_1_audit.json
logs/overnight/GM_V16_1_SLOT1_AND_DIGIT_DAOVIET_AUDIT.md
```
