# Public notebook lever research vs ATH `0.95399`

**Date:** 2026-11-15 · local only · notebooks pulled under `/tmp/s6e11_nb_research/`

## Public LBs that beat our ATH

| Notebook | Author | Public LB | Δ vs ATH |
|---|---|---:|---:|
| Top 10 CPU Blend + EDA | sriram00777 | **0.95422** | **+23e-5** |
| LM Ladder Forecast the Board v7 | goodpjw2008 | **0.95417** | **+18e-5** |
| Shallow-Tree Stack | alexchilton | 0.9539 | −9e-5 |
| Final Ensemble (digit/BiLSTM/Optuna) | mdmuqtadirfuad | 0.95126 | far below |

## What they actually use (that we partly already have)

### 1) Sriram `0.95422` — highest public, simplest unused lever
Pinned dataset `sriram00777/s6e11-pinned-inputs-095422`:

```
rank_fraction( L(goodpjw_0.95417) + 0.15 * (L(RealMLP_PyTabKit_v5) − L(RealMLP_PyTorch_v28)) )
```

- Reference = goodpjw LM Ladder submission (not a new train).
- Lever = **RealMLP version disagreement** (yekenot PyTabKit onehot16 vs older PyTorch).
- Documented as public-feedback-informed; no independent CV claim.

**Local estimate on our ATH:** same Δmlp applied to Slot9

| w | 1−ρ vs ATH (e-4) | notes |
|---:|---:|---|
| 0.08 | 0.64 | inside historical Slot9-like ρ band |
| 0.10 | 1.00 | |
| 0.15 | 2.25 | same weight as sriram |
| 0.20 | 4.01 | larger step |

Sriram file itself is ~25e-4 away from our ATH (different lineage). ATH+Δmlp stays close to ATH while moving in the same *direction* as their neural correction.

### 2) goodpjw LM Ladder `0.95417`
- Starts from Aman Atar **0.95370** Q17 bundle + **70-member OOF bank** + compact LM seed7 + megayak LM + wide LM stack.
- Bold mix: large **+** on `megayak_shallow_tree` (~0.28), `daoviet_gpdvae`; **−** on several LMs (anti-LM overweight).
- Then Areeb-style extrapolations + forecast-guided steps → 0.95417.
- Datasets already mirrored locally under `external/banks/goodpjw2008_*` and Q17 npz.

### 3) Fuad BiLSTM / digit TE / Optuna (`0.95126`)
Useful diversity story (BiLSTM over TE digit/GPT-2 token keys, GBDT trio on original, Optuna FS) but **below ATH** — not a solo path past 0.95399.

### 4) Alex shallow-tree (`0.9539`)
Digit TE + orig20k lookup + published OOF bank — largely already in our lineage / P_super stack.

## Unused / underused vs *our* Slot9 ATH

| Lever | In ATH today? | Pass their score? |
|---|---|---|
| RealMLP PyTabKit − PyTorch residual | **No** (only single `yekenot_realmlp` stream in Bo9) | **Best near-term**: same recipe on ATH or on public 0.95417 |
| goodpjw ladder compounding / forecast steps | Partially (older LM ladder scores in history) | Already at 0.95417 publicly; we can *consume* the submission |
| EBM dig / diglead bank members | Partly via Q17/overnight vault | Secondary; V21 local 4-lev failed 6-gates |
| Fuad BiLSTM token features | **No** | Unlikely alone; optional diversity |
| Full GPT-2 + orig20k GReaT replica (masaya/Deotte) | **In flight** (MPS ~92%) | Real *new* base if fold0 ≥ 0.9505 |
| Digit-key TE / GLM-offset | Tried V21; small calibrated lifts | Not enough alone |

## Honest path ranking to clear `0.95417+`

1. **Consume public neural-delta:** `ATH + w·(L(new_mlp)−L(old_mlp))` with w≈0.08–0.15 (or apply to goodpjw 0.95417 reference). Lowest compute; mirrors proven +5e-5 public step.
2. **Rank-blend ATH toward sriram/goodpjw 0.95422/0.95417** at small w (keep unique E2_heavy_ridge signal).
3. **Finish MPS GPT-2+replica → fold0 canary** for a true unused generative base (gate ≥0.9505 vs masaya f0 0.94888).
4. Skip Fuad BiLSTM as primary ATH attack.

## Artifacts
- Notebooks: `/tmp/s6e11_nb_research/{lm_ladder_95417,sriram_95422,alex_9539}/`
- Sriram pinned CSVs: `/tmp/s6e11_nb_research/sriram_ds/`
- Local banks: `external/banks/goodpjw2008_s6e11-{oof-bank,fraud-series-artifacts,compact-lm-seed7}/`
