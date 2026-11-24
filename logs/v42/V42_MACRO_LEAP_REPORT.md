# V42 MACRO-LEAP Architect Report

**NO_SUBMIT=True** · Peephole micro-jumps **TERMINATED** · Spearhead frozen: Diag_A `f83c899d…` LB=0.95569

## 0. Freeze
- No further `< 0.30e-4` linear bank nudges.
- ID layout: **SEQUENTIAL (train/test id ranges separated → pooled u_g KS fails)** (switches=1)

## Engine 1 — SCAN4 within-cohort id
- Historical SCAN4 0.57489 = **SIZE_LEAK** (integer rank). Normalized `u_g` solo ≈ chance.
- Pooled raw `u_g` KS vs test: typically **fails** under sequential IDs → admitted **hash-rank** KS-safe surrogates (47 feats).
- LGB+offset on KS-safe id feats: Δe5=-12.20 pos=1/5 sf25=6

## Engine 2 — Cohort offset / temperature
- `merch_card_chan/add`: Δe5=0.00 pos=0/5 sf=2 t25=0.00
- `zr_zone/add`: Δe5=0.00 pos=0/5 sf=2 t25=0.00
- `zr_zone/tau`: Δe5=0.00 pos=0/5 sf=0 t25=0.00
- `merch_only/add`: Δe5=0.00 pos=0/5 sf=2 t25=0.00
- `card_chan/add`: Δe5=0.00 pos=0/5 sf=2 t25=0.00
- `merch_only/tau`: Δe5=-0.02 pos=1/5 sf=10 t25=-0.79
- `flag64/add`: Δe5=-0.20 pos=0/5 sf=10 t25=-1.69
- `merch_card_chan/tau`: Δe5=-0.96 pos=2/5 sf=10 t25=-0.88
- **Best:** `merch_card_chan/add` Δe5=**0.00**

## Engine 3 — StageB_v42 + Wilcoxon twilight
- Pillars: 39 (RealMLP×4, L7/LM7, gpt2, P4×3, harvest, stageB68, δ1, spreads, id-hash…)
- StageB_v42: Δe5=**-101.72** pos=0/5 sf=0 t25=-5.21
- Wilcoxon blend: Δe5=**-0.14** pos=1/5 sf=9 twilightρ=1.000
- Missing: ['gpt2_med_4ep: bad shape (2239872,)/(299984,)', 'TabICL*: canary-only (fold0)', 'qwen*: canary-only (fold0)', 'G26/MS: not found as named arrays on disk — using RealMLP+LM contrasts as epistemic spreads']…

## Macro scorecard (natural tip 1–15 e-4, NOT shrunk to 0.15)

| Engine | Δe5 | pos/5 | sf/25 | t25 | disp | KS | tip e4 | cos·S02↓ | cos·S06↓ | macro≥5 | loser≤0 | SHA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---|
| Engine1_id | -12.20 | 1/5 | 6 | -2.35 | 0.890 | 0.1139 | 5.00 | -0.063 | 0.044 | False | False | `220ba88d3dd1…` |
| Engine2_cohort | 0.00 | 0/5 | 2 | 0.00 | 1.000 | 0.0078 | 5.00 | -0.020 | 0.003 | False | False | `55dc8da71701…` |
| Engine3_StageB | -101.72 | 0/5 | 0 | -5.21 | 0.593 | 0.3872 | 5.00 | -0.004 | -0.064 | False | True | `1994a34bf7c9…` |
| Wilcoxon_blend | -0.14 | 1/5 | 9 | -1.79 | 0.752 | 0.0656 | 0.39 | -0.057 | 0.011 | False | False | `c3687e89647c…` |

Staged under `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/out/v42_macro_leap/`

## STOP
NO_SUBMIT=True. Awaiting explicit single-slot fire authorization for a macro-leap candidate.

## Executive verdict
- **No engine cleared the +5 e-5 macro-leap gate** on `z_pub_only + Zr`.
- Engine1 hash-id surrogates are KS-safe but **hurt** OOF (−12 e-5).
- Engine2 cohort EB offsets are **≈0** under capped γ (stronger uncapped versions destroyed AUC).
- Engine3 StageB collapsed (−102 e-5): gpt2 pillar failed shape load; residual hygiene KS/disp failed — **do not submit**.
- Frozen tip remains **Diag_A @ 0.95569**. True breakthrough needs missing G26/MS 5f arrays and/or a redesigned StageB residual (not raw re-stack of public RealMLP already inside `z_pub_only`).
