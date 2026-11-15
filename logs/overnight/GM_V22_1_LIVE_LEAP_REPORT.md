# GM V22.1 — Detailed Live Leap Report

**Date:** 2026-11-15 · local Mac CPU/MPS only · 0 RunPod  
**Prior personal ATH:** `0.95399` (Slot9)  
**Public target:** `sriram_0.95422` / `goodpjw_0.95417`

---

## Status snapshot

| Track | Status | Result |
|---|---|---|
| Public NB lever research | **Done** | `GM_V22_PUBLIC_NB_LEVER_RESEARCH.md` |
| V21 local 4-leverage | **Done** | 0/4 passed 6-gates · no submit |
| V22.1 C1 (ATH+ΔRealMLP) | **Done** | **LB 0.95407** |
| V22.1b C2pure (sriram+prop) | **Done** | **LB 0.95416** (new personal ATH) |
| MPS GPT-2 + orig20k replica | **Not finished** | ~49950/50000 · ETA ~1 min · still alive |

Gap to public #1: **−6e-5** (`0.95416` vs `0.95422`).

---

## 1. What was done

### 1.1 Public notebook audit
Pulled/inspected under `/tmp/s6e11_nb_research/`:
- `sriram_95422` + pinned DS (`reference.csv` = goodpjw 0.95417, `new_neural` / `old_neural`)
- `lm_ladder_95417` (goodpjw)
- `yekenot_pytabkit` / `yekenot_pytorch` **OOF** (`oof_preds.csv` on 699,960 rows)
- Local bank: `external/banks/goodpjw2008_s6e11-oof-bank`

Exact sriram recipe confirmed:
`L(goodpjw_0.95417) + 0.15·(L(RealMLP_PyTabKit_v5) − L(RealMLP_PyTorch_v28))` → **0.95422**.

### 1.2 V21 (dead end for submit)
Four local leverages (Bayes-LM, vault booster, ordered-CTR Cat, minimax portfolio) — **none** cleared 6-gates. No submit.

### 1.3 V22.1 script
`scripts/gm_v22_1_sriram_goodpjw_synthesis_and_live_leap.py`
- Rebuild Slot8/9 anchors via V20 factory
- ΔRealMLP OOF+test from yekenot kernel OOFs (bank RealMLP ≠ exact PyTabKit Δ)
- Distance matrix + R² decomp of `L(sriram)−L(Slot9)`
- Fail-ray checks vs Slot6/Slot7
- Candidates C1 / C2 / C3 → staged CSVs → live submits

### 1.4 Live submissions (today)
| Ref | Candidate | Public LB | Δ vs old ATH |
|---|---|---:|---:|
| 58386493 | C1 `ATH + 0.10·orth(ΔRealMLP)` | **0.95407** | +8e-5 |
| 58387086 | C2pure `sriram + 0.05·orth(prop)` | **0.95416** | +17e-5 |

8 daily slots remaining after C2pure.

---

## 2. What was found

### 2.1 Geometry (test, 299,984 rows) — `1−ρ` × 1e-4

|  | Slot9 | Slot9+d08 | Slot9+d12 | goodpjw | sriram |
|---|---:|---:|---:|---:|---:|
| **Slot9** | 0 | 10.15 | 22.77 | 22.75 | **25.41** |
| Slot9+d08 | 10.15 | 0 | 2.52 | 32.02 | 25.13 |
| goodpjw | 22.75 | 32.02 | 44.17 | 0 | 2.26 |
| sriram | 25.41 | 25.13 | 32.53 | 2.26 | 0 |

- Sriram is a **different manifold** from Slot9 (`~25e-4`), not a small nudge.
- ATH+Δmlp alone cannot close that gap (C1 live only +8e-5).

### 2.2 Fail rays (Slot6 / Slot7)
- `sriram − Slot9`: c6=`−0.112`, c7=`+0.047` → **not** fail-ray aligned
- `ΔRealMLP`: c6=`−0.042`, c7=`−0.024` → clean

### 2.3 Orthogonal decomposition of `L(sriram)−L(Slot9)`

| Basis block | R² |
|---|---:|
| Bo9 reweight | 0.139 |
| ΔRealMLP alone | 0.087 |
| Prop (R2+Cat+U5+E2) | 0.051 |
| Bo9 + Δmlp | 0.243 |
| **Full** | **0.295** |

Full-model prop coefs: **R2/Cat/U5/E2 all negative** (`−0.010…−0.016`); dMLP coef `+0.045`.  
→ Sriram **does not carry** our proprietary Slot5/8/9 engines (inject thesis confirmed).  
→ ~70% of the sriram−Slot9 residual is still **unexplained ladder ε** (goodpjw compounding).

### 2.4 ΔRealMLP on Slot9 OOF (orth)
All weights **hurt** z_S9 OOF (w=0.10 → 1/5, meanΔ=`−10e-5`).  
Raw (non-orth) w=0.08 was milder (3/5, `+3e-5`).  
**Live C1 still scored 0.95407** — OOF/LB mismatch; useful but not the leap path.

ΔRealMLP OOF AUCs: PyTabKit `0.950084` vs PyTorch `0.950016` (tiny absolute gap; ranking residual still LB-valuable).

### 2.5 goodpjw bank members vs z_S9 (orth w=0.012)
All sampled members **5/5** positive, strongest:
- `gbn_bag5_w512` meanΔ `+1.34e-5`, t=4.07
- `gb_bag5` `+1.22`, t=4.43
- `glm_te` / `realmlp_te` ~`+1.0`

Secondary levers for a later bank hill-climb (not used in today’s submits).

### 2.6 Candidates

**C1** — ATH + Δmlp: live **0.95407**. Leaves ~15e-5 on the table vs 0.95422.

**C2** — first run used `wp∈{0.25…0.70}` → OOF destroyed (scale bug).  
Fixed to live-engine scale `wp∈{0.02…0.08}`:
- `C2pure wp=0.05`: ρ vs sriram `4.37e-4`, vsSriProx **5/5** meanΔ `+17.5e-5`, c7=`+0.046`
- Live **0.95416** = **+9e-5** over C1, still **−6e-5** vs sriram

**C3** geodesic blends (not submitted yet): λ=0.25/0.35/0.50 with ρs=`2.0 / 4.0 / 8.2` e-4 — next natural probe if continuing.

### 2.7 Strategic reading
1. Consuming the public manifold (`sriram`) + small proprietary inject beats pure-lineage Δmlp on ATH.
2. Prop inject at wp=0.05 moved **toward** but did not surpass 0.95422 — either under-weight, or prop is only partly orthogonal on the public ladder in LB space.
3. Remaining gap is mostly **goodpjw ladder ε** (R² unexplained ~0.70), not Δmlp.
4. MPS GPT-2+orig20k replica is the only **new trained base** still in flight; fold0 canary gate ≥0.9505 still pending.

---

## 3. Staged artifacts

- `submissions/sub_v22_1_C1_ATH095399_plus_dRealMLP_w0p10.csv`
- `submissions/sub_v22_1_C2_Sriram095422_plus_Proprietary_Slot9_Engines.csv` (fixed C2pure wp=0.05)
- `submissions/sub_v22_1_C3_Geodesic_Sriram095422_x_ATH095399_dMLP.csv`
- `submissions/sub_v22_1_LIVE_leap.csv` → **0.95407**
- `submissions/sub_v22_1b_LIVE_leap.csv` → **0.95416**
- Script: `scripts/gm_v22_1_sriram_goodpjw_synthesis_and_live_leap.py`
- Logs: `GM_V22_1_LIVE_LEAP.log`, `GM_V22_1B_LIVE_LEAP.log`

---

## 4. MPS replica (not finished)

- Target: 50,000 steps · GPT-2 Storyteller + orig20k GReaT-style replica on MPS
- Progress at report time: **~49950/50000** (ep ~19.98, loss ~0.606), process still alive
- Last disk ckpt seen: **48000** (SAVE_STEPS=500; final 50000 save expected at end)
- Trainer lives on pics volume via symlink (ENOSPC workaround)
- After finish: fold0 canary vs masaya f0 `0.94888` (gate ≥ `0.9505`) before any stack use
