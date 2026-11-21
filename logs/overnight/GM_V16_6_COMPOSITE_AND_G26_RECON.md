# GM V16.6 — Composite of 3×`5/5` Signals + G26 Architecture Recon

**STRICT `NO_SUBMIT=True` · 0 GPU pods · no Kaggle submit**

Artifacts: `/tmp/s6e11_v16/out/v16_6.json`, `/tmp/s6e11_v16/out/v16_6_g26_extra.json`,  
`submissions/sub_v16_6_composite_5of5_NOSUBMIT.csv`

Base: true `z_S3 = z_S1 + 0.0055·unit(d1)` on the 0.9524301 ATH stack  
(`AUC z_S1=0.9524347`, `AUC z_S3=0.9524402`). LOFO basis (10):  
`z_ATH, d1, Family_E, MS, G26, LM, L7F, GPD, GB, MLP`.

---

## Step 1 — Forensic + Composite on `z_S3`

### 1.1 Symmetry & double-count forensic (V16.4 gate)

| Signal | `raw_disp` = σ(orth_test)/σ(orth_oof) | max‖β_oof−β_test‖ | ‖corr(orth_t, LIVE resid)‖ | Verdict |
|---|---:|---:|---:|---|
| `Q17_sadam_v4_cat` | **0.8425** (need [0.93,1.07]) | **0.680** (need <0.08) | 0.0005 / bold 0.002 | **FAIL** disp+sym |
| `q80−q20` (L7_s2026 GPU) | 0.9959 ✓ | **0.316** | 0.0034 / bold 0.002 | **FAIL** sym |
| `R2−R3` curv | **0.8747** | **0.463** | 0.0005 / bold 0.000 | **FAIL** disp+sym |

None of the three pass the V16.4 forensic that caught `sameer_glm10f`. Double-count vs LIVE residual is clean for all three (‖corr‖≪0.08) — they are **not** already inside LIVE — but **OOF↔test β asymmetry** is large (especially `sadam`: β(`z_ATH`) −0.74→−0.06, β(`MS`) 0.86→0.25).

**Caveat (inflates max‖Δβ‖):** test basis used the same `lm7_test_5fold` array for both `LM` and `L7F`, so those two β columns are identical on test. That cannot explain the `sadam`/`MS`/`z_ATH` asymmetries, which dominate the fail.

**Verified signals after forensic: `[]`.**

### 1.2 Screens on top of `z_S3`

| Composite | Main folds | Best `w` | meanΔ (×1e5) | sf25 | `t` | Gate (≥19/25 & t≥3.20 & 5/5) |
|---|---|---:|---:|---:|---:|---|
| GPU-pure `orth(q80−q20)+orth(R2−R3)` | **4/5** | 0.02 | +0.55 | 14/25 | 0.55 | **NO** |
| Full-3 equal weights | **5/5** | 0.025 | +2.02 | 17/25 | 2.21 | **NO** |
| Best weighted `(0.7, 0.3, 0.0)` = mostly sadam + q80, **curv dropped** | **5/5** | 0.03 | +3.64 | **18/25** | **3.25** | **NO** (18<19) |

Best weight grid (w1,w2,w3) ∈ {0,0.3,0.5,0.7,1} chose **`(0.7, 0.3, 0.0)`** — curvature adds nothing once sadam+spread are present; GPU-pure alone loses a main fold.

### 1.3 Staging (NOSUBMIT)

Staged **best weighted** on Slot-3 LIVE CSV with quantile-lock:

- File: `submissions/sub_v16_6_composite_5of5_NOSUBMIT.csv`
- `w*=0.03`, `(1−ρ_S)` vs Slot3 ≈ **1.39e-4**
- Soft gate **False** (18/25 sf; forensic of constituents failed)
- **Do not submit.**

**Bottom line Step 1:** Combining the three residual `5/5` signals does **not** clear `≥19/25` / `t≥3.20` under a forensic-clean composite. Best reachable here is **5/5 · 18/25 · t=3.25** on an **unverified** (β-asymmetric) sadam-heavy mix.

---

## Step 2 — Target-First GPT-2 (`MS` / `G26`) Architecture & Fold Recon

### 2.1 Do we have the exact train / architecture script?

**YES — locally, in two copies of the public Matapanino MIT repo:**

| Path | Contents |
|---|---|
| `external/matapanino_gpt2_repo/` | Full pipeline: `train_replica.py`, `train_classifier.py`, `score_rows.py`, `make_predictions.py`, `make_queue.py`, `run_queue.py`, `tabtext.py`, tests |
| `external_bank/masaya_gpt2_repo/` | Same architecture scripts |
| `external/gpt2_target_first_09501/` | Masaya public notebook `s6e11-gpt-2-target-first-classifier-cv-0-9500.ipynb` + kernel metadata |
| `scripts/lm_medium_member.py` | Our compact-token target-first LM (`GPT2LMHeadModel`, gpt2 / gpt2-medium) — related, not the exact G26 score recipe |

Public upstream: `github.com/Matapanino/s6e11-gpt2-target-first` (credited in `/tmp/s6e11_v16/g26/README.md` and `external_bank/masaya_gpt2/README.md`).

**What we do *not* have locally for G26 test completion:**

- Fold-1..4 **test** tensors (`gpt2_s2026_test_f1..f4` / `test_5fold`) — only `gpt2_s2026_test_f0.npy` + full `gpt2_s2026_oof_5fold.npy` under `/tmp/s6e11_v16/g26/`
- Trained `.pt` / `fits/f{k}/final/` weights for seed 2026 (DEAD_ENDS N9: kernel outs = aggregated OOF/HTML only)
- `kaggle` CLI not on PATH in this shell; dataset listing deferred — README on the G26 artifact still says folds 1–4 test scoring was unfinished (~12.6 GPU-h each)

So: **architecture to regenerate G26 test folds 1–4 exists on disk**; **finished fold-1..4 test predictions / checkpoints do not.** Regenerating is a multi-GPU-hour job (0 pods this step — not started).

### 2.2 G26 ↔ L7_s2026 relationship (folds 1–4), controlling `MS` & `L7F`

Setup: OOF on `fold_id ≥ 1` (n=559968).  
`L7_s2026` = mean order LLR from `(N,10,2)` GPU raw (ρ vs `R3` = 0.9999).  
`MS` = `masaya_gpt2` OOF (true GPT-2 seed contrast partner).  
`L7F` = compact-LM seed7 5-fold OOF.

| Metric (folds 1–4) | Value |
|---|---:|
| Spearman(`G26`, `L7_s2026`) | **0.9753** |
| Partial corr `L(G26)` ↔ `L(L7_s2026)` \| `L(MS)`, `L(L7F)` | **0.200** |
| Same partial using `R3` instead of mean-LLR | 0.184 |
| Partial \| `P_super_v4`, `L7F` (weaker MS control) | 0.371 |
| Spearman(`G26−MS`, `L7_s2026−L7F`) **2-seed contrasts** | **0.022** |

Per-fold raw Spearman `G26`–`L7_s2026` stays ~0.975 on **all** folds 0–4; `G26`–`MS` ~0.989–0.990.

**Interpretation:** Levels of G26 and L7_s2026 are tightly aligned, but after removing the shared GPT-2 / LM-seed-7 axes, residual shared structure is only **~0.20**. The **seed-to-seed contrasts are essentially orthogonal** (ρ≈0.02) — so a G26 seed-delta is **not** recoverable as a linear transform of the L7 seed-delta. Having full `L7_s2026_test_0..4` (`preds/l7_s2026_5f/test_{0..4}.npy` + raw `(5,N,10,2)`) does **not** substitute for missing G26 test folds 1–4.

OOF AUCs (sanity): G26 **0.95014**, MS **0.95003**, L7_s2026 **0.94927**, L7F **0.94945**.

---

## Verdicts

1. **Composite:** Forensic rejects all three constituents; best combo is **5/5 · 18/25 · t=3.25** (sadam 0.7 + spread 0.3) — **below** the 19/25 gate. Staged NOSUBMIT only.
2. **G26:** Exact Target-First GPT-2 train/score scripts **are on disk** (`external/matapanino_gpt2_repo/`, `external_bank/masaya_gpt2_repo/`). Missing piece is **compute + fold-1..4 test outputs**, not the architecture definition. L7 full-fold test bank cannot proxy the G26 seed contrast.

`NO_SUBMIT=True` — nothing uploaded to Kaggle in V16.6.
