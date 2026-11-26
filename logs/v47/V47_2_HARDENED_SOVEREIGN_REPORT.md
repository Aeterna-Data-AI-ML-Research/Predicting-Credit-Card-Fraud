# V47.2 HARDENED SOVEREIGN REPORT

**NO_SUBMIT=True · $0.00 RunPod spend · 0 submissions · 0 pods created**

## Step1 — Idle storage teardown
- Deleted `u4ky2oq4pohcyt` (EXITED A40). `list-pods` → **0**. Active billing **$0.00/hr**.

## Step2 — Provenance + `z_sovereign_v1`
- Provenance: **0 rejects** among v0 streams; `P_super_v6` / `soft_diff` excluded; `P_super_v5_harvest` KEEP (self `v4p_hc` + `B_xgb_llr` blend).
- Checkpoints: l7 full-5fold `.pt` = **False** (only `fold_0.pt` / `fold_latest` / `replica`); gpt2 full-5fold `.pt` = **True**.
- Symmetric zero-echo peer residual: pre-unitize hygiene OK (`disp_raw=1.0021`, `KS_raw=0.0051`) but **NOT promoted** (`pos=0/5`, `sf=1/25`; all fold Δ negative vs `z_sovereign_v0`).
- `inner_hgb_sov` hygiene PASS (`disp_raw=0.998`, `KS_raw=0.0027`) → admitted into stack.
- **z0 AUC = 0.952251 → z1 AUC = 0.952249 · Fold-0 = 0.951247**

## Step3A — l7 field24 inference
- **SKIP**: full 5-fold `l7_s2026_5f` weights absent (need `fold_1..4.pt` for honest field24).

## Step3B — HGDT-LM v2 Fold-0 (Apple MPS)
- Pure generative Target-First script ready: `scripts/v47_sovereign/hgdt_v2_mps_fold0.py`
- Throughput gate (50 batches, bs=64, MPS): **383.8 rows/sec** · ETA **422.9 min** > 90 min cap → **STOP clean** (no full train / no arrays written).
- Optional later: free private Kaggle `2× T4` packaging (still under `NO_SUBMIT` until gate clears).

## HARD STOP
- **NO_SUBMIT=True** · **$0.00 spent** · **0 Kaggle submissions** · **0 RunPod pods created**
