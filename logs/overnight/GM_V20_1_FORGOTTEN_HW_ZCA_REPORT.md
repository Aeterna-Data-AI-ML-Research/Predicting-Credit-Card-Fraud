# GM V20.1 — Forgotten Heavyweights + ZCA-Whitened Bo12 → **LIVE Slot9 (LB miss)**

**0 GPU pods · `scripts/gm_v20_1_forgotten_heavyweights_and_zca_whitened_orth_slot9.py`**

| | |
|---|---|
| Slot8 ATH | **0.95395** (still hold) |
| Slot8 rebuild `1-ρ` | **0** (exact) |
| Streams harvested | **58** |
| ZCA + Method-B post-proj | `F2_Bo12 ≈ 0` on all screened streams |
| Submit | **True** — `cache_P2_raw_L7_w0.020` |
| Public LB | **0.95393** (−0.00002 vs ATH) |
| Daily slots left | **1** |

---

## Breakthrough 1 — ZCA + test-side Method-B lock
Naive Bo12 LOFO left `F2≈0.08–0.57` (OOF Σ sparser than Test Σ). Fix:
1. Per-fold ZCA whitening `W_tr=(Σ_tr+λI)^{-1/2}` → Ridge on whitened basis.
2. Test ZCA `W_test` then β̄ transfer.
3. **Method-B post-projection**: LOFO strip vs original `Bo12` on OOF; **test-side Ridge strip `orth_t ~ Bt12` (no `y_test`)** → `F2_Bo12=0.000` guaranteed.

Diag: `gpt2_med` naive F2 `0.571→0.000`; `P_super_v4p_hc` `0.296→0.000`; `Q17_sadam_v4_cat` `0.189→0.000`.

## Breakthrough 2 — Forgotten heavyweights on disk
Loaded & screened: `gpt2_med_4ep_5f`, `gpt2_rev_5f`, v10 (`ridge_resid`, `cb_d4_bag3`, `combo3`, …), `P_super_v4p_hc`, `TabM_5f`, `glm_quad`, Q17 streams, alex peers, cached `W*` / `P*` / `U*`.

## Gate-clear OOF winners (additive on Slot8)
| Candidate | 1-ρ e-4 | c7 | z_S8 | twin | nnls | LB |
|-----------|---------|----|------|------|------|-----|
| `cache_P2_raw_L7_w0.020` ★ submitted | 0.778 | +0.084 | 5/5 m=+13.1 | 5/5 m=+25.1 | 5/5 m=+14.3 | **0.95393** |
| `cache_P2_raw_L7_w0.016` | 0.498 | +0.084 | 5/5 m=+10.7 | 5/5 m=+20.7 | 5/5 m=+11.6 | — |
| `v10_lgb_pure_fe_w0.016` | 0.478 | −0.011 | 5/5 m=+2.95 | 5/5 m=+5.08 | 5/5 m=+3.03 | — |

Near-miss (ρ too small or \|c7\|≥0.20): `v10_ridge_resid_orth_LM`, `v10_combo3_orth_LM` (c7≈+0.26), `v10_cb_d4_bag3*`.

## Decision
**LIVE miss.** OOF gates cleared under ZCA-locked F2, but public LB **0.95393 < 0.95395**. Keep ATH Slot8. **1 daily slot remaining.**

Likely cause: `P2_raw_L7` residual is OOF-optimistic after aggressive test-side Bo12 strip (twin lift ≫ z_S8). Safer next candidate if retrying: `v10_lgb_pure_fe_w0.016` (cleaner c7, smaller ρ-matched lift).

Artifacts: `/tmp/s6e11_v16/out/v20_1.json`, `logs/overnight/GM_V20_1*.log`, `submissions/sub_v20_1_LIVE_slot9_zca_forgotten_hw.csv`.
