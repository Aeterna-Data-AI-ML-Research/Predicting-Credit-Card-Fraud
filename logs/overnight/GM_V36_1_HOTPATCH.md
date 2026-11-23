# GM V36.1 LIVE HOTPATCH — 2026-11-21 ~22:55–22:58Z

## Trigger
`W1_E1` solo **0.9599** with G0 fail (`r²=0.92`, F1/F2 fail) despite `g_fb` 5/5 · 25/25 · +3.3e-4.

## Root causes found
1. **LEAKAGE in `feats_entity_balance`:** `peer_risk` used **all-train** `is_fraud` group means applied to every train row (including future `va`). Impossible honest OOF ≥0.955. Exact site: global `groupby(ekey)` TE before the hotpatch.
2. **Mode-B `r2_B152` on reconstructed `η+δ`:** guaranteed ~0.90–0.98 vs bank — false reject.
3. **F1/F2 too tight for 5-fold bagged trees** + weak test-side Method-B lock.

## W1_E2 decision
Mode-A solos ~**0.952** (honest ceiling). **No y TE in `feats_bpe_lite`.** **Left running** (not killed).
Pre-patch import → queue daemon **re-gates** E2/E3/E4 OOF/test with fixed G0 when each finishes.

## Patches applied (on disk; next workers import fresh)
| File | Change |
|------|--------|
| `v36_1_lib.evaluate_engine_g0` | `solo>0.9550 → LEAKAGE_DETECTED`; competence `[0.9515,0.9550]`; **`r2` on δ**; F1 bagged `[0.84,1.06]` + KS\<0.02; **`disp_1f`∈[0.93,1.07]** when `te_by_fold` present; F2 maxcorr \<0.010 |
| `v36_1_lib.frozen_lofo_orth` | Ridge(α=50) LOFO + Test Gram-Schmidt vs `[z_fb, z_twin, SVD_25(B), B_cols]` |
| `v36_1_lib.train_multiseed_stack` | returns `meta["te_by_fold"]` (5×ntest) for disp_1f |
| `run_wave_engine.feats_entity_balance` | **Fold-safe peer_risk** only (`tr=folds!=f`); no global TE |
| `run_wave_engine` | `W1_E1`→`E1_fixed`; `W3_JEWELS` (+B_xgb_llr, v10_cb_d4); Stage-B skips leaked E1 |
| Supervisor order file | `engine_order.json` injects `W1_E1_fixed` after E4 + jewels |
| Queue | `hotpatch_queue_e1_fixed.py` re-gates E2–E4 + runs E1_fixed after E4 + jewels |

## Queue
- `W1_E1_fixed` after `W1_E4`
- `W3_JEWELS`: P4_L4.2/4.3/4.5 + v10_ridge_resid + B_xgb_llr + v10_cb_d4 through fixed Dual-Baseline G0

## Confirmation
- E2 process left alive at hotpatch time (fold-clean solos ~0.952).
- Supervisor **not** killed.
- Leak guard smoke: `LEAKAGE_DETECTED` on synthetic near-perfect OOF.
- Queue daemon restarted with re-gate + E1_fixed.

## Follow-up 00:18–00:23Z
- Queue crashed re-gating E2: `_gs_strip` flattened SVD (n×k) → shape error. Fixed: column-wise GS.
- E2 re-gated with fixed G0: solo=0.95185, r2_δ=0.995, f1=False, f2_corr=0.017 → **g0/vault False** (spanned, not leakage).
- E3 left running.

## 01:27Z E3 gate fail
- E3 stack finished under **pre-GS-fix import**; gate crashed (`_gs_strip` SVD flatten). **No E3 preds saved**; state=`fail`.
- E4 left running (fresh process, fixed lib).
- Queue updated: **retry W1_E3 after E4**, then supervisor continues `E1_fixed`.

## 02:37Z E1_fixed skipped + E5 leak ingest
- SUP for-iterator predated hotpatch → **skipped W1_E1_fixed**, jumped to W2/E5.
- E5 Mode-A solos ~0.9595 from leaked E1 preds → **killed E5**, quarantined `W1_E1_entity_balance_graph*`.
- Queue now always launches `E1_fixed` after E3 retry, then jewels + clean E5.
- E4 gated: solo 0.95155, f1=True, f2/g0/vault False.

## 03:37Z time-critical
- State had stale E3=fail while PID 91401 still training; E1_fixed never launched.
- **Parallel launch:** W1_E1_fixed + W3_JEWELS now; E3+E7 continue.
- E6 timeout 124; MultiVault UnicodeDecodeError on junk gate file.

## 05:xxZ disarm + morning harvest
- Killed W4_E7 + stopped SUP/queue (04:30Z disarmed); E3+E1_fixed finished cleanly.
- Ran `v36_2_morning_harvest_audit.py` NO_SUBMIT=True → `GM_V36_2_MORNING_HARVEST.md`.
- E1_fixed solo 0.95195 (fold-clean). Pure-residual jewel rescoring: harvest_p4 / P4 show negative w ≈ -0.02..-0.03.
- Kaggle submits: 0.
