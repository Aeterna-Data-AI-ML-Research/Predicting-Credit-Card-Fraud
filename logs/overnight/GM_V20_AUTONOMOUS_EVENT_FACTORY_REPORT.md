# GM V20 Autonomous Event-Driven Factory → **HOLD**

**0 GPU pods · `$0.00/hr` · elapsed ~656s · `scripts/gm_v20_autonomous_event_driven_factory.py`**

| | |
|---|---|
| Slot8 ATH | **0.95395** (held) |
| Slot8 rebuild `1-ρ` | **0** |
| Daily slots this run | **0 used** (1 remaining after V20.1 miss) |
| Gate-clear promotions | **0** |
| Survivor vault | **4** families |
| Submit | **False** |

---

## Operating contract
- Events 1→5 research + vault; Event 6 submits **only** on all 6 hard gates.
- Stricter than V20.1: `t≥2.80`, `sf≥18/25`, `min_fold>+0.05e-5` ×3 bases, `1−ρ∈[0.28,1.10]e-4`, `F3<0.045`.
- ZCA uses **OOF-β transfer** (no test-side Bt refit) to avoid the V20.1 `P2_raw_L7` LB trap (0.95393).

## Event results

### E1 — Recon + forgotten heavyweights
- Kaggle upstream check run (gpt2 dataset / s6e11 datasets / kernels).
- Harvested **37** complete 5-fold streams (gpt2_med/rev, v10_*, P4, TabM, Q17, alex, caches).
- Vaulted: `v10_cb_d4_bag3_orth_LM`, `cache_P1_raw_only`.
- Best near-miss: `v10_cb_d4_bag3_orth_LM_w0.006` — 5/5×3, t≈4.3, sf=18, F2=0.032, but **`1−ρ=0.006`** (Gate 5 fail: too close to Slot8).

### E2 — Standalone Stage-B (`init_score=None`)
- `X_stageB` = 25 cols (LM+tab+U+P4+cents; **no G26_f0 / no KMAT_68**).
- Fit `SB1_cat_obliv`, `SB2_lgb_monotone`, `SB3_ridge_poly2`.
- None cleared survivor hygiene + 4/5×3 after ZCA-Bo12.

### E3 — HierEB + orig20k drift
- 8 vectorized 2-way epistatic excess crosses + ridge/cat condensation on `(g_S8,h_S8)`.
- No new vault survivors under F2/fold floors.

### E4 — Fourier GLM + MLP
- `N1_fourier_glm` multi-α Ridge on RankGauss+Fourier+U+P4 → no vault clear.
- `N2_res_mlp` **skipped** (`torch` absent in `/tmp/s6e11_v15/venv`).

### E5 — Vault synthesis
- Pairwise \|corr\| mean≈0.001 (near-orthogonal vault pair).
- Solver A/B blends vaulted (`E5_solverA`, `E5_solverB_top2`).
- Best synth still **`1−ρ≈0.11`** or fold floors short of Gate 3/5 — **0 promotions**.

### E6 — Promotion
- Gate-clear winners: **0** → **HOLD** last daily slot.
- Staged NOSUBMIT: `sub_v20_staged_top{1,2,3}_NOSUBMIT.csv`.

## Why nothing promoted
The only 5/5×3 / high-t family (`v10_cb_d4_bag3*`) is **too Slot8-collinear** (`1−ρ≪0.28e-4`). Families with enough Spearman displacement fail 5/5 or `t≥2.80` on twin/nnls. Synthesis could not flip both constraints at once.

## Artifacts
- Script: `scripts/gm_v20_autonomous_event_driven_factory.py`
- Vault: `/tmp/s6e11_v16/out/gm_survivor_vault.npz` (+ `gm_survivor_vault_meta.json`)
- State: `/tmp/s6e11_v16/out/gm_v20_factory_state.json`
- Log: `logs/overnight/GM_V20_AUTONOMOUS_EVENT_FACTORY.log`
- Staged: `submissions/sub_v20_staged_top{1..3}_NOSUBMIT.csv`

## Decision
**HOLD ATH `0.95395`.** Preserve remaining daily slot. Next lever: install torch for N2 MLP, or seek a residual with `1−ρ∈[0.28,1.10]` that is still 5/5×3 after ZCA (not the tiny-ρ v10_cb path, not the OOF-optimistic P2 path).
