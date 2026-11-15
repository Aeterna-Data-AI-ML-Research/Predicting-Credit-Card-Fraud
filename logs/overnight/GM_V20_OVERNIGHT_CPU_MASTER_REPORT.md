# GM V20 Overnight CPU Grandmaster Factory → **NEW ATH 0.95399**

**0 GPU pods · `$0.00/hr` · elapsed ~442s · `n_jobs=-1` · `scripts/gm_v20_overnight_cpu_grandmaster_factory.py`**

| | |
|---|---|
| Prior ATH (Slot8) | 0.95395 |
| **Live Slot9** | **`0.95399` (+4e-5)** |
| Chosen | `E2_heavy_ridge_w0.020` |
| Submit | **True** |
| Vault size | 14 |
| Gate-clear winners | 10 |
| Daily slots used | 1 (last remaining) |

---

## Architecture verified
1. **Rule A:** residual targets on Platt `(g_S8, h_S8)`.
2. **Rule B:** LOFO orth on `Bo12` + **Test-Covariance Alignment** on `Bt_all=[Bt12, ray_S6, ray_S7]` → `F2≈0`, `|c6|/|c7|≈0`.
3. **Rule C:** pre-projection `raw_disp` soft-scale into `[0.88,1.12]`.
4. Checkpointing: `/tmp/s6e11_v16/out/gm_overnight_vault.npz`, `gm_overnight_progress.json`.

## Engine outcomes
| Engine | Status | Notes |
|--------|--------|-------|
| E1 HierEB epistatic | ok | no vault clear after Stage-2 |
| E2 heavy+perm | partial | CatBoost `thread_count=-1` cast error; **Ridge cleared all 6 gates** |
| E3 cKDTree knn | failed | same CatBoost cast; knn ridge not reached |
| E4 Regime MoE | ok | no new promotion |
| E5 Vault synth | ok | multiple ★ blends (`S1`/`S2`) also gate-clear |

## Chosen scorecard (`E2_heavy_ridge_w0.020`)
| Gate | Value | Pass |
|------|-------|:----:|
| F2_Bo12 | ~0.000 | ✓ |
| c6 / c7 | ~0.000 | ✓ |
| 1−ρ e-4 | 0.658 | ✓ |
| z_S8 | 5/5 m=+12.1 t=10.7 sf=25 | ✓ |
| twin/nnls | 5/5 (mean lift ≈+15e-5, min≈+9.9e-5) | ✓ |
| **Public LB** | **0.95399** | ✓ ATH |

## Resource monitor (sample)
| Time | CPU% | RSS | free_MB | load |
|------|------|-----|---------|------|
| +1.5m | ~100 | ~1.1GB | 33 | ~5 |
| +3.0m | ~900 (cKDTree workers) | ~1.4GB | 206 | ~5.5 |
| +7m done | 0 | — | 434 | ~4.2 |

Log: `logs/overnight/GM_V20_OVERNIGHT_CPU_MONITOR.log`

## Artifacts
- Submission: `submissions/sub_v20_LIVE_slot9_overnight_cpu_promoted.csv`
- Staged ranks: `submissions/sub_v20_overnight_rank{1..5}_*.csv`
- Vault/meta/state under `/tmp/s6e11_v16/out/gm_overnight_*`
- Factory log: `logs/overnight/GM_V20_OVERNIGHT_CPU_FACTORY.log`

## Decision
**NEW ATH `0.95399`.** Overnight CPU factory promoted `E2_heavy_ridge` (forgotten heavyweight + perm fusion Ridge under two-stage exact basis projection). CatBoost bags failed on `thread_count=-1` (patched for next run); Ridge path alone was sufficient.
