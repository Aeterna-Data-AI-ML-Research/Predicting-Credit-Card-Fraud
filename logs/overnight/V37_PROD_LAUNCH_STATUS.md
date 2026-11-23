# V37 Production GPU Launch — Live Status

**Updated:** 2026-11-22 ~07:22Z  
**NO_SUBMIT=True** · Foreign lock: `efx11br58dkgqp` (siclink) — never touched

## Pod

| Field | Value |
|---|---|
| id | **`khx6qxz80yfenn`** |
| name | `v37-gm-tfm-lm` |
| GPU | NVIDIA A40 44.4GB (Secure, CA-MTL-1) |
| $/hr | **$0.59** (cap $0.89) |
| Watchdog | 7200s → `runpodctl remove pod khx6qxz80yfenn` (pid in `/tmp/watchdog.pid`) |
| Access | `exec-pod` (direct SSH refused from Mac; proxy PTY quirk) |

## Worker preflight (live)

- **TabPFN:** `TABPFN_AUTH_SKIP` — needs `TABPFN_TOKEN` / Prior Labs license (no hardcoded safetensors path)
- **TabICLv2:** OK
- **GPT-2:** OK after `transformers==4.46.3` pin (`True` tok=6407, `False` tok=10352, both length-1)
- Torch: `2.4.1+cu124` (restored after accidental 2.15 upgrade)

## Canary adjustments (A40 wallclock)

TabICL predict ≈ **5 ms/row/estimator** → full 440k × 4 bags × 4 est ≫ 40 min.  
Fold-0 canary uses **4×20k bags, `n_estimators=2`, `kv_cache=True`, chunk 4096, fp32 autocast off**.

## Local $0 audit (done)

See `logs/overnight/v37_local_audit/V37_LOCAL_AUDIT.md`

| Reducer (l7 5-fold LL) | solo | disp | dAUC vs fullbank |
|---|---:|---:|---:|
| R_mean_diff | 0.94927 | 1.033 | −3.3e-3 |
| R_deotte_lse | 0.94921 | 1.032 | −3.4e-3 |
| R_ranklogit | 0.94927 | 1.033 | −3.3e-3 |

| Vault | Result |
|---|---|
| Amanatar Tri-TabPFN (`oof_P2`) | solo **0.94968**, dAUC_fb **−2.9e-3** (hygiene solo OK; disp TBD in npz audit) |
| Busyaprime 124-voices | **403 Forbidden** — V_TabM/V_EBM/V_BPE_GBDT not available |

## Scripts

- `scripts/v37_gpu/run_fold0_tfm_and_transductive_lm.py` — production worker (no self-delete)
- `scripts/v37_gpu/launch_v37_pod.py` — Mac controller (`--attach` supported)
- `scripts/v37_local_parallel_audit.py` — local Bayes/vault

## Next

1. Wait Fold-0 `PROMOTE.json` + finite npy under `/workspace/out/fold_0/`
2. `runpodctl send` from pod → Mac `preds/v37_gpu/` (rsync-before-delete)
3. Gate fail → delete `khx6qxz80yfenn` only; pass → kill 2h WD, arm 4h, folds 1–4
