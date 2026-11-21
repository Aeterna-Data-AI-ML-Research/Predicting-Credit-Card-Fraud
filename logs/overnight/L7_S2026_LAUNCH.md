# L7 SEED=2026 Fold-0 Launch

**UTC:** 2026-11-12T16:45:53Z  
**NO_SUBMIT=True** — all 4 submissions held  
**Phase0:** PASS (`logs/overnight/PHASE0_S2026_PASS.json`) — V=1372 T=72, orders disjoint from seed7, saver shapes OK

## Pod
- **id:** `nnts6ddqh5mprf`
- **GPU:** NVIDIA A40 Secure EU-SE-1 @ **$0.59/hr** (4090/L40S stock NONE at launch)
- **image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (template runpod-torch-v280)
- **SSH:** `ssh -i ~/.ssh/casmi_runpod -p 22131 root@194.68.245.64`
- **Watchdog:** 5h wall (~$2.95) then pkill; hard cap $3.00
- **Never touch:** `lheroidmuab1j4`

## Pace (observed)
- Replica ep1–2: **~1.0 min/epoch** → replica-only ≈ **60 min** (Gate3 wall ≤50m is at risk on A40)
- Config: REP=60, FT=8, BATCH=64, SCORE_BS=512, USE_COMPILE=1, K_test(f0)=10

## Next
1. Monitor polls every 3m for `F0_READY_FOR_LOCAL_CANARY`
2. Rsync `oof_0.npy` + `test_0.npy` → `preds/l7_s2026_5f/`
3. Local 3-gate canary; kill pod on fail or after decision
