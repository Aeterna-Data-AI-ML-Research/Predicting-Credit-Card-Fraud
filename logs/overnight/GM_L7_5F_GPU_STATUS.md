# GM L7 5F Rotated-10 — Live Status

## Phase 0 ($0) — PASS
- Architecture: GPT-2 small compact target-first (`megayak lm_member.py` / `goodpjw seed7`)
- Scoring: full-sequence NLL under False/True × K orders — **not** final-token 2-class logits
- Shapes verified: `lm7_oof_f0.npy` `(139992,10,2)`, `lm7_test_f0.npy` `(299984,4,2)`
- Serializer: `V=1372 T=72 NTOK=1401 NC=24` cache built
- A40 estimate Folds 1–4: **~4.97h / ~$2.93** @ `$0.59/hr` (BATCH=64) — within ≤6.5h / ≤$3.85
- Report: `logs/overnight/GM_L7_5F_PHASE0_PREFLIGHT.md`

## Phase 1 — Pod
- **Did not touch** blacklisted `lheroidmuab1j4`
- Skipped foreign RUNNING A40 `d4w7dded19yo43` (gemma/vLLM P729) — watchdog on it was armed then **disarmed**
- Created network volume `qrgb0y4zgf` (`l7-5f-rot10-vol`, 40GB HP, CA-MTL-1)
- Created pod **`rsoju37zhpzaae`** (`l7-5f-rot10-a40`) — Secure **NVIDIA A40 @ $0.59/hr**, CA-MTL-1, CUDA 13.2
- Image: `runpod/pytorch:1.4.1-rc.171-cu1300-torch2130-ubuntu2404`
- Hard watchdog: 510 min local + on-pod (`scripts/l7_pod_watchdog.sh`) → self-terminate

## Phase 2 — Training (in progress)
- Fold 0 reused from disk
- Folds `1,2,3,4` launched with BATCH=64, SCORE_BS=512, SEED=7, rotated test orders
- Log: `/workspace/l7_work/preds/l7_5f_rot10/train.log` on pod
- Local sync target: `preds/l7_5f_rot10/`

## Next
- Fold-1 canary gate (AUC≥0.9480 + orth lift) before F2–4 continue (script-enforced)
- Stage-B booster → rsync → terminate pod → Phase 4 local NO_SUBMIT report
