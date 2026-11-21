#!/usr/bin/env bash
# Watches canary dir: when TabICL done → stop TabPFN idle; when EP1 ckpt ready → score on cuda:1;
# when EP2 ckpt ready → shard-score va0 on both GPUs.
set -euo pipefail
export HF_HOME=/workspace/hf_cache TRANSFORMERS_CACHE=/workspace/hf_cache
export V33_BUNDLE=/workspace/bundle V33_OUT=/workspace/canary
PY=/workspace/venv/bin/python
LOG=/workspace/logs/supervisor.log
exec > >(tee -a "$LOG") 2>&1
echo "[sup] start $(date -u)"

wait_file() { local f="$1" t="${2:-7200}"; local i=0; while [[ ! -f "$f" ]]; do sleep 5; i=$((i+5)); [[ $i -ge $t ]] && return 1; done; return 0; }

# --- TabICL harvest / kill TabPFN idle ---
while true; do
  if [[ -f /workspace/canary/tabicl_va0.npy ]]; then
    echo "[sup] tabicl present"
    break
  fi
  # if B log shows TabPFN auth fail or mean saved
  if grep -q "TabICL mean AUC" /workspace/logs/worker_b.log 2>/dev/null; then
    echo "[sup] TabICL mean logged — waiting npy"
  fi
  if grep -qiE "TABPFN_TOKEN|401|authentication|gated" /workspace/logs/worker_b.log 2>/dev/null; then
    echo "[sup] TabPFN auth issue detected — killing worker B to free cuda:1"
    pkill -f worker_b_tab_canary || true
    break
  fi
  # if B dead
  if ! pgrep -f worker_b_tab_canary >/dev/null; then
    echo "[sup] worker B dead"
    break
  fi
  sleep 10
done

# If TabICL finished in log but npy missing, give a moment
sleep 2

# Kill B if still running TabPFN after tabicl npy exists
if [[ -f /workspace/canary/tabicl_va0.npy ]]; then
  if pgrep -f worker_b_tab_canary >/dev/null; then
    # only kill if past TabICL mean (into PFN)
    if grep -q "TabICL mean AUC" /workspace/logs/worker_b.log 2>/dev/null; then
      echo "[sup] TabICL done — killing B to free cuda:1 (skip TabPFN wait)"
      pkill -f worker_b_tab_canary || true
    fi
  fi
  echo TABICL_READY > /workspace/canary/TABICL_READY
fi

# --- EP1 score on cuda:1 while EP2 trains on cuda:0 ---
echo "[sup] waiting EP1_CKPT_READY"
if wait_file /workspace/canary/EP1_CKPT_READY 9000; then
  echo "[sup] EP1 ckpt ready — score on cuda:1"
  CUDA_VISIBLE_DEVICES=1 V33_EPOCH=1 V33_SHARD=all V33_SCORE_BS=96 \
    $PY -u /workspace/scripts/score_va_parallel.py \
    > /workspace/logs/score_ep1_cuda1.log 2>&1 &
  echo $! > /workspace/canary/score_ep1.pid
fi

# --- EP2 shard score on both GPUs ---
echo "[sup] waiting EP2_CKPT_READY"
if wait_file /workspace/canary/EP2_CKPT_READY 9000; then
  # wait ep1 score to finish if still running so we don't OOM cuda1 mid-load — or kill it if done enough
  if [[ -f /workspace/canary/qwen15b_ep1_va0.npy ]]; then
    echo "[sup] ep1 score already done"
  else
    echo "[sup] waiting ep1 score finish before dual shard (or 10m)"
    wait_file /workspace/canary/qwen15b_ep1_va0.npy 600 || true
  fi
  # free score process
  pkill -f 'score_va_parallel' || true
  sleep 2
  echo "[sup] EP2 dual-shard score"
  CUDA_VISIBLE_DEVICES=0 V33_EPOCH=2 V33_SHARD=0 V33_SCORE_BS=96 \
    $PY -u /workspace/scripts/score_va_parallel.py > /workspace/logs/score_ep2_s0.log 2>&1 &
  PID0=$!
  CUDA_VISIBLE_DEVICES=1 V33_EPOCH=2 V33_SHARD=1 V33_SCORE_BS=96 \
    $PY -u /workspace/scripts/score_va_parallel.py > /workspace/logs/score_ep2_s1.log 2>&1 &
  PID1=$!
  wait $PID0; wait $PID1
  $PY - <<'PY'
import numpy as np
from pathlib import Path
OUT=Path('/workspace/canary')
a=np.load(OUT/'qwen15b_ep2_shard0_va0.npy')
b=np.load(OUT/'qwen15b_ep2_shard1_va0.npy')
full=np.concatenate([a,b]).astype(np.float32)
assert full.shape[0]==139992, full.shape
np.save(OUT/'qwen15b_ep2_va0.npy', full)
print('merged ep2', full.shape, float(full.mean()))
(OUT/'SCORE_ep2_merged_DONE').write_text('ok')
PY
fi

echo "[sup] done $(date -u)"
date -u +%Y-%m-%dT%H:%M:%SZ > /workspace/canary/SUPERVISOR_DONE
