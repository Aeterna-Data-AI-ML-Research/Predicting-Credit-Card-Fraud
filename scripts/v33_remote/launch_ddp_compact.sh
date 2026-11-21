#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf_cache TRANSFORMERS_CACHE=/workspace/hf_cache
export V33_BUNDLE=/workspace/bundle V33_OUT=/workspace/canary
export V33_EPOCHS="${V33_EPOCHS:-1}"
export V33_PER_GPU_BS="${V33_PER_GPU_BS:-96}"
export V33_SCORE_BS=256
export V33_POS_OVERSAMPLE=6 V33_LR=3e-4 V33_WARMUP=0.04
export V33_NUM_WORKERS="${V33_NUM_WORKERS:-2}"
export V33_GRAD_CKPT="${V33_GRAD_CKPT:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
cd /workspace
echo "[launch-compact] $(date -u) BS=$V33_PER_GPU_BS CKPT=$V33_GRAD_CKPT"
/workspace/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  /workspace/scripts/worker_a_ddp_compact.py 2>&1 | tee /workspace/logs/ddp_compact.log
# If OOM with grad_ckpt=0, auto-retry with ckpt=1 bs=128
ec=${PIPESTATUS[0]}
if [[ $ec -ne 0 ]] && [[ "${V33_GRAD_CKPT}" == "0" ]]; then
  echo "[launch-compact] retry with GRAD_CKPT=1 PER_GPU_BS=128"
  export V33_GRAD_CKPT=1 V33_PER_GPU_BS=128
  /workspace/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
    /workspace/scripts/worker_a_ddp_compact.py 2>&1 | tee -a /workspace/logs/ddp_compact.log
fi
date -u +%Y-%m-%dT%H:%M:%SZ > /workspace/canary/DDP_DONE
