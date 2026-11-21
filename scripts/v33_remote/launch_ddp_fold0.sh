#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export V33_BUNDLE=/workspace/bundle
export V33_OUT=/workspace/canary
export V33_EPOCHS="${V33_EPOCHS:-1}"
export V33_PER_GPU_BS="${V33_PER_GPU_BS:-96}"
export V33_SCORE_BS="${V33_SCORE_BS:-256}"
export V33_MAX_LEN="${V33_MAX_LEN:-112}"
export V33_POS_OVERSAMPLE=6
export V33_LR=3e-4
export V33_WARMUP=0.04
export V33_NUM_WORKERS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=8

cd /workspace
mkdir -p /workspace/canary /workspace/logs
echo "[launch] $(date -u) PER_GPU_BS=$V33_PER_GPU_BS MAX_LEN=$V33_MAX_LEN EPOCHS=$V33_EPOCHS"
/workspace/venv/bin/torchrun --standalone --nproc_per_node=2 \
  /workspace/scripts/worker_a_ddp_fold0.py \
  2>&1 | tee /workspace/logs/ddp_fold0.log
echo "[launch] exit=$? $(date -u)"
date -u +%Y-%m-%dT%H:%M:%SZ > /workspace/canary/DDP_DONE
