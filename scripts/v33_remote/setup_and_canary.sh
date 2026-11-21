#!/usr/bin/env bash
# On-pod: install deps, prefetch weights, arm watchdog, run Fold-0 canary.
set -euo pipefail
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export PIP_CACHE_DIR=/workspace/pip_cache
export V33_BUNDLE=/workspace/bundle
export V33_OUT=/workspace/canary
export PATH="/workspace/venv/bin:$PATH"
PY=${PY:-/workspace/venv/bin/python}
PIP=${PIP:-/workspace/venv/bin/pip}
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$V33_OUT" /workspace/logs

CANARY_SECS="${V33_CANARY_SECS:-5400}"
FULL_SECS="${V33_FULL_SECS:-14400}"

# Ensure venv exists (system-site-packages for image torch)
if [[ ! -x "$PY" ]]; then
  /usr/local/bin/python -m venv --system-site-packages /workspace/venv
  PY=/workspace/venv/bin/python
  PIP=/workspace/venv/bin/pip
fi

# --- Stage-1 watchdog (Fold-0 canary) ---
(
  sleep "$CANARY_SECS"
  echo "[WATCHDOG] canary ${CANARY_SECS}s expired — self-destruct" | tee -a /workspace/logs/watchdog.log
  kill -9 -1 2>/dev/null || true
) &
WD_PID=$!
echo "$WD_PID" > /workspace/canary/watchdog_canary.pid
echo "[setup] armed canary watchdog pid=$WD_PID secs=$CANARY_SECS PY=$PY"

echo "[setup] pip install…"
"$PIP" install -q --upgrade pip
"$PIP" install -q --cache-dir "$PIP_CACHE_DIR" \
  "transformers>=4.44" peft accelerate safetensors sentencepiece \
  tabicl tabpfn scikit-learn scipy pyarrow zstandard pandas numpy

echo "[setup] prefetch HF weights…"
"$PY" - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
mid=os.environ.get("V33_MODEL","Qwen/Qwen2.5-1.5B")
print("tok", mid)
AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
print("model", mid)
AutoModelForCausalLM.from_pretrained(mid, torch_dtype="auto", trust_remote_code=True)
print("ok")
PY

"$PY" - <<'PY' || true
import numpy as np
try:
    from tabpfn import TabPFNClassifier
    X=np.random.randn(50,8).astype(np.float32); y=(X[:,0]>0).astype(int)
    try:
        c=TabPFNClassifier(ignore_pretraining_limits=True, device="cpu")
    except TypeError:
        c=TabPFNClassifier(device="cpu")
    c.fit(X,y)
    print("tabpfn ok")
except Exception as e:
    print("tabpfn prefetch skip", e)
try:
    from tabicl import TabICLClassifier
    X=np.random.randn(80,8).astype(np.float32); y=(X[:,0]>0).astype(int)
    c=TabICLClassifier(); c.fit(X,y); print("tabicl ok")
except Exception as e:
    print("tabicl prefetch skip", e)
PY

NGPU=$("$PY" -c "import torch; print(torch.cuda.device_count())")
echo "[setup] cuda devices=$NGPU"
nvidia-smi -L || true

cd /workspace
if [[ "$NGPU" -ge 2 ]]; then
  echo "[canary] parallel Worker A (cuda:0) + Worker B (cuda:1)"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u /workspace/scripts/worker_b_tab_canary.py \
    2>&1 | tee /workspace/logs/worker_b.log &
  PID_B=$!
  CUDA_VISIBLE_DEVICES=0 "$PY" -u /workspace/scripts/worker_a_qwen_canary.py \
    2>&1 | tee /workspace/logs/worker_a.log &
  PID_A=$!
  wait $PID_B; EC_B=$?
  wait $PID_A; EC_A=$?
  echo "[canary] done A=$EC_A B=$EC_B"
else
  echo "[canary] sequential B then A on single GPU"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u /workspace/scripts/worker_b_tab_canary.py \
    2>&1 | tee /workspace/logs/worker_b.log
  EC_B=$?
  CUDA_VISIBLE_DEVICES=0 "$PY" -u /workspace/scripts/worker_a_qwen_canary.py \
    2>&1 | tee /workspace/logs/worker_a.log
  EC_A=$?
  echo "[canary] done A=$EC_A B=$EC_B"
fi

date -u +%Y-%m-%dT%H:%M:%SZ > /workspace/canary/CANARY_DONE
echo "A=$EC_A B=$EC_B" >> /workspace/canary/CANARY_DONE
exit $(( EC_A != 0 || EC_B != 0 ))
