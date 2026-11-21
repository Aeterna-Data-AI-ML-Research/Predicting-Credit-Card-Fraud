#!/usr/bin/env bash
# Launch full GPT-2 + replica training in local Docker (CPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMG="${IMG:-s6e11-gpt2-storyteller:local}"
CODE="${CODE:-/tmp/s6e11_v16/lm_research/matapanino}"
DATA="${DATA:-$ROOT/data}"
ORIG_DIR="${ORIG_DIR:-$ROOT/original}"
RUN_HOST="${RUN_HOST:-/tmp/s6e11_v16/lm_research/run_docker_gpt2}"
STAGE="${1:-pipeline}"
shift || true

mkdir -p "$RUN_HOST"
# ensure folds script + MPS sync fix present in code mount
test -f "$CODE/train_classifier.py"

if [[ "${BUILD:-1}" == "1" ]]; then
  echo "[run_local] building $IMG …"
  docker build -t "$IMG" "$ROOT/docker/gpt2_storyteller"
fi

# Apply classifier CPU sync fix inside mounted code if not already
# (idempotent patch already applied for MPS; CPU benefits too)

echo "[run_local] stage=$STAGE → $RUN_HOST"
exec docker run --rm \
  --name s6e11_gpt2_storyteller \
  --cpus="8" \
  --memory=14g \
  -e OMP_NUM_THREADS=8 \
  -e MKL_NUM_THREADS=8 \
  -e NPROC=8 \
  -e EPOCHS="${EPOCHS:-}" \
  -e BATCH="${BATCH:-}" \
  -e FOLD="${FOLD:-}" \
  -e SEED="${SEED:-42}" \
  -e ORDERS="${ORDERS:-}" \
  -e SAVE_STEPS="${SAVE_STEPS:-}" \
  -e LOG_STEPS="${LOG_STEPS:-}" \
  -e HF_HOME=/workspace/hf \
  -v "$CODE:/workspace/code:ro" \
  -v "$DATA:/workspace/data:ro" \
  -v "$ORIG_DIR:/workspace/original:ro" \
  -v "$RUN_HOST:/workspace/run" \
  -v /tmp/s6e11_v16/hf_cache:/workspace/hf \
  "$IMG" "$STAGE" "$@"
