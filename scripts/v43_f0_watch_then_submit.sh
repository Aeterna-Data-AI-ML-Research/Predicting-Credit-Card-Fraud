#!/usr/bin/env bash
# Watch Fold-0 → rsync → gate → armed sequential submit (one-at-a-time, wait LB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH_KEY="${POD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
HOST="${POD_HOST:-69.30.85.59}"
PORT="${POD_PORT:-22101}"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$HOME/.runpod/ssh/known_hosts" -p "$PORT" "root@$HOST")
RSYNC=(rsync -azP -e "ssh -p $PORT -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.runpod/ssh/known_hosts")
LOCAL="$ROOT/out/v43_fold0_qwen"
mkdir -p "$LOCAL" "$ROOT/logs/v43"
LOG="$ROOT/logs/v43/V43_F0_WATCH_SUBMIT.log"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -u +%H:%M:%S)] watch start"

while true; do
  if "${SSH[@]}" 'test -f /workspace/out/qwen/meta_f0.json'; then
    echo "[$(date -u +%H:%M:%S)] meta_f0 present — rsync"
    "${RSYNC[@]}" "root@${HOST}:/workspace/out/qwen/" "$LOCAL/" \
      --include='*.npy' --include='*.json' --include='*.log' --include='ckpts/***' --exclude='*'
    break
  fi
  "${SSH[@]}" 'tail -n 3 /workspace/out/qwen/run.log; cat /workspace/out/qwen/ckpts/f0/state.json 2>/dev/null | head -c 200; echo' || true
  sleep 180
done

echo "[$(date -u +%H:%M:%S)] run local gate"
cd "$ROOT"
.venv/bin/python -u scripts/v43_pod_qwen_perfield_llr_and_tabicl/gate_fold0_field24_local.py \
  --qwen-dir "$LOCAL" \
  --modern-oof out/v42_1_modern/z_modern_pub_oof.npy \
  --modern-test out/v42_1_modern/z_modern_pub_test.npy \
  --data data --out logs/v43 --fold 0 \
  && GATE_OK=1 || GATE_OK=0

if [[ "$GATE_OK" != "1" ]]; then
  echo "GATE FAIL — 0 submits"
  exit 2
fi

echo "[$(date -u +%H:%M:%S)] GATE PASS — arm sequential submit"
export V43_ARM_SUBMIT=1
export V43_DAILY_LIMIT=10
.venv/bin/python -u scripts/v43_f0_sequential_submit_ladder.py
echo "[$(date -u +%H:%M:%S)] watch done"
