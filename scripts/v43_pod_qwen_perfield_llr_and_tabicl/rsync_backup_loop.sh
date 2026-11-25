#!/usr/bin/env bash
# Mac-side: pull ckpts + partial score shards every 3 min so a dead pod is not a total loss.
# Usage: POD_HOST=... POD_PORT=... ./rsync_backup_loop.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL="${ROOT}/out/v43_fold0_qwen"
mkdir -p "$LOCAL"
HOST="${POD_HOST:?set POD_HOST}"
PORT="${POD_PORT:?set POD_PORT}"
KEY="${POD_SSH_KEY:-$HOME/.ssh/id_rsa}"
REMOTE_OUT="${REMOTE_OUT:-/workspace/out/qwen}"
RSYNC=(rsync -azP -e "ssh -p $PORT -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null")
echo "backup loop → $LOCAL from root@${HOST}:${REMOTE_OUT}"
while true; do
  "${RSYNC[@]}" "root@${HOST}:${REMOTE_OUT}/ckpts/" "$LOCAL/ckpts/" || true
  "${RSYNC[@]}" "root@${HOST}:${REMOTE_OUT}/" "$LOCAL/" \
    --include='*.npy' --include='*.json' --include='*.log' --include='ckpts/***' --exclude='*' || true
  echo "[$(date -u +%H:%M:%S)] backup tick ok"
  sleep 180
done
