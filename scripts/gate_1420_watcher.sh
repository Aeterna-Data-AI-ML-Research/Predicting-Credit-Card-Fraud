#!/usr/bin/env bash
# Hard VA gate at ~14:20 UTC. NO_SUBMIT=True.
# Running fold0 is NOT touched until VA-complete signal (12x n=139992 or "scoring TEST").
set -euo pipefail
SSD=/tmp/s6e11_v39_ssd
POD=zehvvv1vrbys16
SSH=(ssh -i ~/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p 40603 root@38.65.239.32)
RSYNC_SSH='ssh -i ~/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 40603'
LOG=$SSD/out/gate_1420_watcher.log
mkdir -p "$SSD/out"
echo "NO_SUBMIT=True" > "$SSD/out/NO_SUBMIT"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "armed NO_SUBMIT gate watcher pid=$$"

python3 - <<'PY'
from datetime import datetime, timezone
import time
now = datetime.now(timezone.utc)
target = now.replace(hour=14, minute=15, second=0, microsecond=0)
if target < now:
    target = now
delay = (target - now).total_seconds()
print(f"sleep {delay:.0f}s until {target}", flush=True)
time.sleep(max(0, delay))
PY

log "polling for VA complete (12x n=139992 or scoring TEST)"

for i in $(seq 1 180); do
  out=$("${SSH[@]}" 'echo VA=$(grep -c "scored order=.*n=139992" /workspace/logs/fold0.log); echo TE=$(grep -c "scoring TEST" /workspace/logs/fold0.log || echo 0); tail -2 /workspace/logs/fold0.log' 2>/dev/null | grep -v Warning || true)
  log "poll $i: $out"
  n_va=$(echo "$out" | sed -n 's/^VA=//p' | head -1 | tr -d '[:space:]')
  n_te=$(echo "$out" | sed -n 's/^TE=//p' | head -1 | tr -d '[:space:]')
  n_va=${n_va:-0}
  n_te=${n_te:-0}
  if [ "$n_va" -ge 12 ] || [ "$n_te" -ge 1 ]; then
    log "VA COMPLETE signal — killing fold0 to stop Test burn"
    "${SSH[@]}" 'PID=$(cat /workspace/logs/fold0.pid); kill $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null; echo killed; nvidia-smi --query-gpu=memory.used --format=csv,noheader' 2>/dev/null | tee -a "$LOG" || true
    sleep 8
    log "launching VA-only rescore (saves ll.npy, no Test)"
    "${SSH[@]}" 'nohup python -u /workspace/scripts/score_qwen_va_only.py > /workspace/logs/va_only.log 2>&1 & echo $! > /workspace/logs/va_only.pid; cat /workspace/logs/va_only.pid' 2>/dev/null | tee -a "$LOG"

    ready=0
    for j in $(seq 1 540); do
      if "${SSH[@]}" 'test -f /workspace/out/qwen15b_f0_ll_va.npy' 2>/dev/null; then
        log "VA ll ready (wait iter $j)"
        ready=1
        break
      fi
      if [ "$j" -eq 540 ]; then
        log "VA-only TIMEOUT — terminate pod $POD"
        /Users/mz/.local/bin/runpodctl remove pod "$POD" >>"$LOG" 2>&1 || true
        exit 3
      fi
      sleep 30
    done
    [ "$ready" -eq 1 ]

    rsync -avz -e "$RSYNC_SSH" --no-owner --no-group \
      root@38.65.239.32:/workspace/out/qwen15b_f0_ll.npy \
      root@38.65.239.32:/workspace/out/qwen15b_f0_ll_va.npy \
      "$SSD/out/" 2>&1 | tee -a "$LOG" | grep -v Warning || true

    cd /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud
    if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
    set +e
    "$PY" "$SSD/scripts/gate_va_hard_1420.py" | tee "$SSD/out/gate_va_hard_1420_run.log"
    rc=${PIPESTATUS[0]}
    set -e
    log "gate rc=$rc NO_SUBMIT=True"
    if [ "$rc" -ne 0 ]; then
      log "GATE FAIL — terminate pod $POD"
      /Users/mz/.local/bin/runpodctl remove pod "$POD" >>"$LOG" 2>&1 || true
      exit 1
    fi
    log "GATE PASS — keep pod; NO_SUBMIT (no kaggle submit)"
    exit 0
  fi
  sleep 30
done
log "watcher gave up waiting for VA complete"
exit 4
