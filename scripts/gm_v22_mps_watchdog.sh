#!/usr/bin/env bash
# Restart MPS GPT-2 pipeline if it dies before DONE. Relies on train_replica/train_classifier resume.
set -euo pipefail
RUN=/tmp/s6e11_v16/lm_research/run_mps_gpt2
ROOT=/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud
SCRIPT="$ROOT/scripts/gm_v22_mps_full_gpt2_replica.sh"
LOG="$RUN/watchdog.log"
PID_FILE="$RUN/pipeline.pid"
DONE_FILE="$RUN/DONE"
mkdir -p "$RUN"
echo "watchdog_start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

is_running() {
  if [[ -f "$PID_FILE" ]]; then
    local p
    p=$(cat "$PID_FILE" 2>/dev/null || true)
    if [[ -n "${p:-}" ]] && kill -0 "$p" 2>/dev/null; then
      return 0
    fi
  fi
  pgrep -f 'gm_v22_mps_full_gpt2_replica.sh' >/dev/null && return 0
  pgrep -f 'train_replica.py.*run_mps_gpt2' >/dev/null && return 0
  pgrep -f 'train_classifier.py.*run_mps_gpt2' >/dev/null && return 0
  pgrep -f 'score_rows.py.*run_mps_gpt2' >/dev/null && return 0
  return 1
}

while true; do
  TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ -f "$DONE_FILE" ]]; then
    echo "$TS watchdog_exit DONE present" >> "$LOG"
    exit 0
  fi
  if is_running; then
    echo "$TS ok running" >> "$LOG"
  else
    echo "$TS RESTART launching pipeline (resume from checkpoints)" >> "$LOG"
    nohup env REP_EPOCHS="${REP_EPOCHS:-20}" CLS_EPOCHS="${CLS_EPOCHS:-2}" BATCH="${BATCH:-8}" \
      SAVE_STEPS="${SAVE_STEPS:-500}" SEED="${SEED:-42}" ORDERS="${ORDERS:-30}" \
      bash "$SCRIPT" >> "$RUN/nohup.out" 2>&1 &
    sleep 15
    if ! is_running; then
      echo "$TS RESTART_FAILED" >> "$LOG"
    else
      echo "$TS restart_ok pid=$(cat "$PID_FILE" 2>/dev/null || echo '?')" >> "$LOG"
    fi
  fi
  sleep 60
done
