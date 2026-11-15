#!/usr/bin/env bash
# CPU / memory monitor for MPS GPT-2+replica pipeline
LOG="/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/logs/overnight/GM_V22_MPS_MONITOR.log"
RUN=/tmp/s6e11_v16/lm_research/run_mps_gpt2
DONE_FILE="$RUN/DONE"
mkdir -p "$(dirname "$LOG")" "$RUN"
echo "monitor_start $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG"

find_train_pid() {
  pgrep -f 'train_replica.py.*run_mps_gpt2' | head -1
  pgrep -f 'train_classifier.py.*fits/f' | head -1
  pgrep -f 'score_rows.py.*run_mps_gpt2' | head -1
}

while true; do
  TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ -f "$DONE_FILE" ]]; then
    echo "$TS monitor_exit DONE" >> "$LOG"
    exit 0
  fi

  PID=""
  for cand in $(pgrep -f 'train_replica.py|train_classifier.py|score_rows.py' 2>/dev/null); do
    if ps -p "$cand" -o args= 2>/dev/null | rg -q 'run_mps_gpt2|matapanino'; then
      PID=$cand
      break
    fi
  done
  if [[ -z "$PID" ]]; then
    PID=$(pgrep -f 'gm_v22_mps_full_gpt2_replica.sh' | head -1 || true)
  fi

  # system
  LOAD=$(sysctl -n vm.loadavg 2>/dev/null | tr -d '{}')
  FREE_MB=$(vm_stat 2>/dev/null | awk '/Pages free/{gsub("\\.","",$3); print int($3*4096/1024/1024)}')
  INACT_MB=$(vm_stat 2>/dev/null | awk '/Pages inactive/{gsub("\\.","",$3); print int($3*4096/1024/1024)}')
  WIRED_MB=$(vm_stat 2>/dev/null | awk '/Pages wired/{gsub("\\.","",$4); print int($4*4096/1024/1024)}')
  COMP_MB=$(vm_stat 2>/dev/null | awk '/Pages occupied by compressor/{gsub("\\.","",$5); print int($5*4096/1024/1024)}')
  # pressure
  PRESS=$(memory_pressure 2>/dev/null | awk -F: '/System-wide memory free percentage/{gsub(/ /,"",$2); print $2}' | head -1)

  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    PS=$(ps -o pid=,pcpu=,pmem=,rss=,etime= -p "$PID" 2>/dev/null | head -1)
    RSS_MB=$(ps -o rss= -p "$PID" 2>/dev/null | awk '{print int($1/1024)}')
    # children RSS sum (python workers)
    CRSS=$(ps -o rss= -g "$(ps -o pgid= -p "$PID" | tr -d ' ')" 2>/dev/null | awk '{s+=$1} END{print int(s/1024)}')
    # replica progress snippet
    PROG=""
    if [[ -f "$RUN/replica/progress.jsonl" ]]; then
      PROG=$(tail -1 "$RUN/replica/progress.jsonl" 2>/dev/null | /tmp/s6e11_mps_venv/bin/python -c 'import sys,json; d=json.loads(sys.stdin.read()); print("step={}/{} loss={} rps={} eta_h={}".format(d.get("step"),d.get("total_steps"),d.get("loss"),round(d.get("rows_per_second") or 0,2),round(d.get("eta_hours") or 0,2)))' 2>/dev/null || true)
    fi
    echo "$TS pid=$PID ps=[$PS] rss_mb=$RSS_MB group_rss_mb=${CRSS:-?} free_mb=$FREE_MB inact_mb=$INACT_MB wired_mb=$WIRED_MB comp_mb=$COMP_MB mem_free_pct=${PRESS:-?} load=[$LOAD] $PROG" >> "$LOG"
  else
    echo "$TS train_not_running free_mb=$FREE_MB load=[$LOAD]" >> "$LOG"
  fi
  sleep 30
done
