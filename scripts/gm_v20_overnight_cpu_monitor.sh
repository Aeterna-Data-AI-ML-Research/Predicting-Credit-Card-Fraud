#!/bin/bash
# CPU/memory monitor for overnight factory
LOG="/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/logs/overnight/GM_V20_OVERNIGHT_CPU_MONITOR.log"
PID_FILE="/tmp/s6e11_v16/out/gm_overnight_factory.pid"
echo "monitor_start $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG"
while true; do
  TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
  else
    PID=$(pgrep -f 'gm_v20_overnight_cpu_grandmaster_factory.py' | head -1)
  fi
  if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
    echo "$TS factory_not_running" >> "$LOG"
    # keep monitoring a bit in case it restarts; exit if gone > 2 min
    sleep 30
    PID=$(pgrep -f 'gm_v20_overnight_cpu_grandmaster_factory.py' | head -1)
    if [[ -z "$PID" ]]; then
      echo "$TS monitor_exit factory_done" >> "$LOG"
      exit 0
    fi
  fi
  # process RSS/CPU
  PS=$(ps -o pid=,pcpu=,pmem=,rss=,etime= -p "$PID" 2>/dev/null | head -1)
  # system load + memory
  LOAD=$(sysctl -n vm.loadavg 2>/dev/null || uptime)
  MEM=$(vm_stat 2>/dev/null | awk '/Pages (free|active|inactive|wired|occupied by compressor)/{gsub("\\.","",$NF); print $1,$NF}')
  FREE_MB=$(vm_stat 2>/dev/null | awk '/Pages free/{gsub("\\.","",$3); print int($3*4096/1024/1024)}')
  echo "$TS pid=$PID ps=[$PS] load=[$LOAD] free_mb=$FREE_MB" >> "$LOG"
  sleep 30
done
