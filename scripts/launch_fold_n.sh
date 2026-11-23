#!/bin/bash
# Launch fold N train+score (reuse same worker --fold N)
set -euo pipefail
FOLD="${1:?fold number}"
cd /workspace
export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1
LOG=/workspace/logs/fold${FOLD}.log
PIDF=/workspace/logs/fold${FOLD}.pid
rm -f "$LOG"
nohup python -u /workspace/scripts/run_qwen15b_fold0_storyteller.py \
  --fold "$FOLD" --mode fold_train_score \
  --data_dir /workspace/data \
  --folds_csv /workspace/folds/folds.csv \
  --out_dir /workspace/out \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "LAUNCHED fold=$FOLD pid=$(cat "$PIDF")"
sleep 2
ps -p "$(cat "$PIDF")" -o pid,etime,cmd || true
