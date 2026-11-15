#!/usr/bin/env bash
# Local M3 Pro MPS: full GPT-2 + orig20k replica → fold0 canary.
# Resume-safe: Trainer checkpoints (replica) + resume.pt (classifier); re-run is idempotent.
set -euo pipefail
PY=/tmp/s6e11_mps_venv/bin/python
CODE=/tmp/s6e11_v16/lm_research/matapanino
ROOT=/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud
RUN=/tmp/s6e11_v16/lm_research/run_mps_gpt2
ORIG="$ROOT/original/credit_card_fraud_2026.csv"
DATA="$ROOT/data"
PID_FILE="$RUN/pipeline.pid"
DONE_FILE="$RUN/DONE"

export TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

REP_EPOCHS="${REP_EPOCHS:-20}"
CLS_EPOCHS="${CLS_EPOCHS:-2}"
BATCH="${BATCH:-8}"
SEED="${SEED:-42}"
ORDERS="${ORDERS:-30}"
# Frequent checkpoints so a crash loses ≤ ~SAVE_STEPS of work
SAVE_STEPS="${SAVE_STEPS:-500}"
LOG_STEPS="${LOG_STEPS:-50}"

mkdir -p "$RUN" "$RUN/fits" "$RUN/scores"
echo $$ > "$PID_FILE"
cd "$CODE"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
exec >> "$RUN/pipeline.log" 2>&1

log "MPS GPT-2+replica start pid=$$ REP_EPOCHS=$REP_EPOCHS CLS_EPOCHS=$CLS_EPOCHS BATCH=$BATCH SAVE_STEPS=$SAVE_STEPS"

if [[ ! -f "$RUN/fold_ids.npy" ]]; then
  $PY make_folds.py --train-csv "$DATA/train.csv" --target-col is_fraud --out "$RUN/fold_ids.npy"
fi

# --- Replica (resumes from trainer/checkpoint-* if present) ---
if [[ ! -f "$RUN/replica/train_summary.json" ]]; then
  log "=== REPLICA gpt2 ${REP_EPOCHS}ep (resume-capable) ==="
  # show newest ckpt if any
  if ls "$RUN/replica/trainer"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
    log "resuming replica from $(ls -d "$RUN/replica/trainer"/checkpoint-* | sort -V | tail -1)"
  fi
  $PY train_replica.py --original-csv "$ORIG" --id-col transaction_id --target-col is_fraud \
    --out "$RUN/replica" --model gpt2 --epochs "$REP_EPOCHS" --batch-size "$BATCH" \
    --precision bf16 --save-steps "$SAVE_STEPS" --log-steps "$LOG_STEPS" --seed "$SEED"
else
  log "replica already complete"
fi

# --- Fold0 classify (resumes from fits/f0/resume.pt) ---
FOLD=0
if [[ ! -f "$RUN/fits/f${FOLD}/train_summary.json" ]]; then
  log "=== CLASSIFY fold$FOLD ${CLS_EPOCHS}ep (resume-capable) ==="
  if [[ -f "$RUN/fits/f${FOLD}/resume.pt" ]]; then
    log "resuming classifier from resume.pt"
  fi
  $PY train_classifier.py --train-csv "$DATA/train.csv" --fold-ids "$RUN/fold_ids.npy" \
    --fold "$FOLD" --id-col id --target-col is_fraud --replica "$RUN/replica/final" \
    --out "$RUN/fits/f${FOLD}" --epochs "$CLS_EPOCHS" --batch-size "$BATCH" \
    --precision bf16 --device mps --seed "$SEED" \
    --save-every-steps "$SAVE_STEPS" --log-every-steps "$LOG_STEPS"
else
  log "fold$FOLD classify already complete"
fi

# --- Score OOF (chunked; re-run skips finished chunks inside score_rows) ---
if [[ ! -f "$RUN/early_f0/manifest.json" ]]; then
  log "=== SCORE OOF fold$FOLD orders=$ORDERS ==="
  $PY score_rows.py --model-dir "$RUN/fits/f${FOLD}/final" --csv "$DATA/train.csv" \
    --split oof --fold-ids "$RUN/fold_ids.npy" --fold "$FOLD" --id-col id --target-col is_fraud \
    --out "$RUN/scores/f${FOLD}_oof" --orders "$ORDERS" --seed "$SEED" \
    --precision fp32 --device mps --chunk-rows 1000

  $PY make_predictions.py --scores "$RUN/scores" --train-csv "$DATA/train.csv" \
    --test-csv "$DATA/test.csv" --fold-ids "$RUN/fold_ids.npy" --target-col is_fraud \
    --out "$RUN/early_f0" --allow-partial || true
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$DONE_FILE"
log "DONE early fold0 → $RUN/early_f0"
if [[ -f "$RUN/early_f0/manifest.json" ]]; then
  $PY -c 'import json;from pathlib import Path;m=json.loads(Path("/tmp/s6e11_v16/lm_research/run_mps_gpt2/early_f0/manifest.json").read_text());print("manifest",m.get("auc"),m.get("per_fold"))'
fi
rm -f "$PID_FILE"
