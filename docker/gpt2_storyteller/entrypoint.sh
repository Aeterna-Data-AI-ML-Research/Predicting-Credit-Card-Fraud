#!/usr/bin/env bash
# Stages: replica → fold classifiers → score oof/test → make_predictions
set -euo pipefail
CODE="${CODE:-/workspace/code}"
DATA="${DATA:-/workspace/data}"
ORIG="${ORIG:-/workspace/original/credit_card_fraud_2026.csv}"
RUN="${RUN:-/workspace/run}"
STAGE="${1:-help}"
shift || true

export PYTHONPATH="$CODE:${PYTHONPATH:-}"
cd "$CODE"
mkdir -p "$RUN"

PY=python
NPROC="${NPROC:-8}"
export OMP_NUM_THREADS="$NPROC" MKL_NUM_THREADS="$NPROC"

echo "[entrypoint] stage=$STAGE device=cpu threads=$NPROC run=$RUN"

case "$STAGE" in
  smoke)
    # tiny end-to-end sanity (minutes)
    $PY make_folds.py --train-csv "$DATA/train.csv" --target-col is_fraud --out "$RUN/fold_ids.npy"
    $PY train_replica.py --original-csv "$ORIG" --id-col transaction_id --target-col is_fraud \
      --out "$RUN/replica" --model gpt2 --epochs 1 --batch-size 4 --max-rows 500 --precision fp32 \
      --save-steps 50 --log-steps 10 --stop-after-steps 20 || true
    echo "smoke done (stop-after ok)"
    ;;
  folds)
    $PY make_folds.py --train-csv "$DATA/train.csv" --target-col is_fraud --out "$RUN/fold_ids.npy"
    ;;
  replica)
    EPOCHS="${EPOCHS:-60}"
    BATCH="${BATCH:-8}"
    $PY train_replica.py --original-csv "$ORIG" --id-col transaction_id --target-col is_fraud \
      --out "$RUN/replica" --model gpt2 --epochs "$EPOCHS" --batch-size "$BATCH" --precision fp32 \
      --save-steps "${SAVE_STEPS:-2000}" --log-steps "${LOG_STEPS:-50}" --seed "${SEED:-42}"
    ;;
  classify)
    FOLD="${FOLD:?set FOLD=0..4}"
    EPOCHS="${EPOCHS:-8}"
    BATCH="${BATCH:-8}"
    REPLICA="${REPLICA:-$RUN/replica/final}"
    $PY train_classifier.py --train-csv "$DATA/train.csv" --fold-ids "$RUN/fold_ids.npy" \
      --fold "$FOLD" --id-col id --target-col is_fraud --replica "$REPLICA" \
      --out "$RUN/fits/f${FOLD}" --epochs "$EPOCHS" --batch-size "$BATCH" --precision fp32 \
      --device cpu --seed "${SEED:-42}" --save-every-steps "${SAVE_STEPS:-2000}" \
      --log-every-steps "${LOG_STEPS:-50}"
    ;;
  score-oof)
    FOLD="${FOLD:?}"
    ORDERS="${ORDERS:-30}"
    $PY score_rows.py --model-dir "$RUN/fits/f${FOLD}/final" --csv "$DATA/train.csv" \
      --split oof --fold-ids "$RUN/fold_ids.npy" --fold "$FOLD" --id-col id --target-col is_fraud \
      --out "$RUN/scores/f${FOLD}_oof" --orders "$ORDERS" --seed "${SEED:-42}" \
      --precision fp32 --device cpu --chunk-rows 500
    ;;
  score-test)
    FOLD="${FOLD:?}"
    ORDERS="${ORDERS:-10}"
    $PY score_rows.py --model-dir "$RUN/fits/f${FOLD}/final" --csv "$DATA/test.csv" \
      --split test --id-col id --target-col is_fraud \
      --out "$RUN/scores/f${FOLD}_test" --orders "$ORDERS" --seed "${SEED:-42}" \
      --precision fp32 --device cpu --chunk-rows 500
    ;;
  predict)
    $PY make_predictions.py --scores "$RUN/scores" --train-csv "$DATA/train.csv" \
      --test-csv "$DATA/test.csv" --fold-ids "$RUN/fold_ids.npy" --target-col is_fraud \
      --out "$RUN/preds"
    ;;
  pipeline)
    # Full recipe: replica → fold0 first (early read) → remaining folds
    $0 folds
    $0 replica
    for FOLD in 0 1 2 3 4; do
      export FOLD
      $0 classify
      $0 score-oof
      $0 score-test
      if [[ "$FOLD" == "0" ]]; then
        $PY make_predictions.py --scores "$RUN/scores" --train-csv "$DATA/train.csv" \
          --test-csv "$DATA/test.csv" --fold-ids "$RUN/fold_ids.npy" --target-col is_fraud \
          --out "$RUN/early_f0" --allow-partial || true
        echo "[entrypoint] fold0 early read written to $RUN/early_f0"
      fi
    done
    $0 predict
    ;;
  help|*)
    cat <<EOF
Usage: docker run ... <stage>
  smoke | folds | replica | classify | score-oof | score-test | predict | pipeline
Env: EPOCHS BATCH FOLD SEED ORDERS NPROC REPLICA SAVE_STEPS LOG_STEPS
EOF
    ;;
esac
