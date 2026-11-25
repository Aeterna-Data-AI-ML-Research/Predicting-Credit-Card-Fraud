#!/usr/bin/env bash
# Pod-side Fold-0 runner: loops until meta_f0.json exists; safe to re-SSH and re-run.
# Watchdog: 21600s. NO TabICL. NO Kaggle.
set -euo pipefail
export V43_LAUNCH=1
export V43_K_PAIRS="${V43_K_PAIRS:-2}"
export V43_USE_COMPILE=0
export V43_SKIP_TABICL=1
export V43_EPOCHS="${V43_EPOCHS:-1}"   # canary default 1 epoch (budget)
export V43_CKPT_EVERY="${V43_CKPT_EVERY:-200}"
export V43_SCORE_FLUSH="${V43_SCORE_FLUSH:-64}"
export TORCH_COMPILE_DISABLE=1
echo "[$(date -u +%H:%M:%S)] env EPOCHS=$V43_EPOCHS K_PAIRS=$V43_K_PAIRS CKPT_EVERY=$V43_CKPT_EVERY" | tee -a "${OUT:-/workspace/out/qwen}/run.log"
DATA="${DATA:-/workspace/data}"
OUT="${OUT:-/workspace/out/qwen}"
PKG="${PKG:-/workspace/pkg}"
LOG="${OUT}/run.log"
mkdir -p "$OUT" "$PKG"
cd "$PKG"
echo "[$(date -u +%H:%M:%S)] resumable Fold-0 start OUT=$OUT" | tee -a "$LOG"
WATCHDOG="${V43_WATCHDOG_S:-21600}"
LOCK=/workspace/out/qwen/run.lock
mkdir -p "$(dirname "$LOCK")"
# sole survivor: kill other Fold-0 runners/trainers (path-specific; never pkill broad patterns)
MY=$$
for p in $(pgrep -f '/workspace/pkg/run_fold0_resumable.sh' || true); do
  [[ "$p" == "$MY" ]] && continue
  kill -9 "$p" 2>/dev/null || true
done
for p in $(pgrep -f '/workspace/pkg/qwen_digit_genbayes_24llr.py' || true); do
  kill -9 "$p" 2>/dev/null || true
done
sleep 2
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -u +%H:%M:%S)] another Fold-0 runner holds $LOCK — exit" | tee -a "$LOG"
  exit 0
fi
echo "[$(date -u +%H:%M:%S)] acquired $LOCK pid=$MY" | tee -a "$LOG"
# venv (PEP 668) — pytorch image already has torch
VENV="${VENV:-/workspace/venv}"
PY="$VENV/bin/python"
if [[ ! -x "$PY" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "transformers>=4.44" peft accelerate scikit-learn pandas pyarrow 2>&1 | tee -a "$LOG"
  # torch already in system; expose it into venv via --system-site-packages recreate if needed
fi
# ensure venv can see system torch
if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  rm -rf "$VENV"
  python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q "transformers>=4.44" peft accelerate scikit-learn pandas pyarrow 2>&1 | tee -a "$LOG"
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" | tee -a "$LOG"
touch /workspace/.deps_ok
# retry loop — re-enter after kill/OOM/SSH drop
ATTEMPT=0
while [[ ! -f "$OUT/meta_f0.json" ]]; do
  ATTEMPT=$((ATTEMPT+1))
  echo "[$(date -u +%H:%M:%S)] attempt=$ATTEMPT phase=$(cat $OUT/ckpts/f0/state.json 2>/dev/null || echo none)" | tee -a "$LOG"
  set +e
  timeout "$WATCHDOG" "$PY" -u qwen_digit_genbayes_24llr.py --data "$DATA" --out "$OUT" --fold 0 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
  set -e
  echo "[$(date -u +%H:%M:%S)] exit_rc=$RC" | tee -a "$LOG"
  if [[ -f "$OUT/meta_f0.json" ]]; then
    break
  fi
  if [[ $RC -eq 124 ]]; then
    echo "WATCHDOG fired — leave ckpts; do not delete pod until rsync" | tee -a "$LOG"
    exit 124
  fi
  sleep 15
done
echo "[$(date -u +%H:%M:%S)] DONE meta_f0 present" | tee -a "$LOG"
ls -la "$OUT"/*.npy "$OUT"/meta_f0.json | tee -a "$LOG"
