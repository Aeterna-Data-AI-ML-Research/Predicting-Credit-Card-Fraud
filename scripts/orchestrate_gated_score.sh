#!/usr/bin/env bash
# Fast gated VA score orchestrator. NO_SUBMIT=True.
# Waits for pod SSH, kills leftover scorers, uploads scripts+preds, runs score_qwen_f0_gated.py,
# rsyncs checkpoints, early-kills pod on gate fail.
set -euo pipefail
SSD=/tmp/s6e11_v39_ssd
POD=zehvvv1vrbys16
LOG=$SSD/out/orchestrate_gated.log
mkdir -p "$SSD/out"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# Resolve SSH: prefer direct IP:port from runpodctl if available, else probe
resolve_ssh() {
  # try cached
  if [ -f "$SSD/out/SSH_CMD.txt" ]; then
    # shellcheck disable=SC1090
    source "$SSD/out/SSH_CMD.txt"
  fi
  # probe known direct
  if ssh -i ~/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -p "${SSH_PORT:-40603}" root@"${SSH_HOST:-38.65.239.32}" 'echo OK' 2>/dev/null | grep -q OK; then
    SSH_HOST=${SSH_HOST:-38.65.239.32}
    SSH_PORT=${SSH_PORT:-40603}
    export SSH=(ssh -i ~/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p "$SSH_PORT" root@"$SSH_HOST")
    export RSYNC_SSH="ssh -i ~/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT"
    echo "SSH_HOST=$SSH_HOST" > "$SSD/out/SSH_CMD.txt"
    echo "SSH_PORT=$SSH_PORT" >> "$SSD/out/SSH_CMD.txt"
    return 0
  fi
  return 1
}

log "orchestrator start NO_SUBMIT=True"
# disarm old watcher leftovers
pkill -f 'gate_1420_watcher' 2>/dev/null || true

# wait up to 20 min for SSH
for i in $(seq 1 80); do
  if resolve_ssh; then
    log "SSH ready ${SSH_HOST}:${SSH_PORT}"
    break
  fi
  log "waiting SSH… ($i/80) pod may be reinitializing"
  # refresh status
  /Users/mz/.local/bin/runpodctl pod list 2>/dev/null | head -c 400 || true
  echo
  sleep 15
  if [ "$i" -eq 80 ]; then
    log "SSH TIMEOUT"
    exit 3
  fi
done

log "verify adapter on /workspace"
"${SSH[@]}" 'set -e
ls -lh /workspace/out/qwen15b_lora_f0/adapter_model.safetensors
md5sum /workspace/out/qwen15b_lora_f0/adapter_model.safetensors
test -f /workspace/out/qwen15b_lora_f0/adapter_config.json
'

log "kill any leftover qwen/fold0 scorers (adapter already saved)"
"${SSH[@]}" 'pkill -f "run_qwen15b_fold0_storyteller" 2>/dev/null || true
pkill -f "score_qwen" 2>/dev/null || true
sleep 2
pgrep -af python || echo no_python
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
'

log "upload scripts + preds"
"${SSH[@]}" 'mkdir -p /workspace/scripts /workspace/preds /workspace/out /workspace/logs'
scp -i ~/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -P "$SSH_PORT" \
  "$SSD/scripts/run_qwen15b_fold0_storyteller.py" \
  "$SSD/scripts/score_qwen_f0_gated.py" \
  root@"$SSH_HOST":/workspace/scripts/
scp -i ~/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -P "$SSH_PORT" \
  "$SSD/preds/z_fullbank_oof.npy" \
  "$SSD/preds/z_ATH_twin_oof.npy" \
  root@"$SSH_HOST":/workspace/preds/

log "launch score_qwen_f0_gated.py bs=512"
"${SSH[@]}" 'rm -f /workspace/out/STAGE1_K2_DONE.json /workspace/out/GATE_EARLY_KILL.json /workspace/out/GATE_FAIL.json /workspace/out/GATE_PASS.json /workspace/out/TEST_DONE.json
nohup python -u /workspace/scripts/score_qwen_f0_gated.py --bs 512 \
  > /workspace/logs/gated.log 2>&1 &
echo $! > /workspace/logs/gated.pid
cat /workspace/logs/gated.pid
sleep 3
head -20 /workspace/logs/gated.log
'

# poll stages
for i in $(seq 1 360); do
  state=$("${SSH[@]}" 'ls /workspace/out/STAGE1_K2_DONE.json /workspace/out/GATE_EARLY_KILL.json /workspace/out/STAGE2_K6_DONE.json /workspace/out/GATE_FAIL.json /workspace/out/GATE_PASS.json /workspace/out/TEST_DONE.json 2>/dev/null; tail -3 /workspace/logs/gated.log; ps -p $(cat /workspace/logs/gated.pid) >/dev/null 2>&1 && echo ALIVE || echo DEAD' 2>/dev/null || true)
  log "poll $i: $(echo "$state" | tr '\n' ' | ')"

  if echo "$state" | grep -q STAGE1_K2_DONE; then
    if [ ! -f "$SSD/out/.pulled_k2" ]; then
      log "rsync ll_va_k2 BEFORE gate decision"
      rsync -avz -e "$RSYNC_SSH" --no-owner --no-group \
        root@"$SSH_HOST":/workspace/out/ll_va_k2.npy \
        root@"$SSH_HOST":/workspace/out/qwen15b_f0_ll_va_k2.npy \
        root@"$SSH_HOST":/workspace/out/gate_k2.json \
        root@"$SSH_HOST":/workspace/out/STAGE1_K2_DONE.json \
        "$SSD/out/" || true
      touch "$SSD/out/.pulled_k2"
      log "K2 pulled; gate_k2=$(cat $SSD/out/gate_k2.json 2>/dev/null | head -c 400)"
    fi
  fi

  if echo "$state" | grep -q GATE_EARLY_KILL; then
    log "EARLY KILL — rsync logs then remove pod"
    rsync -avz -e "$RSYNC_SSH" --no-owner --no-group \
      root@"$SSH_HOST":/workspace/out/gate_k2.json \
      root@"$SSH_HOST":/workspace/out/GATE_EARLY_KILL.json \
      root@"$SSH_HOST":/workspace/out/ll_va_k2.npy \
      root@"$SSH_HOST":/workspace/logs/gated.log \
      "$SSD/out/" || true
    /Users/mz/.local/bin/runpodctl remove pod "$POD" || true
    log "pod removed (early kill)"
    exit 2
  fi

  if echo "$state" | grep -q STAGE2_K6_DONE; then
    if [ ! -f "$SSD/out/.pulled_k6" ]; then
      log "rsync ll_va_k6 BEFORE full gate decision"
      rsync -avz -e "$RSYNC_SSH" --no-owner --no-group \
        root@"$SSH_HOST":/workspace/out/ll_va_k6.npy \
        root@"$SSH_HOST":/workspace/out/qwen15b_f0_ll.npy \
        root@"$SSH_HOST":/workspace/out/qwen15b_f0_ll_va.npy \
        root@"$SSH_HOST":/workspace/out/gate_k6.json \
        root@"$SSH_HOST":/workspace/out/STAGE2_K6_DONE.json \
        "$SSD/out/" || true
      touch "$SSD/out/.pulled_k6"
    fi
  fi

  if echo "$state" | grep -q GATE_FAIL; then
    log "FULL GATE FAIL — rsync + remove pod"
    rsync -avz -e "$RSYNC_SSH" --no-owner --no-group \
      root@"$SSH_HOST":/workspace/out/ \
      "$SSD/out/pod_out_fail/" || true
    /Users/mz/.local/bin/runpodctl remove pod "$POD" || true
    exit 1
  fi

  if echo "$state" | grep -q TEST_DONE; then
    log "TEST DONE — rsync (NO_SUBMIT)"
    rsync -avz -e "$RSYNC_SSH" --no-owner --no-group \
      root@"$SSH_HOST":/workspace/out/ll_te_k6.npy \
      root@"$SSH_HOST":/workspace/out/qwen15b_f0_ll_test.npy \
      root@"$SSH_HOST":/workspace/out/GATE_PASS.json \
      root@"$SSH_HOST":/workspace/out/TEST_DONE.json \
      "$SSD/out/" || true
    log "complete NO_SUBMIT — pod kept"
    exit 0
  fi

  if echo "$state" | grep -q DEAD && ! echo "$state" | grep -qE 'GATE_|TEST_DONE|STAGE'; then
    # may still be starting
    if [ "$i" -gt 5 ]; then
      log "gated process dead unexpectedly"
      "${SSH[@]}" 'tail -40 /workspace/logs/gated.log' || true
      exit 4
    fi
  fi
  sleep 20
done
log "orchestrator timeout"
exit 5
