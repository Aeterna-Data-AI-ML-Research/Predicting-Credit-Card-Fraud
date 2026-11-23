#!/usr/bin/env bash
# Deploy fast gated scorer onto new pod. NO_SUBMIT=True.
set -euo pipefail
SSD=/tmp/s6e11_v39_ssd
POD=${POD:-gl54251co7nhkh}
LOG=$SSD/out/deploy_gated.log
mkdir -p "$SSD/out"
exec > >(tee -a "$LOG") 2>&1
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

log "deploy start pod=$POD cost=\$0.89/hr Secure 4090"

# resolve direct SSH from runpodctl / polling get
SSH_HOST=""; SSH_PORT=""
for i in $(seq 1 90); do
  # parse from runpodctl ssh if available
  info=$(/Users/mz/.local/bin/runpodctl pod list 2>/dev/null || true)
  # try MCP-less: curl not available; use python + stored
  # Probe: runpodctl may print ip in older CLI — fallback poll known patterns via ssh proxy later
  if [ -f "$SSD/out/NEW_SSH.txt" ]; then
    # shellcheck disable=SC1090
    source "$SSD/out/NEW_SSH.txt" || true
  fi
  if [ -n "${SSH_HOST:-}" ] && [ -n "${SSH_PORT:-}" ]; then
    if ssh -i ~/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -p "$SSH_PORT" root@"$SSH_HOST" 'echo OK' 2>/dev/null | grep -q OK; then
      log "SSH ready $SSH_HOST:$SSH_PORT"
      break
    fi
  fi
  log "waiting direct SSH ($i/90)…"
  sleep 10
  if [ "$i" -eq 90 ]; then log "SSH timeout"; exit 3; fi
done

SSH=(ssh -i ~/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -p "$SSH_PORT" root@"$SSH_HOST")
RSYNC_SSH="ssh -i ~/.ssh/casmi_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT"

log "prepare dirs + pip peft"
"${SSH[@]}" 'mkdir -p /workspace/{data,folds,preds,out,scripts,logs,hf}
pip install -q peft --break-system-packages 2>/dev/null || pip install -q peft
nvidia-smi -L
'

log "upload data/folds/preds/scripts"
rsync -az -e "$RSYNC_SSH" "$SSD/data/train.csv" "$SSD/data/test.csv" root@"$SSH_HOST":/workspace/data/
rsync -az -e "$RSYNC_SSH" "$SSD/folds/folds.csv" root@"$SSH_HOST":/workspace/folds/
rsync -az -e "$RSYNC_SSH" "$SSD/preds/z_fullbank_oof.npy" "$SSD/preds/z_ATH_twin_oof.npy" root@"$SSH_HOST":/workspace/preds/
rsync -az -e "$RSYNC_SSH" \
  "$SSD/scripts/run_qwen15b_fold0_storyteller.py" \
  "$SSD/scripts/score_qwen_f0_gated.py" \
  root@"$SSH_HOST":/workspace/scripts/

log "upload LoRA adapter (141MB from Mac backup)"
rsync -az --progress -e "$RSYNC_SSH" "$SSD/out/qwen15b_lora_f0/" root@"$SSH_HOST":/workspace/out/qwen15b_lora_f0/
"${SSH[@]}" 'ls -lh /workspace/out/qwen15b_lora_f0/adapter_model.safetensors; md5sum /workspace/out/qwen15b_lora_f0/adapter_model.safetensors'

log "launch score_qwen_f0_gated.py"
"${SSH[@]}" 'rm -f /workspace/out/{STAGE1_K2_DONE,GATE_EARLY_KILL,STAGE2_K6_DONE,GATE_FAIL,GATE_PASS,TEST_DONE}.json /workspace/out/.pulled_* 2>/dev/null
nohup python -u /workspace/scripts/score_qwen_f0_gated.py --bs 512 \
  > /workspace/logs/gated.log 2>&1 &
echo $! > /workspace/logs/gated.pid
sleep 4
head -30 /workspace/logs/gated.log
cat /workspace/logs/gated.pid
'

# reuse orchestrate poll logic inline
rm -f "$SSD/out/.pulled_k2" "$SSD/out/.pulled_k6"
for i in $(seq 1 360); do
  state=$("${SSH[@]}" 'ls /workspace/out/STAGE1_K2_DONE.json /workspace/out/GATE_EARLY_KILL.json /workspace/out/STAGE2_K6_DONE.json /workspace/out/GATE_FAIL.json /workspace/out/GATE_PASS.json /workspace/out/TEST_DONE.json 2>/dev/null; tail -2 /workspace/logs/gated.log; ps -p $(cat /workspace/logs/gated.pid 2>/dev/null) >/dev/null 2>&1 && echo ALIVE || echo DEAD' 2>/dev/null || true)
  log "poll $i: $(echo "$state" | tr '\n' ' | ')"

  if echo "$state" | grep -q STAGE1_K2_DONE && [ ! -f "$SSD/out/.pulled_k2" ]; then
    log "rsync K2 BEFORE gate"
    rsync -az -e "$RSYNC_SSH" root@"$SSH_HOST":/workspace/out/ll_va_k2.npy root@"$SSH_HOST":/workspace/out/gate_k2.json root@"$SSH_HOST":/workspace/out/STAGE1_K2_DONE.json "$SSD/out/" || true
    touch "$SSD/out/.pulled_k2"
    python3 -c "import json;print(json.load(open('$SSD/out/gate_k2.json')))" 2>/dev/null || true
  fi
  if echo "$state" | grep -q GATE_EARLY_KILL; then
    log "EARLY KILL — rsync + terminate $POD"
    rsync -az -e "$RSYNC_SSH" root@"$SSH_HOST":/workspace/out/gate_k2.json root@"$SSH_HOST":/workspace/out/GATE_EARLY_KILL.json root@"$SSH_HOST":/workspace/out/ll_va_k2.npy root@"$SSH_HOST":/workspace/logs/gated.log "$SSD/out/" || true
    /Users/mz/.local/bin/runpodctl remove pod "$POD" || true
    exit 2
  fi
  if echo "$state" | grep -q STAGE2_K6_DONE && [ ! -f "$SSD/out/.pulled_k6" ]; then
    log "rsync K6 BEFORE full gate"
    rsync -az -e "$RSYNC_SSH" root@"$SSH_HOST":/workspace/out/ll_va_k6.npy root@"$SSH_HOST":/workspace/out/qwen15b_f0_ll.npy root@"$SSH_HOST":/workspace/out/gate_k6.json "$SSD/out/" || true
    touch "$SSD/out/.pulled_k6"
  fi
  if echo "$state" | grep -q GATE_FAIL; then
    log "FULL GATE FAIL — rsync + terminate"
    rsync -az -e "$RSYNC_SSH" root@"$SSH_HOST":/workspace/out/ "$SSD/out/pod_fail/" || true
    /Users/mz/.local/bin/runpodctl remove pod "$POD" || true
    exit 1
  fi
  if echo "$state" | grep -q TEST_DONE; then
    log "TEST DONE NO_SUBMIT — rsync keep pod"
    rsync -az -e "$RSYNC_SSH" root@"$SSH_HOST":/workspace/out/ll_te_k6.npy root@"$SSH_HOST":/workspace/out/qwen15b_f0_ll_test.npy root@"$SSH_HOST":/workspace/out/GATE_PASS.json "$SSD/out/" || true
    exit 0
  fi
  sleep 20
done
log timeout; exit 5
