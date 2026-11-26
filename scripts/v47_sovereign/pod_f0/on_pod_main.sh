#!/usr/bin/env bash
# V47 Fold-0 on-pod orchestrator: HGDT-LM then Qwen field24 LLR.
# Dead-man: sleep 16200 && remove THIS pod only. NO_SUBMIT. Never touch other pods.
set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export V43_USE_COMPILE=0
export V43_LAUNCH=1
export V43_FOLD=0
export V43_EPOCHS="${V43_EPOCHS:-4}"
export V43_K_PAIRS=2
export V43_SKIP_TABICL=1
export HF_HOME="${HF_HOME:-/workspace/hf}"
export TRANSFORMERS_CACHE="$HF_HOME"

REPO="${REPO:-/workspace/v47}"
DATA="${DATA:-/workspace/data}"
OUT="${OUT:-/workspace/preds/v47_pod_f0}"
mkdir -p "$OUT" /workspace/logs/v47

POD_ID="${RUNPOD_POD_ID:-${RUNPOD_POD_ID_ENV:-}}"
if [[ -z "${POD_ID}" && -f /etc/runpod-info ]]; then
  POD_ID="$(grep -oE '[a-z0-9]{14,}' /etc/runpod-info | head -1 || true)"
fi
echo "[v47] POD_ID=${POD_ID:-UNKNOWN} OUT=$OUT" | tee /workspace/logs/v47/pod_boot.txt

# Dead-man timer — 4.5h hard cap (~$2.7 @ $0.59/hr A40). Remove ONLY this pod.
if [[ -n "${POD_ID}" ]] && command -v runpodctl >/dev/null 2>&1; then
  ( sleep 16200 && echo "[deadman] removing $POD_ID" && runpodctl remove pod "$POD_ID" ) &
  echo $! > /workspace/logs/v47/deadman.pid
  echo "[v47] dead-man armed pid=$(cat /workspace/logs/v47/deadman.pid) for $POD_ID"
else
  echo "[v47] WARN: no POD_ID/runpodctl — dead-man not armed"
fi

cleanup() {
  echo "[v47] cleanup start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "$OUT"/*.npy 2>/dev/null | tee /workspace/logs/v47/SHA256_POD.txt || true
  ls -la "$OUT" | tee /workspace/logs/v47/OUT_LS.txt || true
}
trap cleanup EXIT

# Sync expected data
if [[ ! -f "$DATA/train.csv" ]]; then
  echo "FATAL: missing $DATA/train.csv" >&2
  exit 2
fi

# Phase HGDT
python3 "$REPO/scripts/v47_sovereign/pod_f0/hgdt_lm_fold0.py" \
  --data "$DATA" --out "$OUT" \
  --pretrain-steps "${HGDT_PRETRAIN_STEPS:-4000}" \
  --finetune-epochs "${HGDT_FINETUNE_EPOCHS:-2}" \
  --bs "${HGDT_BS:-48}" \
  --k-pairs 6 --fold 0 --device cuda \
  2>&1 | tee /workspace/logs/v47/HGDT_F0.log

# Phase Qwen (proven V43 trainer → V47 names)
export OUT_DIR="$OUT"
bash "$REPO/scripts/v47_sovereign/pod_f0/qwen_f0_runner.sh" "$REPO" "$DATA" "$OUT" \
  2>&1 | tee /workspace/logs/v47/QWEN_F0.log

# Manifest
python3 - <<'PY'
import json, time, hashlib
from pathlib import Path
out = Path("/workspace/preds/v47_pod_f0")
files = sorted(out.glob("*.npy"))
man = {"ts": time.time(), "NO_SUBMIT": True, "files": {}}
for p in files:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    man["files"][p.name] = {"sha256": h, "bytes": p.stat().st_size}
Path("/workspace/logs/v47/POD_MANIFEST.json").write_text(json.dumps(man, indent=2))
print(json.dumps(man, indent=2))
PY

echo "[v47] ALL PHASES DONE — ready for rsync"
