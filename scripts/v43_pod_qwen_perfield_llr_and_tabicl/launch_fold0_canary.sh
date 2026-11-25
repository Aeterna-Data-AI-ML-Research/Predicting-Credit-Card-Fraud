#!/usr/bin/env bash
# V43 Fold-0 Canary — Qwen2.5-0.5B (N,24) Per-Field LLR ONLY (TabICL STRIPPED).
# Gemini token: APPROVE_POD_F0_ONLY. Still requires explicit human LAUNCH F0.
# NO_SUBMIT forever inside this script. 0 Kaggle calls.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG="$(cd "$(dirname "$0")" && pwd)"
echo "NO_SUBMIT=True"
echo "MODULE=Qwen2.5-0.5B field24 LLR ONLY (TabICL disabled)"
echo "This script refuses to create pods unless:"
echo "  V43_AUTHORIZE_POD=YES  AND  V43_LAUNCH_F0=YES"
if [[ "${V43_AUTHORIZE_POD:-}" != "YES" || "${V43_LAUNCH_F0:-}" != "YES" ]]; then
  echo "HARD STOP: awaiting explicit user phrase LAUNCH F0."
  echo "Prep checklist (already applied in package):"
  echo "  [x] TabICL removed from boot path"
  echo "  [x] K_PAIRS=2 → K=4 antithetic (matched oof==test)"
  echo "  [x] F.cross_entropy reduction=none (no full-vocab log_softmax)"
  echo "  [x] USE_COMPILE=0 · attn_implementation=sdpa · bf16"
  echo "  [x] y0/y1 prefix length assert + prefix/pad mask on score"
  echo "  [x] post-rsync local gate: gate_fold0_field24_local.py (≥+5e-5, ≥4/5)"
  echo "Suggested GPU (live catalog — re-quote at LAUNCH):"
  echo "  Secure RTX 4090  \$0.89/hr (LOW stock typical)"
  echo "  Secure A40       \$0.59/hr"
  echo "  Secure A100 SXM  \$2.19/hr"
  echo "Est. Fold-0 wall Qwen-only: ~2.5–3.5h on 4090 → ~\$2.2–\$3.1"
  echo "Watchdog: 21600s. rsync .npy BEFORE gate / remove pod."
  exit 0
fi

# ---- authorized path (still no auto-create without RUNPOD_DO_CREATE=1) ----
export V43_LAUNCH=1
export V43_K_PAIRS="${V43_K_PAIRS:-2}"
export V43_USE_COMPILE=0
export V43_SKIP_TABICL=1
WATCHDOG_S="${V43_WATCHDOG_S:-21600}"

echo "AUTHORIZATION DETECTED."
echo "Snapshot foreign pods first:"
runpodctl pod list || true
if [[ "${RUNPOD_DO_CREATE:-}" != "1" ]]; then
  echo "Refusing auto create-pod. Set RUNPOD_DO_CREATE=1 after quoting \$/hr to user."
  echo "On pod (after sync of $PKG + data):"
  echo "  export V43_LAUNCH=1 V43_K_PAIRS=2 V43_USE_COMPILE=0"
  echo "  timeout ${WATCHDOG_S} python qwen_digit_genbayes_24llr.py --data /data --out /out/qwen --fold 0"
  echo "  # DO NOT run tabicl_v2_fullcontext.py"
  echo "Then on Mac: rsync /out/qwen/*.npy → out/v43_fold0_qwen/ ; run gate_fold0_field24_local.py"
  exit 0
fi

echo "RUNPOD_DO_CREATE=1 — create-pod must be issued by the agent with quoted price; this shell does not invent a pod id."
exit 1
