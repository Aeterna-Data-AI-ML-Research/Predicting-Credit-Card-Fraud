#!/usr/bin/env bash
# Hard cost circuit-breaker: terminate THIS pod after 510 minutes (8.5h ≈ $5.01 @ $0.59/hr).
# Never touches foreign pods. Requires RUNPOD_API_KEY + RUNPOD_POD_ID in env.
set -euo pipefail
MINUTES="${WATCHDOG_MINUTES:-510}"
POD_ID="${RUNPOD_POD_ID:-}"
API_KEY="${RUNPOD_API_KEY:-}"
BLACKLIST="lheroidmuab1j4"

if [[ -z "$POD_ID" || -z "$API_KEY" ]]; then
  echo "[watchdog] missing RUNPOD_POD_ID or RUNPOD_API_KEY — refusing to arm" >&2
  exit 1
fi
if [[ "$POD_ID" == *"$BLACKLIST"* ]]; then
  echo "[watchdog] REFUSING blacklisted pod $POD_ID" >&2
  exit 2
fi

echo "[watchdog] armed: will terminate pod=$POD_ID after ${MINUTES}m"
sleep $((MINUTES * 60))
echo "[watchdog] TIMER FIRED — terminating $POD_ID"
# Prefer runpodctl if present; else REST v2
if command -v runpodctl >/dev/null 2>&1; then
  runpodctl delete pod "$POD_ID" || true
else
  curl -sS -X DELETE "https://api.runpod.io/v2/pods/${POD_ID}" \
    -H "Authorization: Bearer ${API_KEY}" || true
fi
echo "[watchdog] terminate requested"
