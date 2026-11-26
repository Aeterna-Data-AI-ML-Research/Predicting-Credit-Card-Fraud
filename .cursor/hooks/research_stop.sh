#!/usr/bin/env bash
# Stop-hook persistence for S6E11 fraud Cursor–Codex research loop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export RESEARCH_ROOT="$ROOT"
INPUT_JSON="$(cat || true)"
export INPUT_JSON
python3 - <<'PY'
import json, os, sys
from pathlib import Path

payload = {}
raw = os.environ.get("INPUT_JSON") or ""
try:
    if raw.strip():
        payload = json.loads(raw)
except json.JSONDecodeError:
    payload = {}

status = payload.get("status") or "completed"
loop_count = int(payload.get("loop_count") or 0)
root = Path(os.environ["RESEARCH_ROOT"])
state_path = root / "artifacts/research_loop/CURRENT_RESEARCH.json"

def emit(obj):
    sys.stdout.write(json.dumps(obj))
    sys.exit(0)

if status == "aborted":
    if state_path.exists():
        try:
            d = json.loads(state_path.read_text())
            d["user_stop"] = True
            d["termination"] = "USER_STOP"
            state_path.write_text(json.dumps(d, indent=2) + "\n")
        except Exception:
            pass
    emit({})

if not state_path.exists():
    emit({})

try:
    state = json.loads(state_path.read_text())
except Exception:
    emit({})

if state.get("project_id") != "s6e11-credit-card-fraud":
    emit({})

if state.get("user_stop") or state.get("termination") in {
    "USER_STOP", "RESOURCE_LIMIT", "BLOCKED_VERIFIED", "PUBLIC_LB_VERIFIED"
}:
    emit({})

max_fu = int(state.get("max_followups") or 10)
if loop_count >= max_fu:
    state["termination"] = "RESOURCE_LIMIT"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    emit({})

budget = state.get("experiment_budget_remaining")
if budget is not None and int(budget) <= 0:
    state["termination"] = "RESOURCE_LIMIT"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    emit({})

milestone = state.get("milestone") or "LOCAL_GATE_PASS"
next_action = state.get("next_action") or "continue_research_loop"
sid = state.get("codex_session_id") or "UNKNOWN"

followup = (
    "Continue the S6E11 fraud Cursor–Codex research loop. "
    "Primary objective: public ROC-AUC > 0.95726 under NO_SUBMIT/NO_GPU/cloud $0. "
    f"Current milestone={milestone}. next_action={next_action}. "
    "Do not stop after a status report. Execute measurable experiments, "
    f"return results to the fraud Codex session {sid}, then continue. "
    "Never resume other projects' Codex sessions. "
    "Retired: signed-CB family. "
    f"loop_count={loop_count}."
)
state["followup_count"] = int(state.get("followup_count") or 0) + 1
state_path.write_text(json.dumps(state, indent=2) + "\n")
emit({"followup_message": followup})
PY
