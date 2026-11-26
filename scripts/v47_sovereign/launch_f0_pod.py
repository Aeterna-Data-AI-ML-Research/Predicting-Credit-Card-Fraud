#!/usr/bin/env python3
"""V47 Fold-0 pod launcher — balance gate + create ONLY a new V47 pod.

NEVER touches PRE_EXISTING pods (e.g. u4ky2oq4pohcyt).
Refuses create when balanceAmount <= 0. NO_SUBMIT.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

NO_SUBMIT = True
assert NO_SUBMIT
REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "logs/v47"
LOG.mkdir(parents=True, exist_ok=True)
PREEXIST = (LOG / "PRE_EXISTING_PODS.txt").read_text().split() if (LOG / "PRE_EXISTING_PODS.txt").exists() else []
HARD_CAP_USD = 3.50
DEADMAN_S = 16200


def main():
    # This script documents the create contract; live MCP create is done by the agent
    # after balance > 0. Local dry-run always exits non-zero on unpaid balance.
    status = {
        "NO_SUBMIT": True,
        "pre_existing_do_not_touch": PREEXIST,
        "hard_cap_usd": HARD_CAP_USD,
        "deadman_s": DEADMAN_S,
        "preferred_gpus": ["NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA A40"],
        "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "on_start": "bash /workspace/v47/scripts/v47_sovereign/pod_f0/on_pod_main.sh",
        "rsync_dest": str(REPO / "preds/v47_pod_f0"),
        "create_allowed": False,
        "block_reason": "RunPod balanceAmount must be > 0 before create-pod. Current check is agent-side.",
    }
    (LOG / "LAUNCH_F0_CONTRACT.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
