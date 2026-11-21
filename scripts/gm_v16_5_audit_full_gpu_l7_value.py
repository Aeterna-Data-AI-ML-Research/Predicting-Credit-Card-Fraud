#!/usr/bin/env python3
"""GM V16.5 diagnostic: f0-in-LIVE audit + full GPU L7 tensor screen. NO_SUBMIT=True."""
from __future__ import annotations

import json
from pathlib import Path

NO_SUBMIT = True
REPO = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
REPORT = REPO / "logs/overnight/GM_V16_5_FULL_GPU_L7_AUDIT.md"
OUT = Path("/tmp/s6e11_v16/out")


def main():
    assert NO_SUBMIT is True
    print("NO_SUBMIT=True — see", REPORT)
    for p in (
        OUT / "v16_5_live_f0_regression.json",
        OUT / "v16_5_gpu_audit.json",
        OUT / "v16_5_d1_grid.json",
    ):
        if p.exists():
            print("artifact", p)


if __name__ == "__main__":
    main()
