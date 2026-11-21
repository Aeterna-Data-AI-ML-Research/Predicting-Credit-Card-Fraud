#!/usr/bin/env python3
"""GM V16.5 Slot3: Slot1 + d1_v4 Δw=0.0055 (w_d1=0.0355). Already submitted → 0.95389.

Re-run with --submit to push again (not recommended; shelf sticky).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
CSV = REPO / "submissions/sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
KAG = "/Users/mz/Library/Python/3.14/bin/kaggle"
REPORT = REPO / "logs/overnight/GM_V16_5_SLOT3_D1_POLISH.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    assert CSV.exists(), CSV
    print("file", CSV, "report", REPORT)
    if args.submit:
        env = {k: v for k, v in __import__("os").environ.items() if "proxy" not in k.lower()}
        subprocess.check_call(
            [
                KAG,
                "competitions",
                "submit",
                "-c",
                "playground-series-s6e11",
                "-f",
                str(CSV),
                "-m",
                "v16.5 Slot3: Slot1 (0.95389) + d1_v4 polish (w_d1=0.0355)",
            ],
            env=env,
        )


if __name__ == "__main__":
    main()
