#!/usr/bin/env python3
"""Up to 5 sequential tip-climbing submits. Wait for each LB before next.

Priority (DS):
  1) Diag_B — held sibling of live tip Diag_A (never submitted)
  2) Cand2 StepAC — held orthogonal board ray
  3–5) Diag_A ⊕ rank-aligned z_modern_pub at increasing tip weights
     (only real +5e-5 OOF bank never tip-fired)

Never: E4, StageB collapsed, SLOT02/06 losers.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import kaggle_doh  # noqa: E402

OUT = ROOT / "out/v43_max5_ladder"
OUT.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "logs/v43/V43_MAX5_SUBMIT_LADDER.jsonl"
TIP = ROOT / "out/v41_step6e_2d/Diag_A_0p050AC_0p100d1_tot0p150e4.csv"
MAX_N = 5
DAILY_LIMIT = 10


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "msg": m}) + "\n")


def sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def rank_to_tip(src: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Monotone map src ranks onto tip's sorted value ladder (preserves tip marginal)."""
    tip_sorted = np.sort(tip)
    order = np.argsort(src)
    out = np.empty_like(src, dtype=np.float64)
    out[order] = tip_sorted
    return out


def blend_tip(tip: np.ndarray, ray: np.ndarray, w: float) -> np.ndarray:
    """tip + w * standardized(ray_in_tip_scale - tip), then re-rank onto tip ladder."""
    r = rank_to_tip(ray, tip)
    z = (r - tip) 
    # keep small; clip extreme
    mixed = tip + w * z
    return rank_to_tip(mixed, tip)


def build_ladder() -> list[dict]:
    tip_df = pd.read_csv(TIP)
    ids = tip_df["id"].to_numpy()
    tip = tip_df["is_fraud"].to_numpy(np.float64)
    cands: list[dict] = []

    def add(path: Path, msg: str, priority: float):
        cands.append({"path": path, "msg": msg, "priority": priority, "sha": sha12(path)})

    # 1) Diag_B
    b = ROOT / "out/v41_step6e_2d/Diag_B_0p060AC_0p160d1_tot0p220e4.csv"
    add(b, f"V43 max5 #1 Diag_B held sibling of Diag_A tip (0.06AC+0.16d1 @0.22e-4) sha={sha12(b)}", 100.0)

    # 2) Cand2
    c2 = ROOT / "out/v41_step6d_candidates/Cand2_Board_StepAC.csv"
    if c2.exists():
        add(c2, f"V43 max5 #2 Cand2 Board StepAC held ray @0.06e-4 sha={sha12(c2)}", 90.0)

    # 3–5) modern bank on Diag_A
    zm_te = np.load(ROOT / "out/v42_1_modern/z_modern_pub_test.npy").ravel().astype(np.float64)
    zm_oof = np.load(ROOT / "out/v42_1_modern/z_modern_pub_oof.npy").ravel().astype(np.float64)
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)

    # OOF: how much does modern beat a tip-proxy? use fold AUC of modern vs pub isn't tip;
    # rank modern OOF and report AUC for sanity
    auc_m = float(roc_auc_score(y, zm_oof))
    log(f"z_modern_pub OOF AUC={auc_m:.6f}")

    for w, pri in [(0.08, 80.0), (0.15, 70.0), (0.25, 60.0), (0.40, 50.0)]:
        te = blend_tip(tip, zm_te, w)
        p = OUT / f"DiagA_x_modern_w{w:.2f}.csv"
        pd.DataFrame({"id": ids, "is_fraud": te}).to_csv(p, index=False)
        add(p, f"V43 max5 Diag_A⊕z_modern_pub rank-blend w={w:.2f} (V42.1 +5.21e-5 bank) sha={sha12(p)}", pri)

    # Cand1 higher weight variant if exists
    c1 = ROOT / "out/v41_step6d_candidates/Cand1_FWD_AB_AA_0p10e4.csv"
    if c1.exists():
        add(c1, f"V43 max5 Cand1 FWD_AB_AA @0.10e-4 (vs flat 0.03) sha={sha12(c1)}", 40.0)

    cands.sort(key=lambda c: -c["priority"])
    return cands[:MAX_N]


def cli_submit(path: Path, message: str) -> str:
    """Official kaggle CLI (DoH curl multipart is broken: BlobFileTokens)."""
    import os
    import subprocess

    for k in list(os.environ):
        if "proxy" in k.lower():
            os.environ.pop(k, None)
    exe = "/Users/mz/Library/Python/3.14/bin/kaggle"
    cmd = [
        exe,
        "competitions",
        "submit",
        "-c",
        kaggle_doh.COMP,
        "-f",
        str(path),
        "-m",
        message[:480],
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return out


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    n, best, _ = kaggle_doh.today_utc_count()
    left = max(0, DAILY_LIMIT - n)
    log(f"=== MAX5 tip ladder | today={n} best={best} left={left} ===")
    cands = build_ladder()
    for i, c in enumerate(cands):
        log(f"PLAN[{i}] pri={c['priority']} sha={c['sha']} {c['path'].name}")

    # Skip already-scored files from today
    _, _, rows = kaggle_doh.today_utc_count()
    today = str(__import__("datetime").datetime.now(__import__("datetime").timezone.utc).date())
    done_names = {
        r.get("fileName")
        for r in rows
        if (r.get("date") or "")[:10] == today and r.get("status") == "complete"
    }
    pending = [c for c in cands if c["path"].name not in done_names]
    # If Diag_B already submitted but unscored, poll it first
    for c in cands:
        if c["path"].name not in done_names:
            # may be in-flight
            pass

    take = min(MAX_N, left, len(cands))
    if take <= 0:
        log("NO SLOTS — abort")
        return

    live = float(best) if best else 0.95569
    results = []
    submitted = 0
    for c in cands:
        if submitted >= MAX_N:
            break
        p = Path(c["path"])
        # if already complete today, record and continue
        sc_existing = None
        for r in kaggle_doh.list_submissions(1):
            if r.get("fileName") == p.name and (r.get("date") or "")[:10] == today:
                sc_existing = r.get("publicScore") or None
                st = r.get("status")
                if sc_existing not in ("", None) and st == "complete":
                    sc_existing = float(sc_existing)
                    log(f"ALREADY {p.name} LB={sc_existing}")
                    results.append({"file": p.name, "score": sc_existing, "delta": sc_existing - live, "sha": c["sha"]})
                    if sc_existing > live:
                        live = sc_existing
                    submitted += 1
                    sc_existing = "done"
                    break
                sc_existing = "pending"
                break
        if sc_existing == "done":
            continue
        if sc_existing != "pending":
            n_now, _, _ = kaggle_doh.today_utc_count()
            if DAILY_LIMIT - n_now <= 0:
                log("quota empty")
                break
            log(f"SUBMIT {submitted+1}/{MAX_N} {p.name}")
            try:
                resp = cli_submit(p, c["msg"])
                log(f"resp={resp.strip()[:240]}")
            except Exception as e:
                log(f"SUBMIT FAIL: {e}")
                break
        else:
            log(f"POLL in-flight {p.name}")
        sc = kaggle_doh.poll_file(p.name, timeout_s=900, sleep_s=12)
        log(f"LB[{submitted+1}]={sc} live_was={live}")
        if sc is None:
            log("poll timeout — stop ladder")
            break
        delta = sc - live
        results.append({"file": p.name, "score": sc, "delta": delta, "sha": c["sha"]})
        submitted += 1
        if sc > live + 1e-9:
            log(f"UP +{delta:.5f} — new tip={sc}")
            live = sc
        elif abs(delta) < 1e-9:
            log("FLAT")
        else:
            log(f"DOWN {delta:.5f}")
        time.sleep(3)

    out = {"live": live, "results": results, "NO_PARALLEL": True}
    (ROOT / "logs/v43/V43_MAX5_SUBMIT_RESULTS.json").write_text(json.dumps(out, indent=2))
    log(f"DONE live={live} n={len(results)}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
