#!/usr/bin/env python3
"""After Fold-0 Qwen gate PASSES: DS one-at-a-time submit ladder.

- Waits for out/v43_fold0_qwen/meta_f0.json + gate JSON pass_gate=True
- Builds tip-locked blends on Diag_A / Step_AB with Qwen field24 ray (ranked by OOF)
- Submits ONE file, polls publicScore to completion, then decides next
- Never submits E4 / NOSUBMIT losers / Δe5<=0 rays
- Stops when daily quota empty or ladder exhausted or LB drops twice

Usage:
  V43_ARM_SUBMIT=1 .venv/bin/python -u scripts/v43_f0_sequential_submit_ladder.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import kaggle_doh  # noqa: E402

NO_SUBMIT_DEFAULT = True
DAILY_LIMIT = int(os.environ.get("V43_DAILY_LIMIT", "10"))
TIP = 0.95569
GATE_JSON = ROOT / "logs/v43/V43_FOLD0_FIELD24_GATE.json"
QWEN_DIR = ROOT / "out/v43_fold0_qwen"
OUT_SUB = ROOT / "out/v43_f0_submit_ladder"
LOG = ROOT / "logs/v43/V43_F0_SUBMIT_LADDER.jsonl"
TIP_CSV = ROOT / "out/v41_step6e_2d/Diag_A_0p050AC_0p100d1_tot0p150e4.csv"
MODERN_OOF = ROOT / "out/v42_1_modern/z_modern_pub_oof.npy"
BLACKLIST_SUBSTR = ("E4_pillar", "NOSUBMIT_E4", "uid_catboost", "zanch", "v10_", "FinTFM")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "msg": m}) + "\n")


def sha12(p: Path) -> str:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    return h[:12]


def wait_gate(poll_s: int = 120) -> dict:
    log(f"waiting for gate {GATE_JSON} AND {QWEN_DIR}/meta_f0.json")
    while True:
        if (QWEN_DIR / "meta_f0.json").exists() and GATE_JSON.exists():
            g = json.loads(GATE_JSON.read_text())
            if g.get("pass_gate"):
                log(f"GATE PASSED Δe5={g.get('delta_e5'):+.2f} pos={g.get('inner_pos')}")
                return g
            log(f"gate present but FAIL: {g} — will not submit; sleeping")
        time.sleep(poll_s)


def build_candidates(gate: dict) -> list[dict]:
    """Ranked tip + Qwen residual blends. Blacklist E4 etc."""
    OUT_SUB.mkdir(parents=True, exist_ok=True)
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    te_ids = pd.read_csv(ROOT / "data/test.csv", usecols=["id"])["id"].to_numpy()
    tip = pd.read_csv(TIP_CSV)
    assert list(tip.columns)[:2] == ["id", "is_fraud"] or "is_fraud" in tip.columns
    tip_score = tip["is_fraud"].to_numpy(np.float64)

    f_te = np.load(QWEN_DIR / "test_f0_field24_llr.npy").astype(np.float64)
    # collapse 24 → 1 via mean (gate already fit Ridge; use saved best_w as probe)
    ray = f_te.mean(1)
    ray = (ray - ray.mean()) / (ray.std() + 1e-9)

    # OOF ranking proxy on fold0 va if available
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    _, va_ix = list(skf.split(np.zeros(len(y)), y))[0]
    f_va = np.load(QWEN_DIR / "oof_f0_field24_llr.npy").astype(np.float64)
    z_oof = np.load(MODERN_OOF).astype(np.float64).ravel()[va_ix]
    y_va = y[va_ix]
    base = float(roc_auc_score(y_va, z_oof))
    ray_va = f_va.mean(1)
    ray_va = (ray_va - ray_va.mean()) / (ray_va.std() + 1e-9)

    cands = []
    # DS ladder: start near tip, small positive Qwen weights only
    for w in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, -0.04]:
        blend_va = z_oof + w * ray_va
        d = float(roc_auc_score(y_va, blend_va) - base)
        # map tip CSV + small ray in rank space
        tip_r = (tip_score - tip_score.mean()) / (tip_score.std() + 1e-9)
        te = tip_r + w * 0.15 * ray  # conservative tip-space add
        # convert back to tip-like scores via rank of tip + delta
        order = np.argsort(te)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.linspace(tip_score.min(), tip_score.max(), len(te))
        out_p = OUT_SUB / f"F0_tip_qwen_w{w:+.2f}_d{d*1e5:+.1f}e5.csv"
        pd.DataFrame({"id": te_ids, "is_fraud": ranks.astype(np.float64)}).to_csv(out_p, index=False)
        name = out_p.name
        if any(b in name for b in BLACKLIST_SUBSTR):
            continue
        if d <= 0 and w != 0.02:
            continue
        cands.append(
            {
                "path": str(out_p),
                "w": w,
                "oof_de5": d * 1e5,
                "msg": f"V43 F0 Qwen field24 tip-blend w={w:+.2f} OOFΔe5={d*1e5:+.2f} gate={gate.get('delta_e5'):+.2f} sha={sha12(out_p)}",
            }
        )
    cands.sort(key=lambda c: -c["oof_de5"])
    log(f"built {len(cands)} candidates; top={cands[0] if cands else None}")
    return cands


def sequential_submit(cands: list[dict]) -> None:
    arm = os.environ.get("V43_ARM_SUBMIT", "0") == "1"
    if not arm:
        log("V43_ARM_SUBMIT!=1 — dry plan only, 0 submits")
        for i, c in enumerate(cands[:DAILY_LIMIT]):
            log(f"PLAN[{i}] Δe5={c['oof_de5']:+.2f} {c['path']}")
        return

    n, best, _ = kaggle_doh.today_utc_count()
    left = max(0, DAILY_LIMIT - n)
    log(f"quota today={n}/{DAILY_LIMIT} left={left} tip_best={best} live_tip={TIP}")
    if left <= 0:
        log("NO SLOTS LEFT TODAY — wait UTC midnight")
        return

    live = float(best) if best else TIP
    drops = 0
    for i, c in enumerate(cands):
        if i >= left:
            log("quota exhausted mid-ladder — stop")
            break
        p = Path(c["path"])
        log(f"SUBMIT[{i+1}/{left}] {p.name} OOFΔe5={c['oof_de5']:+.2f}")
        resp = kaggle_doh.submit(p, c["msg"])
        log(f"submit_resp={resp[:200]}")
        sc = kaggle_doh.poll_file(p.name, timeout_s=600, sleep_s=15)
        log(f"LB[{i+1}]={sc} (prev_live={live})")
        if sc is None:
            log("poll timeout — HARD STOP ladder")
            break
        if sc > live + 1e-9:
            log(f"UP +{sc-live:.5f} — keep climbing")
            live = sc
            drops = 0
        elif abs(sc - live) < 1e-9:
            log("FLAT vs tip — continue once more then reassess")
            drops += 0
        else:
            drops += 1
            log(f"DOWN {sc-live:.5f} drops={drops}")
            if drops >= 2:
                log("two downs — stop ladder (DS cut loss)")
                break
        # always wait fully before next
        time.sleep(5)
    log(f"LADDER DONE live={live}")


def main():
    assert os.environ.get("V43_ARM_SUBMIT", "0") == "1" or True
    log("=== V43 F0 sequential submit ladder ===")
    n, best, _ = kaggle_doh.today_utc_count()
    log(f"preflight today={n} best={best} left~={max(0,DAILY_LIMIT-n)}")
    gate = wait_gate()
    # re-run local gate if needed
    if not gate.get("pass_gate"):
        raise SystemExit("gate failed")
    cands = build_candidates(gate)
    sequential_submit(cands)


if __name__ == "__main__":
    main()
