#!/usr/bin/env python3
"""
V36 — 12h unattended #1-or-bust supervisor (<150MB).
Launches isolated workers; checkpoints; waits NEXT_UTC_MIDNIGHT for ≤2 submits.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from v36_lib import (  # noqa: E402
    ATH_CSV,
    COMP,
    HEARTBEAT,
    KAGGLE,
    LOGS,
    STATE_PATH,
    ensure_dirs,
    heartbeat,
    kaggle_cli,
    load_state,
    next_utc_midnight,
    now_utc,
    parse_submissions_table,
    save_state,
    status_is_complete,
    used_today_utc,
)

PY = str(REPO / ".venv/bin/python")
WORKER = str(REPO / "scripts/v36_workers/run_engine.py")

# wall budgets (seconds)
BUDGET = {
    "t0": 600,
    "phase_a": 1200,
    "H1": 45 * 60,
    "H2": 55 * 60,
    "H3": 50 * 60,
    "H4": 50 * 60,
    "H5": 60 * 60,
    "phase_c": 1200,
}


def free_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 8.0


def run_worker(engine: str, smoke: bool, timeout: int) -> int:
    """Run isolated worker in its own process group; kill group on timeout."""
    waited = 0
    while free_gb() < 4.0 and waited < 1800:
        heartbeat("wait_rss", engine, free_gb=free_gb())
        time.sleep(60)
        waited += 60
    if free_gb() < 4.0:
        print(f"[SUP] RSS wait cap — launching {engine} anyway (free={free_gb():.1f}G)", flush=True)
    cmd = [PY, WORKER, "--engine", engine]
    if smoke:
        cmd.append("--smoke")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["LIGHTGBM_NUM_THREADS"] = "4"
    heartbeat("start_worker", engine, timeout=timeout)
    print(f"[SUP] launch {engine} timeout={timeout}s smoke={smoke}", flush=True)
    # new session so timeout can kill LGB/OpenMP children
    p = subprocess.Popen(cmd, cwd=str(REPO), env=env, start_new_session=True)
    try:
        code = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[SUP] TIMEOUT {engine} — killing process group", flush=True)
        try:
            os.killpg(p.pid, 15)
            time.sleep(2)
            os.killpg(p.pid, 9)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=10)
        except Exception:
            pass
        code = 124
    heartbeat("end_worker", engine, code=code, free_gb=free_gb())
    return code


def mark(st: dict, engine: str, status: str, **extra):
    st.setdefault("engines", {})[engine] = {"status": status, "ts": now_utc().isoformat(), **extra}
    save_state(st)


def wait_until_midnight(st: dict, smoke: bool):
    if smoke:
        print("[SUP] smoke: skip midnight wait", flush=True)
        return
    target = datetime.fromisoformat(st["NEXT_UTC_MIDNIGHT"])
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    while now_utc() < target:
        left = (target - now_utc()).total_seconds()
        heartbeat("wait_midnight", "", left_s=int(left))
        print(f"[SUP] waiting midnight UTC ({left/3600:.2f}h left)", flush=True)
        time.sleep(min(300, max(30, left / 3)))


def poll_submission_score(fname: str, timeout_s: int = 900) -> float | None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        text = kaggle_cli(["competitions", "submissions", "-c", COMP], timeout=90)
        rows = parse_submissions_table(text)
        for r in rows:
            if fname in r.get("fileName", "") or fname in r.get("raw", ""):
                if status_is_complete(r.get("status", "")) and r.get("publicScore") is not None:
                    return float(r["publicScore"])
        time.sleep(30)
        heartbeat("poll_lb", fname)
    return None


def _already_submitted(rows: list, fname: str, submit_log: list) -> bool:
    if any(fname in r.get("fileName", "") for r in rows):
        return True
    return any(e.get("file") == fname for e in submit_log)


def phase_d_submit(st: dict, smoke: bool):
    staged = {}
    pc = LOGS / "v36_ckpt/phase_c.json"
    if pc.exists():
        import json

        staged = json.loads(pc.read_text()).get("staged") or {}
    if not staged:
        print("[SUP] Phase D: nothing staged — hold slots", flush=True)
        st["phase"] = "hold"
        save_state(st)
        return

    wait_until_midnight(st, smoke)
    if smoke:
        print("[SUP] smoke: would submit", staged.get("cand1"), staged.get("cand2"))
        st["submit_log"].append({"smoke": True, "staged": staged})
        save_state(st)
        return

    # used today — AFTER midnight wait so this is the new UTC day
    text = kaggle_cli(["competitions", "submissions", "-c", COMP], timeout=90)
    rows = parse_submissions_table(text)
    mid_today = now_utc().replace(hour=0, minute=0, second=0, microsecond=0)
    used = used_today_utc(rows, mid_today)
    print(f"[SUP] used_today_utc≈{used}", flush=True)
    if used >= 5:
        print("[SUP] quota full — hold", flush=True)
        return

    run_id = st.get("run_id", "v36")
    cand1 = Path(staged["cand1"])
    # unique live name avoids dup-guard false block / double-upload
    live1 = REPO / f"submissions/sub_v36_CAND1_{run_id}.csv"
    live1.write_bytes(cand1.read_bytes())
    if _already_submitted(rows, live1.name, st.get("submit_log", [])):
        print(f"[SUP] dup guard — already submitted {live1.name}", flush=True)
        score = poll_submission_score(live1.name)
    else:
        msg = f"v36.CAND1 0.60w* engine={staged.get('engine')} | #1 hunt | {run_id}"
        out = kaggle_cli(
            ["competitions", "submit", "-c", COMP, "-f", str(live1), "-m", msg],
            timeout=90,
        )
        print(out[-500:], flush=True)
        st.setdefault("submit_log", []).append({"file": live1.name, "msg": msg, "ts": now_utc().isoformat()})
        save_state(st)
        score = poll_submission_score(live1.name)
    print(f"[SUP] CAND1 score={score}", flush=True)
    ath = float(st.get("ath_lb", 0.95548))
    # mean_e4 is already ×1e4; convert to absolute ΔAUC
    dpred = float(staged.get("delta_pred_full_e4", 2.5)) * 1e-4
    if score is None:
        return
    delta = score - ath
    st["cand1_score"] = score
    st["cand1_delta"] = delta
    save_state(st)
    thr = max(0.00025, 0.5 * max(dpred, 0.00025))
    if delta >= thr and used + 1 < 5:
        live2 = REPO / f"submissions/sub_v36_CAND2_{run_id}.csv"
        live2.write_bytes(Path(staged["cand2"]).read_bytes())
        if _already_submitted(rows, live2.name, st.get("submit_log", [])):
            print(f"[SUP] dup guard — skip CAND2 {live2.name}", flush=True)
            return
        msg2 = f"v36.CAND2 1.0w* after CAND1 Δ={delta:.5f} thr={thr:.5f} | {run_id}"
        out2 = kaggle_cli(
            ["competitions", "submit", "-c", COMP, "-f", str(live2), "-m", msg2],
            timeout=90,
        )
        print(out2[-500:], flush=True)
        st["submit_log"].append({"file": live2.name, "msg": msg2, "ts": now_utc().isoformat()})
        score2 = poll_submission_score(live2.name)
        st["cand2_score"] = score2
        save_state(st)
        print(f"[SUP] CAND2 score={score2}", flush=True)
    else:
        print(f"[SUP] skip CAND2 (delta={delta:.5f} thr={thr:.5f})", flush=True)


def write_report(st: dict):
    lines = [
        "# GM V36 12H REPORT",
        "",
        f"- run_id: `{st.get('run_id')}`",
        f"- NEXT_UTC_MIDNIGHT: `{st.get('NEXT_UTC_MIDNIGHT')}`",
        f"- promote_h5: `{st.get('promote_h5')}`",
        f"- engines: `{st.get('engines')}`",
        f"- submit_log: `{st.get('submit_log')}`",
        f"- cand1_score: `{st.get('cand1_score')}` cand2_score: `{st.get('cand2_score')}`",
        "",
        f"Heartbeat: `{HEARTBEAT}`",
        f"State: `{STATE_PATH}`",
        "",
    ]
    pc = LOGS / "v36_ckpt/phase_c.json"
    if pc.exists():
        lines += ["## Phase C", "", "```", pc.read_text()[:8000], "```", ""]
    (LOGS / "GM_V36_12H_REPORT.md").write_text("\n".join(lines))
    print(f"[SUP] wrote {LOGS / 'GM_V36_12H_REPORT.md'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--fresh", action="store_true", help="ignore prior state")
    args = ap.parse_args()
    smoke = args.smoke_test
    ensure_dirs()

    if args.fresh and STATE_PATH.exists():
        STATE_PATH.unlink()

    st = load_state()
    if st.get("NEXT_UTC_MIDNIGHT") is None:
        st["NEXT_UTC_MIDNIGHT"] = next_utc_midnight().isoformat()
        st["smoke"] = smoke
        st.setdefault("ath_lb", 0.95548)
        save_state(st)
    st.setdefault("ath_lb", 0.95548)
    save_state(st)
    print(
        f"[SUP] V36 start run_id={st['run_id']} midnight={st['NEXT_UTC_MIDNIGHT']} smoke={smoke}",
        flush=True,
    )
    heartbeat("supervisor", "boot", smoke=smoke)

    # assert ATH exists
    assert ATH_CSV.exists(), ATH_CSV

    def need(engine: str) -> bool:
        # done → skip; running/fail/missing → (re)run (crash-resume dirty)
        rec = st.get("engines", {}).get(engine, {})
        return rec.get("status") != "done"

    # T0
    if need("t0"):
        mark(st, "t0", "running")
        code = run_worker("t0", smoke, BUDGET["t0"] if not smoke else 180)
        mark(st, "t0", "done" if code == 0 else "fail", code=code)
        if code != 0 and not smoke:
            print("[SUP] T0 failed — abort", flush=True)
            write_report(st)
            return code

    # Phase A concurrent-ish (sequential after t0 for simplicity; budget 20m)
    if need("phase_a"):
        mark(st, "phase_a", "running")
        code = run_worker("phase_a", smoke, BUDGET["phase_a"] if not smoke else 60)
        mark(st, "phase_a", "done" if code == 0 else "fail", code=code)
        intel = LOGS / "GM_V36_PHASE_A_INTEL.json"
        if intel.exists():
            import json

            st["promote_h5"] = bool(json.loads(intel.read_text()).get("promote_h5"))
            save_state(st)

    order = ["H1", "H2", "H3", "H4", "H5"]
    if st.get("promote_h5"):
        order = ["H5"] + [e for e in order if e != "H5"]
        print("[SUP] H5 promoted to #1", flush=True)

    for eng in order:
        if not need(eng):
            print(f"[SUP] skip done {eng}", flush=True)
            continue
        mark(st, eng, "running")
        to = 180 if smoke else BUDGET[eng]
        code = run_worker(eng, smoke, to)
        mark(st, eng, "done" if code == 0 else "fail", code=code)

    if need("phase_c"):
        mark(st, "phase_c", "running")
        code = run_worker("phase_c", smoke, 180 if smoke else BUDGET["phase_c"])
        mark(st, "phase_c", "done" if code == 0 else "fail", code=code)

    st["phase"] = "D"
    save_state(st)
    phase_d_submit(st, smoke)

    st["phase"] = "E"
    save_state(st)
    write_report(st)
    heartbeat("supervisor", "done")
    print("[SUP] V36 finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
