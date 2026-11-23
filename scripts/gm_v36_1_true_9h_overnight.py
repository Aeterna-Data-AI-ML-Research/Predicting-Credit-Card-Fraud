#!/usr/bin/env python3
"""
V36.1 — TRUE 9h unattended #1 campaign supervisor (RSS target <200MB).
Dispatches isolated workers; holds until >= 04:30 UTC; max 2 submits after 00:02 UTC.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from v36_1_lib import (  # noqa: E402
    COMP,
    DEADLINE_UTC,
    HEARTBEAT,
    LOGS,
    MIDNIGHT_EVAL_UTC,
    PRED,
    STATE_PATH,
    SUBMIT_OPEN_UTC,
    ensure_dirs,
    heartbeat,
    kaggle_cli,
    load_state,
    now_utc,
    parse_submissions_table,
    save_state,
    status_is_complete,
    used_today_utc,
)

PY = str(REPO / ".venv/bin/python")
W0 = str(REPO / "scripts/v36_1_workers/w0_repair_baselines_and_intel.py")
WAVE = str(REPO / "scripts/v36_1_workers/run_wave_engine.py")

BUDGET = {
    "W0": 25 * 60,
    "W1_E1": 110 * 60,  # 30×5-fold Cat/LGB/XGB needs >70m on M3
    "W1_E1_fixed": 110 * 60,
    "W1_E2": 110 * 60,
    "W1_E3": 110 * 60,
    "W1_E4": 110 * 60,
    "W2_GPU": 15 * 60,
    "W3_E5": 90 * 60,
    "W3_E6": 50 * 60,
    "W3_JEWELS": 40 * 60,
    "W3_MultiVault": 45 * 60,
    "W4_E7": 100 * 60,
}


def free_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 8.0


def run_worker(cmd: list[str], engine: str, timeout: int) -> int:
    waited = 0
    while free_gb() < 3.5 and waited < 1800:
        heartbeat("wait_rss", engine, free_gb=free_gb())
        time.sleep(60)
        waited += 60
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["LIGHTGBM_NUM_THREADS"] = "4"
    heartbeat("start_worker", engine, timeout=timeout)
    print(f"[SUP] launch {engine} timeout={timeout}s cmd={' '.join(cmd[-4:])}", flush=True)
    p = subprocess.Popen(cmd, cwd=str(REPO), env=env, start_new_session=True)
    try:
        code = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[SUP] TIMEOUT {engine} — killing worker tree", flush=True)
        for sig in (15, 9):
            try:
                os.killpg(p.pid, sig)
            except (ProcessLookupError, PermissionError, OSError) as e:
                print(f"[SUP] killpg({sig}) {e} — fallback p.kill()", flush=True)
                try:
                    p.kill()
                except Exception:
                    pass
            time.sleep(2)
        try:
            p.wait(timeout=15)
        except Exception:
            pass
        code = 124
    heartbeat("end_worker", engine, code=code, free_gb=free_gb())
    return code


def mark(st, engine, status, **extra):
    st.setdefault("engines", {})[engine] = {"status": status, "ts": now_utc().isoformat(), **extra}
    save_state(st)


def need(st, engine) -> bool:
    return st.get("engines", {}).get(engine, {}).get("status") != "done"


def wait_until(target: datetime, label: str):
    while now_utc() < target:
        left = (target - now_utc()).total_seconds()
        heartbeat("wait", label, left_s=int(left))
        print(f"[SUP] waiting {label} ({left/3600:.2f}h left)", flush=True)
        # keep process alive; do light work only via sleep
        time.sleep(min(300, max(30, left / 4)))


def poll_score(fname: str, timeout_s: int = 900) -> float | None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        text = kaggle_cli(["competitions", "submissions", "-c", COMP], timeout=90)
        for r in parse_submissions_table(text):
            if fname in r.get("fileName", "") or fname in r.get("raw", ""):
                if status_is_complete(r.get("status", "")) and r.get("publicScore") is not None:
                    sc = float(r["publicScore"])
                    if sc > 0.90:
                        return sc
        time.sleep(30)
        heartbeat("poll_lb", fname)
    return None


def g1_g2_pass(gate: dict) -> bool:
    g_fb = gate.get("g_fb") or {}
    g_tw = gate.get("g_twin") or {}
    if not gate.get("g0") and not (
        g_fb.get("pos") == 5
        and g_fb.get("sub", 0) >= 20
        and g_fb.get("t25", 0) >= 3.5
        and g_fb.get("mean_e4", 0) >= 2.0
    ):
        # allow rock-solid alternate
        alt = (
            g_fb.get("pos") == 5
            and g_tw.get("pos", 0) >= 4
            and g_fb.get("t25", 0) >= 4.2
            and g_fb.get("mean_e4", 0) >= 1.2
            and g_tw.get("mean_e4", 0) > 0
        )
        if not alt:
            return False
    if g_fb.get("pos", 0) < 5:
        return False
    if g_tw.get("pos", 0) < 4:
        return False
    if g_fb.get("mean_e4", 0) < 1.2:
        return False
    return True


def phase_d_submit(st: dict, smoke: bool):
    staged_p = LOGS / "v36_1_ckpt/staged.json"
    if not staged_p.exists():
        print("[SUP] no staged.json — hold", flush=True)
        return
    staged = json.loads(staged_p.read_text())
    gate = staged.get("gate") or {}
    if not g1_g2_pass(gate):
        print(f"[SUP] G1+G2 FAIL — hold slots. gate_g0={gate.get('g0')}", flush=True)
        st["hold_reason"] = "G1_G2_fail"
        save_state(st)
        return
    if smoke:
        print("[SUP] smoke: would submit", staged.get("cand1"))
        return
    if now_utc() < SUBMIT_OPEN_UTC:
        wait_until(SUBMIT_OPEN_UTC, "submit_open")
    text = kaggle_cli(["competitions", "submissions", "-c", COMP], timeout=90)
    rows = parse_submissions_table(text)
    mid = now_utc().replace(hour=0, minute=0, second=0, microsecond=0)
    used = used_today_utc(rows, mid)
    print(f"[SUP] used_today={used}", flush=True)
    if used >= 5:
        return
    run_id = st["run_id"]
    c1 = Path(staged["cand1"])
    live1 = REPO / f"submissions/sub_v36_1_CAND1_{run_id}.csv"
    live1.write_bytes(c1.read_bytes())
    if any(live1.name in r.get("fileName", "") for r in rows):
        score = poll_score(live1.name)
    else:
        msg = f"v36.1 CAND1 0.60w* | {run_id}"
        print(kaggle_cli(["competitions", "submit", "-c", COMP, "-f", str(live1), "-m", msg], timeout=90)[-400:], flush=True)
        st.setdefault("submit_log", []).append({"file": live1.name, "ts": now_utc().isoformat()})
        save_state(st)
        score = poll_score(live1.name)
    print(f"[SUP] CAND1={score}", flush=True)
    st["cand1_score"] = score
    save_state(st)
    if score is None:
        return
    ath = float(st.get("ath_lb", 0.95548))
    dpred = float(staged.get("delta_pred_full_e4", 2.0)) * 1e-4
    delta = score - ath
    thr = max(0.00020, 0.45 * max(dpred, 0.00020))
    if delta >= thr and used + 1 < 5:
        live2 = REPO / f"submissions/sub_v36_1_CAND2_{run_id}.csv"
        live2.write_bytes(Path(staged["cand2"]).read_bytes())
        msg2 = f"v36.1 CAND2 1.0w* d={delta:.5f} thr={thr:.5f} | {run_id}"
        print(kaggle_cli(["competitions", "submit", "-c", COMP, "-f", str(live2), "-m", msg2], timeout=90)[-400:], flush=True)
        st["submit_log"].append({"file": live2.name, "ts": now_utc().isoformat()})
        st["cand2_score"] = poll_score(live2.name)
        save_state(st)
    else:
        print(f"[SUP] skip CAND2 delta={delta:.5f} thr={thr:.5f}", flush=True)


def write_report(st: dict):
    lines = [
        "# GM V36.1 OVERNIGHT REPORT",
        "",
        f"- run_id: `{st.get('run_id')}`",
        f"- finished_utc: `{now_utc().isoformat()}`",
        f"- engines: `{json.dumps(st.get('engines'), default=str)[:2000]}`",
        f"- submit_log: `{st.get('submit_log')}`",
        f"- hold_reason: `{st.get('hold_reason')}`",
        f"- cand1/cand2: `{st.get('cand1_score')}` / `{st.get('cand2_score')}`",
        "",
    ]
    bm = PRED / "baselines_meta.json"
    if bm.exists():
        lines += ["## Baselines", "```", bm.read_text()[:3000], "```", ""]
    for p in sorted((LOGS / "v36_1_ckpt").glob("*_gate.json")):
        lines += [f"## {p.name}", "```", p.read_text()[:1500], "```", ""]
    staged = LOGS / "v36_1_ckpt/staged.json"
    if staged.exists():
        lines += ["## Staged", "```", staged.read_text()[:2000], "```", ""]
    lines += [
        "## Next-day recommendations",
        "- If vault empty: widen B_clean / fix twin R² / raise Mode-B residual capacity.",
        "- If G0 near-miss 4/5: nested w* on ATH + public-board residual blend (not micro grid).",
        f"- Heartbeat: `{HEARTBEAT}`",
        "",
    ]
    (LOGS / "GM_V36_1_OVERNIGHT_REPORT.md").write_text("\n".join(lines))
    print(f"[SUP] wrote report", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    smoke = args.smoke_test
    ensure_dirs()
    if args.fresh and STATE_PATH.exists():
        STATE_PATH.unlink()
    st = load_state()
    st["smoke"] = smoke
    save_state(st)
    print(f"[SUP] V36.1 start run_id={st['run_id']} deadline={DEADLINE_UTC.isoformat()} smoke={smoke}", flush=True)
    heartbeat("supervisor", "boot")

    # T0
    if need(st, "W0"):
        mark(st, "W0", "running")
        code = run_worker([PY, W0] + (["--smoke"] if smoke else []), "W0", 180 if smoke else BUDGET["W0"])
        mark(st, "W0", "done" if code == 0 else "fail", code=code)
        if code != 0 and not smoke:
            print("[SUP] W0 failed — abort", flush=True)
            write_report(st)
            return code

    # Hotpatch order: de-leaked E1_fixed after E4; jewels before MultiVault
    order = [
        "W1_E1",
        "W1_E2",
        "W1_E3",
        "W1_E4",
        "W1_E1_fixed",
        "W2_GPU",
        "W3_E5",
        "W3_E6",
        "W3_JEWELS",
        "W3_MultiVault",
    ]
    # Allow live order override without full rewrite
    order_path = LOGS / "v36_1_ckpt/engine_order.json"
    if order_path.exists():
        try:
            order = json.loads(order_path.read_text()).get("order", order)
            print(f"[SUP] loaded engine_order.json: {order}", flush=True)
        except Exception as e:
            print(f"[SUP] order override fail {e}", flush=True)
    for eng in order:
        if now_utc() >= DEADLINE_UTC and not smoke:
            break
        # re-read order each step so hotpatch can inject engines mid-run via file
        if order_path.exists():
            try:
                order = json.loads(order_path.read_text()).get("order", order)
            except Exception:
                pass
        if not need(st, eng):
            print(f"[SUP] skip done {eng}", flush=True)
            continue
        mark(st, eng, "running")
        to = 120 if smoke else BUDGET.get(eng, 110 * 60)
        code = run_worker([PY, WAVE, "--engine", eng] + (["--smoke"] if smoke else []), eng, to)
        mark(st, eng, "done" if code == 0 else "fail", code=code)
        # refresh state from disk in case external hotpatch touched it
        st = load_state()

    # Hold / deep work until midnight eval
    if not smoke and now_utc() < MIDNIGHT_EVAL_UTC:
        if need(st, "W4_E7"):
            mark(st, "W4_E7", "running")
            # may run until budget or midnight
            left = max(120, int((MIDNIGHT_EVAL_UTC - now_utc()).total_seconds() + 600))
            to = min(BUDGET["W4_E7"], left)
            code = run_worker([PY, WAVE, "--engine", "W4_E7"], "W4_E7", to)
            mark(st, "W4_E7", "done" if code == 0 else "fail", code=code)
        if now_utc() < MIDNIGHT_EVAL_UTC:
            wait_until(MIDNIGHT_EVAL_UTC, "midnight_eval")
        # re-run multivault after midnight
        mark(st, "W3_MultiVault_post", "running")
        code = run_worker([PY, WAVE, "--engine", "W3_MultiVault"], "W3_MultiVault_post", BUDGET["W3_MultiVault"])
        mark(st, "W3_MultiVault_post", "done" if code == 0 else "fail", code=code)
        phase_d_submit(st, smoke)
        # continue deep exploration if not promoted
        if not st.get("submit_log") and now_utc() < DEADLINE_UTC:
            print("[SUP] no submit — deep explore to 04:15Z", flush=True)
            explore_end = datetime(2026, 11, 22, 4, 15, 0, tzinfo=timezone.utc)
            while now_utc() < explore_end:
                if need(st, "W4_E7_b") or st.get("engines", {}).get("W4_E7_b", {}).get("status") != "done":
                    mark(st, "W4_E7_b", "running")
                    left = int((explore_end - now_utc()).total_seconds())
                    code = run_worker([PY, WAVE, "--engine", "W4_E7"], "W4_E7_b", min(BUDGET["W4_E7"], max(120, left)))
                    mark(st, "W4_E7_b", "done" if code == 0 else "fail", code=code)
                    run_worker([PY, WAVE, "--engine", "W3_MultiVault"], "W3_MultiVault_final", BUDGET["W3_MultiVault"])
                    phase_d_submit(st, smoke)
                    break
                break
    else:
        # smoke or already past midnight
        if need(st, "W4_E7"):
            mark(st, "W4_E7", "running")
            code = run_worker([PY, WAVE, "--engine", "W4_E7"] + (["--smoke"] if smoke else []), "W4_E7", 120 if smoke else BUDGET["W4_E7"])
            mark(st, "W4_E7", "done" if code == 0 else "fail", code=code)
        if need(st, "W3_MultiVault"):
            mark(st, "W3_MultiVault", "running")
            code = run_worker([PY, WAVE, "--engine", "W3_MultiVault"] + (["--smoke"] if smoke else []), "W3_MultiVault", 120 if smoke else BUDGET["W3_MultiVault"])
            mark(st, "W3_MultiVault", "done" if code == 0 else "fail", code=code)
        phase_d_submit(st, smoke)

    if not smoke and now_utc() < DEADLINE_UTC:
        wait_until(DEADLINE_UTC, "hard_deadline")

    write_report(st)
    heartbeat("supervisor", "done")
    print("[SUP] V36.1 finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
