#!/usr/bin/env python3
"""Live hotpatch queue: re-gate completed engines; retry E3 after E4; E1_fixed fallback; jewels."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from v36_1_lib import (  # noqa: E402
    DEADLINE_UTC,
    LOGS,
    PRED,
    atomic_json,
    build_subfolds,
    evaluate_engine_g0,
    load_state,
    load_y_fold_ids,
    now_utc,
    save_state,
)

PY = str(REPO / ".venv/bin/python")
WAVE = str(REPO / "scripts/v36_1_workers/run_wave_engine.py")

REGATE = {
    "W1_E2": "W1_E2_orig20k_inversion_and_bpe_exact",
    "W1_E3": "W1_E3_hier_eb_3d4d_and_stealth_fn",
    "W1_E4": "W1_E4_wilcoxon_twilight_mps_specialist",
    "W1_E1_fixed": "W1_E1_fixed_entity_balance_graph",
}


def _load_baselines(n_tr: int, n_te: int):
    z_fb = np.load(PRED / "z_fullbank_oof.npy").ravel()[:n_tr]
    z_fb_te = np.load(PRED / "z_fullbank_test.npy").ravel()[:n_te]
    z_twin = np.load(PRED / "z_ATH_twin_oof.npy").ravel()[:n_tr]
    z_twin_te = np.load(PRED / "z_ATH_twin_test.npy").ravel()[:n_te]
    Bo = np.load(PRED / "B_clean_oof.npy")[:n_tr]
    Bt = np.load(PRED / "B_clean_test.npy")[:n_te]
    eta = np.load(PRED / "eta_cal_fb_oof.npy").ravel()[:n_tr]
    eta_te = np.load(PRED / "eta_cal_fb_test.npy").ravel()[:n_te]
    return z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te


def regate_if_needed(stem: str) -> bool:
    """Re-run fixed G0 on saved oof/test if gate missing hotpatch tag."""
    gp = LOGS / "v36_1_ckpt" / f"{stem}_gate.json"
    op, tp = PRED / f"{stem}_oof.npy", PRED / f"{stem}_test.npy"
    if not (op.exists() and tp.exists()):
        return False
    if gp.exists():
        try:
            old = json.loads(gp.read_text())
            if str(old.get("hotpatch", "")).startswith("v36.1_g0_leak"):
                return False
        except Exception:
            pass
    y, fold, _, ath = load_y_fold_ids()
    oof = np.load(op).ravel()[: len(y)].astype(float)
    te = np.load(tp).ravel()[: len(ath)].astype(float)
    z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te = _load_baselines(len(y), len(te))
    sf = build_subfolds(y, fold)
    print(f"[HOTPATCH-Q] re-gating {stem} with fixed G0…", flush=True)
    try:
        rec = evaluate_engine_g0(
            stem,
            oof,
            te,
            y,
            fold,
            sf,
            z_fb,
            z_fb_te,
            z_twin,
            z_twin_te,
            Bo,
            Bt,
            eta_fb=eta,
            eta_fb_te=eta_te,
        )
    except Exception as e:
        print(f"[HOTPATCH-Q] REGATE FAIL {stem}: {e}", flush=True)
        return False
    atomic_json(gp, rec)
    print(
        f"[HOTPATCH-Q] REGATE {stem}: solo={rec.get('solo')} g0={rec.get('g0')} "
        f"vault={rec.get('vault')} r2={rec.get('r2_B152')} f1={rec.get('f1')} "
        f"f2={rec.get('f2_ok')} status={rec.get('status')}",
        flush=True,
    )
    return True


def _run_engine(eng: str, via: str) -> int:
    st = load_state()
    st.setdefault("engines", {})[eng] = {
        "status": "running",
        "ts": now_utc().isoformat(),
        "via": via,
    }
    save_state(st)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    print(f"[HOTPATCH-Q] launching {eng}", flush=True)
    p = subprocess.run([PY, WAVE, "--engine", eng], cwd=str(REPO), env=env)
    st = load_state()
    st.setdefault("engines", {})[eng] = {
        "status": "done" if p.returncode == 0 else "fail",
        "ts": now_utc().isoformat(),
        "code": p.returncode,
        "via": via,
    }
    save_state(st)
    return p.returncode


def _supervisor_alive() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-f", "gm_v36_1_true_9h_overnight"], text=True)
        return bool(out.strip())
    except Exception:
        return False


def main():
    print("[HOTPATCH-Q] watching E2–E4 re-gate + E3 retry after E4 + E1_fixed fallback", flush=True)
    regated = set()
    e3_retried = False
    while now_utc() < DEADLINE_UTC:
        st = load_state()
        engines = st.get("engines", {})

        for eng, stem in REGATE.items():
            if eng in regated:
                continue
            if engines.get(eng, {}).get("status") == "done":
                try:
                    if regate_if_needed(stem):
                        regated.add(eng)
                    elif (LOGS / "v36_1_ckpt" / f"{stem}_gate.json").exists():
                        regated.add(eng)
                except Exception as e:
                    print(f"[HOTPATCH-Q] regate loop err {eng}: {e}", flush=True)

        e4 = engines.get("W1_E4", {})
        e3 = engines.get("W1_E3", {})
        e1f = engines.get("W1_E1_fixed", {})

        # After E4: retry E3 if failed (old GS gate crash wiped preds)
        e3_live = False
        try:
            e3_live = bool(
                subprocess.check_output(["pgrep", "-f", "run_wave_engine.py --engine W1_E3"], text=True).strip()
            )
        except Exception:
            e3_live = False
        if e4.get("status") == "done" and e3.get("status") != "done":
            if e3_live or e3.get("status") == "running":
                # wait for in-flight E3 (do not launch a duplicate)
                pass
            elif not e3_retried:
                code = _run_engine("W1_E3", "hotpatch_e3_retry")
                e3_retried = True
                print(f"[HOTPATCH-Q] E3 retry code={code}", flush=True)
                st = load_state()
                engines = st.get("engines", {})
        # If E3 process exited, refresh status from preds
        if e4.get("status") == "done" and not e3_live and e3.get("status") == "running":
            stem = "W1_E3_hier_eb_3d4d_and_stealth_fn"
            ok = (PRED / f"{stem}_oof.npy").exists() and (PRED / f"{stem}_test.npy").exists()
            st = load_state()
            st.setdefault("engines", {})["W1_E3"] = {
                "status": "done" if ok else "fail",
                "ts": now_utc().isoformat(),
                "via": "hotpatch_e3_watch",
            }
            save_state(st)
            engines = st.get("engines", {})
            print(f"[HOTPATCH-Q] E3 watch → {st['engines']['W1_E3']['status']}", flush=True)

        if e1f.get("status") == "done":
            if engines.get("W3_JEWELS", {}).get("status") != "done":
                _run_engine("W3_JEWELS", "hotpatch_queue")
            print("[HOTPATCH-Q] W1_E1_fixed already done — exit", flush=True)
            return 0

        # E1_fixed: SUP's for-iterator may have predated hotpatch injection — always run if missing
        if (
            e4.get("status") == "done"
            and engines.get("W1_E3", {}).get("status") == "done"
            and e1f.get("status") not in ("running", "done")
        ):
            regate_if_needed(REGATE.get("W1_E4", "W1_E4_wilcoxon_twilight_mps_specialist"))
            code = _run_engine("W1_E1_fixed", "hotpatch_queue")
            if load_state().get("engines", {}).get("W3_JEWELS", {}).get("status") != "done":
                _run_engine("W3_JEWELS", "hotpatch_queue")
            # re-run E5 clean (no leaked E1) if prior E5 was contaminated/killed
            e5 = load_state().get("engines", {}).get("W3_E5", {})
            if e5.get("status") != "done":
                _run_engine("W3_E5", "hotpatch_e5_clean")
            return code

        time.sleep(45)
    print("[HOTPATCH-Q] deadline — exit", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
