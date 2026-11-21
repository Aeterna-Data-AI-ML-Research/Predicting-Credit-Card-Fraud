#!/usr/bin/env python3
"""GM V33.0 — RunPod 2×RTX4090 non-bank breakthrough orchestrator.

Preferred: single host gpuCount=2. Fallback: 1× then stream to Pod B if Fold-0 passes.
Zero-orphan teardown via atexit / SIGINT / finally + /tmp/s6e11_v33_active_pods.json.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[1]
BUNDLE_LOCAL = Path("/tmp/s6e11_v33_bundle")
PODS_FILE = Path("/tmp/s6e11_v33_active_pods.json")
CANARY_DIR = REPO / "preds" / "v33_canary"
FOLD5_DIR = REPO / "preds" / "v33_5fold"
REPORT = REPO / "logs" / "overnight" / "GM_V33_0_RUNPOD_CANARY_REPORT.md"
REMOTE_SCRIPTS = REPO / "scripts" / "v33_remote"
SSH_KEY = Path.home() / ".ssh" / "casmi_runpod"
API = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"  # legacy; prefer v2 via MCP when available
REST_V2 = "https://api.runpod.io/v2"

SECURE_2X_HR = 1.78
SECURE_1X_HR = 0.89

# Fold-0 gate
R2_MAX = 0.90
LIFT_MIN = 0.50e-4
SOLO_AUC_MIN = 0.9508
SUBFOLDS_NEED = 4


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def api_key() -> str:
    k = os.environ.get("RUNPOD_API_KEY", "")
    if k:
        return k
    cfg = Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            if line.strip().startswith("apikey"):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("RUNPOD_API_KEY missing")


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read().decode())
    if "errors" in out:
        raise RuntimeError(out["errors"])
    return out["data"]


def rest_v2(method: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{REST_V2}{path}",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read().decode()
        return json.loads(raw) if raw else {}


def load_pods() -> list[str]:
    if PODS_FILE.exists():
        return json.loads(PODS_FILE.read_text()).get("pods", [])
    return []


def save_pods(ids: list[str]) -> None:
    PODS_FILE.write_text(json.dumps({"pods": ids, "ts": time.time()}, indent=2))


def register_pod(pid: str) -> None:
    ids = load_pods()
    if pid not in ids:
        ids.append(pid)
        save_pods(ids)
        log(f"registered pod {pid} -> {PODS_FILE}")


def list_my_pod_ids() -> list[str]:
    data = gql("{ myself { pods { id name desiredStatus } } }")
    return [p["id"] for p in (data["myself"]["pods"] or [])]


def terminate_all_pods(reason: str = "") -> None:
    ids = sorted(set(load_pods() + list_my_pod_ids()))
    log(f"terminate_all_pods ({reason}): {ids}")
    for pid in ids:
        try:
            gql(
                "mutation($id:String!){ podTerminate(input:{podId:$id}) }",
                {"id": pid},
            )
            log(f"  terminated {pid}")
        except Exception as e:
            try:
                rest_v2("DELETE", f"/pods/{pid}")
                log(f"  rest-deleted {pid}")
            except Exception as e2:
                log(f"  fail terminate {pid}: {e} / {e2}")
    save_pods([])
    time.sleep(3)
    left = list_my_pod_ids()
    log(f"active pods after teardown: {left}")
    if left:
        log("WARNING: pods still active — retry")
        for pid in left:
            try:
                gql("mutation($id:String!){ podTerminate(input:{podId:$id}) }", {"id": pid})
            except Exception:
                pass
        time.sleep(2)
        left = list_my_pod_ids()
    assert left == [], f"orphan pods remain: {left}"


_CLEANED = False


def _cleanup_handler(*_a):
    global _CLEANED
    if _CLEANED:
        return
    _CLEANED = True
    try:
        terminate_all_pods("signal/atexit")
    except Exception as e:
        log(f"cleanup error: {e}")


atexit.register(_cleanup_handler)
signal.signal(signal.SIGINT, lambda *_: (_cleanup_handler(), sys.exit(130)))
signal.signal(signal.SIGTERM, lambda *_: (_cleanup_handler(), sys.exit(143)))


def create_pod(gpu_count: int, name: str) -> dict:
    hr = SECURE_2X_HR if gpu_count == 2 else SECURE_1X_HR
    log(f"CREATING pod name={name} gpuCount={gpu_count} SECURE RTX4090 @ ${hr:.2f}/hr")
    body = {
        "name": name,
        "cloud": "SECURE",
        "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "disk": 65,
        "ports": ["22/tcp"],
        "startSsh": True,
        "gpu": {
            "id": "NVIDIA GeForce RTX 4090",
            "count": gpu_count,
            "minCudaVersion": "12.8",
        },
        "env": {"V33_MODEL": "Qwen/Qwen2.5-1.5B"},
    }
    try:
        pod = rest_v2("POST", "/pods", body)
    except Exception as e:
        log(f"REST v2 create failed: {e}; trying GraphQL podFindAndDeployOnDemand")
        # GraphQL fallback
        q = """
        mutation($in: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $in) { id desiredStatus imageName machineId costPerHr }
        }"""
        variables = {
            "in": {
                "name": name,
                "imageName": body["image"],
                "gpuCount": gpu_count,
                "gpuTypeId": "NVIDIA GeForce RTX 4090",
                "cloudType": "SECURE",
                "containerDiskInGb": 65,
                "volumeInGb": 0,
                "ports": "22/tcp",
                "startSsh": True,
                "minCudaVersion": "12.8",
                "env": [{"key": "V33_MODEL", "value": "Qwen/Qwen2.5-1.5B"}],
            }
        }
        pod = gql(q, variables)["podFindAndDeployOnDemand"]
    pid = pod.get("id") or pod.get("podId")
    if not pid:
        raise RuntimeError(f"create returned no id: {pod}")
    register_pod(pid)
    return {"id": pid, "raw": pod, "hr": hr}


def wait_ssh(pod_id: str, timeout: int = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = gql(
            """query($id:String!){
              pod(input:{podId:$id}){
                id desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } }
                machine { gpuDisplayName gpuCount }
              }
            }""",
            {"id": pod_id},
        )
        pod = data["pod"]
        st = pod.get("desiredStatus")
        ports = ((pod.get("runtime") or {}).get("ports")) or []
        ssh = None
        for p in ports:
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                ssh = {"host": p["ip"], "port": int(p["publicPort"])}
                break
        log(f"  wait {pod_id} status={st} ssh={ssh}")
        if ssh and st == "RUNNING":
            # probe
            cmd = [
                "ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=8",
                "-p", str(ssh["port"]), f"root@{ssh['host']}", "echo OK",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and "OK" in r.stdout:
                return {"pod": pod, "ssh": ssh}
        time.sleep(8)
    raise TimeoutError(f"SSH not ready for {pod_id}")


def ssh_run(ssh: dict, cmd: str, check: bool = True, timeout: int = 7200) -> subprocess.CompletedProcess:
    full = [
        "ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=30",
        "-p", str(ssh["port"]), f"root@{ssh['host']}", cmd,
    ]
    log(f"SSH$ {cmd[:120]}…")
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed rc={r.returncode}\nSTDOUT:{r.stdout[-2000:]}\nSTDERR:{r.stderr[-2000:]}")
    return r


def scp_to(ssh: dict, local: Path, remote: str) -> None:
    cmd = [
        "scp", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-P", str(ssh["port"]),
        "-r", str(local), f"root@{ssh['host']}:{remote}",
    ]
    log(f"SCP {local} -> {remote}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)


def scp_from(ssh: dict, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-P", str(ssh["port"]),
        "-r", f"root@{ssh['host']}:{remote}", str(local),
    ]
    log(f"SCP {remote} -> {local}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)


def prepare_local_bundle() -> None:
    BUNDLE_LOCAL.mkdir(parents=True, exist_ok=True)
    # assume already prepared; re-check
    need = ["train.csv.gz", "test.csv.gz", "fold_id.npy", "wide_lm_oof.npy", "orders_4.npy"]
    for n in need:
        assert (BUNDLE_LOCAL / n).exists(), f"missing {BUNDLE_LOCAL/n}"
    fold = np.load(BUNDLE_LOCAL / "fold_id.npy")
    assert fold.shape == (699960,) and list(np.bincount(fold)) == [139992] * 5
    log("local bundle OK")


def rank_logit(x: np.ndarray) -> np.ndarray:
    r = pd.Series(x).rank(method="average").to_numpy()
    u = (r - 0.5) / len(r)
    return logit(np.clip(u, 1e-6, 1 - 1e-6))


def load_bank78_va0(va_mask: np.ndarray) -> np.ndarray:
    bank_dir = REPO / "external" / "banks" / "goodpjw2008_s6e11-oof-bank"
    cols = []
    for p in sorted(bank_dir.glob("oof_*.npy")):
        a = np.load(p)
        if a.ndim > 1:
            a = a.ravel()
        if len(a) == 699960:
            cols.append(a[va_mask].astype(np.float64))
    if not cols:
        raise RuntimeError("no bank oof cols")
    X = np.column_stack(cols)
    log(f"bank78 matrix cols={X.shape[1]} (using all available oof_*.npy)")
    return X


def eval_worker(name: str, pred: np.ndarray, y: np.ndarray, wide: np.ndarray, bank: np.ndarray, fold0_idx: np.ndarray) -> dict:
    # fold0_idx are positions within full train; pred is va0-length
    solo = float(roc_auc_score(y, pred))
    # R² vs bank78 via ridge
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(bank, pred)
    hat = ridge.predict(bank)
    ss_res = float(np.sum((pred - hat) ** 2))
    ss_tot = float(np.sum((pred - pred.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    # optimal w on wide
    best_w, best_lift = 0.0, -1e9
    base = float(roc_auc_score(y, wide))
    for w in np.linspace(0, 0.35, 71):
        b = (1 - w) * wide + w * pred
        lift = float(roc_auc_score(y, b) - base)
        if lift > best_lift:
            best_lift, best_w = lift, float(w)
    # 5 subfolds of va0 by hashing index
    pos = 0
    sub_pos = 0
    rng_idx = np.arange(len(pred))
    # use first 5 equal chunks of va0
    chunks = np.array_split(rng_idx, 5)
    for ch in chunks:
        if len(ch) < 100:
            continue
        b0 = roc_auc_score(y[ch], wide[ch])
        b1 = roc_auc_score(y[ch], (1 - best_w) * wide[ch] + best_w * pred[ch])
        if b1 - b0 >= LIFT_MIN * 0.5:  # soft per-subfold; count positive direction
            # stricter: require positive lift
            if b1 > b0:
                sub_pos += 1
        pos += 1
    # recount with absolute lift threshold relative to overall
    sub_pos = 0
    for ch in chunks:
        b0 = roc_auc_score(y[ch], wide[ch])
        b1 = roc_auc_score(y[ch], (1 - best_w) * wide[ch] + best_w * pred[ch])
        if (b1 - b0) >= LIFT_MIN:
            sub_pos += 1
    passed = (r2 < R2_MAX) and ((best_lift >= LIFT_MIN and sub_pos >= SUBFOLDS_NEED) or solo >= SOLO_AUC_MIN)
    out = {
        "name": name,
        "solo_auc": solo,
        "r2_bank": float(r2),
        "opt_w": best_w,
        "lift_e4": best_lift * 1e4,
        "pos_subfolds": f"{sub_pos}/5",
        "pass": passed,
    }
    log(f"GATE {name}: solo={solo:.6f} R²={r2:.4f} w={best_w:.3f} Δ={best_lift*1e4:.3f}e-4 sub={sub_pos}/5 PASS={passed}")
    return out


def write_report(lines: list[str]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    log(f"wrote {REPORT}")


def main() -> int:
    t_start = time.time()
    CANARY_DIR.mkdir(parents=True, exist_ok=True)
    FOLD5_DIR.mkdir(parents=True, exist_ok=True)
    prepare_local_bundle()
    bal = gql("{ myself { currentSpendPerHr clientBalance } }")
    bal_amt = bal["myself"].get("clientBalance")
    log(f"balance≈{bal_amt} spend/hr={bal['myself'].get('currentSpendPerHr')}")

    topology = "option1_2x"
    hr = SECURE_2X_HR
    pod_info = None
    try:
        pod_info = create_pod(2, "gm-v33-2x4090-canary")
    except Exception as e:
        log(f"2× create failed ({e}); falling back to Option-2 1× @ ${SECURE_1X_HR}/hr")
        topology = "option2_1x"
        hr = SECURE_1X_HR
        pod_info = create_pod(1, "gm-v33-1x4090-podA")

    pid = pod_info["id"]
    ssh = wait_ssh(pid)["ssh"]
    log(f"SSH ready {ssh['host']}:{ssh['port']} topology={topology} ${hr}/hr")

    # upload bundle + scripts
    ssh_run(ssh, "mkdir -p /workspace/bundle /workspace/scripts /workspace/canary /workspace/logs /workspace/pip_cache /workspace/hf_cache")
    for f in BUNDLE_LOCAL.iterdir():
        scp_to(ssh, f, f"/workspace/bundle/{f.name}")
    for f in REMOTE_SCRIPTS.iterdir():
        scp_to(ssh, f, f"/workspace/scripts/{f.name}")
    ssh_run(ssh, "chmod +x /workspace/scripts/setup_and_canary.sh")

    # launch canary in background with nohup so SSH drop doesn't kill it
    log("launching Fold-0 canary (va0-first)…")
    ssh_run(
        ssh,
        "nohup bash /workspace/scripts/setup_and_canary.sh > /workspace/logs/canary_master.log 2>&1 & echo $!",
        timeout=60,
    )

    # poll until CANARY_DONE or timeout
    deadline = time.time() + 5400
    while time.time() < deadline:
        r = ssh_run(ssh, "test -f /workspace/canary/CANARY_DONE && cat /workspace/canary/CANARY_DONE || echo WAIT", check=False, timeout=60)
        out = (r.stdout or "").strip()
        log(f"canary poll: {out[:200]}")
        if out and not out.startswith("WAIT"):
            break
        # tail logs
        r2 = ssh_run(ssh, "tail -n 5 /workspace/logs/canary_master.log /workspace/logs/worker_a.log /workspace/logs/worker_b.log 2>/dev/null || true", check=False, timeout=60)
        print(r2.stdout[-1500:], flush=True)
        time.sleep(45)
    else:
        log("CANARY TIMEOUT — teardown")
        write_report([
            "# GM V33.0 RUNPOD CANARY REPORT",
            "",
            f"- topology: {topology}",
            f"- result: TIMEOUT",
            f"- cost_est_usd: {(time.time()-t_start)/3600*hr:.3f}",
            "- submits: 0",
        ])
        terminate_all_pods("canary_timeout")
        return 2

    # pull predictions
    scp_from(ssh, "/workspace/canary/", CANARY_DIR)
    # flatten if nested
    nested = CANARY_DIR / "canary"
    if nested.is_dir():
        for f in nested.iterdir():
            dest = CANARY_DIR / f.name
            if f.is_file():
                f.replace(dest)

    # local gate
    tr = pd.read_csv(REPO / "data" / "train.csv", usecols=["is_fraud"])
    y_all = tr["is_fraud"].to_numpy(float)
    fold = np.load(BUNDLE_LOCAL / "fold_id.npy")
    wide = np.load(BUNDLE_LOCAL / "wide_lm_oof.npy")
    va = fold == 0
    y = y_all[va]
    w = wide[va]
    bank = load_bank78_va0(va)

    gates = []
    for tag, fname in [
        ("qwen_ep1", "qwen15b_ep1_va0.npy"),
        ("qwen_ep2", "qwen15b_ep2_va0.npy"),
        ("tabicl", "tabicl_va0.npy"),
        ("tabpfn", "tabpfn_va0.npy"),
    ]:
        p = CANARY_DIR / fname
        if p.exists():
            pred = np.load(p).ravel()
            assert len(pred) == va.sum(), (tag, len(pred), va.sum())
            gates.append(eval_worker(tag, pred, y, w, bank, np.where(va)[0]))
        else:
            log(f"missing {fname}")

    any_pass = any(g["pass"] for g in gates)
    cost = (time.time() - t_start) / 3600 * hr

    lines = [
        "# GM V33.0 RUNPOD CANARY REPORT",
        "",
        f"- topology: `{topology}`",
        f"- pod_id: `{pid}`",
        f"- hourly_usd: {hr}",
        f"- elapsed_min: {(time.time()-t_start)/60:.1f}",
        f"- cost_est_usd: {cost:.3f}",
        f"- balance_at_start: {bal_amt}",
        "",
        "## Fold-0 Gate",
        "",
        "| worker | solo_AUC | R²_bank78 | opt_w | ΔAUC e-4 | subfolds | PASS |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for g in gates:
        lines.append(
            f"| {g['name']} | {g['solo_auc']:.6f} | {g['r2_bank']:.4f} | {g['opt_w']:.3f} | "
            f"{g['lift_e4']:.3f} | {g['pos_subfolds']} | {g['pass']} |"
        )
    lines += ["", f"**any_pass={any_pass}**", ""]

    if not any_pass:
        lines += [
            "## Decision",
            "- FAIL Fold-0 gate → immediate teardown, 0 Kaggle submits, no test.csv scoring.",
        ]
        write_report(lines)
        terminate_all_pods("fold0_fail")
        assert list_my_pod_ids() == []
        log("Fold-0 FAIL — clean exit")
        return 0

    # PASS path: reset watchdog to 4h, scale-out folds 1-4
    lines += ["## Decision", "- PASS Fold-0 → arm 14400s watchdog + 5-fold scale-out."]
    ssh_run(
        ssh,
        "kill $(cat /workspace/canary/watchdog_canary.pid) 2>/dev/null; "
        "( sleep 14400; kill -9 -1 ) & echo $! > /workspace/canary/watchdog_full.pid; "
        "echo armed_full",
        check=False,
        timeout=30,
    )

    # For V33.0 hardened canary: if pass, attempt scale-out via remote script if present;
    # otherwise stage honest composite from Fold-0-only signal is NOT allowed for submit.
    # Scale-out: train remaining folds — upload scaleout launcher.
    scaleout = REMOTE_SCRIPTS / "scaleout_5fold.sh"
    if scaleout.exists():
        scp_to(ssh, scaleout, "/workspace/scripts/scaleout_5fold.sh")
        ssh_run(ssh, "nohup bash /workspace/scripts/scaleout_5fold.sh > /workspace/logs/scaleout.log 2>&1 & echo $!", timeout=30)
        deadline = time.time() + 14400
        while time.time() < deadline:
            r = ssh_run(ssh, "test -f /workspace/canary/SCALEOUT_DONE && cat /workspace/canary/SCALEOUT_DONE || echo WAIT", check=False, timeout=60)
            if r.stdout and "WAIT" not in r.stdout:
                break
            time.sleep(90)
        scp_from(ssh, "/workspace/canary/", FOLD5_DIR)

    # Immediate terminate before local stacking / submit
    terminate_all_pods("post_scaleout")
    assert list_my_pod_ids() == []
    cost = (time.time() - t_start) / 3600 * hr
    lines.append(f"- final_cost_est_usd: {cost:.3f}")
    lines.append("- pods_after: []")
    lines.append("- note: 5-fold artifacts pulled if scaleout completed; Kaggle submit gated in postprocess.")
    write_report(lines)
    log("PASS path complete — pods terminated")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log(f"FATAL: {e}")
        try:
            terminate_all_pods("fatal")
        except Exception:
            pass
        # still write a report
        try:
            write_report([
                "# GM V33.0 RUNPOD CANARY REPORT",
                "",
                f"- FATAL: {e}",
                "- submits: 0",
            ])
        except Exception:
            pass
        raise
