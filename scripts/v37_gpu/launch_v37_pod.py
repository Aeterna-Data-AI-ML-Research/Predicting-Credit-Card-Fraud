#!/usr/bin/env python3
"""V37 Mac controller — provision TFM+LM pod, rsync-before-delete, foreign-pod lock.

Never touches pods outside the newly created `v37-gm-tfm-lm` id.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_LOCAL = REPO / "preds" / "v37_gpu"
LOG_LOCAL = REPO / "logs" / "overnight" / "v37_fold0"
STATE = REPO / "logs" / "overnight" / "v37_pod_state.json"
POD_NAME = "v37-gm-tfm-lm"
MAX_USD_HR = 0.89
WATCHDOG_F0 = 7200
WATCHDOG_PROMOTE = 14400
N_TEST = 299_984

# Prefer 48GB; reject <24GB. Cap <= $0.89/hr.
GPU_PRIORITY = [
    # (gpuTypeId, cloud, max_hr, prefer_dcs)
    ("NVIDIA L40S", "COMMUNITY", 0.79, []),
    ("NVIDIA A40", "SECURE", 0.59, ["CA-MTL-1", "EU-SE-1"]),
    ("NVIDIA RTX A6000", "SECURE", 0.69, ["EU-SE-1"]),
    ("NVIDIA L40", "SECURE", 0.82, ["US-KS-2"]),
    ("NVIDIA GeForce RTX 4090", "SECURE", 0.89, ["EU-RO-1", "EUR-IS-1", "US-IL-1"]),
]

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
RUNPODCTL = shutil.which("runpodctl") or str(Path.home() / ".local/bin/runpodctl")


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd, check=True, capture=False, env=None):
    log(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
        env=env or os.environ.copy(),
    )


def list_pod_ids() -> set[str]:
    r = run([RUNPODCTL, "get", "pod"], capture=True, check=False)
    ids = set(re.findall(r"\b([a-z0-9]{10,})\b", r.stdout + r.stderr))
    # also try json
    r2 = run([RUNPODCTL, "get", "pod", "-o", "json"], capture=True, check=False)
    try:
        data = json.loads(r2.stdout or "[]")
        if isinstance(data, list):
            for p in data:
                if isinstance(p, dict) and p.get("id"):
                    ids.add(p["id"])
        elif isinstance(data, dict):
            for p in data.get("pods", data.get("items", [])):
                if p.get("id"):
                    ids.add(p["id"])
    except Exception:
        pass
    return ids


def load_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in [Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"]:
        if p.exists():
            t = p.read_text().strip()
            if t:
                return t
    return None


def create_pod(gpu_type: str, cloud: str, dcs: list[str]) -> str:
    env_args = [
        "NO_SUBMIT=True",
        "ALLOW_KAGGLE_SUBMIT=False",
        "COPYFILE_DISABLE=1",
        "MPLBACKEND=Agg",
        "V37_REPO=/workspace/repo",
        "V37_OUT=/workspace/out",
        "USE_COMPILE=0",
    ]
    hf = load_hf_token()
    if hf:
        env_args.append(f"HF_TOKEN={hf}")
        env_args.append(f"HUGGING_FACE_HUB_TOKEN={hf}")
        log("[launch] HF_TOKEN injected into pod env")
    else:
        log("[launch] WARNING: no HF_TOKEN — TabPFN may TABPFN_AUTH_SKIP")

    cmd = [
        RUNPODCTL,
        "create",
        "pod",
        "--name",
        POD_NAME,
        "--gpuType",
        gpu_type,
        "--gpuCount",
        "1",
        "--imageName",
        IMAGE,
        "--containerDiskSize",
        "50",
        "--volumeSize",
        "80",
        "--volumePath",
        "/workspace",
        "--ports",
        "22/tcp",
        "--startSSH",
        "--args",
        "bash -c 'sleep infinity'",
    ]
    if cloud == "SECURE":
        cmd.append("--secureCloud")
    else:
        cmd.append("--communityCloud")
    for e in env_args:
        cmd.extend(["--env", e])
    if dcs:
        cmd.extend(["--dataCenterId", dcs[0]])

    r = run(cmd, capture=True, check=False)
    out = (r.stdout or "") + (r.stderr or "")
    log(out[-2000:])
    m = re.search(r"(?:pod\s+)?([a-z0-9]{10,})", out, re.I)
    # Prefer explicit id patterns from runpodctl
    m2 = re.findall(r"\b([a-z0-9]{14,})\b", out)
    pod_id = None
    if m2:
        pod_id = m2[-1]
    elif m:
        pod_id = m.group(1)
    if not pod_id or r.returncode != 0:
        raise RuntimeError(f"create failed rc={r.returncode}: {out[-1500:]}")
    return pod_id


def wait_ssh(pod_id: str, timeout=600) -> str:
    """Return ssh destination user@host."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = run([RUNPODCTL, "get", "pod", pod_id, "-o", "json"], capture=True, check=False)
        try:
            data = json.loads(r.stdout or "{}")
            if isinstance(data, list):
                data = data[0] if data else {}
        except Exception:
            data = {}
        # try ssh field
        ssh = data.get("ssh") or {}
        direct = ssh.get("direct") if isinstance(ssh, dict) else None
        proxy = ssh.get("proxy") if isinstance(ssh, dict) else None
        if isinstance(direct, dict) and direct.get("ip"):
            return f"root@{direct['ip']}", direct.get("port", 22)
        if isinstance(proxy, dict) and proxy.get("command"):
            # ssh.runpod.io style — use runpodctl ssh helper if available
            return proxy.get("command"), None
        # fallback parse get pod text
        r2 = run([RUNPODCTL, "get", "pod", pod_id], capture=True, check=False)
        text = r2.stdout + r2.stderr
        if "RUNNING" in text.upper() or "running" in text:
            m = re.search(r"(\S+@ssh\.runpod\.io)", text)
            if m:
                return m.group(1), None
            m = re.search(r"(\d+\.\d+\.\d+\.\d+):(\d+)", text)
            if m:
                return f"root@{m.group(1)}", int(m.group(2))
        time.sleep(8)
    raise TimeoutError(f"SSH not ready for {pod_id}")


def ssh_cmd(dest, port, remote: str, check=True):
    if dest.startswith("ssh "):
        # full command from proxy
        base = dest.split()
        return run(base + [remote], check=check)
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    if port:
        cmd += ["-p", str(port)]
    cmd += [dest, remote]
    return run(cmd, check=check)


def rsync_to(dest, port, local: str, remote: str):
    if "@ssh.runpod.io" in str(dest) or (isinstance(dest, str) and dest.startswith("ssh ")):
        # use runpodctl receive/send if available — fall back to scp via proxy not supported for rsync easily
        # Prefer: tar over ssh proxy command
        userhost = dest if not dest.startswith("ssh ") else dest.split()[-1]
        # runpodctl ssh
        run(
            [
                "bash",
                "-lc",
                f"tar czf - -C {Path(local).parent} {Path(local).name} | "
                f"ssh -o StrictHostKeyChecking=no {userhost} 'mkdir -p {Path(remote).parent} && tar xzf - -C {Path(remote).parent}'",
            ],
            check=True,
        )
        return
    cmd = [
        "rsync",
        "-az",
        "--progress",
        "-e",
        f"ssh -o StrictHostKeyChecking=no -p {port or 22}",
        local.rstrip("/") + "/",
        f"{dest}:{remote.rstrip('/')}/",
    ]
    run(cmd, check=True)


def rsync_from(dest, port, remote: str, local: Path):
    local.mkdir(parents=True, exist_ok=True)
    if isinstance(dest, str) and ("ssh.runpod.io" in dest or dest.startswith("ssh ")):
        userhost = dest if not dest.startswith("ssh ") else dest.split()[-1]
        run(
            [
                "bash",
                "-lc",
                f"ssh -o StrictHostKeyChecking=no {userhost} 'tar czf - -C {Path(remote).parent} {Path(remote).name}' | "
                f"tar xzf - -C {local.parent}",
            ],
            check=False,
        )
        # if remote is /workspace/out, extract as local.name
        return
    cmd = [
        "rsync",
        "-az",
        "--progress",
        "-e",
        f"ssh -o StrictHostKeyChecking=no -p {port or 22}",
        f"{dest}:{remote.rstrip('/')}/",
        str(local).rstrip("/") + "/",
    ]
    run(cmd, check=False)


def verify_npy_dir(d: Path) -> list[str]:
    import numpy as np

    problems = []
    for p in d.rglob("*.npy"):
        if p.name.startswith("._"):
            continue
        try:
            a = np.load(p)
            if a.size == 0:
                problems.append(f"{p}: empty")
            elif not np.isfinite(a.astype(float)).all():
                problems.append(f"{p}: NaN/Inf")
            if p.name.endswith("_test.npy") and a.reshape(-1).shape[0] not in (N_TEST, N_TEST * 4 * 2):
                # allow ll_raw shapes
                if "ll_raw" not in p.name and a.reshape(-1).shape[0] != N_TEST:
                    if a.shape != (N_TEST,) and not (len(a.shape) == 3):
                        problems.append(f"{p}: unexpected shape {a.shape}")
        except Exception as e:
            problems.append(f"{p}: {e}")
    return problems


def safe_remove(pod_id: str, foreign: set[str]):
    if pod_id in foreign:
        log(f"[FATAL] refused to remove foreign pod {pod_id}")
        return
    if not pod_id:
        return
    log(f"[launch] removing OUR pod {pod_id}")
    run([RUNPODCTL, "remove", "pod", pod_id], check=False)


def arm_watchdog(dest, port, seconds: int):
    remote = (
        f"kill $(cat /tmp/watchdog.pid) 2>/dev/null; "
        f"(sleep {seconds} && runpodctl remove pod $RUNPOD_POD_ID) & echo $! > /tmp/watchdog.pid; "
        f"echo armed:{seconds}:pid=$(cat /tmp/watchdog.pid)"
    )
    ssh_cmd(dest, port, remote, check=False)


def kill_watchdog(dest, port):
    ssh_cmd(dest, port, "kill $(cat /tmp/watchdog.pid) 2>/dev/null; rm -f /tmp/watchdog.pid; echo killed", check=False)


def provision() -> tuple[str, set[str]]:
    foreign = list_pod_ids()
    log(f"[launch] FOREIGN_PODS={sorted(foreign)}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for gpu, cloud, price, dcs in GPU_PRIORITY:
        if price > MAX_USD_HR:
            continue
        log(f"[launch] trying {gpu} {cloud} @ ${price}/hr (cap ${MAX_USD_HR})")
        try:
            before = list_pod_ids()
            pod_id = create_pod(gpu, cloud, dcs)
            time.sleep(3)
            after = list_pod_ids()
            new = after - before - foreign
            if pod_id in foreign:
                log("[launch] created id collided with foreign — abort")
                continue
            # confirm
            log(f"[launch] CREATED pod_id={pod_id} gpu={gpu} ${price}/hr NEW={new}")
            STATE.write_text(
                json.dumps(
                    {
                        "pod_id": pod_id,
                        "gpu": gpu,
                        "cloud": cloud,
                        "usd_hr": price,
                        "foreign": sorted(foreign),
                        "name": POD_NAME,
                    },
                    indent=2,
                )
            )
            return pod_id, foreign
        except Exception as e:
            last_err = e
            log(f"[launch] {gpu} failed: {e}")
    raise RuntimeError(f"No GPU under ${MAX_USD_HR}/hr available: {last_err}")


def sync_code(dest, port):
    # essential paths only
    remote_repo = "/workspace/repo"
    ssh_cmd(dest, port, f"mkdir -p {remote_repo}/data {remote_repo}/preds/v36_1 {remote_repo}/scripts/v37_gpu "
            f"{remote_repo}/external/banks/goodpjw2008_s6e11-oof-bank "
            f"{remote_repo}/external/v37_twin_check/candidates/uditjain_orig /workspace/out", check=False)
    # use tar pipelines for reliability over runpod ssh
    paths = [
        ("data/train.csv", "data/"),
        ("data/test.csv", "data/"),
        ("external/banks/goodpjw2008_s6e11-oof-bank/folds.csv", "external/banks/goodpjw2008_s6e11-oof-bank/"),
        ("external/v37_twin_check/candidates/uditjain_orig/credit_card_fraud_2026.csv",
         "external/v37_twin_check/candidates/uditjain_orig/"),
        ("scripts/v37_gpu/run_fold0_tfm_and_transductive_lm.py", "scripts/v37_gpu/"),
        ("preds/v36_1/z_fullbank_oof.npy", "preds/v36_1/"),
        ("preds/v36_1/z_ATH_twin_oof.npy", "preds/v36_1/"),
        ("preds/v36_1/z_fullbank_test.npy", "preds/v36_1/"),
        ("preds/v36_1/B_clean_oof.npy", "preds/v36_1/"),
        ("preds/v36_1/B_clean_test.npy", "preds/v36_1/"),
    ]
    for rel, dest_dir in paths:
        lp = REPO / rel
        if not lp.exists():
            log(f"[sync] MISSING {lp}")
            continue
        log(f"[sync] {rel}")
        if isinstance(dest, str) and "ssh.runpod.io" in dest:
            userhost = dest
            run(
                [
                    "bash",
                    "-lc",
                    f"tar czf - -C {lp.parent} {lp.name} | "
                    f"ssh -o StrictHostKeyChecking=no {userhost} "
                    f"'mkdir -p {remote_repo}/{dest_dir} && tar xzf - -C {remote_repo}/{dest_dir}'",
                ],
                check=True,
            )
        else:
            run(
                [
                    "rsync",
                    "-az",
                    "-e",
                    f"ssh -o StrictHostKeyChecking=no -p {port or 22}",
                    str(lp),
                    f"{dest}:{remote_repo}/{dest_dir}",
                ],
                check=True,
            )


def pull_out(dest, port) -> Path:
    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    LOG_LOCAL.mkdir(parents=True, exist_ok=True)
    if isinstance(dest, str) and "ssh.runpod.io" in dest:
        run(
            [
                "bash",
                "-lc",
                f"ssh -o StrictHostKeyChecking=no {dest} 'tar czf - -C /workspace out' | tar xzf - -C {OUT_LOCAL.parent}",
            ],
            check=False,
        )
        # extracted as OUT_LOCAL.parent/out — move
        extracted = OUT_LOCAL.parent / "out"
        if extracted.exists():
            for p in extracted.iterdir():
                tgt = OUT_LOCAL / p.name
                if tgt.exists():
                    if tgt.is_dir():
                        shutil.rmtree(tgt)
                    else:
                        tgt.unlink()
                shutil.move(str(p), str(tgt))
    else:
        rsync_from(dest, port, "/workspace/out", OUT_LOCAL)
    # mirror gates to logs
    for g in OUT_LOCAL.rglob("gates.json"):
        shutil.copy(g, LOG_LOCAL / f"{g.parent.name}_{g.name}")
    for g in OUT_LOCAL.rglob("PROMOTE.json"):
        shutil.copy(g, LOG_LOCAL / f"{g.parent.name}_{g.name}")
    return OUT_LOCAL


def map_canaries_to_flags(promote: list[str]) -> str:
    flags = set()
    for p in promote:
        if p.startswith("gpt2"):
            flags.add("gpt2")
        elif p in ("tabpfn", "tabicl", "tfm_blend"):
            flags.add("tabpfn")
            flags.add("tabicl")
    return ",".join(sorted(flags)) if flags else ""


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--attach", default="", help="Attach to existing v37 pod id (skip create)")
    ap.add_argument("--foreign", default="", help="Comma-separated foreign pod ids to never touch")
    args = ap.parse_args()
    assert Path(RUNPODCTL).exists(), f"runpodctl missing: {RUNPODCTL}"
    os.environ.setdefault("NO_SUBMIT", "True")
    pod_id = None
    foreign: set[str] = set()
    dest = port = None
    try:
        if args.attach:
            foreign = list_pod_ids() - {args.attach}
            if args.foreign:
                foreign |= set(x.strip() for x in args.foreign.split(",") if x.strip())
            pod_id = args.attach
            if pod_id in foreign:
                raise SystemExit(f"refusing attach to foreign pod {pod_id}")
            log(f"[launch] ATTACH pod_id={pod_id} FOREIGN={sorted(foreign)}")
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps({"pod_id": pod_id, "foreign": sorted(foreign), "attach": True}, indent=2))
        else:
            pod_id, foreign = provision()
        dest, port = wait_ssh(pod_id)
        log(f"[launch] SSH dest={dest} port={port}")
        # inject RUNPOD_POD_ID for watchdog
        ssh_cmd(dest, port, f"export RUNPOD_POD_ID={pod_id}; echo {pod_id} > /etc/runpod_pod_id; "
                f"grep -q RUNPOD_POD_ID /etc/environment || echo RUNPOD_POD_ID={pod_id} >> /etc/environment; "
                f"pip -q install -U tabpfn tabicl transformers accelerate 2>/dev/null | tail -3; true", check=False)
        arm_watchdog(dest, port, WATCHDOG_F0)
        sync_code(dest, port)
        # start Fold-0 worker in background
        ssh_cmd(
            dest,
            port,
            f"cd /workspace/repo && export RUNPOD_POD_ID={pod_id} NO_SUBMIT=True V37_OUT=/workspace/out V37_REPO=/workspace/repo "
            f"&& nohup python -u scripts/v37_gpu/run_fold0_tfm_and_transductive_lm.py --folds 0 "
            f"> /workspace/out/fold0.log 2>&1 & echo $! > /workspace/out/worker.pid && echo STARTED",
            check=False,
        )
        # poll for DONE
        t0 = time.time()
        while time.time() - t0 < WATCHDOG_F0 - 120:
            ssh_cmd(dest, port, "test -f /workspace/out/DONE && echo DONE || (tail -3 /workspace/out/fold0.log; echo WAIT)", check=False)
            pull_out(dest, port)
            if (OUT_LOCAL / "DONE").exists() or (OUT_LOCAL / "fold_0" / "PROMOTE.json").exists():
                break
            time.sleep(60)
        pull_out(dest, port)
        problems = verify_npy_dir(OUT_LOCAL)
        log(f"[launch] verify problems={problems[:10]}")
        promote = []
        pj = OUT_LOCAL / "fold_0" / "PROMOTE.json"
        if pj.exists():
            promote = json.loads(pj.read_text()).get("promote", [])
        log(f"[launch] Fold-0 promote={promote}")
        (LOG_LOCAL / "fold0_controller.json").write_text(
            json.dumps({"pod_id": pod_id, "promote": promote, "problems": problems}, indent=2)
        )

        if not promote:
            log("[launch] Fold-0 FAIL gate — rsynced; deleting pod in finally")
            return 2

        # PASS: kill 2h watchdog, arm 4h, run folds 1-4
        kill_watchdog(dest, port)
        arm_watchdog(dest, port, WATCHDOG_PROMOTE)
        cans = map_canaries_to_flags(promote)
        log(f"[launch] promoting canaries={cans} folds 1-4")
        for k in [1, 2, 3, 4]:
            ssh_cmd(
                dest,
                port,
                f"cd /workspace/repo && export RUNPOD_POD_ID={pod_id} NO_SUBMIT=True V37_OUT=/workspace/out "
                f"&& python -u scripts/v37_gpu/run_fold0_tfm_and_transductive_lm.py --folds {k} --canaries {cans} "
                f"> /workspace/out/fold{k}.log 2>&1",
                check=False,
            )
            pull_out(dest, port)
            probs = verify_npy_dir(OUT_LOCAL / f"fold_{k}")
            log(f"[launch] fold {k} verify={probs}")
            if probs:
                log(f"[launch] fold {k} NaN/verify fail — stop promote loop")
                return 3
        log("[launch] folds 1-4 complete — delete pod in finally")
        return 0
    finally:
        if pod_id:
            # always rsync once more before delete
            try:
                if dest:
                    pull_out(dest, port)
            except Exception as e:
                log(f"[launch] final rsync err: {e}")
            safe_remove(pod_id, foreign)


if __name__ == "__main__":
    raise SystemExit(main())
