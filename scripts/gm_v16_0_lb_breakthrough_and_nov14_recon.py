#!/usr/bin/env python3
"""GM V16.0 — Morning recon: LB breakthrough forensics + V15 targeting fix.

STRICT NO_SUBMIT=True. Does not call kaggle competitions submit.
Stages verified Family A/E NOSUBMIT CSVs at 0.10/0.20/0.35 e-4 via unitize().

Report: logs/overnight/GM_V16_0_MORNING_RECON_AND_LEAK_AUDIT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

NO_SUBMIT = True
REPO = Path(os.environ.get("V16_REPO", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
TMP15 = Path(os.environ.get("V15_TMP", "/tmp/s6e11_v15"))
OUT = Path(os.environ.get("V16_OUT", "/tmp/s6e11_v16/out"))
EXT = REPO / "external/nov14_recon"
SUB = REPO / "submissions"
REPORT = REPO / "logs/overnight/GM_V16_0_MORNING_RECON_AND_LEAK_AUDIT.md"
LIVE_CSV = REPO / "submissions/sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
L7_DIR = REPO / "preds/l7_s2026_5f"
KAG = os.environ.get("KAGGLE_BIN", "/Users/mz/Library/Python/3.14/bin/kaggle")


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def L(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    r = (rankdata(v, method="average") - 0.5) / len(v)
    r = np.clip(r, 1e-12, 1 - 1e-12)
    return np.log(r / (1 - r))


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def quantile_lock(raw, live):
    live = np.asarray(live, float).ravel()
    raw = np.asarray(raw, float).ravel()
    order = np.argsort(raw)
    locked = np.empty_like(live)
    locked[order] = np.sort(live)
    return locked


def env_kaggle(cmd: list[str]) -> str:
    env = os.environ.copy()
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    r = subprocess.run([KAG, *cmd], capture_output=True, text=True, env=env)
    return (r.stdout or "") + (r.stderr or "")


def kill_watcher(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
        log(f"killed watcher PID {pid}")
    except ProcessLookupError:
        log(f"watcher PID {pid} already gone")


def pull_lb(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    txt = env_kaggle(["competitions", "leaderboard", "playground-series-s6e11", "-s", "-v"])
    p = out / "lb_top40.csv"
    p.write_text(txt)
    log(f"wrote {p}")
    return p


def pull_datasets_list(out: Path) -> None:
    txt = env_kaggle(["datasets", "list", "-s", "s6e11", "--sort-by", "updated"])
    (out / "datasets_s6e11.txt").write_text(txt)
    log("datasets list saved")


def file_tree_meta(root: Path) -> dict:
    d = {}
    if not root.exists():
        return d
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith("._"):
            d[p.name] = {"size": p.stat().st_size, "sha256": sha256_file(p)}
    return d


def diff_daoviet() -> dict:
    cur = EXT / "daoviet_apex_1652"
    pri = EXT / "prior_daoviet"
    if not cur.exists() or not any(cur.glob("*.csv")):
        EXT.mkdir(parents=True, exist_ok=True)
        cur.mkdir(parents=True, exist_ok=True)
        log("downloading daoviet/s6e11-gp-dvae-orthogonal-apex …")
        env_kaggle(
            [
                "datasets",
                "download",
                "-d",
                "daoviet/s6e11-gp-dvae-orthogonal-apex",
                "-p",
                str(cur),
                "--unzip",
            ]
        )
    if not any(pri.glob("*")):
        src = REPO / "external/nov13_recon/daoviet_gp_dvae_apex_nov13"
        pri.mkdir(parents=True, exist_ok=True)
        if src.exists():
            import shutil

            for f in src.iterdir():
                if f.is_file() and not f.name.startswith("._"):
                    shutil.copy2(f, pri / f.name)
    C, P = file_tree_meta(cur), file_tree_meta(pri)
    live = pd.read_csv(LIVE_CSV)
    ids = live["id"].to_numpy()
    lv = live.set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    metrics = {}
    for p in sorted(cur.glob("*.csv")):
        if p.name.startswith("._"):
            continue
        df = pd.read_csv(p)
        c = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
        v = df.set_index("id").loc[ids, c].to_numpy(float)
        metrics[p.name] = {
            "1mrho_LIVE": float(1.0 - spearmanr(v, lv).statistic),
            "size": p.stat().st_size,
            "sha256": sha256_file(p),
        }
    oof_auc = None
    oof_p = pri / "oof_ours.npy"
    if oof_p.exists():
        y = pd.read_csv(TMP15 / "train.csv" if (TMP15 / "train.csv").exists() else REPO / "data/train.csv")[
            "is_fraud"
        ].to_numpy()
        oof = np.load(oof_p).ravel()
        if oof.shape[0] == y.shape[0]:
            oof_auc = float(roc_auc_score(y, oof))
    return {
        "current": C,
        "prior": P,
        "removed": sorted(set(P) - set(C)),
        "added": sorted(set(C) - set(P)),
        "metrics_vs_LIVE": metrics,
        "prior_oof_ours_auc": oof_auc,
    }


def leak_audit() -> dict:
    train_p = TMP15 / "train.csv" if (TMP15 / "train.csv").exists() else REPO / "data/train.csv"
    test_p = TMP15 / "test_ids.csv" if (TMP15 / "test_ids.csv").exists() else REPO / "data/test.csv"
    train = pd.read_csv(train_p)
    test = pd.read_csv(test_p)
    y = train["is_fraud"].to_numpy()
    r_id, _ = pointbiserialr(y, train["id"].to_numpy())
    mod = {f"id%{K}": float(pointbiserialr(y, (train["id"] % K).to_numpy())[0]) for K in (2, 5, 10, 16, 100)}
    ord_ = np.argsort(train["id"].to_numpy())
    ys = y[ord_]
    lag = np.zeros_like(ys, float)
    lag[1:] = ys[:-1]
    auc_lag = float(roc_auc_score(ys[1:], lag[1:]))
    feat = [c for c in train.columns if c not in ("id", "is_fraud")]
    hk_tr = pd.util.hash_pandas_object(train[feat], index=False).to_numpy()
    hk_te = pd.util.hash_pandas_object(test[feat], index=False).to_numpy()
    set_tr = set(hk_tr)
    n_match = int(sum(1 for h in hk_te if h in set_tr))
    from collections import Counter

    ctr = Counter(hk_tr)
    train_dup_rows = int(sum(v for v in ctr.values() if v > 1))
    orig_p = REPO / "original/credit_card_fraud_2026.csv"
    test_orig = None
    if orig_p.exists():
        orig = pd.read_csv(orig_p)
        shared = [c for c in feat if c in orig.columns]
        ho = set(pd.util.hash_pandas_object(orig[shared].reset_index(drop=True), index=False))
        ht = pd.util.hash_pandas_object(test[shared].reset_index(drop=True), index=False).to_numpy()
        test_orig = int(sum(1 for h in ht if h in ho))
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "pb_id": float(r_id),
        "id_mod": mod,
        "auc_y_im1": auc_lag,
        "train_dup_rows": train_dup_rows,
        "test_exact_match_train": n_match,
        "test_exact_match_train_pct": float(100.0 * n_match / max(len(hk_te), 1)),
        "test_exact_match_orig": test_orig,
    }


def family_directions():
    """Rebuild Family A/E test directions from V15 staging if present, else from L7 5f."""
    live = pd.read_csv(LIVE_CSV)["is_fraud"].to_numpy(float)
    # Prefer V15 cached family directions
    cache = TMP15 / "out" / "family_dirs.npz"
    if cache.exists():
        z = np.load(cache)
        return live, {k: z[k] for k in z.files if k.startswith("d")}
    # Reconstruct from L7 folds: R3 antithetic mean of 5f as Family A direction vs LIVE lock base
    tests = [np.load(L7_DIR / f"test_{i}.npy").ravel() for i in range(5)]
    oofs = [np.load(L7_DIR / f"oof_{i}.npy").ravel() for i in range(5)]
    # mean of 5 fold test preds as L7F; R3 = antithetic if raw ll available else mean
    L7F = unitize(np.mean(np.column_stack(tests), axis=1))
    # Family A: pure swap direction = L(R3) - L(L7F); use fold0 vs mean as proxy if R3 absent
    R3 = unitize(0.5 * (tests[0] + tests[1]))  # fallback; real R3 from V15 preferred
    dA = unitize(L(R3) - L(L7F))
    # Family E: tail-weighted — weight by LIVE extremity
    w_tail = 0.5 + 0.5 * unitize(np.abs(L(live)))
    w_tail = (w_tail - w_tail.min()) / (w_tail.max() - w_tail.min() + 1e-12)
    dE = unitize((L(R3) - L(L7F)) * w_tail)
    # Prefer phase2 artifacts
    for name, key in (("dA_t.npy", "dA"), ("dE_t.npy", "dE")):
        p = TMP15 / "out" / name
        if p.exists():
            if key == "dA":
                dA = unitize(np.load(p).ravel())
            else:
                dE = unitize(np.load(p).ravel())
    return live, {"dA": dA, "dE": dE}


def bisect_target(live, direction, target: float, tol: float = 0.005e-4):
    u = unitize(direction)
    lo, hi = 0.0, 8.0
    best_w, best_dist, best_cand = 0.0, 0.0, live.copy()
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        cand = quantile_lock(L(live) + mid * u, live)
        dist = 1.0 - float(spearmanr(live, cand).correlation)
        best_w, best_dist, best_cand = mid, dist, cand
        if dist > target:
            hi = mid
        else:
            lo = mid
    err = abs(best_dist - target)
    assert err < tol, f"targeting failed: actual={best_dist} target={target} err={err}"
    return best_cand, best_w, best_dist, err


def stage_nosubmit(live, dirs: dict) -> list[dict]:
    assert NO_SUBMIT is True
    SUB.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(LIVE_CSV)["id"].to_numpy()
    targets = [0.100e-4, 0.200e-4, 0.350e-4]
    tags = ["0p10", "0p20", "0p35"]
    fam_map = [("FamilyA", dirs["dA"]), ("FamilyE", dirs["dE"])]
    cands = []
    for fam, d in fam_map:
        for tag, tgt in zip(tags, targets):
            cand, w, dist, err = bisect_target(live, d, tgt)
            name = f"sub_v16_0_{fam}_{tag}e4_NOSUBMIT.csv"
            for dest in (SUB / name, OUT / name):
                pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(dest, index=False)
            rec = {
                "file": name,
                "family": fam,
                "target": tgt,
                "w": w,
                "1mrho": dist,
                "err": err,
                "NO_SUBMIT": True,
            }
            cands.append(rec)
            log(f"  staged {name} w={w:.6f} 1-ρ={dist:.6e} err={err:.3e}")
    return cands


def measure_v15_rungs() -> dict:
    out = {}
    for key, pat in (
        ("rung1", "sub_v15_LIVE_rung1"),
        ("rung2", "sub_v15_LIVE_rung2"),
    ):
        hits = list(SUB.glob(f"{pat}*.csv")) + list((TMP15 / "out").glob(f"{pat}*.csv"))
        if not hits:
            continue
        live = pd.read_csv(LIVE_CSV)["is_fraud"].to_numpy(float)
        v = pd.read_csv(hits[0])["is_fraud"].to_numpy(float)
        out[key] = float(1.0 - spearmanr(live, v).correlation)
        out[f"{key}_file"] = str(hits[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kill-watcher-pid", type=int, default=25580)
    ap.add_argument("--skip-kaggle-list", action="store_true")
    ap.add_argument("--restage-only", action="store_true")
    args = ap.parse_args()
    assert NO_SUBMIT is True, "NO_SUBMIT guard"
    log(f"NO_SUBMIT={NO_SUBMIT}")
    if args.kill_watcher_pid:
        kill_watcher(args.kill_watcher_pid)

    OUT.mkdir(parents=True, exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)
    (REPO / "logs/overnight").mkdir(parents=True, exist_ok=True)

    if not args.skip_kaggle_list and not args.restage_only:
        try:
            pull_lb(OUT)
            pull_datasets_list(OUT)
        except Exception as e:
            log(f"kaggle list warn: {e}")

    dao = {} if args.restage_only else diff_daoviet()
    leak = {} if args.restage_only else leak_audit()
    rung = measure_v15_rungs()
    live, dirs = family_directions()
    cands = stage_nosubmit(live, dirs)

    payload = {
        "NO_SUBMIT": True,
        "leak": leak,
        "daoviet": dao,
        "cands": cands,
        "bug": "V15 build_rung used L2 unit() so direction std~0.0018; w<=0.08 => 1-ρ~1e-9; fixed to unitize()",
        "rung_actual": rung,
    }
    jp = OUT / "v16_0_recon.json"
    jp.write_text(json.dumps(payload, indent=2))
    (REPO / "logs/overnight/v16_0_recon.json").write_text(json.dumps(payload, indent=2))
    log(f"JSON → {jp}")
    log(f"Report (authoritative narrative) → {REPORT}")
    log("DONE — zero submits.")


if __name__ == "__main__":
    main()
