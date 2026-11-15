#!/usr/bin/env python3
"""GM V22 — Local M3 Pro MPS research for a NEW solo base that can clear ≥0.952+ OOF.

Goal: beat public GPT-2 Storyteller pooled 0.95003 (masaya) with a digit-enriched
target-first GPT-2, then scale to 5-fold if Fold0 canary clears the gate.

Phases:
  pilot   — 40k-fit / full-fold0-heldout, 3 epochs, vanilla vs DIGIT_TEXT (format A/B)
  fold0   — full fold0 train (8 ep) + 30-order OOF score; gate vs masaya fold0 0.94888
  full5   — remaining folds if fold0 ≥ CANARY_MIN

Uses matapanino Deotte recipe under /tmp/s6e11_v16/lm_research/matapanino with MPS.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
CODE = Path("/tmp/s6e11_v16/lm_research/matapanino")
RUN = Path("/tmp/s6e11_v16/lm_research/run_v22")
DATA = REPO / "data"
ORIG = REPO / "original" / "credit_card_fraud_2026.csv"
PY = Path("/tmp/s6e11_mps_venv/bin/python")
LOG = REPO / "logs/overnight"
MASAYA_F0 = 0.9488823016904258
MASAYA_POOLED = 0.9500272037932737
CANARY_MIN = float(os.environ.get("CANARY_MIN", "0.9505"))  # beat masaya f0 by ~+16e-4

T0 = time.time()


def log(*a):
    s = f"[{time.time()-T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.mkdir(parents=True, exist_ok=True)
    with open(LOG / "GM_V22_MPS_NEW_BASE_RESEARCH.log", "a") as f:
        f.write(s + "\n")


def run(cmd, env=None, check=True):
    e = os.environ.copy()
    if env:
        e.update(env)
    e.setdefault("TOKENIZERS_PARALLELISM", "false")
    e.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    log("EXEC", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(CODE), env=e)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode


def ensure_folds():
    RUN.mkdir(parents=True, exist_ok=True)
    fid = RUN / "fold_ids.npy"
    if fid.exists():
        return fid
    run([str(PY), "make_folds.py", "--train-csv", str(DATA / "train.csv"),
         "--target-col", "is_fraud", "--out", str(fid)])
    return fid


def train_fold(fold, out, *, epochs, batch, seed, digit, replica, max_rows=0, device="mps"):
    out = Path(out)
    if (out / "train_summary.json").exists():
        log(f"train fold{fold} already done → {out}")
        return
    env = {"DIGIT_TEXT": "1" if digit else "0"}
    cmd = [
        str(PY), "train_classifier.py",
        "--train-csv", str(DATA / "train.csv"),
        "--fold-ids", str(RUN / "fold_ids.npy"),
        "--fold", str(fold),
        "--id-col", "id",
        "--target-col", "is_fraud",
        "--replica", replica,
        "--out", str(out),
        "--epochs", str(epochs),
        "--batch-size", str(batch),
        "--seed", str(seed),
        "--precision", "bf16",
        "--device", device,
        "--save-every-steps", "5000",
        "--log-every-steps", "50",
    ]
    if max_rows:
        cmd += ["--max-rows", str(max_rows)]
    run(cmd, env=env)


def score_oof(fold, model_dir, out, *, orders, digit, seed=42, device="mps", max_rows=0):
    out = Path(out)
    env = {"DIGIT_TEXT": "1" if digit else "0"}
    cmd = [
        str(PY), "score_rows.py",
        "--model-dir", str(Path(model_dir) / "final"),
        "--csv", str(DATA / "train.csv"),
        "--split", "oof",
        "--fold-ids", str(RUN / "fold_ids.npy"),
        "--fold", str(fold),
        "--id-col", "id",
        "--target-col", "is_fraud",
        "--out", str(out),
        "--orders", str(orders),
        "--seed", str(seed),
        "--precision", "fp32",
        "--device", device,
        "--chunk-rows", "1000",
    ]
    if max_rows:
        cmd += ["--max-rows", str(max_rows)]
    run(cmd, env=env)


def early_auc(scores_dir, fold):
    """Assemble partial/full fold OOF from score chunks and compute AUC."""
    scores_dir = Path(scores_dir)
    chunks = sorted(scores_dir.glob("chunk_*.npz"))
    if not chunks:
        return None, 0
    rows, lp_t, lp_f = [], [], []
    for c in chunks:
        z = np.load(c)
        rows.append(z["rows"])
        # log_p_true / log_p_false shape (n, orders)
        lt = z["log_p_true"]
        lf = z["log_p_false"]
        # log-sum-exp over orders
        def lse(a):
            m = a.max(1, keepdims=True)
            return (m + np.log(np.exp(a - m).sum(1, keepdims=True))).ravel()
        lp_t.append(lse(lt))
        lp_f.append(lse(lf))
    rows = np.concatenate(rows)
    llr = np.concatenate(lp_t) - np.concatenate(lp_f)
    # prior from fit fold
    y_all = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"]).is_fraud.astype(int).values
    fold_ids = np.load(RUN / "fold_ids.npy")
    fit = fold_ids != fold
    prior = y_all[fit].mean()
    logit_prior = np.log(prior / (1 - prior))
    score = llr + logit_prior
    y = y_all[rows]
    return float(roc_auc_score(y, score)), len(rows)


def phase_pilot():
    """Compare vanilla vs digit on a speed-scaled Fold0 canary."""
    ensure_folds()
    replica = os.environ.get("V22_REPLICA", "distilgpt2")  # MPS-friendly; gpt2 ~2× slower
    max_rows = int(os.environ.get("V22_PILOT_ROWS", "20000"))
    epochs = int(os.environ.get("V22_PILOT_EPOCHS", "2"))
    batch = int(os.environ.get("V22_BATCH", "16"))
    log(f"=== PHASE PILOT: vanilla vs DIGIT_TEXT ({replica}, {max_rows} rows, {epochs} ep, batch {batch}) ===")
    results = {}
    for name, digit in [("vanilla", False), ("digit", True)]:
        out_tr = RUN / f"pilot_{name}_f0"
        out_sc = RUN / f"pilot_{name}_f0_oof"
        train_fold(0, out_tr, epochs=epochs, batch=batch, seed=2026, digit=digit,
                   replica=replica, max_rows=max_rows, device="mps")
        score_oof(0, out_tr, out_sc, orders=8, digit=digit, seed=2026, device="mps")
        auc, n = early_auc(out_sc, 0)
        results[name] = {"auc": auc, "n": n}
        log(f"PILOT {name}: AUC={auc} on n={n} (masaya f0={MASAYA_F0:.6f})")
    winner = max(results, key=lambda k: (results[k]["auc"] or 0))
    log(f"PILOT winner={winner} Δdigit-vanilla="
        f"{(results.get('digit',{}).get('auc') or 0)-(results.get('vanilla',{}).get('auc') or 0):+.5f}")
    (RUN / "pilot_results.json").write_text(json.dumps(results, indent=2))
    return winner, results


def phase_fold0(digit: bool, epochs=8, batch=16, seed=2026, orders=30):
    log(f"=== PHASE FOLD0: digit={digit} ep={epochs} batch={batch} seed={seed} orders={orders} ===")
    ensure_folds()
    tag = "digit" if digit else "vanilla"
    out_tr = RUN / f"full_{tag}_s{seed}_f0"
    out_sc = RUN / f"full_{tag}_s{seed}_f0_oof"
    replica = os.environ.get("V22_REPLICA", "distilgpt2")
    train_fold(0, out_tr, epochs=epochs, batch=batch, seed=seed, digit=digit,
               replica=replica, max_rows=0, device="mps")
    # progressive read while scoring
    score_oof(0, out_tr, out_sc, orders=orders, digit=digit, seed=seed, device="mps")
    auc, n = early_auc(out_sc, 0)
    log(f"FOLD0 {tag}: AUC={auc:.6f} n={n}  vs masaya_f0={MASAYA_F0:.6f}  Δ={(auc-MASAYA_F0)*1e5:+.1f}e-5")
    gate = auc >= CANARY_MIN
    summary = {
        "fold0_auc": auc, "n": n, "digit": digit, "seed": seed, "orders": orders,
        "masaya_f0": MASAYA_F0, "gate": gate, "canary_min": CANARY_MIN,
        "proj_pooled_if_delta_transfers": MASAYA_POOLED + (auc - MASAYA_F0),
    }
    (RUN / "fold0_canary.json").write_text(json.dumps(summary, indent=2))
    log(f"GATE {'PASS' if gate else 'FAIL'} canary_min={CANARY_MIN}")
    return summary


def phase_full5(digit: bool, seed=2026, epochs=8, batch=16, orders=30):
    log("=== PHASE FULL5 ===")
    ensure_folds()
    tag = "digit" if digit else "vanilla"
    for fold in range(5):
        out_tr = RUN / f"full_{tag}_s{seed}_f{fold}"
        replica = os.environ.get("V22_REPLICA", "distilgpt2")
        train_fold(fold, out_tr, epochs=epochs, batch=batch, seed=seed, digit=digit,
                   replica=replica, device="mps")
        score_oof(fold, out_tr, RUN / f"full_{tag}_s{seed}_f{fold}_oof",
                  orders=orders, digit=digit, seed=seed, device="mps")
        # also score test
        env = {"DIGIT_TEXT": "1" if digit else "0"}
        run([
            str(PY), "score_rows.py",
            "--model-dir", str(out_tr / "final"),
            "--csv", str(DATA / "test.csv"),
            "--split", "test",
            "--id-col", "id",
            "--target-col", "is_fraud",
            "--out", str(RUN / f"full_{tag}_s{seed}_f{fold}_test"),
            "--orders", str(min(orders, 10)),
            "--seed", str(seed),
            "--precision", "fp32",
            "--device", "mps",
            "--chunk-rows", "1000",
        ], env=env)
    # assemble
    scores = RUN / f"scores_{tag}_s{seed}"
    scores.mkdir(exist_ok=True)
    # make_predictions expects scores/f{k}_oof layout — symlink
    for fold in range(5):
        (scores / f"f{fold}_oof").mkdir(exist_ok=True)
        for c in (RUN / f"full_{tag}_s{seed}_f{fold}_oof").glob("chunk_*.npz"):
            dest = scores / f"f{fold}_oof" / c.name
            if not dest.exists():
                os.symlink(c, dest)
        (scores / f"f{fold}_test").mkdir(exist_ok=True)
        for c in (RUN / f"full_{tag}_s{seed}_f{fold}_test").glob("chunk_*.npz"):
            dest = scores / f"f{fold}_test" / c.name
            if not dest.exists():
                os.symlink(c, dest)
    out_pred = RUN / f"preds_{tag}_s{seed}"
    run([
        str(PY), "make_predictions.py",
        "--scores", str(scores),
        "--train-csv", str(DATA / "train.csv"),
        "--test-csv", str(DATA / "test.csv"),
        "--fold-ids", str(RUN / "fold_ids.npy"),
        "--target-col", "is_fraud",
        "--out", str(out_pred),
    ])
    man = json.loads((out_pred / "manifest.json").read_text())
    pooled = man.get("auc", {}).get("pooled") if isinstance(man.get("auc"), dict) else man.get("auc")
    log(f"FULL5 pooled AUC={pooled}  masaya={MASAYA_POOLED}  Δ={(float(pooled)-MASAYA_POOLED)*1e5:+.1f}e-5")
    # copy to repo preds
    dest_o = REPO / "preds" / f"v22_mps_{tag}_s{seed}_oof.npy"
    dest_t = REPO / "preds" / f"v22_mps_{tag}_s{seed}_test.npy"
    np.save(dest_o, np.load(out_pred / "oof.npy"))
    np.save(dest_t, np.load(out_pred / "test.npy"))
    log(f"saved {dest_o} {dest_t}")
    return pooled


def write_report(extra=""):
    body = [
        "# GM V22 — MPS New Base Research",
        "",
        f"elapsed {time.time()-T0:.0f}s · device MPS (M3 Pro) · clear target solo ≥0.952 / beat masaya 0.95003",
        "",
        extra,
        "",
        f"Run dir: `{RUN}`",
        f"Log: `{LOG / 'GM_V22_MPS_NEW_BASE_RESEARCH.log'}`",
    ]
    (LOG / "GM_V22_MPS_NEW_BASE_RESEARCH.md").write_text("\n".join(body) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "fold0", "full5", "all"], default="all")
    ap.add_argument("--digit", type=int, default=-1, help="1/0; -1=choose from pilot")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--orders", type=int, default=30)
    args = ap.parse_args()

    if not PY.exists():
        log("ERROR missing", PY)
        sys.exit(1)
    log("V22 MPS new-base research start", "phase=", args.phase)

    digit = None if args.digit < 0 else bool(args.digit)
    if args.phase in ("pilot", "all"):
        winner, pilot = phase_pilot()
        if digit is None:
            digit = winner == "digit"
        write_report(f"## Pilot\n```\n{json.dumps(pilot, indent=2)}\n```\nwinner={winner}")

    if args.phase in ("fold0", "all"):
        if digit is None:
            digit = True
        summary = phase_fold0(digit, epochs=args.epochs, batch=args.batch,
                              seed=args.seed, orders=args.orders)
        write_report(f"## Fold0 canary\n```\n{json.dumps(summary, indent=2)}\n```")
        if args.phase == "all" and not summary.get("gate"):
            log("STOP: fold0 failed canary — not launching full5")
            return

    if args.phase in ("full5",) or (args.phase == "all" and digit is not None):
        if digit is None:
            digit = True
        pooled = phase_full5(digit, seed=args.seed, epochs=args.epochs,
                             batch=args.batch, orders=args.orders)
        write_report(f"## Full5 pooled={pooled}")

    log("DONE")


if __name__ == "__main__":
    main()
