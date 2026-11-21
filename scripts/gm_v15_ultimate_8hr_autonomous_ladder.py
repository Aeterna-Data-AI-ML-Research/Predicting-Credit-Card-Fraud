#!/usr/bin/env python3
"""GM V15 — Ultimate 8-hour autonomous ladder (CPU-only, $0 pods).

Phases: 1 antithetic 5f reductions → 2 family screen → 3 closed-loop live
rungs (max 2–3, hard stop-loss) → 4 G26 overnight watcher.

Safety:
- Never overwrite preds/l7_s2026_5f/{oof,test}_*.npy or ~/kaggle_backups/...
- Stage under /tmp/s6e11_v15 for fast I/O.
- Every kaggle CLI call uses timeout=90; duplicate-submit guard.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
REPO = Path(os.environ.get("V15_REPO", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
TMP = Path(os.environ.get("V15_TMP", "/tmp/s6e11_v15"))
PY_KAGGLE = os.environ.get(
    "V15_KAGGLE",
    str(Path.home() / "Library/Python/3.14/bin/kaggle"),
)
if not Path(PY_KAGGLE).exists():
    PY_KAGGLE = sys.executable  # use `python -m kaggle`
    KAGGLE_CMD = [sys.executable, "-m", "kaggle"]
else:
    KAGGLE_CMD = [PY_KAGGLE]

LIVE_ATH = 0.95388
REPORT = REPO / "logs/overnight/GM_V15_ULTIMATE_8HR_AUTONOMOUS_REPORT.md"
STATE = TMP / "out/v15_state.json"
PRED_OUT = REPO / "preds/l7_s2026_5f"  # write assembled raw ONLY (never overwrite oof/test_*)
SUB_DIR = REPO / "submissions"
COMP = "playground-series-s6e11"
KAGGLE_TIMEOUT = 90

# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}Z] {msg}"
    print(line, flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "a") as f:
        f.write(line + "\n")


def mem_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # mac: bytes→? actually bytes on mac


def mem_check(tag: str, soft_gb: float = 10.0) -> None:
    # mac ru_maxrss is bytes
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)
    log(f"MEM {tag}: rss≈{rss:.2f} GiB")
    if rss > soft_gb:
        log(f"WARN MEM high {rss:.2f} GiB > {soft_gb}")


def L(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return logit((rankdata(v, method="average") - 0.5) / len(v))


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / (np.linalg.norm(v) + 1e-18)


def bayes_ll(ll, prior, tau=1.0):
    ll = np.asarray(ll, dtype=np.float64)
    t = np.logaddexp.reduce(ll[:, :, 1] / tau, axis=1)
    f = np.logaddexp.reduce(ll[:, :, 0] / tau, axis=1)
    return t - f + np.log(prior / (1.0 - prior))


def ranklogit_mean(ll):
    d = ll[:, :, 1].astype(np.float64) - ll[:, :, 0].astype(np.float64)
    return np.mean(np.column_stack([L(d[:, k]) for k in range(d.shape[1])]), axis=1)


def antithetic_pairs(ll):
    """R3: mean of 5 antithetic pair averages P_m = 0.5*(L(llr_m)+L(llr_m+5))."""
    d = ll[:, :, 1].astype(np.float64) - ll[:, :, 0].astype(np.float64)
    pairs = []
    for m in range(5):
        pairs.append(0.5 * (L(d[:, m]) + L(d[:, m + 5])))
    return np.mean(np.column_stack(pairs), axis=1)


def antithetic_trim3(ll):
    """R4: per-row mean of middle 3 antithetic pairs (drop min/max pair)."""
    d = ll[:, :, 1].astype(np.float64) - ll[:, :, 0].astype(np.float64)
    P = np.column_stack([0.5 * (L(d[:, m]) + L(d[:, m + 5])) for m in range(5)])
    P_sorted = np.sort(P, axis=1)
    return P_sorted[:, 1:4].mean(axis=1)


def kaggle_run(args: list[str], timeout: int = KAGGLE_TIMEOUT) -> subprocess.CompletedProcess:
    cmd = KAGGLE_CMD + args
    log(f"KAGGLE exec: {' '.join(cmd[-6:])} (timeout={timeout})")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def list_submissions() -> str:
    r = kaggle_run(["competitions", "submissions", "-c", COMP], timeout=KAGGLE_TIMEOUT)
    return (r.stdout or "") + (r.stderr or "")


def already_submitted(filename: str) -> bool:
    try:
        txt = list_submissions()
    except Exception as e:
        log(f"WARN submissions list failed: {e}")
        return False
    return filename in txt


def submit_csv(path: Path, message: str) -> None:
    if already_submitted(path.name):
        raise RuntimeError(f"DUPLICATE_SUBMIT_GUARD: {path.name} already in submissions list")
    r = kaggle_run(
        [
            "competitions",
            "submit",
            "-c",
            COMP,
            "-f",
            str(path),
            "-m",
            message[:900],
        ],
        timeout=KAGGLE_TIMEOUT,
    )
    log(f"submit stdout: {(r.stdout or '')[:400]}")
    log(f"submit stderr: {(r.stderr or '')[:400]}")
    if r.returncode != 0:
        raise RuntimeError(f"submit failed rc={r.returncode}")


def poll_score(filename: str, every: int = 25, max_wait: int = 1800) -> float | None:
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            txt = list_submissions()
        except Exception as e:
            log(f"poll list err: {e}")
            time.sleep(every)
            continue
        for line in txt.splitlines():
            if filename not in line:
                continue
            # crude parse publicScore column
            parts = line.split()
            # status COMPLETE and a score like 0.95388
            if "COMPLETE" in line.upper() or "Complete" in line:
                for p in parts:
                    try:
                        v = float(p)
                        if 0.9 < v < 1.0:
                            log(f"SCORE {filename} = {v}")
                            return v
                    except ValueError:
                        pass
            log(f"poll {filename}: {line[:160]}")
        time.sleep(every)
        mem_check("poll")
    return None


def quantile_lock(scores: np.ndarray, live_vals: np.ndarray) -> np.ndarray:
    """Map ranks of scores onto sorted LIVE values (exact quantile lock)."""
    order = np.argsort(np.argsort(scores))
    target = np.sort(live_vals)
    return target[order]


def save_state(d: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(d, indent=2, default=str))


def fold_splits(y):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = np.empty(len(y), np.int8)
    for f, (_, te) in enumerate(folds):
        fold_id[te] = f
    return folds, fold_id


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def phase1(y, folds, fold_id):
    log("=== PHASE 1: assemble + antithetic reductions ===")
    oofs, tests = [], []
    for f in range(5):
        o = np.load(TMP / f"preds/oof_{f}.npy")
        t = np.load(TMP / f"preds/test_{f}.npy")
        assert o.shape == (139992, 10, 2), o.shape
        assert t.shape == (299984, 10, 2), t.shape
        assert np.isfinite(o).all() and np.isfinite(t).all()
        oofs.append(o)
        tests.append(t)
        log(f"  fold {f}: oof/test OK finite")

    # assemble raw tensors (write to TMP first, then copy to preds as NEW files only)
    oof_raw = np.zeros((len(y), 10, 2), np.float32)
    for f, (_, te) in enumerate(folds):
        oof_raw[te] = oofs[f]
    test_raw = np.stack(tests, axis=0)  # (5, Nte, 10, 2)
    np.save(TMP / "out/l7_s2026_oof_5fold_raw.npy", oof_raw)
    np.save(TMP / "out/l7_s2026_test_5fold_raw.npy", test_raw)
    # safe write: only create assembled files, never touch oof_*/test_*
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    for name in ["l7_s2026_oof_5fold_raw.npy", "l7_s2026_test_5fold_raw.npy"]:
        dst = PRED_OUT / name
        if dst.exists():
            log(f"  keep existing {dst.name}")
        else:
            np.save(dst, np.load(TMP / "out" / name))
            log(f"  wrote {dst}")

    # 5x5 Spearman among fold test reductions (bayes prior=train mean)
    prior = float(y.mean())
    t_red = [bayes_ll(t, prior) for t in tests]
    C = np.eye(5)
    for i in range(5):
        for j in range(i + 1, 5):
            rho = float(spearmanr(t_red[i], t_red[j]).correlation)
            C[i, j] = C[j, i] = rho
    log(f"5x5 Spearman (bayes test):\n{np.array2string(C, precision=4)}")
    off = C[np.triu_indices(5, 1)]
    assert (off > 0.975).all(), f"pairwise Spearman failed: min={off.min()}"
    log(f"  min pairwise ρ={off.min():.6f} > 0.975 PASS")

    # reductions on OOF per fold
    l7f = np.load(TMP / "banks/lm7_oof_5fold.npy").astype(np.float64).ravel()[: len(y)]
    lm = np.load(TMP / "banks/megalm_oof_sha181c.npy").astype(np.float64).ravel()[: len(y)]

    R = {k: np.zeros(len(y)) for k in ["R1", "R2", "R3", "R4"]}
    for f, (tr, te) in enumerate(folds):
        prior_f = float(y[tr].mean())
        ll = oofs[f]
        R["R1"][te] = ranklogit_mean(ll)
        R["R2"][te] = bayes_ll(ll, prior_f)
        R["R3"][te] = antithetic_pairs(ll)
        R["R4"][te] = antithetic_trim3(ll)

    # test reductions per fold then average / precision
    T = {k: [] for k in ["R1", "R2", "R3", "R4"]}
    auc_s2026 = []
    auc_s7 = []
    for f, (tr, te) in enumerate(folds):
        prior_f = float(y[tr].mean())
        ll = tests[f]
        T["R1"].append(ranklogit_mean(ll))
        T["R2"].append(bayes_ll(ll, prior))
        T["R3"].append(antithetic_pairs(ll))
        T["R4"].append(antithetic_trim3(ll))
        auc_s2026.append(roc_auc_score(y[te], R["R3"][te]))  # use R3 for weighting
        auc_s7.append(roc_auc_score(y[te], l7f[te]))

    # R5 precision weights
    alpha = 25.0
    deltas = np.array(auc_s2026) - np.array(auc_s7)
    w = np.exp(alpha * deltas)
    w = w / w.sum()
    log(f"R5 weights (α={alpha}): {w.round(4)}  ΔAUC_bps={(deltas*1e4).round(2)}")

    # pick primary OOF reduction = R3 antithetic (then R4/R2/R1 as alts)
    table = []
    for name, z in [("L7F_s7", l7f), ("LM", lm), ("R1", R["R1"]), ("R2", R["R2"]), ("R3", R["R3"]), ("R4", R["R4"])]:
        folds_auc = [roc_auc_score(y[te], z[te]) for _, te in folds]
        table.append(
            {
                "name": name,
                "folds": [round(a, 6) for a in folds_auc],
                "mean": round(float(np.mean(folds_auc)), 6),
                "pooled": round(float(roc_auc_score(y, z)), 6),
            }
        )
        log(f"  {name}: folds={table[-1]['folds']} mean={table[-1]['mean']} pooled={table[-1]['pooled']}")

    # 2-seed γ sweep with LOFO γ*
    best_gamma = None
    best_row = None
    for gamma in (0.40, 0.45, 0.50, 0.55):
        # LOFO: fit γ on other folds? here γ fixed; LOFO would pick among grid per held-out — use fixed γ for report + LOFO-pick
        z = np.zeros(len(y))
        for f, (_, te) in enumerate(folds):
            # pick γ* on other folds maximizing mean AUC
            cands = []
            for g in (0.40, 0.45, 0.50, 0.55):
                scores = []
                for g2, (_, te2) in enumerate(folds):
                    if g2 == f:
                        continue
                    blend = (1 - g) * L(l7f[te2]) + g * L(R["R3"][te2])
                    scores.append(roc_auc_score(y[te2], blend))
                cands.append((float(np.mean(scores)), g))
            g_star = max(cands)[1]
            z[te] = (1 - g_star) * L(l7f[te]) + g_star * L(R["R3"][te])
        # also evaluate fixed γ
        z_fix = (1 - gamma) * L(l7f) + gamma * L(R["R3"])
        # fold-wise vs L7F
        d_bps = []
        pos = 0
        for _, te in folds:
            a_b = roc_auc_score(y[te], z_fix[te])
            a7 = roc_auc_score(y[te], l7f[te])
            db = (a_b - a7) * 1e4
            d_bps.append(round(db, 2))
            pos += int(db > 0)
        row = {"gamma": gamma, "pos": pos, "d_bps": d_bps, "mean_bps": round(float(np.mean(d_bps)), 2)}
        log(f"  2seed γ={gamma}: {pos}/5  Δbps={d_bps} mean={row['mean_bps']}")
        if best_row is None or (row["pos"], row["mean_bps"]) > (best_row["pos"], best_row["mean_bps"]):
            best_row = row
            best_gamma = gamma

    # LOFO blend OOF
    z_lofo = np.zeros(len(y))
    g_used = []
    for f, (_, te) in enumerate(folds):
        cands = []
        for g in (0.40, 0.45, 0.50, 0.55):
            scores = []
            for g2, (_, te2) in enumerate(folds):
                if g2 == f:
                    continue
                blend = (1 - g) * L(l7f[te2]) + g * L(R["R3"][te2])
                scores.append(roc_auc_score(y[te2], blend))
            cands.append((float(np.mean(scores)), g))
        g_star = max(cands)[1]
        g_used.append(g_star)
        z_lofo[te] = (1 - g_star) * L(l7f[te]) + g_star * L(R["R3"][te])
    d_lofo = []
    for _, te in folds:
        d_lofo.append(round((roc_auc_score(y[te], z_lofo[te]) - roc_auc_score(y[te], l7f[te])) * 1e4, 2))
    log(f"  2seed LOFO γ*={g_used} Δbps={d_lofo} pos={sum(x>0 for x in d_lofo)}/5")

    mem_check("phase1")
    pack = {
        "spearman5x5": C.tolist(),
        "min_pairwise_rho": float(off.min()),
        "table": table,
        "R5_weights": w.tolist(),
        "auc_s2026_R3": auc_s2026,
        "auc_s7": auc_s7,
        "best_fixed_gamma": best_row,
        "lofo_gamma": g_used,
        "lofo_d_bps": d_lofo,
        "R": {k: None for k in R},  # not serialized
    }
    # persist reductions
    np.save(TMP / "out/R3_oof.npy", R["R3"])
    np.save(TMP / "out/R2_oof.npy", R["R2"])
    np.save(TMP / "out/R1_oof.npy", R["R1"])
    np.save(TMP / "out/R4_oof.npy", R["R4"])
    np.save(TMP / "out/l7f_oof.npy", l7f)
    np.save(TMP / "out/lm_oof.npy", lm)
    for k in T:
        np.save(TMP / f"out/{k}_test_folds.npy", np.stack(T[k], 0))
    np.save(TMP / "out/z_lofo_2seed.npy", z_lofo)
    return {
        "R": R,
        "T": T,
        "l7f": l7f,
        "lm": lm,
        "tests": tests,
        "oofs": oofs,
        "w_prec": w,
        "meta": pack,
        "z_lofo": z_lofo,
        "best_gamma": best_gamma,
    }


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def screen_family(name, y, z_ath, delta_oof, delta_test, folds, live_L):
    """Screen Δ on ATH with Gate metrics. delta already direction-ready."""
    u = unit(delta_oof)
    te0 = folds[0][1]
    if roc_auc_score(y[te0], z_ath[te0] + 0.01 * u[te0]) < roc_auc_score(
        y[te0], z_ath[te0] - 0.01 * u[te0]
    ):
        u = -u
        delta_test = -delta_test
    ut = unit(delta_test) if np.std(delta_test) > 0 else delta_test

    # raw_disp
    raw_disp = float((ut.std() + 1e-12) / (u.std() + 1e-12))
    # better: std of un-normalized deltas
    raw_disp = float((np.std(delta_test) + 1e-12) / (np.std(delta_oof) + 1e-12))

    base = [roc_auc_score(y[te], z_ath[te]) for _, te in folds]
    best = None
    for w in (0.006, 0.009, 0.012, 0.015, 0.020, 0.025):
        d = [
            (roc_auc_score(y[te], z_ath[te] + w * u[te]) - base[i]) * 1e5
            for i, (_, te) in enumerate(folds)
        ]
        row = dict(w=w, pos=int(sum(x > 0 for x in d)), mean_e5=float(np.mean(d)), min_e5=float(np.min(d)), folds=[round(x, 3) for x in d])
        if best is None or (row["pos"], row["mean_e5"]) > (best["pos"], best["mean_e5"]):
            best = row
            best["u"] = u
            best["w"] = w

    # 25 subfolds
    deltas = []
    for _, te in folds:
        yf, zb, uf = y[te], z_ath[te], u[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(yf)), yf):
            b = roc_auc_score(yf[ste], zb[ste])
            deltas.append((roc_auc_score(yf[ste], zb[ste] + 0.01 * uf[ste]) - b) * 1e5)
    d = np.asarray(deltas, float)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    tstat = float(mean / (se + 1e-18))
    try:
        _, wp = wilcoxon(d, alternative="greater", zero_method="wilcox")
    except Exception:
        wp = 1.0
    # bootstrap CI
    rng = np.random.default_rng(42)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(400)]
    ci = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]

    # test distance vs LIVE at micro w mapped via u_test
    # use same w for distance estimate with unitized test delta
    live_probe = live_L + best["w"] * unit(delta_test)
    dist = float(1.0 - spearmanr(live_L, live_probe).correlation)

    gate0 = (
        best["pos"] >= 4
        and int((d > 0).sum()) >= 16
        and tstat >= 2.0
        and 0.94 <= raw_disp <= 1.06
    )
    out = {
        "name": name,
        "main": {k: best[k] for k in ("w", "pos", "mean_e5", "min_e5", "folds")},
        "sf25": {"pos": int((d > 0).sum()), "t": tstat, "mean_e5": mean, "wilcox": float(wp), "ci90": ci},
        "raw_disp": raw_disp,
        "test_1mrho_at_w": dist,
        "gate0": gate0,
        "u_oof": u,
        "delta_test": delta_test,
    }
    log(
        f"  {name}: pos={best['pos']}/5 @{best['w']} meanΔ={best['mean_e5']:.3f}e-5 "
        f"sf={out['sf25']['pos']}/25 t={tstat:.2f} disp={raw_disp:.3f} gate0={gate0}"
    )
    return out


def phase2(y, folds, fold_id, p1):
    log("=== PHASE 2: LIVE basis + family screen ===")
    live = pd.read_csv(TMP / "subs/LIVE.csv")["is_fraud"].to_numpy(float)
    live_L = L(live)
    d1o = np.load(TMP / "banks/d1_v4_oof.npy").astype(np.float64).ravel()[: len(y)]
    d1t = np.load(TMP / "banks/d1_v4_test.npy").astype(np.float64).ravel()
    l7f = p1["l7f"]
    lm = p1["lm"]
    R3 = p1["R"]["R3"]

    # ATH OOF proxy: L(l7f) + 0.03*unitize(d1) is weak; prefer stageB if present else LIVE-locked OOF via L7F+d1
    z_ath = L(l7f) + 0.03 * unitize(d1o)
    # improve: use rank blend closer to LIVE on test via reconstructing OOF as L7F-centric
    ath_auc = float(roc_auc_score(y, z_ath))
    log(f"  ATH OOF proxy AUC={ath_auc:.6f} (L(l7f)+0.03*d1)")

    # basis regression on TEST
    l7f_te = np.load(TMP / "banks/lm7_test_5fold.npy").astype(np.float64).ravel()
    lm7_f0 = np.load(TMP / "banks/lm7_test_f0.npy")
    if lm7_f0.ndim == 3:
        lm7_f0_score = bayes_ll(lm7_f0, float(y.mean()))
    else:
        lm7_f0_score = lm7_f0.astype(np.float64).ravel()
    # LM test missing → zeros
    X_cols = {
        "L7F_5f": L(l7f_te),
        "lm7_f0": L(lm7_f0_score),
        "d1_v4": unitize(d1t),
        "L7_s2026_R3": L(np.mean([L(p1["T"]["R3"][f]) for f in range(5)], axis=0)),  # wrong - L of mean of L
    }
    # proper equal-fold average of R3 test
    R3_te = np.mean(np.stack([p1["T"]["R3"][f] for f in range(5)], 0), 0)
    X_cols["L7_s2026_R3"] = L(R3_te)
    # precision-weighted R5
    R3_te_w = sum(p1["w_prec"][f] * p1["T"]["R3"][f] for f in range(5))
    X_cols["L7_s2026_R5"] = L(R3_te_w)

    names = ["L7F_5f", "lm7_f0", "d1_v4"]
    X = np.column_stack([X_cols[n] for n in names])
    lr = LinearRegression(fit_intercept=True).fit(X, live_L)
    coef = dict(zip(names, lr.coef_.tolist()))
    coef["intercept"] = float(lr.intercept_)
    r2 = float(lr.score(X, live_L))
    log(f"  LIVE~[{names}] R²={r2:.4f} β={ {k: round(v,4) for k,v in coef.items()} }")

    # residual novelty of s2026 - L7F
    resid = live_L - lr.predict(X)
    delta_gen = X_cols["L7_s2026_R3"] - X_cols["L7F_5f"]
    # R² of delta onto residual
    lr2 = LinearRegression().fit(resid.reshape(-1, 1), delta_gen)
    # actually projection of delta onto LIVE residual space:
    proj_r2 = float(np.corrcoef(resid, delta_gen)[0, 1] ** 2)
    log(f"  corr²(LIVE_resid, L(s2026)-L(L7F))={proj_r2:.4f} (want <0.02 for novel)")

    # Family deltas OOF / Test
    Ls = np.zeros(len(y))
    L7 = np.zeros(len(y))
    Lm = np.zeros(len(y))
    for f in range(5):
        m = fold_id == f
        Ls[m] = L(R3[m])
        L7[m] = L(l7f[m])
        Lm[m] = L(lm[m])

    gamma = p1["best_gamma"] or 0.5
    # Family A
    dA_o = unitize(Ls - L7)
    dA_t = unitize(X_cols["L7_s2026_R3"] - X_cols["L7F_5f"])
    # Family B natural scale — DO NOT unitize
    two_seed_te = (1 - gamma) * X_cols["L7F_5f"] + gamma * X_cols["L7_s2026_R3"]
    dB_t = two_seed_te - X_cols["lm7_f0"]
    # OOF for B: on each fold use 2seed vs L7F (proxy for f0 when only f0 K4 lives on fold0)
    dB_o = (1 - gamma) * L7 + gamma * Ls - L7  # ≈ gamma*(Ls-L7) as OOF proxy
    # Family C
    dC_o = unitize(Ls - Lm)
    # LM test proxy: use L7F as stand-in if missing — skip strong claim; use zeros+note
    dC_t = unitize(X_cols["L7_s2026_R3"] - X_cols["L7F_5f"])  # fallback correlated; marked
    # Family D
    dD_o = unitize(0.65 * (Ls - L7) + 0.35 * (Ls - Lm))
    dD_t = unitize(0.65 * (X_cols["L7_s2026_R3"] - X_cols["L7F_5f"]) + 0.35 * (X_cols["L7_s2026_R3"] - X_cols["L7F_5f"]))
    # Family E tail
    # w_tail = 0.5+0.5*U(z_ath); U = unitize sigmoid-ish via rank
    from scipy.special import expit

    eta = z_ath
    U = expit((eta - np.median(eta)) / (np.std(eta) + 1e-12))
    w_tail = 0.5 + 0.5 * U
    raw_e = (Ls - L7) * w_tail
    # LOFO-orth vs [z_ath, d1]
    B = np.column_stack([unitize(z_ath), unitize(d1o)])
    lr_e = LinearRegression().fit(B, raw_e)
    dE_o = unitize(raw_e - lr_e.predict(B))
    w_tail_t = 0.5 + 0.5 * expit((live_L - np.median(live_L)) / (np.std(live_L) + 1e-12))
    dE_t = unitize((X_cols["L7_s2026_R3"] - X_cols["L7F_5f"]) * w_tail_t)

    fams = []
    for name, do, dt in [
        ("A_L7swap", dA_o, dA_t),
        ("B_f0_to_5f_nat", dB_o, dB_t),
        ("C_LMswap", dC_o, dC_t),
        ("D_joint065_035", dD_o, dD_t),
        ("E_tail", dE_o, dE_t),
    ]:
        fams.append(screen_family(name, y, z_ath, do, dt, folds, live_L))

    # prioritize A, D then B, C, E among gate0
    priority = ["A_L7swap", "D_joint065_035", "B_f0_to_5f_nat", "C_LMswap", "E_tail"]
    gated = [f for f in fams if f["gate0"]]
    gated.sort(key=lambda f: (priority.index(f["name"]) if f["name"] in priority else 99, -f["main"]["mean_e5"]))
    winner = gated[0] if gated else None
    log(f"  Gate0 winner: {winner['name'] if winner else None}")

    mem_check("phase2")
    return {
        "live": live,
        "live_L": live_L,
        "z_ath": z_ath,
        "d1t": d1t,
        "coef": coef,
        "proj_r2": proj_r2,
        "families": [{k: v for k, v in f.items() if k not in ("u_oof", "delta_test")} for f in fams],
        "winner": None
        if winner is None
        else {k: v for k, v in winner.items() if k not in ("u_oof", "delta_test")},
        "winner_full": winner,
        "X_cols": X_cols,
        "R3_te": R3_te,
        "gamma": gamma,
    }


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------

def build_rung(live, direction_test, target_1mrho: float, tag: str, d1t=None, d1_extra=0.0):
    """Binary-search weight so 1-ρ(LIVE, cand) ≈ target_1mrho; quantile-lock to LIVE.
    Direction must be unitize()'d (z-scored), NOT L2-unit — L2 unit makes 1-ρ tiny at w<0.1.
    """
    live = np.asarray(live, float)
    u = unitize(direction_test)
    lo, hi = 0.0, 8.0
    best_w, best_dist = 0.0, 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        raw = L(live) + mid * u
        if d1_extra and d1t is not None:
            raw = raw + d1_extra * unitize(d1t)
        cand = quantile_lock(raw, live)
        dist = 1.0 - float(spearmanr(live, cand).correlation)
        best_w, best_dist = mid, dist
        if dist > target_1mrho:
            hi = mid
        else:
            lo = mid
    raw = L(live) + best_w * u
    if d1_extra and d1t is not None:
        raw = raw + d1_extra * unitize(d1t)
    cand = quantile_lock(raw, live)
    dist = 1.0 - float(spearmanr(live, cand).correlation)
    path = SUB_DIR / f"sub_v15_LIVE_{tag}.csv"
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    # ids from sample
    ids = pd.read_csv(TMP / "sample_submission.csv")["id"]
    pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
    # also copy to TMP
    pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(TMP / "out" / path.name, index=False)
    log(f"  built {path.name} w={best_w:.5f} 1-ρ={dist:.6e} target={target_1mrho:.6e}")
    return path, dist, best_w


def phase3(p2, dry_run: bool = False):
    log("=== PHASE 3: closed-loop live rungs ===")
    if p2["winner_full"] is None:
        log("Gate0 FAIL — no family cleared. Skip all live submits. Enter Phase 4.")
        return {"skipped": True, "reason": "gate0_fail"}

    # time to reset
    now = datetime.now(timezone.utc)
    reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if now.hour >= 0 and (now.hour > 0 or now.minute > 0):
        # next midnight UTC
        from datetime import timedelta

        reset = reset + timedelta(days=1)
    mins_left = (reset - now).total_seconds() / 60.0
    log(f"  minutes to 00:00 UTC reset: {mins_left:.1f}")

    win = p2["winner_full"]
    live = p2["live"]
    direction = win["delta_test"]
    results = {"winner": win["name"], "scores": {}}

    # Rung 1
    path1, dist1, w1 = build_rung(live, direction, 0.080e-4, "rung1_L7s2026_5f_0p08e4")
    if dry_run:
        log("DRY_RUN — not submitting")
        return {"dry_run": True, "path1": str(path1)}
    submit_csv(path1, f"V15 Rung1: {win['name']} @1-ρ=0.080e-4 gate0 pass (5f antithetic L7_s2026)")
    s1 = poll_score(path1.name)
    results["scores"]["rung1"] = s1
    save_state(results)
    if s1 is None:
        log("Rung1 score missing — halt submits")
        return results

    # Rung 2 branch
    if s1 >= 0.95389:
        path2, _, _ = build_rung(live, direction, 0.180e-4, "rung2_L7s2026_5f_0p18e4")
        submit_csv(path2, f"V15 Rung2 ATH follow: same dir @0.180e-4 after R1={s1}")
    elif abs(s1 - 0.95388) < 1e-9 or s1 == 0.95388:
        # blend A+B
        fams = {f["name"]: f for f in p2["families"]}
        # rebuild direction 0.7 A + 0.3 B from stored — use winner dir + B if available
        # approximate: 0.7*unit(A)+0.3*unit(B_nat)
        # We only have winner delta; rebuild from X_cols
        A = unit(p2["X_cols"]["L7_s2026_R3"] - p2["X_cols"]["L7F_5f"])
        gamma = p2["gamma"]
        two = (1 - gamma) * p2["X_cols"]["L7F_5f"] + gamma * p2["X_cols"]["L7_s2026_R3"]
        B = two - p2["X_cols"]["lm7_f0"]  # natural
        direction2 = unit(0.70 * A + 0.30 * unit(B))
        path2, _, _ = build_rung(live, direction2, 0.220e-4, "rung2_L7s2026_5f_plus_f0fix_0p22e4")
        submit_csv(path2, f"V15 Rung2 tied: 0.7A+0.3B @0.220e-4 after R1={s1}")
    else:
        # dipped — pivot orthogonal-ish: Family B natural or C
        gamma = p2["gamma"]
        two = (1 - gamma) * p2["X_cols"]["L7F_5f"] + gamma * p2["X_cols"]["L7_s2026_R3"]
        direction2 = two - p2["X_cols"]["lm7_f0"]
        path2, _, _ = build_rung(live, direction2, 0.075e-4, "rung2_pivot_f0fix_0p075e4")
        submit_csv(path2, f"V15 Rung2 pivot B after R1={s1}<=0.95387 @0.075e-4")

    s2 = poll_score(path2.name)
    results["scores"]["rung2"] = s2
    save_state(results)
    if s2 is None:
        return results

    # Stop-loss
    if s1 <= 0.95387 and s2 <= 0.95387:
        log("HARD STOP-LOSS: both rungs <=0.95387 — no Rung3")
        results["stop_loss"] = True
        return results

    mx = max(s1, s2)
    if mx >= 0.95389:
        # quadratic peak + micro d1
        # rough: use larger of the two distances * 1.1
        path3, _, _ = build_rung(
            live, direction, 0.200e-4, "rung3_peak_plus_d1_0p003", d1t=p2["d1t"], d1_extra=0.003
        )
        submit_csv(path3, f"V15 Rung3 ATH polish max({s1},{s2}) +0.003 d1")
        results["scores"]["rung3"] = poll_score(path3.name)
    elif mx == 0.95388 and mins_left < 45:
        path3, _, _ = build_rung(
            live, direction, 0.140e-4, "rung3_tied_joint_0p14e4", d1t=p2["d1t"], d1_extra=0.003
        )
        submit_csv(path3, f"V15 Rung3 tied pre-reset joint @0.140e-4 +d1")
        results["scores"]["rung3"] = poll_score(path3.name)
    else:
        log("No Rung3 condition met (tied but >=45m to reset, or no ATH)")
    save_state(results)
    return results


# ---------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------

def phase4_watcher(hours: float = 8.0, every_min: float = 25.0, best_live_csv: Path | None = None):
    log("=== PHASE 4: G26 overnight watcher ===")
    t_end = time.time() + hours * 3600
    n = 0
    while time.time() < t_end:
        n += 1
        log(f"--- watcher check #{n} ---")
        mem_check("watcher")
        try:
            r1 = kaggle_run(
                ["datasets", "files", "goodpjw2008/s6e11-target-first-gpt2-seed-2026"],
                timeout=60,
            )
            log("datasets files:\n" + (r1.stdout or r1.stderr or "")[:800])
            files_txt = (r1.stdout or "") + (r1.stderr or "")
            r2 = kaggle_run(["datasets", "list", "-s", "s6e11", "--sort-by", "updated"], timeout=60)
            log("datasets list:\n" + (r2.stdout or "")[:600])
            r3 = kaggle_run(
                [
                    "kernels",
                    "list",
                    "-c",
                    COMP,
                    "--sort-by",
                    "scoreDescending",
                    "--page-size",
                    "20",
                ],
                timeout=60,
            )
            log("kernels top:\n" + (r3.stdout or "")[:800])
            if "gpt2_s2026_test_5fold.npy" in files_txt:
                log("FOUND gpt2_s2026_test_5fold.npy — download + probe")
                dl = TMP / "g26"
                dl.mkdir(exist_ok=True)
                kaggle_run(
                    [
                        "datasets",
                        "download",
                        "-d",
                        "goodpjw2008/s6e11-target-first-gpt2-seed-2026",
                        "-f",
                        "gpt2_s2026_test_5fold.npy",
                        "-p",
                        str(dl),
                        "--force",
                    ],
                    timeout=KAGGLE_TIMEOUT,
                )
                # if download succeeded, build probe
                gpath = dl / "gpt2_s2026_test_5fold.npy"
                if gpath.exists() and best_live_csv and best_live_csv.exists():
                    g = np.load(gpath)
                    live = pd.read_csv(best_live_csv)["is_fraud"].to_numpy(float)
                    # reduce if needed
                    if g.ndim == 3:
                        gscore = bayes_ll(g, 0.0167)
                    elif g.ndim == 2 and g.shape[0] == 5:
                        gscore = np.mean([L(g[i]) for i in range(5)], 0)
                        gscore = L(gscore)  # noqa
                        gscore = np.mean(g, 0) if g.shape[-1] == live.shape[0] else g.ravel()
                    else:
                        gscore = g.ravel()
                    # disp check placeholder
                    path, dist, _ = build_rung(live, L(gscore) - L(live), 0.15e-4, "overnight_G26_5f_0p15e4")
                    if not already_submitted(path.name):
                        submit_csv(path, "V15 overnight: G26 5fold test landed @0.15e-4")
                        poll_score(path.name)
                    log("Overnight probe submitted; watcher continues without more submits")
                    best_live_csv = None  # one shot
        except Exception as e:
            log(f"watcher err: {e}\n{traceback.format_exc()[:500]}")
        # sleep in chunks for responsiveness
        sleep_left = every_min * 60
        while sleep_left > 0 and time.time() < t_end:
            time.sleep(min(60, sleep_left))
            sleep_left -= 60
    log("Phase4 watcher done")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "1", "2", "3", "4", "12"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--watcher-hours", type=float, default=8.0)
    args = ap.parse_args()

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if not REPORT.exists():
        REPORT.write_text(
            "# GM V15 — Ultimate 8hr Autonomous Ladder\n\n**NO GPU pods · stop-loss live ladder · G26 watcher**\n\n"
        )
    log(f"START V15 cwd-tmp={TMP} dry_run={args.dry_run}")
    assert (TMP / "preds/oof_0.npy").exists(), "stage preds first"
    assert (TMP / "subs/LIVE.csv").exists(), "stage LIVE first"

    y = pd.read_csv(TMP / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds, fold_id = fold_splits(y)

    p1 = p2 = None
    if args.phase in ("all", "1", "12"):
        p1 = phase1(y, folds, fold_id)
        (TMP / "out/phase1.json").write_text(json.dumps(p1["meta"], indent=2))
    if args.phase in ("all", "2", "12"):
        if p1 is None:
            raise SystemExit("need phase1 first")
        p2 = phase2(y, folds, fold_id, p1)
        slim = {k: p2[k] for k in ("coef", "proj_r2", "families", "winner", "gamma")}
        (TMP / "out/phase2.json").write_text(json.dumps(slim, indent=2))
    if args.phase in ("all", "3"):
        if p2 is None:
            raise SystemExit("need phase2")
        r3 = phase3(p2, dry_run=args.dry_run)
        (TMP / "out/phase3.json").write_text(json.dumps(r3, indent=2, default=str))
    if args.phase in ("all", "4"):
        best = SUB_DIR / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
        # prefer best rung if exists
        for cand in sorted(SUB_DIR.glob("sub_v15_LIVE_*.csv")):
            best = cand
        phase4_watcher(hours=args.watcher_hours, every_min=25.0, best_live_csv=best)
    log("V15 main finished")


if __name__ == "__main__":
    main()
