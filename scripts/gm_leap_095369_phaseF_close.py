#!/usr/bin/env python3
"""Phase F — close last ~3e-5 gap to 0.95369 around high-ray/low-v10 band. NO_SUBMIT."""
from __future__ import annotations

import gc
import json
import os
import subprocess
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("NO_SUBMIT", "1")
warnings.filterwarnings("ignore")

VOL = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
WORK = Path(os.environ.get("GM_LEAP_ROOT", "/tmp/gm_deep_tech"))
PRED = VOL / "preds"
ART = WORK / "preds/gm_leap_095369"
LOG = WORK / "logs/overnight"
assert os.environ.get("NO_SUBMIT", "1") != "0"
EPS = 1e-6
T0 = time.time()
TARGET = 0.95369

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def rss():
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return float(out.strip()) / 1024.0


def log(msg):
    print(f"[{time.time()-T0:6.1f}s|RSS={rss():.0f}MB] {msg}", flush=True)


def load_f32(p):
    return np.asarray(np.load(p), dtype=np.float32).ravel()


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS)).astype(np.float32)


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return ((v - float(v.mean())) / (float(v.std()) + 1e-12)).astype(np.float32)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v / (float(np.linalg.norm(v)) + 1e-18)).astype(np.float32)


def spearman_dist(a, b):
    return float(1.0 - spearmanr(a, b).correlation)


def pred_lb_from_t(t):
    return 0.95346910 + 6.99725804e-05 * t


def t_for_target(lb=TARGET):
    return (lb - 0.95346910) / 6.99725804e-05


def lofo_resid(y_vec, bases, folds):
    y_vec = np.asarray(y_vec, dtype=np.float64).ravel()
    B = np.column_stack([unitize(b).astype(np.float64) for b in bases])
    resid = np.zeros_like(y_vec)
    for a, b in folds:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(B[a], y_vec[a])
        resid[b] = y_vec[b] - lr.predict(B[b])
    return unitize(resid)


def fold_aucs(y, z, folds):
    return [float(roc_auc_score(y[b], np.asarray(z, dtype=np.float64)[b])) for _, b in folds]


def eval_delta(y, z_base, z_new, folds, m0=None):
    bf, nf = fold_aucs(y, z_base, folds), fold_aucs(y, z_new, folds)
    d5 = [(a - b) * 1e5 for a, b in zip(nf, bf)]
    out = {"d5_mean": float(np.mean(d5)), "pos": int(sum(x > 0 for x in d5)), "d5": [float(x) for x in d5]}
    if m0 is not None:
        out["d0"] = (
            float(roc_auc_score(y[m0], z_new[m0])) - float(roc_auc_score(y[m0], z_base[m0]))
        ) * 1e5
    return out


def orient_test(u, z_areeb, z_s7_test):
    u = unit(u).astype(np.float32)
    if spearman_dist(z_areeb, z_s7_test.astype(np.float64) + 5 * u.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7_test
    ):
        u = (-u.astype(np.float64)).astype(np.float32)
    return u


def to_z(raw):
    raw = np.asarray(raw, dtype=np.float32).ravel()
    if float(raw.min()) >= -1e-6 and float(raw.max()) <= 1.0 + 1e-6:
        return unitize(zrk(raw))
    return unitize(raw)


def calib_curve(z0, u, z_areeb, ws):
    u = unit(u).astype(np.float64)
    z0 = np.asarray(z0, dtype=np.float64)
    rows = []
    for w in ws:
        z = unitize(z0 + float(w) * u)
        d0 = spearman_dist(z0, z)
        da = spearman_dist(z_areeb, z)
        t = 100.0 * np.sqrt(max(da, 0))
        rows.append({"w": float(w), "d_from_s7": d0, "d_areeb": da, "t": t, "pred": pred_lb_from_t(t)})
    return rows


def row_at_step(curve, step):
    xs = np.array([r["d_from_s7"] for r in curve])
    ys = np.array([r["w"] for r in curve])
    ws = np.array([r["w"] for r in curve])
    ts = np.array([r["t"] for r in curve])
    ps = np.array([r["pred"] for r in curve])
    if step <= xs[0]:
        w = float(ys[0])
    elif step >= xs[-1]:
        w = float(ys[-1])
    else:
        w = float(np.interp(step, xs, ys))
    return w, float(np.interp(w, ws, ts)), float(np.interp(w, ws, ps))


def main():
    log("======== PHASE F CLOSE GAP → 0.95369 (NO_SUBMIT) ========")
    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = pd.read_csv(
        VOL / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv", usecols=["fold"]
    )["fold"].to_numpy(np.int8)
    m0 = fold_id == 0

    z_ref = load_f32(PRED / "gm_v5_overnight/z_ref_old_oof.npy")
    z_ref = unitize(z_ref) if abs(float(z_ref.std()) - 1.0) > 0.2 else z_ref
    d1 = unitize(load_f32(PRED / "gm_v4_polish/d1_v4_oof.npy"))
    w12 = unitize(load_f32(PRED / "gm_v2_ladder/z_lm12_raw_super_oof.npy"))
    mega = zrk(load_f32(VOL / "external/banks/megayak_s6e11-target-first-lm/lm_oof.npy"))
    gpt = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))
    bases = [z_ref, d1, w12, mega, gpt]

    u_sb = unit(d1)
    m1o = unitize(
        0.65 * u_sb.astype(np.float64)
        - 0.35 * (-unit(gpt)).astype(np.float64)
        - 0.20 * gpt.astype(np.float64)
    )
    s6 = unitize(0.70 * m1o.astype(np.float64) + 0.30 * u_sb.astype(np.float64))
    w_s7, dw0 = 0.055, (0.75e-4 / 1e-4) * 0.08
    z_s7 = z_ref.astype(np.float64) + w_s7 * s6.astype(np.float64)
    z_ray = z_ref.astype(np.float64) + (w_s7 + dw0) * s6.astype(np.float64)

    z_areeb = unitize(load_f32(PRED / "gm_breakout/z_areeb.npy"))
    z_s7_test = unitize(
        zrk(
            pd.read_csv(
                VOL / "submissions/sub_master_slot7_verified_s7A.csv", usecols=["is_fraud"]
            )["is_fraud"].to_numpy(np.float64)
        )
    )
    m1 = unit(load_f32(PRED / "gm_breakout/m1.npy"))
    m2 = unit(load_f32(PRED / "gm_breakout/m2.npy"))
    u_ray_a = orient_test(unit(0.70 * m1.astype(np.float64) + 0.30 * m2.astype(np.float64)), z_areeb, z_s7_test)

    v10p = PRED / "v10_transductive_bulsah"
    r_v10 = lofo_resid(to_z(load_f32(v10p / "v10_lgb_d3_bag5_orth_LM_oof.npy")), bases, folds)
    r_soft = lofo_resid(to_z(load_f32(v10p / "soft_diff_orth_LM_oof.npy")), bases, folds)
    r_soft_o = lofo_resid(r_soft, [z_ref, d1, r_v10], folds)
    u_v10_t = orient_test(load_f32(v10p / "v10_lgb_d3_bag5_orth_LM_test.npy"), z_areeb, z_s7_test)
    u_soft_t = orient_test(load_f32(v10p / "soft_diff_orth_LM_test.npy"), z_areeb, z_s7_test)

    # also try pure ray continuation (b=0) and tiny soft
    ws_cal = np.unique(np.concatenate([np.linspace(0, 2, 16), np.linspace(2, 20, 28), np.linspace(20, 50, 16)]))
    steps = np.unique(
        np.concatenate(
            [
                np.linspace(2.5e-4, 3.8e-4, 27),
                np.array([4.0e-4, 4.5e-4, 5.5e-4, 6.5e-4, 8.0e-4]),
            ]
        )
    )
    # Focus on Phase-E winning band: high ray, low v10
    As = np.round(np.concatenate([np.linspace(0.60, 1.10, 11), [0.70, 0.85, 1.00]]), 3)
    Bs = np.round(np.concatenate([np.linspace(0.10, 0.35, 11), [0.0, 0.25]]), 3)
    Cs = [0.0, 0.05, 0.10]
    combos = sorted(set((float(a), float(b), float(c)) for a in As for b in Bs for c in Cs))
    log(f"combos={len(combos)} steps={len(steps)} t_need={t_for_target():.3f}")

    best_safe = None
    best_soft = None  # pos>=3, F0ray>=-5
    hit = None
    n = 0
    for a, b, c in combos:
        u_te = orient_test(
            unit(
                a * u_ray_a.astype(np.float64)
                + b * u_v10_t.astype(np.float64)
                + c * u_soft_t.astype(np.float64)
            ),
            z_areeb,
            z_s7_test,
        )
        u_o = unit(
            a * unitize(s6).astype(np.float64)
            + b * r_v10.astype(np.float64)
            + c * r_soft_o.astype(np.float64)
        )
        curve = calib_curve(z_s7_test, u_te, z_areeb, ws_cal)
        for step in steps:
            w, t, pred = row_at_step(curve, float(step))
            dw = (float(step) / 1e-4) * 0.08
            z_o = z_s7 + dw * unitize(u_o).astype(np.float64)
            ev = eval_delta(y, z_s7, z_o, folds, m0)
            f0_ray = (
                float(roc_auc_score(y[m0], z_o[m0])) - float(roc_auc_score(y[m0], z_ray[m0]))
            ) * 1e5
            row = {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "step": float(step),
                "w": w,
                "t": t,
                "pred_lin": pred,
                "oof": ev["d5_mean"],
                "pos": ev["pos"],
                "f0_ray": float(f0_ray),
                "ge_target": bool(pred >= TARGET - 1e-7),
            }
            safe = ev["pos"] >= 4 and ev["d5_mean"] > 0 and f0_ray >= -1.0
            soft = ev["pos"] >= 3 and ev["d5_mean"] > -5 and f0_ray >= -5.0
            if safe and (best_safe is None or row["pred_lin"] > best_safe["pred_lin"]):
                best_safe = row
                log(
                    f"  SAFE pred={pred:.6f} t={t:.3f} a={a} b={b} c={c} step={step:.2e} "
                    f"OOF={ev['d5_mean']:+.1f}@{ev['pos']}/5 F0ray={f0_ray:+.1f} gap={(TARGET-pred)*1e5:.2f}e-5"
                )
            if soft and (best_soft is None or row["pred_lin"] > best_soft["pred_lin"]):
                best_soft = row
            if safe and row["ge_target"]:
                hit = row
                log(f"  HIT {row}")
        n += 1
        if n % 40 == 0:
            bs = best_safe["pred_lin"] if best_safe else float("nan")
            log(f"  progress {n}/{len(combos)} best_safe={bs:.6f}")
        if n % 10 == 0:
            gc.collect()

    summary = {
        "NO_SUBMIT": True,
        "TARGET": TARGET,
        "best_safe": best_safe,
        "best_soft": best_soft,
        "hit_095369_oof_safe": hit,
        "gap_e5": None if best_safe is None else (TARGET - best_safe["pred_lin"]) * 1e5,
    }
    (ART / "phaseF_close_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    md = []
    md.append("\n---\n\n# Phase F — Close gap to 0.95369\n")
    if best_safe:
        md.append(
            f"**Best OOF-safe:** pred=`{best_safe['pred_lin']:.6f}` t=`{best_safe['t']:.3f}` "
            f"a={best_safe['a']} b={best_safe['b']} c={best_safe['c']} step={best_safe['step']:.2e} "
            f"OOF={best_safe['oof']:+.1f}@{best_safe['pos']}/5 F0ray={best_safe['f0_ray']:+.1f} "
            f"gap=`{(TARGET-best_safe['pred_lin'])*1e5:.2f}e-5`\n\n"
        )
    if best_soft:
        md.append(
            f"**Best soft-safe (pos≥3, F0ray≥−5):** pred=`{best_soft['pred_lin']:.6f}` "
            f"t=`{best_soft['t']:.3f}` a={best_soft['a']} b={best_soft['b']} c={best_soft['c']} "
            f"OOF={best_soft['oof']:+.1f}@{best_soft['pos']}/5 F0ray={best_soft['f0_ray']:+.1f}\n\n"
        )
    if hit:
        md.append(f"**HIT OOF-safe ≥0.95369:** `{hit}`\n\n")
    else:
        md.append("**No OOF-safe hit of 0.95369 in Phase F fine grid.**\n\n")
    md.append("**NO_SUBMIT=True**\n")
    rep = LOG / "GM_LEAP_095369_RESEARCH.md"
    with open(rep, "a") as f:
        f.writelines(md)
    try:
        (VOL / "logs/overnight/GM_LEAP_095369_RESEARCH.md").write_text(rep.read_text())
    except Exception:
        pass

    log(f"SUMMARY → {ART / 'phaseF_close_summary.json'}")
    if hit:
        log(f"HIT: {hit}")
    elif best_safe:
        log(f"SAFE frontier pred={best_safe['pred_lin']:.6f} gap={(TARGET-best_safe['pred_lin'])*1e5:.2f}e-5")
    if best_soft:
        log(f"SOFT frontier pred={best_soft['pred_lin']:.6f}")
    log("======== PHASE F DONE (NO_SUBMIT) ========")


if __name__ == "__main__":
    main()
