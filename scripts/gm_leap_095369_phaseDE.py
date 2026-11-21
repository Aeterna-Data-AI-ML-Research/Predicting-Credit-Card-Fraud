#!/usr/bin/env python3
"""Phase D5+E — finish gated t-push + 2D (v10, soft_diff) geometry. NO_SUBMIT."""
from __future__ import annotations

import gc
import json
import os
import resource
import subprocess
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NO_SUBMIT", "1")
warnings.filterwarnings("ignore")

VOL = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
WORK = Path(os.environ.get("GM_LEAP_ROOT", "/tmp/gm_deep_tech"))
PRED = VOL / "preds"
ART = WORK / "preds/gm_leap_095369"
LOG = WORK / "logs/overnight"
ART.mkdir(parents=True, exist_ok=True)
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
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return float(out.strip()) / 1024.0
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


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
    z = np.asarray(z, dtype=np.float64)
    return [float(roc_auc_score(y[b], z[b])) for _, b in folds]


def eval_delta(y, z_base, z_new, folds, m0=None):
    bf, nf = fold_aucs(y, z_base, folds), fold_aucs(y, z_new, folds)
    d5 = [(a - b) * 1e5 for a, b in zip(nf, bf)]
    out = {
        "d5_mean": float(np.mean(d5)),
        "pos": int(sum(x > 0 for x in d5)),
        "d5": [float(x) for x in d5],
    }
    if m0 is not None:
        out["d0"] = (
            float(roc_auc_score(y[m0], z_new[m0])) - float(roc_auc_score(y[m0], z_base[m0]))
        ) * 1e5
    return out


def weight_for_dist(z0, u, target, hi=200.0, n=64):
    u = unit(u).astype(np.float64)
    z0 = np.asarray(z0, dtype=np.float64)
    lo = 0.0
    best_w, best_err, got = 0.0, 1e9, 0.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        d = spearman_dist(z0, z0 + mid * u)
        err = abs(d - target)
        if err < best_err:
            best_err, best_w, got = err, mid, d
        if d > target:
            hi = mid
        else:
            lo = mid
    return best_w, got


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


def main():
    log("======== PHASE D5+E (NO_SUBMIT) ========")
    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = pd.read_csv(
        VOL / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv", usecols=["fold"]
    )["fold"].to_numpy(np.int8)
    m0 = fold_id == 0
    idx_f0 = np.where(m0)[0]
    y_f0 = y[idx_f0]
    subfolds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_f0)), y_f0))

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
    w_s7, dw = 0.055, (0.75e-4 / 1e-4) * 0.08
    z_s7 = z_ref.astype(np.float64) + w_s7 * s6.astype(np.float64)
    z_ray = z_ref.astype(np.float64) + (w_s7 + dw) * s6.astype(np.float64)
    rayA_f0 = z_ray[idx_f0]
    s7_f0 = z_s7[idx_f0]

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
    u_ray_a = unit(0.70 * m1.astype(np.float64) + 0.30 * m2.astype(np.float64))
    u_ray_a = orient_test(u_ray_a, z_areeb, z_s7_test)

    v10 = PRED / "v10_transductive_bulsah"
    r_v10 = lofo_resid(to_z(load_f32(v10 / "v10_lgb_d3_bag5_orth_LM_oof.npy")), bases, folds)
    r_soft = lofo_resid(to_z(load_f32(v10 / "soft_diff_orth_LM_oof.npy")), bases, folds)
    # orth soft vs v10
    r_soft_o = lofo_resid(r_soft, [z_ref, d1, r_v10], folds)

    blend = load_f32(ART / "u_phaseD_blend_delta_oof.npy")
    z_work = z_ray + blend.astype(np.float64)  # already absolute delta from prior run? check
    # prior saved (z_cur - z_ray); confirm scale
    ev_b = eval_delta(y, z_ray, z_ray + unitize(blend).astype(np.float64) * float(np.std(blend)), folds, m0)
    # safer: reconstruct blend the same way as D2
    z_cur = z_ray.copy()
    for name, r, w in (("v10", r_v10, 0.08), ("soft", r_soft_o, 0.02)):
        z_cur = z_cur + w * unitize(r).astype(np.float64)
    # soft regime lightly already in prior; keep simple 1D blend
    z_work = z_cur
    ev_work = eval_delta(y, z_ray, z_work, folds, m0)
    log(f"  reconstructed blend vsRayA={ev_work['d5_mean']:+.2f}@{ev_work['pos']}/5")

    u_v10_t = orient_test(load_f32(v10 / "v10_lgb_d3_bag5_orth_LM_test.npy"), z_areeb, z_s7_test)
    u_soft_t = orient_test(load_f32(v10 / "soft_diff_orth_LM_test.npy"), z_areeb, z_s7_test)
    # orth soft test vs v10 test + ray
    # use simple mean after orient
    u_sel_t = orient_test(unit(0.5 * u_v10_t.astype(np.float64) + 0.5 * u_soft_t.astype(np.float64)), z_areeb, z_s7_test)
    u_ray_v10 = orient_test(unit(0.55 * u_ray_a.astype(np.float64) + 0.45 * u_v10_t.astype(np.float64)), z_areeb, z_s7_test)
    u_ray_sel = orient_test(unit(0.50 * u_ray_a.astype(np.float64) + 0.50 * u_sel_t.astype(np.float64)), z_areeb, z_s7_test)
    u_ray35 = orient_test(unit(0.35 * u_ray_a.astype(np.float64) + 0.65 * u_sel_t.astype(np.float64)), z_areeb, z_s7_test)

    d_s7 = spearman_dist(z_areeb, z_s7_test)
    log(f"  S7 1-ρ={d_s7:.3e} t={100*np.sqrt(d_s7):.3f} need_t={t_for_target():.3f}")

    # ---- D5 gated ----
    log("======== D5 gated ========")
    u_push = unitize(z_work - z_ray).astype(np.float64)
    best_gated = None
    gated_rows = []
    for w in np.linspace(0.02, 0.35, 34):
        z_o = z_ray + float(w) * u_push
        step = (float(w) / 0.08) * 1e-4
        ww, _ = weight_for_dist(z_s7_test, u_ray_sel, step)
        z_te = unitize(z_s7_test.astype(np.float64) + ww * unit(u_ray_sel).astype(np.float64))
        t = 100.0 * np.sqrt(max(spearman_dist(z_areeb, z_te), 0))
        pred = pred_lb_from_t(t)
        base = [float(roc_auc_score(y_f0[b], rayA_f0[b])) for _, b in subfolds]
        d = [
            (float(roc_auc_score(y_f0[b], z_o[idx_f0][b])) - base[i]) * 1e5
            for i, (_, b) in enumerate(subfolds)
        ]
        f0 = {"mean": float(np.mean(d)), "pos": int(sum(x > 0 for x in d))}
        ev = eval_delta(y, z_ray, z_o, folds, m0)
        row = {
            "w": float(w),
            "t": float(t),
            "pred_lin": float(pred),
            "f0": f0,
            "oof": ev,
            "ge_target": bool(pred >= TARGET - 1e-7),
        }
        gated_rows.append(row)
        if f0["pos"] >= 4 and ev["pos"] >= 4 and ev["d5_mean"] > 0 and f0["mean"] > 0:
            if best_gated is None or row["pred_lin"] > best_gated["pred_lin"]:
                best_gated = row
        if abs(w - 0.08) < 1e-9 or abs(w - 0.16) < 0.011 or abs(w - 0.24) < 0.011 or row["ge_target"]:
            log(
                f"  w={w:.3f} t={t:.3f} pred={pred:.6f} OOF={ev['d5_mean']:+.1f}@{ev['pos']}/5 "
                f"F0={f0['mean']:+.1f}@{f0['pos']}/5 ≥tgt={row['ge_target']}"
            )
    if best_gated:
        log(
            f"  F0-gated frontier pred={best_gated['pred_lin']:.6f} t={best_gated['t']:.3f} "
            f"w={best_gated['w']:.3f} OOF={best_gated['oof']['d5_mean']:+.1f} ≥tgt={best_gated['ge_target']}"
        )
    else:
        log("  no F0-gated row")

    # ---- E: dense 1D geometry + 2D (ray, v10) grid ----
    log("======== E1 dense 1D near collapse ========")
    oof_dirs = {
        "ray_v10": unit(0.55 * unitize(s6).astype(np.float64) + 0.45 * r_v10.astype(np.float64)),
        "ray_sel": unit(0.50 * unitize(s6).astype(np.float64) + 0.50 * unitize(0.5 * r_v10 + 0.5 * r_soft_o).astype(np.float64)),
        "ray35": unit(0.35 * unitize(s6).astype(np.float64) + 0.65 * unitize(0.5 * r_v10 + 0.5 * r_soft_o).astype(np.float64)),
        "v10": unitize(r_v10),
        "soft_o": unitize(r_soft_o),
        "work": unitize(z_work - z_s7),
    }
    te_dirs = {
        "ray_v10": u_ray_v10,
        "ray_sel": u_ray_sel,
        "ray35": u_ray35,
        "v10": u_v10_t,
        "soft_o": u_soft_t,
        "work": orient_test(unit(0.5 * u_ray_a.astype(np.float64) + 0.5 * u_sel_t.astype(np.float64)), z_areeb, z_s7_test),
    }
    dense_steps = np.unique(
        np.concatenate(
            [
                np.linspace(0.8e-4, 3.5e-4, 28),
                np.array([0.35e-4, 0.55e-4, 0.75e-4, 4.0e-4, 4.5e-4, 5.5e-4, 6.5e-4, 8e-4]),
            ]
        )
    )
    geom = []
    safe_frontier = None
    hit = None
    for dname in ("ray_v10", "ray_sel", "ray35", "v10", "work"):
        u_te = te_dirs[dname]
        u_o = oof_dirs[dname]
        for step in dense_steps:
            w, got = weight_for_dist(z_s7_test, u_te, float(step))
            z_te = unitize(z_s7_test.astype(np.float64) + w * unit(u_te).astype(np.float64))
            t = 100.0 * np.sqrt(max(spearman_dist(z_areeb, z_te), 0))
            pred = pred_lb_from_t(t)
            dw = (float(step) / 1e-4) * 0.08
            z_o = z_s7 + dw * unitize(u_o).astype(np.float64)
            ev = eval_delta(y, z_s7, z_o, folds, m0)
            f0_ray = (
                float(roc_auc_score(y[m0], z_o[m0])) - float(roc_auc_score(y[m0], z_ray[m0]))
            ) * 1e5
            row = {
                "dir": dname,
                "step": float(step),
                "t": float(t),
                "pred_lin": float(pred),
                "oof": ev["d5_mean"],
                "pos": ev["pos"],
                "f0_ray": float(f0_ray),
                "ge_target": bool(pred >= TARGET - 1e-7),
            }
            geom.append(row)
            safe = ev["pos"] >= 4 and ev["d5_mean"] > 0 and f0_ray >= -1.0
            if safe and (safe_frontier is None or row["pred_lin"] > safe_frontier["pred_lin"]):
                safe_frontier = row
            if row["ge_target"] and safe:
                hit = row
        # log frontier-ish for this dir
        safe_dir = [r for r in geom if r["dir"] == dname and r["pos"] >= 4 and r["oof"] > 0 and r["f0_ray"] >= -1]
        if safe_dir:
            best = max(safe_dir, key=lambda r: r["pred_lin"])
            log(
                f"  {dname} safe-max pred={best['pred_lin']:.6f} t={best['t']:.3f} "
                f"step={best['step']:.2e} OOF={best['oof']:+.1f}@{best['pos']}/5 F0ray={best['f0_ray']:+.1f}"
            )
        # first collapse
        bad = [r for r in geom if r["dir"] == dname and (r["pos"] < 4 or r["f0_ray"] < -1)]
        if bad:
            first = min(bad, key=lambda r: r["step"])
            log(
                f"  {dname} first-unsafe step={first['step']:.2e} pred={first['pred_lin']:.6f} "
                f"OOF={first['oof']:+.1f}@{first['pos']}/5 F0ray={first['f0_ray']:+.1f}"
            )

    # ---- E2: 2D grid a*ray + b*v10 on TEST, OOF proxy ----
    log("======== E2 2D (ray × v10) grid ========")
    grid = []
    best2d_safe = None
    for a in [0.0, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]:
        for b in [0.0, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2]:
            if a + b < 1e-9:
                continue
            u_te = orient_test(
                unit(a * u_ray_a.astype(np.float64) + b * u_v10_t.astype(np.float64)),
                z_areeb,
                z_s7_test,
            )
            u_o = unit(a * unitize(s6).astype(np.float64) + b * r_v10.astype(np.float64))
            for step in [1.0e-4, 1.5e-4, 2.0e-4, 2.5e-4, 3.0e-4, 4.0e-4, 5.5e-4, 8.0e-4]:
                w, _ = weight_for_dist(z_s7_test, u_te, step)
                z_te = unitize(z_s7_test.astype(np.float64) + w * unit(u_te).astype(np.float64))
                t = 100.0 * np.sqrt(max(spearman_dist(z_areeb, z_te), 0))
                pred = pred_lb_from_t(t)
                dw = (step / 1e-4) * 0.08
                z_o = z_s7 + dw * unitize(u_o).astype(np.float64)
                ev = eval_delta(y, z_s7, z_o, folds, m0)
                f0_ray = (
                    float(roc_auc_score(y[m0], z_o[m0])) - float(roc_auc_score(y[m0], z_ray[m0]))
                ) * 1e5
                row = {
                    "a": a,
                    "b": b,
                    "step": step,
                    "t": float(t),
                    "pred_lin": float(pred),
                    "oof": ev["d5_mean"],
                    "pos": ev["pos"],
                    "f0_ray": float(f0_ray),
                    "ge_target": bool(pred >= TARGET - 1e-7),
                }
                grid.append(row)
                safe = ev["pos"] >= 4 and ev["d5_mean"] > 0 and f0_ray >= -1.0
                if safe and (best2d_safe is None or row["pred_lin"] > best2d_safe["pred_lin"]):
                    best2d_safe = row
                if safe and row["ge_target"] and (hit is None or row["oof"] > hit.get("oof", -1e9)):
                    hit = {**row, "dir": f"2d_a{a}_b{b}"}

    if best2d_safe:
        log(
            f"  2D safe-max pred={best2d_safe['pred_lin']:.6f} t={best2d_safe['t']:.3f} "
            f"a={best2d_safe['a']} b={best2d_safe['b']} step={best2d_safe['step']:.2e} "
            f"OOF={best2d_safe['oof']:+.1f}@{best2d_safe['pos']}/5 F0ray={best2d_safe['f0_ray']:+.1f}"
        )
    # top soft near target
    near = [r for r in grid if r["pred_lin"] >= 0.95365]
    near.sort(key=lambda r: (r["pos"], r["oof"], r["f0_ray"]), reverse=True)
    log("  near≥0.95365 by OOF:")
    for r in near[:8]:
        log(
            f"    a={r['a']:.2f} b={r['b']:.2f} step={r['step']:.2e} pred={r['pred_lin']:.6f} "
            f"OOF={r['oof']:+.1f}@{r['pos']}/5 F0ray={r['f0_ray']:+.1f}"
        )

    # ---- E3: 2D with soft_diff ----
    log("======== E3 2D (v10 × soft) + ray ========")
    best3 = None
    rows3 = []
    for a in [0.3, 0.45, 0.55]:
        for b in [0.3, 0.45, 0.6, 0.8]:
            for c in [0.0, 0.15, 0.3, 0.45]:
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
                for step in [1.2e-4, 1.8e-4, 2.2e-4, 2.8e-4, 3.5e-4, 5.5e-4]:
                    w, _ = weight_for_dist(z_s7_test, u_te, step)
                    z_te = unitize(z_s7_test.astype(np.float64) + w * unit(u_te).astype(np.float64))
                    t = 100.0 * np.sqrt(max(spearman_dist(z_areeb, z_te), 0))
                    pred = pred_lb_from_t(t)
                    dw = (step / 1e-4) * 0.08
                    z_o = z_s7 + dw * unitize(u_o).astype(np.float64)
                    ev = eval_delta(y, z_s7, z_o, folds, m0)
                    f0_ray = (
                        float(roc_auc_score(y[m0], z_o[m0])) - float(roc_auc_score(y[m0], z_ray[m0]))
                    ) * 1e5
                    row = {
                        "a": a,
                        "b": b,
                        "c": c,
                        "step": step,
                        "t": float(t),
                        "pred_lin": float(pred),
                        "oof": ev["d5_mean"],
                        "pos": ev["pos"],
                        "f0_ray": float(f0_ray),
                        "ge_target": bool(pred >= TARGET - 1e-7),
                    }
                    rows3.append(row)
                    safe = ev["pos"] >= 4 and ev["d5_mean"] > 0 and f0_ray >= -1.0
                    if safe and (best3 is None or row["pred_lin"] > best3["pred_lin"]):
                        best3 = row
                    if safe and row["ge_target"]:
                        hit = {**row, "dir": f"3d_a{a}_b{b}_c{c}"}
    if best3:
        log(
            f"  3D safe-max pred={best3['pred_lin']:.6f} t={best3['t']:.3f} "
            f"a={best3['a']} b={best3['b']} c={best3['c']} step={best3['step']:.2e} "
            f"OOF={best3['oof']:+.1f}@{best3['pos']}/5 F0ray={best3['f0_ray']:+.1f}"
        )
    near3 = [r for r in rows3 if r["pred_lin"] >= 0.95365]
    near3.sort(key=lambda r: (r["pos"], r["oof"], r["f0_ray"]), reverse=True)
    for r in near3[:6]:
        log(
            f"    3d a={r['a']:.2f} b={r['b']:.2f} c={r['c']:.2f} step={r['step']:.2e} "
            f"pred={r['pred_lin']:.6f} OOF={r['oof']:+.1f}@{r['pos']}/5 F0ray={r['f0_ray']:+.1f}"
        )

    summary = {
        "NO_SUBMIT": True,
        "TARGET": TARGET,
        "blend_vs_rayA": ev_work,
        "best_gated": best_gated,
        "safe_frontier_1d": safe_frontier,
        "best2d_safe": best2d_safe,
        "best3_safe": best3,
        "hit_095369_oof_safe": hit,
        "near_065_top": near[:5],
        "near3_065_top": near3[:5],
    }
    (ART / "phaseDE_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # append report
    md = []
    md.append("\n---\n\n# Phase D5+E — Dense / 2D / 3D geometry\n")
    md.append(f"Blend (v10+soft) vs RayA: **{ev_work['d5_mean']:+.2f}e-5** `{ev_work['pos']}/5`\n\n")
    if best_gated:
        md.append(
            f"**F0-gated frontier:** pred_lin=`{best_gated['pred_lin']:.6f}` t=`{best_gated['t']:.3f}` "
            f"w=`{best_gated['w']:.3f}` OOF=`{best_gated['oof']['d5_mean']:+.1f}` "
            f"≥tgt=`{best_gated['ge_target']}`\n\n"
        )
    if safe_frontier:
        md.append(
            f"**1D OOF-safe frontier:** `{safe_frontier['dir']}` pred=`{safe_frontier['pred_lin']:.6f}` "
            f"t=`{safe_frontier['t']:.3f}` OOF=`{safe_frontier['oof']:+.1f}`@{safe_frontier['pos']}/5 "
            f"F0ray=`{safe_frontier['f0_ray']:+.1f}`\n\n"
        )
    if best2d_safe:
        md.append(
            f"**2D (ray×v10) safe-max:** pred=`{best2d_safe['pred_lin']:.6f}` "
            f"a={best2d_safe['a']} b={best2d_safe['b']} step={best2d_safe['step']:.2e} "
            f"OOF=`{best2d_safe['oof']:+.1f}`@{best2d_safe['pos']}/5\n\n"
        )
    if best3:
        md.append(
            f"**3D (ray×v10×soft) safe-max:** pred=`{best3['pred_lin']:.6f}` "
            f"a={best3['a']} b={best3['b']} c={best3['c']} step={best3['step']:.2e} "
            f"OOF=`{best3['oof']:+.1f}`@{best3['pos']}/5\n\n"
        )
    if hit:
        md.append(f"**OOF-safe ≥0.95369 HIT:** `{hit}`\n\n")
    else:
        md.append(
            "**Still no OOF-safe path to pred_lin 0.95369.** "
            "Macro Spearman steps that clear t≈3.16 destroy 5F/F0; need new orth signal beyond v10+soft.\n\n"
        )
    md.append("**NO_SUBMIT=True**\n")
    rep = LOG / "GM_LEAP_095369_RESEARCH.md"
    with open(rep, "a") as f:
        f.writelines(md)
    try:
        (VOL / "logs/overnight/GM_LEAP_095369_RESEARCH.md").write_text(rep.read_text())
    except Exception:
        pass

    log(f"SUMMARY → {ART / 'phaseDE_summary.json'}")
    if hit:
        log(f"HIT: {hit}")
    elif safe_frontier:
        log(f"SAFE frontier pred={safe_frontier['pred_lin']:.6f} (need {TARGET})")
    log("======== PHASE D5+E DONE (NO_SUBMIT) ========")


if __name__ == "__main__":
    main()
