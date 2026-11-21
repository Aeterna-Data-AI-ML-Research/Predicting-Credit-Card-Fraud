#!/usr/bin/env python3
"""GM Leap Research toward 0.95369+ — NO_SUBMIT, no CSVs.

Centers on v9 disk-rescue winner v10_lgb_d3_bag5_orth_LM (+57e-5@5/5 raw)
+ L4.6A/B/C + Ray-A Fold0 geometry. Strict sequential float32; soft RSS < 600MB.
"""
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
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NO_SUBMIT", "1")
warnings.filterwarnings("ignore")

VOL = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
WORK = Path(os.environ.get("GM_LEAP_ROOT", "/tmp/gm_deep_tech"))
PRED = VOL / "preds"
ART_V9 = WORK / "preds/gm_architect_v9"
ART = WORK / "preds/gm_leap_095369"
LOG = WORK / "logs/overnight"
VOL_LOG = VOL / "logs/overnight"
VOL_ART = VOL / "preds/gm_leap_095369"
for p in (ART, LOG):
    p.mkdir(parents=True, exist_ok=True)

assert os.environ.get("NO_SUBMIT", "1") != "0"
EPS = 1e-6
RSS_SOFT = 600.0
T0 = time.time()
ATH = 0.95354
TARGET = 0.95369

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def rss_curr():
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
        return float(out.strip()) / 1024.0
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def rss_max():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s|RSS={rss_curr():.0f}/{rss_max():.0f}MB] {msg}", flush=True)
    if rss_curr() > RSS_SOFT:
        print(f"  WARN RSS {rss_curr():.0f} > soft {RSS_SOFT}", flush=True)


def purge():
    gc.collect()


def load_f32(p: Path) -> np.ndarray:
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
        "auc_base": float(roc_auc_score(y, z_base)),
        "auc_new": float(roc_auc_score(y, z_new)),
    }
    if m0 is not None:
        out["d0"] = (
            float(roc_auc_score(y[m0], z_new[m0])) - float(roc_auc_score(y[m0], z_base[m0]))
        ) * 1e5
    return out


def subcv_best(y_f0, z_base, delta, subfolds, ws=None):
    if ws is None:
        ws = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]
    base = [float(roc_auc_score(y_f0[b], z_base[b])) for _, b in subfolds]
    best = None
    rows = []
    u = unitize(delta).astype(np.float64)
    for w in ws:
        z = z_base + w * u
        d = [(float(roc_auc_score(y_f0[b], z[b])) - base[i]) * 1e5 for i, (_, b) in enumerate(subfolds)]
        row = {"w": w, "mean": float(np.mean(d)), "pos": int(sum(x > 0 for x in d)), "d": [float(x) for x in d]}
        rows.append(row)
        if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
            best = row
    return best, rows


def sweep_w(y, z_base, u, folds, m0, ws=None):
    if ws is None:
        ws = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    u = unitize(u).astype(np.float64)
    best = None
    rows = []
    for w in ws:
        ev = eval_delta(y, z_base, z_base + w * u, folds, m0)
        ev["w"] = w
        rows.append(ev)
        if best is None or (ev["pos"], ev["d5_mean"]) > (best["pos"], best["d5_mean"]):
            best = ev
    return best, rows


def regime_expand(r, r_rare, r_disp, r_bnd):
    r = np.asarray(r, dtype=np.float64).ravel()
    return np.column_stack([r, r * r_rare, r * r_disp, r * r_bnd, np.maximum(0.0, r - 1.5)]).astype(np.float32)


def fit_regime_oof(y, r, r_rare, r_disp, r_bnd, folds, C=0.02):
    X = regime_expand(r, r_rare, r_disp, r_bnd)
    oof = np.zeros(len(y), dtype=np.float64)
    for a, b in folds:
        try:
            clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=200)
            clf.fit(X[a], y[a])
            oof[b] = clf.decision_function(X[b])
        except Exception:
            ridge = Ridge(alpha=5000.0)
            ridge.fit(X[a], y[a].astype(np.float64))
            oof[b] = ridge.predict(X[b])
    return unitize(oof)


def pred_lb_from_t(t):
    return 0.95346910 + 6.99725804e-05 * t


def t_for_target(lb=TARGET):
    return (lb - 0.95346910) / 6.99725804e-05


def build_rayA(y, m0, idx_f0, z_ref, d1, folds):
    gpt = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))
    u_sb = unit(d1)
    m1 = unitize(
        0.65 * u_sb.astype(np.float64)
        - 0.35 * (-unit(gpt)).astype(np.float64)
        - 0.20 * gpt.astype(np.float64)
    )
    m2 = unitize(u_sb.astype(np.float64))
    s6 = unitize(0.70 * m1.astype(np.float64) + 0.30 * m2.astype(np.float64))
    w_s7, dw = 0.055, (0.75e-4 / 1e-4) * 0.08
    z_s7 = z_ref.astype(np.float64) + w_s7 * s6.astype(np.float64)
    z_ray = z_ref.astype(np.float64) + (w_s7 + dw) * s6.astype(np.float64)
    base0 = float(roc_auc_score(y[m0], z_ref[m0]))
    log(
        f"  S7_f0={(float(roc_auc_score(y[m0], z_s7[m0]))-base0)*1e5:+.2f}e-5 "
        f"RayA={(float(roc_auc_score(y[m0], z_ray[m0]))-base0)*1e5:+.2f}e-5"
    )
    return z_s7, z_ray, s6


def main():
    log("======== GM LEAP 0.95369+ RESEARCH (NO_SUBMIT) ========")
    log(f"ATH={ATH} TARGET={TARGET}  t_needed≈{t_for_target():.3f}  pred_lin@that={pred_lb_from_t(t_for_target()):.6f}")

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
    z_base = z_ref.astype(np.float64) + 0.04 * d1.astype(np.float64)
    log(f"z_base AUC={roc_auc_score(y, z_base):.7f}")

    z_s7, z_ray, s6 = build_rayA(y, m0, idx_f0, z_ref, d1, folds)
    rayA_f0 = z_ray[idx_f0]
    s7_f0 = z_s7[idx_f0]

    # regimes
    meta = json.loads((PRED / "gm_v5_overnight/X_v5_meta.json").read_text())
    extra = meta.get("extra_names", [])
    base_extra = 68 + int(meta.get("n_kmat5seed", 15))
    Xo = np.load(PRED / "gm_v5_overnight/X_v5_oof.npy", mmap_mode="r")
    if "z_lm_x_rare" in extra:
        r_rare = unitize(np.asarray(Xo[:, base_extra + extra.index("z_lm_x_rare")], dtype=np.float32))
    else:
        r_rare = unitize(np.asarray(Xo[:, 0], dtype=np.float32))
    if "z_lm_x_disp" in extra:
        r_disp0 = unitize(np.asarray(Xo[:, base_extra + extra.index("z_lm_x_disp")], dtype=np.float32))
    else:
        r_disp0 = unitize(np.abs(gpt.astype(np.float64) - d1.astype(np.float64)))
    del Xo
    purge()
    tree = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy"))
    r_disp = unitize(0.5 * r_disp0.astype(np.float64) + 0.5 * np.abs(gpt.astype(np.float64) - tree.astype(np.float64)))
    r_bnd = unitize(np.sqrt(np.clip(expit(z_ref.astype(np.float64)) * (1 - expit(z_ref.astype(np.float64))), 1e-8, None)))

    # ------------------------------------------------------------------
    # Phase A: re-verify disk winners + score L4.6
    # ------------------------------------------------------------------
    log("======== Phase A: verify winners + L4.6 ========")
    cand_paths = {
        "v10_lgb_orth": (
            PRED / "v10_transductive_bulsah/v10_lgb_d3_bag5_orth_LM_oof.npy",
            ART_V9 / "rescued_0_v10_lgb_d3_bag5_orth_LM_oof.npy",
        ),
        "v10_combo3_orth": (PRED / "v10_transductive_bulsah/v10_combo3_orth_LM_oof.npy",),
        "v10_ridge_orth": (PRED / "v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy",),
        "tree_logit_v72": (WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy", ART_V9 / "rescued_1_tree_logit_v72_oof.npy"),
        "L4.6A": (ART_V9 / "P4_L4.6A_stepgrid_5f_oof.npy",),
        "L4.6B": (ART_V9 / "P4_L4.6B_bulsah_5f_oof.npy",),
        "L4.6C": (ART_V9 / "P4_L4.6C_hier2d_5f_oof.npy",),
        "P4_L4.5": (PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy",),
        "P4_L4.3_ext": (PRED / "matrix_v2/P4_L4.3_ext_oof.npy",),
    }

    vectors = {}
    phaseA = []
    for name, paths in cand_paths.items():
        path = next((Path(p) for p in paths if Path(p).exists()), None)
        if path is None:
            log(f"  SKIP {name}")
            continue
        raw = load_f32(path)
        if float(raw.min()) >= -1e-6 and float(raw.max()) <= 1.0 + 1e-6:
            z = unitize(zrk(raw))
        else:
            z = unitize(raw)
        # LOFO orth against [z_ref, d1, w12, mega, gpt]
        r = lofo_resid(z, bases, folds)
        best1, _ = sweep_w(y, z_base, r, folds, m0)
        r_reg = fit_regime_oof(y, r, r_rare, r_disp, r_bnd, folds)
        r_reg = lofo_resid(r_reg, [z_ref, d1], folds)
        bestR, _ = sweep_w(y, z_base, r_reg, folds, m0)
        # vs RayA full OOF (not just f0) — use z_ray as base
        bestRay, _ = sweep_w(y, z_ray, r_reg, folds, m0, ws=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08])
        f0R, _ = subcv_best(y_f0, rayA_f0, r_reg[idx_f0], subfolds)
        f01, _ = subcv_best(y_f0, rayA_f0, r[idx_f0], subfolds)
        row = {
            "name": name,
            "path": str(path),
            "1d": best1,
            "regime": bestR,
            "vs_rayA_5f": bestRay,
            "f0_1d": f01,
            "f0_regime": f0R,
        }
        phaseA.append(row)
        vectors[name] = {"r": r, "r_reg": r_reg}
        log(
            f"  {name}: 1D={best1['d5_mean']:+.2f}@{best1['pos']}/5 "
            f"REG={bestR['d5_mean']:+.2f}@{bestR['pos']}/5 "
            f"vsRayA5F={bestRay['d5_mean']:+.2f}@{bestRay['pos']}/5 "
            f"F0reg={f0R['mean']:+.2f}@{f0R['pos']}/5"
        )
        del raw, z
        purge()

    phaseA.sort(key=lambda r: (r["regime"]["pos"], r["regime"]["d5_mean"]), reverse=True)

    # ------------------------------------------------------------------
    # Phase B: LOFO stack of positive regime winners
    # ------------------------------------------------------------------
    log("======== Phase B: LOFO stack of regime winners ========")
    # pick vectors with regime pos>=4 or vs_rayA pos>=4 or f0 pos>=4
    winners = []
    for row in phaseA:
        if row["regime"]["pos"] >= 4 or row["vs_rayA_5f"]["pos"] >= 4 or row["f0_regime"]["pos"] >= 4:
            winners.append(row["name"])
    if "v10_lgb_orth" not in winners and "v10_lgb_orth" in vectors:
        winners.insert(0, "v10_lgb_orth")
    # always include L4.6 if present for diversity
    for extra_n in ("L4.6A", "L4.6B", "L4.6C", "tree_logit_v72"):
        if extra_n in vectors and extra_n not in winners:
            # include only if not strongly negative
            row = next(r for r in phaseA if r["name"] == extra_n)
            if row["regime"]["d5_mean"] > -5.0:
                winners.append(extra_n)
    log(f"  winner set: {winners}")

    # Build LOFO blend: for each outer fold, Ridge on winner r_reg columns to predict y, with base_margin-ish via residualizing z_ray
    # Target: residual of y after Platt(z_ray) — use y itself with features = winner regs + z_ray col
    ntr = len(y)
    Xw = np.column_stack([vectors[n]["r_reg"] for n in winners if n in vectors]).astype(np.float32)
    # also raw r for v10_lgb (huge 1D)
    if "v10_lgb_orth" in vectors:
        Xw = np.column_stack([Xw, vectors["v10_lgb_orth"]["r"]]).astype(np.float32)
    log(f"  stack X={Xw.shape}")

    stack_oof = np.zeros(ntr, dtype=np.float64)
    for fi, (a, b) in enumerate(folds):
        # features: winners + z_ray
        Xa = np.column_stack([Xw[a], z_ray[a].astype(np.float32)])
        Xb = np.column_stack([Xw[b], z_ray[b].astype(np.float32)])
        try:
            clf = LogisticRegression(C=0.05, solver="newton-cholesky", max_iter=300)
            clf.fit(Xa, y[a])
            stack_oof[b] = clf.decision_function(Xb)
        except Exception:
            ridge = Ridge(alpha=2000.0)
            ridge.fit(Xa, y[a].astype(np.float64))
            stack_oof[b] = ridge.predict(Xb)
        log(f"  stack F{fi}")
        del Xa, Xb
        purge()

    u_stack = unitize(stack_oof)
    # orth stack residual vs z_ray / z_base axes
    u_stack_orth = lofo_resid(u_stack, [z_ref, d1, unitize(z_ray)], folds)
    best_stack, rows_stack = sweep_w(y, z_ray, u_stack_orth, folds, m0)
    best_stack_base, _ = sweep_w(y, z_base, u_stack_orth, folds, m0)
    f0_stack, f0_stack_rows = subcv_best(y_f0, rayA_f0, u_stack_orth[idx_f0], subfolds)
    log(
        f"  STACK vs RayA: {best_stack['d5_mean']:+.2f}e-5@{best_stack['pos']}/5 "
        f"F0={f0_stack['mean']:+.2f}@{f0_stack['pos']}/5"
    )
    log(f"  STACK vs z_base: {best_stack_base['d5_mean']:+.2f}e-5@{best_stack_base['pos']}/5")

    # ------------------------------------------------------------------
    # Phase C: geometric leap — how far along (RayA + stack) to 0.95369 t
    # ------------------------------------------------------------------
    log("======== Phase C: geometry to 0.95369 ========")
    z_s7_test = unitize(
        zrk(
            pd.read_csv(
                VOL / "submissions/sub_master_slot7_verified_s7A.csv", usecols=["is_fraud"]
            )["is_fraud"].to_numpy(np.float64)
        )
    )
    z_areeb = unitize(load_f32(PRED / "gm_breakout/z_areeb.npy"))
    m1 = unit(load_f32(PRED / "gm_breakout/m1.npy"))
    m2 = unit(load_f32(PRED / "gm_breakout/m2.npy"))
    u_ray_a = unit(0.70 * m1.astype(np.float64) + 0.30 * m2.astype(np.float64))
    if spearman_dist(z_areeb, z_s7_test.astype(np.float64) + 5 * u_ray_a.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7_test
    ):
        u_ray_a = (-u_ray_a.astype(np.float64)).astype(np.float32)

    # map OOF stack direction to test via rank alignment of rescued v10 test if present
    v10_te = PRED / "v10_transductive_bulsah/v10_lgb_d3_bag5_orth_LM_test.npy"
    if v10_te.exists():
        u_v10_t = unitize(load_f32(v10_te))
    else:
        u_v10_t = unitize(load_f32(ART_V9 / "rescued_0_v10_lgb_d3_bag5_orth_LM_test.npy"))
    # orient
    if float(np.dot(u_v10_t.astype(np.float64), u_ray_a.astype(np.float64))) < 0:
        # compare via areeb distance increase
        pass
    if spearman_dist(z_areeb, z_s7_test.astype(np.float64) + 5 * u_v10_t.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7_test
    ):
        u_v10_t = (-u_v10_t.astype(np.float64)).astype(np.float32)

    # leap direction: freeze-ish RayA continuation + v10_lgb
    u_leap = unit(0.55 * u_ray_a.astype(np.float64) + 0.45 * u_v10_t.astype(np.float64))
    if spearman_dist(z_areeb, z_s7_test.astype(np.float64) + 5 * u_leap.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7_test
    ):
        u_leap = (-u_leap.astype(np.float64)).astype(np.float32)

    # OOF analog of leap direction
    u_leap_o = unit(
        0.55 * unitize(s6).astype(np.float64)
        + 0.45 * vectors.get("v10_lgb_orth", {}).get("r_reg", vectors.get("v10_lgb_orth", {}).get("r", s6)).astype(np.float64)
    )
    if "v10_lgb_orth" in vectors:
        u_leap_o = unit(0.55 * unitize(s6).astype(np.float64) + 0.45 * vectors["v10_lgb_orth"]["r"].astype(np.float64))

    # From S7 / RayA, sweep macro steps and report OOF + implied LB t
    d_s7_areeb = spearman_dist(z_areeb, z_s7_test)
    t_s7 = 100.0 * np.sqrt(max(d_s7_areeb, 0))
    log(f"  S7 vs areeb: 1-ρ={d_s7_areeb:.3e} t={t_s7:.3f} pred={pred_lb_from_t(t_s7):.6f}")

    def weight_for_dist(z0, u, target, hi=150.0, n=56):
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

    # steps in 1-ρ(S7) and absolute t targets
    geom_rows = []
    for step in [0.10e-4, 0.35e-4, 0.75e-4, 1.20e-4, 2.00e-4, 3.00e-4, 4.50e-4, 6.00e-4]:
        w, got = weight_for_dist(z_s7_test, u_leap, step)
        z = unitize(z_s7_test.astype(np.float64) + w * unit(u_leap).astype(np.float64))
        d_a = spearman_dist(z_areeb, z)
        t = 100.0 * np.sqrt(max(d_a, 0))
        plb = pred_lb_from_t(t)
        # OOF proxy: from z_s7 along u_leap_o with same relative weight scale
        dw = (step / 1e-4) * 0.08
        z_o = z_s7 + dw * unitize(u_leap_o).astype(np.float64)
        ev = eval_delta(y, z_s7, z_o, folds, m0)
        f0 = (
            float(roc_auc_score(y[m0], z_o[m0])) - float(roc_auc_score(y[m0], z_s7[m0]))
        ) * 1e5
        # also vs RayA peak for F0
        f0_vs_ray = (
            float(roc_auc_score(y[m0], z_o[m0])) - float(roc_auc_score(y[m0], z_ray[m0]))
        ) * 1e5
        row = {
            "step_1rho_S7": step,
            "got": got,
            "t": t,
            "pred_lin": plb,
            "oof_vsS7_5F": ev["d5_mean"],
            "oof_pos": ev["pos"],
            "oof_d5": ev["d5"],
            "oof_F0_vsS7": f0,
            "oof_F0_vsRayA": f0_vs_ray,
            "hit_095369": plb >= TARGET - 1e-7,
        }
        geom_rows.append(row)
        log(
            f"  step={step:.2e} t={t:.3f} pred={plb:.6f} "
            f"OOF5FvsS7={ev['d5_mean']:+.2f}@{ev['pos']}/5 F0vsS7={f0:+.2f} F0vsRayA={f0_vs_ray:+.2f}"
        )

    # find OOF-safe frontier: max pred_lin among steps with F0_vsRayA>=0 and pos>=3, else best F0
    safe = [r for r in geom_rows if r["oof_F0_vsRayA"] >= -0.5 and r["oof_pos"] >= 3]
    if not safe:
        safe = [r for r in geom_rows if r["oof_F0_vsS7"] >= 0]
    frontier = max(safe, key=lambda r: (r["pred_lin"], r["oof_F0_vsRayA"])) if safe else geom_rows[0]
    peak_f0 = max(geom_rows, key=lambda r: r["oof_F0_vsS7"])
    log(f"  OOF-safe frontier: step={frontier['step_1rho_S7']:.2e} pred={frontier['pred_lin']:.6f}")
    log(f"  Fold0 peak along leap: step={peak_f0['step_1rho_S7']:.2e} F0vsS7={peak_f0['oof_F0_vsS7']:+.2f}")

    # pure v10 add on RayA (no ray continuation)
    if "v10_lgb_orth" in vectors:
        best_v10_on_ray, v10_rows = sweep_w(
            y, z_ray, vectors["v10_lgb_orth"]["r"], folds, m0, ws=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
        )
        f0_v10, _ = subcv_best(y_f0, rayA_f0, vectors["v10_lgb_orth"]["r"][idx_f0], subfolds)
        log(
            f"  pure v10_lgb on RayA: 5F={best_v10_on_ray['d5_mean']:+.2f}@{best_v10_on_ray['pos']}/5 "
            f"F0Sub={f0_v10['mean']:+.2f}@{f0_v10['pos']}/5"
        )
    else:
        best_v10_on_ray, f0_v10, v10_rows = None, None, []

    np.save(ART / "u_leap_oof.npy", u_leap_o)
    np.save(ART / "u_leap_test.npy", u_leap)
    np.save(ART / "u_stack_orth_oof.npy", u_stack_orth)
    if "v10_lgb_orth" in vectors:
        np.save(ART / "u_v10_lgb_orth_oof.npy", vectors["v10_lgb_orth"]["r"])

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    md = []
    md.append("# GM Leap Research — Path to 0.95369+")
    md.append("")
    md.append(f"**NO_SUBMIT=True**. No submission CSVs. ATH `{ATH}` → target `{TARGET}`.")
    md.append(f"Elapsed `{time.time()-T0:.1f}s`, peak RSS ≈ `{rss_max():.0f}` MB.")
    md.append("")
    md.append(f"Linear LB law needs **t≈{t_for_target():.3f}** for 0.95369 (S7 is t≈{t_s7:.3f}, pred `{pred_lb_from_t(t_s7):.6f}`).")
    md.append("")
    md.append("## Phase A — Disk / L4.6 verification (vs z_base & RayA)")
    md.append("")
    md.append("| name | 1D Δ5F | pos | REG Δ5F | pos | vsRayA Δ5F | pos | F0 REG vsRayA | pos |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in phaseA:
        md.append(
            f"| {r['name']} | {r['1d']['d5_mean']:+.2f} | {r['1d']['pos']}/5 | "
            f"{r['regime']['d5_mean']:+.2f} | {r['regime']['pos']}/5 | "
            f"{r['vs_rayA_5f']['d5_mean']:+.2f} | {r['vs_rayA_5f']['pos']}/5 | "
            f"{r['f0_regime']['mean']:+.2f} | {r['f0_regime']['pos']}/5 |"
        )
    md.append("")
    md.append("## Phase B — LOFO stack")
    md.append("")
    md.append(f"Winners: `{winners}`")
    md.append(
        f"- vs RayA: **{best_stack['d5_mean']:+.2f}e-5** pos `{best_stack['pos']}/5` d5=`{[round(x,2) for x in best_stack['d5']]}`"
    )
    md.append(
        f"- vs z_base: **{best_stack_base['d5_mean']:+.2f}e-5** pos `{best_stack_base['pos']}/5`"
    )
    md.append(f"- Fold0 SubCV vs RayA: **{f0_stack['mean']:+.2f}e-5** pos `{f0_stack['pos']}/5` @w={f0_stack['w']}")
    md.append("")
    md.append("## Phase C — Leap geometry (RayA + v10_lgb direction)")
    md.append("")
    md.append("| 1−ρ(S7) | t | pred_lin | OOF5F vsS7 | pos | F0 vsS7 | F0 vsRayA | ≥0.95369? |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in geom_rows:
        md.append(
            f"| {r['step_1rho_S7']:.2e} | {r['t']:.3f} | {r['pred_lin']:.6f} | "
            f"{r['oof_vsS7_5F']:+.2f} | {r['oof_pos']}/5 | {r['oof_F0_vsS7']:+.2f} | "
            f"{r['oof_F0_vsRayA']:+.2f} | {'YES' if r['hit_095369'] else 'no'} |"
        )
    md.append("")
    md.append(
        f"**OOF-safe frontier:** step=`{frontier['step_1rho_S7']:.2e}` pred_lin=`{frontier['pred_lin']:.6f}` "
        f"(F0vsRayA=`{frontier['oof_F0_vsRayA']:+.2f}`, pos=`{frontier['oof_pos']}/5`)."
    )
    md.append(
        f"**Fold-0 peak along leap:** step=`{peak_f0['step_1rho_S7']:.2e}` F0vsS7=`{peak_f0['oof_F0_vsS7']:+.2f}` "
        f"pred_lin=`{peak_f0['pred_lin']:.6f}`."
    )
    if best_v10_on_ray:
        md.append("")
        md.append("## Pure v10_lgb_orth on RayA (no macro ray push)")
        md.append(
            f"- 5F: **{best_v10_on_ray['d5_mean']:+.2f}e-5** `{best_v10_on_ray['pos']}/5` @w={best_v10_on_ray['w']} "
            f"d5=`{[round(x,2) for x in best_v10_on_ray['d5']]}`"
        )
        md.append(f"- Fold0 SubCV: **{f0_v10['mean']:+.2f}e-5** `{f0_v10['pos']}/5` @w={f0_v10['w']}")
    md.append("")
    md.append("## Verdict")
    md.append("")
    # verdict logic
    if best_v10_on_ray and best_v10_on_ray["pos"] >= 5 and best_v10_on_ray["d5_mean"] > 1.0:
        md.append(
            f"- **Primary unlock:** `v10_lgb_d3_bag5_orth_LM` lifts RayA by "
            f"**{best_v10_on_ray['d5_mean']:+.2f}e-5 (5/5)** — strongest post-S7 signal found."
        )
    if frontier["pred_lin"] >= TARGET:
        md.append(
            f"- Geometry can reach pred_lin `{frontier['pred_lin']:.6f}` while keeping OOF non-destructive "
            f"at step `{frontier['step_1rho_S7']:.2e}`."
        )
    else:
        md.append(
            f"- OOF-safe frontier pred_lin `{frontier['pred_lin']:.6f}` is still "
            f"**below** `{TARGET}`; reaching 0.95369 on the linear law requires larger Spearman steps "
            f"where Fold-0/5F currently degrade — need more orthogonal lift (v10-scale) before macro geometry."
        )
    md.append("")
    md.append("**NO_SUBMIT=True** — research only.")
    report = "\n".join(md) + "\n"
    rp = LOG / "GM_LEAP_095369_RESEARCH.md"
    rp.write_text(report)
    try:
        VOL_LOG.mkdir(parents=True, exist_ok=True)
        (VOL_LOG / "GM_LEAP_095369_RESEARCH.md").write_text(report)
        VOL_ART.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log(f"vol sync warn: {e}")

    summary = {
        "NO_SUBMIT": True,
        "ATH": ATH,
        "TARGET": TARGET,
        "phaseA": phaseA,
        "winners": winners,
        "best_stack": best_stack,
        "best_stack_base": best_stack_base,
        "f0_stack": f0_stack,
        "geom_rows": geom_rows,
        "frontier": frontier,
        "peak_f0": peak_f0,
        "best_v10_on_ray": best_v10_on_ray,
        "f0_v10": f0_v10,
        "rss_max": rss_max(),
        "elapsed": time.time() - T0,
    }
    (ART / "leap_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    try:
        (VOL / "scripts").mkdir(parents=True, exist_ok=True)
        src = Path(__file__).resolve()
        (VOL / "scripts" / src.name).write_text(src.read_text())
    except Exception as e:
        log(f"script sync warn: {e}")
    log(f"REPORT → {rp}")
    log("======== LEAP RESEARCH DONE (NO_SUBMIT) ========")


if __name__ == "__main__":
    main()
