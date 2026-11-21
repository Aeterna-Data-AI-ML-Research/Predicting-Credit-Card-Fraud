#!/usr/bin/env python3
"""Phase D — amplify orth directions to clear pred_lin 0.95369 OOF-safe. NO_SUBMIT."""
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
ART = WORK / "preds/gm_leap_095369"
LOG = WORK / "logs/overnight"
for p in (ART, LOG):
    p.mkdir(parents=True, exist_ok=True)

assert os.environ.get("NO_SUBMIT", "1") != "0"
EPS = 1e-6
RSS_SOFT = 700.0
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
    u = unitize(delta).astype(np.float64)
    for w in ws:
        z = z_base + w * u
        d = [(float(roc_auc_score(y_f0[b], z[b])) - base[i]) * 1e5 for i, (_, b) in enumerate(subfolds)]
        row = {"w": w, "mean": float(np.mean(d)), "pos": int(sum(x > 0 for x in d)), "d": [float(x) for x in d]}
        if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
            best = row
    return best


def sweep_w(y, z_base, u, folds, m0, ws=None):
    if ws is None:
        ws = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    u = unitize(u).astype(np.float64)
    best = None
    for w in ws:
        ev = eval_delta(y, z_base, z_base + w * u, folds, m0)
        ev["w"] = w
        if best is None or (ev["pos"], ev["d5_mean"]) > (best["pos"], best["d5_mean"]):
            best = ev
    return best


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


def build_rayA(y, m0, z_ref, d1):
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
    return z_s7, z_ray, s6


def to_z(raw):
    raw = np.asarray(raw, dtype=np.float32).ravel()
    if float(raw.min()) >= -1e-6 and float(raw.max()) <= 1.0 + 1e-6:
        return unitize(zrk(raw))
    return unitize(raw)


def main():
    log("======== PHASE D: multi-orth amplify → 0.95369 (NO_SUBMIT) ========")
    log(f"t_needed≈{t_for_target():.3f} pred_lin={pred_lb_from_t(t_for_target()):.6f}")

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

    z_s7, z_ray, s6 = build_rayA(y, m0, z_ref, d1)
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
    if spearman_dist(z_areeb, z_s7_test.astype(np.float64) + 5 * u_ray_a.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7_test
    ):
        u_ray_a = (-u_ray_a.astype(np.float64)).astype(np.float32)

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
    tree_p = WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy"
    tree = unitize(load_f32(tree_p)) if tree_p.exists() else unitize(d1)
    r_disp = unitize(0.5 * r_disp0.astype(np.float64) + 0.5 * np.abs(gpt.astype(np.float64) - tree.astype(np.float64)))
    r_bnd = unitize(np.sqrt(np.clip(expit(z_ref.astype(np.float64)) * (1 - expit(z_ref.astype(np.float64))), 1e-8, None)))

    # ------------------------------------------------------------------
    # D1: screen all available orth_LM / strong residuals
    # ------------------------------------------------------------------
    log("======== D1: screen orth catalog ========")
    v10 = PRED / "v10_transductive_bulsah"
    catalog = [
        ("v10_lgb", v10 / "v10_lgb_d3_bag5_orth_LM_oof.npy"),
        ("v10_cb", v10 / "v10_cb_d4_bag3_orth_LM_oof.npy"),
        ("v10_lgb_pure", v10 / "v10_lgb_pure_fe_orth_LM_oof.npy"),
        ("v10_ridge_resid", v10 / "v10_ridge_resid_orth_LM_oof.npy"),
        ("v10_ridge_pure", v10 / "v10_ridge_pure_fe_orth_LM_oof.npy"),
        ("v10_combo3", v10 / "v10_combo3_orth_LM_oof.npy"),
        ("v10_bulsah_clean", v10 / "v10_bulsah_clean_orth_LM_oof.npy"),
        ("soft_diff", v10 / "soft_diff_orth_LM_oof.npy"),
        ("gb_soft", v10 / "gb_soft_bag3_orth_LM_oof.npy"),
        ("strike10_c1", v10 / "strike10_cand1_r_oof.npy"),
        ("strike10_c2", v10 / "strike10_cand2_r_oof.npy"),
        ("macro_orth", PRED / "macro_v11/r_macro_orth_oof.npy"),
        ("tree_logit", WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy"),
        ("L4.6C", WORK / "preds/gm_architect_v9/P4_L4.6C_hier2d_5f_oof.npy"),
        ("P4_L4.5", PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy"),
        ("harvest_xgb", PRED / "matrix_v2/harvest_p4_cuda_xgb_oof.npy"),
    ]
    # also any *orth*_oof under v10
    for p in sorted(v10.glob("*orth*_oof.npy")):
        name = p.stem.replace("_oof", "")
        if not any(str(p) == str(c[1]) for c in catalog):
            catalog.append((name[:28], p))

    screen = []
    keep_vecs = {}
    for name, path in catalog:
        if not Path(path).exists():
            continue
        try:
            z = to_z(load_f32(path))
        except Exception as e:
            log(f"  FAIL load {name}: {e}")
            continue
        r = lofo_resid(z, bases, folds)
        best1 = sweep_w(y, z_base, r, folds, m0)
        # light regime only for promising 1D
        if best1["pos"] >= 3 and best1["d5_mean"] > -2.0:
            r_reg = fit_regime_oof(y, r, r_rare, r_disp, r_bnd, folds)
            r_reg = lofo_resid(r_reg, [z_ref, d1], folds)
            bestR = sweep_w(y, z_base, r_reg, folds, m0)
            bestRay = sweep_w(y, z_ray, r_reg, folds, m0, ws=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10])
            bestRay1 = sweep_w(y, z_ray, r, folds, m0, ws=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12])
            f0R = subcv_best(y_f0, rayA_f0, r_reg[idx_f0], subfolds)
            f01 = subcv_best(y_f0, rayA_f0, r[idx_f0], subfolds)
        else:
            r_reg = r
            bestR = best1
            bestRay = sweep_w(y, z_ray, r, folds, m0, ws=[0.01, 0.02, 0.03, 0.04, 0.05])
            bestRay1 = bestRay
            f0R = subcv_best(y_f0, rayA_f0, r[idx_f0], subfolds)
            f01 = f0R
        row = {
            "name": name,
            "path": str(path),
            "1d": best1,
            "regime": bestR,
            "vsRayA_reg": bestRay,
            "vsRayA_1d": bestRay1,
            "f0_reg": f0R,
            "f0_1d": f01,
        }
        screen.append(row)
        # keep if strong vs RayA on 1d or regime
        score = max(
            (bestRay1["pos"], bestRay1["d5_mean"]),
            (bestRay["pos"], bestRay["d5_mean"]),
            (f01["pos"], f01["mean"]),
            (f0R["pos"], f0R["mean"]),
        )
        if bestRay1["pos"] >= 4 or bestRay["pos"] >= 4 or f01["pos"] >= 4 or f0R["pos"] >= 4 or best1["d5_mean"] > 10:
            keep_vecs[name] = {"r": r.astype(np.float32), "r_reg": r_reg.astype(np.float32)}
        log(
            f"  {name}: 1D={best1['d5_mean']:+.2f}@{best1['pos']}/5 "
            f"Ray1D={bestRay1['d5_mean']:+.2f}@{bestRay1['pos']}/5 "
            f"RayREG={bestRay['d5_mean']:+.2f}@{bestRay['pos']}/5 "
            f"F0={f01['mean']:+.2f}@{f01['pos']}/5"
        )
        del z
        purge()

    screen.sort(
        key=lambda r: (
            r["vsRayA_1d"]["pos"],
            r["vsRayA_1d"]["d5_mean"],
            r["f0_1d"]["pos"],
            r["f0_1d"]["mean"],
        ),
        reverse=True,
    )
    log(f"  kept vectors: {list(keep_vecs.keys())}")

    # ------------------------------------------------------------------
    # D2: greedy orthogonal blend of top directions on RayA
    # ------------------------------------------------------------------
    log("======== D2: greedy multi-orth blend on RayA ========")
    # rank by RayA 1D then F0
    ranked = [r["name"] for r in screen if r["name"] in keep_vecs]
    # force v10_lgb first if present
    if "v10_lgb" in ranked:
        ranked = ["v10_lgb"] + [n for n in ranked if n != "v10_lgb"]

    selected = []
    u_blend = None
    blend_trace = []
    z_cur = z_ray.astype(np.float64).copy()
    for name in ranked[:12]:
        cand = keep_vecs[name]["r"].astype(np.float64)
        # orth vs current score + prior selected
        bases_loc = [z_ref, d1, unitize(z_cur)]
        if selected:
            bases_loc += [keep_vecs[n]["r"] for n in selected]
        u_new = lofo_resid(cand, bases_loc, folds).astype(np.float64)
        best = sweep_w(y, z_cur, u_new, folds, m0, ws=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15])
        f0 = subcv_best(y_f0, z_cur[idx_f0], u_new[idx_f0], subfolds)
        accept = (best["pos"] >= 4 and best["d5_mean"] > 1.0) or (f0["pos"] >= 4 and f0["mean"] > 2.0)
        log(
            f"  try {name}: vs_cur={best['d5_mean']:+.2f}@{best['pos']}/5 "
            f"F0={f0['mean']:+.2f}@{f0['pos']}/5 accept={accept} w={best['w']}"
        )
        blend_trace.append({"name": name, "best": best, "f0": f0, "accept": accept})
        if accept:
            selected.append(name)
            z_cur = z_cur + best["w"] * unitize(u_new).astype(np.float64)
            u_blend = unitize(z_cur - z_ray).astype(np.float32) if u_blend is None else unitize(z_cur - z_ray)
            # also try regime of this residual
            r_reg = keep_vecs[name]["r_reg"].astype(np.float64)
            u_reg = lofo_resid(r_reg, [z_ref, d1, unitize(z_cur)], folds).astype(np.float64)
            bestR = sweep_w(y, z_cur, u_reg, folds, m0, ws=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08])
            f0R = subcv_best(y_f0, z_cur[idx_f0], u_reg[idx_f0], subfolds)
            if (bestR["pos"] >= 4 and bestR["d5_mean"] > 0.5) or (f0R["pos"] >= 4 and f0R["mean"] > 1.0):
                z_cur = z_cur + bestR["w"] * unitize(u_reg).astype(np.float64)
                log(f"    +regime {name}: {bestR['d5_mean']:+.2f}@{bestR['pos']}/5 F0={f0R['mean']:+.2f}")
            purge()

    ev_blend = eval_delta(y, z_ray, z_cur, folds, m0)
    f0_blend = subcv_best(y_f0, rayA_f0, (z_cur - z_ray)[idx_f0], subfolds)
    log(
        f"  BLEND selected={selected} vsRayA={ev_blend['d5_mean']:+.2f}@{ev_blend['pos']}/5 "
        f"F0={f0_blend['mean']:+.2f}@{f0_blend['pos']}/5"
    )
    np.save(ART / "u_phaseD_blend_delta_oof.npy", (z_cur - z_ray).astype(np.float32))

    # ------------------------------------------------------------------
    # D3: newton-cholesky fusion of selected r / r_reg on RayA margin
    # ------------------------------------------------------------------
    log("======== D3: newton fusion of selected ========")
    if not selected:
        selected = [ranked[0]] if ranked else []
    cols = []
    for n in selected:
        cols.append(keep_vecs[n]["r"])
        cols.append(keep_vecs[n]["r_reg"])
    # add v10_lgb raw even if somehow missing from selected
    if "v10_lgb" in keep_vecs and "v10_lgb" not in selected:
        cols.append(keep_vecs["v10_lgb"]["r"])
        cols.append(keep_vecs["v10_lgb"]["r_reg"])
    Xf = np.column_stack(cols + [z_ray.astype(np.float32)]).astype(np.float32)
    log(f"  fusion X={Xf.shape}")
    fus = np.zeros(len(y), dtype=np.float64)
    for fi, (a, b) in enumerate(folds):
        try:
            clf = LogisticRegression(C=0.03, solver="newton-cholesky", max_iter=400)
            clf.fit(Xf[a], y[a])
            fus[b] = clf.decision_function(Xf[b])
        except Exception:
            ridge = Ridge(alpha=1500.0)
            ridge.fit(Xf[a], y[a].astype(np.float64))
            fus[b] = ridge.predict(Xf[b])
        log(f"  fusion F{fi}")
    u_fus = lofo_resid(unitize(fus), [z_ref, d1, unitize(z_ray)], folds)
    bestF = sweep_w(y, z_ray, u_fus, folds, m0, ws=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25])
    f0F = subcv_best(y_f0, rayA_f0, u_fus[idx_f0], subfolds, ws=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10])
    z_fus = z_ray + bestF["w"] * unitize(u_fus).astype(np.float64)
    log(f"  FUSION vsRayA={bestF['d5_mean']:+.2f}@{bestF['pos']}/5 F0={f0F['mean']:+.2f}@{f0F['pos']}/5 w={bestF['w']}")
    np.save(ART / "u_phaseD_fusion_orth_oof.npy", u_fus.astype(np.float32))
    del Xf
    purge()

    # pick best working score among blend / fusion / pure v10
    candidates_z = {"blend": z_cur, "fusion": z_fus}
    if "v10_lgb" in keep_vecs:
        bv = sweep_w(y, z_ray, keep_vecs["v10_lgb"]["r"], folds, m0)
        candidates_z["pure_v10"] = z_ray + bv["w"] * unitize(keep_vecs["v10_lgb"]["r"]).astype(np.float64)
        log(f"  pure_v10 vsRayA={bv['d5_mean']:+.2f}@{bv['pos']}/5 w={bv['w']}")

    best_name, best_z, best_ev = None, None, None
    for nm, zz in candidates_z.items():
        ev = eval_delta(y, z_ray, zz, folds, m0)
        f0 = subcv_best(y_f0, rayA_f0, (zz - z_ray)[idx_f0], subfolds)
        log(f"  pick-cand {nm}: 5F={ev['d5_mean']:+.2f}@{ev['pos']}/5 F0={f0['mean']:+.2f}@{f0['pos']}/5")
        key = (ev["pos"], ev["d5_mean"], f0["pos"], f0["mean"])
        if best_ev is None or key > best_ev:
            best_ev = key
            best_name, best_z = nm, zz

    log(f"  WORKING BASE = {best_name}")
    z_work = best_z

    # ------------------------------------------------------------------
    # D4: leap geometry on TEST (areeb t-law) + OOF proxy — match Phase C
    # ------------------------------------------------------------------
    log("======== D4: leap geometry to 0.95369 ========")

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

    def orient_test(u):
        u = unit(u).astype(np.float32)
        if spearman_dist(z_areeb, z_s7_test.astype(np.float64) + 5 * u.astype(np.float64)) < spearman_dist(
            z_areeb, z_s7_test
        ):
            u = (-u.astype(np.float64)).astype(np.float32)
        return u

    # map working OOF direction → test via available test orth files
    test_dirs = {}
    v10_te = PRED / "v10_transductive_bulsah/v10_lgb_d3_bag5_orth_LM_test.npy"
    if v10_te.exists():
        test_dirs["v10"] = orient_test(load_f32(v10_te))
    # greedy blend of selected test orths
    te_parts = []
    name_to_te = {
        "v10_lgb": v10 / "v10_lgb_d3_bag5_orth_LM_test.npy",
        "v10_cb": v10 / "v10_cb_d4_bag3_orth_LM_test.npy",
        "v10_lgb_pure": v10 / "v10_lgb_pure_fe_orth_LM_test.npy",
        "soft_diff": v10 / "soft_diff_orth_LM_test.npy",
        "gb_soft": v10 / "gb_soft_bag3_orth_LM_test.npy",
        "v10_bulsah_clean": v10 / "v10_bulsah_clean_orth_LM_test.npy",
        "v10_ridge_resid": v10 / "v10_ridge_resid_orth_LM_test.npy",
        "macro_orth": PRED / "macro_v11/r_macro_orth_test.npy",
    }
    for n in selected:
        p = name_to_te.get(n)
        if p is not None and Path(p).exists():
            te_parts.append(orient_test(load_f32(p)).astype(np.float64))
    if te_parts:
        test_dirs["selected_mean"] = orient_test(unit(np.mean(np.column_stack(te_parts), axis=1)))
    # RayA + v10 (Phase C style) and RayA + selected
    if "v10" in test_dirs:
        test_dirs["ray_v10"] = orient_test(
            unit(0.55 * u_ray_a.astype(np.float64) + 0.45 * test_dirs["v10"].astype(np.float64))
        )
    if "selected_mean" in test_dirs:
        test_dirs["ray_sel"] = orient_test(
            unit(0.50 * u_ray_a.astype(np.float64) + 0.50 * test_dirs["selected_mean"].astype(np.float64))
        )
        test_dirs["ray35_sel65"] = orient_test(
            unit(0.35 * u_ray_a.astype(np.float64) + 0.65 * test_dirs["selected_mean"].astype(np.float64))
        )

    # OOF analogs
    oof_dirs = {}
    if "v10_lgb" in keep_vecs:
        oof_dirs["v10"] = unitize(keep_vecs["v10_lgb"]["r"])
        oof_dirs["ray_v10"] = unit(
            0.55 * unitize(s6).astype(np.float64) + 0.45 * keep_vecs["v10_lgb"]["r"].astype(np.float64)
        )
    oof_dirs["work_from_ray"] = unitize(z_work - z_ray)
    oof_dirs["work_from_s7"] = unitize(z_work - z_s7)
    if selected:
        oof_dirs["selected_mean"] = unitize(
            np.mean(np.column_stack([keep_vecs[n]["r"] for n in selected if n in keep_vecs]), axis=1)
        )
        oof_dirs["ray_sel"] = unit(
            0.50 * unitize(s6).astype(np.float64)
            + 0.50 * oof_dirs["selected_mean"].astype(np.float64)
        )

    d_s7_areeb = spearman_dist(z_areeb, z_s7_test)
    t_s7 = 100.0 * np.sqrt(max(d_s7_areeb, 0))
    log(f"  S7 vs areeb: 1-ρ={d_s7_areeb:.3e} t={t_s7:.3f} pred={pred_lb_from_t(t_s7):.6f}")

    steps = [
        0.10e-4, 0.25e-4, 0.35e-4, 0.55e-4, 0.75e-4, 1.00e-4, 1.20e-4, 1.50e-4,
        2.00e-4, 2.50e-4, 3.00e-4, 3.50e-4, 4.00e-4, 4.50e-4, 5.50e-4, 6.50e-4,
        8.00e-4, 10.0e-4,
    ]
    # target 1-ρ for 0.95369: t=3.157 → 1-ρ = (t/100)^2
    target_1rho = (t_for_target() / 100.0) ** 2
    log(f"  target 1-ρ for 0.95369 ≈ {target_1rho:.3e}")

    geom = []
    safe_frontier = None
    hit_095369 = None

    for dname, u_te in test_dirs.items():
        u_o = oof_dirs.get(dname, oof_dirs.get("work_from_ray", oof_dirs.get("v10")))
        if u_o is None:
            continue
        u_o = unitize(u_o)
        for step in steps:
            w, got = weight_for_dist(z_s7_test, u_te, step)
            z_te = unitize(z_s7_test.astype(np.float64) + w * unit(u_te).astype(np.float64))
            d_a = spearman_dist(z_areeb, z_te)
            t = 100.0 * np.sqrt(max(d_a, 0))
            plb = pred_lb_from_t(t)
            dw = (step / 1e-4) * 0.08
            z_o = z_s7 + dw * unitize(u_o).astype(np.float64)
            ev = eval_delta(y, z_s7, z_o, folds, m0)
            f0_s7 = (
                float(roc_auc_score(y[m0], z_o[m0])) - float(roc_auc_score(y[m0], z_s7[m0]))
            ) * 1e5
            f0_ray = (
                float(roc_auc_score(y[m0], z_o[m0])) - float(roc_auc_score(y[m0], z_ray[m0]))
            ) * 1e5
            f0_sub = subcv_best(y_f0, s7_f0, (z_o - z_s7)[idx_f0], subfolds)
            row = {
                "dir": dname,
                "step": float(step),
                "got": float(got),
                "t": float(t),
                "pred_lin": float(plb),
                "oof5f_vs_s7": ev["d5_mean"],
                "pos5": ev["pos"],
                "f0_vs_s7": float(f0_s7),
                "f0_vs_ray": float(f0_ray),
                "f0_sub": f0_sub,
                "ge_target": bool(plb >= TARGET - 1e-7),
            }
            geom.append(row)
            oof_safe = (
                ev["pos"] >= 4
                and ev["d5_mean"] > 0
                and f0_ray >= -1.0
                and f0_sub["pos"] >= 3
            )
            if oof_safe and (safe_frontier is None or row["pred_lin"] > safe_frontier["pred_lin"]):
                safe_frontier = row
            if row["ge_target"] and oof_safe:
                if hit_095369 is None or row["oof5f_vs_s7"] > hit_095369["oof5f_vs_s7"]:
                    hit_095369 = row
            if step in (0.35e-4, 1.20e-4, 2.00e-4, 4.50e-4, 8.00e-4) or row["ge_target"]:
                log(
                    f"  {dname} step={step:.2e} t={t:.3f} pred={plb:.6f} "
                    f"OOF={ev['d5_mean']:+.1f}@{ev['pos']}/5 F0ray={f0_ray:+.1f} "
                    f"sub={f0_sub['mean']:+.1f}@{f0_sub['pos']}/5 ≥tgt={row['ge_target']}"
                )

    # ------------------------------------------------------------------
    # D5: F0-gated push of working OOF on RayA + matched test t
    # ------------------------------------------------------------------
    log("======== D5: F0-gated t push on working blend ========")
    u_push = unitize(z_work - z_ray).astype(np.float64)
    # pick best available test dir for t measurement
    u_te_push = test_dirs.get("ray_sel") or test_dirs.get("ray_v10") or test_dirs.get("v10") or u_ray_a
    best_gated = None
    for w in np.linspace(0.01, 0.40, 40):
        z_o = z_ray + float(w) * u_push
        # map same relative OOF step to test via Spearman matching of OOF delta magnitude proxy
        # use dw scaling: w≈(step/1e-4)*0.08 → step = w/0.08 * 1e-4
        step = (float(w) / 0.08) * 1e-4
        ww, _ = weight_for_dist(z_s7_test, u_te_push, step)
        z_te = unitize(z_s7_test.astype(np.float64) + ww * unit(u_te_push).astype(np.float64))
        d_a = spearman_dist(z_areeb, z_te)
        t = 100.0 * np.sqrt(max(d_a, 0))
        pred = pred_lb_from_t(t)
        base = [float(roc_auc_score(y_f0[b], rayA_f0[b])) for _, b in subfolds]
        d = [
            (float(roc_auc_score(y_f0[b], z_o[idx_f0][b])) - base[i]) * 1e5
            for i, (_, b) in enumerate(subfolds)
        ]
        f0row = {"mean": float(np.mean(d)), "pos": int(sum(x > 0 for x in d)), "d": [float(x) for x in d]}
        ev = eval_delta(y, z_ray, z_o, folds, m0)
        row = {
            "w": float(w),
            "step": float(step),
            "t": float(t),
            "pred_lin": float(pred),
            "f0": f0row,
            "oof": ev,
            "ge_target": bool(pred >= TARGET - 1e-7),
        }
        if f0row["pos"] >= 4 and ev["pos"] >= 4 and ev["d5_mean"] > 0 and f0row["mean"] > -1.0:
            if best_gated is None or row["pred_lin"] > best_gated["pred_lin"]:
                best_gated = row
            if row["ge_target"] and (
                hit_095369 is None or row["oof"]["d5_mean"] > hit_095369.get("oof5f_vs_s7", -1e9)
            ):
                hit_095369 = {
                    "dir": "f0_gated_work",
                    "step": float(step),
                    "t": float(t),
                    "pred_lin": float(pred),
                    "oof5f_vs_s7": ev["d5_mean"],
                    "pos5": ev["pos"],
                    "f0_vs_ray": f0row["mean"],
                    "f0_sub": f0row,
                    "ge_target": True,
                    "w": float(w),
                }
    if best_gated:
        log(
            f"  F0-gated frontier: w={best_gated['w']:.3f} t={best_gated['t']:.3f} "
            f"pred={best_gated['pred_lin']:.6f} OOF={best_gated['oof']['d5_mean']:+.1f}@"
            f"{best_gated['oof']['pos']}/5 F0={best_gated['f0']['mean']:+.1f}@"
            f"{best_gated['f0']['pos']}/5 ≥tgt={best_gated['ge_target']}"
        )
    else:
        log("  F0-gated: no qualifying w")

    soft_hits = [
        r
        for r in geom
        if r["ge_target"] and r["pos5"] >= 3 and r["oof5f_vs_s7"] > -30 and r["f0_vs_ray"] > -50
    ]
    soft_hits.sort(key=lambda r: (r["pos5"], r["oof5f_vs_s7"], r["f0_vs_ray"]), reverse=True)

    summary = {
        "NO_SUBMIT": True,
        "ATH": ATH,
        "TARGET": TARGET,
        "t_needed": t_for_target(),
        "screen_top": [
            {
                "name": r["name"],
                "1d": r["1d"]["d5_mean"],
                "Ray1D": r["vsRayA_1d"]["d5_mean"],
                "Ray1D_pos": r["vsRayA_1d"]["pos"],
                "F0": r["f0_1d"]["mean"],
                "F0_pos": r["f0_1d"]["pos"],
            }
            for r in screen[:15]
        ],
        "selected": selected,
        "working_base": best_name,
        "blend_vs_rayA": ev_blend,
        "fusion": {"best": bestF, "f0": f0F},
        "safe_frontier": safe_frontier,
        "hit_095369_oof_safe": hit_095369,
        "best_gated": best_gated,
        "soft_hits_top3": soft_hits[:3],
        "blend_trace": [
            {"name": t["name"], "accept": t["accept"], "d5": t["best"]["d5_mean"], "pos": t["best"]["pos"]}
            for t in blend_trace
        ],
    }
    (ART / "phaseD_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # append to report
    report_path = LOG / "GM_LEAP_095369_RESEARCH.md"
    md = []
    md.append("\n---\n\n# Phase D — Multi-orth amplify\n")
    md.append(f"Elapsed PhaseD wall from script start `{time.time()-T0:.0f}s`, peak RSS ≈ `{rss_max():.0f}` MB.\n")
    md.append("## D1 Screen (top by RayA 1D)\n")
    md.append("| name | 1D Δ5F | RayA 1D | pos | F0 vsRayA | pos |\n|---|---:|---:|---:|---:|---:|\n")
    for r in screen[:12]:
        md.append(
            f"| {r['name']} | {r['1d']['d5_mean']:+.2f} | {r['vsRayA_1d']['d5_mean']:+.2f} | "
            f"{r['vsRayA_1d']['pos']}/5 | {r['f0_1d']['mean']:+.2f} | {r['f0_1d']['pos']}/5 |\n"
        )
    md.append(f"\n## D2 Greedy blend\nSelected: `{selected}`\n")
    md.append(f"- vs RayA: **{ev_blend['d5_mean']:+.2f}e-5** `{ev_blend['pos']}/5`\n")
    md.append(f"- F0 SubCV: **{f0_blend['mean']:+.2f}e-5** `{f0_blend['pos']}/5`\n")
    md.append(f"\n## D3 Fusion\n- vs RayA: **{bestF['d5_mean']:+.2f}e-5** `{bestF['pos']}/5` @w={bestF['w']}\n")
    md.append(f"- F0: **{f0F['mean']:+.2f}e-5** `{f0F['pos']}/5`\n")
    md.append(f"- Working base: **{best_name}**\n")
    md.append("\n## D4/D5 Geometry verdict\n")
    if hit_095369:
        md.append(
            f"**OOF-safe hit ≥0.95369:** dir=`{hit_095369.get('dir')}` start=`{hit_095369.get('start')}` "
            f"step=`{hit_095369.get('step')}` t=`{hit_095369['t']:.3f}` pred=`{hit_095369['pred_lin']:.6f}` "
            f"OOF=`{hit_095369['oof5f_vs_s7']:+.1f}`@{hit_095369['pos5']}/5\n"
        )
    else:
        md.append("**No OOF-safe geometry yet clears pred_lin 0.95369.**\n")
    if safe_frontier:
        md.append(
            f"- OOF-safe frontier: pred_lin=`{safe_frontier['pred_lin']:.6f}` t=`{safe_frontier['t']:.3f}` "
            f"dir=`{safe_frontier.get('dir')}` OOF=`{safe_frontier['oof5f_vs_s7']:+.1f}`@{safe_frontier['pos5']}/5\n"
        )
    if best_gated:
        md.append(
            f"- F0-gated frontier: pred_lin=`{best_gated['pred_lin']:.6f}` t=`{best_gated['t']:.3f}` "
            f"w=`{best_gated['w']:.3f}` ≥tgt=`{best_gated['ge_target']}`\n"
        )
    if soft_hits:
        md.append("\nSoft hits (pos≥3, OOF>-20, pred≥target):\n")
        for r in soft_hits[:5]:
            md.append(
                f"- {r['dir']}/{r['start']} step={r['step']} t={r['t']:.3f} pred={r['pred_lin']:.6f} "
                f"OOF={r['oof5f_vs_s7']:+.1f}@{r['pos5']}/5\n"
            )
    md.append("\n**NO_SUBMIT=True** — research only.\n")
    with open(report_path, "a") as f:
        f.writelines(md)
    # sync
    vol_rep = VOL / "logs/overnight/GM_LEAP_095369_RESEARCH.md"
    try:
        vol_rep.write_text(report_path.read_text())
    except Exception:
        pass

    log(f"SUMMARY → {ART / 'phaseD_summary.json'}")
    log(f"REPORT appended → {report_path}")
    if hit_095369:
        log(f"HIT OOF-safe 0.95369+ : {hit_095369}")
    elif safe_frontier:
        log(f"SAFE frontier pred={safe_frontier['pred_lin']:.6f} (need {TARGET})")
    else:
        log("No safe frontier recorded")
    log("======== PHASE D DONE (NO_SUBMIT) ========")


if __name__ == "__main__":
    main()
