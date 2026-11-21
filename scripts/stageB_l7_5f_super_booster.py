#!/usr/bin/env python3
"""Phase 3: Stage-B × L7_5F GPU super-booster (<4 min on warm A40).

Builds L7_oof_5f / L7_test_5f from rotated fold tensors, PCHIP-locks test,
trains 5-seed XGB+LGB Stage-B with L7 interaction features.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(os.environ.get("L7_ROOT", "/workspace/l7_work"))
L7 = Path(os.environ.get("L7_OUT", str(ROOT / "preds" / "l7_5f_rot10")))
OUT = Path(os.environ.get("STAGEB_OUT", str(ROOT / "preds" / "stageB_l7_5f")))
PRED = Path(os.environ.get("PRED_ROOT", str(ROOT / "preds")))
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), 1e-6, 1 - 1e-6))


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return ((v - v.mean()) / (v.std() + 1e-12)).astype(np.float32)


def bayes_ll(ll, prior=None):
    t = np.logaddexp.reduce(ll[:, :, 1].astype(np.float64), axis=1)
    f = np.logaddexp.reduce(ll[:, :, 0].astype(np.float64), axis=1)
    z = t - f
    if prior is not None:
        z = z + np.log(prior / (1 - prior))
    return z.astype(np.float32)


def pchip_map(src, ref):
    s = np.sort(np.asarray(src, dtype=np.float64).ravel())
    r = np.sort(np.asarray(ref, dtype=np.float64).ravel())
    # unique for PCHIP
    su, idx = np.unique(s, return_index=True)
    ru = r[idx.clip(0, len(r) - 1)] if len(idx) != len(su) else np.interp(
        np.linspace(0, 1, len(su)), np.linspace(0, 1, len(r)), r
    )
    if len(su) < 4:
        return src.astype(np.float32)
    # match quantiles
    qs = np.linspace(0, 1, min(5000, len(s)))
    xs = np.quantile(s, qs)
    ys = np.quantile(r, qs)
    xs, uidx = np.unique(xs, return_index=True)
    ys = ys[uidx]
    if len(xs) < 4:
        return src.astype(np.float32)
    f = PchipInterpolator(xs, ys, extrapolate=True)
    return f(src).astype(np.float32)


def calibrate_logit_to_prevalence(z, p=None):
    z = np.asarray(z, dtype=np.float64).ravel()
    if p is None:
        # match mean probability ~ train prevalence ~0.0167 via shift
        p = 0.0167
    # find b s.t. mean(sigmoid(z+b)) = p
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        m = float(expit(z + mid).mean())
        if m > p:
            hi = mid
        else:
            lo = mid
    return (z + 0.5 * (lo + hi)).astype(np.float32)


def load_fold_ll(kind: str, f: int):
    p = L7 / f"{kind}_{f}.npy"
    a = np.load(p)
    return a


def build_l7_5f(y, folds):
    ntr = len(y)
    oof_z = np.full(ntr, np.nan, np.float64)
    # per-fold test accumulators over orders
    test_acc = []
    for f, (tr, te) in enumerate(folds):
        o = load_fold_ll("oof", f)
        assert o.shape[0] == ntr and o.shape[2] == 2
        # held-out rows only
        ll = o[te]
        # if fold0-style full tensor with zeros elsewhere, rows outside te may be 0 — use te only
        prior = float(y[tr].mean())
        z = bayes_ll(ll, prior)
        oof_z[te] = z
        t = load_fold_ll("test", f)
        assert t.shape == (299984, 4, 2), t.shape
        test_acc.append(bayes_ll(t, prior))
    assert np.isfinite(oof_z).all()
    test_z = np.mean(np.stack(test_acc, 0), axis=0)
    return oof_z.astype(np.float32), test_z.astype(np.float32)


def std_match_fold(oof, mega, folds):
    out = np.empty_like(oof, dtype=np.float64)
    for tr, te in folds:
        x = oof[te].astype(np.float64)
        m = mega[te].astype(np.float64)
        out[te] = (x - x.mean()) / (x.std() + 1e-12) * (m.std() + 1e-12) + m.mean()
    return out.astype(np.float32)


def main():
    log("======== Stage-B × L7_5F super-booster ========")
    y = pd.read_csv(ROOT / "data" / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))

    l7_o, l7_t = build_l7_5f(y, folds)
    np.save(OUT / "L7_oof_5f_raw.npy", l7_o)
    np.save(OUT / "L7_test_5f_raw.npy", l7_t)
    log(f"L7 raw OOF AUC={roc_auc_score(y, l7_o):.6f}")

    # megayak for std match
    mega_p = None
    for cand in [
        ROOT / "external" / "megayak_s6e11-target-first-lm" / "lm_oof.npy",
        PRED.parent / "external" / "banks" / "megayak_s6e11-target-first-lm" / "lm_oof.npy",
        Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/banks/megayak_s6e11-target-first-lm/lm_oof.npy"),
    ]:
        if cand.exists():
            mega_p = cand
            break
    mega = np.load(mega_p).astype(np.float32).ravel() if mega_p else unitize(l7_o)
    l7_std = std_match_fold(l7_o, mega, folds)
    # PCHIP lock test to OOF law fold-aggregated
    l7_t_lock = pchip_map(l7_t, l7_std)
    disp = float(l7_t_lock.std() / (l7_std.std() + 1e-12))
    ks = float(ks_2samp(l7_std, l7_t_lock).statistic)
    log(f"PCHIP lock disp={disp:.4f} KS={ks:.6f}")
    np.save(OUT / "L7_oof_5f_std.npy", l7_std)
    np.save(OUT / "L7_test_5f_std.npy", l7_t_lock)

    z_ref = None
    for cand in [
        PRED / "gm_v5_overnight" / "z_ref_old_oof.npy",
        ROOT / "preds" / "gm_v5_overnight" / "z_ref_old_oof.npy",
        Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/gm_v5_overnight/z_ref_old_oof.npy"),
    ]:
        if cand.exists():
            z_ref = np.load(cand).astype(np.float32).ravel()
            break
    if z_ref is None:
        raise FileNotFoundError("z_ref_old_oof.npy")
    base_margin = calibrate_logit_to_prevalence(z_ref)
    log(f"base_margin mean_p={float(expit(base_margin).mean()):.7f}")

    # Feature pack (best-effort; missing cols skipped)
    def load_opt(paths, n):
        for p in paths:
            if Path(p).exists():
                a = np.load(p).astype(np.float32).ravel()
                if a.shape[0] == n:
                    return unitize(a)
        return None

    n = len(y)
    feats = {}
    feats["z_L7_5f_bayes"] = unitize(l7_o)
    feats["z_L7_5f_std"] = unitize(l7_std)
    mega_u = unitize(mega[:n])
    feats["z_megayak"] = mega_u
    feats["z_L7_minus_mega"] = unitize(l7_std.astype(np.float64) - mega[:n].astype(np.float64))

    gpt = load_opt(
        [
            PRED / "gm_breakout" / "u_gpt2m_lm.npy",
            ROOT / "preds" / "gm_breakout" / "u_gpt2m_lm.npy",
            "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/gm_breakout/u_gpt2m_lm.npy",
        ],
        n,
    )
    if gpt is not None:
        feats["z_gpt2_med_5f"] = gpt
        feats["z_L7_minus_gpt2m"] = unitize(l7_std.astype(np.float64) - gpt.astype(np.float64))

    # optional KMAT / P4 — skip if absent
    for name, rel in [
        ("z_w12", "gm_breakout/z_areeb.npy"),
    ]:
        v = load_opt([PRED / rel, ROOT / "preds" / rel, f"/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/{rel}"], n)
        if v is not None:
            feats[name] = v
            feats["z_L7_minus_w12"] = unitize(l7_std.astype(np.float64) - v.astype(np.float64))

    cols = sorted(feats.keys())
    X = np.stack([feats[c] for c in cols], axis=1).astype(np.float32)
    log(f"feature matrix {X.shape} cols={cols}")

    # Train XGB + LGB 5 seeds
    oof_x = np.zeros(n, np.float64)
    oof_l = np.zeros(n, np.float64)
    seeds = [0, 1, 2, 3, 4]
    use_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "0") != ""

    try:
        import xgboost as xgb
    except ImportError:
        xgb = None
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None

    for seed in seeds:
        px = np.zeros(n, np.float64)
        pl = np.zeros(n, np.float64)
        for tr, te in folds:
            if xgb is not None:
                dx = xgb.DMatrix(X[tr], label=y[tr], base_margin=base_margin[tr])
                dv = xgb.DMatrix(X[te], label=y[te], base_margin=base_margin[te])
                params = dict(
                    max_depth=3,
                    eta=0.02,
                    subsample=0.8,
                    colsample_bynode=0.35,
                    min_child_weight=180,
                    max_delta_step=0.15,
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    device="cuda" if use_cuda else "cpu",
                    seed=seed,
                )
                bst = xgb.train(params, dx, num_boost_round=400, evals=[(dv, "v")], verbose_eval=False)
                px[te] = bst.predict(dv)
            if lgb is not None:
                dtr = lgb.Dataset(X[tr], label=y[tr], init_score=base_margin[tr])
                params = dict(
                    objective="binary",
                    metric="auc",
                    learning_rate=0.02,
                    num_leaves=6,
                    feature_fraction=0.35,
                    bagging_fraction=0.8,
                    bagging_freq=1,
                    min_sum_hessian_in_leaf=50.0,
                    extra_trees=True,
                    verbosity=-1,
                    seed=seed,
                    device_type="gpu" if use_cuda else "cpu",
                )
                try:
                    booster = lgb.train(params, dtr, num_boost_round=400)
                except Exception:
                    params["device_type"] = "cpu"
                    booster = lgb.train(params, dtr, num_boost_round=400)
                pl[te] = booster.predict(X[te])
        oof_x += px
        oof_l += pl
        log(f"seed {seed}: xgb_auc={roc_auc_score(y, px):.6f} lgb_auc={roc_auc_score(y, pl):.6f}")
    oof_x /= len(seeds)
    oof_l /= len(seeds)
    blend = 0.5 * (oof_x + oof_l)
    # residual vs base
    d1 = unitize(logit(np.clip(blend, 1e-6, 1 - 1e-6)) - base_margin)
    np.save(OUT / "xgb_stageB_L7_5f_oof.npy", oof_x.astype(np.float32))
    np.save(OUT / "lgb_stageB_L7_5f_oof.npy", oof_l.astype(np.float32))
    np.save(OUT / "d1_stageB_L7_5f_oof.npy", d1)

    # 5/5 gate vs z_ref alone and vs z_ref+old d1 if present
    base_auc = np.array([roc_auc_score(y[te], z_ref[te]) for _, te in folds])
    new_auc = np.array([roc_auc_score(y[te], z_ref[te] + 0.08 * d1[te]) for _, te in folds])
    d = (new_auc - base_auc) * 1e5
    pos = int((d > 0).sum())
    log(f"StageB L7 residual vs z_ref: mean={d.mean():+.2f}e-5 pos={pos}/5 folds={d.tolist()}")
    meta = {
        "cols": cols,
        "disp": disp,
        "ks": ks,
        "d_e5": d.tolist(),
        "pos": pos,
        "oof_auc_blend": float(roc_auc_score(y, blend)),
        "elapsed_s": time.time() - T0,
    }
    (OUT / "stageB_l7_5f_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"DONE meta={meta}")


if __name__ == "__main__":
    main()
