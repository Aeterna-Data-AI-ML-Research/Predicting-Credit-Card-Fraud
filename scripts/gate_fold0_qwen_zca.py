#!/usr/bin/env python3
"""V39 Fold-0 ZCA gate for Qwen2.5-1.5B storyteller LL reductions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SSD = Path("/tmp/s6e11_v39_ssd")
ATH_LB = 0.95548


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / max(len(v), 1)


def L(v):
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6))


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def auc(y, p):
    return float(roc_auc_score(y, np.asarray(p, float).ravel()))


def reductions(ll):
    """ll (N,K,2) → three score vectors."""
    # R_mean_diff
    r_md = (ll[:, :, 1] - ll[:, :, 0]).mean(1)
    # R_deotte_lse: logsumexp style soft
    # soft: mean of logsigmoid(ll1-ll0) across orders? use mean LLR then sigmoid
    llr = ll[:, :, 1] - ll[:, :, 0]
    r_deotte = expit(llr).mean(1)  # Deotte-ish bag mean of P(y=1|order)
    # R_ranklogit
    r_rl = expit(L(r_md))
    return {
        "R_mean_diff": U(r_md),
        "R_deotte_lse": U(r_deotte),
        "R_ranklogit": r_rl,
    }


def r2_b40(delta, B):
    d = unit(delta)
    X = np.column_stack([unit(B[:, j]) for j in range(B.shape[1])])
    Xc = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    B40 = Xc @ vt[: min(40, X.shape[1])].T
    ridge = Ridge(alpha=50.0, fit_intercept=False)
    ridge.fit(B40, d)
    pred = ridge.predict(B40)
    return float(1.0 - np.sum((d - pred) ** 2) / (np.sum((d - d.mean()) ** 2) + 1e-12))


def subfold_lift(y, z, u, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    d = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        # pick w on tr
        best_w, best_a = 0.0, -1.0
        for w in np.linspace(-0.06, 0.06, 25):
            a = auc(y[tr], L(z[tr]) + w * u[tr])
            if a > best_a:
                best_a, best_w = a, float(w)
        d.append(auc(y[va], L(z[va]) + best_w * u[va]) - auc(y[va], z[va]))
    d = np.asarray(d)
    return {"mean_d": float(d.mean()), "n_pos": int((d > 0).sum()), "d": d.tolist()}


def main():
    y = pd.read_csv(SSD / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(SSD / "folds/folds.csv")["fold"].to_numpy(int)
    z_fb = np.load(SSD / "preds/z_fullbank_oof.npy").ravel()
    z_tw = np.load(SSD / "preds/z_ATH_twin_oof.npy").ravel()
    B = np.load(SSD / "preds/B_clean_oof.npy")
    ll_path = SSD / "out/qwen15b_f0_ll.npy"
    if not ll_path.exists():
        # try alternate
        ll_path = SSD / "out/qwen15b_f0_ll_va.npy"
    ll = np.load(ll_path)
    assert ll.ndim == 3 and ll.shape[2] == 2, ll.shape
    va = folds == 0
    y0, z0, zt0, B0 = y[va], z_fb[va], z_tw[va], B[va]
    assert len(y0) == ll.shape[0], (len(y0), ll.shape)

    reds = reductions(ll)
    report = {"n_va": int(ll.shape[0]), "K": int(ll.shape[1]), "reductions": {}}
    promote_any = False
    for name, s in reds.items():
        solo = auc(y0, s)
        # delta vs fullbank for R2
        r2 = r2_b40(L(s) - L(z0), B0)
        u = unit(L(s) - L(z0))
        g_fb = subfold_lift(y0, z0, u)
        g_tw = subfold_lift(y0, zt0, u)
        ok = (
            solo >= 0.9490
            and r2 < 0.78
            and g_fb["n_pos"] >= 4
            and g_tw["n_pos"] >= 4
            and g_fb["mean_d"] >= 0.50e-4
            and g_tw["mean_d"] >= 0.50e-4
        )
        report["reductions"][name] = {
            "solo_f0": solo,
            "r2_B40": r2,
            "g_fb": g_fb,
            "g_tw": g_tw,
            "promote": ok,
        }
        print(
            f"{name}: solo={solo:.5f} R2={r2:.3f} "
            f"fbΔ={g_fb['mean_d']:+.2e} ({g_fb['n_pos']}/5) "
            f"twΔ={g_tw['mean_d']:+.2e} ({g_tw['n_pos']}/5) promote={ok}",
            flush=True,
        )
        promote_any = promote_any or ok
    report["promote_any"] = promote_any
    (SSD / "out/fold0_gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"promote_any": promote_any}, indent=2), flush=True)
    return 0 if promote_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
