#!/usr/bin/env python3
"""Hard VA gate (NO_SUBMIT): solo_f0 < 0.9485 OR fb subfold lift < +0.5e-4 → FAIL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SSD = Path("/tmp/s6e11_v39_ssd")
SOLO_MIN = 0.9485
LIFT_MIN = 0.50e-4
NO_SUBMIT = True


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


def subfold_lift(y, z, u, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    d = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        best_w, best_a = 0.0, -1.0
        for w in np.linspace(-0.06, 0.06, 25):
            a = auc(y[tr], L(z[tr]) + w * u[tr])
            if a > best_a:
                best_a, best_w = a, float(w)
        d.append(auc(y[va], L(z[va]) + best_w * u[va]) - auc(y[va], z[va]))
    d = np.asarray(d)
    return float(d.mean()), int((d > 0).sum()), d.tolist()


def main():
    ll_path = SSD / "out/qwen15b_f0_ll.npy"
    if not ll_path.exists():
        ll_path = SSD / "out/qwen15b_f0_ll_va.npy"
    if not ll_path.exists():
        print(json.dumps({"ok": False, "reason": "missing_ll", "NO_SUBMIT": NO_SUBMIT}))
        return 2

    y = pd.read_csv(SSD / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(SSD / "folds/folds.csv")["fold"].to_numpy(int)
    z_fb = np.load(SSD / "preds/z_fullbank_oof.npy").ravel()
    ll = np.load(ll_path)
    va = folds == 0
    y0, z0 = y[va], z_fb[va]
    assert len(y0) == ll.shape[0], (len(y0), ll.shape)

    r_md = (ll[:, :, 1] - ll[:, :, 0]).mean(1)
    s = U(r_md)
    solo = auc(y0, s)
    u = unit(L(s) - L(z0))
    mean_d, n_pos, d = subfold_lift(y0, z0, u)

    fail_solo = solo < SOLO_MIN
    fail_lift = mean_d < LIFT_MIN
    ok = not (fail_solo or fail_lift)
    report = {
        "ok": ok,
        "NO_SUBMIT": NO_SUBMIT,
        "solo_f0": solo,
        "solo_min": SOLO_MIN,
        "fail_solo": fail_solo,
        "fb_mean_d": mean_d,
        "lift_min": LIFT_MIN,
        "fail_lift": fail_lift,
        "fb_n_pos": n_pos,
        "d": d,
        "reduction": "R_mean_diff",
    }
    (SSD / "out/gate_va_hard_1420.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
