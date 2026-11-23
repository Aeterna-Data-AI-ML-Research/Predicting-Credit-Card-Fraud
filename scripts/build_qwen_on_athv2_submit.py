#!/usr/bin/env python3
"""Blend Qwen Fold-0 R_mean_diff residual onto ATH_v2 (0.95550) and write CSV."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

SSD = Path("/tmp/s6e11_v39_ssd")
REPO = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
ATH_V2 = REPO / "submissions/sub_v39_S4_gemini_uid_entity_a004.csv"


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / max(len(v), 1)


def L(v):
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6))


def main():
    ll_va = np.load(SSD / "out/qwen15b_f0_ll.npy")
    ll_te = np.load(SSD / "out/qwen15b_f0_ll_test.npy")
    assert ll_va.ndim == 3 and ll_va.shape[2] == 2
    r_va = (ll_va[:, :, 1] - ll_va[:, :, 0]).mean(1)
    r_te = (ll_te[:, :, 1] - ll_te[:, :, 0]).mean(1)

    y = pd.read_csv(SSD / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(SSD / "folds/folds.csv")["fold"].to_numpy(int)
    va = folds == 0
    y0 = y[va]
    assert len(y0) == len(r_va)

    ath_v2 = pd.read_csv(ATH_V2)
    # For OOF alpha pick: use z_fullbank+uid as proxy for ATH_v2 on fold0
    z_fb = np.load(SSD / "preds/z_fullbank_oof.npy").ravel()
    lgb = np.load(SSD / "out/uid_entity_oof.npy")
    # reconstruct approx ATH_v2 OOF on fold0
    z_ath = L(z_fb)
    b = np.cov(logit(np.clip(lgb, 1e-7, 1 - 1e-7)), z_ath)[0, 1] / np.var(z_ath)
    res_uid = logit(np.clip(lgb, 1e-7, 1 - 1e-7)) - b * z_ath
    res_uid = (res_uid - res_uid.mean()) / (res_uid.std() + 1e-8)
    z_v2 = z_ath + 0.04 * res_uid
    z_v2_0 = z_v2[va]
    u_r = (L(r_va) - L(r_va).mean()) / (L(r_va).std() + 1e-8)

    # orth residual to base on fold0
    beta = np.cov(u_r, z_v2_0)[0, 1] / np.var(z_v2_0)
    res0 = u_r - beta * ((z_v2_0 - z_v2_0.mean()) / (z_v2_0.std() + 1e-8))
    res0 = (res0 - res0.mean()) / (res0.std() + 1e-8)

    best_a, best_auc = 0.03, -1.0
    base = float(roc_auc_score(y0, expit(z_v2_0)))
    for a in np.linspace(-0.06, 0.08, 29):
        auc = float(roc_auc_score(y0, expit(z_v2_0 + a * res0)))
        if auc > best_auc:
            best_auc, best_a = auc, float(a)
    print(json.dumps({"solo_r": float(roc_auc_score(y0, r_va)), "base_v2": base, "best_a": best_a, "best_auc": best_auc, "lift": best_auc - base}, indent=2))

    # test residual: rank-logit then unit, project using train beta onto ATH_v2 test
    z_te = logit(np.clip(ath_v2.iloc[:, 1].to_numpy(float), 1e-7, 1 - 1e-7))
    u_te = L(r_te)
    u_te = (u_te - u_te.mean()) / (u_te.std() + 1e-8)
    z_te_u = (z_te - z_te.mean()) / (z_te.std() + 1e-8)
    # use fold0 beta in standardized space approx
    beta_t = np.cov(u_te, z_te_u)[0, 1] / (np.var(z_te_u) + 1e-12)
    res_t = u_te - beta_t * z_te_u
    res_t = (res_t - res_t.mean()) / (res_t.std() + 1e-8)

    # LB-calibrated cap: uid peak was ~0.04; use min(|best_a|, 0.04) with sign
    a_sub = float(np.clip(best_a, -0.04, 0.04))
    if abs(best_a) < 0.005:
        a_sub = 0.02 * np.sign(best_a) if best_a != 0 else 0.02
    p = expit(z_te + a_sub * res_t)
    out = ath_v2.copy()
    out.iloc[:, 1] = p
    path = REPO / f"submissions/sub_v39_qwen_on_ATHv2_a{a_sub:+.3f}.csv".replace("+", "")
    # cleaner name
    path = REPO / f"submissions/sub_v39_qwen_on_ATHv2_a{a_sub:.3f}.csv"
    out.to_csv(path, index=False)
    meta = {"a_sub": a_sub, "best_a": best_a, "best_auc": best_auc, "base": base, "path": str(path)}
    (SSD / "out/qwen_athv2_submit_meta.json").write_text(json.dumps(meta, indent=2))
    print(path)
    return path


if __name__ == "__main__":
    main()
