#!/usr/bin/env python3
"""Strict CROSS_FOLD adjudication per Codex Round6 protocol."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
SOV = ROOT / "preds/v47_sovereign"
OUT = ROOT / "logs/v47/codex_loop"
FIXED_W = -0.10

def rank_logit(s):
    return logit((rankdata(s) - 0.5) / len(s))

def main():
    y = pd.read_csv(ROOT/"data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy()
    folds = np.load(SOV/"folds_skf42.npy")
    z0 = np.load(SOV/"z_sovereign_v0_oof.npy").ravel()
    cb = np.load(SOV/"catboost_outer_oof.npy").ravel()
    # rebuild candidate with frozen w within each fold
    cand = np.empty_like(z0)
    deltas = []
    ns = []
    for k in range(5):
        va = folds == k
        a = rank_logit(z0[va]); zc = rank_logit(cb[va])
        c = a + FIXED_W * (zc - a)
        cand[va] = c
        d = roc_auc_score(y[va], c) - roc_auc_score(y[va], z0[va])
        deltas.append(float(d)); ns.append(int(va.sum()))
        print(f"fold{k} Δ={d*1e5:+.4f}e-5 n={ns[-1]}")

    # Reproduce F0
    f0_delta = deltas[0]
    f0_ok = abs((roc_auc_score(y[folds==0], cand[folds==0]) - 0.9512507991341679)*1e5 - (roc_auc_score(y[folds==0], cand[folds==0]) - roc_auc_score(y[folds==0], z0[folds==0]))*1e5) < 1e-6
    # actually check cand F0 auc near 0.951336 if CB matches prior; may differ if CB retrain seed path differs
    f0_auc_cand = float(roc_auc_score(y[folds==0], cand[folds==0]))
    f0_auc_base = float(roc_auc_score(y[folds==0], z0[folds==0]))

    eval_idx = [1,2,3,4]
    wmean = sum(deltas[k]*ns[k] for k in eval_idx) / sum(ns[k] for k in eval_idx)
    pos = sum(1 for k in eval_idx if deltas[k] > 0)
    loo = []
    for leave in eval_idx:
        keep = [k for k in eval_idx if k != leave]
        m = sum(deltas[k]*ns[k] for k in keep) / sum(ns[k] for k in keep)
        loo.append({"leave_fold": leave, "weighted_mean_delta": m, "positive": m > 0})

    # Stratified within-fold bootstrap of weighted mean delta
    rng = np.random.default_rng(42)
    B = 2000
    boots = []
    fold_data = []
    for k in eval_idx:
        va = np.flatnonzero(folds == k)
        fold_data.append((y[va], z0[va], cand[va], ns[k]))
    for _ in range(B):
        ds = []
        for yk, bk, ck, n in fold_data:
            # within-class resample
            pos_i = np.flatnonzero(yk == 1); neg_i = np.flatnonzero(yk == 0)
            if len(pos_i)==0 or len(neg_i)==0:
                continue
            pi = rng.choice(pos_i, size=len(pos_i), replace=True)
            ni = rng.choice(neg_i, size=len(neg_i), replace=True)
            idx = np.concatenate([pi, ni])
            d = roc_auc_score(yk[idx], ck[idx]) - roc_auc_score(yk[idx], bk[idx])
            ds.append((d, n))
        if len(ds) < 4:
            continue
        boots.append(sum(d*n for d,n in ds)/sum(n for d,n in ds))
    boots = np.asarray(boots, float)
    ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))] if len(boots) else [None, None]

    # SKF provenance
    rebuilt = np.empty(len(y), dtype=np.int8)
    Xdummy = np.zeros(len(y))
    for k,(_,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(Xdummy,y)):
        rebuilt[va]=k
    folds_ok = bool(np.array_equal(folds, rebuilt))

    criteria = {
        "folds_skf42_ok": folds_ok,
        "weighted_mean_delta_e5": wmean * 1e5,
        "weighted_mean_ge_8e5": wmean >= 8e-5,
        "pos_eval_folds": pos,
        "pos_ge_3_of_4": pos >= 3,
        "loo_all_positive": all(x["positive"] for x in loo),
        "bootstrap_ci95": ci,
        "bootstrap_lb_gt_0": ci[0] is not None and ci[0] > 0,
        "f0_base_auc": f0_auc_base,
        "f0_cand_auc": f0_auc_cand,
        "f0_delta_e5": (f0_auc_cand - f0_auc_base) * 1e5,
    }
    numeric_pass = (
        criteria["weighted_mean_ge_8e5"]
        and criteria["pos_ge_3_of_4"]
        and criteria["loo_all_positive"]
        and criteria["bootstrap_lb_gt_0"]
        and criteria["folds_skf42_ok"]
    )
    # Champion isolation unresolved by default until audited
    champion_isolation_resolved = False
    if numeric_pass and champion_isolation_resolved:
        milestone = "CROSS_FOLD_VERIFIED"
        verdict = "CROSS_FOLD_VERIFIED"
    elif numeric_pass and not champion_isolation_resolved:
        milestone = "LOCAL_GATE_PASS"
        verdict = "MATCHED_ANCHOR_TRANSFER_EVIDENCE — CROSS_FOLD_VERIFIED pending champion isolation audit"
    else:
        milestone = "LOCAL_GATE_PASS"
        verdict = "CROSS_FOLD_FAIL — remain at LOCAL_GATE_PASS"

    out = {
        "verdict": verdict,
        "milestone_achieved": milestone,
        "numeric_pass": numeric_pass,
        "champion_isolation_resolved": champion_isolation_resolved,
        "fixed_w": FIXED_W,
        "per_fold_delta_e5": {str(i): deltas[i]*1e5 for i in range(5)},
        "loo": loo,
        "criteria": criteria,
        "NO_SUBMIT": True,
    }
    path = OUT/"ROUND6_CROSSFOLD_ADJUDICATION.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    # append short verdict
    rep = ROOT/"logs/v47/CODEX_DS_REVIEW_NEXT.md"
    with rep.open("a") as f:
        f.write("\n## Round6 strict adjudication\n\n")
        f.write(f"Verdict: **{verdict}**\n\n")
        f.write(f"Weighted mean Δe-5 (folds1-4) = {wmean*1e5:+.4f}; pos={pos}/4; CI95={ci}; numeric_pass={numeric_pass}.\n")
        f.write(f"Per-fold Δe-5: {[round(d*1e5,4) for d in deltas]}\n")
    print("wrote", path)

if __name__ == "__main__":
    main()
