#!/usr/bin/env python3
"""V37 FIXED local audit — 5-fold LOFO orthogonal blend lift (NOT solo−ensemble).

NO_SUBMIT=True. Does not touch any RunPod.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
os.environ.setdefault("NO_SUBMIT", "True")

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "data" / "train.csv").exists():
    REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "logs" / "overnight" / "v37_local_audit"
PRED = REPO / "preds" / "v36_1"
L7 = REPO / "preds" / "l7_s2026_5f"
FOLDS_CSV = REPO / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv"
AMAN = (
    REPO
    / "external/v37_twin_check/vaults/amanatar_s6e11-sovereign-095423-simplex-and-oof-vault"
    / "s6e11_sovereign_095423_vault.npz"
)
BUSY_OWN = REPO / "external/v37_busyaprime_own"
BUSY_OUT = REPO / "external/v37_busyaprime_out"
W_GRID = np.linspace(-0.06, 0.06, 61)


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


def raw_disp(p_oof, p_te, z_oof, z_te):
    d_o = L(p_oof) - L(z_oof)
    d_t = L(p_te) - L(z_te)
    return float(d_t.std() / (d_o.std() + 1e-12))


def lofo_orth_blend_lift(p_oof, y, folds, z_base, B_oof=None, w_grid=W_GRID):
    """Signed 5-fold LOFO orthogonal blend lift.

    For each held-out fold k:
      - build u = unit(L(p)-L(z)) on all rows
      - if B_oof given: Ridge(α=50) orth of u vs B on folds!=k, apply to fold k
      - choose w* on folds!=k maximizing auc(L(z)+w*u_orth)
      - score ΔAUC on fold k only
    Returns mean ΔAUC, per-fold list, n_pos (>= +0.40e-4), best mean w sign.
    """
    p_oof = np.asarray(p_oof, float).ravel()
    z_base = np.asarray(z_base, float).ravel()
    u_raw = unit(L(p_oof) - L(z_base))
    d_folds, w_folds = [], []
    for k in range(5):
        tr = folds != k
        va = folds == k
        u = u_raw.copy()
        if B_oof is not None:
            Btr = np.column_stack([unit(B_oof[tr, j]) for j in range(B_oof.shape[1])])
            Bva = np.column_stack([unit(B_oof[va, j]) for j in range(B_oof.shape[1])])
            ridge = Ridge(alpha=50.0, fit_intercept=False)
            ridge.fit(Btr, u[tr])
            u_orth = u.copy()
            u_orth[tr] = u[tr] - ridge.predict(Btr)
            u_orth[va] = u[va] - ridge.predict(Bva)
        else:
            u_orth = u
        best_w, best_a = 0.0, -1.0
        for w in w_grid:
            a = auc(y[tr], L(z_base[tr]) + w * u_orth[tr])
            if a > best_a:
                best_a, best_w = a, float(w)
        d = auc(y[va], L(z_base[va]) + best_w * u_orth[va]) - auc(y[va], z_base[va])
        d_folds.append(float(d))
        w_folds.append(best_w)
    d_arr = np.asarray(d_folds)
    return {
        "mean_d_auc": float(d_arr.mean()),
        "d_folds": d_folds,
        "w_folds": w_folds,
        "pos_ge_0p40e4": int((d_arr >= 0.40e-4).sum()),
        "solo_oof": auc(y, p_oof),
        # KEEP for transparency — NOT the lift metric
        "solo_minus_z_solo_NOT_lift": float(auc(y, p_oof) - auc(y, z_base)),
    }


def family_avg_L(cols: list[np.ndarray]) -> np.ndarray:
    Lm = np.column_stack([L(c) for c in cols])
    return expit(Lm.mean(1))


def reducers_from_ll(ll: np.ndarray) -> dict[str, np.ndarray]:
    ll0, ll1 = ll[:, :, 0], ll[:, :, 1]
    r_mean = (ll1 - ll0).mean(1)
    r_lse = logsumexp(ll1, 1) - logsumexp(ll0, 1)
    r_rank = np.mean([L(ll1[:, k] - ll0[:, k]) for k in range(ll.shape[1])], 0)
    return {
        "R_mean_diff": expit(logit(np.clip(U(r_mean), 1e-6, 1 - 1e-6))),
        "R_deotte_lse": expit(logit(np.clip(U(r_lse), 1e-6, 1 - 1e-6))),
        "R_ranklogit": expit(logit(np.clip(U(r_rank), 1e-6, 1 - 1e-6))),
    }


def platt_calibrate_fold(oof_logits_tr, y_tr, logits_va):
    """Per-fold Platt on float64 logits → calibrated logits."""
    # map probs if needed — assume already logit-ish; convert via expit if looks like proba
    def to_logit(x):
        x = np.asarray(x, np.float64).ravel()
        if x.min() >= 0 and x.max() <= 1:
            return logit(np.clip(x, 1e-6, 1 - 1e-6))
        return x

    z_tr = to_logit(oof_logits_tr).reshape(-1, 1)
    z_va = to_logit(logits_va).reshape(-1, 1)
    lr = LogisticRegression(C=1e8, max_iter=500, solver="lbfgs")
    lr.fit(z_tr, y_tr)
    # calibrated probability → logit
    p_va = lr.predict_proba(z_va)[:, 1]
    return logit(np.clip(p_va, 1e-6, 1 - 1e-6))


def deotte_lr_stack(members_oof: dict[str, np.ndarray], members_te: dict[str, np.ndarray], y, folds, C=0.03):
    """Nested 5-fold LogisticRegression(C) on standardized Platt-calibrated logits."""
    names = list(members_oof.keys())
    n = len(y)
    oof_stack = np.zeros(n, np.float64)
    te_acc = []
    for k in range(5):
        tr, va = folds != k, folds == k
        # Platt each member on tr → apply to va and test
        X_tr, X_va, X_te = [], [], []
        for nm in names:
            o = members_oof[nm]
            t = members_te[nm]
            # calibrate using fold tr
            # get calibrated logits for all rows via model fit on tr
            def to_logit(x):
                x = np.asarray(x, np.float64).ravel()
                if np.nanmin(x) >= 0 and np.nanmax(x) <= 1:
                    return logit(np.clip(x, 1e-6, 1 - 1e-6))
                return x

            z = to_logit(o).reshape(-1, 1)
            zt = to_logit(t).reshape(-1, 1)
            lr = LogisticRegression(C=1e8, max_iter=500)
            lr.fit(z[tr], y[tr])
            X_tr.append(logit(np.clip(lr.predict_proba(z[tr])[:, 1], 1e-6, 1 - 1e-6)))
            X_va.append(logit(np.clip(lr.predict_proba(z[va])[:, 1], 1e-6, 1 - 1e-6)))
            X_te.append(logit(np.clip(lr.predict_proba(zt)[:, 1], 1e-6, 1 - 1e-6)))
        X_tr = np.column_stack(X_tr)
        X_va = np.column_stack(X_va)
        X_te = np.column_stack(X_te)
        # standardize by OOF-tr mean/std
        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-12
        X_tr_s = (X_tr - mu) / sd
        X_va_s = (X_va - mu) / sd
        X_te_s = (X_te - mu) / sd
        clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs")
        clf.fit(X_tr_s, y[tr])
        oof_stack[va] = clf.predict_proba(X_va_s)[:, 1]
        te_acc.append(clf.predict_proba(X_te_s)[:, 1])
    te_stack = np.mean(te_acc, axis=0)
    return oof_stack, te_stack


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[fixed] NO_SUBMIT={os.environ.get('NO_SUBMIT')}", flush=True)
    y = pd.read_csv(REPO / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS_CSV)["fold"].to_numpy(int)
    z_fb = np.load(PRED / "z_fullbank_oof.npy").ravel()[: len(y)]
    z_tw = np.load(PRED / "z_ATH_twin_oof.npy").ravel()[: len(y)]
    z_fb_te = np.load(PRED / "z_fullbank_test.npy").ravel()
    z_tw_te = np.load(PRED / "z_ATH_twin_test.npy").ravel()
    B_oof = np.load(PRED / "B_clean_oof.npy")
    B_te = np.load(PRED / "B_clean_test.npy")
    print(f"[fixed] z_fullbank solo={auc(y,z_fb):.5f} z_ATH_twin solo={auc(y,z_tw):.5f}", flush=True)

    report = {
        "gate_canary_read_only": {
            "blend_lift_correct": True,
            "formula": "auc(L(z)+w*u_orth)-auc(z) on held-out va_sf; w* on tr_sf; Ridge orth on tr_sf",
            "raw_disp_caveat": (
                "Pod TabICL scores 4 bags on va0 but only 2 bags on test → "
                "raw_disp = std(delta_te)/std(delta_va) can be inflated/deflated. "
                "Re-run matched-bag gate on Mac after npy sync BEFORE delete decision."
            ),
            "file": "scripts/v37_gpu/run_fold0_tfm_and_transductive_lm.py:gate_canary",
        },
        "amanatar": [],
        "busyaprime": [],
        "l7_bayes": [],
        "l7_contrast": {},
        "deotte_lrstack": {},
    }

    # ----- Amanatar 6 paradigm streams -----
    npz = np.load(AMAN, allow_pickle=True)
    paradigms = [
        ("P1_gpt2", "oof_P1_gpt2_5fold", "test_P1_gpt2_5fold"),
        ("P2_TriTabPFN", "oof_P2_tri_tabpfn", "test_P2_tri_tabpfn"),
        ("P3_gpdvae", "oof_P3_gpdvae_gen", "test_P3_gpdvae_gen"),
        ("P4_deep_neural_gam", "oof_P4_deep_neural_gam", "test_P4_deep_neural_gam"),
        ("P5_boosted_gbdt", "oof_P5_boosted_gbdt_v6", "test_P5_boosted_gbdt_v6"),
        ("P6_shallow_tree_markov", "oof_P6_shallow_tree_markov", "test_P6_shallow_tree_markov"),
    ]
    aman_members_oof, aman_members_te = {}, {}
    for tag, ok, tk in paradigms:
        if ok not in npz or tk not in npz:
            report["amanatar"].append({"name": tag, "error": "missing"})
            continue
        po, pt = npz[ok].astype(np.float64), npz[tk].astype(np.float64)
        aman_members_oof[tag] = po
        aman_members_te[tag] = pt
        lift_fb = lofo_orth_blend_lift(po, y, folds, z_fb, B_oof)
        lift_tw = lofo_orth_blend_lift(po, y, folds, z_tw, B_oof)
        disp = raw_disp(po, pt, z_fb, z_fb_te)
        rec = {
            "name": f"amanatar:{tag}",
            "solo_oof": lift_fb["solo_oof"],
            "raw_disp_fb": disp,
            "LOFO_blend_lift_fb": lift_fb["mean_d_auc"],
            "LOFO_blend_lift_tw": lift_tw["mean_d_auc"],
            "pos_fb": lift_fb["pos_ge_0p40e4"],
            "pos_tw": lift_tw["pos_ge_0p40e4"],
            "d_folds_fb": lift_fb["d_folds"],
            "d_folds_tw": lift_tw["d_folds"],
            "BUGGY_solo_minus_ensemble": lift_fb["solo_minus_z_solo_NOT_lift"],
        }
        report["amanatar"].append(rec)
        print(
            f"[aman] {tag} solo={rec['solo_oof']:.5f} LOFO_fb={rec['LOFO_blend_lift_fb']:+.2e} "
            f"LOFO_tw={rec['LOFO_blend_lift_tw']:+.2e} buggy_sub={rec['BUGGY_solo_minus_ensemble']:+.2e}",
            flush=True,
        )

    # ----- Busyaprime own vault + kernel stack -----
    oof_path = BUSY_OWN / "oof.npy"
    test_path = BUSY_OWN / "test.npy"
    man_path = BUSY_OWN / "manifest.csv"
    if oof_path.exists():
        oof = np.load(oof_path)  # expect (N, M) or similar
        te = np.load(test_path)
        print(f"[busy] oof={oof.shape} test={te.shape}", flush=True)
        if oof.ndim == 1:
            oof = oof.reshape(-1, 1)
            te = te.reshape(-1, 1)
        if oof.shape[0] != len(y) and oof.shape[1] == len(y):
            oof, te = oof.T, te.T
        names = None
        fam_col = None
        man = None
        if man_path.exists():
            man = pd.read_csv(man_path)
            print("[busy] manifest cols", list(man.columns), "rows", len(man), flush=True)
            for c in ("name", "model", "member", "id", "file"):
                if c in man.columns and len(man) == oof.shape[1]:
                    names = man[c].astype(str).tolist()
                    break
            if "family" in man.columns:
                fam_col = man["family"].astype(str).tolist()
        if names is None:
            names = [f"m{i}" for i in range(oof.shape[1])]

        # a priori architectural families (match name OR manifest family string)
        families = {
            "V_EBM": ["ebm", "interpretml", "explainable"],
            "V_TabM": ["tabm", "tab_m"],
            "V_RealMLP": ["realmlp", "real_mlp", "pytabkit"],
            "V_BPE_GBDT": ["bpe", "xgb", "lgb", "catboost", "gbdt", "lightgbm", "xgboost", "booster"],
            "V_JointGLM": ["jointglm", "joint_glm", "probit", "logistic_glm"],
        }

        def match_idxs(keys, fam_name):
            idxs = []
            for i, n in enumerate(names):
                blob = n.lower()
                fam_s = fam_col[i].lower() if fam_col is not None else ""
                blob_all = blob + " " + fam_s
                if fam_name == "V_JointGLM":
                    # pure GLM/probit — exclude tree boosters even if "GLM-start"
                    if any(x in blob_all for x in ("lgb", "xgb", "catboost", "booster", "gbdt")):
                        continue
                    if any(k in blob_all for k in keys) or ("glm" in fam_s and "booster" not in fam_s):
                        idxs.append(i)
                    continue
                if any(k in blob_all for k in keys):
                    idxs.append(i)
            return idxs

        for fam, keys in families.items():
            idxs = match_idxs(keys, fam)
            # dedicated TabM files
            if not idxs and fam == "V_TabM" and (BUSY_OWN / "tabm_oof.npy").exists():
                po = np.load(BUSY_OWN / "tabm_oof.npy").ravel()
                pt = np.load(BUSY_OWN / "tabm_test.npy").ravel()
                lift_fb = lofo_orth_blend_lift(po, y, folds, z_fb, B_oof)
                lift_tw = lofo_orth_blend_lift(po, y, folds, z_tw, B_oof)
                rec = {
                    "name": f"busyaprime:{fam}_tabm_file",
                    "n_members": 1,
                    "solo_oof": lift_fb["solo_oof"],
                    "raw_disp_fb": raw_disp(po, pt, z_fb, z_fb_te),
                    "LOFO_blend_lift_fb": lift_fb["mean_d_auc"],
                    "LOFO_blend_lift_tw": lift_tw["mean_d_auc"],
                    "pos_fb": lift_fb["pos_ge_0p40e4"],
                    "pos_tw": lift_tw["pos_ge_0p40e4"],
                    "BUGGY_solo_minus_ensemble": lift_fb["solo_minus_z_solo_NOT_lift"],
                }
                report["busyaprime"].append(rec)
                print(
                    f"[busy] {fam} solo={rec['solo_oof']:.5f} LOFO_fb={rec['LOFO_blend_lift_fb']:+.2e}",
                    flush=True,
                )
                continue
            if not idxs:
                report["busyaprime"].append(
                    {"name": f"busyaprime:{fam}", "reject": "empty_family", "names_sample": names[:8]}
                )
                continue
            po = family_avg_L([oof[:, i] for i in idxs])
            pt = family_avg_L([te[:, i] for i in idxs])
            lift_fb = lofo_orth_blend_lift(po, y, folds, z_fb, B_oof)
            lift_tw = lofo_orth_blend_lift(po, y, folds, z_tw, B_oof)
            rec = {
                "name": f"busyaprime:{fam}_avg_n{len(idxs)}",
                "n_members": len(idxs),
                "members": [names[i] for i in idxs],
                "solo_oof": lift_fb["solo_oof"],
                "raw_disp_fb": raw_disp(po, pt, z_fb, z_fb_te),
                "LOFO_blend_lift_fb": lift_fb["mean_d_auc"],
                "LOFO_blend_lift_tw": lift_tw["mean_d_auc"],
                "pos_fb": lift_fb["pos_ge_0p40e4"],
                "pos_tw": lift_tw["pos_ge_0p40e4"],
                "BUGGY_solo_minus_ensemble": lift_fb["solo_minus_z_solo_NOT_lift"],
            }
            report["busyaprime"].append(rec)
            print(
                f"[busy] {fam} n={len(idxs)} solo={rec['solo_oof']:.5f} LOFO_fb={rec['LOFO_blend_lift_fb']:+.2e} "
                f"buggy_sub={rec['BUGGY_solo_minus_ensemble']:+.2e}",
                flush=True,
            )
        # kernel nested stack oof if present
        sn = BUSY_OUT / "stack_nested_oof.npy"
        if sn.exists():
            po = np.load(sn).ravel()
            if len(po) == len(y):
                # test from submission_stack if exists
                pt = None
                for cand in [BUSY_OUT / "submission_stack.csv", BUSY_OUT / "submission.csv"]:
                    if cand.exists():
                        df = pd.read_csv(cand)
                        col = [c for c in df.columns if c != "id"][0]
                        pt = df[col].to_numpy(float)
                        break
                lift_fb = lofo_orth_blend_lift(po, y, folds, z_fb, B_oof)
                lift_tw = lofo_orth_blend_lift(po, y, folds, z_tw, B_oof)
                rec = {
                    "name": "busyaprime:kernel_stack_nested_oof",
                    "solo_oof": lift_fb["solo_oof"],
                    "raw_disp_fb": raw_disp(po, pt, z_fb, z_fb_te) if pt is not None else None,
                    "LOFO_blend_lift_fb": lift_fb["mean_d_auc"],
                    "LOFO_blend_lift_tw": lift_tw["mean_d_auc"],
                    "pos_fb": lift_fb["pos_ge_0p40e4"],
                    "pos_tw": lift_tw["pos_ge_0p40e4"],
                    "BUGGY_solo_minus_ensemble": lift_fb["solo_minus_z_solo_NOT_lift"],
                }
                report["busyaprime"].append(rec)
                print(f"[busy] kernel_stack solo={rec['solo_oof']:.5f} LOFO_fb={rec['LOFO_blend_lift_fb']:+.2e}", flush=True)
    else:
        report["busyaprime"].append({"error": f"missing {oof_path}"})

    # ----- L7 Bayes reducers -----
    ll_oof = np.load(L7 / "l7_s2026_oof_5fold_raw.npy")
    ll_te = np.load(L7 / "l7_s2026_test_5fold_raw.npy").mean(0)
    red_o = reducers_from_ll(ll_oof)
    red_t = reducers_from_ll(ll_te)
    for k in red_o:
        lift_fb = lofo_orth_blend_lift(red_o[k], y, folds, z_fb, B_oof)
        lift_tw = lofo_orth_blend_lift(red_o[k], y, folds, z_tw, B_oof)
        rec = {
            "name": f"l7_{k}",
            "solo_oof": lift_fb["solo_oof"],
            "raw_disp_fb": raw_disp(red_o[k], red_t[k], z_fb, z_fb_te),
            "LOFO_blend_lift_fb": lift_fb["mean_d_auc"],
            "LOFO_blend_lift_tw": lift_tw["mean_d_auc"],
            "pos_fb": lift_fb["pos_ge_0p40e4"],
            "pos_tw": lift_tw["pos_ge_0p40e4"],
            "BUGGY_solo_minus_ensemble": lift_fb["solo_minus_z_solo_NOT_lift"],
        }
        report["l7_bayes"].append(rec)
        print(
            f"[l7] {k} solo={rec['solo_oof']:.5f} LOFO_fb={rec['LOFO_blend_lift_fb']:+.2e} "
            f"buggy_sub={rec['BUGGY_solo_minus_ensemble']:+.2e}",
            flush=True,
        )
    # pure reduction contrast R_deotte_lse - R_mean_diff
    contrast = red_o["R_deotte_lse"] - red_o["R_mean_diff"]  # not a proba — use as residual feature via L of ranked?
    # Better: lift of lse vs mean as base? User asked (R_deotte_lse - R_mean_diff) pure reduction contrast
    # Evaluate LOFO blend of the contrast signal as additive feature on z_fb
    # Convert contrast to a score via U
    contrast_score = U(red_o["R_deotte_lse"] - red_o["R_mean_diff"])
    # For lift we need a "prediction" — use mean_diff as base and contrast as residual jewel
    # Direct: treat expit(L(mean)+c*unit(contrast)) — report LOFO of contrast vs z via treating contrast_score as p
    lift_c_fb = lofo_orth_blend_lift(contrast_score, y, folds, z_fb, B_oof)
    lift_c_tw = lofo_orth_blend_lift(contrast_score, y, folds, z_tw, B_oof)
    report["l7_contrast"] = {
        "name": "R_deotte_lse_minus_R_mean_diff",
        "corr_lse_mean": float(np.corrcoef(red_o["R_deotte_lse"], red_o["R_mean_diff"])[0, 1]),
        "LOFO_blend_lift_fb": lift_c_fb["mean_d_auc"],
        "LOFO_blend_lift_tw": lift_c_tw["mean_d_auc"],
        "pos_fb": lift_c_fb["pos_ge_0p40e4"],
        "pos_tw": lift_c_tw["pos_ge_0p40e4"],
        "solo_of_U_contrast": lift_c_fb["solo_oof"],
    }
    print(f"[l7] contrast LOFO_fb={lift_c_fb['mean_d_auc']:+.2e}", flush=True)

    # ----- Deotte C=0.03 LR stack -----
    members_oof = {}
    members_te = {}
    # B_clean columns as weak members? User said B_clean + amanatar + busyaprime fam + gpt2 + L7
    # Use amanatar paradigms + L7 R_deotte_lse + busy tabm if present + a few B_clean PCs via mean of top cols
    for tag, po in aman_members_oof.items():
        members_oof[f"aman_{tag}"] = po
        members_te[f"aman_{tag}"] = aman_members_te[tag]
    members_oof["l7_R_deotte_lse"] = red_o["R_deotte_lse"]
    members_te["l7_R_deotte_lse"] = red_t["R_deotte_lse"]
    members_oof["l7_R_mean_diff"] = red_o["R_mean_diff"]
    members_te["l7_R_mean_diff"] = red_t["R_mean_diff"]
    # busy family means already computed — re-add from report paths
    if (BUSY_OWN / "tabm_oof.npy").exists():
        members_oof["busy_tabm"] = np.load(BUSY_OWN / "tabm_oof.npy").ravel()
        members_te["busy_tabm"] = np.load(BUSY_OWN / "tabm_test.npy").ravel()
    # gpt2 med 4ep — assembled 5-fold or orth artifacts
    gdir = REPO / "preds/gpt2_med_4ep"
    if all((gdir / f"oof_{k}.npy").exists() for k in range(5)):
        po = np.zeros(len(y), np.float64)
        te_parts = []
        for k in range(5):
            po[folds == k] = np.load(gdir / f"oof_{k}.npy").ravel()
            tp = gdir / f"test_{k}.npy"
            if tp.exists():
                te_parts.append(np.load(tp).ravel())
        if te_parts and len(te_parts[0]) == len(z_fb_te):
            members_oof["gpt2_med_4ep"] = po
            members_te["gpt2_med_4ep"] = np.mean(te_parts, axis=0)
            print(f"[deotte] added gpt2_med_4ep solo={auc(y,po):.5f}", flush=True)
    elif (REPO / "preds/cb_native_ctr4_5f/H2_5F_gpt2med_orth_oof.npy").exists():
        po = np.load(REPO / "preds/cb_native_ctr4_5f/H2_5F_gpt2med_orth_oof.npy").ravel()
        pt = np.load(REPO / "preds/cb_native_ctr4_5f/H2_5F_gpt2med_orth_test.npy").ravel()
        if len(po) == len(y):
            members_oof["gpt2_med_4ep"] = po
            members_te["gpt2_med_4ep"] = pt
    # all B_clean columns
    for c in range(B_oof.shape[1]):
        members_oof[f"B{c}"] = B_oof[:, c]
        members_te[f"B{c}"] = B_te[:, c]

    stack_results = {}
    for C in (0.01, 0.03, 0.10):
        oof_s, te_s = deotte_lr_stack(members_oof, members_te, y, folds, C=C)
        solo = auc(y, oof_s)
        lift_fb = lofo_orth_blend_lift(oof_s, y, folds, z_fb, B_oof)
        lift_tw = lofo_orth_blend_lift(oof_s, y, folds, z_tw, B_oof)
        # ATH correlation / orth lift vs z_ATH (ATH twin is proxy for 0.95548 stack)
        corr_ath = float(np.corrcoef(U(oof_s), U(z_tw))[0, 1])
        stack_results[str(C)] = {
            "C": C,
            "solo_oof": solo,
            "vs_fullbank_solo_delta": solo - auc(y, z_fb),
            "LOFO_blend_lift_fb": lift_fb["mean_d_auc"],
            "LOFO_blend_lift_tw_ATH": lift_tw["mean_d_auc"],
            "pos_fb": lift_fb["pos_ge_0p40e4"],
            "pos_tw": lift_tw["pos_ge_0p40e4"],
            "corr_U_vs_ATH_twin": corr_ath,
            "n_members": len(members_oof),
        }
        print(
            f"[deotte C={C}] solo={solo:.5f} vs_fb={solo-auc(y,z_fb):+.2e} "
            f"LOFO_fb={lift_fb['mean_d_auc']:+.2e} LOFO_ATH={lift_tw['mean_d_auc']:+.2e} corr_ATH={corr_ath:.4f}",
            flush=True,
        )
        if C == 0.03:
            np.save(OUT / "z_deotte_lrstack_C003_oof.npy", oof_s.astype(np.float32))
            np.save(OUT / "z_deotte_lrstack_C003_test.npy", te_s.astype(np.float32))
    report["deotte_lrstack"] = stack_results
    report["deotte_members"] = list(members_oof.keys())

    (OUT / "V37_LOCAL_AUDIT_FIXED.json").write_text(json.dumps(report, indent=2))

    # markdown
    lines = [
        "# V37 Local Audit FIXED — LOFO Orthogonal Blend Lift (not solo−ensemble)",
        "",
        f"**NO_SUBMIT=True** · z_fullbank solo={auc(y,z_fb):.5f} · z_ATH_twin solo={auc(y,z_tw):.5f}",
        "",
        "## 1. Pod `gate_canary` read-only (Mac source)",
        "",
        "- **Blend lift math: CORRECT** in `scripts/v37_gpu/run_fold0_tfm_and_transductive_lm.py`:",
        "  `d = auc(y[va_sf], L(z)+w*u_orth) - auc(y[va_sf], z)` with `w*` fit on `tr_sf` only, Ridge orth on `tr_sf`.",
        "- **raw_disp caveat:** TabICL uses **4 bags on va0** but only **2 bags on test** → `raw_disp` can mis-scale.",
        "  After sync, re-run **matched-bag** Fold-0 gate on Mac **before** deleting `khx6qxz80yfenn`.",
        "",
        "## 2. Busyaprime = KERNEL (not dataset)",
        "",
        "- Pulled via `kaggle kernels pull/output`.",
        "- `dataset_sources` include `busyaprime/s6e11-33-own-members` (`oof.npy` / `test.npy`).",
        "",
        "## 3. Amanatar 6-paradigm LOFO blend lifts",
        "",
        "| Stream | solo | LOFO ΔAUC fb | LOFO ΔAUC ATH | pos_fb | **BUGGY solo−z** |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in report["amanatar"]:
        if "error" in r:
            lines.append(f"| {r['name']} | ERR | | | | |")
            continue
        lines.append(
            f"| {r['name']} | {r['solo_oof']:.5f} | {r['LOFO_blend_lift_fb']:+.2e} | "
            f"{r['LOFO_blend_lift_tw']:+.2e} | {r['pos_fb']}/5 | {r['BUGGY_solo_minus_ensemble']:+.2e} |"
        )
    lines += ["", "## 4. Busyaprime family averages (a priori)", ""]
    lines.append("| Family | solo | LOFO fb | LOFO ATH | pos | buggy_sub |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in report["busyaprime"]:
        if "solo_oof" not in r:
            lines.append(f"| {r.get('name', r)} | — | reject/empty | | | |")
            continue
        lines.append(
            f"| {r['name']} | {r['solo_oof']:.5f} | {r['LOFO_blend_lift_fb']:+.2e} | "
            f"{r['LOFO_blend_lift_tw']:+.2e} | {r['pos_fb']}/5 | {r['BUGGY_solo_minus_ensemble']:+.2e} |"
        )
    lines += ["", "## 5. L7 Bayes reducers (honest LOFO)", ""]
    lines.append("| Reducer | solo | LOFO fb | LOFO ATH | buggy_sub |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in report["l7_bayes"]:
        lines.append(
            f"| {r['name']} | {r['solo_oof']:.5f} | {r['LOFO_blend_lift_fb']:+.2e} | "
            f"{r['LOFO_blend_lift_tw']:+.2e} | {r['BUGGY_solo_minus_ensemble']:+.2e} |"
        )
    c = report["l7_contrast"]
    lines += [
        "",
        f"**Contrast** `(R_deotte_lse − R_mean_diff)`: corr(lse,mean)={c['corr_lse_mean']:.6f}; "
        f"LOFO_fb={c['LOFO_blend_lift_fb']:+.2e}; LOFO_ATH={c['LOFO_blend_lift_tw']:+.2e}",
        "",
        "## 6. Deotte `LogisticRegression(C)` meta-stacker",
        "",
        "| C | solo OOF | vs z_fullbank solo | LOFO fb | LOFO vs ATH twin | corr U↔ATH |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for C, r in stack_results.items():
        lines.append(
            f"| {C} | {r['solo_oof']:.5f} | {r['vs_fullbank_solo_delta']:+.2e} | "
            f"{r['LOFO_blend_lift_fb']:+.2e} | {r['LOFO_blend_lift_tw_ATH']:+.2e} | {r['corr_U_vs_ATH_twin']:.4f} |"
        )
    lines += [
        "",
        f"Members ({len(members_oof)}): `{', '.join(members_oof.keys())}`",
        "",
        "## Pod status (read-only)",
        "",
        "- `khx6qxz80yfenn` left **undisturbed** during this audit.",
        "",
    ]
    (OUT / "V37_LOCAL_AUDIT_FIXED.md").write_text("\n".join(lines) + "\n")
    print(f"[fixed] wrote {OUT/'V37_LOCAL_AUDIT_FIXED.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
