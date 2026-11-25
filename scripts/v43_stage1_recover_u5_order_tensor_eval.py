#!/usr/bin/env python3
"""V43 Stage-1 — Recover U5..U10 + 3D order-tensor features; eval on z_modern_pub+Zr.

STRICT NO_SUBMIT=True. $0 local CPU. Zero pods.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
assert NO_SUBMIT
T0 = time.time()
N_TR, N_TE = 699_960, 299_984
ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
SSD = Path("/tmp/s6e11_v39_ssd")
DATA = ROOT / "data" if (ROOT / "data/train.csv").exists() else SSD / "data"
FOLDS_P = SSD / "folds/folds.csv" if (SSD / "folds/folds.csv").exists() else ROOT / "folds/folds.csv"
OUT = ROOT / "out/v43_stage1"
PRED_OUT = ROOT / "preds/v43_order_tensor"
LOG = ROOT / "logs/v43"
FINAL = ROOT / "out/v41_final_10slots"
MOD = ROOT / "out/v42_1_modern"

FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]


def log(m):
    print(f"[{time.time()-T0:5.1f}s] {m}", flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v):
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12))


def L_fold(v, folds):
    v = np.asarray(v, float).ravel()
    out = np.zeros_like(v, dtype=np.float64)
    for k in np.unique(folds):
        m = folds == k
        out[m] = L(v[m])
    return out


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def col_z(X):
    X = np.asarray(X, float)
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-12)


def gini10(Z):
    K = Z.shape[-1]
    s = np.zeros(Z.shape[:-1], dtype=np.float64)
    for i, j in combinations(range(K), 2):
        s += np.abs(Z[..., i] - Z[..., j])
    return s / (K * (K - 1) / 2.0)


def z_goodpjw(df: pd.DataFrame) -> np.ndarray:
    clean = (df["cvv_retry_count"].to_numpy() == 0) & (df["prior_disputes"].to_numpy() == 0)
    for f in FLAGS:
        clean &= df[f].to_numpy().astype(int) == 0
    risk, vel = df["merchant_risk_score"].to_numpy(), df["velocity_score"].to_numpy()
    return (
        (risk < 13.1)
        | ((vel <= 4.1) & (risk <= 20.0))
        | (clean & (((vel <= 4.1) & (risk <= 33.8)) | ((vel <= 6.0) & (risk <= 25.0))))
    )


def apply_zr(scores, zones, u_ref):
    out = L(scores).copy()
    uref = U(u_ref)
    if zones.any() and (~zones).any():
        floor = float(np.min(out[~zones])) - 1.0
        out[zones] = floor + uref[zones]
    return U(out)


def method_b_cont(oof, test, folds):
    o = unit(L_fold(oof, folds))
    t = unit(L(test))
    t = t * (float(o.std()) / (float(t.std()) + 1e-12))
    return o, t, float(np.std(t) / (np.std(o) + 1e-12)), float(ks_2samp(o, t).statistic)


def u_bank_from_llr(ll_pos, ll_neg):
    """U5..U10 from (n, K) pos/neg logits — exact V19.12 recipe."""
    llr = ll_pos - ll_neg
    Z = col_z(llr)
    K = Z.shape[1]
    h = max(K // 2, 1)
    if K >= 2 * h:
        pair_means = np.column_stack([0.5 * (Z[:, m] + Z[:, m + h]) for m in range(h)])
        pair_gaps = np.column_stack([np.abs(Z[:, m] - Z[:, m + h]) for m in range(h)])
    else:
        pair_means = Z
        pair_gaps = np.abs(Z - Z.mean(1, keepdims=True))
    Zp, Zn = col_z(ll_pos), col_z(ll_neg)
    part = np.partition(Z, [min(1, K - 1), max(K - 2, 0)], axis=1)
    top2 = 0.5 * (part[:, -1] + part[:, -2]) if K >= 2 else part[:, -1]
    bot2 = 0.5 * (part[:, 0] + part[:, 1]) if K >= 2 else part[:, 0]
    return {
        "U5_gini10_llr": gini10(Z),
        "U6_pair_std5_llr": pair_means.std(1),
        "U7_fwd_rev_gap": pair_gaps.mean(1),
        "U8_pos_minus_neg_std": Zp.std(1) - Zn.std(1),
        "U9_top2_bot2": top2 - bot2,
        "U10_pair_gap_max": pair_gaps.max(1),
    }


def order_curvature_7(ll_pos, ll_neg):
    """7 order-curvature / epistemic dispersion features from (n,K) logits.

    Assumes first half orders = forward, second half = reverse (antithetic).
    """
    llr = ll_pos - ll_neg  # (n, K)
    K = llr.shape[1]
    h = K // 2
    # R2 bayes LSE over orders (logsumexp of y1 - y0 paths ≈ softmax-mean)
    # use llr as order logits for bayes aggregation
    R2 = logsumexp(llr, axis=1) - np.log(K)
    # R3 rank-logit mean of per-order LLR
    R3 = np.mean(np.stack([L(llr[:, k]) for k in range(K)], 0), 0)
    # curvature
    R2_m_R3 = R2 - R3
    if h >= 1 and K >= 2 * h:
        R5 = np.mean(np.abs(llr[:, :h] - llr[:, h : 2 * h]), axis=1)
    else:
        R5 = np.std(llr, axis=1)
    order_std_y1 = ll_pos.std(1)
    order_std_llr = llr.std(1)
    order_minmax = llr.max(1) - llr.min(1)
    return {
        "R2_bayes_lse": R2,
        "R3_ranklogit": R3,
        "R2_minus_R3_curv": R2_m_R3,
        "R5_antithetic_absdiff": R5,
        "order_std_y1": order_std_y1,
        "order_std_llr": order_std_llr,
        "order_min_max_gap": order_minmax,
    }


def fold_delta(y, p_new, p_base, folds):
    d, pos = [], 0
    for k in range(5):
        m = folds == k
        da = (roc_auc_score(y[m], p_new[m]) - roc_auc_score(y[m], p_base[m])) * 1e5
        d.append(float(da))
        pos += int(da > 0)
    return float(np.mean(d)), pos, d


def sf25(y, a, b, folds) -> int:
    wins = 0
    for k in range(5):
        m = folds == k
        yk, ak, bk = y[m], a[m], b[m]
        skf = StratifiedKFold(5, shuffle=True, random_state=42 + k)
        for _, va in skf.split(np.arange(len(yk)), yk):
            if roc_auc_score(yk[va], ak[va]) > roc_auc_score(yk[va], bk[va]):
                wins += 1
    return wins


def t25(fd):
    x = np.asarray(fd, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-12))


def cos_full(a, b):
    a, b = unit(a), unit(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def lofo_signed_eval(y, folds, base_zr, feat_o, feat_t, Zg, ab, ray_s02, ray_s06, w_grid=None):
    """Strict LOFO signed weights w ∈ [-0.03,+0.03] on continuous ECDF-aligned feat."""
    if w_grid is None:
        w_grid = np.linspace(-0.03, 0.03, 13)
    oo, tt, disp, ks = method_b_cont(feat_o, feat_t, folds)
    oof2 = np.zeros(N_TR)
    chosen = [0.0] * 5
    for f in range(5):
        tr, va = folds != f, folds == f
        best_w, best_a = 0.0, -1.0
        base_L_tr = L(base_zr[tr])
        for w in w_grid:
            cand = base_L_tr + float(w) * oo[tr]
            a = roc_auc_score(y[tr], apply_zr(cand, Zg[tr], base_zr[tr]))
            if a > best_a:
                best_a, best_w = a, float(w)
        chosen[f] = best_w
        oof2[va] = L(base_zr[va]) + best_w * oo[va]
    w_bar = float(np.mean(chosen))
    p_oof = apply_zr(oof2, Zg, base_zr)
    me, pos, fd = fold_delta(y, p_oof, base_zr, folds)
    tip = unit(tt) * (1.0 if abs(w_bar) < 1e-12 else float(np.sign(w_bar)))
    return {
        "mean_e5": me,
        "pos": pos,
        "folds": fd,
        "sf25": sf25(y, p_oof, base_zr, folds),
        "t25": t25(fd),
        "w_lofo": chosen,
        "w_bar": w_bar,
        "disp": disp,
        "KS": ks,
        "cos_S02": cos_full(tip, ray_s02),
        "cos_S06": cos_full(tip, ray_s06),
        "solo_auc": float(roc_auc_score(y, feat_o)),
    }


def main():
    log("=== V43 Stage-1 recover U5 + 3D order tensor | NO_SUBMIT ===")
    OUT.mkdir(parents=True, exist_ok=True)
    PRED_OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS_P)["fold"].to_numpy(int)
    tr = pd.read_csv(DATA / "train.csv")
    Zg = z_goodpjw(tr)

    z_mod = np.load(MOD / "z_modern_pub_oof.npy").ravel()
    z_mod_t = np.load(MOD / "z_modern_pub_test.npy").ravel()
    z_mod_zr = apply_zr(z_mod, Zg, z_mod)
    base_auc = float(roc_auc_score(y, z_mod_zr))
    log(f"z_modern_pub+Zr={base_auc:.6f}")

    ab = pd.read_csv(FINAL / "SLOT01_pub_Step_AB_095569.csv")["is_fraud"].to_numpy(float)
    s02 = pd.read_csv(FINAL / "SLOT02_AB_clip985_areeb_a0p32.csv")["is_fraud"].to_numpy(float)
    s06 = pd.read_csv(FINAL / "SLOT06_AB_C3_harvest_p4_clip985_0p28e4.csv")["is_fraud"].to_numpy(float)
    ray_s02, ray_s06 = L(s02) - L(ab), L(s06) - L(ab)

    # ── l7 full 5-fold raw (N,10,2) / (5,N,10,2) ──
    lo = np.load(ROOT / "preds/l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    lt = np.load(ROOT / "preds/l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    assert lo.shape == (N_TR, 10, 2), lo.shape
    assert lt.shape[0] == 5 and lt.shape[1] == N_TE, lt.shape
    log(f"l7 shapes oof={lo.shape} test={lt.shape}")

    # ── gpt2 reconstruct full OOF (N,8,2); test mean over folds (N,4,2) ──
    gdir = ROOT / "preds/gpt2_med_4ep"
    go = np.zeros((N_TR, 8, 2), dtype=np.float64)
    gt_folds = []
    for k in range(5):
        a = np.load(gdir / f"oof_{k}.npy")
        b = np.load(gdir / f"test_{k}.npy")
        assert a.shape[0] == int((folds == k).sum()), (k, a.shape)
        go[folds == k] = a
        gt_folds.append(b)
    gt = np.mean(np.stack(gt_folds, 0), 0)  # (N_TE, 4, 2)
    log(f"gpt2 shapes oof={go.shape} test={gt.shape}")

    # ── U5..U10 from l7 ──
    u_l7 = u_bank_from_llr(lo[:, :, 1], lo[:, :, 0])
    # test: mean over train-folds' U features
    u_l7_t = {}
    for name in u_l7:
        per = []
        for f in range(5):
            bank = u_bank_from_llr(lt[f, :, :, 1], lt[f, :, :, 0])
            per.append(bank[name])
        u_l7_t[name] = np.mean(per, 0)

    # U5..U10 from gpt2 (K=8 oof; K=4 test — half-split still works)
    u_g = u_bank_from_llr(go[:, :, 1], go[:, :, 0])
    u_g_t = u_bank_from_llr(gt[:, :, 1], gt[:, :, 0])

    # ── 7 order-curvature features ──
    cur_l7 = order_curvature_7(lo[:, :, 1], lo[:, :, 0])
    cur_l7_t = {}
    for name in cur_l7:
        per = [order_curvature_7(lt[f, :, :, 1], lt[f, :, :, 0])[name] for f in range(5)]
        cur_l7_t[name] = np.mean(per, 0)

    cur_g = order_curvature_7(go[:, :, 1], go[:, :, 0])
    cur_g_t = order_curvature_7(gt[:, :, 1], gt[:, :, 0])

    save = {}
    for prefix, bank_o, bank_t in (
        ("l7", u_l7, u_l7_t),
        ("l7", cur_l7, cur_l7_t),
        ("gpt2", u_g, u_g_t),
        ("gpt2", cur_g, cur_g_t),
    ):
        for name, o in bank_o.items():
            t = bank_t[name]
            key = f"{prefix}_{name}"
            np.save(PRED_OUT / f"{key}_oof.npy", o)
            np.save(PRED_OUT / f"{key}_test.npy", t)
            save[key] = {"oof": o, "test": t}

    # also alias canonical U5
    np.save(PRED_OUT / "U5_gini10_llr_oof.npy", u_l7["U5_gini10_llr"])
    np.save(PRED_OUT / "U5_gini10_llr_test.npy", u_l7_t["U5_gini10_llr"])
    log(f"saved {len(list(PRED_OUT.glob('*.npy')))} npy assets")

    # ── U_C_tab note: V22 blend of cat+r2 residuals — reconstruct proxy ──
    # Use R3_ranklogit from l7 as R2-family proxy + B_xgb as tab proxy if present
    uc_note = "U_C_tab historically = 0.55*u_cat + 0.45*u_r2 (V22); rebuilt proxy from l7.R3 + B_xgb_llr"
    xgb_o = np.load(ROOT / "preds/paradigm_B/xgb_llr_oof.npy").ravel()
    xgb_t = np.load(ROOT / "preds/paradigm_B/xgb_llr_test.npy").ravel()
    uc_o = 0.55 * unit(L_fold(xgb_o, folds)) + 0.45 * unit(L_fold(cur_l7["R3_ranklogit"], folds))
    uc_t = 0.55 * unit(L(xgb_t)) + 0.45 * unit(L(cur_l7_t["R3_ranklogit"]))
    np.save(PRED_OUT / "U_C_tab_proxy_oof.npy", uc_o)
    np.save(PRED_OUT / "U_C_tab_proxy_test.npy", uc_t)
    save["U_C_tab_proxy"] = {"oof": uc_o, "test": uc_t}

    # ── orig20k / 0.96153 forensics ──
    forensics = {
        "source": "logs/overnight/GM_V33_1_HARDENED_FORENSICS_REPORT.md",
        "orig20k_leak_free_oof_signal": False,
        "note": (
            "orig20k gen.py is interaction/tree (Linear 0.880 → HistGBDT 0.934 on orig); "
            "synthesizer residual does NOT lift train (0/5). "
            "0.96153@K104 exceeds black-box 60k/80k Query_Pub bound → NOT recoverable from "
            "Public-LB probing alone; requires external/structural info. No gen.py leak found."
        ),
        "absolute_reach_sim": {"0.95632@K40": False, "0.96153@K104": False},
        "local_orig20k_assets": [
            "preds/fast_gpu_harvest/oof_orig20k_knn.npy",
            "preds/v36_1/W1_E2_orig20k_inversion_and_bpe_exact_*.npy",
        ],
    }

    # ── evaluate each feature on z_modern_pub+Zr ──
    results = {}
    # prioritize U5..U10 l7 + key curvature + U_C + gpt2 U5
    eval_keys = (
        [f"l7_{k}" for k in u_l7]
        + [f"l7_{k}" for k in cur_l7]
        + [f"gpt2_{k}" for k in ("U5_gini10_llr", "R3_ranklogit", "R5_antithetic_absdiff", "order_std_llr")]
        + ["U_C_tab_proxy"]
    )
    for key in eval_keys:
        if key not in save:
            # rebuild key from save naming
            continue
        o, t = save[key]["oof"], save[key]["test"]
        # flip if inverted
        if float(roc_auc_score(y, o)) < 0.5:
            o, t = -o, -t
        log(f"eval {key} ...")
        results[key] = lofo_signed_eval(
            y, folds, z_mod_zr, o, t, Zg, ab, ray_s02, ray_s06
        )
        r = results[key]
        log(
            f"  {key}: Δe5={r['mean_e5']:+.2f} pos={r['pos']}/5 sf={r['sf25']}/25 "
            f"t25={r['t25']:.2f} w̄={r['w_bar']:+.4f} cosS02={r['cos_S02']:.3f} cosS06={r['cos_S06']:.3f}"
        )

    # composite: mean of top positive U5/U6/U7
    top = sorted(
        [(k, v["mean_e5"]) for k, v in results.items() if "U5" in k or "U6" in k or "U7" in k],
        key=lambda z: -z[1],
    )[:3]
    report = {
        "NO_SUBMIT": True,
        "pods_started": 0,
        "z_modern_pub_zr": base_auc,
        "gpt2_shape_fix": {"oof": list(go.shape), "test": list(gt.shape)},
        "l7_shape": {"oof": list(lo.shape), "test": list(lt.shape)},
        "U_C_tab": uc_note,
        "forensics_96153_orig20k": forensics,
        "feature_evals": results,
        "top_U_features": top,
        "assets_dir": str(PRED_OUT),
    }
    (LOG / "V43_STAGE1_U5_ORDER_EVAL.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"WROTE {LOG / 'V43_STAGE1_U5_ORDER_EVAL.json'}")
    print(json.dumps({"base": base_auc, "n_eval": len(results), "top": top[:5]}, indent=2))


if __name__ == "__main__":
    main()
