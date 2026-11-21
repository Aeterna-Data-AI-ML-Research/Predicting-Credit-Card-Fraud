#!/usr/bin/env python3
"""GM V17.0 — Fold-0 seed-transfer test, simplex/Ridge weight audit, G26 architecture audit.

STRICT NO_SUBMIT=True. 0 GPU pods.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
TMP15 = Path("/tmp/s6e11_v15")
TMP16 = Path("/tmp/s6e11_v16")
OUT = TMP16 / "out"
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
NB = REPO / "external/nov13_recon/goodpjw_095386/s6e11-lm-ladder-public-and-cv-up-lb-0-95386.ipynb"
MAT = REPO / "external/matapanino_gpt2_repo"
TOK_JSON = (
    REPO
    / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-fraud-series-artifacts/gpt2-tokenizer.json"
)
NO_SUBMIT = True


def L(v):
    v = np.asarray(v, float).ravel()
    r = (rankdata(v, method="average") - 0.5) / len(v)
    r = np.clip(r, 1e-12, 1.0 - 1e-12)
    return np.log(r / (1.0 - r))


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def partial_corr(a, b, controls):
    """Pearson corr of residuals of a,b after OLS on controls (+intercept)."""
    X = np.column_stack([np.asarray(c, float).ravel() for c in controls] + [np.ones(len(a))])
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def screen_noflip(name, u, zbase, y, folds, ws=(0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)):
    """NO flip_to_plus_AUC — evaluate the given signed direction as-is."""
    u = unitize(u)
    base = [roc_auc_score(y[te], zbase[te]) for _, te in folds]
    best = None
    for w in ws:
        d = [
            (roc_auc_score(y[te], zbase[te] + w * u[te]) - base[i]) * 1e5
            for i, (_, te) in enumerate(folds)
        ]
        row = {
            "w": w,
            "pos": int(sum(x > 0 for x in d)),
            "mean": float(np.mean(d)),
            "min": float(np.min(d)),
            "folds": [round(x, 3) for x in d],
        }
        if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
            best = row
    deltas = []
    for _, te in folds:
        yf, zb, uf = y[te], zbase[te], u[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(yf)), yf
        ):
            b = roc_auc_score(yf[ste], zb[ste])
            deltas.append((roc_auc_score(yf[ste], zb[ste] + 0.01 * uf[ste]) - b) * 1e5)
    d25 = np.asarray(deltas, float)
    mean = float(d25.mean())
    se = float(d25.std(ddof=1) / np.sqrt(len(d25)))
    t = mean / (se + 1e-18)
    out = {
        "name": name,
        "main": best,
        "sf25": int((d25 > 0).sum()),
        "t": float(t),
        "mean_sf": mean,
    }
    print(
        f"  {name}: {best['pos']}/5 @{best['w']} mean={best['mean']:.3f} "
        f"min={best['min']:.3f} sf={out['sf25']}/25 t={t:.2f} folds={best['folds']}"
    )
    return out


def flop_forward(n_layer, n_embd, L_seq):
    """Rough GPT-2 forward FLOPs (matmul dominated): attn + MLP."""
    d = n_embd
    # QKV + proj + 2*MLP: ~ 2*n_layer*(4*L*d^2 + 2*L^2*d) is common approx; use:
    # attn: 4*L*d^2 (qkv+proj) + 2*L^2*d (scores+av); mlp: 8*L*d^2
    return n_layer * (12 * L_seq * d * d + 2 * L_seq * L_seq * d)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    assert NO_SUBMIT

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]

    Z = np.load(Q17, allow_pickle=True)
    names = [str(s) for s in Z["stream_names"]]
    XO = {n: Z[f"oof_{n}"].ravel() for n in names}
    XT = {n: Z[f"test_{n}"].ravel() for n in names}

    # ---------- STEP 1 ----------
    print("=== STEP 1: Fold-0 cross-seed ground truth ===")
    # Seed7 f0: (N,4,2) -> mean LLR
    lm7_f0_raw = np.load(TMP15 / "banks/lm7_test_f0.npy")
    lm7_f0 = lm7_f0_raw[:, :, 1] - lm7_f0_raw[:, :, 0]
    lm7_f0 = lm7_f0.mean(1)
    lm7_5 = np.load(TMP15 / "banks/lm7_test_5fold.npy").ravel()
    # also Q17 lm7 test (should match 5fold)
    err_f0_s7 = L(lm7_f0) - L(lm7_5)

    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")  # (5,N)
    R3_L = np.stack([L(R3_te[f]) for f in range(5)], 0)
    R3_5 = L(R3_L.mean(0))
    err_f0_s2026 = R3_L[0] - R3_5

    p_r = float(pearsonr(err_f0_s7, err_f0_s2026)[0])
    s_r = float(spearmanr(err_f0_s7, err_f0_s2026).statistic)
    print(f"  pearson(err_f0_s7, err_f0_s2026)={p_r:.4f}")
    print(f"  spearman(err_f0_s7, err_f0_s2026)={s_r:.4f}")

    # corr(err_f0_s7, L(R3_f)-L(R3_5)) for each f
    fold_corrs = {}
    for f in range(5):
        e_f = R3_L[f] - R3_5
        fold_corrs[f] = {
            "pearson": float(pearsonr(err_f0_s7, e_f)[0]),
            "spearman": float(spearmanr(err_f0_s7, e_f).statistic),
        }
        print(
            f"  corr(err_f0_s7, R3_f{f}-R3_5f): pearson={fold_corrs[f]['pearson']:+.4f} "
            f"spearman={fold_corrs[f]['spearman']:+.4f}"
        )

    # Also 5x5 matrix of seed2026 fold errors vs each other / vs s7
    C = np.zeros((5, 5))
    for f in range(5):
        for g in range(5):
            C[f, g] = float(pearsonr(R3_L[f] - R3_5, R3_L[g] - R3_5)[0])
    print("  C[f,g]=corr(R3_f-R3_5, R3_g-R3_5):\n", np.round(C, 3))

    # leave-one-seed-out: predict err_f0_s2026 from err_f0_s7
    # R^2 of γ·err_s7 explaining err_s2026
    gamma = float(np.dot(err_f0_s7, err_f0_s2026) / (np.dot(err_f0_s7, err_f0_s7) + 1e-18))
    pred = gamma * err_f0_s7
    ss_res = float(np.sum((err_f0_s2026 - pred) ** 2))
    ss_tot = float(np.sum((err_f0_s2026 - err_f0_s2026.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-18)
    # also: recover gap if we only had R3_0 and subtract γ·err_s7 toward R3_5
    recovered = R3_L[0] - gamma * err_f0_s7
    # how close to R3_5?
    mse_raw = float(np.mean((R3_L[0] - R3_5) ** 2))
    mse_deb = float(np.mean((recovered - R3_5) ** 2))
    mse_red = 1.0 - mse_deb / (mse_raw + 1e-18)
    print(f"  γ_hat={gamma:.4f}  R²(err_s2026 ~ γ·err_s7)={r2:.4f}  MSE_reduction={mse_red:.4f}")

    # G26_f0 - MS_5fold vs fold0 errors, controlling for seed contrast
    g26_t = np.load(TMP16 / "g26/gpt2_s2026_test_f0.npy").ravel()
    ms_t = np.load(REPO / "preds/P_super_v4_test.npy").ravel()
    # MS might be P_super not pure masaya — also use Q17 masaya
    ms_q = XT["masaya_gpt2"]
    g26_minus_ms = L(g26_t) - L(ms_q)
    seed_contrast = R3_5 - L(lm7_5)
    c_s7_raw = float(pearsonr(err_f0_s7, g26_minus_ms)[0])
    c_s26_raw = float(pearsonr(err_f0_s2026, g26_minus_ms)[0])
    c_s7_p = partial_corr(err_f0_s7, g26_minus_ms, [seed_contrast])
    c_s26_p = partial_corr(err_f0_s2026, g26_minus_ms, [seed_contrast])
    print(
        f"  corr(err_s7, G26f0-MS): raw={c_s7_raw:.4f} partial|seed={c_s7_p:.4f}"
    )
    print(
        f"  corr(err_s2026, G26f0-MS): raw={c_s26_raw:.4f} partial|seed={c_s26_p:.4f}"
    )

    step1 = {
        "pearson_err_cross_seed": p_r,
        "spearman_err_cross_seed": s_r,
        "fold_corrs_err_s7_vs_R3f_minus_5": fold_corrs,
        "C_R3_fold_errors": C.tolist(),
        "gamma": gamma,
        "R2_err_s2026_on_err_s7": r2,
        "MSE_reduction_debias_R3_0": mse_red,
        "corr_G26f0_minus_MS": {
            "err_s7_raw": c_s7_raw,
            "err_s7_partial_seed": c_s7_p,
            "err_s2026_raw": c_s26_raw,
            "err_s2026_partial_seed": c_s26_p,
        },
        "hypothesis": (
            "PASS"
            if fold_corrs[0]["pearson"] > 0.15
            and all(fold_corrs[f]["pearson"] < 0 for f in range(1, 5))
            else "WEAK/FAIL"
        ),
    }
    print(f"  Fold-0-bias hypothesis: {step1['hypothesis']}")

    # ---------- STEP 2 ----------
    print("=== STEP 2: weight audit (NO flip) ===")
    # bold recipe weights from notebook
    nb = json.load(open(NB))
    src = "".join(nb["cells"][0]["source"])
    i = src.find("RECIPE_JSON = '''")
    j = src.find("'''", i + 17)
    bold = json.loads(src[i + 17 : j])["bold"]
    stream_map = {
        "MS": ["s:masaya_gpt2"],
        "G26": ["s:gpt2_seed2026"],
        "LM": ["s:megayak_5f_lm"],
        "L7F": ["s:lm7_5fold"],
        "GPD": ["s:daoviet_gpdvae"],
        "GB": ["s:gb_bag5"],
        "MLP": ["s:yekenot_realmlp"],
        "wide": ["wide"],
    }
    bold_core = {k: sum(bold.get(p, 0.0) for p in pats) for k, pats in stream_map.items()}
    print("  bold recipe core weights:", {k: round(v, 6) for k, v in bold_core.items()})

    # solo AUCs
    R3_o = np.load(TMP15 / "out/R3_oof.npy").ravel()
    streams_o = {
        "MS": XO["masaya_gpt2"],
        "G26": np.load(TMP16 / "g26/gpt2_s2026_oof_5fold.npy").ravel()[: len(y)],
        "LM": XO["megayak_5f_lm"],
        "L7F": XO["lm7_5fold"],
        "R3_s2026": R3_o,
        "GPD": XO["daoviet_gpdvae"],
        "GB": XO["gb_bag5"],
        "MLP": XO["yekenot_realmlp"],
        "d1_v4": d1,
    }
    solo = {}
    for nm, arr in streams_o.items():
        a = np.asarray(arr, float).ravel()[: len(y)]
        folds_auc = [float(roc_auc_score(y[te], a[te])) for _, te in folds]
        solo[nm] = {"folds": [round(x, 6) for x in folds_auc], "mean": float(np.mean(folds_auc))}
        print(f"  solo {nm}: mean={solo[nm]['mean']:.6f} folds={solo[nm]['folds']}")

    # 2D replace grid: z_ATH + w*(L(R3)-L(stream_j))  NO flip; w=alpha in [0,0.05]
    print("  --- replace grid: z_ATH + α·(L(R3)-L(j)) NO flip ---")
    L_R3 = L(R3_o)
    replace_results = {}
    for j in ("MS", "G26", "LM", "L7F", "GPD", "GB", "MLP"):
        u = L_R3 - L(streams_o[j])  # signed as-is
        scr = screen_noflip(f"R3_minus_{j}", u, z_ath, y, folds)
        replace_results[f"R3_minus_{j}"] = scr
        blend = 0.5 * L(streams_o["L7F"]) + 0.5 * L_R3
        u2 = blend - L(streams_o[j])
        scr2 = screen_noflip(f"L7_2seed_minus_{j}", u2, z_ath, y, folds)
        replace_results[f"L7_2seed_minus_{j}"] = scr2

    print("  --- also α·(L7_2seed - L7F) ---")
    u_blend = (0.5 * L(streams_o["L7F"]) + 0.5 * L_R3) - L(streams_o["L7F"])
    replace_results["L7_2seed_minus_L7F_direct"] = screen_noflip(
        "L7_2seed_minus_L7F", u_blend, z_ath, y, folds
    )

    best_replace = max(
        replace_results.items(),
        key=lambda kv: (kv[1]["main"]["pos"], kv[1]["sf25"], kv[1]["main"]["mean"]),
    )
    print(f"  BEST replace: {best_replace[0]} → {best_replace[1]['main']}")

    # Non-negative Ridge / L2 Logistic LOFO on 9 streams
    print("  --- LOFO non-neg Logistic / Ridge ---")
    bn9 = ["MS", "G26", "LM", "L7F", "R3_s2026", "GPD", "GB", "MLP", "d1_v4"]
    X9 = np.column_stack([unitize(L(streams_o[n]) if n != "d1_v4" else streams_o[n]) for n in bn9])
    w_folds_log = []
    w_folds_ridge = []
    oof_log = np.zeros(len(y))
    oof_ridge = np.zeros(len(y))
    for tr, te in folds:
        # Logistic with L2, non-negative via post-hoc clip+renorm (true simplex soft)
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, fit_intercept=True
        )
        clf.fit(X9[tr], y[tr])
        w = clf.coef_.ravel().copy()
        w = np.clip(w, 0, None)
        if w.sum() > 0:
            w = w / w.sum()
        w_folds_log.append(dict(zip(bn9, map(float, w))))
        oof_log[te] = X9[te] @ w

        rr = Ridge(alpha=10.0, fit_intercept=True, positive=True)
        rr.fit(X9[tr], y[tr].astype(float))
        wr = rr.coef_.ravel().copy()
        if wr.sum() > 0:
            wr = wr / wr.sum()
        w_folds_ridge.append(dict(zip(bn9, map(float, wr))))
        oof_ridge[te] = X9[te] @ wr

    w_mean_log = {k: float(np.mean([wf[k] for wf in w_folds_log])) for k in bn9}
    w_mean_ridge = {k: float(np.mean([wf[k] for wf in w_folds_ridge])) for k in bn9}
    print("  mean LOFO Logistic nonneg w:", {k: round(v, 4) for k, v in w_mean_log.items()})
    print("  mean LOFO Ridge-pos w:", {k: round(v, 4) for k, v in w_mean_ridge.items()})
    for f, wf in enumerate(w_folds_log):
        print(f"    fold{f} log: " + ", ".join(f"{k}={wf[k]:.3f}" for k in bn9 if wf[k] > 0.01))
    auc_log = float(roc_auc_score(y, oof_log))
    auc_ridge = float(roc_auc_score(y, oof_ridge))
    print(f"  OOF AUC simplex-log={auc_log:.7f} ridge-pos={auc_ridge:.7f} z_ATH={roc_auc_score(y,z_ath):.7f}")

    # R3 weight specifically
    step2 = {
        "bold_core_weights": bold_core,
        "solo_auc": solo,
        "replace_grid": {
            k: {kk: vv for kk, vv in v.items()} for k, v in replace_results.items()
        },
        "best_replace": {"name": best_replace[0], **best_replace[1]},
        "lofo_logistic_nonneg": {"per_fold": w_folds_log, "mean": w_mean_log, "oof_auc": auc_log},
        "lofo_ridge_positive": {
            "per_fold": w_folds_ridge,
            "mean": w_mean_ridge,
            "oof_auc": auc_ridge,
        },
        "note": "NO flip_to_plus_AUC on replace grid; direction is L(R3)-L(j) as written",
    }

    # ---------- STEP 3 ----------
    print("=== STEP 3: G26 architecture audit ===")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOK_JSON))
    df = pd.read_csv(REPO / "data/train.csv", dtype=str, keep_default_na=False, nrows=5000)
    cols = [c for c in df.columns if c not in ("id", "is_fraud")]
    def row_text(r, lab="True"):
        body = ", ".join(f"{c} is {str(r[c]).strip() or 'None'}" for c in cols)
        return f"is_fraud is {lab}, {body}<|endoftext|>"

    tl = [len(tok.encode(row_text(df.iloc[i])).ids) for i in range(len(df))]
    L_g26 = float(np.median(tl))
    prefix_len = len(tok.encode("is_fraud is True,").ids)

    # parse make_queue / train defaults from source text
    mq = (MAT / "make_queue.py").read_text()
    tr = (MAT / "train_replica.py").read_text()
    tc = (MAT / "train_classifier.py").read_text()
    sr = (MAT / "score_rows.py").read_text()
    # defaults
    g26_arch = {
        "tokenization": "GPT-2 BPE text: 'col is value' cells joined by ', '; target-first label prefix; NOT 1-token-per-column",
        "n_feature_cols": len(cols),
        "L_seq_median_tokens": L_g26,
        "L_seq_mean": float(np.mean(tl)),
        "L_seq_p95": float(np.percentile(tl, 95)),
        "L_seq_max": int(np.max(tl)),
        "label_prefix_tokens": prefix_len,
        "backbone": "HuggingFace gpt2 (12-layer, n_embd=768, n_head=12, ~124M)",
        "train_replica_epochs": 60,
        "train_classifier_epochs": 8,
        "score_orders_default": 30,
        "score_orders_test_used_by_masaya_g26": 10,
        "source_files": [str(p.relative_to(REPO)) for p in [MAT / "tabtext.py", MAT / "train_replica.py", MAT / "train_classifier.py", MAT / "score_rows.py", MAT / "make_queue.py"]],
    }
    L_l7 = 72.0
    n_layer = 12
    n_embd = 768
    K_g26 = 10
    K_l7 = 10
    flop_g26 = flop_forward(n_layer, n_embd, L_g26)
    flop_l7 = flop_forward(n_layer, n_embd, L_l7)
    # scoring: 2 labels × K orders × N_test rows
    score_flop_ratio = (flop_g26 * 2 * K_g26) / (flop_l7 * 2 * K_l7)
    # wall refs
    wall_l7_4080_min = 75.0
    wall_g26_p100_h = 12.6
    # rough GPU relative throughput (FP16/TF32 inference ballpark)
    rel = {
        "P100": 1.0,
        "RTX_4080_SUPER": 5.5,  # ~ vs P100 for this workload class
        "RTX_4090": 7.5,
        "A100_80GB": 9.0,
    }
    # estimate G26 fold wall from FLOP ratio × L7 wall, scaled by GPU
    # L7 on 4080: 75m; G26 on same GPU ≈ 75 * score_flop_ratio (plus train cost)
    # Full fold = replica amortized? G26 fold = classifier 8ep + score; replica shared.
    # User asked wall per fold to generate test folds 1-4 — that's score_rows dominated (~12.6h said for scoring)
    g26_score_4080_h = wall_g26_p100_h * (rel["P100"] / rel["RTX_4080_SUPER"])
    g26_score_4090_h = wall_g26_p100_h * (rel["P100"] / rel["RTX_4090"])
    g26_score_a100_h = wall_g26_p100_h * (rel["P100"] / rel["A100_80GB"])
    # FLOP-based cross-check from L7
    g26_from_l7_4080_min = wall_l7_4080_min * score_flop_ratio

    print(f"  G26 L_seq median={L_g26:.0f} (BPE text, {len(cols)} cols)")
    print(f"  L7 T={L_l7:.0f} compact tokens, also nl=12 d=768 (NOT 4-layer)")
    print(f"  FLOP/forward ratio G26/L7={flop_g26/flop_l7:.2f}  score FLOP ratio (K=10 both)={score_flop_ratio:.2f}")
    print(
        f"  Est G26 score wall/fold: 4080≈{g26_score_4080_h:.2f}h  4090≈{g26_score_4090_h:.2f}h  "
        f"A100≈{g26_score_a100_h:.2f}h  (from 12.6h P100)"
    )
    print(f"  FLOP-scaled from L7 75m@4080 → G26 score ≈{g26_from_l7_4080_min:.0f} min/fold @4080")

    step3 = {
        **g26_arch,
        "L7_comparison": {
            "n_layer": n_layer,
            "n_embd": n_embd,
            "L_seq": L_l7,
            "V": 1372,
            "K_orders": K_l7,
            "note": "Compact-LM uses GPT-2 12×768 with T≈72 compact tokens (user '4-layer L_seq=25' is outdated vs repo)",
            "wall_min_per_fold_4080": wall_l7_4080_min,
        },
        "flop_forward_g26": flop_g26,
        "flop_forward_l7": flop_l7,
        "flop_ratio_forward": flop_g26 / flop_l7,
        "score_flop_ratio_K10": score_flop_ratio,
        "wall_estimate_hours_score_per_fold": {
            "P100_reported": wall_g26_p100_h,
            "RTX_4080_SUPER": g26_score_4080_h,
            "RTX_4090": g26_score_4090_h,
            "A100_80GB": g26_score_a100_h,
        },
        "flop_scaled_from_L7_min_at_4080": g26_from_l7_4080_min,
        "folds_1_to_4_total_hours": {
            "RTX_4090": 4 * g26_score_4090_h,
            "A100_80GB": 4 * g26_score_a100_h,
            "RTX_4080_SUPER": 4 * g26_score_4080_h,
        },
    }

    payload = {
        "NO_SUBMIT": True,
        "step1": step1,
        "step2": step2,
        "step3": step3,
    }

    def scrub(x):
        if isinstance(x, dict):
            return {k: scrub(v) for k, v in x.items()}
        if isinstance(x, list):
            return [scrub(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return float(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        return x

    OUT.joinpath("v17_0.json").write_text(json.dumps(scrub(payload), indent=2))
    print("WROTE", OUT / "v17_0.json")
    print("DONE")


if __name__ == "__main__":
    main()
