#!/usr/bin/env python3
"""GM V17.2 — Read-only G26 SEED=2026 production blueprint audit + 1f→5f lift sim.

STRICT NO_SUBMIT=True. 0 GPU pods. Does not train or score.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
TMP15 = Path("/tmp/s6e11_v15")
TMP16 = Path("/tmp/s6e11_v16")
OUT = TMP16 / "out"
LOG = REPO / "logs/overnight"
G26_README = TMP16 / "g26/README.md"
MAT = REPO / "external/matapanino_gpt2_repo"
NO_SUBMIT = True


def L(v):
    v = np.asarray(v, float).ravel()
    r = (rankdata(v, method="average") - 0.5) / len(v)
    r = np.clip(r, 1e-12, 1.0 - 1e-12)
    return np.log(r / (1.0 - r))


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    assert NO_SUBMIT
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    readme = G26_README.read_text() if G26_README.exists() else ""
    # Blueprint extracted from source (static)
    blueprint = {
        "NO_SUBMIT": True,
        "g26_readme": readme,
        "pipeline_stages": [
            "make_folds.py — StratifiedKFold(5, shuffle=True, random_state=42)",
            "train_replica.py — ONCE on original 20k CSV (GReaT/be_great), NOT per-fold, NOT on train+test",
            "train_classifier.py — once per fold, continue from replica/final, target-first, 8 epochs",
            "score_rows.py — OOF then test per fold; log p(True) and log p(False)",
            "make_predictions.py — log-sum-exp over orders + prior → oof.npy / test.npy",
        ],
        "train_replica": {
            "once_globally": True,
            "per_fold": False,
            "data": "original CSV only (uditjain13 20k), drop id; NOT competition train+test",
            "model": "gpt2",
            "epochs": 60,
            "batch_size": 16,
            "precision_default": "fp32",
            "seed_default": 42,
            "g26_seed": 2026,
            "published_checkpoint": False,
            "note": "DEAD_ENDS N9 / local search: no replica .pt in public datasets; only OOF/test npy + HTML",
        },
        "train_classifier": {
            "per_fold": True,
            "epochs": 8,
            "batch_size": 32,
            "lr": 5e-5,
            "precision_default": "bf16",
            "seed_default": 42,
            "g26_match_args": {
                "--seed": 2026,
                "--fold": "1..4 (and 0 already done)",
                "--epochs": 8,
                "--batch-size": 32,
                "--lr": 5e-5,
                "--precision": "bf16",
                "--layout": "first",
                "--replica": "RUN/replica/final",
                "fold_ids": "StratifiedKFold seed 42 (same as masaya)",
            },
            "orders_note": "training uses fresh random column orders per step; not --orders (that is score_rows)",
        },
        "score_rows": {
            "orders_default_make_queue": 30,
            "orders_g26_test_actual": 10,
            "orders_masaya_oof": "30 for some folds, 10 for others (manifest)",
            "precision_default": "fp32 (post); bf16/tf32 optional faster + rounding noise",
            "batch_sequences_default": 256,
            "chunk_rows": 2000,
            "true_false_pass": "SINGLE batched pass: concat [ids_true; ids_false] then one forward; bodies_per_batch = batch_sequences//2",
            "padding": "bucketed by exact token length — no pad within a batch",
            "L_seq_median": 224,
            "why_slow_P100": [
                "fp32 full GPT-2 12×768 @ L≈224",
                "K=10 orders × 2 labels × 299984 rows ≈ 6e6 forward sequences per fold",
                "lm_head over full vocab in chunks (memory)",
                "P100 << 4090/A100 throughput",
            ],
            "speedups": {
                "bf16_vs_fp32": "~1.5–2× on Ampere+ (score_rows supports --precision bf16)",
                "batch_512_vs_256": "minor; docstring says batch_sequences sets memory not score",
                "SDPA_FlashAttn": "HF GPT-2 path benefits on 4090/A100 if transformers SDPA enabled; not custom in score_rows",
                "est_wall_score_per_fold_h": {"P100": 12.6, "RTX_4090": 1.68, "A100": 1.4},
            },
        },
        "make_queue_defaults": {
            "--orders": 30,
            "--batch-sequences": 256,
            "--score-precision": "fp32",
            "--stage": "all|oof",
        },
    }

    # Lift simulation
    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, y, fold_id = cache["z_ath"], cache["y"], cache["fold_id"]
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    R3_L = np.stack([L(R3_te[f]) for f in range(5)], 0)
    u_folds = np.stack([unitize(R3_L[f]) for f in range(5)], 0)
    rho = float(
        np.mean(
            [spearmanr(R3_L[f], R3_L[g]).statistic for f in range(5) for g in range(f + 1, 5)]
        )
    )
    sig5 = float(np.std(u_folds[0] - u_folds.mean(0)))
    sig3 = float(np.std(u_folds[0] - u_folds[:3].mean(0)))
    w = 0.226
    rng = np.random.default_rng(2026)

    def mc_main(sig, n=30):
        G = []
        for _ in range(n):
            eps = rng.normal(0, sig, len(y))
            # z_ath already has 5f G26 OOF; +w*ε simulates LIVE's 1f G26; lift = clean - noisy
            d = [
                (
                    roc_auc_score(y[te], z_ath[te])
                    - roc_auc_score(y[te], z_ath[te] + w * eps[te])
                )
                * 1e5
                for _, te in folds
            ]
            G.append(d)
        G = np.asarray(G)
        return {
            "mean": float(G.mean()),
            "std": float(G.mean(1).std()),
            "per_fold_mean": [float(x) for x in G.mean(0)],
            "frac_all5_pos": float(np.mean(np.all(G > 0, 1))),
            "mean_pos_folds": float(np.mean((G > 0).sum(1))),
        }

    g5, g3 = mc_main(sig5), mc_main(sig3)
    idx = rng.choice(len(y), 120000, replace=False)
    ys, zs = y[idx], z_ath[idx]

    def mc_sub(sig, n=25):
        gs = [
            (roc_auc_score(ys, zs) - roc_auc_score(ys, zs + w * rng.normal(0, sig, len(idx))))
            * 1e5
            for _ in range(n)
        ]
        gs = np.asarray(gs)
        return {
            "mean": float(gs.mean()),
            "std": float(gs.std()),
            "p10": float(np.percentile(gs, 10)),
            "p90": float(np.percentile(gs, 90)),
        }

    s5, s3 = mc_sub(sig5), mc_sub(sig3)
    sig98 = float(np.sqrt(0.8 * 0.02))
    s98 = mc_sub(sig98)

    eps = rng.normal(0, sig5, len(y))
    te = folds[0][1]
    g0 = [
        (
            roc_auc_score(y[te[ste]], z_ath[te[ste]])
            - roc_auc_score(y[te[ste]], z_ath[te[ste]] + w * eps[te[ste]])
        )
        * 1e5
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(te)), y[te]
        )
    ]

    lift = {
        "calibration": "σ from unitized L7_s2026 test (f0−mean5); |β_G26|=0.226; Lift=AUC(z)−AUC(z+wε)",
        "l7_test_cross_fold_spearman": rho,
        "sigma_1f_minus_5f": sig5,
        "sigma_1f_minus_3f": sig3,
        "w_g26": w,
        "expected_public_private_AUC_gain_e5": {
            "G26_test_5fold_vs_f0": {"main_folds_MC": g5, "pooled_120k_MC": s5},
            "G26_test_3fold_vs_f0": {"main_folds_MC": g3, "pooled_120k_MC": s3},
            "upper_if_fold_rho_only_0.98": s98,
        },
        "fold0_subfold_gains_e5": [float(x) for x in g0],
        "verdict": (
            f"Expected lift ≈ {s5['mean']:.2f} e-5 (≈{s5['mean']/10:.2f} public ticks) under L7-calibrated σ; "
            f"90%-ish band [{s5['p10']:.1f},{s5['p90']:.1f}] e-5. Not a multi-tick free lunch — fold agreement is already ρ≈{rho:.3f}."
        ),
    }

    payload = {**blueprint, "lift_simulation": lift}
    OUT.joinpath("v17_2.json").write_text(json.dumps(payload, indent=2))
    OUT.joinpath("v17_2_lift.json").write_text(json.dumps(lift, indent=2))
    print(lift["verdict"])
    print("DONE")
    return payload


if __name__ == "__main__":
    main()
