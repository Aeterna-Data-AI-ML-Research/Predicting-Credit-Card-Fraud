#!/usr/bin/env python3
"""Step1: score public TabPFN OOFs on Dual-Baseline G0. NO_SUBMIT=True."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
SSD = Path("/tmp/s6e11_v39_ssd")
OUT = SSD / "out"
OUT.mkdir(exist_ok=True)
SOLO_MIN = 0.9480
LIFT_MIN = 0.40e-4


def log(m):
    print(f"[{time.time()-T0:5.1f}s] {m}", flush=True)


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


def orth_span2(delta, Z2):
    """Ridge(α=50) residual of unit(delta) on unit columns of Z2 (2-col span)."""
    d = unit(delta)
    X = np.column_stack([unit(Z2[:, j]) for j in range(Z2.shape[1])])
    r = Ridge(alpha=50.0, fit_intercept=False)
    r.fit(X, d)
    return d - r.predict(X)


def dual_baseline_gate(y, p, z_fb, z_tw, name: str) -> dict:
    solo = auc(y, p)
    Z2 = np.column_stack([z_fb, z_tw])
    u = orth_span2(L(p) - L(z_fb), Z2)  # orth vs span{fb,tw} of residual vs fb
    # also compute lifts vs each baseline with out-of-subfold w*
    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    def subfold_lift(z, feat):
        d = []
        for tr, va in skf.split(np.zeros(len(y)), y):
            best_w, best_a = 0.0, -1.0
            for w in np.linspace(-0.08, 0.08, 33):
                a = auc(y[tr], L(z[tr]) + w * feat[tr])
                if a > best_a:
                    best_a, best_w = a, float(w)
            d.append(auc(y[va], L(z[va]) + best_w * feat[va]) - auc(y[va], z[va]))
        d = np.asarray(d)
        return {"mean_d": float(d.mean()), "n_pos": int((d > 0).sum()), "d": d.tolist()}

    g_fb = subfold_lift(z_fb, u)
    g_tw = subfold_lift(z_tw, u)
    ok = (
        solo >= SOLO_MIN
        and g_fb["n_pos"] >= 4
        and g_tw["n_pos"] >= 4
        and g_fb["mean_d"] >= LIFT_MIN
        and g_tw["mean_d"] >= LIFT_MIN
    )
    return {
        "name": name,
        "solo_oof": solo,
        "solo_ok": solo >= SOLO_MIN,
        "g_fb": g_fb,
        "g_tw": g_tw,
        "promote": ok,
        "n": int(len(y)),
    }


def main():
    log("=== Public TFM Dual-Baseline G0 | NO_SUBMIT ===")
    y = pd.read_csv(SSD / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    z_fb = np.load(SSD / "preds/z_fullbank_oof.npy").ravel()
    z_tw = np.load(SSD / "preds/z_ATH_twin_oof.npy").ravel()
    assert len(y) == 699960

    report = {"NO_SUBMIT": True, "candidates": {}, "gaurav": "403_FORBIDDEN"}

    # Sadam TabPFN 50k / 100k
    for tag in ["50k", "100k"]:
        oof = pd.read_csv(SSD / f"external/sadam_tabpfn/pair_oof_tabpfn_{tag}.csv")
        te = pd.read_csv(SSD / f"external/sadam_tabpfn/pair_test_tabpfn_{tag}.csv")
        assert len(oof) == 699960 and len(te) == 299984
        # align by id order (should be 0..n-1)
        oof = oof.sort_values("id")
        p = oof["is_fraud"].to_numpy(float)
        g = dual_baseline_gate(y, p, z_fb, z_tw, f"sadam_tabpfn_{tag}")
        report["candidates"][g["name"]] = g
        log(
            f"{g['name']}: solo={g['solo_oof']:.5f} "
            f"fbΔ={g['g_fb']['mean_d']:+.2e}({g['g_fb']['n_pos']}/5) "
            f"twΔ={g['g_tw']['mean_d']:+.2e}({g['g_tw']['n_pos']}/5) promote={g['promote']}"
        )

    # Amanatar Tri-TabPFN family average ONLY
    vault = np.load(SSD / "external/amanatar_vault/s6e11_sovereign_095423_vault.npz", allow_pickle=True)
    p = vault["oof_P2_tri_tabpfn"].astype(float).ravel()
    assert len(p) == 699960
    g = dual_baseline_gate(y, p, z_fb, z_tw, "amanatar_tri_tabpfn")
    report["candidates"][g["name"]] = g
    log(
        f"{g['name']}: solo={g['solo_oof']:.5f} "
        f"fbΔ={g['g_fb']['mean_d']:+.2e}({g['g_fb']['n_pos']}/5) "
        f"twΔ={g['g_tw']['mean_d']:+.2e}({g['g_tw']['n_pos']}/5) promote={g['promote']}"
    )

    report["any_promote"] = any(c["promote"] for c in report["candidates"].values())
    report["need_gpu_canary"] = not report["any_promote"]
    report["qwen_ll_va_k2"] = {
        "exists": (OUT / "ll_va_k2.npy").exists(),
        "note": "no K2 on Mac; Qwen pod terminated (stuck awaiting_container)",
    }
    report["elapsed_s"] = time.time() - T0
    (OUT / "public_tfm_dual_baseline_g0.json").write_text(json.dumps(report, indent=2))
    log(f"any_promote={report['any_promote']} need_gpu={report['need_gpu_canary']}")
    log(f"wrote {OUT/'public_tfm_dual_baseline_g0.json'}")
    return 0 if report["any_promote"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
