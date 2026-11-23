#!/usr/bin/env python3
"""V40: mix new public boards (>ATH 0.95550) with proprietary uid residuals → submit CSVs.

No OOF for public kernels → test-side rank/logit geodesics + proprietary residual axes.
Writes candidates under out/v40_ladder/ and a manifest JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata

SSD = Path("/tmp/s6e11_v39_ssd")
REPO = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
HARVEST = SSD / "external/public_harvest"
OUT = SSD / "out/v40_ladder"
OUT.mkdir(parents=True, exist_ok=True)

N_TE = 299_984


def U(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v: np.ndarray) -> np.ndarray:
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def load_sub(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    col = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
    p = df[col].to_numpy(float)
    assert len(p) == N_TE, f"{path}: {len(p)}"
    return p


def write_sub(name: str, p: np.ndarray, ids: np.ndarray) -> Path:
    p = np.clip(np.asarray(p, float).ravel(), 1e-7, 1 - 1e-7)
    path = OUT / f"{name}.csv"
    pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(path, index=False)
    return path


def orth_vs(cols: list[np.ndarray], target: np.ndarray) -> np.ndarray:
    """Ridge-ish least-squares residual of unit(target) on unit columns."""
    X = np.column_stack([unit(c) for c in cols])
    y = unit(target)
    # closed form with ridge
    xtx = X.T @ X + 50.0 * np.eye(X.shape[1])
    beta = np.linalg.solve(xtx, X.T @ y)
    return unit(y - X @ beta)


def main():
    ids = pd.read_csv(SSD / "data/test.csv", usecols=["id"])["id"].to_numpy()
    ath = load_sub(REPO / "submissions/sub_v39_MAX_uid_peak_a0365.csv")  # LB 0.95550
    areeb = load_sub(HARVEST / "areeb_95567/submission.csv")
    gpw = load_sub(HARVEST / "goodpjw_95562/submission.csv")
    # asterios board peak often submission.csv or extended
    ast = load_sub(HARVEST / "asterios_95552/submission.csv")

    uid = np.load(SSD / "out/uid_v2_test.npy").ravel()
    uid_cb = np.load(SSD / "out/uid_catboost_test.npy").ravel()
    z_fb = np.load(SSD / "preds/z_fullbank_test.npy").ravel()
    z_tw = np.load(SSD / "preds/z_ATH_twin_test.npy").ravel()

    # residual axes (test-side)
    ax_areeb_from_ath = unit(L(areeb) - L(ath))
    ax_gpw_from_ath = unit(L(gpw) - L(ath))
    ax_gpw_from_areeb = unit(L(gpw) - L(areeb))
    ax_ast_from_areeb = unit(L(ast) - L(areeb))
    ax_uid_orth = orth_vs([areeb, gpw, ath], uid)
    ax_cb_orth = orth_vs([areeb, gpw, ath], uid_cb)
    ax_fb_orth = orth_vs([areeb, gpw, ath], z_fb)

    def push(base: np.ndarray, axis: np.ndarray, omr: float) -> np.ndarray:
        # omr in units of std of L(base); convert via logit space then rank-keep soft
        return expit(L(base) + omr * axis)

    cands: dict[str, np.ndarray] = {
        "S01_PUB_areeb_95567": areeb,
        "S02_PUB_goodpjw_95562": gpw,
        "S03_PUB_asterios_95552": ast,
        "S04_cent_areeb_gpw": expit(0.5 * (L(areeb) + L(gpw))),
        "S05_cent_areeb_gpw_ast": expit((L(areeb) + L(gpw) + L(ast)) / 3.0),
        "S06_cent_areeb_gpw_ATH": expit((L(areeb) + L(gpw) + L(ath)) / 3.0),
        # lock public leap then overshoot past areeb along gpw residual
        "S07_areeb_along_gpw_0p20e4": push(areeb, ax_gpw_from_areeb, 0.20e-4),
        "S08_areeb_along_gpw_0p40e4": push(areeb, ax_gpw_from_areeb, 0.40e-4),
        "S09_areeb_along_ast_0p25e4": push(areeb, ax_ast_from_areeb, 0.25e-4),
        # from our ATH along areeb (BOLD public ascent)
        "S10_ATH_along_areeb_0p50e4": push(ath, ax_areeb_from_ath, 0.50e-4),
        "S11_ATH_along_areeb_1p00e4": push(ath, ax_areeb_from_ath, 1.00e-4),
        "S12_ATH_along_gpw_0p50e4": push(ath, ax_gpw_from_ath, 0.50e-4),
        # proprietary orth on NEW public base (areeb)
        "S13_areeb_uid_orth_0p25e4": push(areeb, ax_uid_orth, 0.25e-4),
        "S14_areeb_uid_orth_0p50e4": push(areeb, ax_uid_orth, 0.50e-4),
        "S15_areeb_cb_orth_0p35e4": push(areeb, ax_cb_orth, 0.35e-4),
        "S16_areeb_fb_orth_0p30e4": push(areeb, ax_fb_orth, 0.30e-4),
        # dual: centroid then proprietary micro
        "S17_centAG_uid_0p30e4": push(expit(0.5 * (L(areeb) + L(gpw))), ax_uid_orth, 0.30e-4),
        "S18_areeb_uid_cb_0p20_0p15": push(
            push(areeb, ax_uid_orth, 0.20e-4), ax_cb_orth, 0.15e-4
        ),
        # pullback blend (stability for private)
        "S19_areeb065_ATH035": expit(0.65 * L(areeb) + 0.35 * L(ath)),
        "S20_areeb080_gpw020": expit(0.80 * L(areeb) + 0.20 * L(gpw)),
    }

    manifest = []
    for name, p in cands.items():
        path = write_sub(name, p, ids)
        # diversity vs areeb / ath
        rho_a = float(np.corrcoef(U(p), U(areeb))[0, 1])
        rho_t = float(np.corrcoef(U(p), U(ath))[0, 1])
        manifest.append(
            {
                "name": name,
                "path": str(path),
                "rho_areeb": rho_a,
                "rho_ath": rho_t,
                "mean": float(np.mean(p)),
                "std": float(np.std(p)),
            }
        )
        print(f"wrote {name} rho_areeb={rho_a:.6f} rho_ath={rho_t:.6f}")

    # priority order for today's 10 slots (public lock first, then mixes, then proprietary)
    priority = [
        "S01_PUB_areeb_95567",
        "S02_PUB_goodpjw_95562",
        "S04_cent_areeb_gpw",
        "S07_areeb_along_gpw_0p20e4",
        "S13_areeb_uid_orth_0p25e4",
        "S14_areeb_uid_orth_0p50e4",
        "S15_areeb_cb_orth_0p35e4",
        "S17_centAG_uid_0p30e4",
        "S10_ATH_along_areeb_0p50e4",
        "S19_areeb065_ATH035",
        "S08_areeb_along_gpw_0p40e4",
        "S18_areeb_uid_cb_0p20_0p15",
        "S03_PUB_asterios_95552",
        "S20_areeb080_gpw020",
        "S11_ATH_along_areeb_1p00e4",
        "S16_areeb_fb_orth_0p30e4",
        "S05_cent_areeb_gpw_ast",
        "S09_areeb_along_ast_0p25e4",
        "S12_ATH_along_gpw_0p50e4",
        "S06_cent_areeb_gpw_ATH",
    ]
    meta = {
        "ATH_ref": "sub_v39_MAX_uid_peak_a0365.csv (0.95550)",
        "public": {
            "areeb": "0.95567 claimed",
            "goodpjw": "0.95562 claimed",
            "asterios": "0.95552 claimed",
        },
        "priority": priority,
        "candidates": manifest,
        "daily_cap_assumed": 10,
        "NO_SUBMIT_default": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"n": len(cands), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
