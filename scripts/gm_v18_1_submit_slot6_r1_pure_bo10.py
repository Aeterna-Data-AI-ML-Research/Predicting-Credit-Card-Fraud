#!/usr/bin/env python3
"""GM V18.1 — Submit Slot6: Slot5 (0.95394) + orth(R1_pure|Bo10) @ w=0.022.

0 GPU pods. Local CPU. Submits to Kaggle.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
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
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W_R2 = 0.020
W_SUBMIT = 0.022
OUT_CSV = SUB / "sub_v18_1_LIVE_slot6_Slot5_plus_R1_pure_Bo10_w0p022.csv"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"


def L(v):
    v = np.asarray(v, float).ravel()
    r = (rankdata(v, method="average") - 0.5) / len(v)
    r = np.clip(r, 1e-12, 1.0 - 1e-12)
    return np.log(r / (1.0 - r))


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def qlock(raw, ref):
    raw, ref = np.asarray(raw, float), np.asarray(ref, float)
    out = np.empty_like(ref)
    out[np.argsort(raw)] = np.sort(ref)
    return out


def frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=10.0):
    vo, vt = unitize(v_oof), unitize(v_test)
    orth_o = np.zeros(len(vo))
    betas = []
    for tr, te in folds:
        lr = Ridge(alpha=ridge_alpha, fit_intercept=True)
        lr.fit(Bo[tr], vo[tr])
        betas.append(lr.coef_.copy())
        orth_o[te] = vo[te] - Bo[te] @ lr.coef_ - lr.intercept_
    beta_bar = np.mean(np.stack(betas, 0), axis=0)
    orth_t = vt - Bt @ beta_bar
    return orth_o - orth_o.mean(), orth_t - orth_t.mean(), beta_bar


def score_at_w(u, zbase, y, folds, w):
    u = unitize(u)
    base = [roc_auc_score(y[te], zbase[te]) for _, te in folds]
    d = [
        (roc_auc_score(y[te], zbase[te] + w * u[te]) - base[i]) * 1e5
        for i, (_, te) in enumerate(folds)
    ]
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
    return {
        "w": w,
        "pos": int(sum(x > 0 for x in d)),
        "mean": float(np.mean(d)),
        "min": float(np.min(d)),
        "folds": [round(x, 3) for x in d],
        "sf25": int((d25 > 0).sum()),
        "t": float(t),
    }


def main():
    t0 = time.time()
    SUB.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
    dE_o = unitize(cache["dE"])
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    te0 = folds[0][1]
    n_tr, n_te = len(y), 299984

    Z = np.load(Q17, allow_pickle=True)
    qnames = [str(s) for s in Z["stream_names"]]
    XO = {n: Z[f"oof_{n}"].ravel() for n in qnames}
    XT = {n: Z[f"test_{n}"].ravel() for n in qnames}

    live = pd.read_csv(LIVE)
    ids = live["id"].to_numpy()
    lv = live["is_fraud"].to_numpy(float)
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot5 = pd.read_csv(SLOT5).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    l7f_o, l7f_t = XO["lm7_5fold"], XT["lm7_5fold"]
    R3_t_L = np.stack([L(R3_te[f]) for f in range(5)], 0)

    z_ath_t = L(lv)
    raw_E_o = (L(R3) - L(l7f_o)) * (0.5 + 0.5 * U(z_ath))
    raw_E_t = (L(R3_t_L.mean(0)) - L(l7f_t)) * (0.5 + 0.5 * U(lv))
    Xo_e = np.column_stack([unitize(z_ath), unitize(d1)])
    Xt_e = np.column_stack([unitize(z_ath_t), unitize(d1t)])
    dE_o2, dE_t, _ = frozen_lofo_orth(raw_E_o, raw_E_t, Xo_e, Xt_e, folds, ridge_alpha=1.0)
    dE_o2, dE_t = unitize(dE_o2), unitize(dE_t)
    if np.corrcoef(dE_o, dE_o2)[0, 1] < 0:
        dE_o = -dE_o
    if roc_auc_score(y[te0], z_ath[te0] + 0.01 * dE_o2[te0]) < roc_auc_score(
        y[te0], z_ath[te0] - 0.01 * dE_o2[te0]
    ):
        dE_t = -dE_t

    z_s3 = z_ath + 0.01045 * dE_o + 0.0055 * unitize(d1)

    ms_o = np.load(REPO / "preds/P_super_v4_oof.npy").ravel()
    ms_t = np.load(REPO / "preds/P_super_v4_test.npy").ravel()
    g26_o = np.load(G26_OOF).ravel()[:n_tr]
    g26_t = np.load(G26_T0).ravel()[:n_te]
    Bo9 = np.column_stack(
        [
            unitize(L(ms_o)),
            unitize(L(g26_o)),
            unitize(L(XO["megayak_5f_lm"])),
            unitize(L(l7f_o)),
            unitize(XO["daoviet_gpdvae"]),
            unitize(XO["gb_bag5"]),
            unitize(XO["yekenot_realmlp"]),
            unitize(d1),
            unitize(dE_o),
        ]
    )
    Bt9 = np.column_stack(
        [
            unitize(L(ms_t)),
            unitize(L(g26_t)),
            unitize(L(XT["megayak_5f_lm"])),
            unitize(L(l7f_t)),
            unitize(XT["daoviet_gpdvae"]),
            unitize(XT["gb_bag5"]),
            unitize(XT["yekenot_realmlp"]),
            unitize(d1t),
            unitize(dE_t),
        ]
    )

    y_s3t = unitize(L(slot3))
    ridge_s3 = Ridge(alpha=1.0, fit_intercept=True).fit(Bt9, y_s3t)
    z_slot3_twin = unitize(Bo9 @ ridge_s3.coef_ + ridge_s3.intercept_)
    bank_o, bank_t = [], []
    skip = {
        "megayak_5f_lm",
        "lm7_5fold",
        "gpt2_seed2026",
        "masaya_gpt2",
        "daoviet_gpdvae",
        "gb_bag5",
        "yekenot_realmlp",
    }
    for j, nm in enumerate(BN9):
        if nm not in ("d1_v4", "Family_E"):
            bank_o.append(unitize(Bo9[:, j]))
            bank_t.append(unitize(Bt9[:, j]))
    for n in qnames:
        if n not in skip:
            bank_o.append(unitize(XO[n]))
            bank_t.append(unitize(XT[n]))
    for p in sorted(ALEX.glob("oof_*.npy"))[:8]:
        tp = ALEX / ("test_" + p.name[len("oof_") :])
        if tp.exists():
            bank_o.append(unitize(np.load(p).ravel()[:n_tr]))
            bank_t.append(unitize(np.load(tp).ravel()[:n_te]))
    Bo_bank = np.column_stack(bank_o)
    Bt_bank = np.column_stack(bank_t)
    locked_t = W_D1_LOCK * unitize(d1t) + W_FE_LOCK * unitize(dE_t)
    w_nnls, _ = nnls(Bt_bank, (L(slot3) - locked_t) - (L(slot3) - locked_t).mean())
    z_nnls = W_D1_LOCK * unitize(d1) + W_FE_LOCK * unitize(dE_o) + Bo_bank @ w_nnls

    res = np.load(OUT / "v18_0_residuals.npz")
    R1p, R1p_t = res["R1p_oof"], res["R1p_test"]
    u2, ut2 = res["u2_oof"], res["u2_test"]
    # re-orient u2 vs z_s3 for safety
    if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u2[te0]) < roc_auc_score(
        y[te0], z_s3[te0] - 0.01 * u2[te0]
    ):
        u2, ut2 = -u2, -ut2
    u2, ut2 = unitize(u2), unitize(ut2)

    z_s5 = z_s3 + W_R2 * u2
    z_slot5_twin = z_slot3_twin + W_R2 * u2
    z_nnls_s5 = z_nnls + W_R2 * u2
    Bo10 = np.column_stack([Bo9, u2])
    Bt10 = np.column_stack([Bt9, ut2])

    oo, ot, _ = frozen_lofo_orth(R1p, R1p_t, Bo10, Bt10, folds, ridge_alpha=10.0)
    u = unitize(oo)
    if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u[te0]) < roc_auc_score(
        y[te0], z_s5[te0] - 0.01 * u[te0]
    ):
        u, ot = -u, -ot
    ut = unitize(ot)

    bases = {"z_S5": z_s5, "z_Slot5_twin": z_slot5_twin, "z_NNLS_S5": z_nnls_s5}
    print("=== STEP 1: w-grid on Slot5 baselines + 1-ρ(Slot5) ===")
    grid = {}
    for w in (0.020, 0.022, 0.025, 0.030):
        cand = qlock(L(slot5) + w * ut, lv)
        one_m_rho = (1.0 - float(spearmanr(cand, slot5).correlation)) * 1e4
        per = {b: score_at_w(u, zb, y, folds, w) for b, zb in bases.items()}
        grid[w] = {"one_minus_rho_e4": one_m_rho, "per_base": per}
        print(
            f"  w={w:.3f}  1-ρ={one_m_rho:.3f}e-4  | "
            + " | ".join(
                f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} "
                f"min={per[b]['min']:+.2f} sf{per[b]['sf25']} t={per[b]['t']:.2f}"
                for b in bases
            )
        )

    # build submit @ 0.022
    cand = qlock(L(slot5) + W_SUBMIT * ut, lv)
    assert len(cand) == n_te
    assert np.isfinite(cand).all()
    assert not np.isnan(cand).any()
    pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(OUT_CSV, index=False)
    rho = float(spearmanr(cand, slot5).correlation)
    print(
        f"WROTE {OUT_CSV.name} rows={len(cand)} nans=0 "
        f"1-ρ={((1-rho)*1e4):.3f}e-4 vs Slot5  t={time.time()-t0:.1f}s"
    )

    # Step 2: submit
    msg = (
        "v18.1 Slot6: Slot5 (0.95394) + orth(R1_pure | Bo10_S5) Newton TE "
        "(w=0.022, 5/5 all 3 bases, min_fold=+4.5e-5, t=4.53, F1/F2/F3 pass)"
    )
    print("=== STEP 2: submit ===")
    r = subprocess.run(
        [
            KAG,
            "competitions",
            "submit",
            "-c",
            "playground-series-s6e11",
            "-f",
            str(OUT_CSV),
            "-m",
            msg,
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    print(r.stdout)
    print(r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"submit failed rc={r.returncode}")

    payload = {
        "w_submit": W_SUBMIT,
        "file": OUT_CSV.name,
        "grid": {
            str(w): {
                "one_minus_rho_e4": grid[w]["one_minus_rho_e4"],
                "per_base": grid[w]["per_base"],
            }
            for w in grid
        },
        "elapsed_s": time.time() - t0,
        "submit_stdout": r.stdout,
    }
    # json-safe floats
    def scrub(x):
        if isinstance(x, dict):
            return {k: scrub(v) for k, v in x.items()}
        if isinstance(x, list):
            return [scrub(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return float(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
        return x

    OUT.joinpath("v18_1.json").write_text(json.dumps(scrub(payload), indent=2))
    print("SUBMIT_OK", f"elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
