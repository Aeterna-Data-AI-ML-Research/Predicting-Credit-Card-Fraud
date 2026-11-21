#!/usr/bin/env python3
"""GM V17.7 — Dual-baseline invariance audit (fixed sign, no per-base flip).

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU only.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata, t as tdist, wilcoxon
from sklearn.linear_model import Ridge
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
SLOT1 = SUB / "sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
NO_SUBMIT = True
RIDGE_ALPHA = 10.0
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_GRID = (0.005, 0.010, 0.015, 0.020)
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045


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


def L_fold(raw, fold_id):
    out = np.zeros_like(raw, dtype=np.float64)
    for f in range(5):
        m = fold_id == f
        out[m] = L(raw[m])
    return out


def col_z(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-12)


def gini10(Z):
    K = Z.shape[-1]
    s = np.zeros(Z.shape[:-1], dtype=np.float64)
    for i, j in combinations(range(K), 2):
        s += np.abs(Z[..., i] - Z[..., j])
    return s / (K * (K - 1) / 2.0)


def frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=RIDGE_ALPHA):
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


def forensic_gates(orth_o, orth_t, Bt, live_resid):
    disp = float((orth_t.std() + 1e-12) / (orth_o.std() + 1e-12))
    leaks = [float(np.corrcoef(orth_t, Bt[:, j])[0, 1]) for j in range(Bt.shape[1])]
    max_leak = float(np.max(np.abs(leaks)))
    c_live = float(np.corrcoef(orth_t, live_resid)[0, 1]) if live_resid.std() > 1e-12 else 0.0
    return {
        "disp": disp,
        "max_basis_leak": max_leak,
        "corr_live_resid": c_live,
        "F1": 0.94 <= disp <= 1.06,
        "F2": max_leak < 0.035,
        "F3": abs(c_live) < 0.05,
        "verified": (0.94 <= disp <= 1.06) and (max_leak < 0.035) and (abs(c_live) < 0.05),
    }


def score_fixed(u, zbase, y, folds, w):
    """Score +w*u with FIXED u (no flip)."""
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
    try:
        _, wp = wilcoxon(d25, alternative="greater")
        wp = float(wp)
    except Exception:
        wp = float("nan")
    return {
        "w": w,
        "pos": int(sum(x > 0 for x in d)),
        "mean": float(np.mean(d)),
        "min": float(np.min(d)),
        "folds": [round(x, 3) for x in d],
        "sf25": int((d25 > 0).sum()),
        "t": float(t),
        "wilcoxon_p": wp,
    }


def best_w_row(u, zbase, y, folds):
    rows = [score_fixed(u, zbase, y, folds, w) for w in W_GRID]
    # prefer pos, then mean
    rows.sort(key=lambda r: (r["pos"], r["mean"]), reverse=True)
    return rows[0], rows


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


def main():
    assert NO_SUBMIT
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
    dE_o = unitize(cache["dE"])
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    te0 = folds[0][1]

    Z = np.load(Q17, allow_pickle=True)
    qnames = [str(s) for s in Z["stream_names"]]
    XO = {n: Z[f"oof_{n}"].ravel() for n in qnames}
    XT = {n: Z[f"test_{n}"].ravel() for n in qnames}

    live = pd.read_csv(LIVE)
    ids = live["id"].to_numpy()
    lv = live["is_fraud"].to_numpy(float)
    slot1 = pd.read_csv(SLOT1).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

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

    # z_S3 and Slot1-family bases for d1 micro
    z_s1 = z_ath + 0.01045 * dE_o  # FamilyE layer like Slot1 path
    z_s3 = z_s1 + 0.0055 * unitize(d1)
    print(f"AUC z_S3={roc_auc_score(y,z_s3):.7f}")

    ms_o = np.load(REPO / "preds/P_super_v4_oof.npy").ravel()
    ms_t = np.load(REPO / "preds/P_super_v4_test.npy").ravel()
    g26_o = np.load(G26_OOF).ravel()[: len(y)]
    g26_t = np.load(G26_T0).ravel()[:299984]
    lm_o, lm_t = XO["megayak_5f_lm"], XT["megayak_5f_lm"]
    gpd_o, gpd_t = XO["daoviet_gpdvae"], XT["daoviet_gpdvae"]
    gb_o, gb_t = XO["gb_bag5"], XT["gb_bag5"]
    mlp_o, mlp_t = XO["yekenot_realmlp"], XT["yekenot_realmlp"]

    z_MS_o, z_MS_t = L_fold(ms_o, fold_id), L(ms_t)
    z_G26_o, z_G26_t = L_fold(g26_o, fold_id), L(g26_t)
    z_LM_o, z_LM_t = L_fold(lm_o, fold_id), L(lm_t)
    z_L7F_o, z_L7F_t = L_fold(l7f_o, fold_id), L(l7f_t)
    z_R3_o, z_R3_t = L_fold(R3, fold_id), L(R3_t_L.mean(0))
    z_GB_o, z_GB_t = L_fold(gb_o, fold_id), L(gb_t)
    z_MLP_o, z_MLP_t = L_fold(mlp_o, fold_id), L(mlp_t)
    z_GPD_o, z_GPD_t = L_fold(gpd_o, fold_id), L(gpd_t)

    Bo9 = np.column_stack(
        [
            unitize(L(ms_o)),
            unitize(L(g26_o)),
            unitize(L(lm_o)),
            unitize(L(l7f_o)),
            unitize(gpd_o),
            unitize(gb_o),
            unitize(mlp_o),
            unitize(d1),
            unitize(dE_o),
        ]
    )
    Bt9 = np.column_stack(
        [
            unitize(L(ms_t)),
            unitize(L(g26_t)),
            unitize(L(lm_t)),
            unitize(L(l7f_t)),
            unitize(gpd_t),
            unitize(gb_t),
            unitize(mlp_t),
            unitize(d1t),
            unitize(dE_t),
        ]
    )

    # --- baselines ---
    # A: z_S3
    # B: empirical Slot3 twin (the one that predicted LB drop) Ridge α=1
    y_s3t = unitize(L(slot3))
    ridge_s3 = Ridge(alpha=1.0, fit_intercept=True).fit(Bt9, y_s3t)
    z_slot3_twin = Bo9 @ ridge_s3.coef_ + ridge_s3.intercept_
    r2_s3 = float(ridge_s3.score(Bt9, y_s3t))
    print(f"Slot3 Ridge twin R²={r2_s3:.4f} AUC={roc_auc_score(y,z_slot3_twin):.7f}")
    print(f"  beta={ {BN9[j]: round(float(ridge_s3.coef_[j]),4) for j in range(9)} }")

    # C: NNLS twin (V17.6 fix)
    bank_o, bank_t, bank_n = [], [], []

    def add(n, o, t):
        bank_n.append(n)
        bank_o.append(unitize(o))
        bank_t.append(unitize(t))

    for j, nm in enumerate(BN9):
        if nm in ("d1_v4", "Family_E"):
            continue
        add(f"Bo9_{nm}", Bo9[:, j], Bt9[:, j])
    skip = {
        "megayak_5f_lm",
        "lm7_5fold",
        "gpt2_seed2026",
        "masaya_gpt2",
        "daoviet_gpdvae",
        "gb_bag5",
        "yekenot_realmlp",
    }
    for n in qnames:
        if n not in skip:
            add(f"Q17_{n}", XO[n], XT[n])
    for p in sorted(ALEX.glob("oof_*.npy"))[:12]:
        tp = ALEX / ("test_" + p.name[len("oof_") :])
        if tp.exists():
            add(f"alex_{p.stem[-8:]}", np.load(p).ravel()[: len(y)], np.load(tp).ravel()[:299984])
    Bo_bank = np.column_stack(bank_o)
    Bt_bank = np.column_stack(bank_t)
    locked_t = W_D1_LOCK * unitize(d1t) + W_FE_LOCK * unitize(dE_t)
    resid_t = L(slot3) - locked_t
    w_nnls, _ = nnls(Bt_bank, resid_t - resid_t.mean())
    z_nnls = W_D1_LOCK * unitize(d1) + W_FE_LOCK * unitize(dE_o) + Bo_bank @ w_nnls
    print(f"NNLS twin AUC={roc_auc_score(y,z_nnls):.7f}")

    baselines = {
        "z_S3": z_s3,
        "z_Slot3_twin": unitize(z_slot3_twin),
        "z_NNLS_twin": z_nnls,
    }

    live_L = L(lv)
    live_hat = Ridge(alpha=RIDGE_ALPHA).fit(Bt9, live_L).predict(Bt9)
    live_resid = live_L - live_hat

    # --- signals ---
    print("=== build signals ===")
    P_BPE_o = unitize(0.55 * z_MS_o + 0.45 * z_G26_o)
    P_BPE_t = unitize(0.55 * z_MS_t + 0.45 * z_G26_t)
    P_QLM_o = unitize(0.45 * z_LM_o + 0.30 * z_L7F_o + 0.25 * z_R3_o)
    P_QLM_t = unitize(0.45 * z_LM_t + 0.30 * z_L7F_t + 0.25 * z_R3_t)
    P_TAB_o = unitize(0.50 * z_GB_o + 0.35 * z_MLP_o + 0.15 * z_GPD_o)
    P_TAB_t = unitize(0.50 * z_GB_t + 0.35 * z_MLP_t + 0.15 * z_GPD_t)

    d_all_o = unitize(L_fold(P_BPE_o - P_QLM_o, fold_id))
    d_all_t = unitize(L(P_BPE_t - P_QLM_t))
    # G1e as staged: d * unitize(z_S3) then orth — Slot4 used this + direction
    g1e_o = d_all_o * unitize(z_s3)
    g1e_t = d_all_t * unitize(L(slot3))
    oo_g1e, ot_g1e, _ = frozen_lofo_orth(g1e_o, g1e_t, Bo9, Bt9, folds)
    # Orient G1e to the Slot4 / V17.4 winning direction on z_S3 fold0
    if roc_auc_score(y[te0], z_s3[te0] + 0.01 * oo_g1e[te0]) < roc_auc_score(
        y[te0], z_s3[te0] - 0.01 * oo_g1e[te0]
    ):
        oo_g1e, ot_g1e = -oo_g1e, -ot_g1e
    u_g1e = unitize(oo_g1e)  # +G1e = Slot4 direction

    # G1f family
    d_bt_o = unitize(L_fold(P_BPE_o - P_TAB_o, fold_id))
    d_bt_t = unitize(L(P_BPE_t - P_TAB_t))
    d_qt_o = unitize(L_fold(P_QLM_o - P_TAB_o, fold_id))
    d_qt_t = unitize(L(P_QLM_t - P_TAB_t))
    raw_bt = (d_bt_o * U(z_s3), d_bt_t * U(L(slot3)))
    raw_qt = (d_qt_o * U(z_s3), d_qt_t * U(L(slot3)))

    # U5 / q80 from L7 raw
    raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
    llr_o = raw_o[:, :, 1] - raw_o[:, :, 0]
    llr_t = raw_t[:, :, :, 1] - raw_t[:, :, :, 0]
    U5_o = L_fold(gini10(col_z(llr_o)), fold_id)
    U5_t = L(np.stack([L(gini10(col_z(llr_t[f]))) for f in range(5)], 0).mean(0))
    z10_o = np.stack([L(llr_o[:, k]) for k in range(10)], 1)
    U1_o = L_fold(np.quantile(z10_o, 0.8, 1) - np.quantile(z10_o, 0.2, 1), fold_id)
    z10_t = np.stack(
        [np.stack([L(llr_t[f, :, k]) for k in range(10)], 1) for f in range(5)], 0
    )
    U1_t = L(
        np.stack(
            [
                L(np.quantile(z10_t[f], 0.8, 1) - np.quantile(z10_t[f], 0.2, 1))
                for f in range(5)
            ],
            0,
        ).mean(0)
    )

    signals = {}  # name -> (u_oof fixed, gates, note)

    def register(name, vo, vt, preorient_on_s3=True):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds)
        gates = forensic_gates(oo, ot, Bt9, live_resid)
        u = unitize(oo)
        if preorient_on_s3:
            # define + as S3-helpful on fold0; BOTH + and - will be tested fixed
            if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u[te0]) < roc_auc_score(
                y[te0], z_s3[te0] - 0.01 * u[te0]
            ):
                u = -u
                ot = -ot
        signals[name] = {"u": u, "gates": gates, "ot": unitize(ot)}
        print(
            f"  {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) ver={gates['verified']}"
        )

    signals["G1e"] = {
        "u": u_g1e,
        "gates": forensic_gates(oo_g1e, ot_g1e, Bt9, live_resid),
        "ot": unitize(ot_g1e if np.corrcoef(u_g1e, oo_g1e)[0, 1] > 0 else -ot_g1e),
    }
    print(
        f"  G1e: ver={signals['G1e']['gates']['verified']} "
        f"F2={signals['G1e']['gates']['max_basis_leak']:.4f} "
        f"F3={signals['G1e']['gates']['corr_live_resid']:+.4f}"
    )
    register("G1f_BPE_TAB", *raw_bt)
    register("G1f_QLM_TAB", *raw_qt)
    register("U5_gini10_llr", U5_o, U5_t)
    register("q80_q20", U1_o, U1_t)

    # d1 micro: additive unitize(d1) already partly in bases — use residual orth of d1
    # Relative to z_s1 (Slot1-like without extra d1 polish): add Δw * unitize(d1)
    # Compare absolute stacks: Slot1-proxy, P1, Slot3-proxy on both baselines
    d1u = unitize(d1)
    # orth d1 against Bo9 without d1 column for fairness? User asks Δw relative to Slot1
    # Evaluate absolute: z_s1 + w*d1u as the baseline itself (not an add-on to z_S3)
    d1_stacks = {
        "Slot1_d1_0p030": z_s1 + 0.030 * d1u,
        "P1_d1_0p033": z_s1 + 0.033 * d1u,
        "Slot3_d1_0p0355": z_s1 + 0.0355 * d1u,
    }
    # Also Δ add-ons on z_S3 / twins: ±Δw * orth(d1) — but d1 is in Bo9 so orth≈0.
    # Instead: Δ relative to Slot3 stack on each baseline: add (w-0.0355)*d1u onto a Slot3-like proxy
    # Practical: score Δw * d1u on top of each baseline with FIXED sign (d1u as-is, + and -)

    register("d1_raw_unitize", d1u, unitize(d1t), preorient_on_s3=True)

    print("=== dual-baseline grid ===")
    table = []
    for name, sig in signals.items():
        # only report forensic-clean for dual gate; still evaluate all for completeness
        for sign, tag in ((+1, "+"), (-1, "-")):
            u = sign * sig["u"]
            row = {
                "signal": name,
                "sign": tag,
                "gates": sig["gates"],
                "forensic": sig["gates"]["verified"],
                "by_base": {},
            }
            for bname, zbase in baselines.items():
                best, allw = best_w_row(u, zbase, y, folds)
                row["by_base"][bname] = {"best": best, "all_w": {str(r["w"]): r for r in allw}}
            s3 = row["by_base"]["z_S3"]["best"]
            tw = row["by_base"]["z_Slot3_twin"]["best"]
            nn = row["by_base"]["z_NNLS_twin"]["best"]
            # Dual pass: same sign (already), pos>=4 and mean>0 on BOTH z_S3 and Slot3_twin
            dual = (
                s3["pos"] >= 4
                and s3["mean"] > 0
                and tw["pos"] >= 4
                and tw["mean"] > 0
            )
            dual3 = dual and nn["pos"] >= 4 and nn["mean"] > 0
            row["dual_pass_S3_Slot3twin"] = dual
            row["dual_pass_all3"] = dual3
            table.append(row)
            print(
                f"  {tag}{name}: forensic={row['forensic']} dual={dual} dual3={dual3} | "
                f"S3 {s3['pos']}/5@{s3['w']} m={s3['mean']:+.2f} sf{s3['sf25']} t={s3['t']:.2f} | "
                f"Twin {tw['pos']}/5@{tw['w']} m={tw['mean']:+.2f} sf{tw['sf25']} t={tw['t']:.2f} | "
                f"NNLS {nn['pos']}/5@{nn['w']} m={nn['mean']:+.2f} sf{nn['sf25']} t={nn['t']:.2f}"
            )

    # d1 absolute stack comparison (not orth add-on)
    print("=== d1 absolute stacks ===")
    d1_abs = {}
    for nm, z in d1_stacks.items():
        d1_abs[nm] = {
            "auc": float(roc_auc_score(y, z)),
            "vs_S3": {},
        }
        # compare each stack as baseline replacement? Or score as candidate vs z_S3?
        # Report AUC and fold-AUC vs z_S3 / twin as the score itself
        for bname, zbase in baselines.items():
            # delta of using stack instead of baseline? Better: fold AUC of stack alone
            fold_auc = [roc_auc_score(y[te], z[te]) for _, te in folds]
            base_auc = [roc_auc_score(y[te], zbase[te]) for _, te in folds]
            dlt = [(fold_auc[i] - base_auc[i]) * 1e5 for i in range(5)]
            d1_abs[nm]["vs_S3" if bname == "z_S3" else bname] = {
                "pos": int(sum(x > 0 for x in dlt)),
                "mean": float(np.mean(dlt)),
                "min": float(np.min(dlt)),
                "folds": [round(x, 3) for x in dlt],
                "auc_stack": float(roc_auc_score(y, z)),
            }
        print(
            f"  {nm}: AUC={d1_abs[nm]['auc']:.7f} vs_S3 "
            f"{d1_abs[nm]['vs_S3']['pos']}/5 mean={d1_abs[nm]['vs_S3']['mean']:+.3f} "
            f"vs_twin {d1_abs[nm]['z_Slot3_twin']['pos']}/5 "
            f"mean={d1_abs[nm]['z_Slot3_twin']['mean']:+.3f}"
        )

    # Δw * unitize(d1) on top of each baseline (fixed ±d1 direction)
    print("=== d1 Δw micro on baselines ===")
    d1_micro = []
    for u_use, w_abs, tag in (
        (-unitize(d1u), 0.0055, "minus_d1_0p0055_toward_Slot1"),
        (-unitize(d1u), 0.0025, "minus_d1_0p0025_toward_P1"),
        (+unitize(d1u), 0.0030, "plus_d1_0p0030"),
    ):
        row = {"tag": tag, "w": w_abs, "by_base": {}}
        for bname, zbase in baselines.items():
            row["by_base"][bname] = score_fixed(u_use, zbase, y, folds, w_abs)
        s3, tw = row["by_base"]["z_S3"], row["by_base"]["z_Slot3_twin"]
        nn = row["by_base"]["z_NNLS_twin"]
        dual = s3["pos"] >= 4 and s3["mean"] > 0 and tw["pos"] >= 4 and tw["mean"] > 0
        row["dual_pass"] = dual
        d1_micro.append(row)
        print(
            f"  {tag}: dual={dual} S3 {s3['pos']}/5 m={s3['mean']:+.2f} "
            f"Twin {tw['pos']}/5 m={tw['mean']:+.2f} NNLS {nn['pos']}/5 m={nn['mean']:+.2f}"
        )

    dual_winners = [
        r
        for r in table
        if r["forensic"] and r["dual_pass_S3_Slot3twin"]
    ]
    print(f"DUAL WINNERS (forensic∧S3∩twin)={[ (r['sign']+r['signal']) for r in dual_winners ]}")

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "slot4_feedback": {"lb": 0.95387, "slot3": 0.95389, "delta_e5": -2.0},
            "baselines": {
                "z_S3_auc": float(roc_auc_score(y, z_s3)),
                "z_Slot3_twin_auc": float(roc_auc_score(y, z_slot3_twin)),
                "z_Slot3_twin_r2": r2_s3,
                "z_Slot3_twin_beta": {BN9[j]: float(ridge_s3.coef_[j]) for j in range(9)},
                "z_NNLS_auc": float(roc_auc_score(y, z_nnls)),
            },
            "table": table,
            "d1_abs": d1_abs,
            "d1_micro": d1_micro,
            "dual_winners": [r["sign"] + r["signal"] for r in dual_winners],
            "dual_rule": "forensic F1∧F2∧F3 AND same fixed sign AND pos>=4 AND meanΔ>0 on z_S3 AND z_Slot3_twin",
        }
    )
    OUT.joinpath("v17_7.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v17_7.json")
    print("DONE")


if __name__ == "__main__":
    main()
