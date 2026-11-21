#!/usr/bin/env python3
"""GM V17.9 — Submit R2@0.020 Slot5 + clean R1 F2 leak → R12_clean NOSUBMIT.

0 GPU pods. Step1 submits to Kaggle; Step2 is NO_SUBMIT staging only.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import rankdata, spearmanr, t as tdist, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
TMP15 = Path("/tmp/s6e11_v15")
TMP16 = Path("/tmp/s6e11_v16")
OUT = TMP16 / "out"
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
DATA = REPO / "data"
ORIG = REPO / "original/credit_card_fraud_2026.csv"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT1 = SUB / "sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
RIDGE_ALPHA = 10.0
LAMBDA0 = 20.0
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
NUM_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
    "txn_count_last_24h",
    "card_age_months",
    "customer_age",
    "cvv_retry_count",
]
Q1000_KEYS = {
    "amount_usd_q1000",
    "account_balance_usd_q1000",
    "distance_from_home_km_q1000",
    "velocity_score_q1000",
}


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
    leak_arg = int(np.argmax(np.abs(leaks)))
    c_live = float(np.corrcoef(orth_t, live_resid)[0, 1]) if live_resid.std() > 1e-12 else 0.0
    return {
        "disp": disp,
        "max_basis_leak": max_leak,
        "leak_argmax": leak_arg,
        "leak_by_col": {BN9[j]: leaks[j] for j in range(9)},
        "corr_live_resid": c_live,
        "F1": 0.94 <= disp <= 1.06,
        "F2": max_leak < 0.035,
        "F3": abs(c_live) < 0.05,
        "verified": (0.94 <= disp <= 1.06) and (max_leak < 0.035) and (abs(c_live) < 0.05),
    }


def score_fixed(u, zbase, y, folds, w_grid=(0.01, 0.015, 0.02, 0.025, 0.03)):
    u = unitize(u)
    base = [roc_auc_score(y[te], zbase[te]) for _, te in folds]
    best = None
    for w in w_grid:
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
    try:
        _, wp = wilcoxon(d25, alternative="greater")
        wp = float(wp)
    except Exception:
        wp = float("nan")
    best.update({"sf25": int((d25 > 0).sum()), "t": float(t), "wilcoxon_p": wp})
    return best


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


def newton_delta(keys_tr_tr, g_tr, h_tr, keys_map, lam=LAMBDA0):
    nbin = int(max(keys_tr_tr.max(), keys_map.max()) + 1)
    sg = np.bincount(keys_tr_tr, weights=g_tr, minlength=nbin)
    sh = np.bincount(keys_tr_tr, weights=h_tr, minlength=nbin)
    cnt = np.bincount(keys_tr_tr, minlength=nbin)
    delta = sg / (sh + lam)
    delta[cnt == 0] = 0.0
    return delta[keys_map]


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

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

    z_s3 = z_ath + 0.01045 * dE_o + 0.0055 * unitize(d1)

    ms_o = np.load(REPO / "preds/P_super_v4_oof.npy").ravel()
    ms_t = np.load(REPO / "preds/P_super_v4_test.npy").ravel()
    g26_o = np.load(G26_OOF).ravel()[:n_tr]
    g26_t = np.load(G26_T0).ravel()[:n_te]
    lm_o, lm_t = XO["megayak_5f_lm"], XT["megayak_5f_lm"]
    gpd_o, gpd_t = XO["daoviet_gpdvae"], XT["daoviet_gpdvae"]
    gb_o, gb_t = XO["gb_bag5"], XT["gb_bag5"]
    mlp_o, mlp_t = XO["yekenot_realmlp"], XT["yekenot_realmlp"]

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
    resid_t = L(slot3) - locked_t
    w_nnls, _ = nnls(Bt_bank, resid_t - resid_t.mean())
    z_nnls = W_D1_LOCK * unitize(d1) + W_FE_LOCK * unitize(dE_o) + Bo_bank @ w_nnls
    baselines = {"z_S3": z_s3, "z_Slot3_twin": z_slot3_twin, "z_NNLS_twin": z_nnls}
    live_resid = L(lv) - Ridge(alpha=RIDGE_ALPHA).fit(Bt9, L(lv)).predict(Bt9)

    # ----- Step 1: R2 orth from saved residuals -----
    print("=== STEP 1: build + verify Slot5 CSV ===")
    res = np.load(OUT / "v17_8_residuals.npz")
    R2_oof, R2_test = res["R2_oof"], res["R2_test"]
    oo2, ot2, _ = frozen_lofo_orth(R2_oof, R2_test, Bo9, Bt9, folds, ridge_alpha=10.0)
    u2 = unitize(oo2)
    if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u2[te0]) < roc_auc_score(
        y[te0], z_s3[te0] - 0.01 * u2[te0]
    ):
        u2, ot2 = -u2, -ot2
    ut2 = unitize(ot2)
    gates2 = forensic_gates(u2, ut2, Bt9, live_resid)
    print(f"  R2 forensic ver={gates2['verified']} F2={gates2['max_basis_leak']:.4f}")

    # confirm twin @0.020
    for bname, zb in baselines.items():
        row = score_fixed(u2, zb, y, folds, w_grid=(0.020, 0.025))
        print(
            f"  R2@{bname}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
            f"min={row['min']:+.3f} sf={row['sf25']} t={row['t']:.2f}"
        )

    w_live = 0.020
    z_live = L(slot3) + w_live * ut2
    cand = qlock(z_live, lv)  # qlock onto LIVE sort as requested
    path = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
    pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)

    assert len(cand) == 299984 and np.isfinite(cand).all()
    rho_s3 = float(spearmanr(slot3, cand).statistic)
    rho_s1 = float(spearmanr(slot1, cand).statistic)
    rho_lv = float(spearmanr(lv, cand).statistic)
    # also v17.8 w=0.025 file if present
    p025 = SUB / "sub_v17_8_R2_residual_LGB_w0p025_NOSUBMIT.csv"
    rho025 = None
    if p025.exists():
        c025 = pd.read_csv(p025).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
        rho025 = float(spearmanr(slot3, c025).statistic)
    step1 = {
        "file": path.name,
        "rows": 299984,
        "nans": 0,
        "w": w_live,
        "1mrho_Slot3_e4": (1 - rho_s3) * 1e4,
        "1mrho_Slot1_e4": (1 - rho_s1) * 1e4,
        "1mrho_LIVE_e4": (1 - rho_lv) * 1e4,
        "v178_w025_1mrho_Slot3_e4": None if rho025 is None else (1 - rho025) * 1e4,
        "gates": gates2,
    }
    print(
        f"  VERIFY rows=299984 nans=0 1-ρ Slot3/Slot1/LIVE e-4 = "
        f"{step1['1mrho_Slot3_e4']:.4f}/{step1['1mrho_Slot1_e4']:.4f}/{step1['1mrho_LIVE_e4']:.4f}"
    )
    if rho025 is not None:
        print(f"  v17.8 w0.025 1-ρ(Slot3) e-4 = {step1['v178_w025_1mrho_Slot3_e4']:.4f}")
    print("CSV_READY", path)

    # ----- Step 2: R1_pure without q1000 -----
    print("=== STEP 2: R1_pure (drop q1000) ===")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    # calib
    ab = []
    for tr, va in folds:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s3[tr].reshape(-1, 1), y[tr])
        ab.append((float(clf.coef_[0, 0]), float(clf.intercept_[0])))

    keys_tr, keys_te = {}, {}
    for c in NUM_COLS:
        vtr, vte = train[c].to_numpy(float), test[c].to_numpy(float)
        keys_tr[f"{c}_cents"] = (np.round(vtr * 100).astype(np.int64) % 100) % 100
        keys_te[f"{c}_cents"] = (np.round(vte * 100).astype(np.int64) % 100) % 100
        keys_tr[f"{c}_tenth"] = (np.round(vtr * 10).astype(np.int64) % 10) % 10
        keys_te[f"{c}_tenth"] = (np.round(vte * 10).astype(np.int64) % 10) % 10
        keys_tr[f"{c}_ones"] = (np.abs(vtr).astype(np.int64) % 10) % 10
        keys_te[f"{c}_ones"] = (np.abs(vte).astype(np.int64) % 10) % 10

    amt_all = np.round(
        np.concatenate([train["amount_usd"].to_numpy(float), test["amount_usd"].to_numpy(float)])
        * 100
    ).astype(np.int64)
    uniq, inv = np.unique(amt_all, return_inverse=True)
    cnt = np.bincount(inv)
    freq = np.log1p(cnt[inv])
    keys_tr["amt_freq_bin"] = np.clip((freq[:n_tr] * 5).astype(np.int64), 0, 50)
    keys_te["amt_freq_bin"] = np.clip((freq[n_tr:] * 5).astype(np.int64), 0, 50)

    if ORIG.exists():
        orig = pd.read_csv(ORIG)
        oc = [c for c in orig.columns if "amount" in c.lower() or c.lower() == "amt"]
        if oc:
            from collections import Counter

            oa = np.round(orig[oc[0]].to_numpy(float) * 100).astype(np.int64)
            ctr = Counter(oa.tolist())
            otr = np.array([ctr.get(int(x), 0) for x in amt_all[:n_tr]], dtype=np.int64)
            ote = np.array([ctr.get(int(x), 0) for x in amt_all[n_tr:]], dtype=np.int64)
            keys_tr["amt_orig_bin"] = np.clip(otr, 0, 20)
            keys_te["amt_orig_bin"] = np.clip(ote, 0, 20)

    for cat in ("merchant_category", "channel", "device_type"):
        codes, _ = pd.factorize(pd.concat([train[cat], test[cat]], ignore_index=True))
        keys_tr[f"{cat}_x_amtones"] = codes[:n_tr] * 10 + keys_tr["amount_usd_ones"]
        keys_te[f"{cat}_x_amtones"] = codes[n_tr:] * 10 + keys_te["amount_usd_ones"]

    # also build q1000 for leak diagnosis (per-key orth)
    def joint_q1000(col):
        v = np.concatenate([train[col].to_numpy(float), test[col].to_numpy(float)])
        r = rankdata(v, method="average")
        q = np.clip((r / (len(v) + 1e-12) * 1000).astype(np.int64), 0, 999)
        return q[:n_tr], q[n_tr:]

    q1000_tr, q1000_te = {}, {}
    for c in ("amount_usd", "account_balance_usd", "distance_from_home_km", "velocity_score"):
        qtr, qte = joint_q1000(c)
        q1000_tr[f"{c}_q1000"] = qtr
        q1000_te[f"{c}_q1000"] = qte

    pure_names = [k for k in keys_tr if k not in Q1000_KEYS]
    print(f"  pure keys={len(pure_names)} (excluded q1000)")

    # Diagnose per-key leak of full R1-style single-key TE (alpha=10)
    print("  per-key F2 diagnosis (incl q1000):")
    key_leak = {}
    for kn, ktr, kte in (
        [(k, keys_tr[k], keys_te[k]) for k in pure_names]
        + [(k, q1000_tr[k], q1000_te[k]) for k in q1000_tr]
    ):
        vo = np.zeros(n_tr)
        vt_folds = []
        for f, (tr, va) in enumerate(folds):
            a, b = ab[f]
            p = expit(a * z_s3 + b)
            g, h = y - p, np.clip(p * (1 - p), 1e-6, None)
            vo[va] = newton_delta(ktr[tr], g[tr], h[tr], ktr[va])
            vt_folds.append(newton_delta(ktr[tr], g[tr], h[tr], kte))
        vt = np.mean(np.stack(vt_folds, 0), 0)
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=10.0)
        g = forensic_gates(oo, ot, Bt9, live_resid)
        key_leak[kn] = {
            "max_basis_leak": g["max_basis_leak"],
            "leak_col": BN9[g["leak_argmax"]],
            "disp": g["disp"],
        }
    top_leak = sorted(key_leak.items(), key=lambda kv: -kv[1]["max_basis_leak"])[:12]
    for kn, info in top_leak:
        print(f"    {kn}: leak={info['max_basis_leak']:.4f} ({info['leak_col']})")

    # R1_pure
    R1p_oof = np.zeros(n_tr)
    R1p_te_folds = []
    for f, (tr, va) in enumerate(folds):
        a, b = ab[f]
        p = expit(a * z_s3 + b)
        g, h = y - p, np.clip(p * (1 - p), 1e-6, None)
        Xtr, Xva, Xte = [], [], []
        for kn in pure_names:
            Xtr.append(newton_delta(keys_tr[kn][tr], g[tr], h[tr], keys_tr[kn][tr]))
            Xva.append(newton_delta(keys_tr[kn][tr], g[tr], h[tr], keys_tr[kn][va]))
            Xte.append(newton_delta(keys_tr[kn][tr], g[tr], h[tr], keys_te[kn]))
        Xtr, Xva, Xte = map(np.column_stack, (Xtr, Xva, Xte))
        sw = np.sqrt(h[tr])
        lr = Ridge(alpha=100.0, fit_intercept=True)
        lr.fit(Xtr * sw[:, None], g[tr] * sw)
        R1p_oof[va] = lr.predict(Xva)
        R1p_te_folds.append(lr.predict(Xte))
    R1p_test = np.mean(np.stack(R1p_te_folds, 0), 0)

    # Also residualize each TE column against Bo9 before combining (extra hygiene)
    R1r_oof = np.zeros(n_tr)
    R1r_te_folds = []
    for f, (tr, va) in enumerate(folds):
        a, b = ab[f]
        p = expit(a * z_s3 + b)
        g, h = y - p, np.clip(p * (1 - p), 1e-6, None)
        Xtr, Xva, Xte = [], [], []
        for kn in pure_names:
            dtr = newton_delta(keys_tr[kn][tr], g[tr], h[tr], keys_tr[kn][tr])
            dva = newton_delta(keys_tr[kn][tr], g[tr], h[tr], keys_tr[kn][va])
            dte = newton_delta(keys_tr[kn][tr], g[tr], h[tr], keys_te[kn])
            # residualize vs Bo9 on tr
            rr = Ridge(alpha=1.0, fit_intercept=True).fit(Bo9[tr], dtr)
            Xtr.append(dtr - rr.predict(Bo9[tr]))
            Xva.append(dva - rr.predict(Bo9[va]))
            Xte.append(dte - rr.predict(Bt9))  # use Bt9 for test
        Xtr, Xva, Xte = map(np.column_stack, (Xtr, Xva, Xte))
        sw = np.sqrt(h[tr])
        lr = Ridge(alpha=100.0, fit_intercept=True)
        lr.fit(Xtr * sw[:, None], g[tr] * sw)
        R1r_oof[va] = lr.predict(Xva)
        R1r_te_folds.append(lr.predict(Xte))
    R1r_test = np.mean(np.stack(R1r_te_folds, 0), 0)

    def eval_sig(name, vo, vt, ridge_alpha):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=ridge_alpha)
        gates = forensic_gates(oo, ot, Bt9, live_resid)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s3[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        per = {b: score_fixed(u, zb, y, folds) for b, zb in baselines.items()}
        dual = (
            per["z_S3"]["pos"] >= 4
            and per["z_S3"]["mean"] > 0
            and per["z_Slot3_twin"]["pos"] >= 4
            and per["z_Slot3_twin"]["mean"] > 0
            and per["z_S3"]["sf25"] >= 18
            and per["z_S3"]["t"] >= 2.20
            and per["z_Slot3_twin"]["sf25"] >= 18
            and per["z_Slot3_twin"]["t"] >= 2.20
        )
        print(
            f"  {name} α={ridge_alpha}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{BN9[gates['leak_argmax']]}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) ver={gates['verified']} dual={dual}"
        )
        for b, row in per.items():
            print(
                f"    {b}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
                f"min={row['min']:+.3f} sf={row['sf25']} t={row['t']:.2f}"
            )
        return {
            "gates": gates,
            "per_base": per,
            "dual": dual,
            "u": u,
            "ut": ut,
            "pre_corr": float(np.corrcoef(vo, z_s3)[0, 1]),
        }

    r1_a10 = eval_sig("R1_pure", R1p_oof, R1p_test, 10.0)
    r1_a01 = eval_sig("R1_pure", R1p_oof, R1p_test, 0.1)
    r1r_a10 = eval_sig("R1_residBo9", R1r_oof, R1r_test, 10.0)
    r1r_a01 = eval_sig("R1_residBo9", R1r_oof, R1r_test, 0.1)

    # R12_clean = best clean R1 + R2
    # prefer verified R1 variant
    r1_best_name, r1_best = "R1_pure_a0.1", r1_a01
    for nm, rr in (
        ("R1_pure_a10", r1_a10),
        ("R1_pure_a0.1", r1_a01),
        ("R1_residBo9_a10", r1r_a10),
        ("R1_residBo9_a0.1", r1r_a01),
    ):
        if rr["gates"]["verified"] and (
            not r1_best["gates"]["verified"]
            or rr["per_base"]["z_S3"]["mean"] > r1_best["per_base"]["z_S3"]["mean"]
        ):
            r1_best_name, r1_best = nm, rr

    # R12 always: unitize(R1_pure raw)+unitize(R2) then orth α=0.1
    r12 = eval_sig(
        "R12_clean",
        unitize(R1p_oof) + unitize(R2_oof),
        unitize(R1p_test) + unitize(R2_test),
        0.1,
    )
    r12b = eval_sig(
        "R12_clean",
        unitize(R1p_oof) + unitize(R2_oof),
        unitize(R1p_test) + unitize(R2_test),
        10.0,
    )
    r12r = eval_sig(
        "R12_residR1_R2",
        unitize(R1r_oof) + unitize(R2_oof),
        unitize(R1r_test) + unitize(R2_test),
        0.1,
    )

    staged = []
    # stage best R12 that is verified+dual, else best dual even if soft F2, else R1_best
    for tag, rr in (
        ("R12_clean_a0p1", r12),
        ("R12_clean_a10", r12b),
        ("R12_residR1_R2_a0p1", r12r),
        (r1_best_name, r1_best),
    ):
        if rr["gates"]["verified"] and rr["dual"]:
            wstar = rr["per_base"]["z_Slot3_twin"]["w"]
            path2 = SUB / f"sub_v17_9_{tag}_NOSUBMIT.csv"
            cand2 = qlock(L(slot3) + wstar * rr["ut"], slot3)
            pd.DataFrame({"id": ids, "is_fraud": cand2}).to_csv(path2, index=False)
            staged.append({"file": path2.name, "tag": tag, "w": wstar, "verified": True})
            print("STAGED", path2.name)
            break
    else:
        # stage best dual anyway as diagnostic NOSUBMIT
        tag, rr = "R12_clean_a0p1", r12
        wstar = rr["per_base"]["z_Slot3_twin"]["w"]
        path2 = SUB / "sub_v17_9_R12_clean_NOSUBMIT.csv"
        cand2 = qlock(L(slot3) + wstar * rr["ut"], slot3)
        pd.DataFrame({"id": ids, "is_fraud": cand2}).to_csv(path2, index=False)
        staged.append(
            {
                "file": path2.name,
                "tag": tag,
                "w": wstar,
                "verified": rr["gates"]["verified"],
                "dual": rr["dual"],
            }
        )
        print("STAGED_FALLBACK", path2.name, "ver", rr["gates"]["verified"], "dual", rr["dual"])

    payload = scrub(
        {
            "step1": step1,
            "key_leak_top": [{k: v} for k, v in top_leak],
            "R1_pure_a10": {k: r1_a10[k] for k in ("gates", "per_base", "dual", "pre_corr")},
            "R1_pure_a0.1": {k: r1_a01[k] for k in ("gates", "per_base", "dual", "pre_corr")},
            "R1_residBo9_a10": {k: r1r_a10[k] for k in ("gates", "per_base", "dual", "pre_corr")},
            "R1_residBo9_a0.1": {k: r1r_a01[k] for k in ("gates", "per_base", "dual", "pre_corr")},
            "R12_clean_a0.1": {k: r12[k] for k in ("gates", "per_base", "dual", "pre_corr")},
            "R12_clean_a10": {k: r12b[k] for k in ("gates", "per_base", "dual", "pre_corr")},
            "R12_residR1_R2_a0.1": {k: r12r[k] for k in ("gates", "per_base", "dual", "pre_corr")},
            "staged_nosubmit": staged,
            "elapsed_s": time.time() - t0,
        }
    )
    # drop large arrays from nested
    for k in list(payload.keys()):
        if isinstance(payload[k], dict) and "u" in payload[k]:
            payload[k].pop("u", None)
            payload[k].pop("ut", None)
    OUT.joinpath("v17_9.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v17_9.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE_STEP2")
    return path


if __name__ == "__main__":
    main()
