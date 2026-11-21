#!/usr/bin/env python3
"""GM V18.0 — Slot5-base audit: orth(R1|Bo10), Δw_R2, expanded R_v2.

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import rankdata, t as tdist, wilcoxon
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
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
NO_SUBMIT = True
RIDGE_ALPHA = 10.0
LAMBDA0 = 25.0
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W_R2 = 0.020
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
CAT_COLS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]
BOOL_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]


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


def forensic_gates(orth_o, orth_t, Bt, live_resid, bn=None):
    disp = float((orth_t.std() + 1e-12) / (orth_o.std() + 1e-12))
    leaks = [float(np.corrcoef(orth_t, Bt[:, j])[0, 1]) for j in range(Bt.shape[1])]
    max_leak = float(np.max(np.abs(leaks)))
    leak_arg = int(np.argmax(np.abs(leaks)))
    c_live = float(np.corrcoef(orth_t, live_resid)[0, 1]) if live_resid.std() > 1e-12 else 0.0
    names = bn or [f"c{j}" for j in range(Bt.shape[1])]
    return {
        "disp": disp,
        "max_basis_leak": max_leak,
        "leak_argmax": leak_arg,
        "leak_col": names[leak_arg] if leak_arg < len(names) else "?",
        "corr_live_resid": c_live,
        "F1": 0.94 <= disp <= 1.06,
        "F2": max_leak < 0.035,
        "F3": abs(c_live) < 0.05,
        "verified": (0.94 <= disp <= 1.06) and (max_leak < 0.035) and (abs(c_live) < 0.05),
    }


def score_fixed(u, zbase, y, folds, w_grid=(0.008, 0.01, 0.015, 0.02, 0.025, 0.03)):
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


def newton_delta(keys_tr, g, h, keys_map, lam=LAMBDA0):
    nbin = int(max(int(keys_tr.max()), int(keys_map.max())) + 1)
    sg = np.bincount(keys_tr, weights=g, minlength=nbin)
    sh = np.bincount(keys_tr, weights=h, minlength=nbin)
    cnt = np.bincount(keys_tr, minlength=nbin)
    delta = sg / (sh + lam)
    delta[cnt == 0] = 0.0
    return delta[keys_map]


def main():
    assert NO_SUBMIT
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
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot5_live = pd.read_csv(SLOT5).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

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

    # twins
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

    live_resid = L(lv) - Ridge(alpha=RIDGE_ALPHA).fit(Bt9, L(lv)).predict(Bt9)

    # R2 orth from v17.8
    res = np.load(OUT / "v17_8_residuals.npz")
    R2_oof, R2_test = res["R2_oof"], res["R2_test"]
    oo2, ot2, _ = frozen_lofo_orth(R2_oof, R2_test, Bo9, Bt9, folds, ridge_alpha=10.0)
    u2 = unitize(oo2)
    if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u2[te0]) < roc_auc_score(
        y[te0], z_s3[te0] - 0.01 * u2[te0]
    ):
        u2, ot2 = -u2, -ot2
    ut2 = unitize(ot2)

    # ===== Step 1: Slot5 baselines + Bo10 =====
    print("=== STEP 1: Slot5 baselines + R1|Bo10 + Δw_R2 ===")
    print(
        "V17.9 R1_pure: F2=0.0343, 5/5 all bases @0.03 "
        "S3=+7.10/sf19/t4.34 twin=+10.78/sf18/t4.66 NNLS=+7.82"
    )
    z_s5 = z_s3 + W_R2 * u2
    z_slot5_twin = z_slot3_twin + W_R2 * u2
    z_nnls_s5 = z_nnls + W_R2 * u2
    print(
        f"  AUC z_S5={roc_auc_score(y,z_s5):.7f} "
        f"twin={roc_auc_score(y,z_slot5_twin):.7f} "
        f"nnls={roc_auc_score(y,z_nnls_s5):.7f}"
    )
    Bo10 = np.column_stack([Bo9, u2])
    Bt10 = np.column_stack([Bt9, ut2])
    bn10 = BN9 + ["R2_orth"]
    bases_s5 = {
        "z_S5": z_s5,
        "z_Slot5_twin": z_slot5_twin,
        "z_NNLS_S5": z_nnls_s5,
    }

    # rebuild R1_pure (no q1000)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    keys_tr, keys_te = {}, {}
    for c in NUM_COLS:
        # skip degenerate int cents/tenth for pure integers-ish cols that caused leak
        vtr, vte = train[c].to_numpy(float), test[c].to_numpy(float)
        if c not in ("txn_count_last_24h", "card_age_months", "customer_age", "cvv_retry_count"):
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
    freq = np.log1p(np.bincount(inv)[inv])
    keys_tr["amt_freq_bin"] = np.clip((freq[:n_tr] * 5).astype(np.int64), 0, 50)
    keys_te["amt_freq_bin"] = np.clip((freq[n_tr:] * 5).astype(np.int64), 0, 50)
    if ORIG.exists():
        orig = pd.read_csv(ORIG)
        oc = [c for c in orig.columns if "amount" in c.lower() or c.lower() == "amt"]
        if oc:
            oa = np.round(orig[oc[0]].to_numpy(float) * 100).astype(np.int64)
            ctr = Counter(oa.tolist())
            keys_tr["amt_orig_bin"] = np.clip(
                np.array([ctr.get(int(x), 0) for x in amt_all[:n_tr]], dtype=np.int64), 0, 20
            )
            keys_te["amt_orig_bin"] = np.clip(
                np.array([ctr.get(int(x), 0) for x in amt_all[n_tr:]], dtype=np.int64), 0, 20
            )
    for cat in ("merchant_category", "channel", "device_type"):
        codes, _ = pd.factorize(pd.concat([train[cat], test[cat]], ignore_index=True))
        keys_tr[f"{cat}_x_amtones"] = codes[:n_tr] * 10 + keys_tr["amount_usd_ones"]
        keys_te[f"{cat}_x_amtones"] = codes[n_tr:] * 10 + keys_te["amount_usd_ones"]
    pure_names = list(keys_tr.keys())

    # calib on z_S3 for R1_pure (same as V17.9 — features trained on r_S3)
    ab_s3 = []
    for tr, va in folds:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s3[tr].reshape(-1, 1), y[tr])
        ab_s3.append((float(clf.coef_[0, 0]), float(clf.intercept_[0])))
    R1p = np.zeros(n_tr)
    R1p_te = []
    for f, (tr, va) in enumerate(folds):
        a, b = ab_s3[f]
        p = expit(a * z_s3 + b)
        g, h = y - p, np.clip(p * (1 - p), 1e-6, None)
        Xtr = np.column_stack(
            [newton_delta(keys_tr[k][tr], g[tr], h[tr], keys_tr[k][tr], 20.0) for k in pure_names]
        )
        Xva = np.column_stack(
            [newton_delta(keys_tr[k][tr], g[tr], h[tr], keys_tr[k][va], 20.0) for k in pure_names]
        )
        Xte = np.column_stack(
            [newton_delta(keys_tr[k][tr], g[tr], h[tr], keys_te[k], 20.0) for k in pure_names]
        )
        sw = np.sqrt(h[tr])
        lr = Ridge(alpha=100.0, fit_intercept=True)
        lr.fit(Xtr * sw[:, None], g[tr] * sw)
        R1p[va] = lr.predict(Xva)
        R1p_te.append(lr.predict(Xte))
    R1p_t = np.mean(np.stack(R1p_te, 0), 0)

    results = {}
    oriented = {}

    def eval_on_s5(name, vo, vt, Bo, Bt, bn, w_grid=None):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo, Bt, folds, ridge_alpha=10.0)
        gates = forensic_gates(oo, ot, Bt, live_resid, bn=bn)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s5[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        oriented[name] = (u, ut)
        per = {}
        for bname, zb in bases_s5.items():
            per[bname] = (
                score_fixed(u, zb, y, folds)
                if w_grid is None
                else score_fixed(u, zb, y, folds, w_grid=w_grid)
            )
        ok = (
            gates["verified"]
            and all(per[b]["pos"] >= 5 for b in bases_s5)
            and all(per[b]["sf25"] >= 19 for b in bases_s5)
            and all(per[b]["t"] >= 3.0 for b in bases_s5)
            and all(per[b]["mean"] > 0 for b in bases_s5)
        )
        results[name] = {
            "gates": gates,
            "per_base": per,
            "stage_ok": ok,
            "pre_corr_zS5": float(np.corrcoef(vo, z_s5)[0, 1]),
        }
        print(
            f"  {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{gates['leak_col']}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) ver={gates['verified']} "
            f"stage_ok={ok} preρ={results[name]['pre_corr_zS5']:+.4f}"
        )
        for b, row in per.items():
            print(
                f"    {b}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
                f"min={row['min']:+.3f} sf={row['sf25']} t={row['t']:.2f} folds={row['folds']}"
            )
        return u, ut

    # Δw_R2 on Slot5 baselines (add more R2 weight; u2 already in base so this is EXTRA)
    print("  --- Δw_R2 extras on Slot5 bases ---")
    dw_rows = {}
    for dw in (0.005, 0.008, 0.010):
        dw_rows[dw] = {}
        for bname, zb in bases_s5.items():
            # adding more of same u2
            dw_rows[dw][bname] = score_at_w(u2, zb, y, folds, dw)
        print(
            f"  +Δw_R2={dw}: "
            + " | ".join(
                f"{b} {dw_rows[dw][b]['pos']}/5 m={dw_rows[dw][b]['mean']:+.2f} "
                f"sf{dw_rows[dw][b]['sf25']} t={dw_rows[dw][b]['t']:.2f}"
                for b in bases_s5
            )
        )
    results["delta_w_R2"] = dw_rows

    # R1_pure | Bo10_S5
    eval_on_s5("R1_pure_Bo10", R1p, R1p_t, Bo10, Bt10, bn10)

    # R1|Bo10 + small Δw_R2 composite (fixed signs)
    u_r1, ut_r1 = oriented["R1_pure_Bo10"]
    for dw in (0.005, 0.008):
        # composite in raw space before orth? Better: score u_r1 and also add dw*u2 on bases
        # User wants orth(R1|Bo10)+Δw_R2 as candidate — evaluate as fixed blend of orth units
        vo = unitize(u_r1) + (dw / 0.02) * unitize(u2)  # scale-ish; cleaner: just score jointly
        # Actually evaluate additive: z + w1*u_r1 + dw*u2 — grid w1 with fixed dw
        print(f"  --- R1_Bo10 + Δw_R2={dw} (grid w_R1) ---")
        combo = {}
        for bname, zb in bases_s5.items():
            best = None
            for w1 in (0.01, 0.015, 0.02, 0.025, 0.03):
                # custom score with two additives
                base = [roc_auc_score(y[te], zb[te]) for _, te in folds]
                d = [
                    (
                        roc_auc_score(y[te], zb[te] + w1 * u_r1[te] + dw * u2[te])
                        - base[i]
                    )
                    * 1e5
                    for i, (_, te) in enumerate(folds)
                ]
                row = {
                    "w_R1": w1,
                    "dw_R2": dw,
                    "pos": int(sum(x > 0 for x in d)),
                    "mean": float(np.mean(d)),
                    "min": float(np.min(d)),
                    "folds": [round(x, 3) for x in d],
                }
                # sf/t with total additive direction unitize(w1*u_r1+dw*u2)
                u_comb = unitize(w1 * u_r1 + dw * u2)
                sfrow = score_at_w(u_comb, zb, y, folds, 0.01)
                row["sf25"] = sfrow["sf25"]
                row["t"] = sfrow["t"]
                if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
                    best = row
            combo[bname] = best
            print(
                f"    {bname}: {best['pos']}/5 wR1={best['w_R1']} mean={best['mean']:+.3f} "
                f"min={best['min']:+.3f} sf={best['sf25']} t={best['t']:.2f}"
            )
        results[f"R1_Bo10_plus_dwR2_{dw}"] = combo

    # ===== Step 2: expanded R_v2 on r_S5 =====
    print("=== STEP 2: expanded R_v2 on r_S5 ===", f"t={time.time()-t0:.1f}s")
    # calib on z_S5
    ab_s5 = []
    for tr, va in folds:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s5[tr].reshape(-1, 1), y[tr])
        ab_s5.append((float(clf.coef_[0, 0]), float(clf.intercept_[0])))

    # Build expanded feature matrix for LGB + TE keys
    # All-column freq + orig hits
    orig_df = pd.read_csv(ORIG) if ORIG.exists() else None
    feat_tr = []
    feat_te = []
    feat_names = []
    te_keys_tr = dict(keys_tr)  # start from pure digit keys
    te_keys_te = dict(keys_te)

    def add_cont(name, trv, tev):
        feat_names.append(name)
        feat_tr.append(np.asarray(trv, float))
        feat_te.append(np.asarray(tev, float))

    # per-column train∪test frequency and orig counts
    row_orig_hits_tr = np.zeros(n_tr)
    row_orig_hits_te = np.zeros(n_te)
    row_orig_log_tr = np.zeros(n_tr)
    row_orig_log_te = np.zeros(n_te)
    n_cols_used = 0

    all_cols = [c for c in train.columns if c not in ("id", "is_fraud")]
    for c in all_cols:
        # stringify values for exact match
        trs = train[c].astype(str).to_numpy()
        tes = test[c].astype(str).to_numpy()
        allv = np.concatenate([trs, tes])
        uniq, inv = np.unique(allv, return_inverse=True)
        cnt = np.bincount(inv)
        logf = np.log1p(cnt[inv])
        add_cont(f"logfreq_{c}", logf[:n_tr], logf[n_tr:])
        # TE key: coarse freq bin
        te_keys_tr[f"freqbin_{c}"] = np.clip((logf[:n_tr] * 3).astype(np.int64), 0, 30)
        te_keys_te[f"freqbin_{c}"] = np.clip((logf[n_tr:] * 3).astype(np.int64), 0, 30)
        if orig_df is not None and c in orig_df.columns:
            oc = Counter(orig_df[c].astype(str).tolist())
            otr = np.array([oc.get(x, 0) for x in trs], dtype=np.float64)
            ote = np.array([oc.get(x, 0) for x in tes], dtype=np.float64)
            add_cont(f"origcnt_{c}", np.log1p(otr), np.log1p(ote))
            row_orig_hits_tr += (otr > 0).astype(float)
            row_orig_hits_te += (ote > 0).astype(float)
            row_orig_log_tr += np.log1p(otr)
            row_orig_log_te += np.log1p(ote)
            n_cols_used += 1
            te_keys_tr[f"origbin_{c}"] = np.clip(otr.astype(np.int64), 0, 15)
            te_keys_te[f"origbin_{c}"] = np.clip(ote.astype(np.int64), 0, 15)
    add_cont("row_orig_hits", row_orig_hits_tr, row_orig_hits_te)
    if n_cols_used > 0:
        add_cont("row_orig_logmean", row_orig_log_tr / n_cols_used, row_orig_log_te / n_cols_used)

    # 2-way cat×cat and cat×digit pair freqs (top pairs only for speed)
    pair_list = [
        ("merchant_category", "channel"),
        ("merchant_category", "device_type"),
        ("channel", "device_type"),
        ("card_type", "auth_method"),
        ("merchant_category", "amount_usd_ones"),
        ("channel", "amount_usd_ones"),
        ("device_type", "amount_usd_cents"),
    ]
    for a, b in pair_list:
        if a.endswith("_ones") or a.endswith("_cents"):
            sa_tr, sa_te = keys_tr[a].astype(str), keys_te[a].astype(str)
        else:
            sa_tr, sa_te = train[a].astype(str).to_numpy(), test[a].astype(str).to_numpy()
        if b.endswith("_ones") or b.endswith("_cents"):
            sb_tr = keys_tr.get(b)
            if sb_tr is None and b == "amount_usd_cents":
                sb_tr = keys_tr["amount_usd_cents"]
                sb_te = keys_te["amount_usd_cents"]
            else:
                sb_tr, sb_te = keys_tr[b], keys_te[b]
            sb_tr, sb_te = sb_tr.astype(str), sb_te.astype(str)
        else:
            sb_tr, sb_te = train[b].astype(str).to_numpy(), test[b].astype(str).to_numpy()
        pair_tr = np.char.add(np.char.add(sa_tr.astype(str), "|"), sb_tr.astype(str))
        pair_te = np.char.add(np.char.add(sa_te.astype(str), "|"), sb_te.astype(str))
        allp = np.concatenate([pair_tr, pair_te])
        # factorize for TE keys
        codes, uniques = pd.factorize(allp)
        # cap rare: keep as codes
        te_keys_tr[f"pair_{a}_{b}"] = codes[:n_tr]
        te_keys_te[f"pair_{a}_{b}"] = codes[n_tr:]
        cnt = np.bincount(codes)
        logf = np.log1p(cnt[codes])
        add_cont(f"logfreq_pair_{a}_{b}", logf[:n_tr], logf[n_tr:])

    # raw nums + digit keys as LGB features
    for c in NUM_COLS:
        add_cont(c, train[c].to_numpy(float), test[c].to_numpy(float))
    for k in pure_names:
        add_cont(f"key_{k}", keys_tr[k].astype(float), keys_te[k].astype(float))

    Xtr_lgb = np.column_stack(feat_tr)
    Xte_lgb = np.column_stack(feat_te)
    print(f"  LGB feats={Xtr_lgb.shape[1]} TE keys={len(te_keys_tr)} t={time.time()-t0:.1f}s")

    # R1_v2_pure Newton TE on r_S5
    R1v2 = np.zeros(n_tr)
    R1v2_te = []
    knames = list(te_keys_tr.keys())
    for f, (tr, va) in enumerate(folds):
        a, b = ab_s5[f]
        p = expit(a * z_s5 + b)
        g, h = y - p, np.clip(p * (1 - p), 1e-6, None)
        Xtr = np.column_stack(
            [newton_delta(te_keys_tr[k][tr], g[tr], h[tr], te_keys_tr[k][tr]) for k in knames]
        )
        Xva = np.column_stack(
            [newton_delta(te_keys_tr[k][tr], g[tr], h[tr], te_keys_tr[k][va]) for k in knames]
        )
        Xte = np.column_stack(
            [newton_delta(te_keys_tr[k][tr], g[tr], h[tr], te_keys_te[k]) for k in knames]
        )
        sw = np.sqrt(h[tr])
        lr = Ridge(alpha=100.0, fit_intercept=True)
        lr.fit(Xtr * sw[:, None], g[tr] * sw)
        R1v2[va] = lr.predict(Xva)
        R1v2_te.append(lr.predict(Xte))
    R1v2_t = np.mean(np.stack(R1v2_te, 0), 0)
    print(f"  R1_v2 done t={time.time()-t0:.1f}s")

    # R2_v2 multi-seed LGB
    R2v2 = np.zeros(n_tr)
    R2v2_te_acc = []
    seeds = [42, 2026, 7]
    leaves_list = [6, 10]
    n_models = 0
    for seed in seeds:
        for leaves in leaves_list:
            preds_o = np.zeros(n_tr)
            preds_t = []
            for f, (tr, va) in enumerate(folds):
                a, b = ab_s5[f]
                p = expit(a * z_s5 + b)
                g, h = y - p, np.clip(p * (1 - p), 1e-6, None)
                model = lgb.LGBMRegressor(
                    objective="regression",
                    learning_rate=0.03,
                    num_leaves=leaves,
                    min_data_in_leaf=500,
                    n_estimators=50,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_lambda=1.0,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=4,
                )
                model.fit(Xtr_lgb[tr], g[tr], sample_weight=h[tr])
                preds_o[va] = model.predict(Xtr_lgb[va])
                preds_t.append(model.predict(Xte_lgb))
            R2v2 += preds_o
            R2v2_te_acc.append(np.mean(np.stack(preds_t, 0), 0))
            n_models += 1
    R2v2 /= n_models
    R2v2_t = np.mean(np.stack(R2v2_te_acc, 0), 0)
    print(f"  R2_v2 done models={n_models} t={time.time()-t0:.1f}s")
    print(
        f"  preρ R1v2/zS5={np.corrcoef(R1v2,z_s5)[0,1]:+.4f} "
        f"R2v2/zS5={np.corrcoef(R2v2,z_s5)[0,1]:+.4f}"
    )

    # ===== Step 3: scorecard =====
    print("=== STEP 3: scorecard on Slot5 ===")
    eval_on_s5("R1_v2_pure_Bo10", R1v2, R1v2_t, Bo10, Bt10, bn10)
    eval_on_s5("R2_v2_LGB_Bo10", R2v2, R2v2_t, Bo10, Bt10, bn10)
    eval_on_s5(
        "R12_v2_Bo10",
        unitize(R1v2) + unitize(R2v2),
        unitize(R1v2_t) + unitize(R2v2_t),
        Bo10,
        Bt10,
        bn10,
    )

    # stage
    staged = []
    for name, r in results.items():
        if name == "delta_w_R2" or name.startswith("R1_Bo10_plus"):
            continue
        if r.get("stage_ok"):
            u, ut = oriented[name]
            wstar = r["per_base"]["z_Slot5_twin"]["w"]
            path = SUB / f"sub_v18_0_{name}_w{str(wstar).replace('.','p')}_NOSUBMIT.csv"
            # build on Slot5 live
            cand = qlock(L(slot5_live) + wstar * ut, slot5_live)
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
            staged.append({"file": path.name, "signal": name, "w": wstar})
            print("STAGED", path.name)

    # also consider Δw_R2 alone if all bases pass
    for dw, rows in dw_rows.items():
        ok = all(
            rows[b]["pos"] >= 5 and rows[b]["sf25"] >= 19 and rows[b]["t"] >= 3.0 and rows[b]["mean"] > 0
            for b in bases_s5
        )
        if ok:
            path = SUB / f"sub_v18_0_delta_wR2_{str(dw).replace('.','p')}_NOSUBMIT.csv"
            cand = qlock(L(slot5_live) + dw * ut2, slot5_live)
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
            staged.append({"file": path.name, "signal": f"delta_wR2_{dw}", "w": dw})
            print("STAGED", path.name)

    if not staged:
        print("NO STAGE")

    # strip large from results for JSON
    payload_results = {}
    for k, v in results.items():
        if k == "delta_w_R2" or k.startswith("R1_Bo10_plus"):
            payload_results[k] = v
        else:
            payload_results[k] = {
                "gates": v["gates"],
                "per_base": v["per_base"],
                "stage_ok": v["stage_ok"],
                "pre_corr_zS5": v["pre_corr_zS5"],
            }

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "slot5_lb": 0.95394,
            "auc_z_S5": float(roc_auc_score(y, z_s5)),
            "v17_9_R1_pure_ref": {
                "F2": 0.0343,
                "S3": "5/5 +7.10 sf19 t4.34",
                "twin": "5/5 +10.78 sf18 t4.66",
                "NNLS": "5/5 +7.82",
            },
            "results": payload_results,
            "staged": staged,
            "n_lgb_feats": int(Xtr_lgb.shape[1]),
            "n_te_keys": len(te_keys_tr),
            "elapsed_s": time.time() - t0,
            "stage_rule": "F1∧F2∧F3 ∧ 5/5 ∧ sf>=19 ∧ t>=3.0 on all 3 Slot5 baselines",
        }
    )
    OUT.joinpath("v18_0.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUT / "v18_0_residuals.npz",
        R1p_oof=R1p,
        R1p_test=R1p_t,
        R1v2_oof=R1v2,
        R1v2_test=R1v2_t,
        R2v2_oof=R2v2,
        R2v2_test=R2v2_t,
        u2_oof=u2,
        u2_test=ut2,
        z_s5=z_s5,
    )
    print("WROTE", OUT / "v18_0.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
