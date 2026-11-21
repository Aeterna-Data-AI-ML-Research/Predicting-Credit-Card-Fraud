#!/usr/bin/env python3
"""GM V18.5 — Clean peer homophily: symmetric z_all, zero self-z, pre-orth peer.

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expit
from scipy.stats import rankdata, wilcoxon
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
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
NO_SUBMIT = True
RIDGE_ALPHA = 10.0
M0 = 5.0
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W_R2 = 0.020


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
    try:
        _, wp = wilcoxon(d25, alternative="greater")
        wp = float(wp)
    except Exception:
        wp = float("nan")
    best.update({"sf25": int((d25 > 0).sum()), "t": float(mean / (se + 1e-18)), "wilcoxon_p": wp})
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


def leave_self_peer_mean_std(z, group_ids, m0=M0):
    g = np.asarray(group_ids, np.int64)
    z = np.asarray(z, float)
    nbin = int(g.max()) + 1
    cnt = np.bincount(g, minlength=nbin).astype(np.float64)
    sz = np.bincount(g, weights=z, minlength=nbin)
    sz2 = np.bincount(g, weights=z * z, minlength=nbin)
    c = cnt[g]
    peer = (sz[g] - z) / (c - 1.0 + m0)
    peer = np.where(c <= 1, 0.0, peer)
    # leave-self-out variance for count>=3
    c_lo = c - 1.0
    sum_lo = sz[g] - z
    sum2_lo = sz2[g] - z * z
    # Var = (sum2 - sum^2/n) / (n-1) for n>=2; we need n=c-1 >= 2 => c>=3
    var = np.zeros_like(z)
    ok = c >= 3
    n_lo = c_lo[ok]
    var[ok] = (sum2_lo[ok] - (sum_lo[ok] ** 2) / n_lo) / np.maximum(n_lo - 1.0, 1.0)
    var = np.maximum(var, 0.0)
    peer_std = np.sqrt(var)
    peer_std = np.where(c >= 3, peer_std, 0.0)
    return peer, peer_std, c


def within_group_rank(values, group_ids):
    v = np.asarray(values, float)
    g = np.asarray(group_ids, np.int64)
    out = np.empty(len(v), dtype=np.float64)
    order = np.lexsort((v, g))
    g_s, v_s = g[order], v[order]
    breaks = np.flatnonzero(np.r_[True, g_s[1:] != g_s[:-1], True])
    ranks = np.empty(len(v), dtype=np.float64)
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = b - a
        ranks[a:b] = (rankdata(v_s[a:b], method="average") - 0.5) / n
    out[order] = ranks
    return out


def codes_from_str(arr):
    c, _ = pd.factorize(arr, sort=False)
    return c.astype(np.int64)


def max_basis_leak(v_oof, v_test, Bo, Bt, folds, ridge_alpha=1.0, bn=None):
    oo, ot, _ = frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=ridge_alpha)
    leaks = [abs(float(np.corrcoef(ot, Bt[:, j])[0, 1])) for j in range(Bt.shape[1])]
    j = int(np.argmax(leaks))
    names = bn or [f"c{j}" for j in range(Bt.shape[1])]
    return oo, ot, float(leaks[j]), names[j]


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
    n_all = n_tr + n_te

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
    dE_o, dE_t = unitize(dE_o), unitize(dE_t)

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

    # Symmetric ATH on test: same Bo9 linear recipe (not LIVE logits — those shift Family_E)
    ridge_ath = Ridge(alpha=1.0, fit_intercept=True).fit(Bo9, z_ath)
    z_ath_t_sym = Bt9 @ ridge_ath.coef_ + ridge_ath.intercept_
    z_s3 = z_ath + 0.01045 * dE_o + 0.0055 * unitize(d1)
    z_s3_t = z_ath_t_sym + 0.01045 * dE_t + 0.0055 * unitize(d1t)

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

    res = np.load(OUT / "v17_8_residuals.npz")
    R2_oof, R2_test = res["R2_oof"], res["R2_test"]
    oo2, ot2, _ = frozen_lofo_orth(R2_oof, R2_test, Bo9, Bt9, folds, ridge_alpha=10.0)
    u2 = unitize(oo2)
    if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u2[te0]) < roc_auc_score(
        y[te0], z_s3[te0] - 0.01 * u2[te0]
    ):
        u2, ot2 = -u2, -ot2
    ut2 = unitize(ot2)

    # Fix 1: symmetric Slot5
    z_s5 = z_s3 + W_R2 * u2
    z_s5_t_sym = z_s3_t + W_R2 * ut2
    z_slot5_twin = z_slot3_twin + W_R2 * u2
    z_nnls_s5 = z_nnls + W_R2 * u2
    Bo10 = np.column_stack([Bo9, u2])
    Bt10 = np.column_stack([Bt9, ut2])
    bn10 = BN9 + ["R2_orth"]
    bases_s5 = {"z_S5": z_s5, "z_Slot5_twin": z_slot5_twin, "z_NNLS_S5": z_nnls_s5}

    fe_o = unitize(dE_o)
    fe_t = unitize(dE_t)
    c_oof = float(np.corrcoef(z_s5, fe_o)[0, 1])
    c_te = float(np.corrcoef(z_s5_t_sym, fe_t)[0, 1])
    c_live = float(np.corrcoef(L(slot5), fe_t)[0, 1])
    print("=== STEP 1: symmetric z + clean X_peer ===")
    print(
        f"  corr(z_S5,FE): oof={c_oof:+.4f} test_sym={c_te:+.4f} "
        f"|Δ|={abs(c_oof-c_te):.4f}  (live_Slot5 vs FE={c_live:+.4f})"
    )
    if abs(c_oof - c_te) >= 0.005:
        print(f"  WARN FE corr asymmetry {abs(c_oof-c_te):.4f} >= 0.005 (continuing with twin-sym)")
    else:
        print("  FE corr symmetry OK (|Δ|<0.005)")

    z_sym_all = np.concatenate([unitize(z_s5), unitize(z_s5_t_sym)])
    assert len(z_sym_all) == n_all

    # calib g/h on S5
    ab_s5 = []
    for tr, _ in folds:
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        clf.fit(z_s5[tr].reshape(-1, 1), y[tr])
        ab_s5.append((float(clf.coef_[0, 0]), float(clf.intercept_[0])))
    g5, h5 = [], []
    for f in range(5):
        a, b = ab_s5[f]
        p = expit(a * z_s5 + b)
        g5.append(y - p)
        h5.append(np.clip(p * (1 - p), 1e-6, None))

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    amt = np.concatenate([train["amount_usd"].to_numpy(float), test["amount_usd"].to_numpy(float)])
    bal = np.concatenate(
        [train["account_balance_usd"].to_numpy(float), test["account_balance_usd"].to_numpy(float)]
    )
    vel = np.concatenate(
        [train["velocity_score"].to_numpy(float), test["velocity_score"].to_numpy(float)]
    )
    mr = np.concatenate(
        [
            train["merchant_risk_score"].to_numpy(float),
            test["merchant_risk_score"].to_numpy(float),
        ]
    )
    dist = np.concatenate(
        [
            train["distance_from_home_km"].to_numpy(float),
            test["distance_from_home_km"].to_numpy(float),
        ]
    )
    hrs = np.concatenate(
        [
            train["hours_since_last_txn"].to_numpy(float),
            test["hours_since_last_txn"].to_numpy(float),
        ]
    )
    mer = np.concatenate(
        [
            train["merchant_category"].astype(str).to_numpy(),
            test["merchant_category"].astype(str).to_numpy(),
        ]
    )
    ch = np.concatenate(
        [train["channel"].astype(str).to_numpy(), test["channel"].astype(str).to_numpy()]
    )
    dev = np.concatenate(
        [train["device_type"].astype(str).to_numpy(), test["device_type"].astype(str).to_numpy()]
    )
    cvv = np.concatenate(
        [train["cvv_retry_count"].to_numpy(float), test["cvv_retry_count"].to_numpy(float)]
    )
    foreign = np.concatenate(
        [
            train["is_foreign_transaction"].astype(str).to_numpy(),
            test["is_foreign_transaction"].astype(str).to_numpy(),
        ]
    )
    auth = np.concatenate(
        [train["auth_method"].astype(str).to_numpy(), test["auth_method"].astype(str).to_numpy()]
    )

    key_order = [
        "k1_amount",
        "k2_balance",
        "k3_velocity",
        "k4_mrisk",
        "k5_distance",
        "k6_mer_amt_int",
        "k7_mer_ch_dev_cvv_foreign_auth",
        "k8_cents_mer",
    ]
    key_specs = {
        "k1_amount": codes_from_str(np.round(amt, 8).astype(str)),
        "k2_balance": codes_from_str(np.round(bal, 8).astype(str)),
        "k3_velocity": codes_from_str(np.round(vel, 8).astype(str)),
        "k4_mrisk": codes_from_str(np.round(mr, 8).astype(str)),
        "k5_distance": codes_from_str(np.round(dist, 8).astype(str)),
        "k6_mer_amt_int": codes_from_str(
            np.char.add(np.char.add(mer, "|"), np.round(amt).astype(np.int64).astype(str))
        ),
        "k7_mer_ch_dev_cvv_foreign_auth": codes_from_str(
            (
                pd.Series(mer)
                + "|"
                + pd.Series(ch)
                + "|"
                + pd.Series(dev)
                + "|"
                + pd.Series(cvv.astype(np.int64).astype(str))
                + "|"
                + pd.Series(foreign)
                + "|"
                + pd.Series(auth)
            ).to_numpy()
        ),
        "k8_cents_mer": codes_from_str(
            (
                pd.Series(((np.round(np.abs(amt) * 100).astype(np.int64) % 100) % 100).astype(str))
                + "|"
                + pd.Series(((np.round(np.abs(bal) * 100).astype(np.int64) % 100) % 100).astype(str))
                + "|"
                + pd.Series(mer)
            ).to_numpy()
        ),
    }

    # peer_z raw → pre-orth vs Bo10 (alpha=1)
    peer_orth_tr = []
    peer_orth_te = []
    peer_logcnt_tr = []
    peer_logcnt_te = []
    peer_std_tr = []
    peer_std_te = []
    peer_leak = {}
    for kn in key_order:
        peer, peer_std, cnt = leave_self_peer_mean_std(z_sym_all, key_specs[kn], M0)
        po, pt, leak, lcol = max_basis_leak(
            peer[:n_tr], peer[n_tr:], Bo10, Bt10, folds, ridge_alpha=1.0, bn=bn10
        )
        peer_leak[kn] = {"pre_orth_leak": leak, "col": lcol}
        # after orth, check leak again
        _, _, leak2, lcol2 = max_basis_leak(po, pt, Bo10, Bt10, folds, ridge_alpha=1.0, bn=bn10)
        peer_leak[kn]["post_orth_leak"] = leak2
        peer_leak[kn]["post_col"] = lcol2
        peer_orth_tr.append(unitize(po))
        peer_orth_te.append(unitize(pt))
        peer_logcnt_tr.append(np.log1p(cnt[:n_tr]))
        peer_logcnt_te.append(np.log1p(cnt[n_tr:]))
        peer_std_tr.append(peer_std[:n_tr])
        peer_std_te.append(peer_std[n_tr:])
        print(
            f"  {kn}: pre_leak={leak:.4f}({lcol}) post_leak={leak2:.4f}({lcol2}) "
            f"mean_cnt={cnt.mean():.1f}"
        )

    # pairwise peer diffs among orth peers (no self-z)
    # user asked 8 pairwise diffs e.g. k2-k1, k8-k6 — take a fixed set of 8
    pair_idx = [(1, 0), (7, 5), (1, 2), (0, 5), (3, 2), (4, 0), (6, 1), (7, 1)]  # 0-based
    peer_names = []
    peer_cols_tr = []
    peer_cols_te = []
    for i, kn in enumerate(key_order):
        peer_names.append(f"peer_z_orth_{kn}")
        peer_cols_tr.append(peer_orth_tr[i])
        peer_cols_te.append(peer_orth_te[i])
    for i, kn in enumerate(key_order):
        peer_names.append(f"peer_logcnt_{kn}")
        peer_cols_tr.append(peer_logcnt_tr[i])
        peer_cols_te.append(peer_logcnt_te[i])
    # also include peer_std for k1,k2 (balance/amount) as extra useful — keep within ~24
    # User said 24 = 8 peer_z + 8 logcnt + 8 pairwise. Stick to that.
    for a, b in pair_idx:
        nm = f"peer_diff_{key_order[a]}_minus_{key_order[b]}"
        peer_names.append(nm)
        peer_cols_tr.append(peer_orth_tr[a] - peer_orth_tr[b])
        peer_cols_te.append(peer_orth_te[a] - peer_orth_te[b])

    X_peer_tr = np.column_stack(peer_cols_tr).astype(np.float32)
    X_peer_te = np.column_stack(peer_cols_te).astype(np.float32)
    assert X_peer_tr.shape[1] == 24

    # Cohort-only (zero z)
    g1 = codes_from_str(mer)
    g2 = codes_from_str(
        (pd.Series(mer) + "|" + pd.Series(ch) + "|" + pd.Series(dev)).to_numpy()
    )
    cohort_names = []
    cohort_tr = []
    cohort_te = []
    global_u = {}
    for cname, vals in [
        ("amount_usd", amt),
        ("account_balance_usd", bal),
        ("velocity_score", vel),
        ("merchant_risk_score", mr),
        ("distance_from_home_km", dist),
        ("hours_since_last_txn", hrs),
    ]:
        ug = U(vals)
        global_u[cname] = ug
        u1 = within_group_rank(vals, g1)
        u2 = within_group_rank(vals, g2)
        for nm, arr in [
            (f"rank_{cname}_G1", u1),
            (f"rank_{cname}_G2", u2),
            (f"rankdiff_{cname}_G2_minus_global", u2 - ug),
        ]:
            cohort_names.append(nm)
            cohort_tr.append(arr[:n_tr])
            cohort_te.append(arr[n_tr:])

    for nm, arr in [
        ("drain_ratio", amt / (np.abs(bal) + 10.0)),
        ("rank_amt_minus_bal", global_u["amount_usd"] - global_u["account_balance_usd"]),
        (
            "rank_vel_minus_hrs",
            global_u["velocity_score"] - global_u["hours_since_last_txn"],
        ),
        (
            "rank_mrisk_plus_vel",
            global_u["merchant_risk_score"] + global_u["velocity_score"],
        ),
    ]:
        cohort_names.append(nm)
        cohort_tr.append(arr[:n_tr])
        cohort_te.append(arr[n_tr:])

    X_coh_tr = np.column_stack(cohort_tr).astype(np.float32)
    X_coh_te = np.column_stack(cohort_te).astype(np.float32)
    # 6*3 + 4 = 22
    assert X_coh_tr.shape[1] == 22, X_coh_tr.shape[1]

    X_clean_tr = np.column_stack([X_peer_tr, X_coh_tr]).astype(np.float32)
    X_clean_te = np.column_stack([X_peer_te, X_coh_te]).astype(np.float32)
    clean_names = peer_names + cohort_names
    print(
        f"  X_peer={X_peer_tr.shape[1]} X_cohort={X_coh_tr.shape[1]} "
        f"X_clean={X_clean_tr.shape[1]} t={time.time()-t0:.1f}s"
    )
    max_post = max(v["post_orth_leak"] for v in peer_leak.values())
    print(f"  max post-orth peer_z leak={max_post:.4f} (target <0.015)")

    # ===== STEP 2 =====
    print("=== STEP 2: fit P1..P4 ===")
    lgb_params = dict(
        objective="regression",
        learning_rate=0.03,
        num_leaves=6,
        min_data_in_leaf=500,
        n_estimators=40,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        verbosity=-1,
        n_jobs=-1,
    )
    seeds = [42, 2026, 7]

    def fit_lgb_bag(Xtr, Xte, seeds=seeds):
        oof = np.zeros(n_tr)
        te_acc = []
        gain = np.zeros(Xtr.shape[1])
        n = 0
        for seed in seeds:
            oo = np.zeros(n_tr)
            tef = []
            for f, (tr, va) in enumerate(folds):
                model = lgb.LGBMRegressor(**lgb_params, random_state=seed)
                model.fit(Xtr[tr], g5[f][tr], sample_weight=h5[f][tr])
                oo[va] = model.predict(Xtr[va])
                tef.append(model.predict(Xte))
                if seed == seeds[0]:
                    gain += np.asarray(
                        model.booster_.feature_importance(importance_type="gain"), float
                    )
            oof += oo
            te_acc.append(np.mean(np.stack(tef, 0), 0))
            n += 1
        return oof / n, np.mean(np.stack(te_acc, 0), 0), gain

    # P1 linear on 8 peer_z_orth
    P1 = np.zeros(n_tr)
    P1_te = []
    peer8_tr = np.column_stack(peer_orth_tr)
    peer8_te = np.column_stack(peer_orth_te)
    for f, (tr, va) in enumerate(folds):
        sw = np.sqrt(h5[f][tr])
        lr = Ridge(alpha=100.0, fit_intercept=True)
        lr.fit(peer8_tr[tr] * sw[:, None], g5[f][tr] * sw)
        P1[va] = lr.predict(peer8_tr[va])
        P1_te.append(lr.predict(peer8_te))
    P1_t = np.mean(np.stack(P1_te, 0), 0)

    P2, P2_t, gain2 = fit_lgb_bag(X_peer_tr, X_peer_te)
    P3, P3_t, gain3 = fit_lgb_bag(X_coh_tr, X_coh_te)
    P4, P4_t, gain4 = fit_lgb_bag(X_clean_tr, X_clean_te)
    print(f"  fits done t={time.time()-t0:.1f}s")

    top15_p4 = [
        {"feat": clean_names[i], "gain": float(gain4[i])}
        for i in np.argsort(-gain4)[:15]
    ]
    top15_p3 = [
        {"feat": cohort_names[i], "gain": float(gain3[i])}
        for i in np.argsort(-gain3)[:15]
    ]
    print("  TOP15 P4_peer_clean:")
    for j, row in enumerate(top15_p4, 1):
        print(f"    {j:2d}. {row['feat']:55s} {row['gain']:.2f}")
    print("  TOP15 P3_cohort_only:")
    for j, row in enumerate(top15_p3, 1):
        print(f"    {j:2d}. {row['feat']:55s} {row['gain']:.2f}")

    # ===== STEP 3 =====
    print("=== STEP 3: forensic + scorecard ===")
    signals = {
        "P1_peer_linear": (P1, P1_t),
        "P2_peer_only_lgb": (P2, P2_t),
        "P3_cohort_only_lgb": (P3, P3_t),
        "P4_peer_clean_lgb": (P4, P4_t),
    }
    results = {}
    oriented = {}
    for name, (vo, vt) in signals.items():
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo10, Bt10, folds, ridge_alpha=1.0)
        gates = forensic_gates(oo, ot, Bt10, live_resid, bn=bn10)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s5[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        ut = unitize(ot)
        oriented[name] = (u, ut)
        per = {b: score_fixed(u, zb, y, folds) for b, zb in bases_s5.items()}
        pre = float(np.corrcoef(vo, z_s5)[0, 1])
        corr_r2 = float(np.corrcoef(vo, R2_oof)[0, 1])
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
            "pre_corr_zS5": pre,
            "corr_R2_v178": corr_r2,
            "stage_ok": ok,
        }
        print(
            f"  {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{gates['leak_col']}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) "
            f"preρ={pre:+.4f} ρ(R2)={corr_r2:+.4f} stage_ok={ok}"
        )
        for b, row in per.items():
            print(
                f"    {b}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
                f"min={row['min']:+.3f} folds={row['folds']} "
                f"sf={row['sf25']} t={row['t']:.2f}"
            )

    staged = []
    for name, r in results.items():
        if r["stage_ok"]:
            u, ut = oriented[name]
            w = r["per_base"]["z_Slot5_twin"]["w"]
            path = SUB / f"sub_v18_5_{name}_w{str(w).replace('.','p')}_NOSUBMIT.csv"
            cand = qlock(L(slot5) + w * ut, lv)
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
            staged.append({"file": path.name, "signal": name, "w": w})
            print("STAGED", path.name)
    if not staged:
        print("NO STAGE")

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "fe_corr_check": {
                "oof": c_oof,
                "test_sym": c_te,
                "abs_delta": abs(c_oof - c_te),
                "live_slot5": c_live,
            },
            "peer_leak": peer_leak,
            "n_peer": 24,
            "n_cohort": 22,
            "n_clean": 46,
            "top15_p4": top15_p4,
            "top15_p3": top15_p3,
            "results": results,
            "staged": staged,
            "elapsed_s": time.time() - t0,
            "stage_rule": "F1∧F2∧F3 ∧ 5/5 ∧ sf>=19 ∧ t>=3.0 on all 3 Slot5 baselines",
        }
    )
    OUT.joinpath("v18_5.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUT / "v18_5_clean_peer.npz",
        P1_oof=P1,
        P1_test=P1_t,
        P2_oof=P2,
        P2_test=P2_t,
        P3_oof=P3,
        P3_test=P3_t,
        P4_oof=P4,
        P4_test=P4_t,
        z_s5=z_s5,
        z_s5_t_sym=z_s5_t_sym,
        clean_names=np.array(clean_names, dtype=object),
    )
    print("WROTE", OUT / "v18_5.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
