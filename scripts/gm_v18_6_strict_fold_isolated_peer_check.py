#!/usr/bin/env python3
"""GM V18.6 — Strict fold-isolated peer check (zero cross-fold OOF echo).

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU ~20s.
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


def frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=1.0):
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


def leave_self_peer_pool(z_pool, group_ids_pool, m0=M0):
    """Leave-self-out peer mean within a single pool (va or test)."""
    g = np.asarray(group_ids_pool, np.int64)
    z = np.asarray(z_pool, float)
    # reindex groups dense within pool
    _, g2 = np.unique(g, return_inverse=True)
    nbin = int(g2.max()) + 1
    cnt = np.bincount(g2, minlength=nbin).astype(np.float64)
    sz = np.bincount(g2, weights=z, minlength=nbin)
    c = cnt[g2]
    peer = (sz[g2] - z) / (c - 1.0 + m0)
    peer = np.where(c <= 1, 0.0, peer)
    return peer, c


def codes_from_str(arr):
    c, _ = pd.factorize(arr, sort=False)
    return c.astype(np.int64)


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


def main():
    assert NO_SUBMIT
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
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

    live_resid = L(lv) - Ridge(alpha=10.0).fit(Bt9, L(lv)).predict(Bt9)

    res = np.load(OUT / "v17_8_residuals.npz")
    R2_oof, R2_test = res["R2_oof"], res["R2_test"]
    oo2, ot2, _ = frozen_lofo_orth(R2_oof, R2_test, Bo9, Bt9, folds, ridge_alpha=10.0)
    u2 = unitize(oo2)
    if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u2[te0]) < roc_auc_score(
        y[te0], z_s3[te0] - 0.01 * u2[te0]
    ):
        u2, ot2 = -u2, -ot2
    ut2 = unitize(ot2)

    v185 = np.load(OUT / "v18_5_clean_peer.npz")
    z_s5 = v185["z_s5"]
    z_s5_t_sym = v185["z_s5_t_sym"]
    # prefer consistent rebuild
    z_s5 = z_s3 + W_R2 * u2
    z_s5_t_sym = z_s3_t + W_R2 * ut2
    z_slot5_twin = z_slot3_twin + W_R2 * u2
    z_nnls_s5 = z_nnls + W_R2 * u2
    Bo10 = np.column_stack([Bo9, u2])
    Bt10 = np.column_stack([Bt9, ut2])
    bn10 = BN9 + ["R2_orth"]
    bases_s5 = {"z_S5": z_s5, "z_Slot5_twin": z_slot5_twin, "z_NNLS_S5": z_nnls_s5}

    P1_v = v185["P1_oof"]
    P2_v = v185["P2_oof"]
    P4_v = v185["P4_oof"]

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
    # global codes (same as V18.5) then slice pools
    key_all = {
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

    print("=== STEP 1: strict va-only / test-only peer pools ===")
    z_oof_u = unitize(z_s5)
    z_te_u = unitize(z_s5_t_sym)

    # Also rebuild V18.5-style global peer_z_k2 for sign check
    def global_peer(z_all, gid):
        g = np.asarray(gid, np.int64)
        z = np.asarray(z_all, float)
        nbin = int(g.max()) + 1
        cnt = np.bincount(g, minlength=nbin).astype(np.float64)
        sz = np.bincount(g, weights=z, minlength=nbin)
        c = cnt[g]
        peer = (sz[g] - z) / (c - 1.0 + M0)
        return np.where(c <= 1, 0.0, peer)

    z_sym_all = np.concatenate([z_oof_u, z_te_u])
    peer_k2_global = global_peer(z_sym_all, key_all["k2_balance"])[:n_tr]
    c_p1 = float(np.corrcoef(P1_v, peer_k2_global)[0, 1])
    c_p2 = float(np.corrcoef(P2_v, peer_k2_global)[0, 1])
    c_p4 = float(np.corrcoef(P4_v, peer_k2_global)[0, 1])
    print(
        f"  Sign check vs peer_z_k2_balance (V18.5 global): "
        f"ρ(P1)={c_p1:+.4f} ρ(P2)={c_p2:+.4f} ρ(P4)={c_p4:+.4f}"
    )
    sign = "POSITIVE homophily (+peer_z)" if c_p2 > 0 else "NEGATIVE contrast (−peer_z)"
    print(f"  => P2 uses {sign}")

    # Strict peer_z per key: va-only for OOF, test-only for test
    peer_strict_oof = {kn: np.zeros(n_tr) for kn in key_order}
    peer_strict_cnt = {kn: np.zeros(n_tr) for kn in key_order}
    peer_strict_te = {}
    for kn in key_order:
        # test pool
        peer_t, cnt_t = leave_self_peer_pool(
            z_te_u, key_all[kn][n_tr:], m0=M0
        )
        peer_strict_te[kn] = peer_t
        # per-fold va pool
        for f, (tr, va) in enumerate(folds):
            peer_va, cnt_va = leave_self_peer_pool(
                z_oof_u[va], key_all[kn][va], m0=M0
            )
            peer_strict_oof[kn][va] = peer_va
            peer_strict_cnt[kn][va] = cnt_va
        print(
            f"  {kn}: mean_cnt_va={peer_strict_cnt[kn].mean():.2f} "
            f"mean_cnt_te={cnt_t.mean():.2f} "
            f"ρ(strict,global)_oof={np.corrcoef(peer_strict_oof[kn], global_peer(z_sym_all, key_all[kn])[:n_tr])[0,1]:+.3f}"
        )

    # Pre-orth peer_z_strict vs Bo10
    peer_orth_tr, peer_orth_te = [], []
    for kn in key_order:
        po, pt, _ = frozen_lofo_orth(
            peer_strict_oof[kn], peer_strict_te[kn], Bo10, Bt10, folds, ridge_alpha=1.0
        )
        peer_orth_tr.append(unitize(po))
        peer_orth_te.append(unitize(pt))

    # X_peer_only strict 24
    pair_idx = [(1, 0), (7, 5), (1, 2), (0, 5), (3, 2), (4, 0), (6, 1), (7, 1)]
    peer_cols_tr, peer_cols_te = [], []
    for i in range(8):
        peer_cols_tr.append(peer_orth_tr[i])
        peer_cols_te.append(peer_orth_te[i])
    for i, kn in enumerate(key_order):
        peer_cols_tr.append(np.log1p(peer_strict_cnt[kn]))
        # test logcnt from test pool
        _, cnt_t = leave_self_peer_pool(z_te_u, key_all[kn][n_tr:], m0=M0)
        peer_cols_te.append(np.log1p(cnt_t))
    for a, b in pair_idx:
        peer_cols_tr.append(peer_orth_tr[a] - peer_orth_tr[b])
        peer_cols_te.append(peer_orth_te[a] - peer_orth_te[b])
    X_peer_tr = np.column_stack(peer_cols_tr).astype(np.float32)
    X_peer_te = np.column_stack(peer_cols_te).astype(np.float32)

    # cohort (same as V18.5 — no z)
    g1 = codes_from_str(mer)
    g2 = codes_from_str(
        (pd.Series(mer) + "|" + pd.Series(ch) + "|" + pd.Series(dev)).to_numpy()
    )
    cohort_tr, cohort_te = [], []
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
        for arr in (u1, u2, u2 - ug):
            cohort_tr.append(arr[:n_tr])
            cohort_te.append(arr[n_tr:])
    for arr in (
        amt / (np.abs(bal) + 10.0),
        global_u["amount_usd"] - global_u["account_balance_usd"],
        global_u["velocity_score"] - global_u["hours_since_last_txn"],
        global_u["merchant_risk_score"] + global_u["velocity_score"],
    ):
        cohort_tr.append(arr[:n_tr])
        cohort_te.append(arr[n_tr:])
    X_coh_tr = np.column_stack(cohort_tr).astype(np.float32)
    X_coh_te = np.column_stack(cohort_te).astype(np.float32)
    X_clean_tr = np.column_stack([X_peer_tr, X_coh_tr]).astype(np.float32)
    X_clean_te = np.column_stack([X_peer_te, X_coh_te]).astype(np.float32)
    print(f"  X_peer_strict={X_peer_tr.shape} X_clean={X_clean_tr.shape} t={time.time()-t0:.1f}s")

    # Fit P1/P2/P4 strict
    print("=== Fit P1/P2/P4 strict ===")
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

    def fit_lgb(Xtr, Xte):
        oof = np.zeros(n_tr)
        te_acc = []
        for seed in seeds:
            oo = np.zeros(n_tr)
            tef = []
            for f, (tr, va) in enumerate(folds):
                model = lgb.LGBMRegressor(**lgb_params, random_state=seed)
                model.fit(Xtr[tr], g5[f][tr], sample_weight=h5[f][tr])
                oo[va] = model.predict(Xtr[va])
                tef.append(model.predict(Xte))
            oof += oo
            te_acc.append(np.mean(np.stack(tef, 0), 0))
        return oof / len(seeds), np.mean(np.stack(te_acc, 0), 0)

    P1s = np.zeros(n_tr)
    P1s_te = []
    peer8_tr = np.column_stack(peer_orth_tr)
    peer8_te = np.column_stack(peer_orth_te)
    for f, (tr, va) in enumerate(folds):
        sw = np.sqrt(h5[f][tr])
        lr = Ridge(alpha=100.0, fit_intercept=True)
        lr.fit(peer8_tr[tr] * sw[:, None], g5[f][tr] * sw)
        P1s[va] = lr.predict(peer8_tr[va])
        P1s_te.append(lr.predict(peer8_te))
    P1s_t = np.mean(np.stack(P1s_te, 0), 0)
    P2s, P2s_t = fit_lgb(X_peer_tr, X_peer_te)
    P4s, P4s_t = fit_lgb(X_clean_tr, X_clean_te)
    print(f"  fits done t={time.time()-t0:.1f}s")

    # corr strict vs V18.5
    print(
        f"  ρ(P2_strict,P2_v185)={np.corrcoef(P2s,P2_v)[0,1]:+.4f} "
        f"ρ(P4_strict,P4_v185)={np.corrcoef(P4s,P4_v)[0,1]:+.4f} "
        f"ρ(P1_strict,P1_v185)={np.corrcoef(P1s,P1_v)[0,1]:+.4f}"
    )
    c_p2s_k2 = float(np.corrcoef(P2s, peer_strict_oof["k2_balance"])[0, 1])
    print(f"  ρ(P2_strict, peer_z_k2_strict)={c_p2s_k2:+.4f}")

    print("=== Scorecard strict vs V18.5 ===")
    results = {}
    oriented = {}
    for name, vo, vt, ref in [
        ("P1_strict", P1s, P1s_t, P1_v),
        ("P2_strict", P2s, P2s_t, P2_v),
        ("P4_strict", P4s, P4s_t, P4_v),
    ]:
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
        ok = (
            gates["verified"]
            and all(per[b]["pos"] >= 5 for b in bases_s5)
            and all(per[b]["sf25"] >= 19 for b in bases_s5)
            and all(per[b]["t"] >= 3.0 for b in bases_s5)
        )
        results[name] = {
            "gates": gates,
            "per_base": per,
            "pre_corr_zS5": float(np.corrcoef(vo, z_s5)[0, 1]),
            "corr_R2": float(np.corrcoef(vo, R2_oof)[0, 1]),
            "corr_v185": float(np.corrcoef(vo, ref)[0, 1]),
            "stage_ok": ok,
        }
        print(
            f"  {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{gates['leak_col']}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) "
            f"ρ(v185)={results[name]['corr_v185']:+.3f} stage_ok={ok}"
        )
        for b, row in per.items():
            print(
                f"    {b}: {row['pos']}/5 @{row['w']} mean={row['mean']:+.3f} "
                f"min={row['min']:+.3f} folds={row['folds']} sf={row['sf25']} t={row['t']:.2f}"
            )

    # Compare attenuation vs V18.5 reported means
    v185_ref = {
        "P2": {"z_S5": 18.968, "z_Slot5_twin": 33.544, "z_NNLS_S5": 20.497},
        "P4": {"z_S5": 17.857, "z_Slot5_twin": 31.578, "z_NNLS_S5": 19.560},
    }
    verdict = "PASS_ZERO_ECHO"
    for nm, key in [("P2_strict", "P2"), ("P4_strict", "P4")]:
        r = results[nm]
        if not r["stage_ok"]:
            verdict = "FAIL_GATES_OR_ELITE"
        # if lift collapses to <25% of V18.5, call echo
        ratio = r["per_base"]["z_S5"]["mean"] / v185_ref[key]["z_S5"]
        print(f"  {nm} lift_ratio vs V18.5 z_S5: {ratio:.3f}")
        if ratio < 0.25 and r["per_base"]["z_S5"]["mean"] < 3.0:
            verdict = "FAIL_ECHO_COLLAPSE"

    if verdict == "PASS_ZERO_ECHO" and results["P4_strict"]["stage_ok"]:
        print("VERDICT: ZERO_ECHO — V18.5 P4 safe to submit (after user auth)")
    elif verdict == "PASS_ZERO_ECHO":
        print("VERDICT: SIGNAL_SURVIVES but strict gates soft — review")
    else:
        print(f"VERDICT: {verdict} — DO NOT SUBMIT V18.5 P2/P4")

    staged = []
    # NOSUBMIT only if strict passes — overwrite/refresh ammo
    for name, r in results.items():
        if r["stage_ok"]:
            u, ut = oriented[name]
            w = r["per_base"]["z_Slot5_twin"]["w"]
            path = SUB / f"sub_v18_6_{name}_w{str(w).replace('.','p')}_NOSUBMIT.csv"
            cand = qlock(L(slot5) + w * ut, lv)
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
            staged.append({"file": path.name, "w": w})
            print("STAGED", path.name)
    if not staged:
        print("NO STAGE")

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "sign_check": {
                "rho_P1_peer_k2": c_p1,
                "rho_P2_peer_k2": c_p2,
                "rho_P4_peer_k2": c_p4,
                "interpretation": sign,
            },
            "rho_P2_strict_peer_k2": c_p2s_k2,
            "results": results,
            "v185_ref_means": v185_ref,
            "verdict": verdict,
            "staged": staged,
            "elapsed_s": time.time() - t0,
        }
    )
    OUT.joinpath("v18_6.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v18_6.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
