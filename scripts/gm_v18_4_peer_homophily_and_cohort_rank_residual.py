#!/usr/bin/env python3
"""GM V18.4 — Peer homophily + cohort-rank residual on r_S5 (Bo10).

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
FLOAT_COLS = [
    "amount_usd",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
    "distance_from_home_km",
    "hours_since_last_txn",
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
        "leaks": {names[j]: leaks[j] for j in range(len(names))},
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


def leave_self_peer(z, group_ids, m0=M0):
    """Vectorized leave-self-out peer mean + diff."""
    g = np.asarray(group_ids, np.int64)
    z = np.asarray(z, float)
    nbin = int(g.max()) + 1
    cnt = np.bincount(g, minlength=nbin).astype(np.float64)
    sz = np.bincount(g, weights=z, minlength=nbin)
    c = cnt[g]
    peer = (sz[g] - z) / (c - 1.0 + m0)
    # singleton / empty: peer ~ 0 (only prior)
    peer = np.where(c <= 1, 0.0, peer)
    diff = z - peer
    return peer, diff, c


def within_group_rank(values, group_ids):
    """Average-rank percentile within each group → (0,1)."""
    v = np.asarray(values, float)
    g = np.asarray(group_ids, np.int64)
    out = np.empty(len(v), dtype=np.float64)
    # sort by group then value
    order = np.lexsort((v, g))
    g_s, v_s = g[order], v[order]
    # find group boundaries
    breaks = np.flatnonzero(np.r_[True, g_s[1:] != g_s[:-1], True])
    ranks = np.empty(len(v), dtype=np.float64)
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = b - a
        # average ranks 1..n for ties via rankdata
        ranks[a:b] = rankdata(v_s[a:b], method="average")
        out_slice = (ranks[a:b] - 0.5) / n
        ranks[a:b] = out_slice
    out[order] = ranks
    return out


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

    live_resid = L(lv) - Ridge(alpha=RIDGE_ALPHA).fit(Bt9, L(lv)).predict(Bt9)

    # R2 orth → Slot5 / Bo10
    res = np.load(OUT / "v17_8_residuals.npz")
    R2_oof, R2_test = res["R2_oof"], res["R2_test"]
    oo2, ot2, _ = frozen_lofo_orth(R2_oof, R2_test, Bo9, Bt9, folds, ridge_alpha=10.0)
    u2 = unitize(oo2)
    if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u2[te0]) < roc_auc_score(
        y[te0], z_s3[te0] - 0.01 * u2[te0]
    ):
        u2, ot2 = -u2, -ot2
    ut2 = unitize(ot2)
    z_s5 = z_s3 + W_R2 * u2
    z_slot5_twin = z_slot3_twin + W_R2 * u2
    z_nnls_s5 = z_nnls + W_R2 * u2
    Bo10 = np.column_stack([Bo9, u2])
    Bt10 = np.column_stack([Bt9, ut2])
    bn10 = BN9 + ["R2_orth"]
    bases_s5 = {"z_S5": z_s5, "z_Slot5_twin": z_slot5_twin, "z_NNLS_S5": z_nnls_s5}

    # calib g_S5 / h_S5
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

    # z_all from OOF z_S5 + Slot5 test logits (NO y)
    z_all = np.concatenate([unitize(z_s5), unitize(L(slot5))])
    assert len(z_all) == n_all

    print("=== STEP 1: X_peer ===", f"t={time.time()-t0:.1f}s")
    feat_names = []
    cols = []

    def add(name, v):
        feat_names.append(name)
        cols.append(np.asarray(v, float))

    def codes_from_str(arr):
        c, _ = pd.factorize(arr, sort=False)
        return c.astype(np.int64)

    # Group A — peer keys
    amt = np.concatenate(
        [train["amount_usd"].to_numpy(float), test["amount_usd"].to_numpy(float)]
    )
    bal = np.concatenate(
        [
            train["account_balance_usd"].to_numpy(float),
            test["account_balance_usd"].to_numpy(float),
        ]
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
        [
            train["device_type"].astype(str).to_numpy(),
            test["device_type"].astype(str).to_numpy(),
        ]
    )
    cvv = np.concatenate(
        [
            train["cvv_retry_count"].to_numpy(float),
            test["cvv_retry_count"].to_numpy(float),
        ]
    )
    foreign = np.concatenate(
        [
            train["is_foreign_transaction"].astype(str).to_numpy(),
            test["is_foreign_transaction"].astype(str).to_numpy(),
        ]
    )
    # card_present absent → auth_method as proxy
    auth = np.concatenate(
        [
            train["auth_method"].astype(str).to_numpy(),
            test["auth_method"].astype(str).to_numpy(),
        ]
    )

    key_specs = {
        "k1_amount": codes_from_str(np.round(amt, 8).astype(str)),
        "k2_balance": codes_from_str(np.round(bal, 8).astype(str)),
        "k3_velocity": codes_from_str(np.round(vel, 8).astype(str)),
        "k4_mrisk": codes_from_str(np.round(mr, 8).astype(str)),
        "k5_distance": codes_from_str(np.round(dist, 8).astype(str)),
        "k6_mer_amt_int": codes_from_str(
            np.char.add(
                np.char.add(mer, "|"),
                np.round(amt).astype(np.int64).astype(str),
            )
        ),
        "k7_mer_ch_dev_cvv_foreign_auth": codes_from_str(
            np.char.add(
                np.char.add(
                    np.char.add(
                        np.char.add(
                            np.char.add(np.char.add(mer, "|"), ch),
                            "|",
                        ),
                        dev,
                    ),
                    "|",
                ),
                np.char.add(
                    np.char.add(cvv.astype(np.int64).astype(str), "|"),
                    np.char.add(np.char.add(foreign, "|"), auth),
                ),
            )
        ),
        "k8_cents_mer": codes_from_str(
            np.char.add(
                np.char.add(
                    ((np.round(np.abs(amt) * 100).astype(np.int64) % 100) % 100).astype(str),
                    "|",
                ),
                np.char.add(
                    ((np.round(np.abs(bal) * 100).astype(np.int64) % 100) % 100).astype(str),
                    np.char.add("|", mer),
                ),
            )
        ),
    }

    peer_z_cols = []
    for kn, gid in key_specs.items():
        peer, diff, cnt = leave_self_peer(z_all, gid, M0)
        add(f"peer_z_{kn}", peer)
        add(f"peer_diff_{kn}", diff)
        add(f"peer_logcnt_{kn}", np.log1p(cnt))
        peer_z_cols.append(peer)
        print(f"  {kn}: mean_cnt={cnt.mean():.2f} max={cnt.max():.0f}")

    # Group B — cohort ranks
    g1 = codes_from_str(mer)
    g2 = codes_from_str(np.char.add(np.char.add(np.char.add(mer, "|"), ch), np.char.add("|", dev)))
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
        add(f"rank_{cname}_G1", u1)
        add(f"rank_{cname}_G2", u2)
        add(f"rankdiff_{cname}_G2_minus_global", u2 - ug)

    # Group C — ratios / risk regime
    add("drain_ratio", amt / (np.abs(bal) + 10.0))
    add("rank_amt_minus_bal", global_u["amount_usd"] - global_u["account_balance_usd"])
    add(
        "rank_vel_minus_hrs",
        global_u["velocity_score"] - global_u["hours_since_last_txn"],
    )
    add(
        "rank_mrisk_plus_vel",
        global_u["merchant_risk_score"] + global_u["velocity_score"],
    )
    add("abs_z_S5", np.abs(z_all))
    add("z_S5_rank", U(z_all))

    X_all = np.column_stack(cols).astype(np.float32)
    X_tr, X_te = X_all[:n_tr], X_all[n_tr:]
    print(f"  X_peer={X_all.shape} t={time.time()-t0:.1f}s")

    # ===== STEP 2: fit =====
    print("=== STEP 2: fit R3_peer ===")
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

    # seed 42 single + importance
    R3_s42 = np.zeros(n_tr)
    R3_s42_te = []
    gain = np.zeros(len(feat_names))
    for f, (tr, va) in enumerate(folds):
        model = lgb.LGBMRegressor(**lgb_params, random_state=42)
        model.fit(X_tr[tr], g5[f][tr], sample_weight=h5[f][tr])
        R3_s42[va] = model.predict(X_tr[va])
        R3_s42_te.append(model.predict(X_te))
        gain += np.asarray(model.booster_.feature_importance(importance_type="gain"), float)
    R3_s42_t = np.mean(np.stack(R3_s42_te, 0), 0)
    top15 = [
        {"feat": feat_names[i], "gain": float(gain[i])}
        for i in np.argsort(-gain)[:15]
    ]
    print("  TOP15 gain (R3_peer_s42):")
    for j, row in enumerate(top15, 1):
        print(f"    {j:2d}. {row['feat']:55s} gain={row['gain']:.2f}")

    # 3-seed bag
    seeds = [42, 2026, 7]
    R3_bag = np.zeros(n_tr)
    R3_bag_te = []
    for seed in seeds:
        oof = np.zeros(n_tr)
        tef = []
        for f, (tr, va) in enumerate(folds):
            model = lgb.LGBMRegressor(**lgb_params, random_state=seed)
            model.fit(X_tr[tr], g5[f][tr], sample_weight=h5[f][tr])
            oof[va] = model.predict(X_tr[va])
            tef.append(model.predict(X_te))
        R3_bag += oof
        R3_bag_te.append(np.mean(np.stack(tef, 0), 0))
    R3_bag /= len(seeds)
    R3_bag_t = np.mean(np.stack(R3_bag_te, 0), 0)

    # linear Ridge on 8 peer_z only
    peer_mat_tr = np.column_stack([c[:n_tr] for c in peer_z_cols])
    peer_mat_te = np.column_stack([c[n_tr:] for c in peer_z_cols])
    R3_lin = np.zeros(n_tr)
    R3_lin_te = []
    for f, (tr, va) in enumerate(folds):
        sw = np.sqrt(h5[f][tr])
        lr = Ridge(alpha=10.0, fit_intercept=True)
        lr.fit(peer_mat_tr[tr] * sw[:, None], g5[f][tr] * sw)
        R3_lin[va] = lr.predict(peer_mat_tr[va])
        R3_lin_te.append(lr.predict(peer_mat_te))
    R3_lin_t = np.mean(np.stack(R3_lin_te, 0), 0)
    print(f"  fits done t={time.time()-t0:.1f}s")

    # ===== STEP 3 =====
    print("=== STEP 3: forensic + Slot5 scorecard ===")
    signals = {
        "R3_peer_s42": (R3_s42, R3_s42_t),
        "R3_peer_bag": (R3_bag, R3_bag_t),
        "R3_peer_linear": (R3_lin, R3_lin_t),
    }
    results = {}
    oriented = {}
    for name, (vo, vt) in signals.items():
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo10, Bt10, folds, ridge_alpha=10.0)
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
            path = SUB / f"sub_v18_4_{name}_w{str(w).replace('.','p')}_NOSUBMIT.csv"
            cand = qlock(L(slot5) + w * ut, lv)
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
            staged.append({"file": path.name, "signal": name, "w": w})
            print("STAGED", path.name)
    if not staged:
        print("NO STAGE")

    # strip per-col leaks from JSON size
    for r in results.values():
        r["gates"].pop("leaks", None)

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "n_X_peer": int(X_all.shape[1]),
            "feat_names": feat_names,
            "top15_gain": top15,
            "note_cols": {
                "is_international": "is_foreign_transaction",
                "card_present": "auth_method (proxy; column absent)",
                "hours_since_prev_tx": "hours_since_last_txn",
            },
            "results": results,
            "staged": staged,
            "elapsed_s": time.time() - t0,
            "stage_rule": "F1∧F2∧F3 ∧ 5/5 ∧ sf>=19 ∧ t>=3.0 on all 3 Slot5 baselines",
        }
    )
    OUT.joinpath("v18_4.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        OUT / "v18_4_peer.npz",
        R3_s42_oof=R3_s42,
        R3_s42_test=R3_s42_t,
        R3_bag_oof=R3_bag,
        R3_bag_test=R3_bag_t,
        R3_lin_oof=R3_lin,
        R3_lin_test=R3_lin_t,
        feat_names=np.array(feat_names, dtype=object),
        gain=gain,
    )
    print("WROTE", OUT / "v18_4.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
