#!/usr/bin/env python3
"""Max-leap candidate: CatBoost uid residual orth to [ATH, LGBM-uid], stack on ATH_v2 (0.95550)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

T0 = time.time()
SSD = Path("/tmp/s6e11_v39_ssd")
REPO = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def L(v):
    u = (rankdata(v, method="average") - 0.5) / len(v)
    return logit(np.clip(u, 1e-6, 1 - 1e-6))


def add_uid(df):
    d = df.copy()
    amt_bin = d["amount_usd"].round(0).astype(int).astype(str)
    age_bin = (d["customer_age"] // 5).astype(str)
    bal_bin = d["account_balance_usd"].round(-2).astype(int).astype(str)
    d["uid1"] = (
        d["card_type"].astype(str)
        + "|"
        + d["merchant_category"].astype(str)
        + "|"
        + d["device_type"].astype(str)
        + "|"
        + age_bin
        + "|"
        + amt_bin
    )
    d["uid2"] = (
        d["card_type"].astype(str)
        + "|"
        + d["channel"].astype(str)
        + "|"
        + d["auth_method"].astype(str)
        + "|"
        + bal_bin
    )
    d["uid3"] = (
        d["merchant_category"].astype(str)
        + "|"
        + d["device_type"].astype(str)
        + "|"
        + d["is_foreign_transaction"].astype(int).astype(str)
        + "|"
        + amt_bin
    )
    return d


def oof_te(series, y, folds, m=5):
    oof = np.zeros(len(y))
    gm = float(y.mean())
    for k in range(5):
        trm, vam = folds != k, folds == k
        tmp = pd.DataFrame({"k": series[trm], "y": y[trm]})
        agg = tmp.groupby("k")["y"].agg(["sum", "count"])
        mp = (agg["sum"] + m * gm) / (agg["count"] + m)
        oof[vam] = series[vam].map(mp).fillna(gm).to_numpy()
    tmp = pd.DataFrame({"k": series, "y": y})
    agg = tmp.groupby("k")["y"].agg(["sum", "count"])
    mp = (agg["sum"] + m * gm) / (agg["count"] + m)
    return oof, mp


def main():
    tr = pd.read_csv(SSD / "data/train.csv")
    te = pd.read_csv(SSD / "data/test.csv")
    folds = pd.read_csv(SSD / "folds/folds.csv")["fold"].to_numpy(int)
    y = tr["is_fraud"].to_numpy(float)
    log(f"n_tr={len(tr)} n_te={len(te)}")

    tr2, te2 = add_uid(tr), add_uid(te)
    full = pd.concat([tr2, te2], ignore_index=True)
    for u in ("uid1", "uid2", "uid3"):
        vc = full[u].value_counts()
        tr2[u + "_cnt"] = tr2[u].map(vc).fillna(1).astype(np.float32)
        te2[u + "_cnt"] = te2[u].map(vc).fillna(1).astype(np.float32)
        stats = full.groupby(u)["amount_usd"].agg(["mean", "std"]).reset_index()
        stats.columns = [u, "mu", "sd"]
        for part in (tr2, te2):
            m = part.merge(stats, on=u, how="left")
            part[u + "_amt_z"] = ((part["amount_usd"] - m["mu"]) / (m["sd"].fillna(0) + 1e-5)).astype(
                np.float32
            )
        oof, mp = oof_te(tr2[u], y, folds)
        tr2[u + "_te"] = oof.astype(np.float32)
        te2[u + "_te"] = te2[u].map(mp).fillna(y.mean()).astype(np.float32)

    num = [
        "amount_usd",
        "hours_since_last_txn",
        "txn_count_last_24h",
        "distance_from_home_km",
        "card_age_months",
        "customer_age",
        "account_balance_usd",
        "cvv_retry_count",
        "velocity_score",
        "time_of_day_hour",
        "day_of_week",
        "merchant_risk_score",
        "prior_disputes",
    ]
    flags = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    uids = [c for c in tr2.columns if c.endswith(("_cnt", "_amt_z", "_te"))]
    cat = ["merchant_category", "card_type", "auth_method", "channel", "device_type", "uid1", "uid2", "uid3"]
    feats = num + flags + uids + cat
    X, Xt = tr2[feats].copy(), te2[feats].copy()
    for c in flags:
        X[c] = X[c].astype(int)
        Xt[c] = Xt[c].astype(int)

    oof = np.zeros(len(y))
    tp = np.zeros(len(te2))
    for k in range(5):
        trm, vam = folds != k, folds == k
        model = CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42 + k,
            verbose=False,
            early_stopping_rounds=50,
            thread_count=6,
        )
        model.fit(
            Pool(X[trm], y[trm], cat_features=cat),
            eval_set=Pool(X[vam], y[vam], cat_features=cat),
            use_best_model=True,
        )
        oof[vam] = model.predict_proba(X[vam])[:, 1]
        tp += model.predict_proba(Xt)[:, 1] / 5
        log(f"fold{k} best={model.get_best_iteration()} oof_partial={roc_auc_score(y[vam], oof[vam]):.5f}")

    log(f"CatBoost OOF AUC={roc_auc_score(y, oof):.5f}")
    np.save(SSD / "out/uid_catboost_oof.npy", oof)
    np.save(SSD / "out/uid_catboost_test.npy", tp)

    ath0 = pd.read_csv(SSD / "submissions/ATH.csv")
    sub4 = pd.read_csv(REPO / "submissions/sub_v39_S4_gemini_uid_entity_a004.csv")  # ATH_v2 0.95550
    lgb_o = np.load(SSD / "out/uid_entity_oof.npy")
    lgb_t = np.load(SSD / "out/uid_entity_test.npy")
    z_fb = np.load(SSD / "preds/z_fullbank_oof.npy").ravel()

    pa = np.clip(ath0.iloc[:, 1].to_numpy(float), 1e-7, 1 - 1e-7)
    z_ath_t = logit(pa)
    z_ath_o = L(z_fb)
    beta_u = np.cov(logit(np.clip(lgb_o, 1e-7, 1 - 1e-7)), z_ath_o)[0, 1] / np.var(z_ath_o)
    res_uid_t = logit(np.clip(lgb_t, 1e-7, 1 - 1e-7)) - beta_u * z_ath_t
    res_uid_t = (res_uid_t - res_uid_t.mean()) / (res_uid_t.std() + 1e-8)
    res_uid_o = logit(np.clip(lgb_o, 1e-7, 1 - 1e-7)) - beta_u * z_ath_o
    res_uid_o = (res_uid_o - res_uid_o.mean()) / (res_uid_o.std() + 1e-8)

    z_v2_t = logit(np.clip(sub4.iloc[:, 1].to_numpy(float), 1e-7, 1 - 1e-7))
    # OOF ATH_v2 ≈ ATH + 0.04*res_uid on train
    z_v2_o = z_ath_o + 0.04 * res_uid_o

    z_cb_o = logit(np.clip(oof, 1e-7, 1 - 1e-7))
    z_cb_t = logit(np.clip(tp, 1e-7, 1 - 1e-7))
    Xo = np.column_stack([np.ones(len(y)), z_ath_o, res_uid_o])
    beta, _, _, _ = np.linalg.lstsq(Xo, z_cb_o, rcond=None)
    res_cb_o = z_cb_o - Xo @ beta
    res_cb_o = (res_cb_o - res_cb_o.mean()) / (res_cb_o.std() + 1e-8)
    Xt = np.column_stack([np.ones(len(tp)), z_ath_t, res_uid_t])
    res_cb_t = z_cb_t - Xt @ beta
    res_cb_t = (res_cb_t - res_cb_t.mean()) / (res_cb_t.std() + 1e-8)

    # pick alpha on OOF vs z_v2_o by LOFO mean ΔAUC
    best_a, best_d = 0.035, -1.0
    for a in np.linspace(0.01, 0.08, 15):
        d_folds = []
        for k in range(5):
            vam = folds == k
            base = roc_auc_score(y[vam], expit(z_v2_o[vam]))
            d_folds.append(roc_auc_score(y[vam], expit(z_v2_o[vam] + a * res_cb_o[vam])) - base)
        md = float(np.mean(d_folds))
        log(f"alpha={a:.3f} meanΔ={md:+.2e} pos={(np.array(d_folds)>0).sum()}/5")
        if md > best_d:
            best_d, best_a = md, float(a)

    # also try tiny dual: ATH + a_uid*res_uid + a_cb*res_cb with a_uid near 0.04
    best_dual = None
    best_dual_d = -1.0
    for au in (0.03, 0.035, 0.04, 0.045):
        for ac in (0.015, 0.025, 0.035, 0.045):
            d_folds = []
            for k in range(5):
                vam = folds == k
                base = roc_auc_score(y[vam], expit(z_ath_o[vam]))
                pred = expit(z_ath_o[vam] + au * res_uid_o[vam] + ac * res_cb_o[vam])
                d_folds.append(roc_auc_score(y[vam], pred) - base)
            md = float(np.mean(d_folds))
            if md > best_dual_d:
                best_dual_d = md
                best_dual = (au, ac, md)

    log(f"BEST stack-on-v2 a_cb={best_a:.3f} Δ={best_d:+.2e}")
    log(f"BEST dual au,ac={best_dual}")

    # Prefer dual if better OOF than v2+cb alone; else v2+cb
    # Compare absolute OOF AUC
    a_cb = best_a
    au, ac, _ = best_dual
    oof_v2cb = roc_auc_score(y, expit(z_v2_o + a_cb * res_cb_o))
    oof_dual = roc_auc_score(y, expit(z_ath_o + au * res_uid_o + ac * res_cb_o))
    oof_v2 = roc_auc_score(y, expit(z_v2_o))
    log(f"OOF v2={oof_v2:.6f} v2+cb={oof_v2cb:.6f} dual={oof_dual:.6f}")

    if oof_dual >= oof_v2cb and oof_dual >= oof_v2:
        p = expit(z_ath_t + au * res_uid_t + ac * res_cb_t)
        tag = f"dual_au{au:.3f}_ac{ac:.3f}"
        recipe = "dual"
    elif oof_v2cb >= oof_v2:
        p = expit(z_v2_t + a_cb * res_cb_t)
        tag = f"v2_cb_a{a_cb:.3f}"
        recipe = "v2_cb"
    else:
        # fallback: refine uid alpha near peak 0.0365
        p = expit(z_ath_t + 0.0365 * res_uid_t)
        tag = "uid_a0365"
        recipe = "uid_refine"

    out = sub4.copy()
    out.iloc[:, 1] = p
    path = REPO / f"submissions/sub_v39_MAX_{tag}.csv"
    out.to_csv(path, index=False)
    out.to_csv(SSD / f"out/sub_v39_MAX_{tag}.csv", index=False)
    meta = {
        "recipe": recipe,
        "tag": tag,
        "oof_v2": oof_v2,
        "oof_v2cb": oof_v2cb,
        "oof_dual": oof_dual,
        "best_a_cb": best_a,
        "best_dual": best_dual,
        "cb_oof_auc": float(roc_auc_score(y, oof)),
    }
    (SSD / "out/max_leap_meta.json").write_text(__import__("json").dumps(meta, indent=2))
    log(f"WROTE {path} recipe={recipe} meta={meta}")
    print(path)


if __name__ == "__main__":
    main()
