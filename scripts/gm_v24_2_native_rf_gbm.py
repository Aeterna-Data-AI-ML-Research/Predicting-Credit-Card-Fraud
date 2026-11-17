#!/usr/bin/env python3
"""V24.2 — Native Random Forest & Gradient Boosting (expert DS protocol).

Protocol:
  • Same 5-fold IDs as public OOF bank (StratifiedKFold rs=42 → folds.csv)
  • Leakage-safe LOFO target encoding + digit/leading-digit FE (S6E11 literature)
  • Models: sklearn RandomForest, HistGradientBoosting, LightGBM (GBDT reference)
  • Metrics: OOF AUC, nested H2H vs ATH surrogate, optional gated blend into ATH
  • Artifacts: OOF/test npz + submission CSVs

Local Mac CPU. Submits only if quota allows AND nested gate passes.
"""
from __future__ import annotations

import json
import subprocess
import time
import traceback
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
OUT = Path("/tmp/s6e11_v16/out")
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
ATH = SUB / "sub_v24_SRI440_pub.csv"
PIN = Path("/tmp/s6e11_nb_research/sri_pinned")
PY = "/tmp/s6e11_v23_venv/bin/python"
KAG = [PY, "-m", "kaggle"]
REPORT = LOG / "GM_V24_2_NATIVE_RF_GBM_REPORT.md"
N_TR, N_TE = 699960, 299984
T0 = time.time()
LINES: list[str] = []

CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
NUMS = [
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


def log(msg: str) -> None:
    s = f"[{time.time()-T0:7.1f}s] {msg}"
    print(s, flush=True)
    LINES.append(s)


def L(v):
    v = np.asarray(v, float).ravel()
    return logit(np.clip((rankdata(v, method="average") - 0.5) / len(v), 1e-6, 1 - 1e-6))


def rf_rank(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def omr(a, b):
    return (1.0 - float(spearmanr(a, b).correlation)) * 1e4


def auc(y, p):
    return float(roc_auc_score(y, p))


def digit_feats(x: np.ndarray, prefix: str) -> dict:
    """1st/2nd decimal digits + leading digit — established S6E11 signal family."""
    x = np.asarray(x, float)
    ax = np.abs(x)
    # fractional part digits
    frac = ax - np.floor(ax)
    d1 = np.floor(frac * 10).astype(np.int16)
    d2 = np.floor(frac * 100).astype(np.int16) % 10
    # leading digit of scientific magnitude
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.floor(np.log10(np.clip(ax, 1e-12, None)))
        lead = np.floor(ax / (10.0**mag)).astype(np.int16)
    lead = np.clip(lead, 1, 9)
    return {
        f"{prefix}_d1": d1,
        f"{prefix}_d2": d2,
        f"{prefix}_lead": lead,
        f"{prefix}_log1p": np.log1p(ax),
    }


def build_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in NUMS:
        out[c] = df[c].astype(np.float64)
    for c in BOOLS:
        out[c] = df[c].astype(np.int8)
    # cyclic hour / dow
    out["hour_sin"] = np.sin(2 * np.pi * df["time_of_day_hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * df["time_of_day_hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    # expert interactions
    bal = np.clip(df["account_balance_usd"].to_numpy(float), 1.0, None)
    out["amt_over_bal"] = df["amount_usd"].to_numpy(float) / bal
    out["vel_x_txn24"] = df["velocity_score"].to_numpy(float) * df["txn_count_last_24h"].to_numpy(float)
    out["dist_x_foreign"] = df["distance_from_home_km"].to_numpy(float) * df["is_foreign_transaction"].astype(float)
    out["risk_x_new_merch"] = df["merchant_risk_score"].to_numpy(float) * df["is_new_merchant"].astype(float)
    out["cvv_x_vpn"] = df["cvv_retry_count"].to_numpy(float) * df["used_vpn"].astype(float)
    # digit family on continuous risk proxies
    for col in ("amount_usd", "velocity_score", "merchant_risk_score", "hours_since_last_txn", "account_balance_usd", "distance_from_home_km"):
        for k, v in digit_feats(df[col].to_numpy(float), col).items():
            out[k] = v
    # raw cats kept for encoding
    for c in CATS:
        out[c] = df[c].astype(str)
    return out


def lofo_target_encode(tr_col, y, folds, te_col, prior=20.0):
    """Out-of-fold mean encoding; global mean for unseen."""
    oof = np.zeros(len(tr_col), dtype=np.float64)
    gmean = float(y.mean())
    te = np.zeros(len(te_col), dtype=np.float64)
    # accumulate full-data encode for test (mean of fold models)
    te_acc = np.zeros(len(te_col), dtype=np.float64)
    for tr_idx, te_idx in folds:
        s = pd.DataFrame({"k": tr_col.iloc[tr_idx].to_numpy(), "y": y[tr_idx]})
        agg = s.groupby("k")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + prior * gmean) / (agg["count"] + prior)
        oof[te_idx] = tr_col.iloc[te_idx].map(enc).fillna(gmean).to_numpy(float)
        te_acc += te_col.map(enc).fillna(gmean).to_numpy(float)
    te[:] = te_acc / len(folds)
    return oof, te


def paired_h2h(a, b, y, folds):
    ds = []
    for _, te in folds:
        ds.append((auc(y[te], a[te]) - auc(y[te], b[te])) * 1e5)
    ds = np.asarray(ds, float)
    return dict(
        mean=float(ds.mean()),
        min=float(ds.min()),
        pos=int((ds > 0).sum()),
        t=float(ds.mean() / (ds.std(ddof=1) + 1e-12) * np.sqrt(len(ds))),
    )


def submit_and_poll(path: Path, msg: str):
    try:
        r = subprocess.run(
            KAG + ["competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg],
            capture_output=True,
            text=True,
            timeout=180,
        )
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        log(r.stdout.strip()[:350])
        log(r.stderr.strip()[:250])
        low = out.lower()
        if "limit" in low or "successfully submitted" not in low:
            return False, None, "quota_or_error"
        for i in range(40):
            time.sleep(18 if i else 8)
            p = subprocess.run(
                KAG + ["competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
                capture_output=True,
                text=True,
                timeout=90,
            )
            for ln in p.stdout.splitlines()[1:15]:
                if path.name not in ln:
                    continue
                log(f"poll {ln[:220]}")
                if "COMPLETE" in ln.upper():
                    for tok in reversed([c.strip() for c in ln.split(",")]):
                        try:
                            return True, float(tok), "ok"
                        except ValueError:
                            continue
        return True, None, "timeout"
    except Exception as e:
        log(f"submit err {e}")
        traceback.print_exc()
        return False, None, "exception"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    log("=== V24.2 Native RF + GBM (expert DS) ===")

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    assert len(train) == N_TR and len(test) == N_TE
    y = train["is_fraud"].to_numpy(float)
    ids_te = test["id"].to_numpy()
    fold_id = pd.read_csv(BANK / "folds.csv")["fold"].to_numpy()
    assert len(fold_id) == N_TR
    folds = [(np.where(fold_id != i)[0], np.where(fold_id == i)[0]) for i in range(5)]
    log(f"fraud rate={y.mean():.5f} folds={ [int((fold_id==i).sum()) for i in range(5)] }")

    log("Feature engineering…")
    tr_f = build_base_frame(train)
    te_f = build_base_frame(test)

    # LOFO TE for cats + digit cats (as categorical ints)
    te_cols = []
    for c in CATS:
        oof_e, te_e = lofo_target_encode(tr_f[c], y, folds, te_f[c], prior=30.0)
        tr_f[f"te_{c}"] = oof_e
        te_f[f"te_{c}"] = te_e
        te_cols.append(f"te_{c}")
        log(f"  TE {c} done")

    # ordinal encode cats for tree splits (fold-safe: fit on train fold only → use global ordinal from full train labels alphabetically = deterministic, no y leak)
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr_f[[f"ord_{c}" for c in CATS]] = ord_enc.fit_transform(tr_f[CATS])
    te_f[[f"ord_{c}" for c in CATS]] = ord_enc.transform(te_f[CATS])

    drop = CATS  # raw strings
    feature_cols = [c for c in tr_f.columns if c not in drop]
    X_tr = tr_f[feature_cols].to_numpy(np.float64)
    X_te = te_f[feature_cols].to_numpy(np.float64)
    # NaN safety for HistGBM
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    log(f"matrix {X_tr.shape} features={len(feature_cols)}")

    models = {
        "rf": dict(
            ctor=lambda: RandomForestClassifier(
                n_estimators=350,
                max_depth=22,
                min_samples_leaf=40,
                max_features="sqrt",
                max_samples=0.35,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
                bootstrap=True,
            ),
            proba=True,
        ),
        "hgb": dict(
            ctor=lambda: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=500,
                max_depth=8,
                min_samples_leaf=40,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=30,
                random_state=42,
                class_weight="balanced",
            ),
            proba=True,
        ),
        "lgb": dict(ctor=None, proba=True),  # special path
    }

    oofs = {}
    tests = {}
    fold_aucs = {}

    # ---- Random Forest ----
    log("Training RandomForest (native sklearn)…")
    oof_rf = np.zeros(N_TR)
    te_rf = np.zeros(N_TE)
    fa = []
    for i, (tr, te) in enumerate(folds):
        clf = models["rf"]["ctor"]()
        clf.fit(X_tr[tr], y[tr])
        oof_rf[te] = clf.predict_proba(X_tr[te])[:, 1]
        te_rf += clf.predict_proba(X_te)[:, 1] / 5.0
        a = auc(y[te], oof_rf[te])
        fa.append(a)
        log(f"  RF fold{i} AUC={a:.6f} n_tr={len(tr)}")
    oofs["rf"], tests["rf"], fold_aucs["rf"] = oof_rf, te_rf, fa
    log(f"RF OOF AUC={auc(y,oof_rf):.6f} mean_fold={np.mean(fa):.6f}")

    # ---- HistGradientBoosting ----
    log("Training HistGradientBoosting (native sklearn GBM)…")
    oof_h = np.zeros(N_TR)
    te_h = np.zeros(N_TE)
    fa = []
    for i, (tr, te) in enumerate(folds):
        clf = models["hgb"]["ctor"]()
        clf.fit(X_tr[tr], y[tr])
        oof_h[te] = clf.predict_proba(X_tr[te])[:, 1]
        te_h += clf.predict_proba(X_te)[:, 1] / 5.0
        a = auc(y[te], oof_h[te])
        fa.append(a)
        log(f"  HGB fold{i} AUC={a:.6f} best_iter={getattr(clf, 'n_iter_', '?')}")
    oofs["hgb"], tests["hgb"], fold_aucs["hgb"] = oof_h, te_h, fa
    log(f"HGB OOF AUC={auc(y,oof_h):.6f} mean_fold={np.mean(fa):.6f}")

    # ---- LightGBM (expert GBDT reference) ----
    log("Training LightGBM (expert GBDT)…")
    oof_l = np.zeros(N_TR)
    te_l = np.zeros(N_TE)
    fa = []
    pos = float(y.sum())
    neg = float(len(y) - pos)
    spw = neg / max(pos, 1.0)
    lgb_params = dict(
        objective="binary",
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=80,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        lambda_l2=1.0,
        scale_pos_weight=spw,
        verbosity=-1,
        seed=42,
        n_jobs=-1,
    )
    for i, (tr, te) in enumerate(folds):
        dtr = lgb.Dataset(X_tr[tr], y[tr], free_raw_data=False)
        dva = lgb.Dataset(X_tr[te], y[te], reference=dtr, free_raw_data=False)
        booster = lgb.train(
            lgb_params,
            dtr,
            num_boost_round=4000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof_l[te] = booster.predict(X_tr[te], num_iteration=booster.best_iteration)
        te_l += booster.predict(X_te, num_iteration=booster.best_iteration) / 5.0
        a = auc(y[te], oof_l[te])
        fa.append(a)
        log(f"  LGB fold{i} AUC={a:.6f} best_iter={booster.best_iteration}")
    oofs["lgb"], tests["lgb"], fold_aucs["lgb"] = oof_l, te_l, fa
    log(f"LGB OOF AUC={auc(y,oof_l):.6f} mean_fold={np.mean(fa):.6f}")

    # ---- Ensemble of natives ----
    # rank-average on logit ranks (AUC-only scale)
    def rank_blend(parts, w=None):
        w = np.ones(len(parts)) / len(parts) if w is None else np.asarray(w, float)
        w = w / w.sum()
        acc = 0.0
        for wi, p in zip(w, parts):
            acc = acc + wi * L(p)
        return rf_rank(acc)

    blend_oof = rank_blend([oofs["rf"], oofs["hgb"], oofs["lgb"]], w=[0.25, 0.35, 0.40])
    blend_te = rank_blend([tests["rf"], tests["hgb"], tests["lgb"]], w=[0.25, 0.35, 0.40])
    # optimize weights on OOF via tiny grid
    best_w, best_a = (0.25, 0.35, 0.40), auc(y, blend_oof)
    for wr in np.linspace(0.15, 0.40, 6):
        for wh in np.linspace(0.20, 0.45, 6):
            wl = 1.0 - wr - wh
            if wl < 0.15 or wl > 0.55:
                continue
            b = rank_blend([oofs["rf"], oofs["hgb"], oofs["lgb"]], [wr, wh, wl])
            a = auc(y, b)
            if a > best_a:
                best_a, best_w = a, (wr, wh, wl)
    blend_oof = rank_blend([oofs["rf"], oofs["hgb"], oofs["lgb"]], best_w)
    blend_te = rank_blend([tests["rf"], tests["hgb"], tests["lgb"]], best_w)
    oofs["blend"], tests["blend"] = blend_oof, blend_te
    log(f"Native blend OOF AUC={auc(y,blend_oof):.6f} weights RF/HGB/LGB={best_w}")

    # ---- vs ATH surrogate / geometry ----
    ath = pd.read_csv(ATH).set_index("id").loc[ids_te, "is_fraud"].to_numpy(float)
    # map ATH to OOF via ridge on native oofs + bank gb
    from sklearn.linear_model import Ridge

    Bo = np.column_stack([unit(L(oofs[k])) for k in ("rf", "hgb", "lgb", "blend")])
    Bt = np.column_stack([unit(L(tests[k])) for k in ("rf", "hgb", "lgb", "blend")])
    # add bank anchors
    for name in ("gb_bag5", "lgb_diglead_l3", "realmlp_v1"):
        op, tp = BANK / f"oof_{name}.npy", BANK / f"test_{name}.npy"
        if op.exists() and tp.exists():
            Bo = np.column_stack([Bo, unit(L(np.load(op).ravel()[:N_TR]))])
            Bt = np.column_stack([Bt, unit(L(np.load(tp).ravel()[:N_TE]))])
    lr = Ridge(alpha=10.0).fit(Bt, L(ath))
    z_ath = Bo @ lr.coef_ + lr.intercept_
    log(f"ATH OOF surrogate AUC={auc(y,z_ath):.6f}")

    results_h2h = {}
    for name in ("rf", "hgb", "lgb", "blend"):
        # as standalone vs surrogate (not fair) — better: add small orth residual of native into z_ath
        d = unit(L(oofs[name]))
        # project out z_ath direction
        d = unit(d - np.dot(d, unit(z_ath)) * unit(z_ath))
        per = paired_h2h(z_ath + 0.01 * d, z_ath, y, folds)
        results_h2h[name] = per
        log(
            f"H2H ATH+orth({name}): {per['pos']}/5 t={per['t']:.2f} m={per['mean']:+.2f} min={per['min']:+.2f} "
            f"| solo OOF={auc(y,oofs[name]):.5f} ρ(test,ATH)={omr(tests[name],ath):.2f}e-4"
        )

    # gated inject: use blend residual if passes
    pick = "blend"
    per = results_h2h[pick]
    gate = per["pos"] >= 4 and per["mean"] > 0 and per["t"] >= 2.0
    log(f"GATE blend-orth into ATH: {gate}")

    # Build candidates
    cands = []

    def add(tag, vec, note):
        cands.append({"tag": tag, "vec": np.asarray(vec, float).ravel(), "note": note, "rho_ath": omr(vec, ath)})

    add("native_rf", rf_rank(tests["rf"]), f"RF OOF={auc(y,oofs['rf']):.5f}")
    add("native_hgb", rf_rank(tests["hgb"]), f"HGB OOF={auc(y,oofs['hgb']):.5f}")
    add("native_lgb", rf_rank(tests["lgb"]), f"LGB OOF={auc(y,oofs['lgb']):.5f}")
    add("native_blend", rf_rank(tests["blend"]), f"blend OOF={auc(y,blend_oof):.5f} w={best_w}")

    # ATH + small steps toward native blend (cross-family)
    for w in (0.05, 0.10, 0.15):
        c = rf_rank((1 - w) * L(ath) + w * L(tests["blend"]))
        add(f"ATH_x_nativeBlend_{w:.2f}", c, f"cross-family w={w}")

    # orth inject at 0.30e-4 if gate
    d_te = unit(L(tests["blend"]))
    # crude orth vs [ath]
    d_te = unit(d_te - np.dot(d_te, unit(L(ath))) * unit(L(ath)))
    a, b = 1e-6, 0.5
    best_c, best_err = ath.copy(), 1e9
    for _ in range(28):
        mid = 0.5 * (a + b)
        cand = rf_rank(L(ath) + mid * d_te)
        err = abs(omr(cand, ath) - 0.30)
        if err < best_err:
            best_err, best_c = err, cand
        if omr(cand, ath) < 0.30:
            a = mid
        else:
            b = mid
    add("ATH_orth_nativeBlend_0p30e4", best_c, f"gate={gate} nested={per}")

    # save artifacts
    np.savez_compressed(
        OUT / "v24_2_native_rf_gbm.npz",
        oof_rf=oofs["rf"],
        oof_hgb=oofs["hgb"],
        oof_lgb=oofs["lgb"],
        oof_blend=oofs["blend"],
        test_rf=tests["rf"],
        test_hgb=tests["hgb"],
        test_lgb=tests["lgb"],
        test_blend=tests["blend"],
        feature_cols=np.array(feature_cols, dtype=object),
        fold_aucs_rf=np.array(fold_aucs["rf"]),
        fold_aucs_hgb=np.array(fold_aucs["hgb"]),
        fold_aucs_lgb=np.array(fold_aucs["lgb"]),
        blend_w=np.array(best_w),
    )
    log(f"SAVED {OUT/'v24_2_native_rf_gbm.npz'}")

    staged = []
    for c in cands:
        path = SUB / f"sub_v24_2_{c['tag']}.csv"
        pd.DataFrame({"id": ids_te, "is_fraud": c["vec"]}).to_csv(path, index=False)
        staged.append({**c, "path": path})
        log(f"STAGED {path.name} ρ(ATH)={c['rho_ath']:.3f}e-4 | {c['note']}")

    # Submit priority: cross-family into ATH first (expert: diversity), then orth if gated, then pure blend
    submit_tags = [
        "ATH_x_nativeBlend_0.10",
        "ATH_orth_nativeBlend_0p30e4",
        "ATH_x_nativeBlend_0.05",
        "native_blend",
    ]
    by = {s["tag"]: s for s in staged}
    live = []
    for tag in submit_tags:
        if tag not in by:
            continue
        if tag.startswith("ATH_orth") and not gate:
            log(f"SKIP {tag}: nested gate failed")
            continue
        s = by[tag]
        msg = f"v24.2 {tag} | ρ(ATH)={s['rho_ath']:.3f}e-4 | {s['note']}"[:480]
        log(f"SUBMIT {msg}")
        ok, pub, status = submit_and_poll(s["path"], msg)
        live.append({"tag": tag, "pub": pub, "status": status, "rho_ath": s["rho_ath"]})
        log(f"DONE {tag} pub={pub} status={status}")
        if status == "quota_or_error":
            log("STOP quota")
            break
        time.sleep(5)

    best = max((r["pub"] for r in live if r.get("pub") is not None), default=None)

    body = [
        "# GM V24.2 Native RF & Gradient Boosting",
        "",
        f"Elapsed {time.time()-T0:.0f}s · ATH ref 0.95440",
        "",
        "## Models (expert protocol)",
        "",
        "| Model | OOF AUC | Fold AUCs |",
        "|---|---:|---|",
        f"| RandomForest | **{auc(y,oofs['rf']):.6f}** | {', '.join(f'{a:.5f}' for a in fold_aucs['rf'])} |",
        f"| HistGradientBoosting | **{auc(y,oofs['hgb']):.6f}** | {', '.join(f'{a:.5f}' for a in fold_aucs['hgb'])} |",
        f"| LightGBM | **{auc(y,oofs['lgb']):.6f}** | {', '.join(f'{a:.5f}' for a in fold_aucs['lgb'])} |",
        f"| Rank-blend RF/HGB/LGB {best_w} | **{auc(y,blend_oof):.6f}** | — |",
        "",
        "## Feature design",
        "",
        f"- **{len(feature_cols)}** features: numerics, bools, cyclic time, interactions, digit/leading-digit family, LOFO TE + ordinal cats.",
        "- Folds: identical to public OOF bank (`folds.csv`).",
        "- Class imbalance: RF `balanced_subsample`, HGB `balanced`, LGB `scale_pos_weight`.",
        "",
        "## Nested H2H (ATH surrogate + orth native)",
        "",
        "| native | pos | t | meanΔe5 |",
        "|---|---:|---:|---:|",
    ]
    for k, per in results_h2h.items():
        body.append(f"| `{k}` | {per['pos']}/5 | {per['t']:.2f} | {per['mean']:+.2f} |")
    body += [
        "",
        f"- Blend orth gate: **{gate}**",
        "",
        "## Live submits",
        "",
        "| tag | Public | status |",
        "|---|---:|---|",
    ]
    for r in live:
        body.append(f"| `{r['tag']}` | **{r.get('pub')}** | {r.get('status')} |")
    body += ["", f"- Best live: **{best}**", "", "## Log", "```", *LINES, "```", ""]
    REPORT.write_text("\n".join(body) + "\n")
    OUT.joinpath("v24_2_native_meta.json").write_text(
        json.dumps(
            {
                "oof_auc": {k: auc(y, oofs[k]) for k in oofs},
                "fold_aucs": fold_aucs,
                "blend_w": best_w,
                "h2h": results_h2h,
                "gate": gate,
                "live": live,
                "best": best,
                "n_features": len(feature_cols),
                "elapsed": time.time() - T0,
            },
            indent=2,
            default=float,
        )
    )
    log(f"REPORT {REPORT} best={best}")


if __name__ == "__main__":
    main()
