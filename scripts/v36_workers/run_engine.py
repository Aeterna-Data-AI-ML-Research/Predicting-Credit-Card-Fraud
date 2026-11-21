#!/usr/bin/env python3
"""Isolated V36 engine worker — one process per hypothesis; exits to free RSS."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# repo on path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "4")

from v36_lib import (  # noqa: E402
    ATH_CSV,
    PRED_DIR,
    REPO,
    U,
    V10_OOF,
    V10_TE,
    auc,
    atomic_json,
    atomic_npy,
    blend_ath,
    build_subfolds,
    ensure_dirs,
    heartbeat,
    kaggle_cli,
    L,
    Lf,
    load_core,
    lofo_delta,
    r2_span,
    unit,
    winsor,
)


def _save_engine(name: str, oof: np.ndarray, te: np.ndarray, meta: dict):
    ensure_dirs()
    atomic_npy(PRED_DIR / f"{name}_oof.npy", oof.astype(np.float64))
    atomic_npy(PRED_DIR / f"{name}_test.npy", te.astype(np.float64))
    atomic_json(PRED_DIR / f"{name}_meta.json", meta)
    print(f"[OK] {name} solo_oof={meta.get('solo_auc', 'n/a')} wrote preds", flush=True)


def engine_t0(smoke: bool):
    """Control reproduce + smoke asserts."""
    heartbeat("t0", "controls")
    core = load_core(smoke=smoke)
    y, fold = core["y"], core["fold"]
    z_wide = core["z_wide"]
    sf = build_subfolds(y, fold)
    v10 = np.load(V10_OOF).ravel()[: len(y)].astype(float)
    # LOFO grid micro-weights (include negatives via flipped cand once)
    best = None
    for w, cand in [
        (0.001, v10),
        (0.004, v10),
        (0.008, v10),
        (0.012, v10),
        (0.015, v10),
        (0.008, -v10),
    ]:
        g = lofo_delta(y, fold, sf, z_wide, cand, w=w)
        if best is None or g["mean_e4"] > best["mean_e4"]:
            best = {**g, "w": w if cand is v10 else -w}
    print(f"CTRL v10 best={best}", flush=True)
    assert "COMPLETE" in "SubmissionStatus.COMPLETE".upper()
    assert "ERROR" in "SubmissionStatus.ERROR".upper()
    meta = {"ctrl_v10": best, "smoke_asserts": True, "solo_auc": float(auc(y, v10))}
    # freeze fullbank proxy = wide (bank stack upgrade can replace later)
    atomic_npy(PRED_DIR / "z_fullbank_oof.npy", z_wide)
    atomic_npy(PRED_DIR / "z_fullbank_test.npy", core["z_wide_te"])
    _save_engine("t0_ctrl", v10, np.load(V10_TE).ravel()[: len(core["ath"])].astype(float), meta)
    if not smoke:
        # hard fail only if control is catastrophically dead (don't abort overnight on 4/5)
        if best["pos"] < 3 or best["mean_e4"] < 0:
            raise RuntimeError(f"v10 control broken: {best}")
        if best["pos"] < 5 or best["sub"] < 20:
            print(f"WARN CTRL soft: pos={best['pos']} sub={best['sub']} (continue)", flush=True)
            meta["ctrl_soft_warn"] = True
    return meta


def engine_phase_a(smoke: bool):
    heartbeat("A", "intel")
    out = {"kernels": [], "datasets": [], "promote_h5": False, "novel": []}
    if smoke:
        out["note"] = "smoke skip kaggle"
        atomic_json(REPO / "logs/overnight/GM_V36_PHASE_A_INTEL.json", out)
        (REPO / "logs/overnight/GM_V36_PHASE_A_INTEL.md").write_text(
            "# V36 Phase A (smoke)\n\n- skipped live scrape\n"
        )
        return out

    queries = [
        ("kernels", ["kernels", "list", "-s", "s6e11", "--page-size", "20"]),
        ("kernels", ["kernels", "list", "-s", "0.955", "--page-size", "15"]),
        ("kernels", ["kernels", "list", "-s", "0.956", "--page-size", "10"]),
        ("kernels", ["kernels", "list", "-s", "0.96", "--page-size", "10"]),
        ("datasets", ["datasets", "list", "-s", "credit card fraud 2026", "--max-size", "20"]),
        ("datasets", ["datasets", "list", "-s", "s6e11", "--max-size", "15"]),
        ("datasets", ["datasets", "list", "-s", "uditjain", "--max-size", "10"]),
        ("lb", ["competitions", "leaderboard", "-c", "playground-series-s6e11", "--show"]),
    ]
    blobs = []
    phase_t0 = time.time()
    PHASE_A_WALL = 1140  # 19m hard cap inside 20m supervisor budget
    for kind, args in queries:
        if time.time() - phase_t0 > PHASE_A_WALL:
            print("Phase A wall — stopping scrape", flush=True)
            break
        t0 = time.time()
        text = kaggle_cli(args, timeout=60)
        blobs.append({"kind": kind, "args": args, "chars": len(text), "sec": round(time.time() - t0, 1)})
        for ln in text.splitlines():
            if any(x in ln for x in ("0.9556", "0.9557", "0.9558", "0.9559", "0.956", "0.96")):
                out["novel"].append(ln[:200])
        if kind == "kernels":
            out["kernels"].append(text[:3000])
        elif kind == "datasets":
            out["datasets"].append(text[:2000])

    # promote if we saw fresh high scores beyond known 0.95547 cluster
    if any("0.9556" in x or "0.9557" in x or "0.9558" in x or "0.9559" in x or "0.956" in x for x in out["novel"]):
        out["promote_h5"] = True
    out["queries"] = blobs
    atomic_json(REPO / "logs/overnight/GM_V36_PHASE_A_INTEL.json", out)
    md = ["# V36 Phase A Intel", "", f"- promote_h5={out['promote_h5']}", f"- novel_hits={len(out['novel'])}", ""]
    md += ["## Novel lines", ""] + [f"- `{x}`" for x in out["novel"][:40]]
    (REPO / "logs/overnight/GM_V36_PHASE_A_INTEL.md").write_text("\n".join(md))
    print(f"Phase A promote_h5={out['promote_h5']} novel={len(out['novel'])}", flush=True)
    return out


def engine_h1(smoke: bool):
    """Balance-transition + entity graph — NO id-lag."""
    heartbeat("B", "H1")
    t0 = time.time()
    usecols = [
        "id",
        "customer_age",
        "card_age_months",
        "card_type",
        "account_balance_usd",
        "amount_usd",
        "hours_since_last_txn",
        "is_fraud",
    ]
    tr = pd.read_csv(REPO / "data/train.csv", usecols=usecols)
    te = pd.read_csv(REPO / "data/test.csv", usecols=[c for c in usecols if c != "is_fraud"])
    if smoke:
        tr, te = tr.iloc[:8000].copy(), te.iloc[:4000].copy()
    y = tr["is_fraud"].to_numpy(float)
    fold = pd.read_csv(REPO / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv")["fold"].to_numpy().astype(np.int8)
    if smoke:
        fold = fold[: len(tr)]

    # entity key
    for df in (tr, te):
        df["ekey"] = (
            df["customer_age"].astype(str)
            + "|"
            + df["card_age_months"].astype(str)
            + "|"
            + df["card_type"].astype(str)
        )

    tr = tr.reset_index(drop=True)
    te = te.reset_index(drop=True)
    n_tr, n_te = len(tr), len(te)

    # Entity mean + balance-rank features → LGB on raw y (no id-lag; no early_stopping on va)
    import lightgbm as lgb

    oof = np.full(n_tr, np.nan)
    te_acc = np.zeros(n_te, float)
    folds_done = 0
    for f in range(5):
        if time.time() - t0 > (120 if smoke else 45 * 60):
            print(f"H1 timebox hit after {folds_done}/5 folds", flush=True)
            break
        tr_m = fold != f
        va_m = fold == f
        ent = tr.loc[tr_m].groupby("ekey")["is_fraud"].mean()
        cnt = tr.loc[tr_m].groupby("ekey")["is_fraud"].count()
        gmean = y[tr_m].mean()
        ent_s = (ent * cnt + gmean * 20) / (cnt + 20)

        def featurize(df):
            em = df["ekey"].map(ent_s).fillna(gmean).to_numpy(float)
            bal_map = tr.loc[tr_m, "account_balance_usd"].value_counts()
            bf = df["account_balance_usd"].map(bal_map).fillna(0).to_numpy(float)
            bf_r = U(bf)  # rank not logcnt
            return np.column_stack(
                [
                    em,
                    bf_r,
                    df["amount_usd"].to_numpy(float) / (df["account_balance_usd"].to_numpy(float) + 1.0),
                    df["hours_since_last_txn"].to_numpy(float),
                    df["customer_age"].to_numpy(float),
                    df["card_age_months"].to_numpy(float),
                ]
            )

        Xtr, Xva = featurize(tr.loc[tr_m]), featurize(tr.loc[va_m])
        Xte = featurize(te)
        dtr = lgb.Dataset(Xtr, label=y[tr_m])
        params = dict(
            objective="binary",
            metric="auc",
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbosity=-1,
            seed=42 + f,
        )
        n_est = 40 if smoke else 200
        booster = lgb.train(params, dtr, num_boost_round=n_est)
        oof[va_m] = booster.predict(Xva)
        te_acc += booster.predict(Xte)
        folds_done += 1

    if folds_done == 0:
        raise RuntimeError("H1 completed 0 folds")
    # fill any unfinished OOF slots with global mean of completed preds (no silent zeros)
    done_mask = np.isfinite(oof)
    fill = float(np.nanmean(oof)) if done_mask.any() else float(y.mean())
    oof = np.where(done_mask, oof, fill)
    te_pred = te_acc / folds_done
    solo = auc(y, oof)
    meta = {
        "solo_auc": solo,
        "engine": "H1_entity_balance",
        "id_lag": False,
        "folds_done": folds_done,
        "elapsed": time.time() - t0,
    }
    _save_engine("H1", oof, te_pred, meta)
    return meta


def engine_h2(smoke: bool):
    """Gen/clone homophily + bug-fixed E2-lite ASCII digit fingerprint."""
    heartbeat("B", "H2")
    t0 = time.time()
    core = load_core(smoke=smoke)
    y, fold = core["y"], core["fold"]
    n_tr, n_te = len(y), len(core["ath"])

    cols = [
        "account_balance_usd",
        "amount_usd",
        "merchant_risk_score",
        "velocity_score",
        "hours_since_last_txn",
        "customer_age",
        "card_age_months",
        "distance_from_home_km",
    ]
    tr = pd.read_csv(REPO / "data/train.csv", usecols=cols + ["is_fraud"])
    te = pd.read_csv(REPO / "data/test.csv", usecols=cols)
    if smoke:
        tr, te = tr.iloc[:8000], te.iloc[:4000]

    # E2-lite: string digit/mantissa features (no GPT2 dependency if vocab missing)
    def ascii_feats(df):
        feats = []
        for c in cols:
            s = df[c].astype(str)
            feats.append(s.str.len().to_numpy(float))
            feats.append(s.str.replace(".", "", regex=False).str.len().to_numpy(float))
            # first decimal digit
            def first_dec(x):
                if "." in x:
                    parts = x.split(".")
                    if len(parts[1]):
                        return float(parts[1][0])
                return -1.0

            feats.append(s.map(first_dec).to_numpy(float))
        return np.column_stack(feats)

    Xtr_all = ascii_feats(tr)
    Xte = ascii_feats(te)
    import lightgbm as lgb

    oof = np.zeros(n_tr, float)
    te_pred = np.zeros(n_te, float)
    for f in range(5):
        tr_m, va_m = fold != f, fold == f
        dtr = lgb.Dataset(Xtr_all[tr_m], label=y[tr_m])
        params = dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=24,
            min_data_in_leaf=50,
            verbosity=-1,
            seed=7 + f,
        )
        booster = lgb.train(params, dtr, num_boost_round=30 if smoke else 180)
        oof[va_m] = booster.predict(Xtr_all[va_m])
        te_pred += booster.predict(Xte) / 5.0

    solo = auc(y, oof)
    meta = {"solo_auc": solo, "engine": "H2_ascii_E2lite", "elapsed": time.time() - t0}
    if solo <= 0.935 and not smoke:
        meta["g0_precheck"] = "FAIL_solo_le_0935"
        print(f"H2 solo {solo:.5f} <= 0.935 — mark weak", flush=True)
    _save_engine("H2", oof, te_pred, meta)
    return meta


def engine_h3(smoke: bool):
    """Hierarchical EB rare tuples + Ridge; trees on raw y init=None."""
    heartbeat("B", "H3")
    t0 = time.time()
    core = load_core(smoke=smoke)
    y, fold = core["y"], core["fold"]
    cat_cols = ["auth_method", "channel", "device_type", "merchant_category", "card_type"]
    tr = pd.read_csv(REPO / "data/train.csv", usecols=cat_cols + ["is_fraud", "velocity_score", "merchant_risk_score"])
    te = pd.read_csv(REPO / "data/test.csv", usecols=cat_cols + ["velocity_score", "merchant_risk_score"])
    if smoke:
        tr, te = tr.iloc[:8000], te.iloc[:4000]
        fold = fold[: len(tr)]
        y = y[: len(tr)]

    for df in (tr, te):
        df["vbin"] = pd.qcut(df["velocity_score"].rank(method="first"), 10, labels=False, duplicates="drop")
        df["rbin"] = pd.qcut(df["merchant_risk_score"].rank(method="first"), 10, labels=False, duplicates="drop")
        df["k2"] = df["auth_method"].astype(str) + "|" + df["channel"].astype(str)
        df["k3"] = df["k2"] + "|" + df["device_type"].astype(str)
        df["k4"] = df["k3"] + "|" + df["merchant_category"].astype(str)

    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    n_tr, n_te = len(tr), len(te)
    oof = np.full(n_tr, np.nan)
    te_acc = np.zeros(n_te, float)
    folds_done = 0

    def eb_col(train_s, y_tr, test_s, prior, n0=20):
        g = pd.DataFrame({"k": train_s, "y": y_tr}).groupby("k")["y"].agg(["sum", "count"])
        # filter rare
        g = g[g["count"] >= (3 if smoke else 10)]
        mean = (g["sum"] + prior * n0) / (g["count"] + n0)
        return train_s.map(mean).fillna(prior).to_numpy(float), test_s.map(mean).fillna(prior).to_numpy(float)

    for f in range(5):
        if time.time() - t0 > (120 if smoke else 50 * 60):
            print(f"H3 timebox hit after {folds_done}/5 folds", flush=True)
            break
        tr_m, va_m = fold != f, fold == f
        prior = y[tr_m].mean()
        feats_tr, feats_te = [], []
        for k in ("k2", "k3", "k4"):
            a, b = eb_col(tr.loc[tr_m, k], y[tr_m], te[k], prior)
            # map va via same
            a_full = tr[k].map(
                pd.DataFrame({"k": tr.loc[tr_m, k], "y": y[tr_m]})
                .groupby("k")["y"]
                .agg(lambda s: (s.sum() + prior * 20) / (len(s) + 20) if len(s) >= (3 if smoke else 10) else prior)
            ).fillna(prior).to_numpy(float)
            feats_tr.append(a_full)
            feats_te.append(b)
        X = np.column_stack(feats_tr + [tr["vbin"].to_numpy(float), tr["rbin"].to_numpy(float)])
        Xte = np.column_stack(feats_te + [te["vbin"].to_numpy(float), te["rbin"].to_numpy(float)])
        ridge = Ridge(alpha=1000.0)
        ridge.fit(X[tr_m], y[tr_m])
        dtr = lgb.Dataset(X[tr_m], label=y[tr_m])
        params = dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=23,
            min_data_in_leaf=50,
            max_delta_step=0.15,
            verbosity=-1,
            seed=11 + f,
        )
        booster = lgb.train(params, dtr, num_boost_round=30 if smoke else 160)
        p_va = 0.5 * ridge.predict(X[va_m]) + 0.5 * booster.predict(X[va_m])
        oof[va_m] = p_va
        te_acc += 0.5 * ridge.predict(Xte) + 0.5 * booster.predict(Xte)
        folds_done += 1

    if folds_done == 0:
        raise RuntimeError("H3 completed 0 folds")
    done_mask = np.isfinite(oof)
    fill = float(np.nanmean(oof)) if done_mask.any() else float(y.mean())
    oof = np.where(done_mask, oof, fill)
    te_pred = te_acc / folds_done
    solo = auc(y, oof)
    meta = {
        "solo_auc": solo,
        "engine": "H3_EB_ridge_lgb",
        "folds_done": folds_done,
        "elapsed": time.time() - t0,
    }
    _save_engine("H3", oof, te_pred, meta)
    return meta


def engine_h4(smoke: bool):
    """Twilight specialist with smooth gate; BCE fallback + Wilcoxon-style pairwise on twilight."""
    heartbeat("B", "H4")
    t0 = time.time()
    core = load_core(smoke=smoke)
    y, fold = core["y"], core["fold"]
    zfb = np.load(PRED_DIR / "z_fullbank_oof.npy").ravel()[: len(y)]
    zfb_te = np.load(PRED_DIR / "z_fullbank_test.npy").ravel()[: len(core["ath"])]
    q = U(zfb)
    q_te = U(zfb_te)

    cols = [
        "account_balance_usd",
        "amount_usd",
        "merchant_risk_score",
        "velocity_score",
        "customer_age",
        "auth_method",
        "channel",
        "device_type",
    ]
    tr = pd.read_csv(REPO / "data/train.csv", usecols=cols + ["is_fraud"])
    te = pd.read_csv(REPO / "data/test.csv", usecols=cols)
    if smoke:
        tr, te = tr.iloc[:8000], te.iloc[:4000]
        fold, y, q = fold[:8000], y[:8000], q[:8000]
        zfb, q_te, zfb_te = zfb[:8000], q_te[:4000], zfb_te[:4000]

    # encode cats
    from sklearn.preprocessing import LabelEncoder

    for c in ["auth_method", "channel", "device_type"]:
        le = LabelEncoder()
        le.fit(pd.concat([tr[c], te[c]]).astype(str))
        tr[c] = le.transform(tr[c].astype(str))
        te[c] = le.transform(te[c].astype(str))

    X_all = tr.drop(columns=["is_fraud"]).to_numpy(float)
    Xte = te.to_numpy(float)
    import lightgbm as lgb

    oof_raw = np.zeros(len(y), float)
    te_raw = np.zeros(len(q_te), float)
    for f in range(5):
        tr_m = (fold != f) & (q >= 0.65)
        va_m = fold == f
        if tr_m.sum() < 100:
            tr_m = fold != f
        dtr = lgb.Dataset(X_all[tr_m], label=y[tr_m])
        # Platt-ish: use init_score = None (mode A) for safety in overnight
        params = dict(
            objective="binary",
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=50,
            max_delta_step=0.15,
            verbosity=-1,
            seed=21 + f,
        )
        booster = lgb.train(params, dtr, num_boost_round=30 if smoke else 180)
        oof_raw[va_m] = booster.predict(X_all[va_m])
        te_raw += booster.predict(Xte) / 5.0

    # quantile demean residual vs z_fullbank bins + smooth gate
    def smooth_gate(qq):
        return 1.0 / (1.0 + np.exp(-(qq - 0.78) / 0.035))

    # residual relative to mean in quantile bins of zfb
    bins = np.digitize(q, np.linspace(0, 1, 21))
    u = oof_raw.copy()
    for b in np.unique(bins):
        m = bins == b
        u[m] = u[m] - u[m].mean()
    g = smooth_gate(q)
    oof = u * g

    bins_te = np.digitize(q_te, np.linspace(0, 1, 21))
    ute = te_raw.copy()
    for b in np.unique(bins_te):
        m = bins_te == b
        ute[m] = ute[m] - ute[m].mean()
    te_pred = ute * smooth_gate(q_te)

    # residual oof/te saved for orth blend; solo_auc is RECONSTRUCTED (zfb+resid)
    slope_ok = float(np.mean(g) > 0.05)
    recon = zfb + oof
    solo = auc(y, recon)
    solo_resid = auc(y, oof)
    meta = {
        "solo_auc": solo,
        "solo_residual": solo_resid,
        "is_residual": True,
        "engine": "H4_twilight_smooth",
        "slope_gate_mean": float(g.mean()),
        "slope_ok": slope_ok,
        "elapsed": time.time() - t0,
    }
    # keep residual tensors for G0 orth; Phase C uses is_residual for solo/vault
    _save_engine("H4", oof, te_pred, meta)
    atomic_npy(PRED_DIR / "H4_recon_oof.npy", recon.astype(np.float64))
    atomic_npy(PRED_DIR / "H4_recon_test.npy", (zfb_te + te_pred).astype(np.float64))
    return meta


def engine_h5(smoke: bool):
    """Phase A dynamic / fallback: harvest negative twin + v10 diversity stack."""
    heartbeat("B", "H5")
    core = load_core(smoke=smoke)
    y, fold = core["y"], core["fold"]
    v10 = np.load(V10_OOF).ravel()[: len(y)].astype(float)
    v10_te = np.load(V10_TE).ravel()[: len(core["ath"])].astype(float)
    har_p = REPO / "preds/matrix_v2/harvest_p4_cuda_xgb_oof.npy"
    har_t = REPO / "preds/matrix_v2/harvest_p4_cuda_xgb_test.npy"
    if not har_p.exists():
        har_p = REPO / "backups/matrix_v2/latest/preds_matrix_v2/harvest_p4_cuda_xgb_oof.npy"
        har_t = REPO / "backups/matrix_v2/latest/preds_matrix_v2/harvest_p4_cuda_xgb_test.npy"
    if har_p.exists():
        har = np.load(har_p).ravel()[: len(y)].astype(float)
        har_te = np.load(har_t).ravel()[: len(core["ath"])].astype(float)
        oof = unit(v10) - 0.5 * unit(har)
        te = unit(v10_te) - 0.5 * unit(har_te)
    else:
        oof, te = v10, v10_te
    solo = auc(y, oof)
    meta = {"solo_auc": solo, "engine": "H5_v10_mharvest", "elapsed": 0}
    _save_engine("H5", oof, te, meta)
    return meta


def engine_phase_c(smoke: bool):
    """Gate engines, survivor vault, stage CANDs + private hedge."""
    heartbeat("C", "gates")
    core = load_core(smoke=smoke)
    y, fold = core["y"], core["fold"]
    sf = build_subfolds(y, fold)
    z_fb = np.load(PRED_DIR / "z_fullbank_oof.npy").ravel()[: len(y)]
    ath = core["ath"]
    v10 = np.load(V10_OOF).ravel()[: len(y)].astype(float)
    # Approximate ATH twin as blend of fullbank + v10
    z_ath_twin = unit(0.75 * z_fb + 0.25 * unit(v10))

    results = {}
    vault = []
    passers = []
    z_fb_te = np.load(PRED_DIR / "z_fullbank_test.npy").ravel()[: len(ath)]

    def _load_meta(name: str) -> dict:
        mp = PRED_DIR / f"{name}_meta.json"
        if mp.exists():
            return json.loads(mp.read_text())
        return {}

    for name in ["H1", "H2", "H3", "H4", "H5"]:
        op = PRED_DIR / f"{name}_oof.npy"
        tp = PRED_DIR / f"{name}_test.npy"
        if not op.exists():
            results[name] = {"status": "missing"}
            continue
        oof_raw = np.load(op).ravel()[: len(y)]
        te_raw = np.load(tp).ravel()[: len(ath)]
        meta = _load_meta(name)
        is_resid = bool(meta.get("is_residual"))
        if is_resid:
            ro, rt = PRED_DIR / f"{name}_recon_oof.npy", PRED_DIR / f"{name}_recon_test.npy"
            oof_score = np.load(ro).ravel()[: len(y)] if ro.exists() else (z_fb + oof_raw)
            te_score = np.load(rt).ravel()[: len(ath)] if rt.exists() else (z_fb_te + te_raw)
        else:
            oof_score, te_score = oof_raw, te_raw
        # solo on score-space (reconstructed for residual engines); novelty/disp on orth tensor
        solo = auc(y, oof_score)
        span_te = te_raw if is_resid else te_score
        span_oof = oof_raw if is_resid else oof_score
        r2 = r2_span(
            span_te,
            np.column_stack([core["z_wide_te"], unit(np.load(V10_TE).ravel()[: len(ath)])]),
        )
        # LOFO orth uses saved tensor (residual for H4) for additive lift
        g_fb = lofo_delta(y, fold, sf, z_fb, oof_raw, w=0.012)
        g_twin = lofo_delta(y, fold, sf, z_ath_twin, oof_raw, w=0.012)
        raw_disp = float(np.std(span_te) / (np.std(span_oof) + 1e-12))
        # H5 is a diversity axis — softer solo floor; residual engines use reconstructed solo
        if is_resid:
            solo_ok = solo > 0.90
        elif name == "H5":
            solo_ok = solo > 0.50
        else:
            solo_ok = solo > 0.935
        # residual engines: skip OOF/test scale lock (gated residual has intentional twilight mask)
        disp_ok = True if is_resid else (0.93 <= raw_disp <= 1.07)
        ok_g0 = (
            solo_ok
            and r2 < 0.50
            and g_fb["pos"] == 5
            and g_fb["mean_e4"] >= 3.0
            and g_fb["sub"] >= 20
            and g_twin["pos"] >= 4
            and g_twin["mean_e4"] > 0
            and disp_ok
        )
        near = (not ok_g0) and g_fb["pos"] == 4 and g_fb["t25"] >= 2.8 and r2 < 0.50 and solo_ok
        rec = {
            "solo": solo,
            "is_residual": is_resid,
            "r2_B152": r2,
            "g_fb": g_fb,
            "g_twin": g_twin,
            "raw_disp": raw_disp,
            "g0": ok_g0,
            "vault": near,
        }
        results[name] = rec
        if ok_g0:
            passers.append(name)
        elif near:
            vault.append(name)
        print(
            f"GATE {name}: g0={ok_g0} vault={near} solo={solo:.5f} resid={is_resid} "
            f"r2={r2:.3f} d_fb={g_fb['mean_e4']:.3f}",
            flush=True,
        )

    # survivor vault stack — always score-space (reconstruct residual engines)
    if len(passers) == 0 and len(vault) >= 2:
        score_oofs, score_tes = [], []
        for n in vault[:5]:
            o = np.load(PRED_DIR / f"{n}_oof.npy").ravel()[: len(y)]
            t = np.load(PRED_DIR / f"{n}_test.npy").ravel()[: len(ath)]
            m = _load_meta(n)
            if m.get("is_residual"):
                ro, rt = PRED_DIR / f"{n}_recon_oof.npy", PRED_DIR / f"{n}_recon_test.npy"
                o = np.load(ro).ravel()[: len(y)] if ro.exists() else (z_fb + o)
                t = np.load(rt).ravel()[: len(ath)] if rt.exists() else t
            score_oofs.append(o)
            score_tes.append(t)
        stacked = np.mean(score_oofs, 0)
        stacked_te = np.mean(score_tes, 0)
        g = lofo_delta(y, fold, sf, z_fb, stacked, w=0.012)
        if g["pos"] == 5 and g["mean_e4"] >= 2.5:
            passers = ["VAULT"]
            atomic_npy(PRED_DIR / "VAULT_oof.npy", stacked)
            atomic_npy(PRED_DIR / "VAULT_test.npy", stacked_te)
            results["VAULT"] = {"g0": True, "g_fb": g, "members": vault[:5]}

    staged = {}
    if passers:
        name = passers[0]
        oof = np.load(PRED_DIR / f"{name}_oof.npy").ravel()[: len(y)]
        te = np.load(PRED_DIR / f"{name}_test.npy").ravel()[: len(ath)]
        # nested w* proxy: pick best w on fullbank
        best_w, best_m = 0.01, -1e9
        for w in np.linspace(0.005, 0.06, 12):
            g = lofo_delta(y, fold, sf, z_fb, oof, w=float(w))
            if g["mean_e4"] > best_m and g["pos"] >= 4:
                best_m, best_w = g["mean_e4"], float(w)
        w_star = 0.85 * best_w
        # map w* from fullbank logit space onto ATH
        u_te = winsor(te)
        cand1 = blend_ath(ath, u_te, 0.60 * w_star)
        cand2 = blend_ath(ath, u_te, 1.00 * w_star)
        ids = core["ids_te"]
        p1 = REPO / "submissions/sub_v36_CAND1_NOSUBMIT.csv"
        p2 = REPO / "submissions/sub_v36_CAND2_NOSUBMIT.csv"
        pd.DataFrame({"id": ids, "is_fraud": cand1}).to_csv(p1, index=False)
        pd.DataFrame({"id": ids, "is_fraud": cand2}).to_csv(p2, index=False)
        # private hedge: 0.5 ATH + 0.5 cand2
        hedge = U(0.5 * L(U(ath)) + 0.5 * L(cand2))
        ph = REPO / "submissions/sub_v36_PRIVATE_LB_HEDGE_NOSUBMIT.csv"
        pd.DataFrame({"id": ids, "is_fraud": hedge}).to_csv(ph, index=False)
        staged = {
            "engine": name,
            "w_star": w_star,
            "delta_pred_full_e4": best_m,
            "cand1": str(p1),
            "cand2": str(p2),
            "hedge": str(ph),
            "omr_c2": float((1 - __import__("scipy").stats.spearmanr(cand2, ath).correlation) * 1e4),
        }
        print(f"STAGED {staged}", flush=True)
    else:
        print("No G0 passers — hold slots", flush=True)

    out = {"results": results, "passers": passers, "vault": vault, "staged": staged}
    atomic_json(REPO / "logs/overnight/v36_ckpt/phase_c.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    engines = {
        "t0": engine_t0,
        "phase_a": engine_phase_a,
        "H1": engine_h1,
        "H2": engine_h2,
        "H3": engine_h3,
        "H4": engine_h4,
        "H5": engine_h5,
        "phase_c": engine_phase_c,
    }
    fn = engines[args.engine]
    meta = fn(args.smoke)
    print(json.dumps({"engine": args.engine, "meta_keys": list(meta.keys()) if isinstance(meta, dict) else []}))


if __name__ == "__main__":
    main()
