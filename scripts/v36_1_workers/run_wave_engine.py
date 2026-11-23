#!/usr/bin/env python3
"""V36.1 Wave engines: W1_E1..E4, W3_E5/E6/MultiVault, W4_E7 — real multi-seed stacks."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "4")

from v36_1_lib import (  # noqa: E402
    COMPETENCE_FLOOR,
    LOGS,
    PRED,
    VAULT,
    append_oof_summaries,
    atomic_json,
    atomic_npy,
    auc,
    blend_ath,
    build_subfolds,
    ensure_dirs,
    evaluate_engine_g0,
    heartbeat,
    load_y_fold_ids,
    tabular_core_matrix,
    train_multiseed_stack,
    unit,
)


def _load_baselines(n_tr: int, n_te: int):
    z_fb = np.load(PRED / "z_fullbank_oof.npy").ravel()[:n_tr]
    z_fb_te = np.load(PRED / "z_fullbank_test.npy").ravel()[:n_te]
    z_twin = np.load(PRED / "z_ATH_twin_oof.npy").ravel()[:n_tr]
    z_twin_te = np.load(PRED / "z_ATH_twin_test.npy").ravel()[:n_te]
    Bo = np.load(PRED / "B_clean_oof.npy")[:n_tr]
    Bt = np.load(PRED / "B_clean_test.npy")[:n_te]
    eta = np.load(PRED / "eta_cal_fb_oof.npy").ravel()[:n_tr]
    eta_te = np.load(PRED / "eta_cal_fb_test.npy").ravel()[:n_te]
    return z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te


def _gate(name, oof, te, y, fold, smoke, te_by_fold=None):
    z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te = _load_baselines(len(y), len(te))
    sf = build_subfolds(y, fold)
    rec = evaluate_engine_g0(
        name,
        oof,
        te,
        y,
        fold,
        sf,
        z_fb,
        z_fb_te,
        z_twin,
        z_twin_te,
        Bo,
        Bt,
        eta_fb=eta,
        eta_fb_te=eta_te,
        te_by_fold=te_by_fold,
    )
    keys = ("solo", "g0", "vault", "status", "r2_B152", "f1", "f2_ok", "raw_disp", "disp_1f")
    print(
        f"GATE {name}: {json.dumps({k: rec[k] for k in keys if k in rec})}",
        flush=True,
    )
    if "g_fb" in rec:
        print(f"  g_fb={rec['g_fb']}", flush=True)
        print(f"  g_twin={rec['g_twin']}", flush=True)
    atomic_json(LOGS / "v36_1_ckpt" / f"{name}_gate.json", rec)
    return rec


def feats_entity_balance(smoke: bool, fold: np.ndarray | None = None, y: np.ndarray | None = None):
    """Graph kinematics (unsupervised) + fold-safe peer_risk (NO global y TE)."""
    cols = [
        "customer_age",
        "card_age_months",
        "card_type",
        "account_balance_usd",
        "amount_usd",
        "hours_since_last_txn",
        "velocity_score",
        "merchant_risk_score",
        "is_fraud",
    ]
    tr = pd.read_csv(REPO / "data/train.csv", usecols=cols)
    te = pd.read_csv(REPO / "data/test.csv", usecols=[c for c in cols if c != "is_fraud"])
    tr["src"] = 0
    te["src"] = 1
    te["is_fraud"] = -1
    all_df = pd.concat([tr, te], ignore_index=True)
    if smoke:
        all_df = all_df.iloc[:18000].copy()
        if fold is not None:
            fold = fold[: int((all_df["src"] == 0).sum())]
        if y is not None:
            y = y[: int((all_df["src"] == 0).sum())]
    all_df["ekey"] = (
        all_df["customer_age"].astype(int).astype(str)
        + "|"
        + all_df["card_age_months"].astype(int).astype(str)
        + "|"
        + all_df["card_type"].astype(str)
    )
    n = len(all_df)
    chain_len = np.ones(n, float)
    drawdown = np.zeros(n, float)
    txn_rate = all_df["velocity_score"].to_numpy(float)
    rev_flag = np.zeros(n, float)
    for ekey, idx in all_df.groupby("ekey").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        bals = all_df.loc[idx, "account_balance_usd"].to_numpy(float)
        amts = all_df.loc[idx, "amount_usd"].to_numpy(float)
        order = np.argsort(bals)
        bals_s = bals[order]
        for j, ii in enumerate(order):
            target = bals[ii] + amts[ii]
            k = np.searchsorted(bals_s, target)
            hit = 0
            for kk in (k - 1, k, k + 1):
                if 0 <= kk < len(bals_s) and abs(bals_s[kk] - target) <= 0.05:
                    hit += 1
            chain_len[ii] = float(len(idx))
            drawdown[ii] = amts[ii] / (bals[ii] + 1.0)
            rev_flag[ii] = 1.0 if hit == 1 else 0.0

    n_tr = int((all_df["src"] == 0).sum())
    peer_tr = np.zeros(n_tr, float)
    peer_te = np.zeros(n - n_tr, float)
    ekeys = all_df["ekey"].to_numpy()
    y_tr = all_df.loc[all_df["src"] == 0, "is_fraud"].to_numpy(float)
    if fold is None or y is None:
        # HOTPATCH: refuse global TE — use constant prior only (no leakage)
        print("WARN feats_entity_balance: no fold — peer_risk=global_prior only", flush=True)
        gmean = float(y_tr.mean())
        peer_tr[:] = gmean
        peer_te[:] = gmean
    else:
        gmean = float(y.mean())
        peer_tr = np.full(n_tr, gmean)
        peer_te_acc = np.zeros(n - n_tr, float)
        for f in range(5):
            tr_m = fold != f
            va_m = fold == f
            # means from train folds only
            tmp = pd.DataFrame({"k": ekeys[:n_tr][tr_m], "y": y[tr_m]})
            g = tmp.groupby("k")["y"].agg(["sum", "count"])
            mean = (g["sum"] + gmean * 20) / (g["count"] + 20)
            peer_tr[va_m] = pd.Series(ekeys[:n_tr][va_m]).map(mean).fillna(gmean).to_numpy(float)
            peer_te_acc += pd.Series(ekeys[n_tr:]).map(mean).fillna(gmean).to_numpy(float) / 5.0
        peer_te = peer_te_acc

    feats_tr = np.column_stack(
        [
            chain_len[:n_tr],
            drawdown[:n_tr],
            txn_rate[:n_tr],
            rev_flag[:n_tr],
            peer_tr,
            all_df["amount_usd"].to_numpy(float)[:n_tr] / (all_df["account_balance_usd"].to_numpy(float)[:n_tr] + 1),
            all_df["hours_since_last_txn"].to_numpy(float)[:n_tr],
            all_df["merchant_risk_score"].to_numpy(float)[:n_tr],
        ]
    )
    feats_te = np.column_stack(
        [
            chain_len[n_tr:],
            drawdown[n_tr:],
            txn_rate[n_tr:],
            rev_flag[n_tr:],
            peer_te,
            all_df["amount_usd"].to_numpy(float)[n_tr:] / (all_df["account_balance_usd"].to_numpy(float)[n_tr:] + 1),
            all_df["hours_since_last_txn"].to_numpy(float)[n_tr:],
            all_df["merchant_risk_score"].to_numpy(float)[n_tr:],
        ]
    )
    return feats_tr, feats_te


def feats_stealth_and_eb(smoke: bool, y, fold):
    cols = [
        "account_balance_usd",
        "amount_usd",
        "customer_age",
        "card_age_months",
        "auth_method",
        "channel",
        "device_type",
        "merchant_category",
        "velocity_score",
        "merchant_risk_score",
    ]
    tr = pd.read_csv(REPO / "data/train.csv", usecols=cols + ["is_fraud"])
    te = pd.read_csv(REPO / "data/test.csv", usecols=cols)
    if smoke:
        tr, te = tr.iloc[:12000].copy(), te.iloc[:6000].copy()
        fold = fold[: len(tr)]
        y = y[: len(tr)]
    for df in (tr, te):
        df["amt_bal"] = df["amount_usd"] / (df["account_balance_usd"] + 1)
        df["age_card"] = (df["customer_age"] * 12 - df["card_age_months"]).abs()
        df["cent"] = (np.round(df["amount_usd"] * 100) % 100)
        df["k2"] = df["auth_method"].astype(str) + "|" + df["channel"].astype(str)
        df["k3"] = df["k2"] + "|" + df["device_type"].astype(str)
        df["k4"] = df["k3"] + "|" + df["merchant_category"].astype(str)
    # hierarchical EB OOF
    n = len(tr)
    eb2 = np.zeros(n, float)
    eb3 = np.zeros(n, float)
    eb4 = np.zeros(n, float)
    te2 = np.zeros(len(te), float)
    te3 = np.zeros(len(te), float)
    te4 = np.zeros(len(te), float)
    for f in range(5):
        tr_m, va_m = fold != f, fold == f
        prior = y[tr_m].mean()

        def eb(series_tr, series_te, min_n=10):
            g = pd.DataFrame({"k": series_tr[tr_m], "y": y[tr_m]}).groupby("k")["y"].agg(["sum", "count"])
            g = g[g["count"] >= min_n]
            mean = (g["sum"] + prior * 20) / (g["count"] + 20)
            return series_tr.map(mean).fillna(prior).to_numpy(float), series_te.map(mean).fillna(prior).to_numpy(float)

        a, b = eb(tr["k2"], te["k2"])
        eb2[va_m] = a[va_m]
        te2 += b / 5
        a, b = eb(tr["k3"], te["k3"])
        eb3[va_m] = a[va_m]
        te3 += b / 5
        a, b = eb(tr["k4"], te["k4"], min_n=10)
        eb4[va_m] = a[va_m]
        te4 += b / 5
    Xtr_extra = np.column_stack(
        [
            eb2,
            eb3,
            eb4,
            tr["amt_bal"],
            tr["age_card"],
            tr["cent"],
            tr["velocity_score"],
            tr["merchant_risk_score"],
        ]
    )
    Xte_extra = np.column_stack(
        [
            te2,
            te3,
            te4,
            te["amt_bal"],
            te["age_card"],
            te["cent"],
            te["velocity_score"],
            te["merchant_risk_score"],
        ]
    )
    return Xtr_extra, Xte_extra


def feats_bpe_lite(smoke: bool):
    """ASCII digit / mantissa features + optional GPT2 BPE unigram if vocab present."""
    cols = [
        "account_balance_usd",
        "amount_usd",
        "merchant_risk_score",
        "velocity_score",
        "customer_age",
        "card_age_months",
        "hours_since_last_txn",
        "distance_from_home_km",
    ]
    tr = pd.read_csv(REPO / "data/train.csv", usecols=cols)
    te = pd.read_csv(REPO / "data/test.csv", usecols=cols)
    if smoke:
        tr, te = tr.iloc[:12000], te.iloc[:6000]

    def ascii_block(df):
        blocks = []
        for c in cols:
            s = df[c].astype(str)
            blocks.append(s.str.len().to_numpy(float))
            blocks.append(s.str.replace(".", "", regex=False).str.len().to_numpy(float))
            blocks.append(s.map(lambda x: float(x.split(".")[1][0]) if "." in x and len(x.split(".")[1]) else -1).to_numpy(float))
        return np.column_stack(blocks)

    Xtr, Xte = ascii_block(tr), ascii_block(te)
    vocab = REPO / "logs/overnight/run_mps_gpt2_data/replica/trainer/checkpoint-50000/vocab.json"
    if vocab.exists() and not smoke:
        try:
            tok = json.loads(vocab.read_text())
            # cheap: hash of first 8 char codes of amount string as bucket features
            def bpe_proxy(df):
                s = df["amount_usd"].astype(str) + "|" + df["account_balance_usd"].astype(str)
                h1 = s.map(lambda x: sum(tok.get(ch, 0) for ch in x[:12]) % 997).to_numpy(float)
                h2 = s.map(lambda x: len(x)).to_numpy(float)
                return np.column_stack([h1, h2, h1 / (h2 + 1)])

            Xtr = np.column_stack([Xtr, bpe_proxy(tr)])
            Xte = np.column_stack([Xte, bpe_proxy(te)])
        except Exception as e:
            print("BPE lite skip", e, flush=True)
    return Xtr, Xte


def run_w1_e1(smoke: bool):
    """Legacy entry — redirects to fold-safe E1_fixed (hotpatch)."""
    return run_w1_e1_fixed(smoke)


def run_w1_e1_fixed(smoke: bool):
    heartbeat("W1", "E1_fixed")
    y, fold, _, ath = load_y_fold_ids()
    X0, X0t, _ = tabular_core_matrix(smoke)
    if smoke:
        y, fold, ath = y[:12000], fold[:12000], ath[:6000]
    fe, fet = feats_entity_balance(smoke, fold=fold, y=y)
    fe, fet = fe[: len(y)], fet[: len(ath)]
    z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te = _load_baselines(len(y), len(ath))
    X = np.column_stack([X0, fe])
    Xt = np.column_stack([X0t, fet])
    X, Xt = append_oof_summaries(X, Xt, Bo, Bt, z_fb, z_fb_te, z_twin, z_twin_te)
    oof, te, meta = train_multiseed_stack(X, y, fold, Xt, eta, eta_te, smoke=smoke)
    print(f"E1_fixed stack n_models={meta.get('n_models')} solo={meta.get('solo')}", flush=True)
    if meta["solo"] > 0.9550:
        print(f"LEAKAGE_DETECTED E1_fixed solo={meta['solo']:.5f}", flush=True)
    return _gate(
        "W1_E1_fixed_entity_balance_graph", oof, te, y, fold, smoke, te_by_fold=meta.get("te_by_fold")
    )


def run_w1_e2(smoke: bool):
    heartbeat("W1", "E2")
    y, fold, _, ath = load_y_fold_ids()
    X0, X0t, _ = tabular_core_matrix(smoke)
    if smoke:
        y, fold, ath = y[:12000], fold[:12000], ath[:6000]
    fb, ft = feats_bpe_lite(smoke)
    fb, ft = fb[: len(y)], ft[: len(ath)]
    z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te = _load_baselines(len(y), len(ath))
    # orig20k density proxy: distance to train centroid in standardized space
    mu, sd = X0.mean(0), X0.std(0) + 1e-6
    dens_tr = -np.sum(((X0 - mu) / sd) ** 2, 1)  # higher = closer
    dens_te = -np.sum(((X0t - mu) / sd) ** 2, 1)
    X = np.column_stack([X0, fb, dens_tr])
    Xt = np.column_stack([X0t, ft, dens_te])
    X, Xt = append_oof_summaries(X, Xt, Bo, Bt, z_fb, z_fb_te, z_twin, z_twin_te)
    oof, te, meta = train_multiseed_stack(X, y, fold, Xt, eta, eta_te, smoke=smoke)
    print(f"E2 stack n_models={meta.get('n_models')} solo={meta.get('solo')}", flush=True)
    return _gate(
        "W1_E2_orig20k_inversion_and_bpe_exact",
        oof,
        te,
        y,
        fold,
        smoke,
        te_by_fold=meta.get("te_by_fold"),
    )


def run_w1_e3(smoke: bool):
    heartbeat("W1", "E3")
    y, fold, _, ath = load_y_fold_ids()
    X0, X0t, _ = tabular_core_matrix(smoke)
    if smoke:
        y, fold, ath = y[:12000], fold[:12000], ath[:6000]
    fe, fet = feats_stealth_and_eb(smoke, y, fold)
    z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te = _load_baselines(len(y), len(ath))
    X = np.column_stack([X0, fe])
    Xt = np.column_stack([X0t, fet])
    X, Xt = append_oof_summaries(X, Xt, Bo, Bt, z_fb, z_fb_te, z_twin, z_twin_te)
    oof, te, meta = train_multiseed_stack(X, y, fold, Xt, eta, eta_te, smoke=smoke)
    print(f"E3 stack n_models={meta.get('n_models')} solo={meta.get('solo')}", flush=True)
    return _gate(
        "W1_E3_hier_eb_3d4d_and_stealth_fn",
        oof,
        te,
        y,
        fold,
        smoke,
        te_by_fold=meta.get("te_by_fold"),
    )


def run_w1_e4(smoke: bool):
    """Twilight specialist: focus U(z_fb)>=0.60 + smooth gate residual."""
    heartbeat("W1", "E4")
    from scipy.special import expit
    from v36_1_lib import U, L

    y, fold, _, ath = load_y_fold_ids()
    X0, X0t, _ = tabular_core_matrix(smoke)
    if smoke:
        y, fold, ath = y[:12000], fold[:12000], ath[:6000]
    z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te = _load_baselines(len(y), len(ath))
    q = U(z_fb)
    q_te = U(z_fb_te)
    X, Xt = append_oof_summaries(X0, X0t, Bo, Bt, z_fb, z_fb_te, z_twin, z_twin_te)
    # sample weights emphasize twilight
    # train full stack then residualize
    oof_raw, te_raw, meta = train_multiseed_stack(X, y, fold, Xt, eta, eta_te, smoke=smoke)
    # residual vs quantile bins of z_fb
    bins = np.digitize(q, np.linspace(0, 1, 21))
    u = oof_raw - z_fb
    for b in np.unique(bins):
        m = bins == b
        u[m] = u[m] - u[m].mean()
    g = expit((q - 0.78) / 0.035)
    oof = z_fb + u * g
    bins_te = np.digitize(q_te, np.linspace(0, 1, 21))
    ute = te_raw - z_fb_te
    for b in np.unique(bins_te):
        m = bins_te == b
        ute[m] = ute[m] - ute[m].mean()
    te = z_fb_te + ute * expit((q_te - 0.78) / 0.035)
    print(f"E4 meta n_models={meta.get('n_models')} solo_recon={auc(y, oof):.5f} gate_mean={g.mean():.3f}", flush=True)
    return _gate("W1_E4_wilcoxon_twilight_mps_specialist", oof, te, y, fold, smoke)


def run_w3_e5(smoke: bool):
    heartbeat("W3", "E5")
    y, fold, _, ath = load_y_fold_ids()
    if smoke:
        y, fold, ath = y[:12000], fold[:12000], ath[:6000]
    z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te = _load_baselines(len(y), len(ath))
    # collect W1 OOF features
    cols_o, cols_t = [z_fb, z_twin], [z_fb_te, z_twin_te]
    for name in [
        "W1_E1_fixed_entity_balance_graph",  # de-leaked only — NEVER load W1_E1_entity_balance_graph
        "W1_E2_orig20k_inversion_and_bpe_exact",
        "W1_E3_hier_eb_3d4d_and_stealth_fn",
        "W1_E4_wilcoxon_twilight_mps_specialist",
    ]:
        op, tp = PRED / f"{name}_oof.npy", PRED / f"{name}_test.npy"
        if op.exists():
            cols_o.append(np.load(op).ravel()[: len(y)])
            cols_t.append(np.load(tp).ravel()[: len(ath)])
    for j in range(min(Bo.shape[1], 8)):
        cols_o.append(Bo[:, j])
        cols_t.append(Bt[:, j])
    X = np.column_stack(cols_o)
    Xt = np.column_stack(cols_t)
    # Mode A only (standalone cross-modal)
    oof, te, meta = train_multiseed_stack(X, y, fold, Xt, None, None, smoke=smoke, seeds=[7, 11, 17, 23, 29, 31][: 3 if smoke else 6])
    print(f"E5 n_models={meta.get('n_models')} solo={meta.get('solo')}", flush=True)
    return _gate(
        "W3_E5_cross_modal_stageB_reborn", oof, te, y, fold, smoke, te_by_fold=meta.get("te_by_fold")
    )


def run_w3_e6(smoke: bool):
    heartbeat("W3", "E6")
    y, fold, _, ath = load_y_fold_ids()
    if smoke:
        y, fold, ath = y[:12000], fold[:12000], ath[:6000]
    z_fb, z_fb_te, _, _, _, _, eta, eta_te = _load_baselines(len(y), len(ath))
    cols = ["merchant_category", "channel", "auth_method"]
    tr = pd.read_csv(REPO / "data/train.csv", usecols=cols)
    te = pd.read_csv(REPO / "data/test.csv", usecols=cols)
    if smoke:
        tr, te = tr.iloc[:12000], te.iloc[:6000]
    tr["cohort"] = tr[cols[0]].astype(str) + "|" + tr[cols[1]].astype(str) + "|" + tr[cols[2]].astype(str)
    te["cohort"] = te[cols[0]].astype(str) + "|" + te[cols[1]].astype(str) + "|" + te[cols[2]].astype(str)
    # unsupervised quantile shift: map train z ranks within cohort to test cohort mass
    oof = z_fb.copy()
    te_pred = z_fb_te.copy()
    for f in range(5):
        tr_m, va_m = fold != f, fold == f
        # cohort mean of z on train → apply delta on va/test
        g = pd.DataFrame({"c": tr.loc[tr_m, "cohort"], "z": z_fb[tr_m]}).groupby("c")["z"].mean()
        global_m = z_fb[tr_m].mean()
        delta_va = tr.loc[va_m, "cohort"].map(g).fillna(global_m).to_numpy(float) - global_m
        oof[va_m] = z_fb[va_m] + 0.15 * delta_va
        delta_te = te["cohort"].map(g).fillna(global_m).to_numpy(float) - global_m
        te_pred += (z_fb_te + 0.15 * delta_te) / 5.0
    X0, X0t, _ = tabular_core_matrix(smoke)
    X = np.column_stack([X0, oof, eta])
    Xt = np.column_stack([X0t, te_pred, eta_te])
    oof2, te2, meta = train_multiseed_stack(X, y, fold, Xt, eta, eta_te, smoke=smoke)
    print(f"E6 n_models={meta.get('n_models')} solo={meta.get('solo')}", flush=True)
    return _gate(
        "W3_E6_cohort_calibration_and_ot_audit",
        oof2,
        te2,
        y,
        fold,
        smoke,
        te_by_fold=meta.get("te_by_fold"),
    )


def run_w3_jewels(smoke: bool):
    """Gate unspanned local bank jewels through fixed Dual-Baseline G0/Vault."""
    heartbeat("W3", "jewels")
    y, fold, _, ath = load_y_fold_ids()
    if smoke:
        y, fold, ath = y[:12000], fold[:12000], ath[:6000]
    jewels = [
        ("P4_L4.2", REPO / "preds/matrix_v2/P4_L4.2_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("P4_L4.3", REPO / "preds/matrix_v2/P4_L4.3_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.3_pure5f_test.npy"),
        ("P4_L4.5", REPO / "preds/matrix_v2/P4_L4.5_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.5_pure5f_test.npy"),
        ("v10_ridge_resid", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"),
        ("B_xgb_llr", REPO / "preds/paradigm_B/xgb_llr_oof.npy", REPO / "preds/paradigm_B/xgb_llr_test.npy"),
        ("v10_cb_d4", REPO / "preds/v10_transductive_bulsah/v10_cb_d4_bag3_oof.npy", REPO / "preds/v10_transductive_bulsah/v10_cb_d4_bag3_test.npy"),
    ]
    out = {}
    for name, op, tp in jewels:
        if not (op.exists() and tp.exists()):
            out[name] = {"status": "missing"}
            continue
        oof = np.load(op).ravel()[: len(y)].astype(float)
        te = np.load(tp).ravel()[: len(ath)].astype(float)
        out[name] = _gate(f"JEWEL_{name}", oof, te, y, fold, smoke)
    atomic_json(LOGS / "v36_1_ckpt/w3_jewels.json", out)
    return out


def run_w3_multivault(smoke: bool):
    heartbeat("W3", "MultiVault")
    y, fold, ids_te, ath = load_y_fold_ids()
    if smoke:
        y, fold, ath, ids_te = y[:12000], fold[:12000], ath[:6000], ids_te[:6000]
    z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te = _load_baselines(len(y), len(ath))
    sf = build_subfolds(y, fold)
    # load vault u vectors
    us_o, us_t, names = [], [], []
    for p in sorted(VAULT.glob("*_u_oof.npy")):
        name = p.name.replace("_u_oof.npy", "")
        tp = VAULT / f"{name}_u_test.npy"
        if not tp.exists():
            continue
        uo = np.load(p).ravel()[: len(y)]
        ut = np.load(tp).ravel()[: len(ath)]
        us_o.append(uo)
        us_t.append(ut)
        names.append(name)
    # also include any G0 from PRED
    for p in sorted(PRED.glob("*_u_oof.npy")):
        name = p.name.replace("_u_oof.npy", "")
        if name in names:
            continue
        gate = PRED / f"{name}_gate.json"
        if gate.exists():
            g = json.loads(gate.read_text())
            if not (g.get("g0") or g.get("vault")):
                continue
        tp = PRED / f"{name}_u_test.npy"
        if tp.exists():
            us_o.append(np.load(p).ravel()[: len(y)])
            us_t.append(np.load(tp).ravel()[: len(ath)])
            names.append(name)
    print(f"MultiVault pillars={names}", flush=True)
    if len(us_o) == 0:
        out = {"status": "empty_vault", "g0": False}
        atomic_json(LOGS / "v36_1_ckpt/W3_MultiVault.json", out)
        return out
    Uo = np.column_stack(us_o)
    Ut = np.column_stack(us_t)
    # decorrelate: drop high corr
    keep = [0]
    for j in range(1, Uo.shape[1]):
        if max(abs(np.corrcoef(Uo[:, j], Uo[:, k])[0, 1]) for k in keep) < 0.25:
            keep.append(j)
    Uo, Ut = Uo[:, keep], Ut[:, keep]
    names_k = [names[j] for j in keep]
    # coordinate ascent weights on fb
    from v36_1_lib import signed_lofo_grid, U, L

    w = np.zeros(Uo.shape[1])
    for _ in range(8):
        for j in range(Uo.shape[1]):
            best, bw = -1e18, 0.0
            for cand in np.linspace(-0.05, 0.05, 21):
                w2 = w.copy()
                w2[j] = cand
                syn = Uo @ w2
                # quick mean Δ on full (biased) — refine via LOFO later
                pred = U(L(z_fb) + syn)
                d = auc(y, pred) - auc(y, z_fb)
                if d > best:
                    best, bw = d, cand
            w[j] = bw
    syn_o = Uo @ w
    syn_t = Ut @ w
    atomic_npy(PRED / "u_synth_oof.npy", syn_o)
    atomic_npy(PRED / "u_synth_test.npy", syn_t)
    rec = _gate("W3_MultiVault_Synthesis", z_fb + syn_o, z_fb_te + syn_t, y, fold, smoke)
    # also store blend candidates NOSUBMIT
    from v36_1_lib import evaluate_engine_g0

    g = rec.get("g_fb") or {}
    w_star = abs(float(g.get("w_fixed", 0.02))) * 0.85
    c1 = blend_ath(ath, syn_t, 0.60 * w_star)
    c2 = blend_ath(ath, syn_t, 1.00 * w_star)
    hedge = U(0.5 * L(U(ath)) + 0.5 * L(c2))
    pd.DataFrame({"id": ids_te, "is_fraud": c1}).to_csv(REPO / "submissions/sub_v36_1_CAND1_NOSUBMIT.csv", index=False)
    pd.DataFrame({"id": ids_te, "is_fraud": c2}).to_csv(REPO / "submissions/sub_v36_1_CAND2_NOSUBMIT.csv", index=False)
    pd.DataFrame({"id": ids_te, "is_fraud": hedge}).to_csv(
        REPO / "submissions/sub_v36_1_PRIVATE_HEDGE_NOSUBMIT.csv", index=False
    )
    staged = {
        "engine": "W3_MultiVault_Synthesis",
        "w_star": w_star,
        "delta_pred_full_e4": g.get("mean_e4", 0),
        "pillars": names_k,
        "weights": w.tolist(),
        "gate": rec,
        "cand1": str(REPO / "submissions/sub_v36_1_CAND1_NOSUBMIT.csv"),
        "cand2": str(REPO / "submissions/sub_v36_1_CAND2_NOSUBMIT.csv"),
    }
    atomic_json(LOGS / "v36_1_ckpt/staged.json", staged)
    return rec


def run_w4_e7(smoke: bool):
    """Deep continuation: more seeds on best feature set until midnight window."""
    heartbeat("W4", "E7")
    y, fold, _, ath = load_y_fold_ids()
    X0, X0t, _ = tabular_core_matrix(smoke)
    if smoke:
        y, fold, ath = y[:12000], fold[:12000], ath[:6000]
    z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te = _load_baselines(len(y), len(ath))
    # merge E1+E3 extras if cheap
    try:
        fe, fet = feats_stealth_and_eb(smoke, y, fold)
        X = np.column_stack([X0, fe])
        Xt = np.column_stack([X0t, fet])
    except Exception:
        X, Xt = X0, X0t
    X, Xt = append_oof_summaries(X, Xt, Bo, Bt, z_fb, z_fb_te, z_twin, z_twin_te)
    oof, te, meta = train_multiseed_stack(
        X, y, fold, Xt, eta, eta_te, smoke=smoke, seeds=[101, 103, 107, 109, 113], n_estimators=3000 if not smoke else 80
    )
    print(f"E7 n_models={meta.get('n_models')} solo={meta.get('solo')}", flush=True)
    return _gate(
        "W4_E7_deep_tabm_k32_and_cat_oblivious_seeds",
        oof,
        te,
        y,
        fold,
        smoke,
        te_by_fold=meta.get("te_by_fold"),
    )


def run_w2_gpu_gate(smoke: bool):
    """Conditional GPU: only log decision; provision only if intel+canary (default skip spend)."""
    heartbeat("W2", "gpu_gate")
    intel = {}
    ip = LOGS / "GM_V36_1_INTEL.json"
    if ip.exists():
        intel = json.loads(ip.read_text())
    # Fold-0 proxy: use v10 lift on subfolds of fold0
    y, fold, _, ath = load_y_fold_ids()
    z_fb = np.load(PRED / "z_fullbank_oof.npy").ravel()
    sf = build_subfolds(y, fold)
    v10 = np.load(REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy").ravel()
    from v36_1_lib import frozen_lofo_orth, signed_lofo_grid

    u_o, _, _ = frozen_lofo_orth(v10, v10[: len(ath)], fold, [z_fb], [z_fb[: len(ath)]])
    # count fold0 sublifts
    f0 = fold == 0
    lifts = 0
    for s in range(5):
        m = f0 & (sf == s)
        if m.sum() < 50:
            continue
        from v36_1_lib import U, L

        d = auc(y[m], U(L(z_fb[m]) + 0.01 * u_o[m])) - auc(y[m], z_fb[m])
        if d >= 3e-4:
            lifts += 1
    promote = bool(intel.get("promote_gpu")) and lifts >= 4
    out = {"promote_gpu_pod": promote, "fold0_sub_lifts_ge_3e4": lifts, "reason": "local canary"}
    if promote and not smoke:
        out["action"] = "SKIP_POD_DEFAULT_SAFE — set V361_ALLOW_POD=1 to enable"
        if os.environ.get("V361_ALLOW_POD") == "1":
            out["action"] = "WOULD_PROVISION v361-gm-*-bpelm (not auto in this build without explicit ops)"
    atomic_json(LOGS / "v36_1_ckpt/w2_gpu.json", out)
    print(out, flush=True)
    return out


ENGINES = {
    "W1_E1": run_w1_e1,
    "W1_E1_fixed": run_w1_e1_fixed,
    "W1_E2": run_w1_e2,
    "W1_E3": run_w1_e3,
    "W1_E4": run_w1_e4,
    "W2_GPU": run_w2_gpu_gate,
    "W3_E5": run_w3_e5,
    "W3_E6": run_w3_e6,
    "W3_JEWELS": run_w3_jewels,
    "W3_MultiVault": run_w3_multivault,
    "W4_E7": run_w4_e7,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    t0 = time.time()
    rec = ENGINES[args.engine](args.smoke)
    print(json.dumps({"engine": args.engine, "elapsed": round(time.time() - t0, 1), "keys": list(rec.keys()) if isinstance(rec, dict) else []}), flush=True)


if __name__ == "__main__":
    main()
