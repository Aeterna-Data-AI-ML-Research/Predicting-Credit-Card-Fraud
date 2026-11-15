#!/usr/bin/env python3
"""GM V21 — Local estimate: replace Bo12 pillars with stronger bases (not residual orth).

Tracks:
  A) Import stronger LMs (masaya GPT-2, g26 seed2026 full 5-fold test, lm7, Q17, alex, supers)
  B) Digit-key / joint-key TE + GLM + 4-leaf LGB offset (Fuad/mathsonchain style) — train CPU
  C) Wider public OOF bank + nested-CV forward hill-climb
  D) orig20k exact collisions + knn fraud-rate fingerprints — train CPU LGB

No Kaggle submit. Writes nested-CV vs Slot8-twin delta + LB estimate.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
TMP15, TMP16 = Path("/tmp/s6e11_v15"), Path("/tmp/s6e11_v16")
OUT = TMP16 / "out"
LOG = REPO / "logs/overnight"
DATA = REPO / "data"
SUB = REPO / "submissions"
PRED = REPO / "preds"
ART = REPO / "artifacts"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
ALEX2 = REPO / "external/nov13_recon/alexchilton_oof_predictions"
G26_OOF = TMP16 / "ds/g26_seed2026/gpt2_s2026_oof_5fold.npy"
G26_T = TMP16 / "ds/g26_seed2026/gpt2_s2026_test_5fold.npy"
G26_T0 = TMP16 / "ds/g26_seed2026/gpt2_s2026_test_f0.npy"
MASAYA = TMP16 / "ds/masaya_gpt2"
MATH = TMP16 / "ds/mathsonchain17"
HARV = PRED / "fast_gpu_harvest"
ORIG = REPO / "original/credit_card_fraud_2026.csv"
SLOT8 = SUB / "sub_v19_4_LIVE_slot8_A_cat_wo0.016_wc0.012_wu0.012.csv"
SLOT9 = SUB / "sub_v20_LIVE_slot9_overnight_cpu_promoted.csv"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_3_LIVE_slot3_FamilyE_w0.01045.csv"

N_JOBS = max(1, (os.cpu_count() or 4) - 1)
T0 = time.time()
LINES: list[str] = []


def log(*a):
    s = f"[{time.time()-T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def L(p):
    p = np.clip(np.asarray(p, float).ravel(), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def S(z):
    z = np.asarray(z, float).ravel()
    return 1 / (1 + np.exp(-z))


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def to_prob(a):
    a = np.asarray(a, float).ravel()
    if a.min() < -0.01 or a.max() > 1.01:
        return S(a)
    return np.clip(a, 1e-6, 1 - 1e-6)


def auc(y, p):
    return float(roc_auc_score(y, p))


def put(bank, name, o, t, y, n_tr, n_te, min_auc=0.93):
    o = np.asarray(o, float).ravel()
    t = None if t is None else np.asarray(t, float).ravel()
    if len(o) != n_tr:
        return False
    if t is not None and len(t) != n_te:
        # allow mean over fold axis
        if t.ndim == 2 and t.shape[1] == n_te:
            t = t.mean(0)
        elif t.ndim == 2 and t.shape[0] == n_te:
            t = t.mean(1)
        else:
            t = None
    if t is None:
        return False
    o, t = to_prob(o), to_prob(t)
    try:
        a = auc(y, o)
    except Exception:
        return False
    if a < min_auc or not np.isfinite(a):
        return False
    bank[name] = (o, t, a)
    return True


# ───────────────────────── digit / joint keys ─────────────────────────
CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
NUMS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
]
FLOATS = [
    "amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
    "velocity_score", "merchant_risk_score",
]


def build_key_matrix(full: pd.DataFrame):
    cols = {}
    for c in NUMS + FLAGS + CATS:
        if c in CATS:
            cols[f"v_{c}"] = full[c].astype(str).values
        else:
            cols[f"v_{c}"] = full[c].values
    for c in FLOATS:
        x = full[c].values.astype(float)
        cols[f"cents_{c}"] = (np.round(x * 100) % 100).astype(np.int64)
        cols[f"int_{c}"] = np.floor(x).astype(np.int64)
        cols[f"ten_{c}"] = np.floor(x / 10).astype(np.int64)
        cols[f"dec_{c}"] = np.floor(x * 10 + 1e-6).astype(np.int64)
        i = pd.Series(np.floor(x).astype(np.int64)).astype(str)
        cols[f"l1_{c}"] = (i.str.len() * 10 + i.str[0].astype(int)).values
        cols[f"l2_{c}"] = (i.str.len() * 100 + i.str[:2].astype(int)).values
    # joint keys (kept set from gboost / megayak)
    def q10(c):
        return pd.qcut(full[c], 10, labels=False, duplicates="drop")

    joints = [
        ("j_category_risk", [full["merchant_category"].astype(str), full["merchant_risk_score"]]),
        ("j_foreign_ip", [full["is_foreign_transaction"], full["ip_country_mismatch"]]),
        ("j_category_amount", [full["merchant_category"].astype(str), q10("amount_usd")]),
        ("j_txn_velocity", [full["txn_count_last_24h"], q10("velocity_score")]),
        ("j_category_age", [full["merchant_category"].astype(str), full["customer_age"]]),
        ("j_amount_risk", [q10("amount_usd"), q10("merchant_risk_score")]),
        ("j_hours_velocity", [q10("hours_since_last_txn"), q10("velocity_score")]),
        ("j_age_risk", [full["customer_age"], q10("merchant_risk_score")]),
        ("j_newmerchant_weekday", [full["is_new_merchant"], full["day_of_week"]]),
        ("j_scam_velocity", [full["is_ai_generated_scam_attempt"], q10("velocity_score")]),
        ("j_age_cardage", [full["customer_age"], full["card_age_months"]]),
        ("j_hour_weekday", [full["time_of_day_hour"], full["day_of_week"]]),
        ("j_txn_age", [full["txn_count_last_24h"], full["customer_age"]]),
    ]
    for name, parts in joints:
        k = np.zeros(len(full), np.int64)
        for p in parts:
            u = pd.factorize(p)[0]
            k = k * (u.max() + 1) + u
        cols[name] = pd.factorize(k)[0]
    names = list(cols)
    codes = [pd.factorize(pd.Series(cols[n]))[0].astype(np.int32) for n in names]
    K = np.column_stack(codes)
    ncat = np.array([c.max() + 1 for c in codes], dtype=np.int32)
    return names, K, ncat


def te_rate(k, t, ncat, prior, smooth):
    s = np.bincount(k, weights=t, minlength=ncat)
    n = np.bincount(k, minlength=ncat)
    return (s + prior * smooth) / (n + smooth)


def te_fold(K, y, a, b, n_tr, ncat, smooth=20.0, inner=3, seed=0):
    ya = y[a].astype(np.float64)
    Ka, Kb, Kt = K[a], K[b], K[n_tr:]
    Ea = np.zeros(Ka.shape, np.float32)
    Eb = np.zeros(Kb.shape, np.float32)
    Et = np.zeros(Kt.shape, np.float32)
    inner_f = list(StratifiedKFold(inner, shuffle=True, random_state=seed).split(np.zeros(len(a)), ya))
    for j in range(K.shape[1]):
        for i, o in inner_f:
            Ea[o, j] = te_rate(Ka[i, j], ya[i], ncat[j], ya[i].mean(), smooth)[Ka[o, j]]
        r = te_rate(Ka[:, j], ya, ncat[j], ya.mean(), smooth)
        Eb[:, j] = r[Kb[:, j]]
        Et[:, j] = r[Kt[:, j]]
    return Ea, Eb, Et


def train_te_glm_lgb(tr, te, y, folds, n_jobs=N_JOBS):
    log("=== B) Digit-key TE + GLM + 4-leaf LGB offset ===")
    full = pd.concat([tr, te], ignore_index=True)
    names, K, ncat = build_key_matrix(full)
    n_tr, n_te = len(tr), len(te)
    log(f"keys={len(names)} K={K.shape}")

    oof_glm = np.zeros(n_tr)
    oof_lgb = np.zeros(n_tr)
    oof_boost = np.zeros(n_tr)
    te_glm, te_lgb, te_boost = [], [], []

    for fi, (a, b) in enumerate(folds):
        Ea, Eb, Et = te_fold(K, y, a, b, n_tr, ncat, smooth=20.0, inner=3, seed=42 + fi)
        # GLM on TE
        sc = StandardScaler(with_mean=True, with_std=True)
        Xa = sc.fit_transform(Ea)
        Xb = sc.transform(Eb)
        Xt = sc.transform(Et)
        glm = LogisticRegression(
            C=0.3, solver="lbfgs", max_iter=400, n_jobs=n_jobs,
        )
        glm.fit(Xa, y[a])
        pb = glm.predict_proba(Xb)[:, 1]
        pt = glm.predict_proba(Xt)[:, 1]
        oof_glm[b] = pb
        te_glm.append(pt)

        # shallow LGB on TE
        dtr = lgb.Dataset(Ea, label=y[a])
        dva = lgb.Dataset(Eb, label=y[b], reference=dtr)
        params = dict(
            objective="binary", metric="auc", learning_rate=0.05, num_leaves=4,
            min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
            reg_lambda=1.0, max_bin=255, verbosity=-1, n_jobs=n_jobs,
        )
        booster = lgb.train(
            params, dtr, num_boost_round=4000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        lb = booster.predict(Eb, num_iteration=booster.best_iteration)
        lt = booster.predict(Et, num_iteration=booster.best_iteration)
        oof_lgb[b] = lb
        te_lgb.append(lt)

        # GLM-offset LGB (boost_j style): start from glm logit
        off_a = L(glm.predict_proba(Xa)[:, 1])
        off_b = L(pb)
        off_t = L(pt)
        dtr2 = lgb.Dataset(Ea, label=y[a], init_score=off_a)
        dva2 = lgb.Dataset(Eb, label=y[b], init_score=off_b, reference=dtr2)
        booster2 = lgb.train(
            params, dtr2, num_boost_round=4000, valid_sets=[dva2],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        # predict = sigmoid(init + raw)
        raw_b = booster2.predict(Eb, num_iteration=booster2.best_iteration, raw_score=True)
        raw_t = booster2.predict(Et, num_iteration=booster2.best_iteration, raw_score=True)
        oof_boost[b] = S(off_b + raw_b)
        te_boost.append(S(off_t + raw_t))
        log(
            f"  fold{fi} glm={auc(y[b], pb):.5f} lgb={auc(y[b], lb):.5f} "
            f"boost={auc(y[b], oof_boost[b]):.5f} trees={booster.best_iteration}/{booster2.best_iteration}"
        )

    out = {
        "v21_glm_te": (oof_glm, np.mean(te_glm, 0)),
        "v21_lgb_te": (oof_lgb, np.mean(te_lgb, 0)),
        "v21_boost_j": (oof_boost, np.mean(te_boost, 0)),
    }
    for k, (o, t) in out.items():
        log(f"  {k} OOF AUC={auc(y, o):.6f}")
    return out


# ───────────────────────── orig20k fingerprints ─────────────────────────
def train_orig20k_fingerprints(tr, te, y, folds, orig_path, n_jobs=N_JOBS):
    log("=== D) orig20k collisions + knn fraud fingerprints ===")
    orig = pd.read_csv(orig_path)
    # align columns
    use_cols = [c for c in NUMS + FLAGS + CATS if c in tr.columns and c in orig.columns]
    log(f"orig rows={len(orig)} shared_cols={len(use_cols)}")

    def fingerprint_frame(df):
        # exact amount + coarse mantissa signature
        amt = np.round(df["amount_usd"].astype(float).values, 2)
        risk = np.round(df["merchant_risk_score"].astype(float).values, 1)
        vel = np.round(df["velocity_score"].astype(float).values, 1)
        bal = np.round(df["account_balance_usd"].astype(float).values, 0)
        cat = df["merchant_category"].astype(str).values
        return amt, risk, vel, bal, cat

    o_amt, o_risk, o_vel, o_bal, o_cat = fingerprint_frame(orig)
    o_y = orig["is_fraud"].astype(int).values if "is_fraud" in orig.columns else np.zeros(len(orig), int)

    # collision maps
    from collections import defaultdict

    def build_maps():
        m_amt = defaultdict(list)
        m_pair = defaultdict(list)
        m_trip = defaultdict(list)
        for i in range(len(orig)):
            m_amt[o_amt[i]].append(i)
            m_pair[(o_amt[i], o_cat[i])].append(i)
            m_trip[(o_amt[i], o_cat[i], o_risk[i])].append(i)
        return m_amt, m_pair, m_trip

    m_amt, m_pair, m_trip = build_maps()

    def collision_feats(df):
        amt, risk, vel, bal, cat = fingerprint_frame(df)
        n = len(df)
        hit_amt = np.zeros(n)
        hit_pair = np.zeros(n)
        hit_trip = np.zeros(n)
        fr_amt = np.full(n, o_y.mean())
        fr_pair = np.full(n, o_y.mean())
        fr_trip = np.full(n, o_y.mean())
        for i in range(n):
            ia = m_amt.get(amt[i], [])
            ip = m_pair.get((amt[i], cat[i]), [])
            it = m_trip.get((amt[i], cat[i], risk[i]), [])
            hit_amt[i] = len(ia)
            hit_pair[i] = len(ip)
            hit_trip[i] = len(it)
            if ia:
                fr_amt[i] = o_y[ia].mean()
            if ip:
                fr_pair[i] = o_y[ip].mean()
            if it:
                fr_trip[i] = o_y[it].mean()
        return np.column_stack([hit_amt, hit_pair, hit_trip, fr_amt, fr_pair, fr_trip]).astype(np.float32)

    # knn in standardized numeric space vs orig
    num_cols = [c for c in NUMS if c in orig.columns]
    Xo = orig[num_cols].astype(float).values
    Xtr = tr[num_cols].astype(float).values
    Xte = te[num_cols].astype(float).values
    sc = StandardScaler().fit(Xo)
    Xo_s, Xtr_s, Xte_s = sc.transform(Xo), sc.transform(Xtr), sc.transform(Xte)

    def knn_feats(Xq, k=32):
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=n_jobs)
        nn.fit(Xo_s)
        dist, idx = nn.kneighbors(Xq)
        fr = o_y[idx].mean(1)
        dmean = dist.mean(1)
        dmin = dist[:, 0]
        # fraud-only knn
        pos = np.where(o_y == 1)[0]
        nn_p = NearestNeighbors(n_neighbors=min(16, len(pos)), algorithm="auto", n_jobs=n_jobs)
        nn_p.fit(Xo_s[pos])
        dp, _ = nn_p.kneighbors(Xq)
        return np.column_stack([fr, dmean, dmin, dp.mean(1), dp[:, 0]]).astype(np.float32)

    log("  computing collision feats…")
    Ctr = collision_feats(tr)
    Cte = collision_feats(te)
    log("  computing knn feats (k=32)…")
    Ktr = knn_feats(Xtr_s, 32)
    Kte = knn_feats(Xte_s, 32)

    # also mantissa subspace
    def mant4(df):
        return np.column_stack([
            (np.round(df[c].astype(float).values * 100) % 100) for c in
            ["amount_usd", "account_balance_usd", "velocity_score", "merchant_risk_score"]
        ]).astype(np.float32)

    Mo, Mtr, Mte = mant4(orig), mant4(tr), mant4(te)
    sc2 = StandardScaler().fit(Mo)
    Mo_s, Mtr_s, Mte_s = sc2.transform(Mo), sc2.transform(Mtr), sc2.transform(Mte)

    def knn_mant(Xq, k=24):
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=n_jobs)
        nn.fit(Mo_s)
        dist, idx = nn.kneighbors(Xq)
        return np.column_stack([o_y[idx].mean(1), dist.mean(1), dist[:, 0]]).astype(np.float32)

    log("  mantissa knn…")
    MKtr, MKte = knn_mant(Mtr_s), knn_mant(Mte_s)

    Ftr = np.hstack([Ctr, Ktr, MKtr, tr[num_cols].astype(float).values.astype(np.float32)])
    Fte = np.hstack([Cte, Kte, MKte, te[num_cols].astype(float).values.astype(np.float32)])

    oof = np.zeros(len(tr))
    te_folds = []
    for fi, (a, b) in enumerate(folds):
        dtr = lgb.Dataset(Ftr[a], label=y[a])
        dva = lgb.Dataset(Ftr[b], label=y[b], reference=dtr)
        params = dict(
            objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=1.0, verbosity=-1, n_jobs=n_jobs,
        )
        booster = lgb.train(
            params, dtr, num_boost_round=3000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        oof[b] = booster.predict(Ftr[b], num_iteration=booster.best_iteration)
        te_folds.append(booster.predict(Fte, num_iteration=booster.best_iteration))
        log(f"  fold{fi} orig_lgb={auc(y[b], oof[b]):.5f} trees={booster.best_iteration}")

    # also a pure fingerprint ridge on collision+knn only (no raw nums)
    Gtr = np.hstack([Ctr, Ktr, MKtr])
    Gte = np.hstack([Cte, Kte, MKte])
    oof_r = np.zeros(len(tr))
    te_r = []
    for a, b in folds:
        scg = StandardScaler().fit(Gtr[a])
        rr = Ridge(alpha=10.0).fit(scg.transform(Gtr[a]), y[a])
        oof_r[b] = rr.predict(scg.transform(Gtr[b]))
        te_r.append(rr.predict(scg.transform(Gte)))
    log(f"  v21_orig_lgb OOF={auc(y, oof):.6f}")
    log(f"  v21_orig_ridge OOF={auc(y, oof_r):.6f}")
    return {
        "v21_orig_lgb": (oof, np.mean(te_folds, 0)),
        "v21_orig_ridge": (oof_r, np.mean(te_r, 0)),
    }


# ───────────────────────── bank assembly ─────────────────────────
def load_bank(y, n_tr, n_te):
    log("=== A/C) Assemble wide OOF bank ===")
    bank = {}

    def try_load(name, op, tp):
        if Path(op).exists() and Path(tp).exists():
            put(bank, name, np.load(op), np.load(tp), y, n_tr, n_te)

    # Q17
    if Q17.exists():
        Z = np.load(Q17, allow_pickle=True)
        names = [str(s) for s in Z["stream_names"]]
        for n in names:
            put(bank, f"q17:{n}", Z[f"oof_{n}"], Z[f"test_{n}"], y, n_tr, n_te)
        log(f"  Q17 streams loaded: {len(names)}")

    # LMs
    try_load("masaya_gpt2", MASAYA / "oof.npy", MASAYA / "test.npy")
    g26t = G26_T if G26_T.exists() else G26_T0
    try_load("gpt2_s2026", G26_OOF, g26t)
    try_load("lm7_5fold", TMP15 / "banks/lm7_oof_5fold.npy", TMP15 / "banks/lm7_test_5fold.npy")
    try_load("d1_v4", TMP15 / "banks/d1_v4_oof.npy", TMP15 / "banks/d1_v4_test.npy")

    # supers / champs
    for stem in [
        "P_super_v4", "P_super_v5_full", "P_super_v5_pure", "P_super_v6_grandmaster",
        "P_super_v6_clean", "P_champ", "P_super_champ", "P_restack_pub",
    ]:
        try_load(f"pred:{stem}", PRED / f"{stem}_oof.npy", PRED / f"{stem}_test.npy")

    # artifacts digit/glm
    for stem in [
        "lgb_dig_l7", "lgb_dig_s100_bag5", "lgb_diglead_bag", "lgb_rawdig_l4",
        "glm_ohk_c01", "glm_ohk2_bag", "glm_dl_s50_bag", "ebm_dig", "ebm_b4",
        "realmlp_dig", "realmlp_te", "realmlp_v1",
    ]:
        op = ART / f"{stem}_oof.npy"
        tp = ART / f"{stem}_test.npy"
        if not tp.exists():
            tp = ART / f"{stem}_full.npy"
        try_load(f"art:{stem}", op, tp)

    # mathsonchain 17
    if MATH.exists():
        for op in sorted(MATH.glob("oof_*.npy")):
            stem = op.name[4:-4]
            try_load(f"math:{stem}", op, MATH / f"test_{stem}.npy")

    # harvest
    if HARV.exists():
        pairs = [
            ("harv:knn24", "knn24_oof.npy", "knn24_test.npy"),
            ("harv:orig20k_knn", "oof_orig20k_knn.npy", "test_orig20k_knn.npy"),
            ("harv:tabm", "oof_tabm_5f.npy", "test_tabm_5f.npy"),
            ("harv:xgb", "oof_xgb_fast_harvest.npy", "test_xgb_fast_harvest.npy"),
            ("harv:xgb_clean", "oof_xgb_clean.npy", "test_xgb_clean.npy"),
        ]
        for nm, o, t in pairs:
            try_load(nm, HARV / o, HARV / t)

    # alex
    for alex in (ALEX, ALEX2):
        if not alex.exists():
            continue
        for op in sorted(alex.glob("oof_*.npy"))[:40]:
            tp = alex / ("test_" + op.name[4:])
            try_load(f"alex:{op.stem[4:12]}", op, tp)

    # external_bank masaya
    try_load("eb_masaya", REPO / "external_bank/masaya_gpt2/oof.npy", REPO / "external_bank/masaya_gpt2/test.npy")

    log(f"  bank size={len(bank)}")
    top = sorted(bank.items(), key=lambda kv: -kv[1][2])[:15]
    for n, (_, _, a) in top:
        log(f"    {a:.6f}  {n}")
    return bank


def rebuild_slot8_twin(y, n_tr, n_te, folds, bank):
    """Approximate Slot8 OOF twin from Bo9 pillars + known Slot3/8 test."""
    log("=== Rebuild Slot8 twin OOF (baseline) ===")
    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, dE = cache["z_ath"], cache["d1"], cache["dE"]
    d1t = cache["d1t"] if "d1t" in cache.files else bank.get("d1_v4", (None, None, 0))[1]
    ids = pd.read_csv(LIVE)["id"].to_numpy()
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot8 = pd.read_csv(SLOT8).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot9 = pd.read_csv(SLOT9).set_index("id").loc[ids, "is_fraud"].to_numpy(float) if SLOT9.exists() else slot8

    def need(name):
        if name not in bank:
            raise KeyError(name)
        return bank[name][0], bank[name][1]

    # Bo9-ish
    cols_o, cols_t = [], []
    for key in [
        "pred:P_super_v4", "gpt2_s2026", "q17:megayak_5f_lm", "lm7_5fold",
        "q17:daoviet_gpdvae", "q17:gb_bag5", "q17:yekenot_realmlp",
    ]:
        # fuzzy match
        hit = None
        if key in bank:
            hit = key
        else:
            for k in bank:
                if key.split(":")[-1] in k:
                    hit = k
                    break
        if hit is None:
            log(f"  missing pillar {key}")
            continue
        o, t, _ = bank[hit]
        cols_o.append(unitize(L(o)))
        cols_t.append(unitize(L(t)))
    cols_o.append(unitize(d1))
    cols_t.append(unitize(d1t))
    cols_o.append(unitize(dE))
    # dE test: use zeros placeholder then ridge will absorb — better: use cache if present
    # reconstruct dE_t roughly from overnight if available
    dEt = unitize(L(slot3) - L(pd.read_csv(LIVE)["is_fraud"].to_numpy(float)))  # weak proxy
    if "Family_E" in str(bank.keys()):
        pass
    cols_t.append(dEt)
    Bo = np.column_stack(cols_o)
    Bt = np.column_stack(cols_t)
    # fit twin to Slot8 test logits
    ridge = Ridge(alpha=1.0).fit(Bt, unitize(L(slot8)))
    twin_o = unitize(Bo @ ridge.coef_ + ridge.intercept_)
    twin_oof = S(twin_o)  # not calibrated — use rank-gauss style via logistic on OOF folds
    # better: LOFO calibrate twin to y
    oof_cal = np.zeros(n_tr)
    for a, b in folds:
        lr = LogisticRegression(C=1.0, max_iter=200).fit(twin_o[a].reshape(-1, 1), y[a])
        oof_cal[b] = lr.predict_proba(twin_o[b].reshape(-1, 1))[:, 1]
    a_twin = auc(y, oof_cal)
    log(f"  Slot8-twin OOF AUC={a_twin:.6f}  (LB Slot8=0.95395 Slot9=0.95399)")
    return oof_cal, slot8, slot9, a_twin


# ───────────────────────── nested hill-climb ─────────────────────────
def nested_hillclimb(bank, y, folds, max_members=40, seed_names=None):
    log("=== C) Nested CV forward hill-climb ===")
    names = sorted(bank.keys(), key=lambda n: -bank[n][2])
    # drop near-duplicates: keep if corr < 0.999 with kept
    kept = []
    for n in names:
        o = unitize(L(bank[n][0]))
        ok = True
        for k in kept:
            if abs(np.corrcoef(o, unitize(L(bank[k][0])))[0, 1]) > 0.9995:
                ok = False
                break
        if ok:
            kept.append(n)
    log(f"  unique members={len(kept)} (from {len(names)})")

    # force seed strong LMs / new bases first
    seed_names = seed_names or []
    order = []
    for s in seed_names:
        if s in kept and s not in order:
            order.append(s)
    for n in kept:
        if n not in order:
            order.append(n)

    # nested: for each outer fold, hill-climb on inner 4, score on held-out
    oof_blend = np.zeros(len(y))
    chosen_hist = []
    for fi, (a, b) in enumerate(folds):
        # start with equal blend of top-5 on a
        top5 = sorted(order, key=lambda n: -auc(y[a], bank[n][0][a]))[:5]
        selected = list(top5)
        # current score on a via rank mean
        def blend(idxs, sel):
            R = np.column_stack([rankdata(bank[n][0][idxs]) for n in sel])
            return R.mean(1)

        best = auc(y[a], blend(a, selected))
        improved = True
        while improved and len(selected) < max_members:
            improved = False
            cand_best, cand_n = best, None
            for n in order:
                if n in selected:
                    continue
                sc = auc(y[a], blend(a, selected + [n]))
                if sc > cand_best + 1e-7:
                    cand_best, cand_n = sc, n
            if cand_n is not None:
                selected.append(cand_n)
                best = cand_best
                improved = True
            else:
                # try drop
                if len(selected) <= 3:
                    break
                drop_best, drop_n = best, None
                for n in selected:
                    sc = auc(y[a], blend(a, [x for x in selected if x != n]))
                    if sc > drop_best + 1e-7:
                        drop_best, drop_n = sc, n
                if drop_n is not None:
                    selected = [x for x in selected if x != drop_n]
                    best = drop_best
                    improved = True
                else:
                    break
        oof_blend[b] = blend(b, selected)
        chosen_hist.append(selected)
        log(f"  outer{fi} members={len(selected)} trainAUC={best:.6f} hold={auc(y[b], oof_blend[b]):.6f}")

    nested = auc(y, oof_blend)
    # consensus members
    from collections import Counter
    cnt = Counter(n for sel in chosen_hist for n in sel)
    top_mem = cnt.most_common(25)
    log(f"  NESTED AUC={nested:.6f}")
    log("  frequent members:")
    for n, c in top_mem[:15]:
        log(f"    {c}/5  auc={bank[n][2]:.5f}  {n}")

    # full-data hillclimb for test export estimate
    selected = list(top_mem[:8])  # start
    selected = [n for n, _ in selected if n in bank]
    if not selected:
        selected = order[:5]

    def blend_full(sel, oof=True):
        if oof:
            R = np.column_stack([rankdata(bank[n][0]) for n in sel])
        else:
            R = np.column_stack([rankdata(bank[n][1]) for n in sel])
        return R.mean(1)

    best = auc(y, blend_full(selected, True))
    for _ in range(60):
        gained = False
        for n in order:
            if n in selected or len(selected) >= max_members:
                continue
            sc = auc(y, blend_full(selected + [n], True))
            if sc > best + 1e-7:
                selected.append(n)
                best = sc
                gained = True
                break
        if not gained:
            break
    log(f"  full HC members={len(selected)} OOF={best:.6f}")
    test_pred = blend_full(selected, False)
    return nested, oof_blend, test_pred, selected, best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    log("GM V21 base-pillar replace estimate — local CPU")

    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    y = tr["is_fraud"].astype(int).values
    n_tr, n_te = len(tr), len(te)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n_tr), y))
    log(f"n_tr={n_tr} n_te={n_te} n_jobs={N_JOBS}")

    bank = load_bank(y, n_tr, n_te)

    # B) train TE/GLM/LGB
    te_models = train_te_glm_lgb(tr, te, y, folds)
    for k, (o, t) in te_models.items():
        put(bank, k, o, t, y, n_tr, n_te, min_auc=0.90)

    # D) orig20k
    if ORIG.exists():
        orig_models = train_orig20k_fingerprints(tr, te, y, folds, ORIG)
        for k, (o, t) in orig_models.items():
            put(bank, k, o, t, y, n_tr, n_te, min_auc=0.90)
    else:
        log("ORIG missing — skip D")

    # baseline twin
    try:
        twin_oof, slot8, slot9, a_twin = rebuild_slot8_twin(y, n_tr, n_te, folds, bank)
    except Exception as e:
        log(f"twin rebuild failed: {e}")
        twin_oof, a_twin = None, None
        ids = pd.read_csv(LIVE)["id"].to_numpy()
        slot8 = pd.read_csv(SLOT8).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
        slot9 = pd.read_csv(SLOT9).set_index("id").loc[ids, "is_fraud"].to_numpy(float) if SLOT9.exists() else slot8

    seed = [
        "masaya_gpt2", "gpt2_s2026", "lm7_5fold", "pred:P_super_v6_grandmaster",
        "pred:P_super_v4", "v21_boost_j", "v21_glm_te", "v21_lgb_te",
        "math:boost_j", "math:glm_j", "math:lgb_te", "v21_orig_lgb",
        "harv:orig20k_knn", "harv:tabm",
    ]
    nested, oof_blend, test_pred, selected, full_hc = nested_hillclimb(
        bank, y, folds, max_members=50, seed_names=seed,
    )

    # also NNLS on logits of selected vs y (LOFO)
    oof_nnls = np.zeros(n_tr)
    for a, b in folds:
        Xa = np.column_stack([L(bank[n][0][a]) for n in selected])
        Xb = np.column_stack([L(bank[n][0][b]) for n in selected])
        # nnls on centered
        ya = y[a].astype(float)
        w, _ = nnls(Xa - Xa.mean(0), ya - ya.mean())
        if w.sum() < 1e-12:
            w = np.ones(len(selected)) / len(selected)
        oof_nnls[b] = S((Xb - Xa.mean(0)) @ w + ya.mean())
    a_nnls = auc(y, oof_nnls)
    log(f"  LOFO-NNLS on HC members AUC={a_nnls:.6f}")

    # estimate LB
    # calibrate: Slot9 LB 0.95399; if twin OOF known, bias = LB - twin_OOF
    lb_s9 = 0.95399
    lb_s8 = 0.95395
    if a_twin is not None and a_twin > 0.95:
        bias = lb_s9 - a_twin
        est_nested = nested + bias
        est_nnls = a_nnls + bias
        est_full = full_hc + bias
    else:
        # typical CV≈LB for this comp; use raw
        bias = 0.0
        est_nested, est_nnls, est_full = nested, a_nnls, full_hc

    delta_nested = est_nested - lb_s9
    delta_nnls = est_nnls - lb_s9

    # diversity vs Slot8 test
    rho = 1 - float(np.corrcoef(rankdata(test_pred), rankdata(slot8))[0, 1])
    log(f"  test 1-ρ vs Slot8={rho*1e4:.3f}e-4")

    summary = {
        "nested_auc": nested,
        "full_hc_auc": full_hc,
        "nnls_auc": a_nnls,
        "slot8_twin_oof": a_twin,
        "bias_lb_minus_twin": bias,
        "est_lb_nested": est_nested,
        "est_lb_nnls": est_nnls,
        "est_lb_full_hc": est_full,
        "delta_vs_slot9_nested": delta_nested,
        "delta_vs_slot9_nnls": delta_nnls,
        "bank_size": len(bank),
        "selected": selected,
        "new_base_aucs": {k: float(auc(y, te_models[k][0])) for k in te_models},
    }
    if ORIG.exists():
        summary["new_base_aucs"].update(
            {k: float(auc(y, v[0])) for k, v in {
                "v21_orig_lgb": bank.get("v21_orig_lgb", (np.zeros(n_tr), None, 0))[:1] and None,
            }.items() if False}
        )
    for k in ("v21_orig_lgb", "v21_orig_ridge", "v21_boost_j", "v21_glm_te", "v21_lgb_te",
              "masaya_gpt2", "gpt2_s2026", "math:boost_j", "math:glm_j"):
        if k in bank:
            summary.setdefault("pillar_auc", {})[k] = bank[k][2]

    # save artifacts
    np.savez_compressed(
        OUT / "v21_estimate.npz",
        oof_blend=oof_blend, oof_nnls=oof_nnls, test_pred=test_pred,
        selected=np.array(selected, dtype=object),
    )
    # staged sub (no submit)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    # rank-gauss blend already in [rank space] — map to (0,1) via rank
    sub["is_fraud"] = (rankdata(test_pred) - 0.5) / len(test_pred)
    sub_path = SUB / "sub_v21_NOSUBMIT_wide_hc_estimate.csv"
    sub.to_csv(sub_path, index=False)

    report = [
        "# GM V21 — Base Pillar Replace Estimate (local CPU)",
        "",
        f"**elapsed** {time.time()-T0:.0f}s · bank={len(bank)} · HC members={len(selected)}",
        "",
        "## Score estimate",
        f"- Nested CV AUC: **{nested:.6f}**",
        f"- Full HC OOF AUC: **{full_hc:.6f}**",
        f"- LOFO-NNLS AUC: **{a_nnls:.6f}**",
        f"- Slot8-twin OOF: {a_twin}",
        f"- Bias (Slot9 LB − twin OOF): {bias:.6f}" if a_twin else "- Bias: n/a",
        f"- **Estimated LB (nested): {est_nested:.6f}  (Δ vs Slot9 {delta_nested*1e5:+.2f}e-5)**",
        f"- **Estimated LB (nnls): {est_nnls:.6f}  (Δ vs Slot9 {delta_nnls*1e5:+.2f}e-5)**",
        f"- Slot8 LB=0.95395 · Slot9 ATH=0.95399 · target 0.95784 gap still {(0.95784-est_nested)*1e5:.0f}e-5 if est holds",
        "",
        "## New bases (OOF AUC)",
        "```",
        *[f"{k}: {summary.get('pillar_auc', {}).get(k, 'n/a')}" for k in
          ("v21_boost_j", "v21_glm_te", "v21_lgb_te", "v21_orig_lgb", "v21_orig_ridge",
           "masaya_gpt2", "gpt2_s2026", "math:boost_j", "math:glm_j")],
        "```",
        "",
        "## Selected HC members",
        "```",
        *selected,
        "```",
        "",
        f"Staged (no submit): `{sub_path}`",
        "",
        "## Log",
        "```",
        *LINES,
        "```",
    ]
    rp = LOG / "GM_V21_BASE_PILLAR_REPLACE_ESTIMATE.md"
    rp.write_text("\n".join(report) + "\n")
    OUT.joinpath("v21_estimate.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"REPORT {rp}")
    log(f"EST nested LB≈{est_nested:.6f} (Δ{delta_nested*1e5:+.1f}e-5 vs Slot9)")
    log("DONE")


if __name__ == "__main__":
    main()
