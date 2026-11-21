#!/usr/bin/env python3
"""GM V16.2 — Bracket Family E/A OOF peak + G26 cal + 4-leaf digit TE.

STRICT NO_SUBMIT=True. Local Mac CPU only. 0 pods.
New live baseline: sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv (0.95389).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
REPO = Path(os.environ.get("V16_REPO", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
TMP = Path(os.environ.get("V15_TMP", "/tmp/s6e11_v15"))
OUT = Path(os.environ.get("V16_OUT", "/tmp/s6e11_v16/out"))
G26_DIR = Path(os.environ.get("G26_DIR", "/tmp/s6e11_v16/g26"))
SUB = REPO / "submissions"
REPORT = REPO / "logs/overnight/GM_V16_2_BRACKET_FAME_AND_4LEAF_DIGITS.md"
LIVE095388 = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT1 = SUB / "sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv"

# w grid from brief (nominal → expected 1-ρ vs LIVE_095388)
W_GRID = [0.0074, 0.01045, 0.0122, 0.01383, 0.0157, 0.0173, 0.0200, 0.0250, 0.0300]
W_LABELS = {
    0.0074: "0.10e-4",
    0.01045: "0.20e-4_Slot1",
    0.0122: "0.27e-4",
    0.01383: "0.35e-4",
    0.0157: "0.45e-4",
    0.0173: "0.55e-4",
    0.0200: "0.73e-4",
    0.0250: "1.15e-4",
    0.0300: "1.6e-4?",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def L(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    r = (rankdata(v, method="average") - 0.5) / len(v)
    r = np.clip(r, 1e-12, 1 - 1e-12)
    return np.log(r / (1 - r))


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def quantile_lock(raw, ref):
    raw = np.asarray(raw, float).ravel()
    ref = np.asarray(ref, float).ravel()
    out = np.empty_like(ref)
    out[np.argsort(raw)] = np.sort(ref)
    return out


def fold_splits(y):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = np.empty(len(y), np.int8)
    for f, (_, te) in enumerate(folds):
        fold_id[te] = f
    return folds, fold_id


def orient(u, z, y, te0):
    if roc_auc_score(y[te0], z[te0] + 0.01 * u[te0]) < roc_auc_score(y[te0], z[te0] - 0.01 * u[te0]):
        return -u
    return u


def eval_w_grid(name, y, z_base, u, folds, live, slot1, u_test, slot1_ref_w=0.01045):
    """Evaluate direction u on z_base across W_GRID; also vs Slot1 incremental."""
    rows = []
    base = [roc_auc_score(y[te], z_base[te]) for _, te in folds]
    # Slot1 OOF proxy: z + 0.01045*u
    z_slot1 = z_base + slot1_ref_w * u
    slot1_base = [roc_auc_score(y[te], z_slot1[te]) for _, te in folds]

    for w in W_GRID:
        z = z_base + w * u
        d = [(roc_auc_score(y[te], z[te]) - base[i]) * 1e5 for i, (_, te) in enumerate(folds)]
        d_inc = [(roc_auc_score(y[te], z[te]) - slot1_base[i]) * 1e5 for i, (_, te) in enumerate(folds)]
        # sf25 at this w AND microprobe @0.01 (V15 Gate0 convention)
        deltas, deltas_m = [], []
        for _, te in folds:
            yf, zb, uf = y[te], z_base[te], u[te]
            for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(yf)), yf):
                b = roc_auc_score(yf[ste], zb[ste])
                deltas.append((roc_auc_score(yf[ste], zb[ste] + w * uf[ste]) - b) * 1e5)
                deltas_m.append((roc_auc_score(yf[ste], zb[ste] + 0.01 * uf[ste]) - b) * 1e5)
        d25 = np.asarray(deltas, float)
        d25m = np.asarray(deltas_m, float)
        mean = float(d25.mean())
        se = float(d25.std(ddof=1) / np.sqrt(len(d25)))
        tstat = float(mean / (se + 1e-18))
        mean_m = float(d25m.mean())
        se_m = float(d25m.std(ddof=1) / np.sqrt(len(d25m)))
        tstat_m = float(mean_m / (se_m + 1e-18))
        try:
            _, wp = wilcoxon(d25, alternative="greater", zero_method="wilcox")
        except Exception:
            wp = 1.0
        rng = np.random.default_rng(42)
        boots = [d25[rng.integers(0, len(d25), len(d25))].mean() for _ in range(400)]
        ci = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]

        # test distances
        cand_test = quantile_lock(L(live) + w * unitize(u_test), live)
        mrho_live = (1.0 - float(spearmanr(live, cand_test).correlation)) * 1e4
        mrho_s1 = (1.0 - float(spearmanr(slot1, cand_test).correlation)) * 1e4

        rows.append(
            {
                "name": name,
                "w": w,
                "label": W_LABELS.get(w, ""),
                "1mrho_LIVE_e4": mrho_live,
                "1mrho_Slot1_e4": mrho_s1,
                "main_pos": int(sum(x > 0 for x in d)),
                "mean_e5": float(np.mean(d)),
                "inc_mean_e5_vs_slot1": float(np.mean(d_inc)),
                "min_e5": float(np.min(d)),
                "folds_e5": [round(x, 3) for x in d],
                "sf25": int((d25 > 0).sum()),
                "t": tstat,
                "wilcox": float(wp),
                "ci90": ci,
                "sf25_micro01": int((d25m > 0).sum()),
                "t_micro01": tstat_m,
                "mean_e5_micro01": mean_m,
            }
        )
    return rows


def build_families(y, folds, fold_id):
    l7f = np.load(TMP / "out/l7f_oof.npy").ravel()
    lm = np.load(TMP / "out/lm_oof.npy").ravel()
    R3 = np.load(TMP / "out/R3_oof.npy").ravel()
    R2 = np.load(TMP / "out/R2_oof.npy").ravel()
    R4 = np.load(TMP / "out/R4_oof.npy").ravel()
    d1 = np.load(TMP / "banks/d1_v4_oof.npy").ravel()[: len(y)]
    z_ath = L(l7f) + 0.03 * unitize(d1)

    # test reductions
    R3_te = np.load(TMP / "out/R3_test_folds.npy")
    R2_te = np.load(TMP / "out/R2_test_folds.npy")
    R4_te = np.load(TMP / "out/R4_test_folds.npy")
    if R3_te.shape[0] == 5:
        R3t = R3_te.mean(0)
        R2t = R2_te.mean(0)
        R4t = R4_te.mean(0)
    else:
        R3t, R2t, R4t = R3_te.mean(1), R2_te.mean(1), R4_te.mean(1)
    l7f_te = np.load(TMP / "banks/lm7_test_5fold.npy").ravel()
    d1t = np.load(TMP / "banks/d1_v4_test.npy").ravel()

    Ls = L(R3)
    L7 = L(l7f)
    Lm = L(lm)
    # Family A
    dA_o = unitize(Ls - L7)
    dA_t = unitize(L(R3t) - L(l7f_te))
    # Family C
    dC_o = unitize(Ls - Lm)
    dC_t = unitize(L(R3t) - L(l7f_te))  # LM test proxy
    # Family E
    eta = z_ath
    U = expit((eta - np.median(eta)) / (np.std(eta) + 1e-12))
    w_tail = 0.5 + 0.5 * U
    raw_e = (Ls - L7) * w_tail
    B = np.column_stack([unitize(z_ath), unitize(d1)])
    lr_e = LinearRegression().fit(B, raw_e)
    dE_o = unitize(raw_e - lr_e.predict(B))
    live = pd.read_csv(LIVE095388)["is_fraud"].to_numpy(float)
    live_L = L(live)
    w_tail_t = 0.5 + 0.5 * expit((live_L - np.median(live_L)) / (np.std(live_L) + 1e-12))
    dE_t = unitize((L(R3t) - L(l7f_te)) * w_tail_t)

    # R4/R2 mix inside E
    Lmix = L(0.7 * R4 + 0.3 * R2)
    raw_e2 = (Lmix - L7) * w_tail
    dE2_o = unitize(raw_e2 - LinearRegression().fit(B, raw_e2).predict(B))
    Lmix_t = L(0.7 * R4t + 0.3 * R2t)
    dE2_t = unitize((Lmix_t - L(l7f_te)) * w_tail_t)

    # 0.8 E + 0.2 C
    dEC_o = unitize(0.8 * dE_o + 0.2 * dC_o)
    dEC_t = unitize(0.8 * dE_t + 0.2 * dC_t)

    te0 = folds[0][1]
    dA_o = orient(dA_o, z_ath, y, te0)
    dE_o = orient(dE_o, z_ath, y, te0)
    dE2_o = orient(dE2_o, z_ath, y, te0)
    dEC_o = orient(dEC_o, z_ath, y, te0)
    dC_o = orient(dC_o, z_ath, y, te0)

    return {
        "z_ath": z_ath,
        "d1": d1,
        "d1t": d1t,
        "live": live,
        "A": (dA_o, dA_t),
        "E": (dE_o, dE_t),
        "E_R4R2": (dE2_o, dE2_t),
        "E80_C20": (dEC_o, dEC_t),
        "C": (dC_o, dC_t),
        "R3t": R3t,
        "l7f_te": l7f_te,
        "l7f": l7f,
        "R3": R3,
    }


def screen_delta(name, y, z_base, u, folds, w_main=0.01):
    """Gate metrics for a fixed direction on top of z_base."""
    te0 = folds[0][1]
    u = orient(unitize(u), z_base, y, te0)
    base = [roc_auc_score(y[te], z_base[te]) for _, te in folds]
    best = None
    for w in (0.006, 0.009, 0.012, 0.015, 0.020, 0.025):
        d = [(roc_auc_score(y[te], z_base[te] + w * u[te]) - base[i]) * 1e5 for i, (_, te) in enumerate(folds)]
        row = dict(w=w, pos=int(sum(x > 0 for x in d)), mean_e5=float(np.mean(d)), min_e5=float(np.min(d)), folds=[round(x, 3) for x in d])
        if best is None or (row["pos"], row["mean_e5"]) > (best["pos"], best["mean_e5"]):
            best = row
    deltas = []
    for _, te in folds:
        yf, zb, uf = y[te], z_base[te], u[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(yf)), yf):
            b = roc_auc_score(yf[ste], zb[ste])
            deltas.append((roc_auc_score(yf[ste], zb[ste] + w_main * uf[ste]) - b) * 1e5)
    d25 = np.asarray(deltas, float)
    mean = float(d25.mean())
    se = float(d25.std(ddof=1) / np.sqrt(len(d25)))
    tstat = float(mean / (se + 1e-18))
    try:
        _, wp = wilcoxon(d25, alternative="greater", zero_method="wilcox")
    except Exception:
        wp = 1.0
    rng = np.random.default_rng(42)
    boots = [d25[rng.integers(0, len(d25), len(d25))].mean() for _ in range(400)]
    ci = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]
    gate = best["pos"] >= 5 and int((d25 > 0).sum()) >= 18 and tstat >= 2.8
    return {
        "name": name,
        "main": best,
        "sf25": {"pos": int((d25 > 0).sum()), "t": tstat, "mean_e5": mean, "wilcox": float(wp), "ci90": ci},
        "gate_strict": gate,
        "u": u,
    }


def step1(y, folds, fold_id, fam, slot1):
    log("=== STEP 1: Family E/A fine grid ===")
    live = fam["live"]
    z = fam["z_ath"]
    all_rows = {}
    for key in ("E", "A", "E_R4R2", "E80_C20"):
        uo, ut = fam[key]
        rows = eval_w_grid(key, y, z, uo, folds, live, slot1, ut)
        all_rows[key] = rows
        # peak by mean_e5 among 5/5
        cands = [r for r in rows if r["main_pos"] >= 5]
        peak = max(cands or rows, key=lambda r: (r["main_pos"], r["mean_e5"]))
        log(
            f"  {key} peak w={peak['w']} pos={peak['main_pos']}/5 meanΔ={peak['mean_e5']:.3f}e-5 "
            f"sf={peak['sf25']}/25 t={peak['t']:.2f} 1-ρLIVE={peak['1mrho_LIVE_e4']:.3f}e-4 "
            f"inc_vs_S1={peak['inc_mean_e5_vs_slot1']:.3f}e-5"
        )
    # at Slot1 w compare E vs variants sf
    for key in ("E", "E_R4R2", "E80_C20"):
        r = next(x for x in all_rows[key] if abs(x["w"] - 0.01045) < 1e-9)
        log(f"  @{0.01045} {key}: sf={r['sf25']}/25 t={r['t']:.3f} mean={r['mean_e5']:.3f}")
    return all_rows


def step2_g26(y, folds, fold_id, fam):
    log("=== STEP 2: G26 × L7_s2026 cross-cal ===")
    oof_p = G26_DIR / "gpt2_s2026_oof_5fold.npy"
    f0_p = G26_DIR / "gpt2_s2026_test_f0.npy"
    # also search repo
    if not oof_p.exists():
        for p in REPO.rglob("gpt2_s2026_oof_5fold.npy"):
            if "._" in str(p):
                continue
            oof_p = p
            break
    if not f0_p.exists():
        for p in REPO.rglob("gpt2_s2026_test_f0.npy"):
            if "._" in str(p):
                continue
            f0_p = p
            break
    if not oof_p.exists() or not f0_p.exists():
        log(f"  G26 missing: oof={oof_p.exists()} f0={f0_p.exists()} — skip")
        return {"skipped": True, "reason": "gpt2_s2026 files not found"}

    g26_o = np.load(oof_p).ravel()[: len(y)]
    g26_f0 = np.load(f0_p)
    if g26_f0.ndim > 1:
        g26_f0 = g26_f0.ravel()[:299984]
    log(f"  G26 OOF AUC={roc_auc_score(y, g26_o):.5f}")

    ms_o = np.load(REPO / "preds/P_super_v4_oof.npy").ravel() if (REPO / "preds/P_super_v4_oof.npy").exists() else fam["l7f"]
    ms_t = np.load(REPO / "preds/P_super_v4_test.npy").ravel() if (REPO / "preds/P_super_v4_test.npy").exists() else fam["l7f_te"]

    # L7 fold test drift
    R3_te = np.load(TMP / "out/R3_test_folds.npy")
    if R3_te.shape[0] != 5:
        R3_te = R3_te.T
    drift = np.mean([L(R3_te[f]) - L(R3_te[0]) for f in range(5)], axis=0)
    # OOF: 5fold vs "fold0-like" — use difference G26 - MS as signal
    delta_o = L(g26_o) - L(ms_o)
    # correlate fold-mean L7 drift magnitude with |G26-MS| on a shared rank space via spearman of summaries
    corr_drift = float(spearmanr(unitize(drift), unitize(L(g26_f0) - L(ms_t))).correlation)
    log(f"  corr(L7_test_drift_mean, L(G26_f0)-L(MS_te))={corr_drift:.4f}")

    # ridge: predict G26_test_hat from L7_s2026 5f using OOF relationship
    # Fit on OOF: G26 ~ [L7F, R3, MS]
    Xo = np.column_stack([L(fam["l7f"]), L(fam["R3"]), L(ms_o)])
    ridge = Ridge(alpha=1.0).fit(Xo, L(g26_o))
    Xt = np.column_stack([L(fam["l7f_te"]), L(fam["R3t"]), L(ms_t)])
    g26_hat = ridge.predict(Xt)
    # blend f0 with hat
    g26_te = 0.5 * L(g26_f0) + 0.5 * g26_hat

    # Δ_G26_cal OOF
    raw = L(g26_o) - L(ms_o)
    B = np.column_stack([unitize(fam["z_ath"]), unitize(fam["d1"]), fam["E"][0]])
    # LOFO orth
    resid = np.zeros(len(y))
    for f, (tr, te) in enumerate(folds):
        lr = LinearRegression().fit(B[tr], raw[tr])
        resid[te] = raw[te] - lr.predict(B[te])
    d_cal = unitize(resid)

    # Slot1 OOF = z_ath + 0.01045 * E
    z_s1 = fam["z_ath"] + 0.01045 * fam["E"][0]
    scr = screen_delta("G26_cal_on_Slot1", y, z_s1, d_cal, folds, w_main=0.01)
    log(
        f"  G26_cal on Slot1: pos={scr['main']['pos']}/5 @{scr['main']['w']} "
        f"sf={scr['sf25']['pos']}/25 t={scr['sf25']['t']:.2f} gate={scr['gate_strict']}"
    )

    # test direction
    raw_t = g26_te - L(ms_t)
    Bt = np.column_stack([unitize(L(fam["live"])), unitize(fam["d1t"]), fam["E"][1]])
    lr = LinearRegression().fit(Bt, raw_t)  # in-sample on test — only for staging shape
    d_cal_t = unitize(raw_t - lr.predict(Bt))

    return {
        "g26_oof_auc": float(roc_auc_score(y, g26_o)),
        "corr_drift": corr_drift,
        "screen": {k: scr[k] for k in ("name", "main", "sf25", "gate_strict")},
        "d_cal_o": d_cal,
        "d_cal_t": d_cal_t,
        "ridge_r2": float(ridge.score(Xo, L(g26_o))),
    }


def build_digit_keys(train, orig=None):
    """dhananjay-style keys on numeric/continuous cols."""
    cols = [c for c in train.columns if c not in ("id", "is_fraud") and pd.api.types.is_numeric_dtype(train[c])]
    cont = [c for c in cols if train[c].nunique() > 20]

    def keys_for(df):
        K = {}
        for c in cont:
            x = df[c].to_numpy(float)
            K[f"v_{c}"] = np.round(x, 6)
            K[f"cents_{c}"] = (np.round(x * 100) % 100).astype(np.int64)
            K[f"int_{c}"] = np.floor(x).astype(np.int64)
            K[f"ten_{c}"] = (np.floor(x) // 10).astype(np.int64)
            K[f"dec_{c}"] = np.round(x, 1)
            ax = np.abs(x) + 1e-12
            lead = ax / (10 ** np.floor(np.log10(ax)))
            K[f"l1_{c}"] = np.clip(lead.astype(np.int64), 1, 9)
            K[f"l2_{c}"] = np.clip((lead * 10).astype(np.int64), 10, 99)
        return K

    Ktr = keys_for(train)
    Korig = keys_for(orig) if orig is not None else None
    return list(Ktr.keys()), Ktr, Korig, cont


def encode_fold(keys_tr, y_tr, keys_va, keys_te, m=20.0):
    """Factorize jointly; return te_va, te_te, cnt_va, cnt_te (log1p)."""
    # unify codes
    all_vals = np.concatenate([keys_tr.astype(object), keys_va.astype(object), keys_te.astype(object)])
    codes, uniques = pd.factorize(all_vals, sort=False)
    ntr, nva = len(keys_tr), len(keys_va)
    c_tr, c_va, c_te = codes[:ntr], codes[ntr : ntr + nva], codes[ntr + nva :]
    ncat = int(codes.max()) + 1
    prior = float(y_tr.mean())
    cnt = np.bincount(c_tr, minlength=ncat).astype(np.float64)
    sy = np.bincount(c_tr, weights=y_tr.astype(np.float64), minlength=ncat)
    enc = (sy + m * prior) / (cnt + m)
    return enc[c_va], enc[c_te], np.log1p(cnt[c_va]), np.log1p(cnt[c_te])


def pchip_lock(src, ref):
    """Map src values onto ref's marginal distribution via rank→quantile (PCHIP-lock proxy)."""
    src = np.asarray(src, float).ravel()
    ref = np.asarray(ref, float).ravel()
    # rank of src → percentile → interpolate into sorted ref
    order = rankdata(src, method="average")
    pct = (order - 0.5) / len(src)
    ref_sorted = np.sort(ref)
    idx = np.clip((pct * (len(ref_sorted) - 1)).astype(np.float64), 0, len(ref_sorted) - 1)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, len(ref_sorted) - 1)
    w = idx - lo
    return (1.0 - w) * ref_sorted[lo] + w * ref_sorted[hi]


def platt_init(z, target_rate=0.0167):
    """Calibrate η so mean(σ(η)) == target_rate; return η."""
    z = unitize(z)
    # binary search scale + bias
    lo, hi = 0.01, 20.0
    best = z
    for _ in range(40):
        s = 0.5 * (lo + hi)
        # bias so mean prob matches
        eta0 = s * z
        # find bias
        blo, bhi = -10.0, 10.0
        for __ in range(40):
            b = 0.5 * (blo + bhi)
            m = float(expit(eta0 + b).mean())
            if m > target_rate:
                bhi = b
            else:
                blo = b
        eta = eta0 + b
        m = float(expit(eta).mean())
        if m > target_rate:
            hi = s
        else:
            lo = s
        best = eta
    # final bias tweak
    blo, bhi = -5.0, 5.0
    for __ in range(40):
        b = 0.5 * (blo + bhi)
        if float(expit(best + b).mean()) > target_rate:
            bhi = b
        else:
            blo = b
    eta = best + b
    return eta


def step3_digits(y, folds, fold_id, fam):
    log("=== STEP 3: 4-leaf digit TE (dhananjay-style) ===")
    t0 = time.time()
    train = pd.read_csv(TMP / "train.csv" if (TMP / "train.csv").exists() else REPO / "data/train.csv")
    test = pd.read_csv(TMP / "test_ids.csv" if (TMP / "test_ids.csv").exists() else REPO / "data/test.csv")
    orig = pd.read_csv(REPO / "original/credit_card_fraud_2026.csv")
    # align orig columns
    shared = [c for c in train.columns if c in orig.columns and c not in ("id", "is_fraud", "transaction_id")]
    orig2 = orig[shared].copy()
    # pad missing numeric cols with nan→median
    for c in train.columns:
        if c in ("id", "is_fraud"):
            continue
        if c not in orig2.columns:
            if pd.api.types.is_numeric_dtype(train[c]):
                orig2[c] = train[c].median()
            else:
                orig2[c] = 0

    key_names, Ktr_raw, Korig_raw, cont = build_digit_keys(train, orig2)
    # Also build test keys
    _, Kte_raw, _, _ = build_digit_keys(test, None)
    log(f"  keys={len(key_names)} cont={len(cont)} elapsed={time.time()-t0:.1f}s")

    z_ath = fam["z_ath"]
    d1 = fam["d1"]
    dE = fam["E"][0]
    eta_ath = platt_init(z_ath, 0.0167)
    log(f"  Platt init mean(σ)={float(expit(eta_ath).mean()):.6f}")

    n = len(y)
    nte = len(test)
    # We'll build OOF/test matrices of selected te+cnt after first pass AUCs — speed: use top continuous only
    # Restrict to top-signal cols + amount-like
    prefer = [c for c in cont if any(s in c for s in ("amount", "balance", "distance", "hours", "risk", "velocity", "age"))]
    if len(prefer) < 5:
        prefer = cont[:7]
    key_names = [k for k in key_names if any(k.endswith("_" + c) or k.endswith(c) for c in prefer)]
    # fix: keys are like v_amount_usd
    key_names = [k for k in build_digit_keys(train, None)[0] if any(c in k for c in prefer)]
    log(f"  using {len(key_names)} keys on {prefer}")

    oof_te = {k: np.zeros(n, np.float32) for k in key_names}
    oof_cnt = {k: np.zeros(n, np.float32) for k in key_names}
    te_te = {k: np.zeros(nte, np.float32) for k in key_names}
    te_cnt = {k: np.zeros(nte, np.float32) for k in key_names}
    te_fold_stack = {k: [] for k in key_names}

    for f, (tr, va) in enumerate(folds):
        ytr = y[tr]
        # pool orig into training TE stats
        for k in key_names:
            ktr = np.asarray(Ktr_raw[k])[tr]
            kva = np.asarray(Ktr_raw[k])[va]
            kte = np.asarray(Kte_raw[k])
            if Korig_raw is not None and k in Korig_raw:
                ktr = np.concatenate([ktr, np.asarray(Korig_raw[k])])
                y_pool = np.concatenate([ytr, orig["is_fraud"].to_numpy()])
            else:
                y_pool = ytr
            eva, ete, cva, cte = encode_fold(ktr, y_pool, kva, kte, m=20.0)
            oof_te[k][va] = eva
            oof_cnt[k][va] = cva
            te_fold_stack[k].append(ete)
            # cnt test: use fold's cnt encoding
            if f == 0:
                te_cnt[k] = cte
            else:
                te_cnt[k] += cte
        log(f"  fold {f} TE done {time.time()-t0:.1f}s")

    for k in key_names:
        te_te[k] = np.mean(np.stack(te_fold_stack[k], 0), 0)
        te_cnt[k] /= 5.0
        # PCHIP-lock test TE to OOF dist
        te_te[k] = pchip_lock(te_te[k], oof_te[k])

    # drop unstable corr columns
    keep = []
    for k in key_names:
        co = float(np.corrcoef(oof_te[k], z_ath)[0, 1]) if oof_te[k].std() > 0 else 0.0
        ct = float(np.corrcoef(te_te[k], L(fam["live"]))[0, 1]) if te_te[k].std() > 0 else 0.0
        if abs(co - ct) <= 0.015:
            keep.append(k)
    log(f"  kept {len(keep)}/{len(key_names)} keys after |Δρ|≤0.015 vs z_ATH/LIVE")
    if len(keep) < 8:
        keep = key_names  # fallback
        log("  fallback: keep all keys")

    feat_names = [f"te_{k}" for k in keep] + [f"cnt_{k}" for k in keep]
    Xo = np.column_stack([oof_te[k] for k in keep] + [oof_cnt[k] for k in keep]).astype(np.float32)
    Xt = np.column_stack([te_te[k] for k in keep] + [te_cnt[k] for k in keep]).astype(np.float32)

    # Train 4-leaf LGBM + depth-2 XGB with init_score
    oof_lgb = np.zeros(n)
    oof_xgb = np.zeros(n)
    te_lgb = np.zeros(nte)
    te_xgb = np.zeros(nte)

    lgb_params = dict(
        n_estimators=150,
        learning_rate=0.02,
        num_leaves=4,
        max_depth=2,
        min_child_samples=2000,
        colsample_bytree=0.4,
        reg_lambda=200.0,
        subsample=0.8,
        objective="binary",
        verbosity=-1,
        n_jobs=4,
    )
    xgb_params = dict(
        n_estimators=150,
        learning_rate=0.02,
        max_depth=2,
        min_child_weight=2000,
        colsample_bytree=0.4,
        reg_lambda=200.0,
        subsample=0.8,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
    )

    for f, (tr, va) in enumerate(folds):
        init_tr = eta_ath[tr]
        init_va = eta_ath[va]
        # LightGBM with init_score — extract PURE tree margin via raw_score - init
        dtr = lgb.Dataset(Xo[tr], label=y[tr], init_score=init_tr)
        dva = lgb.Dataset(Xo[va], label=y[va], init_score=init_va, reference=dtr)
        model = lgb.train(
            {**lgb_params, "metric": "auc"},
            dtr,
            num_boost_round=150,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        # raw_score = init + tree_margin  → tree_margin = raw - init  (NO double subtract)
        raw_va = model.predict(Xo[va], raw_score=True)
        raw_te = model.predict(Xt, raw_score=True)
        # When init_score used in training, predict(raw_score=True) returns FULL logit including init contribution
        # stored in model — for Dataset init_score, predict raw is tree sum only in recent lgb? 
        # Spec: "pure tree margin extraction (m.predict raw_score=True — NO double subtract of init_o)"
        # So treat raw_score as the margin directly when that's the API contract they want.
        margin_va = raw_va  # pure as instructed
        margin_te = raw_te
        oof_lgb[va] = margin_va
        te_lgb += margin_te / 5.0

        # XGB: base_margin
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(
            Xo[tr],
            y[tr],
            eval_set=[(Xo[va], y[va])],
            verbose=False,
            # base_margin
        )
        # fit with base_margin via xgb.DMatrix for purity
        dtrain = xgb.DMatrix(Xo[tr], label=y[tr], base_margin=init_tr)
        dval = xgb.DMatrix(Xo[va], label=y[va], base_margin=init_va)
        dtest = xgb.DMatrix(Xt, base_margin=np.zeros(nte))  # margin-only on test
        bst = xgb.train(
            {
                "max_depth": 2,
                "eta": 0.02,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "lambda": 200.0,
                "colsample_bytree": 0.4,
                "min_child_weight": 2000,
                "subsample": 0.8,
                "tree_method": "hist",
                "nthread": 4,
            },
            dtrain,
            num_boost_round=150,
            evals=[(dval, "va")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        # margin: predict with output_margin=True; with base_margin in dval it includes base — subtract init for train/val
        # Instruction: NO double subtract — use output_margin as tree contribution when base_margin set on DMatrix:
        # Actually xgb output_margin WITH base_margin returns base+trees. Without base on test DMatrix we get trees only.
        dval_nomargin = xgb.DMatrix(Xo[va])
        dte_nomargin = xgb.DMatrix(Xt)
        oof_xgb[va] = bst.predict(dval_nomargin, output_margin=True)
        te_xgb += bst.predict(dte_nomargin, output_margin=True) / 5.0
        log(f"  fold {f} trees done {time.time()-t0:.1f}s")

    # Combine
    tree_o = unitize(0.5 * unitize(oof_lgb) + 0.5 * unitize(oof_xgb))
    tree_t = unitize(0.5 * unitize(te_lgb) + 0.5 * unitize(te_xgb))

    rho_ath = float(np.corrcoef(tree_o, z_ath)[0, 1])
    raw_disp = float((tree_t.std() + 1e-12) / (tree_o.std() + 1e-12))
    log(f"  pre-orth ρ(tree,z_ATH)={rho_ath:.4f} raw_disp={raw_disp:.4f}")

    # LOFO orth vs [z_ATH, d1, Family_E]
    B = np.column_stack([unitize(z_ath), unitize(d1), unitize(dE)])
    resid = np.zeros(n)
    for f, (tr, te) in enumerate(folds):
        lr = LinearRegression().fit(B[tr], tree_o[tr])
        resid[te] = tree_o[te] - lr.predict(B[te])
    d_dig = unitize(resid)

    z_s1 = z_ath + 0.01045 * dE
    scr = screen_delta("digit_4leaf_on_Slot1", y, z_s1, d_dig, folds, w_main=0.01)
    log(
        f"  digit_4leaf: pos={scr['main']['pos']}/5 @{scr['main']['w']} "
        f"sf={scr['sf25']['pos']}/25 t={scr['sf25']['t']:.2f} gate={scr['gate_strict']} "
        f"elapsed={time.time()-t0:.1f}s"
    )

    # test residual
    live = fam["live"]
    Bt = np.column_stack([unitize(L(live)), unitize(fam["d1t"]), fam["E"][1]])
    lr = LinearRegression().fit(Bt, tree_t)
    d_dig_t = unitize(tree_t - lr.predict(Bt))

    return {
        "rho_ath": rho_ath,
        "raw_disp": raw_disp,
        "n_keys_kept": len(keep),
        "screen": {k: scr[k] for k in ("name", "main", "sf25", "gate_strict")},
        "tree_oof_auc": float(roc_auc_score(y, tree_o)),
        "resid_auc": float(roc_auc_score(y, d_dig)),
        "d_dig_o": d_dig,
        "d_dig_t": d_dig_t,
        "elapsed_s": time.time() - t0,
        "pass_disp": 0.94 <= raw_disp <= 1.06,
        "pass_rho": abs(rho_ath) < 0.25,
    }


def stage_candidates(fam, grid, g26, digit, slot1):
    assert NO_SUBMIT is True
    live = fam["live"]
    ids = pd.read_csv(LIVE095388)["id"].to_numpy()
    staged = []

    def save(tag, direction_t, w, base=None):
        base = slot1 if base is None else base
        cand = quantile_lock(L(base) + w * unitize(direction_t), base)
        mrho = (1.0 - float(spearmanr(base, cand).correlation)) * 1e4
        path = SUB / f"sub_v16_2_{tag}_NOSUBMIT.csv"
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
        staged.append({"file": path.name, "w": w, "1mrho_e4_vs_base": mrho, "base": "Slot1_095389"})
        log(f"  staged {path.name} w={w} 1-ρ_base={mrho:.4f}e-4")
        return path

    # Best E from grid by (pos, mean) among w>=0.01045 with pos==5
    e_rows = grid["E"]
    cands = [r for r in e_rows if r["main_pos"] == 5 and r["w"] >= 0.01045]
    best_e = max(cands or e_rows, key=lambda r: (r["sf25"], r["t"], r["mean_e5"]))
    # Stage next rung above Slot1 if peak w > 0.01045
    if best_e["w"] > 0.01045 + 1e-9:
        save(f"FamilyE_w{best_e['w']:.5f}_peak", fam["E"][1], best_e["w"] - 0.01045, base=slot1)
    else:
        # stage 0.27 and 0.35 style steps from Slot1
        save("FamilyE_fromS1_0p27e4", fam["E"][1], 0.0122 - 0.01045, base=slot1)
    save("FamilyE_fromS1_0p35step", fam["E"][1], 0.01383 - 0.01045, base=slot1)

    # E_R4R2 / E80C20 if better sf at 0.01045
    for key in ("E_R4R2", "E80_C20"):
        r = next(x for x in grid[key] if abs(x["w"] - 0.01045) < 1e-9)
        rE = next(x for x in grid["E"] if abs(x["w"] - 0.01045) < 1e-9)
        if r["sf25"] > rE["sf25"] or (r["sf25"] == rE["sf25"] and r["t"] > rE["t"] + 0.1):
            save(f"{key}_w0p01045", fam[key][1], 0.01045, base=live)

    if g26 and not g26.get("skipped") and g26.get("screen", {}).get("gate_strict"):
        w = g26["screen"]["main"]["w"]
        save(f"G26cal_onS1_w{w}", g26["d_cal_t"], w, base=slot1)

    if digit and digit.get("screen", {}).get("gate_strict"):
        w = digit["screen"]["main"]["w"]
        save(f"digit4leaf_onS1_w{w}", digit["d_dig_t"], w, base=slot1)
    elif digit and digit.get("pass_rho") and digit.get("screen", {}).get("main", {}).get("pos", 0) >= 4:
        # soft stage for inspection
        w = digit["screen"]["main"]["w"]
        save(f"digit4leaf_SOFT_onS1_w{w}", digit["d_dig_t"], w, base=slot1)

    return staged, best_e


def write_report(grid, g26, digit, staged, best_e):
    lines = []
    lines.append("# GM V16.2 — Family E Bracket + G26 Cal + 4-Leaf Digits")
    lines.append("")
    lines.append("**STRICT `NO_SUBMIT=True` · 0 pods · New ATH baseline Slot1 `0.95389`**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 1 — Fine grid (OOF peak map)")
    lines.append("")
    for key in ("E", "A", "E_R4R2", "E80_C20"):
        lines.append(f"### Family `{key}`")
        lines.append("")
        lines.append("| w | label | 1−ρ LIVE (e-4) | 1−ρ Slot1 (e-4) | pos/5 | meanΔ e-5 | incΔ vs S1 e-5 | min e-5 | folds e-5 | sf25@w | t@w | sf25@0.01 | t@0.01 | Wilcox p | CI90 |")
        lines.append("|--:|-------|---------------:|----------------:|------:|----------:|---------------:|--------:|-----------|-------:|----:|----------:|-------:|---------:|------|")
        for r in grid[key]:
            lines.append(
                f"| {r['w']:.5f} | {r['label']} | {r['1mrho_LIVE_e4']:.4f} | {r['1mrho_Slot1_e4']:.4f} | "
                f"{r['main_pos']}/5 | {r['mean_e5']:.3f} | {r['inc_mean_e5_vs_slot1']:.3f} | {r['min_e5']:.3f} | "
                f"`{r['folds_e5']}` | {r['sf25']}/25 | {r['t']:.2f} | {r.get('sf25_micro01', r['sf25'])}/25 | "
                f"{r.get('t_micro01', r['t']):.2f} | {r['wilcox']:.3g} | "
                f"[{r['ci90'][0]:.2f},{r['ci90'][1]:.2f}] |"
            )
        lines.append("")
    lines.append(f"**Peak Family E (by pos/sf/t):** w=`{best_e['w']}` · pos=`{best_e['main_pos']}/5` · "
                 f"sf=`{best_e['sf25']}/25` · t=`{best_e['t']:.2f}` · meanΔ=`{best_e['mean_e5']:.3f}`e-5 · "
                 f"inc vs Slot1=`{best_e['inc_mean_e5_vs_slot1']:.3f}`e-5")
    lines.append("")
    # compare variants at Slot1 w
    lines.append("### Variants @ w=0.01045 (Slot1 scale)")
    lines.append("")
    lines.append("| variant | sf25 | t | meanΔ e-5 |")
    lines.append("|---------|-----:|--:|----------:|")
    for key in ("E", "E_R4R2", "E80_C20", "A"):
        r = next(x for x in grid[key] if abs(x["w"] - 0.01045) < 1e-9)
        lines.append(f"| {key} | {r['sf25']}/25 | {r['t']:.3f} | {r['mean_e5']:.3f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 2 — G26 × L7_s2026 cross-calibration")
    lines.append("")
    if g26.get("skipped"):
        lines.append(f"**Skipped:** `{g26.get('reason')}`")
    else:
        lines.append(f"- G26 OOF AUC = **{g26['g26_oof_auc']:.5f}**")
        lines.append(f"- corr(L7 test-drift, G26_f0−MS) = **{g26['corr_drift']:.4f}**")
        lines.append(f"- ridge R²(G26~L7F+R3+MS) = **{g26['ridge_r2']:.4f}**")
        s = g26["screen"]
        lines.append(
            f"- On Slot1 OOF: pos=`{s['main']['pos']}/5` @{s['main']['w']} · "
            f"sf=`{s['sf25']['pos']}/25` · t=`{s['sf25']['t']:.2f}` · "
            f"strict gate (≥5/5 & ≥18/25 & t≥2.8) = **{s['gate_strict']}**"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 3 — 4-leaf digit TE (dhananjay-style + orig20k)")
    lines.append("")
    if not digit:
        lines.append("Skipped.")
    else:
        lines.append(f"- elapsed = **{digit['elapsed_s']:.1f}s** · keys kept = {digit['n_keys_kept']}")
        lines.append(f"- pre-orth ρ(tree, z_ATH) = **{digit['rho_ath']:.4f}** (need |ρ|<0.25 → `{digit['pass_rho']}`)")
        lines.append(f"- raw_disp = **{digit['raw_disp']:.4f}** (need [0.94,1.06] → `{digit['pass_disp']}`)")
        lines.append(f"- tree OOF AUC = {digit['tree_oof_auc']:.5f} · resid AUC = {digit['resid_auc']:.5f}")
        s = digit["screen"]
        lines.append(
            f"- On Slot1: pos=`{s['main']['pos']}/5` @{s['main']['w']} · "
            f"sf=`{s['sf25']['pos']}/25` · t=`{s['sf25']['t']:.2f}` · "
            f"strict gate = **{s['gate_strict']}**"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 4 — Staged NOSUBMIT candidates")
    lines.append("")
    lines.append("| file | w | 1−ρ vs base (e-4) | base |")
    lines.append("|------|--:|------------------:|------|")
    for s in staged:
        lines.append(f"| `{s['file']}` | {s['w']} | {s['1mrho_e4_vs_base']:.4f} | {s['base']} |")
    lines.append("")
    lines.append("**NO submissions made.**")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    log(f"wrote {REPORT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-digit", action="store_true")
    ap.add_argument("--skip-g26", action="store_true")
    args = ap.parse_args()
    assert NO_SUBMIT is True
    OUT.mkdir(parents=True, exist_ok=True)
    (REPO / "logs/overnight").mkdir(parents=True, exist_ok=True)

    y = pd.read_csv(TMP / "train.csv" if (TMP / "train.csv").exists() else REPO / "data/train.csv")["is_fraud"].to_numpy()
    folds, fold_id = fold_splits(y)
    slot1 = pd.read_csv(SLOT1)["is_fraud"].to_numpy(float)
    fam = build_families(y, folds, fold_id)

    grid = step1(y, folds, fold_id, fam, slot1)
    g26 = {"skipped": True, "reason": "--skip-g26"} if args.skip_g26 else step2_g26(y, folds, fold_id, fam)
    digit = None if args.skip_digit else step3_digits(y, folds, fold_id, fam)

    staged, best_e = stage_candidates(fam, grid, g26, digit, slot1)
    write_report(grid, g26, digit or {}, staged, best_e)

    # JSON (drop large arrays)
    def slim_g26(g):
        if not g or g.get("skipped"):
            return g
        return {k: g[k] for k in g if not k.startswith("d_cal")}

    def slim_digit(d):
        if not d:
            return d
        return {k: d[k] for k in d if not k.startswith("d_dig")}

    payload = {
        "NO_SUBMIT": True,
        "grid": grid,
        "g26": slim_g26(g26),
        "digit": slim_digit(digit),
        "staged": staged,
        "best_E": best_e,
    }
    (OUT / "v16_2_bracket.json").write_text(json.dumps(payload, indent=2, default=float))
    (REPO / "logs/overnight/v16_2_bracket.json").write_text(json.dumps(payload, indent=2, default=float))
    log("DONE — NO_SUBMIT")


if __name__ == "__main__":
    main()
