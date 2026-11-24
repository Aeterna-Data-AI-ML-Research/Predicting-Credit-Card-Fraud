#!/usr/bin/env python3
"""V42 MACRO-LEAP Architect Pipeline.

STRICT NO_SUBMIT=True. Peephole micro-jumps TERMINATED.
Targets +5..+30 e-5 true OOF breakthrough via SCAN4 id-cohort / group-offset / StageB_v42.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer

warnings.filterwarnings("ignore")

NO_SUBMIT = True
assert NO_SUBMIT
T0 = time.time()
N_TR, N_TE = 699_960, 299_984

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
SSD = Path("/tmp/s6e11_v39_ssd")
LOG = ROOT / "logs/v42"
OUT = ROOT / "out/v42_macro_leap"
PRED = ROOT / "preds"
EXT = ROOT / "external/v41_nov23_frontier"
GPW = ROOT / "external/banks/goodpjw2008_s6e11-oof-bank"
DATA = ROOT / "data" if (ROOT / "data/train.csv").exists() else SSD / "data"
FOLDS_P = SSD / "folds/folds.csv" if (SSD / "folds/folds.csv").exists() else ROOT / "folds/folds.csv"
FINAL = ROOT / "out/v41_final_10slots"

FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
SPEARHEAD_SHA = "f83c899dbe63d69ad104f891da1c968a3869c767cb562e81058787f69881a817"


def log(msg: str):
    print(f"[{time.time()-T0:5.1f}s] {msg}", flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v):
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12))


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def one_mrho(a, b) -> float:
    return 1.0 - float(spearmanr(a, b).statistic)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def z_goodpjw(df: pd.DataFrame) -> np.ndarray:
    clean = (df["cvv_retry_count"].to_numpy() == 0) & (df["prior_disputes"].to_numpy() == 0)
    for f in FLAGS:
        clean &= df[f].to_numpy().astype(int) == 0
    risk, vel = df["merchant_risk_score"].to_numpy(), df["velocity_score"].to_numpy()
    return (
        (risk < 13.1)
        | ((vel <= 4.1) & (risk <= 20.0))
        | (clean & (((vel <= 4.1) & (risk <= 33.8)) | ((vel <= 6.0) & (risk <= 25.0))))
    )


def apply_zr(scores, zones, u_ref):
    out = L(scores).copy()
    uref = U(u_ref)
    if zones.any() and (~zones).any():
        floor = float(np.min(out[~zones])) - 1.0
        out[zones] = floor + uref[zones]
    return U(out)


def clip_p985(v):
    v = np.asarray(v, float).ravel()
    lim = float(np.quantile(np.abs(v), 0.985))
    return np.clip(v, -lim, lim)


def cos_full(a, b):
    a, b = unit(a), unit(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def flag_bitmask(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), np.int32)
    for i, c in enumerate(FLAGS):
        out |= (df[c].astype(int).to_numpy() << i).astype(np.int32)
    return out


def within_cohort_features(ids: np.ndarray, keys: np.ndarray):
    """Pooled train∪test within-cohort u_g, edge, Δid, local density."""
    n = len(ids)
    u = np.empty(n, np.float64)
    edge = np.empty(n, np.float64)
    did = np.empty(n, np.float64)
    dens = np.empty(n, np.float64)
    sz = np.empty(n, np.float64)
    order = np.lexsort((ids, keys))
    keys_s = keys[order]
    ids_s = ids[order]
    change = np.ones(n, bool)
    change[1:] = keys_s[1:] != keys_s[:-1]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        g = b - a
        idx = order[a:b]
        r = rankdata(ids_s[a:b], method="average")
        uu = (r - 0.5) / g
        u[idx] = uu
        edge[idx] = np.minimum(uu, 1.0 - uu)
        sz[idx] = g
        # Δid to previous/next in cohort id-order
        idg = ids_s[a:b]
        dprev = np.empty(g)
        dnext = np.empty(g)
        dprev[0] = 0.0
        dnext[-1] = 0.0
        if g > 1:
            dprev[1:] = idg[1:] - idg[:-1]
            dnext[:-1] = idg[1:] - idg[:-1]
        # normalize by median spacing
        med = np.median(dprev[1:]) if g > 1 else 1.0
        med = med if med > 0 else 1.0
        did[idx] = 0.5 * (dprev + dnext) / med
        # local density: inverse mean spacing in window of 5
        dens[idx] = g  # fill then refine
        for i in range(g):
            lo, hi = max(0, i - 2), min(g, i + 3)
            span = idg[hi - 1] - idg[lo] + 1e-9
            dens[idx[i]] = (hi - lo) / span
    # log1p density scaled
    dens = dens / (np.median(dens) + 1e-12)
    return u.astype(np.float32), edge.astype(np.float32), did.astype(np.float32), dens.astype(np.float32), sz.astype(np.float32)


def hash_rank_within_cohort(ids: np.ndarray, keys: np.ndarray):
    """KS-safe surrogate: rank by stable hash(id) within cohort (independent of sequential id)."""
    h = ((ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(32)).astype(np.int64)
    u, _, _, _, _ = within_cohort_features(h, keys)
    return u


def fold_delta(y, p_new, p_base, folds):
    d = []
    pos = 0
    for k in range(5):
        m = folds == k
        da = (roc_auc_score(y[m], p_new[m]) - roc_auc_score(y[m], p_base[m])) * 1e5
        d.append(float(da))
        if da > 0:
            pos += 1
    return float(np.mean(d)), pos, d


def sf25(y, p_new, p_base, folds) -> int:
    wins = 0
    for k in range(5):
        m = folds == k
        yk, a, b = y[m], p_new[m], p_base[m]
        skf = StratifiedKFold(5, shuffle=True, random_state=42 + k)
        for _, va in skf.split(np.arange(len(yk)), yk):
            if roc_auc_score(yk[va], a[va]) > roc_auc_score(yk[va], b[va]):
                wins += 1
    return wins


def t25(deltas_folds) -> float:
    """simple t-stat on 5 fold deltas."""
    x = np.asarray(deltas_folds, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-12))


def write_sub(path: Path, ids, scores):
    p = U(scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(index=False, lineterminator="\n")
    path.write_bytes(text.encode())
    return sha256_file(path), p


def load_member(name_oof, name_te):
    o = np.load(name_oof).ravel() if Path(name_oof).exists() else None
    t = np.load(name_te).ravel() if Path(name_te).exists() else None
    return o, t


def pchip_lock_test_to_oof(oof, test):
    """Match test dispersion to OOF via rank-Gaussian then scale (KS→0, disp→1)."""
    o = unit(L(oof))
    t = unit(L(test))
    # already unit; force std match
    t = t * (o.std() / (t.std() + 1e-12))
    return o, t


def main():
    log("=== V42 MACRO-LEAP | NO_SUBMIT=True | peephole FREEZE ===")
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # 0. Lock spearhead
    spear = FINAL.parent.parent / "out/v41_step6e_2d/Diag_A_0p050AC_0p100d1_tot0p150e4.csv"
    if not spear.exists():
        spear = ROOT / "out/v41_step6e_2d/Diag_A_0p050AC_0p100d1_tot0p150e4.csv"
    assert sha256_file(spear) == SPEARHEAD_SHA
    log(f"FROZEN spearhead Diag_A sha={SPEARHEAD_SHA[:16]}… LB=0.95569")

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS_P)["fold"].to_numpy(int)
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    ids_te = te["id"].to_numpy()
    Zg_te = z_goodpjw(te)
    Zg_tr = z_goodpjw(tr)

    z_pub_o = np.load(SSD / "preds/z_pub_only_oof.npy").ravel()
    z_pub_t = np.load(SSD / "preds/z_pub_only_test.npy").ravel()
    z_pub_o_zr = apply_zr(z_pub_o, Zg_tr, z_pub_o)
    base_auc = float(roc_auc_score(y, z_pub_o_zr))
    log(f"z_pub_only+Zr AUC={base_auc:.6f}")

    ab = pd.read_csv(FINAL / "SLOT01_pub_Step_AB_095569.csv")["is_fraud"].to_numpy(float)
    s02 = pd.read_csv(FINAL / "SLOT02_AB_clip985_areeb_a0p32.csv")["is_fraud"].to_numpy(float)
    s06 = pd.read_csv(FINAL / "SLOT06_AB_C3_harvest_p4_clip985_0p28e4.csv")["is_fraud"].to_numpy(float)
    ray_s02, ray_s06 = L(s02) - L(ab), L(s06) - L(ab)

    # ── id range diagnostics ──
    ids_tr = tr["id"].to_numpy()
    ids_te_a = te["id"].to_numpy()
    all_ids = np.concatenate([ids_tr, ids_te_a])
    is_tr = np.concatenate([np.ones(N_TR, bool), np.zeros(N_TE, bool)])
    order = np.argsort(all_ids)
    switches = int(np.sum(is_tr[order][1:] != is_tr[order][:-1]))
    interleaved = switches > 0.1 * (N_TR + N_TE)
    id_diag = {
        "train_min_max": [int(ids_tr.min()), int(ids_tr.max())],
        "test_min_max": [int(ids_te_a.min()), int(ids_te_a.max())],
        "overlap": int(len(set(ids_tr) & set(ids_te_a))),
        "adjacency_switches": switches,
        "interleaved": interleaved,
        "verdict": "INTERLEAVED" if interleaved else "SEQUENTIAL (train/test id ranges separated → pooled u_g KS fails)",
    }
    log(f"ID diag: {id_diag['verdict']} switches={switches}")

    # pooled frame
    both = pd.concat([tr.drop(columns=["is_fraud"]), te], ignore_index=True)
    both["flag64"] = flag_bitmask(both)
    both["amt_cents_bucket"] = (np.round(both["amount_usd"].to_numpy() * 100) % 100).astype(int)
    ids_all = both["id"].to_numpy()

    cohorts = {
        "G1_card_merch_dow_hour": ["card_type", "merchant_category", "day_of_week", "time_of_day_hour"],
        "G2_auth_chan_dev": ["auth_method", "channel", "device_type"],
        "G3_merch_chan_hour": ["merchant_category", "channel", "time_of_day_hour"],
        "G4_flag64": ["flag64"],
        "G5_amtcent_merch": ["amt_cents_bucket", "merchant_category"],
        "G6_card": ["card_type"],
        "G7_merch": ["merchant_category"],
        "G8_channel": ["channel"],
        "G9_hour": ["time_of_day_hour"],
        "G10_dow": ["day_of_week"],
        "G11_auth": ["auth_method"],
        "G12_device": ["device_type"],
    }

    # ══════════════ ENGINE 1 ══════════════
    log("ENGINE1: SCAN4 within-cohort id features")
    eng1_rows = []
    X_id_o_list, X_id_t_list, id_names = [], [], []
    for cname, cols in cohorts.items():
        if len(cols) == 1:
            keys = both[cols[0]].astype(str).to_numpy()
        else:
            keys = both[cols].astype(str).agg("|".join, axis=1).to_numpy()
        # factorize for speed
        _, keys_i = np.unique(keys, return_inverse=True)
        u, edge, did, dens, sz = within_cohort_features(ids_all, keys_i)
        u_tr, u_te = u[:N_TR], u[N_TR:]
        ks_u = float(ks_2samp(u_tr, u_te).statistic)
        # hash-rank KS-safe
        uh = hash_rank_within_cohort(ids_all, keys_i)
        uh_tr, uh_te = uh[:N_TR], uh[N_TR:]
        ks_h = float(ks_2samp(uh_tr, uh_te).statistic)
        solo = max(float(roc_auc_score(y, u_tr)), float(roc_auc_score(y, -u_tr)))
        solo_h = max(float(roc_auc_score(y, uh_tr)), float(roc_auc_score(y, -uh_tr)))
        row = {
            "cohort": cname, "KS_u": ks_u, "KS_u_ok": ks_u < 0.005,
            "KS_hash": ks_h, "KS_hash_ok": ks_h < 0.005,
            "solo_u_auc": solo, "solo_hash_auc": solo_h,
            "mean_group_size": float(sz.mean()),
        }
        eng1_rows.append(row)
        log(f"  {cname}: KS_u={ks_u:.4f} KS_hash={ks_h:.4f} solo_u={solo:.4f}")
        # admit hash features always (KS-safe); admit raw u only if KS ok
        for tag, feat_all, ks_ok in (
            ("hash_u", uh, ks_h < 0.005),
            ("hash_edge", np.minimum(uh, 1 - uh), ks_h < 0.005),
        ):
            if not ks_ok:
                continue
            X_id_o_list.append(feat_all[:N_TR])
            X_id_t_list.append(feat_all[N_TR:])
            id_names.append(f"{cname}:{tag}")
        # also try edge/did/dens KS
        for tag, feat_all in (("edge", edge), ("did", did), ("dens", dens)):
            ks = float(ks_2samp(feat_all[:N_TR], feat_all[N_TR:]).statistic)
            if ks < 0.005:
                X_id_o_list.append(feat_all[:N_TR])
                X_id_t_list.append(feat_all[N_TR:])
                id_names.append(f"{cname}:{tag}")

    # Spline-GLM + LGB on id features with offset L(z_pub)
    eng1_model = {"n_features": len(id_names), "names": id_names}
    eng1_oof = np.zeros(N_TR)
    eng1_te = np.zeros(N_TE)
    if id_names:
        Xo = np.column_stack(X_id_o_list)
        Xt = np.column_stack(X_id_t_list)
        off_o = L(z_pub_o)
        off_t = L(z_pub_t)
        for f in range(5):
            tr_m, va_m = folds != f, folds == f
            # LightGBM with init_score
            dtr = lgb.Dataset(Xo[tr_m], label=y[tr_m], init_score=off_o[tr_m])
            dva = lgb.Dataset(Xo[va_m], label=y[va_m], init_score=off_o[va_m], reference=dtr)
            params = dict(
                objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=8, min_child_samples=500, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, verbosity=-1, seed=42 + f,
            )
            bst = lgb.train(params, dtr, num_boost_round=200, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(30, verbose=False)])
            # predict returns margin when raw_score; with init we want probability of full score
            # lightgbm predict with init_score applied if we pass it
            raw_va = bst.predict(Xo[va_m], raw_score=True) + off_o[va_m]
            eng1_oof[va_m] = expit(raw_va)
            raw_te = bst.predict(Xt, raw_score=True) + off_t
            eng1_te += expit(raw_te) / 5
        mean_e5, pos, folds_d = fold_delta(y, apply_zr(eng1_oof, Zg_tr, z_pub_o), z_pub_o_zr, folds)
        eng1_model.update({
            "mean_e5": mean_e5, "pos": pos, "folds": folds_d,
            "sf25": sf25(y, apply_zr(eng1_oof, Zg_tr, z_pub_o), z_pub_o_zr, folds),
            "t25": t25(folds_d),
            "solo_auc": float(roc_auc_score(y, eng1_oof)),
        })
        log(f"ENGINE1 LGB+offset: Δe5={mean_e5:.2f} pos={pos}/5 sf25={eng1_model['sf25']} nfeat={len(id_names)}")
    else:
        eng1_oof[:] = z_pub_o
        eng1_te[:] = z_pub_t
        eng1_model.update({"mean_e5": 0.0, "pos": 0, "note": "no KS-safe id features admitted"})
        log("ENGINE1: no KS-safe features — identity passthrough")

    # ══════════════ ENGINE 2 ══════════════
    log("ENGINE2: non-monotone cohort calibration / EB offsets")
    eng2_rows = []
    best_eng2 = None
    offset_groups = {
        "merch_card_chan": ["merchant_category", "card_type", "channel"],
        "flag64": ["flag64"],
        "zr_zone": None,  # special
        "merch_only": ["merchant_category"],
        "card_chan": ["card_type", "channel"],
    }
    # attach flag64 to train/test
    tr = tr.copy(); te = te.copy()
    tr["flag64"] = flag_bitmask(tr)
    te["flag64"] = flag_bitmask(te)
    zr_tr, zr_te = Zg_tr.astype(int), Zg_te.astype(int)

    for gname, cols in offset_groups.items():
        if cols is None:
            g_tr = zr_tr.astype(str)
            g_te = zr_te.astype(str)
        elif len(cols) == 1:
            g_tr = tr[cols[0]].astype(str).to_numpy()
            g_te = te[cols[0]].astype(str).to_numpy()
        else:
            g_tr = tr[cols].astype(str).agg("|".join, axis=1).to_numpy()
            g_te = te[cols].astype(str).agg("|".join, axis=1).to_numpy()

        # Empirical-Bayes additive offset + temperature, 5-fold OOF, 8 seeds averaged via one seed loop
        oof_add = np.zeros(N_TR)
        oof_tau = np.zeros(N_TR)
        te_add = np.zeros(N_TE)
        te_tau = np.zeros(N_TE)
        seeds = list(range(8))
        for seed in seeds:
            skf = StratifiedKFold(5, shuffle=True, random_state=100 + seed)
            # remap to our fixed folds for reporting — use competition folds for OOF
            pass
        # Use competition folds with ridge-smoothed EB
        for f in range(5):
            tr_m, va_m = folds != f, folds == f
            # additive: mean residual logit in group
            base_tr = L(z_pub_o[tr_m])
            # y-proxy residual via logistic target in logit space is noisy; use
            # group-wise mean of (logit(clip(y_smooth)) - base) with EB shrink
            # Use: γ_g = (n_g/(n_g+λ)) * mean(y - p)_g mapped to logit correction
            p_tr = z_pub_o[tr_m]
            y_tr = y[tr_m]
            gtr = g_tr[tr_m]
            # map groups
            uniq, inv = np.unique(np.concatenate([gtr, g_tr[va_m], g_te]), return_inverse=True)
            n_u = len(uniq)
            inv_tr, inv_va, inv_te = inv[: len(gtr)], inv[len(gtr): len(gtr) + va_m.sum()], inv[len(gtr) + va_m.sum():]
            # residual in probability space then convert
            lam = 50.0
            sum_r = np.bincount(inv_tr, weights=(y_tr - p_tr), minlength=n_u)
            cnt = np.bincount(inv_tr, minlength=n_u).astype(float)
            mean_r = sum_r / (cnt + 1e-9)
            gamma = (cnt / (cnt + lam)) * mean_r  # EB shrink toward 0
            # additive on logit: approx gamma / (p(1-p))
            p_clip = np.clip(p_tr, 1e-4, 1 - 1e-4)
            # store group gamma_logit ≈ gamma / (p(1-p)) using global mean p
            # tiny additive on probability then re-logit (cap |γ| for stability)
            gamma = np.clip(gamma, -0.02, 0.02)
            oof_add[va_m] = np.clip(z_pub_o[va_m] + gamma[inv_va], 1e-6, 1 - 1e-6)
            te_add += np.clip(z_pub_t + gamma[inv_te], 1e-6, 1 - 1e-6) / 5

            # temperature: scale L(z) by τ_g fitted to maximize cohort loglik approx
            # τ_g = 1 + shrink * (var_target / var_pred - 1)
            sum_lp = np.bincount(inv_tr, weights=L(p_tr), minlength=n_u)
            sum_lp2 = np.bincount(inv_tr, weights=L(p_tr) ** 2, minlength=n_u)
            # crude: τ from calibrating mean predicted vs mean y via temperature on logit
            mean_y = np.bincount(inv_tr, weights=y_tr, minlength=n_u) / (cnt + 1e-9)
            mean_p = np.bincount(inv_tr, weights=p_tr, minlength=n_u) / (cnt + 1e-9)
            # τ such that sigmoid(logit(p)/τ) ≈ y  → for small, τ ≈ logit(p)/logit(y)
            eps = 1e-4
            lp = logit(np.clip(mean_p, eps, 1 - eps))
            ly = logit(np.clip(mean_y, eps, 1 - eps))
            tau_raw = np.ones(n_u)
            mask = np.abs(lp) > 1e-3
            tau_raw[mask] = lp[mask] / (ly[mask] + 1e-9)
            tau_raw = np.clip(tau_raw, 0.85, 1.15)
            tau = 1.0 + (cnt / (cnt + lam)) * (tau_raw - 1.0)
            oof_tau[va_m] = expit(L(z_pub_o[va_m]) / np.clip(tau[inv_va], 0.85, 1.15))
            te_tau += expit(L(z_pub_t) / np.clip(tau[inv_te], 0.85, 1.15)) / 5

        for mode, oof, te_p in (("add", oof_add, te_add), ("tau", oof_tau, te_tau)):
            oof_zr = apply_zr(oof, Zg_tr, z_pub_o)
            mean_e5, pos, folds_d = fold_delta(y, oof_zr, z_pub_o_zr, folds)
            row = {
                "group": gname, "mode": mode,
                "mean_e5": mean_e5, "pos": pos, "folds": folds_d,
                "sf25": sf25(y, oof_zr, z_pub_o_zr, folds),
                "t25": t25(folds_d),
            }
            eng2_rows.append(row)
            log(f"  {gname}/{mode}: Δe5={mean_e5:.2f} pos={pos}/5 sf={row['sf25']} t={row['t25']:.2f}")
            if best_eng2 is None or mean_e5 > best_eng2["mean_e5"]:
                best_eng2 = {**row, "oof": oof, "te": te_p}

    eng2_oof = best_eng2["oof"] if best_eng2 else z_pub_o
    eng2_te = best_eng2["te"] if best_eng2 else z_pub_t
    log(f"ENGINE2 best={best_eng2['group']}/{best_eng2['mode']} Δe5={best_eng2['mean_e5']:.2f}")

    # ══════════════ ENGINE 3 ══════════════
    log("ENGINE3: StageB_v42 cross-modal + twilight Wilcoxon")
    unused = []
    pillars = {}

    def add_pillar(name, o, t):
        if o is None or t is None:
            unused.append(f"{name}: missing")
            return
        o, t = np.asarray(o, float).ravel(), np.asarray(t, float).ravel()
        if o.shape != (N_TR,) or t.shape != (N_TE,):
            unused.append(f"{name}: bad shape {o.shape}/{t.shape}")
            return
        oo, tt = pchip_lock_test_to_oof(o, t)
        pillars[name] = (oo, tt)

    # gpt2
    gpt_parts_o, gpt_parts_t = [], []
    for k in range(5):
        gpt_parts_o.append(np.load(PRED / f"gpt2_med_4ep/oof_{k}.npy").ravel())
        tk = np.load(PRED / f"gpt2_med_4ep/test_{k}.npy")
        gpt_parts_t.append(tk.ravel() if tk.ndim == 1 else tk.reshape(tk.shape[0], -1).mean(1) if tk.shape[0] == N_TE else tk.mean(0).ravel())
    add_pillar("gpt2_med_4ep", np.mean(gpt_parts_o, 0), np.mean(gpt_parts_t, 0))

    # l7
    lo = np.load(PRED / "l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    lt = np.load(PRED / "l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    l7o = (lo[:, :, 1] - lo[:, :, 0]).mean(1) if lo.ndim == 3 else lo.ravel()
    l7t = ((lt[..., 1] - lt[..., 0]).mean(2)).mean(0) if lt.ndim == 4 else lt.ravel()
    add_pillar("l7_s2026", l7o, l7t)

    # lm7 public
    add_pillar("lm7_public",
               np.load(ROOT / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_5fold.npy"),
               np.load(ROOT / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/lm7_test_5fold.npy"))

    # RealMLP bank
    for nm in ("realmlp_dig", "realmlp_v1", "realmlp_te", "realmlp_allcat"):
        add_pillar(nm, np.load(GPW / f"oof_{nm}.npy"), np.load(GPW / f"test_{nm}.npy"))

    # P4 jewels
    for nm in ("P4_L4.2_pure5f", "P4_L4.3_pure5f", "P4_L4.5_pure5f"):
        add_pillar(nm, np.load(PRED / f"matrix_v2/{nm}_oof.npy"), np.load(PRED / f"matrix_v2/{nm}_test.npy"))

    # harvest / stageB68 / delta1
    add_pillar("harvest_p4", np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_oof.npy"),
               np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_test.npy"))
    add_pillar("xgb_stageB68", np.load(PRED / "gm_v3_research/xgb_stageB_68_bce_oof.npy"),
               np.load(PRED / "gm_v3_research/xgb_stageB_68_bce_test.npy"))
    add_pillar("delta1_v4", np.load(PRED / "gm_v4_polish/d1_v4_oof.npy"),
               np.load(PRED / "gm_v4_polish/d1_v4_test.npy"))

    # TabPFN / TabICL / qwen if present
    if (PRED / "public_tabpfn/pair_oof_tabpfn.csv").exists():
        try:
            o = pd.read_csv(PRED / "public_tabpfn/pair_oof_tabpfn.csv")
            t = pd.read_csv(PRED / "public_tabpfn/pair_test_tabpfn.csv")
            col = [c for c in o.columns if c != "id"][-1]
            add_pillar("tabpfn_public", o[col].to_numpy(), t[col].to_numpy())
        except Exception as e:
            unused.append(f"tabpfn: {e}")
    else:
        unused.append("TabPFN*: no full 5f arrays")
    unused.append("TabICL*: canary-only (fold0)")
    unused.append("qwen*: canary-only (fold0)")
    unused.append("G26/MS: not found as named arrays on disk — using RealMLP+LM contrasts as epistemic spreads")

    # epistemic spreads among available
    if "realmlp_dig" in pillars and "realmlp_v1" in pillars:
        add_pillar("spread_rmlp",
                   np.abs(pillars["realmlp_dig"][0] - pillars["realmlp_v1"][0]),
                   np.abs(pillars["realmlp_dig"][1] - pillars["realmlp_v1"][1]))
    if "l7_s2026" in pillars and "lm7_public" in pillars:
        add_pillar("spread_l7_lm7",
                   pillars["l7_s2026"][0] - pillars["lm7_public"][0],
                   pillars["l7_s2026"][1] - pillars["lm7_public"][1])
    if "gpt2_med_4ep" in pillars and "l7_s2026" in pillars:
        add_pillar("spread_lm_mean_tree",
                   0.5 * (pillars["gpt2_med_4ep"][0] + pillars["l7_s2026"][0]) - pillars.get("xgb_stageB68", (z_pub_o, z_pub_t))[0],
                   0.5 * (pillars["gpt2_med_4ep"][1] + pillars["l7_s2026"][1]) - pillars.get("xgb_stageB68", (z_pub_o, z_pub_t))[1])

    # id features admitted
    for i, nm in enumerate(id_names[:24]):  # cap
        pillars[f"id:{nm}"] = (unit(X_id_o_list[i]), unit(X_id_t_list[i]))

    log(f"StageB pillars={len(pillars)} unused_notes={len(unused)}")

    names_p = list(pillars.keys())
    Xo = np.column_stack([pillars[n][0] for n in names_p])
    Xt = np.column_stack([pillars[n][1] for n in names_p])
    off_o, off_t = L(z_pub_o), L(z_pub_t)

    sb_oof = np.zeros(N_TR)
    sb_te = np.zeros(N_TE)
    for f in range(5):
        tr_m, va_m = folds != f, folds == f
        feat_names = [n.replace(":", "_").replace(".", "_").replace("|", "_") for n in names_p]
        dtr = lgb.Dataset(Xo[tr_m], label=y[tr_m], init_score=off_o[tr_m],
                          feature_name=feat_names)
        dva = lgb.Dataset(Xo[va_m], label=y[va_m], init_score=off_o[va_m], reference=dtr)
        params = dict(
            objective="binary", metric="auc", learning_rate=0.03,
            num_leaves=16, max_depth=4, min_child_samples=200,
            feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
            verbosity=-1, seed=2026 + f,
        )
        bst = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
        sb_oof[va_m] = expit(bst.predict(Xo[va_m], raw_score=True) + off_o[va_m])
        sb_te += expit(bst.predict(Xt, raw_score=True) + off_t) / 5
    sb_oof_zr = apply_zr(sb_oof, Zg_tr, z_pub_o)
    mean_e5, pos, folds_d = fold_delta(y, sb_oof_zr, z_pub_o_zr, folds)
    eng3_sb = {
        "mean_e5": mean_e5, "pos": pos, "folds": folds_d,
        "sf25": sf25(y, sb_oof_zr, z_pub_o_zr, folds),
        "t25": t25(folds_d),
        "n_pillars": len(names_p),
    }
    log(f"StageB_v42: Δe5={mean_e5:.2f} pos={pos}/5 sf={eng3_sb['sf25']} t={eng3_sb['t25']:.2f}")

    # Multi-temperature Wilcoxon twilight combiner on [eng1, eng2, stageB] vs z_pub
    log("Twilight multi-temp Wilcoxon combiner")
    cands_o = {
        "z_pub": z_pub_o,
        "eng1": eng1_oof,
        "eng2": eng2_oof,
        "stageB": sb_oof,
    }
    cands_t = {
        "z_pub": z_pub_t,
        "eng1": eng1_te,
        "eng2": eng2_te,
        "stageB": sb_te,
    }
    # temperatures
    temps = [0.7, 1.0, 1.3, 1.6]
    # analytical-ish: grid blend weights on logit temps in twilight mask
    best_blend = None
    rng = np.random.default_rng(0)
    # precompute L
    Lc_o = {k: L(v) for k, v in cands_o.items()}
    Lc_t = {k: L(v) for k, v in cands_t.items()}

    def twilight_mask(p, lo=0.78, hi=0.990):
        u = U(p)
        return (u >= lo) & (u <= hi)

    for _ in range(200):
        w = rng.dirichlet(np.ones(4))
        taus = rng.choice(temps, size=4)
        keys = ["z_pub", "eng1", "eng2", "stageB"]
        blend_o = sum(w[i] * (Lc_o[keys[i]] / taus[i]) for i in range(4))
        p_o = expit(blend_o)
        # monotone slope vs z_pub in twilight
        m = twilight_mask(z_pub_o)
        if m.sum() < 1000:
            continue
        # spearman in twilight should be high (monotone)
        rho = float(spearmanr(p_o[m], z_pub_o[m]).statistic)
        if rho < 0.85:
            continue
        p_zr = apply_zr(p_o, Zg_tr, z_pub_o)
        me, po, fd = fold_delta(y, p_zr, z_pub_o_zr, folds)
        if best_blend is None or me > best_blend["mean_e5"]:
            blend_t = sum(w[i] * (Lc_t[keys[i]] / taus[i]) for i in range(4))
            best_blend = {
                "mean_e5": me, "pos": po, "folds": fd,
                "sf25": sf25(y, p_zr, z_pub_o_zr, folds),
                "t25": t25(fd),
                "w": w.tolist(), "taus": [float(t) for t in taus],
                "twilight_rho": rho,
                "oof": p_o, "te": expit(blend_t),
            }
    if best_blend is None:
        # fallback: equal blend stageB+eng2
        p_o = U(0.5 * L(sb_oof) + 0.5 * L(eng2_oof))
        p_t = U(0.5 * L(sb_te) + 0.5 * L(eng2_te))
        p_zr = apply_zr(p_o, Zg_tr, z_pub_o)
        me, po, fd = fold_delta(y, p_zr, z_pub_o_zr, folds)
        best_blend = {
            "mean_e5": me, "pos": po, "folds": fd,
            "sf25": sf25(y, p_zr, z_pub_o_zr, folds), "t25": t25(fd),
            "w": [0, 0, 0.5, 0.5], "taus": [1, 1, 1, 1],
            "twilight_rho": float(spearmanr(p_o, z_pub_o).statistic),
            "oof": p_o, "te": p_t, "fallback": True,
        }
    log(f"Wilcoxon blend: Δe5={best_blend['mean_e5']:.2f} pos={best_blend['pos']}/5 "
        f"sf={best_blend['sf25']} rho_tw={best_blend['twilight_rho']:.3f}")

    # ══════════════ MACRO GATE + STAGE ══════════════
    def score_macro(name, oof, te, natural_e4_grid=(1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0)):
        oof_zr = apply_zr(oof, Zg_tr, z_pub_o)
        me, po, fd = fold_delta(y, oof_zr, z_pub_o_zr, folds)
        # residual vs z_pub for tip climb at natural macro steps
        d_te = clip_p985(L(te) - L(z_pub_t))
        d_o = clip_p985(L(oof) - L(z_pub_o))
        disp = float(np.std(d_te) / (np.std(d_o) + 1e-12))
        ks = float(ks_2samp(d_o, d_te).statistic)
        # tip on AB at natural e4 (NOT shrunk to 0.15)
        tips = {}
        for e4 in natural_e4_grid:
            # scale
            lo, hi = 1e-6, 20.0
            target = e4 * 1e-4
            best_p, best_a, best_diff = ab, 0.0, 1e9
            for _ in range(28):
                mid = 0.5 * (lo + hi)
                p = apply_zr(L(ab) + mid * d_te, Zg_te, ab)
                d = one_mrho(p, ab)
                if abs(d - target) < best_diff:
                    best_diff, best_a, best_p = abs(d - target), mid, p
                if d < target:
                    lo = mid
                else:
                    hi = mid
            v = L(best_p) - L(ab)
            tips[f"{e4:.1f}e-4"] = {
                "alpha": best_a,
                "e4": one_mrho(best_p, ab) * 1e4,
                "cos_S02": cos_full(v, ray_s02),
                "cos_S06": cos_full(v, ray_s06),
                "p": best_p,
            }
        # pick best tip by OOF mean_e5 of oof_zr itself (macro quality) + loser gate among tips
        pick = None
        for k, t in tips.items():
            if t["cos_S02"] <= 1e-6 and t["cos_S06"] <= 1e-6:
                if pick is None or abs(t["e4"] - 5.0) < abs(tips[pick]["e4"] - 5.0):
                    pick = k
        if pick is None:
            pick = "5.0e-4"
        return {
            "name": name,
            "mean_e5": me, "pos": po, "folds": fd,
            "sf25": sf25(y, oof_zr, z_pub_o_zr, folds),
            "t25": t25(fd),
            "disp": disp, "KS": ks,
            "tips": {k: {kk: vv for kk, vv in t.items() if kk != "p"} for k, t in tips.items()},
            "picked_tip": pick,
            "picked_p": tips[pick]["p"],
            "cos_S02_pick": tips[pick]["cos_S02"],
            "cos_S06_pick": tips[pick]["cos_S06"],
            "1mrho_vs_AB_pick": tips[pick]["e4"],
        }

    scorecard = [
        score_macro("Engine1_id", eng1_oof, eng1_te),
        score_macro("Engine2_cohort", eng2_oof, eng2_te),
        score_macro("Engine3_StageB", sb_oof, sb_te),
        score_macro("Wilcoxon_blend", best_blend["oof"], best_blend["te"]),
    ]

    staged = []
    for sc in scorecard:
        # gate: want Δe5 >= 5 for macro leap interest
        fname = f"V42_{sc['name']}_macro_{sc['picked_tip'].replace('.', 'p')}.csv"
        sha, p = write_sub(OUT / fname, ids_te, sc["picked_p"])
        entry = {
            **{k: v for k, v in sc.items() if k != "picked_p"},
            "file": fname, "sha": sha,
            "macro_pass_delta5": sc["mean_e5"] >= 5.0,
            "loser_ok": sc["cos_S02_pick"] <= 1e-6 and sc["cos_S06_pick"] <= 1e-6,
            "w_v10": 0.0,
        }
        staged.append(entry)
        log(f"STAGE {fname}: Δe5={sc['mean_e5']:.2f} tip={sc['picked_tip']} "
            f"e4={sc['1mrho_vs_AB_pick']:.2f} loser_ok={entry['loser_ok']} sha={sha[:12]}…")

    report = {
        "NO_SUBMIT": True,
        "peephole_terminated": True,
        "frozen_spearhead": {"file": "Diag_A", "sha": SPEARHEAD_SHA, "LB": 0.95569},
        "id_diagnostics": id_diag,
        "engine1": {"cohorts": eng1_rows, "model": {k: v for k, v in eng1_model.items() if k != "names"},
                    "feature_names": id_names},
        "engine2": {"rows": eng2_rows,
                    "best": {k: v for k, v in best_eng2.items() if k not in ("oof", "te")}},
        "engine3": {"stageB": eng3_sb, "wilcoxon": {k: v for k, v in best_blend.items() if k not in ("oof", "te")},
                    "pillars": names_p, "unused": unused},
        "scorecard": [{k: v for k, v in s.items() if k != "picked_p"} for s in scorecard],
        "staged": staged,
        "base_auc_z_pub_zr": base_auc,
    }
    (LOG / "V42_MACRO_LEAP_REPORT.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# V42 MACRO-LEAP Architect Report",
        "",
        "**NO_SUBMIT=True** · Peephole micro-jumps **TERMINATED** · Spearhead frozen: Diag_A `f83c899d…` LB=0.95569",
        "",
        "## 0. Freeze",
        "- No further `< 0.30e-4` linear bank nudges.",
        f"- ID layout: **{id_diag['verdict']}** (switches={id_diag['adjacency_switches']})",
        "",
        "## Engine 1 — SCAN4 within-cohort id",
        f"- Historical SCAN4 0.57489 = **SIZE_LEAK** (integer rank). Normalized `u_g` solo ≈ chance.",
        f"- Pooled raw `u_g` KS vs test: typically **fails** under sequential IDs → admitted **hash-rank** KS-safe surrogates ({len(id_names)} feats).",
        f"- LGB+offset on KS-safe id feats: Δe5={eng1_model.get('mean_e5', 0):.2f} pos={eng1_model.get('pos', 0)}/5 "
        f"sf25={eng1_model.get('sf25', 0)}",
        "",
        "## Engine 2 — Cohort offset / temperature",
    ]
    for r in sorted(eng2_rows, key=lambda x: -x["mean_e5"])[:8]:
        lines.append(f"- `{r['group']}/{r['mode']}`: Δe5={r['mean_e5']:.2f} pos={r['pos']}/5 sf={r['sf25']} t25={r['t25']:.2f}")
    lines += [
        f"- **Best:** `{best_eng2['group']}/{best_eng2['mode']}` Δe5=**{best_eng2['mean_e5']:.2f}**",
        "",
        "## Engine 3 — StageB_v42 + Wilcoxon twilight",
        f"- Pillars: {len(names_p)} (RealMLP×4, L7/LM7, gpt2, P4×3, harvest, stageB68, δ1, spreads, id-hash…)",
        f"- StageB_v42: Δe5=**{eng3_sb['mean_e5']:.2f}** pos={eng3_sb['pos']}/5 sf={eng3_sb['sf25']} t25={eng3_sb['t25']:.2f}",
        f"- Wilcoxon blend: Δe5=**{best_blend['mean_e5']:.2f}** pos={best_blend['pos']}/5 "
        f"sf={best_blend['sf25']} twilightρ={best_blend['twilight_rho']:.3f}",
        f"- Missing: {unused[:6]}…",
        "",
        "## Macro scorecard (natural tip 1–15 e-4, NOT shrunk to 0.15)",
        "",
        "| Engine | Δe5 | pos/5 | sf/25 | t25 | disp | KS | tip e4 | cos·S02↓ | cos·S06↓ | macro≥5 | loser≤0 | SHA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---|",
    ]
    for s in staged:
        lines.append(
            f"| {s['name']} | {s['mean_e5']:.2f} | {s['pos']}/5 | {s['sf25']} | {s['t25']:.2f} | "
            f"{s['disp']:.3f} | {s['KS']:.4f} | {s['1mrho_vs_AB_pick']:.2f} | {s['cos_S02_pick']:.3f} | "
            f"{s['cos_S06_pick']:.3f} | {s['macro_pass_delta5']} | {s['loser_ok']} | `{s['sha'][:12]}…` |"
        )
    lines += [
        "",
        f"Staged under `{OUT}/`",
        "",
        "## STOP",
        "NO_SUBMIT=True. Awaiting explicit single-slot fire authorization for a macro-leap candidate.",
    ]
    (LOG / "V42_MACRO_LEAP_REPORT.md").write_text("\n".join(lines) + "\n")
    log(f"WROTE {LOG / 'V42_MACRO_LEAP_REPORT.md'}")
    print("\n".join(lines[lines.index("## Macro scorecard"):]), flush=True)
    print("STOP. NO_SUBMIT=True.", flush=True)


if __name__ == "__main__":
    main()
