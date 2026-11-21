#!/usr/bin/env python3
"""GM V32.1 — Dissect SCAN4 local-rank 0.57489 + entity-graph unlock on wide_lm.

STRICT: NO_SUBMIT = True. Local Mac CPU. Target < 90s.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

NO_SUBMIT = True
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
WIDE_OOF = REPO / "external/nov08_part12/wide_lm_stack_nested.npy"
WIDE_TE = REPO / "external/nov08_part12/wide_lm_stack_test.npy"
OUT = Path("/tmp/s6e11_v32_1")
REPORT = REPO / "logs/overnight/GM_V32_1_DISSECT_0575_AND_GRAPH_UNLOCK.md"
N_TR, N_TE = 699960, 299984
T0 = time.time()
LINES: list[str] = []
RES: dict = {}


def log(msg: str) -> None:
    s = f"[{time.time() - T0:6.1f}s] {msg}"
    print(s, flush=True)
    LINES.append(s)


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def L(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, float).ravel(), 1e-6, 1 - 1e-6)
    return logit(v)


def Lf(v: np.ndarray, fold_id: np.ndarray) -> np.ndarray:
    out = np.zeros(len(v), float)
    for f in np.unique(fold_id):
        m = fold_id == f
        x = np.clip(v[m], 1e-6, 1 - 1e-6)
        # rank-frac then logit within fold (stable)
        r = (pd.Series(x).rank(method="average").to_numpy() - 0.5) / len(x)
        out[m] = logit(np.clip(r, 1e-6, 1 - 1e-6))
    return out


def oof_auc_tree(x: np.ndarray, y: np.ndarray, depth=4, seed=42) -> float:
    X = np.asarray(x, float).reshape(-1, 1)
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=2000, random_state=seed)
        dt.fit(X[tr], y[tr])
        oof[te] = dt.predict_proba(X[te])[:, 1]
    return auc(y, oof)


def oof_auc_te(keys, y, seed=42, min_cnt=5) -> float:
    keys = pd.Series(keys)
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    Xdummy = np.zeros((len(y), 1))
    for tr, te in skf.split(Xdummy, y):
        tab = pd.Series(y[tr]).groupby(keys.iloc[tr]).agg(["mean", "count"])
        means = tab["mean"].where(tab["count"] >= min_cnt)
        oof[te] = keys.iloc[te].map(means).fillna(y[tr].mean()).to_numpy()
    return auc(y, oof)


def leave_self_stats(z: np.ndarray, gkey: np.ndarray) -> dict[str, np.ndarray]:
    """Leave-self-out peer mean/max/std/contrast via group aggregates."""
    s = pd.Series(z)
    g = pd.Series(gkey)
    sum_g = g.map(s.groupby(g).sum())
    cnt_g = g.map(s.groupby(g).count()).astype(float)
    mean_lso = (sum_g - s) / (cnt_g - 1.0).clip(lower=1.0)
    # peer max approx: if self is unique max, use second; else group max
    max_g = g.map(s.groupby(g).max())
    # crude: when z==max and count of max==1, need second max — skip exact; use max of others approx
    # For speed: peer_max = max_g if z < max_g else group second max via transform
    # Use: (sum of squares for std)
    sum2 = g.map((s * s).groupby(g).sum())
    # var of others
    n = (cnt_g - 1.0).clip(lower=1.0)
    mean_lso_safe = mean_lso.copy()
    var_lso = ((sum2 - s * s) / n) - mean_lso_safe**2
    std_lso = np.sqrt(np.clip(var_lso, 0, None))
    # peer max: groupby transform alternative
    # For groups with n>=2: if z < max_g → max_g else NaN→mean of rest
    peer_max = max_g.where(s < max_g, mean_lso)
    contrast = s - mean_lso
    return {
        "peer_mean": mean_lso.to_numpy(float),
        "peer_max": peer_max.to_numpy(float),
        "peer_std": std_lso.to_numpy(float),
        "contrast": contrast.to_numpy(float),
        "cnt": cnt_g.to_numpy(float),
    }


def r2_vs_bank(v: np.ndarray, B: np.ndarray) -> float:
    lr = Ridge(alpha=1.0).fit(B, v)
    pred = lr.predict(B)
    ss_res = ((v - pred) ** 2).sum()
    ss_tot = ((v - v.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-18))


def main() -> None:
    assert NO_SUBMIT is True
    log("=== GM V32.1 dissect 0.57489 local-rank + graph unlock | NO_SUBMIT ===")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    # ── load ──
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    y = tr["is_fraud"].to_numpy(float)
    fold_id = pd.read_csv(BANK / "folds.csv")["fold"].to_numpy()
    z_oof = np.load(WIDE_OOF).ravel()[:N_TR].astype(float)
    z_te = np.load(WIDE_TE).ravel()[:N_TE].astype(float)
    base_auc = auc(y, z_oof)
    log(f"wide_lm AUC={base_auc:.6f} | train={len(tr)} test={len(te)}")

    tr["is_test"] = 0
    te["is_test"] = 1
    te["is_fraud"] = np.nan
    all_df = pd.concat([tr, te], ignore_index=True, sort=False)
    all_df = all_df.sort_values("id").reset_index(drop=True)
    assert len(all_df) == N_TR + N_TE
    is_tr = all_df["is_test"].to_numpy() == 0
    # map train position in all_df
    id_to_pos = {int(i): p for p, i in enumerate(all_df["id"].to_numpy())}
    tr_pos = np.array([id_to_pos[int(i)] for i in tr["id"].to_numpy()])

    # z_all in all_df order
    z_all = np.zeros(len(all_df))
    z_tr_by_id = dict(zip(tr["id"].astype(int), z_oof))
    z_te_by_id = dict(zip(te["id"].astype(int), z_te))
    # fold-aware L for train, rank-logit for test
    zL_tr = Lf(z_oof, fold_id)
    # test: global rank logit of test preds
    r_te = (pd.Series(z_te).rank(method="average").to_numpy() - 0.5) / len(z_te)
    zL_te = logit(np.clip(r_te, 1e-6, 1 - 1e-6))
    for i, row_id in enumerate(all_df["id"].to_numpy()):
        rid = int(row_id)
        if rid in z_tr_by_id:
            # use L_fold value — need index in train
            pass
    # faster: place by masks
    tr_ids = tr["id"].to_numpy()
    te_ids = te["id"].to_numpy()
    pos_tr = all_df["id"].searchsorted(tr_ids)  # all_df sorted by id; ids unique sequential?
    # safer map
    pos_map = pd.Series(np.arange(len(all_df)), index=all_df["id"].to_numpy())
    z_all[pos_map.loc[tr_ids].to_numpy()] = zL_tr
    z_all[pos_map.loc[te_ids].to_numpy()] = zL_te
    # also raw prob space for residual prop
    z_prob_all = np.zeros(len(all_df))
    z_prob_all[pos_map.loc[tr_ids].to_numpy()] = z_oof
    z_prob_all[pos_map.loc[te_ids].to_numpy()] = z_te

    # ═══════════════════════════════════════════════════════════
    # TEST 1 — dissect local-rank
    # ═══════════════════════════════════════════════════════════
    log("TEST1: dissect integer-rank vs group_size vs uniform u_id_grp…")

    def flag_bitmask(df: pd.DataFrame) -> np.ndarray:
        bits = [
            df["is_foreign_transaction"].astype(int).to_numpy(),
            df["is_new_merchant"].astype(int).to_numpy(),
            df["used_vpn"].astype(int).to_numpy(),
            df["ip_country_mismatch"].astype(int).to_numpy(),
            df["billing_shipping_mismatch"].astype(int).to_numpy(),
            df["is_ai_generated_scam_attempt"].astype(int).to_numpy(),
        ]
        bm = np.zeros(len(df), np.int64)
        for i, b in enumerate(bits):
            bm |= b.astype(np.int64) << i
        cvv = df["cvv_retry_count"].clip(0, 7).to_numpy(dtype=np.int64)
        bm |= cvv << 6
        return bm

    groups = {
        "G1_card_merch_dow_hour": (
            all_df["card_type"].astype(str)
            + "|"
            + all_df["merchant_category"].astype(str)
            + "|"
            + all_df["day_of_week"].astype(str)
            + "|"
            + all_df["time_of_day_hour"].astype(str)
        ),
        "G2_exact_balance": all_df["account_balance_usd"].round(6).astype(str),
        "G3_exact_amount": all_df["amount_usd"].round(6).astype(str),
        "G4_merch_chan_dev_auth": (
            all_df["merchant_category"].astype(str)
            + "|"
            + all_df["channel"].astype(str)
            + "|"
            + all_df["device_type"].astype(str)
            + "|"
            + all_df["auth_method"].astype(str)
        ),
        "G5_flag_bitmask": pd.Series(flag_bitmask(all_df).astype(str)),
    }

    t1 = {}
    # also train-only G1 for comparison
    g1_tr_only = (
        tr["card_type"].astype(str)
        + "|"
        + tr["merchant_category"].astype(str)
        + "|"
        + tr["day_of_week"].astype(str)
        + "|"
        + tr["time_of_day_hour"].astype(str)
    )

    def dissect_group2(name: str, gkey_all: pd.Series):
        g = gkey_all
        order = np.lexsort((all_df["id"].to_numpy(), pd.factorize(g, sort=False)[0]))
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        g_sorted = g.to_numpy()[order]
        # stable group sizes + ranks
        # pandas way faster for cumcount after sort
        tmp = pd.DataFrame({"g": g_sorted, "idx": order})
        tmp["rank"] = tmp.groupby("g", sort=False).cumcount()
        tmp["size"] = tmp.groupby("g", sort=False)["g"].transform("size")
        rank_all = np.empty(len(g), int)
        size_all = np.empty(len(g), float)
        rank_all[tmp["idx"].to_numpy()] = tmp["rank"].to_numpy()
        size_all[tmp["idx"].to_numpy()] = tmp["size"].to_numpy(float)
        u_all = (rank_all + 0.5) / size_all

        rank_tr = rank_all[is_tr]
        size_tr = size_all[is_tr]
        u_tr = u_all[is_tr]
        m2 = size_tr >= 2
        corr_u_size = float(np.corrcoef(u_tr[m2], size_tr[m2])[0, 1]) if m2.sum() > 100 else float("nan")
        corr_rank_size = float(np.corrcoef(rank_tr[m2].astype(float), size_tr[m2])[0, 1]) if m2.sum() > 100 else float("nan")

        a_rank = oof_auc_tree(rank_tr, y)
        a_size = oof_auc_tree(size_tr, y)
        a_u = oof_auc_tree(u_tr, y)
        a_u_ge2 = oof_auc_tree(u_tr[m2], y[m2])
        a_rank_ge2 = oof_auc_tree(rank_tr[m2], y[m2])

        row = {
            "n_groups": int(g.nunique()),
            "frac_size_ge2": round(float(m2.mean()), 4),
            "corr(u, size)": round(corr_u_size, 6),
            "corr(rank, size)": round(corr_rank_size, 6),
            "auc_integer_rank": round(a_rank, 5),
            "auc_group_size": round(a_size, 5),
            "auc_u_id_grp": round(a_u, 5),
            "auc_rank_size_ge2": round(a_rank_ge2, 5),
            "auc_u_size_ge2": round(a_u_ge2, 5),
            "verdict": (
                "SIZE_LEAK"
                if (a_size >= a_u - 0.01 and a_size > 0.53 and a_u < a_size + 0.01)
                else ("TRUE_ORDER" if a_u > 0.53 and abs(corr_u_size) < 0.08 else "WEAK/NULL")
            ),
        }
        t1[name] = row
        log(
            f"  {name}: n_g={row['n_groups']} corr(u,size)={corr_u_size:+.4f} | "
            f"AUC rank={a_rank:.4f} size={a_size:.4f} u={a_u:.4f} → {row['verdict']}"
        )
        return rank_all, size_all, u_all, g

    g_cache = {}
    for name, gk in groups.items():
        g_cache[name] = dissect_group2(name, gk if isinstance(gk, pd.Series) else pd.Series(gk))

    # G6: global id quantile train vs all
    u_global_all = (all_df["id"].to_numpy(float) + 0.5) / len(all_df)
    u_global_tr = (tr["id"].to_numpy(float) + 0.5) / N_TR
    t1["G6_global_id_quantile_in_all"] = {
        "auc_u": round(oof_auc_tree(u_global_all[is_tr], y), 5),
        "auc_raw_id": round(oof_auc_tree(tr["id"].to_numpy(float), y), 5),
        "auc_u_train_only_denom": round(oof_auc_tree(u_global_tr, y), 5),
    }
    log(f"  G6 global id AUC u_all={t1['G6_global_id_quantile_in_all']['auc_u']:.4f} raw={t1['G6_global_id_quantile_in_all']['auc_raw_id']:.4f}")

    # train-only G1 (reproduce SCAN4)
    # rank within train-only groups
    order = np.lexsort((tr["id"].to_numpy(), pd.factorize(g1_tr_only, sort=False)[0]))
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    gs = g1_tr_only.to_numpy()[order]
    tmp = pd.DataFrame({"g": gs, "idx": order})
    tmp["rank"] = tmp.groupby("g", sort=False).cumcount()
    tmp["size"] = tmp.groupby("g", sort=False)["g"].transform("size")
    rank_tr_only = np.empty(N_TR, int)
    size_tr_only = np.empty(N_TR, float)
    rank_tr_only[tmp["idx"]] = tmp["rank"]
    size_tr_only[tmp["idx"]] = tmp["size"]
    u_tr_only = (rank_tr_only + 0.5) / size_tr_only
    t1["G1_train_only_SCAN4_repro"] = {
        "auc_integer_rank": round(oof_auc_tree(rank_tr_only, y), 5),
        "auc_group_size": round(oof_auc_tree(size_tr_only, y), 5),
        "auc_u_id_grp": round(oof_auc_tree(u_tr_only, y), 5),
        "corr(u,size)": round(float(np.corrcoef(u_tr_only, size_tr_only)[0, 1]), 6),
    }
    log(
        f"  G1 train-only SCAN4 repro: rank={t1['G1_train_only_SCAN4_repro']['auc_integer_rank']:.4f} "
        f"size={t1['G1_train_only_SCAN4_repro']['auc_group_size']:.4f} "
        f"u={t1['G1_train_only_SCAN4_repro']['auc_u_id_grp']:.4f}"
    )

    # Neighbor lag within groups
    log("TEST1b: within-cohort id-sorted neighbor lags…")
    lag_rows = {}
    y_all = np.full(len(all_df), np.nan)
    y_all[is_tr] = y

    def neighbor_shift(arr, g_s, shift: int):
        """shift>0 → look at i-shift (past); shift<0 → look at i+|shift| (future). Same-group only."""
        out = np.full(len(arr), np.nan)
        if shift > 0:
            valid = np.zeros(len(arr), bool)
            valid[shift:] = g_s[shift:] == g_s[:-shift]
            out[shift:] = arr[:-shift]
            out = np.where(valid, out, np.nan)
        else:
            sh = -shift
            valid = np.zeros(len(arr), bool)
            valid[:-sh] = g_s[:-sh] == g_s[sh:]
            out[:-sh] = arr[sh:]
            out = np.where(valid, out, np.nan)
        return out

    for name in ["G1_card_merch_dow_hour", "G2_exact_balance", "G3_exact_amount", "G4_merch_chan_dev_auth"]:
        _, _, _, g = g_cache[name]
        order = np.lexsort((all_df["id"].to_numpy(), pd.factorize(g, sort=False)[0]))
        g_s = g.to_numpy()[order]
        y_s = y_all[order]
        z_s = z_prob_all[order]
        id_s = all_df["id"].to_numpy(float)[order]
        feat = {}
        for lag, shift in [("-1", 1), ("+1", -1), ("-2", 2), ("+2", -2)]:
            ys = neighbor_shift(y_s, g_s, shift)
            zs = neighbor_shift(z_s, g_s, shift)
            ids = neighbor_shift(id_s, g_s, shift)
            iddiff = np.where(np.isfinite(ids), id_s - ids, np.nan)
            # map back to all_df order
            ys_all = np.empty(len(order))
            zs_all = np.empty(len(order))
            idd_all = np.empty(len(order))
            ys_all[order] = ys
            zs_all[order] = zs
            idd_all[order] = iddiff
            yt, zt, idd = ys_all[is_tr], zs_all[is_tr], idd_all[is_tr]
            m = np.isfinite(yt)
            if m.sum() < 1000:
                feat[lag] = {"n": int(m.sum()), "auc_y_neighbor": None}
                continue
            a_y = auc(y[m], yt[m])
            mz = np.isfinite(zt)
            a_z = auc(y[mz], zt[mz]) if mz.sum() > 1000 else None
            mi = np.isfinite(idd)
            a_id = oof_auc_tree(idd[mi], y[mi]) if mi.sum() > 1000 else None
            feat[lag] = {
                "n": int(m.sum()),
                "auc_y_neighbor": round(a_y, 5),
                "auc_z_neighbor": round(a_z, 5) if a_z is not None else None,
                "auc_id_diff": round(a_id, 5) if a_id is not None else None,
                "mean_y_neighbor_when_y1": round(float(yt[m & (y == 1)].mean()), 5) if (m & (y == 1)).any() else None,
                "base": round(float(y[m].mean()), 5),
            }
        lag_rows[name] = feat
        best = max(
            ((lag, v["auc_y_neighbor"]) for lag, v in feat.items() if v.get("auc_y_neighbor")),
            default=(None, None),
            key=lambda x: x[1] or 0,
        )
        log(f"  lag {name}: best y-neighbor AUC={best}")

    # is_test structure
    log("TEST1c: is_test vs id structure…")
    is_test = all_df["is_test"].to_numpy()
    a_test_id = oof_auc_tree(all_df["id"].to_numpy(float), is_test, seed=0)
    # local test fraction windows
    ids = all_df["id"].to_numpy()
    # since sorted by id and contiguous 0..N-1?
    # check
    id_diff_nn = np.full(len(all_df), np.nan)
    # nearest test row distance for train rows
    test_ids = all_df.loc[~is_tr, "id"].to_numpy()
    tr_ids_sorted = all_df.loc[is_tr, "id"].to_numpy()
    # searchsorted
    idx = np.searchsorted(test_ids, tr_ids_sorted)
    idx = np.clip(idx, 1, len(test_ids) - 1)
    d1 = np.abs(tr_ids_sorted - test_ids[idx])
    d0 = np.abs(tr_ids_sorted - test_ids[idx - 1])
    d_near = np.minimum(d0, d1)
    a_dnear = oof_auc_tree(d_near, y)
    # rolling test fraction in id window k=20,100 on all_df order
    it = is_test.astype(float)
    cs = np.cumsum(np.concatenate([[0.0], it]))
    def roll_frac(k):
        # window centered-ish: [i-k//2, i+k//2]
        half = k // 2
        n = len(it)
        left = np.clip(np.arange(n) - half, 0, n)
        right = np.clip(np.arange(n) + half + 1, 0, n)
        return (cs[right] - cs[left]) / (right - left)

    f20 = roll_frac(20)[is_tr]
    f100 = roll_frac(100)[is_tr]
    t1_split = {
        "auc_id_predicts_is_test": round(a_test_id, 5),
        "auc_id_diff_to_nearest_test_vs_y": round(a_dnear, 5),
        "auc_local_test_frac_k20_vs_y": round(oof_auc_tree(f20, y), 5),
        "auc_local_test_frac_k100_vs_y": round(oof_auc_tree(f100, y), 5),
        "mean_d_near": round(float(d_near.mean()), 3),
    }
    log(f"  is_test: AUC(id→is_test)={a_test_id:.4f} | d_near vs y={a_dnear:.4f} | f20={t1_split['auc_local_test_frac_k20_vs_y']:.4f}")
    RES["test1"] = {"groups": t1, "lags": lag_rows, "split": t1_split}

    # ═══════════════════════════════════════════════════════════
    # TEST 2 — graph propagation on wide_lm
    # ═══════════════════════════════════════════════════════════
    log("TEST2: entity-graph peer props on wide_lm…")
    # bank matrix for R² (top members)
    bank_oofs = []
    for op in sorted(BANK.glob("oof_*.npy"))[:80]:
        try:
            o = np.load(op).ravel()[:N_TR]
            if len(o) == N_TR and np.isfinite(o).all() and np.ptp(o) > 0.01:
                bank_oofs.append(o)
        except Exception:
            pass
    B80 = np.column_stack(bank_oofs) if bank_oofs else np.zeros((N_TR, 1))
    log(f"  bank cols for R²: {B80.shape[1]}")

    entity_keys = {
        "k1_balance": all_df["account_balance_usd"].round(6).astype(str),
        "k2_amount": all_df["amount_usd"].round(6).astype(str),
        "k3_distance": all_df["distance_from_home_km"].round(6).astype(str),
        "k4_velocity": all_df["velocity_score"].round(6).astype(str),
        "k5_merch_risk": all_df["merchant_risk_score"].round(6).astype(str),
        "k6_hours": all_df["hours_since_last_txn"].round(6).astype(str),
        "k7_bal_merch": all_df["account_balance_usd"].round(6).astype(str) + "|" + all_df["merchant_category"].astype(str),
        "k8_amt_merch": all_df["amount_usd"].round(6).astype(str) + "|" + all_df["merchant_category"].astype(str),
    }

    t2 = []
    peer_feat_tr = {}  # store best feats for test4

    def eval_delta(name, feat_tr):
        # incremental: blend / stack with wide_lm via simple Ridge on [z, feat] OOF
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        oof = np.zeros(N_TR)
        X = np.column_stack([z_oof, feat_tr])
        for tri, tei in skf.split(X, y):
            lr = Ridge(alpha=1.0).fit(X[tri], y[tri])
            oof[tei] = lr.predict(X[tei])
        a = auc(y, oof)
        d_e4 = (a - base_auc) * 1e4
        # also solo
        a_solo = auc(y, feat_tr) if np.ptp(feat_tr) > 0 else 0.5
        r2 = r2_vs_bank(feat_tr, B80[:, : min(40, B80.shape[1])])
        return a, d_e4, a_solo, r2

    for kname, gkey in entity_keys.items():
        st = leave_self_stats(z_all, gkey.to_numpy())
        for stat in ["peer_mean", "peer_max", "peer_std", "contrast"]:
            feat_all = st[stat]
            feat_tr = feat_all[is_tr]
            # only where cnt>=2
            m = st["cnt"][is_tr] >= 2
            f = feat_tr.copy()
            f[~m] = f[m].mean() if m.any() else 0.0
            a, d_e4, a_solo, r2 = eval_delta(f"{kname}_{stat}", f)
            row = {
                "name": f"{kname}_{stat}",
                "auc_stack": round(a, 6),
                "delta_e4": round(d_e4, 3),
                "auc_solo": round(a_solo, 5),
                "R2_bank40": round(r2, 4),
                "frac_cnt_ge2": round(float(m.mean()), 4),
            }
            t2.append(row)
            if abs(d_e4) > 1.0:
                peer_feat_tr[row["name"]] = f
        log(f"  {kname} peer_mean Δ={t2[-4]['delta_e4']:+.2f}e-4 stack={t2[-4]['auc_stack']:.6f}")

    # 2-hop: balance → amount peer of peers
    log("  2-hop bipartite balance↔amount…")
    st_bal = leave_self_stats(z_all, entity_keys["k1_balance"].to_numpy())
    # for each row, take peer_mean over balance, then average those over amount group leave-self
    hop1 = st_bal["peer_mean"]
    st_amt_of_hop1 = leave_self_stats(hop1, entity_keys["k2_amount"].to_numpy())
    hop2 = st_amt_of_hop1["peer_mean"]
    f2 = hop2[is_tr]
    a, d_e4, a_solo, r2 = eval_delta("hop2_bal_amt", f2)
    t2.append({"name": "hop2_bal_to_amt", "auc_stack": round(a, 6), "delta_e4": round(d_e4, 3), "auc_solo": round(a_solo, 5), "R2_bank40": round(r2, 4)})
    peer_feat_tr["hop2_bal_to_amt"] = f2
    log(f"  hop2 Δ={d_e4:+.2f}e-4")

    # Fold-safe residual propagation
    log("  fold-safe residual TE on balance/amount…")
    resid = y - z_oof
    for key_name, col in [("balance", "account_balance_usd"), ("amount", "amount_usd")]:
        keys = tr[col].round(6)
        for m_sm in [3, 10, 30]:
            oof_r = np.zeros(N_TR)
            for f in range(5):
                te_m = fold_id == f
                tr_m = ~te_m
                # sum resid / (cnt + m) per key on train fold
                tab = pd.DataFrame({"k": keys[tr_m], "r": resid[tr_m]}).groupby("k")["r"].agg(["sum", "count"])
                sm = (tab["sum"] / (tab["count"] + m_sm)).to_dict()
                oof_r[te_m] = keys[te_m].map(sm).fillna(0.0).to_numpy()
            a, d_e4, a_solo, r2 = eval_delta(f"resid_{key_name}_m{m_sm}", oof_r)
            t2.append({"name": f"resid_{key_name}_m{m_sm}", "auc_stack": round(a, 6), "delta_e4": round(d_e4, 3), "auc_solo": round(a_solo, 5), "R2_bank40": round(r2, 4)})
            if d_e4 > 1:
                peer_feat_tr[f"resid_{key_name}_m{m_sm}"] = oof_r
            log(f"  resid {key_name} m={m_sm}: Δ={d_e4:+.2f}e-4 stack={a:.6f} R²={r2:.3f}")

    t2_sorted = sorted(t2, key=lambda r: -r["delta_e4"])
    RES["test2"] = t2_sorted
    log(f"  TEST2 best: {t2_sorted[0]['name']} Δ={t2_sorted[0]['delta_e4']:+.2f}e-4")

    # ═══════════════════════════════════════════════════════════
    # TEST 3 — float string forensics
    # ═══════════════════════════════════════════════════════════
    log("TEST3: raw CSV float-string forensics…")

    def load_raw_float_strings(path: Path, cols: list[str], n_rows: int) -> dict[str, np.ndarray]:
        # read header
        with open(path) as f:
            header = f.readline().rstrip("\n").split(",")
            idx = {c: header.index(c) for c in cols}
            out = {c: [] for c in cols}
            for i, line in enumerate(f):
                if i >= n_rows:
                    break
                # naive CSV split OK — no commas in float fields
                parts = line.rstrip("\n").split(",")
                for c, j in idx.items():
                    out[c].append(parts[j])
        return {c: np.array(v, dtype=object) for c, v in out.items()}

    float_cols = [
        "amount_usd",
        "account_balance_usd",
        "distance_from_home_km",
        "hours_since_last_txn",
        "velocity_score",
        "merchant_risk_score",
    ]
    raw_tr = load_raw_float_strings(DATA / "train.csv", float_cols, N_TR)
    t3 = []
    str_feats = {}
    for col in float_cols:
        s = raw_tr[col]
        str_len = np.array([len(x) for x in s], float)
        n_dec = np.array([len(x.split(".")[1]) if "." in x else 0 for x in s], float)
        trail = np.array([
            0 if "." not in x else (3 if x.endswith(".00") else 2 if x.endswith(".0") else 1 if x[-1].isdigit() else 0)
            for x in s
        ], float)
        # last digit of cents
        def last_cent(x):
            if "." not in x:
                return 0
            frac = x.split(".")[1]
            return int(frac[-1]) if frac else 0

        last_d = np.array([last_cent(x) for x in s], float)
        # residue vs float64
        parsed = tr[col].to_numpy(float)
        # re-parse from string
        reparsed = np.array([float(x) for x in s])
        residue = parsed - reparsed  # should be 0
        for fname, feat in [
            ("str_len", str_len),
            ("n_dec", n_dec),
            ("trail_pat", trail),
            ("last_digit", last_d),
        ]:
            a_solo = oof_auc_te(feat.astype(int) if feat.dtype == float else feat, y) if fname != "str_len" else oof_auc_tree(feat, y)
            # residual target TE
            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            oof_te = np.zeros(N_TR)
            keys = pd.Series(feat)
            for tri, tei in skf.split(oof_te.reshape(-1, 1), y):
                # TE on residual
                tab = pd.Series(resid[tri]).groupby(keys.iloc[tri]).mean()
                oof_te[tei] = keys.iloc[tei].map(tab).fillna(0).to_numpy()
            a_stack, d_e4, _, r2 = eval_delta(f"{col}_{fname}", oof_te)
            # also stack raw feat
            a2, d2, _, r22 = eval_delta(f"{col}_{fname}_raw", feat)
            row = {
                "name": f"{col}.{fname}",
                "auc_solo_te": round(a_solo, 5),
                "delta_residTE_e4": round(d_e4, 3),
                "delta_raw_e4": round(d2, 3),
                "R2_residTE": round(r2, 4),
                "R2_raw": round(r22, 4),
                "residue_maxabs": float(np.max(np.abs(residue))),
            }
            t3.append(row)
            if abs(d_e4) > 0.5 or abs(d2) > 0.5:
                str_feats[row["name"]] = oof_te if abs(d_e4) >= abs(d2) else feat
        log(f"  {col}: str_len solo≈TE done; best Δ raw/resid logged")

    t3_sorted = sorted(t3, key=lambda r: -max(abs(r["delta_residTE_e4"]), abs(r["delta_raw_e4"])))
    RES["test3"] = t3_sorted
    log(f"  TEST3 best: {t3_sorted[0]}")

    # ═══════════════════════════════════════════════════════════
    # TEST 4 — non-linear meta stacker
    # ═══════════════════════════════════════════════════════════
    log("TEST4: LightGBM L2 over bank+peers+raw…")
    # top 30 diverse bank OOFs by solo AUC with decorrelation
    scored = []
    for op in sorted(BANK.glob("oof_*.npy")):
        try:
            o = np.load(op).ravel()[:N_TR].astype(float)
            if len(o) != N_TR or not np.isfinite(o).all() or np.ptp(o) < 0.01:
                continue
            scored.append((auc(y, o), op.name, o))
        except Exception:
            continue
    scored.sort(reverse=True)
    selected = []
    sel_arr = []
    for a, name, o in scored:
        if len(selected) >= 30:
            break
        if not sel_arr:
            selected.append(name)
            sel_arr.append(o)
            continue
        # skip if corr > 0.995 with any selected
        corrs = [abs(np.corrcoef(o, s)[0, 1]) for s in sel_arr]
        if max(corrs) < 0.995:
            selected.append(name)
            sel_arr.append(o)
    log(f"  selected {len(selected)} bank OOFs")

    # peer features: top by delta
    top_peers = [r["name"] for r in t2_sorted[:8] if r["name"] in peer_feat_tr]
    # raw SCAN6-ish
    raw_cols = [
        "account_balance_usd",
        "amount_usd",
        "customer_age",
        "merchant_risk_score",
        "velocity_score",
        "txn_count_last_24h",
        "hours_since_last_txn",
    ]
    # encode cats briefly
    cat_cols = ["merchant_category", "auth_method", "channel", "device_type", "card_type"]

    X_parts = [np.column_stack(sel_arr), z_oof.reshape(-1, 1)]
    feat_names = list(selected) + ["wide_lm"]
    for pn in top_peers:
        X_parts.append(peer_feat_tr[pn].reshape(-1, 1))
        feat_names.append(pn)
    for c in raw_cols:
        X_parts.append(tr[c].to_numpy(float).reshape(-1, 1))
        feat_names.append(c)
    # label-encode cats globally (leak-light for L2 diagnostic; OK for research)
    for c in cat_cols:
        codes, _ = pd.factorize(tr[c].astype(str))
        X_parts.append(codes.astype(float).reshape(-1, 1))
        feat_names.append(c)
    # add u_id from G1 train-only + G2
    X_parts.append(u_tr_only.reshape(-1, 1))
    feat_names.append("u_id_G1_train")
    # G2 u from cache
    _, _, u_g2, _ = g_cache["G2_exact_balance"]
    X_parts.append(u_g2[is_tr].reshape(-1, 1))
    feat_names.append("u_id_G2_balance")

    X = np.hstack(X_parts)
    log(f"  meta X shape={X.shape}")

    oof_lgb = np.zeros(N_TR)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (tri, tei) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X[tri], y[tri])
        params = dict(
            objective="binary",
            metric="auc",
            max_depth=4,
            num_leaves=12,
            colsample_bytree=0.3,
            min_child_samples=500,
            learning_rate=0.03,
            verbose=-1,
            n_jobs=4,
            seed=42 + fold,
        )
        model = lgb.train(params, dtr, num_boost_round=150)
        oof_lgb[tei] = model.predict(X[tei])
    a_lgb = auc(y, oof_lgb)
    d_lgb = (a_lgb - base_auc) * 1e4
    RES["test4"] = {
        "auc_lgb": round(a_lgb, 6),
        "auc_wide_lm": round(base_auc, 6),
        "delta_e4": round(d_lgb, 3),
        "n_features": X.shape[1],
        "bank_members": selected,
        "peer_feats": top_peers,
    }
    log(f"  LGB L2 AUC={a_lgb:.6f} vs wide_lm={base_auc:.6f} Δ={d_lgb:+.2f}e-4")

    # ── report ──
    write_report(base_auc, t1, lag_rows, t1_split, t2_sorted, t3_sorted, RES["test4"])
    log(f"DONE NO_SUBMIT={NO_SUBMIT} in {time.time()-T0:.1f}s")


def write_report(base_auc, t1, lags, split, t2, t3, t4):
    def fmt_groups():
        lines = []
        for k, v in t1.items():
            if isinstance(v, dict) and "auc_u_id_grp" in v:
                lines.append(
                    f"| `{k}` | {v.get('n_groups','')} | {v.get('corr(u, size)','')} | "
                    f"{v.get('auc_integer_rank','')} | {v.get('auc_group_size','')} | "
                    f"{v.get('auc_u_id_grp','')} | {v.get('verdict', v)} |"
                )
            else:
                lines.append(f"| `{k}` | | | | | | `{v}` |")
        return "\n".join(lines)

    t2_md = "\n".join(
        f"| `{r['name']}` | {r['auc_solo']:.5f} | {r['auc_stack']:.6f} | {r['delta_e4']:+.2f} | {r.get('R2_bank40', r.get('R2',''))} |"
        for r in t2[:20]
    )
    t3_md = "\n".join(
        f"| `{r['name']}` | {r['auc_solo_te']:.5f} | {r['delta_residTE_e4']:+.2f} | {r['delta_raw_e4']:+.2f} | {r['R2_residTE']:.3f} |"
        for r in t3[:15]
    )
    # G1 conclusion
    g1 = t1.get("G1_train_only_SCAN4_repro", {})
    g1_all = t1.get("G1_card_merch_dow_hour", {})
    if g1:
        if g1.get("auc_group_size", 0) >= g1.get("auc_u_id_grp", 0) - 0.005 and g1.get("auc_group_size", 0) > 0.53:
            g1_concl = (
                f"**SIZE LEAK (A):** integer-rank AUC≈{g1.get('auc_integer_rank')} is largely "
                f"**group_size** (AUC={g1.get('auc_group_size')}); uniform `u` drops to {g1.get('auc_u_id_grp')} "
                f"(corr(u,size)={g1.get('corr(u,size)')}). SCAN4 0.57489 was **not** pure within-group order."
            )
        elif g1.get("auc_u_id_grp", 0) > 0.53:
            g1_concl = (
                f"**TRUE ORDER (B):** uniform `u_id_grp` still AUC={g1.get('auc_u_id_grp')} "
                f"with corr(u,size)={g1.get('corr(u,size)')}≈0 — fraud is non-random in id order inside cohorts."
            )
        else:
            g1_concl = f"Inconclusive / weak after normalization: {g1}"
    else:
        g1_concl = "missing"

    md = f"""# GM V32.1 — Dissect SCAN4 `0.57489` Local-Rank + Graph Unlock

**NO_SUBMIT = True** · wide_lm baseline AUC = **{base_auc:.6f}** · generated {time.strftime('%Y-%m-%d %H:%M')}

## Test 1 — Integer Rank vs Group Size vs Uniform `u_id_grp`

### SCAN4 G1 conclusion
{g1_concl}

### All cohorts
| group | n_groups | corr(u,size) | AUC rank | AUC size | AUC u | verdict |
|-------|---------:|-------------:|---------:|---------:|------:|---------|
{fmt_groups()}

### Train/test split structure
```json
{json.dumps(split, indent=2)}
```

### Neighbor lags (id-sorted within group)
```json
{json.dumps(lags, indent=2)[:4000]}
```

## Test 2 — Entity-Graph Peer Propagation on `wide_lm`

Top lifts (Δ in e-4 = bps×0.1? **ΔAUC × 1e4**):

| feature | solo AUC | stack AUC | Δe-4 | R² vs bank40 |
|---------|---------:|----------:|-----:|-------------:|
{t2_md}

## Test 3 — Raw Float-String Forensics

| feature | solo TE/tree AUC | Δ residTE e-4 | Δ raw e-4 | R² residTE |
|---------|-----------------:|--------------:|----------:|-----------:|
{t3_md}

## Test 4 — Non-Linear L2 LightGBM Meta-Stacker

| metric | value |
|--------|------:|
| wide_lm AUC | {t4['auc_wide_lm']:.6f} |
| LGB L2 OOF AUC | {t4['auc_lgb']:.6f} |
| Δ (e-4) | {t4['delta_e4']:+.3f} |
| n_features | {t4['n_features']} |

Peer feats used: `{t4['peer_feats']}`

## Executive Verdict

1. **SCAN4 0.57489 source:** see G1 conclusion above (size leak vs true order).
2. **Graph unlock on wide_lm:** best Δ = **{t2[0]['delta_e4']:+.2f}e-4** via `{t2[0]['name']}` (need ≫ +10e-4 to explain 0.956+).
3. **Float-string:** best `{t3[0]['name']}` Δraw={t3[0]['delta_raw_e4']:+.2f} / Δresid={t3[0]['delta_residTE_e4']:+.2f} e-4.
4. **Non-linear L2:** {t4['delta_e4']:+.2f}e-4 vs linear wide_lm — {"material" if t4['delta_e4']>5 else "small/null"} for a 0.961 leap alone.
5. **NO_SUBMIT** — zero files submitted; slots kept.

## Log
```
{chr(10).join(LINES[-80:])}
```
"""
    REPORT.write_text(md)
    (OUT / "findings.json").write_text(json.dumps(RES, indent=2, default=str))
    log(f"REPORT -> {REPORT}")


if __name__ == "__main__":
    main()
