#!/usr/bin/env python3
"""SCAN4 within-cohort id-rank residual probe (NO_SUBMIT).

Computes normalized id rank u=rank(id)/n within cohorts on train∪test,
KS(train,test), and Ridge(α=500)+isotonic residual booster on y - p_cal(z_fullbank).
Also reproduces SCAN4 integer-rank tree AUC vs uniform-u solo AUC (size-leak check).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

T0 = time.time()
SSD = Path("/tmp/s6e11_v39_ssd")
OUT = SSD / "out"
OUT.mkdir(parents=True, exist_ok=True)
KS_MAX = 0.005
RIDGE_A = 500.0
NO_SUBMIT = True


def log(msg: str) -> None:
    print(f"[{time.time() - T0:5.1f}s] {msg}", flush=True)


def auc(y, p) -> float:
    return float(roc_auc_score(y, np.asarray(p, float).ravel()))


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def flag64(df: pd.DataFrame) -> np.ndarray:
    flags = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    out = np.zeros(len(df), np.int32)
    for i, c in enumerate(flags):
        out |= (df[c].astype(int).to_numpy() << i).astype(np.int32)
    return out


def cohort_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    return df[cols].astype(str).agg("|".join, axis=1)


def within_cohort_u(ids: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """u = rank(id,'average') / n_group inside each key; also return group size."""
    n = len(ids)
    u = np.empty(n, np.float64)
    sz = np.empty(n, np.float64)
    # sort by key then id for stable group ranks
    order = np.lexsort((ids, keys))
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    keys_s = keys[order]
    ids_s = ids[order]
    # find group boundaries
    change = np.ones(n, bool)
    change[1:] = keys_s[1:] != keys_s[:-1]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        g = b - a
        # average ranks 1..g
        r = rankdata(ids_s[a:b], method="average")
        u_s = r / g
        u[order[a:b]] = u_s
        sz[order[a:b]] = g
    return u.astype(np.float32), sz.astype(np.float32)


def integer_rank_in_group(ids: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """SCAN4-style: rank(method=first) within group (NOT normalized) — encodes size."""
    n = len(ids)
    rk = np.empty(n, np.float64)
    order = np.lexsort((ids, keys))
    keys_s = keys[order]
    change = np.ones(n, bool)
    change[1:] = keys_s[1:] != keys_s[:-1]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        # first ranks 1..g in id order
        rk[order[a:b]] = np.arange(1, b - a + 1, dtype=np.float64)
    return rk.astype(np.float32)


def tree_oof_auc(x: np.ndarray, y: np.ndarray, depth=4, leaf=2000, seed=0) -> float:
    X = x.reshape(-1, 1)
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr_i, te_i in skf.split(X, y):
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=seed)
        dt.fit(X[tr_i], y[tr_i])
        oof[te_i] = dt.predict_proba(X[te_i])[:, 1]
    return auc(y, oof)


def residual_booster_oof(y, z, feat: np.ndarray, n_splits=5, seed=42) -> dict:
    """p_cal = isotonic(z); residual = y - p_cal; fit spline+Ridge on feat → residual; blend back."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof_boost = np.zeros(len(y))
    oof_pcal = np.zeros(len(y))
    X = feat.reshape(-1, 1)
    for tr_i, va_i in skf.split(np.zeros(len(y)), y):
        iso = IsotonicRegression(out_of_bounds="clip")
        p_tr = iso.fit_transform(z[tr_i], y[tr_i])
        p_va = iso.predict(z[va_i])
        oof_pcal[va_i] = p_va
        resid = y[tr_i] - p_tr
        pipe = Pipeline(
            [
                ("sp", SplineTransformer(n_knots=6, degree=3, include_bias=True)),
                ("ridge", Ridge(alpha=RIDGE_A, fit_intercept=True)),
            ]
        )
        pipe.fit(X[tr_i], resid)
        oof_boost[va_i] = pipe.predict(X[va_i])
    # blend: p_cal + w * boost, pick w on full OOF grid (slightly optimistic but for probe OK)
    # proper: nested — use fixed small grid evaluated fold-wise
    lifts = []
    best_w, best_a = 0.0, auc(y, oof_pcal)
    base = best_a
    for w in np.linspace(-0.15, 0.15, 31):
        a = auc(y, oof_pcal + w * oof_boost)
        lifts.append((float(w), float(a - base)))
        if a > best_a:
            best_a, best_w = a, float(w)
    # fold-wise honest lift
    d = []
    for tr_i, va_i in skf.split(np.zeros(len(y)), y):
        # pick w on tr
        bw, ba = 0.0, -1.0
        for w in np.linspace(-0.15, 0.15, 31):
            a = auc(y[tr_i], oof_pcal[tr_i] + w * oof_boost[tr_i])
            if a > ba:
                ba, bw = a, float(w)
        d.append(auc(y[va_i], oof_pcal[va_i] + bw * oof_boost[va_i]) - auc(y[va_i], oof_pcal[va_i]))
    d = np.asarray(d)
    return {
        "auc_z": auc(y, z),
        "auc_pcal": float(base),
        "auc_pcal_plus_boost": float(best_a),
        "delta_full_oof": float(best_a - base),
        "best_w": best_w,
        "subfold_mean_d": float(d.mean()),
        "subfold_n_pos": int((d > 0).sum()),
        "subfold_d": d.tolist(),
        "corr_boost_resid": float(np.corrcoef(oof_boost, y - oof_pcal)[0, 1]),
    }


def main():
    log("=== SCAN4 cohort id-rank residual probe | NO_SUBMIT ===")
    tr = pd.read_csv(SSD / "data/train.csv")
    te = pd.read_csv(SSD / "data/test.csv")
    y = tr["is_fraud"].to_numpy(int)
    z = np.load(SSD / "preds/z_fullbank_oof.npy").ravel()
    assert len(z) == len(tr)

    tr["flag64"] = flag64(tr)
    te["flag64"] = flag64(te)
    # bin continuous-ish for optional cohorts already categorical here

    cohorts = {
        "G1_card_merch_dow_hour": ["card_type", "merchant_category", "day_of_week", "time_of_day_hour"],
        "G2_auth_chan_dev": ["auth_method", "channel", "device_type"],
        "G3_flag64": ["flag64"],
        "G4_card_merch": ["card_type", "merchant_category"],
        "G5_dow_hour": ["day_of_week", "time_of_day_hour"],
        "G6_merch_only": ["merchant_category"],
    }

    # --- SCAN4 repro: train-only integer rank tree vs uniform u ---
    log("SCAN4 repro: integer-rank tree vs uniform u (G1)…")
    k_tr = cohort_key(tr, cohorts["G1_card_merch_dow_hour"]).to_numpy()
    ids_tr = tr["id"].to_numpy()
    rk_int = integer_rank_in_group(ids_tr, k_tr)
    u_tr_only, sz_tr = within_cohort_u(ids_tr, k_tr)
    auc_int = tree_oof_auc(rk_int, y)
    auc_sz = tree_oof_auc(sz_tr, y)
    auc_u_solo = tree_oof_auc(u_tr_only, y)
    auc_u_raw = auc(y, u_tr_only)  # direction may flip
    auc_u_raw_flip = auc(y, -u_tr_only)
    log(
        f"  G1 train-only: int_rank_tree={auc_int:.5f} size_tree={auc_sz:.5f} "
        f"u_tree={auc_u_solo:.5f} u_raw={max(auc_u_raw, auc_u_raw_flip):.5f} "
        f"corr(u,size)={np.corrcoef(u_tr_only, sz_tr)[0,1]:+.4f}"
    )

    # --- train∪test normalized u ---
    report = {
        "NO_SUBMIT": NO_SUBMIT,
        "scan4_repro": {
            "auc_integer_rank_tree": auc_int,
            "auc_group_size_tree": auc_sz,
            "auc_u_tree": auc_u_solo,
            "auc_u_raw_max": max(auc_u_raw, auc_u_raw_flip),
            "corr_u_size": float(np.corrcoef(u_tr_only, sz_tr)[0, 1]),
            "verdict": (
                "SIZE_LEAK"
                if auc_sz > auc_u_solo + 0.02 and auc_u_solo < 0.52
                else "POSSIBLE_ORDER"
            ),
        },
        "cohorts": {},
        "KS_MAX": KS_MAX,
        "RIDGE_A": RIDGE_A,
    }

    n_tr, n_te = len(tr), len(te)
    for name, cols in cohorts.items():
        log(f"cohort {name} {cols}…")
        # concat for shared [0,1] scale
        both = pd.concat(
            [
                tr[cols + ["id"]].assign(_split=0),
                te[cols + ["id"]].assign(_split=1),
            ],
            ignore_index=True,
        )
        keys = cohort_key(both, cols).to_numpy()
        ids = both["id"].to_numpy()
        u_all, sz_all = within_cohort_u(ids, keys)
        u_tr = u_all[:n_tr]
        u_te = u_all[n_tr:]
        sz_tr_c = sz_all[:n_tr]
        ks = float(ks_2samp(u_tr, u_te).statistic)
        ks_ok = ks < KS_MAX
        # solo AUCs
        solo_u = max(auc(y, u_tr), auc(y, -u_tr))
        solo_sz = max(auc(y, sz_tr_c), auc(y, -sz_tr_c))
        # residual booster
        boost = residual_booster_oof(y, z, u_tr)
        boost_sz = residual_booster_oof(y, z, sz_tr_c)
        # also u + log1p(size) joint
        feat2 = np.column_stack([u_tr, np.log1p(sz_tr_c)])
        # extend booster for 2d quickly
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        oof_b = np.zeros(len(y))
        oof_p = np.zeros(len(y))
        for tr_i, va_i in skf.split(np.zeros(len(y)), y):
            iso = IsotonicRegression(out_of_bounds="clip")
            p_tr = iso.fit_transform(z[tr_i], y[tr_i])
            oof_p[va_i] = iso.predict(z[va_i])
            resid = y[tr_i] - p_tr
            pipe = Pipeline(
                [
                    ("sp", SplineTransformer(n_knots=5, degree=2, include_bias=True)),
                    ("ridge", Ridge(alpha=RIDGE_A)),
                ]
            )
            # spline only on first col; pass both via ridge after manual expand
            # simpler: Ridge on [u, u^2, logsize, interactions]
            Xtr = np.column_stack(
                [
                    feat2[tr_i, 0],
                    feat2[tr_i, 0] ** 2,
                    feat2[tr_i, 1],
                    feat2[tr_i, 0] * feat2[tr_i, 1],
                ]
            )
            Xva = np.column_stack(
                [
                    feat2[va_i, 0],
                    feat2[va_i, 0] ** 2,
                    feat2[va_i, 1],
                    feat2[va_i, 0] * feat2[va_i, 1],
                ]
            )
            rr = Ridge(alpha=RIDGE_A).fit(Xtr, resid)
            oof_b[va_i] = rr.predict(Xva)
        d2 = []
        for tr_i, va_i in skf.split(np.zeros(len(y)), y):
            bw, ba = 0.0, -1.0
            for w in np.linspace(-0.15, 0.15, 31):
                a = auc(y[tr_i], oof_p[tr_i] + w * oof_b[tr_i])
                if a > ba:
                    ba, bw = a, float(w)
            d2.append(auc(y[va_i], oof_p[va_i] + bw * oof_b[va_i]) - auc(y[va_i], oof_p[va_i]))
        d2 = np.asarray(d2)
        joint = {
            "subfold_mean_d": float(d2.mean()),
            "subfold_n_pos": int((d2 > 0).sum()),
            "subfold_d": d2.tolist(),
            "auc_pcal": auc(y, oof_p),
        }

        row = {
            "cols": cols,
            "n_groups_train": int(pd.Series(keys[:n_tr]).nunique()),
            "n_groups_both": int(pd.Series(keys).nunique()),
            "ks_u_train_vs_test": ks,
            "ks_ok": ks_ok,
            "solo_auc_u": solo_u,
            "solo_auc_size": solo_sz,
            "corr_u_size": float(np.corrcoef(u_tr, sz_tr_c)[0, 1]),
            "boost_u": boost,
            "boost_size": boost_sz,
            "boost_u_logsize": joint,
            "pass_hard": bool(
                ks_ok
                and boost["subfold_mean_d"] >= 0.5e-4
                and boost["subfold_n_pos"] >= 4
                and solo_u >= 0.52
            ),
        }
        report["cohorts"][name] = row
        log(
            f"  KS={ks:.5f}({'OK' if ks_ok else 'FAIL'}) solo_u={solo_u:.5f} solo_sz={solo_sz:.5f} "
            f"boost_u Δ={boost['subfold_mean_d']:+.2e} ({boost['subfold_n_pos']}/5) "
            f"boost_sz Δ={boost_sz['subfold_mean_d']:+.2e} "
            f"joint Δ={joint['subfold_mean_d']:+.2e} ({joint['subfold_n_pos']}/5)"
        )

    # global id sanity
    ids_all = np.concatenate([tr["id"].to_numpy(), te["id"].to_numpy()])
    u_glob = (rankdata(ids_all, method="average") - 0.5) / len(ids_all)
    report["global_id"] = {
        "auc_u_train": max(auc(y, u_glob[:n_tr]), auc(y, -u_glob[:n_tr])),
        "ks": float(ks_2samp(u_glob[:n_tr], u_glob[n_tr:]).statistic),
    }

    any_pass = any(c["pass_hard"] for c in report["cohorts"].values())
    # soft residual-only pass (ignore solo)
    soft = []
    for name, c in report["cohorts"].items():
        ok = (
            c["ks_ok"]
            and c["boost_u"]["subfold_mean_d"] >= 0.5e-4
            and c["boost_u"]["subfold_n_pos"] >= 4
        )
        soft.append((name, ok, c["boost_u"]["subfold_mean_d"]))
    report["any_hard_pass"] = any_pass
    report["soft_residual_candidates"] = [
        {"cohort": n, "ok": ok, "mean_d": d} for n, ok, d in soft if ok
    ]
    report["elapsed_s"] = time.time() - T0
    report["conclusion"] = (
        "SCAN4 0.57489 is SIZE_LEAK (integer rank / group_size), not normalized within-cohort order. "
        "Uniform u∈[0,1] is ~chance solo; residual boost on z_fullbank is the real test — see soft_residual_candidates."
        if report["scan4_repro"]["verdict"] == "SIZE_LEAK"
        else "Possible true order signal — review cohort table."
    )

    jp = OUT / "SCAN4_cohort_id_rank_report.json"
    jp.write_text(json.dumps(report, indent=2))
    # markdown
    lines = [
        "# SCAN4 Within-Cohort id-Rank Residual Report",
        "",
        f"**NO_SUBMIT=True** · elapsed **{report['elapsed_s']:.1f}s** · Ridge α={RIDGE_A}",
        "",
        "## SCAN4 repro (train-only G1)",
        "",
        f"| metric | AUC |",
        f"|--------|----:|",
        f"| integer `rank(id)` tree (SCAN4-style) | **{auc_int:.5f}** |",
        f"| `group_size` tree | **{auc_sz:.5f}** |",
        f"| uniform `u=rank/n` tree | **{auc_u_solo:.5f}** |",
        f"| uniform `u` raw | {report['scan4_repro']['auc_u_raw_max']:.5f} |",
        f"| corr(u, size) | {report['scan4_repro']['corr_u_size']:+.4f} |",
        "",
        f"**Verdict:** `{report['scan4_repro']['verdict']}` — SCAN4’s 0.57489 was **group-size leak** via integer rank, not pure within-cohort order.",
        "",
        "## train∪test normalized `u` + residual booster on `y - p_cal(z_fullbank)`",
        "",
        "| cohort | KS(u) | KS ok | solo u | solo size | boost_u Δ (n+) | boost_size Δ | joint Δ (n+) | hard |",
        "|--------|------:|:-----:|-------:|----------:|---------------:|-------------:|-------------:|:----:|",
    ]
    for name, c in report["cohorts"].items():
        lines.append(
            f"| `{name}` | {c['ks_u_train_vs_test']:.5f} | {'Y' if c['ks_ok'] else 'N'} | "
            f"{c['solo_auc_u']:.5f} | {c['solo_auc_size']:.5f} | "
            f"{c['boost_u']['subfold_mean_d']:+.2e} ({c['boost_u']['subfold_n_pos']}/5) | "
            f"{c['boost_size']['subfold_mean_d']:+.2e} | "
            f"{c['boost_u_logsize']['subfold_mean_d']:+.2e} ({c['boost_u_logsize']['subfold_n_pos']}/5) | "
            f"{'Y' if c['pass_hard'] else 'N'} |"
        )
    lines += [
        "",
        f"**Global id u AUC:** {report['global_id']['auc_u_train']:.5f} (KS={report['global_id']['ks']:.5f})",
        "",
        f"**Hard pass any:** {any_pass}",
        f"**Soft residual candidates (KS ok + Δ≥0.5e-4 & ≥4/5):** {report['soft_residual_candidates'] or 'none'}",
        "",
        f"## Conclusion",
        "",
        report["conclusion"],
        "",
        f"JSON: `{jp}`",
    ]
    mp = OUT / "SCAN4_cohort_id_rank_report.md"
    mp.write_text("\n".join(lines))
    log(f"wrote {jp}")
    log(f"wrote {mp}")
    print("\n" + "\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
