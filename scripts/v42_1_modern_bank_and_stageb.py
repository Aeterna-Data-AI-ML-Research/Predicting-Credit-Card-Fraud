#!/usr/bin/env python3
"""V42.1 — Modern OOF bank ingestion + softened Stage-B redesign.

STRICT NO_SUBMIT=True. Zero submissions.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

NO_SUBMIT = True
assert NO_SUBMIT
T0 = time.time()
N_TR, N_TE = 699_960, 299_984
DEADLINE = datetime(2026, 11, 30, 23, 59, tzinfo=timezone.utc)

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
SSD = Path("/tmp/s6e11_v39_ssd")
LOG = ROOT / "logs/v42"
OUT = ROOT / "out/v42_1_modern"
PRED = ROOT / "preds"
GPW = ROOT / "external/banks/goodpjw2008_s6e11-oof-bank"
BUSY = ROOT / "external/v41_nov23_frontier/busyaprime_52"
AMAN = ROOT / "external/banks_v31/s6e11-rank1-q17-meta-bundle/rank1_q17_streams.npz"
DATA = ROOT / "data" if (ROOT / "data/train.csv").exists() else SSD / "data"
FOLDS_P = SSD / "folds/folds.csv" if (SSD / "folds/folds.csv").exists() else ROOT / "folds/folds.csv"
FINAL = ROOT / "out/v41_final_10slots"

QUARANTINE = (
    "zanch", "v10_ridge", "v10_cb_d4", "v10_lgb", "idf_p3", "clean035",
    "uid_catboost", "fintfm", "fin_tfm", "gb_soft_bag3", "soft_diff",  # soft_diff admitted only as isolated prop below
)
# soft_diff is quarantined from PUBLIC bank but allowed as named proprietary
PRIVATE_ONLY = {"soft_diff_orth_LM", "gpt2_med_4ep", "l7_s2026", "xgb_llr", "delta1_v4", "U5_gini10_llr", "E2_heavy"}

FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]


def log(msg: str):
    print(f"[{time.time()-T0:5.1f}s] {msg}", flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v):
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12))


def L_fold(v, folds):
    v = np.asarray(v, float).ravel()
    out = np.zeros_like(v, dtype=np.float64)
    for k in np.unique(folds):
        m = folds == k
        out[m] = L(v[m])
    return out


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def one_mrho(a, b) -> float:
    return 1.0 - float(spearmanr(a, b).statistic)


def cos_full(a, b):
    a, b = unit(a), unit(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


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


def is_quar(s: str) -> bool:
    s = s.lower()
    return any(q in s for q in QUARANTINE)


def method_b_lock(oof, test, folds):
    """Per-fold L(U) then unit-match (disp→1, KS↓)."""
    o = unit(L_fold(oof, folds))
    t = unit(L(test))
    t = t * (float(o.std()) / (float(t.std()) + 1e-12))
    disp = float(np.std(t) / (np.std(o) + 1e-12))
    ks = float(ks_2samp(o, t).statistic)
    return o, t, disp, ks


def fold_delta(y, p_new, p_base, folds):
    d, pos = [], 0
    for k in range(5):
        m = folds == k
        da = (roc_auc_score(y[m], p_new[m]) - roc_auc_score(y[m], p_base[m])) * 1e5
        d.append(float(da))
        pos += int(da > 0)
    return float(np.mean(d)), pos, d


def sf25(y, a, b, folds) -> int:
    wins = 0
    for k in range(5):
        m = folds == k
        yk, ak, bk = y[m], a[m], b[m]
        skf = StratifiedKFold(5, shuffle=True, random_state=42 + k)
        for _, va in skf.split(np.arange(len(yk)), yk):
            if roc_auc_score(yk[va], ak[va]) > roc_auc_score(yk[va], bk[va]):
                wins += 1
    return wins


def t25(fd):
    x = np.asarray(fd, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-12))


def load_gpt2(folds):
    """Fix Eng3 shape miss: per-fold VA logits → full OOF; mean test bags."""
    gdir = PRED / "gpt2_med_4ep"
    go = np.zeros(N_TR, dtype=np.float64)
    gt = []
    meta = {"oof_raw": [], "test_raw": []}
    for k in range(5):
        a = np.load(gdir / f"oof_{k}.npy")  # (n_va, seeds, 2)
        b = np.load(gdir / f"test_{k}.npy")  # (N_TE, seeds, 2)
        meta["oof_raw"].append(list(a.shape))
        meta["test_raw"].append(list(b.shape))
        go[folds == k] = (a[:, :, 1] - a[:, :, 0]).mean(1)
        gt.append((b[:, :, 1] - b[:, :, 0]).mean(1))
    te = np.mean(gt, 0)
    assert go.shape == (N_TR,) and te.shape == (N_TE,), (go.shape, te.shape)
    return go, te, meta


def load_l7():
    lo = np.load(PRED / "l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    lt = np.load(PRED / "l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    o = (lo[:, :, 1] - lo[:, :, 0]).mean(1) if lo.ndim == 3 else lo.ravel()
    t = ((lt[..., 1] - lt[..., 0]).mean(2)).mean(0) if lt.ndim == 4 else lt.ravel()
    return o.ravel(), t.ravel()


def try_load_pair(name, o_path, t_path):
    if not (Path(o_path).exists() and Path(t_path).exists()):
        return None
    o = np.load(o_path).ravel() if str(o_path).endswith(".npy") else None
    t = np.load(t_path).ravel() if str(t_path).endswith(".npy") else None
    if o is None:
        return None
    if o.shape[0] != N_TR or t.shape[0] != N_TE:
        return {"name": name, "status": "bad_shape", "oof_shape": list(o.shape), "test_shape": list(t.shape)}
    return {"name": name, "o": o, "t": t}


def select_diverse(cols_o, cols_t, y, max_n=36):
    """Keep top-AUC streams with pairwise |ρ|<0.995 on a 20k subsample."""
    rng = np.random.default_rng(42)
    sub = rng.choice(len(y), size=min(20_000, len(y)), replace=False)
    scored = [(float(roc_auc_score(y, o)), i) for i, o in enumerate(cols_o)]
    scored.sort(reverse=True)
    keep = []
    kept_sub = []
    for _, i in scored:
        if len(keep) >= max_n:
            break
        si = cols_o[i][sub]
        if all(abs(float(np.corrcoef(si, kj)[0, 1])) < 0.995 for kj in kept_sub):
            keep.append(i)
            kept_sub.append(si)
    return [cols_o[i] for i in keep], [cols_t[i] for i in keep], keep


def nested_logreg_stack(y, folds, cols_o, cols_t, C_grid=(0.03, 0.1, 0.3, 1.0)):
    """Nested 5-fold LogisticRegression on L_fold columns (step6b recipe)."""
    Fo = np.column_stack([L_fold(c, folds) for c in cols_o])
    Ft = np.column_stack([L(c) for c in cols_t])
    oof = np.zeros(N_TR)
    te = np.zeros(N_TE)
    for f in range(5):
        tr, va = folds != f, folds == f
        best_c, best_s = 0.1, -1.0
        inner = folds[tr]
        for C in C_grid:
            scores = []
            for u in np.unique(inner):
                it, iv = inner != u, inner == u
                clf = LogisticRegression(C=C, solver="lbfgs", max_iter=250, tol=1e-3)
                clf.fit(Fo[tr][it], y[tr][it])
                scores.append(roc_auc_score(y[tr][iv], clf.predict_proba(Fo[tr][iv])[:, 1]))
            m = float(np.mean(scores))
            if m > best_s:
                best_s, best_c = m, C
        clf = LogisticRegression(C=best_c, solver="lbfgs", max_iter=400)
        clf.fit(Fo[tr], y[tr])
        oof[va] = clf.predict_proba(Fo[va])[:, 1]
        te += clf.predict_proba(Ft)[:, 1] / 5
        log(f"    nest fold {f}: C={best_c} inner={best_s:.5f} n_cols={Fo.shape[1]}")
    return U(oof), U(te)


def blend_with_anchor(y, nest_o, nest_t, anchor_o, anchor_t, tag=""):
    """Pick best blend of nest vs locked anchor (z_pub_only / part12 style)."""
    cands = {
        f"{tag}nest": (nest_o, nest_t),
        f"{tag}anchor": (anchor_o, anchor_t),
        f"{tag}0.85a+0.15n": (
            U(0.85 * L(U(anchor_o)) + 0.15 * L(U(nest_o))),
            U(0.85 * L(U(anchor_t)) + 0.15 * L(U(nest_t))),
        ),
        f"{tag}0.70a+0.30n": (
            U(0.70 * L(U(anchor_o)) + 0.30 * L(U(nest_o))),
            U(0.70 * L(U(anchor_t)) + 0.30 * L(U(nest_t))),
        ),
        f"{tag}rankmean": (
            U(0.5 * U(anchor_o) + 0.5 * U(nest_o)),
            U(0.5 * U(anchor_t) + 0.5 * U(nest_t)),
        ),
    }
    # orth nest residual on anchor
    u = unit(L(U(nest_o)))
    bb = unit(L(U(anchor_o)))
    u = unit(u - np.dot(u, bb) * bb)
    ut = unit(L(U(nest_t)))
    bbt = unit(L(U(anchor_t)))
    ut = unit(ut - np.dot(ut, bbt) * bbt)
    for a in (0.01, 0.02, 0.04, 0.08):
        cands[f"{tag}a+{a}orth_n"] = (
            U(L(U(anchor_o)) + a * u),
            U(L(U(anchor_t)) + a * ut),
        )
    best_name, best_a, best_o, best_t = "anchor", -1.0, anchor_o, anchor_t
    for k, (oo, tt) in cands.items():
        a = float(roc_auc_score(y, oo))
        log(f"  blend cand {k}: {a:.6f}")
        if a > best_a:
            best_name, best_a, best_o, best_t = k, a, oo, tt
    return U(best_o), U(best_t), best_name, best_a


def gs_orth(v, bases):
    v = unit(v)
    for b in bases:
        b = unit(b)
        v = unit(v - np.dot(v, b) * b)
    return v


def main():
    log("=== V42.1 modern bank + softened StageB | NO_SUBMIT=True ===")
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    days_left = (DEADLINE - datetime.now(timezone.utc)).total_seconds() / 86400
    try:
        import torch
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        cuda = bool(torch.cuda.is_available())
    except Exception:
        mps, cuda = False, False
    env = {
        "deadline_utc": DEADLINE.isoformat(),
        "days_left": round(days_left, 2),
        "mps": mps, "cuda": cuda,
        "runpodctl": bool(Path("/Users/mz/.local/bin/runpodctl").exists()),
        "note": "GPU for new foundation LLM: local MPS unavailable in this env; runpodctl present for cloud pod.",
    }
    log(f"deadline days_left≈{days_left:.1f} mps={mps} cuda={cuda}")

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS_P)["fold"].to_numpy(int)
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    ids = te["id"].to_numpy()
    Zg_tr, Zg_te = z_goodpjw(tr), z_goodpjw(te)

    z_pub_o = np.load(SSD / "preds/z_pub_only_oof.npy").ravel()
    z_pub_t = np.load(SSD / "preds/z_pub_only_test.npy").ravel()
    z_pub_zr = apply_zr(z_pub_o, Zg_tr, z_pub_o)
    base_auc = float(roc_auc_score(y, z_pub_zr))
    assert abs(base_auc - 0.952614) < 5e-5, base_auc
    log(f"z_pub_only+Zr={base_auc:.6f}")

    ab = pd.read_csv(FINAL / "SLOT01_pub_Step_AB_095569.csv")["is_fraud"].to_numpy(float)
    s02 = pd.read_csv(FINAL / "SLOT02_AB_clip985_areeb_a0p32.csv")["is_fraud"].to_numpy(float)
    s06 = pd.read_csv(FINAL / "SLOT06_AB_C3_harvest_p4_clip985_0p28e4.csv")["is_fraud"].to_numpy(float)
    ray_s02, ray_s06 = L(s02) - L(ab), L(s06) - L(ab)

    # ── inventory ──
    inventory = []
    pub_streams = []  # (name, o, t)
    prop_streams = []

    def admit(name, o, t, family, force_prop=False, residual=False):
        o, t = np.asarray(o, float).ravel(), np.asarray(t, float).ravel()
        row = {
            "asset_name": name, "family": family,
            "oof_shape": list(o.shape), "test_shape": list(t.shape),
        }
        if o.shape != (N_TR,) or t.shape != (N_TE,):
            row.update({"status": "BAD_SHAPE", "solo_oof_auc": None, "disp": None, "KS": None})
            inventory.append(row)
            return False
        if not (np.isfinite(o).all() and np.isfinite(t).all()):
            row["status"] = "NONFINITE"
            inventory.append(row)
            return False
        auc = float(roc_auc_score(y, o))
        # auto-orient inverted residual / regime columns
        if auc < 0.5:
            o, t = -o, -t
            auc = float(roc_auc_score(y, o))
            row["flipped"] = True
        oo, tt, disp, ks = method_b_lock(o, t, folds)
        row.update({
            "solo_oof_auc": auc, "disp": disp, "KS": ks,
            "status": "OK",
            "in_z_pub_only": family == "public" and (not force_prop),
        })
        # gates: residual/orth streams are weak solo by design
        if auc >= 0.9550:
            row["status"] = "REJECT_AUC>=0.9550"
            inventory.append(row)
            return False
        floor = 0.52 if (force_prop and residual) else (0.55 if force_prop else 0.90)
        if auc < floor:
            row["status"] = f"REJECT_AUC<{floor}"
            inventory.append(row)
            return False
        if not (0.98 <= disp <= 1.02) or ks >= 0.005:
            if ks >= 0.05:
                row["status"] = "REJECT_KS"
                inventory.append(row)
                return False
            row["status"] = "OK_LOCKED"
        if is_quar(name) and not force_prop:
            row["status"] = "QUARANTINE"
            inventory.append(row)
            return False
        inventory.append(row)
        # store oriented ranks for stacking (L_fold applied inside nest)
        if family == "public" and not force_prop:
            pub_streams.append((name, U(o), U(t)))
        else:
            prop_streams.append((name, U(o), U(t)))
        return True

    # gpt2 fix
    gpt_o, gpt_t, gpt_meta = load_gpt2(folds)
    log(f"gpt2 shape-fix: oof={gpt_o.shape} test={gpt_t.shape} raw={gpt_meta}")
    admit("gpt2_med_4ep_5f", gpt_o, gpt_t, "proprietary", force_prop=True)

    l7o, l7t = load_l7()
    admit("l7_s2026_5f", l7o, l7t, "proprietary", force_prop=True)

    admit("B_xgb_llr",
          np.load(PRED / "paradigm_B/xgb_llr_oof.npy"),
          np.load(PRED / "paradigm_B/xgb_llr_test.npy"),
          "proprietary", force_prop=True)
    admit("delta1_v4",
          np.load(PRED / "gm_v4_polish/d1_v4_oof.npy"),
          np.load(PRED / "gm_v4_polish/d1_v4_test.npy"),
          "proprietary", force_prop=True, residual=True)
    # soft_diff isolated only (exclude zanch/idf_p3/clean035)
    admit("soft_diff_orth_LM",
          np.load(PRED / "v10_transductive_bulsah/soft_diff_orth_LM_oof.npy"),
          np.load(PRED / "v10_transductive_bulsah/soft_diff_orth_LM_test.npy"),
          "proprietary", force_prop=True, residual=True)

    for nm in ("P4_L4.2_pure5f", "P4_L4.5_pure5f", "P4_L4.3_pure5f"):
        p = PRED / "matrix_v2"
        admit(nm, np.load(p / f"{nm}_oof.npy"), np.load(p / f"{nm}_test.npy"),
              "proprietary", force_prop=True, residual=True)

    # L4.6C top12
    if (PRED / "gm_v9_1_hardened/l46c_top12_5f_oof.npy").exists():
        admit("L4.6C_top12_5f",
              np.load(PRED / "gm_v9_1_hardened/l46c_top12_5f_oof.npy"),
              np.load(PRED / "gm_v9_1_hardened/l46c_top12_5f_test.npy"),
              "proprietary", force_prop=True, residual=True)

    # E2 heavy / U5 — best available proxies
    e2 = try_load_pair("E2_heavy_proxy_v34_3",
                       PRED / "v34_3_principal/E2_oof.npy",
                       PRED / "v34_3_principal/E2_test.npy")
    if e2 and "o" in e2:
        admit("E2_heavy_proxy_v34_3", e2["o"], e2["t"], "proprietary", force_prop=True)
    else:
        inventory.append({"asset_name": "E2_heavy_ridge", "status": "MISSING", "note": "no exact E2_heavy_ridge; v34_3 E2 used if present"})

    inventory.append({"asset_name": "U5_gini10_llr", "status": "MISSING", "note": "not found as named array on disk"})
    inventory.append({"asset_name": "G26_5f", "status": "MISSING", "note": "not in local cache / v43 pinned folder as 5f npy"})
    inventory.append({"asset_name": "MS_5f", "status": "MISSING", "note": "not found as named 5f MS array; using Aman megayak/masaya streams"})

    # Public: gpw70
    man = pd.read_csv(GPW / "manifest.csv")
    n_pub = 0
    for _, r in man.iterrows():
        n = str(r["name"])
        if is_quar(n):
            continue
        op, tp = GPW / f"oof_{n}.npy", GPW / f"test_{n}.npy"
        if op.exists() and tp.exists():
            if admit(f"gpw70:{n}", np.load(op), np.load(tp), "public"):
                n_pub += 1
    log(f"gpw70 admitted={n_pub}")

    # busy52
    if (BUSY / "manifest.csv").exists():
        bman = pd.read_csv(BUSY / "manifest.csv")
        Bo, Bt = np.load(BUSY / "oof.npy"), np.load(BUSY / "test.npy")
        for i, r in bman.iterrows():
            n = str(r["name"])
            if is_quar(n):
                continue
            admit(f"busy52:{n}", Bo[:, i], Bt[:, i], "public")

    # Aman streams
    if AMAN.exists():
        Z = np.load(AMAN, allow_pickle=False)
        for k in Z.files:
            if not k.startswith("oof_"):
                continue
            base = k[4:]
            tk = f"test_{base}"
            if tk not in Z.files:
                continue
            if is_quar(base):
                continue
            admit(f"aman:{base}", Z[k], Z[tk], "public")
        log("aman streams loaded")

    # Part12
    p12 = ROOT / "external/nov08_part12"
    if (p12 / "lm_oof.npy").exists():
        admit("part12:lm", np.load(p12 / "lm_oof.npy"), np.load(p12 / "lm_test.npy"), "public")

    # board-model pinned (goodpjw v43 inputs) — RealMLP / TabM / etc.
    pinned = ROOT / "external/banks_v31/s6e11-board-model-pinned-inputs"
    if pinned.exists():
        for op in sorted(pinned.glob("oof_*.npy")):
            base = op.name[len("oof_"):-4]
            tp = pinned / f"test_{base}.npy"
            if tp.exists():
                admit(f"board:{base}", np.load(op), np.load(tp), "public")

    log(f"inventory OK pub={len(pub_streams)} prop={len(prop_streams)} rows={len(inventory)}")

    # ── z_modern_pub / z_modern_all (anchor = z_pub_only) ──
    log("build z_modern_pub")
    assert len(pub_streams) >= 5, len(pub_streams)
    # Prefer NEW modern public (aman/board/realmlp) + top gpw; always keep diversity
    priority = []
    rest = []
    for i, (n, o, t) in enumerate(pub_streams):
        nl = n.lower()
        if any(k in nl for k in ("aman:", "board:", "realmlp", "yekenot", "megayak", "masaya", "tabpfn", "part12")):
            priority.append(i)
        else:
            rest.append(i)
    ordered = priority + rest
    pub_o_raw = [pub_streams[i][1] for i in ordered]
    pub_t_raw = [pub_streams[i][2] for i in ordered]
    # force-include top priority then diversity fill
    pub_o, pub_t, keep_pub = select_diverse(pub_o_raw, pub_t_raw, y, max_n=28)
    log(f"public diverse keep={len(keep_pub)}/{len(pub_streams)} (priority-first)")
    nest_pub_o, nest_pub_t = nested_logreg_stack(y, folds, pub_o, pub_t)
    z_mod_pub_o, z_mod_pub_t, pub_blend, pub_raw_auc = blend_with_anchor(
        y, nest_pub_o, nest_pub_t, z_pub_o, z_pub_t, tag="pub_"
    )
    z_mod_pub_zr = apply_zr(z_mod_pub_o, Zg_tr, z_mod_pub_o)
    auc_pub = float(roc_auc_score(y, z_mod_pub_zr))
    d_pub, pos_pub, fd_pub = fold_delta(y, z_mod_pub_zr, z_pub_zr, folds)
    log(f"z_modern_pub+Zr={auc_pub:.6f} blend={pub_blend} Δe5={d_pub:.2f} pos={pos_pub}/5")

    log("build z_modern_all")
    # strong proprietary ONLY in L1 (residuals go to StageB)
    strong_prop = []
    residual_prop = []
    for n, o, t in prop_streams:
        if any(k in n for k in ("gpt2", "l7_s2026", "B_xgb", "E2_heavy")):
            strong_prop.append((n, o, t))
        else:
            residual_prop.append((n, o, t))
    all_o = pub_o + [o for _, o, _ in strong_prop]
    all_t = pub_t + [t for _, _, t in strong_prop]
    if len(all_o) > 36:
        all_o, all_t, _ = select_diverse(all_o, all_t, y, max_n=36)
    nest_all_o, nest_all_t = nested_logreg_stack(y, folds, all_o, all_t)
    z_mod_all_o, z_mod_all_t, all_blend, all_raw_auc = blend_with_anchor(
        y, nest_all_o, nest_all_t, z_mod_pub_o, z_mod_pub_t, tag="all_"
    )
    z_mod_all_zr = apply_zr(z_mod_all_o, Zg_tr, z_mod_all_o)
    auc_all = float(roc_auc_score(y, z_mod_all_zr))
    d_all, pos_all, fd_all = fold_delta(y, z_mod_all_zr, z_pub_zr, folds)
    rho_ab_pub = one_mrho(z_mod_pub_t, ab) * 1e4
    rho_ab_all = one_mrho(z_mod_all_t, ab) * 1e4
    log(f"z_modern_all+Zr={auc_all:.6f} blend={all_blend} Δe5={d_all:.2f} pos={pos_all}/5 1-ρvsAB={rho_ab_all:.2f}e-4")

    np.save(OUT / "z_modern_pub_oof.npy", z_mod_pub_o)
    np.save(OUT / "z_modern_pub_test.npy", z_mod_pub_t)
    np.save(OUT / "z_modern_all_oof.npy", z_mod_all_o)
    np.save(OUT / "z_modern_all_test.npy", z_mod_all_t)

    # ── StageB v42.1 ──
    log("StageB v42.1 softened init + GS orth")
    # continuous logit columns for PCHIP/ECDF (Method-B); discrete L46 left raw if multi-dim — here scalar
    cont_names = []
    Xo_list, Xt_list = [], []
    for name, o, t in prop_streams + [(n, o, t) for n, o, t in pub_streams if any(k in n for k in ("realmlp", "megayak", "masaya", "gpt2", "lm7", "tabpfn"))]:
        # continuous model logits only
        if any(k in name for k in ("flag", "count", "bitmask", "L4.6C")):
            # leave discrete untouched — admit raw ranks
            Xo_list.append(unit(o)); Xt_list.append(unit(t))
            cont_names.append(name + ":raw")
            continue
        oo, tt, disp, ks = method_b_lock(o, t, folds)
        Xo_list.append(oo); Xt_list.append(tt)
        cont_names.append(name)

    # build LM bag / RealMLP bag / GBDT bag means
    def bag(substrs, pool):
        oo, tt, n = [], [], 0
        for name, o, t in pool:
            if any(s in name.lower() for s in substrs):
                oo.append(L_fold(o, folds)); tt.append(L(t)); n += 1
        if n == 0:
            return None, None
        return unit(np.mean(oo, 0)), unit(np.mean(tt, 0))

    lm_o, lm_t = bag(["gpt2", "l7_s2026", "megayak", "masaya_gpt2", "lm7"], prop_streams + pub_streams)
    rmlp_o, rmlp_t = bag(["realmlp"], pub_streams)
    gbdt_o, gbdt_t = bag(["xgb", "lgb", "gb_", "gbn", "cat", "stageb", "harvest"], prop_streams + pub_streams)
    if lm_o is not None and rmlp_o is not None:
        Xo_list.append(unit(lm_o - rmlp_o)); Xt_list.append(unit(lm_t - rmlp_t)); cont_names.append("spread_LM_RealMLP")
    if lm_o is not None and gbdt_o is not None:
        Xo_list.append(unit(lm_o - gbdt_o)); Xt_list.append(unit(lm_t - gbdt_t)); cont_names.append("spread_LM_GBDT")

    Xo = np.column_stack(Xo_list)
    Xt = np.column_stack(Xt_list)
    log(f"StageB matrix cols={Xo.shape[1]}")

    # Bug1: softened init_score = 0.25 * L(z_modern_pub)
    init_o = 0.25 * L(z_mod_pub_o)
    init_t = 0.25 * L(z_mod_pub_t)

    sb_margin_o = np.zeros(N_TR)
    sb_margin_t = np.zeros(N_TE)
    feat_names = [c.replace(":", "_").replace(".", "_") for c in cont_names]
    try:
        from catboost import CatBoostClassifier
        have_cb = True
    except Exception:
        have_cb = False
    log(f"StageB trainers: LGB + CatBoost={have_cb}")
    for f in range(5):
        tr_m, va_m = folds != f, folds == f
        dtr = lgb.Dataset(Xo[tr_m], label=y[tr_m], init_score=init_o[tr_m], feature_name=feat_names)
        dva = lgb.Dataset(Xo[va_m], label=y[va_m], init_score=init_o[va_m], reference=dtr)
        params = dict(
            objective="binary", metric="auc", learning_rate=0.03,
            max_depth=3, num_leaves=7, min_child_samples=400,
            feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
            verbosity=-1, seed=42 + f,
        )
        bst = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
        # LightGBM raw_score = learned boost only (relative to softened init)
        raw_va = bst.predict(Xo[va_m], raw_score=True)
        raw_te = bst.predict(Xt, raw_score=True)
        if have_cb:
            # Oblivious depth-4 CatBoost with softened baseline via init_score / baseline
            try:
                cb = CatBoostClassifier(
                    loss_function="Logloss", eval_metric="AUC",
                    depth=4, l2_leaf_reg=30, learning_rate=0.05,
                    iterations=250, random_seed=42 + f, verbose=False,
                    early_stopping_rounds=30, thread_count=4,
                )
                # baseline for train + eval (Bug1 softened init)
                cb.fit(
                    Xo[tr_m], y[tr_m],
                    eval_set=(Xo[va_m], y[va_m]),
                    baseline=init_o[tr_m],
                    eval_baseline=init_o[va_m],
                )
                cb_va = cb.predict(Xo[va_m], prediction_type="RawFormulaVal", baseline=init_o[va_m])
                cb_te = cb.predict(Xt, prediction_type="RawFormulaVal", baseline=init_t)
                # subtract baseline → pure residual boost
                cb_va = np.asarray(cb_va, float).ravel() - init_o[va_m]
                cb_te = np.asarray(cb_te, float).ravel() - init_t
                raw_va = 0.5 * (raw_va + cb_va)
                raw_te = 0.5 * (raw_te + cb_te)
            except Exception as e:
                log(f"  CatBoost fold {f} skipped: {e}")
        sb_margin_o[va_m] = raw_va
        sb_margin_t += raw_te / 5
        log(f"  StageB fold {f} done (lgb_best={bst.best_iteration})")

    # δ = raw_margin (boost) ; already relative to softened init in training
    # ZCA/GS vs [B_modern, S02, S06] on test; LOFO-ish on OOF using modern pub
    B_o = unit(L_fold(z_mod_pub_o, folds))
    B_t = unit(L(z_mod_pub_t))
    d_o = unit(sb_margin_o)
    d_t = unit(sb_margin_t)
    # GS against B_modern
    d_o = gs_orth(d_o, [B_o])
    d_t = gs_orth(d_t, [B_t, unit(ray_s02), unit(ray_s06)])
    # Method-B disp lock post-orth
    d_t = d_t * (d_o.std() / (d_t.std() + 1e-12))
    disp_sb = float(np.std(d_t) / (np.std(d_o) + 1e-12))
    ks_sb = float(ks_2samp(d_o, d_t).statistic)

    # evaluate δ on z_modern_all + Zr
    best = {"mean_e5": -1e9}
    for sgn in (1, -1):
        for a in (0.01, 0.02, 0.04, 0.08, 0.12):
            p = apply_zr(L(z_mod_all_o) + sgn * a * d_o, Zg_tr, z_mod_all_o)
            me, pos, fd = fold_delta(y, p, z_mod_all_zr, folds)
            if me > best["mean_e5"]:
                best = {
                    "mean_e5": me, "pos": pos, "folds": fd, "sign": sgn, "alpha": a,
                    "sf25": sf25(y, p, z_mod_all_zr, folds), "t25": t25(fd),
                    "p_oof": p,
                }
    # also vs z_pub baseline
    p_vs_pub = apply_zr(L(z_mod_all_o) + best["sign"] * best["alpha"] * d_o, Zg_tr, z_pub_o)
    me_pub, pos_pub2, fd_pub2 = fold_delta(y, p_vs_pub, z_pub_zr, folds)

    # tip cosines of residual direction on AB
    tip_dir = best["sign"] * d_t
    # natural macro steps
    tips = {}
    for e4 in (1.0, 2.0, 5.0, 8.0, 12.0, 15.0):
        lo, hi = 1e-6, 25.0
        target = e4 * 1e-4
        best_p, best_a, best_diff = ab, 0.0, 1e9
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            p = apply_zr(L(ab) + mid * tip_dir, Zg_te, ab)
            d = one_mrho(p, ab)
            if abs(d - target) < best_diff:
                best_diff, best_a, best_p = abs(d - target), mid, p
            if d < target:
                lo = mid
            else:
                hi = mid
        v = L(best_p) - L(ab)
        tips[f"{e4:.1f}"] = {
            "e4": one_mrho(best_p, ab) * 1e4, "alpha": best_a,
            "cos_S02": cos_full(v, ray_s02), "cos_S06": cos_full(v, ray_s06),
        }

    # L46C regime eval if present
    l46 = None
    for name, o, t in prop_streams:
        if "L4.6C" in name:
            oo, tt, disp, ks = method_b_lock(o, t, folds)
            # orth vs modern
            oo = gs_orth(oo, [B_o]); tt = gs_orth(tt, [B_t])
            tt = tt * (oo.std() / (tt.std() + 1e-12))
            best46 = {"mean_e5": -1e9}
            for sgn in (1, -1):
                for a in (0.01, 0.02, 0.04, 0.08):
                    p = apply_zr(L(z_mod_all_o) + sgn * a * oo, Zg_tr, z_mod_all_o)
                    me, pos, fd = fold_delta(y, p, z_mod_all_zr, folds)
                    if me > best46["mean_e5"]:
                        best46 = {"mean_e5": me, "pos": pos, "folds": fd, "sign": sgn, "alpha": a,
                                  "sf25": sf25(y, p, z_mod_all_zr, folds), "t25": t25(fd),
                                  "disp": float(np.std(tt)/(np.std(oo)+1e-12)), "KS": float(ks_2samp(oo, tt).statistic)}
            l46 = best46
            break

    # stage diagnostic (NO_SUBMIT)
    te_blend = apply_zr(L(z_mod_all_t) + best["sign"] * best["alpha"] * d_t, Zg_te, z_mod_all_t)
    text = pd.DataFrame({"id": ids, "is_fraud": U(te_blend)}).to_csv(index=False, lineterminator="\n")
    path = OUT / "V42_1_StageB_soft_delta_NOSUBMIT.csv"
    path.write_bytes(text.encode())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    report = {
        "NO_SUBMIT": True,
        "env": env,
        "gpt2_shape_fix": gpt_meta,
        "inventory": inventory,
        "counts": {"public_admitted": len(pub_streams), "prop_admitted": len(prop_streams)},
        "z_modern_pub": {
            "auc_zr": auc_pub, "delta_e5_vs_z_pub": d_pub, "pos": pos_pub, "folds": fd_pub,
            "t25": t25(fd_pub), "sf25": sf25(y, z_mod_pub_zr, z_pub_zr, folds),
            "1mrho_vs_AB_e4": rho_ab_pub,
        },
        "z_modern_all": {
            "auc_zr": auc_all, "delta_e5_vs_z_pub": d_all, "pos": pos_all, "folds": fd_all,
            "t25": t25(fd_all), "sf25": sf25(y, z_mod_all_zr, z_pub_zr, folds),
            "1mrho_vs_AB_e4": rho_ab_all,
        },
        "stageB_v42_1": {
            "mean_e5_vs_modern_all": best["mean_e5"], "pos": best["pos"], "folds": best["folds"],
            "sf25": best["sf25"], "t25": best["t25"], "sign": best["sign"], "alpha": best["alpha"],
            "disp": disp_sb, "KS": ks_sb,
            "delta_e5_vs_z_pub": me_pub, "pos_vs_z_pub": pos_pub2,
            "tips": tips,
            "file": path.name, "sha": sha,
            "bugs_fixed": [
                "init_score=0.25*L(z_modern_pub) not 1.0*z_pub",
                "ECDF/Method-B only on continuous logits; discrete raw",
                "post-orth Method-B disp lock on residual",
            ],
        },
        "L46C_regime": l46,
        "missing": [r for r in inventory if r.get("status") in ("MISSING", "BAD_SHAPE", "QUARANTINE")],
    }
    (LOG / "V42_1_MODERN_BANK_AND_STAGEB_REPORT.json").write_text(json.dumps(report, indent=2, default=str))

    # markdown
    lines = [
        "# V42.1 — Modern OOF Bank + Softened Stage-B",
        "",
        f"**NO_SUBMIT=True** · days left ≈ **{days_left:.1f}** (deadline 2026-11-30) · MPS={mps} CUDA={cuda} runpodctl={env['runpodctl']}",
        "",
        "## 1. Inventory highlights",
        f"- Public admitted: **{len(pub_streams)}** · Proprietary admitted: **{len(prop_streams)}**",
        f"- gpt2 shape-fix: fold VA `(n_va, seeds, 2)` → full OOF `(699960,)` / test mean `(299984,)` ✅",
        f"- soft_diff_orth_LM: isolated admit (folder also has v10_* quarantined — not ingested)",
        f"- Missing: G26_5f, MS_5f, U5_gini10_llr, exact E2_heavy_ridge (proxy used if present)",
        "",
        "| asset | oof | test | solo AUC | disp | KS | status |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    # show key rows
    show = [r for r in inventory if r.get("status") in ("OK", "OK_LOCKED", "MISSING", "BAD_SHAPE", "REJECT_KS", "REJECT_AUC>=0.9550")
            and (r["asset_name"].startswith(("gpt2", "l7", "B_xgb", "delta1", "soft_diff", "P4_", "L4.6", "E2", "G26", "MS", "U5", "aman:", "gpw70:realmlp", "part12"))
                 or "MISSING" in str(r.get("status")))]
    for r in show[:40]:
        lines.append(
            f"| {r['asset_name']} | {r.get('oof_shape')} | {r.get('test_shape')} | "
            f"{r.get('solo_oof_auc') if r.get('solo_oof_auc') is not None else '—'} | "
            f"{r.get('disp') if r.get('disp') is not None else '—'} | "
            f"{r.get('KS') if r.get('KS') is not None else '—'} | {r.get('status')} |"
        )

    lines += [
        "",
        "## 2. z_modern vs z_pub_only (0.952614)",
        "",
        "| stack | AUC+Zr | Δe5 vs z_pub | pos/5 | t25 | sf25 | 1−ρ vs AB e4 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| z_modern_pub | {auc_pub:.6f} | {d_pub:.2f} | {pos_pub}/5 | {t25(fd_pub):.2f} | "
        f"{sf25(y, z_mod_pub_zr, z_pub_zr, folds)} | {rho_ab_pub:.2f} |",
        f"| z_modern_all | {auc_all:.6f} | {d_all:.2f} | {pos_all}/5 | {t25(fd_all):.2f} | "
        f"{sf25(y, z_mod_all_zr, z_pub_zr, folds)} | {rho_ab_all:.2f} |",
        "",
        "## 3. StageB v42.1 (softened)",
        f"- Bugs fixed: init=`0.25*L(z_modern_pub)`; continuous-only Method-B; post-orth disp lock",
        f"- δ vs z_modern_all: Δe5=**{best['mean_e5']:.2f}** pos={best['pos']}/5 sf={best['sf25']} "
        f"t25={best['t25']:.2f} sign={best['sign']} α={best['alpha']} disp={disp_sb:.4f} KS={ks_sb:.4f}",
        f"- δ stack vs z_pub: Δe5=**{me_pub:.2f}** pos={pos_pub2}/5",
        "",
        "| tip e4 | cos·S02↓ | cos·S06↓ |",
        "|---:|---:|---:|",
    ]
    for k, t in tips.items():
        lines.append(f"| {t['e4']:.2f} | {t['cos_S02']:.3f} | {t['cos_S06']:.3f} |")
    if l46:
        lines += ["", f"### L4.6C top12 regime: Δe5={l46['mean_e5']:.2f} pos={l46['pos']}/5 sf={l46['sf25']}"]
    lines += [
        "",
        f"Diagnostic CSV (NOSUBMIT): `{path.name}` sha=`{sha}`",
        "",
        "## STOP",
        "NO_SUBMIT=True. Zero submissions. Awaiting explicit fire command.",
    ]
    (LOG / "V42_1_MODERN_BANK_AND_STAGEB_REPORT.md").write_text("\n".join(lines) + "\n")
    log(f"WROTE {LOG / 'V42_1_MODERN_BANK_AND_STAGEB_REPORT.md'}")
    try:
        i2 = next(i for i, L_ in enumerate(lines) if L_.startswith("## 2."))
        print("\n".join(lines[i2:]), flush=True)
    except StopIteration:
        print("\n".join(lines[-40:]), flush=True)
    print("STOP. NO_SUBMIT=True.", flush=True)


if __name__ == "__main__":
    main()
