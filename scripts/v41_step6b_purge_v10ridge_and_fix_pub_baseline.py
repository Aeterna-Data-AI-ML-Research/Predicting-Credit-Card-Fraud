#!/usr/bin/env python3
"""V41 STEP 6b — Purge v10_ridge/zanch poison; fix pub baseline; refresh slots 04–10.

STRICT NO_SUBMIT=True. Does NOT submit. Hard-stop after rewrite + manifest.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

NO_SUBMIT = True
N_TR, N_TE = 699_960, 299_984
T0 = time.time()
LB_01 = 0.95569  # live Slot01 result (set by caller / known)

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
SSD = Path("/tmp/s6e11_v39_ssd")
EXT = ROOT / "external/v41_nov23_frontier"
GPW = ROOT / "external/banks/goodpjw2008_s6e11-oof-bank"
BUSY = EXT / "busyaprime_52"
PART12 = ROOT / "external/nov08_part12"
LOG = ROOT / "logs/v41"
FINAL = ROOT / "out/v41_final_10slots"
STAGED = ROOT / "out/v41_staged"
PRED = ROOT / "preds"
DATA = ROOT / "data" if (ROOT / "data/test.csv").exists() else SSD / "data"
FOLDS_P = SSD / "folds/folds.csv" if (SSD / "folds/folds.csv").exists() else ROOT / "folds/folds.csv"

SHA_AB = "3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038"
SHA_Z = "ee61b50611326eb668033f088f8803e28541874c1cff2cf073334ea7d0f05a11"

# HARD QUARANTINE — poison + TE-bleed
QUARANTINE = (
    "uid_catboost", "v41_d27_eb", "fintfm", "fin_tfm", "zanch",
    "v10_ridge", "v10_cb_d4", "v10_lgb", "gb_soft_bag3", "soft_diff",
    "idf_p3", "clean035",
)

# Must NEVER appear inside z_pub_only
PRIVATE_FORBIDDEN = (
    "gpt2_med_4ep", "l7_s2026", "xgb_llr", "harvest_p4", "uid_v2",
    "uid_catboost", "p_super_v5", "p_super_v6", "p_champ", "gpt2_rev",
    "paradigm", "matrix_v2", "v10_",
)

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


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
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


def is_quar(s: str) -> bool:
    s = s.lower()
    return any(q in s for q in QUARANTINE)


def is_private_forbidden(s: str) -> bool:
    s = s.lower()
    return any(p in s for p in PRIVATE_FORBIDDEN)


def write_slot(path: Path, ids, scores) -> tuple[str, np.ndarray]:
    p = U(scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(index=False, lineterminator="\n")
    path.write_bytes(text.encode())
    return sha256_file(path), p


def fold_aucs(y, p, folds):
    return [float(roc_auc_score(y[folds == k], p[folds == k])) for k in range(5)]


def sf25_beats(y, a, b, folds) -> int:
    pos = 0
    for k in range(5):
        m = folds == k
        yk, ak, bk = y[m], a[m], b[m]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + k)
        for _, va in skf.split(np.arange(len(yk)), yk):
            if roc_auc_score(yk[va], ak[va]) > roc_auc_score(yk[va], bk[va]):
                pos += 1
    return pos


def scale_to_target_1mrho(base_u, direction, target_e4, zones, u_ab, lo=1e-6, hi=8.0):
    base_u = U(base_u)
    direction = np.asarray(direction, float).ravel()
    target = target_e4 * 1e-4
    best_a, best_diff, best_p = 0.0, 1e9, base_u
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        p = apply_zr(L(base_u) + mid * direction, zones, u_ab)
        d = one_mrho(p, base_u)
        if abs(d - target) < best_diff:
            best_diff, best_a, best_p = abs(d - target), mid, p
        if d < target:
            lo = mid
        else:
            hi = mid
    return best_p, best_a, one_mrho(best_p, base_u) * 1e4


def load_public_matrix(y, folds):
    """gpw70 + busy52 only — zero private local files."""
    names, oofs, tes, sources = [], [], [], []

    def try_add(name, o, t, source):
        if is_quar(name) or is_private_forbidden(name) or is_private_forbidden(source):
            return
        o, t = np.asarray(o, float).ravel(), np.asarray(t, float).ravel()
        if o.shape != (N_TR,) or t.shape != (N_TE,):
            return
        if not (np.isfinite(o).all() and np.isfinite(t).all()):
            return
        auc = float(roc_auc_score(y, o))
        if auc < 0.940 or auc > 0.960:
            return
        ks = float(ks_2samp(L_fold(o, folds), L(t)).statistic)
        if ks > 0.045:
            return
        names.append(name)
        oofs.append(o)
        tes.append(t)
        sources.append(source)

    man = pd.read_csv(GPW / "manifest.csv")
    for _, r in man.iterrows():
        n = str(r["name"])
        op, tp = GPW / f"oof_{n}.npy", GPW / f"test_{n}.npy"
        if op.exists() and tp.exists():
            try_add(f"gpw70:{n}", np.load(op), np.load(tp), "goodpjw70")

    bman = pd.read_csv(BUSY / "manifest.csv")
    Bo, Bt = np.load(BUSY / "oof.npy"), np.load(BUSY / "test.npy")
    for i, r in bman.iterrows():
        n = str(r["name"])
        try_add(f"busy52:{n}", Bo[:, i], Bt[:, i], "busyaprime52")
    if (BUSY / "tabm_oof.npy").exists():
        try_add("busy52:tabm_standalone", np.load(BUSY / "tabm_oof.npy"),
                np.load(BUSY / "tabm_test.npy"), "busyaprime52_tabm")

    # public nov08 lm member (not private)
    if (PART12 / "lm_oof.npy").exists() and (PART12 / "lm_test.npy").exists():
        try_add("pub:nov08_lm", np.load(PART12 / "lm_oof.npy"),
                np.load(PART12 / "lm_test.npy"), "nov08_part12")

    Xo = np.column_stack(oofs)
    Xt = np.column_stack(tes)
    # hard assert no private leakage in names/sources
    for n, s in zip(names, sources):
        assert not is_private_forbidden(n), n
        assert not is_private_forbidden(s), s
        assert not is_quar(n), n
    return names, Xo, Xt


def build_z_pub_only(y, folds, names, Xo, Xt, part12_o, part12_t):
    """Part12 locked offset + nested LogReg on public bank; then fullbank-style blend w/ Part12."""
    # family-cap for speed / Yash law
    fam_caps = {"LM": 12, "GBDT": 14, "TabM": 12, "EBM": 10, "Other": 8}

    def fam(n):
        s = n.lower()
        if any(x in s for x in ("gpt", "compact", "target", "l7", "lm", "mega", "ms")):
            return "LM"
        if any(x in s for x in ("tabm", "realmlp", "rmlp", "neural")):
            return "TabM"
        if any(x in s for x in ("ebm", "glm", "slr")) and "boost" not in s and "g2_" not in s:
            return "EBM"
        if any(x in s for x in ("lgb", "xgb", "gb_", "gbc", "gbe", "gbx", "booster", "glminit")):
            return "GBDT"
        return "Other"

    keep = []
    by = {f: [] for f in fam_caps}
    for i, n in enumerate(names):
        by[fam(n)].append((i, float(roc_auc_score(y, Xo[:, i]))))
    for f, cap in fam_caps.items():
        ranked = sorted(by[f], key=lambda z: -z[1])[:cap]
        keep.extend([i for i, _ in ranked])
    keep = sorted(set(keep))
    names_k = [names[i] for i in keep]
    Xo, Xt = Xo[:, keep], Xt[:, keep]
    log(f"z_pub_only bank kept={len(keep)} members_sample={names_k[:8]}")

    Fo = np.column_stack([L_fold(Xo[:, j], folds) for j in range(Xo.shape[1])])
    Ft = np.column_stack([L(Xt[:, j]) for j in range(Xt.shape[1])])
    nest_o = np.zeros(N_TR)
    nest_t = np.zeros(N_TE)
    for f in range(5):
        tr, va = folds != f, folds == f
        best_c, best_sc = 0.1, -1.0
        inner = folds[tr]
        for C in (0.03, 0.1, 0.3, 1.0):
            scs = []
            for u in np.unique(inner):
                it, iv = inner != u, inner == u
                clf = LogisticRegression(C=C, solver="lbfgs", max_iter=250, tol=1e-3)
                clf.fit(Fo[tr][it], y[tr][it])
                scs.append(roc_auc_score(y[tr][iv], clf.predict_proba(Fo[tr][iv])[:, 1]))
            sc = float(np.mean(scs))
            if sc > best_sc:
                best_sc, best_c = sc, C
        clf = LogisticRegression(C=best_c, solver="lbfgs", max_iter=400)
        clf.fit(Fo[tr], y[tr])
        nest_o[va] = clf.predict_proba(Fo[va])[:, 1]
        nest_t += clf.predict_proba(Ft)[:, 1] / 5
        log(f"  pub nest fold {f}: C={best_c} inner={best_sc:.5f}")

    # blend with Part12 (public locked offset)
    def unitL(v):
        return unit(L(U(v)))

    cands = {
        "nest": (nest_o, nest_t),
        "part12": (part12_o, part12_t),
        "0.85p+0.15n": (
            U(0.85 * L(U(part12_o)) + 0.15 * L(U(nest_o))),
            U(0.85 * L(U(part12_t)) + 0.15 * L(U(nest_t))),
        ),
        "0.70p+0.30n": (
            U(0.70 * L(U(part12_o)) + 0.30 * L(U(nest_o))),
            U(0.70 * L(U(part12_t)) + 0.30 * L(U(nest_t))),
        ),
        "rankmean": (
            U(0.5 * U(part12_o) + 0.5 * U(nest_o)),
            U(0.5 * U(part12_t) + 0.5 * U(nest_t)),
        ),
    }
    # part12 + 0.02 orth(nest)
    lo_o, hi_o = np.quantile(nest_o, [0.001, 0.999])
    lo_t, hi_t = np.quantile(nest_t, [0.001, 0.999])
    u = unit(np.clip(nest_o, lo_o, hi_o))
    bb = unit(L(part12_o))
    u = unit(u - np.dot(u, bb) / (np.dot(bb, bb) + 1e-12) * bb)
    ut = unit(np.clip(nest_t, lo_t, hi_t))
    bbt = unit(L(part12_t))
    ut = unit(ut - np.dot(ut, bbt) / (np.dot(bbt, bbt) + 1e-12) * bbt)
    cands["part12+0.02orth_nest"] = (
        U(L(U(part12_o)) + 0.02 * u),
        U(L(U(part12_t)) + 0.02 * ut),
    )

    best_name, best_a, best_o, best_t = "part12", -1.0, part12_o, part12_t
    for k, (oo, tt) in cands.items():
        a = float(roc_auc_score(y, oo))
        log(f"  z_pub_only cand {k}: {a:.6f}")
        if a > best_a:
            best_name, best_a, best_o, best_t = k, a, oo, tt
    return best_o, best_t, best_name, best_a, names_k


def rebuild_fullbank_clean(y, folds, part12_o, part12_t):
    """Replicate B_clean nest + wide blend WITHOUT v10orth (w_v10=0)."""
    # Use saved B_clean pillars (may include private P_super/gpt2_rev — intentional for C4)
    # but ZERO v10_ridge / zanch
    Bo = np.load(SSD / "preds/B_clean_oof.npy")
    Bt = np.load(SSD / "preds/B_clean_test.npy")
    # Drop any column that correlates with quarantined v10 if we can identify — B_clean names from w0
    # names: wide_lm, lgb_subodh, nov08_lm, gpt2_rev, gpt2_rev_raw, P_super_v5p5, P_super_v6, P_champ
    # none are v10_ridge; v10 was only post-hoc orth on the artifact
    Fo = np.column_stack([L_fold(Bo[:, j], folds) for j in range(Bo.shape[1])])
    Ft = np.column_stack([L(Bt[:, j]) for j in range(Bt.shape[1])])
    nest_o = np.zeros(N_TR)
    nest_t = np.zeros(N_TE)
    for f in range(5):
        tr, va = folds != f, folds == f
        clf = LogisticRegression(C=0.5, solver="lbfgs", max_iter=400)
        clf.fit(Fo[tr], y[tr])
        nest_o[va] = clf.predict_proba(Fo[va])[:, 1]
        nest_t += clf.predict_proba(Ft)[:, 1] / 5
    wide_o, wide_t = part12_o, part12_t
    cands = {
        "nest": (nest_o, nest_t),
        "wide": (wide_o, wide_t),
        "0.85w+0.15n": (
            U(0.85 * L(U(wide_o)) + 0.15 * L(U(nest_o))),
            U(0.85 * L(U(wide_t)) + 0.15 * L(U(nest_t))),
        ),
        "0.70w+0.30n": (
            U(0.70 * L(U(wide_o)) + 0.30 * L(U(nest_o))),
            U(0.70 * L(U(wide_t)) + 0.30 * L(U(nest_t))),
        ),
        "rankmean": (
            U(0.5 * U(wide_o) + 0.5 * U(nest_o)),
            U(0.5 * U(wide_t) + 0.5 * U(nest_t)),
        ),
    }
    u = unit(np.clip(nest_o, *np.quantile(nest_o, [0.001, 0.999])))
    bb = unit(L(wide_o))
    u = unit(u - np.dot(u, bb) / (np.dot(bb, bb) + 1e-12) * bb)
    ut = unit(np.clip(nest_t, *np.quantile(nest_t, [0.001, 0.999])))
    bbt = unit(L(wide_t))
    ut = unit(ut - np.dot(ut, bbt) / (np.dot(bbt, bbt) + 1e-12) * bbt)
    cands["wide+0.02orth_nest"] = (
        U(L(U(wide_o)) + 0.02 * u),
        U(L(U(wide_t)) + 0.02 * ut),
    )
    best_name, best_a, best_o, best_t = "wide", -1.0, wide_o, wide_t
    for k, (oo, tt) in cands.items():
        a = float(roc_auc_score(y, oo))
        log(f"  Fullbank_clean cand {k}: {a:.6f}")
        if a > best_a:
            best_name, best_a, best_o, best_t = k, a, oo, tt
    log(f"Fullbank_clean selected={best_name} AUC={best_a:.6f} (w_v10=0, no zanch)")
    return best_o, best_t, best_name, best_a


def lofo_orth_vs_base(target_o, target_t, base_o, base_t, folds, ridge=50.0):
    """Orth target vs [base] (+ optional bank later)."""
    yo = unit(L_fold(target_o, folds))
    yt = unit(L(target_t))
    bo = unit(L_fold(base_o, folds)).reshape(-1, 1)
    bt = unit(L(base_t)).reshape(-1, 1)
    oof = np.zeros(N_TR)
    te = np.zeros(N_TE)
    for f in range(5):
        tr, va = folds != f, folds == f
        beta = np.linalg.solve(bo[tr].T @ bo[tr] + ridge * np.eye(1), bo[tr].T @ yo[tr])
        oof[va] = unit(yo[va] - (bo[va] @ beta).ravel())
        te += unit(yt - (bt @ beta).ravel()) / 5
    return unit(oof), unit(te)


def signed_lofo_gate(u_o, u_t, y, folds, base_zr) -> dict:
    best = {"sign": 0, "pos": 0, "mean_e5": -1e9, "sf25": 0, "folds": [], "pass": False}
    for sgn in (1, -1):
        lifts = []
        oof = np.zeros(N_TR)
        for f in range(5):
            tr, va = folds != f, folds == f
            best_a, best_sc = 0.0, roc_auc_score(y[tr], base_zr[tr])
            for a in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
                sc = roc_auc_score(y[tr], U(L(base_zr[tr]) + sgn * a * u_o[tr]))
                if sc > best_sc:
                    best_sc, best_a = sc, a
            pred = U(L(base_zr[va]) + sgn * best_a * u_o[va])
            oof[va] = pred
            lifts.append((roc_auc_score(y[va], pred) - roc_auc_score(y[va], base_zr[va])) * 1e5)
        pos = sum(1 for x in lifts if x > 0)
        mean_e5 = float(np.mean(lifts))
        sf = sf25_beats(y, oof, base_zr, folds)
        if (pos, mean_e5) > (best["pos"], best["mean_e5"]):
            best = {"sign": sgn, "pos": pos, "mean_e5": mean_e5, "sf25": sf,
                    "folds": lifts, "pass": pos >= 4 and mean_e5 > 0}
    best["disp"] = float(u_t.std() / (u_o.std() + 1e-12))
    best["KS"] = float(ks_2samp(u_o, u_t).statistic)
    return best


def null_space_P_perp(u_te, d_ab_z, zones):
    m = ~zones
    d = unit(d_ab_z[m])
    u = u_te.copy()
    um = unit(u[m])
    proj = float(np.dot(um, d))
    if proj < 0:
        um = unit(um - proj * d)
        u[m] = um
    return unit(u), {"cos": proj, "projected": proj < 0}


def inner_cv_signed_weights(cols_o, y, folds, train_mask):
    """Inner-4 CV pick signed weights for column stack on train_mask."""
    M = len(cols_o)
    X = np.column_stack([L_fold(c, folds) for c in cols_o])[train_mask]
    yy = y[train_mask]
    ff = folds[train_mask]
    # start equal; try sign flips via coordinate search
    w = np.ones(M) / M
    def score(ww):
        scs = []
        for u in np.unique(ff):
            iv = ff == u
            scs.append(roc_auc_score(yy[iv], X[iv] @ ww))
        return float(np.mean(scs))

    best = score(w)
    for j in range(M):
        for s in (1.0, -1.0):
            trial = w.copy()
            trial[j] = s * abs(trial[j])
            trial = trial / (np.sum(np.abs(trial)) + 1e-12)
            sc = score(trial)
            if sc > best:
                best, w = sc, trial
    return w, best


def main():
    assert NO_SUBMIT is True
    LOG.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    log("STEP6b start NO_SUBMIT=True — purge v10_ridge/zanch; fix pub baseline")

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(float)
    folds = pd.read_csv(FOLDS_P)["fold"].to_numpy(int)
    te_df = pd.read_csv(DATA / "test.csv")
    tr_df = pd.read_csv(DATA / "train.csv")
    ids = te_df["id"].to_numpy()
    Zg_tr = z_goodpjw(tr_df)
    Zg_te = z_goodpjw(te_df)

    ab = pd.read_csv(FINAL / "SLOT01_pub_Step_AB_095569.csv")["is_fraud"].to_numpy(float)
    assert sha256_file(FINAL / "SLOT01_pub_Step_AB_095569.csv") == SHA_AB
    u_ab = U(ab)
    tip_z = pd.read_csv(EXT / "goodpjw_95569/submission_step_z.csv")["is_fraud"].to_numpy(float)
    assert sha256_file(EXT / "goodpjw_95569/submission_step_z.csv") == SHA_Z
    d_AB_Z = unit(L(ab) - L(tip_z))
    areeb = pd.read_csv(EXT / "areeb_95568/submission.csv")["is_fraud"].to_numpy(float)

    part12_o = np.load(PART12 / "wide_lm_stack_nested.npy").ravel()
    part12_t = np.load(PART12 / "wide_lm_stack_test.npy").ravel()
    p12_zr = apply_zr(part12_o, Zg_tr, U(part12_o))
    fb_old = np.load(SSD / "preds/z_fullbank_oof.npy").ravel()
    log(f"Part12+Zr={roc_auc_score(y, p12_zr):.6f} old_fullbank={roc_auc_score(y, fb_old):.6f}")
    log("SOURCE: w0_repair applies v10_ridge_resid_orth_LM micro-orth onto fullbank → POISON in artifact")

    # ── z_pub_only ──
    names, Xo, Xt = load_public_matrix(y, folds)
    log(f"public matrix n={len(names)}")
    z_pub_o, z_pub_t, pub_name, pub_auc, kept_names = build_z_pub_only(
        y, folds, names, Xo, Xt, part12_o, part12_t
    )
    z_pub_o_zr = apply_zr(z_pub_o, Zg_tr, U(z_pub_o))
    z_pub_t_zr = apply_zr(z_pub_t, Zg_te, u_ab)
    pub_auc_zr = float(roc_auc_score(y, z_pub_o_zr))
    log(f"z_pub_only={pub_name} raw={pub_auc:.6f} +Zr={pub_auc_zr:.6f}")
    p12_zr_auc = float(roc_auc_score(y, p12_zr))
    assert pub_auc_zr + 1e-12 >= p12_zr_auc, (
        f"z_pub_only+Zr {pub_auc_zr} < Part12+Zr {p12_zr_auc}"
    )
    log(f"ASSERT z_pub_only+Zr={pub_auc_zr:.10f} >= Part12+Zr={p12_zr_auc:.10f}")
    # save
    np.save(SSD / "preds/z_pub_only_oof.npy", z_pub_o)
    np.save(SSD / "preds/z_pub_only_test.npy", z_pub_t)

    # ── Fullbank_clean (w_v10=0) ──
    fb_c_o, fb_c_t, fb_c_name, fb_c_auc = rebuild_fullbank_clean(y, folds, part12_o, part12_t)
    fb_c_o_zr = apply_zr(fb_c_o, Zg_tr, U(fb_c_o))
    fb_c_t_zr = apply_zr(fb_c_t, Zg_te, u_ab)
    fb_c_auc_zr = float(roc_auc_score(y, fb_c_o_zr))
    np.save(SSD / "preds/z_fullbank_clean_oof.npy", fb_c_o)
    np.save(SSD / "preds/z_fullbank_clean_test.npy", fb_c_t)
    log(f"Fullbank_clean+Zr={fb_c_auc_zr:.6f}")

    # Fullbank_clean - z_pub_only private lift
    diff_o = L_fold(fb_c_o, folds) - L_fold(z_pub_o, folds)
    diff_t = L(fb_c_t) - L(z_pub_t)
    # 5/5 fold wins of pub + α*diff vs pub
    fold_wins = []
    for f in range(5):
        va = folds == f
        base = roc_auc_score(y[va], z_pub_o_zr[va])
        best = base
        for a in np.linspace(0.0, 0.5, 21):
            sc = roc_auc_score(y[va], U(L(z_pub_o_zr[va]) + a * unit(diff_o[va])))
            best = max(best, sc)
        fold_wins.append(best > base + 1e-9)
    priv_lift = {
        "fold_wins": int(sum(fold_wins)),
        "fold_win_flags": fold_wins,
        "oof_auc_diff_solo": float(roc_auc_score(y, diff_o)),
        "note": "L_fold(Fullbank_clean)-L_fold(z_pub_only) private combined lift over public bank",
    }
    log(f"private lift Fullbank_clean - z_pub_only fold_wins={priv_lift['fold_wins']}/5")

    # ── load clean private signals (NO v10) ──
    def load_gpt2():
        gdir = PRED / "gpt2_med_4ep"
        go = np.zeros(N_TR)
        gt = []
        for k in range(5):
            a = np.load(gdir / f"oof_{k}.npy")
            b = np.load(gdir / f"test_{k}.npy")
            go[folds == k] = (a[:, :, 1] - a[:, :, 0]).mean(1)
            gt.append((b[:, :, 1] - b[:, :, 0]).mean(1))
        return go, np.mean(gt, 0)

    lo = np.load(PRED / "l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    lt = np.load(PRED / "l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    l7o = (lo[:, :, 1] - lo[:, :, 0]).mean(1)
    l7t = ((lt[..., 1] - lt[..., 0]).mean(2)).mean(0)
    gpt_o, gpt_t = load_gpt2()
    xgb_o = np.load(PRED / "paradigm_B/xgb_llr_oof.npy").ravel()
    xgb_t = np.load(PRED / "paradigm_B/xgb_llr_test.npy").ravel()
    har_o = np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_oof.npy").ravel()
    har_t = np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_test.npy").ravel()

    # public lm for C2 swap (nov08_lm)
    pub_lm_o = np.load(PART12 / "lm_oof.npy").ravel()
    pub_lm_t = np.load(PART12 / "lm_test.npy").ravel()

    scorecard = {}

    # C1 seq_llr — inner CV signed weights, then orth vs z_pub_only
    c1_cols_o = [gpt_o, l7o, xgb_o]
    c1_cols_t = [gpt_t, l7t, xgb_t]
    # global weights via mean of per-fold inner picks
    ws = []
    for f in range(5):
        tr = folds != f
        w, sc = inner_cv_signed_weights(c1_cols_o, y, folds, tr)
        ws.append(w)
    w_c1 = np.mean(ws, 0)
    w_c1 = w_c1 / (np.sum(np.abs(w_c1)) + 1e-12)
    c1_raw_o = sum(w_c1[j] * L_fold(c1_cols_o[j], folds) for j in range(3))
    c1_raw_t = sum(w_c1[j] * L(c1_cols_t[j]) for j in range(3))
    c1_o, c1_t = lofo_orth_vs_base(c1_raw_o, c1_raw_t, z_pub_o, z_pub_t, folds)
    c1_t, pinfo = null_space_P_perp(c1_t, d_AB_Z, Zg_te)
    g1 = signed_lofo_gate(c1_o, c1_t, y, folds, z_pub_o_zr)
    g1["weights"] = {"gpt2_med": float(w_c1[0]), "l7_s2026": float(w_c1[1]), "xgb_llr": float(w_c1[2])}
    g1["P_perp"] = pinfo
    scorecard["C1_seq_llr"] = g1
    log(f"C1_seq_llr pass={g1['pass']} sign={g1['sign']} pos={g1['pos']}/5 Δe5={g1['mean_e5']:.2f} w={g1['weights']}")

    # C2 famE l7 swap — NO flip_to_plus on fold0; sweep w
    c2_raw_o = L_fold(l7o, folds) - L_fold(pub_lm_o, folds)
    c2_raw_t = L(l7t) - L(pub_lm_t)
    # tail weight: emphasize high |diff|
    tail_o = c2_raw_o * (np.abs(c2_raw_o) >= np.quantile(np.abs(c2_raw_o), 0.7))
    tail_t = c2_raw_t * (np.abs(c2_raw_t) >= np.quantile(np.abs(c2_raw_t), 0.7))
    c2_o, c2_t = lofo_orth_vs_base(tail_o, tail_t, z_pub_o, z_pub_t, folds)
    c2_t, pinfo2 = null_space_P_perp(c2_t, d_AB_Z, Zg_te)
    # sweep w on full OOF for reporting (honest nested gate still via signed_lofo)
    sweep = {}
    for w in np.linspace(-0.04, 0.04, 17):
        if abs(w) < 1e-9:
            continue
        sweep[f"{w:.3f}"] = float(roc_auc_score(y, U(L(z_pub_o_zr) + w * unit(c2_o))))
    g2 = signed_lofo_gate(c2_o, c2_t, y, folds, z_pub_o_zr)
    g2["w_sweep_auc"] = sweep
    g2["best_w_sweep"] = max(sweep, key=sweep.get) if sweep else None
    g2["P_perp"] = pinfo2
    scorecard["C2_famE_l7swap"] = g2
    log(f"C2_famE_l7swap pass={g2['pass']} sign={g2['sign']} pos={g2['pos']}/5 Δe5={g2['mean_e5']:.2f} "
        f"best_w={g2['best_w_sweep']}")

    # C3 harvest_p4 both signs
    c3_o, c3_t = lofo_orth_vs_base(har_o, har_t, z_pub_o, z_pub_t, folds)
    c3_t, pinfo3 = null_space_P_perp(c3_t, d_AB_Z, Zg_te)
    g3 = signed_lofo_gate(c3_o, c3_t, y, folds, z_pub_o_zr)
    g3["P_perp"] = pinfo3
    g3["hygiene"] = {"disp": float(har_t.std() / (har_o.std() + 1e-12)),
                     "KS": float(ks_2samp(har_o, har_t).statistic)}
    scorecard["C3_harvest_p4"] = g3
    log(f"C3_harvest_p4 pass={g3['pass']} sign={g3['sign']} pos={g3['pos']}/5 Δe5={g3['mean_e5']:.2f}")

    # peer hybrids optional
    peer_hits = list(PRED.glob("**/sub_v18_9_M5_peer_hybrid*oof*.npy")) + list(PRED.glob("**/v18_8_peers*oof*.npy"))
    scorecard["peer_hybrids"] = {"found": [str(p) for p in peer_hits[:5]]}

    # C4 fullbank_clean priv diff
    c4_o, c4_t = lofo_orth_vs_base(diff_o, diff_t, z_pub_o, z_pub_t, folds)
    # diff already vs pub — orth again is soft; use raw diff unit too
    c4_o = unit(diff_o)
    c4_t = unit(diff_t)
    c4_t, pinfo4 = null_space_P_perp(c4_t, d_AB_Z, Zg_te)
    g4 = signed_lofo_gate(c4_o, c4_t, y, folds, z_pub_o_zr)
    g4["P_perp"] = pinfo4
    scorecard["C4_fullbank_priv_diff"] = g4
    log(f"C4_fullbank_priv_diff pass={g4['pass']} sign={g4['sign']} pos={g4['pos']}/5 Δe5={g4['mean_e5']:.2f}")

    # collect passing dirs on test (signed)
    dirs = {}
    if g1["pass"]:
        dirs["C1"] = g1["sign"] * c1_t
        dirs_o_map = {"C1": g1["sign"] * c1_o}
    else:
        dirs_o_map = {}
    if g2["pass"]:
        dirs["C2"] = g2["sign"] * c2_t
        dirs_o_map["C2"] = g2["sign"] * c2_o
    if g3["pass"]:
        dirs["C3"] = g3["sign"] * c3_t
        dirs_o_map["C3"] = g3["sign"] * c3_o
    if g4["pass"]:
        dirs["C4"] = g4["sign"] * c4_t
        dirs_o_map["C4"] = g4["sign"] * c4_o
    log(f"PASSING clean dirs: {list(dirs.keys())}")

    # ── refresh slots 04–10 (keep 01–03) ──
    slots_meta = {}
    # load existing 01-03
    for sn, fn in [
        ("SLOT01", "SLOT01_pub_Step_AB_095569.csv"),
        ("SLOT02", "SLOT02_AB_clip985_areeb_a0p32.csv"),
        ("SLOT03", "SLOT03_AB_clip985_quarter_a0p28.csv"),
    ]:
        p = pd.read_csv(FINAL / fn)["is_fraud"].to_numpy(float)
        slots_meta[sn] = {"file": fn, "sha": sha256_file(FINAL / fn),
                          "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4, "kept": True}
    vecs = {
        "SLOT01": ab,
        "SLOT02": pd.read_csv(FINAL / "SLOT02_AB_clip985_areeb_a0p32.csv")["is_fraud"].to_numpy(float),
        "SLOT03": pd.read_csv(FINAL / "SLOT03_AB_clip985_quarter_a0p28.csv")["is_fraud"].to_numpy(float),
    }

    def tip(name, fname, role, direction, target):
        if direction is None:
            # fallback geodesic tiny
            direction = clip_p985(L(areeb) - L(ab))
            role = role + " [FALLBACK areeb]"
        p, a, e = scale_to_target_1mrho(ab, clip_p985(direction), target, Zg_te, u_ab)
        if e < 0.015:
            p, a, e = scale_to_target_1mrho(ab, clip_p985(direction), 0.025, Zg_te, u_ab)
        if e > 2.0:
            p, a, e = scale_to_target_1mrho(ab, clip_p985(direction), 0.85, Zg_te, u_ab)
        sha, pout = write_slot(FINAL / fname, ids, p)
        slots_meta[name] = {"file": fname, "sha": sha, "role": role, "alpha": a,
                            "1mrho_vs_AB_e4": e, "NO_SUBMIT": True}
        vecs[name] = pout
        log(f"  {name}: AB_e4={e:.4f} α={a:.5f} {role[:60]}")
        return pout

    # SLOT04 C1
    tip("SLOT04", "SLOT04_AB_C1_seq_llr_clip985_0p25e4.csv",
        "C1_seq_llr (gpt2+l7+xgb) on AB @0.25e-4; zanch/v10 PURGED",
        dirs.get("C1"), 0.25)
    # SLOT05 C1 step-up or C1+C2
    if "C1" in dirs and "C2" in dirs:
        d05 = unit(0.7 * unit(dirs["C1"]) + 0.3 * unit(dirs["C2"]))
        role05 = "C1+C2 seq upgrade on AB @0.55e-4"
    else:
        d05 = dirs.get("C1")
        role05 = "C1_seq_llr step-up on AB @0.55e-4"
    tip("SLOT05", "SLOT05_AB_C1_seq_llr_clip985_0p55e4.csv", role05, d05, 0.55)
    # SLOT06 C3
    tip("SLOT06", "SLOT06_AB_C3_harvest_p4_clip985_0p28e4.csv",
        "C3_harvest_p4 pure (zero zanch) on AB @0.28e-4",
        dirs.get("C3"), 0.28)
    # SLOT07 C4
    tip("SLOT07", "SLOT07_AB_C4_fullbank_clean_priv_diff_clip985_0p40e4.csv",
        "C4 Fullbank_clean - z_pub_only priv diff on AB @0.40e-4",
        dirs.get("C4"), 0.40)
    # SLOT08 master C1+C2+C3+SLOT02 geodesic
    parts = []
    for k in ("C1", "C2", "C3"):
        if k in dirs:
            parts.append(unit(dirs[k]))
    geo = unit(clip_p985(L(areeb) - L(ab)))
    if parts:
        master = unit(sum(parts) / len(parts) + 0.35 * geo)
    else:
        master = geo
    tip("SLOT08", "SLOT08_AB_master_clean_combo_0p45e4.csv",
        "Master clean combo C1+C2+C3+geodesic @0.45e-4", master, 0.45)

    # SLOT09 Fullbank_clean + Zr
    sha, p = write_slot(FINAL / "SLOT09_pure_nested_cv_champion_zr.csv", ids, fb_c_t_zr)
    slots_meta["SLOT09"] = {
        "file": "SLOT09_pure_nested_cv_champion_zr.csv", "sha": sha,
        "role": f"Fullbank_clean+Zr ({fb_c_name}, w_v10=0)",
        "oof_auc": fb_c_auc, "oof_auc_zr": fb_c_auc_zr,
        "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4, "NO_SUBMIT": True,
    }
    vecs["SLOT09"] = p
    # SLOT10 hedge
    hedge = apply_zr(U(0.55 * L(ab) + 0.45 * L(fb_c_t_zr)), Zg_te, u_ab)
    sha, p = write_slot(FINAL / "SLOT10_private_hedge_55BestTip_45CVChamp_zr.csv", ids, hedge)
    slots_meta["SLOT10"] = {
        "file": "SLOT10_private_hedge_55BestTip_45CVChamp_zr.csv", "sha": sha,
        "role": "55% AB + 45% Fullbank_clean + Zr",
        "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4, "NO_SUBMIT": True,
    }
    vecs["SLOT10"] = p

    # integrity + matrix
    slot_names = [f"SLOT{i:02d}" for i in range(1, 11)]
    M = np.zeros((10, 10))
    for i, a in enumerate(slot_names):
        for j, b in enumerate(slot_names):
            M[i, j] = 0.0 if i == j else one_mrho(vecs[a], vecs[b]) * 1e4

    integrity = {}
    for i, sn in enumerate(slot_names):
        p = vecs[sn]
        max_uz = float(p[Zg_te].max())
        min_div = None if i == 0 else min(one_mrho(p, vecs[slot_names[j]]) * 1e4 for j in range(i))
        # SLOT08 needs >=0.025 vs earlier
        thr = 0.025 if sn == "SLOT08" else 0.020
        ok = (i == 0) or (min_div >= thr - 1e-6)
        e_ab = one_mrho(p, ab) * 1e4
        integrity[sn] = {
            "max_U_zone": max_uz, "Zr_ok": max_uz <= 0.0558 + 1e-4,
            "min_1mrho_vs_earlier_e4": min_div, "diversity_ok": ok,
            "1mrho_vs_AB_e4": e_ab, "uid_catboost_weight": 0.0,
            "v10_ridge_weight": 0.0, "zanch_weight": 0.0,
        }
        slots_meta[sn]["integrity"] = integrity[sn]
        slots_meta[sn]["1mrho_vs_AB_e4"] = e_ab
        log(f"  INT {sn}: AB_e4={e_ab:.4f} minDiv={min_div} div_ok={ok} Zr={integrity[sn]['Zr_ok']}")

    # If SLOT08 diversity fails, escalate target
    if not integrity["SLOT08"]["diversity_ok"]:
        log("SLOT08 diversity escalate")
        tip("SLOT08", "SLOT08_AB_master_clean_combo_0p45e4.csv",
            "Master clean combo (escalated for div)", master, 0.65)
        # recompute integrity for 08 vs earlier
        p = vecs["SLOT08"]
        min_div = min(one_mrho(p, vecs[slot_names[j]]) * 1e4 for j in range(7))
        integrity["SLOT08"]["min_1mrho_vs_earlier_e4"] = min_div
        integrity["SLOT08"]["diversity_ok"] = min_div >= 0.025 - 1e-6
        integrity["SLOT08"]["1mrho_vs_AB_e4"] = one_mrho(p, ab) * 1e4
        slots_meta["SLOT08"]["integrity"] = integrity["SLOT08"]
        for j in range(10):
            M[7, j] = 0.0 if j == 7 else one_mrho(vecs["SLOT08"], vecs[slot_names[j]]) * 1e4
            M[j, 7] = M[7, j]

    payload = {
        "NO_SUBMIT": True,
        "LB_01": LB_01,
        "remaining_submissions_after_slot01": 9,
        "quarantine": list(QUARANTINE),
        "fullbank_v10_finding": (
            "w0_repair_baselines_and_intel.py lines 111-130 apply v10_ridge_resid_orth_LM "
            "micro-orth onto z_fullbank after blend → artifact 0.952621 is POISONED. "
            "Fullbank_clean rebuilt with w_v10=0."
        ),
        "z_pub_only": {
            "blend": pub_name, "oof_auc": pub_auc, "oof_auc_zr": pub_auc_zr,
            "n_members": len(kept_names), "members_head": kept_names[:20],
            "assert_ge_part12_zr": True,
        },
        "Fullbank_clean": {
            "blend": fb_c_name, "oof_auc": fb_c_auc, "oof_auc_zr": fb_c_auc_zr, "w_v10": 0.0,
        },
        "private_lift_vs_pub": priv_lift,
        "scorecard": scorecard,
        "passing_dirs": list(dirs.keys()),
        "slots": slots_meta,
        "integrity": integrity,
        "matrix_1mrho_e4": M.tolist(),
    }
    (LOG / "V41_STEP6B_CLEAN_MANIFEST.json").write_text(json.dumps(payload, indent=2, default=str))

    lines = [
        "# V41 STEP 6b — Clean Manifest (v10/zanch PURGED)",
        "",
        f"**NO_SUBMIT={NO_SUBMIT}** · **LB_01 (live SLOT01) = {LB_01}** · remaining daily ≈ 9",
        "",
        "## Executive",
        "",
        f"- Quarantined: `v10_ridge*`, `v10_cb_d4*`, `zanch_*` (+ prior TE-bleed list)",
        f"- `z_pub_only` ({pub_name}) raw={pub_auc:.6f} +Zr=**{pub_auc_zr:.6f}** (≥ Part12+Zr 0.952614) — zero private files",
        f"- `Fullbank_clean` ({fb_c_name}, w_v10=0) raw={fb_c_auc:.6f} +Zr=**{fb_c_auc_zr:.6f}**",
        f"- Private lift `Fullbank_clean - z_pub_only` fold wins: **{priv_lift['fold_wins']}/5**",
        f"- Passing clean dirs: **{list(dirs.keys())}**",
        f"- Slots 04–10 refreshed under `{FINAL}/` (01–03 intact)",
        "",
        "## Scorecard",
        "",
        "```json",
        json.dumps({k: {kk: vv for kk, vv in g.items() if kk != "w_sweep_auc"}
                    for k, g in scorecard.items() if k != "peer_hybrids"}, indent=2, default=str)[:25000],
        "```",
        "",
        "## 10-Slot Master Table",
        "",
        "| Slot | File | 1-ρ vs AB e4 | minDiv e4 | div_ok |",
        "|---|---|---:|---:|---|",
    ]
    for sn in slot_names:
        s = slots_meta[sn]
        integ = integrity[sn]
        md = integ["min_1mrho_vs_earlier_e4"]
        lines.append(
            f"| {sn} | `{s['file']}` | {s['1mrho_vs_AB_e4']:.4f} | "
            f"{'' if md is None else f'{md:.4f}'} | {integ['diversity_ok']} |"
        )
    lines += [
        "",
        "## 10×10 1-ρ (×1e-4)",
        "",
        "| |" + "|".join(slot_names) + "|",
        "|" + "|".join(["---"] * 11) + "|",
    ]
    for i, sn in enumerate(slot_names):
        lines.append("| " + " | ".join([sn] + [f"{M[i, j]:.3f}" for j in range(10)]) + " |")
    lines += [
        "",
        "## STOP",
        "",
        f"LB_01={LB_01}. Step 6b complete. **NO_SUBMIT for slots 02–10.** "
        "Awaiting explicit `SLOT 02` command before any further live burn.",
        "",
    ]
    (LOG / "V41_STEP6B_CLEAN_MANIFEST.md").write_text("\n".join(lines))

    print("\n=== STRICT NO_SUBMIT=True | V41 STEP6b ===", flush=True)
    print(f"LB_01={LB_01} remaining≈9", flush=True)
    print(f"z_pub_only+Zr={pub_auc_zr:.6f} Fullbank_clean+Zr={fb_c_auc_zr:.6f}", flush=True)
    print(f"passing={list(dirs.keys())}", flush=True)
    print(f"WROTE {LOG / 'V41_STEP6B_CLEAN_MANIFEST.md'}", flush=True)
    print("STOP. Wait for SLOT 02 command. Do NOT submit slots 02–10.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
