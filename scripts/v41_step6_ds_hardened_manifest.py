#!/usr/bin/env python3
"""V41 STEP 6 — DS-Expert Hardened 10-Slot Final Manifest.

STRICT NO_SUBMIT=True. Hard stop after Step 6. ZERO submissions.
Fixes forensic flaws from Steps 1–5b:
  - Residual columns no longer killed by solo-AUC≥0.905
  - 52-col bank collapsed to family macro-pillars (Yash / V33.2)
  - Dual-baseline invariance + AB-twin + P⊥ ladder null-space
  - Concave trust-region tip climbs on Step_AB
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

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
SSD = Path("/tmp/s6e11_v39_ssd")
EXT = ROOT / "external/v41_nov23_frontier"
GPW = ROOT / "external/banks/goodpjw2008_s6e11-oof-bank"
BUSY = EXT / "busyaprime_52"
PART12 = ROOT / "external/nov08_part12"
LOG = ROOT / "logs/v41"
STAGED = ROOT / "out/v41_staged"
FINAL = ROOT / "out/v41_final_10slots"
PRED = ROOT / "preds"
DATA = ROOT / "data" if (ROOT / "data/test.csv").exists() else SSD / "data"
FOLDS_P = SSD / "folds/folds.csv" if (SSD / "folds/folds.csv").exists() else ROOT / "folds/folds.csv"

SHA_AB = "3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038"
SHA_AA = "51706855fa6de98ef88c1d02bebfc24ab340092d05eb513e227e048041c0346c"
SHA_Z = "ee61b50611326eb668033f088f8803e28541874c1cff2cf073334ea7d0f05a11"
SHA_AREEB68 = "3b9fe6d72e3102101ab33d35a17cdb90b1e41de40c3792a7dc1a1cd619d1bc8c"

QUARANTINE = (
    "uid_catboost", "v41_d27_eb", "fintfm", "fin_tfm", "zanch",
    "v10_lgb_d3_bag5", "v10_lgb", "gb_soft_bag3", "soft_diff",
    "idf_p3", "clean035", "fold0_only", "1fold", "f0_only",
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
    uref = U(u_ref) if u_ref is not None else U(scores)
    # if u_ref already uniform ranks, keep order
    if np.min(uref) >= 0 and np.max(uref) <= 1.0 and abs(uref.mean() - 0.5) < 0.05:
        pass
    else:
        uref = U(u_ref)
    if zones.any() and (~zones).any():
        floor = float(np.min(out[~zones])) - 1.0
        out[zones] = floor + uref[zones]
    return U(out)


def clip_p985(v):
    v = np.asarray(v, float).ravel()
    lim = float(np.quantile(np.abs(v), 0.985))
    return np.clip(v, -lim, lim)


def is_quar(name: str, path: str = "") -> bool:
    s = f"{name} {path}".lower()
    return any(q in s for q in QUARANTINE)


def family_of(name: str, hint: str = "") -> str:
    s = f"{name} {hint}".lower()
    if any(x in s for x in ("gpt2", "compact", "target-first", "l7", "lm_", "megayak")):
        return "LM_GPT2"
    if any(x in s for x in ("tabm", "realmlp", "rmlp", "resmlp", "neural")):
        return "TabM_RealMLP"
    if any(x in s for x in (
        "lgb", "xgb", "catboost", "gbdt", "gb_", "gbc", "gbe", "gbx", "gbn", "gbr",
        "lightgbm", "booster", "glm-boost", "glm-start",
    )):
        return "GBDT"
    if any(x in s for x in ("ebm", "slr", "linear", "logistic")) or (
        "glm" in s and "boost" not in s
    ):
        return "EBM_GLM"
    return "Other"


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


def residual_hygiene(oof, te) -> dict:
    oof = np.asarray(oof, float).ravel()
    te = np.asarray(te, float).ravel()
    ok = oof.shape == (N_TR,) and te.shape == (N_TE,)
    nan = int(np.isnan(oof).sum() + np.isinf(oof).sum() + np.isnan(te).sum() + np.isinf(te).sum())
    disp = float(te.std() / (oof.std() + 1e-12)) if ok else float("nan")
    ks = float(ks_2samp(oof, te).statistic) if ok and nan == 0 else float("nan")
    # PCHIP-ish: if discrete few unique, relax — we just report
    nuniq = len(np.unique(np.round(oof, 8)))
    pass_h = ok and nan == 0 and 0.88 <= disp <= 1.12 and ks < 0.020
    return {"disp": disp, "KS": ks, "nan": nan, "nuniq": nuniq, "pass": pass_h, "ok_shape": ok}


def base_hygiene(oof, te, y, folds) -> dict:
    oof = np.asarray(oof, float).ravel()
    te = np.asarray(te, float).ravel()
    ok = oof.shape == (N_TR,) and te.shape == (N_TE,)
    nan = int(np.isnan(oof).sum() + np.isinf(oof).sum() + np.isnan(te).sum() + np.isinf(te).sum())
    auc = float(roc_auc_score(y, oof)) if ok and nan == 0 else float("nan")
    ks_cal = float(ks_2samp(L_fold(oof, folds), L(te)).statistic) if ok and nan == 0 else float("nan")
    pass_h = ok and nan == 0 and 0.905 <= auc <= 0.960 and ks_cal <= 0.045
    return {"auc": auc, "cal_KS": ks_cal, "pass": pass_h, "nan": nan}


def load_public_bank(y, folds) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """gpw70 + busy52 clean public columns (B_52-ish)."""
    rows, oofs, tes = [], [], []

    def add(name, hint, o, t, source, min_auc=0.945, tabm_ex=False):
        if is_quar(name, source):
            rows.append({"name": name, "family": family_of(name, hint), "status": "QUARANTINE"})
            return
        o, t = np.asarray(o, float).ravel(), np.asarray(t, float).ravel()
        if o.shape != (N_TR,) or t.shape != (N_TE,):
            rows.append({"name": name, "status": "SHAPE"})
            return
        h = base_hygiene(o, t, y, folds)
        fam = family_of(name, hint)
        # TabM diversity: allow lower solo AUC floor
        auc_ok = (h["auc"] >= 0.90) if (tabm_ex or fam == "TabM_RealMLP") else (h["auc"] >= min_auc)
        if h["pass"] and auc_ok:
            rows.append({"name": name, "family": fam, "status": "CAND", **h})
            oofs.append(o)
            tes.append(t)
        else:
            rows.append({"name": name, "family": fam, "status": "FAIL", **h})

    man = pd.read_csv(GPW / "manifest.csv")
    for _, r in man.iterrows():
        n = str(r["name"])
        op, tp = GPW / f"oof_{n}.npy", GPW / f"test_{n}.npy"
        if op.exists() and tp.exists():
            add(f"gpw70:{n}", str(r.get("family", "")), np.load(op), np.load(tp), "gpw70")

    bman = pd.read_csv(BUSY / "manifest.csv")
    Bo, Bt = np.load(BUSY / "oof.npy"), np.load(BUSY / "test.npy")
    for i, r in bman.iterrows():
        n = str(r["name"])
        if "fintfm" in n.lower():
            rows.append({"name": f"busy52:{n}", "status": "QUARANTINE"})
            continue
        add(f"busy52:{n}", str(r.get("family", "")), Bo[:, i], Bt[:, i], "busy52",
            tabm_ex=("tabm" in n.lower() or "realmlp" in n.lower() or "rmlp" in n.lower()))
    if (BUSY / "tabm_oof.npy").exists():
        add("busy52:tabm_standalone", "tabm", np.load(BUSY / "tabm_oof.npy"),
            np.load(BUSY / "tabm_test.npy"), "tabm", tabm_ex=True)

    Xo = np.column_stack(oofs) if oofs else np.zeros((N_TR, 0))
    Xt = np.column_stack(tes) if tes else np.zeros((N_TE, 0))
    return rows, Xo, Xt


def nested_family_pillar(Xo, Xt, y, folds, idxs) -> tuple[np.ndarray, np.ndarray]:
    """Inner-fold LogReg on family subset → OOF/test pillar."""
    if not idxs:
        return np.zeros(N_TR), np.zeros(N_TE)
    Fo = np.column_stack([L_fold(Xo[:, j], folds) for j in idxs])
    Ft = np.column_stack([L(Xt[:, j]) for j in idxs])
    oof = np.zeros(N_TR)
    te = np.zeros(N_TE)
    for f in range(5):
        tr, va = folds != f, folds == f
        # pick C on inner 4
        uniq = sorted(np.unique(folds[tr]))
        best_c, best_sc = 0.1, -1.0
        for C in (0.03, 0.1, 0.3, 1.0):
            scs = []
            inner = folds[tr]
            Xtr, ytr = Fo[tr], y[tr]
            for u in uniq:
                it, iv = inner != u, inner == u
                clf = LogisticRegression(C=C, solver="lbfgs", max_iter=250, tol=1e-3)
                clf.fit(Xtr[it], ytr[it])
                scs.append(roc_auc_score(ytr[iv], clf.predict_proba(Xtr[iv])[:, 1]))
            sc = float(np.mean(scs))
            if sc > best_sc:
                best_sc, best_c = sc, C
        clf = LogisticRegression(C=best_c, solver="lbfgs", max_iter=400)
        clf.fit(Fo[tr], y[tr])
        oof[va] = clf.predict_proba(Fo[va])[:, 1]
        te += clf.predict_proba(Ft)[:, 1] / 5
    return oof, te


def load_prop_assets(y, folds) -> dict:
    """Scan proprietary bases + residuals; residual-aware hygiene."""
    assets = {}

    def put(key, kind, o, t, meta=None):
        o, t = np.asarray(o, float).ravel(), np.asarray(t, float).ravel()
        if is_quar(key):
            assets[key] = {"kind": kind, "status": "QUARANTINE"}
            return
        if kind == "residual":
            h = residual_hygiene(o, t)
        else:
            h = base_hygiene(o, t, y, folds)
            # residual-aware: if base AUC low but looks like residual, reclassify
            if not h["pass"] and h.get("auc", 1) < 0.905:
                hr = residual_hygiene(o, t)
                if hr["pass"]:
                    h = {**hr, "auc": h.get("auc"), "reclass": "residual"}
                    kind = "residual"
        assets[key] = {"kind": kind, "oof": o, "test": t, "status": "OK" if h["pass"] else "FAIL", **h, **(meta or {})}

    # bases
    put("prop:xgb_llr", "base",
        np.load(PRED / "paradigm_B/xgb_llr_oof.npy"),
        np.load(PRED / "paradigm_B/xgb_llr_test.npy"))

    lo = np.load(PRED / "l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    lt = np.load(PRED / "l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    l7o = (lo[:, :, 1] - lo[:, :, 0]).mean(1)
    l7t = ((lt[..., 1] - lt[..., 0]).mean(2)).mean(0)
    put("prop:l7_s2026", "base", l7o, l7t)

    gdir = PRED / "gpt2_med_4ep"
    if all((gdir / f"oof_{k}.npy").exists() for k in range(5)):
        go = np.zeros(N_TR)
        gt = []
        for k in range(5):
            a = np.load(gdir / f"oof_{k}.npy")
            b = np.load(gdir / f"test_{k}.npy")
            go[folds == k] = (a[:, :, 1] - a[:, :, 0]).mean(1)
            gt.append((b[:, :, 1] - b[:, :, 0]).mean(1))
        put("prop:gpt2_med_4ep", "base", go, np.mean(gt, 0))

    # lgb_gpt2 variants
    for tag, op, tp in [
        ("lgb_gpt2_3ep_fixed", PRED / "gpt2_3ep/gb_gpt2_3ep_super_oof.npy", PRED / "gpt2_3ep/gb_gpt2_3ep_super_test.npy"),
        ("gb_gpt2_bag5", PRED / "gb_gpt2_bag5_oof.npy", PRED / "gb_gpt2_joint_test.npy"),
    ]:
        if op.exists() and tp.exists():
            put(f"prop:{tag}", "base", np.load(op), np.load(tp))

    # residuals — V34.3 Bug#2: do NOT require AUC≥0.905
    resid_specs = [
        ("resid:v10_ridge_resid",
         PRED / "v10_transductive_bulsah/v10_ridge_resid_oof.npy",
         PRED / "v10_transductive_bulsah/v10_ridge_resid_test.npy"),
        ("resid:v10_ridge_orth_LM",
         PRED / "v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy",
         PRED / "v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"),
        ("resid:harvest_p4_cuda_xgb",
         PRED / "matrix_v2/harvest_p4_cuda_xgb_oof.npy",
         PRED / "matrix_v2/harvest_p4_cuda_xgb_test.npy"),
        ("resid:P4_L4.2_pure5f",
         PRED / "matrix_v2/P4_L4.2_pure5f_oof.npy",
         PRED / "matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("resid:P4_L4.3_pure5f",
         PRED / "matrix_v2/P4_L4.3_pure5f_oof.npy",
         PRED / "matrix_v2/P4_L4.3_pure5f_test.npy"),
        ("resid:P4_L4.5_pure5f",
         PRED / "matrix_v2/P4_L4.5_pure5f_oof.npy",
         PRED / "matrix_v2/P4_L4.5_pure5f_test.npy"),
    ]
    # optional named residuals
    for pat in ("R2_v178", "F_cat_obliv", "U5_gini10_llr", "U6_pair_std5_llr", "U7_fwd_rev_gap", "X_tab45"):
        hits = list(PRED.rglob(f"*{pat}*oof*.npy"))[:3]
        for op in hits:
            tp = Path(str(op).replace("_oof", "_test"))
            if not tp.exists():
                tp = op.parent / op.name.replace("oof", "test")
            if tp.exists():
                resid_specs.append((f"resid:{pat}", op, tp))
                break

    for key, op, tp in resid_specs:
        if op.exists() and tp.exists():
            put(key, "residual", np.load(op), np.load(tp), {"path_oof": str(op)})
        else:
            assets[key] = {"kind": "residual", "status": "MISSING"}

    # uid_v2 as base (will be orth-tested separately)
    uo, ut = SSD / "out/uid_v2_oof.npy", SSD / "out/uid_v2_test.npy"
    if uo.exists() and ut.exists():
        put("prop:uid_v2", "base", np.load(uo), np.load(ut))

    # HARD BAN markers
    assets["prop:uid_catboost"] = {"kind": "base", "status": "HARD_BAN"}
    return assets


def lofo_orth_vs_bank(target_o, target_t, Bo, Bt, folds, ridge=50.0):
    """LOFO-orthogonalize target vs B bank on OOF; apply train-fit betas averaged on test."""
    Mo = np.column_stack([unit(L_fold(Bo[:, j], folds)) for j in range(Bo.shape[1])])
    Mt = np.column_stack([unit(L(Bt[:, j])) for j in range(Bt.shape[1])])
    yo = unit(L_fold(target_o, folds) if target_o.std() > 1e-9 else target_o)
    # if already residual-like, still orth
    oof = np.zeros(N_TR)
    te = np.zeros(N_TE)
    for f in range(5):
        tr, va = folds != f, folds == f
        X, yv = Mo[tr], yo[tr]
        beta = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ yv)
        oof[va] = unit(yo[va] - Mo[va] @ beta)
        te += unit(unit(L(target_t)) - Mt @ beta) / 5
    return unit(oof), unit(te)


def dual_gate(u_o, u_t, y, folds, base_fb, base_twin, Bo_t, zones_te) -> dict:
    """LOFO signed lift on fullbank+Zr and twin+Zr; hygiene; R² vs B_test."""
    def signed_lofo(base):
        best = {"sign": 1, "pos": 0, "mean_e5": -1e9, "sf25": 0, "t25": 0.0, "folds": []}
        for sgn in (1, -1):
            lifts = []
            oof = np.zeros(N_TR)
            for f in range(5):
                tr, va = folds != f, folds == f
                # pick tiny alpha on train via AUC
                best_a, best_sc = 0.0, roc_auc_score(y[tr], base[tr])
                for a in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
                    sc = roc_auc_score(y[tr], U(L(base[tr]) + sgn * a * u_o[tr]))
                    if sc > best_sc:
                        best_sc, best_a = sc, a
                pred = U(L(base[va]) + sgn * best_a * u_o[va])
                oof[va] = pred
                lifts.append((roc_auc_score(y[va], pred) - roc_auc_score(y[va], base[va])) * 1e5)
            pos = sum(1 for x in lifts if x > 0)
            mean_e5 = float(np.mean(lifts))
            # sf25 vs base
            sf = sf25_beats(y, oof, base, folds)
            t25 = float(mean_e5 / (np.std(lifts) / np.sqrt(5) + 1e-12)) if len(lifts) else 0.0
            if (pos, mean_e5) > (best["pos"], best["mean_e5"]):
                best = {"sign": sgn, "pos": pos, "mean_e5": mean_e5, "sf25": sf,
                        "t25": t25, "folds": lifts, "oof": oof}
        return best

    g_fb = signed_lofo(base_fb)
    g_tw = signed_lofo(base_twin)
    # require SAME sign
    same_sign = g_fb["sign"] == g_tw["sign"]
    disp = float(u_t.std() / (u_o.std() + 1e-12))
    ks = float(ks_2samp(u_o, u_t).statistic)
    # R² of u_test ~ B_test
    Mt = np.column_stack([unit(L(Bo_t[:, j])) for j in range(min(Bo_t.shape[1], 40))])  # cap cols for speed
    yt = unit(u_t)
    try:
        beta = np.linalg.lstsq(Mt, yt, rcond=None)[0]
        r2 = float(1.0 - np.sum((yt - Mt @ beta) ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-12))
    except Exception:
        r2 = float("nan")
    pass_gate = (
        same_sign
        and g_fb["pos"] >= 4 and g_tw["pos"] >= 4
        and g_fb["mean_e5"] > 0 and g_tw["mean_e5"] > 0
        and 0.85 <= disp <= 1.15
        and ks < 0.05
    )
    return {
        "fullbank": {k: g_fb[k] for k in ("sign", "pos", "mean_e5", "sf25", "t25", "folds")},
        "twin": {k: g_tw[k] for k in ("sign", "pos", "mean_e5", "sf25", "t25", "folds")},
        "same_sign": same_sign,
        "raw_disp": disp,
        "KS": ks,
        "R2_u_test_on_B52": r2,
        "pass": pass_gate,
        "sign": g_fb["sign"] if same_sign else 0,
    }


def null_space_P_perp(u_te, d_ab_z_te, zones):
    """Zero out negative projection onto d_AB_Z on non-zone test."""
    m = ~zones
    d = unit(d_ab_z_te[m])
    u = u_te.copy()
    um = unit(u[m])
    proj = float(np.dot(um, d))
    cos = proj  # since both unit
    if proj < 0:
        um = um - proj * d
        u[m] = um * (u[m].std() + 1e-12) + u[m].mean()  # restore scale roughly
        u[m] = unit(u[m])
    return unit(u), {"cos_d_AB_Z": cos, "projected": proj < 0}


def scale_to_target_1mrho(base_u, direction, target_e4, zones, u_ab, lo=1e-6, hi=5.0):
    """Binary search α: 1-ρ(U(Zr(L(base)+α*dir)), base)*1e4 ≈ target_e4."""
    base_u = U(base_u)
    direction = np.asarray(direction, float).ravel()
    target = target_e4 * 1e-4
    best_a, best_diff = 0.0, 1e9
    best_p = base_u
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        raw = L(base_u) + mid * direction
        p = apply_zr(raw, zones, u_ab)
        d = one_mrho(p, base_u)
        if abs(d - target) < best_diff:
            best_diff, best_a, best_p = abs(d - target), mid, p
        if d < target:
            lo = mid
        else:
            hi = mid
    return best_p, best_a, one_mrho(best_p, base_u) * 1e4


def combine_dirs(dirs_o, dirs_t, y, folds, weights=None):
    if not dirs_o:
        return np.zeros(N_TR), np.zeros(N_TE)
    if weights is None:
        weights = np.ones(len(dirs_o)) / len(dirs_o)
    wo = sum(w * unit(d) for w, d in zip(weights, dirs_o))
    wt = sum(w * unit(d) for w, d in zip(weights, dirs_t))
    return unit(wo), unit(wt)


def main():
    assert NO_SUBMIT is True
    LOG.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    log("STEP6 DS-hardened start NO_SUBMIT=True")

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(float)
    folds = pd.read_csv(FOLDS_P)["fold"].to_numpy(int)
    te_df = pd.read_csv(DATA / "test.csv")
    tr_df = pd.read_csv(DATA / "train.csv")
    ids = te_df["id"].to_numpy()
    Zg_tr = z_goodpjw(tr_df)
    Zg_te = z_goodpjw(te_df)
    assert int(Zg_te.sum()) == 16_730

    # tips
    ab_path = STAGED / "F1b_pub_Step_AB_095569.csv"
    aa_path = STAGED / "F1_pub_Step_AA_095569.csv"
    z_path = EXT / "goodpjw_95569/submission_step_z.csv"
    if not z_path.exists():
        z_path = EXT / "pinned_csv/Step_Z_095562_from_Tensorflow.csv"
    areeb_path = EXT / "areeb_95568/submission.csv"
    quarter_path = STAGED / "quarter_our_095567.csv"
    assert sha256_file(ab_path) == SHA_AB
    assert sha256_file(aa_path) == SHA_AA
    assert sha256_file(z_path) == SHA_Z
    assert sha256_file(areeb_path) == SHA_AREEB68
    ab = pd.read_csv(ab_path)["is_fraud"].to_numpy(float)
    aa = pd.read_csv(aa_path)["is_fraud"].to_numpy(float)
    tip_z = pd.read_csv(z_path)["is_fraud"].to_numpy(float)
    areeb = pd.read_csv(areeb_path)["is_fraud"].to_numpy(float)
    quarter = pd.read_csv(quarter_path)["is_fraud"].to_numpy(float)
    u_ab = U(ab)

    fb_o = np.load(SSD / "preds/z_fullbank_oof.npy").ravel()
    fb_t = np.load(SSD / "preds/z_fullbank_test.npy").ravel()
    fb_o_zr = apply_zr(fb_o, Zg_tr, U(fb_o))
    fb_t_zr = apply_zr(fb_t, Zg_te, u_ab)
    p12_o = np.load(PART12 / "wide_lm_stack_nested.npy").ravel()
    p12_o_zr = apply_zr(p12_o, Zg_tr, U(p12_o))
    log(f"baselines fb+Zr={roc_auc_score(y, fb_o_zr):.6f} p12+Zr={roc_auc_score(y, p12_o_zr):.6f}")

    # ── §1 public bank → family pillars ──
    bank_rows, Bo, Bt = load_public_bank(y, folds)
    cand = [r for r in bank_rows if r.get("status") == "CAND"]
    log(f"public bank CAND={len(cand)} shape={Bo.shape}")
    families = ["LM_GPT2", "GBDT", "TabM_RealMLP", "EBM_GLM"]
    pillars_o, pillars_t, pillar_meta = {}, {}, {}
    for fam in families:
        idxs = [i for i, r in enumerate(cand) if r["family"] == fam]
        if not idxs:
            continue
        # cap per family for speed
        idxs = sorted(idxs, key=lambda i: cand[i]["auc"], reverse=True)[:18]
        po, pt = nested_family_pillar(Bo, Bt, y, folds, idxs)
        pillars_o[fam] = po
        pillars_t[fam] = pt
        pillar_meta[fam] = {"n": len(idxs), "auc": float(roc_auc_score(y, po)),
                            "members": [cand[i]["name"] for i in idxs[:8]]}
        log(f"  pillar {fam}: n={len(idxs)} OOF={pillar_meta[fam]['auc']:.6f}")

    # S_indep_pillars = nested blend of family pillars (NO Part12)
    fam_names = list(pillars_o.keys())
    if fam_names:
        Po = np.column_stack([L_fold(pillars_o[f], folds) for f in fam_names])
        Pt = np.column_stack([L(pillars_t[f]) for f in fam_names])
        S_ind_o = np.zeros(N_TR)
        S_ind_t = np.zeros(N_TE)
        for f in range(5):
            tr, va = folds != f, folds == f
            clf = LogisticRegression(C=0.3, solver="lbfgs", max_iter=400)
            clf.fit(Po[tr], y[tr])
            S_ind_o[va] = clf.predict_proba(Po[va])[:, 1]
            S_ind_t += clf.predict_proba(Pt)[:, 1] / 5
    else:
        S_ind_o, S_ind_t = fb_o.copy(), fb_t.copy()
    S_ind_o_zr = apply_zr(S_ind_o, Zg_tr, U(S_ind_o))
    div_ind = {
        "1mrho_oof_e4": one_mrho(S_ind_o, fb_o) * 1e4,
        "1mrho_test_e4": one_mrho(S_ind_t, fb_t) * 1e4,
    }
    div_ind["ratio_test_over_oof"] = div_ind["1mrho_test_e4"] / max(div_ind["1mrho_oof_e4"], 1e-12)
    log(f"S_indep vs Fullbank 1-ρ oof/test e4={div_ind['1mrho_oof_e4']:.3f}/{div_ind['1mrho_test_e4']:.3f} "
        f"ratio={div_ind['ratio_test_over_oof']:.2f} AUC={roc_auc_score(y, S_ind_o):.6f}")

    # ── prop assets ──
    assets = load_prop_assets(y, folds)
    for k, v in assets.items():
        if v.get("status") in ("OK", "FAIL", "MISSING", "HARD_BAN", "QUARANTINE"):
            extra = ""
            if "auc" in v:
                extra += f" auc={v['auc']:.5f}" if v["auc"] == v["auc"] else ""
            if "disp" in v:
                extra += f" disp={v['disp']:.3f} KS={v.get('KS', float('nan')):.4f}"
            log(f"  asset {k}: {v.get('kind')} {v.get('status')}{extra}")

    ok_bases = {k: v for k, v in assets.items() if v.get("status") == "OK" and v.get("kind") == "base"}
    ok_res = {k: v for k, v in assets.items() if v.get("status") == "OK" and v.get("kind") == "residual"}
    log(f"OK bases={list(ok_bases)} OK residuals={list(ok_res)}")

    # ── S_cv_champ: nested meta on [Fullbank, S_indep, bases, residuals±] ──
    meta_cols_o = [fb_o, S_ind_o]
    meta_cols_t = [fb_t, S_ind_t]
    meta_names = ["Fullbank", "S_indep_pillars"]
    for k, v in ok_bases.items():
        if k == "prop:uid_v2":
            continue  # handled via orth gate only
        meta_cols_o.append(v["oof"])
        meta_cols_t.append(v["test"])
        meta_names.append(k)
    for k, v in ok_res.items():
        meta_cols_o.append(v["oof"])
        meta_cols_t.append(v["test"])
        meta_names.append(k)

    Mo = np.column_stack([L_fold(c, folds) for c in meta_cols_o])
    Mt = np.column_stack([L(c) for c in meta_cols_t])
    # sign law for residual-like cols: try ± on residual names inside nested fit via feature duplication of flipped
    # Efficient: for residual cols, pick sign by inner CV on train
    S_cv_o = np.zeros(N_TR)
    S_cv_t = np.zeros(N_TE)
    sign_logs = []
    for f in range(5):
        tr, va = folds != f, folds == f
        Xtr = Mo[tr].copy()
        Xva = Mo[va].copy()
        Xte = Mt.copy()
        # pick signs for residual columns
        signs = {}
        for j, name in enumerate(meta_names):
            if not name.startswith("resid:"):
                signs[name] = 1
                continue
            best_s, best_sc = 1, -1.0
            inner = folds[tr]
            for s in (1, -1):
                scs = []
                col = Xtr[:, j] * s
                for u in np.unique(inner):
                    it, iv = inner != u, inner == u
                    # univariate AUC of signed col as quick proxy
                    scs.append(roc_auc_score(y[tr][iv], col[iv]))
                sc = float(np.mean(scs))
                if sc > best_sc:
                    best_sc, best_s = sc, s
            signs[name] = best_s
            Xtr[:, j] *= best_s
            Xva[:, j] *= best_s
            Xte[:, j] *= best_s
        sign_logs.append({"fold": f, "signs": signs})
        # C grid
        best_c, best_sc = 0.1, -1.0
        inner = folds[tr]
        for C in (0.03, 0.1, 0.3, 1.0):
            scs = []
            for u in np.unique(inner):
                it, iv = inner != u, inner == u
                clf = LogisticRegression(C=C, solver="lbfgs", max_iter=300)
                clf.fit(Xtr[it], y[tr][it])
                scs.append(roc_auc_score(y[tr][iv], clf.predict_proba(Xtr[iv])[:, 1]))
            sc = float(np.mean(scs))
            if sc > best_sc:
                best_sc, best_c = sc, C
        clf = LogisticRegression(C=best_c, solver="lbfgs", max_iter=500)
        clf.fit(Xtr, y[tr])
        S_cv_o[va] = clf.predict_proba(Xva)[:, 1]
        S_cv_t += clf.predict_proba(Xte)[:, 1] / 5
        log(f"  S_cv fold {f}: C={best_c} inner={best_sc:.5f} signs_resid="
            f"{ {k:v for k,v in signs.items() if k.startswith('resid:')} }")

    S_cv_o_zr = apply_zr(S_cv_o, Zg_tr, U(S_cv_o))
    S_cv_t_zr = apply_zr(S_cv_t, Zg_te, u_ab)
    cv_auc = float(roc_auc_score(y, S_cv_o))
    cv_auc_zr = float(roc_auc_score(y, S_cv_o_zr))
    cv_report = {
        "oof_auc": cv_auc,
        "oof_auc_zr": cv_auc_zr,
        "folds": fold_aucs(y, S_cv_o, folds),
        "folds_zr": fold_aucs(y, S_cv_o_zr, folds),
        "sf25_vs_fb_zr": sf25_beats(y, S_cv_o_zr, fb_o_zr, folds),
        "sf25_vs_p12_zr": sf25_beats(y, S_cv_o_zr, p12_o_zr, folds),
        "meta_names": meta_names,
        "sign_logs": sign_logs,
    }
    log(f"S_cv_champ raw={cv_auc:.6f} +Zr={cv_auc_zr:.6f} "
        f"sf25 vs fb={cv_report['sf25_vs_fb_zr']}/25 vs p12={cv_report['sf25_vs_p12_zr']}/25")

    # ── §2 z_AB_twin ──
    # B_52_test on non-zone → predict L(AB); apply to OOF
    use_n = min(Bo.shape[1], 52)
    B52o = Bo[:, :use_n]
    B52t = Bt[:, :use_n]
    # rebuild with actual cand order (already)
    Xte_nz = np.column_stack([L(B52t[:, j]) for j in range(use_n)])[~Zg_te]
    y_ab_nz = L(ab)[~Zg_te]
    ridge = Ridge(alpha=100.0, fit_intercept=True)
    ridge.fit(Xte_nz, y_ab_nz)
    pred_te_all = ridge.predict(np.column_stack([L(B52t[:, j]) for j in range(use_n)]))
    twin_te = U(pred_te_all)
    # R² non-zone
    r2_te = float(1 - np.sum((y_ab_nz - ridge.predict(Xte_nz)) ** 2) /
                  (np.sum((y_ab_nz - y_ab_nz.mean()) ** 2) + 1e-12))
    # OOF twin with frozen weights on L_fold
    Xo_f = np.column_stack([L_fold(B52o[:, j], folds) for j in range(use_n)])
    twin_o_raw = Xo_f @ ridge.coef_ + ridge.intercept_
    twin_o = U(twin_o_raw)
    twin_o_zr = apply_zr(twin_o, Zg_tr, U(twin_o))
    twin_te_zr = apply_zr(twin_te, Zg_te, u_ab)
    twin_report = {
        "R2_test_nonzone": r2_te,
        "1mrho_test_vs_AB_e4": one_mrho(twin_te_zr, ab) * 1e4,
        "oof_auc_zr": float(roc_auc_score(y, twin_o_zr)),
        "n_cols": use_n,
    }
    log(f"z_AB_twin R²_nz={r2_te:.4f} 1-ρ vs AB e4={twin_report['1mrho_test_vs_AB_e4']:.3f} "
        f"OOF+Zr={twin_report['oof_auc_zr']:.6f}")

    # ladder diffs
    d_AB_AA = unit(L(ab) - L(aa))
    d_AB_Z = unit(L(ab) - L(tip_z))

    # ── dual-gated orth directions ──
    gate_scorecard = {}
    dirs_o, dirs_t = {}, {}

    def register(tag, o_raw, t_raw):
        uo, ut = lofo_orth_vs_bank(o_raw, t_raw, B52o, B52t, folds)
        ut2, pinfo = null_space_P_perp(ut, d_AB_Z, Zg_te)
        g = dual_gate(uo, ut2, y, folds, fb_o_zr, twin_o_zr, B52t, Zg_te)
        g["P_perp"] = pinfo
        g["cos_d_AB_AA"] = float(np.dot(unit(ut2[~Zg_te]), unit(d_AB_AA[~Zg_te])))
        g["cos_d_AB_Z_after"] = float(np.dot(unit(ut2[~Zg_te]), unit(d_AB_Z[~Zg_te])))
        gate_scorecard[tag] = g
        if g["pass"]:
            sgn = g["sign"]
            dirs_o[tag] = sgn * uo
            dirs_t[tag] = sgn * ut2
        log(f"  GATE {tag}: pass={g['pass']} sign={g.get('sign')} "
            f"fb {g['fullbank']['pos']}/5 Δe5={g['fullbank']['mean_e5']:.2f} "
            f"twin {g['twin']['pos']}/5 Δe5={g['twin']['mean_e5']:.2f} "
            f"disp={g['raw_disp']:.3f} R²={g['R2_u_test_on_B52']:.3f}")
        return g

    # a) seq tree
    seq_parts = []
    for k in ("prop:gpt2_med_4ep", "prop:l7_s2026", "prop:xgb_llr"):
        if k in ok_bases:
            seq_parts.append(k)
    if seq_parts:
        # combine then orth
        co = sum(unit(L_fold(ok_bases[k]["oof"], folds)) for k in seq_parts)
        ct = sum(unit(L(ok_bases[k]["test"])) for k in seq_parts)
        register("u_seq_tree", co, ct)

    # b) residual pillars
    if ok_res:
        co = sum(unit(v["oof"]) for v in ok_res.values())
        ct = sum(unit(v["test"]) for v in ok_res.values())
        register("u_resid_p4_ridge", co, ct)
        for k, v in ok_res.items():
            register(f"u_sub_{k}", v["oof"], v["test"])

    # c) uid_v2 orth
    uid_pass = False
    if "prop:uid_v2" in assets and assets["prop:uid_v2"].get("status") == "OK":
        g = register("u_uid_orth", assets["prop:uid_v2"]["oof"], assets["prop:uid_v2"]["test"])
        uid_pass = g["pass"]
    else:
        gate_scorecard["u_uid_orth"] = {"pass": False, "note": "uid_v2 missing or fail hygiene"}

    # d) ownstack vs twin
    register("u_ownstack_diff",
             clip_p985(L(S_cv_o) - L(twin_o)),
             clip_p985(L(S_cv_t) - L(twin_te)))

    # master combo of passing dirs
    pass_tags = [k for k, g in gate_scorecard.items()
                 if g.get("pass") and k in dirs_o and not k.startswith("u_sub_")]
    # prefer primary tags
    primary = [t for t in ("u_seq_tree", "u_resid_p4_ridge", "u_uid_orth", "u_ownstack_diff") if t in dirs_o]
    if not primary:
        primary = pass_tags
    if primary:
        # precision weights ~ mean_e5
        ws = np.array([max(gate_scorecard[t]["fullbank"]["mean_e5"], 0.1) for t in primary])
        ws = ws / ws.sum()
        mo, mt = combine_dirs([dirs_o[t] for t in primary], [dirs_t[t] for t in primary], y, folds, ws)
        # re-gate master
        g = dual_gate(mo, mt, y, folds, fb_o_zr, twin_o_zr, B52t, Zg_te)
        mt2, pinfo = null_space_P_perp(mt, d_AB_Z, Zg_te)
        g["P_perp"] = pinfo
        g["members"] = primary
        g["weights"] = {t: float(w) for t, w in zip(primary, ws)}
        gate_scorecard["u_master"] = g
        if g["pass"]:
            dirs_o["u_master"] = g["sign"] * mo
            dirs_t["u_master"] = g["sign"] * mt2
        log(f"  GATE u_master: pass={g['pass']} members={primary}")
    else:
        gate_scorecard["u_master"] = {"pass": False, "members": []}
        log("  GATE u_master: no passing primary dirs")

    # ── §3 stage 10 slots ──
    slots = {}
    vecs = {}

    # SLOT01
    sha, p = write_slot(FINAL / "SLOT01_pub_Step_AB_095569.csv", ids, ab)
    assert sha == SHA_AB
    slots["SLOT01"] = {"file": "SLOT01_pub_Step_AB_095569.csv", "sha": sha,
                       "role": "PUB Tip Lock 0.95569", "1mrho_vs_AB_e4": 0.0}
    vecs["SLOT01"] = p

    # SLOT02 geodesic AB × areeb
    d_areeb = clip_p985(L(areeb) - L(ab))
    p2, a2, e2 = scale_to_target_1mrho(ab, d_areeb, 0.029, Zg_te, u_ab)
    # refine with alpha≈0.32 recipe then snap to trust region
    raw2 = L(ab) + 0.32 * d_areeb
    p2b = apply_zr(raw2, Zg_te, u_ab)
    e2b = one_mrho(p2b, ab) * 1e4
    if 0.015 <= e2b <= 0.85:
        p2, e2 = p2b, e2b
        a2 = 0.32
    sha, p = write_slot(FINAL / "SLOT02_AB_clip985_areeb_a0p32.csv", ids, p2)
    slots["SLOT02"] = {"file": "SLOT02_AB_clip985_areeb_a0p32.csv", "sha": sha,
                       "role": "Top-2 Geodesic Clip AB×areeb", "alpha": a2,
                       "1mrho_vs_AB_e4": e2}
    vecs["SLOT02"] = p

    # SLOT03 quarter
    d_q = clip_p985(L(quarter) - L(ab))
    p3, a3, e3 = scale_to_target_1mrho(ab, d_q, 0.037, Zg_te, u_ab)
    raw3 = L(ab) + 0.28 * d_q
    p3b = apply_zr(raw3, Zg_te, u_ab)
    e3b = one_mrho(p3b, ab) * 1e4
    if 0.015 <= e3b <= 0.85:
        p3, e3, a3 = p3b, e3b, 0.28
    # ensure diversity vs slot2
    if one_mrho(p3, vecs["SLOT02"]) * 1e4 < 0.020:
        p3, a3, e3 = scale_to_target_1mrho(ab, d_q, 0.055, Zg_te, u_ab)
    sha, p = write_slot(FINAL / "SLOT03_AB_clip985_quarter_a0p28.csv", ids, p3)
    slots["SLOT03"] = {"file": "SLOT03_AB_clip985_quarter_a0p28.csv", "sha": sha,
                       "role": "Quarter Diversity Clip on AB", "alpha": a3,
                       "1mrho_vs_AB_e4": e3,
                       "1mrho_vs_SLOT02_e4": one_mrho(p, vecs["SLOT02"]) * 1e4}
    vecs["SLOT03"] = p

    def tip_climb(name, fname, role, direction, target_e4):
        if direction is None:
            # fallback: tiny areeb geodesic
            direction = clip_p985(L(areeb) - L(ab))
            role = role + " [FALLBACK areeb geodesic]"
        p, a, e = scale_to_target_1mrho(ab, clip_p985(direction), target_e4, Zg_te, u_ab)
        # clamp trust region
        if e < 0.015:
            p, a, e = scale_to_target_1mrho(ab, clip_p985(direction), 0.025, Zg_te, u_ab)
        if e > 2.0:
            p, a, e = scale_to_target_1mrho(ab, clip_p985(direction), 0.85, Zg_te, u_ab)
        sha, pout = write_slot(FINAL / fname, ids, p)
        slots[name] = {"file": fname, "sha": sha, "role": role, "alpha": a,
                       "1mrho_vs_AB_e4": e}
        vecs[name] = pout
        return pout

    # SLOT04 / 05 seq
    seq_dir = dirs_t.get("u_seq_tree")
    tip_climb("SLOT04", "SLOT04_AB_nonpub_seq_llr_clip985_0p25e4.csv",
              "Dual-Gated seq+LLR residual conservative", seq_dir, 0.25)
    tip_climb("SLOT05", "SLOT05_AB_nonpub_seq_llr_clip985_0p55e4.csv",
              "Dual-Gated seq+LLR residual step-up", seq_dir, 0.55)

    # SLOT06 resid pillars
    resid_dir = dirs_t.get("u_resid_p4_ridge")
    # if master resid failed, try sum of passing sub-residuals
    if resid_dir is None:
        sub = [dirs_t[k] for k in dirs_t if k.startswith("u_sub_resid:")]
        if sub:
            resid_dir = unit(sum(unit(s) for s in sub))
    tip_climb("SLOT06", "SLOT06_AB_prop_resid_pillars_clip985_0p30e4.csv",
              "Dual-Gated pure residual pillars", resid_dir, 0.30)

    # SLOT07 uid_orth OR ownstack_diff
    if uid_pass and "u_uid_orth" in dirs_t:
        tip_climb("SLOT07", "SLOT07_AB_uid_orth_clip985_0p35e4.csv",
                  "Dual-Gated uid_v2 orth", dirs_t["u_uid_orth"], 0.35)
    else:
        tip_climb("SLOT07", "SLOT07_AB_cvchamp_vs_twin_clip985_0p35e4.csv",
                  "Own-stack vs AB-twin clipped residual",
                  dirs_t.get("u_ownstack_diff", clip_p985(L(S_cv_t_zr) - L(twin_te_zr))), 0.35)

    # SLOT08 master combo + slot2 geodesic
    if "u_master" in dirs_t:
        master = unit(0.7 * unit(dirs_t["u_master"]) + 0.3 * unit(clip_p985(L(areeb) - L(ab))))
    elif primary:
        master = unit(sum(unit(dirs_t[t]) for t in primary if t in dirs_t))
    else:
        master = clip_p985(L(areeb) - L(ab))
    tip_climb("SLOT08", "SLOT08_AB_master_dual_gated_combo_0p40e4.csv",
              "Dynamic A/B winner composite master ray", master, 0.40)

    # SLOT09 pure CV champ
    sha, p = write_slot(FINAL / "SLOT09_pure_nested_cv_champion_zr.csv", ids, S_cv_t_zr)
    slots["SLOT09"] = {"file": "SLOT09_pure_nested_cv_champion_zr.csv", "sha": sha,
                       "role": "Pure nested-CV dual-stack champion",
                       "oof_auc": cv_auc, "oof_auc_zr": cv_auc_zr,
                       "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4}
    vecs["SLOT09"] = p

    # SLOT10 private hedge 55 AB / 45 CV
    hedge = apply_zr(U(0.55 * L(ab) + 0.45 * L(S_cv_t_zr)), Zg_te, u_ab)
    sha, p = write_slot(FINAL / "SLOT10_private_hedge_55BestTip_45CVChamp_zr.csv", ids, hedge)
    slots["SLOT10"] = {"file": "SLOT10_private_hedge_55BestTip_45CVChamp_zr.csv", "sha": sha,
                       "role": "Private-LB Bayesian shrinkage hedge",
                       "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4}
    vecs["SLOT10"] = p

    # ── §4 integrity ──
    integrity = {}
    slot_names = [f"SLOT{i:02d}" for i in range(1, 11)]
    for i, sn in enumerate(slot_names):
        p = vecs[sn]
        assert p.shape == (N_TE,)
        assert np.isfinite(p).all()
        assert abs(p.min()) >= 0 and p.max() <= 1.0 + 1e-9
        # Zr: zone ranks should be low
        max_uz = float(p[Zg_te].max()) if Zg_te.any() else 0.0
        # pairwise min 1-ρ vs earlier
        min_div = 1e9
        for j in range(i):
            d = one_mrho(p, vecs[slot_names[j]]) * 1e4
            min_div = min(min_div, d)
        if i == 0:
            min_div = float("nan")
        ok_div = (i == 0) or (min_div >= 0.020 - 1e-6)
        # trust region for tip climbs slots 2-8
        e_ab = one_mrho(p, ab) * 1e4
        trust_ok = True
        if sn in ("SLOT02", "SLOT03", "SLOT04", "SLOT05", "SLOT06", "SLOT07", "SLOT08"):
            trust_ok = 0.015 - 1e-6 <= e_ab <= 2.0 + 1e-6
        integrity[sn] = {
            "rows": int(p.shape[0]),
            "max_U_zone": max_uz,
            "Zr_ok": max_uz <= 0.0558 + 1e-4,
            "min_1mrho_vs_earlier_e4": None if i == 0 else float(min_div),
            "diversity_ok": ok_div,
            "1mrho_vs_AB_e4": e_ab,
            "trust_ok": trust_ok,
            "uid_catboost_weight": 0.0,
        }
        slots[sn]["integrity"] = integrity[sn]
        log(f"  {sn}: 1-ρ AB e4={e_ab:.4f} maxUz={max_uz:.4f} "
            f"minDiv={min_div if i else 0:.4f} Zr_ok={integrity[sn]['Zr_ok']} div_ok={ok_div}")

    # 10x10 matrix
    M = np.zeros((10, 10))
    for i, a in enumerate(slot_names):
        for j, b in enumerate(slot_names):
            M[i, j] = 0.0 if i == j else one_mrho(vecs[a], vecs[b]) * 1e4

    # sequential feedback tree
    feedback = {
        "SLOT01": "LOCK tip. Always keep as pub baseline.",
        "SLOT02": "If SLOT02 ≥ SLOT01: geodesic helps → prefer SLOT02/08 path. If −1 tick: abort geodesic family, keep SLOT01.",
        "SLOT03": "If SLOT03 ≥ max(S01,S02): quarter diversity real. If drop: skip quarter branch.",
        "SLOT04": "If ≥ best tip: unlock SLOT05 step-up. If −1 tick: flip seq sign to −0.20e-4 or SKIP SLOT05.",
        "SLOT05": "Only live if SLOT04 tied/gained. If SLOT05 drops while SLOT04 held: keep SLOT04 scale.",
        "SLOT06": "If gain: residual pillars live. If drop: drop failing sub-pillars, do not raise α.",
        "SLOT07": "If uid_orth path and drop: treat as uid fail confirmation. If ownstack path gains: CV champ injects.",
        "SLOT08": "ALL-NEGATIVE A/B FALLBACK: if SLOT02–07 all ≤ SLOT01−1tick → SLOT08 becomes 0.70*AB+0.30*SLOT09 (private hedge early).",
        "SLOT09": "Honest CV readout. Not a tip climb. Compare private LB vs tip delta.",
        "SLOT10": "Final pick #2 candidate if tip overfits public. Prefer when SLOT09 private ≫ tip private.",
    }

    payload = {
        "NO_SUBMIT": True,
        "step": 6,
        "public_bank": {"n_cand": len(cand), "pillars": pillar_meta},
        "S_indep_vs_fullbank": div_ind,
        "S_cv_champ": cv_report,
        "prop_assets": {k: {kk: vv for kk, vv in v.items() if kk not in ("oof", "test")}
                        for k, v in assets.items()},
        "z_AB_twin": twin_report,
        "dual_baseline_gate_scorecard": {
            k: {kk: vv for kk, vv in g.items() if kk not in ("oof",)}
            for k, g in gate_scorecard.items()
        },
        "passing_dirs": list(dirs_t.keys()),
        "slots": slots,
        "integrity": integrity,
        "matrix_1mrho_e4": M.tolist(),
        "sequential_feedback_tree": feedback,
        "baselines": {
            "fullbank_zr": float(roc_auc_score(y, fb_o_zr)),
            "part12_zr": float(roc_auc_score(y, p12_o_zr)),
            "S_cv_champ_zr": cv_auc_zr,
        },
    }
    (LOG / "V41_FINAL_10SLOT_MANIFEST.json").write_text(json.dumps(payload, indent=2, default=str))

    # markdown
    lines = [
        "# V41 FINAL 10-SLOT MANIFEST — DS-Expert Hardened",
        "",
        f"**NO_SUBMIT={NO_SUBMIT}** · Hard stop after Step 6. **ZERO submissions.**",
        "",
        "## Executive",
        "",
        f"- Public CAND bank → family pillars: `{json.dumps({k: v['auc'] for k,v in pillar_meta.items()})}`",
        f"- S_indep vs Fullbank: 1-ρ oof/test e4={div_ind['1mrho_oof_e4']:.3f}/{div_ind['1mrho_test_e4']:.3f} "
        f"ratio={div_ind['ratio_test_over_oof']:.2f}",
        f"- **S_cv_champ** raw={cv_auc:.6f} +Zr=**{cv_auc_zr:.6f}** "
        f"(sf25 vs fb {cv_report['sf25_vs_fb_zr']}/25, vs p12 {cv_report['sf25_vs_p12_zr']}/25)",
        f"- z_AB_twin: R²_nz={r2_te:.4f} 1-ρ vs AB e4={twin_report['1mrho_test_vs_AB_e4']:.3f} "
        f"OOF+Zr={twin_report['oof_auc_zr']:.6f}",
        f"- Dual-gate PASS dirs: **{list(dirs_t.keys())}**",
        f"- Staged under `{FINAL}/`",
        "",
        "## 1. Residual / base asset scan (V34.3 Bug #2 fix)",
        "",
        "| asset | kind | status | auc/disp | KS |",
        "|---|---|---|---:|---:|",
    ]
    for k, v in assets.items():
        auc = v.get("auc")
        disp = v.get("disp")
        ks = v.get("KS", v.get("cal_KS"))
        a_s = f"{auc:.5f}" if isinstance(auc, float) and auc == auc else ""
        d_s = f"{disp:.3f}" if isinstance(disp, float) and disp == disp else a_s
        k_s = f"{ks:.4f}" if isinstance(ks, float) and ks == ks else ""
        lines.append(f"| `{k}` | {v.get('kind')} | {v.get('status')} | {d_s} | {k_s} |")

    lines += [
        "",
        "## 2. Dual-Baseline Gate Scorecard",
        "",
        "```json",
        json.dumps(payload["dual_baseline_gate_scorecard"], indent=2, default=str)[:30000],
        "```",
        "",
        "## 3. 10-Slot Master Table",
        "",
        "| Slot | File | Role | 1-ρ vs AB e4 | max U_zone | minDiv e4 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for sn in slot_names:
        s = slots[sn]
        integ = integrity[sn]
        md = integ["min_1mrho_vs_earlier_e4"]
        md_s = "" if md is None else f"{md:.4f}"
        lines.append(
            f"| {sn} | `{s['file']}` | {s['role'][:48]} | {s.get('1mrho_vs_AB_e4', float('nan')):.4f} | "
            f"{integ['max_U_zone']:.4f} | {md_s} |"
        )

    lines += [
        "",
        "## 4. 10×10 Pairwise 1-ρ (×1e-4)",
        "",
        "| |" + "|".join(slot_names) + "|",
        "|" + "|".join(["---"] * 11) + "|",
    ]
    for i, sn in enumerate(slot_names):
        row = [sn] + [f"{M[i, j]:.3f}" for j in range(10)]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## 5. Sequential Feedback Tree (1-slot-at-a-time)",
        "",
    ]
    for sn in slot_names:
        lines.append(f"- **{sn}:** {feedback[sn]}")

    lines += [
        "",
        "## 6. S_cv_champ diagnostics",
        "",
        "```json",
        json.dumps({k: cv_report[k] for k in cv_report if k != "sign_logs"}, indent=2),
        "```",
        "",
        "## 7. STOP",
        "",
        "Step 6 complete. **NO_SUBMIT=True. Do not submit.** Awaiting human audit before any live burn.",
        "",
    ]
    (LOG / "V41_FINAL_10SLOT_MANIFEST.md").write_text("\n".join(lines))

    print("\n=== STRICT NO_SUBMIT=True | V41 STEP6 DS-HARDENED ===", flush=True)
    print(f"S_cv_champ={cv_auc:.6f}/+Zr {cv_auc_zr:.6f}", flush=True)
    print(f"passing_dirs={list(dirs_t.keys())}", flush=True)
    print(f"WROTE {LOG / 'V41_FINAL_10SLOT_MANIFEST.md'}", flush=True)
    print(f"STAGED {FINAL}/ SLOT01..10", flush=True)
    print("STOP. Zero submissions. Awaiting audit.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
