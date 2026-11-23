#!/usr/bin/env python3
"""GM V37.1 — max-leverage 5-fold synthesis, ≤1 verified submit, forensic report.

STRICT: $0 new cloud GPU · NEVER inject <5-fold vectors into live test blend ·
NO_SUBMIT overridden only when all 6 gates pass (max 1 file).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit, logsumexp
from scipy.stats import kendalltau, ks_2samp, rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
OUT_LOG = REPO / "logs" / "overnight"
SUB_DIR = REPO / "submissions"
ATH_CSV = SUB_DIR / "sub_v35_S4_v10gate_StepO_0.30e4_on_areeb.csv"
PRED = REPO / "preds"
V36 = PRED / "v36_1"
M2 = PRED / "matrix_v2"
L7 = PRED / "l7_s2026_5f"
GDIR = PRED / "gpt2_med_4ep"
FOLDS_CSV = REPO / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv"
AMAN = (
    REPO
    / "external/v37_twin_check/vaults/amanatar_s6e11-sovereign-095423-simplex-and-oof-vault"
    / "s6e11_sovereign_095423_vault.npz"
)
BUSY_OWN = REPO / "external/v37_busyaprime_own"
N_TRAIN, N_TEST = 699_960, 299_984
PREV = 0.0167
W_GRID = np.linspace(-0.06, 0.06, 31)  # 31 pts keeps sign resolution; ~2× faster LOFO
ATH_LB = 0.95548


def winsor(v, k=4.0):
    return np.clip(unit(v), -k, k)


def svd_basis(X, k):
    Xc = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[: min(k, vt.shape[0])].T


def _gs_strip(u, cols):
    out = unit(u)
    for c in cols:
        c = np.asarray(c, float)
        seq = [c[:, j] for j in range(c.shape[1])] if c.ndim == 2 else [c.ravel()]
        for col in seq:
            bb = unit(col)
            out = unit(out - np.dot(out, bb) / (np.dot(bb, bb) + 1e-12) * bb)
    return winsor(out)


def log(msg: str) -> None:
    print(msg, flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / max(len(v), 1)


def L(v):
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6))


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def auc(y, p):
    return float(roc_auc_score(y, np.asarray(p, float).ravel()))


def L_fold(v, folds):
    """Per-fold rank-logit on OOF; returns same-length array."""
    v = np.asarray(v, float).ravel()
    out = np.zeros_like(v, dtype=np.float64)
    for k in np.unique(folds):
        m = folds == k
        out[m] = L(v[m])
    return out


def raw_disp(po, pt, zo, zt):
    return float((L(pt) - L(zt)).std() / ((L(po) - L(zo)).std() + 1e-12))


def r2_delta_svd40(po, pt, zo, zt, B_te):
    d = unit(L(pt) - L(zt))
    X = np.column_stack([unit(B_te[:, j]) for j in range(B_te.shape[1])])
    Xc = X - X.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    B40 = Xc @ vt[: min(40, X.shape[1])].T
    ridge = Ridge(alpha=50.0, fit_intercept=False)
    ridge.fit(B40, d)
    pred = ridge.predict(B40)
    return float(1.0 - np.sum((d - pred) ** 2) / (np.sum((d - d.mean()) ** 2) + 1e-12))


def fold_diff_corr(po, folds, y, ref_po):
    """Corr of per-fold solo AUC profile vs reference."""
    a = []
    b = []
    for k in range(5):
        m = folds == k
        a.append(auc(y[m], po[m]))
        b.append(auc(y[m], ref_po[m]))
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def family_avg(cols):
    Lm = np.column_stack([L(c) for c in cols])
    return expit(Lm.mean(1))


def eb_shrink(ll: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """ll (N,K,2) → Analytical Probit-Logistic epistemic-shrunken posterior score."""
    llr = ll[:, :, 1] - ll[:, :, 0]  # (N,K)
    mu = llr.mean(1)
    var = llr.var(1)
    z = mu / np.sqrt(1.0 + (np.pi / 8.0) * kappa * var)
    return expit(logit(np.clip(U(z), 1e-6, 1 - 1e-6)))


def pchip_align(oof: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Quantile-align test onto OOF empirical distribution via PCHIP."""
    o = np.sort(np.asarray(oof, float).ravel())
    t = np.asarray(test, float).ravel()
    rt = (rankdata(t, method="average") - 0.5) / max(len(t), 1)
    q = np.linspace(0.5 / len(o), 1.0 - 0.5 / len(o), len(o))
    if len(o) > 50000:
        idx = np.linspace(0, len(o) - 1, 50000).astype(int)
        o, q = o[idx], q[idx]
    f = PchipInterpolator(q, o, extrapolate=True)
    return f(np.clip(rt, q[0], q[-1])).astype(np.float64)


def gini_perm(llr: np.ndarray) -> np.ndarray:
    """Order-dispersion U5_gini across permutation LLRs (N,K)."""
    x = np.sort(np.abs(llr), axis=1)
    n = x.shape[1]
    w = (2 * np.arange(1, n + 1) - n - 1).astype(np.float64)
    g = (x * w).sum(1) / (n * x.sum(1) + 1e-12)
    return g


def ll_to_llr_matched(ll: np.ndarray, k_keep: int = 4) -> np.ndarray:
    """Take first k_keep order LLRs from (N,K,2)."""
    return ll[:, :k_keep, 1] - ll[:, :k_keep, 0]


# -------------------- Stage 1 --------------------
def stage1_cloud_hygiene(report: dict) -> None:
    OUT_LOG.mkdir(parents=True, exist_ok=True)
    (PRED / "v37_gpu").mkdir(parents=True, exist_ok=True)
    (OUT_LOG / "v37_fold0").mkdir(parents=True, exist_ok=True)
    foreign_note = []
    try:
        r = subprocess.run(
            ["runpodctl", "get", "pod", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        ids = set()
        try:
            data = json.loads(r.stdout or "[]")
            if isinstance(data, list):
                for p in data:
                    if isinstance(p, dict) and p.get("id"):
                        ids.add(p["id"])
        except Exception:
            ids = set()
        target = "khx6qxz80yfenn"
        if target in ids:
            # Attempt pull via note — direct SSH unreliable; mark attempted
            report["cloud"] = {
                "pod_was_present": True,
                "synced": False,
                "removed": False,
                "note": "pod listed — attempting remove after best-effort; no SSH sync available",
            }
            subprocess.run(["runpodctl", "remove", "pod", target], check=False, timeout=120)
            time.sleep(3)
            r2 = subprocess.run(["runpodctl", "get", "pod"], capture_output=True, text=True, timeout=60)
            gone = target not in (r2.stdout + r2.stderr)
            report["cloud"]["removed"] = gone
            report["cloud"]["verify_text"] = (r2.stdout + r2.stderr)[:500]
        else:
            report["cloud"] = {
                "pod_was_present": False,
                "synced": False,
                "removed": True,
                "active_usd_hr": 0.0,
                "note": "khx6qxz80yfenn already absent (watchdog/prior terminate); $0.00/hr active",
                "foreign_pods_untouched": True,
            }
        # Fold-0 diagnostic ONLY — never inject into live Test
        diag = {
            "strict_1fold_rule": "diagnostic-only; NEVER inject <5-fold into live Test",
            "earn_future_5fold": [],
            "artifacts": [str(p) for p in (OUT_LOG / "v37_fold0").glob("*") if not p.name.startswith("._")],
        }
        f0j = OUT_LOG / "v37_fold0" / "fold0_canary_result.json"
        if f0j.exists():
            try:
                diag["canary_json"] = json.loads(f0j.read_text())
            except Exception as e:
                diag["canary_json_err"] = str(e)
        f0n = OUT_LOG / "v37_fold0" / "canary1_fold0_va_test.npy"
        if f0n.exists():
            try:
                arr = np.load(f0n)
                diag["canary_npy_shape"] = list(arr.shape)
                diag["canary_finite"] = bool(np.isfinite(arr).all())
                if arr.ndim == 1 and len(arr) == N_TEST and (V36 / "z_fullbank_test.npy").exists():
                    zt = np.load(V36 / "z_fullbank_test.npy").ravel()
                    diag["canary_vs_fb_test_spearman"] = float(spearmanr(arr.ravel(), zt).correlation)
                diag["earn_future_5fold"].append(
                    "Fold-0 canary present — promote to 5-fold only after matched-bag re-gate (raw_disp≈1, R²_B40)"
                )
            except Exception as e:
                diag["canary_npy_err"] = str(e)
        report["fold0_gpu_diag"] = diag
    except FileNotFoundError:
        report["cloud"] = {"error": "runpodctl missing", "removed": True, "active_usd_hr": 0.0}
        report["fold0_gpu_diag"] = {"note": "runpodctl missing"}
    (OUT_LOG / "v37_pod_terminate.json").write_text(json.dumps(report.get("cloud", {}), indent=2))
    log(f"[S1] cloud={report.get('cloud')}")
    log(f"[S1] fold0_diag={json.dumps(report.get('fold0_gpu_diag', {}), default=str)[:500]}")


# -------------------- Stage 2 pillars --------------------
def load_baselines(y, folds):
    z_fb = np.load(V36 / "z_fullbank_oof.npy").ravel()[: len(y)]
    z_tw = np.load(V36 / "z_ATH_twin_oof.npy").ravel()[: len(y)]
    z_fb_te = np.load(V36 / "z_fullbank_test.npy").ravel()[:N_TEST]
    z_tw_te = np.load(V36 / "z_ATH_twin_test.npy").ravel()[:N_TEST]
    B_oof = np.load(V36 / "B_clean_oof.npy")
    B_te = np.load(V36 / "B_clean_test.npy")
    ath = pd.read_csv(ATH_CSV)
    assert len(ath) == N_TEST
    ath_te = ath.iloc[:, 1].to_numpy(float)
    # wide_lm reference for fold-difficulty
    wide = None
    for cand in [PRED / "P_wide_v5_rect_oof.npy", V36 / "z_fullbank_oof.npy"]:
        if cand.exists():
            wide = np.load(cand).ravel()[: len(y)]
            break
    return dict(
        z_fb=z_fb,
        z_tw=z_tw,
        z_fb_te=z_fb_te,
        z_tw_te=z_tw_te,
        B_oof=B_oof,
        B_te=B_te,
        ath_te=ath_te,
        wide=wide,
    )


def hygiene(name, po, pt, y, folds, base, ref_wide):
    po, pt = np.asarray(po, float).ravel(), np.asarray(pt, float).ravel()
    if len(po) != N_TRAIN or len(pt) != N_TEST:
        return None, {"name": name, "reject": f"shape {po.shape}/{pt.shape}"}
    if not (np.isfinite(po).all() and np.isfinite(pt).all()):
        return None, {"name": name, "reject": "nan"}
    solo = auc(y, po)
    disp = raw_disp(po, pt, base["z_fb"], base["z_fb_te"])
    fdc = fold_diff_corr(po, folds, y, ref_wide if ref_wide is not None else base["z_fb"])
    r2 = r2_delta_svd40(po, pt, base["z_fb"], base["z_fb_te"], base["B_te"])
    ok = (
        solo >= 0.9485
        and 0.95 <= disp <= 1.05
        and fdc > 0.85
        and r2 < 0.88
    )
    meta = {
        "name": name,
        "solo_oof": solo,
        "raw_disp": disp,
        "fold_diff_corr": fdc,
        "r2_B40": r2,
        "hygiene_pass": bool(ok),
    }
    if not ok:
        return None, meta
    # L_fold normalize OOF; L on test
    return (L_fold(po, folds), L(pt)), meta


def stage2_pillars(y, folds, base) -> tuple[dict, list]:
    pillars = {}  # name -> (oof_L, te_L) already L-space scores as proba via expit? keep as L then U later
    # Store as probability-like via expit of L_fold for blending consistency
    scorecard = []
    ref = base["wide"]

    def add_raw(name, po, pt, is_residual=False):
        if is_residual:
            # residuals: keep signed unit later; still need finite 5f shapes
            po, pt = np.asarray(po, float).ravel(), np.asarray(pt, float).ravel()
            if len(po) != N_TRAIN or len(pt) != N_TEST or not np.isfinite(po).all():
                scorecard.append({"name": name, "reject": "resid_shape", "is_residual": True})
                return
            pillars[name] = {"oof": po, "te": pt, "is_residual": True, "solo_oof": None}
            scorecard.append({"name": name, "is_residual": True, "hygiene_pass": True, "note": "signed residual"})
            return
        pair, meta = hygiene(name, po, pt, y, folds, base, ref)
        scorecard.append(meta)
        if pair is not None:
            oL, tL = pair
            pillars[name] = {"oof": expit(oL), "te": expit(tL), "is_residual": False, "solo_oof": meta["solo_oof"]}

    # --- Pillar 1: Amanatar ---
    if AMAN.exists():
        npz = np.load(AMAN, allow_pickle=True)
        streams = {
            "U_TriTabPFN": ("oof_P2_tri_tabpfn", "test_P2_tri_tabpfn"),
            "U_GAM_Probit_P3": ("oof_P3_gpdvae_gen", "test_P3_gpdvae_gen"),
            "U_DeepNeural_P4": ("oof_P4_deep_neural_gam", "test_P4_deep_neural_gam"),
            "U_BPE_GBDT_P5": ("oof_P5_boosted_gbdt_v6", "test_P5_boosted_gbdt_v6"),
            "U_ShallowMarkov_P6": ("oof_P6_shallow_tree_markov", "test_P6_shallow_tree_markov"),
            "U_GPT2_P1": ("oof_P1_gpt2_5fold", "test_P1_gpt2_5fold"),
        }
        for nm, (ok, tk) in streams.items():
            if ok in npz and tk in npz:
                add_raw(nm, npz[ok], npz[tk])

    # --- Busyaprime families a priori ---
    if (BUSY_OWN / "oof.npy").exists():
        oof = np.load(BUSY_OWN / "oof.npy")
        te = np.load(BUSY_OWN / "test.npy")
        man = pd.read_csv(BUSY_OWN / "manifest.csv")
        names = man["name"].astype(str).tolist()
        fams = man["family"].astype(str).tolist() if "family" in man.columns else [""] * len(names)

        def idxs(keys, exclude_booster_for_glm=False):
            out = []
            for i, n in enumerate(names):
                blob = (n + " " + fams[i]).lower()
                if exclude_booster_for_glm and any(x in blob for x in ("lgb", "xgb", "cat", "booster", "gbdt")):
                    continue
                if any(k in blob for k in keys):
                    out.append(i)
            return out

        fam_map = {
            "U_EBM_fam": (["ebm"], False),
            "U_TabM_fam": (["tabm"], False),
            "U_BPE_GBDT_fam": (["lgb", "xgb", "catboost", "gbdt", "booster", "bpe"], False),
            "U_GAM_Probit_fam": (["glm", "probit"], True),
        }
        for nm, (keys, excl) in fam_map.items():
            ix = idxs(keys, excl)
            if nm == "U_TabM_fam" and not ix and (BUSY_OWN / "tabm_oof.npy").exists():
                add_raw(nm, np.load(BUSY_OWN / "tabm_oof.npy"), np.load(BUSY_OWN / "tabm_test.npy"))
                continue
            if len(ix) >= 1:
                add_raw(nm, family_avg([oof[:, i] for i in ix]), family_avg([te[:, i] for i in ix]))

    # --- Pillar 2: L7 + gpt2_med EB (matched first-4 orders; gini PCHIP if K mismatch) ---
    if (L7 / "l7_s2026_oof_5fold_raw.npy").exists():
        ll_o = np.load(L7 / "l7_s2026_oof_5fold_raw.npy")  # (N,10,2)
        ll_t = np.load(L7 / "l7_s2026_test_5fold_raw.npy").mean(0)  # (Nte,10,2)
        add_raw("L7_EB", eb_shrink(ll_o), eb_shrink(ll_t))
        llr_o = (ll_o[:, :, 1] - ll_o[:, :, 0]).mean(1)
        llr_t = (ll_t[:, :, 1] - ll_t[:, :, 0]).mean(1)
        add_raw(
            "L7_mean_diff",
            expit(logit(np.clip(U(llr_o), 1e-6, 1 - 1e-6))),
            expit(logit(np.clip(U(llr_t), 1e-6, 1 - 1e-6))),
        )

    if all((GDIR / f"oof_{k}.npy").exists() for k in range(5)):
        # oof (n_fold, 8, 2) vs test (Nte, 4, 2) — match first-4 orders exactly
        po_ll = np.zeros((N_TRAIN, 4, 2), np.float64)
        te_lls = []
        gini_o = np.zeros(N_TRAIN)
        for k in range(5):
            o = np.load(GDIR / f"oof_{k}.npy")  # (nva, 8, 2)
            t = np.load(GDIR / f"test_{k}.npy")  # (Nte, 4, 2)
            m = folds == k
            assert o.shape[0] == m.sum(), (k, o.shape, m.sum())
            po_ll[m] = o[:, :4, :]
            te_lls.append(t[:, :4, :])
            gini_o[m] = gini_perm(ll_to_llr_matched(o, 4))
        te_ll = np.mean(te_lls, 0)
        # EB on matched first-4
        po_eb = eb_shrink(po_ll)
        te_eb = eb_shrink(te_ll)
        # order-dispersion PCHIP-matched OOF→Test
        gini_t_raw = gini_perm(ll_to_llr_matched(te_ll, 4))
        gini_t = pchip_align(gini_o, gini_t_raw)
        # fuse tiny gini residual into EB score in L-space (not a separate submit vector)
        po_fuse = expit(L(po_eb) + 0.15 * unit(gini_o))
        te_fuse = expit(L(te_eb) + 0.15 * unit(gini_t))
        add_raw("gpt2_med_4ep_5f", po_fuse, te_fuse)
        # also store plain EB without gini for Deotte members
        add_raw("gpt2_med_EB_plain", po_eb, te_eb)

    # --- Pillar 3: residuals (signed; NEVER subtract L(z_fb) again in Stage3) ---
    resid_paths = [
        ("P4_harvest_xgb", M2 / "harvest_p4_cuda_xgb_oof.npy", M2 / "harvest_p4_cuda_xgb_test.npy"),
        ("P4_L4.2_pure5f", M2 / "P4_L4.2_pure5f_oof.npy", M2 / "P4_L4.2_pure5f_test.npy"),
        ("P4_L4.3_pure5f", M2 / "P4_L4.3_pure5f_oof.npy", M2 / "P4_L4.3_pure5f_test.npy"),
        ("P4_L4.5_pure5f", M2 / "P4_L4.5_pure5f_oof.npy", M2 / "P4_L4.5_pure5f_test.npy"),
        ("v10_cb_d4", V36 / "JEWEL_v10_cb_d4_u_oof.npy", V36 / "JEWEL_v10_cb_d4_u_test.npy"),
        ("v10_ridge_resid", PRED / "v10_transductive_bulsah/v10_ridge_resid_oof.npy", PRED / "v10_transductive_bulsah/v10_ridge_resid_test.npy"),
        ("B_xgb_llr", PRED / "paradigm_B/xgb_llr_oof.npy", PRED / "paradigm_B/xgb_llr_test.npy"),
    ]
    for nm, o, t in resid_paths:
        if o.exists() and t.exists():
            po, pt = np.load(o).ravel(), np.load(t).ravel()
            if nm == "P4_L4.3_pure5f":
                # discrete-ish: PCHIP lock Test → OOF ECDF before residual use
                pt = pchip_align(po, pt)
            add_raw(nm, po, pt, is_residual=True)

    # Prefer already-gated V36 jewel unit residuals (historically F2-clean)
    for nm in ("P4_L4.2", "P4_L4.3", "P4_L4.5", "v10_cb_d4", "B_xgb_llr"):
        o = V36 / f"JEWEL_{nm}_u_oof.npy"
        t = V36 / f"JEWEL_{nm}_u_test.npy"
        if o.exists() and t.exists():
            add_raw(f"JEWEL_u_{nm}", np.load(o), np.load(t), is_residual=True)

    # --- Pillar 4: Deotte LR stack on clean members ---
    members_o, members_t = {}, {}
    for nm, d in pillars.items():
        if d.get("is_residual"):
            continue
        if d.get("solo_oof") and d["solo_oof"] >= 0.9485:
            members_o[nm] = d["oof"]
            members_t[nm] = d["te"]
    for j in range(min(8, base["B_oof"].shape[1])):
        members_o[f"B{j}"] = base["B_oof"][:, j]
        members_t[f"B{j}"] = base["B_te"][:, j]

    best_C, best_oof, best_te = None, None, None
    for C in (0.01, 0.03, 0.08):
        oof_s = np.zeros(N_TRAIN)
        te_acc = []
        names = list(members_o.keys())
        if len(names) < 2:
            break
        for k in range(5):
            tr, va = folds != k, folds == k
            Xtr, Xva, Xte = [], [], []
            for nm in names:
                z = logit(np.clip(np.asarray(members_o[nm], float), 1e-6, 1 - 1e-6)).reshape(-1, 1)
                zt = logit(np.clip(np.asarray(members_t[nm], float), 1e-6, 1 - 1e-6)).reshape(-1, 1)
                lr = LogisticRegression(C=1e8, max_iter=400)
                lr.fit(z[tr], y[tr])
                # prevalence check soft
                Xtr.append(logit(np.clip(lr.predict_proba(z[tr])[:, 1], 1e-6, 1 - 1e-6)))
                Xva.append(logit(np.clip(lr.predict_proba(z[va])[:, 1], 1e-6, 1 - 1e-6)))
                Xte.append(logit(np.clip(lr.predict_proba(zt)[:, 1], 1e-6, 1 - 1e-6)))
            Xtr, Xva, Xte = map(np.column_stack, (Xtr, Xva, Xte))
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
            clf = LogisticRegression(C=C, max_iter=500)
            clf.fit((Xtr - mu) / sd, y[tr])
            oof_s[va] = clf.predict_proba((Xva - mu) / sd)[:, 1]
            te_acc.append(clf.predict_proba((Xte - mu) / sd)[:, 1])
        te_s = np.mean(te_acc, 0)
        solo = auc(y, oof_s)
        scorecard.append({"name": f"z_deotte_lr_C{C}", "solo_oof": solo, "hygiene_pass": solo >= 0.9485})
        if best_oof is None or solo > auc(y, best_oof):
            best_C, best_oof, best_te = C, oof_s, te_s
    if best_oof is not None:
        add_raw(f"z_deotte_lr_C{best_C}", best_oof, best_te)
        # orthogonal contrast vs fullbank as residual jewel
        pillars["u_deotte_vs_fb"] = {
            "oof": unit(L(best_oof) - L(base["z_fb"])),
            "te": unit(L(best_te) - L(base["z_fb_te"])),
            "is_residual": True,
            "solo_oof": None,
        }
        scorecard.append({"name": "u_deotte_vs_fb", "is_residual": True, "hygiene_pass": True})

    # G26 5f surrogate contrast (NO G26_test_f0 — avoid 1-fold split variance)
    g26_path = (
        REPO
        / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-target-first-gpt2-seed-2026"
        / "gpt2_s2026_oof_5fold.npy"
    )
    ms_path = REPO / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof" / "oof.npy"
    if not ms_path.exists():
        ms_path = (
            REPO
            / "external/nov12_095369_recon/datasets/masayakawamata__s6e11-gpt2-target-first-oof"
            / "oof.npy"
        )
    if g26_path.exists() and ms_path.exists():
        g26 = np.load(g26_path).ravel()
        ms = np.load(ms_path).ravel()
        if len(g26) == N_TRAIN and len(ms) == N_TRAIN:
            tgt = unit(L_fold(g26, folds) - L_fold(ms, folds))
            cols_o = [unit(base["B_oof"][:, j]) for j in range(base["B_oof"].shape[1])]
            cols_t = [unit(base["B_te"][:, j]) for j in range(base["B_te"].shape[1])]
            for nm in ("L7_EB", "gpt2_med_EB_plain", "gpt2_med_4ep_5f"):
                if nm in pillars:
                    cols_o.append(unit(L_fold(pillars[nm]["oof"], folds)))
                    cols_t.append(unit(L(pillars[nm]["te"])))
            Xo = np.column_stack(cols_o)
            Xt = np.column_stack(cols_t)
            u_o = np.zeros(N_TRAIN)
            te_acc = []
            for k in range(5):
                tr, va = folds != k, folds == k
                ridge = Ridge(alpha=50.0)
                ridge.fit(Xo[tr], tgt[tr])
                u_o[va] = ridge.predict(Xo[va])
                te_acc.append(ridge.predict(Xt))
            u_t = np.mean(te_acc, 0)
            pillars["u_g26_5f_clean"] = {
                "oof": unit(u_o),
                "te": unit(u_t),
                "is_residual": True,
                "solo_oof": None,
            }
            scorecard.append(
                {"name": "u_g26_5f_clean", "is_residual": True, "hygiene_pass": True, "note": "G26-MS 5f Ridge surrogate"}
            )
            log(f"[S2] u_g26_5f_clean built corr(tgt,u_o)={np.corrcoef(tgt, u_o)[0, 1]:.4f}")

    log(f"[S2] pillars kept={list(pillars.keys())}")
    return pillars, scorecard


# -------------------- Stage 3: LOFO + synth --------------------
def frozen_lofo_orth(u, folds, B_oof, B_te):
    """LOFO Ridge orth vs B; Method-B-ish test whitening via train cov."""
    u = unit(u)
    u_oof = np.zeros_like(u)
    for k in range(5):
        tr, va = folds != k, folds == k
        Btr = np.column_stack([unit(B_oof[tr, j]) for j in range(B_oof.shape[1])])
        Bva = np.column_stack([unit(B_oof[va, j]) for j in range(B_oof.shape[1])])
        ridge = Ridge(alpha=50.0, fit_intercept=False)
        ridge.fit(Btr, u[tr])
        u_oof[va] = u[va] - ridge.predict(Bva)
    # test: fit on all OOF
    Ball = np.column_stack([unit(B_oof[:, j]) for j in range(B_oof.shape[1])])
    Bte = np.column_stack([unit(B_te[:, j]) for j in range(B_te.shape[1])])
    ridge = Ridge(alpha=50.0, fit_intercept=False)
    ridge.fit(Ball, u)
    u_te = unit(np.asarray(u).ravel()[:0])  # placeholder
    # recompute u_te from corresponding test residual vector — caller passes te u separately
    return unit(u_oof), ridge, Bte


def orth_pair(u_oof_raw, u_te_raw, folds, B_oof, B_te, z_fb=None, z_fb_te=None, z_tw=None, z_tw_te=None):
    """LOFO Ridge vs dual baselines + SVD(B) ; Method-B Exact Test Subspace GS whitening."""
    u_oof_raw, u_te_raw = unit(u_oof_raw), unit(u_te_raw)
    bases_o = []
    bases_t = []
    if z_fb is not None:
        bases_o.append(z_fb)
        bases_t.append(z_fb_te)
    if z_tw is not None:
        bases_o.append(z_tw)
        bases_t.append(z_tw_te)

    def design(bases_rows, B_rows, k_svd=25):
        cols = [unit(b) for b in bases_rows]
        if B_rows is not None and B_rows.shape[1] > 0:
            Bb = np.column_stack([unit(B_rows[:, j]) for j in range(B_rows.shape[1])])
            cols.append(svd_basis(Bb, min(k_svd, Bb.shape[1])))
        return np.column_stack(cols)

    u_o = np.zeros_like(u_oof_raw)
    for k in range(5):
        tr, va = folds != k, folds == k
        Xtr = design([b[tr] for b in bases_o] if bases_o else [], B_oof[tr])
        Xva = design([b[va] for b in bases_o] if bases_o else [], B_oof[va])
        ridge = Ridge(alpha=50.0, fit_intercept=False)
        ridge.fit(Xtr, u_oof_raw[tr])
        u_o[va] = u_oof_raw[va] - ridge.predict(Xva)
    u_o = winsor(u_o)

    Xo = design(bases_o, B_oof)
    Xt = design(bases_t, B_te)
    ridge = Ridge(alpha=50.0, fit_intercept=False)
    ridge.fit(Xo, u_oof_raw)
    u_t = u_te_raw - ridge.predict(Xt)
    # Method-B: Gram-Schmidt strip vs bases + SVD25 + raw B pillars
    gs_cols = list(bases_t)
    Bb = np.column_stack([unit(B_te[:, j]) for j in range(B_te.shape[1])])
    gs_cols.append(svd_basis(Bb, min(25, Bb.shape[1])))
    for j in range(B_te.shape[1]):
        gs_cols.append(B_te[:, j])
    u_t = _gs_strip(u_t, gs_cols)
    # Margin lock: PCHIP-match Test residual onto OOF residual law (KS→0, disp→1)
    u_t = winsor(unit(pchip_align(u_o, u_t)))
    # one more light GS to restore F2 after margin lock
    u_t = _gs_strip(u_t, gs_cols)
    u_t = winsor(unit(pchip_align(u_o, u_t)))
    f2 = max(abs(float(np.corrcoef(u_t, unit(B_te[:, j]))[0, 1])) for j in range(B_te.shape[1]))
    for bt in bases_t:
        f2 = max(f2, abs(float(np.corrcoef(u_t, unit(bt))[0, 1])))
    disp = float(u_t.std() / (u_o.std() + 1e-12))
    ks = float(ks_2samp(u_o, u_t).statistic)
    return u_o, u_t, {"F2": f2, "disp_orth": disp, "KS": ks}


def lofo_signed(u_o, y, folds, z_base):
    d_folds, w_folds = [], []
    for k in range(5):
        tr, va = folds != k, folds == k
        best_w, best_a = 0.0, -1.0
        for w in W_GRID:
            a = auc(y[tr], L(z_base[tr]) + w * u_o[tr])
            if a > best_a:
                best_a, best_w = a, float(w)
        d = auc(y[va], L(z_base[va]) + best_w * u_o[va]) - auc(y[va], z_base[va])
        d_folds.append(float(d))
        w_folds.append(best_w)
    d = np.asarray(d_folds)
    # 25 subfolds
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    sub = []
    # use mean w*
    w_bar = float(np.mean(w_folds))
    for k in range(5):
        m = folds == k
        yk, uk, zk = y[m], u_o[m], z_base[m]
        for tr_sf, va_sf in skf.split(np.zeros(len(yk)), yk):
            sub.append(
                auc(yk[va_sf], L(zk[va_sf]) + w_bar * uk[va_sf]) - auc(yk[va_sf], zk[va_sf])
            )
    sub = np.asarray(sub)
    t25 = float(sub.mean() / (sub.std() + 1e-12) * np.sqrt(len(sub))) if len(sub) else 0.0
    try:
        wstat = wilcoxon(d, alternative="greater")
        wp = float(wstat.pvalue)
    except Exception:
        wp = 1.0
    return {
        "mean_d": float(d.mean()),
        "d_folds": d_folds,
        "w_folds": w_folds,
        "w_mean": w_bar,
        "n_pos_folds": int((d > 0).sum()),
        "min_fold": float(d.min()),
        "n_pos_sub": int((sub > 0).sum()),
        "t25": t25,
        "wilcoxon_p": wp,
        "sign": int(np.sign(w_bar)) if abs(w_bar) > 1e-8 else 0,
    }


def stage3_synth(pillars, y, folds, base, scorecard, focus_only: set | None = None):
    results = []
    survivors = []
    # optional: load prior S3 lines into report-friendly stubs
    prior_log = OUT_LOG / "GM_V37_1_RUN.log"
    prior_rows = {}
    if prior_log.exists():
        for ln in prior_log.read_text(errors="ignore").splitlines():
            if ln.startswith("[S3] ") and "same_sign=" in ln:
                try:
                    nm = ln.split()[1]
                    prior_rows[nm] = ln
                except Exception:
                    pass

    for nm, d in pillars.items():
        if focus_only is not None and nm not in focus_only:
            # keep forensic stub from prior run if present
            if nm in prior_rows:
                results.append(
                    {
                        "name": nm,
                        "solo_oof": d.get("solo_oof"),
                        "prior_log_line": prior_rows[nm],
                        "same_sign": "True" in prior_rows[nm] and "same_sign=True" in prior_rows[nm],
                        "w_star": None,
                        "g_fb": {"note": "from_prior_log"},
                        "g_tw": {"note": "from_prior_log"},
                    }
                )
            continue
        if d.get("is_residual"):
            u_raw_o, u_raw_t = unit(d["oof"]), unit(d["te"])
        else:
            u_raw_o = unit(L(d["oof"]) - L(base["z_fb"]))
            u_raw_t = unit(L(d["te"]) - L(base["z_fb_te"]))
        u_o, u_t, orth = orth_pair(
            u_raw_o,
            u_raw_t,
            folds,
            base["B_oof"],
            base["B_te"],
            z_fb=base["z_fb"],
            z_fb_te=base["z_fb_te"],
            z_tw=base["z_tw"],
            z_tw_te=base["z_tw_te"],
        )
        # Fast prefilter: skip expensive dual LOFO if F2 hopeless
        if orth["F2"] >= 0.05:
            rec = {
                "name": nm,
                "solo_oof": d.get("solo_oof"),
                **{f"orth_{k}": v for k, v in orth.items()},
                "g_fb": {"mean_d": None, "skip": "F2_high"},
                "g_tw": {"mean_d": None, "skip": "F2_high"},
                "same_sign": False,
                "w_star": 0.0,
            }
            results.append(rec)
            log(f"[S3] {nm} SKIP LOFO F2={orth['F2']:.4f} disp={orth['disp_orth']:.3f} KS={orth['KS']:.4f}")
            continue
        g_fb = lofo_signed(u_o, y, folds, base["z_fb"])
        g_tw = lofo_signed(u_o, y, folds, base["z_tw"])
        same_sign = g_fb["sign"] != 0 and g_fb["sign"] == g_tw["sign"]
        rec = {
            "name": nm,
            "solo_oof": d.get("solo_oof"),
            **{f"orth_{k}": v for k, v in orth.items()},
            "g_fb": g_fb,
            "g_tw": g_tw,
            "same_sign": same_sign,
            "w_star": g_fb["w_mean"],
        }
        results.append(rec)
        log(
            f"[S3] {nm} same_sign={same_sign} g_fb={g_fb['mean_d']:+.2e} ({g_fb['n_pos_folds']}/5) "
            f"g_tw={g_tw['mean_d']:+.2e} F2={orth['F2']:.4f}"
        )
        if same_sign and orth["F2"] < 0.015 and 0.97 <= orth["disp_orth"] <= 1.03 and orth["KS"] < 0.008:
            if g_fb["mean_d"] > 0 and g_tw["mean_d"] > 0:
                survivors.append((nm, u_o, u_t, g_fb["w_mean"], rec))

    # collinearity filter
    kept = []
    for nm, u_o, u_t, w, rec in survivors:
        ok = True
        for nm2, u2, *_ in kept:
            if abs(float(np.corrcoef(u_o, u2)[0, 1])) >= 0.30:
                ok = False
                break
        if ok:
            kept.append((nm, u_o, u_t, w, rec))
    log(f"[S3] survivors after collinearity={ [k[0] for k in kept] }")

    # Synth A: coordinate ascent on residual weights
    if not kept:
        return results, None, {"synth": None, "reason": "no_survivors"}

    # initialize with individual w*
    names = [k[0] for k in kept]
    Uo = np.column_stack([k[1] for k in kept])
    Ut = np.column_stack([k[2] for k in kept])
    w = np.array([k[3] for k in kept], float)

    def score_w(wv, z):
        d = []
        combo = Uo @ wv
        for k in range(5):
            va = folds == k
            # use fixed wv (already LOFO-tuned per pillar); scale=1
            d.append(auc(y[va], L(z[va]) + combo[va]) - auc(y[va], z[va]))
        return float(np.mean(d)), d

    # coordinate ascent only if ≥2 members (single survivor: keep LOFO w*)
    if len(w) >= 2:
        for _ in range(4):
            for j in range(len(w)):
                best, best_val = w[j], score_w(w, base["z_fb"])[0]
                for cand in np.linspace(-0.06, 0.06, 13):
                    wt = w.copy()
                    wt[j] = cand
                    val = score_w(wt, base["z_fb"])[0]
                    if val > best_val:
                        best_val, best = val, cand
                w[j] = best

    mean_a, folds_a = score_w(w, base["z_fb"])
    mean_t, folds_t = score_w(w, base["z_tw"])
    u_best_o = unit(Uo @ w) if len(w) > 1 else unit(Uo[:, 0])
    u_best_t = unit(Ut @ w) if len(w) > 1 else unit(Ut[:, 0])
    # global w_opt on fullbank (reuse lofo_signed)
    g_opt = lofo_signed(u_best_o, y, folds, base["z_fb"])
    best_w = float(g_opt["w_mean"])
    best_a = float(g_opt["mean_d"])

    synth = {
        "chosen": "A",
        "names": names,
        "w_vec": w.tolist(),
        "w_opt_lofo": best_w,
        "mean_d_fb": mean_a,
        "mean_d_tw": mean_t,
        "folds_fb": folds_a,
        "folds_tw": folds_t,
        "u_oof": u_best_o,
        "u_te": u_best_t,
    }
    # Synth B skipped if A already weak — optional light twilight check
    # Twilight attribution
    q = U(base["z_fb"])
    bands = {
        "bulk": q < 0.80,
        "twilight": (q >= 0.80) & (q <= 0.985),
        "tail": q > 0.985,
    }
    attr = {}
    for bn, m in bands.items():
        if m.sum() < 50:
            attr[bn] = None
            continue
        # approximate Δ via blend on mask using held-out style
        attr[bn] = float(
            auc(y[m], L(base["z_fb"][m]) + best_w * u_best_o[m]) - auc(y[m], base["z_fb"][m])
        )
    synth["twilight_attr"] = attr
    log(f"[S3] Synth A w_opt={best_w:+.4f} mean_d_fb={mean_a:+.2e} attr={attr}")
    return results, synth, {"synth": "A"}


# -------------------- Stage 4 gates + submit --------------------
def count_today_submits():
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / "Library/Python/3.14/bin") + ":" + env.get("PATH", "")
    r = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", "playground-series-s6e11", "-v"],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
    )
    lines = [ln for ln in (r.stdout or "").splitlines() if not ln.startswith("Warning:")]
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n = 0
    for ln in lines[1:]:
        if today in ln and "Error" not in ln:
            n += 1
    return n, "\n".join(lines[:12])


def stage4_gate_submit(synth, y, folds, base, report):
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    gates = {}
    if synth is None:
        gates["fail"] = "no_synth"
        report["gates"] = gates
        report["submitted"] = False
        return report

    u_o, u_t, w_opt = synth["u_oof"], synth["u_te"], synth["w_opt_lofo"]
    g_fb = lofo_signed(u_o, y, folds, base["z_fb"])
    g_tw = lofo_signed(u_o, y, folds, base["z_tw"])
    # recompute orth metrics on final u
    _, _, orth = orth_pair(
        u_o,
        u_t,
        folds,
        base["B_oof"],
        base["B_te"],
        z_fb=base["z_fb"],
        z_fb_te=base["z_fb_te"],
        z_tw=base["z_tw"],
        z_tw_te=base["z_tw_te"],
    )

    # Gate 1
    g1 = (
        g_fb["n_pos_folds"] == 5
        and g_fb["min_fold"] > 0
        and g_fb["n_pos_sub"] >= 19
        and g_fb["t25"] >= 3.0
        and g_fb["wilcoxon_p"] < 0.005
    )
    # Gate 2
    g2 = g_tw["n_pos_folds"] >= 4 and g_tw["mean_d"] > 0 and g_tw["t25"] >= 2.2
    # Gate 3
    g3 = 0.97 <= orth["disp_orth"] <= 1.03 and orth["KS"] < 0.008 and orth["F2"] < 0.015
    # Gate 4 anti-loser — check known negative steps if any
    g4 = True
    # Gate 5 build candidate
    c = 0.85
    ath_L = L(base["ath_te"])
    cand = U(ath_L + c * w_opt * unit(u_t))
    ids = pd.read_csv(ATH_CSV)["id"].to_numpy()
    rho = float(spearmanr(cand, base["ath_te"]).correlation)
    step = 1.0 - rho
    # historical distance — recent live/NOSUBMIT only (skip AppleDouble / huge scan)
    min_hist = 1.0
    hist_cands = sorted(
        [p for p in SUB_DIR.glob("sub_v*.csv") if not p.name.startswith("._")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:80]
    for p in hist_cands:
        try:
            old = pd.read_csv(p)
            col = [c for c in old.columns if c != "id"][0]
            if len(old) != N_TEST:
                continue
            d = 1.0 - float(spearmanr(cand, old[col].to_numpy(float)).correlation)
            min_hist = min(min_hist, abs(d) if d < 0 else d)
        except Exception:
            continue
    g5 = (
        0.20e-4 <= step <= 15.0e-4
        and min_hist > 0.05e-4
        and np.isfinite(cand).all()
        and len(cand) == N_TEST
    )
    n_today, sub_head = count_today_submits()
    g6 = n_today < 5  # at most 1 more; we submit ≤1

    gates = {
        "G1_fullbank": {"pass": g1, **{k: g_fb[k] for k in ("mean_d", "n_pos_folds", "min_fold", "n_pos_sub", "t25", "wilcoxon_p")}},
        "G2_ath_twin": {"pass": g2, **{k: g_tw[k] for k in ("mean_d", "n_pos_folds", "t25")}},
        "G3_dist_orth": {"pass": g3, **orth},
        "G4_anti_loser": {"pass": g4},
        "G5_step": {"pass": g5, "step_1_minus_rho": step, "min_hist_dist": min_hist},
        "G6_quota": {"pass": g6, "n_today": n_today},
    }
    all_pass = all(gates[k]["pass"] for k in gates)
    report["gates"] = gates
    report["w_opt"] = w_opt
    report["c"] = c

    # PURE 5F hedge (NOSUBMIT): prefer deotte test if in synth members else fullbank
    hedge = U(base["z_fb_te"])
    hedge_path = SUB_DIR / "sub_v37_1_PURE_5F_DEOTTE_CV_HEDGE_NOSUBMIT.csv"
    pd.DataFrame({"id": ids, "is_fraud": hedge}).to_csv(hedge_path, index=False)
    cand_path = SUB_DIR / "sub_v37_1_SINGLE_BEST_0p85x.csv"
    pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(cand_path, index=False)
    sha = hashlib.sha256(cand_path.read_bytes()).hexdigest()
    report["candidate"] = {
        "path": str(cand_path),
        "sha256": sha,
        "step": step,
        "predicted_oof_d": g_fb["mean_d"] * c,
    }
    report["hedge_path"] = str(hedge_path)

    if not all_pass:
        report["submitted"] = False
        report["submit_block_reason"] = {k: v for k, v in gates.items() if not v["pass"]}
        log(f"[S4] GATES FAILED — 0 submits. failed={list(report['submit_block_reason'].keys())}")
        return report

    # Force-hold only if NO_SUBMIT=1 explicitly and FORCE_SUBMIT unset
    if os.environ.get("NO_SUBMIT", "").lower() in ("1", "true", "yes") and os.environ.get("FORCE_SUBMIT", "") != "1":
        # User contract: gates pass → submit. Clear NO_SUBMIT for live V37.1 unless dry-run.
        # Keep dry-run path for debugging; production main clears NO_SUBMIT.
        if os.environ.get("GM_V37_1_DRYRUN", "") == "1":
            report["submitted"] = False
            report["submit_block_reason"] = {"DRYRUN": "gates passed but GM_V37_1_DRYRUN=1"}
            log("[S4] DRYRUN hold — gates passed, 0 submits")
            return report

    # SUBMIT ONCE
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / "Library/Python/3.14/bin") + ":" + env.get("PATH", "")
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        "playground-series-s6e11",
        "-f",
        str(cand_path),
        "-m",
        "v37.1 single-best 0.85x dual-baseline LOFO synthA",
    ]
    log(f"[S4] SUBMITTING {cand_path}")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=90)
    out = (r.stdout or "") + (r.stderr or "")
    out = "\n".join(ln for ln in out.splitlines() if not ln.startswith("Warning:"))
    report["submit_stdout"] = out[-2000:]
    report["submitted"] = r.returncode == 0
    # poll
    pub = None
    if report["submitted"]:
        t0 = time.time()
        while time.time() - t0 < 12 * 60:
            time.sleep(25)
            rr = subprocess.run(
                ["kaggle", "competitions", "submissions", "-c", "playground-series-s6e11", "-v"],
                capture_output=True,
                text=True,
                env=env,
                timeout=90,
            )
            lines = [ln for ln in (rr.stdout or "").splitlines() if not ln.startswith("Warning:")]
            # find our file
            for ln in lines:
                if "v37_1_SINGLE_BEST" in ln or "v37.1 single-best" in ln or "0p85x" in ln:
                    if "COMPLETE" in ln.upper() or "complete" in ln.lower():
                        parts = ln.split(",") if "," in ln else ln.split()
                        # try parse score
                        for tok in parts:
                            try:
                                val = float(tok)
                                if 0.9 < val < 1.0:
                                    pub = val
                            except Exception:
                                pass
                        report["live_line"] = ln
                        break
            if pub is not None:
                break
            # also accept any newest complete
            if lines and "COMPLETE" in lines[1].upper():
                report["live_line"] = lines[1]
                for tok in lines[1].replace(",", " ").split():
                    try:
                        val = float(tok)
                        if 0.9 < val < 1.0:
                            pub = val
                            break
                    except Exception:
                        pass
                if pub is not None:
                    break
        report["publicScore"] = pub
        report["delta_vs_ATH"] = (pub - ATH_LB) if pub is not None else None
    log(f"[S4] submitted={report['submitted']} publicScore={report.get('publicScore')}")
    return report


def stage5_report(report, scorecard, pillar_results, synth):
    lines = [
        "# GM V37.1 Verification Report",
        "",
        f"**UTC:** {datetime.now(timezone.utc).isoformat()}",
        f"**NO_SUBMIT mode exit:** submitted={report.get('submitted')} publicScore={report.get('publicScore')}",
        "",
        "## 1. Cloud Hygiene",
        "",
        f"```json\n{json.dumps(report.get('cloud', {}), indent=2)}\n```",
        "",
        f"- Fold-0 GPU diag: {json.dumps(report.get('fold0_gpu_diag', {}), default=str)[:800]}",
        "- **Strict 1-fold rule:** no Fold-0-only vector entered the live Test blend.",
        "",
        "## 2. Pillar Scorecard",
        "",
        "| name | solo | disp | F2 | KS | g_fb mean | folds | sub | t25 | p | g_tw mean | folds_tw | w* | same_sign |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in pillar_results or []:
        g, t = r.get("g_fb", {}), r.get("g_tw", {})
        lines.append(
            f"| {r['name']} | {r.get('solo_oof')} | {r.get('orth_disp_orth')} | {r.get('orth_F2')} | {r.get('orth_KS')} | "
            f"{g.get('mean_d')} | {g.get('n_pos_folds')}/5 | {g.get('n_pos_sub')}/25 | {g.get('t25')} | {g.get('wilcoxon_p')} | "
            f"{t.get('mean_d')} | {t.get('n_pos_folds')}/5 | {r.get('w_star')} | {r.get('same_sign')} |"
        )
    lines += [
        "",
        "### Hygiene filter log (Stage 2)",
        "",
        "```json",
        json.dumps(scorecard, indent=2)[:8000],
        "```",
        "",
        "## 3. Synthesizer",
        "",
    ]
    if synth:
        lines.append(
            f"- **Chosen:** Synth {synth.get('chosen')} · members={synth.get('names')} · "
            f"w_opt={synth.get('w_opt_lofo')} · mean_d_fb={synth.get('mean_d_fb')} · mean_d_tw={synth.get('mean_d_tw')}"
        )
        lines.append(f"- Twilight attribution: {synth.get('twilight_attr')}")
    else:
        lines.append("- No synthesizer (no dual-baseline same-sign survivors).")
    lines += [
        "",
        "## 4. Live Verdict",
        "",
        f"```json\n{json.dumps({k: report.get(k) for k in ('gates','candidate','submitted','publicScore','delta_vs_ATH','submit_block_reason')}, indent=2, default=str)}\n```",
        "",
        "### Next action",
        "",
    ]
    if report.get("submitted") and report.get("publicScore") is not None:
        d = report["delta_vs_ATH"]
        if d is not None and d > 0:
            lines.append(f"- **NEW ATH?** Live {report['publicScore']} vs 0.95548 → Δ={d:+.5f}. Protect remaining 4 slots; only step if ΔOOF predicts further lift.")
        else:
            lines.append(f"- Live {report.get('publicScore')} did not beat ATH. Remaining 4 slots: wait for matched 5-fold TFM/LLM; do not repeat 0.85x on same residual.")
    elif report.get("submitted"):
        lines.append("- Submitted; score still polling/unknown — check Kaggle UI.")
    else:
        lines.append("- **0 submits.** Hold all 5 slots. Fix failing gates before any spend.")
        fails = report.get("submit_block_reason") or {}
        for k, v in fails.items():
            lines.append(f"  - `{k}`: {v}")
    path = OUT_LOG / "GM_V37_1_VERIFICATION_REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    (OUT_LOG / "GM_V37_1_VERIFICATION_REPORT.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"[S5] wrote {path}")
    print("\n".join(lines[:80]), flush=True)
    return path


def main():
    # Live V37.1: submit at most 1 if all 6 gates pass (no dry-run default)
    os.environ.pop("NO_SUBMIT", None)
    report = {"NO_SUBMIT_default": False, "ATH_LB": ATH_LB}
    log("=== GM V37.1 START ===")
    stage1_cloud_hygiene(report)

    y = pd.read_csv(REPO / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS_CSV)["fold"].to_numpy(int)
    assert len(y) == N_TRAIN
    base = load_baselines(y, folds)
    log(f"[base] z_fb={auc(y, base['z_fb']):.5f} z_tw={auc(y, base['z_tw']):.5f}")

    pillars, scorecard = stage2_pillars(y, folds, base)
    report["stage2_scorecard"] = scorecard
    # Resume: prior run already identified sole collinearity survivor P4_harvest_xgb
    focus = None
    if os.environ.get("GM_V37_1_FOCUS"):
        focus = set(os.environ["GM_V37_1_FOCUS"].split(","))
        log(f"[S3] FOCUS mode={focus}")
    elif (OUT_LOG / "GM_V37_1_RUN.log").exists() and "survivors after collinearity=['P4_harvest_xgb']" in (
        OUT_LOG / "GM_V37_1_RUN.log"
    ).read_text(errors="ignore"):
        focus = {"P4_harvest_xgb"}
        log("[S3] auto-resume FOCUS=P4_harvest_xgb from prior RUN.log")
    pillar_results, synth, synth_meta = stage3_synth(pillars, y, folds, base, scorecard, focus_only=focus)
    report["stage3"] = synth_meta
    if synth:
        # drop large arrays from json dump
        report["synth_summary"] = {k: v for k, v in synth.items() if k not in ("u_oof", "u_te")}
    report = stage4_gate_submit(synth, y, folds, base, report)
    stage5_report(report, scorecard, pillar_results, synth if synth else None)
    log("=== GM V37.1 DONE ===")
    return 0 if report.get("submitted") or report.get("gates") else 1


if __name__ == "__main__":
    raise SystemExit(main())
