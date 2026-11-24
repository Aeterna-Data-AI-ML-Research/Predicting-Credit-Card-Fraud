#!/usr/bin/env python3
"""V41 STEP 5 — Strictly nested dual stack (own nested × Part12) + Zr.

STRICT NO_SUBMIT=True. Writes logs/v41/STEP5_AUDIT.md (+.json) then STOPS.
Does NOT run Step 6.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
N_TR, N_TE = 699_960, 299_984
T0 = time.time()

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
SSD = Path("/tmp/s6e11_v39_ssd")
EXT = ROOT / "external/v41_nov23_frontier"
GPW_BANK = ROOT / "external/banks/goodpjw2008_s6e11-oof-bank"
BUSY = EXT / "busyaprime_52"
PART12 = ROOT / "external/nov08_part12"
LOG = ROOT / "logs/v41"
OUT = ROOT / "out/v41_staged"
PRED = ROOT / "preds"
DATA = ROOT / "data" if (ROOT / "data/test.csv").exists() else SSD / "data"
FOLDS_P = SSD / "folds/folds.csv" if (SSD / "folds/folds.csv").exists() else ROOT / "folds/folds.csv"

SHA_AB = "3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038"

QUARANTINE_SUBSTR = (
    "uid_catboost", "v41_d27_eb", "fintfm", "fin_tfm", "zanch", "v10_lgb",
    "gb_soft_bag3", "soft_diff", "idf_p3", "clean035", "fold0_only", "1fold",
    "f0_only", "g26",
)

FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]


def log(msg: str):
    print(f"[{time.time()-T0:5.1f}s] {msg}", flush=True)


def U(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v: np.ndarray) -> np.ndarray:
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12))


def L_fold(v: np.ndarray, folds: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    out = np.zeros_like(v, dtype=np.float64)
    for k in np.unique(folds):
        m = folds == k
        out[m] = L(v[m])
    return out


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


def apply_zr(scores: np.ndarray, zones: np.ndarray, u_ref: np.ndarray | None = None) -> np.ndarray:
    out = L(scores).copy()
    uref = u_ref if u_ref is not None else U(scores)
    if zones.any() and (~zones).any():
        floor = float(np.min(out[~zones])) - 1.0
        out[zones] = floor + uref[zones]
    return U(out)


def family_of(name: str, fam_hint: str = "") -> str:
    s = f"{name} {fam_hint}".lower()
    if any(x in s for x in ("gpt2", "compact", "target-first", "l7_s2026", "lm_", "megayak")):
        return "LM_GPT2"
    if any(x in s for x in ("tabm", "realmlp", "rmlp", "resmlp", "neural")):
        return "TabM_MLP"
    if any(x in s for x in ("ebm", "glm", "slr", "linear", "logistic")):
        return "EBM_GLM"
    if any(x in s for x in ("uid_v2", "xgb_llr", "proprietary", "paradigm")):
        return "Proprietary"
    if any(x in s for x in ("lgb", "xgb", "catboost", "gbdt", "gb_", "gbc", "gbe", "gbx", "lightgbm", "booster")):
        return "GBDT"
    return "Other"


def is_quarantined(name: str, path: str = "") -> bool:
    s = f"{name} {path}".lower()
    return any(q in s for q in QUARANTINE_SUBSTR)


def hygiene(oof: np.ndarray, te: np.ndarray, y: np.ndarray) -> dict:
    oof = np.asarray(oof, float).ravel()
    te = np.asarray(te, float).ravel()
    ok_shape = oof.shape == (N_TR,) and te.shape == (N_TE,)
    nan_inf = int(np.isnan(oof).sum() + np.isinf(oof).sum() + np.isnan(te).sum() + np.isinf(te).sum())
    auc = float(roc_auc_score(y, oof)) if ok_shape and nan_inf == 0 else float("nan")
    disp = float(te.std() / (oof.std() + 1e-12)) if ok_shape else float("nan")
    ks = float(ks_2samp(oof, te).statistic) if ok_shape and nan_inf == 0 else float("nan")
    pass_h = (
        ok_shape and nan_inf == 0
        and auc >= 0.905 and auc <= 0.960
        and 0.82 <= disp <= 1.18
        and ks <= 0.045
    )
    return {"auc": auc, "disp": disp, "KS": ks, "nan_inf": nan_inf, "pass": pass_h}


def write_u(path: Path, ids: np.ndarray, scores: np.ndarray) -> tuple[str, np.ndarray]:
    p = U(scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(index=False, lineterminator="\n")
    path.write_bytes(text.encode())
    return sha256_file(path), p


def fold_aucs(y, p, folds):
    return [float(roc_auc_score(y[folds == k], p[folds == k])) for k in range(5)]


def sf25_beats(y, a, b, folds) -> int:
    """Count of 25 subfolds where a > b."""
    pos = 0
    for k in range(5):
        m = folds == k
        yk, ak, bk = y[m], a[m], b[m]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + k)
        idx = np.arange(len(yk))
        for _, va in skf.split(idx, yk):
            if roc_auc_score(yk[va], ak[va]) > roc_auc_score(yk[va], bk[va]):
                pos += 1
    return pos


def assemble_candidates(y: np.ndarray, folds: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Return meta list + OOF matrix + Test matrix (raw scores)."""
    rows = []
    oofs, tes = [], []

    def add(name, fam_hint, oof, te, source):
        if is_quarantined(name, source):
            rows.append({"name": name, "family": family_of(name, fam_hint), "source": source,
                         "status": "QUARANTINE", "pass": False})
            return
        h = hygiene(oof, te, y)
        fam = family_of(name, fam_hint)
        rec = {"name": name, "family": fam, "source": source, **h}
        if h["pass"]:
            rec["status"] = "CAND"
            rows.append(rec)
            oofs.append(np.asarray(oof, float).ravel())
            tes.append(np.asarray(te, float).ravel())
        else:
            rec["status"] = "HYGIENE_FAIL"
            rows.append(rec)

    # --- goodpjw 70 ---
    man = pd.read_csv(GPW_BANK / "manifest.csv")
    for _, r in man.iterrows():
        name = str(r["name"])
        op, tp = GPW_BANK / f"oof_{name}.npy", GPW_BANK / f"test_{name}.npy"
        if not (op.exists() and tp.exists()):
            rows.append({"name": name, "family": family_of(name, str(r.get("family", ""))),
                         "source": "goodpjw70", "status": "MISSING", "pass": False})
            continue
        add(f"gpw70:{name}", str(r.get("family", "")), np.load(op), np.load(tp), "goodpjw70")

    # --- busyaprime 52 ---
    bman = pd.read_csv(BUSY / "manifest.csv")
    Bo = np.load(BUSY / "oof.npy")
    Bt = np.load(BUSY / "test.npy")
    assert Bo.shape == (N_TR, 52) and Bt.shape == (N_TE, 52)
    for i, r in bman.iterrows():
        name = str(r["name"])
        if "fintfm" in name.lower() or "fin_tfm" in name.lower():
            rows.append({"name": f"busy52:{name}", "family": "TabM_MLP", "source": "busyaprime52",
                         "status": "QUARANTINE_FinTFM", "pass": False})
            continue
        add(f"busy52:{name}", str(r.get("family", "")), Bo[:, i], Bt[:, i], "busyaprime52")

    # TabM standalone (also may duplicate tabm_dl)
    add("busy52:tabm_standalone", "neural net TabM",
        np.load(BUSY / "tabm_oof.npy"), np.load(BUSY / "tabm_test.npy"), "busyaprime52_tabm")

    # --- proprietary clean ---
    # uid_v2
    add("prop:uid_v2", "Proprietary",
        np.load(SSD / "out/uid_v2_oof.npy"), np.load(SSD / "out/uid_v2_test.npy"), "uid_v2")

    # xgb_llr
    add("prop:xgb_llr", "Proprietary",
        np.load(PRED / "paradigm_B/xgb_llr_oof.npy"),
        np.load(PRED / "paradigm_B/xgb_llr_test.npy"), "paradigm_B")

    # l7
    lo = np.load(PRED / "l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    lt = np.load(PRED / "l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    l7_o = (lo[:, :, 1] - lo[:, :, 0]).mean(1)
    l7_t = ((lt[..., 1] - lt[..., 0]).mean(2)).mean(0)
    add("prop:l7_s2026", "LM_GPT2", l7_o, l7_t, "l7_s2026_5f")

    # gpt2 med 4ep assemble
    gdir = PRED / "gpt2_med_4ep"
    if all((gdir / f"oof_{k}.npy").exists() for k in range(5)):
        gpt_o = np.zeros(N_TR)
        gpt_t_folds = []
        ok = True
        for k in range(5):
            a = np.load(gdir / f"oof_{k}.npy")
            b = np.load(gdir / f"test_{k}.npy")
            if a.ndim == 3:
                zo = (a[:, :, 1] - a[:, :, 0]).mean(1)
            else:
                zo = a.ravel()
            if b.ndim == 3:
                zt = (b[:, :, 1] - b[:, :, 0]).mean(1)
            else:
                zt = b.ravel()
            m = folds == k
            if zo.shape[0] != m.sum():
                ok = False
                break
            gpt_o[m] = zo
            gpt_t_folds.append(zt)
        if ok:
            add("prop:gpt2_med_4ep", "LM_GPT2", gpt_o, np.mean(gpt_t_folds, 0), "gpt2_med_4ep")
        else:
            rows.append({"name": "prop:gpt2_med_4ep", "family": "LM_GPT2", "source": "gpt2_med_4ep",
                         "status": "SHAPE_FAIL", "pass": False})

    # matrix pure5f — likely fail AUC gate but attempt
    for tag in ("P4_L4.2_pure5f", "P4_L4.3_pure5f"):
        op, tp = PRED / f"matrix_v2/{tag}_oof.npy", PRED / f"matrix_v2/{tag}_test.npy"
        if op.exists() and tp.exists():
            add(f"prop:{tag}", "Proprietary", np.load(op), np.load(tp), "matrix_v2")

    # v10 ridge — attempt; quarantine orth/soft siblings by name
    for tag in ("v10_ridge_resid", "v10_ridge_pure_fe"):
        op = PRED / f"v10_transductive_bulsah/{tag}_oof.npy"
        tp = PRED / f"v10_transductive_bulsah/{tag}_test.npy"
        if op.exists() and tp.exists():
            add(f"prop:{tag}", "Proprietary", np.load(op), np.load(tp), "v10_ridge")

    # explicit quarantine markers for excluded known bad
    for name in ("prop:uid_catboost", "prop:v41_d27_eb_zone"):
        rows.append({"name": name, "family": "Proprietary", "source": "HARD_QUARANTINE",
                     "status": "QUARANTINE", "pass": False})

    Xo = np.column_stack(oofs) if oofs else np.zeros((N_TR, 0))
    Xt = np.column_stack(tes) if tes else np.zeros((N_TE, 0))
    return rows, Xo, Xt


def dedup(meta_pass: list[dict], Xo: np.ndarray, Xt: np.ndarray, y: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    """Within-family ρ_oof>0.995 keep higher AUC / lower KS."""
    keep = np.ones(len(meta_pass), bool)
    dropped = []
    for fam in sorted(set(m["family"] for m in meta_pass)):
        idxs = [i for i, m in enumerate(meta_pass) if m["family"] == fam]
        for a in range(len(idxs)):
            i = idxs[a]
            if not keep[i]:
                continue
            for b in range(a + 1, len(idxs)):
                j = idxs[b]
                if not keep[j]:
                    continue
                rho = float(np.corrcoef(rankdata(Xo[:, i]), rankdata(Xo[:, j]))[0, 1])
                if rho > 0.995:
                    # drop worse
                    mi, mj = meta_pass[i], meta_pass[j]
                    drop_j = (mj["auc"], -mj["KS"]) < (mi["auc"], -mi["KS"])
                    loser = j if drop_j else i
                    winner = i if drop_j else j
                    keep[loser] = False
                    dropped.append({
                        "dropped": meta_pass[loser]["name"],
                        "kept": meta_pass[winner]["name"],
                        "family": fam, "rho_oof": rho,
                    })
    idx = np.where(keep)[0]
    kept_meta = [meta_pass[i] for i in idx]
    inv = {}
    for fam in ["LM_GPT2", "GBDT", "TabM_MLP", "EBM_GLM", "Proprietary", "Other"]:
        before = sum(1 for m in meta_pass if m["family"] == fam)
        after = sum(1 for m in kept_meta if m["family"] == fam)
        inv[fam] = {"before_dedup": before, "after_dedup": after}
    return kept_meta, Xo[:, idx], Xt[:, idx], {"dropped": dropped, "by_family": inv}


def platt_lo(train_s, train_y, app_s):
    """Fit Platt on train scores → calibrated log-odds on app."""
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
    x = np.asarray(train_s, float).ravel().reshape(-1, 1)
    lr.fit(x, train_y)
    p = np.clip(lr.predict_proba(np.asarray(app_s, float).ravel().reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
    return logit(p)


def nested_stack(Xo, Xt, y, folds, names, families) -> dict:
    """Strict nested meta with L_fold and Platt-calibrated features."""
    M = Xo.shape[1]
    # Global L_fold / L features for reporting; nested fits recompute safely
    Lf_o = np.column_stack([L_fold(Xo[:, j], folds) for j in range(M)])
    Lf_t = np.column_stack([L(Xt[:, j]) for j in range(M)])

    C_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    ridge_alphas = [1.0, 10.0, 100.0, 1000.0]

    oof_L2 = np.zeros(N_TR)
    oof_Cal = np.zeros(N_TR)
    oof_Sim = np.zeros(N_TR)
    oof_Ens = np.zeros(N_TR)
    te_L2 = np.zeros(N_TE)
    te_Cal = np.zeros(N_TE)
    te_Sim = np.zeros(N_TE)
    te_Ens = np.zeros(N_TE)

    weight_logs = []

    for f in range(5):
        trm, vam = folds != f, folds == f
        # Features: L_fold computed ONLY using train ranks for train rows;
        # for val, rank within val using L() of raw (standard approach: use global L_fold which ranks within each fold separately — val fold ranks among themselves, no train bleed)
        Xtr = Lf_o[trm]
        Xva = Lf_o[vam]
        Xte = Lf_t
        ytr, yva = y[trm], y[vam]

        # Platt-calibrated matrix (fit each col on train_idx only)
        Cal_tr = np.zeros_like(Xtr)
        Cal_va = np.zeros_like(Xva)
        Cal_te = np.zeros_like(Xte)
        for j in range(M):
            Cal_tr[:, j] = platt_lo(Xo[trm, j], ytr, Xo[trm, j])
            Cal_va[:, j] = platt_lo(Xo[trm, j], ytr, Xo[vam, j])
            Cal_te[:, j] = platt_lo(Xo[trm, j], ytr, Xt[:, j])

        # inner 4-fold on train_idx for C / alpha
        inner_folds = folds[trm]
        # remap inner fold ids present
        uniq = sorted(np.unique(inner_folds))
        assert len(uniq) == 4

        def inner_cv_logreg(X, yy, ff, use_cal=False):
            best_c, best_score = C_grid[0], -1.0
            for C in C_grid:
                scores = []
                for u in uniq:
                    itrm, ivam = ff != u, ff == u
                    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=400, n_jobs=1)
                    clf.fit(X[itrm], yy[itrm])
                    scores.append(roc_auc_score(yy[ivam], clf.predict_proba(X[ivam])[:, 1]))
                sc = float(np.mean(scores))
                if sc > best_score:
                    best_score, best_c = sc, C
            return best_c, best_score

        def inner_cv_ridge(X, yy, ff):
            best_a, best_score = ridge_alphas[0], -1.0
            for a in ridge_alphas:
                scores = []
                for u in uniq:
                    itrm, ivam = ff != u, ff == u
                    # non-negative via clip after unconstrained ridge on centered
                    rg = Ridge(alpha=a, fit_intercept=True)
                    rg.fit(X[itrm], yy[itrm])
                    pred = rg.predict(X[ivam])
                    scores.append(roc_auc_score(yy[ivam], pred))
                sc = float(np.mean(scores))
                if sc > best_score:
                    best_score, best_a = sc, a
            return best_a, best_score

        c_l2, s_l2 = inner_cv_logreg(Xtr, ytr, inner_folds)
        c_cal, s_cal = inner_cv_logreg(Cal_tr, ytr, inner_folds)
        a_sim, s_sim = inner_cv_ridge(Xtr, ytr, inner_folds)

        clf_l2 = LogisticRegression(C=c_l2, solver="lbfgs", max_iter=500)
        clf_l2.fit(Xtr, ytr)
        pred_l2_va = clf_l2.predict_proba(Xva)[:, 1]
        pred_l2_te = clf_l2.predict_proba(Xte)[:, 1]

        clf_cal = LogisticRegression(C=c_cal, solver="lbfgs", max_iter=500)
        clf_cal.fit(Cal_tr, ytr)
        pred_cal_va = clf_cal.predict_proba(Cal_va)[:, 1]
        pred_cal_te = clf_cal.predict_proba(Cal_te)[:, 1]

        rg = Ridge(alpha=a_sim, fit_intercept=True)
        rg.fit(Xtr, ytr)
        # simplex-ish: clip coefs to >=0 and renormalize contribution
        coef = np.clip(rg.coef_.ravel(), 0, None)
        if coef.sum() <= 1e-12:
            coef = np.ones(M) / M
        else:
            coef = coef / coef.sum()
        pred_sim_va = Xva @ coef
        pred_sim_te = Xte @ coef

        # ensemble of L2 + Cal
        pred_ens_va = 0.5 * (pred_l2_va + pred_cal_va)
        pred_ens_te = 0.5 * (pred_l2_te + pred_cal_te)

        oof_L2[vam] = pred_l2_va
        oof_Cal[vam] = pred_cal_va
        oof_Sim[vam] = pred_sim_va
        oof_Ens[vam] = pred_ens_va
        te_L2 += pred_l2_te / 5
        te_Cal += pred_cal_te / 5
        te_Sim += pred_sim_te / 5
        te_Ens += pred_ens_te / 5

        # family weight attribution from L2 coefs
        w = clf_l2.coef_.ravel()
        fam_w = {}
        for fam in sorted(set(families)):
            idx = [i for i, f2 in enumerate(families) if f2 == fam]
            fam_w[fam] = float(np.sum(np.abs(w[idx]))) if idx else 0.0
        # named members
        named = {}
        for key in ("tabm", "uid_v2", "realmlp", "rmlp", "ebm", "xgb_llr", "l7", "gpt2"):
            idxs = [i for i, n in enumerate(names) if key in n.lower()]
            named[key] = {names[i]: float(w[i]) for i in idxs} if idxs else {}
        weight_logs.append({
            "fold": f,
            "C_L2": c_l2, "inner_L2": s_l2,
            "C_Cal": c_cal, "inner_Cal": s_cal,
            "alpha_Sim": a_sim, "inner_Sim": s_sim,
            "family_abs_weight": fam_w,
            "named_weights": named,
            "top_abs_coefs": sorted(
                [{"name": names[i], "family": families[i], "coef": float(w[i])} for i in range(M)],
                key=lambda r: abs(r["coef"]), reverse=True,
            )[:15],
        })
        log(f"outer fold {f}: L2 C={c_l2} inner={s_l2:.5f} Cal C={c_cal} Sim a={a_sim}")

    # pick nested winner by OOF AUC
    cands = {
        "Meta_L2_LogReg": (oof_L2, te_L2),
        "Meta_Cal_LogOdds": (oof_Cal, te_Cal),
        "Meta_Simplex_Ridge": (oof_Sim, te_Sim),
        "Meta_Ens_L2_Cal": (oof_Ens, te_Ens),
    }
    scores = {k: float(roc_auc_score(y, v[0])) for k, v in cands.items()}
    winner = max(scores, key=scores.get)
    return {
        "candidates_oof_auc": scores,
        "winner": winner,
        "oof": {k: v[0] for k, v in cands.items()},
        "test": {k: v[1] for k, v in cands.items()},
        "weight_logs": weight_logs,
        "S_own_nested_oof": cands[winner][0],
        "S_own_nested_test": cands[winner][1],
    }


def main():
    assert NO_SUBMIT is True
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log("STEP5 start NO_SUBMIT=True")

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(float)
    folds = pd.read_csv(FOLDS_P)["fold"].to_numpy(int)
    te_df = pd.read_csv(DATA / "test.csv")
    tr_df = pd.read_csv(DATA / "train.csv")
    ids = te_df["id"].to_numpy()
    Zg_tr = z_goodpjw(tr_df)
    Zg_te = z_goodpjw(te_df)
    assert Zg_te.sum() == 16_730

    ab_path = OUT / "F1b_pub_Step_AB_095569.csv"
    if not ab_path.exists():
        ab_path = EXT / "goodpjw_95569/submission.csv"
    assert sha256_file(ab_path) == SHA_AB
    ab = pd.read_csv(ab_path)["is_fraud"].to_numpy(float)

    part12_o = np.load(PART12 / "wide_lm_stack_nested.npy").ravel().astype(float)
    part12_t = np.load(PART12 / "wide_lm_stack_test.npy").ravel().astype(float)
    fb_o = np.load(SSD / "preds/z_fullbank_oof.npy").ravel().astype(float)
    log(f"Part12 AUC={roc_auc_score(y, part12_o):.6f} fullbank={roc_auc_score(y, fb_o):.6f}")

    # 1) assemble + hygiene
    rows, Xo_raw, Xt_raw = assemble_candidates(y, folds)
    cand_meta = [r for r in rows if r.get("status") == "CAND"]
    log(f"candidates after hygiene: {len(cand_meta)} / scanned {len(rows)}")

    kept_meta, Xo, Xt, dedup_info = dedup(cand_meta, Xo_raw, Xt_raw, y)
    names = [m["name"] for m in kept_meta]
    families = [m["family"] for m in kept_meta]
    log(f"after dedup: {len(kept_meta)} | by_family={dedup_info['by_family']}")

    # 2) nested stack
    nest = nested_stack(Xo, Xt, y, folds, names, families)
    S_o = nest["S_own_nested_oof"]
    S_t = nest["S_own_nested_test"]
    winner = nest["winner"]
    log(f"winner={winner} OOF={nest['candidates_oof_auc'][winner]:.6f}")

    # Zr on own nested
    S_o_zr = apply_zr(S_o, Zg_tr)
    S_t_zr = apply_zr(S_t, Zg_te)
    p12_o_zr = apply_zr(part12_o, Zg_tr)
    p12_t_zr = apply_zr(part12_t, Zg_te)

    meta_report = {}
    for k, oof in nest["oof"].items():
        oof_zr = apply_zr(oof, Zg_tr)
        meta_report[k] = {
            "oof_auc": float(roc_auc_score(y, oof)),
            "oof_auc_zr": float(roc_auc_score(y, oof_zr)),
            "folds": fold_aucs(y, oof, folds),
            "folds_zr": fold_aucs(y, oof_zr, folds),
            "sf25_vs_part12": sf25_beats(y, oof_zr, p12_o_zr, folds),
            "sf25_vs_fullbank": sf25_beats(y, oof_zr, apply_zr(fb_o, Zg_tr), folds),
        }
    log(f"own nested+Zr AUC={meta_report[winner]['oof_auc_zr']:.6f} "
        f"sf25 vs P12={meta_report[winner]['sf25_vs_part12']}/25")

    # 3) Yash divergence
    div = {
        "1mrho_oof_e4": one_mrho(S_o, part12_o) * 1e4,
        "1mrho_oof_zr_e4": one_mrho(S_o_zr, p12_o_zr) * 1e4,
        "1mrho_test_e4": one_mrho(S_t, part12_t) * 1e4,
        "1mrho_test_zr_e4": one_mrho(S_t_zr, p12_t_zr) * 1e4,
        "1mrho_test_nonzone_e4": one_mrho(S_t[~Zg_te], part12_t[~Zg_te]) * 1e4,
    }
    div["ratio_test_over_oof"] = div["1mrho_test_e4"] / max(div["1mrho_oof_e4"], 1e-12)
    div["ratio_test_over_oof_zr"] = div["1mrho_test_zr_e4"] / max(div["1mrho_oof_zr_e4"], 1e-12)
    log(f"divergence oof={div['1mrho_oof_e4']:.3f}e-4 test={div['1mrho_test_e4']:.3f}e-4 "
        f"ratio={div['ratio_test_over_oof']:.2f}")

    # nested w* blend
    w_grid = [0.0, 0.25, 0.35, 0.50, 0.65, 0.75, 1.0]
    Ls = L_fold(S_o, folds)
    Lp = L_fold(part12_o, folds)
    # fixed w OOF (with Zr at end)
    fixed = {}
    for w in w_grid:
        blend = U((1 - w) * Ls + w * Lp)
        blend_zr = apply_zr(blend, Zg_tr)
        fixed[str(w)] = {
            "oof_auc": float(roc_auc_score(y, blend)),
            "oof_auc_zr": float(roc_auc_score(y, blend_zr)),
            "folds_zr": fold_aucs(y, blend_zr, folds),
            "sf25_vs_own": sf25_beats(y, blend_zr, S_o_zr, folds),
            "sf25_vs_part12": sf25_beats(y, blend_zr, p12_o_zr, folds),
        }

    # nested w_f*: inner-4 CV pick w inside each outer fold
    oof_wstar = np.zeros(N_TR)
    te_wstar = np.zeros(N_TE)
    w_per_fold = []
    Ls_t = L(S_t)
    Lp_t = L(part12_t)
    for f in range(5):
        trm, vam = folds != f, folds == f
        inner = folds[trm]
        uniq = sorted(np.unique(inner))
        best_w, best_sc = 0.5, -1.0
        for w in w_grid:
            scores = []
            for u in uniq:
                itrm, ivam = inner != u, inner == u
                # map to global indices
                gtr = np.where(trm)[0]
                iva = gtr[ivam]
                blend = U((1 - w) * Ls[iva] + w * Lp[iva])
                # zr within val subset awkward — apply global zr mask on those rows via score demotion
                # simpler: evaluate without zr for inner select, zr only at end (consistent)
                scores.append(roc_auc_score(y[iva], blend))
            sc = float(np.mean(scores))
            if sc > best_sc:
                best_sc, best_w = sc, w
        w_per_fold.append({"fold": f, "w_star": best_w, "inner_auc": best_sc})
        oof_wstar[vam] = U((1 - best_w) * Ls[vam] + best_w * Lp[vam])
        te_wstar += U((1 - best_w) * Ls_t + best_w * Lp_t) / 5  # approx; better blend logits then U once
        log(f"w* fold {f}: w={best_w} inner={best_sc:.5f}")

    # rebuild test w* properly: average fold-specific logit blends then U
    te_logit = np.zeros(N_TE)
    for rec in w_per_fold:
        w = rec["w_star"]
        te_logit += ((1 - w) * Ls_t + w * Lp_t) / 5
    te_wstar = U(te_logit)
    oof_wstar_zr = apply_zr(oof_wstar, Zg_tr)
    te_wstar_zr = apply_zr(te_wstar, Zg_te)

    wstar_report = {
        "per_fold": w_per_fold,
        "mean_w": float(np.mean([r["w_star"] for r in w_per_fold])),
        "oof_auc": float(roc_auc_score(y, oof_wstar)),
        "oof_auc_zr": float(roc_auc_score(y, oof_wstar_zr)),
        "folds_zr": fold_aucs(y, oof_wstar_zr, folds),
        "sf25_vs_own": sf25_beats(y, oof_wstar_zr, S_o_zr, folds),
        "sf25_vs_part12": sf25_beats(y, oof_wstar_zr, p12_o_zr, folds),
        "fixed_grid": fixed,
    }
    log(f"nested w* mean={wstar_report['mean_w']:.3f} OOF+Zr={wstar_report['oof_auc_zr']:.6f}")

    # 4) stage
    staged = {}
    # N1
    sha, p = write_u(OUT / "N1_own_nested_stack_zr.csv", ids, apply_zr(S_t, Zg_te, U(S_t)))
    staged["N1_own_nested_stack_zr.csv"] = {
        "sha": sha, "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4,
        "1mrho_vs_part12_e4": one_mrho(p, p12_t_zr) * 1e4,
        "meta": winner,
    }
    # N2
    sha, p = write_u(OUT / "N2_dual_nested_x_part12_cv_zr.csv", ids, te_wstar_zr)
    staged["N2_dual_nested_x_part12_cv_zr.csv"] = {
        "sha": sha, "mean_w_part12": wstar_report["mean_w"],
        "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4,
        "1mrho_vs_N1_e4": one_mrho(p, staged["N1_own_nested_stack_zr.csv"] and U(apply_zr(S_t, Zg_te))) * 1e4,
    }
    # fix N2 vs N1 using stored
    n1 = pd.read_csv(OUT / "N1_own_nested_stack_zr.csv")["is_fraud"].to_numpy()
    n2 = pd.read_csv(OUT / "N2_dual_nested_x_part12_cv_zr.csv")["is_fraud"].to_numpy()
    staged["N2_dual_nested_x_part12_cv_zr.csv"]["1mrho_vs_N1_e4"] = one_mrho(n2, n1) * 1e4

    # N3 50/50
    n3_score = apply_zr(U(0.5 * L(S_t) + 0.5 * L(part12_t)), Zg_te)
    sha, p = write_u(OUT / "N3_dual_nested_x_part12_50_50_zr.csv", ids, n3_score)
    staged["N3_dual_nested_x_part12_50_50_zr.csv"] = {
        "sha": sha,
        "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4,
        "1mrho_vs_N2_e4": one_mrho(p, n2) * 1e4,
    }

    # Hedges
    for name, w_ab in [("H1_private_hedge_50AB_50N2.csv", 0.50), ("H2_private_hedge_70AB_30N2.csv", 0.70)]:
        # Zr preserve: blend then re-apply Zr with AB zone order
        blend = U(w_ab * L(ab) + (1 - w_ab) * L(n2))
        blend = apply_zr(blend, Zg_te, U(ab))
        sha, p = write_u(OUT / name, ids, blend)
        staged[name] = {
            "sha": sha, "w_AB": w_ab,
            "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4,
            "1mrho_vs_N2_e4": one_mrho(p, n2) * 1e4,
        }
        log(f"staged {name} 1-ρ vs AB e4={staged[name]['1mrho_vs_AB_e4']:.4f}")

    for k, v in staged.items():
        log(f"staged {k}: sha16={v['sha'][:16]}")

    # inventory summary
    inv_before = {}
    for r in rows:
        fam = r.get("family", "Other")
        inv_before[fam] = inv_before.get(fam, 0) + (1 if r.get("status") == "CAND" else 0)
    # recount candidates before dedup from cand_meta families
    inv_table = dedup_info["by_family"]

    payload = {
        "NO_SUBMIT": True,
        "inventory_rows": rows,
        "kept": kept_meta,
        "dedup": dedup_info,
        "inventory_by_family": inv_table,
        "n_kept": len(kept_meta),
        "nested": {
            "winner": winner,
            "candidates_oof_auc": nest["candidates_oof_auc"],
            "meta_report": meta_report,
            "weight_logs": nest["weight_logs"],
        },
        "divergence": div,
        "wstar": wstar_report,
        "staged": staged,
        "part12_oof_auc": float(roc_auc_score(y, part12_o)),
        "part12_oof_auc_zr": float(roc_auc_score(y, p12_o_zr)),
        "step3_note": "uid_catboost excluded; uid_v2 included if hygiene pass",
    }
    (LOG / "STEP5_AUDIT.json").write_text(json.dumps(payload, indent=2, default=str))

    lines = [
        "# V41 STEP5 AUDIT — Nested Dual Stack × Part12 + Zr",
        "",
        f"**NO_SUBMIT={NO_SUBMIT}** · Hard stop. Do not run Step 6 without `PROCEED TO STEP 6`.",
        "",
        "## Executive findings",
        "",
        f"- Kept columns after hygiene+dedup: **{len(kept_meta)}**",
        f"- Inventory by family: `{json.dumps(inv_table)}`",
        f"- Nested winner: **{winner}** OOF={nest['candidates_oof_auc'][winner]:.6f} → +Zr **{meta_report[winner]['oof_auc_zr']:.6f}**",
        f"- Part12 OOF={roc_auc_score(y, part12_o):.6f} → +Zr **{roc_auc_score(y, p12_o_zr):.6f}**",
        f"- Divergence `1−ρ×1e−4`: OOF={div['1mrho_oof_e4']:.3f}, test={div['1mrho_test_e4']:.3f}, "
        f"ratio={div['ratio_test_over_oof']:.2f}",
        f"- Nested `w*` mean Part12 weight={wstar_report['mean_w']:.3f}; dual+Zr OOF=**{wstar_report['oof_auc_zr']:.6f}**",
        f"- Staged: N1/N2/N3 + H1/H2 under `out/v41_staged/`",
        "",
        "## 1. Inventory & quarantine",
        "",
        "```json",
        json.dumps({"by_family": inv_table, "dedup_dropped_n": len(dedup_info["dropped"]),
                    "quarantine_or_fail_n": sum(1 for r in rows if r.get("status") != "CAND")}, indent=2),
        "```",
        "",
        "### Kept members (name / family / auc / KS / disp)",
        "",
        "| name | family | auc | KS | disp |",
        "|---|---|---:|---:|---:|",
    ]
    for m in kept_meta:
        lines.append(f"| `{m['name']}` | {m['family']} | {m['auc']:.5f} | {m['KS']:.4f} | {m['disp']:.3f} |")

    lines += [
        "",
        "## 2. Nested meta-stackers",
        "",
        "```json",
        json.dumps(meta_report, indent=2),
        "```",
        "",
        "### Weight attribution (per outer fold, top coefs)",
        "",
        "```json",
        json.dumps(nest["weight_logs"], indent=2)[:25000],
        "```",
        "",
        "## 3. Yash divergence & w*",
        "",
        "```json",
        json.dumps({"divergence": div, "wstar": {
            "per_fold": w_per_fold, "mean_w": wstar_report["mean_w"],
            "oof_auc_zr": wstar_report["oof_auc_zr"], "folds_zr": wstar_report["folds_zr"],
            "sf25_vs_own": wstar_report["sf25_vs_own"], "sf25_vs_part12": wstar_report["sf25_vs_part12"],
            "fixed_grid_zr": {k: v["oof_auc_zr"] for k, v in fixed.items()},
        }}, indent=2),
        "```",
        "",
        "## 4. Staged files",
        "",
        "```json",
        json.dumps(staged, indent=2),
        "```",
        "",
        "## 5. STOP",
        "",
        "Step 5 complete. **Awaiting human audit.** Reply `PROCEED TO STEP 6` to continue.",
        "",
    ]
    (LOG / "STEP5_AUDIT.md").write_text("\n".join(lines))

    print("\n=== STRICT NO_SUBMIT=True | V41 STEP5 ===")
    print(f"kept={len(kept_meta)} winner={winner}")
    print(f"own+Zr={meta_report[winner]['oof_auc_zr']:.6f} dual_w*+Zr={wstar_report['oof_auc_zr']:.6f}")
    print(f"div oof/test e4={div['1mrho_oof_e4']:.3f}/{div['1mrho_test_e4']:.3f} ratio={div['ratio_test_over_oof']:.2f}")
    print(f"WROTE {LOG/'STEP5_AUDIT.md'}")
    print("STOP. Awaiting PROCEED TO STEP 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
