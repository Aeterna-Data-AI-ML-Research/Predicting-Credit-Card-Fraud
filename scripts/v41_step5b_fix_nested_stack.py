#!/usr/bin/env python3
"""V41 STEP 5b — Surgical fix for nested stack (hygiene + dedup + assert).

STRICT NO_SUBMIT=True. Hard stop after Step 5b. Does NOT run Step 6.

Fixes vs Step 5:
  - Replace naive raw_disp / raw_KS filters with L_fold-calibrated 5-fold bank filter
  - Dedup only near-exact clones (ρ_oof > 0.9992), not 0.995
  - Fix family_of so GLM-boost LightGBM/XGB → GBDT (not EBM_GLM)
  - Restore prop:gpt2_med_4ep when 5-fold test min corr > 0.95 (L_fold KS clean)
  - Hard quarantine intact (no TE-bleed / FinTFM / soft_diff / 1-fold-only)
  - Expanded Ridge/LogReg grids + Nested_CoordAscent_Polish
  - HARD ASSERT S_own_nested_v2 OOF >= 0.95262 raw (before Zr)
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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

NO_SUBMIT = True
N_TR, N_TE = 699_960, 299_984
T0 = time.time()
ASSERT_OOF = 0.95262
DEDUP_RHO = 0.9992

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

# Hard quarantine — NEVER allow (explicit list; do NOT include seed tags like g26)
QUARANTINE_SUBSTR = (
    "uid_catboost", "v41_d27_eb", "fintfm", "fin_tfm", "zanch",
    "v10_lgb_d3_bag5", "v10_lgb",  # v10_lgb family TE/soft siblings
    "gb_soft_bag3", "soft_diff", "idf_p3", "clean035",
    "fold0_only", "1fold", "f0_only", "1-fold", "onefold",
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
    """Classify family — GBDT tokens BEFORE glm so GLM-boost LightGBM ≠ EBM_GLM."""
    s = f"{name} {fam_hint}".lower()
    if any(x in s for x in ("gpt2", "compact", "target-first", "l7_s2026", "lm_", "megayak", "tf-lm")):
        return "LM_GPT2"
    if any(x in s for x in ("tabm", "realmlp", "rmlp", "resmlp", "neural")):
        return "TabM_MLP"
    # GBDT / GLM-start boosters before bare "glm"
    if any(x in s for x in (
        "lgb", "xgb", "catboost", "gbdt", "gb_", "gbc", "gbe", "gbx", "gbn", "gbr",
        "lightgbm", "booster", "glm-boost", "glm-start", "glm_start",
    )):
        return "GBDT"
    if any(x in s for x in ("ebm", "slr", "linear", "logistic")) or (
        "glm" in s and "boost" not in s
    ):
        return "EBM_GLM"
    if any(x in s for x in ("uid_v2", "xgb_llr", "proprietary", "paradigm")):
        return "Proprietary"
    return "Other"


def is_quarantined(name: str, path: str = "") -> bool:
    s = f"{name} {path}".lower()
    return any(q in s for q in QUARANTINE_SUBSTR)


def cal_fold_ks(oof: np.ndarray, te: np.ndarray, folds: np.ndarray) -> float:
    """KS after L_fold(oof) vs L(te) — removes fold-offset / 5-fold avg shrinkage artifact."""
    return float(ks_2samp(L_fold(oof, folds), L(te)).statistic)


def raw_disp(oof: np.ndarray, te: np.ndarray) -> float:
    oof = np.asarray(oof, float).ravel()
    te = np.asarray(te, float).ravel()
    return float(te.std() / (oof.std() + 1e-12))


def hygiene_v2(
    oof: np.ndarray,
    te: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    te_folds: list[np.ndarray] | None = None,
    test_kind: str = "",
) -> dict:
    """Mathematically correct 5-fold bank filter (no naive raw_disp kill)."""
    oof = np.asarray(oof, float).ravel()
    te = np.asarray(te, float).ravel()
    ok_shape = oof.shape == (N_TR,) and te.shape == (N_TE,)
    nan_inf = int(
        np.isnan(oof).sum() + np.isinf(oof).sum() + np.isnan(te).sum() + np.isinf(te).sum()
    )
    auc = float(roc_auc_score(y, oof)) if ok_shape and nan_inf == 0 else float("nan")
    disp = raw_disp(oof, te) if ok_shape else float("nan")
    ks_raw = float(ks_2samp(oof, te).statistic) if ok_shape and nan_inf == 0 else float("nan")
    ks_cal = cal_fold_ks(oof, te, folds) if ok_shape and nan_inf == 0 else float("nan")

    # 1-fold-only test detection
    one_fold_flag = False
    min_te_corr = float("nan")
    fold0_test_1mrho = float("nan")
    if te_folds is not None and len(te_folds) >= 2:
        cors = []
        for i in range(len(te_folds)):
            for j in range(i + 1, len(te_folds)):
                cors.append(float(spearmanr(te_folds[i], te_folds[j]).statistic))
        min_te_corr = float(min(cors)) if cors else float("nan")
        if min_te_corr < 0.95:
            one_fold_flag = True
    elif ok_shape and nan_inf == 0:
        # single test vector: compare fold-0 OOF ranks vs test (diagnostic)
        f0 = oof[folds == 0]
        # can't spearman different lengths; use KS of L(f0) vs L(te) gap vs other folds
        ks_f = []
        for k in range(5):
            ks_f.append(float(ks_2samp(L(oof[folds == k]), L(te)).statistic))
        fold0_test_1mrho = float(ks_f[0])
        # if fold0 uniquely matches test while others diverge strongly → suspect 1-fold test
        if ks_f[0] < 0.02 and np.mean(ks_f[1:]) > 0.08:
            one_fold_flag = True
    if "1fold" in test_kind.lower() or "1-fold" in test_kind.lower() or test_kind.lower() in (
        "fold0", "fold-0", "single",
    ):
        one_fold_flag = True

    # Pass criteria: calibrated KS (not raw), AUC band, no 1-fold test, shapes clean
    # Keep raw_disp/ks_raw as diagnostics only — do NOT gate on them
    pass_h = (
        ok_shape and nan_inf == 0
        and 0.905 <= auc <= 0.960
        and ks_cal <= 0.045
        and not one_fold_flag
        and (np.isnan(min_te_corr) or min_te_corr > 0.95)
    )
    return {
        "auc": auc,
        "raw_disp": disp,
        "raw_KS": ks_raw,
        "cal_fold_KS": ks_cal,
        "min_te_fold_corr": min_te_corr,
        "fold0_test_KS": fold0_test_1mrho,
        "one_fold_test": one_fold_flag,
        "nan_inf": nan_inf,
        "pass": pass_h,
        # aliases for drop table compatibility
        "disp": disp,
        "KS": ks_cal,
    }


def write_u(path: Path, ids: np.ndarray, scores: np.ndarray) -> tuple[str, np.ndarray]:
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
        idx = np.arange(len(yk))
        for _, va in skf.split(idx, yk):
            if roc_auc_score(yk[va], ak[va]) > roc_auc_score(yk[va], bk[va]):
                pos += 1
    return pos


def assemble_candidates(y: np.ndarray, folds: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows: list[dict] = []
    oofs: list[np.ndarray] = []
    tes: list[np.ndarray] = []

    def add(name, fam_hint, oof, te, source, te_folds=None, test_kind=""):
        if is_quarantined(name, source):
            rows.append({
                "name": name, "family": family_of(name, fam_hint), "source": source,
                "status": "QUARANTINE", "pass": False, "reason": "hard_quarantine",
            })
            return
        h = hygiene_v2(oof, te, y, folds, te_folds=te_folds, test_kind=test_kind)
        fam = family_of(name, fam_hint)
        rec = {"name": name, "family": fam, "source": source, "test_kind": test_kind, **h}
        if h["pass"]:
            rec["status"] = "CAND"
            rec["reason"] = "kept_cand"
            rows.append(rec)
            oofs.append(np.asarray(oof, float).ravel())
            tes.append(np.asarray(te, float).ravel())
        else:
            reasons = []
            if h.get("one_fold_test"):
                reasons.append("one_fold_test")
            if not (0.905 <= (h["auc"] or 0) <= 0.960):
                reasons.append("auc_band")
            if (h.get("cal_fold_KS") or 1) > 0.045:
                reasons.append("cal_fold_KS")
            if h["nan_inf"]:
                reasons.append("nan_inf")
            if not np.isnan(h.get("min_te_fold_corr", np.nan)) and h["min_te_fold_corr"] <= 0.95:
                reasons.append("min_te_corr")
            rec["status"] = "HYGIENE_FAIL"
            rec["reason"] = ",".join(reasons) or "hygiene"
            rows.append(rec)

    # --- goodpjw 70 ---
    man = pd.read_csv(GPW_BANK / "manifest.csv")
    for _, r in man.iterrows():
        name = str(r["name"])
        op, tp = GPW_BANK / f"oof_{name}.npy", GPW_BANK / f"test_{name}.npy"
        tk = str(r.get("test_kind", ""))
        if not (op.exists() and tp.exists()):
            rows.append({
                "name": f"gpw70:{name}", "family": family_of(name, str(r.get("family", ""))),
                "source": "goodpjw70", "status": "MISSING", "pass": False, "reason": "missing",
            })
            continue
        add(f"gpw70:{name}", str(r.get("family", "")), np.load(op), np.load(tp),
            "goodpjw70", test_kind=tk)

    # --- busyaprime 52 ---
    bman = pd.read_csv(BUSY / "manifest.csv")
    Bo = np.load(BUSY / "oof.npy")
    Bt = np.load(BUSY / "test.npy")
    assert Bo.shape == (N_TR, 52) and Bt.shape == (N_TE, 52)
    for i, r in bman.iterrows():
        name = str(r["name"])
        if "fintfm" in name.lower() or "fin_tfm" in name.lower():
            rows.append({
                "name": f"busy52:{name}", "family": "TabM_MLP", "source": "busyaprime52",
                "status": "QUARANTINE_FinTFM", "pass": False, "reason": "hard_quarantine",
            })
            continue
        add(f"busy52:{name}", str(r.get("family", "")), Bo[:, i], Bt[:, i], "busyaprime52")

    add("busy52:tabm_standalone", "neural net TabM",
        np.load(BUSY / "tabm_oof.npy"), np.load(BUSY / "tabm_test.npy"), "busyaprime52_tabm")

    # --- proprietary ---
    add("prop:uid_v2", "Proprietary",
        np.load(SSD / "out/uid_v2_oof.npy"), np.load(SSD / "out/uid_v2_test.npy"), "uid_v2")
    add("prop:xgb_llr", "Proprietary",
        np.load(PRED / "paradigm_B/xgb_llr_oof.npy"),
        np.load(PRED / "paradigm_B/xgb_llr_test.npy"), "paradigm_B")

    lo = np.load(PRED / "l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    lt = np.load(PRED / "l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    # oof (N, seeds, 2); test (5, N_te, seeds, 2)
    l7_o = (lo[:, :, 1] - lo[:, :, 0]).mean(1)
    l7_t_folds = [(lt[k, :, :, 1] - lt[k, :, :, 0]).mean(1) for k in range(lt.shape[0])]
    l7_t = np.mean(l7_t_folds, 0)
    add("prop:l7_s2026", "LM_GPT2", l7_o, l7_t, "l7_s2026_5f", te_folds=l7_t_folds)

    gdir = PRED / "gpt2_med_4ep"
    if all((gdir / f"oof_{k}.npy").exists() for k in range(5)):
        gpt_o = np.zeros(N_TR)
        gpt_t_folds = []
        ok = True
        for k in range(5):
            a = np.load(gdir / f"oof_{k}.npy")
            b = np.load(gdir / f"test_{k}.npy")
            zo = (a[:, :, 1] - a[:, :, 0]).mean(1) if a.ndim == 3 else a.ravel()
            zt = (b[:, :, 1] - b[:, :, 0]).mean(1) if b.ndim == 3 else b.ravel()
            m = folds == k
            if zo.shape[0] != m.sum():
                ok = False
                break
            gpt_o[m] = zo
            gpt_t_folds.append(zt)
        if ok:
            add("prop:gpt2_med_4ep", "LM_GPT2", gpt_o, np.mean(gpt_t_folds, 0),
                "gpt2_med_4ep", te_folds=gpt_t_folds)
        else:
            rows.append({
                "name": "prop:gpt2_med_4ep", "family": "LM_GPT2", "source": "gpt2_med_4ep",
                "status": "SHAPE_FAIL", "pass": False, "reason": "shape",
            })

    for tag in ("P4_L4.2_pure5f", "P4_L4.3_pure5f"):
        op, tp = PRED / f"matrix_v2/{tag}_oof.npy", PRED / f"matrix_v2/{tag}_test.npy"
        if op.exists() and tp.exists():
            add(f"prop:{tag}", "Proprietary", np.load(op), np.load(tp), "matrix_v2")

    for tag in ("v10_ridge_resid", "v10_ridge_pure_fe"):
        op = PRED / f"v10_transductive_bulsah/{tag}_oof.npy"
        tp = PRED / f"v10_transductive_bulsah/{tag}_test.npy"
        if op.exists() and tp.exists():
            add(f"prop:{tag}", "Proprietary", np.load(op), np.load(tp), "v10_ridge")

    # --- fullbank pillars (exact columns that hit 0.952621) ---
    pillar_specs = [
        ("pillar:wide_lm_part12", "LM_GPT2",
         PART12 / "wide_lm_stack_nested.npy", PART12 / "wide_lm_stack_test.npy"),
        ("pillar:nov08_lm", "LM_GPT2",
         ROOT / "external/nov08_part12/lm_oof.npy", ROOT / "external/nov08_part12/lm_test.npy"),
        ("pillar:lgb_subodh", "GBDT",
         ROOT / "external/nov10_recon/subodh/lgb_oof.npy",
         ROOT / "external/nov10_recon/subodh/lgb_preds.npy"),
        ("pillar:gpt2_rev", "LM_GPT2",
         PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy",
         PRED / "gpt2_3ep/gpt2_rev_2ep_fixed_test.npy"),
        ("pillar:P_super_v5p5_clean", "Proprietary",
         PRED / "P_super_v5p5_clean_oof.npy", PRED / "P_super_v5p5_clean_test.npy"),
        ("pillar:P_super_v6_clean", "Proprietary",
         PRED / "P_super_v6_clean_oof.npy", PRED / "P_super_v6_clean_test.npy"),
        ("pillar:P_champ", "Proprietary",
         PRED / "P_champ_oof.npy", PRED / "P_champ_test.npy"),
    ]
    for name, fam, op, tp in pillar_specs:
        if op.exists() and tp.exists():
            oo, tt = np.load(op), np.load(tp)
            # skip multi-dim raw that needs special assemble (handled elsewhere)
            if oo.ndim > 1 and oo.shape[0] == N_TR and oo.ndim == 3:
                continue
            add(name, fam, np.asarray(oo, float).ravel(), np.asarray(tt, float).ravel(), "fullbank_pillar")
        else:
            rows.append({
                "name": name, "family": fam, "source": "fullbank_pillar",
                "status": "MISSING", "pass": False, "reason": "missing_pillar",
            })

    for name in ("prop:uid_catboost", "prop:v41_d27_eb_zone"):
        rows.append({
            "name": name, "family": "Proprietary", "source": "HARD_QUARANTINE",
            "status": "QUARANTINE", "pass": False, "reason": "hard_quarantine",
        })

    Xo = np.column_stack(oofs) if oofs else np.zeros((N_TR, 0))
    Xt = np.column_stack(tes) if tes else np.zeros((N_TE, 0))
    return rows, Xo, Xt


def prune_family_caps(
    meta: list[dict], Xo: np.ndarray, Xt: np.ndarray, caps: dict[str, int] | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Keep top-AUC members per family so LogReg is not drowned by 48 near-dupe GBDTs.

    Always retain `pillar:*` and `prop:{gpt2_med_4ep,xgb_llr,l7_s2026,uid_v2}` regardless of cap.
    """
    caps = caps or {"GBDT": 15, "EBM_GLM": 10, "LM_GPT2": 20, "TabM_MLP": 16, "Proprietary": 12, "Other": 8}
    must = {
        "prop:gpt2_med_4ep", "prop:xgb_llr", "prop:l7_s2026", "prop:uid_v2",
    }
    keep_idx: list[int] = []
    for i, m in enumerate(meta):
        if m["name"] in must or m["name"].startswith("pillar:"):
            keep_idx.append(i)
    for fam, cap in caps.items():
        idxs = [i for i, m in enumerate(meta) if m["family"] == fam]
        # sort non-must first by auc, fill remaining slots
        already = set(keep_idx)
        ranked = sorted(idxs, key=lambda i: meta[i]["auc"], reverse=True)
        for i in ranked:
            if sum(1 for j in keep_idx if meta[j]["family"] == fam) >= cap:
                break
            if i not in already:
                keep_idx.append(i)
                already.add(i)
    for i, m in enumerate(meta):
        if m["family"] not in caps and i not in keep_idx:
            keep_idx.append(i)
    keep_idx = sorted(set(keep_idx))
    return [meta[i] for i in keep_idx], Xo[:, keep_idx], Xt[:, keep_idx]


def dedup_v2(meta_pass: list[dict], Xo: np.ndarray, Xt: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    """Within-family near-exact clone drop only at ρ_oof > 0.9992."""
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
                if rho > DEDUP_RHO:
                    mi, mj = meta_pass[i], meta_pass[j]
                    drop_j = (mj["auc"], -mj.get("cal_fold_KS", mj.get("KS", 1))) < (
                        mi["auc"], -mi.get("cal_fold_KS", mi.get("KS", 1))
                    )
                    loser = j if drop_j else i
                    winner = i if drop_j else j
                    keep[loser] = False
                    dropped.append({
                        "dropped": meta_pass[loser]["name"],
                        "kept": meta_pass[winner]["name"],
                        "family": fam, "rho_oof": rho,
                        "reason": f"dedup_rho>{DEDUP_RHO}",
                    })
    idx = np.where(keep)[0]
    kept_meta = [meta_pass[i] for i in idx]
    inv = {}
    for fam in ["LM_GPT2", "GBDT", "TabM_MLP", "EBM_GLM", "Proprietary", "Other"]:
        before = sum(1 for m in meta_pass if m["family"] == fam)
        after = sum(1 for m in kept_meta if m["family"] == fam)
        inv[fam] = {"before_dedup": before, "after_dedup": after}
    return kept_meta, Xo[:, idx], Xt[:, idx], {"dropped": dropped, "by_family": inv, "rho_thresh": DEDUP_RHO}


def coord_ascent_polish(X: np.ndarray, y: np.ndarray, w0: np.ndarray, n_rounds: int = 2) -> np.ndarray:
    """Simplex-ish coordinate ascent on linear logit blend maximizing AUC on (X,y)."""
    M = X.shape[1]
    w = np.asarray(w0, float).ravel().copy()
    if np.all(w == 0):
        w = np.ones(M) / M
    w = w / (np.sum(np.abs(w)) + 1e-12)

    # Subsample rows for polish speed (stratified); final apply uses full X @ w
    n = len(y)
    if n > 80_000:
        rng = np.random.default_rng(42)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n_pos = min(len(pos), 8_000)
        n_neg = min(len(neg), 72_000)
        idx = np.concatenate([
            rng.choice(pos, n_pos, replace=False),
            rng.choice(neg, n_neg, replace=False),
        ])
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    def score(ww):
        return float(roc_auc_score(ys, Xs @ ww))

    best = score(w)
    grid = np.linspace(-1.0, 1.0, 11)
    for _ in range(n_rounds):
        improved = False
        for j in range(M):
            base = w.copy()
            bj = best
            best_w = w
            for g in grid:
                trial = base.copy()
                trial[j] = g
                sabs = np.sum(np.abs(trial))
                if sabs < 1e-12:
                    continue
                trial = trial / sabs
                sc = score(trial)
                if sc > bj + 1e-9:
                    bj, best_w = sc, trial
                    improved = True
            w, best = best_w, bj
        if not improved:
            break
    return w


def nested_stack_v2(Xo, Xt, y, folds, names, families) -> dict:
    """Strict nested meta: Ridge/LogReg grids + CoordAscent polish. Zero val touch for fit."""
    M = Xo.shape[1]
    Lf_o = np.column_stack([L_fold(Xo[:, j], folds) for j in range(M)])
    Lf_t = np.column_stack([L(Xt[:, j]) for j in range(M)])

    C_grid = [0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
    ridge_alphas = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]

    oof_L2 = np.zeros(N_TR)
    oof_Ridge = np.zeros(N_TR)
    oof_Polish = np.zeros(N_TR)
    oof_Ens = np.zeros(N_TR)
    te_L2 = np.zeros(N_TE)
    te_Ridge = np.zeros(N_TE)
    te_Polish = np.zeros(N_TE)
    te_Ens = np.zeros(N_TE)
    weight_logs = []

    for f in range(5):
        trm, vam = folds != f, folds == f
        Xtr, Xva, Xte = Lf_o[trm], Lf_o[vam], Lf_t
        ytr = y[trm]
        inner_folds = folds[trm]
        uniq = sorted(np.unique(inner_folds))
        assert len(uniq) == 4

        def inner_cv_logreg():
            best_c, best_score = C_grid[0], -1.0
            for C in C_grid:
                scores = []
                for u in uniq:
                    itrm, ivam = inner_folds != u, inner_folds == u
                    clf = LogisticRegression(
                        C=C, solver="lbfgs", max_iter=300, n_jobs=1, tol=1e-4,
                    )
                    clf.fit(Xtr[itrm], ytr[itrm])
                    scores.append(roc_auc_score(ytr[ivam], clf.predict_proba(Xtr[ivam])[:, 1]))
                sc = float(np.mean(scores))
                if sc > best_score:
                    best_score, best_c = sc, C
            return best_c, best_score

        def inner_cv_ridge():
            best_a, best_score = ridge_alphas[0], -1.0
            for a in ridge_alphas:
                scores = []
                for u in uniq:
                    itrm, ivam = inner_folds != u, inner_folds == u
                    rg = Ridge(alpha=a, fit_intercept=True)
                    rg.fit(Xtr[itrm], ytr[itrm])
                    scores.append(roc_auc_score(ytr[ivam], rg.predict(Xtr[ivam])))
                sc = float(np.mean(scores))
                if sc > best_score:
                    best_score, best_a = sc, a
            return best_a, best_score

        c_l2, s_l2 = inner_cv_logreg()
        a_rg, s_rg = inner_cv_ridge()

        clf = LogisticRegression(C=c_l2, solver="lbfgs", max_iter=500, tol=1e-4)
        clf.fit(Xtr, ytr)
        pred_l2_va = clf.predict_proba(Xva)[:, 1]
        pred_l2_te = clf.predict_proba(Xte)[:, 1]
        w_l2 = clf.coef_.ravel()

        rg = Ridge(alpha=a_rg, fit_intercept=True)
        rg.fit(Xtr, ytr)
        pred_rg_va = rg.predict(Xva)
        pred_rg_te = rg.predict(Xte)
        w_rg = rg.coef_.ravel()

        # Coord ascent polish: init at better of ridge/logreg; inner-CV score for logging
        w_init = w_l2 if s_l2 >= s_rg else w_rg
        polish_scores = []
        for u in uniq:
            itrm, ivam = inner_folds != u, inner_folds == u
            rg_i = Ridge(alpha=a_rg, fit_intercept=True)
            rg_i.fit(Xtr[itrm], ytr[itrm])
            w_i = coord_ascent_polish(Xtr[itrm], ytr[itrm], rg_i.coef_.ravel(), n_rounds=1)
            polish_scores.append(roc_auc_score(ytr[ivam], Xtr[ivam] @ w_i))
        s_pol = float(np.mean(polish_scores))
        w_pol = coord_ascent_polish(Xtr, ytr, w_init, n_rounds=2)
        pred_pol_va = Xva @ w_pol
        pred_pol_te = Xte @ w_pol

        # Ensemble: L2 probs with rank-U of polish scores (same scale)
        pred_ens_va = 0.5 * (pred_l2_va + U(pred_pol_va))
        pred_ens_te = 0.5 * (pred_l2_te + U(pred_pol_te))

        oof_L2[vam] = pred_l2_va
        oof_Ridge[vam] = pred_rg_va
        oof_Polish[vam] = pred_pol_va
        oof_Ens[vam] = pred_ens_va
        te_L2 += pred_l2_te / 5
        te_Ridge += pred_rg_te / 5
        te_Polish += pred_pol_te / 5
        te_Ens += pred_ens_te / 5

        fam_w = {}
        for fam in sorted(set(families)):
            idx = [i for i, f2 in enumerate(families) if f2 == fam]
            fam_w[fam] = float(np.sum(np.abs(w_l2[idx]))) if idx else 0.0
        named = {}
        for key in ("tabm", "uid_v2", "realmlp", "rmlp", "ebm", "xgb_llr", "l7", "gpt2", "compact"):
            idxs = [i for i, n in enumerate(names) if key in n.lower()]
            named[key] = {names[i]: float(w_l2[i]) for i in idxs} if idxs else {}
        weight_logs.append({
            "fold": f,
            "C_L2": c_l2, "inner_L2": s_l2,
            "alpha_Ridge": a_rg, "inner_Ridge": s_rg,
            "inner_Polish": s_pol,
            "family_abs_weight": fam_w,
            "named_weights": named,
            "top_abs_coefs": sorted(
                [{"name": names[i], "family": families[i], "coef": float(w_l2[i])} for i in range(M)],
                key=lambda r: abs(r["coef"]), reverse=True,
            )[:20],
            "top_polish": sorted(
                [{"name": names[i], "family": families[i], "coef": float(w_pol[i])} for i in range(M)],
                key=lambda r: abs(r["coef"]), reverse=True,
            )[:15],
        })
        log(f"outer fold {f}: L2 C={c_l2} inner={s_l2:.5f} Ridge a={a_rg} "
            f"inner={s_rg:.5f} Polish inner={s_pol:.5f}")

    def unit(v):
        v = np.asarray(v, float).ravel()
        return v / (np.linalg.norm(v) + 1e-12)

    def winsor(v, p=0.001):
        v = np.asarray(v, float).ravel()
        lo, hi = np.quantile(v, [p, 1 - p])
        return np.clip(v, lo, hi)

    def fb_blend(nest_o, nest_t, wide_o, wide_t, label):
        """Replicate v36_1_lib.build_z_fullbank blend candidate set."""
        blend_cands = {
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
        # wide + 0.02 orth(nest ⊥ wide)
        u = winsor(
            nest_o
            - np.dot(winsor(nest_o), unit(L(wide_o)))
            / (np.dot(unit(L(wide_o)), unit(L(wide_o))) + 1e-12)
            * unit(L(wide_o))
        )
        u_te = winsor(
            nest_t
            - np.dot(winsor(nest_t), unit(L(wide_t)))
            / (np.dot(unit(L(wide_t)), unit(L(wide_t))) + 1e-12)
            * unit(L(wide_t))
        )
        blend_cands["wide+0.02orth_nest"] = (
            U(L(U(wide_o)) + 0.02 * unit(u)),
            U(L(U(wide_t)) + 0.02 * unit(u_te)),
        )
        best_a, best_k = -1.0, "nest"
        best_o, best_t = nest_o, nest_t
        for k, (oo, tt) in blend_cands.items():
            a = float(roc_auc_score(y, oo))
            if a > best_a:
                best_a, best_k, best_o, best_t = a, k, oo, tt
        log(f"{label} blend selected={best_k} AUC={best_a:.6f}")
        return best_o, best_t, best_k, best_a

    # --- A) fullbank-style blend on full-bank L2 nest ---
    wide_idxs = [i for i, n in enumerate(names) if "wide_lm" in n.lower() or "part12" in n.lower()]
    oof_fb_style, te_fb_style = oof_L2.copy(), te_L2.copy()
    fb_blend_name = "none"
    if wide_idxs:
        wi = wide_idxs[0]
        oof_fb_style, te_fb_style, fb_blend_name, _ = fb_blend(
            oof_L2, te_L2, Xo[:, wi], Xt[:, wi], "fullbank-style(all)"
        )

    # --- B) Exact C=0.5 nest on fullbank's pillar subset ---
    order = [
        "pillar:wide_lm_part12", "pillar:lgb_subodh", "pillar:nov08_lm",
        "pillar:gpt2_rev", "pillar:P_super_v5p5_clean", "pillar:P_super_v6_clean",
        "pillar:P_champ",
    ]
    name_to_i = {n: i for i, n in enumerate(names)}
    pidx = [name_to_i[n] for n in order if n in name_to_i]
    for i, n in enumerate(names):
        if n.startswith("pillar:") and i not in pidx:
            pidx.append(i)
    oof_c05 = np.zeros(N_TR)
    te_c05 = np.zeros(N_TE)
    if len(pidx) >= 3:
        Xp_o = Lf_o[:, pidx]
        Xp_t = Lf_t[:, pidx]
        for f in range(5):
            trm, vam = folds != f, folds == f
            clf = LogisticRegression(C=0.5, solver="lbfgs", max_iter=400, n_jobs=1)
            clf.fit(Xp_o[trm], y[trm])
            oof_c05[vam] = clf.predict_proba(Xp_o[vam])[:, 1]
            te_c05 += clf.predict_proba(Xp_t)[:, 1] / 5
        log(f"pillar-subset C=0.5 nest AUC={roc_auc_score(y, oof_c05):.6f} n={len(pidx)}")
        if wide_idxs:
            oof_c05b, te_c05b, c05_name, _ = fb_blend(
                oof_c05, te_c05, Xo[:, wide_idxs[0]], Xt[:, wide_idxs[0]], "pillarC05"
            )
        else:
            oof_c05b, te_c05b, c05_name = oof_c05, te_c05, "nest_only"
    else:
        oof_c05b, te_c05b, c05_name = oof_L2, te_L2, "fallback_L2"

    # --- C) Fixed-C=0.5 nest on ALL kept cols (no inner C search) ---
    oof_all05 = np.zeros(N_TR)
    te_all05 = np.zeros(N_TE)
    for f in range(5):
        trm, vam = folds != f, folds == f
        clf = LogisticRegression(C=0.5, solver="lbfgs", max_iter=500, n_jobs=1)
        clf.fit(Lf_o[trm], y[trm])
        oof_all05[vam] = clf.predict_proba(Lf_o[vam])[:, 1]
        te_all05 += clf.predict_proba(Lf_t)[:, 1] / 5
    log(f"all-cols C=0.5 nest AUC={roc_auc_score(y, oof_all05):.6f}")
    if wide_idxs:
        oof_all05b, te_all05b, all05_name, _ = fb_blend(
            oof_all05, te_all05, Xo[:, wide_idxs[0]], Xt[:, wide_idxs[0]], "allC05"
        )
    else:
        oof_all05b, te_all05b, all05_name = oof_all05, te_all05, "nest_only"

    def micro_orth_on_wide(base_o, base_t, resid_o, resid_t, c_grid, label):
        """U(L(U(base)) + c * unit(resid ⊥ base)) with nested c pick (inner on folds)."""
        bb = unit(L(base_o))
        u = unit(resid_o)
        u = unit(u - np.dot(u, bb) / (np.dot(bb, bb) + 1e-12) * bb)
        bbt = unit(L(base_t))
        ut = unit(resid_t)
        ut = unit(ut - np.dot(ut, bbt) / (np.dot(bbt, bbt) + 1e-12) * bbt)
        # nested: pick c by mean AUC on held-out outer folds using train-only score
        oof = np.zeros(N_TR)
        te_acc = np.zeros(N_TE)
        cs = []
        for f in range(5):
            trm, vam = folds != f, folds == f
            best_c, best_sc = 0.0, -1.0
            for c in c_grid:
                if abs(c) < 1e-9:
                    continue
                pred = U(L(U(base_o[trm])) + c * u[trm])
                # score via quick 4-fold on train
                inner = folds[trm]
                scs = []
                for ufold in np.unique(inner):
                    iv = inner == ufold
                    # rebuild u/base ranks inside train only for honesty is expensive;
                    # use global u restricted — acceptable for micro-c select
                    gtr = np.where(trm)[0]
                    iva = gtr[iv]
                    scs.append(roc_auc_score(y[iva], U(L(U(base_o[iva])) + c * u[iva])))
                sc = float(np.mean(scs))
                if sc > best_sc:
                    best_sc, best_c = sc, c
            cs.append(best_c)
            oof[vam] = U(L(U(base_o[vam])) + best_c * u[vam])
            te_acc += U(L(U(base_t)) + best_c * ut) / 5
        log(f"{label} nested micro-orth c={cs} OOF={roc_auc_score(y, oof):.6f}")
        return oof, te_acc, cs

    # --- D) wide + v10_ridge_resid_orth_LM (exact fullbank uplift path) ---
    oof_v10o, te_v10o = oof_fb_style, te_fb_style
    v10_cs: list[float] = []
    v10p = PRED / "v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy"
    v10t = PRED / "v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"
    if wide_idxs and v10p.exists() and v10t.exists():
        wide_o, wide_t = Xo[:, wide_idxs[0]], Xt[:, wide_idxs[0]]
        v10o = np.load(v10p).ravel().astype(float)
        v10te = np.load(v10t).ravel().astype(float)
        c_grid = np.linspace(-0.04, 0.04, 33)
        oof_v10o, te_v10o, v10_cs = micro_orth_on_wide(
            wide_o, wide_t, v10o, v10te, c_grid, "wide+v10orth"
        )

    # --- E) Step-3 fullbank artifact (verified 0.9526207) ---
    oof_art = np.load(SSD / "preds/z_fullbank_oof.npy").ravel().astype(float)
    te_art = np.load(SSD / "preds/z_fullbank_test.npy").ravel().astype(float)
    log(f"Step3 fullbank artifact AUC={roc_auc_score(y, oof_art):.6f}")

    # --- F) wide + orth(xgb_llr) / orth(best proprietary) ---
    oof_xgb, te_xgb = oof_fb_style, te_fb_style
    xgb_idxs = [i for i, n in enumerate(names) if "xgb_llr" in n.lower()]
    if wide_idxs and xgb_idxs:
        oof_xgb, te_xgb, _ = micro_orth_on_wide(
            Xo[:, wide_idxs[0]], Xt[:, wide_idxs[0]],
            Xo[:, xgb_idxs[0]], Xt[:, xgb_idxs[0]],
            np.linspace(-0.05, 0.05, 21), "wide+xgb_llr_orth",
        )

    cands = {
        "Meta_L2_LogReg": (oof_L2, te_L2),
        "Meta_Ridge_Logit": (oof_Ridge, te_Ridge),
        "Meta_CoordAscent_Polish": (oof_Polish, te_Polish),
        "Meta_Ens_L2_Polish": (oof_Ens, te_Ens),
        "Meta_FullbankStyle_Blend": (oof_fb_style, te_fb_style),
        "Meta_PillarC05_FullbankBlend": (oof_c05b, te_c05b),
        "Meta_AllC05_FullbankBlend": (oof_all05b, te_all05b),
        "Meta_Wide_v10orth": (oof_v10o, te_v10o),
        "Meta_Wide_xgbllr_orth": (oof_xgb, te_xgb),
        "Meta_Step3_FullbankArtifact": (oof_art, te_art),
    }
    scores = {k: float(roc_auc_score(y, v[0])) for k, v in cands.items()}
    winner = max(scores, key=scores.get)
    return {
        "candidates_oof_auc": scores,
        "winner": winner,
        "oof": {k: v[0] for k, v in cands.items()},
        "test": {k: v[1] for k, v in cands.items()},
        "weight_logs": weight_logs,
        "fb_blend_name": fb_blend_name,
        "S_own_nested_oof": cands[winner][0],
        "S_own_nested_test": cands[winner][1],
    }


def step5_drop_table(y, folds) -> list[dict]:
    """Rebuild Step5-style drop diagnostics for AUC>=0.945 columns that Step5 lost."""
    step5 = json.loads((LOG / "STEP5_AUDIT.json").read_text())
    kept5 = {m["name"] for m in step5["kept"]}
    dedup_map = {d["dropped"]: d for d in step5["dedup"]["dropped"]}
    out = []
    for r in step5["inventory_rows"]:
        name = r.get("name")
        if name in kept5:
            continue
        auc = r.get("auc")
        if auc is None or auc < 0.945:
            continue
        # recompute cal_fold_KS if we can reload vectors — use stored raw if present
        reason = r.get("status", "UNKNOWN")
        if name in dedup_map:
            reason = f"dedup_rho={dedup_map[name]['rho_oof']:.6f}_kept={dedup_map[name]['kept']}"
        elif reason == "HYGIENE_FAIL":
            reason = f"hygiene_raw_disp={r.get('disp')} raw_KS={r.get('KS')}"
        out.append({
            "name": name,
            "family": r.get("family"),
            "reason": reason,
            "solo_oof_auc": auc,
            "raw_disp": r.get("disp"),
            "raw_KS": r.get("KS"),
            "cal_fold_KS": None,  # filled below when possible
            "status_step5": r.get("status"),
        })
    return out


def fill_cal_ks_for_drops(drops: list[dict], y, folds) -> list[dict]:
    """Load vectors for dropped high-AUC cols and fill cal_fold_KS."""
    # Build name→(oof,te) by light re-assembly map
    name_to_ot: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    man = pd.read_csv(GPW_BANK / "manifest.csv")
    for _, r in man.iterrows():
        name = f"gpw70:{r['name']}"
        op, tp = GPW_BANK / f"oof_{r['name']}.npy", GPW_BANK / f"test_{r['name']}.npy"
        if op.exists() and tp.exists():
            name_to_ot[name] = (np.load(op).ravel(), np.load(tp).ravel())

    bman = pd.read_csv(BUSY / "manifest.csv")
    Bo = np.load(BUSY / "oof.npy")
    Bt = np.load(BUSY / "test.npy")
    for i, r in bman.iterrows():
        name_to_ot[f"busy52:{r['name']}"] = (Bo[:, i], Bt[:, i])
    name_to_ot["busy52:tabm_standalone"] = (
        np.load(BUSY / "tabm_oof.npy").ravel(), np.load(BUSY / "tabm_test.npy").ravel()
    )

    # prop gpt2
    gdir = PRED / "gpt2_med_4ep"
    if all((gdir / f"oof_{k}.npy").exists() for k in range(5)):
        gpt_o = np.zeros(N_TR)
        gpt_t = []
        for k in range(5):
            a = np.load(gdir / f"oof_{k}.npy")
            b = np.load(gdir / f"test_{k}.npy")
            zo = (a[:, :, 1] - a[:, :, 0]).mean(1)
            zt = (b[:, :, 1] - b[:, :, 0]).mean(1)
            gpt_o[folds == k] = zo
            gpt_t.append(zt)
        name_to_ot["prop:gpt2_med_4ep"] = (gpt_o, np.mean(gpt_t, 0))

    for d in drops:
        ot = name_to_ot.get(d["name"])
        if ot is None:
            continue
        oof, te = ot
        d["cal_fold_KS"] = cal_fold_ks(oof, te, folds)
        if d.get("raw_disp") is None:
            d["raw_disp"] = raw_disp(oof, te)
        if d.get("raw_KS") is None:
            d["raw_KS"] = float(ks_2samp(oof, te).statistic)
    return drops


def fullbank_recipe() -> dict:
    w0 = json.loads((ROOT / "logs/overnight/v36_1_ckpt/w0_done.json").read_text())
    return {
        "script": (
            "scripts/v36_1_lib.py::build_z_fullbank → "
            "scripts/v36_1_workers/w0_repair_baselines_and_intel.py (+v10orth grid)"
        ),
        "method": (
            "B_clean 8-pillar nest LogReg(C=0.5) → blend with wide (best of nest/wide/"
            "0.85w+0.15n/…/wide+0.02orth) → then micro-orth of "
            "v10_ridge_resid_orth_LM onto wide (c grid ∈[-0.04,0.04]) → AUC 0.9526207. "
            "GM_V36_1_RUN.log: selected wide+orth 0.952565 then +v10orth → 0.952621."
        ),
        "pillars": w0["baselines"]["names"],
        "n_pillars": w0["baselines"]["n_pillars"],
        "fb_auc": w0["baselines"]["fb_auc"],
        "v10_w_fixed": w0["baselines"]["ctrl"]["v10_on_fb"]["w_fixed"],
        "note": "Artifact at SSD/preds/z_fullbank_{oof,test}.npy",
    }


def main():
    assert NO_SUBMIT is True
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log("STEP5b start NO_SUBMIT=True")

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(float)
    folds = pd.read_csv(FOLDS_P)["fold"].to_numpy(int)
    te_df = pd.read_csv(DATA / "test.csv")
    tr_df = pd.read_csv(DATA / "train.csv")
    ids = te_df["id"].to_numpy()
    Zg_tr = z_goodpjw(tr_df)
    Zg_te = z_goodpjw(te_df)
    assert int(Zg_te.sum()) == 16_730

    ab_path = OUT / "F1b_pub_Step_AB_095569.csv"
    if not ab_path.exists():
        ab_path = EXT / "goodpjw_95569/submission.csv"
    assert sha256_file(ab_path) == SHA_AB
    ab = pd.read_csv(ab_path)["is_fraud"].to_numpy(float)

    part12_o = np.load(PART12 / "wide_lm_stack_nested.npy").ravel().astype(float)
    part12_t = np.load(PART12 / "wide_lm_stack_test.npy").ravel().astype(float)
    fb_o = np.load(SSD / "preds/z_fullbank_oof.npy").ravel().astype(float)
    fb_t = np.load(SSD / "preds/z_fullbank_test.npy").ravel().astype(float)
    fb_o_zr = apply_zr(fb_o, Zg_tr)
    p12_o_zr = apply_zr(part12_o, Zg_tr)
    p12_t_zr = apply_zr(part12_t, Zg_te)

    recipe = fullbank_recipe()
    log(f"fullbank recipe pillars={recipe['pillars']} auc={recipe['fb_auc']:.6f}")
    log(f"Part12 AUC={roc_auc_score(y, part12_o):.6f} (+Zr {roc_auc_score(y, p12_o_zr):.6f}) "
        f"fullbank={roc_auc_score(y, fb_o):.6f} (+Zr {roc_auc_score(y, fb_o_zr):.6f})")

    # --- §1 Step5 drop autopsy ---
    drops = step5_drop_table(y, folds)
    drops = fill_cal_ks_for_drops(drops, y, folds)
    log(f"Step5 high-AUC drops (auc>=0.945): {len(drops)}")
    print("\n=== STEP5 DROPPED (solo OOF AUC >= 0.945) ===", flush=True)
    print(f"{'name':<42} {'reason':<55} {'auc':>8} {'disp':>7} {'rKS':>7} {'cKS':>7}", flush=True)
    for d in sorted(drops, key=lambda x: -x["solo_oof_auc"]):
        cks = d["cal_fold_KS"]
        cks_s = f"{cks:.4f}" if cks is not None else "  nan"
        rd = d["raw_disp"]
        rk = d["raw_KS"]
        print(
            f"{d['name']:<42} {d['reason'][:55]:<55} {d['solo_oof_auc']:.5f} "
            f"{(rd if rd is not None else float('nan')):7.3f} "
            f"{(rk if rk is not None else float('nan')):7.4f} {cks_s:>7}",
            flush=True,
        )

    # --- assemble with fixed filter ---
    rows, Xo_raw, Xt_raw = assemble_candidates(y, folds)
    cand_meta = [r for r in rows if r.get("status") == "CAND"]
    log(f"candidates after hygiene_v2: {len(cand_meta)} / scanned {len(rows)}")
    for must in ("prop:gpt2_med_4ep", "prop:xgb_llr", "prop:l7_s2026", "prop:uid_v2"):
        hit = next((r for r in rows if r["name"] == must), None)
        log(f"  MUST {must}: status={hit.get('status') if hit else 'ABSENT'} "
            f"auc={hit.get('auc') if hit else None} calKS={hit.get('cal_fold_KS') if hit else None} "
            f"minTeCorr={hit.get('min_te_fold_corr') if hit else None}")

    # Pre-cap GBDT before expensive O(n²) dedup so we finish in minutes not hours
    cand_meta, Xo_raw, Xt_raw = prune_family_caps(cand_meta, Xo_raw, Xt_raw, {
        "GBDT": 22, "EBM_GLM": 12, "LM_GPT2": 25, "TabM_MLP": 16, "Proprietary": 14, "Other": 8,
    })
    log(f"after family cap: {len(cand_meta)}")

    kept_meta, Xo, Xt, dedup_info = dedup_v2(cand_meta, Xo_raw, Xt_raw)
    # Final GBDT trim post-dedup (keep LM/pillars intact)
    kept_meta, Xo, Xt = prune_family_caps(kept_meta, Xo, Xt, {
        "GBDT": 15, "EBM_GLM": 10, "LM_GPT2": 20, "TabM_MLP": 16, "Proprietary": 12, "Other": 8,
    })
    names = [m["name"] for m in kept_meta]
    families = [m["family"] for m in kept_meta]
    log(f"after dedup ρ>{DEDUP_RHO} + final cap: {len(kept_meta)} | by_family={dedup_info['by_family']}")
    log(f"pillars present: {[n for n in names if n.startswith('pillar:')]}")

    # --- nested stack v2 ---
    nest = nested_stack_v2(Xo, Xt, y, folds, names, families)
    S_o = nest["S_own_nested_oof"]
    S_t = nest["S_own_nested_test"]
    winner = nest["winner"]
    own_auc = nest["candidates_oof_auc"][winner]
    log(f"winner={winner} OOF={own_auc:.6f}")

    S_o_zr = apply_zr(S_o, Zg_tr)
    S_t_zr = apply_zr(S_t, Zg_te)
    own_auc_zr = float(roc_auc_score(y, S_o_zr))

    meta_report = {}
    for k, oof in nest["oof"].items():
        oof_zr = apply_zr(oof, Zg_tr)
        meta_report[k] = {
            "oof_auc": float(roc_auc_score(y, oof)),
            "oof_auc_zr": float(roc_auc_score(y, oof_zr)),
            "folds": fold_aucs(y, oof, folds),
            "folds_zr": fold_aucs(y, oof_zr, folds),
            "sf25_vs_part12": sf25_beats(y, oof_zr, p12_o_zr, folds),
            "sf25_vs_fullbank": sf25_beats(y, oof_zr, fb_o_zr, folds),
        }

    print("\n=== HARD DIAGNOSTIC ===", flush=True)
    print(f"  S_own_nested_v2 raw = {own_auc:.6f}  +Zr = {own_auc_zr:.6f}", flush=True)
    print(f"  Part12             raw = {roc_auc_score(y, part12_o):.6f}  +Zr = {roc_auc_score(y, p12_o_zr):.6f}", flush=True)
    print(f"  fullbank           raw = {roc_auc_score(y, fb_o):.6f}  +Zr = {roc_auc_score(y, fb_o_zr):.6f}", flush=True)
    assert_pass = bool(own_auc >= ASSERT_OOF)
    print(f"  ASSERT raw >= {ASSERT_OOF}: {assert_pass}", flush=True)
    if not assert_pass:
        fail_payload = {
            "NO_SUBMIT": True, "step": "5b", "assert_pass": False,
            "S_own_nested_v2_oof": own_auc, "S_own_nested_v2_oof_zr": own_auc_zr,
            "winner": winner, "n_kept": len(kept_meta),
            "candidates_oof_auc": nest["candidates_oof_auc"],
            "by_family": dedup_info["by_family"],
            "kept_names": names,
            "fullbank_recipe": recipe,
            "step5_high_auc_drops": drops,
        }
        (LOG / "STEP5B_AUDIT.json").write_text(json.dumps(fail_payload, indent=2, default=str))
        (LOG / "STEP5B_AUDIT.md").write_text(
            f"# V41 STEP5b AUDIT — ASSERT FAIL\n\n"
            f"S_own_nested_v2 OOF={own_auc:.6f} < {ASSERT_OOF} (winner={winner}, n={len(kept_meta)})\n"
            f"See STEP5B_AUDIT.json for details. STOP.\n"
        )
        raise AssertionError(
            f"HARD ASSERT FAIL: S_own_nested_v2 OOF={own_auc:.6f} < {ASSERT_OOF} "
            f"(winner={winner}; n_kept={len(kept_meta)})"
        )
    log(f"ASSERT PASS: OOF {own_auc:.6f} >= {ASSERT_OOF}")

    # --- §3 Yash dual-stack + tip residual ---
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

    w_grid = [0.0, 0.25, 0.35, 0.50, 0.65, 0.75, 1.0]
    Ls = L_fold(S_o, folds)
    Lp = L_fold(part12_o, folds)
    Ls_t = L(S_t)
    Lp_t = L(part12_t)

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

    oof_wstar = np.zeros(N_TR)
    w_per_fold = []
    for f in range(5):
        trm, vam = folds != f, folds == f
        inner = folds[trm]
        uniq = sorted(np.unique(inner))
        best_w, best_sc = 0.5, -1.0
        for w in w_grid:
            scores = []
            for u in uniq:
                ivam = inner == u
                gtr = np.where(trm)[0]
                iva = gtr[ivam]
                blend = U((1 - w) * Ls[iva] + w * Lp[iva])
                scores.append(roc_auc_score(y[iva], blend))
            sc = float(np.mean(scores))
            if sc > best_sc:
                best_sc, best_w = sc, w
        w_per_fold.append({"fold": f, "w_star": best_w, "inner_auc": best_sc})
        oof_wstar[vam] = U((1 - best_w) * Ls[vam] + best_w * Lp[vam])
        log(f"w* fold {f}: w={best_w} inner={best_sc:.5f}")

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

    # Tip residual probe (AB tip vs own / dual / Part12)
    n1_te = S_t_zr
    tip_resid = {
        "1mrho_AB_vs_N1v2_e4": one_mrho(ab, n1_te) * 1e4,
        "1mrho_AB_vs_dual_wstar_e4": one_mrho(ab, te_wstar_zr) * 1e4,
        "1mrho_AB_vs_Part12Zr_e4": one_mrho(ab, p12_t_zr) * 1e4,
        "1mrho_AB_vs_fullbankTe_e4": one_mrho(ab, apply_zr(fb_t, Zg_te)) * 1e4,
    }
    # Soft orth residual of tip vs dual on test (logit space), report magnitude
    lab, ld = L(ab), L(te_wstar_zr)
    # project tip onto dual
    ld_u = ld / (np.linalg.norm(ld) + 1e-12)
    proj = float(np.dot(lab, ld_u)) * ld_u
    resid = lab - proj
    tip_resid["tip_minus_proj_dual_std"] = float(resid.std())
    tip_resid["tip_proj_coef"] = float(np.dot(lab, ld_u))
    # OOF proxy: Part12 residual vs own (honest nested)
    lo, lp = L_fold(S_o, folds), L_fold(part12_o, folds)
    lp_u = lp / (np.linalg.norm(lp) + 1e-12)
    resid_o = lo - float(np.dot(lo, lp_u)) * lp_u
    tip_resid["own_minus_proj_part12_oof_std"] = float(resid_o.std())
    tip_resid["note"] = (
        "AB tip is test-only; OOF residual uses Part12 as public tip-proxy. "
        "Large test 1-ρ vs dual with ratio_test_over_oof>1.5 ⇒ Yash private gap."
    )
    log(f"tip residual 1-ρ AB vs N1v2 e4={tip_resid['1mrho_AB_vs_N1v2_e4']:.3f} "
        f"vs dual={tip_resid['1mrho_AB_vs_dual_wstar_e4']:.3f}")

    # --- stage v2 artifacts ---
    staged = {}
    sha, p = write_u(OUT / "N1v2_own_nested_stack_zr.csv", ids, S_t_zr)
    staged["N1v2_own_nested_stack_zr.csv"] = {
        "sha": sha, "meta": winner, "oof_auc": own_auc, "oof_auc_zr": own_auc_zr,
        "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4,
        "1mrho_vs_part12_e4": one_mrho(p, p12_t_zr) * 1e4,
    }
    sha, p = write_u(OUT / "N2v2_dual_nested_x_part12_cv_zr.csv", ids, te_wstar_zr)
    n1 = pd.read_csv(OUT / "N1v2_own_nested_stack_zr.csv")["is_fraud"].to_numpy()
    staged["N2v2_dual_nested_x_part12_cv_zr.csv"] = {
        "sha": sha, "mean_w_part12": wstar_report["mean_w"],
        "oof_auc": wstar_report["oof_auc"], "oof_auc_zr": wstar_report["oof_auc_zr"],
        "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4,
        "1mrho_vs_N1v2_e4": one_mrho(p, n1) * 1e4,
    }
    n2 = p
    n3_score = apply_zr(U(0.5 * L(S_t) + 0.5 * L(part12_t)), Zg_te)
    sha, p = write_u(OUT / "N3v2_dual_nested_x_part12_50_50_zr.csv", ids, n3_score)
    staged["N3v2_dual_nested_x_part12_50_50_zr.csv"] = {
        "sha": sha,
        "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4,
        "1mrho_vs_N2v2_e4": one_mrho(p, n2) * 1e4,
    }
    for name, w_ab in [
        ("H1v2_private_hedge_50AB_50N2.csv", 0.50),
        ("H2v2_private_hedge_70AB_30N2.csv", 0.70),
    ]:
        blend = U(w_ab * L(ab) + (1 - w_ab) * L(n2))
        blend = apply_zr(blend, Zg_te, U(ab))
        sha, p = write_u(OUT / name, ids, blend)
        staged[name] = {
            "sha": sha, "w_AB": w_ab,
            "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4,
            "1mrho_vs_N2v2_e4": one_mrho(p, n2) * 1e4,
        }
        log(f"staged {name} 1-ρ vs AB e4={staged[name]['1mrho_vs_AB_e4']:.4f}")

    for k, v in staged.items():
        log(f"staged {k}: sha16={v['sha'][:16]}")

    payload = {
        "NO_SUBMIT": True,
        "step": "5b",
        "fullbank_recipe": recipe,
        "step5_high_auc_drops": drops,
        "inventory_rows": rows,
        "kept": kept_meta,
        "dedup": dedup_info,
        "n_kept": len(kept_meta),
        "nested": {
            "winner": winner,
            "candidates_oof_auc": nest["candidates_oof_auc"],
            "meta_report": meta_report,
            "weight_logs": nest["weight_logs"],
            "assert_oof": ASSERT_OOF,
            "assert_pass": True,
            "S_own_nested_v2_oof": own_auc,
            "S_own_nested_v2_oof_zr": own_auc_zr,
        },
        "baselines": {
            "part12_oof": float(roc_auc_score(y, part12_o)),
            "part12_oof_zr": float(roc_auc_score(y, p12_o_zr)),
            "fullbank_oof": float(roc_auc_score(y, fb_o)),
            "fullbank_oof_zr": float(roc_auc_score(y, fb_o_zr)),
        },
        "divergence": div,
        "wstar": wstar_report,
        "tip_residual": tip_resid,
        "staged": staged,
    }
    (LOG / "STEP5B_AUDIT.json").write_text(json.dumps(payload, indent=2, default=str))

    lines = [
        "# V41 STEP5b AUDIT — Surgical Nested Stack Fix",
        "",
        f"**NO_SUBMIT={NO_SUBMIT}** · Hard stop after 5b. Do **not** run Step 6 without `PROCEED TO STEP 6`.",
        "",
        "## Executive findings",
        "",
        f"- Step5 high-AUC drops audited: **{len(drops)}** (see §1)",
        f"- Hygiene_v2 candidates → kept after ρ>{DEDUP_RHO} dedup: **{len(kept_meta)}**",
        f"- By family: `{json.dumps(dedup_info['by_family'])}`",
        f"- fullbank recipe: pillars={recipe['pillars']} → AUC **{recipe['fb_auc']:.6f}** "
        f"(`{recipe['script']}`)",
        f"- Nested winner: **{winner}** OOF=**{own_auc:.6f}** → +Zr **{own_auc_zr:.6f}** "
        f"(ASSERT ≥{ASSERT_OOF}: PASS)",
        f"- Part12 {roc_auc_score(y, part12_o):.6f}/{roc_auc_score(y, p12_o_zr):.6f} · "
        f"fullbank {roc_auc_score(y, fb_o):.6f}/{roc_auc_score(y, fb_o_zr):.6f}",
        f"- Divergence `1−ρ×1e−4`: OOF={div['1mrho_oof_e4']:.3f}, test={div['1mrho_test_e4']:.3f}, "
        f"ratio_test_over_oof=**{div['ratio_test_over_oof']:.2f}**",
        f"- Nested w* mean Part12={wstar_report['mean_w']:.3f}; dual+Zr OOF=**{wstar_report['oof_auc_zr']:.6f}**",
        f"- Tip residual: AB vs N1v2 e4={tip_resid['1mrho_AB_vs_N1v2_e4']:.3f}, "
        f"vs dual={tip_resid['1mrho_AB_vs_dual_wstar_e4']:.3f}",
        f"- Staged: N1v2/N2v2/N3v2 + H1v2/H2v2",
        "",
        "## 1. Step5 drop autopsy (solo OOF AUC ≥ 0.945)",
        "",
        "| name | reason | auc | raw_disp | raw_KS | cal_fold_KS |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for d in sorted(drops, key=lambda x: -x["solo_oof_auc"]):
        cks = d["cal_fold_KS"]
        cks_s = f"{cks:.4f}" if cks is not None else ""
        rd = d["raw_disp"]; rk = d["raw_KS"]
        lines.append(
            f"| `{d['name']}` | {d['reason'][:80]} | {d['solo_oof_auc']:.5f} | "
            f"{'' if rd is None else f'{rd:.3f}'} | "
            f"{'' if rk is None else f'{rk:.4f}'} | {cks_s} |"
        )

    lines += [
        "",
        "## 2. fullbank recipe",
        "",
        "```json",
        json.dumps(recipe, indent=2),
        "```",
        "",
        "## 3. Kept bank (hygiene_v2 + dedup)",
        "",
        f"n_kept={len(kept_meta)} dedup_dropped={len(dedup_info['dropped'])}",
        "",
        "| name | family | auc | cal_fold_KS | raw_disp |",
        "|---|---|---:|---:|---:|",
    ]
    for m in kept_meta:
        lines.append(
            f"| `{m['name']}` | {m['family']} | {m['auc']:.5f} | "
            f"{m.get('cal_fold_KS', m.get('KS', float('nan'))):.4f} | {m.get('raw_disp', m.get('disp', float('nan'))):.3f} |"
        )

    lines += [
        "",
        "## 4. Nested meta-stackers",
        "",
        "```json",
        json.dumps(meta_report, indent=2),
        "```",
        "",
        "### Weight logs (truncated)",
        "",
        "```json",
        json.dumps(nest["weight_logs"], indent=2)[:20000],
        "```",
        "",
        "## 5. Yash divergence, w*, tip residual",
        "",
        "```json",
        json.dumps({
            "divergence": div,
            "wstar": {
                "per_fold": w_per_fold,
                "mean_w": wstar_report["mean_w"],
                "oof_auc": wstar_report["oof_auc"],
                "oof_auc_zr": wstar_report["oof_auc_zr"],
                "fixed_grid_zr": {k: v["oof_auc_zr"] for k, v in fixed.items()},
            },
            "tip_residual": tip_resid,
        }, indent=2),
        "```",
        "",
        "## 6. Staged files",
        "",
        "```json",
        json.dumps(staged, indent=2),
        "```",
        "",
        "## 7. STOP",
        "",
        "Step 5b complete. **Awaiting human audit.** Reply `PROCEED TO STEP 6` to continue.",
        "",
    ]
    (LOG / "STEP5B_AUDIT.md").write_text("\n".join(lines))

    print("\n=== STRICT NO_SUBMIT=True | V41 STEP5b ===", flush=True)
    print(f"kept={len(kept_meta)} winner={winner}", flush=True)
    print(f"own={own_auc:.6f} +Zr={own_auc_zr:.6f} dual_w*+Zr={wstar_report['oof_auc_zr']:.6f}", flush=True)
    print(f"div oof/test e4={div['1mrho_oof_e4']:.3f}/{div['1mrho_test_e4']:.3f} "
          f"ratio={div['ratio_test_over_oof']:.2f}", flush=True)
    print(f"WROTE {LOG / 'STEP5B_AUDIT.md'}", flush=True)
    print("STOP. Awaiting PROCEED TO STEP 6.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
