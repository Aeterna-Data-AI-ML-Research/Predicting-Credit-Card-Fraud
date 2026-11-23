"""V36.1 shared geometry, dual baselines, frozen_lofo_orth, G0/vault gates, train stack."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs" / "overnight"
CKPT = LOGS / "v36_1_ckpt"
STATE_PATH = LOGS / "v36_1_state.json"
HEARTBEAT = LOGS / "GM_V36_1_HEARTBEAT.jsonl"
PRED = REPO / "preds" / "v36_1"
VAULT = REPO / "preds" / "v36_1_vault"
SUB = REPO / "submissions"
KAGGLE = os.path.expanduser("~/Library/Python/3.14/bin/kaggle")
COMP = "playground-series-s6e11"
ATH_CSV = SUB / "sub_v35_S4_v10gate_StepO_0.30e4_on_areeb.csv"
FOLDS_CSV = REPO / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv"
WIDE_OOF = REPO / "external/nov08_part12/wide_lm_stack_nested.npy"
WIDE_TE = REPO / "external/nov08_part12/wide_lm_stack_test.npy"
DEADLINE_UTC = datetime(2026, 11, 22, 6, 30, 0, tzinfo=timezone.utc)
SUBMIT_OPEN_UTC = datetime(2026, 11, 22, 0, 2, 0, tzinfo=timezone.utc)
MIDNIGHT_EVAL_UTC = datetime(2026, 11, 22, 0, 5, 0, tzinfo=timezone.utc)
COMPETENCE_FLOOR = 0.9515
N_THREADS = 4

# Quarantine / poisoned
QUARANTINE_SUBSTR = ("gb_soft_bag3", "soft_diff", "G26", "1fold", "fold0_only")


def ensure_dirs():
    for p in (LOGS, CKPT, PRED, VAULT, SUB):
        p.mkdir(parents=True, exist_ok=True)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def U(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / max(len(v), 1)


def L(v: np.ndarray) -> np.ndarray:
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6))


def Lf(v: np.ndarray, fold: np.ndarray) -> np.ndarray:
    out = np.zeros(len(v), float)
    for f in np.unique(fold):
        m = fold == f
        out[m] = L(v[m])
    return out


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def winsor(v: np.ndarray, k: float = 4.0) -> np.ndarray:
    return np.clip(unit(v), -k, k)


def auc(y, p) -> float:
    return float(roc_auc_score(y, np.asarray(p, float).ravel()))


def atomic_npy(path: Path, arr: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npy")
    # np.save adds .npy; use open binary
    tmp2 = path.parent / (path.name + ".tmp")
    with open(tmp2, "wb") as f:
        np.save(f, np.asarray(arr))
    os.replace(tmp2, path)


def atomic_json(path: Path, obj: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def load_state() -> dict:
    ensure_dirs()
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "run_id": now_utc().strftime("%Y%m%dT%H%M%SZ"),
        "phase": "T0",
        "engines": {},
        "vault": [],
        "g0": [],
        "submit_log": [],
        "ath_lb": 0.95548,
        "baselines": {},
    }


def save_state(st: dict):
    atomic_json(STATE_PATH, st)


def heartbeat(phase: str, engine: str = "", **extra):
    ensure_dirs()
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 1e9
        free = psutil.virtual_memory().available / 1e9
        cpu = psutil.cpu_percent(interval=0.0)
    except Exception:
        rss, free, cpu = -1, -1, -1
    row = {
        "ts": now_utc().isoformat(),
        "phase": phase,
        "engine": engine,
        "rss_gb": round(rss, 3),
        "free_gb": round(free, 3),
        "cpu_pct": cpu,
        **extra,
    }
    with HEARTBEAT.open("a") as f:
        f.write(json.dumps(row) + "\n")


def kaggle_cli(args: list[str], timeout: int = 90) -> str:
    try:
        p = subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ""
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return "\n".join(ln for ln in out.splitlines() if not ln.strip().startswith("Warning:"))


def status_is_complete(status: str) -> bool:
    return "COMPLETE" in str(status).upper()


def status_is_error(status: str) -> bool:
    return "ERROR" in str(status).upper()


def parse_submissions_table(text: str) -> list[dict]:
    rows = []
    for ln in text.splitlines():
        if "SubmissionStatus" not in ln and ".csv" not in ln:
            continue
        parts = re.split(r"\s{2,}", ln.strip())
        if len(parts) < 5:
            continue
        status = next((p for p in parts if "SubmissionStatus" in p or p in ("COMPLETE", "PENDING", "ERROR")), "")
        score = None
        for p in parts[::-1]:
            try:
                score = float(p)
                break
            except ValueError:
                continue
        fname = next((p for p in parts if p.endswith(".csv")), "")
        date = next((p for p in parts if re.match(r"\d{4}-\d{2}-\d{2}", p)), "")
        rows.append({"fileName": fname, "date": date, "status": status, "publicScore": score, "raw": ln})
    return rows


def used_today_utc(rows: list[dict], midnight: datetime) -> int:
    n = 0
    for r in rows:
        if status_is_error(r.get("status", "")):
            continue
        d = r.get("date", "")
        try:
            dt = datetime.fromisoformat(d.replace(" ", "T")).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if dt >= midnight:
            n += 1
    return n


def load_y_fold_ids():
    fold = pd.read_csv(FOLDS_CSV)["fold"].to_numpy().astype(np.int8)
    y = pd.read_csv(REPO / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(float)
    ids_te = pd.read_csv(REPO / "data/test.csv", usecols=["id"])["id"].to_numpy()
    ath = pd.read_csv(ATH_CSV).set_index("id").loc[ids_te, "is_fraud"].to_numpy(float)
    return y, fold, ids_te, ath


def build_subfolds(y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    sf = np.zeros(len(y), np.int8)
    for f in range(5):
        m = fold == f
        idx = np.where(m)[0]
        skf = StratifiedKFold(5, shuffle=True, random_state=42 + f)
        for j, (_, te) in enumerate(skf.split(idx, y[idx])):
            sf[idx[te]] = j
    return sf


def platt_eta(y: np.ndarray, z: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Per-fold 2-param Platt: η = a*z + b so E[σ(η)]≈E[y] on train folds."""
    eta = np.zeros(len(y), float)
    for f in range(5):
        tr, va = fold != f, fold == f
        # fit on train: logistic of z -> y
        zr = z[tr].reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        lr.fit(zr, y[tr])
        a = float(lr.coef_.ravel()[0])
        b = float(lr.intercept_.ravel()[0])
        eta[va] = a * z[va] + b
    # global refine mean match
    for _ in range(3):
        m = expit(eta).mean() - y.mean()
        eta = eta - logit(np.clip(m + 0.5, 1e-6, 1 - 1e-6)) + logit(0.5)  # crude
        # better: shift intercept
        eta = eta - (logit(np.clip(expit(eta).mean(), 1e-6, 1 - 1e-6)) - logit(np.clip(y.mean(), 1e-6, 1 - 1e-6)))
    return eta


def platt_eta_test(y: np.ndarray, z_oof: np.ndarray, z_te: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Bag test η using full-data Platt fit (a,b) averaged over folds."""
    As, Bs = [], []
    for f in range(5):
        tr = fold != f
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
        lr.fit(z_oof[tr].reshape(-1, 1), y[tr])
        As.append(float(lr.coef_.ravel()[0]))
        Bs.append(float(lr.intercept_.ravel()[0]))
    a, b = float(np.mean(As)), float(np.mean(Bs))
    return a * z_te + b


def _pillar_ok(oof: np.ndarray, te: np.ndarray, y: np.ndarray, fold: np.ndarray) -> tuple[bool, dict]:
    if not np.isfinite(oof).all() or not np.isfinite(te).all():
        return False, {"reason": "nan"}
    if oof.std() < 1e-12 or te.std() < 1e-12:
        return False, {"reason": "flat"}
    solo = auc(y, oof)
    if solo < 0.946:
        return False, {"reason": f"solo={solo:.5f}<0.946", "solo": solo}
    lo = Lf(oof, fold)
    lt = L(te)
    ratio = float(lt.std() / (lo.std() + 1e-12))
    if not (0.90 <= ratio <= 1.10):
        return False, {"reason": f"disp={ratio:.4f}", "solo": solo, "disp": ratio}
    return True, {"solo": solo, "disp": ratio}


def discover_bclean_candidates() -> list[tuple[str, Path, Path]]:
    """(name, oof_path, test_path) candidates — exclude quarantine."""
    pairs = []
    known = [
        ("wide_lm", WIDE_OOF, WIDE_TE),
        ("v10_ridge_resid", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"),
        ("l7_s2026", REPO / "preds/l7_s2026_5f/l7_s2026_oof_5fold_raw.npy", REPO / "preds/l7_s2026_5f/l7_s2026_test_5fold_raw.npy"),
        ("P4_L4.2", REPO / "preds/matrix_v2/P4_L4.2_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("P4_L4.3", REPO / "preds/matrix_v2/P4_L4.3_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.3_pure5f_test.npy"),
        ("P4_L4.5", REPO / "preds/matrix_v2/P4_L4.5_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.5_pure5f_test.npy"),
        ("harvest_p4", REPO / "preds/matrix_v2/harvest_p4_cuda_xgb_oof.npy", REPO / "preds/matrix_v2/harvest_p4_cuda_xgb_test.npy"),
        ("d1_v4", REPO / "preds/gm_v4_polish/d1_v4_oof.npy", REPO / "preds/gm_v4_polish/d1_v4_test.npy"),
        ("mlp_subodh", REPO / "external/nov10_recon/subodh/mlp_oof.npy", REPO / "external/nov10_recon/subodh/mlp_preds.npy"),
        ("lgb_subodh", REPO / "external/nov10_recon/subodh/lgb_oof.npy", REPO / "external/nov10_recon/subodh/lgb_preds.npy"),
        ("cat_subodh", REPO / "external/nov10_recon/subodh/cat_oof.npy", REPO / "external/nov10_recon/subodh/cat_preds.npy"),
        ("nov08_lm", REPO / "external/nov08_part12/lm_oof.npy", REPO / "external/nov08_part12/lm_test.npy"),
    ]
    # gpt2 med 5f assemble if fold npy exist
    gdir = REPO / "preds/gpt2_med_4ep"
    if (gdir / "oof_0.npy").exists() and (gdir / "test_0.npy").exists() if False else (gdir / "oof_0.npy").exists():
        pass
    # gpt2 rev
    for name, o, t in [
        ("gpt2_rev", REPO / "preds/gpt2_3ep/gpt2_rev_2ep_fixed_oof.npy", REPO / "preds/gpt2_3ep/gpt2_rev_2ep_fixed_test.npy"),
        ("gpt2_rev_raw", REPO / "preds/gpt2_3ep/gpt2_rev_2ep_oof.npy", REPO / "preds/gpt2_3ep/gpt2_rev_2ep_test.npy"),
    ]:
        known.append((name, o, t))
    # champ stacks (often strong)
    for tag in ["P_super_v5p5_clean", "P_super_v6_clean", "P_champ"]:
        o = REPO / "preds" / f"{tag}_oof.npy"
        t = REPO / "preds" / f"{tag}_test.npy"
        known.append((tag, o, t))

    for name, o, t in known:
        if any(q.lower() in name.lower() or q.lower() in str(o).lower() for q in QUARANTINE_SUBSTR):
            continue
        if o.exists() and t.exists():
            pairs.append((name, o, t))
    return pairs


def assemble_bclean(y, fold) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    cols_o, cols_t, names, meta = [], [], [], []
    n_tr, n_te = len(y), None
    for name, op, tp in discover_bclean_candidates():
        o = np.load(op).ravel().astype(float)
        t = np.load(tp).ravel().astype(float)
        if len(o) != n_tr:
            continue
        if n_te is None:
            n_te = len(t)
        if len(t) != n_te:
            continue
        ok, info = _pillar_ok(o, t, y, fold)
        info = {**info, "name": name, "ok": ok}
        meta.append(info)
        if ok:
            cols_o.append(o)
            cols_t.append(t)
            names.append(name)
    if not cols_o:
        raise RuntimeError("B_clean empty — no pillars passed hygiene")
    Bo = np.column_stack(cols_o)
    Bt = np.column_stack(cols_t)
    return Bo, Bt, names, meta


def nested_ridge_stack(y, fold, Bo: np.ndarray) -> np.ndarray:
    """Nested 5-fold LOFO Logistic on L(U(cols)) — probability OOF (beats raw ridge-on-y)."""
    n, k = Bo.shape
    X = np.column_stack([Lf(Bo[:, j], fold) for j in range(k)])
    oof = np.zeros(n, float)
    for f in range(5):
        tr, va = fold != f, fold == f
        lr = LogisticRegression(C=0.5, solver="lbfgs", max_iter=400, n_jobs=1)
        lr.fit(X[tr], y[tr])
        oof[va] = lr.predict_proba(X[va])[:, 1]
    return oof


def fit_ridge_test(y, fold, Bo, Bt) -> np.ndarray:
    Xo = np.column_stack([Lf(Bo[:, j], fold) for j in range(Bo.shape[1])])
    Xt = np.column_stack([L(Bt[:, j]) for j in range(Bt.shape[1])])
    lr = LogisticRegression(C=0.5, solver="lbfgs", max_iter=400, n_jobs=1)
    lr.fit(Xo, y)
    return lr.predict_proba(Xt)[:, 1]


def build_z_fullbank(y, fold, Bo, Bt, wide_oof, wide_te) -> tuple[np.ndarray, np.ndarray, float]:
    """Ensure z_fullbank OOF AUC >= max(wide, 0.95265) via logistic nest + wide-anchored blends."""
    nest = nested_ridge_stack(y, fold, Bo)
    nest_te = fit_ridge_test(y, fold, Bo, Bt)
    wide = np.asarray(wide_oof, float).ravel()[: len(y)]
    wide_t = np.asarray(wide_te, float).ravel()[: len(Bt)]
    cands = [
        ("nest", nest, nest_te),
        ("wide", wide, wide_t),
        ("0.85w+0.15n", U(0.85 * L(U(wide)) + 0.15 * L(U(nest))), U(0.85 * L(U(wide_t)) + 0.15 * L(U(nest_te)))),
        ("0.70w+0.30n", U(0.70 * L(U(wide)) + 0.30 * L(U(nest))), U(0.70 * L(U(wide_t)) + 0.30 * L(U(nest_te)))),
        ("rankmean", U(0.5 * U(wide) + 0.5 * U(nest)), U(0.5 * U(wide_t) + 0.5 * U(nest_te))),
    ]
    # also try mean of top-3 solo columns
    solos = [auc(y, Bo[:, j]) for j in range(Bo.shape[1])]
    top = np.argsort(solos)[::-1][:3]
    if len(top):
        cands.append(
            (
                "top3mean",
                U(np.mean([U(Bo[:, j]) for j in top], 0)),
                U(np.mean([U(Bt[:, j]) for j in top], 0)),
            )
        )
    best_name, best_o, best_t, best_a = "wide", wide, wide_t, auc(y, wide)
    for name, o, t in cands:
        a = auc(y, o)
        print(f"  z_fb cand {name}: {a:.6f}", flush=True)
        if a > best_a:
            best_name, best_o, best_t, best_a = name, o, t, a
    # micro LOFO boost: add tiny orth of nest vs wide if helpful
    from sklearn.model_selection import StratifiedKFold  # local; fold already given

    u = winsor(nest - np.dot(winsor(nest), unit(L(wide))) / (np.dot(unit(L(wide)), unit(L(wide))) + 1e-12) * unit(L(wide)))
    boost = U(L(U(wide)) + 0.02 * unit(u))
    a_boost = auc(y, boost)
    print(f"  z_fb cand wide+0.02orth_nest: {a_boost:.6f}", flush=True)
    if a_boost > best_a:
        u_te = winsor(
            nest_te
            - np.dot(winsor(nest_te), unit(L(wide_t)))
            / (np.dot(unit(L(wide_t)), unit(L(wide_t))) + 1e-12)
            * unit(L(wide_t))
        )
        best_name, best_o, best_t, best_a = "wide+orth", boost, U(L(U(wide_t)) + 0.02 * unit(u_te)), a_boost
    print(f"z_fullbank selected={best_name} AUC={best_a:.6f}", flush=True)
    return best_o.astype(float), best_t.astype(float), float(best_a)


def ath_twin_nnls(ath, Bo, Bt, fold) -> tuple[np.ndarray, np.ndarray, float]:
    """Non-negative constrained via clipped Ridge on L(U(ATH)) ~ B_test."""
    yt = L(U(ath))
    Xt = np.column_stack([unit(L(Bt[:, j])) for j in range(Bt.shape[1])])
    # Ridge then clip β>=0 and renormalize-ish
    ridge = Ridge(alpha=50.0, fit_intercept=False)
    ridge.fit(Xt, unit(yt))
    beta = np.clip(ridge.coef_.ravel(), 0, None)
    if beta.sum() < 1e-12:
        beta = np.ones_like(beta) / len(beta)
    else:
        beta = beta / (beta.sum() + 1e-12)
    z_te = Xt @ beta
    Xo = np.column_stack([unit(Lf(Bo[:, j], fold)) for j in range(Bo.shape[1])])
    z_oof = Xo @ beta
    r2 = 1.0 - np.sum((unit(yt) - z_te) ** 2) / (np.sum((unit(yt) - unit(yt).mean()) ** 2) + 1e-12)
    return z_oof, z_te, float(r2)


def svd_basis(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    # economy SVD
    try:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return Xc[:, : min(k, X.shape[1])]
    return Xc @ vt[:k].T


def r2_span(u: np.ndarray, B: np.ndarray) -> float:
    u = unit(u)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    X = np.column_stack([unit(B[:, j]) for j in range(B.shape[1])])
    ridge = Ridge(alpha=50.0, fit_intercept=False)
    ridge.fit(X, u)
    pred = ridge.predict(X)
    ss_res = np.sum((u - pred) ** 2)
    ss_tot = np.sum((u - u.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def _gs_strip(u: np.ndarray, cols: list[np.ndarray]) -> np.ndarray:
    """Gram-Schmidt / sequential projection of u onto 1-D columns (test-safe, no labels)."""
    out = unit(u)
    for c in cols:
        c = np.asarray(c, dtype=float)
        if c.ndim == 2:
            # expand matrices (e.g. SVD block) column-wise — never flatten n×k
            seq = [c[:, j] for j in range(c.shape[1])]
        else:
            seq = [c.ravel()]
        for col in seq:
            if len(col) != len(out):
                raise ValueError(f"_gs_strip length mismatch: u={len(out)} col={len(col)}")
            bb = unit(col)
            out = unit(out - np.dot(out, bb) / (np.dot(bb, bb) + 1e-12) * bb)
    return winsor(out)


def frozen_lofo_orth(
    cand_oof: np.ndarray,
    cand_te: np.ndarray,
    fold: np.ndarray,
    bases_oof: list[np.ndarray],
    bases_te: list[np.ndarray],
    B_clean_o: np.ndarray | None = None,
    B_clean_t: np.ndarray | None = None,
    alpha: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Out-of-fold Ridge orth against dual baselines (+ SVD B_clean); test GS lock."""
    n = len(cand_oof)
    u_o = np.zeros(n, float)

    def design(bases, B=None, k_svd=25):
        cols = [unit(b) for b in bases]
        if B is not None and B.shape[1] > 0:
            k = min(k_svd, B.shape[1])
            cols.append(svd_basis(np.column_stack([unit(B[:, j]) for j in range(B.shape[1])]), k))
        return np.column_stack(cols) if cols else np.zeros((len(bases[0]), 1))

    for f in range(5):
        tr, va = fold != f, fold == f
        Xtr = design([b[tr] for b in bases_oof], B_clean_o[tr] if B_clean_o is not None else None)
        Xva = design([b[va] for b in bases_oof], B_clean_o[va] if B_clean_o is not None else None)
        ytr = unit(cand_oof[tr])
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(Xtr, ytr)
        u_o[va] = unit(cand_oof[va]) - ridge.predict(Xva)
    u_o = winsor(u_o)
    # Test: Ridge fit on full OOF design (label-free on test) then Gram-Schmidt vs bases+SVD
    Xt = design(bases_te, B_clean_t)
    Xo_full = design(bases_oof, B_clean_o)
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(Xo_full, unit(cand_oof))
    u_t = unit(cand_te) - ridge.predict(Xt)
    gs_cols: list[np.ndarray] = [np.asarray(b, float).ravel() for b in bases_te]
    if B_clean_t is not None and B_clean_t.shape[1] > 0:
        k = min(25, B_clean_t.shape[1])
        S = svd_basis(
            np.column_stack([unit(B_clean_t[:, j]) for j in range(B_clean_t.shape[1])]), k
        )
        for j in range(S.shape[1]):
            gs_cols.append(S[:, j])
        # also lock against raw B pillars (1-D each)
        for j in range(B_clean_t.shape[1]):
            gs_cols.append(B_clean_t[:, j])
    u_t = _gs_strip(u_t, gs_cols)
    f2 = 0.0
    if B_clean_t is not None:
        for j in range(B_clean_t.shape[1]):
            f2 = max(f2, abs(float(np.corrcoef(u_t, unit(B_clean_t[:, j]))[0, 1])))
    for bt in bases_te:
        f2 = max(f2, abs(float(np.corrcoef(u_t, unit(bt))[0, 1])))
    disp = float(u_t.std() / (u_o.std() + 1e-12))
    meta = {"F2_max_corr": f2, "disp_orth": disp}
    return u_o, u_t, meta


def signed_lofo_grid(
    y: np.ndarray,
    fold: np.ndarray,
    sf: np.ndarray,
    base: np.ndarray,
    u_orth: np.ndarray,
    w_grid: np.ndarray | None = None,
) -> dict:
    """Optimize w on train folds [-f], eval on [f]; identical sign across folds via nested pick."""
    if w_grid is None:
        w_grid = np.linspace(-0.06, 0.06, 49)
    d_outer = []
    sub_ok = 0
    chosen_w = []
    for f in range(5):
        tr, va = fold != f, fold == f
        # pick w on train folds only
        best_w, best_d = 0.0, -1e18
        for w in w_grid:
            if abs(w) < 1e-12:
                continue
            pred_tr = U(L(base[tr]) + w * u_orth[tr])
            d = auc(y[tr], pred_tr) - auc(y[tr], base[tr])
            if d > best_d:
                best_d, best_w = d, float(w)
        chosen_w.append(best_w)
        pred = U(L(base[va]) + best_w * u_orth[va])
        d_outer.append((auc(y[va], pred) - auc(y[va], base[va])) * 1e4)
        for s in range(5):
            m = va & (sf == s)
            if m.sum() < 50:
                continue
            pd_s = U(L(base[m]) + best_w * u_orth[m])
            if (auc(y[m], pd_s) - auc(y[m], base[m])) > 0:
                sub_ok += 1
    d_outer = np.asarray(d_outer, float)
    se = d_outer.std(ddof=1) / np.sqrt(5) if d_outer.std(ddof=1) > 1e-12 else 1e-12
    # same-sign: majority of chosen_w
    signs = np.sign(chosen_w)
    majority = 1.0 if (signs >= 0).sum() >= (signs < 0).sum() else -1.0
    # re-eval with fixed majority sign * |w| mean
    w_fixed = majority * float(np.mean(np.abs(chosen_w)))
    d_fixed = []
    sub_fixed = 0
    for f in range(5):
        va = fold == f
        pred = U(L(base[va]) + w_fixed * u_orth[va])
        d_fixed.append((auc(y[va], pred) - auc(y[va], base[va])) * 1e4)
        for s in range(5):
            m = va & (sf == s)
            if m.sum() < 50:
                continue
            if (auc(y[m], U(L(base[m]) + w_fixed * u_orth[m])) - auc(y[m], base[m])) > 0:
                sub_fixed += 1
    d_fixed = np.asarray(d_fixed, float)
    se_f = d_fixed.std(ddof=1) / np.sqrt(5) if d_fixed.std(ddof=1) > 1e-12 else 1e-12
    return {
        "mean_e4": float(d_fixed.mean()),
        "pos": int((d_fixed > 0).sum()),
        "sub": int(sub_fixed),
        "t25": float(d_fixed.mean() / se_f),
        "w_fixed": w_fixed,
        "nested_w": chosen_w,
        "d": [round(float(x), 4) for x in d_fixed],
        "nested_mean_e4": float(d_outer.mean()),
    }


LEAKAGE_SOLO_CEILING = 0.9550


def evaluate_engine_g0(
    name: str,
    oof_raw: np.ndarray,
    te_raw: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    sf: np.ndarray,
    z_fb: np.ndarray,
    z_fb_te: np.ndarray,
    z_twin: np.ndarray,
    z_twin_te: np.ndarray,
    B_clean_o: np.ndarray,
    B_clean_t: np.ndarray,
    eta_fb: np.ndarray | None = None,
    eta_fb_te: np.ndarray | None = None,
    te_by_fold: np.ndarray | None = None,
) -> dict:
    """Full F1/F2/G0/Vault evaluation (hotpatched: leakage ceil, r2 on δ, bagged F1, disp_1f)."""
    solo_raw = auc(y, oof_raw)
    # Mode-B / reconstructed scores live in probability or logit space near fullbank
    if eta_fb is not None and solo_raw < COMPETENCE_FLOOR:
        recon = eta_fb + oof_raw
        solo = auc(y, recon)
        oof_eval = recon
        te_eval = (eta_fb_te + te_raw) if eta_fb_te is not None else te_raw
        mode_b = True
        delta_o = oof_raw  # pure increment
        delta_t = te_raw if eta_fb_te is None else te_raw  # residual tree output
        delta_mode = "raw_residual"
    else:
        solo = solo_raw
        oof_eval, te_eval = oof_raw, te_raw
        mode_b = bool(eta_fb is not None and solo_raw >= COMPETENCE_FLOOR)
        # Pure increment vs fullbank in rank-logit space (Mode-A absolute preds)
        delta_o = L(U(oof_eval)) - L(U(z_fb))
        delta_t = L(U(te_eval)) - L(U(z_fb_te))
        delta_mode = "logit_minus_zfb"
        if mode_b and eta_fb is not None:
            # absolute Mode-B recon: δ = recon - η
            delta_o = oof_eval - eta_fb
            delta_t = te_eval - (eta_fb_te if eta_fb_te is not None else z_fb_te)
            delta_mode = "recon_minus_eta"

    if solo > LEAKAGE_SOLO_CEILING:
        return {
            "name": name,
            "status": "LEAKAGE_DETECTED",
            "solo": solo,
            "solo_raw": solo_raw,
            "g0": False,
            "vault": False,
            "note": f"solo {solo:.5f} > {LEAKAGE_SOLO_CEILING} — fold/y leakage likely",
        }

    if solo < COMPETENCE_FLOOR:
        return {
            "name": name,
            "status": "COMPETENCE_GAP_FAIL",
            "solo": solo,
            "solo_raw": solo_raw,
            "g0": False,
            "vault": False,
        }

    # F1 on PURE INCREMENT δ (bagged-aware band + optional 1-fold-vs-1-fold)
    o_d = delta_o - delta_o.mean()
    t_d = delta_t - delta_t.mean()
    raw_disp = float(t_d.std() / (o_d.std() + 1e-12))
    ks = float(ks_2samp(unit(delta_o), unit(delta_t)).statistic)
    bagged_ok = (0.84 <= raw_disp <= 1.06) and (ks < 0.020)

    disp_1f = None
    disp_1f_ok = None
    if te_by_fold is not None and np.asarray(te_by_fold).ndim == 2 and te_by_fold.shape[0] == 5:
        ratios = []
        for f in range(5):
            va = fold == f
            te_f = np.asarray(te_by_fold[f], float)
            # stack always returns absolute recon; map into same δ space as delta_o
            if delta_mode in ("raw_residual", "recon_minus_eta") and eta_fb_te is not None:
                d_tf = te_f - eta_fb_te
            else:
                d_tf = L(U(te_f)) - L(U(z_fb_te))
            d_of = delta_o[va]
            ratios.append(float(d_tf.std() / (d_of.std() + 1e-12)))
        disp_1f = float(np.mean(ratios))
        disp_1f_ok = 0.93 <= disp_1f <= 1.07
        # F1: bagged band OR honest 1-fold dispersion (5-fold bag shrink is expected)
        f1 = ((0.84 <= raw_disp <= 1.06) or bool(disp_1f_ok)) and (ks < 0.020)
    else:
        f1 = bagged_ok

    # Novelty R² on δ_test vs SVD_40(B_152) — NOT on reconstructed η+δ
    B152 = svd_basis(
        np.column_stack([unit(B_clean_t[:, j]) for j in range(B_clean_t.shape[1])]),
        min(40, B_clean_t.shape[1]),
    )
    r2 = r2_span(delta_t, B152)

    # Orth on δ (not absolute recon) so Mode-B isn't double-counting η
    u_o, u_t, orth_meta = frozen_lofo_orth(
        delta_o, delta_t, fold, [z_fb, z_twin], [z_fb_te, z_twin_te], B_clean_o, B_clean_t
    )
    f2_ok = orth_meta["F2_max_corr"] < 0.010 and 0.84 <= orth_meta["disp_orth"] <= 1.06

    g_fb = signed_lofo_grid(y, fold, sf, z_fb, u_o)
    g_twin = signed_lofo_grid(y, fold, sf, z_twin, u_o)
    same_sign = np.sign(g_fb["w_fixed"]) == np.sign(g_twin["w_fixed"]) or abs(g_twin["w_fixed"]) < 1e-8

    g0 = (
        f1
        and f2_ok
        and r2 < 0.65
        and g_fb["pos"] == 5
        and g_fb["sub"] >= 19
        and g_fb["t25"] >= 3.0
        and g_fb["mean_e4"] >= 1.5
        and same_sign
        and g_twin["pos"] >= 4
        and g_twin["mean_e4"] > 0
        and g_twin["t25"] >= 1.8
    )
    vault = (
        (not g0)
        and f1
        and f2_ok
        and same_sign
        and g_fb["pos"] >= 4
        and g_fb["sub"] >= 17
        and g_fb["t25"] >= 2.2
        and g_fb["mean_e4"] >= 0.8
        and g_twin["pos"] >= 4
        and g_twin["mean_e4"] > 0
        and g_twin["t25"] >= 2.2
        and g_twin["mean_e4"] >= 0.8
    )
    rec = {
        "name": name,
        "status": "ok",
        "solo": solo,
        "solo_raw": solo_raw,
        "mode_b_style": mode_b,
        "raw_disp": raw_disp,
        "disp_1f": disp_1f,
        "disp_1f_ok": disp_1f_ok,
        "ks": ks,
        "f1": f1,
        "r2_B152": r2,
        "r2_on": "delta",
        "orth": orth_meta,
        "f2_ok": f2_ok,
        "g_fb": g_fb,
        "g_twin": g_twin,
        "same_sign": bool(same_sign),
        "g0": bool(g0),
        "vault": bool(vault),
        "hotpatch": "v36.1_g0_leak_r2delta_f1f2_disp1f",
    }
    atomic_npy(PRED / f"{name}_oof.npy", np.asarray(oof_eval, float))
    atomic_npy(PRED / f"{name}_test.npy", np.asarray(te_eval, float))
    atomic_npy(PRED / f"{name}_delta_oof.npy", np.asarray(delta_o, float))
    atomic_npy(PRED / f"{name}_delta_test.npy", np.asarray(delta_t, float))
    atomic_npy(PRED / f"{name}_u_oof.npy", u_o)
    atomic_npy(PRED / f"{name}_u_test.npy", u_t)
    atomic_json(PRED / f"{name}_gate.json", rec)
    atomic_json(LOGS / "v36_1_ckpt" / f"{name}_gate.json", rec)
    if vault or g0:
        atomic_npy(VAULT / f"{name}_u_oof.npy", u_o)
        atomic_npy(VAULT / f"{name}_u_test.npy", u_t)
        atomic_json(VAULT / f"{name}_gate.json", rec)
    return rec


def blend_ath(ath: np.ndarray, u: np.ndarray, c_w: float) -> np.ndarray:
    return U(L(U(ath)) + c_w * unit(u))


# -------------------- heavy multi-seed stack --------------------

def _has_catboost() -> bool:
    try:
        import catboost  # noqa: F401

        return True
    except Exception:
        return False


def train_multiseed_stack(
    X: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    Xte: np.ndarray,
    eta_oof: np.ndarray | None = None,
    eta_te: np.ndarray | None = None,
    seeds: list[int] | None = None,
    n_estimators: int = 2500,
    lr: float = 0.015,
    smoke: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """5-fold × multi-seed LGB+XGB(+Cat) Mode A and optional Mode B. Returns recon preds."""
    import lightgbm as lgb
    import xgboost as xgb

    seeds = seeds or ([7, 11, 17] if smoke else [7, 11, 17, 23, 29])
    if smoke:
        n_estimators = 80
        seeds = seeds[:2]
    n, nte = len(y), len(Xte)
    oof_sum = np.zeros(n, float)
    te_sum = np.zeros(nte, float)
    te_by_fold_sum = np.zeros((5, nte), float)  # 1-fold test preds for disp_1f
    n_models = 0
    families = ["lgb", "xgb"]
    if _has_catboost():
        families.append("cat")

    modes = [("A", None, None)]
    if eta_oof is not None and eta_te is not None:
        modes.append(("B", eta_oof, eta_te))

    t0 = time.time()
    for mode_name, eta_tr_full, eta_te_full in modes:
        for seed in seeds:
            for fam in families:
                oof_m = np.full(n, np.nan)
                te_acc = np.zeros(nte, float)
                folds_done = 0
                for f in range(5):
                    tr_m, va_m = fold != f, fold == f
                    # internal 85/15 early stop — NEVER on va_m
                    Xtr, ytr = X[tr_m], y[tr_m]
                    idx = np.arange(len(ytr))
                    i_tr, i_es = train_test_split(idx, test_size=0.15, random_state=seed + f, stratify=ytr)
                    te_f = None
                    if fam == "lgb":
                        dtr = lgb.Dataset(Xtr[i_tr], label=ytr[i_tr])
                        des = lgb.Dataset(Xtr[i_es], label=ytr[i_es], reference=dtr)
                        params = dict(
                            objective="binary",
                            learning_rate=lr,
                            num_leaves=31 if mode_name == "A" else 23,
                            min_data_in_leaf=50,
                            feature_fraction=0.5,
                            bagging_fraction=0.8,
                            bagging_freq=1,
                            verbosity=-1,
                            seed=seed + f,
                            num_threads=N_THREADS,
                        )
                        if mode_name == "B":
                            params["max_delta_step"] = 0.15
                            init_tr = eta_tr_full[tr_m][i_tr]
                            init_es = eta_tr_full[tr_m][i_es]
                            dtr.set_init_score(init_tr)
                            des.set_init_score(init_es)
                        booster = lgb.train(
                            params,
                            dtr,
                            num_boost_round=n_estimators,
                            valid_sets=[des],
                            callbacks=[lgb.early_stopping(80 if not smoke else 20, verbose=False)],
                        )
                        raw_va = booster.predict(X[va_m], raw_score=(mode_name == "B"))
                        raw_te = booster.predict(Xte, raw_score=(mode_name == "B"))
                        if mode_name == "B":
                            oof_m[va_m] = eta_tr_full[va_m] + raw_va
                            te_f = eta_te_full + raw_te
                            te_acc += te_f
                        else:
                            oof_m[va_m] = raw_va
                            te_f = raw_te
                            te_acc += raw_te
                    elif fam == "xgb":
                        dtr = xgb.DMatrix(Xtr[i_tr], label=ytr[i_tr])
                        des = xgb.DMatrix(Xtr[i_es], label=ytr[i_es])
                        params = dict(
                            objective="binary:logistic",
                            eta=lr,
                            max_depth=5 if mode_name == "A" else 4,
                            min_child_weight=50,
                            subsample=0.8,
                            colsample_bytree=0.5,
                            tree_method="hist",
                            seed=seed + f,
                            nthread=N_THREADS,
                        )
                        if mode_name == "B":
                            params["max_delta_step"] = 0.15
                            dtr.set_base_margin(eta_tr_full[tr_m][i_tr])
                            des.set_base_margin(eta_tr_full[tr_m][i_es])
                        booster = xgb.train(
                            params,
                            dtr,
                            num_boost_round=n_estimators,
                            evals=[(des, "es")],
                            early_stopping_rounds=80 if not smoke else 20,
                            verbose_eval=False,
                        )
                        dva = xgb.DMatrix(X[va_m])
                        dte = xgb.DMatrix(Xte)
                        if mode_name == "B":
                            dva.set_base_margin(eta_tr_full[va_m])
                            dte.set_base_margin(eta_te_full)
                            oof_m[va_m] = booster.predict(dva, output_margin=True)
                            te_f = booster.predict(dte, output_margin=True)
                            te_acc += te_f
                        else:
                            oof_m[va_m] = booster.predict(dva)
                            te_f = booster.predict(dte)
                            te_acc += te_f
                    else:  # catboost
                        from catboost import CatBoostClassifier, Pool

                        model = CatBoostClassifier(
                            iterations=n_estimators,
                            learning_rate=lr,
                            depth=6 if mode_name == "A" else 4,
                            min_data_in_leaf=50,
                            subsample=0.8,
                            rsm=0.5,
                            random_seed=seed + f,
                            thread_count=N_THREADS,
                            verbose=False,
                            early_stopping_rounds=80 if not smoke else 20,
                            grow_policy="SymmetricTree",
                        )
                        if mode_name == "B":
                            # Mode B requires baseline on train AND eval pools
                            tr_pool = Pool(
                                Xtr[i_tr],
                                ytr[i_tr],
                                baseline=eta_tr_full[tr_m][i_tr],
                            )
                            es_pool = Pool(
                                Xtr[i_es],
                                ytr[i_es],
                                baseline=eta_tr_full[tr_m][i_es],
                            )
                            model.fit(tr_pool, eval_set=es_pool, use_best_model=True)
                            va_pool = Pool(X[va_m], baseline=eta_tr_full[va_m])
                            te_pool = Pool(Xte, baseline=eta_te_full)
                            oof_m[va_m] = model.predict(va_pool, prediction_type="RawFormulaVal")
                            te_f = model.predict(te_pool, prediction_type="RawFormulaVal")
                            te_acc += te_f
                        else:
                            model.fit(
                                Xtr[i_tr],
                                ytr[i_tr],
                                eval_set=(Xtr[i_es], ytr[i_es]),
                                use_best_model=True,
                            )
                            oof_m[va_m] = model.predict_proba(X[va_m])[:, 1]
                            te_f = model.predict_proba(Xte)[:, 1]
                            te_acc += te_f
                    if te_f is not None:
                        te_by_fold_sum[f] += te_f
                    folds_done += 1
                fill = float(np.nanmean(oof_m))
                oof_m = np.where(np.isfinite(oof_m), oof_m, fill)
                te_m = te_acc / max(folds_done, 1)
                oof_sum += oof_m
                te_sum += te_m
                n_models += 1
                print(
                    f"  stack {mode_name}/{fam}/seed{seed} folds={folds_done} "
                    f"solo={auc(y, oof_m):.5f} models={n_models}",
                    flush=True,
                )
    oof = oof_sum / max(n_models, 1)
    te = te_sum / max(n_models, 1)
    te_by_fold = te_by_fold_sum / max(n_models, 1)
    meta = {
        "n_models": n_models,
        "families": families,
        "seeds": seeds,
        "solo": auc(y, oof),
        "elapsed": time.time() - t0,
        "modes": [m[0] for m in modes],
        "te_by_fold": te_by_fold,  # (5, n_te) — for G0 disp_1f
    }
    return oof, te, meta


def tabular_core_matrix(smoke: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Numeric + label-encoded cats from train/test."""
    from sklearn.preprocessing import LabelEncoder

    cols_num = [
        "account_balance_usd",
        "amount_usd",
        "merchant_risk_score",
        "velocity_score",
        "hours_since_last_txn",
        "customer_age",
        "card_age_months",
        "distance_from_home_km",
    ]
    cols_cat = ["auth_method", "channel", "device_type", "merchant_category", "card_type"]
    use = cols_num + cols_cat
    tr = pd.read_csv(REPO / "data/train.csv", usecols=use)
    te = pd.read_csv(REPO / "data/test.csv", usecols=use)
    if smoke:
        tr, te = tr.iloc[:12000].copy(), te.iloc[:6000].copy()
    for c in cols_cat:
        le = LabelEncoder()
        le.fit(pd.concat([tr[c], te[c]]).astype(str))
        tr[c] = le.transform(tr[c].astype(str))
        te[c] = le.transform(te[c].astype(str))
    # engineered domain ratios
    for df in (tr, te):
        df["amt_bal"] = df["amount_usd"] / (df["account_balance_usd"] + 1.0)
        df["age_card_mismatch"] = (df["customer_age"] * 12 - df["card_age_months"]).abs()
        df["cent"] = (np.round(df["amount_usd"] * 100) % 100).astype(float)
        df["log_amt"] = np.log1p(df["amount_usd"])
        df["log_bal"] = np.log1p(df["account_balance_usd"])
    names = list(tr.columns)
    return tr.to_numpy(float), te.to_numpy(float), names


def append_oof_summaries(Xtr, Xte, Bo, Bt, z_fb, z_fb_te, z_twin, z_twin_te) -> tuple[np.ndarray, np.ndarray]:
    extras_tr = [z_fb, z_twin] + [Bo[:, j] for j in range(min(Bo.shape[1], 12))]
    extras_te = [z_fb_te, z_twin_te] + [Bt[:, j] for j in range(min(Bt.shape[1], 12))]
    return np.column_stack([Xtr] + extras_tr), np.column_stack([Xte] + extras_te)
