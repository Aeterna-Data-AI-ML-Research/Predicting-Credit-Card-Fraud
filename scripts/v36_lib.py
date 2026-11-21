"""V36 shared helpers — geometry, gates, CLI parse, atomic IO, state."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs" / "overnight"
CKPT = LOGS / "v36_ckpt"
STATE_PATH = LOGS / "v36_state.json"
HEARTBEAT = LOGS / "GM_V36_HEARTBEAT.jsonl"
PRED_DIR = REPO / "preds" / "v36"
SUB = REPO / "submissions"
KAGGLE = os.path.expanduser("~/Library/Python/3.14/bin/kaggle")
COMP = "playground-series-s6e11"

ATH_CSV = SUB / "sub_v35_S4_v10gate_StepO_0.30e4_on_areeb.csv"
WIDE_OOF = REPO / "external/nov08_part12/wide_lm_stack_nested.npy"
WIDE_TE = REPO / "external/nov08_part12/wide_lm_stack_test.npy"
FOLDS_CSV = REPO / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv"
V10_OOF = REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy"
V10_TE = REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"


def ensure_dirs():
    for p in (LOGS, CKPT, PRED_DIR, SUB):
        p.mkdir(parents=True, exist_ok=True)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def next_utc_midnight(from_dt: datetime | None = None) -> datetime:
    d = from_dt or now_utc()
    floor = d.replace(hour=0, minute=0, second=0, microsecond=0)
    return floor + timedelta(days=1)


def U(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


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
    return float(roc_auc_score(y, p))


def omr(a, b) -> float:
    return float((1.0 - spearmanr(a, b).correlation) * 1e4)


def atomic_npy(path: Path, arr: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.asarray(arr))
    os.replace(tmp, path)


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
        "phase": "t0",
        "engines": {},
        "gate_results": {},
        "submit_log": [],
        "promote_h5": False,
        "NEXT_UTC_MIDNIGHT": None,
        "delta_pred_full": 0.0,
        "smoke": False,
    }


def save_state(st: dict):
    atomic_json(STATE_PATH, st)


def heartbeat(phase: str, engine: str = "", **extra):
    ensure_dirs()
    try:
        import psutil

        proc = psutil.Process()
        rss = proc.memory_info().rss / 1e9
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def status_is_complete(status: str) -> bool:
    return "COMPLETE" in str(status).upper()


def status_is_error(status: str) -> bool:
    return "ERROR" in str(status).upper()


def kaggle_cli(args: list[str], timeout: int = 90) -> str:
    """Run kaggle CLI; strip Warning: lines from stdout."""
    cmd = [KAGGLE, *args]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ""
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    lines = [ln for ln in out.splitlines() if not ln.strip().startswith("Warning:")]
    return "\n".join(lines)


def parse_submissions_table(text: str) -> list[dict]:
    """Best-effort parse of `kaggle competitions submissions` text table."""
    rows = []
    for ln in text.splitlines():
        if "SubmissionStatus" not in ln and ".csv" not in ln:
            continue
        # columns loosely: ref fileName date description status publicScore ...
        parts = re.split(r"\s{2,}", ln.strip())
        if len(parts) < 5:
            continue
        # find status token
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


def build_subfolds(y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    sf = np.zeros(len(y), np.int8)
    for f in range(5):
        m = fold == f
        idx = np.where(m)[0]
        skf = StratifiedKFold(5, shuffle=True, random_state=42 + f)
        for j, (_, te) in enumerate(skf.split(idx, y[idx])):
            sf[idx[te]] = j
    return sf


def _orth_winsor(cand: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Winsorize cand then strip linear projection onto L(base) (same recipe outer+sub)."""
    u = winsor(cand)
    bb = unit(L(base))
    return winsor(u - np.dot(u, bb) / (np.dot(bb, bb) + 1e-12) * bb)


def lofo_delta(
    y: np.ndarray,
    fold: np.ndarray,
    sf: np.ndarray,
    base: np.ndarray,
    cand: np.ndarray,
    w: float = 0.008,
) -> dict:
    """Evaluate U(L(base)+w*orth_winsor(cand)) vs base per outer fold + 25 subfolds."""
    d_outer = []
    sub_ok = 0
    sub_lifts = []
    for f in range(5):
        va = fold == f
        b, c = base[va], cand[va]
        u = _orth_winsor(c, b)
        pred = U(L(b) + w * u)
        d = (auc(y[va], pred) - auc(y[va], b)) * 1e4
        d_outer.append(d)
        for s in range(5):
            m = va & (sf == s)
            if m.sum() < 50:
                sub_lifts.append(0.0)
                continue
            # same orth recipe as outer (fixes inflated sub_ok from raw winsor)
            u_s = _orth_winsor(cand[m], base[m])
            pd_s = U(L(base[m]) + w * u_s)
            lift = (auc(y[m], pd_s) - auc(y[m], base[m])) * 1e4
            sub_lifts.append(lift)
            if lift > 0:
                sub_ok += 1
    d_outer = np.asarray(d_outer, float)
    sub_lifts = np.asarray(sub_lifts, float)
    se = d_outer.std(ddof=1) / np.sqrt(5) if d_outer.std(ddof=1) > 1e-12 else 1e-12
    t = float(d_outer.mean() / se)
    return {
        "mean_e4": float(d_outer.mean()),
        "pos": int((d_outer > 0).sum()),
        "sub": int(sub_ok),
        "t25": t,
        "sub_lifts_nz": int(np.count_nonzero(sub_lifts)),
        "d": [round(float(x), 4) for x in d_outer],
    }


def r2_span(u: np.ndarray, B: np.ndarray) -> float:
    """R² of u explained by columns of B (test span novelty)."""
    u = unit(u)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    X = np.column_stack([unit(B[:, j]) for j in range(B.shape[1])])
    # ridge
    ridge = Ridge(alpha=50.0, fit_intercept=False)
    ridge.fit(X, u)
    pred = ridge.predict(X)
    ss_res = np.sum((u - pred) ** 2)
    ss_tot = np.sum((u - u.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def blend_ath(ath: np.ndarray, u: np.ndarray, c_w: float) -> np.ndarray:
    """CAND = U(L(U(ATH)) + c*w*unit(u)) — never Zr."""
    return U(L(U(ath)) + c_w * unit(u))


def load_core(smoke: bool = False):
    """Load y, fold, wide, ath. Smoke uses first 8k train / 4k test."""
    fold = pd.read_csv(FOLDS_CSV)["fold"].to_numpy().astype(np.int8)
    y = pd.read_csv(REPO / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(float)
    ids_te = pd.read_csv(REPO / "data/test.csv", usecols=["id"])["id"].to_numpy()
    ath = pd.read_csv(ATH_CSV).set_index("id").loc[ids_te, "is_fraud"].to_numpy(float)
    wide = np.load(WIDE_OOF).ravel()[: len(y)].astype(float)
    wide_te = np.load(WIDE_TE).ravel().astype(float)
    if smoke:
        # stratified-ish: take contiguous blocks that cover all folds
        n = 8000
        y, fold, wide = y[:n], fold[:n], wide[:n]
        ids_te, ath, wide_te = ids_te[:4000], ath[:4000], wide_te[:4000]
    return {
        "y": y,
        "fold": fold,
        "wide": wide,
        "wide_te": wide_te,
        "ath": ath,
        "ids_te": ids_te,
        "z_wide": unit(Lf(wide, fold)),
        "z_wide_te": unit(L(wide_te)),
    }
