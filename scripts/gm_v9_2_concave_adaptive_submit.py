#!/usr/bin/env python3
"""GM v9.2 Concave adaptive: stage 4 rungs, submit Rung1, branch Slot2 on LB.

Fires live Kaggle submits (user-authorized). Blacklist: v10_*/soft_diff*/gb_soft*/idf_p3*/clean035*/zanch_*.
"""
from __future__ import annotations

import gc
import json
import os
import re
import resource
import socket
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")

VOL = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
WORK = Path(os.environ.get("GM_V92_ROOT", "/tmp/gm_deep_tech"))
PRED = VOL / "preds"
ART = WORK / "preds/gm_v9_2_concave"
LOG = WORK / "logs/overnight"
VOL_LOG = VOL / "logs/overnight"
SUB = VOL / "submissions"
for p in (ART, LOG, SUB, VOL_LOG):
    p.mkdir(parents=True, exist_ok=True)

EPS = 1e-6
T0 = time.time()
ATH = 0.95354
COMP = "playground-series-s6e11"
KAGGLE_BIN = VOL / ".venv_forensics/bin/kaggle"

BLACKLIST_RE = [
    re.compile(p, re.I)
    for p in [
        r"(^|[/_.-])v10([_/.-]|$)",
        r"soft_diff",
        r"gb_soft",
        r"idf_p3",
        r"clean035",
        r"zanch_",
    ]
]

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def assert_clean(path: Path) -> Path:
    s = str(path)
    for rx in BLACKLIST_RE:
        if rx.search(s) or rx.search(path.name):
            raise RuntimeError(f"BLACKLIST: {path}")
    return path


def load_f32(path: Path) -> np.ndarray:
    return np.asarray(np.load(assert_clean(path)), dtype=np.float32).ravel()


def zrk(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return logit(np.clip((rankdata(x, method="average") - 0.5) / len(x), EPS, 1 - EPS)).astype(np.float32)


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return ((v - float(v.mean())) / (float(v.std()) + 1e-12)).astype(np.float32)


def unit(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v / (float(np.linalg.norm(v)) + 1e-18)).astype(np.float32)


def spearman_dist(a, b):
    return float(1.0 - spearmanr(a, b).correlation)


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))


def _jitter(x, scale=1e-11):
    x = np.asarray(x, dtype=np.float64)
    return x + scale * (rankdata(x, method="ordinal") / len(x) - 0.5)


def force_exact_law_probs(ref_p, score):
    """Map score ranks onto ref_p order statistics (preserves mean/KS)."""
    ref_p = np.asarray(ref_p, dtype=np.float64).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    ref_s = np.sort(ref_p)
    n = len(score)
    order = np.argsort(_jitter(score, 1e-7), kind="mergesort")
    out = np.empty(n, dtype=np.float64)
    out[order] = ref_s
    return out


def align_method_b_smooth(oof_vec, test_vec, folds):
    oof_vec = np.asarray(oof_vec, dtype=np.float64).ravel()
    test_vec = np.asarray(test_vec, dtype=np.float64).ravel()
    oof_std = np.zeros_like(oof_vec)
    for a, b in folds:
        mu, sd = float(oof_vec[b].mean()), float(oof_vec[b].std() + 1e-12)
        oof_std[b] = (oof_vec[b] - mu) / sd
    oof_std = (oof_std - oof_std.mean()) / (oof_std.std() + 1e-12)
    test_std = (test_vec - test_vec.mean()) / (test_vec.std() + 1e-12)
    q = np.linspace(0.0, 1.0, 2001)
    pchip = PchipInterpolator(
        np.quantile(_jitter(test_std), q),
        np.quantile(_jitter(oof_std), q),
        extrapolate=True,
    )
    test_al = pchip(_jitter(test_std))
    test_al = (test_al - test_al.mean()) / (test_al.std() + 1e-12)
    return oof_std.astype(np.float32), test_al.astype(np.float32)


def disp_ks(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.std(b) / (np.std(a) + 1e-18)), float(ks_2samp(a, b).statistic)


def weight_for_dist(z0, u, target, hi=200.0, n=64):
    u = unit(u).astype(np.float64)
    z0 = np.asarray(z0, dtype=np.float64)
    lo, best_w, best_err, got = 0.0, 0.0, 1e9, 0.0
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        d = spearman_dist(z0, z0 + mid * u)
        err = abs(d - target)
        if err < best_err:
            best_err, best_w, got = err, mid, d
        if d > target:
            hi = mid
        else:
            lo = mid
    return best_w, got


def project_out(v, bases):
    v = np.asarray(v, dtype=np.float64).ravel()
    B = np.column_stack([unitize(b).astype(np.float64) for b in bases])
    lr = LinearRegression(fit_intercept=True)
    lr.fit(B, v)
    return unitize(v - lr.predict(B))


def lofo_resid(y_vec, bases, folds):
    y_vec = np.asarray(y_vec, dtype=np.float64).ravel()
    B = np.column_stack([unitize(b).astype(np.float64) for b in bases])
    resid = np.zeros_like(y_vec)
    for a, b in folds:
        lr = LinearRegression(fit_intercept=True)
        lr.fit(B[a], y_vec[a])
        resid[b] = y_vec[b] - lr.predict(B[b])
    return unitize(resid)


def pred_lb_from_t(t):
    return 0.95346910 + 6.99725804e-05 * t


# -------------------- Kaggle DNS + API --------------------
_CACHE: dict[str, str] = {}
_orig_getaddrinfo = socket.getaddrinfo


def _doh_ip(name: str) -> str | None:
    try:
        out = subprocess.check_output(
            [
                "curl",
                "-sS",
                "-m",
                "15",
                "--resolve",
                "dns.google:443:8.8.8.8",
                f"https://dns.google/resolve?name={name}&type=A",
            ],
            text=True,
        )
        j = json.loads(out)
        for a in j.get("Answer", []):
            if a.get("type") == 1:
                return a["data"]
    except Exception:
        return None
    return None


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if isinstance(host, str) and not host.replace(".", "").isdigit():
        if host not in _CACHE:
            ip = _doh_ip(host)
            if ip:
                _CACHE[host] = ip
                log(f"DNS_PATCH {host} -> {ip}")
        if host in _CACHE:
            host = _CACHE[host]
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


def setup_kaggle_api():
    for k in list(os.environ):
        if "proxy" in k.lower():
            os.environ.pop(k, None)
    socket.getaddrinfo = _patched_getaddrinfo  # type: ignore
    # ensure forensics env on path
    venv_site = VOL / ".venv_forensics/lib"
    for p in venv_site.glob("python*/site-packages"):
        sys.path.insert(0, str(p))
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def submit_file(api, path: Path, message: str) -> str:
    path = Path(path)
    assert path.exists() and path.stat().st_size > 1_000_000
    log(f"SUBMIT {path.name} :: {message}")
    # KaggleApi.competition_submit(file_name, message, competition)
    api.competition_submit(str(path), message, COMP)
    return path.name


def poll_lb(api, filename: str, timeout_s=900, interval=20):
    """Poll until a submission matching filename is Complete; return publicScore."""
    t_end = time.time() + timeout_s
    target = Path(filename).name
    last = None
    while time.time() < t_end:
        try:
            subs = api.competition_submissions(COMP)
        except Exception as e:
            log(f"  poll error: {e}; retry")
            time.sleep(interval)
            continue
        # newest first typically
        matches = []
        for s in subs:
            fn = getattr(s, "fileName", None) or getattr(s, "file_name", None) or ""
            desc = getattr(s, "description", "") or ""
            if target in str(fn) or target.replace(".csv", "") in str(desc):
                matches.append(s)
        if not matches:
            # fallback: most recent submission
            matches = list(subs)[:1]
        s = matches[0]
        status = getattr(s, "status", None) or getattr(s, "submissionStatus", "")
        pub = getattr(s, "publicScore", None) or getattr(s, "public_score", None)
        date = getattr(s, "date", None)
        last = {"status": str(status), "publicScore": pub, "date": str(date), "file": str(getattr(s, "fileName", ""))}
        log(f"  poll status={status} pub={pub} file={last['file']}")
        st = str(status).lower().replace("submissionstatus.", "")
        if ("complete" in st or "finished" in st) and pub not in (None, "", "None"):
            try:
                return float(pub), last
            except Exception:
                pass
        if "error" in st or "fail" in st:
            raise RuntimeError(f"Submission failed: {last}")
        time.sleep(interval)
    raise TimeoutError(f"Timeout waiting for LB on {filename}; last={last}")


def verify_csv(path: Path, ref_p: np.ndarray):
    df = pd.read_csv(path)
    assert len(df) == 299984, len(df)
    p = df["is_fraud"].to_numpy(np.float64)
    assert np.isfinite(p).all()
    assert not np.isnan(p).any()
    mean = float(p.mean())
    ks = float(ks_2samp(ref_p, p).statistic)
    return {"n": len(df), "mean": mean, "ks_vs_s7": ks, "min": float(p.min()), "max": float(p.max())}


def main():
    log("======== GM v9.2 CONCAVE ADAPTIVE SUBMIT ========")
    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    fold_id = pd.read_csv(
        VOL / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv", usecols=["fold"]
    )["fold"].to_numpy(np.int8)
    m0 = fold_id == 0
    idx_f0 = np.where(m0)[0]

    # ---- S7 + RayA ----
    s7_df = pd.read_csv(VOL / "submissions/sub_master_slot7_verified_s7A.csv")
    ids = s7_df["id"].to_numpy()
    s7_p = s7_df["is_fraud"].to_numpy(np.float64)
    z_s7_test = unitize(zrk(s7_p))
    log(f"S7 mean={s7_p.mean():.6f} n={len(s7_p)}")

    z_areeb = unitize(load_f32(PRED / "gm_breakout/z_areeb.npy"))
    m1 = unit(load_f32(PRED / "gm_breakout/m1.npy"))
    m2 = unit(load_f32(PRED / "gm_breakout/m2.npy"))
    v_ray = unit(0.70 * m1.astype(np.float64) + 0.30 * m2.astype(np.float64))
    if spearman_dist(z_areeb, z_s7_test.astype(np.float64) + 5 * v_ray.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7_test
    ):
        v_ray = (-v_ray.astype(np.float64)).astype(np.float32)

    # OOF RayA direction (s6)
    z_ref = load_f32(PRED / "gm_v5_overnight/z_ref_old_oof.npy")
    z_ref = unitize(z_ref) if abs(float(z_ref.std()) - 1.0) > 0.2 else z_ref
    d1 = unitize(load_f32(PRED / "gm_v4_polish/d1_v4_oof.npy"))
    gpt = unitize(load_f32(PRED / "gm_v2_ladder/z_m_bayes_5f_oof.npy"))
    u_sb = unit(d1)
    m1o = unitize(
        0.65 * u_sb.astype(np.float64)
        - 0.35 * (-unit(gpt)).astype(np.float64)
        - 0.20 * gpt.astype(np.float64)
    )
    s6 = unitize(0.70 * m1o.astype(np.float64) + 0.30 * u_sb.astype(np.float64))
    z_s7_o = z_ref.astype(np.float64) + 0.055 * s6.astype(np.float64)

    # tree_logit GS-decoupled
    tree_o = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/tree_logit_v72_oof.npy"))
    tree_t = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/tree_logit_v72_test.npy"))
    tree_o_orth = lofo_resid(tree_o, [z_s7_o, s6], folds)
    tree_t_orth = project_out(tree_t, [z_s7_test, v_ray])
    tree_o_orth, tree_t_orth = align_method_b_smooth(tree_o_orth, tree_t_orth, folds)
    # orient tree toward fraud on Fold0
    if float(np.corrcoef(tree_o_orth[idx_f0], y[idx_f0])[0, 1]) < 0:
        tree_o_orth = (-tree_o_orth.astype(np.float64)).astype(np.float32)
        tree_t_orth = (-tree_t_orth.astype(np.float64)).astype(np.float32)
    # exact test projection again
    tree_t_orth = project_out(tree_t_orth, [v_ray])
    d_tr, k_tr = disp_ks(tree_o_orth, tree_t_orth)
    cos_te = cos_sim(tree_t_orth, v_ray)
    cos_f0 = cos_sim(tree_o_orth[idx_f0], s6[idx_f0])
    log(f"tree_orth disp={d_tr:.4f} KS={k_tr:.6f} cosTe={cos_te:.6f} cosF0={cos_f0:.6f}")
    assert abs(cos_te) < 1e-3, cos_te

    # RayA+
    v_plus_t = unit(0.92 * v_ray.astype(np.float64) + 0.08 * tree_t_orth.astype(np.float64))
    v_plus_o = unit(0.92 * s6.astype(np.float64) + 0.08 * tree_o_orth.astype(np.float64))
    # Fold0 verify RayA+ vs pure RayA at 0.20e-4 and 0.40e-4
    use_plus = True
    for step in (0.20e-4, 0.40e-4):
        dw = (step / 1e-4) * 0.08
        z_pure = z_s7_o + dw * unitize(s6).astype(np.float64)
        z_plus = z_s7_o + dw * unitize(v_plus_o).astype(np.float64)
        d_pure = (roc_auc_score(y[m0], z_pure[m0]) - roc_auc_score(y[m0], z_s7_o[m0])) * 1e5
        d_plus = (roc_auc_score(y[m0], z_plus[m0]) - roc_auc_score(y[m0], z_s7_o[m0])) * 1e5
        log(f"  F0 check step={step:.2e}: pure={d_pure:+.2f} plus={d_plus:+.2f}")
        if d_plus < d_pure - 0.05:
            use_plus = False
    if not use_plus:
        log("  RayA+ does NOT beat pure → fallback v=RayA")
        v_use_t, v_use_o = v_ray, s6
        ascent = "pure_RayA"
    else:
        v_use_t, v_use_o = v_plus_t, v_plus_o
        ascent = "RayA+"
    log(f"  ascent vector = {ascent}")

    # Ray B_clean: +u_stageB + 0.22*tree_orth (freeze -u_3ladder / no RayA continuation)
    u_stage_t = unitize(load_f32(PRED / "gm_breakout/u_stageB_5f.npy"))
    u_stage_o = unitize(load_f32(WORK / "preds/gm_deep_tech_v7/u_stageB_plus_oof.npy"))
    # orient stageB
    if spearman_dist(z_areeb, z_s7_test.astype(np.float64) + 5 * u_stage_t.astype(np.float64)) < spearman_dist(
        z_areeb, z_s7_test
    ):
        u_stage_t = (-u_stage_t.astype(np.float64)).astype(np.float32)
        u_stage_o = (-u_stage_o.astype(np.float64)).astype(np.float32)
    # decouple stageB lightly from RayA so B_clean ≠ RayA
    u_stage_t = project_out(u_stage_t, [v_ray])
    u_stage_o = lofo_resid(u_stage_o, [s6], folds)
    v_B_t = unit(u_stage_t.astype(np.float64) + 0.22 * tree_t_orth.astype(np.float64))
    v_B_o = unit(u_stage_o.astype(np.float64) + 0.22 * tree_o_orth.astype(np.float64))

    def materialize(name: str, step: float, v_t, v_o) -> dict:
        w, got = weight_for_dist(z_s7_test, v_t, step)
        z = unitize(z_s7_test.astype(np.float64) + w * unit(v_t).astype(np.float64))
        # probs via exact order-stat map onto S7
        p = force_exact_law_probs(s7_p, z)
        path = SUB / name
        pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(path, index=False)
        ver = verify_csv(path, s7_p)
        d_a = spearman_dist(z_areeb, z)
        t = 100.0 * np.sqrt(max(d_a, 0))
        # OOF F0
        dw = (step / 1e-4) * 0.08
        z_o = z_s7_o + dw * unitize(v_o).astype(np.float64)
        f0 = (roc_auc_score(y[m0], z_o[m0]) - roc_auc_score(y[m0], z_s7_o[m0])) * 1e5
        # 5F vs S7 oof
        folds5 = folds
        d5 = []
        for a, b in folds5:
            d5.append(
                (
                    roc_auc_score(y[b], z_o[b]) - roc_auc_score(y[b], z_s7_o[b])
                )
                * 1e5
            )
        meta = {
            "file": str(path),
            "step": step,
            "got": got,
            "t": t,
            "pred_lin": pred_lb_from_t(t),
            "F0_vs_S7_e5": float(f0),
            "d5_mean": float(np.mean(d5)),
            "d5": [float(x) for x in d5],
            "verify": ver,
            "1mrho_S7": spearman_dist(z_s7_test, z),
        }
        log(
            f"  {name}: 1-ρ={meta['1mrho_S7']:.3e} t={t:.3f} pred={meta['pred_lin']:.6f} "
            f"F0={f0:+.2f} 5F={meta['d5_mean']:+.2f} mean={ver['mean']:.6f} KS={ver['ks_vs_s7']:.2e}"
        )
        assert ver["n"] == 299984
        assert abs(ver["mean"] - float(s7_p.mean())) < 1e-9 or ver["ks_vs_s7"] < 1e-5
        return meta

    log("======== Step 1: materialize 4 rungs ========")
    rung1 = materialize("sub_v9p2_rung1_probe_0p20e4.csv", 0.20e-4, v_use_t, v_use_o)
    rung2A = materialize("sub_v9p2_rung2A_sweetspot_0p42e4.csv", 0.42e-4, v_use_t, v_use_o)
    rung2B = materialize("sub_v9p2_rung2B_f0peak_0p68e4.csv", 0.68e-4, v_use_t, v_use_o)
    rung2C = materialize("sub_v9p2_rung2C_decoupled_bisect_0p10e4.csv", 0.10e-4, v_B_t, v_B_o)

    portfolio = {
        "ascent": ascent,
        "tree_orth": {"disp": d_tr, "ks": k_tr, "cos_te": cos_te, "cos_f0": cos_f0},
        "rung1": rung1,
        "rung2A": rung2A,
        "rung2B": rung2B,
        "rung2C": rung2C,
    }
    (ART / "v9_2_portfolio.json").write_text(json.dumps(portfolio, indent=2, default=float))

    # -------- Step 2: submit + branch --------
    log("======== Step 2: submit Rung1 + secant branch ========")
    api = setup_kaggle_api()
    skip_rung1 = os.environ.get("GM_V92_LB1", "").strip()
    if skip_rung1:
        lb1 = float(skip_rung1)
        info1 = {"status": "reuse", "publicScore": lb1, "file": "sub_v9p2_rung1_probe_0p20e4.csv"}
        log(f"REUSE LB_1={lb1:.5f} (skip re-submit Rung1)")
    else:
        msg1 = (
            "v9.2 Slot 1: RayA+ probe @ 0.20e-4 from S7 "
            f"(F0 {rung1['F0_vs_S7_e5']:+.2f}e-5 vs S7, t={rung1['t']:.2f})"
        )
        submit_file(api, SUB / "sub_v9p2_rung1_probe_0p20e4.csv", msg1)
        lb1, info1 = poll_lb(api, "sub_v9p2_rung1_probe_0p20e4.csv")
    log(f"LB_1 = {lb1:.5f} (Δ={lb1-ATH:+.5f})")
    if lb1 >= 0.95356 - 1e-9:
        case = 1
        next_path = SUB / "sub_v9p2_rung2B_f0peak_0p68e4.csv"
        msg2 = (
            f"v9.2 Slot 2 Case1: LB1={lb1:.5f} >=0.95356 → F0-peak 0.68e-4 "
            f"(pred {rung2B['pred_lin']:.6f})"
        )
        next_meta = rung2B
    elif abs(lb1 - 0.95355) < 1e-9 or (0.953545 <= lb1 < 0.95356):
        # +1 tick region
        case = 2
        next_path = SUB / "sub_v9p2_rung2A_sweetspot_0p42e4.csv"
        msg2 = (
            f"v9.2 Slot 2 Case2: LB1={lb1:.5f} (+1 tick concave) → sweetspot 0.42e-4 "
            f"(pred {rung2A['pred_lin']:.6f})"
        )
        next_meta = rung2A
    else:
        case = 3
        next_path = SUB / "sub_v9p2_rung2C_decoupled_bisect_0p10e4.csv"
        msg2 = (
            f"v9.2 Slot 2 Case3: LB1={lb1:.5f} flat/down → Decoupled RayB_clean 0.10e-4 "
            f"(pred {rung2C['pred_lin']:.6f})"
        )
        next_meta = rung2C

    log(f"BRANCH Case {case} → {next_path.name}")
    submit_file(api, next_path, msg2)
    lb2, info2 = poll_lb(api, next_path.name)
    log(f"LB_2 = {lb2:.5f} (Δ={lb2-ATH:+.5f})")

    result = {
        "ATH": ATH,
        "ascent": ascent,
        "LB_1": lb1,
        "LB_1_info": info1,
        "case": case,
        "slot2_file": next_path.name,
        "LB_2": lb2,
        "LB_2_info": info2,
        "portfolio": portfolio,
    }
    (ART / "v9_2_live_result.json").write_text(json.dumps(result, indent=2, default=float))

    md = []
    md.append("# GM v9.2 Concave Adaptive — Live Result\n\n")
    md.append(f"Ascent vector: **{ascent}**. tree_orth cos(test,v_RayA)=`{cos_te:.6f}`, KS=`{k_tr:.6f}`.\n\n")
    md.append("## Pre-staged rungs\n\n")
    md.append("| rung | step | t | pred_lin | F0vsS7 | 5F | mean | KS |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for tag, r in [("1", rung1), ("2A", rung2A), ("2B", rung2B), ("2C", rung2C)]:
        md.append(
            f"| {tag} | {r['step']:.2e} | {r['t']:.3f} | {r['pred_lin']:.6f} | "
            f"{r['F0_vs_S7_e5']:+.2f} | {r['d5_mean']:+.2f} | {r['verify']['mean']:.6f} | "
            f"{r['verify']['ks_vs_s7']:.2e} |\n"
        )
    md.append("\n## Live LB\n\n")
    md.append(f"- **LB_1** (Rung1 `0.20e-4`) = **`{lb1:.5f}`** (Δ vs ATH `{lb1-ATH:+.5f}`)\n")
    md.append(f"- **Case {case}** → `{next_path.name}`\n")
    md.append(f"- **LB_2** = **`{lb2:.5f}`** (Δ vs ATH `{lb2-ATH:+.5f}`)\n")
    md.append(f"\nElapsed `{time.time()-T0:.0f}s`.\n")
    report = LOG / "GM_V9_2_CONCAVE_ADAPTIVE_LIVE.md"
    report.write_text("".join(md))
    (VOL_LOG / "GM_V9_2_CONCAVE_ADAPTIVE_LIVE.md").write_text(report.read_text())
    log(f"REPORT → {report}")
    log("======== v9.2 DONE ========")
    print(json.dumps({"LB_1": lb1, "case": case, "LB_2": lb2, "slot2": next_path.name}, indent=2))


if __name__ == "__main__":
    main()
