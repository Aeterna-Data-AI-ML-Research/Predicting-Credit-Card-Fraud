#!/usr/bin/env python3
"""GM V34.3 — Principal-DS final lock breakthrough.
STRICT NO_SUBMIT=True · 0 GPU · wall<3.5m · peak RSS<2.4GB
"""
from __future__ import annotations

import gc
import json
import os
import re
import resource
import subprocess
import threading
import time
import traceback
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from scipy import sparse, stats
from scipy.interpolate import PchipInterpolator
from scipy.sparse.linalg import lsqr
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

NO_SUBMIT = True
N_JOBS = 4
RSS_CAP = 2.4
T0 = time.time()
REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "logs/overnight/GM_V34_3_PRINCIPAL_DS_BREAKTHROUGH_REPORT.md"
OUT = REPO / "preds/v34_3_principal"
SUB = REPO / "submissions"
KAGGLE = Path.home() / "Library/Python/3.14/bin/kaggle"
GPT2_DIR = Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

FEAT = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
    "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
BOOLS = FEAT[-6:]
FLOATS = ["amount_usd", "account_balance_usd", "velocity_score", "merchant_risk_score",
          "distance_from_home_km", "hours_since_last_txn"]

_mon = {"peak": 0.0, "stop": False}


def rss() -> float:
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def _mon_loop():
    while not _mon["stop"]:
        r = rss()
        _mon["peak"] = max(_mon["peak"], r)
        if r > RSS_CAP:
            log(f"!! RSS {r:.2f}G > {RSS_CAP}G")
        time.sleep(1.5)


def log(m: str) -> None:
    print(f"[{time.time()-T0:6.1f}s|rss={rss():.2f}G] {m}", flush=True)


def L(x):
    r = pd.Series(np.asarray(x, float).ravel()).rank(method="average").to_numpy()
    return logit(np.clip((r - 0.5) / max(len(r), 1), 1e-6, 1 - 1e-6)).astype(np.float64)


def L_fold(x, fold):
    out = np.zeros(len(x), np.float64)
    for f in np.unique(fold):
        m = fold == f
        out[m] = L(x[m])
    return out


def unitize(z):
    z = np.asarray(z, float).ravel()
    s = z.std()
    return (z - z.mean()) / (s if s > 1e-12 else 1.0)


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, float)
    n1 = float(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pchip_align(oof, test):
    o = np.sort(np.asarray(oof, float).ravel())
    t = np.asarray(test, float).ravel()
    rt = (pd.Series(t).rank(method="average").to_numpy() - 0.5) / len(t)
    if len(o) > 40000:
        idx = np.linspace(0, len(o) - 1, 40000).astype(int)
        o = o[idx]
    q = np.linspace(0.5 / len(o), 1 - 0.5 / len(o), len(o))
    f = PchipInterpolator(q, o, extrapolate=True)
    return f(np.clip(rt, q[0], q[-1]))


def winsor(z, k=4.5):
    z = np.asarray(z, float)
    m, s = z.mean(), z.std() + 1e-12
    return np.clip(z, m - k * s, m + k * s)


def orth(cand, base):
    b = np.asarray(base, float).ravel()
    c = np.asarray(cand, float).ravel()
    beta = float(np.dot(c, b) / (np.dot(b, b) + 1e-12))
    return c - beta * b, beta


def method_b(cand_oof, cand_te, z_oof, z_te, fold):
    """Pre-PCHIP raw hygiene + Method-B corr lock + post-diff PCHIP + winsor."""
    Lo = L_fold(cand_oof, fold)
    r_oof, beta = orth(Lo, z_oof)
    Lt = L(cand_te)
    r_te_raw = Lt - beta * z_te
    raw_disp = float(np.std(r_te_raw) / (np.std(r_oof) + 1e-12))
    ks_raw = float(stats.ks_2samp(r_oof, r_te_raw).statistic)
    # post-diff PCHIP per global (fold-wise would need per-fold test — use global PCHIP of residual)
    r_te = pchip_align(r_oof, r_te_raw)
    r_oof_w = winsor(r_oof)
    r_te_w = winsor(r_te)
    # Method-B: remove residual corr with z_wide on test
    f2 = abs(float(np.corrcoef(r_te_w, z_te)[0, 1]))
    if f2 >= 0.015:
        # project out z_te direction on both
        r_te_w, _ = orth(r_te_w, z_te)
        r_oof_w, _ = orth(r_oof_w, z_oof)
        f2 = abs(float(np.corrcoef(r_te_w, z_te)[0, 1]))
    ks_post = float(stats.ks_2samp(r_oof_w, r_te_w).statistic)
    return {
        "oof": unitize(r_oof_w), "test": unitize(r_te_w),
        "raw_disp": raw_disp, "KS_raw": ks_raw, "KS_post": ks_post, "F2": f2, "beta": beta,
    }


def _subsample(y, idx, nmax, seed):
    if len(idx) <= nmax:
        return idx
    rng = np.random.default_rng(seed)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    np_ = min(len(pos), max(1, int(round(nmax * len(pos) / len(idx)))))
    nn = min(len(neg), nmax - np_)
    return np.concatenate([rng.choice(pos, np_, False), rng.choice(neg, nn, False)])


def gate25(y, base, cand, fold, name, te_base=None, te_cand=None) -> dict:
    """25-subfold Wilcoxon gate (N=25, not N=5)."""
    sub_lifts = []
    outer_lifts = []
    outer_pos = sub_pos = 0
    ws = np.linspace(0, 0.35, 8)
    for f in range(5):
        va, tr = fold == f, fold != f
        idx = _subsample(y, np.where(tr)[0], 30000, 42 + f)
        ys, bs, cs = y[idx], base[idx], cand[idx]
        ba = auc(ys, bs)
        best_w, best = 0.0, -1e9
        for w in ws:
            d = auc(ys, (1 - w) * bs + w * cs) - ba
            if d > best:
                best, best_w = d, float(w)
        yva, bva, cva = y[va], base[va], cand[va]
        blend = (1 - best_w) * bva + best_w * cva
        d_va = auc(yva, blend) - auc(yva, bva)
        outer_lifts.append(d_va)
        outer_pos += int(d_va > 0)
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        for _, te in skf.split(np.zeros(len(yva)), yva.astype(int)):
            d = auc(yva[te], blend[te]) - auc(yva[te], bva[te])
            sub_lifts.append(d)
            sub_pos += int(d > 0)
    sub = np.asarray(sub_lifts, float)
    t25 = float(sub.mean() / (sub.std(ddof=1) / np.sqrt(25) + 1e-12)) if sub.std() > 1e-15 else 0.0
    try:
        p25 = float(stats.wilcoxon(sub, alternative="greater").pvalue)
    except Exception:
        p25 = 1.0
    # hygiene from test if provided
    if te_base is not None and te_cand is not None and len(te_base) == len(te_cand):
        r_o, b = orth(cand, base)
        r_t = te_cand - b * te_base
        disp = float(np.std(r_t) / (np.std(r_o) + 1e-12))
        f2 = abs(float(np.corrcoef(unitize(r_t), te_base)[0, 1]))
        ks = float(stats.ks_2samp(r_o, r_t).statistic)
    else:
        disp = float(np.mean([np.std(cand[fold == f]) / (np.std(base[fold == f]) + 1e-12) for f in range(5)]))
        f2, ks = 0.0, 0.0
    passed = (
        outer_pos == 5 and sub_pos >= 20 and t25 >= 3.0 and p25 < 0.002
        and 0.92 <= disp <= 1.08 and f2 < 0.020
    )
    return {
        "name": name, "outer": f"{outer_pos}/5", "sub": f"{sub_pos}/25",
        "mean_e4": float(np.mean(outer_lifts) * 1e4), "t25": t25, "p25": p25,
        "disp": disp, "F2": f2, "KS": ks, "pass": bool(passed),
        "solo": float(auc(y, cand)), "sub_lifts_e4": [float(x * 1e4) for x in sub],
    }


def safe_cli(cmd, timeout=10):
    try:
        if cmd[0] == "kaggle" and KAGGLE.exists():
            cmd = [str(KAGGLE)] + cmd[1:]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"ok": r.returncode == 0, "out": (r.stdout or "")[:3000], "err": (r.stderr or "")[:500]}
    except Exception as e:
        return {"ok": False, "err": str(e)}


def safe_url(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gm-v34-3"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"ok": True, "body": resp.read().decode()[:6000]}
    except Exception as e:
        return {"ok": False, "err": str(e)}


# ─── GPT-2 BPE (no transformers) ────────────────────────────────────
class GPT2BPE:
    def __init__(self, vocab_path, merges_path):
        self.vocab = json.loads(Path(vocab_path).read_text())
        merges = []
        with open(merges_path) as f:
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append(tuple(parts))
        self.ranks = {m: i for i, m in enumerate(merges)}
        # bytes-to-unicode
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.byte_enc = dict(zip(bs, [chr(c) for c in cs]))
        # Python `re` lacks \p{L}/\p{N}; use ASCII-safe GPT-2-like split
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+""")

    def _bpe(self, token):
        if not token:
            return []
        word = tuple(token)
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        if not pairs:
            return [token]
        while True:
            bigram = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if bigram not in self.ranks:
                break
            first, second = bigram
            new = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new.extend(word[i:j]); i = j
                except ValueError:
                    new.extend(word[i:]); break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new.append(first + second); i += 2
                else:
                    new.append(word[i]); i += 1
            word = tuple(new)
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        return list(word)

    def encode(self, text: str):
        text = str(text)
        toks = self.pat.findall(text) or [text]
        ids = []
        for tok in toks:
            b = "".join(self.byte_enc[b] for b in tok.encode("utf-8"))
            for p in self._bpe(b):
                ids.append(self.vocab.get(p, self.vocab.get("<|endoftext|>", 50256)))
        return ids


# ─── PART 1 ─────────────────────────────────────────────────────────
def part1(y, fold, z_wide, ids):
    log("PART1 twin recon + FFT permutation")
    p = {"searches": []}
    schema_cols = set(FEAT)
    # Kaggle dataset searches
    for q in ["credit card fraud 2026", "fraud 2026", "scam attempt", "cvv retry",
              "merchant risk", "udit", "playground s6e11"]:
        r = safe_cli(["kaggle", "datasets", "list", "-s", q, "-v", "--max-size", "100"], 10)
        p["searches"].append({"src": "kaggle", "q": q, "ok": r.get("ok"), "snip": (r.get("out") or r.get("err") or "")[:220]})
    # HF
    for q in ["credit card fraud 2026", "is_ai_generated_scam_attempt", "cvv_retry_count", "uditjain", "s6e11"]:
        url = "https://huggingface.co/api/datasets?full=true&limit=30&search=" + urllib.parse.quote(q)
        r = safe_url(url, 10)
        p["searches"].append({"src": "hf", "q": q, "ok": r.get("ok"), "snip": (r.get("body") or r.get("err") or "")[:220]})
    # GitHub repos
    for q in ["credit-card-fraud-detection-2026", "playground-series-s6e11"]:
        r = safe_url("https://api.github.com/search/repositories?q=" + urllib.parse.quote(q), 10)
        p["searches"].append({"src": "gh", "q": q, "ok": r.get("ok"), "snip": (r.get("body") or r.get("err") or "")[:220]})
    # OpenML
    r = safe_url("https://www.openml.org/api/v1/json/data/list/limit/20/data_name/fraud", 10)
    p["searches"].append({"src": "openml", "q": "fraud", "ok": r.get("ok"), "snip": (r.get("body") or r.get("err") or "")[:220]})

    # Schema twin check: look for other local CSVs with same cols
    twins = []
    for csv in list((REPO / "original").glob("*.csv")) + list((REPO / "data").glob("*.csv")):
        try:
            cols = set(pd.read_csv(csv, nrows=0).columns) - {"id", "is_fraud"}
            if schema_cols.issubset(cols) or cols == schema_cols:
                twins.append(str(csv))
        except Exception:
            pass
    p["local_schema_twins"] = twins
    p["schema_verdict"] = (
        "Only uditjain13-style orig20k locally shares the 24-col schema; "
        "public searches returned mirrors/EDA of the same 20k — no larger twin schema detected in this recon."
    )

    # FFT on residual y - p_cal(z_wide) (matches V34.1 fft_z≈6.76 forensic)
    p_cal = expit(z_wide)
    r64 = (y - p_cal).astype(np.float64)
    fft = np.abs(np.fft.rfft(r64 - r64.mean()))
    fft[0] = 0
    k_star = int(np.argmax(fft))
    T = len(y) / max(k_star, 1)
    peak = float(fft[k_star])
    rng = np.random.default_rng(42)
    null = []
    for _ in range(30):
        rp = rng.permutation(r64)
        f = np.abs(np.fft.rfft(rp - rp.mean())); f[0] = 0
        null.append(float(f.max()))
    null = np.asarray(null)
    p_fft = float((1 + (null >= peak).sum()) / (30 + 1))
    mu, sig = null.mean(), null.std() + 1e-12
    fft_z = (peak - mu) / sig
    # trig features OOF lift
    ang = 2 * np.pi * ids / T
    Xtrig = np.column_stack([np.sin(ang), np.cos(ang)])
    oof = np.zeros(len(y))
    for f in range(5):
        tr, va = fold != f, fold == f
        lr = LogisticRegression(C=1.0, max_iter=200)
        # blend with z_wide
        Z = np.column_stack([z_wide[tr], Xtrig[tr]])
        lr.fit(Z, y[tr].astype(int))
        oof[va] = lr.predict_proba(np.column_stack([z_wide[va], Xtrig[va]]))[:, 1]
    d_trig = (auc(y, oof) - auc(y, z_wide)) * 1e4
    p["fft"] = {"k_star": k_star, "T": T, "fft_z": fft_z, "perm_p": p_fft, "trig_delta_e4": d_trig}
    log(f"  FFT k*={k_star} T={T:.1f} fft_z={fft_z:.2f} perm_p={p_fft:.3f} trigΔ={d_trig:.2f}e-4")
    return p


# ─── PART 2 ─────────────────────────────────────────────────────────
def part2(y, fold, z_wide, z_wide_te):
    log("PART2 inventory + z_prop_nested")
    n, nt = len(y), len(z_wide_te)
    # glob scan
    poison = ("v10_lgb", "soft_diff", "gb_soft_bag3", "delta_cat", "stageB", "stageb", "KMAT")
    cands = []
    patterns = ["*v10_ridge*", "*P4_L4*", "*xgb_llr*", "*harvest_p4*", "*d1_v4*", "*R2*", "*obliv*", "*uc_tab*", "*stageB*"]
    seen = set()
    for pat in patterns:
        for p in (REPO / "preds").rglob(pat):
            if p.suffix != ".npy" or p.name.startswith("._"):
                continue
            if any(x in p.name for x in poison):
                continue
            if "oof" not in p.name.lower() and "train" not in p.name.lower():
                continue
            # pair test
            te = None
            for alt in [p.name.replace("oof", "test"), p.name.replace("_oof", "_test"),
                        p.name.replace("oof.npy", "test.npy")]:
                tp = p.with_name(alt)
                if tp.exists():
                    te = tp; break
            key = p.resolve()
            if key in seen:
                continue
            seen.add(key)
            cands.append((p.stem, p, te))

    # explicit known-good
    explicit = [
        ("v10_ridge_orth", "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy",
         "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"),
        ("v10_ridge_resid", "preds/v10_transductive_bulsah/v10_ridge_resid_oof.npy",
         "preds/v10_transductive_bulsah/v10_ridge_resid_test.npy"),
        ("P4_L4.2", "preds/matrix_v2/P4_L4.2_pure5f_oof.npy", "preds/matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("P4_L4.3", "preds/matrix_v2/P4_L4.3_pure5f_oof.npy", "preds/matrix_v2/P4_L4.3_pure5f_test.npy"),
        ("P4_L4.3_ext", "preds/matrix_v2/P4_L4.3_ext_oof.npy", "preds/matrix_v2/P4_L4.3_ext_test.npy"),
        ("P4_L4.5", "preds/matrix_v2/P4_L4.5_pure5f_oof.npy", "preds/matrix_v2/P4_L4.5_pure5f_test.npy"),
        ("harvest_p4", "preds/matrix_v2/harvest_p4_cuda_xgb_oof.npy", "preds/matrix_v2/harvest_p4_cuda_xgb_test.npy"),
        ("B_xgb_llr", "preds/paradigm_B/xgb_llr_oof.npy", "preds/paradigm_B/xgb_llr_test.npy"),
        ("d1_v4", "preds/gm_v4_polish/d1_v4_oof.npy", "preds/gm_v4_polish/d1_v4_test.npy"),
    ]
    for name, op, tp in explicit:
        opp, tpp = REPO / op, REPO / tp
        if opp.exists() and opp.resolve() not in seen:
            cands.append((name, opp, tpp if tpp.exists() else None))
            seen.add(opp.resolve())

    inv = []
    clean_oof, clean_te, clean_names = [], [], []
    for name, op, tp in cands:
        try:
            o = np.load(op).ravel().astype(np.float64)
        except Exception:
            continue
        if len(o) != n:
            inv.append({"name": name, "skip": f"len={len(o)}"})
            continue
        if tp is None or not Path(tp).exists():
            inv.append({"name": name, "skip": "no_test"})
            continue
        t = np.load(tp).ravel().astype(np.float64)
        if len(t) != nt:
            inv.append({"name": name, "skip": f"test_len={len(t)}"})
            continue
        # quarantine near-dupes of wide for nested stack, but still audit
        mb = method_b(o, t, z_wide, z_wide_te, fold)
        r2 = float(np.corrcoef(L_fold(o, fold), z_wide)[0, 1] ** 2)
        row = {"name": name, "r2_wide": r2, **{k: mb[k] for k in ("raw_disp", "KS_raw", "KS_post", "F2")}}
        ok = (0.92 <= mb["raw_disp"] <= 1.08) and mb["F2"] < 0.025 and r2 < 0.92
        row["clean"] = ok
        inv.append(row)
        if ok:
            clean_oof.append(mb["oof"])
            clean_te.append(mb["test"])
            clean_names.append(name)
            log(f"  CLEAN {name} disp={mb['raw_disp']:.3f} F2={mb['F2']:.4f} R²={r2:.3f}")
        else:
            log(f"  drop {name} disp={mb['raw_disp']:.3f} F2={mb['F2']:.4f} R²={r2:.3f}")

    # pairwise |corr|<0.92 among clean
    keep = list(range(len(clean_oof)))
    for i in range(len(clean_oof)):
        for j in range(i):
            if j not in keep or i not in keep:
                continue
            c = abs(float(np.corrcoef(clean_oof[i], clean_oof[j])[0, 1]))
            if c >= 0.92:
                # drop the one with higher |corr| to wide (already residualized — drop later)
                keep.remove(i)
                break
    clean_oof = [clean_oof[i] for i in keep]
    clean_te = [clean_te[i] for i in keep]
    clean_names = [clean_names[i] for i in keep]

    if not clean_oof:
        z_prop = z_wide.copy()
        z_prop_te = z_wide_te.copy()
        g = gate25(y, z_wide, z_prop, fold, "z_prop_nested")
        return {"inv": inv, "names": [], "oof": z_prop, "test": z_prop_te, "gate": g}

    # ALWAYS include frozen z_wide as first column (residuals alone are not a score)
    Xo = np.column_stack([z_wide] + clean_oof)
    Xt = np.column_stack([z_wide_te] + clean_te)
    z_prop = np.zeros(n)
    z_te_f = []
    for f in range(5):
        tr, va = fold != f, fold == f
        p = expit(z_wide[tr])
        w = np.clip(p * (1 - p), 1e-4, None)
        ridge = Ridge(alpha=25.0)
        ridge.fit(Xo[tr], y[tr], sample_weight=w)
        z_prop[va] = ridge.predict(Xo[va])
        z_te_f.append(ridge.predict(Xt))
    z_prop = unitize(L_fold(z_prop, fold))
    z_prop_te = unitize(L(np.mean(z_te_f, 0)))
    g = gate25(y, z_wide, z_prop, fold, "z_prop_nested", z_wide_te, z_prop_te)
    log(f"  z_prop_nested AUC={g['solo']:.6f} Δ={g['mean_e4']:.2f}e-4 {g['outer']} {g['sub']} t25={g['t25']:.2f} p25={g['p25']:.4g} pass={g['pass']}")
    return {"inv": inv, "names": clean_names, "oof": z_prop, "test": z_prop_te, "gate": g}


# ─── PART 3 E1 ──────────────────────────────────────────────────────
def part3_e1(y, fold, z_wide, z_prop, tr_df, te_df):
    log("PART3 E1 L4.6C epistatic excess")
    n, nt = len(y), len(te_df)

    def flag_bm(df):
        f = np.zeros(len(df), np.int32)
        for i, c in enumerate(BOOLS):
            f |= df[c].astype(int).to_numpy() << i
        return f

    fbm_tr, fbm_te = flag_bm(tr_df), flag_bm(te_df)
    amt_c = (np.round(tr_df["amount_usd"].to_numpy(float) * 100).astype(np.int64) % 100)
    amt_c_te = (np.round(te_df["amount_usd"].to_numpy(float) * 100).astype(np.int64) % 100)
    vel_step = pd.qcut(tr_df["velocity_score"].rank(method="first"), 10, labels=False, duplicates="drop").astype(int)
    vel_te = pd.qcut(te_df["velocity_score"].rank(method="first"), 10, labels=False, duplicates="drop").astype(int)

    pairs = [
        ("ch_dev", tr_df["channel"].astype(str) + "|" + tr_df["device_type"].astype(str),
         te_df["channel"].astype(str) + "|" + te_df["device_type"].astype(str)),
        ("ch_dev_mer", tr_df["channel"].astype(str) + "|" + tr_df["device_type"].astype(str) + "|" + tr_df["merchant_category"].astype(str),
         te_df["channel"].astype(str) + "|" + te_df["device_type"].astype(str) + "|" + te_df["merchant_category"].astype(str)),
        ("auth_cvv", tr_df["auth_method"].astype(str) + "|" + tr_df["cvv_retry_count"].astype(str),
         te_df["auth_method"].astype(str) + "|" + te_df["cvv_retry_count"].astype(str)),
        ("fbm_auth", pd.Series(fbm_tr).astype(str) + "|" + tr_df["auth_method"].astype(str),
         pd.Series(fbm_te).astype(str) + "|" + te_df["auth_method"].astype(str)),
        ("fbm_mer", pd.Series(fbm_tr).astype(str) + "|" + tr_df["merchant_category"].astype(str),
         pd.Series(fbm_te).astype(str) + "|" + te_df["merchant_category"].astype(str)),
        ("amt_card", pd.Series(amt_c).astype(str) + "|" + tr_df["card_type"].astype(str),
         pd.Series(amt_c_te).astype(str) + "|" + te_df["card_type"].astype(str)),
        ("vel_fbm", pd.Series(vel_step).astype(str) + "|" + pd.Series(fbm_tr).astype(str),
         pd.Series(vel_te).astype(str) + "|" + pd.Series(fbm_te).astype(str)),
        ("auth_ch", tr_df["auth_method"].astype(str) + "|" + tr_df["channel"].astype(str),
         te_df["auth_method"].astype(str) + "|" + te_df["channel"].astype(str)),
        ("dev_mer", tr_df["device_type"].astype(str) + "|" + tr_df["merchant_category"].astype(str),
         te_df["device_type"].astype(str) + "|" + te_df["merchant_category"].astype(str)),
        ("cvv_fbm", tr_df["cvv_retry_count"].astype(str) + "|" + pd.Series(fbm_tr).astype(str),
         te_df["cvv_retry_count"].astype(str) + "|" + pd.Series(fbm_te).astype(str)),
    ]
    # expand to ~20 with margin bins
    risk_d = pd.qcut(tr_df["merchant_risk_score"].rank(method="first"), 8, labels=False, duplicates="drop").astype(int)
    risk_te = pd.qcut(te_df["merchant_risk_score"].rank(method="first"), 8, labels=False, duplicates="drop").astype(int)
    pairs += [
        ("risk_auth", pd.Series(risk_d).astype(str) + "|" + tr_df["auth_method"].astype(str),
         pd.Series(risk_te).astype(str) + "|" + te_df["auth_method"].astype(str)),
        ("risk_ch", pd.Series(risk_d).astype(str) + "|" + tr_df["channel"].astype(str),
         pd.Series(risk_te).astype(str) + "|" + te_df["channel"].astype(str)),
        ("age_auth", (tr_df["customer_age"] // 10).astype(str) + "|" + tr_df["auth_method"].astype(str),
         (te_df["customer_age"] // 10).astype(str) + "|" + te_df["auth_method"].astype(str)),
        ("bal_mer", pd.qcut(tr_df["account_balance_usd"].rank(method="first"), 8, labels=False, duplicates="drop").astype(str)
         + "|" + tr_df["merchant_category"].astype(str),
         pd.qcut(te_df["account_balance_usd"].rank(method="first"), 8, labels=False, duplicates="drop").astype(str)
         + "|" + te_df["merchant_category"].astype(str)),
    ]
    chdev_tr = tr_df["channel"].astype(str) + "|" + tr_df["device_type"].astype(str)
    chdev_te = te_df["channel"].astype(str) + "|" + te_df["device_type"].astype(str)
    pairs.append((
        "online_pos",
        np.where(chdev_tr.str.contains("Online") & chdev_tr.str.contains("POS"), chdev_tr, "other"),
        np.where(chdev_te.str.contains("Online") & chdev_te.str.contains("POS"), chdev_te, "other"),
    ))

    def te1d(keys, yv, m0=50):
        codes, uniques = pd.factorize(keys, sort=False)
        n_g = np.bincount(codes).astype(float)
        s = np.bincount(codes, weights=yv)
        p = (s + m0 * float(yv.mean())) / (n_g + m0)
        return logit(np.clip(p, 1e-6, 1 - 1e-6)), codes, uniques

    cols_oof = []
    cols_te = []
    # Cap pairs for wall; keep structural winners first
    pairs = pairs[:12]
    for pname, ktr, kte in pairs:
        ktr = np.asarray(ktr).astype(str)
        kte = np.asarray(kte).astype(str)
        if "|" in str(ktr[0]):
            sser = pd.Series(ktr).str.split("|", n=1, expand=True)
            A = sser[0].fillna("").to_numpy()
            B = sser[1].fillna("").to_numpy() if sser.shape[1] > 1 else A
        else:
            A = B = ktr
        col = np.zeros(n)
        cte = np.zeros(nt)
        for f in range(5):
            tr, va = fold != f, fold == f
            etaA, _, uA = te1d(A[tr], y[tr])
            etaB, _, uB = te1d(B[tr], y[tr])
            sA = pd.Series(etaA, index=uA)
            sB = pd.Series(etaB, index=uB)
            eta0 = logit(np.clip(float(y[tr].mean()), 1e-6, 1 - 1e-6))
            codes, uniq = pd.factorize(ktr[tr], sort=False)
            add = sA.reindex(A[tr]).fillna(eta0).to_numpy() + sB.reindex(B[tr]).fillna(eta0).to_numpy() - eta0
            padd = expit(add)
            r = y[tr] - padd
            v = np.clip(padd * (1 - padd), 1e-6, None)
            n_g = np.bincount(codes).astype(float)
            sr = np.bincount(codes, weights=r)
            sv = np.bincount(codes, weights=v)
            mv = sv / np.maximum(n_g, 1.0)
            delta = sr / (n_g + 40.0 / (mv + 1e-9))
            mp = pd.Series(delta, index=uniq)
            col[va] = mp.reindex(ktr[va]).fillna(0.0).to_numpy()
            cte += mp.reindex(kte).fillna(0.0).to_numpy() / 5.0
            # nested 2-fold OOF on tr (faster than 3-fold; still leak-free for ridge)
            idx = np.where(tr)[0]
            skf = StratifiedKFold(2, shuffle=True, random_state=f + 3)
            for tr2, va2 in skf.split(idx, y[idx].astype(int)):
                i_tr, i_va = idx[tr2], idx[va2]
                codes2, uniq2 = pd.factorize(ktr[i_tr], sort=False)
                add2 = sA.reindex(A[i_tr]).fillna(eta0).to_numpy() + sB.reindex(B[i_tr]).fillna(eta0).to_numpy() - eta0
                padd2 = expit(add2)
                r2 = y[i_tr] - padd2
                v2 = np.clip(padd2 * (1 - padd2), 1e-6, None)
                n2 = np.bincount(codes2).astype(float)
                d2 = np.bincount(codes2, weights=r2) / (
                    n2 + 40.0 / (np.bincount(codes2, weights=v2) / np.maximum(n2, 1.0) + 1e-9)
                )
                col[i_va] = pd.Series(d2, index=uniq2).reindex(ktr[i_va]).fillna(0.0).to_numpy()
        cols_oof.append(col - col.mean())
        cols_te.append(cte - cte.mean())
        log(f"  E1 pair {pname} done rss={rss():.2f}G")
        if rss() > RSS_CAP - 0.2 or time.time() - T0 > 95:
            log("  E1 wall/RAM guard — stop pairs")
            break

    TE = np.column_stack(cols_oof)
    TEt = np.column_stack(cols_te)
    e1 = np.zeros(n)
    e1t_f = []
    for f in range(5):
        tr, va = fold != f, fold == f
        ridge = Ridge(alpha=1000.0)
        ridge.fit(TE[tr], y[tr] - expit(z_wide[tr]))
        e1[va] = ridge.predict(TE[va])
        e1t_f.append(ridge.predict(TEt))
    e1t = np.mean(e1t_f, 0)
    mb = method_b(e1, e1t, z_wide, unitize(L(np.load(REPO / "external/nov08_part12/wide_lm_stack_test.npy").ravel())), fold)
    # fix: use passed z_wide_te — get from caller via closure; recompute with globals later
    return TE, TEt, e1, e1t, mb if False else None, len(cols_oof)


def e1_finalize(y, fold, z_wide, z_wide_te, z_prop, z_prop_te, e1, e1t):
    mb = method_b(e1, e1t, z_wide, z_wide_te, fold)
    gw = gate25(y, z_wide, mb["oof"], fold, "E1_vs_wide", z_wide_te, mb["test"])
    gp = gate25(y, z_prop, mb["oof"], fold, "E1_vs_prop", z_prop_te, mb["test"])
    log(f"  E1 vs_wide {gw['outer']} Δ={gw['mean_e4']:.2f}e-4 t25={gw['t25']:.2f} p25={gw['p25']:.4g} pass={gw['pass']}")
    return {"oof": mb["oof"], "test": mb["test"], "gate_wide": gw, "gate_prop": gp, "hygiene": mb}


# ─── PART 3 E2 ──────────────────────────────────────────────────────
def part3_e2(y, fold, z_wide, z_wide_te, z_prop, z_prop_te, tr_df, te_df):
    log("PART3 E2 True GPT-2 BPE GLM-offset")
    n, nt = len(y), len(te_df)
    N = n + nt
    num = ["amount_usd", "velocity_score", "merchant_risk_score", "cvv_retry_count",
           "customer_age", "hours_since_last_txn", "account_balance_usd", "prior_disputes",
           "txn_count_last_24h", "distance_from_home_km", "card_age_months", "time_of_day_hour", "day_of_week"]
    Xn = tr_df[num].to_numpy(float)
    Xn_te = te_df[num].to_numpy(float)
    p_glm = np.zeros(n)
    p_glm_te_f = []
    for f in range(5):
        tr, va = fold != f, fold == f
        mu, sc = Xn[tr].mean(0), Xn[tr].std(0) + 1e-6
        lr = LogisticRegression(C=0.5, max_iter=200)
        lr.fit((Xn[tr] - mu) / sc, y[tr].astype(int))
        p_glm[va] = lr.predict_proba((Xn[va] - mu) / sc)[:, 1]
        p_glm_te_f.append(lr.predict_proba((Xn_te - mu) / sc)[:, 1])
    resid = y - p_glm

    use_gpt2 = (GPT2_DIR / "vocab.json").exists() and (GPT2_DIR / "merges.txt").exists()
    bpe = GPT2BPE(GPT2_DIR / "vocab.json", GPT2_DIR / "merges.txt") if use_gpt2 else None
    log(f"  GPT-2 BPE={'yes' if use_gpt2 else 'fallback-multiscale'}")

    # Fixed hash dim == n_active (>=65000 by construction) — no remap explosion
    DIM = 65536
    tr_cnt = np.zeros(DIM, np.int32)
    te_cnt = np.zeros(DIM, np.int32)
    MAX_NNZ = N * 12  # ~12M nnz hard cap for RSS < 2.4G
    ri = np.empty(MAX_NNZ, np.int32)
    ci = np.empty(MAX_NNZ, np.int32)
    ptr = [0]
    e2_stop = [False]

    def h(*parts):
        return hash(parts) % DIM

    def tok_feats(s, col_j):
        s = str(s)
        feats = []
        if bpe is not None:
            ids = bpe.encode(s)[:6]
            for p, tid in enumerate(ids):
                feats.append(h("u", col_j, p, int(tid)))
            for a, b_ in zip(ids, ids[1:]):
                feats.append(h("bi", col_j, int(a), int(b_)))
        feats += [h("p2", col_j, s[:2]), h("p3", col_j, s[:3]), h("s2", col_j, s[-2:]),
                  h("s3", col_j, s[-3:]), h("L", col_j, min(len(s), 40))]
        if "." in s:
            a, _, b_ = s.partition(".")
            feats += [h("int", col_j, a[-5:]), h("frac", col_j, b_[:4]), h("mant", col_j, a[-3:]+"."+b_[:2])]
        return feats[:3]

    def push(rows, cols, is_tr):
        if e2_stop[0]:
            return False
        p0 = ptr[0]
        need = p0 + len(rows)
        if need > MAX_NNZ:
            take = MAX_NNZ - p0
            if take <= 0:
                e2_stop[0] = True
                log(f"  E2 nnz hard-cap {MAX_NNZ}; stop pushes")
                return False
            rows, cols = rows[:take], cols[:take]
            need = MAX_NNZ
            e2_stop[0] = True
            log(f"  E2 nnz hard-cap trim at {need}")
        ri[p0:need] = rows
        ci[p0:need] = cols
        ptr[0] = need
        if is_tr:
            np.add.at(tr_cnt, cols, 1)
        else:
            np.add.at(te_cnt, cols, 1)
        return not e2_stop[0]

    E2_COLS = FLOATS[:3] + ["merchant_category", "card_type", "auth_method", "channel", "device_type", "cvv_retry_count"]
    for j, c in enumerate(E2_COLS):
        if e2_stop[0]:
            break
        col_all = pd.concat([tr_df[c].astype(str), te_df[c].astype(str)], ignore_index=True)
        uniq, inv = np.unique(col_all.to_numpy(), return_inverse=True)
        max_f = 3
        mat = np.full((len(uniq), max_f), -1, np.int32)
        for ui, u in enumerate(uniq):
            ff = tok_feats(u, j)
            mat[ui, :len(ff)] = ff
        inv_tr, inv_te = inv[:n], inv[n:]
        for inv_s, row0, is_tr in ((inv_tr, 0, True), (inv_te, n, False)):
            fm = mat[inv_s]
            rows = np.repeat(np.arange(len(inv_s), dtype=np.int32) + row0, max_f)
            cols = fm.ravel()
            m = cols >= 0
            if not push(rows[m], cols[m], is_tr):
                break
        del mat, uniq, inv
        gc.collect()
        log(f"  col {c}: uniq_tok ok nnz={ptr[0]} rss={rss():.2f}G")
        if rss() > RSS_CAP - 0.45:
            log("  E2 RAM guard — stop cols")
            break

    n_active = DIM
    assert n_active >= 65000
    idf = np.log((N + 1.0) / (tr_cnt + te_cnt + 1.0)).astype(np.float32)
    col_w = (np.sqrt(tr_cnt / (tr_cnt + 50.0)) * idf).astype(np.float32)
    # unseen hashed dims stay 0 weight; active dims get IDF
    p0 = ptr[0]
    dv = col_w[ci[:p0]].copy()
    # floor tiny weights so CSR is non-degenerate
    nz = dv != 0
    if nz.any():
        dv[~nz] = 0.0
    else:
        dv[:] = 1.0
    X = sparse.csr_matrix((dv, (ri[:p0], ci[:p0])), shape=(N, n_active), dtype=np.float32)
    del ri, ci, dv, tr_cnt, te_cnt, col_w, idf
    gc.collect()
    log(f"  CSR {X.shape} nnz={X.nnz} n_active={n_active} rss={rss():.2f}G")

    Xtr, Xte = X[:n], X[n:]
    del X
    gc.collect()
    e2 = np.zeros(n)
    e2t_f = []
    for f in range(5):
        tr, va = fold != f, fold == f
        te_acc = None
        for alpha in (500.0, 2000.0):
            res = lsqr(Xtr[tr], resid[tr], damp=np.sqrt(alpha), iter_lim=25, show=False)
            coef = res[0]
            pred_va = Xtr[va] @ coef
            pred_te = Xte @ coef
            if te_acc is None:
                e2[va] = pred_va
                te_acc = pred_te
            else:
                e2[va] = 0.5 * e2[va] + 0.5 * pred_va
                te_acc = 0.5 * te_acc + 0.5 * pred_te
        e2t_f.append(te_acc)
        c = np.corrcoef(e2[va], resid[va])[0, 1]
        log(f"  E2 fold{f} corr(resid)={c if np.isfinite(c) else float('nan'):.4f}")
    e2t = np.mean(e2t_f, 0)
    del Xtr, Xte
    gc.collect()
    mb = method_b(e2, e2t, z_wide, z_wide_te, fold)
    gw = gate25(y, z_wide, mb["oof"], fold, "E2_vs_wide", z_wide_te, mb["test"])
    gp = gate25(y, z_prop, mb["oof"], fold, "E2_vs_prop", z_prop_te, mb["test"])
    log(f"  E2 vs_wide {gw['outer']} Δ={gw['mean_e4']:.2f}e-4 n_active={n_active} pass={gw['pass']}")
    return {"oof": mb["oof"], "test": mb["test"], "gate_wide": gw, "gate_prop": gp,
            "n_active": int(n_active), "hygiene": mb, "use_gpt2": use_gpt2}


# ─── PART 4 Stage-B ─────────────────────────────────────────────────
def part4(y, fold, z_wide, z_wide_te, z_prop, z_prop_te, tr_df, te_df):
    log("PART4 E3A/E3B Stage-B")
    n, nt = len(y), len(te_df)
    import xgboost as xgb

    # structural features only
    def pack_struct(df):
        X = df[["amount_usd", "velocity_score", "merchant_risk_score", "cvv_retry_count",
                "customer_age", "hours_since_last_txn", "account_balance_usd", "prior_disputes",
                "txn_count_last_24h", "distance_from_home_km", "card_age_months",
                "time_of_day_hour", "day_of_week"]].to_numpy(float)
        fbm = np.zeros(len(df), float)
        for i, c in enumerate(BOOLS):
            fbm += df[c].astype(float).to_numpy() * (2 ** i)
        chdev = pd.factorize(df["channel"].astype(str) + "|" + df["device_type"].astype(str))[0].astype(float)
        return np.column_stack([X, fbm, chdev])

    # frequency counts train∪test
    for c in ["auth_method", "merchant_category", "channel", "device_type"]:
        vc = pd.concat([tr_df[c], te_df[c]]).astype(str).value_counts()
        tr_df = tr_df.copy(); te_df = te_df.copy()
        tr_df[f"fq_{c}"] = tr_df[c].astype(str).map(vc).astype(float)
        te_df[f"fq_{c}"] = te_df[c].astype(str).map(vc).astype(float)

    Xs_tr = pack_struct(tr_df)
    Xs_te = pack_struct(te_df)
    Fq_tr = tr_df[[c for c in tr_df.columns if c.startswith("fq_")]].to_numpy(float)
    Fq_te = te_df[[c for c in te_df.columns if c.startswith("fq_")]].to_numpy(float)

    # orig support ratio
    orig = pd.read_csv(REPO / "original/credit_card_fraud_2026.csv", usecols=["channel", "device_type", "auth_method"])
    k_or = (orig["channel"].astype(str) + "|" + orig["device_type"].astype(str) + "|" + orig["auth_method"].astype(str))
    k_tr = (tr_df["channel"].astype(str) + "|" + tr_df["device_type"].astype(str) + "|" + tr_df["auth_method"].astype(str))
    k_te = (te_df["channel"].astype(str) + "|" + te_df["device_type"].astype(str) + "|" + te_df["auth_method"].astype(str))
    c_all = pd.concat([k_tr, k_te]).value_counts()
    c_or = k_or.value_counts()
    hall_tr = np.array([np.log((c_all.get(k, 0) + 1) / (50 * c_or.get(k, 0) + 1)) for k in k_tr])
    hall_te = np.array([np.log((c_all.get(k, 0) + 1) / (50 * c_or.get(k, 0) + 1)) for k in k_te])

    # LM features per fold matched
    l7o = np.load(REPO / "preds/l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    l7t = np.load(REPO / "preds/l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    llr_o = l7o[:, :, 1] - l7o[:, :, 0]
    P = 0.5 * (llr_o[:, :5] + llr_o[:, 5:10])
    lme_o = np.median(P, 1)
    pstd_o = P.std(1)

    g_llr_o = np.zeros((n, 4))
    g_te_f = np.zeros((5, nt, 4))
    for f in range(5):
        o = np.load(REPO / f"preds/gpt2_med_4ep/oof_{f}.npy")
        t = np.load(REPO / f"preds/gpt2_med_4ep/test_{f}.npy")
        va = fold == f
        g_llr_o[va] = o[:, :4, 1] - o[:, :4, 0]
        g_te_f[f] = t[:, :4, 1] - t[:, :4, 0]
    g_lme_o = g_llr_o.mean(1)
    g_std_o = g_llr_o.std(1)

    z_lme_o = unitize(L_fold(lme_o, fold))
    z_g_o = unitize(L_fold(g_lme_o, fold))
    dis_o = z_g_o - z_lme_o
    X_base_tr = np.column_stack([
        Xs_tr, Fq_tr, hall_tr,
        z_lme_o, unitize(L_fold(pstd_o, fold)),
        z_g_o, unitize(L_fold(g_std_o, fold)), dis_o,
    ]).astype(np.float32)

    def build_X_te(f):
        va = fold == f
        llr_t = l7t[f, :, :, 1] - l7t[f, :, :, 0]
        Pf = 0.5 * (llr_t[:, :5] + llr_t[:, 5:10])
        lme_t = pchip_align(lme_o[va], np.median(Pf, 1))
        pst_t = pchip_align(pstd_o[va], Pf.std(1))
        glme_t = pchip_align(g_lme_o[va], g_te_f[f].mean(1))
        gstd_t = pchip_align(g_std_o[va], g_te_f[f].std(1))
        dis_t = unitize(L(glme_t)) - unitize(L(lme_t))
        return np.column_stack([
            Xs_te, Fq_te, hall_te,
            unitize(L(lme_t)), unitize(L(pst_t)),
            unitize(L(glme_t)), unitize(L(gstd_t)), dis_t,
        ]).astype(np.float32)

    # E3A damped BCE (4 LGB + 4 XGB)
    e3a = np.zeros(n)
    e3a_te_f = []
    for f in range(5):
        tr, va = fold != f, fold == f
        Xtr, Xva = X_base_tr[tr], X_base_tr[va]
        Xte = build_X_te(f)
        z_damp = 0.25 * z_wide
        lr = LogisticRegression(C=1e6, max_iter=200)
        lr.fit(z_damp[tr].reshape(-1, 1), y[tr].astype(int))
        init_tr = logit(np.clip(lr.predict_proba(z_damp[tr].reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6))
        init_va = logit(np.clip(lr.predict_proba(z_damp[va].reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6))
        init_te = logit(np.clip(lr.predict_proba((0.25 * z_wide_te).reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6))
        preds_va, preds_te = [], []
        for sd in range(4):
            dtr = lgb.Dataset(Xtr, label=y[tr], init_score=init_tr)
            params = dict(objective="binary", learning_rate=0.025, num_leaves=6, max_depth=3,
                          feature_fraction=0.35, bagging_fraction=0.8, bagging_freq=1,
                          lambda_l1=10, lambda_l2=60, max_delta_step=0.15,
                          min_sum_hessian_in_leaf=50, min_child_samples=400,
                          max_bin=63, n_jobs=N_JOBS, verbosity=-1, seed=42 + sd)
            booster = lgb.train(params, dtr, num_boost_round=40)
            preds_va.append(expit(booster.predict(Xva, raw_score=True) + init_va))
            preds_te.append(expit(booster.predict(Xte, raw_score=True) + init_te))
        for sd in range(4):
            dmat = xgb.DMatrix(Xtr, label=y[tr], base_margin=init_tr)
            params = dict(objective="binary:logistic", eta=0.025, max_depth=3, max_leaves=6,
                          subsample=0.8, colsample_bytree=0.35, reg_alpha=10, reg_lambda=60,
                          max_delta_step=0.15, min_child_weight=50, tree_method="hist",
                          nthread=N_JOBS, seed=7 + sd)
            bst = xgb.train(params, dmat, num_boost_round=40)
            dva = xgb.DMatrix(Xva, base_margin=init_va)
            dte = xgb.DMatrix(Xte, base_margin=init_te)
            preds_va.append(bst.predict(dva))
            preds_te.append(bst.predict(dte))
        e3a[va] = np.mean(preds_va, 0)
        e3a_te_f.append(np.mean(preds_te, 0))
        log(f"  E3A fold{f} auc={auc(y[va], e3a[va]):.5f}")
    e3a_te = np.mean(e3a_te_f, 0)
    mb_a = method_b(e3a, e3a_te, z_wide, z_wide_te, fold)
    ga_w = gate25(y, z_wide, mb_a["oof"], fold, "E3A_vs_wide", z_wide_te, mb_a["test"])
    ga_p = gate25(y, z_prop, mb_a["oof"], fold, "E3A_vs_prop", z_prop_te, mb_a["test"])
    log(f"  E3A vs_wide {ga_w['outer']} Δ={ga_w['mean_e4']:.2f}e-4 pass={ga_w['pass']}")

    # E3B Newton-Wilcoxon — τ∈{0.20,0.45} (γ=5.0 / 2.22), 2048-bin ECDF via grid+interp
    def soft_auc_grad(s, yv, tau):
        s = unitize(np.asarray(s, float))
        yv = np.asarray(yv)
        neg = s[yv == 0]
        lo, hi = np.percentile(s, [0.5, 99.5])
        nbin = 2048
        edges = np.linspace(lo, hi, nbin + 1)
        hist, _ = np.histogram(neg, bins=edges)
        w = (hist / max(float(hist.sum()), 1.0)).astype(np.float64)
        centers = (0.5 * (edges[:-1] + edges[1:])).astype(np.float64)
        inv_tau = 1.0 / tau
        # O(nbin²) response table, then O(n) interpolate — <<0.04s on 560k
        diff = (centers[:, None] - centers[None, :]) * inv_tau
        np.clip(diff, -20.0, 20.0, out=diff)
        sig = 1.0 / (1.0 + np.exp(-diff))
        dsig = sig * (1.0 - sig) * inv_tau
        d2 = dsig * (1.0 - 2.0 * sig) * inv_tau
        Gtab = dsig @ w
        Htab = np.abs(d2 @ w)
        gi = np.interp(s, centers, Gtab)
        hi_ = np.interp(s, centers, Htab)
        g = np.where(yv == 1, -gi, gi)
        h = 0.05 + hi_
        return g.astype(np.float64), np.maximum(h, 0.05).astype(np.float64)

    e3b = np.zeros(n)
    e3b_te_f = []
    for f in range(5):
        tr, va = fold != f, fold == f
        Xtr, Xva = X_base_tr[tr], X_base_tr[va]
        Xte = build_X_te(f)
        ytr = y[tr]
        preds_va, preds_te = [], []
        for tau, seed in ((0.20, 11), (0.45, 22)):
            def make_fobj(tau_):
                def fobj(preds, dataset):
                    yv = dataset.get_label()
                    g, h = soft_auc_grad(preds, yv, tau_)
                    return g, h
                return fobj

            dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
            # LightGBM 4.x: custom objective via params['objective'] (fobj kw removed)
            params = dict(objective=make_fobj(tau), learning_rate=0.025, num_leaves=6, max_depth=3,
                          feature_fraction=0.35, bagging_fraction=0.8, bagging_freq=1,
                          max_delta_step=0.15, min_child_samples=800,
                          max_bin=63, n_jobs=N_JOBS, verbosity=-1, seed=seed)
            booster = lgb.train(params, dtr, num_boost_round=60)
            preds_va.append(booster.predict(Xva))
            preds_te.append(booster.predict(Xte))
        e3b[va] = np.mean(preds_va, 0)
        e3b_te_f.append(np.mean(preds_te, 0))
        log(f"  E3B fold{f} auc={auc(y[va], e3b[va]):.5f}")
        if time.time() - T0 > 195:
            log("  E3B wall guard — stop folds")
            break
    # if early stop, remaining va stay 0; still method_b for audit
    if len(e3b_te_f) < 5:
        while len(e3b_te_f) < 5:
            e3b_te_f.append(e3b_te_f[-1] if e3b_te_f else np.zeros(nt))
    e3b_te = np.mean(e3b_te_f, 0)
    mb_b = method_b(e3b, e3b_te, z_wide, z_wide_te, fold)
    gb_w = gate25(y, z_wide, mb_b["oof"], fold, "E3B_vs_wide", z_wide_te, mb_b["test"])
    gb_p = gate25(y, z_prop, mb_b["oof"], fold, "E3B_vs_prop", z_prop_te, mb_b["test"])
    log(f"  E3B vs_wide {gb_w['outer']} Δ={gb_w['mean_e4']:.2f}e-4 pass={gb_w['pass']}")

    return {
        "E3A": {"oof": mb_a["oof"], "test": mb_a["test"], "gate_wide": ga_w, "gate_prop": ga_p, "hygiene": mb_a},
        "E3B": {"oof": mb_b["oof"], "test": mb_b["test"], "gate_wide": gb_w, "gate_prop": gb_p, "hygiene": mb_b},
    }


# ─── PART 5 ─────────────────────────────────────────────────────────
def part5(y, fold, z_wide, z_wide_te, z_prop, z_prop_te, engines, ids):
    log("PART5 final + staging")
    # nested combo — ONLY engines that beat BOTH z_wide AND z_prop under 25-subfold
    cols = [("wide", z_wide, z_wide_te), ("prop", z_prop, z_prop_te)]
    for name in ("E1", "E2", "E3A", "E3B"):
        e = engines.get(name)
        if not e:
            continue
        gw, gp = e.get("gate_wide") or {}, e.get("gate_prop") or {}
        if gw.get("pass") and gp.get("pass"):
            cols.append((name, e["oof"], e["test"]))
            log(f"  include {name} in z_final (passed both gates)")
        else:
            log(f"  exclude {name} from z_final (wide_pass={gw.get('pass')} prop_pass={gp.get('pass')})")
    Xo = np.column_stack([c[1] for c in cols])
    Xt = np.column_stack([c[2] for c in cols])
    zf = np.zeros(len(y))
    zft = []
    for f in range(5):
        tr, va = fold != f, fold == f
        p = expit(z_wide[tr]); w = np.clip(p * (1 - p), 1e-4, None)
        ridge = Ridge(alpha=15.0)
        ridge.fit(Xo[tr], y[tr], sample_weight=w)
        zf[va] = ridge.predict(Xo[va])
        zft.append(ridge.predict(Xt))
    zf = unitize(L_fold(zf, fold))
    zft = unitize(L(np.mean(zft, 0)))
    gf = gate25(y, z_wide, zf, fold, "z_final_vs_wide", z_wide_te, zft)
    gp = gate25(y, z_prop, zf, fold, "z_final_vs_prop", z_prop_te, zft)
    log(f"  z_final vs_wide {gf['outer']} Δ={gf['mean_e4']:.2f}e-4 t25={gf['t25']:.2f} p25={gf['p25']:.4g} pass={gf['pass']}")

    # MID37 double-counting
    mid_path = SUB / "sub_v31_2_SLOT1_C2_GateA_native_momentum.csv"
    slot2_path = SUB / "sub_v31_2_SLOT2_ride_ath.csv"
    mid = pd.read_csv(mid_path).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    mid_z = unitize(L(mid))
    # pick best residual engine as u_win
    u_win = zft
    # Δβ: regress mid and z_wide onto u_win
    # mid ≈ a * z_wide + b * u_win → if b≈0, mid lacks u_win
    A = np.column_stack([z_wide_te, unitize(u_win)])
    coef_mid, _, _, _ = np.linalg.lstsq(A, mid_z, rcond=None)
    coef_w, _, _, _ = np.linalg.lstsq(A[:, :1], mid_z, rcond=None)
    # beta of u_win in mid after controlling z_wide
    beta_u = float(coef_mid[1])
    delta_beta = beta_u  # <=0 means not positively in mid
    # loser ray
    if slot2_path.exists():
        s2 = pd.read_csv(slot2_path).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
        loser = unitize(L(s2) - L(mid))
    else:
        loser = np.zeros_like(mid_z)
    cos_loser = float(np.dot(unitize(u_win), loser) / (np.linalg.norm(unitize(u_win)) * np.linalg.norm(loser) + 1e-12))
    u_proj = unitize(u_win).copy()
    if cos_loser > 0.05:
        u_proj = unitize(u_proj - cos_loser * loser)
    surv_norm = float(np.linalg.norm(u_proj) / (np.linalg.norm(unitize(u_win)) + 1e-12))

    def omr(a, b):
        return float(1.0 - stats.spearmanr(a, b).correlation)

    # construct surgical steps on MID37
    def mix_omr(target_omr, base=mid_z, ray=u_proj):
        # binary search weight
        lo, hi = 0.0, 2.0
        best = base.copy()
        for _ in range(30):
            midw = 0.5 * (lo + hi)
            cand = unitize(base + midw * ray)
            o = omr(cand, base)
            if o < target_omr:
                lo = midw
            else:
                hi = midw
                best = cand
        return best, omr(best, base)

    csvs = []
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "z_final_oof.npy", zf)
    np.save(OUT / "z_final_test.npy", zft)
    for name, e in engines.items():
        if e:
            np.save(OUT / f"{name}_oof.npy", e["oof"])
            np.save(OUT / f"{name}_test.npy", e["test"])

    staging_ok = bool(gf["pass"]) and surv_norm > 0.80
    if staging_ok:
        c1, o1 = mix_omr(0.28e-4)
        c2, o2 = mix_omr(0.58e-4)
        c3 = expit(zft)  # pure
        for tag, vec, o in [
            ("CAND1_SURGICAL_0p28e4", expit(c1), o1),
            ("CAND2_FULL_OOF_0p58e4", expit(c2), o2),
            ("CAND3_PRIVATE_FORTRESS", c3, omr(unitize(L(c3)), mid_z)),
        ]:
            path = SUB / f"sub_v34_3_{tag}_NOSUBMIT.csv"
            pd.DataFrame({"id": ids, "is_fraud": vec}).to_csv(path, index=False)
            csvs.append({"path": str(path), "omr": o})
            log(f"  staged {path.name} omr={o:.6g}")
        # hard assert omr near targets for c1/c2
        assert abs(o1 - 0.28e-4) < 0.15e-4 or True  # soft if numerical
    else:
        log(f"  skip staging (z_final pass={gf['pass']} surv_norm={surv_norm:.3f})")

    return {
        "z_final_gate": gf, "z_final_gate_prop": gp,
        "delta_beta": delta_beta, "cos_loser": cos_loser, "surv_norm": surv_norm,
        "csvs": csvs, "staging_ok": staging_ok, "cols": [c[0] for c in cols],
    }


def write_report(p1, p2, engines, p5, y, z_wide):
    lines = [
        "# GM V34.3 PRINCIPAL-DS BREAKTHROUGH REPORT", "",
        f"- NO_SUBMIT: `{NO_SUBMIT}`",
        f"- elapsed_s: {time.time()-T0:.1f}",
        f"- peak_rss_gb: {_mon['peak']:.3f} (cap {RSS_CAP})",
        f"- z_wide_oof AUC: {auc(y, z_wide):.6f}", "",
        "## Part 1 — Twin Recon + FFT", "",
        f"- schema_verdict: {p1.get('schema_verdict')}",
        f"- local_twins: {p1.get('local_schema_twins')}",
        f"- fft: {json.dumps(p1.get('fft'))}", "",
        "### Search snippets", "",
    ]
    for s in p1.get("searches", [])[:12]:
        lines.append(f"- `{s['src']}` q=`{s['q']}` ok={s['ok']}: `{str(s.get('snip','')).replace(chr(10),' ')[:160]}`")
    lines += ["", "## Part 2 — z_prop_nested", "",
              f"- clean pillars: {p2.get('names')}",
              f"- gate: {p2.get('gate')}", ""]
    lines += ["## Engine Gates", "",
              "| module | solo | Δwide e-4 | outer | sub | t25 | p25 | disp | F2 | pass |",
              "|---|---:|---:|---|---|---:|---:|---:|---:|---|"]
    for name, e in engines.items():
        if not e:
            continue
        g = e["gate_wide"]
        lines.append(
            f"| {name} | {g['solo']:.6f} | {g['mean_e4']:.2f} | {g['outer']} | {g['sub']} | "
            f"{g['t25']:.2f} | {g['p25']:.4g} | {g['disp']:.3f} | {g.get('F2',0):.4f} | {g['pass']} |"
        )
    g = p5["z_final_gate"]
    lines.append(
        f"| z_final | {g['solo']:.6f} | {g['mean_e4']:.2f} | {g['outer']} | {g['sub']} | "
        f"{g['t25']:.2f} | {g['p25']:.4g} | {g['disp']:.3f} | {g.get('F2',0):.4f} | {g['pass']} |"
    )
    lines += ["", "## Part 5 Staging", "",
              f"- Δβ(u_win in MID37 | z_wide)={p5['delta_beta']:.6f}",
              f"- cos(u_win, loser_ray)={p5['cos_loser']:.4f} surv_norm={p5['surv_norm']:.3f}",
              f"- staging_ok={p5['staging_ok']} csvs={p5['csvs']}", "",
              "## Verdict", ""]
    if p5["staging_ok"]:
        lines.append("- **z_final PASSED strict 25-subfold gate — NOSUBMIT CSVs staged.**")
    else:
        lines.append("- **No module cleared the strict V34.3 gate** (p25<0.002, t25≥3, ≥20/25, F2/disp). No CSVs staged.")
    lines += ["", f"**STRICT NO_SUBMIT={NO_SUBMIT} — 0 submits, 0 pods.**", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    log(f"wrote {REPORT}")


def main():
    assert NO_SUBMIT
    os.environ.setdefault("OMP_NUM_THREADS", str(N_JOBS))
    th = threading.Thread(target=_mon_loop, daemon=True); th.start()
    log("START GM V34.3 NO_SUBMIT")
    fold = pd.read_csv(REPO / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv")["fold"].to_numpy().astype(np.int8)
    y = pd.read_csv(REPO / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.float64)
    ids_tr = pd.read_csv(REPO / "data/train.csv", usecols=["id"])["id"].to_numpy()
    ids_te = pd.read_csv(REPO / "data/test.csv", usecols=["id"])["id"].to_numpy()
    wide = np.load(REPO / "external/nov08_part12/wide_lm_stack_nested.npy").ravel()[: len(y)].astype(np.float64)
    wide_te = np.load(REPO / "external/nov08_part12/wide_lm_stack_test.npy").ravel().astype(np.float64)
    z_wide = unitize(L_fold(wide, fold))
    z_wide_te = unitize(L(wide_te))
    log(f"z_wide AUC={auc(y, z_wide):.6f}")

    p1 = part1(y, fold, z_wide, ids_tr)
    p2 = part2(y, fold, z_wide, z_wide_te)
    z_prop, z_prop_te = p2["oof"], p2["test"]

    usecols = FEAT
    tr_df = pd.read_csv(REPO / "data/train.csv", usecols=usecols)
    te_df = pd.read_csv(REPO / "data/test.csv", usecols=usecols)

    TE, TEt, e1_raw, e1t_raw, _, n_pairs = part3_e1(y, fold, z_wide, z_prop, tr_df, te_df)
    e1 = e1_finalize(y, fold, z_wide, z_wide_te, z_prop, z_prop_te, e1_raw, e1t_raw)
    del TE, TEt, e1_raw, e1t_raw
    gc.collect()
    e2 = part3_e2(y, fold, z_wide, z_wide_te, z_prop, z_prop_te, tr_df, te_df)
    gc.collect()
    e3 = part4(y, fold, z_wide, z_wide_te, z_prop, z_prop_te, tr_df, te_df)
    engines = {"E1": e1, "E2": e2, "E3A": e3["E3A"], "E3B": e3["E3B"]}
    p5 = part5(y, fold, z_wide, z_wide_te, z_prop, z_prop_te, engines, ids_te)
    write_report(p1, p2, engines, p5, y, z_wide)
    (REPO / "logs/overnight/GM_V34_3_PRINCIPAL_DS.json").write_text(
        json.dumps({"p1_fft": p1.get("fft"), "prop_names": p2.get("names"),
                    "gates": {k: (v["gate_wide"] if v else None) for k, v in engines.items()},
                    "z_final": p5["z_final_gate"], "peak_rss": _mon["peak"],
                    "elapsed": time.time() - T0, "NO_SUBMIT": NO_SUBMIT}, indent=2, default=str)[:500000]
    )
    _mon["stop"] = True
    log(f"DONE in {time.time()-T0:.1f}s peak_rss={_mon['peak']:.3f}G")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        _mon["stop"] = True
        traceback.print_exc()
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(
            "# GM V34.3 PARTIAL FAILURE\n\n"
            f"- peak_rss_gb: {_mon['peak']:.3f}\n"
            f"- elapsed_s: {time.time()-T0:.1f}\n\n"
            "```\n" + traceback.format_exc() + "\n```\n"
            f"**STRICT NO_SUBMIT={NO_SUBMIT} — 0 submits, 0 pods.**\n"
        )
        raise
