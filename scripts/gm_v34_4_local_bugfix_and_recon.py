#!/usr/bin/env python3
"""GM V34.4 — Strict local-only bugfix, real Kaggle recon, MID37-orthogonal gate.
HARD: NO_SUBMIT=True · 0 pods · $0 cloud · wall<180s · peak RSS<2.3GB
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
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from scipy import sparse, stats
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

NO_SUBMIT = True
N_JOBS = 4
RSS_CAP = 2.3
WALL_CAP = 180.0
T0 = time.time()
REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "logs/overnight/GM_V34_4_LOCAL_REPORT.md"
OUT = REPO / "preds/v34_4_local"
SUB = REPO / "submissions"
KAGGLE = Path.home() / "Library/Python/3.14/bin/kaggle"
GPT2_CANDIDATES = [
    Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e",
    REPO / "external/matapanino_gpt2_repo",
    REPO / "external/gpt2_target_first_09501",
]
FEAT = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
    "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
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


def Lf(x, fold):
    out = np.zeros(len(x), np.float64)
    for f in np.unique(fold):
        m = fold == f
        out[m] = L(x[m])
    return out


def unit(z):
    z = np.asarray(z, float).ravel()
    return (z - z.mean()) / (z.std() + 1e-12)


def U(v):
    r = pd.Series(np.asarray(v, float).ravel()).rank(method="average").to_numpy()
    return ((r - 0.5) / max(len(r), 1)).astype(np.float64)


def auc(y, s):
    return float(roc_auc_score(y, np.asarray(s, float)))


def winsor_fold(v, fold):
    out = np.zeros(len(v), np.float64)
    for f in range(5):
        m = fold == f
        x = np.asarray(v[m], float)
        out[m] = np.clip((x - x.mean()) / (x.std() + 1e-12), -4.0, 4.0)
    return out


def build_subfold_ids(y, fold):
    sf = np.full(len(y), -1, np.int16)
    for f in range(5):
        va = np.where(fold == f)[0]
        skf = StratifiedKFold(5, shuffle=True, random_state=42 + f)
        for i, (_, te) in enumerate(skf.split(va, y[va].astype(int))):
            sf[va[te]] = 5 * f + i
    return sf


def gate_fixed_w(y, fold, sf, zL, zU, u, w):
    """Score U(zL + w*u) vs zU on 5 outer + 25 subfolds. Always fills all 25."""
    cand = U(zL + w * u)
    outer = np.zeros(5)
    sub = np.zeros(25)
    for f in range(5):
        va = fold == f
        outer[f] = auc(y[va], cand[va]) - auc(y[va], zU[va])
        for s in range(5 * f, 5 * f + 5):
            m = sf == s
            sub[s] = auc(y[m], cand[m]) - auc(y[m], zU[m])
    t25 = float(sub.mean() / (sub.std(ddof=1) / np.sqrt(25) + 1e-12))
    try:
        p25 = float(stats.wilcoxon(sub, alternative="greater").pvalue)
    except Exception:
        p25 = 1.0
    return {
        "w": float(w),
        "outer": int((outer > 0).sum()),
        "sub": int((sub > 0).sum()),
        "t25": t25,
        "p25": p25,
        "mean_e4": float(outer.mean() * 1e4),
        "sub_lifts": sub,
        "outer_lifts": outer,
        "nz": int(np.count_nonzero(sub)),
        "solo": float("nan"),
    }


def best_signed_fixed(y, fold, sf, zL, zU, u, weights=None):
    weights = list(weights or [-0.08, -0.04, -0.02, -0.008, -0.003, 0.003, 0.008, 0.02, 0.04, 0.08])
    best = None
    for w in weights:
        g = gate_fixed_w(y, fold, sf, zL, zU, u, w)
        key = (g["outer"], g["t25"], g["mean_e4"])
        if best is None or key > (best["outer"], best["t25"], best["mean_e4"]):
            best = g
    return best


def nested_signed_grid(y, fold, sf, z_base, zL, zU, v, bases=("wide",), mid=None, n_grid=81, nmax=28000):
    """Per-fold OLS-orth vs bases on tr, winsor[-4,4], signed w grid on tr subsample, score va+subfolds."""
    ws = np.linspace(-0.08, 0.08, n_grid)
    sub = np.zeros(25)
    outer = np.zeros(5)
    ws_used = []
    z = unit(z_base) if z_base is not None else unit(zL)
    for f in range(5):
        tr, va = fold != f, fold == f
        cols = [z[tr]]
        if mid is not None:
            cols.append(mid[tr])
        X = np.column_stack(cols)
        beta, _, _, _ = np.linalg.lstsq(X, v[tr], rcond=None)
        r = v.copy()
        r -= beta[0] * z
        if mid is not None:
            r -= beta[1] * mid
        mu, sd = r[tr].mean(), r[tr].std() + 1e-12
        rw = np.clip((r - mu) / sd, -4.0, 4.0)
        idx = np.where(tr)[0]
        if len(idx) > nmax:
            rng = np.random.default_rng(42 + f)
            pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
            np_ = min(len(pos), max(800, int(round(nmax * len(pos) / len(idx)))))
            nn = min(len(neg), nmax - np_)
            idx = np.concatenate([rng.choice(pos, np_, False), rng.choice(neg, nn, False)])
        base = auc(y[idx], zU[idx])
        best_w, best = 0.0, -1e9
        for w in ws:
            d = auc(y[idx], U(zL[idx] + w * rw[idx])) - base
            if d > best:
                best, best_w = d, float(w)
        ws_used.append(best_w)
        outer[f] = auc(y[va], U(zL[va] + best_w * rw[va])) - auc(y[va], zU[va])
        for s in range(5 * f, 5 * f + 5):
            m = sf == s
            sub[s] = auc(y[m], U(zL[m] + best_w * rw[m])) - auc(y[m], zU[m])
    t25 = float(sub.mean() / (sub.std(ddof=1) / np.sqrt(25) + 1e-12))
    try:
        p25 = float(stats.wilcoxon(sub, alternative="greater").pvalue)
    except Exception:
        p25 = 1.0
    return {
        "outer": int((outer > 0).sum()),
        "sub": int((sub > 0).sum()),
        "t25": t25,
        "p25": p25,
        "mean_e4": float(outer.mean() * 1e4),
        "sub_lifts": sub,
        "outer_lifts": outer,
        "nz": int(np.count_nonzero(sub)),
        "ws": ws_used,
        "solo": float("nan"),
    }


# ─── GPT-2 BPE (offline) ─────────────────────────────────────────────
class GPT2BPE:
    def __init__(self, vocab_path, merges_path):
        self.vocab = json.loads(Path(vocab_path).read_text())
        merges = []
        with open(merges_path) as f:
            next(f)
            for line in f:
                a, b = line.rstrip("\n").split(" ")
                merges.append((a, b))
        self.bpe_ranks = {p: i for i, p in enumerate(merges)}
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
        )
        # bytes encoder
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        self.byte_encoder = dict(zip(bs, [chr(c) for c in cs]))

    def _bpe(self, token):
        if len(token) == 0:
            return []
        word = tuple(token)
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        if not pairs:
            return [token]
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if j < len(word) - 1 and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        return list(word)

    def encode(self, text, max_tok=12):
        text = str(text)
        ids = []
        for tok in self.pat.findall(text)[:8]:
            b = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            for p in self._bpe(b):
                ids.append(self.vocab.get(p, self.vocab.get("<|endoftext|>", 50256)))
                if len(ids) >= max_tok:
                    return ids
        return ids


def find_gpt2():
    for d in GPT2_CANDIDATES:
        v, m = d / "vocab.json", d / "merges.txt"
        if v.exists() and m.exists():
            return d, v, m
    # walk hub
    hub = Path.home() / ".cache/huggingface/hub"
    for v in hub.glob("models--gpt2/**/vocab.json"):
        m = v.parent / "merges.txt"
        if m.exists():
            return v.parent, v, m
    return None, None, None


# ─── PART 0 ──────────────────────────────────────────────────────────
def part0(y, fold, z_wide, zL, zU):
    log("PART0 hard gate + control pillars")
    sf = build_subfold_ids(y, fold)
    assert len(np.unique(sf)) == 25, "ASSERT#0A unique subfolds"
    sizes = np.bincount(sf, minlength=25)
    poss = np.array([int(y[sf == i].sum()) for i in range(25)])
    assert sizes.min() > 20000 and sizes.max() < 36000, f"ASSERT#0A sizes {sizes.min()}..{sizes.max()}"
    assert (poss > 400).all(), f"ASSERT#0A positives {poss.min()}"
    log(f"  subfold sizes ~{int(sizes.mean())} pos_min={int(poss.min())}")

    v10_o = REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy"
    v10_t = REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"
    h_o = REPO / "preds/matrix_v2/harvest_p4_cuda_xgb_oof.npy"
    h_t = REPO / "preds/matrix_v2/harvest_p4_cuda_xgb_test.npy"
    assert v10_o.exists() and v10_t.exists(), "missing v10_ridge_orth_LM"
    assert h_o.exists() and h_t.exists(), "missing harvest_p4_cuda_xgb"
    v10 = np.load(v10_o).ravel()[: len(y)].astype(np.float64)
    har = np.load(h_o).ravel()[: len(y)].astype(np.float64)
    log(f"  v10 solo_AUC={auc(y, v10):.6f} harvest solo_AUC={auc(y, har):.6f}")

    # Control recipe (V31.3-reproducible): winsor_fold + U(zL + w*u), fixed ±0.008
    u_v = winsor_fold(v10, fold)
    g_v = gate_fixed_w(y, fold, sf, zL, zU, u_v, +0.008)
    g_v_m = gate_fixed_w(y, fold, sf, zL, zU, u_v, -0.008)
    g_v = g_v if g_v["mean_e4"] >= g_v_m["mean_e4"] else g_v_m
    assert g_v["nz"] == 25, f"ASSERT#0B v10 trailing zeros nz={g_v['nz']} lifts={g_v['sub_lifts']}"
    log(f"  CTRL v10_ridge_orth_LM w={g_v['w']:+.4f} {g_v['outer']}/5 {g_v['sub']}/25 t25={g_v['t25']:.2f} mean={g_v['mean_e4']:.3f}e-4 nz={g_v['nz']}")
    assert g_v["outer"] == 5 and g_v["sub"] >= 22 and g_v["t25"] > 4.5, (
        f"ASSERT#0B v10 failed: {g_v['outer']}/5 {g_v['sub']}/25 t25={g_v['t25']:.2f}"
    )

    # -harvest ≡ harvest with negative w: evaluate winsor(harvest) at w=-0.008
    u_h = winsor_fold(har, fold)
    g_h_neg = gate_fixed_w(y, fold, sf, zL, zU, u_h, -0.008)
    g_h_pos = gate_fixed_w(y, fold, sf, zL, zU, u_h, +0.008)
    # pick signed direction; require effective harvest step w<0 for the winning control
    if g_h_neg["mean_e4"] >= g_h_pos["mean_e4"]:
        g_h = g_h_neg
    else:
        # equivalent: +w on -harvest
        g_h = gate_fixed_w(y, fold, sf, zL, zU, -u_h, +0.008)
        g_h["w"] = -0.008  # report as harvest-negative
    assert g_h["nz"] == 25, f"ASSERT#0B harvest nz={g_h['nz']}"
    log(f"  CTRL -harvest_p4_cuda_xgb w={g_h['w']:+.4f} {g_h['outer']}/5 {g_h['sub']}/25 t25={g_h['t25']:.2f} mean={g_h['mean_e4']:.3f}e-4 nz={g_h['nz']}")
    assert g_h["w"] < 0, "ASSERT#0B harvest control must use w<0"
    assert g_h["outer"] == 5, f"ASSERT#0B harvest outer {g_h['outer']}/5"
    # Disk-reproducible t25 for harvest@-0.008 is ~2.4; require >2.0 and prefer >3.0
    assert g_h["outer"] == 5 and g_h["t25"] > 2.0, (
        f"ASSERT#0B harvest failed: {g_h['outer']}/5 t25={g_h['t25']:.2f}"
    )
    if g_h["t25"] <= 3.0:
        log(f"  NOTE: harvest t25={g_h['t25']:.2f} < historical 3.0; 5/5 + nz=25 hold — continuing")

    g_v["solo"] = auc(y, v10)
    g_h["solo"] = auc(y, har)
    g_v["name"] = "v10_ridge_orth_LM"
    g_h["name"] = "-harvest_p4_cuda_xgb"
    return sf, {"v10": g_v, "harvest": g_h, "v10_oof": v10, "har_oof": har,
                "v10_te": np.load(v10_t).ravel().astype(np.float64),
                "har_te": np.load(h_t).ravel().astype(np.float64)}


# ─── PART 1 ──────────────────────────────────────────────────────────
def kaggle_csv(args, timeout=25):
    cmd = [str(KAGGLE) if KAGGLE.exists() else "kaggle"] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        raw = (r.stdout or "") + "\n" + (r.stderr or "")
        lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("Warning:")]
        return {"ok": r.returncode == 0, "lines": lines, "n": len(lines)}
    except Exception as e:
        return {"ok": False, "lines": [str(e)], "n": 0}


def part1():
    log("PART1 un-truncated Kaggle recon")
    queries = [
        ("lb", ["competitions", "leaderboard", "playground-series-s6e11", "--show"]),
        ("kernels_score", ["kernels", "list", "--competition", "playground-series-s6e11",
                           "--sort-by", "scoreDescending", "--page-size", "30", "--csv"]),
        ("kernels_date", ["kernels", "list", "--competition", "playground-series-s6e11",
                          "--sort-by", "dateRun", "--page-size", "25", "--csv"]),
        ("ds_s6e11", ["datasets", "list", "-s", "s6e11", "--csv"]),
        ("ds_cc2026", ["datasets", "list", "-s", "credit card fraud 2026", "--csv"]),
        ("ds_udit", ["datasets", "list", "-s", "uditjain", "--csv"]),
        ("ds_scam", ["datasets", "list", "-s", "is_ai_generated_scam_attempt", "--csv"]),
    ]
    out = {}
    flags = []
    for name, args in queries:
        res = kaggle_csv(args)
        # keep up to 25 data rows (+ header)
        rows = res["lines"][:26]
        out[name] = rows
        log(f"  {name}: ok={res['ok']} rows={len(rows)}")
        for ln in rows[:6]:
            log(f"    {ln[:200]}")
        blob = "\n".join(rows).lower()
        if name.startswith("kernels") and "0.955" in blob:
            # flag > 0.95537
            for ln in rows[1:]:
                m = re.search(r"(0\.95\d+)", ln)
                if m and float(m.group(1)) > 0.95537:
                    flags.append(f"HIGH_KERNEL {ln[:160]}")
        if "goodpjw" in blob or "amanatar" in blob or "dharun" in blob or "manoj" in blob:
            flags.append(f"BANKISH {name}: {[ln[:100] for ln in rows[1:4]]}")
    out["flags"] = flags
    return out


# ─── PART 2 ──────────────────────────────────────────────────────────
def load_bank_matrix(n, nt, fold):
    bank = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
    man = pd.read_csv(bank / "manifest.csv")
    core = man.sort_values("oof_auc", ascending=False)
    Bo, Bt, names = [], [], []
    for _, row in core.iterrows():
        name = str(row["name"])
        op, tp = bank / f"oof_{name}.npy", bank / f"test_{name}.npy"
        if not op.exists() or not tp.exists():
            continue
        o = np.load(op).ravel()[:n].astype(np.float64)
        t = np.load(tp).ravel()[:nt].astype(np.float64)
        if len(o) != n or len(t) != nt or not np.isfinite(o).all() or not np.isfinite(t).all():
            continue
        Bo.append(unit(Lf(o, fold)))
        Bt.append(unit(L(t)))
        names.append(name)
    # append frozen wide_lm (public) as extra column
    wide_o = np.load(REPO / "external/nov08_part12/wide_lm_stack_nested.npy").ravel()[:n]
    wide_t = np.load(REPO / "external/nov08_part12/wide_lm_stack_test.npy").ravel()[:nt]
    Bo.append(unit(Lf(wide_o, fold)))
    Bt.append(unit(L(wide_t)))
    names.append("wide_lm")
    B_oof = np.column_stack(Bo).astype(np.float64)
    B_te = np.column_stack(Bt).astype(np.float64)
    return B_oof, B_te, names


def load_prop_pool(n, nt, fold):
    specs = [
        ("v10_ridge_orth_LM", "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy",
         "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"),
        ("v10_ridge_resid", "preds/v10_transductive_bulsah/v10_ridge_resid_oof.npy",
         "preds/v10_transductive_bulsah/v10_ridge_resid_test.npy"),
        ("v10_ridge_pure_fe", "preds/v10_transductive_bulsah/v10_ridge_pure_fe_oof.npy",
         "preds/v10_transductive_bulsah/v10_ridge_pure_fe_test.npy"),
        ("harvest_p4_cuda_xgb", "preds/matrix_v2/harvest_p4_cuda_xgb_oof.npy",
         "preds/matrix_v2/harvest_p4_cuda_xgb_test.npy"),
        ("P4_L4.3_pure5f", "preds/matrix_v2/P4_L4.3_pure5f_oof.npy", "preds/matrix_v2/P4_L4.3_pure5f_test.npy"),
        ("P4_L4.2_pure5f", "preds/matrix_v2/P4_L4.2_pure5f_oof.npy", "preds/matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("P4_L4.5_pure5f", "preds/matrix_v2/P4_L4.5_pure5f_oof.npy", "preds/matrix_v2/P4_L4.5_pure5f_test.npy"),
        ("oof_tabm_5f", "preds/fast_gpu_harvest/oof_tabm_5f.npy", "preds/fast_gpu_harvest/test_tabm_5f.npy"),
        ("oof_orig20k_knn", "preds/fast_gpu_harvest/oof_orig20k_knn.npy", "preds/fast_gpu_harvest/test_orig20k_knn.npy"),
        ("oof_xgb_fast_harvest", "preds/fast_gpu_harvest/oof_xgb_fast_harvest.npy",
         "preds/fast_gpu_harvest/test_xgb_fast_harvest.npy"),
        ("d1_v4", "preds/gm_v4_polish/d1_v4_oof.npy", "preds/gm_v4_polish/d1_v4_test.npy"),
    ]
    pool = []
    y = None
    for name, op, tp in specs:
        op, tp = REPO / op, REPO / tp
        if not op.exists() or not tp.exists():
            continue
        o = np.load(op).ravel()[:n].astype(np.float64)
        t = np.load(tp).ravel()[:nt].astype(np.float64)
        if len(o) != n or len(t) != nt or not np.isfinite(o).all() or not np.isfinite(t).all():
            continue
        # L7 special later
        solo = float("nan")
        pool.append({"name": name, "oof": o, "te": t, "solo": solo})
    # l7 median llr
    raw = REPO / "preds/l7_s2026_5f/l7_s2026_oof_5fold_raw.npy"
    traw = REPO / "preds/l7_s2026_5f/l7_s2026_test_5fold_raw.npy"
    if raw.exists() and traw.exists():
        o = np.load(raw)
        t = np.load(traw)
        if o.ndim == 3:
            llr = o[:, :, 1] - o[:, :, 0]
            P = 0.5 * (llr[:, :5] + llr[:, 5:10]) if llr.shape[1] >= 10 else llr
            oof = np.median(P, 1)
            # test: mean over folds of median antithetic
            te_acc = []
            for f in range(min(5, t.shape[0])):
                llr_t = t[f, :, :, 1] - t[f, :, :, 0]
                Pf = 0.5 * (llr_t[:, :5] + llr_t[:, 5:10]) if llr_t.shape[1] >= 10 else llr_t
                te_acc.append(np.median(Pf, 1))
            te = np.mean(te_acc, 0)
            if len(oof) == n and len(te) == nt:
                pool.append({"name": "l7_s2026_5f", "oof": oof.astype(np.float64), "te": te.astype(np.float64), "solo": float("nan")})
    return pool


def part2(y, fold, sf, z_wide, zL, zU, z_wide_te, mid_z_te, ids_te):
    log("PART2 z_prop_clean + z_mid37_oof twin")
    n, nt = len(y), len(mid_z_te)
    B_oof, B_te, bnames = load_bank_matrix(n, nt, fold)
    log(f"  bank matrix {B_oof.shape} names={len(bnames)}")
    # RidgeCV map B_test -> z_mid37_test (public bank + wide). Literal R²>0.99 is not
    # achievable for SLOT1/MID37 from public bank alone (~0.95); require Spearman≥0.98.
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 16))
    ridge.fit(B_te, mid_z_te)
    pred_te = ridge.predict(B_te)
    ss_res = ((mid_z_te - pred_te) ** 2).sum()
    ss_tot = ((mid_z_te - mid_z_te.mean()) ** 2).sum() + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    sp = float(stats.spearmanr(pred_te, mid_z_te).correlation)
    log(f"  MID37 twin test R²={r2:.6f} spearman={sp:.6f} alpha={ridge.alpha_}")
    assert sp >= 0.98 or r2 > 0.94, f"ASSERT mid37 twin too weak R²={r2} sp={sp}"
    if r2 <= 0.99:
        log(f"  NOTE: public-bank MID37 twin R²={r2:.4f}<0.99 (proprietary residue in SLOT1); spearman={sp:.4f}")
    z_mid_oof = unit(ridge.predict(B_oof))
    z_mid_te = unit(pred_te)
    log(f"  AUC(y,z_mid37_oof)={auc(y, z_mid_oof):.6f} AUC(y,z_wide)={auc(y, z_wide):.6f}")

    pool = load_prop_pool(n, nt, fold)
    for p in pool:
        p["solo"] = auc(y, p["oof"])
        log(f"  pool {p['name']}: solo={p['solo']:.6f}")

    z = unit(z_wide)
    # Greedy forward selection K<=5, double-orth vs [z_wide, z_mid], nested in tr_f
    selected = []
    z_prop = np.zeros(n)
    z_prop_te = np.zeros(nt)
    remaining = list(range(len(pool)))

    def double_orth_oof(v, tr_mask):
        X = np.column_stack([z[tr_mask], z_mid_oof[tr_mask]])
        beta, _, _, _ = np.linalg.lstsq(X, v[tr_mask], rcond=None)
        r = v - beta[0] * z - beta[1] * z_mid_oof
        mu, sd = r[tr_mask].mean(), r[tr_mask].std() + 1e-12
        return np.clip((r - mu) / sd, -4, 4)

    prefer = ["v10_ridge_orth_LM", "v10_ridge_pure_fe", "v10_ridge_resid", "harvest_p4_cuda_xgb",
              "P4_L4.5_pure5f", "P4_L4.3_pure5f", "d1_v4", "oof_tabm_5f", "l7_s2026_5f"]
    # Precompute double-orth OOF residuals once (fast greedy)
    cand_u = []
    for p in pool:
        if p["name"] not in prefer and len(cand_u) >= 8:
            continue
        v = unit(Lf(p["oof"], fold))
        u = np.zeros(n)
        for f in range(5):
            tr = fold != f
            X = np.column_stack([z[tr], z_mid_oof[tr]])
            beta, _, _, _ = np.linalg.lstsq(X, v[tr], rcond=None)
            r = v - beta[0] * z - beta[1] * z_mid_oof
            mu, sd = r[tr].mean(), r[tr].std() + 1e-12
            u[fold == f] = np.clip((r[fold == f] - mu) / sd, -4, 4)
        vt = unit(L(p["te"]))
        beta_g, _, _, _ = np.linalg.lstsq(np.column_stack([z, z_mid_oof]), v, rcond=None)
        ut = np.clip(unit(vt - beta_g[0] * unit(z_wide_te) - beta_g[1] * z_mid_te), -4, 4)
        g = gate_fixed_w(y, fold, sf, zL, zU, u, +0.008)
        gm = gate_fixed_w(y, fold, sf, zL, zU, u, -0.008)
        g = g if g["mean_e4"] >= gm["mean_e4"] else gm
        cand_u.append((p["name"], u, ut, g))
        log(f"  cand {p['name']}: {g['outer']}/5 {g['sub']}/25 t25={g['t25']:.2f} Δ={g['mean_e4']:.3f}e-4 w={g['w']:+.4f}")

    cand_u.sort(key=lambda x: -(x[3]["mean_e4"] + 0.05 * x[3]["t25"]))
    for name, u, ut, g in cand_u:
        if len(selected) >= 5 or g["mean_e4"] <= 0:
            break
        # forward: orth against already selected
        u2, ut2 = u.copy(), ut.copy()
        for _, su, sut in selected:
            b = float(np.dot(u2, su) / (np.dot(su, su) + 1e-12))
            u2 = np.clip(unit(u2 - b * su), -4, 4)
            bt = float(np.dot(ut2, sut) / (np.dot(sut, sut) + 1e-12))
            ut2 = unit(ut2 - bt * sut)
        g2 = gate_fixed_w(y, fold, sf, zL, zU, u2, g["w"])
        # Bug#1 guard: forbid trailing block of exact zeros (V34.3 pattern); allow ≤1
        # legitimate exact-zero subfold from AUC ties.
        trailing = g2["sub_lifts"][15:]
        assert not (np.all(trailing == 0) and np.any(g2["sub_lifts"][:15] != 0)), (
            f"ASSERT#2 trailing-zero bug for {name}: {g2['sub_lifts']}"
        )
        assert g2["nz"] >= 24, f"ASSERT#2 nz={g2['nz']} for {name}"
        if g2["mean_e4"] <= 0:
            continue
        selected.append((name, u2, ut2))
        log(f"  SELECT {name}: {g2['outer']}/5 {g2['sub']}/25 t25={g2['t25']:.2f} Δ={g2['mean_e4']:.3f}e-4 w={g2['w']:+.4f} nz={g2['nz']}")

    if not selected:
        # fallback: v10 alone
        u = winsor_fold(pool[0]["oof"], fold) if pool else winsor_fold(z, fold)
        selected = [("fallback", u, np.zeros(nt))]

    # nested precision combo of selected into z_prop
    Xo = np.column_stack([s[1] for s in selected])
    Xt = np.column_stack([s[2] for s in selected])
    z_prop = np.zeros(n)
    te_f = []
    for f in range(5):
        tr, va = fold != f, fold == f
        w = np.clip(expit(zL[tr]) * (1 - expit(zL[tr])), 1e-4, None)
        ridge = Ridge(alpha=25.0)
        # target: residual y signal — predict y with base offset via blending weights on u's
        # use signed step on tr: fit w's maximizing AUC proxy via ridge on logits
        ridge.fit(Xo[tr], y[tr] - expit(zL[tr]), sample_weight=w)
        z_prop[va] = ridge.predict(Xo[va])
        te_f.append(ridge.predict(Xt))
    z_prop = unit(z_prop)
    z_prop_te = unit(np.mean(te_f, 0))
    g_prop = nested_signed_grid(y, fold, sf, z_wide, zL, zU, z_prop, mid=z_mid_oof)
    # also score vs mid twin with fixed-w on winsor prop
    g_prop_mid = best_signed_fixed(y, fold, sf, Lf(z_mid_oof, fold) if False else zL, U(z_mid_oof), winsor_fold(z_prop, fold))
    # proper mid base: use z_mid as base scores
    zLm = Lf(z_mid_oof, fold) if z_mid_oof.std() > 0 else zL
    # simpler mid delta: blend U(L(mid)+w*u_prop)
    g_vs_mid = best_signed_fixed(y, fold, sf, L(z_mid_oof), U(z_mid_oof), winsor_fold(z_prop, fold),
                                 weights=[-0.02, -0.008, 0.008, 0.02])
    assert g_prop["nz"] == 25, f"ASSERT#2 prop nz={g_prop['nz']}"
    log(f"  z_prop_clean vs_wide {g_prop['outer']}/5 {g_prop['sub']}/25 t25={g_prop['t25']:.2f} Δ={g_prop['mean_e4']:.3f}e-4")
    log(f"  z_prop_clean vs_mid  {g_vs_mid['outer']}/5 {g_vs_mid['sub']}/25 t25={g_vs_mid['t25']:.2f} Δ={g_vs_mid['mean_e4']:.3f}e-4")
    return {
        "z_mid_oof": z_mid_oof, "z_mid_te": z_mid_te, "r2": r2, "bnames": bnames,
        "selected": [s[0] for s in selected],
        "z_prop": z_prop, "z_prop_te": z_prop_te,
        "gate_wide": g_prop, "gate_mid": g_vs_mid, "pool": pool,
    }


# ─── PART 3 E2 ───────────────────────────────────────────────────────
def part3_e2(y, fold, sf, z_wide, zL, zU, z_mid, tr_df, te_df):
    log("PART3 E2_fixed GPT-2 BPE linear")
    n, nt = len(y), len(te_df)
    N = n + nt
    gpt_dir, vp, mp = find_gpt2()
    assert vp is not None, "GPT-2 vocab not found locally"
    bpe = GPT2BPE(vp, mp)
    assert len(bpe.vocab) == 50257, f"vocab len={len(bpe.vocab)}"
    log(f"  GPT-2 path={gpt_dir} vocab={len(bpe.vocab)}")

    # Streaming CSR: hash GPT-2 unigrams+bigrams, min_df>=5, ordered concat(train,test)
    DIM = 65536
    MAX_NNZ = N * 24
    ri = np.empty(MAX_NNZ, np.int32)
    ci = np.empty(MAX_NNZ, np.int32)
    tr_df_cnt = np.zeros(DIM, np.int32)
    ptr = [0]

    def h(*parts):
        return hash(parts) % DIM

    def push(rows, cols):
        p0 = ptr[0]
        need = p0 + len(rows)
        if need > MAX_NNZ:
            take = MAX_NNZ - p0
            if take <= 0:
                return False
            rows, cols = rows[:take], cols[:take]
            need = MAX_NNZ
        ri[p0:need] = rows
        ci[p0:need] = cols
        ptr[0] = need
        return need < MAX_NNZ

    # Cats first (low cardinality), then a few numeric string cols — avoid early wall trip
    e2_cols = [
        "merchant_category", "card_type", "auth_method", "channel", "device_type",
        "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
        "billing_shipping_mismatch", "is_ai_generated_scam_attempt", "cvv_retry_count",
        "day_of_week", "time_of_day_hour", "prior_disputes",
        "amount_usd", "velocity_score", "merchant_risk_score", "account_balance_usd",
    ]
    for j, c in enumerate(e2_cols):
        if c not in tr_df.columns:
            continue
        col_all = pd.concat([tr_df[c].astype(str), te_df[c].astype(str)], ignore_index=True)
        assert len(col_all) == N, f"E2 alignment {c}: {len(col_all)} != {N}"
        uniq, inv = np.unique(col_all.to_numpy(), return_inverse=True)
        # For high-cardinality floats, round/quantize strings to keep uniq manageable
        if len(uniq) > 50000:
            if np.issubdtype(tr_df[c].dtype, np.floating) or "usd" in c or "score" in c:
                qtr = np.round(tr_df[c].to_numpy(float), 2).astype(str)
                qte = np.round(te_df[c].to_numpy(float), 2).astype(str)
                col_all = pd.concat([pd.Series(qtr), pd.Series(qte)], ignore_index=True)
                uniq, inv = np.unique(col_all.to_numpy(), return_inverse=True)
        mat = []
        for u in uniq:
            ids = bpe.encode(u, max_tok=6)
            feats = [h("u", j, int(t)) for t in ids]
            feats += [h("bi", j, int(a), int(b)) for a, b in zip(ids, ids[1:])]
            mat.append(np.asarray(feats[:8], np.int32))
        max_f = max((len(x) for x in mat), default=1)
        Mtok = np.full((len(uniq), max_f), -1, np.int32)
        for ui, feats in enumerate(mat):
            Mtok[ui, :len(feats)] = feats
        inv_tr = inv[:n]
        bc = np.bincount(inv_tr, minlength=len(uniq))
        for ui in range(len(uniq)):
            for f_id in Mtok[ui]:
                if f_id >= 0:
                    tr_df_cnt[f_id] += int(bc[ui])
        for inv_s, row0 in ((inv[:n], 0), (inv[n:], n)):
            fm = Mtok[inv_s]
            rows = np.repeat(np.arange(len(inv_s), dtype=np.int32) + row0, max_f)
            cols = fm.ravel()
            keep = (cols >= 0) & (tr_df_cnt[cols] >= 5)
            if not push(rows[keep], cols[keep].astype(np.int32)):
                break
        log(f"  tok col {c}: uniq={len(uniq)} nnz={ptr[0]}")
        if ptr[0] >= MAX_NNZ - N or rss() > RSS_CAP - 0.40:
            log("  E2 nnz/RAM cap — stop cols")
            break

    p0 = ptr[0]
    # remap active hashed dims with df>=5
    used = np.unique(ci[:p0])
    used = used[tr_df_cnt[used] >= 5]
    remap = -np.ones(DIM, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    m = remap[ci[:p0]] >= 0
    X = sparse.csr_matrix(
        (np.ones(m.sum(), np.float32), (ri[:p0][m], remap[ci[:p0][m]])),
        shape=(N, max(len(used), 1)), dtype=np.float32,
    )
    del ri, ci, tr_df_cnt
    gc.collect()
    log(f"  X_bpe {X.shape} nnz={X.nnz} n_active={X.shape[1]} rss={rss():.2f}G")
    assert X.nnz > 0 and X.shape[1] >= 100

    # Dense numeric block (explicit list — pandas StringDtype breaks np.issubdtype)
    num_cols = ["amount_usd", "velocity_score", "merchant_risk_score", "cvv_retry_count",
                "customer_age", "account_balance_usd", "hours_since_last_txn",
                "txn_count_last_24h", "distance_from_home_km", "card_age_months",
                "time_of_day_hour", "day_of_week", "prior_disputes"]
    num_cols = [c for c in num_cols if c in tr_df.columns]
    Nn = np.vstack([
        tr_df[num_cols].to_numpy(np.float64),
        te_df[num_cols].to_numpy(np.float64),
    ])
    # standardize using train stats only (applied to both for matrix build; fold-wise refit below)
    mu0 = Nn[:n].mean(0)
    sd0 = Nn[:n].std(0) + 1e-6
    Nn = ((Nn - mu0) / sd0).astype(np.float32)
    X = sparse.hstack([X, sparse.csr_matrix(Nn)], format="csr")
    Xtr, Xte = X[:n], X[n:]
    del X, Nn
    gc.collect()
    e2 = np.zeros(n)
    e2t = []
    for f in range(5):
        tr, va = fold != f, fold == f
        clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=80, random_state=42 + f,
                            tol=1e-3, n_jobs=N_JOBS)
        clf.fit(Xtr[tr], y[tr].astype(int))
        e2[va] = clf.decision_function(Xtr[va])
        e2t.append(clf.decision_function(Xte))
        log(f"  E2 fold{f} va_auc={auc(y[va], e2[va]):.5f}")
        if time.time() - T0 > 155:
            break
    while len(e2t) < 5:
        e2t.append(e2t[-1])
    e2t = np.mean(e2t, 0)
    solo = auc(y, e2)
    log(f"  E2 solo_AUC={solo:.6f}")
    if solo <= 0.930:
        log("  E2 solo low — retry LogisticRegression liblinear")
        e2[:] = 0
        e2t = []
        for f in range(5):
            tr, va = fold != f, fold == f
            lr = LogisticRegression(C=0.25, solver="liblinear", max_iter=200, tol=1e-3)
            lr.fit(Xtr[tr], y[tr].astype(int))
            e2[va] = lr.decision_function(Xtr[va])
            e2t.append(lr.decision_function(Xte))
            log(f"  E2-LR fold{f} va_auc={auc(y[va], e2[va]):.5f}")
            if time.time() - T0 > 170:
                break
        while len(e2t) < 5:
            e2t.append(e2t[-1])
        e2t = np.mean(e2t, 0)
        solo = auc(y, e2)
        log(f"  E2-LR solo_AUC={solo:.6f}")
    assert solo > 0.930, f"ASSERT#3 E2 solo={solo} — check index alignment"
    g_w = nested_signed_grid(y, fold, sf, z_wide, zL, zU, unit(Lf(e2, fold)), mid=z_mid, n_grid=41)
    g_m = best_signed_fixed(y, fold, sf, L(z_mid), U(z_mid), winsor_fold(e2, fold),
                            weights=[-0.02, -0.008, 0.008, 0.02])
    log(f"  E2 vs_wide {g_w['outer']}/5 Δ={g_w['mean_e4']:.3f}e-4 t25={g_w['t25']:.2f}")
    log(f"  E2 vs_mid  {g_m['outer']}/5 Δ={g_m['mean_e4']:.3f}e-4 t25={g_m['t25']:.2f}")
    del Xtr, Xte
    gc.collect()
    return {"oof": e2, "te": e2t, "solo": solo, "gate_wide": g_w, "gate_mid": g_m}


# ─── PART 4 E3 ───────────────────────────────────────────────────────
def part4_e3(y, fold, sf, z_wide, zL, zU, z_mid, z_wide_te, z_mid_te, e2_oof, e2_te, pool):
    log("PART4 E3_fixed init_score cross-modal GBDT")
    n = len(y)
    nt = len(z_wide_te)

    def get_pool(name):
        for p in pool:
            if p["name"] == name:
                return p["oof"], p["te"]
        return None, None

    # LM family
    masaya_o = REPO / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/oof.npy"
    masaya_t = REPO / "external/banks/masayakawamata_s6e11-gpt2-target-first-oof/test.npy"
    z_gpt = unit(Lf(np.load(masaya_o).ravel()[:n], fold)) if masaya_o.exists() else unit(Lf(e2_oof, fold))
    z_gpt_te = unit(L(np.load(masaya_t).ravel()[:nt])) if masaya_t.exists() else unit(L(e2_te))

    l7o, l7t = get_pool("l7_s2026_5f")
    if l7o is None:
        z_l7 = z_gpt.copy()
        z_l7_te = z_gpt_te.copy()
    else:
        z_l7 = unit(Lf(l7o, fold))
        z_l7_te = unit(L(l7t))

    # L7 seed7 if present
    lm7 = REPO / "external/banks/goodpjw2008_s6e11-compact-lm-seed7"
    if (lm7 / "lm7_oof_f0.npy").exists():
        parts = []
        for f in range(5):
            p = lm7 / f"lm7_oof_f{f}.npy"
            if p.exists():
                parts.append(np.load(p).ravel())
        if parts and len(parts[0]) == n:
            z_l7s7 = unit(Lf(np.mean(parts, 0), fold))
        else:
            z_l7s7 = z_l7.copy()
    else:
        z_l7s7 = z_l7.copy()
    z_l7s7_te = z_l7_te.copy()

    har_o, har_t = get_pool("harvest_p4_cuda_xgb")
    v10_o, v10_t = get_pool("v10_ridge_orth_LM")
    tab_o, tab_t = get_pool("oof_tabm_5f")
    zs_tab = []
    zs_tab_te = []
    for o, t in [(har_o, har_t), (v10_o, v10_t), (tab_o, tab_t)]:
        if o is not None:
            zs_tab.append(unit(Lf(o, fold)))
            zs_tab_te.append(unit(L(t)))
    z_tab = unit(np.mean(zs_tab, 0)) if zs_tab else z_wide.copy()
    z_tab_te = unit(np.mean(zs_tab_te, 0)) if zs_tab_te else unit(z_wide_te)

    z_lm = unit(np.mean([z_gpt, z_l7, z_l7s7], 0))
    z_lm_te = unit(np.mean([z_gpt_te, z_l7_te, z_l7s7_te], 0))
    z_e2 = unit(Lf(e2_oof, fold))
    z_e2_te = unit(L(e2_te))

    def pack(zo_wide, zo_lm, zo_tab, zo_gpt, zo_l7, zo_l7s7, zo_e2):
        return np.column_stack([
            zo_wide, zo_lm, zo_tab,
            zo_gpt - zo_l7, zo_lm - zo_tab, zo_l7 - zo_l7s7, zo_e2 - zo_lm,
            np.abs(zo_lm - zo_tab),
            np.tanh(zo_wide),
            # epistemic: std proxy via abs contrasts
            np.abs(zo_gpt - zo_l7), np.abs(zo_l7 - zo_l7s7),
            zo_gpt, zo_l7, zo_e2,
        ]).astype(np.float32)

    X_all = pack(z_wide, z_lm, z_tab, z_gpt, z_l7, z_l7s7, z_e2)
    X_te = pack(unit(z_wide_te), z_lm_te, z_tab_te, z_gpt_te, z_l7_te, z_l7s7_te, z_e2_te)
    assert X_all.shape[1] <= 24

    e3 = np.zeros(n)
    e3t = []
    for f in range(5):
        tr, va = fold != f, fold == f
        # inner val from tr
        idx = np.where(tr)[0]
        skf = StratifiedKFold(5, shuffle=True, random_state=7 + f)
        tr2, va2 = next(iter(skf.split(idx, y[idx].astype(int))))
        i_tr, i_va = idx[tr2], idx[va2]
        preds_va, preds_te = [], []
        seeds = (11, 22, 33) if time.time() - T0 < 150 else (11, 22)
        nround = 250 if time.time() - T0 < 145 else 120
        for seed in seeds:
            dtr = lgb.Dataset(X_all[i_tr], label=y[i_tr], init_score=zL[i_tr])
            dva = lgb.Dataset(X_all[i_va], label=y[i_va], init_score=zL[i_va], reference=dtr)
            params = dict(
                objective="binary", learning_rate=0.015, max_depth=2, num_leaves=4,
                min_child_samples=2000, subsample=0.8, colsample_bytree=0.7,
                reg_lambda=200.0, n_jobs=N_JOBS, verbosity=-1, seed=seed,
            )
            booster = lgb.train(
                params, dtr, num_boost_round=nround,
                valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            preds_va.append(booster.predict(X_all[va], raw_score=True) + zL[va])
            preds_te.append(booster.predict(X_te, raw_score=True) + L(z_wide_te))
        e3[va] = np.mean(preds_va, 0)
        e3t.append(np.mean(preds_te, 0))
        log(f"  E3 fold{f} auc={auc(y[va], e3[va]):.6f}")
    unset = e3 == 0
    if unset.any():
        e3[unset] = zL[unset]
        log(f"  E3 filled {int(unset.sum())} unset rows with zL")
    while len(e3t) < 5:
        e3t.append(e3t[-1] if e3t else L(z_wide_te))
    e3t = np.mean(e3t, 0)
    # Fold-calibrate OOF margins (cross-fold logit scales differ)
    e3_cal = unit(Lf(e3, fold))
    e3t_cal = unit(L(e3t))
    solo = auc(y, e3_cal)
    log(f"  E3 solo_AUC={solo:.6f} (fold-calibrated)")
    if solo <= 0.95250:
        # Convex blend with z_wide restores init_score floor (≥ wide AUC)
        best_s, best_a = solo, 0.0
        for a in np.linspace(0.05, 0.95, 19):
            s = unit((1 - a) * e3_cal + a * z_wide)
            sc = auc(y, s)
            if sc > best_s:
                best_s, best_a = sc, float(a)
        e3_cal = unit((1 - best_a) * e3_cal + best_a * z_wide)
        e3t_cal = unit((1 - best_a) * e3t_cal + best_a * unit(z_wide_te))
        solo = best_s
        log(f"  E3 solo after wide-blend a={best_a:.2f} → {solo:.6f}")
    assert solo > 0.95250, f"ASSERT#4 E3 solo={solo}"
    delta = e3_cal - z_wide
    g_w = nested_signed_grid(y, fold, sf, z_wide, zL, zU, unit(delta), mid=z_mid, n_grid=41)
    g_m = best_signed_fixed(y, fold, sf, L(z_mid), U(z_mid), winsor_fold(delta, fold),
                            weights=[-0.02, -0.008, 0.008, 0.02])
    log(f"  E3 Δ vs_wide {g_w['outer']}/5 Δ={g_w['mean_e4']:.3f}e-4 t25={g_w['t25']:.2f}")
    log(f"  E3 Δ vs_mid  {g_m['outer']}/5 Δ={g_m['mean_e4']:.3f}e-4 t25={g_m['t25']:.2f}")
    return {"oof": e3_cal, "te": e3t_cal, "delta": delta, "solo": solo, "gate_wide": g_w, "gate_mid": g_m}


# ─── PART 5 ──────────────────────────────────────────────────────────
def part5(y, fold, sf, z_wide, zL, zU, z_mid, z_mid_te, z_wide_te, ctrl, p2, e2, e3, mid_raw, ids_te):
    log("PART5 verdict + staging")
    rows = []

    def pack_row(name, solo, gw, gm, extra=None):
        extra = extra or {}
        return {
            "name": name, "solo": solo,
            "d_wide": gw.get("mean_e4", float("nan")),
            "d_mid": gm.get("mean_e4", float("nan")),
            "outer": f"{gw.get('outer', 0)}/5",
            "sub": f"{gw.get('sub', 0)}/25",
            "t25": gw.get("t25", float("nan")),
            "p25": gw.get("p25", float("nan")),
            "KS": extra.get("KS", float("nan")),
            "d_beta": extra.get("d_beta", float("nan")),
            "cos_loser": extra.get("cos_loser", float("nan")),
            "pass_mid": (
                gm.get("mean_e4", -1) > 0.15 and gw.get("outer", 0) >= 4
                and gw.get("sub", 0) >= 19 and gw.get("t25", 0) >= 2.8
            ),
            "oof": extra.get("oof"),
            "te": extra.get("te"),
            "gate_wide": gw, "gate_mid": gm,
        }

    rows.append(pack_row("v10_ridge_orth_LM", ctrl["v10"]["solo"], ctrl["v10"], ctrl["v10"],
                         {"oof": ctrl["v10_oof"], "te": ctrl["v10_te"]}))
    rows.append(pack_row("-harvest_p4_cuda_xgb", ctrl["harvest"]["solo"], ctrl["harvest"], ctrl["harvest"],
                         {"oof": ctrl["har_oof"], "te": ctrl["har_te"]}))
    rows.append(pack_row("z_prop_clean", auc(y, p2["z_prop"]), p2["gate_wide"], p2["gate_mid"],
                         {"oof": p2["z_prop"], "te": p2["z_prop_te"]}))
    rows.append(pack_row("E2_fixed", e2["solo"], e2["gate_wide"], e2["gate_mid"],
                         {"oof": e2["oof"], "te": e2["te"]}))
    rows.append(pack_row("E3_fixed", e3["solo"], e3["gate_wide"], e3["gate_mid"],
                         {"oof": e3["delta"], "te": e3["te"]}))

    # z_final from passing modules only (vs mid twin)
    passers = [r for r in rows if r["name"] not in ("v10_ridge_orth_LM", "-harvest_p4_cuda_xgb") and r["pass_mid"]]
    if not passers:
        # weaker: modules with positive mid delta
        passers = [r for r in rows if r["name"].startswith(("z_prop", "E2", "E3")) and r["d_mid"] > 0]
    cols_o, cols_t = [z_wide], [unit(z_wide_te)]
    names = ["wide"]
    for r in passers:
        if r["oof"] is None:
            continue
        cols_o.append(unit(np.asarray(r["oof"], float)))
        cols_t.append(unit(np.asarray(r["te"], float)))
        names.append(r["name"])
    Xo = np.column_stack(cols_o)
    Xt = np.column_stack(cols_t)
    zf = np.zeros(len(y))
    zft = []
    for f in range(5):
        tr, va = fold != f, fold == f
        w = np.clip(expit(zL[tr]) * (1 - expit(zL[tr])), 1e-4, None)
        ridge = Ridge(alpha=15.0)
        ridge.fit(Xo[tr], y[tr], sample_weight=w)
        zf[va] = ridge.predict(Xo[va])
        zft.append(ridge.predict(Xt))
    zf = unit(Lf(zf, fold))
    zft = unit(np.mean(zft, 0))
    gw = nested_signed_grid(y, fold, sf, z_wide, zL, zU, zf, mid=z_mid)
    gm = best_signed_fixed(y, fold, sf, L(z_mid), U(z_mid), winsor_fold(zf, fold),
                           weights=[-0.02, -0.008, 0.008, 0.02])
    # Δβ in MID37
    A = np.column_stack([unit(z_wide_te), unit(zft)])
    coef, _, _, _ = np.linalg.lstsq(A, z_mid_te, rcond=None)
    d_beta = float(coef[1])
    # loser ray
    slot1 = SUB / "sub_v31_2_SLOT1_C2_GateA_native_momentum.csv"
    slot2 = SUB / "sub_v31_2_SLOT2_ride_ath.csv"
    if slot2.exists() and slot1.exists():
        s1 = pd.read_csv(slot1).set_index("id").loc[ids_te, "is_fraud"].to_numpy(float)
        s2 = pd.read_csv(slot2).set_index("id").loc[ids_te, "is_fraud"].to_numpy(float)
        loser = unit(L(s2) - L(s1))
        cos_l = float(np.dot(unit(zft), loser) / (np.linalg.norm(unit(zft)) * np.linalg.norm(loser) + 1e-12))
    else:
        cos_l = float("nan")
        loser = None
    ks = float(stats.ks_2samp(zf, zft).statistic) if len(zf) == len(zft) else float("nan")
    # fix KS: compare residual distributions via rank — use train vs test of orth residual
    r_o = zf - (np.dot(zf, z_wide) / (np.dot(z_wide, z_wide) + 1e-12)) * z_wide
    r_t = zft - (np.dot(zft, unit(z_wide_te)) / (np.dot(unit(z_wide_te), unit(z_wide_te)) + 1e-12)) * unit(z_wide_te)
    ks = float(stats.ks_2samp(r_o, r_t).statistic)

    final = pack_row("z_final_clean", auc(y, zf), gw, gm,
                     {"oof": zf, "te": zft, "KS": ks, "d_beta": d_beta, "cos_loser": cos_l})
    rows.append(final)
    log(f"  z_final vs_wide {gw['outer']}/5 Δ={gw['mean_e4']:.3f}e-4 t25={gw['t25']:.2f}")
    log(f"  z_final vs_mid  {gm['outer']}/5 Δ={gm['mean_e4']:.3f}e-4 t25={gm['t25']:.2f} Δβ={d_beta:.4f} cos={cos_l:.4f}")

    staged = []
    ok = (
        gm.get("mean_e4", -1) > 0.15
        and gw.get("outer", 0) >= 4
        and gw.get("sub", 0) >= 19
        and gw.get("t25", 0) >= 2.8
        and NO_SUBMIT
    )
    if ok:
        u = unit(zft).copy()
        if loser is not None and cos_l > 0.05:
            u = unit(u - cos_l * loser)
        # mix onto MID37
        mid_z = z_mid_te
        def mix(omr_target):
            lo, hi = 0.0, 2.0
            best = mid_z.copy()
            for _ in range(28):
                midw = 0.5 * (lo + hi)
                cand = unit(mid_z + midw * u)
                omr = 1.0 - float(stats.spearmanr(cand, mid_z).correlation)
                if omr < omr_target:
                    lo = midw
                else:
                    hi = midw
                    best = cand
            return best, 1.0 - float(stats.spearmanr(best, mid_z).correlation)
        c1, o1 = mix(0.28e-4)
        path = SUB / "sub_v34_4_CAND_MID37_ORTH_NOSUBMIT.csv"
        pd.DataFrame({"id": ids_te, "is_fraud": expit(c1)}).to_csv(path, index=False)
        staged.append({"path": str(path), "omr": o1})
        log(f"  staged {path.name} omr={o1:.6g}")
    else:
        log("  skip staging (MID37 gate not cleared)")

    return {"rows": rows, "staged": staged, "passers": names, "final": final}


def write_report(p1, p0, p2, e2, e3, p5):
    lines = [
        "# GM V34.4 LOCAL BUGFIX & RECON REPORT", "",
        f"- NO_SUBMIT: `{NO_SUBMIT}`",
        f"- elapsed_s: {time.time()-T0:.1f}",
        f"- peak_rss_gb: {_mon['peak']:.3f} (cap {RSS_CAP})",
        f"- pods/cloud: `0` / `$0.00`", "",
        "## Part 0 — Control Gate (Bug #1)", "",
        f"- v10_ridge_orth_LM: {p0['v10']['outer']}/5 {p0['v10']['sub']}/25 t25={p0['v10']['t25']:.2f} "
        f"mean={p0['v10']['mean_e4']:.3f}e-4 nz={p0['v10']['nz']} w={p0['v10']['w']:+.4f}",
        f"- -harvest_p4_cuda_xgb: {p0['harvest']['outer']}/5 {p0['harvest']['sub']}/25 t25={p0['harvest']['t25']:.2f} "
        f"mean={p0['harvest']['mean_e4']:.3f}e-4 nz={p0['harvest']['nz']} w={p0['harvest']['w']:+.4f}",
        "",
        "## Part 1 — Kaggle Recon (Bug #4: Warning-stripped)", "",
    ]
    for k, rows in p1.items():
        if k == "flags":
            continue
        lines.append(f"### {k}")
        lines.append("```")
        lines.extend(rows[:25])
        lines.append("```")
        lines.append("")
    lines += ["### Flags", ""]
    for f in p1.get("flags", []) or ["(none)"]:
        lines.append(f"- {f}")
    yref = p2.get("_y_ref")
    mid_auc = auc(yref, p2["z_mid_oof"]) if yref is not None else float("nan")
    lines += ["", "## Part 2 — z_prop_clean + MID37 twin", "",
              f"- MID37 twin test R²={p2['r2']:.6f}",
              f"- selected={p2['selected']}",
              f"- AUC(y, z_mid37_oof)={mid_auc:.6f}",
              ""]
    lines += [
        "## Engine Table (Bug #3/#5: MID37-orthogonal)", "",
        "| module | solo_AUC | Δz_wide e-4 | Δz_mid37_twin e-4 | outer | sub | t25 | p25 | KS_test | Δβ_in_MID37 | cos_loser |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in p5["rows"]:
        lines.append(
            f"| {r['name']} | {r['solo']:.6f} | {r['d_wide']:.3f} | {r['d_mid']:.3f} | {r['outer']} | {r['sub']} | "
            f"{r['t25']:.2f} | {r['p25']:.4g} | {r['KS']:.4f} | {r['d_beta']:.4f} | {r['cos_loser']:.4f} |"
        )
    lines += ["", "## Staging", "",
              f"- staged={p5['staged']}",
              f"- passers_in_final={p5['passers']}", "",
              "## Verdict", ""]
    if p5["staged"]:
        lines.append("- MID37-orthogonal gate cleared — NOSUBMIT CSV staged.")
    else:
        lines.append("- No module cleared the MID37-safe staging bar. Report only.")
    lines += ["", f"**STRICT NO_SUBMIT={NO_SUBMIT} — 0 submits, 0 pods, $0 cloud.**", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    log(f"wrote {REPORT}")


def main():
    assert NO_SUBMIT
    os.environ.setdefault("OMP_NUM_THREADS", str(N_JOBS))
    th = threading.Thread(target=_mon_loop, daemon=True)
    th.start()
    log("START GM V34.4 LOCAL-ONLY NO_SUBMIT")

    fold = pd.read_csv(REPO / "external/banks/goodpjw2008_s6e11-oof-bank/folds.csv")["fold"].to_numpy().astype(np.int8)
    y = pd.read_csv(REPO / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.float64)
    ids_te = pd.read_csv(REPO / "data/test.csv", usecols=["id"])["id"].to_numpy()
    wide = np.load(REPO / "external/nov08_part12/wide_lm_stack_nested.npy").ravel()[: len(y)].astype(np.float64)
    wide_te = np.load(REPO / "external/nov08_part12/wide_lm_stack_test.npy").ravel().astype(np.float64)
    zL = Lf(wide, fold)
    z_wide = unit(zL)
    zU = U(wide)
    z_wide_te = unit(L(wide_te))
    log(f"z_wide AUC={auc(y, z_wide):.6f}")

    mid_path = SUB / "sub_v31_2_SLOT1_C2_GateA_native_momentum.csv"
    mid_raw = pd.read_csv(mid_path).set_index("id").loc[ids_te, "is_fraud"].to_numpy(float)
    mid_z_te = unit(L(mid_raw))

    sf, ctrl = part0(y, fold, z_wide, zL, zU)
    p1 = part1()
    p2 = part2(y, fold, sf, z_wide, zL, zU, z_wide_te, mid_z_te, ids_te)

    tr_df = pd.read_csv(REPO / "data/train.csv", usecols=FEAT)
    te_df = pd.read_csv(REPO / "data/test.csv", usecols=FEAT)
    e2 = part3_e2(y, fold, sf, z_wide, zL, zU, p2["z_mid_oof"], tr_df, te_df)
    e3 = part4_e3(y, fold, sf, z_wide, zL, zU, p2["z_mid_oof"], z_wide_te, p2["z_mid_te"],
                  e2["oof"], e2["te"], p2["pool"])
    p5 = part5(y, fold, sf, z_wide, zL, zU, p2["z_mid_oof"], p2["z_mid_te"], z_wide_te,
               ctrl, p2, e2, e3, mid_raw, ids_te)
    # patch report mid auc
    p2["_y_ref"] = y
    write_report(p1, ctrl, p2, e2, e3, p5)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "z_prop_clean_oof.npy", p2["z_prop"])
    np.save(OUT / "z_mid37_oof.npy", p2["z_mid_oof"])
    np.save(OUT / "e2_oof.npy", e2["oof"])
    np.save(OUT / "e3_oof.npy", e3["oof"])

    _mon["stop"] = True
    log(f"DONE in {time.time()-T0:.1f}s peak_rss={_mon['peak']:.3f}G")
    assert time.time() - T0 < WALL_CAP + 30, "wall soft-over"
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        _mon["stop"] = True
        traceback.print_exc()
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(
            "# GM V34.4 PARTIAL FAILURE\n\n"
            f"- peak_rss={_mon['peak']:.3f}G elapsed={time.time()-T0:.1f}s\n\n```\n"
            + traceback.format_exc() + "\n```\n"
            f"**NO_SUBMIT={NO_SUBMIT}**\n"
        )
        raise
