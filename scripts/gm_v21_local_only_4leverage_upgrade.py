#!/usr/bin/env python3
"""GM V21 — 100% local Mac 4-leverage upgrade vs ATH 0.95399.

ZERO pods / ZERO remote GPU. Local CPU only (float32, n_jobs=4).
Reuses V20 Slot8 rebuild + two-stage orth / 6-gate machinery.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

N_JOBS = 4
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_JOBS)
os.environ["CATBOOST_THREAD_COUNT"] = str(N_JOBS)

REPO = Path(__file__).resolve().parents[1]
TMP15, TMP16 = Path("/tmp/s6e11_v15"), Path("/tmp/s6e11_v16")
OUT, SUB, DATA, LOG = TMP16 / "out", REPO / "submissions", REPO / "data", REPO / "logs/overnight"
PRED = REPO / "preds"
MED, L7DIR, HARV, MV2, V10 = (
    PRED / "gpt2_med_4ep",
    PRED / "l7_s2026_5f",
    PRED / "fast_gpu_harvest",
    PRED / "matrix_v2",
    PRED / "v10_transductive_bulsah",
)
PB = PRED / "paradigm_B"
REV = PRED / "gpt2_rev"
G3 = PRED / "gpt2_3ep"
SLOT9 = SUB / "sub_v20_LIVE_slot9_overnight_cpu_promoted.csv"
REPORT = LOG / "GM_V21_LOCAL_4LEVERAGE_REPORT.md"
NPZ_OUT = OUT / "v21_local_4leverages.npz"
MON_LOG = LOG / "GM_V21_LOCAL_4LEVERAGE_MONITOR.log"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
N_TR, N_TE = 699960, 299984
T0 = time.time()
LINES: list[str] = []


def log(msg: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB on macOS? actually bytes on mac → /1e6 = MB-ish; on Linux KB
    # macOS ru_maxrss is bytes
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    s = f"[{time.time()-T0:7.1f}s|rss={rss_mb:.0f}MB] {msg}"
    print(s, flush=True)
    LINES.append(s)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def load_v20():
    path = REPO / "scripts" / "gm_v20_overnight_cpu_grandmaster_factory.py"
    spec = importlib.util.spec_from_file_location("gm_v20_fac", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def start_monitor(pid: int) -> subprocess.Popen:
    MON_LOG.parent.mkdir(parents=True, exist_ok=True)
    script = f"""
import time, os, subprocess
pid={pid}
log={str(MON_LOG)!r}
while True:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        open(log,'a').write(time.strftime('%Y-%m-%dT%H:%M:%SZ')+' TARGET_EXIT\\n'); break
    try:
        ps=subprocess.check_output(['ps','-p',str(pid),'-o','pcpu=,rss='], text=True).split()
        pcpu, rss_kb = float(ps[0]), int(ps[1])
    except Exception:
        pcpu, rss_kb = -1, -1
    vm=subprocess.check_output(['vm_stat'], text=True)
    page=16384
    free=inact=wired=comp=0
    for line in vm.splitlines():
        if 'Pages free' in line: free=int(line.split(':')[1].strip().rstrip('.'))*page
        if 'Pages inactive' in line: inact=int(line.split(':')[1].strip().rstrip('.'))*page
        if 'Pages wired' in line: wired=int(line.split(':')[1].strip().rstrip('.'))*page
        if 'Pages occupied by compressor' in line: comp=int(line.split(':')[1].strip().rstrip('.'))*page
    load=open('/proc/loadavg').read() if os.path.exists('/proc/loadavg') else subprocess.check_output(['sysctl','-n','vm.loadavg'], text=True).strip()
    open(log,'a').write(
        f"{{time.strftime('%Y-%m-%dT%H:%M:%SZ')}} pid={{pid}} pcpu={{pcpu:.1f}} rss_mb={{rss_kb/1024:.0f}} "
        f"free_mb={{free/1e6:.0f}} inact_mb={{inact/1e6:.0f}} wired_mb={{wired/1e6:.0f}} comp_mb={{comp/1e6:.0f}} load={{load}}\\n"
    )
    time.sleep(30)
"""
    return subprocess.Popen([sys.executable, "-c", script], start_new_session=True)


def f32(a):
    return np.asarray(a, dtype=np.float32)


def unitize_f32(v):
    v = f32(v).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def llr_mean_std(arr):
    """arr (n, k, 2) → mean_llr, std_llr, prec, ll_pos_mean, ll_neg_mean, corr_pn"""
    pos = arr[:, :, 1].astype(np.float32)
    neg = arr[:, :, 0].astype(np.float32)
    llr = pos - neg
    z = llr.mean(1)
    s = llr.std(1) + 1e-6
    prec = z / (0.10 + s)
    # per-row corr across orders (approx via standardized)
    zp = (pos - pos.mean(1, keepdims=True)) / (pos.std(1, keepdims=True) + 1e-6)
    zn = (neg - neg.mean(1, keepdims=True)) / (neg.std(1, keepdims=True) + 1e-6)
    corr = (zp * zn).mean(1)
    return z, s, prec, pos.mean(1), neg.mean(1), corr


def pchip_align_test(raw_t, ref_oof_like):
    """Quantile-align continuous test logits to OOF rank-gauss scale (ECDF/PCHIP-style)."""
    raw_t = f32(raw_t).ravel()
    ref = f32(ref_oof_like).ravel()
    out = np.empty_like(raw_t)
    out[np.argsort(raw_t)] = np.sort(ref[: len(raw_t)] if len(ref) == len(raw_t) else np.sort(ref)[
        (np.linspace(0, len(ref) - 1, len(raw_t))).astype(np.int64)
    ])
    return out


def build_cat_keys(train: pd.DataFrame, test: pd.DataFrame):
    n_tr, n_te = len(train), len(test)
    cols = {}

    def put(name, trv, tev):
        cols[name] = (np.asarray(trv), np.asarray(tev))

    for c in ("merchant_category", "channel", "device_type", "card_type", "auth_method"):
        codes, _ = pd.factorize(pd.concat([train[c].astype(str), test[c].astype(str)], ignore_index=True))
        put(c, codes[:n_tr], codes[n_tr:])

    flags = [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    ftr = np.zeros(n_tr, dtype=np.int32)
    fte = np.zeros(n_te, dtype=np.int32)
    for i, c in enumerate(flags):
        ftr |= (train[c].astype(int).to_numpy() & 1) << i
        fte |= (test[c].astype(int).to_numpy() & 1) << i
    put("flag_bitmask_64", ftr, fte)

    ch = cols["channel"][0]
    dv = cols["device_type"][0]
    cht, dvt = cols["channel"][1], cols["device_type"][1]
    put("ch_x_dev_code", ch * 64 + dv, cht * 64 + dvt)

    def cents(series):
        return (np.round(series.to_numpy(np.float64) * 100).astype(np.int64) % 100).astype(np.int32)

    def ones(series):
        return (np.abs(series.to_numpy(np.float64)).astype(np.int64) % 10).astype(np.int32)

    def tenths(series):
        return (np.round(series.to_numpy(np.float64) * 10).astype(np.int64) % 10).astype(np.int32)

    put("amt_cents", cents(train["amount_usd"]), cents(test["amount_usd"]))
    put("bal_cents", cents(train["account_balance_usd"]), cents(test["account_balance_usd"]))
    put("amt_ones", ones(train["amount_usd"]), ones(test["amount_usd"]))
    put("bal_ones", ones(train["account_balance_usd"]), ones(test["account_balance_usd"]))
    put("amt_tenths", tenths(train["amount_usd"]), tenths(test["amount_usd"]))
    put("bal_tenths", tenths(train["account_balance_usd"]), tenths(test["account_balance_usd"]))
    put("mrisk_cents", cents(train["merchant_risk_score"]), cents(test["merchant_risk_score"]))
    put("vel_cents", cents(train["velocity_score"]), cents(test["velocity_score"]))
    put("cvv_retry_count", train["cvv_retry_count"].astype(np.int32), test["cvv_retry_count"].astype(np.int32))
    put(
        "card_age_mod12",
        (train["card_age_months"].astype(np.int32) % 12),
        (test["card_age_months"].astype(np.int32) % 12),
    )
    put(
        "log2_amt",
        np.floor(np.log2(train["amount_usd"].to_numpy(np.float64) + 1)).astype(np.int32),
        np.floor(np.log2(test["amount_usd"].to_numpy(np.float64) + 1)).astype(np.int32),
    )
    put(
        "log2_bal",
        np.floor(np.log2(train["account_balance_usd"].to_numpy(np.float64) + 1)).astype(np.int32),
        np.floor(np.log2(test["account_balance_usd"].to_numpy(np.float64) + 1)).astype(np.int32),
    )

    # orig20k exact hit if available
    hit_tr = np.zeros(n_tr, dtype=np.int32)
    hit_te = np.zeros(n_te, dtype=np.int32)
    for cand in (
        HARV / "oof_orig20k_knn.npy",
        OUT / "v21_orig_exact_hit_oof.npy",
    ):
        if cand.exists():
            # presence proxy: nonzero knn score as soft hit; binarize at median for flag
            o = np.load(cand).ravel()[:n_tr]
            hit_tr = (o > np.median(o)).astype(np.int32)
            tp = HARV / "test_orig20k_knn.npy"
            if not tp.exists():
                tp = HARV / "orig20k_knn_test.npy"
            if tp.exists():
                t = np.load(tp).ravel()[:n_te]
                hit_te = (t > np.median(t)).astype(np.int32)
            break
    put("orig20k_exact_hit_flag", hit_tr, hit_te)

    names = list(cols.keys())
    Xtr = np.column_stack([cols[n][0] for n in names]).astype(np.int32)
    Xte = np.column_stack([cols[n][1] for n in names]).astype(np.int32)
    return names, Xtr, Xte


class Local4Leverage:
    def __init__(self, v20):
        self.v20 = v20
        self.fac = v20.OvernightFactory()
        self.results = {}
        self.promoted = []

    def rebuild(self):
        log("=== Step1: rebuild ATH anchors (Slot8 stack + Slot9 live) ===")
        v20 = self.v20
        self.fac.rebuild()
        f = self.fac
        # LIVE ATH Slot9
        live9 = pd.read_csv(SLOT9)
        f.slot9 = live9.set_index("id").loc[f.ids, "is_fraud"].to_numpy(float)
        f.live_ath = f.slot9
        # Bo_active = Bo12 + oriented Slot9 residual vs Slot8 rebuild test
        slot8_test = f.slot8
        s9_delta_t = unitize_f32(v20.L(f.slot9) - v20.L(slot8_test))
        # OOF proxy: E2_heavy_ridge from overnight vault if present
        vault = OUT / "gm_overnight_vault.npz"
        if vault.exists():
            Z = np.load(vault)
            if "E2_heavy_ridge_oof" in Z.files:
                u9 = unitize_f32(Z["E2_heavy_ridge_oof"])
                ut9 = unitize_f32(Z["E2_heavy_ridge_test"]) if "E2_heavy_ridge_test" in Z.files else s9_delta_t
            else:
                u9 = unitize_f32(f.z_s8)  # placeholder
                ut9 = s9_delta_t
        else:
            u9, ut9 = unitize_f32(f.z_s8), s9_delta_t
        f.u_s9, f.ut_s9 = u9, ut9
        f.Bo_active = np.column_stack([f.Bo12, u9]).astype(np.float64)
        f.Bt_active = np.column_stack([f.Bt12, ut9]).astype(np.float64)
        f.Bt_all_active = np.column_stack([f.Bt_active, f.bad6, f.bad7]).astype(np.float64)
        # BCE residuals vs Slot9-oriented base (use z_S8 as working base; ATH is Slot9 on test)
        f.g_base = np.zeros(f.n_tr, dtype=np.float32)
        f.h_base = np.zeros(f.n_tr, dtype=np.float32)
        for tr, te in f.folds:
            clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
            clf.fit(f.z_s8[tr].reshape(-1, 1), f.y[tr])
            a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
            p = expit(a * f.z_s8 + b).astype(np.float32)
            f.g_base[te] = f.y[te] - p[te]
            f.h_base[te] = np.clip(p[te] * (1.0 - p[te]), 1e-6, None)
        log(
            f"anchors ready Slot9 1-ρ vs Slot8={v20.one_m_rho_e4(f.slot9, f.slot8):.3f}e-4 "
            f"z_S8 AUC f0={roc_auc_score(f.y[f.te0], f.z_s8[f.te0]):.6f} rss={rss_mb():.0f}MB"
        )
        gc.collect()

    def project(self, vo, vt):
        v20 = self.v20
        f = self.fac
        oo, ot, pre = v20.two_stage_orth(
            vo, vt, f.Bo_active, f.Bt_active, f.Bt_all_active, f.folds, ridge_alpha=10.0
        )
        u, ut = f.orient(oo, ot)
        return u, ut, pre

    def leverage_A(self):
        log("=== Leverage A: per-fold test_f-matched GPT2-med + L7 Bayes ===")
        v20, f = self.v20, self.fac
        feats_o = np.zeros((f.n_tr, 16), dtype=np.float32)
        feats_t = np.zeros((f.n_te, 16), dtype=np.float32)
        ms_o, ms_t = f.ms_o, f.ms_t
        lm_o, lm_t = f.XO["megayak_5f_lm"], f.XT["megayak_5f_lm"]

        # optional fwd/rev
        has_rev = all((REV / f"oof_{i}.npy").exists() and (REV / f"test_{i}.npy").exists() for i in range(5))
        has_g3 = (G3 / "gpt2_3ep_oof.npy").exists() and (G3 / "gpt2_3ep_test.npy").exists()
        g3_o = np.load(G3 / "gpt2_3ep_oof.npy").ravel()[: f.n_tr] if has_g3 else None
        g3_t = np.load(G3 / "gpt2_3ep_test.npy").ravel()[: f.n_te] if has_g3 else None

        te_acc = np.zeros((f.n_te, 16), dtype=np.float32)
        te_cnt = 0
        for fi in range(5):
            og = np.load(MED / f"oof_{fi}.npy")  # (n_te_fold, 8, 2)
            tg = np.load(MED / f"test_{fi}.npy")  # (n_te, 4, 2)
            ol = np.load(L7DIR / f"oof_{fi}.npy")
            tl = np.load(L7DIR / f"test_{fi}.npy")
            # match order dims: use min K
            kg = min(og.shape[1], tg.shape[1])
            kl = min(ol.shape[1], tl.shape[1])
            og, tg = og[:, :kg], tg[:, :kg]
            ol, tl = ol[:, :kl], tl[:, :kl]
            zg, sg, pg, lp, ln, cg = llr_mean_std(og)
            zgt, sgt, pgt, lpt, lnt, cgt = llr_mean_std(tg)
            zl, sl, pl, _, _, cl = llr_mean_std(ol)
            zlt, slt, plt, _, _, clt = llr_mean_std(tl)
            te_idx = f.folds[fi][1]
            # differentials (fold-local L)
            d_ms = v20.L(zg) - v20.L(ms_o[te_idx])
            d_lm = v20.L(zg) - v20.L(lm_o[te_idx])
            d_ms_t = v20.L(zgt) - v20.L(ms_t)
            d_lm_t = v20.L(zgt) - v20.L(lm_t)
            if has_rev:
                orv = np.load(REV / f"oof_{fi}.npy")
                trv = np.load(REV / f"test_{fi}.npy")
                kr = min(orv.shape[1], trv.shape[1], kg)
                zr = (orv[:, :kr, 1] - orv[:, :kr, 0]).mean(1)
                zrt = (trv[:, :kr, 1] - trv[:, :kr, 0]).mean(1)
                if has_g3:
                    d_fr = v20.L(g3_o[te_idx]) - v20.L(zr)
                    d_fr_t = v20.L(g3_t) - v20.L(zrt)
                else:
                    d_fr = v20.L(zg) - v20.L(zr)
                    d_fr_t = v20.L(zgt) - v20.L(zrt)
            else:
                d_fr = np.zeros_like(zg)
                d_fr_t = np.zeros_like(zgt)
            pack_o = np.column_stack(
                [zg, sg, pg, lp, ln, cg, zl, sl, pl, cl, d_ms, d_lm, d_fr, zg - zl, pg - pl, sg]
            ).astype(np.float32)
            pack_t = np.column_stack(
                [zgt, sgt, pgt, lpt, lnt, cgt, zlt, slt, plt, clt, d_ms_t, d_lm_t, d_fr_t, zgt - zlt, pgt - plt, sgt]
            ).astype(np.float32)
            feats_o[te_idx] = pack_o
            te_acc += pack_t
            te_cnt += 1
            del og, tg, ol, tl, pack_o, pack_t
            gc.collect()
        feats_t = (te_acc / max(te_cnt, 1)).astype(np.float32)

        # LOFO Ridge + shallow CatBoost on (g_base) with sample weight h_base
        oof = np.zeros(f.n_tr, dtype=np.float32)
        te_pred = np.zeros(f.n_te, dtype=np.float32)
        for tr, te in f.folds:
            Xtr, Xte = feats_o[tr], feats_o[te]
            ytr = f.g_base[tr]
            wtr = f.h_base[tr]
            ridge = Ridge(alpha=500.0, fit_intercept=True)
            ridge.fit(Xtr, ytr, sample_weight=wtr)
            pr = ridge.predict(Xte).astype(np.float32)
            pt = ridge.predict(feats_t).astype(np.float32)
            # internal 90/10 for cat
            sk = StratifiedKFold(5, shuffle=True, random_state=42)
            # use y labels for split strata
            tr_idx = np.arange(len(tr))
            y_lab = f.y[tr]
            it = sk.split(tr_idx, y_lab)
            tr2, va2 = next(it)
            # expand to absolute? use relative indices into tr
            cb_acc = np.zeros(len(te), dtype=np.float32)
            cb_te = np.zeros(f.n_te, dtype=np.float32)
            for seed in (0, 1, 2, 3, 4):
                if seed >= 2:
                    break  # 2 seeds for mem; spec said shallow 5-seed — use 2 under RSS cap
                cb = CatBoostRegressor(
                    depth=3,
                    l2_leaf_reg=40,
                    iterations=120,
                    learning_rate=0.03,
                    loss_function="RMSE",
                    random_seed=seed,
                    thread_count=N_JOBS,
                    verbose=False,
                    allow_writing_files=False,
                )
                cb.fit(
                    Xtr[tr2],
                    ytr[tr2],
                    sample_weight=wtr[tr2],
                    eval_set=(Xtr[va2], ytr[va2]),
                    use_best_model=True,
                )
                cb_acc += cb.predict(Xte).astype(np.float32)
                cb_te += cb.predict(feats_t).astype(np.float32)
                del cb
            cb_acc /= 2.0
            cb_te /= 2.0
            oof[te] = 0.55 * pr + 0.45 * cb_acc
            te_pred += 0.55 * pt + 0.45 * cb_te
            del ridge, Xtr, Xte
            gc.collect()
        te_pred /= 5.0
        self.results["M_A_bayes_lm"] = {"oof": oof, "test": te_pred}
        log(f"A done oof_std={oof.std():.4f} rss={rss_mb():.0f}MB")
        del feats_o, feats_t
        gc.collect()

    def leverage_B(self):
        log("=== Leverage B: vault fusion booster ===")
        v20, f = self.v20, self.fac

        def lp(po, pt):
            if not Path(po).exists() or not Path(pt).exists():
                return None
            return f32(np.load(po).ravel()[: f.n_tr]), f32(np.load(pt).ravel()[: f.n_te])

        streams = {}
        for name, po, pt in [
            ("b_xgb", PB / "xgb_llr_oof.npy", PB / "xgb_llr_test.npy"),
            ("p42", MV2 / "P4_L4.2_pure5f_oof.npy", MV2 / "P4_L4.2_pure5f_test.npy"),
            ("p43", MV2 / "P4_L4.3_pure5f_oof.npy", MV2 / "P4_L4.3_pure5f_test.npy"),
            ("p45", MV2 / "P4_L4.5_pure5f_oof.npy", MV2 / "P4_L4.5_pure5f_test.npy"),
            ("tabm", HARV / "oof_tabm_5f.npy", HARV / "test_tabm_5f.npy"),
            ("tabm_s", HARV / "oof_tabm_5f_std.npy", HARV / "test_tabm_5f_std.npy"),
            ("knn", HARV / "oof_orig20k_knn.npy", HARV / "test_orig20k_knn.npy"),
            ("v10r", V10 / "v10_ridge_resid_orth_LM_oof.npy", V10 / "v10_ridge_resid_orth_LM_test.npy"),
        ]:
            pair = lp(po, pt)
            if pair is None and name == "knn":
                pair = lp(HARV / "oof_orig20k_knn.npy", HARV / "orig20k_knn_test.npy")
            if pair is not None:
                streams[name] = pair

        # PCHIP-align continuous p43 test to oof scale
        if "p43" in streams:
            o, t = streams["p43"]
            streams["p43"] = (o, f32(pchip_align_test(t, o)))

        knn24_o = f32(np.load(HARV / "knn24_oof.npy")[: f.n_tr])
        knn24_t = f32(np.load(HARV / "knn24_test.npy")[: f.n_te])
        # 18 lowest-collinearity cols vs z_s8
        corr = [abs(float(np.corrcoef(knn24_o[:, j], f.z_s8)[0, 1])) for j in range(knn24_o.shape[1])]
        keep = list(np.argsort(corr)[:18])
        knn24_o, knn24_t = knn24_o[:, keep], knn24_t[:, keep]

        # gpt2m mean from med (averaged fold oofs already in vault E2 or rebuild)
        gpt_o, gpt_t = v20.assemble_gpt2(MED, f.y, f.folds)

        z0 = unitize_f32(f.z_s8)
        feat_names = []
        Fo, Ft = [], []

        def orth_fold(v_o, v_t):
            r_o = np.zeros(f.n_tr, dtype=np.float32)
            betas, intercepts = [], []
            vo_u = unitize_f32(v_o)
            for tr, te in f.folds:
                lr = Ridge(alpha=10.0)
                lr.fit(z0[tr].reshape(-1, 1), vo_u[tr])
                betas.append(float(lr.coef_[0]))
                intercepts.append(float(lr.intercept_))
                r_o[te] = vo_u[te] - float(lr.coef_[0]) * z0[te] - float(lr.intercept_)
            # test orth vs Slot9 live logit (Law: continuous logits only)
            zt = unitize_f32(v20.L(f.live_ath))
            lr = Ridge(alpha=10.0).fit(zt.reshape(-1, 1), unitize_f32(v_t))
            r_t = unitize_f32(v_t) - float(lr.coef_[0]) * zt - float(lr.intercept_)
            return unitize_f32(r_o), unitize_f32(r_t)

        orths = {}
        for name, (vo, vt) in streams.items():
            orths[name] = orth_fold(vo, vt)
        r_gpt = orth_fold(gpt_o, gpt_t)
        orths["gpt2m"] = r_gpt

        def add(name, o, t):
            feat_names.append(name)
            Fo.append(f32(o))
            Ft.append(f32(t))

        for name, (o, t) in orths.items():
            add(f"r_{name}", o, t)
        # agreements
        for a, b, nm in [
            ("gpt2m", "tabm", "agree_lm_tabm"),
            ("gpt2m", "knn", "agree_lm_knn"),
            ("tabm", "knn", "agree_tabm_knn"),
            ("p45", "b_xgb", "agree_p4_bxgb"),
        ]:
            if a in orths and b in orths:
                add(nm, orths[a][0] * orths[b][0], orths[a][1] * orths[b][1])
        if "tabm" in orths and "tabm_s" in streams:
            s = streams["tabm_s"][0]
            st = streams["tabm_s"][1]
            add("tabm_prec", orths["tabm"][0] / (0.05 + np.abs(s)), orths["tabm"][1] / (0.05 + np.abs(st)))
        for j in range(knn24_o.shape[1]):
            add(f"knn24_{j}", knn24_o[:, j], knn24_t[:, j])

        Xo = np.column_stack(Fo).astype(np.float32)
        Xt = np.column_stack(Ft).astype(np.float32)
        log(f"B feature matrix {Xo.shape} rss={rss_mb():.0f}MB")
        del Fo, Ft, streams, orths, knn24_o, knn24_t
        gc.collect()

        try:
            import lightgbm as lgb
        except ImportError:
            lgb = None

        oof = np.zeros(f.n_tr, dtype=np.float32)
        te_pred = np.zeros(f.n_te, dtype=np.float32)
        for tr, te in f.folds:
            ytr, wtr = f.g_base[tr], f.h_base[tr]
            # internal 90/10
            sk = StratifiedKFold(10, shuffle=True, random_state=7)
            tr2, va2 = next(sk.split(np.zeros(len(tr)), f.y[tr]))
            blend_te = np.zeros(len(te), dtype=np.float32)
            blend_t = np.zeros(f.n_te, dtype=np.float32)
            n_m = 0
            if lgb is not None:
                for seed in (0, 1, 2):
                    m = lgb.LGBMRegressor(
                        max_depth=3,
                        num_leaves=6,
                        min_child_samples=800,
                        colsample_bytree=0.25,
                        subsample=0.8,
                        reg_alpha=20.0,
                        reg_lambda=100.0,
                        learning_rate=0.02,
                        n_estimators=90,
                        max_bin=127,
                        n_jobs=N_JOBS,
                        random_state=seed,
                        verbosity=-1,
                    )
                    m.fit(
                        Xo[tr][tr2],
                        ytr[tr2],
                        sample_weight=wtr[tr2],
                        eval_set=[(Xo[tr][va2], ytr[va2])],
                        callbacks=[lgb.early_stopping(20, verbose=False)],
                    )
                    blend_te += m.predict(Xo[te]).astype(np.float32)
                    blend_t += m.predict(Xt).astype(np.float32)
                    n_m += 1
                    del m
            for seed in (0, 1, 2):
                cb = CatBoostRegressor(
                    depth=4,
                    l2_leaf_reg=45.0,
                    rsm=0.30,
                    learning_rate=0.025,
                    iterations=140,
                    thread_count=N_JOBS,
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                )
                cb.fit(
                    Xo[tr][tr2],
                    ytr[tr2],
                    sample_weight=wtr[tr2],
                    eval_set=(Xo[tr][va2], ytr[va2]),
                    use_best_model=True,
                )
                blend_te += cb.predict(Xo[te]).astype(np.float32)
                blend_t += cb.predict(Xt).astype(np.float32)
                n_m += 1
                del cb
            ridge = Ridge(alpha=2000.0)
            ridge.fit(Xo[tr], ytr, sample_weight=wtr)
            blend_te += ridge.predict(Xo[te]).astype(np.float32)
            blend_t += ridge.predict(Xt).astype(np.float32)
            n_m += 1
            oof[te] = blend_te / n_m
            te_pred += blend_t / n_m
            del ridge
            gc.collect()
        te_pred /= 5.0
        self.results["M_B_vault_booster"] = {"oof": oof, "test": te_pred}
        log(f"B done oof_std={oof.std():.4f} rss={rss_mb():.0f}MB")
        del Xo, Xt
        gc.collect()

    def leverage_C(self):
        log("=== Leverage C: ordered-CTR CatBoost on raw cat keys ===")
        f = self.fac
        names, Xtr, Xte = build_cat_keys(f.train, f.test)
        log(f"C keys={len(names)} X={Xtr.shape} rss={rss_mb():.0f}MB")
        cat_idx = list(range(len(names)))
        oof = np.zeros(f.n_tr, dtype=np.float32)
        te_pred = np.zeros(f.n_te, dtype=np.float32)
        for fi, (tr, te) in enumerate(f.folds):
            sk = StratifiedKFold(10, shuffle=True, random_state=42 + fi)
            tr2, va2 = next(sk.split(np.zeros(len(tr)), f.y[tr]))
            acc_te = np.zeros(len(te), dtype=np.float32)
            acc_t = np.zeros(f.n_te, dtype=np.float32)
            for seed in (0, 1):
                cb = CatBoostClassifier(
                    loss_function="Logloss",
                    eval_metric="AUC",
                    depth=5,
                    max_ctr_complexity=4,
                    one_hot_max_size=4,
                    l2_leaf_reg=30.0,
                    learning_rate=0.04,
                    iterations=500,
                    thread_count=N_JOBS,
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                )
                cb.fit(
                    Xtr[tr][tr2],
                    f.y[tr][tr2],
                    cat_features=cat_idx,
                    eval_set=(Xtr[tr][va2], f.y[tr][va2]),
                    use_best_model=True,
                )
                acc_te += cb.predict_proba(Xtr[te])[:, 1].astype(np.float32)
                acc_t += cb.predict_proba(Xte)[:, 1].astype(np.float32)
                del cb
                gc.collect()
            oof[te] = acc_te / 2.0
            te_pred += acc_t / 2.0
            log(f"C fold{fi} AUC={roc_auc_score(f.y[te], oof[te]):.5f} rss={rss_mb():.0f}MB")
        te_pred /= 5.0
        self.results["M_C_ordered_ctr"] = {"oof": oof, "test": te_pred}
        del Xtr, Xte
        gc.collect()

    def leverage_D(self):
        log("=== Leverage D: minimax portfolio synthesis ===")
        v20, f = self.v20, self.fac
        members = []
        for k in ("M_A_bayes_lm", "M_B_vault_booster", "M_C_ordered_ctr"):
            if k in self.results:
                u, ut, pre = self.project(self.results[k]["oof"], self.results[k]["test"])
                self.results[k]["u"], self.results[k]["ut"], self.results[k]["pre"] = u, ut, pre
                members.append((k, u, ut))
                log(f"  projected {k} raw_disp={pre['raw_disp']:.3f} raw_KS={pre['raw_KS']:.4f}")
        # cached survivors
        vault = OUT / "gm_overnight_vault.npz"
        if vault.exists():
            Z = np.load(vault)
            for key, tag in [
                ("prior_cache_P1_raw_only_oof", "P1_raw_only"),
                ("E2_heavy_ridge_oof", "E2_heavy_ridge"),
            ]:
                tk = key.replace("_oof", "_test")
                if key in Z.files and tk in Z.files:
                    u, ut, pre = self.project(Z[key], Z[tk])
                    members.append((tag, u, ut))
                    log(f"  projected cache {tag} raw_disp={pre['raw_disp']:.3f}")
        if not members:
            log("D: no members")
            return
        Uo = np.column_stack([m[1] for m in members]).astype(np.float64)
        Ut = np.column_stack([m[2] for m in members]).astype(np.float64)
        K = Uo.shape[1]
        rng = np.random.default_rng(42)
        best = None
        for _ in range(200):
            a = rng.dirichlet(np.ones(K))
            a = np.clip(a, 0, 0.45)
            a /= a.sum()
            u = v20.unitize(Uo @ a)
            score = 0.0
            for bname, base in f.bases.items():
                folds_d = [
                    (roc_auc_score(f.y[te], (base + 0.012 * u)[te]) - roc_auc_score(f.y[te], base[te])) * 1e5
                    for _, te in f.folds
                ]
                score += float(np.mean(folds_d) + 0.6 * np.min(folds_d))
            if best is None or score > best[0]:
                best = (score, a.copy())
        assert best is not None
        a = best[1]
        u = v20.unitize(Uo @ a)
        ut = v20.unitize(Ut @ a)
        self.results["M_D_minimax_portfolio"] = {
            "oof": u.astype(np.float32),
            "test": ut.astype(np.float32),
            "alpha": {members[i][0]: float(a[i]) for i in range(K)},
            "score": float(best[0]),
        }
        u2, ut2, pre = self.project(u, ut)
        self.results["M_D_minimax_portfolio"]["u"] = u2
        self.results["M_D_minimax_portfolio"]["ut"] = ut2
        self.results["M_D_minimax_portfolio"]["pre"] = pre
        log(f"D alpha={self.results['M_D_minimax_portfolio']['alpha']} score={best[0]:.2f}")

    def scorecard_and_stage(self):
        log("=== Step3: 6-gate scorecard vs ATH 0.95399 ===")
        v20, f = self.v20, self.fac
        rows = []
        for name in ("M_A_bayes_lm", "M_B_vault_booster", "M_C_ordered_ctr", "M_D_minimax_portfolio"):
            if name not in self.results or "u" not in self.results[name]:
                if name in self.results:
                    u, ut, pre = self.project(self.results[name]["oof"], self.results[name]["test"])
                    self.results[name]["u"], self.results[name]["ut"], self.results[name]["pre"] = u, ut, pre
                else:
                    continue
            u = self.results[name]["u"]
            ut = self.results[name]["ut"]
            pre = self.results[name].get("pre", {})
            g12 = v20.forensic(u, ut, f.Bt_active)
            g9 = v20.forensic(u, ut, f.Bt9)
            f3 = abs(float(np.corrcoef(v20.unitize(ut), v20.unitize(v20.L(f.slot8) - v20.L(f.slot5)))[0, 1]))
            best_row = None
            for w in (0.006, 0.008, 0.010, 0.012, 0.016, 0.020):
                oof = {b: f.bases[b] + w * u for b in f.bases}
                per = {b: v20.h2h(oof[b], f.bases[b], f.y, f.folds) for b in f.bases}
                # candidate on Slot8 recipe + w*ut, then measure vs LIVE Slot9
                cand = v20.qlock(
                    v20.L(f.slot3) + v20.W8_O * f.ut_r2 + v20.W8_C * f.ut_cat + v20.W8_U * f.ut_u5 + w * ut,
                    f.lv,
                )
                # blend toward Slot9 for submission cand: Slot9 + small step
                cand9 = v20.qlock(v20.L(f.slot9) + w * ut, f.slot9)
                rho = v20.one_m_rho_e4(cand9, f.slot9)
                delta = v20.unitize(v20.L(cand9) - v20.L(f.slot9))
                c6 = float(np.corrcoef(delta, f.bad6)[0, 1])
                c7 = float(np.corrcoef(delta, f.bad7)[0, 1])
                # Gate1 uses pre-projection raw
                g1_disp = float(pre.get("raw_disp", g12["raw_disp"]))
                g1_ks = float(pre.get("raw_KS", g12["raw_KS"]))
                ok = (
                    0.92 <= g1_disp <= 1.08
                    and g1_ks < 0.025
                    and g9["F2"] < 0.035
                    and g12["F2"] < 0.035
                    and f3 < 0.045
                    and all(per[b]["pos"] == 5 for b in per)
                    and all(per[b]["min"] > 0 for b in per)
                    and all(per[b]["sf25"] >= 18 for b in per)
                    and all(per[b]["t"] >= 2.80 for b in per)
                    and 0.25 <= rho <= 1.10
                    and abs(c6) < 0.20
                    and abs(c7) < 0.20
                )
                row = {
                    "name": f"{name}_w{w:.3f}",
                    "family": name,
                    "w": w,
                    "per": per,
                    "rho": rho,
                    "c6": c6,
                    "c7": c7,
                    "g9": g9,
                    "g12": g12,
                    "f3": f3,
                    "pre_disp": g1_disp,
                    "pre_KS": g1_ks,
                    "ok_gates": ok,
                    "cand": cand9,
                    "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                    "min_lift": float(np.min([per[b]["min"] for b in per])),
                }
                mark = "★" if ok else "~"
                log(
                    f"{mark} {row['name']}: disp={g1_disp:.3f} KS={g1_ks:.4f} F2a={g12['F2']:.4f} "
                    f"ρ={rho:.3f} c7={c7:+.3f} z={per['z_S8']['pos']}/5 m={per['z_S8']['mean']:+.2f} "
                    f"t={per['z_S8']['t']:.2f} sf={per['z_S8']['sf25']} gates={ok}"
                )
                if ok:
                    self.promoted.append(row)
                if best_row is None or row["mean_lift"] > best_row["mean_lift"]:
                    best_row = row
            if best_row:
                rows.append(best_row)
                path = SUB / f"sub_v21_local_4leverage_{name}_NOSUBMIT.csv"
                pd.DataFrame({"id": f.ids, "is_fraud": best_row["cand"]}).to_csv(path, index=False)
                log(f"STAGED {path.name}")

        # save npz
        payload = {}
        for k, v in self.results.items():
            payload[f"{k}_oof"] = f32(v.get("u", v["oof"]))
            payload[f"{k}_test"] = f32(v.get("ut", v["test"]))
        np.savez_compressed(NPZ_OUT, **payload)
        log(f"saved {NPZ_OUT}")

        # submit if gates + positive mean lift vs ATH bases
        submit = False
        pub = None
        chosen = None
        if self.promoted:
            chosen = sorted(self.promoted, key=lambda r: (r["min_lift"], r["mean_lift"]), reverse=True)[0]
            if chosen["mean_lift"] > 0 and chosen["ok_gates"]:
                out_csv = SUB / "sub_v21_local_4leverage_LIVE_candidate.csv"
                pd.DataFrame({"id": f.ids, "is_fraud": chosen["cand"]}).to_csv(out_csv, index=False)
                msg = (
                    f"v21-local4lev {chosen['name']} vs ATH0.95399 | "
                    f"mean={chosen['mean_lift']:+.2f}e-5 min={chosen['min_lift']:+.2f} "
                    f"1-ρ={chosen['rho']:.3f}e-4 c7={chosen['c7']:+.3f}"
                )
                log(f"SUBMIT attempt: {msg}")
                try:
                    r = subprocess.run(
                        [KAG, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(out_csv), "-m", msg],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    log(r.stdout.strip())
                    log(r.stderr.strip())
                    submit = r.returncode == 0 and "400" not in (r.stdout + r.stderr)
                    if submit:
                        for i in range(36):
                            time.sleep(20 if i else 5)
                            p = subprocess.run(
                                [KAG, "competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
                                capture_output=True,
                                text=True,
                                timeout=60,
                            )
                            for ln in p.stdout.strip().splitlines()[1:8]:
                                if "v21_local_4leverage_LIVE" in ln or "local_4leverage" in ln:
                                    cols = ln.split(",")
                                    # fragile CSV parse — also try status field
                                    log(f"poll {ln[:200]}")
                                    if "Complete" in ln:
                                        # public score often near end
                                        for tok in reversed(cols):
                                            tok = tok.strip()
                                            try:
                                                pub = float(tok)
                                                break
                                            except ValueError:
                                                continue
                                    break
                            if pub is not None:
                                break
                except Exception as e:
                    log(f"submit failed: {e}")
            else:
                log("gates/lift insufficient — HOLD")
        else:
            log("NO 6-gate clear — HOLD Slot10")

        # report
        LOG.mkdir(parents=True, exist_ok=True)
        body = [
            "# GM V21 Local-Only 4-Leverage Upgrade",
            "",
            f"**Local CPU only · n_jobs={N_JOBS} · elapsed {time.time()-T0:.0f}s · peak RSS≈{rss_mb():.0f}MB**",
            "",
            f"- Live ATH: `0.95399` (`{SLOT9.name}`)",
            f"- Promoted (6-gate): {len(self.promoted)}",
            f"- Submit: {submit}",
            f"- Public LB: {pub}",
            "",
            "## Scorecard (best w per family)",
            "",
            "| family | w | meanΔe-5 | minΔe-5 | 1-ρ e-4 | c7 | F2 | gates |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        for r in rows:
            body.append(
                f"| {r['family']} | {r['w']:.3f} | {r['mean_lift']:+.2f} | {r['min_lift']:+.2f} | "
                f"{r['rho']:.3f} | {r['c7']:+.3f} | {r['g12']['F2']:.4f} | {r['ok_gates']} |"
            )
        body += ["", "## Log", "```", *LINES[-400:], "```", ""]
        REPORT.write_text("\n".join(body) + "\n")
        log(f"REPORT {REPORT}")
        log(f"DONE submit={submit} pub={pub}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    mon = start_monitor(os.getpid())
    log(f"V21 local 4-leverage START pid={os.getpid()} n_jobs={N_JOBS} monitor={mon.pid}")
    try:
        v20 = load_v20()
        # force n_jobs in imported module paths via env already set
        job = Local4Leverage(v20)
        job.rebuild()
        job.leverage_A()
        job.leverage_B()
        job.leverage_C()
        job.leverage_D()
        job.scorecard_and_stage()
    except Exception as e:
        log(f"FATAL: {e}")
        traceback.print_exc()
        raise
    finally:
        try:
            mon.terminate()
        except Exception:
            pass
        log(f"COMPLETE elapsed={time.time()-T0:.0f}s peak_rss≈{rss_mb():.0f}MB")


if __name__ == "__main__":
    main()
