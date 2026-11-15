#!/usr/bin/env python3
"""GM V20 Autonomous Event-Driven Research & Promotion Factory.

0 GPU pods ($0.00/hr). Local Mac CPU / MPS only.
Events 1→5 research + survivor vault; Event 6 submits Slot9/10 IFF all 6 hard gates.
"""
from __future__ import annotations

import json
import subprocess
import time
import traceback
import warnings
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.optimize import nnls
from scipy.special import expit, logit
from scipy.stats import ks_2samp, norm, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
TMP15, TMP16 = Path("/tmp/s6e11_v15"), Path("/tmp/s6e11_v16")
OUT, SUB, DATA, LOG = TMP16 / "out", REPO / "submissions", REPO / "data", REPO / "logs/overnight"
PRED = REPO / "preds"
V10, MED, REV, MV2 = PRED / "v10_transductive_bulsah", PRED / "gpt2_med_4ep", PRED / "gpt2_rev", PRED / "matrix_v2"
HARV = PRED / "fast_gpu_harvest"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
ORIG = REPO / "data/creditcard.csv"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
SLOT6 = SUB / "sub_v18_1_LIVE_slot6_Slot5_plus_R1_pure_Bo10_w0p022.csv"
SLOT7 = SUB / "sub_v19_0_LIVE_slot7_zero_echo_peer_hybrid.csv"
SLOT8 = SUB / "sub_v19_4_LIVE_slot8_A_cat_wo0.016_wc0.012_wu0.012.csv"
G26_OOF, G26_T0 = TMP16 / "g26/gpt2_s2026_oof_5fold.npy", TMP16 / "g26/gpt2_s2026_test_f0.npy"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
VAULT = OUT / "gm_survivor_vault.npz"
REPORT = LOG / "GM_V20_AUTONOMOUS_EVENT_FACTORY_REPORT.md"
STATE = OUT / "gm_v20_factory_state.json"
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1, W_FE = 0.0355, 0.01045
W8_O, W8_C, W8_U = 0.016, 0.012, 0.012
N_TR, N_TE = 699960, 299984
NUM_COLS = [
    "amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
    "velocity_score", "merchant_risk_score", "txn_count_last_24h", "card_age_months",
    "customer_age", "cvv_retry_count",
]
T0 = time.time()
LINES: list[str] = []


def log(msg: str) -> None:
    s = f"[{time.time()-T0:7.1f}s] {msg}"
    print(s, flush=True)
    LINES.append(s)


def L(v):
    v = np.asarray(v, float).ravel()
    r = np.clip((rankdata(v, method="average") - 0.5) / len(v), 1e-12, 1 - 1e-12)
    return np.log(r / (1 - r))


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def qlock(raw, ref):
    raw, ref = np.asarray(raw, float), np.asarray(ref, float)
    out = np.empty_like(ref)
    out[np.argsort(raw)] = np.sort(ref)
    return out


def L_fold(raw, fold_id):
    out = np.zeros_like(raw, dtype=np.float64)
    for f in range(5):
        m = fold_id == f
        out[m] = L(raw[m])
    return out


def col_z(X):
    X = np.asarray(X, float)
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-12)


def gini10(Z):
    K = Z.shape[-1]
    s = np.zeros(Z.shape[:-1], dtype=np.float64)
    for i, j in combinations(range(K), 2):
        s += np.abs(Z[..., i] - Z[..., j])
    return s / (K * (K - 1) / 2.0)


def one_m_rho_e4(a, b):
    return (1.0 - float(spearmanr(a, b).correlation)) * 1e4


def scrub(x):
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=10.0):
    vo, vt = unitize(v_oof), unitize(v_test)
    orth_o = np.zeros(len(vo))
    betas = []
    for tr, te in folds:
        lr = Ridge(alpha=ridge_alpha, fit_intercept=True)
        lr.fit(Bo[tr], vo[tr])
        betas.append(lr.coef_.copy())
        orth_o[te] = vo[te] - Bo[te] @ lr.coef_ - lr.intercept_
    beta = np.mean(np.stack(betas, 0), 0)
    orth_t = vt - Bt @ beta
    return orth_o - orth_o.mean(), orth_t - orth_t.mean(), beta


def zca_matrix(X, reg=1e-2):
    C = np.cov(np.asarray(X, float), rowvar=False) + reg * np.eye(X.shape[1])
    s, Umat = np.linalg.eigh(C)
    s = np.clip(s, reg, None)
    return Umat @ np.diag(1.0 / np.sqrt(s)) @ Umat.T


def soft_disp_match(oo, ot, lo=0.92, hi=1.08):
    so, st = float(np.std(oo)) + 1e-12, float(np.std(ot)) + 1e-12
    disp = st / so
    if disp < lo:
        ot = ot * (lo * so / st)
    elif disp > hi:
        ot = ot * (hi * so / st)
    return oo, ot


def zca_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=10.0, reg=1e-2):
    """ZCA-whitened LOFO + OOF-beta transfer (no test-side refit — avoids V20.1 LB trap)."""
    vo, vt = unitize(v_oof), unitize(v_test)
    Bo_g, Bt_g = col_z(Bo), col_z(Bt)
    orth_o = np.zeros(len(vo))
    betas, intercepts = [], []
    for tr, te in folds:
        mu, sd = Bo_g[tr].mean(0), Bo_g[tr].std(0) + 1e-12
        Btr = (Bo_g[tr] - mu) / sd
        Bva = (Bo_g[te] - mu) / sd
        W = zca_matrix(Btr, reg)
        Btr_w, Bva_w = Btr @ W, Bva @ W
        mu_y, sd_y = float(vo[tr].mean()), float(vo[tr].std()) + 1e-12
        lr = Ridge(alpha=ridge_alpha, fit_intercept=True)
        lr.fit(Btr_w, (vo[tr] - mu_y) / sd_y)
        orth_o[te] = (vo[te] - mu_y) / sd_y - (Bva_w @ lr.coef_ + lr.intercept_)
        betas.append(lr.coef_.copy())
        intercepts.append(float(lr.intercept_))
    Bt_s = (Bt_g - Bt_g.mean(0)) / (Bt_g.std(0) + 1e-12)
    Wte = zca_matrix(Bt_s, reg)
    beta = np.mean(np.stack(betas, 0), 0)
    icept = float(np.mean(intercepts))
    mu_y, sd_y = float(vo.mean()), float(vo.std()) + 1e-12
    orth_t = (vt - mu_y) / sd_y - ((Bt_s @ Wte) @ beta + icept)
    # 2nd-stage: LOFO strip vs original Bo; transfer mean beta to Test (no y_test fit)
    oo2 = np.zeros(len(vo))
    b2 = []
    for tr, te in folds:
        lr = Ridge(alpha=1.0, fit_intercept=True)
        ytr = orth_o[tr] - orth_o[tr].mean()
        lr.fit(Bo_g[tr], ytr)
        oo2[te] = (orth_o[te] - orth_o[tr].mean()) - (Bo_g[te] @ lr.coef_ + lr.intercept_)
        b2.append(lr.coef_.copy())
    beta2 = np.mean(np.stack(b2, 0), 0)
    ot2 = (orth_t - orth_t.mean()) - Bt_g @ beta2
    oo2, ot2 = soft_disp_match(oo2, ot2)
    return oo2 - oo2.mean(), ot2 - ot2.mean()


def forensic(oo, ot, Bt):
    disp = float(np.std(ot) / (np.std(oo) + 1e-12))
    ut = unitize(ot)
    leaks = [abs(float(np.corrcoef(ut, Bt[:, j])[0, 1])) for j in range(Bt.shape[1])]
    return {
        "raw_disp": disp,
        "raw_KS": float(ks_2samp(unitize(oo), ut).statistic),
        "F2": float(max(leaks)) if leaks else 0.0,
    }


def h2h(cand, base, y, folds):
    d, deltas = [], []
    for _, te in folds:
        d.append((roc_auc_score(y[te], cand[te]) - roc_auc_score(y[te], base[te])) * 1e5)
        yf, c, b = y[te], cand[te], base[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(yf)), yf):
            deltas.append((roc_auc_score(yf[ste], c[ste]) - roc_auc_score(yf[ste], b[ste])) * 1e5)
    d25 = np.asarray(deltas, float)
    mean = float(d25.mean())
    se = float(d25.std(ddof=1) / np.sqrt(len(d25)))
    return {
        "pos": int(sum(x > 0 for x in d)),
        "mean": float(np.mean(d)),
        "min": float(np.min(d)),
        "folds": [round(float(x), 3) for x in d],
        "sf25": int((d25 > 0).sum()),
        "t": float(mean / (se + 1e-18)),
    }


def hygiene_ok(g9, g12, c6, c7):
    return (
        0.90 <= g12["raw_disp"] <= 1.10
        and g12["raw_KS"] < 0.025
        and g9["F2"] < 0.035
        and g12["F2"] < 0.038
        and abs(c6) < 0.20
        and abs(c7) < 0.20
    )


def hard_gates(per, rho, c6, c7, g9, g12, f3):
    """Event-6 promotion gates (strict)."""
    return (
        0.92 <= g12["raw_disp"] <= 1.08
        and g12["raw_KS"] < 0.025
        and g9["F2"] < 0.035
        and g12["F2"] < 0.038
        and f3 < 0.045
        and all(per[b]["pos"] == 5 for b in per)
        and all(per[b]["min"] > 0.05 for b in per)  # > +0.05e-5
        and all(per[b]["sf25"] >= 18 for b in per)
        and all(per[b]["t"] >= 2.80 for b in per)
        and 0.28 <= rho <= 1.10
        and abs(c6) < 0.20
        and abs(c7) < 0.20
    )


def survivor_ok(per, g9, g12, c6, c7):
    return hygiene_ok(g9, g12, c6, c7) and all(per[b]["mean"] > 0 for b in per) and all(
        per[b]["pos"] >= 4 for b in per
    )


def load_pair(po, pt, n_tr=N_TR, n_te=N_TE):
    if not Path(po).exists() or not Path(pt).exists():
        return None
    o = np.load(po).astype(np.float64).ravel()[:n_tr]
    t = np.load(pt).astype(np.float64).ravel()[:n_te]
    if len(o) != n_tr or len(t) != n_te:
        return None
    if not (np.isfinite(o).all() and np.isfinite(t).all()):
        return None
    if float(np.std(o)) < 1e-12 or float(np.std(t)) < 1e-12:
        return None
    return o, t


def assemble_gpt2(path, y, folds):
    zo = np.full(len(y), np.nan)
    zt_f = []
    for fi, (_, b) in enumerate(folds):
        oraw = np.load(path / f"oof_{fi}.npy")
        traw = np.load(path / f"test_{fi}.npy")
        llr = oraw[:, :, 1].astype(np.float64) - oraw[:, :, 0].astype(np.float64)
        zo[b] = llr.mean(1)
        tllr = traw[:, :, 1].astype(np.float64) - traw[:, :, 0].astype(np.float64)
        zt_f.append(tllr.mean(1))
    assert np.isfinite(zo).all()
    return zo.astype(np.float64), np.mean(zt_f, 0)


def bpf_contrast(a_o, a_t, b_o, b_t, fold_id):
    co = L_fold(a_o, fold_id) - L_fold(b_o, fold_id)
    ct = L(a_t) - L(b_t)
    so, st = float(co.std()) + 1e-12, float(ct.std()) + 1e-12
    ct = (ct - ct.mean()) * (so / st) + co.mean()
    return unitize(co), unitize(ct)


def save_report(extra: str = ""):
    LOG.mkdir(parents=True, exist_ok=True)
    body = [
        "# GM V20 Autonomous Event-Driven Factory",
        "",
        f"**0 GPU pods · elapsed {time.time()-T0:.0f}s · `$0.00/hr`**",
        "",
        "```",
        *LINES[-200:],
        "```",
        "",
        extra,
    ]
    REPORT.write_text("\n".join(body) + "\n")


def kaggle_recon():
    info = {"datasets": [], "kernels": [], "gpt2_files": []}
    try:
        r = subprocess.run(
            [KAG, "datasets", "files", "goodpjw2008/s6e11-target-first-gpt2-seed-2026"],
            capture_output=True, text=True, timeout=60,
        )
        info["gpt2_files"] = [ln for ln in r.stdout.splitlines() if ln.strip()][:40]
        log(f"E1 kaggle gpt2 dataset files: {len(info['gpt2_files'])} lines")
        if any("test_5fold" in ln or "gpt2_s2026_test_5fold" in ln for ln in info["gpt2_files"]):
            log("E1 NEW gpt2_s2026_test_5fold detected upstream")
    except Exception as e:
        log(f"E1 kaggle gpt2 files skip: {e}")
    try:
        r = subprocess.run(
            [KAG, "datasets", "list", "-s", "s6e11", "--sort-by", "updated"],
            capture_output=True, text=True, timeout=60,
        )
        info["datasets"] = r.stdout.splitlines()[:15]
        log(f"E1 datasets list ok ({len(info['datasets'])} lines)")
    except Exception as e:
        log(f"E1 datasets list skip: {e}")
    try:
        r = subprocess.run(
            [KAG, "kernels", "list", "-c", "playground-series-s6e11", "--sort-by", "scoreDescending", "--page-size", "30"],
            capture_output=True, text=True, timeout=90,
        )
        info["kernels"] = r.stdout.splitlines()[:35]
        log(f"E1 kernels list ok ({len(info['kernels'])} lines)")
    except Exception as e:
        log(f"E1 kernels list skip: {e}")
    return info


class Factory:
    def __init__(self):
        self.vault: dict[str, dict] = {}
        self.promoted: list[dict] = []
        self.slots_used = 0
        self.max_slots = 1  # V20.1 already burned 1 of 2 today
        self.state = {"events": {}, "submit": False, "public_lbs": []}

    def rebuild_slot8(self):
        log("=== REBUILD Slot8 ATH 0.95395 ===")
        cache = np.load(OUT / "z_ath_cache.npz")
        self.z_ath = cache["z_ath"]
        self.d1 = cache["d1"]
        self.y = cache["y"].astype(np.int32)
        self.fold_id = cache["fold_id"]
        self.d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
        self.dE_o = unitize(cache["dE"])
        self.folds = [(np.where(self.fold_id != f)[0], np.where(self.fold_id == f)[0]) for f in range(5)]
        self.te0 = self.folds[0][1]
        self.n_tr, self.n_te = len(self.y), N_TE

        Z = np.load(Q17, allow_pickle=True)
        self.qnames = [str(s) for s in Z["stream_names"]]
        self.XO = {n: Z[f"oof_{n}"].ravel() for n in self.qnames}
        self.XT = {n: Z[f"test_{n}"].ravel() for n in self.qnames}

        live = pd.read_csv(LIVE)
        self.ids = live["id"].to_numpy()
        self.lv = live["is_fraud"].to_numpy(float)
        self.slot3 = pd.read_csv(SLOT3).set_index("id").loc[self.ids, "is_fraud"].to_numpy(float)
        self.slot5 = pd.read_csv(SLOT5).set_index("id").loc[self.ids, "is_fraud"].to_numpy(float)
        self.slot6 = pd.read_csv(SLOT6).set_index("id").loc[self.ids, "is_fraud"].to_numpy(float)
        self.slot7 = pd.read_csv(SLOT7).set_index("id").loc[self.ids, "is_fraud"].to_numpy(float)
        self.slot8 = pd.read_csv(SLOT8).set_index("id").loc[self.ids, "is_fraud"].to_numpy(float)
        self.bad6 = unitize(L(self.slot6) - L(self.slot5))
        self.bad7 = unitize(L(self.slot7) - L(self.slot5))

        R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
        R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
        l7f_o, l7f_t = self.XO["lm7_5fold"], self.XT["lm7_5fold"]
        R3_t_L = np.stack([L(R3_te[f]) for f in range(5)], 0)
        raw_E_o = (L(R3) - L(l7f_o)) * (0.5 + 0.5 * U(self.z_ath))
        raw_E_t = (L(R3_t_L.mean(0)) - L(l7f_t)) * (0.5 + 0.5 * U(self.lv))
        dE_o2, dE_t, _ = frozen_lofo_orth(
            raw_E_o, raw_E_t,
            np.column_stack([unitize(self.z_ath), unitize(self.d1)]),
            np.column_stack([unitize(L(self.lv)), unitize(self.d1t)]),
            self.folds, ridge_alpha=1.0,
        )
        if np.corrcoef(self.dE_o, unitize(dE_o2))[0, 1] < 0:
            self.dE_o = -self.dE_o
        if roc_auc_score(self.y[self.te0], self.z_ath[self.te0] + 0.01 * unitize(dE_o2)[self.te0]) < roc_auc_score(
            self.y[self.te0], self.z_ath[self.te0] - 0.01 * unitize(dE_o2)[self.te0]
        ):
            dE_t = -dE_t
        self.dE_o, self.dE_t = unitize(self.dE_o), unitize(dE_t)

        ms_o = np.load(PRED / "P_super_v4_oof.npy").ravel()
        ms_t = np.load(PRED / "P_super_v4_test.npy").ravel()
        self.ms_o, self.ms_t = ms_o, ms_t
        g26_o = np.load(G26_OOF).ravel()[: self.n_tr]
        g26_t = np.load(G26_T0).ravel()[: self.n_te]
        self.Bo9 = np.column_stack([
            unitize(L(ms_o)), unitize(L(g26_o)), unitize(L(self.XO["megayak_5f_lm"])),
            unitize(L(l7f_o)), unitize(self.XO["daoviet_gpdvae"]), unitize(self.XO["gb_bag5"]),
            unitize(self.XO["yekenot_realmlp"]), unitize(self.d1), unitize(self.dE_o),
        ])
        self.Bt9 = np.column_stack([
            unitize(L(ms_t)), unitize(L(g26_t)), unitize(L(self.XT["megayak_5f_lm"])),
            unitize(L(l7f_t)), unitize(self.XT["daoviet_gpdvae"]), unitize(self.XT["gb_bag5"]),
            unitize(self.XT["yekenot_realmlp"]), unitize(self.d1t), unitize(self.dE_t),
        ])
        self.z_s3 = self.z_ath + 0.01045 * self.dE_o + 0.0055 * unitize(self.d1)
        ridge_s3 = Ridge(alpha=1.0, fit_intercept=True).fit(self.Bt9, unitize(L(self.slot3)))
        self.z_twin = unitize(self.Bo9 @ ridge_s3.coef_ + ridge_s3.intercept_)

        bank_o, bank_t = [], []
        skip = {"megayak_5f_lm", "lm7_5fold", "gpt2_seed2026", "masaya_gpt2", "daoviet_gpdvae", "gb_bag5", "yekenot_realmlp"}
        for j, nm in enumerate(BN9):
            if nm not in ("d1_v4", "Family_E"):
                bank_o.append(unitize(self.Bo9[:, j]))
                bank_t.append(unitize(self.Bt9[:, j]))
        for n in self.qnames:
            if n not in skip:
                bank_o.append(unitize(self.XO[n]))
                bank_t.append(unitize(self.XT[n]))
        for p in sorted(ALEX.glob("oof_*.npy"))[:8]:
            tp = ALEX / ("test_" + p.name[len("oof_"):])
            if tp.exists():
                bank_o.append(unitize(np.load(p).ravel()[: self.n_tr]))
                bank_t.append(unitize(np.load(tp).ravel()[: self.n_te]))
        Bo_bank = np.column_stack(bank_o)
        Bt_bank = np.column_stack(bank_t)
        locked_t = W_D1 * unitize(self.d1t) + W_FE * unitize(self.dE_t)
        w_nnls, _ = nnls(Bt_bank, (L(self.slot3) - locked_t) - (L(self.slot3) - locked_t).mean())
        self.z_nnls = W_D1 * unitize(self.d1) + W_FE * unitize(self.dE_o) + Bo_bank @ w_nnls

        fam1 = np.load(OUT / "v19_1_families.npz")

        def orient_bo9(vo, vt):
            oo, ot, _ = frozen_lofo_orth(vo, vt, self.Bo9, self.Bt9, self.folds, ridge_alpha=10.0)
            u = unitize(oo)
            if roc_auc_score(self.y[self.te0], self.z_s3[self.te0] + 0.01 * u[self.te0]) < roc_auc_score(
                self.y[self.te0], self.z_s3[self.te0] - 0.01 * u[self.te0]
            ):
                u, ot = -u, -ot
            return unitize(u), unitize(ot)

        self.u_r2, self.ut_r2 = orient_bo9(fam1["R2_v178_orig_oof"], fam1["R2_v178_orig_test"])
        self.u_cat, self.ut_cat = orient_bo9(fam1["F_cat_obliv_oof"], fam1["F_cat_obliv_test"])
        self.r2_raw_o = fam1["R2_v178_orig_oof"].ravel()
        self.r2_raw_t = fam1["R2_v178_orig_test"].ravel()
        self.cat_raw_o = fam1["F_cat_obliv_oof"].ravel()
        self.cat_raw_t = fam1["F_cat_obliv_test"].ravel()

        raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
        raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
        self.raw_o, self.raw_t = raw_o, raw_t
        llr_o = raw_o[:, :, 1] - raw_o[:, :, 0]
        Z_o = col_z(llr_o)
        U5_o = L_fold(gini10(Z_o), self.fold_id)
        U5_t = np.mean(
            np.stack([L(gini10(col_z(raw_t[f, :, :, 1] - raw_t[f, :, :, 0]))) for f in range(5)], 0), 0
        )
        oo5, ot5, _ = frozen_lofo_orth(
            U5_o, U5_t, np.column_stack([self.Bo9, self.u_r2]), np.column_stack([self.Bt9, self.ut_r2]),
            self.folds, ridge_alpha=1.0,
        )
        self.u_u5 = unitize(oo5)
        if roc_auc_score(self.y[self.te0], (self.z_s3 + 0.020 * self.u_r2)[self.te0] + 0.01 * self.u_u5[self.te0]) < roc_auc_score(
            self.y[self.te0], (self.z_s3 + 0.020 * self.u_r2)[self.te0] - 0.01 * self.u_u5[self.te0]
        ):
            self.u_u5, ot5 = -self.u_u5, -ot5
        self.ut_u5 = unitize(ot5)

        # U6..U8 banks
        def u_bank(ll_pos, ll_neg):
            llr = ll_pos - ll_neg
            Z = col_z(llr)
            K = Z.shape[1]
            h = K // 2
            pair_means = np.column_stack([0.5 * (Z[:, m] + Z[:, m + h]) for m in range(h)])
            pair_gaps = np.column_stack([np.abs(Z[:, m] - Z[:, m + h]) for m in range(h)])
            Zp, Zn = col_z(ll_pos), col_z(ll_neg)
            return {
                "U5": gini10(Z),
                "U6": pair_means.std(1),
                "U7": pair_gaps.mean(1),
                "U8": Zp.std(1) - Zn.std(1),
            }

        bank_o = u_bank(raw_o[:, :, 1], raw_o[:, :, 0])
        bank_t = {k: [] for k in bank_o}
        for f in range(5):
            bf = u_bank(raw_t[f, :, :, 1], raw_t[f, :, :, 0])
            for k in bank_t:
                bank_t[k].append(bf[k])
        self.U_o = {k: L_fold(bank_o[k], self.fold_id) for k in bank_o}
        self.U_t = {k: np.mean(np.stack([L(x) for x in bank_t[k]], 0), 0) for k in bank_o}

        self.z_s8 = self.z_s3 + W8_O * self.u_r2 + W8_C * self.u_cat + W8_U * self.u_u5
        self.z_t8 = self.z_twin + W8_O * self.u_r2 + W8_C * self.u_cat + W8_U * self.u_u5
        self.z_n8 = self.z_nnls + W8_O * self.u_r2 + W8_C * self.u_cat + W8_U * self.u_u5
        self.bases = {"z_S8": self.z_s8, "twin": self.z_t8, "nnls": self.z_n8}
        rebuild = qlock(L(self.slot3) + W8_O * self.ut_r2 + W8_C * self.ut_cat + W8_U * self.ut_u5, self.lv)
        rho = one_m_rho_e4(rebuild, self.slot8)
        log(f"Slot8 rebuild 1-ρ={rho:.3e}e-4")
        assert rho < 1e-5, rho

        self.Bo12 = np.column_stack([self.Bo9, self.u_r2, self.u_cat, self.u_u5])
        self.Bt12 = np.column_stack([self.Bt9, self.ut_r2, self.ut_cat, self.ut_u5])
        # Extended orth basis with fail-rays on test only for F3; for orth include unitized rays on OOF via zeros? Use test rays only in F3/c6/c7.
        # OOF proxy for rays: use orth of Slot8 increments is unavailable; orth against Bo12 only + check c6/c7 on test.
        self.Bo_ext = self.Bo12
        self.Bt_ext = np.column_stack([self.Bt12, self.bad6, self.bad7])

        # BCE Newton residuals on Slot8
        self.g_S8 = np.zeros(self.n_tr)
        self.h_S8 = np.zeros(self.n_tr)
        self.p_S8 = np.zeros(self.n_tr)
        for tr, te in self.folds:
            clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
            clf.fit(self.z_s8[tr].reshape(-1, 1), self.y[tr])
            a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
            p = expit(a * self.z_s8 + b)
            self.p_S8[te] = p[te]
            self.g_S8[te] = self.y[te] - p[te]
            self.h_S8[te] = np.clip(p[te] * (1 - p[te]), 1e-6, None)
        log(f"Bo12 ready cols={self.Bo12.shape[1]} g_S8 std={self.g_S8.std():.4f}")

    def orient(self, oo, ot):
        u = unitize(oo)
        if roc_auc_score(self.y[self.te0], self.z_s8[self.te0] + 0.01 * u[self.te0]) < roc_auc_score(
            self.y[self.te0], self.z_s8[self.te0] - 0.01 * u[self.te0]
        ):
            return -u, -unitize(ot)
        return unitize(u), unitize(ot)

    def eval_family(self, name, vo, vt, ws=(0.006, 0.008, 0.010, 0.012, 0.016, 0.020)):
        oo, ot = zca_orth(vo, vt, self.Bo_ext[:, :12], self.Bt12, self.folds)
        # also strip fail-rays lightly on test via OOF-unavailable — check c6/c7 only
        u, ut = self.orient(oo, ot)
        g12 = forensic(u, ut, self.Bt12)
        g9 = forensic(u, ut, self.Bt9)
        f3 = abs(float(np.corrcoef(unitize(ut), unitize(L(self.slot8) - L(self.slot5)))[0, 1]))
        best = None
        for w in ws:
            oof = {b: self.bases[b] + w * u for b in self.bases}
            per = {b: h2h(oof[b], self.bases[b], self.y, self.folds) for b in self.bases}
            cand = qlock(
                L(self.slot3) + W8_O * self.ut_r2 + W8_C * self.ut_cat + W8_U * self.ut_u5 + w * ut,
                self.lv,
            )
            rho = one_m_rho_e4(cand, self.slot8)
            delta = unitize(L(cand) - L(self.slot8))
            c6 = float(np.corrcoef(delta, self.bad6)[0, 1])
            c7 = float(np.corrcoef(delta, self.bad7)[0, 1])
            ok_h = survivor_ok(per, g9, g12, c6, c7)
            ok_g = hard_gates(per, rho, c6, c7, g9, g12, f3)
            row = {
                "name": f"{name}_w{w:.3f}",
                "family": name,
                "w": w,
                "u": u,
                "ut": ut,
                "per": per,
                "rho": rho,
                "c6": c6,
                "c7": c7,
                "g9": g9,
                "g12": g12,
                "f3": f3,
                "cand": cand,
                "ok_survivor": ok_h,
                "ok_gates": ok_g,
                "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                "min_lift": float(np.min([per[b]["min"] for b in per])),
            }
            if ok_g or (ok_h and per["z_S8"]["pos"] >= 4):
                mark = "★" if ok_g else "~"
                log(
                    f"{mark} {row['name']}: ρ={rho:.3f} c7={c7:+.3f} F2={g12['F2']:.3f} "
                    f"z={per['z_S8']['pos']}/5 m={per['z_S8']['mean']:+.2f} t={per['z_S8']['t']:.2f} "
                    f"sf={per['z_S8']['sf25']} gates={ok_g}"
                )
            if ok_h and (best is None or row["min_lift"] > best["min_lift"]):
                best = row
            if ok_g:
                self.promoted.append(row)
        if best is not None:
            self.vault_add(name, best)
        return best

    def vault_add(self, name, row):
        self.vault[name] = {
            "oof": row["u"],
            "test": row["ut"],
            "meta": {
                "name": row["name"],
                "w": row["w"],
                "rho": row["rho"],
                "c6": row["c6"],
                "c7": row["c7"],
                "f2": row["g12"]["F2"],
                "disp": row["g12"]["raw_disp"],
                "KS": row["g12"]["raw_KS"],
                "mean_lift": row["mean_lift"],
                "min_lift": row["min_lift"],
                "per": row["per"],
                "f3": row["f3"],
            },
        }
        log(f"VAULT +{name} (n={len(self.vault)})")

    def persist_vault(self):
        if not self.vault:
            return
        payload = {}
        metas = {}
        for k, v in self.vault.items():
            payload[f"{k}_oof"] = v["oof"]
            payload[f"{k}_test"] = v["test"]
            metas[k] = scrub(v["meta"])
        np.savez_compressed(VAULT, **payload)
        OUT.joinpath("gm_survivor_vault_meta.json").write_text(json.dumps(metas, indent=2))
        log(f"WROTE vault n={len(self.vault)} → {VAULT}")

    def event1(self):
        log("=== EVENT 1: Recon + Forgotten Heavyweight Harvest ===")
        recon = kaggle_recon()
        self.state["events"]["E1_recon"] = scrub({"n_datasets": len(recon["datasets"]), "n_kernels": len(recon["kernels"])})

        streams = {}

        def add(name, o, t):
            o, t = np.asarray(o, float).ravel(), np.asarray(t, float).ravel()
            if len(o) != self.n_tr or len(t) != self.n_te:
                return
            if roc_auc_score(self.y[self.te0], o[self.te0]) < 0.5:
                o, t = -o, -t
            if roc_auc_score(self.y[self.te0], o[self.te0]) <= 0.50:
                return
            streams[name] = (o, t)

        if all((MED / f"oof_{f}.npy").exists() for f in range(5)):
            go, gt = assemble_gpt2(MED, self.y, self.folds)
            add("gpt2_med_4ep_5f", go, gt)
            add("gpt2_med_minus_LM_Bpf", *bpf_contrast(go, gt, self.XO["megayak_5f_lm"], self.XT["megayak_5f_lm"], self.fold_id))
            add("gpt2_med_minus_MS_Bpf", *bpf_contrast(go, gt, self.ms_o, self.ms_t, self.fold_id))
            self.gpt2_med = (go, gt)
        if all((REV / f"oof_{f}.npy").exists() for f in range(5)):
            ro, rt = assemble_gpt2(REV, self.y, self.folds)
            add("gpt2_rev_5f", ro, rt)
        for stem in [
            "v10_ridge_resid", "v10_ridge_resid_orth_LM", "v10_cb_d4_bag3", "v10_cb_d4_bag3_orth_LM",
            "v10_combo3_orth_LM", "v10_lgb_pure_fe", "v10_lgb_d3_bag5", "v10_bulsah_clean_orth_LM",
            "soft_diff_orth_LM",
        ]:
            p = load_pair(V10 / f"{stem}_oof.npy", V10 / f"{stem}_test.npy")
            if p:
                add(stem, *p)
        for stem, po, pt in [
            ("P_super_v4p_hc", PRED / "P_super_v4p_hc_oof.npy", PRED / "P_super_v4p_hc_test.npy"),
            ("TabM_5f", HARV / "oof_tabm_5f.npy", HARV / "test_tabm_5f.npy"),
            ("xgb_llr", PRED / "paradigm_B/xgb_llr_oof.npy", PRED / "paradigm_B/xgb_llr_test.npy"),
            ("P4_L4.2", MV2 / "P4_L4.2_pure5f_oof.npy", MV2 / "P4_L4.2_pure5f_test.npy"),
            ("P4_L4.5", MV2 / "P4_L4.5_pure5f_oof.npy", MV2 / "P4_L4.5_pure5f_test.npy"),
            ("P4_L4.2_ext", MV2 / "P4_L4.2_ext_oof.npy", MV2 / "P4_L4.2_ext_test.npy"),
        ]:
            p = load_pair(po, pt)
            if p:
                add(stem, *p)
                if stem.startswith("P4"):
                    setattr(self, stem.replace(".", "_"), p)
        for n in ["sadam_v4_cat", "daoviet_gpdvae", "alex_glm_gen", "sadam_v6_gb_xgb", "sadam_v6_gb_lgb"]:
            if n in self.XO:
                add(f"Q17_{n}", self.XO[n], self.XT[n])
        for tag in ["15d0d127372ea048", "21b4d1316c55b429"]:
            p = load_pair(ALEX / f"oof_{tag}.npy", ALEX / f"test_{tag}.npy")
            if p:
                add(f"alex_{tag[:8]}", *p)
        for npz_name, keys in [
            ("v20_0_families.npz", ["W1_orth9", "W2_orth9", "W4b_bce_orth9", "W12_orth9"]),
            ("v19_13_families.npz", ["P1_raw_only", "P3b_orth_va", "M4_clean"]),
            ("v19_12_families.npz", ["U6_only", "U7_only", "U67", "M_mod_blend"]),
        ]:
            path = OUT / npz_name
            if not path.exists():
                continue
            Z = np.load(path)
            for key in keys:
                ok, tk = f"{key}_oof", f"{key}_test"
                if ok in Z.files and tk in Z.files:
                    add(f"cache_{key}", Z[ok], Z[tk])

        log(f"E1 harvested {len(streams)} streams")
        for name, (o, t) in streams.items():
            try:
                self.eval_family(name, o, t)
            except Exception as e:
                log(f"E1 eval fail {name}: {e}")
        self.persist_vault()
        self.state["events"]["E1"] = {"n_streams": len(streams), "vault": len(self.vault), "promoted": len(self.promoted)}
        save_report()

    def build_X_stageB(self):
        log("=== EVENT 2: Build X_stageB_v20 ===")
        train = pd.read_csv(DATA / "train.csv")
        test = pd.read_csv(DATA / "test.csv")
        self.train, self.test = train, test
        llr_o = (self.raw_o[:, :, 1] - self.raw_o[:, :, 0]).mean(1)
        llr_t = np.mean(
            [(self.raw_t[f, :, :, 1] - self.raw_t[f, :, :, 0]).mean(1) for f in range(5)], 0
        )
        cols_o, cols_t, names = [], [], []

        def push2(name, o, t):
            oo = np.asarray(o, float).ravel()
            tt = np.asarray(t, float).ravel()
            cols_o.append(unitize(L_fold(oo, self.fold_id)))
            cols_t.append(unitize(L(tt)))
            names.append(name)

        push2("MS", self.ms_o, self.ms_t)
        push2("LM", self.XO["megayak_5f_lm"], self.XT["megayak_5f_lm"])
        push2("L7F", self.XO["lm7_5fold"], self.XT["lm7_5fold"])
        push2("L7_s2026", llr_o, llr_t)
        if hasattr(self, "gpt2_med"):
            push2("gpt2_med", *self.gpt2_med)
        push2("Family_E", self.dE_o, self.dE_t)
        push2("GPD", self.XO["daoviet_gpdvae"], self.XT["daoviet_gpdvae"])
        push2("GB", self.XO["gb_bag5"], self.XT["gb_bag5"])
        push2("MLP", self.XO["yekenot_realmlp"], self.XT["yekenot_realmlp"])
        push2("R2", self.r2_raw_o, self.r2_raw_t)
        push2("CatObliv", self.cat_raw_o, self.cat_raw_t)
        for k in ["U5", "U6", "U7", "U8"]:
            push2(k, self.U_o[k], self.U_t[k])
        for nm in ["P4_L4.2_pure5f", "P4_L4.5_pure5f"]:
            p = load_pair(MV2 / f"{nm}_oof.npy", MV2 / f"{nm}_test.npy")
            if p:
                push2(nm, *p)
        for c in ("amount_usd", "velocity_score", "merchant_risk_score", "distance_from_home_km"):
            vtr, vte = train[c].to_numpy(float), test[c].to_numpy(float)
            push2(
                f"{c}_cents",
                (np.round(vtr * 100).astype(np.int64) % 100).astype(float),
                (np.round(vte * 100).astype(np.int64) % 100).astype(float),
            )
            push2(c, vtr, vte)

        self.Xo_sb = np.column_stack(cols_o).astype(np.float32)
        self.Xt_sb = np.column_stack(cols_t).astype(np.float32)
        self.sb_names = names
        self.lm_idx = [
            i for i, n in enumerate(names) if n in ("MS", "LM", "L7F", "L7_s2026", "gpt2_med", "Family_E")
        ]
        log(f"X_stageB shape={self.Xo_sb.shape} names={names}")
        return self.Xo_sb, self.Xt_sb

    def event2(self):
        log("=== EVENT 2: Standalone Stage-B families ===")
        Xo, Xt = self.build_X_stageB()
        y = self.y

        # SB1 CatBoost oblivious classifiers (reduced for CPU)
        oof1 = np.zeros(self.n_tr)
        te1 = np.zeros(self.n_te)
        cfgs = [(3, 11), (4, 21), (5, 31)]
        for depth, seed in cfgs:
            fo = np.zeros(self.n_tr)
            ft = []
            for tr, te in self.folds:
                model = CatBoostClassifier(
                    depth=depth, l2_leaf_reg=35, iterations=120, learning_rate=0.035,
                    random_seed=seed, verbose=False, allow_writing_files=False,
                    grow_policy="SymmetricTree",
                )
                model.fit(Xo[tr], y[tr])
                fo[te] = model.predict_proba(Xo[te])[:, 1]
                ft.append(model.predict_proba(Xt)[:, 1])
            oof1 += fo
            te1 += np.mean(ft, 0)
        oof1 /= len(cfgs)
        te1 /= len(cfgs)
        self.eval_family("SB1_cat_obliv", oof1, te1)

        # SB2 LGBM monotone on LM cols
        mono = [1 if i in self.lm_idx else 0 for i in range(Xo.shape[1])]
        oof2 = np.zeros(self.n_tr)
        te2 = np.zeros(self.n_te)
        cfgs2 = [(6, 11), (8, 21), (6, 31), (8, 41)]
        for leaves, seed in cfgs2:
            fo = np.zeros(self.n_tr)
            ft = []
            for tr, te in self.folds:
                model = lgb.LGBMClassifier(
                    num_leaves=leaves, min_child_samples=600, colsample_bytree=0.55,
                    subsample=0.80, reg_lambda=40, n_estimators=120, learning_rate=0.03,
                    monotone_constraints=mono, random_state=seed, verbosity=-1, n_jobs=4,
                )
                model.fit(Xo[tr], y[tr])
                fo[te] = model.predict_proba(Xo[te])[:, 1]
                ft.append(model.predict_proba(Xt)[:, 1])
            oof2 += fo
            te2 += np.mean(ft, 0)
        oof2 /= len(cfgs2)
        te2 /= len(cfgs2)
        self.eval_family("SB2_lgb_monotone", oof2, te2)

        # SB3 ridge poly2 on subset
        idx_lm = self.lm_idx
        idx_tab = [i for i, n in enumerate(self.sb_names) if n in ("GPD", "GB", "MLP", "R2", "CatObliv")]
        idx_u = [i for i, n in enumerate(self.sb_names) if n.startswith("U")]
        idx_p4 = [i for i, n in enumerate(self.sb_names) if n.startswith("P4")]
        use = sorted(set(idx_lm + idx_tab + idx_u + idx_p4))[:16]
        Xo_s = col_z(Xo[:, use])
        Xt_s = col_z(Xt[:, use])
        # LM - TAB contrast
        if idx_lm and idx_tab:
            lm_o = Xo[:, idx_lm].mean(1)
            tab_o = Xo[:, idx_tab].mean(1)
            lm_t = Xt[:, idx_lm].mean(1)
            tab_t = Xt[:, idx_tab].mean(1)
            Xo_s = np.column_stack([Xo_s, unitize(lm_o - tab_o)])
            Xt_s = np.column_stack([Xt_s, unitize(lm_t - tab_t)])
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        oof3 = np.zeros(self.n_tr)
        te3s = []
        for tr, te in self.folds:
            Ptr = poly.fit_transform(Xo_s[tr])
            Pva = poly.transform(Xo_s[te])
            Pte = poly.transform(Xt_s)
            # cap dims
            if Ptr.shape[1] > 120:
                Ptr, Pva, Pte = Ptr[:, :120], Pva[:, :120], Pte[:, :120]
            lr = Ridge(alpha=2000.0, fit_intercept=True)
            lr.fit(Ptr, y[tr])
            oof3[te] = lr.predict(Pva)
            te3s.append(lr.predict(Pte))
        te3 = np.mean(te3s, 0)
        self.eval_family("SB3_ridge_poly2", oof3, te3)

        self.persist_vault()
        self.state["events"]["E2"] = {"vault": len(self.vault), "promoted": len(self.promoted)}
        save_report()

    def event3(self):
        log("=== EVENT 3: HierEB + orig20k drift ===")
        train, test = self.train, self.test
        y = self.y
        # 3A orig20k drift on a few keys
        keys_tr, keys_te = {}, {}
        for c in ("amount_usd", "velocity_score", "merchant_risk_score"):
            keys_tr[f"{c}_cents"] = (np.round(train[c].to_numpy(float) * 100).astype(np.int64) % 100)
            keys_te[f"{c}_cents"] = (np.round(test[c].to_numpy(float) * 100).astype(np.int64) % 100)
            keys_tr[f"{c}_tenth"] = (np.round(train[c].to_numpy(float) * 10).astype(np.int64) % 10)
            keys_te[f"{c}_tenth"] = (np.round(test[c].to_numpy(float) * 10).astype(np.int64) % 10)
        for cat in ("merchant_category", "channel", "device_type"):
            codes, _ = pd.factorize(pd.concat([train[cat], test[cat]], ignore_index=True))
            keys_tr[cat] = codes[: self.n_tr]
            keys_te[cat] = codes[self.n_tr :]

        orig_rates = {}
        if ORIG.exists():
            try:
                orig = pd.read_csv(ORIG)
                # use Amount if present
                ac = [c for c in orig.columns if c.lower() in ("amount", "amt")]
                if ac and "Class" in orig.columns:
                    oa = np.round(orig[ac[0]].to_numpy(float) * 100).astype(np.int64) % 100
                    oy = orig["Class"].to_numpy(float)
                    for k in range(100):
                        m = oa == k
                        if m.sum() >= 5:
                            orig_rates[k] = float(oy[m].mean())
            except Exception as e:
                log(f"orig20k parse skip: {e}")

        # δ_synth on amount cents
        oof_d = np.zeros(self.n_tr)
        te_d = np.zeros(self.n_te)
        key = keys_tr["amount_usd_cents"]
        key_te = keys_te["amount_usd_cents"]
        for tr, te in self.folds:
            # empirical rates on tr
            rates = {}
            for k in range(100):
                m = key[tr] == k
                if m.sum() >= 10:
                    rates[k] = float(y[tr][m].mean())
            prior = float(y[tr].mean())
            def score(keys_arr):
                out = np.zeros(len(keys_arr))
                for i, k in enumerate(keys_arr):
                    p_tr = rates.get(int(k), prior)
                    p_or = orig_rates.get(int(k), prior)
                    out[i] = logit(np.clip(p_tr, 1e-4, 1 - 1e-4)) - logit(np.clip(p_or, 1e-4, 1 - 1e-4))
                return out
            # target g_S8 with weight h — ridge on single col is just scaling; use as feature
            xo = score(key[tr]).reshape(-1, 1)
            lr = Ridge(alpha=1500.0).fit(xo, self.g_S8[tr], sample_weight=self.h_S8[tr])
            oof_d[te] = lr.predict(score(key[te]).reshape(-1, 1))
            te_d += lr.predict(score(key_te).reshape(-1, 1))
        te_d /= 5
        self.eval_family("E3A_orig20k_drift", oof_d, te_d)

        # 3B Hierarchical epistatic excess — top crosses
        crosses = [
            ("amount_usd_cents", "merchant_category"),
            ("amount_usd_cents", "channel"),
            ("velocity_score_tenth", "merchant_category"),
            ("amount_usd_tenth", "device_type"),
            ("velocity_score_cents", "channel"),
            ("merchant_risk_score_tenth", "merchant_category"),
            ("amount_usd_cents", "device_type"),
            ("velocity_score_tenth", "channel"),
        ]
        # ensure tenth keys exist for merchant_risk
        for c in ("merchant_risk_score",):
            keys_tr[f"{c}_tenth"] = (np.round(train[c].to_numpy(float) * 10).astype(np.int64) % 10)
            keys_te[f"{c}_tenth"] = (np.round(test[c].to_numpy(float) * 10).astype(np.int64) % 10)
            keys_tr[f"{c}_cents"] = (np.round(train[c].to_numpy(float) * 100).astype(np.int64) % 100)
            keys_te[f"{c}_cents"] = (np.round(test[c].to_numpy(float) * 100).astype(np.int64) % 100)

        feats_o = []
        feats_t = []
        for a, b in crosses:
            if a not in keys_tr or b not in keys_tr:
                continue
            ka, kb = keys_tr[a].astype(np.int64), keys_tr[b].astype(np.int64)
            kat, kbt = keys_te[a].astype(np.int64), keys_te[b].astype(np.int64)
            # pack pair key
            ka_max = int(max(ka.max(), kat.max()) + 1)
            pack = lambda x, y: x * (int(max(kb.max(), kbt.max()) + 1) + 1) + y
            # actually use stable packing
            bmod = int(max(kb.max(), kbt.max()) + 1)
            pk = ka * bmod + kb
            pkt = kat * bmod + kbt
            fo = np.zeros(self.n_tr)
            ft_folds = []
            for tr, te in self.folds:
                ybar = float(y[tr].mean())
                # 1D rates via bincount
                def rate_map(keys):
                    mx = int(keys.max()) + 1
                    cnt = np.bincount(keys, minlength=mx).astype(np.float64)
                    sm = np.bincount(keys, weights=y[tr], minlength=mx)
                    rates = np.full(mx, ybar)
                    m = cnt >= 10
                    rates[m] = sm[m] / cnt[m]
                    return rates

                ra = rate_map(ka[tr])
                rb = rate_map(kb[tr])
                # 2D counts on tr
                mxp = int(pk[tr].max()) + 1
                cnt2 = np.bincount(pk[tr], minlength=mxp).astype(np.float64)
                sm2 = np.bincount(pk[tr], weights=y[tr], minlength=mxp)
                p2 = np.full(mxp, ybar)
                m2 = cnt2 >= 10
                p2[m2] = sm2[m2] / cnt2[m2]

                def excess_keys(ka_arr, kb_arr, pk_arr):
                    pa = ra[np.clip(ka_arr, 0, len(ra) - 1)]
                    pb = rb[np.clip(kb_arr, 0, len(rb) - 1)]
                    eta = (
                        logit(np.clip(pa, 1e-4, 1 - 1e-4))
                        + logit(np.clip(pb, 1e-4, 1 - 1e-4))
                        - logit(np.clip(ybar, 1e-4, 1 - 1e-4))
                    )
                    pkc = np.clip(pk_arr, 0, mxp - 1)
                    n = cnt2[pkc]
                    ex = (p2[pkc] - expit(eta)) * n / (n + 25.0)
                    ex[n < 10] = 0.0
                    return ex

                fo[te] = excess_keys(ka[te], kb[te], pk[te])
                ft_folds.append(excess_keys(kat, kbt, pkt))
            ft = np.mean(ft_folds, 0)
            feats_o.append(unitize(fo))
            feats_t.append(unitize(ft))
            log(f"E3B cross {a}×{b} done")

        if feats_o:
            Fo = np.column_stack(feats_o)
            Ft = np.column_stack(feats_t)
            oof = np.zeros(self.n_tr)
            tes = []
            for tr, te in self.folds:
                lr = Ridge(alpha=2000.0).fit(Fo[tr], self.g_S8[tr], sample_weight=self.h_S8[tr])
                oof[te] = lr.predict(Fo[te])
                tes.append(lr.predict(Ft))
            # also small catboost on excess
            oof_c = np.zeros(self.n_tr)
            tec = []
            for tr, te in self.folds:
                model = CatBoostRegressor(
                    depth=3, iterations=80, learning_rate=0.05, l2_leaf_reg=20,
                    random_seed=42, verbose=False, allow_writing_files=False,
                    grow_policy="SymmetricTree",
                )
                model.fit(Fo[tr], self.g_S8[tr], sample_weight=self.h_S8[tr])
                oof_c[te] = model.predict(Fo[te])
                tec.append(model.predict(Ft))
            self.eval_family("E3B_HierEB_ridge", oof, np.mean(tes, 0))
            self.eval_family("E3B_HierEB_cat", oof_c, np.mean(tec, 0))
            blend_o = unitize(0.5 * unitize(oof) + 0.5 * unitize(oof_c))
            blend_t = unitize(0.5 * unitize(np.mean(tes, 0)) + 0.5 * unitize(np.mean(tec, 0)))
            self.eval_family("E3B_HierEB_blend", blend_o, blend_t)

        self.persist_vault()
        self.state["events"]["E3"] = {"vault": len(self.vault), "promoted": len(self.promoted)}
        save_report()

    def event4(self):
        log("=== EVENT 4: Fourier GLM + Residual MLP ===")
        train, test = self.train, self.test
        # Rank-Gauss tab + Fourier
        cols_o, cols_t = [], []
        for c in ("amount_usd", "velocity_score", "merchant_risk_score", "distance_from_home_km", "account_balance_usd"):
            vtr, vte = train[c].to_numpy(float), test[c].to_numpy(float)
            rtr = (rankdata(vtr) - 0.5) / len(vtr)
            rte = (rankdata(vte) - 0.5) / len(vte)
            cols_o.append(norm.ppf(np.clip(rtr, 1e-6, 1 - 1e-6)))
            cols_t.append(norm.ppf(np.clip(rte, 1e-6, 1 - 1e-6)))
            cents_tr = (np.round(vtr * 100).astype(np.int64) % 100).astype(float)
            cents_te = (np.round(vte * 100).astype(np.int64) % 100).astype(float)
            for p in (100, 50, 25, 10, 5):
                cols_o.append(np.cos(2 * np.pi * cents_tr / p))
                cols_o.append(np.sin(2 * np.pi * cents_tr / p))
                cols_t.append(np.cos(2 * np.pi * cents_te / p))
                cols_t.append(np.sin(2 * np.pi * cents_te / p))
        for k in ["U5", "U6", "U7", "U8"]:
            cols_o.append(unitize(self.U_o[k]))
            cols_t.append(unitize(self.U_t[k]))
        for nm in ["P4_L4.2_pure5f", "P4_L4.5_pure5f"]:
            p = load_pair(MV2 / f"{nm}_oof.npy", MV2 / f"{nm}_test.npy")
            if p:
                cols_o.append(unitize(p[0]))
                cols_t.append(unitize(p[1]))
        Xo = col_z(np.column_stack(cols_o))
        Xt = col_z(np.column_stack(cols_t))

        # N1 multi-alpha ridge ensemble on g_S8
        oof = np.zeros(self.n_tr)
        tes = np.zeros(self.n_te)
        alphas = (300.0, 1000.0, 3000.0)
        for a in alphas:
            fo = np.zeros(self.n_tr)
            ft = []
            for tr, te in self.folds:
                lr = Ridge(alpha=a).fit(Xo[tr], self.g_S8[tr], sample_weight=self.h_S8[tr])
                fo[te] = lr.predict(Xo[te])
                ft.append(lr.predict(Xt))
            oof += fo
            tes += np.mean(ft, 0)
        oof /= len(alphas)
        tes /= len(alphas)
        self.eval_family("N1_fourier_glm", oof, tes)

        # N2 residual MLP (PyTorch MPS/CPU) — compact
        try:
            import torch
            import torch.nn as nn

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            log(f"N2 MLP device={device}")

            class ResMLP(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(d, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.35),
                        nn.Linear(64, 16), nn.SiLU(), nn.Linear(16, 1),
                    )

                def forward(self, x):
                    return self.net(x).squeeze(-1)

            oof_m = np.zeros(self.n_tr)
            te_acc = []
            seeds = (11, 21, 31)
            for seed in seeds:
                torch.manual_seed(seed)
                fo = np.zeros(self.n_tr)
                ft = []
                for tr, te in self.folds:
                    model = ResMLP(Xo.shape[1]).to(device)
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
                    xb = torch.tensor(Xo[tr], dtype=torch.float32, device=device)
                    yb = torch.tensor(self.g_S8[tr], dtype=torch.float32, device=device)
                    wb = torch.tensor(self.h_S8[tr], dtype=torch.float32, device=device)
                    model.train()
                    n = len(tr)
                    for epoch in range(10):
                        # cosine lr
                        lr = 1e-3 * 0.5 * (1 + np.cos(np.pi * epoch / 10))
                        for g in opt.param_groups:
                            g["lr"] = lr
                        perm = np.random.permutation(n)
                        for i in range(0, n, 8192):
                            idx = perm[i : i + 8192]
                            opt.zero_grad()
                            pred = model(xb[idx])
                            loss = (wb[idx] * (pred - yb[idx]) ** 2).mean()
                            loss.backward()
                            opt.step()
                    model.eval()
                    with torch.no_grad():
                        fo[te] = model(torch.tensor(Xo[te], dtype=torch.float32, device=device)).cpu().numpy()
                        ft.append(model(torch.tensor(Xt, dtype=torch.float32, device=device)).cpu().numpy())
                oof_m += fo
                te_acc.append(np.mean(ft, 0))
            oof_m /= len(seeds)
            te_m = np.mean(te_acc, 0)
            self.eval_family("N2_res_mlp", oof_m, te_m)
        except Exception as e:
            log(f"N2 MLP skipped: {e}")
            traceback.print_exc()

        self.persist_vault()
        self.state["events"]["E4"] = {"vault": len(self.vault), "promoted": len(self.promoted)}
        save_report()

    def event5(self):
        log("=== EVENT 5: Multi-Vault Variance-Canceling Synthesis ===")
        if len(self.vault) < 2:
            log("E5 vault too small — skip solvers")
            self.state["events"]["E5"] = {"vault": len(self.vault), "note": "too_small"}
            return
        names = list(self.vault.keys())
        Uo = np.column_stack([self.vault[n]["oof"] for n in names])
        Ut = np.column_stack([self.vault[n]["test"] for n in names])
        K = len(names)
        # correlation matrix
        C = np.corrcoef(Uo.T)
        log("E5 pairwise |corr| mean={:.3f} max={:.3f}".format(
            float(np.mean(np.abs(C - np.eye(K)))), float(np.max(np.abs(C - np.eye(K))))
        ))
        # fold lift matrix
        for i, n in enumerate(names):
            meta = self.vault[n]["meta"]
            log(f"  vault[{n}]: mean={meta['mean_lift']:+.2f} min={meta['min_lift']:+.2f} F2={meta['f2']:.3f} c7={meta['c7']:+.3f}")

        # prune to low-corr diverse set
        keep = []
        for i, n in enumerate(names):
            if all(abs(C[i, j]) < 0.55 for j in keep):
                keep.append(i)
            if len(keep) >= 8:
                break
        if len(keep) < 2:
            keep = list(range(min(5, K)))
        names_k = [names[i] for i in keep]
        Uo = Uo[:, keep]
        Ut = Ut[:, keep]
        K = len(keep)
        log(f"E5 synthesis on K={K}: {names_k}")

        rng = np.random.default_rng(42)

        def eval_alpha(alpha, w=0.012):
            alpha = np.asarray(alpha, float)
            alpha = np.clip(alpha, 0, 0.45)
            if alpha.sum() < 1e-12:
                return None
            alpha = alpha / alpha.sum()
            u = unitize(Uo @ alpha)
            ut = unitize(Ut @ alpha)
            # orient
            u, ut = self.orient(u, ut)
            g12 = forensic(u, ut, self.Bt12)
            g9 = forensic(u, ut, self.Bt9)
            f3 = abs(float(np.corrcoef(unitize(ut), unitize(L(self.slot8) - L(self.slot5)))[0, 1]))
            best = None
            for ww in (w,) if isinstance(w, float) else w:
                oof = {b: self.bases[b] + ww * u for b in self.bases}
                per = {b: h2h(oof[b], self.bases[b], self.y, self.folds) for b in self.bases}
                cand = qlock(
                    L(self.slot3) + W8_O * self.ut_r2 + W8_C * self.ut_cat + W8_U * self.ut_u5 + ww * ut,
                    self.lv,
                )
                rho = one_m_rho_e4(cand, self.slot8)
                delta = unitize(L(cand) - L(self.slot8))
                c6 = float(np.corrcoef(delta, self.bad6)[0, 1])
                c7 = float(np.corrcoef(delta, self.bad7)[0, 1])
                ok = hard_gates(per, rho, c6, c7, g9, g12, f3)
                row = {
                    "name": f"E5_synth_K{K}_w{ww:.3f}",
                    "family": "E5_synth",
                    "w": ww,
                    "alpha": alpha.tolist(),
                    "names": names_k,
                    "u": u, "ut": ut, "per": per, "rho": rho, "c6": c6, "c7": c7,
                    "g9": g9, "g12": g12, "f3": f3, "cand": cand, "ok_gates": ok,
                    "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                    "min_lift": float(np.min([per[b]["min"] for b in per])),
                }
                if best is None or row["min_lift"] > best["min_lift"]:
                    best = row
                if ok:
                    self.promoted.append(row)
                    log(f"★ {row['name']}: ρ={rho:.3f} c7={c7:+.3f} min={row['min_lift']:+.2f} t={per['z_S8']['t']:.2f}")
            return best

        # Solver A: random simplex maximin search (LOFO-style: fit alpha on 4 folds via proxy)
        best_a = None
        for _ in range(80):
            a = rng.dirichlet(np.ones(K))
            a = np.clip(a, 0, 0.45)
            a = a / a.sum()
            # score via fold-floor on z_S8 quickly
            u = unitize(Uo @ a)
            u, _ = self.orient(u, unitize(Ut @ a))
            score = 0.0
            for bname, base in self.bases.items():
                folds_d = []
                for _, te in self.folds:
                    folds_d.append(
                        (roc_auc_score(self.y[te], (base + 0.012 * u)[te]) - roc_auc_score(self.y[te], base[te])) * 1e5
                    )
                score += float(np.mean(folds_d) + 0.5 * np.min(folds_d))
            if best_a is None or score > best_a[0]:
                best_a = (score, a)
        if best_a:
            log(f"E5 SolverA score={best_a[0]:.3f} alpha={np.round(best_a[1],3)}")
            row = eval_alpha(best_a[1], w=(0.008, 0.010, 0.012, 0.016, 0.020))
            if row and survivor_ok(row["per"], row["g9"], row["g12"], row["c6"], row["c7"]):
                self.vault_add("E5_solverA", row)

        # Solver B: equal-weight + top-3 equal
        for mask_n in [K, min(3, K), min(4, K)]:
            a = np.zeros(K)
            # pick top mask_n by mean_lift
            order = np.argsort([-self.vault[names_k[i]]["meta"]["mean_lift"] for i in range(K)])
            a[order[:mask_n]] = 1.0 / mask_n
            row = eval_alpha(a, w=(0.008, 0.010, 0.012, 0.016))
            if row:
                log(f"E5 SolverB top{mask_n}: min={row['min_lift']:+.2f} ρ={row['rho']:.3f} gates={row['ok_gates']}")
                if survivor_ok(row["per"], row["g9"], row["g12"], row["c6"], row["c7"]):
                    self.vault_add(f"E5_solverB_top{mask_n}", row)

        # Solver C: joint (wr2,wcat,wu5,wv) coarse grid
        if K >= 1:
            u = unitize(Uo.mean(1))
            ut = unitize(Ut.mean(1))
            u, ut = self.orient(u, ut)
            g12 = forensic(u, ut, self.Bt12)
            g9 = forensic(u, ut, self.Bt9)
            f3 = abs(float(np.corrcoef(unitize(ut), unitize(L(self.slot8) - L(self.slot5)))[0, 1]))
            for wr2, wcat, wu5, ww in product((0.014, 0.016), (0.010, 0.012), (0.010, 0.012), (0.006, 0.008, 0.010, 0.012)):
                oof = {
                    "z_S8": self.z_s3 + wr2 * self.u_r2 + wcat * self.u_cat + wu5 * self.u_u5 + ww * u,
                    "twin": self.z_twin + wr2 * self.u_r2 + wcat * self.u_cat + wu5 * self.u_u5 + ww * u,
                    "nnls": self.z_nnls + wr2 * self.u_r2 + wcat * self.u_cat + wu5 * self.u_u5 + ww * u,
                }
                per = {b: h2h(oof[b], self.bases[b], self.y, self.folds) for b in self.bases}
                cand = qlock(
                    L(self.slot3) + wr2 * self.ut_r2 + wcat * self.ut_cat + wu5 * self.ut_u5 + ww * ut, self.lv
                )
                rho = one_m_rho_e4(cand, self.slot8)
                delta = unitize(L(cand) - L(self.slot8))
                c6 = float(np.corrcoef(delta, self.bad6)[0, 1])
                c7 = float(np.corrcoef(delta, self.bad7)[0, 1])
                ok = hard_gates(per, rho, c6, c7, g9, g12, f3)
                if ok or (per["z_S8"]["pos"] == 5 and per["z_S8"]["min"] > 0.05 and 0.28 <= rho <= 1.10):
                    log(
                        f"{'★' if ok else '~'} E5C r{wr2:.3f}_c{wcat:.3f}_u{wu5:.3f}_w{ww:.3f}: "
                        f"ρ={rho:.3f} c7={c7:+.3f} z={per['z_S8']['pos']}/5 m={per['z_S8']['mean']:+.2f} "
                        f"t={per['z_S8']['t']:.2f} gates={ok}"
                    )
                if ok:
                    self.promoted.append({
                        "name": f"E5C_r{wr2:.3f}_c{wcat:.3f}_u{wu5:.3f}_w{ww:.3f}",
                        "family": "E5C", "w": ww, "per": per, "rho": rho, "c6": c6, "c7": c7,
                        "g9": g9, "g12": g12, "f3": f3, "cand": cand, "ok_gates": True,
                        "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                        "min_lift": float(np.min([per[b]["min"] for b in per])),
                        "u": u, "ut": ut,
                    })

        self.persist_vault()
        self.state["events"]["E5"] = {"vault": len(self.vault), "promoted": len(self.promoted)}
        save_report()

    def event6(self):
        log("=== EVENT 6: Autonomous Promotion ===")
        # Deduplicate promoted by name, keep best min_lift
        by = {}
        for r in self.promoted:
            if r["name"] not in by or r["min_lift"] > by[r["name"]]["min_lift"]:
                by[r["name"]] = r
        winners = sorted(by.values(), key=lambda r: (r["min_lift"], r["mean_lift"], -abs(r["rho"] - 0.50)), reverse=True)
        log(f"E6 gate-clear winners: {len(winners)}")
        for w in winners[:10]:
            log(f"  {w['name']}: min={w['min_lift']:+.2f} mean={w['mean_lift']:+.2f} ρ={w['rho']:.3f} c7={w['c7']:+.3f} t={w['per']['z_S8']['t']:.2f}")

        # Stage NOSUBMIT top3 always
        near = sorted(
            [v["meta"] for v in self.vault.values()],
            key=lambda m: (m["per"]["z_S8"]["pos"], m["mean_lift"], -abs(m.get("c7", 1))),
            reverse=True,
        )[:3]
        # Also stage from winners if any
        for i, meta in enumerate(near, 1):
            # rebuild cand from vault
            name = meta["name"].rsplit("_w", 1)[0] if "_w" in meta["name"] else meta["name"]
            # find vault key
            key = None
            for k, v in self.vault.items():
                if v["meta"]["name"] == meta["name"] or k == name:
                    key = k
                    break
            if key is None:
                continue
            w = meta["w"]
            ut = self.vault[key]["test"]
            cand = qlock(
                L(self.slot3) + W8_O * self.ut_r2 + W8_C * self.ut_cat + W8_U * self.ut_u5 + w * ut, self.lv
            )
            path = SUB / f"sub_v20_staged_top{i}_NOSUBMIT.csv"
            pd.DataFrame({"id": self.ids, "is_fraud": cand}).to_csv(path, index=False)
            log(f"STAGED {path.name} ← {meta['name']}")

        submit = False
        pub = None
        ref = None
        chosen = None
        if winners and self.slots_used < self.max_slots:
            chosen = winners[0]
            # Re-verify gates once more
            if not hard_gates(chosen["per"], chosen["rho"], chosen["c6"], chosen["c7"], chosen["g9"], chosen["g12"], chosen["f3"]):
                log("E6 chosen failed re-verify — HOLD")
            else:
                out_csv = SUB / "sub_v20_LIVE_slot9_autonomous_promoted.csv"
                arr = chosen["cand"]
                assert len(arr) == self.n_te and np.isfinite(arr).all()
                # KS vs slot8 after qlock should be ~0
                ks = float(ks_2samp(arr, self.slot8).statistic)
                log(f"E6 KS(cand,slot8)={ks:.6f}")
                pd.DataFrame({"id": self.ids, "is_fraud": arr}).to_csv(out_csv, index=False)
                msg = (
                    f"v20-auto Slot9: {chosen['name']} vs Slot8 0.95395 | "
                    f"H2H mean={chosen['mean_lift']:+.2f}e-5 min={chosen['min_lift']:+.2f} "
                    f"1-ρ={chosen['rho']:.3f}e-4 c7={chosen['c7']:+.3f} F2={chosen['g12']['F2']:.3f} t={chosen['per']['z_S8']['t']:.2f}"
                )
                log(f"SUBMIT {msg}")
                r = subprocess.run(
                    [KAG, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(out_csv), "-m", msg],
                    capture_output=True, text=True, timeout=120,
                )
                log(r.stdout.strip())
                log(r.stderr.strip())
                if r.returncode == 0:
                    submit = True
                    self.slots_used += 1
                    for i in range(40):
                        time.sleep(25 if i else 5)
                        p = subprocess.run(
                            [KAG, "competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
                            capture_output=True, text=True, timeout=60,
                        )
                        lines = [ln for ln in p.stdout.strip().splitlines() if ln]
                        if len(lines) < 2:
                            continue
                        hdr = [h.strip() for h in lines[0].split(",")]
                        for ln in lines[1:6]:
                            cols = ln.split(",")
                            row = {hdr[j]: cols[j].strip() for j in range(min(len(hdr), len(cols)))}
                            if "v20_LIVE_slot9_autonomous" in row.get("fileName", "") or "autonomous_promoted" in row.get("fileName", ""):
                                ref = row.get("ref")
                                st = row.get("status", "")
                                log(f"poll {st} score={row.get('publicScore')}")
                                if "complete" in st.lower() and row.get("publicScore"):
                                    pub = float(row["publicScore"])
                                    break
                        if pub is not None:
                            break
                    self.state["public_lbs"].append(pub)
                    log(f"PUBLIC_LB Slot9={pub} (ATH=0.95395)")

                    # Slot 10 decision
                    if pub is not None and pub >= 0.95396 and self.slots_used < 2:
                        log("NEW ATH — attempting Slot10 step-up")
                        # 1.6x step along same vector
                        w2 = min(chosen["w"] * 1.6, 0.032)
                        ut = chosen["ut"]
                        cand2 = qlock(
                            L(self.slot3) + W8_O * self.ut_r2 + W8_C * self.ut_cat + W8_U * self.ut_u5 + w2 * ut,
                            self.lv,
                        )
                        out2 = SUB / "sub_v20_LIVE_slot10_autonomous_ath_step.csv"
                        pd.DataFrame({"id": self.ids, "is_fraud": cand2}).to_csv(out2, index=False)
                        msg2 = f"v20-auto Slot10: 1.6x step of {chosen['name']} after ATH {pub}"
                        r2 = subprocess.run(
                            [KAG, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(out2), "-m", msg2],
                            capture_output=True, text=True, timeout=120,
                        )
                        log(r2.stdout.strip())
                        self.slots_used += 1
                    elif pub is not None and pub <= 0.95395:
                        log("Slot9 <= ATH — HOLD Slot10 (preserve last slot)")
        else:
            log("NO gate-clear winner OR no slots — HOLD")

        self.state["submit"] = submit
        self.state["chosen"] = None if not chosen else scrub({
            "name": chosen["name"], "mean_lift": chosen["mean_lift"], "min_lift": chosen["min_lift"],
            "rho": chosen["rho"], "c6": chosen["c6"], "c7": chosen["c7"],
            "f2": chosen["g12"]["F2"], "per": chosen["per"],
        })
        self.state["public_lb"] = pub
        self.state["ref"] = ref
        self.state["slots_used"] = self.slots_used
        self.state["elapsed_s"] = time.time() - T0
        OUT.joinpath("gm_v20_factory_state.json").write_text(json.dumps(scrub(self.state), indent=2))

        extra = [
            "## Summary",
            f"- Vault size: {len(self.vault)}",
            f"- Gate-clear winners: {len(winners)}",
            f"- Submit: {submit}",
            f"- Public LB: {pub}",
            f"- Slots used this run: {self.slots_used} (max {self.max_slots})",
            "",
            "## Decision",
        ]
        if submit and pub is not None and pub > 0.95395:
            extra.append(f"**NEW ATH {pub}** via `{chosen['name']}`.")
        elif submit:
            extra.append(f"LIVE miss/tie `{pub}` — ATH remains **0.95395**. Slot10 held.")
        else:
            extra.append("**HOLD** ATH `0.95395`. No candidate cleared all 6 hard gates (t≥2.80, sf≥18, min>+0.05e-5 ×3).")
        save_report("\n".join(extra))
        log("DONE")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    log("GM V20 Autonomous Event-Driven Factory START (0 pods)")
    fac = Factory()
    fac.rebuild_slot8()
    try:
        fac.event1()
        if fac.promoted and fac.slots_used < fac.max_slots:
            log("E1 produced gate-clear — continue for deeper ammo before promote")
        fac.event2()
        fac.event3()
        fac.event4()
        fac.event5()
        fac.event6()
    except Exception as e:
        log(f"FATAL: {e}")
        traceback.print_exc()
        save_report(f"FATAL: {e}")
        raise
    finally:
        fac.persist_vault()
        OUT.joinpath("gm_v20_factory_state.json").write_text(json.dumps(scrub(fac.state), indent=2))
        save_report()


if __name__ == "__main__":
    main()
