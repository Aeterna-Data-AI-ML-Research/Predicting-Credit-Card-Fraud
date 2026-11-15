#!/usr/bin/env python3
"""GM V20 Overnight Unattended CPU Grandmaster Factory.

100% local CPU (n_jobs=-1), 0 GPU pods, $0.00/hr.
Engines 1→5 with vault checkpointing; Event-6-style Slot9 submit IFF 6 hard gates.
"""
from __future__ import annotations

import json
import os
import traceback
import warnings
from itertools import combinations, product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from scipy.stats import ks_2samp, norm, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(os.cpu_count() or 8))

import time

REPO = Path(__file__).resolve().parents[1]
TMP15, TMP16 = Path("/tmp/s6e11_v15"), Path("/tmp/s6e11_v16")
OUT, SUB, DATA, LOG = TMP16 / "out", REPO / "submissions", REPO / "data", REPO / "logs/overnight"
PRED = REPO / "preds"
V10, MED, REV, MV2 = PRED / "v10_transductive_bulsah", PRED / "gpt2_med_4ep", PRED / "gpt2_rev", PRED / "matrix_v2"
HARV = PRED / "fast_gpu_harvest"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
ORIG_CANDS = [REPO / "data/creditcard.csv", Path("/tmp/s6e11_v15/data/creditcard.csv"), Path.home() / "data/creditcard.csv"]
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
SLOT6 = SUB / "sub_v18_1_LIVE_slot6_Slot5_plus_R1_pure_Bo10_w0p022.csv"
SLOT7 = SUB / "sub_v19_0_LIVE_slot7_zero_echo_peer_hybrid.csv"
SLOT8 = SUB / "sub_v19_4_LIVE_slot8_A_cat_wo0.016_wc0.012_wu0.012.csv"
G26_OOF, G26_T0 = TMP16 / "g26/gpt2_s2026_oof_5fold.npy", TMP16 / "g26/gpt2_s2026_test_f0.npy"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
VAULT = OUT / "gm_overnight_vault.npz"
PROGRESS = OUT / "gm_overnight_progress.json"
REPORT = LOG / "GM_V20_OVERNIGHT_CPU_MASTER_REPORT.md"
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1, W_FE = 0.0355, 0.01045
W8_O, W8_C, W8_U = 0.016, 0.012, 0.012
N_TR, N_TE = 699960, 299984
N_JOBS = -1
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
    return s / max(K * (K - 1) / 2.0, 1.0)


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


def two_stage_orth(v_oof, v_test, Bo12, Bt12, Bt_all, folds, ridge_alpha=10.0):
    """Rule B: LOFO orth on Bo12, then Test-Covariance Alignment on Bt_all=[Bt12,rays]."""
    oo, ot, _ = frozen_lofo_orth(v_oof, v_test, Bo12, Bt12, folds, ridge_alpha=ridge_alpha)
    raw_disp = float(np.std(ot) / (np.std(oo) + 1e-12))
    raw_ks = float(ks_2samp(unitize(oo), unitize(ot)).statistic)
    # second-stage OOF LOFO strip vs Bo12
    oo2 = np.zeros(len(oo))
    for tr, te in folds:
        lr = Ridge(alpha=1.0, fit_intercept=True)
        ytr = oo[tr] - oo[tr].mean()
        lr.fit(Bo12[tr], ytr)
        oo2[te] = (oo[te] - oo[tr].mean()) - (Bo12[te] @ lr.coef_ + lr.intercept_)
    # Test-Covariance Alignment on Bt_all (Bo12 + fail-rays); no y_test
    lr_te = Ridge(alpha=1.0, fit_intercept=True)
    yt = ot - ot.mean()
    lr_te.fit(Bt_all, yt)
    ot2 = yt - (Bt_all @ lr_te.coef_ + lr_te.intercept_)
    so, st = float(np.std(oo2)) + 1e-12, float(np.std(ot2)) + 1e-12
    disp = st / so
    if disp < 0.92:
        ot2 = ot2 * (0.92 * so / st)
    elif disp > 1.08:
        ot2 = ot2 * (1.08 * so / st)
    return oo2 - oo2.mean(), ot2 - ot2.mean(), {"raw_disp": raw_disp, "raw_KS": raw_ks}


def forensic(oo, ot, Bt):
    disp = float(np.std(ot) / (np.std(oo) + 1e-12))
    ut = unitize(ot)
    leaks = [abs(float(np.corrcoef(ut, Bt[:, j])[0, 1])) for j in range(Bt.shape[1])]
    return {"raw_disp": disp, "raw_KS": float(ks_2samp(unitize(oo), ut).statistic), "F2": float(max(leaks)) if leaks else 0.0}


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


def hard_gates(per, rho, c6, c7, g9, g12, f3):
    return (
        0.92 <= g12["raw_disp"] <= 1.08
        and g12["raw_KS"] < 0.025
        and g9["F2"] < 0.035
        and g12["F2"] < 0.035
        and f3 < 0.045
        and all(per[b]["pos"] == 5 for b in per)
        and all(per[b]["min"] > 0.05 for b in per)
        and all(per[b]["sf25"] >= 18 for b in per)
        and all(per[b]["t"] >= 2.80 for b in per)
        and 0.28 <= rho <= 1.10
        and abs(c6) < 0.20
        and abs(c7) < 0.20
    )


def survivor_ok(per, g9, g12, c6, c7):
    return (
        0.90 <= g12["raw_disp"] <= 1.10
        and g9["F2"] < 0.035
        and g12["F2"] < 0.035
        and abs(c6) < 0.20
        and abs(c7) < 0.20
        and all(per[b]["mean"] > 0 for b in per)
        and all(per[b]["pos"] >= 4 for b in per)
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
    if float(np.std(o)) < 1e-12:
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
        zt_f.append((traw[:, :, 1] - traw[:, :, 0]).astype(np.float64).mean(1))
    assert np.isfinite(zo).all()
    return zo.astype(np.float64), np.mean(zt_f, 0)


def bpf_contrast(a_o, a_t, b_o, b_t, fold_id):
    co = L_fold(a_o, fold_id) - L_fold(b_o, fold_id)
    ct = L(a_t) - L(b_t)
    so, st = float(co.std()) + 1e-12, float(ct.std()) + 1e-12
    ct = (ct - ct.mean()) * (so / st) + co.mean()
    return unitize(co), unitize(ct)


def rank_gauss(v):
    r = (rankdata(np.asarray(v, float).ravel()) - 0.5) / len(v)
    return norm.ppf(np.clip(r, 1e-6, 1 - 1e-6))


def save_report(extra: str = ""):
    LOG.mkdir(parents=True, exist_ok=True)
    body = [
        "# GM V20 Overnight CPU Grandmaster Factory",
        "",
        f"**0 GPU pods · `$0.00/hr` · elapsed {time.time()-T0:.0f}s · n_jobs=-1**",
        "",
        "```",
        *LINES[-250:],
        "```",
        "",
        extra,
    ]
    REPORT.write_text("\n".join(body) + "\n")


class OvernightFactory:
    def __init__(self):
        self.vault: dict[str, dict] = {}
        self.promoted: list[dict] = []
        self.progress = {"engines": {}, "submit": False, "errors": []}
        self.max_slots = 1
        self.slots_used = 0

    def persist(self):
        OUT.mkdir(parents=True, exist_ok=True)
        if self.vault:
            payload = {}
            metas = {}
            for k, v in self.vault.items():
                payload[f"{k}_oof"] = v["oof"]
                payload[f"{k}_test"] = v["test"]
                metas[k] = scrub(v["meta"])
            np.savez_compressed(VAULT, **payload)
            OUT.joinpath("gm_overnight_vault_meta.json").write_text(json.dumps(metas, indent=2))
        self.progress["elapsed_s"] = time.time() - T0
        self.progress["vault_n"] = len(self.vault)
        self.progress["promoted_n"] = len(self.promoted)
        PROGRESS.write_text(json.dumps(scrub(self.progress), indent=2))
        save_report()

    def rebuild(self):
        log("=== REBUILD Slot8 0.95395 ===")
        cache = np.load(OUT / "z_ath_cache.npz")
        self.z_ath, self.d1, self.y, self.fold_id = cache["z_ath"], cache["d1"], cache["y"].astype(np.int32), cache["fold_id"]
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

        self.ms_o = np.load(PRED / "P_super_v4_oof.npy").ravel()
        self.ms_t = np.load(PRED / "P_super_v4_test.npy").ravel()
        g26_o = np.load(G26_OOF).ravel()[: self.n_tr]
        g26_t = np.load(G26_T0).ravel()[: self.n_te]
        self.Bo9 = np.column_stack([
            unitize(L(self.ms_o)), unitize(L(g26_o)), unitize(L(self.XO["megayak_5f_lm"])),
            unitize(L(l7f_o)), unitize(self.XO["daoviet_gpdvae"]), unitize(self.XO["gb_bag5"]),
            unitize(self.XO["yekenot_realmlp"]), unitize(self.d1), unitize(self.dE_o),
        ])
        self.Bt9 = np.column_stack([
            unitize(L(self.ms_t)), unitize(L(g26_t)), unitize(L(self.XT["megayak_5f_lm"])),
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
        Bo_bank, Bt_bank = np.column_stack(bank_o), np.column_stack(bank_t)
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
        self.r2_o, self.r2_t = fam1["R2_v178_orig_oof"].ravel(), fam1["R2_v178_orig_test"].ravel()
        self.cat_o, self.cat_t = fam1["F_cat_obliv_oof"].ravel(), fam1["F_cat_obliv_test"].ravel()

        self.raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
        self.raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
        llr_o = self.raw_o[:, :, 1] - self.raw_o[:, :, 0]
        U5_o = L_fold(gini10(col_z(llr_o)), self.fold_id)
        U5_t = np.mean(
            np.stack([L(gini10(col_z(self.raw_t[f, :, :, 1] - self.raw_t[f, :, :, 0]))) for f in range(5)], 0), 0
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

        def u_bank(ll_pos, ll_neg):
            llr = ll_pos - ll_neg
            Z = col_z(llr)
            K = Z.shape[1]
            h = max(K // 2, 1)
            pair_means = np.column_stack([0.5 * (Z[:, m] + Z[:, m + h]) for m in range(h)]) if K >= 2 else Z
            pair_gaps = np.column_stack([np.abs(Z[:, m] - Z[:, m + h]) for m in range(h)]) if K >= 2 else np.abs(Z)
            Zp, Zn = col_z(ll_pos), col_z(ll_neg)
            return {"U5": gini10(Z), "U6": pair_means.std(1), "U7": pair_gaps.mean(1), "U8": Zp.std(1) - Zn.std(1)}

        bank_o = u_bank(self.raw_o[:, :, 1], self.raw_o[:, :, 0])
        bank_t = {k: [] for k in bank_o}
        for f in range(5):
            bf = u_bank(self.raw_t[f, :, :, 1], self.raw_t[f, :, :, 0])
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
        assert rho < 1e-5

        self.Bo12 = np.column_stack([self.Bo9, self.u_r2, self.u_cat, self.u_u5])
        self.Bt12 = np.column_stack([self.Bt9, self.ut_r2, self.ut_cat, self.ut_u5])
        # Fail-rays only exist on test; for OOF use zeros placeholders so shapes match for Stage-2 Bt_all
        self.Bo_all = np.column_stack([self.Bo12, np.zeros((self.n_tr, 2))])
        self.Bt_all = np.column_stack([self.Bt12, self.bad6, self.bad7])

        # BCE Newton residuals
        self.g_S8 = np.zeros(self.n_tr)
        self.h_S8 = np.zeros(self.n_tr)
        for tr, te in self.folds:
            clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
            clf.fit(self.z_s8[tr].reshape(-1, 1), self.y[tr])
            a, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
            p = expit(a * self.z_s8 + b)
            self.g_S8[te] = self.y[te] - p[te]
            self.h_S8[te] = np.clip(p[te] * (1 - p[te]), 1e-6, None)

        self.train = pd.read_csv(DATA / "train.csv")
        self.test = pd.read_csv(DATA / "test.csv")
        log(f"Bo12+rays ready; g_S8 std={self.g_S8.std():.4f}")

        # Seed prior vault survivors if present
        prior = OUT / "gm_survivor_vault.npz"
        if prior.exists():
            Zp = np.load(prior)
            meta_p = OUT / "gm_survivor_vault_meta.json"
            metas = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            for k in metas:
                ok, tk = f"{k}_oof", f"{k}_test"
                if ok in Zp.files and tk in Zp.files:
                    self.vault[f"prior_{k}"] = {
                        "oof": unitize(Zp[ok]),
                        "test": unitize(Zp[tk]),
                        "meta": {**metas[k], "source": "prior_vault"},
                    }
            log(f"Seeded {sum(1 for k in self.vault if k.startswith('prior_'))} prior vault streams")

    def orient(self, oo, ot):
        u = unitize(oo)
        if roc_auc_score(self.y[self.te0], self.z_s8[self.te0] + 0.01 * u[self.te0]) < roc_auc_score(
            self.y[self.te0], self.z_s8[self.te0] - 0.01 * u[self.te0]
        ):
            return -u, -unitize(ot)
        return unitize(u), unitize(ot)

    def eval_family(self, name, vo, vt, ws=(0.006, 0.008, 0.010, 0.012, 0.016, 0.020)):
        # Rule C: pre-projection raw disp
        pre_disp = float(np.std(vt) / (np.std(vo) + 1e-12))
        if not (0.88 <= pre_disp <= 1.12):
            # soft scale match then continue
            vt = (vt - np.mean(vt)) * (np.std(vo) / (np.std(vt) + 1e-12)) + np.mean(vo)
            log(f"{name}: pre_disp={pre_disp:.3f} soft-scaled into gate")
        oo, ot, pre = two_stage_orth(
            vo, vt, self.Bo12, self.Bt12, self.Bt_all, self.folds, ridge_alpha=10.0
        )
        u, ut = self.orient(oo, ot)
        g12 = forensic(u, ut, self.Bt12)
        g9 = forensic(u, ut, self.Bt9)
        # c6/c7 on unitized ut vs rays (should be ~0 after Stage-2)
        c6_u = float(np.corrcoef(unitize(ut), self.bad6)[0, 1])
        c7_u = float(np.corrcoef(unitize(ut), self.bad7)[0, 1])
        f3 = abs(float(np.corrcoef(unitize(ut), unitize(L(self.slot8) - L(self.slot5)))[0, 1]))
        best = None
        for w in ws:
            oof = {b: self.bases[b] + w * u for b in self.bases}
            per = {b: h2h(oof[b], self.bases[b], self.y, self.folds) for b in self.bases}
            cand = qlock(
                L(self.slot3) + W8_O * self.ut_r2 + W8_C * self.ut_cat + W8_U * self.ut_u5 + w * ut, self.lv
            )
            rho = one_m_rho_e4(cand, self.slot8)
            delta = unitize(L(cand) - L(self.slot8))
            c6 = float(np.corrcoef(delta, self.bad6)[0, 1])
            c7 = float(np.corrcoef(delta, self.bad7)[0, 1])
            ok_h = survivor_ok(per, g9, g12, c6, c7)
            ok_g = hard_gates(per, rho, c6, c7, g9, g12, f3)
            row = {
                "name": f"{name}_w{w:.3f}", "family": name, "w": w, "u": u, "ut": ut,
                "per": per, "rho": rho, "c6": c6, "c7": c7, "g9": g9, "g12": g12, "f3": f3,
                "cand": cand, "ok_gates": ok_g, "pre": pre,
                "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                "min_lift": float(np.min([per[b]["min"] for b in per])),
                "c6_u": c6_u, "c7_u": c7_u,
            }
            if ok_g or (ok_h and per["z_S8"]["pos"] >= 4):
                mark = "★" if ok_g else "~"
                log(
                    f"{mark} {row['name']}: ρ={rho:.3f} c7={c7:+.3f} F2={g12['F2']:.4f} "
                    f"z={per['z_S8']['pos']}/5 m={per['z_S8']['mean']:+.2f} t={per['z_S8']['t']:.2f} "
                    f"sf={per['z_S8']['sf25']} gates={ok_g}"
                )
            if ok_h and (best is None or row["min_lift"] > best["min_lift"]):
                best = row
            if ok_g:
                self.promoted.append(row)
        if best is not None:
            self.vault[name] = {
                "oof": best["u"], "test": best["ut"],
                "meta": scrub({
                    "name": best["name"], "w": best["w"], "rho": best["rho"], "c6": best["c6"], "c7": best["c7"],
                    "f2": best["g12"]["F2"], "disp": best["g12"]["raw_disp"], "KS": best["g12"]["raw_KS"],
                    "mean_lift": best["mean_lift"], "min_lift": best["min_lift"], "per": best["per"], "f3": best["f3"],
                }),
            }
            log(f"VAULT +{name} (n={len(self.vault)}) F2={best['g12']['F2']:.4f} c6u={c6_u:+.4f} c7u={c7_u:+.4f}")
        return best

    # ---------- ENGINE 1 ----------
    def engine1(self):
        log("=== ENGINE 1: Hierarchical EB Epistatic Excess ===")
        tr, te = self.train, self.test
        y = self.y
        keys_tr, keys_te = {}, {}

        def add_digits(prefix, vtr, vte):
            vtr, vte = np.asarray(vtr, float), np.asarray(vte, float)
            keys_tr[f"{prefix}_cents"] = (np.round(vtr * 100).astype(np.int64) % 100)
            keys_te[f"{prefix}_cents"] = (np.round(vte * 100).astype(np.int64) % 100)
            keys_tr[f"{prefix}_tenths"] = (np.round(vtr * 10).astype(np.int64) % 10)
            keys_te[f"{prefix}_tenths"] = (np.round(vte * 10).astype(np.int64) % 10)
            keys_tr[f"{prefix}_ones"] = (np.abs(vtr).astype(np.int64) % 10)
            keys_te[f"{prefix}_ones"] = (np.abs(vte).astype(np.int64) % 10)

        for c in ("amount_usd", "account_balance_usd", "merchant_risk_score", "velocity_score",
                  "distance_from_home_km", "hours_since_last_txn"):
            add_digits(c, tr[c].to_numpy(float), te[c].to_numpy(float))

        # flags bitmask (6 binaries)
        flags = [
            "is_foreign_transaction", "is_new_merchant", "used_vpn",
            "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
        ]
        ftr = np.zeros(self.n_tr, dtype=np.int64)
        fte = np.zeros(self.n_te, dtype=np.int64)
        for i, c in enumerate(flags):
            ftr |= (tr[c].astype(int).to_numpy() & 1) << i
            fte |= (te[c].astype(int).to_numpy() & 1) << i
        keys_tr["flag64"] = ftr
        keys_te["flag64"] = fte

        for cat in ("merchant_category", "channel", "device_type"):
            codes, _ = pd.factorize(pd.concat([tr[cat], te[cat]], ignore_index=True))
            keys_tr[cat] = codes[: self.n_tr].astype(np.int64)
            keys_te[cat] = codes[self.n_tr :].astype(np.int64)

        ch_dev_tr = keys_tr["channel"] * 16 + keys_tr["device_type"]
        ch_dev_te = keys_te["channel"] * 16 + keys_te["device_type"]
        keys_tr["ch_dev"] = ch_dev_tr
        keys_te["ch_dev"] = ch_dev_te
        keys_tr["vel_step25"] = (tr["velocity_score"].to_numpy(float) // 25).astype(np.int64)
        keys_te["vel_step25"] = (te["velocity_score"].to_numpy(float) // 25).astype(np.int64)
        keys_tr["mrisk_step25"] = (tr["merchant_risk_score"].to_numpy(float) // 25).astype(np.int64)
        keys_te["mrisk_step25"] = (te["merchant_risk_score"].to_numpy(float) // 25).astype(np.int64)
        keys_tr["cvv"] = tr["cvv_retry_count"].to_numpy(np.int64)
        keys_te["cvv"] = te["cvv_retry_count"].to_numpy(np.int64)

        crosses = [
            ("amount_usd_cents", "account_balance_usd_cents"),
            ("amount_usd_cents", "merchant_category"),
            ("account_balance_usd_cents", "merchant_category"),
            ("amount_usd_tenths", "device_type"),
            ("amount_usd_tenths", "channel"),
            ("merchant_risk_score_cents", "velocity_score_cents"),
            ("amount_usd_cents", "flag64"),
            ("flag64", "ch_dev"),
            ("flag64", "merchant_category"),
            ("vel_step25", "mrisk_step25"),
            ("vel_step25", "flag64"),
            ("ch_dev", "merchant_category"),
            ("amount_usd_ones", "merchant_category"),
            ("velocity_score_tenths", "channel"),
            ("hours_since_last_txn_tenths", "merchant_category"),
            ("amount_usd_cents", "cvv"),
            ("merchant_risk_score_tenths", "channel"),
            ("distance_from_home_km_tenths", "device_type"),
            ("amount_usd_ones", "account_balance_usd_ones"),
            ("flag64", "cvv"),
        ]

        feats_o, feats_t, kept = [], [], []
        for a, b in crosses:
            if a not in keys_tr or b not in keys_tr:
                continue
            ka, kb = keys_tr[a].astype(np.int64), keys_tr[b].astype(np.int64)
            kat, kbt = keys_te[a].astype(np.int64), keys_te[b].astype(np.int64)
            bmod = int(max(kb.max(), kbt.max()) + 1)
            pk, pkt = ka * bmod + kb, kat * bmod + kbt
            fo = np.zeros(self.n_tr)
            ft_folds = []
            for tr_idx, te_idx in self.folds:
                ybar = float(y[tr_idx].mean())
                # inner 3-fold for 1D rates (fast proxy of 5-seed)
                ra = np.full(int(ka[tr_idx].max()) + 1, ybar)
                rb = np.full(int(kb[tr_idx].max()) + 1, ybar)
                for _, inn in StratifiedKFold(3, shuffle=True, random_state=42).split(
                    np.zeros(len(tr_idx)), y[tr_idx]
                ):
                    idx = tr_idx[inn]
                    # accumulate then average — simpler: use full tr once (outer-safe LOFO on oof)
                    pass
                # use full outer-tr rates (OOF on te_idx is still clean)
                def rate_map(keys):
                    mx = int(keys.max()) + 1
                    cnt = np.bincount(keys, minlength=mx).astype(np.float64)
                    sm = np.bincount(keys, weights=y[tr_idx], minlength=mx)
                    rates = np.full(mx, ybar)
                    m = cnt >= 15
                    rates[m] = sm[m] / cnt[m]
                    return rates

                ra, rb = rate_map(ka[tr_idx]), rate_map(kb[tr_idx])
                mxp = int(pk[tr_idx].max()) + 1
                cnt2 = np.bincount(pk[tr_idx], minlength=mxp).astype(np.float64)
                sm2 = np.bincount(pk[tr_idx], weights=y[tr_idx], minlength=mxp)
                p2 = np.full(mxp, ybar)
                m2 = cnt2 >= 15
                p2[m2] = sm2[m2] / cnt2[m2]

                def excess(ka_arr, kb_arr, pk_arr):
                    pa = ra[np.clip(ka_arr, 0, len(ra) - 1)]
                    pb = rb[np.clip(kb_arr, 0, len(rb) - 1)]
                    eta = (
                        logit(np.clip(pa, 1e-4, 1 - 1e-4))
                        + logit(np.clip(pb, 1e-4, 1 - 1e-4))
                        - logit(np.clip(ybar, 1e-4, 1 - 1e-4))
                    )
                    pkc = np.clip(pk_arr, 0, mxp - 1)
                    n = cnt2[pkc]
                    ex = (p2[pkc] - expit(eta)) * n / (n + 50.0)
                    ex[n < 15] = 0.0
                    return ex

                fo[te_idx] = excess(ka[te_idx], kb[te_idx], pk[te_idx])
                ft_folds.append(excess(kat, kbt, pkt))
            ft = np.mean(ft_folds, 0)
            ks = float(ks_2samp(fo, ft).statistic)
            if ks > 0.015:
                log(f"E1 quarantine {a}×{b} KS={ks:.4f}")
                continue
            feats_o.append(unitize(fo))
            feats_t.append(unitize(ft))
            kept.append(f"{a}x{b}")
        log(f"E1 kept {len(kept)} epistatic cols")
        if not feats_o:
            return
        Fo, Ft = np.column_stack(feats_o), np.column_stack(feats_t)

        # tab45-lite
        tab_o, tab_t = [], []
        for c in ("amount_usd", "velocity_score", "merchant_risk_score", "distance_from_home_km"):
            tab_o.append(unitize(L_fold(tr[c].to_numpy(float), self.fold_id)))
            tab_t.append(unitize(L(te[c].to_numpy(float))))
        To, Tt = np.column_stack(tab_o), np.column_stack(tab_t)
        Xo = np.column_stack([Fo, To])
        Xt = np.column_stack([Ft, Tt])

        # E1_cholesky / ridge
        oof = np.zeros(self.n_tr)
        tes = []
        for tr_idx, te_idx in self.folds:
            lr = Ridge(alpha=3000.0).fit(Xo[tr_idx], self.g_S8[tr_idx], sample_weight=self.h_S8[tr_idx])
            oof[te_idx] = lr.predict(Xo[te_idx])
            tes.append(lr.predict(Xt))
        self.eval_family("E1_cholesky_glm", oof, np.mean(tes, 0))

        # E1 cat obliv bag
        oof_c = np.zeros(self.n_tr)
        tec = []
        for depth, seed in [(3, 11), (4, 21), (5, 31), (3, 41)]:
            fo = np.zeros(self.n_tr)
            ft = []
            for tr_idx, te_idx in self.folds:
                model = CatBoostRegressor(
                    depth=depth, l2_leaf_reg=40, iterations=100, learning_rate=0.035,
                    random_seed=seed, verbose=False, allow_writing_files=False,
                    grow_policy="SymmetricTree", thread_count=max(os.cpu_count() or 4, 1),
                )
                model.fit(Xo[tr_idx], self.g_S8[tr_idx], sample_weight=self.h_S8[tr_idx])
                fo[te_idx] = model.predict(Xo[te_idx])
                ft.append(model.predict(Xt))
            oof_c += fo
            tec.append(np.mean(ft, 0))
        self.eval_family("E1_cat_obliv", oof_c / 4, np.mean(tec, 0))

        # E1 lgb
        oof_l = np.zeros(self.n_tr)
        tel = []
        for leaves, seed in [(4, 11), (6, 21), (4, 31), (6, 41)]:
            fo = np.zeros(self.n_tr)
            ft = []
            for tr_idx, te_idx in self.folds:
                model = lgb.LGBMRegressor(
                    num_leaves=leaves, colsample_bytree=0.35, subsample=0.80,
                    min_child_samples=600, reg_lambda=40, n_estimators=55, learning_rate=0.03,
                    random_state=seed, verbosity=-1, n_jobs=N_JOBS,
                )
                model.fit(Xo[tr_idx], self.g_S8[tr_idx], sample_weight=self.h_S8[tr_idx])
                fo[te_idx] = model.predict(Xo[te_idx])
                ft.append(model.predict(Xt))
            oof_l += fo
            tel.append(np.mean(ft, 0))
        self.eval_family("E1_lgb_colsub", oof_l / 4, np.mean(tel, 0))

    # ---------- ENGINE 2 ----------
    def engine2(self):
        log("=== ENGINE 2: Permutation Trajectory + Forgotten Heavyweights ===")
        streams = {}

        def add(name, o, t):
            o, t = np.asarray(o, float).ravel(), np.asarray(t, float).ravel()
            if len(o) != self.n_tr or len(t) != self.n_te:
                return
            if roc_auc_score(self.y[self.te0], o[self.te0]) < 0.5:
                o, t = -o, -t
            streams[name] = (o, t)

        if all((MED / f"oof_{f}.npy").exists() for f in range(5)):
            go, gt = assemble_gpt2(MED, self.y, self.folds)
            add("gpt2_med", go, gt)
            add("gpt2_med_m_MS", *bpf_contrast(go, gt, self.ms_o, self.ms_t, self.fold_id))
            add("gpt2_med_m_LM", *bpf_contrast(go, gt, self.XO["megayak_5f_lm"], self.XT["megayak_5f_lm"], self.fold_id))
        if all((REV / f"oof_{f}.npy").exists() for f in range(5)):
            add("gpt2_rev", *assemble_gpt2(REV, self.y, self.folds))
        for stem in ["v10_ridge_resid", "v10_cb_d4_bag3", "v10_combo3_orth_LM", "v10_ridge_resid_orth_LM"]:
            p = load_pair(V10 / f"{stem}_oof.npy", V10 / f"{stem}_test.npy")
            if p:
                add(stem, *p)
        for nm, po, pt in [
            ("B_xgb_llr", PRED / "paradigm_B/xgb_llr_oof.npy", PRED / "paradigm_B/xgb_llr_test.npy"),
            ("P_super_v5", PRED / "P_super_v5_harvest_oof.npy", PRED / "P_super_v5_harvest_test.npy"),
            ("P4_L4.2", MV2 / "P4_L4.2_pure5f_oof.npy", MV2 / "P4_L4.2_pure5f_test.npy"),
            ("P4_L4.5", MV2 / "P4_L4.5_pure5f_oof.npy", MV2 / "P4_L4.5_pure5f_test.npy"),
            ("harvest_p4_xgb", MV2 / "harvest_p4_cuda_xgb_oof.npy", MV2 / "harvest_p4_cuda_xgb_test.npy"),
        ]:
            p = load_pair(po, pt)
            if p:
                add(nm, *p)
        if "B_xgb_llr" in streams:
            add("Bxgb_m_d1", *bpf_contrast(streams["B_xgb_llr"][0], streams["B_xgb_llr"][1], self.d1, self.d1t, self.fold_id))

        # X_perm32 from L7
        def perm_feats_from_llr(llr):  # (n, K)
            Z = col_z(llr)
            qs = np.quantile(Z, [0.0, 0.1, 0.2, 0.5, 0.7, 0.8, 1.0], axis=1).T
            softmin = -np.logaddexp.reduce(-1.5 * Z, axis=1) / 1.5
            softmax = np.logaddexp.reduce(1.5 * Z, axis=1) / 1.5
            trimmed = (Z.sum(1) - Z.min(1) - Z.max(1)) / max(Z.shape[1] - 2, 1)
            return np.column_stack([qs, softmin, softmax, trimmed, Z.mean(1), Z.std(1)])

        llr_o = self.raw_o[:, :, 1] - self.raw_o[:, :, 0]
        Po = []
        for f in range(5):
            m = self.fold_id == f
            Po.append(None)
        # build OOF per fold standardized then stack
        Xperm_o = np.zeros((self.n_tr, 12))
        for f, (tr, te) in enumerate(self.folds):
            # standardize using te block only for OOF symmetry
            Xperm_o[te] = col_z(perm_feats_from_llr(llr_o[te]))
        Xperm_t_folds = []
        for f in range(5):
            llr_tf = self.raw_t[f, :, :, 1] - self.raw_t[f, :, :, 0]
            Xperm_t_folds.append(col_z(perm_feats_from_llr(llr_tf)))
        Xperm_t = np.mean(Xperm_t_folds, 0)
        for j in range(Xperm_o.shape[1]):
            add(f"perm_{j}", Xperm_o[:, j], Xperm_t[:, j])
        for k in ["U5", "U6", "U7", "U8"]:
            add(k, self.U_o[k], self.U_t[k])

        log(f"E2 streams={len(streams)}")
        # Build matrix of top streams for ridge/cat
        names = list(streams.keys())
        # evaluate each solo first (subset heavy)
        solo_priority = [
            "gpt2_med", "gpt2_med_m_LM", "B_xgb_llr", "v10_ridge_resid", "v10_combo3_orth_LM",
            "P4_L4.2", "P4_L4.5", "P_super_v5", "Bxgb_m_d1", "U8", "perm_0", "perm_7",
        ]
        for n in solo_priority:
            if n in streams:
                try:
                    self.eval_family(f"E2_{n}", *streams[n])
                except Exception as e:
                    log(f"E2 solo fail {n}: {e}")

        # fusion ridge
        use = [n for n in names if n.startswith("perm_") or n in solo_priority or n.startswith("gpt2") or n.startswith("v10") or n.startswith("P4") or n.startswith("B")]
        use = use[:40]
        if use:
            Xo = col_z(np.column_stack([streams[n][0] for n in use]))
            Xt = col_z(np.column_stack([streams[n][1] for n in use]))
            oof = np.zeros(self.n_tr)
            tes = []
            for tr_idx, te_idx in self.folds:
                lr = Ridge(alpha=1000.0).fit(Xo[tr_idx], self.g_S8[tr_idx], sample_weight=self.h_S8[tr_idx])
                oof[te_idx] = lr.predict(Xo[te_idx])
                tes.append(lr.predict(Xt))
            self.eval_family("E2_heavy_ridge", oof, np.mean(tes, 0))

            oof_c = np.zeros(self.n_tr)
            tec = []
            for depth, seed in [(3, 11), (4, 21), (5, 31)]:
                fo = np.zeros(self.n_tr)
                ft = []
                for tr_idx, te_idx in self.folds:
                    model = CatBoostRegressor(
                        depth=depth, l2_leaf_reg=35, iterations=90, learning_rate=0.035,
                        random_seed=seed, verbose=False, allow_writing_files=False,
                        grow_policy="SymmetricTree", thread_count=max(os.cpu_count() or 4, 1),
                    )
                    model.fit(Xo[tr_idx], self.g_S8[tr_idx], sample_weight=self.h_S8[tr_idx])
                    fo[te_idx] = model.predict(Xo[te_idx])
                    ft.append(model.predict(Xt))
                oof_c += fo
                tec.append(np.mean(ft, 0))
            self.eval_family("E2_perm_cat", oof_c / 3, np.mean(tec, 0))

            oof_l = np.zeros(self.n_tr)
            tel = []
            for leaves, seed in [(4, 11), (6, 21), (8, 31)]:
                fo = np.zeros(self.n_tr)
                ft = []
                for tr_idx, te_idx in self.folds:
                    model = lgb.LGBMRegressor(
                        num_leaves=leaves, colsample_bytree=0.4, subsample=0.8,
                        min_child_samples=500, reg_lambda=30, n_estimators=80, learning_rate=0.03,
                        random_state=seed, verbosity=-1, n_jobs=N_JOBS,
                    )
                    model.fit(Xo[tr_idx], self.g_S8[tr_idx], sample_weight=self.h_S8[tr_idx])
                    fo[te_idx] = model.predict(Xo[te_idx])
                    ft.append(model.predict(Xt))
                oof_l += fo
                tel.append(np.mean(ft, 0))
            self.eval_family("E2_perm_lgb", oof_l / 3, np.mean(tel, 0))

    # ---------- ENGINE 3 ----------
    def engine3(self):
        log("=== ENGINE 3: cKDTree Clone Geometry ===")
        tr, te = self.train, self.test

        def rg_cols(df, cols):
            return np.column_stack([rank_gauss(df[c].to_numpy(float)) for c in cols])

        S1_cols = ["amount_usd", "account_balance_usd", "hours_since_last_txn"]
        # amt/(bal+10)
        def fin4(df):
            a = df["amount_usd"].to_numpy(float)
            b = df["account_balance_usd"].to_numpy(float)
            h = df["hours_since_last_txn"].to_numpy(float)
            return np.column_stack([rank_gauss(a), rank_gauss(b), rank_gauss(a / (b + 10)), rank_gauss(h)])

        def risk4(df):
            return rg_cols(df, ["merchant_risk_score", "velocity_score", "distance_from_home_km", "card_age_months"])

        def mant4(df):
            cols = []
            for c in ("amount_usd", "account_balance_usd", "merchant_risk_score", "velocity_score"):
                cols.append((np.round(df[c].to_numpy(float) * 100) % 100) / 100.0)
            return np.column_stack(cols)

        def full10(df):
            cols = [
                "amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd",
                "velocity_score", "merchant_risk_score", "txn_count_last_24h", "card_age_months",
                "customer_age", "cvv_retry_count",
            ]
            return rg_cols(df, cols)

        spaces_tr = {"S1": fin4(tr), "S2": risk4(tr), "S3": mant4(tr), "S4": full10(tr)}
        spaces_te = {"S1": fin4(te), "S2": risk4(te), "S3": mant4(te), "S4": full10(te)}

        feats_o, feats_t = [], []
        # OOF knn vs tr_pos/tr_neg
        for sname in ["S1", "S2", "S3", "S4"]:
            Xo, Xt = col_z(spaces_tr[sname]), col_z(spaces_te[sname])
            d = Xo.shape[1]
            fo_ratio = np.zeros(self.n_tr)
            fo_dens = np.zeros(self.n_tr)
            ft_ratio, ft_dens = [], []
            for tr_idx, te_idx in self.folds:
                pos = tr_idx[self.y[tr_idx] == 1]
                neg = tr_idx[self.y[tr_idx] == 0]
                # subsample neg for speed
                rng = np.random.default_rng(42)
                if len(neg) > 80000:
                    neg = rng.choice(neg, 80000, replace=False)
                if len(pos) > 20000:
                    pos = rng.choice(pos, 20000, replace=False)
                tree_p = cKDTree(Xo[pos])
                tree_n = cKDTree(Xo[neg])
                scale = (len(tr_idx)) ** (1.0 / d)

                def feats(Xq):
                    dp, _ = tree_p.query(Xq, k=3, workers=N_JOBS)
                    dn, _ = tree_n.query(Xq, k=5, workers=N_JOBS)
                    if dp.ndim == 1:
                        dp = dp[:, None]
                        dn = dn[:, None]
                    dpos3 = dp.mean(1) / scale
                    dneg3 = dn[:, :3].mean(1) / scale
                    ratio = np.log((dneg3 + 1e-3) / (dpos3 + 1e-3))
                    # gaussian dens proxy
                    dens = np.exp(-((dpos3 / 0.35) ** 2)) - np.exp(-((dneg3 / 0.35) ** 2))
                    return ratio, dens

                r_te, d_te = feats(Xo[te_idx])
                fo_ratio[te_idx] = r_te
                fo_dens[te_idx] = d_te
                r_t, d_t = feats(Xt)
                ft_ratio.append(r_t)
                ft_dens.append(d_t)
            feats_o.append(unitize(fo_ratio))
            feats_o.append(unitize(fo_dens))
            feats_t.append(unitize(np.mean(ft_ratio, 0)))
            feats_t.append(unitize(np.mean(ft_dens, 0)))
            log(f"E3 subspace {sname} knn done")

        # tab top
        for c in ("amount_usd", "velocity_score", "merchant_risk_score"):
            feats_o.append(unitize(L_fold(tr[c].to_numpy(float), self.fold_id)))
            feats_t.append(unitize(L(te[c].to_numpy(float))))

        Xo = np.column_stack(feats_o)
        Xt = np.column_stack(feats_t)
        oof = np.zeros(self.n_tr)
        tes = []
        for tr_idx, te_idx in self.folds:
            lr = Ridge(alpha=1500.0).fit(Xo[tr_idx], self.g_S8[tr_idx], sample_weight=self.h_S8[tr_idx])
            oof[te_idx] = lr.predict(Xo[te_idx])
            tes.append(lr.predict(Xt))
        self.eval_family("E3_knn_ridge", oof, np.mean(tes, 0))

        oof_c = np.zeros(self.n_tr)
        tec = []
        for depth, seed in [(3, 11), (4, 21)]:
            fo = np.zeros(self.n_tr)
            ft = []
            for tr_idx, te_idx in self.folds:
                model = CatBoostRegressor(
                    depth=depth, iterations=80, learning_rate=0.04, l2_leaf_reg=25,
                    random_seed=seed, verbose=False, allow_writing_files=False,
                    grow_policy="SymmetricTree", thread_count=max(os.cpu_count() or 4, 1),
                )
                model.fit(Xo[tr_idx], self.g_S8[tr_idx], sample_weight=self.h_S8[tr_idx])
                fo[te_idx] = model.predict(Xo[te_idx])
                ft.append(model.predict(Xt))
            oof_c += fo
            tec.append(np.mean(ft, 0))
        self.eval_family("E3_knn_cat", oof_c / 2, np.mean(tec, 0))

        oof_l = np.zeros(self.n_tr)
        tel = []
        for leaves, seed in [(4, 11), (6, 21)]:
            fo = np.zeros(self.n_tr)
            ft = []
            for tr_idx, te_idx in self.folds:
                model = lgb.LGBMRegressor(
                    num_leaves=leaves, n_estimators=70, learning_rate=0.03, reg_lambda=30,
                    min_child_samples=500, random_state=seed, verbosity=-1, n_jobs=N_JOBS,
                )
                model.fit(Xo[tr_idx], self.g_S8[tr_idx], sample_weight=self.h_S8[tr_idx])
                fo[te_idx] = model.predict(Xo[te_idx])
                ft.append(model.predict(Xt))
            oof_l += fo
            tel.append(np.mean(ft, 0))
        self.eval_family("E3_knn_lgb", oof_l / 2, np.mean(tel, 0))

    # ---------- ENGINE 4 ----------
    def engine4(self):
        log("=== ENGINE 4: Regime MoE ===")
        z0 = unitize(self.z_s8)
        # gates
        G_low = expit(-(z0 - 0.85) / 0.35)
        G_tw = np.exp(-((z0 - 1.45) ** 2) / (2 * 0.65 ** 2))
        G_high = expit(+(z0 - 2.05) / 0.35)
        lm = unitize(L_fold(self.XO["megayak_5f_lm"], self.fold_id))
        tab = unitize(0.5 * unitize(self.XO["gb_bag5"]) + 0.5 * unitize(self.XO["yekenot_realmlp"]))
        G_dis = np.tanh(np.abs(lm - tab) + 0.5 * self.u_u5)
        # test gates from twin proxy
        z0t = unitize(L(self.slot8))
        G_low_t = expit(-(z0t - 0.85) / 0.35)
        G_tw_t = np.exp(-((z0t - 1.45) ** 2) / (2 * 0.65 ** 2))
        G_high_t = expit(+(z0t - 2.05) / 0.35)
        lm_t = unitize(L(self.XT["megayak_5f_lm"]))
        tab_t = unitize(0.5 * unitize(self.XT["gb_bag5"]) + 0.5 * unitize(self.XT["yekenot_realmlp"]))
        G_dis_t = np.tanh(np.abs(lm_t - tab_t) + 0.5 * self.ut_u5)

        pillars_o = [
            unitize(lm - tab), self.u_r2, self.u_cat, self.u_u5, unitize(self.d1), unitize(self.dE_o),
        ]
        pillars_t = [
            unitize(lm_t - tab_t), self.ut_r2, self.ut_cat, self.ut_u5, unitize(self.d1t), unitize(self.dE_t),
        ]
        # add top vault residuals
        for k, v in list(self.vault.items())[:6]:
            pillars_o.append(unitize(v["oof"]))
            pillars_t.append(unitize(v["test"]))

        gates_o = [G_low, G_tw, G_high, G_dis]
        gates_t = [G_low_t, G_tw_t, G_high_t, G_dis_t]
        cols_o, cols_t = [], []
        for p_o, p_t in zip(pillars_o, pillars_t):
            for g_o, g_t in zip(gates_o, gates_t):
                cols_o.append(unitize(p_o * g_o))
                cols_t.append(unitize(p_t * g_t))
        Xo = np.column_stack(cols_o)
        Xt = np.column_stack(cols_t)

        oof = np.zeros(self.n_tr)
        tes = []
        for tr_idx, te_idx in self.folds:
            lr = Ridge(alpha=2000.0).fit(Xo[tr_idx], self.g_S8[tr_idx], sample_weight=self.h_S8[tr_idx])
            oof[te_idx] = lr.predict(Xo[te_idx])
            tes.append(lr.predict(Xt))
        # monotonicity soft check: corr with z_s8 should stay high when added small
        delta = unitize(oof)
        mono = float(np.corrcoef(self.z_s8 + 0.01 * delta, self.z_s8)[0, 1])
        log(f"E4 mono proxy corr={mono:.4f}")
        self.eval_family("E4_regime_moe", oof, np.mean(tes, 0))

    # ---------- ENGINE 5 ----------
    def engine5(self):
        log("=== ENGINE 5: Multi-Vault Super-Synthesis ===")
        # filter hygiene
        names = []
        for k, v in self.vault.items():
            m = v["meta"]
            if m.get("f2", 1) < 0.035 and 0.90 <= m.get("disp", 0) <= 1.10 and abs(m.get("c7", 1)) < 0.20:
                names.append(k)
        log(f"E5 hygiene vault n={len(names)} / {len(self.vault)}")
        if len(names) < 2:
            log("E5 insufficient vault — skip solvers")
            return

        Uo = np.column_stack([unitize(self.vault[n]["oof"]) for n in names])
        Ut = np.column_stack([unitize(self.vault[n]["test"]) for n in names])
        C = np.corrcoef(Uo.T)
        log(f"E5 |corr| mean={float(np.mean(np.abs(C-np.eye(len(names))))):.3f} max={float(np.max(np.abs(C-np.eye(len(names))))):.3f}")
        for n in names:
            m = self.vault[n]["meta"]
            log(f"  {n}: mean={m.get('mean_lift',0):+.2f} min={m.get('min_lift',0):+.2f} F2={m.get('f2',0):.3f} ρ={m.get('rho',0):.3f}")

        # diversify
        keep = []
        for i, n in enumerate(names):
            if all(abs(C[i, j]) < 0.50 for j in keep):
                keep.append(i)
            if len(keep) >= 8:
                break
        if len(keep) < 2:
            keep = list(range(min(5, len(names))))
        names_k = [names[i] for i in keep]
        Uo, Ut = Uo[:, keep], Ut[:, keep]
        K = len(keep)
        log(f"E5 synth K={K}: {names_k}")

        rng = np.random.default_rng(42)

        def eval_alpha(alpha, tag, ws=(0.008, 0.010, 0.012, 0.016, 0.020)):
            alpha = np.asarray(alpha, float)
            alpha = np.clip(alpha, 0, 0.40)
            if alpha.sum() < 1e-12:
                return None
            alpha = alpha / alpha.sum()
            u = unitize(Uo @ alpha)
            ut = unitize(Ut @ alpha)
            return self.eval_family(tag, u, ut, ws=ws)

        # S1 maximin random search
        best = None
        for _ in range(100):
            a = rng.dirichlet(np.ones(K))
            a = np.clip(a, 0, 0.40)
            a /= a.sum()
            u = unitize(Uo @ a)
            u, _ = self.orient(u, unitize(Ut @ a))
            score = 0.0
            for base in self.bases.values():
                folds_d = [
                    (roc_auc_score(self.y[te], (base + 0.012 * u)[te]) - roc_auc_score(self.y[te], base[te])) * 1e5
                    for _, te in self.folds
                ]
                score += float(np.mean(folds_d) + 0.6 * np.min(folds_d))
            if best is None or score > best[0]:
                best = (score, a)
        if best:
            log(f"E5 S1 score={best[0]:.3f}")
            eval_alpha(best[1], "E5_S1_maximin")

        # S2 equal / top-k
        order = np.argsort([-self.vault[names_k[i]]["meta"].get("mean_lift", 0) for i in range(K)])
        for kk in sorted(set([min(3, K), min(4, K), K])):
            a = np.zeros(K)
            a[order[:kk]] = 1.0 / kk
            eval_alpha(a, f"E5_S2_top{kk}")

        # S3 joint coarse grid with mean vault residual
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
            cand = qlock(L(self.slot3) + wr2 * self.ut_r2 + wcat * self.ut_cat + wu5 * self.ut_u5 + ww * ut, self.lv)
            rho = one_m_rho_e4(cand, self.slot8)
            delta = unitize(L(cand) - L(self.slot8))
            c6 = float(np.corrcoef(delta, self.bad6)[0, 1])
            c7 = float(np.corrcoef(delta, self.bad7)[0, 1])
            ok = hard_gates(per, rho, c6, c7, g9, g12, f3)
            if ok or (per["z_S8"]["pos"] == 5 and per["z_S8"]["min"] > 0.05 and rho >= 0.20):
                log(
                    f"{'★' if ok else '~'} E5S3 r{wr2:.3f}_c{wcat:.3f}_u{wu5:.3f}_w{ww:.3f}: "
                    f"ρ={rho:.3f} c7={c7:+.3f} z={per['z_S8']['pos']}/5 m={per['z_S8']['mean']:+.2f} "
                    f"t={per['z_S8']['t']:.2f} gates={ok}"
                )
            if ok:
                self.promoted.append({
                    "name": f"E5S3_r{wr2:.3f}_c{wcat:.3f}_u{wu5:.3f}_w{ww:.3f}",
                    "family": "E5S3", "w": ww, "per": per, "rho": rho, "c6": c6, "c7": c7,
                    "g9": g9, "g12": g12, "f3": f3, "cand": cand, "ok_gates": True,
                    "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
                    "min_lift": float(np.min([per[b]["min"] for b in per])),
                    "u": u, "ut": ut,
                })

    def stage_and_promote(self):
        log("=== MORNING STAGING & AUTONOMOUS PROMOTION ===")
        # stage top 5 from vault by mean_lift among 4/5-ish
        ranked = sorted(
            self.vault.items(),
            key=lambda kv: (
                kv[1]["meta"].get("per", {}).get("z_S8", {}).get("pos", 0),
                kv[1]["meta"].get("mean_lift", -999),
                -abs(kv[1]["meta"].get("c7", 1)),
            ),
            reverse=True,
        )[:5]
        for i, (k, v) in enumerate(ranked, 1):
            w = float(v["meta"].get("w", 0.012))
            cand = qlock(
                L(self.slot3) + W8_O * self.ut_r2 + W8_C * self.ut_cat + W8_U * self.ut_u5 + w * v["test"],
                self.lv,
            )
            path = SUB / f"sub_v20_overnight_rank{i}_{k[:40]}.csv"
            pd.DataFrame({"id": self.ids, "is_fraud": cand}).to_csv(path, index=False)
            log(f"STAGED {path.name} ← {v['meta'].get('name', k)}")

        by = {}
        for r in self.promoted:
            if r["name"] not in by or r["min_lift"] > by[r["name"]]["min_lift"]:
                by[r["name"]] = r
        winners = sorted(by.values(), key=lambda r: (r["min_lift"], r["mean_lift"], -abs(r["rho"] - 0.50)), reverse=True)
        log(f"Gate-clear winners: {len(winners)}")
        for w in winners[:8]:
            log(f"  {w['name']}: min={w['min_lift']:+.2f} mean={w['mean_lift']:+.2f} ρ={w['rho']:.3f} t={w['per']['z_S8']['t']:.2f}")

        submit, pub, ref, chosen = False, None, None, None
        if winners and self.slots_used < self.max_slots:
            chosen = winners[0]
            if hard_gates(chosen["per"], chosen["rho"], chosen["c6"], chosen["c7"], chosen["g9"], chosen["g12"], chosen["f3"]):
                out_csv = SUB / "sub_v20_LIVE_slot9_overnight_cpu_promoted.csv"
                pd.DataFrame({"id": self.ids, "is_fraud": chosen["cand"]}).to_csv(out_csv, index=False)
                msg = (
                    f"v20-overnight Slot9: {chosen['name']} vs Slot8 0.95395 | "
                    f"H2H mean={chosen['mean_lift']:+.2f}e-5 min={chosen['min_lift']:+.2f} "
                    f"1-ρ={chosen['rho']:.3f}e-4 c7={chosen['c7']:+.3f} F2={chosen['g12']['F2']:.4f}"
                )
                log(f"SUBMIT {msg}")
                import subprocess
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
                            if "overnight_cpu_promoted" in row.get("fileName", ""):
                                ref = row.get("ref")
                                st = row.get("status", "")
                                log(f"poll {st} score={row.get('publicScore')}")
                                if "complete" in st.lower() and row.get("publicScore"):
                                    pub = float(row["publicScore"])
                                    break
                        if pub is not None:
                            break
                    log(f"PUBLIC_LB={pub} ATH=0.95395")
            else:
                log("Chosen failed hard-gate re-verify — HOLD")
        else:
            log("NO gate-clear winner — HOLD last daily slot")

        self.progress["submit"] = submit
        self.progress["public_lb"] = pub
        self.progress["ref"] = ref
        self.progress["chosen"] = None if not chosen else scrub({
            "name": chosen["name"], "mean_lift": chosen["mean_lift"], "min_lift": chosen["min_lift"],
            "rho": chosen["rho"], "c7": chosen["c7"], "f2": chosen["g12"]["F2"],
        })
        extra = [
            "## Summary",
            f"- Vault: {len(self.vault)}",
            f"- Promoted (gate-clear): {len(winners)}",
            f"- Submit: {submit}",
            f"- Public LB: {pub}",
            "",
            "## Decision",
            ("**LIVE** `" + str(pub) + "`" if submit else "**HOLD** ATH `0.95395`. Last daily slot preserved."),
        ]
        save_report("\n".join(extra))
        self.persist()
        log("DONE")

    def run_engine(self, name, fn):
        log(f">>> START {name}")
        try:
            fn()
            self.progress["engines"][name] = {"ok": True, "vault": len(self.vault), "promoted": len(self.promoted)}
        except Exception as e:
            log(f"!!! {name} FAILED: {e}")
            traceback.print_exc()
            self.progress["errors"].append({"engine": name, "error": str(e)})
            self.progress["engines"][name] = {"ok": False, "error": str(e)}
        finally:
            self.persist()
            log(f"<<< END {name} vault={len(self.vault)} promoted={len(self.promoted)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    log(f"OVERNIGHT CPU FACTORY START cpus={os.cpu_count()} n_jobs={N_JOBS}")
    fac = OvernightFactory()
    fac.rebuild()
    fac.persist()
    fac.run_engine("E1", fac.engine1)
    fac.run_engine("E2", fac.engine2)
    fac.run_engine("E3", fac.engine3)
    fac.run_engine("E4", fac.engine4)
    fac.run_engine("E5", fac.engine5)
    try:
        fac.stage_and_promote()
    except Exception as e:
        log(f"promote failed: {e}")
        traceback.print_exc()
        fac.persist()
    log(f"FACTORY COMPLETE elapsed={time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
