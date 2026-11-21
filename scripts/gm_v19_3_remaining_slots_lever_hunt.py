#!/usr/bin/env python3
"""GM V19.3 — Hunt remaining daily slots for a real lift over Slot5 ATH 0.95394.

Levers (research-first, submit one at a time only if gates clear):
  L1: R2 weight bracket on Slot3 (0.022–0.030) vs Slot5@0.020
  L2: Anchored Slot3 + F_cat_ms45 (V19.2 cleanest Cat, F2=0.014)
  L3: Anchored Slot3 + F_cat_obliv (V19.1 strongest twin)
  L4: Additive U5_gini on Slot5 (GPU dispersion, forensic-clean family)
  L5: Slot5 + small orth(F_cat_ms45|Bo10) if ρ(R2) low enough after orth
  L6: Rank-blend Slot5 with Cat-boosted Slot3 candidate (Spearman mid)

0 pods. Submit IFF: 5/5×3 H2H vs Slot5, min>0.05, sf>=18, t>=2.8,
1-ρ(Slot5)∈[0.15,0.55]e-4, |corr| vs Slot6/7 losers <0.15, raw_disp∈[0.90,1.10].
"""
from __future__ import annotations

import json
import subprocess
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import ks_2samp, rankdata, spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
TMP15 = Path("/tmp/s6e11_v15")
TMP16 = Path("/tmp/s6e11_v16")
OUT = TMP16 / "out"
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
SLOT6 = SUB / "sub_v18_1_LIVE_slot6_Slot5_plus_R1_pure_Bo10_w0p022.csv"
SLOT7 = SUB / "sub_v19_0_LIVE_slot7_zero_echo_peer_hybrid.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W_R2 = 0.020


def L(v):
    v = np.asarray(v, float).ravel()
    r = (rankdata(v, method="average") - 0.5) / len(v)
    r = np.clip(r, 1e-12, 1.0 - 1e-12)
    return np.log(r / (1.0 - r))


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
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-12)


def gini10(Z):
    K = Z.shape[-1]
    s = np.zeros(Z.shape[:-1], dtype=np.float64)
    for i, j in combinations(range(K), 2):
        s += np.abs(Z[..., i] - Z[..., j])
    return s / (K * (K - 1) / 2.0)


def frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=10.0):
    vo, vt = unitize(v_oof), unitize(v_test)
    orth_o = np.zeros(len(vo))
    betas = []
    for tr, te in folds:
        lr = Ridge(alpha=ridge_alpha, fit_intercept=True)
        lr.fit(Bo[tr], vo[tr])
        betas.append(lr.coef_.copy())
        orth_o[te] = vo[te] - Bo[te] @ lr.coef_ - lr.intercept_
    beta_bar = np.mean(np.stack(betas, 0), axis=0)
    orth_t = vt - Bt @ beta_bar
    return orth_o - orth_o.mean(), orth_t - orth_t.mean(), beta_bar


def h2h(cand, base, y, folds):
    d, deltas = [], []
    for _, te in folds:
        d.append((roc_auc_score(y[te], cand[te]) - roc_auc_score(y[te], base[te])) * 1e5)
        yf, c, b = y[te], cand[te], base[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(yf)), yf
        ):
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


def passes_h2h(per):
    return all(
        per[b]["pos"] >= 5 and per[b]["min"] > 0.05 and per[b]["sf25"] >= 18 and per[b]["t"] >= 2.8
        for b in per
    )


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
    dE_o = unitize(cache["dE"])
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    te0 = folds[0][1]
    n_tr, n_te = len(y), 299984

    Z = np.load(Q17, allow_pickle=True)
    qnames = [str(s) for s in Z["stream_names"]]
    XO = {n: Z[f"oof_{n}"].ravel() for n in qnames}
    XT = {n: Z[f"test_{n}"].ravel() for n in qnames}

    live = pd.read_csv(LIVE)
    ids = live["id"].to_numpy()
    lv = live["is_fraud"].to_numpy(float)
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot5 = pd.read_csv(SLOT5).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot6 = pd.read_csv(SLOT6).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot7 = pd.read_csv(SLOT7).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    bad6 = unitize(L(slot6) - L(slot5))
    bad7 = unitize(L(slot7) - L(slot5))

    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    l7f_o, l7f_t = XO["lm7_5fold"], XT["lm7_5fold"]
    R3_t_L = np.stack([L(R3_te[f]) for f in range(5)], 0)
    raw_E_o = (L(R3) - L(l7f_o)) * (0.5 + 0.5 * U(z_ath))
    raw_E_t = (L(R3_t_L.mean(0)) - L(l7f_t)) * (0.5 + 0.5 * U(lv))
    Xo_e = np.column_stack([unitize(z_ath), unitize(d1)])
    Xt_e = np.column_stack([unitize(L(lv)), unitize(d1t)])
    dE_o2, dE_t, _ = frozen_lofo_orth(raw_E_o, raw_E_t, Xo_e, Xt_e, folds, ridge_alpha=1.0)
    if np.corrcoef(dE_o, unitize(dE_o2))[0, 1] < 0:
        dE_o = -dE_o
    if roc_auc_score(y[te0], z_ath[te0] + 0.01 * unitize(dE_o2)[te0]) < roc_auc_score(
        y[te0], z_ath[te0] - 0.01 * unitize(dE_o2)[te0]
    ):
        dE_t = -dE_t
    dE_o, dE_t = unitize(dE_o), unitize(dE_t)

    ms_o = np.load(REPO / "preds/P_super_v4_oof.npy").ravel()
    ms_t = np.load(REPO / "preds/P_super_v4_test.npy").ravel()
    g26_o = np.load(G26_OOF).ravel()[:n_tr]
    g26_t = np.load(G26_T0).ravel()[:n_te]
    Bo9 = np.column_stack(
        [
            unitize(L(ms_o)),
            unitize(L(g26_o)),
            unitize(L(XO["megayak_5f_lm"])),
            unitize(L(l7f_o)),
            unitize(XO["daoviet_gpdvae"]),
            unitize(XO["gb_bag5"]),
            unitize(XO["yekenot_realmlp"]),
            unitize(d1),
            unitize(dE_o),
        ]
    )
    Bt9 = np.column_stack(
        [
            unitize(L(ms_t)),
            unitize(L(g26_t)),
            unitize(L(XT["megayak_5f_lm"])),
            unitize(L(l7f_t)),
            unitize(XT["daoviet_gpdvae"]),
            unitize(XT["gb_bag5"]),
            unitize(XT["yekenot_realmlp"]),
            unitize(d1t),
            unitize(dE_t),
        ]
    )
    z_s3 = z_ath + 0.01045 * dE_o + 0.0055 * unitize(d1)
    ridge_s3 = Ridge(alpha=1.0, fit_intercept=True).fit(Bt9, unitize(L(slot3)))
    z_twin = unitize(Bo9 @ ridge_s3.coef_ + ridge_s3.intercept_)
    bank_o, bank_t = [], []
    skip = {
        "megayak_5f_lm",
        "lm7_5fold",
        "gpt2_seed2026",
        "masaya_gpt2",
        "daoviet_gpdvae",
        "gb_bag5",
        "yekenot_realmlp",
    }
    for j, nm in enumerate(BN9):
        if nm not in ("d1_v4", "Family_E"):
            bank_o.append(unitize(Bo9[:, j]))
            bank_t.append(unitize(Bt9[:, j]))
    for n in qnames:
        if n not in skip:
            bank_o.append(unitize(XO[n]))
            bank_t.append(unitize(XT[n]))
    for p in sorted(ALEX.glob("oof_*.npy"))[:8]:
        tp = ALEX / ("test_" + p.name[len("oof_") :])
        if tp.exists():
            bank_o.append(unitize(np.load(p).ravel()[:n_tr]))
            bank_t.append(unitize(np.load(tp).ravel()[:n_te]))
    Bo_bank = np.column_stack(bank_o)
    Bt_bank = np.column_stack(bank_t)
    locked_t = W_D1_LOCK * unitize(d1t) + W_FE_LOCK * unitize(dE_t)
    w_nnls, _ = nnls(Bt_bank, (L(slot3) - locked_t) - (L(slot3) - locked_t).mean())
    z_nnls = W_D1_LOCK * unitize(d1) + W_FE_LOCK * unitize(dE_o) + Bo_bank @ w_nnls

    fam1 = np.load(OUT / "v19_1_families.npz")
    fam2 = np.load(OUT / "v19_2_families.npz")

    def orient_bo9(vo, vt):
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds, ridge_alpha=10.0)
        u = unitize(oo)
        if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s3[te0] - 0.01 * u[te0]
        ):
            u, ot = -u, -ot
        return unitize(u), unitize(ot)

    u_r2, ut_r2 = orient_bo9(fam1["R2_v178_orig_oof"], fam1["R2_v178_orig_test"])
    u_cat, ut_cat = orient_bo9(fam1["F_cat_obliv_oof"], fam1["F_cat_obliv_test"])
    u_lgb, ut_lgb = orient_bo9(fam1["F_lgb_ms39_oof"], fam1["F_lgb_ms39_test"])
    u_ms45, ut_ms45 = orient_bo9(fam2["F_cat_ms45_oof"], fam2["F_cat_ms45_test"])
    u_dir, ut_dir = orient_bo9(fam2["F_cat_direct_S5_oof"], fam2["F_cat_direct_S5_test"])
    # direct was trained on S5 residuals — also available as Bo10-orth from npz C_cat
    u_cbo10, ut_cbo10 = fam2["C_cat_orth_S5_oof"], fam2["C_cat_orth_S5_test"]
    u_cbo10, ut_cbo10 = unitize(u_cbo10), unitize(ut_cbo10)
    if roc_auc_score(y[te0], (z_s3 + W_R2 * u_r2)[te0] + 0.01 * u_cbo10[te0]) < roc_auc_score(
        y[te0], (z_s3 + W_R2 * u_r2)[te0] - 0.01 * u_cbo10[te0]
    ):
        u_cbo10, ut_cbo10 = -u_cbo10, -ut_cbo10

    z_s5 = z_s3 + W_R2 * u_r2
    z_t5 = z_twin + W_R2 * u_r2
    z_n5 = z_nnls + W_R2 * u_r2
    bases_s5 = {"z_S5": z_s5, "twin": z_t5, "nnls": z_n5}
    bases_s3 = {"z_S3": z_s3, "twin": z_twin, "nnls": z_nnls}

    # U5
    raw_o = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    raw_t = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy").astype(np.float64)
    llr_o = raw_o[:, :, 1] - raw_o[:, :, 0]
    llr_t = raw_t[:, :, :, 1] - raw_t[:, :, :, 0]
    U5_o = L_fold(gini10(col_z(llr_o)), fold_id)
    U5_t = np.mean(np.stack([L(gini10(col_z(llr_t[f]))) for f in range(5)], 0), 0)
    Bo10 = np.column_stack([Bo9, u_r2])
    Bt10 = np.column_stack([Bt9, ut_r2])
    oo5, ot5, _ = frozen_lofo_orth(U5_o, U5_t, Bo10, Bt10, folds, ridge_alpha=1.0)
    u_u5 = unitize(oo5)
    if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u_u5[te0]) < roc_auc_score(
        y[te0], z_s5[te0] - 0.01 * u_u5[te0]
    ):
        u_u5, ot5 = -u_u5, -ot5
    ut_u5 = unitize(ot5)

    print("=== LEVER HUNT vs Slot5 ATH ===", f"t={time.time()-t0:.1f}s")
    print(
        f"  corr cat/R2={np.corrcoef(u_cat,u_r2)[0,1]:+.3f} "
        f"ms45/R2={np.corrcoef(u_ms45,u_r2)[0,1]:+.3f} "
        f"u5/R2={np.corrcoef(u_u5,u_r2)[0,1]:+.3f}"
    )

    candidates = []

    def eval_cand(name, cand_oof_by_base, cand_test_logit, meta):
        """cand_oof_by_base: dict base_name -> oof score vector to compare vs Slot5 family."""
        per = {b: h2h(cand_oof_by_base[b], bases_s5[b], y, folds) for b in bases_s5}
        cand = qlock(cand_test_logit, lv)
        rho = one_m_rho_e4(cand, slot5)
        delta = unitize(L(cand) - L(slot5))
        c6 = float(np.corrcoef(delta, bad6)[0, 1])
        c7 = float(np.corrcoef(delta, bad7)[0, 1])
        ok = (
            passes_h2h(per)
            and (0.15 <= rho <= 0.55)
            and abs(c6) < 0.15
            and abs(c7) < 0.15
            and np.isfinite(cand).all()
        )
        row = {
            "name": name,
            "per": per,
            "rho": rho,
            "corr6": c6,
            "corr7": c7,
            "ok": ok,
            "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
            "min_lift": float(np.min([per[b]["min"] for b in per])),
            "meta": meta,
            "cand": cand if ok else None,
        }
        mark = "★" if ok else " "
        print(
            f"{mark} {name}: 1-ρ={rho:.3f}e-4 ok={ok} | "
            + " | ".join(
                f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f} "
                f"sf{per[b]['sf25']} t={per[b]['t']:.2f}"
                for b in bases_s5
            )
        )
        candidates.append({k: v for k, v in row.items() if k != "cand"})
        return row if ok else None

    winners = []

    # L1: R2 weight bracket
    print("-- L1 R2 weight bracket --")
    for w in (0.022, 0.024, 0.025, 0.026, 0.028, 0.030):
        oof = {
            "z_S5": z_s3 + w * u_r2,
            "twin": z_twin + w * u_r2,
            "nnls": z_nnls + w * u_r2,
        }
        wrow = eval_cand(
            f"L1_R2_w{w:.3f}",
            oof,
            L(slot3) + w * ut_r2,
            {"lever": "R2_weight", "w": w},
        )
        if wrow:
            winners.append(wrow)

    # L2: anchored ms45
    print("-- L2 anchored F_cat_ms45 --")
    for wo in (0.016, 0.018, 0.020):
        for wc in (0.006, 0.008, 0.010, 0.012, 0.015, 0.018):
            oof = {
                "z_S5": z_s3 + wo * u_r2 + wc * u_ms45,
                "twin": z_twin + wo * u_r2 + wc * u_ms45,
                "nnls": z_nnls + wo * u_r2 + wc * u_ms45,
            }
            wrow = eval_cand(
                f"L2_ms45_wo{wo:.3f}_wc{wc:.3f}",
                oof,
                L(slot3) + wo * ut_r2 + wc * ut_ms45,
                {"lever": "anchored_ms45", "wo": wo, "wc": wc},
            )
            if wrow:
                winners.append(wrow)

    # L3: anchored cat_obliv (broader than V19.2)
    print("-- L3 anchored F_cat_obliv --")
    for wo in (0.016, 0.018, 0.020):
        for wc in (0.006, 0.008, 0.010, 0.012, 0.015):
            oof = {
                "z_S5": z_s3 + wo * u_r2 + wc * u_cat,
                "twin": z_twin + wo * u_r2 + wc * u_cat,
                "nnls": z_nnls + wo * u_r2 + wc * u_cat,
            }
            wrow = eval_cand(
                f"L3_cat_wo{wo:.3f}_wc{wc:.3f}",
                oof,
                L(slot3) + wo * ut_r2 + wc * ut_cat,
                {"lever": "anchored_cat", "wo": wo, "wc": wc},
            )
            if wrow:
                winners.append(wrow)

    # L3b: R2 + ms45 + small lgb39
    print("-- L3b R2+ms45+lgb39 --")
    for wo, wc, wl in [
        (0.018, 0.008, 0.003),
        (0.018, 0.010, 0.003),
        (0.020, 0.008, 0.003),
        (0.020, 0.010, 0.005),
        (0.016, 0.012, 0.005),
    ]:
        oof = {
            "z_S5": z_s3 + wo * u_r2 + wc * u_ms45 + wl * u_lgb,
            "twin": z_twin + wo * u_r2 + wc * u_ms45 + wl * u_lgb,
            "nnls": z_nnls + wo * u_r2 + wc * u_ms45 + wl * u_lgb,
        }
        wrow = eval_cand(
            f"L3b_wo{wo:.3f}_wc{wc:.3f}_wl{wl:.3f}",
            oof,
            L(slot3) + wo * ut_r2 + wc * ut_ms45 + wl * ut_lgb,
            {"lever": "triple", "wo": wo, "wc": wc, "wl": wl},
        )
        if wrow:
            winners.append(wrow)

    # L4: U5 additive on Slot5
    print("-- L4 U5 additive on Slot5 --")
    for w in (0.006, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025):
        oof = {
            "z_S5": z_s5 + w * u_u5,
            "twin": z_t5 + w * u_u5,
            "nnls": z_n5 + w * u_u5,
        }
        wrow = eval_cand(
            f"L4_U5_w{w:.3f}",
            oof,
            L(slot5) + w * ut_u5,
            {"lever": "U5_add", "w": w},
        )
        if wrow:
            winners.append(wrow)

    # L5: Slot5 + direct_S5 cat (trained on S5 residual — natural additive)
    print("-- L5 Slot5 + F_cat_direct_S5 --")
    # re-orth direct vs Bo10
    oo, ot, _ = frozen_lofo_orth(
        fam2["F_cat_direct_S5_oof"], fam2["F_cat_direct_S5_test"], Bo10, Bt10, folds, 10.0
    )
    u_d, ut_d = unitize(oo), unitize(ot)
    if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u_d[te0]) < roc_auc_score(
        y[te0], z_s5[te0] - 0.01 * u_d[te0]
    ):
        u_d, ut_d = -u_d, -ut_d
    print(f"  corr(direct|Bo10, R2)={np.corrcoef(u_d, u_r2)[0,1]:+.4f}")
    for w in (0.006, 0.008, 0.010, 0.012, 0.015, 0.018, 0.020):
        oof = {
            "z_S5": z_s5 + w * u_d,
            "twin": z_t5 + w * u_d,
            "nnls": z_n5 + w * u_d,
        }
        wrow = eval_cand(
            f"L5_direct_w{w:.3f}",
            oof,
            L(slot5) + w * ut_d,
            {"lever": "direct_S5", "w": w},
        )
        if wrow:
            winners.append(wrow)

    # L6: rank-blend Slot5 with strong Slot3+Cat candidate (even if H2H soft)
    print("-- L6 rank-blend Slot5 with Cat-boosted --")
    # pick a strong OOF cat-boosted even if not full clear
    for wo, wc, alpha in [
        (0.020, 0.012, 0.25),
        (0.020, 0.015, 0.30),
        (0.018, 0.012, 0.35),
        (0.020, 0.010, 0.20),
        (0.022, 0.010, 0.25),
    ]:
        boost_o = {
            "z_S5": z_s3 + wo * u_r2 + wc * u_ms45,
            "twin": z_twin + wo * u_r2 + wc * u_ms45,
            "nnls": z_nnls + wo * u_r2 + wc * u_ms45,
        }
        # rank blend: (1-a)*Slot5_family + a*boost
        oof = {
            b: unitize((1 - alpha) * unitize(bases_s5[b]) + alpha * unitize(boost_o[b]))
            for b in bases_s5
        }
        # for AUC we need actual scores not unitize - use L-rank blend
        oof = {
            b: (1 - alpha) * L(bases_s5[b]) + alpha * L(boost_o[b]) for b in bases_s5
        }
        te_logit = (1 - alpha) * L(slot5) + alpha * (L(slot3) + wo * ut_r2 + wc * ut_ms45)
        wrow = eval_cand(
            f"L6_blend_a{alpha:.2f}_wo{wo:.3f}_wc{wc:.3f}",
            oof,
            te_logit,
            {"lever": "rank_blend", "alpha": alpha, "wo": wo, "wc": wc},
        )
        if wrow:
            winners.append(wrow)

    print(f"\nWinners clearing all gates: {len(winners)}")
    submit = False
    pub, ref = None, None
    chosen = None
    if winners:
        # prefer highest mean_lift among those with min_lift>0.2 (safer than micro)
        winners.sort(key=lambda r: (r["min_lift"], r["mean_lift"], r["rho"]), reverse=True)
        chosen = winners[0]
        print(
            f"CHOSEN {chosen['name']} mean={chosen['mean_lift']:+.2f} "
            f"min={chosen['min_lift']:+.2f} 1-ρ={chosen['rho']:.3f}e-4"
        )
        # rebuild cand
        meta = chosen["meta"]
        lever = meta["lever"]
        if lever == "R2_weight":
            cand = qlock(L(slot3) + meta["w"] * ut_r2, lv)
        elif lever == "anchored_ms45":
            cand = qlock(L(slot3) + meta["wo"] * ut_r2 + meta["wc"] * ut_ms45, lv)
        elif lever == "anchored_cat":
            cand = qlock(L(slot3) + meta["wo"] * ut_r2 + meta["wc"] * ut_cat, lv)
        elif lever == "triple":
            cand = qlock(
                L(slot3) + meta["wo"] * ut_r2 + meta["wc"] * ut_ms45 + meta["wl"] * ut_lgb, lv
            )
        elif lever == "U5_add":
            cand = qlock(L(slot5) + meta["w"] * ut_u5, lv)
        elif lever == "direct_S5":
            cand = qlock(L(slot5) + meta["w"] * ut_d, lv)
        elif lever == "rank_blend":
            te_logit = (1 - meta["alpha"]) * L(slot5) + meta["alpha"] * (
                L(slot3) + meta["wo"] * ut_r2 + meta["wc"] * ut_ms45
            )
            cand = qlock(te_logit, lv)
        else:
            raise SystemExit(lever)

        out_csv = SUB / f"sub_v19_3_LIVE_slot8_{chosen['name']}.csv"
        assert len(cand) == n_te and np.isfinite(cand).all()
        assert float(ks_2samp(cand, slot5).statistic) < 0.001
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(out_csv, index=False)
        msg = (
            f"v19.3 Slot8: {chosen['name']} vs Slot5 ATH | "
            f"H2H mean={chosen['mean_lift']:+.2f}e-5 min={chosen['min_lift']:+.2f} "
            f"1-ρ={chosen['rho']:.3f}e-4"
        )
        print("=== SUBMIT ===", msg)
        r = subprocess.run(
            [
                KAG,
                "competitions",
                "submit",
                "-c",
                "playground-series-s6e11",
                "-f",
                str(out_csv),
                "-m",
                msg,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        print(r.stdout)
        print(r.stderr)
        if r.returncode != 0:
            print("SUBMIT FAILED — holding remaining slots")
        else:
            submit = True
            for i in range(30):
                time.sleep(12 if i else 4)
                p = subprocess.run(
                    [
                        KAG,
                        "competitions",
                        "submissions",
                        "-c",
                        "playground-series-s6e11",
                        "--csv",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                lines = [ln for ln in p.stdout.strip().splitlines() if ln]
                if len(lines) < 2:
                    continue
                hdr = [h.strip() for h in lines[0].split(",")]
                for ln in lines[1:5]:
                    cols = ln.split(",")
                    row = {hdr[j]: cols[j].strip() for j in range(min(len(hdr), len(cols)))}
                    if "v19_3" in row.get("fileName", ""):
                        ref = row.get("ref")
                        st = row.get("status", "")
                        print(f"  poll {st} ref={ref} score={row.get('publicScore')}")
                        if "complete" in st.lower() and row.get("publicScore"):
                            pub = row["publicScore"]
                            break
                if pub:
                    break
            print(f"PUBLIC_LB={pub} (ATH Slot5=0.95394)")
    else:
        print("NO GATE-CLEARING WINNER — not submitting a desperate micro-step")

    payload = scrub(
        {
            "submit": submit,
            "n_winners": len(winners),
            "chosen": None
            if not chosen
            else {
                "name": chosen["name"],
                "mean_lift": chosen["mean_lift"],
                "min_lift": chosen["min_lift"],
                "rho": chosen["rho"],
                "corr6": chosen["corr6"],
                "corr7": chosen["corr7"],
                "per": chosen["per"],
                "meta": chosen["meta"],
            },
            "public_lb": pub,
            "ref": ref,
            "top_near_misses": sorted(
                [c for c in candidates if c["per"]["twin"]["pos"] >= 4],
                key=lambda c: (c["per"]["twin"]["pos"], c["mean_lift"], -abs(c["rho"] - 0.30)),
                reverse=True,
            )[:15],
            "elapsed_s": time.time() - t0,
        }
    )
    OUT.joinpath("v19_3.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v19_3.json", f"elapsed={time.time()-t0:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
