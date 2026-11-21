#!/usr/bin/env python3
"""GM V19.4 — Break the ρ↔twin antagonism with U5 diversity + corrected distance band.

Evidence:
  LIVE Slot5 WIN vs Slot3: 1−ρ = 0.763e-4
  LIVE Slot7 FAIL vs Slot5: 1−ρ = 0.148e-4 (micro)
  LIVE Slot4 FAIL vs Slot3: 1−ρ = 0.194e-4
  LIVE Slot6 FAIL vs Slot5: 1−ρ = 0.962e-4 (wrong family)

So the old [0.15,0.55] band rejected Slot5-scale steps and still allowed Slot7 micros.
New band vs Slot5: [0.35, 1.20]e-4 (Slot5-like step, below Slot6-scale poison).

Levers:
  A: anchored Cat + tiny U5 (U5 ⊥ R2 → inflate Spearman without killing twin)
  B: anchored ms45 + U5
  C: Slot5 + U5 alone at higher w with relaxed twin t if z_S5+nnls elite
  D: high-Cat then rank-blend back toward Slot5 to restore twin
  E: Cat_intrabin anchored
  F: soft twin (t≥2.2, sf≥17) IF z_S5+nnls both elite AND 1−ρ∈[0.50,1.00]
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
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
TMP15, TMP16 = Path("/tmp/s6e11_v15"), Path("/tmp/s6e11_v16")
OUT, SUB, LOG = TMP16 / "out", REPO / "submissions", REPO / "logs/overnight"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
SLOT6 = SUB / "sub_v18_1_LIVE_slot6_Slot5_plus_R1_pure_Bo10_w0p022.csv"
SLOT7 = SUB / "sub_v19_0_LIVE_slot7_zero_echo_peer_hybrid.csv"
G26_OOF, G26_T0 = TMP16 / "g26/gpt2_s2026_oof_5fold.npy", TMP16 / "g26/gpt2_s2026_test_f0.npy"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK, W_R2 = 0.0355, 0.01045, 0.020


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


def gate_strict(per, rho, c6, c7):
    ok_h = all(
        per[b]["pos"] >= 5 and per[b]["min"] > 0.05 and per[b]["sf25"] >= 18 and per[b]["t"] >= 2.8
        for b in per
    )
    return ok_h and (0.35 <= rho <= 1.20) and abs(c6) < 0.20 and abs(c7) < 0.20


def gate_soft_twin(per, rho, c6, c7):
    """z_S5+nnls elite; twin allowed softer if distance is Slot5-scale."""
    hard = ("z_S5", "nnls")
    ok_h = all(
        per[b]["pos"] >= 5 and per[b]["min"] > 0.20 and per[b]["sf25"] >= 19 and per[b]["t"] >= 3.0
        for b in hard
    )
    ok_t = (
        per["twin"]["pos"] >= 4
        and per["twin"]["min"] > -0.50
        and per["twin"]["sf25"] >= 16
        and per["twin"]["t"] >= 1.8
        and per["twin"]["mean"] > 0.5
    )
    return ok_h and ok_t and (0.50 <= rho <= 1.00) and abs(c6) < 0.20 and abs(c7) < 0.20


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

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
    u_ms45, ut_ms45 = orient_bo9(fam2["F_cat_ms45_oof"], fam2["F_cat_ms45_test"])
    u_intra, ut_intra = orient_bo9(fam2["F_cat_intrabin_oof"], fam2["F_cat_intrabin_test"])
    u_lgb, ut_lgb = orient_bo9(fam1["F_lgb_ms39_oof"], fam1["F_lgb_ms39_test"])

    z_s5 = z_s3 + W_R2 * u_r2
    z_t5 = z_twin + W_R2 * u_r2
    z_n5 = z_nnls + W_R2 * u_r2
    bases_s5 = {"z_S5": z_s5, "twin": z_t5, "nnls": z_n5}

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

    # U6 = fold-aware KS-style dispersion of llr ranks (if available via max-min)
    U6_o = L_fold(llr_o.max(1) - llr_o.min(1), fold_id)
    U6_t = np.mean(np.stack([L(llr_t[f].max(1) - llr_t[f].min(1)) for f in range(5)], 0), 0)
    oo6, ot6, _ = frozen_lofo_orth(U6_o, U6_t, Bo10, Bt10, folds, ridge_alpha=1.0)
    u_u6 = unitize(oo6)
    if roc_auc_score(y[te0], z_s5[te0] + 0.01 * u_u6[te0]) < roc_auc_score(
        y[te0], z_s5[te0] - 0.01 * u_u6[te0]
    ):
        u_u6, ot6 = -u_u6, -ot6
    ut_u6 = unitize(ot6)

    print("=== V19.4 Cat+U5 diversity / corrected distance band ===")
    print(
        f"corr U5/R2={np.corrcoef(u_u5,u_r2)[0,1]:+.4f} U6/R2={np.corrcoef(u_u6,u_r2)[0,1]:+.4f} "
        f"U5/Cat={np.corrcoef(u_u5,u_cat)[0,1]:+.4f} U6/Cat={np.corrcoef(u_u6,u_cat)[0,1]:+.4f}"
    )
    print(f"LIVE ref: S5vsS3 1-ρ={one_m_rho_e4(slot5,slot3):.3f}e-4")

    candidates = []
    winners = []

    def eval_cand(name, oof, te_logit, meta):
        per = {b: h2h(oof[b], bases_s5[b], y, folds) for b in bases_s5}
        cand = qlock(te_logit, lv)
        rho = one_m_rho_e4(cand, slot5)
        delta = unitize(L(cand) - L(slot5))
        c6 = float(np.corrcoef(delta, bad6)[0, 1])
        c7 = float(np.corrcoef(delta, bad7)[0, 1])
        gs = gate_strict(per, rho, c6, c7)
        gf = gate_soft_twin(per, rho, c6, c7)
        ok = gs or gf
        row = {
            "name": name,
            "per": per,
            "rho": rho,
            "corr6": c6,
            "corr7": c7,
            "gate_strict": gs,
            "gate_soft": gf,
            "ok": ok,
            "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
            "min_lift": float(np.min([per[b]["min"] for b in per])),
            "meta": meta,
            "cand": cand if ok else None,
        }
        mark = "★" if ok else ("~" if per["z_S5"]["pos"] == 5 and per["nnls"]["pos"] == 5 else " ")
        print(
            f"{mark} {name}: 1-ρ={rho:.3f}e-4 strict={gs} soft={gf} | "
            + " | ".join(
                f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f} "
                f"sf{per[b]['sf25']} t={per[b]['t']:.2f}"
                for b in bases_s5
            )
        )
        candidates.append({k: v for k, v in row.items() if k != "cand"})
        if ok:
            winners.append(row)
        return row

    # A: Cat + U5
    print("-- A Cat_obliv + U5 --")
    for wo in (0.016, 0.018, 0.020):
        for wc in (0.006, 0.008, 0.010, 0.012):
            for wu in (0.004, 0.006, 0.008, 0.010, 0.012, 0.015):
                oof = {
                    b: (z_s3 if b == "z_S5" else (z_twin if b == "twin" else z_nnls))
                    + wo * u_r2
                    + wc * u_cat
                    + wu * u_u5
                    for b in bases_s5
                }
                # fix keys properly
                oof = {
                    "z_S5": z_s3 + wo * u_r2 + wc * u_cat + wu * u_u5,
                    "twin": z_twin + wo * u_r2 + wc * u_cat + wu * u_u5,
                    "nnls": z_nnls + wo * u_r2 + wc * u_cat + wu * u_u5,
                }
                eval_cand(
                    f"A_cat_wo{wo:.3f}_wc{wc:.3f}_wu{wu:.3f}",
                    oof,
                    L(slot3) + wo * ut_r2 + wc * ut_cat + wu * ut_u5,
                    {"lever": "cat_u5", "wo": wo, "wc": wc, "wu": wu},
                )

    # B: ms45 + U5
    print("-- B ms45 + U5 --")
    for wo in (0.018, 0.020):
        for wc in (0.008, 0.010, 0.012):
            for wu in (0.006, 0.008, 0.010, 0.012):
                oof = {
                    "z_S5": z_s3 + wo * u_r2 + wc * u_ms45 + wu * u_u5,
                    "twin": z_twin + wo * u_r2 + wc * u_ms45 + wu * u_u5,
                    "nnls": z_nnls + wo * u_r2 + wc * u_ms45 + wu * u_u5,
                }
                eval_cand(
                    f"B_ms45_wo{wo:.3f}_wc{wc:.3f}_wu{wu:.3f}",
                    oof,
                    L(slot3) + wo * ut_r2 + wc * ut_ms45 + wu * ut_u5,
                    {"lever": "ms45_u5", "wo": wo, "wc": wc, "wu": wu},
                )

    # C: Cat + U6
    print("-- C Cat + U6 --")
    for wo in (0.018, 0.020):
        for wc in (0.008, 0.010, 0.012):
            for wu in (0.006, 0.008, 0.010, 0.012):
                oof = {
                    "z_S5": z_s3 + wo * u_r2 + wc * u_cat + wu * u_u6,
                    "twin": z_twin + wo * u_r2 + wc * u_cat + wu * u_u6,
                    "nnls": z_nnls + wo * u_r2 + wc * u_cat + wu * u_u6,
                }
                eval_cand(
                    f"C_cat_u6_wo{wo:.3f}_wc{wc:.3f}_wu{wu:.3f}",
                    oof,
                    L(slot3) + wo * ut_r2 + wc * ut_cat + wu * ut_u6,
                    {"lever": "cat_u6", "wo": wo, "wc": wc, "wu": wu},
                )

    # D: high-Cat then blend toward Slot5
    print("-- D high-Cat blend-back --")
    for wo, wc in [(0.018, 0.015), (0.020, 0.015), (0.018, 0.018), (0.016, 0.015), (0.020, 0.012)]:
        boost_o = {
            "z_S5": z_s3 + wo * u_r2 + wc * u_cat,
            "twin": z_twin + wo * u_r2 + wc * u_cat,
            "nnls": z_nnls + wo * u_r2 + wc * u_cat,
        }
        boost_t = L(slot3) + wo * ut_r2 + wc * ut_cat
        for a in (0.55, 0.65, 0.75, 0.85, 0.90):
            oof = {b: a * L(boost_o[b]) + (1 - a) * L(bases_s5[b]) for b in bases_s5}
            te = a * boost_t + (1 - a) * L(slot5)
            eval_cand(
                f"D_blend_a{a:.2f}_wo{wo:.3f}_wc{wc:.3f}",
                oof,
                te,
                {"lever": "blend_back", "alpha": a, "wo": wo, "wc": wc},
            )

    # E: Cat_intrabin
    print("-- E Cat_intrabin + optional U5 --")
    for wo in (0.018, 0.020):
        for wc in (0.008, 0.010, 0.012, 0.015):
            for wu in (0.0, 0.006, 0.010):
                oof = {
                    "z_S5": z_s3 + wo * u_r2 + wc * u_intra + wu * u_u5,
                    "twin": z_twin + wo * u_r2 + wc * u_intra + wu * u_u5,
                    "nnls": z_nnls + wo * u_r2 + wc * u_intra + wu * u_u5,
                }
                eval_cand(
                    f"E_intra_wo{wo:.3f}_wc{wc:.3f}_wu{wu:.3f}",
                    oof,
                    L(slot3) + wo * ut_r2 + wc * ut_intra + wu * ut_u5,
                    {"lever": "intrabin", "wo": wo, "wc": wc, "wu": wu},
                )

    # F: Cat + lgb39 + U5 (triple + diversity)
    print("-- F Cat+lgb39+U5 --")
    for wo, wc, wl, wu in [
        (0.018, 0.008, 0.003, 0.008),
        (0.018, 0.010, 0.003, 0.010),
        (0.020, 0.008, 0.003, 0.010),
        (0.018, 0.008, 0.005, 0.012),
        (0.016, 0.010, 0.005, 0.010),
    ]:
        oof = {
            "z_S5": z_s3 + wo * u_r2 + wc * u_cat + wl * u_lgb + wu * u_u5,
            "twin": z_twin + wo * u_r2 + wc * u_cat + wl * u_lgb + wu * u_u5,
            "nnls": z_nnls + wo * u_r2 + wc * u_cat + wl * u_lgb + wu * u_u5,
        }
        eval_cand(
            f"F_wo{wo:.3f}_wc{wc:.3f}_wl{wl:.3f}_wu{wu:.3f}",
            oof,
            L(slot3) + wo * ut_r2 + wc * ut_cat + wl * ut_lgb + wu * ut_u5,
            {"lever": "triple_u5", "wo": wo, "wc": wc, "wl": wl, "wu": wu},
        )

    print(f"\nWinners: {len(winners)}")
    submit = False
    pub, ref = None, None
    chosen = None
    if winners:
        # prefer strict gate, then soft; among those prefer rho near 0.70 (Slot5-scale) and high min_lift
        def score(r):
            return (
                int(r["gate_strict"]) * 10
                + int(r["gate_soft"]) * 3
                + min(r["rho"], 1.0) * 2
                - abs(r["rho"] - 0.70)
                + r["min_lift"] * 0.1
                + r["mean_lift"] * 0.05
            )

        winners.sort(key=score, reverse=True)
        chosen = winners[0]
        print(
            f"CHOSEN {chosen['name']} strict={chosen['gate_strict']} soft={chosen['gate_soft']} "
            f"mean={chosen['mean_lift']:+.2f} min={chosen['min_lift']:+.2f} 1-ρ={chosen['rho']:.3f}e-4"
        )
        meta = chosen["meta"]
        lever = meta["lever"]
        if lever == "cat_u5":
            cand = qlock(
                L(slot3) + meta["wo"] * ut_r2 + meta["wc"] * ut_cat + meta["wu"] * ut_u5, lv
            )
        elif lever == "ms45_u5":
            cand = qlock(
                L(slot3) + meta["wo"] * ut_r2 + meta["wc"] * ut_ms45 + meta["wu"] * ut_u5, lv
            )
        elif lever == "cat_u6":
            cand = qlock(
                L(slot3) + meta["wo"] * ut_r2 + meta["wc"] * ut_cat + meta["wu"] * ut_u6, lv
            )
        elif lever == "blend_back":
            boost_t = L(slot3) + meta["wo"] * ut_r2 + meta["wc"] * ut_cat
            cand = qlock(meta["alpha"] * boost_t + (1 - meta["alpha"]) * L(slot5), lv)
        elif lever == "intrabin":
            cand = qlock(
                L(slot3) + meta["wo"] * ut_r2 + meta["wc"] * ut_intra + meta["wu"] * ut_u5, lv
            )
        elif lever == "triple_u5":
            cand = qlock(
                L(slot3)
                + meta["wo"] * ut_r2
                + meta["wc"] * ut_cat
                + meta["wl"] * ut_lgb
                + meta["wu"] * ut_u5,
                lv,
            )
        else:
            raise SystemExit(lever)

        out_csv = SUB / f"sub_v19_4_LIVE_slot8_{chosen['name']}.csv"
        assert len(cand) == n_te and np.isfinite(cand).all()
        assert float(ks_2samp(cand, slot5).statistic) < 0.002
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(out_csv, index=False)
        msg = (
            f"v19.4 Slot8: {chosen['name']} vs Slot5 ATH | "
            f"H2H mean={chosen['mean_lift']:+.2f}e-5 min={chosen['min_lift']:+.2f} "
            f"1-ρ={chosen['rho']:.3f}e-4 strict={chosen['gate_strict']}"
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
        if r.returncode == 0:
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
                    if "v19_4" in row.get("fileName", ""):
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
            print("SUBMIT FAILED")
    else:
        print("NO WINNER — holding")

    near = sorted(
        [c for c in candidates if c["per"]["z_S5"]["pos"] >= 5 and c["per"]["nnls"]["pos"] >= 5],
        key=lambda c: (
            int(c["gate_strict"]),
            int(c["gate_soft"]),
            c["per"]["twin"]["pos"],
            -abs(c["rho"] - 0.70),
            c["mean_lift"],
        ),
        reverse=True,
    )[:20]
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
                "gate_strict": chosen["gate_strict"],
                "gate_soft": chosen["gate_soft"],
                "corr6": chosen["corr6"],
                "corr7": chosen["corr7"],
                "per": chosen["per"],
                "meta": chosen["meta"],
            },
            "public_lb": pub,
            "ref": ref,
            "near_misses": near,
            "elapsed_s": time.time() - t0,
        }
    )
    OUT.joinpath("v19_4.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v19_4.json", f"elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
