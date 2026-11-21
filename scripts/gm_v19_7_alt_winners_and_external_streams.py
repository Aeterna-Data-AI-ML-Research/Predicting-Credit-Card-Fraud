#!/usr/bin/env python3
"""GM V19.7 — Alt Slot8-recipe siblings + unused external streams vs ATH 0.95395.

Submit Slot9 only if 5/5×3, min>0.05, sf≥18, t≥2.8, 1−ρ∈[0.30,1.20], F2<0.035.
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
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
TMP15, TMP16 = Path("/tmp/s6e11_v15"), Path("/tmp/s6e11_v16")
OUT, SUB = TMP16 / "out", REPO / "submissions"
EXT = REPO / "external_bank/fresh_ds"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = EXT / "alexchilton_s6e11-oof-predictions"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
SLOT6 = SUB / "sub_v18_1_LIVE_slot6_Slot5_plus_R1_pure_Bo10_w0p022.csv"
SLOT7 = SUB / "sub_v19_0_LIVE_slot7_zero_echo_peer_hybrid.csv"
SLOT8 = SUB / "sub_v19_4_LIVE_slot8_A_cat_wo0.016_wc0.012_wu0.012.csv"
G26_OOF, G26_T0 = TMP16 / "g26/gpt2_s2026_oof_5fold.npy", TMP16 / "g26/gpt2_s2026_test_f0.npy"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1_LOCK, W_FE_LOCK = 0.0355, 0.01045
W8_O, W8_C, W8_U = 0.016, 0.012, 0.012


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


def main():
    t0 = time.time()
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
    slot8 = pd.read_csv(SLOT8).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
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
    u_lgb, ut_lgb = orient_bo9(fam1["F_lgb_ms39_oof"], fam1["F_lgb_ms39_test"])

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
    if roc_auc_score(y[te0], (z_s3 + W8_O * u_r2)[te0] + 0.01 * u_u5[te0]) < roc_auc_score(
        y[te0], (z_s3 + W8_O * u_r2)[te0] - 0.01 * u_u5[te0]
    ):
        u_u5, ot5 = -u_u5, -ot5
    ut_u5 = unitize(ot5)

    U6_o = L_fold(llr_o.max(1) - llr_o.min(1), fold_id)
    U6_t = np.mean(np.stack([L(llr_t[f].max(1) - llr_t[f].min(1)) for f in range(5)], 0), 0)
    oo6, ot6, _ = frozen_lofo_orth(U6_o, U6_t, Bo10, Bt10, folds, ridge_alpha=1.0)
    u_u6 = unitize(oo6)
    if roc_auc_score(y[te0], (z_s3 + W8_O * u_r2)[te0] + 0.01 * u_u6[te0]) < roc_auc_score(
        y[te0], (z_s3 + W8_O * u_r2)[te0] - 0.01 * u_u6[te0]
    ):
        u_u6, ot6 = -u_u6, -ot6
    ut_u6 = unitize(ot6)

    z_s8 = z_s3 + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_t8 = z_twin + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    z_n8 = z_nnls + W8_O * u_r2 + W8_C * u_cat + W8_U * u_u5
    bases = {"z_S8": z_s8, "twin": z_t8, "nnls": z_n8}
    Bo11 = np.column_stack([Bo9, u_r2, u_cat, u_u5])
    Bt11 = np.column_stack([Bt9, ut_r2, ut_cat, ut_u5])

    print("=== V19.7 alt winners + external vs Slot8 ===")
    winners, candidates = [], []

    def passes(per, rho, c6, c7, f2=0.0):
        return (
            all(
                per[b]["pos"] >= 5
                and per[b]["min"] > 0.05
                and per[b]["sf25"] >= 18
                and per[b]["t"] >= 2.8
                for b in per
            )
            and (0.30 <= rho <= 1.20)
            and abs(c6) < 0.20
            and abs(c7) < 0.20
            and f2 < 0.035
        )

    def eval_te(name, te_logit, oof, meta, f2=0.0):
        per = {b: h2h(oof[b], bases[b], y, folds) for b in bases}
        cand = qlock(te_logit, lv)
        rho = one_m_rho_e4(cand, slot8)
        delta = unitize(L(cand) - L(slot8))
        c6 = float(np.corrcoef(delta, bad6)[0, 1])
        c7 = float(np.corrcoef(delta, bad7)[0, 1])
        ok = passes(per, rho, c6, c7, f2)
        row = {
            "name": name,
            "per": per,
            "rho": rho,
            "corr6": c6,
            "corr7": c7,
            "f2": f2,
            "ok": ok,
            "mean_lift": float(np.mean([per[b]["mean"] for b in per])),
            "min_lift": float(np.min([per[b]["min"] for b in per])),
            "meta": meta,
            "cand": cand if ok else None,
        }
        mark = "★" if ok else ("~" if per["z_S8"]["pos"] >= 4 else " ")
        print(
            f"{mark} {name}: 1-ρ={rho:.3f} f2={f2:.3f} ok={ok} | "
            + " | ".join(
                f"{b} {per[b]['pos']}/5 m={per[b]['mean']:+.2f} min={per[b]['min']:+.2f} "
                f"sf{per[b]['sf25']} t={per[b]['t']:.2f}"
                for b in bases
            )
        )
        candidates.append({k: v for k, v in row.items() if k != "cand"})
        if ok:
            winners.append(row)
        return row

    # A: alt V19.4 recipes vs Slot8
    print("-- A alt recipes --")
    alts = [
        ("alt_wo018_wc012_wu010", 0.018, 0.012, 0.010, u_cat, ut_cat),
        ("alt_wo020_wc012_wu012", 0.020, 0.012, 0.012, u_cat, ut_cat),
        ("alt_ms45_wo020_wc012_wu012", 0.020, 0.012, 0.012, u_ms45, ut_ms45),
        ("alt_u6_wo020_wc012_wu012", 0.020, 0.012, 0.012, u_cat, ut_cat),  # U6 instead of U5 below
    ]
    for name, wo, wc, wu, uc, utc in alts[:3]:
        oof = {
            "z_S8": z_s3 + wo * u_r2 + wc * uc + wu * u_u5,
            "twin": z_twin + wo * u_r2 + wc * uc + wu * u_u5,
            "nnls": z_nnls + wo * u_r2 + wc * uc + wu * u_u5,
        }
        eval_te(
            name,
            L(slot3) + wo * ut_r2 + wc * utc + wu * ut_u5,
            oof,
            {"lever": "alt", "wo": wo, "wc": wc, "wu": wu, "uc": "ms45" if "ms45" in name else "cat"},
        )
    # U6 variant
    oof = {
        "z_S8": z_s3 + 0.020 * u_r2 + 0.012 * u_cat + 0.012 * u_u6,
        "twin": z_twin + 0.020 * u_r2 + 0.012 * u_cat + 0.012 * u_u6,
        "nnls": z_nnls + 0.020 * u_r2 + 0.012 * u_cat + 0.012 * u_u6,
    }
    eval_te(
        "alt_u6_wo020_wc012_wu012",
        L(slot3) + 0.020 * ut_r2 + 0.012 * ut_cat + 0.012 * ut_u6,
        oof,
        {"lever": "alt_u6", "wo": 0.020, "wc": 0.012, "wu": 0.012},
    )

    # B: Slot8 + lgb39 with Bo11 orth (re-orth lgb)
    print("-- B Slot8 + orth(lgb|Bo11) --")
    oo, ot, _ = frozen_lofo_orth(
        fam1["F_lgb_ms39_oof"], fam1["F_lgb_ms39_test"], Bo11, Bt11, folds, 10.0
    )
    u, ut = unitize(oo), unitize(ot)
    if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
        y[te0], z_s8[te0] - 0.01 * u[te0]
    ):
        u, ut = -u, -ut
    f2 = max(abs(float(np.corrcoef(unitize(ot), Bt11[:, j])[0, 1])) for j in range(Bt11.shape[1]))
    print(f"  lgb|Bo11 F2={f2:.4f} corrR2={np.corrcoef(u,u_r2)[0,1]:+.3f}")
    for w in (0.006, 0.008, 0.010, 0.012, 0.015, 0.018):
        oof = {"z_S8": z_s8 + w * u, "twin": z_t8 + w * u, "nnls": z_n8 + w * u}
        eval_te(
            f"B_lgb_bo11_w{w:.3f}",
            L(slot8) + w * ut,
            oof,
            {"lever": "lgb_bo11", "w": w, "ut": "lgb"},
            f2=f2,
        )

    # C: R4
    print("-- C R4 --")
    R4_o = np.load(TMP15 / "out/R4_oof.npy").ravel()
    R4_t = np.mean(np.load(TMP15 / "out/R4_test_folds.npy"), 0)
    oo, ot, _ = frozen_lofo_orth(R4_o, R4_t, Bo11, Bt11, folds, 10.0)
    u, ut = unitize(oo), unitize(ot)
    if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
        y[te0], z_s8[te0] - 0.01 * u[te0]
    ):
        u, ut = -u, -ut
    f2 = max(abs(float(np.corrcoef(unitize(ot), Bt11[:, j])[0, 1])) for j in range(Bt11.shape[1]))
    print(f"  R4|Bo11 F2={f2:.4f}")
    for w in (0.006, 0.008, 0.010, 0.012, 0.015, 0.020):
        oof = {"z_S8": z_s8 + w * u, "twin": z_t8 + w * u, "nnls": z_n8 + w * u}
        eval_te(f"C_R4_w{w:.3f}", L(slot8) + w * ut, oof, {"lever": "R4", "w": w}, f2=f2)

    # D: external submissions as rank-logit signals (test-only → need OOF proxy via train blend)
    # Use OOF npz where available
    print("-- D external OOFs --")
    ext_cands = []

    # wguesdon
    for npz_name in ("raw_lightgbm.npz", "raw_catboost.npz", "encoded_lightgbm.npz", "shallow_lightgbm.npz"):
        p = EXT / "wguesdon_trees" / npz_name
        if not p.exists():
            continue
        z = np.load(p)
        keys = list(z.files)
        # try common patterns
        oof_k = next((k for k in keys if "oof" in k.lower()), None)
        te_k = next((k for k in keys if "test" in k.lower() or "pred" in k.lower()), None)
        if oof_k is None:
            print(f"  skip {npz_name} keys={keys[:8]}")
            continue
        vo = z[oof_k].ravel()[:n_tr]
        if te_k:
            vt = z[te_k].ravel()[:n_te]
        else:
            continue
        if len(vo) != n_tr or len(vt) != n_te:
            print(f"  skip {npz_name} shape oof={vo.shape} te={vt.shape}")
            continue
        ext_cands.append((f"wguesdon_{npz_name}", vo, vt))

    # queyrusi cnn
    for npz_name in ("digit_pair_stump.npz", "oof_stack_reference.npz"):
        p = EXT / "queyrusi_cnn" / npz_name
        if p.exists():
            z = np.load(p)
            print(f"  queyrusi {npz_name} keys={z.files}")
            # handle later if useful

    # malyshev - only CSVs of submissions; skip OOF

    # georgymamarin stage25
    gpath = EXT / "georgymamarin_s6e11-audited-glm-offset-blend/stage25_recorded_submission.csv"
    if gpath.exists():
        gdf = pd.read_csv(gpath)
        # test-only; use as additive after projecting via ridge on Slot8 (no OOF — skip live)
        print(f"  georgymamarin stage25 cols={list(gdf.columns)} n={len(gdf)} (test-only, skip H2H)")

    for ename, vo, vt in ext_cands:
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo11, Bt11, folds, 10.0)
        u, ut = unitize(oo), unitize(ot)
        if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
            y[te0], z_s8[te0] - 0.01 * u[te0]
        ):
            u, ut = -u, -ut
        f2 = max(abs(float(np.corrcoef(unitize(ot), Bt11[:, j])[0, 1])) for j in range(Bt11.shape[1]))
        disp = float(np.std(ot) / (np.std(oo) + 1e-12))
        print(f"  {ename}: F2={f2:.4f} disp={disp:.3f}")
        if f2 >= 0.035 or not (0.90 <= disp <= 1.10):
            continue
        for w in (0.008, 0.010, 0.012, 0.015, 0.020):
            oof = {"z_S8": z_s8 + w * u, "twin": z_t8 + w * u, "nnls": z_n8 + w * u}
            eval_te(
                f"D_{ename}_w{w:.3f}",
                L(slot8) + w * ut,
                oof,
                {"lever": "ext", "ename": ename, "w": w},
                f2=f2,
            )

    # E: Slot8 + tiny ms45 residual (ms45 - cat direction)
    print("-- E Slot8 + orth(ms45 - proj_cat) --")
    # residualize ms45 against [Bo11] already; just add oriented ms45|Bo11
    oo, ot, _ = frozen_lofo_orth(
        fam2["F_cat_ms45_oof"], fam2["F_cat_ms45_test"], Bo11, Bt11, folds, 10.0
    )
    u, ut = unitize(oo), unitize(ot)
    if roc_auc_score(y[te0], z_s8[te0] + 0.01 * u[te0]) < roc_auc_score(
        y[te0], z_s8[te0] - 0.01 * u[te0]
    ):
        u, ut = -u, -ut
    f2 = max(abs(float(np.corrcoef(unitize(ot), Bt11[:, j])[0, 1])) for j in range(Bt11.shape[1]))
    print(f"  ms45|Bo11 F2={f2:.4f} corrCat={np.corrcoef(u,u_cat)[0,1]:+.3f}")
    for w in (0.004, 0.006, 0.008, 0.010, 0.012, 0.015):
        oof = {"z_S8": z_s8 + w * u, "twin": z_t8 + w * u, "nnls": z_n8 + w * u}
        eval_te(f"E_ms45_bo11_w{w:.3f}", L(slot8) + w * ut, oof, {"lever": "ms45_bo11", "w": w}, f2=f2)

    # F: raise U5 a bit while bumping Cat (known strong direction from Slot8 win)
    print("-- F denser Cat/U5 around Slot8 with larger step --")
    for wo, wc, wu in [
        (0.015, 0.014, 0.014),
        (0.014, 0.014, 0.015),
        (0.016, 0.014, 0.014),
        (0.016, 0.013, 0.015),
        (0.017, 0.013, 0.014),
        (0.015, 0.013, 0.015),
        (0.014, 0.015, 0.014),
        (0.016, 0.015, 0.013),
    ]:
        oof = {
            "z_S8": z_s3 + wo * u_r2 + wc * u_cat + wu * u_u5,
            "twin": z_twin + wo * u_r2 + wc * u_cat + wu * u_u5,
            "nnls": z_nnls + wo * u_r2 + wc * u_cat + wu * u_u5,
        }
        eval_te(
            f"F_wo{wo:.3f}_wc{wc:.3f}_wu{wu:.3f}",
            L(slot3) + wo * ut_r2 + wc * ut_cat + wu * ut_u5,
            oof,
            {"lever": "dense", "wo": wo, "wc": wc, "wu": wu},
        )

    print(f"\nWinners: {len(winners)}")
    submit = False
    pub, ref, chosen = None, None, None
    if winners:
        winners.sort(key=lambda r: (r["min_lift"], r["mean_lift"], -abs(r["rho"] - 0.50)), reverse=True)
        chosen = winners[0]
        print(
            f"CHOSEN {chosen['name']} mean={chosen['mean_lift']:+.2f} "
            f"min={chosen['min_lift']:+.2f} 1-ρ={chosen['rho']:.3f}e-4"
        )
        cand = chosen["cand"]
        out_csv = SUB / f"sub_v19_7_LIVE_slot9_{chosen['name']}.csv"
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(out_csv, index=False)
        msg = (
            f"v19.7 Slot9: {chosen['name']} vs Slot8 ATH 0.95395 | "
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
                    if "v19_7" in row.get("fileName", ""):
                        ref = row.get("ref")
                        st = row.get("status", "")
                        print(f"  poll {st} ref={ref} score={row.get('publicScore')}")
                        if "complete" in st.lower() and row.get("publicScore"):
                            pub = row["publicScore"]
                            break
                if pub:
                    break
            print(f"PUBLIC_LB={pub} (ATH Slot8=0.95395)")
    else:
        print("NO WINNER — holding Slot8")

    near = sorted(
        candidates,
        key=lambda c: (c["per"]["z_S8"]["pos"], c["per"]["twin"]["pos"], c["mean_lift"]),
        reverse=True,
    )[:15]
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
                "per": chosen["per"],
                "meta": chosen["meta"],
            },
            "public_lb": pub,
            "ref": ref,
            "near_misses": near,
            "elapsed_s": time.time() - t0,
        }
    )
    OUT.joinpath("v19_7.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v19_7.json", f"elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
