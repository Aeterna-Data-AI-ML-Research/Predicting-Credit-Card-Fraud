#!/usr/bin/env python3
"""GM V17.6 — G26−LM risk-gated audit + fixed non-negative Slot3 twin.

STRICT NO_SUBMIT=True. 0 GPU pods. Local CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata, t as tdist, wilcoxon
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
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
NO_SUBMIT = True
RIDGE_ALPHA = 10.0
BN9 = ["MS", "G26", "LM", "L7F", "GPD", "GB", "MLP", "d1_v4", "Family_E"]
W_D1, W_FE = 0.0355, 0.01045
SIG_1F = 0.0727
STAGE_MIN_SF, STAGE_MIN_T = 20, 2.50
MC_DRAWS = 8


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


def frozen_lofo_orth(v_oof, v_test, Bo, Bt, folds, ridge_alpha=RIDGE_ALPHA):
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


def forensic_gates(orth_o, orth_t, Bt, live_resid):
    disp = float((orth_t.std() + 1e-12) / (orth_o.std() + 1e-12))
    leaks = [float(np.corrcoef(orth_t, Bt[:, j])[0, 1]) for j in range(Bt.shape[1])]
    max_leak = float(np.max(np.abs(leaks)))
    c_live = float(np.corrcoef(orth_t, live_resid)[0, 1]) if live_resid.std() > 1e-12 else 0.0
    return {
        "disp": disp,
        "max_basis_leak": max_leak,
        "leak_argmax": int(np.argmax(np.abs(leaks))),
        "corr_live_resid": c_live,
        "F1": 0.94 <= disp <= 1.06,
        "F2": max_leak < 0.035,
        "F3": abs(c_live) < 0.05,
        "verified": (0.94 <= disp <= 1.06) and (max_leak < 0.035) and (abs(c_live) < 0.05),
    }


def orient_plus(u, zbase, y, te0):
    u = unitize(u)
    flipped = False
    if roc_auc_score(y[te0], zbase[te0] + 0.01 * u[te0]) < roc_auc_score(
        y[te0], zbase[te0] - 0.01 * u[te0]
    ):
        u, flipped = -u, True
    return u, flipped


def score_at_w(u, zbase, y, folds, w=0.035):
    base = [roc_auc_score(y[te], zbase[te]) for _, te in folds]
    d = [
        (roc_auc_score(y[te], zbase[te] + w * u[te]) - base[i]) * 1e5
        for i, (_, te) in enumerate(folds)
    ]
    deltas = []
    for _, te in folds:
        yf, zb, uf = y[te], zbase[te], u[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(yf)), yf
        ):
            b = roc_auc_score(yf[ste], zb[ste])
            deltas.append((roc_auc_score(yf[ste], zb[ste] + 0.01 * uf[ste]) - b) * 1e5)
    d25 = np.asarray(deltas, float)
    mean, se = float(d25.mean()), float(d25.std(ddof=1) / np.sqrt(len(d25)))
    t = mean / (se + 1e-18)
    try:
        _, wp = wilcoxon(d25, alternative="greater")
        wp = float(wp)
    except Exception:
        wp = float("nan")
    tcrit = float(tdist.ppf(0.95, len(d25) - 1))
    return {
        "w": w,
        "pos": int(sum(x > 0 for x in d)),
        "mean": float(np.mean(d)),
        "min": float(np.min(d)),
        "folds": [round(x, 3) for x in d],
        "sf25": int((d25 > 0).sum()),
        "t": float(t),
        "mean_sf": mean,
        "wilcoxon_p": wp,
        "ci90": [float(mean - tcrit * se), float(mean + tcrit * se)],
    }


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


def simplex_lofo(Bo, y, folds):
    """OOF non-negative simplex (w>=0, sum=1) via NNLS on [X, ones] then renormalize."""
    oof = np.zeros(len(y))
    ws = []
    for tr, te in folds:
        X = np.column_stack([Bo[tr], np.ones(len(tr))])
        # target: unitize labels as soft? Use y as float for NNLS vs L? Use y directly in {0,1}
        # Better: NNLS to match a strong proxy — use Ridge-positive via nnls on y
        coef, _ = nnls(X, y[tr].astype(float))
        w, intercept = coef[:-1], coef[-1]
        if w.sum() > 1e-12:
            w = w / w.sum()
        else:
            w = np.ones(Bo.shape[1]) / Bo.shape[1]
        ws.append(w)
        oof[te] = Bo[te] @ w  # drop intercept for rank-AUC; level free
    return oof, np.mean(np.stack(ws, 0), 0)


def main():
    assert NO_SUBMIT
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()
    dE_cache = unitize(cache["dE"]) if "dE" in cache.files else None
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    te0 = folds[0][1]

    Z = np.load(Q17, allow_pickle=True)
    qnames = [str(s) for s in Z["stream_names"]]
    XO = {n: Z[f"oof_{n}"].ravel() for n in qnames}
    XT = {n: Z[f"test_{n}"].ravel() for n in qnames}

    live = pd.read_csv(LIVE)
    ids = live["id"].to_numpy()
    lv = live["is_fraud"].to_numpy(float)
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    l7f_o, l7f_t = XO["lm7_5fold"], XT["lm7_5fold"]
    R3_t_L = np.stack([L(R3_te[f]) for f in range(5)], 0)

    # Family_E
    z_ath_t = L(lv)
    raw_E_o = (L(R3) - L(l7f_o)) * (0.5 + 0.5 * U(z_ath))
    raw_E_t = (L(R3_t_L.mean(0)) - L(l7f_t)) * (0.5 + 0.5 * U(lv))
    Xo_e = np.column_stack([unitize(z_ath), unitize(d1)])
    Xt_e = np.column_stack([unitize(z_ath_t), unitize(d1t)])
    dE_o2, dE_t, _ = frozen_lofo_orth(raw_E_o, raw_E_t, Xo_e, Xt_e, folds, ridge_alpha=1.0)
    dE_o2, dE_t = unitize(dE_o2), unitize(dE_t)
    if roc_auc_score(y[te0], z_ath[te0] + 0.01 * dE_o2[te0]) < roc_auc_score(
        y[te0], z_ath[te0] - 0.01 * dE_o2[te0]
    ):
        dE_o2, dE_t = -dE_o2, -dE_t
    dE_o = dE_cache if dE_cache is not None else dE_o2
    if np.corrcoef(dE_o, dE_o2)[0, 1] < 0:
        dE_o = -dE_o

    z_s3 = z_ath + W_FE * dE_o + 0.0055 * unitize(d1)
    # Slot3 known locked layers use 0.0355 d1 (Slot3) — keep FE at 0.01045 for twin residual
    print(f"AUC z_S3={roc_auc_score(y,z_s3):.7f}")

    ms_o = np.load(REPO / "preds/P_super_v4_oof.npy").ravel()
    ms_t = np.load(REPO / "preds/P_super_v4_test.npy").ravel()
    g26_o = np.load(G26_OOF).ravel()[: len(y)]
    g26_t = np.load(G26_T0).ravel()[:299984]
    lm_o, lm_t = XO["megayak_5f_lm"], XT["megayak_5f_lm"]
    gpd_o, gpd_t = XO["daoviet_gpdvae"], XT["daoviet_gpdvae"]
    gb_o, gb_t = XO["gb_bag5"], XT["gb_bag5"]
    mlp_o, mlp_t = XO["yekenot_realmlp"], XT["yekenot_realmlp"]

    z_MS_o, z_MS_t = L_fold(ms_o, fold_id), L(ms_t)
    z_G26_o, z_G26_t = L_fold(g26_o, fold_id), L(g26_t)
    z_LM_o, z_LM_t = L_fold(lm_o, fold_id), L(lm_t)

    Bo9 = np.column_stack(
        [
            unitize(L(ms_o)),
            unitize(L(g26_o)),
            unitize(L(lm_o)),
            unitize(L(l7f_o)),
            unitize(gpd_o),
            unitize(gb_o),
            unitize(mlp_o),
            unitize(d1),
            unitize(dE_o),
        ]
    )
    Bt9 = np.column_stack(
        [
            unitize(L(ms_t)),
            unitize(L(g26_t)),
            unitize(L(lm_t)),
            unitize(L(l7f_t)),
            unitize(gpd_t),
            unitize(gb_t),
            unitize(mlp_t),
            unitize(d1t),
            unitize(dE_t),
        ]
    )

    # ----- Step 1: baselines -----
    print("=== STEP 1: baselines ===")
    # Expanded bank for NNLS (exclude raw d1/FE cols — locked outside)
    bank_o_cols, bank_t_cols, bank_names = [], [], []

    def add_bank(name, o, t):
        bank_names.append(name)
        bank_o_cols.append(unitize(o))
        bank_t_cols.append(unitize(t))

    # Bo9 without d1 (idx7) and FE (idx8)
    for j, nm in enumerate(BN9):
        if nm in ("d1_v4", "Family_E"):
            continue
        add_bank(f"Bo9_{nm}", Bo9[:, j], Bt9[:, j])
    # Q17 extras not already in Bo9
    skip = {"megayak_5f_lm", "lm7_5fold", "gpt2_seed2026", "masaya_gpt2", "daoviet_gpdvae",
            "gb_bag5", "yekenot_realmlp"}
    for n in qnames:
        if n in skip:
            continue
        add_bank(f"Q17_{n}", XO[n], XT[n])
    # alex matched pairs (cap 12 for speed)
    alex_oofs = sorted(ALEX.glob("oof_*.npy"))[:12]
    for p in alex_oofs:
        tp = ALEX / ("test_" + p.name[len("oof_"):])
        if tp.exists():
            add_bank(f"alex_{p.stem[-8:]}", np.load(p).ravel()[: len(y)], np.load(tp).ravel()[:299984])

    Bo_bank = np.column_stack(bank_o_cols)
    Bt_bank = np.column_stack(bank_t_cols)
    print(f"  NNLS bank cols={len(bank_names)}")

    locked_t = W_D1 * unitize(d1t) + W_FE * unitize(dE_t)
    resid_t = L(slot3) - locked_t
    # NNLS without intercept on unitized bank
    w_nnls, rnorm = nnls(Bt_bank, resid_t - resid_t.mean())
    r2_nnls = 1.0 - float(np.sum((resid_t - resid_t.mean() - Bt_bank @ w_nnls) ** 2)) / float(
        np.sum((resid_t - resid_t.mean()) ** 2) + 1e-18
    )
    z_nnls_t = locked_t + Bt_bank @ w_nnls
    z_nnls_o = W_D1 * unitize(d1) + W_FE * unitize(dE_o) + Bo_bank @ w_nnls
    # Also report full R² vs L(Slot3)
    r2_full = 1.0 - float(np.sum((L(slot3) - z_nnls_t) ** 2)) / float(
        np.sum((L(slot3) - L(slot3).mean()) ** 2) + 1e-18
    )
    auc_nnls = float(roc_auc_score(y, z_nnls_o))
    print(f"  NNLS residual R²={r2_nnls:.4f} full R² vs L(Slot3)={r2_full:.4f} OOF AUC={auc_nnls:.7f}")
    print(f"  NNLS nnz={int((w_nnls>1e-8).sum())}/{len(w_nnls)} rnorm={rnorm:.4f}")
    top = sorted(zip(bank_names, w_nnls), key=lambda kv: -kv[1])[:10]
    print(f"  top weights: {[(n, round(float(w),4)) for n,w in top]}")

    # Simplex LOFO on Bo9
    z_simplex, w_simp = simplex_lofo(Bo9, y, folds)
    auc_simp = float(roc_auc_score(y, z_simplex))
    print(f"  Simplex LOFO AUC={auc_simp:.7f} w={ {BN9[j]: round(float(w_simp[j]),4) for j in range(9)} }")

    baselines = {
        "z_S3": z_s3,
        "z_NNLS_Slot3_twin": z_nnls_o,
        "z_Simplex_LOFO": z_simplex,
    }
    for bn, bz in baselines.items():
        print(f"  BASE {bn} AUC={roc_auc_score(y,bz):.7f}")

    live_L = L(lv)
    live_hat = Ridge(alpha=RIDGE_ALPHA).fit(Bt9, live_L).predict(Bt9)
    live_resid = live_L - live_hat

    # ----- Step 2: candidates -----
    print("=== STEP 2: candidates ===")
    d_g26_lm_o = unitize(L_fold(z_G26_o - z_LM_o, fold_id))
    d_g26_lm_t = unitize(L(z_G26_t - z_LM_t))
    d_blend_o = unitize(L_fold(0.70 * z_G26_o + 0.30 * z_MS_o - z_LM_o, fold_id))
    d_blend_t = unitize(L(0.70 * z_G26_t + 0.30 * z_MS_t - z_LM_t))
    P_BPE_o = unitize(0.55 * z_MS_o + 0.45 * z_G26_o)
    P_QLM_o = unitize(
        0.45 * z_LM_o + 0.30 * L_fold(l7f_o, fold_id) + 0.25 * L_fold(R3, fold_id)
    )
    P_BPE_t = unitize(0.55 * z_MS_t + 0.45 * z_G26_t)
    P_QLM_t = unitize(0.45 * z_LM_t + 0.30 * L(l7f_t) + 0.25 * L(R3_t_L.mean(0)))
    d_all_o = unitize(L_fold(P_BPE_o - P_QLM_o, fold_id))
    d_all_t = unitize(L(P_BPE_t - P_QLM_t))

    U_s3 = U(z_s3)
    U_s3_t = U(L(slot3))
    uz_s3, uz_s3_t = unitize(z_s3), unitize(L(slot3))
    # also centered U for interaction-only
    Uc = unitize(U_s3 - 0.5)
    Uc_t = unitize(U_s3_t - 0.5)

    cands = {
        "R_G26_LM": (d_g26_lm_o * U_s3, d_g26_lm_t * U_s3_t),
        "R_G26_LM_z": (d_g26_lm_o * uz_s3, d_g26_lm_t * uz_s3_t),
        "R_G26_LM_Uc": (d_g26_lm_o * Uc, d_g26_lm_t * Uc_t),  # pure interaction of centered U
        "R_70G26_30MS_LM": (d_blend_o * U_s3, d_blend_t * U_s3_t),
        "R_G1e_all": (d_all_o * U_s3, d_all_t * U_s3_t),
    }

    results = {}
    oriented = {}
    for name, (vo, vt) in cands.items():
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo9, Bt9, folds)
        gates = forensic_gates(oo, ot, Bt9, live_resid)
        per_base = {}
        u_ref = None
        for bname, zbase in baselines.items():
            u, flip = orient_plus(oo, zbase, y, te0)
            if u_ref is None:
                u_ref = u
                oriented[name] = (
                    u,
                    unitize(ot if np.corrcoef(u, oo)[0, 1] > 0 else -ot),
                )
            # align to z_S3 orientation for comparability when scoring other bases
            if bname != "z_S3" and np.corrcoef(u, oriented[name][0])[0, 1] < 0:
                # keep base-specific orient for honest invariance
                pass
            row = score_at_w(u, zbase, y, folds, w=0.035)
            per_base[bname] = {**row, "flipped": flip}
            print(
                f"  {name}@{bname}: F? skip | {row['pos']}/5 mean={row['mean']:.3f} "
                f"min={row['min']:.3f} sf={row['sf25']}/25 t={row['t']:.2f} folds={row['folds']}"
            )
        print(
            f"  META {name}: F1={gates['F1']}({gates['disp']:.3f}) "
            f"F2={gates['F2']}({gates['max_basis_leak']:.4f}:{BN9[gates['leak_argmax']]}) "
            f"F3={gates['F3']}({gates['corr_live_resid']:+.4f}) ver={gates['verified']}"
        )
        results[name] = {"gates": gates, "per_base": per_base}

    # ----- 1-fold MC stress -----
    print("=== STEP 2b: 1-fold G26 MC stress ===")
    rng = np.random.default_rng(2026)
    mc = {}
    for tag, d_builder in (
        (
            "R_G26_LM",
            lambda g26z: unitize(L_fold(g26z - z_LM_o, fold_id)) * U_s3,
        ),
        (
            "R_70G26_30MS_LM",
            lambda g26z: unitize(L_fold(0.70 * g26z + 0.30 * z_MS_o - z_LM_o, fold_id)) * U_s3,
        ),
    ):
        draws = []
        for i in range(MC_DRAWS):
            # add noise on unitized G26 logit scale then rebuild
            g26_noisy = unitize(z_G26_o) + rng.normal(0, SIG_1F, size=len(y))
            # map back to similar scale as z_G26_o
            g26_noisy = g26_noisy * (z_G26_o.std() + 1e-12) + z_G26_o.mean()
            vo = d_builder(g26_noisy)
            vo_u = unitize(vo)
            oo = np.zeros(len(y))
            for tr, te in folds:
                lr = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
                lr.fit(Bo9[tr], vo_u[tr])
                oo[te] = vo_u[te] - Bo9[te] @ lr.coef_ - lr.intercept_
            oo = oo - oo.mean()
            u, _ = orient_plus(oo, z_s3, y, te0)
            # also score on NNLS twin with same u orient relative to z_S3
            row_s3 = score_at_w(u, z_s3, y, folds, 0.035)
            u_b, _ = orient_plus(oo, z_nnls_o, y, te0)
            row_b = score_at_w(u_b, z_nnls_o, y, folds, 0.035)
            u_c, _ = orient_plus(oo, z_simplex, y, te0)
            row_c = score_at_w(u_c, z_simplex, y, folds, 0.035)
            draws.append(
                {
                    "z_S3": {"pos": row_s3["pos"], "sf25": row_s3["sf25"], "mean": row_s3["mean"], "min": row_s3["min"]},
                    "z_NNLS": {"pos": row_b["pos"], "sf25": row_b["sf25"], "mean": row_b["mean"], "min": row_b["min"]},
                    "z_Simplex": {"pos": row_c["pos"], "sf25": row_c["sf25"], "mean": row_c["mean"], "min": row_c["min"]},
                }
            )
            print(
                f"  MC {tag} draw{i}: S3 {row_s3['pos']}/5 sf{row_s3['sf25']} "
                f"NNLS {row_b['pos']}/5 sf{row_b['sf25']} Simp {row_c['pos']}/5 sf{row_c['sf25']}"
            )
        mc[tag] = {
            "draws": draws,
            "frac_S3_5of5_sf20": float(
                np.mean([(d["z_S3"]["pos"] >= 5 and d["z_S3"]["sf25"] >= 20) for d in draws])
            ),
            "frac_all_bases_5of5_sf20": float(
                np.mean(
                    [
                        all(
                            d[b]["pos"] >= 5 and d[b]["sf25"] >= 20
                            for b in ("z_S3", "z_NNLS", "z_Simplex")
                        )
                        for d in draws
                    ]
                )
            ),
        }
        print(
            f"  MC {tag}: frac S3 pass={mc[tag]['frac_S3_5of5_sf20']:.2f} "
            f"all-bases={mc[tag]['frac_all_bases_5of5_sf20']:.2f}"
        )

    # ----- Step 3: stage -----
    print("=== STEP 3: staging decision ===")
    winners = []
    for name, r in results.items():
        g = r["gates"]
        if not g["verified"]:
            continue
        ok_bases = all(
            r["per_base"][b]["pos"] >= 5
            and r["per_base"][b]["sf25"] >= STAGE_MIN_SF
            and r["per_base"][b]["t"] >= STAGE_MIN_T
            and r["per_base"][b]["min"] > 0
            for b in baselines
        )
        mc_ok = True
        if name in mc:
            mc_ok = mc[name]["frac_all_bases_5of5_sf20"] >= 0.75
        elif name == "R_G26_LM_z":
            mc_ok = mc.get("R_G26_LM", {}).get("frac_all_bases_5of5_sf20", 0) >= 0.75
        if ok_bases and mc_ok:
            winners.append(name)
        print(f"  {name}: forensic={g['verified']} bases_ok={ok_bases} mc_ok={mc_ok}")

    staged = []
    for name in winners:
        u_o, u_t = oriented[name]
        wstar = 0.035
        path = SUB / f"sub_v17_6_{name}_w0p035_NOSUBMIT.csv"
        cand = qlock(L(slot3) + wstar * u_t, slot3)
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
        staged.append({"file": path.name, "signal": name, "w": wstar})
        print("STAGED", path.name)

    if not winners:
        print("NO STAGE — invariance or MC failed")

    payload = scrub(
        {
            "NO_SUBMIT": True,
            "pods": 0,
            "baselines": {
                "z_S3_auc": float(roc_auc_score(y, z_s3)),
                "z_NNLS_auc": auc_nnls,
                "z_NNLS_r2_full": r2_full,
                "z_NNLS_r2_resid": r2_nnls,
                "z_NNLS_top_w": [(n, float(w)) for n, w in top],
                "z_NNLS_all_w": {n: float(w) for n, w in zip(bank_names, w_nnls) if w > 1e-8},
                "z_Simplex_auc": auc_simp,
                "z_Simplex_w": {BN9[j]: float(w_simp[j]) for j in range(9)},
            },
            "results": results,
            "mc": mc,
            "winners": winners,
            "staged": staged,
            "stage_rule": "F1&F2&F3 AND 5/5&sf>=20&t>=2.5&min>0 on ALL 3 baselines AND MC all-bases >=75%",
        }
    )
    OUT.joinpath("v17_6.json").write_text(json.dumps(payload, indent=2))
    print("WROTE", OUT / "v17_6.json")
    print("DONE")


if __name__ == "__main__":
    main()
