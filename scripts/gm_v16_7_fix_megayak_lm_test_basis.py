#!/usr/bin/env python3
"""GM V16.7 — Fix LM_test==L7F_test duplicate column; re-audit forensic & composites.

STRICT NO_SUBMIT=True. 0 GPU pods.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
TMP15 = Path("/tmp/s6e11_v15")
TMP16 = Path("/tmp/s6e11_v16")
OUT = TMP16 / "out"
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"
ALEX = REPO / "external_bank/alex_oof"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
BOLD = SUB / "sub_v13_slot1_PURE_goodpjw_bold095386_LIVE.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
G26_OOF = TMP16 / "g26/gpt2_s2026_oof_5fold.npy"
G26_T0 = TMP16 / "g26/gpt2_s2026_test_f0.npy"
NO_SUBMIT = True

DISP_LO, DISP_HI = 0.93, 1.07
MAX_BETA = 0.08
LIVE_CORR_MAX = 0.08


def L(v):
    v = np.asarray(v, float).ravel()
    r = (rankdata(v, method="average") - 0.5) / len(v)
    r = np.clip(r, 1e-12, 1 - 1e-12)
    return np.log(r / (1 - r))


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def qlock(raw, ref):
    raw = np.asarray(raw, float)
    ref = np.asarray(ref, float)
    out = np.empty_like(ref)
    out[np.argsort(raw)] = np.sort(ref)
    return out


def corr_mat(X):
    X = np.asarray(X, float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return np.clip(X.T @ X / X.shape[0], -1, 1)


def lofo_orth_raw(raw, B, folds):
    resid = np.zeros(len(raw))
    for tr, te in folds:
        lr = LinearRegression().fit(B[tr], raw[tr])
        resid[te] = raw[te] - lr.predict(B[te])
    return resid


def forensic(name, raw_o, raw_t, Bo, Bt, bn, folds, live_L, live_hat, bold_L, z_ath_t):
    ro = lofo_orth_raw(raw_o, Bo, folds)
    rt = raw_t - LinearRegression().fit(Bt, raw_t).predict(Bt)
    disp = float((rt.std() + 1e-12) / (ro.std() + 1e-12))
    bo = LinearRegression().fit(Bo, raw_o).coef_
    bt = LinearRegression().fit(Bt, raw_t).coef_
    maxbd = float(np.max(np.abs(bo - bt)))
    orth_t = unitize(rt)
    c_live = float(np.corrcoef(orth_t, live_L - live_hat)[0, 1])
    c_bold = float(np.corrcoef(orth_t, bold_L - z_ath_t)[0, 1])
    pass_disp = DISP_LO <= disp <= DISP_HI
    pass_sym = maxbd < MAX_BETA
    pass_dc = abs(c_live) < LIVE_CORR_MAX and abs(c_bold) < 0.15
    verified = pass_disp and pass_sym and pass_dc
    print(
        f"  {name}: disp={disp:.4f} max|Δβ|={maxbd:.4f} "
        f"corr_LIVE={c_live:.4f} corr_bold={c_bold:.4f} pass={verified}"
    )
    return {
        "name": name,
        "disp": disp,
        "max_beta_diff": maxbd,
        "corr_live_resid": c_live,
        "corr_bold": c_bold,
        "pass_disp": pass_disp,
        "pass_sym": pass_sym,
        "pass_dc": pass_dc,
        "verified": verified,
        "beta_o": dict(zip(bn, map(float, bo))),
        "beta_t": dict(zip(bn, map(float, bt))),
        "orth_o": unitize(ro),
        "orth_t": unitize(rt),
    }


def screen(name, u, zbase, y, folds, te0):
    u = unitize(u)
    if roc_auc_score(y[te0], zbase[te0] + 0.01 * u[te0]) < roc_auc_score(
        y[te0], zbase[te0] - 0.01 * u[te0]
    ):
        u = -u
    base = [roc_auc_score(y[te], zbase[te]) for _, te in folds]
    best = None
    fold_d = None
    for w in (0.006, 0.009, 0.012, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04):
        d = [
            (roc_auc_score(y[te], zbase[te] + w * u[te]) - base[i]) * 1e5
            for i, (_, te) in enumerate(folds)
        ]
        row = {
            "w": w,
            "pos": int(sum(x > 0 for x in d)),
            "mean": float(np.mean(d)),
            "min": float(np.min(d)),
            "folds": [round(x, 3) for x in d],
        }
        if best is None or (row["pos"], row["mean"]) > (best["pos"], best["mean"]):
            best = row
            fold_d = d
    deltas = []
    for _, te in folds:
        yf, zb, uf = y[te], zbase[te], u[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(yf)), yf
        ):
            b = roc_auc_score(yf[ste], zb[ste])
            deltas.append((roc_auc_score(yf[ste], zb[ste] + 0.01 * uf[ste]) - b) * 1e5)
    d25 = np.asarray(deltas, float)
    mean = float(d25.mean())
    se = float(d25.std(ddof=1) / np.sqrt(len(d25)))
    t = mean / (se + 1e-18)
    out = {
        "name": name,
        "main": best,
        "sf25": int((d25 > 0).sum()),
        "t": float(t),
        "mean_sf": mean,
        "fold_d_e5": [float(x) for x in fold_d],
    }
    print(
        f"  SCREEN {name}: pos={best['pos']}/5 @{best['w']} mean={best['mean']:.3f} "
        f"min={best['min']:.3f} sf={out['sf25']}/25 t={t:.2f} folds={best['folds']}"
    )
    return out, u


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    assert NO_SUBMIT

    cache = np.load(OUT / "z_ath_cache.npz")
    z_ath, d1, y, fold_id = cache["z_ath"], cache["d1"], cache["y"], cache["fold_id"]
    d1t = cache["d1t"] if "d1t" in cache.files else np.load(TMP15 / "banks/d1_v4_test.npy").ravel()

    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    te0 = folds[0][1]

    Z = np.load(Q17, allow_pickle=True)
    names = [str(s) for s in Z["stream_names"]]
    XO = {n: Z[f"oof_{n}"].ravel() for n in names}
    XT = {n: Z[f"test_{n}"].ravel() for n in names}

    # --- Step 1: verify megayak LM test is distinct ---
    lm_o = XO["megayak_5f_lm"]
    lm_t = XT["megayak_5f_lm"]
    l7_o = XO["lm7_5fold"]
    l7_t = XT["lm7_5fold"]
    banks_l7t = np.load(TMP15 / "banks/lm7_test_5fold.npy").ravel()
    lm_oof_file = np.load(TMP15 / "out/lm_oof.npy").ravel()

    step1 = {
        "Q17_has_test_megayak_5f_lm": True,
        "spearman_LM_L7F_oof": float(spearmanr(lm_o, l7_o).statistic),
        "spearman_LM_L7F_test": float(spearmanr(lm_t, l7_t).statistic),
        "identical_LM_L7F_test": bool(np.allclose(lm_t, l7_t)),
        "spearman_Q17_l7_vs_banks": float(spearmanr(l7_t, banks_l7t).statistic),
        "spearman_Q17_lm_vs_banks_l7": float(spearmanr(lm_t, banks_l7t).statistic),
        "spearman_Q17_lm_oof_vs_lm_oof_npy": float(spearmanr(lm_o, lm_oof_file).statistic),
        "auc_LM_oof": float(roc_auc_score(y, lm_o)),
        "auc_L7F_oof": float(roc_auc_score(y, l7_o)),
        "bug_was": "V16.4/V16.6 used lm7_test_5fold for BOTH LM_test and L7F_test",
        "fix": "load XT['megayak_5f_lm'] for LM_test",
    }
    print("=== STEP 1: megayak LM test ===")
    for k, v in step1.items():
        print(f"  {k}: {v}")

    # LIVE / Slot3
    live_df = pd.read_csv(LIVE)
    ids = live_df["id"].to_numpy()
    lv = live_df["is_fraud"].to_numpy(float)
    z_ath_t = L(lv)
    bold = pd.read_csv(BOLD).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    slot3 = pd.read_csv(SLOT3).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    # Family E
    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
    l7f = np.load(TMP15 / "out/l7f_oof.npy").ravel() if (TMP15 / "out/l7f_oof.npy").exists() else l7_o
    Ls, L7 = L(R3), L(l7f)
    U = expit((z_ath - np.median(z_ath)) / (np.std(z_ath) + 1e-12))
    w_tail = 0.5 + 0.5 * U
    raw_e = (Ls - L7) * w_tail
    B2 = np.column_stack([unitize(z_ath), unitize(d1)])
    dE = unitize(raw_e - LinearRegression().fit(B2, raw_e).predict(B2))
    if roc_auc_score(y[te0], z_ath[te0] + 0.01 * dE[te0]) < roc_auc_score(
        y[te0], z_ath[te0] - 0.01 * dE[te0]
    ):
        dE = -dE
    z_s1 = z_ath + 0.01045 * dE
    z_s3 = z_s1 + 0.0055 * unitize(d1)
    print(f"AUC z_S1={roc_auc_score(y,z_s1):.7f} z_S3={roc_auc_score(y,z_s3):.7f}")

    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    R3t = R3_te.mean(0) if R3_te.shape[0] == 5 else R3_te.mean(1)
    w_tail_t = 0.5 + 0.5 * expit((z_ath_t - np.median(z_ath_t)) / (np.std(z_ath_t) + 1e-12))
    dE_t = unitize((L(R3t) - L(l7_t)) * w_tail_t)

    # Build FIXED 10-col basis
    bn = ["z_ATH", "d1_v4", "Family_E", "MS", "G26", "LM", "L7F", "GPD", "GB", "MLP"]
    Bo_cols = [
        unitize(z_ath),
        unitize(d1),
        dE,
        unitize(L(np.load(REPO / "preds/P_super_v4_oof.npy").ravel())),
        unitize(L(np.load(G26_OOF).ravel()[: len(y)]))
        if G26_OOF.exists()
        else unitize(XO["gpt2_seed2026"]),
        unitize(L(lm_o)),  # TRUE megayak
        unitize(L(l7_o)),  # L7F
        unitize(XO["daoviet_gpdvae"]),
        unitize(XO["gb_bag5"])
        if "gb_bag5" in XO
        else unitize(np.load(REPO / "preds/gb_gpt2_bag5_oof.npy").ravel()),
        unitize(XO["yekenot_realmlp"]),
    ]
    # G26 test: prefer f0 artifact (same as LIVE weight path); Q17 gpt2_seed2026 is f0-only too
    g26_t = (
        L(np.load(G26_T0).ravel()[:299984])
        if G26_T0.exists()
        else XT["gpt2_seed2026"]
    )
    Bt_cols = [
        unitize(z_ath_t),
        unitize(d1t),
        dE_t,
        unitize(L(np.load(REPO / "preds/P_super_v4_test.npy").ravel())),
        unitize(g26_t),
        unitize(L(lm_t)),  # FIXED: true megayak_5f_lm test
        unitize(L(l7_t)),  # L7F test (distinct)
        unitize(XT["daoviet_gpdvae"]),
        unitize(XT["gb_bag5"])
        if "gb_bag5" in XT
        else unitize(np.load(REPO / "preds/gb_gpt2_bag5_full.npy").ravel()),
        unitize(XT["yekenot_realmlp"]),
    ]
    Bo = np.column_stack(Bo_cols)
    Bt = np.column_stack(Bt_cols)

    # Broken basis (V16.6) for comparison
    Bt_broken = Bt.copy()
    Bt_broken[:, bn.index("LM")] = Bt[:, bn.index("L7F")]

    Co = corr_mat(Bo)
    Ct = corr_mat(Bt)
    Ct_broken = corr_mat(Bt_broken)
    dC = np.abs(Co - Ct)
    dC_broken = np.abs(Co - Ct_broken)
    i, j = np.unravel_index(np.argmax(dC), dC.shape)
    ib, jb = np.unravel_index(np.argmax(dC_broken), dC_broken.shape)
    corr_audit = {
        "corr_oof_LM_L7F": float(Co[bn.index("LM"), bn.index("L7F")]),
        "corr_test_LM_L7F_fixed": float(Ct[bn.index("LM"), bn.index("L7F")]),
        "corr_test_LM_L7F_broken": float(Ct_broken[bn.index("LM"), bn.index("L7F")]),
        "max_abs_corr_diff_fixed": float(dC.max()),
        "max_pair_fixed": [bn[i], bn[j]],
        "max_abs_corr_diff_broken": float(dC_broken.max()),
        "max_pair_broken": [bn[ib], bn[jb]],
        "corr_oof": {bn[a]: {bn[b]: float(Co[a, b]) for b in range(10)} for a in range(10)},
        "corr_test_fixed": {
            bn[a]: {bn[b]: float(Ct[a, b]) for b in range(10)} for a in range(10)
        },
    }
    print("=== basis corr audit ===")
    print(
        f"  corr(LM,L7F) oof={corr_audit['corr_oof_LM_L7F']:.4f} "
        f"test_fixed={corr_audit['corr_test_LM_L7F_fixed']:.4f} "
        f"test_broken={corr_audit['corr_test_LM_L7F_broken']:.4f}"
    )
    print(
        f"  max|Δcorr| fixed={corr_audit['max_abs_corr_diff_fixed']:.4f} "
        f"{corr_audit['max_pair_fixed']}  broken={corr_audit['max_abs_corr_diff_broken']:.4f} "
        f"{corr_audit['max_pair_broken']}"
    )
    # rank / condition
    print(
        f"  cond(Bo)={np.linalg.cond(Bo):.1f} cond(Bt_fixed)={np.linalg.cond(Bt):.1f} "
        f"cond(Bt_broken)={np.linalg.cond(Bt_broken):.1f}"
    )

    # --- Step 2: forensic on candidates ---
    print("=== STEP 2: forensic (fixed basis) ===")
    live_L = L(lv)
    live_hat = LinearRegression().fit(Bt, live_L).predict(Bt)
    bold_L = L(bold)

    # GPU signals
    oof_raw = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy").astype(np.float64)
    llr = oof_raw[:, :, 1] - oof_raw[:, :, 0]
    spread_o = L(np.quantile(llr, 0.8, axis=1) - np.quantile(llr, 0.2, axis=1))
    R2 = np.load(TMP15 / "out/R2_oof.npy").ravel()
    curv_o = unitize(L(R2) - L(R3))

    traw = np.load(TMP15 / "out/l7_s2026_test_5fold_raw.npy")
    tmean = traw.mean(0) if traw.ndim == 4 and traw.shape[0] == 5 else traw
    tllr = tmean[:, :, 1] - tmean[:, :, 0]
    spread_t = L(np.quantile(tllr, 0.8, axis=1) - np.quantile(tllr, 0.2, axis=1))
    R2_te = np.load(TMP15 / "out/R2_test_folds.npy")
    R2t = R2_te.mean(0) if R2_te.shape[0] == 5 else R2_te.mean(1)
    curv_t = unitize(L(R2t) - L(R3t))

    candidates = {
        "q80_q20": (spread_o, spread_t),
        "R2_R3_curv": (curv_o, curv_t),
        "sadam_v4_cat": (XO["sadam_v4_cat"], XT["sadam_v4_cat"]),
        "sameer_glm10f": (XO["sameer_glm10f"], XT["sameer_glm10f"]),
        "daoviet_gpdvae": (XO["daoviet_gpdvae"], XT["daoviet_gpdvae"]),
        "alex_372ea048": (
            np.load(ALEX / "oof_15d0d127372ea048.npy").ravel(),
            np.load(ALEX / "test_15d0d127372ea048.npy").ravel(),
        ),
        "alex_6c55b429": (
            np.load(ALEX / "oof_21b4d1316c55b429.npy").ravel(),
            np.load(ALEX / "test_21b4d1316c55b429.npy").ravel(),
        ),
    }

    forensics = {}
    for name, (ro, rt) in candidates.items():
        forensics[name] = forensic(
            name, L(ro) if name.startswith("alex") or name in ("sadam_v4_cat", "sameer_glm10f", "daoviet_gpdvae") else ro,
            L(rt) if name.startswith("alex") or name in ("sadam_v4_cat", "sameer_glm10f", "daoviet_gpdvae") else rt,
            Bo, Bt, bn, folds, live_L, live_hat, bold_L, z_ath_t,
        )
    # GPU spreads already in L/unitize space — re-do without double L for q80/curv
    # actually spread_o is already L(...); curv is unitize. For alex/sadam we used L.
    # q80/curv: raw already prepared — forensic used as-is above. Good.

    # Also forensic with BROKEN basis for sadam/q80 to show delta
    print("=== forensic BROKEN basis (LM=L7F test) for comparison ===")
    for name in ("sadam_v4_cat", "q80_q20", "R2_R3_curv"):
        ro, rt = candidates[name]
        raw_o = L(ro) if name == "sadam_v4_cat" else ro
        raw_t = L(rt) if name == "sadam_v4_cat" else rt
        forensic(name + "_BROKEN", raw_o, raw_t, Bo, Bt_broken, bn, folds, live_L,
                 LinearRegression().fit(Bt_broken, live_L).predict(Bt_broken), bold_L, z_ath_t)

    # megayak inject probe: is LIVE missing true LM?
    print("=== megayak inject vs LIVE/Slot3 ===")
    delta_lm = unitize(L(lm_t) - L(l7_t))
    # corr of true LM with LIVE / Slot3 vs L7F
    c_lm_live = float(spearmanr(lm_t, lv).statistic)
    c_l7_live = float(spearmanr(l7_t, lv).statistic)
    c_lm_s3 = float(spearmanr(lm_t, slot3).statistic)
    # orth of LM vs L7F on test vs basis without LM
    # OOF screen: inject unitize(L(LM)-L(L7F)) on z_S3
    d_lm_oof = unitize(L(lm_o) - L(l7_o))
    # also orth of LM against full basis without LM column
    Bo_no_lm = np.delete(Bo, bn.index("LM"), axis=1)
    resid_lm = lofo_orth_raw(L(lm_o), Bo_no_lm, folds)
    scr_delta, u_delta = screen("L(LM)-L(L7F)_raw", d_lm_oof, z_s3, y, folds, te0)
    scr_orth_lm, u_orth_lm = screen("orth(LM)_excl_LM_col", resid_lm, z_s3, y, folds, te0)
    # Does LIVE already contain LM? corr(orth_LM_test, LIVE resid)
    Bt_no_lm = np.delete(Bt, bn.index("LM"), axis=1)
    resid_lm_t = L(lm_t) - LinearRegression().fit(Bt_no_lm, L(lm_t)).predict(Bt_no_lm)
    live_hat_nolm = LinearRegression().fit(Bt_no_lm, live_L).predict(Bt_no_lm)
    c_lm_in_live = float(np.corrcoef(unitize(resid_lm_t), live_L - live_hat_nolm)[0, 1])
    megayak_probe = {
        "spearman_LM_vs_LIVE": c_lm_live,
        "spearman_L7F_vs_LIVE": c_l7_live,
        "spearman_LM_vs_Slot3": c_lm_s3,
        "corr_orthLM_vs_LIVE_resid_excl_LM": c_lm_in_live,
        "screen_L_diff": {k: v for k, v in scr_delta.items()},
        "screen_orth_LM": {k: v for k, v in scr_orth_lm.items()},
        "solo_auc_gap_bps": (step1["auc_LM_oof"] - step1["auc_L7F_oof"]) * 1e4,
    }
    print(f"  ρ(LM,LIVE)={c_lm_live:.4f} ρ(L7F,LIVE)={c_l7_live:.4f} corr(orthLM,LIVE_resid)={c_lm_in_live:.4f}")
    print(f"  solo AUC gap LM-L7F = {megayak_probe['solo_auc_gap_bps']:.1f} bps")

    # --- Step 3: screen verified + composites ---
    print("=== STEP 3: screens on z_S3 ===")
    screens = {}
    oriented = {}
    for name, f in forensics.items():
        scr, u = screen(f"orth({name})", f["orth_o"], z_s3, y, folds, te0)
        screens[name] = {**scr, "disp": f["disp"], "max_beta_diff": f["max_beta_diff"], "verified": f["verified"]}
        oriented[name] = (u, f["orth_t"] if np.corrcoef(u, f["orth_o"])[0, 1] > 0 else -f["orth_t"])

    verified = [n for n, f in forensics.items() if f["verified"] and screens[n]["main"]["pos"] >= 5]
    verified_soft = [n for n, f in forensics.items() if f["verified"]]
    print(f"verified={verified_soft}  verified_and_5of5={verified}")

    # composites among verified (or best forensic-cleaner ones)
    pool = verified_soft if verified_soft else []
    # also allow near-pass: disp OK and max_beta < 0.15 for exploratory
    near = [
        n
        for n, f in forensics.items()
        if f["pass_disp"] and f["pass_dc"] and f["max_beta_diff"] < 0.15
    ]
    print(f"near-pass (disp+dc, Δβ<0.15)={near}")

    best_combo = None
    best_u = None
    combo_pool = verified if verified else near
    if len(combo_pool) >= 1:
        mats = {n: oriented[n][0] for n in combo_pool}
        keys = list(mats.keys())
        from itertools import product

        grid = [0.0, 0.3, 0.5, 0.7, 1.0]
        for ws in product(grid, repeat=len(keys)):
            if sum(ws) < 1e-9:
                continue
            u = unitize(sum(w * mats[k] for w, k in zip(ws, keys)))
            if roc_auc_score(y[te0], z_s3[te0] + 0.01 * u[te0]) < roc_auc_score(
                y[te0], z_s3[te0] - 0.01 * u[te0]
            ):
                u = -u
            d = [
                (roc_auc_score(y[te], z_s3[te] + 0.015 * u[te]) - roc_auc_score(y[te], z_s3[te]))
                * 1e5
                for _, te in folds
            ]
            pos = int(sum(x > 0 for x in d))
            mean = float(np.mean(d))
            deltas = []
            for _, te in folds:
                yf, zb, uf = y[te], z_s3[te], u[te]
                for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(
                    np.zeros(len(yf)), yf
                ):
                    b = roc_auc_score(yf[ste], zb[ste])
                    deltas.append((roc_auc_score(yf[ste], zb[ste] + 0.01 * uf[ste]) - b) * 1e5)
            d25 = np.asarray(deltas, float)
            t = float(d25.mean() / (d25.std(ddof=1) / np.sqrt(25) + 1e-18))
            sf = int((d25 > 0).sum())
            row = {"w": dict(zip(keys, ws)), "pos": pos, "mean": mean, "sf": sf, "t": t}
            if best_combo is None or (pos, sf, t, mean) > (
                best_combo["pos"],
                best_combo["sf"],
                best_combo["t"],
                best_combo["mean"],
            ):
                best_combo = row
                best_u = u

    staged = []
    if best_u is not None:
        s_best, u_best = screen("best_composite", best_u, z_s3, y, folds, te0)
        screens["best_composite"] = s_best
        # stage if 5/5
        if s_best["main"]["pos"] >= 5:
            # rebuild test direction
            keys = list(best_combo["w"].keys())
            ut = unitize(
                sum(best_combo["w"][k] * oriented[k][1] for k in keys if best_combo["w"][k] > 0)
            )
            if np.corrcoef(u_best, unitize(sum(best_combo["w"][k] * oriented[k][0] for k in keys)))[
                0, 1
            ] < 0:
                ut = -ut
            wstar = s_best["main"]["w"]
            cand = qlock(L(slot3) + wstar * ut, slot3)
            path = SUB / "sub_v16_7_composite_fixed_basis_NOSUBMIT.csv"
            pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
            mrho = (1 - spearmanr(slot3, cand).statistic) * 1e4
            staged.append(
                {
                    "file": path.name,
                    "kind": "composite",
                    "w": wstar,
                    "weights": best_combo["w"],
                    "1mrho_e4": float(mrho),
                    "pos": s_best["main"]["pos"],
                    "sf25": s_best["sf25"],
                    "t": s_best["t"],
                }
            )
            print(f"STAGED {path.name} 1-ρ={mrho:.3f}e-4")

    # stage best single verified 5/5
    best_single = None
    for name in verified_soft:
        s = screens[name]
        if s["main"]["pos"] < 5:
            continue
        if best_single is None or (s["sf25"], s["t"], s["main"]["mean"]) > (
            best_single["sf25"],
            best_single["t"],
            best_single["main"]["mean"],
        ):
            best_single = {**s, "signal": name}
    if best_single is not None:
        name = best_single["signal"]
        u_o, u_t = oriented[name]
        wstar = best_single["main"]["w"]
        cand = qlock(L(slot3) + wstar * u_t, slot3)
        path = SUB / f"sub_v16_7_{name}_NOSUBMIT.csv"
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
        mrho = (1 - spearmanr(slot3, cand).statistic) * 1e4
        staged.append(
            {
                "file": path.name,
                "kind": "single",
                "signal": name,
                "w": wstar,
                "1mrho_e4": float(mrho),
                "pos": best_single["main"]["pos"],
                "sf25": best_single["sf25"],
                "t": best_single["t"],
            }
        )
        print(f"STAGED {path.name}")

    # if LM delta looks strong and not already in LIVE, stage NOSUBMIT probe
    if scr_delta["main"]["pos"] >= 5 and abs(c_lm_in_live) < 0.5:
        cand = qlock(L(slot3) + scr_delta["main"]["w"] * delta_lm, slot3)
        path = SUB / "sub_v16_7_megayak_minus_l7f_NOSUBMIT.csv"
        pd.DataFrame({"id": ids, "is_fraud": cand}).to_csv(path, index=False)
        staged.append(
            {
                "file": path.name,
                "kind": "megayak_probe",
                "w": scr_delta["main"]["w"],
                "pos": scr_delta["main"]["pos"],
                "sf25": scr_delta["sf25"],
                "t": scr_delta["t"],
                "note": "L(megayak_test)-L(lm7_test); not forensic-orth",
            }
        )
        print(f"STAGED megayak probe {path.name}")

    payload = {
        "NO_SUBMIT": True,
        "step1": step1,
        "corr_audit": {
            k: v
            for k, v in corr_audit.items()
            if k not in ("corr_oof", "corr_test_fixed")
        },
        "corr_oof_matrix": corr_audit["corr_oof"],
        "corr_test_matrix": corr_audit["corr_test_fixed"],
        "forensic": {
            n: {k: v for k, v in f.items() if k not in ("orth_o", "orth_t")}
            for n, f in forensics.items()
        },
        "megayak_probe": megayak_probe,
        "screens": {
            n: {k: v for k, v in s.items()}
            for n, s in screens.items()
        },
        "best_combo": best_combo,
        "staged": staged,
        "verified": verified_soft,
        "verified_5of5": verified,
    }
    OUT.joinpath("v16_7.json").write_text(json.dumps(payload, indent=2, default=float))
    print("WROTE", OUT / "v16_7.json")
    print("DONE")
    return payload


if __name__ == "__main__":
    main()
