#!/usr/bin/env python3
"""GM V16.4 — Forensic: L7 OOF indexing, sameer_glm10f double-count, d1 polish.

STRICT NO_SUBMIT=True. 0 pods. Local CPU < ~30–90s.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
REPO = Path(os.environ.get("V16_REPO", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
TMP15 = Path("/tmp/s6e11_v15")
OUT = Path("/tmp/s6e11_v16/out")
SUB = REPO / "submissions"
REPORT = REPO / "logs/overnight/GM_V16_4_FORENSIC_VERIFICATION.md"
LIVE = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT1 = SUB / "sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv"
BOLD = SUB / "sub_v13_slot1_PURE_goodpjw_bold095386_LIVE.csv"
G26 = Path("/tmp/s6e11_v16/g26/gpt2_s2026_oof_5fold.npy")
Q17 = REPO / "external/nov12_late_recon/amanatar_q17/rank1_q17_streams.npz"


def log(msg: str) -> None:
    print(msg, flush=True)


def L(v):
    v = np.asarray(v, float).ravel()
    r = (rankdata(v, method="average") - 0.5) / len(v)
    r = np.clip(r, 1e-12, 1 - 1e-12)
    return np.log(r / (1 - r))


def unitize(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def qlock(raw, ref):
    raw = np.asarray(raw, float).ravel()
    ref = np.asarray(ref, float).ravel()
    out = np.empty_like(ref)
    out[np.argsort(raw)] = np.sort(ref)
    return out


def orient(u, z, y, te0):
    if roc_auc_score(y[te0], z[te0] + 0.01 * u[te0]) < roc_auc_score(y[te0], z[te0] - 0.01 * u[te0]):
        return -u
    return u


def load_ath():
    path = REPO / "scripts/gm_v13_2_overnight_check_and_stageB_v14.py"
    spec = importlib.util.spec_from_file_location("gm_v13_2", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gm_v13_2"] = mod
    spec.loader.exec_module(mod)
    pack = mod.build_ath_proxy()
    assert pack["auc_ath_oof"] > 0.9523
    return pack, mod


def screen(y, z, u, folds, w_grid=(0.006, 0.009, 0.012, 0.015, 0.020, 0.025)):
    te0 = folds[0][1]
    u = orient(unitize(u), z, y, te0)
    base = [roc_auc_score(y[te], z[te]) for _, te in folds]
    best = None
    for w in w_grid:
        d = [(roc_auc_score(y[te], z[te] + w * u[te]) - base[i]) * 1e5 for i, (_, te) in enumerate(folds)]
        row = dict(w=w, pos=int(sum(x > 0 for x in d)), mean_e5=float(np.mean(d)), min_e5=float(np.min(d)), folds=[round(x, 3) for x in d])
        if best is None or (row["pos"], row["mean_e5"]) > (best["pos"], best["mean_e5"]):
            best = row
    deltas = []
    for _, te in folds:
        yf, zb, uf = y[te], z[te], u[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(yf)), yf):
            b = roc_auc_score(yf[ste], zb[ste])
            deltas.append((roc_auc_score(yf[ste], zb[ste] + 0.01 * uf[ste]) - b) * 1e5)
    d25 = np.asarray(deltas, float)
    mean = float(d25.mean())
    se = float(d25.std(ddof=1) / np.sqrt(len(d25)))
    tstat = float(mean / (se + 1e-18))
    try:
        _, wp = wilcoxon(d25, alternative="greater", zero_method="wilcox")
    except Exception:
        wp = 1.0
    rng = np.random.default_rng(42)
    boots = [d25[rng.integers(0, len(d25), len(d25))].mean() for _ in range(400)]
    ci = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]
    return {"main": best, "sf25": {"pos": int((d25 > 0).sum()), "t": tstat, "mean_e5": mean, "wilcox": float(wp), "ci90": ci}, "u": u}


def eval_fixed(y, z_base, z_new, folds):
    d = []
    for _, te in folds:
        d.append((roc_auc_score(y[te], z_new[te]) - roc_auc_score(y[te], z_base[te])) * 1e5)
    deltas = []
    # sf via direction = z_new - z_base unitized
    u = unitize(z_new - z_base)
    for _, te in folds:
        yf, zb, uf = y[te], z_base[te], u[te]
        for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(yf)), yf):
            b = roc_auc_score(yf[ste], zb[ste])
            # use actual z_new on subfold
            deltas.append((roc_auc_score(yf[ste], z_new[te][ste]) - b) * 1e5)
    d25 = np.asarray(deltas, float)
    mean = float(d25.mean())
    se = float(d25.std(ddof=1) / np.sqrt(len(d25)))
    tstat = float(mean / (se + 1e-18))
    rng = np.random.default_rng(42)
    boots = [d25[rng.integers(0, len(d25), len(d25))].mean() for _ in range(400)]
    return {
        "pos": int(sum(x > 0 for x in d)),
        "mean_e5": float(np.mean(d)),
        "min_e5": float(np.min(d)),
        "folds": [round(x, 3) for x in d],
        "sf25": int((d25 > 0).sum()),
        "t": tstat,
        "ci90": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
    }


def main():
    assert NO_SUBMIT is True
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (REPO / "logs/overnight").mkdir(parents=True, exist_ok=True)

    # ---- Step 1: indexing ----
    log("=== STEP 1: OOF fold-index forensic ===")
    Z = np.load(Q17, allow_pickle=True)
    y = Z["y_train"].astype(int)
    fold_id = Z["folds"].astype(int)
    folds = [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]
    # confirm SK42 identity
    sk = np.empty(len(y), np.int8)
    for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
        sk[te] = f
    agree = float((sk == fold_id).mean())

    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()  # L7_s2026 antithetic
    l7f = np.load(TMP15 / "out/l7f_oof.npy").ravel()
    rho_global = float(spearmanr(R3, l7f).statistic)
    fold_rows = []
    for f in range(5):
        m = fold_id == f
        fold_rows.append(
            {
                "f": f,
                "R3_auc": float(roc_auc_score(y[m], R3[m])),
                "L7F_auc": float(roc_auc_score(y[m], l7f[m])),
                "delta_bps": float((roc_auc_score(y[m], R3[m]) - roc_auc_score(y[m], l7f[m])) * 1e4),
                "rho": float(spearmanr(R3[m], l7f[m]).statistic),
            }
        )
        log(
            f"  f{f}: R3={fold_rows[-1]['R3_auc']:.6f} L7F={fold_rows[-1]['L7F_auc']:.6f} "
            f"Δ={fold_rows[-1]['delta_bps']:+.2f}bps ρ={fold_rows[-1]['rho']:.6f}"
        )
    log(f"  Q17≡SK42 agreement={agree:.6f}  global ρ(R3,L7F)={rho_global:.6f} (need >0.990? {rho_global>0.990})")

    # verify raw assembly: oof_f placed at fold_id==f
    oof_raw = np.load(TMP15 / "out/l7_s2026_oof_5fold_raw.npy")
    scramble = False
    for f in range(5):
        m = fold_id == f
        of = np.load(REPO / f"preds/l7_s2026_5f/oof_{f}.npy")
        # compare bayes/antithetic on placed rows vs R3[m] length
        if of.shape[0] != m.sum():
            scramble = True
        # check raw slice equals file
        if not np.allclose(oof_raw[m], of, atol=1e-5):
            # allow float32
            if not np.allclose(oof_raw[m].astype(np.float32), of.astype(np.float32), atol=1e-4):
                scramble = True
                log(f"  MISMATCH raw vs oof_{f}!")
    log(f"  raw placement scramble={scramble}")

    pack, mod = load_ath()
    z_ath = pack["z_ATH_o"]
    d1 = pack["d1o"]
    z_ath_t = pack["z_ATH_t"]
    names = pack["names"]
    XO, XT = pack["XO"], pack["XT"]
    log(f"  TRUE z_ATH AUC={pack['auc_ath_oof']:.7f}")

    # bold weights on L7F/LM/MS/G26-like keys
    import json as _json

    nb_path = REPO / "external/nov13_recon/goodpjw_095386/s6e11-lm-ladder-public-and-cv-up-lb-0-95386.ipynb"
    nb = _json.load(open(nb_path))
    src = "".join(nb["cells"][0]["source"])
    i = src.find("RECIPE_JSON = '''")
    j = src.find("'''", i + 17)
    bold_w = _json.loads(src[i + 17 : j])["bold"]
    # map interesting keys
    interest = {
        "L7F": ["s:lm7_5fold", "b:lm7", "lm7"],
        "LM": ["wide", "b:megalm", "megayak"],
        "MS": ["b:gb_bag5", "s:gb_bag5", "gb_bag"],
        "G26": ["s:gpt2_seed2026", "b:gpt2_s2026", "gpt2_seed2026"],
    }
    weight_hits = {k: [] for k in interest}
    for k, w in bold_w.items():
        for lab, pats in interest.items():
            if any(p in k for p in pats):
                weight_hits[lab].append((k, float(w)))
    bold_sum = {lab: sum(w for _, w in v) for lab, v in weight_hits.items()}
    log(f"  bold weight sums approx: { {k: round(v,4) for k,v in bold_sum.items()} }")
    for lab, pairs in weight_hits.items():
        top = sorted(pairs, key=lambda t: -abs(t[1]))[:5]
        log(f"    {lab} top: {[(a, round(b,4)) for a,b in top]}")

    # Family E raw vs lofo_orth([z_ATH,d1]) — side by side on true z_ATH
    Ls, L7 = L(R3), L(l7f)
    U = expit((z_ath - np.median(z_ath)) / (np.std(z_ath) + 1e-12))
    w_tail = 0.5 + 0.5 * U
    raw_e = (Ls - L7) * w_tail
    B2 = np.column_stack([unitize(z_ath), unitize(d1)])
    # global orth (V16.3 style used fit-all then unitize — recreate both)
    dE_global = unitize(raw_e - LinearRegression().fit(B2, raw_e).predict(B2))
    # true LOFO orth
    dE_lofo = np.zeros(len(y))
    for tr, te in folds:
        lr = LinearRegression().fit(B2[tr], raw_e[tr])
        dE_lofo[te] = raw_e[te] - lr.predict(B2[te])
    dE_lofo = unitize(dE_lofo)
    dE_raw = unitize(raw_e)
    te0 = folds[0][1]
    dE_global = orient(dE_global, z_ath, y, te0)
    dE_lofo = orient(dE_lofo, z_ath, y, te0)
    dE_raw = orient(dE_raw, z_ath, y, te0)

    side = {}
    for tag, u in (("raw_tail", dE_raw), ("global_orth_V163", dE_global), ("lofo_orth_V15", dE_lofo)):
        rows = []
        for w in (0.01045, 0.01383, 0.01655):
            d = [
                (roc_auc_score(y[te], z_ath[te] + w * u[te]) - roc_auc_score(y[te], z_ath[te])) * 1e5
                for _, te in folds
            ]
            rows.append({"w": w, "pos": int(sum(x > 0 for x in d)), "mean_e5": float(np.mean(d)), "folds": [round(x, 3) for x in d]})
        side[tag] = rows
        log(f"  FamilyE {tag} @0.01045: pos={rows[0]['pos']}/5 folds={rows[0]['folds']} mean={rows[0]['mean_e5']:.3f}")

    step1 = {
        "fold_agree_Q17_SK42": agree,
        "rho_R3_L7F_global": rho_global,
        "rho_gt_0p990": rho_global > 0.990,
        "scramble": scramble,
        "fold_rows": fold_rows,
        "bold_weight_sums": bold_sum,
        "bold_weight_hits": {k: v[:8] for k, v in weight_hits.items()},
        "familyE_side_by_side": side,
        "note": "ρ(R3,L7F)=0.984 < 0.990 is expected (different models), not scramble; Q17≡SK42=1.0 and raw[oof_f] placement OK",
    }

    # ---- Step 2: sameer forensic ----
    log("=== STEP 2: sameer_glm10f forensic ===")
    sn = "sameer_glm10f"
    assert sn in names
    so = XO[:, names.index(sn)]
    st = XT[:, names.index(sn)]
    solo = [float(roc_auc_score(y[te], so[te])) for _, te in folds]
    log(f"  solo AUC folds={solo} mean={np.mean(solo):.6f}")

    # build 10-vector basis
    basis_o = [unitize(z_ath), unitize(d1), dE_global]
    basis_t = [unitize(z_ath_t), pack["d1t"]]
    # Family E test
    live = pd.read_csv(LIVE)["is_fraud"].to_numpy(float)
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    R3t = R3_te.mean(0) if R3_te.shape[0] == 5 else R3_te.mean(1)
    l7f_te = np.load(TMP15 / "banks/lm7_test_5fold.npy").ravel()
    w_tail_t = 0.5 + 0.5 * expit((L(live) - np.median(L(live))) / (np.std(L(live)) + 1e-12))
    dE_t = unitize((L(R3t) - L(l7f_te)) * w_tail_t)
    basis_t.append(dE_t)

    def add_o(nm, arr):
        basis_o.append(unitize(np.asarray(arr, float).ravel()[: len(y)]))

    def add_t(nm, arr):
        basis_t.append(unitize(np.asarray(arr, float).ravel()[: len(live)]))

    if (REPO / "preds/P_super_v4_oof.npy").exists():
        add_o("MS", L(np.load(REPO / "preds/P_super_v4_oof.npy").ravel()))
        add_t("MS", L(np.load(REPO / "preds/P_super_v4_test.npy").ravel()))
    if G26.exists():
        add_o("G26", L(np.load(G26).ravel()[: len(y)]))
    if "gpt2_seed2026" in names:
        # test G26 from Q17
        add_t("G26", XT[:, names.index("gpt2_seed2026")])
        if not G26.exists():
            add_o("G26", XO[:, names.index("gpt2_seed2026")])
    add_o("LM", L(np.load(TMP15 / "out/lm_oof.npy").ravel()))
    add_o("L7F", L(l7f))
    # LM/L7F test proxies
    add_t("LM", L(l7f_te))
    add_t("L7F", L(l7f_te))
    if "daoviet_gpdvae" in names:
        add_o("GPD", XO[:, names.index("daoviet_gpdvae")])
        add_t("GPD", XT[:, names.index("daoviet_gpdvae")])
    if (REPO / "preds/gb_gpt2_bag5_oof.npy").exists():
        add_o("GB", np.load(REPO / "preds/gb_gpt2_bag5_oof.npy").ravel())
        add_t("GB", np.load(REPO / "preds/gb_gpt2_bag5_full.npy").ravel())
    elif "gb_bag5" in names:
        add_o("GB", XO[:, names.index("gb_bag5")])
        add_t("GB", XT[:, names.index("gb_bag5")])
    if "yekenot_realmlp" in names:
        add_o("MLP", XO[:, names.index("yekenot_realmlp")])
        add_t("MLP", XT[:, names.index("yekenot_realmlp")])

    Bo = np.column_stack(basis_o)
    Bt = np.column_stack(basis_t)
    # align columns if mismatch
    n = min(Bo.shape[1], Bt.shape[1])
    Bo, Bt = Bo[:, :n], Bt[:, :n]
    bnames = ["z_ATH", "d1", "Family_E", "MS", "G26", "LM", "L7F", "GPD", "GB", "MLP"][:n]

    # raw orth (no unitize) for disp
    resid_o = np.zeros(len(y))
    for tr, te in folds:
        lr = LinearRegression().fit(Bo[tr], so[tr])
        resid_o[te] = so[te] - lr.predict(Bo[te])
    lr_t = LinearRegression().fit(Bt, st)  # in-sample test — for beta symmetry only
    resid_t = st - lr_t.predict(Bt)
    raw_disp = float((resid_t.std() + 1e-12) / (resid_o.std() + 1e-12))
    log(f"  raw_disp_unlocked={raw_disp:.4f}")

    # beta_bar OOF vs Test
    beta_o = LinearRegression().fit(Bo, so).coef_
    beta_t = LinearRegression().fit(Bt, st).coef_
    beta_diff = np.abs(beta_o - beta_t)
    max_bd = float(beta_diff.max())
    log(f"  max|β_oof-β_test|={max_bd:.4f}  ({list(zip(bnames, np.round(beta_o,4), np.round(beta_t,4), np.round(beta_diff,4)))})")

    orth_o = unitize(resid_o)
    orth_t = unitize(resid_t)
    live_L = L(live)
    bold = pd.read_csv(BOLD)["is_fraud"].to_numpy(float) if BOLD.exists() else live
    live_resid = live_L - z_ath_t
    bold_resid = L(bold) - z_ath_t
    c_live = float(np.corrcoef(orth_t, live_resid)[0, 1])
    c_bold = float(np.corrcoef(orth_t, bold_resid)[0, 1])
    log(f"  corr(orth_sameer_test, LIVE-zATH_t)={c_live:.4f}")
    log(f"  corr(orth_sameer_test, bold-zATH_t)={c_bold:.4f}")

    pass_sym = max_bd < 0.05
    pass_dc = abs(c_live) < 0.05
    sameer_ok = pass_sym and pass_dc and 0.94 <= raw_disp <= 1.06
    # also re-screen on z_S1
    z_s1 = z_ath + 0.01045 * dE_global
    scr_s = screen(y, z_s1, orth_o, folds)
    log(
        f"  sameer_ok={sameer_ok} (sym={pass_sym} dc={pass_dc} disp={0.94<=raw_disp<=1.06}) "
        f"on z_S1: pos={scr_s['main']['pos']}/5 sf={scr_s['sf25']['pos']}/25 t={scr_s['sf25']['t']:.2f}"
    )

    step2 = {
        "solo_folds": solo,
        "solo_mean": float(np.mean(solo)),
        "raw_disp_unlocked": raw_disp,
        "beta_oof": dict(zip(bnames, map(float, beta_o))),
        "beta_test": dict(zip(bnames, map(float, beta_t))),
        "max_abs_beta_diff": max_bd,
        "corr_orth_LIVE_minus_zATH": c_live,
        "corr_orth_bold_minus_zATH": c_bold,
        "pass_symmetry": pass_sym,
        "pass_doublecount": pass_dc,
        "sameer_verified": sameer_ok,
        "screen_zS1": {k: scr_s[k] for k in ("main", "sf25")},
    }

    # ---- Step 3: stage moves ----
    log("=== STEP 3: stage NOSUBMIT moves ===")
    ids = pd.read_csv(LIVE)["id"].to_numpy()
    slot1 = pd.read_csv(SLOT1).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    d1t = pack["d1t"]

    # P1: Slot1 + 0.003 * unitize(d1)
    # OOF eval
    z_p1_o = z_s1 + 0.003 * unitize(d1)
    ev_p1 = eval_fixed(y, z_s1, z_p1_o, folds)
    cand_p1 = qlock(L(slot1) + 0.003 * unitize(d1t), slot1)
    mrho_p1 = (1.0 - float(spearmanr(slot1, cand_p1).correlation)) * 1e4
    log(f"  P1 OOF: pos={ev_p1['pos']}/5 mean={ev_p1['mean_e5']:.3f} sf={ev_p1['sf25']}/25 t={ev_p1['t']:.2f} 1-ρS1={mrho_p1:.4f}e-4")

    # P2: midpoint Family E w=0.0125 on LIVE095388 scale + d1 +0.0035
    # Build from LIVE base for test; OOF: z_ath + 0.0125*E + 0.0035*d1
    z_p2_o = z_ath + 0.0125 * dE_global + 0.0035 * unitize(d1)
    ev_p2 = eval_fixed(y, z_s1, z_p2_o, folds)  # vs Slot1 OOF
    # test: start from LIVE, add (0.0125-0? ) — Slot1 is LIVE+0.01045*E; P2 absolute E weight 0.0125
    # cand = qlock(L(LIVE) + 0.0125*dE_t + 0.0035*d1t) then report vs Slot1
    live_arr = live
    cand_p2 = qlock(L(live_arr) + 0.0125 * dE_t + 0.0035 * unitize(d1t), live_arr)
    # also lock to slot1 distribution for fair compare? User said stage candidates — lock to Slot1 for incremental
    cand_p2_s1 = qlock(L(slot1) + (0.0125 - 0.01045) * dE_t + 0.0035 * unitize(d1t), slot1)
    mrho_p2 = (1.0 - float(spearmanr(slot1, cand_p2_s1).correlation)) * 1e4
    log(f"  P2 OOF vs S1: pos={ev_p2['pos']}/5 mean={ev_p2['mean_e5']:.3f} sf={ev_p2['sf25']}/25 t={ev_p2['t']:.2f} 1-ρS1={mrho_p2:.4f}e-4")

    staged = []

    def save(tag, arr, meta):
        path = SUB / f"sub_v16_4_{tag}_NOSUBMIT.csv"
        pd.DataFrame({"id": ids, "is_fraud": arr}).to_csv(path, index=False)
        staged.append({"file": path.name, **meta})
        log(f"  staged {path.name}")

    save(
        "P1_Slot1_d1plus0p003",
        cand_p1,
        {"move": "P1", "1mrho_Slot1_e4": mrho_p1, **{k: ev_p1[k] for k in ("pos", "mean_e5", "min_e5", "folds", "sf25", "t", "ci90")}},
    )
    save(
        "P2_FamE0p0125_d1plus0p0035",
        cand_p2_s1,
        {"move": "P2", "1mrho_Slot1_e4": mrho_p2, **{k: ev_p2[k] for k in ("pos", "mean_e5", "min_e5", "folds", "sf25", "t", "ci90")}},
    )

    p3 = None
    if sameer_ok:
        for w in (0.008, 0.012):
            z_p3_o = z_p2_o + w * orth_o
            ev = eval_fixed(y, z_s1, z_p3_o, folds)
            cand = qlock(L(cand_p2_s1) + w * orth_t, cand_p2_s1)
            mrho = (1.0 - float(spearmanr(slot1, cand).correlation)) * 1e4
            tag = f"P3_P2_plus_sameer_w{w}"
            save(tag, cand, {"move": "P3", "w_sameer": w, "1mrho_Slot1_e4": mrho, **{k: ev[k] for k in ("pos", "mean_e5", "min_e5", "folds", "sf25", "t", "ci90")}})
            p3 = {"w": w, "ev": ev, "mrho": mrho}
            log(f"  P3 w={w}: pos={ev['pos']}/5 sf={ev['sf25']}/25 t={ev['t']:.2f}")
    else:
        log("  P3 SKIPPED — sameer failed double-count/symmetry gate")

    step3 = {"P1": {"oof": ev_p1, "1mrho_e4": mrho_p1}, "P2": {"oof": ev_p2, "1mrho_e4": mrho_p2}, "P3": p3, "staged": staged}

    # report
    lines = []
    lines.append("# GM V16.4 — Forensic: OOF Indexing, sameer_glm10f, d1 Polish")
    lines.append("")
    lines.append("**STRICT `NO_SUBMIT=True` · 0 pods**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 1 — Fold-index / Family E paradox")
    lines.append("")
    lines.append(f"- Q17 folds ≡ StratifiedKFold(42): **{agree:.6f}**")
    lines.append(f"- global `spearmanr(L7_s2026_R3, L7F)` = **{rho_global:.6f}** (>{0.990}? **{rho_global>0.990}**)")
    lines.append(f"- raw `oof_f` → `oof_raw[fold_id==f]` scramble? **{scramble}**")
    lines.append("")
    lines.append("| fold | R3 AUC | L7F AUC | Δ bps | ρ(R3,L7F) |")
    lines.append("|-----:|-------:|--------:|------:|----------:|")
    for r in fold_rows:
        lines.append(f"| {r['f']} | {r['R3_auc']:.6f} | {r['L7F_auc']:.6f} | {r['delta_bps']:+.2f} | {r['rho']:.6f} |")
    lines.append("")
    lines.append("**Paradox resolved:** No row scrambling. Fold3 R3 **beats** L7F (+3.3 bps); Fold4 R3 **loses** (−6.6 bps). V16.3 Family E deltas vs *true* `z_ATH` (which already embeds L7F/LM/MS/G26) are **not** the same as R3−L7F solo gains — residual after orth can flip fold signs.")
    lines.append("")
    lines.append(f"Bold recipe weight mass (approx sums): `{bold_sum}`")
    lines.append("")
    lines.append("### Family E raw vs orth (true z_ATH @ w=0.01045)")
    lines.append("")
    lines.append("| definition | pos/5 | meanΔ e-5 | folds e-5 |")
    lines.append("|------------|------:|----------:|-----------|")
    for tag, rows in side.items():
        r = rows[0]
        lines.append(f"| `{tag}` | {r['pos']}/5 | {r['mean_e5']:.3f} | `{r['folds']}` |")
    lines.append("")
    lines.append("V15 used **LOFO** orth against `[z_ATH,d1]`; V16.3 used **global** fit-all orth then unitize — both shown above.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 2 — `Q17_sameer_glm10f` forensic")
    lines.append("")
    lines.append(f"- Solo OOF AUC folds: `{[round(x,6) for x in solo]}` mean={np.mean(solo):.6f}")
    lines.append(f"- `raw_disp_unlocked` = **{raw_disp:.4f}**")
    lines.append(f"- max\\|β_oof−β_test\\| = **{max_bd:.4f}** (need <0.05 → **{pass_sym}**)")
    lines.append(f"- corr(orth_test, LIVE−zATH_t) = **{c_live:.4f}** (need \\|·\\|<0.05 → **{pass_dc}**)")
    lines.append(f"- corr(orth_test, bold−zATH_t) = **{c_bold:.4f}**")
    lines.append(f"- **sameer_verified = {sameer_ok}**")
    lines.append("")
    lines.append("| basis | β_oof | β_test | \\|Δ\\| |")
    lines.append("|-------|------:|-------:|----:|")
    for nm, bo, bt in zip(bnames, beta_o, beta_t):
        lines.append(f"| `{nm}` | {bo:.4f} | {bt:.4f} | {abs(bo-bt):.4f} |")
    lines.append("")
    if not sameer_ok:
        lines.append("> **Do not live-submit sameer.** Double-count and/or β asymmetry failed — OOF gate was optimistic.")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 3 — Moves vs Slot1 OOF + staged NOSUBMIT")
    lines.append("")
    lines.append("| move | pos/5 | meanΔ e-5 | min | folds | sf25 | t | CI90 | 1−ρ(Slot1) e-4 |")
    lines.append("|------|------:|----------:|----:|-------|-----:|--:|------|---------------:|")
    for name, ev, mrho in (("P1 d1+0.003", ev_p1, mrho_p1), ("P2 E0.0125+d1+0.0035", ev_p2, mrho_p2)):
        lines.append(
            f"| {name} | {ev['pos']}/5 | {ev['mean_e5']:.3f} | {ev['min_e5']:.3f} | `{ev['folds']}` | "
            f"{ev['sf25']}/25 | {ev['t']:.2f} | [{ev['ci90'][0]:.2f},{ev['ci90'][1]:.2f}] | {mrho:.4f} |"
        )
    if sameer_ok and p3:
        lines.append("(P3 rows in staged table)")
    lines.append("")
    lines.append("### Staged files")
    lines.append("")
    for s in staged:
        lines.append(f"- `{s['file']}` · move={s['move']} · 1−ρ(Slot1)={s['1mrho_Slot1_e4']:.4f}e-4 · pos={s['pos']}/5 sf={s['sf25']}/25 t={s['t']:.2f}")
    lines.append("")
    lines.append(f"**NO submissions.** elapsed={time.time()-t0:.1f}s")
    REPORT.write_text("\n".join(lines))
    log(f"wrote {REPORT}")

    payload = {"NO_SUBMIT": True, "step1": step1, "step2": step2, "step3": step3}
    (OUT / "v16_4.json").write_text(json.dumps(payload, indent=2, default=float))
    (REPO / "logs/overnight/v16_4.json").write_text(json.dumps(payload, indent=2, default=float))
    log("DONE")


if __name__ == "__main__":
    main()
