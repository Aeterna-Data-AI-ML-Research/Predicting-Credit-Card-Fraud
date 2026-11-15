#!/usr/bin/env python3
"""GM V22.1 — Leap past 0.95422 via sriram/goodpjw synthesis + proprietary engines.

Local CPU only. Rebuilds Slot8/9 anchors via V20 factory, audits ΔRealMLP OOF+test,
decomposes sriram_0.95422 vs Slot9, stages C1/C2/C3, submits best OOF-verified leap.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO = Path(__file__).resolve().parents[1]
TMP16 = Path("/tmp/s6e11_v16")
OUT, SUB, LOG = TMP16 / "out", REPO / "submissions", REPO / "logs/overnight"
NB = Path("/tmp/s6e11_nb_research")
SRIRAM_DS = NB / "sriram_ds"
PYTAB_OOF = NB / "yekenot_pytabkit" / "oof_preds.csv"
PYTORCH_OOF = NB / "yekenot_pytorch" / "oof_preds.csv"
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
SLOT9 = SUB / "sub_v20_LIVE_slot9_overnight_cpu_promoted.csv"
SLOT8 = SUB / "sub_v19_4_LIVE_slot8_A_cat_wo0.016_wc0.012_wu0.012.csv"
SLOT5 = SUB / "sub_v17_9_LIVE_slot5_Slot3_plus_R2_LGB_w0p020.csv"
SLOT6 = SUB / "sub_v18_1_LIVE_slot6_Slot5_plus_R1_pure_Bo10_w0p022.csv"
SLOT7 = SUB / "sub_v19_0_LIVE_slot7_zero_echo_peer_hybrid.csv"
SLOT3 = SUB / "sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
REPORT = LOG / "GM_V22_1_LIVE_LEAP_REPORT.md"
N_TR, N_TE = 699960, 299984
W8_O, W8_C, W8_U, W_E2 = 0.016, 0.012, 0.012, 0.020
T0 = time.time()
LINES: list[str] = []


def log(msg: str) -> None:
    s = f"[{time.time()-T0:7.1f}s] {msg}"
    print(s, flush=True)
    LINES.append(s)


def load_v20():
    path = REPO / "scripts" / "gm_v20_overnight_cpu_grandmaster_factory.py"
    spec = importlib.util.spec_from_file_location("gm_v20_fac", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def one_m_rho_e4(a, b):
    return (1.0 - float(spearmanr(a, b).correlation)) * 1e4


def pearson_1mr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return 1.0 - float(np.corrcoef(a, b)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    v20 = load_v20()
    L, U, unitize, qlock = v20.L, v20.U, v20.unitize, v20.qlock
    h2h, frozen_lofo_orth, two_stage_orth, forensic = (
        v20.h2h,
        v20.frozen_lofo_orth,
        v20.two_stage_orth,
        v20.forensic,
    )

    log("=== V22.1 START: rebuild Slot8/9 anchors ===")
    fac = v20.OvernightFactory()
    fac.rebuild()
    f = fac
    ids = f.ids
    y, folds = f.y, f.folds

    # Live submissions aligned to ids
    def load_sub(path):
        df = pd.read_csv(path)
        return df.set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    slot9 = load_sub(SLOT9)
    slot8 = load_sub(SLOT8)
    slot5 = load_sub(SLOT5)
    slot6 = load_sub(SLOT6)
    slot7 = load_sub(SLOT7)
    slot3 = load_sub(SLOT3)
    new_mlp_t = load_sub(SRIRAM_DS / "new_neural.csv")
    old_mlp_t = load_sub(SRIRAM_DS / "old_neural.csv")
    goodpjw = load_sub(SRIRAM_DS / "reference.csv")  # 0.95417
    # Exact sriram 0.95422
    sriram_t = U(L(goodpjw) + 0.15 * (L(new_mlp_t) - L(old_mlp_t)))
    log(f"rebuild sriram vs goodpjw 1-ρ={one_m_rho_e4(sriram_t, goodpjw):.3f}e-4")

    # OOF RealMLP
    o_new = pd.read_csv(PYTAB_OOF).is_fraud.to_numpy(float)
    o_old = pd.read_csv(PYTORCH_OOF).is_fraud.to_numpy(float)
    assert len(o_new) == N_TR and len(o_old) == N_TR
    dmlp_o = L(o_new) - L(o_old)
    dmlp_t = L(new_mlp_t) - L(old_mlp_t)
    log(f"ΔRealMLP oof_std={dmlp_o.std():.4f} test_std={dmlp_t.std():.4f} "
        f"auc_new={roc_auc_score(y,o_new):.6f} auc_old={roc_auc_score(y,o_old):.6f}")

    # Proprietary OOF/test from vault + families
    vault = np.load(OUT / "gm_overnight_vault.npz")
    u_e2, ut_e2 = unitize(vault["E2_heavy_ridge_oof"]), unitize(vault["E2_heavy_ridge_test"])
    # orient E2 like factory
    if roc_auc_score(y[f.te0], f.z_s8[f.te0] + 0.01 * u_e2[f.te0]) < roc_auc_score(
        y[f.te0], f.z_s8[f.te0] - 0.01 * u_e2[f.te0]
    ):
        u_e2, ut_e2 = -u_e2, -ut_e2
    # Slot9 OOF proxy = z_S8 + W_E2 * u_e2 (all OOF-length)
    z_s9 = f.z_s8 + W_E2 * u_e2
    z_t9 = f.z_t8 + W_E2 * u_e2  # twin OOF
    z_n9 = f.z_n8 + W_E2 * u_e2
    bases9 = {"z_S9": z_s9, "twin": z_t9, "nnls": z_n9}

    # Family_E = dE already in factory as dE_o / dE_t
    u_fe, ut_fe = unitize(f.dE_o), unitize(f.dE_t)

    # Active basis including Slot9 engines
    Bo_act = np.column_stack([f.Bo12, u_e2]).astype(np.float64)
    Bt_act = np.column_stack([f.Bt12, ut_e2]).astype(np.float64)
    Bt_all = np.column_stack([Bt_act, f.bad6, f.bad7]).astype(np.float64)

    # ---------- Step1: ΔRealMLP on Slot9 OOF ----------
    log("=== Step1: ΔRealMLP H2H on Slot9 OOF bases ===")
    oo_d, ot_d, pre_d = two_stage_orth(dmlp_o, dmlp_t, Bo_act, Bt_act, Bt_all, folds, ridge_alpha=10.0)
    u_d, ut_d = fac.orient(oo_d, ot_d)
    g12_d = forensic(u_d, ut_d, Bt_act)
    rows_d = []
    for w in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
        per = {b: h2h(bases9[b] + w * u_d, bases9[b], y, folds) for b in bases9}
        rho = one_m_rho_e4(qlock(L(slot9) + w * ut_d, slot9), slot9)
        rows_d.append((w, per, rho))
        log(
            f"  dMLP w={w:.2f}: z9={per['z_S9']['pos']}/5 m={per['z_S9']['mean']:+.2f} "
            f"min={per['z_S9']['min']:+.2f} sf={per['z_S9']['sf25']} t={per['z_S9']['t']:.2f} "
            f"ρ={rho:.3f} disp={pre_d['raw_disp']:.3f} F2={g12_d['F2']:.4f}"
        )

    # Bank member orth lift vs z_S9 (sample of key names)
    log("=== Step1b: bank member orth lift vs z_S9 (selected) ===")
    want = [
        "gb_bag5",
        "ebm_dig",
        "ebm_diglead_bag",
        "ebm_dig_r7",
        "gbn_bag5_w512",
        "realmlp_dig",
        "realmlp_te",
        "lgb_dig_l4",
        "glm_te",
    ]
    for name in want:
        po, pt = BANK / f"oof_{name}.npy", BANK / f"test_{name}.npy"
        if not (po.exists() and pt.exists()):
            continue
        vo, vt = np.load(po).ravel()[:N_TR], np.load(pt).ravel()[:N_TE]
        oo, ot, _ = frozen_lofo_orth(vo, vt, Bo_act, Bt_act, folds, ridge_alpha=10.0)
        u = unitize(oo)
        if roc_auc_score(y[f.te0], z_s9[f.te0] + 0.01 * u[f.te0]) < roc_auc_score(
            y[f.te0], z_s9[f.te0] - 0.01 * u[f.te0]
        ):
            u = -u
        per = h2h(z_s9 + 0.012 * u, z_s9, y, folds)
        log(f"  bank:{name}: {per['pos']}/5 m={per['mean']:+.2f} min={per['min']:+.2f} t={per['t']:.2f}")

    # ---------- Step2: distance matrix + regression decomp ----------
    log("=== Step2: distance matrix + R² decomposition ===")
    # Slot9 + dMLP candidates on test
    slot9_d08 = U(L(slot9) + 0.08 * unitize(dmlp_t))
    slot9_d12 = U(L(slot9) + 0.12 * unitize(dmlp_t))
    names_t = {
        "Slot5": slot5,
        "Slot8": slot8,
        "Slot9": slot9,
        "Slot9+d08": slot9_d08,
        "Slot9+d12": slot9_d12,
        "goodpjw": goodpjw,
        "sriram": sriram_t,
    }
    keys = list(names_t)
    log("1-ρ e-4 (Spearman) matrix:")
    hdr = " ".join(f"{k:>10s}" for k in keys)
    log(f"{'':10s} {hdr}")
    for a in keys:
        cells = []
        for b in keys:
            cells.append(f"{one_m_rho_e4(names_t[a], names_t[b]):10.2f}")
        log(f"{a:10s} " + " ".join(cells))

    # Fail-ray checks
    delta_s9 = unitize(L(sriram_t) - L(slot9))
    c6_s = float(np.corrcoef(delta_s9, f.bad6)[0, 1])
    c7_s = float(np.corrcoef(delta_s9, f.bad7)[0, 1])
    c6_d = float(np.corrcoef(unitize(dmlp_t), f.bad6)[0, 1])
    c7_d = float(np.corrcoef(unitize(dmlp_t), f.bad7)[0, 1])
    log(f"fail-rays: sriram-Slot9 c6={c6_s:+.4f} c7={c7_s:+.4f} | Δmlp c6={c6_d:+.4f} c7={c7_d:+.4f}")

    # Regress L(sriram)-L(slot9) onto test basis blocks
    y_delta = L(sriram_t) - L(slot9)
    blocks = {
        "Bo9": f.Bt9,
        "dMLP": unitize(dmlp_t).reshape(-1, 1),
        "prop_S589": np.column_stack([f.ut_r2, f.ut_cat, f.ut_u5, ut_e2]),
        "full": np.column_stack([f.Bt9, f.ut_r2, f.ut_cat, f.ut_u5, ut_e2, unitize(dmlp_t)]),
    }
    # also prop alone and Bo9+dMLP
    blocks["Bo9+dMLP"] = np.column_stack([f.Bt9, unitize(dmlp_t)])
    blocks["prop+dMLP"] = np.column_stack([f.ut_r2, f.ut_cat, f.ut_u5, ut_e2, unitize(dmlp_t)])
    sst = float(np.var(y_delta) * len(y_delta))
    for name, X in blocks.items():
        lr = Ridge(alpha=1.0, fit_intercept=True).fit(X, y_delta)
        pred = lr.predict(X)
        r2 = 1.0 - float(np.sum((y_delta - pred) ** 2)) / (sst + 1e-18)
        log(f"  R²[{name}]={r2:.4f}")
    # proprietary weight check: coefs on prop columns in full model
    Xf = blocks["full"]
    lr = Ridge(alpha=1.0, fit_intercept=True).fit(Xf, y_delta)
    n_bo9 = f.Bt9.shape[1]
    coef_prop = lr.coef_[n_bo9 : n_bo9 + 4]
    coef_d = lr.coef_[-1]
    log(f"  full coefs prop[R2,Cat,U5,E2]={coef_prop.round(4).tolist()} dMLP={coef_d:+.4f}")

    # ---------- Step3: Candidates ----------
    log("=== Step3: build C1 / C2 / C3 ===")

    def orth_vec(v_o, v_t, Bo=Bo_act, Bt=Bt_act):
        oo, ot, pre = two_stage_orth(v_o, v_t, Bo, Bt, Bt_all, folds, ridge_alpha=10.0)
        u, ut = fac.orient(oo, ot)
        return u, ut, pre

    # Proprietary increment OOF/test (Slot9 vs Slot3 engines)
    # Δ_slot9_vs_slot3 ≈ W8_O*u_r2 + W8_C*u_cat + W8_U*u_u5 + W_E2*u_e2 (+ small Family_E already in z_ath path)
    prop_o = W8_O * f.u_r2 + W8_C * f.u_cat + W8_U * f.u_u5 + W_E2 * u_e2 + 0.01045 * u_fe
    prop_t = W8_O * f.ut_r2 + W8_C * f.ut_cat + W8_U * f.ut_u5 + W_E2 * ut_e2 + 0.01045 * ut_fe

    # Orth prop against goodpjw/realmlp basis on test for C2 inject
    def q17(name, fallback_o, fallback_t):
        if name in f.XO:
            return unitize(L(f.XO[name])), unitize(L(f.XT[name]))
        return unitize(fallback_o), unitize(fallback_t)

    ms_o_u, ms_t_u = q17("masaya_gpt2", f.ms_o, f.ms_t)
    lm_o_u, lm_t_u = q17("megayak_5f_lm", f.ms_o, f.ms_t)
    l7_o_u, l7_t_u = q17("lm7_5fold", f.ms_o, f.ms_t)
    yek_o_u, yek_t_u = q17("yekenot_realmlp", o_old, old_mlp_t)
    B_gp_o = np.column_stack([lm_o_u, ms_o_u, l7_o_u, yek_o_u, unitize(L(o_new)), unitize(L(o_old))])
    B_gp_t = np.column_stack(
        [lm_t_u, ms_t_u, l7_t_u, yek_t_u, unitize(L(new_mlp_t)), unitize(L(old_mlp_t))]
    )

    prop_o_g, prop_t_g, pre_prop = orth_vec(prop_o, prop_t, Bo=B_gp_o, Bt=B_gp_t)
    log(f"prop|goodpjw_basis raw_disp={pre_prop['raw_disp']:.3f} raw_KS={pre_prop['raw_KS']:.4f}")

    # Approximate sriram OOF: L(z_s9-like public) hard — use bold-ish from Q17 + 0.15*Δmlp_oof
    # Practical proxy: unitize(L(z_s9)) is our side; sriram_oof_proxy = U(L(goodpjw_proxy)+0.15*dmlp_o)
    # goodpjw_proxy OOF: NNLS of Q17 streams toward... we don't have goodpjw OOF.
    # Use: sriram_oof_proxy = z_s9 + 0.15*unitize(dmlp_o) shifted? Better:
    # Evaluate C2 OOF as: bases that include injecting prop into (z_s9 + aligned_dmlp) vs z_s9,
    # AND separately score blend of unitized public ladder proxy.
    # Proxy ladder OOF = Ridge predict of L(goodpjw_test) from Bt streams transferred to Bo (dangerous).
    # Honest OOF for C2: treat sriram_oof ≈ z_s9 + α*u_d + β*(public residual)
    # Use: sriram_oof_proxy = unitize(L(oof from blending Q17 to match goodpjw on test via ridge β))

    lr_map = Ridge(alpha=1.0, fit_intercept=True).fit(f.Bt9, L(goodpjw))
    goodpjw_oof_proxy = f.Bo9 @ lr_map.coef_ + lr_map.intercept_
    sriram_oof_proxy = L(U(np.exp(goodpjw_oof_proxy) / (1 + np.exp(goodpjw_oof_proxy))))  # bad
    # cleaner: stay in logit-rank space
    sriram_oof_proxy = goodpjw_oof_proxy + 0.15 * dmlp_o
    sriram_oof_proxy = unitize(sriram_oof_proxy)
    # calibrate scale to z_s9
    sriram_oof_proxy = unitize(sriram_oof_proxy) * float(np.std(z_s9)) + float(np.mean(z_s9))
    log(f"sriram_oof_proxy AUC={roc_auc_score(y, sriram_oof_proxy):.6f} "
        f"z_s9 AUC={roc_auc_score(y, z_s9):.6f}")

    candidates = []

    def eval_cand(tag, oof_score, test_score, vs_base_oof=None):
        """OOF H2H vs bases9; test ρ vs Slot9 and sriram."""
        if vs_base_oof is None:
            vs_base_oof = z_s9
        # For blend candidates, H2H vs z_s9 may be negative — also report vs sriram_oof_proxy
        per9 = {b: h2h(oof_score if b == "z_S9" else (bases9[b] + (oof_score - z_s9)), bases9[b], y, folds) for b in bases9}
        # simpler: direct H2H of oof_score vs z_s9 and vs sriram_oof_proxy
        per_z = h2h(oof_score, z_s9, y, folds)
        per_s = h2h(oof_score, sriram_oof_proxy, y, folds)
        rho9 = one_m_rho_e4(test_score, slot9)
        rhos = one_m_rho_e4(test_score, sriram_t)
        delta = unitize(L(test_score) - L(slot9))
        c6 = float(np.corrcoef(delta, f.bad6)[0, 1])
        c7 = float(np.corrcoef(delta, f.bad7)[0, 1])
        row = {
            "tag": tag,
            "per_z": per_z,
            "per_s": per_s,
            "rho9": rho9,
            "rhos": rhos,
            "c6": c6,
            "c7": c7,
            "test": test_score,
            "oof": oof_score,
            "auc": float(roc_auc_score(y, oof_score)),
            "score": float(per_s["mean"] + 0.5 * per_z["mean"] + 0.3 * per_s["min"]),
        }
        log(
            f"{tag}: AUC={row['auc']:.6f} vs_z9 {per_z['pos']}/5 m={per_z['mean']:+.2f} min={per_z['min']:+.2f} "
            f"vs_sriProx {per_s['pos']}/5 m={per_s['mean']:+.2f} min={per_s['min']:+.2f} "
            f"ρ9={rho9:.3f} ρs={rhos:.3f} c7={c7:+.3f}"
        )
        candidates.append(row)
        return row

    # C1: ATH + orth(Δmlp)
    for w in (0.08, 0.10, 0.12):
        oof = z_s9 + w * u_d
        te = U(L(slot9) + w * ut_d)
        eval_cand(f"C1_ATH_dMLP_w{w:.2f}", oof, te)

    # also raw (non-orth) Δ on test for C1 variant
    for w in (0.08, 0.12):
        oof = z_s9 + w * unitize(dmlp_o)
        # orient
        if roc_auc_score(y[f.te0], oof[f.te0]) < roc_auc_score(y[f.te0], z_s9[f.te0] - w * unitize(dmlp_o)[f.te0]):
            oof = z_s9 - w * unitize(dmlp_o)
            te = U(L(slot9) - w * unitize(dmlp_t))
        else:
            te = U(L(slot9) + w * unitize(dmlp_t))
        eval_cand(f"C1raw_ATH_dMLP_w{w:.2f}", oof, te)

    # C2: sriram + λ*prop_orth + bridge
    # wp must be live-engine scale (~0.02–0.08 of unitize); 0.25–0.70 destroyed OOF in v1.
    for lam in (0.15, 0.25, 0.35):
        for wp in (0.02, 0.04, 0.06, 0.08):
            bridge = L(slot9_d12)
            te = U((1 - lam) * L(sriram_t) + lam * bridge + wp * prop_t_g)
            oof = (1 - lam) * sriram_oof_proxy + lam * (z_s9 + 0.12 * u_d) + wp * prop_o_g
            oof = unitize(oof) * float(np.std(z_s9)) + float(np.mean(z_s9))
            eval_cand(f"C2_sri+prop_l{lam:.2f}_wp{wp:.2f}", oof, te)

    # C2 pure inject: sriram + wp * prop (no λ bridge) — primary leap onto 0.95422 manifold
    for wp in (0.02, 0.035, 0.05, 0.065, 0.08):
        te = U(L(sriram_t) + wp * prop_t_g)
        oof = unitize(sriram_oof_proxy + wp * prop_o_g) * float(np.std(z_s9)) + float(np.mean(z_s9))
        eval_cand(f"C2pure_sri+prop_wp{wp:.3f}", oof, te)

    # C3 geodesic blends
    ath_d = slot9_d12
    for lam in (0.25, 0.35, 0.50):
        te = U((1 - lam) * L(sriram_t) + lam * L(ath_d))
        oof = (1 - lam) * sriram_oof_proxy + lam * (z_s9 + 0.12 * u_d)
        oof = unitize(oof) * float(np.std(z_s9)) + float(np.mean(z_s9))
        eval_cand(f"C3_geo_l{lam:.2f}", oof, te)

    # Pick winners for staging
    # Prefer: positive vs sriram proxy OR strong vs z9 with ρ to sriram decreasing, fail rays ok
    def rank_key(r):
        # prioritize beating sriram proxy on OOF with good min, safe c7, move toward sriram on test
        return (
            r["per_s"]["pos"],
            r["per_s"]["min"],
            r["per_s"]["mean"],
            -abs(r["c7"]),
            -r["rhos"],  # closer to sriram better for leap
        )

    ranked = sorted(candidates, key=rank_key, reverse=True)
    log("=== Top candidates ===")
    for r in ranked[:12]:
        log(
            f"  {r['tag']}: score={r['score']:.2f} vsSri {r['per_s']['pos']}/5 "
            f"m={r['per_s']['mean']:+.2f} min={r['per_s']['min']:+.2f} ρs={r['rhos']:.3f} c7={r['c7']:+.3f}"
        )

    # Stage fixed named files per user request
    def stage(path, vec):
        pd.DataFrame({"id": ids, "is_fraud": np.asarray(vec, float)}).to_csv(path, index=False)
        log(f"STAGED {path.name}")

    # C1 preferred w=0.10 orth
    c1 = next(r for r in candidates if r["tag"] == "C1_ATH_dMLP_w0.10")
    # C2: best C2pure or C2_sri
    c2_pool = [r for r in candidates if r["tag"].startswith("C2")]
    c2 = sorted(c2_pool, key=rank_key, reverse=True)[0]
    c3 = next(r for r in candidates if r["tag"] == "C3_geo_l0.50")

    p1 = SUB / "sub_v22_1_C1_ATH095399_plus_dRealMLP_w0p10.csv"
    p2 = SUB / f"sub_v22_1_C2_Sriram095422_plus_Proprietary_Slot9_Engines.csv"
    p3 = SUB / "sub_v22_1_C3_Geodesic_Sriram095422_x_ATH095399_dMLP.csv"
    stage(p1, c1["test"])
    stage(p2, c2["test"])
    stage(p3, c3["test"])
    # also save the exact c2 tag used
    meta = {
        "c1": c1["tag"],
        "c2": c2["tag"],
        "c3": c3["tag"],
        "c2_metrics": {
            "per_s": c2["per_s"],
            "per_z": c2["per_z"],
            "rho9": c2["rho9"],
            "rhos": c2["rhos"],
            "c6": c2["c6"],
            "c7": c2["c7"],
            "auc": c2["auc"],
        },
        "fail_rays": {"sriram_c6": c6_s, "sriram_c7": c7_s, "dmlp_c6": c6_d, "dmlp_c7": c7_d},
    }
    OUT.joinpath("v22_1_candidates_meta.json").write_text(json.dumps(meta, indent=2, default=float))

    # ---------- Step4: submit ----------
    log("=== Step4: live submission ===")
    mode_1b = os.environ.get("V22_1B", "").strip() in {"1", "true", "TRUE", "yes"}
    # Prefer C2 when OOF beats sriram proxy with clean fail-rays and small move (ρs < 8e-4)
    submit_row = c2
    reason = "C2_super_synthesis"
    c2_ok = (
        c2["per_s"]["pos"] >= 4
        and c2["per_s"]["min"] > -2.0
        and abs(c2["c7"]) < 0.20
        and abs(c2["c6"]) < 0.20
        and c2["rhos"] < 8.0
    )
    if c2_ok:
        submit_row = c2
        reason = "C2_primary_leap"
    elif mode_1b:
        # After C1 live 0.95407: leap via C3 geodesic toward 0.95422 manifold
        submit_row = c3
        reason = "C3_geodesic_leap"
        # Prefer C3_geo_l0.25 if present (closer to sriram)
        for r in candidates:
            if r["tag"] == "C3_geo_l0.25" and abs(r["c7"]) < 0.20:
                submit_row = r
                reason = "C3_geo_l0.25_leap"
                break
        # Override with best C2pure if any is remotely sane
        c2p = [r for r in candidates if r["tag"].startswith("C2pure") and r["rhos"] < 5 and abs(r["c7"]) < 0.20]
        if c2p:
            best = sorted(c2p, key=lambda r: (r["per_s"]["pos"], r["per_s"]["min"], -r["rhos"]), reverse=True)[0]
            if best["per_s"]["pos"] >= 3 and best["per_s"]["mean"] > -5:
                submit_row = best
                reason = "C2pure_fixed_scale"
    else:
        # v1 path: C1 fallback if C2 fails gates
        submit_row = c1
        reason = "C1_fallback_pure_lineage"
        if c3["rhos"] < c1["rhos"] and abs(c3["c7"]) < 0.20 and c3["per_z"]["mean"] > -5:
            submit_row = c3
            reason = "C3_geodesic_fallback"
        if c1["per_z"]["pos"] == 5 and c1["per_z"]["min"] > 0:
            submit_row = c1
            reason = "C1_5of5_dMLP"

    out_live = SUB / ("sub_v22_1b_LIVE_leap.csv" if mode_1b else "sub_v22_1_LIVE_leap.csv")
    pd.DataFrame({"id": ids, "is_fraud": submit_row["test"]}).to_csv(out_live, index=False)
    msg = (
        f"v22.1 leap {reason} {submit_row['tag']} | vsATH ρ={submit_row['rho9']:.3f}e-4 "
        f"vsSri ρ={submit_row['rhos']:.3f}e-4 c7={submit_row['c7']:+.3f} "
        f"OOF vsSriProx {submit_row['per_s']['pos']}/5 m={submit_row['per_s']['mean']:+.2f}"
    )
    log(f"SUBMIT {msg}")
    pub = None
    submit_ok = False
    try:
        r = subprocess.run(
            [KAG, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(out_live), "-m", msg],
            capture_output=True,
            text=True,
            timeout=180,
        )
        log(r.stdout.strip())
        log(r.stderr.strip())
        submit_ok = r.returncode == 0 and "400" not in (r.stdout + r.stderr) and "limit" not in (r.stdout + r.stderr).lower()
        if submit_ok:
            for i in range(40):
                time.sleep(20 if i else 8)
                p = subprocess.run(
                    [KAG, "competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                lines = p.stdout.strip().splitlines()
                hdr = [h.strip() for h in lines[0].split(",")] if lines else []
                for ln in lines[1:12]:
                    if "v22_1_LIVE_leap" in ln or "v22.1" in ln or "v22_1b" in ln:
                        log(f"poll {ln[:240]}")
                        if "COMPLETE" in ln.upper():
                            cols = [c.strip() for c in ln.split(",")]
                            try:
                                ix = hdr.index("publicScore")
                                pub = float(cols[ix])
                            except Exception:
                                for tok in reversed(cols):
                                    tok = tok.strip()
                                    if not tok:
                                        continue
                                    try:
                                        pub = float(tok)
                                        break
                                    except ValueError:
                                        continue
                        break
                if pub is not None:
                    break
    except Exception as e:
        log(f"submit error: {e}")
        traceback.print_exc()

    # Report
    body = [
        "# GM V22.1 Live Leap Report",
        "",
        f"**elapsed** {time.time()-T0:.0f}s · local CPU · ATH was `0.95399` · public target beat `0.95422`",
        "",
        f"- Submit: `{submit_ok}` · reason: `{reason}` · file: `{out_live.name}`",
        f"- Chosen: `{submit_row['tag']}`",
        f"- Public LB: **{pub}**",
        f"- 1−ρ vs Slot9: `{submit_row['rho9']:.3f}e-4` · vs sriram: `{submit_row['rhos']:.3f}e-4`",
        f"- Fail rays sriram−Slot9: c6=`{c6_s:+.4f}` c7=`{c7_s:+.4f}` · Δmlp c6=`{c6_d:+.4f}` c7=`{c7_d:+.4f}`",
        "",
        "## Staged",
        f"- `{p1.name}` ← {c1['tag']}",
        f"- `{p2.name}` ← {c2['tag']}",
        f"- `{p3.name}` ← {c3['tag']}",
        "",
        "## Log",
        "```",
        *LINES,
        "```",
        "",
    ]
    REPORT.write_text("\n".join(body) + "\n")
    log(f"REPORT {REPORT}")
    log(f"DONE pub={pub} submit_ok={submit_ok}")


if __name__ == "__main__":
    main()
