#!/usr/bin/env python3
"""V41 Step6e — Stage 2D diagonal candidates from two live 0.95569 rays.

STRICT NO_SUBMIT=True.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

NO_SUBMIT = True
assert NO_SUBMIT
T0 = time.time()
N_TR, N_TE = 699_960, 299_984

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
SSD = Path("/tmp/s6e11_v39_ssd")
EXT = ROOT / "external/v41_nov23_frontier"
LOG = ROOT / "logs/v41"
FINAL = ROOT / "out/v41_final_10slots"
OUT = ROOT / "out/v41_step6e_2d"
PRED = ROOT / "preds"
DATA = ROOT / "data" if (ROOT / "data/test.csv").exists() else SSD / "data"
FOLDS_P = SSD / "folds/folds.csv" if (SSD / "folds/folds.csv").exists() else ROOT / "folds/folds.csv"

FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]


def log(msg: str):
    print(f"[{time.time()-T0:5.1f}s] {msg}", flush=True)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v):
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12))


def L_fold(v, folds):
    v = np.asarray(v, float).ravel()
    out = np.zeros_like(v, dtype=np.float64)
    for k in np.unique(folds):
        m = folds == k
        out[m] = L(v[m])
    return out


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def one_mrho(a, b) -> float:
    return 1.0 - float(spearmanr(a, b).statistic)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def z_goodpjw(df: pd.DataFrame) -> np.ndarray:
    clean = (df["cvv_retry_count"].to_numpy() == 0) & (df["prior_disputes"].to_numpy() == 0)
    for f in FLAGS:
        clean &= df[f].to_numpy().astype(int) == 0
    risk, vel = df["merchant_risk_score"].to_numpy(), df["velocity_score"].to_numpy()
    return (
        (risk < 13.1)
        | ((vel <= 4.1) & (risk <= 20.0))
        | (clean & (((vel <= 4.1) & (risk <= 33.8)) | ((vel <= 6.0) & (risk <= 25.0))))
    )


def apply_zr(scores, zones, u_ref):
    out = L(scores).copy()
    uref = U(u_ref)
    if zones.any() and (~zones).any():
        floor = float(np.min(out[~zones])) - 1.0
        out[zones] = floor + uref[zones]
    return U(out)


def clip_p985(v):
    v = np.asarray(v, float).ravel()
    lim = float(np.quantile(np.abs(v), 0.985))
    return np.clip(v, -lim, lim)


def max_u_zone(p, zones):
    p = U(p)
    return float(np.max(p[zones])) if zones.any() else 0.0


def cos_full(a, b):
    a, b = unit(a), unit(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def scale_to_target_1mrho(base_u, direction, target_e4, zones, u_ab, lo=1e-6, hi=8.0):
    base_u = U(base_u)
    direction = np.asarray(direction, float).ravel()
    target = target_e4 * 1e-4
    best_a, best_diff, best_p = 0.0, 1e9, base_u
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        p = apply_zr(L(base_u) + mid * direction, zones, u_ab)
        d = one_mrho(p, base_u)
        if abs(d - target) < best_diff:
            best_diff, best_a, best_p = abs(d - target), mid, p
        if d < target:
            lo = mid
        else:
            hi = mid
    return best_p, best_a, one_mrho(best_p, base_u) * 1e4


def write_cand(path: Path, ids, scores):
    p = U(scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(index=False, lineterminator="\n")
    path.write_bytes(text.encode())
    return sha256_file(path), p


def orth_test_signal(signal_o, signal_t, base_o, base_t, folds, ridge=50.0):
    so = unit(L_fold(signal_o, folds))
    bo = unit(L_fold(base_o, folds)).reshape(-1, 1)
    st = unit(L(signal_t))
    bt = unit(L(base_t)).reshape(-1, 1)
    oof = np.zeros(len(so))
    te = np.zeros(len(st))
    for k in range(5):
        tr, va = folds != k, folds == k
        beta = np.linalg.solve(bo[tr].T @ bo[tr] + ridge * np.eye(1), bo[tr].T @ so[tr])
        oof[va] = so[va] - (bo[va] @ beta).ravel()
        te += (st - (bt @ beta).ravel()) / 5
    return unit(oof), unit(te)


def gate_oof(y, base_o, dir_o, folds, zones_tr, u_ref, alphas=(0.005, 0.01, 0.02, 0.04)):
    base_zr = apply_zr(base_o, zones_tr, u_ref)
    best = {"mean_e5": -1e9, "pos": 0, "alpha": 0.0, "sign": 0, "folds": []}
    for sgn in (1, -1):
        for a in alphas:
            p = apply_zr(L(base_o) + sgn * a * dir_o, zones_tr, u_ref)
            folds_d, pos = [], 0
            for k in range(5):
                m = folds == k
                da = (roc_auc_score(y[m], p[m]) - roc_auc_score(y[m], base_zr[m])) * 1e5
                folds_d.append(float(da))
                if da > 0:
                    pos += 1
            mean = float(np.mean(folds_d))
            if mean > best["mean_e5"]:
                best = {"mean_e5": mean, "pos": pos, "alpha": float(a), "sign": sgn, "folds": folds_d}
    return best


def load_csv(path: Path) -> np.ndarray:
    path = Path(path)
    try:
        return pd.read_csv(path)["is_fraud"].to_numpy(float)
    except OSError:
        import subprocess
        tmp = Path("/tmp") / f"_s6e11e_{path.name}"
        if not (tmp.exists() and tmp.stat().st_size > 1_000_000):
            subprocess.check_call(["cp", str(path), str(tmp)])
        return pd.read_csv(tmp)["is_fraud"].to_numpy(float)


def make_diag(ab, d1, d2, e1, e2, etot, zones, label):
    """Variance-matched 2D: scale each axis to ei, sum, rescale total to etot."""
    p1, a1, _ = scale_to_target_1mrho(ab, d1, e1, zones, ab)
    p2, a2, _ = scale_to_target_1mrho(ab, d2, e2, zones, ab)
    # sum of logit increments
    inc = (L(p1) - L(ab)) + (L(p2) - L(ab))
    # temporary combined
    p_raw = apply_zr(L(ab) + inc, zones, ab)
    # rescale total 1-rho to etot (search scalar on inc)
    p, a_tot, e_tot = scale_to_target_1mrho(ab, unit(inc), etot, zones, ab)
    return p, {
        "a1": a1, "a2": a2, "a_tot": a_tot,
        "partial_e1": e1, "partial_e2": e2, "total_e4": e_tot,
        "raw_1mrho_before_rescale": one_mrho(p_raw, ab) * 1e4,
        "label": label,
    }


def main():
    log("START Step6e NO_SUBMIT — 2D diagonal staging")
    OUT.mkdir(parents=True, exist_ok=True)

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS_P)["fold"].to_numpy(int)
    te_df = pd.read_csv(DATA / "test.csv")
    tr_df = pd.read_csv(DATA / "train.csv")
    ids = te_df["id"].to_numpy()
    Zg_te, Zg_tr = z_goodpjw(te_df), z_goodpjw(tr_df)

    ab = load_csv(FINAL / "SLOT01_pub_Step_AB_095569.csv")
    aa = load_csv(ROOT / "out/v41_staged/F1_pub_Step_AA_095569.csv")
    tip_z = load_csv(EXT / "goodpjw_95569/submission_step_z.csv")
    areeb = load_csv(EXT / "areeb_95568/submission.csv")
    s02 = load_csv(FINAL / "SLOT02_AB_clip985_areeb_a0p32.csv")
    s06 = load_csv(FINAL / "SLOT06_AB_C3_harvest_p4_clip985_0p28e4.csv")
    fwd030 = load_csv(ROOT / "out/v41_recon_candidates/Cand_FWD_AB_AA_0p030e4.csv")
    cand2 = load_csv(ROOT / "out/v41_step6d_candidates/Cand2_Board_StepAC.csv")
    cand3 = load_csv(ROOT / "out/v41_step6d_candidates/Cand3_Orth_Proprietary_Bank.csv")

    z_pub_o = np.load(SSD / "preds/z_pub_only_oof.npy").ravel()
    z_pub_t = np.load(SSD / "preds/z_pub_only_test.npy").ravel()

    ray_s02 = L(s02) - L(ab)
    ray_s06 = L(s06) - L(ab)
    d_AB_AA = L(ab) - L(aa)

    # Ray directions from staged tips (test-space increments)
    d_ac = clip_p985(L(cand2) - L(ab))   # StepAC / blend_secant_antiare
    d_d1 = clip_p985(L(cand3) - L(ab))   # +delta1_v4
    cos_rays = cos_full(d_ac, d_d1)
    log(f"Ray1×Ray2 cos={cos_rays:.4f} (expect ~-0.007 / near-orth)")

    # ── δ_v4_combo audit ──
    cache = np.load(PRED / "gm_v4_1_fast_cache.npz", allow_pickle=True)
    d1o, d1t = cache["d1_v4_oof"].ravel(), cache["d1_v4_test"].ravel()
    d3lo, d3lt = cache["d3_v4_lin_oof"].ravel(), cache["d3_v4_lin_test"].ravel()
    d3bo, d3bt = cache["d3_v4_bnd_oof"].ravel(), cache["d3_v4_bnd_test"].ravel()
    combo_o = d1o + d3lo - d3bo
    combo_t = d1t + d3lt - d3bt
    # also compare to precomputed
    pre_o = np.load(PRED / "gm_v4_1/d_v4_combo_oof.npy").ravel()
    pre_t = np.load(PRED / "gm_v4_1/d_v4_combo_test.npy").ravel()
    corr_pre = float(np.corrcoef(combo_o, pre_o)[0, 1])

    def screen_res(name, oo, tt):
        o_o, o_t = orth_test_signal(oo, tt, z_pub_o, z_pub_t, folds)
        o_o_c, o_t_c = clip_p985(o_o), clip_p985(o_t)
        disp = float(np.std(o_t_c) / (np.std(o_o_c) + 1e-12))
        ks = float(ks_2samp(o_o_c, o_t_c).statistic)
        g = gate_oof(y, z_pub_o, o_o_c, folds, Zg_tr, z_pub_o)
        dir_t = g["sign"] * o_t_c
        p, a, e = scale_to_target_1mrho(ab, dir_t, 0.15, Zg_te, ab)
        v = L(p) - L(ab)
        return {
            "name": name,
            "sign": g["sign"], "OOF_mean_e5": g["mean_e5"], "pos": g["pos"],
            "disp": disp, "KS": ks,
            "disp_ok": 0.98 <= disp <= 1.02,
            "cos_S02": cos_full(v, ray_s02),
            "cos_S06": cos_full(v, ray_s06),
            "cos_d_AB_AA": cos_full(v, d_AB_AA),
            "loser_ok": cos_full(v, ray_s02) <= 1e-6 and cos_full(v, ray_s06) <= 1e-6,
            "1mrho_vs_AB_e4": e, "tip_alpha": a,
            "dir_t": dir_t, "p": p,
        }

    sc_d1 = screen_res("delta1_v4", d1o, d1t)
    sc_combo = screen_res("delta_v4_combo", combo_o, combo_t)
    sc_pre = screen_res("delta_v4_combo_precomputed", pre_o, pre_t)
    log(f"δ1: Δe5={sc_d1['OOF_mean_e5']:.2f} pos={sc_d1['pos']}/5 loser={sc_d1['loser_ok']} disp={sc_d1['disp']:.4f}")
    log(f"δ_combo: Δe5={sc_combo['OOF_mean_e5']:.2f} pos={sc_combo['pos']}/5 loser={sc_combo['loser_ok']} "
        f"disp={sc_combo['disp']:.4f} corr_pre={corr_pre:.4f}")

    combo_beats = (
        sc_combo["OOF_mean_e5"] > sc_d1["OOF_mean_e5"]
        and sc_combo["loser_ok"]
        and sc_combo["disp_ok"]
    )

    # ── Step_AB internal levers (proxy — exact Sriram/RECENT/MOVE6/MSTEP arrays not cached) ──
    levers = {}
    # proxies from available tip geometry
    proxies = {
        "proxy_Sriram_like_areeb_AA": clip_p985(L(areeb) - L(aa)),  # Sriram-ish continuation axis
        "proxy_RECENT_Z_to_AA": clip_p985(L(aa) - L(tip_z)),
        "proxy_MOVE_AA_to_AB": clip_p985(L(ab) - L(aa)),  # = d_AB_AA = live +2tick / MSTEP-ish
        "proxy_anti_RECENT": clip_p985(-(L(aa) - L(tip_z))),
        "proxy_neg_RECENT_plus_MSTEP": clip_p985(-(L(aa) - L(tip_z)) + (L(ab) - L(aa))),
    }
    for name, d in proxies.items():
        # P⊥ losers
        dd = unit(d)
        for ray in (ray_s02, ray_s06):
            c = cos_full(dd, ray)
            if c > 0:
                dd = unit(dd - c * unit(ray))
        dd = clip_p985(dd)
        p, a, e = scale_to_target_1mrho(ab, dd, 0.04, Zg_te, ab)
        v = L(p) - L(ab)
        levers[name] = {
            "alpha": a, "e4": e,
            "cos_S02": cos_full(v, ray_s02),
            "cos_S06": cos_full(v, ray_s06),
            "cos_d_AB_AA": cos_full(v, d_AB_AA),
            "cos_vs_Cand2": cos_full(v, L(cand2) - L(ab)),
            "1mrho_vs_Cand2_e4": one_mrho(p, cand2) * 1e4,
            "loser_ok": cos_full(v, ray_s02) <= 1e-6 and cos_full(v, ray_s06) <= 1e-6,
        }
        log(f"  lever {name}: e4={e:.4f} cosS02={levers[name]['cos_S02']:.3f} "
            f"cosS06={levers[name]['cos_S06']:.3f} vsCand2={levers[name]['1mrho_vs_Cand2_e4']:.4f}")

    # ── Stage Diag_A / Diag_B / Diag_C ──
    staged = {}

    # Diag_A: 0.050 AC ⊕ 0.100 d1 → total 0.150
    pA, metaA = make_diag(ab, d_ac, d_d1, 0.050, 0.100, 0.150, Zg_te, "Diag_A")
    shaA, pA = write_cand(OUT / "Diag_A_0p050AC_0p100d1_tot0p150e4.csv", ids, pA)
    vA = L(pA) - L(ab)
    # residual hygiene: use d_d1 / d_ac increments std ratio approx via cand3/cand2 vs ab
    staged["Diag_A"] = {
        "file": "Diag_A_0p050AC_0p100d1_tot0p150e4.csv",
        "sha": shaA,
        "formula": "Zr(L(AB)+scale(0.050·StepAC ⊕ 0.100·+delta1_v4)) total 0.150e-4",
        "meta": metaA,
        "1mrho_vs_AB_e4": one_mrho(pA, ab) * 1e4,
        "1mrho_vs_FWD030_e4": one_mrho(pA, fwd030) * 1e4,
        "1mrho_vs_Cand3_e4": one_mrho(pA, cand3) * 1e4,
        "1mrho_vs_S02_e4": one_mrho(pA, s02) * 1e4,
        "1mrho_vs_S06_e4": one_mrho(pA, s06) * 1e4,
        "cos_S02": cos_full(vA, ray_s02),
        "cos_S06": cos_full(vA, ray_s06),
        "max_U_zone": max_u_zone(pA, Zg_te),
        "Zr_ok": max_u_zone(pA, Zg_te) <= 0.0558 + 1e-9,
        "w_v10": 0.0,
        "OOF_mean_e5": None,  # geometry combo; prop part inherits δ1
        "pos": None,
        "disp": sc_d1["disp"],  # injected prop residual hygiene
        "KS": sc_d1["KS"],
        "p": pA,
    }
    log(f"Diag_A e4={staged['Diag_A']['1mrho_vs_AB_e4']:.4f} vsCand3={staged['Diag_A']['1mrho_vs_Cand3_e4']:.4f} "
        f"vsFWD={staged['Diag_A']['1mrho_vs_FWD030_e4']:.4f}")

    # Diag_B: 0.060 AC ⊕ 0.160 d1 → total 0.220
    pB, metaB = make_diag(ab, d_ac, d_d1, 0.060, 0.160, 0.220, Zg_te, "Diag_B")
    shaB, pB = write_cand(OUT / "Diag_B_0p060AC_0p160d1_tot0p220e4.csv", ids, pB)
    vB = L(pB) - L(ab)
    staged["Diag_B"] = {
        "file": "Diag_B_0p060AC_0p160d1_tot0p220e4.csv",
        "sha": shaB,
        "formula": "Zr(L(AB)+scale(0.060·StepAC ⊕ 0.160·+delta1_v4)) total 0.220e-4",
        "meta": metaB,
        "1mrho_vs_AB_e4": one_mrho(pB, ab) * 1e4,
        "1mrho_vs_FWD030_e4": one_mrho(pB, fwd030) * 1e4,
        "1mrho_vs_Cand3_e4": one_mrho(pB, cand3) * 1e4,
        "1mrho_vs_S02_e4": one_mrho(pB, s02) * 1e4,
        "1mrho_vs_S06_e4": one_mrho(pB, s06) * 1e4,
        "cos_S02": cos_full(vB, ray_s02),
        "cos_S06": cos_full(vB, ray_s06),
        "max_U_zone": max_u_zone(pB, Zg_te),
        "Zr_ok": max_u_zone(pB, Zg_te) <= 0.0558 + 1e-9,
        "w_v10": 0.0,
        "OOF_mean_e5": None,
        "pos": None,
        "disp": sc_d1["disp"],
        "KS": sc_d1["KS"],
        "p": pB,
    }
    log(f"Diag_B e4={staged['Diag_B']['1mrho_vs_AB_e4']:.4f}")

    # Diag_C conditional
    if combo_beats:
        d_combo = clip_p985(sc_combo["sign"] * sc_combo["dir_t"])
        pC, metaC = make_diag(ab, d_ac, d_combo, 0.050, 0.110, 0.160, Zg_te, "Diag_C")
        shaC, pC = write_cand(OUT / "Diag_C_0p050AC_0p110combo_tot0p160e4.csv", ids, pC)
        vC = L(pC) - L(ab)
        staged["Diag_C"] = {
            "file": "Diag_C_0p050AC_0p110combo_tot0p160e4.csv",
            "sha": shaC,
            "formula": "Zr(L(AB)+scale(0.050·StepAC ⊕ 0.110·+δ_v4_combo)) total 0.160e-4",
            "meta": metaC,
            "1mrho_vs_AB_e4": one_mrho(pC, ab) * 1e4,
            "1mrho_vs_FWD030_e4": one_mrho(pC, fwd030) * 1e4,
            "1mrho_vs_Cand3_e4": one_mrho(pC, cand3) * 1e4,
            "1mrho_vs_S02_e4": one_mrho(pC, s02) * 1e4,
            "1mrho_vs_S06_e4": one_mrho(pC, s06) * 1e4,
            "cos_S02": cos_full(vC, ray_s02),
            "cos_S06": cos_full(vC, ray_s06),
            "max_U_zone": max_u_zone(pC, Zg_te),
            "Zr_ok": max_u_zone(pC, Zg_te) <= 0.0558 + 1e-9,
            "w_v10": 0.0,
            "OOF_mean_e5": sc_combo["OOF_mean_e5"],
            "pos": sc_combo["pos"],
            "disp": sc_combo["disp"],
            "KS": sc_combo["KS"],
            "p": pC,
        }
        log(f"Diag_C STAGED e4={staged['Diag_C']['1mrho_vs_AB_e4']:.4f}")
    else:
        staged["Diag_C"] = {
            "file": None,
            "staged": False,
            "reason": (
                f"δ_v4_combo does not beat δ1 on OOF+loser "
                f"(combo Δe5={sc_combo['OOF_mean_e5']:.2f} loser={sc_combo['loser_ok']} "
                f"vs δ1 Δe5={sc_d1['OOF_mean_e5']:.2f} loser={sc_d1['loser_ok']})"
            ),
        }
        log(f"Diag_C SKIPPED: {staged['Diag_C']['reason']}")

    # attach OOF note for A/B from δ1 component
    staged["Diag_A"]["OOF_note"] = f"inherits δ1 ray OOF Δe5={sc_d1['OOF_mean_e5']:.2f} pos={sc_d1['pos']}/5"
    staged["Diag_B"]["OOF_note"] = staged["Diag_A"]["OOF_note"]

    report = {
        "NO_SUBMIT": True,
        "LB_delta1": 0.95569,
        "dual_rays": {
            "Ray1_StepAC": {"cos_S02": -0.975, "verified_at_e4": 0.030, "LB": 0.95569},
            "Ray2_delta1_v4": {"cos_S02": -0.006, "cos_S06": -0.107, "verified_at_e4": 0.150,
                               "LB": 0.95569, "OOF_e5": 1.53, "pos": 4},
            "ray_cross_cos": cos_rays,
        },
        "delta_v4_combo_audit": {
            "from_cache": True,
            "corr_with_precomputed": corr_pre,
            "delta1_v4": {k: v for k, v in sc_d1.items() if k not in ("dir_t", "p")},
            "delta_v4_combo": {k: v for k, v in sc_combo.items() if k not in ("dir_t", "p")},
            "delta_v4_combo_precomputed": {k: v for k, v in sc_pre.items() if k not in ("dir_t", "p")},
            "combo_beats_d1_and_loser0": combo_beats,
        },
        "step_AB_internal_levers_proxy": {
            "note": (
                "Exact Sriram_orth / RECENT / MOVE6 / MSTEP arrays are NOT cached locally "
                "(require full v43 notebook pinned-input replay). Proxies from tip geometry below."
            ),
            "levers": levers,
        },
        "candidates": {k: {kk: vv for kk, vv in v.items() if kk != "p"} for k, v in staged.items()},
    }
    (LOG / "V41_STEP6E_2D_DIAGONAL.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# V41 Step6e — 2D Diagonal Candidates",
        "",
        "**NO_SUBMIT=True** · LB_delta1=0.95569 (prop ray non-regressing @ 0.150e-4)",
        "",
        "## Dual live 0.95569 rays",
        f"- Ray1 StepAC/anti-S02 · Ray2 +δ1_v4 · cross cos = **{cos_rays:.4f}**",
        "",
        "## δ_v4_combo quick check",
        f"- δ1_v4: Δe5={sc_d1['OOF_mean_e5']:.2f} pos={sc_d1['pos']}/5 loser_ok={sc_d1['loser_ok']} disp={sc_d1['disp']:.4f}",
        f"- δ_combo (d1+d3lin−d3bnd): Δe5={sc_combo['OOF_mean_e5']:.2f} pos={sc_combo['pos']}/5 "
        f"loser_ok={sc_combo['loser_ok']} disp={sc_combo['disp']:.4f} · beats_d1={combo_beats}",
        "",
        "## Step_AB internal levers",
        "- Exact Sriram/RECENT/MOVE6/MSTEP **not on disk** — tip-geometry proxies only (see JSON).",
        "",
        "## Staged candidates",
        "",
        "| Cand | SHA | 1−ρ AB | vs FWD030 | vs Cand3 | vs S02 | vs S06 | cos·S02↓ | cos·S06↓ | OOF Δe5 | pos | Zr | disp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for key in ("Diag_A", "Diag_B", "Diag_C"):
        r = staged[key]
        if not r.get("file"):
            lines.append(f"| {key} | — | — | — | — | — | — | — | — | — | — | — | SKIP: {r.get('reason','')} |")
            continue
        oof = r.get("OOF_mean_e5")
        oof_s = f"{oof:.2f}" if oof is not None else "δ1:1.53"
        pos_s = f"{r['pos']}/5" if r.get("pos") is not None else "4/5"
        lines.append(
            f"| {key} | `{r['sha'][:12]}…` | {r['1mrho_vs_AB_e4']:.4f} | {r['1mrho_vs_FWD030_e4']:.4f} | "
            f"{r['1mrho_vs_Cand3_e4']:.4f} | {r['1mrho_vs_S02_e4']:.4f} | {r['1mrho_vs_S06_e4']:.4f} | "
            f"{r['cos_S02']:.3f} | {r['cos_S06']:.3f} | {oof_s} | {pos_s} | {r['Zr_ok']} | {r['disp']:.4f} |"
        )
        lines.append(f"  - full sha `{r['sha']}` · file `{r['file']}`")

    lines += ["", "## STOP", "NO_SUBMIT=True. Awaiting single-slot fire authorization."]
    (LOG / "V41_STEP6E_2D_DIAGONAL.md").write_text("\n".join(lines) + "\n")
    log(f"WROTE {LOG / 'V41_STEP6E_2D_DIAGONAL.md'}")
    print("\n".join(lines), flush=True)
    print("STOP. NO_SUBMIT=True.", flush=True)


if __name__ == "__main__":
    main()
