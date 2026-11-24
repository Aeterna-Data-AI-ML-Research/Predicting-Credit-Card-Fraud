#!/usr/bin/env python3
"""V41 Step6d — goodpjw v43 Board-Model inspection + Step_AC candidate audit.

STRICT NO_SUBMIT=True. Does not submit.
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
from sklearn.linear_model import Ridge
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
STAGED = ROOT / "out/v41_staged"
OUT = ROOT / "out/v41_step6d_candidates"
PRED = ROOT / "preds"
DATA = ROOT / "data" if (ROOT / "data/test.csv").exists() else SSD / "data"
FOLDS_P = SSD / "folds/folds.csv" if (SSD / "folds/folds.csv").exists() else ROOT / "folds/folds.csv"
NB = ROOT / "external/v41_gpw_v43_nb/s6e11-board-model-realmlp-lb-0-95569.ipynb"

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
    for _ in range(40):
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


def lofo_orth(signal, base, folds, ridge=50.0):
    s = unit(L_fold(signal, folds))
    b = unit(L_fold(base, folds)).reshape(-1, 1)
    out = np.zeros_like(s)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        beta = np.linalg.solve(b[tr].T @ b[tr] + ridge * np.eye(1), b[tr].T @ s[tr])
        out[te] = unit(s[te] - (b[te] @ beta).ravel())
    return unit(out)


def orth_test(signal_t, base_t, signal_o, base_o, folds, ridge=50.0):
    """LOFO orth OOF + average test residual using fold betas."""
    so = unit(L_fold(signal_o, folds))
    bo = unit(L_fold(base_o, folds)).reshape(-1, 1)
    st = unit(L(signal_t))
    bt = unit(L(base_t)).reshape(-1, 1)
    oof = np.zeros(len(so))
    te = np.zeros(len(st))
    for k in range(5):
        tr, va = folds != k, folds == k
        beta = np.linalg.solve(bo[tr].T @ bo[tr] + ridge * np.eye(1), bo[tr].T @ so[tr])
        oof[va] = unit(so[va] - (bo[va] @ beta).ravel())
        te += unit(st - (bt @ beta).ravel()) / 5
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
                best = {"mean_e5": mean, "pos": pos, "alpha": float(a), "sign": sgn,
                        "folds": folds_d}
    return best


def feats(vec, ab, d_AB_AA, ray_s02, ray_s06):
    v = L(vec) - L(ab)
    return np.array([
        one_mrho(vec, ab) * 1e4,
        cos_full(v, d_AB_AA),
        cos_full(v, ray_s02),
        cos_full(v, ray_s06),
        np.linalg.norm(unit(v)),
    ], dtype=float)


def main():
    log("START Step6d NO_SUBMIT — board-model v43 inspect + Step_AC audit")
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    # ── (a) Board-model notebook facts ──
    nb_ok = NB.exists()
    board_findings = {
        "notebook": str(NB),
        "pulled": nb_ok,
        "board_model_shipped": False,
        "quote": (
            "The board model is not shipped. It is fitted on 151–233 scored files whose "
            "public score is stated; forecasts and nested CV are reported as measured. "
            "Near the top it forecasts a left-out file to about 0.7 points."
        ),
        "forecasts_reported": {
            "step_AA_v41": {"forecast_pts": 1.05, "LB": 0.95567},
            "step_AB_v43": {"forecast_pts": 0.58, "LB": 0.95569},
        },
        "step_AB_formula": (
            "Zr( Sriram_orthogonal_0.95568 - 0.4*RECENT - 0.15*MOVE6_0.95557 + 0.15*MSTEP )"
        ),
        "step_AA_formula": (
            "Zr( Malyshev_v43 + 0.75*clip99(MOVE_v42→v43) + 1.5*EARLIER )"
        ),
        "pinned_inputs_public": True,
        "pinned_datasets": [
            "asteriospanoras/s6e11-public-board-files (Apache 2.0) — pinned scored CSVs",
            "goodpjw2008/s6e11-oof-bank — 70 OOF members",
            "amanatar rank1_q17_streams.npz bundle",
            "busyaprime TabM / 33 own members",
            "yekenot RealMLP pytabkit + pytorch OOF",
            "malyshevdanil / areebbb / sriram scored tip CSVs (pinned by SHA)",
        ],
        "how_forecast_works": (
            "Offline board model (NOT in notebook) is fitted on public scored files; "
            "for each candidate step it outputs a forecast in 'points' (1 pt = 1e-5 LB). "
            "Notebook only publishes the measured forecasts (+1.05 AA, +0.58 AB) and "
            "exact reconstructible step recipes on public pinned inputs. "
            "We cannot literally retrain goodpjw's board model without that offline fit; "
            "below we fit a LOCAL SURROGATE on our live tip telemetry."
        ),
    }
    log(f"board_model_shipped=False; forecasts AA=+1.05 AB=+0.58; inputs public=True")

    # ── load tips / data ──
    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS_P)["fold"].to_numpy(int)
    te_df = pd.read_csv(DATA / "test.csv")
    tr_df = pd.read_csv(DATA / "train.csv")
    ids = te_df["id"].to_numpy()
    Zg_te, Zg_tr = z_goodpjw(te_df), z_goodpjw(tr_df)

    def load_is_fraud(path: Path) -> np.ndarray:
        """Read CSV resiliently (pics volume intermittently fails open)."""
        import shutil, subprocess
        path = Path(path)
        tmp = Path("/tmp") / f"_s6e11_{path.name}"
        if tmp.exists() and tmp.stat().st_size > 1_000_000:
            return pd.read_csv(tmp)["is_fraud"].to_numpy(float)
        try:
            return pd.read_csv(path)["is_fraud"].to_numpy(float)
        except OSError:
            subprocess.check_call(["cp", str(path), str(tmp)])
            return pd.read_csv(tmp)["is_fraud"].to_numpy(float)

    ab = load_is_fraud(FINAL / "SLOT01_pub_Step_AB_095569.csv")
    aa = load_is_fraud(STAGED / "F1_pub_Step_AA_095569.csv")
    tip_z = load_is_fraud(EXT / "goodpjw_95569/submission_step_z.csv")
    areeb = load_is_fraud(EXT / "areeb_95568/submission.csv")
    quarter = load_is_fraud(STAGED / "quarter_our_095567.csv")
    # prefer pre-staged /tmp copy (pics open can flake)
    mal_path = Path("/tmp/malyshev_v42.csv")
    if not mal_path.exists():
        mal_path = EXT / "pinned_csv/malyshev_v42_from_npz.csv"
    mal42 = load_is_fraud(mal_path)
    s02 = load_is_fraud(FINAL / "SLOT02_AB_clip985_areeb_a0p32.csv")
    s06 = load_is_fraud(FINAL / "SLOT06_AB_C3_harvest_p4_clip985_0p28e4.csv")
    fwd030 = load_is_fraud(ROOT / "out/v41_recon_candidates/Cand_FWD_AB_AA_0p030e4.csv")

    z_pub_o = np.load(SSD / "preds/z_pub_only_oof.npy").ravel()
    z_pub_t = np.load(SSD / "preds/z_pub_only_test.npy").ravel()

    d_AB_AA = L(ab) - L(aa)
    ray_s02 = L(s02) - L(ab)
    ray_s06 = L(s06) - L(ab)

    # B_pub tip bank for multi-orth (test-space)
    B_pub_tips = {
        "Step_Z": tip_z, "malyshev_v42": mal42, "quarter": quarter,
        "Step_AA": aa, "areeb": areeb, "Step_AB": ab,
    }

    # ── local board surrogate from live + ladder telemetry ──
    # points: (vec, LB). Use tip ladder + our 4 live points.
    telemetry = [
        (tip_z, 0.95562, "Step_Z"),
        (aa, 0.95567, "Step_AA"),
        (areeb, 0.95568, "areeb"),
        (ab, 0.95569, "Step_AB"),
        (fwd030, 0.95569, "FWD_AB_AA_0p030"),
        (s02, 0.95568, "SLOT02"),
        (s06, 0.95568, "SLOT06"),
    ]
    # Features relative to AB for non-AB; for AB itself use zeros
    X_list, y_list, names_tel = [], [], []
    for vec, lb, nm in telemetry:
        if nm == "Step_AB":
            X_list.append([0.0, 1.0, 0.0, 0.0, 0.0])  # e4=0, cos_d=1 sentinel, cos losers 0
        else:
            X_list.append(feats(vec, ab, d_AB_AA, ray_s02, ray_s06).tolist())
        y_list.append(lb)
        names_tel.append(nm)
    X = np.asarray(X_list, float)
    yb = np.asarray(y_list, float)
    # Ridge on (LB - 0.95569) in points
    y_pts = (yb - 0.95569) * 1e5
    # leave-one-out style: fit on all (tiny n); report in-sample + LOO
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(X, y_pts)
    pred_pts = ridge.predict(X)
    loo = []
    for i in range(len(X)):
        m = np.ones(len(X), bool); m[i] = False
        if m.sum() < 3:
            loo.append(float("nan"))
            continue
        rr = Ridge(alpha=1.0, fit_intercept=True)
        rr.fit(X[m], y_pts[m])
        loo.append(float(rr.predict(X[i:i+1])[0]))
    surrogate = {
        "n": len(X),
        "telemetry": names_tel,
        "coef": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
        "in_sample_pts": pred_pts.tolist(),
        "loo_pts": loo,
        "note": "LOCAL SURROGATE — not goodpjw board model. Fit on tip geometry features → Δpts vs 0.95569.",
    }

    def forecast_pts(vec):
        if one_mrho(vec, ab) < 1e-12:
            x = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
        else:
            x = feats(vec, ab, d_AB_AA, ray_s02, ray_s06).reshape(1, -1)
        return float(np.asarray(ridge.predict(x)).ravel()[0])

    log(f"local_surrogate fit n={len(X)} intercept={ridge.intercept_:.3f} coef={np.round(ridge.coef_,3)}")

    # ── Cand1: FWD_AB_AA @ 0.10e-4 ──
    d_sec = clip_p985(L(ab) - L(aa))
    p1, a1, e1 = scale_to_target_1mrho(ab, d_sec, 0.100, Zg_te, ab)
    sha1, p1 = write_cand(OUT / "Cand1_FWD_AB_AA_0p10e4.csv", ids, p1)
    v1 = L(p1) - L(ab)
    cand1 = {
        "file": "Cand1_FWD_AB_AA_0p10e4.csv", "sha": sha1, "alpha": a1,
        "1mrho_vs_AB_e4": e1,
        "1mrho_vs_FWD030_e4": one_mrho(p1, fwd030) * 1e4,
        "1mrho_vs_S02_e4": one_mrho(p1, s02) * 1e4,
        "1mrho_vs_S06_e4": one_mrho(p1, s06) * 1e4,
        "max_U_zone": max_u_zone(p1, Zg_te),
        "Zr_ok": max_u_zone(p1, Zg_te) <= 0.0558 + 1e-9,
        "w_v10": 0.0,
        "cos_d_AB_AA": cos_full(v1, d_AB_AA),
        "cos_S02": cos_full(v1, ray_s02),
        "cos_S06": cos_full(v1, ray_s06),
        "loser_ok": cos_full(v1, ray_s02) <= 1e-6 and cos_full(v1, ray_s06) <= 1e-6,
        "board_forecast_pts_local": forecast_pts(p1),
        "formula": "Zr(L(AB)+β·clip_p985(L(AB)-L(AA))) @ 0.100e-4",
    }
    log(f"Cand1 e4={e1:.4f} α={a1:.4f} fc={cand1['board_forecast_pts_local']:.2f} loser_ok={cand1['loser_ok']}")

    # ── Cand2: Board-like Step_AC search in [0.06, 0.18]e-4 ──
    # Directions: (i) AA→AB secant, (ii) AB−Z, (iii) AB−mal42, (iv) AB − (areeb−AB) = anti-areeb
    # (v) AB − (S06−AB) = anti-C3, (vi) blend of (i)+(iv) with loser projection
    dirs = {
        "secant_AB_AA": clip_p985(L(ab) - L(aa)),
        "AB_minus_Z": clip_p985(L(ab) - L(tip_z)),
        "AB_minus_mal42": clip_p985(L(ab) - L(mal42)),
        "anti_areeb": clip_p985(-(L(areeb) - L(ab))),
        "anti_S06": clip_p985(-(L(s06) - L(ab))),
    }
    # project out positive components of loser rays
    def P_perp_losers(d):
        d = unit(d)
        for ray in (ray_s02, ray_s06):
            c = cos_full(d, ray)
            if c > 0:
                d = unit(d - c * unit(ray))
        return clip_p985(d)

    dirs["secant_Pperp"] = P_perp_losers(dirs["secant_AB_AA"])
    dirs["blend_secant_antiare"] = P_perp_losers(
        unit(dirs["secant_AB_AA"]) + 0.5 * unit(dirs["anti_areeb"])
    )

    cand2_best = None
    cand2_grid = []
    for dname, direction in dirs.items():
        for tgt in (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18):
            p, a, e = scale_to_target_1mrho(ab, direction, tgt, Zg_te, ab)
            if not (0.06 - 0.005 <= e <= 0.18 + 0.005):
                continue
            v = L(p) - L(ab)
            c02, c06 = cos_full(v, ray_s02), cos_full(v, ray_s06)
            if c02 > 1e-6 or c06 > 1e-6:
                continue
            fc = forecast_pts(p)
            row = {
                "dir": dname, "target": tgt, "alpha": a, "e4": e,
                "cos_d_AB_AA": cos_full(v, d_AB_AA),
                "cos_S02": c02, "cos_S06": c06,
                "fc_pts": fc, "max_U_zone": max_u_zone(p, Zg_te),
                "p": p,
            }
            cand2_grid.append({k: v for k, v in row.items() if k != "p"})
            if cand2_best is None or fc > cand2_best["fc_pts"]:
                cand2_best = row
    if cand2_best is None:
        # fallback: secant @ 0.10 even if loser gate soft
        p, a, e = scale_to_target_1mrho(ab, dirs["secant_Pperp"], 0.10, Zg_te, ab)
        v = L(p) - L(ab)
        cand2_best = {
            "dir": "secant_Pperp_fallback", "target": 0.10, "alpha": a, "e4": e,
            "cos_d_AB_AA": cos_full(v, d_AB_AA),
            "cos_S02": cos_full(v, ray_s02), "cos_S06": cos_full(v, ray_s06),
            "fc_pts": forecast_pts(p), "max_U_zone": max_u_zone(p, Zg_te), "p": p,
        }
    sha2, p2 = write_cand(OUT / "Cand2_Board_StepAC.csv", ids, cand2_best["p"])
    cand2 = {
        "file": "Cand2_Board_StepAC.csv", "sha": sha2,
        "dir": cand2_best["dir"], "alpha": cand2_best["alpha"],
        "1mrho_vs_AB_e4": cand2_best["e4"],
        "1mrho_vs_FWD030_e4": one_mrho(p2, fwd030) * 1e4,
        "1mrho_vs_S02_e4": one_mrho(p2, s02) * 1e4,
        "1mrho_vs_S06_e4": one_mrho(p2, s06) * 1e4,
        "max_U_zone": cand2_best["max_U_zone"],
        "Zr_ok": cand2_best["max_U_zone"] <= 0.0558 + 1e-9,
        "w_v10": 0.0,
        "cos_d_AB_AA": cand2_best["cos_d_AB_AA"],
        "cos_S02": cand2_best["cos_S02"],
        "cos_S06": cand2_best["cos_S06"],
        "loser_ok": cand2_best["cos_S02"] <= 1e-6 and cand2_best["cos_S06"] <= 1e-6,
        "board_forecast_pts_local": cand2_best["fc_pts"],
        "grid_n_pass": len(cand2_grid),
        "formula": f"Zr(L(AB)+α·{cand2_best['dir']}) @ {cand2_best['e4']:.3f}e-4; loser P⊥",
    }
    log(f"Cand2 dir={cand2['dir']} e4={cand2['1mrho_vs_AB_e4']:.4f} fc={cand2['board_forecast_pts_local']:.2f} pass_grid={len(cand2_grid)}")

    # ── Cand3 screen: proprietary residuals ──
    prop_specs = []
    # FamilyE L7 swap
    lo = np.load(PRED / "l7_s2026_5f/l7_s2026_oof_5fold_raw.npy")
    lt = np.load(PRED / "l7_s2026_5f/l7_s2026_test_5fold_raw.npy")
    # shapes: oof (N, seeds, 2) or similar; test (5, N, seeds, 2)
    if lo.ndim == 3:
        l7o = (lo[:, :, 1] - lo[:, :, 0]).mean(1)
    else:
        l7o = lo.ravel()
    if lt.ndim == 4:
        l7t = ((lt[..., 1] - lt[..., 0]).mean(2)).mean(0)
    else:
        l7t = lt.ravel()
    lm7o = np.load(ROOT / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_5fold.npy").ravel()
    lm7t = np.load(ROOT / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/lm7_test_5fold.npy").ravel()
    famE_o = (L_fold(l7o, folds) - L_fold(lm7o, folds))
    famE_t = (L(l7t) - L(lm7t))
    w_o = np.clip(U(l7o), 0.5, 1.0)
    w_t = np.clip(U(l7t), 0.5, 1.0)
    prop_specs.append(("FamilyE_L7_s2026_5f", famE_o * w_o, famE_t * w_t))

    # StageB / δ1_v4
    if (PRED / "gm_v4_polish/z_stageB_4fam_oof.npy").exists():
        prop_specs.append((
            "StageB_4fam",
            np.load(PRED / "gm_v4_polish/z_stageB_4fam_oof.npy").ravel(),
            np.load(PRED / "gm_v4_polish/z_stageB_4fam_test.npy").ravel(),
        ))
    if (PRED / "gm_v4_polish/d1_v4_oof.npy").exists():
        prop_specs.append((
            "delta1_v4",
            np.load(PRED / "gm_v4_polish/d1_v4_oof.npy").ravel(),
            np.load(PRED / "gm_v4_polish/d1_v4_test.npy").ravel(),
        ))
    # harvest_p4 + sign forced positive in name; gate will pick sign
    prop_specs.append((
        "C3_harvest_p4",
        np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_oof.npy").ravel(),
        np.load(PRED / "matrix_v2/harvest_p4_cuda_xgb_test.npy").ravel(),
    ))
    # StageB LGB / XGB as R2-like / Bo10-ish proxies if R1/R2 pure OOF missing
    if (PRED / "gm_v3_research/lgb_stageB_68_clean_oof.npy").exists():
        prop_specs.append((
            "R2_proxy_lgb_stageB68",
            np.load(PRED / "gm_v3_research/lgb_stageB_68_clean_oof.npy").ravel(),
            np.load(PRED / "gm_v3_research/lgb_stageB_68_clean_test.npy").ravel(),
        ))
    if (PRED / "gm_v3_research/xgb_stageB_68_bce_oof.npy").exists():
        prop_specs.append((
            "R1_proxy_xgb_stageB68",
            np.load(PRED / "gm_v3_research/xgb_stageB_68_bce_oof.npy").ravel(),
            np.load(PRED / "gm_v3_research/xgb_stageB_68_bce_test.npy").ravel(),
        ))

    missing = ["R1_pure_Bo10 (OOF npy not on disk — only historical CSVs)",
               "R2_LGB pure OOF (only LIVE CSV)",
               "StageB_10fam exact (using StageB_4fam / stageB68 proxies)"]

    prop_rows = []
    best_prop = None
    prop_preds = {}
    for name, oo, tt in prop_specs:
        oo, tt = np.asarray(oo, float).ravel(), np.asarray(tt, float).ravel()
        if oo.shape != (N_TR,) or tt.shape != (N_TE,):
            prop_rows.append({"name": name, "status": f"bad_shape o={oo.shape} t={tt.shape}"})
            continue
        # orth vs z_pub_only (=B_pub nest)
        o_o, o_t = orth_test_signal(oo, tt, z_pub_o, z_pub_t, folds)
        o_o_c, o_t_c = clip_p985(o_o), clip_p985(o_t)
        disp = float(np.std(o_t_c) / (np.std(o_o_c) + 1e-12))
        ks = float(ks_2samp(o_o_c, o_t_c).statistic)
        g = gate_oof(y, z_pub_o, o_o_c, folds, Zg_tr, z_pub_o)
        dir_t = g["sign"] * o_t_c
        # tip climb ~0.15e-4 for screening file
        p, a, e = scale_to_target_1mrho(ab, dir_t, 0.15, Zg_te, ab)
        v = L(p) - L(ab)
        row = {
            "name": name, "status": "ok",
            "sign": g["sign"], "OOF_mean_e5": g["mean_e5"], "pos": g["pos"],
            "alpha_oof": g["alpha"], "disp": disp, "KS": ks,
            "1mrho_vs_AB_e4": e, "tip_alpha": a,
            "cos_d_AB_AA": cos_full(v, d_AB_AA),
            "cos_S02": cos_full(v, ray_s02),
            "cos_S06": cos_full(v, ray_s06),
            "loser_ok": cos_full(v, ray_s02) <= 1e-6 and cos_full(v, ray_s06) <= 1e-6,
            "board_forecast_pts_local": forecast_pts(p),
            "max_U_zone": max_u_zone(p, Zg_te),
            "p": p,
        }
        # also cos vs B_pub tip moves
        tip_cos = {tn: cos_full(v, L(tv) - L(ab)) if tn != "Step_AB" else 0.0
                   for tn, tv in B_pub_tips.items()}
        row["cos_vs_B_pub_tips"] = tip_cos
        prop_preds[name] = p
        prop_rows.append({k: val for k, val in row.items() if k != "p"})
        log(f"  prop {name}: Δe5={g['mean_e5']:.2f} pos={g['pos']}/5 sign={g['sign']} "
            f"loser_ok={row['loser_ok']} fc={row['board_forecast_pts_local']:.2f}")
        if row["loser_ok"] and row["OOF_mean_e5"] > 0 and (
            best_prop is None or row["OOF_mean_e5"] > best_prop["OOF_mean_e5"]
        ):
            best_prop = {**{k: val for k, val in row.items() if k != "p"}, "p": p}

    # stage best prop as Cand3 if any; else best by OOF among okish
    if best_prop is None:
        okish = [r for r in prop_rows if r.get("status") == "ok" and r.get("OOF_mean_e5", -1) > 0]
        if okish:
            bp = max(okish, key=lambda r: r["OOF_mean_e5"])
            best_prop = {**bp, "p": prop_preds[bp["name"]]}
    if best_prop is not None and "p" in best_prop:
        sha3, p3 = write_cand(OUT / "Cand3_Orth_Proprietary_Bank.csv", ids, best_prop["p"])
        cand3 = {
            "file": "Cand3_Orth_Proprietary_Bank.csv", "sha": sha3,
            "source": best_prop["name"],
            "1mrho_vs_AB_e4": best_prop["1mrho_vs_AB_e4"],
            "1mrho_vs_FWD030_e4": one_mrho(p3, fwd030) * 1e4,
            "1mrho_vs_S02_e4": one_mrho(p3, s02) * 1e4,
            "1mrho_vs_S06_e4": one_mrho(p3, s06) * 1e4,
            "OOF_mean_e5": best_prop["OOF_mean_e5"], "pos": best_prop["pos"],
            "sign": best_prop["sign"],
            "cos_d_AB_AA": best_prop["cos_d_AB_AA"],
            "cos_S02": best_prop["cos_S02"], "cos_S06": best_prop["cos_S06"],
            "loser_ok": best_prop["loser_ok"],
            "board_forecast_pts_local": best_prop["board_forecast_pts_local"],
            "max_U_zone": best_prop["max_U_zone"],
            "Zr_ok": best_prop["max_U_zone"] <= 0.0558 + 1e-9,
            "w_v10": 0.0,
        }
    else:
        cand3 = {"file": None, "note": "no prop residual passed screen", "sha": None}
        p3 = None

    # pairwise matrix
    named = {
        "AB": ab, "FWD030": fwd030, "S02": s02, "S06": s06,
        "Cand1": p1, "Cand2": p2,
    }
    if p3 is not None:
        named["Cand3"] = p3
    names = list(named.keys())
    mat = [[one_mrho(named[a], named[b]) * 1e4 for b in names] for a in names]

    report = {
        "NO_SUBMIT": True,
        "LB_FWD_AA": 0.95569,
        "board_findings": board_findings,
        "local_board_surrogate": surrogate,
        "Cand1_FWD_AB_AA_0p10e4": cand1,
        "Cand2_Board_StepAC": cand2,
        "Cand2_grid_pass": cand2_grid[:20],
        "Cand3_Orth_Proprietary_Bank": cand3,
        "prop_screen": prop_rows,
        "missing_artifacts": missing,
        "pairwise_1mrho_e4": {"names": names, "matrix": mat},
    }
    (LOG / "V41_STEP6D_BOARD_STEPAC.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# V41 Step6d — Board-Model v43 Inspection & Step_AC Audit",
        "",
        "**NO_SUBMIT=True** · LB_FWD_AA=0.95569 (flat @ 0.030e-4 AA→AB)",
        "",
        "## (a) goodpjw Board Model",
        f"- Notebook pulled: `{NB.name}`",
        "- **Board model NOT shipped** (offline fit on 151–233 scored files; ~0.7pt LOO near tip).",
        "- Forecasts published: **AA +1.05 pts**, **AB +0.58 pts** (1 pt = 1e-5 LB).",
        "- Step_AB recipe (reconstructible): `Zr(Sriram_orth − 0.4·RECENT − 0.15·MOVE6 + 0.15·MSTEP)`.",
        "- Pinned inputs **public/downloadable**: `asteriospanoras/s6e11-public-board-files`, OOF bank, Aman streams, RealMLP OOF, Malyshev/Areeb/Sriram CSVs.",
        "- **Cannot literally refit goodpjw's board model** here; fitted LOCAL SURROGATE on 7 tip/live points.",
        "",
        "## Candidates",
        "",
        "| Cand | 1−ρ AB e4 | cos·d_AB_AA | cos·S02↓ | cos·S06↓ | local fc pts | loser≤0 | notes |",
        "|---|---:|---:|---:|---:|---:|:---:|---|",
        f"| Cand1_FWD_0p10 | {cand1['1mrho_vs_AB_e4']:.4f} | {cand1['cos_d_AB_AA']:.3f} | {cand1['cos_S02']:.3f} | {cand1['cos_S06']:.3f} | {cand1['board_forecast_pts_local']:.2f} | {cand1['loser_ok']} | {cand1['sha'][:12]}… |",
        f"| Cand2_StepAC | {cand2['1mrho_vs_AB_e4']:.4f} | {cand2['cos_d_AB_AA']:.3f} | {cand2['cos_S02']:.3f} | {cand2['cos_S06']:.3f} | {cand2['board_forecast_pts_local']:.2f} | {cand2['loser_ok']} | dir={cand2['dir']} |",
    ]
    if cand3.get("file"):
        lines.append(
            f"| Cand3_{cand3['source']} | {cand3['1mrho_vs_AB_e4']:.4f} | {cand3['cos_d_AB_AA']:.3f} | "
            f"{cand3['cos_S02']:.3f} | {cand3['cos_S06']:.3f} | {cand3['board_forecast_pts_local']:.2f} | "
            f"{cand3['loser_ok']} | OOF Δe5={cand3['OOF_mean_e5']:.2f} pos={cand3['pos']}/5 |"
        )
    else:
        lines.append("| Cand3 | — | — | — | — | — | — | no pass |")

    lines += ["", "### Prop screen (vs z_pub_only+Zr)", "",
              "| name | Δe5 | pos/5 | sign | cos·AA | cos·S02 | cos·S06 | loser≤0 | fc |",
              "|---|---:|---:|---:|---:|---:|---:|:---:|---:|"]
    for r in prop_rows:
        if r.get("status") != "ok":
            lines.append(f"| {r['name']} | — | — | — | — | — | — | — | {r.get('status')} |")
            continue
        lines.append(
            f"| {r['name']} | {r['OOF_mean_e5']:.2f} | {r['pos']}/5 | {r['sign']} | "
            f"{r['cos_d_AB_AA']:.3f} | {r['cos_S02']:.3f} | {r['cos_S06']:.3f} | "
            f"{r['loser_ok']} | {r['board_forecast_pts_local']:.2f} |"
        )

    lines += ["", "### Pairwise 1−ρ (×1e-4)", "",
              "| |" + "|".join(names) + "|",
              "|" + "|".join(["---"] * (len(names) + 1)) + "|"]
    for i, a in enumerate(names):
        lines.append("| " + a + " |" + "|".join(f"{mat[i][j]:.3f}" for j in range(len(names))) + "|")

    lines += [
        "",
        f"Missing exact artifacts: {missing}",
        "",
        "## STOP",
        "NO_SUBMIT=True. Awaiting single-slot authorization.",
    ]
    (LOG / "V41_STEP6D_BOARD_STEPAC.md").write_text("\n".join(lines) + "\n")
    log(f"WROTE {LOG / 'V41_STEP6D_BOARD_STEPAC.md'}")
    print("\n".join(lines[lines.index("## Candidates"):]), flush=True)
    print("STOP. NO_SUBMIT=True.", flush=True)


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
        oof[va] = (so[va] - (bo[va] @ beta).ravel())
        te += (st - (bt @ beta).ravel()) / 5
    return unit(oof), unit(te)


if __name__ == "__main__":
    main()
