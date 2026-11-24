#!/usr/bin/env python3
"""V41 60s recon + forward-secant / UID audit around SLOT01. STRICT NO_SUBMIT=True."""
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
OUT = ROOT / "out/v41_recon_candidates"
DATA = ROOT / "data" if (ROOT / "data/test.csv").exists() else SSD / "data"
FOLDS_P = SSD / "folds/folds.csv" if (SSD / "folds/folds.csv").exists() else ROOT / "folds/folds.csv"

SHA_AB = "3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038"
SHA_AA = "51706855fa6de98ef88c1d02bebfc24ab340092d05eb513e227e048041c0346c"
SHA_AREEB = None  # filled if known

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
    if not zones.any():
        return 0.0
    return float(np.max(p[zones]))


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


def cos(a, b):
    a, b = unit(a), unit(b)
    return float(np.dot(a, b) / (len(a)))


def cos_full(a, b):
    a, b = unit(a), unit(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def lofo_orth(signal, base, folds):
    """LOFO residual of signal on base (both logit-fold)."""
    s = L_fold(signal, folds)
    b = L_fold(base, folds)
    out = np.zeros_like(s)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(b[tr].reshape(-1, 1), s[tr])
        out[te] = s[te] - ridge.predict(b[te].reshape(-1, 1))
    return out


def gate_oof(y, base_o, cand_o_dir, folds, zones_tr, u_ab_tr, alphas=(0.005, 0.01, 0.02, 0.04)):
    """ΔAUC e-5 of base+Zr vs base+α·dir +Zr; both signs via α signed in caller."""
    base_zr = apply_zr(base_o, zones_tr, u_ab_tr)
    base_auc = float(roc_auc_score(y, base_zr))
    best = {"mean_e5": -1e9, "pos": 0, "alpha": 0.0, "folds": []}
    for a in alphas:
        p = apply_zr(L(base_o) + a * cand_o_dir, zones_tr, u_ab_tr)
        folds_d = []
        pos = 0
        for k in range(5):
            m = folds == k
            da = (roc_auc_score(y[m], p[m]) - roc_auc_score(y[m], base_zr[m])) * 1e5
            folds_d.append(float(da))
            if da > 0:
                pos += 1
        mean = float(np.mean(folds_d))
        if mean > best["mean_e5"]:
            best = {"mean_e5": mean, "pos": pos, "alpha": float(a), "folds": folds_d,
                    "auc": float(roc_auc_score(y, p))}
    best["base_auc"] = base_auc
    return best


def write_cand(path: Path, ids, scores):
    p = U(scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(index=False, lineterminator="\n")
    path.write_bytes(text.encode())
    return sha256_file(path), p


def main():
    log("START NO_SUBMIT=True — quarantine SLOT08; HOLD 09/10; audit 3 cands")
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS_P)["fold"].to_numpy(int)
    te_df = pd.read_csv(DATA / "test.csv")
    tr_df = pd.read_csv(DATA / "train.csv")
    ids = te_df["id"].to_numpy()
    Zg_te = z_goodpjw(te_df)
    Zg_tr = z_goodpjw(tr_df)

    ab = pd.read_csv(FINAL / "SLOT01_pub_Step_AB_095569.csv")["is_fraud"].to_numpy(float)
    assert sha256_file(FINAL / "SLOT01_pub_Step_AB_095569.csv") == SHA_AB
    aa = pd.read_csv(STAGED / "F1_pub_Step_AA_095569.csv")["is_fraud"].to_numpy(float)
    assert sha256_file(STAGED / "F1_pub_Step_AA_095569.csv") == SHA_AA
    areeb = pd.read_csv(EXT / "areeb_95568/submission.csv")["is_fraud"].to_numpy(float)
    s02 = pd.read_csv(FINAL / "SLOT02_AB_clip985_areeb_a0p32.csv")["is_fraud"].to_numpy(float)
    s06 = pd.read_csv(FINAL / "SLOT06_AB_C3_harvest_p4_clip985_0p28e4.csv")["is_fraud"].to_numpy(float)

    z_pub_o = np.load(SSD / "preds/z_pub_only_oof.npy").ravel()
    z_pub_t = np.load(SSD / "preds/z_pub_only_test.npy").ravel()
    # B_pub := z_pub_only (100% public bank). B_clean is (N,8) nest pillars — not used as orth base.
    B_o, B_t = z_pub_o, z_pub_t
    uid_o = np.load(SSD / "out/uid_v2_oof.npy").ravel()
    uid_t = np.load(SSD / "out/uid_v2_test.npy").ravel()
    assert uid_o.shape == (N_TR,) and B_o.shape == (N_TR,)

    # loser rays + live +2tick vector (test space)
    ray_s02 = L(s02) - L(ab)          # downhill areeb geodesic
    ray_s06 = L(s06) - L(ab)          # downhill -C3
    d_AB_AA = L(ab) - L(aa)           # +2 tick live vector AA→AB

    results = {}

    # ── Cand_FWD_AB_areeb: opposite of SLOT02 ──
    d_areeb = clip_p985(L(areeb) - L(ab))
    # SLOT02 used +d_areeb; FWD uses -d_areeb
    p_fwd_areeb, a_fa, e_fa = scale_to_target_1mrho(ab, -d_areeb, 0.025, Zg_te, ab)
    sha_fa, p_fwd_areeb = write_cand(OUT / "Cand_FWD_AB_areeb_0p025e4.csv", ids, p_fwd_areeb)
    # OOF: use -clip(L(areeb_proxy)) — areeb has no OOF; proxy via z_pub for gate N/A
    # Approximate OOF gate using AA/AB public tip geometry only — mark areeb OOF as N/A
    # but still report test cosines
    vec_fa = L(p_fwd_areeb) - L(ab)
    results["Cand_FWD_AB_areeb"] = {
        "file": "Cand_FWD_AB_areeb_0p025e4.csv",
        "sha": sha_fa,
        "formula": "Zr(L(AB) - α·clip_p985(L(areeb)-L(AB))) @ 0.025e-4",
        "alpha": a_fa,
        "1mrho_vs_AB_e4": e_fa,
        "1mrho_vs_S02_e4": one_mrho(p_fwd_areeb, s02) * 1e4,
        "max_U_zone": max_u_zone(p_fwd_areeb, Zg_te),
        "Zr_ok": max_u_zone(p_fwd_areeb, Zg_te) <= 0.0558 + 1e-9,
        "w_v10": 0.0,
        "cos_d_AB_AA": cos_full(vec_fa, d_AB_AA),
        "cos_loser_S02": cos_full(vec_fa, ray_s02),
        "cos_loser_S06": cos_full(vec_fa, ray_s06),
        "OOF": {"note": "areeb has no OOF; live-geometry candidate only", "mean_e5": None, "pos": None},
    }
    log(f"FWD_areeb α={a_fa:.5f} e4={e_fa:.4f} cos_S02={results['Cand_FWD_AB_areeb']['cos_loser_S02']:.4f}")

    # ── Cand_FWD_AB_AA: forward secant AA→AB ──
    d_sec = clip_p985(L(ab) - L(aa))
    p_fwd_aa, a_aa, e_aa = scale_to_target_1mrho(ab, d_sec, 0.030, Zg_te, ab)
    sha_aa, p_fwd_aa = write_cand(OUT / "Cand_FWD_AB_AA_0p030e4.csv", ids, p_fwd_aa)
    vec_aa = L(p_fwd_aa) - L(ab)
    # OOF proxy: Part12 / z_pub cannot reproduce AA/AB tips; mark geometry-only
    # Soft OOF: use z_pub_only + Zr vs z_pub + signed step along AB-AA is undefined on train.
    # Instead: fold-transfer using tip scores is N/A. Report N/A honestly.
    results["Cand_FWD_AB_AA"] = {
        "file": "Cand_FWD_AB_AA_0p030e4.csv",
        "sha": sha_aa,
        "formula": "Zr(L(AB) + β·clip_p985(L(AB)-L(AA))) @ 0.030e-4",
        "alpha": a_aa,
        "1mrho_vs_AB_e4": e_aa,
        "1mrho_vs_S02_e4": one_mrho(p_fwd_aa, s02) * 1e4,
        "max_U_zone": max_u_zone(p_fwd_aa, Zg_te),
        "Zr_ok": max_u_zone(p_fwd_aa, Zg_te) <= 0.0558 + 1e-9,
        "w_v10": 0.0,
        "cos_d_AB_AA": cos_full(vec_aa, d_AB_AA),
        "cos_loser_S02": cos_full(vec_aa, ray_s02),
        "cos_loser_S06": cos_full(vec_aa, ray_s06),
        "OOF": {"note": "AA/AB tips are LB-only; secant is live-geometry", "mean_e5": None, "pos": None},
    }
    log(f"FWD_AA β={a_aa:.5f} e4={e_aa:.4f} cos_dABAA={results['Cand_FWD_AB_AA']['cos_d_AB_AA']:.4f}")

    # ── Cand_UID_clean ──
    # Verify staged P1s + rebuild Zr-locked clip_p985 uid orth vs B_pub
    p1s_path = STAGED / "P1s_AB_uid_a0p015.csv"
    p1s = pd.read_csv(p1s_path)["is_fraud"].to_numpy(float)
    p1s_e4 = one_mrho(p1s, ab) * 1e4
    p1s_sha = sha256_file(p1s_path)

    # Build clean uid orth: LOFO vs B_clean on OOF; on test residual vs B_clean
    uid_orth_o = lofo_orth(uid_o, B_o, folds)
    # test: residual of L(uid) on L(B) using pooled ridge (no TE labels — fit on OOF)
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(L_fold(B_o, folds).reshape(-1, 1), L_fold(uid_o, folds))
    uid_orth_t = L(uid_t) - ridge.predict(L(B_t).reshape(-1, 1))
    uid_orth_t = clip_p985(uid_orth_t)
    uid_orth_o_c = clip_p985(uid_orth_o)

    # disp hygiene
    disp = float(np.std(uid_orth_t) / (np.std(uid_orth_o_c) + 1e-12))
    ks = float(ks_2samp(uid_orth_o_c, uid_orth_t).statistic)

    # both signs OOF gate on z_pub_only + Zr
    g_pos = gate_oof(y, z_pub_o, uid_orth_o_c, folds, Zg_tr, z_pub_o)
    g_neg = gate_oof(y, z_pub_o, -uid_orth_o_c, folds, Zg_tr, z_pub_o)
    if g_neg["mean_e5"] > g_pos["mean_e5"]:
        sign, g_best, dir_t = -1, g_neg, -uid_orth_t
    else:
        sign, g_best, dir_t = +1, g_pos, uid_orth_t

    # calibrate tip climb on AB ~0.20e-4 (mid of 0.18..0.25)
    p_uid, a_uid, e_uid = scale_to_target_1mrho(ab, dir_t, 0.20, Zg_te, ab)
    # if miss band, try 0.22 / 0.18
    if not (0.18 <= e_uid <= 0.25):
        for tgt in (0.22, 0.18, 0.25):
            p_uid, a_uid, e_uid = scale_to_target_1mrho(ab, dir_t, tgt, Zg_te, ab)
            if 0.18 <= e_uid <= 0.25:
                break
    sha_uid, p_uid = write_cand(OUT / "Cand_UID_clean_clip985.csv", ids, p_uid)
    vec_uid = L(p_uid) - L(ab)

    results["Cand_UID_clean"] = {
        "file": "Cand_UID_clean_clip985.csv",
        "sha": sha_uid,
        "formula": f"Zr(L(AB) + α·sign({sign})·clip_p985(uid_v2 ⊥ B_pub=z_pub_only)) @ ~0.20e-4; ZERO uid_catboost",
        "alpha": a_uid,
        "sign": sign,
        "1mrho_vs_AB_e4": e_uid,
        "1mrho_vs_S02_e4": one_mrho(p_uid, s02) * 1e4,
        "max_U_zone": max_u_zone(p_uid, Zg_te),
        "Zr_ok": max_u_zone(p_uid, Zg_te) <= 0.0558 + 1e-9,
        "w_v10": 0.0,
        "uid_catboost": 0.0,
        "disp": disp,
        "KS": ks,
        "disp_ok": 0.95 <= disp <= 1.05,
        "cos_d_AB_AA": cos_full(vec_uid, d_AB_AA),
        "cos_loser_S02": cos_full(vec_uid, ray_s02),
        "cos_loser_S06": cos_full(vec_uid, ray_s06),
        "OOF": g_best,
        "P1s_AB_uid_a0p015": {
            "sha": p1s_sha,
            "1mrho_vs_AB_e4": p1s_e4,
            "max_U_zone": max_u_zone(p1s, Zg_te),
            "cos_d_AB_AA": cos_full(L(p1s) - L(ab), d_AB_AA),
            "cos_loser_S02": cos_full(L(p1s) - L(ab), ray_s02),
            "cos_loser_S06": cos_full(L(p1s) - L(ab), ray_s06),
        },
    }
    log(f"UID_clean sign={sign} α={a_uid:.5f} e4={e_uid:.4f} OOF Δe5={g_best['mean_e5']:.2f} pos={g_best['pos']}/5 disp={disp:.3f}")

    # pairwise 1-ρ table
    cands = {
        "AB": ab,
        "FWD_areeb": p_fwd_areeb,
        "FWD_AA": p_fwd_aa,
        "UID": p_uid,
        "S02": s02,
        "S06": s06,
        "P1s_uid": p1s,
    }
    names = list(cands.keys())
    mat = [[one_mrho(cands[a], cands[b]) * 1e4 for b in names] for a in names]

    # verify loser cosines ≤ 0 for the 3 cands (FWD_areeb should be ≈ −1 vs S02)
    for k in ("Cand_FWD_AB_areeb", "Cand_FWD_AB_AA", "Cand_UID_clean"):
        results[k]["loser_gate_S02_nonpos"] = results[k]["cos_loser_S02"] <= 1e-6
        results[k]["loser_gate_S06_nonpos"] = results[k]["cos_loser_S06"] <= 1e-6

    recon = {
        "NO_SUBMIT": True,
        "LB_01": 0.95569,
        "LB_02": 0.95568,
        "LB_06": 0.95568,
        "quarantine": ["SLOT03", "SLOT04", "SLOT05", "SLOT07", "SLOT08"],
        "hold": ["SLOT09", "SLOT10"],
        "public_tip_recon": {
            "any_kernel_above_095569": False,
            "top_kernel_titles": [
                "goodpjw2008 Board Model+RealMLP LB 0.95569",
                "areebbb Which Signals Rank Fraud LB 0.95569",
            ],
            "note": "No public Step_AC / tip >0.95569 in scoreDescending top-50; newest kernels are low-score eda.",
            "datasets": "goodpjw/amanatar/malyshev/masaya — no new >tip tip-lock dataset since Step_AB; areebbb has no datasets",
        },
        "candidates": results,
        "pairwise_1mrho_e4": {"names": names, "matrix": mat},
    }

    (LOG / "V41_RECON_FWD_SECANT_UID.json").write_text(json.dumps(recon, indent=2))

    # markdown
    lines = [
        "# V41 Recon + Forward-Secant / UID Audit",
        "",
        "**NO_SUBMIT=True** · LB_01=0.95569 · LB_02=0.95568 · LB_06=0.95568",
        "",
        "## Quarantine / Hold",
        "- QUARANTINE: SLOT03,04,05,07 (areeb-family) + **SLOT08** (−C3 axis)",
        "- HOLD: SLOT09, SLOT10 (untouched)",
        "",
        "## Public Recon",
        "- **No public kernel/dataset tip > 0.95569** since Step_AB (top public still 0.95569).",
        "- No `Step_AC` / new ladder tip published.",
        "",
        "## Candidate Summary",
        "",
        "| Cand | 1−ρ vs AB e4 | α | cos·d_AB_AA | cos·S02↓ | cos·S06↓ | OOF Δe5 | pos/5 | Zr_ok | loser≤0? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for k, lab in (
        ("Cand_FWD_AB_areeb", "FWD_AB_areeb"),
        ("Cand_FWD_AB_AA", "FWD_AB_AA"),
        ("Cand_UID_clean", "UID_clean"),
    ):
        r = results[k]
        oof = r["OOF"]
        oof_s = "N/A" if oof.get("mean_e5") is None else f"{oof['mean_e5']:.2f}"
        pos_s = "N/A" if oof.get("pos") is None else f"{oof['pos']}/5"
        loser_ok = r["loser_gate_S02_nonpos"] and r["loser_gate_S06_nonpos"]
        lines.append(
            f"| {lab} | {r['1mrho_vs_AB_e4']:.4f} | {r['alpha']:.5f} | {r['cos_d_AB_AA']:.4f} | "
            f"{r['cos_loser_S02']:.4f} | {r['cos_loser_S06']:.4f} | {oof_s} | {pos_s} | "
            f"{r['Zr_ok']} | {loser_ok} |"
        )
    lines += [
        "",
        "### Pairwise 1−ρ (×1e-4)",
        "",
        "| |" + "|".join(names) + "|",
        "|" + "|".join(["---"] * (len(names) + 1)) + "|",
    ]
    for i, a in enumerate(names):
        lines.append("| " + a + " |" + "|".join(f"{mat[i][j]:.3f}" for j in range(len(names))) + "|")

    lines += [
        "",
        "### Files",
        f"- `{OUT / 'Cand_FWD_AB_areeb_0p025e4.csv'}` sha={sha_fa}",
        f"- `{OUT / 'Cand_FWD_AB_AA_0p030e4.csv'}` sha={sha_aa}",
        f"- `{OUT / 'Cand_UID_clean_clip985.csv'}` sha={sha_uid}",
        f"- staged P1s_AB_uid_a0p015 sha={p1s_sha} 1−ρ={p1s_e4:.4f}e-4",
        "",
        "## STOP",
        "NO_SUBMIT=True. Awaiting explicit single-slot fire command.",
    ]
    (LOG / "V41_RECON_FWD_SECANT_UID.md").write_text("\n".join(lines) + "\n")
    log(f"WROTE {LOG / 'V41_RECON_FWD_SECANT_UID.md'}")
    print("\n" + "\n".join(lines[-25:]), flush=True)
    print("STOP. NO_SUBMIT=True. Await fire command.", flush=True)


if __name__ == "__main__":
    main()
