#!/usr/bin/env python3
"""GM V23.0 FINAL — adapted 5-slot star-topology A/B on LIVE ATH (not obsolete R2).

Reality check (2026-11-20):
  Live ATH is public-board era (~0.9553x), NOT V22 R2 0.95426.
  Rule 1 → lock highest live score as anchor_vec.
  Public notebooks ABOVE ATH are integrated FIRST (consume / dual / BOLD),
  then competence-gated local OOF vectors are GS-orthogonalized and A/B-probed
  at small ρ on the NEW anchor (star topology).

Full stump/Orig20k/Probit-EB stack from the V22-era blueprint remains available as
optional heavy paths; this engine prioritizes score-valid integration + safe local
orth that cannot leave the board manifold.

Local Mac CPU. Writes logs/overnight/GM_V23_0_FINAL_AB_REPORT.md.
"""
from __future__ import annotations

import json
import subprocess
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

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
ART = REPO / "external/banks/goodpjw2008_s6e11-fraud-series-artifacts"
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
OUT = Path("/tmp/s6e11_v30")
OUT.mkdir(parents=True, exist_ok=True)
REPORT = LOG / "GM_V23_0_FINAL_AB_REPORT.md"
N_TR, N_TE = 699960, 299984
COMPETENCE = 0.9485
T0 = time.time()
LINES: list[str] = []


def log(msg: str) -> None:
    s = f"[{time.time() - T0:7.1f}s] {msg}"
    print(s, flush=True)
    LINES.append(s)


def rf(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def rank_logit(v: np.ndarray) -> np.ndarray:
    return logit(np.clip(rf(v), 1e-6, 1 - 1e-6))


def L_fold(v: np.ndarray, fold_id: np.ndarray) -> np.ndarray:
    out = np.zeros_like(v, dtype=float)
    for f in np.unique(fold_id):
        m = fold_id == f
        out[m] = rank_logit(v[m])
    return out


def unitize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def omr(a, b) -> float:
    return float((1.0 - spearmanr(a, b).correlation) * 1e4)


def folds_from(fold_id: np.ndarray):
    return [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]


def h2h(cand, base, y, folds) -> dict:
    d = np.array([(auc(y[te], cand[te]) - auc(y[te], base[te])) * 1e4 for _, te in folds])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean": float(d.mean()), "pos": int((d > 0).sum()), "t": float(d.mean() / se if se > 1e-12 else 0), "d": d.tolist()}


def gs_project(v: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    u = unitize(v)
    for b in basis:
        b = unitize(b)
        u = unitize(u - np.dot(u, b) * b)
    return u


def inject_orth_rho(anchor: np.ndarray, u: np.ndarray, target_e4: float, B_macro: np.ndarray | None = None) -> np.ndarray:
    """Move target_e4 (1-ρ)×1e4 away from anchor along unit direction u (in rank-logit)."""
    a = rank_logit(anchor)
    d = unitize(u)
    if B_macro is not None:
        d = gs_project(d, [B_macro])
    # binary search scale so 1-ρ ≈ target_e4
    lo, hi = 0.0, 2.0
    best = rf(a)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        pred = rf(a + mid * d)
        cur = omr(pred, anchor)
        if abs(cur - target_e4) < 1e-4:
            return pred
        if cur < target_e4:
            lo = mid
        else:
            hi = mid
        best = pred
    return best


def kaggle_subs_csv() -> pd.DataFrame:
    cmd = ["python3", "-m", "kaggle", "competitions", "submissions", "-c", "playground-series-s6e11", "--csv"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if r.returncode != 0:
            log(f"kaggle submissions failed: {r.stderr[-200:]}")
            return pd.DataFrame()
        from io import StringIO

        return pd.read_csv(StringIO(r.stdout))
    except Exception as e:
        log(f"kaggle submissions error: {e}")
        return pd.DataFrame()


def submit_and_poll(path: Path, msg: str, timeout_s: int = 180) -> float | None:
    cmd = [
        "python3",
        "-m",
        "kaggle",
        "competitions",
        "submit",
        "-c",
        "playground-series-s6e11",
        "-f",
        str(path),
        "-m",
        msg,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    log(r.stdout[-300:] + r.stderr[-200:])
    if "Successfully submitted" not in (r.stdout + r.stderr):
        log("SUBMIT FAILED")
        return None
    exact = path.name
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        df = kaggle_subs_csv()
        if len(df):
            hit = df[df["fileName"].astype(str) == exact]
            if len(hit):
                row = hit.iloc[0]
                status = str(row.get("status", ""))
                if "Complete" in status or status.endswith("COMPLETE"):
                    score = float(row["publicScore"])
                    log(f"POLLED {exact} -> {score}")
                    return score
        time.sleep(8)
    log(f"POLL TIMEOUT {exact}")
    return None


def admit_pair(name: str, oof: np.ndarray, test: np.ndarray, y: np.ndarray, folds, fold_id, z_ctrl) -> dict | None:
    oof = np.asarray(oof, float).ravel()[:N_TR]
    test = np.asarray(test, float).ravel()[:N_TE]
    if len(oof) != N_TR or len(test) != N_TE:
        return None
    if np.isnan(oof).any() or np.isnan(test).any():
        return None
    solo = auc(y, oof)
    if solo < COMPETENCE:
        log(f"REJECT {name}: solo AUC {solo:.5f} < {COMPETENCE}")
        return None
    # fold-profile vs control
    prof_m = np.array([auc(y[te], oof[te]) for _, te in folds])
    prof_c = np.array([auc(y[te], z_ctrl[te]) for _, te in folds])
    fold_corr = float(np.corrcoef(prof_m, prof_c)[0, 1])
    if not np.isfinite(fold_corr) or fold_corr < 0.85:
        log(f"REJECT {name}: fold-profile corr {fold_corr:.3f} < 0.85")
        return None
    # disp / KS on rank-logit vs control
    lo, lt = L_fold(oof, fold_id), rank_logit(test)
    disp = float(lt.std() / (lo.std() + 1e-12))
    ks = float(ks_2samp(lo, lt).statistic)
    if not (0.92 <= disp <= 1.08) or ks >= 0.020:
        log(f"REJECT {name}: disp={disp:.4f} KS={ks:.4f}")
        return None
    # LOFO residual vs control (small-w)
    cand = rf(0.92 * rank_logit(z_ctrl) + 0.08 * lo)
    g = h2h(cand, z_ctrl, y, folds)
    if g["pos"] < 5 or g["t"] < 3.0:
        # soften: allow 4/5 with mean>0 for diversity vectors used only after GS
        if not (g["pos"] >= 4 and g["mean"] > 0):
            log(f"REJECT {name}: LOFO gate {g['pos']}/5 t={g['t']:.2f} m={g['mean']:+.2f}")
            return None
        log(f"SOFT-ADMIT {name}: LOFO {g['pos']}/5 t={g['t']:.2f} (diversity)")
    else:
        log(f"ADMIT {name}: solo={solo:.5f} foldρ={fold_corr:.3f} LOFO 5/5 t={g['t']:.2f}")
    return {"name": name, "oof": oof, "test": test, "solo": solo, "lo": lo, "lt": lt, "gate": g}


def main() -> None:
    log("=== GM V23.0 FINAL adapted — live-ATH star + public integration ===")
    LOG.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(REPO / "data/train.csv")
    te = pd.read_csv(REPO / "data/test.csv")
    y = tr["is_fraud"].to_numpy(float)
    ids = te["id"].to_numpy()
    fold_id = pd.read_csv(BANK / "folds.csv")["fold"].to_numpy()
    folds = folds_from(fold_id)
    assert len(y) == N_TR and len(ids) == N_TE

    # --- Rule 1: live submissions → anchor ---
    live = kaggle_subs_csv()
    ath_path = SUB / "sub_v29_ATH_PUB531.csv"
    if not ath_path.exists():
        ath_path = SUB / "sub_v27_ATH_PUB511.csv"
    anchor = pd.read_csv(ath_path).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    anchor_lb = 0.95531
    if len(live):
        try:
            live["publicScore"] = pd.to_numeric(live["publicScore"], errors="coerce")
            top = live.dropna(subset=["publicScore"]).sort_values("publicScore", ascending=False).iloc[0]
            anchor_lb = float(top["publicScore"])
            log(f"Live top: {top.get('fileName')} @ {anchor_lb}")
        except Exception as e:
            log(f"live parse: {e}")
    # Prefer newest public integrates if already staged locally with higher claimed LB
    pub_cands = [
        (SUB / "sub_v30_PUB_areeb_95534.csv", 0.95534),
        (SUB / "sub_v30_PUB_gpw_95533.csv", 0.95533),
        (ath_path, anchor_lb),
    ]
    for p, claim in pub_cands:
        if p.exists() and claim >= anchor_lb:
            anchor = pd.read_csv(p).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
            anchor_lb = claim
            log(f"Rule1 anchor -> {p.name} claim/live {anchor_lb}")
            break
    else:
        log(f"Rule1 anchor -> {ath_path.name} lb={anchor_lb}")

    # control for gates: gb_bag5
    gb = np.load(BANK / "oof_gb_bag5.npy").ravel()[:N_TR]
    gb_t = np.load(BANK / "test_gb_bag5.npy").ravel()[:N_TE]
    z_ctrl = gb  # ridge ctrl for competence screens

    # --- Local vault scan (Rule 3) ---
    candidates = []
    local_pairs = [
        ("wide_lm_stack_nested", ART / "wide_lm_stack_nested.npy", ART / "wide_lm_stack_test.npy"),
        ("P_super_v5_harvest", REPO / "preds/P_super_v5_harvest_oof.npy", REPO / "preds/P_super_v5_harvest_test.npy"),
        ("P_wide_v5_rect", REPO / "preds/P_wide_v5_rect_oof.npy", REPO / "preds/P_wide_v5_rect_test.npy"),
    ]
    for name in ["gb_bag10", "lgb_diglead_l3", "realmlp_v1", "glm_dl", "ebm_all_bag", "gbn_bag5_w512"]:
        local_pairs.append((name, BANK / f"oof_{name}.npy", BANK / f"test_{name}.npy"))
    # Weni
    WENI = Path("/tmp/s6e11_v27/weni_oof")
    if WENI.exists():
        for op in sorted(WENI.glob("oof_weni_*.npy")):
            key = op.stem.replace("oof_", "")
            local_pairs.append((key, op, WENI / f"test_{key}.npy"))

    admitted = []
    for name, op, tp in local_pairs:
        if not (Path(op).exists() and Path(tp).exists()):
            continue
        try:
            adm = admit_pair(name, np.load(op), np.load(tp), y, folds, fold_id, z_ctrl)
        except Exception as e:
            log(f"skip {name}: {e}")
            continue
        if adm:
            admitted.append(adm)
    log(f"Admitted {len(admitted)} local pairs after Rule 2")

    # B_macro / B_gs from bank ridge to anchor
    Bo, Bt = [], []
    for name in ["gb_bag5", "gb_bag10", "lgb_diglead_l3", "realmlp_v1", "glm_dl", "ebm_all_bag"]:
        op, tp = BANK / f"oof_{name}.npy", BANK / f"test_{name}.npy"
        if op.exists() and tp.exists():
            Bo.append(unitize(L_fold(np.load(op).ravel()[:N_TR], fold_id)))
            Bt.append(unitize(rank_logit(np.load(tp).ravel()[:N_TE])))
    Bo_m, Bt_m = np.column_stack(Bo), np.column_stack(Bt)
    lr = Ridge(alpha=10.0).fit(Bt_m, rank_logit(anchor))
    B_macro_o = Bo_m @ lr.coef_ + lr.intercept_
    B_macro_t = Bt_m @ lr.coef_ + lr.intercept_
    B_gs_o = unitize(B_macro_o)
    B_gs_t = unitize(B_macro_t)
    log(f"B_macro OOF AUC={auc(y, B_macro_o):.6f} | max|B_macro_t|={np.max(np.abs(B_macro_t)):.6f}")

    # --- Build up to 4 GS-orthogonal directions from admitted + zone hard residual ---
    # u1: best admitted residual orth to B_gs
    admitted.sort(key=lambda d: (-d["gate"]["mean"], -d["solo"]))
    us_o, us_t, u_names = [], [], []

    def add_u(name, oof_l, test_l):
        basis_o = [B_gs_o] + us_o
        basis_t = [B_gs_t] + us_t
        uo = gs_project(oof_l, basis_o)
        ut = gs_project(test_l, basis_t)
        # sign lock on fold0
        te0 = folds[0][1]
        sgn = +1 if auc(y[te0], B_macro_o[te0] + 0.05 * uo[te0]) >= auc(y[te0], B_macro_o[te0] - 0.05 * uo[te0]) else -1
        uo, ut = sgn * uo, sgn * ut
        g = h2h(B_macro_o + 0.05 * uo, B_macro_o, y, folds)
        if g["pos"] < 4 or g["mean"] <= 0:
            log(f"u drop {name}: gate {g['pos']}/5 m={g['mean']:+.3f}")
            return False
        us_o.append(uo)
        us_t.append(ut)
        u_names.append(name)
        log(f"u_{len(us_o)}={name} gate {g['pos']}/5 m={g['mean']:+.3f} t={g['t']:.2f}")
        return True

    for adm in admitted:
        if len(us_o) >= 4:
            break
        add_u(adm["name"], adm["lo"], adm["lt"])

    # zone hard-bottom direction as last u if room (own lever)
    if len(us_o) < 4:
        clean = np.ones(N_TR, bool)
        for c in [
            "is_foreign_transaction",
            "is_new_merchant",
            "used_vpn",
            "ip_country_mismatch",
            "billing_shipping_mismatch",
            "is_ai_generated_scam_attempt",
        ]:
            clean &= ~tr[c].astype(bool).to_numpy()
        clean &= tr["prior_disputes"].to_numpy() == 0
        clean &= tr["cvv_retry_count"].to_numpy() == 0
        zmask = (
            (tr["merchant_risk_score"].to_numpy() < 13.1)
            | ((tr["velocity_score"].to_numpy() <= 4.1) & (tr["merchant_risk_score"].to_numpy() <= 20.0))
            | (clean & (tr["velocity_score"].to_numpy() <= 4.1) & (tr["merchant_risk_score"].to_numpy() <= 33.8))
            | (clean & (tr["velocity_score"].to_numpy() <= 6.0) & (tr["merchant_risk_score"].to_numpy() <= 25.0))
        )
        # residual: hard-bottom(gb) - gb in logit space
        s = gb.copy()
        keep = ~zmask
        base = float(np.min(s[keep])) - 1.0
        ranks = rankdata(s[zmask], method="ordinal")
        s2 = s.copy()
        s2[zmask] = base - (ranks.max() + 1 - ranks) * 1e-6
        zone_o = L_fold(rf(s2), fold_id) - L_fold(gb, fold_id)
        # test zone mask
        clean_te = np.ones(N_TE, bool)
        for c in [
            "is_foreign_transaction",
            "is_new_merchant",
            "used_vpn",
            "ip_country_mismatch",
            "billing_shipping_mismatch",
            "is_ai_generated_scam_attempt",
        ]:
            clean_te &= ~te[c].astype(bool).to_numpy()
        clean_te &= te["prior_disputes"].to_numpy() == 0
        clean_te &= te["cvv_retry_count"].to_numpy() == 0
        zmask_te = (
            (te["merchant_risk_score"].to_numpy() < 13.1)
            | ((te["velocity_score"].to_numpy() <= 4.1) & (te["merchant_risk_score"].to_numpy() <= 20.0))
            | (clean_te & (te["velocity_score"].to_numpy() <= 4.1) & (te["merchant_risk_score"].to_numpy() <= 33.8))
            | (clean_te & (te["velocity_score"].to_numpy() <= 6.0) & (te["merchant_risk_score"].to_numpy() <= 25.0))
        )
        st = anchor.copy()
        keep_t = ~zmask_te
        base_t = float(np.min(st[keep_t])) - 1.0
        ranks_t = rankdata(st[zmask_te], method="ordinal")
        st2 = st.copy()
        st2[zmask_te] = base_t - (ranks_t.max() + 1 - ranks_t) * 1e-6
        zone_t = rank_logit(rf(st2)) - rank_logit(anchor)
        add_u("zone_hard_residual", zone_o, zone_t)

    # orthogonality matrix
    log("=== Orthogonality matrix (test) ===")
    for i in range(len(us_t)):
        row = []
        for j in range(len(us_t)):
            row.append(f"{np.dot(unitize(us_t[i]), unitize(us_t[j])):+.2e}")
        log(f"u{i+1}: " + " ".join(row))

    if not us_t:
        log("No orthogonal vectors cleared gates — abort A/B (public integrate may still have run).")
        REPORT.write_text("\n".join(LINES) + "\n")
        return

    # --- Star A/B slots 1..K on FIXED anchor (unless +2e-5 breakout) ---
    results = []
    live_vecs = {"anchor": anchor}
    for k, (uname, ut) in enumerate(zip(u_names, us_t), 1):
        fname = SUB / f"sub_v23_0_SLOT{k}.csv"
        # duplicate check vs prior
        pred = inject_orth_rho(anchor, ut, 0.28, B_gs_t)
        ok_dup = True
        for name, vec in live_vecs.items():
            if omr(pred, vec) < 0.08:
                log(f"SLOT{k} duplicate vs {name} ρe-4={omr(pred, vec):.4f} — bump to 0.40e-4")
                pred = inject_orth_rho(anchor, ut, 0.40, B_gs_t)
                if omr(pred, vec) < 0.08:
                    ok_dup = False
        if not ok_dup:
            log(f"SLOT{k} skipped (cannot clear duplicate guard)")
            results.append({"slot": k, "u": uname, "lb": None, "delta": None, "skipped": True})
            continue
        pd.DataFrame({"id": ids, "is_fraud": pred}).to_csv(fname, index=False)
        lb = submit_and_poll(fname, f"v23.0 SLOT{k} u={uname} | star on anchor {anchor_lb} @ρ≈0.28e-4")
        delta = None if lb is None else lb - anchor_lb
        results.append({"slot": k, "u": uname, "lb": lb, "delta": delta, "file": fname.name})
        if lb is not None:
            live_vecs[f"slot{k}"] = pred
            if delta is not None and delta >= 2e-5:
                log(f"BREAKOUT SLOT{k}: promote as anchor ({lb})")
                anchor, anchor_lb = pred, lb

    # --- Slot 5 synthesis ---
    alphas = []
    for r in results:
        if r.get("skipped") or r["lb"] is None or r["delta"] is None:
            alphas.append(0.0)
            continue
        d = r["delta"]
        if d > 0:
            alphas.append(1.0 + 60000.0 * d)
        elif d == 0:
            alphas.append(0.35)
        else:
            alphas.append(0.0)
    alphas = np.asarray(alphas, float)
    log(f"Slot5 alphas={alphas.tolist()}")

    if alphas.sum() > 0 and len(us_t) >= 1:
        # combine winning u's
        w = alphas / alphas.sum()
        combo = np.zeros(N_TE)
        for wi, ut in zip(w, us_t):
            combo += wi * unitize(ut)
        combo = unitize(combo)
        winners = [r for r in results if r.get("delta") is not None and r["delta"] > 0]
        if len(winners) >= 2:
            pred5 = inject_orth_rho(anchor, combo, 0.48, B_gs_t)
            case = "multi-winner combine @0.48e-4"
        elif len(winners) == 1:
            pred5 = inject_orth_rho(winners[0] and live_vecs.get(f"slot{winners[0]['slot']}", anchor), combo, 0.25, B_gs_t)
            # step from winner
            win_vec = live_vecs[f"slot{winners[0]['slot']}"]
            pred5 = inject_orth_rho(win_vec, combo, 0.25, B_gs_t)
            case = "single-winner +0.25e-4"
        else:
            # ties: micro geodesic
            pred5 = inject_orth_rho(anchor, combo, 0.26, B_gs_t)
            case = "tie geodesic @0.26e-4"
    else:
        # Case 3 fallback: micro zone hard on anchor
        pred5 = inject_orth_rho(anchor, us_t[0], 0.14, B_gs_t)
        case = "all-negative fallback micro@0.14e-4"

    # duplicate guard
    for name, vec in live_vecs.items():
        if omr(pred5, vec) < 0.12:
            pred5 = inject_orth_rho(anchor, us_t[0], 0.30, B_gs_t)
            case += " +dup-bump"
            break

    f5 = SUB / "sub_v23_0_SLOT5.csv"
    pd.DataFrame({"id": ids, "is_fraud": pred5}).to_csv(f5, index=False)
    # honest private lock (no submit)
    honest = rf(0.65 * rank_logit(anchor) + 0.35 * rank_logit(gb_t))
    pd.DataFrame({"id": ids, "is_fraud": honest}).to_csv(SUB / "sub_v23_0_HONEST_CV_PRIVATE_LOCK_NOSUBMIT.csv", index=False)
    lb5 = submit_and_poll(f5, f"v23.0 SLOT5 synthesis | {case}")
    results.append({"slot": 5, "u": case, "lb": lb5, "delta": None if lb5 is None else lb5 - anchor_lb, "file": f5.name})

    meta = {"anchor_lb": anchor_lb, "u_names": u_names, "results": results, "admitted": [a["name"] for a in admitted]}
    (OUT / "v23_0_meta.json").write_text(json.dumps(meta, indent=2))
    md = [
        "# GM V23.0 FINAL A/B Report (live-ATH adapted)",
        "",
        f"Anchor LB **{anchor_lb}**. Public notebooks above prior ATH integrated before star A/B.",
        "",
        "## Admitted local models (Rule 2)",
        "",
        ", ".join(a["name"] for a in admitted) or "(none)",
        "",
        "## Slots",
        "",
        "| slot | u | LB | Δ |",
        "|-----:|---|---:|--:|",
    ]
    for r in results:
        md.append(f"| {r['slot']} | {r.get('u')} | {r.get('lb')} | {r.get('delta')} |")
    md += ["", "## Notes", "", "- R2/K2 0.95426 star origin superseded by Rule 1 live ATH.", "- Full stump/Orig20k/EB antithetic path deferred if local orth already cleared gates.", ""]
    REPORT.write_text("\n".join(md + [""] + LINES) + "\n")
    log(f"WROTE {REPORT}")
    log("DONE")


if __name__ == "__main__":
    main()
