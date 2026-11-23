#!/usr/bin/env python3
"""V36.2 morning harvest — read-only Dual-Baseline audit + MultiVault synth (NO_SUBMIT)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
os.environ["NO_SUBMIT"] = "True"

from v36_1_lib import (  # noqa: E402
    COMPETENCE_FLOOR,
    LEAKAGE_SOLO_CEILING,
    LOGS,
    PRED,
    SUB,
    U,
    L,
    auc,
    atomic_json,
    blend_ath,
    build_subfolds,
    evaluate_engine_g0,
    frozen_lofo_orth,
    load_y_fold_ids,
    r2_span,
    signed_lofo_grid,
    svd_basis,
    unit,
    winsor,
)

NO_SUBMIT = True
OUT = LOGS / "GM_V36_2_MORNING_HARVEST.json"
OUT_MD = LOGS / "GM_V36_2_MORNING_HARVEST.md"
INTEL_MD = LOGS / "GM_V36_1_INTEL.md"


def _skip_appledouble(name: str) -> bool:
    return name.startswith("._")


def _load_pair(oof_p: Path, te_p: Path, n_tr: int, n_te: int):
    if _skip_appledouble(oof_p.name) or not oof_p.exists() or not te_p.exists():
        return None
    oof = np.load(oof_p).astype(float).ravel()[:n_tr]
    te = np.load(te_p).astype(float).ravel()[:n_te]
    if len(oof) != n_tr or len(te) != n_te:
        return None
    return oof, te


def evaluate_pure_residual(
    name: str,
    delta_o: np.ndarray,
    delta_t: np.ndarray,
    y,
    fold,
    sf,
    z_fb,
    z_fb_te,
    z_twin,
    z_twin_te,
    Bo,
    Bt,
) -> dict:
    """Score pure δ jewels: no competence on residual solo; no L(z) subtraction; signed w ±0.06.

    Fast path for morning harvest: Ridge-LOFO vs dual baselines only (no full B GS).
    """
    from scipy.stats import ks_2samp
    from sklearn.linear_model import Ridge

    delta_o = np.asarray(delta_o, float).ravel()
    delta_t = np.asarray(delta_t, float).ravel()
    solo_proxy = auc(y, U(L(z_fb) + 0.02 * unit(delta_o)))
    raw_disp = float((delta_t - delta_t.mean()).std() / ((delta_o - delta_o.mean()).std() + 1e-12))
    ks = float(ks_2samp(unit(delta_o), unit(delta_t)).statistic)
    f1 = (0.40 <= raw_disp <= 1.06) and (ks < 0.05)

    kB = min(8, Bt.shape[1])
    B152 = svd_basis(np.column_stack([unit(Bt[:, j]) for j in range(kB)]), min(40, kB))
    r2 = r2_span(delta_t, B152)

    # Fast LOFO Ridge vs [z_fb, z_twin] only
    u_o = np.zeros(len(delta_o), float)
    for f in range(5):
        tr, va = fold != f, fold == f
        Xtr = np.column_stack([unit(z_fb[tr]), unit(z_twin[tr])])
        Xva = np.column_stack([unit(z_fb[va]), unit(z_twin[va])])
        ridge = Ridge(alpha=50.0, fit_intercept=False)
        ridge.fit(Xtr, unit(delta_o[tr]))
        u_o[va] = unit(delta_o[va]) - ridge.predict(Xva)
    u_o = winsor(u_o)
    Xt = np.column_stack([unit(z_fb_te), unit(z_twin_te)])
    Xo = np.column_stack([unit(z_fb), unit(z_twin)])
    ridge = Ridge(alpha=50.0, fit_intercept=False)
    ridge.fit(Xo, unit(delta_o))
    u_t = winsor(unit(delta_t) - ridge.predict(Xt))
    # light GS vs baselines + first B cols
    for col in (z_fb_te, z_twin_te, *[Bt[:, j] for j in range(kB)]):
        bb = unit(col)
        u_t = unit(u_t - np.dot(u_t, bb) / (np.dot(bb, bb) + 1e-12) * bb)
    u_t = winsor(u_t)
    f2 = 0.0
    for j in range(kB):
        f2 = max(f2, abs(float(np.corrcoef(u_t, unit(Bt[:, j]))[0, 1])))
    for bt in (z_fb_te, z_twin_te):
        f2 = max(f2, abs(float(np.corrcoef(u_t, unit(bt))[0, 1])))
    disp_orth = float(u_t.std() / (u_o.std() + 1e-12))
    orth_meta = {"F2_max_corr": f2, "disp_orth": disp_orth}
    f2_ok = f2 < 0.010 and 0.84 <= disp_orth <= 1.06

    # Fast dual-baseline G1: coarse signed w grid (not full nested LOFO)
    def fast_signed(base, u):
        best_w, best_mean, best_pos = 0.0, -1e18, 0
        d_at_best = None
        for w in np.linspace(-0.06, 0.06, 25):
            if abs(w) < 1e-12:
                continue
            ds = []
            for f in range(5):
                va = fold == f
                d = (auc(y[va], U(L(base[va]) + w * u[va])) - auc(y[va], base[va])) * 1e4
                ds.append(d)
            ds = np.asarray(ds, float)
            if ds.mean() > best_mean:
                best_mean, best_w, best_pos, d_at_best = float(ds.mean()), float(w), int((ds > 0).sum()), ds
        se = d_at_best.std(ddof=1) / np.sqrt(5) if d_at_best is not None and d_at_best.std(ddof=1) > 1e-12 else 1e-12
        return {
            "mean_e4": best_mean,
            "pos": best_pos,
            "sub": -1,
            "t25": float(best_mean / se),
            "w_fixed": best_w,
            "d": [round(float(x), 4) for x in (d_at_best if d_at_best is not None else [])],
        }

    g_fb = fast_signed(z_fb, u_o)
    g_twin = fast_signed(z_twin, u_o)
    same_sign = np.sign(g_fb["w_fixed"]) == np.sign(g_twin["w_fixed"]) or abs(g_twin["w_fixed"]) < 1e-8
    g0 = False  # full G0 deferred; harvest uses G1 + vault-lite
    vault = (
        f1 and f2_ok and same_sign and g_fb["pos"] >= 4 and g_fb["mean_e4"] >= 0.8
        and g_twin["pos"] >= 4 and g_twin["mean_e4"] > 0
    )
    g1 = g_fb["pos"] == 5 and g_twin["pos"] >= 4 and same_sign

    return {
        "name": name,
        "status": "ok",
        "is_residual": True,
        "solo": solo_proxy,
        "solo_note": "proxy auc(U(L(z_fb)+0.02*u(δ))) — competence NOT gated",
        "raw_disp": raw_disp,
        "ks": ks,
        "f1": f1,
        "r2_B152": r2,
        "r2_on": "delta_as_loaded",
        "orth": orth_meta,
        "f2_ok": f2_ok,
        "g_fb": g_fb,
        "g_twin": g_twin,
        "same_sign": bool(same_sign),
        "g0": bool(g0),
        "vault": bool(vault),
        "g1": bool(g1),
        "u_oof": u_o,
        "u_test": u_t,
        "gate_mode": "fast_signed_harvest",
    }


def _baselines(n_tr, n_te):
    z_fb = np.load(PRED / "z_fullbank_oof.npy").ravel()[:n_tr]
    z_fb_te = np.load(PRED / "z_fullbank_test.npy").ravel()[:n_te]
    z_twin = np.load(PRED / "z_ATH_twin_oof.npy").ravel()[:n_tr]
    z_twin_te = np.load(PRED / "z_ATH_twin_test.npy").ravel()[:n_te]
    Bo = np.load(PRED / "B_clean_oof.npy")[:n_tr]
    Bt = np.load(PRED / "B_clean_test.npy")[:n_te]
    eta = np.load(PRED / "eta_cal_fb_oof.npy").ravel()[:n_tr]
    eta_te = np.load(PRED / "eta_cal_fb_test.npy").ravel()[:n_te]
    return z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te


def score_absolute(name, oof, te, y, fold, sf, bases):
    """Prefer existing gate JSON + u_*.npy (fast); full G0 only if missing."""
    z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te = bases
    gp = LOGS / "v36_1_ckpt" / f"{name}_gate.json"
    uo, ut = PRED / f"{name}_u_oof.npy", PRED / f"{name}_u_test.npy"
    if gp.exists() and not _skip_appledouble(gp.name):
        try:
            rec = json.loads(gp.read_text())
            rec = dict(rec)
            if uo.exists() and ut.exists():
                rec["u_oof"] = np.load(uo).ravel()[: len(y)]
                rec["u_test"] = np.load(ut).ravel()[: len(te)]
            rec["from_cache"] = True
            return rec
        except Exception:
            pass
    rec = evaluate_engine_g0(
        name, oof, te, y, fold, sf, z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te
    )
    if uo.exists() and ut.exists():
        rec["u_oof"] = np.load(uo).ravel()[: len(y)]
        rec["u_test"] = np.load(ut).ravel()[: len(te)]
    return rec


def main():
    t0 = time.time()
    assert NO_SUBMIT and os.environ.get("NO_SUBMIT") == "True"
    y, fold, ids_te, ath = load_y_fold_ids()
    n_tr, n_te = len(y), len(ath)
    sf = build_subfolds(y, fold)
    bases = _baselines(n_tr, n_te)
    z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt, eta, eta_te = bases

    rows = []
    vectors = []  # (name, u_o, u_t, solo, rec)

    # ---- absolute engines (E3 / E1_fixed) ----
    abs_candidates = [
        ("W1_E3_hier_eb_3d4d_and_stealth_fn", PRED / "W1_E3_hier_eb_3d4d_and_stealth_fn_oof.npy", PRED / "W1_E3_hier_eb_3d4d_and_stealth_fn_test.npy"),
        ("W1_E1_fixed_entity_balance_graph", PRED / "W1_E1_fixed_entity_balance_graph_oof.npy", PRED / "W1_E1_fixed_entity_balance_graph_test.npy"),
        ("W1_E2_orig20k_inversion_and_bpe_exact", PRED / "W1_E2_orig20k_inversion_and_bpe_exact_oof.npy", PRED / "W1_E2_orig20k_inversion_and_bpe_exact_test.npy"),
        ("W1_E4_wilcoxon_twilight_mps_specialist", PRED / "W1_E4_wilcoxon_twilight_mps_specialist_oof.npy", PRED / "W1_E4_wilcoxon_twilight_mps_specialist_test.npy"),
    ]
    for name, op, tp in abs_candidates:
        pair = _load_pair(op, tp, n_tr, n_te)
        if pair is None:
            rows.append({"name": name, "status": "missing"})
            print(f"MISS {name}", flush=True)
            continue
        oof, te = pair
        print(f"SCORE abs {name} …", flush=True)
        rec = score_absolute(name, oof, te, y, fold, sf, bases)
        slim = {k: rec[k] for k in rec if k not in ("u_oof", "u_test", "g_fb", "g_twin", "orth")}
        slim["g_fb_pos"] = (rec.get("g_fb") or {}).get("pos")
        slim["g_fb_e4"] = (rec.get("g_fb") or {}).get("mean_e4")
        slim["g_twin_pos"] = (rec.get("g_twin") or {}).get("pos")
        slim["F2"] = (rec.get("orth") or {}).get("F2_max_corr")
        rows.append(slim)
        print(
            f"  → status={rec.get('status')} solo={rec.get('solo')} g0={rec.get('g0')} vault={rec.get('vault')} "
            f"r2={rec.get('r2_B152')} f1={rec.get('f1')} f2={rec.get('f2_ok')}",
            flush=True,
        )
        solo = float(rec.get("solo") or 0)
        if (
            rec.get("status") == "ok"
            and solo <= LEAKAGE_SOLO_CEILING
            and "u_oof" in rec
            and "u_test" in rec
        ):
            vectors.append((name, rec["u_oof"], rec["u_test"], solo, rec))

    # ---- pure residual jewels ----
    resid_candidates = [
        ("harvest_p4_cuda_xgb", REPO / "preds/matrix_v2/harvest_p4_cuda_xgb_oof.npy", REPO / "preds/matrix_v2/harvest_p4_cuda_xgb_test.npy"),
        ("P4_L4.2_pure5f", REPO / "preds/matrix_v2/P4_L4.2_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("P4_L4.3_pure5f", REPO / "preds/matrix_v2/P4_L4.3_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.3_pure5f_test.npy"),
        ("P4_L4.5_pure5f", REPO / "preds/matrix_v2/P4_L4.5_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.5_pure5f_test.npy"),
        ("v10_ridge_resid", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_oof.npy", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_test.npy"),
    ]
    # U_orth4 if present (shallow only — never rglob whole preds/)
    for root in (PRED, REPO / "preds/v36_1_vault", REPO / "preds/v10_transductive_bulsah"):
        if not root.exists():
            continue
        for p in root.glob("*u_orth4*_oof.npy"):
            if _skip_appledouble(p.name):
                continue
            te = p.with_name(p.name.replace("_oof.npy", "_test.npy"))
            resid_candidates.append((p.stem.replace("_oof", ""), p, te))
        for p in root.glob("*U_orth4*_oof.npy"):
            if _skip_appledouble(p.name):
                continue
            te = p.with_name(p.name.replace("_oof.npy", "_test.npy"))
            resid_candidates.append((p.stem.replace("_oof", ""), p, te))

    for name, op, tp in resid_candidates:
        pair = _load_pair(op, tp, n_tr, n_te)
        if pair is None:
            rows.append({"name": name, "status": "missing", "is_residual": True})
            print(f"MISS resid {name}", flush=True)
            continue
        oof, te = pair
        print(f"SCORE resid {name} …", flush=True)
        rec = evaluate_pure_residual(
            name, oof, te, y, fold, sf, z_fb, z_fb_te, z_twin, z_twin_te, Bo, Bt
        )
        slim = {k: rec[k] for k in rec if k not in ("u_oof", "u_test", "g_fb", "g_twin", "orth")}
        slim["g_fb_pos"] = rec["g_fb"]["pos"]
        slim["g_fb_e4"] = rec["g_fb"]["mean_e4"]
        slim["g_fb_w"] = rec["g_fb"]["w_fixed"]
        slim["g_twin_pos"] = rec["g_twin"]["pos"]
        slim["F2"] = rec["orth"]["F2_max_corr"]
        rows.append(slim)
        print(
            f"  → resid solo_proxy={rec['solo']:.5f} g0={rec['g0']} vault={rec['vault']} g1={rec['g1']} "
            f"r2={rec['r2_B152']:.3f} f1={rec['f1']} f2={rec['f2_ok']} w_fb={rec['g_fb']['w_fixed']:.4f} "
            f"pos={rec['g_fb']['pos']}/5 e4={rec['g_fb']['mean_e4']:.3f}",
            flush=True,
        )
        if rec.get("g1") or rec.get("vault") or rec.get("g0") or (rec["g_fb"]["pos"] >= 4 and rec["g_twin"]["pos"] >= 3):
            vectors.append((name, rec["u_oof"], rec["u_test"], float(rec["solo"]), rec))

    # ---- MultiVault synth over non-leaked ----
    # decorrelate + coordinate ascent on fb
    synth = {"status": "empty", "n": 0}
    cand1_path = None
    if vectors:
        # drop high-corr
        keep = [0]
        for j in range(1, len(vectors)):
            uj = unit(vectors[j][1])
            if max(abs(float(np.corrcoef(uj, unit(vectors[k][1]))[0, 1])) for k in keep) < 0.35:
                keep.append(j)
        V = [vectors[j] for j in keep]
        Uo = np.column_stack([unit(v[1]) for v in V])
        Ut = np.column_stack([unit(v[2]) for v in V])
        names = [v[0] for v in V]
        w = np.zeros(Uo.shape[1])
        for _ in range(3):
            for j in range(Uo.shape[1]):
                best, bw = -1e18, 0.0
                for cand in np.linspace(-0.06, 0.06, 13):
                    w2 = w.copy()
                    w2[j] = cand
                    d = auc(y, U(L(z_fb) + Uo @ w2)) - auc(y, z_fb)
                    if d > best:
                        best, bw = d, cand
                w[j] = bw
        syn_o = unit(Uo @ w)
        syn_t = unit(Ut @ w)

        def fast_signed(base, u):
            best_w, best_mean, best_pos, d_at = 0.0, -1e18, 0, None
            for ww in np.linspace(-0.06, 0.06, 25):
                if abs(ww) < 1e-12:
                    continue
                ds = [
                    (auc(y[fold == f], U(L(base[fold == f]) + ww * u[fold == f])) - auc(y[fold == f], base[fold == f]))
                    * 1e4
                    for f in range(5)
                ]
                ds = np.asarray(ds, float)
                if ds.mean() > best_mean:
                    best_mean, best_w, best_pos, d_at = float(ds.mean()), float(ww), int((ds > 0).sum()), ds
            se = d_at.std(ddof=1) / np.sqrt(5) if d_at is not None and d_at.std(ddof=1) > 1e-12 else 1e-12
            return {"mean_e4": best_mean, "pos": best_pos, "w_fixed": best_w, "t25": float(best_mean / se)}

        g_fb = fast_signed(z_fb, syn_o)
        g_twin = fast_signed(z_twin, syn_o)
        same_sign = np.sign(g_fb["w_fixed"]) == np.sign(g_twin["w_fixed"]) or abs(g_twin["w_fixed"]) < 1e-8
        g1 = g_fb["pos"] == 5 and g_twin["pos"] >= 4 and same_sign
        synth = {
            "status": "ok",
            "n": len(names),
            "pillars": names,
            "weights": w.tolist(),
            "g_fb": g_fb,
            "g_twin": g_twin,
            "same_sign": bool(same_sign),
            "g1": bool(g1),
            "delta_e4_fb": g_fb["mean_e4"],
        }
        print(
            f"SYNTH pillars={names} w={np.round(w,4).tolist()} g1={g1} "
            f"fb_pos={g_fb['pos']} twin_pos={g_twin['pos']} e4={g_fb['mean_e4']:.3f}",
            flush=True,
        )
        if g1 and NO_SUBMIT:
            w_star = abs(float(g_fb["w_fixed"])) * 0.85
            c1 = blend_ath(ath, syn_t, 0.60 * w_star)
            cand1_path = SUB / "sub_v36_2_CAND1_NOSUBMIT.csv"
            pd.DataFrame({"id": ids_te, "is_fraud": c1}).to_csv(cand1_path, index=False)
            synth["cand1"] = str(cand1_path)
            print(f"STAGED {cand1_path} (NO_SUBMIT)", flush=True)
        elif not g1:
            print("G1 FAIL — no CAND1 staged", flush=True)

    # ---- INTEL excerpt ----
    intel_lines = []
    if INTEL_MD.exists():
        for ln in INTEL_MD.read_text(errors="replace").splitlines():
            if ln.strip() and not ln.startswith("#"):
                intel_lines.append(ln.strip())
            if len(intel_lines) >= 12:
                break

    elapsed = time.time() - t0
    report = {
        "elapsed_s": elapsed,
        "NO_SUBMIT": True,
        "kaggle_submissions": 0,
        "rows": rows,
        "synth": {k: v for k, v in synth.items() if k not in ("g_fb", "g_twin")}
        | {
            "g_fb_pos": (synth.get("g_fb") or {}).get("pos"),
            "g_fb_e4": (synth.get("g_fb") or {}).get("mean_e4"),
            "g_twin_pos": (synth.get("g_twin") or {}).get("pos"),
            "g1": synth.get("g1"),
        },
        "intel_top": intel_lines[:12],
    }
    # scrub numpy from rows
    for r in report["rows"]:
        for k in list(r.keys()):
            if isinstance(r[k], np.ndarray):
                del r[k]
            elif isinstance(r[k], (np.floating, np.integer)):
                r[k] = float(r[k]) if isinstance(r[k], np.floating) else int(r[k])

    atomic_json(OUT, report)

    # markdown table
    lines = [
        "# GM V36.2 Morning Harvest Audit",
        f"- elapsed: {elapsed:.1f}s · NO_SUBMIT=True · kaggle_submits=0",
        "",
        "| name | status | solo | g0 | vault | f1 | f2 | r2 | disp | fb_pos | fb_e4 |",
        "|---|---|---:|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('name')} | {r.get('status')} | {r.get('solo')} | {r.get('g0')} | {r.get('vault')} | "
            f"{r.get('f1')} | {r.get('f2_ok')} | {r.get('r2_B152')} | {r.get('raw_disp')} | "
            f"{r.get('g_fb_pos')} | {r.get('g_fb_e4')} |"
        )
    lines += [
        "",
        "## MultiVault synth",
        f"- g1={synth.get('g1')} pillars={synth.get('pillars')} cand1={synth.get('cand1')}",
        "",
        "## INTEL (top)",
        *[f"- {x}" for x in intel_lines[:8]],
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"WROTE {OUT} + {OUT_MD} in {elapsed:.1f}s", flush=True)
    print("=== SUMMARY TABLE ===", flush=True)
    print("\n".join(lines[3:3 + len(rows) + 2]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
