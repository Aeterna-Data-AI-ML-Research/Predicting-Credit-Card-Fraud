#!/usr/bin/env python3
"""GM V31.0 — Zero-slot forensic autopsy of 0.95534 + OOF basis rebuild.

STRICT: NO_SUBMIT = True. Never calls kaggle competitions submit.
Writes logs/overnight/GM_V31_0_AUTOPSY_095534_REPORT.md
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

NO_SUBMIT = True
REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
OUT = Path("/tmp/s6e11_v31_autopsy")
REPORT = LOG / "GM_V31_0_AUTOPSY_095534_REPORT.md"
N_TR, N_TE = 699960, 299984
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
    return {
        "mean": float(d.mean()),
        "pos": int((d > 0).sum()),
        "t": float(d.mean() / se if se > 1e-12 else 0.0),
        "d": [round(float(x), 4) for x in d],
    }


def gs_project(v: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    u = unitize(v)
    for b in basis:
        b = unitize(b)
        u = unitize(u - np.dot(u, b) * b)
    return u


def r2_fit(X: np.ndarray, y: np.ndarray) -> float:
    lr = Ridge(alpha=1.0).fit(X, y)
    pred = lr.predict(X)
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-18))


def load_sub(path: Path, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path)
    col = "is_fraud" if "is_fraud" in df.columns else [c for c in df.columns if c != "id"][0]
    return df.set_index("id").loc[ids, col].to_numpy(float)


def load_pair(oof_p: Path, te_p: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not (oof_p.exists() and te_p.exists()):
        return None
    o = np.load(oof_p).ravel()[:N_TR].astype(float)
    t = np.load(te_p).ravel()[:N_TE].astype(float)
    if len(o) != N_TR or len(t) != N_TE or np.isnan(o).any() or np.isnan(t).any():
        return None
    return o, t


def _softmax2(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e[..., 1] / (e.sum(axis=-1) + 1e-12)


def stack_gpt2_med(dirp: Path, fold_id: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Decode (n_val, n_orders, 2) logits per fold → stitched OOF + mean test probs."""
    oofs = [dirp / f"oof_{i}.npy" for i in range(5)]
    tests = [dirp / f"test_{i}.npy" for i in range(5)]
    if not (all(p.exists() for p in oofs) and all(p.exists() for p in tests)):
        return None
    oof = np.full(N_TR, np.nan)
    for f in range(5):
        arr = np.load(oofs[f])
        te = np.where(fold_id == f)[0]
        if arr.ndim == 3 and arr.shape[0] == len(te):
            oof[te] = _softmax2(arr).mean(axis=1)
        else:
            return None
    t_parts = []
    for f in range(5):
        arr = np.load(tests[f])
        if arr.ndim == 3:
            t_parts.append(_softmax2(arr).mean(axis=1))
        else:
            t_parts.append(arr.ravel()[:N_TE])
    if not np.isfinite(oof).all():
        return None
    return oof.astype(float), np.mean(t_parts, axis=0).astype(float)


def main() -> None:
    assert NO_SUBMIT, "NO_SUBMIT must remain True"
    log("=== GM V31.0 autopsy — NO_SUBMIT=True ===")
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(REPO / "data/train.csv")
    te = pd.read_csv(REPO / "data/test.csv")
    y = tr["is_fraud"].to_numpy(float)
    ids = te["id"].to_numpy()
    fold_id = pd.read_csv(BANK / "folds.csv")["fold"].to_numpy()
    folds = folds_from(fold_id)

    # --- Live public vectors ---
    areeb_p = SUB / "sub_v30_PUB_areeb_95534.csv"
    gpw_p = SUB / "sub_v30_PUB_gpw_95533.csv"
    gpw36_p = Path("/tmp/s6e11_v31_autopsy/gpw_95536_out/submission.csv")
    if not areeb_p.exists() or not gpw_p.exists():
        raise SystemExit("missing local ATH CSVs")
    areeb = load_sub(areeb_p, ids)
    gpw = load_sub(gpw_p, ids)
    La, Lg = rank_logit(areeb), rank_logit(gpw)
    log(f"omr(areeb,gpw)={omr(areeb, gpw):.6f}e-4  corr={spearmanr(areeb, gpw).correlation:.8f}")
    if gpw36_p.exists():
        gpw36 = load_sub(gpw36_p, ids)
        log(f"NEW public goodpjw 0.95536 local: omr(areeb)={omr(gpw36, areeb):.6f} omr(gpw33)={omr(gpw36, gpw):.6f}")
    else:
        gpw36 = None
        log("gpw 0.95536 output not cached")

    # Reproduce Areeb residual50 math if step files exist
    step_n = Path("/tmp/s6e11_v30/gpw_95533/submission.csv")
    step_m = Path("/tmp/s6e11_v30/gpw_95533/submission_step_m.csv")
    mal = SUB / "sub_v30_PUB_malyshev.csv"
    # Malyshev as published was residual100 from areeb; for Areeb recipe F is older fusion.
    # Use gpw N/M from local v30 pin.
    recipe_ok = False
    if step_n.exists() and step_m.exists():
        N = rank_logit(load_sub(step_n, ids))
        M = rank_logit(load_sub(step_m, ids))
        # F from malyshev dataset LEADER/PROBE pins if present
        fus_candidates = list((REPO / "external/banks_v31/s6e11-nov20-late-public-residual").glob("*.csv"))
        F = None
        for p in fus_candidates:
            try:
                v = load_sub(p, ids)
                # fusion was 0.95527-ish; far from areeb
                if 0.5 < omr(v, areeb) < 50:
                    F = rank_logit(v)
                    log(f"fusion candidate {p.name} omr(areeb)={omr(v, areeb):.4f}")
                    break
            except Exception:
                pass
        if F is None and mal.exists():
            # fallback: cannot perfectly recover F; skip exact byte match
            log("Areeb F (Malyshev pre-areeb fusion) not found in banks_v31; math documented from source")
        else:
            d = N - M
            e = F - M
            beta = float((d @ e) / (e @ e))
            r = d - beta * e
            cand = rf(N + 0.5 * r)
            log(f"Areeb residual50 reproduce: beta={beta:.6f} omr(cand,areeb)={omr(cand, areeb):.8f}")
            recipe_ok = omr(cand, areeb) < 1e-3
    log(f"Areeb residual50 local reproduce ok={recipe_ok}")

    # --- Stale V22.5 Bt vs updated Bt_v31 ---
    Bt_names_v22 = ["gb_bag5", "gb_bag10", "lgb_diglead_l3", "realmlp_v1", "glm_dl", "ebm_all_bag"]
    Bt_v22 = []
    for name in Bt_names_v22:
        tp = BANK / f"test_{name}.npy"
        if tp.exists():
            Bt_v22.append(unitize(rank_logit(np.load(tp).ravel()[:N_TE])))
    Bt_v22_m = np.column_stack(Bt_v22)
    r2_v22 = r2_fit(Bt_v22_m, La)
    log(f"R² Ridge(Bt_v22 → L(areeb)) = {r2_v22:.6f}  (n_cols={Bt_v22_m.shape[1]})")

    # Bt_v31: bank models + public board ladder steps + RealMLP pins if present
    Bt_v31_cols = []
    Bt_v31_names = []
    for name in Bt_names_v22 + [
        "gbn_bag5_w512",
        "lgb_diglead_l4",
        "realmlp_te",
        "xgb_te_d2",
    ]:
        tp = BANK / f"test_{name}.npy"
        if tp.exists():
            Bt_v31_cols.append(unitize(rank_logit(np.load(tp).ravel()[:N_TE])))
            Bt_v31_names.append(f"bank:{name}")
    # public ladder steps
    for label, p in [
        ("gpw33", gpw_p),
        ("areeb34", areeb_p),
        ("step_i", Path("/tmp/s6e11_v30/gpw_95533/submission_step_i.csv")),
        ("step_m", Path("/tmp/s6e11_v30/gpw_95533/submission_step_m.csv")),
        ("step_n", Path("/tmp/s6e11_v30/gpw_95533/submission.csv")),
        ("base_95442", Path("/tmp/s6e11_v30/gpw_95533/submission_0p95442.csv")),
    ]:
        if p.exists():
            Bt_v31_cols.append(unitize(rank_logit(load_sub(p, ids))))
            Bt_v31_names.append(label)
    if gpw36 is not None:
        Bt_v31_cols.append(unitize(rank_logit(gpw36)))
        Bt_v31_names.append("gpw36")
    Bt_v31_m = np.column_stack(Bt_v31_cols)
    r2_v31 = r2_fit(Bt_v31_m, La)
    r2_v31_gpw = r2_fit(Bt_v31_m, Lg)
    log(f"R² Ridge(Bt_v31 → L(areeb)) = {r2_v31:.6f}  cols={len(Bt_v31_names)}")
    log(f"R² Ridge(Bt_v31 → L(gpw33)) = {r2_v31_gpw:.6f}")
    # R² of areeb on ONLY [gpw, step_m, step_n] — residual recipe span
    span = []
    for p in [gpw_p, Path("/tmp/s6e11_v30/gpw_95533/submission_step_m.csv"), Path("/tmp/s6e11_v30/gpw_95533/submission.csv")]:
        if p.exists():
            span.append(unitize(rank_logit(load_sub(p, ids))))
    if span:
        r2_span = r2_fit(np.column_stack(span), La)
        log(f"R² Ridge([gpw,step_m,step_n] → L(areeb)) = {r2_span:.6f}")

    # Why LOFO aborted: B_macro was Ridge(Bt_v22 → L(anchor)) with tiny residual variance in bank space
    lr22 = Ridge(alpha=10.0).fit(Bt_v22_m, La)
    B_macro_t = Bt_v22_m @ lr22.coef_ + lr22.intercept_
    resid = La - B_macro_t
    log(
        f"stale B_macro: R²={r2_v22:.4f}  resid_std={resid.std():.4f}  "
        f"max|B_macro_t|={np.max(np.abs(B_macro_t)):.4f} (NOT unitized — V23 bug)"
    )
    # On OOF side, same mapping using Bo
    Bo_v22 = []
    for name in Bt_names_v22:
        op = BANK / f"oof_{name}.npy"
        if op.exists():
            Bo_v22.append(unitize(L_fold(np.load(op).ravel()[:N_TR], fold_id)))
    Bo_v22_m = np.column_stack(Bo_v22)
    B_macro_o = Bo_v22_m @ lr22.coef_ + lr22.intercept_
    z_honest = rf(B_macro_o)
    log(f"z_honest_v31 (B_macro_o replay) OOF AUC={auc(y, z_honest):.6f}")

    # Honest CV-led file from gpw if present
    cv_led = Path("/tmp/s6e11_v30/gpw_95533/submission_cv_led.csv")
    if cv_led.exists():
        log(f"gpw submission_cv_led omr(areeb)={omr(load_sub(cv_led, ids), areeb):.4f} (CV-backed safer pick per goodpjw)")

    # --- Proprietary absence audit ---
    prop_specs = []
    # Prefer rebuilt caches from this autopsy; else decode live
    gpt2_cache = (
        Path("/tmp/s6e11_v31_autopsy/gpt2_med_oof_rebuilt.npy"),
        Path("/tmp/s6e11_v31_autopsy/gpt2_med_test_rebuilt.npy"),
    )
    if gpt2_cache[0].exists() and gpt2_cache[1].exists():
        prop_specs.append(("U_B_gpt2med_5f", np.load(gpt2_cache[0]), np.load(gpt2_cache[1])))
    else:
        gpt2 = stack_gpt2_med(REPO / "preds/gpt2_med_4ep", fold_id)
        if gpt2:
            prop_specs.append(("U_B_gpt2med_5f", gpt2[0], gpt2[1]))
    l7_cache = (
        Path("/tmp/s6e11_v31_autopsy/l7_s2026_oof_rebuilt.npy"),
        Path("/tmp/s6e11_v31_autopsy/l7_s2026_test_rebuilt.npy"),
    )
    if l7_cache[0].exists() and l7_cache[1].exists():
        prop_specs.append(("U_L7_s2026_5f", np.load(l7_cache[0]), np.load(l7_cache[1])))
    else:
        # raw L7 is (N,10,2) / (5,N,10,2) — decode if present
        op = REPO / "preds/l7_s2026_5f/l7_s2026_oof_5fold_raw.npy"
        tp = REPO / "preds/l7_s2026_5f/l7_s2026_test_5fold_raw.npy"
        if op.exists() and tp.exists():
            ao, at = np.load(op), np.load(tp)
            if ao.ndim == 3 and at.ndim == 4:
                prop_specs.append(
                    (
                        "U_L7_s2026_5f",
                        _softmax2(ao).mean(axis=1),
                        _softmax2(at).mean(axis=0).mean(axis=1),
                    )
                )
    log("NOTE: P4_L4.*_pure5f_* are residual-scale; skipped as standalone probs")
    for name, op, tp in [
        ("v10_ridge_resid", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_oof.npy", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_test.npy"),
    ]:
        pair = load_pair(op, tp)
        if pair and float(np.ptp(pair[0])) > 0.05:
            prop_specs.append((name, pair[0], pair[1]))
        elif pair:
            log(f"skip {name}: residual-scale ptp={float(np.ptp(pair[0])):g}")
    for wp in [
        REPO / "external/banks/goodpjw2008_s6e11-fraud-series-artifacts/wide_lm_stack_nested.npy",
        REPO / "external/nov08_part12/wide_lm_stack_nested.npy",
    ]:
        pair = load_pair(wp, wp.parent / "wide_lm_stack_test.npy")
        if pair:
            prop_specs.append(("wide_lm_stack_nested", pair[0], pair[1]))
            break

    basis_t = [unitize(La), unitize(Lg)]
    if gpw36 is not None:
        basis_t.append(unitize(rank_logit(gpw36)))
    # also project out B_macro_t unitized
    basis_t.append(unitize(B_macro_t))

    audit_rows = []
    for name, oof, test in prop_specs:
        solo = auc(y, oof)
        # raw corr to ATH
        corr_a = float(spearmanr(test, areeb).correlation)
        corr_g = float(spearmanr(test, gpw).correlation)
        # residual after GS vs B_v31 test basis
        u_t = gs_project(rank_logit(test), basis_t)
        # OOF residual vs B_macro for LOFO
        u_o = gs_project(L_fold(oof, fold_id), [unitize(B_macro_o)])
        # LOFO small-w on honest replay
        cand = rf(0.92 * rank_logit(z_honest) + 0.08 * L_fold(oof, fold_id))
        g_raw = h2h(cand, z_honest, y, folds)
        cand_u = rf(rank_logit(z_honest) + 0.05 * u_o)
        g_u = h2h(cand_u, z_honest, y, folds)
        # also vs gb_bag5 honest
        gb = np.load(BANK / "oof_gb_bag5.npy").ravel()[:N_TR]
        cand_gb = rf(0.92 * rank_logit(gb) + 0.08 * L_fold(oof, fold_id))
        g_gb = h2h(cand_gb, gb, y, folds)
        row = {
            "name": name,
            "solo_oof_auc": round(solo, 6),
            "spearman_test_vs_areeb": round(corr_a, 6),
            "spearman_test_vs_gpw": round(corr_g, 6),
            "omr_test_vs_areeb_e4": round(omr(test, areeb), 4),
            "abs_corr_u_vs_La": round(abs(float(np.corrcoef(u_t, unitize(La))[0, 1])), 8),
            "lofo_vs_Bmacro_pos": g_raw["pos"],
            "lofo_vs_Bmacro_mean_e4": round(g_raw["mean"], 3),
            "lofo_vs_Bmacro_t": round(g_raw["t"], 2),
            "lofo_orth_u_pos": g_u["pos"],
            "lofo_orth_u_mean_e4": round(g_u["mean"], 3),
            "lofo_orth_u_t": round(g_u["t"], 2),
            "lofo_vs_gb_pos": g_gb["pos"],
            "lofo_vs_gb_mean_e4": round(g_gb["mean"], 3),
            "lofo_vs_gb_t": round(g_gb["t"], 2),
            "absent_from_095534": abs(corr_a) < 0.9995,  # not a near-duplicate of ATH
        }
        audit_rows.append(row)
        log(
            f"PROP {name}: solo={solo:.5f} ρ(areeb)={corr_a:.4f} omr={omr(test, areeb):.2f}e-4 | "
            f"LOFO@gb {g_gb['pos']}/5 m={g_gb['mean']:+.2f}t={g_gb['t']:.1f} | "
            f"orth@Bmacro {g_u['pos']}/5 m={g_u['mean']:+.2f}"
        )

    # Local "member-step" toy: apply small proprietary residual on L(areeb) using LOFO-gated weight
    # (mechanism upgrade test — CV only, no submit)
    mech = []
    for name, oof, test in prop_specs:
        u_t = gs_project(rank_logit(test), [unitize(La), unitize(Lg)])
        u_o = gs_project(L_fold(oof, fold_id), [unitize(B_macro_o)])
        # search alpha on OOF vs B_macro
        best = None
        for a in [0.02, 0.05, 0.08, 0.12, 0.20]:
            cand = rf(rank_logit(z_honest) + a * u_o)
            g = h2h(cand, z_honest, y, folds)
            if best is None or (g["mean"], g["t"]) > (best["mean"], best["t"]):
                best = {**g, "alpha": a}
        mech.append({"name": name, **{k: best[k] for k in ("alpha", "mean", "pos", "t")}})
        log(f"MECH local member-step {name}: best α={best['alpha']} LOFO {best['pos']}/5 m={best['mean']:+.3f} t={best['t']:.2f}")

    # Save artifacts
    art = {
        "NO_SUBMIT": True,
        "r2_Bt_v22_to_areeb": r2_v22,
        "r2_Bt_v31_to_areeb": r2_v31,
        "r2_Bt_v31_to_gpw": r2_v31_gpw,
        "Bt_v31_names": Bt_v31_names,
        "areeb_residual50_reproduced": recipe_ok,
        "proprietary_audit": audit_rows,
        "local_member_step_mech": mech,
        "new_public_above_ath": {
            "goodpjw_0.95536": gpw36 is not None,
            "note": "Kernel slug renamed to …-lb-0-95536; Step O member-step from N+1.5r file",
        },
    }
    (OUT / "autopsy_metrics.json").write_text(json.dumps(art, indent=2))
    log(f"wrote {OUT / 'autopsy_metrics.json'}")

    # --- Report ---
    absent = [r for r in audit_rows if r["absent_from_095534"]]
    strong = [r for r in audit_rows if r["lofo_vs_gb_pos"] >= 4 and r["lofo_vs_gb_mean_e4"] > 0]
    md = f"""# GM V31.0 — Forensic Autopsy of `0.95534` & OOF Basis Rebuild

**NO_SUBMIT = True** · Generated from `scripts/gm_v31_0_autopsy_095534_and_rebuild_oof.py`  
Sources pulled to `/tmp/s6e11_v31_autopsy/` · Datasets to `external/banks_v31/`

## Executive verdict

The leap **0.95426 → 0.95534 (~+108e-5)** is **not** a new raw feature / id-leak / orig20k lookup. It is a **public-board member-step ladder** (goodpjw) starting from yekenot RealMLP + public OOF banks, then a **+1e-5 orthogonal residual half-step** (Areeb) on top of step N.

| File | LB | What it actually is |
|------|-----|---------------------|
| goodpjw board-model step N | 0.95533 | Member-step: `L_out = (1−Σw)·L(base) + Σ w_k L(member_k)` with ~152 members; weights chosen by an offline **board model** (regression on ~167–173 scored public CSVs) + nested CV — **not** label pseudo-labeling |
| Areeb residual50 | **0.95534** | `d=L(N)−L(M)`, `e=L(F)−L(M)`, `r=d−βe`, `P=rank(L(N)+0.5 r)` with β≈0.011 → r ≈ step-N direction |
| Malyshev residual100 | 0.95534 | `L(P)+1.0·(L(P)−L(Q))` + soft zone floors — continuation of Areeb, not a new engine |
| goodpjw step O (overnight) | **0.95536** | New public peak (as of autopsy): member-step from N+1.5r file; **do not confuse with our locked 0.95534** |

## 1. Exact recipes (from pulled source)

### 1.1 Areeb (`areebbb/s6e11-which-signals-rank-fraud-lb-0-95534`)

- `dataset_sources`: **[]** (no OOF datasets)
- `kernel_sources`: Malyshev fusion, Asterios, goodpjw board-model
- Math (logit-rank space):

```text
N = L(goodpjw submission.csv)          # step N, LB 0.95533
M = L(goodpjw submission_step_m.csv)   # step M, LB 0.95531
F = L(Malyshev fusion submission)      # comparison direction ~0.95527
d = N - M
e = F - M
β = (d·e)/(e·e)                        # reported ≈ 0.011
r = d - β e                            # ≈ step-N direction
P = rank( N + 0.5 * r )                # scored 0.95534 (+1e-5)
```

Variants that tied/lost: `N+0.25 d` → 0.95533; `0.85 N + 0.15 F` → 0.95533.

### 1.2 goodpjw board-model (`…-lb-0-95536`, was `…-95533`)

- **What “board-model + RealMLP” means:**
  1. **Base:** public 0.95442 stack built from yekenot RealMLP (PyTabKit + PyTorch) + other public bases.
  2. **Members:** ~87–152 OOF/test streams from `s6e11-oof-bank`, `fraud-series-npz`, Aman Atar Q17 bundle, busyaprime TabM/BPE, Weni BPE, wide LM stack, etc.
  3. **Member step:** frozen weights `STEP_*` applied as  
     `apply(w, base) = (1−Σw)·L(base) + Σ_k w_k·L(member_k)` then zone-aware re-rank (`Zr`).
  4. **Board model (not shipped):** offline regressor fitted on **scored public CSVs** to **forecast** which candidate step gains public LB; nested CV on replayed OOF used as a second gate. This is **score-forecasting / step selection**, not distillation of `is_fraud` pseudo-labels onto test.
- Ladder (public points, 1 pt = 1e-5): RealMLP 0.95442 → … → M 0.95531 → **N 0.95533** → (others) → **O 0.95536**.
- Honest CV file they recommend for private: `submission_step_i.csv` / `submission_cv_led.csv` (OOF-positive), while late steps N/O are **public-led** and often **negative** on replayed OOF.

### 1.3 Malyshev

Pure public residual continuation of Areeb vs goodpjw + soft structural zero-fraud zones. SHA-locked recipe `residual100`.

### 1.4 Datasets downloaded to `external/banks_v31/`

- `goodpjw2008/s6e11-board-model-pinned-inputs` (Weni BPE OOF/test, TabM, Davut bag, Malyshev step-L file, …)
- `malyshevdanil/s6e11-nov20-late-public-residual`
- `amanatar/s6e11-rank1-q17-meta-bundle`

Also listed top-30 kernels (scoreDescending): **goodpjw 0.95536 now #1**; Bhuvan “40 scored files” 0.95534; Rishabh GS N+1.5r 0.95535 referenced in goodpjw text.

## 2. Why `gm_v23_0` LOFO aborted

### Measured R² (this run)

| Basis | Target | R² |
|-------|--------|-----|
| Stale `Bt_v22` (6 bank models) | `L(areeb_095534)` | **{r2_v22:.4f}** |
| Updated `Bt_v31` (bank + ladder steps + ATH) | `L(areeb)` | **{r2_v31:.4f}** |
| `Bt_v31` | `L(gpw_095533)` | **{r2_v31_gpw:.4f}** |

### Root causes

1. **Wrong manifold:** V23 built `B_macro = Ridge(Bt_v22 → L(anchor))` after swapping anchor to Areeb. Areeb lives almost entirely in the **public ladder span** (steps M/N + tiny residual), not in the 6-column V22 bank. Residual directions from local OOFs, after GS against that mis-specified `B_macro`, had **no LOFO edge** (`pos ≤ 1/5`).
2. **No OOF for ATH:** Areeb/gpw late steps are **test-only public constructions**. Projecting local `u` orth to `L(areeb)` on test while gating LOFO on `B_macro_o` (bank replay) **mismatches train/test bases** → gates kill every `u_k`.
3. **Bug hygiene:** V23 logged `max|B_macro_t|=13` because it never unitized `B_macro_t` before the orth check (cosmetic), but the real failure was the LOFO gate on a stale control.

### Fix for V31+ (implemented here as measurement protocol, not submit)

- Treat `B_v31_test = span{{L(areeb), L(gpw), L(step_m), L(step_n), …}}`.
- Gate LOFO on **honest** OOF anchor (`gb_bag5` or gpw `submission_cv_led` replay), **not** on Ridge-to-Areeb.
- On test, require `|corr(u, L(areeb))|` and `|corr(u, L(gpw))|` ~ 0 after GS before any future star probe.

## 3. Proprietary engines vs `0.95534` basis

Absent = Spearman vs Areeb not ≈ 1 (not a near-duplicate of the public ATH vector).

| Engine | Solo OOF AUC | ρ(test, areeb) | omr(areeb)×1e4 | LOFO vs gb 5/5? | Notes |
|--------|-------------:|---------------:|---------------:|-----------------|-------|
"""
    for r in audit_rows:
        md += (
            f"| `{r['name']}` | {r['solo_oof_auc']:.5f} | {r['spearman_test_vs_areeb']:.4f} | "
            f"{r['omr_test_vs_areeb_e4']:.2f} | {r['lofo_vs_gb_pos']}/5 m={r['lofo_vs_gb_mean_e4']:+.2f} | "
            f"{'ABSENT from ATH' if r['absent_from_095534'] else 'near-ATH'} |\n"
        )
    md += f"""

**Count absent from 0.95534 manifold:** {len(absent)} / {len(audit_rows)}  
**Count with positive LOFO vs gb_bag5 (pos≥4, mean>0):** {len(strong)}

### Local upgrade of their mechanism (CV only)

Same geometry as a member-step / residual add-on: `z_honest + α · GS(u | B_macro)` with α grid-searched on 5-fold H2H (no submit):

| Proprietary u | best α | LOFO pos | mean Δ×1e4 | t |
|---------------|-------:|---------:|-----------:|--:|
"""
    for m in mech:
        md += f"| `{m['name']}` | {m['alpha']} | {m['pos']}/5 | {m['mean']:+.3f} | {m['t']:.2f} |\n"

    md += f"""

Interpretation: if a row shows **pos≥4 and mean>0** on honest OOF after orth to bank `B_macro`, that proprietary channel is a **candidate** to plug into a goodpjw-style member-step **with our stronger LMs included** — but only after rebuilding an OOF replay of the 0.95534 ladder (or staying on `submission_cv_led` / step I for private-safe blends).

## 4. What we did *not* do

- **No Kaggle submissions** (`NO_SUBMIT=True`).
- Did not shuffle or re-upload public CSVs.
- Did not claim private-LB gains from public-led steps N/O (goodpjw themselves flag nested CV negative for those).

## 5. Artifacts

- Kernel sources: `/tmp/s6e11_v31_autopsy/{{areeb,gpw_95536,malyshev,bhuvan,asterios}}/`
- Metrics JSON: `/tmp/s6e11_v31_autopsy/autopsy_metrics.json`
- Datasets: `external/banks_v31/`
- This report: `logs/overnight/GM_V31_0_AUTOPSY_095534_REPORT.md`

## Log tail

```
{chr(10).join(LINES[-40:])}
```
"""
    REPORT.write_text(md)
    log(f"REPORT -> {REPORT}")
    log("DONE (NO_SUBMIT)")


if __name__ == "__main__":
    main()
