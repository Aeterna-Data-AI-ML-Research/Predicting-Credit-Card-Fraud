#!/usr/bin/env python3
"""GM V16.3 — Slot2 Family E 0.50e-4 submit + true 0.95243 ATH orthogonal screen.

Step1: live submit (optional --skip-submit). Step2: NO_SUBMIT research on true build_ath_proxy().
Do NOT submit V16.2 G26cal / digit4leaf (z_ATH proxy bug).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
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

REPO = Path(os.environ.get("V16_REPO", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
TMP15 = Path(os.environ.get("V15_TMP", "/tmp/s6e11_v15"))
OUT = Path(os.environ.get("V16_OUT", "/tmp/s6e11_v16/out"))
SUB = REPO / "submissions"
REPORT = REPO / "logs/overnight/GM_V16_3_SLOT2_AND_TRUE_ATH_SCREEN.md"
LIVE095388 = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT1 = SUB / "sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv"
SLOT2 = SUB / "sub_v16_3_LIVE_slot2_FamilyE_0p50e4.csv"
KAG = os.environ.get("KAGGLE_BIN", "/Users/mz/Library/Python/3.14/bin/kaggle")
COMP = "playground-series-s6e11"
G26_DIR = Path("/tmp/s6e11_v16/g26")

W_TRUE = [0.01045, 0.01383, 0.01655, 0.0200, 0.0250, 0.0300, 0.0350]


def log(msg: str) -> None:
    print(msg, flush=True)


def L(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    r = (rankdata(v, method="average") - 0.5) / len(v)
    r = np.clip(r, 1e-12, 1 - 1e-12)
    return np.log(r / (1 - r))


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def qlock(raw, ref):
    raw = np.asarray(raw, float).ravel()
    ref = np.asarray(ref, float).ravel()
    out = np.empty_like(ref)
    out[np.argsort(raw)] = np.sort(ref)
    return out


def env_kaggle(args):
    env = os.environ.copy()
    for k in list(env):
        if "proxy" in k.lower():
            env.pop(k, None)
    r = subprocess.run([KAG, *args], capture_output=True, text=True, env=env)
    return (r.stdout or "") + (r.stderr or "")


def build_family_E_test(live):
    R3_te = np.load(TMP15 / "out/R3_test_folds.npy")
    R3t = R3_te.mean(0) if R3_te.shape[0] == 5 else R3_te.mean(1)
    l7f_te = np.load(TMP15 / "banks/lm7_test_5fold.npy").ravel()
    live_L = L(live)
    w_tail = 0.5 + 0.5 * expit((live_L - np.median(live_L)) / (np.std(live_L) + 1e-12))
    return unitize((L(R3t) - L(l7f_te)) * w_tail)


def step1_build_and_submit(skip_submit: bool) -> dict:
    live_df = pd.read_csv(LIVE095388)
    ids = live_df["id"].to_numpy()
    live = live_df["is_fraud"].to_numpy(float)
    slot1 = pd.read_csv(SLOT1).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    dE = build_family_E_test(live)
    target = 0.5000e-4
    lo, hi = 0.0, 8.0
    best_w = best_cand = best_d = None
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        cand = qlock(L(live) + mid * dE, live)
        d = 1.0 - float(spearmanr(live, cand).correlation)
        best_w, best_cand, best_d = mid, cand, d
        if d > target:
            hi = mid
        else:
            lo = mid
    err = abs(best_d - target)
    inc = 1.0 - float(spearmanr(slot1, best_cand).correlation)
    assert len(best_cand) == 299984 and np.isfinite(best_cand).all()
    assert err < 0.002e-4, (best_d, err)
    pd.DataFrame({"id": ids, "is_fraud": best_cand}).to_csv(SLOT2, index=False)
    out = {
        "file": str(SLOT2),
        "w": best_w,
        "1mrho_LIVE": best_d,
        "1mrho_Slot1": inc,
        "err": err,
        "publicScore": None,
        "submitted": False,
    }
    log(f"Slot2 built w={best_w:.6f} 1-ρLIVE={best_d:.6e} 1-ρSlot1={inc:.6e}")
    if skip_submit:
        poll = env_kaggle(["competitions", "submissions", "-c", COMP])
        (OUT / "v16_3_poll.txt").write_text(poll)
        for line in poll.splitlines():
            if "sub_v16_3_LIVE_slot2" in line and "COMPLETE" in line.upper():
                for tok in line.split():
                    if tok.startswith("0.95") and len(tok) <= 8:
                        out["publicScore"] = float(tok)
                        out["submitted"] = True
                        break
                break
        log(f"Slot2 score={out['publicScore']}")
        return out

    msg = (
        "v16.3 Slot2: Step winning 5-fold L7_s2026 Family E from 0.20e-4 (0.95389) "
        "to 0.50e-4 (w=0.0165, inc 1-rho=0.068e-4 vs Slot1)"
    )
    txt = env_kaggle(["competitions", "submit", "-c", COMP, "-f", str(SLOT2), "-m", msg])
    log(txt[-400:])
    out["submitted"] = "Successfully submitted" in txt
    for _ in range(12):
        time.sleep(25)
        poll = env_kaggle(["competitions", "submissions", "-c", COMP])
        (OUT / "v16_3_poll.txt").write_text(poll)
        for line in poll.splitlines():
            if "sub_v16_3_LIVE_slot2" in line and "COMPLETE" in line.upper():
                for tok in line.split():
                    if tok.startswith("0.95") and len(tok) <= 8:
                        out["publicScore"] = float(tok)
                        break
                break
        if out["publicScore"] is not None:
            break
    log(f"Slot2 publicScore={out['publicScore']}")
    return out


def load_v13_build_ath():
    path = REPO / "scripts/gm_v13_2_overnight_check_and_stageB_v14.py"
    spec = importlib.util.spec_from_file_location("gm_v13_2", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gm_v13_2"] = mod
    spec.loader.exec_module(mod)
    pack = mod.build_ath_proxy()
    auc = float(pack["auc_ath_oof"])
    assert auc > 0.9523, f"ATH proxy AUC={auc} — not true 0.95243 build!"
    log(f"TRUE z_ATH OOF AUC={auc:.7f} PASS (>0.9523)")
    return pack


def orient(u, z, y, te0):
    if roc_auc_score(y[te0], z[te0] + 0.01 * u[te0]) < roc_auc_score(y[te0], z[te0] - 0.01 * u[te0]):
        return -u
    return u


def screen_on(name, y, z_base, u, folds, w_grid=None):
    te0 = folds[0][1]
    u = orient(unitize(u), z_base, y, te0)
    base = [roc_auc_score(y[te], z_base[te]) for _, te in folds]
    best = None
    for w in w_grid or (0.006, 0.009, 0.012, 0.015, 0.020, 0.025, 0.030):
        d = [(roc_auc_score(y[te], z_base[te] + w * u[te]) - base[i]) * 1e5 for i, (_, te) in enumerate(folds)]
        row = dict(w=w, pos=int(sum(x > 0 for x in d)), mean_e5=float(np.mean(d)), min_e5=float(np.min(d)), folds=[round(x, 3) for x in d])
        if best is None or (row["pos"], row["mean_e5"]) > (best["pos"], best["mean_e5"]):
            best = row
    # sf25 micro @0.01
    deltas = []
    for _, te in folds:
        yf, zb, uf = y[te], z_base[te], u[te]
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
    raw_disp = 1.0  # unitized OOF-only here; caller may override
    gate = best["pos"] >= 5 and int((d25 > 0).sum()) >= 18 and tstat >= 2.0 and 0.95 <= raw_disp <= 1.05
    return {
        "name": name,
        "main": best,
        "sf25": {"pos": int((d25 > 0).sum()), "t": tstat, "mean_e5": mean, "wilcox": float(wp), "ci90": ci},
        "raw_disp": raw_disp,
        "gate": gate,
        "u": u,
    }


def eval_w_on_true(name, y, z, u, folds, ws):
    rows = []
    base = [roc_auc_score(y[te], z[te]) for _, te in folds]
    for w in ws:
        d = [(roc_auc_score(y[te], z[te] + w * u[te]) - base[i]) * 1e5 for i, (_, te) in enumerate(folds)]
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
        rows.append(
            {
                "w": w,
                "pos": int(sum(x > 0 for x in d)),
                "mean_e5": float(np.mean(d)),
                "min_e5": float(np.min(d)),
                "folds": [round(x, 3) for x in d],
                "sf25": int((d25 > 0).sum()),
                "t": tstat,
            }
        )
    return rows


def lofo_orth(raw, B, folds):
    resid = np.zeros(len(raw))
    for tr, te in folds:
        lr = LinearRegression().fit(B[tr], raw[tr])
        resid[te] = raw[te] - lr.predict(B[te])
    return unitize(resid)


def build_families_oof(y, folds, fold_id, z_ath, d1):
    l7f = np.load(TMP15 / "out/l7f_oof.npy").ravel()
    lm = np.load(TMP15 / "out/lm_oof.npy").ravel()
    R3 = np.load(TMP15 / "out/R3_oof.npy").ravel()
    R2 = np.load(TMP15 / "out/R2_oof.npy").ravel()
    Ls, L7, Lm = L(R3), L(l7f), L(lm)
    dA = unitize(Ls - L7)
    eta = z_ath
    U = expit((eta - np.median(eta)) / (np.std(eta) + 1e-12))
    w_tail = 0.5 + 0.5 * U
    raw_e = (Ls - L7) * w_tail
    B = np.column_stack([unitize(z_ath), unitize(d1)])
    dE = unitize(raw_e - LinearRegression().fit(B, raw_e).predict(B))
    te0 = folds[0][1]
    dA = orient(dA, z_ath, y, te0)
    dE = orient(dE, z_ath, y, te0)
    # curvature R2 - R3
    d_curv = unitize(L(R2) - L(R3))
    d_curv = orient(d_curv, z_ath, y, te0)
    return {"dA": dA, "dE": dE, "d_curv": d_curv, "l7f": l7f, "lm": lm, "R3": R3, "R2": R2}


def step2_true_ath_screen(pack) -> dict:
    log("=== STEP 2: true 0.95243 ATH screen (NO_SUBMIT) ===")
    y = pack["y"]
    folds = pack["folds"]
    fold_id = pack["fold_id"]
    z_ath = pack["z_ATH_o"]
    d1 = pack["d1o"]
    Z = pack["Z"]
    names = pack["names"]
    XO = pack["XO"]

    fam = build_families_oof(y, folds, fold_id, z_ath, d1)
    z_s1 = z_ath + 0.01045 * fam["dE"]
    log(f"  z_ATH AUC={roc_auc_score(y, z_ath):.7f}  z_S1 AUC={roc_auc_score(y, z_s1):.7f}")

    # Fine grid on TRUE z_ATH
    grid = {
        "E": eval_w_on_true("E", y, z_ath, fam["dE"], folds, W_TRUE),
        "A": eval_w_on_true("A", y, z_ath, fam["dA"], folds, W_TRUE),
    }
    for key in ("E", "A"):
        peak = max(grid[key], key=lambda r: (r["pos"], r["mean_e5"]))
        log(
            f"  TRUE {key} peak w={peak['w']} pos={peak['pos']}/5 mean={peak['mean_e5']:.3f}e-5 "
            f"sf={peak['sf25']}/25 t={peak['t']:.2f}"
        )

    # Full 11-vector basis for lofo_orth
    # [z_ATH, d1, Family_E, MS, G26, LM, L7F, GPD, MLP, GB] — use available
    basis_cols = [unitize(z_ath), unitize(d1), fam["dE"]]
    basis_names = ["z_ATH", "d1_v4", "Family_E"]

    def add_basis(nm, arr):
        a = np.asarray(arr, float).ravel()
        if len(a) != len(y):
            return
        basis_cols.append(unitize(a))
        basis_names.append(nm)

    # MS
    if (REPO / "preds/P_super_v4_oof.npy").exists():
        add_basis("MS", L(np.load(REPO / "preds/P_super_v4_oof.npy").ravel()))
    # G26
    g26_p = G26_DIR / "gpt2_s2026_oof_5fold.npy"
    if g26_p.exists():
        add_basis("G26", L(np.load(g26_p).ravel()[: len(y)]))
    elif "gpt2_seed2026" in names:
        add_basis("G26", XO[:, names.index("gpt2_seed2026")])
    add_basis("LM", L(fam["lm"]))
    add_basis("L7F", L(fam["l7f"]))
    # GPD / MLP / GB from bank or preds
    for p, nm in [
        (REPO / "preds/ladder/glm_quad_oof.npy", "GPD"),
        (REPO / "preds/mlp_oof.npy", "MLP"),
        (REPO / "preds/gb_gpt2_bag5_oof.npy", "GB"),
        (REPO / "external_bank/goodpjw_artifacts/gb_bag5_oof.npy", "GB"),
    ]:
        if p.exists() and nm not in basis_names:
            add_basis(nm, np.load(p).ravel()[: len(y)])
    # Q17 gb_bag5 / daoviet as GPD-ish if missing
    if "GPD" not in basis_names and "daoviet_gpdvae" in names:
        add_basis("GPD", XO[:, names.index("daoviet_gpdvae")])
    if "GB" not in basis_names and "gb_bag5" in names:
        add_basis("GB", XO[:, names.index("gb_bag5")])
    if "MLP" not in basis_names and "yekenot_realmlp" in names:
        add_basis("MLP", XO[:, names.index("yekenot_realmlp")])

    B11 = np.column_stack(basis_cols)
    log(f"  basis ({B11.shape[1]}): {basis_names}")

    addons = []

    # Add-On 1: R2_bayes - R3_antithetic curvature
    raw = fam["d_curv"]
    u = lofo_orth(raw, B11, folds)
    # raw_disp proxy using test if available
    scr = screen_on("R2_minus_R3_curv", y, z_s1, u, folds)
    R2t = np.load(TMP15 / "out/R2_test_folds.npy")
    R3t = np.load(TMP15 / "out/R3_test_folds.npy")
    R2t = R2t.mean(0) if R2t.shape[0] == 5 else R2t.mean(1)
    R3t = R3t.mean(0) if R3t.shape[0] == 5 else R3t.mean(1)
    ut = unitize(L(R2t) - L(R3t))
    scr["raw_disp"] = float((ut.std() + 1e-12) / (u.std() + 1e-12))
    scr["gate"] = scr["main"]["pos"] >= 5 and scr["sf25"]["pos"] >= 18 and scr["sf25"]["t"] >= 2.0 and 0.95 <= scr["raw_disp"] <= 1.05
    addons.append({k: scr[k] for k in ("name", "main", "sf25", "raw_disp", "gate")})
    log(f"  {scr['name']}: pos={scr['main']['pos']}/5 sf={scr['sf25']['pos']}/25 t={scr['sf25']['t']:.2f} disp={scr['raw_disp']:.3f} gate={scr['gate']}")

    # Add-On 2: Q17 / alex streams
    q17_targets = [
        "yekenot_realmlp",
        "alex_glm_gen",
        "sadam_v4_cat",
        "megayak_shallow_tree",
        "sameer_glm10f",
        "gpt2_seed2026",
        "gb_bag5",
        "daoviet_gpdvae",
    ]
    for sn in q17_targets:
        if sn not in names:
            continue
        raw = XO[:, names.index(sn)]
        u = lofo_orth(raw, B11, folds)
        # test disp
        XT = pack["XT"]
        ut = unitize(XT[:, names.index(sn)])
        # orth on test vs LIVE basis approx — use std ratio of unitized residual vs oof
        scr = screen_on(f"Q17_{sn}", y, z_s1, u, folds)
        scr["raw_disp"] = float((ut.std() + 1e-12) / (u.std() + 1e-12))
        scr["gate"] = (
            scr["main"]["pos"] >= 5
            and scr["sf25"]["pos"] >= 18
            and scr["sf25"]["t"] >= 2.0
            and 0.95 <= scr["raw_disp"] <= 1.05
        )
        addons.append({k: scr[k] for k in ("name", "main", "sf25", "raw_disp", "gate")})
        log(
            f"  Q17_{sn}: pos={scr['main']['pos']}/5 @{scr['main']['w']} "
            f"sf={scr['sf25']['pos']}/25 t={scr['sf25']['t']:.2f} disp={scr['raw_disp']:.3f} gate={scr['gate']}"
        )

    # alexchilton a few
    alex = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
    if alex.exists():
        for p in sorted(alex.glob("oof_*.npy"))[:6]:
            if p.name.startswith("._"):
                continue
            o = np.load(p).ravel()
            if len(o) != len(y):
                continue
            u = lofo_orth(L(o), B11, folds)
            tp = alex / p.name.replace("oof_", "test_")
            disp = 1.0
            if tp.exists():
                disp = float((unitize(np.load(tp).ravel()).std() + 1e-12) / (u.std() + 1e-12))
            scr = screen_on(f"alex_{p.stem[-8:]}", y, z_s1, u, folds)
            scr["raw_disp"] = disp
            scr["gate"] = (
                scr["main"]["pos"] >= 5
                and scr["sf25"]["pos"] >= 18
                and scr["sf25"]["t"] >= 2.0
                and 0.95 <= scr["raw_disp"] <= 1.05
            )
            addons.append({k: scr[k] for k in ("name", "main", "sf25", "raw_disp", "gate")})
            log(
                f"  alex_{p.stem[-8:]}: pos={scr['main']['pos']}/5 sf={scr['sf25']['pos']}/25 "
                f"t={scr['sf25']['t']:.2f} gate={scr['gate']}"
            )

    # Add-On 3: d1 micro weight polish on top of Family E (z = MIX + w_d1*d1 + 0.01045*E)
    # Here z_ath already has 0.03*d1 baked in. Polish = extra Δw * d1 on z_S1.
    d1_rows = []
    base = [roc_auc_score(y[te], z_s1[te]) for _, te in folds]
    for dw in (-0.004, +0.003, +0.006):
        z = z_s1 + dw * unitize(d1)
        d = [(roc_auc_score(y[te], z[te]) - base[i]) * 1e5 for i, (_, te) in enumerate(folds)]
        # also sf25 of dw*d1 direction
        u = orient(unitize(d1), z_s1, y, folds[0][1])
        deltas = []
        for _, te in folds:
            yf, zb, uf = y[te], z_s1[te], u[te]
            for _, ste in StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(yf)), yf):
                b = roc_auc_score(yf[ste], zb[ste])
                # evaluate at |dw| scale
                deltas.append((roc_auc_score(yf[ste], zb[ste] + abs(dw) * np.sign(dw) * uf[ste]) - b) * 1e5)
        d25 = np.asarray(deltas, float)
        mean = float(d25.mean())
        se = float(d25.std(ddof=1) / np.sqrt(len(d25)))
        tstat = float(mean / (se + 1e-18))
        row = {
            "dw": dw,
            "w_d1_effective": 0.030 + dw,
            "pos": int(sum(x > 0 for x in d)),
            "mean_e5": float(np.mean(d)),
            "min_e5": float(np.min(d)),
            "folds": [round(x, 3) for x in d],
            "sf25": int((d25 > 0).sum()),
            "t": tstat,
        }
        d1_rows.append(row)
        log(f"  d1 Δw={dw:+.3f} (w={0.03+dw:.3f}): pos={row['pos']}/5 mean={row['mean_e5']:.3f}e-5 sf={row['sf25']}/25 t={row['t']:.2f}")

    return {
        "auc_ath": float(roc_auc_score(y, z_ath)),
        "auc_s1": float(roc_auc_score(y, z_s1)),
        "basis_names": basis_names,
        "grid": grid,
        "addons": addons,
        "d1_polish": d1_rows,
    }


def write_report(slot2, screen):
    lines = []
    lines.append("# GM V16.3 — Slot2 Family E 0.50e-4 + True 0.95243 ATH Screen")
    lines.append("")
    lines.append("**0 pods · Do NOT submit V16.2 G26cal / digit4leaf (weak z_ATH proxy bug)**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 1 — Live Slot 2")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|------|------:|")
    lines.append(f"| File | `{Path(slot2['file']).name}` |")
    lines.append(f"| w | {slot2['w']:.6f} |")
    lines.append(f"| 1−ρ(LIVE_095388) | **{slot2['1mrho_LIVE']:.6e}** |")
    lines.append(f"| 1−ρ(Slot1_095389) | **{slot2['1mrho_Slot1']:.6e}** ({slot2['1mrho_Slot1']*1e4:.4f}e-4) |")
    lines.append(f"| Public score | **{slot2.get('publicScore')}** |")
    if slot2.get("publicScore") is not None:
        lines.append(f"| Δ vs Slot1 0.95389 | {(slot2['publicScore']-0.95389)*10000:+.1f} bps |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 2 — True `build_ath_proxy()` screen (`NO_SUBMIT`)")
    lines.append("")
    lines.append(f"- Asserted true ATH OOF AUC = **{screen['auc_ath']:.7f}** (>0.9523)")
    lines.append(f"- Slot1 OOF `z_S1 = z_ATH + 0.01045·Family_E` AUC = **{screen['auc_s1']:.7f}**")
    lines.append(f"- Full basis: `{screen['basis_names']}`")
    lines.append("")
    lines.append("### True Family E / A fine grid on z_ATH 0.95243")
    lines.append("")
    for key in ("E", "A"):
        lines.append(f"#### `{key}`")
        lines.append("")
        lines.append("| w | pos/5 | meanΔ e-5 | min e-5 | folds | sf25@0.01 | t |")
        lines.append("|--:|------:|----------:|--------:|-------|----------:|--:|")
        for r in screen["grid"][key]:
            lines.append(
                f"| {r['w']:.5f} | {r['pos']}/5 | {r['mean_e5']:.3f} | {r['min_e5']:.3f} | "
                f"`{r['folds']}` | {r['sf25']}/25 | {r['t']:.2f} |"
            )
        lines.append("")
    lines.append("### Orthogonal add-ons on z_S1 (full-basis lofo_orth)")
    lines.append("")
    lines.append("| name | pos/5 | w* | meanΔ e-5 | sf25 | t | raw_disp | gate |")
    lines.append("|------|------:|---:|----------:|-----:|--:|---------:|:----:|")
    for a in screen["addons"]:
        lines.append(
            f"| `{a['name']}` | {a['main']['pos']}/5 | {a['main']['w']} | {a['main']['mean_e5']:.3f} | "
            f"{a['sf25']['pos']}/25 | {a['sf25']['t']:.2f} | {a['raw_disp']:.3f} | **{a['gate']}** |"
        )
    lines.append("")
    lines.append("### d1_v4 micro weight polish on z_S1")
    lines.append("")
    lines.append("| Δw | w_d1 eff | pos/5 | meanΔ e-5 | min | folds | sf25 | t |")
    lines.append("|---:|---------:|------:|----------:|----:|-------|-----:|--:|")
    for r in screen["d1_polish"]:
        lines.append(
            f"| {r['dw']:+.3f} | {r['w_d1_effective']:.3f} | {r['pos']}/5 | {r['mean_e5']:.3f} | "
            f"{r['min_e5']:.3f} | `{r['folds']}` | {r['sf25']}/25 | {r['t']:.2f} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- Slot2 live result logged above.")
    lines.append("- V16.2 G26cal/digit **not submitted** (false orthogonality under weak z_ATH).")
    lines.append("- Next add-ons must clear gates on **true** z_S1 with full-basis lofo_orth.")
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    log(f"wrote {REPORT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-submit", action="store_true")
    ap.add_argument("--skip-screen", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (REPO / "logs/overnight").mkdir(parents=True, exist_ok=True)

    slot2 = step1_build_and_submit(skip_submit=args.skip_submit)
    if args.skip_screen:
        screen = {}
    else:
        pack = load_v13_build_ath()
        screen = step2_true_ath_screen(pack)
    write_report(slot2, screen if screen else {"auc_ath": None, "auc_s1": None, "basis_names": [], "grid": {"E": [], "A": []}, "addons": [], "d1_polish": []})
    payload = {"slot2": slot2, "screen": screen, "NO_SUBMIT_step2": True}
    (OUT / "v16_3.json").write_text(json.dumps(payload, indent=2, default=float))
    (REPO / "logs/overnight/v16_3.json").write_text(json.dumps(payload, indent=2, default=float))
    log("DONE")


if __name__ == "__main__":
    main()
