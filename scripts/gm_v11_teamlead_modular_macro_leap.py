#!/usr/bin/env python3
"""GM v11 Team-Lead Modular Macro-Leap — Modules A/B/C/D, strict 5/5 gates.

STRICT NO_SUBMIT=True. Immutable cand2 (0.95355) anchor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gm_v11_shared import (  # noqa: E402
    ART,
    BLACKLIST,
    LOG,
    NO_SUBMIT,
    PRED,
    RECON,
    SUB,
    VOL,
    force_law,
    g26_status,
    gate_pass,
    L,
    L_fold,
    load_anchor_and_bank,
    log,
    lofo_eval,
    pchip_lock,
    project_out,
    setup_folds,
    sha16,
    spearman_dist,
    unit,
    unitize,
    weight_force,
    write_nosubmit,
)

# Reuse pillar implementations
from gm_v11_1_nonlinear_stageB_wilcoxon_and_g26_denoiser import (  # noqa: E402
    pillar1_g26_denoise_and_lmtrim,
    pillar2_wilcoxon,
    pillar3_stageB,
)


def module_A(pack):
    log("======== MODULE A: zero-sum bank + v17→s352 ========")
    y, folds = pack["y"], pack["folds"]
    bases_o, bases_t = pack["bases_o"], pack["bases_t"]
    zS, zP = pack["eval_S7matched_355"], pack["eval_simple_355"]

    swaps = [
        ("LM_trim", -(pack["LMo"] - pack["Fo"]), -(pack["LM"] - pack["F"])),
        ("GPD-MLP", pack["GPDo"] - pack["MLPo"], pack["GPD"] - pack["MLP"]),
        ("GPD-GB", pack["GPDo"] - pack["GBo"], pack["GPD"] - pack["GB"]),
        ("L7F-LM", pack["L7Fo"] - pack["LMo"], pack["L7F"] - pack["LM"]),
        ("GPD-MS", pack["GPDo"] - pack["MSo"], pack["GPD"] - pack["MS"]),
        ("L7F-MS", pack["L7Fo"] - pack["MSo"], pack["L7F"] - pack["MS"]),
        ("d1", pack["d1o"], pack["d1t"]),
    ]
    if pack["Gmedo"] is not None:
        swaps.append(("Gmed-MS", pack["Gmedo"] - pack["MSo"], pack["Gmedt"] - pack["MS"]))
    # bank vs MS/MLP only (cap compute)
    for i, (o, t) in enumerate(zip(pack["bank_o"][:8], pack["bank_t"][:8])):
        for cname, co, ct in [("MS", pack["MSo"], pack["MS"]), ("MLP", pack["MLPo"], pack["MLP"])]:
            swaps.append((f"bank{i}-{cname}", o - co, t - ct))

    winners = []
    # prefilter with fixed-w fold scan
    from _gm_v11_shared import fold_aucs

    base_fa = fold_aucs(y, zS, folds)
    m0 = folds[0][1]
    pre = []
    for name, o, t in swaps:
        do = project_out(o, bases_o)
        if np.linalg.norm(do) < 1e-8:
            continue
        u = unit(do)
        if roc_auc_score(y[m0], zS[m0] + 0.04 * u[m0]) < roc_auc_score(y[m0], zS[m0] - 0.04 * u[m0]):
            u = -u
        d = [(roc_auc_score(y[te], (zS + 0.04 * u)[te]) - b) * 1e5 for b, (_, te) in zip(base_fa, folds)]
        if sum(x > 0 for x in d) >= 4 and d[0] > 0:
            pre.append((name, do, t, d))
    pre.sort(key=lambda r: (sum(x > 0 for x in r[3]), float(np.mean(r[3]))), reverse=True)
    log(f"  A prefilter survivors={len(pre)} / tested={len(swaps)}")
    for name, do, t, _ in pre[:12]:
        rS = lofo_eval(y, zS, do, folds)
        if not (rS["pos5"] == 5 and rS["d0_e5"] > 0):
            continue
        rP = lofo_eval(y, zP, do, folds)
        if rP["pos5"] < 4:
            continue
        dt = project_out(t, bases_t)
        dt = pchip_lock(dt, do)
        dt = project_out(dt, bases_t)
        winners.append(
            {
                "name": name,
                "S7m": {k: v for k, v in rS.items() if k != "u"},
                "simple": {k: v for k, v in rP.items() if k != "u"},
                "u_o": rS["u"],
                "u_t": unit(dt),
                "pass": True,
            }
        )
        log(f"  A PASS {name}: mean={rS['mean_d_e5']} folds={rS['fold_d_e5']}")

    if not winners:
        # include LM_trim even if not full gate for reporting
        do = project_out(-(pack["LMo"] - pack["Fo"]), bases_o)
        rS = lofo_eval(y, zS, do, folds)
        rP = lofo_eval(y, zP, do, folds)
        log(f"  A near LM_trim: {rS['pos5']}/5 mean={rS['mean_d_e5']}")
        return {
            "pass": gate_pass(rS, rP),
            "S7m": {k: v for k, v in rS.items() if k != "u"},
            "simple": {k: v for k, v in rP.items() if k != "u"},
            "u_o": rS["u"],
            "u_t": unit(project_out(-(pack["LM"] - pack["F"]), bases_t)),
            "winners": [],
            "n_tested": len(swaps),
        }

    # combine winners equal weight
    Uo = unit(np.mean([unit(w["u_o"]) for w in winners], 0))
    Ut = unit(np.mean([unit(w["u_t"]) for w in winners], 0))
    rS = lofo_eval(y, zS, Uo, folds)
    rP = lofo_eval(y, zP, Uo, folds)
    return {
        "pass": gate_pass(rS, rP),
        "S7m": {k: v for k, v in rS.items() if k != "u"},
        "simple": {k: v for k, v in rP.items() if k != "u"},
        "u_o": rS["u"],
        "u_t": Ut,
        "winners": [w["name"] for w in winners],
        "n_tested": len(swaps),
    }


def rankdata_safe(z):
    from scipy.stats import rankdata

    z = np.asarray(z, dtype=np.float64).ravel()
    return (rankdata(z, method="average") - 0.5) / len(z)


def module_C(pack, basis_us):
    log("======== MODULE C: MoE twilight/disagree ========")
    y, folds = pack["y"], pack["folds"]
    bases_o, bases_t = pack["bases_o"], pack["bases_t"]
    z0 = pack["eval_S7matched_355"]
    # disagreement
    d_fam_o = unitize(np.abs(pack["LMo"] - pack["GBo"]))
    d_fam_t = unitize(np.abs(pack["LM"] - pack["GB"]))
    q = rankdata_safe(z0)
    # smooth twilight |g'| bounded via scaled logistic of distance to band center
    center, width = 0.90, 0.12
    g_tw_o = np.tanh((1.0 - np.abs(q - center) / width).clip(0, 1) * 0.15)
    # test: use cand2 ranks
    qt = rankdata_safe(L(pack["cand2"]))
    g_tw_t = np.tanh((1.0 - np.abs(qt - center) / width).clip(0, 1) * 0.15)
    g_dis_o = np.tanh(d_fam_o / 2.0) * 0.15
    g_dis_t = np.tanh(d_fam_t / 2.0) * 0.15

    # top 4 u signals
    us = []
    for name, uo, ut in basis_us[:4]:
        us.append((name, unit(uo), unit(ut)))
    if not us:
        return {"pass": False, "S7m": {"pos5": 0, "mean_d_e5": 0, "d0_e5": 0, "fold_d_e5": []}, "simple": {"pos5": 0}}

    # interaction features
    feats_o, feats_t = [], []
    for name, uo, ut in us:
        feats_o += [uo * g_tw_o, uo * g_dis_o]
        feats_t += [ut * g_tw_t, ut * g_dis_t]
    # LOFO ridge on interactions; residualize main effects
    from sklearn.linear_model import Ridge

    delta_o = np.zeros(len(y))
    delta_t_acc = []
    Xo = np.column_stack(feats_o)
    Xt = np.column_stack(feats_t)
    for f, (tr, te) in enumerate(folds):
        # residual y-proxy: leave as predicting residual of labels via logistic linear — use AUC direction via z0 lift
        # Fit w to maximize soft AUC via ridge on (label - mean) with features
        yy = y[tr].astype(np.float64) - y[tr].mean()
        ridge = Ridge(alpha=50.0)
        ridge.fit(Xo[tr], yy)
        delta_o[te] = ridge.predict(Xo[te])
        delta_t_acc.append(ridge.predict(Xt))
    delta_o = project_out(delta_o, bases_o + [u for _, u, _ in us])
    delta_t = project_out(np.mean(delta_t_acc, 0), bases_t + [u for _, _, u in us])
    delta_t = pchip_lock(delta_t, delta_o)
    delta_t = project_out(delta_t, bases_t)

    rS = lofo_eval(y, z0, delta_o, folds)
    rP = lofo_eval(y, pack["eval_simple_355"], delta_o, folds)
    log(f"  MoE: S7m {rS['pos5']}/5 mean={rS['mean_d_e5']} folds={rS['fold_d_e5']}")
    return {
        "pass": gate_pass(rS, rP),
        "S7m": {k: v for k, v in rS.items() if k != "u"},
        "simple": {k: v for k, v in rP.items() if k != "u"},
        "u_o": rS["u"],
        "u_t": unit(delta_t),
    }


def stage_teamlead(pack, mods):
    log("======== STEP5: promotion + stage on cand2 ========")
    cand2, ids = pack["cand2"], pack["ids"]
    passed = []
    rows = []
    for name, m in mods.items():
        ok = bool(m.get("pass"))
        rows.append((name, m.get("S7m", {}), m.get("simple", {}), ok))
        log(
            f"  {name}: GATE={'PASS' if ok else 'FAIL'} "
            f"S7m={m.get('S7m',{}).get('pos5')}/5 mean={m.get('S7m',{}).get('mean_d_e5')} "
            f"simple={m.get('simple',{}).get('pos5')}/5"
        )
        if ok and "u_t" in m:
            passed.append((name, m["u_t"], m["u_o"], m["S7m"]["mean_d_e5"]))

    cands = {}
    if not passed:
        # fallback: strongest mean module even if fail — marked best-effort
        ranked = sorted(mods.items(), key=lambda kv: kv[1].get("S7m", {}).get("mean_d_e5", -1e9), reverse=True)
        name, m = ranked[0]
        u = m.get("u_t", unit(pack["u_g26ms_t"]))
        note = "best_effort_no_5of5"
        for slot, tgt, fname in [
            ("slot1", 0.112e-4, "sub_v11_slot1_macro_pass_0p11e4_NOSUBMIT.csv"),
            ("slot2", 0.220e-4, "sub_v11_slot2_macro_pass_0p22e4_NOSUBMIT.csv"),
            ("slot3", 0.120e-4, "sub_v11_slot3_top_single_module_0p12e4_NOSUBMIT.csv"),
        ]:
            p, w, got = weight_force(cand2, u, tgt)
            path = SUB / fname
            write_nosubmit(path, ids, p)
            cands[slot] = {"path": str(path), "1rho": got, "w": w, "sha16": sha16(path), "note": note, "module": name}
            log(f"  {slot}: 1ρ={got:.6e} ({note} from {name})")
        return {"cands": cands, "passed": [], "rows": rows}

    U = unit(np.mean([unit(u) for _, u, _, _ in passed], 0))
    best = max(passed, key=lambda t: t[3])
    # neg probe cos
    neg = {}
    for label, path in [
        ("v92_rung1", SUB / "sub_v9p2_rung1_probe_0p20e4.csv"),
        ("v10_slot1", SUB / "sub_v10_slot1_s354pub_plus_clean5f_stageB_0p12e4_LIVE.csv"),
    ]:
        if path.exists():
            from _gm_v11_shared import load_prob

            neg[label] = float(
                np.dot(U, unit(unitize(L(load_prob(path))) - unitize(L(cand2))))
            )

    specs = [
        ("slot1", 0.112e-4, U, "sub_v11_slot1_macro_pass_0p11e4_NOSUBMIT.csv"),
        ("slot2", 0.220e-4, U, "sub_v11_slot2_macro_pass_0p22e4_NOSUBMIT.csv"),
        ("slot3", 0.120e-4, unit(best[1]), "sub_v11_slot3_top_single_module_0p12e4_NOSUBMIT.csv"),
    ]
    for slot, tgt, u, fname in specs:
        p, w, got = weight_force(cand2, u, tgt)
        path = SUB / fname
        write_nosubmit(path, ids, p)
        cands[slot] = {
            "path": str(path),
            "1rho": got,
            "w": w,
            "sha16": sha16(path),
            "modules": [x[0] for x in passed],
            "cos_negs": neg,
            "mean": float(p.mean()),
        }
        log(f"  {slot}: 1ρ={got:.6e} → {fname}")
    # verify immutable
    assert spearman_dist(force_law(cand2, unitize(L(cand2))), cand2) < 1e-12
    # combined OOF check
    if len(passed) >= 1:
        rS = lofo_eval(pack["y"], pack["eval_S7matched_355"], unit(np.mean([unit(x[2]) for x in passed], 0)), pack["folds"])
        rP = lofo_eval(pack["y"], pack["eval_simple_355"], unit(np.mean([unit(x[2]) for x in passed], 0)), pack["folds"])
        log(f"  Δ_macro_pass OOF S7m {rS['pos5']}/5 mean={rS['mean_d_e5']} simple {rP['pos5']}/5")
    return {"cands": cands, "passed": [x[0] for x in passed], "rows": rows, "cos_negs": neg}


def write_report(g26, mods, staged, recipe):
    md = []
    md.append("# GM v11 Team-Lead Modular Macro-Leap Report\n\n")
    md.append("`NO_SUBMIT=True` · immutable anchor **cand2 = 0.95355** · $0.00/hr GPU\n\n")
    md.append("## Engineering laws\n\n")
    md.append("1. Immutable cand2 anchor · 2. Double orth vs S7 & (G26−MS) · 3. Contrast PCHIP · 4. Transductive priors only (no zero-grad pseudo) · 5. Independent 5/5 LOFO gate\n\n")
    md.append("## G26 recipe / status\n\n")
    md.append(f"- has_test_5fold: **{g26.get('has_test_5fold')}** (remote `{g26.get('has_test_5fold_remote')}`)\n")
    md.append(f"- files: `{g26.get('files')}`\n")
    for k, v in recipe.items():
        md.append(f"- **{k}:** {v}\n")
    md.append("\n## 4-Module Promotion Scorecard\n\n")
    md.append("| Module | S7m pos | meanΔe-5 | d0 | folds | simple pos | GATE |\n|---|---:|---:|---:|---|---:|---|\n")
    for name, m in mods.items():
        s, p = m.get("S7m", {}), m.get("simple", {})
        md.append(
            f"| {name} | {s.get('pos5')}/5 | {s.get('mean_d_e5')} | {s.get('d0_e5')} | "
            f"{s.get('fold_d_e5')} | {p.get('pos5')}/5 | {'PASS' if m.get('pass') else 'FAIL'} |\n"
        )
    md.append(f"\nPassed: `{staged.get('passed')}`\n")
    if staged.get("cos_negs"):
        md.append(f"cos vs neg probes: `{staged['cos_negs']}`\n")
    md.append("\n## Staged NOSUBMIT (cand2 + w·Δ)\n\n")
    for k, v in staged.get("cands", {}).items():
        md.append(f"### {k}\n\n")
        for kk, vv in v.items():
            md.append(f"- **{kk}:** `{vv}`\n")
        md.append("\n")
    md.append("---\n`NO_SUBMIT=True` — prior v11 s354-climb cands remain unsubmitted.\n")
    path = LOG / "GM_V11_TEAMLEAD_MACRO_REPORT.md"
    path.write_text("".join(md))
    log(f"wrote {path}")
    return path


def main():
    assert NO_SUBMIT
    log("======== GM v11 TEAMLEAD MODULAR MACRO-LEAP (NO_SUBMIT) ========")
    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds, fold_id = setup_folds(y)
    g26 = g26_status()
    recipe = {
        "arch": "GPT-2 small target-first (Deotte/masaya)",
        "classifier": "8 ep AdamW lr 5e-5→0 batch32 bf16",
        "orders": 10,
        "train_gpu_h": "~60",
        "score_gpu_h_per_fold_test": "~12.6",
        "folds_1_4_est_h": "~50.4",
        "oof_auc_G26": 0.950140,
    }
    log(f"  G26 5fold test uploaded? {g26.get('has_test_5fold_remote')}")
    pack = load_anchor_and_bank(y, fold_id, folds)

    # Module A
    modA = module_A(pack)

    # Pillar1 for denoise features + Module B/D via v11.1 pillars
    p1 = pillar1_g26_denoise_and_lmtrim(pack)
    modB = pillar2_wilcoxon(pack, p1)
    modB = {**modB, "name": "B_wilcoxon"}

    basis_us = [
        ("LM_trim", p1["lm_trim"]["u_o"], p1["lm_trim_t"]),
        ("GPD-MLP", project_out(pack["GPDo"] - pack["MLPo"], pack["bases_o"]), project_out(pack["GPD"] - pack["MLP"], pack["bases_t"])),
        ("L7F-LM", project_out(pack["L7Fo"] - pack["LMo"], pack["bases_o"]), project_out(pack["L7F"] - pack["LM"], pack["bases_t"])),
        ("G26MS", project_out(pack["u_g26ms_o"], pack["bases_o"]), project_out(pack["u_g26ms_t"], pack["bases_t"])),
    ]
    modC = module_C(pack, basis_us)
    modD = pillar3_stageB(pack, p1)

    # Also expose P1 combo as optional
    a2 = dict(p1["combo"])
    a2["u_t"] = unit(p1["delta_t"])
    a2["pass"] = bool(p1["combo"]["pass"])
    mods = {
        "A_zerosum": modA,
        "B_wilcoxon": modB,
        "C_moe": modC,
        "D_stageB": modD,
        "A2_LMtrim_G26denoise": a2,
    }

    staged = stage_teamlead(pack, mods)
    write_report(g26, mods, staged, recipe)
    log("DONE teamlead modular")


if __name__ == "__main__":
    main()
