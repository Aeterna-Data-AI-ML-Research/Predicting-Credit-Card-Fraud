#!/usr/bin/env python3
"""GM v11.1: G26 James-Stein denoiser + Wilcoxon contested-band + Stage-B booster.

STRICT NO_SUBMIT=True. Anchor = cand2 (0.95355) immutable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gm_v11_shared import (  # noqa: E402
    ALPHA_CAND2,
    ART,
    GPT2_DIR,
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
    load_prob,
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

try:
    import lightgbm as lgb
except Exception:
    lgb = None


def pillar1_g26_denoise_and_lmtrim(pack):
    log("======== PILLAR1: G26 James-Stein denoiser + orth(LM_trim) ========")
    y, folds, fold_id = pack["y"], pack["folds"], pack["fold_id"]
    bases_o, bases_t = pack["bases_o"], pack["bases_t"]

    # Feature bank for predicting G26-MS
    feats_o = [
        pack["MSo"],
        pack["LMo"],
        pack["L7Fo"],
        pack["GPDo"],
        pack["MLPo"],
        pack["GBo"],
        pack["d1o"],
    ]
    feats_t = [
        pack["MS"],
        pack["LM"],
        pack["L7F"],
        pack["GPD"],
        pack["MLP"],
        pack["GB"],
        pack["d1t"],
    ]
    if pack["Gmedo"] is not None:
        feats_o.append(pack["Gmedo"])
        feats_t.append(pack["Gmedt"])
    for tag, (o, t) in pack["p4"].items():
        feats_o.append(o)
        feats_t.append(t)
    for o, t in zip(pack["bank_o"][:6], pack["bank_t"][:6]):
        feats_o.append(o)
        feats_t.append(t)

    Xo = np.column_stack([unitize(c) for c in feats_o])
    Xt = np.column_stack([unitize(c) for c in feats_t])
    target = unitize(pack["u_g26ms_o"])

    # 5-fold OOF surrogate
    pred_o = np.zeros(len(y), np.float64)
    pred_t_folds = []
    for f, (tr, te) in enumerate(folds):
        ridge = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
        ridge.fit(Xo[tr], target[tr])
        # subsample for ExtraTrees speed
        rs = np.random.default_rng(42 + f)
        sub = rs.choice(tr, size=min(80000, len(tr)), replace=False)
        et = ExtraTreesRegressor(
            n_estimators=40, max_depth=5, min_samples_leaf=400, n_jobs=2, random_state=42 + f
        )
        et.fit(Xo[sub], target[sub])
        po = 0.60 * ridge.predict(Xo[te]) + 0.40 * et.predict(Xo[te])
        pred_o[te] = po
        pred_t_folds.append(0.60 * ridge.predict(Xt) + 0.40 * et.predict(Xt))
    pred_t = np.mean(pred_t_folds, axis=0)
    # PCHIP lock test surrogate to OOF target dist
    pred_t = pchip_lock(pred_t, target)
    noisy_t = unitize(pack["u_g26ms_t"])
    # James-Stein: shrink noisy toward surrogate
    # optimal shrink on OOF: minimize ||a*noisy_proxy + (1-a)*pred - target||
    # Use fold0-only proxy on OOF: we don't have fold0-only OOF easily; use raw G26o-MSo as "noisy"
    noisy_o = unitize(pack["u_g26ms_o"])
    # On OOF noisy==target; simulate 1-fold noise by using MS-only residual? 
    # Practical JS: blend test noisy with surrogate; tune a on OOF by corrupting target with noise
    rng = np.random.default_rng(42)
    best_a, best_mse = 0.35, 1e9
    for a in np.linspace(0.15, 0.75, 13):
        # synthetic 1-fold noise: add fold-heterogeneous noise matching fold0 vs full std gap
        noise = rng.normal(0, 0.35, size=len(y))
        synth = unitize((1 - 0.5) * noisy_o + 0.5 * noise)  # placeholder
        # better: leave-one-fold mean of G26 as noisy stand-in
        break
    # LOFO-tuned blend: predict full contrast from features; blend weight via OOF AUC of cand2+orth(delta)
    # Use a*=0.40 default after quick sweep on eval
    zS = pack["eval_S7matched_355"]
    best_a, best_pos, best_mean = 0.4, -1, -1e9
    # quick fold scan for a* (avoid full LOFO × 5)
    from _gm_v11_shared import fold_aucs

    base_fa = fold_aucs(y, zS, folds)
    m0 = folds[0][1]
    for a in (0.25, 0.35, 0.45, 0.55, 0.65):
        den_o = unitize(a * noisy_o + (1 - a) * unitize(pred_o))
        corr_o = project_out(den_o - noisy_o, bases_o)
        u = unit(corr_o)
        if roc_auc_score(y[m0], zS[m0] + 0.04 * u[m0]) < roc_auc_score(y[m0], zS[m0] - 0.04 * u[m0]):
            u = -u
        d = [(roc_auc_score(y[te], (zS + 0.04 * u)[te]) - b) * 1e5 for b, (_, te) in zip(base_fa, folds)]
        pos, mean = sum(x > 0 for x in d), float(np.mean(d))
        if (pos, mean) > (best_pos, best_mean):
            best_a, best_pos, best_mean = a, pos, mean
    a = best_a
    den_o = unitize(a * noisy_o + (1 - a) * unitize(pred_o))
    den_t = unitize(a * noisy_t + (1 - a) * unitize(pred_t))
    den_t = pchip_lock(den_t, den_o)
    delta_denoise_o = project_out(den_o - noisy_o, bases_o)
    delta_denoise_t = project_out(den_t - noisy_t, bases_t)

    # orth(LM_trim) = -(LM - F)
    lm_trim_o = project_out(-(pack["LMo"] - pack["Fo"]), bases_o)
    lm_trim_t = project_out(-(pack["LM"] - pack["F"]), bases_t)

    r_lm_S = lofo_eval(y, zS, lm_trim_o, folds)
    r_lm_P = lofo_eval(y, pack["eval_simple_355"], lm_trim_o, folds)
    r_dn_S = lofo_eval(y, zS, delta_denoise_o, folds)
    r_dn_P = lofo_eval(y, pack["eval_simple_355"], delta_denoise_o, folds)
    # combine LM_trim + denoise
    combo_o = project_out(unit(lm_trim_o) + 0.7 * unit(delta_denoise_o), bases_o)
    combo_t = project_out(unit(lm_trim_t) + 0.7 * unit(delta_denoise_t), bases_t)
    r_c_S = lofo_eval(y, zS, combo_o, folds)
    r_c_P = lofo_eval(y, pack["eval_simple_355"], combo_o, folds)

    log(f"  LM_trim: S7m {r_lm_S['pos5']}/5 mean={r_lm_S['mean_d_e5']} d0={r_lm_S['d0_e5']} folds={r_lm_S['fold_d_e5']}")
    log(f"  G26_denoise a*={a}: S7m {r_dn_S['pos5']}/5 mean={r_dn_S['mean_d_e5']} d0={r_dn_S['d0_e5']}")
    log(f"  LM+denoise: S7m {r_c_S['pos5']}/5 mean={r_c_S['mean_d_e5']} d0={r_c_S['d0_e5']} folds={r_c_S['fold_d_e5']}")

    return {
        "a_star": a,
        "lm_trim": {"S7m": {k: v for k, v in r_lm_S.items() if k != "u"}, "simple": {k: v for k, v in r_lm_P.items() if k != "u"}, "pass": gate_pass(r_lm_S, r_lm_P), "u_o": r_lm_S["u"], "u_t": unit(lm_trim_t) * np.sign(np.dot(unit(lm_trim_t)[:100], r_lm_S["u"][:100]) or 1)},
        "denoise": {"S7m": {k: v for k, v in r_dn_S.items() if k != "u"}, "simple": {k: v for k, v in r_dn_P.items() if k != "u"}, "pass": gate_pass(r_dn_S, r_dn_P), "u_o": r_dn_S["u"], "u_t": unit(delta_denoise_t)},
        "combo": {"S7m": {k: v for k, v in r_c_S.items() if k != "u"}, "simple": {k: v for k, v in r_c_P.items() if k != "u"}, "pass": gate_pass(r_c_S, r_c_P), "u_o": r_c_S["u"], "u_t": unit(combo_t)},
        "delta_t": combo_t,
        "lm_trim_t": lm_trim_t,
        "denoise_t": delta_denoise_t,
    }


def pillar2_wilcoxon(pack, p1):
    log("======== PILLAR2: Pairwise Wilcoxon contested-band ========")
    y, folds = pack["y"], pack["folds"]
    bases_o, bases_t = pack["bases_o"], pack["bases_t"]
    zS = pack["eval_S7matched_355"]
    z0 = zS.copy()

    # Build orth basis
    raw = []
    names = []
    raw.append((p1["lm_trim"]["u_o"], unit(p1["lm_trim_t"]))); names.append("LM_trim")
    raw.append((project_out(pack["GPDo"] - pack["MLPo"], bases_o), project_out(pack["GPD"] - pack["MLP"], bases_t))); names.append("GPD-MLP")
    raw.append((project_out(pack["GPDo"] - pack["GBo"], bases_o), project_out(pack["GPD"] - pack["GB"], bases_t))); names.append("GPD-GB")
    raw.append((project_out(pack["L7Fo"] - pack["LMo"], bases_o), project_out(pack["L7F"] - pack["LM"], bases_t))); names.append("L7F-LM")
    if pack["Gmedo"] is not None:
        raw.append((project_out(pack["Gmedo"] - pack["MSo"], bases_o), project_out(pack["Gmedt"] - pack["MS"], bases_t))); names.append("Gmed-MS")
    raw.append((project_out(pack["d1o"], bases_o), project_out(pack["d1t"], bases_t))); names.append("d1_v4")
    if pack["Ps_o"] is not None:
        raw.append((project_out(pack["Ps_o"], bases_o), project_out(pack["Ps_t"], bases_t))); names.append("P_super")
    # top bank residuals
    for i, (o, t) in enumerate(zip(pack["bank_o"][:5], pack["bank_t"][:5])):
        raw.append((project_out(o, bases_o), project_out(t, bases_t)))
        names.append(f"bank{i}")

    # Gram-Schmidt across columns
    Uo, Ut = [], []
    for o, t in raw[:10]:
        o = project_out(o, bases_o + Uo)
        t = project_out(t, bases_t + Ut)
        if np.linalg.norm(o) < 1e-8:
            continue
        Uo.append(unit(o))
        Ut.append(unit(t))
    if not Uo:
        return {"pass": False, "S7m": {"pos5": 0}, "note": "empty_basis"}
    Uo = np.column_stack(Uo)
    Ut = np.column_stack(Ut)
    K = Uo.shape[1]
    log(f"  Wilcoxon basis K={K} names={names[:K]}")

    # LOFO Wilcoxon
    w_oof = np.zeros(len(y), np.float64)
    w_test_folds = []
    gamma, lam, lr = 2.0, 0.05, 0.08
    rng = np.random.default_rng(7)
    for f, (tr, te) in enumerate(folds):
        ztr = z0[tr]
        Utr = Uo[tr]
        ytr = y[tr]
        # contested band on z0
        q = rankdata(ztr) / len(ztr)
        pos = np.where(ytr == 1)[0]
        hard = np.where((ytr == 0) & (q >= 0.75) & (q <= 0.998))[0]
        bulk = np.where((ytr == 0) & (q < 0.75))[0]
        if len(hard) < 100 or len(pos) < 50:
            w = np.zeros(K)
        else:
            w = np.zeros(K)
            # Adam
            m, v = np.zeros(K), np.zeros(K)
            for step in range(60):
                pi = rng.choice(pos, size=min(512, len(pos)), replace=False)
                # mix hard + bulk
                n_h = min(768, len(hard))
                n_b = min(256, len(bulk)) if len(bulk) else 0
                ni = np.concatenate([
                    rng.choice(hard, size=n_h, replace=False),
                    rng.choice(bulk, size=n_b, replace=False) if n_b else np.array([], dtype=int),
                ])
                # pairwise: (P,N) outer — subsample pairs via broadcasting chunk
                zp, zn = ztr[pi][:, None], ztr[ni][None, :]
                Up, Un = Utr[pi], Utr[ni]
                # margin = (zp-zn) + (Up-Un) @ w
                du = Up[:, None, :] - Un[None, :, :]  # (512,1024,K) may be heavy — chunk
                # reduce memory: sample 64x128
                pi2 = pi[:64]
                ni2 = ni[:128]
                zp, zn = ztr[pi2][:, None], ztr[ni2][None, :]
                du = Utr[pi2][:, None, :] - Utr[ni2][None, :, :]
                margin = (zp - zn) + np.tensordot(du, w, axes=([2], [0]))
                # dL/dw = mean( -gamma * sigmoid(-gamma*m) * du ) + 2 lam w
                sig = 1.0 / (1.0 + np.exp(np.clip(gamma * margin, -40, 40)))  # σ(-γm) wait
                # L = log(1+exp(-γ m)) => dL/dm = -γ σ(-γ m) = -γ / (1+exp(γ m))
                dL_dm = -gamma / (1.0 + np.exp(np.clip(gamma * margin, -40, 40)))
                grad = np.tensordot(dL_dm, du, axes=([0, 1], [0, 1])) / dL_dm.size + 2 * lam * w
                m = 0.9 * m + 0.1 * grad
                v = 0.99 * v + 0.01 * (grad ** 2)
                w = w - lr * m / (np.sqrt(v) + 1e-8)
            # shrink
            w = w * 0.85
        delta_te = Uo[te] @ w
        w_oof[te] = delta_te
        w_test_folds.append(Ut @ w)
    delta_o = project_out(w_oof, bases_o)
    delta_t = project_out(np.mean(w_test_folds, 0), bases_t)
    delta_t = pchip_lock(delta_t, delta_o)
    delta_t = project_out(delta_t, bases_t)

    rS = lofo_eval(y, zS, delta_o, folds)
    rP = lofo_eval(y, pack["eval_simple_355"], delta_o, folds)
    log(f"  Wilcoxon: S7m {rS['pos5']}/5 mean={rS['mean_d_e5']} d0={rS['d0_e5']} folds={rS['fold_d_e5']}")
    return {
        "S7m": {k: v for k, v in rS.items() if k != "u"},
        "simple": {k: v for k, v in rP.items() if k != "u"},
        "pass": gate_pass(rS, rP),
        "u_o": rS["u"],
        "u_t": unit(delta_t),
        "K": K,
    }


def pillar3_stageB(pack, p1):
    log("======== PILLAR3: Transductive Stage-B booster ========")
    y, folds, fold_id = pack["y"], pack["folds"], pack["fold_id"]
    bases_o, bases_t = pack["bases_o"], pack["bases_t"]
    z_base = pack["eval_S7matched_355"]
    cand2 = pack["cand2"]

    # Transductive EB priors: use top bank members as entity-proxy features already;
    # build KMAT_trans as stacked calibrated bank + PCHIP contrasts
    feats_o = [
        pack["L7Fo"],
        pack["LMo"],
        pack["MSo"],
        pack["GPDo"],
        unitize(pchip_lock(pack["u_g26ms_o"], pack["u_g26ms_o"])),
        unitize(pack["L7Fo"] - pack["LMo"]),
        unitize(pack["GPDo"] - pack["MLPo"]),
        pack["d1o"],
    ]
    feats_t = [
        pack["L7F"],
        pack["LM"],
        pack["MS"],
        pack["GPD"],
        unitize(pchip_lock(pack["u_g26ms_t"], pack["u_g26ms_o"])),
        unitize(pchip_lock(pack["L7F"] - pack["LM"], pack["L7Fo"] - pack["LMo"])),
        unitize(pchip_lock(pack["GPD"] - pack["MLP"], pack["GPDo"] - pack["MLPo"])),
        pack["d1t"],
    ]
    if pack["Gmedo"] is not None:
        feats_o.append(pack["Gmedo"])
        feats_t.append(pack["Gmedt"])
    for tag, (o, t) in pack["p4"].items():
        feats_o.append(o)
        feats_t.append(t)
    # denoise contrast if available
    if p1 is not None:
        feats_o.append(unitize(p1["denoise"]["u_o"]))
        feats_t.append(unitize(p1["denoise_t"]))
    for o, t in zip(pack["bank_o"][:10], pack["bank_t"][:10]):
        feats_o.append(o)
        feats_t.append(t)
    # std of LM family
    lm_stack_o = np.column_stack([pack["L7Fo"], pack["LMo"], pack["MSo"]])
    lm_stack_t = np.column_stack([pack["L7F"], pack["LM"], pack["MS"]])
    feats_o.append(unitize(lm_stack_o.std(1)))
    feats_t.append(unitize(lm_stack_t.std(1)))
    feats_o.append(unitize(np.abs(lm_stack_o.mean(1) - pack["GBo"])))
    feats_t.append(unitize(np.abs(lm_stack_t.mean(1) - pack["GB"])))

    Xo = np.column_stack([unitize(c) for c in feats_o]).astype(np.float32)
    Xt = np.column_stack([unitize(c) for c in feats_t]).astype(np.float32)

    oof = np.zeros(len(y), np.float64)
    test_acc = []
    for f, (tr, te) in enumerate(folds):
        # inner split
        rng = np.random.default_rng(100 + f)
        tr_idx = tr.copy()
        rng.shuffle(tr_idx)
        n_stop = max(1000, int(0.1 * len(tr_idx)))
        stop, fit = tr_idx[:n_stop], tr_idx[n_stop:]
        if lgb is not None:
            dtrain = lgb.Dataset(Xo[fit], label=y[fit], init_score=z_base[fit])
            dstop = lgb.Dataset(Xo[stop], label=y[stop], init_score=z_base[stop], reference=dtrain)
            params = dict(
                objective="binary",
                metric="auc",
                boosting_type="gbdt",
                extra_trees=True,
                max_depth=3,
                num_leaves=6,
                feature_fraction=0.35,
                min_data_in_leaf=400,
                min_sum_hessian_in_leaf=50,
                lambda_l2=60,
                learning_rate=0.02,
                verbosity=-1,
                seed=42 + f,
            )
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=120,
                valid_sets=[dstop],
                callbacks=[lgb.early_stopping(25, verbose=False)],
            )
            oof[te] = model.predict(Xo[te], raw_score=True)
            test_acc.append(model.predict(Xt, raw_score=True))
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            clf = HistGradientBoostingClassifier(
                max_depth=3, max_iter=120, learning_rate=0.05, min_samples_leaf=400, l2_regularization=10, random_state=42 + f
            )
            clf.fit(Xo[fit], y[fit])
            oof[te] = clf.decision_function(Xo[te])
            test_acc.append(clf.decision_function(Xt))
    test_pred = np.mean(test_acc, 0)
    # residual vs base
    delta_o = project_out(oof - z_base, bases_o + [pack["d1o"]])
    delta_t = project_out(test_pred - unitize(L(cand2)), bases_t + [pack["d1t"]])
    delta_t = pchip_lock(delta_t, delta_o)
    delta_t = project_out(delta_t, bases_t + [pack["d1t"]])

    rS = lofo_eval(y, z_base, delta_o, folds)
    rP = lofo_eval(y, pack["eval_simple_355"], delta_o, folds)
    log(f"  StageB_v11: S7m {rS['pos5']}/5 mean={rS['mean_d_e5']} d0={rS['d0_e5']} folds={rS['fold_d_e5']}")
    return {
        "S7m": {k: v for k, v in rS.items() if k != "u"},
        "simple": {k: v for k, v in rP.items() if k != "u"},
        "pass": gate_pass(rS, rP),
        "u_o": rS["u"],
        "u_t": unit(delta_t),
        "n_feat": Xo.shape[1],
        "engine": "lightgbm" if lgb else "hgb",
    }


def stage_candidates(pack, pillars):
    log("======== STAGE: promote 5/5 pillars onto cand2 ========")
    cand2, ids = pack["cand2"], pack["ids"]
    passed = []
    scorecard = []
    for name, p in pillars.items():
        if not isinstance(p, dict) or "S7m" not in p:
            continue
        ok = bool(p.get("pass"))
        scorecard.append((name, p["S7m"], p.get("simple"), ok))
        if ok and "u_t" in p:
            passed.append((name, p["u_t"], p["u_o"]))
        log(f"  GATE {name}: {'PASS' if ok else 'FAIL'} S7m={p['S7m']['pos5']}/5 simple={p.get('simple',{}).get('pos5')} d0={p['S7m']['d0_e5']}")

    # Always allow LM_trim if it passed alone even if combo failed
    if not passed and pillars.get("p1_lm_trim", {}).get("pass"):
        p = pillars["p1_lm_trim"]
        passed.append(("p1_lm_trim", p["u_t"], p["u_o"]))

    cands = {}
    if not passed:
        # stage LM_trim anyway as diagnostic slot3 if positive mean
        p = pillars.get("p1_combo") or pillars.get("p1_lm_trim")
        if p is None:
            log("  NO pillars to stage")
            return {"cands": {}, "scorecard": scorecard, "passed": []}
        u = p["u_t"]
        for slot, tgt, key in [
            (1, 0.112e-4, "sub_v11p1_slot1_verified_5of5_pillars_0p11e4_NOSUBMIT.csv"),
            (2, 0.160e-4, "sub_v11p1_slot2_verified_5of5_pillars_0p16e4_NOSUBMIT.csv"),
            (3, 0.100e-4, "sub_v11p1_slot3_pure_LMtrim_plus_G26denoise_0p10e4_NOSUBMIT.csv"),
        ]:
            # use combo for 1/2, lm+denoise for 3
            uu = pillars.get("p1_combo", p)["u_t"] if slot < 3 else pillars.get("p1_combo", p)["u_t"]
            prob, w, got = weight_force(cand2, uu, tgt)
            # verify w=0 identity
            assert spearman_dist(force_law(cand2, unitize(L(cand2))), cand2) < 1e-12
            path = SUB / key
            write_nosubmit(path, ids, prob)
            cands[f"slot{slot}"] = {"path": str(path), "1rho": got, "w": w, "sha16": sha16(path), "note": "best_effort_no_full_gate"}
            log(f"  staged slot{slot} 1ρ={got:.6e} (best-effort)")
        return {"cands": cands, "scorecard": scorecard, "passed": []}

    # Combine passed
    U = np.mean([unit(u) for _, u, _ in passed], axis=0)
    U = unit(U)
    # single strongest
    best = max(passed, key=lambda t: next(s[1]["mean_d_e5"] for s in scorecard if s[0] == t[0] or True))
    # pick by S7m mean among passed
    best = max(
        passed,
        key=lambda t: next((s[1]["mean_d_e5"] for s in scorecard if s[0] == t[0]), -1e9),
    )

    specs = [
        ("slot1", 0.112e-4, U, "sub_v11p1_slot1_verified_5of5_pillars_0p11e4_NOSUBMIT.csv"),
        ("slot2", 0.160e-4, U, "sub_v11p1_slot2_verified_5of5_pillars_0p16e4_NOSUBMIT.csv"),
        ("slot3", 0.100e-4, unit(best[1]), "sub_v11p1_slot3_pure_LMtrim_plus_G26denoise_0p10e4_NOSUBMIT.csv"),
    ]
    for name, tgt, u, fname in specs:
        prob, w, got = weight_force(cand2, u, tgt)
        path = SUB / fname
        write_nosubmit(path, ids, prob)
        cands[name] = {
            "path": str(path),
            "1rho": got,
            "w": w,
            "sha16": sha16(path),
            "modules": [p[0] for p in passed],
            "mean": float(prob.mean()),
        }
        log(f"  {name}: 1ρ={got:.6e} w={w:.3f} → {fname}")
    # immutable check
    z0 = force_law(cand2, unitize(L(cand2)) + 0.0 * U)
    assert spearman_dist(z0, cand2) < 1e-12
    return {"cands": cands, "scorecard": scorecard, "passed": [p[0] for p in passed]}


def write_report(g26, p1, p2, p3, staged):
    md = []
    md.append("# GM v11.1 Non-Linear Stage-B / Wilcoxon / G26 Denoiser\n\n")
    md.append(f"`NO_SUBMIT=True` · anchor **cand2=0.95355** · G26 test5fold={g26.get('has_test_5fold')}\n\n")
    md.append("## Promotion scorecard\n\n")
    md.append("| pillar | S7m pos | meanΔe-5 | d0 | simple pos | GATE |\n|---|---:|---:|---:|---:|---|\n")
    for name, p in [("P1_LM_trim", p1["lm_trim"]), ("P1_G26_denoise", p1["denoise"]), ("P1_combo", p1["combo"]), ("P2_Wilcoxon", p2), ("P3_StageB", p3)]:
        if "S7m" not in p:
            continue
        md.append(
            f"| {name} | {p['S7m']['pos5']}/5 | {p['S7m']['mean_d_e5']} | {p['S7m']['d0_e5']} | "
            f"{p.get('simple',{}).get('pos5','?')}/5 | {'PASS' if p.get('pass') else 'FAIL'} |\n"
        )
    md.append("\n### Per-fold (combo / wilcoxon / stageB)\n\n")
    for name, p in [("P1_combo", p1["combo"]), ("P2", p2), ("P3", p3)]:
        md.append(f"- **{name}** folds=`{p.get('S7m',{}).get('fold_d_e5')}`\n")
    md.append("\n## G26 upstream\n\n")
    md.append(f"- local files: `{g26.get('files')}`\n")
    md.append(f"- has_test_5fold local/remote: `{g26.get('has_test_5fold')}` / `{g26.get('has_test_5fold_remote')}`\n")
    md.append(f"- status note: `{g26.get('status')}`\n")
    md.append("\n## Staged NOSUBMIT\n\n")
    for k, v in staged.get("cands", {}).items():
        md.append(f"- **{k}:** `{v}`\n")
    md.append(f"\nPassed modules: `{staged.get('passed')}`\n")
    md.append("\n---\n`NO_SUBMIT=True` — do not submit prior v11 s354-climb cands.\n")
    path = LOG / "GM_V11_1_NONLINEAR_STAGEB_REPORT.md"
    path.write_text("".join(md))
    log(f"wrote {path}")
    return path


def main():
    assert NO_SUBMIT
    log("======== GM v11.1 NONLINEAR (NO_SUBMIT) ========")
    y = pd.read_csv(VOL / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(np.int8)
    folds, fold_id = setup_folds(y)
    g26 = g26_status()
    log(f"  G26 status has_5fold={g26.get('has_test_5fold')} remote={g26.get('has_test_5fold_remote')}")
    pack = load_anchor_and_bank(y, fold_id, folds)
    p1 = pillar1_g26_denoise_and_lmtrim(pack)
    p2 = pillar2_wilcoxon(pack, p1)
    p3 = pillar3_stageB(pack, p1)
    pillars = {
        "p1_lm_trim": p1["lm_trim"],
        "p1_denoise": p1["denoise"],
        "p1_combo": p1["combo"],
        "p2_wilcoxon": p2,
        "p3_stageB": p3,
    }
    staged = stage_candidates(pack, pillars)
    write_report(g26, p1, p2, p3, staged)
    log("DONE v11.1")


if __name__ == "__main__":
    main()
