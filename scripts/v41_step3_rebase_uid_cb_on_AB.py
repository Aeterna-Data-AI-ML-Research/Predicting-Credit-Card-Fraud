#!/usr/bin/env python3
"""V41 STEP 3 — Rebase proprietary uid/CB onto Step_AB (0.95569).

STRICT NO_SUBMIT=True. Writes logs/v41/STEP3_AUDIT.md (+.json) then STOPS.
Does NOT run Step 4.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp, pearsonr, rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
N_TR, N_TE = 699_960, 299_984
T0 = time.time()

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
SSD = Path("/tmp/s6e11_v39_ssd")
EXT = ROOT / "external/v41_nov23_frontier"
PART12 = ROOT / "external/nov08_part12"
LOG = ROOT / "logs/v41"
OUT = ROOT / "out/v41_staged"
SSD_OUT = Path("/tmp/s6e11_v41_ssd/out/v41_staged")
DATA = ROOT / "data" if (ROOT / "data/test.csv").exists() else SSD / "data"
FOLDS = SSD / "folds/folds.csv" if (SSD / "folds/folds.csv").exists() else ROOT / "folds/folds.csv"

SHA_AB = "3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038"
SHA_AA = "51706855fa6de98ef88c1d02bebfc24ab340092d05eb513e227e048041c0346c"
SHA_Z = "ee61b50611326eb668033f088f8803e28541874c1cff2cf073334ea7d0f05a11"
SHA_AREEB68 = "3b9fe6d72e3102101ab33d35a17cdb90b1e41de40c3792a7dc1a1cd619d1bc8c"
SHA_Q = "60ec5bd11eca0445c3b0b222a2715d3ffeee9594968cbce56061db16d87f5c62"

QUARANTINE_SUBSTR = ("zanch", "v10_lgb", "soft_diff", "gb_soft_bag3", "G26", "1fold", "fold0_only")


def log(msg: str):
    print(f"[{time.time()-T0:5.1f}s] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def U(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v: np.ndarray) -> np.ndarray:
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def one_mrho(a, b) -> float:
    return 1.0 - float(spearmanr(a, b).statistic)


def load_sub(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    col = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
    return df["id"].to_numpy(), df[col].to_numpy(float)


def zone_mask(test: pd.DataFrame) -> np.ndarray:
    flags = [
        "is_new_merchant",
        "is_foreign_transaction",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    clean = test.cvv_retry_count.eq(0) & test.prior_disputes.eq(0) & ~test[flags].astype(bool).any(axis=1)
    risk, velocity = test.merchant_risk_score, test.velocity_score
    return (
        (risk < 13.1)
        | ((velocity <= 4.1) & (risk <= 20))
        | (clean & (((velocity <= 4.1) & (risk <= 33.8)) | ((velocity <= 6) & (risk <= 25))))
    ).to_numpy()


def orth_vs(cols: list[np.ndarray], target: np.ndarray, ridge: float = 50.0) -> np.ndarray:
    X = np.column_stack([unit(c) for c in cols])
    y = unit(target)
    beta = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)
    return unit(y - X @ beta)


def apply_zr(score: np.ndarray, zones: np.ndarray, u_ab: np.ndarray) -> np.ndarray:
    """Keep zones bottom-ranked in Step_AB internal order; blend non-zones freely."""
    out = np.asarray(score, float).copy()
    if zones.any() and (~zones).any():
        floor = float(np.min(out[~zones])) - 1.0
        # floor + U_AB preserves AB order among zones (NOT floor - U_AB, which reverses)
        out[zones] = floor + u_ab[zones]
    return out


def write_u_sub(path: Path, ids: np.ndarray, scores: np.ndarray) -> str:
    """Write uniform-rank submission in [0,1]."""
    p = U(scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(index=False, lineterminator="\n")
    path.write_bytes(text.encode())
    SSD_OUT.mkdir(parents=True, exist_ok=True)
    (SSD_OUT / path.name).write_bytes(text.encode())
    return sha256_file(path)


def hygiene(name: str, oof_p: Path, te_p: Path, y: np.ndarray) -> dict:
    oof = np.load(oof_p).ravel().astype(float)
    te = np.load(te_p).ravel().astype(float)
    assert oof.shape == (N_TR,), (name, oof.shape)
    assert te.shape == (N_TE,), (name, te.shape)
    nan_inf = int(np.isnan(oof).sum() + np.isinf(oof).sum() + np.isnan(te).sum() + np.isinf(te).sum())
    auc = float(roc_auc_score(y, oof))
    raw_disp = float(te.std() / (oof.std() + 1e-12))
    ks = float(ks_2samp(oof, te).statistic)
    return {
        "name": name,
        "oof_path": str(oof_p),
        "test_path": str(te_p),
        "oof_shape": list(oof.shape),
        "test_shape": list(te.shape),
        "nan_inf": nan_inf,
        "oof_auc": auc,
        "raw_disp": raw_disp,
        "KS_raw": ks,
    }


def path_quarantine_ok(paths: list[Path]) -> dict:
    hits = []
    for p in paths:
        s = str(p).lower()
        for q in QUARANTINE_SUBSTR:
            if q.lower() in s:
                hits.append({"path": str(p), "substr": q})
        # also scan tiny sidecar json/log if present
        for side in p.parent.glob(p.stem.split("_oof")[0].split("_test")[0] + "*"):
            if side.suffix in {".json", ".txt", ".log", ".py"} and side.stat().st_size < 200_000:
                txt = side.read_text(errors="ignore").lower()
                for q in QUARANTINE_SUBSTR:
                    if q.lower() in txt:
                        hits.append({"path": str(side), "substr": q})
    return {"quarantine_hits": hits, "clean": len(hits) == 0}


def fold_delta_auc(y, base, cand, folds) -> dict:
    d = []
    for k in range(5):
        m = folds == k
        d.append(roc_auc_score(y[m], cand[m]) - roc_auc_score(y[m], base[m]))
    d = np.asarray(d, float)
    return {
        "pos5": int((d > 0).sum()),
        "mean_e5": float(d.mean() * 1e5),
        "min_e5": float(d.min() * 1e5),
        "folds_e5": [float(x * 1e5) for x in d],
        "tstat": float(d.mean() / (d.std(ddof=1) / np.sqrt(5) + 1e-18)),
    }


def sf25_pos(y, base, cand, folds) -> int:
    """25 stratified subfolds: 5 outer × 5 inner on each outer val."""
    pos = 0
    for k in range(5):
        m = folds == k
        yk, bk, ck = y[m], base[m], cand[m]
        if yk.sum() < 10 or (1 - yk).sum() < 10:
            continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + k)
        idx = np.arange(len(yk))
        for _, va in skf.split(idx, yk):
            if roc_auc_score(yk[va], ck[va]) > roc_auc_score(yk[va], bk[va]):
                pos += 1
    return pos


def lofo_orth_residual(base_o: np.ndarray, prop_o: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """LOFO: residual of L(prop) after OLS on [1, L(base)] fit on train_idx only."""
    Lb, Lp = L(base_o), L(prop_o)
    res = np.zeros(len(base_o), float)
    for k in range(5):
        tr, va = folds != k, folds == k
        X = np.column_stack([np.ones(tr.sum()), Lb[tr]])
        beta, _, _, _ = np.linalg.lstsq(X, Lp[tr], rcond=None)
        Xv = np.column_stack([np.ones(va.sum()), Lb[va]])
        res[va] = unit(Lp[va] - Xv @ beta)
    return res


def md_table(df: pd.DataFrame, fmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join([""] + [str(c) for c in cols]) + " |",
        "| " + " | ".join(["---"] * (len(cols) + 1)) + " |",
    ]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:{fmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    assert NO_SUBMIT is True
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log("STEP3 start NO_SUBMIT=True")

    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(float)
    folds = pd.read_csv(FOLDS)["fold"].to_numpy(int)
    assert len(y) == N_TR and len(folds) == N_TR
    test = pd.read_csv(DATA / "test.csv")
    ids = test["id"].to_numpy()
    zones = zone_mask(test)
    n_z = int(zones.sum())
    assert n_z == 16_730, n_z
    outside = ~zones

    # --- paths ---
    p_uid_o, p_uid_t = SSD / "out/uid_v2_oof.npy", SSD / "out/uid_v2_test.npy"
    p_cb_o, p_cb_t = SSD / "out/uid_catboost_oof.npy", SSD / "out/uid_catboost_test.npy"
    p_ent_o, p_ent_t = SSD / "out/uid_entity_oof.npy", SSD / "out/uid_entity_test.npy"
    p_fb_o = SSD / "preds/z_fullbank_oof.npy"
    p_fb_t = SSD / "preds/z_fullbank_test.npy"
    p_wide_o = PART12 / "wide_lm_stack_nested.npy"
    p_wide_t = PART12 / "wide_lm_stack_test.npy"
    p_bo_o = SSD / "preds/B_clean_oof.npy"
    p_bo_t = SSD / "preds/B_clean_test.npy"

    ab_path = OUT / "F1b_pub_Step_AB_095569.csv"
    if not ab_path.exists():
        ab_path = EXT / "goodpjw_95569/submission.csv"
    aa_path = EXT / "goodpjw_95569/submission_step_aa.csv"
    z_path = EXT / "goodpjw_95569/submission_step_z.csv"
    areeb68_path = EXT / "areeb_95568/submission.csv"
    q_path = Path("/tmp/s6e11_v39_ssd/out/v40_ladder/S01_PUB_areeb_95567.csv")

    assert sha256_file(ab_path) == SHA_AB
    assert sha256_file(aa_path) == SHA_AA
    assert sha256_file(z_path) == SHA_Z
    assert sha256_file(areeb68_path) == SHA_AREEB68
    assert sha256_file(q_path) == SHA_Q

    _, ab = load_sub(ab_path)
    _, aa = load_sub(aa_path)
    _, zz = load_sub(z_path)
    _, areeb68 = load_sub(areeb68_path)
    _, quarter = load_sub(q_path)
    u_ab = U(ab)
    Lab, Laa, Lz, Lar, Lq = map(L, [ab, aa, zz, areeb68, quarter])

    uid_o = np.load(p_uid_o).ravel().astype(float)
    uid_t = np.load(p_uid_t).ravel().astype(float)
    cb_o = np.load(p_cb_o).ravel().astype(float)
    cb_t = np.load(p_cb_t).ravel().astype(float)

    # ========== 1. Provenance & hygiene ==========
    hyg = [
        hygiene("uid_v2 (LGBM uidA–E)", p_uid_o, p_uid_t, y),
        hygiene("uid_catboost (CB)", p_cb_o, p_cb_t, y),
        hygiene("uid_entity (LGBM v1, 0.95550 heritage)", p_ent_o, p_ent_t, y),
    ]
    qcheck = path_quarantine_ok([p_uid_o, p_uid_t, p_cb_o, p_cb_t, p_ent_o, p_ent_t,
                                 ROOT / "scripts/build_max_leap_sub.py",
                                 ROOT / "scripts/v40_public_orth_ladder.py"])
    # also scan source texts for quarantine
    src_hits = []
    for sp in [ROOT / "scripts/build_max_leap_sub.py", ROOT / "scripts/v40_public_orth_ladder.py"]:
        if sp.exists():
            txt = sp.read_text(errors="ignore")
            for q in QUARANTINE_SUBSTR:
                if re.search(q, txt, re.I):
                    src_hits.append({"script": str(sp), "substr": q})
    qcheck["source_hits"] = src_hits
    qcheck["clean"] = qcheck["clean"] and len(src_hits) == 0

    provenance = {
        "uid_v2": {
            "builder": "inline V39 session → saved uid_v2_{oof,test}.npy (LightGBM uidA–E + TE/cnt/z)",
            "meta": json.loads((SSD / "out/uid_v2_meta.json").read_text()) if (SSD / "out/uid_v2_meta.json").exists() else {},
            "depends": ["data/train.csv", "data/test.csv", "folds", "NO zanch/soft_diff"],
        },
        "uid_catboost": {
            "builder": "scripts/build_max_leap_sub.py",
            "depends": ["uid1/2/3 CatBoost features", "NO zanch/soft_diff"],
            "note": "Solo OOF AUC ~0.988 is extremely high (TE-rich CatBoost); treat as overfit-risk channel, use only via orth/clip.",
        },
        "uid_entity_v1": {
            "builder": "V39 gemini uid_entity → ATH_v2 0.95550 heritage (sub_v39_S4 / MAX_uid_peak)",
            "ATH_0p95550": "submissions/sub_v39_MAX_uid_peak_a0365.csv",
        },
        "v40_ladder_blend_operator": {
            "T1_areeb_uid_a0pXX": "expit( L(areeb) + α * orth_ridge50([areeb,gpw,ath], uid_v2_test) )",
            "T2_areeb_cb_a0pXX": "expit( L(areeb) + α * orth_ridge50([areeb,gpw,ath], uid_catboost_test) )",
            "T3_areeb_mixuid_a0pXX": "expit( (1-α)*L(areeb) + α*L(uid_v2_test) )  # raw macro, not orth",
            "S13_S15_micro": "same orth axis with omr∈{0.25e-4..0.50e-4}",
            "prop_form_in_T1T2": "orthogonalized residual axis (NOT raw model prediction in the α term)",
            "created_by": "ad-hoc shell after v40_public_orth_ladder.py (transcript bold prep)",
        },
        "hygiene": hyg,
        "quarantine_assert": qcheck,
    }
    log(f"hygiene uid_auc={hyg[0]['oof_auc']:.5f} cb_auc={hyg[1]['oof_auc']:.5f} quarantine_clean={qcheck['clean']}")

    # ========== 2. OOF baselines + audits ==========
    fb_o = np.load(p_fb_o).ravel().astype(float)
    wide_o = np.load(p_wide_o).ravel().astype(float)
    B = np.load(p_bo_o)
    assert B.shape == (N_TR, 8)
    Bo = np.mean(np.column_stack([L(B[:, i]) for i in range(8)]), axis=1)

    base_aucs = {
        "fullbank": float(roc_auc_score(y, fb_o)),
        "wide_lm_stack_nested_part12": float(roc_auc_score(y, wide_o)),
        "Bo_meanL_B_clean8": float(roc_auc_score(y, Bo)),
        "ATH_twin": float(roc_auc_score(y, np.load(SSD / "preds/z_ATH_twin_oof.npy").ravel())),
    }
    for k, v in base_aucs.items():
        if k in ("fullbank", "wide_lm_stack_nested_part12"):
            assert v > 0.9524, (k, v)
    # primary OOF base = fullbank (highest honest)
    base_o = fb_o
    base_name = "fullbank"
    log(f"baselines {base_aucs}")

    # also evaluate on wide_lm as secondary
    bases = {"fullbank": fb_o, "wide_lm_part12": wide_o}

    alphas_dir = [0.02, 0.04, 0.07, 0.10, 0.15]
    alphas_clip = [0.04, 0.07, 0.10, 0.15]
    props = {
        "uid": (uid_o, uid_t),
        "cb": (cb_o, cb_t),
    }
    # OOF-optimal uid+CB mix via LOFO orth @α=0.07 (avoids raw CB 0.988 TE-inflated AUC dominating)
    mix_grid = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
    best_mix, best_mix_score = 0.5, -1e9
    Lb0 = L(base_o)
    for w in mix_grid:
        mix = expit((1 - w) * L(uid_o) + w * L(cb_o))
        res = lofo_orth_residual(base_o, mix, folds)
        cand = expit(Lb0 + 0.07 * res)
        st = fold_delta_auc(y, base_o, cand, folds)
        # prefer stable: pos5 then mean then min
        score = st["pos5"] * 1e3 + st["mean_e5"] + 0.1 * st["min_e5"]
        if score > best_mix_score:
            best_mix_score, best_mix = score, w
    mix_o = expit((1 - best_mix) * L(uid_o) + best_mix * L(cb_o))
    mix_t = expit((1 - best_mix) * L(uid_t) + best_mix * L(cb_t))
    props["uid_cb_mix"] = (mix_o, mix_t)
    log(f"best uid+CB mix w_cb={best_mix} lofo_score={best_mix_score:.3f}")

    oof_audit = {"baselines": base_aucs, "primary_base": base_name, "uid_cb_mix_w_cb": best_mix, "rows": []}

    for bname, bvec in bases.items():
        Lb = L(bvec)
        for pname, (po, _pt) in props.items():
            for sign in (+1, -1):
                # a) direct macro
                for a in alphas_dir:
                    if sign > 0:
                        cand = expit((1 - a) * Lb + a * L(po))
                    else:
                        cand = expit((1 - a) * Lb + a * L(-po))
                    st = fold_delta_auc(y, bvec, cand, folds)
                    # sf25 only on primary base to keep runtime ~2–3 min
                    sf = sf25_pos(y, bvec, cand, folds) if bname == "fullbank" else -1
                    oof_audit["rows"].append({
                        "mode": "direct_macro", "base": bname, "prop": pname, "sign": sign, "alpha": a,
                        **st, "sf25": sf,
                    })
                # b) lofo orth
                res = lofo_orth_residual(bvec, po, folds)
                for a in alphas_dir:
                    cand = expit(Lb + sign * a * res)
                    st = fold_delta_auc(y, bvec, cand, folds)
                    sf = sf25_pos(y, bvec, cand, folds) if bname == "fullbank" else -1
                    oof_audit["rows"].append({
                        "mode": "lofo_orth", "base": bname, "prop": pname, "sign": sign, "alpha": a,
                        **st, "sf25": sf,
                    })
                # c) clipped residual
                resid = L(po) - Lb
                for clip_p in (0.985, 0.99):
                    cap = float(np.quantile(np.abs(resid), clip_p))
                    rclip = np.clip(resid, -cap, cap)
                    for a in alphas_clip:
                        cand = expit(Lb + sign * a * rclip)
                        st = fold_delta_auc(y, bvec, cand, folds)
                        # sf25 for clip99 on fullbank only
                        sf = sf25_pos(y, bvec, cand, folds) if (bname == "fullbank" and clip_p == 0.99) else -1
                        oof_audit["rows"].append({
                            "mode": f"clip_p{int(round(clip_p * 1000))}", "base": bname, "prop": pname,
                            "sign": sign, "alpha": a, "cap": cap, **st, "sf25": sf,
                        })

    # top OOF rows
    rows_df = pd.DataFrame(oof_audit["rows"])
    top = rows_df.sort_values(["pos5", "mean_e5", "min_e5"], ascending=False).head(25)
    oof_audit["top25"] = top.to_dict(orient="records")
    # best clipped vs unclipped at same α for uid/cb on fullbank +
    clip_compare = []
    for pname in ("uid", "cb", "uid_cb_mix"):
        for a in alphas_clip:
            sub = rows_df[(rows_df.base == "fullbank") & (rows_df.prop == pname) & (rows_df.sign == 1) & (rows_df.alpha == a)]
            d = sub[sub["mode"] == "direct_macro"]
            c985 = sub[sub["mode"] == "clip_p985"]
            c99 = sub[sub["mode"] == "clip_p990"]
            clip_compare.append({
                "prop": pname, "alpha": a,
                "direct_mean_e5": float(d["mean_e5"].iloc[0]) if len(d) else None,
                "clip985_mean_e5": float(c985["mean_e5"].iloc[0]) if len(c985) else None,
                "clip99_mean_e5": float(c99["mean_e5"].iloc[0]) if len(c99) else None,
                "direct_min_e5": float(d["min_e5"].iloc[0]) if len(d) else None,
                "clip99_min_e5": float(c99["min_e5"].iloc[0]) if len(c99) else None,
            })
    oof_audit["clip_vs_direct_fullbank_pos"] = clip_compare
    log(f"OOF rows={len(rows_df)} top pos5={int(top.iloc[0]['pos5'])} mean={top.iloc[0]['mean_e5']:.3f}")

    # ========== 3. Test-space geometry vs Step_AB ==========
    Luid, Lcb = L(uid_t), L(cb_t)
    d_uid = Luid - Lab
    d_cb = Lcb - Lab
    climb = {
        "d_AB_AA": Lab - Laa,
        "d_AB_Z": Lab - Lz,
        "d_areeb_AA": Lar - Laa,
    }
    align = {}
    for cname, cv in climb.items():
        for pname, pv in (("uid", d_uid), ("cb", d_cb)):
            x, yv = pv[outside], cv[outside]
            pear = float(pearsonr(x, yv).statistic)
            cos = float(np.dot(unit(x), unit(yv)))
            align[f"{pname}_vs_{cname}"] = {"pearson_nonzone": pear, "cosine_nonzone": cos}
    # tree re-dilution: negative projection onto d_AB_Z?
    align["tree_redilution_check"] = {
        "uid_proj_on_d_AB_Z_negative": align["uid_vs_d_AB_Z"]["pearson_nonzone"] < 0,
        "cb_proj_on_d_AB_Z_negative": align["cb_vs_d_AB_Z"]["pearson_nonzone"] < 0,
        "note": "If True, blending that prop without orth may reverse Z→AB climb (tree re-dilution).",
    }

    # zone ranks of raw props
    zone_geom = {
        "n_zones": n_z,
        "uid_zone_U": {
            "min": float(U(uid_t)[zones].min()), "median": float(np.median(U(uid_t)[zones])),
            "p90": float(np.quantile(U(uid_t)[zones], 0.9)), "p99": float(np.quantile(U(uid_t)[zones], 0.99)),
            "max": float(U(uid_t)[zones].max()),
            "n_gt_0558": int((U(uid_t)[zones] > 0.0558).sum()),
            "n_gt_050": int((U(uid_t)[zones] > 0.50).sum()),
        },
        "cb_zone_U": {
            "min": float(U(cb_t)[zones].min()), "median": float(np.median(U(cb_t)[zones])),
            "p90": float(np.quantile(U(cb_t)[zones], 0.9)), "p99": float(np.quantile(U(cb_t)[zones], 0.99)),
            "max": float(U(cb_t)[zones].max()),
            "n_gt_0558": int((U(cb_t)[zones] > 0.0558).sum()),
            "n_gt_050": int((U(cb_t)[zones] > 0.50).sum()),
        },
        "AB_zone_U_max": float(u_ab[zones].max()),
    }

    # unmasked vs Zr-preserved blend at α=0.07 uid
    a_demo = 0.07
    blend_unmasked = (1 - a_demo) * Lab + a_demo * Luid
    blend_zr = apply_zr(blend_unmasked.copy(), zones, u_ab)
    # for unmasked, just use blend; for zr use apply_zr
    u_un = U(blend_unmasked)
    u_zr = U(blend_zr)
    zone_jump = {
        "alpha_demo": a_demo,
        "prop": "uid",
        "unmasked_zone_gt_0558": int((u_un[zones] > 0.0558).sum()),
        "unmasked_zone_gt_050": int((u_un[zones] > 0.50).sum()),
        "zr_zone_gt_0558": int((u_zr[zones] > 0.0558).sum()),
        "zr_zone_gt_050": int((u_zr[zones] > 0.50).sum()),
        "1mrho_unmasked_vs_zr_e4": one_mrho(u_un, u_zr) * 1e4,
        "default": "PRESERVE Zr (Step_AB zone lock) unless Step4 proves zone fraud positives",
    }
    # also cb
    blend_un_cb = (1 - a_demo) * Lab + a_demo * Lcb
    blend_zr_cb = apply_zr(blend_un_cb.copy(), zones, u_ab)
    zone_jump["cb_unmasked_zone_gt_0558"] = int((U(blend_un_cb)[zones] > 0.0558).sum())
    zone_jump["cb_unmasked_zone_gt_050"] = int((U(blend_un_cb)[zones] > 0.50).sum())
    zone_jump["cb_1mrho_unmasked_vs_zr_e4"] = one_mrho(U(blend_un_cb), U(blend_zr_cb)) * 1e4
    log(f"zone jump uid unmasked gt0.0558={zone_jump['unmasked_zone_gt_0558']} 1-ρ e4={zone_jump['1mrho_unmasked_vs_zr_e4']:.4f}")

    # ========== 4. Stage candidates ==========
    # Prefer OOF-informed sign for staging; default + for diversity on tip unless strong evidence
    def best_sign(pname: str) -> int:
        sub = rows_df[
            (rows_df["base"] == "fullbank")
            & (rows_df["prop"] == pname)
            & (rows_df["mode"] == "lofo_orth")
            & (rows_df["alpha"] == 0.07)
        ]
        if len(sub) == 0:
            return +1
        # pick sign with higher mean_e5 among pos5>=2 else higher mean
        sub = sub.sort_values(["pos5", "mean_e5"], ascending=False)
        return int(sub.iloc[0]["sign"])

    signs = {p: best_sign(p) for p in ("uid", "cb", "uid_cb_mix")}
    log(f"staging signs {signs}")

    def make_blend(prop_t: np.ndarray, alpha: float, sign: int, clipped: bool) -> np.ndarray:
        if clipped:
            resid = L(prop_t) - Lab
            cap = float(np.quantile(np.abs(resid[outside]), 0.99))
            score = Lab + sign * alpha * np.clip(resid, -cap, cap)
        else:
            # direct macro on non-zones; Zr restore
            if sign > 0:
                score = (1 - alpha) * Lab + alpha * L(prop_t)
            else:
                score = (1 - alpha) * Lab + alpha * L(-prop_t)
        return apply_zr(score, zones, u_ab)

    # sweet-spot scan
    stage_alphas = [0.04, 0.07, 0.10, 0.15]
    sweet = []
    for pname, (_po, pt) in props.items():
        for a in stage_alphas:
            for clipped in (False, True):
                sc = make_blend(pt, a, signs[pname], clipped)
                mrho = one_mrho(U(sc), ab) * 1e4
                band = "sweet" if 0.15 <= mrho <= 1.50 else ("noop" if mrho < 0.015 else ("overshoot" if mrho > 3.0 else "edge"))
                sweet.append({"prop": pname, "alpha": a, "clipped_p99": clipped, "sign": signs[pname],
                              "1mrho_vs_AB_e4": mrho, "band": band})
    oof_audit["test_sweet_spot"] = sweet

    staged = {}

    def stage(name: str, prop_t: np.ndarray, alpha: float, sign: int, clipped: bool = False):
        sc = make_blend(prop_t, alpha, sign, clipped)
        sha = write_u_sub(OUT / name, ids, sc)
        p = U(sc)
        assert len(p) == N_TE and np.isfinite(p).all()
        assert p.min() >= 0 and p.max() <= 1
        assert np.array_equal(ids, test["id"].to_numpy())
        mrho = one_mrho(p, ab) * 1e4
        staged[name] = {
            "sha": sha, "alpha": alpha, "sign": sign, "clipped_p99": clipped,
            "1mrho_vs_AB_e4": mrho,
            "band": ("sweet" if 0.15 <= mrho <= 1.50 else ("noop" if mrho < 0.015 else ("overshoot" if mrho > 3.0 else "edge"))),
            "rows": N_TE, "nan_inf": 0,
        }
        log(f"staged {name} 1-ρe4={mrho:.4f} band={staged[name]['band']}")

    # P1 uid
    for a, tag in [(0.04, "a0p04"), (0.07, "a0p07"), (0.10, "a0p10")]:
        stage(f"P1_AB_uid_{tag}.csv", uid_t, a, signs["uid"], False)
    # best clipped uid if OOF clip99 mean better than direct at same α on fullbank
    clip_uid_better = False
    best_clip_a = 0.07
    for a in (0.04, 0.07, 0.10):
        row = next(r for r in clip_compare if r["prop"] == "uid" and r["alpha"] == a)
        if row["clip99_mean_e5"] is not None and row["direct_mean_e5"] is not None:
            if row["clip99_mean_e5"] > row["direct_mean_e5"] and row["clip99_min_e5"] >= row["direct_min_e5"] - 1.0:
                clip_uid_better = True
                best_clip_a = a
                break
    if clip_uid_better:
        stage(f"P1_AB_uid_clip99_a{str(best_clip_a).replace('.','p')}.csv", uid_t, best_clip_a, signs["uid"], True)
    else:
        # still stage one clipped reference at 0.07 for completeness
        stage("P1_AB_uid_clip99_a0p07.csv", uid_t, 0.07, signs["uid"], True)
        staged["P1_AB_uid_clip99_a0p07.csv"]["NOTE"] = "OOF did not clearly prefer clip; staged as reference"

    # P2 cb
    for a, tag in [(0.04, "a0p04"), (0.07, "a0p07"), (0.10, "a0p10")]:
        stage(f"P2_AB_cb_{tag}.csv", cb_t, a, signs["cb"], False)

    # P3 mix at 0.05 and 0.09
    for a, tag in [(0.05, "a0p05"), (0.09, "a0p09")]:
        stage(f"P3_AB_uid_cb_oofmix_{tag}.csv", mix_t, a, signs["uid_cb_mix"], False)

    # pairwise among P* + F1b + quarter
    names = sorted(staged.keys()) + ["F1b_pub_Step_AB_095569.csv", "quarter_our_095567.csv"]
    # ensure F1b present in OUT
    if not (OUT / "F1b_pub_Step_AB_095569.csv").exists():
        import shutil
        shutil.copy2(ab_path, OUT / "F1b_pub_Step_AB_095569.csv")
    # write quarter alias for matrix
    import shutil
    shutil.copy2(q_path, OUT / "quarter_our_095567.csv")

    vecs = {}
    for n in names:
        if n in staged or n.startswith("F1b") or n.startswith("quarter"):
            _, v = load_sub(OUT / n)
            vecs[n] = v
    order = [n for n in names if n in vecs]
    M = np.zeros((len(order), len(order)))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            M[i, j] = 0.0 if i == j else one_mrho(vecs[a], vecs[b]) * 1e4
    prim_mat = pd.DataFrame(M, index=order, columns=order)

    # ========== write audit ==========
    lines = []
    lines.append("# V41 STEP3 AUDIT — Rebase uid/CB onto Step_AB (0.95569)")
    lines.append("")
    lines.append(f"**NO_SUBMIT={NO_SUBMIT}** · Hard stop. Do not run Step 4 without `PROCEED TO STEP 4`.")
    lines.append("")
    lines.append("## Executive findings")
    lines.append("")
    lines.append(f"- Tip lock: `F1b_pub_Step_AB_095569.csv` sha `{SHA_AB[:16]}…` (0.95569).")
    lines.append(f"- Quarantine clean: **{qcheck['clean']}** (no zanch/v10_lgb/soft_diff in uid/CB paths or builders).")
    lines.append(f"- V40 T1/T2 operator was **`L(areeb)+α·orth([areeb,gpw,ath], prop)`**, not raw macro; T3 was raw macro.")
    lines.append(f"- OOF baselines: fullbank={base_aucs['fullbank']:.6f}, Part12 wide={base_aucs['wide_lm_stack_nested_part12']:.6f} (both >0.9524).")
    lines.append(f"- uid+CB OOF mix weight w_cb={best_mix}.")
    lines.append(f"- Staging signs from LOFO@α=0.07: {signs}.")
    lines.append("- Default **Zr preserved** on all staged P* (zone lock from Step_AB).")
    lines.append("")
    lines.append("## 1. Asset provenance & hygiene")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(provenance, indent=2)[:20000])
    lines.append("```")
    lines.append("")
    lines.append("## 2. Honest 5-fold / 25-subfold OOF audit")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"baselines": base_aucs, "uid_cb_mix_w_cb": best_mix, "top25": oof_audit["top25"],
                             "clip_vs_direct": clip_compare}, indent=2)[:50000])
    lines.append("```")
    lines.append("")
    lines.append("## 3. Test-space geometry vs Step_AB + Zr")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"alignment": align, "zone_geom": zone_geom, "zone_jump": zone_jump}, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 4. Staged P* candidates (Zr-preserved, NO_SUBMIT)")
    lines.append("")
    lines.append("### Sweet-spot scan (`1−ρ×1e−4` vs Step_AB)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(sweet, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Staged files")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(staged, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Pairwise `1−ρ` (×1e−4)")
    lines.append("")
    lines.append(md_table(prim_mat))
    lines.append("")
    lines.append("## 5. STOP")
    lines.append("")
    lines.append("Step 3 complete. **Awaiting human audit.** Reply `PROCEED TO STEP 4` to continue.")
    lines.append("")

    (LOG / "STEP3_AUDIT.md").write_text("\n".join(lines))
    payload = {
        "NO_SUBMIT": True,
        "provenance": provenance,
        "oof_audit": {
            "baselines": base_aucs,
            "uid_cb_mix_w_cb": best_mix,
            "top25": oof_audit["top25"],
            "clip_vs_direct": clip_compare,
            "n_rows": len(rows_df),
        },
        "alignment": align,
        "zone_geom": zone_geom,
        "zone_jump": zone_jump,
        "signs": signs,
        "sweet": sweet,
        "staged": staged,
        "primary_matrix_1mrho_e4": prim_mat.to_dict(),
    }
    (LOG / "STEP3_AUDIT.json").write_text(json.dumps(payload, indent=2, default=str))

    print("\n=== STRICT NO_SUBMIT=True | V41 STEP3 ===")
    print(f"quarantine_clean={qcheck['clean']}")
    print(f"baselines={base_aucs}")
    print(f"signs={signs} mix_w_cb={best_mix}")
    print(prim_mat.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWROTE {LOG/'STEP3_AUDIT.md'}")
    print("STOP. Awaiting PROCEED TO STEP 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
