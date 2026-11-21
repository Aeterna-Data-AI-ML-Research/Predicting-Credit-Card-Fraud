#!/usr/bin/env python3
"""GM V16.1 — Submit Family E 0.20e-4 Slot1 + daoviet / digits-leak audit.

Step 1: live submit + poll (optional --skip-submit if already done).
Steps 2–3: local CPU diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(os.environ.get("V16_REPO", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
TMP15 = Path(os.environ.get("V15_TMP", "/tmp/s6e11_v15"))
OUT = Path(os.environ.get("V16_OUT", "/tmp/s6e11_v16/out"))
EXT = REPO / "external/nov14_recon"
SUB = REPO / "submissions"
REPORT = REPO / "logs/overnight/GM_V16_1_SLOT1_AND_DIGIT_DAOVIET_AUDIT.md"
KAG = os.environ.get("KAGGLE_BIN", "/Users/mz/Library/Python/3.14/bin/kaggle")
LIVE_CSV = SUB / "sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv"
SLOT1_SRC = SUB / "sub_v16_0_FamilyE_0p20e4_NOSUBMIT.csv"
SLOT1_DST = SUB / "sub_v16_1_LIVE_slot1_FamilyE_0p20e4.csv"
COMP = "playground-series-s6e11"


def log(msg: str) -> None:
    print(msg, flush=True)


def L(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    r = (rankdata(v, method="average") - 0.5) / len(v)
    r = np.clip(r, 1e-12, 1.0 - 1e-12)
    return np.log(r / (1.0 - r))


def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def env_kaggle(args: list[str]) -> str:
    env = os.environ.copy()
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    r = subprocess.run([KAG, *args], capture_output=True, text=True, env=env)
    return (r.stdout or "") + (r.stderr or "")


def load_sub(path: Path, ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path)
    c = "is_fraud" if "is_fraud" in df.columns else df.columns[-1]
    return df.set_index("id").loc[ids, c].to_numpy(float)


def step1_submit(skip: bool) -> dict:
    live = pd.read_csv(LIVE_CSV)
    src = pd.read_csv(SLOT1_SRC)
    assert len(src) == 299984
    assert src["is_fraud"].isna().sum() == 0
    rho = 1.0 - float(spearmanr(live["is_fraud"], src["is_fraud"]).statistic)
    assert abs(rho - 0.200e-4) < 0.005e-4, rho
    src.to_csv(SLOT1_DST, index=False)
    log(f"Step1: wrote {SLOT1_DST.name} rows=299984 1-ρ={rho:.6e}")
    out = {"file": str(SLOT1_DST), "1mrho": rho, "submitted": False, "publicScore": None}
    if skip:
        # Prefer a fresh poll so we don't pick an older 0.95388 line by mistake
        poll_txt = env_kaggle(["competitions", "submissions", "-c", COMP])
        (OUT / "submissions_poll.txt").write_text(poll_txt)
        for line in poll_txt.splitlines():
            if "sub_v16_1_LIVE_slot1" in line and "COMPLETE" in line.upper():
                for tok in line.split():
                    if tok.startswith("0.95") and len(tok) <= 8:
                        out["publicScore"] = float(tok)
                        out["submitted"] = True
                        break
                break
        log(f"Step1 skip-submit; score={out['publicScore']}")
        return out

    msg = (
        "v16.1 Slot1: ATH 0.95388 + 5-fold L7_s2026 (K=10) Family E "
        "tail-weighted 2-seed swap (1-rho=0.20e-4, 5/5 main, 19/25 sf, t=3.14)"
    )
    txt = env_kaggle(
        ["competitions", "submit", "-c", COMP, "-f", str(SLOT1_DST), "-m", msg]
    )
    log(txt[-500:])
    out["submitted"] = "Successfully submitted" in txt or "success" in txt.lower()
    score = None
    for _ in range(12):
        time.sleep(25)
        poll = env_kaggle(["competitions", "submissions", "-c", COMP])
        (OUT / "submissions_poll.txt").write_text(poll)
        for line in poll.splitlines():
            if "sub_v16_1_LIVE_slot1" in line:
                log("  " + line[:220])
                if "COMPLETE" in line.upper():
                    for tok in line.split():
                        if tok.startswith("0.95") and len(tok) <= 8:
                            score = float(tok)
                            break
                break
        if score is not None:
            break
    out["publicScore"] = score
    log(f"Step1 publicScore={score}")
    return out


def build_Bt_full(ids: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Assemble full test basis streams (logit-rank)."""
    streams: dict[str, np.ndarray] = {}

    def add(name: str, arr):
        a = np.asarray(arr, float).ravel()
        if a.shape[0] != len(ids):
            return
        if not np.isfinite(a).all():
            a = np.nan_to_num(a, nan=0.0)
        streams[name] = L(a)

    add("LIVE_095388", load_sub(LIVE_CSV, ids))
    bold = SUB / "sub_v13_slot1_PURE_goodpjw_bold095386_LIVE.csv"
    if bold.exists():
        add("bold_095386", load_sub(bold, ids))
    for p, nm in [
        (SUB / "sub_v12p3_slot1_PURE_Q17_anchor095370_LIVE.csv", "anchor_095370"),
        (SUB / "sub_v12p2_slot4_midpoint_cand2_s2026_x_anchor095370_NOSUBMIT.csv", "cand2"),
    ]:
        if p.exists():
            add(nm, load_sub(p, ids))

    # L7 / LM banks
    if (TMP15 / "banks/lm7_test_5fold.npy").exists():
        add("L7F", np.load(TMP15 / "banks/lm7_test_5fold.npy").ravel())
    if (TMP15 / "banks/lm7_test_f0.npy").exists():
        f0 = np.load(TMP15 / "banks/lm7_test_f0.npy")
        add("G26_f0", f0.ravel() if f0.ndim == 1 else f0.mean(axis=tuple(range(1, f0.ndim))))
    if (TMP15 / "out/R3_test_folds.npy").exists():
        R3 = np.load(TMP15 / "out/R3_test_folds.npy")
        # shape (5, n_test) or (n_test, 5)
        if R3.ndim == 2:
            if R3.shape[0] == 5:
                add("L7_s2026_5f", R3.mean(0))
            elif R3.shape[1] == 5:
                add("L7_s2026_5f", R3.mean(1))
    if (TMP15 / "banks/d1_v4_test.npy").exists():
        add("d1_v4", np.load(TMP15 / "banks/d1_v4_test.npy").ravel())
    if (TMP15 / "banks/megalm_oof_sha181c.npy").exists():
        # megalm may be OOF only — skip for test basis if wrong length
        pass
    if (REPO / "preds/P_super_v4_test.npy").exists():
        add("MS", np.load(REPO / "preds/P_super_v4_test.npy").ravel())
    if (REPO / "preds/P_super_v3_test.npy").exists():
        add("Psuper_v3", np.load(REPO / "preds/P_super_v3_test.npy").ravel())

    # common GB / MLP / GPD from goodpjw / preds
    for p, nm in [
        (REPO / "preds/gb_gpt2_bag5_full.npy", "GB"),
        (REPO / "external_bank/goodpjw_artifacts/gb_bag5_full.npy", "GB"),
        (REPO / "preds/ladder/glm_quad_test.npy", "GPD"),
        (REPO / "preds/mlp_test.npy", "MLP"),
    ]:
        if p.exists() and nm not in streams:
            add(nm, np.load(p).ravel())

    # LM placeholder = L7F if missing
    if "LM" not in streams and "L7F" in streams:
        streams["LM"] = streams["L7F"].copy()

    # alexchilton test streams (cap to keep regression stable)
    alex = REPO / "external_bank/fresh_ds/alexchilton_s6e11-oof-predictions"
    if alex.exists():
        tests = sorted(alex.glob("test_*.npy"))
        for i, p in enumerate(tests[:12]):
            add(f"alex_{p.stem[-8:]}", np.load(p).ravel())

    # Q17 / amanatar CSVs as streams
    for p in [
        REPO / "external_bank/amanatar_rank3_artifacts/sub_unbundled_newton_cholesky.csv",
        REPO / "external_bank/amanatar_rank3_artifacts/sub_titan_sovereign_fusion_50_50.csv",
        REPO / "external_bank/amanatar_rank3_artifacts/sub_titan_triad_40_30_30.csv",
    ]:
        if p.exists():
            add(f"Q17_{p.stem[:20]}", load_sub(p, ids))

    names = list(streams.keys())
    X = np.column_stack([streams[n] for n in names])
    log(f"Bt_full: {len(names)} streams, X={X.shape}")
    return names, X


def step2_daoviet(ids: np.ndarray) -> dict:
    names, X = build_Bt_full(ids)
    dao_dir = EXT / "daoviet_apex_1652"
    results = {}
    live = load_sub(LIVE_CSV, ids)
    for fname in ("submission_exact_095388.csv", "submission_orthogonal_apex_095370plus.csv"):
        p = dao_dir / fname
        if not p.exists():
            continue
        y = L(load_sub(p, ids))
        # Ridge for stability with collinear public streams
        lr = Ridge(alpha=1.0, fit_intercept=True).fit(X, y)
        pred = lr.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-18)
        # also OLS R² for reporting
        ols = LinearRegression().fit(X, y)
        r2_ols = float(ols.score(X, y))
        coef = ols.coef_
        order = np.argsort(-np.abs(coef))[:8]
        top8 = [(names[i], float(coef[i])) for i in order]
        resid = y - ols.predict(X)
        # LOFO-ish residual = resid (already orthogonal to span of Bt in LS sense)
        corr_live = float(np.corrcoef(resid, L(live))[0, 1]) if resid.std() > 0 else 0.0
        # correlate residual with each basis column
        resid_corrs = sorted(
            ((names[i], float(np.corrcoef(resid, X[:, i])[0, 1])) for i in range(X.shape[1])),
            key=lambda t: -abs(t[1]),
        )[:5]
        one_mrho = float(1.0 - spearmanr(load_sub(p, ids), live).statistic)
        results[fname] = {
            "1mrho_LIVE": one_mrho,
            "R2_ols": r2_ols,
            "R2_ridge": r2,
            "top8_weights": top8,
            "resid_std": float(resid.std()),
            "resid_corr_LIVE": corr_live,
            "resid_top_corrs": resid_corrs,
            "intercept": float(ols.intercept_),
        }
        log(f"  {fname}: 1-ρ={one_mrho:.4e} R²_ols={r2_ols:.6f} R²_ridge={r2:.6f}")
        log(f"    top8={[(n, round(w, 4)) for n, w in top8]}")

    # If exact_095388 highly spanned, reconstruct OOF blend proxy
    exact = results.get("submission_exact_095388.csv")
    recon = None
    if exact and exact["R2_ols"] > 0.998:
        # map top weights onto available OOF proxies
        train_p = TMP15 / "train.csv" if (TMP15 / "train.csv").exists() else REPO / "data/train.csv"
        ytr = pd.read_csv(train_p)["is_fraud"].to_numpy()
        z_ath = None
        if (TMP15 / "out/l7f_oof.npy").exists() and (TMP15 / "banks/d1_v4_oof.npy").exists():
            z_ath = L(np.load(TMP15 / "out/l7f_oof.npy").ravel()) + 0.03 * unitize(
                np.load(TMP15 / "banks/d1_v4_oof.npy").ravel()[: len(ytr)]
            )
        # approximate daoviet OOF as weighted sum of available OOF streams matching names
        oof_map = {}
        if (TMP15 / "out/l7f_oof.npy").exists():
            oof_map["L7F"] = L(np.load(TMP15 / "out/l7f_oof.npy").ravel())
            oof_map["LM"] = oof_map["L7F"]
        if (TMP15 / "out/R3_oof.npy").exists():
            oof_map["L7_s2026_5f"] = L(np.load(TMP15 / "out/R3_oof.npy").ravel())
        if (TMP15 / "banks/d1_v4_oof.npy").exists():
            oof_map["d1_v4"] = unitize(np.load(TMP15 / "banks/d1_v4_oof.npy").ravel()[: len(ytr)])
        if (REPO / "preds/P_super_v4_oof.npy").exists():
            oof_map["MS"] = L(np.load(REPO / "preds/P_super_v4_oof.npy").ravel())
        # use LIVE-locked ATH as our_ATH
        if z_ath is not None:
            # reconstruct using matched OOF columns from top weights where available
            wdict = dict(exact["top8_weights"])
            pieces = []
            wsum = 0.0
            for n, w in exact["top8_weights"]:
                if n in oof_map:
                    pieces.append(w * oof_map[n])
                    wsum += abs(w)
            if pieces:
                dao_oof = sum(pieces)
                blend = 0.5 * z_ath + 0.5 * L(dao_oof)
                folds = StratifiedKFold(5, shuffle=True, random_state=42)
                fold_aucs = []
                for tr, va in folds.split(np.zeros(len(ytr)), ytr):
                    fold_aucs.append(float(roc_auc_score(ytr[va], blend[va])))
                # 25 subfolds: 5x5 nested on main folds
                sf = []
                for tr, va in folds.split(np.zeros(len(ytr)), ytr):
                    inner = StratifiedKFold(5, shuffle=True, random_state=7)
                    for _, va2 in inner.split(np.zeros(len(va)), ytr[va]):
                        idx = va[va2]
                        sf.append(float(roc_auc_score(ytr[idx], blend[idx])))
                recon = {
                    "matched_weight_abs_sum": wsum,
                    "oof_auc_mean5": float(np.mean(fold_aucs)),
                    "oof_fold_aucs": fold_aucs,
                    "sf25_mean": float(np.mean(sf)),
                    "sf25_pos_vs_zath": None,
                }
                if z_ath is not None:
                    z_folds = [float(roc_auc_score(ytr[va], z_ath[va])) for _, va in folds.split(np.zeros(len(ytr)), ytr)]
                    recon["zath_fold_aucs"] = z_folds
                    recon["lift_mean_bps"] = float(10000 * (np.mean(fold_aucs) - np.mean(z_folds)))
                log(f"  recon blend OOF mean={recon['oof_auc_mean5']:.6f} lift_bps={recon.get('lift_mean_bps')}")
    results["_recon_blend"] = recon
    results["_basis_names"] = names
    results["_n_streams"] = len(names)
    return results


def smooth_te(keys_tr, y_tr, keys_va, m: float, prior: float):
    """Bincount smoothed TE: (sum_y + m*prior)/(cnt + m)."""
    keys_tr = keys_tr.astype(np.int64)
    keys_va = keys_va.astype(np.int64)
    ncat = int(max(keys_tr.max(), keys_va.max()) + 1)
    cnt = np.bincount(keys_tr, minlength=ncat).astype(np.float64)
    sy = np.bincount(keys_tr, weights=y_tr.astype(np.float64), minlength=ncat)
    enc = (sy + m * prior) / (cnt + m)
    return enc[keys_va]


def factorize_pair(a, b):
    """Map (a,b) pairs to dense int codes."""
    s = a.astype(np.int64) * (b.max() + 1).astype(np.int64) + b.astype(np.int64)
    # may overflow — use pandas factorize
    codes, _ = pd.factorize(pd.Series(list(zip(a.tolist(), b.tolist()))), sort=False)
    return codes.astype(np.int64)


def step3_digit_te() -> dict:
    train_p = TMP15 / "train.csv" if (TMP15 / "train.csv").exists() else REPO / "data/train.csv"
    train = pd.read_csv(train_p)
    y = train["is_fraud"].to_numpy()
    orig_p = REPO / "original/credit_card_fraud_2026.csv"
    orig = pd.read_csv(orig_p) if orig_p.exists() else None

    num_cols = [
        c
        for c in train.columns
        if c not in ("id", "is_fraud") and pd.api.types.is_numeric_dtype(train[c])
    ]
    # continuous-ish floats
    cont = [c for c in num_cols if train[c].nunique() > 50]
    log(f"Step3: {len(num_cols)} numeric, {len(cont)} continuous cols")

    # z_ATH OOF proxy
    l7f = np.load(TMP15 / "out/l7f_oof.npy").ravel()
    d1 = np.load(TMP15 / "banks/d1_v4_oof.npy").ravel()[: len(y)]
    z_ath = L(l7f) + 0.03 * unitize(d1)
    z_auc = float(roc_auc_score(y, z_ath))

    # Precompute key matrices for train (+ optionally append orig for counts only via TE on train folds)
    def make_keys(df):
        keys = {}
        for c in cont:
            x = df[c].to_numpy(float)
            # exact value via rounded string-ish float6
            keys[f"v_{c}"] = pd.factorize(np.round(x, 6))[0]
            cents = (np.round(x * 100) % 100).astype(np.int64)
            keys[f"cents_{c}"] = cents
            keys[f"cents3_{c}"] = (np.round(x * 1000) % 1000).astype(np.int64) % 100  # last 2 of 3dp
            keys[f"int_{c}"] = np.floor(x).astype(np.int64)
            # fractional part bucket
            keys[f"frac10_{c}"] = (np.round((x - np.floor(x)) * 10) % 10).astype(np.int64)
            # leading 2 digits of abs
            ax = np.abs(x) + 1e-12
            lead = (ax / 10 ** np.floor(np.log10(ax))).astype(np.float64)
            keys[f"lead2_{c}"] = np.clip((lead * 10).astype(np.int64), 10, 99)
        return keys

    Ktr = make_keys(train)
    key_names = list(Ktr.keys())

    # pairwise on top 4 continuous by variance (amount etc.)
    pair_cols = cont[:4]
    for i, a in enumerate(pair_cols):
        for b in pair_cols[i + 1 :]:
            # cents pair
            ca = (np.round(train[a].to_numpy(float) * 100) % 100).astype(np.int64)
            cb = (np.round(train[b].to_numpy(float) * 100) % 100).astype(np.int64)
            Ktr[f"pair_cents_{a[:8]}_{b[:8]}"] = (ca * 100 + cb).astype(np.int64)
    key_names = list(Ktr.keys())
    log(f"  {len(key_names)} digit/exact keys")

    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    fold_id = np.zeros(len(y), dtype=np.int32)
    for i, (_, va) in enumerate(folds.split(np.zeros(len(y)), y)):
        fold_id[va] = i

    results_m = {}
    for m in (10.0, 30.0):
        oof = np.zeros(len(y), dtype=np.float64)
        # average of all key TEs
        n_keys = 0
        oof_mat = np.zeros((len(y), len(key_names)), dtype=np.float32)
        for f in range(5):
            tr = fold_id != f
            va = fold_id == f
            ytr = y[tr]
            prior = float(ytr.mean())
            # optionally augment with orig for TE stats (same schema numeric cols)
            for j, kn in enumerate(key_names):
                ktr = Ktr[kn][tr]
                kva = Ktr[kn][va]
                if orig is not None and kn.startswith(("cents_", "int_", "frac", "lead", "v_")):
                    # rebuild key on orig for this feature if possible
                    pass  # keep train-only for speed/schema safety
                enc = smooth_te(ktr, ytr, kva, m=m, prior=prior)
                oof_mat[va, j] = enc
        # mean TE vector + shallow aggregation: rank-mean of top keys by solo AUC
        solo = []
        for j, kn in enumerate(key_names):
            try:
                a = float(roc_auc_score(y, oof_mat[:, j]))
            except Exception:
                a = 0.5
            solo.append((kn, a))
        solo.sort(key=lambda t: -t[1])
        topk = [j for j, (kn, a) in enumerate(solo[:40])]
        # remap indices
        top_names = [solo[i][0] for i in range(min(40, len(solo)))]
        top_idx = [key_names.index(n) for n in top_names]
        oof = oof_mat[:, top_idx].mean(axis=1)
        # also LOFO-orth vs [z_ath, d1]
        B = np.column_stack([unitize(z_ath), unitize(d1)])
        lr = LinearRegression().fit(B, oof)
        resid = oof - lr.predict(B)
        # fold AUCs for oof, z_ath, z_ath+resid
        fold_te, fold_z, fold_blend = [], [], []
        for f in range(5):
            va = fold_id == f
            fold_te.append(float(roc_auc_score(y[va], oof[va])))
            fold_z.append(float(roc_auc_score(y[va], z_ath[va])))
            blend = L(z_ath) + 0.15 * unitize(resid)
            fold_blend.append(float(roc_auc_score(y[va], blend[va])))
        corr = float(spearmanr(oof, z_ath).statistic)
        results_m[f"m{int(m)}"] = {
            "te_oof_auc": float(roc_auc_score(y, oof)),
            "z_ath_auc": z_auc,
            "spearman_te_zath": corr,
            "fold_te": fold_te,
            "fold_z": fold_z,
            "fold_blend": fold_blend,
            "lift_all5": all(fold_blend[i] > fold_z[i] for i in range(5)),
            "lift_mean_bps": float(10000 * (np.mean(fold_blend) - np.mean(fold_z))),
            "top10_solo_keys": solo[:10],
            "resid_auc": float(roc_auc_score(y, resid)),
            "n_keys": len(key_names),
        }
        log(
            f"  m={m}: TE_AUC={results_m[f'm{int(m)}']['te_oof_auc']:.5f} "
            f"ρ(zATH)={corr:.4f} blend_lift_bps={results_m[f'm{int(m)}']['lift_mean_bps']:.2f} "
            f"all5={results_m[f'm{int(m)}']['lift_all5']}"
        )
        log(f"    top keys: {solo[:5]}")

    # kernel FE summary pointers
    kernels = {
        "digits": str(EXT / "kernels/digits/s6e11-fraud-part-2-digits-leak-the-label.ipynb"),
        "daoviet": str(EXT / "kernels/daoviet/s6e11-on-the-journey-to-your-first-top-100.ipynb"),
        "asterios": str(EXT / "kernels/asterios/s6e11-how-much-of-the-public-top-is-real.ipynb"),
    }
    return {"z_ath_auc": z_auc, "by_m": results_m, "kernels": kernels, "n_cont": len(cont)}


def write_report(slot1: dict, dao: dict, digit: dict) -> None:
    lines = []
    lines.append("# GM V16.1 — Slot1 Family E Submit + Digits / daoviet Audit")
    lines.append("")
    lines.append(f"**Submitted:** `{SLOT1_DST.name}`  ")
    lines.append(f"**Public LB:** `{slot1.get('publicScore')}`  ")
    lines.append(f"**1−ρ(LIVE):** `{slot1.get('1mrho'):.6e}`  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 1 — Live Slot 1")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|------|------:|")
    lines.append(f"| File | `{SLOT1_DST.name}` |")
    lines.append(f"| Rows / NaNs | 299984 / 0 |")
    lines.append(f"| 1−ρ(LIVE_095388) | {slot1.get('1mrho'):.6e} |")
    lines.append(f"| Public score | **{slot1.get('publicScore')}** |")
    lines.append(f"| vs LIVE 0.95388 | {('+' if slot1.get('publicScore') and slot1['publicScore']>=0.95388 else '')}{((slot1.get('publicScore') or 0)-0.95388)*10000:.1f} bps |")
    lines.append("")
    lines.append("Message: Family E tail-weighted 5-fold L7_s2026 2-seed swap @ 0.20e-4 (5/5 main, 19/25 sf, t=3.14).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 2 — daoviet 16:52 CSV decomposition on `Bt_full`")
    lines.append("")
    lines.append(f"Basis width: **{dao.get('_n_streams')}** streams (`{', '.join((dao.get('_basis_names') or [])[:16])}…`).")
    lines.append("")
    for fname, r in dao.items():
        if fname.startswith("_"):
            continue
        lines.append(f"### `{fname}`")
        lines.append("")
        lines.append(f"- `1−ρ(LIVE)` = **{r['1mrho_LIVE']:.6e}**")
        lines.append(f"- OLS R² = **{r['R2_ols']:.6f}** · Ridge(α=1) R² = **{r['R2_ridge']:.6f}**")
        lines.append(f"- resid std = {r['resid_std']:.6f} · corr(resid, LIVE) = {r['resid_corr_LIVE']:.4f}")
        lines.append("- Top-8 OLS weights:")
        lines.append("")
        lines.append("| stream | weight |")
        lines.append("|--------|-------:|")
        for n, w in r["top8_weights"]:
            lines.append(f"| `{n}` | {w:.6f} |")
        lines.append("")
        if r["R2_ols"] < 0.999:
            lines.append(f"R² < 0.999 → residual inspected; top residual corrs: `{r['resid_top_corrs']}`")
            lines.append("")
        else:
            lines.append("R² ≥ 0.999 → nearly spanned by known public/our basis.")
            lines.append("")
    recon = dao.get("_recon_blend")
    if recon:
        lines.append("### OOF reconstruction blend `0.5·L(z_ATH)+0.5·L(dao_recon)`")
        lines.append("")
        lines.append(f"- 5-fold mean AUC = **{recon['oof_auc_mean5']:.6f}**")
        lines.append(f"- fold AUCs = {recon['oof_fold_aucs']}")
        lines.append(f"- lift vs z_ATH = **{recon.get('lift_mean_bps'):.2f} bps**")
        lines.append(f"- sf25 mean = {recon['sf25_mean']:.6f}")
        lines.append("")
    else:
        lines.append("_OOF reconstruct skipped (R²≤0.998 or insufficient OOF-mapped streams)._")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Step 3 — Digits / exact-value TE vs quantile-binned `z_ATH`")
    lines.append("")
    lines.append("### Kernel FE (downloaded)")
    lines.append("")
    lines.append("- **digits-leak (dhananjayk2609):** keys `v_` (exact value), `cents_`, `int_`, `ten_`, `dec_`, `l1_`/`l2_` leading digits; OOF TE with smoothing m≈20; shallow 4-leaf LGBM / depth-2 XGB. Claims classic track plateaus ~0.947 while digit TE lifts ~0.950.")
    lines.append("- **daoviet journey:** multi-environment file discovery + public ladder blend / GP-DVAE orthogonal apex narrative; pairs with dataset CSVs (exact_095388 / orthogonal_apex).")
    lines.append("- **asterios:** public LB = 20% noise, winner’s curse, nested CV for private survival — not a digit recipe.")
    lines.append("")
    lines.append(f"`z_ATH` OOF proxy AUC = **{digit.get('z_ath_auc'):.6f}** (L(L7F)+0.03·d1). Continuous cols used: {digit.get('n_cont')}.")
    lines.append("")
    for mk, r in (digit.get("by_m") or {}).items():
        lines.append(f"### Smoothing `{mk}`")
        lines.append("")
        lines.append(f"| metric | value |")
        lines.append(f"|--------|------:|")
        lines.append(f"| Digit/exact TE OOF AUC | {r['te_oof_auc']:.6f} |")
        lines.append(f"| spearman(TE, z_ATH) | {r['spearman_te_zath']:.4f} |")
        lines.append(f"| resid AUC after LOFO-orth([z_ATH,d1]) | {r['resid_auc']:.6f} |")
        lines.append(f"| blend lift mean (bps) | {r['lift_mean_bps']:.2f} |")
        lines.append(f"| lift on all 5 folds? | **{r['lift_all5']}** |")
        lines.append("")
        lines.append(f"- fold TE: {r['fold_te']}")
        lines.append(f"- fold z_ATH: {r['fold_z']}")
        lines.append(f"- fold blend: {r['fold_blend']}")
        lines.append(f"- top solo keys: `{r['top10_solo_keys'][:8]}`")
        lines.append("")
    lines.append("### Interpretation")
    lines.append("")
    lines.append("Our L7/GPT-2 path uses coarse 100-bin quantile tokens (V≈1372), which erase cents/mantissas. If digit/exact TE residual lifts all 5 folds after orthogonalization to `z_ATH`, that is a genuine orthogonal edge for Stage-B; otherwise the digit signal is already absorbed by the public ATH stack.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("```")
    lines.append(f"{SLOT1_DST}")
    lines.append(f"{EXT}/daoviet_apex_1652/")
    lines.append(f"{EXT}/kernels/{{digits,daoviet,asterios}}/")
    lines.append(f"{REPORT}")
    lines.append("```")
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    log(f"wrote {REPORT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-submit", action="store_true", help="Assume Slot1 already submitted; still poll/parse score")
    ap.add_argument("--skip-digit", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (REPO / "logs/overnight").mkdir(parents=True, exist_ok=True)

    slot1 = step1_submit(skip=args.skip_submit)
    # if skip and no score, hardcode from morning poll if present
    if slot1.get("publicScore") is None:
        # one quick poll
        poll = env_kaggle(["competitions", "submissions", "-c", COMP])
        (OUT / "submissions_poll.txt").write_text(poll)
        for line in poll.splitlines():
            if "sub_v16_1_LIVE_slot1" in line and "COMPLETE" in line.upper():
                for tok in line.split():
                    if tok.startswith("0.95") and len(tok) <= 8:
                        slot1["publicScore"] = float(tok)
                        break

    ids = pd.read_csv(LIVE_CSV)["id"].to_numpy()
    log("=== Step 2 daoviet decomposition ===")
    dao = step2_daoviet(ids)
    if args.skip_digit:
        digit = {"z_ath_auc": None, "by_m": {}, "kernels": {}, "n_cont": 0}
    else:
        log("=== Step 3 digit/exact TE ===")
        digit = step3_digit_te()

    payload = {"slot1": slot1, "daoviet": dao, "digit": digit}
    # make JSON-safe
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    (OUT / "v16_1_audit.json").write_text(json.dumps(payload, indent=2, default=conv))
    (REPO / "logs/overnight/v16_1_audit.json").write_text(json.dumps(payload, indent=2, default=conv))
    write_report(slot1, dao, digit)
    log("DONE")


if __name__ == "__main__":
    main()
