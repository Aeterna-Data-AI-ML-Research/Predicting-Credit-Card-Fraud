#!/usr/bin/env python3
"""V41 STEP 4 — kabartay D27 / Zero-Fraud Zones vs Step_AB Zr mask.

STRICT NO_SUBMIT=True. Writes logs/v41/STEP4_AUDIT.md (+.json) then STOPS.
Does NOT run Step 5.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import ks_2samp, rankdata, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
N_TR, N_TE = 699_960, 299_984
BASE_RATE = 0.016699525687182123
T0 = time.time()

ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
SSD = Path("/tmp/s6e11_v39_ssd")
EXT = ROOT / "external/v41_nov23_frontier"
PART12 = ROOT / "external/nov08_part12"
LOG = ROOT / "logs/v41"
OUT = ROOT / "out/v41_staged"
PRED = ROOT / "preds"
SSD_PRED = Path("/tmp/s6e11_v41_ssd/preds")
DATA = ROOT / "data" if (ROOT / "data/test.csv").exists() else SSD / "data"
FOLDS_P = SSD / "folds/folds.csv" if (SSD / "folds/folds.csv").exists() else ROOT / "folds/folds.csv"
NB = EXT / "kabartay/s6e11-own-models-only-honest-5-fold-cv.ipynb"

FLAGS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
SHA_AB = "3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038"


def log(msg: str):
    print(f"[{time.time()-T0:5.1f}s] {msg}", flush=True)


def U(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v: np.ndarray) -> np.ndarray:
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12))


def one_mrho(a, b) -> float:
    return 1.0 - float(spearmanr(a, b).statistic)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_mask(df: pd.DataFrame) -> np.ndarray:
    ok = (df["cvv_retry_count"].to_numpy() == 0) & (df["prior_disputes"].to_numpy() == 0)
    for f in FLAGS:
        ok &= df[f].to_numpy().astype(int) == 0
    return ok


def z_goodpjw(df: pd.DataFrame) -> np.ndarray:
    """Published Zr / areeb / kabartay hand zones (identical on test = 16,730)."""
    clean = clean_mask(df)
    risk, vel = df["merchant_risk_score"].to_numpy(), df["velocity_score"].to_numpy()
    return (
        (risk < 13.1)
        | ((vel <= 4.1) & (risk <= 20.0))
        | (clean & (((vel <= 4.1) & (risk <= 33.8)) | ((vel <= 6.0) & (risk <= 25.0))))
    )


# Kabartay section-8 zone specs: (cond before risk edge, published edge, strict <)
def kabartay_zones():
    return [
        (lambda d: np.ones(len(d), bool), 13.1, True),
        (lambda d: d["velocity_score"].to_numpy() <= 4.1, 20.0, False),
        (lambda d: clean_mask(d) & (d["velocity_score"].to_numpy() <= 4.1), 33.8, False),
        (lambda d: clean_mask(d) & (d["velocity_score"].to_numpy() <= 6.0), 25.0, False),
    ]


def zone_rows_kabartay(d_fit: pd.DataFrame, y_fit: np.ndarray, d_app: pd.DataFrame) -> np.ndarray:
    """Exact kabartay zone_rows: merchant-risk edges refit on d_fit; velocity edges published."""
    z = np.zeros(len(d_app), bool)
    m_fit = d_fit["merchant_risk_score"].to_numpy()
    m_app = d_app["merchant_risk_score"].to_numpy()
    for cond, pub, strict in kabartay_zones():
        fraud = cond(d_fit) & (y_fit == 1)
        low = float(m_fit[fraud].min()) if fraud.any() else np.inf
        edge, s = (low, True) if low <= pub else (pub, strict)
        z |= cond(d_app) & ((m_app < edge) if s else (m_app <= edge))
    return z


def apply_zr_hard(scores: np.ndarray, zones: np.ndarray, u_base: np.ndarray) -> np.ndarray:
    """Hard bottom-rank: floor + U_base order among zones."""
    out = L(scores).copy()
    if zones.any() and (~zones).any():
        floor = float(np.min(out[~zones])) - 1.0
        out[zones] = floor + u_base[zones]
    elif zones.any():
        out[zones] = -1e9 + u_base[zones]
    return U(out)


def soft_penalty(scores: np.ndarray, zones: np.ndarray, lam: float) -> np.ndarray:
    return U(L(scores) - lam * zones.astype(float))


def fold_stats(y, base, cand, folds) -> dict:
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
        "tstat5": float(d.mean() / (d.std(ddof=1) / np.sqrt(5) + 1e-18)),
    }


def sf25(y, base, cand, folds) -> dict:
    diffs = []
    for k in range(5):
        m = folds == k
        yk, bk, ck = y[m], base[m], cand[m]
        if yk.sum() < 10 or (len(yk) - yk.sum()) < 10:
            continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + k)
        idx = np.arange(len(yk))
        for _, va in skf.split(idx, yk):
            diffs.append(roc_auc_score(yk[va], ck[va]) - roc_auc_score(yk[va], bk[va]))
    diffs = np.asarray(diffs, float)
    pos = int((diffs > 0).sum())
    # wilcoxon vs 0
    try:
        if np.allclose(diffs, 0):
            wp = 1.0
        else:
            wp = float(wilcoxon(diffs, alternative="greater").pvalue)
    except Exception:
        wp = float("nan")
    t25 = float(diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)) + 1e-18)) if len(diffs) else 0.0
    return {"sf25_pos": pos, "sf25_n": int(len(diffs)), "t25": t25, "wilcoxon_p_25": wp,
            "sf25_mean_e5": float(diffs.mean() * 1e5) if len(diffs) else 0.0}


def eval_mech(y, base, cand, folds, tag: str) -> dict:
    st = fold_stats(y, base, cand, folds)
    sf = sf25(y, base, cand, folds)
    return {"tag": tag, **st, **sf}


def write_u_sub(path: Path, ids: np.ndarray, scores: np.ndarray) -> tuple[str, np.ndarray]:
    p = U(scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(index=False, lineterminator="\n")
    path.write_bytes(text.encode())
    return sha256_file(path), p


def build_cell_keys(df: pd.DataFrame) -> np.ndarray:
    """Discrete cell keys for D27-style mining beyond the 4 hand zones."""
    risk = pd.cut(df["merchant_risk_score"], bins=[-np.inf, 10, 13.1, 20, 25, 33.8, 50, np.inf],
                  labels=False, duplicates="drop")
    vel = pd.cut(df["velocity_score"], bins=[-np.inf, 2, 4.1, 6, 10, np.inf],
                 labels=False, duplicates="drop")
    clean = clean_mask(df).astype(int)
    cvv = np.clip(df["cvv_retry_count"].to_numpy().astype(int), 0, 3)
    disp = np.clip(df["prior_disputes"].to_numpy().astype(int), 0, 3)
    # pack into int64 key
    key = (
        risk.astype(np.int64) * 100_000
        + vel.astype(np.int64) * 10_000
        + clean * 1_000
        + cvv * 100
        + disp
    )
    return key.to_numpy() if hasattr(key, "to_numpy") else np.asarray(key, np.int64)


def mine_zero_cells(keys_fit, y_fit, keys_app, n_min: int) -> np.ndarray:
    """Cells with 0 fraud and count>=n_min on fit → mask on app."""
    # aggregate on fit
    df = pd.DataFrame({"k": keys_fit, "y": y_fit})
    agg = df.groupby("k")["y"].agg(["sum", "count"])
    zero = agg[(agg["sum"] == 0) & (agg["count"] >= n_min)].index.to_numpy()
    if len(zero) == 0:
        return np.zeros(len(keys_app), bool)
    return np.isin(keys_app, zero)


def main():
    assert NO_SUBMIT is True
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    SSD_PRED.mkdir(parents=True, exist_ok=True)
    log("STEP4 start NO_SUBMIT=True")

    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    y = tr["is_fraud"].to_numpy(float)
    folds = pd.read_csv(FOLDS_P)["fold"].to_numpy(int)
    ids = te["id"].to_numpy()
    assert len(y) == N_TR and len(te) == N_TE

    ab_path = OUT / "F1b_pub_Step_AB_095569.csv"
    if not ab_path.exists():
        ab_path = EXT / "goodpjw_95569/submission.csv"
    assert sha256_file(ab_path) == SHA_AB
    ab = pd.read_csv(ab_path)["is_fraud"].to_numpy(float)
    u_ab = U(ab)

    fb_o = np.load(SSD / "preds/z_fullbank_oof.npy").ravel().astype(float)
    wide_o = np.load(PART12 / "wide_lm_stack_nested.npy").ravel().astype(float)
    assert abs(roc_auc_score(y, fb_o) - 0.9526207252382072) < 1e-9 or roc_auc_score(y, fb_o) > 0.9524
    assert roc_auc_score(y, wide_o) > 0.9524
    log(f"baselines fullbank={roc_auc_score(y, fb_o):.6f} part12={roc_auc_score(y, wide_o):.6f}")

    # ========== 1. Parse + overlap ==========
    lineage = {
        "notebook": str(NB),
        "section": "8 · Zero-fraud zones",
        "discovery": (
            "Hand rules published by daoviet/goodpjw/Weni Mohamed; found on FULL train "
            "(potential selection bias). kabartay mitigates by refitting each merchant-risk "
            "edge inside train_idx (just below lowest fraud in the zone-condition, never above "
            "published edge). Velocity edges stay published (residual optimism)."
        ),
        "N_min": "NOT used in kabartay hand rules; applied in our supplemental cell-miner.",
        "zones": [
            "Z1: merchant_risk_score < 13.1",
            "Z2: velocity_score <= 4.1 AND merchant_risk_score <= 20",
            "Z3: clean(8 flags) AND velocity <= 4.1 AND risk <= 33.8",
            "Z4: clean(8 flags) AND velocity <= 6.0 AND risk <= 25",
        ],
        "postprocess": (
            "OOF: zoned[val]=stack_oof.min()-1 for zone rows. "
            "Test: p[z_test] *= 1e-6 (soft bottom)."
        ),
        "reported": "Markdown: small gain every fold; 0 held-out frauds in refitted zones.",
    }

    Zg_te = z_goodpjw(te)
    Zg_tr = z_goodpjw(tr)
    # kabartay published edges on full train (no refit needed when 0 frauds → same as Zr)
    Zd27_te_pub = zone_rows_kabartay(tr, y, te)
    Zd27_tr_pub = zone_rows_kabartay(tr, y, tr)

    overlap = {
        "test": {
            "n_Z_goodpjw": int(Zg_te.sum()),
            "n_Z_D27_pub": int(Zd27_te_pub.sum()),
            "intersection": int((Zg_te & Zd27_te_pub).sum()),
            "D27_minus_goodpjw": int((Zd27_te_pub & ~Zg_te).sum()),
            "goodpjw_minus_D27": int((Zg_te & ~Zd27_te_pub).sum()),
            "identical": bool(np.array_equal(Zg_te, Zd27_te_pub)),
        },
        "train": {
            "n_Z_goodpjw": int(Zg_tr.sum()),
            "n_Z_D27_pub": int(Zd27_tr_pub.sum()),
            "intersection": int((Zg_tr & Zd27_tr_pub).sum()),
            "D27_minus_goodpjw": int((Zd27_tr_pub & ~Zg_tr).sum()),
            "goodpjw_minus_D27": int((Zg_tr & ~Zd27_tr_pub).sum()),
            "fraud_in_Z_goodpjw": int(y[Zg_tr].sum()),
            "fraud_rate_Z_goodpjw": float(y[Zg_tr].mean()) if Zg_tr.any() else None,
            "baseline_fraud_rate": float(y.mean()),
        },
    }
    log(f"overlap test inter={overlap['test']['intersection']} D27\\Zr={overlap['test']['D27_minus_goodpjw']} "
        f"train frauds in Zr={overlap['train']['fraud_in_Z_goodpjw']}")

    # ========== 2. Strict OOF leakage audit ==========
    keys_tr = build_cell_keys(tr)
    keys_te = build_cell_keys(te)

    oof_leak = {"kabartay_edge_refit": [], "cell_miner": {}}
    # kabartay OOF edge-refit held-out frauds
    n_val_zone = n_val_fraud = 0
    n_inc_val = n_inc_fraud = 0
    Zd27_oof = np.zeros(N_TR, bool)  # OOF-applied kabartay zones on train
    for k in range(5):
        trm, vam = folds != k, folds == k
        z = zone_rows_kabartay(tr.iloc[trm], y[trm], tr.iloc[vam])
        Zd27_oof[vam] = z
        nf = int(y[vam][z].sum())
        n_val_zone += int(z.sum())
        n_val_fraud += nf
        # incremental vs Zr on val
        zg = Zg_tr[vam]
        inc = z & ~zg
        n_inc_val += int(inc.sum())
        n_inc_fraud += int(y[vam][inc].sum())
        oof_leak["kabartay_edge_refit"].append({
            "fold": k,
            "n_val_zone": int(z.sum()),
            "n_val_fraud": nf,
            "val_fraud_rate": float(y[vam][z].mean()) if z.any() else 0.0,
            "n_inc_vs_Zr": int(inc.sum()),
            "n_inc_fraud": int(y[vam][inc].sum()),
        })
    oof_leak["kabartay_edge_refit_sum"] = {
        "n_val_zone": n_val_zone,
        "n_val_fraud": n_val_fraud,
        "n_inc_vs_Zr": n_inc_val,
        "n_inc_fraud": n_inc_fraud,
        "baseline": BASE_RATE,
    }
    log(f"kabartay OOF zones val_rows={n_val_zone} frauds={n_val_fraud} inc_vs_Zr={n_inc_val} inc_fraud={n_inc_fraud}")

    # cell miner across N_min
    for n_min in [20, 35, 50, 80, 120]:
        rows = []
        n_vz = n_vf = n_iv = n_if = 0
        z_oof = np.zeros(N_TR, bool)
        z_te_votes = []
        for k in range(5):
            trm, vam = folds != k, folds == k
            z_va = mine_zero_cells(keys_tr[trm], y[trm], keys_tr[vam], n_min)
            z_te_k = mine_zero_cells(keys_tr[trm], y[trm], keys_te, n_min)
            # transductive filter: keep cells with test_cnt>=2
            # recompute with test support
            df = pd.DataFrame({"k": keys_tr[trm], "y": y[trm]})
            agg = df.groupby("k")["y"].agg(["sum", "count"])
            zero = set(agg[(agg["sum"] == 0) & (agg["count"] >= n_min)].index.tolist())
            te_vc = pd.Series(keys_te).value_counts()
            keep = {c for c in zero if int(te_vc.get(c, 0)) >= 2}
            z_va = np.isin(keys_tr[vam], list(keep))
            z_te_k = np.isin(keys_te, list(keep))
            z_oof[vam] = z_va
            z_te_votes.append(z_te_k)
            zg = Zg_tr[vam]
            inc = z_va & ~zg
            n_vz += int(z_va.sum()); n_vf += int(y[vam][z_va].sum())
            n_iv += int(inc.sum()); n_if += int(y[vam][inc].sum())
            rows.append({
                "fold": k, "n_val_zone": int(z_va.sum()), "n_val_fraud": int(y[vam][z_va].sum()),
                "n_inc": int(inc.sum()), "n_inc_fraud": int(y[vam][inc].sum()),
                "n_test_zone": int(z_te_k.sum()),
            })
        # consensus test mask: in zone on majority of fold models (>=3/5)
        z_te_stack = np.stack(z_te_votes, 0)
        z_te_maj = z_te_stack.sum(0) >= 3
        oof_leak["cell_miner"][str(n_min)] = {
            "folds": rows,
            "sum_n_val_zone": n_vz,
            "sum_n_val_fraud": n_vf,
            "sum_n_inc_vs_Zr": n_iv,
            "sum_n_inc_fraud": n_if,
            "inc_fraud_rate": float(n_if / n_iv) if n_iv else 0.0,
            "test_majority_n": int(z_te_maj.sum()),
            "test_inc_vs_Zr": int((z_te_maj & ~Zg_te).sum()),
        }
        log(f"N_min={n_min}: val_zone={n_vz} val_fraud={n_vf} inc={n_iv} inc_fraud={n_if} "
            f"test_maj={z_te_maj.sum()} test_inc={int((z_te_maj & ~Zg_te).sum())}")

    # pick best N_min: maximize test_inc with 0 inc fraud preferred, else lowest fraud rate
    best_nmin, best_score = 50, -1e18
    for n_min_s, rec in oof_leak["cell_miner"].items():
        # score: prefer 0 frauds, then more incremental coverage with low rate
        score = -1e6 * rec["sum_n_inc_fraud"] + rec["test_inc_vs_Zr"] - 100 * rec["inc_fraud_rate"]
        if score > best_score:
            best_score, best_nmin = score, int(n_min_s)
    log(f"selected N_min={best_nmin}")

    # rebuild selected OOF/test masks for miner
    n_min = best_nmin
    Zmine_oof = np.zeros(N_TR, bool)
    z_te_votes = []
    for k in range(5):
        trm, vam = folds != k, folds == k
        df = pd.DataFrame({"k": keys_tr[trm], "y": y[trm]})
        agg = df.groupby("k")["y"].agg(["sum", "count"])
        zero = set(agg[(agg["sum"] == 0) & (agg["count"] >= n_min)].index.tolist())
        te_vc = pd.Series(keys_te).value_counts()
        keep = {c for c in zero if int(te_vc.get(c, 0)) >= 2}
        Zmine_oof[vam] = np.isin(keys_tr[vam], list(keep))
        z_te_votes.append(np.isin(keys_te, list(keep)))
    Zmine_te = np.stack(z_te_votes, 0).sum(0) >= 3
    Zmine_inc_te = Zmine_te & ~Zg_te
    Zmine_inc_oof = Zmine_oof & ~Zg_tr

    # kabartay OOF test mask (fit edges on full train for test = pub)
    Zd27_te = Zd27_te_pub
    Zd27_inc_te = Zd27_te & ~Zg_te  # empty if identical
    Zd27_inc_oof = Zd27_oof & ~Zg_tr

    # ========== 3. OOF evaluation ==========
    bases = {"fullbank": fb_o, "part12": wide_o}
    mech_results = []

    for bname, bvec in bases.items():
        u_b = U(bvec)
        # A1.1 Z_goodpjw alone
        cand = apply_zr_hard(bvec, Zg_tr, u_b)
        mech_results.append({**eval_mech(y, bvec, cand, folds, f"A1_Zr_goodpjw@{bname}"), "base": bname})
        # A1.2 kabartay OOF zones alone
        cand = apply_zr_hard(bvec, Zd27_oof, u_b)
        mech_results.append({**eval_mech(y, bvec, cand, folds, f"A1_kabartay_oof@{bname}"), "base": bname})
        # A1.3 union Zr ∪ kabartay_oof
        union = Zg_tr | Zd27_oof
        cand_u = apply_zr_hard(bvec, union, u_b)
        cand_z = apply_zr_hard(bvec, Zg_tr, u_b)
        # marginal of adding D27\Zr on top of Zr-already
        st_u = eval_mech(y, bvec, cand_u, folds, f"A1_union_Zr_U_kab@{bname}")
        # marginal vs Zr-applied base
        st_marg = eval_mech(y, cand_z, cand_u, folds, f"A1_marginal_kab_on_Zr@{bname}")
        mech_results.append({**st_u, "base": bname})
        mech_results.append({**st_marg, "base": bname, "note": "Δ vs Zr-already-applied"})

        # A1 mined incremental
        cand_m = apply_zr_hard(bvec, Zg_tr | Zmine_oof, u_b)
        st_mu = eval_mech(y, bvec, cand_m, folds, f"A1_union_Zr_U_mine{n_min}@{bname}")
        st_mm = eval_mech(y, cand_z, cand_m, folds, f"A1_marginal_mine{n_min}_on_Zr@{bname}")
        mech_results.append({**st_mu, "base": bname})
        mech_results.append({**st_mm, "base": bname})

        # A2 soft λ on incremental mined \ Zr (or kabartay inc if any)
        inc_mask = Zmine_inc_oof if Zmine_inc_oof.any() else Zd27_inc_oof
        for lam in [0.25, 0.50, 1.0, 2.0, 4.0]:
            # start from Zr-hard then soft-penalize incremental
            base_zr = apply_zr_hard(bvec, Zg_tr, u_b)
            cand = soft_penalty(base_zr, inc_mask, lam) if inc_mask.any() else base_zr
            mech_results.append({
                **eval_mech(y, base_zr if inc_mask.any() else bvec, cand, folds,
                            f"A2_soft_lam{lam}_inc_on_Zr@{bname}"),
                "base": bname, "lambda": lam, "n_inc_oof": int(inc_mask.sum()),
            })

    log(f"mechanisms evaluated: {len(mech_results)}")

    # Mechanism B: EB continuous feature
    # OOF: for each fold, compute cell log-odds from train_idx with Laplace smoothing
    eb_oof = np.zeros(N_TR)
    eb_te_folds = []
    prior = float(y.mean())
    for k in range(5):
        trm, vam = folds != k, folds == k
        df = pd.DataFrame({"k": keys_tr[trm], "y": y[trm]})
        agg = df.groupby("k")["y"].agg(["sum", "count"])
        # Empirical-Bayes: (sum + a*prior) / (count + a), a=10
        a = 10.0
        rate = (agg["sum"] + a * prior) / (agg["count"] + a)
        # zero-zone posterior: P(zero-fraud cell | data) ≈ exp(-expected frauds) rough
        # use logit(rate) - logit(prior) as continuous contrast
        lo = logit(np.clip(rate.to_numpy(), 1e-6, 1 - 1e-6)) - logit(np.clip(prior, 1e-6, 1 - 1e-6))
        mp = dict(zip(agg.index.to_numpy(), lo))
        eb_oof[vam] = pd.Series(keys_tr[vam]).map(mp).fillna(0.0).to_numpy()
        eb_te_folds.append(pd.Series(keys_te).map(mp).fillna(0.0).to_numpy())
    eb_te = np.mean(np.stack(eb_te_folds, 0), axis=0)
    # hygiene
    eb_disp = float(eb_te.std() / (eb_oof.std() + 1e-12))
    eb_ks = float(ks_2samp(eb_oof, eb_te).statistic)
    eb_nan = int(np.isnan(eb_oof).sum() + np.isnan(eb_te).sum() + np.isinf(eb_oof).sum() + np.isinf(eb_te).sum())
    eb_path_o = PRED / "v41_d27_eb_zone_oof.npy"
    eb_path_t = PRED / "v41_d27_eb_zone_test.npy"
    np.save(eb_path_o, eb_oof)
    np.save(eb_path_t, eb_te)
    np.save(SSD_PRED / "v41_d27_eb_zone_oof.npy", eb_oof)
    np.save(SSD_PRED / "v41_d27_eb_zone_test.npy", eb_te)
    eb_hyg = {
        "oof_path": str(eb_path_o), "test_path": str(eb_path_t),
        "nan_inf": eb_nan, "raw_disp": eb_disp, "KS_raw": eb_ks,
        "pass_disp": 0.92 <= eb_disp <= 1.08, "pass_ks": eb_ks < 0.02,
        "oof_std": float(eb_oof.std()), "te_std": float(eb_te.std()),
    }
    log(f"EB feature disp={eb_disp:.4f} KS={eb_ks:.4f} pass={eb_hyg['pass_disp'] and eb_hyg['pass_ks']}")

    # LOFO orth of EB on bases
    for bname, bvec in bases.items():
        Lb = L(bvec)
        # residual of EB after OLS on L(base) LOFO
        res = np.zeros(N_TR)
        for k in range(5):
            trm, vam = folds != k, folds == k
            X = np.column_stack([np.ones(trm.sum()), Lb[trm]])
            beta, _, _, _ = np.linalg.lstsq(X, eb_oof[trm], rcond=None)
            Xv = np.column_stack([np.ones(vam.sum()), Lb[vam]])
            r = eb_oof[vam] - Xv @ beta
            res[vam] = (r - r.mean()) / (r.std() + 1e-12)
        for sign in (+1, -1):
            for a in [0.02, 0.05, 0.10, 0.20]:
                cand = U(Lb + sign * a * res)
                mech_results.append({
                    **eval_mech(y, bvec, cand, folds, f"B_EB_lofo_s{sign}_a{a}@{bname}"),
                    "base": bname, "sign": sign, "alpha": a,
                })

    # ========== 4. Disagreement + conditional staging ==========
    # For kabartay pub \ Zr (empty) and mined \ Zr
    def zone_rank_stats(mask_te: np.ndarray) -> dict:
        if not mask_te.any():
            return {"n_rows": 0}
        u = u_ab[mask_te]
        return {
            "n_rows": int(mask_te.sum()),
            "min": float(u.min()), "median": float(np.median(u)),
            "p90": float(np.quantile(u, 0.9)), "p99": float(np.quantile(u, 0.99)),
            "max": float(u.max()),
            "n_gt_010": int((u > 0.10).sum()),
            "n_gt_050": int((u > 0.50).sum()),
            "n_gt_080": int((u > 0.80).sum()),
        }

    disagree = {
        "kabartay_pub_minus_Zr": zone_rank_stats(Zd27_inc_te),
        "mined_minus_Zr": zone_rank_stats(Zmine_inc_te),
        "n_min_selected": n_min,
    }

    # OOF gate for staging: marginal mine on Zr must be positive on fullbank or part12
    marg_fb = next(r for r in mech_results if r["tag"].startswith(f"A1_marginal_mine{n_min}_on_Zr@fullbank"))
    marg_p12 = next(r for r in mech_results if r["tag"].startswith(f"A1_marginal_mine{n_min}_on_Zr@part12"))
    kab_marg_fb = next(r for r in mech_results if r["tag"].startswith("A1_marginal_kab_on_Zr@fullbank"))

    oof_positive = (
        Zmine_inc_te.any()
        and marg_fb["pos5"] >= 3
        and marg_fb["mean_e5"] > 0
        and marg_p12["mean_e5"] > -1.0  # not clearly harmful on part12
    )
    kab_inc_empty = not Zd27_inc_te.any()

    staged = {}
    if kab_inc_empty and not oof_positive:
        live_stage = "SKIP (keep for Step 5 own-stack only)"
        live_reason = (
            f"Z_D27_kabartay ≡ Z_goodpjw on test (inc=0). "
            f"Mined N_min={n_min} test_inc={int(Zmine_inc_te.sum())} "
            f"OOF marginal pos5={marg_fb['pos5']} mean_e5={marg_fb['mean_e5']:.3f} — "
            f"{'positive' if oof_positive else 'not clearly positive'}; LIVE_STAGE=SKIP."
        )
    elif oof_positive:
        live_stage = "STAGE"
        live_reason = f"Mined incremental zones OOF-positive; staging hard+soft on Step_AB."
        # Z1 hard
        sc = apply_zr_hard(ab, Zg_te | Zmine_te, u_ab)
        sha, p = write_u_sub(OUT / "Z1_AB_plus_D27_hard_zr.csv", ids, sc)
        staged["Z1_AB_plus_D27_hard_zr.csv"] = {
            "sha": sha, "1mrho_vs_AB_e4": one_mrho(p, ab) * 1e4,
            "n_extra_zones": int(Zmine_inc_te.sum()),
        }
        # Z2 soft at best λ from OOF
        soft_rows = [r for r in mech_results if r["tag"].startswith("A2_soft_lam") and r["base"] == "fullbank"]
        soft_rows = sorted(soft_rows, key=lambda r: (r["pos5"], r["mean_e5"]), reverse=True)
        best_lam = soft_rows[0].get("lambda", 1.0) if soft_rows else 1.0
        # soft on AB: Zr already in AB; only soft-penalize incremental
        sc2 = soft_penalty(ab, Zmine_inc_te, best_lam)
        # also preserve Zr hard
        sc2 = apply_zr_hard(sc2, Zg_te, u_ab)
        sha2, p2 = write_u_sub(OUT / f"Z2_AB_plus_D27_soft_lam{best_lam}.csv".replace(".", "p"), ids, sc2)
        staged[f"Z2_AB_plus_D27_soft_lam{best_lam}.csv".replace(".", "p")] = {
            "sha": sha2, "lambda": best_lam, "1mrho_vs_AB_e4": one_mrho(p2, ab) * 1e4,
        }
    else:
        live_stage = "SKIP (keep for Step 5 own-stack only)"
        live_reason = (
            f"Incremental mined test zones={int(Zmine_inc_te.sum())} but OOF gate failed "
            f"(pos5={marg_fb['pos5']} mean_e5={marg_fb['mean_e5']:.3f})."
        )

    log(f"LIVE_STAGE={live_stage}")

    # ========== write audit ==========
    payload = {
        "NO_SUBMIT": True,
        "lineage": lineage,
        "overlap": overlap,
        "oof_leak": oof_leak,
        "selected_n_min": n_min,
        "disagree": disagree,
        "eb_hygiene": eb_hyg,
        "mechanisms": mech_results,
        "marginal_gates": {
            "kabartay_on_Zr_fullbank": kab_marg_fb,
            "mine_on_Zr_fullbank": marg_fb,
            "mine_on_Zr_part12": marg_p12,
        },
        "LIVE_STAGE": live_stage,
        "LIVE_REASON": live_reason,
        "staged": staged,
        "step5_note": (
            "uid_catboost TE-leaked (OOF=0.98852, disp=0.561, KS=0.095) — EXCLUDE from Step5 nested CV "
            "unless rebuilt leak-free. uid_v2 + P1s_AB_uid_* remain clean."
        ),
    }
    (LOG / "STEP4_AUDIT.json").write_text(json.dumps(payload, indent=2, default=str))

    # top mechanism table
    top = sorted(mech_results, key=lambda r: (r["pos5"], r["mean_e5"]), reverse=True)[:20]

    lines = [
        "# V41 STEP4 AUDIT — kabartay D27 Zero-Fraud Zones vs Step_AB Zr",
        "",
        f"**NO_SUBMIT={NO_SUBMIT}** · Hard stop. Do not run Step 5 without `PROCEED TO STEP 5`.",
        "",
        "## Executive findings",
        "",
        f"- Kabartay §8 hand zones **≡** `Z_goodpjw` on test: `|∩|={overlap['test']['intersection']}`, "
        f"`|D27\\\\Zr|={overlap['test']['D27_minus_goodpjw']}`, identical={overlap['test']['identical']}.",
        f"- Train `Z_goodpjw`: n={overlap['train']['n_Z_goodpjw']}, **frauds={overlap['train']['fraud_in_Z_goodpjw']}** "
        f"(rate 0 vs baseline {overlap['train']['baseline_fraud_rate']:.5f}).",
        f"- Kabartay OOF edge-refit: held-out zone frauds = **{oof_leak['kabartay_edge_refit_sum']['n_val_fraud']}**; "
        f"incremental vs Zr = **{oof_leak['kabartay_edge_refit_sum']['n_inc_vs_Zr']}**.",
        f"- Cell-miner N_min={n_min}: test incremental vs Zr = **{int(Zmine_inc_te.sum())}**; "
        f"OOF inc frauds = {oof_leak['cell_miner'][str(n_min)]['sum_n_inc_fraud']}.",
        f"- **LIVE_STAGE = `{live_stage}`**",
        f"- Reason: {live_reason}",
        f"- EB feature: disp={eb_disp:.4f} KS={eb_ks:.4f} "
        f"({'PASS' if eb_hyg['pass_disp'] and eb_hyg['pass_ks'] else 'FAIL hygiene'}) → `{eb_path_o.name}`.",
        "- Step5 note: **exclude TE-leaked `uid_catboost`** from nested CV.",
        "",
        "## 1. Lineage & overlap",
        "",
        "```json",
        json.dumps({"lineage": lineage, "overlap": overlap}, indent=2),
        "```",
        "",
        "## 2. Strict OOF leakage",
        "",
        "```json",
        json.dumps(oof_leak, indent=2)[:30000],
        "```",
        "",
        "## 3. Mechanisms (top 20 by pos5/meanΔ)",
        "",
        "```json",
        json.dumps(top, indent=2),
        "```",
        "",
        "### Marginal gates",
        "",
        "```json",
        json.dumps(payload["marginal_gates"], indent=2, default=str),
        "```",
        "",
        "## 4. Disagreement vs Step_AB + staging",
        "",
        "```json",
        json.dumps({"disagree": disagree, "LIVE_STAGE": live_stage, "LIVE_REASON": live_reason,
                    "staged": staged, "eb_hygiene": eb_hyg}, indent=2),
        "```",
        "",
        "## 5. STOP",
        "",
        "Step 4 complete. **Awaiting human audit.** Reply `PROCEED TO STEP 5` to continue.",
        "",
    ]
    (LOG / "STEP4_AUDIT.md").write_text("\n".join(lines))

    print("\n=== STRICT NO_SUBMIT=True | V41 STEP4 ===")
    print(f"Zr≡kabartay_pub: {overlap['test']['identical']}")
    print(f"train frauds in Zr: {overlap['train']['fraud_in_Z_goodpjw']}")
    print(f"LIVE_STAGE={live_stage}")
    print(f"mine N_min={n_min} test_inc={int(Zmine_inc_te.sum())} marg_fb pos5={marg_fb['pos5']} mean_e5={marg_fb['mean_e5']:.3f}")
    zr_fb = next(r for r in mech_results if r["tag"] == "A1_Zr_goodpjw@fullbank")
    print(f"A1 Zr on fullbank: pos5={zr_fb['pos5']} mean_e5={zr_fb['mean_e5']:.3f} sf25={zr_fb['sf25_pos']}/{zr_fb['sf25_n']}")
    print(f"WROTE {LOG/'STEP4_AUDIT.md'}")
    print("STOP. Awaiting PROCEED TO STEP 5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
