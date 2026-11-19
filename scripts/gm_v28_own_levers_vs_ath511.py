#!/usr/bin/env python3
"""V28 — own levers vs ATH 0.95511 (NOT public-board consume).

Levers:
  L1  Structural zero-fraud zone soft-floor mined from train (Z1–Z4 style).
  L2  LOFO HistGradientBoosting residual, GS-orth to local control, small-w.
  L3  Weni BPE/TabM/RealMLP diversity channels as small-w into ATH (gated).

Gates: nested 5-fold H2H vs gb_bag5 and vs ATH bank-surrogate.
Stages CSVs under submissions/sub_v28_*.csv. Does not submit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
WENI = Path("/tmp/s6e11_v27/weni_oof")
ATH = REPO / "submissions/sub_v27_ATH_PUB511.csv"
SUB = REPO / "submissions"
OUT = Path("/tmp/s6e11_v28")
OUT.mkdir(parents=True, exist_ok=True)
LOG = REPO / "logs/overnight/GM_V28_OWN_LEVERS_REPORT.md"

CLEAN_FLAGS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
    # two more binary-ish cleans often used with "8 clean flags"
    # prior_disputes==0 and cvv_retry_count==0 treated as clean
]


def log(msg: str) -> None:
    print(msg, flush=True)


def rf(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v: np.ndarray) -> np.ndarray:
    return logit(np.clip(rf(v), 1e-6, 1.0 - 1e-6))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def omr(a: np.ndarray, b: np.ndarray) -> float:
    return float((1.0 - spearmanr(a, b).correlation) * 1e4)


def h2h(cand: np.ndarray, base: np.ndarray, y: np.ndarray, folds: list) -> dict:
    deltas = []
    for _, te in folds:
        deltas.append((auc(y[te], cand[te]) - auc(y[te], base[te])) * 1e4)
    d = np.asarray(deltas, float)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    t = mean / se if se > 1e-12 else 0.0
    return {"mean": mean, "pos": int((d > 0).sum()), "t": t, "d": d.tolist()}


def folds_from_ids(fold_id: np.ndarray) -> list:
    return [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]


def all_clean(df: pd.DataFrame) -> np.ndarray:
    m = np.ones(len(df), dtype=bool)
    for c in [
        "is_foreign_transaction",
        "is_new_merchant",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]:
        m &= ~df[c].astype(bool).to_numpy()
    m &= df["prior_disputes"].to_numpy() == 0
    m &= df["cvv_retry_count"].to_numpy() == 0
    return m


def mine_zones(tr: pd.DataFrame, te: pd.DataFrame) -> dict[str, np.ndarray]:
    """Re-derive structural zero-fraud zones from train (support + 0 frauds)."""
    y = tr["is_fraud"].to_numpy()
    clean_tr = all_clean(tr)
    clean_te = all_clean(te)

    # Z1: low merchant risk
    z1_tr = tr["merchant_risk_score"].to_numpy() < 13.1
    z1_te = te["merchant_risk_score"].to_numpy() < 13.1
    # Z2: low velocity + moderate risk
    z2_tr = (tr["velocity_score"].to_numpy() <= 4.1) & (tr["merchant_risk_score"].to_numpy() <= 20.0)
    z2_te = (te["velocity_score"].to_numpy() <= 4.1) & (te["merchant_risk_score"].to_numpy() <= 20.0)
    # Z3 / Z4: clean flags + velocity/risk
    z3_tr = clean_tr & (tr["velocity_score"].to_numpy() <= 4.1) & (tr["merchant_risk_score"].to_numpy() <= 33.8)
    z3_te = clean_te & (te["velocity_score"].to_numpy() <= 4.1) & (te["merchant_risk_score"].to_numpy() <= 33.8)
    z4_tr = clean_tr & (tr["velocity_score"].to_numpy() <= 6.0) & (tr["merchant_risk_score"].to_numpy() <= 25.0)
    z4_te = clean_te & (te["velocity_score"].to_numpy() <= 6.0) & (te["merchant_risk_score"].to_numpy() <= 25.0)

    zones = {
        "Z1": (z1_tr, z1_te),
        "Z2": (z2_tr, z2_te),
        "Z3": (z3_tr, z3_te),
        "Z4": (z4_tr, z4_te),
    }
    for name, (mtr, mte) in zones.items():
        n, fr = int(mtr.sum()), float(y[mtr].sum())
        log(f"zone {name}: train_rows={n} frauds={fr:.0f} test_rows={int(mte.sum())}")
        assert fr == 0.0, f"{name} not zero-fraud on train"
    union_tr = z1_tr | z2_tr | z3_tr | z4_tr
    union_te = z1_te | z2_te | z3_te | z4_te
    log(f"zone UNION: train={union_tr.sum()} frauds={y[union_tr].sum():.0f} test={union_te.sum()}")
    return {
        "Z1": z1_te,
        "Z12": z1_te | z2_te,
        "Z1234": union_te,
        "Z1_tr": z1_tr,
        "Z12_tr": z1_tr | z2_tr,
        "Z1234_tr": union_tr,
    }


def soft_floor(scores: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """Pull masked rows toward the bottom in rank-logit space."""
    s = L(scores)
    # move masked rows down by a fraction of the score range
    span = float(np.percentile(s, 95) - np.percentile(s, 5))
    s2 = s.copy()
    s2[mask] = s2[mask] - strength * span
    return rf(s2)


def encode_xy(tr: pd.DataFrame, te: pd.DataFrame):
    y = tr["is_fraud"].to_numpy(float)
    drop = {"id", "is_fraud"}
    cols = [c for c in tr.columns if c not in drop]
    cat, num = [], []
    for c in cols:
        if pd.api.types.is_bool_dtype(tr[c]) or tr[c].dtype == object or not pd.api.types.is_numeric_dtype(tr[c]):
            cat.append(c)
        else:
            num.append(c)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xtr_num = tr[num].to_numpy(dtype=np.float64)
    Xte_num = te[num].to_numpy(dtype=np.float64)
    Xtr_cat = enc.fit_transform(tr[cat].astype(str)) if cat else np.zeros((len(tr), 0))
    Xte_cat = enc.transform(te[cat].astype(str)) if cat else np.zeros((len(te), 0))
    Xtr = np.column_stack([Xtr_num, Xtr_cat])
    Xte = np.column_stack([Xte_num, Xte_cat])
    return Xtr, Xte, y


def main() -> None:
    t0 = time.time()
    log("=== V28 own levers vs ATH 0.95511 ===")
    tr = pd.read_csv(REPO / "data/train.csv")
    te = pd.read_csv(REPO / "data/test.csv")
    ath = pd.read_csv(ATH).set_index("id").loc[te["id"], "is_fraud"].to_numpy(float)
    ids = te["id"].to_numpy()
    fold_id = pd.read_csv(BANK / "folds.csv")["fold"].to_numpy()
    folds = folds_from_ids(fold_id)
    y = tr["is_fraud"].to_numpy(float)
    N_TR, N_TE = len(tr), len(te)
    assert len(fold_id) == N_TR

    gb = np.load(BANK / "oof_gb_bag5.npy").ravel()[:N_TR]
    gb_t = np.load(BANK / "test_gb_bag5.npy").ravel()[:N_TE]
    log(f"gb_bag5 OOF AUC={auc(y, gb):.6f}")

    # ATH bank surrogate (same recipe as V24/V25)
    members = [
        "gb_bag5",
        "gb_bag10",
        "lgb_diglead_l3",
        "realmlp_v1",
        "glm_dl",
        "ebm_all_bag",
        "gbn_bag5_w512",
    ]
    Bo, Bt = [], []
    for name in members:
        op, tp = BANK / f"oof_{name}.npy", BANK / f"test_{name}.npy"
        if op.exists() and tp.exists():
            Bo.append(unit(L(np.load(op).ravel()[:N_TR])))
            Bt.append(unit(L(np.load(tp).ravel()[:N_TE])))
    Bo_m, Bt_m = np.column_stack(Bo), np.column_stack(Bt)
    lr = Ridge(alpha=10.0).fit(Bt_m, L(ath))
    z_ath = Bo_m @ lr.coef_ + lr.intercept_
    log(f"ATH bank surrogate AUC={auc(y, z_ath):.6f}")

    zones = mine_zones(tr, te)
    report = {"ath": str(ATH), "levers": {}}

    # -------------------- L1 zone soft-floor --------------------
    log("--- L1 zone soft-floor ---")
    l1 = {}
    for zname, strength in [
        ("Z1", 0.15),
        ("Z1", 0.30),
        ("Z12", 0.15),
        ("Z12", 0.25),
        ("Z1234", 0.10),
        ("Z1234", 0.20),
        ("Z1234", 0.35),
    ]:
        m_te = zones[zname]
        m_tr = zones[f"{zname}_tr"]
        # gate on local control: soft-floor gb OOF in-zone
        base_o = gb.copy()
        cand_o = soft_floor(base_o, m_tr, strength)
        # also vs ATH surrogate
        cand_z = soft_floor(rf(z_ath), m_tr, strength)
        # compare in score space via soft_floor of probability-like rf(z_ath)
        g_gb = h2h(cand_o, base_o, y, folds)
        g_ath = h2h(L(cand_z), z_ath, y, folds)
        tag = f"zone_{zname}_s{strength:.2f}".replace(".", "p")
        l1[tag] = {
            "zone": zname,
            "strength": strength,
            "test_hit": int(m_te.sum()),
            "gb": g_gb,
            "ath_surr": g_ath,
            "rho_ath": None,
        }
        log(
            f"L1 {tag}: gb {g_gb['pos']}/5 m={g_gb['mean']:+.2f} t={g_gb['t']:.2f} | "
            f"athS {g_ath['pos']}/5 m={g_ath['mean']:+.2f} t={g_ath['t']:.2f}"
        )
        # stage test file on real ATH
        pred = soft_floor(ath, m_te, strength)
        l1[tag]["rho_ath"] = omr(pred, ath)
        path = SUB / f"sub_v28_L1_{tag}.csv"
        pd.DataFrame({"id": ids, "is_fraud": pred}).to_csv(path, index=False)

    report["levers"]["L1"] = l1

    # -------------------- L2 LOFO HGB residual --------------------
    log("--- L2 LOFO HGB residual ---")
    Xtr, Xte, y = encode_xy(tr, te)
    oof = np.zeros(N_TR)
    te_parts = []
    for tr_idx, te_idx in folds:
        clf = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=80,
            l2_regularization=1.0,
            random_state=2026,
        )
        clf.fit(Xtr[tr_idx], y[tr_idx])
        oof[te_idx] = clf.predict_proba(Xtr[te_idx])[:, 1]
        te_parts.append(clf.predict_proba(Xte)[:, 1])
    te_pred = np.mean(te_parts, axis=0)
    log(f"L2 HGB OOF AUC={auc(y, oof):.6f}")

    # orth to ATH surrogate / gb
    d = unit(L(oof))
    d = unit(d - np.dot(d, unit(z_ath)) * unit(z_ath))
    te0 = folds[0][1]
    sgn = +1 if auc(y[te0], z_ath[te0] + 0.05 * d[te0]) >= auc(y[te0], z_ath[te0] - 0.05 * d[te0]) else -1
    g_ath = h2h(z_ath + 0.05 * sgn * d, z_ath, y, folds)
    # gb + small w rank blend
    g_gb = h2h(rf(0.92 * L(gb) + 0.08 * L(oof)), gb, y, folds)
    log(f"L2 GATE ath-orth {g_ath['pos']}/5 m={g_ath['mean']:+.2f} | gb+8% {g_gb['pos']}/5 m={g_gb['mean']:+.2f}")

    l2 = {"solo_auc": auc(y, oof), "ath_orth": g_ath, "gb8": g_gb, "sgn": sgn}
    for w in (0.03, 0.05, 0.08, 0.12):
        pred = rf((1 - w) * L(ath) + w * L(te_pred))
        tag = f"ATH_x_hgb_{w:.2f}".replace(".", "p")
        path = SUB / f"sub_v28_L2_{tag}.csv"
        pd.DataFrame({"id": ids, "is_fraud": pred}).to_csv(path, index=False)
        l2[tag] = {"rho_ath": omr(pred, ath), "w": w}
        log(f"L2 STAGED {path.name} ρ(ATH)={l2[tag]['rho_ath']:.3f}e-4")
    # orth inject on test via ridge-mapped direction approx: use raw small-w only (safer)
    report["levers"]["L2"] = l2
    np.savez_compressed(OUT / "v28_l2_hgb.npz", oof=oof, test=te_pred)

    # -------------------- L3 Weni diversity --------------------
    log("--- L3 Weni diversity channels ---")
    l3 = {}
    weni_files = sorted(WENI.glob("oof_weni_*.npy"))
    for op in weni_files:
        key = op.stem.replace("oof_", "")
        tp = WENI / f"test_{key}.npy"
        if not tp.exists():
            continue
        oo = np.load(op).ravel()[:N_TR]
        tt = np.load(tp).ravel()[:N_TE]
        solo = auc(y, oo)
        d = unit(L(oo))
        d = unit(d - np.dot(d, unit(z_ath)) * unit(z_ath))
        sgn = +1 if auc(y[te0], z_ath[te0] + 0.05 * d[te0]) >= auc(y[te0], z_ath[te0] - 0.05 * d[te0]) else -1
        g_ath = h2h(z_ath + 0.05 * sgn * d, z_ath, y, folds)
        g_gb = h2h(rf(0.92 * L(gb) + 0.08 * L(oo)), gb, y, folds)
        l3[key] = {
            "solo": solo,
            "ath_orth": g_ath,
            "gb8": g_gb,
            "sgn": int(sgn),
            "rho_raw": omr(tt, ath),
        }
        log(
            f"L3 {key}: solo={solo:.5f} ath-orth {g_ath['pos']}/5 m={g_ath['mean']:+.2f} | "
            f"gb+8% {g_gb['pos']}/5 m={g_gb['mean']:+.2f} ρ={l3[key]['rho_raw']:.1f}e-4"
        )
        if g_gb["pos"] >= 4 and g_gb["mean"] > 0:
            for w in (0.05, 0.08):
                pred = rf((1 - w) * L(ath) + w * L(tt))
                tag = f"ATH_x_{key}_{w:.2f}".replace(".", "p")
                path = SUB / f"sub_v28_L3_{tag}.csv"
                pd.DataFrame({"id": ids, "is_fraud": pred}).to_csv(path, index=False)
                l3[key][tag] = omr(pred, ath)
                log(f"L3 STAGED {path.name} ρ={l3[key][tag]:.3f}e-4")

    # rank-mean of gated weni members
    gated = [k for k, g in l3.items() if g["gb8"]["pos"] >= 4 and g["gb8"]["mean"] > 0]
    if gated:
        oo_rm = rf(np.mean([L(np.load(WENI / f"oof_{k}.npy").ravel()[:N_TR]) for k in gated], axis=0))
        tt_rm = rf(np.mean([L(np.load(WENI / f"test_{k}.npy").ravel()[:N_TE]) for k in gated], axis=0))
        g_gb = h2h(rf(0.92 * L(gb) + 0.08 * L(oo_rm)), gb, y, folds)
        log(f"L3 rankmean({len(gated)}): gb+8% {g_gb['pos']}/5 m={g_gb['mean']:+.2f}")
        for w in (0.05, 0.08):
            pred = rf((1 - w) * L(ath) + w * L(tt_rm))
            path = SUB / f"sub_v28_L3_ATH_x_weni_rankmean_{w:.2f}.csv".replace(".", "p", 1)
            # fix name
            path = SUB / f"sub_v28_L3_ATH_x_weniRM_{str(w).replace('.','p')}.csv"
            pd.DataFrame({"id": ids, "is_fraud": pred}).to_csv(path, index=False)
            log(f"L3 STAGED {path.name} ρ={omr(pred, ath):.3f}e-4")
        l3["weni_rankmean"] = {"members": gated, "gb8": g_gb}

    report["levers"]["L3"] = l3

    # -------------------- pick queue for tomorrow --------------------
    picks = []
    # L1: require gb gate ≥4/5 and mean>0
    for tag, g in l1.items():
        if g["gb"]["pos"] >= 4 and g["gb"]["mean"] > 0:
            picks.append(("L1", tag, g["gb"]["mean"], g["rho_ath"], SUB / f"sub_v28_L1_{tag}.csv"))
    # L2: prefer if gb8 ≥4/5
    if l2["gb8"]["pos"] >= 4 and l2["gb8"]["mean"] > 0:
        picks.append(("L2", "ATH_x_hgb_0p05", l2["gb8"]["mean"], l2["ATH_x_hgb_0p05"]["rho_ath"], SUB / "sub_v28_L2_ATH_x_hgb_0p05.csv"))
        picks.append(("L2", "ATH_x_hgb_0p08", l2["gb8"]["mean"], l2["ATH_x_hgb_0p08"]["rho_ath"], SUB / "sub_v28_L2_ATH_x_hgb_0p08.csv"))
    # L3 gated
    for key, g in l3.items():
        if key == "weni_rankmean":
            continue
        if g.get("gb8", {}).get("pos", 0) >= 4 and g["gb8"]["mean"] > 0:
            for wtag in list(g):
                if wtag.startswith("ATH_x_"):
                    picks.append(("L3", wtag, g["gb8"]["mean"], g[wtag], SUB / f"sub_v28_L3_{wtag}.csv"))

    picks.sort(key=lambda t: (-t[2], t[3] if t[3] is not None else 99))
    report["tomorrow_queue"] = [
        {"family": a, "tag": b, "gate_mean_e4": c, "rho_ath_e4": d, "path": str(e)} for a, b, c, d, e in picks[:10]
    ]
    log("=== tomorrow queue (top) ===")
    for row in report["tomorrow_queue"]:
        log(f"  {row['family']} {row['tag']} gateΔ={row['gate_mean_e4']:+.2f}e-4 ρATH={row['rho_ath_e4']}")

    (OUT / "v28_meta.json").write_text(json.dumps(report, indent=2, default=str))
    # markdown report
    lines = [
        "# V28 Own Levers vs ATH 0.95511",
        "",
        "Goal: find **non-board-consume** levers (structural zones, own HGB residual, Weni diversity) gated nested vs ATH surrogate / gb_bag5.",
        "",
        f"ATH lock: `{ATH.name}` (public 0.95511).",
        "",
        "## L1 Structural zone soft-floor",
        "",
        "| tag | gb gate | athS gate | ρATH e-4 |",
        "|-----|---------|-----------|---------:|",
    ]
    for tag, g in l1.items():
        lines.append(
            f"| `{tag}` | {g['gb']['pos']}/5 {g['gb']['mean']:+.2f} | "
            f"{g['ath_surr']['pos']}/5 {g['ath_surr']['mean']:+.2f} | {g['rho_ath']:.3f} |"
        )
    lines += [
        "",
        "## L2 LOFO HGB residual",
        "",
        f"- solo AUC={l2['solo_auc']:.5f}",
        f"- ath-orth {l2['ath_orth']['pos']}/5 mean={l2['ath_orth']['mean']:+.2f}",
        f"- gb+8% {l2['gb8']['pos']}/5 mean={l2['gb8']['mean']:+.2f}",
        "",
        "## L3 Weni diversity",
        "",
        "| channel | solo | ath-orth | gb+8% | ρ raw |",
        "|---------|-----:|----------|-------|------:|",
    ]
    for key, g in l3.items():
        if key == "weni_rankmean":
            continue
        lines.append(
            f"| `{key}` | {g['solo']:.5f} | {g['ath_orth']['pos']}/5 {g['ath_orth']['mean']:+.2f} | "
            f"{g['gb8']['pos']}/5 {g['gb8']['mean']:+.2f} | {g['rho_raw']:.1f} |"
        )
    lines += ["", "## Tomorrow submit queue", ""]
    for i, row in enumerate(report["tomorrow_queue"], 1):
        lines.append(
            f"{i}. **{row['family']}** `{row['tag']}` gateΔ={row['gate_mean_e4']:+.2f}e-4 "
            f"ρATH={row['rho_ath_e4']} → `{Path(row['path']).name}`"
        )
    lines += [
        "",
        "## Law",
        "",
        "Own levers must clear nested gb/ATH-surrogate gates **and** stay near ATH (ρ ≲ a few e-4) "
        "before burning public slots. Do not replace the board ATH with a solo native.",
        "",
        f"Elapsed {time.time()-t0:.1f}s.",
    ]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines) + "\n")
    log(f"WROTE {LOG}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
