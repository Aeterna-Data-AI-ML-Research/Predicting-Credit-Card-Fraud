#!/usr/bin/env python3
"""Audit unused self-owned OOF streams vs z_sovereign_v0 Fold-0; fixed 95:5 blends."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "preds"
SOV = PRED / "v47_sovereign"
USED = {
    "B_xgb_llr",
    "P4_L4.2_pure5f",
    "P4_L4.3_pure5f",
    "P4_L4.5_pure5f",
    "P_super_v4p_hc",
    "P_super_v5_harvest",
    "delta1_v4",
    "gpt2_med_4ep_mean",
    "gpt2_med_4ep_shrink",
    "l7_R5",
    "l7_U5_gini10",
    "l7_curv",
    "l7_mean10",
    "zone_honest",
}
BASE_AUC = 0.9512507991341679


def rank_logit(scores: np.ndarray) -> np.ndarray:
    return logit((rankdata(scores) - 0.5) / len(scores))


def main() -> None:
    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy()
    folds = np.load(SOV / "folds_skf42.npy")
    valid = np.flatnonzero(folds == 0)
    y0 = y[valid]
    base = np.load(SOV / "z_sovereign_v0_oof.npy").ravel()[valid]
    base_auc = float(roc_auc_score(y0, base))
    assert abs(base_auc - BASE_AUC) < 1e-12

    def subfold_deltas(candidate: np.ndarray) -> list[float]:
        out = []
        for _, i in StratifiedKFold(5, shuffle=True, random_state=7).split(base, y0):
            out.append(
                (roc_auc_score(y0[i], candidate[i]) - roc_auc_score(y0[i], base[i])) * 1e5
            )
        return out

    cands: list[tuple[str, str, np.ndarray]] = []
    for p in sorted(PRED.glob("P_*_oof.npy")):
        name = p.name.replace("_oof.npy", "")
        if name in USED:
            continue
        try:
            a = np.load(p)
        except Exception:
            continue
        if a.ndim != 1 or a.shape[0] != len(y):
            continue
        cands.append((name, str(p), a.astype(np.float64)))

    rows = []
    best = None
    for name, path, a in cands:
        s = a[valid]
        if not np.isfinite(s).all():
            continue
        solo = float(roc_auc_score(y0, s))
        corr = float(spearmanr(base, s).correlation)
        blend = 0.95 * rank_logit(base) + 0.05 * rank_logit(s)
        auc = float(roc_auc_score(y0, blend))
        d = auc - base_auc
        subs = subfold_deltas(blend)
        pos = int(sum(x > 0 for x in subs))
        verd = "GO" if d >= 8e-5 and pos >= 4 else "NO-GO"
        rec = {
            "name": name,
            "path": path,
            "solo_auc": solo,
            "spearman_vs_base": corr,
            "blend_auc": auc,
            "delta_e5": d * 1e5,
            "pos5": pos,
            "sub_e5": subs,
            "verdict": verd,
        }
        rows.append(rec)
        key = (rec["delta_e5"], rec["pos5"], rec["solo_auc"])
        if best is None or key > (best["delta_e5"], best["pos5"], best["solo_auc"]):
            best = rec

    rows.sort(key=lambda r: (r["delta_e5"], r["pos5"], r["solo_auc"]), reverse=True)
    print(f"Audited {len(rows)} unused OOF streams; base={base_auc:.16f}")
    print("TOP 15 by Δe-5 of fixed 95:5 blend:")
    for r in rows[:15]:
        print(
            f"  {r['name']}: solo={r['solo_auc']:.6f} ρ={r['spearman_vs_base']:.4f} "
            f"Δe-5={r['delta_e5']:+.4f} pos={r['pos5']}/5 {r['verdict']}"
        )

    out = {
        "base_auc": base_auc,
        "n_audited": len(rows),
        "top": rows[:25],
        "best": best,
        "VERIFIED_LIFT": bool(best and best["verdict"] == "GO"),
        "NO_SUBMIT": True,
    }
    out_dir = ROOT / "logs/v47/codex_loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ROUND4_PSUPER_AUDIT.json").write_text(json.dumps(out, indent=2))

    rep = ROOT / "logs/v47/CODEX_DS_REVIEW_NEXT.md"
    lines = [
        "\n## Round4 P_super / unused OOF audit (fixed 95:5 rank-logit vs champion)\n",
        f"Audited {len(rows)} streams. VERIFIED_LIFT={'YES' if out['VERIFIED_LIFT'] else 'NO'}.\n",
    ]
    if best:
        lines.append(
            f"Best: `{best['name']}` solo={best['solo_auc']:.6f} "
            f"Δe-5={best['delta_e5']:+.6f} pos={best['pos5']}/5 {best['verdict']}\n"
        )
    lines.append(
        "\n| stream | solo | ρ | Δe-5 | pos/5 | verdict |\n|---|---:|---:|---:|---:|---|\n"
    )
    for r in rows[:12]:
        lines.append(
            f"| {r['name']} | {r['solo_auc']:.6f} | {r['spearman_vs_base']:.4f} | "
            f"{r['delta_e5']:+.4f} | {r['pos5']}/5 | {r['verdict']} |\n"
        )
    with rep.open("a") as f:
        f.writelines(lines)
    print("BEST", best["name"] if best else None, "VERIFIED_LIFT", out["VERIFIED_LIFT"])


if __name__ == "__main__":
    main()
