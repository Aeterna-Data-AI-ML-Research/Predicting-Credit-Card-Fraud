#!/usr/bin/env python3
"""V37 $0 local parallel audit — Bayes reducers on l7 + public vault hygiene.

NO_SUBMIT=True. n_jobs<=4.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1].parent if (Path(__file__).parent.name == "scripts") else Path(__file__).resolve().parents[2]
# scripts/v37_local... → parents[1]=scripts, parents[2]=repo — fix:
REPO = Path(__file__).resolve().parents[1]
if not (REPO / "data" / "train.csv").exists():
    REPO = Path(__file__).resolve().parents[2]

OUT = REPO / "logs" / "overnight" / "v37_local_audit"
PRED = REPO / "preds" / "v36_1"
L7 = REPO / "preds" / "l7_s2026_5f"
VAULT = REPO / "external" / "v37_twin_check" / "vaults"
FOLDS = REPO / "external" / "banks" / "goodpjw2008_s6e11-oof-bank" / "folds.csv"
PREV = 0.0167
os.environ.setdefault("NO_SUBMIT", "True")


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / max(len(v), 1)


def L(v):
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6))


def auc(y, p):
    return float(roc_auc_score(y, np.asarray(p, float).ravel()))


def raw_disp(p_oof, p_te, z_oof, z_te, folds):
    # use fold0 for va-like; overall OOF vs test
    d_o = L(p_oof) - L(z_oof)
    d_t = L(p_te) - L(z_te)
    return float(d_t.std() / (d_o.std() + 1e-12))


def reducers_from_ll(ll: np.ndarray) -> dict[str, np.ndarray]:
    """ll (N,K,2) → three readouts as probabilities via unit→sigmoid."""
    ll0, ll1 = ll[:, :, 0], ll[:, :, 1]
    r_mean = (ll1 - ll0).mean(1)
    r_lse = logsumexp(ll1, 1) - logsumexp(ll0, 1)
    r_rank = np.mean([L(ll1[:, k] - ll0[:, k]) for k in range(ll.shape[1])], 0)
    return {
        "R_mean_diff": expit(L(r_mean) * 0 + (rankdata(r_mean) - 0.5) / len(r_mean) * 0 + logit(np.clip(U(r_mean), 1e-6, 1 - 1e-6))),
        "R_deotte_lse": expit(logit(np.clip(U(r_lse), 1e-6, 1 - 1e-6))),
        "R_ranklogit": expit(logit(np.clip(U(r_rank), 1e-6, 1 - 1e-6))),
    }


def score_reducer(name, p_oof, p_te, y, folds, z_fb, z_tw, z_fb_te, z_tw_te):
    solo = auc(y, p_oof)
    # per-fold ΔAUC vs baselines
    d_fb, d_tw = [], []
    for f in range(5):
        m = folds == f
        d_fb.append(auc(y[m], p_oof[m]) - auc(y[m], z_fb[m]))
        d_tw.append(auc(y[m], p_oof[m]) - auc(y[m], z_tw[m]))
    return {
        "name": name,
        "solo_oof": solo,
        "raw_disp_fb": raw_disp(p_oof, p_te, z_fb, z_fb_te, folds),
        "mean_d_auc_fb": float(np.mean(d_fb)),
        "mean_d_auc_tw": float(np.mean(d_tw)),
        "d_auc_fb_folds": [float(x) for x in d_fb],
        "d_auc_tw_folds": [float(x) for x in d_tw],
    }


def audit_l7_bayes():
    y = pd.read_csv(REPO / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS)["fold"].to_numpy(int)
    z_fb = np.load(PRED / "z_fullbank_oof.npy").ravel()[: len(y)]
    z_tw = np.load(PRED / "z_ATH_twin_oof.npy").ravel()[: len(y)]
    z_fb_te = np.load(PRED / "z_fullbank_test.npy").ravel()
    z_tw_te = np.load(PRED / "z_ATH_twin_test.npy").ravel()
    ll_oof = np.load(L7 / "l7_s2026_oof_5fold_raw.npy")  # (N,10,2)
    # test: mean over fold models' LL then reduce, OR reduce per fold then mean proba
    ll_te_f = np.load(L7 / "l7_s2026_test_5fold_raw.npy")  # (5,Nte,10,2)
    red_o = reducers_from_ll(ll_oof)
    # average fold test LLs then reduce
    ll_te = ll_te_f.mean(axis=0)
    red_t = reducers_from_ll(ll_te)
    rows = []
    for k in red_o:
        rows.append(score_reducer(f"l7_{k}", red_o[k], red_t[k], y, folds, z_fb, z_tw, z_fb_te, z_tw_te))
    return rows


def kaggle_dl(slug: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / "Library/Python/3.14/bin") + ":" + env.get("PATH", "")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (r.stdout or "") + (r.stderr or "")
    lines = [ln for ln in out.splitlines() if not ln.startswith("Warning:")]
    print("\n".join(lines[-8:]), flush=True)
    return r.returncode == 0


def load_pred_candidates(root: Path):
    """Find oof/test npy pairs; return list of (name, oof, test)."""
    found = []
    for p in root.rglob("*.npy"):
        if p.name.startswith("._"):
            continue
        n = p.name.lower()
        if "oof" in n and "test" not in n:
            # pair
            te = None
            for cand in [
                p.with_name(p.name.replace("oof", "test")),
                p.with_name(p.name.replace("OOF", "test")),
                p.parent / p.name.replace("oof", "test"),
            ]:
                if cand.exists():
                    te = cand
                    break
            try:
                o = np.load(p)
                if o.ndim > 1:
                    o = o.reshape(o.shape[0], -1)
                    if o.shape[1] > 1:
                        # multi-model — defer to family avg
                        found.append((p.stem, o, np.load(te) if te else None, "matrix"))
                        continue
                    o = o.ravel()
                t = np.load(te).ravel() if te else None
                found.append((p.stem, o, t, "vector"))
            except Exception:
                continue
    return found


def family_average(mats: list[np.ndarray]) -> np.ndarray:
    stacked = np.column_stack([m if m.ndim == 1 else m.mean(1) for m in mats])
    # average in rank-logit space
    Lcols = np.column_stack([L(stacked[:, j]) for j in range(stacked.shape[1])])
    return expit(Lcols.mean(1))


def hygiene_screen(name, p_oof, p_te, y, folds, z_fb, z_tw, z_fb_te):
    if p_oof is None or len(p_oof) != len(y):
        return {"name": name, "reject": "shape", "solo_oof": None}
    solo = auc(y, p_oof)
    if p_te is None or len(p_te) != len(z_fb_te):
        disp = None
    else:
        disp = raw_disp(p_oof, p_te, z_fb, z_fb_te, folds)
    ok = solo >= 0.9480 and disp is not None and 0.94 <= disp <= 1.06
    return {
        "name": name,
        "solo_oof": solo,
        "raw_disp": disp,
        "pass_hygiene": bool(ok),
        "d_auc_fb": float(auc(y, p_oof) - auc(y, z_fb)),
        "d_auc_tw": float(auc(y, p_oof) - auc(y, z_tw)),
    }


def audit_vault(slug: str, family_rules: dict):
    dest = VAULT / slug.replace("/", "_")
    ok = kaggle_dl(slug, dest)
    y = pd.read_csv(REPO / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = pd.read_csv(FOLDS)["fold"].to_numpy(int)
    z_fb = np.load(PRED / "z_fullbank_oof.npy").ravel()[: len(y)]
    z_tw = np.load(PRED / "z_ATH_twin_oof.npy").ravel()[: len(y)]
    z_fb_te = np.load(PRED / "z_fullbank_test.npy").ravel()
    cands = load_pred_candidates(dest)
    rows = []
    # family collapse
    families: dict[str, list] = {k: [] for k in family_rules}
    singles = []
    for stem, o, t, kind in cands:
        low = stem.lower()
        matched = None
        for fam, keys in family_rules.items():
            if any(k.lower() in low for k in keys):
                matched = fam
                break
        if matched:
            families[matched].append((stem, o, t))
        else:
            singles.append((stem, o, t, kind))

    for fam, items in families.items():
        if not items:
            rows.append({"name": f"{slug}:{fam}", "reject": "empty_family"})
            continue
        # a priori family average ONLY — no cherry-pick
        oofs = []
        tes = []
        for stem, o, t in items:
            if isinstance(o, np.ndarray) and o.ndim == 2:
                for j in range(o.shape[1]):
                    oofs.append(o[:, j])
                    if t is not None and isinstance(t, np.ndarray):
                        tes.append(t[:, j] if t.ndim == 2 else t)
            else:
                oofs.append(np.asarray(o).ravel())
                if t is not None:
                    tes.append(np.asarray(t).ravel())
        p_o = family_average(oofs)
        p_t = family_average(tes) if tes else None
        rows.append(hygiene_screen(f"{slug}:{fam}_avg_n{len(oofs)}", p_o, p_t, y, folds, z_fb, z_tw, z_fb_te))
    rows.append({"slug": slug, "n_files": len(cands), "download_ok": ok, "n_singles_ignored": len(singles)})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[audit] NO_SUBMIT={os.environ.get('NO_SUBMIT')} out={OUT}", flush=True)
    results = {"l7_bayes": None, "vaults": []}

    def job_l7():
        return ("l7", audit_l7_bayes())

    def job_aman():
        # Tri-TabPFN family
        return (
            "aman",
            audit_vault(
                "amanatar/s6e11-sovereign-095423-simplex-and-oof-vault",
                {"V_TriTabPFN": ["tabpfn", "tri", "pfn", "sovereign"]},
            ),
        )

    def job_busy():
        return (
            "busy",
            audit_vault(
                "busyaprime/s6e11-124-voices-the-top-rebuilt-lb-0-95465",
                {
                    "V_TabM": ["tabm", "tab_m"],
                    "V_EBM": ["ebm", "interpretml"],
                    "V_BPE_GBDT": ["bpe", "xgb", "lgb", "catboost", "gbdt"],
                },
            ),
        )

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(job_l7), ex.submit(job_aman), ex.submit(job_busy)]
        for fut in as_completed(futs):
            tag, rows = fut.result()
            if tag == "l7":
                results["l7_bayes"] = rows
            else:
                results["vaults"].extend(rows)
            print(f"[audit] finished {tag}", flush=True)

    (OUT / "V37_LOCAL_AUDIT.json").write_text(json.dumps(results, indent=2))
    # markdown summary
    lines = ["# V37 Local Parallel Audit", "", "## L7 Bayes reducers", ""]
    for r in results["l7_bayes"] or []:
        lines.append(
            f"- **{r['name']}**: solo={r['solo_oof']:.5f} disp={r['raw_disp_fb']:.4f} "
            f"dAUC_fb={r['mean_d_auc_fb']:+.2e} dAUC_tw={r['mean_d_auc_tw']:+.2e}"
        )
    lines += ["", "## Vault family averages (hygiene solo≥0.9480 & disp∈[0.94,1.06])", ""]
    for r in results["vaults"]:
        if "solo_oof" in r and r.get("solo_oof") is not None:
            lines.append(
                f"- **{r['name']}**: solo={r['solo_oof']:.5f} disp={r.get('raw_disp')} "
                f"pass={r.get('pass_hygiene')} d_fb={r.get('d_auc_fb')} d_tw={r.get('d_auc_tw')}"
            )
        else:
            lines.append(f"- {r}")
    (OUT / "V37_LOCAL_AUDIT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"[audit] wrote {OUT/'V37_LOCAL_AUDIT.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
