#!/usr/bin/env python3
"""V47.2 Step4 — Fold-0 / 5-fold gate + HARD STOP report. NO_SUBMIT · $0."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

NO_SUBMIT = True
assert NO_SUBMIT
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" if (REPO / "data/train.csv").exists() else Path("/tmp/s6e11_v39_ssd/data")
OUT = REPO / "preds/v47_sovereign"
LOGD = REPO / "logs/v47"
POD = REPO / "preds/v47_pod_f0"
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def hygiene(o, t):
    o, t = np.asarray(o, float).ravel(), np.asarray(t, float).ravel()
    return float((t.std()+1e-12)/(o.std()+1e-12)), float(ks_2samp(o, t).statistic)


def nested_lr(X, y, C=0.05):
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=400)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    return oof


def gate_vs(z_base, z_cand, y):
    d = []
    for _, va in StratifiedKFold(5, shuffle=True, random_state=7).split(z_base, y):
        d.append(roc_auc_score(y[va], z_cand[va]) - roc_auc_score(y[va], z_base[va]))
    d = np.asarray(d)
    return {
        "delta_auc": float(roc_auc_score(y, z_cand) - roc_auc_score(y, z_base)),
        "delta_e5": float((roc_auc_score(y, z_cand) - roc_auc_score(y, z_base)) * 1e5),
        "pos": int((d > 0).sum()),
        "sub": d.tolist(),
        "auc_base": float(roc_auc_score(y, z_base)),
        "auc_cand": float(roc_auc_score(y, z_cand)),
    }


def main():
    LOGD.mkdir(parents=True, exist_ok=True)
    y = pd.read_csv(DATA / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = np.load(OUT / "folds_skf42.npy")
    va = folds == 0
    y_va = y[va]

    z0 = np.load(OUT / "z_sovereign_v0_oof.npy").ravel()
    z1_path = OUT / "z_sovereign_v1_oof.npy"
    z1 = np.load(z1_path).ravel() if z1_path.exists() else z0
    z1t = np.load(OUT / "z_sovereign_v1_test.npy").ravel() if (OUT / "z_sovereign_v1_test.npy").exists() else None

    report = {
        "NO_SUBMIT": True,
        "spend_usd": 0.0,
        "pods_created": 0,
        "z_sovereign_v0_auc": float(roc_auc_score(y, z0)),
        "z_sovereign_v1_auc": float(roc_auc_score(y, z1)),
        "z_sovereign_v1_f0": float(roc_auc_score(y_va, z1[va])),
    }
    if z1t is not None:
        d, ks = hygiene(z1, z1t)
        report["z1_hygiene"] = {"disp_raw": d, "KS_raw": ks}

    # Step3A
    report["step3a"] = {"status": "SKIP", "reason": "l7 full 5-fold .pt weights absent (only fold_0/latest/replica)"}
    prov = json.loads((LOGD / "STEP2_PROVENANCE.json").read_text()) if (LOGD / "STEP2_PROVENANCE.json").exists() else {}
    if prov.get("checkpoints", {}).get("l7_has_full_5fold_weights"):
        report["step3a"]["status"] = "WOULD_RUN"
    else:
        report["step3a"]["checkpoints"] = prov.get("checkpoints", {})

    # Step3B HGDT
    hg_f = OUT / "hgdt_v2_f0_field24_llr_oof.npy"
    if not hg_f.exists():
        hg_f = POD / "hgdt_v2_f0_field24_llr_oof.npy"
    skip = OUT / "HGDT_V2_SKIPPED.json"
    if not skip.exists():
        skip = POD / "HGDT_V2_SKIPPED.json"
    thr = OUT / "HGDT_V2_THROUGHPUT.json"
    if not thr.exists():
        thr = POD / "HGDT_V2_THROUGHPUT.json"

    if skip.exists():
        report["step3b"] = {"status": "SKIPPED_ETA", **json.loads(skip.read_text())}
    elif hg_f.exists():
        f24 = np.load(hg_f)
        if f24.shape[0] == len(y):
            f24 = f24[va]
        elif f24.shape[0] != len(y_va):
            raise SystemExit(f"bad hgdt shape {f24.shape}")
        sum_llr = f24.sum(1)
        std_p = OUT / "hgdt_v2_f0_field24_std_oof.npy"
        if not std_p.exists():
            std_p = POD / "hgdt_v2_f0_field24_std_oof.npy"
        X = f24
        if std_p.exists():
            s24 = np.load(std_p)
            if s24.shape[0] == len(y):
                s24 = s24[va]
            X = np.hstack([f24, s24])
        oof_lr = nested_lr(X, y_va)
        g_sum = gate_vs(z1[va], sum_llr, y_va)
        g_lr = gate_vs(z1[va], oof_lr, y_va)
        # cross-modal: P4 + field24
        p4 = []
        for tag in ("P4_L4.2_pure5f", "P4_L4.3_pure5f", "P4_L4.5_pure5f"):
            p = REPO / "preds/matrix_v2" / f"{tag}_oof.npy"
            if p.exists():
                p4.append(np.load(p).ravel()[va, None])
        Xc = np.hstack(p4 + [f24]) if p4 else f24
        try:
            import lightgbm as lgb
            oof_cm = np.zeros(len(y_va))
            for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=42).split(Xc, y_va):
                dtr = lgb.Dataset(Xc[tr_i], label=y_va[tr_i], init_score=0.25 * z1[va][tr_i])
                bst = lgb.train(dict(objective="binary", max_depth=3, learning_rate=0.05,
                                     min_child_samples=200, verbosity=-1), dtr, num_boost_round=150)
                oof_cm[va_i] = bst.predict(Xc[va_i], raw_score=False)
            g_cm = gate_vs(z1[va], oof_cm, y_va)
        except Exception as e:
            g_cm = {"error": str(e)}
        report["step3b"] = {
            "status": "DONE",
            "solo_sum_auc": float(roc_auc_score(y_va, sum_llr)),
            "solo_lr_auc": float(roc_auc_score(y_va, oof_lr)),
            "gate_sum": g_sum,
            "gate_lr": g_lr,
            "gate_cross": g_cm,
            "R2_vs_z1": float(r2_score(z1[va], sum_llr)),
            "throughput": json.loads(thr.read_text()) if thr.exists() else None,
        }
    else:
        report["step3b"] = {
            "status": "PENDING_OR_MISSING",
            "throughput": json.loads(thr.read_text()) if thr.exists() else None,
        }

    # peer meta
    if (LOGD / "STEP2_Z_SOVEREIGN_V1.json").exists():
        report["step2"] = json.loads((LOGD / "STEP2_Z_SOVEREIGN_V1.json").read_text())

    (LOGD / "STEP4_GATE.json").write_text(json.dumps(report, indent=2, default=float))

    md = [
        "# V47.2 HARDENED SOVEREIGN REPORT\n\n",
        "**NO_SUBMIT=True · $0.00 RunPod · 0 submissions · 0 pods created**\n\n",
        "## Step1 — Pod teardown\n",
        "- Removed `u4ky2oq4pohcyt`; `list-pods` empty; **$0.00/hr**\n\n",
        "## Step2 — Provenance + z_sovereign_v1\n",
        f"- z0 AUC={report['z_sovereign_v0_auc']:.6f} → z1 AUC=**{report['z_sovereign_v1_auc']:.6f}** · F0=**{report['z_sovereign_v1_f0']:.6f}**\n",
    ]
    if "step2" in report:
        p = report["step2"].get("peer", {})
        md.append(f"- peer promote=`{p.get('promote')}` pos={p.get('pos5')}/5 sf={p.get('sf25')}/25 hygiene=`{p.get('hygiene')}`\n")
        md.append(f"- streams: `{report['step2'].get('names')}`\n")
    md += [
        "\n## Step3A — l7 field24 inference\n",
        f"- `{report['step3a']}`\n",
        "\n## Step3B — HGDT-LM v2 Fold-0\n",
        f"- `{json.dumps(report['step3b'], indent=2, default=float)}`\n",
        "\n## HARD STOP\n",
        "- **NO_SUBMIT=True** · **$0.00 spent** · **0 Kaggle submissions**\n",
    ]
    (LOGD / "V47_2_HARDENED_SOVEREIGN_REPORT.md").write_text("".join(md))
    log(f"wrote report status3b={report['step3b'].get('status')}")


if __name__ == "__main__":
    main()
