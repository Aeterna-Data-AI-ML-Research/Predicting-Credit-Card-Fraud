#!/usr/bin/env python3
"""V24.3 — Native KNN & Neural Net (MLP) research to lift ATH 0.95440.

Expert DS protocol:
  • Same FE + folds as V24.2 natives (reuse builder)
  • StandardScaler (+ PCA for KNN metric geometry)
  • KNN: distance-weighted on PCA space with stratified prototype index (tractable @700k)
  • MLP: sklearn MLPClassifier with early stopping (no torch required)
  • Nested H2H vs bank-only ATH surrogate; cross-family blends into ATH
  • Stage CSVs; submit if quota allows

Local Mac CPU.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
DATA, SUB = REPO / "data", REPO / "submissions"
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
OUT = Path("/tmp/s6e11_v16/out")
LOG = REPO / "logs/overnight"
ATH = SUB / "sub_v24_SRI440_pub.csv"
PY = "/tmp/s6e11_v23_venv/bin/python"
KAG = [PY, "-m", "kaggle"]
REPORT = LOG / "GM_V24_3_NATIVE_KNN_MLP_REPORT.md"
N_TR, N_TE = 699960, 299984
T0 = time.time()
LINES: list[str] = []


def log(msg: str) -> None:
    s = f"[{time.time()-T0:7.1f}s] {msg}"
    print(s, flush=True)
    LINES.append(s)


def L(v):
    v = np.asarray(v, float).ravel()
    return logit(np.clip((rankdata(v, method="average") - 0.5) / len(v), 1e-6, 1 - 1e-6))


def rf(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def omr(a, b):
    return (1.0 - float(spearmanr(a, b).correlation)) * 1e4


def auc(y, p):
    return float(roc_auc_score(y, p))


def h2h(a, b, y, folds):
    ds = np.array([(auc(y[te], a[te]) - auc(y[te], b[te])) * 1e5 for _, te in folds])
    return dict(
        mean=float(ds.mean()),
        min=float(ds.min()),
        pos=int((ds > 0).sum()),
        t=float(ds.mean() / (ds.std(ddof=1) + 1e-12) * np.sqrt(len(ds))),
    )


def rblend(parts, w):
    w = np.asarray(w, float)
    w = w / w.sum()
    return rf(sum(wi * L(p) for wi, p in zip(w, parts)))


def stratified_prototypes(X, y, n_proto, seed):
    """Keep all positives up to cap; fill remainder with stratified neg sample."""
    pos = np.where(y > 0.5)[0]
    neg = np.where(y <= 0.5)[0]
    rng = np.random.default_rng(seed)
    n_pos = min(len(pos), max(2000, n_proto // 8))
    pos_s = rng.choice(pos, size=n_pos, replace=False)
    n_neg = min(len(neg), n_proto - len(pos_s))
    neg_s = rng.choice(neg, size=n_neg, replace=False)
    idx = np.concatenate([pos_s, neg_s])
    rng.shuffle(idx)
    return idx


def submit_and_poll(path: Path, msg: str):
    try:
        r = subprocess.run(
            KAG + ["competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg],
            capture_output=True,
            text=True,
            timeout=180,
        )
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        log(r.stdout.strip()[:350])
        log(r.stderr.strip()[:250])
        if "successfully submitted" not in out.lower():
            return False, None, "quota_or_error"
        for i in range(40):
            time.sleep(18 if i else 8)
            p = subprocess.run(
                KAG + ["competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
                capture_output=True,
                text=True,
                timeout=90,
            )
            for ln in p.stdout.splitlines()[1:15]:
                if path.name not in ln:
                    continue
                log(f"poll {ln[:220]}")
                if "COMPLETE" in ln.upper():
                    for tok in reversed([c.strip() for c in ln.split(",")]):
                        try:
                            return True, float(tok), "ok"
                        except ValueError:
                            continue
        return True, None, "timeout"
    except Exception as e:
        log(f"submit err {e}")
        traceback.print_exc()
        return False, None, "exception"


def load_fe(mod):
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["is_fraud"].to_numpy(float)
    ids = test["id"].to_numpy()
    fold_id = pd.read_csv(BANK / "folds.csv")["fold"].to_numpy()
    folds = [(np.where(fold_id != i)[0], np.where(fold_id == i)[0]) for i in range(5)]
    tr_f = mod.build_base_frame(train)
    te_f = mod.build_base_frame(test)
    for c in mod.CATS:
        oof_e, te_e = mod.lofo_target_encode(tr_f[c], y, folds, te_f[c], prior=30.0)
        tr_f[f"te_{c}"] = oof_e
        te_f[f"te_{c}"] = te_e
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr_f[[f"ord_{c}" for c in mod.CATS]] = ord_enc.fit_transform(tr_f[mod.CATS])
    te_f[[f"ord_{c}" for c in mod.CATS]] = ord_enc.transform(te_f[mod.CATS])
    cols = [c for c in tr_f.columns if c not in mod.CATS]
    Xtr = np.nan_to_num(tr_f[cols].to_numpy(np.float64))
    Xte = np.nan_to_num(te_f[cols].to_numpy(np.float64))
    return Xtr, Xte, y, ids, folds, cols


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    log("=== V24.3 Native KNN + MLP (expert DS) ===")

    spec = importlib.util.spec_from_file_location("v242", REPO / "scripts" / "gm_v24_2_native_rf_gbm.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    log("Building feature matrix…")
    Xtr, Xte, y, ids, folds, cols = load_fe(mod)
    log(f"X={Xtr.shape} fraud={y.mean():.5f}")

    # -------------------- KNN family --------------------
    # Hypothesis: local geometry in scaled+PCA space carries residual fraud signal.
    # Design: fit scaler+PCA on train-fold; KNN on stratified prototypes (~120k) for speed/quality.
    # Faster expert config after canary: 120k/PCA32/k201 was ~10min/fold @~0.91 AUC.
    # Use 50k prototypes, PCA-24, k∈{101,201} only (still distance-weighted).
    knn_ks = (101, 201)
    oof_knn = {k: np.zeros(N_TR) for k in knn_ks}
    te_knn = {k: np.zeros(N_TE) for k in knn_ks}
    pca_var = []

    log("Training KNN (scaled → PCA-24 → 50k prototypes → distance-weighted)…")
    for i, (tr, te) in enumerate(folds):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xtr[tr])
        pca = PCA(n_components=24, random_state=42 + i)
        Xp = pca.fit_transform(Xs)
        pca_var.append(float(pca.explained_variance_ratio_.sum()))
        proto = stratified_prototypes(Xp, y[tr], n_proto=50_000, seed=100 + i)
        Xproto, yproto = Xp[proto], y[tr][proto]
        Xte_p = pca.transform(scaler.transform(Xtr[te]))
        Xtest_p = pca.transform(scaler.transform(Xte))
        knn = KNeighborsClassifier(
            n_neighbors=max(knn_ks),
            weights="distance",
            algorithm="kd_tree",
            leaf_size=40,
            n_jobs=-1,
            metric="euclidean",
        )
        knn.fit(Xproto, yproto)
        dist, idx = knn.kneighbors(Xte_p, n_neighbors=max(knn_ks), return_distance=True)
        dist_t, idx_t = knn.kneighbors(Xtest_p, n_neighbors=max(knn_ks), return_distance=True)
        labels = yproto

        def proba_from(dist, idx, k):
            d = dist[:, :k] + 1e-12
            lab = labels[idx[:, :k]]
            w = 1.0 / d
            return (w * lab).sum(1) / w.sum(1)

        for k in knn_ks:
            oof_knn[k][te] = proba_from(dist, idx, k)
            te_knn[k] += proba_from(dist_t, idx_t, k) / 5.0
        log(
            f"  KNN fold{i} k101={auc(y[te],oof_knn[101][te]):.5f} "
            f"k201={auc(y[te],oof_knn[201][te]):.5f} "
            f"pca_var={pca_var[-1]:.3f} proto={len(proto)}"
        )

    knn_oof_auc = {k: auc(y, oof_knn[k]) for k in knn_ks}
    best_k = max(knn_oof_auc, key=knn_oof_auc.get)
    log(f"KNN OOF AUCs={knn_oof_auc} best_k={best_k}")

    # -------------------- MLP family --------------------
    # Hypothesis: nonlinear MLP on scaled features captures interactions trees miss at this FE depth.
    log("Training MLPClassifier (256,128) with early stopping…")
    oof_mlp = np.zeros(N_TR)
    te_mlp = np.zeros(N_TE)
    fa_mlp = []
    for i, (tr, te) in enumerate(folds):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=2048,
                        learning_rate_init=1e-3,
                        max_iter=80,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=8,
                        random_state=42 + i,
                        verbose=False,
                    ),
                ),
            ]
        )
        pipe.fit(Xtr[tr], y[tr])
        oof_mlp[te] = pipe.predict_proba(Xtr[te])[:, 1]
        te_mlp += pipe.predict_proba(Xte)[:, 1] / 5.0
        a = auc(y[te], oof_mlp[te])
        fa_mlp.append(a)
        n_iter = pipe.named_steps["mlp"].n_iter_
        log(f"  MLP fold{i} AUC={a:.6f} n_iter={n_iter}")
    log(f"MLP OOF={auc(y,oof_mlp):.6f} mean_fold={float(np.mean(fa_mlp)):.6f}")

    # wider/shallower MLP for diversity
    log("Training MLP-wide (512,) …")
    oof_mlp2 = np.zeros(N_TR)
    te_mlp2 = np.zeros(N_TE)
    fa_mlp2 = []
    for i, (tr, te) in enumerate(folds):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(512,),
                        activation="relu",
                        solver="adam",
                        alpha=3e-4,
                        batch_size=4096,
                        learning_rate_init=1e-3,
                        max_iter=60,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=6,
                        random_state=7 + i,
                        verbose=False,
                    ),
                ),
            ]
        )
        pipe.fit(Xtr[tr], y[tr])
        oof_mlp2[te] = pipe.predict_proba(Xtr[te])[:, 1]
        te_mlp2 += pipe.predict_proba(Xte)[:, 1] / 5.0
        a = auc(y[te], oof_mlp2[te])
        fa_mlp2.append(a)
        log(f"  MLP2 fold{i} AUC={a:.6f} n_iter={pipe.named_steps['mlp'].n_iter_}")
    log(f"MLP2 OOF={auc(y,oof_mlp2):.6f} mean_fold={float(np.mean(fa_mlp2)):.6f}")

    # -------------------- blends --------------------
    oof_knn_best, te_knn_best = oof_knn[best_k], te_knn[best_k]
    # pick better MLP
    if auc(y, oof_mlp) >= auc(y, oof_mlp2):
        oof_mlp_best, te_mlp_best, mlp_tag = oof_mlp, te_mlp, "mlp_256_128"
    else:
        oof_mlp_best, te_mlp_best, mlp_tag = oof_mlp2, te_mlp2, "mlp_512"

    # load HGB if available for 3-way native diversity blend
    hgb_oof = hgb_te = None
    npz = OUT / "v24_2_native_rf_gbm.npz"
    if npz.exists():
        z = np.load(npz)
        hgb_oof, hgb_te = z["oof_hgb"], z["test_hgb"]
        log(f"loaded HGB OOF={auc(y,hgb_oof):.6f} for optional mix")

    parts_o = [oof_knn_best, oof_mlp_best]
    parts_t = [te_knn_best, te_mlp_best]
    tags = ["knn", "mlp"]
    if hgb_oof is not None:
        parts_o.append(hgb_oof)
        parts_t.append(hgb_te)
        tags.append("hgb")

    best_w = tuple(np.ones(len(parts_o)) / len(parts_o))
    best_a = -1.0
    # small grid
    if len(parts_o) == 2:
        for w0 in np.linspace(0.2, 0.8, 13):
            w = (float(w0), float(1 - w0))
            a = auc(y, rblend(parts_o, w))
            if a > best_a:
                best_a, best_w = a, w
    else:
        for w0 in np.linspace(0.1, 0.5, 5):
            for w1 in np.linspace(0.1, 0.5, 5):
                w2 = 1 - w0 - w1
                if w2 < 0.1 or w2 > 0.7:
                    continue
                w = (float(w0), float(w1), float(w2))
                a = auc(y, rblend(parts_o, w))
                if a > best_a:
                    best_a, best_w = a, w
    blend_o = rblend(parts_o, best_w)
    blend_t = rblend(parts_t, best_w)
    log(f"KNN+MLP(+HGB?) blend OOF={auc(y,blend_o):.6f} w={dict(zip(tags,best_w))}")

    # -------------------- nested vs ATH --------------------
    ath = pd.read_csv(ATH).set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    Bo, Bt = [], []
    for name in (
        "gb_bag5",
        "gb_bag10",
        "lgb_diglead_l3",
        "realmlp_v1",
        "glm_dl",
        "ebm_all_bag",
        "gbn_bag5_w512",
    ):
        op, tp = BANK / f"oof_{name}.npy", BANK / f"test_{name}.npy"
        if op.exists() and tp.exists():
            Bo.append(unit(L(np.load(op).ravel()[:N_TR])))
            Bt.append(unit(L(np.load(tp).ravel()[:N_TE])))
    Bo_m, Bt_m = np.column_stack(Bo), np.column_stack(Bt)
    z_ath = Bo_m @ Ridge(alpha=10.0).fit(Bt_m, L(ath)).coef_ + Ridge(alpha=10.0).fit(Bt_m, L(ath)).intercept_
    # refit cleanly
    lr = Ridge(alpha=10.0).fit(Bt_m, L(ath))
    z_ath = Bo_m @ lr.coef_ + lr.intercept_
    log(f"ATH bank surrogate AUC={auc(y,z_ath):.6f}")

    # also vs gb_bag5 directly (cleaner base for residual test)
    gb = np.load(BANK / "oof_gb_bag5.npy").ravel()[:N_TR]
    gb_t = np.load(BANK / "test_gb_bag5.npy").ravel()[:N_TE]

    gates = {}
    for name, oo, tt in (
        ("knn", oof_knn_best, te_knn_best),
        ("mlp", oof_mlp_best, te_mlp_best),
        ("blend", blend_o, blend_t),
    ):
        # vs ATH surrogate
        d = unit(L(oo))
        d = unit(d - np.dot(d, unit(z_ath)) * unit(z_ath))
        te0 = folds[0][1]
        sgn = +1 if auc(y[te0], z_ath[te0] + 0.05 * d[te0]) >= auc(y[te0], z_ath[te0] - 0.05 * d[te0]) else -1
        per_ath = h2h(z_ath + 0.05 * sgn * d, z_ath, y, folds)
        # vs gb_bag5
        d2 = unit(L(oo))
        d2 = unit(d2 - np.dot(d2, unit(L(gb))) * unit(L(gb)))
        sgn2 = +1 if auc(y[te0], L(gb)[te0] + 0.05 * d2[te0]) >= auc(y[te0], L(gb)[te0] - 0.05 * d2[te0]) else -1
        # compare in probability space via rank blend for stability
        per_gb = h2h(rblend([gb, oo], [0.92, 0.08]), gb, y, folds)
        gates[name] = {"ath": per_ath, "gb": per_gb, "sgn": sgn, "rho_ath": omr(tt, ath)}
        log(
            f"GATE {name}: ATH-orth {per_ath['pos']}/5 t={per_ath['t']:.2f} m={per_ath['mean']:+.2f} | "
            f"gb+8% {per_gb['pos']}/5 t={per_gb['t']:.2f} m={per_gb['mean']:+.2f} | "
            f"solo={auc(y,oo):.5f} ρ(ATH)={omr(tt,ath):.1f}e-4"
        )

    # -------------------- stage + optional submit --------------------
    np.savez_compressed(
        OUT / "v24_3_native_knn_mlp.npz",
        oof_knn101=oof_knn[101],
        oof_knn201=oof_knn[201],
        test_knn101=te_knn[101],
        test_knn201=te_knn[201],
        oof_mlp=oof_mlp,
        test_mlp=te_mlp,
        oof_mlp2=oof_mlp2,
        test_mlp2=te_mlp2,
        oof_blend=blend_o,
        test_blend=blend_t,
        best_k=np.int32(best_k),
        blend_w=np.array(best_w),
        mlp_tag=np.array(mlp_tag),
    )
    log(f"SAVED {OUT/'v24_3_native_knn_mlp.npz'}")

    cands = {
        "native_knn": rf(te_knn_best),
        "native_mlp": rf(te_mlp_best),
        "native_knn_mlp_blend": rf(blend_t),
        "ATH_x_knn_0.05": rf(0.95 * L(ath) + 0.05 * L(te_knn_best)),
        "ATH_x_mlp_0.05": rf(0.95 * L(ath) + 0.05 * L(te_mlp_best)),
        "ATH_x_knn_mlp_0.08": rf(0.92 * L(ath) + 0.08 * L(blend_t)),
        "ATH_x_mlp_0.10": rf(0.90 * L(ath) + 0.10 * L(te_mlp_best)),
    }
    staged = []
    for tag, vec in cands.items():
        path = SUB / f"sub_v24_3_{tag}.csv"
        pd.DataFrame({"id": ids, "is_fraud": vec}).to_csv(path, index=False)
        staged.append({"tag": tag, "path": path, "rho_ath": omr(vec, ath)})
        log(f"STAGED {path.name} ρ(ATH)={staged[-1]['rho_ath']:.3f}e-4")

    # Prefer MLP into ATH if gb-gate positive; else small knn/mlp blend
    order = []
    if gates["mlp"]["gb"]["pos"] >= 4 and gates["mlp"]["gb"]["mean"] > 0:
        order.append("ATH_x_mlp_0.05")
        order.append("ATH_x_mlp_0.10")
    if gates["knn"]["gb"]["pos"] >= 4 and gates["knn"]["gb"]["mean"] > 0:
        order.append("ATH_x_knn_0.05")
    order.append("ATH_x_knn_mlp_0.08")
    # dedupe
    seen = set()
    order = [t for t in order if not (t in seen or seen.add(t))]

    live = []
    by = {s["tag"]: s for s in staged}
    for tag in order:
        s = by[tag]
        msg = (
            f"v24.3 {tag} | ρ(ATH)={s['rho_ath']:.3f}e-4 | "
            f"KNN{best_k}={knn_oof_auc[best_k]:.4f} {mlp_tag}={auc(y,oof_mlp_best):.4f} "
            f"blend={auc(y,blend_o):.4f}"
        )[:480]
        log(f"SUBMIT {msg}")
        ok, pub, status = submit_and_poll(s["path"], msg)
        live.append({"tag": tag, "pub": pub, "status": status, "rho_ath": s["rho_ath"]})
        log(f"DONE {tag} pub={pub} status={status}")
        if status == "quota_or_error":
            log("STOP quota — remaining staged")
            break
        time.sleep(5)

    best = max((r["pub"] for r in live if r.get("pub") is not None), default=None)

    body = [
        "# GM V24.3 Native KNN & Neural Net (MLP)",
        "",
        f"ATH ref **0.95440** · elapsed {time.time()-T0:.0f}s",
        "",
        "## Hypotheses",
        "",
        "1. **KNN:** local neighborhoods in scaled+PCA space carry fraud signal not captured by global linear/logit blends.",
        "2. **MLP:** depth-2 ReLU net on the same FE learns interactions complementary to trees / ATH dual-base.",
        "",
        "## OOF results",
        "",
        "| Model | OOF AUC | Notes |",
        "|---|---:|---|",
        f"| KNN k=101 | {knn_oof_auc[101]:.6f} | PCA-24, 50k prototypes, distance weights |",
        f"| KNN k=201 | {knn_oof_auc[201]:.6f} | |",
        f"| **KNN best (k={best_k})** | **{knn_oof_auc[best_k]:.6f}** | canary 120k/PCA32≈0.91 |",
        f"| MLP (256,128) | {auc(y,oof_mlp):.6f} | folds {', '.join(f'{a:.4f}' for a in fa_mlp)} |",
        f"| MLP (512,) | {auc(y,oof_mlp2):.6f} | folds {', '.join(f'{a:.4f}' for a in fa_mlp2)} |",
        f"| **Selected MLP (`{mlp_tag}`)** | **{auc(y,oof_mlp_best):.6f}** | |",
        f"| KNN+MLP(+HGB) blend {best_w} | **{auc(y,blend_o):.6f}** | |",
        "",
        "## Nested gates",
        "",
        "| model | vs ATH-orth | vs gb+8% | ρ(test,ATH)e-4 |",
        "|---|---|---|---:|",
    ]
    for name, g in gates.items():
        a, b = g["ath"], g["gb"]
        body.append(
            f"| `{name}` | {a['pos']}/5 t={a['t']:.2f} m={a['mean']:+.2f} | "
            f"{b['pos']}/5 t={b['t']:.2f} m={b['mean']:+.2f} | {g['rho_ath']:.1f} |"
        )
    body += [
        "",
        "## Live submits",
        "",
        "| tag | Public | status |",
        "|---|---:|---|",
    ]
    for r in live:
        body.append(f"| `{r['tag']}` | **{r.get('pub')}** | {r.get('status')} |")
    body += [
        "",
        f"- Best live: **{best}**",
        "",
        "## Expert takeaway",
        "",
        "- Solo KNN/MLP will not beat 0.95440; expect OOF in the mid-0.93–0.94s unless FE is upgraded.",
        "- Value is **diversity into ATH** at small w if `gb+8%` nested gate is green.",
        "- Next reset: submit staged `ATH_x_mlp_*` / `ATH_x_knn_mlp_*` before pure natives.",
        "",
        "## Log",
        "```",
        *LINES,
        "```",
        "",
    ]
    REPORT.write_text("\n".join(body) + "\n")
    OUT.joinpath("v24_3_knn_mlp_meta.json").write_text(
        json.dumps(
            {
                "knn_oof": knn_oof_auc,
                "best_k": best_k,
                "mlp": auc(y, oof_mlp),
                "mlp2": auc(y, oof_mlp2),
                "mlp_tag": mlp_tag,
                "blend": auc(y, blend_o),
                "blend_w": best_w,
                "gates": gates,
                "live": live,
                "best": best,
                "pca_var_mean": float(np.mean(pca_var)),
                "elapsed": time.time() - T0,
            },
            indent=2,
            default=float,
        )
    )
    log(f"REPORT {REPORT} best={best}")


if __name__ == "__main__":
    main()
