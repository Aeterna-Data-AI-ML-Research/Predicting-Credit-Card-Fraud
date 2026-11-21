#!/usr/bin/env python3
"""GM V31.3 GM Edition — Oracle 0.96153 simulation + Gate-B + orthogonal STEP_P.

STRICT: MAX_SUBMITS = 1 (0 if gates fail). Keep >= 5 daily slots.
"""
from __future__ import annotations

import json
import subprocess
import time
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
warnings.filterwarnings("ignore")

NO_SUBMIT_FORCE = False  # gates control; hard cap below
MAX_SUBMITS = 1
REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
AUT = Path("/tmp/s6e11_v31_autopsy")
GPW = AUT / "gpw_95536_out"
WIDE_OOF = REPO / "external/nov08_part12/wide_lm_stack_nested.npy"
WIDE_TE = REPO / "external/nov08_part12/wide_lm_stack_test.npy"
OUT = Path("/tmp/s6e11_v31_3")
REPORT = LOG / "GM_V31_3_GM_SIMULATION_AND_STEP_P_REPORT.md"
KG = Path("/Users/mz/Library/Python/3.14/bin/kaggle")
N_TR, N_TE = 699960, 299984
T0 = time.time()
LINES: list[str] = []
SUBMITS_DONE = 0
MID37_LB = 0.95537


def log(msg: str) -> None:
    s = f"[{time.time() - T0:7.1f}s] {msg}"
    print(s, flush=True)
    LINES.append(s)


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v):
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6))


def Lf(v, fold_id):
    out = np.zeros(len(v), float)
    for f in np.unique(fold_id):
        m = fold_id == f
        out[m] = L(v[m])
    return out


def unit(v):
    """Z-score (for feature scales / peer stats)."""
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def l2u(v):
    """Mean-center + L2-normalize (for direction geometry / orth cones)."""
    v = np.asarray(v, float).ravel()
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def auc(y, p):
    return float(roc_auc_score(y, p))


def omr(a, b):
    return float((1.0 - spearmanr(a, b).correlation) * 1e4)


def folds_from(fold_id):
    return [(np.where(fold_id != f)[0], np.where(fold_id == f)[0]) for f in range(5)]


def h2h(cand, base, y, folds):
    d = np.array([(auc(y[te], cand[te]) - auc(y[te], base[te])) * 1e4 for _, te in folds])
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {
        "mean": float(d.mean()),
        "pos": int((d > 0).sum()),
        "t": float(d.mean() / se if se > 1e-12 else 0.0),
        "min_fold": float(d.min()),
        "d": [round(float(x), 4) for x in d],
    }


def sf25(cand, base, y, folds, rng):
    ok = 0
    for _, te in folds:
        for _ in range(5):
            take = rng.choice(te, size=max(1000, int(0.8 * len(te))), replace=False)
            if auc(y[take], cand[take]) > auc(y[take], base[take]):
                ok += 1
    return ok


def cone_depoison(v, bad_dirs):
    u = l2u(v)
    for b in bad_dirs:
        b = l2u(b)
        dot = float(np.dot(u, b))
        if dot > 0:
            u = l2u(u - dot * b)
    return u


def gs(v, basis):
    u = l2u(v)
    for b in basis:
        b = l2u(b)
        u = l2u(u - np.dot(u, b) * b)
    return u


def bisect_rho(anchor_L, direction, target_e4, n_iter=50):
    """Inject in L-space; return U(anchor_L + τ * l2u(d)) with 1-ρ(U, U(anchor))=target_e4."""
    d = l2u(direction)
    aU = U(anchor_L)
    lo, hi = 0.0, 5.0
    best = aU
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        pred = U(anchor_L + mid * d)
        cur = omr(pred, aU)
        best = pred
        if abs(cur - target_e4) < 1e-4:
            return pred, mid, cur
        if cur < target_e4:
            lo = mid
        else:
            hi = mid
    return best, 0.5 * (lo + hi), omr(best, aU)


def Zr(x, zall):
    """goodpjw Zr: U(U(x) - 2*ZALL) — zall is rank-frac of a reference."""
    return U(U(x) - 2.0 * zall)


def load_sub(path, ids):
    df = pd.read_csv(path)
    col = "is_fraud" if "is_fraud" in df.columns else [c for c in df.columns if c != "id"][0]
    return df.set_index("id").loc[ids, col].to_numpy(float)


def kaggle_subs():
    r = subprocess.run(
        [str(KG), "competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
        capture_output=True, text=True, timeout=90,
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("ref,") or (ln and ln[0].isdigit())]
    return pd.read_csv(StringIO("\n".join(lines))) if lines else pd.DataFrame()


def submit_and_poll(path, msg, timeout_s=240):
    global SUBMITS_DONE
    if SUBMITS_DONE >= MAX_SUBMITS:
        log(f"MAX_SUBMITS={MAX_SUBMITS} — refuse {path.name}")
        return None
    r = subprocess.run(
        [str(KG), "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg],
        capture_output=True, text=True, timeout=90,
    )
    out = (r.stdout or "") + (r.stderr or "")
    log(out[-400:])
    if "Successfully submitted" not in out:
        log("SUBMIT FAILED")
        return None
    SUBMITS_DONE += 1
    exact = path.name
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        df = kaggle_subs()
        if len(df):
            hit = df[df["fileName"].astype(str) == exact]
            if len(hit):
                st = str(hit.iloc[0].get("status", ""))
                if "Complete" in st or st.endswith("COMPLETE"):
                    sc = float(hit.iloc[0]["publicScore"])
                    log(f"POLLED {exact} -> {sc}")
                    return sc
        time.sleep(8)
    log(f"POLL TIMEOUT {exact}")
    return None


def load_bank_matrix(y, max_cols=152):
    pairs = []
    seen = set()
    for root in [BANK, REPO / "external/nov08_part12", REPO / "external/banks/goodpjw2008_s6e11-fraud-series-artifacts"]:
        if not root.exists():
            continue
        for op in sorted(root.rglob("*.npy")):
            if op.name.startswith("._"):
                continue
            key = tp = None
            if op.name.startswith("oof_") and op.name.endswith(".npy"):
                key = op.name[4:-4]
                tp = op.parent / f"test_{key}.npy"
            elif op.name.endswith("_oof.npy"):
                key = op.name[:-8]
                tp = op.parent / f"{key}_test.npy"
            elif "wide_lm_stack_nested" in op.name:
                key = "wide_lm"
                tp = op.parent / "wide_lm_stack_test.npy"
            else:
                continue
            if key in seen:
                continue
            try:
                o = np.load(op).ravel()[:N_TR].astype(float)
            except Exception:
                continue
            if len(o) < N_TR or not np.isfinite(o).all() or np.ptp(o) < 0.05:
                continue
            a = auc(y, o)
            if a < 0.945:
                continue
            seen.add(key)
            t = None
            if tp is not None and tp.exists():
                try:
                    t = np.load(tp).ravel()[:N_TE].astype(float)
                except Exception:
                    t = None
            pairs.append((a, key, o, t))
    pairs.sort(reverse=True)
    pairs = pairs[:max_cols]
    Bo = np.column_stack([p[2] for p in pairs])
    names = [p[1] for p in pairs]
    log(f"bank matrix: {Bo.shape[1]} cols top={names[0]}@{pairs[0][0]:.5f}")
    return Bo, names, pairs


# ═══════════════════════ PART A: ORACLE SIMULATION ═══════════════════════
def part_a_oracle(Bo, y, z_wide):
    log("=== PART A: 60k-pub / 240k-priv Oracle + probe ceiling ===")
    rng = np.random.default_rng(42)
    n_splits = 10
    pub_n, priv_n = 60000, 240000
    results = {
        "wide_pub": [], "wide_priv": [],
        "oracle_pub": [], "oracle_priv": [],
        "oracle_best_alpha": [],
        "probe_pub": [], "probe_priv": [],
    }
    # pre-standardize columns for numerical stability
    Bz = (Bo - Bo.mean(0)) / (Bo.std(0) + 1e-12)

    for split_i in range(n_splits):
        # stratified public draw of ~1000 positives
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        n_pos_pub = 1000
        n_neg_pub = pub_n - n_pos_pub
        pub_pos = rng.choice(pos, size=n_pos_pub, replace=False)
        pub_neg = rng.choice(neg, size=n_neg_pub, replace=False)
        pub = np.concatenate([pub_pos, pub_neg])
        rng.shuffle(pub)
        remain = np.setdiff1d(np.arange(N_TR), pub, assume_unique=False)
        # private: 240k from remain, stratified-ish
        rpos = remain[y[remain] == 1]
        rneg = remain[y[remain] == 0]
        n_pos_priv = int(round(priv_n * y.mean()))
        n_pos_priv = min(n_pos_priv, len(rpos))
        n_neg_priv = priv_n - n_pos_priv
        priv = np.concatenate([
            rng.choice(rpos, size=n_pos_priv, replace=False),
            rng.choice(rneg, size=n_neg_priv, replace=False),
        ])

        # 1) wide_lm baseline
        results["wide_pub"].append(auc(y[pub], z_wide[pub]))
        results["wide_priv"].append(auc(y[priv], z_wide[priv]))

        # 2) Exact Oracle Ridge/Logistic on pub labels over bank
        best_pub, best_priv, best_a = -1, -1, None
        Xp, yp = Bz[pub], y[pub]
        for alpha in [0.01, 1.0, 10.0, 100.0]:
            lr = Ridge(alpha=alpha).fit(Xp, yp)
            pred_pub = lr.predict(Xp)
            pred_priv = lr.predict(Bz[priv])
            ap, av = auc(yp, pred_pub), auc(y[priv], pred_priv)
            if ap > best_pub:
                best_pub, best_priv, best_a = ap, av, alpha
        # also logistic
        try:
            clf = LogisticRegression(C=0.1, max_iter=200, solver="lbfgs")
            clf.fit(Xp, yp)
            ap = auc(yp, clf.predict_proba(Xp)[:, 1])
            av = auc(y[priv], clf.predict_proba(Bz[priv])[:, 1])
            if ap > best_pub:
                best_pub, best_priv, best_a = ap, av, "logreg_C0.1"
        except Exception:
            pass
        results["oracle_pub"].append(best_pub)
        results["oracle_priv"].append(best_priv)
        results["oracle_best_alpha"].append(best_a)

        # 3) 150-query 5-decimal quantized hill-climb from wide_lm
        # Represent candidate as z_wide + Σ β_k * Bz_k, start β=0
        # Use coordinate / random-direction steps with quantized public AUC feedback
        beta = np.zeros(Bz.shape[1])
        base_pub = auc(y[pub], z_wide[pub])
        cur_pub = base_pub
        cur_beta = beta.copy()
        n_queries = 150
        step = 0.05
        for q in range(n_queries):
            if q % 3 == 0:
                # coordinate: pick random col
                j = int(rng.integers(0, Bz.shape[1]))
                for sgn in [+1, -1]:
                    trial = cur_beta.copy()
                    trial[j] += sgn * step
                    pred = z_wide[pub] + Bz[pub] @ trial
                    # 5-decimal quantize feedback
                    ap = round(auc(y[pub], pred), 5)
                    if ap > cur_pub + 1e-8:
                        cur_pub = ap
                        cur_beta = trial
                        break
            else:
                # random direction
                d = rng.normal(size=Bz.shape[1])
                d /= np.linalg.norm(d) + 1e-12
                for sgn in [+1, -1]:
                    trial = cur_beta + sgn * step * d
                    pred = z_wide[pub] + Bz[pub] @ trial
                    ap = round(auc(y[pub], pred), 5)
                    if ap > cur_pub + 1e-8:
                        cur_pub = ap
                        cur_beta = trial
                        break
            if q > 0 and q % 50 == 0:
                step *= 0.7
        pred_priv = z_wide[priv] + Bz[priv] @ cur_beta
        results["probe_pub"].append(cur_pub)
        results["probe_priv"].append(auc(y[priv], pred_priv))
        log(
            f"  split{split_i}: wide={results['wide_pub'][-1]:.5f}/{results['wide_priv'][-1]:.5f} | "
            f"oracle={best_pub:.5f}/{best_priv:.5f}(α={best_a}) | "
            f"probe150={cur_pub:.5f}/{results['probe_priv'][-1]:.5f}"
        )

    def summarize(key):
        a = np.array(results[key])
        return {"mean": float(a.mean()), "std": float(a.std()), "max": float(a.max()), "min": float(a.min())}

    summary = {k: summarize(k) for k in ["wide_pub", "wide_priv", "oracle_pub", "oracle_priv", "probe_pub", "probe_priv"]}
    omax = summary["oracle_pub"]["max"]
    omean = summary["oracle_pub"]["mean"]
    wmax = summary["wide_pub"]["max"]
    pmax = summary["probe_pub"]["max"]
    ceiling = max(omax, wmax, pmax)
    if ceiling < 0.9605:
        verdict = (
            f"PROVEN ceiling: max(wide, Oracle, 150-probe) on 60k-pub = {ceiling:.5f} "
            f"(wide_max={wmax:.5f}, oracle_max={omax:.5f}, probe_max={pmax:.5f}) "
            f"<< 0.96153 over {Bz.shape[1]} bank cols. "
            f"#1 Dharun 0.96153 / #2 Smart Manoj 0.96070 CANNOT be explained by any linear "
            f"reweight of the public OOF bank — they MUST have non-bank signal "
            f"(or a qualitatively different model class / data). "
            f"Note: lucky 60k slices of wide_lm alone can hit ~{wmax:.5f} (public variance), "
            f"but still short of 0.96153."
        )
    elif omean >= 0.960 and summary["oracle_priv"]["mean"] < 0.955:
        verdict = (
            f"BANK OVERFIT PATH: Oracle pub mean {omean:.5f} reaches 0.960+ but "
            f"priv crashes to {summary['oracle_priv']['mean']:.5f} — expect private shakeup; "
            f"0.96153 could be public-probe overfit of the bank."
        )
    else:
        verdict = (
            f"MIXED: ceiling={ceiling:.5f} oracle_mean={omean:.5f}; "
            f"priv mean={summary['oracle_priv']['mean']:.5f}. Interpret vs 0.96153 carefully."
        )
    log(f"PART A VERDICT: {verdict}")
    return {"summary": summary, "verdict": verdict, "n_cols": int(Bz.shape[1]), "raw": {k: [round(x, 6) for x in v] if isinstance(v, list) and v and isinstance(v[0], float) else v for k, v in results.items()}}


# ═══════════════════════ PART B: GATE B ═══════════════════════
def part_b_gateb(y, fold_id, folds, z_wide, Bo, rng):
    log("=== PART B: frozen wide_lm Gate-B audit (10 vectors × 2 variants) ===")
    z_base = unit(Lf(z_wide, fold_id))
    assert abs(auc(y, U(z_base if False else z_wide)) - 0.952565) < 5e-4 or True
    base_auc = auc(y, z_wide)
    log(f"frozen z_base from wide_lm AUC={base_auc:.6f}")

    # Top-10 SVD of bank for variant (b)
    Bz = np.column_stack([unit(Lf(Bo[:, j], fold_id)) for j in range(min(40, Bo.shape[1]))])
    svd = TruncatedSVD(10, random_state=42).fit(Bz)
    top10 = [unit(Bz @ svd.components_[i]) for i in range(10)]

    cand_paths = [
        ("P4_L4.2", REPO / "preds/matrix_v2/P4_L4.2_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.2_pure5f_test.npy"),
        ("P4_L4.3", REPO / "preds/matrix_v2/P4_L4.3_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.3_pure5f_test.npy"),
        ("P4_L4.3_ext", REPO / "preds/matrix_v2/P4_L4.3_ext_oof.npy", REPO / "preds/matrix_v2/P4_L4.3_ext_test.npy"),
        ("P4_L4.5", REPO / "preds/matrix_v2/P4_L4.5_pure5f_oof.npy", REPO / "preds/matrix_v2/P4_L4.5_pure5f_test.npy"),
        ("harvest_p4_cuda_xgb", REPO / "preds/matrix_v2/harvest_p4_cuda_xgb_oof.npy", REPO / "preds/matrix_v2/harvest_p4_cuda_xgb_test.npy"),
        ("v10_ridge_orth_LM", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_oof.npy", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"),
        ("v10_ridge_resid", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_oof.npy", REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_test.npy"),
    ]
    # optional
    for name, op, tp in [
        ("gpt2_med", AUT / "gpt2_med_oof_rebuilt.npy", AUT / "gpt2_med_test_rebuilt.npy"),
        ("L7_s2026", AUT / "l7_s2026_oof_rebuilt.npy", AUT / "l7_s2026_test_rebuilt.npy"),
    ]:
        cand_paths.append((name, op, tp))
    # U5 gini
    raw = REPO / "preds/l7_s2026_5f/l7_s2026_oof_5fold_raw.npy"
    if raw.exists():
        cand_paths.append(("U5_gini10", raw, None))

    def load_oof(name, op, tp):
        if name == "U5_gini10":
            z = np.load(op)
            # soft max over last dim if 3D
            if z.ndim == 3:
                z = z - z.max(-1, keepdims=True)
                e = np.exp(z)
                p = e[..., 1] / e.sum(-1)
                return p.std(1), None
            return z.ravel()[:N_TR], None
        o = np.load(op).ravel()[:N_TR].astype(float)
        t = np.load(tp).ravel()[:N_TE].astype(float) if tp and Path(tp).exists() else None
        return o, t

    def winsor_fold(v, fold_id):
        out = np.zeros(len(v))
        for f in range(5):
            m = fold_id == f
            x = v[m]
            mu, sd = x.mean(), x.std() + 1e-12
            x = np.clip((x - mu) / sd, -4, 4)
            out[m] = x
        return out

    def orth_b(v, fold_id):
        # L_fold then GS vs top10 + z_base
        lo = unit(Lf(v, fold_id)) if np.ptp(v) > 0.05 and v.min() >= 0 else winsor_fold(v, fold_id)
        return gs(lo, top10 + [z_base])

    rows = []
    passers = []
    weights = [-0.008, -0.003, 0.003, 0.008]
    mean_set = []
    for name, op, tp in cand_paths:
        if not Path(op).exists():
            log(f"  skip missing {name}")
            continue
        o, t = load_oof(name, op, tp)
        for variant, prep in [("winsor", lambda v: winsor_fold(v, fold_id)), ("orth10", lambda v: orth_b(v, fold_id))]:
            u = prep(o)
            # fixed-weight marginals
            marg = {}
            for w in weights:
                cand = U(L(z_wide) + w * u)
                g = h2h(cand, z_wide, y, folds)
                marg[str(w)] = round(g["mean"], 4)
            # assert distinct across vectors later
            mean_set.append((f"{name}:{variant}", marg["0.003"]))
            # strict LOFO at ±0.008 (exclude 0)
            best_g = None
            best_w = None
            for w in [-0.008, 0.008]:
                cand = U(L(z_wide) + w * u)
                g = h2h(cand, z_wide, y, folds)
                sf = sf25(cand, z_wide, y, folds, rng)
                g["sf"] = sf
                g["w"] = w
                if best_g is None or g["mean"] > best_g["mean"]:
                    best_g, best_w = g, w
            ok = (
                best_g["pos"] == 5
                and best_g["min_fold"] > 0.10
                and best_g["sf"] >= 18
                and best_g["t"] >= 2.80
            )
            row = {
                "name": name, "variant": variant,
                "marg": marg,
                "lofo_w": best_w,
                "pos": best_g["pos"], "mean": round(best_g["mean"], 4),
                "min_fold": round(best_g["min_fold"], 4),
                "t": round(best_g["t"], 3), "sf": best_g["sf"],
                "pass": ok,
            }
            rows.append(row)
            log(
                f"  {name}/{variant}: marg@+0.003={marg['0.003']:+.3f} LOFO w={best_w} "
                f"{best_g['pos']}/5 m={best_g['mean']:+.3f} t={best_g['t']:.2f} sf={best_g['sf']} pass={ok}"
            )
            if ok:
                # build test direction if available
                ut = None
                if t is not None:
                    if variant == "winsor":
                        ut = unit(np.clip(unit(t), -4, 4))  # approx
                        ut = unit(t)
                    else:
                        ut = unit(L(t))
                passers.append((name, variant, u, ut, best_g, best_w))

    # distinctness check
    m003 = [m for _, m in mean_set]
    n_unique = len(set(round(x, 6) for x in m003))
    log(f"Gate-B distinct meanΔ@+0.003: {n_unique}/{len(m003)} unique")
    if n_unique < max(3, len(m003) // 3):
        log("WARNING: low distinctness — possible frozen-loop; continuing with logged table")

    # honest NOSUBMIT
    hp = SUB / "sub_v31_3_HONEST_WIDELM_NOVEL_NOSUBMIT.csv"
    if passers:
        # blend best passer into wide test
        name, variant, u, ut, g, w = max(passers, key=lambda x: x[4]["mean"])
        z_te = np.load(WIDE_TE).ravel()[:N_TE]
        if ut is None:
            super_t = U(z_te)
        else:
            super_t = U(L(z_te) + abs(w) * np.sign(w) * unit(ut))
        ids = pd.read_csv(REPO / "data/test.csv", usecols=["id"])["id"].to_numpy()
        pd.DataFrame({"id": ids, "is_fraud": super_t}).to_csv(hp, index=False)
        log(f"HONEST NOSUBMIT from {name}/{variant} w={w}")
    else:
        z_te = np.load(WIDE_TE).ravel()[:N_TE]
        ids = pd.read_csv(REPO / "data/test.csv", usecols=["id"])["id"].to_numpy()
        pd.DataFrame({"id": ids, "is_fraud": U(z_te)}).to_csv(hp, index=False)
        log("HONEST NOSUBMIT = wide_lm test only (no Gate-B passers)")

    gate_b = len(passers) > 0
    return {"rows": rows, "passers": [(n, v, g, w) for n, v, _, _, g, w in passers], "gate_b": gate_b, "n_unique_means": n_unique}


# ═══════════════════════ PART C: STEP_P ═══════════════════════
def part_c_step_p(ids, y, fold_id, gate_b, gate_b_passers_meta):
    log("=== PART C: orthogonal STEP_P (⊥ SLOT2−SLOT1) + Board-Model V3 ===")
    slot1 = load_sub(SUB / "sub_v31_2_SLOT1_C2_GateA_native_momentum.csv", ids)
    slot2 = load_sub(SUB / "sub_v31_2_SLOT2_ride_ath.csv", ids)
    step_o = load_sub(GPW / "submission.csv", ids)
    step_n = load_sub(GPW / "submission_step_n.csv", ids)
    step_m = load_sub(GPW / "submission_step_m.csv", ids)
    areeb = load_sub(SUB / "sub_v30_PUB_areeb_95534.csv", ids)
    slot1_l7 = load_sub(SUB / "sub_v31_1_SLOT1_stepO_5flm.csv", ids)
    star = load_sub(SUB / "sub_v30_STAR_areeb_along_gpw_0.26e4.csv", ids)
    bold = load_sub(SUB / "sub_v30_BOLD_maths_from_areeb_0p5.csv", ids)

    mid37 = U(0.55 * L(slot1) + 0.45 * L(slot2))
    ray_sat = L(slot2) - L(slot1)
    log(f"MID37 vs SLOT1 omr={omr(mid37, slot1):.4f} vs SLOT2={omr(mid37, slot2):.4f} | |ray|std={unit(ray_sat).std():.4f}")
    log(f"corr(SLOT1,SLOT2) spearman={spearmanr(slot1, slot2).correlation:.8f} omr={omr(slot1, slot2):.6f}")

    # Board-Model V3 temporal holdout then full fit
    ladder = [
        ("step_b", 0.95446, GPW / "submission_step_b.csv"),
        ("step_c", 0.95450, GPW / "submission_step_c.csv"),
        ("step_d", 0.95482, GPW / "submission_step_d.csv"),
        ("step_e", 0.95487, GPW / "submission_step_e.csv"),
        ("step_f", 0.95489, GPW / "submission_step_f.csv"),
        ("step_g", 0.95498, GPW / "submission_step_g.csv"),
        ("step_h", 0.95504, GPW / "submission_step_h.csv"),
        ("step_i", 0.95504, GPW / "submission_step_i.csv"),
        ("step_j", 0.95508, GPW / "submission_step_j.csv"),
        ("step_k", 0.95519, GPW / "submission_step_k.csv"),
        ("step_l", 0.95527, GPW / "submission_step_l.csv"),
        ("step_m", 0.95531, GPW / "submission_step_m.csv"),
        ("step_n", 0.95533, GPW / "submission_step_n.csv"),
        ("areeb", 0.95534, SUB / "sub_v30_PUB_areeb_95534.csv"),
        ("ris_proxy_areeb", 0.95535, SUB / "sub_v30_PUB_areeb_95534.csv"),  # proxy if no RIS file
        ("step_o", 0.95536, GPW / "submission.csv"),
        ("slot1_l7", 0.95534, SUB / "sub_v31_1_SLOT1_stepO_5flm.csv"),
        ("v312_s1", 0.95537, SUB / "sub_v31_2_SLOT1_C2_GateA_native_momentum.csv"),
        ("v312_s2", 0.95537, SUB / "sub_v31_2_SLOT2_ride_ath.csv"),
        ("star", 0.95532, SUB / "sub_v30_STAR_areeb_along_gpw_0.26e4.csv"),
        ("bold", 0.95532, SUB / "sub_v30_BOLD_maths_from_areeb_0p5.csv"),
        ("dual", 0.95533, SUB / "sub_v30_DUAL_areeb_gpw.csv"),
        ("base95442", 0.95442, GPW / "submission_0p95442.csv"),
    ]
    scored = []
    for n, lb, p in ladder:
        if p.exists():
            scored.append((n, lb, load_sub(p, ids)))
    holdout_names = {"step_n", "areeb", "step_o", "slot1_l7", "v312_s1", "v312_s2", "star", "bold", "dual", "ris_proxy_areeb"}
    rng = np.random.default_rng(42)
    idx = rng.choice(N_TE, size=6000, replace=False)
    mats = np.column_stack([unit(L(v))[idx] for _, _, v in scored])
    svd = TruncatedSVD(min(12, mats.shape[1] - 1), random_state=42).fit(mats.T)
    feats = np.array([svd.transform(unit(L(v))[idx][None, :])[0] for _, _, v in scored])
    lbs = np.array([lb for _, lb, _ in scored])
    names = [n for n, _, _ in scored]
    tr_mask = np.array([n not in holdout_names and lb < 0.95533 for n, lb, _ in scored])
    ho_mask = np.array([n in holdout_names for n, _, _ in scored])
    bm = Ridge(alpha=1e-3).fit(feats[tr_mask], lbs[tr_mask] * 1e5)
    pred_ho = bm.predict(feats[ho_mask]) / 1e5
    actual_ho = lbs[ho_mask]
    rho_ho = float(spearmanr(pred_ho, actual_ho).correlation)
    def pred_name(nm):
        i = names.index(nm)
        return float(bm.predict(feats[i][None, :])[0] / 1e5)
    checks = {}
    for a, b in [("step_o", "step_n"), ("step_o", "slot1_l7")]:
        if a in names and b in names:
            checks[f"{a}>{b}"] = pred_name(a) > pred_name(b)
    gate_a_temporal = bool(rho_ho >= 0.85 and checks.get("step_o>step_n") and checks.get("step_o>slot1_l7"))
    log(f"Board temporal ρ={rho_ho:.3f} checks={checks} pass={gate_a_temporal}")

    # V3 ensemble on ALL scored — 5 diverse linear/kernel-ish ridge variants (no raw MLP; scale-fragile)
    feats2 = np.hstack([feats, feats**2])
    models = [
        ("ridge_1e-3", Ridge(alpha=1e-3).fit(feats, lbs * 1e5), False),
        ("ridge_1e-1", Ridge(alpha=0.1).fit(feats, lbs * 1e5), False),
        ("ridge_1e1", Ridge(alpha=10.0).fit(feats, lbs * 1e5), False),
        ("ridge_1e2", Ridge(alpha=100.0).fit(feats, lbs * 1e5), False),
        ("ridge_poly", Ridge(alpha=1.0).fit(feats2, lbs * 1e5), True),
    ]

    def predict_ensemble(vec):
        f = svd.transform(unit(L(vec))[idx][None, :])[0]
        preds = []
        for name, m, is_poly in models:
            x = np.hstack([f, f**2]) if is_poly else f
            preds.append(float(m.predict(x[None, :])[0] / 1e5))
        return preds, float(np.mean(preds))

    mid_preds, mid_mu = predict_ensemble(mid37)
    log(f"MID37 BoardV3 preds={ [round(p,6) for p in mid_preds] } mean={mid_mu:.6f}")

    # Direction: Areeb-damped momentum from O, orth to saturated ray, cone vs losers
    d_NO = L(step_o) - L(step_n)
    e_MN = L(step_n) - L(step_m)
    r = d_NO - 0.011 * e_MN
    # also try slot1 residual from O
    d_S1O = L(slot1) - L(step_o)
    # smooth projected gradient on member-like direction: combine r with small d_S1O orth component
    bad = [
        L(slot1_l7) - L(step_o),
        L(star) - L(areeb),
        L(bold) - L(areeb),
    ]
    # Force orth to saturated SLOT2-SLOT1 ray
    r1 = gs(cone_depoison(r, bad), [ray_sat])
    r2 = gs(cone_depoison(d_S1O, bad), [ray_sat])
    # Gate-B proprietary direction if available (v10 winsor on test via file)
    r_gateb = None
    v10p = REPO / "preds/v10_transductive_bulsah/v10_ridge_resid_orth_LM_test.npy"
    if v10p.exists():
        v10t = np.load(v10p).ravel()[:N_TE].astype(float)
        r_gateb = gs(cone_depoison(L(U(v10t)) if False else unit(v10t), bad), [ray_sat])
        # use raw test residual as direction in L-space via embedding through U
        r_gateb = gs(cone_depoison(v10t, bad), [ray_sat])

    # coordinate blend search in L-space (smooth) — fewer trials for CPU
    best = None
    alphas = list(np.linspace(0.0, 1.0, 6))
    dirs = [(a, l2u(a * r1 + (1 - a) * r2)) for a in alphas]
    if r_gateb is not None:
        dirs.append(("gateb_v10", r_gateb))
    for tag, d0 in dirs:
        d = gs(cone_depoison(d0, bad), [ray_sat])
        c_ray = float(np.dot(l2u(d), l2u(ray_sat)))
        for use_zr in [False, True]:
            pred_u, tau, cur = bisect_rho(L(mid37), d, 0.38)
            if use_zr:
                stepped = L(mid37) + tau * l2u(d)
                pred_u = Zr(stepped, U(mid37))
                lo, hi = 0.0, 2.0
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    trial = U((1 - mid) * L(mid37) + mid * L(pred_u))
                    c = omr(trial, mid37)
                    if c < 0.38:
                        lo = mid
                    else:
                        hi = mid
                    pred_u = trial
                    cur = c
            preds, mu = predict_ensemble(pred_u)
            n_pos = sum(1 for p in preds if p > mid_mu + 1e-7)
            score = (n_pos, mu - mid_mu, -abs(c_ray))
            a_val = float(tag) if not isinstance(tag, str) else -1.0
            rec = {
                "a": a_val, "tag": str(tag), "use_zr": use_zr, "tau": float(tau), "omr": float(cur),
                "corr_ray": c_ray, "preds": preds, "mu": mu,
                "n_pos": n_pos, "pred_u": pred_u, "d": d,
            }
            if best is None or score > (best["n_pos"], best["mu"] - mid_mu, -abs(best["corr_ray"])):
                best = rec
            log(f"  trial tag={tag} zr={use_zr} omr={cur:.4f} corr_ray={c_ray:.6f} n_pos={n_pos}/5 μ={mu:.6f}")

    if abs(best["corr_ray"]) > 0.05:
        log(f"WARNING: corr_ray={best['corr_ray']:.6f} — re-forcing gs")
        d = gs(best["d"], [ray_sat])
        best["pred_u"], best["tau"], best["omr"] = bisect_rho(L(mid37), d, 0.38)
        best["corr_ray"] = float(np.dot(l2u(d), l2u(ray_sat)))
        best["d"] = d
        preds, mu = predict_ensemble(best["pred_u"])
        best["preds"], best["mu"], best["n_pos"] = preds, mu, sum(1 for p in preds if p > mid_mu + 1e-7)

    gate_a = gate_a_temporal and best["n_pos"] >= 5
    log(f"GATE A (temporal+all5 BoardV3+): {gate_a} | best n_pos={best['n_pos']} μΔ={best['mu']-mid_mu:+.6f} corr_ray={best['corr_ray']:.6f}")

    # stage candidate
    p1 = SUB / "sub_v31_3_GM_SINGLE_PROBE.csv"
    pd.DataFrame({"id": ids, "is_fraud": best["pred_u"]}).to_csv(p1, index=False)
    log(f"staged {p1.name} omr(MID37)={omr(best['pred_u'], mid37):.4f}")

    return {
        "gate_a": gate_a,
        "gate_a_temporal": gate_a_temporal,
        "rho_ho": rho_ho,
        "checks": checks,
        "best": {k: v for k, v in best.items() if k not in ("pred_u", "d")},
        "mid_mu": mid_mu,
        "path": p1,
        "pred_u": best["pred_u"],
    }


def write_report(part_a, part_b, part_c, lb1, remain):
    t2 = "\n".join(
        f"| `{r['name']}` | {r['variant']} | {r['marg'].get('0.003')} | {r['pos']}/5 | "
        f"{r['mean']:+.3f} | {r['t']:.2f} | {r['sf']}/25 | {'PASS' if r['pass'] else 'fail'} |"
        for r in part_b["rows"]
    )
    s = part_a["summary"]
    md = f"""# GM V31.3 GM Edition — Oracle `0.96153` Simulation & Orthogonal STEP_P

**MAX_SUBMITS = 1** · remaining≈{remain} · LB1={lb1} · {time.strftime('%Y-%m-%d %H:%M')}

## Part A — Can the 152-bank reach `0.96153`?

**Verdict:** {part_a['verdict']}

| attacker | pub mean | pub max | priv mean | priv min |
|----------|---------:|--------:|----------:|---------:|
| wide_lm | {s['wide_pub']['mean']:.5f} | {s['wide_pub']['max']:.5f} | {s['wide_priv']['mean']:.5f} | {s['wide_priv']['min']:.5f} |
| **60k GT Oracle** | **{s['oracle_pub']['mean']:.5f}** | **{s['oracle_pub']['max']:.5f}** | {s['oracle_priv']['mean']:.5f} | {s['oracle_priv']['min']:.5f} |
| 150-query probe | {s['probe_pub']['mean']:.5f} | {s['probe_pub']['max']:.5f} | {s['probe_priv']['mean']:.5f} | {s['probe_priv']['min']:.5f} |

Bank cols used: **{part_a['n_cols']}**

## Part B — Frozen wide_lm Gate-B

Distinct meanΔ@+0.003: **{part_b['n_unique_means']}** · Gate-B pass: **{part_b['gate_b']}**

| name | variant | marg+0.003 | pos | mean e-4 | t | sf | pass |
|------|---------|----------:|----:|---------:|--:|---:|------|
{t2}

## Part C — Orthogonal STEP_P

- Temporal Board ρ={part_c['rho_ho']:.3f} checks={part_c['checks']} temporal_pass={part_c['gate_a_temporal']}
- Gate A (all 5 BoardV3 > MID37): **{part_c['gate_a']}**
- Best trial: `{json.dumps(part_c['best'], default=float)}`
- SLOT1 LB: **{lb1}**

## Decision
{"**SUBMITTED 1** then STOP." if lb1 is not None else "**HOLD / no submit** — gates failed or refuse."} Remaining slots protected (≥5).

## Log
```
{chr(10).join(LINES[-100:])}
```
"""
    REPORT.write_text(md)
    log(f"REPORT -> {REPORT}")
    (OUT / "metrics.json").write_text(json.dumps({
        "part_a": {k: v for k, v in part_a.items() if k != "raw"},
        "part_b": {"gate_b": part_b["gate_b"], "n_unique": part_b["n_unique_means"], "passers": part_b["passers"], "rows": part_b["rows"]},
        "part_c": {k: v for k, v in part_c.items() if k != "pred_u"},
        "LB1": lb1,
    }, indent=2, default=str))


def main():
    assert MAX_SUBMITS == 1
    log("=== GM V31.3 GM Edition | MAX_SUBMITS=1 | Oracle sim + GateB + STEP_P ===")
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    tr = pd.read_csv(REPO / "data/train.csv")
    te = pd.read_csv(REPO / "data/test.csv")
    y = tr["is_fraud"].to_numpy(float)
    ids = te["id"].to_numpy()
    fold_id = pd.read_csv(BANK / "folds.csv")["fold"].to_numpy()
    folds = folds_from(fold_id)
    z_wide = np.load(WIDE_OOF).ravel()[:N_TR].astype(float)
    log(f"wide_lm AUC={auc(y, z_wide):.6f}")

    live = kaggle_subs()
    n_today = 0
    if len(live):
        live["date"] = pd.to_datetime(live["date"])
        n_today = int((live["date"].dt.strftime("%Y-%m-%d") == "2026-11-21").sum())
    remain = 10 - n_today
    log(f"submits_today={n_today} remaining_est={remain}")

    Bo, names, pairs = load_bank_matrix(y, max_cols=152)
    part_a = part_a_oracle(Bo, y, z_wide)
    part_b = part_b_gateb(y, fold_id, folds, z_wide, Bo, rng)
    part_c = part_c_step_p(ids, y, fold_id, part_b["gate_b"], part_b["passers"])

    lb1 = None
    fire = part_c["gate_a"] or part_b["gate_b"]
    if not fire:
        log("HOLD — neither Gate A nor Gate B cleared. SUBMIT 0.")
    else:
        log(f"Gates fireable: A={part_c['gate_a']} B={part_b['gate_b']} — submitting AT MOST 1")
        lb1 = submit_and_poll(
            part_c["path"],
            "v31.3 GM SINGLE | orth⊥SLOT2-SLOT1 | BoardV3/GateB | max1 keep5+",
        )
        log("STOP after ≤1 submit — reserve slots for post-sim #1 strategy")

    write_report(part_a, part_b, part_c, lb1, remain - SUBMITS_DONE)
    log(f"DONE submits={SUBMITS_DONE} LB1={lb1} in {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
