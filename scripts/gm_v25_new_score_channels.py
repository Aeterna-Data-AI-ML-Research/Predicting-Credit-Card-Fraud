#!/usr/bin/env python3
"""V25 — Research NEW score channels beyond dual_GB+Δmlp_v15 / Ray-1 / natives.

Channels under test (all vs ATH 0.95440):
  C1  Alex-fresh multi-family (40-model bank updated 2026-11-17)
  C2  Mathsonchain 17-single external bank
  C3  Near-peer disagreement / triple-average of public ATH-band files
  C4  Cross-family composites (tree / digit+LM / NN) into ATH at small w

Gates: nested H2H vs ATH bank-surrogate + vs gb_bag5 raw blend; ρ(ATH) geometry.
Quota: stage always; submit only if remaining > 0.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
OUT = Path("/tmp/s6e11_v16/out")
DOC = REPO / "docs"
PIN = Path("/tmp/s6e11_nb_research/sri_pinned")
ALEX = Path("/tmp/s6e11_nb_research/alex_fresh")
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
MATH = REPO / "external/banks/mathsonchain_s6e11-fraud-oof-test-17-single-models"
ATH = SUB / "sub_v24_SRI440_pub.csv"
GPW = Path("/tmp/s6e11_nb_research/gpw_95432/out")
BHU = Path("/tmp/s6e11_nb_research/bhuvan_95426")
PY = "/tmp/s6e11_v23_venv/bin/python"
KAG = [PY, "-m", "kaggle"]
REPORT = LOG / "GM_V25_NEW_SCORE_CHANNELS_REPORT.md"
DOC_REPORT = DOC / "V25_NEW_SCORE_CHANNELS.md"
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


def rblend(parts, w):
    z = np.zeros_like(rf(parts[0]))
    for p, wi in zip(parts, w):
        z += wi * rf(p)
    return z


def load_folds_y_ids():
    te = pd.read_csv(SUB / "sub_v24_SRI440_pub.csv")
    ids = te["id"].to_numpy()
    train = pd.read_csv(REPO / "data/train.csv")
    y = train["is_fraud"].to_numpy(float)
    fold_id = pd.read_csv(BANK / "folds.csv")["fold"].to_numpy()
    folds = [(np.where(fold_id != i)[0], np.where(fold_id == i)[0]) for i in range(5)]
    assert len(y) == N_TR == len(fold_id)
    return y, ids, folds


def h2h(a, b, y, folds):
    ds = np.array([(auc(y[te], a[te]) - auc(y[te], b[te])) * 1e5 for _, te in folds])
    return dict(
        mean=float(ds.mean()),
        min=float(ds.min()),
        pos=int((ds > 0).sum()),
        t=float(ds.mean() / (ds.std(ddof=1) + 1e-12) * np.sqrt(len(ds))),
        ds=ds.tolist(),
    )


def gs_orth(d, base):
    d, base = unit(d), unit(base)
    return unit(d - np.dot(d, base) * base)


def load_csv_vec(path, ids):
    df = pd.read_csv(path)
    col = [c for c in df.columns if c != "id"][0]
    return df.set_index("id").loc[ids, col].to_numpy(float)


def alex_pairs():
    man = json.loads((ALEX / "oof_manifest.json").read_text())
    out = []
    for k, meta in man.items():
        op, tp = ALEX / f"oof_{k}.npy", ALEX / f"test_{k}.npy"
        if op.exists() and tp.exists():
            out.append((k, meta, np.load(op).ravel()[:N_TR], np.load(tp).ravel()[:N_TE]))
    return out


def math_pairs():
    out = []
    if not MATH.exists():
        return out
    for op in sorted(MATH.glob("oof_*.npy")):
        tag = op.name[len("oof_") : -4]
        tp = MATH / f"test_{tag}.npy"
        if tp.exists():
            out.append((tag, np.load(op).ravel()[:N_TR], np.load(tp).ravel()[:N_TE]))
    return out


def family_rankmean(pairs, keys):
    oo, tt = [], []
    for k, meta, o, t in pairs:
        if k in keys:
            oo.append(o)
            tt.append(t)
    if not oo:
        return None, None
    w = tuple([1 / len(oo)] * len(oo))
    return rblend(oo, w), rblend(tt, w)


def gate_member(name, oo, tt, z_ath, gb, y, folds, ath, gb_t=None):
    te0 = folds[0][1]
    d = gs_orth(L(oo), z_ath)
    sgn = +1 if auc(y[te0], z_ath[te0] + 0.05 * d[te0]) >= auc(y[te0], z_ath[te0] - 0.05 * d[te0]) else -1
    per_ath = h2h(z_ath + 0.05 * sgn * d, z_ath, y, folds)
    per_gb = h2h(rblend([gb, oo], [0.92, 0.08]), gb, y, folds)
    # raw small-w into ATH surrogate (no orth) — closer to board inject
    per_raw = h2h(rblend([rf(z_ath), oo], [0.95, 0.05]), rf(z_ath), y, folds)
    return {
        "name": name,
        "solo": auc(y, oo),
        "ath_orth": per_ath,
        "gb8": per_gb,
        "raw5": per_raw,
        "sgn": sgn,
        "rho_ath": omr(tt, ath),
        "rho_gb": omr(tt, gb_t) if (tt is not None and gb_t is not None) else None,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    log("=== V25 NEW SCORE CHANNELS ===")

    y, ids, folds = load_folds_y_ids()
    ath = load_csv_vec(ATH, ids)
    log(f"y fraud={y.mean():.5f} ATH loaded ρ self=0")

    # bank ATH surrogate (same recipe as V24.3)
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
    lr = Ridge(alpha=10.0).fit(Bt_m, L(ath))
    z_ath = Bo_m @ lr.coef_ + lr.intercept_
    gb = np.load(BANK / "oof_gb_bag5.npy").ravel()[:N_TR]
    gb_t = np.load(BANK / "test_gb_bag5.npy").ravel()[:N_TE]
    log(f"ATH surrogate AUC={auc(y,z_ath):.6f} gb_bag5={auc(y,gb):.6f}")

    # -------------------- C1 Alex fresh --------------------
    log("C1 loading alex_fresh 40-model bank…")
    alex = alex_pairs()
    log(f"C1 pairs={len(alex)}")
    alex_gates = []
    for k, meta, o, t in alex:
        g = gate_member(f"alex:{meta.get('name','?')}|{k[:8]}", o, t, z_ath, gb, y, folds, ath, gb_t)
        g["key"] = k
        g["meta"] = meta
        alex_gates.append(g)

    # rank by raw5 then gb8 then solo
    def score_g(g):
        return (
            g["raw5"]["pos"],
            g["raw5"]["mean"],
            g["gb8"]["pos"],
            g["gb8"]["mean"],
            g["solo"],
        )

    alex_gates.sort(key=score_g, reverse=True)
    log("C1 top-12 by raw5/gb8:")
    for g in alex_gates[:12]:
        log(
            f"  {g['solo']:.5f} raw5 {g['raw5']['pos']}/5 t={g['raw5']['t']:.2f} m={g['raw5']['mean']:+.2f} | "
            f"gb8 {g['gb8']['pos']}/5 m={g['gb8']['mean']:+.2f} | ath-orth {g['ath_orth']['pos']}/5 | "
            f"ρATH={g['rho_ath']:.1f} | {g['name']}"
        )

    # family composites
    man = {k: m for k, m, _, _ in [(a[0], a[1], a[2], a[3]) for a in alex]}
    # curated families by name keywords
    def keys_matching(*substrs):
        ks = []
        for k, meta, _, _ in alex:
            nm = (meta.get("name") or "").lower()
            if any(s in nm for s in substrs):
                ks.append(k)
        return ks

    fam = {
        "alex_part5_lgb": keys_matching("part 5"),
        "alex_digit": keys_matching("digit"),
        "alex_nn": keys_matching("realmlp", "tabm"),
        "alex_joint_glm": keys_matching("joint glm", "boosted from the joint"),
        "alex_depth1": keys_matching("depth 1"),
        "alex_cat": keys_matching("catboost"),
        "alex_top5_solo": [g["key"] for g in sorted(alex_gates, key=lambda x: -x["solo"])[:5]],
    }
    # unique digit+GPT-2
    fam["alex_digit_gpt2"] = keys_matching("gpt-2")

    channel_oof_test = {}
    fam_gates = []
    for fname, ks in fam.items():
        oo, tt = family_rankmean(alex, set(ks))
        if oo is None:
            log(f"C1 family {fname}: empty")
            continue
        channel_oof_test[fname] = (oo, tt)
        g = gate_member(fname, oo, tt, z_ath, gb, y, folds, ath, gb_t)
        g["keys"] = ks
        fam_gates.append(g)
        log(
            f"C1 FAM {fname}: n={len(ks)} solo={g['solo']:.5f} raw5 {g['raw5']['pos']}/5 m={g['raw5']['mean']:+.2f} "
            f"gb8 {g['gb8']['pos']}/5 m={g['gb8']['mean']:+.2f} ρATH={g['rho_ath']:.1f}"
        )

    # -------------------- C2 Mathson --------------------
    log("C2 mathsonchain 17-single…")
    math = math_pairs()
    log(f"C2 pairs={len(math)}")
    math_gates = []
    math_oo, math_tt = [], []
    for tag, o, t in math:
        g = gate_member(f"math:{tag}", o, t, z_ath, gb, y, folds, ath, gb_t)
        g["key"] = tag
        math_gates.append(g)
        math_oo.append(o)
        math_tt.append(t)
    math_gates.sort(key=score_g, reverse=True)
    for g in math_gates[:8]:
        log(
            f"  {g['solo']:.5f} raw5 {g['raw5']['pos']}/5 m={g['raw5']['mean']:+.2f} | "
            f"gb8 {g['gb8']['pos']}/5 | {g['name']}"
        )
    if math_oo:
        oo = rblend(math_oo, tuple([1 / len(math_oo)] * len(math_oo)))
        tt = rblend(math_tt, tuple([1 / len(math_tt)] * len(math_tt)))
        channel_oof_test["math_rankmean17"] = (oo, tt)
        g = gate_member("math_rankmean17", oo, tt, z_ath, gb, y, folds, ath, gb_t)
        fam_gates.append(g)
        log(
            f"C2 FAM math_rankmean17 solo={g['solo']:.5f} raw5 {g['raw5']['pos']}/5 m={g['raw5']['mean']:+.2f} "
            f"gb8 {g['gb8']['pos']}/5 ρATH={g['rho_ath']:.1f}"
        )

    # -------------------- C3 peer disagreement / triple --------------------
    log("C3 peer disagreement + triple…")
    peers = {}
    for tag, path in (
        ("sri440", ATH),
        ("gpw32", next(GPW.glob("*.csv"), None) if GPW.exists() else None),
        ("bhuvan", next(BHU.glob("*.csv"), None) if BHU.exists() else None),
        ("dual_r2_gpw", SUB / "sub_v24_DUAL_R2_GPW32_dmlp30.csv"),
        ("r2", SUB / "sub_v22_3_LIVE_ridge_continuation.csv"),
    ):
        if path is None:
            # search common names
            continue
        if Path(path).exists():
            peers[tag] = load_csv_vec(path, ids)
            log(f"  peer {tag} ρATH={omr(peers[tag], ath):.3f}e-4")
    # also pin references
    for tag, p in (
        ("step11", PIN / "reference_step11.csv"),
        ("bhuvan_pin", PIN / "reference_bhuvan.csv"),
        ("n15", PIN / "new_neural_v15.csv"),
        ("n_old", PIN / "old_neural.csv"),
    ):
        if p.exists():
            peers[tag] = load_csv_vec(p, ids)
            log(f"  peer {tag} ρATH={omr(peers[tag], ath):.3f}e-4")

    # find gpw/bhuvan if missed
    if "gpw32" not in peers:
        for p in Path("/tmp/s6e11_nb_research/gpw_95432").rglob("*.csv"):
            peers["gpw32"] = load_csv_vec(p, ids)
            log(f"  peer gpw32 ← {p} ρATH={omr(peers['gpw32'], ath):.3f}e-4")
            break
    if "bhuvan" not in peers:
        for p in Path("/tmp/s6e11_nb_research/bhuvan_95426").rglob("*.csv"):
            peers["bhuvan"] = load_csv_vec(p, ids)
            log(f"  peer bhuvan ← {p} ρATH={omr(peers['bhuvan'], ath):.3f}e-4")
            break

    test_channels = {}  # tag -> test vector (rank-frac)
    # disagreement: ATH + α * unit(L(peer)-L(ATH))  [peer pull]
    for peer_name in ("gpw32", "bhuvan", "dual_r2_gpw", "step11", "n15"):
        if peer_name not in peers:
            continue
        delta = unit(L(peers[peer_name]) - L(ath))
        for a, lab in ((0.15, "015"), (0.30, "030"), (0.50, "050")):
            vec = rf(L(ath) + a * delta)
            tag = f"DIS_{peer_name}_{lab}"
            test_channels[tag] = vec
            log(f"  {tag} ρATH={omr(vec, ath):.3f}e-4")

    # anti-disagreement (away from peer)
    for peer_name in ("r2", "gpw32"):
        if peer_name not in peers:
            continue
        delta = unit(L(ath) - L(peers[peer_name]))
        vec = rf(L(ath) + 0.30 * delta)
        tag = f"ANTI_{peer_name}_030"
        test_channels[tag] = vec
        log(f"  {tag} ρATH={omr(vec, ath):.3f}e-4")

    # triple average of near-tops
    trip = [peers[k] for k in ("sri440", "gpw32", "bhuvan") if k in peers]
    if len(trip) >= 2:
        # if sri440 is ATH itself, triple of step11, bhuvan, gpw32
        trip2 = [peers[k] for k in ("step11", "bhuvan", "gpw32", "bhuvan_pin") if k in peers]
        # unique by content hash
        uniq = []
        seen = set()
        for v in trip2:
            h = hash(v.tobytes()[::1000])
            if h not in seen:
                seen.add(h)
                uniq.append(v)
        if len(uniq) >= 2:
            vec = rblend(uniq, tuple([1 / len(uniq)] * len(uniq)))
            test_channels["TRIPLE_near"] = vec
            log(f"  TRIPLE_near n={len(uniq)} ρATH={omr(vec, ath):.3f}e-4")
        # dual of top two distinct from ATH
        if "step11" in peers and "bhuvan_pin" in peers:
            # already ATH recipe bases — try step11×gpw32 dual without neural
            if "gpw32" in peers:
                vec = rblend([peers["step11"], peers["gpw32"]], (0.5, 0.5))
                test_channels["DUAL_step11_gpw32"] = vec
                log(f"  DUAL_step11_gpw32 ρATH={omr(vec, ath):.3f}e-4")
        # NEW neural channel: dual bases + OLD neural (falsify v15 necessity) already known
        # NEW: dual + average(n15, something) — try dual_GB with half n15 + digit-gpt2 residual on test only later

    # dual_GB + n15 already ATH. Try dual_GB + alex digit-gpt2 residual as NEW neural axis
    if "alex_digit_gpt2" in channel_oof_test and "step11" in peers and "bhuvan_pin" in peers and "n15" in peers:
        _, dig_t = channel_oof_test["alex_digit_gpt2"]
        # replace neural with dig_gpt2
        G, B = peers["step11"], peers["bhuvan_pin"]
        d_new = L(dig_t) - L(peers.get("n_old", peers["n15"]))
        for w in (0.22, 0.30):
            b1 = unit(L(G) + w * unit(d_new))
            b2 = unit(L(B) + w * unit(d_new))
            vec = rf(0.5 * (b1 + b2))
            tag = f"DUAL_GB_x_digitGPT2_w{w:.2f}"
            test_channels[tag] = vec
            log(f"  {tag} ρATH={omr(vec, ath):.3f}e-4")
        # blend n15 with digitGPT2 residual
        d_mix = unit(0.5 * unit(L(peers["n15"]) - L(peers.get("n_old", peers["n15"]))) + 0.5 * unit(L(dig_t) - L(peers.get("n_old", peers["n15"]))))
        for w in (0.30, 0.35):
            b1 = unit(L(G) + w * d_mix)
            b2 = unit(L(B) + w * d_mix)
            vec = rf(0.5 * (b1 + b2))
            tag = f"DUAL_GB_n15mixDigit_w{w:.2f}"
            test_channels[tag] = vec
            log(f"  {tag} ρATH={omr(vec, ath):.3f}e-4")

    # -------------------- C4 ATH × channel small-w --------------------
    log("C4 building ATH×channel injects…")
    # pick best family channels by raw5 gate
    fam_gates_sorted = sorted(fam_gates, key=score_g, reverse=True)
    for g in fam_gates_sorted:
        if g["name"] not in channel_oof_test:
            continue
        _, tt = channel_oof_test[g["name"]]
        for w, lab in ((0.05, "05"), (0.08, "08"), (0.12, "12")):
            vec = rf(0.0 * ath)  # placate linter
            vec = rf((1 - w) * L(ath) + w * L(tt))
            tag = f"ATH_x_{g['name']}_{lab}"
            test_channels[tag] = vec

    # best single alex members (top 3 by raw5 with solo>=0.948)
    best_singles = [g for g in alex_gates if g["solo"] >= 0.948][:5]
    key_to_test = {k: t for k, _, _, t in alex}
    for g in best_singles:
        tt = key_to_test[g["key"]]
        for w, lab in ((0.05, "05"), (0.08, "08")):
            tag = f"ATH_x_alex_{g['key'][:8]}_{lab}"
            test_channels[tag] = rf((1 - w) * L(ath) + w * L(tt))
        # orth inject at 0.30e-4 target — approximate with small alpha
        d = gs_orth(L(tt), L(ath))
        # scale alpha so 1-ρ ≈ 0.30e-4 (binary search)
        lo, hi = 1e-4, 0.05
        best_a, best_diff = 0.01, 1e9
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            vec = rf(L(ath) + mid * d)
            r = omr(vec, ath)
            if abs(r - 0.30) < best_diff:
                best_diff, best_a = abs(r - 0.30), mid
            if r < 0.30:
                lo = mid
            else:
                hi = mid
        tag = f"ATH_orth_alex_{g['key'][:8]}_0p30"
        test_channels[tag] = rf(L(ath) + best_a * d)
        log(f"  orth {tag} ρATH={omr(test_channels[tag], ath):.3f}e-4 (target 0.30)")

    # -------------------- stage + select --------------------
    # Prefer channels that: (a) family raw5 pos>=4, or (b) are structural dual/disagree near ATH
    priority = []
    for g in fam_gates_sorted:
        if g["raw5"]["pos"] >= 4 and g["raw5"]["mean"] > 0:
            priority.append(f"ATH_x_{g['name']}_05")
            priority.append(f"ATH_x_{g['name']}_08")
    # structural
    for tag in list(test_channels):
        if tag.startswith("DUAL_GB_") or tag.startswith("TRIPLE") or tag.startswith("DIS_"):
            priority.append(tag)
    # best alex singles
    for g in best_singles[:3]:
        priority.append(f"ATH_x_alex_{g['key'][:8]}_05")
        priority.append(f"ATH_orth_alex_{g['key'][:8]}_0p30")

    # dedupe preserve order
    seen = set()
    priority = [t for t in priority if t in test_channels and not (t in seen or seen.add(t))]

    staged = []
    for tag, vec in test_channels.items():
        path = SUB / f"sub_v25_{tag}.csv"
        pd.DataFrame({"id": ids, "is_fraud": vec}).to_csv(path, index=False)
        staged.append({"tag": tag, "path": str(path), "rho_ath": omr(vec, ath)})
    staged.sort(key=lambda s: s["rho_ath"])
    log(f"STAGED {len(staged)} CSVs")
    for s in staged[:15]:
        log(f"  {s['tag']:40} ρATH={s['rho_ath']:.3f}e-4")

    # meta dump
    meta = {
        "alex_top": [
            {
                "key": g["key"],
                "name": g["name"],
                "solo": g["solo"],
                "raw5": {k: g["raw5"][k] for k in ("mean", "pos", "t")},
                "gb8": {k: g["gb8"][k] for k in ("mean", "pos", "t")},
                "ath_orth": {k: g["ath_orth"][k] for k in ("mean", "pos", "t")},
                "rho_ath": g["rho_ath"],
            }
            for g in alex_gates[:20]
        ],
        "families": [
            {
                "name": g["name"],
                "solo": g["solo"],
                "raw5": {k: g["raw5"][k] for k in ("mean", "pos", "t")},
                "gb8": {k: g["gb8"][k] for k in ("mean", "pos", "t")},
                "rho_ath": g["rho_ath"],
                "keys": g.get("keys"),
            }
            for g in fam_gates_sorted
        ],
        "math_top": [
            {
                "name": g["name"],
                "solo": g["solo"],
                "raw5": {k: g["raw5"][k] for k in ("mean", "pos", "t")},
                "gb8": {k: g["gb8"][k] for k in ("mean", "pos", "t")},
            }
            for g in math_gates[:10]
        ],
        "priority": priority,
        "staged": staged,
        "elapsed": time.time() - T0,
    }
    (OUT / "v25_new_channels_meta.json").write_text(json.dumps(meta, indent=2))

    # optional submit first priority if quota
    live = []
    for tag in priority[:2]:
        s = next(x for x in staged if x["tag"] == tag)
        msg = f"v25.{tag} | ρ(ATH)={s['rho_ath']:.3f}e-4 | new-channel research"[
            :480
        ]
        log(f"SUBMIT try {msg}")
        try:
            r = subprocess.run(
                KAG
                + [
                    "competitions",
                    "submit",
                    "-c",
                    "playground-series-s6e11",
                    "-f",
                    s["path"],
                    "-m",
                    msg,
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            out = (r.stdout or "") + (r.stderr or "")
            log(out[-500:].replace("\n", " | "))
            status = "ok" if r.returncode == 0 and "error" not in out.lower() and "quota" not in out.lower() else "quota_or_error"
            if "limit" in out.lower() or "quota" in out.lower() or "403" in out or "429" in out:
                status = "quota_or_error"
            live.append({"tag": tag, "status": status})
            if status != "ok":
                log("STOP quota — remaining staged")
                break
        except Exception as e:
            log(f"submit fail {e}")
            live.append({"tag": tag, "status": "error"})
            break

    meta["live"] = live
    (OUT / "v25_new_channels_meta.json").write_text(json.dumps(meta, indent=2))

    # reports
    lines = [
        "# GM V25 New Score Channels",
        "",
        f"ATH **0.95440** · elapsed {time.time()-T0:.0f}s",
        "",
        "## Channel map",
        "",
        "| ID | Channel | Idea |",
        "|----|---------|------|",
        "| C1 | Alex-fresh 40-model | New public OOF bank (Nov17); family rank-means + best singles |",
        "| C2 | Mathsonchain 17 | Alternate external singles bank |",
        "| C3 | Peer disagreement / triple | Pull/push vs gpw/bhuvan/step11; dual without v15; digitGPT2 neural swap |",
        "| C4 | ATH × channel | Small-w / micro-orth injects of gated channels |",
        "",
        "## C1 Alex top singles (raw5 gate)",
        "",
        "| solo | raw5 | gb8 | ath-orth | ρATH e-4 | model |",
        "|-----:|------|-----|----------|--------:|-------|",
    ]
    for g in alex_gates[:12]:
        lines.append(
            f"| {g['solo']:.5f} | {g['raw5']['pos']}/5 {g['raw5']['mean']:+.2f} | "
            f"{g['gb8']['pos']}/5 {g['gb8']['mean']:+.2f} | {g['ath_orth']['pos']}/5 | "
            f"{g['rho_ath']:.1f} | {g['name']} |"
        )
    lines += ["", "## Family / bank composites", "", "| name | solo | raw5 | gb8 | ρATH |", "|------|-----:|------|-----|-----:|"]
    for g in fam_gates_sorted:
        lines.append(
            f"| `{g['name']}` | {g['solo']:.5f} | {g['raw5']['pos']}/5 {g['raw5']['mean']:+.2f} | "
            f"{g['gb8']['pos']}/5 {g['gb8']['mean']:+.2f} | {g['rho_ath']:.1f} |"
        )
    lines += [
        "",
        "## Submit priority (staged)",
        "",
    ]
    for i, tag in enumerate(priority[:12], 1):
        s = next(x for x in staged if x["tag"] == tag)
        lines.append(f"{i}. `{s['path'].split('/')[-1]}` — ρATH={s['rho_ath']:.3f}e-4")
    lines += [
        "",
        "## Live",
        "",
        "| tag | status |",
        "|-----|--------|",
    ]
    for Lrow in live:
        lines.append(f"| `{Lrow['tag']}` | {Lrow['status']} |")
    if not live:
        lines.append("| — | none attempted / quota |")
    lines += [
        "",
        "## Verdict rules",
        "",
        "- **VALID channel candidate:** family/single raw5 ≥4/5 and meanΔ>0, or structural dual/disagree with ρATH∈[0.15, 3.0]e-4 near ATH manifold.",
        "- **INVALID:** ath-orth 0/5 with raw5 fail; solo≪0.94; ρATH≫10e-4 unless intentional peer blend.",
        "- Exhausted (not retested as primary): Ray-1 crest, ±twilight, bank micro-GS on R2, CSV roundtrip.",
        "",
        "## Log",
        "```",
        *LINES,
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines))
    DOC_REPORT.write_text(
        "\n".join(
            [
                "# V25 New Score Channels",
                "",
                "Research for lift past ATH **0.95440** via *new channels* (not Ray-1 / twilight / native solo).",
                "",
                "See `logs/overnight/GM_V25_NEW_SCORE_CHANNELS_REPORT.md` for full gates.",
                "",
                "## What was screened",
                "",
                "1. **Alex-fresh** 40 OOF+test models (dataset update 2026-11-17).",
                "2. **Mathsonchain** 17 singles.",
                "3. **Peer disagreement / triple / dual_GB×digitGPT2** structural recipes.",
                "4. **ATH × channel** small-w and 0.30e-4 orth injects.",
                "",
                f"## Priority queue ({len(priority)} gated)",
                "",
                *[f"- `{t}`" for t in priority[:15]],
                "",
                f"Staged CSVs: `submissions/sub_v25_*.csv` ({len(staged)} files).",
                "",
            ]
        )
    )
    log(f"REPORT {REPORT}")
    log(f"DOC {DOC_REPORT}")
    log("DONE")


if __name__ == "__main__":
    main()
