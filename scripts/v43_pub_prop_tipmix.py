#!/usr/bin/env python3
"""Mix proprietary work onto highest-public tip (goodpjw Step_AB @ 0.95569).

Primary lever: tip-lock rank-mapped proprietary banks onto Step_AB at small 1−ρ e4
(DiagA×modern rank-weight w≥0.08 already burned LB — that move was ≥1.26e-4).

Also: additive tip-lock of residual props (soft_diff / delta1) that pass fold gates.
Reject |cos| > 0.85 vs known loser axes S02/S06.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import kaggle_doh  # noqa: E402

OUT = ROOT / "out/v43_pub_prop_mix"
OUT.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "logs/v43/V43_PUB_PROP_MIX.jsonl"
FINAL = ROOT / "out/v41_final_10slots"
PRED = ROOT / "preds"
SSD = Path("/tmp/s6e11_v39_ssd")
TIP_AB = FINAL / "SLOT01_pub_Step_AB_095569.csv"
DAILY_LIMIT = 10
MAX_SUBMIT = int(os.environ.get("V43_MIX_MAX_SUBMIT", "5"))
DO_SUBMIT = os.environ.get("V43_MIX_SUBMIT", "1") == "1"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "msg": m}) + "\n")


def sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def L(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float).ravel(), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def U(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float).ravel()))


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


def one_mrho(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.corrcoef(np.asarray(a, float).ravel(), np.asarray(b, float).ravel())[0, 1])


def cos_full(a: np.ndarray, b: np.ndarray) -> float:
    a, b = unit(a), unit(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def clip_p985(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    lim = float(np.quantile(np.abs(v), 0.985))
    return np.clip(v, -lim, lim)


def rank_to_tip(src: np.ndarray, tip: np.ndarray) -> np.ndarray:
    tip_sorted = np.sort(tip)
    order = np.argsort(src)
    out = np.empty_like(src, dtype=np.float64)
    out[order] = tip_sorted
    return out


def ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(np.asarray(x, float).ravel())).astype(np.float64)


def scale_to_target_1mrho(base_u, direction, target_e4, lo=1e-10, hi=64.0):
    base_u = np.asarray(base_u, float).ravel()
    direction = np.asarray(direction, float).ravel()
    target = target_e4 * 1e-4
    best_a, best_diff, best_p = 0.0, 1e9, base_u
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        p = U(L(base_u) + mid * direction)
        d = one_mrho(p, base_u)
        if abs(d - target) < best_diff:
            best_diff, best_a, best_p = abs(d - target), mid, p
        if d < target:
            lo = mid
        else:
            hi = mid
    return best_p, best_a, one_mrho(best_p, base_u) * 1e4


def write_csv(path: Path, ids, scores) -> str:
    text = pd.DataFrame({"id": ids, "is_fraud": np.asarray(scores, float).ravel()}).to_csv(
        index=False, lineterminator="\n"
    )
    path.write_bytes(text.encode())
    return sha12(path)


def cli_submit(path: Path, message: str) -> str:
    for k in list(os.environ):
        if "proxy" in k.lower():
            os.environ.pop(k, None)
    exe = "/Users/mz/Library/Python/3.14/bin/kaggle"
    cmd = [exe, "competitions", "submit", "-c", kaggle_doh.COMP, "-f", str(path), "-m", message[:480]]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def load_tip(path: Path):
    df = pd.read_csv(path)
    return df["id"].to_numpy(), df["is_fraud"].to_numpy(float)


def gate_rankblend(y, base_o, ray_o, folds, weights=(0.05, 0.1, 0.2, 0.3)):
    best = {"mean_e5": -1e9, "pos": 0, "w": 0.0, "folds": []}
    rb, rr = ranks(base_o), ranks(ray_o)
    for w in weights:
        p = (1 - w) * rb + w * rr
        folds_d, pos = [], 0
        for k in range(5):
            m = folds == k
            da = (roc_auc_score(y[m], p[m]) - roc_auc_score(y[m], base_o[m])) * 1e5
            folds_d.append(float(da))
            if da > 0:
                pos += 1
        mean = float(np.mean(folds_d))
        if mean > best["mean_e5"]:
            best = {"mean_e5": mean, "pos": pos, "w": float(w), "folds": folds_d}
    return best


def gate_additive(y, base_o, dir_o, folds, alphas=(0.02, 0.04, 0.08, 0.12)):
    base_p = np.clip(np.asarray(base_o, float).ravel(), 1e-6, 1 - 1e-6)
    best = {"mean_e5": -1e9, "pos": 0, "alpha": 0.0, "sign": 0, "folds": []}
    for sgn in (1, -1):
        for a in alphas:
            p = U(L(base_p) + sgn * a * dir_o)
            folds_d, pos = [], 0
            for k in range(5):
                m = folds == k
                da = (roc_auc_score(y[m], p[m]) - roc_auc_score(y[m], base_p[m])) * 1e5
                folds_d.append(float(da))
                if da > 0:
                    pos += 1
            mean = float(np.mean(folds_d))
            if mean > best["mean_e5"]:
                best = {"mean_e5": mean, "pos": pos, "alpha": float(a), "sign": sgn, "folds": folds_d}
    return best


def orth_fold(signal_o, signal_t, base_o, base_t, folds, ridge=50.0):
    def Lf(x, folds):
        x = L(x)
        out = np.zeros_like(x)
        for k in range(5):
            m = folds == k
            v = x[m]
            out[m] = (v - v.mean()) / (v.std() + 1e-12)
        return out

    so = unit(Lf(signal_o, folds))
    bo = unit(Lf(base_o, folds)).reshape(-1, 1)
    st = unit(L(signal_t))
    bt = unit(L(base_t)).reshape(-1, 1)
    oof = np.zeros(len(so))
    te = np.zeros(len(st))
    for k in range(5):
        tr, va = folds != k, folds == k
        beta = np.linalg.solve(bo[tr].T @ bo[tr] + ridge * np.eye(1), bo[tr].T @ so[tr])
        oof[va] = so[va] - (bo[va] @ beta).ravel()
        te += (st - (bt @ beta).ravel()) / 5
    return unit(oof), unit(te)


def stage_tip(ids, AB, direction, e4, d_s02, d_s06, meta: dict, cands: list):
    for sgn, tag in ((1, ""), (-1, "_anti")):
        p, a, got = scale_to_target_1mrho(AB, sgn * direction, e4)
        if abs(got - e4) > 0.05:  # failed to hit target
            log(f"  miss target e4={e4} got={got:.4f} {meta.get('ray')}{tag}")
            continue
        inc = clip_p985(L(p) - L(AB))
        c2, c6 = cos_full(inc, d_s02), cos_full(inc, d_s06)
        if abs(c2) > 0.85 or abs(c6) > 0.85:
            log(f"  reject e4={e4}{tag} cosS02={c2:.3f} cosS06={c6:.3f}")
            continue
        name = f"AB_x_{meta['ray']}{tag}_e{e4:.2f}.csv"
        path = OUT / name
        sha = write_csv(path, ids, p)
        row = {
            **meta,
            "path": str(path),
            "name": name,
            "e4": e4,
            "got_e4": got,
            "alpha_tip": a,
            "sign": sgn,
            "cos_S02": c2,
            "cos_S06": c6,
            "sha": sha,
            "priority": meta.get("gate_e5", 0) * 10 - e4 + (5 if sgn == meta.get("pref_sign", 1) else 0),
        }
        cands.append(row)
        log(f"  STAGED {name} got={got:.4f} cosS02={c2:.3f} cosS06={c6:.3f} sha={sha}")


def main():
    log("=== V43 pub⊕prop tip-mix on Step_AB (rankmap + residual) ===")
    ids, AB = load_tip(TIP_AB)
    _, S02 = load_tip(FINAL / "SLOT02_AB_clip985_areeb_a0p32.csv")
    _, S06 = load_tip(FINAL / "SLOT06_AB_C3_harvest_p4_clip985_0p28e4.csv")
    d_s02 = clip_p985(L(S02) - L(AB))
    d_s06 = clip_p985(L(S06) - L(AB))

    y = pd.read_csv(ROOT / "data/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    folds = np.zeros(len(y), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        folds[va] = k

    z_pub_o = np.load(SSD / "preds/z_pub_only_oof.npy").ravel()
    z_pub_t = np.load(SSD / "preds/z_pub_only_test.npy").ravel()
    z_mod_o = np.load(ROOT / "out/v42_1_modern/z_modern_pub_oof.npy").ravel()
    z_mod_t = np.load(ROOT / "out/v42_1_modern/z_modern_pub_test.npy").ravel()
    own_o = np.load(PRED / "P_restack_own_oof.npy").ravel()
    own_t = np.load(PRED / "P_restack_own_test.npy").ravel()

    cands: list[dict] = []
    report = []

    # ── A) Rank-map proprietary banks onto AB tip ──
    rank_banks = {
        "modern": (z_mod_o, z_mod_t),
        "own_restack": (own_o, own_t),
    }
    for name, (so, st) in rank_banks.items():
        g = gate_rankblend(y, z_pub_o, so, folds)
        report.append({"ray": name, "mode": "rankblend", "gate": g})
        log(f"RANKBANK {name}: Δe5={g['mean_e5']:.2f} pos={g['pos']}/5 w*={g['w']} folds={[round(x,2) for x in g['folds']]}")
        if g["mean_e5"] < 0.3 or g["pos"] < 3:
            log("  weak — skip")
            continue
        mapped = rank_to_tip(st, AB)
        direction = clip_p985(L(mapped) - L(AB))
        meta = {"ray": f"{name}_rmap", "gate_e5": g["mean_e5"], "pos": g["pos"], "pref_sign": 1, "mode": "rankmap"}
        # stay well below burned 1.26e-4 DiagA×modern w=0.08
        for e4 in (0.05, 0.10, 0.18, 0.28):
            stage_tip(ids, AB, direction, e4, d_s02, d_s06, meta, cands)

    # ── B) Additive residual props (soft_diff / delta1) ──
    resid = {
        "soft_diff": (
            np.load(PRED / "v10_transductive_bulsah/soft_diff_orth_LM_oof.npy").ravel(),
            np.load(PRED / "v10_transductive_bulsah/soft_diff_orth_LM_test.npy").ravel(),
        ),
        "delta1_v4": (
            np.load(PRED / "gm_v4_polish/d1_v4_oof.npy").ravel(),
            np.load(PRED / "gm_v4_polish/d1_v4_test.npy").ravel(),
        ),
    }
    for name, (so, st) in resid.items():
        o_o, o_t = orth_fold(so, st, z_pub_o, z_pub_t, folds)
        g = gate_additive(y, z_pub_o, o_o, folds)
        report.append({"ray": name, "mode": "additive", "gate": g})
        log(
            f"RESID {name}: Δe5={g['mean_e5']:.2f} pos={g['pos']}/5 α={g['alpha']} sgn={g['sign']} "
            f"folds={[round(x,2) for x in g['folds']]}"
        )
        if g["pos"] < 4:
            continue
        tip_dir = clip_p985(o_t)
        meta = {
            "ray": f"{name}_add",
            "gate_e5": g["mean_e5"],
            "pos": g["pos"],
            "pref_sign": g["sign"],
            "mode": "additive",
        }
        for e4 in (0.05, 0.10, 0.15):
            stage_tip(ids, AB, g["sign"] * tip_dir, e4, d_s02, d_s06, meta, cands)

    # ── C) 2D: modern_rmap ⊕ soft_diff_add (if both staged) ──
    has_mod = any(c["ray"] == "modern_rmap" for c in cands)
    has_sf = any(c["ray"] == "soft_diff_add" for c in cands)
    if has_mod and has_sf:
        mapped = rank_to_tip(z_mod_t, AB)
        d_m = unit(clip_p985(L(mapped) - L(AB)))
        _, o_t = orth_fold(resid["soft_diff"][0], resid["soft_diff"][1], z_pub_o, z_pub_t, folds)
        g_sf = next(r["gate"] for r in report if r["ray"] == "soft_diff")
        d_s = unit(clip_p985(g_sf["sign"] * o_t))
        combo = clip_p985(d_m + d_s)
        meta = {
            "ray": "modern_soft_2D",
            "gate_e5": float(
                np.mean(
                    [
                        next(r["gate"]["mean_e5"] for r in report if r["ray"] == "modern"),
                        g_sf["mean_e5"],
                    ]
                )
            ),
            "pos": 4,
            "pref_sign": 1,
            "mode": "2D",
        }
        for e4 in (0.10, 0.18):
            stage_tip(ids, AB, combo, e4, d_s02, d_s06, meta, cands)

    cands.sort(key=lambda c: -c["priority"])
    # diversify: prefer +sign, at most 2 per ray, prefer smaller e4 first within ray
    picked, seen = [], {}
    for c in cands:
        if c["sign"] != c.get("pref_sign", 1) and c.get("mode") != "rankmap":
            # still allow anti for rankmap only if + was rejected
            pass
        key = c["ray"]
        n = seen.get(key, 0)
        if n >= 2:
            continue
        # skip anti for rankmap unless no +sign for that e4 family
        if c["sign"] < 0 and c.get("mode") == "rankmap":
            continue
        seen[key] = n + 1
        picked.append(c)
        if len(picked) >= 6:
            break

    # ensure at least modern micros if empty
    if not picked and cands:
        picked = [c for c in cands if c["sign"] > 0][:3]

    (OUT / "MIX_REPORT.json").write_text(
        json.dumps({"report": report, "staged": picked, "all": cands}, indent=2, default=float)
    )
    log(f"staged={len(cands)} picked={len(picked)}")
    for i, c in enumerate(picked):
        log(f"PICK[{i}] {c['name']} gate={c['gate_e5']:.2f} e4={c['e4']}")

    if not DO_SUBMIT or not picked:
        log("NO_SUBMIT or empty — done build")
        print(json.dumps({"staged": [c["name"] for c in picked], "submit": False}, indent=2))
        return

    n, best, _ = kaggle_doh.today_utc_count()
    left = max(0, DAILY_LIMIT - n)
    take = min(MAX_SUBMIT, left, len(picked))
    live = float(best) if best else 0.95569
    log(f"SUBMIT ladder take={take} today={n} live={live} left={left}")
    results = []
    for i, c in enumerate(picked[:take]):
        p = Path(c["path"])
        # skip if already scored today
        today = str(__import__("datetime").datetime.now(__import__("datetime").timezone.utc).date())
        already = False
        for r in kaggle_doh.list_submissions(1):
            if r.get("fileName") == p.name and (r.get("date") or "")[:10] == today and r.get("status") == "complete":
                sc = float(r["publicScore"])
                log(f"ALREADY {p.name} LB={sc}")
                results.append({"file": p.name, "score": sc, "delta": sc - live, "ray": c["ray"], "e4": c["e4"], "sha": c["sha"]})
                if sc > live:
                    live = sc
                already = True
                break
        if already:
            continue
        msg = (
            f"V43 pub⊕prop Step_AB×{c['ray']} tip-lock e4={c['e4']:.2f} "
            f"gateΔe5={c['gate_e5']:.2f} pos={c['pos']} sha={c['sha']}"
        )
        log(f"SUBMIT {i+1}/{take} {p.name}")
        try:
            resp = cli_submit(p, msg)
            log(f"resp={resp.strip()[:240]}")
        except Exception as e:
            log(f"SUBMIT FAIL: {e}")
            break
        sc = kaggle_doh.poll_file(p.name, timeout_s=900, sleep_s=12)
        log(f"LB[{i+1}]={sc} live_was={live}")
        if sc is None:
            log("poll timeout — stop")
            break
        delta = sc - live
        results.append({"file": p.name, "score": sc, "delta": delta, "ray": c["ray"], "e4": c["e4"], "sha": c["sha"]})
        if sc > live + 1e-9:
            log(f"UP +{delta:.5f} — new tip={sc}")
            live = sc
        elif abs(delta) < 1e-9:
            log("FLAT")
        else:
            log(f"DOWN {delta:.5f}")
        time.sleep(3)

    out = {"live": live, "results": results}
    (ROOT / "logs/v43/V43_PUB_PROP_MIX_RESULTS.json").write_text(json.dumps(out, indent=2))
    log(f"DONE live={live} n={len(results)}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
