#!/usr/bin/env python3
"""V24 — Different levers: sriram 0.95440 recipe, Δmlp v15, goodpjw 0.95432, peer blends.

Exhausted (do not rehash): Ray-1 crest, ±twilight orth, GPW/sameer/alex micro-orth, anti-K2.
"""
from __future__ import annotations

import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[1]
SUB = REPO / "submissions"
LOG = REPO / "logs/overnight"
OUT = Path("/tmp/s6e11_v16/out")
PIN = Path("/tmp/s6e11_nb_research/sri_pinned")
GPW = Path("/tmp/s6e11_nb_research/gpw_95432/out")
SRI_OUT = Path("/tmp/s6e11_nb_research/sri_95440/out")
P0 = Path("/tmp/s6e11_p0_blend/in")
R2P = SUB / "sub_v22_3_LIVE_ridge_continuation.csv"
PY = "/tmp/s6e11_v23_venv/bin/python"
KAG = [PY, "-m", "kaggle"]
REPORT = LOG / "GM_V24_NEW_LEVERS_REPORT.md"
N_TE = 299984
T0 = time.time()
LINES: list[str] = []


def log(msg: str) -> None:
    s = f"[{time.time()-T0:7.1f}s] {msg}"
    print(s, flush=True)
    LINES.append(s)


def L(v):
    v = np.asarray(v, float).ravel()
    return logit((rankdata(v, method="average") - 0.5) / len(v))


def rf(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def omr(a, b):
    return (1.0 - float(spearmanr(a, b).correlation)) * 1e4


def csv_roundtrip(ids, vec):
    """Match sriram: write/read CSV so float32 path matches public file."""
    tmp = OUT / "_v24_roundtrip.csv"
    pd.DataFrame({"id": ids, "is_fraud": np.asarray(vec, float).ravel()}).to_csv(tmp, index=False)
    return pd.read_csv(tmp).set_index("id").loc[ids, "is_fraud"].to_numpy(float)


def reconstruct_sri95440(ids):
    """U((CSV_roundtrip(U(L(G)+.30*(L(N)-L(C)))) + CSV_roundtrip(U(L(B)+.30*(L(N)-L(C)))))/2)"""
    G = pd.read_csv(PIN / "reference_step11.csv").set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    B = pd.read_csv(PIN / "reference_bhuvan.csv").set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    N = pd.read_csv(PIN / "new_neural_v15.csv").set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    C = pd.read_csv(PIN / "old_neural.csv").set_index("id").loc[ids, "is_fraud"].to_numpy(float)
    dN = L(N) - L(C)
    branch_g = unit(L(G) + 0.30 * dN)
    branch_b = unit(L(B) + 0.30 * dN)
    g_rt = csv_roundtrip(ids, branch_g)
    b_rt = csv_roundtrip(ids, branch_b)
    return unit((unit(g_rt) + unit(b_rt)) / 2.0)


def blend(a, b, w_b):
    return rf((1 - w_b) * L(a) + w_b * L(b))


def bisect_toward(base, peer, target_rho_vs_base, n=28):
    """Find w in blend(base, peer, w) with 1-ρ(cand, base) ≈ target."""
    a, b = 1e-4, 0.5
    best_w, best_c, best_err = 0.0, base.copy(), 1e9
    for _ in range(n):
        mid = 0.5 * (a + b)
        cand = blend(base, peer, mid)
        rho = omr(cand, base)
        err = abs(rho - target_rho_vs_base)
        if err < best_err:
            best_err, best_w, best_c = err, mid, cand
        if rho < target_rho_vs_base:
            a = mid
        else:
            b = mid
    return best_w, best_c, omr(best_c, base)


def submit_and_poll(path: Path, msg: str):
    try:
        r = subprocess.run(
            KAG + ["competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg],
            capture_output=True,
            text=True,
            timeout=180,
        )
        out = r.stdout + "\n" + r.stderr
        log(r.stdout.strip()[:400])
        log(r.stderr.strip()[:350])
        low = out.lower()
        if r.returncode != 0 or "limit" in low or ("400" in out and "successfully" not in low):
            return False, None, "quota_or_error"
        for i in range(45):
            time.sleep(18 if i else 8)
            p = subprocess.run(
                KAG + ["competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
                capture_output=True,
                text=True,
                timeout=90,
            )
            for ln in p.stdout.strip().splitlines()[1:20]:
                if path.name not in ln:
                    continue
                log(f"poll {ln[:240]}")
                if "COMPLETE" in ln.upper():
                    for tok in reversed([c.strip() for c in ln.split(",")]):
                        try:
                            return True, float(tok), "ok"
                        except ValueError:
                            continue
        return True, None, "timeout"
    except Exception as e:
        log(f"submit error: {e}")
        traceback.print_exc()
        return False, None, "exception"


def stage(path: Path, ids, vec):
    v = np.asarray(vec, float).ravel()
    assert len(v) == N_TE and np.isfinite(v).all()
    pd.DataFrame({"id": ids, "is_fraud": v}).to_csv(path, index=False)
    log(f"STAGED {path.name}")


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log("=== V24 NEW LEVERS (sri95440 / Δmlp_v15 / gpw95432 / peer) ===")

    sample = pd.read_csv(R2P)
    ids = sample["id"].to_numpy()
    assert len(ids) == N_TE

    def load_sub(path):
        return pd.read_csv(path).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    r2 = load_sub(R2P)
    sri_pub = load_sub(SRI_OUT / "submission.csv")
    gpw32 = load_sub(GPW / "submission.csv")
    gpw_s12 = load_sub(GPW / "submission_step12.csv") if (GPW / "submission_step12.csv").exists() else None
    s95417 = load_sub(P0 / "alex/submission.csv") if (P0 / "alex/submission.csv").exists() else None

    # old sriram construct for reference
    old_ref = load_sub(PIN / "reference.csv")
    old_new = load_sub(PIN / "new_neural.csv")
    old_old = load_sub(PIN / "old_neural.csv")
    sri_old = rf(L(old_ref) + 0.15 * (L(old_new) - L(old_old)))

    # reconstruct 0.95440
    sri440 = reconstruct_sri95440(ids)
    # map unit scores to (0,1)-ish via rank_frac for submission
    sri440_sub = rf(sri440)
    # compare to published
    log(
        f"recon vs pub sri: ρ={omr(sri440_sub, sri_pub):.4f}e-4 "
        f"max|d|={np.max(np.abs(sri440_sub - rf(sri_pub))):.6g} "
        f"spearman={float(spearmanr(sri440_sub, sri_pub).correlation):.8f}"
    )
    log(
        f"geometry vs R2: sri440={omr(sri440_sub,r2):.3f}e-4 sri_pub={omr(sri_pub,r2):.3f}e-4 "
        f"gpw32={omr(gpw32,r2):.3f}e-4 old_sri={omr(sri_old,r2):.3f}e-4"
    )
    log(
        f"logit corr R2×sri440={float(np.corrcoef(L(r2),L(sri440_sub))[0,1]):.4f} "
        f"R2×gpw32={float(np.corrcoef(L(r2),L(gpw32))[0,1]):.4f}"
    )

    # Lever A: apply Δmlp_v15 onto R2 at several weights (NEW neural version)
    N15 = load_sub(PIN / "new_neural_v15.csv")
    C = load_sub(PIN / "old_neural.csv")
    d15 = L(N15) - L(C)
    # also old Δmlp for contrast
    d_old = L(old_new) - L(old_old)
    log(f"Δmlp_v15 vs Δmlp_old corr={float(np.corrcoef(unit(d15), unit(d_old))[0,1]):.4f}")

    cands = []

    def add(tag, vec, note):
        v = np.asarray(vec, float).ravel()
        cands.append(
            {
                "tag": tag,
                "vec": v,
                "note": note,
                "rho_r2": omr(v, r2),
                "rho_sri440": omr(v, sri440_sub),
            }
        )

    # 1) exact / near sriram 0.95440 (consume new public high)
    add("SRI440_recon", sri440_sub, "reconstructed sriram 0.95440 recipe w=0.30 v15 dual-base")
    add("SRI440_pub", rf(sri_pub), "published sriram submission.csv (0.95440)")

    # 2) R2 ← toward sri440 (cross-family to new high) at controlled ρ
    for tgt, tag in [(0.30, "R2_to_SRI440_0p30e4"), (0.60, "R2_to_SRI440_0p60e4"), (1.20, "R2_to_SRI440_1p20e4")]:
        w, c, rho = bisect_toward(r2, sri440_sub, tgt)
        add(tag, c, f"blend w={w:.4f} toward sri440 | ρ={rho:.3f}e-4")

    # 3) R2 + w * Δmlp_v15 (direct new neural lever — different from old 0.15 path)
    for w in (0.08, 0.15, 0.22, 0.30):
        c = rf(L(r2) + w * d15)
        add(f"R2_dmlp_v15_w{w:.2f}", c, f"R2 + {w:.2f}*(L(N15)-L(C))")

    # 4) goodpjw 0.95432 blend / step12
    add("GPW32_pub", rf(gpw32), "goodpjw ladder-meets-stack 0.95432")
    w, c, rho = bisect_toward(r2, gpw32, 0.35)
    add("R2_to_GPW32_0p35e4", c, f"w={w:.4f} ρ={rho:.3f}")
    if gpw_s12 is not None:
        w, c, rho = bisect_toward(r2, gpw_s12, 0.35)
        add("R2_to_GPW_S12_0p35e4", c, f"w={w:.4f} ρ={rho:.3f}")

    # 5) Areeb-style: sri440 × S95417 90/10 (true near-high stack blend)
    if s95417 is not None:
        add("SRI440_x_S95417_90_10", blend(sri440_sub, s95417, 0.10), "Areeb on new high × shallow 0.95417")
        add("R2_x_S95417_95_5", blend(r2, s95417, 0.05), "soft staged leftover")

    # 6) Dual-base on R2 like sriram: avg(R2+0.30 d15, gpw32+0.30 d15)
    b1 = unit(L(r2) + 0.30 * d15)
    b2 = unit(L(gpw32) + 0.30 * d15)
    dual = unit((unit(csv_roundtrip(ids, b1)) + unit(csv_roundtrip(ids, b2))) / 2.0)
    add("DUAL_R2_GPW32_dmlp30", rf(dual), "sriram-style dual-base avg with R2+gpw32 @0.30 v15")

    # Rank for submit: prefer different levers, not near-duplicates of R2
    for c in cands:
        log(f"CAND {c['tag']:28s} ρR2={c['rho_r2']:7.3f}e-4 ρ440={c['rho_sri440']:7.3f}e-4 | {c['note']}")

    # Stage all
    staged = []
    for c in cands:
        path = SUB / f"sub_v24_{c['tag']}.csv"
        stage(path, ids, c["vec"])
        staged.append({**c, "path": path})

    # Submit order — different levers, highest EV first
    order = [
        "SRI440_pub",  # consume the new public high
        "R2_to_SRI440_0p60e4",  # move ATH toward it
        "R2_dmlp_v15_w0.22",  # new neural axis on our ATH
        "DUAL_R2_GPW32_dmlp30",  # sriram dual-base recipe on our files
        "SRI440_x_S95417_90_10",  # Areeb on new high
        "R2_to_GPW32_0p35e4",
        "R2_to_SRI440_1p20e4",
        "R2_dmlp_v15_w0.30",
        "R2_x_S95417_95_5",
    ]

    by_tag = {s["tag"]: s for s in staged}
    results = []
    for tag in order:
        if tag not in by_tag:
            continue
        s = by_tag[tag]
        msg = f"v24.{tag} | ρ(R2)={s['rho_r2']:.3f}e-4 ρ(440)={s['rho_sri440']:.3f}e-4 | {s['note']}"[:480]
        log(f"SUBMIT {msg}")
        ok, pub, status = submit_and_poll(s["path"], msg)
        results.append({"tag": tag, "pub": pub, "ok": ok, "status": status, "rho_r2": s["rho_r2"]})
        log(f"DONE {tag} pub={pub} status={status}")
        if status == "quota_or_error":
            log("STOP quota")
            break
        time.sleep(6)

    best = max((r["pub"] for r in results if r.get("pub") is not None), default=None)
    body = [
        "# GM V24 New Levers Report",
        "",
        f"Prior ATH `0.95426` · elapsed {time.time()-T0:.0f}s",
        "",
        "## Live",
        "| tag | Public | ρ(R2) e-4 | status |",
        "|---|---:|---:|---|",
    ]
    for r in results:
        body.append(f"| `{r['tag']}` | **{r.get('pub')}** | {r.get('rho_r2', float('nan')):.3f} | {r.get('status')} |")
    body += [
        "",
        f"- Best this batch: **{best}**",
        "",
        "## Lever thesis",
        "- Sriram 0.95440: dual-base (step11+Bhuvan) + **w=0.30 Δmlp PyTabKit v15** (was 0.15/v5).",
        "- goodpjw ladder-meets-stack **0.95432** peer.",
        "- Avoided re-running orth/twilight/Ray-1.",
        "",
        "## Log",
        "```",
        *LINES,
        "```",
        "",
    ]
    REPORT.write_text("\n".join(body) + "\n")
    # meta without vec/path objects
    meta = {
        "results": results,
        "cands": [{"tag": c["tag"], "rho_r2": c["rho_r2"], "rho_sri440": c["rho_sri440"], "note": c["note"]} for c in cands],
        "best": best,
        "elapsed": time.time() - T0,
    }
    OUT.joinpath("v24_new_levers_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    log(f"REPORT {REPORT} best={best}")


if __name__ == "__main__":
    main()
