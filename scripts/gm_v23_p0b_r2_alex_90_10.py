#!/usr/bin/env python3
"""V23 P0b — Canonical 90/10 cross-family blend: R2_095426 × alex shallow-tree.

Force w_S=0.10 (Areeb-style) when logit Spearman ρ(X,S) ∈ [0.975, 0.995].
Also stage areeb-replica (gpw/lm × alex) for diagnostics; submit best R2@0.10 pair.
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

IN = Path("/tmp/s6e11_p0_blend/in")
R2 = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v22_3_LIVE_ridge_continuation.csv")
REPO_SUB = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions")
SUB = Path("/tmp/s6e11_repo_research/submissions")
LOG = Path("/tmp/s6e11_repo_research/logs/overnight")
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
N_TE = 299984
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def rank_frac(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def logit_rank(v):
    return logit(rank_frac(v))


def one_m_rho_e4(a, b):
    return (1.0 - float(spearmanr(a, b).correlation)) * 1e4


def spearman(a, b):
    return float(spearmanr(a, b).correlation)


def load_csv(path: Path):
    df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
    assert len(df) == N_TE and "is_fraud" in df.columns
    return df["id"].to_numpy(), df["is_fraud"].to_numpy(float)


def blend90_10(x, s):
    """Canonical Areeb: rank_frac(0.90*L(X) + 0.10*L(S))."""
    return rank_frac(0.90 * logit_rank(x) + 0.10 * logit_rank(s))


def submit_and_poll(path: Path, msg: str):
    pub = None
    r = subprocess.run(
        [KAG, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg],
        capture_output=True,
        text=True,
        timeout=120,
    )
    log(r.stdout.strip()[:400])
    log(r.stderr.strip()[:400])
    blob = (r.stdout + r.stderr).lower()
    ok = r.returncode == 0 and "400" not in (r.stdout + r.stderr) and "limit" not in blob
    if not ok:
        return False, None
    for i in range(40):
        time.sleep(18 if i else 8)
        p = subprocess.run(
            [KAG, "competitions", "submissions", "-c", "playground-series-s6e11", "--csv"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        lines = p.stdout.strip().splitlines()
        if not lines:
            continue
        hdr = [h.strip() for h in lines[0].split(",")]
        for ln in lines[1:15]:
            if path.name not in ln:
                continue
            log(f"poll {ln[:240]}")
            if "COMPLETE" in ln.upper():
                cols = [c.strip() for c in ln.split(",")]
                try:
                    pub = float(cols[hdr.index("publicScore")])
                except Exception:
                    for tok in reversed(cols):
                        try:
                            pub = float(tok)
                            break
                        except ValueError:
                            continue
                return True, pub
    return ok, pub


def main():
    for d in (SUB, LOG, REPO_SUB):
        d.mkdir(parents=True, exist_ok=True)

    files = {
        "R2_095426": R2,
        "sri_95422": IN / "sri" / "submission.csv",
        "gpw_95421": IN / "gpw21" / "submission.csv",
        "lm_95417": IN / "lm17" / "submission.csv",
        "alex_main": IN / "alex" / "submission.csv",
        "alex_s417": IN / "alex" / "submission_s417_wide10.csv",
        "alex_cvled": IN / "alex" / "submission_cvled_wide10.csv",
        "areeb_95421": IN / "areeb" / "submission.csv",
    }
    vecs = {}
    ids = None
    for name, path in files.items():
        if not path.exists():
            log(f"MISSING {name}: {path}")
            continue
        i, v = load_csv(path)
        if ids is None:
            ids = i
        else:
            assert np.array_equal(ids, i), name
        vecs[name] = v
        log(f"loaded {name} mean={v.mean():.6f}")

    assert "R2_095426" in vecs, "R2 ATH required"
    lm = [n for n in ("R2_095426", "sri_95422", "gpw_95421", "lm_95417") if n in vecs]
    tree = [n for n in ("alex_main", "alex_s417", "alex_cvled") if n in vecs]

    log("=== 90/10 screen (force w_S=0.10) ===")
    rows = []
    for xname in lm:
        for sname in tree:
            rho = spearman(logit_rank(vecs[xname]), logit_rank(vecs[sname]))
            band = 0.975 <= rho <= 0.995
            b = blend90_10(vecs[xname], vecs[sname])
            rx = one_m_rho_e4(b, vecs[xname])
            rs = one_m_rho_e4(b, vecs[sname])
            # soft gate: diversity band + non-trivial step (not noop, not huge)
            soft = band and 0.20 <= rx <= 2.50
            row = {
                "x": xname,
                "s": sname,
                "rho": rho,
                "band": band,
                "soft": soft,
                "rx": rx,
                "rs": rs,
                "vec": b,
            }
            rows.append(row)
            log(
                f"  0.90*{xname}+0.10*{sname}: ρ={rho:.6f} band={band} "
                f"1−ρ(X)={rx:.3f}e-4 1−ρ(S)={rs:.3f}e-4 soft={soft}"
            )

    # Prefer R2 × alex_main in band; else best soft; else R2×alex_main forced
    pref = [r for r in rows if r["x"] == "R2_095426" and r["s"] == "alex_main" and r["band"]]
    if not pref:
        pref = [r for r in rows if r["x"] == "R2_095426" and r["band"]]
    if not pref:
        pref = [r for r in rows if r["soft"]]
    if not pref:
        pref = [r for r in rows if r["x"] == "R2_095426" and r["s"] == "alex_main"]
    # among prefs, closest ρ to 0.98
    winner = min(pref, key=lambda r: abs(r["rho"] - 0.98))

    tag = f"sub_v23_p0b_{winner['x']}_90_10_{winner['s']}.csv"
    for dest in (SUB / tag, REPO_SUB / tag):
        pd.DataFrame({"id": ids, "is_fraud": winner["vec"]}).to_csv(dest, index=False)
    live_name = "sub_v23_p0b_LIVE_R2_alex_90_10.csv"
    for dest in (SUB / live_name, REPO_SUB / live_name):
        pd.DataFrame({"id": ids, "is_fraud": winner["vec"]}).to_csv(dest, index=False)
    live = REPO_SUB / live_name

    if "areeb_95421" in vecs:
        log(f"vs areeb_95421: 1−ρ={one_m_rho_e4(winner['vec'], vecs['areeb_95421']):.3f}e-4")
    if "sri_95422" in vecs:
        log(f"vs sri_95422: 1−ρ={one_m_rho_e4(winner['vec'], vecs['sri_95422']):.3f}e-4")
    log(f"vs R2: 1−ρ={one_m_rho_e4(winner['vec'], vecs['R2_095426']):.3f}e-4")

    msg = (
        f"v23.p0b 90/10 {winner['x']}+0.10*{winner['s']} | "
        f"ρ(XS)={winner['rho']:.4f} 1−ρ(R2)={winner['rx']:.3f}e-4 band={winner['band']}"
    )
    log(f"SUBMIT {msg}")
    ok, pub = submit_and_poll(live, msg)

    body = [
        "# GM V23 P0b R2×Alex 90/10 Report",
        "",
        f"**elapsed** {time.time()-T0:.0f}s",
        "",
        f"## Decision: `{'SUBMIT' if ok else 'FAIL'}` · Public LB: **{pub}**",
        f"- Canonical blend: `0.90 * {winner['x']} + 0.10 * {winner['s']}`",
        f"- logit ρ(X,S)={winner['rho']:.6f} (target ≈0.98)",
        f"- 1−ρ(blend,X)={winner['rx']:.3f}e-4 · band={winner['band']} · soft={winner['soft']}",
        f"- File: `{live_name}`",
        "",
        "## All 90/10 pairs",
    ]
    for r in rows:
        body.append(
            f"- `0.90*{r['x']}+0.10*{r['s']}`: ρ={r['rho']:.6f} "
            f"1−ρ(X)={r['rx']:.3f}e-4 band={r['band']} soft={r['soft']}"
        )
    report = LOG / "GM_V23_P0B_R2_ALEX_90_10_REPORT.md"
    report.write_text("\n".join(body) + "\n")
    # mirror into volume logs if present
    vol_log = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/logs/overnight")
    if vol_log.exists():
        (vol_log / report.name).write_text(report.read_text())
    meta = {"winner": {k: v for k, v in winner.items() if k != "vec"}, "pub": pub, "ok": ok}
    Path("/tmp/s6e11_p0_blend/out").mkdir(parents=True, exist_ok=True)
    Path("/tmp/s6e11_p0_blend/out/v23_p0b_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    log(f"REPORT {report}")
    log(f"DONE pub={pub} ok={ok}")


if __name__ == "__main__":
    main()
