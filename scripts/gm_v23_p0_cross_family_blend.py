#!/usr/bin/env python3
"""V23 P0 — Cross-family logit blend (Areeb-style): LM/sriram × shallow-tree.

R2 ATH CSV is offline with the volume; use public LM-family peers on the same
board band (sriram 0.95422 / goodpjw 0.95421 / lm17 0.95417) as X, and alex
shallow / s417_wide10 as S. Gate: logit Spearman ρ(X,S)∈[0.975,0.995],
1−ρ(blend,X)∈[0.22e-4,0.42e-4], prefer w≈0.10.
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
OUT = Path("/tmp/s6e11_p0_blend/out")
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
    df = pd.read_csv(path)
    assert "id" in df.columns and "is_fraud" in df.columns
    df = df.sort_values("id").reset_index(drop=True)
    assert len(df) == N_TE
    return df["id"].to_numpy(), df["is_fraud"].to_numpy(float)


def blend(x, s, w_s):
    """rank_frac( (1-w)*L(X) + w*L(S) )."""
    return rank_frac((1.0 - w_s) * logit_rank(x) + w_s * logit_rank(s))


def submit_and_poll(path: Path, msg: str):
    pub = None
    ok = False
    r = subprocess.run(
        [KAG, "competitions", "submit", "-c", "playground-series-s6e11", "-f", str(path), "-m", msg],
        capture_output=True,
        text=True,
        timeout=120,
    )
    log(r.stdout.strip()[:300])
    log(r.stderr.strip()[:300])
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
            log(f"poll {ln[:220]}")
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
    OUT.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)

    files = {
        # LM / board-band family (stand-ins for offline R2_095426)
        "sri_95422": IN / "sri" / "submission.csv",
        "gpw_95421": IN / "gpw21" / "submission.csv",
        "lm_95417": IN / "lm17" / "submission.csv",
        # Shallow / tree family
        "alex_main": IN / "alex" / "submission.csv",
        "alex_s417": IN / "alex" / "submission_s417_wide10.csv",
        "alex_cvled": IN / "alex" / "submission_cvled_wide10.csv",
        # Reference blend
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
        log(f"loaded {name}: mean={v.mean():.6f}")

    lm_names = [n for n in ("sri_95422", "gpw_95421", "lm_95417") if n in vecs]
    tree_names = [n for n in ("alex_s417", "alex_cvled", "alex_main") if n in vecs]
    log(f"LM pool={lm_names} TREE pool={tree_names}")

    # Pairwise ρ
    log("=== logit-rank Spearman matrix (selected) ===")
    pairs = []
    for xname in lm_names:
        for sname in tree_names:
            rho = spearman(logit_rank(vecs[xname]), logit_rank(vecs[sname]))
            d = one_m_rho_e4(vecs[xname], vecs[sname])
            ok_band = 0.975 <= rho <= 0.995
            log(f"  {xname} × {sname}: ρ={rho:.6f}  1−ρ={d:.3f}e-4  band={ok_band}")
            pairs.append((xname, sname, rho, d, ok_band))

    # Prefer pairs in diversity band; else take closest below 0.995
    cand_pairs = [p for p in pairs if p[4]]
    if not cand_pairs:
        # relax: take lowest ρ among pairs still < 0.999
        loose = [p for p in pairs if p[2] < 0.999]
        loose.sort(key=lambda t: t[2])
        cand_pairs = loose[:3]
        log(f"NO pair in [0.975,0.995] — using lowest-ρ loose pairs: {[(a,b,r) for a,b,r,_,_ in cand_pairs]}")
    else:
        # prefer sri as X (our manifold), then highest tree LB proxy (s417)
        def key(t):
            x, s, rho, _, _ = t
            return (0 if x == "sri_95422" else 1, 0 if "s417" in s else 1, -rho)

        cand_pairs = sorted(cand_pairs, key=key)

    results = []
    for xname, sname, rho, d, _ in cand_pairs:
        x, s = vecs[xname], vecs[sname]
        best = None
        for w in (0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
            b = blend(x, s, w)
            r_x = one_m_rho_e4(b, x)
            r_s = one_m_rho_e4(b, s)
            gate = 0.22 <= r_x <= 0.42
            row = {
                "x": xname,
                "s": sname,
                "w": w,
                "rho_xs": rho,
                "rho_bx": r_x,
                "rho_bs": r_s,
                "gate": gate,
                "vec": b,
            }
            results.append(row)
            log(
                f"  blend {xname}+{w:.2f}*{sname}: 1−ρ(X)={r_x:.3f}e-4 1−ρ(S)={r_s:.3f}e-4 gate={gate}"
            )
            if gate and (best is None or abs(w - 0.10) < abs(best["w"] - 0.10)):
                best = row
        # also keep exact 0.90/0.10 even if slightly outside gate (Areeb canonical)
        if best is None:
            b = blend(x, s, 0.10)
            best = {
                "x": xname,
                "s": sname,
                "w": 0.10,
                "rho_xs": rho,
                "rho_bx": one_m_rho_e4(b, x),
                "rho_bs": one_m_rho_e4(b, s),
                "gate": 0.22 <= one_m_rho_e4(b, x) <= 0.42,
                "vec": b,
            }
            log(f"  FALLBACK canonical w=0.10 gate={best['gate']} ρ(X)={best['rho_bx']:.3f}e-4")

        # stage this pair's best
        tag = f"sub_v23_p0_{xname}_w{best['w']:.2f}_{sname}.csv"
        path = SUB / tag
        pd.DataFrame({"id": ids, "is_fraud": best["vec"]}).to_csv(path, index=False)
        best["path"] = str(path)
        log(f"STAGED {path.name} gate={best['gate']}")

        # Prefer first gated sri×s417 or first gated overall
        if best.get("gate") and xname == "sri_95422":
            winner = best
            break
    else:
        # pick best gated among all results, else canonical sri×s417@0.10
        gated = [r for r in results if r.get("gate")]
        if gated:
            winner = min(gated, key=lambda r: (0 if r["x"] == "sri_95422" else 1, abs(r["w"] - 0.10)))
        else:
            xname, sname = cand_pairs[0][0], cand_pairs[0][1]
            b = blend(vecs[xname], vecs[sname], 0.10)
            path = SUB / f"sub_v23_p0_{xname}_w0.10_{sname}.csv"
            pd.DataFrame({"id": ids, "is_fraud": b}).to_csv(path, index=False)
            winner = {
                "x": xname,
                "s": sname,
                "w": 0.10,
                "rho_xs": cand_pairs[0][2],
                "rho_bx": one_m_rho_e4(b, vecs[xname]),
                "rho_bs": one_m_rho_e4(b, vecs[sname]),
                "gate": False,
                "vec": b,
                "path": str(path),
            }

    # also stage LIVE name
    live = SUB / "sub_v23_p0_LIVE_cross_family_blend.csv"
    pd.DataFrame({"id": ids, "is_fraud": winner["vec"]}).to_csv(live, index=False)
    log(
        f"WINNER {winner['x']} + {winner['w']:.2f}*{winner['s']} | "
        f"ρ(XS)={winner['rho_xs']:.6f} 1−ρ(X)={winner['rho_bx']:.3f}e-4 gate={winner['gate']}"
    )

    # compare to areeb if present
    if "areeb_95421" in vecs:
        log(f"vs areeb_95421: 1−ρ={one_m_rho_e4(winner['vec'], vecs['areeb_95421']):.3f}e-4")

    msg = (
        f"v23.p0 cross-family {winner['x']}+{winner['w']:.2f}*{winner['s']} | "
        f"ρ(XS)={winner['rho_xs']:.4f} 1−ρ(X)={winner['rho_bx']:.3f}e-4 gate={winner['gate']}"
    )
    log(f"SUBMIT {msg}")
    ok, pub = submit_and_poll(live, msg)

    report = [
        "# GM V23 P0 Cross-Family Blend Report",
        "",
        f"**elapsed** {time.time()-T0:.0f}s",
        "",
        f"## Decision: `{'SUBMIT' if ok else 'FAIL'}` · Public LB: **{pub}**",
        f"- Blend: `{winner['x']} + {winner['w']:.2f} * {winner['s']}`",
        f"- logit ρ(X,S)={winner['rho_xs']:.6f}",
        f"- 1−ρ(blend,X)={winner['rho_bx']:.3f}e-4 · gate={winner['gate']}",
        f"- File: `{live.name}`",
        "",
        "Note: personal R2_095426 CSV offline (volume); X is public LM-band peer on the same board.",
        "",
        "## Pair screen",
    ]
    for xname, sname, rho, d, okb in pairs:
        report.append(f"- `{xname}` × `{sname}`: ρ={rho:.6f} 1−ρ={d:.3f}e-4 band={okb}")
    report_path = LOG / "GM_V23_P0_CROSS_FAMILY_BLEND_REPORT.md"
    report_path.write_text("\n".join(report) + "\n")
    meta = {"winner": {k: v for k, v in winner.items() if k != "vec"}, "pub": pub, "ok": ok}
    OUT.joinpath("v23_p0_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    log(f"REPORT {report_path}")
    log(f"DONE pub={pub} ok={ok}")


if __name__ == "__main__":
    main()
