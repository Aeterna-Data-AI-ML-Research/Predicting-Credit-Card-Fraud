#!/usr/bin/env python3
"""V23.1 — GPW compressed-ORTH @0.30e-4 from R2 + Areeb anti-K2 reflect @~0.45e-4."""
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
BANK = REPO / "external/banks/goodpjw2008_s6e11-oof-bank"
FAM = np.load("/tmp/s6e11_v16/out/v20_1_families.npz")
NB = Path("/tmp/s6e11_nb_research")
R2 = SUB / "sub_v22_3_LIVE_ridge_continuation.csv"
K2 = SUB / "sub_v22_5_LIVE_crest_orth4.csv"
KAG = "/tmp/s6e11_v15/venv/bin/kaggle"
REPORT = LOG / "GM_V23_1_GPW_ORTH_AND_ANTIK2_REPORT.md"
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


def project_out(v, B):
    v = np.asarray(v, float).ravel()
    X = np.column_stack([np.ones(len(v)), np.asarray(B, float)])
    xtx = X.T @ X
    xtx.flat[:: xtx.shape[0] + 1] += 1e-8
    return v - X @ np.linalg.solve(xtx, X.T @ v)


def bisect_vs(ref, base_L, d, target, n=28):
    d = unit(d)
    a, b = 1e-6, 1.5
    best_w, best_c, best_err = 0.0, rf(base_L), 1e9
    for _ in range(n):
        mid = 0.5 * (a + b)
        cand = rf(base_L + mid * d)
        rho = omr(cand, ref)
        err = abs(rho - target)
        if err < best_err:
            best_err, best_w, best_c = err, mid, cand
        if rho < target:
            a = mid
        else:
            b = mid
    return best_w, best_c, omr(best_c, ref)


def load_test(name: str):
    p = BANK / f"test_{name}.npy"
    if p.exists():
        return np.load(p).ravel()[:N_TE]
    key = f"{name}_test"
    if key in FAM.files:
        return FAM[key].ravel()[:N_TE]
    if name.startswith("Q17_") and f"{name}_test" in FAM.files:
        return FAM[f"{name}_test"].ravel()[:N_TE]
    return None


def submit_and_poll(path: Path, msg: str):
    pub = None
    ok = False
    try:
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
            for ln in p.stdout.strip().splitlines()[1:15]:
                if path.name not in ln:
                    continue
                log(f"poll {ln[:240]}")
                if "COMPLETE" in ln.upper():
                    cols = [c.strip() for c in ln.split(",")]
                    for tok in reversed(cols):
                        try:
                            pub = float(tok)
                            return True, pub
                        except ValueError:
                            continue
    except Exception as e:
        log(f"submit error: {e}")
        traceback.print_exc()
    return ok, pub


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    SUB.mkdir(parents=True, exist_ok=True)
    log("=== V23.1 GPW_comp_ORTH + anti-K2 ===")

    sample = pd.read_csv(R2)
    ids = sample["id"].to_numpy()
    assert len(ids) == N_TE

    def load_sub(path):
        return pd.read_csv(path).set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    r2 = load_sub(R2)
    k2 = load_sub(K2)
    raw_gp = load_sub(NB / "sriram_ds/reference.csv")
    raw_new = load_sub(NB / "sriram_ds/new_neural.csv")
    raw_old = load_sub(NB / "sriram_ds/old_neural.csv")
    sri = rf(L(raw_gp) + 0.15 * (L(raw_new) - L(raw_old)))
    log(f"anchors ρ(sri): R2={omr(r2,sri):.3f}e-4 K2={omr(k2,sri):.3f}e-4 K2−R2={omr(k2,r2):.3f}e-4")

    members = {
        "Q17_megayak_shallow_tree": +0.2108,
        "Q17_sameer_glm10f": +0.0905,
        "Q17_daoviet_gpdvae": +0.0118,
        "Q17_alex_glm_gen": -0.0716,
        "ebm_all_bag": +0.1680,
        "ebm_dig_bag": +0.1393,
        "ebm_dig": -0.1256,
        "gb_bag5": -0.0826,
        "gbn_bag5_w512": -0.0290,
        "glm_ohk_c01": +0.1127,
        "glm_dl_s50_bag": +0.1079,
    }
    d = np.zeros(N_TE)
    used = []
    for name, w in members.items():
        ot = load_test(name)
        if ot is None:
            log(f"MISS {name}")
            continue
        d = d + w * unit(L(ot))
        used.append(name)
    d = unit(d)
    log(f"GPW dir members={len(used)}")

    B = np.column_stack(
        [
            unit(L(sri)),
            unit(L(raw_gp)),
            unit(L(raw_new)),
            unit(L(raw_old)),
            unit(L(FAM["Q17_megayak_shallow_tree_test"].ravel()[:N_TE])),
            unit(L(FAM["Q17_sameer_glm10f_test"].ravel()[:N_TE])),
        ]
    )
    d_orth = unit(project_out(d, B))
    log(f"max|corr| raw={float(np.max(np.abs(np.corrcoef(d, B.T)[0,1:]))):.4f} "
        f"orth={float(np.max(np.abs(np.corrcoef(d_orth, B.T)[0,1:]))):.6f}")

    results = []

    # P0: GPW_comp_ORTH @ 0.30e-4
    w0, c0, rho0 = bisect_vs(r2, L(r2), d_orth, 0.30)
    path0 = SUB / "sub_v23_1_GPW_comp_ORTH_0p30e4.csv"
    pd.DataFrame({"id": ids, "is_fraud": c0}).to_csv(path0, index=False)
    msg0 = (
        f"v23.1 GPW_comp_ORTH @0.30e-4 from R2 | ρ(R2)={rho0:.3f}e-4 "
        f"ρ(sri)={omr(c0,sri):.3f}e-4 members={len(used)} nested_screen=5/5_t9"
    )
    log(f"STAGED {path0.name}")
    log(f"SUBMIT {msg0}")
    ok0, pub0 = submit_and_poll(path0, msg0)
    results.append({"tag": "GPW_comp_ORTH_0p30", "pub": pub0, "ok": ok0, "rho_r2": rho0, "w": w0})
    log(f"P0 DONE pub={pub0}")
    time.sleep(10)

    # P1: anti-K2 Areeb reflect
    d_anti = L(r2) - L(k2)
    c1 = rf(L(r2) + 1.0 * d_anti)
    rho1 = omr(c1, r2)
    path1 = SUB / "sub_v23_1_AREEB_antiK2_reflect.csv"
    pd.DataFrame({"id": ids, "is_fraud": c1}).to_csv(path1, index=False)
    msg1 = (
        f"v23.1 AREEB R2+1.0*(R2-K2) anti-orth reflect | ρ(R2)={rho1:.3f}e-4 "
        f"ρ(sri)={omr(c1,sri):.3f}e-4 ρ(K2)={omr(c1,k2):.3f}e-4"
    )
    log(f"STAGED {path1.name}")
    log(f"SUBMIT {msg1}")
    ok1, pub1 = submit_and_poll(path1, msg1)
    results.append({"tag": "AREEB_antiK2", "pub": pub1, "ok": ok1, "rho_r2": rho1})
    log(f"P1 DONE pub={pub1}")

    best = max((r["pub"] for r in results if r.get("pub") is not None), default=None)
    body = [
        "# GM V23.1 GPW_ORTH + Anti-K2 Report",
        "",
        f"ATH hold reference `0.95426` · elapsed {time.time()-T0:.0f}s",
        "",
        "| tag | Public | ρ(R2) e-4 |",
        "|---|---:|---:|",
    ]
    for r in results:
        body.append(f"| `{r['tag']}` | **{r.get('pub')}** | {r.get('rho_r2', float('nan')):.3f} |")
    body += ["", f"- Best among probes: **{best}**", "", "## Log", "```", *LINES, "```", ""]
    REPORT.write_text("\n".join(body) + "\n")
    Path("/tmp/s6e11_v16/out/v23_1_submit_meta.json").write_text(
        json.dumps({"results": results, "elapsed": time.time() - T0}, indent=2, default=float)
    )
    log(f"REPORT {REPORT} best={best}")


if __name__ == "__main__":
    main()
