#!/usr/bin/env python3
"""V41 STEP 1 — Harvest, SHA-pin & exact reconstruction of public frontier.

STRICT NO_SUBMIT=True. Writes logs/v41/STEP1_AUDIT.md then exits.
Does NOT run Steps 2–6.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr

NO_SUBMIT = True
N_TE = 299_984
ROOT = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
SSD_EXT = Path("/tmp/s6e11_v41_ssd/external/v41_nov23_frontier")
EXT = ROOT / "external/v41_nov23_frontier"
LOG = ROOT / "logs/v41"
KAGGLE = Path("/Users/mz/Library/Python/3.14/bin/kaggle")
PYDATA = ROOT / "data" if (ROOT / "data/test.csv").exists() else Path("/tmp/s6e11_v39_ssd/data")
OUR_95567 = Path("/tmp/s6e11_v39_ssd/out/v40_ladder/S01_PUB_areeb_95567.csv")
if not OUR_95567.exists():
    OUR_95567 = ROOT / "submissions/ATH_v3_areeb_95567.csv"

LB = {
    "our_095567": 0.95567,
    "Step_V": 0.95558,
    "Step_W": 0.95559,
    "Step_X": 0.95560,
    "Step_Y_095561": 0.95561,
    "Step_Z_095562": 0.95562,
    "Step_AA": 0.95569,  # goodpjw board tip (AA file used inside areeb recipe)
    "dennis_clipped": None,  # API may not expose; unscored direction historically
    "areeb_095568": 0.95568,
    "malyshev_v42": 0.95565,
}


def sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = "\n".join(ln for ln in (r.stdout + r.stderr).splitlines() if not ln.startswith("Warning:"))
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ranks(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def logit_ranks(v: np.ndarray) -> np.ndarray:
    return logit(np.clip(ranks(v), 1e-12, 1 - 1e-12))


def load_sub(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    assert list(df.columns)[:2] == ["id", "is_fraud"] or (
        "id" in df.columns and "is_fraud" in df.columns
    ), path
    ids = df["id"].to_numpy()
    p = df["is_fraud"].to_numpy(float)
    assert len(p) == N_TE, (path, len(p))
    assert np.isfinite(p).all() and not np.isnan(p).any()
    return ids, p


def integrity_row(name: str, path: Path, reported_lb) -> dict:
    ids, p = load_sub(path)
    u = ranks(p)
    # uniform rank file ≈ equals (rank-0.5)/N of itself → unique ranks, u≈p if already ranks
    is_ur = bool(np.allclose(p, u, atol=1e-12)) or bool(np.allclose(p, ranks(p), atol=1e-9))
    # stricter: file values equal uniform ranks of themselves
    is_uniform_rank = bool(np.max(np.abs(p - u)) < 1e-9)
    return {
        "name": name,
        "filename": str(path.relative_to(EXT)) if str(path).startswith(str(EXT)) else path.name,
        "path": str(path),
        "reported_LB": reported_lb,
        "sha256": sha256_file(path),
        "sha16": sha256_file(path)[:16],
        "rows": int(len(p)),
        "min": float(np.min(p)),
        "max": float(np.max(p)),
        "mean": float(np.mean(p)),
        "is_uniform_rank": is_uniform_rank,
        "n_nan": int(np.isnan(p).sum()),
        "n_inf": int(np.isinf(p).sum()),
        "id0": int(ids[0]),
        "id_last": int(ids[-1]),
    }


def ensure_synced():
    """Copy SSD harvest into repo external/ (lightweight pointer sync)."""
    EXT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    if SSD_EXT.exists():
        # symlink or copy key dirs
        for name in [
            "areeb_95568",
            "dennis",
            "goodpjw_board",
            "goodpjw_95569",
            "nov23_frontier",
            "nov23_round2",
            "busyaprime_52",
            "kabartay",
            "expanded_frontier",
        ]:
            src, dst = SSD_EXT / name, EXT / name
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError:
                    shutil.copytree(src, dst)


def export_npz_as_csv(npz_path: Path, key: str, ids: np.ndarray, out: Path) -> Path:
    with np.load(npz_path, allow_pickle=False) as z:
        assert key in z.files, (key, z.files)
        assert np.array_equal(z["id"], ids)
        p = z[key].astype(float)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(out, index=False)
    return out


def reproduce_areeb_095568(ids: np.ndarray, test: pd.DataFrame) -> dict:
    """Exact recipe from areeb notebook cell 10 (clipped Step AA on quarter-step)."""
    round2 = EXT / "nov23_round2/additional_scored_rankings.npz"
    aa_path = EXT / "goodpjw_95569/submission_step_aa.csv"
    if not aa_path.exists():
        aa_path = EXT / "goodpjw_board/submission_step_aa.csv"
    areeb_exec = EXT / "areeb_95568/submission.csv"

    with np.load(round2, allow_pickle=False) as z:
        assert np.array_equal(z["id"], ids)
        v42 = z["own_v42"].astype(float)
        step_z = z["Z"].astype(float)

    aa = pd.read_csv(aa_path)
    assert aa.id.to_numpy().tolist() == ids.tolist() or np.array_equal(aa.id.to_numpy(), ids)
    aa_p = aa.set_index("id").loc[ids, "is_fraud"].to_numpy(float)

    flags = [
        "is_new_merchant",
        "is_foreign_transaction",
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_ai_generated_scam_attempt",
    ]
    clean = test.cvv_retry_count.eq(0) & test.prior_disputes.eq(0) & ~test[flags].astype(bool).any(axis=1)
    risk, velocity = test.merchant_risk_score, test.velocity_score
    zones = (
        (risk < 13.1)
        | ((velocity <= 4.1) & (risk <= 20))
        | (clean & (((velocity <= 4.1) & (risk <= 33.8)) | ((velocity <= 6) & (risk <= 25))))
    ).to_numpy()

    v42_logit = logit_ranks(v42)
    # v43 reconstruction used as intermediate in notebook
    v43 = ranks(ranks(v42_logit - (logit_ranks(step_z) - v42_logit)) - 2 * zones)
    base = v43  # notebook re-reads CSV; ranks of that
    # re-derive quarter-step exactly as notebook
    base_logit = logit_ranks(base)
    delta = base_logit - v42_logit
    score = base_logit.copy()
    score[~zones] += 0.25 * delta[~zones]
    score[zones] -= 100
    quarter = ranks(score)
    quarter_sha = hashlib.sha256(
        pd.DataFrame({"id": ids, "is_fraud": quarter}).to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()
    anchor = logit_ranks(quarter)

    residual = logit_ranks(aa_p) - anchor
    cap = float(np.quantile(np.abs(residual[~zones]), 0.95))
    score2 = anchor.copy()
    score2[~zones] += 0.70 * np.clip(residual[~zones], -cap, cap)
    score2[zones] -= 100
    final = ranks(score2)
    final_csv = pd.DataFrame({"id": ids, "is_fraud": final}).to_csv(index=False, lineterminator="\n")
    final_sha = hashlib.sha256(final_csv.encode()).hexdigest()

    out_path = EXT / "recon/areeb_095568_rebuilt.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(final_csv.encode())

    _, exec_p = load_sub(areeb_exec)
    max_abs = float(np.max(np.abs(final - exec_p)))
    max_rank = float(np.max(np.abs(rankdata(final) - rankdata(exec_p))))
    rho = float(spearmanr(final, exec_p).statistic)
    return {
        "space": "logit(uniform_rank) with zone mask; residual=AA-quarter; clip p95 abs outside zones; weight 0.70; then uniform ranks",
        "NOT_Y_minus_Z": True,
        "cap_p95": cap,
        "weight": 0.70,
        "quarter_sha256": quarter_sha,
        "final_sha256": final_sha,
        "exec_sha256": sha256_file(areeb_exec),
        "max_abs_diff": max_abs,
        "max_rank_diff": max_rank,
        "one_minus_spearman": 1.0 - rho,
        "sha_prefix_match_notebook": final_sha.startswith("3b9fe6d72e3102101ab33d35a17cdb90b1e41de40c3792"),
        "zones_n": int(zones.sum()),
        "rebuilt_path": str(out_path),
    }


def reproduce_dennis_current(ids: np.ndarray) -> dict:
    """Exact recipe from dennis notebook (current): parent + 0.12*clip_p985(alt-parent)."""
    parent_sha = "3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038"
    alt_sha = "3b9fe6d72e3102101ab33d35a17cdb90b1e41de40c3792a7dc1a1cd619d1bc8c"
    # locate parent/alt among known harvest paths first (avoid scanning recon/)
    parent = alt = None
    candidates = [
        EXT / "goodpjw_95569/submission.csv",
        EXT / "goodpjw_board/submission.csv",
        EXT / "areeb_95568/submission.csv",
        EXT / "recon/areeb_095568_rebuilt.csv",
        SSD_EXT / "goodpjw_95569/submission.csv",
        SSD_EXT / "goodpjw_board/submission.csv",
        SSD_EXT / "areeb_95568/submission.csv",
    ]
    for p in candidates:
        if not p.exists() or p.stat().st_size < 1_000_000:
            continue
        dig = sha256_file(p)
        if dig == parent_sha and parent is None:
            parent = p
        if dig == alt_sha and alt is None:
            alt = p
    if alt is None:
        alt = EXT / "areeb_95568/submission.csv"
    dennis_exec = EXT / "dennis/submission.csv"

    def L(path: Path) -> np.ndarray:
        df = pd.read_csv(path).set_index("id", verify_integrity=True)
        p = df.loc[ids, "is_fraud"].to_numpy(float)
        return logit_ranks(p)

    info = {"parent_found": parent is not None, "parent_path": str(parent) if parent else None, "alt_path": str(alt)}
    if parent is None:
        info["error"] = f"parent sha {parent_sha[:16]} not in harvest; cannot exact-rebuild current dennis"
        # still compare dennis vs areeb
        _, d = load_sub(dennis_exec)
        _, a = load_sub(alt)
        info["dennis_vs_areeb_1mrho"] = 1.0 - float(spearmanr(d, a).statistic)
        return info

    left, right = L(parent), L(alt)
    gap = right - left
    limit = float(np.quantile(np.abs(gap), 0.985))
    candidate = ranks(left + 0.12 * np.clip(gap, -limit, limit))
    out = EXT / "recon/dennis_current_rebuilt.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "is_fraud": candidate}).to_csv(out, index=False)
    _, exec_p = load_sub(dennis_exec)
    rho = float(spearmanr(candidate, exec_p).statistic)
    info.update(
        {
            "space": "logit(uniform_rank); residual=alt-parent; clip quantile abs 0.985; weight 0.12; then uniform ranks",
            "NOT_Y_minus_Z_026": True,
            "limit_p985": limit,
            "weight": 0.12,
            "max_abs_diff": float(np.max(np.abs(candidate - exec_p))),
            "max_rank_diff": float(np.max(np.abs(rankdata(candidate) - rankdata(exec_p)))),
            "one_minus_spearman": 1.0 - rho,
            "rebuilt_sha256": sha256_file(out),
            "exec_sha256": sha256_file(dennis_exec),
        }
    )
    return info


def reproduce_historical_YZ_clip(ids: np.ndarray) -> dict:
    """Malyshev-documented dennis formula: +0.26 * clip_p99(Y-Z) in some space — probe spaces."""
    with np.load(EXT / "nov23_frontier/additional_scored_rankings.npz", allow_pickle=False) as z:
        Y, Z = z["Y"].astype(float), z["Z"].astype(float)
        assert np.array_equal(z["id"], ids)
    logits = np.load(EXT / "nov23_frontier/new_member_logits.npy")
    names = json.loads((EXT / "nov23_frontier/new_member_names.json").read_text())
    # external:dennis_Nov23 is index 1
    dennis_idx = names.index("external:dennis_Nov23")
    dennis_logit = logits[dennis_idx].astype(float)
    # convert logit members? new_member_logits may already be logit-ranks; also try rank form
    dennis_from_logit = ranks(dennis_logit)  # if already logit, ranks(logit) still ok for compare

    results = {}
    for space, Ys, Zs in [
        ("raw_file_values", Y, Z),
        ("uniform_rank", ranks(Y), ranks(Z)),
        ("logit_rank", logit_ranks(Y), logit_ranks(Z)),
    ]:
        r = Ys - Zs
        # if Y,Z already uniform ranks in NPZ, raw==uniform
        p99 = float(np.quantile(np.abs(r), 0.99))
        for base_name, base in [("Z_raw", Z), ("Z_logit", logit_ranks(Z)), ("Y_raw", Y)]:
            if "logit" in base_name:
                rebuilt = ranks(base + 0.26 * np.clip(r if space == "logit_rank" else logit_ranks(Y) - logit_ranks(Z), -p99, p99))
                # only meaningful when space matches
            else:
                add = 0.26 * np.clip(r, -p99, p99)
                if space == "logit_rank":
                    rebuilt = ranks(logit_ranks(base) + add)
                else:
                    rebuilt = ranks(base + add)
            # compare to dennis_Nov23 exported rank
            rho = float(spearmanr(rebuilt, dennis_from_logit).statistic)
            key = f"{space}__on_{base_name}"
            results[key] = {
                "p99": p99,
                "one_minus_spearman_vs_dennis_Nov23": 1.0 - rho,
                "max_abs_vs_dennis_rank": float(np.max(np.abs(rebuilt - dennis_from_logit))),
            }
    # also direct: assume NPZ Y/Z already ranks; rebuild on Z in logit space as Malyshev text
    r = logit_ranks(Y) - logit_ranks(Z)
    p99 = float(np.quantile(np.abs(r), 0.99))
    rebuilt = ranks(logit_ranks(Z) + 0.26 * np.clip(r, -p99, p99))
    best = {
        "documented_formula": "Z + 0.26 * clip(logit_rank(Y)-logit_rank(Z), ±p99_abs) → uniform ranks",
        "p99": p99,
        "one_minus_spearman_vs_dennis_Nov23": 1.0 - float(spearmanr(rebuilt, dennis_from_logit).statistic),
        "max_abs": float(np.max(np.abs(rebuilt - dennis_from_logit))),
        "space_probe": results,
    }
    # export dennis_Nov23 as CSV for integrity table
    out = EXT / "recon/dennis_Nov23_from_npz.csv"
    pd.DataFrame({"id": ids, "is_fraud": dennis_from_logit}).to_csv(out, index=False)
    best["dennis_Nov23_csv"] = str(out)
    best["dennis_Nov23_sha256"] = sha256_file(out)
    return best


def pairwise_matrix(vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    keys = list(vectors.keys())
    M = np.zeros((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i == j:
                M[i, j] = 0.0
            else:
                rho = float(spearmanr(vectors[a], vectors[b]).statistic)
                M[i, j] = (1.0 - rho) * 1e4
    return pd.DataFrame(M, index=keys, columns=keys)


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    # simple markdown
    cols = list(df.columns)
    lines = ["| " + " | ".join([""] + [str(c) for c in cols]) + " |",
             "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    assert NO_SUBMIT is True
    ensure_synced()
    sample = pd.read_csv(PYDATA / "test.csv", usecols=["id"])
    ids = sample["id"].to_numpy()
    test = pd.read_csv(PYDATA / "test.csv")
    assert len(ids) == N_TE

    # Export NPZ V–Z / own_v42 as CSVs for SHA table
    pin_dir = EXT / "pinned_csv"
    pin_dir.mkdir(parents=True, exist_ok=True)
    npz = EXT / "nov23_round2/additional_scored_rankings.npz"
    with np.load(npz, allow_pickle=False) as z:
        assert np.array_equal(z["id"], ids)
        for key, name in [
            ("V", "Step_V_from_npz.csv"),
            ("W", "Step_W_from_npz.csv"),
            ("X", "Step_X_from_npz.csv"),
            ("Y", "Step_Y_095561_from_npz.csv"),
            ("Z", "Step_Z_095562_from_npz.csv"),
            ("own_v42", "malyshev_v42_from_npz.csv"),
        ]:
            pd.DataFrame({"id": ids, "is_fraud": z[key].astype(float)}).to_csv(pin_dir / name, index=False)

    # Prefer goodpjw executed step CSVs (authoritative) when present
    gpw = EXT / "goodpjw_95569"
    if not (gpw / "submission_step_z.csv").exists():
        gpw = EXT / "goodpjw_board"

    files = {
        "our_095567": OUR_95567,
        "Step_V": gpw / "submission_step_v.csv",
        "Step_W": gpw / "submission_step_w.csv",
        "Step_X": gpw / "submission_step_x.csv",
        "Step_Y_095561": gpw / "submission_step_y.csv",
        "Step_Z_095562": gpw / "submission_step_z.csv",
        "Step_AA": gpw / "submission_step_aa.csv",
        "areeb_095568": EXT / "areeb_95568/submission.csv",
        "dennis_clipped": EXT / "dennis/submission.csv",
        "malyshev_v42": pin_dir / "malyshev_v42_from_npz.csv",
    }

    # Verify NPZ Y/Z match goodpjw CSVs
    match_yz = {}
    with np.load(npz, allow_pickle=False) as z:
        for lab, key, csvk in [
            ("Y", "Y", "Step_Y_095561"),
            ("Z", "Z", "Step_Z_095562"),
            ("V", "V", "Step_V"),
        ]:
            _, p = load_sub(files[csvk])
            match_yz[lab] = {
                "max_abs_npz_vs_csv": float(np.max(np.abs(z[key].astype(float) - p))),
                "1mrho": 1.0 - float(spearmanr(z[key].astype(float), p).statistic),
            }

    integrity = [integrity_row(n, p, LB.get(n)) for n, p in files.items() if p.exists()]
    integ_df = pd.DataFrame(integrity)

    # id alignment check vs test
    for row in integrity:
        ids_f, _ = load_sub(Path(row["path"]))
        assert np.array_equal(ids_f, ids), row["name"]

    areeb_recon = reproduce_areeb_095568(ids, test)
    dennis_recon = reproduce_dennis_current(ids)
    yz_hist = reproduce_historical_YZ_clip(ids)

    # add dennis_Nov23 to integrity
    integrity.append(integrity_row("dennis_Nov23_npz", Path(yz_hist["dennis_Nov23_csv"]), None))
    integ_df = pd.DataFrame(integrity)

    vectors = {}
    for n, p in files.items():
        if p.exists():
            _, vectors[n] = load_sub(p)
    vectors["dennis_Nov23_npz"] = load_sub(Path(yz_hist["dennis_Nov23_csv"]))[1]

    # matrix as specified (+ Step_AA, dennis variants for audit)
    order = [
        "our_095567",
        "Step_V",
        "Step_W",
        "Step_X",
        "Step_Y_095561",
        "Step_Z_095562",
        "dennis_clipped",
        "areeb_095568",
    ]
    mat = pairwise_matrix({k: vectors[k] for k in order})

    # inventory for later steps
    inventory = {
        "busyaprime_52": {
            "path": str(EXT / "busyaprime_52"),
            "files": sorted([p.name for p in (EXT / "busyaprime_52").glob("*")]) if (EXT / "busyaprime_52").exists() else [],
        },
        "kabartay_d27_notebook": {
            "path": str(EXT / "kabartay"),
            "files": sorted([p.name for p in (EXT / "kabartay").glob("*")]) if (EXT / "kabartay").exists() else [],
            "note": "Notebook pulled for Step 4; D27 zone rules to be parsed then.",
        },
        "goodpjw_step_AA_LB_note": "goodpjw board tip now publishes Step AA; areeb 0.95568 uses AA sha 51706855…",
    }

    # write audit
    lines = []
    lines.append("# V41 STEP1 AUDIT — Frontier Harvest & Exact Reconstruction")
    lines.append("")
    lines.append(f"**NO_SUBMIT={NO_SUBMIT}** · Hard stop after this file. Do not run Step 2 without `PROCEED TO STEP 2`.")
    lines.append("")
    lines.append("## 0. Critical formula finding (read first)")
    lines.append("")
    lines.append(
        "Public description “0.26 × clip99(Y−Z)” is **Malyshev’s documentation of an earlier dennis direction**, "
        "not the executed `areeb 0.95568` recipe."
    )
    lines.append("")
    lines.append("**`areeb 0.95568` exact space (notebook cell 10):**")
    lines.append("1. Inputs: Malyshev `own_v42`, goodpjw `Step Z`, goodpjw `Step AA` (sha `51706855…`), test zone mask.")
    lines.append("2. Build quarter-step (our `0.95567`, sha `60ec5bd1…`) in **logit(uniform rank)** with zone demotion.")
    lines.append(
        "3. `residual = logit_rank(Step_AA) − logit_rank(quarter)`; "
        "`cap = quantile(|residual| outside zones, 0.95)`; "
        "`score = quarter + 0.70 * clip(residual, ±cap)` outside zones; zones −100; emit uniform ranks."
    )
    lines.append("")
    lines.append("**Current `dennis` notebook (executed CSV):**")
    lines.append(
        "`parent + 0.12 * clip_p985(areeb_095568 − parent)` in logit-rank space "
        f"(parent sha `{dennis_recon.get('parent_path')}` / found={dennis_recon.get('parent_found')})."
    )
    lines.append("")
    lines.append("## 1. SHA-256 & File Integrity Table")
    lines.append("")
    show = integ_df[
        ["name", "reported_LB", "sha16", "rows", "min", "max", "mean", "is_uniform_rank", "n_nan", "n_inf"]
    ].copy()
    show["sha16"] = integ_df["sha256"].str[:16]
    lines.append(md_table(show.set_index("name"), ".6g"))
    lines.append("")
    lines.append("Full SHA-256:")
    lines.append("")
    for r in integrity:
        lines.append(f"- `{r['name']}`: `{r['sha256']}`")
    lines.append("")
    lines.append("### NPZ V/Y/Z vs goodpjw CSV alignment")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(match_yz, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 2. Exact Algebraic Reproduction")
    lines.append("")
    lines.append("### 2.1 areeb 0.95568")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(areeb_recon, indent=2))
    lines.append("```")
    lines.append("")
    lines.append(
        f"**Verdict:** `1-ρ={areeb_recon['one_minus_spearman']:.3e}`, "
        f"max_abs={areeb_recon['max_abs_diff']:.3e}, "
        f"sha_prefix_ok={areeb_recon['sha_prefix_match_notebook']}."
    )
    lines.append("")
    lines.append("### 2.2 dennis (current executed) + historical Y−Z probe")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"dennis_current": dennis_recon, "historical_YZ_0p26_p99": yz_hist}, indent=2, default=str)[:12000])
    lines.append("```")
    lines.append("")
    lines.append("## 3. Pairwise `1 − ρ` Spearman (× 1e−4)")
    lines.append("")
    lines.append(md_table(mat, ".4f"))
    lines.append("")
    lines.append("## 4. Inventory ready for Steps 4–5")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(inventory, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 5. STOP")
    lines.append("")
    lines.append("Step 1 complete. **Awaiting human audit.** Reply `PROCEED TO STEP 2` to continue.")
    lines.append("")

    audit_path = LOG / "STEP1_AUDIT.md"
    audit_path.write_text("\n".join(lines))
    # also dump machine JSON
    (LOG / "STEP1_AUDIT.json").write_text(
        json.dumps(
            {
                "NO_SUBMIT": True,
                "integrity": integrity,
                "areeb_recon": areeb_recon,
                "dennis_recon": dennis_recon,
                "yz_hist_summary": {k: v for k, v in yz_hist.items() if k != "space_probe"},
                "pairwise_1mrho_e4": mat.to_dict(),
                "match_yz": match_yz,
                "inventory": inventory,
            },
            indent=2,
            default=str,
        )
    )

    # console summaries
    print("=== STRICT NO_SUBMIT=True | V41 STEP1 ===")
    print(show.to_string(index=False))
    print("\n=== Pairwise 1-ρ (×1e-4) ===")
    print(mat.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== areeb recon ===")
    print(json.dumps({k: areeb_recon[k] for k in [
        "space", "max_abs_diff", "max_rank_diff", "one_minus_spearman",
        "sha_prefix_match_notebook", "final_sha256", "exec_sha256", "quarter_sha256"
    ]}, indent=2))
    print("\n=== dennis current ===")
    print(json.dumps(dennis_recon, indent=2, default=str)[:2000])
    print(f"\nWROTE {audit_path}")
    print("STOP. Awaiting PROCEED TO STEP 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
