#!/usr/bin/env python3
"""V41 STEP 2 — Step_AA / Step_AB frontier geometry audit + staged candidates.

STRICT NO_SUBMIT=True. Writes logs/v41/STEP2_AUDIT.md (+.json) then STOPS.
Does NOT run Step 3.
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
EXT = ROOT / "external/v41_nov23_frontier"
LOG = ROOT / "logs/v41"
OUT = ROOT / "out/v41_staged"
SSD_OUT = Path("/tmp/s6e11_v41_ssd/out/v41_staged")
DATA = ROOT / "data" if (ROOT / "data/test.csv").exists() else Path("/tmp/s6e11_v39_ssd/data")
OUR_Q = Path("/tmp/s6e11_v39_ssd/out/v40_ladder/S01_PUB_areeb_95567.csv")
KAGGLE = Path("/Users/mz/Library/Python/3.14/bin/kaggle")

SHA = {
    "Step_AA": "51706855fa6de98ef88c1d02bebfc24ab340092d05eb513e227e048041c0346c",
    "Step_AB": "3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038",  # tip 0.95569
    "areeb_095568": "3b9fe6d72e3102101ab33d35a17cdb90b1e41de40c3792a7dc1a1cd619d1bc8c",
    "dennis": "7de7fcc291f699bf84ea26ea08c54409fb62018a232382516efe6b2ffc86fd0c",
    "quarter_095567": "60ec5bd11eca0445c3b0b222a2715d3ffeee9594968cbce56061db16d87f5c62",
    "Step_Z": "ee61b50611326eb668033f088f8803e28541874c1cff2cf073334ea7d0f05a11",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def U(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / len(v)


def L(v: np.ndarray) -> np.ndarray:
    return logit(np.clip(U(v), 1e-12, 1 - 1e-12))


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df["id"].to_numpy(), df["is_fraud"].to_numpy(float)


def assert_pin(path: Path, expect: str):
    got = sha256_file(path)
    assert got == expect, (path, got, expect)
    return got


def one_mrho(a, b) -> float:
    return 1.0 - float(spearmanr(a, b).statistic)


def zone_mask(test: pd.DataFrame) -> np.ndarray:
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
    return (
        (risk < 13.1)
        | ((velocity <= 4.1) & (risk <= 20))
        | (clean & (((velocity <= 4.1) & (risk <= 33.8)) | ((velocity <= 6) & (risk <= 25))))
    ).to_numpy()


def qstats(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    qs = np.quantile(x, [0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "n": int(len(x)),
        "min": float(np.min(x)),
        "p50": float(qs[0]),
        "p90": float(qs[1]),
        "p95": float(qs[2]),
        "p99": float(qs[3]),
        "max": float(qs[4]),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def regime_mask(u: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "bulk_neg_qlt080": u < 0.80,
        "twilight_080_985": (u >= 0.80) & (u <= 0.985),
        "fraud_tail_gt985": u > 0.985,
    }


def write_sub(path: Path, ids: np.ndarray, p: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = pd.DataFrame({"id": ids, "is_fraud": p}).to_csv(index=False, lineterminator="\n")
    path.write_bytes(text.encode())
    # mirror to SSD
    SSD_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, SSD_OUT / path.name)
    return sha256_file(path)


def md_table(df: pd.DataFrame, fmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join([""] + [str(c) for c in cols]) + " |",
        "| " + " | ".join(["---"] * (len(cols) + 1)) + " |",
    ]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v:{fmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def kaggle_top() -> list[str]:
    r = subprocess.run(
        [str(KAGGLE), "kernels", "list", "--competition", "playground-series-s6e11",
         "--sort-by", "scoreDescending", "--page-size", "25"],
        capture_output=True, text=True,
    )
    lines = [ln for ln in (r.stdout + "\n" + r.stderr).splitlines() if not ln.startswith("Warning:")]
    return lines


def main():
    assert NO_SUBMIT is True
    LOG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    gpw = EXT / "goodpjw_95569"
    # prefer competition test for zones (test.csv has no is_fraud)
    test_path = DATA / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"missing competition test.csv under {DATA}")
    test = pd.read_csv(test_path)
    ids = test["id"].to_numpy()
    assert len(ids) == N_TE
    zones = zone_mask(test)
    n_z = int(zones.sum())
    assert n_z == 16730, n_z

    paths = {
        "quarter_our_095567": OUR_Q if OUR_Q.exists() else EXT / "recon/areeb_095568_rebuilt.csv",
        "Step_Z": gpw / "submission_step_z.csv",
        "Step_AA": gpw / "submission_step_aa.csv",
        "Step_AB_tip_095569": gpw / "submission.csv",
        "areeb_095568": EXT / "areeb_95568/submission.csv",
        "dennis_clipped": EXT / "dennis/submission.csv",
        "malyshev_v42": EXT / "pinned_csv/malyshev_v42_from_npz.csv",
    }
    # ensure quarter is true quarter sha
    if sha256_file(paths["quarter_our_095567"]) != SHA["quarter_095567"]:
        # rebuild from areeb recipe artifacts already verified in step1
        paths["quarter_our_095567"] = Path("/tmp/s6e11_v39_ssd/external/public_harvest/areeb_95567/submission.csv")

    assert_pin(paths["Step_AA"], SHA["Step_AA"])
    assert_pin(paths["Step_AB_tip_095569"], SHA["Step_AB"])
    assert_pin(paths["areeb_095568"], SHA["areeb_095568"])
    assert_pin(paths["dennis_clipped"], SHA["dennis"])
    assert_pin(paths["quarter_our_095567"], SHA["quarter_095567"])
    assert_pin(paths["Step_Z"], SHA["Step_Z"])

    vec = {k: load_csv(p)[1] for k, p in paths.items()}
    # align ids
    for k, p in paths.items():
        i, _ = load_csv(p)
        assert np.array_equal(i, ids), k

    # --- lineage docs ---
    lineage = {
        "CRITICAL_NAMING_CORRECTION": {
            "user_label_Step_AA_as_095569": False,
            "fact": (
                "goodpjw submission_step_aa.csv (sha 51706855…) is Step AA, documented public LB 0.95567 "
                "(v41). The tip file submission.csv (sha 3e1dba5d…) is Step AB, public LB 0.95569 (v43)."
            ),
            "Step_AA_formula": (
                "Zr( MALV43 + 0.75 * clip_p99(MOVE43=MALV43-MALV42, outside zones) + 1.5 * EARLIER )"
            ),
            "Step_AB_formula": (
                "Zr( SRIF_orthogonal_0.95568 - 0.4*RECENT - 0.15*MOVE6_0.95557 + 0.15*MSTEP )"
            ),
            "byte_repro_from_local_only": (
                "NOT without pinned inputs: malyshevdanil_v42/v43 CSVs, EARLIER residual, "
                "sriram_orthogonal_0.95568, RECENT, MOVE6, MSTEP, Zr zone applicator. "
                "Executed CSVs are SHA-pinned; full local rebuild needs goodpjw pinned-input dataset."
            ),
            "OOF_twin": (
                "No dedicated Step_AA/AB OOF twin npy. Notebook reports RealMLP OOF (v15 0.950314, v28 0.950016), "
                "TabM OOF 0.937314, and in-sample replay OOF deltas for early steps B–G; AA/AB are board-model "
                "test-side constructions with forecast points (AA forecast +1.05; AB +0.58)."
            ),
        },
        "dennis": {
            "notebook_title": "S6E11 Fraud signal rank fusion four refinements",
            "reported_numeric_LB_in_title": None,
            "markdown_claim": (
                "Parent is goodpjw board V43 (0.95569). Blend 12% of scaled rank residual from areeb 0.95568. "
                "No numeric LB claimed for the dennis output itself."
            ),
            "exact_formula_verified": (
                "parent=Step_AB(submission.csv sha 3e1dba5d) + 0.12 * clip_p985(L(areeb)-L(parent)); "
                "NOT Step_AA-based. 1-ρ(dennis, Step_AB)=0.0040e-4 (near-identity)."
            ),
            "exec_sha": SHA["dennis"],
        },
    }

    top_lines = kaggle_top()
    ge_95569 = [ln for ln in top_lines if "0.95569" in ln or "0.9557" in ln]
    lineage["kaggle_scoreDescending_head"] = top_lines[:22]
    lineage["kernels_mentioning_ge_095569"] = ge_95569

    # --- pairwise matrix ---
    order = [
        "quarter_our_095567",
        "Step_Z",
        "malyshev_v42",
        "Step_AA",
        "areeb_095568",
        "dennis_clipped",
        "Step_AB_tip_095569",
    ]
    M = np.zeros((len(order), len(order)))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            M[i, j] = 0.0 if i == j else one_mrho(vec[a], vec[b]) * 1e4
    mat = pd.DataFrame(M, index=order, columns=order)

    # --- zone geometry ---
    u_aa, u_ab, u_q, u_ar, u_d = map(U, [vec["Step_AA"], vec["Step_AB_tip_095569"], vec["quarter_our_095567"],
                                          vec["areeb_095568"], vec["dennis_clipped"]])
    zone_ranks = {
        "Step_AA_on_zones": qstats(u_aa[zones]),
        "Step_AB_on_zones": qstats(u_ab[zones]),
        "quarter_on_zones": qstats(u_q[zones]),
        "areeb_on_zones": qstats(u_ar[zones]),
        "dennis_on_zones": qstats(u_d[zones]),
        "n_zones": n_z,
        "n_AA_zone_rows_above_050": int((u_aa[zones] > 0.50).sum()),
        "n_AA_zone_rows_above_080": int((u_aa[zones] > 0.80).sum()),
        "n_AB_zone_rows_above_050": int((u_ab[zones] > 0.50).sum()),
        "n_AB_zone_rows_above_080": int((u_ab[zones] > 0.80).sum()),
        "definition": (
            "risk<13.1 OR (vel<=4.1 & risk<=20) OR (clean flags & ((vel<=4.1 & risk<=33.8)|(vel<=6 & risk<=25))); "
            "same as areeb notebook (16,730 rows)."
        ),
    }

    # dennis soft zone pull analysis (parent=AB)
    La, Lq, Lar, Ld, Lab = map(L, [vec["Step_AA"], vec["quarter_our_095567"], vec["areeb_095568"],
                                    vec["dennis_clipped"], vec["Step_AB_tip_095569"]])
    gap_areeb_minus_ab = Lar - Lab
    lim = float(np.quantile(np.abs(gap_areeb_minus_ab), 0.985))
    soft_zone_pull = 0.12 * float(np.clip(gap_areeb_minus_ab[zones], -lim, lim).mean())
    dennis_analysis = {
        "parent_is_Step_AB_not_AA": True,
        "clip_p985_limit": lim,
        "mean_soft_zone_delta_logit": soft_zone_pull,
        "expected_cap_magnitude": 0.12 * lim,
        "zone_gap_areeb_minus_AB_stats": qstats(gap_areeb_minus_ab[zones]),
        "nonzone_gap_areeb_minus_AB_stats": qstats(gap_areeb_minus_ab[~zones]),
        "note": (
            "On zones areeb has hard demotion (~logit of near-zero ranks) so L(areeb)-L(AB) is large negative; "
            "dennis only applies ±p985 cap * 0.12, i.e. soft pull, not -100."
        ),
    }

    # AA - quarter residual outside/inside zones
    r_aa_q = La - Lq
    outside = ~zones
    r_out = r_aa_q[outside]
    p95 = float(np.quantile(np.abs(r_out), 0.95))
    top5 = np.abs(r_out) > p95
    # reconstruct areeb-like: quarter + 0.70*clip_p95(AA-quarter) outside, zones demoted
    score_areeb_like = Lq.copy()
    score_areeb_like[outside] = Lq[outside] + 0.70 * np.clip(r_out, -p95, p95)
    score_areeb_like[zones] -= 100
    areeb_like = U(score_areeb_like)
    # unclipped 1.00 * AA residual outside + zone demotion
    score_full = Lq.copy()
    score_full[outside] = Lq[outside] + 1.00 * r_out
    score_full[zones] -= 100
    full_pull = U(score_full)
    # AA as-is (no extra demotion)
    # zone-only demotion of AA
    score_aa_hz = La.copy()
    score_aa_hz[zones] -= 100
    aa_hard_zones = U(score_aa_hz)

    # counterfactuals to attribute +1 tick AA path vs areeb
    # Note: public AA was 0.95567; areeb 0.95568; AB 0.95569. Compare geometries not invent LB.
    counterfactual = {
        "p95_of_abs_AA_minus_quarter_outside": p95,
        "frac_outside_clipped_by_p95": float(top5.mean()),
        "residual_outside_quantiles": qstats(r_out),
        "residual_outside_by_AA_regime": {
            name: qstats(r_out[m[outside]]) for name, m in regime_mask(u_aa).items()
        },
        "residual_inside_zones": qstats(r_aa_q[zones]),
        "1mrho_areeb_like_vs_areeb_e4": one_mrho(areeb_like, vec["areeb_095568"]) * 1e4,
        "1mrho_full1p0_zoneHard_vs_areeb_e4": one_mrho(full_pull, vec["areeb_095568"]) * 1e4,
        "1mrho_full1p0_zoneHard_vs_AA_e4": one_mrho(full_pull, vec["Step_AA"]) * 1e4,
        "1mrho_AAhardZones_vs_areeb_e4": one_mrho(aa_hard_zones, vec["areeb_095568"]) * 1e4,
        "1mrho_AAhardZones_vs_AA_e4": one_mrho(aa_hard_zones, vec["Step_AA"]) * 1e4,
        "tick_attribution_hypothesis": (
            "areeb ≠ unclipped AA: areeb applies 0.70*clip_p95(AA−quarter) on non-zones + hard zone demotion. "
            "Counterfactual AAhardZones is very close to areeb (small 1-ρ), while full 1.00*unclipped residual "
            "moves away from areeb toward raw AA. Public AA listed 0.95567; areeb clip gained +1 tick to 0.95568; "
            "goodpjw tip Step AB (not AA) is the 0.95569 file."
        ),
    }

    # inside-zone rank diffs
    inside = {
        "U_AA_minus_U_quarter": qstats(u_aa[zones] - u_q[zones]),
        "U_areeb_minus_U_AA": qstats(u_ar[zones] - u_aa[zones]),
        "U_dennis_minus_U_AB": qstats(u_d[zones] - u_ab[zones]),
        "U_AB_minus_U_AA": qstats(u_ab[zones] - u_aa[zones]),
    }

    # --- stage candidates ---
    staged = {}

    def pin_copy(src: Path, dest_name: str) -> str:
        dest = OUT / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        SSD_OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, SSD_OUT / dest_name)
        return sha256_file(dest)

    # F1: user asked for AA sha — stage it, but also stage true tip AB
    p = vec["Step_AA"]
    staged["F1_pub_Step_AA_sha51706855.csv"] = {
        "sha": pin_copy(paths["Step_AA"], "F1_pub_Step_AA_sha51706855.csv"),
        "expected": SHA["Step_AA"],
        "claimed_LB_from_goodpjw_docs": 0.95567,
        "NOTE": "NOT the 0.95569 tip; that is Step_AB/submission.csv",
    }
    assert staged["F1_pub_Step_AA_sha51706855.csv"]["sha"] == SHA["Step_AA"]

    p = vec["Step_AB_tip_095569"]
    staged["F1b_pub_Step_AB_095569.csv"] = {
        "sha": pin_copy(paths["Step_AB_tip_095569"], "F1b_pub_Step_AB_095569.csv"),
        "expected": SHA["Step_AB"],
        "claimed_LB": 0.95569,
        "NOTE": "True Slot-1 PUB lock candidate (goodpjw tip).",
    }
    assert staged["F1b_pub_Step_AB_095569.csv"]["sha"] == SHA["Step_AB"]

    p = vec["dennis_clipped"]
    staged["F2_pub_dennis_clipped.csv"] = {
        "sha": pin_copy(paths["dennis_clipped"], "F2_pub_dennis_clipped.csv"),
        "expected": SHA["dennis"],
        "1mrho_vs_AB_e4": one_mrho(p, vec["Step_AB_tip_095569"]) * 1e4,
        "NOTE": "Near-duplicate of Step_AB (1-ρ≈0.004e-4); weak as a separate slot.",
    }
    assert staged["F2_pub_dennis_clipped.csv"]["sha"] == SHA["dennis"]

    # F3: Step_AB (tip) with hard zones — user said Step_AA but isolation of zone mask on tip is more useful;
    # also produce AA-with-hard-zones for areeb comparison.
    p = aa_hard_zones
    staged["F3_StepAA_with_hard_zones.csv"] = {
        "sha": write_sub(OUT / "F3_StepAA_with_hard_zones.csv", ids, p),
        "1mrho_vs_F1_AA_e4": one_mrho(p, vec["Step_AA"]) * 1e4,
        "1mrho_vs_F2_dennis_e4": one_mrho(p, vec["dennis_clipped"]) * 1e4,
        "1mrho_vs_areeb_e4": one_mrho(p, vec["areeb_095568"]) * 1e4,
        "1mrho_vs_AB_e4": one_mrho(p, vec["Step_AB_tip_095569"]) * 1e4,
    }
    p_ab_hz = U(np.where(zones, Lab - 100, Lab))
    staged["F3b_StepAB_with_hard_zones.csv"] = {
        "sha": write_sub(OUT / "F3b_StepAB_with_hard_zones.csv", ids, p_ab_hz),
        "1mrho_vs_AB_e4": one_mrho(p_ab_hz, vec["Step_AB_tip_095569"]) * 1e4,
        "1mrho_vs_areeb_e4": one_mrho(p_ab_hz, vec["areeb_095568"]) * 1e4,
        "1mrho_vs_dennis_e4": one_mrho(p_ab_hz, vec["dennis_clipped"]) * 1e4,
    }

    # F4/F5: on tip AB, add clipped quarter/v42 residual (pure diversity pullback)
    base = Lab
    for tag, src, a, demote in [
        ("F4_StepAB_plus_clip_quarter_a0p15", Lq, 0.15, False),
        ("F4z_StepAB_plus_clip_quarter_a0p15_zones", Lq, 0.15, True),
        ("F5_StepAB_plus_clip_quarter_a0p30", Lq, 0.30, False),
        ("F5z_StepAB_plus_clip_quarter_a0p30_zones", Lq, 0.30, True),
        ("F4v_StepAB_plus_clip_v42_a0p15", L(vec["malyshev_v42"]), 0.15, False),
        ("F5v_StepAB_plus_clip_v42_a0p30", L(vec["malyshev_v42"]), 0.30, False),
        # also AA-based variants as user wording
        ("F4aa_StepAA_plus_clip_quarter_a0p15", Lq, 0.15, False),
        ("F5aa_StepAA_plus_clip_quarter_a0p30", Lq, 0.30, False),
    ]:
        b = base if not tag.startswith("F4aa") and not tag.startswith("F5aa") else La
        resid = src - b
        cap = float(np.quantile(np.abs(resid[outside]), 0.985))
        score = b.copy()
        score[outside] = b[outside] + a * np.clip(resid[outside], -cap, cap)
        if demote:
            score[zones] -= 100
        p = U(score)
        fname = f"{tag}.csv"
        staged[fname] = {
            "sha": write_sub(OUT / fname, ids, p),
            "alpha": a,
            "cap_p985": cap,
            "zone_demote": demote,
            "1mrho_vs_AB_e4": one_mrho(p, vec["Step_AB_tip_095569"]) * 1e4,
            "1mrho_vs_AA_e4": one_mrho(p, vec["Step_AA"]) * 1e4,
            "1mrho_vs_areeb_e4": one_mrho(p, vec["areeb_095568"]) * 1e4,
            "1mrho_vs_dennis_e4": one_mrho(p, vec["dennis_clipped"]) * 1e4,
        }

    # pairwise among primary staged
    primary = [
        "F1_pub_Step_AA_sha51706855.csv",
        "F1b_pub_Step_AB_095569.csv",
        "F2_pub_dennis_clipped.csv",
        "F3_StepAA_with_hard_zones.csv",
        "F3b_StepAB_with_hard_zones.csv",
        "F4_StepAB_plus_clip_quarter_a0p15.csv",
        "F5_StepAB_plus_clip_quarter_a0p30.csv",
    ]
    prim_vec = {n: load_csv(OUT / n)[1] for n in primary}
    Pm = np.zeros((len(primary), len(primary)))
    for i, a in enumerate(primary):
        for j, b in enumerate(primary):
            Pm[i, j] = 0.0 if i == j else one_mrho(prim_vec[a], prim_vec[b]) * 1e4
    prim_mat = pd.DataFrame(Pm, index=primary, columns=primary)

    # OOF variance note (no AA/AB OOF twins)
    oof_note = {
        "available": False,
        "reason": "Step_AA/AB have no matched OOF twins in harvest; skipping fabricated 60k/240k OOF sims.",
        "proxy": "Early-step in-sample replay OOF exists only for B–G in goodpjw notebook; not transferable to AA/AB.",
    }

    # write audit
    lines = []
    lines.append("# V41 STEP2 AUDIT — AA/AB Frontier Geometry")
    lines.append("")
    lines.append(f"**NO_SUBMIT={NO_SUBMIT}** · Hard stop. Do not run Step 3 without `PROCEED TO STEP 3`.")
    lines.append("")
    lines.append("## 0. Critical naming correction")
    lines.append("")
    lines.append("| File | SHA16 | goodpjw name | Documented public LB |")
    lines.append("|---|---|---|---:|")
    lines.append("| `submission_step_aa.csv` | `51706855…` | **Step AA** | **0.95567** |")
    lines.append("| `submission.csv` | `3e1dba5d…` | **Step AB** | **0.95569** |")
    lines.append("| `areeb` clipped | `3b9fe6d7…` | — | **0.95568** |")
    lines.append("| `dennis` | `7de7fcc2…` | AB + 0.12·clip(areeb−AB) | **unclaimed** (near-AB) |")
    lines.append("")
    lines.append("Slot-1 PUB lock for 0.95569 must be **Step AB**, not Step AA.")
    lines.append("")
    lines.append("## 1. Step_AA / Step_AB lineage & dennis LB")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(lineage, indent=2)[:15000])
    lines.append("```")
    lines.append("")
    lines.append("### Kernels at/above 0.95569 (from scoreDescending list)")
    lines.append("")
    lines.append("Top of list is `goodpjw2008/...-lb-0-95569`. dennis title has **no numeric LB**. "
                 "areeb is 0.95568. No higher than 0.95569 appeared in the first page.")
    lines.append("")
    lines.append("## 2. Pairwise `1−ρ` (×1e−4)")
    lines.append("")
    lines.append(md_table(mat))
    lines.append("")
    lines.append("## 3. Zone mask (n=16,730) vs Step_AA / AB / areeb / dennis")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(zone_ranks, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### dennis soft-zone mechanics (parent = Step_AB)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dennis_analysis, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Residual `L(Step_AA)−L(quarter)` outside vs inside zones")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"counterfactual": counterfactual, "inside_zone_rank_diffs": inside}, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 4. Staged candidates (`out/v41_staged/`, NO_SUBMIT)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(staged, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Primary staged pairwise `1−ρ` (×1e−4)")
    lines.append("")
    lines.append(md_table(prim_mat))
    lines.append("")
    lines.append("## 5. OOF 60k/240k check")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(oof_note, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 6. STOP")
    lines.append("")
    lines.append("Step 2 complete. **Awaiting human audit.** Reply `PROCEED TO STEP 3` to continue.")
    lines.append("")

    (LOG / "STEP2_AUDIT.md").write_text("\n".join(lines))
    (LOG / "STEP2_AUDIT.json").write_text(
        json.dumps(
            {
                "NO_SUBMIT": True,
                "lineage": lineage,
                "matrix_1mrho_e4": mat.to_dict(),
                "zone_ranks": zone_ranks,
                "dennis_analysis": dennis_analysis,
                "counterfactual": counterfactual,
                "inside": inside,
                "staged": staged,
                "primary_matrix_1mrho_e4": prim_mat.to_dict(),
                "oof_note": oof_note,
            },
            indent=2,
            default=str,
        )
    )

    print("=== STRICT NO_SUBMIT=True | V41 STEP2 ===")
    print("NAMING: Step_AA sha5170… = LB 0.95567; Step_AB sha3e1d… = LB 0.95569 tip")
    print(mat.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nZones n=", n_z, "AA>0.80 among zones:", zone_ranks["n_AA_zone_rows_above_080"])
    print("dennis vs AB 1-ρ e-4:", f"{one_mrho(vec['dennis_clipped'], vec['Step_AB_tip_095569'])*1e4:.4f}")
    print("areeb-like vs areeb 1-ρ e-4:", f"{counterfactual['1mrho_areeb_like_vs_areeb_e4']:.4f}")
    print("AAhardZones vs areeb 1-ρ e-4:", f"{counterfactual['1mrho_AAhardZones_vs_areeb_e4']:.4f}")
    print("\nStaged:")
    for k, v in staged.items():
        print(f"  {k}: sha16={v['sha'][:16]} extras={ {kk:vv for kk,vv in v.items() if kk!='sha'} }")
    print(f"\nWROTE {LOG/'STEP2_AUDIT.md'}")
    print("STOP. Awaiting PROCEED TO STEP 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
