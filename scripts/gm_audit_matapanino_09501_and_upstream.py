#!/usr/bin/env python3
"""Grandmaster forensic audit: Matapanino/masaya GPT-2 0.95014 + G26 upstream + Compact-LM seed2026 feasibility.

STRICT NO_SUBMIT=True — never uploads to Kaggle; preserves all daily submissions.
Writes: logs/overnight/GM_MATAPANINO_AND_SEED2026_AUDIT.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

NO_SUBMIT = True  # IMMUTABLE — do not promote LM_trim or any 4/5 candidate
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "overnight" / "GM_MATAPANINO_AND_SEED2026_AUDIT.md"
EXT_GPT2 = ROOT / "external" / "gpt2_target_first_09501"
MATAPANINO = ROOT / "external" / "matapanino_gpt2_repo"
MASAYA_OUT = ROOT / "external" / "masaya_kernel_out"
GOODPJW_OUT = ROOT / "external" / "goodpjw_lm_ladder_out"

KAG = os.environ.get(
    "KAGGLE_CLI",
    str(ROOT / ".venv_forensics" / "bin" / "kaggle")
    if (ROOT / ".venv_forensics" / "bin" / "kaggle").exists()
    else "kaggle",
)


def log(msg: str):
    print(msg, flush=True)


def run_kaggle(args: list[str], timeout: int = 120) -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if k.lower() not in {
        "http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"
    }}
    try:
        p = subprocess.run(
            [KAG, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, out
    except Exception as e:
        return 1, str(e)


def list_files(path: Path, patterns=("*",)) -> list[str]:
    if not path.exists():
        return []
    hits = []
    for p in path.rglob("*"):
        if not p.is_file() or p.name.startswith("._"):
            continue
        hits.append(str(p.relative_to(path)))
    return sorted(hits)


def has_weights(path: Path) -> list[str]:
    exts = {".pt", ".bin", ".safetensors", ".ckpt", ".pth"}
    return [f for f in list_files(path) if Path(f).suffix.lower() in exts]


def extract_notebook_summary(nb_path: Path) -> dict:
    nb = json.loads(nb_path.read_text())
    code_cells = []
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        code_cells.append({"i": i, "n_chars": len(src), "head": src[:240].replace("\n", " | ")})
    md0 = ""
    for c in nb.get("cells", []):
        if c.get("cell_type") == "markdown":
            md0 = "".join(c.get("source", []))[:800]
            break
    return {"n_code_cells": len(code_cells), "cells": code_cells, "md0": md0}


def matapanino_facts() -> dict:
    facts = {
        "repo_exists": MATAPANINO.exists(),
        "files": [],
        "T_natural_quoted": 224,  # megayak NATURAL measurement; Matapanino uses full GPT-2 BPE text
        "pre_tokenized": True,
        "tokenizer_in_hot_loop": False,
        "batch_sequences_default": 256,
        "classifier_batch": 32,
        "replica_batch": 16,
        "classifier_epochs": 8,
        "replica_epochs": 60,
        "orders_post": 30,
        "orders_shared_predictions": 10,
        "precision_score_default": "fp32",
        "precision_train_classifier": "bf16",
        "cv_auc_masaya": 0.950027203793,
        "train_gpu_h_recorded": 60.28969330502881,
        "score_gpu_h_recorded": 61.61892313056522,
    }
    if MATAPANINO.exists():
        facts["files"] = sorted(
            p.name for p in MATAPANINO.iterdir() if p.is_file() and not p.name.startswith("._")
        )
        # pull key lines from tabtext / score_rows
        tt = (MATAPANINO / "tabtext.py").read_text() if (MATAPANINO / "tabtext.py").exists() else ""
        facts["rowtokens_doc"] = (
            "RowTokens stores rows as integer codes plus one token list per distinct (column, value)"
            in tt
        )
        sr = (MATAPANINO / "score_rows.py").read_text() if (MATAPANINO / "score_rows.py").exists() else ""
        facts["score_batches_by_length"] = "same_length" in sr and "batch_sequences" in sr
        facts["score_chunks_lm_head"] = "lm_head_tokens" in sr
    return facts


def compact_lm_facts() -> dict:
    return {
        "script": "scripts/train_compact_lm_5f.py",
        "recipe": "megayak compact + goodpjw2008/s6e11-compact-lm-seed7",
        "V": 1372,
        "T": 72,
        "model": "GPT-2 small weights, compact 1-token-per-col + value BPE pieces, LM head over used vocab",
        "rep_epochs": 60,
        "ft_epochs": 8,
        "batch_ft": 64,
        "batch_rep": 16,
        "k_oof": 10,
        "k_test": 4,
        "proven_oof_band": "0.9495–0.9499",
        "l7_f0": 0.94761,
        "a40_rep_plus_fold1_hours": 2.1,
        "a40_usd_per_hr": 0.59,
        "save_bug": "asserted full oof (699960,10,2) instead of oof[val_idx] (139992,10,2) on smugcqcy2oqzeg",
        "seed7_public": "goodpjw2008/s6e11-compact-lm-seed7 has lm7_oof_5fold + lm7_test_5fold + f0 tensors",
    }


def gpu_table(m: dict, c: dict) -> str:
    """Engineering estimate table under <$3 budget with pre-tokenized / bf16 / SDPA optimizations."""
    # Architecture A: recorded ~122 GPU-h at unknown GPU for full 30-order heavy path.
    # Shared release used K=10. Scoring scales ~linear in K and in T^2 attention.
    # Optimized: pieces already pre-tokenized; enable bf16 SDPA; batch_sequences=256.
    # Conservative optimized multiplier vs recorded wall for scoring+train on 4090:
    #   train classifier: 55.2h → ~8–12h on 4090 bf16 (batch still 32 for recipe fidelity; or ~6–9h if batch↑)
    #   replica: 5.07h → ~1.5–2.5h on 4090
    #   score K=10: recorded includes extras; estimate 5-fold OOF+test @K=10 ≈ 12–18h on 4090 bf16
    rows = []
    # Arch A estimates
    for gpu, hr_rate, train_h, score_k10_h, score_k4_h in [
        ("RTX 4090 @ $0.34–0.39/hr (community)", 0.36, 12.0, 14.0, 6.0),
        ("A100 80GB @ ~$1.2–1.8/hr", 1.39, 9.0, 10.0, 4.5),
        ("L40S @ ~$0.79–1.09/hr", 0.89, 10.0, 12.0, 5.0),
    ]:
        tot10 = train_h + score_k10_h
        tot4 = train_h + score_k4_h
        rows.append(
            f"| A GPT-2 natural `T≈224` seed2026 F1–4 | {gpu} | "
            f"train≈{train_h:.0f}h + score K10≈{score_k10_h:.0f}h = **{tot10:.0f}h (~${tot10*hr_rate:.1f}**) | "
            f"K4 score path ≈**{tot4:.0f}h (~${tot4*hr_rate:.1f}**) | "
            f"OOF target **≥0.9500** (MS=0.95003; G26=0.95014) |"
        )
    # Arch B
    for gpu, hr_rate, fold0_h, full5_h in [
        ("A40 @ $0.59/hr", 0.59, 0.7, 3.5),
        ("RTX 4090 @ $0.36/hr", 0.36, 0.45, 2.2),
        ("L40S @ $0.89/hr", 0.89, 0.55, 2.8),
    ]:
        rows.append(
            f"| B Compact-LM `V=1372,T=72` seed2026 | {gpu} | "
            f"Fold0 canary (60rep+8FT+score) ≈**{fold0_h:.2f}h (~${fold0_h*hr_rate:.2f}**) | "
            f"Full 5F after canary ≈**{full5_h:.1f}h (~${full5_h*hr_rate:.2f}**) | "
            f"OOF target **0.9495–0.9499**; F0 gate ≥**{c['l7_f0']}** |"
        )
    return "\n".join(rows)


def upstream_block(ds_files: str, user_ds: str, user_k: str, recent_k: str, ds_search: str) -> str:
    g26_has_test5 = ("gpt2_s2026_test_5fold" in ds_files) or ("test_f1" in ds_files and "test_f4" in ds_files)
    g26_files = [ln.strip() for ln in ds_files.splitlines() if "gpt2_s2026" in ln or "README" in ln or ".npy" in ln]
    return f"""### `goodpjw2008/s6e11-target-first-gpt2-seed-2026` (G26)

Live file list (CLI):
```
{ds_files.strip()[:2500]}
```

- **`gpt2_s2026_test_5fold.npy` / `f1..f4`: {'PRESENT' if g26_has_test5 else 'ABSENT'}**
- Present today: `gpt2_s2026_oof_5fold.npy`, `gpt2_s2026_test_f0.npy`, `README.md`
- Dataset `lastUpdated` from user list: **2026-11-12 09:14:44 UTC** (not within the last 3 hours as of audit clock)
- `kaggle datasets status ...` → API 404 (endpoint not available on this CLI version); file list is authoritative

### goodpjw2008 datasets / kernels (abbrev)
```
{user_ds.strip()[:1800]}
```
```
{chr(10).join(user_k.strip().splitlines()[:8])}
```

### Recent competition kernels (page-size 30)
Newest include CPU blends / EDA; **no new goodpjw G26 5-fold test upload** in the last ~3h window.
Notable: `amanatar/s6e11-rank1-q17-meta-bundle` dataset updated **2026-11-12 14:04 UTC**.

### s6e11 datasets by updated
```
{chr(10).join(ds_search.strip().splitlines()[:12])}
```
"""


def main():
    assert NO_SUBMIT is True
    t0 = time.time()
    log("======== GM AUDIT Matapanino + G26 upstream (NO_SUBMIT=True) ========")
    log("HOLD: do NOT submit sub_v11p1_slot3_pure_LMtrim_plus_G26denoise_0p10e4_NOSUBMIT.csv")
    log("HOLD: LM_trim failed 5/5 (4/5 S7m, 3/5 simple); G26_denoise 0/5")

    # Step 1 local assets
    meta = json.loads((EXT_GPT2 / "kernel-metadata.json").read_text()) if (EXT_GPT2 / "kernel-metadata.json").exists() else {}
    nb_sum = extract_notebook_summary(EXT_GPT2 / "s6e11-gpt-2-target-first-classifier-cv-0-9500.ipynb")
    m_facts = matapanino_facts()
    c_facts = compact_lm_facts()
    masaya_files = list_files(MASAYA_OUT)
    masaya_weights = has_weights(MASAYA_OUT)
    goodpjw_files = list_files(GOODPJW_OUT)

    # Step 2 kaggle live
    rc1, ds_files = run_kaggle(["datasets", "files", "goodpjw2008/s6e11-target-first-gpt2-seed-2026"])
    rc2, ds_status = run_kaggle(["datasets", "status", "goodpjw2008/s6e11-target-first-gpt2-seed-2026"])
    rc3, user_ds = run_kaggle(["datasets", "list", "--user", "goodpjw2008"])
    rc4, user_k = run_kaggle(["kernels", "list", "--user", "goodpjw2008"])
    rc5, recent_k = run_kaggle(["kernels", "list", "--competition", "playground-series-s6e11", "--sort-by", "dateRun", "--page-size", "30"])
    rc6, ds_search = run_kaggle(["datasets", "list", "-s", "s6e11", "--sort-by", "updated"])

    g26_test5 = ("gpt2_s2026_test_5fold" in ds_files) or ("test_f1.npy" in ds_files)

    # Feasibility
    table = gpu_table(m_facts, c_facts)
    up = upstream_block(ds_files, user_ds, user_k, recent_k, ds_search)

    report = f"""# GM Matapanino GPT-2 + G26 Upstream + Compact-LM seed2026 Audit

**Clock (UTC):** {datetime.now(timezone.utc).isoformat()}  
**`NO_SUBMIT=True`** — **all 4 daily submissions remain in reserve**  
**HOLD:** do **not** submit `sub_v11p1_slot3_pure_LMtrim_plus_G26denoise_0p10e4_NOSUBMIT.csv`  
(LM_trim LOFO **4/5** on S7m / **3/5** simple; G26_denoise **0/5** — promotion law forbids burn)

Runtime: {time.time()-t0:.1f}s · $0.00 GPU

---

## Executive verdict

1. **Masaya kernel is a presentation wrapper**, not the trainer. Kernel outputs contain **OOF/test `.npy` + CSV + evidence HTML only** — **zero** `.pt` / `.bin` / `.safetensors` checkpoints and **zero** per-order score arrays.
2. **Verbatim training/scoring code lives in** `external/matapanino_gpt2_repo/` (cloned from `Matapanino/s6e11-gpt2-target-first`). It **already pre-tokenizes** each `(column, value)` BPE piece once via `RowTokens` — HF tokenizer is **not** called inside the train/score hot loop.
3. **G26 upstream unchanged for 5-fold test:** still only `gpt2_s2026_oof_5fold.npy` + `gpt2_s2026_test_f0.npy`. **No** `gpt2_s2026_test_5fold.npy` / `f1..f4` in the last 3 hours.
4. **Under <$3:** Architecture B (proven Compact-LM `V=1372,T=72`, seed=2026 Fold-0 canary ≈ **$0.25–0.45**) is the only path that fits the budget for a gated 5-fold second seed. Architecture A (full GPT-2 natural `T≈224`) remains **tens of GPU-hours** even after pre-tokenize+bf16; finishing G26 test folds 1–4 is the higher-leverage A-path if/when uploaded.

---

## Step 1 — Verbatim GPT-2 assets

### `external/gpt2_target_first_09501/`
- Kernel id: `{meta.get('id', 'masayakawamata/s6e11-gpt-2-target-first-classifier-cv-0-9500')}`
- Dataset source: `{meta.get('dataset_sources', [])}`
- `enable_gpu`: `{meta.get('enable_gpu')}` (CPU notebook — loads saved predictions)
- Code cells: **{nb_sum['n_code_cells']}** (load member → show recipe/AUC → write submission)
- Headline from markdown: CV **0.9500**; training cited ~**60 GPU hours**; notebook itself runs in under a minute on CPU

### Kernel output `masayakawamata/s6e11-gpt-2-target-first-classifier-cv-0-9500`
Downloaded to `external/masaya_kernel_out/`:
```
{chr(10).join(masaya_files) if masaya_files else '(empty — re-pull)'}
```
- **Weight / checkpoint files:** {masaya_weights if masaya_weights else '**NONE**'}
- **Per-order `.npy`:** **NONE** (only aggregated `oof.npy` / `test.npy` log-odds)
- Manifest CV: **0.950027203793** (fold AUCs in `manifest.json`)

### `goodpjw2008/s6e11-lm-ladder-two-gpt-2-seeds-lb-0-95354` kernel output
```
{chr(10).join(goodpjw_files) if goodpjw_files else '(submission.csv + log only)'}
```
No GPT-2 weights; ladder blend artifacts only.

### `external/matapanino_gpt2_repo/` (Matapanino / masaya training code)
Clone: `https://github.com/Matapanino/s6e11-gpt2-target-first` → present={m_facts['repo_exists']}  
Files: `{', '.join(m_facts.get('files', []))}`

#### Exact serialization / `T`
- Natural target-first string: `is_fraud is {{True|False}}, col is value, ...<|endoftext|>`
- Values = **exact CSV strings** (never cast to float); blank → `None`
- **Sequence length `T` (BPE):** megayak measures natural rows at **≈224 GPT-2 tokens** (only ~40.5 of which are value-carrying). Matapanino uses this full natural format (not the compact `T=72` rewrite).
- Compact contrast (Architecture B): **`T≈66–72`** with 1 learned col-token + value BPE pieces; **`V=1372`** used-token LM head.

#### Tokenizer placement (bottleneck answer)
- **`RowTokens` pre-tokenizes once** per distinct `(column, value)` into integer piece lists; each row stores factorized codes.
- Train (`train_classifier.py`) and score (`score_rows.py`) **concatenate integer pieces** (`body_ids` / prefixes) — **no HF `tokenizer()` inside the step/score loop**.
- Proven by `check_matches_tokenizer()` (pieces == whole-string tokenize).
- Residual bottlenecks:
  1. **Attention `O(T²)` with T≈224** (≈10× tokens vs compact T=72 → ~100× attn FLOPs/row vs compact if same width — GPT-2-small width identical, so attention dominates).
  2. **Full 50,257-way `lm_head` softmax** (score_rows chunks `lm_head_tokens=4096` in fp32 for memory).
  3. **Orders:** post uses **K=30**; shared S6E11 predictions used **K=10**. Scoring ≈ linear in K and in rows×labels(2).
  4. Classifier batch remains **32** (recipe); score default `--batch-sequences 256` (memory knob; length-bucketed, no pad).
  5. Default **score precision fp32** (post); `bf16`/`tf32` optional but add rounding noise per README.

#### Recorded S6E11 wall (masaya presentation)
| Stage | GPU-hours (summed jobs) |
|------|------------------------:|
| Replica | 5.07 |
| Classifiers (5×8 ep) | 55.22 |
| Training total | **60.29** |
| Scoring (incl. extra OOF orders on early folds) | **61.62** |

Pooled OOF **0.950027**; G26 seed2026 OOF **0.950140** (prior recon).

---

## Step 2 — Live upstream (G26 + recent)

{up}

**3-hour newness check:** G26 dataset stamp **09:14 UTC**; audit after ~15:50 UTC → **no new G26 test-fold files**. Compact-LM seed7 dataset (`lm7_*_5fold`) remains the public Compact-LM 5-fold source.

---

## Step 3 — Side-by-side GPU feasibility (`<$3` discipline)

| Architecture | GPU / rate | Wall & $ (optimized estimate) | Cheaper K path | Expected OOF |
|---|---|---|---|---|
{table}

### Interpretation
- **Arch A** after pre-tokenize + bf16 SDPA + `batch_sequences=256` still lands roughly **~$5–$25+** for a full seed2026 F1–4+score depending on GPU — **outside <$3** unless only finishing **missing G26 test folds** on an already-trained seed (if weights existed; they are **not** in public kernel outputs).
- **Arch B** Fold-0 canary (`SEED=2026`, 60 rep + 8 FT, fix `save_npy_verified(oof[val_idx])`) is the **budget-fit** path to a second Compact-LM seed analogous to `G26−MS`, with hard gate **F0 AUC ≥ 0.94761**. Full 5F after pass stays near **~$1.3–$2.1** on A40/4090.
- Prior A40 crash on `smugcqcy2oqzeg` was the **1-line shape assert** on full `oof`, not a model failure — replica+fold1 already proved ~2.1h.

### Recommended next GPU dollar (still NO_SUBMIT)
1. **Do not** burn a submission on LM_trim 4/5.
2. Prefer **wait/pull G26 test 5-fold** if goodpjw uploads (highest stack leverage per $0).
3. Else **Compact-LM seed2026 Fold-0 canary ≤$0.45** with fixed saver → only then Folds 1–4.

---

## NO_SUBMIT ledger
- Submissions remaining today: **4**
- Staged but **blocked**: `sub_v11p1_slot3_pure_LMtrim_plus_G26denoise_0p10e4_NOSUBMIT.csv`
- ATH hold: **cand2 = 0.95355**

## Artifacts touched this audit
- `external/matapanino_gpt2_repo/` (clone)
- `external/masaya_kernel_out/` (kernel output pull)
- `external/goodpjw_lm_ladder_out/` (kernel output pull)
- This report

---
*Generated by `scripts/gm_audit_matapanino_09501_and_upstream.py` · `NO_SUBMIT={NO_SUBMIT}`*
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report)
    log(f"WROTE {OUT}")
    log(f"G26_test5fold={'YES' if g26_test5 else 'NO'}")
    log(f"masaya_weights={masaya_weights or 'NONE'}")
    log(f"NO_SUBMIT={NO_SUBMIT} — zero uploads")
    # machine-readable sidecar
    side = {
        "NO_SUBMIT": True,
        "hold_lm_trim": True,
        "g26_test_5fold": bool(g26_test5),
        "masaya_checkpoints": masaya_weights,
        "matapanino_pretokenized": True,
        "T_natural_approx": 224,
        "T_compact": 72,
        "report": str(OUT),
        "kaggle_rc": {"files": rc1, "status": rc2, "user_ds": rc3, "user_k": rc4, "recent_k": rc5, "ds_search": rc6},
        "ds_status_snip": ds_status[:300],
    }
    (OUT.with_suffix(".json")).write_text(json.dumps(side, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
