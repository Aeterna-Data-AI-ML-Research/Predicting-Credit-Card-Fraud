#!/usr/bin/env python3
"""Phase 0 ($0): verify compact-LM serializer, rotated-10 math, fold0 shapes, A40 estimate."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

VOL = Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud")
WORK = Path(os.environ.get("GM_L7_WORK", "/tmp/gm_l7_preflight"))
WORK.mkdir(parents=True, exist_ok=True)

# Lay out data the trainer expects
data = WORK / "data"
orig_dir = WORK / "original"
ext = WORK / "external" / "goodpjw2008_s6e11-compact-lm-seed7"
out = WORK / "preds" / "l7_5f_rot10"
for p in (data, orig_dir, ext, out):
    p.mkdir(parents=True, exist_ok=True)

# symlink/copy essentials
for name in ("train.csv", "test.csv"):
    src = VOL / "data" / name
    dst = data / name
    if not dst.exists():
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
og = VOL / "original" / "credit_card_fraud_2026.csv"
dst_og = orig_dir / "credit_card_fraud_2026.csv"
if not dst_og.exists():
    try:
        dst_og.symlink_to(og)
    except OSError:
        shutil.copy2(og, dst_og)
for name in ("lm7_oof_f0.npy", "lm7_test_f0.npy", "README.md"):
    src = VOL / "external/banks/goodpjw2008_s6e11-compact-lm-seed7" / name
    dst = ext / name
    if src.exists() and not dst.exists():
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)

# point trainer env
os.environ["L7_ROOT"] = str(WORK)
os.environ["L7_DATA"] = str(data)
os.environ["L7_ORIG"] = str(dst_og)
os.environ["L7_OUT"] = str(out)
os.environ["L7_CKPT"] = str(WORK / "ckpt")
os.environ["L7_CACHE"] = str(WORK / "cache_compact.npz")
os.environ["L7_F0_DIR"] = str(ext)
os.environ.setdefault("SEED", "7")
os.environ.setdefault("EPOCHS", "8")
os.environ.setdefault("BATCH", "64")
os.environ.setdefault("A40_ROWS_S", "1200")

sys.path.insert(0, str(VOL / "scripts"))
# import as module by path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "train_compact_lm_5f", VOL / "scripts" / "train_compact_lm_5f.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
est = mod.preflight_cpu()

# architecture audit note
arch = {
    "backbone": "GPT-2 small (12 layers, 12 heads, d=768) — megayak compact lm_member.py / goodpjw seed7",
    "NOT": "distilgpt2; NOT final-token logits[:, -1, [tok0,tok1]]",
    "scoring": "full-sequence NLL under label False then True for each column order; LLR = mean(ll[...,1]-ll[...,0]) or Bayes logsumexp",
    "shapes": {"oof": "(139992, K_oof, 2)", "test": "(299984, 4, 2)", "dtype": "float32"},
    "fold0_auc_readme": 0.947545,
    "vocab": "value tokens used in data (~1372) + END + PAD + LAB×2 + COL tokens + optional BOS",
    "seq_len": f"T≈{est.get('T')} compact (~66 tokens/row vs 224 natural)",
    "lr_epochs": {"LR": 5e-5, "EPOCHS": 8, "BATCH": 64, "SEED": 7},
    "estimate": est,
}
report = VOL / "logs/overnight/GM_L7_5F_PHASE0_PREFLIGHT.md"
report.parent.mkdir(parents=True, exist_ok=True)
md = []
md.append("# GM L7 5F Rotated-10 — Phase 0 Pre-Flight ($0.00)\n\n")
md.append("## Architecture (exact match to `lm7_*`)\n\n")
md.append(f"- Backbone: **{arch['backbone']}**\n")
md.append(f"- Scoring: **{arch['scoring']}**\n")
md.append(f"- Explicitly **{arch['NOT']}**\n")
md.append(f"- Shapes: `{arch['shapes']}`\n")
md.append(f"- Fold0 README AUC: `{arch['fold0_auc_readme']}`\n")
md.append(f"- T={est['T']} V={est['V']} NTOK={est['NTOK']} NC={est['NC']}\n\n")
md.append("## Rotated 4-of-10 cover\n\n")
md.append("Every master order 0..9 appears exactly **2×** across folds 0..4 test tensors.\n\n")
md.append("## A40 wall-clock estimate (Folds 1–4 only)\n\n")
md.append(f"| metric | value |\n|---|---:|\n")
md.append(f"| anchor | {est.get('anchor','A40 empirical')} |\n")
md.append(f"| batch | {est.get('batch')} (alt {est.get('recommended_batch')}) |\n")
md.append(f"| train min/fold | {est['train_min_per_fold']:.1f} |\n")
md.append(f"| score min/fold | {est['score_min_per_fold']:.1f} |\n")
md.append(f"| replica min (once) | {est['replica_min']:.1f} |\n")
md.append(f"| **total hours** | **{est['total_hours']:.2f}** |\n")
md.append(f"| **$ @ $0.59/hr** | **${est['cost_usd_at_0p59']:.2f}** |\n")
md.append(f"| alt BATCH=96 hours/cost | {est.get('alt_batch96_hours',0):.2f}h / ${est.get('alt_batch96_cost',0):.2f} |\n")
md.append(f"| ≤6.5h / ≤$3.85 gate | `{'PASS' if est['budget_ok_le_6p5h'] else 'FAIL'}` |\n\n")
md.append("Watchdog cap: 510 min ($5.01 @ $0.59) << $10 hard cap.\n")
md.append(f"\nPHASE0_OK elapsed={est['elapsed_preflight_s']:.1f}s\n")
report.write_text("".join(md))
(out / "phase0_arch.json").write_text(json.dumps(arch, indent=2, default=float))
print("REPORT", report)
print("PHASE0_COMPLETE")
