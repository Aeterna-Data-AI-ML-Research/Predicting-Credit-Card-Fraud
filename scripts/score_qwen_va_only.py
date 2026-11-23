#!/usr/bin/env python3
"""Score Fold-0 VA only from saved LoRA; save ll.npy before any Test pass."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen15b_fold0_storyteller import (  # noqa: E402
    build_chunks,
    pretok_chunks,
    score_ll_orders,
)

T0 = time.time()
MODEL_ID = "Qwen/Qwen2.5-1.5B"
BS = 48
K_PAIRS = 3


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def antithetic(n_parts: int, k_pairs: int, seed: int):
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(k_pairs):
        o = rng.permutation(n_parts).tolist()
        orders.append(o)
        orders.append(list(reversed(o)))
    return orders


def main():
    data = Path("/workspace/data")
    folds_csv = Path("/workspace/folds/folds.csv")
    out = Path("/workspace/out")
    adapter = out / "qwen15b_lora_f0"
    device = torch.device("cuda")

    tr = pd.read_csv(data / "train.csv")
    folds = pd.read_csv(folds_csv)["fold"].to_numpy(int)
    va_idx = np.where(folds == 0)[0]
    log(f"VA-only n_va={len(va_idx)} adapter={adapter}")

    chunks_tr = build_chunks(tr)
    n_parts = len(chunks_tr[0])
    tok = AutoTokenizer.from_pretrained(
        adapter if (adapter / "tokenizer_config.json").exists() else MODEL_ID,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.to(device).eval()
    pad_id = tok.pad_token_id
    prefix_ids = {
        0: tok.encode("0", add_special_tokens=False),
        1: tok.encode("1", add_special_tokens=False),
    }
    log("pretok VA…")
    tok_va = pretok_chunks([chunks_tr[i] for i in va_idx], tok)
    orders = antithetic(n_parts, K_PAIRS, seed=39)
    log(f"K={len(orders)} scoring VA bs={BS}…")
    ll_va = score_ll_orders(model, tok_va, prefix_ids, pad_id, device, orders, bs=BS)
    np.save(out / "qwen15b_f0_ll.npy", ll_va)
    np.save(out / "qwen15b_f0_ll_va.npy", ll_va)
    meta = {
        "n_va": int(len(va_idx)),
        "orders": orders,
        "bs": BS,
        "K": len(orders),
        "elapsed": time.time() - T0,
        "va_only": True,
    }
    (out / "qwen15b_f0_va_only_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"SAVED VA ll {ll_va.shape} elapsed={meta['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
