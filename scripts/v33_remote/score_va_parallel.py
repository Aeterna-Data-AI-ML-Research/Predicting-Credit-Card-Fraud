#!/usr/bin/env python3
"""Score va0 LLR from a saved ep{N}_lora + ep{N}_sub.pt checkpoint.

Env:
  V33_EPOCH=1
  V33_SHARD=all|0|1   (0=first half, 1=second half, all=full)
  V33_SCORE_BS=96
  V33_SCORE_ORDERS=4
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import logit
from sklearn.metrics import roc_auc_score

T0 = time.time()
BUNDLE = Path(os.environ.get("V33_BUNDLE", "/workspace/bundle"))
OUT = Path(os.environ.get("V33_OUT", "/workspace/canary"))
HF = os.environ.get("HF_HOME", "/workspace/hf_cache")
MODEL_ID = os.environ.get("V33_MODEL", "Qwen/Qwen2.5-1.5B")
EPOCH = int(os.environ.get("V33_EPOCH", "1"))
SHARD = os.environ.get("V33_SHARD", "all")
SCORE_BS = int(os.environ.get("V33_SCORE_BS", "96"))
MAX_LEN = int(os.environ.get("V33_MAX_LEN", "320"))
N_ORDERS = int(os.environ.get("V33_SCORE_ORDERS", "4"))


def log(m):
    print(f"[{time.time()-T0:7.1f}s][S{EPOCH}/{SHARD}] {m}", flush=True)


# reuse serialization from dual worker via import path
import sys
sys.path.insert(0, "/workspace/scripts")
from worker_a_qwen_dual import (  # noqa
    FEATS, RowTokens, pretok_matrix, build_seq, pack_batch, seq_ll, r3_ranklogit_mean, raw_cell_one
)


@torch.no_grad()
def main():
    os.environ.setdefault("HF_HOME", HF)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tr = pd.read_csv(BUNDLE / "train.csv.gz")
    fold = np.load(BUNDLE / "fold_id.npy")
    orders = np.load(BUNDLE / "orders_4.npy")[:N_ORDERS]
    y = tr["is_fraud"].to_numpy(float)
    wide = np.load(BUNDLE / "wide_lm_oof.npy")
    va = np.where(fold == 0)[0]
    if SHARD == "0":
        va_local = va[: len(va) // 2]
        tag = f"ep{EPOCH}_shard0"
    elif SHARD == "1":
        va_local = va[len(va) // 2 :]
        tag = f"ep{EPOCH}_shard1"
    else:
        va_local = va
        tag = f"ep{EPOCH}_full"
    log(f"scoring n={len(va_local)} / va0={len(va)} orders={len(orders)} bs={SCORE_BS}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rt = RowTokens(tok)
    col_toks = pretok_matrix(tr, rt)

    ckpt_lora = OUT / f"ep{EPOCH}_lora"
    ckpt_sub = OUT / f"ep{EPOCH}_sub.pt"
    assert ckpt_lora.exists() and ckpt_sub.exists(), f"missing {ckpt_lora} or {ckpt_sub}"
    ck = torch.load(ckpt_sub, map_location="cpu")
    used = ck["used"]
    id_map = {tid: i for i, tid in enumerate(used)}
    V = len(used)
    H = ck["sub"]["weight"].shape[1]

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    )
    backbone = PeftModel.from_pretrained(base.model, str(ckpt_lora))
    del base
    sub = nn.Linear(H, V, bias=False)
    sub.load_state_dict(ck["sub"])
    backbone.to(device).eval()
    sub = sub.to(device=device, dtype=torch.float32).eval()

    llrs = []
    for oi, order in enumerate(orders):
        scores = np.zeros(len(va_local), np.float64)
        for start in range(0, len(va_local), SCORE_BS):
            chunk = va_local[start : start + SCORE_BS]
            seqs = [build_seq(col_toks, rt, int(ii), order, rt.y1) for ii in chunk]
            seqs += [build_seq(col_toks, rt, int(ii), order, rt.y0) for ii in chunk]
            x, m = pack_batch(seqs, id_map, device)
            h = backbone(input_ids=x, attention_mask=m.long()).last_hidden_state
            ll = seq_ll(sub, h, x, m)
            n = len(chunk)
            scores[start : start + n] = (ll[:n] - ll[n:]).float().cpu().numpy()
        llrs.append(scores)
        log(f"order {oi} mean_llr={scores.mean():.4f}")
    mean_llr = np.mean(llrs, axis=0).astype(np.float32)
    r3 = r3_ranklogit_mean(llrs).astype(np.float32)
    out_p = OUT / f"qwen15b_{tag}_va0.npy"
    np.save(out_p, mean_llr)
    np.save(OUT / f"qwen15b_{tag}_r3_va0.npy", r3)
    # if full shard, also write canonical name + quick metrics
    if SHARD == "all":
        np.save(OUT / f"qwen15b_ep{EPOCH}_va0.npy", mean_llr)
        auc = float(roc_auc_score(y[va_local], mean_llr))
        w = wide[va_local]
        base = float(roc_auc_score(y[va_local], w))
        best_lift = max(roc_auc_score(y[va_local], (1 - wt) * w + wt * mean_llr) - base for wt in np.linspace(0, 0.3, 31))
        meta = {"epoch": EPOCH, "solo_auc": auc, "lift_e4": best_lift * 1e4, "n": len(va_local)}
        (OUT / f"qwen15b_ep{EPOCH}_score_meta.json").write_text(json.dumps(meta, indent=2))
        log(f"DONE solo={auc:.6f} lift={best_lift*1e4:.3f}e-4 -> {out_p}")
    else:
        # save index map for merge
        np.save(OUT / f"qwen15b_{tag}_idx.npy", va_local.astype(np.int64))
        log(f"DONE shard -> {out_p}")
    (OUT / f"SCORE_{tag}_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


if __name__ == "__main__":
    main()
