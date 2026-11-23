#!/usr/bin/env python3
"""FAST: score TEST only (K=2, bs=64) for last-slot Qwen leap on ATH_v2."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
COLS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score",
    "prior_disputes", "merchant_category", "card_type", "auth_method", "channel",
    "device_type", "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
    "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
MAX_LEN, MODEL_ID, BS = 144, "Qwen/Qwen2.5-1.5B", 64


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def flag64(df):
    out = np.zeros(len(df), np.int32)
    for i, c in enumerate(FLAGS):
        out |= (df[c].astype(int).to_numpy() << i).astype(np.int32)
    return out


def _col_to_str(s):
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
        return s.astype(int).astype(str).to_numpy()
    if pd.api.types.is_float_dtype(s):
        arr = s.to_numpy(dtype=np.float64, copy=False)
        rounded = np.rint(arr)
        is_int = np.abs(arr - rounded) < 1e-9
        out = np.empty(len(arr), dtype=object)
        out[is_int] = rounded[is_int].astype(np.int64).astype(str)
        ni = ~is_int
        if ni.any():
            out[ni] = [f"{v:.6g}" for v in arr[ni]]
        return out
    return s.astype(str).str.replace(" ", "_", regex=False).str[:24].to_numpy()


def build_chunks(df):
    f64 = flag64(df)
    active = [c for c in COLS if c not in FLAGS]
    mats = [(" f64=" + pd.Series(f64).astype(str)).to_numpy()]
    for c in active:
        mats.append((f" {c}=" + pd.Series(_col_to_str(df[c]))).to_numpy())
    return np.column_stack(mats).tolist()


def pretok(chunks, tok):
    n, np_ = len(chunks), len(chunks[0])
    part_ids = []
    for j in range(np_):
        part_ids.append(tok([chunks[i][j] for i in range(n)], add_special_tokens=False, padding=False)["input_ids"])
        if (j + 1) % 5 == 0:
            log(f"pretok {j+1}/{np_}")
    return [[part_ids[j][i] for j in range(np_)] for i in range(n)]


@torch.no_grad()
def score_ll(model, tok_chunks, prefix_ids, pad_id, device, orders, bs=BS):
    model.eval()
    n, K = len(tok_chunks), len(orders)
    out = np.zeros((n, K, 2), np.float32)
    for ki, order in enumerate(orders):
        for label in (0, 1):
            pref = prefix_ids[label]
            for start in range(0, n, bs):
                end = min(start + bs, n)
                batch_ids = []
                for i in range(start, end):
                    ids = list(pref)
                    for j in order:
                        ids.extend(tok_chunks[i][j])
                    batch_ids.append(ids[:MAX_LEN])
                max_l = max(len(x) for x in batch_ids)
                input_ids = torch.full((len(batch_ids), max_l), pad_id, dtype=torch.long, device=device)
                attn = torch.zeros((len(batch_ids), max_l), dtype=torch.long, device=device)
                for bi, ids in enumerate(batch_ids):
                    input_ids[bi, : len(ids)] = torch.tensor(ids, device=device)
                    attn[bi, : len(ids)] = 1
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                shift = logits[:, :-1, :].float()
                tgt = input_ids[:, 1:]
                selected = shift.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                lse = torch.logsumexp(shift, dim=-1)
                mask = (tgt != pad_id) & (attn[:, 1:] == 1)
                out[start:end, ki, label] = ((selected - lse) * mask).sum(1).cpu().numpy()
                del logits, shift
            torch.cuda.empty_cache()
            log(f"order={ki} label={label}")
    return out


def main():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path("/workspace/out")
    adapter = out / "qwen15b_lora_f0"
    te = pd.read_csv("/workspace/data/test.csv")
    log(f"n_te={len(te)}")
    chunks = build_chunks(te)
    n_parts = len(chunks[0])
    tok = AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, str(adapter)).to(device).eval()
    pad_id = tok.pad_token_id
    prefix_ids = {0: tok.encode("0", add_special_tokens=False), 1: tok.encode("1", add_special_tokens=False)}
    log("pretok test…")
    tok_te = pretok(chunks, tok)
    rng = np.random.default_rng(39)
    o = rng.permutation(n_parts).tolist()
    orders = [o, list(reversed(o))]
    ll = score_ll(model, tok_te, prefix_ids, pad_id, device, orders, bs=BS)
    np.save(out / "qwen15b_f0_ll_test.npy", ll)
    # also mean-diff score vector for quick CSV on pod
    r = (ll[:, :, 1] - ll[:, :, 0]).mean(1)
    np.save(out / "qwen15b_f0_R_mean_diff_test.npy", r.astype(np.float32))
    (out / "qwen15b_f0_fast_meta.json").write_text(json.dumps({"orders": orders, "elapsed": time.time() - T0}, indent=2))
    log(f"DONE shape={ll.shape} elapsed={time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
