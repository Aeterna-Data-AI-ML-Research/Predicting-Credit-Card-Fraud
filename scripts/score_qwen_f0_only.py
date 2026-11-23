#!/usr/bin/env python3
"""Score Fold-0 Qwen LoRA adapter → qwen15b_f0_ll.npy (memory-safe)."""
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
MAX_LEN = 144
MODEL_ID = "Qwen/Qwen2.5-1.5B"
BS = 48
K_PAIRS = 1  # K=2 antithetic — speed for last daily slot


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def flag64(df):
    out = np.zeros(len(df), np.int32)
    for i, c in enumerate(FLAGS):
        out |= (df[c].astype(int).to_numpy() << i).astype(np.int32)
    return out


def _col_to_str(s: pd.Series) -> np.ndarray:
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


def pretok_chunks(chunks, tok):
    n, n_parts = len(chunks), len(chunks[0])
    part_ids = []
    for j in range(n_parts):
        texts = [chunks[i][j] for i in range(n)]
        part_ids.append(tok(texts, add_special_tokens=False, padding=False)["input_ids"])
        if (j + 1) % 5 == 0:
            log(f"pretok {j+1}/{n_parts}")
    return [[part_ids[j][i] for j in range(n_parts)] for i in range(n)]


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
                    parts = tok_chunks[i]
                    for j in order:
                        ids.extend(parts[j])
                    batch_ids.append(ids[:MAX_LEN])
                max_l = max(len(x) for x in batch_ids)
                input_ids = torch.full((len(batch_ids), max_l), pad_id, dtype=torch.long, device=device)
                attn = torch.zeros((len(batch_ids), max_l), dtype=torch.long, device=device)
                for bi, ids in enumerate(batch_ids):
                    input_ids[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                    attn[bi, : len(ids)] = 1
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                # memory-safe LL: avoid full log_softmax materialization in bf16 huge copy
                shift_logits = logits[:, :-1, :].float()
                tgt = input_ids[:, 1:]
                selected = shift_logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                lse = torch.logsumexp(shift_logits, dim=-1)
                mask = (tgt != pad_id) & (attn[:, 1:] == 1)
                ll = ((selected - lse) * mask).sum(dim=1)
                out[start:end, ki, label] = ll.float().cpu().numpy()
                del logits, shift_logits, selected, lse
            torch.cuda.empty_cache()
            log(f"scored order={ki} label={label} n={n}")
    return out


def antithetic(n_parts, k_pairs=1, seed=39):
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(k_pairs):
        o = rng.permutation(n_parts).tolist()
        orders.append(o)
        orders.append(list(reversed(o)))
    return orders


def main():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    data = Path("/workspace/data")
    out = Path("/workspace/out")
    adapter = out / "qwen15b_lora_f0"
    assert adapter.exists(), adapter

    tr = pd.read_csv(data / "train.csv")
    te = pd.read_csv(data / "test.csv")
    folds = pd.read_csv("/workspace/folds/folds.csv")["fold"].to_numpy(int)
    va_idx = np.where(folds == 0)[0]
    log(f"n_va={len(va_idx)} n_te={len(te)}")

    log("chunks…")
    chunks_tr = build_chunks(tr)
    chunks_te = build_chunks(te)
    n_parts = len(chunks_tr[0])

    tok = AutoTokenizer.from_pretrained(adapter if (adapter / "tokenizer_config.json").exists() else MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.to(device)
    model.eval()
    pad_id = tok.pad_token_id
    prefix_ids = {0: tok.encode("0", add_special_tokens=False), 1: tok.encode("1", add_special_tokens=False)}

    log("pretok VA…")
    tok_va = pretok_chunks([chunks_tr[i] for i in va_idx], tok)
    log("pretok TEST…")
    tok_te = pretok_chunks(chunks_te, tok)
    orders = antithetic(n_parts, K_PAIRS, seed=39)
    log(f"orders={orders}")

    log("score VA…")
    ll_va = score_ll(model, tok_va, prefix_ids, pad_id, device, orders, bs=BS)
    np.save(out / "qwen15b_f0_ll.npy", ll_va)
    np.save(out / "qwen15b_f0_ll_va.npy", ll_va)
    log("score TEST…")
    ll_te = score_ll(model, tok_te, prefix_ids, pad_id, device, orders, bs=BS)
    np.save(out / "qwen15b_f0_ll_test.npy", ll_te)
    meta = {"n_va": int(len(va_idx)), "n_te": int(len(te)), "orders": orders, "bs": BS, "K": len(orders), "elapsed": time.time() - T0}
    (out / "qwen15b_f0_score_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"DONE {meta}")


if __name__ == "__main__":
    main()
