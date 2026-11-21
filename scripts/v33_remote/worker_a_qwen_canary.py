#!/usr/bin/env python3
"""Worker A — Qwen2.5-1.5B active-subvocab fp32 LLR LoRA Fold-0 canary (va0 only)."""
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
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = os.environ.get("V33_MODEL", "Qwen/Qwen2.5-1.5B")
MAX_EPOCHS = int(os.environ.get("V33_EPOCHS", "2"))
TRAIN_BS = int(os.environ.get("V33_TRAIN_BS", "16"))
SCORE_BS = int(os.environ.get("V33_SCORE_BS", "64"))
MAX_LEN = int(os.environ.get("V33_MAX_LEN", "320"))
TRAIN_SUB = int(os.environ.get("V33_TRAIN_SUB", "0"))
GRAD_CKPT = os.environ.get("V33_GRAD_CKPT", "1") == "1"
MAX_STEPS = int(os.environ.get("V33_MAX_STEPS", "0"))  # per epoch; 0=full
N_SCORE_ORDERS = int(os.environ.get("V33_SCORE_ORDERS", "4"))


def log(m):
    print(f"[{time.time()-T0:7.1f}s][A] {m}", flush=True)


CAT_MAP = {
    "merchant_category": {
        "online_retail": "Online Retail", "grocery": "Grocery", "travel": "Travel",
        "crypto": "Crypto", "gambling": "Gambling", "entertainment": "Entertainment",
        "restaurant": "Restaurant", "gas": "Gas", "utilities": "Utilities", "other": "Other",
    },
    "card_type": {"credit": "Credit", "debit": "Debit", "prepaid": "Prepaid"},
    "auth_method": {
        "pin": "PIN", "chip": "Chip", "swipe": "Swipe", "online": "Online",
        "contactless": "Contactless", "no_auth": "No Auth", "biometric": "Biometric",
    },
    "channel": {"pos": "POS Term", "online": "Online", "atm": "ATM", "mobile": "Mobile"},
    "device_type": {"mobile": "Mobile", "desktop": "Desktop", "pos": "POS", "other": "Other"},
}

FEATS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
assert len(FEATS) == 24


def raw_cell_one(col, v):
    if col in CAT_MAP:
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return CAT_MAP[col].get(s, str(v))
    if isinstance(v, (bool, np.bool_)) or col.startswith("is_") or col in {
        "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
        "is_foreign_transaction", "is_new_merchant",
    }:
        return "1" if bool(v) else "0"
    if isinstance(v, float):
        s = f"{v}"
        if "e" in s.lower():
            s = format(v, ".10f").rstrip("0").rstrip(".")
        return s
    return str(v)


def rank_logit(x):
    r = pd.Series(x).rank(method="average").to_numpy()
    u = (r - 0.5) / len(r)
    return logit(np.clip(u, 1e-6, 1 - 1e-6))


def r3_ranklogit_mean(llrs):
    return np.mean([rank_logit(a) for a in llrs], axis=0)


class RowTokens:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.cache = {}  # (col_idx, raw_str) -> np.uint32
        self.nl_id = tokenizer.encode("\n", add_special_tokens=False)
        self.y1 = np.asarray(tokenizer.encode("y: 1\n", add_special_tokens=False), np.uint32)
        self.y0 = np.asarray(tokenizer.encode("y: 0\n", add_special_tokens=False), np.uint32)
        assert len(self.y1) == len(self.y0)
        self.used = set(self.y1.tolist() + self.y0.tolist() + self.nl_id)

    def ensure(self, col_idx, raw):
        key = (col_idx, raw)
        if key not in self.cache:
            col = FEATS[col_idx]
            ids = self.tok.encode(f"{col}: {raw}", add_special_tokens=False)
            self.used.update(ids)
            self.cache[key] = np.asarray(ids, dtype=np.uint32)
        return self.cache[key]

    def assemble(self, pieces, y_prefix):
        # pieces: list of uint32 arrays for body lines (already include "col: val")
        parts = [y_prefix]
        nl = np.asarray(self.nl_id, np.uint32)
        for k, p in enumerate(pieces):
            parts.append(p)
            if k < len(pieces) - 1:
                parts.append(nl)
        return np.concatenate(parts)


def pretok_matrix(df, rt: RowTokens):
    """Pre-tokenize every cell once; return list of 24 arrays of object-dtype token arrays per row."""
    log("pre-tokenize unique cells + row index…")
    n = len(df)
    # per-column: map raw->tokens, then array of references per row
    col_toks = []
    for j, col in enumerate(FEATS):
        series = df[col]
        # unique raw strings
        raws = [raw_cell_one(col, v) for v in series.unique()]
        mapping = {r: rt.ensure(j, r) for r in raws}
        # per-row token arrays (object)
        arr = np.empty(n, dtype=object)
        # vectorized via map on values
        vals = series.map(lambda v, c=col: raw_cell_one(c, v)).to_numpy()
        for i, r in enumerate(vals):
            arr[i] = mapping[r]
        col_toks.append(arr)
        if j % 6 == 0:
            log(f"  col {j}/{len(FEATS)} cache={len(rt.cache)}")
    log(f"cache keys={len(rt.cache)} used_tokens={len(rt.used)}")
    return col_toks


def build_seq(col_toks, rt, row_i, order, y_prefix):
    pieces = [col_toks[int(j)][row_i] for j in order]
    return rt.assemble(pieces, y_prefix)


def pack_batch(seqs, id_map, device, max_len=MAX_LEN):
    lens = [min(len(s), max_len) for s in seqs]
    T = max(lens) if lens else 1
    B = len(seqs)
    x = torch.zeros(B, T, dtype=torch.long, device=device)
    m = torch.zeros(B, T, dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        s = s[:max_len]
        mapped = torch.tensor([id_map[int(t)] for t in s], dtype=torch.long, device=device)
        x[i, : len(mapped)] = mapped
        m[i, : len(mapped)] = True
    return x, m


def seq_nll(sub, h, x, msk):
    logits = sub(h[:, :-1].float())
    tgt = x[:, 1:]
    m2 = msk[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    nllp = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    nllp = nllp.masked_fill(~m2, 0.0)
    return nllp.sum(dim=1) / m2.sum(dim=1).clamp_min(1)


def seq_ll(sub, h, x, msk):
    logits = sub(h[:, :-1].float())
    tgt = x[:, 1:]
    m2 = msk[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    gather = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    gather = gather.masked_fill(~m2, 0.0)
    return gather.sum(dim=1)


@torch.no_grad()
def score_va(model, sub_head, id_map, col_toks, rt, va_idx, orders, device):
    model.eval()
    sub_head.eval()
    use_orders = orders[:N_SCORE_ORDERS]
    llrs = []
    for oi, order in enumerate(use_orders):
        scores = np.zeros(len(va_idx), np.float64)
        for start in range(0, len(va_idx), SCORE_BS):
            chunk = va_idx[start : start + SCORE_BS]
            seqs = []
            for ii in chunk:
                seqs.append(build_seq(col_toks, rt, int(ii), order, rt.y1))
            for ii in chunk:
                seqs.append(build_seq(col_toks, rt, int(ii), order, rt.y0))
            x, m = pack_batch(seqs, id_map, device)
            h = model(input_ids=x, attention_mask=m.long()).last_hidden_state
            ll = seq_ll(sub_head, h, x, m)
            n = len(chunk)
            scores[start : start + n] = (ll[:n] - ll[n:]).float().cpu().numpy()
        llrs.append(scores)
        log(f"  score order {oi}/{len(use_orders)} mean_llr={scores.mean():.4f}")
    mean_llr = np.mean(llrs, axis=0).astype(np.float32)
    r3 = r3_ranklogit_mean(llrs).astype(np.float32)
    return mean_llr, r3, float(np.std(mean_llr))


def main():
    os.environ.setdefault("HF_HOME", HF)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} model={MODEL_ID}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tr = pd.read_csv(BUNDLE / "train.csv.gz")
    fold = np.load(BUNDLE / "fold_id.npy")
    orders = np.load(BUNDLE / "orders_4.npy")
    y = tr["is_fraud"].to_numpy(float)
    wide = np.load(BUNDLE / "wide_lm_oof.npy")
    va = np.where(fold == 0)[0]
    tr_idx = np.where(fold != 0)[0]
    if TRAIN_SUB > 0:
        rng = np.random.default_rng(0)
        pos = tr_idx[y[tr_idx] == 1]
        neg = tr_idx[y[tr_idx] == 0]
        take = min(max(TRAIN_SUB - len(pos), 0), len(neg))
        tr_idx = np.concatenate([pos, rng.choice(neg, size=take, replace=False)])
        rng.shuffle(tr_idx)
        log(f"TRAIN_SUB active n={len(tr_idx)}")
    log(f"tr={len(tr_idx)} va={len(va)} wide_auc={roc_auc_score(y[va], wide[va]):.6f}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rt = RowTokens(tok)
    col_toks = pretok_matrix(tr, rt)
    used = sorted(rt.used)
    log(f"|V_tab|={len(used)}")
    np.save(OUT / "used_token_ids.npy", np.asarray(used, np.int32))
    id_map = {tid: i for i, tid in enumerate(used)}
    V = len(used)

    log("load backbone bf16…")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    backbone = base.model
    lm_w = base.lm_head.weight.detach()
    H = lm_w.shape[1]
    sub = nn.Linear(H, V, bias=False)
    with torch.no_grad():
        sub.weight.copy_(lm_w[used].float())
    sub.weight.requires_grad = True
    del base

    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    backbone = get_peft_model(backbone, lora)
    if GRAD_CKPT:
        try:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        log("gradient checkpointing ON")
    backbone.to(device)
    sub = sub.to(device=device, dtype=torch.float32)
    log(f"trainable ready |V_tab|={V} H={H} train_bs={TRAIN_BS} score_bs={SCORE_BS}")

    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": 2e-4}, {"params": [sub.weight], "lr": 5e-5}],
        weight_decay=0.01,
    )
    steps_per = max(1, len(tr_idx) // TRAIN_BS)
    if MAX_STEPS > 0:
        steps_per = min(steps_per, MAX_STEPS)
    total_steps = steps_per * MAX_EPOCHS
    warmup = max(1, int(0.03 * total_steps))
    order0 = orders[0]
    log(f"steps_per_epoch={steps_per} max_steps={MAX_STEPS} score_orders={N_SCORE_ORDERS}")

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * t))

    def run_epoch(epoch):
        backbone.train()
        sub.train()
        rng = np.random.default_rng(epoch)
        order = tr_idx.copy()
        rng.shuffle(order)
        step0 = (epoch - 1) * steps_per
        losses = []
        n_steps = 0
        for bi, start in enumerate(range(0, len(order), TRAIN_BS)):
            if n_steps >= steps_per:
                break
            step = step0 + n_steps
            mult = lr_at(step)
            for pg, base_lr in zip(opt.param_groups, [2e-4, 5e-5]):
                pg["lr"] = base_lr * mult
            batch_idx = order[start : start + TRAIN_BS]
            # pack pos+neg in one forward for throughput
            seqs = [build_seq(col_toks, rt, int(ii), order0, rt.y1) for ii in batch_idx]
            seqs += [build_seq(col_toks, rt, int(ii), order0, rt.y0) for ii in batch_idx]
            x, m = pack_batch(seqs, id_map, device)
            h = backbone(input_ids=x, attention_mask=m.long()).last_hidden_state
            nll = seq_nll(sub, h, x, m)
            n = len(batch_idx)
            loss = (nll[:n] - 0.5 * nll[n:]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + [sub.weight], 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            n_steps += 1
            if n_steps % 20 == 0 or n_steps == 1:
                log(f"ep{epoch} step{n_steps}/{steps_per} loss={losses[-1]:.4f} lr={opt.param_groups[0]['lr']:.2e}")
        return float(np.mean(losses))

    best = {"epoch": 0, "lift": -1e9, "auc": 0.0}
    results = {}
    for ep in range(1, MAX_EPOCHS + 1):
        log(f"=== EPOCH {ep} ===")
        run_epoch(ep)
        log(f"score va0 at epoch {ep}")
        mean_llr, r3, std_llr = score_va(backbone, sub, id_map, col_toks, rt, va, orders, device)
        np.save(OUT / f"qwen15b_ep{ep}_va0.npy", mean_llr)
        np.save(OUT / f"qwen15b_ep{ep}_r3_va0.npy", r3)
        auc = float(roc_auc_score(y[va], mean_llr))
        w = wide[va]
        base_auc = float(roc_auc_score(y[va], w))
        lifts = [roc_auc_score(y[va], (1 - wt) * w + wt * mean_llr) - base_auc for wt in np.linspace(0, 0.3, 31)]
        lift = float(max(lifts))
        results[f"ep{ep}"] = {"auc": auc, "lift": lift, "std_llr": std_llr}
        log(f"ep{ep} solo_auc={auc:.6f} best_lift={lift*1e4:.3f}e-4 std={std_llr:.4f}")
        if lift > best["lift"] or (abs(lift - best["lift"]) < 1e-9 and auc > best["auc"]):
            best = {"epoch": ep, "lift": lift, "auc": auc}
            torch.save({"sub": sub.state_dict(), "used": used}, OUT / "best_sub.pt")
            backbone.save_pretrained(OUT / "best_lora")

    del opt
    meta = {"best": best, "results": results, "V_tab": V, "model": MODEL_ID}
    (OUT / "worker_a_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"DONE best={best}")


if __name__ == "__main__":
    main()
