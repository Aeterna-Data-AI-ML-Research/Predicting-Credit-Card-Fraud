#!/usr/bin/env python3
"""Worker A dual-GPU aware: train on one GPU, save ckpt BEFORE score, 6× pos oversample.

Env:
  V33_CUDA=0
  V33_POS_OVERSAMPLE=6
  V33_NEG_CAP=100000   # neg exposures per epoch (plus 6× all pos)
  V33_TRAIN_BS=32
  V33_SKIP_SCORE=1     # leave scoring to score_va_parallel.py on other GPU
  V33_EPOCHS=2
  V33_RESUME=          # path to best_lora to continue ep2 only
  V33_START_EPOCH=1
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
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = os.environ.get("V33_MODEL", "Qwen/Qwen2.5-1.5B")
MAX_EPOCHS = int(os.environ.get("V33_EPOCHS", "2"))
START_EPOCH = int(os.environ.get("V33_START_EPOCH", "1"))
TRAIN_BS = int(os.environ.get("V33_TRAIN_BS", "32"))
SCORE_BS = int(os.environ.get("V33_SCORE_BS", "96"))
MAX_LEN = int(os.environ.get("V33_MAX_LEN", "320"))
GRAD_CKPT = os.environ.get("V33_GRAD_CKPT", "1") == "1"
POS_OS = int(os.environ.get("V33_POS_OVERSAMPLE", "6"))
NEG_CAP = int(os.environ.get("V33_NEG_CAP", "100000"))
SKIP_SCORE = os.environ.get("V33_SKIP_SCORE", "1") == "1"
N_SCORE_ORDERS = int(os.environ.get("V33_SCORE_ORDERS", "4"))
RESUME = os.environ.get("V33_RESUME", "").strip()
MAX_STEPS = int(os.environ.get("V33_MAX_STEPS", "0"))  # 0=full stream; pos-first so cap still covers 6× pos
POS_FIRST = os.environ.get("V33_POS_FIRST", "1") == "1"


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
        self.cache = {}
        self.nl_id = tokenizer.encode("\n", add_special_tokens=False)
        self.y1 = np.asarray(tokenizer.encode("y: 1\n", add_special_tokens=False), np.uint32)
        self.y0 = np.asarray(tokenizer.encode("y: 0\n", add_special_tokens=False), np.uint32)
        self.used = set(self.y1.tolist() + self.y0.tolist() + self.nl_id)

    def ensure(self, col_idx, raw):
        key = (col_idx, raw)
        if key not in self.cache:
            ids = self.tok.encode(f"{FEATS[col_idx]}: {raw}", add_special_tokens=False)
            self.used.update(ids)
            self.cache[key] = np.asarray(ids, dtype=np.uint32)
        return self.cache[key]

    def assemble(self, pieces, y_prefix):
        parts = [y_prefix]
        nl = np.asarray(self.nl_id, np.uint32)
        for k, p in enumerate(pieces):
            parts.append(p)
            if k < len(pieces) - 1:
                parts.append(nl)
        return np.concatenate(parts)


def pretok_matrix(df, rt: RowTokens):
    log("pre-tokenize…")
    n = len(df)
    col_toks = []
    for j, col in enumerate(FEATS):
        raws = [raw_cell_one(col, v) for v in df[col].unique()]
        mapping = {r: rt.ensure(j, r) for r in raws}
        vals = df[col].map(lambda v, c=col: raw_cell_one(c, v)).to_numpy()
        arr = np.empty(n, dtype=object)
        for i, r in enumerate(vals):
            arr[i] = mapping[r]
        col_toks.append(arr)
    log(f"|V_tab| candidates used={len(rt.used)} cache={len(rt.cache)}")
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
    return gather.masked_fill(~m2, 0.0).sum(dim=1)


def make_epoch_indices(pos_idx, neg_idx, rng, pos_os=POS_OS, neg_cap=NEG_CAP):
    """All positives × pos_os, plus neg_cap negatives.

    If POS_FIRST: stream = shuffle(pos×os) then shuffle(negs) so a MAX_STEPS cap
    still covers every fraud row × os before burning steps on negatives.
    """
    pos = np.tile(pos_idx, pos_os)
    rng.shuffle(pos)
    n_neg = min(neg_cap, max(len(neg_idx), 1))
    if len(neg_idx) >= n_neg:
        neg = rng.choice(neg_idx, size=n_neg, replace=False)
    else:
        neg = rng.choice(neg_idx, size=n_neg, replace=True)
    rng.shuffle(neg)
    if POS_FIRST:
        return np.concatenate([pos, neg])
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return idx


def main():
    os.environ.setdefault("HF_HOME", HF)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} POS_OS={POS_OS} NEG_CAP={NEG_CAP} SKIP_SCORE={SKIP_SCORE} bs={TRAIN_BS} accum=1")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    tr = pd.read_csv(BUNDLE / "train.csv.gz")
    fold = np.load(BUNDLE / "fold_id.npy")
    orders = np.load(BUNDLE / "orders_4.npy")
    y = tr["is_fraud"].to_numpy(float)
    wide = np.load(BUNDLE / "wide_lm_oof.npy")
    va = np.where(fold == 0)[0]
    tr_mask = fold != 0
    pos_idx = np.where(tr_mask & (y == 1))[0]
    neg_idx = np.where(tr_mask & (y == 0))[0]
    log(f"tr0_pos={len(pos_idx)} tr0_neg={len(neg_idx)} va0={len(va)} wide={roc_auc_score(y[va], wide[va]):.6f}")
    log(f"epoch_rows≈{len(pos_idx)*POS_OS + NEG_CAP} pos_exposures={len(pos_idx)*POS_OS} "
        f"eff_bs={TRAIN_BS} steps≈{(len(pos_idx)*POS_OS + NEG_CAP)//TRAIN_BS}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rt = RowTokens(tok)
    col_toks = pretok_matrix(tr, rt)
    used = sorted(rt.used)
    np.save(OUT / "used_token_ids.npy", np.asarray(used, np.int32))
    id_map = {tid: i for i, tid in enumerate(used)}
    V = len(used)
    log(f"|V_tab|={V}")

    if RESUME and (OUT / "best_lora").exists():
        log(f"resume LoRA from {OUT / 'best_lora'}")
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
        )
        backbone = PeftModel.from_pretrained(base.model if hasattr(base, "model") else base, OUT / "best_lora", is_trainable=True)
        # reload sub head
        ck = torch.load(OUT / "best_sub.pt", map_location="cpu")
        H = ck["sub"]["weight"].shape[1]
        used = ck.get("used", used)
        id_map = {tid: i for i, tid in enumerate(used)}
        V = len(used)
        sub = nn.Linear(H, V, bias=False)
        sub.load_state_dict(ck["sub"])
        del base
    else:
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
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
    backbone.to(device)
    sub = sub.to(device=device, dtype=torch.float32)

    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": 2e-4}, {"params": [sub.weight], "lr": 5e-5}],
        weight_decay=0.01,
    )
    order0 = orders[0]
    audit = {
        "per_device_batch_size": TRAIN_BS,
        "grad_accum": 1,
        "effective_batch_size": TRAIN_BS,
        "pos_oversample": POS_OS,
        "neg_cap": NEG_CAP,
        "tr0_pos": int(len(pos_idx)),
        "rows_per_epoch": int(len(pos_idx) * POS_OS + NEG_CAP),
        "pos_exposures_per_epoch": int(len(pos_idx) * POS_OS),
    }
    (OUT / "worker_a_audit.json").write_text(json.dumps(audit, indent=2))
    log(f"AUDIT {audit}")

    def run_epoch(epoch):
        backbone.train()
        sub.train()
        rng = np.random.default_rng(epoch * 17 + 3)
        order = make_epoch_indices(pos_idx, neg_idx, rng)
        n_pos_seen = int(np.isin(order, pos_idx).sum())
        steps = max(1, len(order) // TRAIN_BS)
        if MAX_STEPS > 0:
            steps = min(steps, MAX_STEPS)
        # steps needed to cover all pos×os when POS_FIRST
        pos_steps_need = int(np.ceil(len(pos_idx) * POS_OS / TRAIN_BS))
        log(f"ep{epoch} stream_len={len(order)} pos_in_stream={n_pos_seen} "
            f"steps={steps} pos_steps_need={pos_steps_need} pos_first={POS_FIRST} "
            f"pos_rate={n_pos_seen/len(order):.4f}")
        if POS_FIRST and steps < pos_steps_need:
            log(f"WARNING: MAX_STEPS={MAX_STEPS} < pos_steps_need={pos_steps_need} — raising steps")
            steps = pos_steps_need
        warmup = max(1, int(0.03 * steps))
        losses = []
        n_steps = 0
        for start in range(0, len(order), TRAIN_BS):
            if n_steps >= steps:
                break
            batch_idx = order[start : start + TRAIN_BS]
            if len(batch_idx) < max(4, TRAIN_BS // 4):
                break
            # cosine
            if n_steps < warmup:
                mult = n_steps / warmup
            else:
                t = (n_steps - warmup) / max(1, steps - warmup)
                mult = 0.5 * (1 + np.cos(np.pi * t))
            for pg, base_lr in zip(opt.param_groups, [2e-4, 5e-5]):
                pg["lr"] = base_lr * mult
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
            if n_steps % 25 == 0 or n_steps == 1:
                log(f"ep{epoch} step{n_steps}/{steps} loss={losses[-1]:.4f} lr={opt.param_groups[0]['lr']:.2e}")
        return float(np.mean(losses)), n_steps, n_pos_seen

    results = {}
    for ep in range(START_EPOCH, MAX_EPOCHS + 1):
        log(f"=== EPOCH {ep} TRAIN ===")
        mean_loss, n_steps, n_pos_seen = run_epoch(ep)
        # SAVE BEFORE SCORE — critical for dual-GPU handoff
        torch.save({"sub": sub.state_dict(), "used": used, "epoch": ep}, OUT / f"ep{ep}_sub.pt")
        torch.save({"sub": sub.state_dict(), "used": used, "epoch": ep}, OUT / "best_sub.pt")
        backbone.save_pretrained(OUT / f"ep{ep}_lora")
        backbone.save_pretrained(OUT / "best_lora")
        flag = OUT / f"EP{ep}_CKPT_READY"
        flag.write_text(json.dumps({"epoch": ep, "steps": n_steps, "pos_seen": n_pos_seen, "mean_loss": mean_loss}))
        log(f"wrote {flag} steps={n_steps} pos_seen={n_pos_seen} loss={mean_loss:.4f}")
        results[f"ep{ep}"] = {"steps": n_steps, "pos_seen": n_pos_seen, "mean_loss": mean_loss}

        if not SKIP_SCORE:
            from worker_a_qwen_canary import score_va  # optional local
            mean_llr, r3, std_llr = score_va(backbone, sub, id_map, col_toks, rt, va, orders, device)
            np.save(OUT / f"qwen15b_ep{ep}_va0.npy", mean_llr)

    del opt
    (OUT / "worker_a_meta.json").write_text(json.dumps({"results": results, "audit": audit, "V_tab": V}, indent=2))
    (OUT / "TRAIN_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    log("TRAIN_DONE")


if __name__ == "__main__":
    main()
