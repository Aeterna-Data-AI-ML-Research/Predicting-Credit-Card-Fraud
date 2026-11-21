#!/usr/bin/env python3
"""GM V33 — 2-GPU DDP Qwen2.5-1.5B Fold-0 canary with FIXED CE loss + shuffled 6× pos.

Loss: standard causal F.cross_entropy on post-prefix tokens (always > 0).
Data: concat(neg_tr0, repeat(pos_tr0, 6)) then DistributedSampler shuffle each epoch.
Target: ~2370 steps / epoch @ global_bs=256, <0.45s/step, ~15–20 min train + score.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

T0 = time.time()
BUNDLE = Path(os.environ.get("V33_BUNDLE", "/workspace/bundle"))
OUT = Path(os.environ.get("V33_OUT", "/workspace/canary"))
HF = os.environ.get("HF_HOME", "/workspace/hf_cache")
MODEL_ID = os.environ.get("V33_MODEL", "Qwen/Qwen2.5-1.5B")
MAX_EPOCHS = int(os.environ.get("V33_EPOCHS", "1"))  # gate after ep1; ep2 optional
PER_GPU_BS = int(os.environ.get("V33_PER_GPU_BS", "64"))
SCORE_BS = int(os.environ.get("V33_SCORE_BS", "256"))
MAX_LEN = int(os.environ.get("V33_MAX_LEN", "112"))
POS_OS = int(os.environ.get("V33_POS_OVERSAMPLE", "6"))
LR = float(os.environ.get("V33_LR", "3e-4"))
WARMUP_RATIO = float(os.environ.get("V33_WARMUP", "0.04"))
N_ORDERS = int(os.environ.get("V33_SCORE_ORDERS", "4"))
NUM_WORKERS = int(os.environ.get("V33_NUM_WORKERS", "4"))
USE_GRAD_CKPT = os.environ.get("V33_GRAD_CKPT", "0") == "1"


def log(m: str) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"[{time.time()-T0:7.1f}s][DDP] {m}", flush=True)


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
        self.y1 = tokenizer.encode("y: 1\n", add_special_tokens=False)
        self.y0 = tokenizer.encode("y: 0\n", add_special_tokens=False)
        assert len(self.y1) == len(self.y0)
        self.prefix_len = len(self.y1)
        self.used = set(self.y1 + self.y0 + self.nl_id)

    def ensure(self, col_idx, raw):
        key = (col_idx, raw)
        if key not in self.cache:
            ids = self.tok.encode(f"{FEATS[col_idx]}: {raw}", add_special_tokens=False)
            self.used.update(ids)
            self.cache[key] = ids
        return self.cache[key]

    def build_full(self, pieces, y_ids):
        ids = list(y_ids)
        for k, p in enumerate(pieces):
            ids.extend(p)
            if k < len(pieces) - 1:
                ids.extend(self.nl_id)
        return ids


def pretok_cols(df, rt: RowTokens):
    log("pre-tokenize unique cells…")
    col_toks = []
    n = len(df)
    for j, col in enumerate(FEATS):
        uniq = [raw_cell_one(col, v) for v in df[col].unique()]
        mapping = {r: rt.ensure(j, r) for r in uniq}
        vals = df[col].map(lambda v, c=col: raw_cell_one(c, v)).to_numpy()
        arr = np.empty(n, dtype=object)
        for i, r in enumerate(vals):
            arr[i] = mapping[r]
        col_toks.append(arr)
    log(f"cache={len(rt.cache)} |V|={len(rt.used)} prefix_len={rt.prefix_len}")
    return col_toks


def materialize_tensor(row_indices, col_toks, rt, order, y_ids, id_map, max_len=MAX_LEN):
    """Build (N, max_len) int64 input_ids + attention + labels (-100 on prefix/pad)."""
    N = len(row_indices)
    input_ids = np.zeros((N, max_len), dtype=np.int64)
    attn = np.zeros((N, max_len), dtype=np.int64)
    labels = np.full((N, max_len), -100, dtype=np.int64)
    plen = rt.prefix_len
    for i, ri in enumerate(row_indices):
        pieces = [col_toks[int(j)][int(ri)] for j in order]
        ids_orig = rt.build_full(pieces, y_ids)[:max_len]
        mapped = [id_map[t] for t in ids_orig]
        L = len(mapped)
        input_ids[i, :L] = mapped
        attn[i, :L] = 1
        # causal labels: predict token t from 0..t-1; ignore prefix positions in loss
        # labels[k] = input_ids[k] for k >= plen, else -100; shift handled in loss via [:,1:]
        if L > plen:
            labels[i, plen:L] = mapped[plen:L]
    return input_ids, attn, labels


class TensorDS(Dataset):
    def __init__(self, input_ids, attn, labels):
        self.x = input_ids  # np int64 (N, MAX_LEN)
        self.m = attn
        self.y = labels
        self.lens = attn.sum(axis=1).astype(np.int32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        L = int(self.lens[i])
        return self.x[i, :L], self.m[i, :L], self.y[i, :L]


def collate_pad(batch):
    xs, ms, ys = zip(*batch)
    T = max(x.shape[0] for x in xs)
    B = len(xs)
    x = torch.zeros(B, T, dtype=torch.long)
    m = torch.zeros(B, T, dtype=torch.long)
    y = torch.full((B, T), -100, dtype=torch.long)
    for i, (xi, mi, yi) in enumerate(zip(xs, ms, ys)):
        L = xi.shape[0]
        x[i, :L] = torch.from_numpy(np.asarray(xi))
        m[i, :L] = torch.from_numpy(np.asarray(mi))
        y[i, :L] = torch.from_numpy(np.asarray(yi))
    return x, m, y


def causal_ce(logits, labels):
    """logits: (B,T,V) raw; labels: (B,T) with -100 ignore. CE over positions 1..T-1."""
    # standard shift
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def main():
    os.environ.setdefault("HF_HOME", HF)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    local_rank, rank, world = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    log(f"world={world} local_rank={local_rank} bs/gpu={PER_GPU_BS} max_len={MAX_LEN} lr={LR}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    tr = pd.read_csv(BUNDLE / "train.csv.gz")
    fold = np.load(BUNDLE / "fold_id.npy")
    orders = np.load(BUNDLE / "orders_4.npy")
    y = tr["is_fraud"].to_numpy(float)
    wide = np.load(BUNDLE / "wide_lm_oof.npy")
    va_idx = np.where(fold == 0)[0]
    tr_mask = fold != 0
    pos_idx = np.where(tr_mask & (y == 1))[0]
    neg_idx = np.where(tr_mask & (y == 0))[0]
    # FULL fold-0 train coverage: all negs + 6× all pos, THEN shuffle via sampler
    train_indices = np.concatenate([neg_idx, np.repeat(pos_idx, POS_OS)])
    assert len(train_indices) == len(neg_idx) + len(pos_idx) * POS_OS
    pos_rate = (len(pos_idx) * POS_OS) / len(train_indices)
    log(f"tr0_pos={len(pos_idx)} tr0_neg={len(neg_idx)} stream={len(train_indices)} "
        f"pos_rate={pos_rate:.4f} (~9.2% target) va0={len(va_idx)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rt = RowTokens(tok)
    col_toks = pretok_cols(tr, rt)
    used = sorted(rt.used)
    id_map = {tid: i for i, tid in enumerate(used)}
    V = len(used)
    if rank == 0:
        np.save(OUT / "used_token_ids.npy", np.asarray(used, np.int32))
        (OUT / "ddp_audit.json").write_text(json.dumps({
            "per_gpu_bs": PER_GPU_BS, "world": world, "global_bs": PER_GPU_BS * world,
            "grad_accum": 1, "max_len": MAX_LEN, "pos_os": POS_OS,
            "stream_rows": int(len(train_indices)), "pos_rate": pos_rate,
            "steps_per_epoch_approx": int(len(train_indices) / (PER_GPU_BS * world)),
            "V_tab": V, "loss": "F.cross_entropy causal ignore_index=-100",
            "shuffle": "DistributedSampler seed=2026+ep",
        }, indent=2))
    log(f"|V_tab|={V} global_bs={PER_GPU_BS*world} steps≈{len(train_indices)/(PER_GPU_BS*world):.0f}")

    # model
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
    if USE_GRAD_CKPT:
        try:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        log("grad_ckpt ON")
    else:
        if hasattr(backbone, "gradient_checkpointing_disable"):
            backbone.gradient_checkpointing_disable()
        log("grad_ckpt OFF")
    backbone.to(device)
    sub = sub.to(device=device, dtype=torch.float32)

    class Wrap(nn.Module):
        def __init__(self, bb, head):
            super().__init__()
            self.bb = bb
            self.head = head

        def forward(self, input_ids, attention_mask):
            h = self.bb(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.head(h.float())

    model = Wrap(backbone, sub)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Precompute BODY (no y-prefix) mapped ids once per unique tr0 row, then stamp stream
    order0 = orders[0]
    log("precompute body tokens for unique tr0 rows…")
    tr0_rows = np.where(tr_mask)[0]
    body_cache = {}  # row -> list[int] mapped body (+ newlines), no y prefix
    nl_m = [id_map[t] for t in rt.nl_id]
    for k, ri in enumerate(tr0_rows):
        pieces = [col_toks[int(j)][int(ri)] for j in order0]
        body = []
        for pi, p in enumerate(pieces):
            body.extend(id_map[t] for t in p)
            if pi < len(pieces) - 1:
                body.extend(nl_m)
        body_cache[int(ri)] = body
        if rank == 0 and k % 100000 == 0:
            log(f"  body {k}/{len(tr0_rows)}")
    y1_m = [id_map[t] for t in rt.y1]
    y0_m = [id_map[t] for t in rt.y0]
    plen = len(y1_m)
    assert plen == rt.prefix_len

    N = len(train_indices)
    log(f"materialize stream N={N}…")
    input_ids = np.zeros((N, MAX_LEN), dtype=np.int64)
    attn = np.zeros((N, MAX_LEN), dtype=np.int64)
    labels = np.full((N, MAX_LEN), -100, dtype=np.int64)
    for i, ri in enumerate(train_indices):
        ri = int(ri)
        pref = y1_m if y[ri] >= 0.5 else y0_m
        full = (pref + body_cache[ri])[:MAX_LEN]
        L = len(full)
        input_ids[i, :L] = full
        attn[i, :L] = 1
        if L > plen:
            labels[i, plen:L] = full[plen:L]
        if rank == 0 and i > 0 and i % 150000 == 0:
            log(f"  stream {i}/{N}")
    lens = attn.sum(axis=1)
    log(f"seq_len mean={lens.mean():.1f} p95={np.percentile(lens,95):.0f} max={lens.max()} max_len={MAX_LEN}")
    del body_cache

    ds = TensorDS(input_ids, attn, labels)
    steps_per_epoch = len(train_indices) // (PER_GPU_BS * world)
    total_steps = steps_per_epoch * MAX_EPOCHS
    warmup = max(1, int(WARMUP_RATIO * total_steps))

    opt = torch.optim.AdamW(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": LR},
            {"params": [sub.weight], "lr": LR * 0.25},
        ],
        weight_decay=0.01,
    )

    def lr_mult(step):
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * t))

    global_step = 0
    for ep in range(1, MAX_EPOCHS + 1):
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=2026 + ep)
        loader = DataLoader(
            ds, batch_size=PER_GPU_BS, sampler=sampler, num_workers=NUM_WORKERS,
            pin_memory=True, drop_last=True, persistent_workers=NUM_WORKERS > 0,
            collate_fn=collate_pad,
        )
        model.train()
        sampler.set_epoch(ep)
        losses = []
        t_ep = time.time()
        for bi, (bx, bm, by) in enumerate(loader):
            mult = lr_mult(global_step)
            for pg, base in zip(opt.param_groups, [LR, LR * 0.25]):
                pg["lr"] = base * mult
            bx = bx.to(device, non_blocking=True)
            bm = bm.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            logits = model(bx, bm)
            loss = causal_ce(logits, by)
            # HARD GUARD — CE must be positive and finite
            lv = float(loss.item())
            assert torch.isfinite(loss) and 0.05 < lv < 10.0, f"BROKEN LOSS: {lv}"
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(lv)
            global_step += 1
            if rank == 0 and (bi % 50 == 0 or bi == 0):
                dt = time.time() - t_ep
                sps = (bi + 1) / max(1e-6, dt)
                log(f"ep{ep} step{bi}/{len(loader)} loss={lv:.4f} lr={opt.param_groups[0]['lr']:.2e} "
                    f"{sps:.2f} it/s (~{1/sps:.2f}s/step)")
        if rank == 0:
            log(f"ep{ep} DONE mean_loss={float(np.mean(losses)):.4f} "
                f"min={min(losses):.4f} max={max(losses):.4f} time={time.time()-t_ep:.0f}s")
            # save ckpt BEFORE score
            torch.save({"sub": sub.state_dict(), "used": used, "epoch": ep}, OUT / f"ep{ep}_sub.pt")
            torch.save({"sub": sub.state_dict(), "used": used, "epoch": ep}, OUT / "best_sub.pt")
            backbone.save_pretrained(OUT / f"ep{ep}_lora")
            backbone.save_pretrained(OUT / "best_lora")
            (OUT / f"EP{ep}_CKPT_READY").write_text(json.dumps({
                "epoch": ep, "mean_loss": float(np.mean(losses)),
                "steps": len(losses), "sec": time.time() - t_ep,
            }))
        dist.barrier()

        # ---- sharded va0 score (all ranks) ----
        score_va_ddp(model, sub, id_map, col_toks, rt, orders, va_idx, y, wide, device, rank, world, ep)

    if rank == 0:
        (OUT / "TRAIN_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    dist.barrier()
    dist.destroy_process_group()
    log("ALL_DONE")


@torch.inference_mode()
def score_va_ddp(model, sub, id_map, col_toks, rt, orders, va_idx, y, wide, device, rank, world, ep):
    model.eval()
    # unwrap DDP
    net = model.module if isinstance(model, DDP) else model
    # shard va0
    n = len(va_idx)
    chunk = (n + world - 1) // world
    lo, hi = rank * chunk, min(n, (rank + 1) * chunk)
    local = va_idx[lo:hi]
    log(f"score ep{ep} shard[{lo}:{hi}] n={len(local)}")

    def pack(seqs):
        lens = [min(len(s), MAX_LEN) for s in seqs]
        T = max(lens) if lens else 1
        B = len(seqs)
        x = torch.zeros(B, T, dtype=torch.long, device=device)
        m = torch.zeros(B, T, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            s = s[:MAX_LEN]
            mapped = torch.tensor([id_map[int(t)] for t in s], dtype=torch.long, device=device)
            x[i, : len(mapped)] = mapped
            m[i, : len(mapped)] = True
        return x, m

    def seq_ll(h, x, msk, prefix_len):
        logits = net.head(h.float()) if hasattr(net, "head") else sub(h.float())
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = x[:, 1:]
        m2 = msk[:, 1:].clone()
        # only count post-prefix feature tokens
        # positions in shifted seq: index j predicts token j+1; ignore if (j+1) < prefix_len
        pos = torch.arange(m2.shape[1], device=device).unsqueeze(0).expand_as(m2)
        m2 = m2 & ((pos + 1) >= prefix_len)
        gather = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).masked_fill(~m2, 0.0)
        return gather.sum(dim=1)

    use_orders = orders[:N_ORDERS]
    llrs = []
    plen = rt.prefix_len
    for oi, order in enumerate(use_orders):
        scores = np.zeros(len(local), np.float64)
        for s in range(0, len(local), SCORE_BS):
            batch = local[s : s + SCORE_BS]
            pos_seqs, neg_seqs = [], []
            for ii in batch:
                pieces = [col_toks[int(j)][int(ii)] for j in order]
                pos_seqs.append(rt.build_full(pieces, rt.y1))
                neg_seqs.append(rt.build_full(pieces, rt.y0))
            xp, mp = pack(pos_seqs)
            xn, mn = pack(neg_seqs)
            hp = net.bb(input_ids=xp, attention_mask=mp.long()).last_hidden_state
            hn = net.bb(input_ids=xn, attention_mask=mn.long()).last_hidden_state
            llp = seq_ll(hp, xp, mp, plen)
            lln = seq_ll(hn, xn, mn, plen)
            scores[s : s + len(batch)] = (llp - lln).float().cpu().numpy()
        llrs.append(scores)
        log(f"  order{oi} mean_llr={scores.mean():.4f}")
    mean_llr = np.mean(llrs, axis=0).astype(np.float32)
    # gather to rank0
    parts = [None] * world
    dist.all_gather_object(parts, (lo, hi, mean_llr))
    if rank == 0:
        full = np.zeros(n, np.float32)
        for plo, phi, arr in parts:
            full[plo:phi] = arr
        np.save(OUT / f"qwen15b_ep{ep}_va0.npy", full)
        # also r3 from single-order mean for now (full r3 needs all orders gathered — approx via mean)
        auc = float(roc_auc_score(y[va_idx], full))
        w = wide[va_idx]
        base = float(roc_auc_score(y[va_idx], w))
        best_lift = max(
            roc_auc_score(y[va_idx], (1 - wt) * w + wt * full) - base
            for wt in np.linspace(0, 0.3, 31)
        )
        meta = {"epoch": ep, "solo_auc": auc, "lift_e4": best_lift * 1e4, "wide_base": base}
        (OUT / f"qwen15b_ep{ep}_score_meta.json").write_text(json.dumps(meta, indent=2))
        (OUT / f"SCORE_ep{ep}_DONE").write_text(json.dumps(meta))
        log(f"SCORE ep{ep} solo={auc:.6f} lift={best_lift*1e4:.3f}e-4")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
