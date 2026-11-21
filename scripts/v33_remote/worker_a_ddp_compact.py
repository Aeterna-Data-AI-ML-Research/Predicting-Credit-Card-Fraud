#!/usr/bin/env python3
"""GM V33 — 2-GPU DDP Qwen Fold-0 with COMPACT markers (0% truncation) + FA2 path.

Serialization (~78–88 toks):
  Prefix: " Y1" / " Y0" (2 toks each)
  18 non-flag cols -> single-token markers " A".." R" + exact digit/category body
  6 bool flags -> marker " S" + six '0'/'1' digits
  orders_4 permutes the 19 field blocks

Train: concat(all neg, 6×pos), DistributedSampler shuffle, order = orders_4[i % 4]
Forward: backbone only (no 151k lm_head), attention_mask=None, right-pad eos, labels -100
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
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

T0 = time.time()
BUNDLE = Path(os.environ.get("V33_BUNDLE", "/workspace/bundle"))
OUT = Path(os.environ.get("V33_OUT", "/workspace/canary"))
HF = os.environ.get("HF_HOME", "/workspace/hf_cache")
MODEL_ID = os.environ.get("V33_MODEL", "Qwen/Qwen2.5-1.5B")
MAX_EPOCHS = int(os.environ.get("V33_EPOCHS", "1"))
PER_GPU_BS = int(os.environ.get("V33_PER_GPU_BS", "96"))
SCORE_BS = int(os.environ.get("V33_SCORE_BS", "256"))
POS_OS = int(os.environ.get("V33_POS_OVERSAMPLE", "6"))
LR = float(os.environ.get("V33_LR", "3e-4"))
WARMUP_RATIO = float(os.environ.get("V33_WARMUP", "0.04"))
N_ORDERS = int(os.environ.get("V33_SCORE_ORDERS", "4"))
NUM_WORKERS = int(os.environ.get("V33_NUM_WORKERS", "2"))
USE_GRAD_CKPT = os.environ.get("V33_GRAD_CKPT", "0") == "1"
LEN_SAMPLE = int(os.environ.get("V33_LEN_SAMPLE", "20000"))


def log(m: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[{time.time()-T0:7.1f}s][CMP] {m}", flush=True)


# 18 non-flag + 6 flags packed
NONFLAG = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
]  # 18
FLAGS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]  # 6 -> block " S"
# 19 blocks total; orders_4 was over 24 feats — remap: use orders over 19 blocks
N_BLOCKS = 19  # 18 nonflag + 1 flag-pack

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


def raw_val(col, v):
    if col in CAT_MAP:
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return CAT_MAP[col].get(s, str(v))
    if col in FLAGS or isinstance(v, (bool, np.bool_)):
        return "1" if bool(v) else "0"
    if isinstance(v, float):
        s = f"{v}"
        if "e" in s.lower():
            s = format(v, ".10f").rstrip("0").rstrip(".")
        return s
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def pick_markers(tok):
    """Pick 19 single-token markers ' A'.. and prefixes ' Y0'/' Y1'."""
    markers = []
    # Prefer leading-space letter tokens (Qwen BPE often encodes " A" as 1 tok)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for ch in alphabet:
        cand = f" {ch}"
        ids = tok.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            markers.append((cand, ids[0]))
        if len(markers) >= N_BLOCKS:
            break
    if len(markers) < N_BLOCKS:
        # fallback: uncommon unicode letters with space
        for ch in "αβγδεζηθικλμνξοπρστυφχψω":
            cand = f" {ch}"
            ids = tok.encode(cand, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in {m[1] for m in markers}:
                markers.append((cand, ids[0]))
            if len(markers) >= N_BLOCKS:
                break
    assert len(markers) >= N_BLOCKS, f"only found {len(markers)} single-token markers"
    markers = markers[:N_BLOCKS]

    def find_pref(label):
        for cand in [f" Y{label}", f"Y{label}", f" y{label}"]:
            ids = tok.encode(cand, add_special_tokens=False)
            if len(ids) == 2:
                return cand, ids
        # force 2-token by encoding separately
        a = tok.encode(" Y", add_special_tokens=False)
        b = tok.encode(str(label), add_special_tokens=False)
        ids = a + b
        assert len(ids) >= 2
        return f" Y{label}", ids[:2] if len(ids) > 2 else ids

    y0_s, y0 = find_pref("0")
    y1_s, y1 = find_pref("1")
    # Prefer same length prefixes differing at last token
    assert len(y0) == len(y1), (y0, y1)
    # verify markers are single-token and disjoint from digit chars used in values
    for s, i in markers:
        assert len(tok.encode(s, add_special_tokens=False)) == 1
    return markers, y0_s, y0, y1_s, y1


def block_value_str(df, row_i, block_j):
    """Return string body for block j (0..18)."""
    if block_j < 18:
        col = NONFLAG[block_j]
        return raw_val(col, df[col].iloc[row_i])
    # flag pack: 6 digits
    return "".join(raw_val(c, df[c].iloc[row_i]) for c in FLAGS)


def encode_row(df, row_i, block_order, markers, y_ids, tok, used: set, body_cache: dict):
    """Return list of original token ids for one row with given y prefix + block order."""
    ids = list(y_ids)
    used.update(y_ids)
    for bj in block_order:
        bj = int(bj)
        m_str, m_id = markers[bj]
        ids.append(m_id)
        used.add(m_id)
        body = block_value_str(df, row_i, bj)
        key = (bj, body)
        if key not in body_cache:
            bids = tok.encode(body, add_special_tokens=False)
            body_cache[key] = bids
            used.update(bids)
        else:
            used.update(body_cache[key])
        ids.extend(body_cache[key])
    return ids


def make_block_orders(orders24: np.ndarray) -> np.ndarray:
    """Map 24-col antithetic perms to 19-block perms.

    Blocks 0..17 = NONFLAG cols (match original feat index where possible).
    Block 18 = flag pack (always last in base, but permutes with others).
    We derive 4 perms by projecting orders_4 onto nonflag indices + inserting flag block.
    """
    # original FEATS index for nonflag
    orig_nonflag_idx = list(range(18))  # NONFLAG aligns with first 18 of old 24? 
    # Old FEATS had 13 nums + 5 cats + 6 flags. NONFLAG here = 13+5=18, FLAGS=6.
    # orders_4 permutes 0..23. Project: take order of indices that fall in 0..17 for nonflag,
    # and place flag-block (18) at the position where the first flag index (18) appears in π.
    out = []
    for pi in orders24:
        pi = list(map(int, pi))
        non = [x for x in pi if x < 18]
        # position of flag-pack: median of flag positions in π
        flag_pos = [i for i, x in enumerate(pi) if x >= 18]
        insert_at = int(np.median(flag_pos)) if flag_pos else len(non)
        # build 19-perm: nonflag order as non, insert 18 at insert_at scaled
        insert_at = min(insert_at, 18)
        blocks = list(non)  # values 0..17
        # map insert_at from 24-space to 19-space roughly
        ins = min(18, max(0, int(round(insert_at * 19 / 24))))
        blocks = blocks[:ins] + [18] + blocks[ins:]
        assert len(blocks) == 19 and sorted(blocks) == list(range(19)), blocks
        out.append(blocks)
    return np.asarray(out, dtype=np.int64)


class CompactDS(Dataset):
    def __init__(self, input_ids, lengths, pad_id, prefix_len, max_len):
        self.x = input_ids  # (N, max_len) int64, already right-padded
        self.L = lengths    # (N,) true lengths
        self.pad_id = pad_id
        self.prefix_len = prefix_len
        self.max_len = max_len

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], int(self.L[i])


def collate_fixed(batch, prefix_len):
    xs, Ls = zip(*batch)
    x = torch.stack([torch.from_numpy(np.asarray(xi)) for xi in xs])  # (B, T) already padded
    L = torch.tensor(Ls, dtype=torch.long)
    B, T = x.shape
    labels = x.clone()
    # -100 on prefix and pad region
    pos = torch.arange(T).unsqueeze(0).expand(B, -1)
    labels[pos < prefix_len] = -100
    labels[pos >= L.unsqueeze(1)] = -100
    return x, labels


def causal_ce(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


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
    log(f"world={world} bs/gpu={PER_GPU_BS} grad_ckpt={USE_GRAD_CKPT}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tr = pd.read_csv(BUNDLE / "train.csv.gz")
    fold = np.load(BUNDLE / "fold_id.npy")
    orders24 = np.load(BUNDLE / "orders_4.npy")
    block_orders = make_block_orders(orders24)
    y = tr["is_fraud"].to_numpy(float)
    wide = np.load(BUNDLE / "wide_lm_oof.npy")
    va_idx = np.where(fold == 0)[0]
    tr_mask = fold != 0
    pos_idx = np.where(tr_mask & (y == 1))[0]
    neg_idx = np.where(tr_mask & (y == 0))[0]
    train_indices = np.concatenate([neg_idx, np.repeat(pos_idx, POS_OS)])
    pos_rate = (len(pos_idx) * POS_OS) / len(train_indices)
    log(f"stream={len(train_indices)} pos_rate={pos_rate:.4f} va0={len(va_idx)} block_orders={block_orders.shape}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.eos_token_id
    markers, y0_s, y0_ids, y1_s, y1_ids = pick_markers(tok)
    prefix_len = len(y0_ids)
    log(f"prefix Y0={y0_s}{y0_ids} Y1={y1_s}{y1_ids} markers={[m[0] for m in markers]}")

    # --- length survey on 20k sample (TRUE y prefix, cycling orders) ---
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(train_indices, size=min(LEN_SAMPLE, len(train_indices)), replace=False)
    used = set(y0_ids + y1_ids + [m[1] for m in markers])
    body_cache: dict = {}
    sample_lens = []
    for i, ri in enumerate(sample_idx):
        ri = int(ri)
        yids = y1_ids if y[ri] >= 0.5 else y0_ids
        bord = block_orders[i % 4]
        ids = encode_row(tr, ri, bord, markers, yids, tok, used, body_cache)
        sample_lens.append(len(ids))
    sample_lens = np.asarray(sample_lens)
    max_obs = int(sample_lens.max())
    max_len = int(np.ceil(max_obs / 8) * 8)
    # safety margin +8
    max_len = max(max_len, int(np.ceil((max_obs + 4) / 8) * 8))
    n_trunc_probe = int((sample_lens > max_len).sum())
    log(f"LEN_SURVEY n={len(sample_lens)} min={sample_lens.min()} mean={sample_lens.mean():.1f} "
        f"p95={np.percentile(sample_lens,95):.0f} max={max_obs} -> max_len={max_len} trunc_probe={n_trunc_probe}")
    assert n_trunc_probe == 0, f"truncation in survey: {n_trunc_probe}"

    # materialize full stream with order = orders[i % 4]
    N = len(train_indices)
    log(f"materialize N={N} max_len={max_len}…")
    input_ids = np.full((N, max_len), pad_id, dtype=np.int64)
    lengths = np.zeros(N, dtype=np.int32)
    n_trunc = 0
    for i, ri in enumerate(train_indices):
        ri = int(ri)
        yids = y1_ids if y[ri] >= 0.5 else y0_ids
        bord = block_orders[i % 4]  # ALL 4 antithetic orders in epoch 1
        ids = encode_row(tr, ri, bord, markers, yids, tok, used, body_cache)
        L = len(ids)
        if L > max_len:
            n_trunc += 1
            ids = ids[:max_len]
            L = max_len
        input_ids[i, :L] = ids
        lengths[i] = L
        if rank == 0 and i > 0 and i % 150000 == 0:
            log(f"  {i}/{N}")
    log(f"TRUNCATED_ROWS={n_trunc} / {N}  (MUST BE 0)")
    assert n_trunc == 0, f"FATAL truncation: {n_trunc}"
    used_list = sorted(used)
    id_map = {tid: i for i, tid in enumerate(used_list)}  # orig -> compact
    V = len(used_list)
    # remap input_ids to compact vocab
    log(f"|V_tab|={V} remapping…")
    remap = np.zeros(max(used_list) + 1, dtype=np.int64)
    for tid, ci in id_map.items():
        remap[tid] = ci
    # pad / eos also in map
    if pad_id not in id_map:
        id_map[pad_id] = V
        used_list.append(pad_id)
        V += 1
        remap = np.zeros(max(max(used_list), pad_id) + 1, dtype=np.int64)
        for tid, ci in id_map.items():
            remap[tid] = ci
    compact = remap[input_ids]
    pad_c = id_map[pad_id]
    if rank == 0:
        np.save(OUT / "used_token_ids.npy", np.asarray(used_list, np.int32))
        (OUT / "compact_audit.json").write_text(json.dumps({
            "max_len": max_len, "max_obs": max_obs, "truncated_rows": int(n_trunc),
            "V_tab": V, "prefix_len": prefix_len, "pos_rate": pos_rate,
            "stream": N, "global_bs": PER_GPU_BS * world,
            "steps_approx": N // (PER_GPU_BS * world),
            "markers": [m[0] for m in markers],
            "y0": y0_s, "y1": y1_s,
            "grad_ckpt": USE_GRAD_CKPT, "attention_mask": None,
        }, indent=2))

    # ---- model: backbone ONLY, delete lm_head ----
    full = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    )
    lm_w = full.lm_head.weight.detach().clone()
    backbone = full.model
    # delete huge head so it can never fire
    try:
        del full.lm_head
    except Exception:
        pass
    del full
    torch.cuda.empty_cache()

    H = lm_w.shape[1]
    sub = nn.Linear(H, V, bias=False)
    with torch.no_grad():
        # init from original lm_head rows for used tokens
        rows = []
        for tid in used_list:
            rows.append(lm_w[tid].float())
        sub.weight.copy_(torch.stack(rows, 0))
    sub.weight.requires_grad = True
    del lm_w

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

        def forward(self, input_ids):
            # attention_mask=None unlocks fused causal SDPA; right-pad never attended (causal)
            out = self.bb(input_ids=input_ids, attention_mask=None, use_cache=False)
            h = out.last_hidden_state
            return self.head(h.float())

    model = DDP(Wrap(backbone, sub), device_ids=[local_rank], find_unused_parameters=False)

    ds = CompactDS(compact, lengths, pad_c, prefix_len, max_len)
    steps_per = len(train_indices) // (PER_GPU_BS * world)
    total_steps = steps_per * MAX_EPOCHS
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
            collate_fn=lambda b: collate_fixed(b, prefix_len),
        )
        model.train()
        sampler.set_epoch(ep)
        losses = []
        t_ep = time.time()
        for bi, (bx, by) in enumerate(loader):
            for pg, base in zip(opt.param_groups, [LR, LR * 0.25]):
                pg["lr"] = base * lr_mult(global_step)
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            logits = model(bx)  # no attention_mask
            loss = causal_ce(logits, by)
            lv = float(loss.item())
            assert torch.isfinite(loss) and 0.05 < lv < 10.0, f"BROKEN LOSS: {lv}"
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(lv)
            global_step += 1
            if rank == 0 and bi % 50 == 0:
                dt = time.time() - t_ep
                sps = (bi + 1) / max(1e-6, dt)
                log(f"ep{ep} step{bi}/{len(loader)} loss={lv:.4f} {sps:.2f} it/s (~{1/sps:.2f}s/step) "
                    f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        if rank == 0:
            log(f"ep{ep} DONE mean_loss={float(np.mean(losses)):.4f} time={time.time()-t_ep:.0f}s")
            torch.save({"sub": sub.state_dict(), "used": used_list, "epoch": ep,
                        "max_len": max_len, "prefix_len": prefix_len,
                        "markers": [m[0] for m in markers],
                        "y0": y0_ids, "y1": y1_ids,
                        "block_orders": block_orders.tolist()}, OUT / f"ep{ep}_sub.pt")
            backbone.save_pretrained(OUT / f"ep{ep}_lora")
            (OUT / f"EP{ep}_CKPT_READY").write_text(json.dumps({
                "mean_loss": float(np.mean(losses)), "steps": len(losses),
                "sec": time.time() - t_ep, "max_len": max_len, "truncated": 0,
            }))
        dist.barrier()
        score_va(model, sub, id_map, used_list, markers, y0_ids, y1_ids, block_orders,
                 tr, va_idx, y, wide, tok, device, rank, world, ep, max_len, prefix_len, pad_id)

    if rank == 0:
        (OUT / "TRAIN_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    dist.destroy_process_group()
    log("ALL_DONE")


@torch.inference_mode()
def score_va(model, sub, id_map, used_list, markers, y0_ids, y1_ids, block_orders,
             tr, va_idx, y, wide, tok, device, rank, world, ep, max_len, prefix_len, pad_id):
    model.eval()
    net = model.module if isinstance(model, DDP) else model
    n = len(va_idx)
    chunk = (n + world - 1) // world
    lo, hi = rank * chunk, min(n, (rank + 1) * chunk)
    local = va_idx[lo:hi]
    log(f"score ep{ep} [{lo}:{hi}] n={len(local)}")
    used = set()

    body_cache: dict = {}

    def build(ri, bord, yids):
        ids = encode_row(tr, int(ri), bord, markers, yids, tok, used, body_cache)
        mapped = [id_map[t] for t in ids[:max_len]]
        return mapped

    def pack(seqs):
        T = max(len(s) for s in seqs)
        B = len(seqs)
        x = torch.full((B, T), id_map[pad_id], dtype=torch.long, device=device)
        L = torch.zeros(B, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, device=device)
            L[i] = len(s)
        return x, L

    def seq_ll(x, L):
        h = net.bb(input_ids=x, attention_mask=None, use_cache=False).last_hidden_state
        logits = net.head(h.float())
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = x[:, 1:]
        B, Tm = tgt.shape
        pos = torch.arange(Tm, device=device).unsqueeze(0).expand(B, -1)
        # predict tokens at positions 1..L-1; ignore prefix and pad
        m2 = (pos + 1 >= prefix_len) & (pos + 1 < L.unsqueeze(1))
        g = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).masked_fill(~m2, 0.0)
        return g.sum(dim=1)

    # score all 4 block orders
    llrs = []
    for oi in range(min(N_ORDERS, len(block_orders))):
        bord = block_orders[oi]
        scores = np.zeros(len(local), np.float64)
        for s in range(0, len(local), SCORE_BS):
            batch = local[s : s + SCORE_BS]
            pos_s = [build(ii, bord, y1_ids) for ii in batch]
            neg_s = [build(ii, bord, y0_ids) for ii in batch]
            xp, Lp = pack(pos_s)
            xn, Ln = pack(neg_s)
            scores[s : s + len(batch)] = (seq_ll(xp, Lp) - seq_ll(xn, Ln)).float().cpu().numpy()
        llrs.append(scores)
        log(f"  order{oi} mean_llr={scores.mean():.4f}")
    mean_llr = np.mean(llrs, axis=0).astype(np.float32)
    parts = [None] * world
    dist.all_gather_object(parts, (lo, hi, mean_llr))
    if rank == 0:
        from scipy.special import logit
        full = np.zeros(n, np.float32)
        for plo, phi, arr in parts:
            full[plo:phi] = arr
        np.save(OUT / f"qwen15b_ep{ep}_va0.npy", full)
        auc = float(roc_auc_score(y[va_idx], full))
        w = wide[va_idx]
        base = float(roc_auc_score(y[va_idx], w))
        best_lift = max(roc_auc_score(y[va_idx], (1 - wt) * w + wt * full) - base
                        for wt in np.linspace(0, 0.3, 31))
        meta = {"epoch": ep, "solo_auc": auc, "lift_e4": best_lift * 1e4, "wide_base": base, "max_len": max_len}
        (OUT / f"qwen15b_ep{ep}_score_meta.json").write_text(json.dumps(meta, indent=2))
        (OUT / f"SCORE_ep{ep}_DONE").write_text(json.dumps(meta))
        log(f"SCORE ep{ep} solo={auc:.6f} lift={best_lift*1e4:.3f}e-4")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
