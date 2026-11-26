#!/usr/bin/env python3
"""V47 HGDT-LM Fold-0 — Hybrid Generative-Discriminative Dual-Stream Transformer.

Phases: A1 unsupervised next-field on train∪test∪orig (no test labels);
A2 joint generative+discriminative LLR on tr0; A3 K=12 antithetic field-LLR.

NO_SUBMIT. Sovereignty: raw CSVs only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

NO_SUBMIT = True
assert NO_SUBMIT
T0 = time.time()

FEATS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
CONT = {
    "amount_usd", "hours_since_last_txn", "distance_from_home_km",
    "account_balance_usd", "velocity_score", "merchant_risk_score",
}


def log(m: str) -> None:
    print(f"[{time.time()-T0:7.1f}s][HGDT] {m}", flush=True)


def antithetic_orders(n: int = 24, k_pairs: int = 6, seed: int = 47) -> np.ndarray:
    rng = np.random.RandomState(seed)
    orders = []
    for _ in range(k_pairs):
        o = rng.permutation(n)
        orders.append(o)
        orders.append(o[::-1].copy())
    return np.stack(orders, 0)  # (2*k_pairs, n)


class FieldTokenizer:
    """Per-column fine quantile/category IDs + continuous attrs."""

    def __init__(self, n_q: int = 100):
        self.n_q = n_q
        self.edges = {}
        self.cat_maps = {}
        self.offsets = {}
        self.vocab = 2  # PAD=0, UNK=1

    def fit(self, dfs: list[pd.DataFrame]):
        off = 2
        for j, c in enumerate(FEATS):
            self.offsets[c] = off
            col = pd.concat([d[c] for d in dfs])
            if c in CONT or np.issubdtype(col.dtype, np.number):
                qs = np.linspace(0, 1, self.n_q + 1)
                edges = np.unique(np.quantile(col.astype(float).to_numpy(), qs))
                self.edges[c] = edges
                off += max(1, len(edges))
            else:
                cats = {v: i for i, v in enumerate(sorted(col.astype(str).unique()))}
                self.cat_maps[c] = cats
                off += len(cats) + 1
        self.vocab = off
        log(f"tokenizer vocab={self.vocab}")
        return self

    def encode_row(self, row) -> tuple[np.ndarray, np.ndarray]:
        ids = np.zeros(24, np.int64)
        cont = np.zeros((24, 2), np.float32)  # z-ish value + subfloat cents
        for j, c in enumerate(FEATS):
            v = row[c]
            base = self.offsets[c]
            if c in self.edges:
                x = float(v)
                e = self.edges[c]
                b = int(np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 1))
                ids[j] = base + b
                cont[j, 0] = np.tanh(x / (abs(x) + 1.0))  # bounded
                cont[j, 1] = (x % 1.0) if np.isfinite(x) else 0.0
            else:
                cats = self.cat_maps[c]
                ids[j] = base + cats.get(str(v), len(cats))
                cont[j, 0] = 0.0
                cont[j, 1] = 0.0
        return ids, cont

    def encode_frame(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ids = np.zeros((len(df), 24), np.int64)
        cont = np.zeros((len(df), 24, 2), np.float32)
        # vectorized-ish loop in chunks
        records = df[FEATS].to_dict("records")
        for i, row in enumerate(records):
            ids[i], cont[i] = self.encode_row(row)
            if i % 100000 == 0 and i:
                log(f"  encoded {i}/{len(df)}")
        return ids, cont


class HGDT(nn.Module):
    def __init__(self, vocab: int, d_model=512, n_layers=6, n_heads=8, n_fields=24):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.cont_proj = nn.ModuleList([nn.Linear(2, d_model) for _ in range(n_fields)])
        self.y_emb = nn.Embedding(3, d_model)  # 0/1/unlabeled=2
        self.pos = nn.Embedding(n_fields + 1, d_model)  # + y prefix
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab)
        self.d_model = d_model
        self.n_fields = n_fields

    def embed(self, ids, cont, y_tok, order):
        # ids (B,24), cont (B,24,2), order (B,24)
        B = ids.size(0)
        ids_o = torch.gather(ids, 1, order)
        cont_o = torch.gather(cont, 1, order.unsqueeze(-1).expand(-1, -1, 2))
        e = self.tok(ids_o)
        # Vectorized per-field continuous proj in canonical space, then gather by order
        ce_can = torch.stack(
            [self.cont_proj[j](cont[:, j, :]) for j in range(self.n_fields)], dim=1
        )  # (B,24,d)
        ce = torch.gather(ce_can, 1, order.unsqueeze(-1).expand(-1, -1, self.d_model))
        e = e + ce
        y_e = self.y_emb(y_tok).unsqueeze(1)
        x = torch.cat([y_e, e], 1)
        pos = self.pos(torch.arange(self.n_fields + 1, device=ids.device)).unsqueeze(0)
        return x + pos

    def forward(self, ids, cont, y_tok, order, causal=True):
        x = self.embed(ids, cont, y_tok, order)
        if causal:
            T = x.size(1)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            h = self.enc(x, mask=mask)
        else:
            h = self.enc(x)
        return self.head(h)  # (B, T, V) — predict next token ids along sequence


def causal_field_nll(logits, ids, order):
    """Next-field NLL on token ids (positions 1..24 predict ordered field tokens)."""
    # logits: (B,25,V) with pos0=y; targets = ordered ids at pos1..24
    B = ids.size(0)
    tgt = torch.gather(ids, 1, order)  # (B,24)
    pred = logits[:, :-1, :]  # (B,24,V) predict each field given prefix
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="none").view(B, 24)


@torch.no_grad()
def score_llr(model, ids, cont, orders, device, bs=64):
    """orders: (K,24). Returns seq_llr (N,K,2) and field24 (N,24), field_std (N,24)."""
    model.eval()
    N = len(ids)
    K = len(orders)
    seq = np.zeros((N, K, 2), np.float32)
    field = np.zeros((N, 24, K), np.float32)
    ids_t = torch.as_tensor(ids, dtype=torch.long)
    cont_t = torch.as_tensor(cont, dtype=torch.float32)
    for ki, order in enumerate(orders):
        ord_t = torch.as_tensor(order, dtype=torch.long, device=device).unsqueeze(0)
        for yv in (0, 1):
            nlls = []
            field_nll = np.zeros((N, 24), np.float32)
            for i0 in range(0, N, bs):
                sl = slice(i0, i0 + bs)
                b = ids_t[sl].to(device)
                c = cont_t[sl].to(device)
                B = b.size(0)
                o = ord_t.expand(B, -1)
                ytok = torch.full((B,), yv, device=device, dtype=torch.long)
                logits = model(b, c, ytok, o, causal=True)
                nll = causal_field_nll(logits, b, o)  # (B,24) in order space
                # unpermute to canonical
                inv = torch.empty_like(o)
                inv.scatter_(1, o, torch.arange(24, device=device).unsqueeze(0).expand(B, -1))
                nll_can = torch.gather(nll, 1, inv)
                field_nll[i0 : i0 + B] = nll_can.float().cpu().numpy()
                nlls.append(nll.sum(1).float().cpu().numpy())
            seq[:, ki, yv] = -np.concatenate(nlls)  # loglik
            field[:, :, ki] = -field_nll  # loglik per field
        log(f"  scored order {ki+1}/{K}")
    # LLR = logp(y=1) - logp(y=0)
    seq_llr = np.stack([seq[:, :, 0], seq[:, :, 1]], -1)  # keep both; consumer does diff
    # Actually store (N,K,2) as loglik under y=0 and y=1
    field_llr = field[:, :, :].mean(-1)  # placeholder — proper: mean over K of (ll1-ll0)
    # recompute field LLR properly
    # We need per-order y=0 and y=1 — redo light: use last pass structure
    # Simpler API: return seq as (N,K,2) loglik, field_llr computed outside in runner
    return seq, field  # field currently last y only — fix in runner loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--pretrain-steps", type=int, default=4000)
    ap.add_argument("--finetune-epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--k-pairs", type=int, default=6)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--lambda-disc", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    tr = pd.read_csv(args.data / "train.csv")
    te = pd.read_csv(args.data / "test.csv")
    orig = None
    for p in [args.data / "orig20k.csv", Path("/data/orig20k.csv")]:
        if p.exists():
            orig = pd.read_csv(p)
            break
    y = tr["is_fraud"].to_numpy(int)
    folds = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        folds[va] = k
    tr0 = folds != args.fold
    va0 = folds == args.fold
    log(f"tr0={tr0.sum()} va0={va0.sum()}")

    dfs = [tr, te] + ([orig] if orig is not None else [])
    tok = FieldTokenizer(n_q=80).fit(dfs)
    log("encoding frames…")
    id_tr, c_tr = tok.encode_frame(tr)
    id_te, c_te = tok.encode_frame(te)
    id_unlab = [id_tr, id_te]
    c_unlab = [c_tr, c_te]
    if orig is not None:
        id_o, c_o = tok.encode_frame(orig)
        id_unlab.append(id_o)
        c_unlab.append(c_o)
    id_u = np.concatenate(id_unlab)
    c_u = np.concatenate(c_unlab)
    log(f"unlab rows={len(id_u)}")

    model = HGDT(tok.vocab, args.d_model, args.layers, args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # A1 pretrain
    log("Phase A1 unsupervised…")
    model.train()
    rng = np.random.RandomState(0)
    orders_pool = antithetic_orders(24, 4, seed=0)
    for step in range(args.pretrain_steps):
        ix = rng.randint(0, len(id_u), size=args.bs)
        ids = torch.as_tensor(id_u[ix], device=device)
        cont = torch.as_tensor(c_u[ix], device=device)
        o = torch.as_tensor(orders_pool[rng.randint(0, len(orders_pool))], device=device).unsqueeze(0).expand(args.bs, -1)
        ytok = torch.full((args.bs,), 2, device=device, dtype=torch.long)  # unlabeled
        logits = model(ids, cont, ytok, o)
        nll = causal_field_nll(logits, ids, o).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()
        if step % 200 == 0:
            log(f"  A1 step {step} nll={float(nll):.4f}")
    torch.save({"model": model.state_dict(), "vocab": tok.vocab}, args.out / "hgdt_a1.pt")

    # A2 finetune fold0
    log("Phase A2 joint finetune…")
    id_tr0 = torch.as_tensor(id_tr[tr0], dtype=torch.long)
    c_tr0 = torch.as_tensor(c_tr[tr0], dtype=torch.float32)
    y_tr0 = torch.as_tensor(y[tr0], dtype=torch.long)
    pi1 = float(y[tr0].mean())
    prior = math.log(pi1 / (1 - pi1 + 1e-12))
    for ep in range(args.finetune_epochs):
        perm = rng.permutation(len(id_tr0))
        total = 0.0
        for i0 in range(0, len(perm), args.bs):
            ix = perm[i0 : i0 + args.bs]
            ids = id_tr0[ix].to(device)
            cont = c_tr0[ix].to(device)
            yy = y_tr0[ix].to(device)
            o = torch.as_tensor(orders_pool[rng.randint(0, len(orders_pool))], device=device).unsqueeze(0).expand(len(ix), -1)
            # generative NLL under true y
            logits_t = model(ids, cont, yy, o)
            nll_t = causal_field_nll(logits_t, ids, o).sum(1)
            # LLR via both y
            with torch.no_grad():
                pass
            logits1 = model(ids, cont, torch.ones_like(yy), o)
            logits0 = model(ids, cont, torch.zeros_like(yy), o)
            ll1 = -causal_field_nll(logits1, ids, o).sum(1)
            ll0 = -causal_field_nll(logits0, ids, o).sum(1)
            llr = ll1 - ll0 + prior
            bce = F.binary_cross_entropy_with_logits(llr, yy.float())
            loss = nll_t.mean() + args.lambda_disc * bce
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(ix)
        log(f"  A2 ep{ep} loss={total/len(id_tr0):.4f}")
    torch.save({"model": model.state_dict(), "vocab": tok.vocab, "prior": prior}, args.out / "hgdt_a2.pt")

    # A3 score
    log("Phase A3 K-order scoring…")
    orders = antithetic_orders(24, args.k_pairs, seed=47)
    assert len(orders) == 2 * args.k_pairs

    def score_split(ids_np, cont_np):
        N = len(ids_np)
        K = len(orders)
        seq = np.zeros((N, K, 2), np.float32)
        field_ll = np.zeros((N, 24, K, 2), np.float32)
        model.eval()
        with torch.no_grad():
            for ki, order in enumerate(orders):
                ord_t = torch.as_tensor(order, dtype=torch.long, device=device)
                for yv in (0, 1):
                    for i0 in range(0, N, args.bs):
                        sl = slice(i0, min(i0 + args.bs, N))
                        ids = torch.as_tensor(ids_np[sl], device=device)
                        cont = torch.as_tensor(cont_np[sl], device=device)
                        B = ids.size(0)
                        o = ord_t.unsqueeze(0).expand(B, -1)
                        ytok = torch.full((B,), yv, device=device, dtype=torch.long)
                        logits = model(ids, cont, ytok, o)
                        nll = causal_field_nll(logits, ids, o)
                        inv = torch.empty_like(o)
                        inv.scatter_(1, o, torch.arange(24, device=device).unsqueeze(0).expand(B, -1))
                        nll_can = torch.gather(nll, 1, inv)
                        seq[i0 : i0 + B, ki, yv] = (-nll.sum(1)).float().cpu().numpy()
                        field_ll[i0 : i0 + B, :, ki, yv] = (-nll_can).float().cpu().numpy()
                log(f"  order {ki+1}/{K} done for N={N}")
        field_llr = (field_ll[:, :, :, 1] - field_ll[:, :, :, 0]).mean(2)
        field_std = (field_ll[:, :, :, 1] - field_ll[:, :, :, 0]).std(2)
        return seq, field_llr, field_std

    # va0 as oof proxy for fold0 + full test
    seq_va, f_va, s_va = score_split(id_tr[va0], c_tr[va0])
    seq_te, f_te, s_te = score_split(id_te, c_te)
    # pack oof full with zeros elsewhere
    seq_oof = np.zeros((len(y), len(orders), 2), np.float32)
    f_oof = np.zeros((len(y), 24), np.float32)
    s_oof = np.zeros((len(y), 24), np.float32)
    seq_oof[va0] = seq_va
    f_oof[va0] = f_va
    s_oof[va0] = s_va
    np.save(args.out / "hgdt_f0_seq_llr_oof.npy", seq_oof)
    np.save(args.out / "hgdt_f0_seq_llr_test.npy", seq_te)
    np.save(args.out / "hgdt_f0_field24_llr_oof.npy", f_oof)
    np.save(args.out / "hgdt_f0_field24_llr_test.npy", f_te)
    np.save(args.out / "hgdt_f0_field24_std_oof.npy", s_oof)
    np.save(args.out / "hgdt_f0_field24_std_test.npy", s_te)
    meta = {"orders": len(orders), "vocab": tok.vocab, "prior": prior, "va0": int(va0.sum()), "NO_SUBMIT": True}
    (args.out / "hgdt_f0_meta.json").write_text(json.dumps(meta, indent=2))
    log("HGDT Fold0 DONE")


if __name__ == "__main__":
    main()
