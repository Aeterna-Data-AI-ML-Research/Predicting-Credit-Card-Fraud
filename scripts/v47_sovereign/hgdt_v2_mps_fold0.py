#!/usr/bin/env python3
"""V47.2 HGDT-LM v2 — Pure generative Target-First on Apple MPS Fold-0.

T=25 = [CLS_y, col_0..col_23], causal continuous residual shifted by 1,
pure next-token CE (no discriminative shortcut). Throughput gate 90 min.
NO_SUBMIT=True · $0 RunPod.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

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
    print(f"[{time.time()-T0:7.1f}s][HGDT2] {m}", flush=True)


def antithetic_orders(n=24, k_pairs=4, seed=47):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(k_pairs):
        o = rng.permutation(n)
        out.append(o)
        out.append(o[::-1].copy())
    return np.stack(out, 0)


class FieldTok:
    def __init__(self, n_q=100):
        self.n_q = n_q
        self.edges = {}
        self.cat_maps = {}
        self.offsets = {}
        self.vocab = 4  # PAD=0, UNK=1, Y0=2, Y1=3

    def fit(self, dfs):
        off = 4
        for c in FEATS:
            self.offsets[c] = off
            col = pd.concat([d[c] for d in dfs])
            is_num = c in CONT
            if not is_num:
                try:
                    is_num = np.issubdtype(np.dtype(col.dtype), np.number)
                except TypeError:
                    # pandas StringDtype etc.
                    is_num = False
                if not is_num:
                    try:
                        pd.to_numeric(col, errors="raise")
                        is_num = True
                    except Exception:
                        is_num = False
            if is_num:
                qs = np.linspace(0, 1, self.n_q + 1)
                edges = np.unique(np.quantile(col.astype(float).to_numpy(), qs))
                self.edges[c] = edges
                off += max(1, len(edges))
            else:
                cats = {v: i for i, v in enumerate(sorted(col.astype(str).unique()))}
                self.cat_maps[c] = cats
                off += len(cats) + 1
        self.vocab = off
        log(f"vocab={self.vocab}")
        return self

    def encode(self, df: pd.DataFrame):
        n = len(df)
        ids = np.zeros((n, 24), np.int64)
        cont = np.zeros((n, 24, 3), np.float32)  # z_val, cents, mantissa
        for j, c in enumerate(FEATS):
            base = self.offsets[c]
            if c in self.edges:
                x = df[c].to_numpy(float)
                e = self.edges[c]
                b = np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 1)
                ids[:, j] = base + b
                cont[:, j, 0] = np.tanh(x / (np.abs(x) + 1.0)).astype(np.float32)
                cont[:, j, 1] = (x % 1.0).astype(np.float32)
                # mantissa first digit of abs scientific-ish
                ax = np.abs(x) + 1e-12
                mant = ax / (10.0 ** np.floor(np.log10(ax)))
                cont[:, j, 2] = ((mant - 1.0) / 9.0).astype(np.float32)
            else:
                cats = self.cat_maps[c]
                s = df[c].astype(str).to_numpy()
                ids[:, j] = base + np.array([cats.get(v, len(cats)) for v in s], np.int64)
        return ids, cont


class HGDTv2(nn.Module):
    def __init__(self, vocab, d_model=384, n_layers=6, n_heads=6, n_fields=24):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.y_emb = nn.Embedding(3, d_model)  # 0/1/unlab=2
        self.pos = nn.Embedding(n_fields + 1, d_model)
        self.cont_proj = nn.ModuleList([nn.Linear(3, d_model) for _ in range(n_fields)])
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab)
        self.n_fields = n_fields
        self.d_model = d_model

    def forward(self, ids, cont, y_tok, order):
        # ids (B,24), cont (B,24,3), order (B,24)
        B = ids.size(0)
        ids_o = torch.gather(ids, 1, order)
        # Continuous dual-stream with causal shift-1: input at field position t
        # gets cont of ordered field t-1 only (never the target field's continuous).
        ce_can = torch.stack([self.cont_proj[j](cont[:, j, :]) for j in range(self.n_fields)], dim=1)
        ce = torch.gather(ce_can, 1, order.unsqueeze(-1).expand(-1, -1, self.d_model))
        ce_shift = torch.zeros_like(ce)
        ce_shift[:, 1:, :] = ce[:, :-1, :]
        e = self.tok(ids_o) + ce_shift
        y_e = self.y_emb(y_tok).unsqueeze(1)
        x = torch.cat([y_e, e], 1)
        x = x + self.pos(torch.arange(self.n_fields + 1, device=ids.device)).unsqueeze(0)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.enc(x, mask=mask)
        return self.head(h)  # (B,25,V)


def field_nll(logits, ids, order):
    """NLL of ordered field tokens at positions 1..24 (given causal prefix)."""
    tgt = torch.gather(ids, 1, order)
    pred = logits[:, :-1, :]
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="none").view(ids.size(0), 24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eta-cap-min", type=float, default=90.0)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--pretrain-epochs", type=int, default=1)
    ap.add_argument("--finetune-epochs", type=int, default=3)
    ap.add_argument("--k-pairs", type=int, default=4)
    ap.add_argument("--bench-batches", type=int, default=200)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device={device}")

    tr = pd.read_csv(args.data / "train.csv")
    te = pd.read_csv(args.data / "test.csv")
    orig = None
    for p in (args.data / "orig20k.csv", Path("/tmp/s6e11_v39_ssd/data/orig20k.csv"), Path("/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/original/credit_card_fraud_2026.csv")):
        if p.exists():
            orig = pd.read_csv(p)
            break
    y = tr["is_fraud"].to_numpy(int)
    folds = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        folds[va] = k
    tr0 = folds != 0
    va0 = folds == 0
    log(f"tr0={tr0.sum()} va0={va0.sum()}")

    dfs = [tr, te] + ([orig] if orig is not None else [])
    tok = FieldTok(n_q=100).fit(dfs)
    log("encoding…")
    id_tr, c_tr = tok.encode(tr)
    id_te, c_te = tok.encode(te)
    id_u = [id_tr, id_te]
    c_u = [c_tr, c_te]
    if orig is not None:
        ido, co = tok.encode(orig)
        id_u.append(ido)
        c_u.append(co)
    id_u = np.concatenate(id_u)
    c_u = np.concatenate(c_u)
    log(f"unlab={len(id_u)}")

    model = HGDTv2(tok.vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    orders_pool = antithetic_orders(24, 4, seed=0)
    rng = np.random.RandomState(0)

    # Throughput gate
    log(f"throughput bench {args.bench_batches} batches…")
    model.train()
    t_b = time.time()
    for bi in range(args.bench_batches):
        ix = rng.randint(0, len(id_u), size=args.bs)
        ids = torch.as_tensor(id_u[ix], device=device)
        cont = torch.as_tensor(c_u[ix], device=device)
        o = torch.as_tensor(orders_pool[rng.randint(0, len(orders_pool))], device=device).unsqueeze(0).expand(args.bs, -1)
        ytok = torch.full((args.bs,), 2, device=device, dtype=torch.long)
        logits = model(ids, cont, ytok, o)
        loss = field_nll(logits, ids, o).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
    dt = time.time() - t_b
    rows_sec = args.bench_batches * args.bs / max(dt, 1e-6)
    # ETA: A1 1ep over unlab + A2 3ep over tr0 + score K=8 * (va0+test) * 2 y
    steps_a1 = math.ceil(len(id_u) / args.bs) * args.pretrain_epochs
    steps_a2 = math.ceil(int(tr0.sum()) / args.bs) * args.finetune_epochs
    score_rows = (int(va0.sum()) + len(id_te)) * (2 * args.k_pairs) * 2  # orders * y0/y1
    score_steps = math.ceil(score_rows / args.bs)
    total_steps = steps_a1 + steps_a2 + score_steps
    eta_min = (total_steps / max(args.bench_batches / dt, 1e-9)) / 60.0
    # better: use rows/sec
    train_rows = len(id_u) * args.pretrain_epochs + int(tr0.sum()) * args.finetune_epochs
    score_row_passes = (int(va0.sum()) + len(id_te)) * (2 * args.k_pairs) * 2
    eta_min = (train_rows + score_row_passes) / rows_sec / 60.0
    bench = {"rows_sec": rows_sec, "eta_min": eta_min, "device": str(device), "bs": args.bs}
    (args.out / "HGDT_V2_THROUGHPUT.json").write_text(json.dumps(bench, indent=2))
    log(f"rows/sec={rows_sec:.1f} ETA={eta_min:.1f} min (cap={args.eta_cap_min})")
    if eta_min > args.eta_cap_min:
        log("STOP — ETA above cap. Packaging note: free Kaggle 2×T4 optional.")
        (args.out / "HGDT_V2_SKIPPED.json").write_text(json.dumps({**bench, "skipped": True, "NO_SUBMIT": True}, indent=2))
        return

    # A1
    log("Phase A1 unsupervised…")
    n_steps = math.ceil(len(id_u) / args.bs) * args.pretrain_epochs
    for step in range(n_steps):
        ix = rng.randint(0, len(id_u), size=args.bs)
        ids = torch.as_tensor(id_u[ix], device=device)
        cont = torch.as_tensor(c_u[ix], device=device)
        o = torch.as_tensor(orders_pool[rng.randint(0, len(orders_pool))], device=device).unsqueeze(0).expand(args.bs, -1)
        ytok = torch.full((args.bs,), 2, device=device, dtype=torch.long)
        loss = field_nll(model(ids, cont, ytok, o), ids, o).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            log(f"  A1 {step}/{n_steps} loss={float(loss):.4f}")
    torch.save({"model": model.state_dict(), "vocab": tok.vocab}, args.out / "hgdt_v2_a1.pt")

    # A2
    log("Phase A2 target-first FT…")
    id_tr0 = id_tr[tr0]
    c_tr0 = c_tr[tr0]
    y_tr0 = y[tr0]
    n_steps = math.ceil(len(id_tr0) / args.bs) * args.finetune_epochs
    for step in range(n_steps):
        ix = rng.randint(0, len(id_tr0), size=args.bs)
        ids = torch.as_tensor(id_tr0[ix], device=device)
        cont = torch.as_tensor(c_tr0[ix], device=device)
        yy = torch.as_tensor(y_tr0[ix], device=device, dtype=torch.long)
        o = torch.as_tensor(orders_pool[rng.randint(0, len(orders_pool))], device=device).unsqueeze(0).expand(args.bs, -1)
        loss = field_nll(model(ids, cont, yy, o), ids, o).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            log(f"  A2 {step}/{n_steps} loss={float(loss):.4f}")
    torch.save({"model": model.state_dict(), "vocab": tok.vocab}, args.out / "hgdt_v2_a2.pt")

    # A3 score
    orders = antithetic_orders(24, args.k_pairs, seed=47)
    log(f"Phase A3 score K={len(orders)}")

    def score_split(ids_np, cont_np):
        N = len(ids_np)
        K = len(orders)
        seq = np.zeros((N, K, 2), np.float32)
        field = np.zeros((N, 24, K, 2), np.float32)
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
                        nll = field_nll(model(ids, cont, ytok, o), ids, o)
                        inv = torch.empty_like(o)
                        inv.scatter_(1, o, torch.arange(24, device=device).unsqueeze(0).expand(B, -1))
                        nll_can = torch.gather(nll, 1, inv)
                        seq[i0:i0+B, ki, yv] = (-nll.sum(1)).float().cpu().numpy()
                        field[i0:i0+B, :, ki, yv] = (-nll_can).float().cpu().numpy()
                log(f"  order {ki+1}/{K}")
        f_llr = (field[:, :, :, 1] - field[:, :, :, 0]).mean(2)
        f_std = (field[:, :, :, 1] - field[:, :, :, 0]).std(2)
        return seq, f_llr, f_std

    seq_va, f_va, s_va = score_split(id_tr[va0], c_tr[va0])
    seq_te, f_te, s_te = score_split(id_te, c_te)
    # assert sum field ≈ seq
    seq_llr_va = seq_va[:, :, 1] - seq_va[:, :, 0]
    sum_f = f_va.sum(1)
    # mean over orders of seq vs sum field — approximate check on mean LLR
    diff = np.abs(seq_llr_va.mean(1) - sum_f)
    log(f"assert |mean_seq_llr - sum_field| max={diff.max():.4e} mean={diff.mean():.4e}")

    seq_oof = np.zeros((len(y), len(orders), 2), np.float32)
    f_oof = np.zeros((len(y), 24), np.float32)
    s_oof = np.zeros((len(y), 24), np.float32)
    seq_oof[va0] = seq_va
    f_oof[va0] = f_va
    s_oof[va0] = s_va
    np.save(args.out / "hgdt_v2_f0_seq_llr_oof.npy", seq_oof)
    np.save(args.out / "hgdt_v2_f0_seq_llr_test.npy", seq_te)
    np.save(args.out / "hgdt_v2_f0_field24_llr_oof.npy", f_oof)
    np.save(args.out / "hgdt_v2_f0_field24_llr_test.npy", f_te)
    np.save(args.out / "hgdt_v2_f0_field24_std_oof.npy", s_oof)
    np.save(args.out / "hgdt_v2_f0_field24_std_test.npy", s_te)
    (args.out / "hgdt_v2_f0_meta.json").write_text(json.dumps({
        "K": len(orders), "vocab": tok.vocab, "NO_SUBMIT": True, "throughput": bench,
        "sum_vs_seq_max_abs": float(diff.max()),
    }, indent=2))
    log("HGDT-v2 Fold0 DONE")


if __name__ == "__main__":
    main()
