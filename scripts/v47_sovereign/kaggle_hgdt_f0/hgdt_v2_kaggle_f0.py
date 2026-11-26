#!/usr/bin/env python3
"""V47.3 HGDT-LM v2 — Private Kaggle CUDA Fold-0 canary.

Pure Target-First Generative Bayes (T=25). NO_SUBMIT=True.
Sovereign: competition train/test only. 0 competitor preds. 0 RunPod.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

NO_SUBMIT = True
assert NO_SUBMIT
T0 = time.time()

OUT = Path("/kaggle/working")
BS = 256
PRETRAIN_EPOCHS = 1
FINETUNE_EPOCHS = 3
K_PAIRS = 4
NW = 2


def find_data_dir() -> Path:
    """Locate competition CSVs under /kaggle/input (layout can vary)."""
    roots = [
        Path("/kaggle/input/playground-series-s6e11"),
        Path("/kaggle/input"),
    ]
    for root in roots:
        if not root.exists():
            continue
        direct = root / "train.csv"
        if direct.exists():
            return root
        for p in root.rglob("train.csv"):
            if (p.parent / "test.csv").exists():
                return p.parent
    # debug listing
    inp = Path("/kaggle/input")
    listing = []
    if inp.exists():
        for p in sorted(inp.rglob("*"))[:80]:
            listing.append(str(p))
    raise FileNotFoundError(
        "train.csv not found under /kaggle/input. listing=" + json.dumps(listing[:40])
    )

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


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        self.vocab = 4

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
        cont = np.zeros((n, 24, 3), np.float32)
        for j, c in enumerate(FEATS):
            base = self.offsets[c]
            if c in self.edges:
                x = df[c].to_numpy(float)
                e = self.edges[c]
                b = np.clip(np.searchsorted(e, x, side="right") - 1, 0, len(e) - 1)
                ids[:, j] = base + b
                cont[:, j, 0] = np.tanh(x / (np.abs(x) + 1.0)).astype(np.float32)
                cont[:, j, 1] = (x % 1.0).astype(np.float32)
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
        self.y_emb = nn.Embedding(3, d_model)
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
        ids_o = torch.gather(ids, 1, order)
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
        return self.head(h)


def field_nll(logits, ids, order):
    tgt = torch.gather(ids, 1, order)
    pred = logits[:, :-1, :]
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="none").view(ids.size(0), 24)


def make_loader(ids, cont, y=None, shuffle=True):
    ids_t = torch.from_numpy(ids.astype(np.int64))
    cont_t = torch.from_numpy(cont.astype(np.float32))
    if y is None:
        y_t = torch.full((len(ids),), 2, dtype=torch.long)
    else:
        y_t = torch.from_numpy(y.astype(np.int64))
    ds = TensorDataset(ids_t, cont_t, y_t)
    return DataLoader(
        ds, batch_size=BS, shuffle=shuffle, num_workers=NW,
        pin_memory=True, drop_last=False,
    )


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    log(f"device={device} gpu={torch.cuda.get_device_name(0)}")

    data = find_data_dir()
    log(f"data_dir={data}")
    tr = pd.read_csv(data / "train.csv")
    te = pd.read_csv(data / "test.csv")
    y = tr["is_fraud"].to_numpy(int)
    folds = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        folds[va] = k
    tr0 = folds != 0
    va0 = folds == 0
    assert int(va0.sum()) == 139992, int(va0.sum())
    assert int(tr0.sum()) == 559968, int(tr0.sum())
    log(f"tr0={tr0.sum()} va0={va0.sum()} test={len(te)}")

    tok = FieldTok(n_q=100).fit([tr, te])
    log("encoding…")
    id_tr, c_tr = tok.encode(tr)
    id_te, c_te = tok.encode(te)
    id_u = np.concatenate([id_tr, id_te])
    c_u = np.concatenate([c_tr, c_te])
    log(f"unlab={len(id_u)} (train∪test, 0 labels)")

    model = HGDTv2(tok.vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    try:
        scaler = torch.amp.GradScaler("cuda")
    except Exception:
        from torch.cuda.amp import GradScaler as _GS
        scaler = _GS()
    orders_pool = antithetic_orders(24, 4, seed=0)
    rng = np.random.RandomState(0)

    def train_epoch(loader, labeled: bool):
        model.train()
        total, n = 0.0, 0
        for ids, cont, ytok in loader:
            B = ids.size(0)
            ids = ids.to(device, non_blocking=True)
            cont = cont.to(device, non_blocking=True)
            ytok = ytok.to(device, non_blocking=True)
            o = torch.as_tensor(
                orders_pool[rng.randint(0, len(orders_pool))],
                device=device, dtype=torch.long,
            ).unsqueeze(0).expand(B, -1)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(ids, cont, ytok, o)
                # CE in float32 for stability
                loss = field_nll(logits.float(), ids, o).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * B
            n += B
        return total / max(n, 1)

    # Phase A1
    log("Phase A1 unsupervised (1 ep, train∪test)…")
    for ep in range(PRETRAIN_EPOCHS):
        loader = make_loader(id_u, c_u, y=None, shuffle=True)
        loss = train_epoch(loader, labeled=False)
        log(f"  A1 ep{ep} loss={loss:.4f}")
    torch.save({"model": model.state_dict(), "vocab": tok.vocab}, OUT / "hgdt_v2_a1.pt")

    # Phase A2
    log("Phase A2 Target-First FT (3 ep, tr0)…")
    for ep in range(FINETUNE_EPOCHS):
        loader = make_loader(id_tr[tr0], c_tr[tr0], y=y[tr0], shuffle=True)
        loss = train_epoch(loader, labeled=True)
        log(f"  A2 ep{ep} loss={loss:.4f}")
    torch.save({"model": model.state_dict(), "vocab": tok.vocab}, OUT / "hgdt_v2_a2.pt")

    # Phase A3 score
    orders = antithetic_orders(24, K_PAIRS, seed=47)
    assert len(orders) == 8
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
                    for i0 in range(0, N, BS):
                        sl = slice(i0, min(i0 + BS, N))
                        ids = torch.as_tensor(ids_np[sl], device=device)
                        cont = torch.as_tensor(cont_np[sl], device=device)
                        B = ids.size(0)
                        o = ord_t.unsqueeze(0).expand(B, -1)
                        ytok = torch.full((B,), yv, device=device, dtype=torch.long)
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            logits = model(ids, cont, ytok, o)
                        nll = field_nll(logits.float(), ids, o)
                        inv = torch.empty_like(o)
                        inv.scatter_(1, o, torch.arange(24, device=device).unsqueeze(0).expand(B, -1))
                        nll_can = torch.gather(nll, 1, inv)
                        seq[i0:i0 + B, ki, yv] = (-nll.sum(1)).float().cpu().numpy()
                        field[i0:i0 + B, :, ki, yv] = (-nll_can).float().cpu().numpy()
                log(f"  order {ki+1}/{K} done N={N}")
        # per-order field LLR then mean/std across orders
        field_llr_k = field[:, :, :, 1] - field[:, :, :, 0]  # (N,24,K)
        seq_llr_k = seq[:, :, 1] - seq[:, :, 0]  # (N,K)
        sum_field_k = field_llr_k.sum(1)  # (N,K)
        max_abs = float(np.max(np.abs(sum_field_k - seq_llr_k)))
        assert max_abs < 1e-3, f"sum(field24)-seq_llr max_abs={max_abs}"
        f_llr = field_llr_k.mean(2)
        f_std = field_llr_k.std(2)
        return seq, f_llr, f_std, max_abs, seq_llr_k

    seq_va, f_va, s_va, max_abs_va, seq_llr_va = score_split(id_tr[va0], c_tr[va0])
    seq_te, f_te, s_te, max_abs_te, seq_llr_te = score_split(id_te, c_te)
    log(f"assert max|sum(field)-seq| va={max_abs_va:.3e} te={max_abs_te:.3e}")

    # Solo AUCs
    y_va = y[va0]
    mean_seq = seq_llr_va.mean(1)
    solo_auc = float(roc_auc_score(y_va, mean_seq))
    per_order_auc = [float(roc_auc_score(y_va, seq_llr_va[:, k])) for k in range(seq_llr_va.shape[1])]
    sum_field_auc = float(roc_auc_score(y_va, f_va.sum(1)))
    log(f"solo mean_seq AUC={solo_auc:.6f} sum_field AUC={sum_field_auc:.6f}")
    log(f"per-order AUCs={per_order_auc}")

    # Save va-sized oof as specified
    paths = {
        "hgdt_v2_f0_seq_llr_oof.npy": seq_va,
        "hgdt_v2_f0_seq_llr_test.npy": seq_te,
        "hgdt_v2_f0_field24_llr_oof.npy": f_va,
        "hgdt_v2_f0_field24_llr_test.npy": f_te,
        "hgdt_v2_f0_field24_std_oof.npy": s_va,
        "hgdt_v2_f0_field24_std_test.npy": s_te,
    }
    for name, arr in paths.items():
        p = OUT / name
        np.save(p, arr.astype(np.float32))
        log(f"saved {name} shape={arr.shape}")

    # sanity
    assert seq_va.shape == (139992, 8, 2), seq_va.shape
    assert seq_te.shape == (299984, 8, 2), seq_te.shape
    assert f_va.shape == (139992, 24), f_va.shape
    assert f_te.shape == (299984, 24), f_te.shape
    for name in paths:
        a = np.load(OUT / name)
        assert np.isfinite(a).all(), name

    checksums = {name: sha256_file(OUT / name) for name in paths}
    meta = {
        "NO_SUBMIT": True,
        "private_kaggle": True,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "vocab": tok.vocab,
        "K": 8,
        "bs": BS,
        "solo_fold0_auc_mean_seq": solo_auc,
        "solo_fold0_auc_sum_field": sum_field_auc,
        "per_order_aucs": per_order_auc,
        "max_abs_sum_field_minus_seq_va": max_abs_va,
        "max_abs_sum_field_minus_seq_te": max_abs_te,
        "wall_clock_seconds": time.time() - T0,
        "shapes": {k: list(v.shape) for k, v in paths.items()},
        "sha256": checksums,
        "sovereignty": "train∪test competition only; 0 competitor preds",
    }
    (OUT / "hgdt_v2_f0_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"DONE wall={meta['wall_clock_seconds']:.1f}s AUC={solo_auc:.6f}")


if __name__ == "__main__":
    main()
