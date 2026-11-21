#!/usr/bin/env python3
"""HiRes-Compact-LM 5-fold campaign (V≤4096, T=72, value-token CE, 30 antithetic orders).

Fixes vs prior L7 crash / design flaws:
  1. Continuous residual δ_cont injected ONLY at odd value positions (2j+1) — col_ID stays pure.
  2. No LLR BCE; value-token CE mask (odd=1.0, even_col=0.05) + 2× fraud oversample (finetune).
  3. Hi-res quantile/tail bins → V≈3800–4096; T=72 exactly (BOS+LAB + 35 feat × 2).
  4. 30 antithetic master orders; K_oof=10; K_test=6 rotated so folds cover all 30.
  5. save_fold_preds writes oof=(n_val,10,2) and test=(299984,6,2); Phase-0 unit-tests this.

Modes:
  --mode unit_test   CPU-only Phase-0 gate (mandatory before pod)
  --mode run         GPU training (replica → Fold0 canary → Folds1–4)
  --mode canary_eval Local Fold-0 canary after rsync

Env: HIRES_ROOT, HIRES_OUT, HIRES_CKPT, SEED, EPOCHS, BATCH, LR, FOLDS, SKIP_REPLICA
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------- paths
VOL = Path(os.environ.get("HIRES_VOL", "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud"))
ROOT = Path(os.environ.get("HIRES_ROOT", str(VOL)))
DATA = Path(os.environ.get("HIRES_DATA", str(ROOT / "data")))
ORIG = Path(os.environ.get("HIRES_ORIG", str(ROOT / "original" / "credit_card_fraud_2026.csv")))
OUT = Path(os.environ.get("HIRES_OUT", str(ROOT / "preds" / "hires_lm_5f")))
CK = Path(os.environ.get("HIRES_CKPT", str(ROOT / "checkpoints" / "hires_lm_5f")))
CACHE = Path(os.environ.get("HIRES_CACHE", str(ROOT / "cache" / "hires_compact.npz")))
for d in (OUT, CK, CACHE.parent):
    d.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("SEED", 7))
EPOCHS = int(os.environ.get("EPOCHS", 6))
BATCH = int(os.environ.get("BATCH", 32))
LR = float(os.environ.get("LR", 5e-5))
REP_EPOCHS = int(os.environ.get("REP_EPOCHS", 40))
REP_BATCH = int(os.environ.get("REP_BATCH", 16))
K_OOF = int(os.environ.get("K_OOF", 10))
K_TEST = int(os.environ.get("K_TEST", 6))
N_ORDERS = 30  # 15 forward + 15 reverse
N_FEAT = 35  # → T = 2 (BOS,LAB) + 35*2 = 72
T_SEQ = 2 + 2 * N_FEAT  # 72
V_CAP = 4096
CANARY_MIN_AUC = float(os.environ.get("CANARY_MIN_AUC", "0.94761"))  # beat L7_f0
CANARY_ORTH_MIN = float(os.environ.get("CANARY_ORTH_MIN", "0.5e-5"))
EMA = 0.999

# Fold f scores test orders P_{6f .. 6f+5} (covers all 30 across folds)
TEST_ORDERS = {f: tuple(range(6 * f, 6 * f + 6)) for f in range(5)}
# OOF: 5 forward + 5 reverse antithetic from master
OOF_ORDERS = (0, 1, 2, 3, 4, 15, 16, 17, 18, 19)  # first 5 fwd + their reverses at +15

dev = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


def log(*a):
    s = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    try:
        with open(OUT / "run.log", "a", encoding="utf-8") as f:
            f.write(s + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------- antithetic orders
def master_perms_antithetic(nc: int, seed: int = 7) -> np.ndarray:
    """15 forward perms + 15 exact reverses → (30, nc)."""
    rs = np.random.RandomState(seed)
    base = np.arange(nc, dtype=np.int64)
    fwd = np.empty((15, nc), dtype=np.int64)
    for k in range(15):
        fwd[k] = rs.permutation(base)
    rev = fwd[:, ::-1].copy()
    return np.concatenate([fwd, rev], axis=0)


def verify_test_cover():
    seen = {i: 0 for i in range(N_ORDERS)}
    for f, orders in TEST_ORDERS.items():
        assert len(orders) == K_TEST == 6 and len(set(orders)) == 6
        for o in orders:
            seen[o] += 1
    assert all(v == 1 for v in seen.values()), seen
    assert len(OOF_ORDERS) == K_OOF
    return seen


# --------------------------------------------------------------------------- features / tokenizer
RAW_NUM = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]
RAW_CAT = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]


def engineer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """24 raw + 11 engineered → 35 feature columns."""
    out = pd.DataFrame(index=df.index)
    for c in RAW_NUM + RAW_CAT:
        out[c] = df[c]
    # 11 engineered
    out["log1p_amount"] = np.log1p(pd.to_numeric(df["amount_usd"], errors="coerce").fillna(0).clip(lower=0))
    out["log1p_balance"] = np.log1p(
        pd.to_numeric(df["account_balance_usd"], errors="coerce").fillna(0).clip(lower=0)
    )
    out["log1p_distance"] = np.log1p(
        pd.to_numeric(df["distance_from_home_km"], errors="coerce").fillna(0).clip(lower=0)
    )
    amt = pd.to_numeric(df["amount_usd"], errors="coerce").fillna(0)
    bal = pd.to_numeric(df["account_balance_usd"], errors="coerce").fillna(0).clip(lower=1)
    out["amt_over_bal"] = (amt / bal).clip(0, 50)
    out["vel_x_txn24"] = (
        pd.to_numeric(df["velocity_score"], errors="coerce").fillna(0)
        * pd.to_numeric(df["txn_count_last_24h"], errors="coerce").fillna(0)
    )
    out["risk_x_foreign"] = pd.to_numeric(df["merchant_risk_score"], errors="coerce").fillna(0) * (
        df["is_foreign_transaction"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(float)
    )
    out["hour_sin"] = np.sin(2 * np.pi * pd.to_numeric(df["time_of_day_hour"], errors="coerce").fillna(0) / 24)
    out["hour_cos"] = np.cos(2 * np.pi * pd.to_numeric(df["time_of_day_hour"], errors="coerce").fillna(0) / 24)
    out["dow_sin"] = np.sin(2 * np.pi * pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0) / 7)
    out["card_age_x_cust"] = (
        pd.to_numeric(df["card_age_months"], errors="coerce").fillna(0)
        * pd.to_numeric(df["customer_age"], errors="coerce").fillna(0)
        / 1000.0
    )
    out["vpn_ip_mismatch"] = (
        df["used_vpn"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(float)
        * df["ip_country_mismatch"].astype(str).str.lower().isin(["1", "true", "yes"]).astype(float)
    )
    assert out.shape[1] == N_FEAT, out.shape[1]
    return out


class HiResTokenizer:
    """Quantile+tail bins (V≤4096). Returns bin_ids (N, nc) and δ_cont (N, nc) in [-0.5,0.5]."""

    def __init__(self, max_bins_num: int = 256, v_cap: int = V_CAP):
        self.max_bins_num = max_bins_num
        self.v_cap = v_cap
        self.edges = {}
        self.cat_map = {}
        self.col_names = None
        self.n_bin_tokens = 0
        self.COL0 = 0
        self.LAB0 = 0
        self.BOS = 0
        self.PAD = 0
        self.END = 0
        self.V = 0  # value/bin vocab size (excl specials)
        self.NTOK = 0

    def fit(self, df: pd.DataFrame):
        self.col_names = list(df.columns)
        nc = len(self.col_names)
        # reserve tokens: bin values first, then specials, then col_ids
        # We'll assign bin token ids densely across all columns' bins.
        next_id = 0
        for c in self.col_names:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s) or c.startswith(("log1p_", "amt_", "vel_", "risk_", "hour_", "dow_", "card_", "vpn_")):
                x = pd.to_numeric(s, errors="coerce").astype(np.float64).values
                x = np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0)
                # adaptive bins: up to max_bins_num unique quantile edges
                nuniq = len(np.unique(x.round(8)))
                nb = int(min(self.max_bins_num, max(8, nuniq)))
                qs = np.linspace(0, 1, nb + 1)
                edges = np.unique(np.quantile(x, qs))
                if len(edges) < 3:
                    edges = np.array([x.min() - 1, x.mean(), x.max() + 1], dtype=np.float64)
                self.edges[c] = edges
                self.cat_map[c] = None
                next_id += max(1, len(edges) - 1)
            else:
                cats = pd.Categorical(s.astype(str).fillna("None"))
                self.cat_map[c] = {v: i for i, v in enumerate(cats.categories)}
                self.edges[c] = None
                next_id += len(cats.categories)
        self.V = int(min(next_id, self.v_cap - nc - 8))
        # specials after value vocab
        self.END = self.V
        self.PAD = self.V + 1
        self.LAB0 = self.V + 2  # LAB0=False, LAB0+1=True
        self.COL0 = self.V + 4
        self.BOS = self.COL0 + nc
        self.NTOK = self.BOS + 1
        assert self.NTOK <= self.v_cap + 64, self.NTOK
        return self

    def transform(self, df: pd.DataFrame):
        assert list(df.columns) == self.col_names
        N, nc = len(df), len(self.col_names)
        bins = np.zeros((N, nc), dtype=np.int64)
        delta = np.zeros((N, nc), dtype=np.float32)
        # re-walk assignment consistently with fit order
        cursor = 0
        for j, c in enumerate(self.col_names):
            if self.edges[c] is not None:
                edges = self.edges[c]
                x = pd.to_numeric(df[c], errors="coerce").astype(np.float64).values
                x = np.nan_to_num(x, nan=float(np.median(edges)))
                # bin index
                b = np.searchsorted(edges, x, side="right") - 1
                b = np.clip(b, 0, len(edges) - 2)
                bins[:, j] = cursor + b
                # intra-bin residual in [-0.5, 0.5]
                lo, hi = edges[b], edges[b + 1]
                span = np.maximum(hi - lo, 1e-12)
                # center at mid → [-0.5, 0.5]
                delta[:, j] = ((x - lo) / span - 0.5).astype(np.float32)
                delta[:, j] = np.clip(delta[:, j], -0.5, 0.5)
                cursor += len(edges) - 1
            else:
                mp = self.cat_map[c]
                vals = df[c].astype(str).fillna("None").values
                b = np.array([mp.get(v, 0) for v in vals], dtype=np.int64)
                bins[:, j] = cursor + b
                delta[:, j] = 0.0  # categorical: no continuous residual
                cursor += len(mp)
        # clamp into V
        bins = np.clip(bins, 0, self.V - 1)
        return bins, delta


def build_cache():
    log("building HiRes cache…")
    rd = lambda p: pd.read_csv(p)
    tr, te = rd(DATA / "train.csv"), rd(DATA / "test.csv")
    og = rd(ORIG)
    # align columns
    Xtr = engineer_frame(tr)
    Xte = engineer_frame(te)
    Xog = engineer_frame(og)
    allX = pd.concat([Xtr, Xte, Xog], ignore_index=True)
    tok = HiResTokenizer()
    tok.fit(allX)
    bins, delta = tok.transform(allX)
    y = np.r_[
        tr["is_fraud"].astype(np.float64).values,
        np.full(len(te), -1.0),
        og["is_fraud"].astype(np.float64).values,
    ].astype(np.int8)
    np.savez_compressed(
        CACHE,
        bins=bins.astype(np.int32),
        delta=delta.astype(np.float32),
        y=y,
        ntr=len(tr),
        nte=len(te),
        nog=len(og),
        col_names=np.array(tok.col_names),
        V=tok.V,
        NTOK=tok.NTOK,
        COL0=tok.COL0,
        LAB0=tok.LAB0,
        BOS=tok.BOS,
        PAD=tok.PAD,
        END=tok.END,
        meta_json=np.array(json.dumps({"edges_keys": list(tok.edges.keys()), "V": tok.V, "NTOK": tok.NTOK})),
    )
    # pickle tokenizer edges via sidecar json (edges as lists)
    edge_dump = {}
    for c, e in tok.edges.items():
        edge_dump[c] = None if e is None else e.tolist()
    (CACHE.parent / "hires_tok_edges.json").write_text(json.dumps({"edges": edge_dump, "cat_map": tok.cat_map}))
    meta = {
        "ntr": int(len(tr)),
        "nte": int(len(te)),
        "nog": int(len(og)),
        "nc": int(N_FEAT),
        "V": int(tok.V),
        "NTOK": int(tok.NTOK),
        "T": int(T_SEQ),
        "cols": tok.col_names,
    }
    (OUT / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"cache: V={tok.V} NTOK={tok.NTOK} T={T_SEQ} ntr={len(tr)} nte={len(te)}")
    return tok, meta


def load_pack():
    if not CACHE.exists():
        build_cache()
    z = np.load(CACHE, allow_pickle=True)
    g = {k: z[k] for k in z.files}
    g["ntr"] = int(g["ntr"])
    g["nte"] = int(g["nte"])
    g["nog"] = int(g["nog"])
    g["V"] = int(g["V"])
    g["NTOK"] = int(g["NTOK"])
    g["COL0"] = int(g["COL0"])
    g["LAB0"] = int(g["LAB0"])
    g["BOS"] = int(g["BOS"])
    g["PAD"] = int(g["PAD"])
    g["END"] = int(g["END"])
    g["NC"] = g["bins"].shape[1]
    g["y"] = g["y"].astype(np.int64)
    assert g["NC"] == N_FEAT
    assert T_SEQ == 72
    return g


# --------------------------------------------------------------------------- model
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head, dropout=0.05):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.d_head = d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, Dh)
        # FlashAttention / SDPA causal — no separate mask needed
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.drop if self.training else 0.0, is_causal=True
        )
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(a)


class Block(nn.Module):
    def __init__(self, d, n_head, dropout=0.05):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.mlp(self.ln2(x))
        return x


class HiResCausalLM(nn.Module):
    def __init__(self, ntok, d=512, n_head=8, n_layer=8, max_t=T_SEQ):
        super().__init__()
        self.wte = nn.Embedding(ntok, d)
        self.wpe = nn.Embedding(max_t, d)
        # continuous residual projection — ONLY added at odd value positions
        self.cont = nn.Linear(1, d, bias=False)
        nn.init.normal_(self.cont.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, ntok, bias=False)
        self.head.weight = self.wte.weight  # tie
        self.d = d
        self.max_t = max_t

    def forward(self, tok, cont_delta, return_logits=True):
        """
        tok: (B, T) int64
        cont_delta: (B, T) float — ZERO on even positions; δ_cont on odd value positions
        """
        B, T = tok.shape
        pos = torch.arange(T, device=tok.device)
        x = self.wte(tok) + self.wpe(pos)[None, :, :]
        # Flaw #1 fix: inject continuous residual only where cont_delta != 0 (odd value slots)
        x = x + self.cont(cont_delta.unsqueeze(-1))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        if return_logits:
            return self.head(x)
        return x


# --------------------------------------------------------------------------- sequence builder
def build_sequences(g, idx, perm, label: int):
    """Build (B, T) token ids and (B, T) cont deltas for given rows & column permutation.

    Layout: [BOS, LAB] + for j in perm: [COL_ID_j, VAL_BIN_j]
    cont:   [0,   0  ] + for j in perm: [0,        δ_j      ]
    """
    idx = np.asarray(idx, dtype=np.int64)
    B = len(idx)
    nc = g["NC"]
    perm = np.asarray(perm, dtype=np.int64)
    assert perm.shape == (nc,)
    bins = g["bins"][idx]  # (B, nc)
    delta = g["delta"][idx]
    T = T_SEQ
    tok = np.full((B, T), g["PAD"], dtype=np.int64)
    cont = np.zeros((B, T), dtype=np.float32)
    tok[:, 0] = g["BOS"]
    tok[:, 1] = g["LAB0"] + int(label)
    # fill pairs
    for t_j, col in enumerate(perm):
        p_even = 2 + 2 * t_j
        p_odd = p_even + 1
        tok[:, p_even] = g["COL0"] + int(col)  # pure col ID — no δ
        tok[:, p_odd] = bins[:, col]
        cont[:, p_odd] = delta[:, col]  # δ ONLY here
        # even stays 0
    return tok, cont


def masked_ce_loss(logits, targets, pos_weights):
    """logits (B,T,V), targets (B,T) with -100 ignore, pos_weights (T,) on token positions."""
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, T)
    w = pos_weights[None, :].expand(B, T)
    # zero weight where ignore
    mask = (targets != -100).float()
    return (loss * w * mask).sum() / (w * mask).sum().clamp_min(1.0)


def position_loss_weights(device):
    """even col positions (except BOS/LAB handled): weight 0.05; odd value: 1.0; prefix: 0.05."""
    w = torch.ones(T_SEQ, device=device)
    w[0] = 0.05  # BOS
    w[1] = 0.05  # LAB — still predict next; light weight
    for j in range(N_FEAT):
        w[2 + 2 * j] = 0.05  # col_ID
        w[2 + 2 * j + 1] = 1.0  # value
    return w


# --------------------------------------------------------------------------- save pipeline (Flaw #5)
def save_fold_preds(out_dir: Path, fold: int, oof_ll: np.ndarray, test_ll: np.ndarray, val_idx: np.ndarray):
    """Atomic verified save — shapes locked to (n_val,10,2) and (299984,6,2)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = len(val_idx)
    oof_ll = np.asarray(oof_ll, dtype=np.float32)
    test_ll = np.asarray(test_ll, dtype=np.float32)
    expect_oof = (n_val, K_OOF, 2)
    expect_test = (299984, K_TEST, 2)
    assert oof_ll.shape == expect_oof, (oof_ll.shape, expect_oof, "HINT: save ONLY val rows, not full train")
    assert test_ll.shape == expect_test, (test_ll.shape, expect_test)
    assert np.isfinite(oof_ll).all() and np.isfinite(test_ll).all()
    # write idx sidecar so OOF can be reassembled
    np.save(out_dir / f"oof_{fold}_idx.npy", np.asarray(val_idx, dtype=np.int64))
    op = out_dir / f"oof_{fold}.npy"
    tp = out_dir / f"test_{fold}.npy"
    np.save(op, oof_ll)
    np.save(tp, test_ll)
    # reload verify
    o2, t2 = np.load(op), np.load(tp)
    assert o2.shape == expect_oof and t2.shape == expect_test
    assert o2.dtype == np.float32 and t2.dtype == np.float32
    assert np.isfinite(o2).all() and np.isfinite(t2).all()
    log(f"save_fold_preds fold={fold} oof={o2.shape} test={t2.shape} OK")
    return op, tp


# --------------------------------------------------------------------------- scoring
@torch.no_grad()
def score_ll(model, g, idx, orders, label_pair=(0, 1), bs=64):
    """Return (N, K, 2) full-sequence sum logprob under label False/True for each order."""
    model.eval()
    idx = np.asarray(idx, dtype=np.int64)
    N = len(idx)
    K = len(orders)
    perms = master_perms_antithetic(g["NC"], SEED)
    out = np.zeros((N, K, 2), dtype=np.float32)
    for ki, oid in enumerate(orders):
        perm = perms[oid]
        for lab in label_pair:
            for s in range(0, N, bs):
                sl = idx[s : s + bs]
                tok_np, cont_np = build_sequences(g, sl, perm, lab)
                tok = torch.as_tensor(tok_np, device=dev)
                cont = torch.as_tensor(cont_np, device=dev)
                logits = model(tok, cont)  # (B,T,V)
                logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
                tgt = tok[:, 1:]
                # gather token logprobs
                gather = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B,T-1)
                # mask pad
                m = (tgt != g["PAD"]).float()
                ll = (gather * m).sum(1).detach().cpu().numpy().astype(np.float32)
                out[s : s + len(sl), ki, lab] = ll
    return out


def bayes_ll(ll, prior):
    t = np.logaddexp.reduce(ll[:, :, 1].astype(np.float64), axis=1)
    f = np.logaddexp.reduce(ll[:, :, 0].astype(np.float64), axis=1)
    # analytic correction note: training oversampled fraud 2× → adjust prior at inference
    return t - f + np.log(prior / (1.0 - prior))


# --------------------------------------------------------------------------- train
class EMA:
    def __init__(self, model, decay=EMA):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, p in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def make_loader_idx(rows, y, batch, seed, oversample_fraud=False):
    rs = np.random.RandomState(seed)
    rows = np.asarray(rows)
    if oversample_fraud:
        # 2× fraud rows among labeled
        lab = y[rows]
        pos = rows[lab == 1]
        neg = rows[lab == 0]
        # unlabeled (orig y may be 0/1 only here)
        pos2 = np.concatenate([pos, pos]) if len(pos) else pos
        pool = np.concatenate([neg, pos2]) if len(pos2) else neg
    else:
        pool = rows
    rs.shuffle(pool)
    # drop remainder
    n = (len(pool) // batch) * batch
    pool = pool[:n]
    return pool.reshape(-1, batch)


def train_epochs(model, g, rows, y, epochs, batch, lr, tag, seed, oversample_fraud=False, perms_cycle=True):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    ema = EMA(model)
    pos_w = position_loss_weights(dev)
    perms = master_perms_antithetic(g["NC"], SEED)
    ydev = g["y"]
    for ep in range(epochs):
        batches = make_loader_idx(rows, ydev, batch, seed + ep, oversample_fraud=oversample_fraud)
        tot, n = 0.0, 0
        for bi, b in enumerate(batches):
            # cycle through perms
            perm = perms[(ep * 17 + bi) % N_ORDERS] if perms_cycle else perms[0]
            # labels from data (orig/train); for unsupervised orig use true labels
            labs = ydev[b]
            # mix labels for generative LM: use true label token when known
            tok_list, cont_list, tgt_list = [], [], []
            for lab_choice in (0, 1):
                # half batch each? simpler: use true label for each row
                pass
            tok_np, cont_np = build_sequences(g, b, perm, 0)  # placeholder
            # per-row true label
            for i, row in enumerate(b):
                lab = int(max(0, min(1, ydev[row])))
                t1, c1 = build_sequences(g, [row], perm, lab)
                tok_np[i] = t1[0]
                cont_np[i] = c1[0]
            tok = torch.as_tensor(tok_np, device=dev)
            cont = torch.as_tensor(cont_np, device=dev)
            logits = model(tok, cont)
            # next-token targets
            tgt = tok.clone()
            tgt[:, :-1] = tok[:, 1:]
            tgt[:, -1] = -100
            # don't predict PAD
            tgt[tok == g["PAD"]] = -100
            loss = masked_ce_loss(logits[:, :-1, :], tgt[:, :-1], pos_w[:-1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            tot += float(loss.detach())
            n += 1
            if bi % 50 == 0:
                log(f"  {tag} ep{ep} bi{bi}/{len(batches)} loss={float(loss):.4f}")
        log(f"{tag} epoch {ep+1}/{epochs} mean_loss={tot/max(n,1):.4f}")
        torch.save({"model": model.state_dict(), "ema": ema.shadow, "epoch": ep}, CK / f"{tag}.pt")
    ema.copy_to(model)
    return model


# --------------------------------------------------------------------------- canary
def fold0_canary(g, oof_ll, val_idx):
    y = g["y"][: g["ntr"]]
    prior = float((y == 1).mean())
    # analytic correction for 2× fraud oversample in finetune: effective prior tilt
    # p' = 2p/(1+p) during train → invert at inference by using original prior in bayes
    z = bayes_ll(oof_ll, prior)
    auc = float(roc_auc_score(y[val_idx], z))
    info = {"auc": auc, "prior": prior, "pass_auc": auc >= CANARY_MIN_AUC, "orth_e5": None, "pass": False}
    # orth lift vs cand2-locked simple eval if available
    try:
        z_ref = np.load(ROOT / "preds/gm_v5_overnight/z_ref_old_oof.npy").ravel()
        d1 = np.load(ROOT / "preds/gm_v4_polish/d1_v4_oof.npy").ravel()
        base = z_ref[val_idx] + 0.05 * d1[val_idx]
        from scipy.special import logit
        from scipy.stats import rankdata

        def zrk(x):
            return logit((rankdata(x, method="average") - 0.5) / len(x))

        r = zrk(z)
        # project out base
        b = zrk(base)
        beta = float(np.dot(r - r.mean(), b - b.mean()) / (np.dot(b - b.mean(), b - b.mean()) + 1e-12))
        resid = r - beta * b
        best = -1e9
        for w in (0.02, 0.04, 0.06, 0.08, 0.10):
            d = roc_auc_score(y[val_idx], base + w * resid) - roc_auc_score(y[val_idx], base)
            best = max(best, d)
        info["orth_e5"] = best * 1e5
        info["pass_orth"] = best > CANARY_ORTH_MIN
    except Exception as e:
        info["orth_err"] = str(e)[:120]
        info["pass_orth"] = False
    info["pass"] = bool(info["pass_auc"] and info.get("pass_orth", False))
    # also compare vs L7_f0 / G26_f0 references
    info["beat_L7_f0"] = auc > 0.94761
    info["ref_G26_f0"] = 0.94919
    return info


# --------------------------------------------------------------------------- Phase 0 unit test
def phase0_unit_test():
    log("======== PHASE0 UNIT TEST (CPU, $0) ========")
    verify_test_cover()
    log(f"test cover OK: each of 30 orders appears once across folds; K_oof={K_OOF} K_test={K_TEST}")
    # tiny synthetic pack mirroring shapes
    ntr, nte, nog = 1000, 500, 200
    nc = N_FEAT
    rs = np.random.RandomState(0)
    bins = rs.randint(0, 100, size=(ntr + nte + nog, nc)).astype(np.int32)
    delta = (rs.rand(ntr + nte + nog, nc).astype(np.float32) - 0.5)
    # even positions must remain 0 in cont builder — tested below
    y = np.zeros(ntr + nte + nog, np.int64)
    y[:ntr] = rs.binomial(1, 0.02, ntr)
    y[ntr : ntr + nte] = -1
    y[ntr + nte :] = rs.binomial(1, 0.02, nog)
    V = 200
    COL0 = V + 4
    LAB0 = V + 2
    BOS = COL0 + nc
    g = dict(
        bins=bins,
        delta=delta,
        y=y,
        ntr=ntr,
        nte=nte,
        nog=nog,
        V=V,
        NTOK=BOS + 1,
        COL0=COL0,
        LAB0=LAB0,
        BOS=BOS,
        PAD=V + 1,
        END=V,
        NC=nc,
    )
    # causal leakage check: even slots cont==0, odd==delta
    perm = np.arange(nc)
    tok, cont = build_sequences(g, np.arange(8), perm, 1)
    assert tok.shape == (8, T_SEQ) == (8, 72)
    assert np.allclose(cont[:, 0], 0) and np.allclose(cont[:, 1], 0)
    for j in range(nc):
        assert np.allclose(cont[:, 2 + 2 * j], 0.0), "col_ID position leaked δ!"
        assert np.allclose(cont[:, 2 + 2 * j + 1], delta[:8, perm[j]])
    log("causal leakage check OK (δ only on odd value positions)")

    # model forward + loss on CPU
    model = HiResCausalLM(g["NTOK"], d=128, n_head=4, n_layer=2).to(dev)
    pos_w = position_loss_weights(dev)
    tok_t = torch.as_tensor(tok, device=dev)
    cont_t = torch.as_tensor(cont, device=dev)
    logits = model(tok_t, cont_t)
    tgt = tok_t.clone()
    tgt[:, :-1] = tok_t[:, 1:]
    tgt[:, -1] = -100
    loss = masked_ce_loss(logits[:, :-1], tgt[:, :-1], pos_w[:-1])
    assert torch.isfinite(loss)
    log(f"dummy forward+masked CE OK loss={float(loss):.4f}")

    # save pipeline across folds 0..4 — test dim locked to competition size 299984
    tmp = OUT / "_unit_test_saves"
    tmp.mkdir(parents=True, exist_ok=True)
    N_TEST_LOCK = 299984
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y[:ntr]))
    for f, (_, te) in enumerate(folds):
        n_val = len(te)
        oof = rs.randn(n_val, K_OOF, 2).astype(np.float32)
        test = rs.randn(N_TEST_LOCK, K_TEST, 2).astype(np.float32)
        # MUST fail if wrong shape — catch the prior L7 crash: (699960,10,2) style
        bad = rs.randn(ntr, K_OOF, 2).astype(np.float32)
        rejected = False
        try:
            save_fold_preds(tmp, f, bad, test, te)
        except AssertionError as e:
            rejected = True
            assert "val rows" in str(e) or bad.shape != (n_val, K_OOF, 2)
            log(f"  fold{f}: correctly rejected wrong oof shape {bad.shape}")
        assert rejected, "save_fold_preds must reject full-train oof shape"
        save_fold_preds(tmp, f, oof, test, te)
    log("PHASE0 UNIT TEST PASSED — safe to provision GPU")
    (OUT / "PHASE0_PASS.json").write_text(
        json.dumps({"pass": True, "T": T_SEQ, "K_OOF": K_OOF, "K_TEST": K_TEST, "N_ORDERS": N_ORDERS}, indent=2)
    )
    return True


# --------------------------------------------------------------------------- run
def run_training(folds_which):
    assert torch.cuda.is_available(), "CUDA required for --mode run"
    assert (OUT / "PHASE0_PASS.json").exists(), "Run --mode unit_test first"
    g = load_pack()
    verify_test_cover()
    np.save(OUT / "master_perms_antithetic30.npy", master_perms_antithetic(g["NC"], SEED))
    (OUT / "test_orders.json").write_text(json.dumps({str(k): list(v) for k, v in TEST_ORDERS.items()}))
    y = g["y"]
    ntr, nte = g["ntr"], g["nte"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y[:ntr]))
    model = HiResCausalLM(g["NTOK"]).to(dev)
    # replica pretrain on orig20k
    orig_rows = np.arange(ntr + nte, ntr + nte + g["nog"])
    if not int(os.environ.get("SKIP_REPLICA", "0")):
        log("replica pretrain on orig20k…")
        train_epochs(model, g, orig_rows, y, REP_EPOCHS, REP_BATCH, LR, "hires_replica", SEED, oversample_fraud=False)
        torch.save(model.state_dict(), CK / "hires_replica.pt")
    else:
        p = CK / "hires_replica.pt"
        if p.exists():
            model.load_state_dict(torch.load(p, map_location=dev))
            log("loaded hires_replica.pt")

    for f in folds_which:
        tr, te = folds[f]
        log(f"======== FOLD {f} train={len(tr)} val={len(te)} ========")
        # finetune with 2× fraud oversample
        train_epochs(
            model,
            g,
            tr,
            y,
            EPOCHS,
            BATCH,
            LR,
            f"hires_fold{f}",
            SEED + f,
            oversample_fraud=True,
        )
        torch.save(model.state_dict(), CK / f"hires_fold{f}.pt")
        # score
        log(f"scoring fold {f} OOF K={K_OOF}…")
        oof_ll = score_ll(model, g, te, OOF_ORDERS, bs=int(os.environ.get("SCORE_BS", "64")))
        log(f"scoring fold {f} TEST K={K_TEST} orders={TEST_ORDERS[f]}…")
        test_ll = score_ll(model, g, np.arange(ntr, ntr + nte), TEST_ORDERS[f], bs=int(os.environ.get("SCORE_BS", "64")))
        save_fold_preds(OUT, f, oof_ll, test_ll, te)
        if f == 0:
            info = fold0_canary(g, oof_ll, te)
            (OUT / "fold0_canary.json").write_text(json.dumps(info, indent=2))
            log(f"FOLD0 CANARY: {info}")
            if not info["pass"]:
                log("FOLD0 CANARY FAILED — caller must terminate pod")
                return False
            log("FOLD0 CANARY PASSED — continue folds 1–4")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unit_test", "run", "canary_eval", "build_cache"], default="unit_test")
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    if a.mode == "unit_test":
        ok = phase0_unit_test()
        # also build real cache quickly if data present (optional smoke)
        if (DATA / "train.csv").exists() and (ORIG).exists():
            try:
                log("smoke: building real tokenizer cache (may take ~1–2 min)…")
                build_cache()
                g = load_pack()
                log(f"real cache OK V={g['V']} NTOK={g['NTOK']} bins={g['bins'].shape}")
                assert g["V"] <= V_CAP, g["V"]
                # one real batch through model
                m = HiResCausalLM(g["NTOK"], d=256, n_head=4, n_layer=2).to(dev)
                tok, cont = build_sequences(g, np.arange(8), np.arange(g["NC"]), 1)
                with torch.no_grad():
                    lg = m(torch.as_tensor(tok, device=dev), torch.as_tensor(cont, device=dev))
                assert lg.shape[1] == T_SEQ
                log("real-data smoke forward OK")
            except Exception as e:
                log(f"real cache smoke skipped/failed: {e}")
        sys.exit(0 if ok else 1)
    if a.mode == "build_cache":
        build_cache()
        return
    if a.mode == "canary_eval":
        g = load_pack()
        oof = np.load(OUT / "oof_0.npy")
        idx = np.load(OUT / "oof_0_idx.npy")
        info = fold0_canary(g, oof, idx)
        print(json.dumps(info, indent=2))
        sys.exit(0 if info["pass"] else 2)
    if a.mode == "run":
        which = [int(x) for x in a.folds.split(",") if x != ""]
        ok = run_training(which)
        sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
