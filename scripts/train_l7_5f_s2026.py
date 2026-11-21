#!/usr/bin/env python3
"""Compact-LM V=1372 T≈72 — SEED=2026 second-seed 5-fold (proven L7 recipe + 4 upgrades).

Upgrades vs smugcqcy2oqzeg / train_compact_lm_5f.py:
  1. SEED=2026 for init, replica/FT shuffles, AND 10 antithetic eval column orders
     (5 RandomState(2026) perms + their exact reverses) — disjoint from seed=7 L7F.
  2. Fast inference: SCORE_BS=512, bf16 SDPA, optional torch.compile (scoring only).
  3. Fixed per-fold saver: oof_f.npy = (len(val_idx), 10, 2); test_0 = (299984, 10, 2);
     folds 1–4 test = (299984, 6, 2) rotated cover of the 10 orders.
  4. 3-Gate Fold-0 canary (solo AUC/corr, two-seed gain, cand2 stack ΔF0 + wall ≤50m).

STRICT NO_SUBMIT during canary; never touches foreign pod lheroidmuab1j4.

Modes:
  phase0  — CPU 1-batch unit test across f=0..4 → PHASE0_S2026_PASS.json
  build_cache / run / canary_check
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

ROOT = Path(os.environ.get("L7_ROOT", "/workspace/l7_work"))
DATA = Path(os.environ.get("L7_DATA", str(ROOT / "data")))
ORIG = Path(os.environ.get("L7_ORIG", str(ROOT / "original" / "credit_card_fraud_2026.csv")))
OUT = Path(os.environ.get("L7_OUT", str(ROOT / "preds" / "l7_s2026_5f")))
CK = Path(os.environ.get("L7_CKPT", str(ROOT / "ckpt" / "l7_s2026_5f")))
CACHE = Path(os.environ.get("L7_CACHE", str(ROOT / "cache_compact.npz")))
L7_S7_DIR = Path(
    os.environ.get(
        "L7_S7_DIR",
        str(ROOT / "external" / "goodpjw2008_s6e11-compact-lm-seed7"),
    )
)
for d in (OUT, CK):
    d.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("SEED", 2026))
EPOCHS = int(os.environ.get("EPOCHS", 8))
BATCH = int(os.environ.get("BATCH", 64))
LR = float(os.environ.get("LR", 5e-5))
REP_EPOCHS = int(os.environ.get("REP_EPOCHS", 60))
REP_BATCH = int(os.environ.get("REP_BATCH", 16))
K_OOF = int(os.environ.get("K_OOF", 10))
SCORE_BS = int(os.environ.get("SCORE_BS", 512))
USE_COMPILE = int(os.environ.get("USE_COMPILE", "1"))
WALL_F0_MAX_MIN = float(os.environ.get("WALL_F0_MAX_MIN", "50"))
# 3-gate thresholds
GATE1_AUC_MIN = float(os.environ.get("GATE1_AUC_MIN", "0.9472"))
GATE1_CORR_LO = float(os.environ.get("GATE1_CORR_LO", "0.980"))
GATE1_CORR_HI = float(os.environ.get("GATE1_CORR_HI", "0.996"))
GATE2_DELTA = float(os.environ.get("GATE2_DELTA", "15e-5"))
GATE3_STACK_MIN = float(os.environ.get("GATE3_STACK_MIN", "0.5e-5"))
BOS = 0
LMAX = 6
dev = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False

# Fold0: all 10 orders. Folds1–4: 6 rotated (cover each order ≥2× across 1–4).
TEST_ORDERS = {
    0: tuple(range(10)),
    1: (0, 1, 2, 3, 4, 5),
    2: (2, 3, 4, 5, 6, 7),
    3: (4, 5, 6, 7, 8, 9),
    4: (6, 7, 8, 9, 0, 1),
}


def log(*a):
    s = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "run.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def master_perms(nc: int, seed: int = 2026) -> np.ndarray:
    """10 antithetic column orders: 5 RNG(seed) perms + exact reverses. Shape (10, nc)."""
    rs = np.random.RandomState(seed)
    base = np.arange(nc, dtype=np.int64)
    out = np.empty((10, nc), dtype=np.int64)
    for k in range(5):
        p = rs.permutation(base)
        out[k] = p
        out[k + 5] = p[::-1].copy()
    return out


def verify_orders_disjoint_from_seed7(nc: int) -> dict:
    p7 = master_perms_seed7_style(nc, 7)
    p26 = master_perms(nc, 2026)
    # compare first-5 randoms; also full set as tuples
    s7 = {tuple(r.tolist()) for r in p7}
    s26 = {tuple(r.tolist()) for r in p26}
    inter = s7 & s26
    return {
        "n_intersect_orders": len(inter),
        "disjoint": len(inter) == 0,
        "seed7_nuniq": len(s7),
        "seed2026_nuniq": len(s26),
    }


def master_perms_seed7_style(nc: int, seed: int = 7) -> np.ndarray:
    """Exact seed=7 generator used by train_compact_lm_5f (10 independent perms)."""
    rs = np.random.RandomState(seed)
    base = np.arange(nc, dtype=np.int64)
    out = np.empty((10, nc), dtype=np.int64)
    for k in range(10):
        out[k] = rs.permutation(base)
    return out


def verify_test_cover():
    assert len(TEST_ORDERS[0]) == 10
    seen = {i: 0 for i in range(10)}
    for f in range(1, 5):
        orders = TEST_ORDERS[f]
        assert len(orders) == 6 and len(set(orders)) == 6
        for o in orders:
            seen[o] += 1
    assert all(v >= 2 for v in seen.values()), seen
    return seen


# ---------------------------------------------------------------- serializer / cache
def _gpt2_encode(text: str) -> list[int]:
    """GPT-2 BPE encode. Prefer tiktoken; fall back to transformers."""
    try:
        import tiktoken

        return tiktoken.get_encoding("gpt2").encode(text)
    except Exception:
        from transformers import GPT2TokenizerFast

        return GPT2TokenizerFast.from_pretrained("gpt2").encode(text)


def build_cache():
    rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    train_csv = DATA / "train.csv"
    test_csv = DATA / "test.csv"
    if not train_csv.exists():
        train_csv = DATA / "playground-series-s6e11" / "train.csv"
        test_csv = DATA / "playground-series-s6e11" / "test.csv"
    tr, te = rd(train_csv), rd(test_csv)
    og = rd(ORIG)
    cols = [c for c in tr.columns if c not in ("id", "is_fraud")]
    allv = pd.concat([tr[cols], te[cols], og[cols]], ignore_index=True)
    used, per = set(), []
    lab = [_gpt2_encode(" False")[0], _gpt2_encode(" True")[0]]
    for c in cols:
        codes, uniq = pd.factorize(allv[c], sort=False)
        ids = [_gpt2_encode(" " + (str(v).strip() or "None")) for v in uniq]
        assert codes.min() >= 0 and max(map(len, ids)) <= (LMAX - 1), c
        used.update(i for x in ids for i in x)
        per.append((codes, ids))
    vocab = np.array(sorted(used))
    remap = {g: i for i, g in enumerate(vocab)}
    V = len(vocab)
    COL0 = V + 4
    toks = np.full((len(allv), len(cols), LMAX), -1, np.int16)
    for j, (codes, ids) in enumerate(per):
        t = np.full((len(ids), LMAX), -1, np.int16)
        t[:, 0] = COL0 + j
        for i, x in enumerate(ids):
            t[i, 1 : 1 + len(x)] = [remap[g] for g in x]
        toks[:, j, :] = t[codes]
    y = np.r_[
        tr.is_fraud.astype(float).values,
        np.full(len(te), -1.0),
        og.is_fraud.astype(float).values,
    ].astype(np.int8)
    colids = np.full((len(cols), 8), -1, np.int64)
    for j, c in enumerate(cols):
        x = _gpt2_encode(" " + c)[:8]
        colids[j, : len(x)] = x
    np.savez(
        CACHE,
        toks=toks,
        y=y,
        vocab=vocab,
        colids=colids,
        ntr=len(tr),
        nte=len(te),
        lab=np.array(lab),
        comma=np.array(_gpt2_encode(",")),
        pre=np.array([], np.int64),
        labn=np.array([0, 0], np.int64),
    )
    meta = {
        "ntr": int(len(tr)),
        "nte": int(len(te)),
        "nog": int(len(og)),
        "nc": int(len(cols)),
        "V": int(V),
        "T_est": int((toks >= 0).sum((1, 2)).max()) + 1,
        "cols": cols,
        "lab_tok": lab,
        "SEED": SEED,
    }
    (OUT / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_data():
    if not CACHE.exists():
        build_cache()
    z = np.load(CACHE)
    g = dict(
        toks=z["toks"],
        y=z["y"].astype(np.int64),
        vocab=z["vocab"],
        colids=z["colids"],
        ntr=int(z["ntr"]),
        nte=int(z["nte"]),
        lab=z["lab"],
        comma=z["comma"],
    )
    g["V"] = len(g["vocab"])
    g["END"], g["PAD"], g["LAB"], g["COL0"] = g["V"], g["V"] + 1, g["V"] + 2, g["V"] + 4
    g["NC"] = g["toks"].shape[1]
    g["BOS"] = g["COL0"] + g["NC"]
    g["NTOK"] = g["BOS"] + 1
    g["PRE"] = [g["BOS"]] if BOS else []
    g["LABID"] = [g["LAB"], g["LAB"] + 1]
    g["T"] = int((g["toks"] >= 0).sum((1, 2)).max()) + len(g["PRE"]) + 1
    return g


class Rows:
    def __init__(self, g):
        self.t = torch.from_numpy(g["toks"]).to(dev)
        self.g, self.T = g, g["T"]
        self.P = len(g["PRE"])
        self.lens = (g["toks"] >= 0).sum((1, 2)) + self.P + 1
        self.pre = torch.tensor(g["PRE"], dtype=torch.long, device=dev)
        self.labid = torch.tensor(g["LABID"], dtype=torch.long, device=dev)
        self.master = torch.as_tensor(master_perms(g["NC"], SEED), device=dev)

    def rand_perm(self, n, gen):
        return torch.rand(n, self.g["NC"], device=dev, generator=gen).argsort(1)

    def fixed_perm(self, n, order_idx: int):
        return self.master[int(order_idx)].unsqueeze(0).expand(n, -1).contiguous()

    def batches(self, rows, bs, rs, group=64):
        o = rs.permutation(rows)[: len(rows) // bs * bs]
        for c in range(0, len(o), bs * group):
            s = o[c : c + bs * group]
            o[c : c + bs * group] = s[np.argsort(self.lens[s], kind="stable")]
        b = o.reshape(-1, bs)[rs.permutation(len(o) // bs)]
        return torch.as_tensor(b, device=dev), self.lens[b].max(1).tolist()

    def assemble(self, idx, lab, perm, T=None, full=False):
        g, T = self.g, T or self.T
        B = idx.numel()
        seg = self.t[idx].long().gather(1, perm[:, :, None].expand(-1, -1, LMAX)).reshape(B, -1)
        valid = seg >= 0
        pos = torch.where(valid, valid.cumsum(1) + self.P, torch.full_like(seg, T + 1))
        buf = torch.full((B, T + 2), g["PAD"], dtype=torch.long, device=dev)
        buf.scatter_(1, pos, seg.clamp_min(0))
        buf = buf[:, : T + 1]
        if self.P:
            buf[:, : self.P] = self.pre
        buf[:, self.P] = self.labid[lab]
        cur, nxt = buf[:, :-1], buf[:, 1:]
        tgt = torch.where(nxt < g["V"], nxt, torch.full_like(nxt, g["END"]))
        tgt = torch.where(cur == g["PAD"], torch.full_like(tgt, -100), tgt)
        if not full:
            tgt[:, : self.P + 1] = -100
        return cur.contiguous(), tgt


class Block(nn.Module):
    def __init__(s, d, nh, p):
        super().__init__()
        s.ln_1, s.ln_2 = nn.LayerNorm(d, eps=1e-5), nn.LayerNorm(d, eps=1e-5)
        s.qkv, s.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.fc, s.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        s.nh, s.p, s.drop = nh, p, nn.Dropout(p)

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln_1(x)).split(D, 2)
        sh = lambda t: t.view(B, T, s.nh, D // s.nh).transpose(1, 2)
        a = F.scaled_dot_product_attention(
            sh(q), sh(k), sh(v), dropout_p=s.p if s.training else 0.0, is_causal=True
        )
        x = x + s.drop(s.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + s.drop(s.out(F.gelu(s.fc(s.ln_2(x)), approximate="tanh")))


class LM(nn.Module):
    def __init__(s, ntok, nout, T, nl=12, nh=12, d=768, p=0.1):
        super().__init__()
        s.wte, s.wpe = nn.Embedding(ntok, d), nn.Embedding(T, d)
        s.h = nn.ModuleList(Block(d, nh, p) for _ in range(nl))
        s.ln_f, s.drop, s.nout = nn.LayerNorm(d, eps=1e-5), nn.Dropout(p), nout

    def hidden(s, x, lab=None):
        e = s.wte(x) + s.wpe.weight[: x.shape[1]]
        h = s.drop(e if lab is None else e + s.wte(lab)[:, None, :])
        for b in s.h:
            h = b(h)
        return s.ln_f(h)

    def forward(s, x, lab=None):
        h = s.hidden(x, lab).float()
        with torch.autocast("cuda", enabled=False):
            return h @ s.wte.weight[: s.nout].T


def copy_blocks(m, sd):
    P = "transformer."
    with torch.no_grad():
        for i, b in enumerate(m.h):
            q = f"{P}h.{i}."
            for name, key in (("ln_1", "ln_1"), ("ln_2", "ln_2")):
                getattr(b, name).weight.copy_(sd[q + key + ".weight"])
                getattr(b, name).bias.copy_(sd[q + key + ".bias"])
            for name, key in (
                ("qkv", "attn.c_attn"),
                ("proj", "attn.c_proj"),
                ("fc", "mlp.c_fc"),
                ("out", "mlp.c_proj"),
            ):
                getattr(b, name).weight.copy_(sd[q + key + ".weight"].T)
                getattr(b, name).bias.copy_(sd[q + key + ".bias"])
        m.ln_f.weight.copy_(sd[P + "ln_f.weight"])
        m.ln_f.bias.copy_(sd[P + "ln_f.bias"])


def from_gpt2(g):
    from transformers import GPT2LMHeadModel

    sd = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
    m = LM(g["NTOK"], g["V"] + 1, g["T"])
    copy_blocks(m, sd)
    w = sd["transformer.wte.weight"]
    with torch.no_grad():
        m.wpe.weight.copy_(sd["transformer.wpe.weight"][: g["T"]])
        m.wte.weight[: g["V"]] = w[torch.from_numpy(g["vocab"])]
        m.wte.weight[g["END"]] = w[int(g["comma"][0])]
        m.wte.weight[g["BOS"]] = w[50256]
        m.wte.weight[g["PAD"]] = w.mean(0)
        m.wte.weight[g["LAB"] : g["LAB"] + 2] = w[torch.from_numpy(g["lab"])]
        for j, r in enumerate(g["colids"]):
            m.wte.weight[g["COL0"] + j] = w[torch.from_numpy(r[r >= 0])].mean(0)
    return m.to(dev)


def save_atomic(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(m, R, ytr, rows, epochs, bs, lr, name, seed):
    per = len(rows) // bs
    total = max(per * epochs, 1)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0, fused=(dev == "cuda"))
    ck, start = CK / f"{name}.pt", 0
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        m.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        start = int(s["epoch"]) + 1
        log(f"{name}: resumed after epoch {start}")
    gen = torch.Generator(device=dev)
    m.train()
    t0, seen, run = time.time(), 0, torch.zeros((), device=dev)
    for ep in range(start, epochs):
        run.zero_()
        gen.manual_seed(seed * 1000 + ep)
        bt, tb = R.batches(rows, bs, np.random.RandomState((seed * 1000 + ep) % 2**31))
        for s in range(per):
            step = ep * per + s
            warm = min(1.0, (step + 1) / 100)
            decay = 0.5 * (1.0 + np.cos(np.pi * step / total))
            for grp in opt.param_groups:
                grp["lr"] = lr * warm * decay
            idx = bt[s]
            inp, tgt = R.assemble(idx, ytr[idx], R.rand_perm(bs, gen), tb[s])
            if dev == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = m(inp)
            else:
                lg = m(inp)
            loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += loss.detach()
            seen += bs
            if (s + 1) % 2000 == 0:
                log(
                    f"{name} ep {ep + 1}/{epochs} step {s + 1}/{per} loss {run.item() / 2000:.4f} "
                    f"{seen / (time.time() - t0):.0f} rows/s"
                )
                run.zero_()
        save_atomic({"model": m.state_dict(), "opt": opt.state_dict(), "epoch": ep}, ck)
        log(f"{name}: epoch {ep + 1}/{epochs} done, {(time.time() - t0) / 60:.1f} min")
    return seen / max(time.time() - t0, 1e-6)


@torch.no_grad()
def score_orders(m, R, rows, order_ids, bs=512, dtype=torch.bfloat16):
    """Score fixed master-perm orders → float32 [n, K, 2] log-likelihoods. No grads."""
    m.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    srt = np.argsort(R.lens[rows], kind="stable")
    ls = R.lens[rows][srt]
    rows_t = torch.as_tensor(rows[srt], device=dev)
    K = len(order_ids)
    out = torch.zeros((len(rows), K, 2), dtype=torch.float32, device=dev)
    for ki, oid in enumerate(order_ids):
        for s in range(0, len(rows), bs):
            idx = rows_t[s : s + bs]
            n = idx.numel()
            perm = R.fixed_perm(n, oid)
            inp, tgt = R.assemble(idx, torch.zeros(n, dtype=torch.long, device=dev), perm, int(ls[s + n - 1]))
            both = torch.cat([inp, inp])
            both[n:, R.P] = R.labid[1]
            if dev == "cuda":
                with torch.autocast("cuda", dtype=dtype):
                    lg = m(both)
            else:
                lg = m(both)
            nll = (
                F.cross_entropy(
                    lg.float().view(-1, lg.shape[-1]),
                    tgt.repeat(2, 1).reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                .view(2 * n, -1)
                .sum(1)
            )
            out[s : s + n, ki, 0] = -nll[:n]
            out[s : s + n, ki, 1] = -nll[n:]
    res = np.empty((len(rows), K, 2), np.float32)
    res[srt] = out.cpu().numpy()
    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def maybe_compile(m):
    if USE_COMPILE and hasattr(torch, "compile") and dev == "cuda":
        try:
            m = torch.compile(m, mode="reduce-overhead")
            log("torch.compile enabled (reduce-overhead)")
        except Exception as e:
            log(f"torch.compile skipped: {e}")
    return m


def bayes(ll, prior):
    t = np.logaddexp.reduce(ll[:, :, 1].astype(np.float64), axis=1)
    f = np.logaddexp.reduce(ll[:, :, 0].astype(np.float64), axis=1)
    return t - f + np.log(prior / (1 - prior))


def save_npy_verified(path: Path, arr: np.ndarray, expect_shape):
    assert arr.shape == expect_shape, (arr.shape, expect_shape)
    assert arr.dtype == np.float32
    assert np.isfinite(arr).all()
    np.save(path, arr)
    reload = np.load(path)
    assert reload.shape == expect_shape and np.isfinite(reload).all()
    log(f"  wrote {path.name} shape={reload.shape} nbytes={reload.nbytes} nan=0")
    return reload.nbytes


def replica(g, R, ydev):
    p = CK / "replica.pt"
    rows = np.arange(g["ntr"] + g["nte"], len(g["y"]))
    if int(os.environ.get("SKIP_REPLICA", "0")):
        log("replica: SKIP_REPLICA=1 → init from gpt2 only")
        return from_gpt2(g)
    if p.exists():
        try:
            obj = torch.load(p, map_location=dev, weights_only=False)
            if isinstance(obj, dict) and "model" in obj and "epoch" in obj:
                ep_done = int(obj["epoch"]) + 1
                log(f"replica: found ckpt after epoch {ep_done}/{REP_EPOCHS}")
                if ep_done >= REP_EPOCHS:
                    m = LM(g["NTOK"], g["V"] + 1, g["T"]).to(dev)
                    m.load_state_dict(obj["model"])
                    log("replica: complete — loaded final weights")
                    return m
                log(f"replica: resuming from epoch {ep_done}")
            else:
                m = LM(g["NTOK"], g["V"] + 1, g["T"]).to(dev)
                m.load_state_dict(obj["model"] if isinstance(obj, dict) and "model" in obj else obj)
                log("replica: loaded bare state_dict (assumed complete)")
                return m
        except Exception as e:
            log(f"replica: ckpt load failed ({e}); deleting and retraining")
            p.unlink(missing_ok=True)
    m = from_gpt2(g)
    train(m, R, ydev, rows, REP_EPOCHS, REP_BATCH, LR, "replica", SEED)
    save_atomic(m.state_dict(), p)
    return m


def _rank_logit(x):
    from scipy.special import logit
    from scipy.stats import rankdata

    return logit((rankdata(x, method="average") - 0.5) / len(x))


def fold0_three_gates(y, fold_rows, ll_s2026, prior, wall_min: float) -> dict:
    """Gate1 solo/corr, Gate2 two-seed, Gate3 cand2 stack ΔF0 + wall."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _gm_v11_shared import L_fold, load_anchor_and_bank, setup_folds

    z26 = bayes(ll_s2026, prior)
    auc26 = float(roc_auc_score(y[fold_rows], z26))

    # seed7 fold0 OOF (val-only 139992×10×2)
    s7_path = L7_S7_DIR / "lm7_oof_f0.npy"
    alt = [
        Path("external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_f0.npy"),
        ROOT / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_f0.npy",
    ]
    if not s7_path.exists():
        for a in alt:
            if a.exists():
                s7_path = a
                break
    assert s7_path.exists(), s7_path
    ll7 = np.load(s7_path).astype(np.float32)
    assert ll7.shape == ll_s2026.shape, (ll7.shape, ll_s2026.shape)
    # prior for seed7 fold0 = train mean on complementary fold
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    a0, b0 = folds[0]
    assert np.array_equal(b0, fold_rows)
    prior7 = float(y[a0].mean())
    z7 = bayes(ll7, prior7)
    auc7 = float(roc_auc_score(y[fold_rows], z7))
    corr = float(np.corrcoef(z26, z7)[0, 1])

    gate1 = bool(auc26 >= GATE1_AUC_MIN and GATE1_CORR_LO <= corr <= GATE1_CORR_HI)

    # Gate2: avg of rank-logits
    z_avg = 0.5 * _rank_logit(z26) + 0.5 * _rank_logit(z7)
    auc_avg = float(roc_auc_score(y[fold_rows], z_avg))
    gate2 = bool(auc_avg > auc7 + GATE2_DELTA)

    # Gate3: stack residual onto eval_S7matched_355 fold0
    folds_full, fold_id = setup_folds(y)
    pack = load_anchor_and_bank(y, fold_id, folds_full)
    L26 = _rank_logit(z26)
    L7 = _rank_logit(z7)
    # also try vs LM_f0
    mega_path = None
    for cand in [
        L7_S7_DIR / "megalm_oof_sha181c.npy",
        Path("external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/megalm_oof_sha181c.npy"),
    ]:
        if cand.exists():
            mega_path = cand
            break
    delta_vs_l7 = L26 - L7
    base = pack["eval_S7matched_355"][fold_rows]
    yf = y[fold_rows]
    base_auc = float(roc_auc_score(yf, base))
    best_d, best_w, best_tag = -1e9, None, None
    for tag, dlt in (("vs_L7s7", delta_vs_l7),):
        for w in (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
            d = float(roc_auc_score(yf, base + w * dlt) - base_auc)
            if d > best_d:
                best_d, best_w, best_tag = d, w, tag
    if mega_path is not None:
        Lm = _rank_logit(np.load(mega_path).astype(np.float64).ravel()[fold_rows])
        dlt = L26 - Lm
        for w in (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
            d = float(roc_auc_score(yf, base + w * dlt) - base_auc)
            if d > best_d:
                best_d, best_w, best_tag = d, w, "vs_LM"
    # Wall clock is logged only — never a kill/fail gate (user override).
    wall_ok = wall_min <= WALL_F0_MAX_MIN
    gate3 = bool(best_d > GATE3_STACK_MIN)

    info = {
        "auc_s2026": auc26,
        "auc_s7": auc7,
        "corr": corr,
        "auc_two_seed": auc_avg,
        "two_seed_delta": auc_avg - auc7,
        "stack_best_d": best_d,
        "stack_best_w": best_w,
        "stack_best_tag": best_tag,
        "wall_min": wall_min,
        "wall_ok": wall_ok,
        "wall_gate_disabled": True,
        "gate1": gate1,
        "gate2": gate2,
        "gate3": gate3,
        "pass": bool(gate1 and gate2 and gate3),
        "NO_SUBMIT": True,
    }
    return info


def run_folds(g, which, stop_after_f0_canary=True):
    cover = verify_test_cover()
    log(f"test cover folds1-4 counts={cover}")
    disj = verify_orders_disjoint_from_seed7(g["NC"])
    log(f"orders vs seed7: {disj}")
    assert disj["disjoint"], disj

    R = Rows(g)
    y = g["y"]
    ydev = torch.as_tensor(np.maximum(y, 0), device=dev)
    ntr, nte = g["ntr"], g["nte"]
    test = np.arange(ntr, ntr + nte)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y[:ntr]))
    P = master_perms(g["NC"], SEED)
    np.save(OUT / "master_perms_seed2026.npy", P)
    (OUT / "test_orders.json").write_text(json.dumps({str(k): list(v) for k, v in TEST_ORDERS.items()}))

    t_campaign = time.time()
    m_rep = replica(g, R, ydev)
    log(f"replica ready, {(time.time() - t_campaign) / 60:.1f} min")
    del m_rep
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for f in which:
        f = int(f)
        if (OUT / f"test_{f}.npy").exists() and (OUT / f"oof_{f}.npy").exists():
            log(f"fold {f}: already scored")
            if f == 0 and stop_after_f0_canary:
                # still run canary on existing
                pass
            else:
                continue
        t1 = time.time()
        a, b = folds[f]
        need_train = not ((OUT / f"oof_{f}.npy").exists() and (OUT / f"test_{f}.npy").exists())
        if need_train:
            m = replica(g, R, ydev)
            train(m, R, ydev, a, EPOCHS, BATCH, LR, f"fold_{f}", SEED + f)
            m = maybe_compile(m)
            t2 = time.time()
            prior = float(y[a].mean())
            # OOF: all 10 orders — FIXED saver: val-only (len(b), 10, 2)
            ll = score_orders(m, R, b, list(range(K_OOF)), bs=SCORE_BS)
            assert ll.shape == (len(b), K_OOF, 2)
            save_npy_verified(OUT / f"oof_{f}.npy", ll, (len(b), K_OOF, 2))
            z = bayes(ll, prior)
            auc = float(roc_auc_score(y[b], z))
            log(f"fold {f}: AUC {auc:.6f}")
            # TEST
            orders = list(TEST_ORDERS[f])
            lt = score_orders(m, R, test, orders, bs=SCORE_BS)
            save_npy_verified(OUT / f"test_{f}.npy", lt, (nte, len(orders), 2))
            wall_so_far = (time.time() - t_campaign) / 60
            (OUT / f"fold_{f}_meta.json").write_text(
                json.dumps(
                    {
                        "fold": f,
                        "auc": auc,
                        "prior": prior,
                        "test_orders": orders,
                        "oof_shape": list(ll.shape),
                        "test_shape": list(lt.shape),
                        "train_min": (t2 - t1) / 60,
                        "score_min": (time.time() - t2) / 60,
                        "wall_from_start_min": wall_so_far,
                        "SCORE_BS": SCORE_BS,
                        "SEED": SEED,
                    },
                    indent=2,
                )
            )
            log(
                f"fold {f}: train {(t2 - t1) / 60:.1f} min, score {(time.time() - t2) / 60:.1f} min, "
                f"wall {wall_so_far:.1f} min"
            )
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            ll = np.load(OUT / f"oof_{f}.npy")
            prior = float(y[a].mean())
            wall_so_far = (time.time() - t_campaign) / 60

        if f == 0:
            if int(os.environ.get("CANARY_LOCAL_ONLY", "0")) == 1:
                meta_wall = {"wall_min": wall_so_far, "NO_SUBMIT": True, "pending_local_3gate": True}
                (OUT / "F0_READY_FOR_LOCAL_CANARY").write_text(json.dumps(meta_wall, indent=2))
                log(f"F0 scored — CANARY_LOCAL_ONLY=1 wall={wall_so_far:.1f}m; rsync oof_0/test_0 for Mac 3-gate")
                if stop_after_f0_canary and max(int(x) for x in which) == 0:
                    return True
            else:
                info = fold0_three_gates(
                    y[:ntr], b, ll if need_train else np.load(OUT / "oof_0.npy"), prior, wall_so_far
                )
                (OUT / "fold0_canary_3gate.json").write_text(json.dumps(info, indent=2))
                log(f"FOLD0 3-GATE CANARY: {info}")
                if not info["pass"]:
                    log("CANARY FAIL — stop; kill pod (do not continue folds 1–4)")
                    (OUT / "CANARY_FAIL").write_text(json.dumps(info, indent=2))
                    return False
                (OUT / "CANARY_PASS").write_text(json.dumps(info, indent=2))
                if stop_after_f0_canary and max(int(x) for x in which) == 0:
                    return True
    log("all requested folds finished")
    (OUT / "DONE").write_text(json.dumps({"wall_min": (time.time() - t_campaign) / 60}))
    return True


def phase0_cpu():
    """1-batch CPU unit test across f=0..4 + shape/saver contracts → PHASE0_S2026_PASS.json."""
    t0 = time.time()
    report = {"NO_SUBMIT": True, "SEED": SEED, "checks": {}}
    cover = verify_test_cover()
    report["checks"]["test_cover"] = cover
    # perms antithetic + disjoint
    nc_probe = 32
    P = master_perms(nc_probe, SEED)
    assert P.shape == (10, nc_probe)
    for k in range(5):
        assert np.array_equal(P[k + 5], P[k][::-1]), k
    assert len({tuple(r.tolist()) for r in P}) == 10
    disj = verify_orders_disjoint_from_seed7(nc_probe)
    report["checks"]["disjoint_seed7_nc32"] = disj
    assert disj["disjoint"]

    # need real cache for V/T — resolve local Mac paths if /workspace absent
    global DATA, ORIG, CACHE, OUT, CK, L7_S7_DIR
    repo = Path(__file__).resolve().parents[1]
    if not (DATA / "train.csv").exists() and (repo / "data" / "train.csv").exists():
        DATA = repo / "data"
        ORIG = repo / "original" / "credit_card_fraud_2026.csv"
        if not ORIG.exists():
            ORIG = repo / "external/nov12_late_recon/detective_data/credit_card_fraud_2026.csv"
        CACHE = repo / "cache_compact.npz"
        OUT = repo / "preds" / "l7_s2026_5f"
        CK = repo / "ckpt" / "l7_s2026_5f"
        L7_S7_DIR = repo / "external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7"
        OUT.mkdir(parents=True, exist_ok=True)
        CK.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        build_cache()
    g = load_data()
    report["checks"]["V"] = g["V"]
    report["checks"]["T"] = g["T"]
    report["checks"]["NTOK"] = g["NTOK"]
    report["checks"]["NC"] = g["NC"]
    assert g["V"] == 1372, g["V"]
    assert 66 <= g["T"] <= 80, g["T"]

    disj2 = verify_orders_disjoint_from_seed7(g["NC"])
    report["checks"]["disjoint_seed7"] = disj2
    assert disj2["disjoint"], disj2
    np.save(OUT / "master_perms_seed2026.npy", master_perms(g["NC"], SEED))

    R = Rows(g)
    y = g["y"]
    ntr, nte = g["ntr"], g["nte"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y[:ntr]))
    # tiny LM for shape-only forward (random init, 1 layer would change T — use full but 1 batch CPU)
    m = LM(g["NTOK"], g["V"] + 1, g["T"]).to(dev)
    m.eval()
    for f in range(5):
        a, b = folds[f]
        # 1-batch assemble train
        idx = torch.as_tensor(a[:4], device=dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(SEED + f)
        perm = R.rand_perm(4, gen)
        inp, tgt = R.assemble(idx, torch.as_tensor(np.maximum(y[a[:4]], 0), device=dev), perm)
        with torch.no_grad():
            lg = m(inp)
        assert lg.shape == (4, inp.shape[1], g["V"] + 1)
        # fake oof/test shapes via save_npy_verified dry-run arrays
        k_test = len(TEST_ORDERS[f])
        oof_fake = np.zeros((len(b), 10, 2), np.float32)
        test_fake = np.zeros((nte, k_test, 2), np.float32)
        # write to tmp then remove — verifies saver contract
        op = OUT / f"_phase0_oof_{f}.npy"
        tp = OUT / f"_phase0_test_{f}.npy"
        save_npy_verified(op, oof_fake, (len(b), 10, 2))
        save_npy_verified(tp, test_fake, (nte, k_test, 2))
        op.unlink()
        tp.unlink()
        report["checks"][f"fold{f}_oof_shape"] = [len(b), 10, 2]
        report["checks"][f"fold{f}_test_shape"] = [nte, k_test, 2]
        report["checks"][f"fold{f}_n_val"] = int(len(b))
        log(f"phase0 fold{f}: oof={(len(b),10,2)} test={(nte,k_test,2)} fwd_ok")

    # seed7 reference present for canary
    s7 = None
    for cand in [
        L7_S7_DIR / "lm7_oof_f0.npy",
        Path("external/nov12_095369_recon/datasets/goodpjw2008__s6e11-compact-lm-seed7/lm7_oof_f0.npy"),
    ]:
        if cand.exists():
            s7 = cand
            break
    assert s7 is not None and np.load(s7).shape == (folds[0][1].shape[0], 10, 2)
    report["checks"]["l7_s7_f0"] = str(s7)
    report["SCORE_BS"] = SCORE_BS
    report["elapsed_s"] = round(time.time() - t0, 2)
    report["pass"] = True
    outp = OUT / "PHASE0_S2026_PASS.json"
    # also write to logs/overnight for Mac visibility
    alt = Path("logs/overnight/PHASE0_S2026_PASS.json")
    alt.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    alt.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("PHASE0_S2026_PASS", flush=True)
    return report


def canary_check_local():
    """Re-run 3-gate on synced oof_0.npy without GPU."""
    oof = np.load(OUT / "oof_0.npy")
    meta = json.loads((OUT / "fold_0_meta.json").read_text()) if (OUT / "fold_0_meta.json").exists() else {}
    # load y
    g = load_data()
    ntr = g["ntr"]
    y = g["y"][:ntr]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y))
    a, b = folds[0]
    prior = float(meta.get("prior", y[a].mean()))
    wall = float(meta.get("wall_from_start_min", 0))
    info = fold0_three_gates(y, b, oof, prior, wall)
    print(json.dumps(info, indent=2))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["phase0", "build_cache", "run", "canary_check"])
    ap.add_argument("--folds", default=os.environ.get("FOLDS", "0"))
    ap.add_argument("--continue-after-canary", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if a.continue_after_canary:
        os.environ["CONTINUE_AFTER_CANARY"] = "1"
    if a.mode == "phase0":
        phase0_cpu()
        return
    if a.mode == "build_cache":
        print(json.dumps(build_cache(), indent=2))
        return
    if a.mode == "canary_check":
        info = canary_check_local()
        sys.exit(0 if info["pass"] else 2)
    assert torch.cuda.is_available(), "CUDA required for run mode"
    g = load_data()
    log(
        f"start SEED={SEED} epochs={EPOCHS} batch={BATCH} SCORE_BS={SCORE_BS} "
        f"T={g['T']} V={g['V']} NTOK={g['NTOK']} gpu={torch.cuda.get_device_name(0)}"
    )
    which = [x for x in a.folds.split(",") if x != ""]
    stop = not a.continue_after_canary
    ok = run_folds(g, which, stop_after_f0_canary=stop)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
