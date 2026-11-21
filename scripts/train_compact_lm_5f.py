#!/usr/bin/env python3
"""Compact-LM (L7) 5-fold rotated-10 trainer — A40 campaign.

Architecture match for goodpjw2008/s6e11-compact-lm-seed7:
  GPT-2 small (12L/12H/768d), compact one-token-per-column row format (megayak lm_member.py),
  target-first scoring via full-sequence NLL under label False/True for K column orders.
  Output tensors: OOF (139992, K_oof, 2), TEST (299984, 4, 2) float32 log-likelihoods.

Rotated 4-of-10 test design (seed=7 master perms P_0..P_9):
  Fold0 test orders {0,1,2,3}  — already on disk as lm7_test_f0.npy (skip train)
  Fold1 test orders {4,5,6,7}
  Fold2 test orders {8,9,0,1}
  Fold3 test orders {2,3,4,5}
  Fold4 test orders {6,7,8,9}
  Averaged across folds: every test row sees all 10 orders twice (20 evals).

Env:
  SEED=7 EPOCHS=8 BATCH=64 LR=5e-5 K_OOF=10 K_TEST=4
  L7_ROOT=/workspace/l7_work  L7_OUT=/workspace/preds/l7_5f_rot10
  FOLDS=1,2,3,4  SKIP_REPLICA=0  CANARY_MIN_AUC=0.9480
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
ROOT = Path(os.environ.get("L7_ROOT", "/workspace/l7_work"))
DATA = Path(os.environ.get("L7_DATA", str(ROOT / "data")))
ORIG = Path(os.environ.get("L7_ORIG", str(ROOT / "original" / "credit_card_fraud_2026.csv")))
OUT = Path(os.environ.get("L7_OUT", str(ROOT / "preds" / "l7_5f_rot10")))
CK = Path(os.environ.get("L7_CKPT", str(ROOT / "ckpt" / "l7_5f_rot10")))
CACHE = Path(os.environ.get("L7_CACHE", str(ROOT / "cache_compact.npz")))
F0_DIR = Path(
    os.environ.get(
        "L7_F0_DIR",
        str(ROOT / "external" / "goodpjw2008_s6e11-compact-lm-seed7"),
    )
)
for d in (OUT, CK):
    d.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("SEED", 7))
EPOCHS = int(os.environ.get("EPOCHS", 8))
BATCH = int(os.environ.get("BATCH", 64))
LR = float(os.environ.get("LR", 5e-5))
REP_EPOCHS = int(os.environ.get("REP_EPOCHS", 60))
REP_BATCH = int(os.environ.get("REP_BATCH", 16))
K_OOF = int(os.environ.get("K_OOF", 10))
K_TEST = int(os.environ.get("K_TEST", 4))
SCORE_BS = int(os.environ.get("SCORE_BS", 256))
CANARY_MIN_AUC = float(os.environ.get("CANARY_MIN_AUC", "0.9480"))
CANARY_ORTH_MIN = float(os.environ.get("CANARY_ORTH_MIN", "2.0e-5"))
BOS = 0
LABEMB = 0
NATURAL = 0
LMAX = 6
dev = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False

# Fold → 4 master-perm indices for TEST scoring (rotated cover of 0..9)
TEST_ORDERS = {
    0: (0, 1, 2, 3),
    1: (4, 5, 6, 7),
    2: (8, 9, 0, 1),
    3: (2, 3, 4, 5),
    4: (6, 7, 8, 9),
}


def log(*a):
    s = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(OUT / "run.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def master_perms(nc: int, seed: int = 7) -> np.ndarray:
    """10 deterministic column permutations P_0..P_9, shape (10, nc)."""
    rs = np.random.RandomState(seed)
    base = np.arange(nc, dtype=np.int64)
    out = np.empty((10, nc), dtype=np.int64)
    for k in range(10):
        out[k] = rs.permutation(base)
    return out


def verify_rotation_cover():
    seen = {i: 0 for i in range(10)}
    for f, orders in TEST_ORDERS.items():
        assert len(orders) == 4 and len(set(orders)) == 4, (f, orders)
        for o in orders:
            seen[o] += 1
    assert all(v == 2 for v in seen.values()), seen
    return seen


# ---------------------------------------------------------------- serializer / cache
def build_cache():
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    train_csv = DATA / "train.csv"
    test_csv = DATA / "test.csv"
    if not train_csv.exists():
        # alternate layout: playground-series-s6e11/
        train_csv = DATA / "playground-series-s6e11" / "train.csv"
        test_csv = DATA / "playground-series-s6e11" / "test.csv"
    tr, te = rd(train_csv), rd(test_csv)
    og = rd(ORIG)
    cols = [c for c in tr.columns if c not in ("id", "is_fraud")]
    allv = pd.concat([tr[cols], te[cols], og[cols]], ignore_index=True)
    used, per = set(), []
    pre = tok.encode("is_fraud is")
    lab = [tok.encode(" False")[0], tok.encode(" True")[0]]
    for c in cols:
        codes, uniq = pd.factorize(allv[c], sort=False)
        ids = [tok.encode(" " + (str(v).strip() or "None")) for v in uniq]
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
        x = tok.encode(" " + c)[:8]
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
        comma=np.array(tok.encode(",")),
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
        self.master = torch.as_tensor(master_perms(g["NC"], SEED), device=dev)  # (10, NC)

    def rand_perm(self, n, gen):
        return torch.rand(n, self.g["NC"], device=dev, generator=gen).argsort(1)

    def fixed_perm(self, n, order_idx: int):
        p = self.master[int(order_idx)].unsqueeze(0).expand(n, -1)
        return p.contiguous()

    def batches(self, rows, bs, rs, group=64):
        o = rs.permutation(rows)[: len(rows) // bs * bs]
        for c in range(0, len(o), bs * group):
            s = o[c : c + bs * group]
            o[c : c + bs * group] = s[np.argsort(self.lens[s], kind="stable")]
        b = o.reshape(-1, bs)[rs.permutation(len(o) // bs)]
        return torch.as_tensor(b, device=dev), self.lens[b].max(1).tolist()

    def assemble_any(self, idx, lab, gen, T):
        raise RuntimeError("NATURAL/anywhere not used in compact L7")

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
        s = torch.load(ck, map_location=dev, weights_only=False)  # trusted local ckpt
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
            # warmup 100 steps + linear cosine-ish decay to 0
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
def score_orders(m, R, rows, order_ids, bs=256, dtype=torch.float16):
    """Score fixed master-perm orders → float32 [n, K, 2] log-likelihoods."""
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
    # train() checkpoints to CK/replica.pt with {"model","opt","epoch"}; resume if incomplete
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
                # incomplete: let train() resume via same ckpt path
                log(f"replica: resuming from epoch {ep_done}")
            else:
                # bare state_dict = treated as finished
                m = LM(g["NTOK"], g["V"] + 1, g["T"]).to(dev)
                m.load_state_dict(obj["model"] if isinstance(obj, dict) and "model" in obj else obj)
                log("replica: loaded bare state_dict (assumed complete)")
                return m
        except Exception as e:
            log(f"replica: ckpt load failed ({e}); deleting and retraining")
            p.unlink(missing_ok=True)
    m = from_gpt2(g)
    train(m, R, ydev, rows, REP_EPOCHS, REP_BATCH, LR, "replica", SEED)
    # final bare weights for fast reload
    save_atomic(m.state_dict(), p)
    return m


def fold_canary(y, oof_ll, fold_rows, prior, z_ref=None, d1=None, mega=None):
    """Solo AUC on fold + optional orth lift vs megayak on z_ref+d1."""
    z = bayes(oof_ll, prior)
    auc = float(roc_auc_score(y[fold_rows], z))
    info = {"auc": auc, "orth_e5": None, "pass": auc >= CANARY_MIN_AUC}
    if z_ref is not None and d1 is not None and mega is not None:
        # rank-logit residual of L7 vs mega, lift on base
        from scipy.special import logit
        from scipy.stats import rankdata

        def zrk(x):
            return logit((rankdata(x, method="average") - 0.5) / len(x))

        idx = fold_rows
        zl = zrk(z)
        zm = zrk(mega[idx])
        # beta = OLS of zl on zm
        beta = float(np.dot(zl - zl.mean(), zm - zm.mean()) / (np.dot(zm - zm.mean(), zm - zm.mean()) + 1e-12))
        r = zl - beta * zm
        base = z_ref[idx] + d1[idx]
        # search small positive weight
        best = -1e9
        for w in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
            d = roc_auc_score(y[idx], base + w * r) - roc_auc_score(y[idx], base)
            best = max(best, d)
        info["orth_e5"] = best * 1e5
        info["pass"] = info["pass"] and (best > CANARY_ORTH_MIN)
        info["beta"] = beta
    return info


def run_folds(g, which):
    cover = verify_rotation_cover()
    log(f"rotation cover counts={cover}")
    R = Rows(g)
    y = g["y"]
    ydev = torch.as_tensor(np.maximum(y, 0), device=dev)
    ntr, nte = g["ntr"], g["nte"]
    test = np.arange(ntr, ntr + nte)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(ntr), y[:ntr]))
    # persist master perms
    np.save(OUT / "master_perms_seed7.npy", master_perms(g["NC"], SEED))
    (OUT / "test_orders.json").write_text(json.dumps({str(k): list(v) for k, v in TEST_ORDERS.items()}))

    # Fold0 already on disk — copy/symlink into OUT
    f0_o = F0_DIR / "lm7_oof_f0.npy"
    f0_t = F0_DIR / "lm7_test_f0.npy"
    if f0_o.exists() and f0_t.exists():
        o0 = np.load(f0_o)
        t0 = np.load(f0_t)
        # OOF for fold0 is only the held-out rows of fold0 stored as (139992,10,2) in train order —
        # goodpjw stores full-train-length with fold0 OOF rows filled. Keep as oof_0.npy identically.
        save_npy_verified(OUT / "oof_0.npy", o0.astype(np.float32), (139992, 10, 2))
        save_npy_verified(OUT / "test_0.npy", t0.astype(np.float32), (299984, 4, 2))
        log("fold 0: reused lm7_* from disk (SKIP train)")
    else:
        log(f"WARN: fold0 assets missing under {F0_DIR}")

    # optional canary bases
    z_ref = d1 = mega = None
    for cand in [
        ROOT / "preds" / "gm_v5_overnight" / "z_ref_old_oof.npy",
        Path("/workspace/preds/gm_v5_overnight/z_ref_old_oof.npy"),
    ]:
        if cand.exists():
            z_ref = np.load(cand).astype(np.float64).ravel()
            break
    for cand in [
        ROOT / "preds" / "gm_v4_polish" / "d1_v4_oof.npy",
        Path("/workspace/preds/gm_v4_polish/d1_v4_oof.npy"),
    ]:
        if cand.exists():
            d1 = np.load(cand).astype(np.float64).ravel()
            break
    for cand in [
        ROOT / "external" / "banks" / "megayak_s6e11-target-first-lm" / "lm_oof.npy",
        Path("/workspace/external/megayak_s6e11-target-first-lm/lm_oof.npy"),
    ]:
        if cand.exists():
            mega = np.load(cand).astype(np.float64).ravel()
            break

    t0 = time.time()
    m_rep = replica(g, R, ydev)
    log(f"replica ready, {(time.time() - t0) / 60:.1f} min")
    del m_rep
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for f in which:
        f = int(f)
        done = OUT / f"test_{f}.npy"
        if done.exists() and (OUT / f"oof_{f}.npy").exists():
            log(f"fold {f}: already scored")
            continue
        t1 = time.time()
        a, b = folds[f]
        m = replica(g, R, ydev)
        train(m, R, ydev, a, EPOCHS, BATCH, LR, f"fold_{f}", SEED + f)
        t2 = time.time()
        prior = float(y[a].mean())
        # OOF: all 10 master perms
        ll = score_orders(m, R, b, list(range(K_OOF)), bs=SCORE_BS)
        # Place into full-length (139992, K, 2) like lm7_oof_f0
        oof_full = np.zeros((ntr, K_OOF, 2), np.float32)
        oof_full[b] = ll
        save_npy_verified(OUT / f"oof_{f}.npy", oof_full, (139992, K_OOF, 2))
        z = bayes(ll, prior)
        auc = float(roc_auc_score(y[b], z))
        log(f"fold {f}: AUC {auc:.6f} (one order {roc_auc_score(y[b], ll[:, 0, 1] - ll[:, 0, 0]):.6f})")
        # TEST: rotated 4
        orders = list(TEST_ORDERS[f])
        assert len(orders) == K_TEST
        lt = score_orders(m, R, test, orders, bs=SCORE_BS)
        save_npy_verified(OUT / f"test_{f}.npy", lt, (299984, K_TEST, 2))
        (OUT / f"fold_{f}_meta.json").write_text(
            json.dumps(
                {
                    "fold": f,
                    "auc": auc,
                    "prior": prior,
                    "test_orders": orders,
                    "train_min": (t2 - t1) / 60,
                    "score_min": (time.time() - t2) / 60,
                },
                indent=2,
            )
        )
        log(f"fold {f}: train {(t2 - t1) / 60:.1f} min, score {(time.time() - t2) / 60:.1f} min")

        if f == 1:
            info = fold_canary(y, ll, b, prior, z_ref, d1, mega)
            (OUT / "fold1_canary.json").write_text(json.dumps(info, indent=2))
            log(f"FOLD1 CANARY: {info}")
            if not info["pass"]:
                log("CANARY FAIL — stopping before folds 2–4")
                del m
                return False
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log("all requested folds finished")
    return True


def preflight_cpu():
    """Mac/CPU: serializer + rotation math + shape contracts. No GPU spend."""
    t0 = time.time()
    cover = verify_rotation_cover()
    print("rotation_cover", cover)
    P = master_perms(32, 7)  # NC checked after cache; 32 placeholder until build
    assert P.shape[0] == 10 and len({tuple(r) for r in P}) == 10
    meta = build_cache() if not CACHE.exists() else json.loads((OUT / "cache_meta.json").read_text()) if (OUT / "cache_meta.json").exists() else build_cache()
    if not CACHE.exists():
        meta = build_cache()
    g = load_data()
    P = master_perms(g["NC"], SEED)
    np.save(OUT / "master_perms_seed7.npy", P)
    # assemble one CPU batch
    R = Rows(g)
    idx = torch.tensor([0, 1, 2, 3], device=dev)
    perm = R.fixed_perm(4, 0)
    inp, tgt = R.assemble(idx, torch.tensor([0, 1, 0, 1], device=dev), perm)
    assert inp.shape[0] == 4 and tgt.shape == inp.shape
    # fold0 shape contract
    f0_o = F0_DIR / "lm7_oof_f0.npy"
    f0_t = F0_DIR / "lm7_test_f0.npy"
    assert f0_o.exists() and f0_t.exists(), F0_DIR
    o = np.load(f0_o)
    t = np.load(f0_t)
    assert o.shape == (139992, 10, 2) and t.shape == (299984, 4, 2)
    assert o.dtype == np.float32 and t.dtype == np.float32
    assert np.isfinite(o).all() and np.isfinite(t).all()
    # A40 estimate anchored on our prior A40 compact/GPT-2 campaigns
    # (monitor_tick*: ~960–1000 rows/s @ BATCH=32 → ~9.9 min/epoch; score≈20.5 min for K_oof=10+K_test=4).
    # BATCH=64 on 48GB A40 → ~1.7× train throughput vs BATCH=32.
    ntr = g["ntr"]
    n_train = int(0.8 * ntr)
    rows_s_b32 = 970.0
    train_min_b32_ep = (n_train / rows_s_b32) / 60.0
    batch_speedup = max(1.0, BATCH / 32.0) * 0.85  # sub-linear scaling
    train_min_fold = train_min_b32_ep * EPOCHS / batch_speedup
    # Empirical score wall-clock on A40 for (140k×10 + 300k×4) both-label passes:
    score_min_fold = 20.5 * ((K_OOF / 10.0 + K_TEST / 4.0) / 2.0)
    # Replica on 20k orig × 60 ep @ REP_BATCH=16 ≈ 35 min on A40 (conservative)
    replica_min = 35.0
    folds_todo = 4
    total_min = replica_min + folds_todo * (train_min_fold + score_min_fold)
    total_h = total_min / 60.0
    cost = total_h * 0.59
    # Aggressive but still safe plan if borderline: SCORE_BS=512, BATCH=96
    train_min_b96 = train_min_b32_ep * EPOCHS / (max(1.0, 96 / 32.0) * 0.80)
    total_h_b96 = (replica_min + folds_todo * (train_min_b96 + score_min_fold * 0.85)) / 60.0
    est = {
        "device_assumed": "A40",
        "anchor": "A40 monitor ~970 rows/s @bs32; score≈20.5m/fold; BATCH=64 → ~1.7× train",
        "epochs": EPOCHS,
        "batch": BATCH,
        "train_min_per_fold": train_min_fold,
        "score_min_per_fold": score_min_fold,
        "replica_min": replica_min,
        "folds": folds_todo,
        "total_hours": total_h,
        "cost_usd_at_0p59": cost,
        "alt_batch96_hours": total_h_b96,
        "alt_batch96_cost": total_h_b96 * 0.59,
        "budget_ok_le_6p5h": (total_h <= 6.5) or (total_h_b96 <= 6.5),
        "recommended_batch": BATCH if total_h <= 6.5 else 96,
        "NC": g["NC"],
        "V": g["V"],
        "T": g["T"],
        "NTOK": g["NTOK"],
        "cache": str(CACHE),
        "scoring": "full-sequence NLL under False/True × K orders (NOT final-token 2-class logits)",
        "elapsed_preflight_s": time.time() - t0,
        "meta": meta if isinstance(meta, dict) else {},
    }
    (OUT / "phase0_estimate.json").write_text(json.dumps(est, indent=2, default=float))
    print(json.dumps(est, indent=2, default=float))
    assert est["budget_ok_le_6p5h"], est
    print("PHASE0_OK", flush=True)
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["preflight", "build_cache", "run", "canary_only"])
    ap.add_argument("--folds", default=os.environ.get("FOLDS", "1,2,3,4"))
    a = ap.parse_args()
    torch.manual_seed(SEED)
    if a.mode == "preflight":
        preflight_cpu()
        return
    if a.mode == "build_cache":
        print(json.dumps(build_cache(), indent=2))
        return
    assert torch.cuda.is_available(), "CUDA required for run mode"
    g = load_data()
    log(
        f"start: epochs {EPOCHS} batch {BATCH} lr {LR} K {K_OOF}/{K_TEST} seed {SEED} "
        f"T {g['T']} V {g['V']} NTOK {g['NTOK']} gpu {torch.cuda.get_device_name(0)}"
    )
    ok = run_folds(g, [x for x in a.folds.split(",") if x != ""])
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
