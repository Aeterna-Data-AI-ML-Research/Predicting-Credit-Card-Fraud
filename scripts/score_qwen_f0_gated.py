#!/usr/bin/env python3
"""Fast gated Fold-0 Qwen score from saved LoRA.

Stage1: K=2 (1 antithetic pair, 4 passes) → ll_va_k2.npy → early kill if solo<0.9478
Stage2: remaining pairs → K=6 → ll_va_k6.npy → full gate
Stage3 (if pass): Test with length-sorted bs=512, checkpoint per order-pair.
NO_SUBMIT=True — never kaggle-submit from this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen15b_fold0_storyteller import (  # noqa: E402
    MAX_LEN,
    MODEL_ID,
    build_chunks,
    pretok_chunks,
)

T0 = time.time()
BS = 512
K_PAIRS = 3
EARLY_SOLO_MIN = 0.9478
FULL_SOLO_MIN = 0.9485
LIFT_MIN = 0.50e-4
NO_SUBMIT = True


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def antithetic(n_parts: int, k_pairs: int, seed: int = 39):
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(k_pairs):
        o = rng.permutation(n_parts).tolist()
        orders.append(o)
        orders.append(list(reversed(o)))
    return orders


def U(v):
    v = np.asarray(v, float).ravel()
    return (rankdata(v, method="average") - 0.5) / max(len(v), 1)


def L(v):
    return logit(np.clip(U(v), 1e-6, 1 - 1e-6))


def unit(v):
    v = np.asarray(v, float).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)


def auc(y, p):
    return float(roc_auc_score(y, np.asarray(p, float).ravel()))


def reductions(ll):
    llr = ll[:, :, 1] - ll[:, :, 0]
    r_md = llr.mean(1)
    r_deotte = expit(llr).mean(1)
    r_rl = expit(L(r_md))
    return {
        "mean_diff": U(r_md),
        "logsumexp": U(r_deotte),
        "ranklogit_mean": r_rl,
    }


def subfold_lift(y, z, u, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    d = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        best_w, best_a = 0.0, -1.0
        for w in np.linspace(-0.06, 0.06, 25):
            a = auc(y[tr], L(z[tr]) + w * u[tr])
            if a > best_a:
                best_a, best_w = a, float(w)
        d.append(auc(y[va], L(z[va]) + best_w * u[va]) - auc(y[va], z[va]))
    d = np.asarray(d)
    return {"mean_d": float(d.mean()), "n_pos": int((d > 0).sum()), "d": d.tolist()}


def gate_metrics(ll, y0, z_fb0, z_tw0):
    reds = reductions(ll)
    out = {}
    best_solo = -1.0
    for name, s in reds.items():
        solo = auc(y0, s)
        best_solo = max(best_solo, solo)
        u_fb = unit(L(s) - L(z_fb0))
        u_tw = unit(L(s) - L(z_tw0))
        g_fb = subfold_lift(y0, z_fb0, u_fb)
        g_tw = subfold_lift(y0, z_tw0, u_tw)
        out[name] = {"solo_f0": solo, "g_fb": g_fb, "g_tw": g_tw}
        log(
            f"gate[{name}] solo={solo:.5f} "
            f"fbΔ={g_fb['mean_d']:+.2e}({g_fb['n_pos']}/5) "
            f"twΔ={g_tw['mean_d']:+.2e}({g_tw['n_pos']}/5)"
        )
    out["best_solo"] = best_solo
    return out


@torch.inference_mode()
def score_pass_sorted(model, tok_chunks, pref, pad_id, device, order, bs=BS, max_tokens=24576):
    """One (order,label) pass; length-sorted dynamic batching padded to batch max.

    Target bs=512 for short rows; auto-shrink so B*max_l <= max_tokens to avoid
    4090 OOM on full-vocab logits (observed OOM at fixed bs=256 previously).
    """
    n = len(tok_chunks)
    seqs = []
    for i in range(n):
        ids = list(pref)
        parts = tok_chunks[i]
        for j in order:
            ids.extend(parts[j])
        seqs.append(ids[:MAX_LEN])
    lengths = np.array([len(s) for s in seqs], dtype=np.int32)
    order_idx = np.argsort(lengths)  # short→long
    ll = np.empty(n, dtype=np.float32)

    pos = 0
    while pos < n:
        # grow batch under token budget
        batch_ix = []
        max_l = 0
        while pos < n and len(batch_ix) < bs:
            i = int(order_idx[pos])
            L_ = int(lengths[i])
            new_max = max(max_l, L_)
            if batch_ix and (len(batch_ix) + 1) * new_max > max_tokens:
                break
            batch_ix.append(i)
            max_l = new_max
            pos += 1
            if len(batch_ix) == 1 and max_l * 1 > max_tokens:
                # single long row — still must run
                break
        batch = [seqs[i] for i in batch_ix]
        B = len(batch)
        input_ids = torch.full((B, max_l), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((B, max_l), dtype=torch.long, device=device)
        for bi, ids in enumerate(batch):
            L_ = len(ids)
            input_ids[bi, :L_] = torch.tensor(ids, dtype=torch.long, device=device)
            attn[bi, :L_] = 1
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        # keep bf16 logits; float only for gather/lse numerics on slice
        shift = logits[:, :-1, :]
        tgt = input_ids[:, 1:]
        selected = shift.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float()
        # chunked logsumexp over vocab to cut peak float memory
        V = shift.shape[-1]
        chunk = 16384
        lse = None
        for c0 in range(0, V, chunk):
            part = torch.logsumexp(shift[:, :, c0 : c0 + chunk].float(), dim=-1)
            lse = part if lse is None else torch.logaddexp(lse, part)
        mask = (tgt != pad_id) & (attn[:, 1:] == 1)
        vals = ((selected - lse) * mask).sum(dim=1).float().cpu().numpy()
        for bi, i in enumerate(batch_ix):
            ll[i] = vals[bi]
        del logits, shift, selected, lse, input_ids, attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ll


@torch.inference_mode()
def score_orders(
    model, tok_chunks, prefix_ids, pad_id, device, orders, out_arr, ki0, partial_path, tag, bs=512
):
    """Score orders into out_arr[:, ki0:ki0+len(orders), :]; checkpoint after each order-pair."""
    for local_i, order in enumerate(orders):
        ki = ki0 + local_i
        for label in (0, 1):
            t1 = time.time()
            out_arr[:, ki, label] = score_pass_sorted(
                model, tok_chunks, prefix_ids[label], pad_id, device, order, bs=bs
            )
            log(
                f"{tag} order={ki} label={label} n={len(tok_chunks)} "
                f"bs={bs} {time.time()-t1:.1f}s"
            )
            np.save(partial_path, out_arr)
        if (local_i + 1) % 2 == 0:
            log(f"{tag} checkpointed pair ending ki={ki} → {partial_path}")
    return out_arr


def load_model(adapter: Path, device):
    tok = AutoTokenizer.from_pretrained(
        adapter if (adapter / "tokenizer_config.json").exists() else MODEL_ID,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.to(device).eval()
    return tok, model


def write_gate_flag(out: Path, name: str, payload: dict):
    p = out / name
    p.write_text(json.dumps(payload, indent=2))
    log(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/workspace/data")
    ap.add_argument("--folds_csv", default="/workspace/folds/folds.csv")
    ap.add_argument("--out_dir", default="/workspace/out")
    ap.add_argument("--preds_dir", default="/workspace/preds")
    ap.add_argument("--adapter", default="/workspace/out/qwen15b_lora_f0")
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--skip_test", action="store_true")
    args = ap.parse_args()
    score_bs = int(args.bs)

    device = torch.device("cuda")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter = Path(args.adapter)
    assert (adapter / "adapter_model.safetensors").exists(), adapter

    # marker for orchestrator
    write_gate_flag(out, "NO_SUBMIT.json", {"NO_SUBMIT": True})

    tr = pd.read_csv(Path(args.data_dir) / "train.csv")
    te = pd.read_csv(Path(args.data_dir) / "test.csv")
    folds = pd.read_csv(args.folds_csv)["fold"].to_numpy(int)
    y = tr["is_fraud"].to_numpy(int)
    va_idx = np.where(folds == 0)[0]
    y0 = y[va_idx]

    preds = Path(args.preds_dir)
    z_fb = np.load(preds / "z_fullbank_oof.npy").ravel()[va_idx]
    z_tw = np.load(preds / "z_ATH_twin_oof.npy").ravel()[va_idx]

    log(f"device={device} bs={score_bs} n_va={len(va_idx)} n_te={len(te)} NO_SUBMIT={NO_SUBMIT}")
    log(f"adapter={adapter}")
    log("build chunks…")
    chunks_tr = build_chunks(tr)
    n_parts = len(chunks_tr[0])
    orders_all = antithetic(n_parts, K_PAIRS, seed=39)
    assert len(orders_all) == 6, len(orders_all)
    orders_k2 = orders_all[:2]
    orders_rest = orders_all[2:]

    tok, model = load_model(adapter, device)
    pad_id = tok.pad_token_id
    prefix_ids = {
        0: tok.encode("0", add_special_tokens=False),
        1: tok.encode("1", add_special_tokens=False),
    }

    log("pretok VA…")
    tok_va = pretok_chunks([chunks_tr[i] for i in va_idx], tok)

    # ---------- Stage 1: K=2 ----------
    ll_va = np.zeros((len(va_idx), 6, 2), np.float32)
    partial = out / "ll_va_partial.npy"
    log("STAGE1 K=2 (4 passes)…")
    score_orders(
        model, tok_va, prefix_ids, pad_id, device, orders_k2, ll_va, 0, partial, "VA", bs=score_bs
    )
    k2_path = out / "ll_va_k2.npy"
    np.save(k2_path, ll_va[:, :2, :])
    np.save(out / "qwen15b_f0_ll_va_k2.npy", ll_va[:, :2, :])
    write_gate_flag(
        out,
        "STAGE1_K2_DONE.json",
        {"path": str(k2_path), "shape": list(ll_va[:, :2, :].shape), "t": time.time() - T0},
    )
    log(f"SAVED {k2_path}")

    m_k2 = gate_metrics(ll_va[:, :2, :], y0, z_fb, z_tw)
    early_kill = m_k2["best_solo"] < EARLY_SOLO_MIN
    write_gate_flag(
        out,
        "gate_k2.json",
        {"early_kill": early_kill, "EARLY_SOLO_MIN": EARLY_SOLO_MIN, "metrics": m_k2, "NO_SUBMIT": True},
    )
    if early_kill:
        write_gate_flag(out, "GATE_EARLY_KILL.json", {"reason": "k2_solo_dead", "best_solo": m_k2["best_solo"]})
        log(f"HARD EARLY KILL: best_solo={m_k2['best_solo']:.5f} < {EARLY_SOLO_MIN}")
        return 2

    # ---------- Stage 2: remaining → K=6 ----------
    log("STAGE2 K=6 remaining 2 pairs (8 passes)…")
    score_orders(
        model, tok_va, prefix_ids, pad_id, device, orders_rest, ll_va, 2, partial, "VA", bs=score_bs
    )
    k6_path = out / "ll_va_k6.npy"
    np.save(k6_path, ll_va)
    np.save(out / "qwen15b_f0_ll.npy", ll_va)
    np.save(out / "qwen15b_f0_ll_va.npy", ll_va)
    write_gate_flag(out, "STAGE2_K6_DONE.json", {"path": str(k6_path), "shape": list(ll_va.shape)})
    log(f"SAVED {k6_path}")

    m_k6 = gate_metrics(ll_va, y0, z_fb, z_tw)
    md = m_k6["mean_diff"]
    full_ok = (
        md["solo_f0"] >= FULL_SOLO_MIN
        and md["g_fb"]["n_pos"] >= 4
        and md["g_fb"]["mean_d"] >= LIFT_MIN
    )
    any_ok = False
    for name, m in m_k6.items():
        if name == "best_solo":
            continue
        if (
            m["solo_f0"] >= FULL_SOLO_MIN
            and m["g_fb"]["n_pos"] >= 4
            and m["g_fb"]["mean_d"] >= LIFT_MIN
        ):
            any_ok = True
    full_ok = full_ok or any_ok
    write_gate_flag(
        out,
        "gate_k6.json",
        {"ok": full_ok, "FULL_SOLO_MIN": FULL_SOLO_MIN, "LIFT_MIN": LIFT_MIN, "metrics": m_k6, "NO_SUBMIT": True},
    )
    if not full_ok:
        write_gate_flag(out, "GATE_FAIL.json", {"reason": "k6_full_gate", "metrics": m_k6})
        log("FULL GATE FAIL — stop before Test")
        return 1

    write_gate_flag(out, "GATE_PASS.json", {"metrics": m_k6, "NO_SUBMIT": True})
    log("FULL GATE PASS — scoring Test (NO_SUBMIT)")

    if args.skip_test:
        log("skip_test set — done")
        return 0

    # ---------- Stage 3: Test ----------
    log("pretok TEST…")
    chunks_te = build_chunks(te)
    tok_te = pretok_chunks(chunks_te, tok)
    ll_te = np.zeros((len(te), 6, 2), np.float32)
    te_partial = out / "ll_te_partial.npy"
    for pair in range(3):
        sl = orders_all[2 * pair : 2 * pair + 2]
        score_orders(
            model, tok_te, prefix_ids, pad_id, device, sl, ll_te, 2 * pair, te_partial, "TE", bs=score_bs
        )
        pair_path = out / f"ll_te_pair{pair}.npy"
        np.save(pair_path, ll_te[:, : 2 * (pair + 1), :])
        log(f"SAVED {pair_path}")
    np.save(out / "ll_te_k6.npy", ll_te)
    np.save(out / "qwen15b_f0_ll_test.npy", ll_te)
    write_gate_flag(out, "TEST_DONE.json", {"shape": list(ll_te.shape), "NO_SUBMIT": True})
    log(f"DONE elapsed={time.time()-T0:.1f}s NO_SUBMIT={NO_SUBMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
