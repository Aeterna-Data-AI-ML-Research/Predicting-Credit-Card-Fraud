#!/usr/bin/env python3
"""Module A — Qwen2.5-0.5B digit GenBayes with exact (N,24) per-field LLR.

RESUMABLE: atomic checkpoints under out/ckpts/f{fold}/ every CKPT_EVERY steps +
each epoch end + each score (y,order) shard. Re-running the same --out resumes
from the latest state.json. NO_SUBMIT. TabICL never called.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from serializer_24field import (
    FEATS,
    FieldSerializer,
    accumulate_field_llr,
    antithetic_orders,
    raw_cell,
)

NO_SUBMIT = True
assert NO_SUBMIT
T0 = time.time()
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
MODEL_ID = os.environ.get("V43_MODEL", "Qwen/Qwen2.5-0.5B")
MAX_LEN = int(os.environ.get("V43_MAX_LEN", "384"))
EPOCHS = int(os.environ.get("V43_EPOCHS", "4"))
K_PAIRS = int(os.environ.get("V43_K_PAIRS", "2"))  # → 4 orders
TRAIN_BS = int(os.environ.get("V43_TRAIN_BS", "8"))
SCORE_BS = int(os.environ.get("V43_SCORE_BS", "32"))
CKPT_EVERY = int(os.environ.get("V43_CKPT_EVERY", "200"))  # train steps
SCORE_FLUSH = int(os.environ.get("V43_SCORE_FLUSH", "64"))  # batches between score mmap flush
LR = float(os.environ.get("V43_LR", "1e-4"))
LORA_R = int(os.environ.get("V43_LORA_R", "64"))
USE_LORA = os.environ.get("V43_USE_LORA", "1") == "1"
USE_COMPILE = os.environ.get("V43_USE_COMPILE", "0") == "1"
FOLD = int(os.environ.get("V43_FOLD", "0"))
SEED = 42


def log(m: str) -> None:
    print(f"[{time.time()-T0:7.1f}s][A] {m}", flush=True)


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def atomic_np_save(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_base = path.parent / (path.stem + ".tmp")
    np.save(tmp_base, arr)  # writes stem.tmp.npy
    os.replace(Path(str(tmp_base) + ".npy"), path)


def load_frames(data_dir: Path):
    tr = pd.read_csv(data_dir / "train.csv")
    te = pd.read_csv(data_dir / "test.csv")
    y = tr["is_fraud"].to_numpy(int)
    return tr, te, y


def pretok_rows(df: pd.DataFrame, ser: FieldSerializer | None = None, cache_path: Path | None = None) -> list[list[str]]:
    """Fast column-wise pretok; optional pickle cache for resume."""
    import pickle

    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            raws = pickle.load(f)
        log(f"pretok cache hit n={len(raws)} ← {cache_path}")
        return raws
    n = len(df)
    log(f"pretok building n={n} …")
    cols = {}
    for c in FEATS:
        s = df[c]
        cols[c] = [raw_cell(c, v) for v in s.to_numpy()]
        log(f"  pretok col {c} done")
    raws = [[cols[c][i] for c in FEATS] for i in range(n)]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.parent / f"{cache_path.name}.{os.getpid()}.tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(raws, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
            log(f"pretok cached → {cache_path}")
        except Exception as e:
            log(f"pretok cache write skipped: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
    return raws


def build_model(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    if USE_LORA:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_R * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, cfg)
        log(f"LoRA r={LORA_R} on all linear projections")
    if USE_COMPILE:
        raise SystemExit("V43_USE_COMPILE must stay 0")
    model.to(device)
    return model, tok


def token_ll_cross_entropy(logits, input_ids, attn, pad_id):
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:].clone()
    shift_attn = attn[:, 1:]
    shift_labels = shift_labels.masked_fill((shift_attn == 0) | (shift_labels == pad_id), -100)
    B, Tm1, V = shift_logits.shape
    nll = F.cross_entropy(
        shift_logits.reshape(-1, V).float(),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(B, Tm1)
    tok_ll = (-nll).float()
    return tok_ll * (shift_labels != -100).to(tok_ll.dtype)


def ckpt_dir(out: Path, fold: int) -> Path:
    d = out / "ckpts" / f"f{fold}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_train_ckpt(ckd: Path, model, opt, sched, ep: int, step: int, global_step: int, rng_state, perm) -> None:
    import torch

    payload = {
        "epoch": ep,
        "step_in_epoch": step,
        "global_step": global_step,
        "rng_state": rng_state,
        "perm": np.asarray(perm),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": {"EPOCHS": EPOCHS, "K_PAIRS": K_PAIRS, "SEED": SEED, "MODEL_ID": MODEL_ID},
    }
    tmp = ckd / "train_ckpt.tmp.pt"
    final = ckd / "train_ckpt.pt"
    torch.save(payload, tmp)
    os.replace(tmp, final)
    atomic_write_json(
        ckd / "state.json",
        {
            "phase": "train",
            "epoch": ep,
            "step_in_epoch": step,
            "global_step": global_step,
            "ts": time.time(),
            "NO_SUBMIT": True,
        },
    )
    log(f"CKPT saved ep={ep} step={step} global={global_step} → {final}")


def load_train_ckpt(ckd: Path, model, opt, sched, device):
    import torch

    p = ckd / "train_ckpt.pt"
    if not p.exists():
        return None
    blob = torch.load(p, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"], strict=True)
    opt.load_state_dict(blob["opt"])
    sched.load_state_dict(blob["sched"])
    # move opt state tensors to device
    for st in opt.state.values():
        for k, v in list(st.items()):
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)
    log(f"CKPT resume ep={blob['epoch']} step={blob['step_in_epoch']} global={blob['global_step']}")
    return blob


def train_fold(model, tok, ser, tr_df, y_tr, device, ckd: Path):
    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    model.train()
    n = len(tr_df)
    raws = pretok_rows(tr_df, ser, cache_path=ckd / "pretok_train.pkl")
    opt = AdamW((p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=0.01)
    steps_per = max(n // TRAIN_BS, 1)
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(steps_per // 10, 1), num_training_steps=steps_per * EPOCHS
    )
    rng = np.random.default_rng(SEED)
    orders = antithetic_orders(24, K_PAIRS, seed=39)

    start_ep, start_step, global_step = 0, 0, 0
    resume_perm = None
    blob = load_train_ckpt(ckd, model, opt, sched, device)
    if blob is not None:
        start_ep = int(blob["epoch"])
        start_step = int(blob["step_in_epoch"])
        global_step = int(blob["global_step"])
        if blob.get("rng_state") is not None:
            rng.bit_generator.state = blob["rng_state"]
        if blob.get("perm") is not None:
            resume_perm = np.asarray(blob["perm"])

    state = json.loads((ckd / "state.json").read_text()) if (ckd / "state.json").exists() else {}
    if state.get("phase") in ("score", "done"):
        log("train already finished per state.json — skip train")
        return model

    for ep in range(start_ep, EPOCHS):
        if ep == start_ep and resume_perm is not None and start_step > 0:
            perm = resume_perm
        else:
            perm = rng.permutation(n)
        step0 = start_step if ep == start_ep else 0
        losses = []
        n_steps = (n + TRAIN_BS - 1) // TRAIN_BS
        for si in range(step0, n_steps):
            s0 = si * TRAIN_BS
            batch_ix = perm[s0 : s0 + TRAIN_BS]
            seqs = []
            for i in batch_ix:
                o = orders[int(rng.integers(0, len(orders)))]
                ids, _, _ = ser.assemble(raws[i], o, int(y_tr[i]))
                seqs.append(ids[:MAX_LEN])
            max_l = max(len(s) for s in seqs)
            pad_id = tok.pad_token_id
            input_ids = torch.full((len(seqs), max_l), pad_id, dtype=torch.long, device=device)
            attn = torch.zeros((len(seqs), max_l), dtype=torch.long, device=device)
            for bi, ids in enumerate(seqs):
                L_ = len(ids)
                input_ids[bi, :L_] = torch.tensor(ids, dtype=torch.long, device=device)
                attn[bi, :L_] = 1
            out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
            loss = out.loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            global_step += 1
            if si % 200 == 0:
                log(f"ep{ep} step{si}/{n_steps} loss={np.mean(losses[-50:]):.4f}")
            next_step = si + 1
            if global_step % CKPT_EVERY == 0 or next_step >= n_steps:
                save_train_ckpt(
                    ckd, model, opt, sched, ep, next_step, global_step, rng.bit_generator.state, perm
                )
        log(f"epoch {ep} mean_loss={float(np.mean(losses)) if losses else float('nan'):.4f}")
        start_step = 0
        resume_perm = None
        save_train_ckpt(ckd, model, opt, sched, ep + 1, 0, global_step, rng.bit_generator.state, perm)

    atomic_write_json(
        ckd / "state.json",
        {"phase": "score", "epoch": EPOCHS, "global_step": global_step, "ts": time.time(), "NO_SUBMIT": True},
    )
    # freeze weights for score resume
    import torch

    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        ckd / "model_final.pt",
    )
    log("train complete → phase=score")
    return model


def _score_arrays(ckd: Path, tag: str, n: int, K: int):
    """Memory-map / load partial score tensors; create if missing."""
    paths = {
        "seq": ckd / f"partial_{tag}_seq.npy",
        "f0": ckd / f"partial_{tag}_f0.npy",
        "f1": ckd / f"partial_{tag}_f1.npy",
        "prog": ckd / f"partial_{tag}_progress.json",
    }
    if paths["seq"].exists():
        seq = np.load(paths["seq"], mmap_mode="r+")
        f0 = np.load(paths["f0"], mmap_mode="r+")
        f1 = np.load(paths["f1"], mmap_mode="r+")
        prog = json.loads(paths["prog"].read_text()) if paths["prog"].exists() else {"done": []}
    else:
        seq = np.lib.format.open_memmap(paths["seq"], mode="w+", dtype=np.float64, shape=(n, K, 2))
        f0 = np.lib.format.open_memmap(paths["f0"], mode="w+", dtype=np.float64, shape=(n, K, 24))
        f1 = np.lib.format.open_memmap(paths["f1"], mode="w+", dtype=np.float64, shape=(n, K, 24))
        seq[:] = 0
        f0[:] = 0
        f1[:] = 0
        prog = {"done": []}
        atomic_write_json(paths["prog"], prog)
    return seq, f0, f1, prog, paths


def score_rows_field_llr(model, tok, ser, raws, orders, device, ckd: Path, tag: str):
    import torch

    model.eval()
    n = len(raws)
    K = len(orders)
    seq_ll, f0, f1, prog, paths = _score_arrays(ckd, tag, n, K)
    done = set(tuple(x) for x in prog.get("done", []))
    pad_id = tok.pad_token_id
    prefix_len = len(ser.y0)
    assert prefix_len == len(ser.y1)
    body_lo = max(prefix_len - 1, 0)

    with torch.no_grad():
        for yi, y_label in enumerate((0, 1)):
            bucket = f0 if yi == 0 else f1
            for ok, order in enumerate(orders):
                key = (yi, ok)
                if key in done or [yi, ok] in prog.get("done", []):
                    log(f"score skip done y={y_label} order={ok}/{K}")
                    continue
                log(f"score y={y_label} order={ok}/{K}")
                flush_i = 0
                for s0 in range(0, n, SCORE_BS):
                    batch = list(range(s0, min(s0 + SCORE_BS, n)))
                    packed, spans_b = [], []
                    for i in batch:
                        ids, _, spans = ser.assemble(raws[i], order, y_label)
                        packed.append(ids[:MAX_LEN])
                        spans_b.append(spans)
                    max_l = max(len(s) for s in packed)
                    input_ids = torch.full((len(packed), max_l), pad_id, dtype=torch.long, device=device)
                    attn = torch.zeros((len(packed), max_l), dtype=torch.long, device=device)
                    for bi, ids in enumerate(packed):
                        L_ = len(ids)
                        input_ids[bi, :L_] = torch.tensor(ids, dtype=torch.long, device=device)
                        attn[bi, :L_] = 1
                    logits = model(input_ids=input_ids, attention_mask=attn).logits
                    tok_ll = token_ll_cross_entropy(logits, input_ids, attn, pad_id)
                    if body_lo > 0:
                        tok_ll[:, :body_lo] = 0
                    for bi, i in enumerate(batch):
                        tl = tok_ll[bi].detach().cpu().numpy()
                        seq_ll[i, ok, yi] = float(tl[body_lo:].sum())
                        bucket[i, ok] = accumulate_field_llr(tl, spans_b[bi])
                    flush_i += 1
                    if flush_i % SCORE_FLUSH == 0:
                        seq_ll.flush()
                        f0.flush()
                        f1.flush()
                        atomic_write_json(
                            ckd / "state.json",
                            {
                                "phase": "score",
                                "tag": tag,
                                "y": y_label,
                                "order": ok,
                                "row": s0 + len(batch),
                                "ts": time.time(),
                                "NO_SUBMIT": True,
                            },
                        )
                seq_ll.flush()
                f0.flush()
                f1.flush()
                prog.setdefault("done", []).append([yi, ok])
                atomic_write_json(paths["prog"], prog)
                log(f"score shard done y={y_label} order={ok} flushed")

    field_llr = np.asarray(f1 - f0).mean(1)
    field_std = np.asarray(f1 - f0).std(1)
    return np.asarray(seq_ll), field_llr, field_std


def finalize_fold(out: Path, fold: int, seq_va, f_va, s_va, seq_te, f_te, s_te):
    tag = f"f{fold}"
    atomic_np_save(out / f"oof_{tag}_seq_llr.npy", seq_va.astype(np.float32))
    atomic_np_save(out / f"test_{tag}_seq_llr.npy", seq_te.astype(np.float32))
    atomic_np_save(out / f"oof_{tag}_field24_llr.npy", f_va.astype(np.float32))
    atomic_np_save(out / f"test_{tag}_field24_llr.npy", f_te.astype(np.float32))
    atomic_np_save(out / f"oof_{tag}_field24_std.npy", s_va.astype(np.float32))
    atomic_np_save(out / f"test_{tag}_field24_std.npy", s_te.astype(np.float32))
    meta = {
        "fold": fold,
        "model": MODEL_ID,
        "K": int(seq_va.shape[1]),
        "epochs": EPOCHS,
        "lora": USE_LORA,
        "lora_r": LORA_R,
        "NO_SUBMIT": True,
        "shapes": {
            "seq_va": list(seq_va.shape),
            "seq_te": list(seq_te.shape),
            "field_va": list(f_va.shape),
            "field_te": list(f_te.shape),
        },
    }
    atomic_write_json(out / f"meta_{tag}.json", meta)


def main():
    import torch

    if os.environ.get("V43_LAUNCH", "0") != "1":
        log("DRY-RUN / harness only. Set V43_LAUNCH=1 to train+score on GPU. Exit.")
        print(json.dumps({"status": "dry_run", "NO_SUBMIT": True, "pods": 0}))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=FOLD)
    ap.add_argument("--full5", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} model={MODEL_ID} CKPT_EVERY={CKPT_EVERY}")
    tr, te, y = load_frames(args.data)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    fold_ids = args.fold if not args.full5 else list(range(5))
    if isinstance(fold_ids, int):
        fold_ids = [fold_ids]

    from transformers import AutoTokenizer

    tok_probe = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    ser = FieldSerializer(tok_probe)
    assert len(ser.y0) == len(ser.y1), (len(ser.y0), len(ser.y1))
    log(f"prefix_len y0=y1={len(ser.y0)}")
    for d in "0123456789":
        ids = tok_probe.encode(d, add_special_tokens=False)
        assert len(ids) == 1, f"digit {d!r} not single-token: {ids}"
    orders = antithetic_orders(24, K_PAIRS, seed=39)
    assert len(orders) == 2 * K_PAIRS
    log(f"orders K={len(orders)}")

    for f in fold_ids:
        ckd = ckpt_dir(args.out, f)
        tag = f"f{f}"
        done_marker = args.out / f"meta_{tag}.json"
        if done_marker.exists() and (args.out / f"oof_{tag}_field24_llr.npy").exists():
            log(f"fold {f} already finalized — skip")
            continue

        tr_ix, va_ix = folds[f]
        log(f"=== FOLD {f} train={len(tr_ix)} va={len(va_ix)} ckpt={ckd} ===")
        model, tok = build_model(device)
        ser = FieldSerializer(tok)

        st_path = ckd / "state.json"
        phase = json.loads(st_path.read_text()).get("phase") if st_path.exists() else "train"
        if phase == "done":
            log(f"fold {f} phase=done")
            continue

        if phase in ("train", None) or not (ckd / "model_final.pt").exists():
            model = train_fold(model, tok, ser, tr.iloc[tr_ix].reset_index(drop=True), y[tr_ix], device, ckd)
        else:
            blob = torch.load(ckd / "model_final.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(blob, strict=True)
            model.to(device)
            log("loaded model_final.pt for score resume")

        va_raws = pretok_rows(tr.iloc[va_ix].reset_index(drop=True), ser, cache_path=ckd / "pretok_va.pkl")
        te_raws = pretok_rows(te, ser, cache_path=ckd / "pretok_te.pkl")
        seq_va, f_va, s_va = score_rows_field_llr(model, tok, ser, va_raws, orders, device, ckd, "va")
        seq_te, f_te, s_te = score_rows_field_llr(model, tok, ser, te_raws, orders, device, ckd, "te")
        finalize_fold(args.out, f, seq_va, f_va, s_va, seq_te, f_te, s_te)
        atomic_write_json(ckd / "state.json", {"phase": "done", "ts": time.time(), "NO_SUBMIT": True})
        log(f"fold {f} DONE → {args.out}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
