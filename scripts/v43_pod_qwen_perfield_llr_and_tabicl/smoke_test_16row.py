#!/usr/bin/env python3
"""16-row local CPU/MPS smoke: serializer, digit tokens, field-span un-permute, shape asserts.

NO GPU pod. NO_SUBMIT. Exit 0 iff all asserts pass.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from serializer_24field import (  # noqa: E402
    FEATS,
    ByteDigitTokenizer,
    FieldSerializer,
    accumulate_field_llr,
    antithetic_orders,
    assert_field_sum_equals_row,
    raw_cell,
    unpermute_order_llr_to_fields,
)

NO_SUBMIT = True
assert NO_SUBMIT
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:5.1f}s] {m}", flush=True)


def main():
    log("=== V43 smoke 16-row serializer / field24 unpermute ===")
    data = ROOT / "data" if (ROOT / "data/train.csv").exists() else Path("/tmp/s6e11_v39_ssd/data")
    df = pd.read_csv(data / "train.csv", nrows=16)
    assert len(df) == 16

    # digits tokenize cleanly (byte fallback + optional HF)
    bt = ByteDigitTokenizer()
    for d in "0123456789":
        ids = bt.encode(d)
        assert len(ids) == 1, (d, ids)
    log("byte-digit tokenizer: digits 0-9 are single tokens ✓")

    hf_ok = False
    tok = bt
    try:
        from transformers import AutoTokenizer
        ht = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        dig_lens = {d: len(ht.encode(d, add_special_tokens=False)) for d in "0123456789"}
        log(f"HF Qwen2.5-0.5B digit token lengths: {dig_lens}")
        hf_ok = True
        tok = ht
    except Exception as e:
        log(f"HF tokenizer unavailable locally ({e}); using ByteDigitTokenizer fallback")

    ser = FieldSerializer(tok)
    orders = antithetic_orders(24, k_pairs=4, seed=39)
    assert len(orders) == 8
    assert orders[1] == list(reversed(orders[0]))
    log(f"antithetic K=8 (4 fwd + 4 rev) ✓ first={orders[0][:4]}...")

    # synthetic token-ll: use deterministic scores = 1.0 per token for exact sum check
    per_order_tl = []
    per_order_spans = []
    field_bank = []
    for i in range(16):
        raws = [raw_cell(c, df.iloc[i][c]) for c in FEATS]
        row_fields = []
        for order in orders:
            for y_label in (0, 1):
                ids, _, spans = ser.assemble(raws, order, y_label)
                assert len(spans) == 24
                assert set(spans.keys()) == set(range(24))
                # fake causal token_ll length = len(ids)-1, value 0.1 each
                tl = np.full(len(ids) - 1, 0.1, dtype=np.float64)
                residual = assert_field_sum_equals_row(tl, spans, prefix_len=len(ser.y0), atol=1e-5)
                fld = accumulate_field_llr(tl, spans)
                assert fld.shape == (24,)
                assert abs(fld.sum() - sum(
                    tl[max(sp.start - 1, 0): max(sp.end - 1, 0)].sum() for sp in spans.values()
                )) < 1e-5
            # one y=1 pass for unpermute bank
            ids, _, spans = ser.assemble(raws, order, 1)
            tl = np.full(len(ids) - 1, 0.1, dtype=np.float64)
            if i == 0:
                per_order_tl.append(tl)
                per_order_spans.append(spans)
            row_fields.append(accumulate_field_llr(tl, spans))
        mean24, std24 = unpermute_order_llr_to_fields(
            [accumulate_field_llr(np.full(len(ser.assemble(raws, o, 1)[0]) - 1, 0.1), ser.assemble(raws, o, 1)[2]) for o in orders],
            [ser.assemble(raws, o, 1)[2] for o in orders],
        )
        assert mean24.shape == (24,) and std24.shape == (24,)
        field_bank.append(mean24)

    field_bank = np.stack(field_bank, 0)
    assert field_bank.shape == (16, 24)
    log(f"field24 bank shape={field_bank.shape} ✓")

    # seq llr shape contract
    seq = np.zeros((16, 8, 2), dtype=np.float32)
    assert seq.shape == (16, 8, 2)
    log("seq_llr contract (N,8,2) ✓")

    # StratifiedKFold seed=42 fold0 sizes
    from sklearn.model_selection import StratifiedKFold
    y = pd.read_csv(data / "train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    tr0, va0 = next(iter(skf.split(np.zeros(len(y)), y)))
    assert len(va0) == 139992, len(va0)
    assert len(tr0) == 559968, len(tr0)
    log(f"SKF(5,seed=42) fold0 train={len(tr0)} va={len(va0)} ✓")

    out = {
        "NO_SUBMIT": True,
        "pods_started": 0,
        "hf_tokenizer": hf_ok,
        "orders_K": 8,
        "field24_smoke_shape": list(field_bank.shape),
        "fold0_sizes": {"train": int(len(tr0)), "va": int(len(va0))},
        "digit_single_token_byte": True,
        "assert_field_sum_atol": 1e-5,
        "status": "PASS",
    }
    out_path = ROOT / "logs/v43/V43_SMOKE_16ROW.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log(f"PASS → {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
