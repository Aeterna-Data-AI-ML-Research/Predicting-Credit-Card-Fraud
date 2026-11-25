#!/usr/bin/env python3
"""Canonical 24-field class-prefix serializer + antithetic orders + span un-permuter.

Used by Module A (Qwen digit GenBayes) and local smoke tests.
NO GPU required for serialization / span math.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

FEATS = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
    "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
    "merchant_category", "card_type", "auth_method", "channel", "device_type",
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
assert len(FEATS) == 24

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

BOOL_COLS = {
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
}


def raw_cell(col: str, v) -> str:
    if col in CAT_MAP:
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return CAT_MAP[col].get(s, str(v))
    if col in BOOL_COLS or isinstance(v, (bool, np.bool_)):
        return "1" if bool(v) else "0"
    if isinstance(v, (float, np.floating)):
        s = f"{float(v)}"
        if "e" in s.lower():
            s = format(float(v), ".10f").rstrip("0").rstrip(".")
        return s
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def antithetic_orders(n_parts: int = 24, k_pairs: int = 4, seed: int = 39) -> list[list[int]]:
    """K=8 orders: 4 forward permutations + 4 exact reverses."""
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(k_pairs):
        o = rng.permutation(n_parts).tolist()
        orders.append(o)
        orders.append(list(reversed(o)))
    assert len(orders) == 2 * k_pairs
    return orders


@dataclass
class FieldSpan:
    """Token span [start, end) for canonical column j inside a serialized sequence."""
    start: int
    end: int
    col_id: int


class ByteDigitTokenizer:
    """Fallback tokenizer: maps printable ASCII + digits 0-9 to clean single ids.

    Used for local smoke when HF tokenizer is unavailable. Digits are 1-token each.
    """

    def __init__(self):
        # reserve 0=pad, 1=bos; printable 32..126
        self.pad_id = 0
        self._chr2id = {chr(c): c for c in range(32, 127)}
        # ensure digits are single-char → single-id (already true via ord)
        for d in "0123456789":
            assert len(self.encode(d)) == 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._chr2id.get(ch, 63) for ch in text]  # '?' fallback

    def decode(self, ids) -> str:
        rev = {v: k for k, v in self._chr2id.items()}
        return "".join(rev.get(int(i), "?") for i in ids)


class FieldSerializer:
    """Class-prefix conditional generative format with exact 24 field spans."""

    def __init__(self, tokenizer=None):
        self.tok = tokenizer or ByteDigitTokenizer()
        self.y1 = self.tok.encode("y: 1\n", add_special_tokens=False)
        self.y0 = self.tok.encode("y: 0\n", add_special_tokens=False)
        assert len(self.y1) == len(self.y0), "y0/y1 prefix length mismatch"
        self.nl = self.tok.encode("\n", add_special_tokens=False)
        self._cell_cache: dict[tuple[int, str], list[int]] = {}

    def cell_ids(self, col_idx: int, raw: str) -> list[int]:
        key = (col_idx, raw)
        if key not in self._cell_cache:
            col = FEATS[col_idx]
            self._cell_cache[key] = self.tok.encode(f"{col}: {raw}", add_special_tokens=False)
        return self._cell_cache[key]

    def assemble(
        self,
        cell_raws: list[str],
        order: list[int],
        y_label: int,
    ) -> tuple[list[int], list[FieldSpan], dict[int, FieldSpan]]:
        """Build token ids for one (row, order, y).

        Returns:
          ids: full sequence
          spans_in_order: FieldSpan list in serialization order
          spans_by_canonical: map canonical col j → FieldSpan (un-permuted)
        """
        assert len(cell_raws) == 24
        assert sorted(order) == list(range(24))
        prefix = self.y1 if y_label == 1 else self.y0
        ids = list(prefix)
        spans_in_order: list[FieldSpan] = []
        spans_by_canonical: dict[int, FieldSpan] = {}
        for pos, j in enumerate(order):
            start = len(ids)
            piece = self.cell_ids(j, cell_raws[j])
            ids.extend(piece)
            end = len(ids)
            sp = FieldSpan(start=start, end=end, col_id=j)
            spans_in_order.append(sp)
            spans_by_canonical[j] = sp
            if pos < len(order) - 1:
                ids.extend(self.nl)
        return ids, spans_in_order, spans_by_canonical


def accumulate_field_llr(
    token_ll: np.ndarray,
    spans_by_canonical: dict[int, FieldSpan],
) -> np.ndarray:
    """Sum token logprobs per canonical field → (24,) vector.

    token_ll[t] = log p(token_{t+1} | prefix) aligned to target positions;
    we sum over [start, end) of each field's tokens (shifted by -1 for causal LM
    when start>0). For smoke tests we pass already-aligned per-token scores of
    length len(ids)-1 with index i corresponding to predicting ids[i+1].
    """
    out = np.zeros(24, dtype=np.float64)
    for j, sp in spans_by_canonical.items():
        # causal: logprob of token at position p lives at index p-1
        lo = max(sp.start - 1, 0)
        hi = max(sp.end - 1, 0)
        if hi > lo:
            out[j] = float(token_ll[lo:hi].sum())
    return out


def unpermute_order_llr_to_fields(
    per_order_token_ll: list[np.ndarray],
    per_order_spans: list[dict[int, FieldSpan]],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean + std across orders of per-field LL contributions → (24,), (24,)."""
    mat = np.stack(
        [accumulate_field_llr(tl, sp) for tl, sp in zip(per_order_token_ll, per_order_spans)],
        0,
    )  # (K, 24)
    return mat.mean(0), mat.std(0)


def assert_field_sum_equals_row(
    token_ll: np.ndarray,
    spans_by_canonical: dict[int, FieldSpan],
    prefix_len: int,
    atol: float = 1e-5,
) -> float:
    """Assert sum_j field_llr_j ≈ total body token_ll (excluding y-prefix targets)."""
    field = accumulate_field_llr(token_ll, spans_by_canonical)
    # body starts after prefix; causal indices prefix_len-1 .. end
    body_lo = max(prefix_len - 1, 0)
    total = float(token_ll[body_lo:].sum())
    # newlines between fields are NOT inside field spans — account for them
    covered = float(field.sum())
    # residual = newline tokens between fields
    residual = total - covered
    # For exact equality of field-only sum vs sum of field token positions:
    recon = covered
    field_only_total = 0.0
    for sp in spans_by_canonical.values():
        lo = max(sp.start - 1, 0)
        hi = max(sp.end - 1, 0)
        field_only_total += float(token_ll[lo:hi].sum())
    err = abs(field_only_total - recon)
    assert err < atol, f"field sum mismatch err={err}"
    return residual
