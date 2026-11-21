# GM V19.13 — Ultra-clean 999k peer → **HOLD**

**0 GPU pods · elapsed 359s · script `scripts/gm_v19_13_label_free_999k_peer_reconstruction_and_slot9.py`**

| | |
|---|---|
| Slot8 ATH | **0.95395** (hold) |
| Slot8 rebuild `1-ρ` | **0** (exact) |
| Submit | **False** — 0 winners |
| Daily slots used | **0** (2 remain) |
| `corr(e_va_noise, M4)` | **−0.000000** (Gram-Schmidt ✓) |
| X_raw / X_symD | 57 / 36 cols |

---

## Flaw locks verified
1. Leave-self-out peer means only (`peer = (Σ−self)/(n−1+m0)`).
2. Raw bank keys = pure numerical `k1..k6` (amt/bal/dist cents & rounds) — **no** mrisk/vel/flag/merchant in keys.
3. Attributes `A1..A6` never average amt/bal/dist; freq regress-out `[log1p(cnt), shrink]` on every column.
4. `e_va_noise = unitize(0.5(M1+M2) − β·M4)` with `corr(e_va, M4) ≡ 0` (not the old `0.5(M1+M2)−M4` that was anti-aligned with M4).
5. `D_f` = fast q20 LGB (`n_estimators=100`, `max_bin=63`) + L7 coarse as `A7` only.

## M4 / e_va_noise diagnostic
| Metric | Value |
|--------|-------|
| `corr(0.5(M1+M2), M4)` | +0.155 |
| `corr(e_va_noise, M4)` | **0.000000** |
| M4_clean F2_Bo12 | 0.042 |
| M4_clean **c7** | **+0.491** (Slot7 fail ray still inside M4) |
| M4_clean folds @0.012 | [+0.61, +0.77, +3.02, +2.54, +3.94] m=+2.18 5/5 |

**Conclusion:** Gram-Schmidt strips *va-pool noise* from the M1/M2 mix, but **M4’s geometry itself is Slot7-aligned**. Orth vs `e_va_noise` cannot depoison M4’s `c7≈+0.49`.

---

## Family scorecard vs Slot8

| Family | F2_Bo9 | F2_Bo12 | disp | KS | c7 | corr(δ,e_va) | Best H2H | Gate fails |
|--------|--------|---------|------|-----|-----|--------------|----------|------------|
| **P1_raw_only** | **0.008** ✓ | **0.023** ✓ | 0.995 | 0.020 | **−0.011** ✓ | −0.025 ✓ | 4/5 m≈+1.0 min<0 | Gate3/4 (not 5/5) |
| P2_raw_L7 | 0.414 ❌ | 0.414 ❌ | 1.05 | 0.004 | +0.071 | +0.033 | **5/5×3 +8…+15e-5** | Gate2 F2 (L7→Bo12) |
| P3a_orth_M5 | 0.013 ✓ | 0.025 ✓ | 0.995 | 0.033 ❌ | +0.004 | −0.086 ❌ | ≤2/5 | KS + eva + H2H |
| P3b_orth_va | 0.014 ✓ | 0.027 ✓ | 0.994 | 0.034 ❌ | +0.067 | −0.029 | ≤3/5 | KS + H2H |
| M4_clean | 0.033 ✓ | 0.042 ❌ | 0.986 | 0.010 | **+0.491** ❌ | +0.038 | 5/5 but fail-ray | Gate6 |
| P4_P1_P3b | 0.010 ✓ | 0.031 ✓ | 1.00 | 0.008 | +0.038 | −0.037 | 4/5 | Gate3 |
| P4_P1_P2 | 0.271 ❌ | 0.271 ❌ | 1.00 | 0.005 | +0.039 | +0.005 | 5/5×3 huge | Gate2 |
| P4_P1_M4c | 0.023 ✓ | 0.023 ✓ | 1.00 | 0.005 | **+0.398** ❌ | +0.018 | 5/5×3 | Gate6 |

### Best clean candidate: `P1_raw_only`
- Passes F2, anti-fail-ray, scale, `corr(δ,e_va)≤0.05`.
- **Fails** uniform 5/5×3: always one soft fold (`min≈−1…−4e-5`), `t≪2.8`, and at `w` with `1−ρ∈[0.30,1.10]` twin is fragile.
- Mean OOF lift ~+1e-5 is real but **not gate-clear**.

### P2 / P4 with L7
Huge OOF (same class as historical P4) but **`F2_Bo12≈0.41`** — L7-coarse peer means re-enter Bo12 (L7F pillar). Forbidden by Gate 2.

---

## 6-gate summary
| Gate | P1 | P2 | M4_clean |
|------|----|----|----------|
| 1 Scale | ✓ | ✓ | ✓ |
| 2 F2 | ✓ | ✗ | ✗ (0.042) |
| 3 5/5×3 | ✗ 4/5 | ✓ | ✓ |
| 4 sf/t | ✗ | ✓ | soft t |
| 5 1−ρ band | conflict w/ H2H | ok at mid w | mid w small ρ |
| 6 anti-fail / eva | ✓ | soft | ✗ c7=+0.49 |

**0 candidates cleared all 6.**

---

## Diagnosis
1. Ultra-clean numerical peers (**P1**) achieve the F2 + anti-fail-ray contract but lack fold-uniform AUC vs saturated Slot8.
2. Any reintroduction of L7/M4 peer geometry restores OOF lift **and** either F2 leak or Slot7 fail-ray (`c7≈+0.4…+0.5`).
3. `e_va_noise ⊥ M4` is correctly constructed; it does **not** equal “M4 without Slot7 poison.”

## Decision
**HOLD** ATH `0.95395`. Keep remaining daily slots.

Artifacts: `/tmp/s6e11_v16/out/v19_13.json`, `v19_13_families.npz`, `logs/overnight/GM_V19_13_ULTRA_CLEAN.log`.
