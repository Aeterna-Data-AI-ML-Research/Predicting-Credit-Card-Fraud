# GM V19.2 — Additive / Anchored CatObliv on Slot5 → **HOLD (NO Submit)**

**0 GPU pods · ~1646s**  
Script: `scripts/gm_v19_2_additive_oblivious_cat_and_intrabin_on_slot5.py`  
Artifacts: `/tmp/s6e11_v16/out/v19_2.json`, `v19_2_families.npz`

---

## Verdict

**Do not submit.** Anchored stacking *does* unlock **5/5×3 H2H vs Slot5** (V19.1’s replacement trap is real), but **no candidate jointly clears the Spearman live-transfer band** `1−ρ ∈ [0.15, 0.50]e-4` with twin `sf≥18` & `t≥2.8`.

| Gate | Best near-miss `w_R2=0.018 + w_cat=0.008` |
|---|---|
| 5/5×3 H2H, min&gt;0.05 | ✓ (+1.51 / +1.96 / +1.47; mins +0.64 / +0.11 / +0.54) |
| sf / t vs Slot5 | ✓ 19/18/20 · 3.74 / **2.80** / 3.76 |
| `1−ρ(Slot5)` | **✗ 0.075e-4** (&lt; 0.15 floor) |
| Neg-ray Slot6/7 | ✓ +0.045 / −0.001 |
| **Submit** | **False — HOLD 0.95394** |

Denser grid (`w_orig∈[0.012,0.020]`, `w_cat∈[0.004,0.019]`): **FULL CLEAR count = 0**.

---

## Step 1 — Why additive orth-to-`Bo10` fails; why anchored works

| Metric | Value |
|---|---|
| `corr(F_cat_Bo9, R2_v178)` | **+0.707** (complementary but heavily overlapping) |
| `C_cat = orth(Cat \| Bo10_S5)` `raw_disp` / `raw_KS` | 1.000 / 0.020 |
| F2 vs Bo9 / vs Bo10 | **0.042** / **0.233** (R2 column absorbs Cat) |
| `corr(resid(C_cat, u_R2), u_R2)` | −0.006 |

**Geometry 1 (pure additive `z_S5 + w·C_cat|Bo10`):** all `w∈[0.005,0.020]` → **4/5 or worse**, twin min negative. Orthing Cat against `Bo10=[Bo9,R2]` deletes the shared +0.71 geometry that made Cat strong on Slot3.

**Geometry 2 (anchored joint on Slot3):** keep `w_R2>0` + add Cat — **does** beat Slot5 5/5×3 at small Cat weight.

### Per-fold H2H table — best 5/5×3 hit (`w_orig=0.018`, `w_cat=0.008`, `w_lgb=0`)

ΔAUC (e-5) vs Slot5 (`z_S3+0.020·R2`):

| Base | F0 | F1 | F2 | F3 | F4 | mean | min | sf | t |
|---|---|---|---|---|---|---|---|---|---|
| z_S5 | +2.42 | +0.72 | +2.22 | +0.64 | +1.55 | **+1.51** | +0.64 | 19 | 3.74 |
| twin | +2.93 | +0.11 | +3.36 | +0.74 | +2.68 | **+1.96** | +0.11 | 18 | 2.80 |
| NNLS | +2.20 | +0.58 | +2.26 | +0.54 | +1.75 | **+1.47** | +0.54 | 20 | 3.76 |

### Rho ↔ twin-t tradeoff (denser grid excerpt)

| `w_orig` | `w_cat` | `1−ρ`×1e4 | 5/5 mins | sf/t ok | in band |
|---|---|---|---|---|---|
| 0.018 | 0.007 | 0.054 | ✓ | ✓ | ✗ |
| **0.018** | **0.008** | **0.075** | **✓** | **✓** | **✗** |
| 0.018 | 0.011 | 0.161 | ✓ | ✗ twin t=2.32 | ✓ |
| 0.019 | 0.006 | 0.049 | ✓ | ✓ | ✗ |
| 0.019 | 0.010 | 0.155 | ✓ | ✗ twin t=2.19 | ✓ |
| 0.020 | 0.010 | 0.185 | ✓ | ✗ twin sf17 t=1.94 | ✓ |

**Pattern:** configs strong enough for live-band Spearman distance lose twin subfold `t`/`sf`; configs that keep twin `t≥2.8` are micro-steps (`1−ρ&lt;0.10e-4`) — same failure class as Slot7.

---

## Step 2 — New oblivious / intrabin families

| Family | target | `raw_disp` | `raw_KS` | F2(Bo9) | fam_ok | Additive on S5 |
|---|---|---|---|---|---|---|
| `F_cat_ms45` (20× depth 2–5) | `(g_S3,h_S3)` | 1.000 | **0.007** | **0.014** | ✓ | ✗ 4/5 |
| `F_cat_direct_S5` (15×) | `(g_S5,h_S5)` | 1.000 | 0.014 | 0.020 | ✓ | ✗ 4/5 |
| `F_cat_intrabin` (+18 intra-q) | `(g_S5,h_S5)` | 1.000 | 0.022 | 0.020 | ✓ | ✗ 4/5 |

Intrabin bank is hygienic (no L4.6C F2 spike), but **does not unlock additive 5/5×3 on Slot5**.

---

## Step 3 — Decision

All 6 hard gates required. Only one candidate cleared H2H 5/5×3; it failed gate 5 (`1−ρ` band). Extended search: **0** full clears.

**HOLD Slot5 ATH `0.95394`. No Kaggle submit. 3 daily slots preserved.**

---

## Implications

1. **V19.1 diagnosis confirmed:** do not delete R2; Cat and LGB leaf-wise splits are complementary (`ρ≈0.71`) but not interchangeable.
2. **Orth-to-`Bo10` additive is the wrong operator** when the new signal correlates with R2 — use anchored joint weights instead.
3. **Live-transfer band and twin-t are currently antagonistic** for Cat+R2 mixes — pushing `1−ρ` into `[0.15,0.50]e-4` systematically softens twin subfolds below `t=2.8`.
4. Next attempt needs either (a) a Slot5-orthogonal residual with `ρ(R2)≪0.5` and native `raw_disp≈1`, or (b) a calibrated joint path that improves twin-t *inside* the Spearman band — not another micro-anchored polish.

---

*End GM V19.2*
