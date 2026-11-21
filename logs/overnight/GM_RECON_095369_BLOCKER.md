# GM Recon — Public `0.95369` Blocker (`NO_SUBMIT=True`)

Generated after local Kaggle catalog pull into `external/nov12_095369_recon/` (~103s).  
**Hard rule honored:** A40 training pods were **not** touched (`rsoju37zhpzaae` / live `smugcqcy2oqzeg`).

---

## Executive verdict

The public leap past `0.95347` is **not** a mystery new GBDT family. It is a published **LM-ladder upgrade** whose key ingredient is:

1. **`lm7_test_5fold.npy` / `lm7_oof_5fold.npy`** — goodpjw2008’s compact LM seed-7 now ships **all 5 folds** (dataset `goodpjw2008/s6e11-compact-lm-seed7` was updated; Fold-0-only tensors kept for compatibility).
2. **Two GPT-2 seeds** — move weight from masayakawamata’s public GPT-2 (`MS`) toward goodpjw’s seed-2026 GPT-2 (`G26`) at `0.06`.
3. **GP+DVAE orthogonal step** (daoviet) at `0.05`, plus half tree de-bias (`−0.05 MLP −0.025 GB`).

Documented public ceiling in pulled notebooks today: **`0.95354`** (goodpjw / areebb rebuild). Ablation tables also cite **`0.95368`** (daoviet “journey” notebook) as a claimed Public LB for a GP/DVAE apex stage — treat as author-reported, not a SHA-locked submission we verified.

**So: the 0.9536x wall people are clustering on still rests on 5-fold L7 (`L7F`), not 1-fold `lm7_test_f0.npy`.**  
Our ongoing A40 `L7_5F_K10` remains the right structural play if we need an *independent* 5-fold L7 (rotated-10) rather than cloning goodpjw’s published tensor.

---

## Exact ladder mathematics (public `0.95354` recipe)

From `areebbb/s6e11-three-ladders-two-seeds-lb-0-95354` (byte-rebuild of goodpjw):

```text
# 0.95352 intermediate (5-fold L7 already inside):
s352 = A + 0.6*(A - F)
     + 0.09*(LM  - F)     # megalm_test_sha3168  (frozen megayak)
     + 0.09*(L7F - F)     # lm7_test_5fold.npy   ★ 5-FOLD L7
     + 0.05*(GPD - F)     # daoviet GP+DVAE test_ours.npy
     - 0.05*(MLP - F)     # oof-bank MLP bag
     - 0.025*(GB - F)     # oof-bank GB bag

# 0.95354 two-seed rung:
score = logit_rank(s352) + 0.06*(G26 - MS)
# G26 = gpt2_s2026_test_f0.npy   (goodpjw seed-2026)
# MS  = masayakawamata public GPT-2 test.npy
```

All inputs are logit-ranks: `L(v) = logit((rankdata(v)-0.5)/N)`.

### Ladder history (as published)

| Step | Change | Public LB |
|---|---|---:|
| Strike 9 (Moran Z) | anchor `F` | 0.95330 |
| Strike 10 (amanatar) | apex `A` | 0.95340 |
| LM Ladder Part 1 | `+0.6(A−F)+…+0.06(LM7_f0−F)` | 0.95346 |
| areebb v17 | two-step residual | 0.95347 |
| **LM7 on 5 folds** | **`L7F = lm7_test_5fold`** | **0.95348** |
| both LMs @ 0.09 + GPD 5% | `s352` | 0.95352 |
| **+0.06 (G26−MS)** | two GPT-2 seeds | **0.95354** |

---

## Dataset smoking gun: `goodpjw2008/s6e11-compact-lm-seed7` (updated)

README now states **“All five folds (new in this version)”**:

| file | shape (verified load) | role |
|---|---|---|
| **`lm7_oof_5fold.npy`** | `(699960,)` float64 | 5-fold OOF log-odds; OOF AUC **0.949448** |
| **`lm7_test_5fold.npy`** | `(299984,)` float64 | 5-fold avg test log-odds (**4 orders/model**) |
| `lm7_oof_f0.npy` | `(139992, 10, 2)` float32 | Fold-0 only (legacy) |
| `lm7_test_f0.npy` | `(299984, 4, 2)` float32 | Fold-0 only (legacy) |
| `megalm_test_sha3168.npy` | `(299984,)` float32 | frozen megayak 2026-11-12 |
| `megalm_oof_sha181c.npy` | `(699960,)` float32 | frozen megayak OOF |
| `megalm_test_v_nov08.npy` | `(299984,)` float32 | frozen 11-08 copy |

**Answer to Q3:** The recipes that moved the pack past `0.95347` **no longer rely only on 1-fold `lm7_test_f0.npy`**. They switched to **`lm7_test_5fold.npy` (`L7F`)**. Fold-0 tensors remain in the dataset for older notebooks.

---

## Notable kernels pulled (25) — score / L7 notes

| Kernel | Peak score cited | L7 dependency |
|---|---:|---|
| `goodpjw2008/s6e11-lm-ladder-two-gpt-2-seeds-lb-0-95354` | **0.95354** | **`lm7_test_5fold`** |
| `areebbb/s6e11-three-ladders-two-seeds-lb-0-95354` | **0.95354** | **`lm7_test_5fold`** (exact rebuild) |
| `daoviet/s6e11-on-the-journey-to-your-first-top-100` | verified **0.95351**; table cites **0.95368 / 0.95378** | reconstruct path still uses **`lm7_test_f0`** + `0.06*(L7−F)`; apex often loads prebaked `submission_095351` |
| `alexchilton/credit-card-fraud-lb-0-9530-shallow-tree-stack` | ≤0.95347 | uses `lm7_test_f0` |
| `lucifer19/the-golden-signal-consensus-blend-v22` | ≤0.95347 | blend / no new L7 5f |
| Older areebb / Strike ladders | 0.95347 | 1-fold LM7 era |

No pulled notebook title literally contained **`0.95369`**. The “everyone at 0.95369” cluster is consistent with **private micro-tweaks on the public 0.95354 two-seed + 5-fold-L7 stack** (noise band ~1e−5 on public 20%).

---

## Datasets pulled

- `goodpjw2008/s6e11-compact-lm-seed7` ★ **updated 5-fold L7**
- `amanatar/s6e11-strike10-podium-apex-095340`
- `moranzavdi/s6e11-strike9-lb-095330`
- `alexchilton/s6e11-oof-predictions` (79 files)
- `dariushafshar/s6e11-golem-oof-library` (24 files)

Also referenced by the 0.95354 recipe (not all re-downloaded here):  
`goodpjw2008/s6e11-target-first-gpt2-seed-2026`, `masayakawamata/s6e11-gpt2-target-first-oof`, `daoviet/s6e11-gp-dvae-orthogonal-apex`, `goodpjw2008/s6e11-oof-bank`.

---

## Implication for our A40 `L7_5F_K10` campaign

| | Public `L7F` (goodpjw) | Our `L7_5F` (in flight) |
|---|---|---|
| Folds | 5 | 5 (Fold0 reused + F1–4) |
| Test orders | 4 / fold model, mean | **rotated 4-of-10** covering all 10 orders 2× |
| Already on Public LB | **Yes — inside 0.95348→0.95354** | No (training) |
| Differentiation | Everyone can load the npy | Independent seed-7 retrain + rotated design + Stage-B×L7 |

**Do not waste a submit cloning `lm7_test_5fold` into S7** — that direction is already saturated publicly. Use our rotated-10 5-fold L7 as an **orthogonal residual** vs published `L7F`, then Stage-B.

---

## Artifacts

- Script: `scripts/gm_recon_095369_blocker.py` (`NO_SUBMIT=True`)
- Pulls: `external/nov12_095369_recon/`
- JSON: `external/nov12_095369_recon/recon_095369.json`
- CLI dump: `logs/overnight/recon_095369_cli.txt`
- This report: `logs/overnight/GM_RECON_095369_BLOCKER.md`
