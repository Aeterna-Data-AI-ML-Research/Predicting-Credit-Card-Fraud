# V23.1 Lift Research (2026-11-16 after crest / −twilight)

Personal ATH: **0.95426** (R2 / K2 / SLOT1 / Ray-1 1.55 / Ray-1 1.75 — all tied).  
Quota after morning probes: **~5 submits left today**.

---

## 1. What today’s live probes closed

| Probe | Public | Verdict |
|-------|-------:|---------|
| P0 sri+0.05 alex | 0.95423 | −3e-5 |
| P0b R2+0.10 alex | 0.95425 | −1e-5 |
| Pure Ray-1 @ 1.55e-4 sri | **0.95426** | flat crest |
| Pure Ray-1 @ 1.75e-4 sri | **0.95426** | flat crest |
| −U_orth twilight @ 0.35e-4 | 0.95417 | −9e-5 (same miss class as +twilight) |

**Closed:** Ray-1 interior, both twilight orth signs, weak alex cross-family blends against our ATH.  
**Still open:** new *bank-recipe* axes, anti-orth uniform (non-twilight), true peer files at 0.95417–0.95421.

---

## 2. Public GM playbook (fresh pulls)

| Source | LB | Usable idea |
|--------|----|-------------|
| goodpjw `…public-and-cv-up…` (today) | 0.95421 | Forecast-led steps + **dual board/CV files**; step9 weights push **+megayak_shallow / +ebm_*_bag / +sameer / +glm_ohk**, **−gb_bag / −alex_glm / −ebm_dig** |
| Areeb `two-families-one-blend` | 0.95421 | 90/10 of two **near-equal** files (ρ≈0.983): ladder 0.95419 × shallow stack 0.95417 |
| Asterios `score is not real` | 0.95422 | Public-slice ladders often lose private — keep nested gate |
| alexchilton 15-model OOF set | — | Already partly in `v20_1_families` (`alex_15d0d127`, `alex_21b4d131`); full set still worth a residual screen |

Our ATH is **already above** goodpjw/Areeb public files. Copying their *blend of 0.95419×0.95417* is still EV if we can obtain those CSVs; blending our R2 with raw Q17 shallow OOF at 3–10% **destroys** geometry (ρ(R2)≫1e-3) and nested 0/5.

---

## 3. Local quantitative screen (`v23_1_research_screen.json`)

Compressed goodpjw step9 direction from **11 local bank/Q17 members** (shallow, sameer, gpdvae, alex_glm, ebm_all/dig bags, gb_bag5, gbn_w512, glm_ohk_c01, glm_dl_s50_bag), mapped against an R2 OOF surrogate (AUC≈0.9510).

### Nested H2H leaders (must-have for board file)

| Tag | Nested | ρ(R2) e-4 | Note |
|-----|--------|----------:|------|
| `SINGLE_sameer +` @0.28–0.40 | **5/5 t=11.7** | 0.28–0.40 | Strong OOF; sameer already inside prior U_orth4 (public noop when bundled) |
| `GPW_comp_ORTH` @0.22–0.45 | **5/5 t=9.0** | 0.22–0.45 | **New axis** (ebm bags + glm_ohk − gb); raw (non-orth) GPW fails 2/5 |
| `SINGLE_glm_ohk_c01 +` | 5/5 t=4.2 | 0.28–0.40 | Secondary |
| `MID_dmlp` | 3/5 − | — | Kill |
| Raw `GPW_comp` (no orth) | 2/5 − | — | Kill |
| `BLEND_R2_shallow` 3–10% | 0/5 | 5–60 | Kill (wrong peer distance) |

### Geometry-only (no nested yet)

| Tag | ρ(R2) e-4 | Idea |
|-----|----------:|------|
| `AREEB R2+0.5·(R2−K2)` | 0.11 | Soft undo of K2 orth |
| `AREEB R2+1.0·(R2−K2)` | 0.45 | Full reflect away from K2 |
| `AREEB R2+1.0·(R2−P2)` | 0.40 | Extra Ray-1 (crest already flat) |

Uniform **+orth** was a public noop; twilight **±orth** hurt. **Uniform −orth / anti-K2** has **not** been live-tested.

---

## 4. Ranked next probes (GM discipline)

### Board file (public lift)

1. **P0 — `GPW_comp_ORTH` @ ~0.30e-4 from R2**  
   New recipe orth (not V1–V14 mix). Nested 5/5 t≈9. Expect: tie / +1e-5; stop if ≤0.95425.

2. **P1 — Anti-K2 Areeb `R2 + 1.0·(R2−K2)`** (~0.45e-4)  
   First uniform anti-orth probe. Cheap; orthogonal hypothesis to P0.

3. **P2 — (only if P0 ties)** `SINGLE_sameer` @ 0.28e-4 **or** `GPW_comp_ORTH` @ 0.45e-4  
   Avoid if P0 already regresses.

4. **P3 — Acquire true Areeb peers** (SHA `29e5ab0f…` = 0.95419 ladder, `36686835…` = 0.95417 stack) → 90/10 with **R2 as X** if `0.975<ρ<0.995`.

### Private / dual-file (do not burn board quota lightly)

5. **CV-max file:** larger `GPW_comp_ORTH` or nested-max bank step; ship as `*_PRIVATE_LOCK_NOSUBMIT.csv` until endgame.

### Do not prioritize

| Idea | Why |
|------|-----|
| More Ray-1 in [1.45, 2.05]e-4 | Live-flat |
| ± twilight orth | Live −9…−14e-5 |
| R2×alex 5–10% | Live −1…−3e-5 |
| Raw Q17 shallow 3–10% blend | Nested 0/5, ρ explosion |
| Mid-band Δmlp | Nested 3/5 |
| Chase 0.955–0.959 without recipe | Asterios / public-slice risk |

---

## 5. Constraints that still bind

- Stay near sriram manifold: prefer `1−ρ(R2) ∈ [0.22, 0.45]e-4` for board probes.
- Sign-lock + nested `pos≥4` (prefer 5/5) before submit.
- Exact-fname Kaggle poll.
- Never GS-project after twilight gating (if twilight ever retried — currently **retired**).
- Keep `sub_v22_3_LIVE_ridge_continuation.csv` as private lock until a CV file beats it nested **and** public ≥0.95425.

---

## 6. Live follow-through (same morning)

| Probe | Public | vs ATH |
|-------|-------:|--------|
| `GPW_comp_ORTH` @0.30e-4 | **0.95425** | −1e-5 |
| `AREEB R2+1.0·(R2−K2)` | **0.95424** | −2e-5 |

Nested 5/5 on GPW_ORTH **did not transfer**. Uniform anti-K2 also soft-regressed. Update: bank-recipe orth at ≤0.45e-4 is in the same **5th-decimal / slight-negative** band as SLOT1/K2 packaging — not a board path.

**Executed live (quota→0):**

| Probe | Public |
|-------|-------:|
| R2 × gpw21 90/10 | **0.95426** (tie) |
| alex-15 `4188470e` ORTH @0.30e-4 | 0.95423 |
| SAMEER ORTH @0.28e-4 | 0.95425 |

**Staged for next daily reset:** `gpw21×S95417` Areeb recipe, R2×S95417 95/5, soft anti-K2, GPW@0.45, CV-max lock.

---

## 7. Bottom line

The sriram Ray-1 crest is **exhausted**, and **new bank-weighted orth + anti-K2** just failed live (−1…−2e-5). Remaining lift is unlikely from another micro-orth on R2. Next board EV is **true near-peer cross-family blend** (matched public files) or a **genuinely new base** (fresh alex/GPT-2 member), not another GS residual.
