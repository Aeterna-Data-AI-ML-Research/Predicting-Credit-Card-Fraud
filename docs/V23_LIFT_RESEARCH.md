# V23 Lift Research (2026-11-16 morning)

Personal ATH: **0.95426** (tied with Sriram / several others).  
Public board now stretches to **0.95931** (#1) with a dense band **0.95429–0.95449** above us.  
Today’s Kaggle quota: **0 submits so far** (full day available).

Local volume `/Volumes/pics/...` was offline at research time; artifacts still in `/tmp/s6e11_nb_research/` and `/tmp/s6e11_v16/out/`.

---

## 1. Board geometry (what “up” means)

| Band | Scores | Interpretation |
|------|--------|----------------|
| Extreme | 0.95568–0.95931 | Likely heavy public-slice overfitting / undisclosed stacks; **do not chase blindly** |
| Near-top | 0.95429–0.95449 | Achievable with **cross-family** blends + gated micro-steps |
| Our plateau | **0.95426** | sriram-manifold Ray-1 R2 / K2 / SLOT1 |
| Noise floor | ±1e-5 | Asterios: many “+2e-5 public” probes lose private |

Asterios Panoras (*LB 0.95422 \| This Score Is Not Real*, 22 votes) rebuilds public ladders on nested OOF and shows public-board-steered probes often **hurt private** while CV-backed 90/10 family blends hold.

goodpjw2008 now publishes a **public-score forecast** before each probe (9/12 up since v4) and ships three files: board file / **CV-best** / bold.

---

## 2. What V22 already proved (constraints)

**Keep**

- Stay on sriram rank-frac manifold; Ray-1 micro-steps (`1−ρ ≈ 0.5–1.6e-4`) moved LB.
- Sign-lock + `disp=1` before Test inject; exact-fname poll.

**Avoid / invert**

- `+U_orth_super` in twilight (`tw≈0.78`) → **0.95416 / 0.95412** (anti-aligned). Use `V22_6_TWI_SIGN=-1` if retrying.
- Post-gate `B_gs` projection (kills twilight).
- z9 self-overlap gates; Ray-1 overshoot to `2.05e-4` (tie only).
- Uniform orth `≲0.32e-4` (5th-decimal blindness / noop).

---

## 3. Ranked new lift hypotheses (for V23)

### P0 — Cross-family 90/10 blend (highest EV, <60s CPU)

**Idea (Areeb v22):** two files at ~same public score, rank-ρ≈0.98, blend `0.90·LM + 0.10·Tree`.

| Ingredient | Local path / source | Public |
|------------|---------------------|--------|
| X (LM / sriram lineage) | ATH `R2` 0.95426 or `lm_ladder_95417` | 0.95417–0.95426 |
| S (shallow / alex) | `/tmp/s6e11_nb_research/alex_9539` + newer alex 0.9539 notebook | ~0.9539–0.95417 |

**Protocol:** measure Spearman of logit-ranks; if `0.975 < ρ < 0.995`, grid `w∈{0.05…0.20}` of S into X; require nested H2H vs R2 `5/5` and `1−ρ(R2)∈[0.22e-4,0.42e-4]`.  
**Expected:** +1e-5…+3e-5 public if families are truly different (Areeb got 0.95419→0.95421).

### P1 — Ray-1 crest interior only (no orth)

R2=`1.25e-4`, K2=`2.05e-4` tied. Probe pure Ray-1 at **`1.55e-4` and `1.75e-4`** from sriram (between R2 and K2), **zero** orth residual.  
V22.5 apex story still unused in the *interior*.

### P2 — Flipped twilight orth (`−U_orth`)

V22.6 showed `+orth` twilight hurts. One gated probe: `inject_twilight_rho(..., sign=-1)` @ `0.35–0.40e-4` from R2. Hard stop if OOF ridge not `5/5`.

### P3 — Dual-file strategy (board vs private)

Copy goodpjw: every day ship  
1. **Board file** — small public-forecast step from R2  
2. **CV file** — max nested AUC near R2 (for endgame / private)  
Never promote board-only steps that fail nested `pos≥4` / meanΔ≥0.

### P4 — New digit-shallow family residual (medium, ~30–90 min)

kiza EXP031 / historical digit TE: 1st/2nd decimal digits of risk/velocity/hours as shallow-tree family.  
Build LOFO OOF+test, GS-orth vs `B_macro`, screen vs `z_R2_sri_ridge` (`5/5`, `t≥3`). If pass, inject at `0.30e-4` — **new signal**, not rehash of V1–V14 that failed nnls.

### P5 — Fresh public members into the bank (recon only first)

Pull / hash-check:

- `goodpjw2008/s6e11-target-first-gpt2-seed-2026` (updated Nov 14)
- `alexchilton/s6e11-oof-predictions` (15 models, Nov 15)
- `jamaluddin99/s6e11-fraud-rank-blends` (0.9538x — baseline only)

Replay onto OOF; only keep members with nested lift vs R2 surrogate.

### P6 — Do **not** prioritize (low EV / high risk)

| Idea | Why skip |
|------|----------|
| Chase 0.955–0.959 without recipe | Opaque; likely public-slice fit |
| Re-run V22.6 SLOT2–5 recipes | Live-scored regressions |
| More `+orth` twilight | Proven −10e-5…−14e-5 |
| Larger Ray-1 past 2.1e-4 | K2 already tied; overshoot |

---

## 4. Suggested day plan (5–8 slots)

| Slot | Probe | Gate |
|------|-------|------|
| 1 | P0 blend `0.90 R2 + 0.10 alex/tree` | nested 5/5; ρ(R2)∈[0.22,0.42]e-4 |
| 2 | P1 Ray-1 @ `1.55e-4` sri | vs R2 OOF ≥0; hold if LB≤0.95425 |
| 3 | Adaptive: if S1≥0.95427 continue family; elif S1 tie try P2 −twilight | — |
| 4 | P1 alt `1.75e-4` **or** P3 CV-max file | — |
| 5+ | Hold best; optional P4 if family built | — |

Always keep **`sub_v22_3_LIVE_ridge_continuation.csv`** as private lock until a CV file beats it nested **and** public ≥0.95425.

---

## 5. Local inputs already on disk

```
/tmp/s6e11_nb_research/sriram_95422/
/tmp/s6e11_nb_research/lm_ladder_95417/
/tmp/s6e11_nb_research/alex_9539/
/tmp/s6e11_nb_research/sriram_ds/
/tmp/s6e11_v16/out/gm_overnight_vault.npz
```

Scripts: `scripts/gm_v22_6_unattended_5slot_number1_ladder.py` (`V22_6_TWI_SIGN=-1`), V22.3/V22.5 builders.

---

## 6. Bottom line

The next **+1e-5…+3e-5** is more likely from **cross-family logit blend** and/or **Ray-1 crest interior** than from another orth residual on the same sriram manifold. Treat scores >0.955 as a separate game until a reproducible public recipe appears.
