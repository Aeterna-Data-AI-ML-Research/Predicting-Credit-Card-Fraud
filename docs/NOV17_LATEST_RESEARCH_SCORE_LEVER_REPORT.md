# S6E11 — Latest Research, Score Increases & Lever Verdicts

**Date:** 2026-11-17  
**Competition:** Predicting Credit Card Fraud (Playground S6E11)  
**Current personal ATH:** **0.95440** (`sub_v24_SRI440_pub.csv` / dual-base + Δmlp_v15)  
**Prior plateau:** 0.95426 (V22.3 R2 Ray-1)  
**Scope:** V22 → V24.3 (live + staged research). Companions: `THINGS_THAT_WORKED.md`, `DEAD_ENDS.md`, `logs/overnight/GM_V2{2,3,4}_*`.

---

## 1. Executive summary

| Metric | Value |
|--------|------:|
| ATH before V24 new levers | **0.95426** |
| ATH after V24 | **0.95440** |
| Δ public | **+0.00014** |
| What moved the board | Dual-base (goodpjw step11 + Bhuvan) + **w=0.30 Δmlp PyTabKit API15** |
| What did **not** move past 0.95426 | Ray-1 crest interior, ±twilight orth, bank micro-GS orth, weak alex/sameer blends |
| Natives (RF/HGB/LGB/KNN/MLP) | Competent diversity; **not** solo ATH; staged as small-w into ATH (quota-blocked) |

**One-line law:** Lift after the sriram Ray-1 plateau came from **adopting / reconstructing a stronger dual-base + newer neural residual**, not from another orth/twilight micro-step on R2.

---

## 2. Public score ladder (verified)

### 2.1 Path to current ATH

| Era | Submit / tag | Public | Δ vs prior ATH | Verdict |
|-----|--------------|-------:|---------------:|---------|
| Slot9 era | Slot9 lineage | 0.95399 | — | Base climb |
| V22.1 C1 | lineage + dMLP | 0.95407 | +8e-5 | VALID |
| V22.1 C2pure | overshot microleap | 0.95416 | +9e-5 | VALID |
| V22.2h P2 | unbundled microleap | 0.95425 | +9e-5 | VALID |
| **V22.3 R2** | Ray-1 @ 1.25e-4 sri | **0.95426** | +1e-5 | VALID (plateau) |
| V22.5 K2 | Ray-1 2.05e-4 + orth4 | 0.95426 | 0 | TIE / exhausted crest |
| V22.6 SLOT1 | GS-projected twilight | 0.95426 | 0 | NOOP |
| V23 morning | Ray-1 1.55 / 1.75 | 0.95426 | 0 | Flat crest |
| **V24 SRI440** | dual_GB + w0.30 Δmlp_v15 | **0.95440** | **+14e-5** | **VALID ATH** |
| V24 dual R2×gpw32@0.30 v15 | same recipe family | **0.95440** | +14e-5 vs R2 | VALID (ties ATH) |

### 2.2 V24 live batch (same day, vs then-ATH 0.95426)

| Tag | Public | vs 0.95426 | Class |
|-----|-------:|----------:|-------|
| `SRI440_pub` | **0.95440** | **+14e-5** | **ATH / VALID** |
| `DUAL_R2_GPW32_dmlp30` | **0.95440** | **+14e-5** | **VALID** (ties) |
| `SRI440_x_S95417_90_10` | 0.95436 | +10e-5 | Soft valid (below ATH) |
| `R2_to_SRI440_1p20e4` | 0.95434 | +8e-5 | Partial move toward ATH |
| `R2_to_SRI440_0p60e4` | 0.95432 | +6e-5 | Partial |
| `R2_dmlp_v15_w0.22` | 0.95432 | +6e-5 | Neural residual alone (partial) |
| `R2_dmlp_v15_w0.30` | 0.95432 | +6e-5 | Same class |
| `R2_to_GPW32_0p35e4` | 0.95430 | +4e-5 | Weak |
| `R2_x_S95417_95_5` | 0.95426 | 0 | Null |
| Bold `SRI440 + 0.5·(SRI440−R2)` | 0.95439 | −1e-5 vs ATH | Near-miss / invalid for lift |

---

## 3. VALID levers & techniques

### 3.1 Proven public lifts (keep / reuse)

| Lever | Technique | Evidence | Notes |
|-------|-----------|----------|-------|
| **Sriram dual-base + Δmlp_v15** | `U(avg(U(L(G)+wΔv15), U(L(B)+wΔv15)))`, w=0.30 | Public **0.95440** | G=step11, B=Bhuvan, N15=PyTabKit API15, C=PyTorch v28 |
| **Dual of near-tops + v15** | Same formula with (R2, gpw32) | Public **0.95440** | Confirms first-order: dual + v15, not only peer CSVs |
| **Sriram rank-frac manifold** | `rank_frac(L(gp)+α·(L(new)−L(old)))` | All V22 ATH steps | Stay near manifold for board probes |
| **Tiny `1−ρ` micro-steps** | P2 0.55e-4 → R2 1.25e-4 | +3e-5 then +1e-5 | Worked **until crest saturated** |
| **Sign-lock + disp=1** | Match OOF variance before Test inject | V22.5 vectors clean | Ops prerequisite |
| **Inject Δz residual, never α=1 overwrite** | Preserve LB backbone | Harvest / stack audits | Permanent rule |
| **Nested H2H + MC pub/priv gates** | pos≥4/5, t, CI, MC gap | V24.1 screen | Methodology VALID even when hyp staged |
| **Discrete TE / digit / token LLR** | After KMAT_108 | Stacking ladder Nov 5–7 | Continuous manifold FE ≈0 lift |
| **Target-first GPT-2 seeds** | Masaya / G26 | ATH seeds historically | Expensive; weights often unpublished |
| **Q17 + tiny private residual** | L(Q17)+0.03·unitize(d1_v4) | 0.95371 era | Pattern: external apex + small private Δ |

### 3.2 Factorial factors of the 0.95440 recipe (V24.1)

| Factor | Status | Finding |
|--------|--------|---------|
| **Neural version (v15 vs v5)** | First-order VALID | Mean ρ(ATH): v15 **1.42** vs v5 **4.10** (×e-4) |
| **Base dual(G,B)** | First-order VALID | dual_GB **1.80** ≪ single bases ~2.6 ≪ dual(R2,gpw32) **4.00** |
| **Weight w** | Continuous VALID knob | Exact match @0.30; 0.22/0.35/0.40 smooth (0.17–0.68 e-4) |
| **CSV roundtrip** | **NULL** | rt True ≡ False on ρ — drop from design |

### 3.3 Research-valid but not yet live-scored (staged)

| ID / artifact | Why still valid research | Blocker |
|---------------|--------------------------|---------|
| V24.1 **H1** w=0.35 dual_GB v15 | Continuous knob next to ATH | Quota |
| V24.1 **H5** dual(ATH,B)@0.30 | Bhuvan diversity into ATH | Quota |
| V24.1 **H7** ATH+ORTH(realmlp_v1)@0.30e-4 | Nested 5/5 t=6.24 gated residual | Quota; MC gap +0.37 optimistic |
| V24.2 **ATH×nativeHGB 0.08** | Cross-family small-w; HGB OOF 0.9446 | Orth gate vs ATH surrogate 0/5; try raw small-w anyway |
| V24.3 **ATH×mlp 0.05 / knn_mlp 0.08** | MLP OOF 0.9394; mild diversity | Quota; knn alone FAIL |

### 3.4 Meta-techniques (always valid process)

- Locked CV: `StratifiedKFold(5, shuffle=True, random_state=42)` / bank `folds.csv`
- Fold-0 canary before full GPU
- `NO_SUBMIT` + staged CSVs when quota=0 or gate fails
- Exact-filename Kaggle poll in multi-slot ladders
- Prefer public-transfer family history over OOF 5/5 alone (L3/L4 counterexamples)
- Dual board/CV file discipline (goodpjw / Asterios public-slice warning)

---

## 4. INVALID / exhausted / dead levers

### 4.1 Live public regressions or ties that close the family

| Lever | Public | Verdict |
|-------|-------:|---------|
| Pure Ray-1 @ 1.55e-4 / 1.75e-4 | 0.95426 | **Exhausted crest** |
| Ray-1 overshoot 2.05e-4 + orth4 (K2) | 0.95426 | Tie only |
| `+U_orth` twilight (tw≈0.78) | 0.95416 / 0.95412 | **Anti-aligned** |
| `−U_orth` twilight @ 0.35e-4 | 0.95417 | Same miss class (−9e-5) |
| Post-gate `B_gs` after twilight | 0.95426 noop | Kills localization |
| P0 sri+0.05 alex / R2+0.10 alex | 0.95423 / 0.95425 | Soft regression |
| `GPW_comp_ORTH` @0.30e-4 | 0.95425 | Nested 5/5 **did not transfer** |
| Anti-K2 `R2+1.0·(R2−K2)` | 0.95424 | Soft regression |
| alex-15 ORTH @0.30e-4 | 0.95423 | Dead for board |
| SAMEER ORTH @0.28e-4 | 0.95425 | Dead for board |
| R2 × gpw21 90/10 | 0.95426 | Tie only |
| Bold ATH away from R2 (0.5·) | 0.95439 | −1e-5 vs ATH |
| Uniform orth ≲0.32e-4 | often tie | 5th-decimal blindness |
| Chase 0.955–0.959 without recipe | — | Opaque / public-slice risk |

### 4.2 Nested / OOF dead ends (do not re-run)

| Family | Result | Law |
|--------|--------|-----|
| Native RF/HGB/LGB **orth into ATH** | 0/5 nested | Use small raw-w blend, not GS-orth at probe scale |
| Solo KNN (PCA-24, 50k, k=201) | OOF **0.9162**; gb+8% 1/5 | Too weak / correlated |
| Solo MLP / native trees as ATH | OOF 0.93–0.94 vs bank ~0.9515 | Diversity only |
| TabPFN / testbag into newton | ρ≈1 with bag5; nested Δ≈0 | No new signal |
| Un-anchored wide Stage-1 TE GLM | Overfit; poisons boosters | Never replace z_champ that way |
| Continuous FE / TabR / TabM / DAE on top of champ | Nested Δ ≤ 0 | Discrete TE law |
| Soft pseudo-label TE | Reinforces bias | Dead |
| HiRes Compact-LM short CE | Fold0 0.894 ≪ L7 | Wrong recipe |
| P_super / stageB on Q17+L2 | Public red despite OOF 5/5 | Transfer fail |
| `gb_gpt2_bag5` as newton column | Nested regresses | Collinear |

### 4.3 Ops pitfalls (invalid process)

- Uncalibrated rank-logit `base_margin` → early-stop iter=1
- α=1 overwrite of LB anchor with OOF-tuned weight
- Soft-orth optimism without double-orth / LOFO
- Dual-prefix score bs=1536 OOM on A5000
- Treating public fusion CSVs as nested-CV evidence

---

## 5. Expert DS natives bake-off (V24.2–V24.3)

Same 62-col FE (LOFO TE + digit + interactions), bank folds.

| Model | OOF AUC | vs ATH-orth | vs gb+8% | Role |
|-------|--------:|-------------|----------|------|
| RandomForest | 0.9347 | 0/5 | — | Weak native |
| LightGBM | 0.9404 | 0/5 | — | Mid native |
| HistGB | **0.9446** | 0/5 | — | Best sklearn native |
| RF/HGB/LGB blend | 0.9447 | 0/5 | — | ≈ HGB |
| KNN k=201 | 0.9162 | 0/5 | **1/5 FAIL** | Drop solo |
| MLP (256,128) | 0.9383 | — | — | Runner-up NN |
| MLP (512,) | **0.9394** | 1/5 | 3/5 weak | Best NN native |
| KNN+MLP+HGB (0.1/0.2/0.7) | **0.9450** | 0/5 | **5/5** | Diversity blend |

Public bank reference: `gb_bag5` OOF ≈ **0.9515** — natives trail ~70–350e-4 AUC.

**Verdict:** Natives increase score **only** as small cross-family mass into ATH (priority: HGB 0.08, MLP 0.05, knn_mlp 0.08). Solo natives and GS-orth injects are **invalid** for ATH chase.

---

## 6. Bank residual screen (V24.1) — gated candidates

Gate: fold pos≥4, bootstrap CI_lo>0, MC priv_pos≥0.55, t≥2.5 · 23/70 passed.

| Residual | Nested | t | meanΔe5 | MC gap | Note |
|----------|--------|--:|--------:|-------:|------|
| `realmlp_v1` | 5/5 | 6.24 | +1.36 | +0.37 | Top; board-optimistic gap |
| `gb_bag5` | 5/5 | 4.85 | +0.20 | +0.03 | Conservative |
| `realmlp_dig_s100` | 5/5 | 4.70 | +2.19 | −0.28 | Prefer for private-aware |
| `lgb_diglead_l3` | 5/5 | 3.66 | +2.32 | −0.01 | Digit family |

---

## 7. Staged next-reset priority (quota was 0)

1. **V24.1 H1** (`w=0.35` dual_GB v15) → H5 → H7 → H4  
2. **V24.2** `ATH_x_nativeHGB_0.08` → `ATH_x_nativeBlend_0.05`  
3. **V24.3** `ATH_x_mlp_0.05` → `ATH_x_knn_mlp_0.08` → `ATH_x_mlp_0.10`  
4. Skip: pure natives, pure KNN, Ray-1 / twilight / old-R2 micro-GS

Kill lines: any board probe ≤ **0.95439** vs current ATH (except intentional private locks).

---

## 8. Technique taxonomy (quick reference)

| Technique | Valid? | Where it applies |
|-----------|:------:|------------------|
| Dual near-top bases + newer neural Δ | **Yes** | Current ATH generator |
| w-grid on identified recipe | **Yes** (staged) | H1–H3 |
| Ray-1 micro-step on sriram | Was yes → **exhausted** | Closed at 0.95426 crest |
| Twilight ±orth | **No** | Live −9…−14e-5 |
| Bank micro-GS orth @ ≤0.45e-4 | **No** for board | 5th-decimal / slight negative |
| Cross-family 90/10 matched peers | **Conditional** | Needs true near-equal public peers; wrong distance kills |
| Small-w native HGB/MLP into ATH | **Plausible** | Staged; not live yet |
| Solo RF / KNN / MLP / HGB | **No** for ATH | Diversity only |
| CSV roundtrip | **Null** | Drop |
| Nested + MC gates | **Yes** (method) | Always |
| Public-slice ladder chase >0.955 | **No** without recipe | Asterios warning |

---

## 9. Artifact index

| Artifact | Path |
|----------|------|
| This report | `docs/NOV17_LATEST_RESEARCH_SCORE_LEVER_REPORT.md` |
| V24 new levers | `docs/V24_NEW_LEVERS_RESEARCH.md`, `logs/overnight/GM_V24_NEW_LEVERS_REPORT.md` |
| Professor study | `docs/V24_1_PROFESSOR_LEVER_STUDY.md`, `logs/overnight/GM_V24_1_PROFESSOR_LEVER_STUDY.md` |
| Native RF/GBM | `docs/V24_2_NATIVE_RF_GBM.md`, `logs/overnight/GM_V24_2_NATIVE_RF_GBM_REPORT.md` |
| KNN/MLP | `logs/overnight/GM_V24_3_NATIVE_KNN_MLP_REPORT.md` |
| V23 research | `docs/V23_LIFT_RESEARCH.md`, `docs/V23_1_LIFT_RESEARCH.md` |
| V22 ladder | `docs/V22_LADDER_SUMMARY.md` |
| Historical keep/kill | `docs/THINGS_THAT_WORKED.md`, `docs/DEAD_ENDS.md` |
| ATH CSV | `submissions/sub_v24_SRI440_pub.csv` |
| Predictions | `/tmp/s6e11_v16/out/v24_{1,2,3}_*.npz` / `*_meta.json` |

---

*Generated 2026-11-17 · ATH 0.95440 · Research complete through V24.3; board slots for staged hyps pending next quota reset.*
