# S6E11 Things That Worked

Catalogue of approaches that **cleared a measured gate** (nested OOF, Fold-0 canary, or public LB) and should be **reused as building blocks**, not rediscovered.  
Companion to `docs/DEAD_ENDS.md`. Policy: prefer **STRICT NO_SUBMIT** diagnostics; promote only after the stated gate.

Locked CV: `StratifiedKFold(5, shuffle=True, random_state=42)`.

---

## 0. Current ATH lock (2026-11-15) — V22.3 R2

| Item | Detail |
|------|--------|
| **Public ATH** | **0.95426** |
| Formula | Ray-1 ridge continuation on sriram manifold @ `1−ρ(sriram)=1.25e-4` from P2 `0.95425` |
| Path | Slot9 `0.95399` → C1 `0.95407` → C2pure `0.95416` → P2 `0.95425` → **R2 `0.95426`** |
| File | `submissions/sub_v22_3_LIVE_ridge_continuation.csv` |
| Also tied ATH | V22.5 K2 · V22.6 SLOT1 |
| Gate for next inject | Prefer Ray-1 crest `[1.45e-4, 1.85e-4]`; twilight orth default **sign −1**; never GS-project after `G_twi` |
| Summary | `docs/V22_LADDER_SUMMARY.md` · `logs/overnight/GM_V22_3_RIDGE_CONTINUATION_REPORT.md` |

### V22 building blocks that transferred

| Block | Why keep |
|------|----------|
| sriram `rank_frac(L(gp)+0.15·(L(new)−L(old)))` manifold | All ATH lifts after 0.95407 lived here |
| Tiny `1−ρ` micro-steps | P2 `0.55e-4` (+3e-5 LB), R2 `1.25e-4` (+1e-5) |
| Sign-lock + `disp=1` var-match before Test inject | V22.5 vectors `5/5`, `max\|B_dir\|=0` |
| Exact-fname Kaggle poll | Required for multi-slot ladders |

---

## 0b. Prior ATH lock (2026-11-12 late) — superseded

| Item | Detail |
|------|--------|
| **Public ATH** | **0.95371** (L2) — superseded by V22 |
| Formula | `L(Q17 test_anchor_prob) + 0.03 · unitize(d1_v4)` |
| File | `submissions/sub_v12p3_slot2_anchor095370_plus_d1v4_5of5_LIVE.csv` |

Artifacts: `logs/overnight/GM_V12_3_LADDER_LIVE_REPORT.md`, `GM_V12_DAY_SUMMARY_2026-11-12.md`.

---

## 1. Early campaign (Nov 2) — public OOF ensembles

| Item | Detail |
|------|--------|
| Public OOF bank + rank/logit hillclimb | CV 0.95111 → public **0.95163** (first breakout) |
| Core rule / Stage 1 split proof | Adv AUC ≈ 0.50080; locked SKF(5, rs=42) |
| Stage 2 promotion gate | Reject unless OOF↑ **and** ≥4/5 folds↑ |
| TE control family | `te_xgb` / `te_xgb_smooth50` (~0.9493–0.9494 OOF) |
| Stage 5 shallow leaves | Best TE-LGBM at leaves=3 |

See also: `docs/AUDIT_REPORT.md` §What worked (Nov 2 snapshot).

---

## 2. Stacking ladder that compounded (Nov 5–7)

| Milestone | Nested / note | Keep |
|-----------|---------------|------|
| Part-8B 74-col newton-cholesky + hill + `gb_super` | `P_super_champ` ≈ 0.95214 | Bank recipe |
| `glm_quad` + orth into bank | `P_super_v3` **0.952149** | Keep |
| L1 public LM members + L4 regime | `P_super_v4` **0.952257** | Keep |
| `+12%` hill + `4%` GB_super multi-seed | `P_super_v4p_hc` **0.9523071** | **Step-0 champ** |
| Discrete residual heads (P4 L4.2/L4.3/L4.5 + `B_xgb_llr`) | `P_super_v5_harvest` **0.9523401** | Orth inject pattern |
| Wide + megayak + harvest residual | `P_super_v6_grandmaster` nested **0.9525275** (5/5 vs wide) | Residual-on-public formula |

**Generator law (validated):** after `KMAT_108`, continuous manifold methods add ≈0; lift comes from **discrete TE / digit / step-grid / token LLR**.

---

## 3. Target-first LMs that actually scored

| Member | OOF / note | Role |
|--------|------------|------|
| **Masaya GPT-2** (natural text, K=10) | Pooled **0.950027**; Fold0 **0.94888** | `MS` seed in ATH |
| **G26 GPT-2 seed2026** | OOF **0.950140** (5-fold); test **f0 only** public | `G26` seed; `G26−MS` orth axis |
| **0.5·G26 + 0.5·MS** | OOF **0.950439** | Two-seed rung |
| **Compact-LM L7** (`V=1372`, `T≈72`, seed7) | Band **0.9495–0.9499**; Fold0 ≈ **0.94761** | Public `goodpjw2008/s6e11-compact-lm-seed7`; second-seed candidate via seed2026 |
| **megayak compact** | Compact token format; NATURAL≈224 vs compact≈66–72 | Design source for L7 |

**Engineering that worked (Matapanino audit):**

- Pre-tokenize `(column, value)` BPE pieces once (`RowTokens`) — **no HF tokenizer in hot loop**.
- Target-first Bayes: score under True/False, log-sum-exp over orders + prior.
- Compact path: 1 learned col-token + value pieces + used-vocab LM head (`V=1372`) → trainable on A40 in hours, not days.
- Phase-0 **save shape unit test** (`oof=(n_val,K,2)`, `test=(299984,K_test,2)`) — mandatory before GPU (prevents `smugcqcy2oqzeg`-class crashes).

---

## 4. Nov 12 V12 ladder wins (zero-pod)

| Item | Detail |
|------|--------|
| **Pure Q17 anchor** (`test_anchor_prob` / `anchor_095370`) | Public **0.95370** (L1) — submit verified external apex as-is |
| **+0.03·unitize(d1_v4)** on that anchor | Public **0.95371** (L2) — only private residual that transferred (~55% of OOF) |
| Zero-pod recon + Kaggle CLI | Recovered Q17 bank without GPU; `kaggle competitions submit --competition` |

---

## 5. Ops / process wins

| Practice | Why it worked |
|----------|----------------|
| Fold-0 canary before Folds 1–4 | Killed HiRes at ~$0.44 instead of full 5F |
| Hard pod watchdog (`sleep 300m` + spend cap) | Bound runaway GPU cost |
| Ignore foreign / blacklisted pods (`lheroidmuab1j4`) | No accidental spend |
| **Hard zero-pod rule** when ordered | Prevents spend / state drift during local-only days |
| `NO_SUBMIT` + staged `*_NOSUBMIT.csv` | Preserves daily slots; 4/5 near-misses do not burn |
| Double-orth before LOFO | Stops false “5/5” from soft-orth optimism |
| Platt / prevalence-calibrated `base_margin` | Avoids early-stop collapse on rank-logit offsets |
| Inject **Δz residual**, never α=1 overwrite of LB anchor | Preserves ATH backbone |
| Trust **public transfer history**, not OOF 5/5 alone | L3/L4 both 5/5 OOF → public red |

---

## 6. Highest-leverage next actions (reuse these wins)

1. Finish **Compact-LM seed2026 K=10 folds 1–4** (fold0 local only today); re-test pure two-seed residual with subfold LOFO vs L2.
2. Recon for any new public **0.9538x** artifact before burning quota.
3. Do **not** re-blend Q17+d1 with P_super / stageB / sov+G26 without a new family.
4. Hold submits unless gate: LOFO 5/5 **and** prior public-transfer family (d1-like).

---

*Last updated: 2026-11-12 · ATH 0.95371 · Companion to `docs/DEAD_ENDS.md`.*
