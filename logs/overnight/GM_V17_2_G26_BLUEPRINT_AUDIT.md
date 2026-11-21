# GM V17.2 — G26 (`SEED=2026`) Production Blueprint Audit

**STRICT `NO_SUBMIT=True` · 0 GPU pods · read-only**

Script: `scripts/gm_v17_2_g26_blueprint_audit.py`  
Artifacts: `/tmp/s6e11_v16/g26/`, `/tmp/s6e11_v16/out/v17_2_lift.json`  
Sources: `external/matapanino_gpt2_repo/`, `external/gpt2_target_first_09501/`, `external_bank/masaya_gpt2/`, `/tmp/s6e11_v16/g26/README.md`

---

## Step 1 — Exact `G26` pipeline stages & hyperparameters

### README (`/tmp/s6e11_v16/g26/README.md`) — how goodpjw2008 built G26

Second seed of the **target-first GPT-2** classifier (Chris Deotte S6E10 recipe), trained with public Matapanino code (`github.com/Matapanino/s6e11-gpt2-target-first`, MIT) that produced masayakawamata’s seed-42 OOF. **Same settings, seed 2026.**

| Artifact | Shape | Notes |
|---|---|---|
| `gpt2_s2026_oof_5fold.npy` | `(699960,)` | OOF log-odds; **5-fold AUC 0.950140** (folds 0.94919 / 0.95037 / 0.95524 / 0.95021 / 0.94586); vs masaya seed 0.950027, rank corr 0.989 |
| `gpt2_s2026_test_f0.npy` | `(299984,)` | **Fold-0 model only**, **K=10** column orders; folds 1–4 “still scoring (~12.6 GPU-h each)” |

Folds: `StratifiedKFold(5, shuffle=True, random_state=42)` on `train.csv` — **identical** to masaya `fold_ids.npy`.

Scoring recipe (README): rows written **label first**, shuffled column order; **log-sum-exp over 10 orders** of `log p(row \| True) − log p(row \| False)` + fold training prior log-odds.

Masaya public kernel (`external/gpt2_target_first_09501/`) is **presentation-only** (GPU off): dataset source `masayakawamata/s6e11-gpt2-target-first-oof` ships `oof.npy` / `test.npy` / `fold_ids.npy` / `manifest.json` — **no** `.pt` / safetensors. Manifest: pooled AUC **0.950027**, diagnostics show AUC@K=1/6/10; model string `gpt2_target_first`.

---

### Pipeline stages (`make_queue.py`)

```
folds → replica (ONCE) → fit_f0 → score_oof_f0
      → fit_f{1..4} → score_oof_f{1..4}
      → [--stage all] score_test_f{0..4} → make_predictions → xgb → stack
```

| Stage | Script | Scope |
|---|---|---|
| 0 | `make_folds.py` | Once; SKFold seed **42** |
| 1 | `train_replica.py` | **Once globally** → `RUN/replica/final` |
| 2 | `train_classifier.py` | **Per fold** ← continues from replica |
| 3 | `score_rows.py` | Per fold × {oof, test} |
| 4 | `make_predictions.py` | Log-sum-exp over orders + prior → `oof.npy` / `test.npy` |

---

### Question A — `train_replica.py`: once globally or per fold?

**Once globally.** Not per fold. Not unsupervised DA on competition `train+test`.

- `make_queue.py` emits a single `"replica"` job; every `fit_f{k}` has `after=["replica"]` and `--replica RUN/replica/final`.
- Docstring / post call: `GReaT(llm="gpt2", batch_size=16, epochs=60).fit(original_df)` on the **original 20k CSV** (uditjain13), drop id — **not** competition train∪test.
- Defaults: `--epochs 60`, `--batch-size 16`, `--precision fp32`, `--seed 42`, `--model gpt2`.
- **G26 match:** same hyperparams with `--seed 2026`.

**Published replica checkpoint?** **No.**  
`docs/DEAD_ENDS.md` **N9**: Masaya/goodpjw kernel outs = aggregated OOF/test + HTML only; weights not public. Local datasets (`external_bank/masaya_gpt2/`, `external/gpt2_target_first_09501/`, `/tmp/s6e11_v16/g26/`) contain **npy + README/manifest only** — no `replica/final` safetensors. Regenerating G26 requires **re-running replica** (or obtaining a private copy).

---

### Question B — `train_classifier.py`: exact args for folds 1..4 to match `gpt2_s2026_oof_5fold.npy`

`make_queue.fit(fold)` does **not** pass `--seed` (defaults to 42). G26 must override seed explicitly.

| Arg | Value to match G26 |
|---|---|
| `--seed` | **2026** |
| `--fold` | `1`, `2`, `3`, `4` (fold 0 already done for test) |
| `--epochs` | **8** |
| `--batch-size` | **32** |
| `--lr` | **5e-5** (AdamW, linear decay → 0) |
| `--precision` | **bf16** (post / default) |
| `--layout` | **first** (target-first) |
| `--label-text` | **truefalse** |
| `--replica` | `RUN/replica/final` (seed-2026 replica) |
| `--fold-ids` | same `fold_ids.npy` as masaya (SKFold seed 42) |

**No `--orders` on `train_classifier.py`.** Training draws a **fresh random column order per step** via `random_orders([seed, fold, step, …])`. `--orders` belongs only to `score_rows.py`.

Queue-equivalent command (per fold `f`):

```bash
python train_classifier.py \
  --train-csv train.csv --fold-ids fold_ids.npy --fold f \
  --id-col id --target-col is_fraud \
  --replica RUN/replica/final --out RUN/fits/f{f} \
  --epochs 8 --batch-size 32 --lr 5e-5 --precision bf16 \
  --layout first --seed 2026
```

Then score OOF/test with the **same** `--seed 2026` so chunk orders match G26:

```bash
python score_rows.py \
  --model-dir RUN/fits/f{f}/final --csv {train|test}.csv \
  --split {oof|test} --fold-ids fold_ids.npy --fold f \
  --id-col id --target-col is_fraud --out RUN/scores/f{f}_{split} \
  --orders 10 --batch-sequences 256 --precision fp32 --seed 2026
```

(`--orders 10` matches G26 README / shipped `test_f0`; Matapanino `make_queue` default is **30**. Masaya manifest diagnostics quote AUC at K∈{1,6,10}.)

---

### Question C — Why `score_rows.py` is slow on Kaggle P100; y=0/y=1; speedups

**Cost drivers (G26 test, one fold):**

| Factor | Detail |
|---|---|
| Sequences | `N_test × K × 2` = `299984 × 10 × 2 ≈ 6.0e6` forwards / fold |
| Length | BPE median **L≈224** (vs Compact-LM L7 T=72 → ~3.2× FLOP/forward) |
| Precision default | **`fp32`** (`--precision` default; post code); bf16/tf32 optional + rounding noise |
| Batch | `--batch-sequences 256` (memory knob; scores independent of it) |
| Padding | **None** — buckets by exact token length |
| lm_head | Full-vocab softmax in **fp32 chunks** of 4096 tokens (memory) |
| Wall (README) | **~12.6 GPU-h / fold** on the hardware that scored G26 test_f0 (P100-class) |

**y=0 and y=1: single batched pass (not two separate passes).**  
In `score_chunk` (`layout=first`):

```text
bodies_per_batch = batch_sequences // 2
ids_true  = [prefix_True  ‖ body]
ids_false = [prefix_False ‖ body]
input_ids = concat([ids_true, ids_false])   # one forward
```

Same column orders under both labels; one transformer call scores both.

**Estimated speedups (vs P100 @ fp32 K=10):**

| Change | Effect |
|---|---|
| `bf16` vs `fp32` | ~**1.5–2×** on Ampere+ (`score_rows --precision bf16`); transformer autocast; lm_head still fp32 chunks |
| `batch_sequences` 256→512 | Minor wall; docstring: sets memory, not speed |
| SDPA / FlashAttention | HF GPT-2 path can use SDPA on 4090/A100 if enabled by transformers; not a custom path in `score_rows` |
| Wall / fold (prior V17.0 scale) | P100 **~12.6 h** → RTX 4090 **~1.7 h** → A100 **~1.4 h**; folds 1–4 test ≈ **6–7 h** total on one 4090/A100 |

`train_classifier.py` already defaults to **bf16** @ bs=32 — the scoring stage is the bottleneck, not the 8-epoch fit.

---

## Step 2 — Expected ensemble lift: 1-fold `G26_test_f0` → 5-fold (and 3-fold)

### Calibration (STRICT formula)

LIVE carries **|β_G26| ≈ 0.226**. OOF stack already uses true **5-fold** G26 (`0.95014`); Test uses only **f0**.

σ calibrated on **L7_s2026** unitized test folds (proxy for fold-dispersion of a same-recipe LM):

| Quantity | Value |
|---|---|
| Mean pairwise Spearman across 5 L7 test folds | **ρ ≈ 0.9923** |
| `σ_1f−5f = std(u_f0 − mean_5 u_f)` | **0.0727** |
| `σ_1f−3f = std(u_f0 − mean_3 u_f)` | **0.0659** |

Noise model: ε ~ N(0, σ²) on unitized scale; **lift** =  
`AUC(z_ATH) − AUC(z_ATH + w·ε)` with `w=0.226`  
(= gain of replacing noisy 1-fold G26 with clean 5-fold G26 already in `z_ATH`).

Variance-reduction intuition: 5-fold averaging cuts 1-fold residual variance by ~(1−ρ)·4/5; here ρ is already ~0.992, so residual σ is small → **sub-tick** expected lift under this calibration.

### Monte Carlo results (`/tmp/s6e11_v16/out/v17_2_lift.json`)

Units: **e-5 AUC** (÷10 ≈ public LB ticks of 1e-4).

| Replacement | Main 5 OOF folds MC (mean) | Pooled 120k MC mean [p10, p90] |
|---|---|---|
| **G26_test 5-fold vs f0** | **+0.24 e-5** (std across runs ~1.09; mean pos folds 2.8/5; all-5-pos 10%) | **+1.38 e-5** [−1.89, +5.68] |
| **G26_test 3-fold (f0..2) vs f0** | **+0.20 e-5** | **+0.74 e-5** [−2.02, +4.33] |
| Upper if fold ρ were only 0.98 (`σ=√(0.8·0.02)`) | — | **+2.54 e-5** [−2.90, +7.30] |

Fold-0 stratified subfolds (one draw, e-5): `[−1.25, +7.50, +1.77, −1.33, +8.60]` — noisy, sign-mixed; consistent with ~0–few e-5 scale, not multi-tick.

### Public / Private expected gain (verdict)

| Scenario | Expected Public & Private AUC gain |
|---|---|
| Finish **true 5-fold** `G26_test` | **≈ +0.2 … +1.4 e-5** (~**0.02–0.14** of a 1e-4 tick); 90%-ish band roughly **−2 … +6 e-5** |
| Finish **3-fold** (f0..2 mean) | **≈ +0.2 … +0.7 e-5** — slightly weaker, still sub-tick |
| Optimistic low-ρ (ρ=0.98) upper | still **≲ +3 e-5** mean |

**Not a multi-tick free lunch.** Fold agreement on the L7 proxy is already ρ≈0.992, so 1f→5f variance reduction at LIVE’s 0.226 weight is tiny versus ATH saturation (`≤ +0.5e-5` linear contrasts). Completing G26 test folds remains **correct for fidelity / removing f0-only bias**, but **expected LB move under this noise model is ≪ 1 public tick**.

Caveats: (1) σ from Compact-LM L7, not G26 itself (G26 test folds 1–4 unavailable); (2) assumes additive unitized noise at fixed w; (3) does not model f0 systematic bias beyond isotropic ε (V17.0: f0 cross-seed ρ≈0.33, R²≈0.11 — weak debias).

---

## Production blueprint (if regenerating G26 test f1–4)

1. **Replica once** (`--seed 2026`, ep60, bs16, fp32) on original 20k — **must train**; not downloadable.
2. **Classifier** per fold 1..4: `--seed 2026 --epochs 8 --batch-size 32 --lr 5e-5 --precision bf16 --layout first`.
3. **Score test** each fold: `--seed 2026 --orders 10 --precision fp32|bf16 --batch-sequences 256`.
4. **Average** `test_f{0..4}` (or f0..2 for 3-fold interim) → replace LIVE’s G26 slot.
5. **Runtime (0 pods this step):** plan ~**1.4–1.7 h/fold** score on A100/4090 bf16; ~**6–7 h** for four missing test folds. Do **not** expect ≥1e-4 LB from this alone.

---

## Guardrails confirmation

| Check | Status |
|---|---|
| `NO_SUBMIT=True` | **Held** — no Kaggle submit |
| GPU pods launched | **0** |
| Training / scoring executed | **None** — source + MC only |

---

*End GM V17.2*
