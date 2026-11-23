# V37 STEP 1 — Twin Hunt + Deotte S6E10 Clone + Fold-0 GPU Canary Staging

**Timestamp:** 2026-11-22 (local)  
**Mode:** `NO_SUBMIT=True` · $0 external APIs only · no CPU GBDT/MLP training  
**Verdict:** **LAUNCH RunPod 1× RTX 4090 / L40S immediately** for Fold-0 TFM+LM canary (hard cap **$8**, self-destruct **10800s**).

---

## STEP 1A — Global Twin / Parent Dataset & LB Recon

### HuggingFace Hub (`/api/datasets?search=…`)

| Query | Hits (schema-relevant) |
|---|---|
| `is_ai_generated_scam_attempt` | **0** |
| `cvv_retry_count` | **0** |
| `merchant_risk_score` | **0** |
| `uditjain` | 2 unrelated (`adaption-financial-math-qa`, `veneer-bench`) |
| `credit-card-fraud-detection-2026` | **0** |
| `s6e11` | **0** |

**Dump:** `external/v37_twin_check/api_dumps/hf_hub.json`

### OpenML REST

Name-list endpoints mostly return **HTTP 412**. Recovered:

- `Credit_Card_Fraud` (did **46568**) — classic PCA fraud schema, **not** S6E11 columns.
- Known `creditcard` (did **1597**) — V1–V28 + Amount + Class — **no** S6E11 schema twin.

**Dump:** `external/v37_twin_check/api_dumps/openml.json`

### Kaggle `datasets list -s` (Warning: lines stripped)

| Query | Material finding |
|---|---|
| `is_ai_generated_scam_attempt` | No datasets |
| `cvv_retry_count` | Misc train dumps; not S6E11 twins |
| `merchant_risk_score` | Generic payment-fraud sets; weak column overlap |
| `uditjain` | Author PEFT/QA sets — **no** S6E11 twin |
| `credit-card-fraud-detection-2026` | **`uditjain13/…` = competition ORIGINAL (20k)**; Mohan 2026 sets ≠ schema; public GPT-2 OOF banks |
| `s6e11` | OOF/board artifact datasets only |

**Dump:** `external/v37_twin_check/api_dumps/kaggle_list.txt`

### Downloaded candidates → `external/v37_twin_check/`

| Dataset | Rows | Schema ∩ S6E11 | Test key overlap (`amt\|bal` / multi-col) |
|---|---:|---:|---|
| `uditjain13/credit-card-fraud-detection-2026` | **20,000** | **25/26** (full parent schema) | **0** exact full-row hits vs `test.csv` **and** vs `train.csv` (fraud rate 0.01695 ≈ PREV) |
| `mohankrishnathalla/credit-card-fraud-detection-dataset-2026` | (downloaded) | **5** | not a twin |
| `harshjain123/synthetic-credit-card-fraud-interpretable` | large | **7** | not a twin |

**Conclusion 1A:** No public external set **>20k** shares S6E11’s schema *and* overlaps `test.csv`. The only exact-schema parent is the **20k original** (already used as `orig20k`). Twin/leak shortcut is **closed**.

### Leaderboard Top-25 (`playground-series-s6e11`, 2026-11-22T06:24Z)

| Rank | Team | Score | Subs |
|---:|---|---:|---:|
| 1 | Dharun MK | **0.96242** | 114 |
| 2 | Smart Manoj | 0.96140 | 210 |
| 3 | 강북구집돌이 | 0.95754 | 192 |
| 4 | srikalyan | 0.95703 | 117 |
| 5 | Nibedita Biswal | 0.95680 | 66 |
| … | … | … | … |
| 25 | Abdelghani Charouiti | 0.95557 | 32 |

**Our ATH:** ~**0.95548** → gap to #1 ≈ **+694 bps**.  
**File:** `external/v37_twin_check/lb/playground-series-s6e11-publicleaderboard-*.csv` + `api_dumps/lb_top25.csv`

### G26 Hub check — `goodpjw2008/s6e11-target-first-gpt2-seed-2026`

| File | Present? |
|---|---|
| `gpt2_s2026_oof_5fold.npy` | yes |
| `gpt2_s2026_test_5fold.npy` | yes (aggregate) |
| `gpt2_s2026_test_f0.npy` | **yes — Fold 0 only** |
| Test Folds **1–4** of G26 (`0.95014` OOF) | **NOT uploaded** |

Cannot harvest G26 per-fold test residuals for 1–4 without re-running the model.

---

## STEP 1B — Chris Deotte S6E10 / S6E9 Playbook Clone

**Clone:** `git clone --depth 1 https://github.com/cdeotte/kaggle-playground-solutions.git` → `external/cdeotte_2026_1st/`

### S6E10 — *TFMs and LLMs with tabular data* (1st place)

**Stack CV 0.96284 / Private 0.96314** from 27 own models (L2 logistic). Key recipes:

#### Tabular Foundation Models (zero-shot, **full outer-train context**, raw features)

| Model | Recipe (exact) | OOF AUC |
|---|---|---:|
| **TabPFN-3.5** | `TabPFNClassifier(model_path='tabpfn-v3.5-20260909.safetensors', device='cuda', ignore_pretraining_limits=True, n_estimators=8, fit_mode='fit_with_cache')`; cat indices passed; query chunk 50k | 0.961145 |
| **TabICLv2** | `TabICLClassifier(n_estimators=8, batch_size=1, kv_cache=False)`; **fp16 autocast disabled** (precision bug at ~560k context) | 0.958088 |
| **TabFM 1.0** | bf16, 8 estimators + **exact context cache** (`tabfm_cache.py`) so train context processed once | 0.958871 |
| + Kumo / EXAONE / LimiX / Causilo / Mitra | same full-context zero-shot pattern | lower |

**Note:** There is **no package named “TabPack”** in Deotte’s repo. Ensemble diversity comes from `n_estimators` + full-context TFMs. Our Fold-0 canary implements **stratified context-bagging (32×10k)** as the VRAM-safe 4090/L40S analogue of full-context TabPFN when ~560k context is too large / too slow for a ≤40 min canary.

#### LLM / GPT-2 target-first (“Storyteller”)

1. Start from **GReaT replica** trained on **original data only**.
2. Fine-tune causal LM on fold outer-train rows written **label first**:  
   `satisfaction is True, Age is 36, Class is Eco, …` with **random field order** each step.  
   AdamW **lr=5e-5**, batch 32, **8 epochs**, bf16, grad clip 1; medium/large use EMA 0.9999.
3. Score with Bayes:  
   `LLR = logsumexp_o log p(fields | True, order_o) − logsumexp_o log p(fields | False, order_o)`  
   over **K=30** field orders; `sigmoid(LLR)` → OOF.  
   GPT-2 small target-first OOF **0.962425** (strongest single model).

Also: target-last “Detective”, label-anywhere, GPT-Neo 125M, hybrid BCE, BatchEnsemble heads, replica-world transfer, Q-FT (did not help stack).

### S6E9 — *The AI Relay* (1st place)

Same generator family: **GReaT label-first GPT-2**. Winning stack = **60 models** L2 LR (XGBoost TE banks + RealMLP/TabM + **GPT-2 target-first / chains / Yes-No / likelihood features**). Confirms target-first LM + TE/GBDT diversity — not GBDT alone — is the #1 pattern for GReaT playgrounds.

---

## STEP 1C — Fold-0 GPU Canary Staged

**Script:** `scripts/v37_gpu/run_fold0_tfm_and_transductive_lm.py`  
**Local smoke (`--smoke --skip2`):** OK — Fold-0 va=**139,992**, test=**299,984**, feature matrix **32 cols** (24 raw + cents/mantissa + fold-clean/transductive freq). Smoke logistic bags correctly **FAIL** gate (solo_f0≈0.71); plumbing + self-destruct + `NO_SUBMIT` verified.

### Canary 1 — Deotte-style TabPFN bags (~15 min GPU)

- Prefer `tabpfn.TabPFNClassifier` on CUDA; else logistic fallback (CPU smoke only).
- **32** stratified bags × **10,000** train rows (`pos:neg` @ prevalence **0.0167**).
- Features: raw + exact cents/mantissa + fold-clean KMAT/freq + train∪test freq bins.
- Probas **recalibrated** so mean ≈ 0.0167.
- Score Fold-0 va ∪ full test.

### Canary 2 — Transductive target-first compact LM (~25 min GPU)

- Compact vocab **V ≤ 2048**; Tiny GRU-LM (GPU) pretrained on Fold-0 train with **Y-first** tokens.
- Every row includes transductive tokens: `amt_freq_bin`, `bal_freq_bin`, `cents_exact`, `orig20k_min_dist_bin`, `flag64_id`.
- Score **K=10** antithetic reverse-pair column permutations → mean LLR → recalibrated P.
- (Full Deotte GPT-2-small + 30 orders is the promote-path upgrade if canary clears gate.)

### Hard Fold-0 gate

Against `z_fullbank_oof[f0]` and `z_ATH_twin_oof[f0]`:

1. `R²(δ_test | SVD_40(B_152)) < 0.75`
2. `≥ 4/5` Fold-0 subfolds with `ΔAUC ≥ +2.5e-4`

**Pass** → continue Folds 1–4 for that canary.  
**Fail** → rsync `logs/overnight/v37_fold0/` and `runpodctl remove pod $RUNPOD_POD_ID`.

### Pod contract

```bash
# on pod boot
(sleep 10800 && runpodctl remove pod $RUNPOD_POD_ID) &
export NO_SUBMIT=True
python -u scripts/v37_gpu/run_fold0_tfm_and_transductive_lm.py
```

Hard spend cap: **$8**. No Kaggle submit from this worker.

---

## Launch recommendation

| Question | Answer |
|---|---|
| Twin / parent leak found? | **No** (orig20k only; 0 test overlap) |
| G26 test folds 1–4 on Hub? | **No** (f0 + 5fold agg only) |
| Deotte #1 recipe clear? | **Yes** — TabPFN/TabICL full-context TFMs + GPT-2 target-first Storyteller |
| Fold-0 canary staged + smoke OK? | **Yes** |
| **Launch 4090/L40S now?** | **YES — immediately** |

CPU GBDT/MLP waves are spanned; #1 is **~+0.007** away and sits on TFM+LLM stack territory Deotte already proved on GReaT playgrounds. Fold-0 GPU canary is the cheapest falsifier before committing to full 5-fold.

---

## Artifacts

- `external/cdeotte_2026_1st/` — Deotte solutions clone  
- `external/v37_twin_check/` — LB, API dumps, candidate downloads  
- `scripts/v37_gpu/run_fold0_tfm_and_transductive_lm.py` — ready worker  
- `logs/overnight/v37_fold0_smoke1.txt` + `logs/overnight/v37_fold0/fold0_canary_result.json` — local smoke  
