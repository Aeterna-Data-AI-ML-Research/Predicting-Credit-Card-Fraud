# V43 Stage-1 — Local Prep + Pod Spec (NO_SUBMIT · 0 pods)

**HARD STOP.** `NO_SUBMIT=True`. **0 GPU pods started** (`runpodctl pod list` empty).  
Await explicit authorization before Fold-0 canary launch.

---

## 0. Tooling

| Tool | Path / status |
|---|---|
| Kaggle CLI | `/Users/mz/Library/Python/3.14/bin/kaggle` ✓ (v2.2.4) |
| runpodctl | `/Users/mz/.local/bin/runpodctl` ✓ (3.1.1) |
| Pods running | **0** |

---

## 1. Recovered assets & `0.96153` / `orig20k` forensics ($0 CPU)

### Shapes fixed / extracted
| Source | OOF | Test |
|---|---|---|
| `l7_s2026_5f` | `(699960, 10, 2)` | `(5, 299984, 10, 2)` |
| `gpt2_med_4ep_5f` (shape-fixed) | `(699960, 8, 2)` | `(299984, 4, 2)` mean over folds |

Recipe recovered from `scripts/gm_v16_9_*` / `gm_v19_12_*` (`u_bank_from_llr`, `gini10`).

**Saved → `preds/v43_order_tensor/` (112 `.npy`):**
- `U5..U10` from l7 + gpt2 (`U5_gini10_llr` canonical alias)
- 7 order-curvature feats each: `R2_bayes_lse`, `R3_ranklogit`, `R2_minus_R3_curv`, `R5_antithetic_absdiff`, `order_std_y1`, `order_std_llr`, `order_min_max_gap`
- `U_C_tab_proxy` = `0.55·B_xgb + 0.45·l7.R3` (V22 formula; original Cat/R2 resid tensors not on disk as named files)

### Forensics (`GM_V33_1_HARDENED_FORENSICS`)
- **orig20k** `gen.py` = interaction/tree (Linear 0.880 → depth-2 HistGBDT **0.934**); synthesizer residual **does not** lift train (**0/5**).
- Simulated Public-LB probing max Δ ≪ Ahmet/Dharun breakaways.
- **`0.96153@K104` exceeds black-box 60k/80k `Query_Pub` bound → NOT leak-free OOF from LB probing**; needs external/structural info. No public `gen.py` leak found.

### Eval on `z_modern_pub+Zr = 0.952652`  
(LOFO `w ∈ [-0.03,+0.03]`, continuous Method-B ECDF)

| feature | Δe5 | pos/5 | sf/25 | t25 | cos·S02 | cos·S06 |
|---|---:|---:|---:|---:|---:|---:|
| **l7_R5_antithetic_absdiff** | **+1.06** | **4/5** | **15** | **1.05** | +0.008 | −0.037 |
| l7_order_std_llr | +0.84 | 3/5 | 13 | 0.83 | +0.010 | −0.048 |
| **l7_U5_gini10_llr** | **+0.74** | **4/5** | **14** | **1.58** | −0.006 | −0.077 |
| l7_U7_fwd_rev_gap | +0.66 | 4/5 | 14 | 1.94 | −0.005 | −0.066 |
| l7_U6_pair_std5_llr | +0.52 | 4/5 | 14 | 2.07 | −0.009 | −0.086 |
| gpt2_R3_ranklogit | +0.39 | 4/5 | 16 | 0.95 | +0.089 | +0.026 |
| U_C_tab_proxy | +0.22 | 3/5 | 15 | 1.12 | +0.068 | +0.026 |
| gpt2_U5_gini10_llr | −0.52 | 2/5 | 8 | −1.10 | +0.005 | −0.043 |

**Note:** Historical `U_C_tab` (sf 22/25, t=5.77) not fully recovered as live tensors; proxy is weak. Recovered `U5` is positive but micro (≪ +5e-5 gate). Best local order signal = **l7 R5 antithetic**.

JSON detail: `logs/v43/V43_STAGE1_U5_ORDER_EVAL.json`

---

## 2. Pod harness (built · dry-run verified · 0 pods)

Package: `scripts/v43_pod_qwen_perfield_llr_and_tabicl/`

| Module | File | Status |
|---|---|---|
| Serializer + unpermute | `serializer_24field.py` | ✓ |
| A — Qwen2.5-0.5B `(N,24)` field LLR | `qwen_digit_genbayes_24llr.py` | dry-run exits without GPU |
| B — TabICLv2 full-context | `tabicl_v2_fullcontext.py` | dry-run exits without GPU |
| Smoke 16-row | `smoke_test_16row.py` | **PASS** (`logs/v43/V43_SMOKE_16ROW.json`) |
| Launch gate | `launch_fold0_canary.sh` | refuses unless `V43_AUTHORIZE_POD=YES` |

### Smoke asserts passed
- Digits `0–9` single-token (byte tokenizer; HF transformers not in local venv — pod will use real Qwen tok)
- Antithetic K=8 (4 fwd + 4 exact rev)
- `assert unpermuted_field_sum == field token LL` to `1e-5`
- `field24` shape `(16, 24)`; seq contract `(N, 8, 2)`
- `StratifiedKFold(5, shuffle=True, random_state=42)` fold0 = **559968 / 139992**

### Module A contract (on pod with `V43_LAUNCH=1`)
- Backbone `Qwen/Qwen2.5-0.5B` bf16 SDPA, LoRA-r64 (or FFT), cosine LR, **4 epochs** Fold-0
- Class-prefix `y: 0/1` + K=8 antithetic column perms
- Saves `oof_f0_seq_llr.npy (139992,8,2)`, `test_f0_seq_llr.npy (299984,8,2)`, `*_field24_llr.npy (N,24)`, `*_field24_std.npy (N,24)` with canonical un-permute

### Module B
- `TabICLClassifier`, stratified ~150k context chunks × multi-estimator × feat-perm views; Fold-0 va+test (~5 min GPU)

### Live catalog prices (Secure POD, read this session)
| GPU | $/hr secure | Stock (at read) |
|---|---:|---|
| **RTX 4090** 24GB | **$0.89** | LOW (EU-RO-1) |
| **A100 SXM** 80GB | **$2.19** | LOW |
| **H100 SXM** 80GB | **$4.39** | LOW |
| Community 4090 | $0.34 | **NONE** at read |

### Wall-clock / $ estimate (Qwen 0.5B LoRA + TabICL)

| Scope | 4090 | A100 SXM | H100 SXM |
|---|---|---|---|
| **Fold-0 canary** (train 4ep + score K=8×2 + TabICL ~5m) | **~3.0–4.0 h → $2.7–$3.6** | **~2.0–2.5 h → $4.4–$5.5** | **~1.5–2.0 h → $6.6–$8.8** |
| **Full 5-fold** (~5× F0) | **~15–20 h → $13–$18** | **~10–13 h → $22–$28** | **~8–10 h → $35–$44** |

Recommend **1× Secure 4090** for Fold-0 canary (best $/hr with stock). Gate: solo OOF ≥ 0.948 + field24 ZCA vs `z_modern_pub` before scaling to 5-fold.

---

## 3. STOP

- Report: this file + `V43_STAGE1_U5_ORDER_EVAL.json` + `V43_SMOKE_16ROW.json`
- **NO_SUBMIT=True. Zero submissions. Zero pods.**
- Next: await explicit **“AUTHORIZE V43 FOLD-0 GPU CANARY”** before any `create-pod` / `V43_LAUNCH=1`.
