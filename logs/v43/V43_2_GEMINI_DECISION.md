# Gemini decision after V43.2 local 4-engine (2026-11-24)

**Token:** `APPROVE_POD_F0_ONLY`  
**Submit:** **NO** (not `APPROVE_SUBMIT_NOW`)  
**E4 NOSUBMIT CSV:** **do NOT submit** (worth zero)

## Gemini rationale (verbatim bullets)

* **Why all 4 local CPU engines hit exact +0.00e-5 on z_modern_pub (0.952652):** z_modern_pub already linearly spans 147 models across BPE-GPT2, Quantile-LMs, RealMLP/PyTabKit, TabPFN, GLMs, and shallow/deep GBDTs (0.952652 OOF), leaving zero unspanned variance on standard 24-column discrete/sub-float engineered features. Furthermore, E1 and E2 collapsed on pre-unitize raw dispersion (0.750, mirroring our earlier G_lgb 0.62–0.84 inner-TE bagging shrinkage trap), confirming no honest local tabular residual exists here.

* **Verdict on NOSUBMIT_E4_pillar_moe (0.00e-5, sf=12/25):** Worth zero—do NOT submit. sf=12/25 is a coin flip (48%), Δe5 = +0.00, and cos·S06 ≈ ±1.0 shows the orthogonalizer leaked residual covariance onto our live loser axis S06, virtually guaranteeing ≤ 0.95569 and wasting a slot.

* **Next concrete move (LAUNCH F0 — Qwen2.5-0.5B (N, 24) Per-Field LLR ONLY; skip TabICL):** Authorize **LAUNCH F0**, **stripping TabICL out of the Fold-0 script before boot** because TabICLv2 scored 0.94324 (0/5 subfolds, -4.7e-5) on A40 in V39 and wastes ~20–40 minutes. Dedicate 100% of the pod to **Qwen2.5-0.5B single-digit BPE + exact (N, 24) un-permuted per-field conditional LLR extraction**, then fit fold-clean Ridge + shallow GBDT on the 24-column LLR matrix against z_modern_pub[f0].

* **Risks before booting F0:**
  1. Never materialize full (B, T, 151936) float32 log_softmax; use `F.cross_entropy(..., reduction='none')` / fused chunked gather in bf16 with `USE_COMPILE=0`, `attn_implementation="sdpa"`, pre-tokenized uint16 field arrays, matched `K_oof == K_test == 4`.
  2. Identical y=1/y=0 prefix token length; mask prefix+pad; snapshot foreign pods; 21600s watchdog; rsync `.npy` before gate/delete; hard ≥ +5.0e-5 (≥4/5 subfolds on z_modern_pub[f0]) with **0 Kaggle submissions**.

## Agent status
- V43.2 suite finished; report at `logs/v43/V43_2_LOCAL_4ENGINES_REPORT.md`
- HARD STOP held: **0 submissions, 0 pods touched**
- Awaiting explicit user `LAUNCH F0` (Qwen-only, TabICL stripped per Gemini)
