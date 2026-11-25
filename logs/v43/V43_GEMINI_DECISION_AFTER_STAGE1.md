# Gemini decision after V43 Stage-1 (2026-11-24)

**Token:** `APPROVE_POD_F0_ONLY`  
**Submit tonight:** **NO** (not `APPROVE_SUBMIT_NOW`)

## Rationale (Gemini)
- R5/U5 micro rays (+1.06 / +0.74 e-5) are **below** the 1-tick bar; Cand3 (+1.53e-5) already printed **flat 0.95569** today — don't peephole.
- True escape = Qwen2.5-0.5B `(N,24)` field-LLR + TabICL Fold-0 canary (~$2.7–3.6), **0 LB burn** until ≥+5e-5 Fold-0 gate.

## Guardrails Gemini required
1. Snapshot pods first; never touch foreign pods; 6h (21600s) dead-man watchdog.
2. TabICL Fold-0 first (≤20 min) → rsync → then Qwen Fold-0; SHA-verify before `remove pod`.
3. Gate on z_modern_pub[f0]+Zr: solo seq ≥0.9485; nested field24+TabICL orth ΔAUC_f0 ≥ +5.0e-5 with ≥4/5 subfolds; disp/KS/cos checks.
4. Hard stop after Fold-0 sync: **0 Kaggle submissions**; report scorecard before Folds 1–4 or any submit.

**Agent status:** consultation done; **no submit fired**; awaiting user `LAUNCH F0` (or equivalent) to spend pod budget under these rules.
