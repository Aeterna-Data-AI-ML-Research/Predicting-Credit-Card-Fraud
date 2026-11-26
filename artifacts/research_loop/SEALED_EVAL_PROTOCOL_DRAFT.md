# Sealed evaluation protocol (draft) — S6E11 fraud

**Status:** DRAFT pending Astra-high review. Not yet unblinded.

## What this is not
- Not a new random reshuffle of the same SKF(5, rs=42) rows already used for LOCAL_GATE_PASS / signed-CB / Round6–7 research.
- Not “folds 1–4” of the historical SKF as untouched holdout (already exposed for several recipes).
- Not public LB verification (NO_SUBMIT remains).

## Purpose
Provide a **fresh sealed evaluator** for recipes developed after CROSS_FOLD_FAIL, so that `CROSS_FOLD_VERIFIED` / `INDEPENDENT_HOLDOUT_VERIFIED` mean something beyond matched Fold-0.

## Candidate sealed designs (pick one after Astra-high)

### A. Time-aware / entity-blocked holdout (preferred if keys exist)
If merchant/card/entity IDs exist with leakage risk, hold out entire entities never used in any prior research fit. Size for resolving ΔAUC ≈ 8e-5 at 95% paired power using F0 variance.

### B. Frozen secondary seed panel (diagnostic only)
A second SKF seed (e.g. rs=2027) used **once** after recipe freeze. Label as **transport diagnostic**, not independent holdout, because rows overlap the historical training universe.

### C. Competition public LB (only after SUBMISSION_READY)
Requires explicit user arm of SUBMIT. Counts as `PUBLIC_LB_VERIFIED` iff COMPLETE publicScore > 0.95726.

## Freeze-before-touch rules
1. Freeze baseline, candidate, preprocessing, hyperparams, ranking batch, coefficient(s), bootstrap unit.
2. Write SHA256 of code + arrays + protocol to `artifacts/research_loop/SEALED_LOCK.json`.
3. Score both arms once on identical sealed rows.
4. Require ΔAUC ≥ +8e-5 and paired 95% LB > 0 (within-fold / within-class as applicable).
5. Miss ⇒ no retune-and-rescore on same sealed cohort.

## Isolation requirements for any arm compared on sealed data
All of: feature prep, selection, base-model fits, stacking, and hyperparameter / track selection must exclude sealed rows (and their entities if blocked).

## Public champion
Preserve actual public pin **0.95726**. Local `z_sovereign_v0` remains the OOF research baseline, not the public claim.


## Decision (Astra-high 2026-11-26)

**Use B once** as frozen transport diagnostic. **Reserve A** for true independent verification when unexposed cohort+keys exist. **C** only after authorized submit with publicScore > 0.95726.
