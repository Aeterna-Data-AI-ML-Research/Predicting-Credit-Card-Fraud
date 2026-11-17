# V24.1 Professor Lever Study (methods memo)

**ATH:** 0.95440 · **Quota today:** exhausted (hypotheses staged, not live-scored)

---

## Research question

Public AUC depends only on the **ordering** of test scores. Given an identified generative model for the ATH file, which **factors** are necessary, which are continuous knobs, and what **unused orthogonal signal** remains in the public OOF bank?

## Generative model (identified)

\[
s^\star = U\!\left(\tfrac12\big(U(L(G)+w\Delta_{v15})+U(L(B)+w\Delta_{v15})\big)\right),\quad w=0.30
\]

- \(G\): goodpjw step11 · \(B\): Bhuvan · \(\Delta_{v15}=L(N_{15})-L(C)\) (PyTabKit API15 − PyTorch v28)
- Reconstruction spearman to live ATH: **0.99999996** (recipe locked)

## Factorial ablation (identifiability)

| Factor | Finding |
|--------|---------|
| **Neural version** | First-order. Mean \(\rho\)(ATH): v15 **1.42** vs v5 **4.10** (×e-4). Old Δmlp is the wrong axis. |
| **Base** | First-order. dual(G,B) **1.80** ≪ G-only / B-only ~2.6 ≪ dual(R2,gpw32) **4.00**. Dual of the two public near-tops is necessary. |
| **Weight \(w\)** | Continuous knob. Exact match at 0.30; neighbors 0.22/0.35/0.40 are smooth displacements (0.17–0.68 e-4). |
| **CSV roundtrip** | **Null effect** (rt True ≡ False on ρ). Drop from future design. |

**Implication:** Do not hunt new orth/Ray-1 on the old manifold. Allocate board slots to **(i)** \(w\)-grid on dual(G,B)+v15 and **(ii)** residuals that survive nested+MC gates against the ATH surrogate.

## Nested residual screen (70 bank members)

ATH OOF surrogate AUC ≈ 0.9497 (imperfect; interpret ranks, not absolute Δ).

**Gate:** fold pos≥4, bootstrap CI on mean ΔAUC\({\times}10^5\) excludes 0, MC private-up rate ≥0.55, \(t\ge 2.5\).

Top gated residuals (examples):

| member | nested | t | meanΔ | MC gap |
|--------|--------|--:|------:|-------:|
| `realmlp_v1` | 5/5 | 6.24 | +1.36 | +0.37 |
| `gb_bag5` | 5/5 | 4.85 | +0.20 | +0.03 |
| `realmlp_dig_s100` | 5/5 | 4.70 | +2.19 | −0.28 |
| `lgb_diglead_l3` | 5/5 | 3.66 | +2.32 | −0.01 |

MC gap > 0 ⇒ publicly optimistic; prefer small/negative gap when choosing board files.

## Falsifiable hypotheses (staged)

| ID | Claim | ρ(ATH) e-4 | Prior | Kill if |
|----|-------|----------:|-------|---------|
| **H1** | \(w=0.35\) dual_GB v15 beats ATH | 0.171 | med | ≤0.95439 |
| **H2** | \(w=0.40\) still lifts | 0.683 | low-med | ≤0.95439 |
| **H3** | half-extrapolate w30→w40 | 0.171 | med | ≤ATH |
| **H4** | dual(R2,gpw32)@0.35 | 2.809 | med | ≤0.95439 |
| **H5** | dual(ATH,B)@0.30 Bhuvan diversity | 1.640 | med | ≤ATH |
| **H6** | bold ATH away from dual_live | 0.494 | low | ≤ATH |
| **H7** | ATH+ORTH(`realmlp_v1`)@0.30e-4 | 0.300 | med-high | ≤0.95439 |

Submit priority after reset: **H1 → H3 → H5 → H7 → H4**.

## Design rules (professor constraints)

1. One primary contrast per submit; pre-register kill line.
2. Do not declare discovery on a single +1e-5; require a second confirming contrast.
3. Exclude exhausted families a priori: Ray-1 crest, ±twilight orth, micro-GS orth on old R2.
4. Prefer hyps with MC priv_pos ≥ 0.55 and CI_lo > 0 on nested Δ.

## Artifacts

- Report: `logs/overnight/GM_V24_1_PROFESSOR_LEVER_STUDY.md`
- Meta: `/tmp/s6e11_v16/out/v24_1_professor_study_meta.json`
- Ablation table: `/tmp/s6e11_v16/out/v24_1_factorial_ablation.csv`
- Staged CSVs: `submissions/sub_v24_1_H{1..7}.csv`
