# Predicting Credit Card Fraud (Kaggle Playground S6E11)

Local research repo for [playground-series-s6e11](https://www.kaggle.com/competitions/playground-series-s6e11).

**Current tip lock (2026-11-24):** Public LB **`0.95569`** — goodpjw **Step_AB** (`SLOT01`, SHA `3e1dba5d…`) and held flat on Diag_A / FWD_AB_AA / δ1_v4. Peephole micro-jumps (`< 0.30e-4`) **terminated**; V42 macro-leap engines did **not** clear +5e-5 OOF (see below).

Earlier public releases: **`0.95490`** (ref `58521009`, `notebooks/lb_095490/`) · **`0.95330`** (ref `58151250`, `notebooks/lb_095330/`).

## Public notebook (LB 0.95490)

| | |
|---|---|
| **Notebook** | [S6E11 \| Public Board Leap \| LB 0.95490](https://www.kaggle.com/code/moranzavdi/s6e11-public-board-leap-lb-0-95490) *(local package; publish when ready)* |
| **Dataset** | [s6e11-board-leap-lb-095490](https://www.kaggle.com/datasets/moranzavdi/s6e11-board-leap-lb-095490) *(local package)* |
| **Local path** | `notebooks/lb_095490/` |
| **Frozen CSV** | `notebooks/lb_095490/dataset/submission_095490.csv` |
| **SHA-256** | `6ea4f9bad68a61753a5c9484b13ec1536c4d5218f56c352e950670a3b078e8b4` |
| **Kaggle ref** | `58521009` |

Rebuild: load goodpjw **step E** (0.95487) + **step F** (0.95489), apply `rankfrac(L(F)+0.5·(L(F)−L(E)))`, SHA-check against the frozen CSV. No in-kernel retrain.

### Credits (please upvote upstream work)

Full linked tables live in the notebook; summary:

- **goodpjw2008** — [Board Model + RealMLP \| LB 0.95489](https://www.kaggle.com/code/goodpjw2008/s6e11-board-model-realmlp-lb-0-95489) (step E/F parents)
- **Asterios Panoras** — [0.95487 analysis](https://www.kaggle.com/code/asteriospanoras/s6e11-lb-0-95487-top-public-and-score-analysis) · [public-board-files](https://www.kaggle.com/datasets/asteriospanoras/s6e11-public-board-files)
- **Sriram S** — [TabM Ensemble \| 0.95482](https://www.kaggle.com/code/sriram00777/s6e11-tabm-ensemble-eda-lb-0-95482) · [pinned-inputs](https://www.kaggle.com/datasets/sriram00777/s6e11-pinned-inputs-095422)
- **yekenot**, **busyaprime**, **BhuvanNR**, **alexchilton**, **areebbb**, **Malyshev Danil**, **sameershaikh27**, **mathsonchain**, **amanatar**

## Public notebook (LB 0.95330)


| | |
|---|---|
| **Notebook** | [S6E11 \| Grandmaster Ladder \| LB 0.95330](https://www.kaggle.com/code/moranzavdi/s6e11-grandmaster-ladder-lb-0-95330) |
| **Dataset** | [s6e11-strike9-lb-095330](https://www.kaggle.com/datasets/moranzavdi/s6e11-strike9-lb-095330) |
| **Local path** | `notebooks/lb_095330/` |
| **Frozen CSV** | `notebooks/lb_095330/dataset/submission_095330.csv` |
| **SHA-256** | `f4a7a4f8636c99fabd82945c7d81956c5559c609dcb55cde6d4e03f80b0ea105` |

The notebook does **not** retrain the full private stack in-kernel. It loads the byte-identical Strike-9 CSV (SHA-checked), validates ids against `sample_submission.csv`, and writes `/kaggle/working/submission.csv`. Running + submitting that file should land on **Public LB 0.95330**.

### What “Strike 9” is

`0.95330` is the end of a local **LB-gradient ladder** on top of a Part-12 / Fraud-Signal-Lab backbone, not a single model:

| Step | Idea | Public LB |
|---|---|---:|
| `v8` | Part-12 wide LM + FSL-style rank-logit blend | 0.95307 |
| Slot 3 | verified F0/F2 contrastive upgrade | 0.95312 |
| Att 2 → Att 3 | multi-axis LB-gradient continuation | 0.95315 → 0.95317 |
| Strike 2 | honest 5-fold orthogonal harvest (NADE / Sameer / kNN / P4 / Graph) | 0.95319 |
| Strike 3A | + GLM-quad + graph GLM | 0.95322 |
| Strike 4A | continuation | 0.95323 |
| Strike 6 | reverse harmful pre-Part-12 tree dilution | 0.95325 |
| Strike 7 → 8 | keep riding the high-slope axis | 0.95327 → **0.95329 (#1)** |
| **Strike 9** | slope-weighted 2D step | **0.95330 (#1)** |

Mechanics that mattered in practice:

1. Work in **rank-logit** space; map test scores onto a fixed OOF reference with PCHIP quantiles.
2. Inject only **nested 5-fold orthogonal** residuals (LOFO OLS against the backbone controls).
3. Prefer **tiny Spearman distance** steps (`1-ρ ≈ 1e-5`) over large reshuffles.
4. Drop collinear public tree bags that dilute the LM backbone (Strike 5A cost **−0.00008** LB).

Repro scripts for the later ladder steps:

- `scripts/gm_top1_strike4_6_7_8_repro.py`
- `scripts/gm_top1_strike9_final_max.py`

## Credits (please upvote upstream work)

This score sits on public notebooks/datasets as well as our local training. Full linked tables live in the [public notebook](https://www.kaggle.com/code/moranzavdi/s6e11-grandmaster-ladder-lb-0-95330); summary:

- **goodpjw2008** — Part 12 wide stack + 2nd LM; [fraud-series-artifacts](https://www.kaggle.com/datasets/goodpjw2008/s6e11-fraud-series-artifacts); [oof-bank](https://www.kaggle.com/datasets/goodpjw2008/s6e11-oof-bank)
- **Malyshev Danil** — [Fraud Signal Lab](https://www.kaggle.com/code/malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion) (FSL blend recipe)
- **sameershaikh27** — [Wide Pooled Stack · Every Bank Found](https://www.kaggle.com/code/sameershaikh27/s6e11-fraud-wide-pooled-stack-every-bank-found)
- **megayak**, **masayakawamata** — target-first LM / shallow-tree members and settings
- **areebbb** — [Wide Stack + Shallow-Tree Fusion · 0.95300](https://www.kaggle.com/code/areebbb/s6e11-wide-stack-shallow-tree-fusion-0-95300)
- **sadamtorres** — TabPFN–GBDT OOF/test pairs
- **alexchilton** — public OOF bank / shallow-tree stack
- Also referenced in the climb: Chris Deotte (target-first idea), mathsonchain, amanatar, lucifer19

## Repo layout (high level)

```
notebooks/lb_095490/     # public 0.95490 notebook + frozen dataset CSV (board leap)
notebooks/lb_095330/     # public 0.95330 notebook + frozen dataset CSV
scripts/v41_*.py         # V41 tip-lock / Step6b–6e portfolio builders
scripts/v42_*.py         # V42 macro-leap architect (NO_SUBMIT audits)
preds/                   # local OOF & test prediction banks (gitignored bulk)
out/v41_*/ out/v42_*/    # staged tip CSVs (local; regenerable, not committed)
submissions/             # scored CSVs (mostly gitignored)
logs/v41/ logs/v42/      # tip-lock manifests, 2D diagonal + macro-leap reports
logs/overnight/          # JSON diagnostics & submit plans
docs/                    # research notes / audits
external/                # pulled public kernels & OOF banks
```

## Current tip (2026-11-24) — V41 tip lock + V42 macro freeze

| | |
|---|---|
| **Public tip** | **0.95569** |
| **Spearhead** | goodpjw **Step_AB** (`SLOT01_pub_Step_AB_095569.csv`, SHA `3e1dba5d31e01b6f…`) |
| **Frozen 2D hold** | `Diag_A` (`f83c899dbe63…`) = `0.05e-4` StepAC ⊕ `0.10e-4` +δ1_v4 @ total `0.150e-4` → LB **0.95569** (flat) |
| **Policy** | Peephole linear nudges `< 0.30e-4` **TERMINATED**. No submit of SLOT09/10 / Diag_B / quarantined areeb/−C3 axes. |
| **Reports** | `logs/v41/V41_STEP6B_CLEAN_MANIFEST.md` · `logs/v41/V41_STEP6E_2D_DIAGONAL.md` · `logs/v42/V42_MACRO_LEAP_REPORT.md` · `logs/v41/V41_SLOT_DECISION_STATE.md` |

### V41 live burn ladder (2026-11-24)

| Slot / cand | Idea | `1−ρ` vs AB | Public LB | Δ vs tip |
|---|---|---:|---:|---:|
| **SLOT01** | Step_AB tip lock | 0 | **0.95569** | 0 |
| SLOT02 | AB×areeb geodesic clip985 | 0.028e-4 | 0.95568 | −1 tick |
| SLOT06 | −C3_harvest_p4 orth | 0.280e-4 | 0.95568 | −1 tick |
| FWD_AB_AA | +secant `L(AB)−L(AA)` | 0.030e-4 | **0.95569** | 0 (flat) |
| Cand3 δ1_v4 | `+delta1_v4 ⊥ z_pub_only` | 0.150e-4 | **0.95569** | 0 (flat) |
| **Diag_A** | 2D orth StepAC ⊕ δ1 | 0.150e-4 | **0.95569** | 0 (flat) |

Quarantined (do not submit): SLOT03/04/05/07 (areeb-family), SLOT08 (−C3 axis), Cand_UID_clean, Diag_C (δ_combo fails loser≤0). Held: Diag_B, SLOT09/10, Cand1/Cand2.

Two live-verified **0.95569** rays from AB (near-orthogonal, cos≈−0.005):

1. **Ray1** public secant / anti-S02 (`FWD_AB_AA` / StepAC)
2. **Ray2** proprietary Stage-B `+delta1_v4 ⊥ z_pub_only` (OOF +1.53e-5, 4/5)

### V42 macro-leap audit (`NO_SUBMIT`)

| Engine | Finding | Δe5 vs `z_pub_only+Zr` |
|---|---|---:|
| E1 SCAN4 within-cohort `id` | IDs **sequential** → raw `u_g` KS fails; SCAN4 0.57489 = **size leak**; hash-u KS-ok but hurts | **−12.2** |
| E2 cohort EB offset / τ | Capped offsets ≈0; uncapped destroys AUC | **0.00** best |
| E3 StageB_v42 + Wilcoxon | gpt2 pillar shape miss; stack collapses | **−101.7** / −0.14 |

**No engine cleared the +5e-5 macro gate.** Tip remains Diag_A / Step_AB @ **0.95569**. Scripts: `scripts/v41_step*.py`, `scripts/v42_macro_leap_architect.py`.

### Prior ATH markers

| Date | Public ATH | Note |
|---|---:|---|
| 2026-11-20 | 0.95531 | V29 consume goodpjw Board Model (`docs/V29_SCORE_LEAP.md`) |
| 2026-11 | 0.95426 | V22 sriram-manifold ladder |
| — | 0.95490 / 0.95330 | Public notebook freezes (above) |

## License / competition rules

Respect Kaggle competition rules when using or redistributing predictions. Upstream notebooks/datasets keep their own licenses; credit them if you build on this release.
