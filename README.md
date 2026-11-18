# Predicting Credit Card Fraud (Kaggle Playground S6E11)

Local research repo for [playground-series-s6e11](https://www.kaggle.com/competitions/playground-series-s6e11).

Public highlight: frozen submission **Public LB `0.95490`** (Kaggle ref `58521009`) — BOLD extrapolate on **goodpjw2008** Board Model steps E/F — published under `notebooks/lb_095490/` with full upstream credits. Earlier open release: **Public LB `0.95330`** (ref `58151250`) under `notebooks/lb_095330/`.

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
scripts/                 # local research / ladder / retake builders
preds/                   # local OOF & test prediction banks (gitignored bulk)
submissions/             # scored CSVs (mostly gitignored)
logs/overnight/          # JSON diagnostics & submit plans
docs/                    # research notes / audits
external/                # pulled public kernels & OOF banks
```

## Current local ATH (2026-11-18)

| | |
|---|---|
| **Public ATH** | **0.95490** |
| **Recipe** | V26b BOLD F←E @0.5 on goodpjw Board Model steps (E≈0.95487, F≈0.95489) |
| **CSV** | `submissions/sub_v26b_ATH_PUB490.csv` (= `notebooks/lb_095490/dataset/submission_095490.csv`) |
| **Kaggle ref** | `58521009` |
| **Public notebook** | `notebooks/lb_095490/` |
| **Day reports** | `docs/V26_SCORE_LEAP.md` · `docs/V25_NEW_SCORE_CHANNELS.md` · `docs/NOV17_LATEST_RESEARCH_SCORE_LEVER_REPORT.md` |

### V22 live ladder (sriram `0.95422` → ATH `0.95426`)

| Step | Idea | Public LB |
|---|---|---:|
| V22.1 C1 | ATH lineage + ΔRealMLP | 0.95407 |
| V22.1 C2pure | sriram + proprietary Slot9 engines | 0.95416 |
| V22.2h P2 | micro-geodesic sriram × C1 + 5F LM | 0.95425 |
| **V22.3 R2** | P2 continuation + de-diluted dMLP @ `1.25e-4` | **0.95426** |
| V22.5 K2 | Ray-1 crest + signed orth4 @ `2.05e-4` | 0.95426 (tie) |
| V22.6 SLOT1–5 | unattended 5-slot ladder (twilight orth hurt) | 0.95426…0.95412 |

Key scripts: `scripts/gm_v22_{1,2_hardened,3,4,5,6}_*.py`.  
Next reset: prefer **Ray-1 crest** in `[1.45e-4, 1.85e-4]` and **`−U_orth` in twilight** (`V22_6_TWI_SIGN=-1`); do not GS-project after quantile gating.

## Local next steps (beyond 0.95330)

Historical SoftPseudo retake artifacts (pre-V22) remain under `submissions/sub_retake_R*.csv` / `scripts/gm_day2_*`. The active climb is the **sriram-manifold V22 ladder** documented above.

## License / competition rules

Respect Kaggle competition rules when using or redistributing predictions. Upstream notebooks/datasets keep their own licenses; credit them if you build on this release.
