# Sovereign Fold-0 review

## Hypotheses ranked by expected AUC lift per CPU cost

1. Conditional pooling of self-trained permutation logits. Fit order contrasts, dispersion, and risk-regime interactions with regularized logistic regression. This directly tests the fixed pooling assumptions in the champion while keeping supervised combiner fits inside Fold-0.
2. Add a fresh outer-train tabular anchor. Its fraud ranking can supply signal absent from both language models and gives the combiner a second model family without depending on the lineage of stored super-stacks.
3. Add low-degree merchant-risk and velocity splines to the cross-family combiner. These let the effect of the raw risk variables change around known risk ranges.
4. Rebuild the strongest super-stack inputs inside the outer training split. Preserve the original F0 holdout through every upstream model fit, track choice, and calibration before testing a new stack. This has a higher CPU cost and should begin with the strongest distinct pillar.

## Evaluation contract

Matched saved `z_sovereign_v0` on 139,992 rows using the verified SKF rs=42 Fold-0 indices. Each candidate score comes from a logistic combiner fit on the other four SKF rs=7 subfolds inside Fold-0. Rank transforms pool unlabeled Fold-0 scores. Clipping and scaling parameters for learned features come from each combiner training split. The champion scores serve as the reference only. Several legacy component builders fit on other global OOF rows or selected tracks across all folds, so the candidate uses the Fold-0 slices from the self-trained L7 and GPT-2 tensors.

An independent reread reproduced both AUCs and all five subfold deltas for each saved candidate. The consolidated L7 slice matched the original Fold-0 tensor element for element.

Promotion requires pooled ΔAUC ≥ +8e-5 and at least four positive subfolds.

| Experiment | Base AUC | Candidate AUC | ΔAUC × 10⁵ | Positive | Verdict |
|---|---:|---:|---:|---:|---|
| Conditional LM pooling | 0.9512507991341679 | 0.9469439662423098 | −430.683289185807 | 0/5 | NO-GO |
| Conditional LM pooling + outer-train LightGBM | 0.9512507991341679 | 0.9471946036884362 | −405.61954457316585 | 0/5 | NO-GO |

Conditional LM subfold deltas × 10⁵ were −260.43679743, −384.24439645, −471.23052332, −571.74404216, and −390.83059727. The stand-alone pool did not recover the champion's ranking.

The LightGBM learner used the 559,968 outer training rows and fixed boosting rounds. Its solo Fold-0 AUC was 0.943526723763. The combined model improved on the first candidate by +25.06374461e-5 and remained −405.61954457e-5 below the champion. Its subfold deltas × 10⁵ were −281.16711946, −368.57441510, −448.02133503, −489.74187323, and −375.12706136.

**VERIFIED_LIFT=NO.** Both runs are NO-GO. The next spline check has low expected lift given the remaining gap.

Artifacts are `scripts/v47_sovereign/conditional_f0.py`, `logs/v47/conditional_f0.json`, and `preds/v47_sovereign/conditional_f0_oof.npy`. The second run wrote `logs/v47/conditional_f0_tabular.json` and `preds/v47_sovereign/conditional_f0_tabular_oof.npy`. Both arrays are in canonical Fold-0 row order.

NO_SUBMIT=True · SUBMIT_ARMED=False · RunPod $0 · cloud $0 · public competitor tip CSVs read 0.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/v47_sovereign/conditional_f0.py --tabular --regime-splines
```

## Round 2 and champion-anchored digit LightGBM

Both used the matched base AUC 0.9512507991341679.
- Splines scored 0.9472101119386994, Δe-5 −404.0687195468462, 0/5, NO-GO. Subfold Δe-5 [−263.06786414, −382.58348708, −398.34402753, −491.39856126, −419.78763748].
- Fixed 95:5 rank-logit digit LightGBM blend scored 0.9511974636845753, Δe-5 −5.333544959262859, 2/5, NO-GO. Subfold Δe-5 [+9.21261409, −13.78399548, +4.14560903, −17.65578332, −8.96789346].
- VERIFIED_LIFT=NO. NO_SUBMIT=True, SUBMIT_ARMED=False, RunPod $0, cloud $0, public tip CSVs read 0.

## Champion-anchored CatBoost

Fixed 95:5 rank-logit CatBoost blend — base 0.9512507991341679, candidate 0.9511643271947364, Δe-5 -8.647193943145, 0/5, NO-GO. Solo 0.9435214983935254. Subfold Δe-5 [-3.8728680986643838, -16.663422460538868, -10.406800914430914, -4.386723255411784, -6.261191880496675].

## Round4 P_super / unused OOF audit (fixed 95:5 rank-logit vs champion)
Audited 30 streams. VERIFIED_LIFT=NO.
Best: `P_wide_v5_rect` solo=0.951971 Δe-5=+5.134291 pos=5/5 NO-GO

| stream | solo | ρ | Δe-5 | pos/5 | verdict |
|---|---:|---:|---:|---:|---|
| P_wide_v5_rect | 0.951971 | 0.9966 | +5.1343 | 5/5 | NO-GO |
| P_super_v8_part12_rect | 0.951870 | 0.9959 | +5.0426 | 5/5 | NO-GO |
| P_super_v6_grandmaster | 0.951937 | 0.9970 | +4.7056 | 5/5 | NO-GO |
| P_super_v7_cpu | 0.951937 | 0.9970 | +4.7056 | 5/5 | NO-GO |
| P_super_v9_frontier_rect | 0.951838 | 0.9963 | +4.6301 | 5/5 | NO-GO |
| P_restack_own | 0.951738 | 0.9960 | +4.4899 | 5/5 | NO-GO |
| P_restack_pub | 0.951598 | 0.9957 | +3.7249 | 5/5 | NO-GO |
| P_super_v7_regime_only | 0.951425 | 0.9960 | +2.9151 | 5/5 | NO-GO |
| P_super_v7 | 0.951496 | 0.9969 | +2.8067 | 5/5 | NO-GO |
| P_super_v5p5_rect | 0.951580 | 0.9982 | +2.3040 | 5/5 | NO-GO |
| P_super_v5_rect | 0.951578 | 0.9981 | +2.2636 | 5/5 | NO-GO |
| P_super_v5p5_clean | 0.951513 | 0.9983 | +1.8688 | 5/5 | NO-GO |

## Round 5 — Cross-fitted signed CatBoost residual

GO on the matched Fold-0 gate. Base AUC 0.9512507991341679, candidate AUC 0.9513363354184922, Δe-5 +8.55362843242835, 4/5 positive.
Subfold Δe-5 [-0.054328810994430654, 22.50765027270729, 16.481323720174412, 1.0733471795032479, 5.942298877881225].
On each SKF rs=7 scoring subfold, the other four subfolds selected the signed weight by AUC from [0, .025, -.025, .05, -.05, .1, -.1]. All five selected -.1. Candidate = 1.10 × rank-logit(champion) − 0.10 × rank-logit(CatBoost).
The five Round 4 leaders inherit public prediction arrays and were excluded. This matched result inherits the saved champion's upstream OOF provenance and has no independent holdout confirmation. NO_SUBMIT=True, SUBMIT_ARMED=False, RunPod $0, cloud $0, tip CSVs read 0.


## Round6 adversarial protocol

### What could make the local pass a false positive

- Fold-0 was reused, and the margin is thin (+8.55e-5 clears by only 0.55e-5).
- Five nested weight choices share overlapping training sets; all hit grid endpoint −0.10.
- Champion upstream OOF provenance remains the main isolation risk; CB fold isolation alone is insufficient.
- Prefer within-fold AUC deltas over pooled cross-fold rank comparisons; row bootstrap is conditional on frozen predictions.

### CROSS_FOLD_VERIFIED (strict)

On folds 1–4 for the one frozen recipe (w=−0.10, no retune):

1. Provenance: SKF rs=42 alignment, CB outer-train isolation, reproduce F0 within tolerance, then exclude F0 from adjudication.
2. Fold-size-weighted mean within-fold ΔAUC ≥ +8e-5; ≥3/4 strictly positive; leave-one-eval-fold-out weighted mean still > 0.
3. Two-sided 95% paired interval lower bound > 0 for that weighted statistic (within-fold, within-class resample). Interval is conditional on frozen predictions.
4. If numbers pass but champion isolation unresolved → record **matched-anchor transfer evidence**; keep CROSS_FOLD_VERIFIED pending.

### INDEPENDENT_HOLDOUT_VERIFIED

Genuinely unexposed labeled cohort; freeze recipe before unblinding; ΔAUC ≥ +8e-5 and paired 95% LB > 0; no retune-after-miss. Locks unchanged.

### Dev isolation

Adjudicate frozen candidate first. Complementary residuals (e.g. signed digit LGBM) only in Fold-0/dev nested fits. Any design prompted by folds 1–4 needs a fresh sealed evaluator.

## Round6 strict adjudication

Verdict: **CROSS_FOLD_FAIL — remain at LOCAL_GATE_PASS**

Per-fold Δe-5 (w=−0.10 frozen): [8.5536, -20.1038, 4.0431, 1.1155, -5.9661]
Weighted mean Δe-5 folds1-4 = -5.2278; pos=2/4; LOO all positive=False; numeric_pass=False.
F0 sensitivity peak at w=−0.125 (+9.07e-5) but cannot change frozen eval coefficient.
Milestone remains LOCAL_GATE_PASS. CROSS_FOLD_VERIFIED not achieved.

## Round6 Codex critique

CROSS_FOLD_FAIL. Retire frozen −0.10 signed CatBoost as promotion candidate.
Champion provenance: global OOF matrix → each outer fold’s 559,968 meta-train rows can include scored-fold labels via base-model fits; P4 tracks selected across all 5 folds. Saved champion cannot certify strict isolation.
Ranked next: (1) isolated champion diagnostic (2) Fold-0-only complementary signed digit LGBM (3) drop signed-CB family near-term.
CROSS_FOLD_VERIFIED criteria unchanged and unmet.

## Round7 F0 signed digit LGBM (dev only)

Δe-5=-0.8489 pos=2/5 weights=[-0.025, -0.025, -0.05, -0.05, -0.05] NO-GO. Solo=0.888207.

## Round7 corrected digit LightGBM recipe

Round7 used a different feature recipe. Its numeric-only input omitted the native categorical columns and five categorical cents features used in the earlier 0.9368051080303192 solo run. The original Round7 result remains NO-GO (−0.848928e-5, 2/5). This rerun restores the earlier recipe and scores Fold-0 only. NO_SUBMIT=True, SUBMIT_ARMED=False, RunPod $0, tip CSVs read 0.

Corrected Fold-0 dev result — base 0.9512507991341679, solo 0.9368051080303192, candidate 0.9512048308028603, Δe-5 -4.596833130754, 0/5, NO-GO. Weights [-0.05, 0.0, -0.025, 0.0, 0.0]. Subfold Δe-5 [-13.57444149207021, 0.0, -3.297820319669853, 0.0, 0.0]. Milestone remains LOCAL_GATE_PASS.

## Isolated champion Fold-0 diagnostic

Corrected Round7b reproduced solo AUC 0.9368051080303192. The signed blend scored 0.9512048308028603, Δe-5 −4.5968, 0/5, NO-GO.

The isolated two-stream stack scored 0.9446644275289812 against saved champion 0.9512507991341679, Δe-5 -658.63716052. Isolated LR 0.9413067436937739, isolated LightGBM 0.9440808429637034. All fitting used the outer training partition. Only historical Fold-0 was scored. This is a smaller-stack diagnostic. Milestone remains LOCAL_GATE_PASS. NO_SUBMIT=True, SUBMIT_ARMED=False, RunPod $0, tip CSVs read 0.

## Round8 isolated champion F0 diagnostic (COMPLETE)

Two-stream isolated stack (one-hot LR + native-cat LGBM), all fits inside Fold-0 train, inner SKF(3,rs=2026).
- saved champion F0 AUC 0.9512507991341679
- isolated stack 0.9446644275289812 · Δe-5 **−658.64** vs saved champion
- isolated LR 0.9413067 · isolated LGBM 0.9440808
- Artifact: `logs/v47/codex_loop/ISOLATED_CHAMPION_F0_DIAGNOSTIC.json`
- Interprets as: isolation procedure works; small self-owned stack is far below historical 14-stream champion (expected). Not a public LB claim. Signed-CB remains retired.

## Astra-high provenance strategy

**The public score gap is unexplained.** The supplied 0.95726 public score and the saved sovereign stack’s ~0.952 local OOF score come from different recipes and evaluation rows. The gap could reflect stronger features or ensembling, a different mix of easy and hard fraud–legitimate pairs, sampling variation, or leaderboard selection across attempts. There is no matched evidence to assign the gap to any of those causes. Keep **0.95726 as the public target** and use paired local gains to select candidates. A local AUC plus an assumed leaderboard offset cannot establish a public win.

**Round 8 establishes the CPU development anchor at 0.9446644275289812.** It beats its isolated raw LightGBM component by **+58.36e-5**. Its **−658.64e-5** gap to the saved Fold-0 champion combines differences in stream coverage, training size, stacking, and provenance. The experiment cannot estimate how much the champion’s global OOF reuse or P4 selection inflated its score. The research priority is to recover missing signal with reproducible streams before spending more CPU on small blend-weight gains.

### Ranked P2 experiments

Use the frozen Round 8 stack as the paired control. Run one prespecified candidate per hypothesis on historical Fold-0. Record solo AUC, paired ΔAUC, and the five existing dev subfold deltas. Use **+8e-5 and at least 4/5 positive** as a dev advance criterion. Historical Fold-0 has already informed research choices, so even a pass only earns a place in the next frozen evaluation.

| Rank | Experiment | Why it could generalize |
|---|---|---|
| 1 | **Nonlinear additive logistic stream.** Fit splines on numeric columns, one-hot encode native categories, and test a fixed 15% rank-logit blend with the isolated anchor. | Recovers smooth univariate effects the raw LR misses and gives the stack a regularized alternative to tree splits. One CPU fit makes the incremental test cheap. |
| 2 | **Smoothed interaction model.** Use a small fixed set of crosses such as `auth_method × channel`, `merchant_category × card_type`, and a coarse `velocity_score × merchant_risk_score` grid. Fit regularized one-hot effects and fall back to main effects for unseen combinations. | Tests whether a few repeatable fraud regimes add signal beyond the additive model. Sparse combinations can otherwise create unstable gains. |
| 3 | **CPU histogram booster on structured features.** Fit a fixed XGBoost histogram recipe with raw fields and train-fitted coarse bins or frequency features. Compare it against the same-feature LightGBM control before adding either stream to the stack. | Separates feature gain from learner diversity. The corrected digit residual’s failure lowers the priority of another cents-only experiment. |
| 4 | **Rebuild one auditable P4 feature family.** Extract one deterministic feature generator, rebuild its supervised dependencies inside each training partition, and freeze a single learner before historical Fold-0 scoring. | This is the closest CPU path to recovering signal from the stronger legacy stack. It earns its higher cost only if its source lineage and incremental value can be demonstrated. |

### Expanding the isolated anchor

The CPU-rebuildable core is **raw one-hot LR, raw native-categorical LightGBM, spline and interaction LR, and a small CPU histogram booster**. A P4 feature family can join once its inputs, base margins, and track choice can all be regenerated inside the split. Saved P4 tracks, super-stack scores, and LM or LLR arrays stay outside this build unless their complete outer-isolated lineage can be reproduced under the CPU lock.

For each outer split, generate fresh three-fold inner OOF scores. Fit vocabularies, bins, frequencies, supervised encoders, and base models inside the applicable training partitions. Fit rank references, scaling, and the regularized meta-model on the inner OOF design. Score the outer rows with the inner-model ensemble using the same reference mapping as Round 8. Record code and data hashes, row IDs, split indices, feature lists, seeds, and score hashes. If a stream has label-dependent features or track selection, apply an additional training-only split wherever those choices would otherwise see the rows they score.

### Sealed evaluation decision

**Use B once as a frozen transport diagnostic. Reserve A for independent verification.** A second SKF seed can test split sensitivity, but every row comes from the examined development universe. The training schema has a row `id`, `merchant_category`, and transaction attributes. It exposes no evident card or merchant entity identifier or transaction timestamp. A requires a genuinely unexposed, relevant labeled cohort with documented lineage and reliable grouping or time keys where applicable. If such a cohort is unavailable, independent verification stays pending. C requires separate authorization after submission readiness and a completed public score **strictly above 0.95726**.

Before touching either evaluator, freeze:

- **Recipes and artifacts.** One baseline and one candidate, exact preprocessing, feature sources, fitting and calibration rules, seeds, blend formula, inference ranking batch, and CPU environment. Record hashes in `artifacts/research_loop/SEALED_LOCK.json`.
- **Cohort and evaluation design.** Lock row IDs or the split generator, duplicate and entity policy, and scoring code. For A, size the cohort using a stated power target and development estimates of paired variance and positive counts. A 95% confidence level is distinct from statistical power.
- **Decision rules.** For B, apply the weighted within-fold **+8e-5**, **≥3/4 positive**, positive leave-one-fold-out means, and paired 95% lower bound above zero as transport diagnostics. For A, require paired **ΔAUC ≥ +8e-5** and a two-sided paired 95% lower bound above zero, resampling intact independent groups where applicable. Score both arms once, release the prespecified metrics, and require a fresh evaluator for a revised recipe.

The workspace is read-only. This section is ready for Cursor to append to `logs/v47/CODEX_DS_REVIEW_NEXT.md`. The command runs P2 experiment 1 on historical Fold-0, verifies the saved anchor’s identity, and appends the measured result. The spline model uses only outer-training rows. The blend weight is fixed before scoring.


## P2 additive spline Fold-0 development result

Isolated anchor 0.9446644275289812. Spline solo 0.9420998870266402. Fixed 85:15 blend 0.9445494165545395, Δe-5 -11.50109744, 1/5 positive, NO-GO. Subfold Δe-5 [-1.7385219520993367, -1.7850895044024817, 0.34222663695127054, -41.76720546359425, -8.76566863267092]. Historical Fold-0 only.

## P2 batch vs isolated anchor (F0 DEV)

- interaction_lr_fixed15: Δe-5=-31.638509120923697 pos=0/5 NO-GO solo=0.9405163047461014
- hist_xgb_fixed15: Δe-5=20.975771086673145 pos=5/5 DEV_ADVANCE solo=0.9448430754846596

## P2 hist-XGB inner-OOF integration

Historical Fold-0 only. Saved control reproduced True. Matched control 0.9446644275289812, candidate 0.9449195163309662, Δe-5 +25.50888020, 4/5 positive, DEV_ADVANCE. Subfold Δe-5 [16.72551253024679, 38.76748728006696, 65.80084883095472, -0.015555756216478756, 11.036809041509166].

## P2 isolated P4 L4.3 step-grid stream

Historical Fold-0. Matched control 0.9449195163309662, candidate 0.9449231812391763, Δe-5 +0.36649082, 3/5 positive, NO-GO. Subfold Δe-5 [-2.8716657244576282, 3.469282645474081, 14.101293017776406, -22.36917745128686, 10.819028454356339].

## Frozen B transport diagnostic

B_TRANSPORT_PASS. Weighted within-fold Δe-5 +39.05502189, 4/4 positive, leave-one-fold-out Δe-5 [40.52200007100455, 39.5157910477247, 38.8978761704221, 37.284420260511965], paired 95% interval [28.699695908202397, 48.532718555475626]. Historical rows under SKF seed 2027. Independent holdout verification remains pending.

## P2 cross-fitted pair-rate XGB

Historical Fold-0. Matched control 0.9449195163309662, candidate 0.9449210301649766, Δe-5 +0.15138340, 2/5 positive, NO-GO. Subfold Δe-5 [-10.617401921753089, 14.1487746369795, 38.064935482118315, -22.236953523357972, -14.241294823813533].

## P2 isolated CPU MLP

Historical Fold-0. Matched control 0.9449195163309662, candidate 0.9449708343713743, MLP solo 0.9411922362945062, Δe-5 +5.13180404, 4/5 positive, NO-GO. Subfold Δe-5 [5.836466553466035, -4.796457885702932, 16.901329138241383, 3.0567060981701033, 0.46667268673861173].

## P2 hist-XGB capacity replacement

Historical Fold-0. Matched control 0.9449195163309662, candidate 0.9452913382101480, Δe-5 +37.18218792, 3/5 positive, NO-GO. Subfold Δe-5 [39.02360881764544, -10.268145279579421, -16.177986473797645, 109.47363443210011, 64.774168920112].

## P2 isolated decimal-token likelihood

Historical Fold-0. Control 0.9449195163309662, candidate 0.9461443510359124, solo 0.9463649791318605, Δe-5 +122.48347049, 5/5 positive, DEV_ADVANCE. Subfold Δe-5 [80.86455454876517, 155.54338590190176, 178.5334141921502, 78.75101588810108, 115.21370847906721].

## Round16 note

Decimal-token DEV_ADVANCE confirmed by Cursor execution (+122.48e-5, 5/5). Absolute still ~0.946 vs public 0.95726.

## Frozen decimal-token B transport diagnostic

B_TRANSPORT_PASS. Weighted within-fold Δe-5 +124.38848643, 4/4 positive, leave-one-fold-out Δe-5 [125.50248877179668, 129.21070304465394, 120.61665261578665, 122.22410130764418], paired 95% interval [104.4049907435196, 145.909275833013]. Historical rows under SKF seed 2031. Independent holdout verification remains pending.

## P2 decimal character n-gram LR

Historical Fold-0. Control 0.9461443510359124, candidate 0.9468323528460475, lexical solo 0.8229832175646649, Δe-5 +68.80018101, 5/5 positive, DEV_ADVANCE. Subfold Δe-5 [8.226934237609562, 75.33853834387205, 97.7679278729493, 120.3782195457026, 37.48937250179773].

## Frozen decimal-token B transport diagnostic

B_TRANSPORT_PASS. Weighted within-fold Δe-5 +34.83313868, 4/4 positive, leave-one-fold-out Δe-5 [30.5239177821585, 33.88557771720387, 38.83272892640788, 36.09033031409803], paired 95% interval [18.895251891339893, 50.67845670831878]. Historical rows under SKF seed 2033. Independent holdout verification remains pending.

## P2 magnitude-conditioned decimal likelihood

Historical Fold-0. Control 0.9468323528460475, candidate 0.9468408296948083, solo 0.9463624519305245, Δe-5 +0.84768488, 2/5 positive, NO-GO. Subfold Δe-5 [-1.0167248916337357, -4.625710193972843, 13.992402724205544, -6.766753957798688, 3.1578185136327264].

## P2 decimal character context MLP

Historical Fold-0. Control 0.9468323528460475, candidate 0.9466928967162500, solo 0.9223623544483182, Δe-5 -13.94561298, 0/5 positive, NO-GO. Subfold Δe-5 [-10.904568494196631, -17.95179140716696, -15.579089859174555, -10.865695723016877, -15.252418978439763].

## P2 decimal dependency tree

Historical Fold-0. Control 0.9468323528460475, candidate 0.9467584423095654, solo 0.8465517262549191, Δe-5 -7.39105365, 1/5 positive, NO-GO. Subfold Δe-5 [9.398884303524202, -6.9928941019758994, -5.195622579090564, -4.837840185922815, -24.95921086270725].

## P2 decimal dependency tree

Historical Fold-0. Control 0.9468323528460475, candidate 0.9467584423095654, solo 0.8465517262549191, Δe-5 -7.39105365, 1/5 positive, NO-GO. Subfold Δe-5 [9.398884303524202, -6.9928941019758994, -5.195622579090564, -4.837840185922815, -24.95921086270725].

## P2 five-stream pairwise meta-model

Historical Fold-0. Control 0.9468323528460475, candidate 0.9469293656707550, Δe-5 +9.70128247, 5/5 positive, DEV_ADVANCE. Subfold Δe-5 [2.8794269831711183, 11.059793668488638, 26.763678584773576, 3.5156009068004757, 6.922311520041191].

## Frozen five-stream pairwise B transport diagnostic

B_TRANSPORT_PASS. Weighted within-fold Δe-5 +17.58198793, 4/4 positive, leave-one-fold-out Δe-5 [19.073435170750365, 16.420316247378608, 21.481811744979613, 13.352388560405883], paired 95% interval [11.207188985734444, 24.934166431320598]. Historical rows under SKF seed 2037. Independent holdout verification remains pending.

## Champion 14-stream pairwise head diagnostic

NO-GO. Folds1–4 weighted Δe-5 +4.54425331, 3/4 positive, LOO [2.444109164349377, 7.304466430010001, 5.3895103176548895, 3.0389273239728176]. Control repro max|Δ| vs saved z0 = 0.000e+00. NO_SUBMIT.

## Champion-native character n-gram stream

NO-GO. Historical folds 1–4 weighted Δe-5 -2.64421167, 1/4 positive, LOO [-0.4342804358866322, -4.017068138787809, -2.951201225901793, -3.1742968755373484]. Saved control reproduced exactly. Legacy champion provenance caveat applies. NO_SUBMIT.

## Champion-native decimal-token hist-XGB stream

NO-GO. Historical folds 1–4 weighted Δe-5 +0.09406993, 3/4 positive, LOO [-0.9829051601739636, 1.3214843981571949, -0.03448971742647801, 0.07219018931155692]. Saved control reproduced exactly. Legacy champion provenance caveat applies. NO_SUBMIT.

## Champion 14-stream hard-negative pairwise diagnostic

NO-GO. Folds1–4 weighted Δe-5 +7.36231180, 4/4 positive, LOO [7.067492035225016, 9.551890989593451, 7.406071273215649, 5.42379288960652]. Control repro max|Δ| vs saved z0 = 0.000e+00. NO_SUBMIT.

## Frozen five-stream pairwise B transport diagnostic

B_TRANSPORT_FAIL. Weighted within-fold Δe-5 -7.15677172, 0/4 positive, leave-one-fold-out Δe-5 [-4.715873314309338, -7.936985061529621, -8.396951533221989, -7.577276987643286], paired 95% interval [-12.760869192083094, -1.4981917662654234]. Historical rows under SKF seed 2039. Independent holdout verification remains pending.

## Champion 14-stream regime-interaction head diagnostic

NO-GO. Folds1–4 weighted Δe-5 -1.26726408, 1/4 positive, LOO [-0.6144969773765515, -1.9403314299837329, -1.1702682196745864, -1.343959679493227]. Control repro max|Δ| vs saved z0 = 0.000e+00. NO_SUBMIT.

## Champion harvest stream ablation

NO-GO. Folds1–4 weighted Δe-5 -5.20988932, 0/4 positive, LOO [-5.797068990494776, -6.8469857037234405, -3.298791290976914, -4.896711292146083]. Control repro max|Δ| vs saved z0 = 0.000e+00. NO_SUBMIT.

## Champion harvest stream ablation

NO-GO. Folds1–4 weighted Δe-5 -5.20988932, 0/4 positive, LOO [-5.797068990494776, -6.8469857037234405, -3.298791290976914, -4.896711292146083]. Control repro max|Δ| vs saved z0 = 0.000e+00. NO_SUBMIT.

## Champion 14-stream rank-Gaussian diagnostic

NO-GO. Folds1–4 weighted Δe-5 -3.01054840, 0/4 positive, LOO [-3.9386635859672836, -1.7429218161539122, -2.5798930068080907, -3.7807151803607533]. Control repro max|Δ| vs saved z0 = 0.000e+00. NO_SUBMIT.
