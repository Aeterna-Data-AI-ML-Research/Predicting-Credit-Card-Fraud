# V42.1 — Modern OOF Bank + Softened Stage-B

**NO_SUBMIT=True** · days left ≈ **6.4** (deadline 2026-11-30) · MPS=True CUDA=False runpodctl=True  
**ZERO SUBMISSIONS.** Script: `scripts/v42_1_modern_bank_and_stageb.py`

## Verdict
`z_modern_pub` via nested LogReg + `rankmean` blend vs locked `z_pub_only` delivers **+5.21 e-5** OOF AUC+Zr (`0.952652` vs `0.952614`), **pos 4/5**, sf25=16. Softened Stage-B δ is **flat** vs `z_modern_all` (Δe5≈0); L4.6C regime also flat. No live fire.

## 1. Inventory highlights
- Public admitted: **147** (gpw70 + Aman rank1_q17 + board-pinned RealMLP/TabM/weni + part12) · Proprietary: **8**
- gpt2 Engine-3 shape-fix: fold VA `(n_va, seeds, 2)` → full OOF `(699960,)` / test mean `(299984,)` ✅ (was wrongly `.ravel()`'d)
- soft_diff_orth_LM: isolated admit only (v10_*/zanch/idf_p3/clean035 quarantined)
- Missing / not on disk (kaggle CLI unavailable this env): named `G26_5f`, `MS_5f`, `U5_gini10_llr`, exact `E2_heavy_ridge` (proxy if present). Aman `masaya_gpt2` / `megayak_*` used as MS-family stand-ins; gpw70+board RealMLP cover RealMLP 5f.

| asset | oof | test | solo AUC | disp | KS | status |
|---|---|---|---:|---:|---:|---|
| gpt2_med_4ep_5f | [699960] | [299984] | 0.9482383414278046 | 0.9999999999979997 | 9.285978232131953e-06 | OK |
| l7_s2026_5f | [699960] | [299984] | 0.9492651606850682 | 0.9999999999979999 | 1.0338589755548577e-05 | OK |
| B_xgb_llr | [699960] | [299984] | 0.9519489439041481 | 0.9999999999979999 | 7.53873759007373e-06 | OK |
| delta1_v4 | [699960] | [299984] | 0.7618614554123042 | 0.9999999999979999 | 7.53873759007373e-06 | OK |
| soft_diff_orth_LM | [699960] | [299984] | 0.576130001602512 | 0.999999999998 | 7.53873759007373e-06 | OK |
| P4_L4.5_pure5f | [699960] | [299984] | 0.5344172721214207 | 0.9999999999979998 | 0.006145307729553906 | OK_LOCKED |
| P4_L4.3_pure5f | [699960] | [299984] | 0.5276782100230668 | 0.9999999999979999 | 0.035396151962247546 | OK_LOCKED |
| L4.6C_top12_5f | [699960] | [299984] | 0.8326874417025272 | 0.9999999999979997 | 1.3583291065016878e-05 | OK |
| U5_gini10_llr | None | None | — | — | — | MISSING |
| G26_5f | None | None | — | — | — | MISSING |
| MS_5f | None | None | — | — | — | MISSING |
| gpw70:realmlp_dig_s2 | [699960] | [299984] | 0.9486712882932222 | 0.9999999999979997 | 9.059096024888191e-06 | OK |
| gpw70:realmlp_dig_s100 | [699960] | [299984] | 0.9485764989745433 | 0.9999999999979998 | 9.293788618913723e-06 | OK |
| gpw70:realmlp_dig | [699960] | [299984] | 0.9485288055034796 | 0.9999999999979999 | 9.2974080664332e-06 | OK |
| gpw70:realmlp_te | [699960] | [299984] | 0.9477104886524447 | 0.9999999999980002 | 9.288645193450407e-06 | OK |
| gpw70:realmlp_te_s2 | [699960] | [299984] | 0.9476213940339099 | 0.9999999999980002 | 9.194539559276294e-06 | OK |
| gpw70:realmlp_te_cnt | [699960] | [299984] | 0.9476016088196754 | 0.9999999999979997 | 9.273976906309933e-06 | OK |
| gpw70:realmlp_cnt | [699960] | [299984] | 0.946643645217877 | 0.9999999999979997 | 9.277215359315694e-06 | OK |
| gpw70:realmlp_allcat | [699960] | [299984] | 0.9463971628346843 | 0.999999999998 | 9.289026187880856e-06 | OK |
| gpw70:realmlp_rawdig | [699960] | [299984] | 0.9461483965429945 | 0.9999999999979998 | 9.313790828580082e-06 | OK |
| gpw70:realmlp_v1 | [699960] | [299984] | 0.9458383039303639 | 0.9999999999979999 | 9.271309945019235e-06 | OK |
| aman:anchor_095370 | [699960] | [299984] | 0.9525424154358397 | 0.9999999999979998 | 0.0069598707835612445 | OK_LOCKED |
| aman:megayak_5f_lm | [699960] | [299984] | 0.9504282674228537 | 0.9999999999979999 | 1.333366348788001e-05 | OK |
| aman:masaya_gpt2 | [699960] | [299984] | 0.9500272047255064 | 0.9999999999979999 | 7.704965477473014e-06 | OK |
| aman:lm7_5fold | [699960] | [299984] | 0.9494475201107185 | 0.9999999999979999 | 7.53873759007373e-06 | OK |
| aman:gpt2_seed2026 | [699960] | [299984] | 0.9501396986388475 | 0.9999999999979997 | 8.78032237075499e-06 | OK |
| aman:tabpfn_slimypunk_tp1 | [699960] | [299984] | 0.9492943359707291 | 0.9999999999979999 | 7.53873759007373e-06 | OK |
| aman:sadam_tabpfn_100k_tp3 | [699960] | [299984] | 0.9488330057825877 | 0.9999999999979999 | 0.00017905536176094012 | OK |
| aman:yekenot_realmlp | [699960] | [299984] | 0.9500162915719409 | 0.999999999998 | 9.297598563468013e-06 | OK |
| aman:alex_glm_gen | [699960] | [299984] | 0.9504716815719364 | 0.9999999999979997 | 1.039524363331823e-05 | OK |
| aman:daoviet_gpdvae | [699960] | [299984] | 0.946489033767143 | 0.9999999999979999 | 9.27340541459487e-06 | OK |
| aman:gbn_bag5_w512 | [699960] | [299984] | 0.951222973548657 | 0.9999999999979998 | 1.1701102181405076e-05 | OK |
| aman:sameer_glm10f | [699960] | [299984] | 0.951594937279145 | 0.9999999999979997 | 1.4717054410695596e-05 | OK |
| aman:sadam_v6_gb_xgb | [699960] | [299984] | 0.9507706939127138 | 0.9999999999979999 | 9.276643867628387e-06 | OK |
| aman:sadam_v6_gb_lgb | [699960] | [299984] | 0.9509418011006124 | 0.9999999999979998 | 9.265785525292003e-06 | OK |
| aman:sadam_v4_cat | [699960] | [299984] | 0.9500714069842968 | 0.9999999999979999 | 9.315124309156042e-06 | OK |
| aman:gb_bag5 | [699960] | [299984] | 0.9515058163094932 | 0.9999999999979997 | 1.0524210261708689e-05 | OK |
| aman:megayak_shallow_tree | [699960] | [299984] | 0.9514702806472366 | 0.9999999999979999 | 1.2491856205082419e-05 | OK |
| part12:lm | [699960] | [299984] | 0.9504282674228537 | 0.9999999999979999 | 1.333366348788001e-05 | OK |

## 2. z_modern vs z_pub_only (0.952614)

Method: priority-first diverse keep (28) → nested 5-fold `LogisticRegression` on `L_fold` → blend grid vs locked `z_pub_only` (winner **`pub_rankmean`**).  
`z_modern_all` = same nest + strong prop (`gpt2`/`l7`/`B_xgb`); blend winner collapsed to **`all_anchor`** (= z_modern_pub). Residuals (`soft_diff`, `delta1`, `L4.6C`, P4) reserved for Stage-B only.

| stack | AUC+Zr | Δe5 vs z_pub | pos/5 | t25 | sf25 | 1−ρ vs AB e4 |
|---|---:|---:|---:|---:|---:|---:|
| z_modern_pub | **0.952652** | **+5.21** | **4/5** | 1.43 | 16 | 200.92 |
| z_modern_all | 0.952652 | +5.21 | 4/5 | 1.43 | 16 | 200.92 |

## 3. StageB v42.1 (softened)
- Bug1: `init_score = 0.25 * L(z_modern_pub)` (not `1.0 * z_pub_only`)
- Bug2: Method-B / ECDF only on continuous model-logits; discrete/regime raw
- Bug3: post-orth Method-B `disp=1.0000`; shallow LGB `max_depth=3, num_leaves=7, min_child_samples=400, colsample=0.5`
- CatBoost depth-4 `l2_leaf_reg=30` attempted; this CatBoost build rejected `eval_baseline` → **LGB-only residual** this run
- δ vs z_modern_all: Δe5=**0.00** pos=3/5 sf=13 t25=1.26 sign=+1 α=0.01 disp=1.0000 KS=0.0301
- δ stack vs z_pub: Δe5=**+5.21** pos=4/5 (inherits modern bank lift; no extra Stage-B leap)

| tip e4 | cos·S02↓ | cos·S06↓ |
|---:|---:|---:|
| 1.00 | -0.108 | -0.999 |
| 2.00 | -0.108 | -0.999 |
| 5.00 | -0.108 | -0.999 |
| 8.00 | -0.108 | -0.999 |
| 12.00 | -0.108 | -0.999 |
| 15.00 | -0.108 | -0.999 |

### L4.6C top12 regime: Δe5=0.00 pos=2/5 sf=12

Diagnostic CSV (NOSUBMIT): `V42_1_StageB_soft_delta_NOSUBMIT.csv` sha=`33852654a535348abb69531c5e5d2ab8bb6b9c86261177fa51a5015341473dc0`

## STOP
NO_SUBMIT=True. Zero submissions. Awaiting explicit fire command.
