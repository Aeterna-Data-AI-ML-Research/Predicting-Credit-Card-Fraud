# GM V33.1 HARDENED FORENSICS REPORT

- NO_SUBMIT: `True`
- elapsed_s: 211.5

## Part 1 — v33_canary Post-Mortem

- **float_format:** exact decimal strings via f'{v}' / format; NOT binned; digit+dot chars tokenized by Qwen BPE
- **serialization:** compact single-token markers ' A'..' S' + body (digits/categories); prefix ' Y0'/' Y1' (2 toks)
- **max_len_88:** LEN_SURVEY 20k rows max_obs=80 → ceil((80+4)/8)*8=88; assert 0 truncated
- **V_tab_89_90:** active vocab = unique tokens in compact serialization (~digits, markers, cat words, pad) sliced from full 151936
- **sub_head_init:** nn.Linear(H,V) copied from lm_head.weight[used_token_ids].float(); backbone=model.model, lm_head deleted
- **scoring:** FULL generative NLL LLR: sum logp over post-prefix tokens for y:1 seq minus y:0 seq; mean over K=4 antithetic block orders (NOT single target-token logit)
- **training_loss:** causal F.cross_entropy on true-y prefix sequences (pos+neg stream), NOT LLR loss

### Fold-0 5-Subfold OOS

| metric | value |
|---|---|
| wide AUC | 0.951774 |
| qwen solo | 0.946923 |
| tabicl solo | 0.942573 |
| OOS blend AUC | 0.951559 |
| Δ vs wide (e-4) | -2.144 |
| pos subfolds | 2/5 |
| corr(qwen,wide) | 0.9684 |
| verdict | CLOSED — OOS blend does not beat wide; canary non-bank signal null under leakage-free 5-subfold |

### Proprietary inventory (R² vs L(wide))

- `P4_L4.2`: exists=True R²=0.00024424754504447164
- `P4_L4.3`: exists=True R²=0.003531963617407911
- `P4_L4.5`: exists=True R²=0.0025253371263439395
- `B_xgb_llr`: exists=True R²=0.9903329558709598
- `gpt2_3ep`: exists=True R²=0.9374178275088271
- `v10_ridge_orth`: exists=True R²=0.0009197086890787693
- `delta_cat`: exists=True R²=0.9982031617477742

**z_prop_anchor** cols=['wide', 'P4_L4.2', 'P4_L4.3', 'P4_L4.5', 'B_xgb_llr', 'gpt2_3ep', 'v10_ridge_orth', 'delta_cat'] AUC=0.952730 Δ_wide=1.64e-4

### Generative multi-order scan

- `l7_mean_llr`: 1/5 sub=3/25 Δ=-0.63e-4 t=-0.97 pass=False
- `l7_std_llr`: 0/5 sub=4/25 Δ=-0.22e-4 t=-1.16 pass=False
- `l7_joint_mean`: 0/5 sub=1/25 Δ=-0.41e-4 t=-1.00 pass=False
- `l7_anchor_x_std`: 0/5 sub=1/25 Δ=-0.22e-4 t=-1.00 pass=False
- `gpt2med_mean_llr`: 0/5 sub=2/25 Δ=-0.03e-4 t=-1.00 pass=False
- `gpt2med_std_llr`: 0/5 sub=0/25 Δ=0.00e-4 t=0.00 pass=False

## Part 2 — Udit / Breakaway Footprint

*(Backfilled with `/Users/mz/Library/Python/3.14/bin/kaggle` — full dump: `logs/overnight/GM_V33_1_PART2_FOOTPRINT.json`)*

### Udit Jain (`uditjain13`)
- **Dataset** `uditjain13/credit-card-fraud-detection-2026` live (id=11219727, downloads=5285, votes=69). Subtitle: “20,000 transactions, 1.7% fraud - readable columns, not PCA mystery”. Description names `is_ai_generated_scam_attempt`, velocity/cvv/auth/VPN features — **no public `gen.py` attached / listed**.
- **Kernels:** many 2026 EDA baselines (e-commerce, insurance, stroke, heart, NeuroGolf…) — **none titled S6E11 / credit-card-fraud / gen.py**.
- **GitHub user repos:** only public `uditjain13/mewtwo` (no S6E11 / gen.py).
- **GitHub code search** `is_ai_generated_scam_attempt`: **401 Unauthorized** (unauthenticated API).
- **GitHub repo search** `credit-card-fraud-detection-2026`: 20 hits (community forks/mirrors, e.g. `VinayMR0/...`) — **no authoritative Udit `gen.py` leak**.

### Breakaway accounts (public kernels/datasets)
| account | public kernels | public datasets | S6E11-relevant? |
|---|---|---|---|
| `dharunmk` | Not found | none | **no public footprint** |
| `smartmanoj` | TPU/batch-invariance/Gemma MREs (no S6E11) | none | **no S6E11 kernel** |
| `ahmetbugrahanaltunok` | S6E9 17th-place only | none | **no S6E11 kernel** |
| `srikalyan` | Not found | Nemotron offline pkgs | **no S6E11** |
| `lebxgue` | Not found | none | **no public footprint** |
| `starkhushi` | Credit-card EDA + “28th→243rd overnight” (Nov 1–2) | present | **EDA only — no breakaway recipe** |

### New S6E11 kernels (2026-11-21, last ~12h)
- `dennis6886/s6e11-fraud-signal-rank-fusion-*` (rank-fusion refinements)
- `goutamsharma12/spline-logistic-regression`
- `olegpan` / `vladboy` LogReg baselines, HitTmg Optuna HGB/CB/XGB/LGBM, etc.
- **None from the 6 breakaway accounts; no public `gen.py` / leak kernel detected.**

## Part 3 — orig20k `gen.py` Structure

- n=20000 pos=339 prev=0.01695
- **Form verdict: `interaction_or_tree`**

| model | orig20k 5-fold AUC |
|---|---|
| linear_logit | 0.880399 |
| step_additive | 0.893758 |
| pairwise_L1 | 0.460593 |
| gbdt_d2 | 0.934158 |
| gbdt_d3 | 0.931276 |
| gbdt_d4 | 0.929010 |

### Top-15 linear terms

- `ip_country_mismatch`: +1.5171
- `cvv==0`: -1.3680
- `is_new_merchant`: +1.3426
- `cvv_retry_count`: +1.2231
- `auth_method_No Authentication`: +1.1638
- `billing_shipping_mismatch`: +1.0769
- `is_foreign_transaction`: +1.0159
- `used_vpn`: +0.9228
- `auth_method_Biometric`: -0.6791
- `disp==0`: -0.6484
- `is_ai_generated_scam_attempt`: +0.6255
- `auth_method_PIN`: -0.5884
- `cvv==2`: +0.4998
- `device_type_Windows PC`: -0.3736
- `card_type_Mastercard`: -0.3662

- |corr| mean orig=0.0172 (max=0.6224) vs train=0.0193
- velocity R² from other X on orig20k: 0.4582
- vel synth resid gate on train: 0/5 Δ=-0.69e-4 pass=False

## Part 4 — M1 / M2 Gates

### M1 families=['P_tree', 'P_bpe_lm', 'P_compact_lm', 'P_glm_mlp']
- vs wide: {'name': 'M1_vs_wide', 'outer_pos': '2/5', 'sub_pos': '8/25', 'mean_lift_e4': -0.430026548676965, 't_stat': -1.4163126954175846, 'wilcoxon_p': 0.8125, 'disp': 0.9999999999994487, 'pass': False, 'lifts_e4': [-1.0943434566401677, 0.10906207941530077, 0.0, 0.08106637184979171, -1.2459177380097497]}
- vs anchor: None
### M2 n_cols=16
- vs wide: {'name': 'M2_TE_vs_wide', 'outer_pos': '0/5', 'sub_pos': '0/25', 'mean_lift_e4': -0.9567617455450428, 't_stat': -0.9999999895480772, 'wilcoxon_p': 1.0, 'disp': 0.9999999999756802, 'pass': False, 'lifts_e4': [-4.783808727725214, 0.0, 0.0, 0.0, 0.0]}
- vs anchor: None

## Part 5 — 60k/80k Adaptive LB Simulation

- Pub_sim n=60000 pos=1002 · Priv_sim n=79992 · bank_cols=40

| K | Start_Pub | A_Pub | B_Pub | C_Pub | Start_Priv | A_Priv | B_Priv | C_Priv |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.95075 | 0.94956 | 0.95086 | 0.95078 | 0.95257 | 0.95136 | 0.95247 | 0.95252 |
| 104 | 0.95075 | 0.94993 | 0.95105 | 0.95078 | 0.95257 | 0.95103 | 0.95244 | 0.95252 |
| 200 | 0.95075 | 0.94995 | 0.95151 | 0.95078 | 0.95257 | 0.95096 | 0.95203 | 0.95253 |

### Conclusion

- Start_Pub=0.95075; max Δe-4 @K40/104/200 = 1.10/3.00/7.60
- Ahmet jump from ATH (0.95632-0.95537)=0.00095; sim can match jump≈False
- Dharun jump from ATH=0.00616; sim can match jump≈False
- Absolute reach 0.95632@K40=False; 0.96153@K104=False
- Note: Absolute Pub_sim AUCs are not identical to live Public LB; interpret primarily via Δ from start and whether +60–120e-4 jumps appear within K queries under 5-decimal oracle.

## Global Verdicts

1. **v33_canary CLOSED:** compact generative LLR (`max_len=88`, `|V_tab|≈89`, full-seq NLL LLR × K=4 orders) has **no leakage-free Fold-0 OOS lift** (2/5 subfolds, Δ=−2.14e-4). Order-epistemic / joint-density scans on l7 + gpt2_med also **FAIL** 25-subfold gate.
2. **`gen.py` form = interaction/tree, not pure additive logit:** orig20k ladder Linear 0.880 → Step 0.894 → Depth-2 HistGBDT **0.934** (+5.4pp). Features nearly independent in orig (`|corr|mean=0.017`) but `velocity_score` has latent R²=0.46 from other X; synthesizer residual **does not** lift train (0/5).
3. **M1 (calibrated family fusion) & M2 (Fisher Residual-TE):** both **FAIL** 25-subfold gate vs `L_fold(wide_lm)` (M1 2/5 Δ=−0.43e-4; M2 0/5 Δ=−0.96e-4). Do not pursue.
4. **LB overfitting cannot explain breakaways:** under exact 5-decimal `Query_Pub` on 60k/80k Fold-0 split, max Δ@K40/104/200 ≈ **+1.1 / +3.0 / +7.6 e-4** — far below Ahmet (+9.5e-4 abs from ATH) and Dharun (+61.6e-4). **0.95632@40 and 0.96153@104 exceed the black-box query bound → external/structural information required** (not recoverable from Public-LB probing alone). Public recon finds **no `gen.py` leak** from Udit or the 6 breakaway accounts.

**STRICT NO_SUBMIT=True — 0 Kaggle submits, 0 GPU pods.** · wall≈211.6s CPU
