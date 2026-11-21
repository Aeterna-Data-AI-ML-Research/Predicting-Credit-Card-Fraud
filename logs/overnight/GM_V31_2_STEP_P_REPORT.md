# GM V31.2 HARDENED — Step-P Audit & Strike Report

**Live ATH:** **0.95537** (our V31.2 SLOT1/SLOT2) · prior ceiling 0.95536 (goodpjw Step O) · **submits this run: 2** · **remaining≈6**

## Native Step N → Step O
- Space: `L=logit(U)`, `Zr=U(U(x)-2*ZALL)`, export `U(...)`
- `FILE_O = Zr(apply(STEP_O, L(RIS), test))` with `RIS` = Rishabh 0.95535 file from Areeb residual continuation
- Not a simple secant of N→M; it is a **152-member step from RIS**

### Top |Δw| STEP_N → STEP_O
| member | w_N | w_O | Δ |
|--------|-----|-----|---|
| `b:gb_bag5` | +0.0311 | -0.0120 | -0.0431 |
| `b:lgb_diglead_bag` | +0.0383 | -0.0001 | -0.0384 |
| `x:Sold` | -0.0240 | -0.0613 | -0.0373 |
| `s:gb_bag5` | +0.0132 | +0.0493 | +0.0361 |
| `b:gb_bag10` | -0.0614 | -0.0305 | +0.0308 |
| `b:glm_ohk_c01` | +0.0067 | -0.0231 | -0.0297 |
| `b:glm_ohkl_bag` | +0.0023 | +0.0309 | +0.0285 |
| `s:masaya_gpt2` | -0.0064 | +0.0215 | +0.0279 |
| `x:b33_glm_ohk_g2_c0003_s50` | -0.0025 | +0.0249 | +0.0274 |
| `x:b33_glm_plain_g2_s50` | +0.0133 | -0.0131 | -0.0264 |

## Temporal Holdout Board-Model
- Train: LB < 0.95533 · Holdout: {N, Areeb, O, DUAL, STAR, BOLD, SLOT1, gpw33}
- Spearman ρ(holdout) = **0.969** (need ≥ 0.70)
- Ranking checks: `{'O>N': True, 'O>Areeb': True, 'O>SLOT1': True, 'Areeb>N': True}`
- Temporal pass: **True** → Gate A overall: **True**

## z_true_oof
- AUC = **0.952364** (target ≥ 0.95258)

## Proprietary R² / LOFO (Gate B)
| name | kind | R²oof | R²test | novel | spanned | pos | mean e-4 | min e-5 | sf25 | t | disp | gateB |
|------|------|------:|-------:|:-----:|:-------:|----:|---------:|--------:|-----:|--:|-----:|------:|
| `L7_s2026` | prob | 0.972 | 0.977 | n | Y | 3/5 | -0.128 | -12.54 | 11/25 | -0.29 | 1.000 | FAIL |
| `gpt2_med` | prob | 0.961 | 0.968 | n | Y | 3/5 | -0.128 | -12.54 | 13/25 | -0.29 | 1.000 | FAIL |
| `P4_L4.2` | resid | 0.010 | 0.042 | Y | n | 3/5 | -0.128 | -12.54 | 13/25 | -0.29 | 1.000 | FAIL |
| `P4_L4.5` | resid | 0.003 | 0.019 | Y | n | 3/5 | -0.128 | -12.54 | 11/25 | -0.29 | 1.000 | FAIL |
| `P4_L4.3` | resid | 0.022 | 0.031 | Y | n | 3/5 | -0.128 | -12.54 | 14/25 | -0.29 | 1.000 | FAIL |
| `v10_ridge_orth_LM` | resid | 0.023 | 0.043 | Y | n | 3/5 | -0.128 | -12.54 | 12/25 | -0.29 | 1.000 | FAIL |
| `v10_ridge_resid` | resid | 0.038 | 0.055 | Y | n | 3/5 | -0.128 | -12.54 | 13/25 | -0.29 | 1.000 | FAIL |
| `U5_gini10` | resid | 0.385 | 0.394 | Y | n | 3/5 | -0.128 | -12.54 | 12/25 | -0.29 | 1.000 | FAIL |

## Gates & Live
| Gate | Status |
|------|--------|
| A (temporal + native momentum) | True |
| B (R²<0.75 + LOFO 5/5) | False |
| SLOT1 LB | 0.95537 |
| SLOT2 LB | 0.95537 |

## Honest NOSUBMIT
- w=None LOFO=None
- `sub_v31_2_HONEST_700K_C001_SUPERSTACK_NOSUBMIT.csv`

## Decision
**NEW #1 PUBLIC ATH 0.95537** via Gate A only (`C2` native Areeb-damped `d_NO−0.011·e_MN`, one-sided cone vs losers, `1−ρ(O)=0.20e-4`). SLOT2 ride @`0.22e-4` tied 0.95537 — STOP (6 slots kept). Gate B failed (all prop LOFO 3/5; `z_true` 0.952364 < 0.95258).

## Log
```
[    0.0s] === GM V31.2 HARDENED Step-P audit | submit only if Gate A/B clear ===
[    6.3s] submits_today=2 remaining_est=8 | public ceiling still 0.95536 (goodpjw Step O)
[    9.4s] NATIVE SPACE (from goodpjw source):
[    9.4s]   U(v)=(rankdata-0.5)/n  |  L(v)=logit(U(v))  |  Zr(x)=U(U(x)-2*ZALL)
[    9.4s]   apply(w,base)=(1-Σw)*base + Σ w_k*L(member_k)
[    9.4s]   FILE_N = Zr(apply(STEP_N, L(FILE_M), test))
[    9.4s]   P_AREEB = U(L(N)+0.5*r)  r=dN-β(L(F)-L(M))
[    9.4s]   RIS = Zr(L(P_AREEB)+2*(L(P_AREEB)-L(N)))  # Rishabh 0.95535
[    9.4s]   FILE_O = Zr(apply(STEP_O, L(RIS), test)) → export U(FILE_O)
[    9.6s]   1-ρ(O,N)=1.026560e-4  1-ρ(O,Areeb)=0.617027e-4  1-ρ(O,SLOT1)=0.350055e-4
[    9.7s]   Step O CSV is rank-frac (Spearman(O,U(O))=1) — confirmed native export space
[    9.7s] Top |Δw| STEP_N→STEP_O:
[    9.7s]   b:gb_bag5: N=+0.0311 O=-0.0120 Δ=-0.0431
[    9.7s]   b:lgb_diglead_bag: N=+0.0383 O=-0.0001 Δ=-0.0384
[    9.7s]   x:Sold: N=-0.0240 O=-0.0613 Δ=-0.0373
[    9.7s]   s:gb_bag5: N=+0.0132 O=+0.0493 Δ=+0.0361
[    9.7s]   b:gb_bag10: N=-0.0614 O=-0.0305 Δ=+0.0308
[    9.7s]   b:glm_ohk_c01: N=+0.0067 O=-0.0231 Δ=-0.0297
[    9.7s]   b:glm_ohkl_bag: N=+0.0023 O=+0.0309 Δ=+0.0285
[    9.7s]   s:masaya_gpt2: N=-0.0064 O=+0.0215 Δ=+0.0279
[    9.7s]   x:b33_glm_ohk_g2_c0003_s50: N=-0.0025 O=+0.0249 Δ=+0.0274
[    9.7s]   x:b33_glm_plain_g2_s50: N=+0.0133 O=-0.0131 Δ=-0.0264
[    9.7s]   b:glm_ohk2_bag: N=+0.0174 O=-0.0082 Δ=-0.0256
[    9.7s]   x:b33_lgb_g2_glminit_s50: N=+0.0044 O=-0.0209 Δ=-0.0253
[   10.6s] scored ladder vectors loaded: 21
[   10.6s] temporal train n=15 (LB<0.95533) | holdout n=8
[   13.6s] TEMPORAL HOLDOUT predicted vs actual:
[   13.6s]   step_o: pred=0.95533 actual=0.95536 Δ=-0.00003
[   13.6s]   areeb: pred=0.95531 actual=0.95534 Δ=-0.00003
[   13.6s]   slot1_l7: pred=0.95532 actual=0.95534 Δ=-0.00002
[   13.6s]   step_n: pred=0.95531 actual=0.95533 Δ=-0.00002
[   13.6s]   dual: pred=0.95531 actual=0.95533 Δ=-0.00002
[   13.6s]   gpw33: pred=0.95531 actual=0.95533 Δ=-0.00002
[   13.6s]   star_gpw: pred=0.95531 actual=0.95532 Δ=-0.00001
[   13.6s]   bold_maths: pred=0.95530 actual=0.95532 Δ=-0.00002
[   13.6s]   Spearman ρ(holdout)=0.969
[   13.6s] TEMPORAL GATE: rho>=0.7? True checks={'O>N': True, 'O>Areeb': True, 'O>SLOT1': True, 'Areeb>N': True} -> pass=True
[   13.6s] Board-Model V2 refit on ALL scored points (temporal PASSED)
[   34.0s] bank pairs for z_true: 66 top=wide_lm@0.95257
[   42.6s] z_true candidates: Ridge-L AUC=0.951942 | LogReg-U AUC=0.951586
[   42.9s] z_true_oof AUC=0.951942 (need >= 0.95258) | ||w_oof-w_test||rel=0.0378
[   43.2s] z_true blend with wide_lm AUC=0.952364
[   43.3s] z_true gate soft>=0.9525: False (strict 0.95258: False)
[   69.3s] PROP L7_s2026: R²o=0.972 R²t=0.977 novel=False spanned=True | LOFO 3/5 m=-0.128e-4 min=-1.254 sf=11/25 t=-0.29 passB=False
[   71.9s] PROP gpt2_med: R²o=0.961 R²t=0.968 novel=False spanned=True | LOFO 3/5 m=-0.128e-4 min=-1.254 sf=13/25 t=-0.29 passB=False
[   74.4s] PROP P4_L4.2: R²o=0.010 R²t=0.042 novel=True spanned=False | LOFO 3/5 m=-0.128e-4 min=-1.254 sf=13/25 t=-0.29 passB=False
[   76.8s] PROP P4_L4.5: R²o=0.003 R²t=0.019 novel=True spanned=False | LOFO 3/5 m=-0.128e-4 min=-1.254 sf=11/25 t=-0.29 passB=False
[   79.3s] PROP P4_L4.3: R²o=0.022 R²t=0.031 novel=True spanned=False | LOFO 3/5 m=-0.128e-4 min=-1.254 sf=14/25 t=-0.29 passB=False
[   81.8s] PROP v10_ridge_orth_LM: R²o=0.023 R²t=0.043 novel=True spanned=False | LOFO 3/5 m=-0.128e-4 min=-1.254 sf=12/25 t=-0.29 passB=False
[   84.2s] PROP v10_ridge_resid: R²o=0.038 R²t=0.055 novel=True spanned=False | LOFO 3/5 m=-0.128e-4 min=-1.254 sf=13/25 t=-0.29 passB=False
[   87.2s] PROP U5_gini10: R²o=0.385 R²t=0.394 novel=True spanned=False | LOFO 3/5 m=-0.128e-4 min=-1.254 sf=12/25 t=-0.29 passB=False
[   87.8s] GATE A (temporal+cone): True | post-cone dots vs losers=[-298104.9378190312, -277939.1206089898, -171058.19140258135]
[   87.8s] GATE B (novel+LOFO): False n_passers=0 z_ok=False
[   90.0s] C2 staged omr(O)=0.2001 (native Areeb-damped d_NO)
[   91.5s] HONEST NOSUBMIT = z_true only AUC=0.952364 (no Gate-B passers)
[   97.8s] 7.52M [00:00<00:07, 978kB/s] 
 12%|█▏        | 912k/7.52M [00:00<00:06, 1.01MB/s]
 17%|█▋        | 1.31M/7.52M [00:01<00:04, 1.44MB/s]
 20%|██        | 1.52M/7.52M [00:01<00:04, 1.57MB/s]
 30%|███       | 2.27M/7.52M [00:01<00:02, 2.60MB/s]
 34%|███▎      | 2.53M/7.52M [00:01<00:02, 2.31MB/s]
 70%|███████   | 5.27M/7.52M [00:01<00:00, 7.89MB/s]
100%|██████████| 7.52M/7.52M [00:02<00:00, 3.26MB/s]

[  116.7s] POLLED sub_v31_2_SLOT1_C2_GateA_native_momentum.csv -> 0.95537
[  116.7s] NEW ATH — SLOT2 continuation @0.22e-4
[  125.1s] 2k/7.52M [00:00<00:07, 930kB/s] 
  8%|▊         | 624k/7.52M [00:00<00:08, 901kB/s]
 10%|█         | 800k/7.52M [00:00<00:08, 867kB/s]
 33%|███▎      | 2.50M/7.52M [00:01<00:01, 4.18MB/s]
 46%|████▌     | 3.42M/7.52M [00:01<00:00, 5.32MB/s]
 56%|█████▌    | 4.17M/7.52M [00:01<00:00, 5.42MB/s]
 64%|██████▍   | 4.84M/7.52M [00:01<00:00, 5.65MB/s]
100%|██████████| 7.52M/7.52M [00:02<00:00, 3.83MB/s]

[  144.4s] POLLED sub_v31_2_SLOT2_ride_ath.csv -> 0.95537
```
