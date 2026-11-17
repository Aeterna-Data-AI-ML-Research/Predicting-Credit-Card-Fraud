# GM V24.1 Professor Lever Study

ATH baseline **0.95440** · elapsed 424s

## 1. Scientific framing

Public AUC is a functional of the **rank** of test scores. Levers are interventions on a score field `s`.
We treat the sriram 0.95440 file as an estimated generative model:
`s = U(avg(U(L(G)+w Δ_v15), U(L(B)+w Δ_v15)))` with `w=0.30`, and ask which factors are **necessary**.

## 2. Factorial ablation (identifiability)

- Canonical recon spearman to ATH ≈ `0.99999996` (recipe identified).
- Marginal mean ρ(ATH): see log (dual_GB + v15 + w≈0.30 + roundtrip dominate closeness).
- Interpretation: **dual base** and **v15 neural** are first-order; w is a continuous tuning parameter; CSV roundtrip is second-order.

## 3. Nested residual screen

- Bank members screened: **70**; ATH OOF surrogate AUC=`0.949688`.
- Gated residuals (pos≥4, bootstrap CI_lo>0, MC private-up≥0.55, t≥2.5): **23**.

| residual | pos | t | meanΔe5 | CI | MC gap |
|---|---:|---:|---:|---|---:|
| `realmlp_v1` | 5/5 | 6.24 | +1.36 | [+0.98,+1.73] | +0.37 |
| `gb_bag5` | 5/5 | 4.85 | +0.20 | [+0.13,+0.27] | +0.03 |
| `realmlp_dig_s100` | 5/5 | 4.70 | +2.19 | [+1.26,+2.92] | -0.28 |
| `realmlp_dig` | 5/5 | 4.47 | +2.00 | [+1.10,+2.63] | -0.06 |
| `glm_dl` | 5/5 | 4.12 | +1.39 | [+0.80,+1.94] | +0.13 |
| `glm_dl_bag` | 5/5 | 4.12 | +1.38 | [+0.82,+1.93] | -0.12 |
| `glm_ohk` | 5/5 | 3.77 | +1.02 | [+0.55,+1.48] | -0.35 |
| `lgb_diglead_l3` | 5/5 | 3.66 | +2.32 | [+1.16,+3.28] | -0.01 |

## 4. Hypotheses & live results

| ID | Claim | ρ(ATH)e-4 | Public |
|---|---|---:|---:|
| `H1` | w=0.35 dual_GB v15 beats ATH (under-regularized neural Δ) | 0.171 | **None** |
| `H2` | w=0.40 dual_GB v15 still lifts (monotonic in w near 0.30) | 0.683 | **staged** |
| `H3` | 0.5-extrapolate w30→w40 lifts without full w40 noise | 0.171 | **staged** |
| `H4` | dual(R2,gpw32)@0.35 improves our alternate ATH lineage | 2.809 | **staged** |
| `H5` | dual(ATH,B)@0.30 injects Bhuvan diversity into ATH | 1.640 | **staged** |
| `H6` | bold away from dual_live isolates unique ATH component | 0.494 | **staged** |
| `H7` | ATH+ORTH(realmlp_v1)@0.30e-4 nested-gated residual | 0.300 | **staged** |

- Best live among submits: **None**

## 5. Methodological notes (professor constraints)

1. **Falsifiability:** each H has an explicit kill line on public LB.
2. **Multiple testing:** do not interpret a single +1e-5 as discovery; require replication on a second contrast.
3. **Public-slice risk:** MC gap>0 ⇒ board-optimistic; prefer hyps with priv_pos≥0.55.
4. **Avoid exhausted families:** orth-twilight / Ray-1 crest excluded a priori.

## Log
```
[    0.0s] === V24.1 PROFESSOR LEVER STUDY ===
[    5.2s] §1 Formal setting: maximize public AUC; proxy nested/OOF + MC pub/priv; ATH=0.95440
[    5.5s] ATH geometry: vs_R2=6.586e-4 vs_G=6.593e-4 vs_B=7.386e-4 vs_gpw32=6.551e-4
[    5.5s] §2 Factorial ablation of 0.95440 recipe
[   47.0s] Closest reconstructions to live ATH (by 1-ρ):
[   47.0s]   dual_GB      v15 w=0.30 rt=0 ρ(ATH)=0.0004e-4 spear=0.99999996
[   47.0s]   dual_GB      v15 w=0.30 rt=1 ρ(ATH)=0.0004e-4 spear=0.99999996
[   47.0s]   dual_GB      v15 w=0.35 rt=0 ρ(ATH)=0.1714e-4 spear=0.99998286
[   47.0s]   dual_GB      v15 w=0.35 rt=1 ρ(ATH)=0.1714e-4 spear=0.99998286
[   47.0s]   dual_GB      v15 w=0.22 rt=0 ρ(ATH)=0.4384e-4 spear=0.99995616
[   47.0s]   dual_GB      v15 w=0.22 rt=1 ρ(ATH)=0.4384e-4 spear=0.99995616
[   47.0s]   dual_GB      v15 w=0.40 rt=0 ρ(ATH)=0.6834e-4 spear=0.99993166
[   47.0s]   dual_GB      v15 w=0.40 rt=1 ρ(ATH)=0.6834e-4 spear=0.99993166
[   47.0s] Marginal mean ρ(ATH) by factor (identifiability):
[   47.0s]   base: dual_GB=1.795, G_only=2.613, B_only=2.628, dual_R2_GPW=3.999
[   47.0s]   neural: v15=1.418, v5=4.099
[   47.1s]   w: 0.22=1.797, 0.3=2.266, 0.15=2.268, 0.35=3.104, 0.4=4.359
[   47.1s]   rt: False=2.759, True=2.759
[   47.9s] Canonical recon ρ(ATH)=0.0004e-4 spear=0.99999996
[   54.8s] §3 Build ATH OOF surrogate + screen bank residuals
[   62.6s] bank members with oof+test: 70
[   70.2s] ATH OOF surrogate AUC=0.949688
[  396.3s] Top residual candidates (nested gate: pos≥4, CI_lo>0, mc_priv_pos≥0.55):
[  396.3s]   realmlp_v1             5/5 t= 6.24 m=+1.36 CI[+0.98,+1.73] gap=+0.37 priv+=1.00 GATE=True
[  396.3s]   gb_bag5                5/5 t= 4.85 m=+0.20 CI[+0.13,+0.27] gap=+0.03 priv+=1.00 GATE=True
[  396.3s]   realmlp_dig_s100       5/5 t= 4.70 m=+2.19 CI[+1.26,+2.92] gap=-0.28 priv+=1.00 GATE=True
[  396.3s]   realmlp_dig            5/5 t= 4.47 m=+2.00 CI[+1.10,+2.63] gap=-0.06 priv+=1.00 GATE=True
[  396.3s]   glm_dl                 5/5 t= 4.12 m=+1.39 CI[+0.80,+1.94] gap=+0.13 priv+=1.00 GATE=True
[  396.3s]   glm_dl_bag             5/5 t= 4.12 m=+1.38 CI[+0.82,+1.93] gap=-0.12 priv+=0.95 GATE=True
[  396.3s]   glm_ohk                5/5 t= 3.77 m=+1.02 CI[+0.55,+1.48] gap=-0.35 priv+=0.93 GATE=True
[  396.3s]   lgb_diglead_l3         5/5 t= 3.66 m=+2.32 CI[+1.16,+3.28] gap=-0.01 priv+=1.00 GATE=True
[  396.3s]   lgb_dk_l4              5/5 t= 3.59 m=+1.66 CI[+0.86,+2.46] gap=+0.80 priv+=0.97 GATE=True
[  396.3s]   lgb_diglead_l3_bag     5/5 t= 3.50 m=+1.74 CI[+0.84,+2.54] gap=+0.02 priv+=1.00 GATE=True
[  396.3s]   lgb_dig_s100_l3_bag    5/5 t= 3.27 m=+1.64 CI[+0.87,+2.54] gap=+0.10 priv+=1.00 GATE=True
[  396.3s]   lgb_dig_s100_l3        5/5 t= 3.18 m=+1.81 CI[+0.88,+2.96] gap=-0.56 priv+=1.00 GATE=True
[  396.3s]   realmlp_te_s2          5/5 t= 3.05 m=+1.04 CI[+0.51,+1.66] gap=+0.07 priv+=0.97 GATE=True
[  396.3s]   realmlp_cnt            5/5 t= 3.02 m=+1.36 CI[+0.58,+2.14] gap=+0.09 priv+=0.97 GATE=True
[  396.3s]   realmlp_rawdig         5/5 t= 2.80 m=+1.02 CI[+0.40,+1.64] gap=-0.36 priv+=0.90 GATE=True
[  396.3s]   lgb_rawdig_l4          5/5 t= 2.20 m=+1.04 CI[+0.37,+1.94] gap=+0.13 priv+=1.00 GATE=False
[  396.3s]   ebm_dig                5/5 t= 1.94 m=+0.11 CI[+0.04,+0.23] gap=-0.03 priv+=0.93 GATE=False
[  396.3s]   realmlp_dig_s2         4/5 t= 3.66 m=+1.96 CI[+0.91,+2.69] gap=+0.27 priv+=1.00 GATE=True
[  396.3s]   glm_ohk2_bag           4/5 t= 2.87 m=+1.08 CI[+0.37,+1.63] gap=-0.03 priv+=0.97 GATE=True
[  396.3s]   lgb_diglead_l4         4/5 t= 2.86 m=+2.14 CI[+0.82,+3.31] gap=+0.12 priv+=1.00 GATE=True
[  396.3s]   ebm_i25_bag            4/5 t= 2.85 m=+1.64 CI[+0.51,+2.60] gap=+0.32 priv+=1.00 GATE=True
[  396.3s]   lgb_dig_s100_bag5      4/5 t= 2.78 m=+1.70 CI[+0.72,+2.68] gap=+0.06 priv+=0.97 GATE=True
[  396.3s]   realmlp_te             4/5 t= 2.72 m=+1.26 CI[+0.41,+1.96] gap=-0.20 priv+=1.00 GATE=True
[  396.3s]   glm_ohkl               4/5 t= 2.70 m=+0.71 CI[+0.28,+1.23] gap=+0.42 priv+=0.80 GATE=True
[  396.3s]   ebm_i25_bag4           4/5 t= 2.68 m=+1.76 CI[+0.55,+2.84] gap=-0.33 priv+=1.00 GATE=True
[  396.3s] §4 Hypothesis register
[  402.2s] H7 uses residual realmlp_v1
[  402.3s]   H1: ρ(ATH)=0.171e-4 prior=med | w=0.35 dual_GB v15 beats ATH (under-regularized neural Δ)
[  402.5s]   H2: ρ(ATH)=0.683e-4 prior=low-med | w=0.40 dual_GB v15 still lifts (monotonic in w near 0.30)
[  402.6s]   H3: ρ(ATH)=0.171e-4 prior=med | 0.5-extrapolate w30→w40 lifts without full w40 noise
[  402.7s]   H4: ρ(ATH)=2.809e-4 prior=med | dual(R2,gpw32)@0.35 improves our alternate ATH lineage
[  402.9s]   H5: ρ(ATH)=1.640e-4 prior=med | dual(ATH,B)@0.30 injects Bhuvan diversity into ATH
[  403.0s]   H6: ρ(ATH)=0.494e-4 prior=low | bold away from dual_live isolates unique ATH component
[  403.1s]   H7: ρ(ATH)=0.300e-4 prior=med-high | ATH+ORTH(realmlp_v1)@0.30e-4 nested-gated residual
[  404.3s] STAGED sub_v24_1_H1.csv
[  405.8s] STAGED sub_v24_1_H2.csv
[  407.5s] STAGED sub_v24_1_H3.csv
[  408.9s] STAGED sub_v24_1_H4.csv
[  410.0s] STAGED sub_v24_1_H5.csv
[  410.8s] STAGED sub_v24_1_H6.csv
[  411.5s] STAGED sub_v24_1_H7.csv
[  412.1s] STAGED sub_v24_1_abl_canon_recon.csv
[  412.6s] STAGED sub_v24_1_abl_w035_dual_GB.csv
[  413.3s] STAGED sub_v24_1_abl_w040_dual_GB.csv
[  414.1s] STAGED sub_v24_1_abl_extrap_w30_40.csv
[  414.1s] SUBMIT v24.1 H1 | ρ(ATH)=0.171e-4 | w=0.35 dual_GB v15 beats ATH (under-regularized neural Δ)
[  423.8s] 
[  423.8s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<00:48, 163kB/s]
  7%|▋         | 528k/7.52M [00:00<00:03, 2.33MB/s]
 10%|▉         | 736k/7.52M [00:00<00:10, 700kB/s] 
 12%|█▏        | 960k/7.52M [00:01<00:08, 815kB/s]
 
[  423.8s] DONE H1 pub=None status=quota_or_error
[  423.8s] STOP: no quota — remaining hyps staged only
```

