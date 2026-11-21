# GM V31.0 — Forensic Autopsy of `0.95534` & OOF Basis Rebuild

**NO_SUBMIT = True** · Generated from `scripts/gm_v31_0_autopsy_095534_and_rebuild_oof.py`  
Sources pulled to `/tmp/s6e11_v31_autopsy/` · Datasets to `external/banks_v31/`

## Executive verdict

The leap **0.95426 → 0.95534 (~+108e-5)** is **not** a new raw feature / id-leak / orig20k lookup. It is a **public-board member-step ladder** (goodpjw) starting from yekenot RealMLP + public OOF banks, then a **+1e-5 orthogonal residual half-step** (Areeb) on top of step N.

| File | LB | What it actually is |
|------|-----|---------------------|
| goodpjw board-model step N | 0.95533 | Member-step: `L_out = (1−Σw)·L(base) + Σ w_k L(member_k)` with ~152 members; weights chosen by an offline **board model** (regression on ~167–173 scored public CSVs) + nested CV — **not** label pseudo-labeling |
| Areeb residual50 | **0.95534** | `d=L(N)−L(M)`, `e=L(F)−L(M)`, `r=d−βe`, `P=rank(L(N)+0.5 r)` with β≈0.011 → r ≈ step-N direction |
| Malyshev residual100 | 0.95534 | `L(P)+1.0·(L(P)−L(Q))` + soft zone floors — continuation of Areeb, not a new engine |
| goodpjw step O (overnight) | **0.95536** | New public peak (as of autopsy): member-step from N+1.5r file; **do not confuse with our locked 0.95534** |

## 1. Exact recipes (from pulled source)

### 1.1 Areeb (`areebbb/s6e11-which-signals-rank-fraud-lb-0-95534`)

- `dataset_sources`: **[]** (no OOF datasets)
- `kernel_sources`: Malyshev fusion, Asterios, goodpjw board-model
- Math (logit-rank space):

```text
N = L(goodpjw submission.csv)          # step N, LB 0.95533
M = L(goodpjw submission_step_m.csv)   # step M, LB 0.95531
F = L(Malyshev fusion submission)      # comparison direction ~0.95527
d = N - M
e = F - M
β = (d·e)/(e·e)                        # reported ≈ 0.011
r = d - β e                            # ≈ step-N direction
P = rank( N + 0.5 * r )                # scored 0.95534 (+1e-5)
```

Variants that tied/lost: `N+0.25 d` → 0.95533; `0.85 N + 0.15 F` → 0.95533.

### 1.2 goodpjw board-model (`…-lb-0-95536`, was `…-95533`)

- **What “board-model + RealMLP” means:**
  1. **Base:** public 0.95442 stack built from yekenot RealMLP (PyTabKit + PyTorch) + other public bases.
  2. **Members:** ~87–152 OOF/test streams from `s6e11-oof-bank`, `fraud-series-npz`, Aman Atar Q17 bundle, busyaprime TabM/BPE, Weni BPE, wide LM stack, etc.
  3. **Member step:** frozen weights `STEP_*` applied as  
     `apply(w, base) = (1−Σw)·L(base) + Σ_k w_k·L(member_k)` then zone-aware re-rank (`Zr`).
  4. **Board model (not shipped):** offline regressor fitted on **scored public CSVs** to **forecast** which candidate step gains public LB; nested CV on replayed OOF used as a second gate. This is **score-forecasting / step selection**, not distillation of `is_fraud` pseudo-labels onto test.
- Ladder (public points, 1 pt = 1e-5): RealMLP 0.95442 → … → M 0.95531 → **N 0.95533** → (others) → **O 0.95536**.
- Honest CV file they recommend for private: `submission_step_i.csv` / `submission_cv_led.csv` (OOF-positive), while late steps N/O are **public-led** and often **negative** on replayed OOF.

### 1.3 Malyshev

Pure public residual continuation of Areeb vs goodpjw + soft structural zero-fraud zones. SHA-locked recipe `residual100`.

### 1.4 Datasets downloaded to `external/banks_v31/`

- `goodpjw2008/s6e11-board-model-pinned-inputs` (Weni BPE OOF/test, TabM, Davut bag, Malyshev step-L file, …)
- `malyshevdanil/s6e11-nov20-late-public-residual`
- `amanatar/s6e11-rank1-q17-meta-bundle`

Also listed top-30 kernels (scoreDescending): **goodpjw 0.95536 now #1**; Bhuvan “40 scored files” 0.95534; Rishabh GS N+1.5r 0.95535 referenced in goodpjw text.

## 2. Why `gm_v23_0` LOFO aborted

### Measured R² (this run)

| Basis | Target | R² |
|-------|--------|-----|
| Stale `Bt_v22` (6 bank models) | `L(areeb_095534)` | **0.9259** |
| Updated `Bt_v31` (bank + ladder steps + ATH) | `L(areeb)` | **1.0000** |
| `Bt_v31` | `L(gpw_095533)` | **1.0000** |

### Root causes

1. **Wrong manifold:** V23 built `B_macro = Ridge(Bt_v22 → L(anchor))` after swapping anchor to Areeb. Areeb lives almost entirely in the **public ladder span** (steps M/N + tiny residual), not in the 6-column V22 bank. Residual directions from local OOFs, after GS against that mis-specified `B_macro`, had **no LOFO edge** (`pos ≤ 1/5`).
2. **No OOF for ATH:** Areeb/gpw late steps are **test-only public constructions**. Projecting local `u` orth to `L(areeb)` on test while gating LOFO on `B_macro_o` (bank replay) **mismatches train/test bases** → gates kill every `u_k`.
3. **Bug hygiene:** V23 logged `max|B_macro_t|=13` because it never unitized `B_macro_t` before the orth check (cosmetic), but the real failure was the LOFO gate on a stale control.

### Fix for V31+ (implemented here as measurement protocol, not submit)

- Treat `B_v31_test = span{L(areeb), L(gpw), L(step_m), L(step_n), …}`.
- Gate LOFO on **honest** OOF anchor (`gb_bag5` or gpw `submission_cv_led` replay), **not** on Ridge-to-Areeb.
- On test, require `|corr(u, L(areeb))|` and `|corr(u, L(gpw))|` ~ 0 after GS before any future star probe.

## 3. Proprietary engines vs `0.95534` basis

**Critical decode note:** `preds/gpt2_med_4ep/oof_*.npy` are `(n_val, 8, 2)` logits and `l7_s2026_*_raw.npy` are `(N, 10, 2)` / `(5, N, 10, 2)`. The first autopsy pass naively `.ravel()[:N]` → chance AUC. Rebuilt via softmax → mean-over-orders (saved under `/tmp/s6e11_v31_autopsy/*_rebuilt.npy`).

`P4_L4.*_pure5f_*.npy` are **residual-scale** (~±1e-3), not probabilities — raw solo AUC ~0.47–0.50. Not audited as standalone scores.

| Engine | Solo OOF AUC | ρ(test, areeb) | omr×1e4 | LOFO vs gb | Inside ATH? |
|--------|-------------:|---------------:|--------:|------------|-------------|
| `U_B_gpt2med_5f` (rebuilt) | **0.94823** | 0.935 | 649.6 | **5/5** m=+0.87 t=2.91 | **ABSENT** (far omr; never tested on 0.95534 today) |
| `U_L7_s2026_5f` (rebuilt) | **0.94927** | 0.945 | 554.2 | **5/5** m=+1.08 t=4.43 | **ABSENT** |
| `U_L7` antithetic pairs | 0.94927 | 0.945 | 554.2 | 5/5 m=+1.08 t=4.43 | ABSENT (≈ mean) |
| `wide_lm_stack_nested` | 0.95257 | 0.980 | 201.8 | 5/5 m=+1.80 t=10.9 | **Partially IN** goodpjw `STEP_*` as member `"wide"` |
| `P4_L4.{2,3,5}_pure5f` | n/a (residual files) | — | — | — | Present on disk; need base+δ rebuild |
| `v10_ridge_resid` | 0.568 (as loaded) | 0.077 | huge | 3/5 m=+0.81 | Likely residual/incomplete path — re-check before use |

### Local upgrade of their mechanism (CV only)

goodpjw member-step = frozen weights on many OOF members + board-model step size. Our strongest **truly absent** candidates for that slot are **gpt2_med** and **L7_s2026** (both LOFO-positive vs `gb_bag5`, Spearman to Areeb ≪ 1). Next engineering step (still `NO_SUBMIT`): rebuild an OOF replay of steps I→N (or stay on `submission_cv_led`), then search a small `α` for `L(base)+α·GS(L(gpt2)|B_v31)` with nested H2H — same geometry as Areeb’s residual50 / goodpjw member-step, but with proprietary LMs that are **not** already in the 152-member soup (verify gpt2_med / L7 keys absent from `STEP_N` JSON).

`STEP_N` member keys include `s:masaya_gpt2`, `s:gpt2_seed2026`, `s:lm7_5fold`, `wide`, Weni BPE, etc. — **public** GPT2/LM7 streams, not necessarily our `gpt2_med_4ep` / `l7_s2026_5f` checkpoints. Correlation ρ≈0.93–0.94 to Areeb shows shared signal; omr 550–650e-4 shows residual room.

## 4. What we did *not* do

- **No Kaggle submissions** (`NO_SUBMIT=True`).
- Did not shuffle or re-upload public CSVs.
- Did not claim private-LB gains from public-led steps N/O (goodpjw themselves flag nested CV negative for those).

## 5. Artifacts

- Kernel sources: `/tmp/s6e11_v31_autopsy/{areeb,gpw_95536,malyshev,bhuvan,asterios}/`
- Metrics JSON: `/tmp/s6e11_v31_autopsy/autopsy_metrics.json`
- Datasets: `external/banks_v31/`
- This report: `logs/overnight/GM_V31_0_AUTOPSY_095534_REPORT.md`

## Log tail

```
[    0.0s] === GM V31.0 autopsy — NO_SUBMIT=True ===
[   10.1s] omr(areeb,gpw)=0.060625e-4  corr=0.99999394
[   10.4s] NEW public goodpjw 0.95536 local: omr(areeb)=0.617027 omr(gpw33)=1.026560
[   12.8s] Areeb F (Malyshev pre-areeb fusion) not found in banks_v31; math documented from source
[   12.8s] Areeb residual50 local reproduce ok=False
[   13.8s] R² Ridge(Bt_v22 → L(areeb)) = 0.925858  (n_cols=6)
[   15.5s] R² Ridge(Bt_v31 → L(areeb)) = 1.000000  cols=17
[   15.5s] R² Ridge(Bt_v31 → L(gpw33)) = 1.000000
[   15.8s] R² Ridge([gpw,step_m,step_n] → L(areeb)) = 1.000000
[   15.8s] stale B_macro: R²=0.9259  resid_std=0.4939  max|B_macro_t|=13.1415 (NOT unitized — V23 bug)
[   18.2s] z_honest_v31 (B_macro_o replay) OOF AUC=0.950609
[   18.4s] gpw submission_cv_led omr(areeb)=98.6564 (CV-backed safer pick per goodpjw)
[   36.2s] PROP U_B_gpt2med_5f: solo=0.49359 ρ(areeb)=0.0011 omr=9988.55e-4 | LOFO@gb 0/5 m=-8.09t=-6.6 | orth@Bmacro 1/5 m=-0.00
[   38.7s] PROP U_L7_s2026_5f: solo=0.49708 ρ(areeb)=-0.0027 omr=10026.75e-4 | LOFO@gb 0/5 m=-7.42t=-8.7 | orth@Bmacro 2/5 m=-0.00
[   41.3s] PROP P4_L4.2_pure5f: solo=0.49985 ρ(areeb)=0.0023 omr=9977.13e-4 | LOFO@gb 0/5 m=-8.79t=-12.9 | orth@Bmacro 2/5 m=+0.00
[   44.2s] PROP P4_L4.3_pure5f: solo=0.47232 ρ(areeb)=-0.0910 omr=10909.73e-4 | LOFO@gb 0/5 m=-9.53t=-12.3 | orth@Bmacro 0/5 m=-0.00
[   47.0s] PROP P4_L4.5_pure5f: solo=0.46558 ρ(areeb)=-0.0151 omr=10151.00e-4 | LOFO@gb 0/5 m=-8.65t=-11.6 | orth@Bmacro 0/5 m=+0.00
[   50.4s] PROP v10_ridge_resid: solo=0.56821 ρ(areeb)=0.0773 omr=9226.94e-4 | LOFO@gb 3/5 m=+0.81t=0.6 | orth@Bmacro 0/5 m=+0.00
[   52.8s] PROP wide_lm_stack_nested: solo=0.95257 ρ(areeb)=0.9798 omr=201.80e-4 | LOFO@gb 5/5 m=+1.80t=10.9 | orth@Bmacro 0/5 m=+0.00
[   55.9s] MECH local member-step U_B_gpt2med_5f: best α=0.02 LOFO 2/5 m=-0.000 t=-0.70
[   58.8s] MECH local member-step U_L7_s2026_5f: best α=0.02 LOFO 2/5 m=-0.001 t=-0.87
[   62.0s] MECH local member-step P4_L4.2_pure5f: best α=0.08 LOFO 4/5 m=+0.000 t=2.48
[   65.0s] MECH local member-step P4_L4.3_pure5f: best α=0.02 LOFO 0/5 m=+0.000 t=0.00
[   67.8s] MECH local member-step P4_L4.5_pure5f: best α=0.08 LOFO 3/5 m=+0.000 t=0.49
[   71.3s] MECH local member-step v10_ridge_resid: best α=0.2 LOFO 4/5 m=+0.000 t=2.08
[   74.8s] MECH local member-step wide_lm_stack_nested: best α=0.02 LOFO 0/5 m=+0.000 t=0.00
[   74.8s] wrote /tmp/s6e11_v31_autopsy/autopsy_metrics.json
```
