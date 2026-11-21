# GM V15 — Ultimate 8hr Autonomous Ladder

**Updated:** 2026-11-13T20:21:08.231040+00:00  
**Pods:** 0 · **NO GPU** · staged `/tmp/s6e11_v15/` · originals in `preds/l7_s2026_5f/` untouched

## Executive result

| Item | Result |
|---|---|
| Gate0 winner | **Family A (`A_L7swap`)** — also E_tail cleared |
| Rung1 live | **0.95388** (tied ATH) — ⚠️ targeting bug made 1-ρ≈0 (≈LIVE copy) |
| Rung2 live | **0.95388** (tied) — same bug |
| Rung3 | **skipped** (tied but >45m to 00:00 UTC; stop-loss N/A) |
| Bug fix | `build_rung` now uses **`unitize`** not L2 `unit` (w*≈0.0068 for 0.080e-4) |
| Overnight | G26 watcher started — **no submit** unless `gpt2_s2026_test_5fold.npy` lands |

## Phase 1 — Antithetic 5-fold

- Shapes OK · 0 NaN/Inf · min pairwise Spearman(test)=**0.991638** (>0.975)
- Assembled: `preds/l7_s2026_5f/l7_s2026_{oof,test}_5fold_raw.npy`
- R1≈R3≈R4 (ranklogit / antithetic / trim3); R2 Bayes slightly lower mean
- **2-seed LOFO vs L7F: 5/5 folds +** Δbps=[5.06, 4.86, 3.67, 6.59, 2.11] γ*=[0.45, 0.45, 0.45, 0.45, 0.5]

### Reduction table (mean / pooled)

- `L7F_s7`: folds=[0.947545, 0.949767, 0.955166, 0.949065, 0.945676] mean=0.949444 pooled=0.949448
- `LM`: folds=[0.948506, 0.950682, 0.956085, 0.950604, 0.946381] mean=0.950452 pooled=0.950428
- `R1`: folds=[0.947517, 0.949662, 0.954906, 0.949395, 0.945015] mean=0.949299 pooled=0.949299
- `R2`: folds=[0.947493, 0.949578, 0.954835, 0.94933, 0.944992] mean=0.949246 pooled=0.949212
- `R3`: folds=[0.947517, 0.949662, 0.954906, 0.949395, 0.945015] mean=0.949299 pooled=0.949299
- `R4`: folds=[0.947516, 0.949663, 0.954906, 0.949399, 0.945028] mean=0.949302 pooled=0.949303

## Phase 2 — LIVE basis + families

- LIVE ~ [L7F_5f, lm7_f0, d1_v4] R²=0.9285 β={"L7F_5f": 1.0046, "lm7_f0": -0.0399, "d1_v4": -0.0589, "intercept": 0.0}
- Novelty corr²(LIVE_resid, L(s2026)-L(L7F))=**0.0127** (<0.02 ✓)
- ATH OOF proxy used for screen: L(l7f)+0.03·d1 (AUC≈0.94956) — conservative vs true 0.95388 OOF

| Family | pos/5 | meanΔ e-5 | sf25 | t | disp | Gate0 |
|---|---|---|---|---|---|---|
| A_L7swap | 5/5 @0.025 | 0.022 | 17/25 | 2.96 | 1.000 | True |
| B_f0_to_5f_nat | 5/5 @0.025 | 0.022 | 17/25 | 2.96 | 1.564 | False |
| C_LMswap | 5/5 @0.025 | 0.017 | 13/25 | 1.44 | 1.000 | False |
| D_joint065_035 | 5/5 @0.02 | 0.006 | 12/25 | 0.80 | 1.000 | False |
| E_tail | 5/5 @0.025 | 0.026 | 19/25 | 3.14 | 1.000 | True |

**Winner:** `A_L7swap`

## Phase 3 — Live ladder

1. `sub_v15_LIVE_rung1_L7s2026_5f_0p08e4.csv` → **0.95388** (intended 0.080e-4; actual 1-ρ≈0 due to L2-unit bug)
2. `sub_v15_LIVE_rung2_L7s2026_5f_plus_f0fix_0p22e4.csv` → **0.95388** (same)
3. No Rung3 (tied ATH, ~223m to reset >45m)

**Quota:** 5 remaining after Rung2 (per CLI). Pre-reset stop-loss not triggered (neither ≤0.95387).

## Phase 4 — Watcher

Poll every 25m for `gpt2_s2026_test_5fold.npy`. Dataset currently only has `gpt2_s2026_test_f0.npy` + OOF.

## Artifacts

- Script: `scripts/gm_v15_ultimate_8hr_autonomous_ladder.py`
- Staging: `/tmp/s6e11_v15/` · state JSON under `out/`
- Home backup still: `~/kaggle_backups/l7_s2026_5f_v14/`

---

## Live append log

# GM V15 — Ultimate 8hr Autonomous Ladder

**NO GPU pods · stop-loss live ladder · G26 watcher**

[20:13:55Z] START V15 cwd-tmp=/tmp/s6e11_v15 dry_run=False
[20:13:56Z] === PHASE 1: assemble + antithetic reductions ===
[20:13:56Z]   fold 0: oof/test OK finite
[20:13:56Z]   fold 1: oof/test OK finite
[20:13:56Z]   fold 2: oof/test OK finite
[20:13:57Z]   fold 3: oof/test OK finite
[20:13:57Z]   fold 4: oof/test OK finite
[20:14:09Z]   wrote /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/l7_s2026_5f/l7_s2026_oof_5fold_raw.npy
[20:14:40Z]   wrote /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/preds/l7_s2026_5f/l7_s2026_test_5fold_raw.npy
[20:14:45Z] 5x5 Spearman (bayes test):
[[1.     0.9926 0.9921 0.992  0.9922]
 [0.9926 1.     0.9923 0.9922 0.9926]
 [0.9921 0.9923 1.     0.9916 0.9922]
 [0.992  0.9922 0.9916 1.     0.9921]
 [0.9922 0.9926 0.9922 0.9921 1.    ]]
[20:14:45Z]   min pairwise ρ=0.991638 > 0.975 PASS
[20:14:54Z] R5 weights (α=25.0): [0.2006 0.2002 0.1994 0.2024 0.1974]  ΔAUC_bps=[-0.27 -1.05 -2.61  3.3  -6.61]
[20:14:56Z]   L7F_s7: folds=[0.947545, 0.949767, 0.955166, 0.949065, 0.945676] mean=0.949444 pooled=0.949448
[20:14:57Z]   LM: folds=[0.948506, 0.950682, 0.956085, 0.950604, 0.946381] mean=0.950452 pooled=0.950428
[20:14:57Z]   R1: folds=[0.947517, 0.949662, 0.954906, 0.949395, 0.945015] mean=0.949299 pooled=0.949299
[20:14:57Z]   R2: folds=[0.947493, 0.949578, 0.954835, 0.94933, 0.944992] mean=0.949246 pooled=0.949212
[20:14:58Z]   R3: folds=[0.947517, 0.949662, 0.954906, 0.949395, 0.945015] mean=0.949299 pooled=0.949299
[20:14:58Z]   R4: folds=[0.947516, 0.949663, 0.954906, 0.949399, 0.945028] mean=0.949302 pooled=0.949303
[20:15:03Z]   2seed γ=0.4: 5/5  Δbps=[4.92, 4.73, 3.67, 6.26, 2.6] mean=4.44
[20:15:08Z]   2seed γ=0.45: 5/5  Δbps=[5.05, 4.86, 3.67, 6.57, 2.4] mean=4.51
[20:15:14Z]   2seed γ=0.5: 5/5  Δbps=[5.07, 4.88, 3.59, 6.78, 2.11] mean=4.49
[20:15:19Z]   2seed γ=0.55: 5/5  Δbps=[5.01, 4.8, 3.41, 6.92, 1.73] mean=4.37
[20:15:24Z]   2seed LOFO γ*=[0.45, 0.45, 0.45, 0.45, 0.5] Δbps=[5.06, 4.86, 3.67, 6.59, 2.11] pos=5/5
[20:15:25Z] MEM phase1: rss≈0.98 GiB
[20:15:25Z] === PHASE 2: LIVE basis + family screen ===
[20:15:25Z]   ATH OOF proxy AUC=0.949558 (L(l7f)+0.03*d1)
[20:15:26Z]   LIVE~[['L7F_5f', 'lm7_f0', 'd1_v4']] R²=0.9285 β={'L7F_5f': 1.0046, 'lm7_f0': -0.0399, 'd1_v4': -0.0589, 'intercept': 0.0}
[20:15:26Z]   corr²(LIVE_resid, L(s2026)-L(L7F))=0.0127 (want <0.02 for novel)
[20:15:28Z]   A_L7swap: pos=5/5 @0.025 meanΔ=0.022e-5 sf=17/25 t=2.96 disp=1.000 gate0=True
[20:15:29Z]   B_f0_to_5f_nat: pos=5/5 @0.025 meanΔ=0.022e-5 sf=17/25 t=2.96 disp=1.564 gate0=False
[20:15:30Z]   C_LMswap: pos=5/5 @0.025 meanΔ=0.017e-5 sf=13/25 t=1.44 disp=1.000 gate0=False
[20:15:32Z]   D_joint065_035: pos=5/5 @0.02 meanΔ=0.006e-5 sf=12/25 t=0.80 disp=1.000 gate0=False
[20:15:33Z]   E_tail: pos=5/5 @0.025 meanΔ=0.026e-5 sf=19/25 t=3.14 disp=1.000 gate0=True
[20:15:33Z]   Gate0 winner: A_L7swap
[20:15:34Z] MEM phase2: rss≈0.98 GiB
[20:15:34Z] V15 main finished
[20:15:55Z] START V15 cwd-tmp=/tmp/s6e11_v15 dry_run=False
[20:15:55Z] === PHASE 1: assemble + antithetic reductions ===
[20:15:55Z]   fold 0: oof/test OK finite
[20:15:56Z]   fold 1: oof/test OK finite
[20:15:56Z]   fold 2: oof/test OK finite
[20:15:56Z]   fold 3: oof/test OK finite
[20:15:57Z]   fold 4: oof/test OK finite
[20:15:57Z]   keep existing l7_s2026_oof_5fold_raw.npy
[20:15:58Z]   keep existing l7_s2026_test_5fold_raw.npy
[20:15:59Z] 5x5 Spearman (bayes test):
[[1.     0.9926 0.9921 0.992  0.9922]
 [0.9926 1.     0.9923 0.9922 0.9926]
 [0.9921 0.9923 1.     0.9916 0.9922]
 [0.992  0.9922 0.9916 1.     0.9921]
 [0.9922 0.9926 0.9922 0.9921 1.    ]]
[20:16:00Z]   min pairwise ρ=0.991638 > 0.975 PASS
[20:16:09Z] R5 weights (α=25.0): [0.2006 0.2002 0.1994 0.2024 0.1974]  ΔAUC_bps=[-0.27 -1.05 -2.61  3.3  -6.61]
[20:16:10Z]   L7F_s7: folds=[0.947545, 0.949767, 0.955166, 0.949065, 0.945676] mean=0.949444 pooled=0.949448
[20:16:11Z]   LM: folds=[0.948506, 0.950682, 0.956085, 0.950604, 0.946381] mean=0.950452 pooled=0.950428
[20:16:12Z]   R1: folds=[0.947517, 0.949662, 0.954906, 0.949395, 0.945015] mean=0.949299 pooled=0.949299
[20:16:12Z]   R2: folds=[0.947493, 0.949578, 0.954835, 0.94933, 0.944992] mean=0.949246 pooled=0.949212
[20:16:12Z]   R3: folds=[0.947517, 0.949662, 0.954906, 0.949395, 0.945015] mean=0.949299 pooled=0.949299
[20:16:13Z]   R4: folds=[0.947516, 0.949663, 0.954906, 0.949399, 0.945028] mean=0.949302 pooled=0.949303
[20:16:18Z]   2seed γ=0.4: 5/5  Δbps=[4.92, 4.73, 3.67, 6.26, 2.6] mean=4.44
[20:16:22Z]   2seed γ=0.45: 5/5  Δbps=[5.05, 4.86, 3.67, 6.57, 2.4] mean=4.51
[20:16:26Z]   2seed γ=0.5: 5/5  Δbps=[5.07, 4.88, 3.59, 6.78, 2.11] mean=4.49
[20:16:31Z]   2seed γ=0.55: 5/5  Δbps=[5.01, 4.8, 3.41, 6.92, 1.73] mean=4.37
[20:16:36Z]   2seed LOFO γ*=[0.45, 0.45, 0.45, 0.45, 0.5] Δbps=[5.06, 4.86, 3.67, 6.59, 2.11] pos=5/5
[20:16:37Z] MEM phase1: rss≈0.86 GiB
[20:16:37Z] === PHASE 2: LIVE basis + family screen ===
[20:16:38Z]   ATH OOF proxy AUC=0.949558 (L(l7f)+0.03*d1)
[20:16:39Z]   LIVE~[['L7F_5f', 'lm7_f0', 'd1_v4']] R²=0.9285 β={'L7F_5f': 1.0046, 'lm7_f0': -0.0399, 'd1_v4': -0.0589, 'intercept': 0.0}
[20:16:40Z]   corr²(LIVE_resid, L(s2026)-L(L7F))=0.0127 (want <0.02 for novel)
[20:16:42Z]   A_L7swap: pos=5/5 @0.025 meanΔ=0.022e-5 sf=17/25 t=2.96 disp=1.000 gate0=True
[20:16:43Z]   B_f0_to_5f_nat: pos=5/5 @0.025 meanΔ=0.022e-5 sf=17/25 t=2.96 disp=1.564 gate0=False
[20:16:44Z]   C_LMswap: pos=5/5 @0.025 meanΔ=0.017e-5 sf=13/25 t=1.44 disp=1.000 gate0=False
[20:16:46Z]   D_joint065_035: pos=5/5 @0.02 meanΔ=0.006e-5 sf=12/25 t=0.80 disp=1.000 gate0=False
[20:16:47Z]   E_tail: pos=5/5 @0.025 meanΔ=0.026e-5 sf=19/25 t=3.14 disp=1.000 gate0=True
[20:16:49Z]   Gate0 winner: A_L7swap
[20:16:50Z] MEM phase2: rss≈0.98 GiB
[20:16:50Z] === PHASE 3: closed-loop live rungs ===
[20:16:51Z]   minutes to 00:00 UTC reset: 223.1
[20:17:02Z]   built sub_v15_LIVE_rung1_L7s2026_5f_0p08e4.csv w=0.08000 1-ρ=3.719996e-09 target=8.000000e-06
[20:17:03Z] KAGGLE exec: -m kaggle competitions submissions -c playground-series-s6e11 (timeout=90)
[20:17:05Z] KAGGLE exec: -c playground-series-s6e11 -f /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v15_LIVE_rung1_L7s2026_5f_0p08e4.csv -m V15 Rung1: A_L7swap @1-ρ=0.080e-4 gate0 pass (5f antithetic L7_s2026) (timeout=90)
[20:17:11Z] submit stdout: Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
6 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[20:17:11Z] submit stderr: 
  0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<03:04, 42.5kB/s]
  2%|▏         | 144k/7.52M [00:00<00:24, 321kB/s]  
  8%|▊         | 624k/7.52M [00:00<00:06, 1.06MB/s]
 10%|▉         | 752k/7.52M [00:00<00:07, 942kB/s] 
 12%|█▏        | 944k/7.52M [00:01<00:07, 925kB/s]
 15%|█▌        | 1.14M/7.52M [00:01<00:06, 1.00MB/s]
 28%|██▊       | 2.14M/7.52M [00:01<00:02
[20:17:12Z] KAGGLE exec: -m kaggle competitions submissions -c playground-series-s6e11 (timeout=90)
[20:17:13Z] poll sub_v15_LIVE_rung1_L7s2026_5f_0p08e4.csv: 58320228  sub_v15_LIVE_rung1_L7s2026_5f_0p08e4.csv                        2026-11-13 20:17:09.440000  V15 Rung1: A_L7swap @1-ρ=0.080e-4 gate0 pass (5f antitheti
[20:17:39Z] MEM poll: rss≈0.98 GiB
[20:17:40Z] KAGGLE exec: -m kaggle competitions submissions -c playground-series-s6e11 (timeout=90)
[20:17:41Z] SCORE sub_v15_LIVE_rung1_L7s2026_5f_0p08e4.csv = 0.95388
[20:17:50Z]   built sub_v15_LIVE_rung2_L7s2026_5f_plus_f0fix_0p22e4.csv w=0.08000 1-ρ=3.715927e-09 target=2.200000e-05
[20:17:50Z] KAGGLE exec: -m kaggle competitions submissions -c playground-series-s6e11 (timeout=90)
[20:17:52Z] KAGGLE exec: -c playground-series-s6e11 -f /Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/submissions/sub_v15_LIVE_rung2_L7s2026_5f_plus_f0fix_0p22e4.csv -m V15 Rung2 tied: 0.7A+0.3B @0.220e-4 after R1=0.95388 (timeout=90)
[20:17:57Z] submit stdout: Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
5 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[20:17:57Z] submit stderr: 
  0%|          | 0.00/7.52M [00:00<?, ?B/s]
  2%|▏         | 160k/7.52M [00:00<00:04, 1.56MB/s]
  7%|▋         | 560k/7.52M [00:00<00:06, 1.21MB/s]
  9%|▉         | 688k/7.52M [00:00<00:06, 1.08MB/s]
 11%|█         | 864k/7.52M [00:00<00:06, 1.00MB/s]
 44%|████▎     | 3.28M/7.52M [00:00<00:00, 6.23MB/s]
 83%|████████▎ | 6.25M/7.52M [00:01<00:00, 12.0MB/s]
100%|██████████| 7.52M/7.52M [00:01<00:00
[20:17:57Z] KAGGLE exec: -m kaggle competitions submissions -c playground-series-s6e11 (timeout=90)
[20:17:59Z] poll sub_v15_LIVE_rung2_L7s2026_5f_plus_f0fix_0p22e4.csv: 58320243  sub_v15_LIVE_rung2_L7s2026_5f_plus_f0fix_0p22e4.csv             2026-11-13 20:17:55.413000  V15 Rung2 tied: 0.7A+0.3B @0.220e-4 after R1=0.95388      
[20:18:24Z] MEM poll: rss≈0.98 GiB
[20:18:24Z] KAGGLE exec: -m kaggle competitions submissions -c playground-series-s6e11 (timeout=90)
[20:18:25Z] SCORE sub_v15_LIVE_rung2_L7s2026_5f_plus_f0fix_0p22e4.csv = 0.95388
[20:18:25Z] No Rung3 condition met (tied but >=45m to reset, or no ATH)
[20:18:25Z] === PHASE 4: G26 overnight watcher ===
[20:18:25Z] --- watcher check #1 ---
[20:18:26Z] MEM watcher: rss≈0.98 GiB
[20:18:26Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[20:18:27Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[20:18:29Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[20:18:30Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[20:18:30Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[20:18:31Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[20:19:31Z] Phase4 watcher done
[20:19:32Z] V15 main finished
[20:21:09Z] START V15 cwd-tmp=/tmp/s6e11_v15 dry_run=False
[20:21:10Z] === PHASE 4: G26 overnight watcher ===
[20:21:11Z] --- watcher check #1 ---
[20:21:12Z] MEM watcher: rss≈0.22 GiB
[20:21:12Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[20:21:14Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[20:21:15Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[20:21:18Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[20:21:19Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[20:21:26Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[20:46:53Z] --- watcher check #2 ---
[20:46:53Z] MEM watcher: rss≈0.22 GiB
[20:46:53Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[20:46:57Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[20:46:58Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[20:46:59Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[20:47:01Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[20:47:03Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[21:12:05Z] --- watcher check #3 ---
[21:12:06Z] MEM watcher: rss≈0.22 GiB
[21:12:06Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[21:12:07Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[21:12:07Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[21:12:09Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[21:12:09Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[21:12:10Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[21:37:10Z] --- watcher check #4 ---
[21:37:12Z] MEM watcher: rss≈0.22 GiB
[21:37:12Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[21:37:13Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[21:37:13Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[21:37:15Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[21:37:15Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[21:37:16Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[22:02:16Z] --- watcher check #5 ---
[22:02:17Z] MEM watcher: rss≈0.22 GiB
[22:02:17Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[22:02:18Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[22:02:19Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[22:02:19Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[22:02:19Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[22:02:20Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[22:27:21Z] --- watcher check #6 ---
[22:27:21Z] MEM watcher: rss≈0.22 GiB
[22:27:21Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[22:27:22Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[22:27:22Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[22:27:23Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[22:27:23Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[22:27:24Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[22:52:24Z] --- watcher check #7 ---
[22:52:25Z] MEM watcher: rss≈0.22 GiB
[22:52:25Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[22:52:26Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[22:52:26Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[22:52:27Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[22:52:27Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[22:52:28Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[23:17:28Z] --- watcher check #8 ---
[23:17:28Z] MEM watcher: rss≈0.22 GiB
[23:17:28Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[23:17:30Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[23:17:30Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[23:17:30Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[23:17:31Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[23:17:31Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[23:42:31Z] --- watcher check #9 ---
[23:42:32Z] MEM watcher: rss≈0.22 GiB
[23:42:32Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[23:42:33Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[23:42:33Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[23:42:34Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[23:42:34Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[23:42:34Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[00:07:35Z] --- watcher check #10 ---
[00:07:35Z] MEM watcher: rss≈0.22 GiB
[00:07:35Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[00:07:36Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[00:07:36Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[00:07:38Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[00:07:38Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[00:07:38Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[00:32:38Z] --- watcher check #11 ---
[00:32:38Z] MEM watcher: rss≈0.22 GiB
[00:32:38Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[00:32:40Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[00:32:40Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[00:32:41Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[00:32:41Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[00:32:41Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[00:57:41Z] --- watcher check #12 ---
[00:57:41Z] MEM watcher: rss≈0.22 GiB
[00:57:41Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[00:57:42Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[00:57:42Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[00:57:44Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[00:57:44Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[00:57:44Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[01:22:44Z] --- watcher check #13 ---
[01:22:45Z] MEM watcher: rss≈0.22 GiB
[01:22:45Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[01:22:46Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[01:22:46Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[01:22:47Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[01:22:47Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[01:22:47Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[01:47:48Z] --- watcher check #14 ---
[01:47:48Z] MEM watcher: rss≈0.22 GiB
[01:47:48Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[01:47:49Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[01:47:49Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[01:47:51Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[01:47:51Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[01:47:51Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[02:12:52Z] --- watcher check #15 ---
[02:12:52Z] MEM watcher: rss≈0.22 GiB
[02:12:52Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[02:12:53Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[02:12:53Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[02:12:54Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[02:12:54Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[02:12:54Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[02:37:55Z] --- watcher check #16 ---
[02:37:55Z] MEM watcher: rss≈0.22 GiB
[02:37:55Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[02:37:55Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[02:37:56Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[02:37:57Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[02:37:57Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[02:37:57Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[03:02:58Z] --- watcher check #17 ---
[03:02:58Z] MEM watcher: rss≈0.22 GiB
[03:02:58Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[03:02:59Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[03:02:59Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[03:03:00Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[03:03:00Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[03:03:01Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[03:28:01Z] --- watcher check #18 ---
[03:28:02Z] MEM watcher: rss≈0.22 GiB
[03:28:02Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[03:28:03Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[03:28:03Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[03:28:05Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[03:28:05Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[03:28:05Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[03:53:05Z] --- watcher check #19 ---
[03:53:05Z] MEM watcher: rss≈0.22 GiB
[03:53:05Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[03:53:07Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[03:53:07Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[03:53:08Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[03:53:08Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[03:53:08Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

[04:18:08Z] --- watcher check #20 ---
[04:18:08Z] MEM watcher: rss≈0.22 GiB
[04:18:09Z] KAGGLE exec: /private/tmp/s6e11_v15/venv/bin/python -m kaggle datasets files goodpjw2008/s6e11-target-first-gpt2-seed-2026 (timeout=60)
[04:18:10Z] datasets files:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
name                         size  creationDate                
------------------------  -------  --------------------------  
README.md                    1343  2026-11-12 09:14:45.794000  
gpt2_s2026_oof_5fold.npy  5599808  2026-11-12 09:14:46.075000  
gpt2_s2026_test_f0.npy    2400000  2026-11-12 09:14:45.915000  

[04:18:10Z] KAGGLE exec: datasets list -s s6e11 --sort-by updated (timeout=60)
[04:18:11Z] datasets list:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                     title                                                  size  lastUpdated                 downloadCount  voteCount  usabilityRating  
------------------------------------------------------  -----------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
daoviet/s6e11-gp-dvae-orthogonal-apex                   S6E11 GP and DVAE Orthogonal
[04:18:12Z] KAGGLE exec: -c playground-series-s6e11 --sort-by scoreDescending --page-size 20 (timeout=60)
[04:18:13Z] kernels top:
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)

