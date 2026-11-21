# GM V16.0 — Morning Recon & Leak Audit

**STRICT `NO_SUBMIT=True` · 0 GPU pods · 2026-11-14 ~04:20–06:30 UTC**

Watcher PID **25580** terminated (already gone). No Kaggle submissions in this step.

---

## Executive verdict

1. **Overnight public LB jumped `0.95388 → 0.95784` (+396 bps)** in a tight UTC-midnight cluster (00:06–00:37Z). Top movers did **not** publish kernels/datasets explaining the leap. No new discussion thread documents a fresh leak recipe overnight.
2. **`daoviet/s6e11-gp-dvae-orthogonal-apex` @ Nov-13 16:52 UTC** replaced prior OOF/test `.npy` + older CSVs with **two submission CSVs only** (`exact_095388` ≈LIVE+1.75e-4; `orthogonal_apex_095370plus` ≈8.72e-4). Not a new orthogonal bank.
3. **Leak audit (id / Time / exact dups / orig):** **negative** for classical id-order, lag-label, and exact train↔test row match. Digit-value TE remains the known synthetic-data path (historical notebook), not a new overnight discovery.
4. **V15 targeting bug confirmed:** Rung1/Rung2 used L2 `unit()` → actual `1−ρ(LIVE) ≈ 3.72e-9` (≈LIVE). Fixed via `unitize()` (z-score). **Family A/E NOSUBMIT** candidates staged at verified `0.10 / 0.20 / 0.35 × 10⁻⁴` (`|err| ≪ 0.005e-4`).

---

## Step 1 — Overnight LB forensics (`0.95388 → 0.95784`)

### Public LB (show) — top of board (as of morning recon)

| Rank | Team | Score | Last submission (UTC) |
|-----:|------|------:|-----------------------|
| 1 | Smart Manoj | **0.95784** | 2026-11-14 **00:37:11** |
| 2 | Dharun MK | **0.95604** | 2026-11-14 **00:25:43** |
| 3 | tine.sh agent | **0.95442** | 2026-11-14 **00:11:56** |
| 4 | Ahmet Bugrahan ALTUNOK | **0.95437** | 2026-11-14 **00:25:57** |
| 5 | srikalyan | **0.95432** | 2026-11-14 **00:06:19** |
| 6 | 강북구집돌이 | 0.95406 | 2026-11-13 14:09:13 |
| 7 | Md. Hamid Hosen | 0.95404 | 2026-11-14 03:32:07 |
| 8 | gold | 0.95400 | 2026-11-13 13:54:08 |
| 9 | Devyanshi_Parihar | 0.95394 | 2026-11-13 16:04:35 |
| 10 | Ravi Ramakrishnan | 0.95394 | 2026-11-13 14:32:10 |
| 11 | Popla69 | 0.95393 | 2026-11-13 18:02:35 |
| 12 | kazuhiro3381 | 0.95392 | 2026-11-14 01:23:01 |
| 13 | aphoristicartist | 0.95391 | 2026-11-14 00:58:19 |
| 14 | Aman Atar | 0.95390 | 2026-11-14 03:54:28 |
| 15 | Chase Hendrick | 0.95389 | 2026-11-14 03:42:04 |
| 16–… | Santiago Maniches, queyrusi, Sanglle, Aditya Goyal, **Moran Zavdi**, … | **0.95388** | various (ours 2026-11-13 20:17:55) |

API returns ~20 rows/page; page 2 is a long **0.95388 plate** (Areeb Ali, daoviet, Knight Fork, …). Submission **counts** are not exposed by `kaggle competitions leaderboard -s`.

### Jump pattern

- **+396 bps** to #1 landed within **~30 min after 00:00 UTC daily reset**.
- Scores `0.95432–0.95784` all have last-submit timestamps in the **00:06–00:37Z** window → coordinated private probes / blends after reset, **not** a slow public kernel race.
- The old ATH shelf (`0.95388`) remains densely occupied; our LIVE (`sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv`) still sits on that shelf.

### Kernels (last ~18–36h) — relevant finds

| Kernel | Last run | Signal |
|--------|----------|--------|
| `asteriospanoras/s6e11-how-much-of-the-public-top-is-real` | Nov-14 03:22Z | Public LB = 20% noise / winner’s curse; nested CV for private survival |
| `jamaluddin99/s6e11-rank-blend-of-top-public-lb-0-95384` | Nov-13 08:13Z | Honest rank-blend of public files → **0.95384**; weaker files don’t beat best single |
| `dhananjayk2609/s6e11-fraud-part-2-digits-leak-the-label` | (series) | **Digit / cents / leading-digit TE** lifts OOF ~0.947→~0.950 — synthetic generative artifact, **not new overnight** |
| `areebbb/s6e11-bold-then-bolder-lb-0-95388` | Nov-13 09:49Z | Public 0.95388 ladder (pre-jump) |
| `daoviet/s6e11-on-the-journey-to-your-first-top-100` | Nov-13 16:53Z | Paired with dataset update 16:52Z |
| `goodpjw2008/...0-95386` | Nov-13 07:22Z | Still the strong public LM ladder reference |

**No kernel from Smart Manoj / Dharun MK / tine.sh agent / srikalyan** explaining `0.954+`. Top jump looks **private**.

### Discussions (API `/competitions/.../topics`)

Recent threads (none announcing a 0.957 breakthrough):

- “data cleaning” (Nov-13)
- “Is your CV supporting the rise in leaderboard score?”
- “Probing the public board, simulated on OOF…”
- “0.95347 reference vs our own models…”
- Older: “Sharing Code for 500 Models!”

No overnight sticky leak dump.

---

## Step 2 — Dataset pull & `daoviet` 16:52 UTC diff

### Watched / fresh `s6e11` datasets (`datasets list -s s6e11 --sort-by updated`)

| Dataset | lastUpdated (UTC) | Note |
|---------|-------------------|------|
| **`daoviet/s6e11-gp-dvae-orthogonal-apex`** | **2026-11-13 16:52:04** | **This update** |
| `alexchilton/s6e11-oof-predictions` | 2026-11-13 08:51 | 15-model OOF bank (unchanged overnight) |
| `amanatar/s6e11-rank1-q17-meta-bundle` | 2026-11-12 | older |
| `goodpjw2008/s6e11-target-first-gpt2-seed-2026` | 2026-11-12 | older |
| others | ≤ Nov-12 | no newer “smoking gun” dataset in last 24h beyond daoviet |

Pulled into `external/nov14_recon/daoviet_apex_1652/`. Prior pull mirrored from `external/nov13_recon/daoviet_gp_dvae_apex_nov13/` → `external/nov14_recon/prior_daoviet/`.

### File-level diff (prior 05:29-era pull vs 16:52 UTC)

| Status | File | Size | SHA256 (prefix) |
|--------|------|-----:|-----------------|
| **REMOVED** | `oof_ours.npy` | 2,799,968 | `91be476ddb00…` |
| **REMOVED** | `test_ours.npy` | 1,200,064 | `96f3bdde816e…` |
| **REMOVED** | `submission_095351.csv` | 5,999,693 | `f16b86b3bc0a…` |
| **REMOVED** | `submission_095354_exact.csv` | 8,180,417 | `92e3acbe5a0d…` |
| **REMOVED** | `submission_summit_095370.csv` | 6,274,032 | `927e8583e696…` |
| **REMOVED** | `submission_verified_095370.csv` | 8,180,417 | `b5e82aaec6ac…` |
| **ADDED** | `submission_exact_095388.csv` | 7,880,432 | `b09796945412…` |
| **ADDED** | `submission_orthogonal_apex_095370plus.csv` | 8,180,417 | `1fa5e78cf614…` |

### Metrics vs our LIVE (`sub_v13_slot2_bold_plus_d1_w0.03_LIVE.csv`)

| File | `1 − spearmanr(LIVE)` | Notes |
|------|----------------------:|-------|
| `submission_exact_095388.csv` | **1.748e-4** | Near-LIVE public 0.95388 clone / slight step |
| `submission_orthogonal_apex_095370plus.csv` | **8.725e-4** | Larger displacement; no OOF → no AUC/R² on Bt |
| prior `oof_ours.npy` (archived) | — | **OOF AUC = 0.94649** on train; `.npy` removed from public dataset at 16:52 |

**Conclusion:** 16:52 update is a **submission-swap / cleanup**, not a new GP/DVAE orthogonal feature dump. Cannot recompute OOF AUC or `R²` on Bt for the new files (CSV test preds only).

Kernels / notebooks archived under `external/nov14_recon/kernels/` and `/tmp/s6e11_v16/kernels/`.

---

## Step 3 — Fast leak diagnostics (`train` / `test` / `orig20k`)

| Hypothesis | Result |
|------------|--------|
| `pointbiserial(is_fraud, id)` | **−6.4e-4** (noise) |
| `id % K` (K=2…100) | \|r\| ≤ **2e-3** |
| Adjacent label as score (`y[i±1]` by id order) | AUC **≈0.4996** (no autocorr) |
| `hours_since_last_txn` | pb ≈ **−0.020** (weak feature, not leak) |
| `time_of_day_hour` / `day_of_week` | ≈0 / +0.0037 |
| Exact duplicate feature rows in train | **0** |
| Exact train↔test feature-hash matches | **0 / 299984 (0%)** |
| Exact train↔orig on 24 shared cols | **0** |
| Exact test↔orig on 24 shared cols | **0 / 299984 (0%)** |

**Why +400 bps then?** Most consistent with **private blends / aggressive public probing after daily reset** on a noisy 20% public slice (asterios thesis), possibly plus stronger digit-TE / proprietary stacks — **not** a simple `id`/`Time`/`exact-dup` leak we can reproduce from CSV columns alone. Digit-value TE remains the known synthetic edge (Part-2 notebook), already in the public meta-game below 0.954.

---

## Step 4 — V15 targeting bug + staged Family A/E (`NO_SUBMIT`)

### Exact bug

In overnight V15, `build_rung` applied **L2** `unit(v) = v / ‖v‖₂`. For a typical direction with `‖v‖₂ ~ O(√n)·σ`, the L2-unit vector has **tiny per-row std (~0.0018)**. With bisection weights `w ≲ 0.08–0.22`, the logit-space step was negligible after `quantile_lock` → submissions were **numerically LIVE**.

**Measured on written rungs:**

| File | Intended `1−ρ` | **Actual `1−ρ(LIVE)`** |
|------|---------------:|-----------------------:|
| `sub_v15_LIVE_rung1_…0p08e4.csv` | 0.080e-4 | **3.720e-9** |
| `sub_v15_LIVE_rung2_…0p22e4.csv` | 0.220e-4 | **3.716e-9** |

Both scored **0.95388** on LB (identical to LIVE within noise).

### Fix (already in `scripts/gm_v15_ultimate_8hr_autonomous_ladder.py`)

```python
def unitize(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    return (v - v.mean()) / (v.std() + 1e-12)

# build_rung:
u = unitize(direction_test)   # NOT unit() / L2
raw = L(live) + mid * u
cand = quantile_lock(raw, live)
```

Comment in code: *Direction must be unitize()'d (z-scored), NOT L2-unit — L2 unit makes 1−ρ tiny at w\<0.1.*

### Staged verified NOSUBMIT candidates (L7_s2026_5f winners)

Gate0 winners from V15:

- **Family E** — Tail-Weighted 5-Fold 2-Seed L7 Swap — **5/5 main, 19/25 sf, t=3.14**
- **Family A** — Pure 5-Fold 2-Seed L7 Swap — **5/5 main, 17/25 sf, t=2.96**

Assert: `abs(actual_1m_rho − target) < 0.005e-4` before save.

| File | Family | Target `1−ρ` | Actual `1−ρ` | \|err\| | `w` |
|------|--------|-------------:|-------------:|-------:|----:|
| `submissions/sub_v16_0_FamilyA_0p10e4_NOSUBMIT.csv` | A | 0.100e-4 | 1.000000e-05 | 6.1e-14 | 0.00757 |
| `submissions/sub_v16_0_FamilyA_0p20e4_NOSUBMIT.csv` | A | 0.200e-4 | 2.000000e-05 | 5.8e-13 | 0.01070 |
| `submissions/sub_v16_0_FamilyA_0p35e4_NOSUBMIT.csv` | A | 0.350e-4 | 3.500000e-05 | 1.1e-12 | 0.01416 |
| `submissions/sub_v16_0_FamilyE_0p10e4_NOSUBMIT.csv` | E | 0.100e-4 | 1.000000e-05 | 2.1e-13 | 0.00739 |
| `submissions/sub_v16_0_FamilyE_0p20e4_NOSUBMIT.csv` | E | 0.200e-4 | 2.000000e-05 | 6.9e-13 | 0.01045 |
| `submissions/sub_v16_0_FamilyE_0p35e4_NOSUBMIT.csv` | E | 0.350e-4 | 3.500000e-05 | 3.3e-13 | 0.01383 |

Machine JSON: `logs/overnight/v16_0_recon.json`.  
Script stub: `scripts/gm_v16_0_lb_breakthrough_and_nov14_recon.py` (`NO_SUBMIT=True`).

**Not submitted.** Authorize live upload separately.

---

## Artifacts

```
external/nov14_recon/
  daoviet_apex_1652/          # 16:52 UTC pull
  prior_daoviet/              # pre-16:52 archive (npy + old csv)
  lb_top40.csv / lb_page2.csv / datasets_s6e11.txt
  kernels/                    # asterios, jamaluddin, digits-leak, …
submissions/sub_v16_0_Family{A,E}_{0p10,0p20,0p35}e4_NOSUBMIT.csv
logs/overnight/v16_0_recon.json
logs/overnight/GM_V16_0_MORNING_RECON_AND_LEAK_AUDIT.md   # this file
```

## Recommended next (when submit unlocked)

1. Prefer **Family E @ 0.10e-4** then **0.20e-4** (stronger Gate0) — one live probe after reset, stop-loss if ≤LIVE.
2. Do **not** chase 0.95784 with large `1−ρ` steps without private CV support (asterios / winner’s curse).
3. Treat daoviet 16:52 as LIVE-adjacent public files only; keep using our L7_s2026_5f OOF bank as the controlled direction.
