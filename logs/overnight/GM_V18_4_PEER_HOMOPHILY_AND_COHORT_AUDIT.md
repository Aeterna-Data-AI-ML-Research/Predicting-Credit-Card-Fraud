# GM V18.4 — Peer Homophily + Cohort-Rank Residual on `r_S5`

**STRICT `NO_SUBMIT=True` · 0 GPU pods · local CPU (~58s)**  
Script: `scripts/gm_v18_4_peer_homophily_and_cohort_rank_residual.py`  
Artifacts: `/tmp/s6e11_v16/out/v18_4.json`, `v18_4_peer.npz`

Column proxies: `hours_since_last_txn`; `is_foreign_transaction`; `auth_method` (no `card_present` in data).

---

## Verdict

**No stage.** Peer/cohort family is **nearly orthogonal to `R2_v178`** (`ρ ≈ +0.019`) and shows **huge OOF lifts** (+17…+30 e-5, 5/5×3, sf25, t>12), but **fails F2** via leak into **`Family_E`** (and linear also fails F3). Do not submit.

| Candidate | preρ(z_S5) | ρ(R2) | F1 | F2 | F3 | Elite S5? | Stage |
|---|---|---|---|---|---|---|---|
| **R3_peer_s42** | +0.096 | **+0.019** | ✓ 0.996 | ✗ **0.048→Family_E** | ✓ | 5/5×3 sf25 | ✗ |
| **R3_peer_bag** | +0.113 | +0.020 | ✓ | ✗ **0.053→Family_E** | ✓ | 5/5×3 sf25 | ✗ |
| **R3_peer_linear** | +0.066 | +0.023 | ✓ | ✗ **0.086→Family_E** | ✗ +0.089 | 5/5×3 | ✗ |

---

## Step 1 — `X_peer` (48 cols, `N=999,944`, **zero `y`**)

`z_all = concat[unitize(z_S5_oof), unitize(L(Slot5_test))]`

### Group A — leave-self-out peer (`m0=5`)

| Key | mean count | max count |
|---|---|---|
| k1 amount | 409 | 2747 |
| k2 balance | 170 | 1442 |
| k3 velocity | 7385 | 69426 |
| k4 mrisk | 2435 | 4973 |
| k5 distance | 659 | 2109 |
| k6 mer×amt_int | 2885 | 7069 |
| k7 mer×ch×dev×cvv×foreign×auth | 1563 | 7934 |
| k8 (amt_cents, bal_cents, mer) | **16.3** | 67 |

Features per key: `peer_z`, `peer_diff`, `peer_logcnt`.

### Groups B/C

Within-`merchant` and within-`(mer,ch,dev)` percentile ranks + G2−global diffs; `drain_ratio`, rank diffs/sums; `abs_z_S5`, `z_S5_rank`.

---

## Step 2 — Fit on `g_S5` / `h_S5`

LGB: `num_leaves=6`, `n_estimators=40`, `min_data=500`, `lr=0.03` · s42 + bag `[42,2026,7]` · Ridge on 8 `peer_z` only.

### Top 15 gain (`R3_peer_s42`)

| # | Feature | Gain |
|---|---|---|
| 1 | **`peer_z_k2_balance`** | 129.9 |
| 2 | `abs_z_S5` | 68.9 |
| 3 | `z_S5_rank` | 47.2 |
| 4 | `peer_diff_k1_amount` | 33.2 |
| 5 | `peer_z_k1_amount` | 27.7 |
| 6 | `peer_diff_k8_cents_mer` | 24.0 |
| 7 | `peer_diff_k2_balance` | 22.1 |
| 8–13 | peer distance / logcnt / mer×amt / mrisk / velocity | … |
| 14–15 | cohort mrisk G1; hours G2−global | … |

**Signal is real peer/cohort structure**, not cents/logfreq rediscovery (`ρ(R2)≈0.02`). Dominant driver: **balance-collision peer mean of `z_S5`**.

---

## Step 3 — Forensic & Slot5 scorecard

Orth vs **`Bo10_S5 = [Bo9, R2_orth]`**.

| Signal | z_S5 | Slot5_twin | NNLS_S5 |
|---|---|---|---|
| R3_peer_s42 @0.03 | **5/5 +17.1** min+13.9 sf**25** t=14.9 | **+28.7** sf25 t=13.1 | **+18.5** sf25 |
| R3_peer_bag @0.03 | +17.7 sf25 | +30.1 | +19.3 |
| R3_peer_linear @0.03 | +23.4 sf25 | +43.1 | +25.1 |

Folds for s42 on z_S5: `[+13.9, +17.8, +17.9, +17.1, +18.9]` — uniformly strong, but **forensic F2 blocks staging**.

**Why F2?** `peer_z` is a leave-self-out smooth of **`z_S5` itself**, which already embeds `Family_E`. Orth removes linear Bo10, yet test residual still couples to Family_E at 0.048–0.086.

---

## Staged NOSUBMIT

**None** (`F2` / `F3` gate).

---

## Guardrails

| Check | Status |
|---|---|
| Submit | **NOT done** |
| Pods | **0** |
| ATH | **0.95394** |

---

## Bottom line

1. Peer homophily is the **first family since Slot5 with near-zero `ρ(R2)`** and dual-elite OOF strength — directionally the right hunt.  
2. **Cannot stage** until F2&lt;0.035: next step should **peer a Bo9-orthogonal score** (or residualize `peer_z_*` against Bo10 *before* LGB), not raw `z_S5`.  
3. Hold Slot5; treat V18.4 as a **blueprint**, not ammo.

---

*End GM V18.4*
