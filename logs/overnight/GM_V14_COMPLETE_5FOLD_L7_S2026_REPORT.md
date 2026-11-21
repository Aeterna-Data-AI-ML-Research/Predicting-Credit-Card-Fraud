# GM V14 — Complete Hardened 5-Fold L7_s2026 Report

**STRICT NO_SUBMIT=True** · no live Kaggle submit from this pipeline.  
**Updated:** 2026-11-13 ~19:54 UTC — **ALL 8 FOLDS LOCAL** · pod terminated.

## Status — COMPLETE

| Field | Value |
|---|---|
| Pod | `li1uaxfpuiqf59` · RTX 4080 SUPER · **DELETED** (was $0.28/hr) |
| Local bank | `preds/l7_s2026_5f/` — `oof/test_{0..4}` sized ✅ |
| Emergency backup | `~/kaggle_backups/l7_s2026_5f_v14/` (used while `/Volumes/pics` was unmounted; restored) |
| `ALL_FOLDS_DONE` | present |
| `ckpts/fold_latest.pt` | 164.6 MB fp16 |

### Fold AUC / wall (Bayes OOF)

| Fold | Seed | OOF AUC | Train | Score | Wall | Local |
|---|---|---|---|---|---|---|
| 0 | 2026 | **0.947493** | 70.3 m | 26.0 m | (prior) | ✅ |
| 1 | 2027 | **0.949578** | 48.9 m | 26.8 m | **75.7 m** | ✅ |
| 2 | 2028 | **0.954835** | 48.9 m | 26.8 m | **75.7 m** | ✅ |
| 3 | 2029 | **0.949330** | 50.3 m | 26.8 m | **77.1 m** | ✅ |
| 4 | 2030 | **0.944992** | 50.6 m | 26.8 m | **77.5 m** | ✅ |

Fold-1 canary: solo AUC **0.94958 > 0.9485** — **PASS**.

---

## Step 1 — Recon

| Item | Status |
|---|---|
| Active pods (pre-launch) | **0** |
| `orders_10` / `master_perms_seed2026` | `(10, 24)` int64 — **NC=24** |
| `oof/test_{0..4}` | **ON DISK** |

## Step 2 — Kaggle upstream

| Check | Result |
|---|---|
| Public board | Cap still **0.95388** (areebbb bold) |
| seed-2026 full 5-fold test dump | still missing upstream |

## Step 3 — Preflight / parity (HARD GATE)

Script: `scripts/gm_v14_complete_hardened_pipeline.py`

```
rho(llr)=0.99997335 · max|Δ|=0.064293 · pass=True · replica.pt strict=True OK
```

## Step 4 — Pod resume — **DONE**

Resume folds 1–4 on 4080 SUPER completed; artifacts restored to pics after remount.

## Step 5 — Assemble + ATH 4-bar — **READY TO RUN**

Inputs present locally. Next:

- `l7_s2026_oof_5fold_raw.npy` `(699960, 10, 2)`
- `l7_s2026_test_5fold_raw.npy` `(5, 299984, 10, 2)`
- `R_bayes_5f` / `R_ranklogit_5f`
- Moves 1–3 on ATH 0.95388 + NOSUBMIT staging

## Related reports

| Doc | Content |
|---|---|
| `GM_V14_1_WATCHDOG_CANARY_STEP5_PREP.md` | Canary gates + L7F/LM refs |
| `GM_V14_2_FOLD1_VERIFICATION.md` | F1 two-seed (+4.47 bps) + F0+F1 ATH Moves (weak) |

## Guardrails

- **NO_SUBMIT=True**
- Foreign pod `lheroidmuab1j4` never touched
- Zero live Kaggle submit from V14
