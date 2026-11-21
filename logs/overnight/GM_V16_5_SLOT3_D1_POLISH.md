# GM V16.5 — Slot3: Slot1 + d1_v4 polish (w_d1=0.0355)

**0 pods · Submitted · Public = `0.95389` (tie)**

---

## Step 1 — d1 fine grid on true z_S1 (0.95243 + Family E 0.01045)

| Δw | w_d1 | pos/5 | meanΔ e-5 | min e-5 | folds | sf25 | t | 1−ρ(Slot1) e-4 |
|---:|-----:|------:|----------:|--------:|-------|-----:|--:|---------------:|
| +0.0030 | 0.0330 | **5** | 0.345 | **+0.047** | `[0.591,0.344,0.164,0.576,0.047]` | **18** | **3.31** | 0.013 |
| +0.0040 | 0.0340 | 5 | 0.435 | +0.061 | `[0.781,0.441,0.161,0.732,0.061]` | 17 | 2.99 | 0.023 |
| +0.0050 | 0.0350 | 4 | 0.503 | −0.008 | `[1.026,0.468,0.134,0.894,-0.008]` | 17 | 2.71 | 0.037 |
| **+0.0055** | **0.0355** | **5** | 0.551 | +0.008 | `[1.122,0.49,0.132,1.0,0.008]` | **17** | **2.91** | **0.044** |
| +0.0060 | 0.0360 | 5 | 0.587 | +0.002 | `[1.193,0.506,0.147,1.087,0.002]` | 17 | 2.94 | 0.053 |

**Strict select** (largest of {0.0050,0.0055} with 5/5, min≥0.02e-5, sf≥18, t≥3.0): **none**.  
Soft choice for requested Slot3 file: **+0.0055** (5/5, min=+0.008, sf=17, t=2.91).

Only full hard-gate point remains **+0.0030** (already staged as V16.4 P1).

---

## Step 2 — Live Slot 3

| Item | Value |
|------|------:|
| File | `sub_v16_5_LIVE_slot3_Slot1_plus_d1_w0p0355.csv` |
| Construction | `qlock(L(Slot1) + 0.0055·unitize(d1_v4_test), LIVE)` |
| Rows / NaNs | 299984 / 0 |
| 1−ρ(Slot1) | 0.0443e-4 |
| 1−ρ(LIVE_095388) | 0.243e-4 |
| Kaggle ref | `58332767` |
| **Public score** | **`0.95389`** |
| Δ vs Slot1 | **+0.0 bps** |

---

## Verdict
Slot3 **tied** again. Public shelf at 0.95389 is sticky; d1 +0.0055 did not cross another public tick. Prefer waiting / private CV over further tiny rank steps on the same shelf.
