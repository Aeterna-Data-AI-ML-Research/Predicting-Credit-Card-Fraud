# V43 Fold-0 — Qwen-only ready checklist

**Token:** `APPROVE_POD_F0_ONLY`  
**Status:** package hardened · **0 pods** · awaiting explicit user `LAUNCH F0`  
**NO_SUBMIT=True** (gate + pod scripts never call Kaggle)

## Locked in (this session)
| Item | Status |
|---|---|
| TabICL stripped from boot path | ✓ (`V43_SKIP_TABICL=1`; launch never calls it) |
| `K_PAIRS=2` → K=4 antithetic | ✓ |
| Fused `cross_entropy` (no full-vocab softmax) | ✓ |
| `USE_COMPILE=0` + SDPA + bf16 | ✓ |
| y0/y1 prefix length assert + prefix/pad mask | ✓ |
| Local Ridge+GBDT gate script | ✓ `gate_fold0_field24_local.py` |
| 21600s watchdog in launch docs | ✓ |
| E4 NOSUBMIT CSV | quarantined — **do not submit** |

## Live Secure POD quote (catalog read this session)
| GPU | $/hr | Stock |
|---|---:|---|
| **RTX 4090** | **$0.89** | LOW |
| A40 | $0.59 | LOW |
| A100 SXM | $2.19 | LOW |
| H100 SXM | $4.39 | MEDIUM |

**Recommend:** 1× Secure **RTX 4090** @ **$0.89/hr** · est. **2.5–3.5 h** → **~$2.2–$3.1** (Qwen-only; TabICL time removed).

## Gate after rsync
- nested ΔAUC on `z_modern_pub[f0]+Zr` ≥ **+5.0e-5**
- ≥ **4/5** inner subfolds positive
- **0** Kaggle submissions; report scorecard before Folds 1–4

## To boot
Reply exactly: **`LAUNCH F0`**
