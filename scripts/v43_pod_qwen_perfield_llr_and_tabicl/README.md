# V43 Fold-0 — Qwen2.5-0.5B Per-Field LLR ONLY

**Gemini token: `APPROVE_POD_F0_ONLY` · TabICL STRIPPED · NO_SUBMIT=True**

TabICLv2 is quarantined (`tabicl_v2_fullcontext.py` kept on disk for archaeology only — **do not boot**). V39 A40: TabICL 0.94324, 0/5, −4.7e-5.

## Modules
| File | Role |
|---|---|
| `serializer_24field.py` | `y: 0/1` prefix + 24 field spans + antithetic orders + unpermute |
| `qwen_digit_genbayes_24llr.py` | Qwen2.5-0.5B LoRA-r64, 4ep, CE-gather LLR → `(N,K,2)` + `(N,24)` |
| `gate_fold0_field24_local.py` | Mac post-rsync Ridge+GBDT gate vs `z_modern_pub[f0]` |
| `smoke_test_16row.py` | Local CPU serializer smoke |
| `launch_fold0_canary.sh` | Needs `V43_AUTHORIZE_POD=YES` **and** `V43_LAUNCH_F0=YES` |
| ~~`tabicl_v2_fullcontext.py`~~ | **DISABLED for F0** |

## Gemini lockdowns (baked in)
- `V43_K_PAIRS=2` → **K=4** (2 fwd + 2 exact rev); matched oof==test
- `F.cross_entropy(..., reduction="none")` — **no** `(B,T,151936)` log_softmax
- `V43_USE_COMPILE=0`, `attn_implementation="sdpa"`, bf16
- Assert `len(y0)==len(y1)`; mask prefix + pad before seq/field sums
- Gate: nested ΔAUC_f0 ≥ **+5.0e-5** with **≥4/5** subfolds; **0 Kaggle submits**
- Watchdog **21600s**; rsync `.npy` **before** gate / pod delete; never touch foreign pods

## Local smoke
```bash
.venv/bin/python scripts/v43_pod_qwen_perfield_llr_and_tabicl/smoke_test_16row.py
```

## Boot (ONLY after user types `LAUNCH F0`)
```bash
export V43_LAUNCH=1 V43_K_PAIRS=2 V43_USE_COMPILE=0
timeout 21600 python qwen_digit_genbayes_24llr.py --data /data --out /out/qwen --fold 0
# DO NOT run tabicl
```

## Post-rsync gate (Mac)
```bash
.venv/bin/python scripts/v43_pod_qwen_perfield_llr_and_tabicl/gate_fold0_field24_local.py \
  --qwen-dir out/v43_fold0_qwen \
  --modern-oof out/v42_1_modern/z_modern_pub_oof.npy \
  --modern-test out/v42_1_modern/z_modern_pub_test.npy \
  --data data --out logs/v43 --fold 0
```

## Outputs (Fold-0)
- `oof_f0_seq_llr.npy` `(139992, 4, 2)`  ← K=4
- `test_f0_seq_llr.npy` `(299984, 4, 2)`
- `oof_f0_field24_llr.npy` / `_std.npy` `(139992, 24)`
- `test_f0_field24_llr.npy` / `_std.npy` `(299984, 24)`
