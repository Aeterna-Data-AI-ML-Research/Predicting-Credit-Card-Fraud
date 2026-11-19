# V28c Lift More (quota blocked)

Daily Kaggle submit returned **400** (quota exhausted). Staged stronger own levers for next window.

## Nested gates (board surrogates)

- Pure soft75 / hard Z1234: still **5/5** on gb/ATH/SUB (~+0.49–0.54e-4).
- **Best combo:** `hard_x_weni_0p08` — zone then Weni RealMLP-BPE-v2 small-w.
  - vs ATH_surr 5/5 +0.785
  - vs SUB_surr 5/5 +0.785
  - vs gb 5/5 +1.150
  - vs zone-only 4/5 +0.254
  - Simulated public **0.95513–0.95515**
  - File: `submissions/sub_v28c_BEST_hard_x_weni_0p08.csv`

## Next submit order (when quota resets)

1. `sub_v28c_BEST_*` (zone×weni combo)
2. `sub_v28_L1_zone_Z1234_s0p75.csv` / `sub_v28c_LOCK_hard.csv`
3. `sub_v28_L3_ATH_x_weni_realmlp_bpe_v2_0p05.csv`

