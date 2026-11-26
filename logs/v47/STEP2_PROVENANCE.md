# V47.2 STEP2 — Provenance

- v0 streams: `['B_xgb_llr', 'P4_L4.2_pure5f', 'P4_L4.3_pure5f', 'P4_L4.5_pure5f', 'P_super_v4p_hc', 'P_super_v5_harvest', 'delta1_v4', 'gpt2_med_4ep_mean', 'gpt2_med_4ep_shrink', 'l7_R5', 'l7_U5_gini10', 'l7_curv', 'l7_mean10', 'zone_honest']`
- **P_super_v5_harvest**: `KEEP` finalize blends v4p_hc + paradigm_B xgb_llr (self); no tip CSV loads
- **B_xgb_llr**: `KEEP` 
- **P4_L4.2_pure5f**: `KEEP` 
- **P4_L4.3_pure5f**: `KEEP` 
- **P4_L4.5_pure5f**: `KEEP` 
- **delta1_v4**: `KEEP` 
- **l7_s2026_5f**: `KEEP` builder mentions goodpjw seed7 path as optional compare; our s2026 preds/ckpts are self-trained
- **gpt2_med_4ep**: `KEEP` 
- **P_super_v4p_hc**: `KEEP` 
- **oof_xgb_clean**: `KEEP` 
- **SCAN_P_super_v6**: `EXCLUDED_FROM_V1` 
- **SCAN_soft_diff**: `EXCLUDED_FROM_V1` 

## Checkpoints
```json
{
  "l7_s2026_5f": [
    "fold_0.pt",
    "fold_latest.pt",
    "replica.pt"
  ],
  "l7_has_full_5fold_weights": false,
  "gpt2_med_4ep": [
    "fold_0.pt",
    "fold_1.pt",
    "fold_2.pt",
    "fold_3.pt",
    "fold_4.pt"
  ],
  "gpt2_has_full_5fold_weights": true
}
```
