# V28 Own Levers vs ATH 0.95511

Goal: find **non-board-consume** levers (structural zones, own HGB residual, Weni diversity) gated nested vs ATH surrogate / gb_bag5.

ATH lock: `sub_v27_ATH_PUB511.csv` (public 0.95511).

## L1 Structural zone soft-floor

| tag | gb gate | athS gate | ρATH e-4 |
|-----|---------|-----------|---------:|
| `zone_Z1_s0p15` | 5/5 +0.25 | 5/5 +0.25 | 0.000 |
| `zone_Z1_s0p30` | 5/5 +0.33 | 5/5 +0.33 | 0.000 |
| `zone_Z12_s0p15` | 5/5 +0.30 | 5/5 +0.30 | 0.000 |
| `zone_Z12_s0p25` | 5/5 +0.37 | 5/5 +0.37 | 0.000 |
| `zone_Z1234_s0p10` | 5/5 +0.32 | 5/5 +0.32 | 0.001 |
| `zone_Z1234_s0p20` | 5/5 +0.45 | 5/5 +0.45 | 0.001 |
| `zone_Z1234_s0p35` | 5/5 +0.52 | 5/5 +0.51 | 0.001 |

## L2 LOFO HGB residual

- solo AUC=0.94357
- ath-orth 4/5 mean=+0.00
- gb+8% 1/5 mean=-0.03

## L3 Weni diversity

| channel | solo | ath-orth | gb+8% | ρ raw |
|---------|-----:|----------|-------|------:|
| `weni_bpe_cat_d3` | 0.95067 | 2/5 +0.00 | 1/5 -0.16 | 378.1 |
| `weni_bpe_cat_d5` | 0.95068 | 2/5 -0.00 | 1/5 -0.15 | 374.6 |
| `weni_bpe_glm` | 0.95027 | 3/5 +0.00 | 0/5 -0.30 | 411.0 |
| `weni_bpe_lgb_l4` | 0.95093 | 4/5 +0.00 | 4/5 -0.04 | 344.3 |
| `weni_realmlp_bpe` | 0.94368 | 3/5 -0.00 | 1/5 -0.10 | 893.4 |
| `weni_realmlp_bpe_v2` | 0.95001 | 4/5 +0.00 | 5/5 +0.61 | 363.5 |
| `weni_tabm_bpe` | 0.94842 | 2/5 +0.00 | 3/5 +0.29 | 384.3 |

## Tomorrow submit queue

1. **L3** `ATH_x_weni_realmlp_bpe_v2_0p05` gateΔ=+0.61e-4 ρATH=0.7800334402541242 → `sub_v28_L3_ATH_x_weni_realmlp_bpe_v2_0p05.csv`
2. **L3** `ATH_x_weni_realmlp_bpe_v2_0p08` gateΔ=+0.61e-4 ρATH=2.0021940169978603 → `sub_v28_L3_ATH_x_weni_realmlp_bpe_v2_0p08.csv`
3. **L1** `zone_Z1234_s0p35` gateΔ=+0.52e-4 ρATH=0.0006771411809047834 → `sub_v28_L1_zone_Z1234_s0p35.csv`
4. **L1** `zone_Z1234_s0p20` gateΔ=+0.45e-4 ρATH=0.0006771411809047834 → `sub_v28_L1_zone_Z1234_s0p20.csv`
5. **L1** `zone_Z12_s0p25` gateΔ=+0.37e-4 ρATH=0.00021355147872270663 → `sub_v28_L1_zone_Z12_s0p25.csv`
6. **L1** `zone_Z1_s0p30` gateΔ=+0.33e-4 ρATH=6.152792053626399e-05 → `sub_v28_L1_zone_Z1_s0p30.csv`
7. **L1** `zone_Z1234_s0p10` gateΔ=+0.32e-4 ρATH=0.0006771411809047834 → `sub_v28_L1_zone_Z1234_s0p10.csv`
8. **L1** `zone_Z12_s0p15` gateΔ=+0.30e-4 ρATH=0.00021355147872270663 → `sub_v28_L1_zone_Z12_s0p15.csv`
9. **L1** `zone_Z1_s0p15` gateΔ=+0.25e-4 ρATH=6.152792053626399e-05 → `sub_v28_L1_zone_Z1_s0p15.csv`

## Law

Own levers must clear nested gb/ATH-surrogate gates **and** stay near ATH (ρ ≲ a few e-4) before burning public slots. Do not replace the board ATH with a solo native.

Elapsed 112.2s.
