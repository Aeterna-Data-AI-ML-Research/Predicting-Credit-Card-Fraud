# SCAN4 Within-Cohort id-Rank Residual Report

**NO_SUBMIT=True** · elapsed **389.6s** · Ridge α=500.0

## SCAN4 repro (train-only G1)

| metric | AUC |
|--------|----:|
| integer `rank(id)` tree (SCAN4-style) | **0.57489** |
| `group_size` tree | **0.60045** |
| uniform `u=rank/n` tree | **0.50546** |
| uniform `u` raw | 0.50362 |
| corr(u, size) | -0.0244 |

**Verdict:** `SIZE_LEAK` — SCAN4’s 0.57489 was **group-size leak** via integer rank, not pure within-cohort order.

## train∪test normalized `u` + residual booster on `y - p_cal(z_fullbank)`

| cohort | KS(u) | KS ok | solo u | solo size | boost_u Δ (n+) | boost_size Δ | joint Δ (n+) | hard |
|--------|------:|:-----:|-------:|----------:|---------------:|-------------:|-------------:|:----:|
| `G1_card_merch_dow_hour` | 0.93859 | N | 0.50359 | 0.60127 | -2.12e-05 (2/5) | -5.67e-05 | +3.26e-05 (3/5) | N |
| `G2_auth_chan_dev` | 0.98942 | N | 0.50026 | 0.55432 | -5.80e-05 (2/5) | +5.31e-05 | +3.36e-05 (4/5) | N |
| `G3_flag64` | 0.99883 | N | 0.50028 | 0.75614 | +4.50e-05 (4/5) | +4.31e-06 | +6.38e-05 (4/5) | N |
| `G4_card_merch` | 0.99654 | N | 0.50134 | 0.60052 | -5.37e-05 (3/5) | +2.79e-05 | +2.64e-05 (3/5) | N |
| `G5_dow_hour` | 0.99047 | N | 0.50141 | 0.50325 | -5.19e-05 (2/5) | +5.45e-05 | +6.15e-05 (3/5) | N |
| `G6_merch_only` | 0.99837 | N | 0.50143 | 0.62123 | -5.35e-05 (2/5) | -2.50e-05 | +7.69e-05 (4/5) | N |

**Global id u AUC:** 0.50145 (KS=1.00000)

**Hard pass any:** False
**Soft residual candidates (KS ok + Δ≥0.5e-4 & ≥4/5):** none

## Conclusion

SCAN4 0.57489 is SIZE_LEAK (integer rank / group_size), not normalized within-cohort order. Uniform u∈[0,1] is ~chance solo; residual boost on z_fullbank is the real test — see soft_residual_candidates.

JSON: `/tmp/s6e11_v39_ssd/out/SCAN4_cohort_id_rank_report.json`