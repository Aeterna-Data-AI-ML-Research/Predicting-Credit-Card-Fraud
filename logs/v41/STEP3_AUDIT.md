# V41 STEP3 AUDIT — Rebase uid/CB onto Step_AB (0.95569)

**NO_SUBMIT=True** · Hard stop. Do not run Step 4 without `PROCEED TO STEP 4`.

## Executive findings

- Tip: `F1b_pub_Step_AB_095569.csv` (`3e1dba5d…`, 0.95569).
- Quarantine clean: **True** (no zanch/v10_lgb/soft_diff).
- V40 T1/T2 operator: `expit(L(areeb)+α·orth_ridge50([areeb,gpw,ath], prop))`; T3 = raw macro.
- OOF baselines: fullbank=0.952621, Part12 wide=0.952565 (both >0.9524).
- Hygiene: uid OOF AUC=0.93485, KS=0.0219; CB AUC=0.98852 (TE-inflated), KS=0.0947, disp=0.561.
- Staging signs all `+1`; uid+CB mix `w_cb=1.0` (P3 ≈ pure CB).
- **Zr fixed** (`floor+U_AB`); unmasked α=0.07 lifts 977 uid / 1217 CB zone rows above U=0.0558.
- No tree re-dilution vs `d_AB_Z` (pearson uid/cb > 0).
- Requested α≥0.04 mostly **overshoot**; sweet extras staged as `P1s_*` / `P2s_*`.

## 1–2. Provenance / OOF

Full tables in `STEP3_AUDIT.json`. Top OOF lifts are CB/direct on fullbank (TE-inflated — interpret cautiously).

## 3. Geometry

```json
{
  "alignment": {
    "uid_vs_d_AB_AA": {
      "pearson_nonzone": 0.1929800467086093,
      "cosine_l2_nonzone": 0.19274696613769388
    },
    "cb_vs_d_AB_AA": {
      "pearson_nonzone": 0.09438279315387191,
      "cosine_l2_nonzone": 0.09401123590270855
    },
    "uid_vs_d_AB_Z": {
      "pearson_nonzone": 0.03374020305763103,
      "cosine_l2_nonzone": 0.03369945176792578
    },
    "cb_vs_d_AB_Z": {
      "pearson_nonzone": 0.05931710539107902,
      "cosine_l2_nonzone": 0.05908359142217663
    },
    "uid_vs_d_areeb_AA": {
      "pearson_nonzone": 0.02788129797989345,
      "cosine_l2_nonzone": 0.027847623053582113
    },
    "cb_vs_d_areeb_AA": {
      "pearson_nonzone": -0.01956339509345091,
      "cosine_l2_nonzone": -0.0194863797704037
    },
    "tree_redilution_check": {
      "uid_proj_on_d_AB_Z_negative": false,
      "cb_proj_on_d_AB_Z_negative": false,
      "note": "If True, blending that prop without orth may reverse Z\u2192AB climb (tree re-dilution)."
    }
  },
  "zone_jump": {
    "alpha_demo": 0.07,
    "prop": "uid",
    "unmasked_zone_gt_0558": 977,
    "unmasked_zone_gt_050": 0,
    "zr_zone_gt_0558": 0,
    "zr_zone_gt_050": 0,
    "1mrho_unmasked_vs_zr_e4": 0.03732510449383497,
    "cb_unmasked_zone_gt_0558": 1217,
    "cb_unmasked_zone_gt_050": 0,
    "cb_1mrho_unmasked_vs_zr_e4": 0.07040143485625983,
    "default": "PRESERVE Zr (Step_AB zone lock) unless Step4 proves zone fraud positives",
    "Zr_bugfix": "floor+U_AB (was floor-U_AB reversing zone order \u2192 false 3.47e-4 floor)"
  },
  "zone_geom": {
    "n_zones": 16730,
    "uid_zone_U": {
      "min": 1.6667555602965492e-06,
      "median": 0.042082244386367276,
      "p90": 0.18848705264280766,
      "p99": 0.5601669089018069,
      "max": 0.9659931863032695,
      "n_gt_0558": 6958,
      "n_gt_050": 246
    },
    "cb_zone_U": {
      "min": 1.6667555602965492e-06,
      "median": 0.1066990239479439,
      "p90": 0.42313323377246786,
      "p99": 0.8339105752306787,
      "max": 0.9929379566910235,
      "n_gt_0558": 11486,
      "n_gt_050": 1218
    },
    "AB_zone_U_max": 0.05576797429196224
  }
}
```

## 4. Staged candidates

### Requested P* (Zr-preserved)

| file | 1−ρ×1e−4 | band |
|---|---:|---|
| `P1_AB_uid_a0p04.csv` | 1.7568 | **edge** |
| `P1_AB_uid_a0p07.csv` | 5.4093 | **overshoot** |
| `P1_AB_uid_a0p10.csv` | 11.0926 | **overshoot** |
| `P1_AB_uid_clip99_a0p04.csv` | 1.7193 | **edge** |
| `P2_AB_cb_a0p04.csv` | 3.3729 | **overshoot** |
| `P2_AB_cb_a0p07.csv` | 10.3909 | **overshoot** |
| `P2_AB_cb_a0p10.csv` | 21.3235 | **overshoot** |
| `P3_AB_uid_cb_oofmix_a0p05.csv` | 5.2808 | **overshoot** |
| `P3_AB_uid_cb_oofmix_a0p09.csv` | 17.2418 | **overshoot** |

### Sweet-spot extras (hit [0.15,1.50])

| file | α | 1−ρ×1e−4 | band |
|---|---:|---:|---|
| `P1s_AB_uid_a0p015.csv` | 0.015 | 0.2459 | **sweet** |
| `P1s_AB_uid_a0p02.csv` | 0.02 | 0.4376 | **sweet** |
| `P1s_AB_uid_a0p03.csv` | 0.03 | 0.9866 | **sweet** |
| `P2s_AB_cb_a0p01.csv` | 0.01 | 0.2094 | **sweet** |
| `P2s_AB_cb_a0p015.csv` | 0.015 | 0.4717 | **sweet** |
| `P2s_AB_cb_a0p02.csv` | 0.02 | 0.8397 | **sweet** |

### Orth-push extras

| file | omr | 1−ρ×1e−4 | band |
|---|---:|---:|---|
| `P1m_AB_uid_orth_2e3.csv` | 0.002 | 0.0001 | noop |
| `P1m_AB_uid_orth_5e3.csv` | 0.005 | 0.0007 | noop |
| `P2m_AB_cb_orth_2e3.csv` | 0.002 | 0.0001 | noop |
| `P2m_AB_cb_orth_5e3.csv` | 0.005 | 0.0007 | noop |

### Pairwise `1−ρ` (×1e−4)

|  | P1_AB_uid_a0p04.csv | P1_AB_uid_a0p07.csv | P1_AB_uid_a0p10.csv | P1_AB_uid_clip99_a0p04.csv | P2_AB_cb_a0p04.csv | P2_AB_cb_a0p07.csv | P2_AB_cb_a0p10.csv | P3_AB_uid_cb_oofmix_a0p05.csv | P3_AB_uid_cb_oofmix_a0p09.csv | F1b_pub_Step_AB_095569.csv | quarter_our_095567.csv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1_AB_uid_a0p04.csv | 0.0000 | 1.0018 | 4.0242 | 0.0045 | 3.3634 | 9.0456 | 18.6345 | 4.8270 | 15.0013 | 1.7568 | 2.5080 |
| P1_AB_uid_a0p07.csv | 1.0018 | 0.0000 | 1.0114 | 1.0381 | 5.6808 | 10.3513 | 18.9212 | 6.8081 | 15.6283 | 5.4093 | 6.3464 |
| P1_AB_uid_a0p10.csv | 4.0242 | 1.0114 | 0.0000 | 4.0924 | 10.0222 | 13.6744 | 21.2172 | 10.8111 | 18.2675 | 11.0926 | 12.2163 |
| P1_AB_uid_clip99_a0p04.csv | 0.0045 | 1.0381 | 4.0924 | 0.0000 | 3.3455 | 9.0425 | 18.6464 | 4.8140 | 15.0082 | 1.7193 | 2.4686 |
| P2_AB_cb_a0p04.csv | 3.3634 | 5.6808 | 10.0222 | 3.3455 | 0.0000 | 1.9263 | 7.7464 | 0.2134 | 5.3704 | 3.3729 | 4.2086 |
| P2_AB_cb_a0p07.csv | 9.0456 | 10.3513 | 13.6744 | 9.0425 | 1.9263 | 0.0000 | 1.9491 | 0.8582 | 0.8651 | 10.3909 | 11.4765 |
| P2_AB_cb_a0p10.csv | 18.6345 | 18.9212 | 21.2172 | 18.6464 | 7.7464 | 1.9491 | 0.0000 | 5.3911 | 0.2176 | 21.3235 | 22.6603 |
| P3_AB_uid_cb_oofmix_a0p05.csv | 4.8270 | 6.8081 | 10.8111 | 4.8140 | 0.2134 | 0.8582 | 5.3911 | 0.0000 | 3.4447 | 5.2808 | 6.1996 |
| P3_AB_uid_cb_oofmix_a0p09.csv | 15.0013 | 15.6283 | 18.2675 | 15.0082 | 5.3704 | 0.8651 | 0.2176 | 3.4447 | 0.0000 | 17.2418 | 18.4947 |
| F1b_pub_Step_AB_095569.csv | 1.7568 | 5.4093 | 11.0926 | 1.7193 | 3.3729 | 10.3909 | 21.3235 | 5.2808 | 17.2418 | 0.0000 | 0.5047 |
| quarter_our_095567.csv | 2.5080 | 6.3464 | 12.2163 | 2.4686 | 4.2086 | 11.4765 | 22.6603 | 6.1996 | 18.4947 | 0.5047 | 0.0000 |

## 5. STOP

Step 3 complete. **Awaiting human audit.** Reply `PROCEED TO STEP 4` to continue.
