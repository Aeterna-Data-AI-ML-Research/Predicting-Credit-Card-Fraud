# V41 Step6d — Board-Model v43 Inspection & Step_AC Audit

**NO_SUBMIT=True** · LB_FWD_AA=0.95569 (flat @ 0.030e-4 AA→AB)

## (a) goodpjw Board Model
- Notebook pulled: `s6e11-board-model-realmlp-lb-0-95569.ipynb`
- **Board model NOT shipped** (offline fit on 151–233 scored files; ~0.7pt LOO near tip).
- Forecasts published: **AA +1.05 pts**, **AB +0.58 pts** (1 pt = 1e-5 LB).
- Step_AB recipe (reconstructible): `Zr(Sriram_orth − 0.4·RECENT − 0.15·MOVE6 + 0.15·MSTEP)`.
- Pinned inputs **public/downloadable**: `asteriospanoras/s6e11-public-board-files`, OOF bank, Aman streams, RealMLP OOF, Malyshev/Areeb/Sriram CSVs.
- **Cannot literally refit goodpjw's board model** here; fitted LOCAL SURROGATE on 7 tip/live points.

## Candidates

| Cand | 1−ρ AB e4 | cos·d_AB_AA | cos·S02↓ | cos·S06↓ | local fc pts | loser≤0 | notes |
|---|---:|---:|---:|---:|---:|:---:|---|
| Cand1_FWD_0p10 | 0.1000 | 0.657 | -0.959 | -0.086 | -0.29 | True | 888c1c85e8a3… |
| Cand2_StepAC | 0.0600 | 0.656 | -0.975 | -0.094 | -0.23 | True | dir=blend_secant_antiare |
| Cand3_delta1_v4 | 0.1500 | -0.007 | -0.006 | -0.107 | -0.83 | True | OOF Δe5=1.53 pos=4/5 |

### Prop screen (vs z_pub_only+Zr)

| name | Δe5 | pos/5 | sign | cos·AA | cos·S02 | cos·S06 | loser≤0 | fc |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| FamilyE_L7_s2026_5f | 1.06 | 4/5 | -1 | -0.049 | 0.065 | 0.007 | False | -0.87 |
| StageB_4fam | 1.83 | 2/5 | 1 | 0.004 | -0.004 | 0.151 | False | -0.83 |
| delta1_v4 | 1.53 | 4/5 | 1 | -0.007 | -0.006 | -0.107 | True | -0.83 |
| C3_harvest_p4 | 2.45 | 4/5 | -1 | -0.055 | 0.108 | 0.999 | False | -0.87 |
| R2_proxy_lgb_stageB68 | 1.54 | 2/5 | 1 | 0.003 | -0.003 | 0.151 | False | -0.83 |
| R1_proxy_xgb_stageB68 | 1.42 | 2/5 | 1 | 0.007 | -0.009 | 0.150 | False | -0.83 |

### Pairwise 1−ρ (×1e-4)

| |AB|FWD030|S02|S06|Cand1|Cand2|Cand3|
|---|---|---|---|---|---|---|---|
| AB |0.000|0.030|0.028|0.280|0.100|0.060|0.150|
| FWD030 |0.030|0.000|0.113|0.330|0.021|0.005|0.181|
| S02 |0.028|0.113|0.000|0.282|0.229|0.167|0.178|
| S06 |0.280|0.330|0.282|0.000|0.417|0.371|0.474|
| Cand1 |0.100|0.021|0.229|0.417|0.000|0.006|0.253|
| Cand2 |0.060|0.005|0.167|0.371|0.006|0.000|0.211|
| Cand3 |0.150|0.181|0.178|0.474|0.253|0.211|0.000|

Missing exact artifacts: ['R1_pure_Bo10 (OOF npy not on disk — only historical CSVs)', 'R2_LGB pure OOF (only LIVE CSV)', 'StageB_10fam exact (using StageB_4fam / stageB68 proxies)']

## STOP
NO_SUBMIT=True. Awaiting single-slot authorization.
