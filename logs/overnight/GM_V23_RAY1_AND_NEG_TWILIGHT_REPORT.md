# GM V23 Ray-1 Crest + −U_orth Twilight Report

**elapsed** 183s · ATH was `0.95426`

## Live results
| tag | Public LB | ρ(sri) e-4 | ρ(R2) e-4 | notes |
|---|---:|---:|---:|---|
| `R1_1p55e4` | **0.95426** | 1.550 | 0.016 | pure Ray-1 |
| `R1_1p75e4` | **0.95426** | 1.750 | 0.042 | pure Ray-1 |
| `NEG_TWI_0p35e4` | **0.95417** | 1.476 | 0.350 | tw=0.779 |

- Best among probes: **0.95426** (ATH hold if `< 0.95426`)

## Log
```
[    0.0s] === V23 Ray-1 crest + −U_orth twilight ===
[    1.5s] anchors 1−ρ(sri): P2=0.550e-4 R2=1.250e-4 R2−P2=0.400e-4
[    5.2s] STAGED sub_v23_R1_1p55e4_pure_ray1.csv
[    5.2s] SUBMIT v23.R1_1p55e4 pure Ray-1 no-orth | ρ(sri)=1.550e-4 ρ(R2)=0.016e-4 ρ(P2)=0.542e-4 w=0.02977
[    9.9s] Submission ref: 58420156
7 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[    9.9s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  2%|▏         | 144k/7.52M [00:00<00:05, 1.38MB/s]
  7%|▋         | 512k/7.52M [00:00<00:09, 762kB/s] 
  8%|▊         | 624k/7.52M [00:00<00:09, 757kB/s]
 11%|█         | 864k/7.52M [00:01<00:07, 898kB/s]
 31%|███▏      | 2.36M/7.52M [00:01<00:01, 3.66MB/s]
 66%|██████▌   | 4.95M/7.52M [00:01<00:00, 8.53MB/s]
 83%|████████▎ | 6.20M/7.52M [00:01<00:00, 7.9
[   19.1s] poll 58420156,sub_v23_R1_1p55e4_pure_ray1.csv,2026-11-16 06:48:13.117000,v23.R1_1p55e4 pure Ray-1 no-orth | ρ(sri)=1.550e-4 ρ(R2)=0.016e-4 ρ(P2)=0.542e-4 w=0.02977,SubmissionStatus.PENDING,,
[   37.8s] poll 58420156,sub_v23_R1_1p55e4_pure_ray1.csv,2026-11-16 06:48:13.117000,v23.R1_1p55e4 pure Ray-1 no-orth | ρ(sri)=1.550e-4 ρ(R2)=0.016e-4 ρ(P2)=0.542e-4 w=0.02977,SubmissionStatus.COMPLETE,0.95426,
[   37.8s] R1_1p55e4 DONE pub=0.95426 ok=True
[   53.6s] STAGED sub_v23_R1_1p75e4_pure_ray1.csv
[   53.6s] SUBMIT v23.R1_1p75e4 pure Ray-1 no-orth | ρ(sri)=1.750e-4 ρ(R2)=0.042e-4 ρ(P2)=0.644e-4 w=0.03163
[   58.6s] Submission ref: 58420179
6 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[   58.6s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<00:48, 164kB/s]
  7%|▋         | 544k/7.52M [00:00<00:07, 1.00MB/s]
  9%|▊         | 672k/7.52M [00:00<00:07, 988kB/s] 
 11%|█         | 848k/7.52M [00:00<00:07, 925kB/s]
 36%|███▌      | 2.69M/7.52M [00:01<00:01, 4.55MB/s]
 82%|████████▏ | 6.14M/7.52M [00:01<00:00, 11.4MB/s]
100%|██████████| 7.52M/7.52M [00:01<00:00, 4.
[   67.4s] poll 58420179,sub_v23_R1_1p75e4_pure_ray1.csv,2026-11-16 06:49:01.773000,v23.R1_1p75e4 pure Ray-1 no-orth | ρ(sri)=1.750e-4 ρ(R2)=0.042e-4 ρ(P2)=0.644e-4 w=0.03163,SubmissionStatus.PENDING,,
[   86.2s] poll 58420179,sub_v23_R1_1p75e4_pure_ray1.csv,2026-11-16 06:49:01.773000,v23.R1_1p75e4 pure Ray-1 no-orth | ρ(sri)=1.750e-4 ρ(R2)=0.042e-4 ρ(P2)=0.644e-4 w=0.03163,SubmissionStatus.COMPLETE,0.95426,
[   86.2s] R1_1p75e4 DONE pub=0.95426 ok=True
[   98.2s] === rebuild factory + U_orth4 (V1+V7+V8+V2) for −twilight ===
[  136.6s]   V1: sgn=+1 ridge 5/5 t=6.46 m=+2.10
[  138.9s]   V8: sgn=+1 ridge 5/5 t=5.11 m=+1.69
[  141.4s]   V7: sgn=+1 ridge 5/5 t=6.53 m=+2.39
[  143.7s]   V2: sgn=+1 ridge 5/5 t=4.83 m=+1.83
[  143.8s] U_orth4 max|B|=0.000000
[  148.3s] STAGED sub_v23_NEG_TWI_Uorth_0p35e4.csv
[  148.3s] SUBMIT v23.NEG_TWI −U_orth4 @0.35e-4 from R2 | ρ(R2)=0.350e-4 ρ(sri)=1.476e-4 tw=0.779 sign=-1
[  154.4s] Submission ref: 58420245
5 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[  154.4s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  2%|▏         | 144k/7.52M [00:00<00:05, 1.36MB/s]
  8%|▊         | 624k/7.52M [00:00<00:06, 1.10MB/s]
 10%|▉         | 752k/7.52M [00:00<00:06, 1.02MB/s]
 12%|█▏        | 928k/7.52M [00:00<00:07, 978kB/s] 
 17%|█▋        | 1.31M/7.52M [00:01<00:04, 1.61MB/s]
 20%|██        | 1.52M/7.52M [00:01<00:03, 1.69MB/s]
 32%|███▏      | 2.39M/7.52M [00:01<00:01, 3
[  163.8s] poll 58420245,sub_v23_NEG_TWI_Uorth_0p35e4.csv,2026-11-16 06:50:37.520000,v23.NEG_TWI −U_orth4 @0.35e-4 from R2 | ρ(R2)=0.350e-4 ρ(sri)=1.476e-4 tw=0.779 sign=-1,SubmissionStatus.PENDING,,
[  183.0s] poll 58420245,sub_v23_NEG_TWI_Uorth_0p35e4.csv,2026-11-16 06:50:37.520000,v23.NEG_TWI −U_orth4 @0.35e-4 from R2 | ρ(R2)=0.350e-4 ρ(sri)=1.476e-4 tw=0.779 sign=-1,SubmissionStatus.COMPLETE,0.95417,
[  183.1s] NEG_TWI DONE pub=0.95417 ok=True tw=0.779
```

