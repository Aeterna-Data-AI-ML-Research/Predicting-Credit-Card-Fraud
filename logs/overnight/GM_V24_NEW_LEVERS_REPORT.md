# GM V24 New Levers Report

Prior ATH `0.95426` · elapsed 395s

## Live
| tag | Public | ρ(R2) e-4 | status |
|---|---:|---:|---|
| `SRI440_pub` | **0.9544** | 6.586 | ok |
| `R2_to_SRI440_0p60e4` | **0.95432** | 0.600 | ok |
| `R2_dmlp_v15_w0.22` | **0.95432** | 3.306 | ok |
| `DUAL_R2_GPW32_dmlp30` | **0.9544** | 7.011 | ok |
| `SRI440_x_S95417_90_10` | **0.95436** | 6.795 | ok |
| `R2_to_GPW32_0p35e4` | **0.9543** | 0.350 | ok |
| `R2_to_SRI440_1p20e4` | **0.95434** | 1.200 | ok |
| `R2_dmlp_v15_w0.30` | **0.95432** | 6.143 | ok |
| `R2_x_S95417_95_5` | **0.95426** | 0.252 | ok |

- Best this batch: **0.9544**

## Lever thesis
- Sriram 0.95440: dual-base (step11+Bhuvan) + **w=0.30 Δmlp PyTabKit v15** (was 0.15/v5).
- goodpjw ladder-meets-stack **0.95432** peer.
- Avoided re-running orth/twilight/Ray-1.

## Log
```
[    0.0s] === V24 NEW LEVERS (sri95440 / Δmlp_v15 / gpw95432 / peer) ===
[    1.7s] recon vs pub sri: ρ=0.0004e-4 max|d|=0.0137941 spearman=0.99999996
[    1.9s] geometry vs R2: sri440=6.588e-4 sri_pub=6.586e-4 gpw32=9.886e-4 old_sri=1.250e-4
[    2.1s] logit corr R2×sri440=0.9993 R2×gpw32=0.9989
[    2.2s] Δmlp_v15 vs Δmlp_old corr=0.8614
[   30.3s] CAND SRI440_recon                 ρR2=  6.588e-4 ρ440=  0.000e-4 | reconstructed sriram 0.95440 recipe w=0.30 v15 dual-base
[   30.3s] CAND SRI440_pub                   ρR2=  6.586e-4 ρ440=  0.000e-4 | published sriram submission.csv (0.95440)
[   30.3s] CAND R2_to_SRI440_0p30e4          ρR2=  0.300e-4 ρ440=  4.078e-4 | blend w=0.2132 toward sri440 | ρ=0.300e-4
[   30.3s] CAND R2_to_SRI440_0p60e4          ρR2=  0.600e-4 ρ440=  3.214e-4 | blend w=0.3015 toward sri440 | ρ=0.600e-4
[   30.3s] CAND R2_to_SRI440_1p20e4          ρR2=  1.200e-4 ρ440=  2.166e-4 | blend w=0.4265 toward sri440 | ρ=1.200e-4
[   30.3s] CAND R2_dmlp_v15_w0.08            ρR2=  0.438e-4 ρ440=  5.249e-4 | R2 + 0.08*(L(N15)-L(C))
[   30.3s] CAND R2_dmlp_v15_w0.15            ρR2=  1.538e-4 ρ440=  4.794e-4 | R2 + 0.15*(L(N15)-L(C))
[   30.3s] CAND R2_dmlp_v15_w0.22            ρR2=  3.306e-4 ρ440=  5.009e-4 | R2 + 0.22*(L(N15)-L(C))
[   30.3s] CAND R2_dmlp_v15_w0.30            ρR2=  6.143e-4 ρ440=  6.070e-4 | R2 + 0.30*(L(N15)-L(C))
[   30.3s] CAND GPW32_pub                    ρR2=  9.886e-4 ρ440=  6.550e-4 | goodpjw ladder-meets-stack 0.95432
[   30.3s] CAND R2_to_GPW32_0p35e4           ρR2=  0.350e-4 ρ440=  5.071e-4 | w=0.1879 ρ=0.350
[   30.3s] CAND R2_to_GPW_S12_0p35e4         ρR2=  0.350e-4 ρ440=  5.172e-4 | w=0.2268 ρ=0.350
[   30.3s] CAND SRI440_x_S95417_90_10        ρR2=  6.795e-4 ρ440=  1.006e-4 | Areeb on new high × shallow 0.95417
[   30.3s] CAND R2_x_S95417_95_5             ρR2=  0.252e-4 ρ440=  6.573e-4 | soft staged leftover
[   30.3s] CAND DUAL_R2_GPW32_dmlp30         ρR2=  7.011e-4 ρ440=  1.970e-4 | sriram-style dual-base avg with R2+gpw32 @0.30 v15
[   31.4s] STAGED sub_v24_SRI440_recon.csv
[   32.1s] STAGED sub_v24_SRI440_pub.csv
[   32.8s] STAGED sub_v24_R2_to_SRI440_0p30e4.csv
[   34.0s] STAGED sub_v24_R2_to_SRI440_0p60e4.csv
[   34.6s] STAGED sub_v24_R2_to_SRI440_1p20e4.csv
[   35.7s] STAGED sub_v24_R2_dmlp_v15_w0.08.csv
[   36.3s] STAGED sub_v24_R2_dmlp_v15_w0.15.csv
[   37.4s] STAGED sub_v24_R2_dmlp_v15_w0.22.csv
[   38.4s] STAGED sub_v24_R2_dmlp_v15_w0.30.csv
[   39.5s] STAGED sub_v24_GPW32_pub.csv
[   40.1s] STAGED sub_v24_R2_to_GPW32_0p35e4.csv
[   40.8s] STAGED sub_v24_R2_to_GPW_S12_0p35e4.csv
[   41.7s] STAGED sub_v24_SRI440_x_S95417_90_10.csv
[   43.0s] STAGED sub_v24_R2_x_S95417_95_5.csv
[   43.7s] STAGED sub_v24_DUAL_R2_GPW32_dmlp30.csv
[   43.7s] SUBMIT v24.SRI440_pub | ρ(R2)=6.586e-4 ρ(440)=0.000e-4 | published sriram submission.csv (0.95440)
[   48.7s] Submission ref: 58468079
9 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[   48.7s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<00:53, 147kB/s]
  7%|▋         | 512k/7.52M [00:00<00:08, 901kB/s] 
  8%|▊         | 592k/7.52M [00:00<00:09, 807kB/s]
 10%|█         | 800k/7.52M [00:00<00:08, 873kB/s]
 41%|████      | 3.09M/7.52M [00:01<00:00, 5.35MB/s]
 58%|█████▊    | 4.33M/7.52M [00:01<00:00, 6.97MB
[   57.4s] poll 58468079,sub_v24_SRI440_pub.csv,2026-11-17 12:12:18.207000,v24.SRI440_pub | ρ(R2)=6.586e-4 ρ(440)=0.000e-4 | published sriram submission.csv (0.95440),SubmissionStatus.PENDING,,
[   76.2s] poll 58468079,sub_v24_SRI440_pub.csv,2026-11-17 12:12:18.207000,v24.SRI440_pub | ρ(R2)=6.586e-4 ρ(440)=0.000e-4 | published sriram submission.csv (0.95440),SubmissionStatus.COMPLETE,0.95440,
[   76.2s] DONE SRI440_pub pub=0.9544 status=ok
[   82.2s] SUBMIT v24.R2_to_SRI440_0p60e4 | ρ(R2)=0.600e-4 ρ(440)=3.214e-4 | blend w=0.3015 toward sri440 | ρ=0.600e-4
[   87.2s] Submission ref: 58468092
8 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[   87.2s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<01:23, 94.2kB/s]
 14%|█▎        | 1.02M/7.52M [00:00<00:01, 4.65MB/s]
 21%|██        | 1.58M/7.52M [00:00<00:02, 2.50MB/s]
 26%|██▌       | 1.95M/7.52M [00:00<00:02, 2.37MB/s]
 30%|███       | 2.27M/7.52M [00:01<00:02, 2.23MB/s]
 39%|███▊      | 2.91M/7.52M [00:01<00:01, 
[   95.9s] poll 58468092,sub_v24_R2_to_SRI440_0p60e4.csv,2026-11-17 12:12:56.730000,v24.R2_to_SRI440_0p60e4 | ρ(R2)=0.600e-4 ρ(440)=3.214e-4 | blend w=0.3015 toward sri440 | ρ=0.600e-4,SubmissionStatus.PENDING,,
[  114.7s] poll 58468092,sub_v24_R2_to_SRI440_0p60e4.csv,2026-11-17 12:12:56.730000,v24.R2_to_SRI440_0p60e4 | ρ(R2)=0.600e-4 ρ(440)=3.214e-4 | blend w=0.3015 toward sri440 | ρ=0.600e-4,SubmissionStatus.COMPLETE,0.95432,
[  114.7s] DONE R2_to_SRI440_0p60e4 pub=0.95432 status=ok
[  120.7s] SUBMIT v24.R2_dmlp_v15_w0.22 | ρ(R2)=3.306e-4 ρ(440)=5.009e-4 | R2 + 0.22*(L(N15)-L(C))
[  126.2s] Submission ref: 58468106
7 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[  126.2s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<03:31, 37.1kB/s]
 10%|▉         | 736k/7.52M [00:00<00:03, 1.83MB/s] 
 15%|█▌        | 1.14M/7.52M [00:01<00:07, 922kB/s]
 20%|█▉        | 1.47M/7.52M [00:01<00:05, 1.13MB/s]
 32%|███▏      | 2.39M/7.52M [00:01<00:02, 2.28MB/s]
 47%|████▋     | 3.52M/7.52M [00:01<00:01, 3
[  135.3s] poll 58468106,sub_v24_R2_dmlp_v15_w0.22.csv,2026-11-17 12:13:35.297000,v24.R2_dmlp_v15_w0.22 | ρ(R2)=3.306e-4 ρ(440)=5.009e-4 | R2 + 0.22*(L(N15)-L(C)),SubmissionStatus.PENDING,,
[  154.5s] poll 58468106,sub_v24_R2_dmlp_v15_w0.22.csv,2026-11-17 12:13:35.297000,v24.R2_dmlp_v15_w0.22 | ρ(R2)=3.306e-4 ρ(440)=5.009e-4 | R2 + 0.22*(L(N15)-L(C)),SubmissionStatus.COMPLETE,0.95432,
[  154.5s] DONE R2_dmlp_v15_w0.22 pub=0.95432 status=ok
[  160.5s] SUBMIT v24.DUAL_R2_GPW32_dmlp30 | ρ(R2)=7.011e-4 ρ(440)=1.970e-4 | sriram-style dual-base avg with R2+gpw32 @0.30 v15
[  165.7s] Submission ref: 58468120
6 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[  165.7s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<03:27, 37.8kB/s]
  7%|▋         | 512k/7.52M [00:00<00:11, 660kB/s]  
  7%|▋         | 576k/7.52M [00:01<00:11, 621kB/s]
 11%|█         | 816k/7.52M [00:01<00:09, 771kB/s]
 42%|████▏     | 3.14M/7.52M [00:01<00:00, 4.72MB/s]
 58%|█████▊    | 4.39M/7.52M [00:01<00:00, 6.32
[  174.5s] poll 58468120,sub_v24_DUAL_R2_GPW32_dmlp30.csv,2026-11-17 12:14:15.140000,v24.DUAL_R2_GPW32_dmlp30 | ρ(R2)=7.011e-4 ρ(440)=1.970e-4 | sriram-style dual-base avg with R2+gpw32 @0.30 v15,SubmissionStatus.PENDING,,
[  193.2s] poll 58468120,sub_v24_DUAL_R2_GPW32_dmlp30.csv,2026-11-17 12:14:15.140000,v24.DUAL_R2_GPW32_dmlp30 | ρ(R2)=7.011e-4 ρ(440)=1.970e-4 | sriram-style dual-base avg with R2+gpw32 @0.30 v15,SubmissionStatus.COMPLETE,0.95440,
[  193.2s] DONE DUAL_R2_GPW32_dmlp30 pub=0.9544 status=ok
[  199.2s] SUBMIT v24.SRI440_x_S95417_90_10 | ρ(R2)=6.795e-4 ρ(440)=1.006e-4 | Areeb on new high × shallow 0.95417
[  204.8s] Submission ref: 58468140
5 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[  204.8s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<03:21, 39.0kB/s]
  6%|▋         | 496k/7.52M [00:00<00:11, 658kB/s]  
  7%|▋         | 560k/7.52M [00:01<00:11, 617kB/s]
 10%|█         | 800k/7.52M [00:01<00:09, 775kB/s]
 27%|██▋       | 2.03M/7.52M [00:01<00:02, 2.82MB/s]
 42%|████▏     | 3.14M/7.52M [00:01<00:01, 4.45
[  214.0s] poll 58468140,sub_v24_SRI440_x_S95417_90_10.csv,2026-11-17 12:14:54.250000,v24.SRI440_x_S95417_90_10 | ρ(R2)=6.795e-4 ρ(440)=1.006e-4 | Areeb on new high × shallow 0.95417,SubmissionStatus.PENDING,,
[  232.7s] poll 58468140,sub_v24_SRI440_x_S95417_90_10.csv,2026-11-17 12:14:54.250000,v24.SRI440_x_S95417_90_10 | ρ(R2)=6.795e-4 ρ(440)=1.006e-4 | Areeb on new high × shallow 0.95417,SubmissionStatus.COMPLETE,0.95436,
[  232.7s] DONE SRI440_x_S95417_90_10 pub=0.95436 status=ok
[  238.7s] SUBMIT v24.R2_to_GPW32_0p35e4 | ρ(R2)=0.350e-4 ρ(440)=5.071e-4 | w=0.1879 ρ=0.350
[  244.5s] Submission ref: 58468156
4 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[  244.5s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<03:32, 37.1kB/s]
  9%|▊         | 656k/7.52M [00:00<00:08, 847kB/s]  
 10%|▉         | 752k/7.52M [00:01<00:08, 813kB/s]
 13%|█▎        | 976k/7.52M [00:01<00:07, 881kB/s]
 18%|█▊        | 1.36M/7.52M [00:01<00:05, 1.26MB/s]
 35%|███▌      | 2.64M/7.52M [00:01<00:01, 3.33
[  253.2s] poll 58468156,sub_v24_R2_to_GPW32_0p35e4.csv,2026-11-17 12:15:34.150000,v24.R2_to_GPW32_0p35e4 | ρ(R2)=0.350e-4 ρ(440)=5.071e-4 | w=0.1879 ρ=0.350,SubmissionStatus.PENDING,,
[  272.0s] poll 58468156,sub_v24_R2_to_GPW32_0p35e4.csv,2026-11-17 12:15:34.150000,v24.R2_to_GPW32_0p35e4 | ρ(R2)=0.350e-4 ρ(440)=5.071e-4 | w=0.1879 ρ=0.350,SubmissionStatus.COMPLETE,0.95430,
[  272.0s] DONE R2_to_GPW32_0p35e4 pub=0.9543 status=ok
[  278.0s] SUBMIT v24.R2_to_SRI440_1p20e4 | ρ(R2)=1.200e-4 ρ(440)=2.166e-4 | blend w=0.4265 toward sri440 | ρ=1.200e-4
[  283.0s] Submission ref: 58468168
3 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[  283.0s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<03:27, 37.9kB/s]
  7%|▋         | 512k/7.52M [00:00<00:10, 712kB/s]  
  8%|▊         | 624k/7.52M [00:01<00:10, 684kB/s]
 10%|█         | 800k/7.52M [00:01<00:09, 723kB/s]
 32%|███▏      | 2.39M/7.52M [00:01<00:01, 3.38MB/s]
 47%|████▋     | 3.52M/7.52M [00:01<00:00, 4.92
[  291.7s] poll 58468168,sub_v24_R2_to_SRI440_1p20e4.csv,2026-11-17 12:16:12.533000,v24.R2_to_SRI440_1p20e4 | ρ(R2)=1.200e-4 ρ(440)=2.166e-4 | blend w=0.4265 toward sri440 | ρ=1.200e-4,SubmissionStatus.PENDING,,
[  310.5s] poll 58468168,sub_v24_R2_to_SRI440_1p20e4.csv,2026-11-17 12:16:12.533000,v24.R2_to_SRI440_1p20e4 | ρ(R2)=1.200e-4 ρ(440)=2.166e-4 | blend w=0.4265 toward sri440 | ρ=1.200e-4,SubmissionStatus.COMPLETE,0.95434,
[  310.5s] DONE R2_to_SRI440_1p20e4 pub=0.95434 status=ok
[  316.5s] SUBMIT v24.R2_dmlp_v15_w0.30 | ρ(R2)=6.143e-4 ρ(440)=6.070e-4 | R2 + 0.30*(L(N15)-L(C))
[  321.9s] Submission ref: 58468189
2 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[  321.9s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<03:35, 36.4kB/s]
  9%|▊         | 656k/7.52M [00:00<00:08, 843kB/s]  
 10%|█         | 784k/7.52M [00:01<00:08, 864kB/s]
 13%|█▎        | 1.00M/7.52M [00:01<00:07, 952kB/s]
 17%|█▋        | 1.27M/7.52M [00:01<00:05, 1.26MB/s]
 19%|█▉        | 1.42M/7.52M [00:01<00:04, 1.2
[  330.8s] poll 58468189,sub_v24_R2_dmlp_v15_w0.30.csv,2026-11-17 12:16:51.343000,v24.R2_dmlp_v15_w0.30 | ρ(R2)=6.143e-4 ρ(440)=6.070e-4 | R2 + 0.30*(L(N15)-L(C)),SubmissionStatus.PENDING,,
[  350.0s] poll 58468189,sub_v24_R2_dmlp_v15_w0.30.csv,2026-11-17 12:16:51.343000,v24.R2_dmlp_v15_w0.30 | ρ(R2)=6.143e-4 ρ(440)=6.070e-4 | R2 + 0.30*(L(N15)-L(C)),SubmissionStatus.COMPLETE,0.95432,
[  350.0s] DONE R2_dmlp_v15_w0.30 pub=0.95432 status=ok
[  356.0s] SUBMIT v24.R2_x_S95417_95_5 | ρ(R2)=0.252e-4 ρ(440)=6.573e-4 | soft staged leftover
[  360.7s] Submission ref: 58468214
1 submissions remaining today.
Successfully submitted to Predicting Credit Card Fraud
[  360.7s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<00:49, 160kB/s]
  7%|▋         | 528k/7.52M [00:00<00:06, 1.17MB/s]
  8%|▊         | 624k/7.52M [00:00<00:08, 839kB/s] 
 11%|█         | 864k/7.52M [00:01<00:08, 862kB/s]
 30%|███       | 2.27M/7.52M [00:01<00:01, 3.28MB/s]
 45%|████▌     | 3.39M/7.52M [00:01<00:00, 4.82M
[  369.5s] poll 58468214,sub_v24_R2_x_S95417_95_5.csv,2026-11-17 12:17:30.423000,v24.R2_x_S95417_95_5 | ρ(R2)=0.252e-4 ρ(440)=6.573e-4 | soft staged leftover,SubmissionStatus.PENDING,,
[  388.7s] poll 58468214,sub_v24_R2_x_S95417_95_5.csv,2026-11-17 12:17:30.423000,v24.R2_x_S95417_95_5 | ρ(R2)=0.252e-4 ρ(440)=6.573e-4 | soft staged leftover,SubmissionStatus.COMPLETE,0.95426,
[  388.7s] DONE R2_x_S95417_95_5 pub=0.95426 status=ok
```

