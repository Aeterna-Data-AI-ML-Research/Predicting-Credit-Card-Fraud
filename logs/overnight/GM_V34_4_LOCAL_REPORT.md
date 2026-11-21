# GM V34.4 LOCAL BUGFIX & RECON REPORT

- NO_SUBMIT: `True`
- elapsed_s: 145.0
- peak_rss_gb: 2.527 (cap 2.3)
- pods/cloud: `0` / `$0.00`

## Part 0 — Control Gate (Bug #1)

- v10_ridge_orth_LM: 5/5 22/25 t25=6.17 mean=0.271e-4 nz=25 w=+0.0080
- -harvest_p4_cuda_xgb: 5/5 16/25 t25=2.43 mean=0.123e-4 nz=25 w=-0.0080

## Part 1 — Kaggle Recon (Bug #4: Warning-stripped)

### lb
```
Next Page Token = CfDJ8CXkJvnxKG5Ktgac76JjV5HqWnL3piFeELvuBi0BujUEGfGfXkEk9omCjJq72YSEQdV5S8v2CQWQTsSKv2UnqOI
  teamId  teamName                submissionDate              score    
--------  ----------------------  --------------------------  -------  
17268683  Dharun MK               2026-11-21 00:06:37.450000  0.96153  
17244109  Smart Manoj             2026-11-21 01:24:19.653000  0.96070  
17244552  강북구집돌이                  2026-11-21 01:19:49.256000  0.95754  
17243337  Ahmet Bugrahan ALTUNOK  2026-11-20 18:01:31.913000  0.95632  
17337252  srikalyan               2026-11-21 00:33:45.993000  0.95627  
17392080  Weni Mohamed            2026-11-21 10:28:53.900000  0.95621  
17356299  Lebxgue.                2026-11-21 10:21:42.570000  0.95613  
17244702  EunSeob Sim             2026-11-21 14:07:45.833000  0.95568  
17407149  Arpit Jain              2026-11-21 10:31:16.396000  0.95565  
17269262  Japan OpenData Lab      2026-11-21 13:27:08.600000  0.95565  
17251560  starkhushi              2026-11-21 07:07:01.353000  0.95565  
17265195  odie7710                2026-11-21 09:33:00.156000  0.95559  
17244399  Aman Atar               2026-11-21 15:55:49.240000  0.95551  
17244439  kazuhiro3381            2026-11-21 02:26:49.793000  0.95551  
17245310  奶龙                      2026-11-21 14:22:58.053000  0.95547  
17317597  jvachier                2026-11-21 14:31:58.936000  0.95547  
17245747  Areeb Ali               2026-11-21 13:57:03.070000  0.95547  
17352647  IEEM-AI-2026-114034558  2026-11-21 15:46:10.666000  0.95547  
17249269  yukiZ                   2026-11-21 14:31:06.530000  0.95547  
17312842  Ansh Kashyap877         2026-11-21 14:31:24.730000  0.95547  
```

### kernels_score
```
ref,title,author,lastRunTime,totalVotes
daoviet/s6e11-on-the-journey-to-your-first-top-100,S6E11 | On the journey to your first top 100 🚀,daoviet,2026-11-21 15:21:43.123000,21
asteriospanoras/s6e11-lb-0-95547-will-the-top-survive-private,S6E11 | LB 0.95547 | Will the Top Survive Private?,Asterios Panoras,2026-11-21 15:05:33.867000,54
areebbb/s6e11-which-signals-rank-fraud-lb-0-95547,S6E11 | Which Signals Rank Fraud? | LB 0.95547,Areeb Ali,2026-11-21 14:07:47.893000,6
dennis6886/s6e11-which-signals-rank-fraud-v3-reproduction,S6E11 Which Signals Rank Fraud V3 reproduction,奶龙,2026-11-21 14:08:24.377000,1
dennis6886/s6e11-catboost-within-fraud-risk-bands,S6E11 CatBoost within fraud risk bands,奶龙,2026-11-21 14:15:34.533000,2
goodpjw2008/s6e11-board-model-realmlp-lb-0-95546,S6E11 | Board Model + RealMLP | LB 0.95546,goodpjw2008,2026-11-21 13:35:35.030000,46
dennis6886/s6e11-fraud-signal-rank-fusion-four-refinements,S6E11 Fraud signal rank fusion four refinements,奶龙,2026-11-21 11:48:56.980000,2
malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion,Fraud Signal Lab | GPT-2 and GLM Fusion,Malyshev Danil,2026-11-21 11:36:32.933000,47
bhuvannr2006/s6e11-40-scored-files-one-forecast-lb-0-95534,"S6E11 | 40 Scored Files, One Forecast | LB 0.95534",BhuvanNR,2026-11-21 02:19:22.783000,13
rishabhbafnaiiitd/credit-card-fraud-gram-schmidt-optimization,Credit Card Fraud: Gram-Schmidt Optimization,Rishabh Bafna,2026-11-20 19:08:12.880000,9
mayankraaz/s6e11-fraud-lab-learn-build-review,"S6E11 | Fraud Lab: Learn, Build & Review",MAYANK RAJ,2026-11-20 09:02:21.827000,4
alexchilton/credit-card-fraud-lb-0-9550-shallow-tree-stack,Credit Card Fraud | LB 0.9550 | Shallow-Tree Stack,alex chilton,2026-11-19 07:08:40.490000,72
rishabhbafnaiiitd/credit-card-fraud-digit-forensics-frontier-fusion,Credit Card Fraud Digit Forensics Frontier Fusion,Rishabh Bafna,2026-11-19 01:33:34.973000,3
busyaprime/s6e11-124-voices-the-top-rebuilt-lb-0-95465,S6E11 | 124 Voices + the Top Rebuilt | LB 0.95465,Akmal Xodarev (Busya PRIME),2026-11-18 00:11:40.867000,11
goodpjw2008/s6e11-lm-ladder-ladder-meets-stack-lb-0-95432,S6E11 LM Ladder | Ladder Meets Stack | LB 0.95432,goodpjw2008,2026-11-17 06:40:30.427000,61
areebbb/s6e11-split-then-push-lb-0-95431,"S6E11 | Split, Then Push | LB 0.95431",Areeb Ali,2026-11-17 04:53:50.747000,65
saamhm/eda-blend-lb-0-95239-credit-card-fraud,EDA | blend | LB-0.95239 | Credit Card Fraud,samsul hoque mondal,2026-11-17 06:00:11.917000,63
bhuvannr2006/s6e11-lm-ladder-forecast-the-board-lb-0-95417,S6E11 LM Ladder | Forecast the Board | LB 0.95417,BhuvanNR,2026-11-15 13:02:08.287000,8
muhammadshoaibaltaf/s6e11-0-95521-learning-to-rerank-step-l,S6E11 | 0.95521 | Learning to Rerank Step L,Muhammad Shoaib Altaf,2026-11-20 17:58:57.890000,12
amanatar/s6e11-sovereign-0-95447-frontier-dual-shield,S6E11 | Sovereign 0.95447+ Frontier & Dual Shield,Aman Atar,2026-11-17 15:55:10.213000,8
mobeenfatimah/s6e11-getting-high-score-on-lb,S6E11 | Getting High Score on LB,Mobeen Fatima,2026-11-14 10:19:25.087000,6
modansaif/s6e11-fraud-fusion-reproduction-credited,S6E11 Fraud Fusion Reproduction (credited),SAIF MODAN,2026-11-12 18:36:26.740000,0
davutkursun/s6e11-what-worked-what-didn-t-lb-0-9526,S6E11 | What Worked & What Didn't | LB 0.9526,DavutKursun,2026-11-20 19:41:23.100000,11
amanatar/s6e11-podium-apex-public-lb-0-95340,S6E11 | Podium Apex | Public LB 0.95340,Aman Atar,2026-11-10 13:58:10.947000,4
```

### kernels_date
```
ref,title,author,lastRunTime,totalVotes
hongbotrn/s6e11-fraud-research,s6e11_fraud_research,Hoàng Bảo Trần,2026-11-21 16:02:40.733000,0
obiaf88/predicting-credit-fraud,Predicting Credit Fraud,De Marchi Fabio,2026-11-21 15:43:02.747000,0
olegpan/s6e11-logreg-baseline-eceab6,S6E11 | LogReg | BaseLine eceab6,Oleg Pan,2026-11-21 15:35:25.193000,0
daoviet/s6e11-on-the-journey-to-your-first-top-100,S6E11 | On the journey to your first top 100 🚀,daoviet,2026-11-21 15:21:43.123000,21
asteriospanoras/s6e11-lb-0-95547-will-the-top-survive-private,S6E11 | LB 0.95547 | Will the Top Survive Private?,Asterios Panoras,2026-11-21 15:05:33.867000,54
mnkaggler/credit-card-fraud-detection-xgboost-lgbm,Credit Card Fraud Detection | XGBoost & LGBM,MN Kaggler,2026-11-21 14:40:48.553000,0
dennis6886/s6e11-catboost-within-fraud-risk-bands,S6E11 CatBoost within fraud risk bands,奶龙,2026-11-21 14:15:34.533000,2
dennis6886/s6e11-which-signals-rank-fraud-v3-reproduction,S6E11 Which Signals Rank Fraud V3 reproduction,奶龙,2026-11-21 14:08:24.377000,1
areebbb/s6e11-which-signals-rank-fraud-lb-0-95547,S6E11 | Which Signals Rank Fraud? | LB 0.95547,Areeb Ali,2026-11-21 14:07:47.893000,6
goodpjw2008/s6e11-board-model-realmlp-lb-0-95546,S6E11 | Board Model + RealMLP | LB 0.95546,goodpjw2008,2026-11-21 13:35:35.030000,46
goutamsharma12/spline-logistic-regression,Spline Logistic Regression,Goutam sharma,2026-11-21 12:13:57.523000,1
dennis6886/s6e11-fraud-signal-rank-fusion-four-refinements,S6E11 Fraud signal rank fusion four refinements,奶龙,2026-11-21 11:48:56.980000,2
malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion,Fraud Signal Lab | GPT-2 and GLM Fusion,Malyshev Danil,2026-11-21 11:36:32.933000,47
skull237/mori23784aab3d79b,mori23784aab3d79b,skull 237,2026-11-21 11:21:11.280000,0
dhananjayk2609/s6e11-fraud-part-3-final-blend-lb-0-953,"S6E11 Fraud | Part 3: Final Blend, LB 0.953",Dhananjay Kothawale,2026-11-21 10:40:19.250000,4
dhananjayk2609/s6e11-fraud-part-2-eda-and-modelling-plan,S6E11 Fraud | Part 2: EDA and Modelling Plan,Dhananjay Kothawale,2026-11-21 10:40:00.253000,15
dhananjayk2609/s6e11-fraud-part-1-shared-helper-module,S6E11 Fraud | Part 1: Shared Helper Module,Dhananjay Kothawale,2026-11-21 10:39:55.137000,8
mathsonchain/s6e11-fraud-what-worked-what-didn-t,"S6E11 Fraud: What Worked, What Didn’t",MathsOnChain,2026-11-21 10:15:20.907000,9
davutkursun/s6e11-the-0-9526-wall-who-still-adds-signal,S6E11 | The 0.9526 Wall: Who Still Adds Signal?,DavutKursun,2026-11-21 10:01:04.167000,1
hittmg/261121playground-s6e11day3-1-lgbm-optuna,261121Playground S6E11Day3-1 LGBM Optuna,HitTmg,2026-11-21 08:17:01.093000,0
gauravduttaiiitb/ex30-s6e11-flaml-roc-auc-lgbm,Ex30 : S6E11 : FLAML : roc_auc : lgbm,Gaurav Dutta,2026-11-21 07:45:59.947000,0
logeshwari12345/predicting-credit-card-fraud,Predicting Credit Card Fraud,Logeshwari Somasundaram,2026-11-21 07:18:00.373000,5
hittmg/261121playground-s6e11day3-4-hgb-optuna,261121Playground S6E11Day3-4 HGB Optuna,HitTmg,2026-11-21 07:16:11.347000,0
hittmg/261121playground-s6e11day3-3-cb-optuna,261121Playground S6E11Day3-3 CB Optuna,HitTmg,2026-11-21 07:13:14.613000,0
```

### ds_s6e11
```
ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating
moranzavdi/s6e11-strike9-lb-095330,S6E11 Strike9 LB 0.95330 Frozen Submission,3619611,2026-11-09 09:08:05.577000,219,0,0.64705884
goodpjw2008/s6e11-fraud-series-artifacts,S6E11 Fraud Series Artifacts,135687080,2026-11-07 23:52:13.130000,2258,6,0.4375
goodpjw2008/s6e11-oof-bank,"S6E11 OOF Bank: 70 Members, Same 5 Folds",259968203,2026-11-03 16:00:24.543000,3927,4,0.64705884
masayakawamata/s6e11-gpt2-target-first-oof,S6E11 GPT-2 Target-First OOF and Test Predictions,7731555,2026-11-03 13:21:09.043000,817,3,0.5
wguesdon/s6e11-tree-ensemble-predictions,S6E11 Tree Ensemble Predictions,115874960,2026-11-01 13:52:49.857000,550,1,0.47058824
busyaprime/s6e11-33-own-members,"S6E11 Own Members, 52 OOF and Test Predictions",196624446,2026-11-18 00:04:51.127000,374,3,0.5294118
megayak/s6e11-target-first-lm,S6E11 target-first LM predictions,4601297,2026-11-19 00:45:33.823000,1091,2,0.3125
asteriospanoras/s6e11-public-board-files,S6E11 Public Board Files (pinned copies),99693558,2026-11-20 18:44:21.027000,330,2,0.5294118
georgymamarin/s6e11-audited-glm-offset-blend,S6E11 Fraud: AUC vs Review Budgets,2333,2026-11-05 19:38:35.113000,95,14,0.47058824
malyshevdanil/s6e11-pooled-stack-v9,S6E11 Pooled Stack V9 Artifacts,122363140,2026-11-03 02:38:22.743000,530,2,0.29411766
sometimessubodh/ps6e11-dataset,PS6E11 38 Features and Leak-Safe OOF/Test Preds,131372929,2026-11-09 11:51:13.290000,574,4,1
dariushafshar/s6e11-golem-oof-library,"S6E11 OOF library - 9 models, frozen folds",76962207,2026-11-10 15:38:57.440000,419,2,0.9411765
nina2025/ps-s6e11-03,PS-s6e11-03,52826565,2026-11-06 17:27:39.573000,129,3,0.1764706
amanatar/s6e11-rank1-q17-meta-bundle,S6E11 Rank #1 Q17 Meta Bundle,64592945,2026-11-12 14:04:45.277000,360,3,0.1875
mathsonchain/s6e11-fraud-oof-test-17-single-models,"S6E11 Fraud: OOF + Test, 17 Single Models",120568923,2026-11-07 21:47:57.393000,601,1,0.75
amanatar/s6e11-sovereign-095423-simplex-and-oof-vault,S6E11 Sovereign 0.95423+ Simplex & OOF Vault,50633003,2026-11-16 18:45:26.847000,128,2,0.64705884
sadamtorres/ps-s6e11-tabpfn-gbdt-oof-test-pairs,PS S6E11 · TabPFN-3.5 + GBDT OOF/test pairs,149150734,2026-11-04 19:51:32.317000,1121,1,0.64705884
malyshevdanil/s6e11-digit-token-full-refit-artifacts,S6E11 Digit Token Full Refit Artifacts,8588007,2026-11-02 08:22:01.453000,307,3,0.3125
goodpjw2008/s6e11-compact-lm-seed7,S6E11 Compact Target-First LM Seed 7,29140672,2026-11-12 08:04:21.997000,711,1,0.375
goodpjw2008/s6e11-target-first-gpt2-seed-2026,S6E11 Target-First GPT-2 Seed 2026,9723214,2026-11-14 18:23:23.293000,534,1,0.375
```

### ds_cc2026
```
ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating
uditjain13/credit-card-fraud-detection-2026,Credit Card Fraud Detection 2026,665173,2026-07-16 19:17:54.650000,5293,70,1
mohankrishnathalla/credit-card-fraud-detection-dataset-2026,Credit Card Fraud Detection Dataset 2026,6976318,2026-11-20 17:57:55.380000,1312,29,1
mohankrishnathalla/credit-card-fraud-detection-with-drift-2026,Credit Card Fraud Detection with Drift 2026,8689341,2026-11-20 17:57:58.967000,210,5,1
hassanjameelahmed/e-commerce-customer-ds,ecommerce-customer-behavior-analysis-eda,1007281,2026-01-08 11:07:25.823000,177,16,1
dianatofficial/payments-fraud-operations-benchmark-2026,Synthetic Payments Fraud Operations 2026,3383372,2026-10-11 04:43:03.647000,88,5,1
busyaprime/s6e11-33-own-members,"S6E11 Own Members, 52 OOF and Test Predictions",196624446,2026-11-18 00:04:51.127000,374,3,0.5294118
masayakawamata/s6e11-gpt2-target-first-oof,S6E11 GPT-2 Target-First OOF and Test Predictions,7731555,2026-11-03 13:21:09.043000,817,3,0.5
generaterealdata/synthetic-card-transactions-fraud-labels,Synthetic Card Transactions with Fraud Labels,22976758,2026-10-13 13:31:16.730000,26,1,1
goodpjw2008/s6e11-fraud-series-artifacts,S6E11 Fraud Series Artifacts,135687080,2026-11-07 23:52:13.130000,2258,6,0.4375
nicolasssoares/brazilian-benefit-card-transactions-synthetic,Brazilian Benefit-Card Transactions (Synthetic),864955925,2026-11-02 18:07:31.813000,12,1,0.9411765
dariushafshar/s6e11-golem-oof-library,"S6E11 OOF library - 9 models, frozen folds",76962207,2026-11-10 15:38:57.440000,419,2,0.9411765
bhavyashah7645/fraudxai-us-100k-seed42,FraudxAI US Card Transactions (100k Synthetic),62585267,2026-10-11 10:36:46.483000,106,1,1
generaterealdata/banking-payments-knowledge-graph-neo4j,Bank Fraud & AML Knowledge Graph (Neo4j),76899614,2026-11-18 15:14:16.480000,4,2,1
bhavyashah7645/fraudxai-india-100k-seed42,FraudxAI India Card Transactions (100k Synthetic),64176189,2026-10-11 13:44:09.547000,82,1,1
quantumkaran/indian-credit-card-fraud-detection-dataset,Indian Credit Card Fraud Detection Dataset,31854323,2026-07-26 18:25:07.507000,62,1,0.47058824
greatcool/financial-transactions-and-fraud-detection-dataset,Financial Transactions & Fraud Detection Dataset,194351,2026-08-26 16:06:16.167000,129,0,1
aryansingh2000/indian-credit-card-analytics,Indian credit card analytics dataset,41046378,2026-11-14 18:28:38.077000,11,1,0.7647059
aryansingh2000/indian-credit-card-usages-and-analysis,Indian Credit Card usages and analysis,41046378,2026-11-18 04:58:27.197000,14,1,0.64705884
quailmodel/card-fraud-v1-sample,QM Synthetic Card Fraud Transactions (Sample),13237755,2026-10-09 23:00:26.517000,30,1,0.88235295
roshaaann30/financial-transaction-intelligence-dataset-2026,Financial Transaction Intelligence Dataset 2026,670352,2026-07-23 06:29:25.127000,160,0,0.88235295
```

### ds_udit
```
ref,title,size,lastUpdated,downloadCount,voteCount,usabilityRating
uditjain13/adaption-financial-math-qa,Adaption Financial Math QA (Personal Finance),2356400,2026-07-25 10:56:37.883000,16,2,0.625
uditjain13/nemotron-30b-science-expert-peft,Nemotron-30B Science Expert PEFT,22000223,2026-04-22 16:25:05.113000,54,4,0.9375
uditjain13/nemotron-30b-code-hyper-reasoner-peft,Nemotron-30B Code Hyper-Reasoner PEFT,22011029,2026-04-22 16:23:44.847000,58,4,0.9375
uditjain13/nemotron-30b-math-reasoner-peft,Nemotron-30B Math Reasoner PEFT,22010886,2026-04-22 16:23:03.147000,74,3,0.9375
uditjain13/nemotron-30b-multi-domain-merged-peft,Nemotron-30B Multi-Domain Merged PEFT,30569278,2026-04-22 16:15:05.080000,118,4,0.9375
```

### ds_scam
```
No datasets found
```

### Flags

- BANKISH lb: ['  teamId  teamName                submissionDate              score    ', '--------  ----------------------  --------------------------  -------  ', '17268683  Dharun MK               2026-11-21 00:06:37.450000  0.96153  ']
- HIGH_KERNEL asteriospanoras/s6e11-lb-0-95547-will-the-top-survive-private,S6E11 | LB 0.95547 | Will the Top Survive Private?,Asterios Panoras,2026-11-21 15:05:33.867000,54
- HIGH_KERNEL areebbb/s6e11-which-signals-rank-fraud-lb-0-95547,S6E11 | Which Signals Rank Fraud? | LB 0.95547,Areeb Ali,2026-11-21 14:07:47.893000,6
- HIGH_KERNEL goodpjw2008/s6e11-board-model-realmlp-lb-0-95546,S6E11 | Board Model + RealMLP | LB 0.95546,goodpjw2008,2026-11-21 13:35:35.030000,46
- BANKISH kernels_score: ['daoviet/s6e11-on-the-journey-to-your-first-top-100,S6E11 | On the journey to your first top 100 🚀,da', 'asteriospanoras/s6e11-lb-0-95547-will-the-top-survive-private,S6E11 | LB 0.95547 | Will the Top Surv', 'areebbb/s6e11-which-signals-rank-fraud-lb-0-95547,S6E11 | Which Signals Rank Fraud? | LB 0.95547,Are']
- HIGH_KERNEL asteriospanoras/s6e11-lb-0-95547-will-the-top-survive-private,S6E11 | LB 0.95547 | Will the Top Survive Private?,Asterios Panoras,2026-11-21 15:05:33.867000,54
- HIGH_KERNEL areebbb/s6e11-which-signals-rank-fraud-lb-0-95547,S6E11 | Which Signals Rank Fraud? | LB 0.95547,Areeb Ali,2026-11-21 14:07:47.893000,6
- HIGH_KERNEL goodpjw2008/s6e11-board-model-realmlp-lb-0-95546,S6E11 | Board Model + RealMLP | LB 0.95546,goodpjw2008,2026-11-21 13:35:35.030000,46
- BANKISH kernels_date: ['hongbotrn/s6e11-fraud-research,s6e11_fraud_research,Hoàng Bảo Trần,2026-11-21 16:02:40.733000,0', 'obiaf88/predicting-credit-fraud,Predicting Credit Fraud,De Marchi Fabio,2026-11-21 15:43:02.747000,0', 'olegpan/s6e11-logreg-baseline-eceab6,S6E11 | LogReg | BaseLine eceab6,Oleg Pan,2026-11-21 15:35:25.1']
- BANKISH ds_s6e11: ['moranzavdi/s6e11-strike9-lb-095330,S6E11 Strike9 LB 0.95330 Frozen Submission,3619611,2026-11-09 09:', 'goodpjw2008/s6e11-fraud-series-artifacts,S6E11 Fraud Series Artifacts,135687080,2026-11-07 23:52:13.', 'goodpjw2008/s6e11-oof-bank,"S6E11 OOF Bank: 70 Members, Same 5 Folds",259968203,2026-11-03 16:00:24.']
- BANKISH ds_cc2026: ['uditjain13/credit-card-fraud-detection-2026,Credit Card Fraud Detection 2026,665173,2026-07-16 19:17', 'mohankrishnathalla/credit-card-fraud-detection-dataset-2026,Credit Card Fraud Detection Dataset 2026', 'mohankrishnathalla/credit-card-fraud-detection-with-drift-2026,Credit Card Fraud Detection with Drif']

## Part 2 — z_prop_clean + MID37 twin

- MID37 twin test R²=0.953779
- selected=['v10_ridge_orth_LM', 'harvest_p4_cuda_xgb', 'P4_L4.3_pure5f', 'd1_v4', 'P4_L4.5_pure5f']
- AUC(y, z_mid37_oof)=0.951058

## Engine Table (Bug #3/#5: MID37-orthogonal)

| module | solo_AUC | Δz_wide e-4 | Δz_mid37_twin e-4 | outer | sub | t25 | p25 | KS_test | Δβ_in_MID37 | cos_loser |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| v10_ridge_orth_LM | 0.536503 | 0.271 | 0.271 | 5/5 | 22/25 | 6.17 | 2.086e-06 | nan | nan | nan |
| -harvest_p4_cuda_xgb | 0.473781 | 0.123 | 0.123 | 5/5 | 16/25 | 2.43 | 0.01275 | nan | nan | nan |
| z_prop_clean | 0.756255 | 0.255 | 0.236 | 2/5 | 14/25 | 0.82 | 0.2289 | nan | nan | nan |
| E2_fixed | 0.937583 | -0.344 | 0.167 | 1/5 | 13/25 | -1.23 | 0.7711 | nan | nan | nan |
| E3_fixed | 0.952563 | -0.583 | -0.008 | 0/5 | 5/25 | -2.38 | 0.9947 | nan | nan | nan |
| z_final_clean | 0.949611 | 0.466 | 0.411 | 4/5 | 17/25 | 1.59 | 0.02256 | 0.0491 | 0.1171 | -0.1688 |

## Staging

- staged=[]
- passers_in_final=['wide', 'z_prop_clean', 'E2_fixed']

## Verdict

- No module cleared the MID37-safe staging bar. Report only.

**STRICT NO_SUBMIT=True — 0 submits, 0 pods, $0 cloud.**
