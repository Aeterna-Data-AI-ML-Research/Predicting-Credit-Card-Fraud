# V47 STEP1 — Schema Sister-Dataset Scan

**Goal:** Explain rumor scores like `0.96153` and map schema sisters.

## Verdict

Public LB scores ≥0.961 typically come from (a) different evaluation slices, (b) notebooks that still blend public tip CSVs, or (c) datasets/schemas that are sisters of Udit Jain 2026 orig20k — NOT from a verified sovereign self-train stack. Local orig20k: 20000 rows, 25/25 train-col overlap, schema flags present=True.

## Local orig20k

- path: `/tmp/s6e11_v39_ssd/data/orig20k.csv`
- rows: **20000** · cols: **26**
- overlap with train: **25** → `['account_balance_usd', 'amount_usd', 'auth_method', 'billing_shipping_mismatch', 'card_age_months', 'card_type', 'channel', 'customer_age', 'cvv_retry_count', 'day_of_week', 'device_type', 'distance_from_home_km', 'hours_since_last_txn', 'ip_country_mismatch', 'is_ai_generated_scam_attempt', 'is_foreign_transaction', 'is_fraud', 'is_new_merchant', 'merchant_category', 'merchant_risk_score', 'prior_disputes', 'time_of_day_hour', 'txn_count_last_24h', 'used_vpn', 'velocity_score']`
- `is_ai_generated_scam_attempt`: True
- `cvv_retry_count`: True
- frauds: 339

## HuggingFace hits (top)

- `uditjain/adaption-financial-math-qa` (q=uditjain, dl=529)
- `uditjain/veneer-bench` (q=uditjain, dl=87)

## OpenML hits

- (none / API miss)

## Kaggle dataset list excerpts

### query `is_ai_generated_scam_attempt`

```
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
No datasets found

```

### query `cvv_retry_count`

```
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
No datasets found

```

### query `Credit Card Fraud Detection 2026`

```
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
ref                                                             title                                              size  lastUpdated                 downloadCount  voteCount  usabilityRating  
--------------------------------------------------------------  -------------------------------------------  ----------  --------------------------  -------------  ---------  ---------------  
mohankrishnathalla/credit-card-fraud-detection-dataset-2026     Credit Card Fraud Detection Dataset 2026        6976318  2026-11-25 17:02:31.600000           1436         29                1  
mohankrishnathalla/credit-card-fraud-detection-with-drift-2026  Credit Card Fraud Detection with Drift 2026     8689341  2026-11-25 17:02:35.233000            286          5                1  
rajandasthrice/credittransact-a-profile-driven-dataset          CreditTransAct: A Profile-Driven Dataset              0  2026-10-05 16:14:20.140000              7          0        0.4117647  

```

### query `uditjain`

```
Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.4), please consider upgrading to the latest version (2.2.5)
No datasets found

```

## Notes / errors

- OpenML credit card fraud: HTTPError: HTTP Error 412: Precondition Failed
- OpenML udit jain: HTTPError: HTTP Error 412: Precondition Failed
- OpenML cvv_retry: HTTPError: HTTP Error 412: Precondition Failed
