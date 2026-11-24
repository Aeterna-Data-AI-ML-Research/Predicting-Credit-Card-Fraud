# V41 Recon + Forward-Secant / UID Audit

**NO_SUBMIT=True** · LB_01=0.95569 · LB_02=0.95568 · LB_06=0.95568

## Quarantine / Hold
- QUARANTINE: SLOT03,04,05,07 (areeb-family) + **SLOT08** (−C3 axis)
- HOLD: SLOT09, SLOT10 (untouched)

## Public Recon
- **No public kernel/dataset tip > 0.95569** since Step_AB (top public still 0.95569).
- No `Step_AC` / new ladder tip published.

## Candidate Summary

| Cand | 1−ρ vs AB e4 | α | cos·d_AB_AA | cos·S02↓ | cos·S06↓ | OOF Δe5 | pos/5 | Zr_ok | loser≤0? |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| FWD_AB_areeb | 0.0250 | 0.30524 | 0.6403 | -0.9905 | -0.1083 | N/A | N/A | True | True |
| FWD_AB_AA | 0.0300 | 0.37896 | 0.6570 | -0.9582 | -0.0863 | N/A | N/A | True | True |
| UID_clean | 0.2000 | 0.01444 | -0.0373 | 0.0409 | 0.0057 | 0.00 | 3/5 | True | False |

### Pairwise 1−ρ (×1e-4)

| |AB|FWD_areeb|FWD_AA|UID|S02|S06|P1s_uid|
|---|---|---|---|---|---|---|---|
| AB |0.000|0.025|0.030|0.200|0.028|0.280|0.246|
| FWD_areeb |0.025|0.000|0.002|0.229|0.105|0.329|0.241|
| FWD_AA |0.030|0.002|0.000|0.236|0.113|0.330|0.238|
| UID |0.200|0.229|0.236|0.000|0.223|0.477|0.858|
| S02 |0.028|0.105|0.113|0.223|0.000|0.282|0.305|
| S06 |0.280|0.329|0.330|0.477|0.282|0.000|0.556|
| P1s_uid |0.246|0.241|0.238|0.858|0.305|0.556|0.000|

### Files
- `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/out/v41_recon_candidates/Cand_FWD_AB_areeb_0p025e4.csv` sha=e50346bd099e96a0317f98baa00c8c673112f05463a5b0b8eb5f4b43342802ef
- `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/out/v41_recon_candidates/Cand_FWD_AB_AA_0p030e4.csv` sha=6feb162b7a617f435cf81dd3a29ca825f75ebf7aa918adac09c6f01ee9245a37
- `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/out/v41_recon_candidates/Cand_UID_clean_clip985.csv` sha=76e734e81c6deff9320a8117d808f43b3ac57bc4c43b6b40dc35626637a081a7
- staged P1s_AB_uid_a0p015 sha=b8c5b68dccde6941a9648421972655b28bcabe8e6602221dea186980aa30e757 1−ρ=0.2459e-4

## STOP
NO_SUBMIT=True. Awaiting explicit single-slot fire command.
