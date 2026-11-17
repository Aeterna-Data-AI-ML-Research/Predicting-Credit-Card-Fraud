# GM V23.1b Execute-All Research Report

ATH reference `0.95426` · daily quota exhausted (0 remaining)

## Live submits

| tag | Public | vs ATH |
|---|---:|---:|
| `R2_gpw21_90_10` | **0.95426** | tie |
| `ALEX15_4188470e_ORTH_0p30` (LGB 3-leaf Part5, 5 seeds) | **0.95423** | −3e-5 |
| `SAMEER_ORTH_0p28e4` | **0.95425** | −1e-5 |
| `gpw21_S95417_90_10` | quota block | — |

## Staged for next reset (not submitted)

- `sub_v23_1b_gpw21_S95417_90_10.csv` — true Areeb peer recipe (ρ(R2)=6.67e-4)
- `sub_v23_1b_R2_S95417_95_5.csv` — soft shallow 5%
- `sub_v23_1b_SOFT_antiK2_0p5.csv`
- `sub_v23_1b_GPW_comp_ORTH_0p45e4.csv`
- `sub_v23_1b_CVMAX_PRIVATE_LOCK_NOSUBMIT.csv`

## Takeaway

Near-peer LM blend with goodpjw 0.95421 **ties** ATH. New alex-15 orth and pure sameer orth **do not lift**. Remaining EV tomorrow: submit `gpw21×S95417` Areeb recipe and soft S95417 5% first.

## Log (tail)
```
 39%|███▉      | 2.94M/7.52M [00:01<00:01, 
[  139.5s] poll 58421235,sub_v23_1b_SAMEER_ORTH_0p28e4.csv,2026-11-16 07:19:37.870000,v23.1b SAMEER_ORTH_0p28e4 | ρ(R2)=0.280e-4 ρ(sri)=1.571e-4 | ρ(R2)=0.280e-4,SubmissionStatus.PENDING,,
[  158.7s] poll 58421235,sub_v23_1b_SAMEER_ORTH_0p28e4.csv,2026-11-16 07:19:37.870000,v23.1b SAMEER_ORTH_0p28e4 | ρ(R2)=0.280e-4 ρ(sri)=1.571e-4 | ρ(R2)=0.280e-4,SubmissionStatus.COMPLETE,0.95425,
[  158.7s] DONE SAMEER_ORTH_0p28e4 pub=0.95425 status=ok
[  166.7s] SUBMIT v23.1b gpw21_S95417_90_10 | ρ(R2)=6.667e-4 ρ(sri)=5.989e-4 | Areeb recipe on available peers (gpw21≈0.95421 × S95417)
[  170.0s] 
[  170.0s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<01:27, 89.5kB/s]
  2%|▏         | 144k/7.52M [00:00<00:14, 523kB/s]  
  8%|▊         | 592k/7.52M [00:00<00:06, 1.07MB/s]
  9%|▉         | 720k/7.52M [00:00<00:06, 1.04MB/s]
 11%|█         | 832k/7.52M [00:00<00:06, 1.03MB/s]
 12%|█▏        | 928k/7.52M [00:01<00:07, 984k
[  170.0s] DONE gpw21_S95417_90_10 pub=None status=quota_or_error
[  170.0s] STOP: quota exhausted or submit error
Traceback (most recent call last):
  File "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/scripts/gm_v23_1_execute_all_research.py", line 476, in <module>
    main()
    ~~~~^^
  File "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/scripts/gm_v23_1_execute_all_research.py", line 459, in main
    json.dumps(
    ~~~~~~~~~~^
        {
        ^
    ...<8 lines>...
        default=float,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/Versions/3.14/lib/python3.14/json/__init__.py", line 242, in dumps
    **kw).encode(obj)
          ~~~~~~^^^^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/Versions/3.14/lib/python3.14/json/encoder.py", line 202, in encode
    chunks = self.iterencode(o, _one_shot=True)
  File "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/Versions/3.14/lib/python3.14/json/encoder.py", line 263, in iterencode
    return _iterencode(o, 0)
TypeError: float() argument must be a string or a real number, not 'PosixPath'
when serializing dict item 'path'
when serializing list item 0
when serializing dict item 'staged'
```
