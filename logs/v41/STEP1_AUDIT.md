# V41 STEP1 AUDIT — Frontier Harvest & Exact Reconstruction

**NO_SUBMIT=True** · Hard stop after this file. Do not run Step 2 without `PROCEED TO STEP 2`.

## 0. Critical formula finding (read first)

Public description “0.26 × clip99(Y−Z)” is **Malyshev’s documentation of an earlier dennis direction**, not the executed `areeb 0.95568` recipe.

**`areeb 0.95568` exact space (notebook cell 10):**
1. Inputs: Malyshev `own_v42`, goodpjw `Step Z`, goodpjw `Step AA` (sha `51706855…`), test zone mask.
2. Build quarter-step (our `0.95567`, sha `60ec5bd1…`) in **logit(uniform rank)** with zone demotion.
3. `residual = logit_rank(Step_AA) − logit_rank(quarter)`; `cap = quantile(|residual| outside zones, 0.95)`; `score = quarter + 0.70 * clip(residual, ±cap)` outside zones; zones −100; emit uniform ranks.

**Current `dennis` notebook (executed CSV):**
`parent + 0.12 * clip_p985(areeb_095568 − parent)` in logit-rank space (parent sha `/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/v41_nov23_frontier/goodpjw_95569/submission.csv` / found=True).

## 1. SHA-256 & File Integrity Table

|  | reported_LB | sha16 | rows | min | max | mean | is_uniform_rank | n_nan | n_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| our_095567 | 0.95567 | 60ec5bd11eca0445 | 299984 | 1.66676e-06 | 0.999998 | 0.5 | True | 0 | 0 |
| Step_V | 0.95558 | ac6962c8f32f6d13 | 299984 | 1.66676e-06 | 0.999998 | 0.5 | True | 0 | 0 |
| Step_W | 0.95559 | abc5d4a49535086e | 299984 | 1.66676e-06 | 0.999998 | 0.5 | True | 0 | 0 |
| Step_X | 0.9556 | 5c7e5518ac4beae1 | 299984 | 1.66676e-06 | 0.999998 | 0.5 | True | 0 | 0 |
| Step_Y_095561 | 0.95561 | 6a153c97a54fe055 | 299984 | 1.66676e-06 | 0.999998 | 0.5 | True | 0 | 0 |
| Step_Z_095562 | 0.95562 | ee61b50611326eb6 | 299984 | 1.66676e-06 | 0.999998 | 0.5 | True | 0 | 0 |
| Step_AA | 0.95569 | 51706855fa6de98e | 299984 | 1.66676e-06 | 0.999998 | 0.5 | True | 0 | 0 |
| areeb_095568 | 0.95568 | 3b9fe6d72e310210 | 299984 | 1.66676e-06 | 0.999998 | 0.5 | True | 0 | 0 |
| dennis_clipped | nan | 7de7fcc291f699bf | 299984 | 1.66676e-06 | 0.999998 | 0.5 | True | 0 | 0 |
| malyshev_v42 | 0.95565 | cde594c2098f4e07 | 299984 | 1.66676e-06 | 0.999998 | 0.5 | True | 0 | 0 |
| dennis_Nov23_npz | nan | c226b49911d20295 | 299984 | 1.66676e-06 | 0.999998 | 0.5 | True | 0 | 0 |

Full SHA-256:

- `our_095567`: `60ec5bd11eca0445c3b0b222a2715d3ffeee9594968cbce56061db16d87f5c62`
- `Step_V`: `ac6962c8f32f6d132af3787874f02ea7afdbfde9bde5317d663be6879b2092b3`
- `Step_W`: `abc5d4a49535086e7f1c6e9cce8b0763253db1f4d6a0778227c651bcdd59b125`
- `Step_X`: `5c7e5518ac4beae1680675c585f006a7956d220338abf72a6951ae83f81579b8`
- `Step_Y_095561`: `6a153c97a54fe0554504b74d8e79917dbe0834e0c058d387f3146a993c063290`
- `Step_Z_095562`: `ee61b50611326eb668033f088f8803e28541874c1cff2cf073334ea7d0f05a11`
- `Step_AA`: `51706855fa6de98ef88c1d02bebfc24ab340092d05eb513e227e048041c0346c`
- `areeb_095568`: `3b9fe6d72e3102101ab33d35a17cdb90b1e41de40c3792a7dc1a1cd619d1bc8c`
- `dennis_clipped`: `7de7fcc291f699bf84ea26ea08c54409fb62018a232382516efe6b2ffc86fd0c`
- `malyshev_v42`: `cde594c2098f4e076e755ea1227cb60e8ff05a69bd304234f7944263295c4102`
- `dennis_Nov23_npz`: `c226b49911d20295f5c46ac73a51946cb79ffb4222f691179f9012c3d10d7b05`

### NPZ V/Y/Z vs goodpjw CSV alignment

```json
{
  "Y": {
    "max_abs_npz_vs_csv": 0.0,
    "1mrho": 0.0
  },
  "Z": {
    "max_abs_npz_vs_csv": 0.0,
    "1mrho": 0.0
  },
  "V": {
    "max_abs_npz_vs_csv": 0.0,
    "1mrho": 0.0
  }
}
```

## 2. Exact Algebraic Reproduction

### 2.1 areeb 0.95568

```json
{
  "space": "logit(uniform_rank) with zone mask; residual=AA-quarter; clip p95 abs outside zones; weight 0.70; then uniform ranks",
  "NOT_Y_minus_Z": true,
  "cap_p95": 0.02086590712294964,
  "weight": 0.7,
  "quarter_sha256": "60ec5bd11eca0445c3b0b222a2715d3ffeee9594968cbce56061db16d87f5c62",
  "final_sha256": "3b9fe6d72e3102101ab33d35a17cdb90b1e41de40c3792a7dc1a1cd619d1bc8c",
  "exec_sha256": "3b9fe6d72e3102101ab33d35a17cdb90b1e41de40c3792a7dc1a1cd619d1bc8c",
  "max_abs_diff": 0.0,
  "max_rank_diff": 0.0,
  "one_minus_spearman": 0.0,
  "sha_prefix_match_notebook": true,
  "zones_n": 16730,
  "rebuilt_path": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/v41_nov23_frontier/recon/areeb_095568_rebuilt.csv"
}
```

**Verdict:** `1-ρ=0.000e+00`, max_abs=0.000e+00, sha_prefix_ok=True.

### 2.2 dennis (current executed) + historical Y−Z probe

```json
{
  "dennis_current": {
    "parent_found": true,
    "parent_path": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/v41_nov23_frontier/goodpjw_95569/submission.csv",
    "alt_path": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/v41_nov23_frontier/areeb_95568/submission.csv",
    "space": "logit(uniform_rank); residual=alt-parent; clip quantile abs 0.985; weight 0.12; then uniform ranks",
    "NOT_Y_minus_Z_026": true,
    "limit_p985": 0.10542461491305088,
    "weight": 0.12,
    "max_abs_diff": 0.0,
    "max_rank_diff": 0.0,
    "one_minus_spearman": 0.0,
    "rebuilt_sha256": "7de7fcc291f699bf84ea26ea08c54409fb62018a232382516efe6b2ffc86fd0c",
    "exec_sha256": "7de7fcc291f699bf84ea26ea08c54409fb62018a232382516efe6b2ffc86fd0c"
  },
  "historical_YZ_0p26_p99": {
    "documented_formula": "Z + 0.26 * clip(logit_rank(Y)-logit_rank(Z), \u00b1p99_abs) \u2192 uniform ranks",
    "p99": 0.06364578033863297,
    "one_minus_spearman_vs_dennis_Nov23": 0.0,
    "max_abs": 0.0,
    "space_probe": {
      "raw_file_values__on_Z_raw": {
        "p99": 0.006733692463598062,
        "one_minus_spearman_vs_dennis_Nov23": 2.4382364571806647e-08,
        "max_abs_vs_dennis_rank": 0.0023801269401034775
      },
      "raw_file_values__on_Z_logit": {
        "p99": 0.006733692463598062,
        "one_minus_spearman_vs_dennis_Nov23": 7.759519575767371e-07,
        "max_abs_vs_dennis_rank": 0.0036535281881699966
      },
      "raw_file_values__on_Y_raw": {
        "p99": 0.006733692463598062,
        "one_minus_spearman_vs_dennis_Nov23": 2.9474485926495042e-05,
        "max_abs_vs_dennis_rank": 0.028994879726918776
      },
      "uniform_rank__on_Z_raw": {
        "p99": 0.006733692463598062,
        "one_minus_spearman_vs_dennis_Nov23": 2.4382364571806647e-08,
        "max_abs_vs_dennis_rank": 0.0023801269401034775
      },
      "uniform_rank__on_Z_logit": {
        "p99": 0.006733692463598062,
        "one_minus_spearman_vs_dennis_Nov23": 7.759519575767371e-07,
        "max_abs_vs_dennis_rank": 0.0036535281881699966
      },
      "uniform_rank__on_Y_raw": {
        "p99": 0.006733692463598062,
        "one_minus_spearman_vs_dennis_Nov23": 2.9474485926495042e-05,
        "max_abs_vs_dennis_rank": 0.028994879726918776
      },
      "logit_rank__on_Z_raw": {
        "p99": 0.06364578033863297,
        "one_minus_spearman_vs_dennis_Nov23": 0.0,
        "max_abs_vs_dennis_rank": 0.0
      },
      "logit_rank__on_Z_logit": {
        "p99": 0.06364578033863297,
        "one_minus_spearman_vs_dennis_Nov23": 0.0,
        "max_abs_vs_dennis_rank": 0.0
      },
      "logit_rank__on_Y_raw": {
        "p99": 0.06364578033863297,
        "one_minus_spearman_vs_dennis_Nov23": 2.962118495142274e-05,
        "max_abs_vs_dennis_rank": 0.028491519547709218
      }
    },
    "dennis_Nov23_csv": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/v41_nov23_frontier/recon/dennis_Nov23_from_npz.csv",
    "dennis_Nov23_sha256": "c226b49911d20295f5c46ac73a51946cb79ffb4222f691179f9012c3d10d7b05"
  }
}
```

## 3. Pairwise `1 − ρ` Spearman (× 1e−4)

|  | our_095567 | Step_V | Step_W | Step_X | Step_Y_095561 | Step_Z_095562 | dennis_clipped | areeb_095568 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| our_095567 | 0.0000 | 4.2710 | 3.5935 | 3.1432 | 2.7689 | 2.1021 | 0.4238 | 0.0712 |
| Step_V | 4.2710 | 0.0000 | 0.0906 | 0.1698 | 0.4767 | 1.0109 | 6.4945 | 5.2887 |
| Step_W | 3.5935 | 0.0906 | 0.0000 | 0.0504 | 0.3624 | 0.6034 | 5.7837 | 4.5069 |
| Step_X | 3.1432 | 0.1698 | 0.0504 | 0.0000 | 0.1491 | 0.3654 | 5.2918 | 4.0319 |
| Step_Y_095561 | 2.7689 | 0.4767 | 0.3624 | 0.1491 | 0.0000 | 0.2966 | 4.8867 | 3.6544 |
| Step_Z_095562 | 2.1021 | 1.0109 | 0.6034 | 0.3654 | 0.2966 | 0.0000 | 4.0611 | 2.7904 |
| dennis_clipped | 0.4238 | 6.4945 | 5.7837 | 5.2918 | 4.8867 | 4.0611 | 0.0000 | 0.2270 |
| areeb_095568 | 0.0712 | 5.2887 | 4.5069 | 4.0319 | 3.6544 | 2.7904 | 0.2270 | 0.0000 |

## 4. Inventory ready for Steps 4–5

```json
{
  "busyaprime_52": {
    "path": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/v41_nov23_frontier/busyaprime_52",
    "files": [
      "manifest.csv",
      "oof.npy",
      "tabm_oof.npy",
      "tabm_test.npy",
      "test.npy"
    ]
  },
  "kabartay_d27_notebook": {
    "path": "/Volumes/pics/kaggle/cc/Predicting-Credit-Card-Fraud/external/v41_nov23_frontier/kabartay",
    "files": [
      "kernel-metadata.json",
      "s6e11-own-models-only-honest-5-fold-cv.ipynb"
    ],
    "note": "Notebook pulled for Step 4; D27 zone rules to be parsed then."
  },
  "goodpjw_step_AA_LB_note": "goodpjw board tip now publishes Step AA; areeb 0.95568 uses AA sha 51706855\u2026"
}
```

## 5. STOP

Step 1 complete. **Awaiting human audit.** Reply `PROCEED TO STEP 2` to continue.
