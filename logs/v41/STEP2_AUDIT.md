# V41 STEP2 AUDIT — AA/AB Frontier Geometry

**NO_SUBMIT=True** · Hard stop. Do not run Step 3 without `PROCEED TO STEP 3`.


## Executive findings (audit)

1. **Naming correction (critical):** `sha 51706855…` is **Step_AA = public LB 0.95567**, not 0.95569. The tip `submission.csv` (`sha 3e1dba5d…`) is **Step_AB = 0.95569**. Slot-1 PUB lock = **F1b** (`F1b_pub_Step_AB_095569.csv`).
2. **Zone mask already baked into AA and AB:** all 16,730 zone rows have uniform rank ≪ 0.5 in both (`AA_gt080=0`, `AB_gt080=0`). Therefore `F3_StepAA_with_hard_zones` ≡ `F1` (byte-identical after rank rewrite / same order), and `F3b` ≡ `F1b`. Hard-zone demotion is **not** a free +tick on the tip.
3. **areeb recipe exact:** `quarter + 0.70·clip_p95(AA−quarter)` outside + hard zones → `1−ρ = 0` vs pulled areeb (`0.95568`). AAhardZones vs areeb `1−ρ×1e4 = 0.0235` (= AA vs areeb): areeb’s +1 tick over AA comes from the **0.70·clip_p95 pull toward quarter on non-zones**, not from demoting more zone rows.
4. **dennis parent is Step_AB (tip), not AA:** `dennis = AB + 0.12·clip_p985(areeb−AB)`; `1−ρ(dennis,AB)×1e4 = 0.0040` (near-identity). Soft zone pull caps at `±0.12·p985`; does **not** hard-demote. No numeric LB in dennis title/markdown.
5. **Obsolete:** do **not** stage Z-base `Y−Z` / `0.26·clip99` blends (0.95562 shelf).
6. **OOF:** no Step_AA/AB OOF twins → 60k/240k stratified sims skipped.
7. **Kaggle board:** tip remains `goodpjw … lb-0-95569`; no public kernel `> 0.95569` on first scoreDescending page.

## 0. Critical naming correction

| File | SHA16 | goodpjw name | Documented public LB |
|---|---|---|---:|
| `submission_step_aa.csv` | `51706855…` | **Step AA** | **0.95567** |
| `submission.csv` | `3e1dba5d…` | **Step AB** | **0.95569** |
| `areeb` clipped | `3b9fe6d7…` | — | **0.95568** |
| `dennis` | `7de7fcc2…` | AB + 0.12·clip(areeb−AB) | **unclaimed** (near-AB) |

Slot-1 PUB lock for 0.95569 must be **Step AB**, not Step AA.

## 1. Step_AA / Step_AB lineage & dennis LB

```json
{
  "CRITICAL_NAMING_CORRECTION": {
    "user_label_Step_AA_as_095569": false,
    "fact": "goodpjw submission_step_aa.csv (sha 51706855\u2026) is Step AA, documented public LB 0.95567 (v41). The tip file submission.csv (sha 3e1dba5d\u2026) is Step AB, public LB 0.95569 (v43).",
    "Step_AA_formula": "Zr( MALV43 + 0.75 * clip_p99(MOVE43=MALV43-MALV42, outside zones) + 1.5 * EARLIER )",
    "Step_AB_formula": "Zr( SRIF_orthogonal_0.95568 - 0.4*RECENT - 0.15*MOVE6_0.95557 + 0.15*MSTEP )",
    "byte_repro_from_local_only": "NOT without pinned inputs: malyshevdanil_v42/v43 CSVs, EARLIER residual, sriram_orthogonal_0.95568, RECENT, MOVE6, MSTEP, Zr zone applicator. Executed CSVs are SHA-pinned; full local rebuild needs goodpjw pinned-input dataset.",
    "OOF_twin": "No dedicated Step_AA/AB OOF twin npy. Notebook reports RealMLP OOF (v15 0.950314, v28 0.950016), TabM OOF 0.937314, and in-sample replay OOF deltas for early steps B\u2013G; AA/AB are board-model test-side constructions with forecast points (AA forecast +1.05; AB +0.58)."
  },
  "dennis": {
    "notebook_title": "S6E11 Fraud signal rank fusion four refinements",
    "reported_numeric_LB_in_title": null,
    "markdown_claim": "Parent is goodpjw board V43 (0.95569). Blend 12% of scaled rank residual from areeb 0.95568. No numeric LB claimed for the dennis output itself.",
    "exact_formula_verified": "parent=Step_AB(submission.csv sha 3e1dba5d) + 0.12 * clip_p985(L(areeb)-L(parent)); NOT Step_AA-based. 1-\u03c1(dennis, Step_AB)=0.0040e-4 (near-identity).",
    "exec_sha": "7de7fcc291f699bf84ea26ea08c54409fb62018a232382516efe6b2ffc86fd0c"
  },
  "kaggle_scoreDescending_head": [
    "ref                                                                  title                                               author                       lastRunTime                 totalVotes  ",
    "-------------------------------------------------------------------  --------------------------------------------------  ---------------------------  --------------------------  ----------  ",
    "goodpjw2008/s6e11-board-model-realmlp-lb-0-95569                     S6E11 | Board Model + RealMLP | LB 0.95569          goodpjw2008                  2026-11-24 03:06:07.297000          68  ",
    "dennis6886/s6e11-fraud-signal-rank-fusion-four-refinements           S6E11 Fraud signal rank fusion four refinements     \u5976\u9f99                           2026-11-24 06:19:55.257000          15  ",
    "areebbb/s6e11-which-signals-rank-fraud-lb-0-95568                    S6E11 | Which Signals Rank Fraud? | LB 0.95568      Areeb Ali                    2026-11-24 00:45:23.713000          30  ",
    "malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion                  Fraud Signal Lab | GPT-2 and GLM Fusion             Malyshev Danil               2026-11-23 09:19:32.690000          60  ",
    "daoviet/s6e11-on-the-journey-to-your-first-top-100                   S6E11 | On the journey to your first top 100 \ud83d\ude80      daoviet                      2026-11-23 14:27:05.200000          25  ",
    "mzishran/credit-card-fraud-rank-diversity-and-blend-search           Credit Card Fraud |Rank Diversity and Blend Search  Mirza Ishran                 2026-11-23 15:25:19.883000          21  ",
    "rishabhbafnaiiitd/credit-card-fraud-beginner-friendly-sota-guide     Credit Card Fraud: Beginner-Friendly SOTA Guide     Rishabh Bafna                2026-11-23 04:20:10.633000          13  ",
    "davutkursun/s6e11-what-a-board-model-learns                          S6E11 | What a Board Model Learns                   DavutKursun                  2026-11-22 16:53:38.180000           3  ",
    "asteriospanoras/s6e11-lb-0-95552-will-the-top-survive-private        S6E11 | LB 0.95552 | Will the Top Survive Private?  Asterios Panoras             2026-11-21 20:53:34.267000          58  ",
    "dennis6886/s6e11-which-signals-rank-fraud-v3-reproduction            S6E11 Which Signals Rank Fraud V3 reproduction      \u5976\u9f99                           2026-11-21 14:08:24.377000           5  ",
    "dennis6886/s6e11-catboost-within-fraud-risk-bands                    S6E11 CatBoost within fraud risk bands              \u5976\u9f99                           2026-11-21 14:15:34.533000           7  ",
    "bhuvannr2006/s6e11-40-scored-files-one-forecast-lb-0-95534           S6E11 | 40 Scored Files, One Forecast | LB 0.95534  BhuvanNR                     2026-11-21 02:19:22.783000          14  ",
    "alexchilton/credit-card-fraud-lb-0-9550-shallow-tree-stack           Credit Card Fraud | LB 0.9550 | Shallow-Tree Stack  alex chilton                 2026-11-23 08:05:58.627000          73  ",
    "rishabhbafnaiiitd/credit-card-fraud-digit-forensics-frontier-fusion  Credit Card Fraud Digit Forensics Frontier Fusion   Rishabh Bafna                2026-11-19 01:33:34.973000           3  ",
    "busyaprime/s6e11-124-voices-the-top-rebuilt-lb-0-95465               S6E11 | 124 Voices + the Top Rebuilt | LB 0.95465   Akmal Xodarev (Busya PRIME)  2026-11-18 00:11:40.867000          11  ",
    "goodpjw2008/s6e11-lm-ladder-ladder-meets-stack-lb-0-95432            S6E11 LM Ladder | Ladder Meets Stack | LB 0.95432   goodpjw2008                  2026-11-17 06:40:30.427000          61  ",
    "areebbb/s6e11-split-then-push-lb-0-95431                             S6E11 | Split, Then Push | LB 0.95431               Areeb Ali                    2026-11-17 04:53:50.747000          65  ",
    "saamhm/eda-blend-lb-0-95239-credit-card-fraud                        EDA | blend | LB-0.95239 | Credit Card Fraud        samsul hoque mondal          2026-11-17 06:00:11.917000          64  ",
    "bhuvannr2006/s6e11-lm-ladder-forecast-the-board-lb-0-95417           S6E11 LM Ladder | Forecast the Board | LB 0.95417   BhuvanNR                     2026-11-15 13:02:08.287000           8  ",
    "muhammadshoaibaltaf/s6e11-0-95521-learning-to-rerank-step-l          S6E11 | 0.95521 | Learning to Rerank Step L         Muhammad Shoaib Altaf        2026-11-20 17:58:57.890000          12  "
  ],
  "kernels_mentioning_ge_095569": [
    "goodpjw2008/s6e11-board-model-realmlp-lb-0-95569                     S6E11 | Board Model + RealMLP | LB 0.95569          goodpjw2008                  2026-11-24 03:06:07.297000          68  "
  ]
}
```

### Kernels at/above 0.95569 (from scoreDescending list)

Top of list is `goodpjw2008/...-lb-0-95569`. dennis title has **no numeric LB**. areeb is 0.95568. No higher than 0.95569 appeared in the first page.

## 2. Pairwise `1−ρ` (×1e−4)

|  | quarter_our_095567 | Step_Z | malyshev_v42 | Step_AA | areeb_095568 | dennis_clipped | Step_AB_tip_095569 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| quarter_our_095567 | 0.0000 | 2.1021 | 0.6408 | 0.1628 | 0.0712 | 0.4238 | 0.5047 |
| Step_Z | 2.1021 | 0.0000 | 0.4249 | 3.2089 | 2.7904 | 4.0611 | 4.2631 |
| malyshev_v42 | 0.6408 | 0.4249 | 0.0000 | 1.3279 | 1.0549 | 1.9165 | 2.0647 |
| Step_AA | 0.1628 | 3.2089 | 1.3279 | 0.0000 | 0.0235 | 0.1668 | 0.2199 |
| areeb_095568 | 0.0712 | 2.7904 | 1.0549 | 0.0235 | 0.0000 | 0.2270 | 0.2909 |
| dennis_clipped | 0.4238 | 4.0611 | 1.9165 | 0.1668 | 0.2270 | 0.0000 | 0.0040 |
| Step_AB_tip_095569 | 0.5047 | 4.2631 | 2.0647 | 0.2199 | 0.2909 | 0.0040 | 0.0000 |

## 3. Zone mask (n=16,730) vs Step_AA / AB / areeb / dennis

```json
{
  "Step_AA_on_zones": {
    "n": 16730,
    "min": 1.6667555602965492e-06,
    "p50": 0.027884820523761268,
    "p90": 0.050191343538322046,
    "p95": 0.05297965891514214,
    "p99": 0.05521031121659821,
    "max": 0.05576797429196224,
    "mean": 0.027884820523761265,
    "std": 0.016099308606938204
  },
  "Step_AB_on_zones": {
    "n": 16730,
    "min": 1.6667555602965492e-06,
    "p50": 0.027884820523761268,
    "p90": 0.050191343538322046,
    "p95": 0.05297965891514214,
    "p99": 0.05521031121659821,
    "max": 0.05576797429196224,
    "mean": 0.027884820523761265,
    "std": 0.016099308606938204
  },
  "quarter_on_zones": {
    "n": 16730,
    "min": 1.6667555602965492e-06,
    "p50": 0.027884820523761268,
    "p90": 0.050191343538322046,
    "p95": 0.05297965891514214,
    "p99": 0.05521031121659821,
    "max": 0.05576797429196224,
    "mean": 0.027884820523761265,
    "std": 0.016099308606938204
  },
  "areeb_on_zones": {
    "n": 16730,
    "min": 1.6667555602965492e-06,
    "p50": 0.027884820523761268,
    "p90": 0.050191343538322046,
    "p95": 0.05297965891514214,
    "p99": 0.05521031121659821,
    "max": 0.05576797429196224,
    "mean": 0.027884820523761265,
    "std": 0.016099308606938204
  },
  "dennis_on_zones": {
    "n": 16730,
    "min": 1.6667555602965492e-06,
    "p50": 0.027884820523761268,
    "p90": 0.050191343538322046,
    "p95": 0.05297965891514214,
    "p99": 0.05521031121659821,
    "max": 0.05576797429196224,
    "mean": 0.027884820523761265,
    "std": 0.016099308606938204
  },
  "n_zones": 16730,
  "n_AA_zone_rows_above_050": 0,
  "n_AA_zone_rows_above_080": 0,
  "n_AB_zone_rows_above_050": 0,
  "n_AB_zone_rows_above_080": 0,
  "definition": "risk<13.1 OR (vel<=4.1 & risk<=20) OR (clean flags & ((vel<=4.1 & risk<=33.8)|(vel<=6 & risk<=25))); same as areeb notebook (16,730 rows)."
}
```

### dennis soft-zone mechanics (parent = Step_AB)

```json
{
  "parent_is_Step_AB_not_AA": true,
  "clip_p985_limit": 0.10542461491305088,
  "mean_soft_zone_delta_logit": -0.00026410849570451106,
  "expected_cap_magnitude": 0.012650953789566106,
  "zone_gap_areeb_minus_AB_stats": {
    "n": 16730,
    "min": -0.3585976250434828,
    "p50": -0.006521273601177047,
    "p90": 0.11888582451905308,
    "p95": 0.1491236128591817,
    "p99": 0.24536111368398264,
    "max": 0.5194302053827107,
    "mean": 5.308896710700091e-20,
    "std": 0.09139623797061157
  },
  "nonzone_gap_areeb_minus_AB_stats": {
    "n": 283254,
    "min": -0.5810732214064398,
    "p50": -3.1615146522223636e-05,
    "p90": 0.01577265261152956,
    "p95": 0.0217673791364277,
    "p99": 0.04165559575759556,
    "max": 0.8251196550349018,
    "mean": -1.2542501354969395e-20,
    "std": 0.015769720775789656
  },
  "note": "On zones areeb has hard demotion (~logit of near-zero ranks) so L(areeb)-L(AB) is large negative; dennis only applies \u00b1p985 cap * 0.12, i.e. soft pull, not -100."
}
```

### Residual `L(Step_AA)−L(quarter)` outside vs inside zones

```json
{
  "counterfactual": {
    "p95_of_abs_AA_minus_quarter_outside": 0.02086590712294964,
    "frac_outside_clipped_by_p95": 0.050001059120083034,
    "residual_outside_quantiles": {
      "n": 283254,
      "min": -0.164032983372163,
      "p50": -4.339512490125763e-05,
      "p90": 0.012146573508050696,
      "p95": 0.016441001544161105,
      "p99": 0.028092817121198793,
      "max": 0.22570533772388757,
      "mean": 0.0,
      "std": 0.010598848556231314
    },
    "residual_outside_by_AA_regime": {
      "bulk_neg_qlt080": {
        "n": 223257,
        "min": -0.05826364697957498,
        "p50": -8.239749712857236e-05,
        "p90": 0.011959109118585375,
        "p95": 0.01623475073387368,
        "p99": 0.027829473405051545,
        "max": 0.12700506612671703,
        "mean": -7.419966287814334e-06,
        "std": 0.010350270707121828
      },
      "twilight_080_985": {
        "n": 55497,
        "min": -0.05265682713428488,
        "p50": 5.6189035485987304e-05,
        "p90": 0.011966167017713405,
        "p95": 0.015683423376671744,
        "p99": 0.023554727293450314,
        "max": 0.05249609768466934,
        "mean": 2.0809383136261883e-05,
        "std": 0.009717276315293772
      },
      "fraud_tail_gt985": {
        "n": 4500,
        "min": -0.164032983372163,
        "p50": 0.0006940504575982942,
        "p90": 0.02826088293191819,
        "p95": 0.03699906852373497,
        "p99": 0.05384921424679703,
        "max": 0.22570533772388757,
        "mean": 0.00011148912835676446,
        "std": 0.024321843898987613
      }
    },
    "residual_inside_zones": {
      "n": 16730,
      "min": -0.21379334691894503,
      "p50": -0.0008153000450032977,
      "p90": 0.05236392832229764,
      "p95": 0.06428740009890174,
      "p99": 0.08542842034596565,
      "max": 0.2554486883830318,
      "mean": 5.308896710700091e-20,
      "std": 0.041646512953736864
    },
    "1mrho_areeb_like_vs_areeb_e4": 0.0,
    "1mrho_full1p0_zoneHard_vs_areeb_e4": 0.019133691697703625,
    "1mrho_full1p0_zoneHard_vs_AA_e4": 0.004322339540729914,
    "1mrho_AAhardZones_vs_areeb_e4": 0.023456031240653985,
    "1mrho_AAhardZones_vs_AA_e4": 0.0,
    "tick_attribution_hypothesis": "areeb \u2260 unclipped AA: areeb applies 0.70*clip_p95(AA\u2212quarter) on non-zones + hard zone demotion. Counterfactual AAhardZones is very close to areeb (small 1-\u03c1), while full 1.00*unclipped residual moves away from areeb toward raw AA. Public AA listed 0.95567; areeb clip gained +1 tick to 0.95568; goodpjw tip Step AB (not AA) is the 0.95569 file."
  },
  "inside_zone_rank_diffs": {
    "U_AA_minus_U_quarter": {
      "n": 16730,
      "min": -0.002986825964051417,
      "p50": -6.667022241186454e-06,
      "p90": 0.001306736359272493,
      "p95": 0.001955270947783878,
      "p99": 0.00355868979678915,
      "max": 0.006477012107312392,
      "mean": -8.295151110468892e-22,
      "std": 0.0011365404319033036
    },
    "U_areeb_minus_U_AA": {
      "n": 16730,
      "min": -0.006477012107312392,
      "p50": 6.667022241186454e-06,
      "p90": 0.0014967464931463018,
      "p95": 0.0018500986719291734,
      "p99": 0.00232345725105339,
      "max": 0.002986825964051417,
      "mean": 8.295151110468892e-22,
      "std": 0.0011365404319033036
    },
    "U_dennis_minus_U_AB": {
      "n": 16730,
      "min": -0.0006867032908421736,
      "p50": -3.3335111205914925e-06,
      "p90": 0.00032001706757693593,
      "p95": 0.000383353778868209,
      "p99": 0.0004800256013654039,
      "max": 0.0005833644461037887,
      "mean": 4.147575555234446e-22,
      "std": 0.00022537793029255547
    },
    "U_AB_minus_U_AA": {
      "n": 16730,
      "min": -0.021824497306523014,
      "p50": 7.000373353245604e-05,
      "p90": 0.001443410315216814,
      "p95": 0.0019201024054616225,
      "p99": 0.003300076004053529,
      "max": 0.010700570697103846,
      "mean": 1.6590302220937785e-21,
      "std": 0.0013961962123477355
    }
  }
}
```

## 4. Staged candidates (`out/v41_staged/`, NO_SUBMIT)

```json
{
  "F1_pub_Step_AA_sha51706855.csv": {
    "sha": "51706855fa6de98ef88c1d02bebfc24ab340092d05eb513e227e048041c0346c",
    "expected": "51706855fa6de98ef88c1d02bebfc24ab340092d05eb513e227e048041c0346c",
    "claimed_LB_from_goodpjw_docs": 0.95567,
    "NOTE": "NOT the 0.95569 tip; that is Step_AB/submission.csv"
  },
  "F1b_pub_Step_AB_095569.csv": {
    "sha": "3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038",
    "expected": "3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038",
    "claimed_LB": 0.95569,
    "NOTE": "True Slot-1 PUB lock candidate (goodpjw tip)."
  },
  "F2_pub_dennis_clipped.csv": {
    "sha": "7de7fcc291f699bf84ea26ea08c54409fb62018a232382516efe6b2ffc86fd0c",
    "expected": "7de7fcc291f699bf84ea26ea08c54409fb62018a232382516efe6b2ffc86fd0c",
    "1mrho_vs_AB_e4": 0.004038074070678022,
    "NOTE": "Near-duplicate of Step_AB (1-\u03c1\u22480.004e-4); weak as a separate slot."
  },
  "F3_StepAA_with_hard_zones.csv": {
    "sha": "51706855fa6de98ef88c1d02bebfc24ab340092d05eb513e227e048041c0346c",
    "1mrho_vs_F1_AA_e4": 0.0,
    "1mrho_vs_F2_dennis_e4": 0.16683004261608403,
    "1mrho_vs_areeb_e4": 0.023456031240653985,
    "1mrho_vs_AB_e4": 0.21991635567597534
  },
  "F3b_StepAB_with_hard_zones.csv": {
    "sha": "3e1dba5d31e01b6f36554459cfe092cf4c760b5d7d228f23bd4e2777dfc86038",
    "1mrho_vs_AB_e4": 0.0,
    "1mrho_vs_areeb_e4": 0.290898400540085,
    "1mrho_vs_dennis_e4": 0.004038074070678022
  },
  "F4_StepAB_plus_clip_quarter_a0p15.csv": {
    "sha": "b9465d0917dcf4076433a590add9ba913dadf76199f84b1f99802b1ed3ab00ec",
    "alpha": 0.15,
    "cap_p985": 0.05941847194926209,
    "zone_demote": false,
    "1mrho_vs_AB_e4": 0.010518409954141106,
    "1mrho_vs_AA_e4": 0.15188417542222332,
    "1mrho_vs_areeb_e4": 0.2016920473502104,
    "1mrho_vs_dennis_e4": 0.0026333170555226104
  },
  "F4z_StepAB_plus_clip_quarter_a0p15_zones.csv": {
    "sha": "b9465d0917dcf4076433a590add9ba913dadf76199f84b1f99802b1ed3ab00ec",
    "alpha": 0.15,
    "cap_p985": 0.05941847194926209,
    "zone_demote": true,
    "1mrho_vs_AB_e4": 0.010518409954141106,
    "1mrho_vs_AA_e4": 0.15188417542222332,
    "1mrho_vs_areeb_e4": 0.2016920473502104,
    "1mrho_vs_dennis_e4": 0.0026333170555226104
  },
  "F5_StepAB_plus_clip_quarter_a0p30.csv": {
    "sha": "7b2ee3ae6f735be43413ea9edc594ad2fcc875615847b869e085a2e8b447b735",
    "alpha": 0.3,
    "cap_p985": 0.05941847194926209,
    "zone_demote": false,
    "1mrho_vs_AB_e4": 0.042082006753707546,
    "1mrho_vs_AA_e4": 0.10469692738990233,
    "1mrho_vs_areeb_e4": 0.13328207545737492,
    "1mrho_vs_dennis_e4": 0.02228630478828819
  },
  "F5z_StepAB_plus_clip_quarter_a0p30_zones.csv": {
    "sha": "7b2ee3ae6f735be43413ea9edc594ad2fcc875615847b869e085a2e8b447b735",
    "alpha": 0.3,
    "cap_p985": 0.05941847194926209,
    "zone_demote": true,
    "1mrho_vs_AB_e4": 0.042082006753707546,
    "1mrho_vs_AA_e4": 0.10469692738990233,
    "1mrho_vs_areeb_e4": 0.13328207545737492,
    "1mrho_vs_dennis_e4": 0.02228630478828819
  },
  "F4v_StepAB_plus_clip_v42_a0p15.csv": {
    "sha": "8a93c965a99427cc76728dfcd73a567a474940d29e96932e669ae3bf65d2f62e",
    "alpha": 0.15,
    "cap_p985": 0.11151614091274586,
    "zone_demote": false,
    "1mrho_vs_AB_e4": 0.04343210153523458,
    "1mrho_vs_AA_e4": 0.12961502869446306,
    "1mrho_vs_areeb_e4": 0.1540993097004506,
    "1mrho_vs_dennis_e4": 0.025961168765853415
  },
  "F5v_StepAB_plus_clip_v42_a0p30.csv": {
    "sha": "c48315376c1b7923ebbc924cb009128029863e4ad16df07f9917684506308f82",
    "alpha": 0.3,
    "cap_p985": 0.11151614091274586,
    "zone_demote": false,
    "1mrho_vs_AB_e4": 0.17376535787483327,
    "1mrho_vs_AA_e4": 0.12600859536804343,
    "1mrho_vs_areeb_e4": 0.10394186165663122,
    "1mrho_vs_dennis_e4": 0.13478609750650072
  },
  "F4aa_StepAA_plus_clip_quarter_a0p15.csv": {
    "sha": "d933a19ff425efbf819c822ee8c18e24eb99635009195dd55c14ca8d429f5fb2",
    "alpha": 0.15,
    "cap_p985": 0.02983064596541737,
    "zone_demote": false,
    "1mrho_vs_AB_e4": 0.24012945619156234,
    "1mrho_vs_AA_e4": 0.0034836653028946785,
    "1mrho_vs_areeb_e4": 0.011477239868495914,
    "1mrho_vs_dennis_e4": 0.18320510736247186
  },
  "F5aa_StepAA_plus_clip_quarter_a0p30.csv": {
    "sha": "5c31f75bcb359a08e92cbe679e542e9706ebd40ed1ade5f016e540ef2d4e1d4c",
    "alpha": 0.3,
    "cap_p985": 0.02983064596541737,
    "zone_demote": false,
    "1mrho_vs_AB_e4": 0.2673746830528234,
    "1mrho_vs_AA_e4": 0.01394258974274365,
    "1mrho_vs_areeb_e4": 0.006378844144006024,
    "1mrho_vs_dennis_e4": 0.20659444438964059
  }
}
```

### Primary staged pairwise `1−ρ` (×1e−4)

|  | F1_pub_Step_AA_sha51706855.csv | F1b_pub_Step_AB_095569.csv | F2_pub_dennis_clipped.csv | F3_StepAA_with_hard_zones.csv | F3b_StepAB_with_hard_zones.csv | F4_StepAB_plus_clip_quarter_a0p15.csv | F5_StepAB_plus_clip_quarter_a0p30.csv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1_pub_Step_AA_sha51706855.csv | 0.0000 | 0.2199 | 0.1668 | 0.0000 | 0.2199 | 0.1519 | 0.1047 |
| F1b_pub_Step_AB_095569.csv | 0.2199 | 0.0000 | 0.0040 | 0.2199 | 0.0000 | 0.0105 | 0.0421 |
| F2_pub_dennis_clipped.csv | 0.1668 | 0.0040 | 0.0000 | 0.1668 | 0.0040 | 0.0026 | 0.0223 |
| F3_StepAA_with_hard_zones.csv | 0.0000 | 0.2199 | 0.1668 | 0.0000 | 0.2199 | 0.1519 | 0.1047 |
| F3b_StepAB_with_hard_zones.csv | 0.2199 | 0.0000 | 0.0040 | 0.2199 | 0.0000 | 0.0105 | 0.0421 |
| F4_StepAB_plus_clip_quarter_a0p15.csv | 0.1519 | 0.0105 | 0.0026 | 0.1519 | 0.0105 | 0.0000 | 0.0106 |
| F5_StepAB_plus_clip_quarter_a0p30.csv | 0.1047 | 0.0421 | 0.0223 | 0.1047 | 0.0421 | 0.0106 | 0.0000 |

## 5. OOF 60k/240k check

```json
{
  "available": false,
  "reason": "Step_AA/AB have no matched OOF twins in harvest; skipping fabricated 60k/240k OOF sims.",
  "proxy": "Early-step in-sample replay OOF exists only for B\u2013G in goodpjw notebook; not transferable to AA/AB."
}
```

## 6. STOP

Step 2 complete. **Awaiting human audit.** Reply `PROCEED TO STEP 3` to continue.

## 7. Filename aliases (user Step-2 naming)

| User-requested name | Points to | Note |
|---|---|---|
| `F1_pub_Step_AA_095569.csv` | `F1_pub_Step_AA_sha51706855.csv` | **Misnomer kept as alias** — AA is 0.95567; tip is F1b |
| `F4_StepAA_plus_clipped_quarter_resid_a0p15.csv` | `F4aa_…` | Pure AA + α·clip_p985(quarter−AA) |
| `F5_StepAA_plus_clipped_quarter_resid_a0p30.csv` | `F5aa_…` | α=0.30 |

## 8. `L(Step_AB)−L(quarter)` outside zones (tip geometry)

```json
{
  "residual_outside_quantiles": {
    "n": 283254,
    "min": -0.8392470901212858,
    "p50": -5.7056405679045685e-05,
    "p90": 0.02137550465921621,
    "p95": 0.029196467952526856,
    "p99": 0.05347077571880993,
    "max": 0.5810732214064398,
    "mean": -5.017000541987758e-20
  },
  "residual_outside_by_AB_regime": {
    "bulk_neg_qlt080": {
      "n": 223257,
      "min": -0.32084636478862105,
      "p50": -0.000134891788727165,
      "p90": 0.02093239324944458,
      "p95": 0.028541588237105987,
      "p99": 0.05371853963922009,
      "max": 0.302839571608175,
      "mean": -2.1081263609842898e-05
    },
    "twilight_080_985": {
      "n": 55497,
      "min": -0.8392470901212858,
      "p50": 0.00014961167420501909,
      "p90": 0.02124721268976755,
      "p95": 0.028186072972812136,
      "p99": 0.04297940594794449,
      "max": 0.24678170434796853,
      "mean": 3.535227563390657e-05
    },
    "fraud_tail_gt985": {
      "n": 4500,
      "min": -0.26829732587370003,
      "p50": 0.002496715334800559,
      "p90": 0.051968197563650606,
      "p95": 0.06998182720397966,
      "p99": 0.10162265440025307,
      "max": 0.5810732214064398,
      "mean": 0.0006099098730861757
    }
  },
  "residual_inside_zones": {
    "n": 16730,
    "min": -0.5194302053827107,
    "p50": 0.006521273601177047,
    "p90": 0.1114443543438532,
    "p95": 0.13886609461760985,
    "p99": 0.19367544727015337,
    "max": 0.3585976250434828,
    "mean": -5.308896710700091e-20
  },
  "zone_U_stats": {
    "AA": {
      "n": 16730,
      "min": 1.6667555602965492e-06,
      "p50": 0.027884820523761268,
      "p90": 0.050191343538322046,
      "p95": 0.05297965891514214,
      "p99": 0.05521031121659821,
      "max": 0.05576797429196224,
      "mean": 0.027884820523761265
    },
    "AB": {
      "n": 16730,
      "min": 1.6667555602965492e-06,
      "p50": 0.027884820523761268,
      "p90": 0.050191343538322046,
      "p95": 0.05297965891514214,
      "p99": 0.05521031121659821,
      "max": 0.05576797429196224,
      "mean": 0.027884820523761265
    },
    "quarter": {
      "n": 16730,
      "min": 1.6667555602965492e-06,
      "p50": 0.027884820523761268,
      "p90": 0.050191343538322046,
      "p95": 0.05297965891514214,
      "p99": 0.05521031121659821,
      "max": 0.05576797429196224,
      "mean": 0.027884820523761265
    },
    "areeb": {
      "n": 16730,
      "min": 1.6667555602965492e-06,
      "p50": 0.027884820523761268,
      "p90": 0.050191343538322046,
      "p95": 0.05297965891514214,
      "p99": 0.05521031121659821,
      "max": 0.05576797429196224,
      "mean": 0.027884820523761265
    },
    "dennis": {
      "n": 16730,
      "min": 1.6667555602965492e-06,
      "p50": 0.027884820523761268,
      "p90": 0.050191343538322046,
      "p95": 0.05297965891514214,
      "p99": 0.05521031121659821,
      "max": 0.05576797429196224,
      "mean": 0.027884820523761265
    }
  },
  "n_zone_above": {
    "AA_gt050": 0,
    "AA_gt080": 0,
    "AB_gt050": 0,
    "AB_gt080": 0,
    "quarter_gt050": 0,
    "quarter_gt080": 0
  },
  "F3_identity": {
    "F3_AA_hard_zones_identical_to_AA": true,
    "F3b_AB_hard_zones_identical_to_AB": true,
    "reason": "Both Step_AA and Step_AB already place all 16,730 zone rows at bottom (max zone U << 0.5); extra -100 is a no-op on rank order."
  }
}
```

## STOP

**NO_SUBMIT=True.** Step 2 artifacts ready for human audit. Do **not** run Step 3 until `PROCEED TO STEP 3`.
