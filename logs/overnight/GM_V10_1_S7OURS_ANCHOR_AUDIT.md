# GM v10.1 — Anchor on `z_S7_ours` Forensic Audit (`NO_SUBMIT=True`)

Generated after v10 live probes both scored **0.95352** (−2 ticks vs ATH 0.95354).

## Executive verdict

1. Our v10 s354_pub reconstruction MATCHES the official areebb/goodpjw 0.95354 recipe (1-ρ ≈ 0 vs official submission). It does NOT collapse to s352 (1-ρ(s354,s352)≈3.09e-5). Slot1 LB=0.95352 is therefore NOT a reduction bug — it is a crest/noise outcome from sitting on s354_pub (1-ρ=0.12e-4) while being 4.02e-4 away from proven z_S7_ours.
2. GM_V10_LIVE reported OOF AUC 0.952498 from a wide_lm-based s354_o PROXY, NOT from the calibrated evaluator z_ref_old + δ1_v4 (=0.9526953). Apples-to-oranges — that number never claimed to be the S7 OOF evaluator.
3. v10 probes left the 0.10–0.15e-4 S7 trust region by 30–40×, anchoring on s354_pub instead. Public LB then behaved like the s352/s354 crest noise band (0.95352).

## Step 1 — Verbatim notebook recipe vs our reconstr

### G26 / input shapes

| asset | shape | SHA matches notebook |
|---|---|---|
| A | `[299984]` | True |
| F | `[299984]` | True |
| LM | `[299984]` | True |
| L7F | `[299984]` | True |
| GPD | `[299984]` | True |
| MLP | `[299984]` | True |
| GB | `[299984]` | True |
| MS | `[299984]` | True |
| G26 | `[299984]` | True |

**G26 reduction:** NONE — file is already (299984,) log-odds; notebook does logit_rank(np.load) only

### Exact published formula (areebb / goodpjw cell)

```python
s352 = A + 0.6*(A-F) + 0.09*(LM-F) + 0.09*(L7F-F) + 0.05*(GPD-F) - 0.05*(MLP-F) - 0.025*(GB-F)
score = logit_rank(s352) + 0.06*(G26 - MS)
submission = (rankdata(score)-0.5)/n
```

### Verbatim code cells — `areebbb`

<details><summary>cell 0 (markdown)</summary>

```python
# S6E11 | Three Ladders, Two Seeds: LB 0.95354

**Our earlier versions** (all in the version history): v19 0.95348, v18 0.95346, v17 0.95347, v16 0.95343, v15 0.95333, v13 0.95330, v12 0.95300, v11 0.95291, v10 0.95288, v9 0.95239.

This version rebuilds the current top public recipe exactly: it moves 0.06 of the weight of a public target-first GPT-2 onto a second seed of the same model, on a ladder file that keeps our half tree move (v17) and our 0.09 language-model weight (v19). Its public score is shown in the box above. Read **How to read the gaps** before you use it as a final pick. Full sources and credit are at the end.

## Our ladder

| Version | What changed | Public LB |
|---|---|---|
| v12 | wide LM stack + shallow-tree fusion, CV-backed | 0.95300 |
| v15 | stepped away from the MLP and GB bags | 0.95333 |
| v16 | + 0.12 of a public target-first LM | 0.95343 |
| v17 | current LM + half tree move | 0.95347 |
| v18 | + 5% TabPFN-3.5 | 0.95346 |
| v19 | LM step 0.12 to 0.09 | 0.95348 |
| **v20** | **two GPT-2 seeds** | **0.95354** |

## What it does

1. Loads nine public files, each found and checked by its SHA-256:
   - `A`: the Strike 10 file (0.95340).
   - `F`: the Strike 9 file (0.95330), the common base of every step.
   - `LM`: a public target-first LM, a frozen copy of its current test predictions.
   - `L7F`: a compact target-first LM (seed 7), five fold models averaged.
   - `GPD`: GP + DVAE test predictions.
   - `MLP` and `GB`: two bags from a public OOF bank.
   - `MS`: the public target-first GPT-2 that sits inside the stacks every ladder file is built on.
   - `G26`: a second seed of that GPT-2 (seed 2026, fold-0 model).
2. Turns each into logit ranks, `logit((rank - 0.5) / n)`. AUC only sees order, so this puts all nine on one scale.
3. Builds the ladder file, ranks it again, and adds the two-seed rung:

```
s352  = A + 0.6 (A - F) + 0.09 (LM - F) + 0.09 (L7F - F) + 0.05 (GPD - F) - 0.05 (MLP - F) - 0.025 (GB - F)
score = L(s352) + 0.06 (G26 - MS)
```

4. Ranks the result, writes `submission.csv` and checks its SHA-256.

## Why two seeds

- Every ladder file already holds `MS` through its stacks, and adding more of it loses points.
- `G26` is the same model trained with another seed: OOF AUC 0.95014 and rank correlation 0.989 with the public seed. Moving 0.06 of the weight from one seed to the other leans the file on the average of two seeds, which cancels part of one seed's noise.
- It is the first rung on the ladder that gains on both boards: +2 points public, and +1.9e-5 on the nested OOF of the wide LM stack, in all five folds.

## What our own probes found

| Probe on our v17 (0.95347) | Public LB |
|---|---|
| **LM step 0.12 to 0.09 (v19)** | **0.95348** |
| more of both LMs (+0.04 LM, +0.03 seed-7 LM) | 0.95345 |
| less seed-7 LM (-0.02) | 0.95347 |
| + 10% / + 20% of a GPT-2 + GLM fusion file | 0.95347 / 0.95347 |
| + 5% / + 2% TabPFN-3.5 | 0.95346 / 0.95346 |
| 10% mix with the sovereign stack | 0.95346 |

The language model sits near 0.09, and new model families (TabPFN, GBDT files, other stacks) lose a point. The gains now come from new seeds and folds of the same target-first models, which is what the two-seed rung does.

## CV evidence

On the nested OOF of the wide LM stack (AUC 0.952565):

| Move | CV change |
|---|---|
| full tree move: 0.10 away from MLP + 0.05 away from GB (v15) | -7.4e-5 |
| + 5% public LM | -1.5e-5 |
| + 5% TabPFN-3.5 | +0.38e-5 |
| + 0.06 (G26 - MS), the two-seed rung | +1.9e-5 |

Every earlier ladder step lost on CV. The two-seed rung is the first one that holds up there.

## How to read the gaps

- The public LB is about 20% of the test set, roughly 60k rows. 0.00001 is noise and 0.00003 is barely more.
- Every ladder file, this one included, holds LB-gradient steps that can fade on the private 80%. **For your two final picks, pair a ladder file with a CV-backed one**, such as our v12 (0.95300).
- The next real gain is more seeds and folds of the target-first models, checked on OOF first.

## Built on public work

This version rebuilds goodpjw2008's [S6E11 LM Ladder | Two GPT-2 Seeds | LB 0.95354](https://www.kaggle.com/code/goodpjw2008/s6e11-lm-ladder-two-gpt-2-seeds-lb-0-95354) exactly, from the same public inputs, and its output matches his file byte for byte. The two-seed rung, the seed-2026 model, the five-fold seed-7 LM, the 0.95352 ladder file and the +1.9e-5 CV figure are his. His ladder file uses the tree step from our v17. The public ladder it climbs:

| Step | What changed | Public LB |
|---|---|---|
| Strike 9 (Moran Z) | public ladder file | 0.95330 |
| Strike 10 (amanatar) | climb from Strike 9 | 0.95340 |
| LM Ladder Part 1 (goodpjw2008) | + 0.6 (A - F) and a seed-7 compact LM | 0.95346 |
| 0.95352 file (goodpjw2008) | seed-7 LM on all five folds, both LMs at 0.09, 5% GP + DVAE (daoviet) | 0.95352 |
| Two GPT-2 Seeds (goodpjw2008) | 0.06 of masayakawamata's GPT-2 moved to a second seed | 0.95354 |

## Credits

- **goodpjw2008**:
  - [S6E11 LM Ladder | Two GPT-2 Seeds | LB 0.95354](https://www.kaggle.com/code/goodpjw2008/s6e11-lm-ladder-two-gpt-2-seeds-lb-0-95354), the recipe this version rebuilds.
  - The [seed-2026 target-first GPT-2](https://www.kaggle.com/datasets/goodpjw2008/s6e11-target-first-gpt2-seed-2026) (`G26`), the [compact target-first LM, seed 7](https://www.kaggle.com/datasets/goodpjw2008/s6e11-compact-lm-seed7) (`L7F`) and the [OOF Bank](https://www.kaggle.com/datasets/goodpjw2008/s6e11-oof-bank) (`MLP`, `GB`).
  - [s6e11-fraud-series-artifacts](https://www.kaggle.com/datasets/goodpjw2008/s6e11-fraud-series-artifacts) and the wide LM stack behind every CV check.
- **daoviet**, [S6E11 GP and DVAE Orthogonal Apex](https://www.kaggle.com/datasets/daoviet/s6e11-gp-dvae-orthogonal-apex) (`GPD`).
- **masayakawamata**, [S6E11 GPT-2 Target-First OOF and Test Predictions](https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof) (`MS`) and the [GPT-2 Target-First Classifier](https://www.kaggle.com/code/masayakawamata/s6e11-gpt-2-target-first-classifier-cv-0-9500).
- **megayak**, [S6E11 target-first LM predictions](https://www.kaggle.com/datasets/megayak/s6e11-target-first-lm) (`LM`), and the compact row format and training script behind the seed-7 LM.
- **amanatar**, [S6E11 | Podium Apex | Public LB 0.95340](https://www.kaggle.com/code/amanatar/s6e11-podium-apex-public-lb-0-95340) and the [Strike 10 dataset](https://www.kaggle.com/datasets/amanatar/s6e11-strike10-podium-apex-095340) (`A`), and the sovereign vault we probed.
- **Moran Z**, [S6E11 | Grandmaster Ladder | LB 0.95330](https://www.kaggle.com/code/moranzavdi/s6e11-grandmaster-ladder-lb-0-95330) and the [Strike9 dataset](https://www.kaggle.com/datasets/moranzavdi/s6e11-strike9-lb-095330) (`F`).
- **Chris Deotte**, [S6E10 1st place writeup](https://www.kaggle.com/competitions/playground-series-s6e10/writeups/1st-place-tfms-and-llms-with-tabular-data): the target-first language-model idea.
- **slimypunk03** and **philippsinger**: the TabPFN-3.5 runs we probed ([1](https://www.kaggle.com/code/slimypunk03/tabpfn-starter-ps-season-6-episode-11-0-94909), [2](https://www.kaggle.com/code/philippsinger/tabpfn-3-5-starter-playground-series-s6e11)).
- **Malyshev Danil**, [Fraud Signal Lab | GPT-2 and GLM Fusion](https://www.kaggle.com/code/malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion): the 65/35 backbone of our earlier versions and the fusion file we probed.
- The wide stack pools public predictions from Sadam Torres, alexchilton, Malyshev Danil, lazyyeou, megayak, Will, queyrusi, Vladimir Demidov, Nilanjan_Saha123, masayakawamata, Sameer Shaikh and goodpjw2008.

If this helps, please upvote the source notebooks too, starting with goodpjw2008, daoviet, masayakawamata and megayak.
```

</details>

<details><summary>cell 1 (code)</summary>

```python
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit

# Public inputs, found and checked by SHA-256 (sources in Credits)
EXPECTED = {
    'A': ('submission_095340.csv', '12b525b84f29e32d6511e9d6105f6dc5eac79ae41b10140eed3353ea57afc159'),  # Strike 10 file, 0.95340
    'F': ('submission_095330.csv', 'f4a7a4f8636c99fabd82945c7d81956c5559c609dcb55cde6d4e03f80b0ea105'),  # Strike 9 file, 0.95330: the common base
    'LM': ('megalm_test_sha3168.npy', '3168779be644a530c6bdc4c141cae168bfc2feb01ac21090c917cc978a7687d3'),  # public target-first LM, frozen copy
    'L7F': ('lm7_test_5fold.npy', '68639fec79f081e97d40185443fa3903c9db3e62698abb01dfe31dcee217462a'),  # compact target-first LM, seed 7, five folds
    'GPD': ('test_ours.npy', '96f3bdde816e604438e61eac9722f7a7402f1c39bb8efe8999793b23f91d02f2'),  # GP + DVAE
    'MLP': ('test_gbn_bag5_w512.npy', '035d6544d7c03108fc42ee296a8660a45825e3b1b19b7cabce1c3f42cac4e7cb'),  # OOF bank: MLP bag
    'GB': ('test_gb_bag5.npy', '7d54e5f17e0707d88fec1a2cccb05885978ae6cc2a595c5f4575c7b7d45a9bb8'),  # OOF bank: GB bag
    'MS': ('test.npy', '6c27aff7be744f7a0cc17809e0cd04fcdea0664234af88e3040a25235fb67906'),  # public target-first GPT-2 inside the stacks
    'G26': ('gpt2_s2026_test_f0.npy', '40e08058ed7632ee328c2fe7d97040b856b36d2636f32cc6d2781ec9db080695'),  # second seed of that GPT-2 (seed 2026)
}
OUT_SHA = '1e2e72fa26309ce630a19e0df780ae63d525901f33dc23f9058581ced5e9d6e0'  # the public 0.95354 file
INPUT = Path('/kaggle/input')
OUT = Path('/kaggle/working/submission.csv')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def locate(name, digest):
    hits = [p for p in sorted(INPUT.rglob(name)) if sha256(p) == digest]
    assert hits, (name, digest)
    return hits[0]

def logit_rank(v):
    return logit((rankdata(v) - 0.5) / len(v))

paths = {key: locate(name, digest) for key, (name, digest) in EXPECTED.items()}
samples = sorted(INPUT.rglob('sample_submission.csv'))
sample = pd.read_csv(([p for p in samples if 'playground' in str(p)] or samples)[0])
csvs = {key: pd.read_csv(paths[key]) for key in ('A', 'F')}
for sub in csvs.values():
    assert (sub['id'].to_numpy() == sample['id'].to_numpy()).all()

A = logit_rank(csvs['A']['is_fraud'].to_numpy())
F = logit_rank(csvs['F']['is_fraud'].to_numpy())
LM, L7F, GPD, MLP, GB, MS, G26 = (logit_rank(np.load(paths[k])) for k in ('LM', 'L7F', 'GPD', 'MLP', 'GB', 'MS', 'G26'))

# the 0.95352 ladder file: Strike 10 extrapolated from Strike 9, two LMs at 0.09, 5% GP + DVAE, half the tree move
s352 = A + 0.6 * (A - F) + 0.09 * (LM - F) + 0.09 * (L7F - F) + 0.05 * (GPD - F) - 0.05 * (MLP - F) - 0.025 * (GB - F)
# the new rung (0.95354): move 0.06 of the weight from the public GPT-2 seed to a second seed
score = logit_rank(s352) + 0.06 * (G26 - MS)
print('rank correlation with the 0.95352 file:', round(float(np.corrcoef(rankdata(score), rankdata(s352))[0, 1]), 6))
print('correlation of the new step with the ladder direction:', round(float(np.corrcoef(G26 - MS, s352 - F)[0, 1]), 3))

sample.assign(is_fraud=(rankdata(score) - 0.5) / len(score)).to_csv(OUT, index=False)
assert sha256(OUT) == OUT_SHA, sha256(OUT)
print('wrote', OUT, len(sample), 'rows, sha', OUT_SHA[:16])
```

</details>

### Verbatim code cells — `goodpjw`

<details><summary>cell 1 (markdown)</summary>

```python
# S6E11 LM Ladder: a second GPT-2 seed, 0.95354

**Public LB 0.95354**, above every public notebook when this version was written (best 0.95351).

The public board is led by *ladders*: a strong file plus small steps that the public score rewards. This notebook keeps one ladder
and adds a rung in each version. Every probe is listed, including the ones that lost.

| version | public LB | what it added |
|---|---|---|
| v1 | 0.95346 | a step towards a target-first LM we retrained ourselves (seed 7, fold-0 model) |
| **v2** | **0.95354** | that LM on all five folds, the two LMs at equal weight on daoviet's file, and a **second GPT-2 seed** |

**The new rung.** Every ladder file contains masayakawamata's public target-first GPT-2: it is a member of the stacks they are built on.
Adding *more* of it lost 14 points in our v1 probes. We trained a second seed of the same model (seed 2026, OOF AUC 0.95014,
rank correlation 0.989 with the public seed) and moved 0.06 of the weight from the public seed to ours, so the file leans on the
average of two seeds instead of one. It is the first step on this ladder that gains on **both** boards: +2 points public, and
+1.9e-5 on the nested OOF of our wide LM stack, in all five folds.

| ingredient | what it is |
|---|---|
| `A` | amanatar's Strike 10 file (public 0.95340) |
| `F` | Moran Z's Strike 9 file (public 0.95330), the common base every step is measured from |
| `LM` | megayak's target-first LM, test predictions (CC0; frozen copy in our dataset) |
| `L7F` | our compact target-first LM (seed 7), five fold models averaged ([dataset](https://www.kaggle.com/datasets/goodpjw2008/s6e11-compact-lm-seed7)) |
| `GPD` | daoviet's GP + DVAE test predictions ([dataset](https://www.kaggle.com/datasets/daoviet/s6e11-gp-dvae-orthogonal-apex)) |
| `MLP`, `GB` | two bags from our OOF bank, for the tree step in Areeb Ali's v17 |
| `MS` | masayakawamata's target-first GPT-2, test predictions ([dataset](https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof)) |
| **`G26`** | **our second seed of that GPT-2 (seed 2026), fold-0 model** ([dataset](https://www.kaggle.com/datasets/goodpjw2008/s6e11-target-first-gpt2-seed-2026)) |

```
s352  = A + 0.6 (A - F) + 0.09 (LM - F) + 0.09 (L7F - F) + 0.05 (GPD - F) - 0.05 (MLP - F) - 0.025 (GB - F)   # 0.95352
score = L(s352) + 0.06 (G26 - MS)                                                                            # 0.95354
```
All inputs are logit ranks, `L(v) = logit((rank(v) - 0.5) / n)`. `s352` is our 0.95352 file; it is put back on the logit-rank
scale before the new rung is added, exactly as the scored file was built.
```

</details>

<details><summary>cell 3 (code)</summary>

```python
import os, hashlib, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.special import logit
warnings.filterwarnings('ignore')

INPUT = Path(os.environ.get('KAGGLE_INPUT', '/kaggle/input'))
WORK = Path(os.environ.get('KAGGLE_WORKING', '/kaggle/working'))

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def find(name, digest=None, inside=None):   # a file anywhere under /kaggle/input, checked by SHA-256
    hits = [p for p in sorted(INPUT.rglob(name)) if inside is None or inside in str(p)]
    if digest is not None:
        hits = [p for p in hits if sha256(p) == digest]
    assert hits, (name, digest, inside)
    return hits[0]

def L(v):   # logit rank: AUC only sees order, so every input goes on one scale
    return logit((rankdata(v) - 0.5) / len(v))

# Public inputs, each checked byte for byte
EXPECTED = {
    'A':   ('submission_095340.csv', '12b525b84f29e32d6511e9d6105f6dc5eac79ae41b10140eed3353ea57afc159'),   # Strike 10, amanatar
    'F':   ('submission_095330.csv', 'f4a7a4f8636c99fabd82945c7d81956c5559c609dcb55cde6d4e03f80b0ea105'),   # Strike 9, Moran Z
    'LM':  ('megalm_test_sha3168.npy', '3168779be644a530c6bdc4c141cae168bfc2feb01ac21090c917cc978a7687d3'), # megayak LM (frozen copy)
    'L7F': ('lm7_test_5fold.npy', '68639fec79f081e97d40185443fa3903c9db3e62698abb01dfe31dcee217462a'),      # our LM7, 5 folds
    'GPD': ('test_ours.npy', '96f3bdde816e604438e61eac9722f7a7402f1c39bb8efe8999793b23f91d02f2'),           # daoviet GP + DVAE
    'MLP': ('test_gbn_bag5_w512.npy', '035d6544d7c03108fc42ee296a8660a45825e3b1b19b7cabce1c3f42cac4e7cb'),  # our OOF bank
    'GB':  ('test_gb_bag5.npy', '7d54e5f17e0707d88fec1a2cccb05885978ae6cc2a595c5f4575c7b7d45a9bb8'),        # our OOF bank
    'MS':  ('test.npy', '6c27aff7be744f7a0cc17809e0cd04fcdea0664234af88e3040a25235fb67906'),                # masayakawamata GPT-2
    'G26': ('gpt2_s2026_test_f0.npy', '40e08058ed7632ee328c2fe7d97040b856b36d2636f32cc6d2781ec9db080695'),  # our GPT-2 seed 2026
}
P = {k: find(n, h) for k, (n, h) in EXPECTED.items()}
sample = pd.read_csv(find('sample_submission.csv', inside='playground-series-s6e11'))
csv = {k: pd.read_csv(P[k]) for k in ('A', 'F')}
for d in csv.values():
    assert (d['id'].to_numpy() == sample['id'].to_numpy()).all()
A, F = L(csv['A']['is_fraud'].to_numpy()), L(csv['F']['is_fraud'].to_numpy())
X = {k: L(np.load(P[k])) for k in ('LM', 'L7F', 'GPD', 'MLP', 'GB', 'MS', 'G26')}
r = lambda a, b: np.corrcoef(rankdata(a), rankdata(b))[0, 1]   # rank correlation
print('rank correlation  G26 ~ MS %.4f | L7F ~ LM %.4f | GPD ~ Strike 10 %.4f' % (r(X['G26'], X['MS']), r(X['L7F'], X['LM']), r(X['GPD'], A)))
```

</details>

<details><summary>cell 4 (code)</summary>

```python
LM, L7F, GPD, MLP, GB, MS, G26 = (X[k] for k in ('LM', 'L7F', 'GPD', 'MLP', 'GB', 'MS', 'G26'))
s352 = A + 0.6 * (A - F) + 0.09 * (LM - F) + 0.09 * (L7F - F) + 0.05 * (GPD - F) - 0.05 * (MLP - F) - 0.025 * (GB - F)   # 0.95352
score = L(s352) + 0.06 * (G26 - MS)   # 0.95354: the 0.95352 file, back on the logit-rank scale, plus the new rung
sub = sample[['id']].copy()
sub['is_fraud'] = (rankdata(score) - 0.5) / len(score)
sub.to_csv(WORK / 'submission.csv', index=False)
print(sub.shape, '| rank correlation with the 0.95352 file %.6f, with Strike 10 %.6f' % (r(score, s352), r(score, A)))
print('correlation of the new step with the ladder direction (s352 - F): %.3f' % np.corrcoef(G26 - MS, s352 - F)[0, 1])
sub.head()
```

</details>

<details><summary>cell 5 (markdown)</summary>

```python
## How to read this

* **The gaps are a few hundred-thousandths** on the public 20% of the test rows. Files this close can swap order on the private 80%,
  so a ladder rung is a bet, not a measurement. `A` and `F` publish no OOF, so the file as a whole has no out-of-fold check.
* **What CV can say** (right panel of the figure): each move applied to the nested OOF of our wide LM stack (CV 0.952565).
  The GPT-2 seed swap and the LM swap both gain, in at least four of five folds. They point the same way as the public board.
  daoviet's GP + DVAE step costs about 0.5e-5 at 0.05, within noise. The GPT-2 swap is the one move that is nearly
  orthogonal to the ladder (correlation about 0 with `s352 - F`), so it adds something the ladder files did not have.
* **What comes next:** the other four fold models of GPT-2 seed 2026 are still scoring the test rows. When they finish, `G26`
  becomes a 5-fold average, as `L7F` did for our LM.
* **For final picks**, pair a file like this with one that is backed by out-of-fold CV.

## Credits

**masayakawamata** and **Matapanino** (the target-first GPT-2 code and its first seed), **daoviet** (GP + DVAE predictions and the
step to 0.95351), **Areeb Ali** (v17 and the megayak-LM trim in v19; with **samanyu1808**, the LM-step idea on these files),
**amanatar** (Strike 10), **Moran Z** (Strike 9), **megayak** (target-first LM predictions and the compact LM code we retrained),
**Chris Deotte** (the target-first recipe, S6E10). If this was useful, please upvote them too.
```

</details>

### Reconstruction checks

- All input SHAs OK: **True**
- 1−ρ(recon s354, s352) = `3.088207e-05`
- 1−ρ(recon s354, official pub 0.95354) = `0.0`
- max |Δrank| vs official = `0.0`
- CSV byte-SHA may differ from OUT_SHA due to pandas float formatting; rank identity is the real check

### Why probes scored 0.95352 — distances

| pair | 1−ρ |
|---|---:|
| slot1_vs_s354 | `1.200001e-05` |
| slot1_vs_s352 | `4.234805e-05` |
| slot1_vs_S7 | `4.020927e-04` |
| slot3_vs_s354 | `9.187537e-05` |
| slot3_vs_S7 | `3.126464e-04` |
| S7_vs_s354 | `3.722933e-04` |

## Step 2 — Components A–D on `z_ref_old + 0.05·δ1_v4` (AUC **0.9526953**)

Evaluator folds: [0.952011, 0.952665, 0.95737, 0.952878, 0.948539]

1−ρ(S7, s354_pub) = `3.722933e-04`

### A — `u_L7_5f_swap` (5-fold L7 residual orth to S7)

Note: OOF uses orth(zrk(L7F_5fold) - S7_proxy). Fold0-only 1fold tensor cannot form a full 699960 1fold vector; L7F solo OOF AUC=0.949448. flipped=True

**Best w=0.02:** pos=2/5 · meanΔ=-0.002e-5 · ΔF0=0.009e-5 · folds=[0.009, -0.01, -0.006, 0.005, -0.007]

| w | pos/5 | meanΔe-5 | ΔF0 e-5 | fold Δe-5 |
|---:|---:|---:|---:|---|
| 0.02 | 2 | -0.002 | 0.009 | [0.009, -0.01, -0.006, 0.005, -0.007] |
| 0.04 | 1 | -0.005 | 0.012 | [0.012, -0.006, -0.007, -0.001, -0.024] |
| 0.06 | 2 | -0.004 | 0.023 | [0.023, -0.013, -0.011, 0.004, -0.026] |
| 0.08 | 2 | -0.004 | 0.028 | [0.028, -0.009, -0.011, 0.005, -0.033] |
| 0.1 | 2 | -0.004 | 0.032 | [0.032, -0.011, -0.011, 0.002, -0.03] |

L7F solo OOF AUC=0.9494475201107185 folds=[0.94754, 0.94977, 0.95517, 0.94907, 0.94568]

### B — `u_GPD_orth`

**Best w=0.06:** pos=3/5 · meanΔ=0.006e-5 · ΔF0=0.009e-5 · folds=[0.009, 0.016, -0.01, 0.023, -0.008]

| w | pos/5 | meanΔe-5 | ΔF0 e-5 | fold Δe-5 |
|---:|---:|---:|---:|---|
| 0.02 | 3 | 0.001 | 0.006 | [0.006, 0.007, -0.007, 0.013, -0.012] |
| 0.04 | 3 | 0.004 | 0.007 | [0.007, 0.013, -0.009, 0.021, -0.011] |
| 0.06 | 3 | 0.006 | 0.009 | [0.009, 0.016, -0.01, 0.023, -0.008] |
| 0.08 | 3 | 0.006 | 0.005 | [0.005, 0.034, -0.015, 0.018, -0.013] |
| 0.1 | 3 | 0.005 | 0.001 | [0.001, 0.04, -0.015, 0.02, -0.021] |

### C — `u_G26_MS_orth`

Note: OOF uses 5-fold G26_oof; test G26 is still Fold-0-only

**Best w=0.1:** pos=5/5 · meanΔ=0.022e-5 · ΔF0=0.015e-5 · folds=[0.015, 0.038, 0.005, 0.031, 0.024]

| w | pos/5 | meanΔe-5 | ΔF0 e-5 | fold Δe-5 |
|---:|---:|---:|---:|---|
| 0.02 | 4 | 0.003 | 0.0 | [0.0, 0.002, 0.004, 0.011, -0.001] |
| 0.04 | 5 | 0.012 | 0.006 | [0.006, 0.018, 0.0, 0.028, 0.007] |
| 0.06 | 5 | 0.013 | 0.007 | [0.007, 0.023, 0.002, 0.025, 0.008] |
| 0.08 | 5 | 0.017 | 0.008 | [0.008, 0.034, 0.005, 0.025, 0.014] |
| 0.1 | 5 | 0.022 | 0.015 | [0.015, 0.038, 0.005, 0.031, 0.024] |

### D — `δ_v10_stageB_L7_5f` (+δ1 blend)

**Best w=0.08:** pos=3/5 · meanΔ=-0.0e-5 · ΔF0=0.004e-5 · folds=[0.004, -0.007, 0.003, -0.009, 0.008]

| w | pos/5 | meanΔe-5 | ΔF0 e-5 | fold Δe-5 |
|---:|---:|---:|---:|---|
| 0.02 | 0 | -0.003 | -0.003 | [-0.003, -0.004, -0.002, -0.0, -0.004] |
| 0.04 | 0 | -0.003 | 0.0 | [0.0, -0.007, -0.005, -0.002, -0.002] |
| 0.06 | 1 | -0.002 | 0.004 | [0.004, -0.006, -0.0, -0.006, -0.001] |
| 0.08 | 3 | -0.0 | 0.004 | [0.004, -0.007, 0.003, -0.009, 0.008] |
| 0.1 | 2 | 0.001 | 0.007 | [0.007, -0.001, -0.001, -0.016, 0.014] |

## Implication for next probes (`NO_SUBMIT` guidance)

- Do **not** re-anchor on `s354_pub` (leaves S7 by ~4e-4).
- Next candidates must satisfy `1−ρ(z_S7_ours) ≤ 0.15e-4`.
- Prefer components with pos≥4/5 on the 0.9526953 evaluator before any live submit.
- `smugcqcy2oqzeg` left untouched (independent L7_5F seed still training).

`NO_SUBMIT=True` — no Kaggle upload.
