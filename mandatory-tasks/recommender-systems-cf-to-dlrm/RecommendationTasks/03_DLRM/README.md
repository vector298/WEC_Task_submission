# Task 03: DLRM from Scratch

A Deep Learning Recommendation Model written in PyTorch for click prediction on
the supplied CTR dataset, compared with Task 02 on the same split and protocol.

## Contents
- `notebook.ipynb`: data and split check, DLRM implementation, first run,
  sweep, ablations, selected model, test, uncertainty (Kaggle, CPU)
- `results/`: loss curves, calibration plot and the results table
- `LOG.md`: timestamped log

## Data and split
Identical to Task 02: stratified 80/20 split of `train.csv` (seed 42), 32,000
training rows (1,024 clicks) and 8,000 validation rows (256 clicks);
`test.csv` untouched until the end. All preprocessing statistics come from the
training part only (median fill, clipped negatives, log1p and standardisation
for the 13 dense features; indices for the 26 categorical fields, 0 for missing
or unseen, 1 for fewer than 5 train occurrences; 9,912 embedding rows). The
notebook asserts that the split fingerprint equals Task 02's (clicks 1,024 and
256; categorical index sums 38,103,084 and 9,050,373).

## DLRM research paper notes (Naumov et al., 2019, arXiv 1906.00091)
- Embedding tables represent the categorical fields; an MLP processes the dense
  fields; an interaction layer combines them; a second MLP produces the click
  probability.
- The bottom MLP outputs a vector of the same length as the embeddings, so
  dense and categorical information share one space.
- The interaction layer takes the dot product of every distinct pair among the
  embeddings and the processed dense vector, as in factorisation machines, and
  concatenates the results with the dense vector. This models second-order
  interactions only; higher orders must come from the MLPs.
- At production scale the embedding tables dominate memory, which motivates
  model parallelism in the paper. Here the tables are tiny (9,912 rows).
   

## Implementation
Bottom MLP 13 to 32 to d (ReLU on every layer), 26 embedding tables of length d
(normal initialisation with standard deviation 1/sqrt(d)), the 27 vectors
interacted by all 27 x 26 / 2 = 351 pairwise dot products, concatenated with the
dense vector, then a top MLP with ReLU and optional dropout to one logit.
Checks: the batched interaction matches a plain loop over pairs, and the
parameter count matches a hand calculation (d = 8: 134,409, of which 79,296 are
embeddings). Training protocol identical to Task 02: binary cross-entropy,
Adam (lr 1e-3, weight decay 1e-5), batch 512, early stopping on validation
PR-AUC (patience 3, at most 15 epochs), threshold from validation F1. The
concatenation ablation replaces the interaction layer with plain concatenation
of the dense vector and the 26 embeddings.

## Results
**First run** (d = 8, top (128, 64), seed 42): best epoch 3, val PR-AUC 0.0630,
ROC-AUC 0.6756, log loss 0.1371; train loss fell from 0.130 to 0.109 while val
loss rose from 0.1371 to 0.1475.

**Sweep** (validation, mean over seeds 42, 43, 44):

| Config | Interaction | Params | PR-AUC (std) | AUC | Log loss |
|---|---|---|---|---|---|
| D1 d8 top(128,64) dr0.0 | dot | 134,409 | 0.0735 (0.0083) | 0.6919 | 0.1357 |
| D1n same, N(0,1) init | dot | 134,409 | 0.0687 (0.0084) | 0.6592 | 0.1412 |
| P1 d4 top(32) dr0.0 | dot | 51,653 | 0.0698 (0.0040) | 0.6878 | 0.1374 |
| P1c d4 top(32) dr0.0 | concat | 43,749 | 0.0892 (0.0056) | 0.7231 | 0.1315 |
| P2 d8 top(64) dr0.3 | dot | 103,113 | 0.0780 (0.0063) | 0.6949 | 0.1357 |
| P2c d8 top(64) dr0.3 | concat | 93,961 | 0.0921 (0.0026) | 0.7331 | 0.1307 |
| D4 d4 top(64) dr0.3 | dot | 63,077 | 0.0738 (0.0072) | 0.6982 | 0.1364 |
| **D5 d16 top(64) dr0.3** | dot | 183,185 | **0.0836 (0.0026)** | 0.7106 | 0.1338 |

Selection rule (fixed in advance): highest mean PR-AUC among the dot-product
configurations, then the fewest parameters within one standard deviation of it.
D5 was selected; it uses the largest d tried, so the optimum may lie beyond 16.

**Ablation** (matched pairs; val PR-AUC for seeds 42, 43, 44):

| Pair | Dot | Concat | Difference |
|---|---|---|---|
| P1 (d4, top 32, dropout 0) | 0.0722, 0.0641, 0.0731 | 0.0812, 0.0929, 0.0933 | -0.009, -0.029, -0.020 |
| P2 (d8, top 64, dropout 0.3) | 0.0692, 0.0839, 0.0809 | 0.0898, 0.0957, 0.0909 | -0.021, -0.012, -0.010 |

Removing the interaction layer improved PR-AUC in all six comparisons (AUC and
log loss agree). Caveats: three seeds per pair; pairing by seed is loose because
the models have different shapes; removing the interactions also shrinks the
top MLP's input.

**Initialisation check:** with PyTorch's default N(0, 1), the first
configuration was worse in all three seeds (PR-AUC 0.0569, 0.0757, 0.0735
against 0.0630, 0.0832, 0.0743), so the 1/sqrt(d) choice is not what made DLRM
weak.

**Selected model (D5, seed 42)** at threshold 0.09; 183,185 parameters, 0.733 MB
of float32 weights, 8 s training on CPU:

| Split | ROC-AUC (95% CI) | PR-AUC (95% CI) | Log loss (95% CI) | Acc | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Validation | 0.712 [0.681, 0.744] | 0.0831 [0.0667, 0.1091] | 0.1338 [0.1216, 0.1463] | 0.9066 | 0.1463 | 0.1034 | 0.25 |
| Test (one run) | 0.681 [0.655, 0.708] | 0.0665 [0.0557, 0.0846] | 0.1427 [0.1306, 0.1556] | 0.9031 | 0.1053 | 0.0762 | 0.1701 |

Calibration: mean predicted probability 0.0342 (val) and 0.0332 (test) against
observed 0.0320 and 0.0335; the highest bin is overconfident (about 0.12
predicted against about 0.084 observed, read off the plot).

## Comparison with Task 02
| | Task 02 selected (F) | Task 03 selected (D5) |
|---|---|---|
| Parameters / weight memory | 43,457 / 0.17 MB | 183,185 / 0.73 MB |
| Training time per epoch (CPU, approximate) | about 0.5 s | about 1.0 s |
| Val PR-AUC, 3-seed mean (std) | 0.1036 (0.0075) | 0.0836 (0.0026) |
| Val AUC / log loss, 3-seed mean | 0.7187 / 0.1315 | 0.7106 / 0.1338 |
| Test ROC-AUC / PR-AUC / log loss (one run) | 0.6676 / 0.0647 / 0.1441 | 0.681 / 0.0665 / 0.1427 |

On validation (three seeds) Task 02's model leads by 0.020 PR-AUC, 2.7 of its
standard deviations. On test DLRM is slightly ahead, but the bootstrap intervals
overlap heavily, so there is no measurable difference. The two ran on Colab CPU
and Kaggle CPU; the Task 02 sweep reproduced identically across the two, and the
library versions differ only in pandas (2.2.3 against 2.3.3). Task 02 used
PyTorch's default N(0, 1) embedding initialisation; DLRM used 1/sqrt(d), which
did not hurt it (see the initialisation check).

## What DLRM adds and what it costs
- **In principle:** it computes second-order feature interactions explicitly as
  dot products of embeddings (the idea behind matrix factorisation and
  factorisation machines, extended to all field pairs), where a plain MLP must
  learn such cross-terms implicitly, and it keeps dense and categorical features
  in one embedding space.
- **Cost:** 351 extra inputs to the top MLP, more parameters (4 times as many as
  Task 02's selected model here), about twice the time per epoch, and an extra
  design constraint (the bottom MLP must output the embedding length).
- **Observed here:** no gain. In the controlled ablation, removing the
  interactions improved PR-AUC in every comparison, and on test DLRM is not
  distinguishable from Task 02. With 1,024 training clicks and sparse,
  partly rare-valued categorical fields, the extra interaction inputs probably
  add ways to memorise; this was not tested. DLRM targets far larger data.

## Key findings
- The first DLRM overfitted within three epochs; the sweep shows smaller
  networks, dropout and a larger d help somewhat, but the architecture stays
  below Task 02's model on validation.
- Interactions hurt in 6 of 6 matched comparisons; the initialisation choice is
  not the cause.
- Val and test disagree about which of DLRM and Task 02 is ahead, and the test
  difference is inside the bootstrap noise.
- Results depend on the device and seed (Task 02 showed 0.0056 PR-AUC between
  CPU and GPU for the same seed), so single-run differences below about 0.01
  are not trusted.

## Limitations and not done
- Three seeds per configuration; one split; 256 validation and 335 test clicks.
- The sweep grid is small and D5 sits at its edge (d = 16); no learning-rate or
  weight-decay search; the bottom MLP and top MLP shapes were barely varied (no
  dense-MLP ablation); I did not test why the interactions hurt.
- The ablation changes the top MLP's input size as well as removing the
  interactions.
- Comparison across all three tasks, including Task 01 (collaborative
  filtering, a different dataset and metric): to be added after Task 01.

## Reproduce
Kaggle notebook, CPU only, Internet on. Python 3.13.15, numpy 2.1.3, pandas
2.3.3, torch 2.11.0+cpu, scikit-learn 1.6.1, matplotlib 3.10.0. Seeds: split 42;
sweep 42, 43, 44; selected model 42.
