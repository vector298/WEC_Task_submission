# Task 02: Neural CTR Prediction

Predicting ad clicks with a vanilla neural network on the supplied dataset.

## Contents
- `notebook.ipynb`: exploration, preprocessing, baseline, architecture sweep,
  selected model, test, intervals and checks (Kaggle)
- `results/`: loss curves, reliability plot, results table
- `LOG.md`: timestamped log

## Data
`train.csv` (40,000 impressions) and `test.csv` (10,000): 13 integer features,
26 categorical features, binary `label`. Click rate 3.2% (1,280 clicks) in train
and 3.35% (335) in test. Missing values reach 40% (integer) and 36%
(categorical). Categoricals are hashed strings with 3 to 13,508 distinct values
(15 columns above 1,000); on average 9.2% of non-missing test values never
appear in train (up to 36%). Integers are heavily skewed (skewness 108 for
`integer_feature_1`); only `integer_feature_8` has negatives (2,455 rows).

## Split and preprocessing
The supplied train file is split 80/20, stratified, seed 42: 32,000 training
rows (1,024 clicks) and 8,000 validation rows (256 clicks). `test.csv` is
untouched until the final evaluation. All statistics come from the training
part only: integer missing values filled with the train median, negatives
clipped to 0, log1p, standardised with train mean and standard deviation.
Categorical values map to index 0 (missing or unseen), 1 (fewer than 5 train
occurrences) or a per-value index. Embedding tables total 9,912 rows (4 to 1,291
per column). 14.1% of validation and 15.7% of test categorical cells map to the
unknown index. Missing and unseen share one index, a simplification.
Split fingerprint (Task 03 reproduces it): train clicks 1,024, val clicks 256,
categorical index sums 38,103,084 (train) and 9,050,373 (val).

## Model
Learned embeddings per categorical column, concatenated with the 13 dense
features, then fully connected ReLU layers with dropout, one logit, binary
cross-entropy, Adam (lr 1e-3, weight decay 1e-5), batch 512. The epoch is
chosen by validation PR-AUC with early stopping (patience 3, at most 15 epochs).
The decision threshold is chosen on validation to maximise F1.

## Metrics
PR-AUC (average precision; random ranker about 0.032), ROC-AUC, log loss
(constant-rate predictor 0.1416 on val, 0.1467 on test), accuracy (majority
class 0.968 on val), and F1, precision and recall at the validation-chosen
threshold. A reliability plot (8 quantile bins) is in `results/`.

## Baseline
8-d embeddings, hidden (256, 128), dropout 0.2 (169,153 parameters): best epoch
2, val ROC-AUC 0.7179, PR-AUC 0.0993, log loss 0.1313. Train loss fell from
0.177 to 0.126 while validation loss rose from 0.1306 to 0.1350.

## Architecture sweep (validation, mean of seeds 42, 43, 44)
| Config | Params | Best epoch | PR-AUC (std) | AUC | Log loss |
|---|---|---|---|---|---|
| A 8d (256,128) dropout 0.2 | 169,153 | 3.7 | 0.0994 (0.0042) | 0.7148 | 0.1327 |
| B 4d (64) dropout 0.2 | 47,265 | 10.7 | 0.1025 (0.0006) | 0.7189 | 0.1316 |
| C 8d (128,64) dropout 0.3 | 116,033 | 3.7 | 0.1000 (0.0054) | 0.7228 | 0.1318 |
| D 16d (256,128) dropout 0.5 | 301,697 | 3.3 | 0.1017 (0.0068) | 0.7213 | 0.1318 |
| E 8d (512,256,128) dropout 0.3 | 357,313 | 1.7 | 0.0980 (0.0093) | 0.7257 | 0.1313 |
| F 4d (32) dropout 0.0 | 43,457 | 13.3 | 0.1036 (0.0075) | 0.7187 | 0.1315 |

Selection rule fixed in advance: highest mean validation PR-AUC; among
configurations within one standard deviation of it, the fewest parameters. All
six qualified, so F was selected.

## Selected model: F
43,457 parameters (91% in embeddings), 0.17 MB of float32 weights, 8 s training
time. Threshold 0.10.

| Split | ROC-AUC (95% CI) | PR-AUC (95% CI) | Log loss (95% CI) | Acc | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Validation | 0.720 [0.688, 0.752] | 0.1014 [0.0790, 0.1396] | 0.1315 [0.1188, 0.1442] | 0.9416 | 0.1616 | 0.1495 | 0.1758 |
| Test (one run) | 0.668 [0.635, 0.694] | 0.0647 [0.0540, 0.0814] | 0.1441 [0.1310, 0.1575] | 0.9415 | 0.0845 | 0.0888 | 0.0806 |

Intervals are 95% bootstrap intervals (1,000 resamples of the stored
predictions); they capture noise from the number of clicks only.

## Key findings
- The signal is weak but real: validation PR-AUC is about 3.2 times the click
  rate, and log loss beats the constant-rate predictor by 0.0101 (7%).
- The six architectures are not separated: mean PR-AUC spans 0.0056, about the
  size of the seed standard deviations. Larger networks overfit within 2 to 4
  epochs, while the small ones were still improving at the 15-epoch cap and may
  be undertrained.
- Test performance is clearly weaker than validation: PR-AUC is 1.9 times the
  test click rate (3.1 times on validation), and log loss beats the constant
  predictor by only 0.0026. The intervals only just touch, and ROC-AUC fell too
  (0.720 to 0.668), so test ranking is worse, not just the probability scale
  (the model under-predicts mildly on both splits: 88% and 80% of the observed
  rate).
- The two files differ: missing-rate gaps of up to 2.6 percentage points
  (about 8 standard errors by a rough estimate), and more unknown categories in
  test. I did not show that this explains the drop; selection on validation
  probably explains little (config spread 0.0056, against a gap of 0.037).
- Accuracy (0.94) is below the majority-class 0.968 because the F1-optimal
  threshold flags about 4% of impressions; accuracy is not informative here.

## Not done
Deep & Cross Network (bonus), a train-vs-test classifier to test the
distribution-shift explanation, a missing-value indicator, more epochs for the
small networks, other learning rates and regularisation.

## Reproduce
Google colab notebook with Internet on (the data is fetched from the public task
repository). Hardware and versions: python 3.13.15 | numpy 2.1.3 | pandas 2.2.3 | torch 2.11.0+cpu | scikit-learn 1.6.1 | matplotlib 3.10.0
device: cpu | cpu
Seeds: split 42; sweep seeds 42, 43, 44; selected model 42.
