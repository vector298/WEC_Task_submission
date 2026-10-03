# Task 01: Collaborative Filtering (memory-based and matrix factorisation)

## Contents
- `notebook.ipynb`: data exploration, split, baselines, memory-based CF, matrix
  factorisation from scratch, ranking metrics, test (Kaggle)
- `results/`: figures and the results table
- `LOG.md`: timestamped log

## Dataset: MovieLens 1M
GroupLens; Harper and Konstan, "The MovieLens Datasets: History and Context", ACM
TiiS, 2015. 1,000,209 ratings from 1 to 5 by 6,040 users on 3,706 movies (density
4.47%, no duplicate pairs). Every user has at least 20 ratings (median 96, maximum
2,314); items have 1 to 3,428 (median 123); 7.8% of items have fewer than 5.
Ratings skew positive (mean 3.582; 4 is the most common at 34.9%).
It suits both algorithms: explicit ratings let both predict a rating and be scored
with RMSE and MAE; the matrix is sparse but not empty; and it is large enough for
stable estimates (about 100,000 ratings each for validation and test) yet small
enough for dense matrices. Limits: explicit, not implicit, feedback; only users
who chose to rate appear; popularity is heavily skewed; small by industrial
standards. I first ran a pilot on MovieLens 100K (baselines and the neighbourhood
grid only; same method) and then used 1M for everything reported here.

## Setup
Random 80/10/10 split of the ratings (seed 42): 800,167 train, 100,020 validation,
100,022 test. Every statistic is learned from the training part only. In
validation, no user and 19 items (0.02%) are unseen in training and fall back to
the baseline. Settings are chosen on validation; test is run once, with models
trained on the train split only. Metrics: RMSE and MAE (rating prediction, 95%
bootstrap intervals), plus top-10 precision, recall and NDCG and a within-user
NDCG as ranking views.

## Baselines
Global mean, item mean, user mean, and a bias baseline (mu + b_u + b_i, biases
shrunk by 10). The bias baseline is 18.7% below the global mean on validation
RMSE and is the reference that matters.

## A. Memory-based CF
Prediction = baseline + the similarity-weighted average of the user's residuals
(rating minus baseline) on the neighbours. Similarity: cosine of residual vectors
(missing = 0) times shrinkage n/(n + lambda); each row keeps its k most similar
neighbours overall, and only neighbours the user rated contribute; with none, the
prediction falls back to the baseline. Grid: item-item and user-user, k in 50 to
800, lambda in 0, 25, 100 (and 200, 400 at k = 200, 400). Selected: item-item
k = 200, shrinkage 400 (val RMSE 0.8575, effectively tied with shrinkage 100 to
400); user-user k = 400, shrinkage 100 (0.8694). 27 of the first 30 settings beat
the bias baseline; item-item beat user-user in all 15 matched settings (paired
bootstrap of the RMSE difference 0.0117 [0.0104, 0.0130]).

## B. Matrix factorisation (from scratch)
r_hat = mu + b_u + b_i + p_u . q_i; squared error plus L2 penalty; mini-batch SGD
with hand-written gradients (batch 256, lr 0.01), factors N(0, 0.1^2); early
stopping on validation RMSE (patience 5, at most 100 epochs); predictions clipped
to [1, 5]. Grid: factors 16, 32, 64 x reg 0.02, 0.05, 0.1. Regularisation 0.05 was
best at every size (val RMSE 0.8533, 0.8525, 0.8520); reg 0.02 overfit (peaks at
epochs 11 to 18, worse with more factors); reg 0.1 was slower and ended worse (the
16-factor run hit the 100-epoch cap). Selected: 64 factors, reg 0.05 (val RMSE
0.8520, MAE 0.6693, best epoch 25); over seeds 42, 43, 44: mean 0.8532, std 0.0008.
The training curve shows overfitting (train RMSE falls to about 0.67 while
validation RMSE flattens near 0.853).

## Results on test (one run)
| Method | Test RMSE (95% CI) | Test MAE |
|---|---|---|
| Global mean | 1.1160 [1.1114, 1.1206] | 0.9327 |
| Item mean | 0.9742 [0.9699, 0.9786] | 0.7782 |
| User mean | 1.0334 [1.0287, 1.0379] | 0.8273 |
| Bias baseline | 0.9031 [0.8989, 0.9073] | 0.7153 |
| User-user (k 400, shrinkage 100) | 0.8643 [0.8601, 0.8684] | 0.6792 |
| Item-item (k 200, shrinkage 400) | 0.8523 [0.8481, 0.8563] | 0.6679 |
| **Matrix factorisation (64, reg 0.05)** | **0.8471 [0.8431, 0.8512]** | **0.6668** |

RMSE(item-item) - RMSE(MF) = 0.0052 [0.0036, 0.0068] on test (0.0055 on
validation). Validation RMSEs were: bias 0.9085, user-user 0.8694, item-item
0.8575, MF 0.8520; every method is about 0.005 lower on test (an easier split).

**Ranking (test; top-10 over all unseen movies, relevant = rating >= 4, train and
validation movies excluded; 5,749 users):**

| Method | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Random | 0.0030 | 0.0033 | 0.0035 |
| Popularity | **0.0894** | **0.0944** | **0.1201** |
| Bias baseline | 0.0340 | 0.0371 | 0.0450 |
| Item-item | 0.0271 | 0.0178 | 0.0283 |
| User-user | 0.0001 | 0.0003 | 0.0002 |
| MF | 0.0414 | 0.0453 | 0.0528 |

Within-user NDCG@10 (gain = rating; chance level 0.8686): bias baseline 0.9332,
user-user 0.9419, item-item 0.9443, MF 0.9445.

## When is each approach preferable?
Measured on this dataset:
- **Accuracy:** MF has the lowest RMSE on test (0.8471) and validation, but its
  advantage over item-item is small (0.0052 RMSE, 0.6%; the MAE gap is 0.0011).
  Item-item beats user-user by about 0.012 RMSE (0.0117 validation, 0.0120 test).
  Both beat the bias baseline by 4 to 6%.
- **Cost:** the neighbourhood methods need similarity matrices that grow with the
  square of users or items (item-item 3,706^2 = 13.7 million entries; user-user
  6,040^2 = 36.5 million) and had no training step (about 5 s per setting here).
  MF stores (6,040 + 3,706) x 64 + biases = 633,490 parameters (about one per
  training rating), predicts with a dot product, but needs training (about 70 s per
  configuration) and has a sharp regularisation optimum.
- **Top-N:** on offline top-10, popularity beat every rating-based method; MF was
  the best of those; my user-user implementation produced almost no hits (see
  below).

In general (reasoning, not tested here): neighbourhood methods are easy to explain
("people like you liked X") and can use new ratings without retraining, which suits
small or fast-changing data and cases where explanations matter; matrix
factorisation compresses the data into shared factors and tends to do better when
the matrix is sparse and large, at the price of training, tuning and less direct
interpretability. Both have no information about a brand-new user or movie.

## Limitations and not done
- One split; the best of 59 settings was chosen on the same validation ratings.
- MF was run with one seed in the grid (three for the selected configuration);
  the 16-factor reg 0.1 run hit the epoch cap; reg values between 0.02 and 0.1
  were not tried.
- The neighbourhood top-k is taken over all neighbours with only raters
  contributing (a simplification); no damping or minimum support, which probably
  explains user-user's near-empty top-10 (untested).
- Ranking treats unrated movies as irrelevant; no popularity-debiased evaluation;
  no implicit-feedback models; models trained on the train split only (not
  retrained on train plus validation for the test run).
- Cold start (new users or movies) was not studied beyond the baseline fallback.

## Reproduce
Kaggle notebook, Internet on (the data is fetched from GroupLens). [hardware: CPU
or GPU accelerator setting; the code is numpy and runs on the CPU]. [paste the
versions and seeds lines printed by cell 10]. Seeds: split 42; MF 42, 43, 44 (final
model 42). The whole notebook takes about 25 minutes.
