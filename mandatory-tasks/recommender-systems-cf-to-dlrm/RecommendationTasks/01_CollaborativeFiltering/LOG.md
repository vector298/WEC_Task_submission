## 2026-10-03: Task 01 data, split, baselines and memory-based CF

>> Setup: MovieLens 100K (943 users, 1,682 items, 100,000 ratings), random
80/10/10 split of the ratings, seed 42; all statistics from the training part.

>> Baselines: global, item and user means, and a bias baseline (mu + b_u + b_i,
biases shrunk by 10). Memory-based CF: cosine similarity of residuals over the
bias baseline, top-k neighbours, shrinkage n/(n + lambda); item-item and
user-user; k in 20, 50, 100, 200, 400 and lambda in 0, 25. Kaggle, CPU.

>> Result: val RMSE / MAE: global mean 1.1320 / 0.9485, item mean 1.0295 /
0.8179, user mean 1.0457 / 0.8386, bias baseline 0.9494 / 0.7517. Best
item-item (k 200, lambda 25) 0.9169 / 0.7179; best user-user (k 400, lambda
25) 0.9277 / 0.7270. 21 of 10,000 val ratings involve items unseen in train.

>> Observation: biases explain much of the structure (16% below the global
mean); the best neighbourhood method improves on the bias baseline by 0.033
RMSE (3.4%). Only 11 of 20 settings beat the bias baseline; shrinkage helped
in all 10 matched comparisons; item-item was better than user-user at the
best settings and in 8 of 10 matched settings. The best user-user k and the
best shrinkage sit at the edge of the grid, so the grid is widened next.

>> Next: wider grid with a paired bootstrap, then matrix factorisation.


## 2026-10-0X: Task 01 seed check and ranking on validation


>> Setup: the selected MF configuration (64 factors, reg 0.05) repeated with seeds 43
and 44. Ranking: for each user all movies unrated in train ranked by predicted
rating (unclipped), top-10 against validation ratings >= 4 (5,722 users with at
least one), with random, popularity and bias-baseline lists as references; also a
within-user NDCG@10 over each user's own held-out validation ratings (gain =
rating, chance level = constant predictions).

>> Result: MF val RMSE 0.8520, 0.8536, 0.8538 (mean 0.8532, std 0.0008). Top-10
precision / NDCG: random 0.0031 / 0.0036, popularity 0.0768 / 0.1033, bias
baseline 0.0321 / 0.0426, item-item 0.0244 / 0.0260, user-user 0.0001 / 0.0002, MF
0.0392 / 0.0505. Within-user NDCG@10: chance 0.8691, item mean 0.9323, bias
baseline 0.9325, user-user 0.9419, item-item 0.9453, MF 0.9452.

>> Observation: the MF margin over item-item survives seed noise (0.0043 on the seed
mean). Popularity beats every rating-based method on top-10; MF is the best of
those. User-user's top-10 is almost empty of hits; my untested explanation is
extreme predictions from single raters on obscure movies. Within each user's own
items, item-item and MF tie and all methods are well above chance. The bias
baseline equals the item mean because a user bias cannot change a user's own order.

>> Next: test run once, README.

## 2026-10-0X: Task 01 final test (one run)


>> Setup: all models trained on the train split only, settings from validation;
rating prediction on the 100,022 test ratings with 1,000-resample bootstrap
intervals; top-10 ranking excluding train and validation movies; within-user NDCG.

>> Result: test RMSE / MAE: global mean 1.1160 / 0.9327, item mean 0.9742 / 0.7782,
user mean 1.0334 / 0.8273, bias baseline 0.9031 / 0.7153, user-user 0.8643 / 0.6792,
item-item 0.8523 / 0.6679, MF 0.8471 / 0.6668. RMSE(item-item) - RMSE(MF) =
0.0052 [0.0036, 0.0068]. Top-10 precision: popularity 0.0894, MF 0.0414, bias
baseline 0.0340, item-item 0.0271, random 0.0030, user-user 0.0001.

>> Observation: the method ordering from validation holds on test; every method is
about 0.005 RMSE lower on test (an easier split for all). MF is best on RMSE and
MAE, but only 0.005 RMSE ahead of item-item and tied on MAE. On offline top-10
popularity wins clearly.

>> Next: README and commit.
