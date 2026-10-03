## 2026-10-03: Task 02 architecture sweep, selected model and test
>> Hypothesis:
- val has only 256 clicks, so single-run metrics are noisy and gaps of 0.005
  PR-AUC may not be real; I average over three seeds
- smaller networks or more dropout should overfit less than the baseline
- differences between architectures may be small against the seed spread
- selection rule fixed before running: highest mean val PR-AUC; among
  configurations within one standard deviation of it, the one with fewer
  parameters

>> Setup: six MLP configurations x seeds 42, 43, 44; early stopping on val
PR-AUC (patience 3, at most 15 epochs); selected model retrained with seed
42; threshold from val F1; test run once.

>> Result: mean val PR-AUC 0.0994 (A), 0.1025 (B), 0.1000 (C), 0.1017 (D),
0.0980 (E), 0.1036 (F); seed std 0.0006 to 0.0093. All six within one std of
the best, so the rule selected F (4-d embeddings, one hidden layer of 32, no
dropout, 43,457 parameters). Final: val ROC-AUC 0.7204, PR-AUC 0.1014, log
loss 0.1315, F1 0.1616 at threshold 0.10. Test (one run): ROC-AUC 0.6676,
PR-AUC 0.0647, log loss 0.1441, F1 0.0845, precision 0.0888, recall 0.0806.

>> Observation: the sweep does not separate the architectures (spread 0.0056,
about the size of the seed std); larger models overfit within 2 to 4 epochs
and the small ones were still improving at the 15-epoch cap, so F may be
undertrained. Test is much weaker than val (PR-AUC 1.9x the click rate
against 3.1x on val; log loss 0.1441 against about 0.1467 for a constant
predictor). Cause not diagnosed; candidates: selection on val, differences
between test.csv and train.csv, and sampling noise (335 test clicks).

>> Next: bootstrap intervals for val and test, then Task 03 (DLRM) on the same
split and preprocessing.


## 2026-10-03: bootstrap intervals and val/test distribution check
>> Hypothesis:
- with 256 val clicks and 335 test clicks the PR-AUC intervals should be
  wide and the val and test intervals might overlap
- if the model is miscalibrated on test, mean predicted probability should
  differ visibly from the observed click rate
- if test.csv differs from train.csv, the missing-rate and unknown-category
  gaps may show it

>> Setup: 1,000 bootstrap resamples of the stored val and test predictions (no
retraining, no second test run); distribution checks on the raw files.

>> Result: val ROC-AUC 0.720 [0.688, 0.752], PR-AUC 0.1014 [0.0790, 0.1396],
log loss 0.1315 [0.1188, 0.1442]; test ROC-AUC 0.668 [0.635, 0.694], PR-AUC
0.0647 [0.0540, 0.0814], log loss 0.1441 [0.1310, 0.1575]. Mean predicted
probability 0.0282 (val) and 0.0269 (test) against observed 0.0320 and
0.0335. Largest train-vs-test missing-rate gap 0.026 (integer_feature_6 and
_10). Unknown-index share 0.141 (val), 0.157 (test). Predicted-positive share
at threshold 0.10: 0.0376 (val), 0.0304 (test).

>> Observation: the intervals are wide but only just overlap, so noise alone is
a stretch as the explanation for the test drop. Selection on val probably
explains little (config spread 0.0056, late-epoch spread about 0.004, against
a gap of 0.037; my inference). The model under-predicts mildly on both
splits (88% and 80% of the observed rate), but ROC-AUC fell as well, so
ranking is worse on test, not only the scale. The missing-rate gaps of up to
2.6 points are about 8 standard errors by my rough estimate, so test.csv
likely differs from train.csv; I did not show that this explains the drop.

>> Next: README, commit, then Task 03 on the same split.
