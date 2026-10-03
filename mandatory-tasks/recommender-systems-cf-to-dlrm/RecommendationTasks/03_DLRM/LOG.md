## 2026-10-03: Task 03, first DLRM run
>> Hypothesis:
- the dot products give the top MLP explicit pairwise cross-features that the
  Task 02 network had to find implicitly, so DLRM might do better; with only
  1,024 training clicks they could also add noise
- the top MLP input grows to d + 351, so it may overfit sooner than Task 02's
  model
- architecture differences in Task 02 were about the size of the seed spread,
  so a single run needs three seeds before I believe it
>> Setup: identical split and encoding to Task 02 (fingerprint reproduced and
asserted), Colab CPU. DLRM written from scratch: bottom MLP (13 -> 32 -> 8,
ReLU), 26 embedding tables of length 8 (normal init, std 1/sqrt(8)), the 27
vectors interacted by all 351 pairwise dot products, concatenated with the
dense vector, top MLP (128, 64) -> 1 logit. Same protocol as Task 02: Adam
1e-3, batch 512, early stopping on val PR-AUC (patience 3), seed 42.

>> Result: 134,409 parameters, 0.538 MB, 6 s. Best epoch 3: val ROC-AUC 0.6756,
PR-AUC 0.0630, log loss 0.1371, F1 0.1092, precision 0.0663, recall 0.3086,
accuracy 0.8389. Train loss fell from 0.130 to 0.109 (epochs 3 to 6) while val
loss rose from 0.1371 to 0.1475. Task 02, same seed and hardware: baseline
PR-AUC 0.1049, selected model 0.1014.

>> Observation: the first DLRM is clearly worse than Task 02 (PR-AUC about 0.04
lower, far more than the seed spread) and overfits within three epochs; by
epochs 5 and 6 its val log loss (0.1447, 0.1475) is above the constant-rate
predictor (0.1416). Cause not diagnosed; candidates: too much capacity for
1,024 clicks, my embedding init differing from Task 02's default, an
untuned setup.

>> Next: three-seed sweep with smaller sizes, dropout, an init check and a
no-interaction ablation.


## 2026-10-03: Task 03 selected DLRM, test and uncertainty
Hypothesis:
 - well previously , the model had overfitted as we could clearly see that the validation loss increased whereas the training loss significantly decreased.
 
Setup: selected configuration D5 (d = 16, bottom (32), top (64), dropout 0.3,
dot-product interactions) retrained with seed 42; threshold chosen on val;
test run once; 1,000-resample bootstrap on val and test. Kaggle, CPU only
(python 3.13.15, torch 2.11.0+cpu, pandas 2.3.3).
Result: best epoch 5; 183,185 parameters, 0.733 MB, 8 s. Val ROC-AUC 0.7124,
PR-AUC 0.0831, log loss 0.1338, F1 0.1463 at threshold 0.09. Test (one run):
ROC-AUC 0.681 [0.655, 0.708], PR-AUC 0.0665 [0.0557, 0.0846], log loss 0.1427,
F1 0.1053, precision 0.0762, recall 0.1701. Task 02 selected model on test:
ROC-AUC 0.6676, PR-AUC 0.0647, log loss 0.1441, F1 0.0845. Mean predicted
probability 0.0342 (val), 0.0332 (test) against observed 0.0320, 0.0335.
Flagged share at the threshold 7.7% (val), 7.5% (test).
Observation: the seed-42 run matches the 3-seed mean (0.0831 against 0.0836).
It still overfits (val loss up from epoch 4, above the constant predictor by
epoch 7) and is overconfident in the top probability bin. On test DLRM is
slightly ahead of Task 02 on every measure, but the intervals overlap
heavily, so there is no measurable difference; on val Task 02 led by 0.018
(0.020 on 3-seed means), and the ablation showed the interaction layer hurt in
all six comparisons. The ordering flips between val and test, which I treat as
noise from 256 and 335 clicks.
Next: README, commit, then Task 01.
