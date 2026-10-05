## 2026-10-05: subtask 3 first training run (12-epoch cap) and retrieval baseline

Setup: Kaggle, one GPU; grounded and ungrounded models (embedding 256, hidden 256, dropout 0.3, Adam 1e-3,
batch 128, clipping 1.0, teacher forcing 1.0 down to 0.6), early stopping on validation loss, at most 12
epochs; TF-IDF retrieval baseline over 7,855 training histories (9,102 features).
Result: both models trained for the full 12 epochs with validation loss falling at every epoch: grounded
6.185 to 5.446 (perplexity 232), ungrounded 6.120 to 5.283 (perplexity 197); train loss 5.726 and 5.448 at
epoch 12; about 9 s and 5 s per epoch. The retrieval baseline built in 1 s.
Observation: the 12-epoch cap was too low (both models underfitted; I had predicted early overfitting). At
equal budget the ungrounded model has the lower validation loss, contrary to my hypothesis; a plausible but
untested reason is that the grounded model has more to learn and trains more slowly. Both models are
retrained with an 80-epoch cap and the same early stopping before the test set is used. The test set has
not been evaluated yet.
Next: retrain with the longer cap, then the test evaluation once.

## 2026-10-05: subtask 3 longer training, test evaluation, document-swap test

>> Setup: both models retrained with an 80-epoch cap and early stopping (patience 2) on validation loss; test set
(933 examples from 27 conversations) evaluated once with greedy decoding (up to 30 tokens); metrics BLEU,
ROUGE-L, distinct-1/2, length and a grounding rate; document-swap test (documents shuffled across test
examples; 94% received a different movie); TF-IDF retrieval baseline. Kaggle, one GPU.

>> Result: both models stopped after about 22 to 23 epochs; validation loss flattened near 5.1 to 5.2 while train
loss kept falling (about 4.5 grounded, 4.8 ungrounded); the ungrounded validation loss was slightly lower
(about 5.12 against 5.16). Test: retrieval BLEU 0.42, ROUGE-L 0.064, distinct-2 0.447; ungrounded BLEU 0.72,
ROUGE-L 0.108, distinct-2 0.055, length 6.7; grounded BLEU 0.98, ROUGE-L 0.107, distinct-2 0.072, length 8.2;
references distinct-2 0.684, length 12.6. Grounding rate: grounded 0.282, ungrounded 0.000, retrieval 0.040,
references 0.100. Swap test: grounding to the true document 0.282 falls to 0.047 and grounding to the swapped
document is 0.391; 91.7% of outputs changed; ROUGE-L 0.107 to 0.105, BLEU 0.98 to 0.82.

>> Observation: both generators overfit after about epoch 17 and produce short, generic, repetitive replies
(distinct-2 about a tenth of the references'). The grounded model measurably uses the document (its content
words follow the swapped document) but this does not improve ROUGE-L, BLEU or validation loss over the
ungrounded model; it appears to copy names and titles. Retrieval is more diverse but off-topic and has the
lowest overlap. BLEU is below 1 for every system, so ROUGE-L is the more usable metric. My first
code-mixing split put 810 of 933 examples in one group and 35 in the other, so it is recomputed with terciles.

>> Next: extra diagnostics (best epochs, terciles, grounding counts), README.
