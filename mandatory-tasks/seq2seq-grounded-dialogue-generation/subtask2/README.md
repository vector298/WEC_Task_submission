# Subtask 2: Attention and Decoding Strategies (English to French)

## Goal
Extend the subtask 1 LSTM encoder-decoder with attention, re-train it under the same protocol, and
compare decoding strategies at inference on the same dataset.

## Status
Implementation written in `notebook.ipynb`. STATUS: [replace with one line: "training and evaluation
complete, results below" OR "training and evaluation not completed before the deadline; no attention
results are claimed here"].

## Baseline to beat (subtask 1, measured)
Same data, vocabularies, hyperparameters, seed and budget.

| Model | Best validation BLEU (greedy) | Test BLEU (greedy) |
|---|---|---|
| Subtask 1, no attention | 6.37 (epoch 9 of 10) | 9.41, 95% interval [9.17, 9.66] |

Baseline test BLEU by source length (tokens): 19.67 (1 to 10), 12.66 (11 to 20), 8.40 (21 to 30),
5.71 (31 to 50), 3.99 (over 50). Baseline validation BLEU by epoch: 2.47, 3.52, 4.38, 5.01, 5.42,
5.37, 5.82, 5.87, 6.37, 5.85. The baseline was still improving when its 10-epoch budget ended, and
22.8% of its outputs repeat a trigram (references: 3.8%).

## Method
**Attention model (Luong "general").** Everything is identical to subtask 1 (embedding 256, hidden 512,
2 LSTM layers, dropout 0.3, vocabularies of 10,000, Adam 1e-3, batch 128, gradient clipping 1.0,
scheduled teacher forcing from 1.0 to 0.5, at most 10 epochs, early stopping on greedy validation
BLEU, seed 42) except for attention, so a difference can be attributed to it:
- the encoder returns all hidden states, not only the final one (the final state still initialises
  the decoder);
- at each decoding step the decoder scores every encoder state with a learned bilinear form of its
  current LSTM output, masks padding positions with minus infinity before the softmax, forms the
  weighted context vector, and predicts from tanh(W [context ; decoder state]);
- by calculation the model has 17,606,416 + 262,144 + 524,800 = 18,393,360 parameters (the notebook
  prints the actual count); checks in the notebook confirm that attention weights sum to 1, that
  padding receives exactly zero weight, and that the model can memorise a small batch.

**Decoding strategies** (all on the validation set of 890 sentences, because beam search is run
sentence by sentence): greedy; beam search with widths 3 and 5 and length normalisation (score divided
by length); temperature sampling (0.7 and 1.0); top-k sampling (k = 10); top-p sampling (p = 0.9).
Sampled results are the mean of 3 seeds. Metrics: BLEU (sacrebleu, lowercase, 13a), distinct-1 and
distinct-2 (unique n-grams over all n-grams, a diversity measure), the share of outputs with a
repeated trigram, output-to-reference length ratio, and decoding time.

## Hypotheses 
- Attention removes the fixed-vector bottleneck, so the BLEU decline with source length should become
  shallower than the baseline's (19.67 down to 3.99).
- Beam search should raise BLEU slightly over greedy and reduce repetition; wide beams may shorten
  outputs.
- Sampling should lower BLEU and raise diversity (distinct-n), with top-p and top-k closer to greedy
  than unrestricted temperature 1.0.



If attention training was stopped early, compare with the baseline at the same epoch (baseline
validation BLEU: epoch 1 2.47, epoch 2 3.52, epoch 3 4.38, epoch 4 5.01, epoch 5 5.42).



Example outputs for three validation sentences under every strategy are printed by the last cell of
the notebook.

## Limitations
One seed per model; the baseline was undertrained (validation loss still falling at the epoch cap),
so both models were limited by the same budget; the decoding comparison uses the small validation
set (890 sentences), so BLEU differences of a few tenths are within noise; BLEU penalises valid
paraphrases; the corpus is noisy (misaligned pairs, stage directions, about 5% `<unk>`).

## Reproduce
Kaggle, Tesla T4 (one GPU), Internet on. Python 3.13.15, torch 2.11.0+cu128, sacrebleu 2.6.0.
Seed 42. Run the subtask 1 notebook cells for data and vocabularies first (the attention cells reuse
them).
