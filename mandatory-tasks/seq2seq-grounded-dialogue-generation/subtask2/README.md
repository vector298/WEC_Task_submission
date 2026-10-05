# Subtask 2: Attention and Decoding Strategies (English to French)

## Goal
Add attention to the subtask 1 LSTM encoder-decoder, retrain it under the same protocol, and compare decoding
strategies at inference on the same data.

## Method
**Attention model (Luong "general").** Identical to subtask 1 (embedding 256, hidden 512, 2 LSTM layers, dropout
0.3, vocabularies of 10,000, Adam 1e-3, batch 128, gradient clipping 1.0, scheduled teacher forcing 1.0 to 0.5,
10 epochs, seed 42, best checkpoint by greedy validation BLEU) except for attention, so differences can be
attributed to it. The encoder returns all hidden states; at every decoding step the decoder scores each
encoder state with a learned bilinear form of its current LSTM output, masks padding with minus infinity before
the softmax, forms the weighted context vector, and predicts from tanh(W [context ; decoder state]). The model
has 18,393,360 parameters (baseline 17,606,416 plus 786,944). Checks: attention weights sum to 1, padding gets
exactly zero weight, and the model memorises a small batch.

**Decoding strategies** (compared on the 890 validation sentences because beam search runs sentence by
sentence): greedy; beam search (widths 3 and 5, length-normalised); temperature sampling (0.7 and 1.0); top-k
(10); top-p (0.9); sampled rows are means over 3 seeds. Metrics: sacrebleu BLEU (lowercase, 13a), distinct-1 and
distinct-2, the share of outputs with a repeated trigram, output-to-reference length ratio, and time. The
strategy for the test run (beam 5, plus greedy) was chosen on validation.

## Hypotheses (stated before running)
- Attention removes the fixed-vector bottleneck, so the BLEU decline with source length should be shallower
  than the baseline's.
- Beam search should raise BLEU slightly over greedy and reduce repetition; wide beams may shorten outputs.
- Sampling should lower BLEU and raise diversity, with top-p and top-k closer to greedy than temperature 1.0.

## Results
**Training at an equal budget (10 epochs, same protocol):**

| Epoch | Baseline val BLEU | Attention val BLEU | Baseline val loss | Attention val loss |
|---|---|---|---|---|
| 1 | 2.47 | 6.74 | 3.538 | 3.037 |
| 2 | 3.52 | 11.80 | 3.237 | 2.465 |
| 3 | 4.38 | 14.68 | 3.095 | 2.284 |
| 4 | 5.01 | 16.38 | 3.040 | 2.212 |
| 5 | 5.42 | 17.13 | 2.997 | 2.195 |
| 6 | 5.37 | 17.02 | 2.985 | 2.201 |
| 7 | 5.82 | 17.53 | 2.951 | 2.183 |
| 8 | 5.87 | 17.19 | 2.923 | 2.168 |
| 9 | 6.37 | **18.18** | 2.906 | 2.154 |
| 10 | 5.85 | 17.53 | 2.885 | 2.167 |

Best epoch 9 for both; attention reaches 2.9 times the baseline's best validation BLEU and exceeds it after one
epoch. An epoch takes about 251 s against 218 s. Attention plateaus near epoch 9; the baseline was still
improving at epoch 10.

**Test (one run, epoch-9 checkpoints; sacrebleu lowercase 13a, 95% bootstrap interval from 1,000 resamples):**

| System | Test BLEU | 95% interval | Repeated trigram |
|---|---|---|---|
| Baseline (subtask 1), greedy | 9.41 | [9.17, 9.66] | 22.8% |
| Attention, greedy | 25.50 | [25.10, 25.87] | 8.6% |
| Attention, beam 5 | 27.72 | [27.33, 28.11] | 7.0% |

On the 8,463 rows whose English is not in the training data: 25.47 (greedy) and 27.68 (beam 5).

**Test BLEU by source length (tokens):**

| Source length | Sentences | Baseline | Attention greedy | Attention beam 5 |
|---|---|---|---|---|
| 1 to 10 | 2,325 | 19.66 | 28.85 | 29.97 |
| 11 to 20 | 3,226 | 12.66 | 26.51 | 28.45 |
| 21 to 30 | 1,682 | 8.40 | 25.46 | 27.75 |
| 31 to 50 | 1,117 | 5.71 | 24.18 | 26.53 |
| over 50 | 247 | 3.99 | 22.42 | 26.50 |

**Decoding strategies (validation, 890 sentences):**

| Strategy | Val BLEU | distinct-1 | distinct-2 | Repeated trigram % | Length ratio | Seconds |
|---|---|---|---|---|---|---|
| greedy | 18.18 | 0.191 | 0.604 | 10.6 | 0.969 | 0.4 |
| beam 3 | 19.89 | 0.196 | 0.617 | 7.2 | 0.978 | 23.1 |
| beam 5 | 20.21 | 0.197 | 0.622 | 6.9 | 0.979 | 24.2 |
| temperature 0.7 | 13.61 | 0.209 | 0.663 | 5.2 | 0.979 | 2.4 |
| temperature 1.0 | 8.13 | 0.250 | 0.763 | 1.5 | 0.999 | 2.0 |
| top-k 10 | 10.55 | 0.207 | 0.667 | 5.1 | 0.989 | 2.1 |
| top-p 0.9 | 10.31 | 0.228 | 0.719 | 2.7 | 0.991 | 3.5 |

(Greedy is batched and beam search runs one sentence at a time, so the time ratio partly reflects the
implementation.)

## Analysis
- **Attention and the bottleneck.** Attention improves BLEU at every length. The baseline's BLEU falls 15.7
  points from the shortest to the longest bucket, attention's 6.4 (greedy) and 3.5 (beam 5), and the gain grows
  with length (+9.2 points for 1 to 10 tokens, +18.4 for over 50, greedy). This supports the fixed-size context
  vector as a main cause of the baseline's long-sentence failure. It is association on identical sentences, not
  a controlled proof; the over-50 bucket has 247 sentences and exceeds the lengths seen in training.
- **Unknown words remain a limit.** BLEU is 20.75 for sources containing an `<unk>` against 32.26 without
  (baseline 6.46 and 13.96); attention does not fix a word-level vocabulary.
- **Decoding.** Beam search raises validation BLEU by 1.70 (width 3) and 2.03 (width 5) and cuts repetition
  from 10.6% to 6.9%; going from width 3 to 5 adds only 0.32. Sampling lowers BLEU by 4.6 to 10 points and
  raises diversity (distinct-2 from 0.604 to 0.663 and 0.763) and lowers repetition. With one reference per
  sentence, beam search is the right choice for translation; sampling would suit open-ended generation
  (subtask 3). Among the sampling strategies, temperature 0.7 keeps the most BLEU; top-p and top-k are about
  equal; I had expected top-p and top-k to beat low temperature, and they did not.
- **Examples.** Greedy: "elle `<unk>` en hiver et `<unk>` en été." (rare verbs become `<unk>`). "voici est la
  rivière `<unk>` dans groenland groenland." (a repeated word and a grammar error; beam 5 gives "c'est la
  rivière `<unk>` dans le groenland du groenland", still repeating). Sampled outputs hallucinate ("il se
  connecte dans hiver", "19ème nord").
- **Attention maps** (`results/s2_attention.png`; two validation sentences, greedy output). The weights follow a
  rough diagonal and align content words with their translations: rivière with river, dans with in, groenland with
  greenland, pétrole with oil, problème with problem, charbon with coal, sérieux with serious, and the final
  "." with ".". Where French order differs from English the weights reorder locally: for "the <unk> river" the
  output "la rivière <unk>" has "rivière" attending to "river" and the following "<unk>" attending to both
  "<unk>" and "river". Function words spread their attention over neighbours ("voici" over "this is the", then
  "est" on "the"; "le" and "pétrole" both on "oil"). Two failure modes are visible: the two generated "groenland"
  tokens both attend to the same source word "greenland" (the model has no coverage mechanism, which fits the
  repeated word), and the generated "<unk>" attends to the source "<unk>" and "river", so the rare word's
  identity is already lost at the input. Attention weights show where the model looked, not necessarily why it
  produced a word. 

## Limitations
One seed per model; the baseline was still improving at the end of its budget while attention had plateaued,
so a longer baseline run might narrow the gap (not measured); decoding strategies are compared on a small
validation set (890 sentences) where differences of a few tenths are within noise; test BLEU is higher than
validation BLEU for both models (probably a harder validation set; not investigated); BLEU penalises valid
paraphrases; no coverage mechanism, so repeated words remain; word-level vocabulary; noisy corpus
(misaligned pairs, about 5% `<unk>`); beam search was evaluated per sentence, not batched.

## Reproduce
Kaggle, Tesla T4 (one GPU), Internet on. Python 3.13.15, torch 2.11.0+cu128, sacrebleu 2.6.0. Seed 42. The
attention cells (11 to 17) of `notebook.ipynb` reuse the data and vocabularies built by cells 1 to 5 (the
subtask 1 part of the same notebook).
