# Subtask 1: LSTM Encoder-Decoder (English to French)

## Contents
- `notebook.ipynb`: data preparation, model, training, test evaluation, failure analysis (Kaggle, Tesla T4)
- `results/`: training curves
- `LOG.md`: timestamped log

## Data
Provided English-French CSVs (TED-talk style): train 232,825 pairs, validation 890, test 8,597; no
missing sides; exact duplicate pairs 2,054 / 7 / 71. Cleaning (all splits): whitespace collapsed,
apostrophes normalised, French stage directions (--Rires--, --Faux sanglot--, --Applaudissements--)
removed by a whitelist (14 training segments). English is untouched. An earlier, looser rule deleted
real words between em-dashes and was replaced. Training filter only: both sides 1 to 50 tokens and
length ratio within 1/3 to 3 (220,442 of 232,825 pairs kept; 12,320 too long, 65 extreme ratio);
validation and test are never filtered. The corpus is noisy (misaligned pairs such as a French line
of "-" for an English sentence). Leakage: 12 of 890 validation and 96 of 8,597 test pairs also occur
in train (134 test English sentences, 1.6%).

## Preprocessing
Word-level tokenizer (lowercase, punctuation split). French is split after apostrophes (l' idee),
chosen over keeping them attached because it lowered the validation `<unk>` rate from 7.06% to 5.88%.
Vocabularies of 10,000 words per language (minimum frequency 2) plus `<pad>`, `<sos>`, `<eos>`,
`<unk>`. Token `<unk>` rate: train EN 3.44%, FR 4.48%; validation EN 4.20%, FR 5.41%.

## Model and training
LSTM encoder-decoder from scratch: embedding 256, hidden 512, 2 layers, dropout 0.3 (17,606,416
parameters). The encoder's final state initialises the decoder (a single fixed context vector);
padding is handled by packing and masking. Adam 1e-3, batch 128 (length-bucketed), gradient clipping
1.0, scheduled teacher forcing from 1.0 down to 0.5 (0.1 per epoch), at most 10 epochs, best
checkpoint by greedy validation BLEU. About 218 s per epoch on one Tesla T4.

| Epoch | Train loss | Val loss | Val BLEU |
|---|---|---|---|
| 1 | 4.083 | 3.538 | 2.47 |
| 5 | 3.206 | 2.997 | 5.42 |
| 9 | 3.006 | 2.906 | **6.37** |
| 10 | 2.942 | 2.885 | 5.85 |

The model is undertrained, not overfitted: validation loss fell at every epoch and the run stopped at
the epoch cap. Validation BLEU is noisy (about 0.5 between epochs on 890 sentences).

## Test results (greedy, one run)
sacrebleu, lowercase, 13a tokenisation (nrefs:1|case:lc|eff:no|tok:13a|smooth:exp|version:2.6.0).
**Test BLEU 9.41, 95% bootstrap interval [9.17, 9.66]**; 9.37 on the 8,463 rows whose English is not
in train (51.01 on the 134 overlapping rows).

| Source length | Sentences | BLEU | Output/reference length |
|---|---|---|---|
| 1 to 10 | 2,325 | 19.67 | 0.90 |
| 11 to 20 | 3,226 | 12.66 | 0.91 |
| 21 to 30 | 1,682 | 8.40 | 0.90 |
| 31 to 50 | 1,117 | 5.71 | 0.87 |
| over 50 | 247 | 3.99 | 0.74 |

## Failure analysis
- **Long sentences:** BLEU falls steadily with source length (association; the attention model in
  subtask 2 tests whether the fixed context vector is the cause). Sentences over 50 tokens were
  excluded from training.
- **Unknown words:** 43.6% of test sources contain an `<unk>`; BLEU 6.46 with against 13.96 without
  (within 11 to 20 tokens: 9.22 against 15.72). 62.8% of outputs contain `<unk>`.
- **Degeneration:** 22.8% of outputs repeat a trigram (references: 3.8%); outputs start correctly and
  then loop ("et, et, et", "ce que ce que") or emit `<unk>` and commas.
- **BLEU limits:** a fluent paraphrase ("et c'est ce qu'ils ont fait" for "Et elles se sont tues.")
  scores badly against one reference.
- Likely causes: the single fixed-size context vector, a word-level vocabulary, exposure bias with
  greedy decoding, a short training budget, and noisy alignment.

## Limitations
One run, one seed; validation BLEU (6.37) is 3 points below test BLEU (9.41), not investigated;
undertrained; no subword vocabulary; BLEU only.

## Reproduce
Kaggle, Tesla T4, one GPU. Python 3.13.15, torch 2.11.0+cu128, sacrebleu 2.6.0. Seeds: 42.
