## 2026-10-04: seq2seq subtask 1 setup and checks

>> Setup: Kaggle, Tesla T4, python 3.13.15, torch 2.11.0+cu128, sacrebleu 2.6.0. Data: the
provided English-to-French CSVs (train 232,826 lines, val 891, test 8,598, header
included). Word-level tokenizer (lowercase, punctuation split, apostrophes kept inside
words, Devanagari-safe), vocabulary with <pad>, <sos>, <eos>, <unk>. Model: LSTM
encoder-decoder from scratch, final encoder state as context vector, teacher forcing,
padding handled by packing and masking. Sanity checks on random data.

>> Result: parameter count 455,656 as calculated; logits (8, 8, 1000); loss on one
memorised random batch fell from 6.919 (ln V = 6.908) to 0.0026 in 300 steps; BLEU
sanity check 53.7 as calculated.

>> Observation: the pipeline works end to end before touching the real data. The corpus is
noisy (misaligned pairs, stage-direction markers, duplicated sentences, double spaces),
and the val set is small (about 890 pairs), so val BLEU will be noisy; the test BLEU is
reported once.

>> Next: data cleaning and statistics, vocabulary and batching, then training.

## 2026-10-04: seq2seq subtask 1 cleaning correction and leakage check

>> Setup: rerun of the data preparation with stage-direction removal restricted to French and
to capitalised segments of at most 30 characters; leakage check between train and val/test.

>> Result: 96 French training segments removed; the most common were --Rires-- (9), --Faux
sanglot-- (2), --Applaudissements-- (2), but also named asides such as -- Docteur Moskowitz --
and -- Dr. Holly --. English rows with (Laughter)-style tags: 0 in every split. Pairs also in

>> train: val 12 of 890, test 96 of 8,597 (English sentences: 14 and 134). Training pairs kept
220,445 of 232,825. French validation <unk> rate 7.06% (a) against 5.88% (b, chosen).
>> Observation: the capital-letter rule still deleted real French asides (the impact is about
0.04% of the training rows, but the description would be wrong), so it was replaced by a
whitelist of the stage-direction words actually seen. Leakage is about 1% of the test pairs;
test BLEU will be reported on all rows and on the rows whose English sentence is not in train.

>> Next: full training run.

## 2026-10-04: seq2seq subtask 1 stage-direction rule corrected again

>> Setup: rerun of the data preparation with the French stage-direction rule replaced by a
whitelist (Rires, Applaudissements, Faux sanglot, Musique, Vidéo, plus one following word).

>> Result: 14 French training segments removed, all genuine (--Rires-- 9, --Faux sanglot-- 2,
--Applaudissements-- 2, --Rires Applaudissements-- 1). Pairs kept 220,442 of 232,825.
French validation <unk> rate 7.06% (a) against 5.88% (b, chosen); English 4.39%.

>> Observation: the capital-letter rule had removed named em-dash asides (for example
-- Docteur Moskowitz --); the whitelist keeps them. English sources are left untouched.

>> Next: full training run.

## 2026-10-04: seq2seq subtask 1 full training run

>> Setup: Kaggle Tesla T4, one GPU. LSTM encoder-decoder from scratch (embedding 256, hidden
512, 2 layers, dropout 0.3, 17,606,416 parameters), word-level vocabularies of 10,000, Adam
1e-3, batch 128 with length bucketing, gradient clipping 1.0, scheduled teacher forcing 1.0
down to 0.5 by 0.1 per epoch, at most 10 epochs, early stopping on greedy validation BLEU
(patience 2), best checkpoint kept.

>> Result: validation BLEU 2.47, 3.52, 4.38, 5.01, 5.42, 5.37, 5.82, 5.87, 6.37, 5.85 over the
ten epochs; best 6.37 at epoch 9. Validation loss fell every epoch from 3.538 to 2.885; train
loss 4.083 to 2.942 (higher than validation loss from epoch 2 on). About 218 s per epoch,
about 36 minutes in total.

>> Observation: no overfitting within the budget; the run stopped at the epoch cap with
validation loss still falling, so the model is undertrained. Train loss is above validation
loss because it includes dropout and partly self-fed inputs, and it rose slightly while the
teacher-forcing ratio fell. Validation BLEU is noisy (swings of about 0.5 between epochs on
890 sentences), so the epoch-9 versus epoch-10 choice is within noise. Subtask 2 will use the
same 10-epoch budget so the attention comparison is like for like.

>> Next: test BLEU once, BLEU by length, <unk> effect, examples.
