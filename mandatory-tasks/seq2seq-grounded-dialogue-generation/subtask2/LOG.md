## 2026-10-05: seq2seq subtask 2 full results

>> Setup: Luong "general" attention added to the subtask 1 model (identical data, vocabularies, embedding 256,
hidden 512, 2 layers, dropout 0.3, Adam 1e-3, batch 128, clipping 1.0, teacher forcing 1.0 down to 0.5, seed
42, 10 epochs), 18,393,360 parameters, Tesla T4, about 251 s per epoch. Best checkpoint by validation BLEU
(epoch 9). Decoding strategies compared on the 890 validation sentences; test run once (greedy and beam 5).

>> Result: validation BLEU by epoch 6.74, 11.80, 14.68, 16.38, 17.13, 17.02, 17.53, 17.19, 18.18, 17.53 against
the baseline's 2.47 to 6.37 (best epoch 9 for both). Test BLEU: baseline 9.41 [9.17, 9.66]; attention greedy
25.50 [25.10, 25.87]; attention beam 5 27.72 [27.33, 28.11]; repeated trigrams 22.8% / 8.6% / 7.0%. Test BLEU
by source length (baseline / attention greedy / beam 5): 19.66 / 28.85 / 29.97 (1 to 10), 12.66 / 26.51 /
28.45 (11 to 20), 8.40 / 25.46 / 27.75 (21 to 30), 5.71 / 24.18 / 26.53 (31 to 50), 3.99 / 22.42 / 26.50
(over 50). Sources with an <unk>: attention BLEU 20.75 with, 32.26 without. Decoding (validation BLEU):
greedy 18.18, beam 3 19.89, beam 5 20.21, temperature 0.7 13.61, temperature 1.0 8.13, top-k 10 10.55,
top-p 0.9 10.31; repeated trigram 10.6% (greedy), 6.9% (beam 5), 1.5 to 5.2% (sampling); distinct-2 0.604
(greedy), 0.763 (temperature 1.0).

>> Observation: at an equal 10-epoch budget attention is 2.9 times the baseline on validation BLEU and 2.7
times (greedy) on test. The BLEU decline with source length is 15.7 points for the baseline against 6.4
(greedy) and 3.5 (beam 5) for attention, consistent with the fixed-vector bottleneck being a main cause.
Attention plateaued near epoch 9; the baseline was still improving. Beam search helps BLEU and repetition;
sampling lowers BLEU and raises diversity. <unk> still costs about 11 BLEU points. Test BLEU is higher than
validation BLEU for both models (a harder small validation set; not investigated). An earlier interrupted
run reproduced epochs 1 to 5 exactly, so training is deterministic with this seed.

>> Next: attention heatmaps, README, commit.
