Git w# Log: Shopee Part B (text matching)

## 2026-10-01: split, metric and TF-IDF baseline
Hypothesis: I did not record a prediction before running the sweep.
Setup: group split 70/15/15 (seed 42); TF-IDF fitted on train; cosine
similarity on val; thresholds 0.20 to 0.95 in steps of 0.05.
Result: best F1 0.7632 at threshold 0.45; floor 0.4629.
Observation: [what surprised you or didn't]
Next: precision and recall per threshold, error analysis, a second
representation.


## 2026-10-01: Experiment 1, keep single-character tokens
Hypothesis: the default tokenizer drops tokens shorter than two
characters, so "8 Watt" and "6 Watt" in the Ecolink titles became
identical vectors and scored 1.0. If I keep single characters, I expect
that pair to drop below 1.0 (roughly 0.85 to 0.95) but stay above the
0.45 threshold, because one token carries a small share of a long
title's weight. I expect overall F1 to change only slightly.
Setup: TfidfVectorizer with token_pattern r"(?u)\b\w+\b", fitted on the
train split, same val threshold sweep (0.20 to 0.95, steps of 0.05) as
the baseline.
>> Result: best val F1 0.7650 at threshold 0.45 (baseline 0.7632 at 0.45).
Ecolink 8W vs 6W similarity went from 1.0 to 0.931.

>> Observation: the prediction held. The pair dropped below 1.0, inside my
predicted 0.85 to 0.95 range, but stayed far above the 0.45 threshold, so
it is still a false match. Separating it would need a threshold above
0.93, and the sweep shows F1 near the floor at 0.90 and 0.95. The overall
gain (+0.0018) is small enough that I would not call it a real
improvement from a single split.

>> Next:  experiment 2, fitting the vectoriser on the val titles, to test why
RT100Q and RT130 scored 1.0.


