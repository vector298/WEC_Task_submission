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



## 2026-10-02: Experiment 2, fit the vectoriser on the val titles
Hypothesis: RT100Q and RT130 were not in the train vocabulary, so the
vectoriser ignored both and the two powerbank titles became identical.
If I fit on the val titles (no labels are used), both codes become real
tokens, so I expect the pair to drop below 1.0. I expect overall F1 to
rise slightly, because val-only words now count.
Setup: TfidfVectorizer(lowercase=True) fitted on val.title, same sweep.

Result: best val F1 0.7763 at threshold 0.40 (baseline 0.7632 at 0.45).
My first two lookups of the RT100Q/RT130 pair picked the wrong listings,
so I selected the highest-scoring of the 14 false matches. Its
similarity: baseline 1.000, single chars kept 1.000, fitted on val 0.906.

Observation: the hypothesis held. Both codes were missing from the train
vocabulary, so they were ignored; once the vectoriser saw the val titles
the pair dropped from 1.0 to 0.906. It is still far above the threshold,
because the titles share nearly every other word. All four RT100Q/RT130
pairs I printed fell under val-fitting. Overall F1 rose by 0.0131 and
the best threshold moved to 0.40. At test time I must fit on the test
titles in the same way

Next: experiment 3, character n-grams.


## 2026-10-02: Experiment 3, character n-grams
Hypothesis: character n-grams (2 to 4) will link titles that share no whole
word but share letters, such as truncations ("tablets" vs "tab") and
spelling variants ("mermed" vs "mermaid"), so recall should rise. They may
also add false matches between similar-looking brand names, and they will
not separate model-code variants like RT100Q and RT130.
Setup: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
lowercase=True), fitted on val.title like experiment 2 so the comparison is
fair. Same threshold sweep (0.20 to 0.95, steps of 0.05) on val.
Result: best val F1 0.7774 at threshold 0.45 (experiment 2, word TF-IDF
fitted on val: 0.7763 at 0.40). At a fixed threshold of 0.45, per-listing
precision / recall: word 0.878 / 0.774, character n-grams 0.839 / 0.810.
Observation: the direction of the hypothesis held at 0.45: recall rose by
0.036 and precision fell by 0.039, so F1 barely moved (+0.0011, too small
to call a gain from one split). I then plotted precision against recall
over all thresholds for both methods. The curves overlap from recall 0.4 to
0.85 and differ only above recall 0.9, where precision is under 0.6. So the
recall gain at 0.45 was movement along one curve: character chunks overlap
more easily, giving higher similarity scores, so a fixed cutoff lets more
pairs through. In aggregate I found no better matching. I did not inspect
individual pairs, so character features may still link specific truncation
cases, and I did not check whether they separate RT100Q from RT130.
Next: use word TF-IDF fitted on val as the final method (same performance,
simpler, scores explainable by shared words), run the precision, recall and
F1 threshold analysis, fix the threshold on val, and report on test once.

