Git w# Log: Shopee Part B (text matching)

## 2026-10-01: split, metric and TF-IDF baseline
Hypothesis: I did not record a prediction before running the sweep.
Setup: group split 70/15/15 (seed 42); TF-IDF fitted on train; cosine
similarity on val; thresholds 0.20 to 0.95 in steps of 0.05.
Result: best F1 0.7632 at threshold 0.45; floor 0.4629.
Observation: [what surprised you or didn't]
Next: precision and recall per threshold, error analysis, a second
representation.

