## 2026-10-02: baseline, equal-weight score fusion (result line, update)
Result: with w = 0.5 on raw scores, best val F1 0.8521 at threshold 0.52
(precision 0.917, recall 0.856; threshold step 0.01). On the coarse grid:
0.8515 at 0.50. Text alone 0.7763, image alone 0.6898 on that grid.

## 2026-10-02: fine search over weight and threshold

>> Setup: weight on text 0.30 to 0.70 in steps of 0.05, threshold 0.35 to 0.65
in steps of 0.01, for raw and for centred image similarity; selection by
val F1.

>> Result: raw best w = 0.45, threshold 0.52, F1 0.8524 (precision 0.905,
recall 0.867). Centred best w = 0.55, threshold 0.44, F1 0.8494. Raw at
w = 0.50: 0.8521.

>> Observation: the top is flat (0.8524, 0.8521, 0.8496 for w = 0.45, 0.50,
0.55), so equal weights are within 0.0003 of the best and weight tuning
added nothing measurable. Raw and centred differ by 0.003, a tie on one

>> split; I selected raw as the simpler one.

>>Next: ablation.

## 2026-10-02: ablation, including phash

>> Setup: A text only, B image only (raw ResNet), C final fusion (w_text 0.45,
raw image, threshold 0.52), D = C plus phash similarity at weights 0.1, 0.2
and 0.3 (threshold per row chosen on val).

>>Result: A 0.7770, B 0.6917, C 0.8524, D 0.8525 / 0.8490 / 0.8395.

>>Observation: both modalities contribute (C is +0.075 over A and +0.161 over
B) and fusion raises precision and recall together. Phash gives +0.0001 at
weight 0.1 and hurts at higher weights, so it adds nothing; a possible
reason is that identical images already get nearly identical embeddings
(untested).

>>Next: test run, error analysis.

## 2026-10-02: final evaluation on test

>> Setup: weight 0.45 and thresholds fixed from val (A 0.43, B 0.79, C 0.52);
test titles fitted for TF-IDF; ResNet test embeddings; one run each.

>> Result: test floor 0.4401. A 0.7741, B 0.6800, C 0.8508.

>> Observation: the final system is 0.0016 below val and closes about 73% of
the floor-to-perfect gap on both splits (72.5% val, 73.4% test). A and B
reproduce Parts B and C exactly. Nothing was changed after seeing test.

>> Next: error analysis, README, report.
