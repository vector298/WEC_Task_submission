## 2026-10-02: fine search over weight and threshold
>> Hypothesis:
- The coarse grid showed a flat top between w = 0.4 and 0.6, so a finer
  grid should land in that range
- the gain over the coarse best (0.8515) should be small
- raw and centred image similarity should end up close, in which case I
  keep the simpler raw version

>> Setup: weight on text 0.30 to 0.70 in steps of 0.05, threshold 0.35 to 0.65
in steps of 0.01, for raw and centred image similarity; selection by val F1.

>> Result: raw best w = 0.45, threshold 0.52, F1 0.8524 (precision 0.905,
recall 0.867); centred best 0.8494 at w = 0.55, threshold 0.44. Raw at
w = 0.50: 0.8521.

>> Observation: the top is flat (0.8524, 0.8521, 0.8496 for w = 0.45, 0.50,
0.55), so equal weights are within 0.0003 of the best and weight tuning
added nothing measurable. Raw and centred differ by 0.003, a tie on one

>> split; I selected raw as the simpler one.

>> Next: ablation.

## 2026-10-02: ablation, including phash
>> Hypothesis:
- text and image fused should beat each alone, so both contribute
- phash should add little: identical images already get nearly identical
  ResNet embeddings
- phash could add false matches, since 147 phashes span more than one group
  (Part A)

>> Setup: A text only, B image only (raw ResNet), C final fusion (w_text 0.45,
raw image, threshold 0.52), D = C plus phash at weights 0.1, 0.2, 0.3.

>> Result: A 0.7770, B 0.6917, C 0.8524, D 0.8525 / 0.8490 / 0.8395.

>> Observation: C is +0.075 over A and +0.161 over B and raises precision and
recall together. Phash gives +0.0001 at weight 0.1 and hurts at higher
weights, so it adds nothing; a possible reason is that identical images
already get nearly identical embeddings (untested).

>> Next: test run and error analysis.

## 2026-10-02: final evaluation on test
>> Hypothesis:
- test F1 should be slightly below val, as in Parts B and C, since test has
  larger groups and a lower floor
- the share of the floor-to-perfect gap closed should stay about the same
- weights and thresholds are fixed from val; nothing changes after seeing
  test

>> Setup: weight 0.45, thresholds fixed from val (A 0.43, B 0.79, C 0.52); one
run each.

>> Result: test floor 0.4401. A 0.7741, B 0.6800, C 0.8508.

>> Observation: C is 0.0016 below val and closes about 73% of the gap on both
splits (72.5% val, 73.4% test). A and B reproduce Parts B and C exactly.

## 2026-10-02: error analysis of the final system

>> Hypothesis:
- fusion should remove many of text's false matches (images separate
  products with different photos) and recover pairs text misses
- fusion should also add some new false matches, since adding two moderate
  scores can clear the bar
- errors should split into variants that look alike in both channels, and
  pairs neither channel links

>> Setup: val pool, unordered pairs, final fusion threshold 0.52; "high" means
the modality alone would match at its own val threshold (text 0.43, image
0.79).

>> Result: fused 8,315 correct / 1,729 false / 2,696 missed (text 7,110 /
2,551 / 3,901). Against text, fusion removed 1,531 false matches, added 709,
found 1,763 true pairs and lost 558. False matches: 120 both high, 900 text
high only, 256 image high only, 453 both low. Missed: 0 both high (by
construction), 558 text high only, 59 image high only, 2,079 both low.

>> Observation: over half of the false matches are driven by text alone; 77%
of misses are low in both channels, so fusion cannot recover them. Top false
matches: Ecolink 6W vs 8W (text 1.00, image 0.99) and extra-bubble-wrap
listings in different groups. Several misses have a confident text score and
a low image score (Salsa Dynamatte lip cream), which averaging dilutes.
Next: Experiment 3 (variant-aware penalty), then README and report.
