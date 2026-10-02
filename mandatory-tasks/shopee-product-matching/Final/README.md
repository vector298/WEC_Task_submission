# Finale: Multimodal Product Matching

A product matching system that fuses title text and image similarity.

## Contents
- `notebook.ipynb`: split, components, baseline, experiments, ablation,
  error analysis, test (Kaggle)
- `results/`: figures and the results table
- `report.pdf`: the report (the sections below)
- `LOG.md`: timestamped experiment log

## Setup
Same split as Parts B and C (by `label_group`, 70/15/15 of groups, seed 42).
Metric: mean set-F1 per listing. Floors (predict only itself): val 0.4629,
test 0.4401. Weights and thresholds are chosen on val; test is run once per
configuration.

## 1. Approach
Two similarity matrices over the pool being searched are combined at the
score level:
- **Text:** lowercased word TF-IDF fitted on the pool's titles (no labels),
  cosine similarity.
- **Image:** frozen ImageNet ResNet50 (torchvision IMAGENET1K_V2), classifier
  removed, 2048-d L2-normalised features, cosine similarity. Embedded in this
  notebook (val and test images, 57 s on a CUDA GPU).
- **Fusion:** S = w * S_text + (1 - w) * S_image on raw scores. Two listings
  match when S >= threshold; each listing is matched to every listing in its
  pool at or above the threshold (itself included).
- **Final configuration:** w_text = 0.45, threshold 0.52, both chosen on val.

## 2. Experiments
| Experiment | Change | Best val F1 |
|---|---|---|
| Baseline | equal weights (0.5), raw scores | 0.8521 (threshold 0.52) |
| 1 | weight on text swept 0.0 to 1.0 (coarse), then 0.30 to 0.70 | 0.8524 (w 0.45, thr 0.52) |
| 2 | image similarity mean-centred (no labels), same sweeps | 0.8494 (w 0.55, thr 0.44) |

Coarse weight sweep (raw): w = 0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0 gives F1 0.6898,
0.7760, 0.8463, 0.8515, 0.8432, 0.8137, 0.7763. Single runs on one split.

## 3. Analysis
- Fusion beats both single signals: +0.075 over text (0.7770) and +0.160 over
  image (0.6917) on val, and it raises precision and recall together (text
  0.863 / 0.790; fusion 0.905 / 0.867).
- The top of the weight curve is flat: w = 0.45, 0.50 and 0.55 give 0.8524,
  0.8521 and 0.8496, so tuning the weight added nothing measurable over equal
  weights.
- Mean-centring the image similarity did not help (0.8494 against 0.8524, a tie
  on one split), so I kept the simpler raw version. My expectation that the
  mismatch between score scales (unrelated pairs average 0.004 for text and
  0.234 for image) would matter was not supported.
- By true group size, fusion is best in every bucket. Image alone drops from
  0.765 (pairs) to 0.486 (groups of 11+), text from 0.803 to 0.686, and fusion
  peaks at 0.882 for groups of 4 to 5. The 11+ bucket is noisy (at most about
  57 groups).

## 4. Error analysis (val, final system)
Over all val pairs the final system has 8,315 correct matches, 1,729 false
matches and 2,696 misses (pooled precision 0.828, recall 0.755), against 7,110 /
2,551 / 3,901 for text alone and 4,516 / 1,422 / 6,495 for image alone.
Against text, fusion removed 1,531 false matches (60%) and found 1,763 true
pairs text missed, but introduced 709 new false matches and lost 558.

| | Both modalities high | Text high only | Image high only | Both low |
|---|---|---|---|---|
| False matches (1,729) | 120 | 900 | 256 | 453 |
| Missed pairs (2,696) | 0 (by construction) | 558 | 59 | 2,079 |

| Case | Predicted | Correct | Why it failed | Possible fix (untested) |
|---|---|---|---|---|
| Ecolink 6 Watt vs 8 Watt | match (fused 1.00) | different products | titles differ in one digit, which the default tokenizer drops; identical photo | keep numeric tokens and penalise mismatching numbers |
| "Extra bubble wrap" listings | match (0.96, 0.92) | different groups | near-identical titles and photos of a generic add-on item; the listing does not show what separates the groups | none visible in the data |
| Ring light selfie LED | match (0.92) | different groups | titles identical apart from capitals, similar photos | finer image detail such as colour |
| Salsa Dynamatte lip cream | no match (0.47, 0.50) | same product | titles clearly name one product (text 0.55, 0.64) but photos differ (0.39); averaging pulled it under 0.52 | non-linear fusion that trusts a confident text score |
| Cream Malam vs Cream Siang Tabita | no match (0.47) | same group | titles contain opposite words (malam, night; siang, day): a bundle or label noise | none if the label is noisy |
| "Soes GG 2kg" vs "Juara Snack ... Sus Coklat GG 2kg" | no match (0.47) | same group | short title in a variant spelling, few shared tokens | synonym knowledge |

77% of missed pairs are low in both channels, so neither signal links them.

## 5. Ablation study (val; test in the last column where run)
| Configuration | Text | Image | Other | Threshold | Val F1 | Test F1 |
|---|---|---|---|---|---|---|
| A text only | yes | no | no | 0.43 | 0.7770 | 0.7741 |
| B image only (frozen ResNet50) | no | yes | no | 0.79 | 0.6917 | 0.6800 |
| **C text + image (final)** | yes | yes | no | 0.52 | **0.8524** | **0.8508** |
| D = C + phash, weight 0.1 | yes | yes | phash | 0.53 | 0.8525 | not run |
| D = C + phash, weight 0.2 | yes | yes | phash | 0.53 | 0.8490 | not run |
| D = C + phash, weight 0.3 | yes | yes | phash | 0.53 | 0.8395 | not run |

Test floor 0.4401. Each test number is one run with the val-chosen weight and
threshold. The final system closes about 73% of the gap between the floor and a
perfect score on both splits (72.5% val, 73.4% test). The perceptual hash adds
nothing (+0.0001 at weight 0.1, worse above).

## 6. Limitations
- The image encoder in the final system is the frozen ImageNet ResNet50. In
  Part C, fine-tuning its last block with an ArcFace loss raised image-only val
  F1 from 0.6917 to 0.7733 (test 0.68 to 0.7743). I did not use it in the
  fusion because I could not transfer its embeddings between notebooks in time,
  so I have not measured the fused result with it.
- One global weight and one global threshold serve groups of very different
  sizes; the largest groups score worst (see section 3).
- Variants separated only by a number or code, and identical photos in
  different groups, cannot be separated by these signals.
- Each result is one run on one split; differences of about 0.002 to 0.003
  (weights, centring) are within what I would expect from noise. Val and test
  are not equally hard (test floor 0.4401 against 0.4629).
- The text vectoriser is fitted on the pool being searched, which uses no
  labels but assumes all titles are available at indexing time.
- Some groups may be mislabelled or bundles (the Cream Malam and Siang pair);
  no model can fix those.

## 7. Future improvements
- Use the fine-tuned image encoder from Part C in the fusion and re-tune the
  weight and threshold on val.
- A numeric-token rule: when both titles contain numbers with units and none
  match, subtract a penalty (idea from the 6 Watt vs 8 Watt error).
- A learned, non-linear fusion of the two scores, to handle cases where one
  signal is confident and the other is not (Salsa Dynamatte).
- Group-size-aware matching (for example a relative threshold or a top-k
  rule), since one threshold serves pairs and groups of 51.
- CLIP's image-to-title similarity as a third signal.
- Repeating runs over several splits to measure variance.

## External resources
torchvision ResNet50 (ImageNet weights), scikit-learn (TF-IDF, cosine
similarity, GroupShuffleSplit). Kaggle Shopee - Price Match Guarantee data.
AI assistance: [state honestly how you used AI tools, for example for
explanation, debugging and drafting cells, and confirm you can explain every
part].

## How to rerun
Open the notebook on Kaggle with the Shopee competition data attached, GPU on,
then Restart and Run All.
