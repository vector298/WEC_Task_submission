# Shopee Part B: Text-Based Product Matching

Matching Shopee listings that describe the same product, using titles only.

## Contents
- `notebook.ipynb`: split, metric, baseline, four experiments, threshold
  analysis, error analysis, conclusions (run on Kaggle, no GPU needed
  except that the embedding run is faster with one)
- `results/`: plots and the experiment table
- `LOG.md`: timestamped log of each experiment, with hypotheses and results

## Data and setup
Kaggle "Shopee - Price Match Guarantee" (`train.csv` only; not committed).
The data is split by `label_group` (70/15/15 of groups, seed 42), so no
product appears in two splits.

| Split | Listings | Groups |
|---|---|---|
| train | 23,724 | 7,709 |
| val | 5,115 | 1,652 |
| test | 5,411 | 1,653 |

Each split is its own search pool: a listing looks for partners inside its
own split. Models are chosen on val; test is used once, at the end.

## Metric
Mean set-based F1 per listing: for listing i, with predicted set P and true
group T, F1 = 2|P ∩ T| / (|P| + |T|), averaged over listings. P always
contains the listing itself.

| Baseline | Val | Test |
|---|---|---|
| Predict everything | 0.0021 | not run |
| Predict only itself (floor) | 0.4629 | 0.4401 |
| Perfect | 1.0000 | 1.0000 |

The floor is high because 63% of groups are pairs, so predicting no
matches already gets about half the credit. Results are read against it.

## Experiments
Pipeline: title, text representation, cosine similarity, threshold.

| # | Representation | Fit on | Best threshold | Val F1 | Test F1 |
|---|---|---|---|---|---|
| Baseline | word TF-IDF | train | 0.45 | 0.7632 | not run |
| 1 | word TF-IDF, single-character tokens kept | train | 0.45 | 0.7650 | not run |
| 2 (final) | word TF-IDF | val / test titles | 0.43 | 0.7770 | 0.7741 |
| 3 | character n-grams (2 to 4, char_wb) | val titles | 0.45 | 0.7774 | not run |
| 4 | multilingual MiniLM sentence embeddings | none (pretrained) | 0.75 | 0.6424 | not run |

Each experiment is one run on one split, so differences of a few thousandths
(for example 0.7632 vs 0.7650) are within what I would expect from noise.
Only the final method was run on test.

## Key findings
- Word TF-IDF with cosine similarity closes about 58% of the gap between the
  floor and a perfect score on val, and 60% on test (0.7741 against a floor
  of 0.4401).
- Fitting the vectoriser on the titles being searched (no labels used)
  added 0.013 val F1. Vocabulary that exists only in the pool, such as the
  model codes RT100Q and RT130, was otherwise ignored.
- The default tokenizer drops single-character tokens, so "8 Watt" and "6
  Watt" became identical titles (similarity 1.0). Keeping them lowered it to
  0.931, still far above any threshold that works.
- Character n-grams did not help in aggregate: their precision-recall curve
  overlaps word TF-IDF's from recall 0.4 to 0.85.
- A multilingual sentence-embedding model did worse (val F1 0.6424). On a
  random sample of 400,000 val pairs (333 true pairs), unrelated pairs
  scored a mean of 0.219 with embeddings against 0.004 with TF-IDF, so
  different products sit close together under that model.
- Variants that differ in one number or code, identical titles across
  groups, synonyms, and titles with no shared word are the remaining errors.

## Threshold analysis (word TF-IDF, fit on val)
| Threshold | Precision | Recall | Set-F1 |
|---|---|---|---|
| 0.20 | 0.403 | 0.937 | 0.5084 |
| 0.40 | 0.834 | 0.815 | 0.7763 |
| 0.45 | 0.878 | 0.774 | 0.7748 |
| 0.95 | 0.998 | 0.370 | 0.5062 |

Precision and recall cross at about 0.39. F1 is flat between 0.40 and 0.45,
so the exact threshold matters little. The final threshold, 0.43, is the F1
maximum on val over a sweep from 0.30 to 0.50 in steps of 0.01 (val F1
0.7770). At 0.95 the recall of 0.370 is only about 0.05 above the 0.323 a
listing gets by predicting only itself.

## Not done
- A stemmer or lemmatiser (mixed Indonesian and English titles, and a risk of
  mangling model codes)
- Fine-tuning an embedding model, or trying other embedding models
- Repeating runs over several splits to measure variance

## Limitations
- The vectoriser is fitted on the pool being searched, which uses no labels
  but assumes all titles are available at indexing time.
- Val and test are not equally hard (test has about 3.27 listings per group
  against 3.10 for val, and a lower floor).

## How to rerun
Open `notebook.ipynb` on Kaggle with the Shopee competition data attached,
turn Internet on (for the embedding model), then Restart and Run All.
