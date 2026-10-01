# Shopee Part A: Dataset Exploration

## Contents
- `notebook.ipynb`: the exploration (run on Kaggle, where the images
  are mounted at /kaggle/input/competitions/shopee-product-matching/)
- `LOG.md`: timestamped notes from the work

## Data
Kaggle "Shopee - Price Match Guarantee" (slug shopee-product-matching).
Not committed. Only `train.csv` and `train_images/` were used.

## What the notebook covers
[one line per section: counts, group sizes, title length, duplicate
images and phash, hard examples, challenges, answers]

## Key findings
- 34,250 listings in 11,014 groups; 63% of groups are pairs, the
  largest has 51 listings.
- [duplicate images: 46 files and 147 phashes span more than one group]
- [titles: median 53 characters; 106 of 1,308 repeated titles span
  more than one group]
- [one finding from your hard examples, in your words]

## How to rerun
Open the notebook on Kaggle with the competition data attached and
use Restart and Run All. No GPU needed.

## Limitations
[anything you didn't check, e.g. the TF-IDF cosine comparison]
