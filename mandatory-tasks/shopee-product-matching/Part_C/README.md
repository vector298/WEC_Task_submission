# Shopee Part C: Image-Based Product Matching

Matching listings that are the same product using images only.

## Contents
- `notebook.ipynb`: preprocessing, embeddings, experiments, phash reference,
  nearest-neighbour examples, error analysis (run on Kaggle with a GPU)
- `results/`: plots, error-analysis figures and the results table
- `LOG.md`: timestamped experiment log

## Setup
Same split as Part B (by `label_group`, 70/15/15 of groups, seed 42), so
results are comparable. Metric: mean set-F1 per listing. Floors (predict
only itself): val 0.4629, test 0.4401. Only val and test images are embedded,
because the models are frozen and pretrained.

## Method
Image, then embedding, then cosine similarity, then a threshold.
Embeddings are L2-normalised, so a dot product equals cosine similarity.
- **ResNet50** (ImageNet weights): classifier removed, 2048-d pooled
  features, torchvision preprocessing.
- **CLIP ViT-B/32** image encoder: 512-d.
- **Mean-centring** (no labels): subtract the pool's mean embedding, then
  re-normalise.
- **Perceptual hash** (non-learned reference): Hamming distance on the 64-bit
  `image_phash`.

## Results (val; threshold chosen on val)
| Experiment | Representation | Dim | Similarity | Threshold | Val F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Reference | image_phash | 64 bits | Hamming | max distance 10 | 0.6074 | n/a | n/a |
| Baseline (selected) | ResNet50 | 2048 | cosine | 0.790 | 0.6917 | 0.938 | 0.626 |
| Exp 1 | CLIP ViT-B/32 | 512 | cosine | 0.830 | 0.6616 | 0.918 | 0.602 |
| Exp 2a | ResNet50, centred | 2048 | cosine | 0.735 | 0.6900 | 0.950 | 0.615 |
| Exp 2b | CLIP, centred | 512 | cosine | 0.635 | 0.6765 | 0.894 | 0.641 |

Test (single run, ResNet50 raw, threshold 0.79): F1 0.68 against a test
floor of 0.4401. Word TF-IDF from Part B reached 0.777 on val and 0.774 on
test. Each configuration is one run on one split; differences of about 0.002
(ResNet raw against centred) are treated as ties.

## Key findings
- Learned embeddings add about 0.08 F1 over the perceptual hash and about
  0.23 over the floor on val; images alone close about 43% of the gap to a
  perfect score, text alone about 58%.
- ResNet50 beat CLIP in aggregate (its precision-recall curve is above
  CLIP's from recall about 0.5 to 0.9). A possible reason is that CLIP's
  training aligns images with captions, which image-only comparison does not
  use; untested.
- Mean-centring helped CLIP (+0.015) and not ResNet50.
- The image method operates at high precision and lower recall (0.94 and
  0.63) compared with text.
- Test F1 is 0.012 below val, but the share of the gap closed is about the
  same (42.8% against 42.6%), so the drop is mostly the lower test floor.

## Nearest-neighbour analysis
[Describe the figure from cell 6: for each of three query images (from groups
of at least 4 listings), how many of the five nearest neighbours are the same
product (green) and which are different products (red), and why you think so.]

## Error analysis (ResNet50, threshold 0.79, val)
Over val pairs: 4,516 correct, 1,422 false matches, 6,495 missed. Pooled pair
recall is about 0.41 against 0.626 per listing; probably because large
groups contribute many pairs (untested).
- **False matches** (the four highest-scoring wrong pairs): [confirm
  against your figure] rolls of bubble wrap in near-identical photos with
  different group labels. These are the worst cases, not typical ones.
- **Missed matches** (a random sample of true pairs below the threshold):
  [confirm] the same product in different compositions: plain photo against
  promotional banner, different arrangements.

## Answers to the questions to consider
**What information does an image embedding capture?** ResNet50 features come
from a network trained to classify ImageNet categories, so they capture
shapes, textures, colours and layout. They have no notion of "the same
product", and the failures above show they respond to composition and
background.

**Why might two images of the same product have different embeddings?**
Different angle, background, crop or promotional overlay changes the
features. In the missed-match sample the pairs differ in composition.

**Why might different products have highly similar embeddings?** Generic
products photographed alike (bubble wrap) look the same; Part A also showed
photos reused across variants such as sizes.

**Which similarity metric works best?** I used cosine on L2-normalised
embeddings. For unit vectors, squared Euclidean distance is 2 minus 2 times
cosine, so both rank pairs identically; I did not run Euclidean separately.
Mean-centred cosine helped CLIP and was a tie for ResNet50. Other metrics
and learned metrics were not tried.

**How does the threshold affect results?** [From your plot: below about 0.6
almost every pair passes and F1 falls under the floor; at 0.79 precision is
0.938 and recall 0.626; above that precision approaches 1 and recall falls.]
Best thresholds differ by method (0.79 ResNet, 0.83 CLIP), so thresholds
cannot be compared across models.

**What are the computational challenges?** A dense similarity matrix is 105
MB for 5,115 images but 4.69 GB for all 34,250; embedding took 69 s
(ResNet50) and 36 s (CLIP) for 5,115 images on a GPU, about 7.7 and 4.0
minutes extrapolated to all images at the same speed. Larger pools need
chunked or approximate nearest-neighbour search (not tried).

## Not done
Fine-tuning a vision model, other architectures, approximate nearest-neighbour
search, repeating runs over several splits.

## How to rerun
Open the notebook on Kaggle with the Shopee competition data attached, GPU
and Internet on, then Restart and Run All (several minutes).
