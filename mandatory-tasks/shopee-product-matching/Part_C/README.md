# Shopee Part C: Image-Based Product Matching

Matching listings that are the same product using images only.

## Contents
- `notebook.ipynb`: preprocessing, frozen and fine-tuned embeddings, CLIP,
  phash reference, nearest-neighbour and error analysis (Kaggle, GPU)
- `results/`: plots, error figures and the results table
- `LOG.md`: timestamped experiment log

## Setup
Same split as Part B (by `label_group`, 70/15/15 of groups, seed 42), so
results are comparable. Metric: mean set-F1 per listing. Floors (predict only
itself): val 0.4629, test 0.4401. Train images are used only by the
fine-tuning experiment; the frozen models embed val and test images only.

## Method
Image, embedding, cosine similarity, threshold. Embeddings are
L2-normalised, so a dot product equals cosine similarity.
- **Frozen ResNet50** (ImageNet weights): classifier removed, 2048-d pooled
  features.
- **CLIP ViT-B/32** image encoder: 512-d.
- **Mean-centring** (no labels): subtract the pool's mean embedding.
- **Perceptual hash** (non-learned reference): Hamming distance on the 64-bit
  `image_phash`.
- **Fine-tuned ResNet50** (final): only `layer4` trained (15.0 M parameters),
  plus a 2048-to-512 projection with BatchNorm, ArcFace loss (s = 30, margin
  ramped 0 to 0.3 over epoch 1) over the 7,709 train groups, AdamW (layer4
  5e-5, projection and head 1e-3), cosine schedule, 4 epochs, mixed precision,
  random crop / flip / colour jitter. The best epoch by val F1 (epoch 4) is used.

## Results (threshold chosen on val)
| Experiment | Representation | Dim | Threshold | Val F1 | Precision | Recall | Test F1 |
|---|---|---|---|---|---|---|---|
| Reference | image_phash | 64 bits | max distance 10 | 0.6074 | n/a | n/a | not run |
| Baseline | ResNet50, frozen | 2048 | 0.790 | 0.6917 | 0.938 | 0.626 | 0.68 |
| Exp 1 | CLIP ViT-B/32 | 512 | 0.830 | 0.6616 | 0.918 | 0.602 | not run |
| Exp 2a | ResNet50, centred | 2048 | 0.735 | 0.6900 | 0.950 | 0.615 | not run |
| Exp 2b | CLIP, centred | 512 | 0.635 | 0.6765 | 0.894 | 0.641 | not run |
| **Exp 3 (final)** | **ResNet50, fine-tuned** | **512** | **0.445** | **0.7733** | **0.909** | **0.752** | **0.7743** |

Each row is a single run on one split; gaps of about 0.002 are treated as
ties. The test split was run once for the frozen baseline and once for the
final model, with val-chosen thresholds. Word TF-IDF from Part B scored 0.7770
(val) and 0.7741 (test).

## Key findings
- Fine-tuning raised val F1 from 0.6917 to 0.7733 and test F1 from 0.68 to
  0.7743. It closes about 58% (val) and 60% (test) of the gap between the
  floor and a perfect score, against 43% for the frozen model, and matches
  text alone.
- At val-chosen thresholds, fine-tuning found 1,930 true pairs the frozen
  model missed and lost 35 (6,411 correct pairs against 4,516), but it
  removed 1,009 false matches and introduced 1,087, so false matches stayed
  about the same (1,500 against 1,422).
- Among the frozen models, ResNet50 beat CLIP in aggregate; a possible reason
  (untested) is that CLIP's training aligns images with captions, which
  image-only comparison does not use.
- A perceptual hash reaches 0.6074; learned embeddings add about 0.08 (frozen)
  to 0.17 (fine-tuned).
- Attempt 1 at fine-tuning collapsed (val similarity mean 0.995, F1 0.476).
  Attempt 2 changed the projection layer, the margin schedule and the learning
  rate together, so I did not isolate which fixed it.

## Nearest-neighbour analysis
Three val queries from groups of at least four listings (see
`results/neighbours_frozen_vs_finetuned.png`): same-product neighbours in the
top 5: frozen 4 of 11 possible, fine-tuned 7 of 11. Query 2330 accounts for
the gain; query 3786 finds none under either model. Three queries is
anecdotal.

## Error analysis (fine-tuned, val)
- **False matches:** the four highest-scoring are all one pair of
  bubble-wrap groups, similarity about 1.00 under both models. Identical
  photos in different groups cannot be separated by an image model.
- **Missed pairs:** a niqab listing and knitted-cap listings in one group (may
  be a wholesale listing or label noise), and pairs whose titles clearly name
  the same product but whose photos differ (Nano Spray 30ml, BEBWHITE C).
  A text signal would link the latter.

## Answers to the questions to consider
**What information does an image embedding capture?** Frozen ResNet50
features come from ImageNet classification and capture shapes, textures,
colour and layout; they have no notion of the same product. The fine-tuned
embedding was trained with group labels to place photos of one product close
together, which raised F1 by 0.08.

**Why might two images of the same product have different embeddings?**
Different angle, background, crop or promotional overlay; missed pairs above
differ in composition or show different items.

**Why might different products have highly similar embeddings?** Generic
products photographed alike; Part A also showed one photo reused across
variants. Fine-tuning cannot fix identical photos (the bubble-wrap groups).

**Which similarity metric works best?** Cosine on L2-normalised embeddings.
For unit vectors squared Euclidean distance is 2 minus 2 times cosine, so both
rank pairs identically; I did not run Euclidean separately. Mean-centred
cosine helped CLIP and tied for ResNet50.

**How does the threshold affect results?** Low thresholds admit almost every
pair (precision near zero); high ones keep only near-duplicates. Best
thresholds differ by model (0.79 frozen, 0.445 fine-tuned) because unrelated
pairs average 0.234 for the frozen model and 0.003 for the fine-tuned one, so
thresholds cannot be compared across models.

**What are the computational challenges?** A dense val similarity matrix is
105 MB for 5,115 images but 4.69 GB for all 34,250. Embedding the val images
took 36 s for CLIP and 48 to 69 s for ResNet50 across runs on a CUDA GPU.
Fine-tuning took about 171 s per epoch, about 11.5 minutes for four epochs.
Larger pools need chunked or approximate nearest-neighbour search (not tried).

## Not done
Unfreezing more layers or tuning hyperparameters, other architectures,
isolating which change fixed attempt 1, approximate nearest-neighbour search,
repeated runs over several splits. The collapse monitor in the training loop
(mean cosine between batch embeddings) was uninformative, because BatchNorm
forces it to about -0.016 regardless of the embeddings.

## How to rerun
Open the notebook on Kaggle with the Shopee competition data attached, GPU and
Internet on, then Restart and Run All (about 12 minutes of it is fine-tuning).
