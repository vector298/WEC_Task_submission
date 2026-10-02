# Log: Shopee Part C (image matching)

<!-- One entry per session:
## date: what I did
Hypothesis:
Setup:
Result:
Observation:
Next:
-->

## 2026-10-02: baseline, ResNet50 image embeddings + cosine
>> Hypothesis: [your prediction. Mine would be: image similarity will score
below word TF-IDF (val 0.777), because sellers reuse one photo across
variants and one product has very different photos, but it should link
some listings whose titles share no words.]

>> Setup: torchvision ResNet50 (ImageNet weights), classifier head removed,
2048-d pooled features, L2-normalised, val images only (frozen model, so
train images are not needed). Embedding the 5,115 val images took 69 s on
a CUDA GPU. Cosine similarity, threshold sweep 0.30 to 0.975 in steps of
0.025.

>> Result: best val F1 0.69 at threshold 0.775 to 0.80 (floor 0.4629).

>> Observation: [where precision and recall cross; what the red neighbours
look like.]

>> Next: a second configuration (CLIP image encoder) for comparison.




## 2026-10-02: Experiment 1, CLIP image embeddings
>> Hypothesis: CLIP is generally usd=ed for matching images to text but since that is not happening here , ResNET generally performs better 

>> Setup: openai/clip-vit-base-patch32 image encoder, 512-d, L2-normalised,
val images only (36 s on a CUDA GPU; ResNet50 took 69 s, not a controlled
comparison). Cosine similarity, threshold sweep 0.40 to 0.975 in steps of
0.025.

>> Result: best val F1 0.6612 at threshold 0.825 (floor 0.4629; ResNet50
[raw value from cell 11]). Mean raw similarity 0.454.

>> Observation: the precision-recall curve for ResNet50 lies above CLIP's from
recall about 0.5 to 0.9 and they converge above 0.9, so ResNet50 was better
in aggregate here. This is one frozen CLIP variant used image-only, so it
does not show CLIP is worse in general.

>> Next: mean-centred embeddings.

## 2026-10-02:## Experiment 2: mean-centred embeddings

Subtracting the average embedding and re-normalising removes the component
every image shares, which spreads the similarities out (cosines can become
negative). It uses no labels. Best val F1: ResNet50 0.6891 (threshold
0.725), CLIP 0.6751 (0.625). Centring raised CLIP by 0.014 over its raw
0.6612; [ResNet50 raw vs centred]. On unit-length vectors cosine and
Euclidean distance rank pairs identically (squared distance = 2 - 2 cos),
so I report cosine only..


## 2026-10-02: error analysis and final test run



>> Setup: ResNet50 raw, threshold 0.79 (val-chosen). Pairs counted over val.

>>Test: embeddings of the test images, threshold 0.79, a single run.

>>Result: val pairs: 4,516 correct, 1,422 false matches, 6,495 missed.

>>Test F1 0.68 against a test floor of 0.4401 (val 0.6917 against 0.4629).

>>Observation: false matches are generic products photographed alike (bubble
wrap); missed matches are true pairs whose photos differ in composition or
use a promotional banner. Test F1 is 0.012 below val, but the share of the
floor-to-perfect gap closed is about the same (42.8% vs 42.6%), so the drop
is mostly the lower test floor. Nothing was changed after seeing the test
result.

>>Next: README, then the Finale.



## Error analysis (ResNet50 raw, threshold 0.79, val)

Over all val pairs: 4,516 correct matches, 1,422 false matches and 6,495
missed true pairs. (Pooled pair recall is about 0.41, below the per-listing
mean of 0.626, probably because large groups contribute many pairs;
untested.)

**False matches (the four highest-scoring wrong pairs).** [Confirm against
your figure:] all four show rolls of bubble wrap with near-identical
photos but different group labels. Generic products photographed in the
same way are indistinguishable by appearance; the variant that separates
the groups (for example length) is not visible.

**Missed matches (a random sample of true pairs below the threshold).**
[Confirm:] the pairs differ in composition: a plain product photo against
a promotional banner or a different arrangement (scarves, a skincare
bottle against an "11.11" banner, a handheld fan against a text banner,
lipsticks). A frozen ImageNet model has no notion of "the same product" and
responds to layout and background.

Both failure types match Part A: reused photos across variants, and one
product shown in very different images.

## Experiment 3: fine-tuning ResNet50 on the group labels

**Motivation.** The frozen ImageNet features respond to composition and
background and have no notion of "the same product". The error analysis showed
this: the same product in different compositions was missed, and generic items
photographed alike were matched. Fine-tuning with the `label_group` labels
should teach the embedding that photos of one product belong together.

**Setup.**
- Training data: the train split only (23,724 images, 7,709 groups). Each
  group is a class, used only as a training signal. Val and test products are
  never seen in training, so the evaluation still measures matching of unseen
  products.
- Only the last block of ResNet50 (`layer4`, about 15 M parameters) is
  trained. The earlier blocks stay frozen, which is faster and overfits less.
- Loss: ArcFace, which adds an angular margin to the correct group's angle so
  that same-group embeddings are pulled together.
- Augmentation: random crop, horizontal flip and small colour jitter.
- Val F1 is checked after every epoch, and the best epoch is kept. The
  threshold is chosen on val and test is run once.

**Attempt 1 failed.** The embeddings collapsed: on val the similarity between
different images had a mean of 0.995 (minimum 0.94), and F1 was 0.476, barely
above the floor of 0.4629 and far below the frozen 0.6917. The epoch log was not
saved, and I did not diagnose the cause.

**Attempt 2 changes** (standard remedies, not proven to be the cause): a
projection layer (2048 to 512 numbers with BatchNorm) before the ArcFace head;
a margin that ramps from 0 to 0.3 over the first epoch; a lower learning rate
for `layer4` (5e-5); and a monitor that stops training if the mean cosine
between embeddings in a batch exceeds 0.95.


## 2026-10-02: Experiment 3b, fine-tune layer4 with ArcFace (attempt 2)

>> Setup: ResNet50 ImageNet weights, layer4 trainable (15.0 M parameters), a
projection layer (2048 to 512, BatchNorm), ArcFace head (s = 30), margin
ramped 0 to 0.3 during epoch 1, AdamW (layer4 5e-5, projection and head
1e-3), cosine schedule, 4 epochs, mixed precision, random crop / flip /
colour jitter, val checked every epoch.

>> Result: mean train loss 9.005, 4.843, 1.919, 1.067; best val F1 by epoch
0.7494, 0.7648, 0.7720, 0.7729 (coarse grid); refined: 0.7733 at threshold
0.445, precision 0.909, recall 0.752 (frozen: 0.6917, 0.938, 0.626). About
171 s per epoch. Val similarity min / mean / max -0.287 / 0.003 / 1.0.

>> Observation: no collapse; fine-tuning raised val F1 by 0.082 and closes
about 58% of the floor-to-perfect gap (frozen 43%), close to text alone
(0.777). Gains flatten by epoch 3. I changed the projection layer, margin
ramp and learning rate together, so I did not isolate which fixed attempt
1. The "mean batch cos" monitor stayed at -0.015 throughout, which
BatchNorm forces regardless of collapse, so it was not informative.

>> Next: run test once with threshold 0.445, then use the fine-tuned
embeddings in the Finale.


## 2026-10-02: final evaluation of the fine-tuned model on test
>> Hypothesis:
- the fine-tuned model should score above the frozen 0.68 on test, as it
  did on val
- test should be slightly below val, as in Parts B and C, since test has
  larger groups and a lower floor
- weights and threshold are fixed from val and nothing changes after seeing
  test
>> Setup: epoch-4 weights reloaded (check: val F1 0.7733 reproduced), test
images embedded once, threshold 0.445 chosen on val.

>> Result: test F1 0.7743 against a test floor of 0.4401 (frozen: 0.68).
Closes 59.7% of the floor-to-perfect gap on test (57.8% on val; frozen
42.8% on test).

>> Observation: fine-tuning raised test F1 by 0.094. Test was 0.001 above val
instead of below it; with a lower test floor I treat that as noise from one
split. Fine-tuned image alone ties word TF-IDF on test (0.7741). Nothing
was changed after seeing the result.

>> Next: error analysis of the fine-tuned model, README, then the Finale with
the new embeddings.
