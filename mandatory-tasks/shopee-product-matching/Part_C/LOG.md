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
