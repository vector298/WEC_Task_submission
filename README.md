# WEC Intelligence SIG Recruitment 2026: Task Submission

**Candidate:** Ishaan (GitHub: [vector298](https://github.com/vector298)), Mining Engineering, NITK (batch of 2029)
**Task statement:** [WebClub-NITK/Intelligence-SIG-Recs-2026](https://github.com/WebClub-NITK/Intelligence-SIG-Recs-2026)

This repository is **private**; the reviewers `PranavBhatP`, `Vivek-k7` and `Ds0uz4` have been added as collaborators.

## What is here

Three mandatory tasks (all completed) and one optional task:

| Task | Folder | Headline result |
|---|---|---|
| Shopee product matching (Parts A to C and Finale) | `mandatory-tasks/shopee-product-matching/` | Text + image score fusion reaches test F1 **0.8508**, from a self-match floor of 0.4401 (text alone 0.7741, image alone 0.6800) |
| Recommender systems: collaborative filtering to DLRM | `mandatory-tasks/recommender-systems-cf-to-dlrm/RecommendationTasks/` | MovieLens 1M: matrix factorisation test RMSE **0.8471** (item-item 0.8523, bias baseline 0.9031). CTR data: MLP test PR-AUC 0.0647; DLRM from scratch 0.0665 (no measurable gain) |
| Seq2seq: translation to grounded dialogue | `mandatory-tasks/seq2seq-grounded-dialogue-generation/` | English to French test BLEU **9.41** without attention, **25.50** with attention (27.72 with beam 5); a grounded Hinglish dialogue model that measurably uses its document but does not beat the ungrounded one on ROUGE-L |
| Neural ODEs (optional) | `non-mandatory-tasks/neural-odes/` | Damped pendulum: after 1,200 epochs the Neural ODE has the lower extrapolation RMSE (0.289 against 0.389 for an LSTM), but the LSTM wins on 5 of 7 damping values |

## Repository map

```text
mandatory-tasks/
├── shopee-product-matching/
│   ├── Part_A/            EDA
│   ├── Part_B/            text matching (TF-IDF and alternatives)
│   ├── Part_C/            image matching (pHash, ResNet50, CLIP, ArcFace fine-tuning)
│   └── Final/             multimodal fusion, ablation, error analysis, report.pdf
├── recommender-systems-cf-to-dlrm/RecommendationTasks/
│   ├── 01_CollaborativeFiltering/   memory-based CF and matrix factorisation (MovieLens 1M)
│   ├── 02_NeuralCTR/                neural click prediction
│   ├── 03_DLRM/                     DLRM implemented from scratch
│   └── recommender_systems_report.pdf
└── seq2seq-grounded-dialogue-generation/
    ├── subtask1/          LSTM encoder-decoder (English to French)
    ├── subtask2/          attention and decoding strategies
    ├── subtask3/          document-grounded Hinglish dialogue
    └── report.pdf         3-page report covering all three subtasks
non-mandatory-tasks/
└── neural-odes/           Neural ODE against LSTM on a damped pendulum
```

Every task folder contains `README.md` (method, results, analysis, limitations), `LOG.md` (a dated log in the form
hypothesis, setup, result, observation), `notebook.ipynb` (with its saved outputs) and `results/` (figures and tables).
Start with each folder's README; the logs show how decisions were made, including the mistakes.

## Summary of the work

**Shopee product matching.** Mean per-listing F1 on a group-disjoint 70/15/15 split (floor from returning only the listing itself:
0.4629 validation, 0.4401 test). Text: word TF-IDF fitted on the unlabelled evaluation-pool titles, threshold 0.43 (validation 0.7770, test 0.7741);
character n-grams and multilingual sentence embeddings were no better or worse. Image: frozen ResNet50 0.6917 validation; fine-tuning with an
ArcFace loss (last block plus a 512-d projection) reached 0.7733 validation and 0.7743 test. Finale: score-level fusion of text and (frozen) image
similarity, weight 0.45 and threshold 0.52 chosen on validation (0.8524 validation, 0.8508 test), closing about 73% of the gap to a perfect score;
adding pHash did not help. Errors are dominated by products that share a photo but differ in a size token (for example 6W against 8W) and by
same-product pairs with different photos and loose titles. `Final/report.pdf` has the seven-section report.

**Recommender systems.** *Collaborative filtering* (MovieLens 1M, random 80/10/10 split of the ratings): bias baseline, item-item and user-user
neighbourhoods (residuals over the baseline, shrinkage) and biased matrix factorisation written from scratch (hand-written SGD). MF is best on RMSE
(0.8471, 95% interval [0.8431, 0.8512]) but only 0.0052 RMSE ahead of item-item (paired bootstrap [0.0036, 0.0068]); on an offline top-10 list a
popularity baseline beat every rating-based method. *Neural CTR* (supplied advertising data, 3.2% clicks): embedding MLP chosen by a six-architecture,
three-seed sweep (the architectures were not separated); test PR-AUC 0.0647 (random 0.0335), weaker than validation, cause not established.
*DLRM* (from scratch, same split and protocol): the interaction layer did not help; replacing it by concatenation improved validation PR-AUC in all
six matched comparisons, and the selected DLRM is indistinguishable from the MLP on test (PR-AUC 0.0665).

**Seq2seq.** Subtask 1: 2-layer LSTM encoder-decoder, word-level vocabularies, test BLEU 9.41 [9.17, 9.66]; BLEU falls from 19.7 to 4.0 as source length
grows, 43.6% of sources contain an unknown word, and 22.8% of outputs repeat a trigram. Subtask 2: Luong attention under the identical protocol gives
25.50 [25.10, 25.87] (2.9 times the baseline's validation BLEU), a 2.5 to 4.5 times flatter length curve, and beam search adds about 2 BLEU over greedy;
sampling trades BLEU for diversity. Subtask 3: dual-attention grounded generator against an ungrounded twin and a TF-IDF retrieval baseline on 27 test
conversations: all systems score very low (BLEU below 1; ROUGE-L 0.107 grounded, 0.108 ungrounded, 0.064 retrieval), both generators give generic replies,
and a document-swap test shows the grounded model does depend on the document (grounding to the true document 0.282 falls to 0.047), but rarely and without a measurable metric gain.

**Neural ODEs.** Hand-written RK4 and a small vector-field network against a teacher-forced LSTM, with whole damping values held out, a 10 s training window and a 30 s extrapolation target, and a noise experiment. The
Neural ODE is far more robust to training noise inside the training window, and with enough training it extrapolates with 26% lower overall RMSE, but that aggregate
is dominated by the two most lightly damped
