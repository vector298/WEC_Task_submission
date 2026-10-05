## Results
Training: both models were trained with an 80-epoch cap and early stopping on validation loss (patience 2).
The grounded model ran 23 epochs (best epoch 21: validation loss 5.161, perplexity 174.3, train loss 4.680)
and the ungrounded model 22 epochs (best epoch 20: validation loss 5.114, perplexity 166.3, train loss 4.954).
Validation loss flattened while train loss kept falling, so both overfit after roughly epoch 17; the grounded
model overfits more (train-validation gap 0.48 against 0.16). A first run with a 12-epoch cap left both
underfitted and was replaced before the test set was used.

**Test set (933 examples from 27 conversations, greedy decoding, one run):**

| System | BLEU | ROUGE-L | distinct-1 | distinct-2 | Length | Grounding (mean per reply) |
|---|---|---|---|---|---|---|
| Retrieval (TF-IDF) | 0.42 | 0.064 | 0.159 | 0.447 | 10.6 | 0.040 |
| Ungrounded generator | 0.72 | 0.108 | 0.014 | 0.055 | 6.7 | 0.000 |
| Grounded generator | 0.98 | 0.107 | 0.017 | 0.072 | 8.2 | 0.282 |
| Human reference | n/a | n/a | 0.208 | 0.684 | 12.6 | 0.100 |

BLEU is below 1 for every system (many valid replies, one reference), so BLEU differences are not meaningful;
ROUGE-L is the more usable measure. The grounding rate uses only content words (words outside the 200 most
frequent training words) and the generators produce very few:

| System | Replies with at least one content word | Content words per reply | Share of content words found in the document |
|---|---|---|---|
| Retrieval | 87.1% | 3.65 | 0.057 |
| Ungrounded | 2.8% | 0.03 | 0.000 |
| Grounded | 10.1% | 0.40 | 0.125 |
| Human reference | 92.7% | 4.79 | 0.108 |

The mean-per-reply grounding in the table above is therefore computed over about 94 replies for the grounded
model (10.1% of 933, approximate) and about 26 for the ungrounded one; the pooled share in the second table
(0.125 against 0.108 for the references) is the safer figure.

**Document-swap test (grounded model; 94% of the examples received a different movie's document):**

| | BLEU | ROUGE-L | Grounding to the true document | Grounding to the swapped document |
|---|---|---|---|---|
| Correct document | 0.98 | 0.107 | 0.282 | n/a |
| Swapped document | 0.82 | 0.105 | 0.047 | 0.391 |

91.7% of the outputs changed when the document was swapped.

**By code-mixing proxy (terciles of the proxy on the test set: 0.20 and 0.37; BLEU / ROUGE-L):**

| Group | Examples | Retrieval | Ungrounded | Grounded |
|---|---|---|---|---|
| Lowest third (most Hindi-heavy, proxy 0.00 to 0.20) | 341 | 0.33 / 0.064 | 0.88 / 0.106 | 0.69 / 0.110 |
| Highest third (most English-heavy, proxy 0.37 to 1.00) | 313 | 0.38 / 0.064 | 0.79 / 0.099 | 0.71 / 0.098 |

A first split with fixed thresholds put 810 of the 933 examples in one group and only 35 in the other and is
not used.

**Distinct replies out of 933:** retrieval 485, ungrounded 360, grounded 485. Most common outputs: ungrounded
"kya" (43), "ye ek hai hai" (27), "mujhe , hai ki hai" (25); grounded "kya" (44), "mujhe , hai ki hai ." (35),
"mujhe lagta hai ki hai ." (24); retrieval "mene is dunkirk ko dekha he" (22), "hello" (19).

## Analysis
- **The grounded model uses the document, but rarely and without a measurable benefit.** Its outputs change
  with the document (91.7% changed when swapped) and its content words follow the swapped document (grounding
  0.391 to the swapped one, 0.047 to the true one), and 12.5% of its content words occur in the document (the
  references: 10.8%). But only about 10% of its replies contain any content word, so this rests on roughly 94
  replies and about 370 words. ROUGE-L (0.107 against 0.108), BLEU and validation loss (5.161 against 5.114)
  are not better than the ungrounded model's; the examples suggest it mostly copies names and titles (for
  example "la land") and not facts.
- **Both generators collapse into generic replies.** Distinct-2 is 0.055 and 0.072 against 0.684 for the
  references, replies are short (6.7 and 8.2 tokens against 12.6), and the most common outputs are "kya" and
  sentence frames with the content missing ("mujhe lagta hai ki hai ."). Greedy decoding always picks the
  frequent words, so content words are rare. Probable causes (not tested): very little data (about 7,900
  training examples), maximum-likelihood training, and greedy decoding; sampling or a diversity penalty were
  not tried.
- **Retrieval** returns real replies (distinct-2 0.447) but only 485 distinct replies for 933 examples, often
  for the wrong context (a Frozen reply for a La La Land conversation), and has the lowest overlap with the
  references (ROUGE-L 0.064).
- **Overfitting:** with 6 to 7 million parameters for about 95,000 target tokens both models overfit after about
  epoch 17; the grounded model, with more capacity (7.1M against 6.2M parameters), has the lower train loss and
  the higher validation loss, which probably explains why the document gave no validation gain (not tested).
- **Code-mixing:** I expected the most Hindi-heavy replies to be harder. The results do not show it: ROUGE-L is
  0.106 to 0.110 in the Hindi-heavy third and 0.098 to 0.099 in the English-heavy third, differences of about
  0.01 on 341 and 313 examples, within noise.

## Limitations
Very little data (about 7,900 training examples; 933 test examples from 27 conversations), so every number is
noisy; BLEU and ROUGE-L are weak measures of dialogue quality, there is no human evaluation, and BLEU is below 1
throughout; the grounding rate and the code-mixing proxy are my own rough measures, and the grounding evidence
rests on about 10% of the grounded model's replies; the model sees only the current document section; one
seed; word-level vocabulary (9.0% of document tokens are `<unk>`, mostly names); greedy decoding only; the first
run (12-epoch cap) was underfitted and was replaced before the test set was used.
