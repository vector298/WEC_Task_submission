# Subtask 3: Document-Grounded Dialogue Generation in Hinglish

## Goal
Generate the next reply in a conversation, in Hinglish, conditioned on two sources at once: the
conversation history and a grounding Wikipedia document, and show that the document is actually used.

## Status
Implementation written in `notebook.ipynb`. STATUS: All models trained and evaluated
## Data (verified)
CMU Hinglish DoG (`festvox/cmu_hinglish_dog`). Each Hugging Face row is one utterance with `translation`
(`en` and `hi_en`), `uid`, `docIdx`, plus conversation-level fields (`date`, `wikiDocumentIdx`,
`rating`, `status`, `user2_id`, and others). **The grounding documents are not in these rows**: they
are in the original English repo `festvox/datasets-CMU_DoG`, folder `WikiData`: 30 movie documents of
4 sections each (section 0 is structured: movie name, year, director, genre, introduction, rating,
cast and critical response; sections 1 to 3 are plain text, about 126, 155 and 151 words on average).
`wikiDocumentIdx` (0 to 29) selects the document and the turn's `docIdx` is the section being shown
when the utterance was said, so the right grounding text for a reply is that section of that movie.
The English repo has 3,373 / 229 / 619 conversation files for train / validation / test and 4,112
conversations in total (about 21 turns each, per its README); the notebook prints the sizes of the
Hinglish version actually used.

## Preprocessing
- Conversations are rebuilt by grouping rows on date, wikiDocumentIdx and user2_id (rows are in order).
- One example per turn after the first: input = previous 3 turns (Hinglish, joined with a `<sep>`
  token, last 60 tokens), document = the current section (first 150 tokens), target = the turn's
  Hinglish text (first 30 tokens).
- Tokenizer: lowercase word-level, punctuation split, Devanagari-safe. One shared vocabulary for
  history, replies and documents (up to 12,000 words, minimum frequency 2, built from the training
  split only) so names and titles share embeddings. **Embeddings are learned from scratch**
  (randomly initialised and trained jointly with the model; no pretrained vectors).
- Code-mixing proxy (my own, rough): the share of a reply's Hinglish tokens that also occur in its
  English version. Replies with a share of at most 0.5 are treated as heavily code-mixed and 0.95 or
  more as mostly English-like.

## Models
- **Grounded model:** two LSTM encoders (history and document) with a shared embedding table; the
  decoder state starts from a learned combination of both final states; at every step the decoder
  attends separately over the history states and the document states (Luong "general" attention with
  padding masks) and predicts from tanh(W [history context ; document context ; decoder state]).
- **Ungrounded baseline:** the same network without the document encoder and document attention.
  Same data, hyperparameters (embedding 256, hidden 256, dropout 0.3), seed and early stopping, so the
  only difference is the document.
- **Retrieval baseline:** TF-IDF similarity between the conversation history and each training
  history; the reply of the most similar training history is returned.
- **Training:** Adam 1e-3, batch 128, gradient clipping 1.0, scheduled teacher forcing from 1.0 down to
  0.6, early stopping on validation loss (patience 2), at most 12 epochs and a 25-minute cap, seed 42.
  Decoding: greedy, up to 30 tokens.

## Evaluation
BLEU (whitespace tokens, lowercase, one reference truncated to the same 30 tokens), ROUGE-L
(sentence-level F-measure, averaged), distinct-1 and distinct-2, mean reply length, and a **grounding
rate** (share of a reply's content words, those outside the 200 most frequent training words, that
occur in the true document section; computed for the references too). Results are also split by the
code-mixing proxy. **Document-swap test:** the grounded model is run with documents shuffled across
test examples; if it really uses the document, its grounding to the true document should fall and
its grounding to the swapped document should rise.

## Hypotheses (stated before running)
- The grounded model should have lower validation loss and higher grounding rate than the ungrounded
  one, and its outputs should change when the document is swapped.
- BLEU and ROUGE-L will be low for every system (many valid replies, one reference), so the
  differences between systems matter more than the absolute values.
- Heavily code-mixed references should be harder (lower scores) than mostly English-like ones.
- The retrieval baseline should be fluent but off-topic for the specific document.

## Results


## Qualitative analysis


## Limitations
Word-level vocabulary (rare names and Hinglish spellings become `<unk>`); single reference per reply;
BLEU and ROUGE-L are weak measures of dialogue quality and there is no human evaluation; the
code-mixing measure and the grounding rate are my own rough proxies; only the current document section
is given to the model; one seed and one run; models are small and trained for a short budget; decoding
is greedy, so generic or repetitive replies are likely.

## Reproduce
Kaggle with a GPU and Internet on. The Hugging Face dataset is loaded with `datasets` and the 30
documents are downloaded from the original GitHub repo (`festvox/datasets-CMU_DoG`). Seed 42.
