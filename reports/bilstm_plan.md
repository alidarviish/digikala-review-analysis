# A recurrent model: BiLSTM trained from scratch (written before running)

## Why

The models so far read the text in two ways: FaBERT, a pretrained Transformer encoder, and
TF-IDF with a linear model, a bag of words. TookaBERT-Large, XLM-R and mmBERT are also
Transformer encoders, and FaBERT and TookaBERT-Large share 77 to 81% of their mistakes
(`reports/error_overlap.json`). To learn which mistakes come from the data and which from a
way of reading, four models that read differently are compared review by review: FaBERT,
TF-IDF, this BiLSTM, and a small pretrained decoder (its own plan, after its Persian support
and licence are checked).

The BiLSTM reads the words in order, like a person, but knows nothing before training: all
it knows comes from our training reviews.

## Model

- Input: the normalized text of the cache, as for every other model; only the review text.
- Words: the project's `persian` token pattern (`src/digikala_reviews/tfidf.py`): words kept
  whole across half-spaces, and every emoji and punctuation mark as its own token. So emoji,
  which FaBERT sees as `[UNK]` (`reports/unknown_tokens.json`), are read.
- Vocabulary: tokens seen at least 5 times in `train_supervised`, at most 200,000; one id
  for padding and one for any other token. At most 128 tokens per review (99% of reviews
  have at most 62 words; longer ones are cut at the end).
- Network: embeddings of size 300 trained from scratch; a bidirectional LSTM; the mean and
  the maximum of its outputs over the real tokens, side by side; dropout; two linear heads
  (recommendation and rating band), as FaBERT.
- Training: the sum of the two cross-entropies, no class weights (as FaBERT); AdamW, weight
  decay 0.01, 6% warmup then linear decay; batch 256; gradient norm clipped at 1.0.
- Labels with offsets: per-class offsets tuned on the other half by product, exactly as for
  FaBERT and TF-IDF.

## Tuning (fixed budget, decided now)

- 8 settings: hidden size 256 or 512 per direction, 1 or 2 LSTM layers, learning rate 1e-3
  or 2e-3; dropout 0.3 and embedding size 300 for all.
- Each setting: one seed (2026), the same fixed 500,000 training reviews, 2 epochs, scored
  on the fixed 50,000-review validation sample used by every FaBERT check (the mean Macro F1
  of the two heads with offsets).
- The best setting is trained once on all 3,552,498 training reviews, up to 4 epochs, four
  checks per epoch on the same 50,000 reviews, the best check kept and stopping after two
  checks in a row without improvement, as the first FaBERT run.
- Nothing else is tuned. If the best setting is on an edge of the grid (for example 2 layers
  and 512 both best), that is written down, not chased with more runs.

## Comparison set

FaBERT's checks, and now the BiLSTM's tuning, used the 50,000-review sample, so it is left
out of every comparison: the four models are compared on the other 872,795 validation
reviews (fewer per head, only reviews with that label). The numbers of FaBERT and TF-IDF on
this set are recomputed, so they will differ slightly from the whole-validation numbers.

## Outputs

- `artifacts/bilstm/` the vocabulary, the kept weights and the validation predictions, with
  the same columns as the Transformer runs (log-probabilities and labels with offsets of each
  head), so the four-model comparison reads all models the same way.
- `reports/bilstm/tuning.json` the 8 settings and their scores; `reports/bilstm/final.json`
  checks, scores and timing of the full run.
- Both runs record the commit, the cache hashes and the environment, as every other run.

## Expected

- Tuning: 2 layers do better than 1, and the 8 settings are within 1.5 points of each other.
- Full run: recommendation 0.69 to 0.71 and rating band 0.65 to 0.68 (Macro F1 with
  offsets): close to TF-IDF (0.7017 and 0.6683 on all of validation) and 2 to 4 points below
  FaBERT, since it starts from nothing.
- `no_idea` and `medium` stay the weakest classes, at least as weak as with FaBERT.
- Time on the molab GPU: under 2 minutes per tuning run, 20 to 60 minutes for the full run.

## How the BiLSTM is used later

It is always one of the four models of the comparison, whatever its score, since the point
is how it reads, not to replace FaBERT. If it is more than 5 points below FaBERT on a head,
the comparison is reported both with and without it, so a weak model does not fill the
"only one model wrong" group with its own weakness.

The four-model comparison gets its own plan before it runs. It reports exact counts, per
head and per true class: every one of the 16 right/wrong patterns of the four models, and
the groups all right, all wrong, exactly one right (which one), exactly two and exactly
three right; and for "all wrong", how many reviews have each known cause (the buyer's two
labels disagreeing, a rate of exactly 3, a text also in train with another label, emoji,
very short text) and how many have none.

The test set is not read.
