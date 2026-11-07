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

## Result

`reports/bilstm/tuning.json` and `reports/bilstm/final.json`, from commit 528b676 on the molab
GPU. The vocabulary has 55,254 ids: only that many tokens are seen 5 times or more, so the
200,000 cap was never reached.

Tuning, mean Macro F1 of the two heads with offsets on the 50,000-review sample, each setting
best at epoch 2:

| Setting (hidden, layers, learning rate) | Score |
|---|---|
| 512, 2, 2e-3 | 0.6836 |
| 512, 1, 2e-3 | 0.6833 |
| 256, 2, 2e-3 | 0.6826 |
| 512, 2, 1e-3 | 0.6812 |
| 256, 2, 1e-3 | 0.6802 |
| 256, 1, 2e-3 | 0.6796 |
| 512, 1, 1e-3 | 0.6768 |
| 256, 1, 1e-3 | 0.6749 |

The best setting is on the edge of the grid in all three directions and at the last epoch;
as planned, this is written down and not chased.

Full run (512, 2 layers, 2e-3) on all 3,552,498 training reviews: the kept check is at epoch
3.0 (0.7020 on the sample); training stopped after epoch 3.5; 18.7 minutes of training. On
the whole validation set, Macro F1 with offsets:

| Head | BiLSTM [95%] | FaBERT (kept run) | TF-IDF |
|---|---|---|---|
| recommendation | 0.7176 [0.7161, 0.7190] | 0.7324 | 0.7017 |
| rating band | 0.6826 [0.6813, 0.6839] | 0.6922 | 0.6683 |

F1 per class, BiLSTM / FaBERT: `recommended` 0.942 / 0.944, `not_recommended` 0.719 / 0.744,
`no_idea` 0.492 / 0.510; `low` 0.696 / 0.711, `medium` 0.466 / 0.479, `high` 0.885 / 0.887.

| | Expected | Found |
|---|---|---|
| 2 layers better than 1 | yes | yes in all four pairs, but by only 0.0003 at 512 and 2e-3 |
| Spread of the 8 settings | within 1.5 points | 0.87 points: right |
| Recommendation | 0.69 to 0.71 | 0.7176: wrong, higher |
| Rating band | 0.65 to 0.68 | 0.6826: wrong, a little higher |
| Against FaBERT | 2 to 4 points lower | 1.0 to 1.5 points lower: wrong, closer |
| Against TF-IDF | close | 1.4 to 1.6 points higher: wrong, better |
| `no_idea` and `medium` the weakest | yes | yes, about 1.3 to 1.8 points below FaBERT |
| Time | under 2 minutes per setting, 20 to 60 for the full run | about 1.5 and 19 minutes: right |

A model that starts from nothing and reads only our reviews comes within 1 to 1.5 points of
the pretrained FaBERT, and the same two classes are the weakest. It is less than 5 points
behind FaBERT on both heads, so the four-model comparison uses it without a separate report
without it.

## Two more seeds (written before running)

The decision checks (`reports/decision_recheck.json`) and the three-model comparison
(`reports/reading_models.json`) rest on one training run of each model, and their intervals
measure only which validation reviews happen to be drawn, not the chance in training
itself. The BiLSTM is cheap to train again, so the full run is repeated with training seeds
2027 and 2028 (`scripts/train_bilstm.py --seed`, steps 82 and 83): the same best setting,
vocabulary, training reviews and choosing sample; only the initial weights, dropout and the
order of the reviews change. `scripts/check_bilstm_seeds.py` (step 84,
`reports/bilstm_seeds.json`) then reports the spread of Macro F1 over the three seeds,
FaBERT minus BiLSTM for each seed, and the three-model comparison redone with each seed.

Expected:

- The three seeds are within 0.3 points of each other on each head (the FaBERT pilots on
  500,000 reviews moved by 0.06 to 0.09 points between seeds; a model trained from scratch
  moves more).
- FaBERT stays ahead of the BiLSTM by about 1 point on both heads with every seed, and the
  BiLSTM stays ahead of TF-IDF.
- The counts of the three-model comparison change by under 3% from one seed to another for
  all right and all wrong, and the share of all wrong with the same wrong label stays within
  one point of 90% and 94%: the shared mistakes do not depend on one run.
