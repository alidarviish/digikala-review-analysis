# Models that read differently, review by review (written before running)

## Why

Every model so far is weakest on `no_idea` and `medium`, and the label analyses
(`reports/rating_boundaries.json`, `reports/rate_recommendation.json`,
`reports/label_conflict.json`) point at the labels. Models that read the text in different
ways give a direct test: a review that every model gets wrong, each in its own way of
reading, most likely has something in the data that no reader can recover from the text; a
review only one model gets right shows what that way of reading adds.

## Models

| Name | How it reads | Predictions |
|---|---|---|
| FaBERT | pretrained Transformer encoder, word pieces | `artifacts/transformer_no_early_stop` (the kept run) |
| TF-IDF | bag of words and word pairs, linear | `artifacts/tfidf_final` |
| BiLSTM | recurrent, word by word in order, trained from scratch | `artifacts/bilstm` (`reports/bilstm_plan.md`) |
| decoder | small pretrained generative model (later, its own plan) | added as a fourth column when it exists |

Each model's labels with offsets are used, as saved. Before anything is counted, each model
must score on the whole validation set exactly the Macro F1 recorded for it
(`reports/error_overlap.json` for FaBERT and TF-IDF, `reports/bilstm/final.json` for the
BiLSTM), and its predictions must join validation one to one by id.

## Reviews

Validation only, without the 50,000-review sample used for FaBERT's checks and the BiLSTM's
tuning: 872,795 reviews, and for each head those with its label. The test set is not read.

## What is reported (`scripts/compare_reading_models.py`, `reports/reading_models.json`)

Exact counts and shares, for each head, overall and for each true class:

1. Macro F1 of each model on these reviews.
2. Every right/wrong pattern of the models (8 with three models, 16 with four), with its count.
3. Groups: all right, all wrong, exactly one right (and which model), exactly two right
   (with three models: which one is wrong).
4. For "all wrong": how many reviews the models agree on the same wrong label, and which
   true label goes to which agreed label.
5. Known causes, counted in each group (all right, some right, all wrong), a review can have
   several:
   - the buyer's two labels disagree, by the mapping used in `reports/label_sample_plan.md`
     (high with `recommended`, medium with `no_idea`, low with `not_recommended`); and the
     strong case, `recommended` with low or `not_recommended` with high;
   - a rate of exactly 3;
   - the same text is also in train (`text_seen_in_train`);
   - an emoji in the text;
   - a very short text, 3 words or fewer;
   - none of these.

Report only: nothing in the models, labels or splits changes.

## Expected

- Recommendation: all right 78 to 82%, all wrong 8 to 10%; rating band: all right 70 to 74%,
  all wrong 14 to 17% (with TookaBERT-Large instead of the BiLSTM, `reports/error_overlap.json`
  found 81.4% and 9.0%, 72.9% and 15.8%; the BiLSTM reads more differently from FaBERT than
  Large does, so a little less agreement).
- On "all wrong", the models give the same wrong label on at least 70% of the reviews.
- Exactly one right: FaBERT alone most often, TF-IDF alone least often.
- The buyer's two labels disagree at least three times as often in "all wrong" as in "all
  right"; for rating band, at least half of "all wrong" has a rate of exactly 3.
- No known cause: under 40% of "all wrong" for rating band; more for recommendation, where
  the causes measured so far are weaker.

## Added after the first run, before running again

The first run (`reports/reading_models.json`, commit d5ebe40) left 55.1% of the high reviews
that all three models got wrong without any known cause; on 33,037 of them all three said
medium. In the hand reading (`reports/error_sample_plan.md`, group C), reviews rated high
but read as medium were mostly "good, but ..." texts. Hypothesis: these are mostly reviews
rated exactly 4, where the text reads like a 3. One cause is added, "a rate of exactly 4",
and the comparison is run again; nothing else changes.

Expected, before running again: of the high reviews all three models get wrong, at least 70%
have a rate of exactly 4 (against about 40% of all high reviews: 227,836 of 586,764 high
reviews with a whole rate in validation are rated 4), and no known cause drops from 55.1%
to under 20% for them. For the other classes and groups the new cause changes little.
