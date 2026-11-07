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

## Added after the second run, before running again: the negative classes

With a rate of exactly 4, the high reviews all three models get wrong have a known cause in
84% of cases. The negative classes now have the most errors without one: 51.9% of the
`not_recommended` reviews all three get wrong, and 36.5% of the `low` ones.

1. One more cause, "a rate of exactly 2": the rating boundary analysis put the model's
   effective low/medium edge near 2.0, and reviews rated 2 are right only half the time
   (`reports/rating_boundaries.json`). The comparison is run again; nothing else changes.
2. `scripts/analyze_negative_errors.py` (`reports/negative_errors.json`): for
   `not_recommended` and `low`, the reviews all three models get wrong with no known cause,
   against the reviews all three get right: what the models said, their rates, their length,
   and the words most typical of each group (log-odds ratio with an informative Dirichlet
   prior, counted once per review, words in at least 30 reviews). Report only.

Expected, before running:

- Of the `low` reviews all three get wrong, at least 60% have a rate of exactly 2 (against
  38% of all `low` reviews with a whole rate: 29,963 of 78,926), and no known cause drops
  from 36.5% to under 15%. "A rate of exactly 2" changes little for `not_recommended`.
- The unexplained `not_recommended` reviews are mostly called `no_idea` by all three, are
  longer than the ones all three get right, and their typical words are about delivery, the
  seller, packaging, price or the original/fake question rather than the product being bad,
  with contrast words ("ولی", "اما"); the typical words of the ones all three get right are
  plain verdicts ("اصلا", "پیشنهاد نمیکنم", "افتضاح").

## Result

`reports/reading_models.json` (the third run, commit 25b8e6c, with every cause above) and
`reports/negative_errors.json`. All three models scored their recorded Macro F1 on the whole
validation set and joined it one to one. On the 872,795 compared reviews (745,860 with a
recommendation label, 797,508 with a rating band), Macro F1 with offsets: FaBERT 0.7324 and
0.6921, BiLSTM 0.7175 and 0.6825, TF-IDF 0.7019 and 0.6682.

| | Recommendation | Rating band |
|---|---|---|
| All right | 604,117 (81.0%) | 577,201 (72.4%) |
| All wrong | 66,339 (8.9%) | 124,315 (15.6%) |
| Exactly one right | 29,869 (4.0%) | 40,425 (5.1%) |
| Exactly two right | 45,535 (6.1%) | 55,567 (7.0%) |
| Only FaBERT / only TF-IDF / only BiLSTM right | 10,650 / 11,595 / 7,624 | 13,453 / 17,444 / 9,528 |
| All wrong with the same wrong label | 59,722 (90.0%) | 117,190 (94.3%) |
| All wrong with no known cause | 17.1% | 8.9% |

The largest shared mistakes: `no_idea` called recommended (20,120) and `recommended` called
`no_idea` (19,069); `medium` called high (55,654) and high called medium (33,037). Share of
each true class that all three get wrong: `recommended` 4.0%, `not_recommended` 21.1%,
`no_idea` 32.6%; `low` 25.6%, `medium` 43.6%, high 6.8%.

Known causes, all wrong against all right: the buyer's two labels disagree in 39.1% against
11.2% (recommendation) and 51.9% against 4.7% (rating band); a rate of exactly 3 in 39.8%
against 11.7% and 46.3% against 8.6%. A text also in train and a short text are more common
among the reviews all three get right (repeated short texts are easy); emoji make no
difference.

By true class, among the reviews all three get wrong:

| True class | Main cause | No known cause |
|---|---|---|
| `medium` | rate exactly 3: 88.0%; buyer's labels disagree: 71.5% | 1.6% |
| high | rate exactly 4: 73.1% (38.8% of all high reviews with a whole rate) | 16.2% (55.1% before the rate-4 cause) |
| `low` | rate exactly 2: 56.5% (38% of all low reviews with a whole rate) | 18.6% (36.5% before the rate-2 cause) |
| `no_idea` | rate exactly 3: 53.3% | 5.6% |
| `recommended` | buyer's labels disagree: 42.8%; rate 3: 34.5%; rate 4: 30.9% | 19.5% |
| `not_recommended` | buyer's labels disagree: 33.5%; rate 3: 23.4%; rate 2: 22.7% | 34.4% (51.9% before the rate-2 cause) |

The unexplained negative reviews (`reports/negative_errors.json`; 5,197 `not_recommended`
and 3,795 `low`) are mostly rated 1 (3,015 and 3,279) and called `no_idea` or `medium` by all
three (3,449 and 2,688); they are about as long as the ones all three get right (median 11
against 10 words). Their most typical words are contrast and middle words: "ولی" (in 23.9%
against 6.7%), "اما", "خوبه", "عالی", "معمولی", "قیمت", "مناسب", "نسبت", "البته"; the
typical words of the ones all three get right are plain verdicts: "اصلا" (27.6% against
3.4%), "نمیکنم", "نخرید", "افتضاح", "مرجوع".

| | Expected | Found |
|---|---|---|
| All right / all wrong, recommendation | 78 to 82% / 8 to 10% | 81.0% / 8.9%: right |
| All right / all wrong, rating band | 70 to 74% / 14 to 17% | 72.4% / 15.6%: right |
| Same wrong label on "all wrong" | at least 70% | 90.0% and 94.3%: right |
| Only one right: FaBERT most, TF-IDF least | yes | TF-IDF most, BiLSTM least: wrong |
| Buyer's labels disagree, all wrong against all right | at least 3 times | 3.5 and 11 times: right |
| Rating band, all wrong with rate exactly 3 | at least half | 46.3%: wrong, a little under |
| No known cause, rating band / recommendation | under 40% / more | 8.9% / 17.1%: right |
| High all wrong with rate 4 / no cause | at least 70% / under 20% | 73.1% / 16.2%: right |
| Low all wrong with rate 2 / no cause | at least 60% / under 15% | 56.5% / 18.6%: both wrong |
| Rate 2 changes little for `not_recommended` | yes | no cause from 51.9% to 34.4%: wrong |
| Unexplained negatives: called `no_idea`, longer, about delivery, seller or price, with contrast words | | called `no_idea`: right; longer: no; delivery and seller words: no; price and contrast words: right |

What it means: three models that read the text in different ways make 90 to 94% of their
shared mistakes with the same wrong label, and those mistakes sit where the label and the
text disagree: a middle rate (3, or 4 for high, or 2 for low), or the buyer's two labels
pointing different ways. Even the mistakes without a measured cause are mostly mixed or
price-relative texts with a strict rating. Whether this is the text or a bias the three
models learned together from the same labels is tested by a reader that never saw the
labels (`reports/independent_reader_plan.md`), and whether it depends on one training run by
`reports/bilstm_plan.md` ("Two more seeds") and `reports/full_seeds_plan.md`.
