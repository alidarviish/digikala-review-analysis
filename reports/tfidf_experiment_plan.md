# TF-IDF experiments: plan written before the results

This plan was written and committed while the first tuning run (stages 1 and 2 of
`scripts/tune_tfidf.py`) was still running, before any of its scores were seen. Writing
the questions and the expected outcomes down first keeps the reading of the results honest:
an experiment that changes nothing is still a result, and nothing is chosen after the fact
because it happened to look good.

## Reference model

For each head, the converged run with the highest validation Macro F1 among stages 1 and 2
(ties go to fewer features, then to no class weights). Every experiment below changes
exactly one thing in that model and is scored on the same validation reviews.

## How a difference is judged

- The score is validation Macro F1. Test is not used until the final model is fixed.
- Each experiment is compared with the reference on the same reviews, with a paired
  bootstrap interval (95%) for the difference in Macro F1.
- A change is **adopted** only if the whole interval is above zero. If the interval
  contains zero, the result is "no measurable difference" and the simpler model is kept.
- If several changes are adopted, the combination is fitted and checked once more before
  it becomes the final model.

## Experiments

| # | Question | Why we ask (evidence from the data audit) | Expected outcome |
|---|---|---|---|
| E1 | Do per-class decision offsets help Macro F1? | The small classes (`not_recommended` 8.2%, `no_idea` 9.7%, `low` 9.1% of all reviews) tend to be under-predicted by a model trained for overall likelihood | +1 to +3 points |
| E2 | Do character n-grams next to word n-grams help? | Informal spellings (`پشیمونم`) and joined forms (`وکاربردی`) are common and split the counts of one word over several forms (section 6 of notebook 01); character n-grams alone scored 3 points lower (`reports/memory_measurements.json`), but may add to words | 0 to +1 point |
| E3 | Does removing the most frequent words help? | Negation words appear in 42.2% of `not_recommended` reviews against 8.8% of `recommended` ones, and frequent words such as `خیلی` ("very") and `ولی` ("but") carry meaning (section 6 of notebook 01) | Worse, or no difference |
| E4 | Does an ordinal model for the rating band reduce large mistakes? | Logistic regression does not know that low < medium < high; two binary models ("above low?", "above medium?") use that order | Fewer two-step errors (low vs high); Macro F1 about the same |
| E5 | Does sublinear term frequency matter? | Reviews are short (median 6 words), so repeated words are rare | Small or no difference |
| E6 | How do other linear models compare? | The project asks for logistic regression as the baseline; a linear SVM and Complement Naive Bayes show how much room the classic approach has | SVM close to LR; Naive Bayes lower |
| E7 | What if a rating of 0 counted as `low`? | Section 4 of notebook 01 found that 0 means "no rating recorded" | `low` gets mixed with positive reviews and its F1 on real low ratings drops |

## Details

- **E1.** Validation is split into two halves by product. Offsets added to the log
  probabilities are searched on one half and scored on the other, then the halves are
  swapped, so the offsets are never scored on the reviews they were tuned on. If adopted,
  the offsets used for the test set are searched on the whole validation set.
- **E2.** Character n-grams (`char_wb`, 3 to 4 characters, seen in at least 5 reviews) are
  joined with the reference word features. If memory runs out on Colab (12 GB), the run
  moves to a larger machine or the character features get a cap, and the report says so.
- **E3.** The 100 words that appear in the most training reviews are removed. The list is
  saved with the result, together with which negation words it contains.
- **E4.** Frank and Hall's method: P(low) = 1 - P(above low), P(medium) = P(above low) -
  P(above medium), P(high) = P(above medium). Scored with Macro F1 and the ordinal distance.
- **E5.** Plain term frequency instead of `1 + log(tf)`.
- **E6.** `LinearSVC` with C in {0.1, 1} and `ComplementNB` with alpha in {0.1, 1}.
  Logistic regression stays the reported baseline whatever the outcome.
- **E7.** Report only; the decision on zero ratings is not changed by this experiment. The
  model is trained with zero-rated reviews labeled `low` and scored twice on validation:
  with zeros labeled `low`, and on the reviews with a real rating only. Training reviews
  that have neither a recommendation nor a rating band are not in `train_supervised.csv`,
  so about 17,700 zero-rated reviews are missing from this variant; the report notes it.

## Experiments added after stage 1

E8 to E12 were added after the stage 1 scores and the first stage 3 result (E1) were seen.
They come from a discussion with classmates about spelling, conflicting labels, how the
test set should be chosen and how much data is needed. Their expected outcomes were written
down here before any of them was run. They run on a second machine (molab, 4 CPUs and
32 GB of memory, see `notebooks/molab_tfidf_extra.py`) next to stage 3, so the reference
models are fitted again there; if they do not reproduce the stage 2 scores (0.6802 and
0.6459) to three decimals, the report says so before any result is read.

None of them reads the test set. E9 and E10 build their own splits, but only out of the
reviews that are already in train and validation.

| # | Question | Why we ask | Expected outcome |
|---|---|---|---|
| E8 | Does fixing frequent misspellings help? | Misspellings split one word over several features (`پشنهاد`, `پیسنهاد` and `پیشنهد` for `پیشنهاد`), and `و` joined to the next word (`وقیمت`) hides a frequent word | No measurable difference: the list changes about 2% of the reviews, in similar shares across classes |
| E9 | How much higher is the score with a random split instead of a split by product? | The model never sees the product, but reviews of one product share names and complaints, so a random split may reward remembering products | Random split higher, by less than 1 point: most reviews are short and general, and 37% of the test texts already appear in train with the split by product |
| E10 | How does the model do on the newest reviews when it is trained only on older ones? | A deployed model always meets reviews written after it was trained; the reviews carry a date (1395 to 1402) | Lower than a split by product of the same size, by 1 to 2 points |
| E11 | Does removing conflicting labels from the training reviews help? | The same text sometimes comes with different labels (`عالی`: about 94,000 recommended, 293 not recommended) | No measurable difference: on such texts the model already predicts the majority label |
| E12 | How much training data is needed? | Most reviews are short and repeat the same phrases, so far fewer than 3.5 million may be enough | Large gain from 10,000 to 100,000 reviews, less than 1 point from 1,000,000 to all |

### Details of E8 to E12

- **Reference.** For each head the stage 3 reference: features `word123_min5_persian`;
  recommendation with balanced class weights and C = 0.5, rating band without weights and
  C = 1. Fitted on train and scored on validation as in stage 2.
- **E8.** The list is `resources/spelling_fixes.csv` (590 rows marked `fix`). It was built
  from word counts on the training reviews only: a word seen in at least 30 reviews, not in
  the hazm word list even after removing common prefixes and suffixes, one letter away from
  a word seen at least 5 times more often. Every row was checked by hand, with example
  sentences for the unclear ones; spoken forms (`قیمتشم`, `اینا`) and formal verbs (`دارد`)
  are left alone. Only whole words are replaced, after normalization, in train and
  validation alike. Compared with the reference on the same validation reviews; adopted only
  if the whole interval is above zero, and then the same step runs before every model.
- **E9 and E10.** Report only. The pool is train plus validation. The validation share is
  the share of reviews from the newest year (1402) in the pool, so that all three splits
  have about the same size: (a) by product: whole products drawn at random (seed 2026) for
  validation; (b) random: single reviews drawn at random (seed 2026); (c) by time: every
  review from 1402 in validation, everything older in train. Each split gets its own
  features (fitted on its own training part) and the reference model settings. Reported for
  each: Macro F1 with its bootstrap interval, the sizes, and the share of validation texts
  that also appear in its training part. E9 reads (b) against (a), E10 reads (c) against
  (a). The validation sets differ, so there is no paired interval; the gap is read next to
  the two intervals.
- **E11.** For each head, training texts that occur more than once with more than one label
  keep only the rows with the most common label of that text; a text whose top labels are
  tied is kept whole. The removed rows are counted in the report. Same features and model
  settings as the reference; scored on the untouched validation reviews and compared with
  the reference.
- **E12.** Report only. Random samples of 10,000, 100,000 and 1,000,000 labeled training
  reviews of each head (seed 2026, each sample inside the next); all of them is the
  reference. Features are fitted on the sample itself, as they would be for a smaller data
  set. Scored on the full validation set, with a bootstrap interval.

## The final model (written after stages 3 and E8 to E12, before it was fitted)

Adopted changes, each with its whole interval above zero against the reference:

| Head | Adopted | Not adopted |
|---|---|---|
| recommendation | E1 offsets (+0.0217), E11 conflicts removed (+0.0027), E8 spelling fixes (+0.0009) | E2, E3, E5 (no difference or worse); E6 and E7 are report only |
| rating band | E1 offsets (+0.0217), E11 conflicts removed (+0.0014), E2 character n-grams (+0.0017) | E3, E4, E5, E8 (no difference or worse); E6 and E7 are report only |

`scripts/final_tfidf.py` fits each head with all of its adopted changes together and scores
it on validation against the reference, refitted on the same machine. Three comparisons are
reported: the final model against the reference, against the reference with offsets (does
anything add to E1?), and the final model without offsets against the reference. Training
scores on 300,000 training reviews are reported next to validation scores. Expected
outcome: close to the sum of the single gains, about +2.5 points for each head against the
reference, and less than 0.5 point against the reference with offsets. Logistic regression
stays the model (decision D48): the linear SVM of E6 did not converge and was scored
without offsets.
