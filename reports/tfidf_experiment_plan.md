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
