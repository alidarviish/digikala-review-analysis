# The main model decisions again, with product-level intervals (written before running)

## Why

Every comparison between models so far used a paired bootstrap that draws reviews one by
one (`bootstrap_difference` in `src/digikala_reviews/metrics.py`). The label analyses since
then showed that reviews of the same product are not independent (843,458 reviews with a
band come from 48,755 products), and from them on the intervals draw products. A
review-level interval is too narrow when reviews cluster, so some of the small differences
that decided the final system may not hold. This check recomputes those decisions with the
same predictions, nothing trained again.

## Comparisons (`scripts/recheck_decisions.py`, `reports/decision_recheck.json`)

On the whole validation set, as the decisions were made, for each head, Macro F1 with
offsets, paired, drawing products with replacement (a review's weight is the number of times
its product was drawn), 1,000 rounds, seed 2026:

| Decision | Variant | Reference | Recorded (review-level) |
|---|---|---|---|
| D58 final system | FaBERT + TF-IDF | FaBERT | +0.0020 [+0.0014, +0.0026]; +0.0013 [+0.0009, +0.0018] |
| D65 best single model | further pretrained FaBERT | FaBERT | +0.0012 [+0.0004, +0.0021]; +0.0009 [+0.0002, +0.0016] |
| D65 final system unchanged | further pretrained FaBERT + TF-IDF | FaBERT + TF-IDF | +0.0001 [-0.0007, +0.0008]; +0.0008 [+0.0002, +0.0015] |
| model order | FaBERT | TF-IDF | |
| model order | BiLSTM | TF-IDF | |
| model order | FaBERT | BiLSTM | |

The combinations are rebuilt exactly as in steps 60 and 75 (`cross_fitted_blend`: weight and
offsets tuned on the other product half). Before any interval is computed, every model and
combination must score the Macro F1 recorded for it, and every set of predictions must join
validation one to one by id. The test set is not read.

## Rule (fixed now)

A decision that needed an interval above zero keeps standing if the product-level interval
is still above zero. If it now includes zero, it is marked a tie; by the project's rule a tie
goes to the simpler or usual setup, and what that changes is written in the decision log and
agreed before anything else changes.

## Expected

- Product-level intervals are 1.3 to 2 times as wide as the review-level ones.
- D58 stands on both heads (the recorded intervals start well above zero).
- D65, further pretrained FaBERT alone: the rating band interval includes zero; recommendation
  stays just above zero.
- D65, combinations: still no measurable difference on recommendation; rating band now
  includes zero too, so the final system stays as it is either way.
- The order FaBERT > BiLSTM > TF-IDF holds on both heads with intervals well above zero.

## Result

`reports/decision_recheck.json`, commit d54d70f. Every model and combination scored its
recorded Macro F1 first. Differences with product-level 95% intervals (recommendation; rating
band):

| Decision | Variant against reference | Recommendation | Rating band | Still stands |
|---|---|---|---|---|
| D58 final system | FaBERT + TF-IDF against FaBERT | +0.0020 [+0.0014, +0.0026] | +0.0013 [+0.0009, +0.0018] | yes |
| D65 best single model | further pretrained against FaBERT | +0.0012 [+0.0004, +0.0021] | +0.0009 [+0.0003, +0.0016] | yes |
| D65 final system unchanged | further pretrained + TF-IDF against FaBERT + TF-IDF | +0.0001 [-0.0008, +0.0008] | +0.0008 [+0.0002, +0.0015] | yes (as before: a tie on recommendation) |
| model order | FaBERT against TF-IDF | +0.0308 [+0.0295, +0.0320] | +0.0239 [+0.0229, +0.0250] | yes |
| model order | BiLSTM against TF-IDF | +0.0159 [+0.0146, +0.0172] | +0.0143 [+0.0132, +0.0153] | yes |
| model order | FaBERT against BiLSTM | +0.0149 [+0.0137, +0.0161] | +0.0096 [+0.0087, +0.0105] | yes |

No decision changes.

| | Expected | Found |
|---|---|---|
| Product-level intervals 1.3 to 2 times as wide | yes | about the same width as the review-level ones: wrong |
| D58 stands | yes | right |
| D65 alone: rating band includes zero | yes | it stays above zero: wrong |
| D65 combinations: rating band includes zero | yes | it stays above zero: wrong |
| Order FaBERT > BiLSTM > TF-IDF, far from zero | yes | right |

Why the intervals did not widen: in a paired comparison both models are scored on the same
drawn products, so what a product does to one model it mostly does to the other, and it
cancels in the difference. Clustering by product widens the interval of a share or of one
model's score (as in the label analyses), not of the difference between two models on the
same reviews. What these intervals still leave out is the chance in training itself (one run
per model), measured in `reports/full_seeds_plan.md`.
