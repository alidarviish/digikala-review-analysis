# Calibration of the final FaBERT model (written before running)

## Why

The model gives a probability with every answer. Two later steps use these probabilities as
they are: the semi-supervised step keeps a guess on an unlabeled review only when its
probability is at least 0.9 (`PSEUDO_LABEL_THRESHOLD`), and the service will return them to
the user. A probability of 0.9 should then mean right about 9 times out of 10. The leave-out
runs (`reports/difficulty_plan.md`) showed that a model can keep its Macro F1 while its
probabilities get worse, so this is measured on its own.

## Method

Temperature scaling: for each head, the log probabilities are divided by one number T and
normalized again. T above 1 makes the model less sure, below 1 more sure. It never changes
which class comes first, so Macro F1 without offsets stays exactly the same; with offsets it
can move a little, since the offsets are searched again on the scaled probabilities.

`scripts/calibrate.py` (step 55, on your own machine) reads the validation predictions of the
final FaBERT model (`artifacts/transformer_no_early_stop`). T is fitted (lowest mean loss) on
one half of validation by product and scored on the other, and the other way round, as for
the offsets. Before and after it reports the mean loss, the Brier score, the expected
calibration error (ECE: the gap between the top probability and the share right, in 15 bins,
weighted by the reviews in each bin), and how many reviews reach 0.9 and how often they are
right. The T fitted on all of validation is kept for the semi-supervised step and the
service, as the offsets are. Nothing is trained and the test set is not read.

Checked on made-up data first: a model made too sure by a factor of 1.8 gets T = 1.78, and
its ECE falls from 0.118 to 0.018.

## Expected

- The final model was kept at its best check, where its validation loss was lowest, so it is
  only a little too sure: T between 1.0 and 1.3 for both heads, about the same on the two
  halves (within 0.05).
- ECE before 0.01 to 0.04, after below 0.01.
- Mean loss lower by less than 0.01 for each head.
- Among reviews with a top probability of at least 0.9: right in 1 to 4 points fewer cases
  than their mean probability before, within 1 point after; their share of all reviews falls.
- Macro F1 without offsets unchanged; with offsets within 0.002.

If T is close to 1 and the ECE already small, the probabilities are kept as they are and the
step is recorded as a check that passed.

## Result

`reports/calibration.json`, `reports/figures/results_calibration.png`. All of validation,
temperature fitted on one half by product and scored on the other.

| | Recommendation | Rating band | Expected |
|---|---|---|---|
| Temperature (halves) | 1.124 (1.127, 1.121) | 1.108 (1.108, 1.108) | 1.0 to 1.3, halves within 0.05: right |
| ECE before / after | 0.0162 / 0.0055 | 0.0240 / 0.0046 | 0.01 to 0.04, then below 0.01: right |
| Mean loss before / after | 0.3121 / 0.3095 | 0.4925 / 0.4898 | lower by less than 0.01: right |
| Brier before / after | 0.1720 / 0.1712 | 0.2812 / 0.2801 | |
| Reviews with p >= 0.9, before | 72.3%, 96.9% right, mean p 97.7% | 45.6%, 92.9% right, mean p 94.4% | right 1 to 4 points below the mean p: wrong for recommendation (0.8 below), right for rating band (1.5) |
| The same, after | 69.1%, 97.3% right, mean p 97.0% | 37.2%, 93.9% right, mean p 93.6% | within 1 point, and fewer reviews: right |
| Macro F1 without / with offsets, before | 0.7253 / 0.7324 | 0.6799 / 0.6922 | |
| The same, after | 0.7253 / 0.7327 | 0.6799 / 0.6923 | unchanged / within 0.002: right |

What it means:

- The final model is a little too sure, more so for the rating band, and one number per head
  fixes nearly all of it: after scaling, the probability is within half a point of how often
  the model is right in every bin that holds at least 0.1% of the reviews.
- The scores do not change. What changes is what the probabilities can be used for: at the
  pseudo-label threshold of 0.9, the rating band keeps 37% of the reviews instead of 46%, and
  those it keeps are right more often (93.9% against 92.9%).
- Kept: the temperatures fitted on all of validation (recommendation 1.124, rating band 1.108)
  are applied to the probabilities in the semi-supervised step and in the service, before the
  offsets.
