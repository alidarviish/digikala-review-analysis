# Saying "uncertain" instead of guessing (written before running)

## Why

Part of the mistakes cannot be removed by any model that reads only the text: the same text
carries different labels, and a reader that never saw the labels agrees with the models, not
the label, on many of them (`reports/text_ceiling_plan.md`, `reports/independent_reader_plan.md`).
A product can still use the system well if it says "uncertain" where it is least sure, instead
of guessing. This measures how much that helps, before the test set is read.

## Method (`scripts/uncertain_output.py`, `reports/uncertain.json`)

- The final system (D58): the weighted mean of FaBERT's and TF-IDF's probabilities, then the
  per-class offsets. Its answer for a review is the class chosen with the offsets; its
  **confidence** is the mixed probability of that class.
- The 10% of reviews with the lowest confidence are answered "uncertain" (the share chosen
  with the user before running).
- Cross-fitted, as everything else: on each product half of validation, the weight, the
  offsets and the confidence threshold (the 10th percentile) are fitted on the other half, so
  no review is judged with a threshold fitted on it. For the test set and the offline script,
  the same three are fitted once on all of validation and saved.
- Reported for each head, on the whole validation set: the share answered "uncertain"; accuracy,
  Macro F1 and each class's recall on the answered reviews against all reviews; the true
  classes of the "uncertain" reviews against all reviews, and how often the system would have
  been right on them; and the coverage curve: accuracy and Macro F1 when only the most
  confident 50%, 55%, ..., 100% are answered.

The test set is not read.

## Expected

- About 10% answered "uncertain" (9.5 to 10.5%, since the threshold comes from the other half).
- At 90% coverage, accuracy rises from 87.4% to 91 to 92% for recommendation and from 79.5% to
  83 to 84% for rating band.
- The "uncertain" reviews are rich in the hard classes: `no_idea` and `medium` at least three
  times their share among all reviews; the system would have been right on under 55% of them.
- Macro F1 on the answered reviews rises less than accuracy, since many of the hard-class
  reviews it keeps are still wrong.

## Result

`reports/uncertain.json` (commit d324ed2). The system's labels give exactly step 60's Macro F1
on both heads (checked in the script). The fitted weight was the same on both product halves for
recommendation (0.7); for rating band it was 0.8 on one half and 0.7 on the other. Saved for the
test set, fitted on all of validation: recommendation weight 0.7, threshold 0.5873; rating band
weight 0.8, threshold 0.4929.

| | Recommendation | Rating band |
|---|---|---|
| Answered "uncertain" | 10.0% (78,888) | 9.9% (83,902) |
| Accuracy: all, answered | 0.8733, 0.9188 | 0.7957, 0.8407 |
| Macro F1: all, answered | 0.7344, 0.7633 | 0.6935, 0.7078 |
| Recall, all to answered | recommended 0.937 to 0.973, not_recommended 0.732 to 0.865, no_idea 0.543 to 0.429 | low 0.685 to 0.766, medium 0.468 to 0.346, high 0.898 to 0.962 |
| Among the "uncertain", times their share | recommended 0.52, not_recommended 2.22, no_idea 3.35 | low 1.38, medium 2.03, high 0.67 |
| The system right on the "uncertain" | 46.4% | 38.8% |

Coverage curve, accuracy (the most confident share answered):

| Coverage | 50% | 60% | 70% | 80% | 90% | 95% | 100% |
|---|---|---|---|---|---|---|---|
| Recommendation | 0.983 | 0.977 | 0.969 | 0.952 | 0.919 | 0.898 | 0.873 |
| Rating band | 0.924 | 0.910 | 0.894 | 0.874 | 0.841 | 0.820 | 0.796 |

| | Expected | Found |
|---|---|---|
| Share answered "uncertain" | 9.5 to 10.5% | 10.0% and 9.9%: right |
| Accuracy at 90% coverage, recommendation | 91 to 92% | 91.9%: right |
| Accuracy at 90% coverage, rating band | 83 to 84% | 84.1%: wrong, a little above |
| `no_idea` at least 3 times its share | yes | 3.35: right |
| `medium` at least 3 times its share | yes | 2.03: wrong |
| The system right on under 55% of the "uncertain" | yes | 46.4% and 38.8%: right |
| Macro F1 rises less than accuracy | yes | +0.029 against +0.046, and +0.014 against +0.045: right |

What it means:

- Saying "uncertain" on 10% of reviews sets aside 42% of the mistakes on recommendation and
  30% on rating band: accuracy rises 4.5 points on both heads, and the system would have been
  wrong on more than half of the reviews it sets aside.
- The cost falls on the hard classes. The "uncertain" reviews are rich in `no_idea` and
  `medium`, and many of them were answered right: the recall of `no_idea` among the answered
  reviews falls from 0.543 to 0.429, and of `medium` from 0.468 to 0.346. So "uncertain" is
  close to a fourth answer, "mixed or unclear opinion", rather than a filter that keeps every
  class equally well; Macro F1 rises much less than accuracy.
- `medium` is less concentrated than expected (2.03 times its share, against `low` 1.38): the
  rating band is unsure across the low and medium boundary, not only on `medium`.
- For a product: at 80% coverage the recommendation head is right 95.2% of the time, at 50%
  98.3%; the threshold is chosen on validation and is read once on test, in the final
  evaluation.
