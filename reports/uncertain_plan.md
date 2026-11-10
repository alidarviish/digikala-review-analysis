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
