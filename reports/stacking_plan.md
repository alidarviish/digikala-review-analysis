# Stacking: both models and both heads together (written before running)

The final system combines FaBERT and TF-IDF with a weighted mean of their probabilities, for
each head on its own (`reports/ensemble_plan.md`). The two heads are strongly related: a review
that does not recommend the product usually has a low rating, and one with a medium rating is
often `no_idea`. Each head now decides without looking at the other.

## Method

`scripts/check_stacking.py` (step 63, on your own machine). For each head a logistic
regression reads 12 numbers per review: the log probabilities of FaBERT and TF-IDF for both
heads (floored at -20). It is fitted on one half of validation by product, its offsets are
searched on the same half, and it predicts the other half; then the other way round. With 12
inputs and about 400,000 reviews per half, the small model cannot learn much by heart, so its
offsets are searched on the half it was fitted on. It is compared on the same reviews with the
weighted mean of the two models for that head (refitted the same way) and with FaBERT alone,
by the paired bootstrap. Only saved validation predictions are used; the test set is not read.

The rule: for a head, stacking replaces the weighted mean only if the whole 95% interval of the
difference to the weighted mean is above zero. Both need the same two models at prediction
time, so the cost is the same.

## Expected

- Rating band: 0.1 to 0.5 point above the weighted mean, the interval above zero; the
  recommendation probabilities help to tell `low` from `medium`.
- Recommendation: 0 to 0.3 point above the weighted mean; it may tie.
- No class loses more than 0.5 point of F1.
