# A chooser with more information (written before running)

## The question

On about a tenth of the validation reviews at least one of the three final models is right
and another wrong; about half of the perfect-chooser room this gives is chance, and about 4
points on each head comes from the models really making different mistakes
(`reports/error_overlap_plan.md`, steps 71 and 72). The weighted mean recovers 0.2 point of
it, and stacking with a linear model on the probabilities recovered nothing
(`reports/stacking_plan.md`). Can a non-linear model that also sees the length of the text and
whether the models agree learn when to trust which model?

## Method

Step 73, `scripts/check_chooser.py`, on your own machine. For each head, gradient-boosted
trees (scikit-learn's `HistGradientBoostingClassifier`: at most 300 rounds, learning rate 0.1,
early stopping on 10% of the fitting data) learn the class from:

- the log probabilities of the models for both heads, floored at -20;
- the length of the text in words and in characters (the text is the only model input, so
  its length is allowed);
- for each pair of models, whether their most likely classes for this head agree.

The chooser is fitted on one half of validation by product and predicts the other half, then
the other way round. Its offsets are searched on predictions it did not fit: within the
fitting half, a second split by product gives every review a prediction from a model that did
not see it. A test on made-up data checks that it learns "on short texts trust model A, on
long texts model B", which the probabilities alone cannot show.

Two versions, each against the kept combination of FaBERT and TF-IDF (step 60) on the same
reviews:

1. FaBERT and TF-IDF only. Rule: it replaces the kept combination if the whole 95% interval is
   above zero on both heads.
2. All three final models. Large costs about three times as much at prediction time, so, as in
   `reports/large_final_plan.md`: the interval above zero on both heads and a mean gain of at
   least 0.3 point.

XGBoost is tried only if a version gains, to check that the result does not depend on one
implementation. Nothing outside validation is read; the test set is not read.

## Expected

- FaBERT and TF-IDF: 0 to 0.3 point above the kept combination.
- All three models: 0.2 to 0.5 point above it.
- No class loses more than 0.5 point of F1.
- 20 to 60 minutes in all (four versions and heads, six fits each).

## Result

`reports/chooser.json`, all of validation, against the kept combination of FaBERT and TF-IDF
(0.7344 and 0.6935).

| Head | Version | Chooser | Difference [95%] | Expected |
|---|---|---|---|---|
| recommendation | FaBERT and TF-IDF | 0.7325 | -0.0019 [-0.0024, -0.0013], worse | 0 to 0.3 point better: wrong |
| recommendation | three models | 0.7347 | +0.0003 [-0.0005, +0.0010], no measurable difference | 0.2 to 0.5 point better: wrong |
| rating band | FaBERT and TF-IDF | 0.6940 | +0.0005 [+0.0000, +0.0009], better | 0 to 0.3 point: right, at the low end |
| rating band | three models | 0.6957 | +0.0021 [+0.0016, +0.0027], better | 0.2 to 0.5 point: right, at the low end |

F1 per class, kept -> chooser: FaBERT and TF-IDF: `recommended` 0.943 -> 0.944,
`not_recommended` 0.749 -> 0.746, `no_idea` 0.511 -> 0.507; `low`, `medium`, `high` within
0.001. Three models: `recommended` 0.943 -> 0.946, `not_recommended` 0.749 -> 0.746,
`no_idea` 0.511 -> 0.513; `low` 0.714 -> 0.718, `medium` 0.479 -> 0.482, `high` 0.888 ->
0.887. No class moved by more than 0.5 point, as expected. Each version took about one
minute, not the 20 to 60 expected (early stopping ends the fits early).

What it means:

- Neither version passes its rule: FaBERT and TF-IDF lose on recommendation, and the three
  models gain nothing measurable there.
- The chooser does no better than the plain averages. On recommendation it loses about 0.2
  point against the weighted mean, as the linear stacking did (-0.0020); with three models it
  ends below the plain mean of the three (0.7367, step 71) on recommendation and level with
  it on the rating band (0.6959). The text length and the agreement between the models do
  not tell it which model is right.
- The room measured in steps 71 and 72 is real, but these signals cannot reach it.
- Kept: the weighted mean of FaBERT and TF-IDF. XGBoost is not tried, since no version gained.
