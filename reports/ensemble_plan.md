# Combined models, and the length of the input (written before running)

No change to the training of FaBERT has beaten the usual one (`reports/hard_examples_plan.md`,
`reports/difficulty_plan.md`, `reports/learning_rate_plan.md`). Two questions remain about the
model as a whole: whether combining models helps, and whether the input is cut too short.

## Length of the input

FaBERT reads at most 128 tokens of a review. The tokenizer comparison
(`reports/tokenizers/compare.json`, 50,000 training reviews) already counted the reviews that
are longer: with FaBERT's tokenizer 0.21% are longer than 128 tokens and 0.03% longer than 256
(median 7 tokens, 99th percentile 72). Even if every one of these reviews were wrong now and
right with a longer input, Macro F1 could move by about 0.2 point at most, and on average
less. No run is made for it; the limit of 128 stays.

## Combined models

`scripts/check_ensemble.py` (step 60, on your own machine) uses only saved validation
predictions; nothing is trained and the test set is not read.

1. **FaBERT and TF-IDF together.** For each head, the mean of the two models' probabilities,
   with weight w for FaBERT and 1 - w for TF-IDF (w = 0, 0.1, ..., 1). The weight and the
   offsets are chosen on one half of validation by product and scored on the other, and the
   other way round, then compared with FaBERT alone on the same reviews (paired bootstrap).
   TF-IDF is three points of Macro F1 lower, but it reads the text in another way and may be
   wrong on other reviews.
2. **Three FaBERT seeds together.** The three pilot runs of step 41 (500,000 reviews, seeds
   2026 to 2028): the mean of their probabilities against each seed alone, all with
   cross-fitted offsets. This tells whether training several final models would pay.

The rule: a combination is used only if, for a head, the whole 95% interval of the difference
is above zero. Combining FaBERT with TF-IDF costs little at prediction time (TF-IDF is fast);
several FaBERT models cost their training time again and as much again at prediction time.

Expected:

- FaBERT and TF-IDF: weight 0.7 to 0.9 for FaBERT on both halves; Macro F1 0.2 to 0.8 point
  higher than FaBERT alone for each head, with the whole interval above zero.
- Three seeds: 0.3 to 0.8 point above each seed alone for each head, the interval above zero
  for at least two of the three seeds.

## Result

`reports/ensemble.json`, all of validation.

FaBERT and TF-IDF together, against FaBERT alone:

| Head | FaBERT alone | Together | Difference [95%] | FaBERT weight (each half) | Expected |
|---|---|---|---|---|---|
| recommendation | 0.7324 | 0.7344 | +0.0020 [+0.0014, +0.0026], better | 0.7 and 0.7 | 0.2 to 0.8 point, weight 0.7 to 0.9: right, at the low end |
| rating band | 0.6922 | 0.6935 | +0.0013 [+0.0009, +0.0018], better | 0.8 and 0.7 | right for the weight; the gain is below 0.2 point: wrong, smaller |

F1 per class, FaBERT alone -> together: `recommended` 0.944 -> 0.943, `not_recommended` 0.744 ->
0.749, `no_idea` 0.510 -> 0.511; `low` 0.711 -> 0.714, `medium` 0.479 -> 0.479, `high` 0.887 ->
0.888. The gain comes mostly from `not_recommended` and `low`.

Three pilot seeds together, against each seed alone:

| Head | Seeds alone | Together | Differences [95%] | Expected |
|---|---|---|---|---|
| recommendation | 0.7235, 0.7237, 0.7241 | 0.7223 | -0.0012 [-0.0019, -0.0005], -0.0013 [-0.0020, -0.0007], -0.0018 [-0.0025, -0.0011]: worse than every seed | 0.3 to 0.8 point better: wrong |
| rating band | 0.6858, 0.6867, 0.6864 | 0.6872 | +0.0014 [+0.0009, +0.0018], +0.0005 [-0.0000, +0.0010], +0.0008 [+0.0003, +0.0013] | 0.3 to 0.8 point better: wrong, smaller |

What it means:

- FaBERT and TF-IDF together beat FaBERT alone on both heads, with the whole interval above
  zero, so by the rule the combination is used. The gain is small (0.2 and 0.1 point), and it
  costs only the TF-IDF model at prediction time, which is fast.
- Three copies of FaBERT trained with other seeds do not help: they make nearly the same
  mistakes, and for recommendation the mean of their probabilities is even slightly worse.
  Why it is worse was not measured; one possible reason is that the offsets are searched in
  steps of 0.1, which fits the smoother mean less well. Several final FaBERT models are not
  trained.
- Kept: the final system combines FaBERT and TF-IDF. The weight and the offsets used on the
  test set and in the service are fitted on all of validation when the test set is scored,
  as the offsets of each model were; the temperature of FaBERT
  (`reports/calibration_plan.md`) is applied before combining, and the combined probabilities
  are checked for calibration again at that point.
