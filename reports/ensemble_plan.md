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
