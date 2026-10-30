# TookaBERT-Large trained like the kept FaBERT model (written before running)

FaBERT was chosen over TookaBERT-Large on the pilot (`reports/transformer_model_choice.md`).
With 100,000 training reviews the large model led by 0.26 point, more than the spread between
seeds; with 500,000 reviews it led on every seed but only by 0.12 point (0.7058 against
0.7046), just below the spread, so by the rule it was a tie and the faster model was kept.
Whether the gap grows with all 3.55 million training reviews and four epochs was never
measured. This run measures it.

## Method

Step 61 (`scripts/train_transformer.py --large`, GPU) trains TookaBERT-Large exactly like the
kept FaBERT run (step 33): every labeled training review, four epochs, four checks per epoch
on the same 50,000 validation reviews, the best check kept, everything after training scored
with the kept weights. Only the model and its learning rate change (2e-5, as in the pilot;
3e-5 was the best of three for FaBERT, `reports/learning_rate_plan.md`, and was not tried for
the large model). FaBERT's run took 121 minutes; the large model was 2.5 times slower on the
pilot, so about 5 hours are expected. A resume point is saved at every check.

Step 62 (`scripts/compare_fabert_runs.py --large`, on your own machine) compares the two on
the whole validation set, both with cross-fitted offsets, per head with the paired bootstrap.

## Rule, fixed before the run

TookaBERT-Large replaces FaBERT only if both of these hold:

1. on both heads the whole 95% interval of the difference is above zero, and
2. the mean gain over the two heads is at least 0.3 point of Macro F1.

The second condition is there because the large model costs about three times as much at
prediction time and in memory, and 2.5 times the training time; a gain smaller than the
largest gain any other change brought (0.2 point, combining with TF-IDF) is not worth that.
If it replaces FaBERT, the calibration, the combination with TF-IDF and the analyses are made
again for it. Otherwise FaBERT stays, and the large model is reported as an upper bound.

## Expected

- The kept check is in epoch 2 to 3.5, as for FaBERT (2.75).
- The large model is 0.1 to 0.3 point better on each head, with the whole interval above zero
  on both (the validation set is large enough to measure 0.1 point).
- The mean gain is below 0.3 point, so FaBERT stays.
- Training takes 4.5 to 6 hours.
