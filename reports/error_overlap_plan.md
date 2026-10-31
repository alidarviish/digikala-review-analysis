# Do the final models make different mistakes? (written before running)

## The question

Combining models helps only where they make different mistakes. FaBERT and TF-IDF together
gained 0.1 to 0.2 point; three FaBERT seeds together gained nothing, and stacking gained
nothing (`reports/ensemble_plan.md`, `reports/stacking_plan.md`). Before looking for more
models to combine, this measures how much room there is: on how many reviews are all the
final models wrong, where no combination can help, and how high could a perfect choice
between them score?

## Method

Step 71, `scripts/check_error_overlap.py`, on your own machine, from the saved validation
predictions of the final FaBERT, TookaBERT-Large (`reports/large_final_plan.md`) and TF-IDF
models, each with its cross-fitted offsets. For each head:

1. The share of reviews all three get right, all three get wrong, and the rest (some right),
   overall and per true class; for each pair, the share of either model's mistakes that both
   make.
2. The perfect-chooser bound: the true label wherever at least one model has it, otherwise
   FaBERT's label. No real combination of these three can score above its Macro F1.
3. FaBERT and Large averaged, then combined with TF-IDF exactly as in step 60, against the
   kept combination of FaBERT and TF-IDF on the same reviews.

Rule for 3: Large costs about three times as much as FaBERT at prediction time, so, as in
`reports/large_final_plan.md`, the three models replace the kept combination only if the
whole 95% interval is above zero on both heads and the mean gain is at least 0.3 point.

Nothing is trained and the test set is not read.

## Expected

- Recommendation: about 85% of reviews all right, 8 to 10% all wrong, 5 to 7% some right.
  Of the reviews labeled `no_idea`, 40 to 55% are wrong in all three models (the label and
  the reading of the text differ there, `reports/label_sample_plan.md`).
- FaBERT and Large share 70 to 80% of their mistakes; FaBERT and TF-IDF 55 to 70%.
- The perfect chooser is 0.03 to 0.08 above the kept combination for each head: room exists
  only on paper, since no real combination knows which model is right.
- The three models gain 0 to 0.2 point over FaBERT and TF-IDF, below the 0.3 point asked
  for, so the kept combination stays.
