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

## Result

`reports/error_overlap.json`, all of validation.

| | Recommendation | Rating band |
|---|---|---|
| All three right | 81.4% | 72.9% |
| All three wrong | 9.0% | 15.8% |
| Some right | 9.6% | 11.3% |
| All wrong, per true class | recommended 4.1%, not_recommended 20.9%, no_idea 33.1% | low 25.1%, medium 44.0%, high 7.1% |
| Some right, per true class | recommended 6.0%, not_recommended 18.8%, no_idea 27.1% | low 16.6%, medium 19.8%, high 8.3% |
| Both wrong, share of either wrong: FaBERT & Large / FaBERT & TF-IDF / Large & TF-IDF | 77.1% / 55.6% / 54.7% | 81.0% / 64.3% / 65.4% |
| Macro F1: FaBERT / Large / TF-IDF | 0.7324 / 0.7324 / 0.7017 | 0.6922 / 0.6938 / 0.6683 |
| FaBERT and TF-IDF (kept) | 0.7344 | 0.6935 |
| Perfect chooser | 0.8080 | 0.7609 |
| Three models | 0.7367, +0.0023 [+0.0018, +0.0028], better | 0.6959, +0.0023 [+0.0018, +0.0028], better |

In the three-model combination the averaged transformers get weight 0.7 (recommendation) and
0.8 (rating band) on both halves. F1 per class, kept -> three models: `recommended` 0.943 ->
0.944, `not_recommended` 0.749 -> 0.752, `no_idea` 0.511 -> 0.514; `low` 0.714 -> 0.719,
`medium` 0.479 -> 0.482, `high` 0.888 -> 0.886.

| | Expected | Found |
|---|---|---|
| Recommendation: all right / all wrong / some right | about 85% / 8 to 10% / 5 to 7% | 81.4% / 9.0% / 9.6%: all wrong right, the rest wrong (more room than expected) |
| `no_idea` wrong in all three | 40 to 55% | 33.1%: wrong, lower |
| Shared mistakes, FaBERT & Large / FaBERT & TF-IDF | 70 to 80% / 55 to 70% | recommendation 77.1% / 55.6%: right; rating band 81.0%: a little above |
| Perfect chooser above the kept combination | 0.03 to 0.08 | 0.074 and 0.067: right |
| Three models over the kept combination | 0 to 0.2 point | 0.23 point on each head: wrong, a little more |
| The choice | the kept combination stays | stays: the interval is above zero on both heads, but the mean gain, 0.23 point, is below the 0.3 point fixed before the run |

What it means:

- About a tenth of the reviews are wrong in every model, mostly `no_idea`, `not_recommended`,
  `low` and `medium`; no combination of these models can fix them.
- On another tenth at least one model is right. A perfect chooser would gain about 7 points,
  but the real combinations recover only a small part of it: averaging gains 0.2 point, and
  stacking, which tries to learn when to trust which model, gained nothing
  (`reports/stacking_plan.md`). The models do not show, in their probabilities, when they are
  the one that is right.
- FaBERT and Large share four fifths of their mistakes; TF-IDF, a different kind of model,
  shares only a little more than half, which is why it is the one that helps.
- By the rule fixed before the run, the kept combination of FaBERT and TF-IDF stays. The
  three models are reported as the best combination measured on validation.

## Step 72: how much of that room is chance (added after step 71, before running)

The perfect chooser counts a review as right whenever any model has its label. Where the
label is unclear, a model can have it by chance, so the bound is too high, and more so the
more models there are. Three FaBERT runs that differ only in their seed (step 41) know the
same things, so their perfect-chooser gain is what chance alone gives.
`scripts/check_seed_oracle.py` measures it for the three seeds and for each pair, next to the
gain of the three final models over FaBERT alone (0.0756 and 0.0687).

The seeds were trained on 500,000 reviews for one epoch and vary more than models trained on
all reviews, so their gain overstates chance a little.

Rule: if the three seeds gain more than half as much as the three final models, at most half
of the room is real, a real chooser could hope for a small part of that, and the idea of a
chooser with more information (text length, agreement between the models, the words) is
dropped. Otherwise it gets its own plan.

Expected: the seeds have "some right" on 4 to 6% of the recommendation reviews (the final
models 9.6%) and gain 0.03 to 0.045 for each head, 40 to 60% of the final models' gain: the
rule is near its edge.

### Result of step 72

`reports/seed_oracle.json`.

| | Recommendation | Rating band |
|---|---|---|
| Seeds alone | 0.7235, 0.7237, 0.7241 | 0.6858, 0.6867, 0.6864 |
| Some right: three seeds / three final models | 3.5% / 9.6% | 4.4% / 11.3% |
| Perfect-chooser gain, two seeds | +0.0242 | +0.0216 |
| Perfect-chooser gain, three seeds | +0.0358 | +0.0319 |
| Perfect-chooser gain, three final models over FaBERT | +0.0756 | +0.0687 |
| Seeds' gain as a share of the final models' gain | 47% | 46% |

| | Expected | Found |
|---|---|---|
| Some right, three seeds (recommendation) | 4 to 6% | 3.5%: wrong, lower |
| Perfect-chooser gain, three seeds | 0.03 to 0.045 | 0.0358 and 0.0319: right |
| Share of the final models' gain | 40 to 60% | 47% and 46%: right |

What it means:

- Chance alone, with models that know the same things, gives about half of the
  perfect-chooser room. The other half, about 4 points on each head, comes from the three
  final models really making different mistakes: they disagree with one of them right almost
  three times as often as the seeds.
- Below half, so by the rule the idea of a chooser with more information gets its own plan.
  The seeds overstate chance a little, so the real part may be somewhat larger; it is still
  only a bound, and a real chooser can recover only a part of it.
