# Hard examples: giving the hard training reviews more weight (written before running)

The loss analysis (`reports/loss_analysis_plan.md`) found that most of the validation loss
that is left sits in reviews the model finds hard: the ambiguous classes (`no_idea`, `medium`),
long reviews and texts not seen in training. Hard sampling trains with more weight on such
reviews, in the hope that the model learns them better.

There is a known risk. A review can be hard because it is difficult, or because its label is
wrong or ambiguous. The hand-labeled sample (`reports/label_sample_plan.md`) found that when
FaBERT disagrees with the original label, the hand label sides with FaBERT in 55% of the cases
(21 of 38), and half of the 38 are `no_idea` reviews. Weighting hard reviews up may therefore
teach the model the noise of the labels. Only a measurement can tell which effect wins.

## Method

Focal loss: the loss of each review is multiplied by (1 - p)^gamma, where p is the probability
the model gives to the true class at that moment. A review the model already gets right with
p = 0.95 counts 0.05 (gamma 1) or 0.0025 (gamma 2) times as much; a review with p = 0.2 counts
0.8 or 0.64 times. gamma = 0 is the usual loss. The weighting follows the model while it trains,
so no first pass is needed. Only the review body is used and validation and test are not
changed.

## Step 1: check on the data (CPU)

`scripts/check_hard_examples.py` reads the validation probabilities of the three FaBERT pilot
runs on 500,000 training reviews (step 35) and the validation labels. Nothing is trained. For
gamma 0, 1 and 2 it reports which share of the total weight would go to each class, each
review length and texts seen or new, and how much of the weight of the hardest reviews
(p < 0.2) falls on texts whose copies carry more than one label, where no model can be right
every time. Validation stands in for training here: the training probabilities were not
saved, and both are drawn from the same reviews by product.

Expected:

- With gamma 2, `no_idea` gets 30 to 45% of the recommendation weight (11% of the reviews, so
  11% of the weight at gamma 0) and `medium` 35 to 50% of the rating-band weight (19%).
- Long reviews (16 words and more) gain 3 to 8 points of weight share; texts seen in training
  lose most of theirs.
- 10 to 25% of the weight of the hardest reviews falls on texts with more than one label.

### Result of step 1

`reports/hard_examples_check.json` (validation, the three runs of step 35 pooled).

| | Found (gamma 0 / 1 / 2) | Expected at gamma 2 |
|---|---|---|
| `no_idea` share of the recommendation weight | 11.3 / 44.2 / 55.1% | 30 to 45%: wrong, much more |
| `recommended` share | 79.1 / 36.3 / 24.8% | |
| `medium` share of the rating-band weight | 18.8 / 46.0 / 60.9% | 35 to 50%: wrong, more |
| Reviews of 16+ words | recommendation 17.7 / 24.0 / 24.5%, rating band 17.3 / 19.7 / 20.1% | +3 to +8 points: right for recommendation (+6.8), just below for rating band (+2.8) |
| Texts seen in training | recommendation 35.9 / 25.1 / 24.0%, rating band 37.3 / 33.4 / 32.6% | lose most of their share: wrong, they lose a third at most |
| Hardest reviews (p < 0.2) | recommendation 5.5% of reviews, 47.9% of the weight at gamma 2; rating band 10.0%, 52.2% | |
| Hardest reviews on texts with more than one label | recommendation 22.8% (all reviews 20.5%), rating band 33.2% (26.4%) | 10 to 25%: right for recommendation, more for rating band |

What it means:

- Focal loss moves far more weight than expected. At gamma 2 about half of all the training
  weight goes to the 5 to 10% hardest reviews, and the ambiguous class becomes the largest:
  `no_idea` 55% of the recommendation weight, `medium` 61% of the rating band.
- The hardest reviews are only a little more often texts with mixed labels than reviews in
  general (23% against 21%, 33% against 26%), so they are not mostly the reviews no model can
  get right; but those can only be seen among repeated texts, and the hand-labeled sample
  suggests more label noise in `no_idea`.
- gamma 2 is a large change; gamma 1 already moves most of the weight. A milder gamma 0.5 is
  added to the GPU pilot so that the effect can be followed in steps.

## Step 2: GPU pilot

Steps 41 to 45 of `notebooks/run_all.py` train FaBERT on the same 500,000 training reviews and
seeds 2026 to 2028 as step 35, scored on all of validation, with the validation loss measured
10 times during training on a fixed sample of 50,000 validation reviews (`--checks 10`):

| Step | Run | Option |
|---|---|---|
| 41 | the usual training again, now measured during training | |
| 42 | focal loss, gamma 0.5 (added after step 1) | `--focal 0.5` |
| 43 | focal loss, gamma 1 | `--focal 1` |
| 44 | focal loss, gamma 2 | `--focal 2` |
| 45 | class weights: every class counts as much in total (added after step 1) | `--class-weights` |

Focal loss is the mean over the labeled reviews of -(1 - p)^gamma log p, as in its original
form; the weights are not rescaled, so the loss is smaller than the usual one. Class weights
answer a separate question: step 1 showed that focal loss mostly moves weight between classes,
so if class weights give the same result, it is the class balance that matters and not the
difficulty of single reviews.

Step 46 (`scripts/compare_hard_examples.py`, local) compares each run with step 41 on the same
reviews (paired bootstrap, offsets cross-fitted by product) and checks that step 41 scores as
step 35. The rule as before: a change is kept only if it leads by more than the spread between
seeds; a tie goes to the usual training, which is as fast and simpler. Every run also gets an
experiment card, with the validation loss of both setups drawn during training.

Expected:

- Step 41 scores within 0.001 of step 35 (mean Macro F1 of the two heads): measuring during
  training does not change the training, only the GPU's rounding may differ.
- gamma 0.5 and gamma 1: a tie (within 0.002).
- gamma 2: a tie or slightly worse (0 to 0.004 lower), with `no_idea` and `medium` F1 up a
  little and `not_recommended` and `low` down.
- Class weights: a tie (within 0.002), with per-class changes in the same direction as gamma 1:
  the offsets already move the decisions towards the small classes, so the weights add little.
- The validation loss of every variant is higher than the usual one at the end, since they
  train for another balance of the classes than the true one, and the loss measures the
  probabilities as they are.
- On the curves the focal runs start slower and end level with the reference.
