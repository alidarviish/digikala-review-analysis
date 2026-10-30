# How hard each training review is, and two uses of it (written before running)

Giving the hard training reviews more weight did not help (`reports/hard_examples_plan.md`):
the more weight moved to them, the worse the scores, and the ambiguous classes did not improve.
The hand-labeled sample suggests why: many hard reviews have an ambiguous or wrong label. Two
experiments follow from this, and both need to know how hard each training review is:

1. **Leave out the hardest labels.** If the hardest reviews are mostly noisy labels, training
   without them should help or at least not hurt.
2. **Curriculum** (suggested by the course teacher): train on the easy reviews first and add
   the hard ones later.

## Step 1: difficulty of every review in the training sample (GPU)

A model's loss on reviews it was trained on says little, since it has partly learned them. The
difficulty is therefore measured out of fold: the 500,000 training reviews of the pilot (step
35) are split in two halves by product, FaBERT is trained on one half (1 epoch, the pilot
settings) and scores the other, and the other way round. Each review then has, for each head,
the probability that a model which never saw it gives to its label. Only the review body is
used; validation and test are not read.

`scripts/score_difficulty.py` (step 47, GPU) writes `reports/difficulty/n500000.parquet` (id,
and per head the probability of the true class); it is small and is kept with the reports, so
the later GPU steps find it after a new set-up. `scripts/check_difficulty.py` (step 48)
summarizes it in `reports/difficulty/summary.json`: the share of each class, review length and
text that comes more than once in training among the hardest 5% and 10% of each head, and how
many of them are texts whose copies carry more than one label, each next to all reviews.

Expected:

- Recommendation: `no_idea` makes up 35 to 50% of the hardest 10% (11% of the reviews),
  `not_recommended` 15 to 25%. Rating band: `medium` 40 to 55% of the hardest 10%.
- 20 to 35% of the hardest 10% are texts with more than one label, above the share among all
  reviews.

## Step 2: leaving out the hardest labels, and curriculum (GPU)

Steps 49 to 51: FaBERT on the same 500,000 reviews and seeds 2026 to 2028, scored on all of
validation and measured 10 times during training, against the usual training of step 41
(compared in step 52, `scripts/compare_hard_examples.py --set difficulty`, with cards):

| Run | What changes |
|---|---|
| leave out 5% | for each head, the label of its 5% hardest reviews is not used (the review still trains the other head) |
| leave out 10% | the same with 10% |
| curriculum | every review is used once, ordered from easy to hard: the reviews are put into 10 groups by difficulty (mean over the heads with a label), the groups follow each other from the easiest, and the order inside a group is random |

Expected:

- Leave out 5%: a tie (within 0.002 of the mean Macro F1 of the two heads); the validation loss
  a little lower on `recommended` and `high` and higher on `no_idea` and `medium`.
- Leave out 10%: a tie or slightly worse (0 to 0.003 lower).
- Curriculum: a tie or slightly worse (0 to 0.004 lower); on the curves the validation loss
  falls faster in the first half of the epoch and ends level or a little higher, since the
  last updates see only the hardest reviews.

The rule as before: a change is kept only if it leads by more than the spread between seeds; a
tie goes to the usual training.

## Results

### Step 1

`reports/difficulty/summary.json`. Each half took about 2.2 minutes to train. Shares among all
labeled reviews / the hardest 5% / the hardest 10%:

| | Found | Expected (hardest 10%) |
|---|---|---|
| `no_idea` share, recommendation | 11.2 / 60.4 / 59.8% | 35 to 50%: wrong, more |
| `not_recommended` share | 9.6 / 20.6 / 19.1% | 15 to 25%: right |
| `medium` share, rating band | 18.6 / 59.8 / 70.5% | 40 to 55%: wrong, more |
| Texts with more than one recommendation label, recommendation | 24.3 / 27.4 / 19.9% | 20 to 35%, above all reviews: wrong, below |
| The same, rating band | 26.0 / 23.4 / 29.0% | right |
| Texts repeated in training, recommendation | 35.5 / 29.3 / 21.8% | |
| Hardest 10%, probability of the own label below | recommendation 0.374, rating band 0.197 | |

The hardest reviews are mostly the ambiguous class of each head, and for recommendation mostly
texts that come only once, not texts with conflicting copies. The column of texts with more
than one label is counted on the recommendation label for both heads, so for the rating band it
is only a rough sign.

### Step 2

`reports/transformer_pilot_whole/difficulty_n500000.json`, the cards in
`reports/figures/experiments/difficulty_*`. Mean Macro F1 of the two heads, three seeds each;
the usual training (step 41) scores 0.7051. Every run took about 4.3 minutes per seed.

| Run | Mean Macro F1 | Lead | Spread between seeds | Verdict | Expected |
|---|---|---|---|---|---|
| leave out 5% | 0.7037 | -0.0013 | 0.0006 | worse | a tie: wrong |
| leave out 10% | 0.7026 | -0.0025 | 0.0010 | worse | 0 to 0.003 lower: right in size |
| curriculum | 0.1455 | -0.5596 | 0.0192 | worse | 0 to 0.004 lower: wrong, by far |

Change in F1 per class (points, three seeds pooled; outside the 95% interval unless marked):

| Run | recommended | not_recommended | no_idea | low | medium | high |
|---|---|---|---|---|---|---|
| leave out 5% | +0.01 (no difference) | -0.25 | -0.32 | +0.22 | -0.38 | -0.06 |
| leave out 10% | 0.00 (no difference) | -0.17 | -0.82 | +0.02 (no difference) | -0.39 | -0.13 |
| curriculum | -93.3 | -40.9 | -39.0 | -49.3 | -23.7 | -86.2 |

What it means:

- Leaving out the hardest labels hurts, and more with 10% than with 5%. The losses fall on the
  ambiguous classes, whose reviews make up most of the left-out labels: the hard labels carry
  useful signal and are not mostly noise.
- The curriculum as planned breaks the model. The validation loss falls while the easy groups
  train (recommendation 1.74 at a tenth of the epoch down to 0.50 at nine tenths) and jumps to
  3.30 / 3.43 after the last group, which holds only the hardest reviews, mostly `no_idea` and
  `medium`. The last updates teach the model those classes alone, and `recommended` and `high`
  are nearly lost. The fault is in the design (strict easy to hard, with the hardest group alone
  at the end), not in the idea of a curriculum as such.
- The validation loss of the leave-out runs rises during training instead of falling
  (recommendation 0.3169 -> 0.4459, rating band 0.4969 -> 0.6208 with 5% left out), against a
  small expected fall: wrong. Without the reviews that contradict its guesses the model grows
  overconfident, and the validation reviews like them cost much more. The offsets hide most of
  this in Macro F1, but the probabilities get worse.
- The difficulty is out of fold by product, not by text: a text that comes in both halves (about
  a third of the reviews are repeated texts) was seen by the model that scores it, so repeated
  texts look easier than they are. This does not change the verdicts, but the hardest reviews
  are partly hard because they are new texts.
- Kept: the usual training. Reference with checks against step 35: +0.0004, +0.0007, +0.0004.
