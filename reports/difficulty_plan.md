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
