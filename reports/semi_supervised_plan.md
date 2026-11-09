# Semi-supervised learning: pseudo-labels for the hidden recommendation labels (written before running)

## The question

20% of the labeled training reviews of each recommendation class (736,134 reviews) had their
recommendation label hidden by the shared rule (`src/digikala_reviews/mask.py`); they are in
`train_pseudo_unlabeled.csv` with their text only. Every model so far was trained without them
(3,552,498 reviews). The question: if the final model guesses their labels and the confident
guesses are added to training, does the recommendation classifier get better?

Their true labels (`audit/masked_labels_reference.csv`) are never used for training; they are
read only to measure how often the guesses are right.

## Method

1. **Teacher.** The final FaBERT model (`artifacts/transformer_no_early_stop/model.pt`) scores
   the 736,134 hidden reviews (both heads, GPU). Its recommendation probabilities are scaled
   with the temperature 1.124 (`reports/calibration_plan.md`), so that 0.9 means right 9 times
   in 10.
2. **Pseudo-labels.** A review keeps its guessed recommendation label when the top
   probability is at least 0.9 (`PSEUDO_LABEL_THRESHOLD`). Its rating band is not used: it was
   hidden with the rest of the row. The guesses are checked against the true labels: share
   kept, accuracy, and the share of each class, overall and per class.
3. **Student, on the pilot.** FaBERT is trained as in step 41 (500,000 training reviews, 1
   epoch, seeds 2026 to 2028, all of validation), with the pseudo-labeled reviews added in the
   same proportion as in the full data (the hidden reviews are 20.7% of the labeled ones, so a
   fixed random 103,600 of them, of which those above 0.9 are kept). It is compared with step
   41 on the same validation reviews by the usual rule (paired bootstrap per seed; a lead
   smaller than the spread between seeds is a tie, and a tie goes to the usual training).
4. **Full run, only if the pilot wins.** If the pilot student beats step 41 by the rule, FaBERT
   is trained again on all 3.55 million reviews plus all pseudo-labeled ones, exactly as the
   kept run, and compared on the whole validation set.

The test set is not read at any step.

## What has to be prepared

- The hidden reviews' text is not in the cache that molab downloads. `scripts/cache_normalized.py`
  also writes `train_pseudo_unlabeled.parquet` (normalized the same way), to be added to the
  Drive cache folder.
- The final model's weights are not in the repository (263 MB). The file is added to the same
  Drive folder as `fabert_final.pt`, so molab can load it, together with the updated
  `cache_info.json` that holds the hash of the new cache file. The guesses record the hash of
  the weights they were made with, and the check on your own machine compares it with the
  local file.
- New steps: teacher scoring (GPU), checking the guesses against the true labels (on your own
  machine, where the audit file is), the pilot student (GPU, with a test run), and the
  comparison (on your own machine).

## Expected

- About 65 to 72% of the hidden reviews reach 0.9, as on validation (69.1% after scaling),
  and 96 to 98% of those guesses are right.
- The kept guesses are about 88 to 92% `recommended`, 7 to 10% `not_recommended` and only
  1 to 3% `no_idea` (against 11.3% `no_idea` among all hidden reviews): the confident guesses
  are mostly the easy reviews.
- The pilot student ties with step 41 (within the spread between seeds); if anything,
  `no_idea` F1 falls a little, since the added reviews teach it little about that class. The
  model already has 3.55 million labeled reviews in the full data; reviews it was already sure
  of add little new.
- So the full run is most likely not made, and the answer to the question is "no measurable
  gain", with the measured quality of the guesses.

## Result

`reports/pseudo_labels/scoring.json` and `check.json` (the guesses, step 64 and 65),
`reports/transformer_pilot_whole/pseudo_n500000.json` (the pilot, step 67), all of validation.
Scoring the 736,134 hidden reviews took about 9 minutes; the guesses were made with the final
FaBERT weights (the hash matches the local `model.pt`).

The guesses against the true labels (true classes: `recommended` 79.1%, `no_idea` 11.3%,
`not_recommended` 9.6%):

| Threshold | Kept | Right | Kept classes: recommended / not_recommended / no_idea | Kept and right, share of each true class: recommended / not_recommended / no_idea |
|---|---|---|---|---|
| 0.8 | 574,432 (78.0%) | 96.0% | 93.4% / 6.4% / 0.3% | 88.8% / 46.0% / 1.5% |
| 0.9 | 508,744 (69.1%) | 97.3% | 95.7% / 4.2% / 0.0% | 81.5% / 28.6% / 0.2% |
| 0.95 | 404,952 (55.0%) | 98.4% | 97.3% / 2.7% / 0.0% | 66.6% / 14.9% / 0.0% |

The pilot student (500,000 reviews plus 71,740 of a fixed random 103,608 hidden reviews whose
guess reaches 0.9) against step 41, three seeds:

| Head | Step 41 | With pseudo-labels | Difference per seed [95%] |
|---|---|---|---|
| recommendation | 0.7235, 0.7237, 0.7241 | 0.7226, 0.7225, 0.7233 | -0.0009 [-0.0016, -0.0002] worse; -0.0012 [-0.0019, -0.0005] worse; -0.0008 [-0.0016, +0.0000] no measurable difference |
| rating band | 0.6858, 0.6867, 0.6864 | 0.6865, 0.6864, 0.6869 | +0.0007 [+0.0001, +0.0013] better; -0.0003 [-0.0009, +0.0003] and +0.0005 [-0.0001, +0.0011] no measurable difference |

Mean Macro F1 of the two heads: 0.7047 against 0.7051, lead -0.0003, spread between seeds
0.0007: a tie, so the usual training is kept. Each seed took 5.0 minutes instead of 4.4.

F1 per class with offsets, mean of three seeds, step 41 -> with pseudo-labels: `recommended`
0.943 -> 0.944, `not_recommended` 0.732 -> 0.732, `no_idea` 0.496 -> 0.492; `low` 0.702 ->
0.704, `medium` 0.472 -> 0.470, `high` 0.885 -> 0.886.

| | Expected | Found |
|---|---|---|
| Share reaching 0.9 | 65 to 72% | 69.1%: right |
| Kept guesses right | 96 to 98% | 97.3%: right |
| Kept classes | 88 to 92% recommended, 7 to 10% not_recommended, 1 to 3% no_idea | 95.7%, 4.2%, 0.0%: wrong, even more one-sided |
| Pilot student | a tie; `no_idea` F1 a little lower | a tie (recommendation slightly lower in two seeds); `no_idea` 0.4 point lower: right |
| Full run | most likely not made | not made: the pilot did not win |

What it means:

- The confident guesses are right almost every time, but they are nearly all `recommended`:
  only 0.2% of the hidden `no_idea` reviews and 28.6% of the hidden `not_recommended` ones
  get a label at 0.9, against 81.5% of the `recommended` ones. A lower threshold does not
  change this (0.3% `no_idea` kept at 0.8).
- So the added reviews are mostly reviews the model already gets right, of the class it
  already knows best; they teach it nothing about `no_idea`, the class that limits Macro F1.
- Kept: the final system as before (`reports/ensemble_plan.md`). The answer to the question
  is "no measurable gain", with the measured quality of the guesses above.

## Against the project's protocol (added 2026-11-09, before the official run)

A check of the project definition against this pilot found three differences, so it does not
count as the required semi-supervised experiment; it stays as a pilot, and its result above
stands as such:

1. The 0.9 threshold was applied to probabilities scaled with a temperature fitted on the
   validation labels (`reports/calibration_plan.md`). The project asks that no validation or
   test label be used to make or select pseudo-labels; the threshold must be applied to the
   model's own probabilities.
2. The student was FaBERT trained from its pretrained weights on the 500,000-review pilot.
   The project asks for a copy of the supervised model itself, fine-tuned further on
   `train_supervised` plus the accepted pseudo-labeled reviews, with the rating band loss
   masked for those reviews.
3. The project asks for both models to be compared on the same validation and test sets, with
   accuracy, Macro F1, Macro precision, Macro recall and per-class scores, and for a notebook
   `04_semi_supervised.ipynb` with the fixed 0.90 threshold.

The official run follows the project's protocol and gets its own plan before it runs.
