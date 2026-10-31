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
