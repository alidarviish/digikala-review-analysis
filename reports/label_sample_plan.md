# Labeling a validation sample by hand (written before labeling)

## Why

The same text often comes with different labels (E11), the recommendation was given by the
buyer, and `no_idea` is a self-reported, fuzzy answer. Scores of 0.70 to 0.73 may therefore
be close to what any reader could reach. This sample measures how often a careful reader
disagrees with the original label, per class, and how consistent that reader is with
themself. It also puts the small differences between models (for example 0.1 point between
TookaBERT-Large and FaBERT) next to the size of the label noise.

## Design

- From validation only, with seed 2026 (`scripts/make_label_sample.py`); the test set is not
  read. 100 reviews labeled `recommended`, 50 `not_recommended` and 50 `no_idea`, each text
  at most once, so the small classes get enough reviews to measure.
- 20 of these reviews appear a second time, in shuffled order, to measure how often the same
  reader gives the same label twice. One reader labels the whole file.
- The reader sees only the row number and the text, and types 1 (recommended), 2 (not
  recommended) or 3 (no idea), judging from the text alone. The original labels and the
  model predictions are not looked at until the file is finished.
- `scripts/score_label_sample.py` compares the hand labels with the original labels (per
  class, and weighted by the share of each class in validation), with the kept FaBERT
  predictions with offsets, and the repeated reviews with their first showing. Report only:
  nothing in the models changes because of this sample.

## Expected

- Hand label equal to the original label: about 90% for `recommended`, 75 to 85% for
  `not_recommended`, 40 to 60% for `no_idea`.
- The same hand label on a repeated review: about 90%.
- FaBERT agrees with the hand labels about as often as with the original labels.

With 50 reviews per small class, a share has an uncertainty of about plus or minus 10 to 14
points, so only large differences between classes can be read from it.
