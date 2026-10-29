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

## Added after labeling, before scoring

- Where the hand label and the original label differ, the buyer's star rating (never shown
  while labeling) is read as a third opinion: high = recommended, medium = no idea, low = not
  recommended. Expected: on those reviews the rating sides with the hand label more often
  than with the original label.
- Rule used for code 3, fixed while labeling: 3 only when the text itself is neutral or gives
  no verdict; when it leans one way, that way, even without full certainty.

## Result

`reports/label_sample.json`, 200 reviews labeled from the text alone, 20 repeats.

| Original label | Hand label agrees | Hand labels (recommended / not / no idea) | Expected |
|---|---|---|---|
| recommended (100) | 95% | 95 / 0 / 5 | about 90% |
| not_recommended (50) | 94% | 1 / 47 / 2 | 75 to 85% |
| no_idea (50) | 34% | 12 / 21 / 17 | 40 to 60% |

- Weighted by the share of each label in validation, the hand label agrees with the original
  on 88% of the reviews. The same hand label was given twice on all 20 repeats (expected
  about 90%).
- FaBERT agrees with the hand labels on 82.5% of the sample and with the original labels on
  81.0%: about the same, as expected. On this sample the model is not "more right" than the
  original labels.
- The rating prediction was wrong. Where the hand label differs from the original (39
  reviews with a rating), the buyer's rating sides with the original label in 69% and with
  the hand label in 23%. Of the 41 differences, 33 are reviews labeled `no_idea` by the
  buyer; 21 of them read as not recommended from the text, and 16 of those 21 have a medium
  rating.

What it means: the labels `recommended` and `not_recommended` are clean (about 95%). The
disagreement sits almost entirely in `no_idea`, and there the buyer's own rating agrees with
the buyer's label, not with the reading of the text. Buyers often choose "no idea" while
writing a mild complaint, and the rule used here (a text that leans one way gets that side)
reads the same text as not recommended. This is a difference in what `no_idea` means to the
buyer, which the text alone does not show, rather than wrong labels; it puts a low ceiling
on `no_idea` for any model that reads only the text. With 50 reviews per small class, each
share is uncertain by about 10 to 14 points.

Added after the first scoring: on the 38 reviews of the sample where FaBERT differs from the
original label (19 `no_idea`, 13 `not_recommended`, 6 `recommended` by the original label),
the hand label sides with FaBERT on 55.3% and with the original label on 39.5%. With 38
reviews each share is uncertain by about 16 points, so this does not separate the two; and
half of these reviews are `no_idea`, where the reading of the text and the buyer's label
differ for the reason above.
