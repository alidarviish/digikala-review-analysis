# How far can any text-only model go? A ceiling from repeated texts (written before running)

## Why

The reading-model comparison (`reports/reading_models_plan.md`) shows that three models
which read differently agree on the same wrong label for 90 to 94% of the reviews they all
get wrong. That is evidence, not proof, that a better model could not do much better. For
reviews whose exact text appears more than once there is a proof: a model that sees only
the text must give the same answer to the same text, so on those reviews no text-only model
can be right more often than by giving every text its most frequent label.

## Design (`scripts/estimate_text_ceiling.py`, `reports/text_ceiling.json`)

- Reviews: the same 872,795 validation reviews as the reading-model comparison (the
  50,000-review choosing sample left out), for each head those with its label. Texts are the
  normalized texts the models were trained on, compared exactly. The test set is not read.
- For texts seen at least k times among these reviews (k = 2, 5 and 20):
  - **ceiling**: the share of reviews that the most frequent label of their own text
    covers. No text-only predictor can be right more often on these reviews: it is an exact
    upper bound on accuracy, computed with the validation labels themselves (so it is
    optimistic, never too low).
  - the Macro F1 of giving every text its most frequent validation label, as a reference
    (Macro F1 is not decided text by text, so this is not a strict bound for it).
  - **train majority**: the most frequent label of the same text in `train_supervised`,
    which a model could learn; for texts not in train, nothing is counted.
  - FaBERT, TF-IDF and the BiLSTM on the same reviews: accuracy and Macro F1 (labels with
    offsets, as saved).
  - how many distinct texts and reviews this covers, and its share of all compared reviews.
- The 20 most frequent texts: their label counts and what each model says most often.

Caveat written now: repeated texts are mostly short, common phrases, so the ceiling holds
for this part of the data, not for long unique reviews. The saved labels with offsets use
offsets tuned on one of two product halves, so a model can give the same text two answers;
this can lift a model very slightly above a per-text ceiling and is reported, not hidden.

## Expected

- Texts seen at least 5 times: accuracy ceiling 88 to 92% for recommendation and 82 to 86%
  for rating band; FaBERT within 2 points of the ceiling on both heads.
- The train majority is within 1 point of the ceiling (the same phrases carry the same mix of
  labels in train and validation).
- On these reviews, the best possible per-text answer almost never gives `no_idea` or
  `medium`: the most frequent label of a repeated text is rarely one of them, so their
  recall stays low for any text-only model.
