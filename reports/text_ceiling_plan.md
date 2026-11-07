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

## Result

`reports/text_ceiling.json`, commit 3e8e982, on the same 872,795 reviews. Accuracy (the
ceiling is an exact bound for accuracy only):

| Head | Texts seen at least | Reviews (share) | Ceiling | Train majority | FaBERT | BiLSTM | TF-IDF |
|---|---|---|---|---|---|---|---|
| recommendation | 2 | 227,829 (30.6%) | 0.9313 | 0.9237 | 0.9222 | 0.9218 | 0.9176 |
| recommendation | 5 | 184,278 (24.7%) | 0.9296 | 0.9275 | 0.9262 | 0.9261 | 0.9229 |
| recommendation | 20 | 146,711 (19.7%) | 0.9305 | 0.9299 | 0.9291 | 0.9289 | 0.9269 |
| rating band | 2 | 256,705 (32.2%) | 0.8539 | 0.8394 | 0.8321 | 0.8319 | 0.8295 |
| rating band | 5 | 211,200 (26.5%) | 0.8482 | 0.8451 | 0.8363 | 0.8363 | 0.8331 |
| rating band | 20 | 171,824 (21.6%) | 0.8485 | 0.8481 | 0.8406 | 0.8404 | 0.8361 |

On texts seen at least 5 times, FaBERT is 0.34 point under the ceiling for recommendation
(99.6% of it) and 1.19 points under for rating band (98.6% of it).

Macro F1 on the same reviews (k at least 5): FaBERT 0.7334 and 0.6545; giving every text its
most frequent label: 0.7238 and 0.6540. The models' Macro F1 is a little higher than that
answer's, because the offsets trade accuracy for recall of the small classes: as said in the
design, that answer is not a bound for Macro F1.

The most frequent label of a repeated text gives `no_idea` to 7,860 reviews (13,471 truly
`no_idea`) and `medium` to 13,976 (33,654 truly `medium`), k at least 5. The most frequent
texts carry mixed labels: "خوب بود" in rating band is high 9,410, medium 2,250, low 228;
"خوبه" is high 9,887, medium 1,877, low 293.

| | Expected | Found |
|---|---|---|
| Ceiling, recommendation, k at least 5 | 88 to 92% | 92.96%: wrong, higher |
| Ceiling, rating band, k at least 5 | 82 to 86% | 84.82%: right |
| FaBERT within 2 points of the ceiling | yes | 0.34 and 1.19 points: right |
| Train majority within 1 point of the ceiling | yes | right for k at least 5 and 20; for k at least 2 in rating band 1.45 points: wrong |
| The best per-text answer almost never gives `no_idea` or `medium` | yes | it gives them to about half as many reviews as truly have them: wrong |

What it means: on a quarter of the data the limit of any model that reads only the text can
be computed exactly, and the models reach 98.6 to 99.6% of it. What remains wrong there comes
from the same text carrying different labels, which no text-only model can fix. The ceiling
holds for repeated, mostly short texts; for long unique reviews it is not a bound, but the
reading-model comparison and the independent reader point the same way.
