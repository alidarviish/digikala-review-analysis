# An independent reader that never saw our labels (written before running)

## Why

FaBERT, TF-IDF and the BiLSTM read the text in three different ways, and when all three are
wrong they give the same wrong label 90 to 94% of the time, with every BiLSTM seed
(`reports/reading_models.json`, `reports/bilstm_seeds.json`). But all three learned from the
same training labels, so their agreement could be a bias they share from those labels
rather than what the text says. The only reader so far that did not learn from the labels is
one person reading 20 reviews per group (`reports/error_sample_plan.md`). A large language
model asked about each review, with no training on our data, is a second independent reader,
on many more reviews.

## Reader

`Qwen/Qwen3-8B` (Apache 2.0, 8.2 billion parameters, trained on over 100 languages including
Persian), run on the molab GPU with its weights downloaded from the Hugging Face hub, so no
review leaves the machines the project already uses. Thinking mode off. For each review it is
asked one question per head, in English, with the review text in Persian; the answer is read
from the probabilities of the three answer tokens ("1", "2", "3") after the question, so it
is deterministic and cannot answer outside the three options:

- recommendation: would the writer recommend the product? 1 yes, 2 no, 3 no clear opinion
- rating band: from the text alone, what rating did the writer most likely give? 1 low
  (1 to 2 stars), 2 medium (about 3), 3 high (4 or 5)

The exact prompts are in `scripts/independent_reader.py`, fixed before running. No example
reviews are shown in the prompt (zero-shot), so nothing of our labels reaches the reader.

## Sample

From the 872,795 compared validation reviews (`scripts/compare_reading_models.py`), seed
2026, each review at most once:

- for each head and each true class: 300 reviews all three models got right, and 300 all
  three got wrong with the same wrong label (fewer if the group is smaller);
- 1,800 reviews per head in all at most (3 classes, 2 groups, 300 each).

Only review ids and groups are written to git (`reports/independent_reader/sample.json`);
the texts are read from the validation cache on molab.

## Steps

1. local: `scripts/independent_reader.py --make-sample` (needs the three models' predictions)
2. molab GPU: `scripts/independent_reader.py` reads the sample, the texts and asks the
   reader; writes `reports/independent_reader/answers.json` (ids and answers, no text)
3. local: `scripts/independent_reader.py --score` writes `reports/independent_reader.json`

## What is reported

For each head, group and true class: how often the reader agrees with the true label, and,
in "all wrong, same label", how often it agrees with the three models instead, with Wilson
95% intervals (one review per draw; reviews of one product are rare in a sample this
small).

## Rule (fixed now)

- The reader is usable if, on the "all right" reviews, it agrees with the true label at least
  70% of the time on each head. Otherwise nothing is concluded from it.
- If usable: the shared-bias objection is rejected when, on "all wrong, same label", the
  reader agrees with the three models at least twice as often as with the true label. If it
  agrees with the true label more often than with the models, the objection stands and the
  report says so.

## Expected

- On "all right", the reader agrees with the label 85 to 95% of the time for recommendation
  and 75 to 85% for rating band.
- On "all wrong, same label", it agrees with the three models 60 to 75% of the time and with
  the true label 15 to 30%; the clearest cases are `medium` reviews the models call high and
  `no_idea` reviews they call recommended.
