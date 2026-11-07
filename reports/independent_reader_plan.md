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

## Result

`reports/independent_reader.json`, answers from commit 36ed533 on the molab GPU
(`Qwen/Qwen3-8B`, transformers 5.21.0, 0.8 minutes for 3,600 questions). The exact revision
of the reader's weights was not recorded: transformers 5 no longer puts it where the script
reads it, so `reader_revision` is empty. Nearly all of the next-token probability fell on the
three digits (mean 1.000 on both heads), so the reader always answered with one of the options.

The rule, as fixed before running:

| Head | All right: agrees with label [95%] | All wrong, same label: with label | with the models | Verdict |
|---|---|---|---|---|
| recommendation | 0.708 [0.677, 0.737] | 0.324 | 0.472 (1.46 times) | inconclusive |
| rating band | 0.712 [0.682, 0.741] | 0.280 | 0.590 (2.11 times) | shared-bias objection rejected |

The reader passes the 70% check on both heads by its point value, but the lower end of both
intervals is under 70%, so it is a weak reader and is reported as one. It is weak on the
middle classes: on reviews all three models got right it agrees with `no_idea` 26% and with
`medium` 38% of the time, against 89 to 98% on the other classes.

| | Expected | Found |
|---|---|---|
| All right, recommendation | 85 to 95% | 70.8%: wrong, lower |
| All right, rating band | 75 to 85% | 71.2%: wrong, lower |
| All wrong, with the models | 60 to 75% | 47.2% and 59.0%: wrong, lower |
| All wrong, with the label | 15 to 30% | 32.4% (wrong, a little higher) and 28.0% (right) |
| Clearest: `medium` called high, `no_idea` called recommended | yes | with the models 72% and 82% of the time: right |

### Each mistake beside the reviews of the same class (added after seeing the results)

The reader reads middle texts as negative: of the `no_idea` reviews all three models got
right it calls 68% `not_recommended`, and of the right `medium` reviews it calls 60% `low`.
So agreeing with a negative label says little. To separate what the text says from the
reader's own habit, `--score` now also sets each pair (true label, the three models' label)
beside the "all right" reviews of the same true class: how often the reader gives the
models' label in each. This is report only; the rule and its verdicts above are unchanged.

| True label | The models said | Reviews | Reader gives the models' label | Same, on right reviews of that class | Reader gives the true label |
|---|---|---|---|---|---|
| `medium` | high | 266 | 68.8% | 2.3% | 28.2% |
| `high` | medium | 281 | 63.3% | 19.3% | 5.0% |
| `low` | medium | 212 | 25.5% | 1.3% | 72.2% |
| `low` | high | 88 | 73.9% | 1.3% | 10.2% |
| `no_idea` | recommended | 227 | 76.7% | 6.0% | 18.5% |
| `recommended` | no_idea | 277 | 35.0% | 9.7% | 9.4% |
| `not_recommended` | no_idea | 255 | 11.4% | 1.7% | 83.9% |

(Pairs under 75 reviews are in the report: `high`/low 19, `medium`/low 34,
`recommended`/not_recommended 23, `not_recommended`/recommended 45, `no_idea`/not_recommended 73.)

In every one of the 12 pairs the reader gives the models' label more often than on right
reviews of the same class. In the largest pairs that go up (`medium` called high, `no_idea`
called recommended), on `high` reviews called medium (most of them rate 4,
`reports/reading_models.json`), and on
`recommended` reviews the models call `no_idea` (the reader gives `recommended` to only 9%,
against 89% of right ones), a reader that never saw our labels reads the text as the models
do. Where the true label is negative and the models say a middle class (`low`/medium,
`not_recommended`/no_idea), the reader mostly keeps the negative label; since it reads middle
texts as negative anyway, these pairs cannot decide between the label and the models, and
they are what leaves the recommendation verdict inconclusive.

What this supports: the shared mistakes in which the models call a text more positive or
more middle than its label, and those on `high` reviews read as medium, are what the text says to an
independent reader, not only a bias the three models learned together. For negative labels
the models call middle, this reader cannot tell. Its own weakness on the middle classes may
come from the prompt, or be one more sign that a middle opinion is hard to see in the text;
this run cannot separate the two.
