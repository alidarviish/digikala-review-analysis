# Training reviews merged by text (written before running)

The loss analysis (`reports/loss_analysis_plan.md`) found that 33.5% of the training reviews
repeat an earlier text exactly and 25.5% belong to a text with more than one recommendation
label, while on texts with many copies the model is already at the lowest loss possible.

Idea: one training row per different text. Its target for each head is the share of each
label among its copies (for example 90% `recommended`, 10% `no_idea`), and its number of
copies is kept. Only the review body is used, as the project requires; validation and test are
not changed.

Two ways to train on the merged rows:

- **A, weight = copies.** The loss is exactly the loss on the separate rows, so the scores
  should stay the same while one epoch takes about a third fewer steps.
- **B, weight = 1.** Every text counts once. Frequent short texts ("عالی بود") lose weight
  and the rare texts, where most of the loss is, gain it. The class shares change, so the
  offsets tuned on validation matter more.

## Step 1: check on the data (this file)

`scripts/check_merged_texts.py` reads only the training cache and writes
`reports/merged_texts_check.json`: the number of different texts and steps per epoch, how many
copies texts have, the most copied texts, texts with more than one label, and the share of
each class as now and if each text counts once.

Expected:

- 3,552,498 rows become about 2,362,000 different texts (66.5%, from the 33.5% found in the
  loss analysis); steps per epoch 55,508 -> about 36,900.
- 93 to 97% of texts appear once. Fewer than 200 texts have 1,000 or more copies, together
  holding 10 to 20% of the rows; the most copied are short praise such as "عالی" or "عالی بود".
- Recommendation: 2 to 5% of the labeled texts have more than one label, holding 25 to 30% of
  the labeled rows. Rating band: more, 4 to 10% of texts and 30 to 45% of rows.
- If each text counts once, `recommended` falls by 3 to 8 points and `no_idea` and
  `not_recommended` rise; `high` falls and `low` and `medium` rise.

### Result of step 1

`reports/merged_texts_check.json`.

| | Found | Expected |
|---|---|---|
| Different texts | 2,363,237 of 3,552,498 rows (66.5%) | about 2,362,000: right |
| Steps per epoch | 55,508 -> 36,926 | about 36,900: right |
| Texts that appear once | 95.35% of texts, 63.4% of rows | 93 to 97%: right |
| Texts with 1,000+ copies | under 0.005% of texts (fewer than 120), 13.8% of rows | under 200 texts, 10 to 20%: right |
| Most copied | "عالی" 92,467, "خوب بود" 54,172, "خوبه" 50,564, "خوب" 43,512, "عالی بود" 20,055 | short praise: right |
| Recommendation, more than one label | 0.89% of texts, 24.2% of labeled rows | 2 to 5%, 25 to 30%: both lower |
| Rating band, more than one label | 1.62% of texts, 30.4% of rows | 4 to 10%, 30 to 45%: texts lower, rows right |
| Each text once: recommendation | `recommended` 79.1 -> 74.8%, `not_recommended` 9.6 -> 12.3%, `no_idea` 11.3 -> 12.9% | fall of 3 to 8 points: right |
| Each text once: rating band | `low` 10.0 -> 12.5%, `medium` 18.6 -> 19.9%, `high` 71.5 -> 67.6% | right |

What it means:

- Mixed labels belong to very few texts, but these are the frequent short ones: under 1% of
  the texts hold about a quarter of the rows. Merging turns each of them into one row with
  its label shares.
- A cuts one epoch by a third with the same loss.
- B shrinks the frequent short texts (13.8% of the rows come from fewer than 120 texts) to
  almost nothing, and moves the class shares by 1 to 4 points towards the small classes.
  Both are worth a GPU pilot.

## Step 2: GPU pilot

Steps 37 and 38 of `notebooks/run_all.py` train FaBERT exactly as in step 35 (the same 500,000
training reviews, seeds 2026 to 2028, 1 epoch, scored on all of validation), with one change:
`scripts/pilot_transformers.py --merge copies` or `--merge once`. The runs of step 35 are the
reference, so nothing else is trained again. Step 39 (`scripts/compare_merged_runs.py`,
local) compares each seed on the same reviews (paired bootstrap, offsets cross-fitted by
product) and decides with the rule of step 36: a lead smaller than the spread between seeds
is a tie, and a tie goes to the faster setup.

The merged loss with weight = copies is tested to be exactly the loss on the separate rows
(`tests/test_merge_texts.py`), and the runs without `--merge` are unchanged.

Expected:

- Inside the 500,000-review sample fewer texts repeat than in all the training reviews: about
  350,000 to 380,000 different texts, so 5,500 to 6,000 steps instead of 7,813 and about 3.0
  to 3.4 training minutes instead of 4.35.
- A (copies): the same loss but about 28% fewer updates in one epoch, so slightly worse:
  mean Macro F1 of the two heads 0.001 to 0.005 below the separate runs. If so, A only pays
  off when the saved time is spent on more epochs, which is a separate question.
- B (once): a tie, within 0.003 either way; the offsets make up for the shift in class
  shares. The `no_idea` F1 may rise a little and `recommended` fall a little.
- The rule keeps whichever is faster on a tie, so a tie in B would make B the setup to try
  on all the training reviews.

### Result of step 2

`reports/transformer_pilot_whole/merged_vs_separate_n500000.json`. Mean Macro F1 of the two
heads, three seeds each; the separate runs are those of step 35.

| | Merged | Separate | Lead of merged | Spread between seeds | Verdict |
|---|---|---|---|---|---|
| A, weight = copies | 0.7043 | 0.7046 | -0.0002 | 0.0006 | tie, merged is faster |
| B, each text once | 0.7045 | 0.7046 | -0.0001 | 0.0006 | tie, merged is faster |

| | Expected | Found |
|---|---|---|
| Different texts in the sample | 350,000 to 380,000 | 362,642: right |
| Steps, training minutes | 5,500 to 6,000; 3.0 to 3.4 | 5,667; 3.3 instead of 4.3: right |
| A | 0.001 to 0.005 worse | 0.0002 worse, a tie: wrong, closer than expected |
| B | a tie within 0.003 | 0.0001 worse, a tie: right |
| Per class | `no_idea` a little higher, `recommended` a little lower | `no_idea` +0.002 (A) and +0.001 (B), `not_recommended` -0.004 (A) and -0.002 (B), `recommended` unchanged: half right |

Per seed and head the differences are at most 0.0012 either way; recommendation is 0.001
lower in two of the three seeds of A (whole interval below zero), rating band 0.001 higher in
one.

What it means:

- Training on label shares instead of single labels does not change the score: the mixed
  labels of the frequent short texts were not holding the model back, which fits the loss
  analysis (on those texts the model was already at the floor).
- Merging saves time only: 24% of the training time on this sample, about a third on all the
  training reviews (36,926 instead of 55,508 steps per epoch).
- By the rule a tie goes to the faster setup. If the final model is trained again after a
  later experiment, it will be trained on merged texts with weight = copies (once the same option is
  added to `scripts/train_transformer.py`): it keeps exactly the loss of the separate rows
  and saves a third of the time. B is no better and changes what the loss counts, so it
  is not used. The current final model is not trained again for this alone.
