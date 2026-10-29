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

## Step 2: GPU pilot (only if step 1 supports it)

FaBERT on 500,000 training rows, 3 seeds, scored on all of validation with cross-fitted
offsets, against the current setup on the same seeds: A first (should tie and be faster),
then B. The rule as before: a change is kept only if the whole 95% interval is above zero; a
tie goes to the faster setup. The expected results for step 2 will be written here before it
runs.
