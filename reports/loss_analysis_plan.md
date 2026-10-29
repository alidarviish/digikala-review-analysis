# Where the validation loss comes from (written before running)

`scripts/analyze_loss.py`, on the final FaBERT model (trained for every epoch, kept at epoch
2.75). Report only: it decides which of the next steps are worth a GPU run (removing repeated
training texts, the other text fields, curriculum or hard sampling). The test set is not read.

## Questions

1. **Floor.** A text that comes with different labels cannot be fitted by any model that reads
   only the text: the best it can do is the share of each label among the copies, and the
   entropy of those shares is the lowest loss possible on them. How close is the model to that
   floor on texts with many copies?
2. **Groups.** Which classes, review lengths, and texts (seen exactly in training, nearly the
   same as a training text, or new) carry the loss?
3. **Repeats in training.** How many training reviews repeat a text exactly or nearly, and how
   many belong to a text that comes with more than one recommendation label?

"Nearly the same" means the same letters once spacing, punctuation, digits and stretched
letters are removed ("عاااالی!!" and "عالی").

## Expected

- On texts with 100 or more copies the model is within 0.05 of the floor for both heads; on
  all texts with 2 or more copies its loss is 1.3 to 1.6 times the floor.
- Recommendation: `no_idea` (about 10% of the rows) carries 35 to 50% of the loss and
  `not_recommended` 15 to 25%. Rating band: `medium` (about 18% of the rows) carries 40 to 50%.
- The mean loss is lowest for one-word reviews and highest for 4 to 15 words.
- Mean loss: exact text in train < nearly the same < new; new texts carry over 60% of the loss.
- Training reviews: 35 to 50% repeat an earlier text exactly, 3 to 8 points more nearly; 25 to
  40% belong to a text with more than one recommendation label.

## Result

`reports/loss_analysis.json`. Validation loss of the final model: recommendation 0.3121,
rating band 0.4925.

| | Recommendation | Rating band | Expected |
|---|---|---|---|
| Texts with 100+ copies: floor / model | 0.2058 / 0.2084 (14.8% of rows) | 0.4336 / 0.4373 (16.6%) | within 0.05: right |
| Texts with 2+ copies: floor / model | 0.1814 / 0.2254 (30.7%), 1.24 times | 0.3767 / 0.4460 (32.3%), 1.18 times | 1.3 to 1.6 times: a little closer |
| Small classes' share of the loss | `no_idea` 11.3% of rows, 48.7% of loss; `not_recommended` 9.6%, 21.5% | `medium` 18.8% of rows, 53.4% of loss | 35 to 50%, 15 to 25%: right; `medium` 40 to 50%: higher |
| Mean loss by length | lowest for one word (0.216), rising to 0.441 for 32+ words | 0.402 to 0.576, same order | highest for 4 to 15 words: wrong, it keeps rising |
| Mean loss: exact / nearly / new | 0.229 / 0.235 / 0.363; new texts 72.0% of the loss | 0.449 / 0.438 / 0.521; new 64.0% | order and share: right |

Training reviews: 33.5% repeat an earlier text exactly (expected 35 to 50%), 35.8% once
spacing, punctuation, digits and stretched letters are ignored (expected 3 to 8 points more,
found 2.3), and 25.5% belong to a text with more than one recommendation label (expected 25
to 40%).

What it means:

- On repeated texts the model is already at the floor: what loss is left there comes from
  the texts' own mixed labels and cannot be removed by any model that reads only the text.
- Texts that are nearly but not exactly the same as a training text are only 2.2% of
  validation, and the model scores them like exact copies. Near copies add little either to
  leakage or to removing repeats.
- The loss that can still be reduced sits in new texts (about two thirds of it), in the
  ambiguous classes (`no_idea`, `medium`: about half of each head's loss) and in longer
  reviews. This points to the other text fields of a review (title, advantages,
  disadvantages) as the next thing to check, rather than to removing repeated texts, which
  would mostly save training time.
