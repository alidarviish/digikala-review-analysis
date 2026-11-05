# Domain adaptation: FaBERT pretrained further on the review texts (written before running)

## The question

FaBERT was pretrained on general Persian text. The reviews are short, informal and full of
shop words, misspellings and colloquial forms. If FaBERT first learns more about this kind of
text, by the same masked-word task it was pretrained with, and is fine-tuned after that, does
it classify better?

Every training change so far tied or lost (`reports/hard_examples_plan.md`,
`reports/difficulty_plan.md`, `reports/learning_rate_plan.md`, `reports/semi_supervised_plan.md`),
and a model three times as large gained 0.08 point (`reports/large_final_plan.md`). This one
changes what the encoder knows before fine-tuning, not how it is fine-tuned.

## Method

1. **Further pretraining** (step 68, `scripts/adapt_pretrain.py`, GPU). Only the text of the
   training reviews: the 3,552,498 labeled ones and the 736,134 whose recommendation label
   was hidden. No label is read; the validation and test texts are not used. A fixed 1% of
   the texts is held out to measure the masked-word loss before and after, with the same
   masks both times.
   - Each text is tokenized as for fine-tuning (at most 128 tokens, with its special
     tokens); the texts are joined end to end and cut into blocks of 128 tokens, so no
     compute goes to padding (the median review is 7 tokens).
   - One pass over the blocks, batch 128, learning rate 5e-5 with 6% warmup and a linear
     decay, weight decay 0.01; 15% of the tokens masked as in BERT, a new mask for every batch.
2. **Pilot** (step 69). FaBERT fine-tuned exactly as in step 41 (500,000 training reviews, 1
   epoch, seeds 2026 to 2028, all of validation), but starting from the further pretrained
   weights; the hash of those weights is part of each run's settings.
3. **Comparison** (step 70) with step 41 on the same validation reviews by the usual rule:
   paired bootstrap per seed; a lead smaller than the spread between seeds is a tie, and a
   tie goes to the usual training.
4. **Full run, only if the pilot wins.** FaBERT is then fine-tuned again from these weights
   on all 3.55 million reviews, exactly as the kept run, compared with it on the whole
   validation set, and combined with TF-IDF again.

The test set is not read at any step.

## Expected

- The held-out masked-word loss after is 10 to 30% lower than before.
- The further pretraining takes 15 to 45 minutes on the molab GPU; the pilot about 5 minutes
  per seed, as in step 41.
- The pilot ties with step 41 (within the spread between seeds), at most 0.3 point better on
  the mean of the two heads; no class moves by more than 0.5 point of F1. The model already
  learns the shop words from 3.55 million labeled reviews during fine-tuning, and the limit
  measured so far is in the labels (`reports/label_sample_plan.md`), which this does not touch.
- So the full run is most likely not made.

## Result

`reports/domain_adaptation/pretrain.json` (step 68), `reports/transformer_pilot_whole/adapted_n500000.json`
(step 70), the card in `reports/figures/experiments/domain_adapted`.

Further pretraining: 4,288,632 texts (42,886 held out), 438,622 blocks of 128 tokens, 3,427
steps in 5.4 minutes, peak GPU memory 15.4 GB. The held-out masked-word loss fell from 4.047
to 2.103, 48% lower.

The pilot against step 41, three seeds:

| Head | Step 41 | Further pretrained | Difference per seed [95%] |
|---|---|---|---|
| recommendation | 0.7235, 0.7237, 0.7241 | 0.7257, 0.7263, 0.7254 | +0.0022 [+0.0014, +0.0029]; +0.0026 [+0.0019, +0.0033]; +0.0013 [+0.0006, +0.0021]: better in all three |
| rating band | 0.6858, 0.6867, 0.6864 | 0.6886, 0.6883, 0.6888 | +0.0028 [+0.0022, +0.0034]; +0.0016 [+0.0010, +0.0022]; +0.0023 [+0.0018, +0.0029]: better in all three |

Mean Macro F1 of the two heads: 0.7072 against 0.7051, lead +0.0021, spread between seeds
0.0006: a lead, so the further pretrained start is chosen. The validation loss at the end is
lower on both heads (0.3157 against 0.3187, 0.4951 against 0.4974). Each seed took 4.4
minutes, as before.

F1 per class with offsets, mean of three seeds, step 41 -> further pretrained: `recommended`
0.943 -> 0.944, `not_recommended` 0.732 -> 0.735, `no_idea` 0.496 -> 0.498; `low` 0.702 ->
0.708, `medium` 0.472 -> 0.472, `high` 0.885 -> 0.886.

| | Expected | Found |
|---|---|---|
| Held-out masked-word loss | 10 to 30% lower | 48% lower: wrong, much more |
| Time of the further pretraining | 15 to 45 minutes | 5.4 minutes: wrong, faster |
| Pilot | a tie, at most 0.3 point better | 0.21 point better, every seed and head above zero: wrong, a win |
| Classes | none moves by more than 0.5 point | largest `low` +0.6: wrong for one class |

What it means:

- This is the first training change that wins by the rule, on every seed and both heads.
  Every earlier change tied or lost (`reports/hard_examples_plan.md`,
  `reports/difficulty_plan.md`, `reports/learning_rate_plan.md`,
  `reports/semi_supervised_plan.md`), and a model three times as large gained 0.08 point.
- FaBERT knew the language of the reviews much less well than expected: one pass over their
  text halved its masked-word loss in five minutes. What helps is knowing this kind of text,
  not a bigger model or a different way of fine-tuning.
- The gain spreads over most classes and is largest on `low` and `not_recommended`;
  `no_idea` and `medium`, where the labels limit any model (`reports/label_sample_plan.md`),
  barely move.
- By the plan, the full run is made next: FaBERT fine-tuned from these weights on all 3.55
  million reviews, exactly as the kept run, compared with it on the whole validation set and
  combined with TF-IDF again.

## The full run (written after the pilot, before running)

Step 74 (`scripts/train_transformer.py --adapted`, GPU): FaBERT starts from the further
pretrained weights and is trained exactly as the kept run (`artifacts/transformer_no_early_stop`):
all 3,552,498 training reviews, 4 epochs, four checks per epoch on the same 50,000 validation
reviews, the best check kept. The hash of the starting weights is part of the settings. If the
molab machine was reset and the weights of step 68 are gone, step 68 is run again first; its
new hash is then recorded.

Step 75 (`scripts/compare_fabert_runs.py --adapted`, on your own machine), on the whole
validation set:

1. The new run against the kept FaBERT run, each with its cross-fitted offsets. Rule: it
   replaces the kept run if the whole 95% interval is above zero on both heads (same model,
   same cost).
2. Each run combined with the final TF-IDF model as in step 60, and the two combinations
   compared on the same reviews. Rule: the new combination becomes the final system if the
   whole interval is above zero on both heads.

Expected:

- The kept check is in epoch 2 to 3.5, as for the kept run (2.75); training takes 2 to 2.5
  hours (the kept run took 121 minutes).
- The gain is smaller than on the pilot, since with seven times as many labeled reviews the
  usual run learns more of the review language during fine-tuning: 0.05 to 0.2 point on each
  head, with the interval above zero on at least one head.
- Combined with TF-IDF, the gain shrinks a little more (TF-IDF already adds some of the
  review-specific words): 0 to 0.15 point.

### Result of the full run

`reports/transformer_final/adapted.json` and `reports/transformer_final/adapted_vs_fabert.json`,
whole validation set, Macro F1 with cross-fitted offsets.

The machine had been reset, so step 68 was run again first: held-out masked-word loss 4.047 to
2.103 (2.103 before), 5.5 minutes (5.4 before), new weights `ddc497a6...` (`3132dcf6...` before;
a GPU run is not exactly repeatable). Step 74 started from the new weights and records their
hash; the pilot weights are kept outside git.

| Head | Kept FaBERT | Further pretrained | Difference [95%] | With TF-IDF: kept / further pretrained | Difference [95%] |
|---|---|---|---|---|---|
| recommendation | 0.7324 | 0.7337 | +0.0012 [+0.0004, +0.0021]: better | 0.7344 / 0.7345 | +0.0001 [-0.0007, +0.0008]: no measurable difference |
| rating band | 0.6922 | 0.6931 | +0.0009 [+0.0002, +0.0016]: better | 0.6935 / 0.6944 | +0.0008 [+0.0002, +0.0015]: better |

Training took 127 minutes; the kept check is at epoch 2.75 (step 152,647), as for the kept run.
Validation loss: 0.3125 against 0.3121 (recommendation), 0.4921 against 0.4925 (rating band).
F1 per class, kept -> further pretrained: `recommended` 0.944 -> 0.944, `not_recommended` 0.744
-> 0.744, `no_idea` 0.510 -> 0.513; `low` 0.711 -> 0.713, `medium` 0.479 -> 0.479, `high`
0.887 -> 0.888.

| | Expected | Found |
|---|---|---|
| Kept check | epoch 2 to 3.5 | 2.75: right |
| Training time | 2 to 2.5 hours | 127 minutes: right |
| Model alone | 0.05 to 0.2 point per head, above zero on at least one | 0.12 and 0.09 point, above zero on both: right, a little better |
| With TF-IDF | 0 to 0.15 point | 0.01 and 0.08 point: right |

Decision, by the rules written above:

1. The interval is above zero on both heads, so the further pretrained run replaces the kept
   FaBERT run as the best single model (0.7337 and 0.6931).
2. Combined with TF-IDF, the interval is above zero for rating band but not for
   recommendation, so the final system stays as it is: the kept FaBERT run combined with TF-IDF
   (0.7344 and 0.6935). The rule asks for both heads, and it is not changed after the result.

What it means:

- The gain of the pilot holds on the full run, smaller, as expected: knowing the review language
  helps a little even after fine-tuning on 3.55 million reviews.
- TF-IDF already gives the combination most of what the further pretraining adds on
  recommendation; on rating band a small part of the gain is left.
- `medium` does not move at all, alone or combined, as the label analyses predict
  (`reports/rating_boundaries.json`, `reports/rate_recommendation.json`,
  `reports/label_conflict.json`): its errors are limited by the labels, not by how well the model
  knows the language.
