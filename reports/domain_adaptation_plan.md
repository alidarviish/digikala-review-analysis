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
