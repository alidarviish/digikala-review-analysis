# Choosing the Transformer model (written before any of it was run)

## Candidates

| Model | Size | Why it is a candidate | Published sentiment score |
|---|---|---|---|
| PartAI/TookaBERT-Base | about 0.1B | Newest Persian BERT, trained on over 500 GB of Persian text including forums and blogs | DeepSentiPers 83.9 F1 |
| sbunlp/fabert | 124M | Trained on 20 million Persian blog posts, much of it informal; WordPiece adapted to the half-space | MirasOpinion 87.5, DeepSentiPers 79.9 macro F1 |
| FacebookAI/xlm-roberta-base | 278M (mostly embeddings) | The usual multilingual reference | DeepSentiPers 84.0 F1 |
| jhu-clsp/mmBERT-base | 307M (110M without embeddings) | Modern encoder (ModernBERT design), 1800+ languages including Persian, up to 8192 tokens | No separate Persian number found |
| HooshvareLab/bert-fa-zwnj-base | about 110M | The classic Persian reference (ParsBERT) | DeepSentiPers 80.2 F1 |

Left out: TookaBERT-Large (access on request, three times larger; reconsidered if GPU time
allows), Persian-ModernBert (no published results), and any model already fine-tuned on
Digikala reviews (it may have seen our validation and test texts).

Published scores differ by about a point between the best candidates, and on other data,
so the choice is made on our own reviews in two steps.

## Step 1: tokenizers (`scripts/compare_tokenizers.py`)

50,000 labeled validation reviews, normalized as in training. Expected, before running:

- Persian tokenizers (TookaBERT, FaBERT, ParsBERT) cut words into fewer pieces than the
  multilingual ones (XLM-R, mmBERT).
- With any tokenizer, fewer than 5% of reviews are longer than 128 tokens and fewer than 1%
  longer than 256 (99% of reviews have at most 62 words).
- Unknown tokens are rare (well under 1%) for all but possibly ParsBERT, whose vocabulary is
  the oldest.

## Step 2: a small equal fine-tuning run (`scripts/pilot_transformers.py`, on a GPU)

Step 1 left every candidate in (reports/tokenizers/compare.json): at most 0.4% of reviews
are longer than 128 tokens with any tokenizer, and unknown tokens are under 0.3%. The
Persian tokenizers use about 20% fewer tokens than XLM-R and mmBERT. TookaBERT-Large is
added, as the GPU is large enough.

Same settings for every model: 100,000 training reviews and 50,000 validation reviews
(fixed samples, seed 2026); one encoder with two linear heads on the first token; the sum
of the two cross-entropies, without class weights; full fine-tuning, 1 epoch, batch 64,
learning rate 3e-5 (2e-5 for the large model), AdamW with weight decay 0.01, 6% warmup then
linear decay, 128 tokens; per-class offsets tuned on the other half of the sample, as for
TF-IDF. The final TF-IDF model is scored on the same 50,000 reviews, and each model is
compared with it (paired bootstrap). A sample of 50,000 is enough to see a difference of
about one point (reports/tfidf_analysis/sample.json).

Expected, before running:

- TookaBERT-Large scores highest.
- TookaBERT-Base, FaBERT and XLM-R are within one point of each other.
- ParsBERT is the lowest of the Persian models.
- Even with 100,000 training reviews, the best model matches or beats the final TF-IDF
  model (trained on all of train) with offsets, most of all on the rating band.
- A base model takes under 10 minutes, the large one two to three times as long.

The best model becomes the main model, fine-tuned afterwards on more data.

## Result of step 2 (one seed)

Macro F1 with offsets on the same 50,000 validation reviews (reports/transformer_pilot/):

| Model | Recommendation | Rating band | Mean | Minutes |
|---|---|---|---|---|
| TookaBERT-Large | 0.7154 | 0.6772 | 0.6963 | 2.2 |
| FaBERT | 0.7003 | 0.6795 | 0.6899 | 0.9 |
| TookaBERT-Base | 0.7050 | 0.6702 | 0.6876 | 0.9 |
| XLM-R base | 0.7037 | 0.6708 | 0.6873 | 1.2 |
| mmBERT-base | 0.6963 | 0.6736 | 0.6850 | 1.8 |
| ParsBERT | 0.6873 | 0.6592 | 0.6733 | 0.9 |

Four of five expectations held; the large model led on the mean and on recommendation but
not on the rating band. ParsBERT is dropped (lowest), and so are XLM-R and mmBERT (slower and
not better). The lead of the large model (0.6 points on the mean) is within what one
training run can change, so the choice is not yet made.

## Step 3: seeds and more data (written before running)

- TookaBERT-Large, FaBERT and TookaBERT-Base again with training seeds 2027 and 2028 (same
  samples, so only the training randomness changes). With seed 2026, three runs per model.
- TookaBERT-Large and FaBERT on 500,000 training reviews (seed 2026).

Expected:

- Between seeds, Macro F1 moves by about 0.3 to 0.5 points.
- The large model stays ahead on the mean of three seeds and on recommendation; on the
  rating band it is level with FaBERT.
- With 500,000 reviews every model gains one to two points, the large one the most.

The model with the best mean over seeds becomes the main model; a lead smaller than the
spread between seeds counts as a tie, and a tie goes to the faster model.

## Result of step 3 and the choice

Mean of three seeds with 100,000 training reviews: TookaBERT-Large 0.6949 (seeds 0.6910 to
0.6974), FaBERT 0.6925 (0.6899 to 0.6947), TookaBERT-Base 0.6875. With 500,000 reviews:
FaBERT 0.7057 in 4.3 minutes, TookaBERT-Large 0.7037 in 11 minutes. On the same 50,000
reviews FaBERT with 500,000 reviews beats the final TF-IDF model by 2.7 points on
recommendation and 1.7 on the rating band, both intervals above zero
(reports/transformer_pilot/summary.json).

The seeds moved the large model by up to 0.6 point on the mean (expected 0.3 to 0.5), and
the large model did not gain more from more data. Its lead over FaBERT (0.25 point) is
smaller than the spread between seeds, so by the rule above it is a tie, and FaBERT is
chosen: 2.6 times faster, a third of the memory, and fewer tokens per review.

## Training the chosen model (written before running)

`scripts/train_transformer.py`: FaBERT on every labeled training review, with the settings
of the comparison runs, up to 4 epochs. Four checks per epoch on the same 50,000 validation
reviews (mean Macro F1 of the two heads with offsets); the best check is kept, and training
stops after two checks in a row without gain. The kept model then scores the whole
validation set and is compared with the final TF-IDF model on the same reviews. The test set
is not read.

Expected:

- The best check comes in the second epoch.
- On the whole validation set, with offsets: recommendation about 0.73 to 0.74 and rating
  band about 0.69 to 0.70, about 3 and 2.5 points above TF-IDF.
- no_idea and medium stay the weakest classes, with higher recall than with TF-IDF.

If the score is still rising at the end of the fourth epoch, more epochs are tried in a
separate run.

## Two runs added after training (written before running them)

**Pilot on the whole validation set** (`pilot_transformers.py --whole-validation`). The
pilot runs were scored on the 50,000-review sample. The same runs (both models, three seeds,
500,000 reviews) are repeated and scored on all validation reviews, with results kept apart
in `reports/transformer_pilot_whole/`. Expected: every score within 0.3 point of the sample
score, and the same choice (FaBERT), since the large model's lead was smaller than the seed
spread.

**FaBERT for 4 epochs without early stopping** (`train_transformer.py --no-early-stop`).
The first run stopped at epoch 1.75 after two checks without gain, so nobody knows what
happens later. This run uses the same settings, trains all 4 epochs and keeps the best
check. The validation loss of each head is now logged at every check. Expected:

- the best check comes in epoch 3 or 4, about 0.3 to 0.8 point above the first run;
- the validation loss starts to rise after epoch 2 while Macro F1 stays flat or rises a
  little (the model grows more confident on the reviews it gets wrong).

It replaces the first run only if, on the whole validation set with offsets
(`scripts/compare_fabert_runs.py`), the whole 95% interval of the difference is above zero,
judged for each head separately. If the best
check is the last one and the score is still rising, 6 to 8 epochs are tried next. Training
the lower layers with smaller learning rates (layer-wise decay) is a separate run after
this one.

## Result of the pilot on the whole validation set

`reports/transformer_pilot_whole/summary.json`. The prediction was wrong: 10 of the 28
scores moved by more than 0.3 point from the sample score, the largest by 0.9 point (FaBERT
and mmBERT on recommendation). On the whole validation set the seeds move a model by only
about 0.1 point, so most of the seed spread seen before was noise of the 50,000-review
sample. Mean of three seeds with 100,000 reviews: TookaBERT-Large 0.6956 (0.6951 to
0.6960), FaBERT 0.6929 (0.6925 to 0.6936), TookaBERT-Base 0.6856 (0.6841 to 0.6868). Here
the lead of the large model (0.26 point) is larger than the spread, so by the rule it is no
longer a tie. With 500,000 reviews and one seed: TookaBERT-Large 0.7050, FaBERT 0.7043,
both about 2 points above TF-IDF on each head with the whole interval above zero.

## TookaBERT-Large against FaBERT on 500,000 reviews (written before running)

The choice has to be settled before FaBERT is trained for longer, and the model is trained
on all 3.5 million reviews in the end, so the comparison is made at 500,000 reviews with
three seeds, as at 100,000: `pilot_transformers.py` with seeds 2027 and 2028 for both models
(step 35), then `scripts/compare_pilot_models.py` (step 36), which compares the two models
directly on the same reviews for each seed (paired bootstrap, per head) and applies the
rule above to the mean over the three seeds.

Expected:

- the lead of the large model stays below 0.15 point and below the spread between seeds,
  so it is a tie and FaBERT stays the model;
- per seed, the interval of the difference includes zero for recommendation; for the
  rating band it may be slightly in favour of the large model.

If the large model wins by the rule, it replaces FaBERT, and the longer training run (step
33) is made with it instead.

## Result of TookaBERT-Large against FaBERT on 500,000 reviews, and the choice

`reports/transformer_pilot_whole/large_vs_fabert_n500000.json`. TookaBERT-Large minus FaBERT
on the same reviews:

| Seed | Recommendation [95%] | Rating band [95%] |
|---|---|---|
| 2026 | +0.0001 [-0.0008, +0.0009] | +0.0013 [+0.0007, +0.0020] |
| 2027 | +0.0017 [+0.0008, +0.0026] | +0.0013 [+0.0006, +0.0020] |
| 2028 | +0.0019 [+0.0011, +0.0028] | +0.0008 [+0.0002, +0.0015] |

Mean over the three seeds: TookaBERT-Large 0.7058 (0.7050 to 0.7062, 11.0 minutes), FaBERT
0.7046 (0.7043 to 0.7049, 4.3 minutes). The lead of the large model (0.00117) is just below
the spread between seeds (0.00124), so by the rule it is a tie and FaBERT stays the model.
The predictions of a tie and of a small gain for the large model on the rating band were
right; the interval for recommendation included zero for one seed only, not for all three.

The result sits at the edge of the rule: the large model is slightly better on every seed,
by 0.1 to 0.2 point, and the seed ranges of the two models do not overlap. FaBERT is kept
because the rule was fixed before the runs, the gain is far smaller than the label noise
measured earlier, and the large model costs 2.6 times the training time, three times the
memory and a slower service. If time and GPU allow once every step is done with FaBERT,
TookaBERT-Large is trained once with the final settings and reported as an upper bound.
