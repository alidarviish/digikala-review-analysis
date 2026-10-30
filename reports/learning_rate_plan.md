# Learning rate and layer-wise decay (written before running)

Every Transformer run so far used one learning rate, 3e-5 (2e-5 for TookaBERT-Large), taken
from common practice and never tuned. `reports/transformer_model_choice.md` also left a run
open: training the lower layers with smaller learning rates (layer-wise decay). Both are
measured here on the FaBERT pilot, so the choice of the final settings rests on a measurement.

## Method

Steps 56 to 58 of `notebooks/run_all.py` train FaBERT as in step 41 (the same 500,000 training
reviews, seeds 2026 to 2028, 1 epoch, all of validation, the validation loss measured 10
times), with one change each:

| Step | Run | Option |
|---|---|---|
| 56 | learning rate 2e-5 | `--learning-rate 2e-5` |
| 57 | learning rate 5e-5 | `--learning-rate 5e-5` |
| 58 | layer-wise decay 0.9: the heads keep 3e-5, each encoder layer further down gets 0.9 times the one above, the embeddings 3e-5 x 0.9^13 = 7.6e-6 | `--layer-decay 0.9` |

The warm-up and the linear fall of the learning rate stay as they are, for every layer.
Step 59 (`scripts/compare_hard_examples.py --set learning_rate`) compares each run with step
41 by the same rule as before: per seed on the same reviews with cross-fitted offsets, and a
lead smaller than the spread between seeds is a tie, which goes to the usual setting. Each run
gets an experiment card.

## Expected

- Learning rate 2e-5: a tie or slightly worse (0 to 0.003 lower): with one epoch, a smaller
  step learns a little less.
- Learning rate 5e-5: a tie (within 0.002); the validation loss falls faster at the first
  checks.
- Layer-wise decay 0.9: a tie (within 0.002). It mainly protects what the model learned before
  fine-tuning, which matters most with little data; here there are 500,000 reviews.
- No class changes by more than 0.5 points of F1.

If one of them wins by the rule, the final FaBERT model is trained again with it and compared
with the kept one on the whole validation set, and TookaBERT-Large, which is trained once as an
upper bound, uses the same setting.

## Result

`reports/transformer_pilot_whole/learning_rate_n500000.json`, the cards in
`reports/figures/experiments/learning_rate_*` and `layer_decay_0.9`. Mean Macro F1 of the two
heads over three seeds; the usual training (step 41, learning rate 3e-5) scores 0.7051.

| Run | Mean Macro F1 | Lead | Spread between seeds | Verdict | Minutes per seed | Expected |
|---|---|---|---|---|---|---|
| learning rate 2e-5 | 0.7041 | -0.0010 | 0.0008 | worse | 4.3 | a tie or 0 to 0.003 lower: right |
| learning rate 5e-5 | 0.7043 | -0.0008 | 0.0009 | tie | 4.4 | a tie: right |
| layer decay 0.9 | 0.7032 | -0.0019 | 0.0010 | worse, in all six seeds and heads | 5.4 | a tie: wrong |

Change in F1 per class (points, three seeds pooled; outside the 95% interval unless marked):

| Run | recommended | not_recommended | no_idea | low | medium | high |
|---|---|---|---|---|---|---|
| learning rate 2e-5 | 0.00 (no difference) | -0.28 | -0.09 | -0.07 | -0.27 | +0.13 |
| learning rate 5e-5 | +0.01 | -0.14 | -0.20 | +0.11 | -0.22 | -0.04 |
| layer decay 0.9 | +0.03 | -0.44 | -0.33 | -0.10 | -0.29 | -0.01 (no difference) |

No class moved by more than 0.5 points, as expected. The validation loss at the end is within
0.002 of the usual one in every run.

What it means:

- 3e-5 is the best of the three learning rates: a smaller one and a larger one both score
  lower, so the usual setting sits near the best value for one epoch on this data.
- Layer-wise decay is worse and a quarter slower (each parameter gets its own group in the
  optimizer). Slowing the lower layers costs more than it protects with 500,000 reviews.
- Kept: the usual settings. The final FaBERT model is not trained again.
