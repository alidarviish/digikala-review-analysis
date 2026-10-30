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
