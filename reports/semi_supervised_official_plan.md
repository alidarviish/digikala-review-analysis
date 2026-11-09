# Semi-supervised learning as the project defines it (written before running)

## Why a second run

The pseudo-label run in `reports/semi_supervised_plan.md` was a pilot: it applied the 0.90
threshold to probabilities scaled with a temperature fitted on validation labels, trained the
student from FaBERT's pretrained weights on 500,000 reviews, and did not compare on the test
set. This run follows the project's protocol step by step, on the recommendation head only.

## Steps

1. **Supervised model (the teacher).** The kept FaBERT run (`artifacts/transformer_no_early_stop`,
   fine-tuned on `train_supervised`, seed 2026). Nothing is trained in this step.
2. **Pseudo-labels.** The teacher scores the 736,134 hidden reviews of
   `train_pseudo_unlabeled` (normalized as every other text). For each review the class with
   the highest probability is taken, from the model's own softmax: no temperature, no offsets,
   nothing fitted on any label. A guess is accepted when that probability is at least 0.90
   (`PSEUDO_LABEL_THRESHOLD`); the others are left out. The hidden reviews' true labels are not
   read in this step.
3. **Semi-supervised model (the student).** A copy of the supervised model, fine-tuned further
   for one epoch on `train_supervised` plus the accepted pseudo-labeled reviews, shuffled
   together. Pseudo-labeled reviews count only for the recommendation loss; their rating band
   loss is masked (their band was hidden with them). Settings identical to the supervised run:
   AdamW, learning rate 3e-5, weight decay 0.01, 6% warmup then linear decay, batch 64, 128
   tokens, dropout 0.1, bfloat16, seed 2026; four checks per epoch on the same 50,000-review
   validation sample, and the best check is kept. The starting model is scored on the sample
   first, as check 0, so if no check beats it the starting weights are kept and the report says
   so.
4. **Control.** The same continued fine-tuning, with the same settings and seed, on
   `train_supervised` alone. Without it, a difference between the student and the supervised
   model could come from the extra epoch rather than from the pseudo-labels.
5. **Quality of the pseudo-labels**, after training (allowed by the project): the guesses
   against the reference labels of the hidden reviews (`data/splits/audit/masked_labels_reference.csv`):
   share accepted, class shares, and accuracy overall, per true class and per guessed class.
   For the 0.80 and 0.95 thresholds the same is reported from the same guesses, without
   training (optional in the project).
6. **Comparison on validation**, now: the supervised model, the control and the student, on
   the whole validation set: accuracy, Macro F1, Macro precision, Macro recall and every
   class's precision, recall and F1, for the recommendation head (the rating band is reported
   too), with and without offsets; paired differences with intervals that draw products.
7. **Comparison on test**, once, at the end, in the single reading of the test set, with the
   same scores; the list of models scored there is fixed before the test set is read.
8. `notebooks/04_semi_supervised.ipynb` shows steps 2 to 7 from the saved reports, with the
   fixed 0.90 threshold.

Only `train_supervised`, the hidden reviews' text and the validation set are used to train and
choose; no validation or test label makes or selects a pseudo-label.

## Rule (fixed now)

The pseudo-labels help if the student beats the control on recommendation Macro F1 with
offsets, on the whole validation set, with the whole product-level 95% interval above zero
and by more than 0.0012 (the spread of the kept run over three training seeds,
`reports/full_seeds_plan.md`). Otherwise the answer is "no measurable gain". The student
against the supervised model is reported as the project asks, whatever it shows. The final
system changes only if the rule is met, and only after talking it over.

## Expected

- Accepted at 0.90: 75 to 82% of the hidden reviews (more than the pilot's 69.1%, because the
  model's own probabilities are more confident than the temperature-scaled ones), and 96 to 98%
  of the accepted guesses right.
- Accepted classes: 93 to 97% `recommended`, 3 to 6% `not_recommended`, under 1% `no_idea`
  (against 11.3% `no_idea` among the hidden reviews, known from the pilot's check).
- Control against the supervised model: within 0.1 point on both heads (the kept run was
  already at its best check, 2.75 epochs, and the validation loss rose after epoch 3).
- Student against the control: a tie on recommendation (within 0.1 point), `no_idea` F1 a
  little lower, rating band unchanged; so "no measurable gain", as in the pilot.
- Time on the molab GPU: about 10 minutes for the pseudo-labels, about 40 minutes for the
  student (about 4.1 million reviews) and 32 for the control; the two can run side by side.

The test set is not read until the final evaluation.

## Added after the first run, before running again

The first run (commit e86f8eb) is recorded but not imported: 72.4% of the hidden reviews
reached 0.90 (533,043 pseudo-labeled reviews), and both the student and the control kept
check 0. No check beat the starting model on the 50,000-review sample (0.7126): the student's
checks were 0.7099, 0.7094, 0.7108 and 0.7097, the control's 0.7078, 0.7086, 0.7079 and
0.7079, and the validation loss rose from 0.808 to about 0.82. So by the rule the answer is
already "no measurable gain", but the semi-supervised model it selects is the supervised model
itself, and the comparison the project asks for (supervised against semi-supervised, on
validation and test) would compare a model with itself.

Added now: the student and the control are trained again exactly as before, and the weights at
the end of the epoch are also kept and scored on the whole validation set
(`model_last.pt`, `validation_predictions_last.parquet`, "last_check" in the report). The best
check is still chosen and the rule above is unchanged. The end-of-epoch student is the model the
semi-supervised training actually made, and it is the one compared with the supervised model as
the project asks (on validation now, on test at the end); the end-of-epoch control shows how
much of any difference is the extra epoch. Report only.

Expected for the end-of-epoch models, on the whole validation set, recommendation Macro F1 with
offsets: the control 0.2 to 0.5 point under the supervised model, the student 0.1 to 0.4 under
it, and the student 0.1 to 0.3 above the control (on the sample the student was above the
control at every check). Rating band: both within 0.3 point of the supervised model.

## Result

`reports/semi_supervised.json` (commit 2b57540). The guesses: step 99, commit e86f8eb (the
scoring code did not change between the two commits); the student and the control: steps 100
and 101 run again with the end-of-epoch weights kept. All three started from this machine's
kept weights (hash 2828ba5); the guesses use the model's own softmax (temperature 1.0); the
student read exactly the guesses that were checked.

Pseudo-labels, against the reference labels of the hidden reviews (79.1% `recommended`,
11.3% `no_idea`, 9.6% `not_recommended`), read after training:

| Threshold | Kept | Right | Kept classes: recommended / not_recommended / no_idea | Kept and right, share of each true class |
|---|---|---|---|---|
| 0.80 | 79.8% | 95.5% | 92.6% / 6.9% / 0.5% | 89.9% / 50.1% / 2.9% |
| **0.90 (used)** | **72.4% (533,043)** | **96.9%** | 94.9% / 5.0% / 0.1% | 84.4% / 34.7% / 0.4% |
| 0.95 | 60.5% | 98.0% | 96.4% / 3.6% / 0.0% | 72.3% / 21.6% / 0.0% |

The extra epoch: neither the student nor the control found a check better than the starting
model on the 50,000-review sample (0.7126); the student's checks were 0.7072 to 0.7124, the
control's 0.7070 to 0.7104. So both kept check 0, and the semi-supervised model chosen by the
rule is the supervised model itself.

Whole validation set, recommendation (788,717 reviews), with offsets unless said:

| Model | Accuracy | Macro F1 | Macro precision | Macro recall | Macro F1 without offsets | F1: recommended / not_recommended / no_idea |
|---|---|---|---|---|---|---|
| Supervised (kept FaBERT) | 0.8736 | 0.7324 | 0.7346 | 0.7320 | 0.7253 | 0.944 / 0.744 / 0.510 |
| Control, end of epoch | 0.8731 | 0.7300 | 0.7342 | 0.7278 | 0.7174 | 0.944 / 0.739 / 0.507 |
| Student, end of epoch | 0.8756 | 0.7310 | 0.7359 | 0.7268 | 0.7179 | 0.945 / 0.744 / 0.503 |

| Difference in Macro F1 with offsets [95%, products drawn] | Recommendation | Rating band |
|---|---|---|
| Student minus control, best checks (the rule) | +0.0000 | +0.0000 |
| Student minus control, end of epoch | +0.0010 [+0.0002, +0.0017] | +0.0018 [+0.0012, +0.0025] |
| Student, end of epoch, minus supervised | -0.0015 [-0.0023, -0.0007] | -0.0001 [-0.0007, +0.0006] |
| Control, end of epoch, minus supervised | -0.0024 [-0.0033, -0.0016] | -0.0019 [-0.0026, -0.0013] |

Verdict by the rule: **no measurable gain**. The student did not beat the control: by the rule
both are the starting model.

| | Expected | Found |
|---|---|---|
| Accepted at 0.90 | 75 to 82% | 72.4%: wrong, a little fewer |
| Accepted guesses right | 96 to 98% | 96.9%: right |
| Accepted classes | 93 to 97% recommended, 3 to 6% not_recommended, under 1% no_idea | 94.9%, 5.0%, 0.1%: right |
| Control against supervised (best checks) | within 0.1 point | the same model: right |
| Student against control (best checks) | a tie | the same model: right |
| End of epoch: control under supervised | 0.2 to 0.5 point | 0.24: right |
| End of epoch: student under supervised | 0.1 to 0.4 point | 0.15: right |
| End of epoch: student above control | 0.1 to 0.3 point | 0.10: right, at the edge |
| End of epoch: rating band within 0.3 point | yes | -0.01 and -0.19: right |
| `no_idea` F1 a little lower with the pseudo-labels | yes | 0.503 against 0.507 (control) and 0.510: right |

What it means:

- The model was already at its best: one more epoch at the same learning rate makes it worse
  (control: -0.24 point on recommendation), with or without pseudo-labels.
- The pseudo-labels soften that drop (the student ends 0.10 point above the control on
  recommendation and 0.18 on rating band, measurable), but the semi-supervised model ends 0.15
  point under the supervised one on recommendation and ties it on rating band.
- The confident guesses are almost all `recommended` (94.9%) and right 96.9% of the time, but
  they cover only 0.4% of the hidden `no_idea` reviews: they add reviews of the class the model
  already knows, so recall moves towards `recommended` (accuracy 0.8756 against 0.8736, `no_idea`
  recall 0.515 against 0.540) and Macro F1 does not rise.
- The answer to the project's question, "can unlabeled reviews improve the classifier?", is no
  for this model and protocol, as in the pilot. The final system does not change.
- The scores on the sample were not identical between the first and the second run with the
  same seed (for example the student's first check 0.7099 and 0.7072): training on the GPU is
  not exactly repeatable, which is why the decisions rest on intervals and on the seed spread.

The end-of-epoch student is the semi-supervised model compared with the supervised one on the
test set at the end.
