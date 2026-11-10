# The final evaluation on the test set (written before the test set is read)

## Why now, and what is read

Every required model is trained and every choice is made on validation (D73). The test set
is read now, once, for the models below and no other; nothing is fitted, tuned or chosen on
it. A later extra experiment with inputs other than `body` (D70) may read it once more for one
model chosen before, reported apart; it cannot change anything here.

## The models (fixed now)

| Model | Why it is scored on test | Weights, offsets, threshold |
|---|---|---|
| **Final system** (official result) | the system the project delivers | FaBERT and TF-IDF mixed: weight, offsets and "uncertain" threshold of `reports/uncertain.json`, "for_test" |
| FaBERT (the supervised model) | compared with TF-IDF "under the same evaluation protocol", and with the semi-supervised model "on the same test set" | the kept run (seed 2026), its offsets for the test set |
| TF-IDF | the baseline compared with FaBERT | `artifacts/tfidf_final/model.joblib`, its offsets for the test set |
| Semi-supervised student | the project's comparison of supervised and semi-supervised models on test | the student at the end of its epoch (`model_last.pt`), its offsets for the test set |

The other models (BiLSTM, TookaBERT-Large, the further pretrained FaBERT, other mixtures) are
reported on validation only.

## Method

1. `scripts/score_test_set.py` (steps 106 and 107, molab GPU): FaBERT and the student score the
   test texts. Only `id` and the normalized text are read, no label. Same scoring as for
   validation: the tokenizer of hub commit 4cad2850ab81, 128 tokens, bfloat16. The weights come
   from the cache folder (`fabert_final.pt`, `semi_student_last.pt`); their hashes are recorded.
2. `scripts/final_evaluation.py` (step 108, your machine): checks that the scored weights are
   this machine's kept files and that the predictions are unchanged, scores TF-IDF, reads the
   test labels (here only) and computes everything below. It runs once: it stops if its report
   exists.

For every model and head, on all test reviews that have that head's label:

- accuracy, Macro F1, Macro precision, Macro recall, balanced accuracy, precision, recall and F1
  of each class, the confusion matrix; for the rating band the ordinal distance error (mean
  number of bands off, and the share off by two);
- with offsets (the official labels) and without;
- a 95% interval of Macro F1 that draws products (1,000 rounds, seed 2026);
- the validation scores recorded before, next to the test ones.

Paired differences, with intervals that draw products and the change of every score:
FaBERT minus TF-IDF, the final system minus FaBERT, and the student minus FaBERT (the change
after adding pseudo-labeled data, which the project asks for).

For the final system also: the share answered "uncertain", and its scores on the reviews it
answers. The coverage curve stays on validation (`reports/uncertain.json`).

Tables and figures, made from `reports/final_evaluation.json` with the other figures: the four
models on both heads with every score; the confusion matrices; validation against test; the
supervised and semi-supervised models on test with the change of each score.

## Expected

Validation, with offsets (Macro F1): TF-IDF 0.7017 and 0.6683, FaBERT 0.7324 and 0.6922, the
final system 0.7344 and 0.6935, the student 0.7310 and 0.6921 (recommendation and rating band).

- Every model's test Macro F1 within 0.5 point of its validation one: the split is by product
  and the parts are balanced on both labels.
- The same order on both heads: the final system, then FaBERT, then TF-IDF; FaBERT 2 to 3.5
  points above TF-IDF on both heads, with the whole interval above zero.
- The final system above FaBERT by 0.05 to 0.3 point on both heads.
- The student 0 to 0.3 point under FaBERT on recommendation.
- 9.5 to 10.5% of the reviews answered "uncertain" on each head.

The test set has not been read.

## Result

`reports/final_evaluation.json` (commit f2fd9cf; the test texts were scored on molab by steps 106
and 107 at commit 622c56b, with this machine's kept weights: FaBERT 2828ba5, the student
deaf0d8). The test labels were read once, by step 108.

Test set, with offsets (788,717 reviews with a recommendation label, 843,458 with a rating band):

| Model | Recommendation: accuracy | Macro F1 [95%, products] | Balanced acc. | Validation F1 | Rating band: accuracy | Macro F1 [95%, products] | Balanced acc. | Ordinal error | Validation F1 |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF | 0.8490 | 0.7021 [0.7005, 0.7036] | 0.7111 | 0.7017 | 0.7765 | 0.6684 [0.6668, 0.6700] | 0.6617 | 0.241 | 0.6683 |
| FaBERT | 0.8732 | 0.7346 [0.7331, 0.7361] | 0.7374 | 0.7324 | 0.7941 | 0.6918 [0.6903, 0.6934] | 0.6810 | 0.218 | 0.6922 |
| **Final system** | 0.8734 | **0.7349** [0.7334, 0.7364] | 0.7382 | 0.7344 | 0.7929 | **0.6927** [0.6911, 0.6942] | 0.6827 | 0.219 | 0.6935 |
| Semi-supervised student | 0.8756 | 0.7330 [0.7314, 0.7345] | 0.7292 | 0.7310 | 0.7916 | 0.6911 [0.6896, 0.6927] | 0.6807 | 0.220 | 0.6921 |

F1 of each class, with offsets:

| Model | recommended | not_recommended | no_idea | low | medium | high |
|---|---|---|---|---|---|---|
| TF-IDF | 0.928 | 0.706 | 0.472 | 0.681 | 0.449 | 0.875 |
| FaBERT | 0.943 | 0.746 | 0.514 | 0.710 | 0.478 | 0.887 |
| Final system | 0.943 | 0.748 | 0.513 | 0.711 | 0.481 | 0.885 |
| Student | 0.945 | 0.744 | 0.510 | 0.710 | 0.479 | 0.885 |

| Difference in Macro F1 with offsets [95%, products] | Recommendation | Rating band |
|---|---|---|
| FaBERT minus TF-IDF | +0.0325 [+0.0312, +0.0339] | +0.0234 [+0.0224, +0.0245] |
| Final system minus FaBERT | +0.0003 [-0.0002, +0.0009] | +0.0009 [+0.0005, +0.0013] |
| Student minus FaBERT (after adding pseudo-labels) | -0.0016 [-0.0025, -0.0008] | -0.0006 [-0.0014, -0.0000] |

The semi-supervised student against the supervised model, recommendation, change of each score:
accuracy +0.0024, Macro F1 -0.0016, Macro precision +0.0043, Macro recall (= balanced accuracy)
-0.0082; recall of `recommended` +0.0074, `not_recommended` -0.0097, `no_idea` -0.0224; F1 of
`recommended` +0.0016, `not_recommended` -0.0016, `no_idea` -0.0048.

The final system answered "uncertain" on 10.0% of the reviews of each head; on the reviews it
answered, accuracy 0.9189 and Macro F1 0.7645 for recommendation (validation 0.9188 and 0.7633),
0.8396 and 0.7093 for rating band (validation 0.8407 and 0.7078).

| | Expected | Found |
|---|---|---|
| Test within 0.5 point of validation | yes | every model, -0.10 to +0.22 point: right |
| Order: final system, FaBERT, TF-IDF | yes | yes on both heads: right (on recommendation the final system leads FaBERT by 0.03 point only) |
| FaBERT 2 to 3.5 points above TF-IDF, interval above zero | yes | +3.25 and +2.34: right |
| Final system 0.05 to 0.3 point above FaBERT | yes | recommendation +0.03 with the interval across zero: wrong; rating band +0.09: right |
| Student 0 to 0.3 point under FaBERT on recommendation | yes | -0.16: right |
| 9.5 to 10.5% "uncertain" | yes | 10.0% and 10.0%: right |

What it means:

- The validation scores hold on the unseen products: no model moved more than a quarter point,
  so nothing was fitted to validation by the many choices made on it.
- The fine-tuned Transformer beats the TF-IDF baseline clearly on both heads (+3.3 and +2.3
  points), on every class.
- Mixing in TF-IDF (D58) helps the rating band a little (+0.09 point, measurable) but adds
  nothing measurable on recommendation on test (+0.03, interval across zero), where validation
  had shown +0.20. The final system stays as chosen; on recommendation it is as good as FaBERT
  alone.
- The semi-supervised model is worse than the supervised one on Macro F1 (-0.16 point on
  recommendation, measurable) while its accuracy is higher (+0.24 point): the pseudo-labels,
  almost all `recommended`, pull the model towards that class and away from `no_idea` (recall
  -2.2 points). The answer to "can unlabeled reviews improve the classifier?" is no, for this
  model and protocol, on test as on validation.
- Answering "uncertain" on 10% of reviews raises accuracy to 91.9% and 84.0%, exactly as on
  validation.
