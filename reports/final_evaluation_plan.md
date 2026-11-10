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
