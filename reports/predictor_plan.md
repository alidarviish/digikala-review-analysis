# Offline prediction (written before running)

## Why

The project asks for the saved artifacts to support a simple offline prediction script: one
review body goes in; it is normalized and tokenized, scored by the TF-IDF and the fine-tuned
Transformer classifiers for both recommendation and rating band, with probabilities and an
explanation, and shown on short, ambiguous, negative and mixed reviews.

## Method

- `src/digikala_reviews/predictor.py` loads everything once (as a service would at start-up)
  and answers for a text:
  - only the text goes in; it is normalized with the training normalizer;
  - TF-IDF: `artifacts/tfidf_final/model.joblib`, with its offsets for the test set;
  - FaBERT: `artifacts/transformer_no_early_stop/model.pt` on the CPU in float32, cut at 128
    tokens as in training, with its offsets for the test set;
  - the final system: the weighted mean of both models' probabilities and its offsets,
    answering `uncertain` when the mixed probability of its answer is under the threshold
    (`reports/uncertain.json`, "for_test"); its best guess is given too;
  - out: label and probabilities of each model and head, the normalized text, the tokens, the
    token count and whether the text was cut, the latency of each model, and the version
    (code commit, normalizer version, hashes of the model files);
  - explanation: whether the two models agree, and the words of the review that pushed the
    TF-IDF model most towards its answer (the word's weight for that class minus its mean
    weight over the classes, times its TF-IDF value).
- `scripts/predict.py`: the command line (a readable table, `--json`, `--demo`).
- `scripts/save_fabert_files.py`: saves FaBERT's tokenizer and configuration once into
  `artifacts/fabert_files/`, from the hub commit the model was trained with (4cad285; it stops
  if the hub has moved), so prediction downloads nothing. The files are not put in git: the
  model's license is not declared. `build_model` gains an option to build the encoder from the
  configuration only; training is unchanged.
- `scripts/check_predictor.py` (step 105): 2,000 fixed validation reviews (seed 2026), from the
  raw body, through the whole path, against what was saved during training.
- `--demo`: eleven reviews written for this purpose, none from the data: short, ambiguous,
  negative, mixed, negation, sarcasm, a spelling variant and one long review.

The test set is not read.

## Rules (fixed now)

| What | Passes when |
|---|---|
| Normalized text | equal to the text in the cache on every review |
| TF-IDF probabilities | largest difference under 1e-6, the same labels |
| FaBERT | the same label without offsets on at least 99.5% of the reviews; mean absolute difference of the probabilities under 0.005 |
| The final system | the same best guess on at least 99.5% |

FaBERT cannot be exactly equal: the saved predictions were made on the GPU with bfloat16,
these on the CPU with float32.

## Expected

- Normalization and TF-IDF: equal.
- FaBERT: the same label on 99.7 to 99.95% of the reviews; mean probability difference 0.001
  to 0.003.
- Latency of one review on the CPU: TF-IDF under 5 ms, FaBERT 20 to 80 ms.
- Under 5% of the sampled reviews are longer than 128 tokens.
- In the demo, "بد نیست" and the sarcastic review are missed by at least one of the two models;
  a mixed review gets `uncertain`, or `no_idea` or `medium`.
