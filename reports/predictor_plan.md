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

## Result

`reports/predictor_check.json` (commit 233e53a), `reports/fabert_files.json` and
`reports/predict_demo.md` (commit 8a38c72; the code did not change between the two). The hub
was still at the commit the model was trained with (4cad2850ab81); three files were saved
(`config.json`, `tokenizer.json`, `tokenizer_config.json`).

Check on 2,000 validation reviews, from the raw body:

| | Recommendation | Rating band | Rule |
|---|---|---|---|
| Normalized text equal to the cache | 100% | 100% | every review: passed |
| TF-IDF: same label; largest probability difference | 100%; 0 | 100%; 0 | passed |
| FaBERT: same label without offsets | 100% | 100% | at least 99.5%: passed |
| FaBERT: probability difference, mean (largest) | 0.0002 (0.0047) | 0.0003 (0.0037) | under 0.005: passed |
| Final system: same best guess | 100% | 100% | at least 99.5%: passed |
| Final system: same "uncertain" | 99.90% (2 reviews) | 100% | reported |

Tokens: median 9 (the median review has 6 words); 0.25% of the reviews are longer than 128
tokens. Latency of one review on this machine's CPU, after the first: normalization 0.1 ms,
TF-IDF 1.6 ms (95%: 4.0), FaBERT 44 ms (95%: 87). The first review after loading is slower
(about 250 ms for FaBERT). FaBERT on the 2,000 reviews in batches: 168 s.

Demo (`reports/predict_demo.md`), recommendation and rating band of the final system:

| Review | Recommendation | Rating band |
|---|---|---|
| short, "عالی" | recommended | high |
| short, "بد بود" | not_recommended | low |
| ambiguous | no_idea | medium |
| negative | not_recommended | low |
| mixed, quality against price | uncertain [no_idea] | uncertain [medium] |
| mixed, sound against battery | recommended | uncertain [medium] |
| negation, "بد نیست" | no_idea | medium |
| negation, "اصلا ارزش خرید نداره" | not_recommended | low |
| sarcasm | recommended | high |
| spelling variant | recommended | high |
| long, cut at 128 tokens | uncertain [recommended] | medium |

| | Expected | Found |
|---|---|---|
| Normalization and TF-IDF equal | yes | yes: right |
| FaBERT, same label | 99.7 to 99.95% | 100%: wrong, better |
| FaBERT, mean probability difference | 0.001 to 0.003 | 0.0002 and 0.0003: wrong, smaller |
| TF-IDF latency | under 5 ms | 1.6 ms (95%: 4.0): right |
| FaBERT latency | 20 to 80 ms | 44 ms median: right; 95%: 87 ms, above |
| Longer than 128 tokens | under 5% | 0.25%: right |
| "بد نیست" missed by at least one model | yes | wrong: both read it as lukewarm (`no_idea`, `medium`), not as negative |
| The sarcastic review missed by at least one model | yes | right: both answer `recommended` |
| A mixed review gets `uncertain`, `no_idea` or `medium` | yes | right: both mixed reviews, on at least one head |

What it means:

- The offline path gives the same answers as the models did during training: TF-IDF exactly,
  FaBERT on every label of the sample, with the small differences expected from the GPU's
  bfloat16; so the scores measured on validation hold for this script.
- One review takes about 50 ms on a laptop CPU, almost all of it FaBERT; TF-IDF alone is 30
  times faster.
- The demo shows the known limits: sarcasm is read from its positive words; a mixed opinion
  falls to `no_idea`, `medium` or `uncertain`; FaBERT's answer on the long review rests on its
  first 128 tokens only. A plain negation ("بد نیست") is read as lukewarm, as many reviewers mean it.
