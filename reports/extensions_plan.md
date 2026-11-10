# The optional extensions of the project (written before any of them is trained)

The project lists several optional extensions; all of them are done here. This plan fixes, before anything is trained,
what each one is, how it is scored, what is expected, and which models are scored on the test set. The official
system and its results (`reports/final_evaluation.json`) do not change, whatever these show.

## The test set, a second and last time

The test set was read once for the four official models (D73). The models of extensions 1 to 5 below are scored on it
**once more, all together, by one script, after all of them are trained**, and reported apart from the official
result, as extensions. The list is fixed now:

| Model | Extension |
|---|---|
| FaBERT with a third head for the exact rate | 1 |
| TF-IDF ridge regression for the exact rate (its baseline) | 1 |
| FaBERT with the encoder frozen (feature extraction) | 2 |
| Semi-supervised student with threshold 0.80, end of epoch | 3 |
| Semi-supervised student with threshold 0.95, end of epoch | 3 |
| Semi-supervised student with pseudo-labels for both heads, end of epoch | 4 |
| Semi-supervised student with the naturally unlabeled reviews added, end of epoch | 5 |

Everything about them (settings, offsets, the checkpoint kept) is fixed on validation before that reading. Nothing
read on the test set changes any of them.

## 1. Exact-rate regression

*"Add a regression head beside the recommendation and rating_band classification heads: total_loss =
recommendation_loss + λ1 × rating_band_loss + λ2 × rate_regression_loss. Mask each loss when its target is
unavailable. Report MAE, RMSE, MSE, R², Spearman correlation, predicted-versus-actual plot, and residual analysis."*

- FaBERT exactly as the kept run (same settings, seed 2026, 4 epochs, best of 16 checks on the same 50,000-review
  sample chosen by the mean Macro F1 of the two classification heads), plus a linear head on the first token that
  predicts the rate. λ1 = 1 (as in the kept run). The rate target is standardized with the mean and standard deviation
  of the training rates, and λ2 = 1, so the three losses start at a similar size; neither λ is tuned.
- A rate of 0 means "no rating recorded" and is masked, like a missing label. Predictions are clipped to [0, 5] (and
  scored on the reviews with a real rate, 1 to 5 and fractions).
- Baseline: ridge regression on the TF-IDF features of the final TF-IDF model (alpha 1, fitted on train). A constant
  (the training mean) is also reported: its R² is about 0.
- Reported on validation (and on test in the second reading): MAE, RMSE, MSE, R², Spearman; the predicted-versus-actual
  plot; residuals by true rate, by recommendation class and by review length; the classification heads' Macro F1
  against the kept run (does the third head change them?).
- Expected: FaBERT R² 0.45 to 0.60, MAE 0.55 to 0.75, Spearman 0.55 to 0.65; TF-IDF ridge 0.05 to 0.10 lower in R²;
  residuals largest for reviews rated 1 to 3 (predictions pulled towards the mean, since 71% of the rates are high);
  the classification heads within 0.2 point of the kept run.

## 2. Frozen feature extraction

*"The frozen feature-extraction comparison is an optional Phase 2 extension."*

- The same network, but the pretrained FaBERT encoder (hub commit 4cad2850ab81) is frozen: only the two linear heads
  are trained, on the first token's output. Same data, checks and offsets as the kept run, 4 epochs, best check kept.
  The heads start from zero and nothing else moves, so they get a learning rate of 1e-3 (the kept run's 3e-5 is for
  fine-tuning the whole encoder); the rate is fixed now, not tuned.
- Compared with the fine-tuned FaBERT and with TF-IDF on validation (and test).
- Expected: far below fine-tuning and below TF-IDF: Macro F1 0.60 to 0.66 for recommendation and 0.56 to 0.62 for
  the rating band. The pretrained first token is not trained to summarize a review's sentiment.

## 3. Threshold sweep

*"A sweep over confidence thresholds such as 0.80, 0.90, and 0.95 ... [is an] optional Phase 2 extension."*

- Two more students, exactly as the official one (one more epoch from the teacher, end of epoch kept), with the
  teacher's guesses accepted at 0.80 (587,492 reviews, 95.5% right) and at 0.95 (445,123, 98.0% right). The guesses
  are those already made and checked (`reports/semi_supervised/pseudo_labels.parquet`).
- Compared with the supervised model, the control and the 0.90 student, with every required score.
- Expected: no threshold gives a measurable gain over the control by the official rule (more than 0.12 point, interval
  above zero). 0.80 adds more `not_recommended` and `no_idea` guesses (6.9% and 0.5%) but more wrong ones: within 0.2
  point of the 0.90 student. 0.95 is still more one-sided: also within 0.2 point.

## 4. Pseudo-labels for the rating band

*"Pseudo-labeling for rating_band ... [is an] optional Phase 2 extension."*

- The teacher also guesses the rating band of the hidden reviews (their band is hidden with them), accepted at 0.90
  of its own band softmax. A student is trained as the official one, with both kinds of guesses (each review counts for
  each head whose guess is accepted). Compared with the official student (recommendation guesses only).
- The quality of the band guesses is checked afterwards against the reference file
  (`data/splits/audit/masked_labels_reference.csv`, which has both labels).
- Expected: band guesses accepted for 40 to 50% of the hidden reviews (on validation 45.6% of reviews reach 0.9), at
  least 90% right and over 90% of them `high`; band Macro F1 within 0.2 point of the official student.

## 5. The naturally unlabeled reviews

*"The main comparison must use this common artificial pool; naturally unlabeled rows are an optional extension."*

- `natural_unlabeled.csv`: 625,700 training reviews whose buyers gave no recommendation. The teacher guesses them;
  those at 0.90 are added to the official student's data (hidden pool plus natural pool). Their quality cannot be
  checked (they have no reference label).
- Expected: a smaller share accepted than for the hidden pool (these reviews are older and differ, notebook 01,
  section 6.5), mostly `recommended`; no measurable gain over the official student.

## 6. Memory use (no test needed)

The peak memory of the offline predictor (process RSS) after loading and while answering, next to its latency.
Expected: 1.0 to 1.6 GB, mostly FaBERT's weights in float32 and the TF-IDF vocabulary.

## 7. FastAPI service and 8. Docker (no test needed)

Built on the same artifacts and `src/digikala_reviews/predictor.py`, as the project describes: the endpoints, the
failure codes (422, 400, 503, 500), at least five tests, no logging of review text, a documented maximum length; then
a Dockerfile, checked inside the running container, with image size, loading time and the CPU assumption reported.
Each gets its own short plan before it is built.

## Order and cost

Extensions 1 to 5 need the GPU (molab); they run side by side where memory allows: about 2 hours for the regression
run, 1 hour for the frozen run, about 1 hour for each student, and a few minutes for the new guesses. Extensions 6 to
8 run on this machine meanwhile. Then the second test reading, then the report sections.
