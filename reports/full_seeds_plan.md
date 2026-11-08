# How much the full runs depend on chance (written before running)

## Why

Every design decision for FaBERT was made on the pilot (500,000 training reviews, all of
validation) with three seeds, and the seeds moved Macro F1 by only 0.06 to 0.09 points
(`reports/seed_oracle.json`); the further pretrained FaBERT (D65) beat every pilot seed on both
heads. But the full runs on all 3,552,498 training reviews were each trained once: the kept
FaBERT run, which gives the official numbers, and the BiLSTM, whose two more seeds
(`reports/bilstm_seeds.json`) differed by up to a point because one stopped early at epoch 1.5.
The kept FaBERT run trains every epoch and keeps the best check, so early stopping cannot do
this to it; how much its own numbers move from one run to another has not been measured.

## Runs

- FaBERT, the kept run (`scripts/train_transformer.py --no-early-stop`), again with training
  seeds 2027 and 2028 (`--seed`, steps 88 and 89): the same settings, training reviews and
  50,000-review choosing sample; only the initial weights of the heads, dropout and the order
  of the reviews change. About 2 hours each on the molab GPU.
- The BiLSTM with the best setting of its tuning, trained for all 4 epochs with the best check
  kept, as the kept FaBERT run (`scripts/train_bilstm.py --no-early-stop`, seeds 2026, 2027 and
  2028, steps 90 to 92). About 22 minutes each.
- Every run gets its own names, so the existing runs stay as they are.

## What is reported

`scripts/check_full_seeds.py` (step 93, `reports/full_seeds.json`), on the whole validation set:

- Macro F1 with offsets of each run, and the spread (highest minus lowest) of each model;
- FaBERT minus BiLSTM and BiLSTM minus TF-IDF with each seed;
- D58 with each FaBERT seed: FaBERT + TF-IDF minus FaBERT, paired product-level bootstrap;
- where the full further pretrained run stands against the three FaBERT seeds;
- the three-model comparison (all right, all wrong, the same wrong label, no known cause) with
  each pair of seeds, on the same 872,795 reviews.

Every run's saved labels must score its recorded Macro F1 first.

## Rules (fixed now)

- The official model stays the kept FaBERT run (seed 2026), whatever the other seeds score.
  Picking the best of three seeds would choose on chance and make the reported number look
  better than a new run would be.
- The official numbers are reported with the three-seed spread next to them.
- D58 stands with a seed if the whole interval is above zero; if it fails with any seed, the
  report says so, and the final system is discussed again with the user before anything changes.
- D65 is not decided again here (it was decided on the pilot with three seeds); where the full
  further pretrained run stands against the three seeds is reported only.

## Expected

- FaBERT: the three seeds within 0.2 points of each other on each head (the pilot's 0.06 to
  0.09, plus the choice of the best check).
- BiLSTM without early stopping: the three seeds within 0.3 points, each kept at epoch 2.5 or
  later, and none more than 0.1 point below the early-stopped seed 2026 run (0.7176 and 0.6826).
- FaBERT ahead of the BiLSTM by at least 0.8 point with each pair of seeds, and the BiLSTM ahead
  of TF-IDF.
- D58 stands with every FaBERT seed.
- The counts of the three-model comparison change by under 3% from one pair of seeds to another.

The test set is not read.

## Result

`reports/full_seeds.json` (commit c5b243f); the five runs from commit 33c7688 on the molab
GPU, all five at the same time (so each took longer: about 298 minutes per FaBERT run and 99
per BiLSTM run, against 121 and about 20 alone). Every run's fingerprint differs from the run it
repeats only in the seed (and, for the BiLSTM, the patience); the data hashes, GPU and torch are
the same. Every run scored its recorded Macro F1, and FaBERT + TF-IDF with seed 2026 scored the
recorded D58 value.

Macro F1 with offsets on the whole validation set:

| Model | Seed 2026 | 2027 | 2028 | Spread |
|---|---|---|---|---|
| FaBERT, recommendation | 0.7324 | 0.7317 | 0.7312 | 0.12 point |
| FaBERT, rating band | 0.6922 | 0.6929 | 0.6927 | 0.07 point |
| BiLSTM every epoch, recommendation | 0.7195 | 0.7192 | 0.7182 | 0.13 point |
| BiLSTM every epoch, rating band | 0.6838 | 0.6842 | 0.6811 | 0.31 point |
| BiLSTM with early stopping (for reference), recommendation | 0.7176 | 0.7081 | 0.7182 | 1.01 points |
| BiLSTM with early stopping (for reference), rating band | 0.6826 | 0.6765 | 0.6811 | 0.61 point |

Kept checks: FaBERT at epochs 2.75, 2.0 and 2.0; the BiLSTM at 3.75, 3.75 and 2.5 (seed 2028
kept the same check as with early stopping, so the same predictions).

| | Recommendation | Rating band |
|---|---|---|
| FaBERT minus BiLSTM, seeds 2026 / 2027 / 2028 | +0.0129 / +0.0126 / +0.0130 | +0.0084 / +0.0087 / +0.0117 |
| BiLSTM minus TF-IDF | +0.0179 / +0.0175 / +0.0166 | +0.0155 / +0.0159 / +0.0128 |
| D58, FaBERT + TF-IDF minus FaBERT, seed 2026 | +0.0020 [+0.0014, +0.0026] | +0.0013 [+0.0009, +0.0018] |
| D58, seed 2027 | +0.0022 [+0.0015, +0.0029] | +0.0004 [+0.0001, +0.0007] |
| D58, seed 2028 | +0.0025 [+0.0018, +0.0032] | +0.0014 [+0.0010, +0.0018] |
| Further pretrained FaBERT against the three seeds | 0.7337, above every seed (0.7312 to 0.7324) | 0.6931, above every seed (0.6922 to 0.6929) |

The three-model comparison with each pair of seeds (FaBERT and BiLSTM of the same seed, with
TF-IDF), on the same 872,795 reviews:

| Seeds | Rec: all right / all wrong / same wrong label | Band: all right / all wrong / same wrong label |
|---|---|---|
| 2026 | 604,466 / 66,392 / 90.1% | 577,256 / 124,834 / 94.4% |
| 2027 | 606,007 / 67,534 / 90.3% | 579,883 / 126,413 / 94.5% |
| 2028 | 605,012 / 67,364 / 90.3% | 577,907 / 125,703 / 94.4% |

Relative spread of the counts: all right 0.25% and 0.45%, all wrong 1.70% and 1.26%, all wrong
with the same label 1.93% and 1.41%, all wrong with no known cause 3.64% and 1.63%.

| | Expected | Found |
|---|---|---|
| FaBERT seeds within 0.2 point | yes | 0.12 and 0.07: right |
| BiLSTM every epoch within 0.3 point | yes | 0.13: right; rating band 0.31: wrong, just over |
| BiLSTM kept at epoch 2.5 or later | yes | 3.75, 3.75, 2.5: right |
| No BiLSTM seed more than 0.1 point below 0.7176 and 0.6826 | yes | recommendation right; rating band seed 2028 0.15 below: wrong |
| FaBERT at least 0.8 point ahead of the BiLSTM with each pair | yes | 1.26 to 1.30 and 0.84 to 1.17: right |
| BiLSTM ahead of TF-IDF | yes | right |
| D58 stands with every FaBERT seed | yes | right, every interval above zero (rating band with seed 2027 only +0.0004) |
| Counts change by under 3% | yes | right for all right, all wrong and the same wrong label; "no known cause" on recommendation 3.64% |

What it means:

- The official FaBERT numbers move by about 0.1 point from one training run to another, far
  less than every difference a decision rests on, except the smallest: the final system's gain
  on rating band is +0.0004 to +0.0014 depending on the seed, measurable with each seed but
  small. D58 stands. The official model stays the kept run (seed 2026), as fixed before running;
  it happens to be the highest of the three on recommendation and the lowest on rating band.
- The BiLSTM's one-point drop with seed 2027 came from early stopping: trained every epoch, its
  three seeds are within 0.13 point on recommendation and 0.31 on rating band.
- The shared mistakes of the three models, and their same wrong label (90% and 94%), do not
  depend on any one training run of FaBERT or of the BiLSTM.
