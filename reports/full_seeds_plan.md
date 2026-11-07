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
