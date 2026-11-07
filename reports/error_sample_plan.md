# Reading rating band errors by hand (written before reading)

## Why

The rating boundary, rate recommendation and label conflict analyses (`reports/rating_boundaries.json`,
`reports/rate_recommendation.json`, `reports/label_conflict.json`) found that:

- 88% of the medium mistakes are reviews rated exactly 3, not reviews near the band edges.
- Among reviews rated 3, those FaBERT reads as high were recommended by the buyer 81% of the
  time, against 32% for those it reads as medium.
- Reviews the buyer recommended but rated medium give 51,855 band mistakes, 34.5% of all band
  mistakes on reviews with both labels; FaBERT's recommendation head gets them right.
- A band mistake is far more likely when the recommendation is also mistaken (0.586 against
  0.156).

These are counts. Whether the texts themselves read positive, mixed or neutral has not been
looked at. This sample reads them, and also covers the required qualitative analysis of
errors.

## Design

- From validation only, with seed 2026 (`scripts/make_error_sample.py`); the test set is not
  read. FaBERT's final run, saved band labels with offsets. Reviews with both labels only.
- Five groups, 20 reviews each:

| Group | Buyer's recommendation | True band | FaBERT's band |
|---|---|---|---|
| A | recommended | medium | high |
| B | no_idea | medium | low or high |
| C | recommended | high | medium |
| D | recommended | low | medium or high |
| E (control) | any | any | right |

- Each text at most once and each product at most once; texts also in train are left out
  (repeat texts are H4, measured already). 10 of the 100 reviews are shown again, to measure
  how consistent the reader is.
- The reader sees only the row number and the original text, in shuffled order, and fills
  three columns, judging from the text alone. The key (group, rate, labels, predictions) is
  not opened until the file is finished.

### Codes

- `reading`: 1 positive, 2 negative, 3 mixed (praise and complaint), 4 neutral or no verdict.
- `stars`: the band the text alone suggests: 1 low (below 2.5), 2 medium, 3 high (4 or more),
  0 cannot tell.
- `tags` (optional, any letters together, for example `ab`):
  a complaint about delivery, seller, packaging or price rather than the product;
  b "good but" or mixed judgement; c negation or sarcasm; d too short or no information;
  e judged against expectations or price ("fine for this price"); f other.

## Analysis (after reading, `scripts/score_error_sample.py`, report only)

- Per group: shares of each `reading`, `stars` and tag.
- Per group: `stars` equal to the true band, and equal to FaBERT's band.
- Repeats: same codes on both showings.
- Nothing in the models, labels or splits changes because of this sample. With 20 reviews per
  group, a share is uncertain by about plus or minus 20 points, so only large differences can
  be read.

## Expected

- A: `reading` positive in at least 60%, `stars` high in at least 60% (the text agrees with
  FaBERT, not with the rate).
- B: `reading` mixed or neutral in at least 60%.
- C: `reading` mixed, or tag e or d, in at least 50%.
- D: `stars` not low in at least 50%, with tag a frequent.
- E: `stars` equal to the true band in about 75%.
- Repeats: the same `stars` in about 80%.

## Decision rule (fixed before reading)

- If in A `stars` is high in at least 60%, and in E `stars` equals the true band in at least
  60%, H2' is supported by reading: these mistakes come from the rate disagreeing with the
  text, a ceiling for any model that reads only the text. They are reported as such, not
  treated as something to train away.
- If E is below 60%, the reader cannot recover the band from text alone even when the model
  can, and the A result is not read as evidence either way.

## Result

`reports/error_sample.json`, commit a9cea67. One reader (the project's author) coded the 110
rows without seeing the key. Counts out of 20 per group:

| Group | Reading: positive / negative / mixed / neutral | Stars: low / medium / high / cannot tell | Stars = true band | Stars = FaBERT's band | Most common tags |
|---|---|---|---|---|---|
| A (medium, FaBERT high) | 15 / 0 / 5 / 0 | 1 / 11 / 8 / 0 | 11 | 8 | b 4, e 3 |
| B (`no_idea` and medium, FaBERT low or high) | 2 / 12 / 4 / 2 | 2 / 9 / 7 / 2 | 9 | 3 | c 6, a 4, b 4 |
| C (high, FaBERT medium) | 3 / 2 / 15 / 0 | 6 / 14 / 0 / 0 | 0 | 14 | b 11, e 5 |
| D (`recommended` and low, FaBERT medium or high) | 6 / 5 / 9 / 0 | 3 / 12 / 5 / 0 | 3 | 10 | b 7, e 5, c 4 |
| E (control, FaBERT right) | 11 / 1 / 8 / 0 | 2 / 6 / 12 / 0 | 13 | 13 | b 7 |

The 10 repeated rows got the same codes both times (10 of 10 for reading, stars and tags).

| | Expected | Found |
|---|---|---|
| A reading positive at least 60% | yes | 75%: right |
| A stars high at least 60% | yes | 40%: wrong |
| B reading mixed or neutral at least 60% | yes | 30% (60% read negative): wrong |
| C reading mixed, or tag e or d, at least 50% | yes | 80%: right |
| D stars not low at least 50%, tag a frequent | yes | 85%: right; tag a in 15%: wrong |
| E stars equal to the true band, about 75% | yes | 65%: about right |
| Repeats with the same stars, about 80% | yes | 100% |

The rule: E is above 60%, so the reader can recover the band from the text; but in A the
text reads high in only 40%, under 60%, so H2' (these medium mistakes come from a rate that
disagrees with a high-reading text) is **not supported by reading**. What reading does show:
A texts mostly read positive but at a medium strength; C texts (rated high, read medium by
FaBERT) are mostly "good, but" and read medium to the reader too, 14 of 20 agreeing with
FaBERT and none with the label; D texts (recommended but rated low) mostly do not read low.

Limits: one reader, 20 reviews per group (a share is uncertain by about plus or minus 20
points), only rating band mistakes. A second, independent reader on many more reviews is
`reports/independent_reader_plan.md`; it agreed with the models on the high reviews read as
medium (63% against 19% on right high reviews).
