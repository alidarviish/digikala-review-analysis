# Where the rating band and recommendation mistakes come from: four analyses

Written after the four analyses ran. Each was planned in the project's working notes before it
ran, with its expected results and, where there was one, its decision rule; those are quoted
here as they were written, and wrong expectations are marked. All four read only saved
validation predictions: nothing is trained, the rate is used only to group reviews and never
as an input, and the test set is not read. Intervals draw products with replacement (1,000
rounds, seed 2026), because reviews of one product are not independent: 843,458 validation
reviews with a band come from 48,755 products.

## 1. Are band mistakes made near the band edges? (H2)

`scripts/analyze_rating_boundaries.py`, `reports/rating_boundaries.json` (commit 8a0d9d5).
The band comes from the buyer's rate (low below 2.5, medium from 2.5 to below 4.0, high from
4.0). If the text says little about whether the rate was 3.9 or 4.1, the model should be wrong
more often near 2.5 and 4.0. No numeric expectation was written before this run.

- Fractional rates (4.0% of reviews, 9,690 products) are much harder: FaBERT's Macro F1 0.579
  [0.574, 0.584] against 0.697 [0.696, 0.699] on whole rates.
- Among fractional medium rates, the share right falls steadily from 0.506 [0.483, 0.530] in
  [2.50, 2.75) to 0.221 [0.206, 0.235] in [3.75, 4.00), while the share called high rises from
  24% to 76%; fractional low rates fall from 0.669 to 0.338 towards 2.5.
- Share right by whole rate: 1 0.795, 2 0.505, 3 0.486 (41.8% called high, 9.6% low), 4 0.808,
  5 0.953. The model's effective edges sit lower than the defined ones: about 3.0 for
  medium/high and about 2.0 for low/medium.
- Of the 83,886 medium mistakes, 73,932 (88.1%) are reviews rated exactly 3 and 9,954 (11.9%)
  fractional ones.
- FaBERT + TF-IDF against FaBERT: +0.0013 [+0.0009, +0.0018] overall, about +0.009 on rates 1
  and 2, -0.0036 [-0.0045, -0.0027] on rate 3.

H2 holds in its own form (mistakes grow towards an edge), but closeness to an edge explains at
most 12% of the medium mistakes; most come from reviews rated exactly 3, 42% of which the model
reads as high. That led to H2'.

## 2. Do the buyers who rate 3 themselves recommend the product? (H2')

`scripts/analyze_rate_recommendation.py`, `reports/rate_recommendation.json` (commit
2a42db2). The buyer chooses the recommendation label apart from the rate, so it is evidence
that does not come from any model. 713,182 reviews have both a band and a buyer's
recommendation.

Rule written before running: H2' is supported if, among reviews rated 3, the share the buyer
recommended is at least 0.15 higher where FaBERT predicts high than where it predicts medium,
with the interval above zero.

| | Expected | Found |
|---|---|---|
| Share the buyer recommended, rate 5 | above 0.9 | 0.989: right |
| Rate 4 | 0.75 to 0.85 | 0.930: wrong, higher |
| Rate 3 | 0.40 to 0.60 | 0.501 (`no_idea` 0.428, `not_recommended` 0.071): right |
| Rate 1 | under 0.15 | 0.042: right |
| Rate 3, predicted high minus predicted medium | 0.15 to 0.30 | +0.492 [+0.487, +0.497]: wrong, far larger |
| FaBERT's recommendation head on rate 3 | 0.5 to 0.6 recommended | 0.543: right |

Among reviews rated 3, those FaBERT reads as high were recommended by the buyer 81.1%
[80.7, 81.4] of the time (53,662 reviews), those it reads as medium 31.8% (62,324 reviews,
`no_idea` 61.4%). For fractional rates from 2.5 to 3 the difference is +0.562, from 3 to 4
+0.374. Verdict by the rule: H2' supported. At least about 43,500 of the 83,886 medium
mistakes (53,662 x 0.811) are reviews the buyer recommended but rated 3: a conflict inside the
buyer's own two labels, which no model that reads only the text can resolve. This is a
correlation; whether those texts read positive was tested by hand (`reports/error_sample_plan.md`)
and by an independent reader (`reports/independent_reader_plan.md`).

## 3. The two labels against each other

`scripts/analyze_label_conflict.py`, `reports/label_conflict.json` (commit c176f64).

Rule written before running: if the share of medium mistakes that fall on `no_idea` reviews is
at least twice the base share of `no_idea`, and the difference below has an interval above
zero, H1 (`no_idea`) and H2' share a cause and the hand reading samples both cells; otherwise
they are looked at separately.

Band recall by the buyer's recommendation: `recommended` (560,167 reviews) low 0.151, medium
0.297, high 0.913; `not_recommended` (67,056) low 0.802, medium 0.466, high 0.409; `no_idea`
(85,959, 68% medium) low 0.392, medium 0.694, high 0.417; no recommendation label (130,276)
Macro F1 0.674, close to all reviews.

On the 713,182 reviews with both labels:

- `recommended` and medium (73,748 reviews): band wrong 70.3%, both heads wrong only 1.4%;
  the recommendation head says recommended and the band head says high for 50,814 of them.
  This cell holds 51,855 band mistakes, 34.5% of all 150,276.
- `no_idea` and medium (58,164): band wrong 30.6%, recommendation wrong 41.5%, both 28.7%.
- A band mistake is far more likely when the recommendation is also wrong: 0.586 against
  0.156, a difference of +0.430 [+0.426, +0.434].

| | Expected | Found |
|---|---|---|
| Medium recall among `recommended` | 0.20 to 0.35 | 0.297: right |
| Medium recall among `no_idea` | 0.55 to 0.70 | 0.694: right |
| Base shares: medium about 0.20, `no_idea` about 0.12 | | 0.199 and 0.121: right |
| `no_idea` mistakes on medium reviews | 0.40 to 0.55 | 0.611: wrong, higher |
| Medium mistakes on `no_idea` reviews | 0.20 to 0.35 | 0.237: right |
| Band wrong given recommendation wrong, minus given right | +0.10 to +0.20 | +0.430: wrong, far larger |

Verdict by the rule: 0.237 [0.234, 0.241] is just under twice the base share (0.242), so the
first condition fails and the two were looked at separately; the rule was not changed after
seeing this. Seen the other way round, `no_idea` mistakes fall on medium reviews 3.1 times
as often as the base share, and part of that link may come from the encoder both heads share.
The largest single source of band mistakes is reviews the buyer recommended but rated medium,
which the model calls high while its own recommendation head agrees with the buyer: the model
sides with the buyer's recommendation, not with the buyer's rate.

## 4. Does the tokenizer turn many reviews into unknown tokens?

`scripts/check_unknown_tokens.py` with `scripts/bert_wordpiece.py`, `reports/unknown_tokens.json`
(commit 9391af2). A suggestion from outside the project, taken as a hypothesis: FaBERT's
tokenizer marks many reviews as unknown because of spacing or similar, so the text should have
been changed before training. The tokenizer was rebuilt in plain Python (the local machine has
no tokenizer library) and reproduced all 10 numbers and 17 probes recorded in
`reports/tokenizers/compare.json` exactly.

Rule written before running: if fixing every unknown token could gain under 0.3 accuracy
points on each head, changing the text and training again is not worth it and the hypothesis
is rejected for this model; otherwise a pilot is planned.

| | Expected | Found |
|---|---|---|
| Share of tokens unknown | 0.05 to 0.1% | 0.0689% of 10,473,077: right |
| Reviews with at least one unknown token | 0.5 to 3% | 6,851 (0.74%): right |
| Mostly emoji | yes | yes: emoji alone 2,391, emoji joined to Persian letters 4,567 (as "بود😍"); Latin letters and digits almost none: right |
| Reviews made only of unknown tokens | a few hundred to a few thousand | 162: wrong, fewer |
| Most that a fix could gain | under 0.1 point | 0.00: right |

Half-spaces are removed by the tokenizer's own normalizer, which joins the word parts (as if
written without the half-space) and never makes an unknown token; 2.68% of reviews have one.
The model does better, not worse, on reviews with an unknown token: share right 0.898 against
0.873 (recommendation) and 0.881 against 0.794 (rating band). Verdict by the rule: rejected
for FaBERT. This does not show that emoji carry no information (reviews with emoji are short
and plain); only that where they appear the rest of the text is enough, and they are rare.
