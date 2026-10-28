# All results

Written by `scripts/summarize_results.py` from the files in `reports/`; do not edit by hand.
Macro F1 on the validation set. A difference is shown with its 95% bootstrap interval
[low, high]: a change counts only when the whole interval is above zero.

## TF-IDF stages 1 and 2 (no offsets)

| Head | Stage | Features | Class weight | C | Converged | Macro F1 | Commit |
|---|---|---|---|---|---|---|---|
| rating_band | features | word123_min5_persian | balanced | 1.0 | yes | 0.6443 | a7c1389 |
| rating_band | regularization | word123_min5_persian | none | 0.25 | yes | 0.6365 | a7c1389 |
| rating_band | regularization | word123_min5_persian | none | 0.5 | yes | 0.6426 | a7c1389 |
| rating_band | features | word123_min5_persian | none | 1.0 | yes | 0.6459 | a7c1389 |
| rating_band | regularization | word123_min5_persian | none | 2.0 | yes | 0.6450 | a7c1389 |
| rating_band | regularization | word123_min5_persian | none | 4.0 | yes | 0.6425 | a7c1389 |
| rating_band | features | word12_min2 | balanced | 1.0 | yes | 0.6422 | a7c1389 |
| rating_band | features | word12_min2 | none | 1.0 | yes | 0.6436 | a7c1389 |
| rating_band | features | word12_min5 | balanced | 1.0 | yes | 0.6400 | a7c1389 |
| rating_band | features | word12_min5 | none | 1.0 | yes | 0.6431 | a7c1389 |
| rating_band | features | word12_min5_persian | balanced | 1.0 | yes | 0.6404 | a7c1389 |
| rating_band | features | word12_min5_persian | none | 1.0 | yes | 0.6431 | a7c1389 |
| recommendation | regularization | word123_min5_persian | balanced | 0.25 | yes | 0.6784 | a7c1389 |
| recommendation | regularization | word123_min5_persian | balanced | 0.5 | yes | 0.6802 | a7c1389 |
| recommendation | features | word123_min5_persian | balanced | 1.0 | yes | 0.6789 | a7c1389 |
| recommendation | regularization | word123_min5_persian | balanced | 2.0 | yes | 0.6758 | a7c1389 |
| recommendation | regularization | word123_min5_persian | balanced | 4.0 | yes | 0.6712 | a7c1389 |
| recommendation | features | word123_min5_persian | none | 1.0 | yes | 0.6783 | a7c1389 |
| recommendation | features | word12_min2 | balanced | 1.0 | yes | 0.6768 | a7c1389 |
| recommendation | features | word12_min2 | none | 1.0 | yes | 0.6756 | a7c1389 |
| recommendation | features | word12_min5 | balanced | 1.0 | yes | 0.6751 | a7c1389 |
| recommendation | features | word12_min5 | none | 1.0 | yes | 0.6782 | a7c1389 |
| recommendation | features | word12_min5_persian | balanced | 1.0 | yes | 0.6735 | a7c1389 |
| recommendation | features | word12_min5_persian | none | 1.0 | yes | 0.6759 | a7c1389 |

## TF-IDF experiments (each against the reference, no offsets unless named)

| Experiment | Head | Variant | Difference to reference [95%] | Verdict | Converged | Commit |
|---|---|---|---|---|---|---|
| E1 | rating_band | offsets | +0.0217 [+0.0209, +0.0226] | better | yes | 7efaee5 |
| E1 | recommendation | offsets | +0.0217 [+0.0207, +0.0227] | better | yes | 92c13ad |
| E2 | rating_band | word_plus_char | +0.0017 [+0.0011, +0.0024] | better | yes | 7efaee5 |
| E2 | recommendation | word_plus_char | -0.0003 [-0.0009, +0.0003] | no measurable difference | yes | 7efaee5 |
| E3 | rating_band | frequent_words_removed | -0.0798 [-0.0812, -0.0785] | worse | yes | 7efaee5 |
| E3 | recommendation | frequent_words_removed | -0.0714 [-0.0727, -0.0701] | worse | yes | 7efaee5 |
| E4 | rating_band | ordinal | -0.0017 [-0.0024, -0.0009] | worse | yes | 7efaee5 |
| E5 | rating_band | plain_tf | -0.0003 [-0.0008, +0.0002] | no measurable difference | yes | 7efaee5 |
| E5 | recommendation | plain_tf | -0.0001 [-0.0004, +0.0002] | no measurable difference | yes | 7efaee5 |
| E6 | rating_band | svc_C0.1 | -0.0111 [-0.0118, -0.0104] | worse | yes | 7efaee5 |
| E6 | recommendation | complement_nb_alpha0.1 | -0.0281 [-0.0291, -0.0269] | worse | yes | 7efaee5 |
| E6 | recommendation | complement_nb_alpha1 | -0.0260 [-0.0271, -0.0248] | worse | yes | 7efaee5 |
| E6 | recommendation | svc_C0.1 | +0.0188 [+0.0180, +0.0197] | better | yes | 92c13ad |
| E6 | recommendation | svc_C1 | +0.0054 [+0.0043, +0.0064] | better | **no** | 7efaee5 |
| E7 | rating_band | zero_as_low | -0.0109 [-0.0117, -0.0101] | worse | yes | 7efaee5 |
| E8 | rating_band | spelling_fixed | -0.0000 [-0.0004, +0.0004] | no measurable difference | yes | 2d9bb48 |
| E8 | recommendation | spelling_fixed | +0.0009 [+0.0006, +0.0012] | better | yes | 2d9bb48 |
| E9 | rating_band | by_product | score 0.6448 [0.6438, 0.6459] | report only | yes | 2d9bb48 |
| E9 | rating_band | random | score 0.6464 [0.6452, 0.6475] | report only | yes | 2d9bb48 |
| E9 | recommendation | by_product | score 0.6808 [0.6798, 0.6819] | report only | yes | 2d9bb48 |
| E9 | recommendation | random | score 0.6803 [0.6791, 0.6814] | report only | yes | 2d9bb48 |
| E10 | rating_band | by_time | score 0.6708 [0.6697, 0.6719] | report only | yes | 2d9bb48 |
| E10 | recommendation | by_time | score 0.6780 [0.6770, 0.6791] | report only | yes | 2d9bb48 |
| E11 | rating_band | majority_only | +0.0014 [+0.0006, +0.0022] | better | yes | 2d9bb48 |
| E11 | recommendation | majority_only | +0.0027 [+0.0019, +0.0034] | better | yes | 2d9bb48 |
| E12 | rating_band | n10000 | score 0.5496 [0.5481, 0.5510] | report only | yes | 2d9bb48 |
| E12 | rating_band | n100000 | score 0.6074 [0.6060, 0.6087] | report only | yes | 2d9bb48 |
| E12 | rating_band | n1000000 | score 0.6355 [0.6342, 0.6368] | report only | yes | 2d9bb48 |
| E12 | recommendation | n10000 | score 0.6150 [0.6136, 0.6164] | report only | yes | 2d9bb48 |
| E12 | recommendation | n100000 | score 0.6487 [0.6473, 0.6499] | report only | yes | 2d9bb48 |
| E12 | recommendation | n1000000 | score 0.6720 [0.6706, 0.6733] | report only | yes | 2d9bb48 |
| reference | rating_band | fit | score 0.6465 [0.6452, 0.6479] | report only | yes | 2d9bb48 |
| reference | recommendation | fit | score 0.6795 [0.6781, 0.6809] | report only | yes | 2d9bb48 |

## TF-IDF combined run (final_tfidf.py)

| Head | Comparison | Variant | Reference | Difference [95%] | Commit |
|---|---|---|---|---|---|
| rating_band | final vs reference | 0.6650 | 0.6465 | +0.0185 [+0.0175, +0.0195] | d4e095c |
| rating_band | final vs reference with offsets | 0.6650 | 0.6683 | -0.0033 [-0.0041, -0.0025] | d4e095c |
| rating_band | final without offsets vs reference | 0.6517 | 0.6465 | +0.0052 [+0.0044, +0.0059] | d4e095c |
| recommendation | final vs reference | 0.7021 | 0.6795 | +0.0226 [+0.0217, +0.0236] | d4e095c |
| recommendation | final vs reference with offsets | 0.7021 | 0.7017 | +0.0005 [-0.0002, +0.0011] | d4e095c |
| recommendation | final without offsets vs reference | 0.6836 | 0.6795 | +0.0041 [+0.0034, +0.0047] | d4e095c |

## TF-IDF model kept: reference with offsets (fit_final_tfidf.py)

| Head | Without offsets | With offsets | Commit |
|---|---|---|---|
| recommendation | 0.6795 | 0.7017 | 4da9c60 |
| rating_band | 0.6465 | 0.6683 | 4da9c60 |

## Transformer pilot (1 epoch)

Scored with offsets on the same 50,000 validation reviews, as is TF-IDF (the kept model).

| Model | Seed | Training reviews | Minutes | Recommendation (vs TF-IDF [95%]) | Rating band (vs TF-IDF [95%]) | Mean |
|---|---|---|---|---|---|---|
| FacebookAI/xlm-roberta-base | 2026 | 100,000 | 1.22 | 0.7037 (+0.0064 [+0.0004, +0.0124]) | 0.6708 (+0.0004 [-0.0045, +0.0052]) | 0.6873 |
| HooshvareLab/bert-fa-zwnj-base | 2026 | 100,000 | 0.86 | 0.6873 (-0.0100 [-0.0159, -0.0038]) | 0.6592 (-0.0112 [-0.0158, -0.0068]) | 0.6733 |
| PartAI/TookaBERT-Base | 2026 | 100,000 | 0.87 | 0.7050 (+0.0077 [+0.0017, +0.0133]) | 0.6702 (-0.0003 [-0.0048, +0.0045]) | 0.6876 |
| PartAI/TookaBERT-Base | 2027 | 100,000 | 0.85 | 0.7047 (+0.0074 [+0.0018, +0.0131]) | 0.6718 (+0.0013 [-0.0033, +0.0061]) | 0.6882 |
| PartAI/TookaBERT-Base | 2028 | 100,000 | 0.86 | 0.7051 (+0.0078 [+0.0022, +0.0136]) | 0.6682 (-0.0023 [-0.0069, +0.0024]) | 0.6866 |
| PartAI/TookaBERT-Large | 2026 | 100,000 | 2.2 | 0.7154 (+0.0181 [+0.0125, +0.0237]) | 0.6772 (+0.0067 [+0.0019, +0.0113]) | 0.6963 |
| PartAI/TookaBERT-Large | 2027 | 100,000 | 2.22 | 0.7057 (+0.0084 [+0.0027, +0.0139]) | 0.6763 (+0.0059 [+0.0012, +0.0107]) | 0.6910 |
| PartAI/TookaBERT-Large | 2028 | 100,000 | 2.19 | 0.7155 (+0.0182 [+0.0128, +0.0240]) | 0.6793 (+0.0089 [+0.0042, +0.0134]) | 0.6974 |
| PartAI/TookaBERT-Large | 2026 | 500,000 | 11.03 | 0.7197 (+0.0224 [+0.0166, +0.0284]) | 0.6877 (+0.0173 [+0.0129, +0.0217]) | 0.7037 |
| jhu-clsp/mmBERT-base | 2026 | 100,000 | 1.78 | 0.6963 (-0.0010 [-0.0072, +0.0047]) | 0.6736 (+0.0032 [-0.0020, +0.0079]) | 0.6850 |
| sbunlp/fabert | 2026 | 100,000 | 0.86 | 0.7003 (+0.0030 [-0.0033, +0.0091]) | 0.6795 (+0.0091 [+0.0045, +0.0135]) | 0.6899 |
| sbunlp/fabert | 2027 | 100,000 | 0.86 | 0.7072 (+0.0098 [+0.0040, +0.0152]) | 0.6783 (+0.0078 [+0.0031, +0.0123]) | 0.6927 |
| sbunlp/fabert | 2028 | 100,000 | 0.85 | 0.7118 (+0.0144 [+0.0084, +0.0203]) | 0.6776 (+0.0072 [+0.0029, +0.0116]) | 0.6947 |
| sbunlp/fabert | 2026 | 500,000 | 4.3 | 0.7241 (+0.0268 [+0.0211, +0.0325]) | 0.6872 (+0.0167 [+0.0125, +0.0214]) | 0.7056 |

Commit 1647e23 (uncommitted)

## FaBERT on all training reviews

Checks on the 50,000-review validation sample, with offsets:

| Epoch | Step | Mean | Recommendation | Rating band |  |
|---|---|---|---|---|---|
| 0.25 | 13,877 | 0.7009 | 0.7157 | 0.6861 | kept |
| 0.5 | 27,754 | 0.7038 | 0.7218 | 0.6858 | kept |
| 0.75 | 41,631 | 0.7059 | 0.7235 | 0.6882 | kept |
| 1.0 | 55,508 | 0.7071 | 0.7252 | 0.6889 | kept |
| 1.25 | 69,385 | 0.7104 | 0.7310 | 0.6898 | kept |
| 1.5 | 83,262 | 0.7085 | 0.7282 | 0.6889 |  |
| 1.75 | 97,139 | 0.7103 | 0.7325 | 0.6881 |  |

Commit 2e54cac. Whole validation set:

| Head | FaBERT without offsets | FaBERT with offsets | TF-IDF with offsets | Difference [95%] |
|---|---|---|---|---|
| recommendation | 0.7224 | 0.7298 | 0.7017 | +0.0281 [+0.0268, +0.0294] |
| rating_band | 0.6797 | 0.6908 | 0.6683 | +0.0225 [+0.0215, +0.0235] |

| Head | FaBERT validation loss | TF-IDF validation loss |
|---|---|---|
| recommendation | 0.3117 | 0.5004 |
| rating_band | 0.4933 | 0.5220 |

FaBERT loss of both heads: training 0.7874 (logged steps before the kept one), validation 0.8050.

## Not measured

| What | Why |
|---|---|
| E6 linear SVM with offsets against logistic regression with offsets | never run: E6 compared the SVM without offsets against the reference without offsets |
| E6 rating band: SVM C=1, Complement Naive Bayes alpha 0.1 and 1 | run on molab, lost when the notebook was relaunched before the download |
| Transformer pilot runs on the whole validation set | by design they were scored on a 50,000-review sample to save time |
| FaBERT validation loss at each check | not logged during training; only the kept model's validation loss is known |
| FaBERT trained past epoch 1.75 | early stopping ended the run after two checks without gain |
| Test set | kept untouched until the end; scored once for each final model |
