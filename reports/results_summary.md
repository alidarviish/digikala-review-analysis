# All results

Written by `scripts/summarize_results.py` from the files in `reports/`; do not edit by hand.
Macro F1 on the validation set, except where a section says test. A difference is shown with
its 95% bootstrap interval [low, high]: a change counts only when the whole interval is above zero.

## TF-IDF stages 1 and 2 (no offsets)

| Head | Stage | Features | Class weight | C | Converged | Macro F1 | Made with |
|---|---|---|---|---|---|---|---|
| rating_band | features | word123_min5_persian | balanced | 1.0 | yes | 0.6443 | a7c1389 · Colab |
| rating_band | regularization | word123_min5_persian | none | 0.25 | yes | 0.6365 | a7c1389 · Colab |
| rating_band | regularization | word123_min5_persian | none | 0.5 | yes | 0.6426 | a7c1389 · Colab |
| rating_band | features | word123_min5_persian | none | 1.0 | yes | 0.6459 | a7c1389 · Colab |
| rating_band | regularization | word123_min5_persian | none | 2.0 | yes | 0.6450 | a7c1389 · Colab |
| rating_band | regularization | word123_min5_persian | none | 4.0 | yes | 0.6425 | a7c1389 · Colab |
| rating_band | features | word12_min2 | balanced | 1.0 | yes | 0.6422 | a7c1389 · Colab |
| rating_band | features | word12_min2 | none | 1.0 | yes | 0.6436 | a7c1389 · Colab |
| rating_band | features | word12_min5 | balanced | 1.0 | yes | 0.6400 | a7c1389 · Colab |
| rating_band | features | word12_min5 | none | 1.0 | yes | 0.6431 | a7c1389 · Colab |
| rating_band | features | word12_min5_persian | balanced | 1.0 | yes | 0.6404 | a7c1389 · Colab |
| rating_band | features | word12_min5_persian | none | 1.0 | yes | 0.6431 | a7c1389 · Colab |
| recommendation | regularization | word123_min5_persian | balanced | 0.25 | yes | 0.6784 | a7c1389 · Colab |
| recommendation | regularization | word123_min5_persian | balanced | 0.5 | yes | 0.6802 | a7c1389 · Colab |
| recommendation | features | word123_min5_persian | balanced | 1.0 | yes | 0.6789 | a7c1389 · Colab |
| recommendation | regularization | word123_min5_persian | balanced | 2.0 | yes | 0.6758 | a7c1389 · Colab |
| recommendation | regularization | word123_min5_persian | balanced | 4.0 | yes | 0.6712 | a7c1389 · Colab |
| recommendation | features | word123_min5_persian | none | 1.0 | yes | 0.6783 | a7c1389 · Colab |
| recommendation | features | word12_min2 | balanced | 1.0 | yes | 0.6768 | a7c1389 · Colab |
| recommendation | features | word12_min2 | none | 1.0 | yes | 0.6756 | a7c1389 · Colab |
| recommendation | features | word12_min5 | balanced | 1.0 | yes | 0.6751 | a7c1389 · Colab |
| recommendation | features | word12_min5 | none | 1.0 | yes | 0.6782 | a7c1389 · Colab |
| recommendation | features | word12_min5_persian | balanced | 1.0 | yes | 0.6735 | a7c1389 · Colab |
| recommendation | features | word12_min5_persian | none | 1.0 | yes | 0.6759 | a7c1389 · Colab |

## TF-IDF experiments (each against the reference, no offsets unless named)

| Experiment | Head | Variant | Difference to reference [95%] | Verdict | Converged | Made with |
|---|---|---|---|---|---|---|
| E1 | rating_band | offsets | +0.0217 [+0.0209, +0.0226] | better | yes | 7efaee5 · molab CPU |
| E1 | recommendation | offsets | +0.0217 [+0.0207, +0.0227] | better | yes | 92c13ad · Colab |
| E2 | rating_band | word_plus_char | +0.0017 [+0.0011, +0.0024] | better | yes | 7efaee5 · molab CPU |
| E2 | recommendation | word_plus_char | -0.0003 [-0.0009, +0.0003] | no measurable difference | yes | 7efaee5 · molab CPU |
| E3 | rating_band | frequent_words_removed | -0.0798 [-0.0812, -0.0785] | worse | yes | 7efaee5 · molab CPU |
| E3 | recommendation | frequent_words_removed | -0.0714 [-0.0727, -0.0701] | worse | yes | 7efaee5 · molab CPU |
| E4 | rating_band | ordinal | -0.0017 [-0.0024, -0.0009] | worse | yes | 7efaee5 · molab CPU |
| E5 | rating_band | plain_tf | -0.0003 [-0.0008, +0.0002] | no measurable difference | yes | 7efaee5 · molab CPU |
| E5 | recommendation | plain_tf | -0.0001 [-0.0004, +0.0002] | no measurable difference | yes | 7efaee5 · molab CPU |
| E6 | rating_band | complement_nb_alpha0.1 | -0.0209 [-0.0222, -0.0196] | worse | yes | 927438a · molab CPU |
| E6 | rating_band | complement_nb_alpha1 | -0.0208 [-0.0220, -0.0195] | worse | yes | 927438a · molab CPU |
| E6 | rating_band | svc_C0.1 | -0.0111 [-0.0118, -0.0104] | worse | yes | 7efaee5 · molab CPU |
| E6 | rating_band | svc_C1 | -0.0076 [-0.0086, -0.0068] | worse | **no** | 927438a · molab CPU |
| E6 | recommendation | complement_nb_alpha0.1 | -0.0281 [-0.0291, -0.0269] | worse | yes | 7efaee5 · molab CPU |
| E6 | recommendation | complement_nb_alpha1 | -0.0260 [-0.0271, -0.0248] | worse | yes | 7efaee5 · molab CPU |
| E6 | recommendation | svc_C0.1 | +0.0188 [+0.0180, +0.0197] | better | yes | 92c13ad · Colab |
| E6 | recommendation | svc_C1 | +0.0054 [+0.0043, +0.0064] | better | **no** | 7efaee5 · molab CPU |
| E7 | rating_band | zero_as_low | -0.0109 [-0.0117, -0.0101] | worse | yes | 7efaee5 · molab CPU |
| E8 | rating_band | spelling_fixed | -0.0000 [-0.0004, +0.0004] | no measurable difference | yes | 2d9bb48 · molab CPU |
| E8 | recommendation | spelling_fixed | +0.0009 [+0.0006, +0.0012] | better | yes | 2d9bb48 · molab CPU |
| E9 | rating_band | by_product | score 0.6448 [0.6438, 0.6459] | report only | yes | 2d9bb48 · molab CPU |
| E9 | rating_band | random | score 0.6464 [0.6452, 0.6475] | report only | yes | 2d9bb48 · molab CPU |
| E9 | recommendation | by_product | score 0.6808 [0.6798, 0.6819] | report only | yes | 2d9bb48 · molab CPU |
| E9 | recommendation | random | score 0.6803 [0.6791, 0.6814] | report only | yes | 2d9bb48 · molab CPU |
| E10 | rating_band | by_time | score 0.6708 [0.6697, 0.6719] | report only | yes | 2d9bb48 · molab CPU |
| E10 | recommendation | by_time | score 0.6780 [0.6770, 0.6791] | report only | yes | 2d9bb48 · molab CPU |
| E11 | rating_band | majority_only | +0.0014 [+0.0006, +0.0022] | better | yes | 2d9bb48 · molab CPU |
| E11 | recommendation | majority_only | +0.0027 [+0.0019, +0.0034] | better | yes | 2d9bb48 · molab CPU |
| E12 | rating_band | n10000 | score 0.5496 [0.5481, 0.5510] | report only | yes | 2d9bb48 · molab CPU |
| E12 | rating_band | n100000 | score 0.6074 [0.6060, 0.6087] | report only | yes | 2d9bb48 · molab CPU |
| E12 | rating_band | n1000000 | score 0.6355 [0.6342, 0.6368] | report only | yes | 2d9bb48 · molab CPU |
| E12 | recommendation | n10000 | score 0.6150 [0.6136, 0.6164] | report only | yes | 2d9bb48 · molab CPU |
| E12 | recommendation | n100000 | score 0.6487 [0.6473, 0.6499] | report only | yes | 2d9bb48 · molab CPU |
| E12 | recommendation | n1000000 | score 0.6720 [0.6706, 0.6733] | report only | yes | 2d9bb48 · molab CPU |
| E13 | rating_band | svc_C0.1_with_offsets | +0.0007 [+0.0000, +0.0014] | better | yes | 1be1f21 · molab CPU |
| E13 | recommendation | svc_C0.1_with_offsets | +0.0025 [+0.0019, +0.0032] | better | yes | 1be1f21 · molab CPU |
| reference | rating_band | fit | score 0.6465 [0.6452, 0.6479] | report only | yes | 2d9bb48 · molab CPU |
| reference | recommendation | fit | score 0.6795 [0.6781, 0.6809] | report only | yes | 2d9bb48 · molab CPU |

## TF-IDF combined run (final_tfidf.py)

| Head | Comparison | Variant | Reference | Difference [95%] | Made with |
|---|---|---|---|---|---|
| rating_band | final vs reference | 0.6650 | 0.6465 | +0.0185 [+0.0175, +0.0195] | d4e095c · molab CPU |
| rating_band | final vs reference with offsets | 0.6650 | 0.6683 | -0.0033 [-0.0041, -0.0025] | d4e095c · molab CPU |
| rating_band | final without offsets vs reference | 0.6517 | 0.6465 | +0.0052 [+0.0044, +0.0059] | d4e095c · molab CPU |
| recommendation | final vs reference | 0.7021 | 0.6795 | +0.0226 [+0.0217, +0.0236] | d4e095c · molab CPU |
| recommendation | final vs reference with offsets | 0.7021 | 0.7017 | +0.0005 [-0.0002, +0.0011] | d4e095c · molab CPU |
| recommendation | final without offsets vs reference | 0.6836 | 0.6795 | +0.0041 [+0.0034, +0.0047] | d4e095c · molab CPU |

## TF-IDF model kept: reference with offsets (fit_final_tfidf.py)

| Head | Without offsets | With offsets | Made with |
|---|---|---|---|
| recommendation | 0.6795 | 0.7017 | 4da9c60 · molab CPU |
| rating_band | 0.6465 | 0.6683 | 4da9c60 · molab CPU |

## Transformer pilot (1 epoch)

Scored with offsets on the same 50,000 validation reviews, as is TF-IDF (the kept model).

| Model | Seed | Training reviews | Minutes | Machine | Recommendation (vs TF-IDF [95%]) | Rating band (vs TF-IDF [95%]) | Mean |
|---|---|---|---|---|---|---|---|
| FacebookAI/xlm-roberta-base | 2026 | 100,000 | 1.22 | molab GPU | 0.7037 (+0.0064 [+0.0004, +0.0124]) | 0.6708 (+0.0004 [-0.0045, +0.0052]) | 0.6873 |
| HooshvareLab/bert-fa-zwnj-base | 2026 | 100,000 | 0.86 | molab GPU | 0.6873 (-0.0100 [-0.0159, -0.0038]) | 0.6592 (-0.0112 [-0.0158, -0.0068]) | 0.6733 |
| PartAI/TookaBERT-Base | 2026 | 100,000 | 0.87 | molab GPU | 0.7050 (+0.0077 [+0.0017, +0.0133]) | 0.6702 (-0.0003 [-0.0048, +0.0045]) | 0.6876 |
| PartAI/TookaBERT-Base | 2027 | 100,000 | 0.85 | molab GPU | 0.7047 (+0.0074 [+0.0018, +0.0131]) | 0.6718 (+0.0013 [-0.0033, +0.0061]) | 0.6882 |
| PartAI/TookaBERT-Base | 2028 | 100,000 | 0.86 | molab GPU | 0.7051 (+0.0078 [+0.0022, +0.0136]) | 0.6682 (-0.0023 [-0.0069, +0.0024]) | 0.6866 |
| PartAI/TookaBERT-Large | 2026 | 100,000 | 2.2 | molab GPU | 0.7154 (+0.0181 [+0.0125, +0.0237]) | 0.6772 (+0.0067 [+0.0019, +0.0113]) | 0.6963 |
| PartAI/TookaBERT-Large | 2027 | 100,000 | 2.22 | molab GPU | 0.7057 (+0.0084 [+0.0027, +0.0139]) | 0.6763 (+0.0059 [+0.0012, +0.0107]) | 0.6910 |
| PartAI/TookaBERT-Large | 2028 | 100,000 | 2.19 | molab GPU | 0.7155 (+0.0182 [+0.0128, +0.0240]) | 0.6793 (+0.0089 [+0.0042, +0.0134]) | 0.6974 |
| PartAI/TookaBERT-Large | 2026 | 500,000 | 11.03 | molab GPU | 0.7197 (+0.0224 [+0.0166, +0.0284]) | 0.6877 (+0.0173 [+0.0129, +0.0217]) | 0.7037 |
| jhu-clsp/mmBERT-base | 2026 | 100,000 | 1.78 | molab GPU | 0.6963 (-0.0010 [-0.0072, +0.0047]) | 0.6736 (+0.0032 [-0.0020, +0.0079]) | 0.6850 |
| sbunlp/fabert | 2026 | 100,000 | 0.86 | molab GPU | 0.7003 (+0.0030 [-0.0033, +0.0091]) | 0.6795 (+0.0091 [+0.0045, +0.0135]) | 0.6899 |
| sbunlp/fabert | 2027 | 100,000 | 0.86 | molab GPU | 0.7072 (+0.0098 [+0.0040, +0.0152]) | 0.6783 (+0.0078 [+0.0031, +0.0123]) | 0.6927 |
| sbunlp/fabert | 2028 | 100,000 | 0.85 | molab GPU | 0.7118 (+0.0144 [+0.0084, +0.0203]) | 0.6776 (+0.0072 [+0.0029, +0.0116]) | 0.6947 |
| sbunlp/fabert | 2026 | 500,000 | 4.3 | molab GPU | 0.7241 (+0.0268 [+0.0211, +0.0325]) | 0.6872 (+0.0167 [+0.0125, +0.0214]) | 0.7056 |

Summary made with 58a68b1 · local

## Transformer pilot on the whole validation set (1 epoch, mean over seeds, with offsets)

| Model | Training reviews | Seeds | Recommendation | Rating band | Mean | Seeds range | Minutes |
|---|---|---|---|---|---|---|---|
| PartAI/TookaBERT-Large | 100,000 | 3 | 0.7119 | 0.6792 | 0.6956 | 0.6951 to 0.6960 | 2.2 |
| sbunlp/fabert | 100,000 | 3 | 0.7102 | 0.6757 | 0.6929 | 0.6925 to 0.6936 | 0.9 |
| jhu-clsp/mmBERT-base | 100,000 | 1 | 0.7053 | 0.6719 | 0.6886 | 0.6886 to 0.6886 | 1.8 |
| PartAI/TookaBERT-Base | 100,000 | 3 | 0.7013 | 0.6699 | 0.6856 | 0.6841 to 0.6868 | 0.9 |
| FacebookAI/xlm-roberta-base | 100,000 | 1 | 0.7004 | 0.6693 | 0.6848 | 0.6848 to 0.6848 | 1.2 |
| HooshvareLab/bert-fa-zwnj-base | 100,000 | 1 | 0.6881 | 0.6579 | 0.6730 | 0.6730 to 0.6730 | 0.9 |
| PartAI/TookaBERT-Large | 500,000 | 3 | 0.7243 | 0.6872 | 0.7058 | 0.7050 to 0.7062 | 11.0 |
| sbunlp/fabert | 500,000 | 3 | 0.7231 | 0.6861 | 0.7046 | 0.7043 to 0.7049 | 4.3 |

Summary made with 085b8f7 · local

TookaBERT-Large minus FaBERT, both on 500,000 reviews, same reviews per seed:

| Seed | Recommendation [95%] | Rating band [95%] |
|---|---|---|
| 2026 | +0.0001 [-0.0008, +0.0009] | +0.0013 [+0.0007, +0.0020] |
| 2027 | +0.0017 [+0.0008, +0.0026] | +0.0013 [+0.0006, +0.0020] |
| 2028 | +0.0019 [+0.0011, +0.0028] | +0.0008 [+0.0002, +0.0015] |

Lead of TookaBERT-Large +0.00117, spread between seeds 0.00124: tie, chosen sbunlp__fabert (a lead smaller than the spread between seeds is a tie; a tie goes to the faster model). Made with 42ec92d.

## FaBERT on all training reviews (the kept run: every epoch, best check kept)

Checks on the 50,000-review validation sample, with offsets:

| Epoch | Step | Mean | Recommendation | Rating band | Validation loss |  |
|---|---|---|---|---|---|---|
| 0.25 | 13,877 | 0.7007 | 0.7169 | 0.6844 | 0.8321 | kept |
| 0.5 | 27,754 | 0.7047 | 0.7230 | 0.6864 | 0.8201 | kept |
| 0.75 | 41,631 | 0.7090 | 0.7274 | 0.6905 | 0.8104 | kept |
| 1.0 | 55,508 | 0.7081 | 0.7265 | 0.6896 | 0.8131 |  |
| 1.25 | 69,385 | 0.7078 | 0.7252 | 0.6905 | 0.8083 |  |
| 1.5 | 83,262 | 0.7091 | 0.7282 | 0.6899 | 0.8097 | kept |
| 1.75 | 97,139 | 0.7098 | 0.7287 | 0.6908 | 0.8034 | kept |
| 2.0 | 111,016 | 0.7125 | 0.7322 | 0.6928 | 0.8048 | kept |
| 2.25 | 124,893 | 0.7101 | 0.7313 | 0.6889 | 0.8131 |  |
| 2.5 | 138,770 | 0.7116 | 0.7306 | 0.6927 | 0.8127 |  |
| 2.75 | 152,647 | 0.7131 | 0.7325 | 0.6937 | 0.8075 | kept |
| 3.0 | 166,524 | 0.7116 | 0.7345 | 0.6887 | 0.8104 |  |
| 3.25 | 180,401 | 0.7072 | 0.7262 | 0.6882 | 0.8394 |  |
| 3.5 | 194,278 | 0.7103 | 0.7326 | 0.6879 | 0.8365 |  |
| 3.75 | 208,155 | 0.7093 | 0.7307 | 0.6878 | 0.8363 |  |
| 4.0 | 222,032 | 0.7087 | 0.7297 | 0.6877 | 0.8372 |  |

Trained with 287977c · molab GPU. Whole validation set (compared with 59a3ce9 · local):

| Head | FaBERT without offsets | FaBERT with offsets | TF-IDF with offsets | Difference [95%] |
|---|---|---|---|---|
| recommendation | 0.7253 | 0.7324 | 0.7017 | +0.0308 [+0.0295, +0.0321] |
| rating_band | 0.6799 | 0.6922 | 0.6683 | +0.0239 [+0.0229, +0.0249] |

| Head | FaBERT validation loss | TF-IDF validation loss |
|---|---|---|
| recommendation | 0.3121 | 0.5004 |
| rating_band | 0.4925 | 0.5220 |

FaBERT loss of both heads: training 0.7505 (logged steps before the kept one), validation 0.8046.

Kept run (best check at epoch 2.75) against the first run, stopped early (best check at epoch 1.25), both with offsets on the whole validation set (made with 82ab4df):

| Head | Kept run | First run | Difference [95%] | Verdict |
|---|---|---|---|---|
| recommendation | 0.7324 | 0.7298 | +0.0027 [+0.0017, +0.0036] | better |
| rating_band | 0.6922 | 0.6908 | +0.0014 [+0.0006, +0.0022] | better |

## Final models per class (whole validation set, with offsets)

Balanced accuracy is the mean recall over the classes, so the large classes do not
weigh more; plain accuracy does, and is shown only next to it.

| Head | Class | Reviews | FaBERT precision / recall / F1 | TF-IDF precision / recall / F1 |
|---|---|---|---|---|
| recommendation | recommended | 624,006 | 0.948 / 0.940 / 0.944 | 0.943 / 0.913 / 0.928 |
| recommendation | not_recommended | 75,666 | 0.773 / 0.716 / 0.744 | 0.730 / 0.685 / 0.706 |
| recommendation | no_idea | 89,045 | 0.483 / 0.540 / 0.510 | 0.420 / 0.535 / 0.471 |
| rating_band | low | 84,407 | 0.750 / 0.676 / 0.711 | 0.714 / 0.647 / 0.679 |
| rating_band | medium | 158,755 | 0.486 / 0.472 / 0.479 | 0.451 / 0.451 / 0.451 |
| rating_band | high | 600,296 | 0.878 / 0.897 / 0.887 | 0.869 / 0.881 / 0.875 |

| Head | FaBERT Macro F1 / balanced accuracy / accuracy | TF-IDF Macro F1 / balanced accuracy / accuracy |
|---|---|---|
| recommendation | 0.7324 / 0.7320 / 0.8736 | 0.7017 / 0.7110 / 0.8487 |
| rating_band | 0.6922 / 0.6813 / 0.7945 | 0.6683 / 0.6596 / 0.7765 |

## Hand-labeled validation sample (reports/label_sample_plan.md)

200 reviews labeled again from the text alone. Made with bd13b11.

| Original label | Reviews | Hand label agrees | FaBERT agrees with the hand label |
|---|---|---|---|
| recommended | 100 | 95% | 97% |
| not_recommended | 50 | 94% | 80% |
| no_idea | 50 | 34% | 56% |

## Training changes on the FaBERT pilot (500,000 reviews, 1 epoch, 3 seeds, with offsets)

Mean Macro F1 of the two heads over the seeds. A lead smaller than the spread between
seeds is a tie.

| Experiment | Variant | Score | Usual training | Lead | Spread | Verdict | Made with |
|---|---|---|---|---|---|---|---|
| Texts merged (reports/merged_texts_plan.md) | copies | 0.7043 | 0.7046 | -0.0002 | 0.0006 | tie | 73437c8 |
| Texts merged (reports/merged_texts_plan.md) | once | 0.7045 | 0.7046 | -0.0001 | 0.0006 | tie | 73437c8 |
| Hard reviews weighted up (reports/hard_examples_plan.md) | focal 0.5 | 0.7047 | 0.7051 | -0.0004 | 0.0006 | tie | 9622331 |
| Hard reviews weighted up (reports/hard_examples_plan.md) | focal 1 | 0.7046 | 0.7051 | -0.0005 | 0.0006 | tie | 9622331 |
| Hard reviews weighted up (reports/hard_examples_plan.md) | focal 2 | 0.7036 | 0.7051 | -0.0015 | 0.0012 | worse | 9622331 |
| Hard reviews weighted up (reports/hard_examples_plan.md) | class weights | 0.7021 | 0.7051 | -0.0029 | 0.0018 | worse | 9622331 |
| Difficulty used (reports/difficulty_plan.md) | leave out 5% | 0.7037 | 0.7051 | -0.0013 | 0.0006 | worse | f090d6c |
| Difficulty used (reports/difficulty_plan.md) | leave out 10% | 0.7026 | 0.7051 | -0.0025 | 0.0010 | worse | f090d6c |
| Difficulty used (reports/difficulty_plan.md) | curriculum | 0.1455 | 0.7051 | -0.5596 | 0.0192 | worse | f090d6c |
| Gradual curriculum (reports/difficulty_plan.md) | gradual curriculum | 0.7029 | 0.7051 | -0.0021 | 0.0009 | worse | bd13b11 |
| Learning rate (reports/learning_rate_plan.md) | learning rate 2e-5 | 0.7041 | 0.7051 | -0.0010 | 0.0008 | worse | 3f49121 |
| Learning rate (reports/learning_rate_plan.md) | learning rate 5e-5 | 0.7043 | 0.7051 | -0.0008 | 0.0009 | tie | 3f49121 |
| Learning rate (reports/learning_rate_plan.md) | layer decay 0.9 | 0.7032 | 0.7051 | -0.0019 | 0.0010 | worse | 3f49121 |
| Pseudo-labels (reports/semi_supervised_plan.md) | pseudo-labels 0.9 | 0.7047 | 0.7051 | -0.0003 | 0.0007 | tie | 2279ab7 |
| Domain-adapted FaBERT (reports/domain_adaptation_plan.md) | domain-adapted | 0.7072 | 0.7051 | +0.0021 | 0.0006 | better | be670fa |

## Calibration of the final FaBERT model (reports/calibration_plan.md)

Temperature scaling, fitted on one half of validation by product and scored on the other. Made with a067b69.

| Head | Temperature | ECE | Mean loss | Reviews with p >= 0.9 | Macro F1 with offsets |
|---|---|---|---|---|---|
| recommendation | 1.124 | 0.0162 -> 0.0055 | 0.3121 -> 0.3095 | 72.3%, 96.9% right -> 69.1%, 97.3% right | 0.7324 -> 0.7327 |
| rating_band | 1.108 | 0.0240 -> 0.0046 | 0.4925 -> 0.4898 | 45.6%, 92.9% right -> 37.2%, 93.9% right | 0.6922 -> 0.6923 |

## Every required score of the final models (whole validation set, with offsets)

Made with 2204338. Macro precision and recall are plain means over the classes; the ordinal error is how many bands a prediction is off on average, and the share off by two.

recommendation (788,717 reviews):

| Model | Accuracy | Macro F1 | Macro precision | Macro recall | Balanced accuracy |
|---|---|---|---|---|---|
| tfidf | 0.8487 | 0.7017 | 0.6976 | 0.7110 | 0.7110 |
| fabert | 0.8736 | 0.7324 | 0.7346 | 0.7320 | 0.7320 |
| adapted | 0.8731 | 0.7337 | 0.7337 | 0.7356 | 0.7356 |
| large | 0.8759 | 0.7324 | 0.7397 | 0.7269 | 0.7269 |
| bilstm | 0.8694 | 0.7176 | 0.7302 | 0.7083 | 0.7083 |
| fabert + tfidf | 0.8733 | 0.7344 | 0.7327 | 0.7376 | 0.7376 |

rating_band (843,458 reviews):

| Model | Accuracy | Macro F1 | Macro precision | Macro recall | Balanced accuracy | Ordinal error | Off by two |
|---|---|---|---|---|---|---|---|
| tfidf | 0.7765 | 0.6683 | 0.6781 | 0.6596 | 0.6596 | 0.2405 | 1.70% |
| fabert | 0.7945 | 0.6922 | 0.7046 | 0.6813 | 0.6813 | 0.2176 | 1.21% |
| adapted | 0.7958 | 0.6931 | 0.7050 | 0.6826 | 0.6826 | 0.2165 | 1.23% |
| large | 0.7942 | 0.6938 | 0.7022 | 0.6861 | 0.6861 | 0.2182 | 1.24% |
| bilstm | 0.7903 | 0.6826 | 0.6970 | 0.6701 | 0.6701 | 0.2233 | 1.36% |
| fabert + tfidf | 0.7957 | 0.6935 | 0.7046 | 0.6836 | 0.6836 | 0.2167 | 1.24% |

## The main decisions, with product-level intervals (reports/decision_recheck_plan.md)

Made with d54d70f.

| Decision | Comparison | recommendation | rating_band |
|---|---|---|---|
| D58 final system | fabert + tfidf against fabert | +0.0020 [+0.0014, +0.0026] | +0.0013 [+0.0009, +0.0018] |
| D65 best single model | adapted against fabert | +0.0012 [+0.0004, +0.0021] | +0.0009 [+0.0003, +0.0016] |
| D65 final system unchanged | adapted + tfidf against fabert + tfidf | +0.0001 [-0.0008, +0.0008] | +0.0008 [+0.0002, +0.0015] |
| model order | fabert against tfidf | +0.0308 [+0.0295, +0.0320] | +0.0239 [+0.0229, +0.0250] |
| model order | bilstm against tfidf | +0.0159 [+0.0146, +0.0172] | +0.0143 [+0.0132, +0.0153] |
| model order | fabert against bilstm | +0.0149 [+0.0137, +0.0161] | +0.0096 [+0.0087, +0.0105] |

## The full runs with three training seeds (reports/full_seeds_plan.md)

Made with c5b243f. The official model is the seed 2026 run.

| Model | Head | Seed 2026 | Seed 2027 | Seed 2028 | Spread |
|---|---|---|---|---|---|
| FaBERT | recommendation | 0.7324 | 0.7317 | 0.7312 | 0.0012 |
| BiLSTM, every epoch | recommendation | 0.7195 | 0.7192 | 0.7182 | 0.0013 |
| FaBERT | rating_band | 0.6922 | 0.6929 | 0.6927 | 0.0007 |
| BiLSTM, every epoch | rating_band | 0.6838 | 0.6842 | 0.6811 | 0.0031 |

D58 (FaBERT + TF-IDF minus FaBERT) with each FaBERT seed:

| Head | Seed 2026 | Seed 2027 | Seed 2028 |
|---|---|---|---|
| recommendation | +0.0020 [+0.0014, +0.0026] | +0.0022 [+0.0015, +0.0029] | +0.0025 [+0.0018, +0.0032] |
| rating_band | +0.0013 [+0.0009, +0.0018] | +0.0004 [+0.0001, +0.0007] | +0.0014 [+0.0010, +0.0018] |

## Three models that read differently (reports/reading_models_plan.md)

FaBERT, TF-IDF and the BiLSTM on 872,795 validation reviews. Made with 25b8e6c.

| Head | Reviews | All right | All wrong | All wrong, same label | All wrong, no known cause |
|---|---|---|---|---|---|
| recommendation | 745,860 | 81.0% | 8.9% | 90.0% | 17.1% |
| rating_band | 797,508 | 72.4% | 15.6% | 94.3% | 8.9% |

## The accuracy ceiling of any text-only model on repeated texts (reports/text_ceiling_plan.md)

Texts seen at least 5 times. Made with 3e8e982.

| Head | Reviews | Ceiling | FaBERT | BiLSTM | TF-IDF |
|---|---|---|---|---|---|
| recommendation | 184,278 (24.7%) | 0.9296 | 0.9262 | 0.9261 | 0.9229 |
| rating_band | 211,200 (26.5%) | 0.8482 | 0.8363 | 0.8363 | 0.8331 |

## A reader that never saw the labels (reports/independent_reader_plan.md)

Qwen/Qwen3-8B, zero-shot. Made with 59a3ce9.

| Head | All right: agrees with the label | All wrong, same label: with the label | with the models | Verdict |
|---|---|---|---|---|
| recommendation | 70.8% | 32.4% | 47.2% | inconclusive: the reader sides with the models more often, but not twice as often |
| rating_band | 71.2% | 28.0% | 59.0% | shared-bias objection rejected: the reader sides with the models at least twice as often |

## Ids removed by cleaning rule P7 (report only)

245 ids, 490 rows; every pair shares a product and a creation time. Made with aa52c1e.

| What differs | Ids |
|---|---|
| label | 101 |
| text edit | 111 |
| rate 0 | 4 |
| two rates | 21 |
| spacing | 8 |

## The test set: the four official models (reports/final_evaluation_plan.md)

Made with f2fd9cf. Test Macro F1 with offsets tuned on all of validation, its 95% interval drawing products, and the validation Macro F1 beside it.

recommendation (788,717 reviews):

| Model | Accuracy | Macro F1 [95%] | Balanced accuracy | Validation Macro F1 |
|---|---|---|---|---|
| tfidf | 0.8490 | 0.7021 [0.7005, 0.7036] | 0.7111 | 0.7017 |
| fabert | 0.8732 | 0.7346 [0.7331, 0.7361] | 0.7374 | 0.7324 |
| final system | 0.8734 | 0.7349 [0.7334, 0.7364] | 0.7382 | 0.7344 |
| student | 0.8756 | 0.7330 [0.7314, 0.7345] | 0.7292 | 0.7310 |

The final system answers "uncertain" on 10.0% of them; on the others accuracy 0.9189.

rating_band (843,458 reviews):

| Model | Accuracy | Macro F1 [95%] | Balanced accuracy | Validation Macro F1 |
|---|---|---|---|---|
| tfidf | 0.7765 | 0.6684 [0.6668, 0.6700] | 0.6617 | 0.6683 |
| fabert | 0.7941 | 0.6918 [0.6903, 0.6934] | 0.6810 | 0.6922 |
| final system | 0.7929 | 0.6927 [0.6911, 0.6942] | 0.6827 | 0.6935 |
| student | 0.7916 | 0.6911 [0.6896, 0.6927] | 0.6807 | 0.6921 |

The final system answers "uncertain" on 10.0% of them; on the others accuracy 0.8396.

## Semi-supervised learning, the project's protocol (reports/semi_supervised_official_plan.md)

Made with 2b57540. Threshold 0.9: 533,043 of 736,134 hidden reviews accepted, 96.9% of them right; the rule (more than 0.0012 over the control, interval above zero): pseudo-labels help = False. Checks kept (epochs): supervised 2.75, control 0.0, student 0.0; the student and the control kept their start, the teacher, because no later check was better, so the comparison uses both at the end of their epoch.

| Model (validation) | recommendation | rating_band |
|---|---|---|
| supervised | 0.7324 | 0.6922 |
| control | 0.7324 | 0.6922 |
| student | 0.7324 | 0.6922 |
| control at the end | 0.7300 | 0.6903 |
| student at the end | 0.7310 | 0.6921 |

| Recommendation difference | Macro F1 [95%] |
|---|---|
| student minus control | +0.0000 [+0.0000, +0.0000] |
| student minus supervised | +0.0000 [+0.0000, +0.0000] |
| control minus supervised | +0.0000 [+0.0000, +0.0000] |
| student at the end minus control at the end | +0.0010 [+0.0002, +0.0017] |
| student at the end minus supervised | -0.0015 [-0.0023, -0.0007] |
| control at the end minus supervised | -0.0024 [-0.0033, -0.0016] |

## The extensions (reports/extensions_plan.md)

Validation made with 1c01e0e; the test set read a second and last time with b28f2c8 (D74). The exact rate is scored on the reviews with a valid rate (D75).

Exact rate:

| Model | Val MAE | Val RMSE | Val R2 | Val Spearman | Test MAE | Test RMSE | Test R2 | Test Spearman |
|---|---|---|---|---|---|---|---|---|
| FaBERT rate head | 0.546 | 0.740 | 0.584 | 0.674 | 0.547 | 0.741 | 0.582 | 0.672 |
| TF-IDF ridge | 0.609 | 0.798 | 0.516 | 0.621 | 0.609 | 0.799 | 0.515 | 0.621 |
| constant (training mean) | 0.877 | 1.146 | 0.000 | – | 0.877 | 1.147 | 0.000 | – |

recommendation, Macro F1 with offsets:

| Model | Validation | Test |
|---|---|---|
| fabert | 0.7324 | 0.7346 |
| student 0.90 | 0.7310 | 0.7330 |
| rate model | 0.7327 | 0.7338 |
| frozen | 0.6592 | 0.6605 |
| student 0.80 | 0.7308 | 0.7322 |
| student 0.95 | 0.7310 | 0.7323 |
| student both heads | 0.7299 | 0.7321 |
| student natural | 0.7306 | 0.7325 |
| tfidf | 0.7017 | 0.7021 |

| Test difference | Macro F1 [95%] |
|---|---|
| rate model minus fabert | -0.0008 [-0.0017, +0.0001] |
| frozen minus fabert | -0.0741 [-0.0756, -0.0726] |
| frozen minus tfidf | -0.0416 [-0.0430, -0.0401] |
| student 0.80 minus student 0.90 | -0.0008 [-0.0016, -0.0001] |
| student 0.95 minus student 0.90 | -0.0007 [-0.0015, +0.0000] |
| student both heads minus student 0.90 | -0.0009 [-0.0017, -0.0001] |
| student natural minus student 0.90 | -0.0005 [-0.0012, +0.0003] |
| student 0.80 minus fabert | -0.0024 [-0.0032, -0.0017] |
| student 0.95 minus fabert | -0.0023 [-0.0031, -0.0015] |
| student both heads minus fabert | -0.0025 [-0.0034, -0.0017] |
| student natural minus fabert | -0.0020 [-0.0029, -0.0012] |

rating_band, Macro F1 with offsets:

| Model | Validation | Test |
|---|---|---|
| fabert | 0.6922 | 0.6918 |
| student 0.90 | 0.6921 | 0.6911 |
| rate model | 0.6926 | 0.6920 |
| frozen | 0.6174 | 0.6167 |
| student 0.80 | 0.6923 | 0.6910 |
| student 0.95 | 0.6915 | 0.6912 |
| student both heads | 0.6919 | 0.6908 |
| student natural | 0.6922 | 0.6912 |
| tfidf | 0.6683 | 0.6684 |

| Test difference | Macro F1 [95%] |
|---|---|
| rate model minus fabert | +0.0002 [-0.0005, +0.0009] |
| frozen minus fabert | -0.0751 [-0.0765, -0.0737] |
| frozen minus tfidf | -0.0517 [-0.0530, -0.0503] |
| student 0.80 minus student 0.90 | -0.0001 [-0.0007, +0.0005] |
| student 0.95 minus student 0.90 | +0.0000 [-0.0006, +0.0006] |
| student both heads minus student 0.90 | -0.0003 [-0.0009, +0.0002] |
| student natural minus student 0.90 | +0.0001 [-0.0005, +0.0007] |
| student 0.80 minus fabert | -0.0007 [-0.0014, -0.0001] |
| student 0.95 minus fabert | -0.0006 [-0.0013, -0.0000] |
| student both heads minus fabert | -0.0010 [-0.0016, -0.0003] |
| student natural minus fabert | -0.0006 [-0.0013, +0.0001] |

Pseudo-labels help by the rule (on validation): student 0.80 no, student 0.95 no, student both heads no, student natural no. Band guesses at 0.9: 45.2% kept, 92.4% right. Naturally unlabeled reviews at 0.9: 77.1%.

## The service (CPU)

| What | Measured | Made with |
|---|---|---|
| Predictor against the saved predictions: passed; predicted rate, mean difference | True; 0.0012 | 5a0c528 |
| Models in memory | 1.585 GB | 5a0c528 |
| Process after loading, peak while loading | 2.175 GB, 2.707 GB | 5a0c528 |
| One review (median, 95th percentile) | 103 ms, 202 ms | 5a0c528 |
| Docker image | 2.447 GB | 5a0c528 |
| Container ready after (models loaded in) | 43.3 s (38.9 s) | 5a0c528 |
| Health check; /health and /ready from inside; GPU inside | healthy; {'/health': 200, '/ready': 200}; False | 5a0c528 |
| Requests answered as expected; review text in the log | True; False | 5a0c528 |

## Not measured

| What | Why |
|---|---|
| Class-balanced pseudo-labels | not part of the project's protocol or of the extensions plan (reports/extensions_plan.md) |
| The quality of the naturally unlabeled reviews' pseudo-labels | those reviews have no label to check them against |
| The service on a GPU | it is built and checked for the CPU only (reports/docker_check.json) |
| The test set for any model outside the two fixed lists | read once for the four official models and once for the extensions (D73, D74) |
