# Offline prediction: examples

Made by `scripts/predict.py --demo` on 2026-11-10, code 8a38c72. The reviews were written for this demonstration; none is from the data. Labels with each model's offsets for the test set; the final system answers `uncertain` when its confidence is under the threshold fitted on validation (`reports/uncertain.json`), and its best guess is shown in brackets.

## short: عالی

3 tokens; latency: TF-IDF 33.2 ms, FaBERT 252.7 ms.

| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |
|---|---|---|---|---|
| recommendation | recommended (0.94) | recommended (0.99) | recommended (0.98) | عالی |
| rating_band | high (0.97) | high (0.97) | high (0.97) | عالی |

## short: بد بود

4 tokens; latency: TF-IDF 1.5 ms, FaBERT 50.6 ms.

| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |
|---|---|---|---|---|
| recommendation | not_recommended (0.88) | not_recommended (0.82) | not_recommended (0.84) | بد بود, بد, بود |
| rating_band | low (0.90) | low (0.87) | low (0.87) | بد, بد بود, بود |

## ambiguous: معمولیه، فعلا دارم استفاده می‌کنم ببینم چطوره

11 tokens; latency: TF-IDF 1.3 ms, FaBERT 62.6 ms.

| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |
|---|---|---|---|---|
| recommendation | no_idea (0.86) | no_idea (0.62) | no_idea (0.69) | فعلا, معمولیه ،, معمولیه, چطوره, فعلا دارم |
| rating_band | medium (0.62) | medium (0.78) | medium (0.74) | معمولیه ،, معمولیه, فعلا, چطوره, دارم استفاده می‌کنم |

## negative: بعد از دو هفته خراب شد و پشتیبانی هم جواب نداد. پولم هدر رفت.

18 tokens; latency: TF-IDF 1.4 ms, FaBERT 63.9 ms.

| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |
|---|---|---|---|---|
| recommendation | not_recommended (0.93) | not_recommended (0.94) | not_recommended (0.94) | خراب, پولم, جواب نداد, هدر, رفت |
| rating_band | low (0.79) | low (0.89) | low (0.87) | پولم, خراب, رفت, جواب نداد, هدر |

## mixed: کیفیتش خوبه ولی برای این قیمت خیلی گرونه

11 tokens; latency: TF-IDF 3.6 ms, FaBERT 52.2 ms.

| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |
|---|---|---|---|---|
| recommendation | no_idea (0.67) | no_idea (0.44) | uncertain [no_idea] (0.51) | ولی, گرونه, این قیمت, قیمت خیلی گرونه, برای این |
| rating_band | high (0.73) | medium (0.48) | uncertain [medium] (0.43) | خوبه ولی, کیفیتش خوبه, برای, این قیمت خیلی, برای این قیمت |

## mixed: صداش عالیه ولی باتریش زود تموم میشه و شارژرش هم داغ میکنه

16 tokens; latency: TF-IDF 1.5 ms, FaBERT 57.0 ms.

| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |
|---|---|---|---|---|
| recommendation | recommended (0.48) | recommended (0.72) | recommended (0.65) | عالیه, عالیه ولی, هم, تموم, و |
| rating_band | medium (0.31) | medium (0.29) | uncertain [medium] (0.29) | ولی باتریش, ولی, تموم میشه و, زود, تموم میشه |

## negation: بد نیست

4 tokens; latency: TF-IDF 1.1 ms, FaBERT 48.3 ms.

| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |
|---|---|---|---|---|
| recommendation | no_idea (0.86) | no_idea (0.50) | no_idea (0.60) | نیست, بد, بد نیست |
| rating_band | medium (0.62) | medium (0.58) | medium (0.59) | نیست, بد نیست, بد |

## negation: اصلا ارزش خرید نداره

6 tokens; latency: TF-IDF 1.2 ms, FaBERT 48.4 ms.

| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |
|---|---|---|---|---|
| recommendation | not_recommended (0.99) | not_recommended (0.96) | not_recommended (0.97) | اصلا, خرید نداره, ارزش خرید نداره, نداره, اصلا ارزش |
| rating_band | low (0.97) | low (0.93) | low (0.94) | اصلا, ارزش خرید نداره, خرید نداره, اصلا ارزش, نداره |

## sarcasm: واقعا عالی بود، روز دوم شکست!

10 tokens; latency: TF-IDF 1.3 ms, FaBERT 47.2 ms.

| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |
|---|---|---|---|---|
| recommendation | recommended (0.42) | recommended (0.81) | recommended (0.70) | عالی بود ،, واقعا عالی, عالی بود, واقعا, عالی |
| rating_band | medium (0.32) | high (0.80) | high (0.74) | واقعا عالی, ، روز, شکست !, روز دوم, ! |

## spelling: خیلی خوبع دستتون درد نکنه

8 tokens; latency: TF-IDF 1.4 ms, FaBERT 47.0 ms.

| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |
|---|---|---|---|---|
| recommendation | recommended (0.95) | recommended (0.99) | recommended (0.98) | خیلی خوبع, درد نکنه, نکنه, خوبع, دستتون درد |
| rating_band | high (0.99) | high (0.96) | high (0.97) | خیلی خوبع, درد نکنه, خوبع, نکنه, دستتون درد |

## long: این گوشی رو حدود سه ماه پیش خریدم و تا الان تجربهٔ نسبتاً خوبی داشتم. صفحه‌نمایشش روشن و رنگ‌هاش طبیعیه و زیر نور آفتاب هم به‌خوبی دیده میشه. دوربینش در روز عکس‌های خوبی می‌گیره ولی در شب نویز زیادی داره و جزئیات از بین میره. باتریش با استفادهٔ معمولی یک روز کامل دوام میاره، اما وقتی بازی می‌کنم زود داغ میشه و شارژش سریع کم میشه. بلندگوش صدای بلندی داره ولی کیفیتش متوسطه. نرم‌افزارش گاهی کند میشه و چند تا برنامهٔ اضافه داره که نمیشه پاکشون کرد. بسته‌بندی سالم بود و ارسال هم به‌موقع انجام شد. در کل برای این قیمت انتخاب بدی نیست، ولی اگه بودجهٔ بیشتری دارید شاید مدل بالاتر ارزشش رو داشته باشه.

146 tokens (cut at 128); latency: TF-IDF 2.3 ms, FaBERT 249.4 ms.

| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |
|---|---|---|---|---|
| recommendation | no_idea (0.75) | recommended (0.69) | uncertain [recommended] (0.55) | ولی, متوسطه, اما, گاهی, شاید |
| rating_band | medium (0.41) | medium (0.65) | medium (0.61) | ولی, متوسطه, نسبتاً, کیفیتش متوسطه ., و چند تا |
