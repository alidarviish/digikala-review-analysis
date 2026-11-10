# Error cases (validation, final system)

Made by `scripts/make_error_cases.py`.

## 1. negation

> بد نبود خوب بود

rate 3.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | no_idea | no_idea (0.76) | recommended (0.72) | recommended (0.57) | نبود, بد نبود خوب, بد, نبود خوب بود |
| rating_band | medium | medium (0.40) | medium (0.40) | medium (0.40) | بد نبود خوب, نبود, خوب بود, بد نبود |

## 2. negation

> هم پلاستیکی بود وهم وسایل جانبی از جمله شیر اتصال به شلنگ همراش نبود

rate 0.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | not_recommended (0.53) | not_recommended (0.66) | not_recommended (0.62) | نبود, پلاستیکی, اتصال به, اتصال |
| rating_band | none | | | | |

## 3. negation

> کاربردش راحت نیست

rate 0.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | not_recommended | no_idea (0.60) | no_idea (0.56) | no_idea (0.57) | راحت نیست, نیست, کاربردش |
| rating_band | none | | | | |

## 4. sarcasm

> واقعن عالی بود

rate 2.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | not_recommended | recommended (0.88) | recommended (1.00) | recommended (0.96) | عالی بود, واقعن, عالی, واقعن عالی |
| rating_band | low | high (0.95) | high (0.97) | high (0.96) | واقعن عالی, واقعن, عالی بود, عالی |

## 5. sarcasm

> عالی

rate 5.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | not_recommended | recommended (0.94) | recommended (0.99) | recommended (0.98) | عالی |
| rating_band | high | high (0.97) | high (0.97) | high (0.97) | عالی |

## 6. sarcasm

> عالی بود، یکیشو هدیه دادم به بابای عشقم، اون یکیش تا از اون یکیش جدا افتاد کمرش شکست و انداختمش دور، پس نتیجه میگیریم حتما از هر دو با هم استفاده کنید 😘

rate 3.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | not_recommended | recommended (0.28) | recommended (1.00) | recommended (0.78) | عالی بود ،, حتما, عالی بود, هدیه |
| rating_band | medium | medium (0.27) | high (0.99) | high (0.86) | شکست و, هدیه دادم, با هم, هدیه |

## 7. short

> عالی بود

rate 3.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | no_idea | recommended (0.96) | recommended (0.99) | recommended (0.98) | عالی بود, عالی |
| rating_band | medium | high (0.93) | high (0.93) | high (0.93) | عالی بود, عالی |

## 8. short

> کیفیت صفحاتش پایینه

rate 0.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | not_recommended | no_idea (0.43) | no_idea (0.55) | no_idea (0.51) | پایینه, صفحاتش |
| rating_band | none | | | | |

## 9. short

> متاسفانه ماندگاری نداره

rate 0.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | not_recommended | no_idea (0.59) | no_idea (0.48) | no_idea (0.51) | متاسفانه, نداره, ماندگاری نداره, ماندگاری |
| rating_band | none | | | | |

## 10. mixed

> بد نبود کیفیتش خوبه اما رنگشو زیاد دوست نداشتم

rate 4.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | no_idea (0.92) | no_idea (0.47) | no_idea (0.60) | اما, نبود, زیاد, دوست نداشتم |
| rating_band | high | medium (0.68) | medium (0.48) | medium (0.52) | بد نبود کیفیتش, اما, زیاد, دوست نداشتم |

## 11. mixed

> جنسش به نظرم خیلی معمولیه من توی شگفت انگیز با قیمت 230 خریدم در همون اندازه قیمت بد نیست ولی نه بیشتر

rate 4.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | no_idea (0.86) | no_idea (0.67) | no_idea (0.73) | معمولیه, معمولیه من, بد نیست ولی, ولی |
| rating_band | high | medium (0.86) | medium (0.82) | medium (0.83) | معمولیه من, معمولیه, بد نیست ولی, خیلی معمولیه |

## 12. mixed

> جنسش کاملا معمولیه بعد از یک ماه استفاده پوسته پوسته شد ولی با این قیمت بد نیست.

rate 3.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | not_recommended | no_idea (0.82) | no_idea (0.73) | no_idea (0.76) | کاملا معمولیه, معمولیه, بد نیست ., ولی |
| rating_band | medium | medium (0.82) | medium (0.84) | medium (0.83) | کاملا معمولیه, معمولیه, پوسته پوسته شد, پوسته شد |

## 13. spelling

> کوچک وضعیف است وبی کیفیت

rate 3.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | no_idea | not_recommended (0.95) | not_recommended (0.92) | not_recommended (0.93) | وبی, وضعیف, وبی کیفیت, کوچک |
| rating_band | medium | low (0.75) | low (0.88) | low (0.84) | وبی کیفیت, وبی, وضعیف, کوچک |

## 14. spelling

> بوش خوبه ملایمه ولی نرم کنندگی نداره تاثییر نداره انگار

rate 0.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | not_recommended | no_idea (0.48) | no_idea (0.71) | no_idea (0.64) | نداره, ولی, انگار, ملایمه |
| rating_band | none | | | | |

## 15. spelling

> چرا هروز قیمت افزایش پیدا میکنه

rate 2.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | no_idea | recommended (0.26) | recommended (0.88) | recommended (0.69) | پیدا, هروز, افزایش, میکنه |
| rating_band | low | high (0.60) | high (0.73) | high (0.70) | پیدا, هروز, افزایش, افزایش پیدا میکنه |

## 16. duplicate

> خوب

rate 1.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | no_idea | recommended (0.66) | recommended (0.93) | recommended (0.85) | خوب |
| rating_band | low | high (0.84) | high (0.84) | high (0.84) | خوب |

## 17. duplicate

> بوی جالبی نداره

rate 5.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | no_idea (0.55) | no_idea (0.65) | no_idea (0.62) | نداره, جالبی نداره, جالبی, بوی |
| rating_band | high | medium (0.60) | medium (0.63) | medium (0.62) | جالبی نداره, جالبی, نداره, بوی |

## 18. duplicate

> خوبه ولی خیلی نازکه

rate 5.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | no_idea (0.85) | recommended (0.68) | no_idea (0.47) | ولی, نازکه, خیلی نازکه, ولی خیلی نازکه |
| rating_band | high | medium (0.49) | medium (0.38) | medium (0.40) | ولی, نازکه, خوبه ولی, خیلی نازکه |

## 19. contradictory

> خوبه

rate 2.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | no_idea | recommended (0.67) | recommended (0.94) | recommended (0.86) | خوبه |
| rating_band | low | high (0.82) | high (0.84) | high (0.83) | خوبه |

## 20. contradictory

> معمولی

rate 3.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | no_idea (0.86) | no_idea (0.59) | no_idea (0.67) | معمولی |
| rating_band | medium | medium (0.80) | medium (0.76) | medium (0.77) | معمولی |

## 21. contradictory

> قیمت بالا

rate 4.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | no_idea (0.53) | no_idea (0.41) | no_idea (0.45) | قیمت بالا, قیمت |
| rating_band | high | medium (0.47) | medium (0.51) | medium (0.49) | قیمت بالا, قیمت |

## 22. boundary

> یک ماهه استفاده میکنم، خوله مشکلی نداشته، فقط فیک و سنگین بودنش حس بدی به آدم میده

rate 4.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | recommended (0.48) | no_idea (0.72) | no_idea (0.59) | مشکلی, ، فقط, به آدم, استفاده میکنم |
| rating_band | high | medium (0.33) | medium (0.71) | medium (0.59) | نداشته, به آدم میده, حس, حس بدی |

## 23. boundary

> بدک نیست

rate 4.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | none | | | | |
| rating_band | high | medium (0.59) | medium (0.57) | medium (0.58) | بدک, نیست |

## 24. boundary

> رایحه معمولی داره و ارزش خرید داره

rate 4.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | recommended (0.46) | recommended (0.91) | recommended (0.77) | ارزش خرید داره, خرید داره, داره, و |
| rating_band | high | medium (0.69) | medium (0.44) | medium (0.51) | معمولی داره و, معمولی, معمولی داره, رایحه معمولی |

## 25. disagreement (FaBERT right, TF-IDF wrong)

> کیفیتش مثل قبل نیست ولی باز خوبه

rate 4.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | no_idea (0.62) | recommended (0.78) | recommended (0.62) | ولی, باز خوبه, نیست, مثل قبل نیست |
| rating_band | high | medium (0.33) | medium (0.36) | medium (0.36) | قبل نیست, ولی, مثل قبل نیست, نیست |

## 26. disagreement (FaBERT right, TF-IDF wrong)

> خوبه بدک نیست

rate 4.0; text seen in train: yes

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | no_idea (0.67) | recommended (0.77) | recommended (0.64) | بدک, نیست, خوبه بدک نیست, خوبه بدک |
| rating_band | high | medium (0.35) | medium (0.39) | medium (0.38) | بدک, نیست, خوبه بدک, خوبه |

## 27. disagreement (TF-IDF right, FaBERT wrong)

> خوب بود ولی من معمولی میخواستم برام کبریتی فرستادن

rate 4.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | no_idea | no_idea (0.73) | recommended (0.68) | recommended (0.53) | ولی, معمولی, من معمولی, کبریتی |
| rating_band | high | medium (0.40) | medium (0.32) | medium (0.34) | من معمولی, معمولی, ولی, خوب بود |

## 28. disagreement (TF-IDF right, FaBERT wrong)

> یکم سخت در میاد

rate 4.0; text seen in train: no

| Head | True | TF-IDF | FaBERT | Final | TF-IDF words |
|---|---|---|---|---|---|
| recommendation | recommended | recommended (0.40) | no_idea (0.43) | no_idea (0.44) | یکم, یکم سخت, میاد, سخت در میاد |
| rating_band | high | medium (0.46) | medium (0.43) | medium (0.44) | یکم, سخت, میاد, سخت در |
