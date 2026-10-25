import pytest

from digikala_reviews.normalize import ZWNJ, normalize

CASES = [
    # (raw, expected, what it checks)
    ("\u0643\u064a\u0641\u064a\u062a \u0639\u0627\u0644\u064a", "کیفیت عالی", "Arabic kaf and yeh"),
    ("قیمت ۱۲۳ و \u0661\u0662\u0663 و 123", "قیمت 123 و 123 و 123", "all digits become Latin"),
    ("خوب\r\nبود\tواقعا", "خوب بود واقعا", "line breaks and tabs"),
    ("  خیلی    خوب  ", "خیلی خوب", "repeated and surrounding spaces"),
    ("ع\u0640\u0640\u0640الی", "عالی", "tatweel"),
    ("خوب\u200bبود\ufeff", "خوب بود", "zero-width space becomes a space, BOM is removed"),
    ("خو\u200dب", "خوب", "zero-width joiner is removed"),
    ("عالیییییی", "عالی", "repeated letters"),
    ("خیییییلی خوبه", "خیلی خوبه", "repeated letters inside a word"),
    ("مدد", "مدد", "two identical letters stay"),
    ("قیمتش 1000 تومن", "قیمتش 1000 تومن", "repeated digits stay"),
    ("Samsung GOOD", "samsung good", "Latin is lowercased"),
    ("سلام <br> خوب", "سلام خوب", "HTML tag"),
    ("ببینید https://example.com/a?b=1 خوبه", "ببینید خوبه", "URL"),
    (f"می{ZWNJ}خواهم", f"می{ZWNJ}خواهم", "half-space is kept"),
    (f"می{ZWNJ}{ZWNJ}خواهم", f"می{ZWNJ}خواهم", "repeated half-space"),
    (f"خوب {ZWNJ}بود{ZWNJ}", "خوب بود", "half-space next to a space or at the end"),
    (f"خوب {ZWNJ} {ZWNJ}بود", "خوب بود", "mixed run of spaces and half-spaces"),
    ("ار\u0627\u0640\u0654ه", "ار\u0623ه", "letter and mark split by tatweel are joined"),
    ("\u0640\u0640\u0640", "", "tatweel only becomes empty"),
    ("عالی بود 😍👍", "عالی بود 😍👍", "emoji stay"),
    ("\ufb8e\ufbff\ufed4\ufbff\ufe96", "کیفیت", "Arabic presentation forms (NFKC)"),
    ("", "", "empty text"),
]


@pytest.mark.parametrize("raw, expected, rule", CASES, ids=[c[2] for c in CASES])
def test_normalize(raw, expected, rule):
    assert normalize(raw) == expected


NEGATIONS = [f"نمی{ZWNJ}خرم", "نه", "خوب نبود", "اصلا راضی نیستم", "نخرید"]


@pytest.mark.parametrize("text", NEGATIONS, ids=[f"negation {i}" for i in range(1, len(NEGATIONS) + 1)])
def test_negation_is_untouched(text):
    assert normalize(text) == text


MESSY = "!!!؟؟؟ خوووب\n\n\u200c\u200c\ufefb \u0665\u0665\u0665"


@pytest.mark.parametrize("raw", [c[0] for c in CASES] + [MESSY], ids=[c[2] for c in CASES] + ["messy text"])
def test_normalizing_twice_changes_nothing(raw):
    once = normalize(raw)
    assert normalize(once) == once
