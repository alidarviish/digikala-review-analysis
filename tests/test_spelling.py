import pandas as pd

from digikala_reviews.spelling import fix_spelling, load_fixes
from digikala_reviews.tfidf import ZWNJ

FIXES = {"پشنهاد": "پیشنهاد", "وقیمت": "و قیمت", "خوبع": "خوبه"}


def test_whole_words_are_fixed():
    texts = pd.Series(["پشنهاد میکنم", "خوبع وقیمت مناسب", "بدون غلط"])
    assert fix_spelling(texts, FIXES).tolist() == ["پیشنهاد میکنم", "خوبه و قیمت مناسب", "بدون غلط"]


def test_words_inside_longer_words_are_left_alone():
    texts = pd.Series(["پشنهادش", "اپشنهاد", f"پشنهاد{ZWNJ}ها", "پشنهاد!"])
    assert fix_spelling(texts, FIXES).tolist() == ["پشنهادش", "اپشنهاد", f"پشنهاد{ZWNJ}ها", "پیشنهاد!"]


def test_fixing_twice_changes_nothing_more():
    texts = pd.Series(["پشنهاد وقیمت خوبع"])
    once = fix_spelling(texts, FIXES)
    assert fix_spelling(once, FIXES).tolist() == once.tolist()


def test_the_list_in_the_repository_loads():
    fixes = load_fixes()
    assert len(fixes) > 500
    assert fixes["پشنهاد"] == "پیشنهاد" and fixes["خئب"] == "خوب"
    assert "دارد" not in fixes  # a formal verb, marked keep
