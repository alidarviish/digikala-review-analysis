import re

from digikala_reviews.tfidf import FEATURE_SETUPS, TOKEN_PATTERNS, ZWNJ, make_vectorizer


def tokens(pattern, text):
    return re.findall(TOKEN_PATTERNS[pattern], text)


def test_sklearn_pattern_splits_half_space_and_drops_emoji():
    assert tokens("sklearn", f"نمی{ZWNJ}خرم و 😍") == ["نمی", "خرم"]


def test_persian_pattern_keeps_half_space_words_short_words_and_emoji():
    assert tokens("persian", f"نمی{ZWNJ}خرم و 😍!") == [f"نمی{ZWNJ}خرم", "و", "😍", "!"]


def test_persian_pattern_keeps_emoji_marks_and_vowel_marks_attached():
    assert tokens("persian", "عالی ✌️ 👍🏻") == ["عالی", "✌️", "👍🏻"]
    assert tokens("persian", "کتابٌ خوب") == ["کتابٌ", "خوب"]


def test_every_setup_builds_and_learns_from_text():
    texts = [f"خیلی خوب بود 😍 نمی{ZWNJ}خرم"] * 6 + ["بد نبود"] * 6
    for name in FEATURE_SETUPS:
        matrix = make_vectorizer(name).fit_transform(texts)
        assert matrix.shape[0] == 12 and matrix.shape[1] > 0
        assert matrix.dtype == "float32"
