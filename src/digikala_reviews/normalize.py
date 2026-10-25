"""Persian text normalization, shared by training, evaluation and inference.

Every model sees text that went through `normalize`, so the same review always becomes
the same string. Only changes that do not alter meaning are made (see section 6.7 of
notebooks/01_descriptive_analysis.ipynb for how often each pattern occurs):

1. Unicode NFKC, which turns Arabic presentation forms and full-width characters into
   their standard letters
2. Arabic yeh and kaf become Persian yeh and kaf
3. Persian and Arabic digits become Latin digits
4. HTML tags and URLs are removed
5. Tatweel and invisible characters are removed; a zero-width space becomes a normal space,
   because it separates words. The half-space is kept.
6. A Persian/Arabic character repeated three or more times is kept once (خیییلی -> خیلی)
7. Latin letters are lowercased
8. All whitespace (line breaks, tabs, repeated spaces) becomes a single space
9. Half-spaces are kept, but repeated ones and ones next to a space are cleaned up

Not changed on purpose: emoji, punctuation, negation words, stopwords and spelling.
A misspelled word can also be a real word (for example a name), so spelling is left
to the models, which see each word in its context.

Invisible characters are written as escape codes so that they can be read in the source.
"""

import re
import unicodedata

NORMALIZER_VERSION = "1"

ZWNJ = "\u200c"  # half-space
ZERO_WIDTH_SPACE = "\u200b"

ARABIC_TO_PERSIAN = str.maketrans({"\u064a": "\u06cc", "\u0649": "\u06cc", "\u0643": "\u06a9"})
PERSIAN_DIGITS = "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
ARABIC_DIGITS = "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
TO_LATIN_DIGITS = str.maketrans(PERSIAN_DIGITS + ARABIC_DIGITS, "0123456789" * 2)

# Tatweel, zero-width joiner, direction marks and the byte-order mark
INVISIBLE = "\u0640\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff"
REMOVE_INVISIBLE = str.maketrans({ZERO_WIDTH_SPACE: " "} | {char: None for char in INVISIBLE})

HTML_TAG = re.compile(r"<[A-Za-z/][^>]*>")
URL = re.compile(r"https?://\S+|www\.\S+")
REPEATED_PERSIAN_CHAR = re.compile(r"([\u0600-\u06ff])\1{2,}")
WHITESPACE = re.compile(r"\s+")
REPEATED_ZWNJ = re.compile(ZWNJ + "{2,}")
ZWNJ_NEXT_TO_SPACE = re.compile(f" ?{ZWNJ} | {ZWNJ}")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ARABIC_TO_PERSIAN).translate(TO_LATIN_DIGITS)
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" ", text)
    text = text.translate(REMOVE_INVISIBLE)
    text = REPEATED_PERSIAN_CHAR.sub(r"\1", text)
    text = text.lower()
    text = WHITESPACE.sub(" ", text)
    text = REPEATED_ZWNJ.sub(ZWNJ, text)
    text = ZWNJ_NEXT_TO_SPACE.sub(" ", text)
    return text.strip(" " + ZWNJ)
