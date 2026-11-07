"""FaBERT's tokenizer in plain Python, read from its tokenizer.json, for machines without the
tokenizers library.

It follows the steps the tokenizers library runs for this file:

1. Special tokens written literally in the text ([UNK], [PAD], ...) are taken out first.
2. BertNormalizer: control and format characters are removed (the half-space is one), other
   white space becomes a space, CJK ideographs get spaces around them, accents are stripped
   (decomposition, then non-spacing marks removed, because lowercase is on), and letters are
   lowercased.
3. BertPreTokenizer: split on white space; every punctuation character is its own word.
4. WordPiece: longest match from the left, "##" for the following pieces; a word longer than
   max_input_chars_per_word, or one that cannot be fully matched, becomes one [UNK].

Each token keeps the span of the original text it came from. check_unknown_tokens.py checks
this module against the numbers the real tokenizer gave in reports/tokenizers/compare.json
before using it.
"""

import json
import re
import unicodedata
from pathlib import Path

CJK = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
]
PUNCTUATION = {"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps"}
OTHER = {"Cc", "Cf", "Cn", "Co", "Cs"}


def is_whitespace(c: str) -> bool:
    return c in " \t\n\r" or unicodedata.category(c) == "Zs"


def is_control(c: str) -> bool:
    return c not in "\t\n\r" and unicodedata.category(c) in OTHER


def is_punctuation(c: str) -> bool:
    return (c.isascii() and not c.isalnum() and c.isprintable() and c != " ") or (
        unicodedata.category(c) in PUNCTUATION
    )


def is_cjk(c: str) -> bool:
    code = ord(c)
    return any(a <= code <= b for a, b in CJK)


class BertWordPiece:
    def __init__(self, path: Path):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        model, normalizer, pre = spec["model"], spec["normalizer"], spec["pre_tokenizer"]
        assert model["type"] == "WordPiece", model["type"]
        assert normalizer["type"] == "BertNormalizer", normalizer["type"]
        assert pre["type"] == "BertPreTokenizer", pre["type"]
        self.vocab = model["vocab"]
        self.unk = model["unk_token"]
        self.unk_id = self.vocab[self.unk]
        self.prefix = model["continuing_subword_prefix"]
        self.max_chars = model["max_input_chars_per_word"]
        self.clean = normalizer["clean_text"]
        self.chinese = normalizer["handle_chinese_chars"]
        self.lowercase = normalizer["lowercase"]
        strip = normalizer["strip_accents"]
        self.strip = self.lowercase if strip is None else strip
        added = [t for t in spec["added_tokens"] if t["special"]]
        assert all(not t["normalized"] and not t["single_word"] for t in added)
        self.added = {t["content"]: t["id"] for t in added}
        pattern = "|".join(re.escape(t) for t in sorted(self.added, key=len, reverse=True))
        self.added_pattern = re.compile(f"({pattern})") if pattern else None
        self.memo = {}

    def vocab_size(self) -> int:
        return len(self.vocab)

    def normalize(self, text: str, offset: int) -> list[tuple[str, int]]:
        """Normalized characters, each with the position of the original character."""
        out = []
        for i, c in enumerate(text):
            position = offset + i
            if self.clean:
                if ord(c) in (0, 0xFFFD) or is_control(c):
                    continue
                if is_whitespace(c):
                    c = " "
            pieces = [" ", c, " "] if self.chinese and is_cjk(c) else [c]
            for p in pieces:
                if self.strip:
                    p = "".join(x for x in unicodedata.normalize("NFD", p) if unicodedata.category(x) != "Mn")
                if self.lowercase:
                    p = p.lower()
                out.extend((x, position) for x in p)
        return out

    def words(self, chars: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
        words, current = [], []
        for c, position in chars:
            if c.isspace():
                if current:
                    words.append(current)
                current = []
            elif is_punctuation(c):
                if current:
                    words.append(current)
                words.append([(c, position)])
                current = []
            else:
                current.append((c, position))
        if current:
            words.append(current)
        return words

    def wordpiece(self, word: str) -> list[tuple[int, int, int]]:
        """(token id, start, end) in characters of the normalized word."""
        if word in self.memo:
            return self.memo[word]
        if len(word) > self.max_chars:
            result = [(self.unk_id, 0, len(word))]
        else:
            result, start = [], 0
            while start < len(word):
                end, found = len(word), None
                while start < end:
                    piece = word[start:end] if start == 0 else self.prefix + word[start:end]
                    if piece in self.vocab:
                        found = self.vocab[piece]
                        break
                    end -= 1
                if found is None:
                    result = [(self.unk_id, 0, len(word))]
                    break
                result.append((found, start, end))
                start = end
        self.memo[word] = result
        return result

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Token ids and (start, end) spans in the original text, without [CLS] and [SEP]."""
        ids, spans = [], []
        segments = self.added_pattern.split(text) if self.added_pattern else [text]
        offset = 0
        for segment in segments:
            if segment in self.added:
                ids.append(self.added[segment])
                spans.append((offset, offset + len(segment)))
            elif segment:
                for word in self.words(self.normalize(segment, offset)):
                    letters = "".join(c for c, _ in word)
                    for token, a, b in self.wordpiece(letters):
                        ids.append(token)
                        spans.append((word[a][1], word[b - 1][1] + 1))
            offset += len(segment)
        return ids, spans

    def tokens(self, text: str) -> list[str]:
        inverse = {v: k for k, v in self.vocab.items()}
        return [inverse[i] for i in self.encode(text)[0]]
