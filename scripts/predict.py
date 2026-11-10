"""Predict recommendation and rating band for review texts, offline (plan: reports/predictor_plan.md).

The text is normalized and tokenized, scored by TF-IDF, by FaBERT and by the final system
(both mixed, with "uncertain" for its least confident answers), and explained by the words
that pushed the TF-IDF model most. Only the text is given; no label or rate.

Needs artifacts/fabert_files once (scripts/save_fabert_files.py); runs on the CPU.

Usage:
    uv run python scripts/predict.py "متن نظر"
    uv run python scripts/predict.py --json "متن نظر"
    uv run python scripts/predict.py --demo      # the examples below; writes reports/predict_demo.md
"""

import argparse
import json
from datetime import UTC, datetime

from digikala_reviews.config import REPORTS_DIR
from digikala_reviews.predictor import Predictor
from digikala_reviews.tfidf import HEADS

DEMO_OUTPUT = REPORTS_DIR / "predict_demo.md"
# Written for this demonstration, not taken from the data
DEMO = [
    ("short", "عالی"),
    ("short", "بد بود"),
    ("ambiguous", "معمولیه، فعلا دارم استفاده می‌کنم ببینم چطوره"),
    ("negative", "بعد از دو هفته خراب شد و پشتیبانی هم جواب نداد. پولم هدر رفت."),
    ("mixed", "کیفیتش خوبه ولی برای این قیمت خیلی گرونه"),
    ("mixed", "صداش عالیه ولی باتریش زود تموم میشه و شارژرش هم داغ میکنه"),
    ("negation", "بد نیست"),
    ("negation", "اصلا ارزش خرید نداره"),
    ("sarcasm", "واقعا عالی بود، روز دوم شکست!"),
    ("spelling", "خیلی خوبع دستتون درد نکنه"),
    (
        "long",
        "این گوشی رو حدود سه ماه پیش خریدم و تا الان تجربهٔ نسبتاً خوبی داشتم. صفحه‌نمایشش روشن و "
        "رنگ‌هاش طبیعیه و زیر نور آفتاب هم به‌خوبی دیده میشه. دوربینش در روز عکس‌های خوبی می‌گیره ولی "
        "در شب نویز زیادی داره و جزئیات از بین میره. باتریش با استفادهٔ معمولی یک روز کامل دوام میاره، "
        "اما وقتی بازی می‌کنم زود داغ میشه و شارژش سریع کم میشه. بلندگوش صدای بلندی داره ولی کیفیتش "
        "متوسطه. نرم‌افزارش گاهی کند میشه و چند تا برنامهٔ اضافه داره که نمیشه پاکشون کرد. بسته‌بندی "
        "سالم بود و ارسال هم به‌موقع انجام شد. در کل برای این قیمت انتخاب بدی نیست، ولی اگه بودجهٔ "
        "بیشتری دارید شاید مدل بالاتر ارزشش رو داشته باشه.",
    ),
]


def show(result: dict) -> None:
    print(f"text:       {result['text']}")
    print(f"normalized: {result['normalized_text']}")
    cut = " (cut at 128)" if result["truncated"] else ""
    print(f"tokens:     {result['token_count']}{cut}: {' '.join(result['tokens'][:30])}")
    for head in HEADS:
        print(f"{head}:")
        for name, answer in result["models"].items():
            probabilities = ", ".join(f"{c} {p:.3f}" for c, p in answer[head]["probabilities"].items())
            print(f"  {name:<13} {answer[head]['label']:<16} ({probabilities})")
        e = result["explanation"][head]
        words = ", ".join(w["term"] for w in e["tfidf_words_for_its_answer"]) or "none"
        print(f"  models agree: {'yes' if e['models_agree'] else 'no'}; TF-IDF words for its answer: {words}")
    latency = ", ".join(f"{k} {v:.1f} ms" for k, v in result["latency_ms"].items())
    print(f"latency: {latency}\n")


def demo_report(predictor: Predictor) -> str:
    lines = [
        "# Offline prediction: examples",
        "",
        f"Made by `scripts/predict.py --demo` on {datetime.now(UTC).date()}, code "
        f"{predictor.version['code']['commit'][:7]}. The reviews were written for this demonstration; "
        "none is from the data. Labels with each model's offsets for the test set; the final system "
        "answers `uncertain` when its confidence is under the threshold fitted on validation "
        "(`reports/uncertain.json`), and its best guess is shown in brackets.",
        "",
    ]
    for kind, text in DEMO:
        r = predictor.predict(text)
        lines += [f"## {kind}: {text}", ""]
        lines.append(
            f"{r['token_count']} tokens{' (cut at 128)' if r['truncated'] else ''}; latency: TF-IDF "
            f"{r['latency_ms']['tfidf']:.1f} ms, FaBERT {r['latency_ms']['fabert']:.1f} ms."
        )
        lines += [
            "",
            "| Head | TF-IDF | FaBERT | Final system | TF-IDF words for its answer |",
            "|---|---|---|---|---|",
        ]
        for head in HEADS:
            cells = []
            for name in ("tfidf", "fabert"):
                a = r["models"][name][head]
                cells.append(f"{a['label']} ({a['probabilities'][a['label']]:.2f})")
            f = r["models"]["final_system"][head]
            final = f["label"] if f["label"] != "uncertain" else f"uncertain [{f['best_guess']}]"
            words = ", ".join(w["term"] for w in r["explanation"][head]["tfidf_words_for_its_answer"])
            lines.append(f"| {head} | {cells[0]} | {cells[1]} | {final} ({f['confidence']:.2f}) | {words} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("texts", nargs="*")
    parser.add_argument("--json", action="store_true", help="print the full answer as JSON")
    parser.add_argument("--demo", action="store_true", help="run the examples and write the report")
    args = parser.parse_args()
    predictor = Predictor()
    if args.demo:
        DEMO_OUTPUT.write_text(demo_report(predictor), encoding="utf-8")
        print(f"saved {DEMO_OUTPUT}")
        return
    for text in args.texts:
        result = predictor.predict(text)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            show(result)


if __name__ == "__main__":
    main()
