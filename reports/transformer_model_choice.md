# Choosing the Transformer model (written before any of it was run)

## Candidates

| Model | Size | Why it is a candidate | Published sentiment score |
|---|---|---|---|
| PartAI/TookaBERT-Base | about 0.1B | Newest Persian BERT, trained on over 500 GB of Persian text including forums and blogs | DeepSentiPers 83.9 F1 |
| sbunlp/fabert | 124M | Trained on 20 million Persian blog posts, much of it informal; WordPiece adapted to the half-space | MirasOpinion 87.5, DeepSentiPers 79.9 macro F1 |
| FacebookAI/xlm-roberta-base | 278M (mostly embeddings) | The usual multilingual reference | DeepSentiPers 84.0 F1 |
| jhu-clsp/mmBERT-base | 307M (110M without embeddings) | Modern encoder (ModernBERT design), 1800+ languages including Persian, up to 8192 tokens | No separate Persian number found |
| HooshvareLab/bert-fa-zwnj-base | about 110M | The classic Persian reference (ParsBERT) | DeepSentiPers 80.2 F1 |

Left out: TookaBERT-Large (access on request, three times larger; reconsidered if GPU time
allows), Persian-ModernBert (no published results), and any model already fine-tuned on
Digikala reviews (it may have seen our validation and test texts).

Published scores differ by about a point between the best candidates, and on other data,
so the choice is made on our own reviews in two steps.

## Step 1: tokenizers (`scripts/compare_tokenizers.py`)

50,000 labeled validation reviews, normalized as in training. Expected, before running:

- Persian tokenizers (TookaBERT, FaBERT, ParsBERT) cut words into fewer pieces than the
  multilingual ones (XLM-R, mmBERT).
- With any tokenizer, fewer than 5% of reviews are longer than 128 tokens and fewer than 1%
  longer than 256 (99% of reviews have at most 62 words).
- Unknown tokens are rare (well under 1%) for all but possibly ParsBERT, whose vocabulary is
  the oldest.

## Step 2: a small equal fine-tuning run (`scripts/pilot_transformers.py`, on a GPU)

Step 1 left every candidate in (reports/tokenizers/compare.json): at most 0.4% of reviews
are longer than 128 tokens with any tokenizer, and unknown tokens are under 0.3%. The
Persian tokenizers use about 20% fewer tokens than XLM-R and mmBERT. TookaBERT-Large is
added, as the GPU is large enough.

Same settings for every model: 100,000 training reviews and 50,000 validation reviews
(fixed samples, seed 2026); one encoder with two linear heads on the first token; the sum
of the two cross-entropies, without class weights; full fine-tuning, 1 epoch, batch 64,
learning rate 3e-5 (2e-5 for the large model), AdamW with weight decay 0.01, 6% warmup then
linear decay, 128 tokens; per-class offsets tuned on the other half of the sample, as for
TF-IDF. The final TF-IDF model is scored on the same 50,000 reviews, and each model is
compared with it (paired bootstrap). A sample of 50,000 is enough to see a difference of
about one point (reports/tfidf_analysis/sample.json).

Expected, before running:

- TookaBERT-Large scores highest.
- TookaBERT-Base, FaBERT and XLM-R are within one point of each other.
- ParsBERT is the lowest of the Persian models.
- Even with 100,000 training reviews, the best model matches or beats the final TF-IDF
  model (trained on all of train) with offsets, most of all on the rating band.
- A base model takes under 10 minutes, the large one two to three times as long.

The best model becomes the main model, fine-tuned afterwards on more data.
