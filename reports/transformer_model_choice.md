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

## Step 2: a small equal fine-tuning run (later, on a GPU)

The models left after step 1 are fine-tuned with the same settings on the same sample, and
scored on a fixed sample of 50,000 validation reviews (enough to see a difference of about
one point; see reports/tfidf_analysis/sample.json). The best one becomes the main model.
