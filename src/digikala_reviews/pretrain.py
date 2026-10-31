"""Continued masked-word pretraining of FaBERT on the review texts
(reports/domain_adaptation_plan.md)."""

import itertools

import numpy as np

IGNORE = -100  # label of a token that was not chosen; the loss skips it


def pack_blocks(token_lists, block_size: int) -> np.ndarray:
    """Join the tokenized texts end to end (each with its own special tokens) and cut the
    stream into blocks of block_size tokens. The texts are short, so packing wastes no
    compute on padding. The last, shorter piece is dropped."""
    stream = np.fromiter(itertools.chain.from_iterable(token_lists), dtype=np.int32)
    n_blocks = len(stream) // block_size
    return stream[: n_blocks * block_size].reshape(n_blocks, block_size)


def mask_tokens(ids, rng, mask_id: int, vocab_size: int, special_ids, share: float = 0.15):
    """BERT masking: a share of the tokens that are not special tokens is chosen; of those,
    80% become the mask token, 10% a random token and 10% stay as they are. Returns the
    inputs and the labels (the original token where chosen, IGNORE elsewhere)."""
    ids = np.asarray(ids)
    chosen = ~np.isin(ids, np.asarray(special_ids)) & (rng.random(ids.shape) < share)
    labels = np.where(chosen, ids, IGNORE)
    inputs = ids.copy()
    r = rng.random(ids.shape)
    inputs[chosen & (r < 0.8)] = mask_id
    swapped = chosen & (r >= 0.8) & (r < 0.9)
    inputs[swapped] = rng.integers(0, vocab_size, int(swapped.sum()))
    return inputs, labels
