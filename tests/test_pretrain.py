import numpy as np
import pytest

from digikala_reviews.pretrain import IGNORE, mask_tokens, pack_blocks


def test_pack_blocks_joins_texts_and_drops_the_short_end():
    blocks = pack_blocks([[1, 5, 2], [1, 6, 7, 2], [1, 8, 2]], 4)
    assert blocks.tolist() == [[1, 5, 2, 1], [6, 7, 2, 1]]  # the last two tokens are dropped


def test_mask_tokens_chooses_a_share_of_the_ordinary_tokens():
    rng = np.random.default_rng(0)
    ids = np.full((200, 128), 50)
    ids[:, 0], ids[:, 64] = 1, 2  # special tokens
    inputs, labels = mask_tokens(ids, rng, mask_id=3, vocab_size=100, special_ids=[1, 2])
    chosen = labels != IGNORE
    assert chosen.mean() == pytest.approx(0.15 * 126 / 128, abs=0.01)
    assert not chosen[:, [0, 64]].any()  # special tokens are never chosen
    assert (labels[chosen] == 50).all()
    assert (inputs[chosen] == 3).mean() == pytest.approx(0.8, abs=0.03)
    assert (inputs[~chosen] == ids[~chosen]).all()  # the rest is left as it was


def test_mask_tokens_is_the_same_for_the_same_seed():
    ids = np.arange(1000).reshape(10, 100) % 97 + 3
    a = mask_tokens(ids, np.random.default_rng(5), 0, 100, [1, 2])
    b = mask_tokens(ids, np.random.default_rng(5), 0, 100, [1, 2])
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
