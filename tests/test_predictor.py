import numpy as np
import pytest
from scipy import sparse

from digikala_reviews.predictor import final_answer, offsets_array, top_words


def test_offsets_follow_the_order_of_the_labels():
    assert offsets_array({"b": 0.2, "a": 0.0}, ["a", "b"]).tolist() == [0.0, 0.2]


def test_final_answer_mixes_adds_offsets_and_marks_low_confidence():
    log_fabert = np.log(np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]]))
    log_tfidf = np.log(np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]]))
    codes, confidence, unsure = final_answer(log_fabert, log_tfidf, 0.5, np.zeros(3), threshold=0.5)
    assert codes.tolist() == [0, 1]
    assert np.allclose(confidence, [0.8, 0.425])
    assert unsure.tolist() == [False, True]
    # an offset can change the answer; the confidence is then the mixed probability of the new one
    codes, confidence, _ = final_answer(log_fabert, log_tfidf, 0.5, np.array([0.0, 0.0, 2.0]), threshold=0.5)
    assert codes.tolist() == [0, 2] and np.isclose(confidence[1], 0.225)


def test_top_words_are_the_terms_pushing_towards_the_class():
    coef = np.array([[2.0, -1.0, 0.0], [-2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    names = np.array(["good", "bad", "the"])
    row = sparse.csr_matrix(np.array([[0.5, 0.5, 0.7]]))
    words = top_words(row, coef, names, code=0)
    assert [w["term"] for w in words] == ["good"]  # "bad" pushes away, "the" not at all
    assert words[0]["push"] == pytest.approx(1.0)
    assert [w["term"] for w in top_words(row, coef, names, code=1)] == ["bad"]


def test_a_model_built_from_a_configuration_takes_saved_weights():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from digikala_reviews.finetune import build_model

    config = transformers.BertConfig(
        vocab_size=50, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16
    )
    first = build_model(torch, "unused", 0.1, ["a", "b"], config=config)
    second = build_model(torch, "unused", 0.1, ["a", "b"], config=config)
    second.load_state_dict(first.state_dict())  # strict: every weight has a place
    first.eval(), second.eval()
    ids = torch.tensor([[1, 5, 7, 2]])
    mask = torch.ones_like(ids)
    assert torch.allclose(first(ids, mask)["a"], second(ids, mask)["a"])


def test_the_rate_head_gives_one_number_per_review():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from digikala_reviews.finetune import build_model

    config = transformers.BertConfig(
        vocab_size=50, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16
    )
    model = build_model(torch, "unused", 0.1, ["a", "b"], config=config, rate_head=True)
    ids = torch.tensor([[1, 5, 7, 2], [1, 6, 2, 0]])
    found = model(ids, (ids > 0).long())
    assert set(found) == {"a", "b", "rate"}
    assert found["a"].shape == (2, 3) and found["rate"].shape == (2,)
