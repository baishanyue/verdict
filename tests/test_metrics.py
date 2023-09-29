from __future__ import annotations

import math

import pytest

from verdict.metrics import (
    char_bleu4,
    exact_match,
    lcs_length,
    mrr,
    ndcg_at_k,
    normalize_text,
    recall_at_k,
    rouge_l,
    token_f1,
    tokenize,
)


def test_normalisation_and_tokenisation() -> None:
    assert normalize_text("  The   CAT\n") == "the cat"
    assert tokenize("The cat, sat!") == ["the", "cat", "sat"]
    # NFKC folds the fullwidth letters.
    assert tokenize("ＣＡＴ") == ["cat"]


def test_exact_match_normalises_case_punctuation_and_articles() -> None:
    assert exact_match("The Cat.", "cat") == 1.0
    assert exact_match("a dog", "dog") == 1.0
    assert exact_match("cat", "dog") == 0.0
    assert exact_match("", "") == 1.0
    assert exact_match("", "cat") == 0.0
    assert exact_match("the", "") == 0.0
    assert exact_match("the", "a") == 1.0


def test_token_f1_known_value() -> None:
    assert token_f1("the cat sat", "the cat sat on the mat") == pytest.approx(2 / 3)
    assert token_f1("cat", "the cat") == 1.0
    assert token_f1("cat", "dog") == 0.0
    assert token_f1("", "") == 1.0
    assert token_f1("cat", "") == 0.0
    assert token_f1("the", "") == 0.0
    assert token_f1("the", "the") == 1.0


def test_rouge_l_known_value_and_lcs() -> None:
    assert lcs_length(["a", "b", "c"], ["a", "c"]) == 2
    assert lcs_length([], ["a"]) == 0
    assert rouge_l("the cat sat", "the cat sat on the mat") == pytest.approx(2 / 3)
    assert rouge_l("cat", "the cat") == 1.0
    assert rouge_l("cat", "dog") == 0.0
    assert rouge_l("", "") == 1.0
    assert rouge_l("cat", "") == 0.0
    assert rouge_l("the", "") == 0.0


def test_char_bleu_identical_and_disjoint() -> None:
    sentence = "the cat sat on the mat"
    assert char_bleu4(sentence, sentence) == pytest.approx(1.0)
    assert char_bleu4("zzzz", "abcd") == 0.0
    assert char_bleu4("", "") == 1.0
    assert char_bleu4("cat", "") == 0.0
    # Documented consequence of no smoothing: too short for a 4-gram.
    assert char_bleu4("cat", "cat") == 0.0


def test_char_bleu_brevity_penalty() -> None:
    # All n-gram precisions are 1, so the score is exactly the brevity penalty.
    assert char_bleu4("abcd", "abcdefgh") == pytest.approx(math.exp(-1.0))


def test_recall_at_k() -> None:
    retrieved = ["b", "a", "c"]
    relevant = ["a", "c"]
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert recall_at_k(["a", "a", "b"], ["a", "b"], 2) == 0.5
    assert recall_at_k(["a"], [], 5) == 0.0
    assert recall_at_k("a", "a", 1) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(retrieved, relevant, 0)


def test_mrr() -> None:
    assert mrr(["x", "a", "c"], ["a"]) == 0.5
    assert mrr(["x", "a", "c"], ["a"], k=1) == 0.0
    assert mrr(["x", "a", "c"], ["a"], k=2) == 0.5
    assert mrr(["x", "x", "a"], ["a"]) == pytest.approx(1 / 3)
    assert mrr(["x"], ["a"]) == 0.0
    assert mrr(["a"], []) == 0.0
    with pytest.raises(ValueError):
        mrr(["a"], ["a"], k=0)


def test_ndcg_at_k() -> None:
    score = ndcg_at_k(["a", "x", "c"], ["a", "c"], 3)
    expected = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert score == pytest.approx(expected)
    assert ndcg_at_k(["x", "a"], ["a"], 2) == pytest.approx(1.0 / math.log2(3))
    assert ndcg_at_k(["a", "x"], ["a"], 2) == 1.0
    assert ndcg_at_k(["x"], ["a"], 1) == 0.0
    assert ndcg_at_k(["a"], [], 1) == 0.0
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], ["a"], 0)


def test_metric_type_errors() -> None:
    with pytest.raises(TypeError):
        exact_match(1, "a")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        recall_at_k([1], ["a"], 1)  # type: ignore[list-item]
