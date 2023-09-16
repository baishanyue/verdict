"""Reference metrics for question answering and retrieval.

Every metric here is a pure function of strings or sequences of identifiers. No
model, tokenizer download or network access is involved, which keeps scores
stable across machines and cheap enough to run in CI.

Normalisation is deliberately simple: Unicode NFKC, lower case, punctuation
stripped through word tokenisation, and English articles removed for the answer
metrics. Retrieval metrics compare identifiers exactly (or verbatim passage
text, if that is what a dataset stores in ``contexts``).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

__all__ = [
    "char_bleu4",
    "content_tokens",
    "exact_match",
    "lcs_length",
    "mrr",
    "ndcg_at_k",
    "normalize_text",
    "recall_at_k",
    "rouge_l",
    "token_f1",
    "tokenize",
]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_ARTICLES = frozenset({"a", "an", "the"})


def normalize_text(text: str) -> str:
    """Apply NFKC, lower-case and whitespace normalisation.

    Punctuation is retained here; :func:`tokenize` removes it when word
    boundaries are needed. Character-level BLEU uses this form directly so that
    spacing and punctuation differences still count.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def tokenize(text: str) -> list[str]:
    """Split normalised text into Unicode word tokens."""
    return _TOKEN_RE.findall(normalize_text(text))


def content_tokens(text: str) -> list[str]:
    """Return word tokens with English articles removed.

    This is the token form used by exact match, token F1, ROUGE-L and the
    heuristic judge's coverage dimension. Removing ``a``, ``an`` and ``the``
    stops article noise from changing whether two answers agree.
    """
    return [token for token in tokenize(text) if token not in _ARTICLES]


def _coerce_ids(values: Sequence[str] | str, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    if not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{name} must contain only strings")
        result.append(value)
    return tuple(result)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Return the length of the longest common subsequence of two token lists."""
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 when the normalised strings are identical, else 0.0.

    ``EM = 1[normalise(prediction) == normalise(reference)]`` where
    normalisation is case folding, punctuation removal, article removal and
    whitespace collapse. Empty answers match each other.
    """
    predicted = normalize_text(prediction)
    expected = normalize_text(reference)
    if not predicted or not expected:
        return 1.0 if predicted == expected else 0.0
    return 1.0 if content_tokens(prediction) == content_tokens(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Return token-level F1 over normalised words.

    With ``C`` the multiset overlap of prediction and reference tokens::

        P = |C| / |prediction|
        R = |C| / |reference|
        F1 = 2 * P * R / (P + R)

    Returns 1.0 when both sides are empty and 0.0 when exactly one is empty.
    """
    predicted_text = normalize_text(prediction)
    expected_text = normalize_text(reference)
    if not predicted_text and not expected_text:
        return 1.0
    if not predicted_text or not expected_text:
        return 0.0
    predicted = content_tokens(prediction)
    expected = content_tokens(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """Return ROUGE-L F1 based on the longest common subsequence.

    With ``L`` the LCS length in normalised tokens::

        P = L / |prediction|
        R = L / |reference|
        F1 = 2 * P * R / (P + R)

    Empty/empty scores 1.0; empty on one side scores 0.0.
    """
    predicted_text = normalize_text(prediction)
    expected_text = normalize_text(reference)
    if not predicted_text and not expected_text:
        return 1.0
    if not predicted_text or not expected_text:
        return 0.0
    predicted = content_tokens(prediction)
    expected = content_tokens(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    length = lcs_length(predicted, expected)
    if length == 0:
        return 0.0
    precision = length / len(predicted)
    recall = length / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def _char_ngrams(text: str, order: int) -> Counter[str]:
    return Counter(text[index : index + order] for index in range(len(text) - order + 1))


def char_bleu4(prediction: str, reference: str) -> float:
    """Return sentence-level character BLEU-4 with a brevity penalty.

    Normalise both strings, count character n-grams for ``n = 1..4`` and clip
    each count by the reference count::

        p_n = clipped_n / possible_n
        BP  = 1                       if len(prediction) >= len(reference)
        BP  = exp(1 - len(reference) / len(prediction))  otherwise
        BLEU = BP * (p_1 * p_2 * p_3 * p_4) ** (1/4)

    No smoothing is applied, so any zero precision makes the score zero. That is
    deliberately harsh: a three-character answer cannot supply a 4-gram and is
    not rewarded. Empty/empty scores 1.0; empty on one side scores 0.0.
    """
    predicted = normalize_text(prediction)
    expected = normalize_text(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    log_precision = 0.0
    for order in range(1, 5):
        possible = len(predicted) - order + 1
        if possible <= 0:
            return 0.0
        predicted_ngrams = _char_ngrams(predicted, order)
        expected_ngrams = _char_ngrams(expected, order)
        clipped = sum(
            min(count, expected_ngrams[gram])
            for gram, count in predicted_ngrams.items()
        )
        if clipped == 0:
            return 0.0
        log_precision += math.log(clipped / possible)
    brevity = 1.0
    if len(predicted) < len(expected):
        brevity = math.exp(1.0 - len(expected) / len(predicted))
    return brevity * math.exp(log_precision / 4.0)


def recall_at_k(
    retrieved: Sequence[str] | str,
    relevant: Sequence[str] | str,
    k: int,
) -> float:
    """Return hit rate over the gold identifiers in the top ``k`` results.

    ``Recall@k = |set(retrieved[:k]) & set(relevant)| / |set(relevant)|``.
    Duplicate retrieved ids count once. Returns 0.0 when ``relevant`` is empty.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    ranked = _coerce_ids(retrieved, "retrieved")
    gold = set(_coerce_ids(relevant, "relevant"))
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def mrr(
    retrieved: Sequence[str] | str,
    relevant: Sequence[str] | str,
    k: int | None = None,
) -> float:
    """Return reciprocal rank of the first relevant result.

    ``MRR = 1 / rank`` for the first hit within the top ``k`` (all results when
    ``k`` is ``None``), otherwise 0.0. Duplicate ids are ignored after their
    first appearance. Returns 0.0 when ``relevant`` is empty.
    """
    if k is not None and k <= 0:
        raise ValueError("k must be a positive integer or None")
    ranked = _coerce_ids(retrieved, "retrieved")
    gold = set(_coerce_ids(relevant, "relevant"))
    if not gold:
        return 0.0
    limit = len(ranked) if k is None else k
    seen: set[str] = set()
    for position, identifier in enumerate(ranked[:limit], start=1):
        if identifier in seen:
            continue
        seen.add(identifier)
        if identifier in gold:
            return 1.0 / position
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str] | str,
    relevant: Sequence[str] | str,
    k: int,
) -> float:
    """Return binary-relevance nDCG over the top ``k`` results.

    ``DCG@k = sum_{i=1..k} rel_i / log2(i + 1)`` and
    ``IDCG@k = sum_{i=1..min(|relevant|, k)} 1 / log2(i + 1)``;
    the score is ``DCG@k / IDCG@k``. Duplicate ids count once. Returns 0.0 when
    ``relevant`` is empty.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    ranked = _coerce_ids(retrieved, "retrieved")
    gold = set(_coerce_ids(relevant, "relevant"))
    if not gold:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for position, identifier in enumerate(ranked[:k], start=1):
        if identifier in seen:
            continue
        seen.add(identifier)
        if identifier in gold:
            dcg += 1.0 / math.log2(position + 1)
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(len(gold), k) + 1)
    )
    return dcg / ideal if ideal else 0.0
