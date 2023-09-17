"""A single registry of every metric the runner can score.

Each metric declares its family and its callable. The runner resolves requested
metric names through this table instead of a dispatch chain, so adding a metric
means adding one line here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .metrics import (char_bleu4, exact_match, mrr, ndcg_at_k, recall_at_k,
                      rouge_l, token_f1)


@dataclass(frozen=True)
class Metric:
    name: str
    family: str          # "text" or "retrieval"
    fn: Callable
    higher_is_better: bool = True


def _metric(name, family, fn):
    return Metric(name=name, family=family, fn=fn)


REGISTRY: dict[str, Metric] = {
    m.name: m for m in (
        _metric("exact_match", "text", exact_match),
        _metric("token_f1", "text", token_f1),
        _metric("rouge_l", "text", rouge_l),
        _metric("char_bleu4", "text", char_bleu4),
        _metric("recall_at_k", "retrieval", recall_at_k),
        _metric("mrr", "retrieval", mrr),
        _metric("ndcg_at_k", "retrieval", ndcg_at_k),
    )
}


def get_metric(name: str) -> Metric:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown metric: {name}") from exc


def metric_names(family: str | None = None) -> tuple[str, ...]:
    return tuple(
        name for name, metric in REGISTRY.items()
        if family is None or metric.family == family
    )


def describe() -> list[dict]:
    return [
        {"name": m.name, "family": m.family, "higher_is_better": m.higher_is_better}
        for m in REGISTRY.values()
    ]
