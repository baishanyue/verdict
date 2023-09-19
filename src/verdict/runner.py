"""Suite execution and aggregation.

The runner joins cases to predictions by id, evaluates every requested metric
for every case exactly once, and aggregates the per-case scores. Results are
sorted by case id so two runs over the same suite produce byte-identical JSON
regardless of prediction file order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from .suites.datasets import Case
from .judge.judge import Judge, Judgment
from .metrics import (
    char_bleu4,
    exact_match,
    mrr,
    ndcg_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)

__all__ = [
    "DEFAULT_METRICS",
    "CaseResult",
    "MetricSummary",
    "RETRIEVAL_METRICS",
    "RunError",
    "RunResult",
    "TEXT_METRICS",
    "load_run",
    "percentile",
    "run_suite",
    "save_run",
    "summarize",
]

from .registry import get_metric, metric_names

TEXT_METRICS = metric_names("text")
RETRIEVAL_METRICS = metric_names("retrieval")
DEFAULT_METRICS = TEXT_METRICS
SUPPORTED_METRICS = TEXT_METRICS + RETRIEVAL_METRICS + ("judge",)


class RunError(ValueError):
    """Raised for invalid suites, predictions or serialised run files."""


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Aggregate statistics for one metric across the evaluated cases."""

    count: int
    mean: float
    median: float
    p95: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "p95": self.p95,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One case's prediction, metric scores and optional judgment."""

    case_id: str
    prediction: str
    scores: Mapping[str, float]
    judge: Judgment | None = None
    missing_prediction: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "prediction": self.prediction,
            "missing_prediction": self.missing_prediction,
            "scores": {name: float(value) for name, value in self.scores.items()},
            "judge": self.judge.to_dict() if self.judge is not None else None,
        }


@dataclass(frozen=True, slots=True)
class RunResult:
    """A complete, serialisable evaluation run."""

    suite: str
    metrics: tuple[str, ...]
    k: int
    cases: tuple[CaseResult, ...]
    summary: Mapping[str, MetricSummary]
    missing_predictions: tuple[str, ...] = ()
    ignored_predictions: tuple[str, ...] = ()
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "metrics": list(self.metrics),
            "k": self.k,
            "summary": {
                name: self.summary[name].to_dict() for name in self.metrics
                if name in self.summary
            },
            "cases": [case.to_dict() for case in self.cases],
            "missing_predictions": list(self.missing_predictions),
            "ignored_predictions": list(self.ignored_predictions),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n"

    def save(self, path: str | Path) -> None:
        """Write the run as UTF-8 JSON, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunResult":
        """Rebuild a run from :meth:`to_dict` output.

        Aggregates are recomputed from the per-case scores so a hand-edited
        summary cannot leak into a regression comparison.
        """
        if not isinstance(raw, Mapping):
            raise RunError("run file must contain a JSON object")
        metrics_raw = raw.get("metrics", [])
        if not isinstance(metrics_raw, list) or not all(
            isinstance(name, str) for name in metrics_raw
        ):
            raise RunError("run 'metrics' must be a list of strings")
        metrics = tuple(metrics_raw)
        if not metrics:
            raise RunError("run file must list at least one metric")
        if len(set(metrics)) != len(metrics):
            raise RunError("run 'metrics' must not contain duplicates")
        cases_raw = raw.get("cases")
        if not isinstance(cases_raw, list) or not cases_raw:
            raise RunError("run file must contain at least one case")
        k_raw = raw.get("k", 5)
        if not isinstance(k_raw, int) or k_raw <= 0:
            raise RunError("run 'k' must be a positive integer")

        case_results: list[CaseResult] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(cases_raw):
            if not isinstance(item, Mapping):
                raise RunError(f"case #{index} must be an object")
            case_id = item.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise RunError(f"case #{index} needs a non-empty string 'id'")
            if case_id in seen_ids:
                raise RunError(f"duplicate case id {case_id!r}")
            seen_ids.add(case_id)
            prediction = item.get("prediction", "")
            if not isinstance(prediction, str):
                raise RunError(f"case {case_id!r} 'prediction' must be a string")
            scores_raw = item.get("scores", {})
            if not isinstance(scores_raw, Mapping):
                raise RunError(f"case {case_id!r} 'scores' must be an object")
            scores: dict[str, float] = {}
            for name, value in scores_raw.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise RunError(f"case {case_id!r} score {name!r} must be a number")
                if not math.isfinite(float(value)):
                    raise RunError(f"case {case_id!r} score {name!r} must be finite")
                scores[str(name)] = float(value)
            judge_raw = item.get("judge")
            judgment: Judgment | None = None
            if isinstance(judge_raw, Mapping):
                judgment = Judgment.from_dict(judge_raw)
            elif judge_raw is not None:
                raise RunError(f"case {case_id!r} 'judge' must be an object or null")
            case_results.append(
                CaseResult(
                    case_id=case_id,
                    prediction=prediction,
                    scores=scores,
                    judge=judgment,
                    missing_prediction=bool(item.get("missing_prediction", False)),
                )
            )
        case_results.sort(key=lambda result: result.case_id)
        summary: dict[str, MetricSummary] = {}
        for name in metrics:
            missing_scores = [
                result.case_id for result in case_results if name not in result.scores
            ]
            if missing_scores:
                raise RunError(
                    f"run case rows are missing scores for {name!r}: "
                    + ", ".join(missing_scores)
                )
            summary[name] = summarize(
                [result.scores[name] for result in case_results]
            )
        missing = raw.get("missing_predictions", [])
        ignored = raw.get("ignored_predictions", [])
        if not isinstance(missing, list) or not isinstance(ignored, list):
            raise RunError("run prediction id fields must be lists")
        return cls(
            suite=str(raw.get("suite", "suite")),
            metrics=metrics,
            k=k_raw,
            cases=tuple(case_results),
            summary=summary,
            missing_predictions=tuple(sorted(str(item) for item in missing)),
            ignored_predictions=tuple(sorted(str(item) for item in ignored)),
            schema_version=int(raw.get("schema_version", 1)),
        )


def percentile(values: Sequence[float], q: float) -> float:
    """Return the ``q`` quantile using linear interpolation.

    ``q`` is in ``[0, 1]``. With ``n`` sorted values, the position is
    ``(n - 1) * q`` and the result interpolates between neighbouring values.
    This makes p95 well defined for small case counts as well.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values: Sequence[float]) -> MetricSummary:
    """Aggregate a non-empty sequence into mean, median and p95."""
    if not values:
        raise ValueError("cannot summarise an empty set of scores")
    numbers = [float(value) for value in values]
    return MetricSummary(
        count=len(numbers),
        mean=mean(numbers),
        median=median(numbers),
        p95=percentile(numbers, 0.95),
        minimum=min(numbers),
        maximum=max(numbers),
    )


def _resolve_retrieval(case: Case) -> tuple[tuple[str, ...], tuple[str, ...]]:
    relevant = case.relevant or case.contexts
    return case.retrieved, relevant


def run_suite(
    cases: Sequence[Case],
    predictions: Mapping[str, str],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    k: int = 5,
    judge: Judge | None = None,
    suite: str = "suite",
) -> RunResult:
    """Evaluate ``predictions`` over ``cases`` for every requested metric.

    Args:
        cases: Validated cases; ids must be unique and non-empty.
        predictions: ``{case_id: prediction}``. Missing predictions are scored
            as empty strings and listed in ``RunResult.missing_predictions``;
            ids that do not match a case are listed as ignored.
        metrics: Metric names from ``exact_match``, ``token_f1``, ``rouge_l``,
            ``char_bleu4``, ``recall_at_k``, ``mrr``, ``ndcg_at_k`` and
            ``judge``. Order is preserved in the output.
        k: Cutoff used by the retrieval metrics.
        judge: Required only when ``"judge"`` is in ``metrics``.
        suite: Label stored in the run.

    Returns:
        A :class:`RunResult` with per-case rows sorted by case id.

    Raises:
        RunError: For empty/duplicate cases, unknown metrics, a missing judge
            or malformed prediction values.
    """
    if not cases:
        raise RunError("at least one case is required")
    if k <= 0:
        raise RunError("k must be a positive integer")
    requested = tuple(metrics)
    if not requested:
        raise RunError("at least one metric is required")
    if len(set(requested)) != len(requested):
        raise RunError("metrics must not contain duplicates")
    unknown = [name for name in requested if name not in SUPPORTED_METRICS]
    if unknown:
        raise RunError(
            "unknown metric(s): "
            + ", ".join(unknown)
            + "; available: "
            + ", ".join(SUPPORTED_METRICS)
        )
    if "judge" in requested and judge is None:
        raise RunError("metric 'judge' requires a judge instance")

    case_list = list(cases)
    case_ids = [case.id for case in case_list]
    if len(set(case_ids)) != len(case_ids):
        raise RunError("case ids must be unique")
    known = set(case_ids)
    for case_id, prediction in predictions.items():
        if not isinstance(case_id, str) or not isinstance(prediction, str):
            raise RunError("predictions must map strings to strings")

    missing = sorted(known - set(predictions))
    ignored = sorted(set(predictions) - known)

    results: list[CaseResult] = []
    for case in sorted(case_list, key=lambda item: item.id):
        prediction = predictions.get(case.id, "")
        missing_prediction = case.id not in predictions
        scores: dict[str, float] = {}
        judgment: Judgment | None = None
        for name in requested:
            if name == "judge":
                assert judge is not None  # checked above
                judgment = judge.judge(case, prediction)
                score = judgment.score
            else:
                metric = get_metric(name)
                if metric.family == "retrieval":
                    retrieved, relevant = _resolve_retrieval(case)
                    score = metric.fn(retrieved, relevant, k)
                else:
                    score = metric.fn(prediction, case.expected)
            score = float(score)
            if not math.isfinite(score):
                raise RunError(f"metric {name!r} returned a non-finite score")
            scores[name] = score
        results.append(
            CaseResult(
                case_id=case.id,
                prediction=prediction,
                scores=scores,
                judge=judgment,
                missing_prediction=missing_prediction,
            )
        )

    summary = {
        name: summarize([result.scores[name] for result in results])
        for name in requested
    }
    return RunResult(
        suite=suite,
        metrics=requested,
        k=k,
        cases=tuple(results),
        summary=summary,
        missing_predictions=tuple(missing),
        ignored_predictions=tuple(ignored),
    )


def load_run(path: str | Path) -> RunResult:
    """Read a run JSON file and recompute its aggregates."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RunError(f"cannot read {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RunError(f"{source} is not valid JSON: {exc}") from exc
    return RunResult.from_dict(raw)


def save_run(result: RunResult, path: str | Path) -> None:
    """Persist a run result as JSON."""
    result.save(path)
