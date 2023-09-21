"""Rubric-based answer judging with an offline default and a pluggable adapter.

Two implementations share the :class:`Judge` protocol:

* :class:`HeuristicJudge` is deterministic, runs on the standard library and is
  explicitly a **baseline, not an LLM**. It scores weighted rubric dimensions
  from token and keyword overlap.
* :class:`LLMJudge` is an adapter around any ``complete(prompt) -> str``
  callable. No network client is imported or called at import time; tests inject
  a fake callable and production code injects its own client.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..suites.datasets import Case
from ..metrics import content_tokens, normalize_text, tokenize

__all__ = [
    "DEFAULT_RUBRIC",
    "HeuristicJudge",
    "Judge",
    "JudgeError",
    "Judgment",
    "LLMJudge",
    "Rubric",
    "parse_judgment",
]


class JudgeError(ValueError):
    """Raised when a judge response cannot be interpreted."""


@dataclass(frozen=True, slots=True)
class Judgment:
    """The result of judging one prediction.

    Attributes:
        score: Normalised quality score in ``[0.0, 1.0]``.
        rationale: Short human-readable explanation of the score.
        judge: Identifier of the judge that produced it.
        criteria: Optional per-dimension component scores.
    """

    score: float
    rationale: str
    judge: str
    criteria: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "rationale": self.rationale,
            "judge": self.judge,
            "criteria": dict(self.criteria),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Judgment":
        try:
            score = float(raw["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise JudgeError("judgment is missing a numeric 'score'") from exc
        if not 0.0 <= score <= 1.0:
            raise JudgeError("judgment score must be between 0.0 and 1.0")
        criteria_raw = raw.get("criteria", {})
        if not isinstance(criteria_raw, Mapping):
            raise JudgeError("judgment 'criteria' must be an object")
        try:
            criteria = {
                str(key): float(value) for key, value in criteria_raw.items()
            }
        except (TypeError, ValueError) as exc:
            raise JudgeError("judgment criteria values must be numbers") from exc
        return cls(
            score=score,
            rationale=str(raw.get("rationale", "")),
            judge=str(raw.get("judge", "unknown")),
            criteria=criteria,
        )


@runtime_checkable
class Judge(Protocol):
    """Protocol implemented by answer judges."""

    name: str

    def judge(self, case: Case, prediction: str) -> Judgment:
        """Score ``prediction`` for ``case`` and return a :class:`Judgment`."""
        ...


@dataclass(frozen=True, slots=True)
class Rubric:
    """Weights and optional required keywords for :class:`HeuristicJudge`.

    The three dimensions are weighted and renormalised over the dimensions that
    apply (grounding is skipped when a case has no contexts). When
    ``required_keywords`` is non-empty it contributes 30 percent of the coverage
    dimension, so a fluent answer that omits a required fact is capped.
    """

    coverage_weight: float = 0.55
    grounding_weight: float = 0.25
    concision_weight: float = 0.20
    required_keywords: tuple[str, ...] = ()
    max_length_ratio: float = 2.5
    pass_threshold: float = 0.6

    def __post_init__(self) -> None:
        weights = (self.coverage_weight, self.grounding_weight, self.concision_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("rubric weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one rubric weight must be positive")
        if self.max_length_ratio <= 1.5:
            raise ValueError("max_length_ratio must be greater than 1.5")
        if not 0.0 <= self.pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be between 0.0 and 1.0")
        for keyword in self.required_keywords:
            if not isinstance(keyword, str) or not keyword.strip():
                raise ValueError("required keywords must be non-blank strings")

    def describe(self) -> str:
        """Return a compact rubric description suitable for a prompt."""
        return (
            f"coverage {self.coverage_weight:.2f}, "
            f"grounding {self.grounding_weight:.2f}, "
            f"concision {self.concision_weight:.2f}; "
            f"required keywords: {', '.join(self.required_keywords) or 'none'}"
        )


DEFAULT_RUBRIC = Rubric()


def _keyword_present(keyword: str, normalized_text: str) -> bool:
    needle = normalize_text(keyword)
    return bool(needle) and f" {needle} " in f" {normalized_text} "


class HeuristicJudge:
    """Deterministic offline rubric baseline. This is not a language model.

    Dimensions:

    * **coverage** — fraction of reference content tokens present in the
      prediction. If ``required_keywords`` are configured they supply 30 percent
      of this dimension.
    * **grounding** — fraction of prediction content tokens found in the case
      contexts or the reference; only applied when contexts exist.
    * **concision** — 1.0 up to 1.5x the reference token count, then decaying
      linearly to 0.0 at ``max_length_ratio``; 0.0 for an empty prediction.

    The weighted score is a baseline signal for smoke tests, not a replacement
    for human or model review.
    """

    name = "heuristic-baseline"
    is_llm = False

    def __init__(self, rubric: Rubric | None = None) -> None:
        self.rubric = rubric or DEFAULT_RUBRIC

    def judge(self, case: Case, prediction: str) -> Judgment:
        if not isinstance(prediction, str):
            raise TypeError("prediction must be a string")
        predicted_tokens = tokenize(prediction)
        expected_tokens = tokenize(case.expected)
        context_tokens = tokenize(" ".join(case.contexts))

        expected_terms = set(content_tokens(case.expected))
        predicted_terms = set(content_tokens(prediction))

        if not expected_terms:
            coverage = 1.0 if not predicted_terms else 0.0
        else:
            coverage = len(expected_terms & predicted_terms) / len(expected_terms)
        if self.rubric.required_keywords:
            normalized_prediction = normalize_text(prediction)
            hits = sum(
                1
                for keyword in self.rubric.required_keywords
                if _keyword_present(keyword, normalized_prediction)
            )
            keyword_score = hits / len(self.rubric.required_keywords)
            coverage = 0.7 * coverage + 0.3 * keyword_score

        components: dict[str, float] = {"coverage": coverage}
        weights: dict[str, float] = {"coverage": self.rubric.coverage_weight}

        if case.contexts:
            if not predicted_terms:
                grounding = 0.0
            else:
                supported = predicted_terms & (set(context_tokens) | expected_terms)
                grounding = len(supported) / len(predicted_terms)
            components["grounding"] = grounding
            weights["grounding"] = self.rubric.grounding_weight

        if not predicted_terms or not expected_tokens:
            concision = 0.0
        else:
            ratio = len(predicted_tokens) / len(expected_tokens)
            if ratio <= 1.5:
                concision = 1.0
            else:
                concision = max(
                    0.0,
                    1.0
                    - (ratio - 1.5) / (self.rubric.max_length_ratio - 1.5),
                )
        components["concision"] = concision
        weights["concision"] = self.rubric.concision_weight

        active_weight = sum(weights.values())
        score = sum(components[name] * weights[name] for name in components) / active_weight
        score = min(1.0, max(0.0, score))
        rationale = (
            " ".join(f"{name}={value:.2f}" for name, value in components.items())
            + f" -> {score:.2f} (deterministic heuristic baseline, not an LLM)"
        )
        return Judgment(
            score=score,
            rationale=rationale,
            judge=self.name,
            criteria={name: round(value, 6) for name, value in components.items()},
        )


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")


def parse_judgment(raw: str, *, judge_name: str = "llm") -> Judgment:
    """Parse a judge response containing a JSON object.

    Accepts optional Markdown code fences and surrounding prose around a single
    object. The object must contain a numeric ``score`` in ``[0.0, 1.0]`` and
    optionally ``rationale`` (string) and ``criteria`` (object of numbers).

    Raises:
        JudgeError: If no valid object with a valid score can be found.
    """
    if not isinstance(raw, str):
        raise JudgeError("judge response must be a string")
    text = _FENCE_RE.sub("", raw.strip())
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    payload: Any = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        break
    if not isinstance(payload, Mapping):
        raise JudgeError("judge response did not contain a JSON object")
    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise JudgeError("judge response needs a numeric 'score'")
    if not 0.0 <= float(score) <= 1.0:
        raise JudgeError("judge score must be between 0.0 and 1.0")
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        raise JudgeError("judge 'rationale' must be a string")
    criteria_raw = payload.get("criteria", {})
    if not isinstance(criteria_raw, Mapping):
        raise JudgeError("judge 'criteria' must be an object")
    criteria: dict[str, float] = {}
    for key, value in criteria_raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JudgeError("judge criteria values must be numbers")
        criteria[str(key)] = float(value)
    return Judgment(
        score=float(score),
        rationale=rationale,
        judge=judge_name,
        criteria=criteria,
    )


class LLMJudge:
    """Adapter that delegates judging to an injected completion callable.

    The interface is deliberately tiny::

        LLMJudge(lambda prompt: my_client.complete(prompt))

    The callable receives the prompt built by :meth:`build_prompt` and must
    return a string containing a JSON object with a ``score`` between 0.0 and
    1.0 and an optional ``rationale``. Network access, retries and timeouts are
    the caller's responsibility, so importing this module never opens a socket.
    """

    name = "llm"
    is_llm = True

    def __init__(
        self,
        complete: Callable[[str], str],
        *,
        model: str = "injected",
        rubric: Rubric | None = None,
    ) -> None:
        if not callable(complete):
            raise TypeError("complete must be callable")
        self._complete = complete
        self.model = model
        self.rubric = rubric or DEFAULT_RUBRIC

    def build_prompt(self, case: Case, prediction: str) -> str:
        """Build the rubric prompt sent to the completion callable."""
        contexts = "\n".join(f"- {text}" for text in case.contexts) or "- (none)"
        return (
            "You are grading one answer against a reference answer.\n"
            f"Rubric: {self.rubric.describe()}.\n"
            "Return only a JSON object of the form "
            '{"score": <float 0.0-1.0>, "rationale": "<one sentence>", '
            '"criteria": {"coverage": <float>, "grounding": <float>, '
            '"concision": <float>}}.\n\n'
            f"Question:\n{case.input}\n\n"
            f"Reference answer:\n{case.expected}\n\n"
            f"Contexts:\n{contexts}\n\n"
            f"Candidate answer:\n{prediction}\n"
        )

    def judge(self, case: Case, prediction: str) -> Judgment:
        if not isinstance(prediction, str):
            raise TypeError("prediction must be a string")
        raw = self._complete(self.build_prompt(case, prediction))
        return parse_judgment(raw, judge_name=self.name)
