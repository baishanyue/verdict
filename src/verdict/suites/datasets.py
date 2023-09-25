"""JSONL dataset loading, validation, splitting and deterministic sampling.

A *case* is one evaluation item. The canonical record is::

    {"id": "q01",
     "input": "What does RRF combine?",
     "expected": "Ranked lists from multiple retrievers.",
     "contexts": ["..."],          # optional, gold evidence
     "retrieved": ["doc-3"],       # optional, ranked retrieval output
     "relevant": ["doc-3"]}        # optional, gold ids for retrieval metrics

``relevant`` is an additive field used by the retrieval metrics; when it is
absent the runner falls back to comparing ``retrieved`` against ``contexts``.
Any additional keys are preserved in :attr:`Case.metadata` instead of being
rejected, so a dataset can carry annotations the runner does not interpret.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Case",
    "DatasetError",
    "load_cases",
    "load_predictions",
    "sample_cases",
    "split_cases",
    "validate_case",
]

_REQUIRED = ("id", "input", "expected")
_OPTIONAL = ("contexts", "retrieved", "relevant")


class DatasetError(ValueError):
    """Raised when a JSONL dataset is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class Case:
    """One evaluation case.

    Attributes:
        id: Unique, non-empty identifier used to join predictions to cases.
        input: The question or instruction given to the system under test.
        expected: Reference answer used by the text metrics and the judge.
        contexts: Gold evidence passages, if the task is retrieval grounded.
        retrieved: Ranked identifiers produced by a retriever, most relevant first.
        relevant: Gold identifiers used by Recall@k, MRR and nDCG@k. When empty,
            the runner uses ``contexts`` as the relevance set.
        metadata: Any extra keys from the JSONL record, kept unmodified.
    """

    id: str
    input: str
    expected: str
    contexts: tuple[str, ...] = ()
    retrieved: tuple[str, ...] = ()
    relevant: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serialisable record close to the input format."""
        record: dict[str, Any] = {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
        }
        if self.contexts:
            record["contexts"] = list(self.contexts)
        if self.retrieved:
            record["retrieved"] = list(self.retrieved)
        if self.relevant:
            record["relevant"] = list(self.relevant)
        for key, value in self.metadata.items():
            record.setdefault(key, value)
        return record


def _fail(message: str, *, path: Path | None = None, line: int | None = None) -> DatasetError:
    if path is not None:
        where = str(path)
        if line is not None:
            where = f"{where}:{line}"
        return DatasetError(f"{where}: {message}")
    if line is not None:
        return DatasetError(f"line {line}: {message}")
    return DatasetError(message)


def _text_list(value: Any, field_name: str, *, line: int | None) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _fail(f"'{field_name}' must be a list of strings", line=line)
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise _fail(
                f"'{field_name}[{index}]' must be a string",
                line=line,
            )
        if not item.strip():
            raise _fail(
                f"'{field_name}[{index}]' must not be blank",
                line=line,
            )
        items.append(item)
    return tuple(items)


def validate_case(raw: Any, *, line: int | None = None) -> Case:
    """Validate one decoded JSON record and return a :class:`Case`.

    Required keys are ``id``, ``input`` and ``expected``. ``id`` must be a
    non-empty string; answer fields must be strings (an empty ``expected`` is
    allowed so abstention cases can be represented). Optional list fields must
    contain non-blank strings. Extra keys become :attr:`Case.metadata`.

    Raises:
        DatasetError: If the record has the wrong shape or types.
    """
    if not isinstance(raw, Mapping):
        raise _fail("each line must be a JSON object", line=line)
    for key in _REQUIRED:
        if key not in raw:
            raise _fail(f"missing required field '{key}'", line=line)

    case_id = raw["id"]
    if not isinstance(case_id, str) or not case_id.strip():
        raise _fail("'id' must be a non-empty string", line=line)
    case_input = raw["input"]
    if not isinstance(case_input, str):
        raise _fail("'input' must be a string", line=line)
    expected = raw["expected"]
    if not isinstance(expected, str):
        raise _fail("'expected' must be a string", line=line)

    contexts = _text_list(raw.get("contexts", []), "contexts", line=line)
    retrieved = _text_list(raw.get("retrieved", []), "retrieved", line=line)
    relevant = _text_list(raw.get("relevant", []), "relevant", line=line)

    metadata = {
        key: value
        for key, value in raw.items()
        if key not in _REQUIRED and key not in _OPTIONAL
    }
    return Case(
        id=case_id,
        input=case_input,
        expected=expected,
        contexts=contexts,
        retrieved=retrieved,
        relevant=relevant,
        metadata=metadata,
    )


def load_cases(path: str | Path) -> list[Case]:
    """Load and validate a UTF-8 JSONL case file.

    Blank lines are skipped. Case ids must be unique within the file; duplicate
    ids would make the prediction join ambiguous. Input order is preserved, so
    splitting and sampling are reproducible for a fixed file.
    """
    source = Path(path)
    cases: list[Case] = []
    seen: set[str] = set()
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise _fail(
                        f"invalid JSON ({exc.msg})",
                        path=source,
                        line=line_number,
                    ) from exc
                try:
                    case = validate_case(raw, line=line_number)
                except DatasetError as exc:
                    raise _fail(str(exc), path=source) from exc
                if case.id in seen:
                    raise _fail(
                        f"duplicate case id {case.id!r}",
                        path=source,
                        line=line_number,
                    )
                seen.add(case.id)
                cases.append(case)
    except OSError as exc:
        raise DatasetError(f"cannot read {source}: {exc}") from exc
    return cases


def load_predictions(path: str | Path) -> dict[str, str]:
    """Load a UTF-8 JSONL prediction file as ``{case_id: prediction}``.

    Each non-blank line must be an object with string fields ``id`` and
    ``prediction``. Duplicate ids are rejected. Insertion order follows the
    file; consumers should not rely on it for report ordering.
    """
    source = Path(path)
    predictions: dict[str, str] = {}
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise _fail(
                        f"invalid JSON ({exc.msg})",
                        path=source,
                        line=line_number,
                    ) from exc
                if not isinstance(raw, Mapping):
                    raise _fail(
                        "each line must be a JSON object",
                        path=source,
                        line=line_number,
                    )
                case_id = raw.get("id")
                prediction = raw.get("prediction")
                if not isinstance(case_id, str) or not case_id.strip():
                    raise _fail(
                        "'id' must be a non-empty string",
                        path=source,
                        line=line_number,
                    )
                if not isinstance(prediction, str):
                    raise _fail(
                        "'prediction' must be a string",
                        path=source,
                        line=line_number,
                    )
                if case_id in predictions:
                    raise _fail(
                        f"duplicate prediction id {case_id!r}",
                        path=source,
                        line=line_number,
                    )
                predictions[case_id] = prediction
    except OSError as exc:
        raise DatasetError(f"cannot read {source}: {exc}") from exc
    return predictions


def split_cases(
    cases: Sequence[Case],
    *,
    ratio: float = 0.8,
    seed: int = 0,
) -> tuple[list[Case], list[Case]]:
    """Shuffle and split cases into ``(first, second)`` partitions.

    The shuffle uses :class:`random.Random` with the supplied seed, so the same
    input sequence and seed always produce the same partitions. The first
    partition receives ``int(len(cases) * ratio)`` items. For an 80/20 split of
    ten cases that is eight and two.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be strictly between 0 and 1")
    items = list(cases)
    random.Random(seed).shuffle(items)
    cut = int(len(items) * ratio)
    return items[:cut], items[cut:]


def sample_cases(
    cases: Sequence[Case],
    n: int,
    *,
    seed: int = 0,
    replace: bool = False,
) -> list[Case]:
    """Draw ``n`` cases with a seeded RNG.

    Without replacement the result is an ordered sample of distinct cases and
    ``n`` may not exceed ``len(cases)``. With ``replace=True`` cases can repeat.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    items = list(cases)
    if not replace and n > len(items):
        raise ValueError(
            f"cannot sample {n} cases without replacement from {len(items)}"
        )
    if not items:
        if n:
            raise ValueError("cannot sample from an empty case list")
        return []
    rng = random.Random(seed)
    if replace:
        return [items[rng.randrange(len(items))] for _ in range(n)]
    return rng.sample(items, n)
