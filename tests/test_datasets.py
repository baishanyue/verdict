from __future__ import annotations

import json
from pathlib import Path

import pytest

from verdict.suites.datasets import (
    Case,
    DatasetError,
    load_cases,
    load_predictions,
    sample_cases,
    split_cases,
    validate_case,
)

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_load_example_cases() -> None:
    cases = load_cases(ROOT / "examples" / "cases.jsonl")
    assert len(cases) == 10
    assert cases[0].id == "q01"
    assert cases[0].contexts
    assert cases[0].retrieved[0] == "doc-rrf"
    assert cases[0].relevant == ("doc-rrf",)
    assert all(isinstance(case, Case) for case in cases)


def test_load_predictions_covers_examples() -> None:
    cases = load_cases(ROOT / "examples" / "cases.jsonl")
    predictions = load_predictions(ROOT / "examples" / "predictions.jsonl")
    assert set(predictions) == {case.id for case in cases}
    assert predictions["q02"] == "To stop short outputs from inflating precision."


def test_validate_case_defaults_and_metadata() -> None:
    case = validate_case(
        {
            "id": "x1",
            "input": "Q?",
            "expected": "A.",
            "license": "internal",
        }
    )
    assert case.contexts == ()
    assert case.retrieved == ()
    assert case.relevant == ()
    assert case.metadata == {"license": "internal"}
    record = case.to_record()
    assert record["id"] == "x1"
    assert record["license"] == "internal"
    assert "contexts" not in record


@pytest.mark.parametrize(
    "row, message",
    [
        ({"input": "q", "expected": "a"}, "missing required field 'id'"),
        ({"id": " ", "input": "q", "expected": "a"}, "'id' must be a non-empty string"),
        ({"id": "i", "input": 3, "expected": "a"}, "'input' must be a string"),
        ({"id": "i", "input": "q", "expected": []}, "'expected' must be a string"),
        ({"id": "i", "input": "q", "expected": "a", "contexts": "x"}, "must be a list"),
        (
            {"id": "i", "input": "q", "expected": "a", "retrieved": [1]},
            r"retrieved\[0\]",
        ),
        (
            {"id": "i", "input": "q", "expected": "a", "relevant": [" "]},
            "must not be blank",
        ),
    ],
)
def test_validate_case_rejects_bad_shapes(row: dict, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        validate_case(row, line=4)


def test_load_cases_reports_line_and_duplicate(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "input": "q", "expected": "e"}\n\nnot-json\n')
    with pytest.raises(DatasetError, match=r"bad\.jsonl:3: invalid JSON"):
        load_cases(bad)

    duplicate = write_jsonl(
        tmp_path / "dup.jsonl",
        [
            {"id": "a", "input": "q", "expected": "e"},
            {"id": "a", "input": "q2", "expected": "e2"},
        ],
    )
    with pytest.raises(DatasetError, match=r"dup\.jsonl:2: duplicate case id"):
        load_cases(duplicate)


def test_load_cases_missing_file() -> None:
    with pytest.raises(DatasetError, match="cannot read"):
        load_cases("/definitely/not/here.jsonl")


def test_load_predictions_validation(tmp_path: Path) -> None:
    good = write_jsonl(
        tmp_path / "p.jsonl",
        [{"id": "a", "prediction": ""}, {"id": "b", "prediction": "answer"}],
    )
    assert load_predictions(good) == {"a": "", "b": "answer"}

    duplicate = write_jsonl(
        tmp_path / "p2.jsonl",
        [{"id": "a", "prediction": "x"}, {"id": "a", "prediction": "y"}],
    )
    with pytest.raises(DatasetError, match="duplicate prediction id"):
        load_predictions(duplicate)

    bad = write_jsonl(tmp_path / "p3.jsonl", [{"id": "a", "prediction": 1}])
    with pytest.raises(DatasetError, match="'prediction' must be a string"):
        load_predictions(bad)


def test_split_cases_is_seeded_and_complete() -> None:
    cases = load_cases(ROOT / "examples" / "cases.jsonl")
    first = split_cases(cases, ratio=0.8, seed=7)
    second = split_cases(cases, ratio=0.8, seed=7)
    assert [case.id for case in first[0]] == [case.id for case in second[0]]
    assert len(first[0]) == 8 and len(first[1]) == 2
    ids = [case.id for case in first[0] + first[1]]
    assert sorted(ids) == sorted(case.id for case in cases)
    assert split_cases(cases, ratio=0.8, seed=7) != split_cases(cases, seed=8)
    with pytest.raises(ValueError):
        split_cases(cases, ratio=1.0)


def test_sample_cases_is_seeded_and_bounded() -> None:
    cases = load_cases(ROOT / "examples" / "cases.jsonl")
    first = sample_cases(cases, 4, seed=3)
    second = sample_cases(cases, 4, seed=3)
    assert [case.id for case in first] == [case.id for case in second]
    assert len({case.id for case in first}) == 4
    repeated = sample_cases(cases, 30, seed=1, replace=True)
    assert len(repeated) == 30
    with pytest.raises(ValueError):
        sample_cases(cases, 11)
    with pytest.raises(ValueError):
        sample_cases(cases, -1)
    with pytest.raises(ValueError):
        sample_cases([], 1)
    assert sample_cases([], 0) == []
