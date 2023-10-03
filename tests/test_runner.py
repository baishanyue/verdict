from __future__ import annotations

import json
import math

import pytest

from verdict.suites.datasets import Case
from verdict.judge.judge import Judgment
from verdict.runner import (
    RunError,
    RunResult,
    load_run,
    percentile,
    run_suite,
    summarize,
)


class CountingJudge:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, case: Case, prediction: str) -> Judgment:
        self.calls += 1
        return Judgment(
            score=0.5,
            rationale=f"scored {case.id}",
            judge=self.name,
            criteria={"fixed": 0.5},
        )


def make_cases() -> list[Case]:
    return [
        Case(
            id="b",
            input="beta?",
            expected="alpha beta",
            contexts=("alpha beta",),
            retrieved=("d1", "d2"),
            relevant=("d1",),
        ),
        Case(id="a", input="alpha?", expected="alpha", contexts=("alpha",)),
    ]


def test_run_suite_aggregates_and_sorts() -> None:
    result = run_suite(
        make_cases(),
        {"a": "alpha", "b": "alpha"},
        metrics=("exact_match", "token_f1", "rouge_l", "char_bleu4"),
        suite="demo",
    )
    assert [case.case_id for case in result.cases] == ["a", "b"]
    assert result.summary["exact_match"].mean == pytest.approx(0.5)
    assert result.summary["exact_match"].median == pytest.approx(0.5)
    assert result.summary["exact_match"].p95 == pytest.approx(0.95)
    assert result.summary["token_f1"].mean == pytest.approx((1.0 + 2 / 3) / 2)
    assert result.summary["token_f1"].count == 2
    assert result.suite == "demo"
    assert result.missing_predictions == ()
    assert result.ignored_predictions == ()


def test_run_suite_is_order_independent() -> None:
    cases = make_cases()
    first = run_suite(cases, {"a": "alpha", "b": "alpha"}, metrics=("exact_match",))
    second = run_suite(
        list(reversed(cases)),
        {"b": "alpha", "a": "alpha"},
        metrics=("exact_match",),
    )
    assert first.to_json() == second.to_json()


def test_run_suite_judge_is_called_once_per_case() -> None:
    judge = CountingJudge()
    result = run_suite(
        make_cases(),
        {"a": "alpha", "b": "beta"},
        metrics=("exact_match", "judge"),
        judge=judge,
    )
    assert judge.calls == 2
    assert all(case.judge is not None for case in result.cases)
    assert result.summary["judge"].mean == 0.5
    assert result.cases[0].judge is not None
    assert result.cases[0].judge.rationale == "scored a"


def test_run_suite_handles_missing_and_ignored_predictions() -> None:
    result = run_suite(
        make_cases(),
        {"a": "alpha", "z": "extra"},
        metrics=("exact_match", "token_f1"),
    )
    assert result.missing_predictions == ("b",)
    assert result.ignored_predictions == ("z",)
    missing_case = next(case for case in result.cases if case.case_id == "b")
    assert missing_case.missing_prediction is True
    assert missing_case.prediction == ""
    assert missing_case.scores["exact_match"] == 0.0


def test_run_suite_retrieval_metrics() -> None:
    cases = [
        Case(
            id="r1",
            input="q",
            expected="a",
            retrieved=("d1", "d2", "d3"),
            relevant=("d1", "d3"),
        ),
        Case(
            id="r2",
            input="q",
            expected="a",
            contexts=("d9",),
            retrieved=("d9",),
        ),
        Case(id="r3", input="q", expected="a"),
    ]
    result = run_suite(
        cases,
        {"r1": "a", "r2": "a", "r3": "a"},
        metrics=("recall_at_k", "mrr", "ndcg_at_k"),
        k=5,
    )
    assert result.summary["recall_at_k"].mean == pytest.approx(2 / 3)
    assert result.summary["mrr"].mean == pytest.approx(2 / 3)
    assert result.summary["ndcg_at_k"].mean == pytest.approx(
        (1.5 / (1.0 + 1.0 / math.log2(3)) + 1.0) / 3
    )
    assert result.cases[2].scores["recall_at_k"] == 0.0


def test_run_suite_validation_errors() -> None:
    cases = make_cases()
    with pytest.raises(RunError, match="at least one case"):
        run_suite([], {})
    with pytest.raises(RunError, match="unknown metric"):
        run_suite(cases, {}, metrics=("nope",))
    with pytest.raises(RunError, match="duplicates"):
        run_suite(cases, {}, metrics=("exact_match", "exact_match"))
    with pytest.raises(RunError, match="at least one metric"):
        run_suite(cases, {}, metrics=())
    with pytest.raises(RunError, match="requires a judge"):
        run_suite(cases, {}, metrics=("judge",))
    with pytest.raises(RunError, match="k must be"):
        run_suite(cases, {}, metrics=("exact_match",), k=0)
    with pytest.raises(RunError, match="unique"):
        run_suite([cases[0], cases[0]], {}, metrics=("exact_match",))


def test_run_result_roundtrip_and_summary_recompute(tmp_path) -> None:
    result = run_suite(
        make_cases(),
        {"a": "alpha", "b": "alpha"},
        metrics=("exact_match", "token_f1"),
        suite="roundtrip",
    )
    raw = result.to_dict()
    raw["summary"]["exact_match"]["mean"] = 99.0  # must be ignored on load
    restored = RunResult.from_dict(raw)
    assert restored.summary["exact_match"].mean == pytest.approx(0.5)
    assert restored.to_json() == result.to_json()

    path = tmp_path / "run.json"
    result.save(path)
    restored_file = load_run(path)
    assert restored_file.suite == "roundtrip"
    assert restored_file.cases == result.cases


def test_run_result_from_dict_rejects_malformed() -> None:
    good = run_suite(make_cases(), {"a": "a", "b": "b"}, metrics=("exact_match",))
    payload = good.to_dict()
    with pytest.raises(RunError):
        RunResult.from_dict({})
    bad = json.loads(json.dumps(payload))
    bad["metrics"] = []
    with pytest.raises(RunError):
        RunResult.from_dict(bad)
    bad = json.loads(json.dumps(payload))
    bad["cases"][0]["scores"] = {}
    with pytest.raises(RunError, match="missing scores"):
        RunResult.from_dict(bad)
    bad = json.loads(json.dumps(payload))
    bad["cases"][1]["id"] = bad["cases"][0]["id"]
    with pytest.raises(RunError, match="duplicate case id"):
        RunResult.from_dict(bad)
    bad = json.loads(json.dumps(payload))
    bad["cases"][0]["scores"]["exact_match"] = math.inf
    with pytest.raises(RunError, match="finite"):
        RunResult.from_dict(bad)


def test_percentile_and_summarize() -> None:
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
    assert percentile([3.0], 0.5) == 3.0
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)
    summary = summarize([0.0, 0.5, 1.0])
    assert summary.count == 3
    assert summary.median == 0.5
    assert summary.minimum == 0.0
    assert summary.maximum == 1.0
    with pytest.raises(ValueError):
        summarize([])
