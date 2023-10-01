from __future__ import annotations

import pytest

from verdict.regression import (
    MetricComparison,
    RegressionReport,
    Tolerance,
    compare_runs,
)
from verdict.runner import CaseResult, RunResult, summarize


def make_run(suite: str, means: dict[str, float]) -> RunResult:
    result = CaseResult(case_id="c1", prediction="", scores=dict(means))
    summary = {
        name: summarize([value]) for name, value in means.items()
    }
    return RunResult(
        suite=suite,
        metrics=tuple(means),
        k=5,
        cases=(result,),
        summary=summary,
    )


def test_identical_runs_pass() -> None:
    baseline = make_run("baseline", {"exact_match": 0.7, "token_f1": 0.8})
    report = compare_runs(baseline, baseline)
    assert report.ok is True
    assert report.exit_code == 0
    assert report.failed == ()
    assert {row.metric for row in report.rows} == {"exact_match", "token_f1"}


def test_tolerance_is_max_of_absolute_and_relative() -> None:
    assert Tolerance(0.01, 0.02).allowed_drop(0.5) == pytest.approx(0.01)
    assert Tolerance(0.01, 0.02).allowed_drop(1.0) == pytest.approx(0.02)
    baseline = make_run("baseline", {"exact_match": 0.5})
    within = make_run("candidate", {"exact_match": 0.495})
    assert compare_runs(within, baseline).ok is True
    beyond = make_run("candidate", {"exact_match": 0.48})
    report = compare_runs(beyond, baseline)
    assert report.ok is False
    assert report.exit_code == 1
    row = report.rows[0]
    assert row.delta == pytest.approx(-0.02)
    assert row.allowed_drop == pytest.approx(0.01)
    assert row.relative_delta == pytest.approx(-0.04)
    assert "exceeds" in row.note


def test_relative_tolerance_scales_for_large_baselines() -> None:
    baseline = make_run("baseline", {"token_f1": 1.0})
    candidate = make_run("candidate", {"token_f1": 0.985})
    assert compare_runs(candidate, baseline).ok is True


def test_per_metric_override_and_unknown_override() -> None:
    baseline = make_run("baseline", {"token_f1": 0.5})
    candidate = make_run("candidate", {"token_f1": 0.499})
    strict = compare_runs(
        candidate,
        baseline,
        overrides={"token_f1": Tolerance(0.0, 0.0)},
    )
    assert strict.ok is False
    with pytest.raises(ValueError, match="unknown metric"):
        compare_runs(
            candidate,
            baseline,
            overrides={"nope": Tolerance()},
        )


def test_missing_and_new_metrics() -> None:
    baseline = make_run("baseline", {"exact_match": 0.7, "rouge_l": 0.6})
    candidate = make_run("candidate", {"exact_match": 0.7, "judge": 0.9})
    report = compare_runs(candidate, baseline)
    rows = {row.metric: row for row in report.rows}
    assert rows["rouge_l"].passed is False
    assert "missing" in rows["rouge_l"].note
    assert rows["judge"].passed is True
    assert "new metric" in rows["judge"].note
    assert report.failed == ("rouge_l",)
    assert report.ok is False


def test_compare_accepts_mappings() -> None:
    baseline = make_run("baseline", {"exact_match": 0.7})
    candidate = make_run("candidate", {"exact_match": 0.7})
    report = compare_runs(candidate.to_dict(), baseline.to_dict())
    assert report.ok is True
    assert report.candidate_suite == "candidate"
    with pytest.raises(ValueError, match="invalid candidate run"):
        compare_runs({"cases": []}, baseline)
    with pytest.raises(TypeError):
        compare_runs(3, baseline)  # type: ignore[arg-type]


def test_tolerance_validation() -> None:
    with pytest.raises(ValueError):
        Tolerance(absolute=-0.1)
    with pytest.raises(ValueError):
        Tolerance(relative=float("inf"))


def test_report_serialisation_and_markdown() -> None:
    baseline = make_run("baseline", {"exact_match": 0.8})
    candidate = make_run("candidate", {"exact_match": 0.5})
    report = compare_runs(candidate, baseline)
    payload = report.to_dict()
    assert payload["ok"] is False
    assert payload["exit_code"] == 1
    assert payload["failed"] == ["exact_match"]
    assert payload["rows"][0]["delta"] == pytest.approx(-0.3)
    markdown = report.render_markdown()
    assert "Regression gate: FAIL" in markdown
    assert "| `exact_match` |" in markdown
    assert "Failed metrics" in markdown


def test_metric_comparison_to_dict() -> None:
    row = MetricComparison(
        metric="x",
        baseline_mean=1.0,
        candidate_mean=0.5,
        delta=-0.5,
        allowed_drop=0.02,
        passed=False,
        note="drop exceeds tolerance",
    )
    payload = row.to_dict()
    assert payload["metric"] == "x"
    assert payload["passed"] is False
    assert payload["relative_delta"] is None
    assert isinstance(RegressionReport, type)
