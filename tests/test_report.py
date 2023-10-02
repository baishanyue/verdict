from __future__ import annotations

import pytest

from verdict.suites.datasets import Case
from verdict.judge.judge import HeuristicJudge
from verdict.reports.report import render_html, render_markdown, write_report
from verdict.runner import run_suite


def make_runs():
    cases = [
        Case(id="q1", input="alpha?", expected="alpha beta", contexts=("alpha beta",)),
        Case(id="q2", input="gamma?", expected="gamma delta", contexts=("gamma delta",)),
    ]
    baseline = run_suite(
        cases,
        {"q1": "alpha beta", "q2": "gamma delta"},
        metrics=("exact_match", "token_f1"),
        suite="baseline",
    )
    candidate = run_suite(
        cases,
        {"q1": "alpha beta", "q2": "<script>alert(1)</script> | wrong"},
        metrics=("exact_match", "token_f1", "judge"),
        judge=HeuristicJudge(),
        suite="candidate",
    )
    return baseline, candidate


def test_markdown_report_sections_and_escaping() -> None:
    baseline, candidate = make_runs()
    markdown = render_markdown(candidate, baseline=baseline)
    assert markdown.startswith("# Verdict report: candidate")
    assert "## Summary" in markdown
    assert "## Per-case scores" in markdown
    assert "## Judge notes" in markdown
    assert "| `exact_match` |" in markdown
    assert "Regression gate:" in markdown
    assert "FAIL" in markdown
    assert "\\|" in markdown
    assert "<script>" in markdown  # markdown is text; HTML escaping is tested below


def test_markdown_report_without_regression() -> None:
    _, candidate = make_runs()
    markdown = render_markdown(candidate, title="Custom title")
    assert markdown.startswith("# Custom title")
    assert "Baseline" not in markdown
    assert "Regression gate" not in markdown


def test_html_report_is_self_contained_and_escaped() -> None:
    baseline, candidate = make_runs()
    page = render_html(candidate, baseline=baseline)
    assert page.startswith("<!doctype html>")
    assert "#7EE7D0" in page
    assert "http://" not in page
    assert "https://" not in page
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "Judge notes" in page
    assert "EVALUATION REPORT" in page
    assert "FAIL" in page


def test_html_report_without_regression() -> None:
    _, candidate = make_runs()
    page = render_html(candidate)
    assert "Primary mean" in page
    assert "Gate" not in page
    assert "Regression" not in page


def test_write_report_infers_format(tmp_path) -> None:
    _, candidate = make_runs()
    markdown_path = write_report(candidate, tmp_path / "report.md")
    html_path = write_report(candidate, tmp_path / "report.html")
    assert markdown_path.read_text(encoding="utf-8").startswith("#")
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
    explicit = write_report(candidate, tmp_path / "custom.out", fmt="html")
    assert explicit.read_text(encoding="utf-8").startswith("<!doctype html>")
    with pytest.raises(ValueError):
        write_report(candidate, tmp_path / "report.txt")
    with pytest.raises(ValueError):
        write_report(candidate, tmp_path / "report.txt", fmt="pdf")
