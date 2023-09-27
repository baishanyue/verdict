from __future__ import annotations

import json
from pathlib import Path

import pytest

from verdict.cli import main
from verdict.suites.datasets import load_cases
from verdict.runner import load_run

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples" / "cases.jsonl"
PREDICTIONS = ROOT / "examples" / "predictions.jsonl"


def run_cli(tmp_path: Path, name: str = "run.json", *extra: str) -> Path:
    out = tmp_path / name
    code = main(
        [
            "run",
            "--cases",
            str(CASES),
            "--predictions",
            str(PREDICTIONS),
            "--suite",
            "cli-test",
            "--out",
            str(out),
            *extra,
        ]
    )
    assert code == 0
    return out


def write_empty_predictions(path: Path) -> Path:
    cases = load_cases(CASES)
    path.write_text(
        "\n".join(
            json.dumps({"id": case.id, "prediction": ""}) for case in cases
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_run_command_writes_artifacts(tmp_path, capsys) -> None:
    out = run_cli(tmp_path)
    captured = capsys.readouterr()
    assert "suite: cli-test" in captured.out
    assert "wrote:" in captured.out
    result = load_run(out)
    assert result.suite == "cli-test"
    assert len(result.cases) == 10
    assert "judge" in result.metrics
    assert result.summary["judge"].mean > 0.5


def test_run_json_flag(tmp_path, capsys) -> None:
    out = run_cli(tmp_path, "run.json", "--json")
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["suite"] == "cli-test"
    assert payload["summary"]["exact_match"]["count"] == 10
    assert out.exists()


def test_run_without_judge(tmp_path) -> None:
    out = tmp_path / "run.json"
    code = main(
        [
            "run",
            "--cases",
            str(CASES),
            "--predictions",
            str(PREDICTIONS),
            "--metrics",
            "exact_match",
            "--judge",
            "none",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert load_run(out).metrics == ("exact_match",)


def test_run_warns_on_missing_predictions(tmp_path, capsys) -> None:
    cases = load_cases(CASES)
    partial = tmp_path / "partial.jsonl"
    partial.write_text(
        json.dumps({"id": cases[0].id, "prediction": cases[0].expected}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "run.json"
    code = main(
        [
            "run",
            "--cases",
            str(CASES),
            "--predictions",
            str(partial),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert "warning: missing predictions" in capsys.readouterr().err
    result = load_run(out)
    assert len(result.missing_predictions) == 9


def test_compare_command_pass_and_fail(tmp_path, capsys) -> None:
    baseline = run_cli(tmp_path, "baseline.json", "--judge", "none")
    candidate = tmp_path / "candidate.json"
    worse = write_empty_predictions(tmp_path / "empty.jsonl")
    code = main(
        [
            "run",
            "--cases",
            str(CASES),
            "--predictions",
            str(worse),
            "--judge",
            "none",
            "--out",
            str(candidate),
        ]
    )
    assert code == 0
    capsys.readouterr()

    same = main(["compare", "--baseline", str(baseline), "--candidate", str(baseline)])
    assert same == 0
    assert "PASS" in capsys.readouterr().out

    failed = main(
        ["compare", "--baseline", str(baseline), "--candidate", str(candidate)]
    )
    assert failed == 1
    output = capsys.readouterr().out
    assert "FAIL" in output
    assert "exact_match" in output


def test_compare_json_output(tmp_path, capsys) -> None:
    baseline = run_cli(tmp_path, "baseline.json", "--judge", "none")
    capsys.readouterr()
    code = main(
        [
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(baseline),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["exit_code"] == 0


def test_report_command(tmp_path, capsys) -> None:
    baseline = run_cli(tmp_path, "baseline.json", "--judge", "none")
    worse = write_empty_predictions(tmp_path / "empty.jsonl")
    candidate = tmp_path / "candidate.json"
    assert (
        main(
            [
                "run",
                "--cases",
                str(CASES),
                "--predictions",
                str(worse),
                "--judge",
                "none",
                "--out",
                str(candidate),
            ]
        )
        == 0
    )
    capsys.readouterr()
    html = tmp_path / "report.html"
    code = main(
        [
            "report",
            "--run",
            str(candidate),
            "--baseline",
            str(baseline),
            "--out",
            str(html),
        ]
    )
    assert code == 0
    page = html.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert "FAIL" in page
    assert "wrote:" in capsys.readouterr().out

    markdown = tmp_path / "report.md"
    assert (
        main(["report", "--run", str(candidate), "--out", str(markdown)]) == 0
    )
    assert markdown.read_text(encoding="utf-8").startswith("# Verdict report")


def test_help_without_arguments(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "run" in output and "compare" in output and "report" in output


def test_no_command_prints_help(capsys) -> None:
    assert main([]) == 0
    assert "usage: verdict" in capsys.readouterr().out


def test_subcommand_help() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--help"])
    assert excinfo.value.code == 0


def test_version(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "0.2.0" in capsys.readouterr().out


def test_invalid_inputs_exit_2(tmp_path, capsys) -> None:
    code = main(["run", "--cases", str(tmp_path / "missing.jsonl")])
    assert code == 2
    assert "error:" in capsys.readouterr().err

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    code = main(["compare", "--baseline", str(bad), "--candidate", str(bad)])
    assert code == 2
    assert "error:" in capsys.readouterr().err
