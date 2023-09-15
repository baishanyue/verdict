"""Command line interface for Verdict."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .suites.datasets import DatasetError, load_cases, load_predictions
from .judge.judge import HeuristicJudge, JudgeError
from .regression import Tolerance, compare_runs
from .reports.report import write_report
from .runner import RunError, load_run, run_suite

__all__ = ["build_parser", "main"]

_EXAMPLES = """examples:
  verdict run --cases examples/cases.jsonl --predictions examples/predictions.jsonl
  verdict compare --baseline baseline.json --candidate run.json
  verdict report --run run.json --baseline baseline.json --out report.html
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the ``verdict`` command."""
    parser = argparse.ArgumentParser(
        prog="verdict",
        description=(
            "Evaluation and observability for LLM and RAG systems. "
            "Deterministic metrics, offline by default."
        ),
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{run,compare,report}")

    run_parser = subparsers.add_parser(
        "run",
        help="evaluate predictions over a case file",
        description="Evaluate a prediction file against a JSONL case file.",
    )
    run_parser.add_argument(
        "--cases", "-c", default="examples/cases.jsonl", help="JSONL case file"
    )
    run_parser.add_argument(
        "--predictions",
        "-p",
        default="examples/predictions.jsonl",
        help="JSONL prediction file",
    )
    run_parser.add_argument(
        "--metrics",
        "-m",
        default="exact_match,token_f1,rouge_l,char_bleu4",
        help=(
            "comma-separated metrics: exact_match, token_f1, rouge_l, "
            "char_bleu4, recall_at_k, mrr, ndcg_at_k, judge"
        ),
    )
    run_parser.add_argument(
        "--judge",
        choices=("heuristic", "none"),
        default="heuristic",
        help="heuristic appends the 'judge' metric; none leaves it out",
    )
    run_parser.add_argument(
        "--k", type=int, default=5, help="retrieval cutoff for retrieval metrics"
    )
    run_parser.add_argument(
        "--suite", default=None, help="suite label (default: case file stem)"
    )
    run_parser.add_argument(
        "--out",
        "-o",
        default="artifacts/run.json",
        help="output run JSON path",
    )
    run_parser.add_argument(
        "--json", action="store_true", help="print the full run JSON to stdout"
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="gate a candidate run against a baseline",
        description=(
            "Compare per-metric means and fail when a drop exceeds the "
            "absolute or relative tolerance."
        ),
    )
    compare_parser.add_argument(
        "--baseline", "-b", required=True, help="baseline run JSON"
    )
    compare_parser.add_argument(
        "--candidate", "-c", required=True, help="candidate run JSON"
    )
    compare_parser.add_argument(
        "--abs-tol", type=float, default=0.01, help="absolute tolerance (default 0.01)"
    )
    compare_parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.02,
        help="relative tolerance as a fraction (default 0.02)",
    )
    compare_parser.add_argument(
        "--json", action="store_true", help="print the gate report as JSON"
    )

    report_parser = subparsers.add_parser(
        "report",
        help="render a Markdown or HTML report",
        description="Render a Markdown or single-file HTML report for a run.",
    )
    report_parser.add_argument("--run", "-r", required=True, help="run JSON")
    report_parser.add_argument(
        "--baseline", "-b", default=None, help="optional baseline run JSON"
    )
    report_parser.add_argument(
        "--out",
        "-o",
        default="report.md",
        help="output path; .md or .html selects the format",
    )
    report_parser.add_argument(
        "--format",
        dest="fmt",
        choices=("md", "html"),
        default=None,
        help="override the format inferred from the output suffix",
    )
    report_parser.add_argument("--title", default=None, help="report title")
    report_parser.add_argument(
        "--abs-tol", type=float, default=0.01, help="absolute tolerance (default 0.01)"
    )
    report_parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.02,
        help="relative tolerance as a fraction (default 0.02)",
    )
    return parser


def _parse_metrics(value: str) -> tuple[str, ...]:
    metrics = tuple(item.strip() for item in value.split(",") if item.strip())
    if not metrics:
        raise ValueError("at least one metric is required")
    return metrics


def _run_command(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    predictions = load_predictions(args.predictions)
    metrics = _parse_metrics(args.metrics)
    judge = HeuristicJudge() if args.judge == "heuristic" else None
    if judge is not None and "judge" not in metrics:
        metrics = metrics + ("judge",)
    suite = args.suite or Path(args.cases).stem
    result = run_suite(
        cases,
        predictions,
        metrics=metrics,
        k=args.k,
        judge=judge,
        suite=suite,
    )
    result.save(args.out)
    if result.missing_predictions:
        print(
            "warning: missing predictions for "
            + ", ".join(result.missing_predictions),
            file=sys.stderr,
        )
    if result.ignored_predictions:
        print(
            "warning: ignored predictions for "
            + ", ".join(result.ignored_predictions),
            file=sys.stderr,
        )
    if args.json:
        print(result.to_json(), end="")
    else:
        print(f"suite: {result.suite}")
        print(f"cases: {len(result.cases)}")
        for name in result.metrics:
            summary = result.summary[name]
            print(
                f"  {name:<14} mean={summary.mean:.4f}  "
                f"median={summary.median:.4f}  p95={summary.p95:.4f}"
            )
        print(f"wrote: {Path(args.out)}")
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    tolerance = Tolerance(absolute=args.abs_tol, relative=args.rel_tol)
    baseline = load_run(args.baseline)
    candidate = load_run(args.candidate)
    report = compare_runs(candidate, baseline, tolerance=tolerance)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render_markdown(), end="")
    return report.exit_code


def _report_command(args: argparse.Namespace) -> int:
    run = load_run(args.run)
    baseline = load_run(args.baseline) if args.baseline else None
    regression = None
    if baseline is not None:
        tolerance = Tolerance(absolute=args.abs_tol, relative=args.rel_tol)
        regression = compare_runs(run, baseline, tolerance=tolerance)
    path = write_report(
        run,
        args.out,
        fmt=args.fmt,
        baseline=baseline,
        regression=regression,
        title=args.title,
    )
    print(f"wrote: {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    ``0`` means success, ``1`` means a failed regression gate, and ``2`` means
    invalid input or an evaluation error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "compare":
            return _compare_command(args)
        if args.command == "report":
            return _report_command(args)
    except (DatasetError, RunError, JudgeError, OSError, ValueError) as exc:
        print(f"verdict: error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")
    return 2  # pragma: no cover - argparse exits first


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
