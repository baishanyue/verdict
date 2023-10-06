#!/usr/bin/env python3
"""Measure Verdict metric throughput on this machine.

The benchmark builds a deterministic synthetic QA/retrieval suite, warms up
each operation, then reports the median of several timed passes. It measures
this implementation only; the numbers are indicative and not comparative
claims about other projects.

Usage::

    python benchmarks/bench.py
    python benchmarks/bench.py --n 5000 --repeat 7 --out benchmarks/latest.md
"""

from __future__ import annotations

import argparse
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verdict.suites.datasets import Case  # noqa: E402
from verdict.judge.judge import HeuristicJudge  # noqa: E402
from verdict.metrics import (  # noqa: E402
    char_bleu4,
    exact_match,
    mrr,
    ndcg_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)
from verdict.runner import run_suite  # noqa: E402

WORDS = (
    "retrieval context ranking evidence metric answer chunk recall precision "
    "vector sparse corpus latency judge calibration trace span baseline "
    "candidate tolerance split sample prompt token window overlap"
).split()


def cpu_label() -> str:
    try:
        value = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if value:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def build_suite(n: int) -> tuple[list[Case], dict[str, str]]:
    rng = random.Random(2026)
    cases: list[Case] = []
    predictions: dict[str, str] = {}

    def sentence(length: int) -> str:
        return " ".join(rng.choice(WORDS) for _ in range(length))

    for index in range(n):
        expected = sentence(16 + index % 9)
        contexts = (sentence(28), sentence(26))
        retrieved = [f"doc-{rng.randrange(400):03d}" for _ in range(10)]
        relevant = (
            [retrieved[0], f"doc-gold-{index}"]
            if index % 3 == 0
            else [retrieved[0], retrieved[1]]
        )
        style = index % 5
        if style == 0:
            prediction = expected
        elif style == 1:
            prediction = " ".join(expected.split()[:8])
        elif style == 2:
            prediction = sentence(14)
        elif style == 3:
            words = expected.split()
            words[len(words) // 2] = rng.choice(WORDS)
            prediction = " ".join(words)
        else:
            prediction = expected + " " + sentence(6)
        case = Case(
            id=f"bench-{index:05d}",
            input=f"question {index}",
            expected=expected,
            contexts=contexts,
            retrieved=tuple(retrieved),
            relevant=tuple(relevant),
        )
        cases.append(case)
        predictions[case.id] = prediction
    return cases, predictions


def median_seconds(operation, repeat: int) -> float:
    operation()  # warm-up
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def measure(n: int, repeat: int) -> tuple[list[tuple[str, float]], dict[str, float]]:
    cases, predictions = build_suite(n)
    judge = HeuristicJudge()

    def each_metric(function) -> None:
        for case in cases:
            function(case, predictions[case.id])

    operations = {
        "exact_match": lambda: each_metric(
            lambda case, prediction: exact_match(prediction, case.expected)
        ),
        "token_f1": lambda: each_metric(
            lambda case, prediction: token_f1(prediction, case.expected)
        ),
        "rouge_l": lambda: each_metric(
            lambda case, prediction: rouge_l(prediction, case.expected)
        ),
        "char_bleu4": lambda: each_metric(
            lambda case, prediction: char_bleu4(prediction, case.expected)
        ),
        "recall_at_5": lambda: each_metric(
            lambda case, _prediction: recall_at_k(case.retrieved, case.relevant, 5)
        ),
        "mrr": lambda: each_metric(
            lambda case, _prediction: mrr(case.retrieved, case.relevant, 5)
        ),
        "ndcg_at_5": lambda: each_metric(
            lambda case, _prediction: ndcg_at_k(case.retrieved, case.relevant, 5)
        ),
        "heuristic_judge": lambda: each_metric(
            lambda case, prediction: judge.judge(case, prediction)
        ),
        "full_suite": lambda: run_suite(
            cases,
            predictions,
            metrics=(
                "exact_match",
                "token_f1",
                "rouge_l",
                "char_bleu4",
                "recall_at_k",
                "mrr",
                "ndcg_at_k",
                "judge",
            ),
            k=5,
            judge=judge,
            suite="bench",
        ),
    }

    rows: list[tuple[str, float]] = []
    for name, operation in operations.items():
        rows.append((name, median_seconds(operation, repeat)))
    return rows, {"n": n, "repeat": repeat}


def render_markdown(rows: list[tuple[str, float]], n: int, repeat: int) -> str:
    lines = [
        "| Operation | Median total | Per case | Throughput |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, seconds in rows:
        per_case = seconds / n * 1_000_000.0
        throughput = n / seconds if seconds else 0.0
        lines.append(
            f"| `{name}` | {seconds * 1000:.1f} ms | {per_case:.2f} us | "
            f"{throughput:,.0f} cases/s |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure Verdict metric throughput (indicative, not comparative).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The suite is generated deterministically from seed 2026. "
            "Numbers vary with CPU load, Python build and input lengths."
        ),
    )
    parser.add_argument("--n", type=int, default=2000, help="cases (default 2000)")
    parser.add_argument(
        "--repeat", type=int, default=5, help="timed passes; median is reported"
    )
    parser.add_argument("--out", type=Path, default=None, help="optional Markdown output")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")

    rows, meta = measure(args.n, args.repeat)
    markdown = render_markdown(rows, args.n, args.repeat)
    header = (
        f"# Verdict metric benchmark\n\n"
        f"- Cases: {args.n:,}\n"
        f"- Timed passes: {args.repeat} (median reported)\n"
        f"- CPU: {cpu_label()} / {platform.machine()}\n"
        f"- Platform: {platform.platform()}\n"
        f"- Python: {platform.python_version()}\n"
        f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"- Note: indicative measurements for this machine, not comparative claims.\n"
    )
    output = header + "\n" + markdown + "\n"
    if args.json:
        import json

        payload = {
            **meta,
            "cpu": cpu_label(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rows": [
                {
                    "operation": name,
                    "median_seconds": seconds,
                    "microseconds_per_case": seconds / args.n * 1_000_000.0,
                    "cases_per_second": args.n / seconds if seconds else 0.0,
                }
                for name, seconds in rows
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(output)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"wrote: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
