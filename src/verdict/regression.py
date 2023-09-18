"""Regression gating between a baseline run and a candidate run.

Scores are assumed to be *higher is better* (accuracy, F1, nDCG, judge score).
A candidate may drop by ``max(absolute, relative * |baseline|)`` before the gate
fails; the absolute floor keeps small baselines from failing on noise while the
relative term scales with strong baselines.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .runner import RunResult

__all__ = [
    "DEFAULT_TOLERANCE",
    "MetricComparison",
    "RegressionReport",
    "Tolerance",
    "compare_runs",
]


@dataclass(frozen=True, slots=True)
class Tolerance:
    """Allowed mean-score drop, combining an absolute and a relative term.

    A metric passes when ``candidate_mean - baseline_mean >=
    -max(absolute, relative * |baseline_mean|)``.
    """

    absolute: float = 0.01
    relative: float = 0.02

    def __post_init__(self) -> None:
        for name, value in (("absolute", self.absolute), ("relative", self.relative)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"tolerance {name} must be a finite non-negative number")

    def allowed_drop(self, baseline_mean: float) -> float:
        """Return the permitted drop for a given baseline score."""
        return max(self.absolute, self.relative * abs(baseline_mean))


DEFAULT_TOLERANCE = Tolerance()


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """One metric's baseline/candidate comparison and gate decision."""

    metric: str
    baseline_mean: float | None
    candidate_mean: float | None
    delta: float | None
    allowed_drop: float | None
    passed: bool
    note: str = ""
    relative_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "delta": self.delta,
            "relative_delta": self.relative_delta,
            "allowed_drop": self.allowed_drop,
            "passed": self.passed,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class RegressionReport:
    """Result of a regression gate over every metric in either run."""

    baseline_suite: str
    candidate_suite: str
    tolerance: Tolerance
    rows: tuple[MetricComparison, ...]

    @property
    def passed(self) -> tuple[str, ...]:
        return tuple(row.metric for row in self.rows if row.passed)

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(row.metric for row in self.rows if not row.passed)

    @property
    def ok(self) -> bool:
        return all(row.passed for row in self.rows)

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "baseline_suite": self.baseline_suite,
            "candidate_suite": self.candidate_suite,
            "tolerance": {
                "absolute": self.tolerance.absolute,
                "relative": self.tolerance.relative,
            },
            "rows": [row.to_dict() for row in self.rows],
            "passed": list(self.passed),
            "failed": list(self.failed),
        }

    def render_markdown(self) -> str:
        """Render a compact Markdown gate summary."""
        lines = [
            f"## Regression gate: {'PASS' if self.ok else 'FAIL'}",
            "",
            f"Baseline `{self.baseline_suite}` to candidate `{self.candidate_suite}`. "
            f"Tolerance: absolute {self.tolerance.absolute:.3f}, "
            f"relative {self.tolerance.relative:.1%}.",
            "",
            "| Metric | Baseline | Candidate | Delta | Allowed drop | Status | Note |",
            "| --- | ---: | ---: | ---: | ---: | :---: | --- |",
        ]
        for row in self.rows:
            baseline = "n/a" if row.baseline_mean is None else f"{row.baseline_mean:.4f}"
            candidate = "n/a" if row.candidate_mean is None else f"{row.candidate_mean:.4f}"
            delta = "n/a" if row.delta is None else f"{row.delta:+.4f}"
            allowed = "n/a" if row.allowed_drop is None else f"{row.allowed_drop:.4f}"
            status = "pass" if row.passed else "FAIL"
            note = row.note.replace("|", "\\|") if row.note else ""
            lines.append(
                f"| `{row.metric}` | {baseline} | {candidate} | {delta} | "
                f"{allowed} | {status} | {note} |"
            )
        if self.failed:
            lines.extend(
                [
                    "",
                    "Failed metrics: " + ", ".join(f"`{name}`" for name in self.failed),
                ]
            )
        return "\n".join(lines) + "\n"


def _coerce_run(value: RunResult | Mapping[str, Any], label: str) -> RunResult:
    if isinstance(value, RunResult):
        return value
    if isinstance(value, Mapping):
        try:
            return RunResult.from_dict(value)
        except ValueError as exc:
            raise ValueError(f"invalid {label} run: {exc}") from exc
    raise TypeError(f"{label} run must be a RunResult or a mapping")


def compare_runs(
    candidate: RunResult | Mapping[str, Any],
    baseline: RunResult | Mapping[str, Any],
    *,
    tolerance: Tolerance | None = None,
    overrides: Mapping[str, Tolerance] | None = None,
) -> RegressionReport:
    """Compare candidate metrics against baseline metrics.

    Args:
        candidate: The run being gated.
        baseline: The reference run that must not be undercut.
        tolerance: Default gate tolerance.
        overrides: Per-metric tolerances, keyed by metric name.

    Metrics present only in the baseline fail with a "missing" note. Metrics
    added by the candidate pass but are labelled as ungated, since there is no
    baseline value to regress against.
    """
    candidate_run = _coerce_run(candidate, "candidate")
    baseline_run = _coerce_run(baseline, "baseline")
    default = tolerance or DEFAULT_TOLERANCE
    overrides = dict(overrides or {})

    ordered = list(baseline_run.metrics)
    for name in candidate_run.metrics:
        if name not in ordered:
            ordered.append(name)
    unknown = sorted(set(overrides) - set(ordered))
    if unknown:
        raise ValueError(
            "tolerance override for unknown metric(s): " + ", ".join(unknown)
        )

    rows: list[MetricComparison] = []
    for name in ordered:
        metric_tolerance = overrides.get(name, default)
        baseline_summary = baseline_run.summary.get(name)
        candidate_summary = candidate_run.summary.get(name)
        if baseline_summary is None and candidate_summary is None:
            rows.append(
                MetricComparison(
                    metric=name,
                    baseline_mean=None,
                    candidate_mean=None,
                    delta=None,
                    allowed_drop=None,
                    passed=False,
                    note="no aggregate available",
                )
            )
            continue
        if baseline_summary is None:
            rows.append(
                MetricComparison(
                    metric=name,
                    baseline_mean=None,
                    candidate_mean=candidate_summary.mean if candidate_summary else None,
                    delta=None,
                    allowed_drop=None,
                    passed=True,
                    note="new metric; no baseline to gate",
                )
            )
            continue
        if candidate_summary is None:
            rows.append(
                MetricComparison(
                    metric=name,
                    baseline_mean=baseline_summary.mean,
                    candidate_mean=None,
                    delta=None,
                    allowed_drop=None,
                    passed=False,
                    note="missing from candidate run",
                )
            )
            continue
        delta = candidate_summary.mean - baseline_summary.mean
        allowed = metric_tolerance.allowed_drop(baseline_summary.mean)
        relative = (
            delta / abs(baseline_summary.mean)
            if baseline_summary.mean != 0.0
            else None
        )
        passed = delta >= -allowed - 1e-12
        note = "within tolerance" if passed else "drop exceeds tolerance"
        rows.append(
            MetricComparison(
                metric=name,
                baseline_mean=baseline_summary.mean,
                candidate_mean=candidate_summary.mean,
                delta=delta,
                allowed_drop=allowed,
                passed=passed,
                note=note,
                relative_delta=relative,
            )
        )

    return RegressionReport(
        baseline_suite=baseline_run.suite,
        candidate_suite=candidate_run.suite,
        tolerance=default,
        rows=tuple(rows),
    )
