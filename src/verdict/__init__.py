"""Verdict: evaluation and observability for LLM and RAG systems.

The package is pure standard library. Importing it performs no I/O, opens no
sockets and starts no background work.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .suites.datasets import (
    Case,
    DatasetError,
    load_cases,
    load_predictions,
    sample_cases,
    split_cases,
)
from .judge.judge import (
    DEFAULT_RUBRIC,
    HeuristicJudge,
    Judge,
    JudgeError,
    Judgment,
    LLMJudge,
    Rubric,
)
from .metrics import (
    char_bleu4,
    exact_match,
    mrr,
    ndcg_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)
from .regression import (
    DEFAULT_TOLERANCE,
    MetricComparison,
    RegressionReport,
    Tolerance,
    compare_runs,
)
from .reports.report import render_html, render_markdown, write_report
from .runner import (
    DEFAULT_METRICS,
    RETRIEVAL_METRICS,
    TEXT_METRICS,
    CaseResult,
    MetricSummary,
    RunError,
    RunResult,
    load_run,
    run_suite,
    save_run,
    summarize,
)

__all__ = [
    "Case",
    "CaseResult",
    "DEFAULT_METRICS",
    "DEFAULT_RUBRIC",
    "DEFAULT_TOLERANCE",
    "DatasetError",
    "HeuristicJudge",
    "Judge",
    "JudgeError",
    "Judgment",
    "LLMJudge",
    "MetricComparison",
    "MetricSummary",
    "RETRIEVAL_METRICS",
    "RegressionReport",
    "Rubric",
    "RunError",
    "RunResult",
    "TEXT_METRICS",
    "Tolerance",
    "__version__",
    "char_bleu4",
    "compare_runs",
    "exact_match",
    "load_cases",
    "load_predictions",
    "load_run",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "render_html",
    "render_markdown",
    "rouge_l",
    "run_suite",
    "sample_cases",
    "save_run",
    "split_cases",
    "summarize",
    "token_f1",
    "write_report",
]
