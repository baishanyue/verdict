"""Markdown and self-contained HTML reports for evaluation runs."""

from __future__ import annotations

import html
from pathlib import Path

from ..regression import RegressionReport, compare_runs
from ..runner import RunResult

__all__ = ["render_html", "render_markdown", "write_report"]


def _short(text: str, limit: int = 96) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _md_cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _gate_status(regression: RegressionReport | None) -> str:
    if regression is None:
        return ""
    return "PASS" if regression.ok else "FAIL"


def _summary_rows(
    run: RunResult,
    regression: RegressionReport | None,
) -> list[list[str]]:
    by_metric = (
        {row.metric: row for row in regression.rows} if regression is not None else {}
    )
    rows: list[list[str]] = []
    for name in run.metrics:
        summary = run.summary[name]
        row = [
            f"`{name}`",
            str(summary.count),
            f"{summary.mean:.4f}",
            f"{summary.median:.4f}",
            f"{summary.p95:.4f}",
        ]
        if regression is not None:
            comparison = by_metric.get(name)
            if comparison is None or comparison.baseline_mean is None:
                baseline_text = "n/a"
            else:
                baseline_text = f"{comparison.baseline_mean:.4f}"
            delta_text = (
                "n/a"
                if comparison is None or comparison.delta is None
                else f"{comparison.delta:+.4f}"
            )
            status = "n/a" if comparison is None else ("pass" if comparison.passed else "FAIL")
            row.extend([baseline_text, delta_text, status])
        rows.append(row)
    return rows


def render_markdown(
    run: RunResult,
    *,
    baseline: RunResult | None = None,
    regression: RegressionReport | None = None,
    title: str | None = None,
) -> str:
    """Render a Markdown report for one run.

    Passing ``baseline`` computes a regression summary automatically unless an
    explicit ``regression`` report is supplied.
    """
    if baseline is not None and regression is None:
        regression = compare_runs(run, baseline)
    heading = title or f"Verdict report: {run.suite}"
    lines = [
        f"# {heading}",
        "",
        f"**Suite:** `{run.suite}` | **Cases:** {len(run.cases)} | "
        f"**Metrics:** {len(run.metrics)} | **Retrieval k:** {run.k}",
        "",
    ]
    if regression is not None:
        gate = _gate_status(regression)
        lines.extend(
            [
                f"**Regression gate:** {gate} "
                f"({len(regression.passed)}/{len(regression.rows)} metrics within tolerance, "
                f"absolute {regression.tolerance.absolute:.3f}, "
                f"relative {regression.tolerance.relative:.1%})",
                "",
            ]
        )

    lines.extend(["## Summary", ""])
    if regression is None:
        lines.extend(
            [
                "| Metric | Cases | Mean | Median | p95 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Metric | Cases | Mean | Median | p95 | Baseline | Delta | Gate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
    for row in _summary_rows(run, regression):
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Per-case scores", ""])
    header = ["Case", *[f"`{name}`" for name in run.metrics], "Prediction"]
    divider = ["---", *[":---:" for _ in run.metrics], "---"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(divider) + " |")
    for case in run.cases:
        cells = [f"`{_md_cell(case.case_id)}`"]
        for name in run.metrics:
            cells.append(f"{case.scores[name]:.4f}")
        prediction = _short(case.prediction) if case.prediction.strip() else "(empty)"
        if case.missing_prediction:
            prediction = "(missing prediction)"
        cells.append(_md_cell(prediction))
        lines.append("| " + " | ".join(cells) + " |")

    judgments = [case for case in run.cases if case.judge is not None]
    if judgments:
        lines.extend(["", "## Judge notes", ""])
        lines.extend(
            [
                "| Case | Score | Rationale |",
                "| --- | ---: | --- |",
            ]
        )
        for case in judgments:
            assert case.judge is not None
            lines.append(
                "| `"
                + _md_cell(case.case_id)
                + "` | "
                + f"{case.judge.score:.3f} | "
                + _md_cell(case.judge.rationale)
                + " |"
            )

    if run.missing_predictions:
        lines.extend(
            [
                "",
                "## Missing predictions",
                "",
                *[f"- `{_md_cell(case_id)}`" for case_id in run.missing_predictions],
            ]
        )
    if run.ignored_predictions:
        lines.extend(
            [
                "",
                "## Ignored predictions",
                "",
                "Prediction ids with no matching case:",
                "",
                *[f"- `{_md_cell(case_id)}`" for case_id in run.ignored_predictions],
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "Generated by Verdict. Scores are deterministic for a fixed run file; "
            "judge rows from a model adapter depend on the injected model.",
            "",
        ]
    )
    return "\n".join(lines)


_CSS = """
:root {
  --bg: #080B10;
  --panel: #0C1118;
  --panel-2: #0A0F15;
  --line: #1C2530;
  --text: #E6EDF3;
  --muted: #8B98A9;
  --accent: #7EE7D0;
  --blue: #8FB8FF;
  --violet: #B9A6FF;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
html { color-scheme: dark; }
body {
  margin: 0;
  background:
    radial-gradient(1000px 460px at 82% -8%, rgba(126, 231, 208, 0.09), transparent 62%),
    radial-gradient(800px 420px at -6% 8%, rgba(143, 184, 255, 0.06), transparent 60%),
    var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 40px 28px 72px; }
.top {
  display: flex; justify-content: space-between; gap: 16px;
  padding-bottom: 18px; border-bottom: 1px solid var(--line);
  font-family: var(--mono); font-size: 11px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted);
}
.top .mark { color: var(--text); }
.hero { padding: 56px 0 36px; }
.kicker {
  margin: 0 0 14px; font-family: var(--mono); font-size: 11px;
  letter-spacing: 0.18em; text-transform: uppercase; color: var(--accent);
}
h1 { margin: 0 0 10px; font-size: clamp(38px, 6vw, 64px); line-height: 1.02; letter-spacing: -0.04em; font-weight: 600; }
.lede { margin: 0; color: var(--muted); font-family: var(--mono); font-size: 12px; letter-spacing: 0.04em; }
.lede span { color: var(--text); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 8px 0 40px; }
.card { border: 1px solid var(--line); background: linear-gradient(180deg, var(--panel), var(--panel-2)); padding: 18px 18px 16px; border-radius: 2px; }
.card .label { font-family: var(--mono); font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); }
.card .value { margin-top: 8px; font-size: 28px; letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
.card .value.accent { color: var(--accent); }
.card .value.fail { color: #FF9E9E; }
.gate { border: 1px solid var(--line); border-left: 2px solid var(--accent); padding: 14px 18px; margin: 0 0 36px; font-size: 14px; background: rgba(126, 231, 208, 0.04); }
.gate.fail { border-left-color: #FF9E9E; background: rgba(255, 158, 158, 0.05); }
.gate strong { color: var(--accent); }
.gate.fail strong { color: #FF9E9E; }
h2 { margin: 42px 0 14px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); font-weight: 500; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { padding: 11px 12px; text-align: left; font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--line); white-space: nowrap; background: var(--panel); }
td { padding: 10px 12px; border-bottom: 1px solid rgba(28, 37, 48, 0.7); vertical-align: top; }
tr:last-child td { border-bottom: 0; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.mono, .mono { font-family: var(--mono); font-size: 12px; }
td.pred { color: var(--muted); max-width: 420px; }
.pass { color: var(--accent); font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; }
.fail { color: #FF9E9E; font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; }
.muted { color: var(--muted); }
.note { margin-top: 10px; color: var(--muted); font-size: 12px; }
ul.plain { margin: 0; padding-left: 18px; color: var(--muted); font-family: var(--mono); font-size: 12px; }
footer { margin-top: 64px; padding-top: 18px; border-top: 1px solid var(--line); display: flex; justify-content: space-between; gap: 16px; color: var(--muted); font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; }
@media (max-width: 640px) {
  .wrap { padding: 28px 18px 56px; }
  .hero { padding: 38px 0 26px; }
  .top { font-size: 10px; }
  footer { flex-direction: column; }
}
"""


def _html_summary(run: RunResult, regression: RegressionReport | None) -> str:
    by_metric = (
        {row.metric: row for row in regression.rows} if regression is not None else {}
    )
    head = ["Metric", "Cases", "Mean", "Median", "p95"]
    if regression is not None:
        head.extend(["Baseline", "Delta", "Gate"])
    cells = [
        "".join(
            f'<th class="{"num" if index > 0 else ""}">{html.escape(name)}</th>'
            for index, name in enumerate(head)
        )
    ]
    rows: list[str] = []
    for name in run.metrics:
        summary = run.summary[name]
        comparison = by_metric.get(name)
        row = [
            f'<td class="mono">{html.escape(name)}</td>',
            f'<td class="num">{summary.count}</td>',
            f'<td class="num">{summary.mean:.4f}</td>',
            f'<td class="num">{summary.median:.4f}</td>',
            f'<td class="num">{summary.p95:.4f}</td>',
        ]
        if regression is not None:
            baseline = (
                "n/a"
                if comparison is None or comparison.baseline_mean is None
                else f"{comparison.baseline_mean:.4f}"
            )
            delta = (
                "n/a"
                if comparison is None or comparison.delta is None
                else f"{comparison.delta:+.4f}"
            )
            status = (
                "n/a"
                if comparison is None
                else (
                    '<span class="pass">pass</span>'
                    if comparison.passed
                    else '<span class="fail">FAIL</span>'
                )
            )
            row.extend(
                [
                    f'<td class="num">{baseline}</td>',
                    f'<td class="num">{delta}</td>',
                    f'<td>{status}</td>',
                ]
            )
        rows.append("<tr>" + "".join(row) + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(cells)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _html_cases(run: RunResult) -> str:
    head = ["Case", *run.metrics, "Prediction"]
    header = "".join(
        f'<th class="{"num" if 0 < index <= len(run.metrics) else ""}">'
        f"{html.escape(name)}</th>"
        for index, name in enumerate(head)
    )
    rows: list[str] = []
    for case in run.cases:
        cells = [f'<td class="mono">{html.escape(case.case_id)}</td>']
        for name in run.metrics:
            cells.append(f'<td class="num">{case.scores[name]:.4f}</td>')
        if case.missing_prediction:
            prediction = '<span class="fail">missing prediction</span>'
        elif case.prediction.strip():
            prediction = html.escape(_short(case.prediction, 120))
        else:
            prediction = '<span class="muted">(empty)</span>'
        cells.append(f'<td class="pred">{prediction}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _html_judges(run: RunResult) -> str:
    judgments = [case for case in run.cases if case.judge is not None]
    if not judgments:
        return ""
    rows: list[str] = []
    for case in judgments:
        assert case.judge is not None
        rows.append(
            "<tr>"
            f'<td class="mono">{html.escape(case.case_id)}</td>'
            f'<td class="num">{case.judge.score:.3f}</td>'
            f"<td>{html.escape(case.judge.rationale)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Case</th><th class=\"num\">Score</th><th>Rationale</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def render_html(
    run: RunResult,
    *,
    baseline: RunResult | None = None,
    regression: RegressionReport | None = None,
    title: str | None = None,
) -> str:
    """Render a single-file dark HTML report with inline CSS and no scripts."""
    if baseline is not None and regression is None:
        regression = compare_runs(run, baseline)
    page_title = title or f"Verdict report: {run.suite}"
    primary = run.summary[run.metrics[0]] if run.metrics else None
    gate_html = ""
    if regression is not None:
        gate_html = (
            f'<div class="gate{" fail" if not regression.ok else ""}">'
            f"<strong>{'PASS' if regression.ok else 'FAIL'}</strong> "
            f"{len(regression.passed)}/{len(regression.rows)} metrics within "
            f"tolerance (absolute {regression.tolerance.absolute:.3f}, "
            f"relative {regression.tolerance.relative:.1%})."
            "</div>"
        )
    cards = [
        ("Cases", str(len(run.cases)), ""),
        ("Metrics", str(len(run.metrics)), ""),
        (
            "Primary mean",
            "n/a" if primary is None else f"{primary.mean:.3f}",
            "accent",
        ),
    ]
    if regression is not None:
        cards.append(
            (
                "Gate",
                "PASS" if regression.ok else "FAIL",
                "" if regression.ok else "fail",
            )
        )
    cards_html = "".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value {css}">{html.escape(value)}</div></div>'
        for label, value, css in cards
    )
    missing_html = ""
    if run.missing_predictions:
        items = "".join(
            f"<li>{html.escape(case_id)}</li>" for case_id in run.missing_predictions
        )
        missing_html = (
            '<h2>Missing predictions</h2><ul class="plain">' + items + "</ul>"
        )
    ignored_html = ""
    if run.ignored_predictions:
        items = "".join(
            f"<li>{html.escape(case_id)}</li>" for case_id in run.ignored_predictions
        )
        ignored_html = (
            '<h2>Ignored predictions</h2><p class="note">'
            "Prediction ids with no matching case.</p>"
            '<ul class="plain">' + items + "</ul>"
        )
    judges_html = _html_judges(run)
    judges_section = (
        f"<h2>Judge notes</h2>{judges_html}" if judges_html else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>{html.escape(page_title)}</title>
<style>{_CSS}</style>
</head>
<body>
<main class="wrap">
<header class="top">
  <div class="mark">Verdict</div>
  <div>EVALUATION REPORT / {html.escape(run.suite)}</div>
</header>
<section class="hero">
  <p class="kicker">Run report</p>
  <h1>{html.escape(run.suite)}</h1>
  <p class="lede"><span>{len(run.cases)}</span> cases &nbsp;/&nbsp; <span>{len(run.metrics)}</span> metrics &nbsp;/&nbsp; retrieval k=<span>{run.k}</span></p>
</section>
<div class="cards">{cards_html}</div>
{gate_html}
<h2>Summary</h2>
{_html_summary(run, regression)}
<h2>Per-case scores</h2>
{_html_cases(run)}
{judges_section}
{missing_html}
{ignored_html}
<footer>
  <div>Verdict / {html.escape(run.suite)}</div>
  <div>Deterministic metrics. Injected judge. No network required.</div>
</footer>
</main>
</body>
</html>
"""


def write_report(
    run: RunResult,
    path: str | Path,
    *,
    fmt: str | None = None,
    baseline: RunResult | None = None,
    regression: RegressionReport | None = None,
    title: str | None = None,
) -> Path:
    """Write a Markdown or HTML report, inferring the format from the suffix."""
    target = Path(path)
    suffix = target.suffix.lower()
    chosen = fmt or {".html": "html", ".htm": "html", ".md": "md", ".markdown": "md"}.get(
        suffix
    )
    if chosen == "html":
        content = render_html(run, baseline=baseline, regression=regression, title=title)
    elif chosen == "md":
        content = render_markdown(
            run, baseline=baseline, regression=regression, title=title
        )
    else:
        raise ValueError("report format must be 'md' or 'html'")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target
