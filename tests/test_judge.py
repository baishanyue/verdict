from __future__ import annotations

import pytest

from verdict.suites.datasets import Case
from verdict.judge.judge import (
    HeuristicJudge,
    Judge,
    JudgeError,
    Judgment,
    LLMJudge,
    Rubric,
    parse_judgment,
)


def make_case(**kwargs) -> Case:
    defaults = {
        "id": "c1",
        "input": "What did the cat do?",
        "expected": "The cat sat on the mat.",
        "contexts": ("The cat sat on the mat.",),
    }
    defaults.update(kwargs)
    return Case(**defaults)


def test_heuristic_judge_is_deterministic_and_labelled() -> None:
    judge = HeuristicJudge()
    case = make_case()
    first = judge.judge(case, "The cat sat on the mat.")
    second = judge.judge(case, "The cat sat on the mat.")
    assert first.score == pytest.approx(1.0)
    assert first == second
    assert first.judge == "heuristic-baseline"
    assert "not an LLM" in first.rationale
    assert set(first.criteria) == {"coverage", "grounding", "concision"}
    assert isinstance(judge, Judge)
    assert judge.is_llm is False


def test_heuristic_judge_partial_and_empty() -> None:
    judge = HeuristicJudge()
    case = make_case()
    partial = judge.judge(case, "cat")
    assert partial.score == pytest.approx(0.5875)
    assert partial.criteria["coverage"] == pytest.approx(0.25)
    empty = judge.judge(case, "")
    assert empty.score == 0.0
    assert empty.criteria["concision"] == 0.0


def test_heuristic_judge_without_contexts_renormalises() -> None:
    judge = HeuristicJudge()
    case = make_case(contexts=())
    result = judge.judge(case, "The cat sat on the mat.")
    assert "grounding" not in result.criteria
    assert result.score == pytest.approx(1.0)


def test_required_keywords_cap_fluent_answers() -> None:
    plain = HeuristicJudge().judge(make_case(), "cat")
    strict = HeuristicJudge(
        Rubric(required_keywords=("sat on the mat",))
    ).judge(make_case(), "cat")
    assert strict.score < plain.score
    assert strict.criteria["coverage"] < plain.criteria["coverage"]


def test_rubric_validation() -> None:
    with pytest.raises(ValueError):
        Rubric(coverage_weight=-1)
    with pytest.raises(ValueError):
        Rubric(coverage_weight=0, grounding_weight=0, concision_weight=0)
    with pytest.raises(ValueError):
        Rubric(max_length_ratio=1.0)
    with pytest.raises(ValueError):
        Rubric(pass_threshold=2.0)
    with pytest.raises(ValueError):
        Rubric(required_keywords=("",))
    assert "coverage" in Rubric().describe()


def test_llm_judge_uses_injected_callable() -> None:
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return '{"score": 0.8, "rationale": "mostly right", "criteria": {"coverage": 0.9}}'

    judge = LLMJudge(complete, model="fake-model")
    result = judge.judge(make_case(), "The cat sat down.")
    assert result.score == 0.8
    assert result.judge == "llm"
    assert result.rationale == "mostly right"
    assert result.criteria == {"coverage": 0.9}
    assert len(prompts) == 1
    assert "What did the cat do?" in prompts[0]
    assert "The cat sat on the mat." in prompts[0]
    assert "JSON" in prompts[0]
    assert judge.is_llm is True


def test_llm_judge_requires_callable() -> None:
    with pytest.raises(TypeError):
        LLMJudge(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"score": 0.5, "rationale": "ok"}\n```',
        'Here is the grade: {"score": 0.5, "rationale": "ok"} thanks.',
        '{"score": 0.5, "rationale": "ok", "criteria": {"coverage": 0.5}}',
    ],
)
def test_parse_judgment_accepts_fences_and_prose(raw: str) -> None:
    result = parse_judgment(raw)
    assert result.score == 0.5
    assert result.judge == "llm"


@pytest.mark.parametrize(
    "raw",
    [
        "no object here",
        '{"score": 2.0}',
        '{"score": true}',
        '{"score": "0.5"}',
        '{"rationale": "missing score"}',
        '{"score": 0.5, "criteria": {"x": "high"}}',
    ],
)
def test_parse_judgment_rejects_bad_payloads(raw: str) -> None:
    with pytest.raises(JudgeError):
        parse_judgment(raw)


def test_judgment_roundtrip() -> None:
    original = Judgment(
        score=0.75,
        rationale="fine",
        judge="custom",
        criteria={"coverage": 0.8},
    )
    restored = Judgment.from_dict(original.to_dict())
    assert restored == original
    with pytest.raises(JudgeError):
        Judgment.from_dict({"score": 1.5})
    with pytest.raises(JudgeError):
        Judgment.from_dict({"score": 0.5, "criteria": {"x": "bad"}})
