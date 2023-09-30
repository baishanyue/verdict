import pytest

from verdict.registry import REGISTRY, describe, get_metric, metric_names


def test_every_metric_has_a_family_and_callable():
    assert REGISTRY
    for metric in REGISTRY.values():
        assert metric.family in {"text", "retrieval"}
        assert callable(metric.fn)


def test_text_and_retrieval_families_are_disjoint():
    text = set(metric_names("text"))
    retrieval = set(metric_names("retrieval"))
    assert text and retrieval
    assert text.isdisjoint(retrieval)
    assert set(metric_names()) == text | retrieval


def test_unknown_metric_raises():
    with pytest.raises(KeyError, match="unknown metric"):
        get_metric("bleu_9")


@pytest.mark.parametrize("name", metric_names("text"))
def test_text_metrics_score_identical_strings_at_one(name):
    metric = get_metric(name)
    assert metric.fn("the same answer", "the same answer") == pytest.approx(1.0)


@pytest.mark.parametrize("name", metric_names("retrieval"))
def test_retrieval_metrics_are_zero_without_hits(name):
    metric = get_metric(name)
    score = metric.fn(["a", "b", "c"], ["z"], 3)
    assert 0.0 <= float(score) <= 1.0


def test_describe_is_json_shaped():
    rows = describe()
    assert [row["name"] for row in rows] == list(metric_names())
    assert all({"name", "family", "higher_is_better"} <= set(row) for row in rows)
