"""Evaluation tests stubs."""

from RouterGym.evaluation.metrics import evaluate, MetricResult


def test_metrics_stub() -> None:
    """Placeholder test for metric evaluation."""
    result = evaluate({})
    assert isinstance(result, MetricResult)
