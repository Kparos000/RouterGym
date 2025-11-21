"""Benchmark analyzer stubs."""

from typing import Any, Dict

from .metrics import MetricResult, evaluate


def summarize(run_artifacts: Dict[str, Any]) -> MetricResult:
    """Summarize run artifacts into aggregate metrics."""
    return evaluate(run_artifacts)
