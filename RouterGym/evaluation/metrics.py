"""Evaluation metrics stubs."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MetricResult:
    """Container for metric scores."""

    groundedness: float | None = None
    schema_validity: float | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    fallback_rate: float | None = None
    accuracy: float | None = None


def evaluate(outputs: Dict[str, Any]) -> MetricResult:
    """Stub evaluation over model outputs."""
    return MetricResult()
