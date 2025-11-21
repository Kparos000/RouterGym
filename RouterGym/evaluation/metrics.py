"""Evaluation metrics implementations."""

from dataclasses import dataclass
from typing import Any, Dict, List
import json


@dataclass
class MetricResult:
    """Container for metric scores."""

    groundedness: float | None = None
    schema_validity: float | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    fallback_rate: float | None = None
    accuracy: float | None = None


def accuracy_score(predicted: str, true: str) -> float:
    """Simple accuracy: 1.0 if match else 0.0."""
    return 1.0 if predicted == true else 0.0


def groundedness_score(output: str, kb_snippets: List[str]) -> float:
    """Keyword overlap between output and KB snippets."""
    if not kb_snippets:
        return 0.0
    tokens_out = set(output.lower().split())
    score_sum = 0
    for snippet in kb_snippets:
        tokens_snip = set(snippet.lower().split())
        overlap = tokens_out.intersection(tokens_snip)
        score_sum += len(overlap) / max(len(tokens_snip), 1)
    return score_sum / len(kb_snippets)


def schema_validity_score(json_output: str) -> float:
    """Check required schema fields."""
    try:
        data = json.loads(json_output)
    except Exception:
        return 0.0
    required = {"classification", "answer", "reasoning"}
    return 1.0 if required.issubset(set(data.keys())) else 0.0


def latency_score(latency_ms: float) -> float:
    """Placeholder: lower latency is better; return inverse scaled."""
    if latency_ms <= 0:
        return 0.0
    return 1_000 / (latency_ms + 1)


def token_cost_estimate(model_name: str, token_count: int) -> float:
    """Placeholder token cost estimate."""
    slm_rate = 0.0001
    llm_rate = 0.001
    rate = slm_rate if model_name.startswith("slm") else llm_rate
    return rate * token_count


def compute_all_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    """Compute all metrics from a record."""
    output = record.get("output", "")
    label = record.get("label", "")
    kb_snippets = record.get("kb_snippets", [])
    latency = record.get("latency_ms", 0.0)
    model_name = record.get("model", "slm")
    tokens = record.get("tokens", 0)

    acc = accuracy_score(record.get("predicted", ""), label) if "predicted" in record else 0.0
    grounded = groundedness_score(output, kb_snippets)
    schema = schema_validity_score(output)
    latency_val = latency_score(latency)
    cost = token_cost_estimate(model_name, tokens)

    return {
        "accuracy": acc,
        "groundedness": grounded,
        "schema_validity": schema,
        "latency": latency_val,
        "cost": cost,
    }


def evaluate(outputs: Dict[str, Any]) -> MetricResult:
    """Wrapper to compute MetricResult from outputs dict."""
    metrics = compute_all_metrics(outputs)
    return MetricResult(
        accuracy=metrics["accuracy"],
        groundedness=metrics["groundedness"],
        schema_validity=metrics["schema_validity"],
        latency_ms=outputs.get("latency_ms"),
        cost_usd=metrics["cost"],
        fallback_rate=outputs.get("fallback_rate"),
    )
