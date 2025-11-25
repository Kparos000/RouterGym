"""Evaluation metrics tests."""

from __future__ import annotations

from RouterGym.evaluation import metrics


def test_groundedness_similarity() -> None:
    score = metrics.groundedness_score("hello world", ["hello there"])
    assert score > 0


def test_schema_validity() -> None:
    valid = metrics.schema_validity({"reasoning": "r", "final_answer": "a", "predicted_category": "access"})
    assert valid == 1


def test_router_conversion() -> None:
    stats = [{"model_used": "slm"}, {"model_used": "llm"}, {"model_used": "slm"}]
    rate = metrics.router_conversion_rate(stats)
    assert rate == 2 / 3


def test_accuracy_and_cost() -> None:
    rec = {
        "output": {"final_answer": "hello", "reasoning": "r", "predicted_category": "access"},
        "label": "access",
        "predicted_category": "access",
        "kb_snippets": ["hello world"],
        "kb_attached": True,
        "model_used": "llm",
        "latency_ms": 10.0,
        "prompt_text": "prompt",
    }
    metrics_out = metrics.compute_all_metrics(rec)
    assert metrics_out["accuracy"] == 1.0
    cheaper = metrics.estimate_cost_usd("slm", "p", "a")
    expensive = metrics.estimate_cost_usd("llm", "p", "a")
    assert cheaper < expensive
    rec_missing_kb = {
        "output": {"final_answer": "hello", "reasoning": "r", "predicted_category": "unknown"},
        "label": "access",
        "predicted_category": "unknown",
        "kb_snippets": [],
        "kb_attached": False,
        "model_used": "slm",
        "latency_ms": 5.0,
        "prompt_text": "prompt",
    }
    metrics_out2 = metrics.compute_all_metrics(rec_missing_kb)
    assert metrics_out2["accuracy"] == 0.0
    assert metrics_out2["groundedness"] == 0.0
