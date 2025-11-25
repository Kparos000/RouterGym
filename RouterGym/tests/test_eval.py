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


def test_token_cost() -> None:
    cost = metrics.token_cost("slm_model", "one two three")
    assert cost > 0
