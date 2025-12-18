"""Tests for response generator and contracts."""

from __future__ import annotations

import json

from RouterGym.agents import generator as gen
from RouterGym.label_space import CANONICAL_LABELS


def test_build_prompt_includes_kb() -> None:
    prompt = gen.build_prompt("ticket text", ["kb1", "kb2"])
    assert "ticket text" in prompt
    assert "KB Reference 1" in prompt
    assert "kb2" in prompt


def test_json_contract_validation() -> None:
    contract = gen.JSONContract()
    ok, parsed = contract.validate(json.dumps({"a": 1, "b": 2}))
    assert ok and parsed["a"] == 1
    ok, parsed = contract.validate("not json")
    assert not ok and parsed is None


def test_schema_contract_fields() -> None:
    contract = gen.SchemaContract()
    payload = {
        "reasoning": "why",
        "final_answer": "ans",
        "predicted_category": "access",
    }
    ok, errors = contract.validate(payload)
    assert ok and not errors


class DummyModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        if self.calls == 1:
            return "invalid json"
        return json.dumps({"reasoning": "r", "final_answer": "a", "predicted_category": "access"})


def test_self_repair_fallback(monkeypatch) -> None:
    model = DummyModel()
    contract = gen.SchemaContract()
    repair = gen.SelfRepair(max_retries=2)

    # Force repair model to be the local dummy to keep test deterministic.
    monkeypatch.setattr(gen, "get_repair_model", lambda: model)

    fixed = repair.repair(model, "prompt", "invalid", contract)
    assert fixed["final_answer"] == "a"
    assert fixed["predicted_category"]


def test_repair_uses_llm(monkeypatch):
    """Ensure repair escalates to the dedicated LLM engine."""
    called = {}

    class RepairModel:
        def __call__(self, prompt: str, **kwargs):
            called["used"] = True
            return json.dumps({"reasoning": "r", "final_answer": "a", "predicted_category": "access"})

    monkeypatch.setattr(gen, "get_repair_model", lambda: RepairModel())
    model = DummyModel()
    contract = gen.SchemaContract()
    repair = gen.SelfRepair(max_retries=1)
    _ = repair.repair(model, "prompt", "invalid", contract)
    assert called.get("used")


class DummyEncoderClassifier:
    def __init__(self, *args, **kwargs):
        self.backend_name = "encoder_calibrated"

    def predict_proba(self, text: str):
        total = len(CANONICAL_LABELS)
        probs = {label: 0.0 for label in CANONICAL_LABELS}
        probs[CANONICAL_LABELS[0]] = 0.6
        remaining = (1.0 - probs[CANONICAL_LABELS[0]]) / float(total - 1)
        for label in CANONICAL_LABELS[1:]:
            probs[label] = remaining
        return probs


def test_run_ticket_pipeline(monkeypatch):
    monkeypatch.setattr(gen, "EncoderClassifier", DummyEncoderClassifier)
    result = gen.run_ticket_pipeline("simple hardware test ticket text")
    assert result["original_query"] == "simple hardware test ticket text"
    assert result["rewritten_query"] == "simple hardware test ticket text"
    assert result["topic_group"] in CANONICAL_LABELS
    assert result["classifier_label"] == result["classification"]["label"]
    assert isinstance(result["classifier_confidence_bucket"], str)
    assert result["classifier_backend"] == "encoder_calibrated"
    assert result["memory_mode"] == "none"
    assert result["kb_policy_ids"] == []
    assert result["kb_categories"] == []
    assert "classification" in result
    cls = result["classification"]
    assert cls["label"] in CANONICAL_LABELS
    assert isinstance(cls["confidence"], float)
    assert cls["confidence_bucket"] in {"high", "medium", "low"}
    assert isinstance(result["resolution_steps"], list)
    assert result["resolution_steps"] == []
    assert "final_answer" in result and isinstance(result["final_answer"], str)
    assert isinstance(result["escalation_flags"]["needs_human"], bool)
    assert isinstance(result["escalation_flags"]["needs_llm_escalation"], bool)
    assert isinstance(result["escalation_flags"]["policy_gap"], bool)
    assert isinstance(result["metrics"], dict)
    for key in ("latency_ms", "total_input_tokens", "total_output_tokens", "total_cost_usd"):
        assert key in result["metrics"]
