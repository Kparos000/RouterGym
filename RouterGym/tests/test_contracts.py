"""Contract enforcement tests."""

from __future__ import annotations

import json

import pytest

from RouterGym.contracts.json_contract import JSONContract, validate_agent_output
from RouterGym.contracts.schema_contract import AgentOutputSchema, SchemaContract
from RouterGym.agents.generator import SelfRepair


class DummyModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return "not json"
        return json.dumps({"reasoning": "r", "final_answer": "a", "predicted_category": "access"})


def test_json_contract_valid() -> None:
    jc = JSONContract()
    ok, parsed = jc.validate('{"a":1}')
    assert ok and parsed["a"] == 1


def test_json_contract_invalid() -> None:
    jc = JSONContract()
    ok, parsed = jc.validate("not json")
    assert not ok and parsed is None


def test_schema_contract_missing_fields() -> None:
    sc = SchemaContract()
    ok, errors = sc.validate({"reasoning": "x"})
    assert not ok
    assert errors


def test_self_repair_fills_schema() -> None:
    model = DummyModel()
    repair = SelfRepair()
    sc = SchemaContract()
    repaired = repair.repair(model, "prompt", "not json", sc)
    assert sc.validate(repaired)[0]
    assert repaired["predicted_category"]


def _build_agent_output_payload() -> dict:
    return {
        "original_query": "laptop is broken",
        "rewritten_query": "laptop is broken",
        "category": "Hardware",
        "classifier_label": "Hardware",
        "classifier_backend": "encoder_calibrated",
        "classifier_confidence": 0.9,
        "classifier_confidence_bucket": "high",
        "classification": {
            "label": "Hardware",
            "confidence": 0.9,
            "confidence_bucket": "high",
        },
        "router_name": "slm_dominant",
        "model_used": "slm1",
        "context_mode": "none",
        "memory_mode": "none",
        "kb_policy_ids": [],
        "kb_categories": [],
        "resolution_steps": [],
        "reasoning": "dummy",
        "escalation": {
            "agent_escalation": False,
            "human_escalation": False,
            "reasons": [],
        },
        "metrics": {
            "latency_ms": 1.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
        },
    }


def test_agent_output_schema_happy_path() -> None:
    payload = _build_agent_output_payload()
    schema = AgentOutputSchema()
    ok, errors = schema.validate(dict(payload))
    assert ok and not errors
    validated = validate_agent_output(payload)
    assert validated["category"] == "Hardware"
    assert validated["context_mode"] == "none"


def test_agent_output_schema_missing_field() -> None:
    payload = _build_agent_output_payload()
    payload.pop("classifier_backend")
    schema = AgentOutputSchema()
    ok, errors = schema.validate(payload)
    assert not ok
    with pytest.raises(ValueError):
        validate_agent_output(payload)


def test_agent_output_schema_invalid_category() -> None:
    payload = _build_agent_output_payload()
    payload["category"] = "invalid"
    schema = AgentOutputSchema()
    ok, errors = schema.validate(dict(payload))
    assert not ok
    with pytest.raises(ValueError):
        validate_agent_output(payload)


def test_agent_output_schema_invalid_metrics_type() -> None:
    payload = _build_agent_output_payload()
    payload["metrics"]["cost_usd"] = "bad"  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_agent_output(payload)


def test_agent_output_schema_invalid_bucket() -> None:
    payload = _build_agent_output_payload()
    payload["classification"]["confidence_bucket"] = "unknown"
    with pytest.raises(ValueError):
        validate_agent_output(payload)
    payload = _build_agent_output_payload()
    payload["classifier_confidence_bucket"] = "unknown"
    with pytest.raises(ValueError):
        validate_agent_output(payload)
