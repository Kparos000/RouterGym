"""Tests for response generator and contracts."""

from __future__ import annotations

import json

from RouterGym.agents import generator as gen


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
    }
    ok, errors = contract.validate(payload)
    assert ok and not errors


class DummyModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return "invalid json"
        return json.dumps({"reasoning": "r", "final_answer": "a"})


def test_self_repair_fallback(monkeypatch) -> None:
    model = DummyModel()
    contract = gen.SchemaContract()
    repair = gen.SelfRepair(max_retries=2)

    # Force repair model to be the local dummy to keep test deterministic.
    monkeypatch.setattr(gen, "get_repair_model", lambda: model)

    fixed = repair.repair(model, "prompt", "invalid", contract)
    assert fixed["final_answer"] == "a"


def test_repair_uses_llm(monkeypatch):
    """Ensure repair escalates to the dedicated LLM engine."""
    called = {}

    class RepairModel:
        def __call__(self, prompt: str, **kwargs):
            called["used"] = True
            return json.dumps({"reasoning": "r", "final_answer": "a"})

    monkeypatch.setattr(gen, "get_repair_model", lambda: RepairModel())
    model = DummyModel()
    contract = gen.SchemaContract()
    repair = gen.SelfRepair(max_retries=1)
    _ = repair.repair(model, "prompt", "invalid", contract)
    assert called.get("used")
