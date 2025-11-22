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
        "classification": "class",
        "reasoning": "why",
        "action_steps": [],
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
        return json.dumps({"classification": "c", "answer": "a", "reasoning": "r"})


def test_self_repair_fallback() -> None:
    model = DummyModel()
    contract = gen.SchemaContract()
    repair = gen.SelfRepair(max_retries=2)
    fixed = repair.repair(model, "prompt", "invalid", contract)
    data = json.loads(fixed)
    assert data["final_answer"] == "a"
