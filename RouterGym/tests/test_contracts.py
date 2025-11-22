"""Contract enforcement tests."""

from __future__ import annotations

import json

from RouterGym.contracts.json_contract import JSONContract
from RouterGym.contracts.schema_contract import SchemaContract
from RouterGym.agents.generator import SelfRepair


class DummyModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return "not json"
        if self.calls == 2:
            return json.dumps({"classification": "c", "reasoning": "r", "action_steps": [], "final_answer": "a"})
        return prompt


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
    ok, errors = sc.validate({"classification": "x"})
    assert not ok
    assert errors


def test_self_repair_fills_schema() -> None:
    model = DummyModel()
    repair = SelfRepair()
    sc = SchemaContract()
    repaired = repair.repair(model, "prompt", "not json", sc)
    parsed = json.loads(repaired)
    assert sc.validate(parsed)[0]
