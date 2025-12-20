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
    # Stub model registry to avoid real network calls.
    class FakeModel:
        def __call__(self, prompt: str, **kwargs):
            return json.dumps(
                {
                    "final_answer": "answer",
                    "reasoning": "reason",
                    "resolution_steps": ["step1"],
                }
            )

    monkeypatch.setattr(gen, "load_models", lambda sanity=True, slm_subset=None: {"slm1": FakeModel()})

    result = gen.run_ticket_pipeline(
        ticket={"text": "simple hardware test ticket text"},
        base_model_name="slm1",
        memory_mode="none",
        router_mode="slm_only",
    )
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


def test_run_ticket_pipeline_with_kb(monkeypatch):
    monkeypatch.setattr(gen, "EncoderClassifier", DummyEncoderClassifier)

    class FakeModel:
        def __call__(self, prompt: str, **kwargs):
            return json.dumps(
                {
                    "final_answer": "answer",
                    "reasoning": "reason",
                    "resolution_steps": ["step1"],
                }
            )

    from RouterGym.memory.base import MemoryBase, MemoryRetrieval  # type: ignore

    class DummyMemory(MemoryBase):
        def update(self, item, metadata=None):
            return None

        def summarize(self):
            return "summary"

        def retrieve(self, query=None):
            return MemoryRetrieval(
                retrieved_context="ctx",
                retrieval_metadata={
                    "mode": "rag_dense",
                    "query": query or "",
                    "snippets": [
                        {"policy_id": "hardware.doc1", "category": "Hardware", "text": "content"},
                        {"policy_id": "hardware.doc1", "category": "Hardware", "text": "content2"},
                    ],
                },
                retrieval_cost_tokens=0,
                relevance_score=1.0,
                retrieval_latency_ms=1.0,
            )

    monkeypatch.setattr(gen, "load_models", lambda sanity=True, slm_subset=None: {"slm1": FakeModel()})
    monkeypatch.setattr(gen, "get_memory_class", lambda mode: DummyMemory)

    result = gen.run_ticket_pipeline(
        ticket={"text": "ticket with kb"},
        base_model_name="slm1",
        memory_mode="rag_dense",
        router_mode="slm_only",
    )

    assert result["memory_mode"] == "rag_dense"
    assert result["kb_policy_ids"] == ["hardware.doc1"]
    assert result["kb_categories"] == ["Hardware"]


def test_slm_dominant_escalates_with_reasons(monkeypatch):
    class LowConfClassifier:
        def __init__(self, *args, **kwargs):
            self.backend_name = "encoder_calibrated"

        def predict_proba(self, text: str):
            probs = {label: 0.0 for label in CANONICAL_LABELS}
            probs[CANONICAL_LABELS[0]] = 0.2
            remaining = (1.0 - probs[CANONICAL_LABELS[0]]) / float(len(CANONICAL_LABELS) - 1)
            for label in CANONICAL_LABELS[1:]:
                probs[label] = remaining
            return probs

    from RouterGym.memory.base import MemoryBase, MemoryRetrieval  # type: ignore

    class DummyMemory(MemoryBase):
        def update(self, item, metadata=None):
            return None

        def summarize(self):
            return "summary"

        def retrieve(self, query=None):
            return MemoryRetrieval(
                retrieved_context="ctx",
                retrieval_metadata={"mode": "rag_dense", "query": query or "", "snippets": []},
                retrieval_cost_tokens=0,
                relevance_score=0.0,
                retrieval_latency_ms=1.0,
            )

    class BaseModel:
        def __call__(self, prompt: str, **kwargs):
            return json.dumps({"final_answer": "short", "reasoning": "r", "resolution_steps": []})

    class EscalationModel:
        def __call__(self, prompt: str, **kwargs):
            return json.dumps(
                {"final_answer": "This is a longer detailed answer for the user", "reasoning": "rr", "resolution_steps": ["step1"]}
            )

    monkeypatch.setattr(gen, "EncoderClassifier", LowConfClassifier)
    monkeypatch.setattr(gen, "load_models", lambda sanity=True, slm_subset=None: {"slm1": BaseModel(), "llm1": EscalationModel()})
    monkeypatch.setattr(gen, "get_memory_class", lambda mode: DummyMemory)
    monkeypatch.setattr(gen, "validate_agent_output", lambda payload: dict(payload))

    result = gen.run_ticket_pipeline(
        ticket={"text": "ticket needing escalation"},
        base_model_name="slm1",
        escalation_model_name="llm1",
        memory_mode="rag_dense",
        router_mode="slm_dominant",
    )

    reasons = result["escalation_flags"]["reasons"]
    assert result["model_name"] == "llm1"
    assert result["escalation_flags"]["needs_llm_escalation"] is True
    assert "low_confidence" in reasons
    assert "weak_kb" in reasons
    assert "short_answer" in reasons


def test_slm_dominant_no_escalation(monkeypatch):
    class HighConfClassifier:
        def __init__(self, *args, **kwargs):
            self.backend_name = "encoder_calibrated"

        def predict_proba(self, text: str):
            probs = {label: 0.0 for label in CANONICAL_LABELS}
            probs[CANONICAL_LABELS[0]] = 0.9
            remaining = (1.0 - probs[CANONICAL_LABELS[0]]) / float(len(CANONICAL_LABELS) - 1)
            for label in CANONICAL_LABELS[1:]:
                probs[label] = remaining
            return probs

    from RouterGym.memory.base import MemoryBase, MemoryRetrieval  # type: ignore

    class DummyMemory(MemoryBase):
        def update(self, item, metadata=None):
            return None

        def summarize(self):
            return "summary"

        def retrieve(self, query=None):
            return MemoryRetrieval(
                retrieved_context="ctx",
                retrieval_metadata={"mode": "rag_dense", "query": query or "", "snippets": []},
                retrieval_cost_tokens=0,
                relevance_score=0.2,
                retrieval_latency_ms=1.0,
            )

    class BaseModel:
        def __call__(self, prompt: str, **kwargs):
            return json.dumps(
                {
                    "final_answer": "This is a sufficiently detailed answer with enough tokens to avoid escalation",
                    "reasoning": "r",
                    "resolution_steps": ["step1"],
                }
            )

    monkeypatch.setattr(gen, "EncoderClassifier", HighConfClassifier)
    monkeypatch.setattr(
        gen,
        "load_models",
        lambda sanity=True, slm_subset=None: {"slm1": BaseModel(), "llm1": BaseModel()},
    )
    monkeypatch.setattr(gen, "get_memory_class", lambda mode: DummyMemory)
    monkeypatch.setattr(gen, "validate_agent_output", lambda payload: dict(payload))

    result = gen.run_ticket_pipeline(
        ticket={"text": "ticket with good SLM answer"},
        base_model_name="slm1",
        escalation_model_name="llm1",
        memory_mode="rag_dense",
        router_mode="slm_dominant",
    )

    assert result["model_name"] == "slm1"
    assert result["escalation_flags"]["needs_llm_escalation"] is False
    assert result["escalation_flags"]["reasons"] == []
