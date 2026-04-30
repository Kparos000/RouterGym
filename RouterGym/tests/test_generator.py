"""Tests for response generator and contracts."""

from __future__ import annotations

import json

from RouterGym.agents import generator as gen
from RouterGym.classifiers import encoder_classifier as enc
from RouterGym.label_space import CANONICAL_LABELS


def test_build_prompt_includes_kb() -> None:
    prompt = gen.build_prompt("ticket text", ["kb1", "kb2"])
    assert "ticket text" in prompt
    assert "KB Reference 1" in prompt
    assert "kb2" in prompt
    assert "ticket_request" in prompt
    assert "ticket_id" in prompt
    assert "Do NOT return benchmark metadata" in prompt


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
            return json.dumps(
                {"reasoning": "r", "final_answer": "a", "predicted_category": "access"}
            )

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
                    "ticket_request": "User reports a hardware issue with a simple test ticket.",
                    "final_answer": "answer",
                    "reasoning": "reason",
                    "resolution_steps": ["step1", "step2"],
                }
            )

    monkeypatch.setattr(
        gen, "load_models", lambda sanity=True, slm_subset=None: {"slm1": FakeModel()}
    )

    result = gen.run_ticket_pipeline(
        ticket={"text": "simple hardware test ticket text"},
        base_model_name="slm1",
        memory_mode="none",
        router_mode="slm_only",
    )
    assert result["original_query"] == "simple hardware test ticket text"
    assert result["ticket_request"] == "User reports a hardware issue with a simple test ticket."
    assert result["rewritten_query"] == result["ticket_request"]
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
    assert result["resolution_steps"] == ["step1", "step2"]
    assert "final_answer" in result and isinstance(result["final_answer"], str)
    assert isinstance(result["escalation_flags"]["needs_human"], bool)
    assert isinstance(result["escalation_flags"]["needs_llm_escalation"], bool)
    assert isinstance(result["escalation_flags"]["policy_gap"], bool)
    assert result["generation_valid"] is True
    assert result["has_real_final_answer"] is True
    assert result["has_resolution_steps"] is True
    assert result["placeholder_answer"] is False
    assert result["raw_response_saved"] is True
    assert result["raw_model_response_text"]
    assert result["parsed_output_before_validation"]["final_answer"] == "answer"
    assert result["parse_error"] is None
    assert result["validation_error"] is None
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
                    "resolution_steps": ["step1", "step2"],
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

    monkeypatch.setattr(
        gen, "load_models", lambda sanity=True, slm_subset=None: {"slm1": FakeModel()}
    )
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
    assert result["ticket_request"] == "ticket with kb"


def test_run_ticket_pipeline_normalizes_natural_language_output(monkeypatch):
    monkeypatch.setattr(gen, "EncoderClassifier", DummyEncoderClassifier)

    class FakeModel:
        def __call__(self, prompt: str, **kwargs):
            return (
                "Answer: Reset the VPN connection and sign in again.\n"
                "Reasoning: This addresses the reported remote access issue.\n"
                "1. Disconnect from VPN.\n"
                "2. Reconnect using SSO.\n"
            )

    monkeypatch.setattr(
        gen, "load_models", lambda sanity=True, slm_subset=None: {"slm1": FakeModel()}
    )

    result = gen.run_ticket_pipeline(
        ticket={"text": "vpn issue"},
        base_model_name="slm1",
        memory_mode="none",
        router_mode="slm_only",
    )

    assert result["generation_valid"] is True
    assert result["parse_error"] == "Model output is not valid JSON."
    assert result["ticket_request"] == "vpn issue"
    assert result["final_answer"] == "Reset the VPN connection and sign in again."
    assert result["reasoning"] == "This addresses the reported remote access issue."
    assert result["resolution_steps"] == ["Disconnect from VPN.", "Reconnect using SSO."]
    assert result["placeholder_answer"] is False
    assert result["raw_response_saved"] is True


def test_run_ticket_pipeline_preserves_raw_text_on_malformed_output(monkeypatch):
    monkeypatch.setattr(gen, "EncoderClassifier", DummyEncoderClassifier)

    class FakeModel:
        def __call__(self, prompt: str, **kwargs):
            return "???"

    monkeypatch.setattr(
        gen, "load_models", lambda sanity=True, slm_subset=None: {"slm1": FakeModel()}
    )

    result = gen.run_ticket_pipeline(
        ticket={"text": "bad output case"},
        base_model_name="slm1",
        memory_mode="none",
        router_mode="slm_only",
    )

    assert result["generation_valid"] is False
    assert result["placeholder_answer"] is True
    assert result["has_real_final_answer"] is False
    assert result["has_resolution_steps"] is False
    assert result["raw_response_saved"] is True
    assert result["raw_model_response_text"] == "???"


def test_run_ticket_pipeline_normalizes_extra_metadata_to_minimal_payload(monkeypatch):
    monkeypatch.setattr(gen, "EncoderClassifier", DummyEncoderClassifier)

    class FakeModel:
        def __call__(self, prompt: str, **kwargs):
            return json.dumps(
                {
                    "ticket_id": "999",
                    "original_query": "wrong source",
                    "rewritten_query": "User cannot access VPN after password reset.",
                    "topic_group": "Access",
                    "model_name": "llm1",
                    "router_mode": "llm_only",
                    "classifier_label": "Access",
                    "classifier_confidence": 0.99,
                    "memory_mode": "rag_bm25",
                    "kb_policy_ids": ["SHOULD_NOT_SURVIVE"],
                    "kb_categories": ["SHOULD_NOT_SURVIVE"],
                    "final_answer": "Reset the VPN session and sign in again.",
                    "reasoning": "This matches the VPN access recovery procedure.",
                    "predicted_category": "Access",
                    "resolution_steps": ["Disconnect VPN.", "Reconnect with SSO."],
                    "escalation_flags": {
                        "needs_human": False,
                        "needs_llm_escalation": False,
                        "policy_gap": False,
                        "reasons": [],
                    },
                }
            )

    monkeypatch.setattr(
        gen, "load_models", lambda sanity=True, slm_subset=None: {"llm1": FakeModel()}
    )

    result = gen.run_ticket_pipeline(
        ticket={"text": "vpn issue", "ticket_id": "9"},
        base_model_name="llm1",
        memory_mode="none",
        router_mode="llm_only",
    )

    assert result["generation_valid"] is True
    assert result["ticket_id"] == "9"
    assert result["original_query"] == "vpn issue"
    assert result["ticket_request"] == "User cannot access VPN after password reset."
    assert result["rewritten_query"] == result["ticket_request"]
    assert result["kb_policy_ids"] == []
    assert result["kb_categories"] == []
    assert result["parsed_output_before_validation"]["ticket_id"] == "999"


def test_truncated_json_does_not_pass_with_empty_steps(monkeypatch):
    monkeypatch.setattr(gen, "EncoderClassifier", DummyEncoderClassifier)

    class FakeModel:
        def __call__(self, prompt: str, **kwargs):
            return (
                '{"ticket_request":"User cannot access VPN.","final_answer":"Reset the VPN session and sign '
                'in again.","reasoning":"VPN access issue.","predicted_category":"Access","resolution_steps":['
            )

    monkeypatch.setattr(
        gen, "load_models", lambda sanity=True, slm_subset=None: {"llm1": FakeModel()}
    )

    result = gen.run_ticket_pipeline(
        ticket={"text": "vpn issue"},
        base_model_name="llm1",
        memory_mode="none",
        router_mode="llm_only",
    )

    assert result["generation_valid"] is False
    assert result["has_resolution_steps"] is False
    assert result["placeholder_answer"] is False
    assert result["final_answer"] == "Reset the VPN session and sign in again."


def test_natural_language_without_steps_is_invalid(monkeypatch):
    monkeypatch.setattr(gen, "EncoderClassifier", DummyEncoderClassifier)

    class FakeModel:
        def __call__(self, prompt: str, **kwargs):
            return "Answer: Reset the VPN connection and sign in again.\nReasoning: This addresses the issue."

    monkeypatch.setattr(
        gen, "load_models", lambda sanity=True, slm_subset=None: {"slm1": FakeModel()}
    )

    result = gen.run_ticket_pipeline(
        ticket={"text": "vpn issue"},
        base_model_name="slm1",
        memory_mode="none",
        router_mode="slm_only",
    )

    assert result["generation_valid"] is False
    assert result["has_resolution_steps"] is False


def test_generation_invalid_when_resolution_steps_empty(monkeypatch):
    monkeypatch.setattr(gen, "EncoderClassifier", DummyEncoderClassifier)

    class FakeModel:
        def __call__(self, prompt: str, **kwargs):
            return json.dumps(
                {
                    "ticket_request": "User cannot access VPN.",
                    "final_answer": "Reset the VPN session.",
                    "reasoning": "This should fix access.",
                    "predicted_category": "Access",
                    "resolution_steps": [],
                }
            )

    monkeypatch.setattr(
        gen, "load_models", lambda sanity=True, slm_subset=None: {"llm1": FakeModel()}
    )

    result = gen.run_ticket_pipeline(
        ticket={"text": "vpn issue"},
        base_model_name="llm1",
        memory_mode="none",
        router_mode="llm_only",
    )

    assert result["generation_valid"] is False
    assert result["has_resolution_steps"] is False


def test_missing_ticket_request_is_derived_from_ticket(monkeypatch):
    monkeypatch.setattr(gen, "EncoderClassifier", DummyEncoderClassifier)

    class FakeModel:
        def __call__(self, prompt: str, **kwargs):
            return json.dumps(
                {
                    "final_answer": "Reset the VPN session and sign in again.",
                    "reasoning": "This should resolve the issue.",
                    "predicted_category": "Access",
                    "resolution_steps": ["Disconnect VPN.", "Reconnect with SSO."],
                }
            )

    monkeypatch.setattr(
        gen, "load_models", lambda sanity=True, slm_subset=None: {"llm1": FakeModel()}
    )

    result = gen.run_ticket_pipeline(
        ticket={"text": "Please reset my VPN access after the password change."},
        base_model_name="llm1",
        memory_mode="none",
        router_mode="llm_only",
    )

    assert result["generation_valid"] is True
    assert result["ticket_request"] == "Please reset my VPN access after the password change."


def test_llm_only_pipeline_works_with_explicit_encoder_fallback(monkeypatch, tmp_path):
    missing_head_path = tmp_path / "missing_encoder_calibrated_head.npz"
    monkeypatch.setenv("ROUTERGYM_ENCODER_HEAD_MODE", "calibrated")
    monkeypatch.setenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK", "1")
    monkeypatch.setattr(enc, "CALIBRATED_HEAD_PATH", missing_head_path)
    monkeypatch.setattr(enc.EncoderClassifier, "_maybe_load_centroids", lambda self: None)

    class BrokenSentenceTransformer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no sentence transformer for test")

    class FakeModel:
        def __call__(self, prompt: str, **kwargs):
            return json.dumps(
                {
                    "final_answer": "Reset the VPN session and sign in again.",
                    "reasoning": "This resolves the access issue.",
                    "predicted_category": "Access",
                    "resolution_steps": ["Disconnect VPN.", "Reconnect with SSO."],
                }
            )

    monkeypatch.setattr(enc, "SentenceTransformer", BrokenSentenceTransformer)
    monkeypatch.setattr(gen, "EncoderClassifier", enc.EncoderClassifier)
    monkeypatch.setattr(
        gen, "load_models", lambda sanity=True, slm_subset=None: {"llm1": FakeModel()}
    )

    result = gen.run_ticket_pipeline(
        ticket={"text": "vpn issue", "ticket_id": "7"},
        base_model_name="llm1",
        memory_mode="none",
        router_mode="llm_only",
    )

    assert result["generation_valid"] is True
    assert result["ticket_request"] == "vpn issue"
    assert result["final_answer"] == "Reset the VPN session and sign in again."
    assert result["resolution_steps"] == ["Disconnect VPN.", "Reconnect with SSO."]
    assert result["raw_model_response_text"]
    assert result["classifier_backend"] == "encoder_centroid"
    assert result["parse_error"] is None
    assert result["validation_error"] is None


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
                {
                    "final_answer": "This is a longer detailed answer for the user",
                    "reasoning": "rr",
                    "resolution_steps": ["step1"],
                }
            )

    monkeypatch.setattr(gen, "EncoderClassifier", LowConfClassifier)
    monkeypatch.setattr(
        gen,
        "load_models",
        lambda sanity=True, slm_subset=None: {"slm1": BaseModel(), "llm1": EscalationModel()},
    )
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
    assert result["generation_valid"] is True
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
    assert result["generation_valid"] is True
