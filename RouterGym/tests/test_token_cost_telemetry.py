"""Token and cost telemetry tests."""

from __future__ import annotations

import json
from typing import Any, Dict

import RouterGym.agents.generator as gen
from RouterGym.engines.telemetry import (
    ModelCallTelemetry,
    aggregate_model_call_telemetry,
    estimate_text_tokens,
    invoke_model_with_telemetry,
)
from RouterGym.label_space import CANONICAL_LABELS
from RouterGym.memory.base import MemoryBase, MemoryRetrieval


def test_estimated_token_count_and_costs() -> None:
    class EstimatedModel:
        backend_used = "unknown"

        def __call__(self, prompt: str, **kwargs: Any) -> str:
            return "abcd1234"

    output_text, telemetry = invoke_model_with_telemetry(
        EstimatedModel(),
        "abcdefghijkl",
        model_key="slm1",
    )
    assert output_text == "abcd1234"
    assert telemetry.token_count_method == "estimated"
    assert telemetry.input_tokens == estimate_text_tokens("abcdefghijkl")
    assert telemetry.output_tokens == estimate_text_tokens("abcd1234")
    assert telemetry.total_cost_usd == telemetry.input_cost_usd + telemetry.output_cost_usd


def test_measured_token_count_path() -> None:
    class MeasuredModel:
        backend_used = "hf_inference"

        def generate(self, prompt: str, **kwargs: Any) -> str:
            self.last_usage = {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16}
            return "done"

    _, telemetry = invoke_model_with_telemetry(
        MeasuredModel(),
        "prompt",
        model_key="llm1",
    )
    assert telemetry.token_count_method == "measured"
    assert telemetry.input_tokens == 10
    assert telemetry.output_tokens == 6
    assert telemetry.total_tokens == 16
    assert telemetry.total_cost_usd > 0.0


def test_aggregate_model_call_telemetry_splits_slm_vs_llm() -> None:
    summary = aggregate_model_call_telemetry(
        [
            ModelCallTelemetry(
                model_key="slm1",
                model_name="mistralai/Mistral-7B-Instruct-v0.3",
                model_family="slm",
                backend_used="hf_inference",
                input_tokens=20,
                output_tokens=10,
                total_tokens=30,
                token_count_method="estimated",
                input_cost_usd=0.01,
                output_cost_usd=0.02,
                total_cost_usd=0.03,
            ),
            ModelCallTelemetry(
                model_key="llm1",
                model_name="Qwen/Qwen2-72B-Instruct",
                model_family="llm",
                backend_used="hf_inference",
                input_tokens=40,
                output_tokens=20,
                total_tokens=60,
                token_count_method="measured",
                input_cost_usd=0.20,
                output_cost_usd=0.30,
                total_cost_usd=0.50,
            ),
        ]
    )
    assert summary["total_tokens"] == 90
    assert summary["slm_total_tokens"] == 30
    assert summary["llm_total_tokens"] == 60
    assert summary["slm_cost_usd"] == 0.03
    assert summary["llm_cost_usd"] == 0.50
    assert summary["token_count_method_summary"] == "mixed"
    assert summary["pricing_version"] == "normalized_v1"


def test_run_ticket_pipeline_emits_telemetry_fields(monkeypatch: Any) -> None:
    class FakeClassifier:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.backend_name = "encoder_calibrated"

        def predict_proba(self, text: str) -> Dict[str, float]:
            probs = {label: 0.0 for label in CANONICAL_LABELS}
            probs["Access"] = 0.9
            remaining = (1.0 - probs["Access"]) / float(len(CANONICAL_LABELS) - 1)
            for label in CANONICAL_LABELS:
                if label != "Access":
                    probs[label] = remaining
            return probs

    class FakeMemory(MemoryBase):
        def update(self, item: Any, metadata: Dict[str, Any] | None = None) -> None:
            return None

        def summarize(self) -> str:
            return "memory summary"

        def retrieve(self, query: str | None = None) -> MemoryRetrieval:
            return MemoryRetrieval(
                retrieved_context="ctx",
                retrieval_metadata={"mode": "rag_dense", "query": query or "", "snippets": []},
                retrieval_cost_tokens=0,
                relevance_score=0.2,
                retrieval_latency_ms=1.0,
            )

    class FakeSLM:
        model_key = "slm1"
        backend_used = "hf_inference"

        def __call__(self, prompt: str, **kwargs: Any) -> str:
            self.last_usage = {"input_tokens": 25, "output_tokens": 15, "total_tokens": 40}
            return json.dumps(
                {
                    "rewritten_query": "rewrite",
                    "final_answer": "Detailed answer with enough information to satisfy the user request.",
                    "reasoning": "Reasoning text",
                    "predicted_category": "Access",
                    "resolution_steps": ["step 1", "step 2", "step 3"],
                }
            )

    monkeypatch.setattr(gen, "EncoderClassifier", FakeClassifier)
    monkeypatch.setattr(gen, "get_memory_class", lambda mode: FakeMemory)
    monkeypatch.setattr(
        gen,
        "load_models",
        lambda sanity=True, slm_subset=None: {"slm1": FakeSLM()},
    )
    monkeypatch.setattr(gen, "validate_agent_output", lambda payload: dict(payload))

    result = gen.run_ticket_pipeline(
        ticket={"text": "Please reset my VPN access", "ticket_id": "1"},
        base_model_name="slm1",
        memory_mode="rag_dense",
        router_mode="slm_only",
    )

    assert result["total_tokens"] == 40
    assert result["total_cost_usd"] > 0.0
    assert result["slm_cost_usd"] == result["total_cost_usd"]
    assert result["llm_cost_usd"] == 0.0
    assert result["pricing_version"] == "normalized_v1"
    assert result["token_count_method_summary"] == "measured"
    assert len(result["model_call_telemetry"]) == 1
