"""Integration tests for unified memory suite."""

from __future__ import annotations

from typing import Any

import pandas as pd

from RouterGym.memory.none import NoneMemory
from RouterGym.memory.transcript import TranscriptMemory
from RouterGym.memory.rag import RAGMemory
from RouterGym.memory.salience import SalienceGatedMemory
from RouterGym.memory.base import MemoryRetrieval
from RouterGym.memory import rag as rag_module
from RouterGym.routing.router_engine import RouterEngine
from RouterGym.experiments import run_grid


def _patch_rag(monkeypatch: Any) -> None:
    def fake_retrieve(query: str, top_k: int = 3):
        return [{"chunk": f"kb for {query}", "score": 0.75}]

    monkeypatch.setattr(rag_module.kb_loader, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag_module.kb_loader, "load_kb", lambda *args, **kwargs: {"doc": "kb"})
    monkeypatch.setattr(rag_module.kb_loader, "kb_hash", lambda: "hash")


def test_all_memory_modes_retrieve(monkeypatch: Any) -> None:
    _patch_rag(monkeypatch)
    ticket = {"text": "need vpn fix"}
    for cls in (NoneMemory, TranscriptMemory, RAGMemory, SalienceGatedMemory):
        mem = cls()
        mem.load(ticket)
        mem.update(ticket["text"])
        payload = mem.retrieve(ticket["text"])
        assert isinstance(payload.retrieved_context, str)
        assert "retrieval_cost_tokens" in payload.as_dict()


def test_salience_and_transcript_quality(monkeypatch: Any) -> None:
    _patch_rag(monkeypatch)
    transcript = TranscriptMemory(k=2)
    transcript.update("reset password")
    transcript.update("still locked")
    t_payload = transcript.retrieve()
    assert t_payload.relevance_score > 0

    salience = SalienceGatedMemory(top_k=1)
    salience.update("short note")
    salience.update("longer description with rare tokens xyzzy plugh")
    s_payload = salience.retrieve()
    assert "rare" in s_payload.retrieved_context.lower()
    assert s_payload.relevance_score >= 0


def test_router_engine_includes_memory_metadata(monkeypatch: Any) -> None:
    _patch_rag(monkeypatch)
    engine = RouterEngine("tfidf")
    memory_payload = MemoryRetrieval(
        retrieved_context="ctx",
        retrieval_metadata={"mode": "transcript"},
        retrieval_cost_tokens=42,
        relevance_score=0.8,
    )
    summary = engine.classify_ticket({"text": "vpn"}, memory_payload, "transcript")
    payload = summary.as_dict("tfidf")
    assert payload["memory_context_used"] == "ctx"
    assert payload["memory_relevance_score"] == 0.8
    assert payload["memory_cost_tokens"] == 42
    assert payload["memory_mode"] == "transcript"


def test_run_grid_logs_memory_fields(monkeypatch: Any) -> None:
    sample_record = {
        "ticket_id": 1,
        "router": "stub",
        "memory": "transcript",
        "memory_mode": "transcript",
        "model_used": "slm",
        "result": "success",
        "accuracy": 1.0,
        "groundedness": 0.5,
        "schema_validity": 1.0,
        "latency_ms": 10.0,
        "cost_usd": 0.001,
        "classifier_mode": "tfidf",
        "classifier_label": "access",
        "classifier_confidence": 0.7,
        "classifier_latency_ms": 1.0,
        "classifier_token_cost": 0.0,
        "classifier_accuracy": 1.0,
        "classifier_efficiency_score": 1.0,
        "json_valid": True,
        "schema_valid": True,
        "gold_category": "access",
        "predicted_category": "access",
        "memory_context": "ctx",
        "memory_cost_tokens": 10,
        "memory_relevance_score": 0.4,
        "retrieved_context_length": 3,
        "retrieval_latency_ms": 2.0,
    }

    monkeypatch.setattr(run_grid, "run_config", lambda *args, **kwargs: [sample_record])
    monkeypatch.setattr(run_grid, "load_models", lambda *args, **kwargs: {})

    df = run_grid.run_full_grid(
        tickets=[{"id": 1, "text": "hi", "category": "access", "gold_category": "access"}],
        routers=["llm_first"],
        memories=["transcript"],
        models=["slm1"],
        kb_retriever={},
        classifier_modes=["tfidf"],
    )
    assert isinstance(df, pd.DataFrame)
    assert "memory_cost_tokens" in df.columns
    assert df.loc[0, "memory_mode"] == "transcript"