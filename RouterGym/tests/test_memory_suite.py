"""Integration tests for unified memory suite."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from RouterGym.memory import MEMORY_MODES, get_memory_class, resolve_memory_mode
from RouterGym.memory.base import MemoryRetrieval
from RouterGym.data.policy_kb import kb_loader


def _synthetic_kb() -> Dict[str, str]:
    return {
        "kb/vpn.md": "vpn reset instructions token",
        "kb/printer.md": "printer offline troubleshooting uniqueprintertoken",
        "kb/hr.md": "pto policy and leave form guidance",
    }


def _patch_kb(monkeypatch: Any, kb: Dict[str, str]) -> None:
    def fake_load_kb(*args: Any, **kwargs: Any) -> Dict[str, str]:
        return kb

    def fake_retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_tokens = set(query.lower().split())
        hits: List[Dict[str, Any]] = []
        for path, text in kb.items():
            overlap = len(q_tokens & set(text.lower().split()))
            if overlap:
                hits.append({"chunk": text, "text": text, "score": float(overlap), "source": path})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]

    monkeypatch.setattr(kb_loader, "load_kb", fake_load_kb)
    monkeypatch.setattr(kb_loader, "retrieve", fake_retrieve)
    monkeypatch.setattr(kb_loader, "kb_hash", lambda: "hash")
    monkeypatch.setattr(kb_loader, "load_cached_embeddings", lambda: {})
    monkeypatch.setattr(kb_loader, "save_cached_embeddings", lambda data: None)


@pytest.mark.parametrize("memory_mode", MEMORY_MODES)
def test_memory_modes_retrieve(monkeypatch: Any, memory_mode: str) -> None:
    kb = _synthetic_kb()
    _patch_kb(monkeypatch, kb)
    memory_cls = get_memory_class(memory_mode)
    assert memory_cls is not None
    mem = memory_cls()
    ticket = {"text": "vpn reset now"}
    mem.load(ticket)
    mem.update(ticket["text"])
    payload: MemoryRetrieval = mem.retrieve(ticket["text"])
    data = payload.as_dict()
    for key in ["retrieved_context", "retrieval_latency_ms", "memory_cost_tokens", "memory_relevance_score"]:
        assert key in data
    assert isinstance(data["retrieved_context"], str)
    assert data["retrieval_latency_ms"] >= 0.0


def test_bm25_unique_keyword(monkeypatch: Any) -> None:
    kb = _synthetic_kb()
    _patch_kb(monkeypatch, kb)
    memory_cls = get_memory_class(resolve_memory_mode("rag_bm25"))
    mem = memory_cls()
    payload = mem.retrieve("uniqueprintertoken")
    assert "uniqueprintertoken" in payload.retrieved_context
    assert payload.memory_relevance_score > 0.0


def test_hybrid_latency_and_context(monkeypatch: Any) -> None:
    kb = _synthetic_kb()
    _patch_kb(monkeypatch, kb)
    memory_cls = get_memory_class(resolve_memory_mode("rag_hybrid"))
    mem = memory_cls()
    mem.update("vpn reset transcript mention")
    payload = mem.retrieve("vpn reset")
    assert payload.retrieved_context  # some context returned
    bm25_latency = payload.retrieval_metadata["bm25"].get("retrieval_latency_ms", 0.0)
    dense_latency = payload.retrieval_metadata["dense"].get("retrieval_latency_ms", 0.0)
    assert payload.retrieval_latency_ms >= max(bm25_latency, dense_latency)
    assert payload.memory_relevance_score >= 0.0
