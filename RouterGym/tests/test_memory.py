"""Memory tests."""

from typing import Any, Dict

from RouterGym.memory.none import NoneMemory
from RouterGym.memory.transcript import TranscriptMemory
from RouterGym.memory.rag import RAGMemory
from RouterGym.memory.bm25 import BM25Memory
from RouterGym.memory.hybrid import HybridRAGMemory
from RouterGym.data.policy_kb import kb_loader


def _patch_kb(monkeypatch: Any, kb: Dict[str, str]) -> None:
    def fake_load_kb(*args: Any, **kwargs: Any) -> Dict[str, str]:
        return kb

    def fake_retrieve(query: str, top_k: int = 3):
        tokens = set(query.lower().split())
        hits = []
        for path, text in kb.items():
            overlap = len(tokens & set(text.lower().split()))
            if overlap:
                hits.append({"chunk": text, "text": text, "score": float(overlap)})
        return hits[:top_k]

    monkeypatch.setattr(kb_loader, "load_kb", fake_load_kb)
    monkeypatch.setattr(kb_loader, "retrieve", fake_retrieve)
    monkeypatch.setattr(kb_loader, "kb_hash", lambda: "hash")
    monkeypatch.setattr(kb_loader, "load_cached_embeddings", lambda: {})
    monkeypatch.setattr(kb_loader, "save_cached_embeddings", lambda data: None)


def test_none_memory() -> None:
    mem = NoneMemory()
    mem.load({"text": "text"})
    payload = mem.retrieve("text")
    assert payload.retrieved_context == ""
    assert payload.retrieval_cost_tokens == 0
    assert payload.retrieval_latency_ms >= 0.0


def test_transcript_memory_add_and_get() -> None:
    mem = TranscriptMemory(k=2)
    mem.update("hello")
    mem.update("world")
    payload = mem.retrieve()
    assert "hello" in payload.retrieved_context
    assert "world" in payload.retrieved_context
    assert payload.relevance_score > 0
    assert payload.retrieval_latency_ms >= 0.0


def test_rag_dense_memory_retrieval(monkeypatch: Any) -> None:
    """Dense RAG should format snippets and carry scores."""
    kb = {"doc.md": "vpn reset instructions", "other.md": "printer offline"}
    _patch_kb(monkeypatch, kb)
    mem = RAGMemory(top_k=1)
    mem.load({"text": "vpn reset"})
    payload = mem.retrieve("vpn reset")
    assert "KB Reference" in payload.retrieved_context
    assert payload.relevance_score > 0.0
    assert payload.retrieval_latency_ms >= 0.0
    assert payload.retrieval_metadata["mode"] == "rag_dense"


def test_bm25_memory_overlap(monkeypatch: Any) -> None:
    kb = {"doc.md": "uniqueprintertoken network", "other.md": "something else"}
    _patch_kb(monkeypatch, kb)
    mem = BM25Memory(top_k=1)
    payload = mem.retrieve("uniqueprintertoken")
    assert "uniqueprintertoken" in payload.retrieved_context
    assert payload.memory_relevance_score > 0
    assert payload.retrieval_metadata["mode"] == "rag_bm25"


def test_hybrid_memory_fuses(monkeypatch: Any) -> None:
    kb = {"doc.md": "vpn reset instructions", "doc2.md": "printer offline guidance"}
    _patch_kb(monkeypatch, kb)
    mem = HybridRAGMemory(top_k=2, alpha=0.5)
    mem.update("vpn reset in transcript")
    result = mem.retrieve("vpn reset")
    assert result.retrieved_context
    bm25_latency = result.retrieval_metadata["bm25"].get("retrieval_latency_ms", 0.0)
    dense_latency = result.retrieval_metadata["dense"].get("retrieval_latency_ms", 0.0)
    assert result.retrieval_latency_ms >= max(bm25_latency, dense_latency)
    assert result.memory_relevance_score >= 0.0
