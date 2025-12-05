"""Memory tests."""

from typing import Any

from RouterGym.memory.none import NoneMemory
from RouterGym.memory.transcript import TranscriptMemory
from RouterGym.memory.rag import RAGMemory
from RouterGym.memory.salience import SalienceGatedMemory
from RouterGym.memory import rag as rag_module


def test_none_memory() -> None:
    mem = NoneMemory()
    mem.load({"text": "text"})
    payload = mem.retrieve("text")
    assert payload.retrieved_context == ""
    assert payload.retrieval_cost_tokens == 0


def test_transcript_memory_add_and_get() -> None:
    mem = TranscriptMemory(k=2)
    mem.update("hello")
    mem.update("world")
    payload = mem.retrieve()
    assert "hello" in payload.retrieved_context
    assert "world" in payload.retrieved_context
    assert payload.relevance_score > 0


def test_rag_memory_retrieval(monkeypatch: Any) -> None:
    """RAGMemory should format retrieved snippets."""
    mem = RAGMemory(top_k=1)

    def fake_retrieve(query: str, top_k: int = 3):
        return [{"chunk": "snippet", "score": 1.0}]

    monkeypatch.setattr(rag_module.kb_loader, "retrieve", fake_retrieve)
    mem.load({"text": "ticket text"})
    mem.update("ticket text")
    payload = mem.retrieve("ticket text")
    assert "KB Reference" in payload.retrieved_context
    assert "snippet" in payload.retrieved_context
    assert payload.relevance_score == 1.0


def test_salience_gated_memory(monkeypatch: Any) -> None:
    """Salience memory gates based on length heuristic."""
    mem = SalienceGatedMemory(top_k=1)
    mem.update("short")
    mem.update("this is a much longer ticket text that should retrieve")
    payload = mem.retrieve()
    assert payload.retrieved_context
    assert len(payload.retrieved_context.splitlines()) <= 1
