"""Memory tests."""

from typing import Any

from RouterGym.memory.none import NoneMemory
from RouterGym.memory.transcript import TranscriptMemory
from RouterGym.memory.rag import RAGMemory
from RouterGym.memory.salience import SalienceGatedMemory
from RouterGym.data import kb_loader


def test_none_memory() -> None:
    mem = NoneMemory()
    mem.add("text")
    assert mem.get_context() == ""


def test_transcript_memory_add_and_get() -> None:
    mem = TranscriptMemory()
    mem.add("hello")
    mem.add("world")
    ctx = mem.get_context()
    assert "hello" in ctx and "world" in ctx


def test_rag_memory_retrieval(monkeypatch: Any) -> None:
    """RAGMemory should format retrieved snippets."""
    mem = RAGMemory(top_k=1)

    def fake_retrieve(query: str, top_k: int = 3):
        return [{"chunk": "snippet", "score": 1.0}]

    monkeypatch.setattr(kb_loader, "retrieve", fake_retrieve)
    mem.add("ticket text")
    ctx = mem.get_context()
    assert "KB Reference" in ctx
    assert "snippet" in ctx


def test_salience_gated_memory(monkeypatch: Any) -> None:
    """Salience memory gates based on length heuristic."""
    mem = SalienceGatedMemory(top_k=1)

    def fake_retrieve(query: str, top_k: int = 3):
        return [{"chunk": "snippet", "score": 1.0}]

    monkeypatch.setattr(kb_loader, "retrieve", fake_retrieve)
    mem.add("short")
    # Short text should return recent note, not KB
    assert "Recent note" in mem.get_context()
    mem.add("this is a much longer ticket text that should retrieve")
    assert "KB Reference" in mem.get_context()
