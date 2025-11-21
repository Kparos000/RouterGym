"""Memory tests stubs."""

from RouterGym.memory.none import NullMemory
from RouterGym.memory.transcript import TranscriptMemory
from RouterGym.memory.rag import RAGMemory
from RouterGym.memory.salience import SalienceMemory


def test_null_memory_stub() -> None:
    """Placeholder test for null memory backend."""
    assert NullMemory().fetch() == ""


def test_transcript_memory_stub() -> None:
    """Placeholder test for transcript memory backend."""
    mem = TranscriptMemory()
    mem.add("hello")
    assert "hello" in mem.get_context()


def test_rag_memory_stub() -> None:
    """Placeholder test for RAG memory backend."""
    mem = RAGMemory()
    mem.upsert("doc")
    assert mem.retrieve("q")


def test_salience_memory_stub() -> None:
    """Placeholder test for salience memory backend."""
    mem = SalienceMemory()
    mem.upsert("doc")
    assert mem.retrieve("q")
