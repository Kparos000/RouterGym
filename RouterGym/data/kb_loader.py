"""Compatibility wrapper to policy_kb loader."""

from RouterGym.data.policy_kb.kb_loader import load_kb  # type: ignore


def retrieve(query: str, top_k: int = 3):
    """Return the first top_k docs as a simple list."""
    kb = load_kb()
    items = list(kb.items())[:top_k]
    return [{"filename": fname, "text": text, "chunk": text, "score": 1.0} for fname, text in items]


__all__ = ["load_kb", "retrieve"]
