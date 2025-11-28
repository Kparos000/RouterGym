"""Utilities for normalizing KB retrieval hits."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def coerce_kb_hits(hits: Any) -> List[Dict[str, str]]:
    """Ensure KB hits are a list of dicts with text and source keys."""
    normalized: List[Dict[str, str]] = []
    if not hits:
        return normalized
    for h in hits:
        if isinstance(h, dict):
            normalized.append(
                {
                    "text": str(h.get("text") or h.get("chunk") or h.get("content") or ""),
                    "source": str(h.get("source") or h.get("path") or h.get("id") or h.get("filename") or ""),
                }
            )
        else:
            normalized.append({"text": str(h), "source": ""})
    return normalized


def rerank_and_trim_hits(query: str, hits: List[Dict[str, str]], top_k: int = 3, max_chars: int = 400) -> List[str]:
    """Simple token-overlap reranker + trimmer for KB hits."""
    if not hits:
        return []
    q_tokens = set((query or "").lower().split())

    def score(hit: Dict[str, str]) -> Tuple[int, int]:
        text = hit.get("text", "")
        t_tokens = set(text.lower().split())
        overlap = len(q_tokens & t_tokens)
        return (overlap, -len(text))

    ranked = sorted(hits, key=score, reverse=True)
    snippets: List[str] = []
    for hit in ranked[:top_k]:
        text = hit.get("text", "")
        snippets.append(text[:max_chars])
    return snippets


__all__ = ["coerce_kb_hits", "rerank_and_trim_hits"]
