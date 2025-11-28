"""Utilities for normalizing KB retrieval hits."""

from __future__ import annotations

from typing import Any, Dict, List


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


__all__ = ["coerce_kb_hits"]
