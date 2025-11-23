"""Knowledge base loader for markdown policies."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from RouterGym.utils.kb_utils import coerce_kb_hits

KB_ROOT = Path(__file__).resolve().parent


def load_kb(base_path: Path | None = None) -> Dict[str, str]:
    """Load all markdown files and return dict of filename -> text."""
    base = base_path or KB_ROOT
    articles: Dict[str, str] = {}
    for md_path in base.rglob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        articles[str(md_path)] = text
    return articles


def retrieve(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Simple retrieval by returning the first top_k articles."""
    kb = load_kb()
    items = list(kb.items())[:top_k]
    hits = [{"text": text, "source": fname} for fname, text in items]
    return coerce_kb_hits(hits)


__all__ = ["load_kb", "retrieve"]
