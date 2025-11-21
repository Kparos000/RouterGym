"""Knowledge base loader for markdown policies."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


KB_ROOT = Path(__file__).resolve().parent


def load_kb(base_path: Path | None = None) -> List[Dict[str, str]]:
    """Load all markdown files and return list of dicts with filename and text."""
    base = base_path or KB_ROOT
    articles: List[Dict[str, str]] = []
    for md_path in base.rglob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        articles.append({"filename": str(md_path), "text": text})
    return articles


__all__ = ["load_kb"]
