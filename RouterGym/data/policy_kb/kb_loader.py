"""Knowledge base loader for markdown policies."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List

from RouterGym.utils.kb_utils import coerce_kb_hits

KB_ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "kb_embeddings.pkl"


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

def kb_hash(base_path: Path | None = None) -> str:
    base = base_path or KB_ROOT
    h = hashlib.sha256()
    for md_path in sorted(base.rglob("*.md")):
        h.update(md_path.read_bytes())
    return h.hexdigest()


def load_cached_embeddings() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            return pickle.loads(CACHE_PATH.read_bytes())
        except Exception:
            return {}
    return {}


def save_cached_embeddings(data: Dict[str, Any]) -> None:
    try:
        CACHE_PATH.write_bytes(pickle.dumps(data))
    except Exception:
        pass


__all__ = ["load_kb", "retrieve"]
