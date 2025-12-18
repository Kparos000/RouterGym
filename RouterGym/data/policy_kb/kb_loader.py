"""Knowledge base loader for markdown policies."""

from __future__ import annotations

import hashlib
import json
import re
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, TypedDict

from RouterGym.label_space import CANONICAL_LABEL_SET, canonicalize_label
from RouterGym.utils.kb_utils import coerce_kb_hits

KB_ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "kb_embeddings.pkl"
INDEX_PATH = KB_ROOT / "policy_kb_index.json"


class KBArticle(TypedDict):
    id: str
    category: str
    title: str
    summary: str
    content: str
    escalation_notes: str
    tags: List[str]
    path: str


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


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return slug.strip("_")


def _extract_section(sections: Dict[str, List[str]], key_match: Iterable[str]) -> str:
    for key in sections:
        norm = key.lower().strip()
        for target in key_match:
            if norm == target or target in norm:
                return "\n".join(sections[key]).strip()
    return ""


def _parse_markdown(md_path: Path) -> KBArticle:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    title = md_path.stem
    sections: Dict[str, List[str]] = {}
    current: str | None = None

    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip() or title
            current = None
        elif line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        else:
            if current is not None:
                sections[current].append(line)

    description = _extract_section(sections, ["description"])
    escalation_notes = _extract_section(sections, ["when to escalate"])

    category_dir = md_path.parent.name
    category = canonicalize_label(category_dir)
    rel_path = md_path.relative_to(KB_ROOT).as_posix()
    slug = _slugify(md_path.stem)
    article_id = f"{category_dir}.{slug}"
    tags_set: List[str] = []
    for part in [category_dir] + slug.split("_"):
        if part and part not in tags_set:
            tags_set.append(part)
    tags = tags_set[:8]

    return KBArticle(
        id=article_id,
        category=category,
        title=title,
        summary=description,
        content=text,
        escalation_notes=escalation_notes,
        tags=tags,
        path=rel_path,
    )


def build_kb_index(base_path: Path | None = None, output_path: Path | None = None) -> Path:
    """Parse markdown articles into a structured JSON index."""
    base = base_path or KB_ROOT
    out = output_path or INDEX_PATH
    articles: List[KBArticle] = []
    seen_ids: Set[str] = set()

    for md_path in sorted(base.rglob("*.md")):
        if md_path.name.startswith(".") or "__pycache__" in md_path.parts:
            continue
        article = _parse_markdown(md_path)
        if article["category"] not in CANONICAL_LABEL_SET:
            raise RuntimeError(f"Unexpected category '{article['category']}' in {md_path}")
        if article["id"] in seen_ids:
            raise RuntimeError(f"Duplicate KB article id detected: {article['id']} from {md_path}")
        seen_ids.add(article["id"])
        articles.append(article)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    return out.resolve()


def load_kb_index(base_path: Path | None = None, index_path: Path | None = None) -> List[KBArticle]:
    """Load the structured KB index, building it if missing."""
    base = base_path or KB_ROOT
    idx_path = index_path or INDEX_PATH
    if not idx_path.exists():
        build_kb_index(base_path=base, output_path=idx_path)
    data = json.loads(idx_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"KB index at {idx_path} is not a list")
    articles: List[KBArticle] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise RuntimeError(f"KB index entry is not a dict: {entry}")
        category = entry.get("category")
        if category not in CANONICAL_LABEL_SET:
            raise RuntimeError(f"KB index entry has unexpected category '{category}'")
        if not entry.get("id") or not entry.get("title") or not entry.get("content"):
            raise RuntimeError(f"KB index entry missing required fields: {entry}")
        # Coerce minimal shape
        article: KBArticle = {
            "id": str(entry["id"]),
            "category": str(category),
            "title": str(entry["title"]),
            "summary": str(entry.get("summary", "")),
            "content": str(entry["content"]),
            "escalation_notes": str(entry.get("escalation_notes", "")),
            "tags": [str(t) for t in entry.get("tags", [])],
            "path": str(entry.get("path", "")),
        }
        articles.append(article)
    return articles

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


__all__ = [
    "load_kb",
    "load_kb_index",
    "build_kb_index",
    "kb_hash",
    "load_cached_embeddings",
    "save_cached_embeddings",
    "retrieve",
]
