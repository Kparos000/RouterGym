"""Knowledge base loader for Markdown policies with FAISS retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

KB_ROOT = Path(__file__).resolve().parent / "policy_kb"
_MODEL_NAME = "all-MiniLM-L6-v2"
_index = None
_metadata: List[Dict[str, str]] = []
_encoder = None


def _load_encoder():
    """Load sentence transformer encoder."""
    global _encoder
    if _encoder is None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed")
        _encoder = SentenceTransformer(_MODEL_NAME)
    return _encoder


def _load_markdown_files(base_path: Path) -> List[Tuple[str, str]]:
    """Load all markdown files recursively."""
    files: List[Tuple[str, str]] = []
    for path in base_path.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        files.append((str(path), text))
    return files


def _chunk_markdown(text: str) -> List[str]:
    """Split markdown into chunk-sized paragraphs/headings."""
    parts = re.split(r"\n\s*\n", text)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned


def _embed_chunks(chunks: List[str]) -> np.ndarray:
    """Embed text chunks using sentence transformers."""
    encoder = _load_encoder()
    embeddings = encoder.encode(chunks, convert_to_numpy=True)
    return embeddings.astype("float32")


def load_kb(base_path: Path | None = None) -> None:
    """Load markdown KB, embed, and build a FAISS index."""
    global _index, _metadata
    if base_path is None:
        base_path = KB_ROOT
    if not base_path.exists():
        raise FileNotFoundError(f"KB path does not exist: {base_path}")
    docs = _load_markdown_files(base_path)
    chunks: List[str] = []
    meta: List[Dict[str, str]] = []
    for filename, text in docs:
        for chunk in _chunk_markdown(text):
            chunks.append(chunk)
            meta.append({"filename": filename, "chunk": chunk})
    if not chunks:
        _index = None
        _metadata = []
        return
    if faiss is None:
        raise ImportError("faiss is required to build the KB index")
    vectors = _embed_chunks(chunks)
    dim = vectors.shape[1]
    _index = faiss.IndexFlatIP(dim)
    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)
    _index.add(vectors)
    _metadata = meta


def retrieve(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Retrieve top-k KB chunks for a query."""
    if _index is None:
        load_kb()
    if _index is None:
        return []
    encoder = _load_encoder()
    vec = encoder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    scores, indices = _index.search(vec, top_k)
    results: List[Dict[str, str]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(_metadata):
            entry = _metadata[int(idx)]
            results.append({"filename": entry["filename"], "chunk": entry["chunk"], "score": float(score)})
    return results


__all__ = ["load_kb", "retrieve"]
