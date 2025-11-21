"""Knowledge base loader for Markdown policies with FAISS retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
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
KB_INDEX: Any = None
KB_CHUNKS: List[Dict[str, str]] = []
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
    """Split markdown into chunks using headings or blank lines."""
    chunks: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.strip().startswith("#") and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        elif line.strip() == "":
            if current:
                chunks.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return [c for c in chunks if c]


def load_kb(base_path: Path | None = None) -> Tuple[List[str], List[Dict[str, str]]]:
    """Load markdown KB and return chunks with metadata."""
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
    return chunks, meta


def embed_kb(chunks: List[str]) -> np.ndarray:
    """Embed text chunks using sentence transformers."""
    if not chunks:
        return np.zeros((0, 0), dtype="float32")
    encoder = _load_encoder()
    embeddings = encoder.encode(chunks, convert_to_numpy=True)
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> Any:
    """Create an in-memory FAISS index from embeddings."""
    if embeddings is None or embeddings.size == 0:
        return None
    if faiss is None:
        raise ImportError("faiss is required to build the KB index")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def refresh_index(base_path: Path | None = None) -> None:
    """Load KB, embed, and build FAISS index; update globals."""
    global KB_INDEX, KB_CHUNKS
    chunks, meta = load_kb(base_path)
    embeddings = embed_kb(chunks)
    KB_INDEX = build_faiss_index(embeddings)
    KB_CHUNKS = meta


def retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve top-k KB chunks for a query."""
    if KB_INDEX is None:
        refresh_index()
    if KB_INDEX is None:
        return []
    encoder = _load_encoder()
    vec = encoder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)
    scores, indices = KB_INDEX.search(vec, top_k)
    results: List[Dict[str, Any]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(KB_CHUNKS):
            entry = KB_CHUNKS[int(idx)]
            results.append({"filename": entry["filename"], "chunk": entry["chunk"], "score": float(score)})
    return results


__all__ = ["load_kb", "embed_kb", "build_faiss_index", "retrieve", "KB_INDEX", "KB_CHUNKS", "refresh_index"]
