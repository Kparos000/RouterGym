"""Knowledge base loader for Markdown policies with FAISS indexing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


class MarkdownKB:
    """Simple Markdown knowledge base with embedding + FAISS retrieval."""

    def __init__(self, folder_path: str | Path, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"KB folder does not exist: {self.folder_path}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name) if SentenceTransformer else None
        self.index = None
        self.paragraphs: List[str] = []

    def load_markdown_kb(self) -> List[str]:
        """Load markdown files and split into paragraphs."""
        paragraphs: List[str] = []
        for md_file in self.folder_path.glob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
            paragraphs.extend(chunks)
        self.paragraphs = paragraphs
        return paragraphs

    def embed(self) -> np.ndarray:
        """Embed paragraphs using sentence-transformers (or zero fallback)."""
        if not self.paragraphs:
            self.load_markdown_kb()
        if self.model:
            return np.array(self.model.encode(self.paragraphs))
        # Fallback: deterministic dummy vectors
        return np.zeros((len(self.paragraphs), 384), dtype="float32")

    def build_index(self) -> None:
        """Build a FAISS index from embeddings."""
        if faiss is None:
            raise ImportError("faiss is required to build the knowledge base index.")
        embeddings = self.embed().astype("float32")
        dim = embeddings.shape[1] if embeddings.size else 0
        self.index = faiss.IndexFlatL2(dim)
        if embeddings.size:
            self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top paragraphs for a query."""
        if self.index is None:
            self.build_index()
        if self.index is None:
            return []
        if self.model:
            query_vec = np.array(self.model.encode([query])).astype("float32")
        else:
            query_vec = np.zeros((1, 384), dtype="float32")
        distances, indices = self.index.search(query_vec, top_k)  # type: ignore[assignment]
        results: List[Tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.paragraphs):
                results.append((self.paragraphs[idx], float(dist)))
        return results

