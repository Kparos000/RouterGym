"""Tests for KB loader with mocked dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from RouterGym.data import kb_loader


class DummyEncoder:
    def encode(self, texts, convert_to_numpy: bool = True):  # type: ignore[override]
        return np.array([[float(i + 1)] for i, _ in enumerate(texts)])


class DummyIndex:
    def __init__(self, dim: int) -> None:
        self.vectors: list[np.ndarray] = []

    def add(self, vectors) -> None:  # type: ignore[override]
        self.vectors.extend(list(vectors))

    def search(self, vec, k: int):
        n = min(k, len(self.vectors))
        idxs = np.arange(n).reshape(1, -1)
        scores = np.ones_like(idxs, dtype="float32")
        return scores, idxs


def test_load_kb_and_retrieve(tmp_path: Path, monkeypatch: Any) -> None:
    """Ensure KB loads and retrieval returns results."""
    # Create dummy markdown
    kb_dir = tmp_path / "policy_kb"
    kb_dir.mkdir()
    (kb_dir / "doc.md").write_text("# Title\n\nChunk one\n\nChunk two", encoding="utf-8")

    # Mock encoder and faiss
    monkeypatch.setattr(kb_loader, "SentenceTransformer", DummyEncoder)
    dummy_faiss = type(
        "F",
        (),
        {
            "IndexFlatIP": DummyIndex,
            "normalize_L2": lambda x: x,
        },
    )
    monkeypatch.setattr(kb_loader, "faiss", dummy_faiss)

    chunks, meta = kb_loader.load_kb(kb_dir)
    assert chunks
    assert meta

    embeddings = kb_loader.embed_kb(chunks)
    assert embeddings.shape[0] == len(chunks)

    kb_loader.refresh_index(kb_dir)
    results = kb_loader.retrieve("query", top_k=2)
    assert isinstance(results, list)
    # Retrieval should return at most top_k entries
    assert len(results) <= 2
