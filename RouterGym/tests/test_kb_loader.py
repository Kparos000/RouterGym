"""Tests for KB loader."""

import pytest

from RouterGym.data import kb_loader


@pytest.mark.requires_faiss
def test_load_and_retrieve_kb() -> None:
    """Ensure KB can be loaded and queried."""
    try:
        kb_loader.load_kb()
    except ImportError:
        pytest.skip("Dependencies for KB loader not installed")
    results = kb_loader.retrieve("password reset", top_k=2)
    assert isinstance(results, list)
    # If KB exists, expect entries; if empty KB, retrieval may be empty.
    assert all("filename" in r and "chunk" in r for r in results)
