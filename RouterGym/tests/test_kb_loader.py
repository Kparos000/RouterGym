"""Tests for KB loader."""

from __future__ import annotations

from RouterGym.data import kb_loader


def test_load_kb_basic(tmp_path):
    kb_dir = tmp_path / "policy_kb"
    kb_dir.mkdir()
    (kb_dir / "doc.md").write_text("content", encoding="utf-8")
    kb = kb_loader.load_kb(kb_dir)
    assert len(kb) == 1


def test_retrieve_returns_entries(tmp_path):
    kb_dir = tmp_path / "policy_kb"
    kb_dir.mkdir()
    (kb_dir / "doc1.md").write_text("content1", encoding="utf-8")
    (kb_dir / "doc2.md").write_text("content2", encoding="utf-8")
    kb_loader.load_kb(kb_dir)  # build mapping
    results = kb_loader.retrieve("query", top_k=1)
    assert isinstance(results, list)
    assert results
