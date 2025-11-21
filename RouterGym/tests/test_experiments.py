"""Tests for experiment runner utilities."""

from __future__ import annotations

from typing import Any, List, Dict

from RouterGym.experiments import run_grid
from RouterGym.routing.llm_first import LLMFirstRouter


def test_run_single_and_config(monkeypatch: Any) -> None:
    """Run single ticket and a minimal config without filesystem side effects."""
    ticket = {"id": 1, "text": "test ticket", "category": "access"}
    kb_retriever = None
    router = LLMFirstRouter()

    single = run_grid.run_single(ticket, router, "none", kb_retriever)
    assert single["router"] == "llm_first"
    assert single["memory"] == "none"

    tickets: List[Dict[str, Any]] = [ticket]
    config_results = run_grid.run_config("llm_first", "none", tickets, kb_retriever)
    assert isinstance(config_results, list)
    assert config_results
