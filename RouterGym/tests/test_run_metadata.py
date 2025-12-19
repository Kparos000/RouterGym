"""Tests for run metadata logging in grid experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from RouterGym.experiments import run_grid


def test_log_run_metadata_smoke(tmp_path: Path, monkeypatch: Any) -> None:
    tickets = [{"id": i, "text": f"ticket {i}", "category": "access"} for i in range(1, 6)]

    class FakeRouter:
        def route(self, ticket: dict, **kwargs: Any):
            return {
                "strategy": "llm_first",
                "target_model": "slm",
                "model_used": "slm",
                "final_output": {"final_answer": "ans", "reasoning": "r", "predicted_category": "access"},
                "json_valid": True,
                "schema_valid": True,
                "kb_attached": False,
                "kb_snippets": [],
            }

    class FakeKB:
        def retrieve(self, query: str, top_k: int = 3):
            return [{"text": "kb"}]

    log_path = tmp_path / "run_metadata.csv"
    monkeypatch.setattr(run_grid, "init_router", lambda name=None: FakeRouter())
    monkeypatch.setattr(run_grid, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_grid, "_results_output_path", lambda ticket_count, override=None: tmp_path / "results.csv" if override is None else override)
    df = run_grid.run_full_grid(
        tickets=tickets,
        kb_retriever=FakeKB(),
        routers=["slm_only"],
        memories=["none"],
        models=["slm1"],
        classifier_modes=["tfidf"],
        output_path=tmp_path / "results.csv",
    )
    assert len(df) == len(tickets)

    assert log_path.exists()
    with log_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # header + one row
    header = rows[0]
    data = rows[1]
    row_dict = dict(zip(header, data))
    assert row_dict["backend"]
    assert row_dict["num_tickets"] == str(len(tickets))
    assert float(row_dict["wall_clock_seconds"]) >= 0.0
