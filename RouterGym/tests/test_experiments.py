"""Tests for experiment runner utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Dict
import sys

import pandas as pd

from RouterGym.experiments import run_grid
from RouterGym.routing.llm_first import LLMFirstRouter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import run_experiments  # type: ignore  # noqa: E402


def test_run_single_and_config(monkeypatch: Any) -> None:
    """Run single ticket and a minimal config without filesystem side effects."""
    ticket = {"id": 1, "text": "test ticket", "category": "access"}
    kb_retriever = None
    router = LLMFirstRouter()

    single = run_grid.run_single(ticket, router, "none", kb_retriever, models=None)
    assert single["router"] == "llm_first"
    assert single["memory"] == "none"

    tickets: List[Dict[str, Any]] = [ticket]
    config_results = run_grid.run_config("llm_first", "none", tickets, kb_retriever, models=None)
    assert isinstance(config_results, list)
    assert config_results


def test_run_pipeline_with_mocked_outputs(tmp_path: Path, monkeypatch: Any) -> None:
    """Run the top-level pipeline with mocked grid and analyzer to avoid heavy ops."""
    dummy_df = pd.DataFrame(
        [
            {
                "router": "llm_first",
                "memory": "none",
                "model": "slm1",
                "accuracy": 1.0,
                "cost_usd": 0.001,
                "latency_ms": 1.0,
            }
        ]
    )

    def fake_grid(**kwargs):
        return dummy_df

    monkeypatch.setattr(run_experiments.analyzer, "export_all_figures", lambda df, output_dir=None: None)
    monkeypatch.setattr(run_experiments.eval_stats, "export_anova_results", lambda df, filename=None: tmp_path / "anova.csv")

    run_experiments.run_pipeline(base_dir=tmp_path, grid_runner=fake_grid)
    assert (tmp_path / "results.csv").exists()


def test_sanity_forces_llm(monkeypatch: Any, tmp_path: Path) -> None:
    """Sanity mode should force LLM usage."""
    captured = {}

    def fake_grid_runner(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame([{"router": "llm_first"}])

    monkeypatch.setattr(run_experiments, "run_full_grid", fake_grid_runner)
    monkeypatch.setattr(run_experiments.analyzer, "export_all_figures", lambda df, output_dir=None: None)
    monkeypatch.setattr(run_experiments.eval_stats, "export_anova_results", lambda df, filename=None: tmp_path / "anova.csv")
    run_experiments.run_pipeline(base_dir=tmp_path, grid_runner=fake_grid_runner, force_llm=True)
    assert captured.get("force_llm") is True
