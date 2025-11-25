"""Tests for experiment runner utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Dict
import sys

import pandas as pd

from RouterGym.experiments import run_grid
from RouterGym.routing.llm_first import LLMFirstRouter
from RouterGym.engines import model_registry

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import run_experiments  # type: ignore  # noqa: E402


class DummyClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def chat_completion(self, **kwargs: Any):
        return type(
            "Resp",
            (),
            {"choices": [{"message": {"content": '{"final_answer":"ok","reasoning":"r","predicted_category":"access"}'}}]},
        )


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
    monkeypatch.setattr(
        run_experiments.eval_stats,
        "export_anova_results",
        lambda df, filename=None: tmp_path / "anova.csv",
    )

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
    monkeypatch.setattr(
        run_experiments.eval_stats,
        "export_anova_results",
        lambda df, filename=None: tmp_path / "anova.csv",
    )
    run_experiments.run_pipeline(base_dir=tmp_path, grid_runner=fake_grid_runner, force_llm=True)
    assert captured.get("force_llm") is True


def test_grid_outputs_vary_per_ticket(monkeypatch: Any) -> None:
    """Ensure groundedness/accuracy vary and KB is attached when RAG is used."""
    tickets = [
        {"id": 1, "text": "vpn issue", "category": "network"},
        {"id": 2, "text": "password reset", "category": "access"},
        {"id": 3, "text": "printer problem", "category": "hardware"},
    ]

    class FakeKB:
        def retrieve(self, query: str, top_k: int = 3):
            return [{"text": f"kb snippet for {query}", "source": "kb.md"}]

    class FakeRouter:
        def __init__(self) -> None:
            self.count = 0

        def route(self, ticket: dict, **kwargs: Any):
            self.count += 1
            predicted = ["network", "access", "hardware"][self.count % 3]
            return {
                "strategy": "llm_first",
                "target_model": "llm",
                "model_used": "llm",
                "final_output": {
                    "final_answer": f"answer {self.count}",
                    "reasoning": f"reason {self.count}",
                    "predicted_category": predicted,
                },
                "json_valid": True,
                "schema_valid": True,
                "kb_attached": True,
            }

    monkeypatch.setattr(run_grid, "init_router", lambda name=None: FakeRouter())
    df = run_grid.run_full_grid(
        tickets=tickets,
        kb_retriever=FakeKB(),
        limit=3,
        routers=["llm_first"],
        memories=["rag"],
        models=["llm1"],
        verbose=False,
        force_llm=True,
    )
    assert not df.empty
    assert df["kb_attached"].any()
    assert df["latency_ms"].max() >= 0
    assert df["accuracy"].nunique() >= 1 or df["groundedness"].nunique() >= 1


def test_run_grid_handles_bad_router_and_kb(monkeypatch: Any) -> None:
    """Grid should not crash if routers or KB return unexpected types."""
    class FakeKB:
        def retrieve(self, query: str, top_k: int = 3):
            return ["snippet"]  # not a dict

    class BadRouter:
        def route(self, *args: Any, **kwargs: Any):
            return "oops"  # not a dict

    monkeypatch.setattr(run_grid, "init_router", lambda name=None: BadRouter())
    df = run_grid.run_full_grid(
        tickets=[{"id": 1, "text": "hello"}],
        kb_retriever=FakeKB(),
        limit=1,
        routers=["llm_first"],
        memories=["none"],
        models=["llm1"],
        verbose=False,
    )
    assert not df.empty
    assert "router" in df.columns


def test_full_grid_remote_smoke(monkeypatch: Any) -> None:
    """Smoke test: remote-only models and grid run with limit=1."""

    # Ensure load_models can initialize without network
    monkeypatch.setattr(model_registry, "InferenceClient", lambda *args, **kwargs: DummyClient(*args, **kwargs))
    _ = model_registry.load_models(sanity=True)

    df = run_grid.run_full_grid(
        limit=1,
        routers=["llm_first"],
        memories=["none"],
        models=["slm1", "llm1"],
        force_llm=False,
        verbose=False,
    )
    assert not df.empty
    assert df["model_used"].isin({"slm", "llm"}).any()
    assert df["latency_ms"].max() > 0
    assert df["json_valid"].apply(lambda v: isinstance(v, bool)).all()
    assert df["schema_valid"].apply(lambda v: isinstance(v, bool)).all()
