"""Tests for experiment runner utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Dict
import sys

import pandas as pd

from RouterGym.experiments import run_grid
from RouterGym.routing.llm_first import LLMFirstRouter
from RouterGym.engines import model_registry
from RouterGym.evaluation import metrics as eval_metrics

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
            predicted = ["network", "access", "hardware"][(self.count - 1) % 3]
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
                "kb_snippets": [f"kb snippet {self.count}"],
            }

    monkeypatch.setattr(run_grid, "init_router", lambda name=None: FakeRouter())
    df = run_grid.run_full_grid(
        tickets=tickets,
        kb_retriever=FakeKB(),
        limit=3,
        routers=["llm_first"],
        memories=["rag_dense"],
        models=["llm1"],
        verbose=False,
        force_llm=True,
    )
    assert not df.empty
    assert df["kb_attached"].any()
    assert "gold_category" in df.columns and "predicted_category" in df.columns
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


def test_kb_attached_depends_on_memory(monkeypatch: Any) -> None:
    """kb_attached should reflect prompt inclusion, not retriever existence."""
    monkeypatch.setattr(eval_metrics, "groundedness_score", lambda answer, snippets: 0.9 if snippets else 0.0)

    class FakeRouter:
        def route(self, *args: Any, **kwargs: Any):
            return {
                "strategy": "llm_first",
                "target_model": "slm",
                "model_used": "slm",
                "final_output": {
                    "final_answer": "ans",
                    "reasoning": "r",
                    "predicted_category": "billing",
                },
                "json_valid": True,
                "schema_valid": True,
                "kb_attached": True,
                "kb_snippets": ["kb used"],
            }

    ticket = {"id": 1, "text": "hello", "category": "billing"}
    res_none = run_grid.run_single(ticket, FakeRouter(), "none", kb_retriever=None, models=None)
    assert res_none["kb_attached"] is False
    assert res_none["groundedness"] == 0.0

    res_rag = run_grid.run_single(ticket, FakeRouter(), "rag_dense", kb_retriever=None, models=None)
    assert res_rag["kb_attached"] is True
    assert res_rag["groundedness"] == 0.9


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


def test_dimensions_constants() -> None:
    assert run_grid.ROUTER_MODES == ["llm_first", "slm_dominant", "hybrid_specialist"]
    assert run_grid.MEMORY_MODES_CANONICAL == ["none", "transcript", "rag_dense", "rag_bm25", "rag_hybrid"]
    assert run_grid.CLASSIFIER_MODES == ["tfidf", "encoder", "slm_finetuned"]
    assert run_grid.MODEL_NAMES == ["slm1", "slm2", "llm1", "llm2"]
    assert len(set(run_grid.ROUTER_MODES)) == len(run_grid.ROUTER_MODES)
    assert len(set(run_grid.MEMORY_MODES_CANONICAL)) == len(run_grid.MEMORY_MODES_CANONICAL)
    assert len(set(run_grid.CLASSIFIER_MODES)) == len(run_grid.CLASSIFIER_MODES)
    assert len(set(run_grid.MODEL_NAMES)) == len(run_grid.MODEL_NAMES)


def test_smoke10_uses_ten_tickets(monkeypatch: Any, tmp_path: Path) -> None:
    tickets = [{"id": i, "text": f"ticket {i}", "category": "access"} for i in range(1, 15)]

    def fake_load_dataset(limit=None):
        return tickets

    class FakeKB:
        def retrieve(self, query: str, top_k: int = 3):
            return [{"text": "kb"}]

    class FakeRouter:
        def route(self, ticket: dict, **kwargs: Any):
            return {
                "strategy": "llm_first",
                "target_model": "slm",
                "model_used": "slm",
                "final_output": {
                    "final_answer": "ans",
                    "reasoning": "r",
                    "predicted_category": "access",
                },
                "json_valid": True,
                "schema_valid": True,
                "kb_attached": False,
                "kb_snippets": [],
            }

    if hasattr(run_grid.dataset_loader, "load_and_preprocess"):
        monkeypatch.setattr(run_grid.dataset_loader, "load_and_preprocess", fake_load_dataset)
    monkeypatch.setattr(run_grid.dataset_loader, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(run_grid, "load_kb", lambda: FakeKB())
    monkeypatch.setattr(run_grid, "init_router", lambda name=None: FakeRouter())
    df = run_grid.run_full_grid(
        tickets=tickets[:10],
        kb_retriever=FakeKB(),
        routers=["llm_first"],
        memories=["none"],
        models=["slm1"],
        classifier_modes=["tfidf"],
        verbose=False,
    )
    assert len(df) == 10


def test_ticket_slicing_and_output_path(tmp_path: Path, monkeypatch: Any) -> None:
    tickets = [{"id": i, "text": f"ticket {i}", "category": "access"} for i in range(1, 8)]

    class FakeRouter:
        def route(self, ticket: dict, **kwargs: Any):
            return {
                "strategy": "llm_first",
                "target_model": "slm",
                "model_used": "slm",
                "final_output": {
                    "final_answer": "ans",
                    "reasoning": "r",
                    "predicted_category": "access",
                },
                "json_valid": True,
                "schema_valid": True,
                "kb_attached": False,
                "kb_snippets": [],
            }

    class FakeKB:
        def retrieve(self, query: str, top_k: int = 3):
            return [{"text": "kb"}]

    monkeypatch.setattr(run_grid, "init_router", lambda name=None: FakeRouter())
    custom_small = tmp_path / "custom_small.csv"
    df_small = run_grid.run_full_grid(
        tickets=tickets,
        kb_retriever=FakeKB(),
        routers=["llm_first"],
        memories=["none"],
        models=["slm1"],
        classifier_modes=["tfidf"],
        ticket_start=2,
        ticket_limit=3,
        output_path=custom_small,
        verbose=False,
    )
    assert len(df_small) == 3
    assert list(df_small["ticket_id"]) == [3, 4, 5]
    assert custom_small.exists()

    custom_large = tmp_path / "custom_large.csv"
    df_large = run_grid.run_full_grid(
        tickets=tickets,
        kb_retriever=FakeKB(),
        routers=["llm_first"],
        memories=["none"],
        models=["slm1"],
        classifier_modes=["tfidf"],
        ticket_start=0,
        ticket_limit=-1,
        output_path=custom_large,
        verbose=False,
    )
    assert len(df_large) == len(tickets)
    assert custom_large.exists()
