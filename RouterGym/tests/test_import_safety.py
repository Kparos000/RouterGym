"""Structural import-safety tests for lazy module boundaries."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import huggingface_hub


def _fresh_import(module_name: str):
    for name in [key for key in sys.modules if key == module_name or key.startswith(f"{module_name}.")]:
        sys.modules.pop(name, None)
    return importlib.import_module(module_name)


def test_import_routing_package_does_not_pull_generator() -> None:
    _fresh_import("RouterGym.routing")
    sys.modules.pop("RouterGym.agents.generator", None)
    module = importlib.import_module("RouterGym.routing")
    assert hasattr(module, "__all__")
    assert "RouterGym.agents.generator" not in sys.modules


def test_import_generator_does_not_import_memory_backends() -> None:
    _fresh_import("RouterGym.memory")
    sys.modules.pop("RouterGym.memory.rag", None)
    sys.modules.pop("RouterGym.memory.bm25", None)
    sys.modules.pop("RouterGym.memory.hybrid", None)
    _fresh_import("RouterGym.agents.generator")
    assert "RouterGym.memory.rag" not in sys.modules
    assert "RouterGym.memory.bm25" not in sys.modules
    assert "RouterGym.memory.hybrid" not in sys.modules


def test_import_model_registry_has_no_env_or_client_side_effects(monkeypatch: Any) -> None:
    calls = {"dotenv": 0, "client": 0}

    fake_dotenv = types.ModuleType("dotenv")

    def fake_load_dotenv(*args: Any, **kwargs: Any) -> bool:
        calls["dotenv"] += 1
        return True

    class FailIfConstructed:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls["client"] += 1

    fake_dotenv.load_dotenv = fake_load_dotenv  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setattr(huggingface_hub, "InferenceClient", FailIfConstructed)

    module = _fresh_import("RouterGym.engines.model_registry")
    assert module.SLM_MODELS
    assert calls["dotenv"] == 0
    assert calls["client"] == 0


def test_import_run_agentic_eval_has_no_runtime_side_effects(monkeypatch: Any) -> None:
    import RouterGym.agents.generator as generator
    import RouterGym.data.tickets.dataset_loader as dataset_loader

    calls = {"dataset": 0, "pipeline": 0}

    def fail_dataset(*args: Any, **kwargs: Any) -> Any:
        calls["dataset"] += 1
        raise AssertionError("load_dataset should not be called during import")

    def fail_pipeline(*args: Any, **kwargs: Any) -> Any:
        calls["pipeline"] += 1
        raise AssertionError("run_ticket_pipeline should not be called during import")

    monkeypatch.setattr(dataset_loader, "load_dataset", fail_dataset)
    monkeypatch.setattr(generator, "run_ticket_pipeline", fail_pipeline)

    reloaded = _fresh_import("RouterGym.experiments.run_agentic_eval")
    assert hasattr(reloaded, "run_agentic_eval")
    assert calls["dataset"] == 0
    assert calls["pipeline"] == 0
