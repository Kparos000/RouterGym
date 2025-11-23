"""Model registry tests."""

from __future__ import annotations

from typing import Any

from RouterGym.engines import model_registry


class DummyPipeline:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return f"pipeline:{prompt}"


class DummyClient:
    def __init__(self, model: str) -> None:
        self.model = model

    def chat_completion(self, **kwargs: Any):
        return type("Resp", (), {"choices": [{"message": {"content": f"remote:{self.model}"}}]})


def test_sanity_loads_small_model(monkeypatch: Any) -> None:
    calls = {}

    def fake_pipeline(task: str, model: str, tokenizer: Any = None, device: int = -1):
        calls["model"] = model
        return DummyPipeline()

    monkeypatch.setattr(model_registry, "AutoTokenizer", type("Tok", (), {"from_pretrained": staticmethod(lambda m: m)}))  # type: ignore[arg-type]
    monkeypatch.setattr(model_registry, "AutoModelForCausalLM", type("Model", (), {"from_pretrained": staticmethod(lambda m, torch_dtype=None: m)}))  # type: ignore[arg-type]
    monkeypatch.setattr(model_registry, "pipeline", fake_pipeline)
    models = model_registry.load_models(sanity=True)
    assert len(models) == 1
    assert "phi-3" in str(calls.get("model", "")).lower()


def test_large_models_use_remote(monkeypatch: Any) -> None:
    remote_called = []

    def fake_client(model: str, token: Any = None):
        remote_called.append(model)
        return DummyClient(model)

    def fake_pipeline(task: str, model: str, tokenizer: Any = None, device: int = -1):
        return DummyPipeline()

    monkeypatch.setattr(model_registry, "AutoTokenizer", type("Tok", (), {"from_pretrained": staticmethod(lambda m: None)}))
    monkeypatch.setattr(model_registry, "AutoModelForCausalLM", type("Model", (), {"from_pretrained": staticmethod(lambda m, torch_dtype=None: None)}))
    monkeypatch.setattr(model_registry, "InferenceClient", fake_client)
    monkeypatch.setattr(model_registry, "pipeline", fake_pipeline)
    models = model_registry.load_models(sanity=False)
    assert any("72b" in m.lower() for m in remote_called)
    assert any("llama-3" in m.lower() for m in remote_called)
    assert any(name.startswith("slm") for name in models)
    assert any(name.startswith("llm") for name in models)


def test_get_repair_model(monkeypatch: Any) -> None:
    """Repair model should be a remote engine, not a local pipeline."""
    class DummyClient:
        def __init__(self, model: str) -> None:
            self.model = model

        def text_generation(self, prompt: str, **kwargs: Any) -> str:
            return f"remote:{self.model}"

    called = {}

    def fake_client(model: str, token: Any = None):
        called["model"] = model
        return DummyClient(model)

    monkeypatch.setattr(model_registry, "InferenceClient", fake_client)
    engine = model_registry.get_repair_model()
    assert hasattr(engine, "generate")
    assert "72b" in called.get("model", "").lower() or "70b" in called.get("model", "").lower()
