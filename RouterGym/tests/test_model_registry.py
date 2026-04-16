"""Model registry tests for remote-only engines."""

from __future__ import annotations

from typing import Any, List

from RouterGym.engines import model_registry


class DummyClient:
    def __init__(
        self,
        model: str | None = None,
        token: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.init_args = {"model": model, "token": token, "timeout": timeout}
        self.calls: List[dict[str, Any]] = []

    def chat_completion(self, **kwargs: Any):
        self.calls.append(kwargs)
        return type(
            "Resp",
            (),
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"final_answer":"ok","reasoning":"r","predicted_category":"access"}'
                        }
                    }
                ]
            },
        )


def test_load_models_all_remote(monkeypatch: Any) -> None:
    """All model entries should be remote engines and respect subsets."""
    created: list[str] = []

    def fake_client(
        model: str | None = None,
        token: str | None = None,
        timeout: int | None = None,
    ):
        created.append(model or "")
        return DummyClient(model=model, token=token, timeout=timeout)

    monkeypatch.setattr(model_registry, "InferenceClient", fake_client)
    monkeypatch.setenv("ROUTERGYM_MODEL_BACKEND", "hf_inference")
    models = model_registry.load_models(sanity=False)
    assert set(models.keys()) == {"slm1", "slm2", "llm1", "llm2"}
    assert all(isinstance(engine, model_registry.RemoteInferenceEngine) for engine in models.values())
    assert "openai/gpt-oss-20b" in created
    assert "Qwen/Qwen2.5-14B-Instruct" in created


def test_sanity_includes_slm_and_llm(monkeypatch: Any) -> None:
    """Sanity mode should still produce one SLM and one LLM engine."""
    monkeypatch.setattr(model_registry, "InferenceClient", lambda *args, **kwargs: DummyClient(*args, **kwargs))
    monkeypatch.setenv("ROUTERGYM_MODEL_BACKEND", "hf_inference")
    models = model_registry.load_models(sanity=True)
    kinds = {getattr(engine, "kind", "") for engine in models.values()}
    assert "slm" in kinds
    assert "llm" in kinds
    assert all(isinstance(engine, model_registry.RemoteInferenceEngine) for engine in models.values())


def test_remote_engine_response_format(monkeypatch: Any) -> None:
    """Ensure response_format is passed to chat_completion."""
    captured = {}

    class CapturingClient(DummyClient):
        def chat_completion(self, **kwargs: Any):
            captured.update(kwargs)
            return super().chat_completion(**kwargs)

    monkeypatch.setattr(model_registry, "InferenceClient", lambda *args, **kwargs: CapturingClient(*args, **kwargs))
    engine = model_registry.RemoteInferenceEngine("model", token="tkn", max_retries=0)
    _ = engine.generate("hi")
    assert captured.get("model") == "model"
    assert captured.get("response_format", {}).get("type") == "json_object"
    assert captured.get("messages")
    assert captured.get("max_tokens") is not None
    assert captured.get("temperature") is not None
    assert engine.last_endpoint_path == "chat_completion"


def test_remote_engine_falls_back_to_text_generation(monkeypatch: Any) -> None:
    class FallbackClient(DummyClient):
        def chat_completion(self, **kwargs: Any):
            raise RuntimeError("chat_completion unavailable")

        def text_generation(self, prompt: str, **kwargs: Any):
            self.calls.append({"prompt": prompt, **kwargs})
            return '{"final_answer":"ok","reasoning":"r","predicted_category":"access"}'

    monkeypatch.setattr(model_registry, "InferenceClient", lambda *args, **kwargs: FallbackClient(*args, **kwargs))
    engine = model_registry.RemoteInferenceEngine("model", token="tkn", max_retries=0)
    text = engine.generate("hi")
    assert "final_answer" in text
    assert engine.last_endpoint_path == "text_generation"


def test_remote_engine_falls_back_to_plain_chat_when_json_mode_fails(monkeypatch: Any) -> None:
    class PlainChatFallbackClient(DummyClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.chat_calls = 0

        def chat_completion(self, **kwargs: Any):
            self.chat_calls += 1
            if self.chat_calls == 1:
                raise RuntimeError("json mode unsupported")
            self.calls.append(kwargs)
            return type(
                "Resp",
                (),
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"final_answer":"ok","reasoning":"r","predicted_category":"access"}'
                            }
                        }
                    ]
                },
            )

    monkeypatch.setattr(model_registry, "InferenceClient", lambda *args, **kwargs: PlainChatFallbackClient(*args, **kwargs))
    engine = model_registry.RemoteInferenceEngine("model", token="tkn", max_retries=0)
    text = engine.generate("hi", max_new_tokens=80)
    assert "final_answer" in text
    assert engine.last_endpoint_path == "chat_completion_plain"


def test_get_repair_model(monkeypatch: Any) -> None:
    """Repair model should return strongest LLM remote engine."""
    monkeypatch.setattr(model_registry, "InferenceClient", lambda *args, **kwargs: DummyClient(*args, **kwargs))
    engine = model_registry.get_repair_model()
    assert isinstance(engine, model_registry.RemoteInferenceEngine)
    assert engine.kind == "llm"


def test_llm_entries_match_frozen_models() -> None:
    assert model_registry.LLM_MODELS["llm1"].hf_id == "openai/gpt-oss-20b"
    assert model_registry.LLM_MODELS["llm2"].hf_id == "Qwen/Qwen2.5-14B-Instruct"
