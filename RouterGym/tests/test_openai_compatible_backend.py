"""OpenAI-compatible backend tests."""

from __future__ import annotations

import json
from typing import Any

from RouterGym.engines import model_registry
from RouterGym.engines.openai_compatible import OpenAICompatibleEngine
from RouterGym.scripts import smoke_openai_compatible_model as smoke_script


class DummyClient:
    def __init__(self, model: str | None = None, token: str | None = None, timeout: int | None = None) -> None:
        self.model = model
        self.token = token
        self.timeout = timeout


def test_backend_selection_accepts_openai_aliases(monkeypatch: Any) -> None:
    monkeypatch.setenv("ROUTERGYM_MODEL_BACKEND", "openai_compatible")
    assert model_registry.get_model_backend() == "openai_compatible"
    monkeypatch.setenv("ROUTERGYM_MODEL_BACKEND", "vllm_openai")
    assert model_registry.get_model_backend() == "openai_compatible"


def test_load_models_openai_backend_uses_openai_for_llms(monkeypatch: Any) -> None:
    monkeypatch.setattr(model_registry, "InferenceClient", lambda *args, **kwargs: DummyClient(*args, **kwargs))
    monkeypatch.setenv("ROUTERGYM_MODEL_BACKEND", "openai_compatible")
    monkeypatch.setenv("ROUTERGYM_OPENAI_BASE_URL", "http://localhost:9000")
    monkeypatch.setenv("ROUTERGYM_OPENAI_API_KEY", "secret-key")
    models = model_registry.load_models(sanity=False)
    assert isinstance(models["slm1"], model_registry.RemoteInferenceEngine)
    assert isinstance(models["slm2"], model_registry.RemoteInferenceEngine)
    assert isinstance(models["llm1"], OpenAICompatibleEngine)
    assert isinstance(models["llm2"], OpenAICompatibleEngine)
    assert models["llm1"].base_url == "http://localhost:9000/v1"
    assert models["llm1"].api_key == "secret-key"
    assert models["llm1"].backend_used == "openai_compatible"


def test_openai_compatible_engine_request_path_and_usage(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"final_answer":"ok","reasoning":"r","predicted_category":"Access"}'
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
                }
            ).encode("utf-8")

    def fake_urlopen(req: Any, timeout: int = 0) -> FakeResponse:
        captured["url"] = req.full_url
        captured["authorization"] = req.headers.get("Authorization")
        captured["content_type"] = req.headers.get("Content-type")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    from RouterGym.engines import openai_compatible as openai_module

    monkeypatch.setattr(openai_module.request, "urlopen", fake_urlopen)
    engine = OpenAICompatibleEngine(
        "mistralai/Mistral-Small-24B-Instruct-2501",
        model_key="llm1",
        base_url="http://localhost:8000",
        api_key="test-key",
        max_retries=0,
    )
    output = engine.generate("hello", max_new_tokens=50, temperature=0.0)
    assert "final_answer" in output
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["content_type"] == "application/json"
    assert captured["body"]["model"] == "mistralai/Mistral-Small-24B-Instruct-2501"
    assert engine.last_endpoint_path == "openai_chat_completions"
    assert engine.last_usage == {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18}


def test_openai_compatible_engine_records_missing_content_error(monkeypatch: Any) -> None:
    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning": "thinking",
                            },
                            "finish_reason": "length",
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(req: Any, timeout: int = 0) -> FakeResponse:
        del req, timeout
        return FakeResponse()

    from RouterGym.engines import openai_compatible as openai_module

    monkeypatch.setattr(openai_module.request, "urlopen", fake_urlopen)
    engine = OpenAICompatibleEngine(
        "openai/gpt-oss-20b",
        model_key="llm1",
        base_url="http://localhost:8000",
        api_key="test-key",
        max_retries=0,
    )
    output = engine.generate("hello", max_new_tokens=50, temperature=0.0)
    assert "LLM unavailable" in output
    assert engine.last_endpoint_path == "openai_chat_completions"
    assert engine.last_error is not None
    assert engine.last_error["error_type"] == "ValueError"
    assert engine.last_error["phase"] == "response_parsing"
    assert engine.last_error["finish_reason"] == "length"
    assert engine.last_error["reasoning_present"] == "True"


def test_smoke_script_surfaces_backend_error_on_failure(monkeypatch: Any) -> None:
    class FakeEngine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            self.last_endpoint_path = ""
            self.last_error = {
                "error_type": "HTTPError",
                "message": "HTTP Error 401: Unauthorized",
            }

        def generate(self, *args: Any, **kwargs: Any) -> str:
            del args, kwargs
            return json.dumps(
                {
                    "final_answer": "LLM unavailable",
                    "reasoning": "timeout or error",
                    "predicted_category": "unknown",
                }
            )

    monkeypatch.setattr(smoke_script, "OpenAICompatibleEngine", FakeEngine)
    result = smoke_script.run_smoke_test(model_key="llm1", base_url="http://localhost:8123")
    assert result["status"] == "failure"
    assert result["model_id"] == "openai/gpt-oss-20b"
    assert result["max_new_tokens"] == 512
    assert result["backend_error"] == {
        "error_type": "HTTPError",
        "message": "HTTP Error 401: Unauthorized",
    }


def test_smoke_script_dry_run() -> None:
    result = smoke_script.run_smoke_test(model_key="llm1", dry_run=True, base_url="http://localhost:8123")
    assert result["status"] == "dry_run"
    assert result["model_key"] == "llm1"
    assert result["model_id"] == "mistralai/Mistral-Small-24B-Instruct-2501"
    assert result["base_url"] == "http://localhost:8123"
    assert result["max_new_tokens"] == 512
    assert result["backend_error"] is None
