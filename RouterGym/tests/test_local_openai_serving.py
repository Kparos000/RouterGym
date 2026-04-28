"""Local OpenAI-serving gateway and assertion tests."""

from __future__ import annotations

from typing import Any

import pytest

from RouterGym.engines import model_registry
from RouterGym.engines.local_openai_gateway import (
    GatewayConfigError,
    build_gateway_models_payload,
    build_gateway_routes,
)
from RouterGym.engines.openai_compatible import OpenAICompatibleEngine
from RouterGym.scripts import assert_local_openai_serving


@pytest.fixture
def gateway_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("ROUTERGYM_GATEWAY_SLM1_UPSTREAM_BASE_URL", "http://127.0.0.1:8101/v1")
    monkeypatch.setenv("ROUTERGYM_GATEWAY_SLM2_UPSTREAM_BASE_URL", "http://127.0.0.1:8102/v1")
    monkeypatch.setenv("ROUTERGYM_GATEWAY_LLM1_UPSTREAM_BASE_URL", "http://127.0.0.1:8103/v1")
    monkeypatch.setenv("ROUTERGYM_GATEWAY_LLM2_UPSTREAM_BASE_URL", "http://127.0.0.1:8104/v1")
    monkeypatch.setenv(
        "ROUTERGYM_GATEWAY_LLM2_REPLICA_BASE_URLS",
        "http://127.0.0.1:8105/v1,http://127.0.0.1:8106/v1",
    )


def test_gateway_route_table_includes_all_four_keys(gateway_env: None) -> None:
    routes = build_gateway_routes()
    assert set(routes) == {"slm1", "slm2", "llm1", "llm2"}
    assert routes["slm1"].upstream_model_id == "mistralai/Mistral-7B-Instruct-v0.3"
    assert routes["slm2"].upstream_model_id == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert routes["llm1"].upstream_model_id == "mistralai/Mistral-Small-24B-Instruct-2501"
    assert routes["llm2"].upstream_model_id == "Qwen/Qwen2.5-14B-Instruct"
    assert routes["llm2"].upstream_base_urls == (
        "http://127.0.0.1:8104/v1",
        "http://127.0.0.1:8105/v1",
        "http://127.0.0.1:8106/v1",
    )
    payload = build_gateway_models_payload(routes)
    ids = [str(item.get("id")) for item in payload["data"]]
    assert ids == ["slm1", "slm2", "llm1", "llm2"]


def test_gateway_route_table_requires_local_upstreams(monkeypatch: Any) -> None:
    monkeypatch.delenv("ROUTERGYM_GATEWAY_SLM1_UPSTREAM_BASE_URL", raising=False)
    with pytest.raises(GatewayConfigError):
        build_gateway_routes(["slm1"])


def test_local_serving_assertion_fails_if_slm_is_not_local(monkeypatch: Any) -> None:
    monkeypatch.setenv("ROUTERGYM_MODEL_BACKEND", "openai_compatible")
    fake_remote = model_registry.RemoteInferenceEngine.__new__(model_registry.RemoteInferenceEngine)
    fake_remote.backend_used = "hf_inference"
    fake_remote.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    fake_remote.model_key = "slm1"
    fake_remote.kind = "slm"

    fake_local = OpenAICompatibleEngine(
        model_registry.LLM_MODELS["llm1"].hf_id,
        model_key="llm1",
        request_model_name="llm1",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
    )

    monkeypatch.setattr(
        assert_local_openai_serving,
        "load_models",
        lambda sanity=False: {"slm1": fake_remote, "llm1": fake_local},
    )
    monkeypatch.setattr(
        assert_local_openai_serving,
        "_fetch_gateway_models",
        lambda base_url: {
            "data": [
                {"id": "slm1", "routergym_upstream_model_id": "mistralai/Mistral-7B-Instruct-v0.3"}
            ]
        },
    )
    monkeypatch.setattr(
        assert_local_openai_serving,
        "run_smoke_test",
        lambda **kwargs: {"status": "success", "model_key": kwargs["model_key"]},
    )
    result = assert_local_openai_serving.run_assertion(model_keys=["slm1"])
    assert result["status"] == "failure"
    assert result["failures"][0]["reason"] == "not_local_openai_engine"


def test_local_serving_assertion_succeeds_for_all_local_keys(
    monkeypatch: Any, gateway_env: None
) -> None:
    monkeypatch.setenv("ROUTERGYM_MODEL_BACKEND", "openai_compatible")
    fake_models = {
        key: OpenAICompatibleEngine(
            entry.hf_id,
            model_key=key,
            request_model_name=key,
            kind=entry.kind,
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
        )
        for key, entry in model_registry.ALL_MODELS.items()
    }
    monkeypatch.setattr(
        assert_local_openai_serving, "load_models", lambda sanity=False: fake_models
    )
    monkeypatch.setattr(
        assert_local_openai_serving,
        "_fetch_gateway_models",
        lambda base_url: build_gateway_models_payload(build_gateway_routes()),
    )
    monkeypatch.setattr(
        assert_local_openai_serving,
        "run_smoke_test",
        lambda **kwargs: {"status": "success", "model_key": kwargs["model_key"]},
    )
    result = assert_local_openai_serving.run_assertion(model_keys=["slm1", "slm2", "llm1", "llm2"])
    assert result["status"] == "success"
    assert result["failures"] == []
