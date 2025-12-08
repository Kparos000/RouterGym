"""Backend switch tests for model registry."""

from __future__ import annotations

from typing import Any

import pytest

from RouterGym.engines import model_registry


@pytest.mark.skipif(model_registry.LocalVLLMEngine is None, reason="vLLM not installed")
def test_get_model_backend_vllm(monkeypatch: Any) -> None:
    monkeypatch.setenv("ROUTERGYM_MODEL_BACKEND", "vllm_local")
    assert model_registry.get_model_backend() == "vllm_local"
