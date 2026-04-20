"""Frozen benchmark specification tests."""

from __future__ import annotations

from RouterGym import benchmark_spec
from RouterGym.engines import pricing


def test_benchmark_spec_is_frozen() -> None:
    assert benchmark_spec.BENCHMARK_SPEC_VERSION == "dissertation_run_v1"
    assert benchmark_spec.PRODUCTION_CHUNK_SIZE == 5000
    assert benchmark_spec.MEMORY_MODES == ["none", "rag_bm25", "rag_dense", "rag_hybrid"]
    assert benchmark_spec.FINAL_CONFIG_COUNT == 36


def test_model_lineup_and_pricing_version_match() -> None:
    assert benchmark_spec.MODEL_LINEUP["llm1"] == "mistralai/Mistral-Small-24B-Instruct-2501"
    assert benchmark_spec.MODEL_LINEUP["llm2"] == "Qwen/Qwen2.5-14B-Instruct"
    assert benchmark_spec.PRICING_VERSION == "normalized_v3"
    assert pricing.PRICING_VERSION == "normalized_v3"
    assert pricing.PRICING_TABLE["llm1"].model_name == benchmark_spec.MODEL_LINEUP["llm1"]
    assert pricing.PRICING_TABLE["llm2"].model_name == benchmark_spec.MODEL_LINEUP["llm2"]
