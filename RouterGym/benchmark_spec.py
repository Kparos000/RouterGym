"""Frozen benchmark specification for the dissertation runs.

This module is the single lightweight reference for the final benchmark
lineup, router/model matrix, memory modes, pricing version, and intended
production chunk size.
"""

from __future__ import annotations

from typing import Dict, List


BENCHMARK_SPEC_VERSION = "dissertation_run_v1"
PRICING_VERSION = "normalized_v2"
PRODUCTION_CHUNK_SIZE = 5000
MEMORY_MODES = ["none", "rag_bm25", "rag_dense", "rag_hybrid"]

MODEL_LINEUP: Dict[str, str] = {
    "slm1": "mistralai/Mistral-7B-Instruct-v0.3",
    "slm2": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llm1": "mistralai/Mistral-Small-24B-Instruct-2501",
    "llm2": "Qwen/Qwen2.5-32B-Instruct-AWQ",
}

ROUTER_MODEL_CONFIGS: List[Dict[str, str]] = [
    {"router_mode": "slm_only", "base_model": "slm1"},
    {"router_mode": "slm_only", "base_model": "slm2"},
    {"router_mode": "llm_only", "base_model": "llm1"},
    {"router_mode": "llm_only", "base_model": "llm2"},
    {"router_mode": "slm_dominant", "base_model": "slm1", "escalation_model": "llm1"},
    {"router_mode": "slm_dominant", "base_model": "slm1", "escalation_model": "llm2"},
    {"router_mode": "slm_dominant", "base_model": "slm2", "escalation_model": "llm1"},
    {"router_mode": "slm_dominant", "base_model": "slm2", "escalation_model": "llm2"},
    {"router_mode": "hybrid_specialist", "base_model": "slm1"},
]


def build_final_benchmark_matrix() -> List[Dict[str, str]]:
    """Return the frozen 36-configuration benchmark matrix."""

    return [
        {**config, "memory_mode": memory_mode}
        for config in ROUTER_MODEL_CONFIGS
        for memory_mode in MEMORY_MODES
    ]


FINAL_CONFIG_COUNT = len(build_final_benchmark_matrix())


__all__ = [
    "BENCHMARK_SPEC_VERSION",
    "FINAL_CONFIG_COUNT",
    "MEMORY_MODES",
    "MODEL_LINEUP",
    "PRICING_VERSION",
    "PRODUCTION_CHUNK_SIZE",
    "ROUTER_MODEL_CONFIGS",
    "build_final_benchmark_matrix",
]
