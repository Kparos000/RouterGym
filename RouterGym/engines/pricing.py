"""Centralized normalized pricing for benchmark telemetry.

The benchmark uses a versioned, normalized pricing table rather than live
vendor prices. This keeps SLM-vs-LLM comparisons reproducible across
environments while still preserving clear relative cost differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


PRICING_VERSION = "normalized_v3"
PRICING_SOURCE = "normalized_benchmark_model"


@dataclass(frozen=True)
class PricingEntry:
    """Normalized token pricing for a benchmark model."""

    model_key: str
    model_name: str
    family: str
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float
    pricing_version: str = PRICING_VERSION
    pricing_source: str = PRICING_SOURCE
    notes: str = ""


PRICING_TABLE: Dict[str, PricingEntry] = {
    "slm1": PricingEntry(
        model_key="slm1",
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        family="slm",
        input_cost_per_1k_tokens=0.0005,
        output_cost_per_1k_tokens=0.0005,
    ),
    "slm2": PricingEntry(
        model_key="slm2",
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        family="slm",
        input_cost_per_1k_tokens=0.0007,
        output_cost_per_1k_tokens=0.0007,
    ),
    "llm1": PricingEntry(
        model_key="llm1",
        model_name="mistralai/Mistral-Small-24B-Instruct-2501",
        family="llm",
        input_cost_per_1k_tokens=0.006,
        output_cost_per_1k_tokens=0.006,
    ),
    "llm2": PricingEntry(
        model_key="llm2",
        model_name="Qwen/Qwen2.5-14B-Instruct",
        family="llm",
        input_cost_per_1k_tokens=0.003,
        output_cost_per_1k_tokens=0.003,
    ),
}

_FAMILY_FALLBACKS = {
    "slm": "slm1",
    "llm": "llm1",
}


def resolve_pricing_entry(model_key_or_name: str) -> PricingEntry:
    """Resolve a pricing entry from a model key, HF id, or family alias."""

    candidate = str(model_key_or_name or "").strip()
    normalized = candidate.lower()
    if normalized in PRICING_TABLE:
        return PRICING_TABLE[normalized]
    if normalized in _FAMILY_FALLBACKS:
        return PRICING_TABLE[_FAMILY_FALLBACKS[normalized]]
    for entry in PRICING_TABLE.values():
        if candidate == entry.model_name:
            return entry
    raise KeyError(f"Unknown pricing key: {model_key_or_name}")


def calculate_call_costs(
    model_key_or_name: str, input_tokens: int, output_tokens: int
) -> Dict[str, float | str]:
    """Calculate normalized input/output/total costs for a model call."""

    entry = resolve_pricing_entry(model_key_or_name)
    safe_input = max(int(input_tokens), 0)
    safe_output = max(int(output_tokens), 0)
    input_cost = (safe_input / 1000.0) * entry.input_cost_per_1k_tokens
    output_cost = (safe_output / 1000.0) * entry.output_cost_per_1k_tokens
    total_cost = input_cost + output_cost
    return {
        "model_key": entry.model_key,
        "model_name": entry.model_name,
        "model_family": entry.family,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "pricing_version": entry.pricing_version,
        "pricing_source": entry.pricing_source,
    }


__all__ = [
    "PRICING_SOURCE",
    "PRICING_TABLE",
    "PRICING_VERSION",
    "PricingEntry",
    "calculate_call_costs",
    "resolve_pricing_entry",
]
