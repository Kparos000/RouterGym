"""Token and cost telemetry helpers for benchmark model calls."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from RouterGym.engines.pricing import PRICING_SOURCE, PRICING_VERSION, calculate_call_costs


def estimate_text_tokens(text: str) -> int:
    """Deterministically estimate tokens using the benchmark fallback rule."""

    raw = str(text or "")
    if not raw:
        return 0
    return max(int(math.ceil(len(raw) / 4.0)), 1)


def _coerce_usage_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    return None


def _extract_usage_dict(data: Any) -> Optional[Dict[str, int]]:
    if data is None:
        return None
    usage: Any = None
    if isinstance(data, dict):
        usage = data.get("usage", data)
    else:
        usage = getattr(data, "usage", data)

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    if isinstance(usage, dict):
        prompt_tokens = _coerce_usage_value(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        )
        completion_tokens = _coerce_usage_value(
            usage.get("completion_tokens", usage.get("output_tokens"))
        )
        total_tokens = _coerce_usage_value(usage.get("total_tokens"))
    else:
        prompt_tokens = _coerce_usage_value(
            getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None))
        )
        completion_tokens = _coerce_usage_value(
            getattr(usage, "completion_tokens", getattr(usage, "output_tokens", None))
        )
        total_tokens = _coerce_usage_value(getattr(usage, "total_tokens", None))

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    safe_prompt = prompt_tokens or 0
    safe_completion = completion_tokens or 0
    safe_total = total_tokens if total_tokens is not None else safe_prompt + safe_completion
    if total_tokens is not None and prompt_tokens is None and completion_tokens is not None:
        safe_prompt = max(safe_total - safe_completion, 0)
    if total_tokens is not None and completion_tokens is None and prompt_tokens is not None:
        safe_completion = max(safe_total - safe_prompt, 0)
    if safe_total == 0:
        safe_total = safe_prompt + safe_completion
    return {
        "input_tokens": safe_prompt,
        "output_tokens": safe_completion,
        "total_tokens": safe_total,
    }


def _extract_output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output and isinstance(output[0], dict):
        first = output[0]
        if "generated_text" in first:
            return str(first["generated_text"])
    if isinstance(output, dict):
        if "generated_text" in output:
            return str(output["generated_text"])
        if "text" in output:
            return str(output["text"])
    if output is not None and hasattr(output, "choices"):
        choices = getattr(output, "choices", None)
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message") or {}
                return str(message.get("content", ""))
            message = getattr(first, "message", None)
            if isinstance(message, dict):
                return str(message.get("content", ""))
            if message is not None and hasattr(message, "__getitem__"):
                try:
                    return str(message["content"])
                except Exception:
                    return ""
    return str(output)


def _call_model_raw(
    model: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Any:
    if hasattr(model, "generate"):
        try:
            return model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except TypeError:
            return model.generate(prompt)  # type: ignore[call-arg]
    if callable(model):
        try:
            return model(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        except TypeError:
            return model(prompt)
    return str(prompt)


def _backend_used(model: Any) -> str:
    backend = getattr(model, "backend_used", None)
    if isinstance(backend, str) and backend.strip():
        return backend
    return "unknown"


@dataclass(frozen=True)
class ModelCallTelemetry:
    """Per-call token and cost telemetry."""

    model_key: str
    model_name: str
    model_family: str
    backend_used: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    token_count_method: str
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    pricing_version: str = PRICING_VERSION
    pricing_source: str = PRICING_SOURCE

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def invoke_model_with_telemetry(
    model: Any,
    prompt: str,
    *,
    model_key: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tuple[str, ModelCallTelemetry]:
    """Invoke a model and return normalized text plus telemetry."""

    raw_output = _call_model_raw(model, prompt, max_new_tokens, temperature, top_p)
    output_text = _extract_output_text(raw_output)

    usage = _extract_usage_dict(raw_output)
    if usage is None:
        usage = _extract_usage_dict(getattr(model, "last_usage", None))

    if usage is None:
        input_tokens = estimate_text_tokens(prompt)
        output_tokens = estimate_text_tokens(output_text)
        total_tokens = input_tokens + output_tokens
        token_count_method = "estimated"
    else:
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        total_tokens = usage["total_tokens"] or (input_tokens + output_tokens)
        token_count_method = "measured"

    cost_info = calculate_call_costs(model_key, input_tokens, output_tokens)
    telemetry = ModelCallTelemetry(
        model_key=str(cost_info["model_key"]),
        model_name=str(cost_info["model_name"]),
        model_family=str(cost_info["model_family"]),
        backend_used=_backend_used(model),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        token_count_method=token_count_method,
        input_cost_usd=float(cost_info["input_cost_usd"]),
        output_cost_usd=float(cost_info["output_cost_usd"]),
        total_cost_usd=float(cost_info["total_cost_usd"]),
        pricing_version=str(cost_info["pricing_version"]),
        pricing_source=str(cost_info["pricing_source"]),
    )
    return output_text, telemetry


def summarize_token_count_methods(records: Sequence[ModelCallTelemetry]) -> str:
    """Summarize token counting provenance across one ticket."""

    methods = sorted({record.token_count_method for record in records if record.token_count_method})
    if not methods:
        return "estimated"
    if len(methods) == 1:
        return methods[0]
    return "mixed"


def aggregate_model_call_telemetry(records: Sequence[ModelCallTelemetry]) -> Dict[str, Any]:
    """Aggregate per-call telemetry into ticket-level totals."""

    summary: Dict[str, Any] = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "total_input_cost_usd": 0.0,
        "total_output_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "slm_input_tokens": 0,
        "slm_output_tokens": 0,
        "slm_total_tokens": 0,
        "slm_cost_usd": 0.0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_total_tokens": 0,
        "llm_cost_usd": 0.0,
        "token_count_method_summary": summarize_token_count_methods(records),
        "pricing_version": PRICING_VERSION,
        "pricing_source": PRICING_SOURCE,
    }
    for record in records:
        summary["total_input_tokens"] += record.input_tokens
        summary["total_output_tokens"] += record.output_tokens
        summary["total_tokens"] += record.total_tokens
        summary["total_input_cost_usd"] += record.input_cost_usd
        summary["total_output_cost_usd"] += record.output_cost_usd
        summary["total_cost_usd"] += record.total_cost_usd
        family = record.model_family.lower()
        if family == "slm":
            summary["slm_input_tokens"] += record.input_tokens
            summary["slm_output_tokens"] += record.output_tokens
            summary["slm_total_tokens"] += record.total_tokens
            summary["slm_cost_usd"] += record.total_cost_usd
        elif family == "llm":
            summary["llm_input_tokens"] += record.input_tokens
            summary["llm_output_tokens"] += record.output_tokens
            summary["llm_total_tokens"] += record.total_tokens
            summary["llm_cost_usd"] += record.total_cost_usd
    return summary


def telemetry_records_as_dicts(records: Sequence[ModelCallTelemetry]) -> List[Dict[str, Any]]:
    """Serialize model-call telemetry records for JSON outputs."""

    return [record.as_dict() for record in records]


__all__ = [
    "ModelCallTelemetry",
    "aggregate_model_call_telemetry",
    "estimate_text_tokens",
    "invoke_model_with_telemetry",
    "summarize_token_count_methods",
    "telemetry_records_as_dicts",
]
