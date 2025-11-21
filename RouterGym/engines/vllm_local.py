"""Local vLLM engine wrapper for SLMs.

Uses vLLM's LlamaEngine (or LLM fallback) to load local models with optional CPU
execution. Exposes a simple generate API suitable for routing/memory pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from vllm import LlamaEngine, LLM, SamplingParams
except Exception:  # pragma: no cover - optional dependency
    LlamaEngine = None  # type: ignore
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore


class LocalVLLMEngine:
    """Local vLLM engine targeting small models."""

    def __init__(
        self,
        model: str,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        enforce_cpu: bool = False,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if LlamaEngine is None and LLM is None:
            raise ImportError("vLLM is not installed. Install vllm to use LocalVLLMEngine.")

        kwargs: Dict[str, Any] = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
        }
        if max_model_len:
            kwargs["max_model_len"] = max_model_len
        if enforce_cpu:
            kwargs["device"] = "cpu"
        if engine_kwargs:
            kwargs.update(engine_kwargs)

        # Prefer LlamaEngine if available, otherwise fall back to LLM wrapper.
        if LlamaEngine is not None:
            self._engine = LlamaEngine(**kwargs)  # type: ignore[arg-type]
        else:
            self._engine = LLM(**kwargs)  # type: ignore[arg-type]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        """Generate text from the local model."""
        if SamplingParams is None:
            raise RuntimeError("SamplingParams unavailable; ensure vllm is installed correctly.")
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        outputs: List[Any] = self._engine.generate(prompt, params)  # type: ignore[attr-defined]
        # vLLM returns a list of RequestOutput; take first completion text.
        return outputs[0].outputs[0].text if outputs else ""
