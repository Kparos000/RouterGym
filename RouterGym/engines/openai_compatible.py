"""OpenAI-compatible engine for dedicated vLLM-style serving."""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Optional
from urllib import request


DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "http://localhost:8000/v1"


def normalize_openai_compatible_base_url(base_url: str) -> str:
    """Normalize an OpenAI-compatible base URL to include the /v1 prefix."""

    raw = str(base_url or "").strip() or DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    normalized = raw.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


class OpenAICompatibleEngine:
    """Minimal OpenAI-compatible HTTP client for vLLM-style chat serving."""

    def __init__(
        self,
        model_id: str,
        *,
        model_key: Optional[str] = None,
        request_model_name: Optional[str] = None,
        kind: str = "llm",
        base_url: str = DEFAULT_OPENAI_COMPATIBLE_BASE_URL,
        api_key: str = "EMPTY",
        timeout: int = 30,
        max_retries: int = 1,
    ) -> None:
        self.model_key = model_key or model_id
        self.model_name = model_id
        self.request_model_name = request_model_name or self.model_key
        self.kind = kind
        self.base_url = normalize_openai_compatible_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backend_used = "openai_compatible"
        self.last_usage: Optional[Dict[str, int]] = None
        self.last_endpoint_path = ""
        self.last_error: Optional[Dict[str, str]] = None

    def _extract_content(self, payload: Dict[str, Any]) -> Optional[str]:
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_value = item.get("text", "")
                        if isinstance(text_value, str):
                            text_parts.append(text_value)
                if text_parts:
                    return "".join(text_parts)
        text = first.get("text")
        if isinstance(text, str):
            return text
        return None

    def _extract_usage(self, payload: Dict[str, Any]) -> Optional[Dict[str, int]]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None

        def _coerce(value: Any) -> int:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return max(int(value), 0)
            return 0

        input_tokens = _coerce(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        output_tokens = _coerce(
            usage.get("completion_tokens", usage.get("output_tokens", 0))
        )
        total_tokens = _coerce(usage.get("total_tokens", input_tokens + output_tokens))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens or (input_tokens + output_tokens),
        }

    def _build_missing_content_error(self, payload: Dict[str, Any]) -> Dict[str, str]:
        choices = payload.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return {
                "error_type": "ValueError",
                "message": "OpenAI-compatible response did not include any choices.",
                "phase": "response_parsing",
            }

        first = choices[0]
        if not isinstance(first, dict):
            return {
                "error_type": "ValueError",
                "message": "OpenAI-compatible response choice was not an object.",
                "phase": "response_parsing",
            }

        message = first.get("message", {})
        message_keys = ""
        reasoning_present = "False"
        if isinstance(message, dict):
            message_keys = ",".join(sorted(str(key) for key in message.keys()))
            reasoning_present = str(bool(message.get("reasoning")))

        return {
            "error_type": "ValueError",
            "message": "OpenAI-compatible response did not include assistant message content.",
            "phase": "response_parsing",
            "finish_reason": str(first.get("finish_reason") or ""),
            "reasoning_present": reasoning_present,
            "message_keys": message_keys,
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        """Call an OpenAI-compatible chat completions endpoint."""

        del kwargs
        fallback = json.dumps(
            {
                "final_answer": "LLM unavailable",
                "reasoning": "timeout or error",
                "predicted_category": "unknown",
            }
        )
        self.last_usage = None
        self.last_endpoint_path = ""
        self.last_error = None
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.request_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        for _attempt in range(max(1, self.max_retries + 1)):
            req = request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    self.last_error = {
                        "error_type": "ValueError",
                        "message": "OpenAI-compatible response body was not a JSON object.",
                        "phase": "response_parsing",
                    }
                    continue
                self.last_usage = self._extract_usage(parsed)
                self.last_endpoint_path = "openai_chat_completions"
                content = self._extract_content(parsed)
                if content is not None:
                    self.last_error = None
                    return content
                self.last_error = self._build_missing_content_error(parsed)
            except Exception as exc:
                self.last_usage = None
                self.last_endpoint_path = ""
                self.last_error = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "stack_trace": traceback.format_exc(),
                }
                continue
        return fallback

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)


__all__ = [
    "DEFAULT_OPENAI_COMPATIBLE_BASE_URL",
    "OpenAICompatibleEngine",
    "normalize_openai_compatible_base_url",
]
