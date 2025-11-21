"""Remote vLLM engine wrapper.

Calls a remote vLLM server's /v1/completions endpoint using a pooled session.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import requests


class RemoteVLLMEngine:
    """HTTP client for remote vLLM instances."""

    def __init__(
        self,
        model: str,
        endpoint: str,
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
        **extra: Any,
    ) -> str:
        """Call remote vLLM /v1/completions."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        payload.update(extra)
        url = f"{self.endpoint}/v1/completions"
        resp = self.session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("text", "") or ""
