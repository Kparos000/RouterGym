"""Smoke-test a RouterGym logical model key through an OpenAI-compatible gateway.

Examples:
    python -m RouterGym.scripts.smoke_openai_compatible_model --model slm1 --dry-run
    python -m RouterGym.scripts.smoke_openai_compatible_model --model llm2
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from RouterGym.engines.model_registry import (
    ALL_MODELS,
    _get_openai_compatible_api_key,
    _get_openai_compatible_base_url,
)
from RouterGym.engines.openai_compatible import (
    OpenAICompatibleEngine,
    get_openai_compatible_model_override,
)


DEFAULT_PROMPT = (
    "Return strict JSON with keys final_answer, reasoning, predicted_category. "
    "Ticket: reset VPN access."
)
DEFAULT_SMOKE_MAX_NEW_TOKENS = 80


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test an OpenAI-compatible RouterGym model endpoint."
    )
    parser.add_argument(
        "--model", choices=sorted(ALL_MODELS.keys()), required=True, help="Logical model key."
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt to send.")
    parser.add_argument(
        "--base-url", type=str, default=None, help="Override OpenAI-compatible base URL."
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="Override OpenAI-compatible API key."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print resolved config without sending a request."
    )
    return parser


def run_smoke_test(
    *,
    model_key: str,
    prompt: str = DEFAULT_PROMPT,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    entry = ALL_MODELS[model_key]
    resolved_base_url = base_url or _get_openai_compatible_base_url()
    resolved_api_key = api_key or _get_openai_compatible_api_key()
    request_model_name = get_openai_compatible_model_override() or model_key
    max_new_tokens = DEFAULT_SMOKE_MAX_NEW_TOKENS
    payload: Dict[str, Any] = {
        "status": "dry_run" if dry_run else "pending",
        "model_key": model_key,
        "model_id": entry.hf_id,
        "request_model_name": request_model_name,
        "backend_used": "openai_compatible",
        "base_url": resolved_base_url,
        "max_new_tokens": max_new_tokens,
        "endpoint_path": "",
        "output_preview": "",
        "backend_error": None,
    }
    if dry_run:
        return payload

    engine = OpenAICompatibleEngine(
        entry.hf_id,
        model_key=model_key,
        request_model_name=request_model_name,
        kind=entry.kind,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        timeout=30,
        max_retries=0,
    )
    output = engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    payload["endpoint_path"] = engine.last_endpoint_path or "none"
    payload["output_preview"] = output[:300].replace("\n", " ")
    payload["backend_error"] = engine.last_error
    payload["status"] = "success" if "LLM unavailable" not in output else "failure"
    return payload


def main() -> None:
    args = build_parser().parse_args()
    result = run_smoke_test(
        model_key=args.model,
        prompt=args.prompt,
        base_url=args.base_url,
        api_key=args.api_key,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
