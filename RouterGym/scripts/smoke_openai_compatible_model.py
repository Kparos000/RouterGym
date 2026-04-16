"""Smoke-test a dedicated OpenAI-compatible / vLLM-served larger model.

Examples:
    python -m RouterGym.scripts.smoke_openai_compatible_model --model llm1 --dry-run
    python -m RouterGym.scripts.smoke_openai_compatible_model --model llm2
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from RouterGym.engines.model_registry import (
    LLM_MODELS,
    _get_openai_compatible_api_key,
    _get_openai_compatible_base_url,
)
from RouterGym.engines.openai_compatible import OpenAICompatibleEngine


DEFAULT_PROMPT = (
    "Return strict JSON with keys final_answer, reasoning, predicted_category. "
    "Ticket: reset VPN access."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test an OpenAI-compatible larger-model endpoint.")
    parser.add_argument("--model", choices=sorted(LLM_MODELS.keys()), required=True, help="Logical model key.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt to send.")
    parser.add_argument("--base-url", type=str, default=None, help="Override OpenAI-compatible base URL.")
    parser.add_argument("--api-key", type=str, default=None, help="Override OpenAI-compatible API key.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without sending a request.")
    return parser


def run_smoke_test(
    *,
    model_key: str,
    prompt: str = DEFAULT_PROMPT,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    entry = LLM_MODELS[model_key]
    resolved_base_url = base_url or _get_openai_compatible_base_url()
    resolved_api_key = api_key or _get_openai_compatible_api_key()
    payload: Dict[str, Any] = {
        "status": "dry_run" if dry_run else "pending",
        "model_key": model_key,
        "model_id": entry.hf_id,
        "backend_used": "openai_compatible",
        "base_url": resolved_base_url,
        "endpoint_path": "",
        "output_preview": "",
    }
    if dry_run:
        return payload

    engine = OpenAICompatibleEngine(
        entry.hf_id,
        model_key=model_key,
        kind=entry.kind,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        timeout=30,
        max_retries=0,
    )
    output = engine.generate(prompt, max_new_tokens=80, temperature=0.0)
    payload["endpoint_path"] = engine.last_endpoint_path or "none"
    payload["output_preview"] = output[:300].replace("\n", " ")
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
