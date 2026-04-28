"""Assert that all requested RouterGym model keys resolve through local OpenAI-compatible serving."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List
from urllib import request

from RouterGym.engines.model_registry import ALL_MODELS, get_model_backend, load_models
from RouterGym.engines.openai_compatible import (
    OpenAICompatibleEngine,
    normalize_openai_compatible_base_url,
)
from RouterGym.scripts.smoke_openai_compatible_model import run_smoke_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assert that all requested RouterGym models are locally served."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(ALL_MODELS.keys()),
        required=True,
        help="Logical model keys to verify.",
    )
    parser.add_argument("--base-url", type=str, default=None, help="Override gateway base URL.")
    parser.add_argument("--api-key", type=str, default=None, help="Override gateway API key.")
    return parser


def _fetch_gateway_models(base_url: str) -> Dict[str, Any]:
    endpoint = f"{normalize_openai_compatible_base_url(base_url)}/models"
    req = request.Request(endpoint, method="GET")
    with request.urlopen(req, timeout=30) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("Gateway /v1/models response was not a JSON object.")
    return parsed


def run_assertion(
    *, model_keys: List[str], base_url: str | None = None, api_key: str | None = None
) -> Dict[str, Any]:
    backend = get_model_backend()
    if backend != "openai_compatible":
        raise RuntimeError(
            f"Expected ROUTERGYM_MODEL_BACKEND=openai_compatible but resolved {backend!r}."
        )

    models = load_models(sanity=False)
    local_resolution: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []

    for model_key in model_keys:
        engine = models.get(model_key)
        entry = ALL_MODELS[model_key]
        local_resolution[model_key] = {
            "engine_type": type(engine).__name__ if engine is not None else "missing",
            "backend_used": getattr(engine, "backend_used", ""),
            "model_name": getattr(engine, "model_name", ""),
            "request_model_name": getattr(engine, "request_model_name", ""),
        }
        if not isinstance(engine, OpenAICompatibleEngine):
            failures.append(
                {
                    "model_key": model_key,
                    "reason": "not_local_openai_engine",
                    "engine_type": type(engine).__name__ if engine is not None else "missing",
                }
            )
            continue
        if getattr(engine, "model_name", "") != entry.hf_id:
            failures.append(
                {
                    "model_key": model_key,
                    "reason": "model_id_mismatch",
                    "expected_model_id": entry.hf_id,
                    "actual_model_id": getattr(engine, "model_name", ""),
                }
            )
        if getattr(engine, "request_model_name", "") != model_key:
            failures.append(
                {
                    "model_key": model_key,
                    "reason": "request_model_name_mismatch",
                    "expected_request_model_name": model_key,
                    "actual_request_model_name": getattr(engine, "request_model_name", ""),
                }
            )

    inferred_base_url = str(
        base_url or getattr(next(iter(models.values())), "base_url", "http://127.0.0.1:8000/v1")
    )
    normalized_base_url = normalize_openai_compatible_base_url(inferred_base_url)
    gateway_models = _fetch_gateway_models(normalized_base_url)
    gateway_index = {
        str(item.get("id")): item
        for item in gateway_models.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }

    smoke_results: List[Dict[str, Any]] = []
    for model_key in model_keys:
        entry = ALL_MODELS[model_key]
        model_payload = gateway_index.get(model_key)
        if model_payload is None:
            failures.append({"model_key": model_key, "reason": "missing_from_gateway_models"})
        else:
            upstream_model_id = str(model_payload.get("routergym_upstream_model_id") or "")
            if upstream_model_id != entry.hf_id:
                failures.append(
                    {
                        "model_key": model_key,
                        "reason": "gateway_model_id_mismatch",
                        "expected_model_id": entry.hf_id,
                        "actual_model_id": upstream_model_id,
                    }
                )
        smoke = run_smoke_test(
            model_key=model_key,
            base_url=normalized_base_url,
            api_key=api_key,
            dry_run=False,
        )
        smoke_results.append(smoke)
        if smoke.get("status") != "success":
            failures.append(
                {
                    "model_key": model_key,
                    "reason": "smoke_failure",
                    "backend_error": smoke.get("backend_error"),
                    "output_preview": smoke.get("output_preview"),
                }
            )

    return {
        "status": "success" if not failures else "failure",
        "backend": backend,
        "base_url": normalized_base_url,
        "models_checked": model_keys,
        "local_resolution": local_resolution,
        "gateway_models": gateway_index,
        "smoke_results": smoke_results,
        "failures": failures,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_assertion(
        model_keys=list(args.models), base_url=args.base_url, api_key=args.api_key
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
