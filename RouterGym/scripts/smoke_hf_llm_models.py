"""Live smoke-test the larger HF Inference models behind llm1/llm2."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict

from RouterGym.engines.model_registry import LLM_MODELS, _get_token, load_models

DEFAULT_PROMPT = (
    "Return strict JSON with keys final_answer, reasoning, predicted_category. "
    "Ticket: reset VPN access."
)


def _force_hf_backend() -> None:
    os.environ["ROUTERGYM_MODEL_BACKEND"] = "hf_inference"


def _exception_payload(exc: BaseException, *, phase: str) -> Dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "stack_trace": traceback.format_exc(),
        "phase": phase,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test llm1/llm2 through Hugging Face Inference.")
    parser.add_argument("--model", choices=sorted(LLM_MODELS.keys()), required=True, help="Logical model key.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt to send.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional path to write the JSON result.")
    return parser


def run_live_smoke(*, model_key: str, prompt: str = DEFAULT_PROMPT) -> Dict[str, Any]:
    _force_hf_backend()
    token = _get_token()
    entry = LLM_MODELS[model_key]
    payload: Dict[str, Any] = {
        "status": "pending",
        "model_key": model_key,
        "model_id": entry.hf_id,
        "backend_used": "hf_inference",
        "endpoint_path": "",
        "output_preview": "",
        "token_present": bool(token),
        "error": None,
    }
    try:
        models = load_models(sanity=False, slm_subset=[model_key])
        engine = models[model_key]
        output = engine.generate(prompt, max_new_tokens=80, temperature=0.0)
        payload["backend_used"] = str(getattr(engine, "backend_used", "hf_inference"))
        payload["endpoint_path"] = str(getattr(engine, "last_endpoint_path", "") or "none")
        payload["output_preview"] = output[:400].replace("\n", " ")
        if "LLM unavailable" in output:
            payload["status"] = "failure"
            payload["error"] = getattr(engine, "last_error", None) or {
                "error_type": "ModelUnavailable",
                "message": "Model returned the benchmark fallback payload.",
                "stack_trace": "",
                "phase": "generate",
            }
        else:
            payload["status"] = "success"
    except Exception as exc:
        payload["status"] = "failure"
        payload["error"] = _exception_payload(exc, phase="script")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    result = run_live_smoke(model_key=args.model, prompt=args.prompt)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
