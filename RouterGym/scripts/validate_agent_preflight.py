"""Local-only validation for classifier/runtime dependencies before GPU runs."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from RouterGym.agents import generator as gen
from RouterGym.classifiers.encoder_classifier import (
    CALIBRATED_HEAD_PATH,
    ALLOWED_ENCODER_HEAD_MODES,
    DEFAULT_ENCODER_HEAD_MODE,
    encoder_fallback_enabled,
    ensure_encoder_classifier_ready,
    resolve_encoder_head_mode,
)
from RouterGym.data.tickets import dataset_loader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local agent preflight dependencies without calling real models."
    )
    parser.add_argument(
        "--encoder-head-mode",
        choices=sorted(ALLOWED_ENCODER_HEAD_MODES),
        default=None,
        help=f"Encoder head mode to validate (default {DEFAULT_ENCODER_HEAD_MODE}).",
    )
    parser.add_argument(
        "--allow-encoder-fallback",
        action="store_true",
        help="Explicitly allow fallback to centroid mode when calibrated head is unavailable.",
    )
    parser.add_argument(
        "--ticket-start",
        type=int,
        default=0,
        help="Dataset start index for the sample validation ticket.",
    )
    return parser


class FakeModel:
    """Deterministic fake model used to validate the generation/output path."""

    model_key = "llm1"
    backend_used = "mock_local"

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        self.last_usage = {"input_tokens": 32, "output_tokens": 18, "total_tokens": 50}
        del prompt, kwargs
        return json.dumps(
            {
                "final_answer": "Reset the VPN session and sign in again.",
                "reasoning": "The ticket is an access issue and the KB-guided answer is complete.",
                "predicted_category": "Access",
                "resolution_steps": [
                    "Disconnect the VPN client.",
                    "Reconnect using SSO and confirm access.",
                ],
            }
        )


def _set_encoder_env(head_mode: Optional[str], allow_fallback: bool) -> Dict[str, Optional[str]]:
    previous = {
        "ROUTERGYM_ENCODER_HEAD_MODE": os.getenv("ROUTERGYM_ENCODER_HEAD_MODE"),
        "ROUTERGYM_ALLOW_ENCODER_FALLBACK": os.getenv("ROUTERGYM_ALLOW_ENCODER_FALLBACK"),
    }
    if head_mode is not None:
        os.environ["ROUTERGYM_ENCODER_HEAD_MODE"] = resolve_encoder_head_mode(head_mode)
    elif not os.getenv("ROUTERGYM_ENCODER_HEAD_MODE"):
        os.environ["ROUTERGYM_ENCODER_HEAD_MODE"] = DEFAULT_ENCODER_HEAD_MODE
    if allow_fallback:
        os.environ["ROUTERGYM_ALLOW_ENCODER_FALLBACK"] = "1"
    return previous


def _restore_env(previous: Dict[str, Optional[str]]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_validation(*, ticket_start: int = 0) -> Dict[str, Any]:
    dataset = dataset_loader.load_dataset(n=1, start=ticket_start)
    if dataset.empty:
        raise RuntimeError("Dataset load returned no rows for preflight validation.")
    sample_row = dataset.iloc[0]

    classifier = ensure_encoder_classifier_ready(
        head_mode=resolve_encoder_head_mode(),
        use_lexical_prior=True,
    )

    original_load_models = gen.load_models
    try:
        gen.load_models = lambda sanity=True, slm_subset=None: {"llm1": FakeModel()}
        result = gen.run_ticket_pipeline(
            ticket={"text": str(sample_row["text"]), "ticket_id": str(ticket_start)},
            router_mode="llm_only",
            memory_mode="none",
            base_model_name="llm1",
        )
    finally:
        gen.load_models = original_load_models

    required_fields = [
        "final_answer",
        "resolution_steps",
        "raw_model_response_text",
        "generation_valid",
    ]
    missing_fields = [field for field in required_fields if field not in result]
    if missing_fields:
        raise RuntimeError(f"Missing required top-level output fields: {missing_fields}")

    return {
        "dataset_path": str(dataset_loader.DEFAULT_PATH),
        "sample_ticket_index": int(ticket_start),
        "sample_ticket_text_preview": str(sample_row["text"])[:120],
        "encoder_head_mode": resolve_encoder_head_mode(),
        "allow_encoder_fallback": encoder_fallback_enabled(),
        "encoder_backend": classifier.backend_name,
        "encoder_calibrated_head_path": str(CALIBRATED_HEAD_PATH),
        "encoder_calibrated_head_exists": bool(CALIBRATED_HEAD_PATH.exists()),
        "required_fields_present": True,
        "generation_valid": bool(result["generation_valid"]),
        "final_answer_preview": str(result["final_answer"])[:200],
        "resolution_steps_count": len(result["resolution_steps"])
        if isinstance(result["resolution_steps"], list)
        else 0,
        "raw_response_saved": bool(result.get("raw_response_saved")),
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    previous_env = _set_encoder_env(args.encoder_head_mode, args.allow_encoder_fallback)
    try:
        summary = run_validation(ticket_start=int(args.ticket_start))
    finally:
        _restore_env(previous_env)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
