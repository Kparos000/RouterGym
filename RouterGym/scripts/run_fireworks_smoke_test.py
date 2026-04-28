"""Run a 50-ticket Fireworks smoke validation using the existing agentic pipeline.

This script is intentionally narrow:
- router_mode = slm_dominant
- memory_mode = rag_bm25
- base SLM = slm1
- escalation LLMs = llm1 then llm2

The benchmark spec stays unchanged. Fireworks-specific model IDs are applied only
for this script's process so the final benchmark registry remains intact.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from RouterGym.agents import generator as generator_module
from RouterGym.agents.generator import run_ticket_pipeline
from RouterGym.data.tickets.dataset_loader import load_dataset
from RouterGym.engines.model_registry import (
    LLM_MODELS,
    ModelEntry,
    load_models as registry_load_models,
)

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_OUTPUT_DIR = Path("test_logs_fireworks")
DEFAULT_TICKET_LIMIT = 50
DEFAULT_TICKET_START = 0
BASE_MODEL_KEY = "slm1"
ROUTER_MODE = "slm_dominant"
MEMORY_MODE = "rag_bm25"

# Fireworks OpenAI-compatible model identifiers use the accounts/... path.
FIREWORKS_MODEL_OVERRIDES: Dict[str, str] = {
    "llm1": "accounts/fireworks/models/mistral-small-24b-instruct-2501",
    "llm2": "accounts/fireworks/models/qwen2p5-32b-instruct",
}


class MissingFireworksApiKeyError(RuntimeError):
    """Raised when Fireworks smoke validation is requested without an API key."""


class SharedClassifierFactory:
    """Return one preloaded encoder classifier instance for repeated smoke runs."""

    def __init__(self, classifier: Any) -> None:
        self._classifier = classifier

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return self._classifier


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _json_block(records: List[Dict[str, Any]], limit: Optional[int] = None) -> str:
    if not records:
        return "(none)"
    selected = records if limit is None else records[:limit]
    return "\n\n".join(json.dumps(record, ensure_ascii=False, indent=2) for record in selected)


def _configure_fireworks_env(api_key: str) -> None:
    os.environ["ROUTERGYM_MODEL_BACKEND"] = "openai_compatible"
    os.environ["ROUTERGYM_OPENAI_BASE_URL"] = FIREWORKS_BASE_URL
    os.environ["ROUTERGYM_OPENAI_API_KEY"] = api_key


def _require_fireworks_api_key() -> str:
    api_key = (os.getenv("ROUTERGYM_OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise MissingFireworksApiKeyError(
            "ROUTERGYM_OPENAI_API_KEY is not set. Export a Fireworks API key before running this smoke test."
        )
    return api_key


@contextmanager
def _temporary_fireworks_model_mapping() -> Iterator[None]:
    original = {key: LLM_MODELS[key] for key in FIREWORKS_MODEL_OVERRIDES}
    try:
        for key, model_id in FIREWORKS_MODEL_OVERRIDES.items():
            LLM_MODELS[key] = ModelEntry(name=key, hf_id=model_id, kind="llm")
        yield
    finally:
        for key, entry in original.items():
            LLM_MODELS[key] = entry


@contextmanager
def _patched_generator_runtime(models: Dict[str, Any]) -> Iterator[None]:
    from RouterGym.classifiers.encoder_classifier import EncoderClassifier

    classifier = EncoderClassifier(head_mode="calibrated", use_lexical_prior=True)
    original_load_models = generator_module.load_models
    original_encoder = generator_module.EncoderClassifier
    try:
        generator_module.load_models = lambda sanity=True, slm_subset=None, force_llm=False: models
        generator_module.EncoderClassifier = SharedClassifierFactory(classifier)
        yield
    finally:
        generator_module.load_models = original_load_models
        generator_module.EncoderClassifier = original_encoder


def _build_exception_payload(exc: BaseException) -> Dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "stack_trace": traceback.format_exc(),
    }


def _extract_backend_error(model: Any) -> Optional[Dict[str, str]]:
    error = getattr(model, "last_error", None)
    if isinstance(error, dict) and error.get("error_type"):
        return {
            "error_type": str(error.get("error_type", "BackendError")),
            "message": str(error.get("message", "")),
            "stack_trace": str(error.get("stack_trace", "")),
        }
    return None


def _derive_failure_payload(
    result: Dict[str, Any], models: Dict[str, Any], llm_model_key: str
) -> Optional[Dict[str, str]]:
    final_answer = str(result.get("final_answer", "") or "").strip()
    if final_answer not in {"LLM unavailable", "No valid answer produced"}:
        return None

    llm_engine = models.get(llm_model_key)
    slm_engine = models.get(BASE_MODEL_KEY)
    for engine in (llm_engine, slm_engine):
        backend_error = _extract_backend_error(engine)
        if backend_error is not None:
            return backend_error
    return {
        "error_type": "PipelineReturnedFallback",
        "message": final_answer or "Pipeline returned an empty answer",
        "stack_trace": "",
    }


def _result_record(
    *,
    ticket_id: str,
    gold_label: str,
    llm_model_key: str,
    llm_model_id: str,
    result: Optional[Dict[str, Any]],
    error: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    payload = result or {}
    metrics_obj = payload.get("metrics")
    metrics: Dict[str, Any] = metrics_obj if isinstance(metrics_obj, dict) else {}
    return {
        "ticket_id": ticket_id,
        "gold_label": gold_label,
        "smoke_llm_key": llm_model_key,
        "smoke_llm_model_id": llm_model_id,
        "predicted_category": payload.get("topic_group")
        or payload.get("classifier_label")
        or "unknown",
        "final_answer": payload.get("final_answer", ""),
        "reasoning": payload.get("reasoning", ""),
        "escalated": bool(payload.get("escalated", False)),
        "final_model": payload.get("final_model", payload.get("model_name", BASE_MODEL_KEY)),
        "latency_ms": float(metrics.get("latency_ms", payload.get("latency_ms", 0.0)) or 0.0),
        "total_tokens": int(payload.get("total_tokens", metrics.get("total_tokens", 0)) or 0),
        "total_cost_usd": float(
            payload.get("total_cost_usd", metrics.get("total_cost_usd", 0.0)) or 0.0
        ),
        "routing_policy_version": payload.get("routing_policy_version", ""),
        "error": error,
    }


def _summarize_run(records: List[Dict[str, Any]], target_count: int) -> Dict[str, Any]:
    success_count = sum(1 for record in records if not record.get("error"))
    failure_count = len(records) - success_count
    total_latency = sum(float(record.get("latency_ms", 0.0) or 0.0) for record in records)
    avg_latency = total_latency / len(records) if records else 0.0
    total_tokens = sum(int(record.get("total_tokens", 0) or 0) for record in records)
    total_cost = sum(float(record.get("total_cost_usd", 0.0) or 0.0) for record in records)
    escalated_count = sum(1 for record in records if bool(record.get("escalated", False)))
    final_model_counter = Counter(
        str(record.get("final_model", "")) for record in records if record.get("final_model")
    )
    error_counter = Counter(
        str(record["error"].get("error_type", "unknown"))
        for record in records
        if isinstance(record.get("error"), dict)
    )
    return {
        "target_tickets": target_count,
        "processed_tickets": len(records),
        "success_count": success_count,
        "failure_count": failure_count,
        "avg_latency_ms": avg_latency,
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "escalation_rate": (escalated_count / len(records)) if records else 0.0,
        "model_usage_breakdown": dict(final_model_counter),
        "failure_types": dict(error_counter),
    }


def _summarize_failure_types(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter(
        str(record["error"].get("error_type", "unknown"))
        for record in records
        if isinstance(record.get("error"), dict)
    )
    return dict(counter)


def _format_summary(
    run_summaries: Dict[str, Dict[str, Any]],
    overall_summary: Dict[str, Any],
    notes: Optional[List[str]] = None,
) -> str:
    lines: List[str] = []
    lines.append("Fireworks Smoke Test Summary")
    lines.append("=" * 29)
    lines.append("")
    if notes:
        lines.append("Notes:")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    for model_key, summary in run_summaries.items():
        lines.append(f"{model_key}:")
        lines.append(f"  target_tickets: {summary['target_tickets']}")
        lines.append(f"  processed_tickets: {summary['processed_tickets']}")
        lines.append(f"  success_count: {summary['success_count']}")
        lines.append(f"  failure_count: {summary['failure_count']}")
        lines.append(f"  avg_latency_ms: {summary['avg_latency_ms']:.2f}")
        lines.append(f"  total_tokens: {summary['total_tokens']}")
        lines.append(f"  total_cost_usd: {summary['total_cost_usd']:.6f}")
        lines.append(f"  escalation_rate: {summary['escalation_rate']:.3f}")
        lines.append(
            f"  model_usage_breakdown: {json.dumps(summary['model_usage_breakdown'], ensure_ascii=False)}"
        )
        lines.append(f"  failure_types: {json.dumps(summary['failure_types'], ensure_ascii=False)}")
        lines.append("")
    lines.append("overall:")
    lines.append(f"  total_ticket_runs: {overall_summary['target_tickets']}")
    lines.append(f"  processed_tickets: {overall_summary['processed_tickets']}")
    lines.append(f"  success_count: {overall_summary['success_count']}")
    lines.append(f"  failure_count: {overall_summary['failure_count']}")
    lines.append(f"  avg_latency_ms: {overall_summary['avg_latency_ms']:.2f}")
    lines.append(f"  total_tokens: {overall_summary['total_tokens']}")
    lines.append(f"  total_cost_usd: {overall_summary['total_cost_usd']:.6f}")
    lines.append(f"  escalation_rate: {overall_summary['escalation_rate']:.3f}")
    lines.append(
        f"  model_usage_breakdown: {json.dumps(overall_summary['model_usage_breakdown'], ensure_ascii=False)}"
    )
    lines.append(
        f"  failure_types: {json.dumps(overall_summary['failure_types'], ensure_ascii=False)}"
    )
    return "\n".join(lines) + "\n"


def _write_combined_results(
    path: Path,
    summary_text: str,
    llm1_results: List[Dict[str, Any]],
    llm2_results: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> None:
    content = "\n\n".join(
        [
            summary_text.strip(),
            "llm1_results_first_10:\n" + _json_block(llm1_results, limit=10),
            "llm2_results_first_10:\n" + _json_block(llm2_results, limit=10),
            "all_failures:\n" + _json_block(failures, limit=None),
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def _load_smoke_tickets(limit: int, start: int) -> List[Dict[str, str]]:
    df = load_dataset(n=limit, start=start)
    tickets: List[Dict[str, str]] = []
    for idx, row in df.iterrows():
        tickets.append(
            {
                "ticket_id": str(start + int(idx)),
                "text": str(row["text"]),
                "gold_label": str(row.get("label", "")),
            }
        )
    return tickets


def _run_model_smoke(
    *,
    llm_model_key: str,
    tickets: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    model_id = FIREWORKS_MODEL_OVERRIDES[llm_model_key]
    models = registry_load_models(sanity=True, slm_subset=[BASE_MODEL_KEY, llm_model_key])
    with _patched_generator_runtime(models):
        for index, ticket in enumerate(tickets, start=1):
            print(f"[{llm_model_key}] ticket {index}/{len(tickets)} id={ticket['ticket_id']}")
            result: Optional[Dict[str, Any]] = None
            error: Optional[Dict[str, str]] = None
            try:
                result = run_ticket_pipeline(
                    ticket={"text": ticket["text"], "ticket_id": ticket["ticket_id"]},
                    router_mode=ROUTER_MODE,
                    memory_mode=MEMORY_MODE,
                    base_model_name=BASE_MODEL_KEY,
                    escalation_model_name=llm_model_key,
                )
                error = _derive_failure_payload(result, models, llm_model_key)
            except Exception as exc:
                error = _build_exception_payload(exc)
            record = _result_record(
                ticket_id=ticket["ticket_id"],
                gold_label=ticket["gold_label"],
                llm_model_key=llm_model_key,
                llm_model_id=model_id,
                result=result,
                error=error,
            )
            records.append(record)
    return records


def run_fireworks_smoke_test(output_dir: Path, limit: int, start: int) -> int:
    output_dir = _ensure_output_dir(output_dir)
    llm1_results_path = output_dir / "smoke_llm1_results.jsonl"
    llm2_results_path = output_dir / "smoke_llm2_results.jsonl"
    failures_path = output_dir / "smoke_failures.jsonl"
    summary_path = output_dir / "smoke_summary.txt"
    combined_path = output_dir / "ALL_FIREWORKS_RESULTS.txt"

    notes: List[str] = [
        "Fireworks base URL fixed to https://api.fireworks.ai/inference/v1",
        "Temporary smoke model mapping only; final benchmark spec is unchanged.",
    ]

    llm_records: Dict[str, List[Dict[str, Any]]] = {"llm1": [], "llm2": []}
    failures: List[Dict[str, Any]] = []

    try:
        api_key = _require_fireworks_api_key()
        _configure_fireworks_env(api_key)
        tickets = _load_smoke_tickets(limit=limit, start=start)
        with _temporary_fireworks_model_mapping():
            for llm_model_key in ("llm1", "llm2"):
                llm_records[llm_model_key] = _run_model_smoke(
                    llm_model_key=llm_model_key, tickets=tickets
                )
                failures.extend(
                    record
                    for record in llm_records[llm_model_key]
                    if isinstance(record.get("error"), dict)
                )
    except Exception as exc:
        notes.append(
            "Run did not start cleanly. Fireworks serverless support for the target model pages currently appears unavailable."
        )
        config_error = _build_exception_payload(exc)
        for llm_model_key in ("llm1", "llm2"):
            failures.append(
                {
                    "ticket_id": "",
                    "gold_label": "",
                    "smoke_llm_key": llm_model_key,
                    "smoke_llm_model_id": FIREWORKS_MODEL_OVERRIDES[llm_model_key],
                    "predicted_category": "unknown",
                    "final_answer": "",
                    "reasoning": "",
                    "escalated": False,
                    "final_model": "",
                    "latency_ms": 0.0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "routing_policy_version": "",
                    "error": config_error,
                }
            )
        llm_records["llm1"] = []
        llm_records["llm2"] = []

    _write_jsonl(llm1_results_path, llm_records["llm1"])
    _write_jsonl(llm2_results_path, llm_records["llm2"])
    _write_jsonl(failures_path, failures)

    run_summaries = {
        "llm1": _summarize_run(llm_records["llm1"], target_count=limit),
        "llm2": _summarize_run(llm_records["llm2"], target_count=limit),
    }
    overall_summary = _summarize_run(
        llm_records["llm1"] + llm_records["llm2"], target_count=limit * 2
    )
    if failures:
        overall_summary["failure_count"] = max(int(overall_summary["failure_count"]), len(failures))
        overall_summary["failure_types"] = _summarize_failure_types(failures)
    summary_text = _format_summary(run_summaries, overall_summary, notes=notes)
    summary_path.write_text(summary_text, encoding="utf-8")
    _write_combined_results(
        combined_path,
        summary_text=summary_text,
        llm1_results=llm_records["llm1"],
        llm2_results=llm_records["llm2"],
        failures=failures,
    )

    success_threshold = int(limit * 2 * 0.9)
    success_count = overall_summary["success_count"]
    status = "SUCCESS" if success_count >= success_threshold else "FAILURE"
    print(status)
    print(f"ALL_FIREWORKS_RESULTS={combined_path}")
    print(summary_text.strip())
    return 0 if status == "SUCCESS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a 50-ticket Fireworks smoke validation.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for smoke logs."
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_TICKET_LIMIT, help="Tickets per LLM run."
    )
    parser.add_argument(
        "--start", type=int, default=DEFAULT_TICKET_START, help="0-based ticket start index."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run_fireworks_smoke_test(args.output_dir, args.limit, args.start))


if __name__ == "__main__":
    main()
