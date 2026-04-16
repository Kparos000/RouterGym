"""Run a 50-ticket HF Inference smoke validation for llm1 and llm2."""

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
from RouterGym.engines.model_registry import LLM_MODELS, load_models as registry_load_models

DEFAULT_OUTPUT_DIR = Path("test_logs_hf_smoke")
DEFAULT_TICKET_LIMIT = 50
DEFAULT_TICKET_START = 0
BASE_MODEL_KEY = "slm1"
ROUTER_MODE = "slm_dominant"
MEMORY_MODE = "rag_bm25"


class SharedClassifierFactory:
    """Return one preloaded encoder classifier instance for repeated smoke runs."""

    def __init__(self, classifier: Any) -> None:
        self._classifier = classifier

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return self._classifier


def _force_hf_backend() -> None:
    os.environ["ROUTERGYM_MODEL_BACKEND"] = "hf_inference"


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_text(path: Path) -> str:
    if not path.exists():
        return "(missing)"
    return path.read_text(encoding="utf-8").strip() or "(empty)"


def _json_block(records: List[Dict[str, Any]], limit: Optional[int] = None) -> str:
    if not records:
        return "(none)"
    selected = records if limit is None else records[:limit]
    return "\n\n".join(json.dumps(record, ensure_ascii=False, indent=2) for record in selected)


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


def _exception_payload(exc: BaseException, *, phase: str, llm_model_key: str) -> Dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "stack_trace": traceback.format_exc(),
        "phase": phase,
        "model_key": llm_model_key,
        "backend": "hf_inference",
    }


def _extract_backend_error(model: Any, *, llm_model_key: str, phase: str) -> Optional[Dict[str, str]]:
    error = getattr(model, "last_error", None)
    if isinstance(error, dict) and error.get("error_type"):
        payload = {
            "error_type": str(error.get("error_type", "BackendError")),
            "message": str(error.get("message", "")),
            "stack_trace": str(error.get("stack_trace", "")),
            "phase": str(error.get("phase", phase)),
            "model_key": llm_model_key,
            "backend": str(getattr(model, "backend_used", "hf_inference")),
        }
        return payload
    return None


def _derive_failure_payload(result: Dict[str, Any], models: Dict[str, Any], llm_model_key: str) -> Optional[Dict[str, str]]:
    final_answer = str(result.get("final_answer", "") or "").strip()
    if final_answer not in {"LLM unavailable", "No valid answer produced"}:
        return None

    llm_engine = models.get(llm_model_key)
    slm_engine = models.get(BASE_MODEL_KEY)
    for engine in (llm_engine, slm_engine):
        if engine is None:
            continue
        backend_error = _extract_backend_error(engine, llm_model_key=llm_model_key, phase="generate")
        if backend_error is not None:
            return backend_error
    return {
        "error_type": "PipelineReturnedFallback",
        "message": final_answer or "Pipeline returned an empty answer",
        "stack_trace": "",
        "phase": "pipeline",
        "model_key": llm_model_key,
        "backend": "hf_inference",
    }


def _result_record(
    *,
    ticket_id: str,
    gold_label: str,
    llm_model_key: str,
    result: Optional[Dict[str, Any]],
    error: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    payload = result or {}
    metrics_obj = payload.get("metrics")
    metrics: Dict[str, Any] = metrics_obj if isinstance(metrics_obj, dict) else {}
    return {
        "ticket_id": ticket_id,
        "gold_label": gold_label,
        "llm_model_key": llm_model_key,
        "llm_model_id": LLM_MODELS[llm_model_key].hf_id,
        "success": error is None,
        "escalated": bool(payload.get("escalated", False)),
        "final_model": payload.get("final_model", payload.get("model_name", BASE_MODEL_KEY)),
        "predicted_category": payload.get("topic_group") or payload.get("classifier_label") or "unknown",
        "latency_ms": float(metrics.get("latency_ms", payload.get("latency_ms", 0.0)) or 0.0),
        "total_tokens": int(payload.get("total_tokens", metrics.get("total_tokens", 0)) or 0),
        "total_cost_usd": float(payload.get("total_cost_usd", metrics.get("total_cost_usd", 0.0)) or 0.0),
        "backend_used": "hf_inference",
        "error": error,
    }


def _summarize_run(records: List[Dict[str, Any]], target_count: int) -> Dict[str, Any]:
    success_count = sum(1 for record in records if bool(record.get("success", False)))
    failure_count = len(records) - success_count
    total_latency = sum(float(record.get("latency_ms", 0.0) or 0.0) for record in records)
    avg_latency = total_latency / len(records) if records else 0.0
    total_tokens = sum(int(record.get("total_tokens", 0) or 0) for record in records)
    total_cost = sum(float(record.get("total_cost_usd", 0.0) or 0.0) for record in records)
    escalated_count = sum(1 for record in records if bool(record.get("escalated", False)))
    model_usage = Counter(str(record.get("final_model", "")) for record in records if record.get("final_model"))
    failure_types = Counter(
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
        "model_usage_breakdown": dict(model_usage),
        "failure_types": dict(failure_types),
    }


def _format_summary(run_summaries: Dict[str, Dict[str, Any]], overall_summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("HF Smoke 50 Summary")
    lines.append("===================")
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
        lines.append(f"  model_usage_breakdown: {json.dumps(summary['model_usage_breakdown'], ensure_ascii=False)}")
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
    lines.append(f"  model_usage_breakdown: {json.dumps(overall_summary['model_usage_breakdown'], ensure_ascii=False)}")
    lines.append(f"  failure_types: {json.dumps(overall_summary['failure_types'], ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def _write_combined_results(
    *,
    output_path: Path,
    summary_text: str,
    llm1_smoke_text: str,
    llm2_smoke_text: str,
    llm1_records: List[Dict[str, Any]],
    llm2_records: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
) -> None:
    content = "\n\n".join(
        [
            summary_text.strip(),
            "live_llm1_smoke:\n" + llm1_smoke_text,
            "live_llm2_smoke:\n" + llm2_smoke_text,
            "smoke50_llm1_first_10:\n" + _json_block(llm1_records, limit=10),
            "smoke50_llm2_first_10:\n" + _json_block(llm2_records, limit=10),
            "all_failures:\n" + _json_block(failures, limit=None),
        ]
    )
    output_path.write_text(content + "\n", encoding="utf-8")


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


def _run_model_smoke(*, llm_model_key: str, tickets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    models = registry_load_models(sanity=False, slm_subset=[BASE_MODEL_KEY, llm_model_key])
    records: List[Dict[str, Any]] = []
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
                error = _exception_payload(exc, phase="ticket_pipeline", llm_model_key=llm_model_key)
            record = _result_record(
                ticket_id=ticket["ticket_id"],
                gold_label=ticket["gold_label"],
                llm_model_key=llm_model_key,
                result=result,
                error=error,
            )
            records.append(record)
    return records


def run_hf_smoke_50(output_dir: Path, limit: int, start: int) -> int:
    _force_hf_backend()
    output_dir.mkdir(parents=True, exist_ok=True)
    tickets = _load_smoke_tickets(limit=limit, start=start)

    llm1_results = _run_model_smoke(llm_model_key="llm1", tickets=tickets)
    llm2_results = _run_model_smoke(llm_model_key="llm2", tickets=tickets)
    failures = [record for record in llm1_results + llm2_results if isinstance(record.get("error"), dict)]

    llm1_results_path = output_dir / "smoke50_llm1_results.jsonl"
    llm2_results_path = output_dir / "smoke50_llm2_results.jsonl"
    failures_path = output_dir / "smoke50_failures.jsonl"
    summary_path = output_dir / "smoke50_summary.txt"
    combined_path = output_dir / "ALL_HF_SMOKE_RESULTS.txt"

    _write_jsonl(llm1_results_path, llm1_results)
    _write_jsonl(llm2_results_path, llm2_results)
    _write_jsonl(failures_path, failures)

    run_summaries = {
        "llm1": _summarize_run(llm1_results, target_count=limit),
        "llm2": _summarize_run(llm2_results, target_count=limit),
    }
    overall_summary = _summarize_run(llm1_results + llm2_results, target_count=limit * 2)
    summary_text = _format_summary(run_summaries, overall_summary)
    summary_path.write_text(summary_text, encoding="utf-8")

    _write_combined_results(
        output_path=combined_path,
        summary_text=summary_text,
        llm1_smoke_text=_read_text(output_dir / "live_llm1_smoke.txt"),
        llm2_smoke_text=_read_text(output_dir / "live_llm2_smoke.txt"),
        llm1_records=llm1_results,
        llm2_records=llm2_results,
        failures=failures,
    )

    success_threshold = int(limit * 2 * 0.9)
    success_count = overall_summary["success_count"]
    status = "SUCCESS" if success_count >= success_threshold else "FAILURE"
    print(status)
    print(f"ALL_HF_SMOKE_RESULTS={combined_path}")
    print(summary_text.strip())
    return 0 if status == "SUCCESS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 50-ticket HF smoke validation for llm1 and llm2.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for HF smoke logs.")
    parser.add_argument("--limit", type=int, default=DEFAULT_TICKET_LIMIT, help="Tickets per llm run.")
    parser.add_argument("--start", type=int, default=DEFAULT_TICKET_START, help="0-based start index in tickets.csv.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run_hf_smoke_50(args.output_dir, args.limit, args.start))


if __name__ == "__main__":
    main()
