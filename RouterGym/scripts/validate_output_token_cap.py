"""Validate whether a lower output token cap is safe on a small benchmark sample.

Example:
    python -m RouterGym.scripts.validate_output_token_cap --sample-size 6 --tight-cap 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Optional

from RouterGym.agents.generator import resolve_max_output_tokens, run_ticket_pipeline
from RouterGym.contracts.json_contract import validate_agent_output
from RouterGym.data.tickets.dataset_loader import load_dataset
from RouterGym.engines.model_registry import get_model_backend
from RouterGym.experiments.chunked_execution import backend_override

DEFAULT_OUTPUT_DIR = Path("RouterGym/results/token_cap_validation")
FALLBACK_ANSWERS = {"No valid answer produced", "LLM unavailable"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare the default output cap with a lower cap.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for report files.")
    parser.add_argument("--sample-size", type=int, default=6, help="Number of tickets to evaluate.")
    parser.add_argument("--start", type=int, default=0, help="0-based dataset start index.")
    parser.add_argument("--tight-cap", type=int, default=200, help="Lower cap to compare against the current default.")
    parser.add_argument("--router-mode", type=str, default="slm_dominant", help="Router mode for validation.")
    parser.add_argument("--memory-mode", type=str, default="rag_bm25", help="Memory mode for validation.")
    parser.add_argument("--base-model-name", type=str, default="slm1", help="Base model for validation.")
    parser.add_argument(
        "--escalation-model-name",
        type=str,
        default="llm1",
        help="Escalation model for validation.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf_inference", "openai_compatible", "vllm_local"],
        default=None,
        help="Optional backend override for the validation run.",
    )
    return parser


def _obvious_truncation(result: Dict[str, Any], cap: int) -> bool:
    final_answer = str(result.get("final_answer", "") or "").strip()
    if not final_answer or final_answer in FALLBACK_ANSWERS:
        return False
    total_output_tokens = int(result.get("total_output_tokens", result.get("metrics", {}).get("total_output_tokens", 0)) or 0)
    near_cap = total_output_tokens >= max(int(cap * 0.9), cap - 8)
    lowered = final_answer.lower().rstrip()
    bad_suffixes = (" and", " or", " to", " for", " with", " because", " if", " then", " -", ":", ",", "(")
    ends_abruptly = lowered.endswith(bad_suffixes) or final_answer[-1] not in ".!?)]}\"'"
    return near_cap and ends_abruptly


def _evaluate_row(result: Dict[str, Any], cap: int) -> Dict[str, Any]:
    try:
        validate_agent_output(result)
        schema_valid = True
    except Exception as exc:
        schema_valid = False
        schema_error = str(exc)
    else:
        schema_error = ""

    resolution_steps = result.get("resolution_steps", [])
    usable_response = (
        str(result.get("final_answer", "") or "").strip() not in FALLBACK_ANSWERS
        and isinstance(resolution_steps, list)
        and len(resolution_steps) > 0
    )
    return {
        "ticket_id": str(result.get("ticket_id", "")),
        "schema_valid": schema_valid,
        "schema_error": schema_error,
        "usable_response": usable_response,
        "obvious_truncation": _obvious_truncation(result, cap),
        "latency_ms": float(result.get("metrics", {}).get("latency_ms", 0.0) or 0.0),
        "total_tokens": int(result.get("total_tokens", 0) or 0),
        "total_output_tokens": int(result.get("total_output_tokens", 0) or 0),
        "total_cost_usd": float(result.get("total_cost_usd", 0.0) or 0.0),
        "final_answer_preview": str(result.get("final_answer", "") or "")[:200],
    }


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    success_rows = [row for row in rows if not row.get("exception")]
    schema_valid_rows = [row for row in success_rows if bool(row.get("schema_valid"))]
    usable_rows = [row for row in success_rows if bool(row.get("usable_response"))]
    truncated_rows = [row for row in success_rows if bool(row.get("obvious_truncation"))]
    return {
        "row_count": len(rows),
        "success_count": len(success_rows),
        "failure_count": len(rows) - len(success_rows),
        "schema_valid_count": len(schema_valid_rows),
        "schema_valid_rate": round(len(schema_valid_rows) / len(rows), 3) if rows else 0.0,
        "usable_response_count": len(usable_rows),
        "usable_response_rate": round(len(usable_rows) / len(rows), 3) if rows else 0.0,
        "obvious_truncation_count": len(truncated_rows),
        "obvious_truncation_rate": round(len(truncated_rows) / len(rows), 3) if rows else 0.0,
        "avg_latency_ms": round(fmean(float(row["latency_ms"]) for row in success_rows), 2) if success_rows else 0.0,
        "avg_total_tokens": round(fmean(int(row["total_tokens"]) for row in success_rows), 2) if success_rows else 0.0,
        "avg_total_output_tokens": round(
            fmean(int(row["total_output_tokens"]) for row in success_rows), 2
        )
        if success_rows
        else 0.0,
        "total_cost_usd": round(sum(float(row["total_cost_usd"]) for row in success_rows), 6),
    }


def _scenario_label(cap: int, default_cap: int) -> str:
    return "current_default" if cap == default_cap else f"cap_{cap}"


def _run_scenario(
    *,
    tickets: List[Dict[str, str]],
    router_mode: str,
    memory_mode: str,
    base_model_name: str,
    escalation_model_name: Optional[str],
    cap: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for ticket in tickets:
        try:
            result = run_ticket_pipeline(
                ticket={"text": ticket["text"], "ticket_id": ticket["ticket_id"]},
                router_mode=router_mode,
                memory_mode=memory_mode,
                base_model_name=base_model_name,
                escalation_model_name=escalation_model_name,
                max_output_tokens=cap,
            )
            row = _evaluate_row(result, cap)
        except Exception as exc:
            row = {
                "ticket_id": ticket["ticket_id"],
                "schema_valid": False,
                "schema_error": "",
                "usable_response": False,
                "obvious_truncation": False,
                "latency_ms": 0.0,
                "total_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "final_answer_preview": "",
                "exception": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        rows.append(row)
    return {
        "cap": cap,
        "rows": rows,
        "summary": _aggregate_rows(rows),
    }


def _recommendation(default_summary: Dict[str, Any], tight_summary: Dict[str, Any], default_cap: int, tight_cap: int) -> str:
    if (
        float(default_summary["usable_response_rate"]) == 0.0
        and float(tight_summary["usable_response_rate"]) == 0.0
    ):
        return (
            f"Keep the current default cap of {default_cap}. "
            f"This sample was inconclusive because neither cap produced usable benchmark answers on the active backend."
        )
    schema_drop = float(default_summary["schema_valid_rate"]) - float(tight_summary["schema_valid_rate"])
    usable_drop = float(default_summary["usable_response_rate"]) - float(tight_summary["usable_response_rate"])
    truncation_increase = int(tight_summary["obvious_truncation_count"]) - int(default_summary["obvious_truncation_count"])
    latency_reduction = float(default_summary["avg_latency_ms"]) - float(tight_summary["avg_latency_ms"])
    cost_reduction = float(default_summary["total_cost_usd"]) - float(tight_summary["total_cost_usd"])

    if schema_drop > 0.05 or usable_drop > 0.05 or truncation_increase > 0:
        return (
            f"Keep the current default cap of {default_cap}. "
            f"The tighter cap of {tight_cap} reduced answer quality or increased truncation risk."
        )
    if latency_reduction > 0.0 or cost_reduction > 0.0:
        return (
            f"Adopt the tighter cap of {tight_cap}. "
            f"It preserved schema/response quality on this sample and reduced runtime/cost."
        )
    return (
        f"Keep the current default cap of {default_cap}. "
        f"The tighter cap of {tight_cap} did not show material runtime/cost savings on this sample."
    )


def _write_text_report(path: Path, payload: Dict[str, Any]) -> None:
    comparisons = payload["comparisons"]
    default_summary = comparisons["current_default"]["summary"]
    tight_summary = comparisons["tight_cap"]["summary"]
    lines = [
        "Output Token Cap Validation",
        f"backend: {payload['backend_name']}",
        f"default_cap: {payload['default_cap']}",
        f"tight_cap: {payload['tight_cap']}",
        f"sample_size: {payload['sample_size']}",
        "",
        "Current default summary:",
        json.dumps(default_summary, indent=2),
        "",
        "Tighter cap summary:",
        json.dumps(tight_summary, indent=2),
        "",
        f"recommendation: {payload['recommendation']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    default_cap = resolve_max_output_tokens(None)
    if args.tight_cap >= default_cap:
        raise ValueError(
            f"tight_cap must be lower than the current default cap ({default_cap}); received {args.tight_cap}."
        )

    df = load_dataset(n=args.sample_size, start=args.start)
    tickets = [
        {"ticket_id": str(args.start + idx), "text": str(row["text"])}
        for idx, (_, row) in enumerate(df.iterrows())
    ]

    backend_name = args.backend or get_model_backend()
    with backend_override(args.backend):
        default_run = _run_scenario(
            tickets=tickets,
            router_mode=args.router_mode,
            memory_mode=args.memory_mode,
            base_model_name=args.base_model_name,
            escalation_model_name=args.escalation_model_name,
            cap=default_cap,
        )
        tight_run = _run_scenario(
            tickets=tickets,
            router_mode=args.router_mode,
            memory_mode=args.memory_mode,
            base_model_name=args.base_model_name,
            escalation_model_name=args.escalation_model_name,
            cap=args.tight_cap,
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "backend_name": backend_name,
        "default_cap": default_cap,
        "tight_cap": args.tight_cap,
        "sample_size": len(tickets),
        "start": args.start,
        "router_mode": args.router_mode,
        "memory_mode": args.memory_mode,
        "base_model_name": args.base_model_name,
        "escalation_model_name": args.escalation_model_name,
        "comparisons": {
            _scenario_label(default_cap, default_cap): default_run,
            "tight_cap": tight_run,
        },
    }
    report_payload["recommendation"] = _recommendation(
        default_run["summary"],
        tight_run["summary"],
        default_cap,
        args.tight_cap,
    )

    json_path = output_dir / "output_token_cap_validation.json"
    txt_path = output_dir / "output_token_cap_validation.txt"
    json_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_text_report(txt_path, report_payload)
    print(json.dumps({"json_report_path": str(json_path), "text_report_path": str(txt_path), **report_payload}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
