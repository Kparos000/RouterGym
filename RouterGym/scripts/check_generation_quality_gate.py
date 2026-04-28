"""Fail fast when preflight outputs show broken generation quality."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PLACEHOLDER_FINAL_ANSWER = "No valid answer produced"
PLACEHOLDER_THRESHOLD = 0.02
EMPTY_STEPS_THRESHOLD = 0.05
RAW_RESPONSE_THRESHOLD = 1.0
GENERATION_VALID_THRESHOLD = 0.95


@dataclass(frozen=True)
class QualityGateThresholds:
    placeholder_answer_max_rate: float = PLACEHOLDER_THRESHOLD
    empty_steps_max_rate: float = EMPTY_STEPS_THRESHOLD
    min_raw_response_saved_rate: float = RAW_RESPONSE_THRESHOLD
    min_generation_valid_rate: float = GENERATION_VALID_THRESHOLD

    def as_dict(self) -> Dict[str, float]:
        return {
            "placeholder_answer_max_rate": float(self.placeholder_answer_max_rate),
            "empty_steps_max_rate": float(self.empty_steps_max_rate),
            "min_raw_response_saved_rate": float(self.min_raw_response_saved_rate),
            "min_generation_valid_rate": float(self.min_generation_valid_rate),
        }


def build_thresholds(
    *,
    placeholder_answer_max_rate: float = PLACEHOLDER_THRESHOLD,
    empty_steps_max_rate: float = EMPTY_STEPS_THRESHOLD,
    min_raw_response_saved_rate: float = RAW_RESPONSE_THRESHOLD,
    min_generation_valid_rate: float = GENERATION_VALID_THRESHOLD,
) -> QualityGateThresholds:
    return QualityGateThresholds(
        placeholder_answer_max_rate=float(placeholder_answer_max_rate),
        empty_steps_max_rate=float(empty_steps_max_rate),
        min_raw_response_saved_rate=float(min_raw_response_saved_rate),
        min_generation_valid_rate=float(min_generation_valid_rate),
    )


def _discover_result_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    merged = sorted(input_path.rglob("*_results_merged.jsonl"))
    if merged:
        return merged
    return sorted(path for path in input_path.rglob("*.jsonl") if "__results" in path.name)


def _iter_jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def summarize_quality_for_file(
    path: Path,
    *,
    thresholds: Optional[QualityGateThresholds] = None,
    allow_full_slm_dominant_escalation: bool = False,
) -> Dict[str, Any]:
    active_thresholds = thresholds or build_thresholds()
    rows = list(_iter_jsonl_rows(path))
    row_count = len(rows)
    config_identifier = str(
        rows[0].get("config_identifier")
        if rows
        else path.stem.replace("__results_merged", "").replace("__results", "")
    )
    placeholder_count = 0
    empty_steps_count = 0
    raw_response_saved_count = 0
    generation_valid_count = 0
    escalated_count = 0
    has_escalation_field = False

    for row in rows:
        final_answer = str(row.get("final_answer", "") or "")
        placeholder = (
            bool(row.get("placeholder_answer")) or final_answer == PLACEHOLDER_FINAL_ANSWER
        )
        if placeholder:
            placeholder_count += 1

        resolution_steps = row.get("resolution_steps", [])
        if not isinstance(resolution_steps, list) or not resolution_steps:
            empty_steps_count += 1

        if (
            bool(row.get("raw_response_saved"))
            and str(row.get("raw_model_response_text", "") or "").strip()
        ):
            raw_response_saved_count += 1

        if bool(row.get("generation_valid")):
            generation_valid_count += 1

        if "escalated" in row:
            has_escalation_field = True
            if bool(row.get("escalated")):
                escalated_count += 1

    summary: Dict[str, Any] = {
        "config_identifier": config_identifier,
        "results_path": str(path),
        "row_count": row_count,
        "placeholder_answer_rate": _rate(placeholder_count, row_count),
        "empty_resolution_steps_rate": _rate(empty_steps_count, row_count),
        "raw_response_saved_rate": _rate(raw_response_saved_count, row_count),
        "generation_valid_rate": _rate(generation_valid_count, row_count),
        "escalation_rate": _rate(escalated_count, row_count) if has_escalation_field else None,
        "thresholds": active_thresholds.as_dict(),
        "failures": [],
    }

    failures: List[str] = []
    if summary["placeholder_answer_rate"] > active_thresholds.placeholder_answer_max_rate:
        failures.append(
            "placeholder_answer_rate "
            f"{summary['placeholder_answer_rate']:.4f} exceeds {active_thresholds.placeholder_answer_max_rate:.2f}"
        )
    if summary["empty_resolution_steps_rate"] > active_thresholds.empty_steps_max_rate:
        failures.append(
            "empty_resolution_steps_rate "
            f"{summary['empty_resolution_steps_rate']:.4f} exceeds {active_thresholds.empty_steps_max_rate:.2f}"
        )
    if summary["raw_response_saved_rate"] < active_thresholds.min_raw_response_saved_rate:
        failures.append(
            "raw_response_saved_rate "
            f"{summary['raw_response_saved_rate']:.4f} is below {active_thresholds.min_raw_response_saved_rate:.2f}"
        )
    if summary["generation_valid_rate"] < active_thresholds.min_generation_valid_rate:
        failures.append(
            "generation_valid_rate "
            f"{summary['generation_valid_rate']:.4f} is below {active_thresholds.min_generation_valid_rate:.2f}"
        )
    escalation_rate = summary["escalation_rate"]
    if (
        not allow_full_slm_dominant_escalation
        and config_identifier.startswith("slm_dominant__")
        and escalation_rate == 1.0
    ):
        failures.append("slm_dominant escalation_rate is 1.0")
    summary["failures"] = failures
    summary["passes_quality_gate"] = not failures
    return summary


def summarize_quality(
    input_path: Path,
    *,
    thresholds: Optional[QualityGateThresholds] = None,
    allow_full_slm_dominant_escalation: bool = False,
) -> Dict[str, Any]:
    active_thresholds = thresholds or build_thresholds()
    result_files = _discover_result_files(input_path)
    configs = [
        summarize_quality_for_file(
            path,
            thresholds=active_thresholds,
            allow_full_slm_dominant_escalation=allow_full_slm_dominant_escalation,
        )
        for path in result_files
    ]
    failed_configs = [
        summary["config_identifier"] for summary in configs if not summary["passes_quality_gate"]
    ]
    return {
        "input_path": str(input_path),
        "result_file_count": len(result_files),
        "failed_config_count": len(failed_configs),
        "failed_configs": failed_configs,
        "thresholds": active_thresholds.as_dict(),
        "passes_quality_gate": not failed_configs,
        "configs": configs,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail a preflight run when generation quality regressions appear."
    )
    parser.add_argument(
        "--input-path", required=True, help="Merged results file, config directory, or output root."
    )
    parser.add_argument("--output-path", help="Optional path to write the gate summary as JSON.")
    parser.add_argument(
        "--placeholder-answer-max-rate",
        type=float,
        default=PLACEHOLDER_THRESHOLD,
        help=f"Fail when placeholder answer rate exceeds this value (default {PLACEHOLDER_THRESHOLD}).",
    )
    parser.add_argument(
        "--empty-steps-max-rate",
        type=float,
        default=EMPTY_STEPS_THRESHOLD,
        help=f"Fail when empty resolution step rate exceeds this value (default {EMPTY_STEPS_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-raw-response-saved-rate",
        type=float,
        default=RAW_RESPONSE_THRESHOLD,
        help=f"Fail when raw_response_saved rate falls below this value (default {RAW_RESPONSE_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-generation-valid-rate",
        type=float,
        default=GENERATION_VALID_THRESHOLD,
        help=f"Fail when generation_valid rate falls below this value (default {GENERATION_VALID_THRESHOLD}).",
    )
    parser.add_argument(
        "--allow-slm-dominant-full-escalation",
        action="store_true",
        help="Allow slm_dominant escalation_rate == 1.0 without failing the gate.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    thresholds = build_thresholds(
        placeholder_answer_max_rate=args.placeholder_answer_max_rate,
        empty_steps_max_rate=args.empty_steps_max_rate,
        min_raw_response_saved_rate=args.min_raw_response_saved_rate,
        min_generation_valid_rate=args.min_generation_valid_rate,
    )
    summary = summarize_quality(
        Path(args.input_path),
        thresholds=thresholds,
        allow_full_slm_dominant_escalation=bool(args.allow_slm_dominant_full_escalation),
    )
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["passes_quality_gate"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
