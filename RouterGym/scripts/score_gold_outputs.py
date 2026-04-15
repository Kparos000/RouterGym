"""Score JSONL outputs against the frozen gold-eval set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from RouterGym.evaluation.gold_scoring import build_gold_index, score_record_against_gold

DEFAULT_GOLD_PATH = Path(__file__).resolve().parents[1] / "data" / "gold_eval" / "gold_eval_final.jsonl"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _ticket_index(record: Mapping[str, Any]) -> Optional[int]:
    for key in ("ticket_index", "ticket_id"):
        if key not in record:
            continue
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            return int(str(value))
        except (TypeError, ValueError):
            continue
    return None


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_scored.jsonl")


def score_records_against_gold(
    records: Sequence[Mapping[str, Any]],
    gold_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Annotate each record with deterministic gold-quality component scores."""
    gold_index = build_gold_index(gold_records)
    scored_records: List[Dict[str, Any]] = []
    for record in records:
        ticket_id = _ticket_index(record)
        gold_record = gold_index.get(ticket_id) if ticket_id is not None else None
        score_result = score_record_against_gold(record, gold_record)
        merged = dict(record)
        merged.update(score_result.as_dict())
        scored_records.append(merged)
    return scored_records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score JSONL outputs against gold_eval_final.jsonl.")
    parser.add_argument("--input-path", type=str, required=True, help="JSONL outputs to score.")
    parser.add_argument("--gold-path", type=str, default=str(DEFAULT_GOLD_PATH))
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    gold_path = Path(args.gold_path)
    output_path = Path(args.output_path) if args.output_path else _default_output_path(input_path)

    records = _read_jsonl(input_path)
    gold_records = _read_jsonl(gold_path)
    sliced_records = records[args.start : args.start + args.limit if args.limit is not None else None]
    scored = score_records_against_gold(sliced_records, gold_records)
    _write_jsonl(scored, output_path)

    print(f"Scored {len(scored)} records against {gold_path}")
    print(f"Wrote scored output to {output_path}")


if __name__ == "__main__":
    main()
