"""Freeze reviewed or approved gold-eval records into a final JSONL set."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

FINAL_SCHEMA_VERSION = "gold_eval_final/v1"
DEFAULT_GOLD_DIR = Path(__file__).resolve().parents[1] / "data" / "gold_eval"
DEFAULT_DRAFT_PATH = DEFAULT_GOLD_DIR / "gold_eval_auto.jsonl"
DEFAULT_REVIEW_QUEUE_PATH = DEFAULT_GOLD_DIR / "gold_eval_review_queue.jsonl"
DEFAULT_REVIEWED_PATH = DEFAULT_GOLD_DIR / "gold_eval_reviewed.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_GOLD_DIR / "gold_eval_final.jsonl"
DEFAULT_METADATA_PATH = DEFAULT_GOLD_DIR / "gold_eval_final_metadata.json"

APPROVED_REVIEW_STATUSES = {
    "approved",
    "reviewed_approved",
    "human_approved",
    "reviewed",
}


def _read_jsonl(path: Path, required: bool = True) -> List[Dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required JSONL file: {path}")
        return []
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


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _ticket_index(record: Mapping[str, Any]) -> Optional[int]:
    value = record.get("ticket_index", record.get("ticket_id"))
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _record_complete(record: Mapping[str, Any]) -> bool:
    resolution = record.get("gold_resolution")
    return (
        _ticket_index(record) is not None
        and bool(str(record.get("topic_group", "")).strip())
        and bool(str(record.get("ticket_text", "")).strip())
        and isinstance(resolution, dict)
    )


def _is_reviewed_approved(record: Mapping[str, Any]) -> bool:
    if bool(record.get("approved", False)):
        return True
    review_status = str(record.get("review_status", "")).strip().lower()
    return review_status in APPROVED_REVIEW_STATUSES


def _merge_record(
    base_record: Mapping[str, Any], override_record: Mapping[str, Any]
) -> Dict[str, Any]:
    merged = dict(base_record)
    for key in ("topic_group", "ticket_text", "gold_resolution", "review_notes", "review_status"):
        if key in override_record and override_record.get(key) not in (None, ""):
            merged[key] = override_record.get(key)
    if "approved" in override_record:
        merged["approved"] = bool(override_record.get("approved"))
    return merged


def _build_final_record(
    record: Mapping[str, Any],
    *,
    frozen_at: str,
    review_status: str,
    source_file: Path,
    source_record_type: str,
    review_source_file: Optional[Path],
) -> Dict[str, Any]:
    return {
        "ticket_index": int(record["ticket_index"]),
        "topic_group": str(record.get("topic_group", "")),
        "ticket_text": str(record.get("ticket_text", "")),
        "gold_resolution": dict(record.get("gold_resolution", {})),
        "review_status": review_status,
        "gold_provenance": {
            "schema_version": FINAL_SCHEMA_VERSION,
            "frozen_at": frozen_at,
            "source_file": str(source_file),
            "source_record_type": source_record_type,
            "review_source_file": str(review_source_file) if review_source_file else "",
            "source_needs_human_review": bool(record.get("needs_human_review", False)),
            "source_review_reasons": list(record.get("review_reasons", []) or []),
        },
    }


def finalize_gold_records(
    draft_records: Sequence[Mapping[str, Any]],
    review_queue_records: Sequence[Mapping[str, Any]],
    reviewed_records: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    draft_source_file: Path = DEFAULT_DRAFT_PATH,
    review_source_file: Optional[Path] = None,
    allow_auto_approved: bool = True,
    frozen_at: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Freeze approved records and return final records plus metadata."""
    frozen_timestamp = frozen_at or datetime.now(timezone.utc).isoformat()
    review_queue_ids = {
        ticket_id
        for ticket_id in (_ticket_index(record) for record in review_queue_records)
        if ticket_id is not None
    }

    reviewed_by_id: Dict[int, Dict[str, Any]] = {}
    for record in reviewed_records or []:
        ticket_id = _ticket_index(record)
        if ticket_id is None:
            continue
        reviewed_by_id[ticket_id] = dict(record)

    final_records: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    included_counts = {"auto_approved": 0, "human_reviewed": 0}

    for draft_record in sorted(draft_records, key=lambda item: int(item.get("ticket_index", 0))):
        ticket_id = _ticket_index(draft_record)
        if ticket_id is None:
            exclusions.append({"ticket_index": None, "reason": "missing_ticket_index"})
            continue

        reviewed_override = reviewed_by_id.get(ticket_id)
        if reviewed_override and _is_reviewed_approved(reviewed_override):
            merged = _merge_record(draft_record, reviewed_override)
            if not _record_complete(merged):
                exclusions.append(
                    {"ticket_index": ticket_id, "reason": "reviewed_record_incomplete"}
                )
                continue
            final_records.append(
                _build_final_record(
                    merged,
                    frozen_at=frozen_timestamp,
                    review_status=str(reviewed_override.get("review_status", "human_approved"))
                    or "human_approved",
                    source_file=draft_source_file,
                    source_record_type="reviewed_override",
                    review_source_file=review_source_file,
                )
            )
            included_counts["human_reviewed"] += 1
            continue

        if ticket_id in review_queue_ids or bool(draft_record.get("needs_human_review", False)):
            exclusions.append(
                {
                    "ticket_index": ticket_id,
                    "reason": "needs_human_review",
                    "review_reasons": list(draft_record.get("review_reasons", []) or []),
                }
            )
            continue

        if allow_auto_approved and _record_complete(draft_record):
            final_records.append(
                _build_final_record(
                    draft_record,
                    frozen_at=frozen_timestamp,
                    review_status="auto_approved",
                    source_file=draft_source_file,
                    source_record_type="auto_draft",
                    review_source_file=None,
                )
            )
            included_counts["auto_approved"] += 1
        else:
            exclusions.append({"ticket_index": ticket_id, "reason": "not_auto_approved"})

    metadata: Dict[str, Any] = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "frozen_at": frozen_timestamp,
        "sample_count": len(final_records),
        "input_counts": {
            "draft_records": len(draft_records),
            "review_queue_records": len(review_queue_records),
            "reviewed_records": len(reviewed_records or []),
        },
        "included_counts": included_counts,
        "excluded_count": len(exclusions),
        "exclusions": exclusions,
    }
    return final_records, metadata


def finalize_gold_files(
    *,
    draft_path: Path,
    review_queue_path: Path,
    reviewed_path: Optional[Path],
    output_path: Path,
    metadata_path: Path,
    allow_auto_approved: bool = True,
    legacy_metadata_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Finalize the gold set from on-disk JSONL inputs."""
    draft_records = _read_jsonl(draft_path, required=True)
    review_queue_records = _read_jsonl(review_queue_path, required=False)
    reviewed_records = _read_jsonl(reviewed_path, required=False) if reviewed_path else []

    final_records, metadata = finalize_gold_records(
        draft_records,
        review_queue_records,
        reviewed_records,
        draft_source_file=draft_path,
        review_source_file=reviewed_path,
        allow_auto_approved=allow_auto_approved,
    )
    metadata.update(
        {
            "input_files": {
                "draft_path": str(draft_path),
                "review_queue_path": str(review_queue_path),
                "reviewed_path": str(reviewed_path) if reviewed_path else "",
            },
            "output_file": str(output_path),
        }
    )

    _write_jsonl(final_records, output_path)
    _write_json(metadata, metadata_path)
    if legacy_metadata_path:
        _write_json(metadata, legacy_metadata_path)
    return final_records, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze approved gold-eval records into gold_eval_final.jsonl."
    )
    parser.add_argument("--draft-path", type=str, default=str(DEFAULT_DRAFT_PATH))
    parser.add_argument("--review-queue-path", type=str, default=str(DEFAULT_REVIEW_QUEUE_PATH))
    parser.add_argument(
        "--reviewed-path",
        type=str,
        default=str(DEFAULT_REVIEWED_PATH),
        help="Optional reviewed override JSONL; ignored if the file does not exist.",
    )
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--metadata-path", type=str, default=str(DEFAULT_METADATA_PATH))
    parser.add_argument(
        "--allow-auto-approved", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    reviewed_path = Path(args.reviewed_path) if args.reviewed_path else None
    if reviewed_path is not None and not reviewed_path.exists():
        reviewed_path = None

    final_records, metadata = finalize_gold_files(
        draft_path=Path(args.draft_path),
        review_queue_path=Path(args.review_queue_path),
        reviewed_path=reviewed_path,
        output_path=Path(args.output_path),
        metadata_path=Path(args.metadata_path),
        allow_auto_approved=bool(args.allow_auto_approved),
    )
    print(f"Wrote {len(final_records)} final gold records to {args.output_path}")
    print(f"Metadata written to {args.metadata_path}")
    print(f"Excluded {metadata['excluded_count']} records during freeze")


if __name__ == "__main__":
    main()
