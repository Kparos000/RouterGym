"""Tests for freezing approved gold-eval records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from RouterGym.scripts import finalize_gold_eval as finalize


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def _draft_record(ticket_index: int, *, needs_review: bool = False, summary: str = "Summary") -> Dict[str, Any]:
    return {
        "ticket_index": ticket_index,
        "topic_group": "Access",
        "ticket_text": f"Ticket {ticket_index}",
        "gold_resolution": {
            "summary": summary,
            "steps": ["Verify account", "Provision VPN", "Ask user to test"],
            "escalation_required": False,
            "escalation_reason": "",
            "kb_policies": ["access.vpn"],
            "acceptance_criteria": ["VPN connects", "User confirms access"],
        },
        "needs_human_review": needs_review,
        "review_reasons": ["summary_missing"] if needs_review else [],
    }


def test_finalize_gold_records_excludes_review_queue_and_sets_provenance() -> None:
    final_records, metadata = finalize.finalize_gold_records(
        draft_records=[_draft_record(1), _draft_record(2, needs_review=True)],
        review_queue_records=[_draft_record(2, needs_review=True)],
        reviewed_records=[],
        draft_source_file=Path("gold_eval_auto.jsonl"),
        allow_auto_approved=True,
        frozen_at="2026-04-15T00:00:00+00:00",
    )

    assert len(final_records) == 1
    assert final_records[0]["ticket_index"] == 1
    assert final_records[0]["review_status"] == "auto_approved"
    assert final_records[0]["gold_provenance"]["source_file"] == "gold_eval_auto.jsonl"
    assert final_records[0]["gold_provenance"]["schema_version"] == finalize.FINAL_SCHEMA_VERSION
    assert metadata["sample_count"] == 1
    assert metadata["excluded_count"] == 1
    assert metadata["exclusions"][0]["ticket_index"] == 2


def test_finalize_gold_files_prefers_reviewed_override(tmp_path: Path) -> None:
    draft_path = tmp_path / "gold_eval_auto.jsonl"
    review_queue_path = tmp_path / "gold_eval_review_queue.jsonl"
    reviewed_path = tmp_path / "gold_eval_reviewed.jsonl"
    output_path = tmp_path / "gold_eval_final.jsonl"
    metadata_path = tmp_path / "gold_eval_final_metadata.json"

    _write_jsonl(draft_path, [_draft_record(7, needs_review=True, summary="Auto summary")])
    _write_jsonl(review_queue_path, [_draft_record(7, needs_review=True, summary="Auto summary")])
    _write_jsonl(
        reviewed_path,
        [
            {
                "ticket_index": 7,
                "gold_resolution": {
                    "summary": "Human reviewed summary",
                    "steps": ["Verify account", "Provision VPN", "Confirm VPN works"],
                    "escalation_required": False,
                    "escalation_reason": "",
                    "kb_policies": ["access.vpn"],
                    "acceptance_criteria": ["VPN connects", "User confirms access"],
                },
                "review_status": "human_approved",
                "approved": True,
            }
        ],
    )

    final_records, metadata = finalize.finalize_gold_files(
        draft_path=draft_path,
        review_queue_path=review_queue_path,
        reviewed_path=reviewed_path,
        output_path=output_path,
        metadata_path=metadata_path,
        allow_auto_approved=True,
    )

    assert output_path.exists()
    assert metadata_path.exists()
    assert len(final_records) == 1
    assert final_records[0]["ticket_index"] == 7
    assert final_records[0]["review_status"] == "human_approved"
    assert final_records[0]["gold_resolution"]["summary"] == "Human reviewed summary"
    assert metadata["included_counts"]["human_reviewed"] == 1
