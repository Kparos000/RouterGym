from __future__ import annotations

import json
import uuid
from pathlib import Path

from RouterGym.scripts import check_generation_quality_gate as quality_gate


def _temp_dir() -> Path:
    root = Path.cwd() / ".tmp_quality_gate"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"run_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_quality_gate_catches_placeholder_answer() -> None:
    tmp_dir = _temp_dir()
    result_path = tmp_dir / "slm_only__base_slm1__mem_rag_bm25__results_merged.jsonl"
    _write_rows(
        result_path,
        [
            {
                "config_identifier": "slm_only__base_slm1__mem_rag_bm25",
                "final_answer": "No valid answer produced",
                "resolution_steps": [],
                "raw_model_response_text": "",
                "raw_response_saved": False,
                "generation_valid": False,
                "placeholder_answer": True,
                "escalated": False,
            }
        ],
    )

    summary = quality_gate.summarize_quality(result_path)
    assert summary["passes_quality_gate"] is False
    failures = summary["configs"][0]["failures"]
    assert any("placeholder_answer_rate" in failure for failure in failures)


def test_quality_gate_accepts_healthy_rows() -> None:
    tmp_dir = _temp_dir()
    result_path = tmp_dir / "slm_only__base_slm1__mem_rag_bm25__results_merged.jsonl"
    _write_rows(
        result_path,
        [
            {
                "config_identifier": "slm_only__base_slm1__mem_rag_bm25",
                "final_answer": "Reset the VPN client and sign in again.",
                "resolution_steps": ["Disconnect VPN", "Reconnect with SSO"],
                "raw_model_response_text": "{\"final_answer\":\"ok\"}",
                "raw_response_saved": True,
                "generation_valid": True,
                "placeholder_answer": False,
                "escalated": False,
            }
            for _ in range(10)
        ],
    )

    summary = quality_gate.summarize_quality(result_path)
    assert summary["passes_quality_gate"] is True
    assert summary["configs"][0]["failures"] == []


def test_quality_gate_rejects_full_slm_dominant_escalation_by_default() -> None:
    tmp_dir = _temp_dir()
    result_path = tmp_dir / "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25__results_merged.jsonl"
    _write_rows(
        result_path,
        [
            {
                "config_identifier": "slm_dominant__base_slm1__esc_llm2__mem_rag_bm25",
                "final_answer": "Use the knowledge-base procedure to restore access.",
                "resolution_steps": ["Verify account", "Reset session"],
                "raw_model_response_text": "{\"final_answer\":\"ok\"}",
                "raw_response_saved": True,
                "generation_valid": True,
                "placeholder_answer": False,
                "escalated": True,
            }
            for _ in range(4)
        ],
    )

    blocked = quality_gate.summarize_quality(result_path)
    allowed = quality_gate.summarize_quality(
        result_path,
        allow_full_slm_dominant_escalation=True,
    )
    assert blocked["passes_quality_gate"] is False
    assert any(
        "slm_dominant escalation_rate is 1.0" in failure
        for failure in blocked["configs"][0]["failures"]
    )
    assert allowed["passes_quality_gate"] is True
