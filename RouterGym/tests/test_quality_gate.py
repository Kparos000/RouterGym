from __future__ import annotations

import json
import uuid
from pathlib import Path

from RouterGym.scripts import check_generation_quality_gate as quality_gate
from RouterGym.scripts.summarize_benchmark_results import summarize_benchmark_results


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
                "raw_model_response_text": '{"final_answer":"ok"}',
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
                "raw_model_response_text": '{"final_answer":"ok"}',
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


def test_quality_gate_catches_missing_raw_response() -> None:
    tmp_dir = _temp_dir()
    result_path = tmp_dir / "slm_only__base_slm1__mem_rag_bm25__results_merged.jsonl"
    _write_rows(
        result_path,
        [
            {
                "config_identifier": "slm_only__base_slm1__mem_rag_bm25",
                "final_answer": "Reset the VPN session.",
                "resolution_steps": ["Disconnect", "Reconnect"],
                "raw_model_response_text": "",
                "raw_response_saved": False,
                "generation_valid": True,
                "placeholder_answer": False,
                "escalated": False,
            }
        ],
    )

    summary = quality_gate.summarize_quality(result_path)
    assert summary["passes_quality_gate"] is False
    failures = summary["configs"][0]["failures"]
    assert any("raw_response_saved_rate" in failure for failure in failures)


def test_quality_gate_catches_empty_steps_even_with_answer() -> None:
    tmp_dir = _temp_dir()
    result_path = tmp_dir / "slm_only__base_slm1__mem_rag_bm25__results_merged.jsonl"
    _write_rows(
        result_path,
        [
            {
                "config_identifier": "slm_only__base_slm1__mem_rag_bm25",
                "final_answer": "Reset the VPN session and try again.",
                "resolution_steps": [],
                "raw_model_response_text": '{"final_answer":"ok"}',
                "raw_response_saved": True,
                "generation_valid": True,
                "placeholder_answer": False,
                "escalated": False,
            }
        ],
    )

    summary = quality_gate.summarize_quality(result_path)
    assert summary["passes_quality_gate"] is False
    failures = summary["configs"][0]["failures"]
    assert any("empty_resolution_steps_rate" in failure for failure in failures)


def test_quality_gate_fails_on_exception_style_row() -> None:
    tmp_dir = _temp_dir()
    result_path = tmp_dir / "llm_only__base_llm1__mem_rag_bm25__results_merged.jsonl"
    _write_rows(
        result_path,
        [
            {
                "config_identifier": "llm_only__base_llm1__mem_rag_bm25",
                "success": False,
                "error": {
                    "error_type": "RuntimeError",
                    "message": "generation failed before answer",
                },
            }
        ],
    )

    summary = quality_gate.summarize_quality(result_path)
    assert summary["passes_quality_gate"] is False
    failures = summary["configs"][0]["failures"]
    assert any("empty_resolution_steps_rate" in failure for failure in failures)
    assert any("raw_response_saved_rate" in failure for failure in failures)
    assert any("generation_valid_rate" in failure for failure in failures)


def test_result_summary_script_summarizes_tiny_fixture() -> None:
    tmp_dir = _temp_dir()
    result_path = (
        tmp_dir
        / "openai_compatible"
        / "slm_only__base_slm1__mem_rag_bm25"
        / "merged"
        / "slm_only__base_slm1__mem_rag_bm25__results_merged.jsonl"
    )
    _write_rows(
        result_path,
        [
            {
                "ticket_id": "1",
                "success": True,
                "gold_label": "Access",
                "predicted_category": "Access",
                "final_answer": "Reset the VPN session.",
                "resolution_steps": ["Disconnect", "Reconnect"],
                "raw_response_saved": True,
                "generation_valid": True,
                "parse_error": None,
                "metrics": {
                    "latency_ms": 10.0,
                    "total_input_tokens": 20,
                    "total_output_tokens": 5,
                    "total_tokens": 25,
                    "total_cost_usd": 0.01,
                },
            },
            {
                "ticket_id": "2",
                "success": True,
                "gold_label": "Hardware",
                "predicted_category": "Access",
                "final_answer": "Check the laptop.",
                "resolution_steps": [],
                "raw_response_saved": True,
                "generation_valid": False,
                "parse_error": "Model output is not valid JSON.",
                "metrics": {
                    "latency_ms": 30.0,
                    "total_input_tokens": 40,
                    "total_output_tokens": 15,
                    "total_tokens": 55,
                    "total_cost_usd": 0.03,
                },
            },
        ],
    )

    summary = summarize_benchmark_results(tmp_dir)
    config = summary["configs"]["slm_only__base_slm1__mem_rag_bm25"]
    assert config["row_count"] == 2
    assert config["success_rate"] == 1.0
    assert config["generation_valid_rate"] == 0.5
    assert config["raw_response_saved_rate"] == 1.0
    assert config["empty_resolution_steps_rate"] == 0.5
    assert config["parse_error_rate"] == 0.5
    assert config["gold_label_distribution"] == {"Access": 1, "Hardware": 1}
    assert config["predicted_category_distribution"] == {"Access": 2}
    assert config["average_latency_ms"] == 20.0
    assert len(config["sample_bad_rows"]) == 1
