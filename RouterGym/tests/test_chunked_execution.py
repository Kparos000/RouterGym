"""Tests for chunked/resumable production execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from RouterGym.experiments import chunked_execution


def _temp_dir() -> Path:
    root = Path.cwd() / ".tmp_chunked_testdirs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"run_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_df(size: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"ticket {idx}" for idx in range(size)],
            "label": ["Access" for _ in range(size)],
        }
    )


def _patch_dataset(monkeypatch: Any, size: int) -> None:
    df = _make_df(size)

    def fake_load_ticket_dataset(*, limit: int | None, start: int):
        sliced = df.iloc[start:]
        if limit is not None:
            sliced = sliced.head(limit)
        return sliced.reset_index(drop=True)

    monkeypatch.setattr(chunked_execution, "load_ticket_dataset", fake_load_ticket_dataset)


def _sample_config() -> Dict[str, str]:
    return {
        "router_mode": "slm_dominant",
        "base_model": "slm1",
        "escalation_model": "llm1",
        "memory_mode": "rag_bm25",
    }


@pytest.fixture(autouse=True)
def _stub_dependency_validation(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        chunked_execution,
        "validate_benchmark_dependencies",
        lambda: {
            "encoder_head_mode": "calibrated",
            "allow_encoder_fallback": False,
            "classifier_backend": "encoder_calibrated",
            "calibrated_head_path": "RouterGym/classifiers/encoder_calibrated_head.npz",
            "calibrated_head_exists": True,
        },
    )


def _write_fake_chunk_outputs(
    *,
    output_root: Path,
    config_identifier: str,
    backend_name: str,
    chunk_spec: Dict[str, int],
) -> Dict[str, Any]:
    config_dir = chunked_execution.get_config_output_dir(
        output_root, backend_name, config_identifier
    )
    results_path = chunked_execution.chunk_results_path(config_dir, chunk_spec)
    failures_path = chunked_execution.chunk_failures_path(config_dir, chunk_spec)
    metadata_path = chunked_execution.chunk_metadata_path(config_dir, chunk_spec)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        for ticket_index in range(int(chunk_spec["start"]), int(chunk_spec["end_exclusive"])):
            handle.write(
                json.dumps(
                    {"ticket_index": ticket_index, "chunk_index": int(chunk_spec["chunk_index"])}
                )
                + "\n"
            )
    failures_path.write_text("", encoding="utf-8")

    metadata = {
        "config_identifier": config_identifier,
        "backend_name": backend_name,
        "chunk_index": int(chunk_spec["chunk_index"]),
        "start": int(chunk_spec["start"]),
        "end_exclusive": int(chunk_spec["end_exclusive"]),
        "ticket_count": int(chunk_spec["ticket_count"]),
        "row_count": int(chunk_spec["ticket_count"]),
        "success_count": int(chunk_spec["ticket_count"]),
        "failure_count": 0,
        "results_path": str(results_path),
        "failures_path": str(failures_path),
        "metadata_path": str(metadata_path),
        "completed_at": "2026-04-22T00:00:00+00:00",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def test_write_json_survives_concurrent_writers() -> None:
    tmp_path = _temp_dir()
    status_path = tmp_path / "run_status.json"

    def write_status(worker_index: int) -> None:
        for sequence in range(50):
            chunked_execution._write_json(
                status_path,
                {
                    "worker_index": worker_index,
                    "sequence": sequence,
                },
            )

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(write_status, range(12)))

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert isinstance(payload["worker_index"], int)
    assert isinstance(payload["sequence"], int)
    assert not list(tmp_path.glob("*run_status.json*.tmp"))
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_emit_progress_update_keeps_running_when_backend_summary_races(monkeypatch: Any) -> None:
    tmp_path = _temp_dir()
    config = _sample_config()
    config_identifier = chunked_execution.build_config_identifier(config)
    backend_name = "openai_compatible"
    manifest = chunked_execution.initialize_manifest(
        config_identifier=config_identifier,
        config=config,
        output_root=tmp_path,
        backend_details={"backend_name": backend_name},
        total_tickets_expected=2,
        ticket_start=0,
        ticket_limit=2,
        chunk_size=1,
        max_output_tokens=None,
    )
    chunk_plan = chunked_execution.build_chunk_plan(total_tickets=2, chunk_size=1)

    def racing_backend_summary(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args, kwargs
        raise FileNotFoundError("simulated shared status race")

    monkeypatch.setattr(
        chunked_execution,
        "write_backend_status_summary",
        racing_backend_summary,
    )

    status_payload = chunked_execution._emit_progress_update(
        output_root=tmp_path,
        backend_name=backend_name,
        manifest=manifest,
        chunk_plan=chunk_plan,
        current_status="running",
    )

    status_path = chunked_execution.config_status_path(Path(manifest["output_files"]["config_dir"]))
    stored_status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["current_status"] == "running"
    assert stored_status["config_identifier"] == config_identifier
    assert stored_status["current_status"] == "running"
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_build_chunk_plan_boundaries() -> None:
    plan = chunked_execution.build_chunk_plan(total_tickets=12, chunk_size=5, start_index=0)
    assert plan == [
        {"chunk_index": 0, "start": 0, "end_exclusive": 5, "ticket_count": 5},
        {"chunk_index": 1, "start": 5, "end_exclusive": 10, "ticket_count": 5},
        {"chunk_index": 2, "start": 10, "end_exclusive": 12, "ticket_count": 2},
    ]


def test_production_chunk_size_defaults_to_100_ticket_boundaries() -> None:
    plan = chunked_execution.build_chunk_plan(total_tickets=250, chunk_size=100, start_index=0)
    assert plan == [
        {"chunk_index": 0, "start": 0, "end_exclusive": 100, "ticket_count": 100},
        {"chunk_index": 1, "start": 100, "end_exclusive": 200, "ticket_count": 100},
        {"chunk_index": 2, "start": 200, "end_exclusive": 250, "ticket_count": 50},
    ]


def test_build_parallel_config_plan_assigns_gpu_slots() -> None:
    tmp_path = _temp_dir()
    plan = chunked_execution.build_parallel_config_plan(
        output_root=tmp_path,
        backend_name="openai_compatible",
        config_ids=[
            "slm_only__base_slm1__mem_none",
            "slm_only__base_slm2__mem_none",
            "llm_only__base_llm1__mem_none",
        ],
        parallel_workers=2,
        gpu_ids=["0", "1"],
        ticket_start=0,
        ticket_limit=47000,
        chunk_size=100,
    )

    assert [entry["config_identifier"] for entry in plan] == [
        "slm_only__base_slm1__mem_none",
        "slm_only__base_slm2__mem_none",
        "llm_only__base_llm1__mem_none",
    ]
    assert [entry["worker_slot"] for entry in plan] == [0, 1, 0]
    assert [entry["gpu_id"] for entry in plan] == ["0", "1", "0"]
    assert Path(plan[0]["manifest_path"]).parts[-2:] == (
        "slm_only__base_slm1__mem_none",
        "manifest.json",
    )
    assert Path(plan[1]["output_dir"]).parts[-2:] == (
        "openai_compatible",
        "slm_only__base_slm2__mem_none",
    )
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_manifest_creation_and_update(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=4)
    tmp_path = _temp_dir()

    def fake_run_ticket_pipeline_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        ticket = kwargs["ticket"]
        return {
            "ticket_id": ticket["ticket_id"],
            "topic_group": "Access",
            "final_answer": "ok",
            "reasoning": "r",
            "model_name": kwargs["base_model_name"],
            "metrics": {
                "latency_ms": 1.0,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.001,
            },
            "total_tokens": 15,
            "total_cost_usd": 0.001,
        }

    monkeypatch.setattr(
        chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call
    )

    result = chunked_execution.run_config_chunked(
        config=_sample_config(),
        output_root=tmp_path,
        chunk_size=2,
        ticket_start=0,
        ticket_limit=4,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
    )

    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    assert manifest["backend_name"] == "openai_compatible"
    assert manifest["pricing_version"] == "normalized_v3"
    assert manifest["total_tickets_expected"] == 4
    assert manifest["chunk_size"] == 2
    assert len(manifest["completed_chunks"]) == 2
    assert manifest["failed_chunks"] == []
    status_path = chunked_execution.config_status_path(Path(manifest["output_files"]["config_dir"]))
    progress_log = chunked_execution.config_progress_log_path(
        Path(manifest["output_files"]["config_dir"])
    )
    backend_summary = chunked_execution.backend_status_summary_path(tmp_path, "openai_compatible")
    assert status_path.exists()
    assert progress_log.exists()
    assert backend_summary.exists()

    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    backend_payload = json.loads(backend_summary.read_text(encoding="utf-8"))
    assert status_payload["config_identifier"] == result["config_identifier"]
    assert status_payload["completed_chunks"] == 2
    assert status_payload["pending_chunks"] == 0
    assert status_payload["failed_chunks"] == 0
    assert status_payload["last_completed_ticket_index"] == 3
    assert status_payload["current_status"] == "completed"
    assert any(
        entry["config_identifier"] == result["config_identifier"]
        for entry in backend_payload["configs"]
    )
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_missing_calibrated_head_fails_before_chunk_processing(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=4)
    tmp_path = _temp_dir()
    execute_calls = {"count": 0}

    def fail_dependency_check() -> Dict[str, Any]:
        raise RuntimeError(
            "Missing calibrated encoder head. Run python -m "
            "RouterGym.scripts.train_encoder_calibrated_head or enable explicit fallback."
        )

    def unexpected_execute_chunk(**kwargs: Any) -> Dict[str, Any]:
        del kwargs
        execute_calls["count"] += 1
        raise AssertionError("execute_chunk should not run when dependency validation fails")

    monkeypatch.setattr(chunked_execution, "validate_benchmark_dependencies", fail_dependency_check)
    monkeypatch.setattr(chunked_execution, "execute_chunk", unexpected_execute_chunk)

    with pytest.raises(RuntimeError, match="Missing calibrated encoder head"):
        chunked_execution.run_config_chunked(
            config=_sample_config(),
            output_root=tmp_path,
            chunk_size=2,
            ticket_start=0,
            ticket_limit=4,
            backend_name="openai_compatible",
            resume=True,
            dry_run=False,
        )

    assert execute_calls["count"] == 0
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_manifest_records_100_ticket_chunk_outputs(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=250)
    tmp_path = _temp_dir()

    def fake_run_ticket_pipeline_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        ticket = kwargs["ticket"]
        return {
            "ticket_id": ticket["ticket_id"],
            "topic_group": "Access",
            "final_answer": "ok",
            "reasoning": "r",
            "model_name": kwargs["base_model_name"],
            "metrics": {
                "latency_ms": 1.0,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.001,
            },
            "total_tokens": 15,
            "total_cost_usd": 0.001,
        }

    monkeypatch.setattr(
        chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call
    )

    result = chunked_execution.run_config_chunked(
        config=_sample_config(),
        output_root=tmp_path,
        ticket_start=0,
        ticket_limit=250,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["chunk_size"] == 100
    assert len(manifest["completed_chunks"]) == 3
    assert manifest["completed_chunks"][0]["results_path"].endswith(
        "chunk_0000__tickets_000000_000099__results.jsonl"
    )
    assert manifest["completed_chunks"][1]["results_path"].endswith(
        "chunk_0001__tickets_000100_000199__results.jsonl"
    )
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_execute_chunk_marks_generation_invalid_rows_as_failures(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=1)
    tmp_path = _temp_dir()

    def fake_run_ticket_pipeline_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        ticket = kwargs["ticket"]
        return {
            "ticket_id": ticket["ticket_id"],
            "topic_group": "Access",
            "final_answer": "No valid answer produced",
            "reasoning": "Generation invalid: malformed output",
            "model_name": kwargs["base_model_name"],
            "generation_valid": False,
            "generation_invalid_reason": "malformed output",
            "raw_model_response_text": "???",
            "raw_response_saved": True,
            "metrics": {
                "latency_ms": 1.0,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.001,
            },
            "total_tokens": 15,
            "total_cost_usd": 0.001,
        }

    monkeypatch.setattr(
        chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call
    )

    metadata = chunked_execution.execute_chunk(
        config_identifier=chunked_execution.build_config_identifier(_sample_config()),
        config=_sample_config(),
        output_root=tmp_path,
        backend_name="openai_compatible",
        chunk_spec={"chunk_index": 0, "start": 0, "end_exclusive": 1, "ticket_count": 1},
    )

    results_path = Path(metadata["results_path"])
    failures_path = Path(metadata["failures_path"])
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = [
        json.loads(line)
        for line in failures_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert metadata["failure_count"] == 1
    assert rows[0]["success"] is False
    assert rows[0]["error"]["error_type"] == "GenerationInvalidError"
    assert failures[0]["raw_model_response_text"] == "???"
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_quality_abort_passes_when_rows_are_valid(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=4)
    tmp_path = _temp_dir()

    def fake_run_ticket_pipeline_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        ticket = kwargs["ticket"]
        return {
            "ticket_id": ticket["ticket_id"],
            "topic_group": "Access",
            "final_answer": "Reset the VPN session and sign in again.",
            "reasoning": "Valid structured answer.",
            "resolution_steps": ["Disconnect VPN", "Reconnect with SSO"],
            "model_name": kwargs["base_model_name"],
            "generation_valid": True,
            "placeholder_answer": False,
            "has_real_final_answer": True,
            "has_resolution_steps": True,
            "raw_response_saved": True,
            "raw_model_response_text": '{"final_answer":"ok"}',
            "metrics": {
                "latency_ms": 1.0,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.001,
            },
            "total_tokens": 15,
            "total_cost_usd": 0.001,
        }

    monkeypatch.setattr(
        chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call
    )

    result = chunked_execution.run_config_chunked(
        config=_sample_config(),
        output_root=tmp_path,
        chunk_size=2,
        ticket_start=0,
        ticket_limit=4,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
        enable_quality_abort=True,
    )

    assert result["status"] == "completed"
    assert result["quality_gate"]["enabled"] is True
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["run_status"] == "completed"
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_quality_abort_fails_and_writes_report(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=2)
    tmp_path = _temp_dir()

    def fake_run_ticket_pipeline_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        ticket = kwargs["ticket"]
        return {
            "ticket_id": ticket["ticket_id"],
            "topic_group": "Access",
            "final_answer": "No valid answer produced",
            "reasoning": "Generation invalid: placeholder",
            "resolution_steps": [],
            "model_name": kwargs["base_model_name"],
            "generation_valid": False,
            "generation_invalid_reason": "placeholder output",
            "placeholder_answer": True,
            "has_real_final_answer": False,
            "has_resolution_steps": False,
            "raw_response_saved": True,
            "raw_model_response_text": "bad raw output",
            "metrics": {
                "latency_ms": 1.0,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.001,
            },
            "total_tokens": 15,
            "total_cost_usd": 0.001,
        }

    monkeypatch.setattr(
        chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call
    )

    result = chunked_execution.run_config_chunked(
        config=_sample_config(),
        output_root=tmp_path,
        chunk_size=1,
        ticket_start=0,
        ticket_limit=2,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
        enable_quality_abort=True,
    )

    assert result["status"] == "failed_quality_gate"
    report_json = Path(result["quality_gate_failure_report_json"])
    report_md = Path(result["quality_gate_failure_report_md"])
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["run_status"] == "failed_quality_gate"
    assert report_json.exists()
    assert report_md.exists()
    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_payload["passes_quality_gate"] is False
    assert "placeholder_answer_rate" in report_md.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="failed the quality gate"):
        chunked_execution.run_config_chunked(
            config=_sample_config(),
            output_root=tmp_path,
            chunk_size=1,
            ticket_start=0,
            ticket_limit=2,
            backend_name="openai_compatible",
            resume=True,
            dry_run=False,
            enable_quality_abort=True,
        )
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_runtime_manifest_includes_git_sha_and_model_ids(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=2)
    tmp_path = _temp_dir()

    def fake_run_ticket_pipeline_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        ticket = kwargs["ticket"]
        return {
            "ticket_id": ticket["ticket_id"],
            "topic_group": "Access",
            "final_answer": "Reset the VPN session.",
            "reasoning": "Valid structured answer.",
            "resolution_steps": ["Disconnect", "Reconnect"],
            "model_name": kwargs["base_model_name"],
            "generation_valid": True,
            "placeholder_answer": False,
            "has_real_final_answer": True,
            "has_resolution_steps": True,
            "raw_response_saved": True,
            "raw_model_response_text": '{"final_answer":"ok"}',
            "metrics": {
                "latency_ms": 1.0,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.001,
            },
            "total_tokens": 15,
            "total_cost_usd": 0.001,
        }

    monkeypatch.setattr(
        chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call
    )

    result = chunked_execution.run_config_chunked(
        config=_sample_config(),
        output_root=tmp_path,
        chunk_size=1,
        ticket_start=0,
        ticket_limit=2,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
        enable_quality_abort=True,
        command_line_args=["--config-id", "example"],
    )

    runtime_manifest = json.loads(Path(result["runtime_manifest_path"]).read_text(encoding="utf-8"))
    assert runtime_manifest["git"]["commit_sha"]
    assert runtime_manifest["model_ids"]["slm1"] == "mistralai/Mistral-7B-Instruct-v0.3"
    assert runtime_manifest["model_ids"]["llm1"] == "mistralai/Mistral-Small-24B-Instruct-2501"
    assert runtime_manifest["quality_gate"]["enabled"] is True
    assert runtime_manifest["command_line_args"] == ["--config-id", "example"]
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_chunked_execution_persists_and_passes_max_output_tokens(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=4)
    tmp_path = _temp_dir()
    seen_caps: List[int | None] = []

    def fake_run_ticket_pipeline_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args
        seen_caps.append(kwargs.get("max_output_tokens"))
        ticket = kwargs["ticket"]
        return {
            "ticket_id": ticket["ticket_id"],
            "topic_group": "Access",
            "final_answer": "ok",
            "reasoning": "r",
            "model_name": kwargs["base_model_name"],
            "metrics": {
                "latency_ms": 1.0,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.001,
            },
            "total_tokens": 15,
            "total_cost_usd": 0.001,
        }

    monkeypatch.setattr(
        chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call
    )

    result = chunked_execution.run_config_chunked(
        config=_sample_config(),
        output_root=tmp_path,
        chunk_size=2,
        ticket_start=0,
        ticket_limit=4,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
        max_output_tokens=123,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["generation_settings"]["max_output_tokens"] == 123
    assert seen_caps == [123, 123, 123, 123]
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_progress_log_contains_chunk_save_lines(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=4)
    tmp_path = _temp_dir()

    def fake_run_ticket_pipeline_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        ticket = kwargs["ticket"]
        return {
            "ticket_id": ticket["ticket_id"],
            "topic_group": "Access",
            "final_answer": "ok",
            "reasoning": "r",
            "model_name": kwargs["base_model_name"],
            "metrics": {
                "latency_ms": 1.0,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.001,
            },
            "total_tokens": 15,
            "total_cost_usd": 0.001,
        }

    monkeypatch.setattr(
        chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call
    )
    result = chunked_execution.run_config_chunked(
        config=_sample_config(),
        output_root=tmp_path,
        chunk_size=2,
        ticket_start=0,
        ticket_limit=4,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
    )

    progress_lines = Path(result["progress_log_path"]).read_text(encoding="utf-8").splitlines()
    assert any("startup" in line for line in progress_lines)
    assert any("tickets 0-1 saved" in line for line in progress_lines)
    assert any("tickets 2-3 saved" in line for line in progress_lines)
    assert any("completed | completed 2/2 chunks" in line for line in progress_lines)
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_resume_skips_completed_chunks(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=4)
    tmp_path = _temp_dir()
    seen_ticket_ids: List[str] = []

    def fake_run_ticket_pipeline_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        ticket = kwargs["ticket"]
        seen_ticket_ids.append(str(ticket["ticket_id"]))
        return {
            "ticket_id": ticket["ticket_id"],
            "topic_group": "Access",
            "final_answer": "ok",
            "reasoning": "r",
            "model_name": kwargs["base_model_name"],
            "metrics": {
                "latency_ms": 1.0,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.001,
            },
            "total_tokens": 15,
            "total_cost_usd": 0.001,
        }

    monkeypatch.setattr(
        chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call
    )

    chunked_execution.run_config_chunked(
        config=_sample_config(),
        output_root=tmp_path,
        chunk_size=2,
        ticket_start=0,
        ticket_limit=4,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
    )
    assert seen_ticket_ids == ["0", "1", "2", "3"]

    seen_ticket_ids.clear()
    second_result = chunked_execution.run_config_chunked(
        config=_sample_config(),
        output_root=tmp_path,
        chunk_size=2,
        ticket_start=0,
        ticket_limit=4,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
    )
    assert seen_ticket_ids == []
    merged_results = (
        Path(second_result["merged_results_path"]).read_text(encoding="utf-8").splitlines()
    )
    assert len(merged_results) == 4
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_resume_is_independent_per_config(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=4)
    tmp_path = _temp_dir()
    seen_calls: List[tuple[str, str]] = []

    def fake_run_ticket_pipeline_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args
        seen_calls.append((str(kwargs["base_model_name"]), str(kwargs["ticket"]["ticket_id"])))
        ticket = kwargs["ticket"]
        return {
            "ticket_id": ticket["ticket_id"],
            "topic_group": "Access",
            "final_answer": "ok",
            "reasoning": "r",
            "model_name": kwargs["base_model_name"],
            "metrics": {
                "latency_ms": 1.0,
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.001,
            },
            "total_tokens": 15,
            "total_cost_usd": 0.001,
        }

    monkeypatch.setattr(
        chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call
    )

    chunked_execution.run_benchmark_matrix_chunked(
        output_root=tmp_path,
        config_ids=["slm_only__base_slm1__mem_none"],
        chunk_size=2,
        ticket_start=0,
        ticket_limit=4,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
    )
    assert seen_calls == [("slm1", "0"), ("slm1", "1"), ("slm1", "2"), ("slm1", "3")]

    seen_calls.clear()
    results = chunked_execution.run_benchmark_matrix_chunked(
        output_root=tmp_path,
        config_ids=["slm_only__base_slm1__mem_none", "slm_only__base_slm2__mem_none"],
        chunk_size=2,
        ticket_start=0,
        ticket_limit=4,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
    )

    assert seen_calls == [("slm2", "0"), ("slm2", "1"), ("slm2", "2"), ("slm2", "3")]
    config_ids = [entry["config_identifier"] for entry in results]
    assert config_ids == ["slm_only__base_slm1__mem_none", "slm_only__base_slm2__mem_none"]
    assert Path(results[0]["manifest_path"]) != Path(results[1]["manifest_path"])
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_interrupted_run_resumes_from_last_completed_chunk(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=6)
    tmp_path = _temp_dir()
    config = _sample_config()
    config_identifier = chunked_execution.build_config_identifier(config)
    executed_chunks: List[int] = []

    def interrupting_execute_chunk(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args
        chunk_spec = dict(kwargs["chunk_spec"])
        chunk_index = int(chunk_spec["chunk_index"])
        executed_chunks.append(chunk_index)
        if chunk_index == 1:
            raise KeyboardInterrupt()
        return _write_fake_chunk_outputs(
            output_root=tmp_path,
            config_identifier=config_identifier,
            backend_name="openai_compatible",
            chunk_spec=chunk_spec,
        )

    monkeypatch.setattr(chunked_execution, "execute_chunk", interrupting_execute_chunk)

    with pytest.raises(KeyboardInterrupt):
        chunked_execution.run_config_chunked(
            config=config,
            output_root=tmp_path,
            chunk_size=2,
            ticket_start=0,
            ticket_limit=6,
            backend_name="openai_compatible",
            resume=True,
            dry_run=False,
        )

    manifest_path = chunked_execution.get_manifest_path(
        tmp_path, "openai_compatible", config_identifier
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert executed_chunks == [0, 1]
    assert [entry["chunk_index"] for entry in manifest["completed_chunks"]] == [0]
    assert manifest["failed_chunks"] == []

    resumed_chunks: List[int] = []

    def resumed_execute_chunk(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args
        chunk_spec = dict(kwargs["chunk_spec"])
        resumed_chunks.append(int(chunk_spec["chunk_index"]))
        return _write_fake_chunk_outputs(
            output_root=tmp_path,
            config_identifier=config_identifier,
            backend_name="openai_compatible",
            chunk_spec=chunk_spec,
        )

    monkeypatch.setattr(chunked_execution, "execute_chunk", resumed_execute_chunk)

    result = chunked_execution.run_config_chunked(
        config=config,
        output_root=tmp_path,
        chunk_size=2,
        ticket_start=0,
        ticket_limit=6,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
    )

    assert result["startup_resume_state"] == {
        "completed_chunks": 1,
        "pending_chunks": 2,
        "failed_chunks": 0,
        "total_chunks": 3,
    }
    assert resumed_chunks == [1, 2]
    merged_results = Path(result["merged_results_path"]).read_text(encoding="utf-8").splitlines()
    assert len(merged_results) == 6
    assert [json.loads(line)["ticket_index"] for line in merged_results] == [0, 1, 2, 3, 4, 5]
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_failed_chunks_retry_without_duplicating_completed_output(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=6)
    tmp_path = _temp_dir()
    config = _sample_config()
    config_identifier = chunked_execution.build_config_identifier(config)
    first_attempts: List[int] = []

    def flaky_execute_chunk(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args
        chunk_spec = dict(kwargs["chunk_spec"])
        chunk_index = int(chunk_spec["chunk_index"])
        first_attempts.append(chunk_index)
        if chunk_index == 1:
            raise RuntimeError("simulated worker failure")
        return _write_fake_chunk_outputs(
            output_root=tmp_path,
            config_identifier=config_identifier,
            backend_name="openai_compatible",
            chunk_spec=chunk_spec,
        )

    monkeypatch.setattr(chunked_execution, "execute_chunk", flaky_execute_chunk)

    first_result = chunked_execution.run_config_chunked(
        config=config,
        output_root=tmp_path,
        chunk_size=2,
        ticket_start=0,
        ticket_limit=6,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
    )

    assert first_attempts == [0, 1, 2]
    assert first_result["final_resume_state"] == {
        "completed_chunks": 2,
        "pending_chunks": 0,
        "failed_chunks": 1,
        "total_chunks": 3,
    }

    manifest_path = chunked_execution.get_manifest_path(
        tmp_path, "openai_compatible", config_identifier
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [entry["chunk_index"] for entry in manifest["completed_chunks"]] == [0, 2]
    assert [entry["chunk_index"] for entry in manifest["failed_chunks"]] == [1]

    retry_attempts: List[int] = []

    def successful_execute_chunk(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        del args
        chunk_spec = dict(kwargs["chunk_spec"])
        retry_attempts.append(int(chunk_spec["chunk_index"]))
        return _write_fake_chunk_outputs(
            output_root=tmp_path,
            config_identifier=config_identifier,
            backend_name="openai_compatible",
            chunk_spec=chunk_spec,
        )

    monkeypatch.setattr(chunked_execution, "execute_chunk", successful_execute_chunk)

    second_result = chunked_execution.run_config_chunked(
        config=config,
        output_root=tmp_path,
        chunk_size=2,
        ticket_start=0,
        ticket_limit=6,
        backend_name="openai_compatible",
        resume=True,
        dry_run=False,
    )

    assert second_result["startup_resume_state"] == {
        "completed_chunks": 2,
        "pending_chunks": 0,
        "failed_chunks": 1,
        "total_chunks": 3,
    }
    assert retry_attempts == [1]
    merged_results = (
        Path(second_result["merged_results_path"]).read_text(encoding="utf-8").splitlines()
    )
    assert len(merged_results) == 6
    assert [json.loads(line)["ticket_index"] for line in merged_results] == [0, 1, 2, 3, 4, 5]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["failed_chunks"] == []
    assert [entry["chunk_index"] for entry in manifest["completed_chunks"]] == [0, 1, 2]
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_merge_completed_chunks_preserves_order() -> None:
    tmp_path = _temp_dir()
    config = _sample_config()
    config_id = chunked_execution.build_config_identifier(config)
    config_dir = chunked_execution.get_config_output_dir(tmp_path, "openai_compatible", config_id)
    chunks_dir = config_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk0 = {"chunk_index": 0, "start": 0, "end_exclusive": 2, "ticket_count": 2}
    chunk1 = {"chunk_index": 1, "start": 2, "end_exclusive": 4, "ticket_count": 2}

    results0 = chunked_execution.chunk_results_path(config_dir, chunk0)
    results1 = chunked_execution.chunk_results_path(config_dir, chunk1)
    failures0 = chunked_execution.chunk_failures_path(config_dir, chunk0)
    failures1 = chunked_execution.chunk_failures_path(config_dir, chunk1)
    meta0 = chunked_execution.chunk_metadata_path(config_dir, chunk0)
    meta1 = chunked_execution.chunk_metadata_path(config_dir, chunk1)

    results0.write_text('{"ticket_index": 0}\n{"ticket_index": 1}\n', encoding="utf-8")
    results1.write_text('{"ticket_index": 2}\n{"ticket_index": 3}\n', encoding="utf-8")
    failures0.write_text("", encoding="utf-8")
    failures1.write_text('{"ticket_index": 2, "success": false}\n', encoding="utf-8")
    meta0.write_text("{}", encoding="utf-8")
    meta1.write_text("{}", encoding="utf-8")

    manifest_path = config_dir / "manifest.json"
    manifest = {
        "config_identifier": config_id,
        "output_files": {"config_dir": str(config_dir)},
        "completed_chunks": [
            {
                "chunk_index": 0,
                "results_path": str(results0),
                "failures_path": str(failures0),
                "metadata_path": str(meta0),
            },
            {
                "chunk_index": 1,
                "results_path": str(results1),
                "failures_path": str(failures1),
                "metadata_path": str(meta1),
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    merged = chunked_execution.merge_completed_chunks(manifest_path)
    merged_results = Path(merged["merged_results_path"]).read_text(encoding="utf-8").splitlines()
    merged_failures = Path(merged["merged_failures_path"]).read_text(encoding="utf-8").splitlines()
    assert merged_results == [
        '{"ticket_index": 0}',
        '{"ticket_index": 1}',
        '{"ticket_index": 2}',
        '{"ticket_index": 3}',
    ]
    assert merged_failures == ['{"ticket_index": 2, "success": false}']
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_dry_run_summary_includes_manifest_and_first_chunk(monkeypatch: Any) -> None:
    _patch_dataset(monkeypatch, size=100)
    tmp_path = _temp_dir()
    summary = chunked_execution.run_benchmark_matrix_chunked(
        output_root=tmp_path,
        config_id="slm_only__base_slm1__mem_none",
        chunk_size=40,
        ticket_start=0,
        ticket_limit=100,
        backend_name="openai_compatible",
        resume=True,
        dry_run=True,
    )[0]

    assert summary["status"] == "dry_run"
    assert summary["config_identifier"] == "slm_only__base_slm1__mem_none"
    assert summary["chunk_size"] == 40
    assert summary["manifest_path"].endswith("manifest.json")
    assert "chunk_0000" in summary["first_chunk_path"]
    assert summary["resume_state"] == {
        "completed_chunks": 0,
        "pending_chunks": 3,
        "failed_chunks": 0,
        "total_chunks": 3,
    }
    assert "pending" in summary["resume_behavior_summary"]
    shutil.rmtree(tmp_path, ignore_errors=True)
