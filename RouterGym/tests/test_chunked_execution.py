"""Tests for chunked/resumable production execution."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

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
    assert plan[0]["manifest_path"].endswith("slm_only__base_slm1__mem_none\\manifest.json")
    assert plan[1]["output_dir"].endswith("openai_compatible\\slm_only__base_slm2__mem_none")
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

    monkeypatch.setattr(chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call)

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

    monkeypatch.setattr(chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call)

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

    monkeypatch.setattr(chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call)

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
    assert seen_ticket_ids == []
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

    monkeypatch.setattr(chunked_execution, "run_ticket_pipeline_call", fake_run_ticket_pipeline_call)

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
    assert "pending" in summary["resume_behavior_summary"]
    shutil.rmtree(tmp_path, ignore_errors=True)
