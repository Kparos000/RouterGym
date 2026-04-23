"""Chunked, resumable benchmark execution over the frozen benchmark matrix.

This module wraps the existing per-ticket pipeline without changing routing or
benchmark semantics. It adds:

- deterministic chunk planning
- per-config manifests
- incremental chunk output files
- resumable execution
- merge/finalize helpers
"""

from __future__ import annotations

import json
import os
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from RouterGym.benchmark_spec import (
    BENCHMARK_SPEC_VERSION,
    PRICING_VERSION,
    PRODUCTION_CHUNK_SIZE,
    build_final_benchmark_matrix,
)
from RouterGym.engines.model_registry import get_model_backend


DEFAULT_OUTPUT_ROOT = Path("RouterGym/results/production_runs")


def load_ticket_dataset(*, limit: Optional[int], start: int):
    """Lazy proxy to the ticket dataset loader."""

    from RouterGym.data.tickets.dataset_loader import load_dataset as _load_dataset

    return _load_dataset(n=limit, start=start)


def run_ticket_pipeline_call(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Lazy proxy to the benchmark ticket pipeline."""

    from RouterGym.agents.generator import run_ticket_pipeline as _run_ticket_pipeline

    return _run_ticket_pipeline(*args, **kwargs)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_config_identifier(config: Mapping[str, str]) -> str:
    """Build a stable identifier for one frozen benchmark configuration."""

    parts = [
        str(config["router_mode"]).strip(),
        f"base_{str(config['base_model']).strip()}",
    ]
    escalation_model = str(config.get("escalation_model", "") or "").strip()
    if escalation_model:
        parts.append(f"esc_{escalation_model}")
    parts.append(f"mem_{str(config['memory_mode']).strip()}")
    return "__".join(parts)


def get_frozen_config_map() -> Dict[str, Dict[str, str]]:
    """Return the frozen benchmark matrix indexed by config identifier."""

    return {
        build_config_identifier(config): dict(config)
        for config in build_final_benchmark_matrix()
    }


def resolve_selected_configs(
    *,
    config_id: Optional[str] = None,
    config_ids: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Dict[str, str]]]:
    """Return one or many frozen configs while preserving explicit input order."""

    if config_id and config_ids:
        raise ValueError("Provide either config_id or config_ids, not both.")

    config_map = get_frozen_config_map()
    if config_id:
        selected_ids = [config_id]
    elif config_ids:
        selected_ids = [str(item) for item in config_ids]
    else:
        selected_ids = sorted(config_map)

    selected: List[Tuple[str, Dict[str, str]]] = []
    for selected_id in selected_ids:
        if selected_id not in config_map:
            raise KeyError(f"Unknown config identifier: {selected_id}")
        selected.append((selected_id, dict(config_map[selected_id])))
    return selected


def build_chunk_plan(
    *,
    total_tickets: int,
    chunk_size: int,
    start_index: int = 0,
) -> List[Dict[str, int]]:
    """Create deterministic chunk boundaries for a ticket slice."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if total_tickets < 0:
        raise ValueError("total_tickets must be >= 0")

    plan: List[Dict[str, int]] = []
    offset = 0
    chunk_index = 0
    while offset < total_tickets:
        chunk_start = start_index + offset
        chunk_end_exclusive = min(chunk_start + chunk_size, start_index + total_tickets)
        plan.append(
            {
                "chunk_index": chunk_index,
                "start": chunk_start,
                "end_exclusive": chunk_end_exclusive,
                "ticket_count": chunk_end_exclusive - chunk_start,
            }
        )
        offset += chunk_size
        chunk_index += 1
    return plan


def resolve_backend_details(backend_override: Optional[str] = None) -> Dict[str, Any]:
    """Return the active backend plus relevant serving metadata."""

    backend_name = (backend_override or get_model_backend()).strip().lower()
    if not backend_name:
        backend_name = "hf_inference"
    details: Dict[str, Any] = {"backend_name": backend_name}
    if backend_name == "openai_compatible":
        details["openai_base_url"] = (
            os.getenv("ROUTERGYM_OPENAI_BASE_URL")
            or os.getenv("ROUTERGYM_VLLM_BASE_URL")
            or "http://localhost:8000/v1"
        )
        details["openai_api_key_present"] = bool(
            os.getenv("ROUTERGYM_OPENAI_API_KEY") or os.getenv("ROUTERGYM_VLLM_API_KEY")
        )
    return details


def build_parallel_config_plan(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    backend_name: str,
    config_id: Optional[str] = None,
    config_ids: Optional[Sequence[str]] = None,
    parallel_workers: int = 1,
    gpu_ids: Optional[Sequence[str]] = None,
    ticket_start: int = 0,
    ticket_limit: Optional[int] = None,
    chunk_size: int = PRODUCTION_CHUNK_SIZE,
) -> List[Dict[str, Any]]:
    """Build a deterministic config-level worker assignment plan."""

    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be > 0")

    selected = resolve_selected_configs(config_id=config_id, config_ids=config_ids)
    normalized_gpu_ids = [str(gpu_id).strip() for gpu_id in (gpu_ids or []) if str(gpu_id).strip()]
    if normalized_gpu_ids and parallel_workers > len(normalized_gpu_ids):
        raise ValueError("parallel_workers cannot exceed the number of provided gpu_ids.")

    worker_count = min(parallel_workers, len(selected))
    if normalized_gpu_ids:
        worker_count = min(worker_count, len(normalized_gpu_ids))
    worker_count = max(worker_count, 1)

    plan: List[Dict[str, Any]] = []
    for queue_index, (selected_id, config) in enumerate(selected):
        worker_slot = queue_index % worker_count
        gpu_id = normalized_gpu_ids[worker_slot] if normalized_gpu_ids else None
        config_dir = get_config_output_dir(output_root, backend_name, selected_id)
        plan.append(
            {
                "queue_index": queue_index,
                "worker_slot": worker_slot,
                "gpu_id": gpu_id,
                "config_identifier": selected_id,
                "config": dict(config),
                "backend_name": backend_name,
                "output_dir": str(config_dir),
                "manifest_path": str(get_manifest_path(output_root, backend_name, selected_id)),
                "ticket_start": int(ticket_start),
                "ticket_limit": ticket_limit,
                "chunk_size": int(chunk_size),
            }
        )
    return plan


@contextmanager
def backend_override(backend_name: Optional[str]) -> Iterator[None]:
    """Temporarily override ROUTERGYM_MODEL_BACKEND for one execution scope."""

    if not backend_name:
        yield
        return
    original = os.getenv("ROUTERGYM_MODEL_BACKEND")
    os.environ["ROUTERGYM_MODEL_BACKEND"] = backend_name
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("ROUTERGYM_MODEL_BACKEND", None)
        else:
            os.environ["ROUTERGYM_MODEL_BACKEND"] = original


def get_config_output_dir(output_root: Path, backend_name: str, config_identifier: str) -> Path:
    """Return the directory for one config/backend production run."""

    return output_root / backend_name / config_identifier


def get_manifest_path(output_root: Path, backend_name: str, config_identifier: str) -> Path:
    """Return the manifest path for one config/backend production run."""

    return get_config_output_dir(output_root, backend_name, config_identifier) / "manifest.json"


def chunk_file_stem(chunk_spec: Mapping[str, int]) -> str:
    """Return a deterministic chunk stem used across result/failure/meta files."""

    end_inclusive = int(chunk_spec["end_exclusive"]) - 1
    return (
        f"chunk_{int(chunk_spec['chunk_index']):04d}"
        f"__tickets_{int(chunk_spec['start']):06d}_{end_inclusive:06d}"
    )


def chunk_results_path(config_dir: Path, chunk_spec: Mapping[str, int]) -> Path:
    return config_dir / "chunks" / f"{chunk_file_stem(chunk_spec)}__results.jsonl"


def chunk_failures_path(config_dir: Path, chunk_spec: Mapping[str, int]) -> Path:
    return config_dir / "chunks" / f"{chunk_file_stem(chunk_spec)}__failures.jsonl"


def chunk_metadata_path(config_dir: Path, chunk_spec: Mapping[str, int]) -> Path:
    return config_dir / "chunks" / f"{chunk_file_stem(chunk_spec)}__metadata.json"


def merged_results_path(config_dir: Path, config_identifier: str) -> Path:
    return config_dir / "merged" / f"{config_identifier}__results_merged.jsonl"


def merged_failures_path(config_dir: Path, config_identifier: str) -> Path:
    return config_dir / "merged" / f"{config_identifier}__failures_merged.jsonl"


def _load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def initialize_manifest(
    *,
    config_identifier: str,
    config: Mapping[str, str],
    output_root: Path,
    backend_details: Mapping[str, Any],
    total_tickets_expected: int,
    ticket_start: int,
    ticket_limit: Optional[int],
    chunk_size: int,
) -> Dict[str, Any]:
    """Build an initial manifest for one config/backend run."""

    backend_name = str(backend_details["backend_name"])
    config_dir = get_config_output_dir(output_root, backend_name, config_identifier)
    return {
        "spec_version": BENCHMARK_SPEC_VERSION,
        "pricing_version": PRICING_VERSION,
        "backend_name": backend_name,
        "backend_details": dict(backend_details),
        "config_identifier": config_identifier,
        "config": dict(config),
        "total_tickets_expected": int(total_tickets_expected),
        "ticket_start": int(ticket_start),
        "ticket_limit": ticket_limit,
        "chunk_size": int(chunk_size),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "last_run_started_at": "",
        "last_run_completed_at": "",
        "run_status": "initialized",
        "output_files": {
            "config_dir": str(config_dir),
            "chunks_dir": str(config_dir / "chunks"),
            "merged_dir": str(config_dir / "merged"),
            "merged_results_path": "",
            "merged_failures_path": "",
        },
        "completed_chunks": [],
        "failed_chunks": [],
    }


def ensure_manifest(
    *,
    config_identifier: str,
    config: Mapping[str, str],
    output_root: Path,
    backend_details: Mapping[str, Any],
    total_tickets_expected: int,
    ticket_start: int,
    ticket_limit: Optional[int],
    chunk_size: int,
) -> Dict[str, Any]:
    """Load or initialize a manifest, rejecting incompatible resume settings."""

    manifest_path = get_manifest_path(output_root, str(backend_details["backend_name"]), config_identifier)
    existing = _load_manifest(manifest_path)
    if existing is None:
        manifest = initialize_manifest(
            config_identifier=config_identifier,
            config=config,
            output_root=output_root,
            backend_details=backend_details,
            total_tickets_expected=total_tickets_expected,
            ticket_start=ticket_start,
            ticket_limit=ticket_limit,
            chunk_size=chunk_size,
        )
        _write_manifest(manifest_path, manifest)
        return manifest

    if int(existing.get("chunk_size", chunk_size)) != int(chunk_size):
        raise ValueError("Existing manifest chunk_size does not match requested chunk_size.")
    if int(existing.get("total_tickets_expected", total_tickets_expected)) != int(total_tickets_expected):
        raise ValueError("Existing manifest total_tickets_expected does not match requested ticket set.")
    if str(existing.get("backend_name", "")) != str(backend_details["backend_name"]):
        raise ValueError("Existing manifest backend_name does not match requested backend.")
    return existing


def _completed_chunk_map(manifest: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        int(entry["chunk_index"]): dict(entry)
        for entry in manifest.get("completed_chunks", [])
        if isinstance(entry, dict) and "chunk_index" in entry
    }


def _failed_chunk_map(manifest: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        int(entry["chunk_index"]): dict(entry)
        for entry in manifest.get("failed_chunks", [])
        if isinstance(entry, dict) and "chunk_index" in entry
    }


def _set_completed_chunk(manifest: Dict[str, Any], entry: Mapping[str, Any]) -> None:
    completed = _completed_chunk_map(manifest)
    failed = _failed_chunk_map(manifest)
    index = int(entry["chunk_index"])
    completed[index] = dict(entry)
    failed.pop(index, None)
    manifest["completed_chunks"] = [completed[idx] for idx in sorted(completed)]
    manifest["failed_chunks"] = [failed[idx] for idx in sorted(failed)]


def _set_failed_chunk(manifest: Dict[str, Any], entry: Mapping[str, Any]) -> None:
    completed = _completed_chunk_map(manifest)
    failed = _failed_chunk_map(manifest)
    index = int(entry["chunk_index"])
    failed[index] = dict(entry)
    manifest["completed_chunks"] = [completed[idx] for idx in sorted(completed)]
    manifest["failed_chunks"] = [failed[idx] for idx in sorted(failed)]


def _update_manifest_status(manifest: Dict[str, Any], chunk_plan: Sequence[Mapping[str, int]]) -> None:
    completed_count = len(manifest.get("completed_chunks", []))
    failed_count = len(manifest.get("failed_chunks", []))
    total_chunks = len(chunk_plan)
    if total_chunks and completed_count == total_chunks:
        status = "completed"
    elif completed_count or failed_count:
        status = "partial"
    else:
        status = "initialized"
    manifest["run_status"] = status
    manifest["updated_at"] = _utc_now()


def describe_resume_behavior(manifest: Mapping[str, Any], chunk_plan: Sequence[Mapping[str, int]]) -> str:
    """Return a short, human-readable summary of resume behavior."""

    completed = _completed_chunk_map(manifest)
    failed = _failed_chunk_map(manifest)
    pending = len(chunk_plan) - len(completed)
    return (
        f"{len(completed)} completed chunk(s) will be skipped; "
        f"{pending} chunk(s) pending; "
        f"{len(failed)} failed chunk(s) eligible for retry"
    )


def _is_chunk_complete(config_dir: Path, manifest: Mapping[str, Any], chunk_spec: Mapping[str, int]) -> bool:
    completed = _completed_chunk_map(manifest)
    entry = completed.get(int(chunk_spec["chunk_index"]))
    if entry is None:
        return False
    results_path = Path(str(entry.get("results_path", chunk_results_path(config_dir, chunk_spec))))
    metadata_path = Path(str(entry.get("metadata_path", chunk_metadata_path(config_dir, chunk_spec))))
    return results_path.exists() and metadata_path.exists()


def _chunk_error_record(
    *,
    config_identifier: str,
    config: Mapping[str, str],
    chunk_spec: Mapping[str, int],
    exc: BaseException,
) -> Dict[str, Any]:
    return {
        "config_identifier": config_identifier,
        "router_mode": config.get("router_mode", ""),
        "memory_mode": config.get("memory_mode", ""),
        "base_model_name": config.get("base_model", ""),
        "escalation_model_name": config.get("escalation_model", ""),
        "chunk_index": int(chunk_spec["chunk_index"]),
        "start": int(chunk_spec["start"]),
        "end_exclusive": int(chunk_spec["end_exclusive"]),
        "ticket_count": int(chunk_spec["ticket_count"]),
        "failed_at": _utc_now(),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _jsonl_write(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _load_ticket_slice(start: int, limit: int) -> List[Dict[str, Any]]:
    df = load_ticket_dataset(limit=limit, start=start)
    tickets: List[Dict[str, Any]] = []
    for local_idx, row in df.iterrows():
        global_index = start + int(local_idx)
        tickets.append(
            {
                "ticket_index": global_index,
                "ticket_id": str(global_index),
                "text": str(row["text"]),
                "gold_label": str(row.get("label", "")),
            }
        )
    return tickets


def _result_record(
    *,
    config_identifier: str,
    config: Mapping[str, str],
    backend_name: str,
    chunk_spec: Mapping[str, int],
    ticket: Mapping[str, Any],
    result: Optional[Mapping[str, Any]],
    error: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = dict(result or {})
    record: Dict[str, Any] = {
        "config_identifier": config_identifier,
        "router_mode": config.get("router_mode", ""),
        "memory_mode": config.get("memory_mode", ""),
        "base_model_name": config.get("base_model", ""),
        "escalation_model_name": config.get("escalation_model", ""),
        "backend_name": backend_name,
        "chunk_index": int(chunk_spec["chunk_index"]),
        "ticket_index": int(ticket["ticket_index"]),
        "ticket_id": str(ticket["ticket_id"]),
        "gold_label": str(ticket["gold_label"]),
        "success": error is None,
        "error": dict(error) if error is not None else None,
    }
    record.update(payload)
    return record


def execute_chunk(
    *,
    config_identifier: str,
    config: Mapping[str, str],
    output_root: Path,
    backend_name: str,
    chunk_spec: Mapping[str, int],
) -> Dict[str, Any]:
    """Execute one chunk and write result/failure artifacts atomically."""

    config_dir = get_config_output_dir(output_root, backend_name, config_identifier)
    results_path = chunk_results_path(config_dir, chunk_spec)
    failures_path = chunk_failures_path(config_dir, chunk_spec)
    metadata_path = chunk_metadata_path(config_dir, chunk_spec)

    temp_results = results_path.with_suffix(results_path.suffix + ".tmp")
    temp_failures = failures_path.with_suffix(failures_path.suffix + ".tmp")
    temp_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")

    tickets = _load_ticket_slice(int(chunk_spec["start"]), int(chunk_spec["ticket_count"]))
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for ticket in tickets:
        error: Optional[Dict[str, Any]] = None
        result: Optional[Dict[str, Any]] = None
        try:
            result = run_ticket_pipeline_call(
                ticket={"text": ticket["text"], "ticket_id": ticket["ticket_id"]},
                router_mode=str(config["router_mode"]),
                memory_mode=str(config["memory_mode"]),
                base_model_name=str(config["base_model"]),
                escalation_model_name=(str(config["escalation_model"]) if config.get("escalation_model") else None),
            )
        except Exception as exc:  # pragma: no cover - exercised via tests with mocks
            error = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        record = _result_record(
            config_identifier=config_identifier,
            config=config,
            backend_name=backend_name,
            chunk_spec=chunk_spec,
            ticket=ticket,
            result=result,
            error=error,
        )
        results.append(record)
        if error is not None:
            failures.append(record)

    metadata: Dict[str, Any] = {
        "config_identifier": config_identifier,
        "backend_name": backend_name,
        "chunk_index": int(chunk_spec["chunk_index"]),
        "start": int(chunk_spec["start"]),
        "end_exclusive": int(chunk_spec["end_exclusive"]),
        "ticket_count": int(chunk_spec["ticket_count"]),
        "row_count": len(results),
        "success_count": sum(1 for row in results if bool(row.get("success"))),
        "failure_count": len(failures),
        "results_path": str(results_path),
        "failures_path": str(failures_path),
        "metadata_path": str(metadata_path),
        "completed_at": _utc_now(),
    }

    _jsonl_write(temp_results, results)
    _jsonl_write(temp_failures, failures)
    temp_metadata.parent.mkdir(parents=True, exist_ok=True)
    temp_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_results.replace(results_path)
    temp_failures.replace(failures_path)
    temp_metadata.replace(metadata_path)
    return metadata


def run_config_chunked(
    *,
    config: Mapping[str, str],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    chunk_size: int = PRODUCTION_CHUNK_SIZE,
    ticket_start: int = 0,
    ticket_limit: Optional[int] = None,
    backend_name: Optional[str] = None,
    resume: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run one frozen config in deterministic chunks with manifest-based resume."""

    backend_details = resolve_backend_details(backend_name)
    config_identifier = build_config_identifier(config)

    ticket_df = load_ticket_dataset(limit=ticket_limit, start=ticket_start)
    total_tickets_expected = len(ticket_df)
    chunk_plan = build_chunk_plan(
        total_tickets=total_tickets_expected,
        chunk_size=chunk_size,
        start_index=ticket_start,
    )

    manifest = ensure_manifest(
        config_identifier=config_identifier,
        config=config,
        output_root=output_root,
        backend_details=backend_details,
        total_tickets_expected=total_tickets_expected,
        ticket_start=ticket_start,
        ticket_limit=ticket_limit,
        chunk_size=chunk_size,
    )
    manifest_path = get_manifest_path(output_root, str(backend_details["backend_name"]), config_identifier)
    resume_summary = describe_resume_behavior(manifest, chunk_plan)
    config_dir = get_config_output_dir(output_root, str(backend_details["backend_name"]), config_identifier)

    if dry_run:
        return {
            "status": "dry_run",
            "config_identifier": config_identifier,
            "chunk_size": chunk_size,
            "total_tickets_expected": total_tickets_expected,
            "manifest_path": str(manifest_path),
            "first_chunk_path": (
                str(chunk_results_path(config_dir, chunk_plan[0])) if chunk_plan else ""
            ),
            "resume_behavior_summary": resume_summary,
        }

    with backend_override(backend_name):
        manifest["last_run_started_at"] = _utc_now()
        manifest["updated_at"] = _utc_now()
        manifest["run_status"] = "running"
        _write_manifest(manifest_path, manifest)

        for chunk_spec in chunk_plan:
            if resume and _is_chunk_complete(config_dir, manifest, chunk_spec):
                continue
            try:
                chunk_metadata = execute_chunk(
                    config_identifier=config_identifier,
                    config=config,
                    output_root=output_root,
                    backend_name=str(backend_details["backend_name"]),
                    chunk_spec=chunk_spec,
                )
                _set_completed_chunk(manifest, chunk_metadata)
            except Exception as exc:  # pragma: no cover - exercised via tests with mocks
                _set_failed_chunk(
                    manifest,
                    _chunk_error_record(
                        config_identifier=config_identifier,
                        config=config,
                        chunk_spec=chunk_spec,
                        exc=exc,
                    ),
                )
            _update_manifest_status(manifest, chunk_plan)
            _write_manifest(manifest_path, manifest)

    merged = merge_completed_chunks(manifest_path)
    manifest = _load_manifest(manifest_path) or manifest
    manifest["last_run_completed_at"] = _utc_now()
    manifest["output_files"]["merged_results_path"] = merged["merged_results_path"]
    manifest["output_files"]["merged_failures_path"] = merged["merged_failures_path"]
    _update_manifest_status(manifest, chunk_plan)
    _write_manifest(manifest_path, manifest)
    return {
        "status": manifest["run_status"],
        "config_identifier": config_identifier,
        "manifest_path": str(manifest_path),
        "merged_results_path": merged["merged_results_path"],
        "merged_failures_path": merged["merged_failures_path"],
        "resume_behavior_summary": resume_summary,
    }


def merge_completed_chunks(manifest_path: Path) -> Dict[str, str]:
    """Merge completed chunk outputs into final per-config JSONL artifacts."""

    manifest = _load_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    config_identifier = str(manifest["config_identifier"])
    config_dir = Path(str(manifest["output_files"]["config_dir"]))
    merged_results = merged_results_path(config_dir, config_identifier)
    merged_failures = merged_failures_path(config_dir, config_identifier)
    merged_results.parent.mkdir(parents=True, exist_ok=True)

    completed_chunks = sorted(
        (dict(entry) for entry in manifest.get("completed_chunks", []) if isinstance(entry, dict)),
        key=lambda entry: int(entry["chunk_index"]),
    )

    with merged_results.open("w", encoding="utf-8") as results_handle:
        for entry in completed_chunks:
            source = Path(str(entry["results_path"]))
            if source.exists():
                results_handle.write(source.read_text(encoding="utf-8"))

    with merged_failures.open("w", encoding="utf-8") as failures_handle:
        for entry in completed_chunks:
            source = Path(str(entry["failures_path"]))
            if source.exists():
                failures_handle.write(source.read_text(encoding="utf-8"))

    return {
        "merged_results_path": str(merged_results),
        "merged_failures_path": str(merged_failures),
    }


def run_benchmark_matrix_chunked(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config_id: Optional[str] = None,
    config_ids: Optional[Sequence[str]] = None,
    chunk_size: int = PRODUCTION_CHUNK_SIZE,
    ticket_start: int = 0,
    ticket_limit: Optional[int] = None,
    backend_name: Optional[str] = None,
    resume: bool = True,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Run one or many frozen configs through the chunked execution wrapper."""

    selected = resolve_selected_configs(config_id=config_id, config_ids=config_ids)

    results: List[Dict[str, Any]] = []
    for selected_id, config in selected:
        result = run_config_chunked(
            config=config,
            output_root=output_root,
            chunk_size=chunk_size,
            ticket_start=ticket_start,
            ticket_limit=ticket_limit,
            backend_name=backend_name,
            resume=resume,
            dry_run=dry_run,
        )
        result.setdefault("config_identifier", selected_id)
        results.append(result)
    return results


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "backend_override",
    "build_chunk_plan",
    "build_config_identifier",
    "chunk_failures_path",
    "chunk_file_stem",
    "chunk_metadata_path",
    "chunk_results_path",
    "build_parallel_config_plan",
    "describe_resume_behavior",
    "ensure_manifest",
    "execute_chunk",
    "get_config_output_dir",
    "get_frozen_config_map",
    "get_manifest_path",
    "merge_completed_chunks",
    "resolve_backend_details",
    "resolve_selected_configs",
    "run_benchmark_matrix_chunked",
    "run_config_chunked",
]
