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
import platform
import time
import threading
import subprocess
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from RouterGym.benchmark_spec import (
    BENCHMARK_SPEC_VERSION,
    PRICING_VERSION,
    PRODUCTION_CHUNK_SIZE,
    build_final_benchmark_matrix,
)
from RouterGym.engines.model_registry import LLM_MODELS, SLM_MODELS, get_model_backend
from RouterGym.scripts.check_generation_quality_gate import (
    EMPTY_STEPS_THRESHOLD,
    GENERATION_VALID_THRESHOLD,
    PLACEHOLDER_THRESHOLD,
    RAW_RESPONSE_THRESHOLD,
    build_thresholds,
    summarize_quality,
)


DEFAULT_OUTPUT_ROOT = Path("RouterGym/results/production_runs")
_JSON_WRITE_GUARDS: Dict[str, threading.Lock] = {}
_JSON_WRITE_GUARDS_LOCK = threading.Lock()
RELEVANT_ENV_VAR_NAMES = [
    "ROUTERGYM_MODEL_BACKEND",
    "ROUTERGYM_OPENAI_BASE_URL",
    "ROUTERGYM_OPENAI_API_KEY",
    "ROUTERGYM_VLLM_BASE_URL",
    "ROUTERGYM_VLLM_API_KEY",
    "ROUTERGYM_WORKER_SLOT",
    "ROUTERGYM_ASSIGNED_GPU_ID",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
]


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


def config_progress_log_path(config_dir: Path) -> Path:
    return config_dir / "progress.log"


def config_status_path(config_dir: Path) -> Path:
    return config_dir / "status.json"


def runtime_manifest_path(config_dir: Path) -> Path:
    return config_dir / "runtime_manifest.json"


def quality_gate_report_json_path(config_dir: Path) -> Path:
    return config_dir / "quality_gate_failure_report.json"


def quality_gate_report_md_path(config_dir: Path) -> Path:
    return config_dir / "quality_gate_failure_report.md"


def backend_status_summary_path(output_root: Path, backend_name: str) -> Path:
    return output_root / backend_name / "run_status.json"


def merged_results_path(config_dir: Path, config_identifier: str) -> Path:
    return config_dir / "merged" / f"{config_identifier}__results_merged.jsonl"


def merged_failures_path(config_dir: Path, config_identifier: str) -> Path:
    return config_dir / "merged" / f"{config_identifier}__failures_merged.jsonl"


def _load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _get_json_write_lock(path: Path) -> threading.Lock:
    path_key = str(path.resolve())
    with _JSON_WRITE_GUARDS_LOCK:
        lock = _JSON_WRITE_GUARDS.get(path_key)
        if lock is None:
            lock = threading.Lock()
            _JSON_WRITE_GUARDS[path_key] = lock
        return lock


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[PermissionError] = None
    with _get_json_write_lock(path):
        for attempt in range(5):
            temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                temp_path.write_text(
                    json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                os.replace(temp_path, path)
                return
            except PermissionError as exc:
                last_error = exc
                if attempt == 4:
                    raise
                time.sleep(0.01 * float(attempt + 1))
            finally:
                temp_path.unlink(missing_ok=True)
    if last_error is not None:
        raise last_error


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    return datetime.fromisoformat(raw_value)


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _execution_context() -> Dict[str, Optional[str]]:
    worker_slot_raw = str(os.getenv("ROUTERGYM_WORKER_SLOT", "")).strip()
    gpu_id = (
        str(os.getenv("ROUTERGYM_ASSIGNED_GPU_ID", "")).strip()
        or str(os.getenv("CUDA_VISIBLE_DEVICES", "")).strip()
        or None
    )
    return {
        "worker_slot": worker_slot_raw or None,
        "gpu_id": gpu_id,
    }


def _safe_subprocess_output(command: Sequence[str], *, timeout_seconds: int = 5) -> Optional[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception:
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def _git_value(*args: str) -> str:
    output = _safe_subprocess_output(["git", *args])
    return str(output or "").strip()


def _package_version(distribution_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution_name)
    except Exception:
        return None


def _torch_runtime_details() -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "torch_version": _package_version("torch"),
        "cuda_available": None,
        "cuda_version": None,
        "device_count": None,
    }
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return details

    try:
        details["torch_version"] = str(getattr(torch, "__version__", details["torch_version"]))
        cuda = getattr(torch, "cuda", None)
        if cuda is not None:
            details["cuda_available"] = bool(cuda.is_available())
            details["device_count"] = int(cuda.device_count()) if details["cuda_available"] else 0
        torch_version = getattr(torch, "version", None)
        details["cuda_version"] = str(getattr(torch_version, "cuda", "")) or None
    except Exception:
        return details
    return details


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality_gate_settings(
    *,
    enable_quality_abort: bool,
    quality_check_after_chunks: Optional[int],
    total_tickets_expected: int,
    chunk_size: int,
    placeholder_answer_max_rate: float,
    empty_steps_max_rate: float,
    min_raw_response_saved_rate: float,
    min_generation_valid_rate: float,
    allow_slm_dominant_full_escalation: bool,
) -> Dict[str, Any]:
    if quality_check_after_chunks is not None:
        interval = max(int(quality_check_after_chunks), 1)
    else:
        short_run_threshold = max(int(chunk_size) * 5, 500)
        interval = 1 if int(total_tickets_expected) <= short_run_threshold else 5
    thresholds = build_thresholds(
        placeholder_answer_max_rate=placeholder_answer_max_rate,
        empty_steps_max_rate=empty_steps_max_rate,
        min_raw_response_saved_rate=min_raw_response_saved_rate,
        min_generation_valid_rate=min_generation_valid_rate,
    )
    return {
        "enabled": bool(enable_quality_abort),
        "check_after_chunks": interval,
        "allow_slm_dominant_full_escalation": bool(allow_slm_dominant_full_escalation),
        "thresholds": thresholds.as_dict(),
    }


def _build_runtime_manifest(
    *,
    config_identifier: str,
    backend_name: str,
    config: Mapping[str, str],
    chunk_size: int,
    command_line_args: Sequence[str],
    quality_gate_settings: Mapping[str, Any],
) -> Dict[str, Any]:
    execution_context = _execution_context()
    git_head = _git_value("rev-parse", "HEAD")
    git_branch = _git_value("rev-parse", "--abbrev-ref", "HEAD")
    git_dirty = bool(_git_value("status", "--porcelain"))
    nvidia_summary = _safe_subprocess_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    head_path = Path("RouterGym/classifiers/encoder_calibrated_head.npz")
    torch_details = _torch_runtime_details()
    env_names_present = [name for name in RELEVANT_ENV_VAR_NAMES if os.getenv(name)]

    return {
        "generated_at": _utc_now(),
        "config_identifier": config_identifier,
        "backend": backend_name,
        "chunk_size": int(chunk_size),
        "command_line_args": [str(item) for item in command_line_args],
        "git": {
            "commit_sha": git_head,
            "branch": git_branch,
            "dirty_worktree": git_dirty,
        },
        "benchmark_config": dict(config),
        "model_ids": {
            "slm1": SLM_MODELS["slm1"].hf_id,
            "slm2": SLM_MODELS["slm2"].hf_id,
            "llm1": LLM_MODELS["llm1"].hf_id,
            "llm2": LLM_MODELS["llm2"].hf_id,
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "runtime_packages": {
            "vllm_version": _package_version("vllm"),
            "torch_version": torch_details["torch_version"],
        },
        "cuda": {
            "available": torch_details["cuda_available"],
            "cuda_version": torch_details["cuda_version"],
            "device_count": torch_details["device_count"],
            "visible_gpu_ids": execution_context["gpu_id"],
            "nvidia_smi_summary": nvidia_summary,
        },
        "huggingface": {
            "hf_home": os.getenv("HF_HOME"),
            "huggingface_hub_cache": os.getenv("HUGGINGFACE_HUB_CACHE"),
        },
        "env": {
            "present_var_names": env_names_present,
            "worker_slot": execution_context["worker_slot"],
            "assigned_gpu_id": execution_context["gpu_id"],
        },
        "encoder_calibrated_head": {
            "path": str(head_path),
            "exists": head_path.exists(),
            "sha256": _sha256_file(head_path),
        },
        "quality_gate": dict(quality_gate_settings),
    }


def _write_runtime_manifest(config_dir: Path, payload: Mapping[str, Any]) -> Path:
    output_path = runtime_manifest_path(config_dir)
    _write_json(output_path, payload)
    return output_path


def _write_quality_gate_failure_reports(config_dir: Path, summary: Mapping[str, Any]) -> Dict[str, str]:
    json_path = quality_gate_report_json_path(config_dir)
    md_path = quality_gate_report_md_path(config_dir)
    _write_json(json_path, summary)

    config_summary = None
    configs = summary.get("configs", [])
    if isinstance(configs, list) and configs:
        config_summary = configs[0]
    failures = list(config_summary.get("failures", [])) if isinstance(config_summary, dict) else []
    lines = [
        "# Quality Gate Failure Report",
        "",
        f"- generated_at: {_utc_now()}",
        f"- input_path: {summary.get('input_path', '')}",
        f"- passes_quality_gate: {summary.get('passes_quality_gate', False)}",
        "",
        "## Thresholds",
        "",
        "```json",
        json.dumps(summary.get("thresholds", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Failures",
        "",
    ]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- No detailed failures recorded.")
    lines.extend(
        [
            "",
            "## Full Summary",
            "",
            "```json",
            json.dumps(summary, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "quality_gate_failure_report_json": str(json_path),
        "quality_gate_failure_report_md": str(md_path),
    }


def _status_context_suffix(status_payload: Mapping[str, Any]) -> str:
    worker_slot = str(status_payload.get("worker_slot", "") or "").strip()
    gpu_id = str(status_payload.get("gpu_id", "") or "").strip()
    parts: List[str] = []
    if worker_slot:
        parts.append(f"worker {worker_slot}")
    if gpu_id:
        parts.append(f"gpu {gpu_id}")
    return f" [{' | '.join(parts)}]" if parts else ""


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
    max_output_tokens: Optional[int],
    quality_gate_settings: Optional[Mapping[str, Any]] = None,
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
        "generation_settings": {
            "max_output_tokens": (int(max_output_tokens) if max_output_tokens is not None else None),
        },
        "quality_gate": dict(quality_gate_settings or {}),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "last_run_started_at": "",
        "last_run_completed_at": "",
        "first_execution_started_at": "",
        "run_status": "initialized",
        "output_files": {
            "config_dir": str(config_dir),
            "chunks_dir": str(config_dir / "chunks"),
            "merged_dir": str(config_dir / "merged"),
            "progress_log_path": str(config_progress_log_path(config_dir)),
            "status_path": str(config_status_path(config_dir)),
            "runtime_manifest_path": str(runtime_manifest_path(config_dir)),
            "quality_gate_failure_report_json": str(quality_gate_report_json_path(config_dir)),
            "quality_gate_failure_report_md": str(quality_gate_report_md_path(config_dir)),
            "backend_status_summary_path": str(backend_status_summary_path(output_root, backend_name)),
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
    max_output_tokens: Optional[int],
    quality_gate_settings: Optional[Mapping[str, Any]] = None,
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
            max_output_tokens=max_output_tokens,
            quality_gate_settings=quality_gate_settings,
        )
        _write_manifest(manifest_path, manifest)
        return manifest

    if int(existing.get("chunk_size", chunk_size)) != int(chunk_size):
        raise ValueError("Existing manifest chunk_size does not match requested chunk_size.")
    if int(existing.get("total_tickets_expected", total_tickets_expected)) != int(total_tickets_expected):
        raise ValueError("Existing manifest total_tickets_expected does not match requested ticket set.")
    if str(existing.get("backend_name", "")) != str(backend_details["backend_name"]):
        raise ValueError("Existing manifest backend_name does not match requested backend.")
    generation_settings = existing.setdefault("generation_settings", {})
    existing_max_output_tokens = generation_settings.get("max_output_tokens")
    requested_max_output_tokens = (
        int(max_output_tokens) if max_output_tokens is not None else None
    )
    if existing_max_output_tokens is None and "max_output_tokens" not in generation_settings:
        generation_settings["max_output_tokens"] = requested_max_output_tokens
    elif existing_max_output_tokens != requested_max_output_tokens:
        raise ValueError(
            "Existing manifest max_output_tokens does not match requested max_output_tokens."
        )
    output_files = existing.setdefault("output_files", {})
    config_dir = get_config_output_dir(output_root, str(backend_details["backend_name"]), config_identifier)
    output_files.setdefault("config_dir", str(config_dir))
    output_files.setdefault("chunks_dir", str(config_dir / "chunks"))
    output_files.setdefault("merged_dir", str(config_dir / "merged"))
    output_files.setdefault("progress_log_path", str(config_progress_log_path(config_dir)))
    output_files.setdefault("status_path", str(config_status_path(config_dir)))
    output_files.setdefault("runtime_manifest_path", str(runtime_manifest_path(config_dir)))
    output_files.setdefault("quality_gate_failure_report_json", str(quality_gate_report_json_path(config_dir)))
    output_files.setdefault("quality_gate_failure_report_md", str(quality_gate_report_md_path(config_dir)))
    output_files.setdefault(
        "backend_status_summary_path",
        str(backend_status_summary_path(output_root, str(backend_details["backend_name"]))),
    )
    existing.setdefault("first_execution_started_at", "")
    if quality_gate_settings:
        existing["quality_gate"] = dict(quality_gate_settings)
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
    if str(manifest.get("run_status", "")).strip() == "failed_quality_gate":
        manifest["updated_at"] = _utc_now()
        return
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


def _reset_manifest_for_fresh_run(manifest: Dict[str, Any]) -> None:
    manifest["completed_chunks"] = []
    manifest["failed_chunks"] = []
    manifest["run_status"] = "initialized"
    manifest["last_run_completed_at"] = ""
    manifest["updated_at"] = _utc_now()
    output_files = manifest.setdefault("output_files", {})
    output_files["merged_results_path"] = ""
    output_files["merged_failures_path"] = ""


def describe_resume_behavior(manifest: Mapping[str, Any], chunk_plan: Sequence[Mapping[str, int]]) -> str:
    """Return a short, human-readable summary of resume behavior."""

    config_dir = Path(str(manifest.get("output_files", {}).get("config_dir", "")))
    resume_state = summarize_resume_state(config_dir, manifest, chunk_plan)
    return (
        f"{resume_state['completed_chunks']} completed chunk(s) will be skipped; "
        f"{resume_state['pending_chunks']} chunk(s) pending; "
        f"{resume_state['failed_chunks']} failed chunk(s) eligible for retry"
    )


def _is_chunk_complete(config_dir: Path, manifest: Mapping[str, Any], chunk_spec: Mapping[str, int]) -> bool:
    completed = _completed_chunk_map(manifest)
    entry = completed.get(int(chunk_spec["chunk_index"]))
    if entry is None:
        return False
    results_path = Path(str(entry.get("results_path", chunk_results_path(config_dir, chunk_spec))))
    metadata_path = Path(str(entry.get("metadata_path", chunk_metadata_path(config_dir, chunk_spec))))
    return results_path.exists() and metadata_path.exists()


def summarize_resume_state(
    config_dir: Path,
    manifest: Mapping[str, Any],
    chunk_plan: Sequence[Mapping[str, int]],
) -> Dict[str, int]:
    """Return completed/pending/failed chunk counts for a manifest-backed run."""

    failed = _failed_chunk_map(manifest)
    completed_count = 0
    pending_count = 0
    failed_count = 0

    for chunk_spec in chunk_plan:
        chunk_index = int(chunk_spec["chunk_index"])
        if _is_chunk_complete(config_dir, manifest, chunk_spec):
            completed_count += 1
        elif chunk_index in failed:
            failed_count += 1
        else:
            pending_count += 1

    return {
        "completed_chunks": completed_count,
        "pending_chunks": pending_count,
        "failed_chunks": failed_count,
        "total_chunks": len(chunk_plan),
    }


def _completed_tickets_count(manifest: Mapping[str, Any], chunk_plan: Sequence[Mapping[str, int]], config_dir: Path) -> int:
    return sum(
        int(chunk_spec["ticket_count"])
        for chunk_spec in chunk_plan
        if _is_chunk_complete(config_dir, manifest, chunk_spec)
    )


def _last_completed_ticket_index(manifest: Mapping[str, Any], chunk_plan: Sequence[Mapping[str, int]], config_dir: Path) -> int:
    completed_ticket_indices = [
        int(chunk_spec["end_exclusive"]) - 1
        for chunk_spec in chunk_plan
        if _is_chunk_complete(config_dir, manifest, chunk_spec)
    ]
    return max(completed_ticket_indices, default=-1)


def build_config_status_payload(
    *,
    output_root: Path,
    backend_name: str,
    manifest: Mapping[str, Any],
    chunk_plan: Sequence[Mapping[str, int]],
    current_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a machine-readable per-config status payload."""

    config_dir = Path(str(manifest["output_files"]["config_dir"]))
    resume_state = summarize_resume_state(config_dir, manifest, chunk_plan)
    completed_tickets = _completed_tickets_count(manifest, chunk_plan, config_dir)
    total_tickets_expected = int(manifest.get("total_tickets_expected", 0))
    last_completed_ticket_index = _last_completed_ticket_index(manifest, chunk_plan, config_dir)

    run_started_at = (
        _parse_iso_datetime(str(manifest.get("first_execution_started_at", "")))
        or _parse_iso_datetime(str(manifest.get("last_run_started_at", "")))
        or _parse_iso_datetime(str(manifest.get("created_at", "")))
    )
    elapsed_seconds = (
        max((datetime.now(timezone.utc) - run_started_at).total_seconds(), 0.0)
        if run_started_at is not None
        else 0.0
    )
    eta_seconds: Optional[float]
    completed_chunks = int(resume_state["completed_chunks"])
    remaining_chunks = int(resume_state["pending_chunks"]) + int(resume_state["failed_chunks"])
    if completed_chunks > 0 and remaining_chunks > 0:
        eta_seconds = (elapsed_seconds / float(completed_chunks)) * float(remaining_chunks)
    else:
        eta_seconds = 0.0 if remaining_chunks == 0 else None

    execution_context = _execution_context()
    return {
        "config_identifier": str(manifest["config_identifier"]),
        "backend_name": backend_name,
        "manifest_path": str(get_manifest_path(output_root, backend_name, str(manifest["config_identifier"]))),
        "progress_log_path": str(config_progress_log_path(config_dir)),
        "status_path": str(config_status_path(config_dir)),
        "worker_slot": execution_context["worker_slot"],
        "gpu_id": execution_context["gpu_id"],
        "completed_chunks": completed_chunks,
        "pending_chunks": int(resume_state["pending_chunks"]),
        "failed_chunks": int(resume_state["failed_chunks"]),
        "total_chunks": int(resume_state["total_chunks"]),
        "completed_tickets": completed_tickets,
        "total_tickets_expected": total_tickets_expected,
        "last_completed_ticket_index": last_completed_ticket_index,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "elapsed_hms": _format_duration(elapsed_seconds),
        "eta_seconds": None if eta_seconds is None else round(eta_seconds, 2),
        "eta_hms": None if eta_seconds is None else _format_duration(eta_seconds),
        "current_status": str(current_status or manifest.get("run_status", "initialized")),
        "updated_at": _utc_now(),
    }


def write_backend_status_summary(output_root: Path, backend_name: str) -> Dict[str, Any]:
    """Write a compact top-level status summary across all manifests for one backend."""

    backend_root = output_root / backend_name
    config_statuses: List[Dict[str, Any]] = []
    for manifest_path in sorted(backend_root.glob("*/manifest.json")):
        manifest = _load_manifest(manifest_path)
        if manifest is None:
            continue
        config_dir = Path(str(manifest["output_files"]["config_dir"]))
        status_path = config_status_path(config_dir)
        status_payload = _load_json(status_path)
        if status_payload is None:
            chunk_size = int(manifest.get("chunk_size", PRODUCTION_CHUNK_SIZE))
            total_tickets = int(manifest.get("total_tickets_expected", 0))
            ticket_start = int(manifest.get("ticket_start", 0))
            chunk_plan = build_chunk_plan(
                total_tickets=total_tickets,
                chunk_size=chunk_size,
                start_index=ticket_start,
            )
            status_payload = build_config_status_payload(
                output_root=output_root,
                backend_name=backend_name,
                manifest=manifest,
                chunk_plan=chunk_plan,
            )
        config_statuses.append(dict(status_payload))

    summary = {
        "generated_at": _utc_now(),
        "backend_name": backend_name,
        "summary_path": str(backend_status_summary_path(output_root, backend_name)),
        "config_count": len(config_statuses),
        "configs": sorted(config_statuses, key=lambda item: str(item["config_identifier"])),
    }
    _write_json(backend_status_summary_path(output_root, backend_name), summary)
    return summary


def _append_progress_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _emit_progress_update(
    *,
    output_root: Path,
    backend_name: str,
    manifest: Mapping[str, Any],
    chunk_plan: Sequence[Mapping[str, int]],
    current_status: Optional[str],
    progress_line: Optional[str] = None,
) -> Dict[str, Any]:
    config_dir = Path(str(manifest["output_files"]["config_dir"]))
    status_payload = build_config_status_payload(
        output_root=output_root,
        backend_name=backend_name,
        manifest=manifest,
        chunk_plan=chunk_plan,
        current_status=current_status,
    )
    _write_json(config_status_path(config_dir), status_payload)
    try:
        write_backend_status_summary(output_root, backend_name)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    if progress_line:
        _append_progress_line(config_progress_log_path(config_dir), progress_line)
        print(progress_line, flush=True)
    return status_payload


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


def _maybe_evaluate_quality_gate(
    *,
    config_dir: Path,
    config_identifier: str,
    settings: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not bool(settings.get("enabled")):
        return None
    thresholds_raw = settings.get("thresholds", {})
    thresholds = build_thresholds(
        placeholder_answer_max_rate=float(thresholds_raw.get("placeholder_answer_max_rate", PLACEHOLDER_THRESHOLD)),
        empty_steps_max_rate=float(thresholds_raw.get("empty_steps_max_rate", EMPTY_STEPS_THRESHOLD)),
        min_raw_response_saved_rate=float(thresholds_raw.get("min_raw_response_saved_rate", RAW_RESPONSE_THRESHOLD)),
        min_generation_valid_rate=float(thresholds_raw.get("min_generation_valid_rate", GENERATION_VALID_THRESHOLD)),
    )
    summary = summarize_quality(
        config_dir,
        thresholds=thresholds,
        allow_full_slm_dominant_escalation=bool(settings.get("allow_slm_dominant_full_escalation", False)),
    )
    summary["checked_at"] = _utc_now()
    summary["config_identifier"] = config_identifier
    summary["quality_gate_settings"] = dict(settings)
    return summary


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
    max_output_tokens: Optional[int] = None,
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
                max_output_tokens=max_output_tokens,
            )
            if not bool(result.get("generation_valid", True)):
                error = {
                    "error_type": "GenerationInvalidError",
                    "message": str(
                        result.get("generation_invalid_reason")
                        or result.get("validation_error")
                        or result.get("parse_error")
                        or "generation_valid=false"
                    ),
                    "traceback": "",
                }
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
    max_output_tokens: Optional[int] = None,
    enable_quality_abort: bool = False,
    quality_check_after_chunks: Optional[int] = None,
    placeholder_answer_max_rate: float = PLACEHOLDER_THRESHOLD,
    empty_steps_max_rate: float = EMPTY_STEPS_THRESHOLD,
    min_raw_response_saved_rate: float = RAW_RESPONSE_THRESHOLD,
    min_generation_valid_rate: float = GENERATION_VALID_THRESHOLD,
    allow_slm_dominant_full_escalation: bool = False,
    command_line_args: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run one frozen config in deterministic chunks with manifest-based resume."""

    backend_details = resolve_backend_details(backend_name)
    backend_name_resolved = str(backend_details["backend_name"])
    config_identifier = build_config_identifier(config)

    ticket_df = load_ticket_dataset(limit=ticket_limit, start=ticket_start)
    total_tickets_expected = len(ticket_df)
    chunk_plan = build_chunk_plan(
        total_tickets=total_tickets_expected,
        chunk_size=chunk_size,
        start_index=ticket_start,
    )
    quality_gate_settings = _quality_gate_settings(
        enable_quality_abort=enable_quality_abort,
        quality_check_after_chunks=quality_check_after_chunks,
        total_tickets_expected=total_tickets_expected,
        chunk_size=chunk_size,
        placeholder_answer_max_rate=placeholder_answer_max_rate,
        empty_steps_max_rate=empty_steps_max_rate,
        min_raw_response_saved_rate=min_raw_response_saved_rate,
        min_generation_valid_rate=min_generation_valid_rate,
        allow_slm_dominant_full_escalation=allow_slm_dominant_full_escalation,
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
        max_output_tokens=max_output_tokens,
        quality_gate_settings=quality_gate_settings,
    )
    manifest_path = get_manifest_path(output_root, backend_name_resolved, config_identifier)
    config_dir = get_config_output_dir(output_root, backend_name_resolved, config_identifier)
    if not resume:
        _reset_manifest_for_fresh_run(manifest)
        manifest["quality_gate"] = dict(quality_gate_settings)
        _write_manifest(manifest_path, manifest)
    elif str(manifest.get("run_status", "")).strip() == "failed_quality_gate":
        raise ValueError(
            "Existing manifest failed the quality gate. Rerun explicitly with --no-resume after fixing the issue."
        )
    resume_state = summarize_resume_state(config_dir, manifest, chunk_plan)
    resume_summary = describe_resume_behavior(manifest, chunk_plan)

    if dry_run:
        dry_run_status = _emit_progress_update(
            output_root=output_root,
            backend_name=backend_name_resolved,
            manifest=manifest,
            chunk_plan=chunk_plan,
            current_status="dry_run",
        )
        return {
            "status": "dry_run",
            "config_identifier": config_identifier,
            "chunk_size": chunk_size,
            "max_output_tokens": max_output_tokens,
            "total_tickets_expected": total_tickets_expected,
            "manifest_path": str(manifest_path),
            "first_chunk_path": (
                str(chunk_results_path(config_dir, chunk_plan[0])) if chunk_plan else ""
            ),
            "resume_state": dict(resume_state),
            "status_path": str(config_status_path(config_dir)),
            "backend_status_summary_path": str(backend_status_summary_path(output_root, backend_name_resolved)),
            "progress_log_path": str(config_progress_log_path(config_dir)),
            "runtime_manifest_path": str(runtime_manifest_path(config_dir)),
            "status_payload": dict(dry_run_status),
            "resume_behavior_summary": resume_summary,
            "quality_gate": dict(quality_gate_settings),
        }

    with backend_override(backend_name):
        if not str(manifest.get("first_execution_started_at", "")).strip():
            manifest["first_execution_started_at"] = _utc_now()
        manifest["last_run_started_at"] = _utc_now()
        manifest["updated_at"] = _utc_now()
        manifest["run_status"] = "running"
        manifest["quality_gate"] = dict(quality_gate_settings)
        _write_manifest(manifest_path, manifest)
        runtime_manifest_payload = _build_runtime_manifest(
            config_identifier=config_identifier,
            backend_name=backend_name_resolved,
            config=config,
            chunk_size=chunk_size,
            command_line_args=command_line_args or [],
            quality_gate_settings=quality_gate_settings,
        )
        _write_runtime_manifest(config_dir, runtime_manifest_payload)
        startup_status = build_config_status_payload(
            output_root=output_root,
            backend_name=backend_name_resolved,
            manifest=manifest,
            chunk_plan=chunk_plan,
            current_status="running",
        )
        startup_status = _emit_progress_update(
            output_root=output_root,
            backend_name=backend_name_resolved,
            manifest=manifest,
            chunk_plan=chunk_plan,
            current_status="running",
            progress_line=(
                f"config {config_identifier}{_status_context_suffix(startup_status)}: startup | "
                f"completed {resume_state['completed_chunks']} chunk(s), "
                f"pending {resume_state['pending_chunks']}, failed {resume_state['failed_chunks']} | "
                f"elapsed {startup_status['elapsed_hms']} | "
                f"eta {startup_status['eta_hms'] or 'unknown'}"
            ),
        )
        try:
            for chunk_spec in chunk_plan:
                if resume and _is_chunk_complete(config_dir, manifest, chunk_spec):
                    continue
                try:
                    chunk_metadata = execute_chunk(
                        config_identifier=config_identifier,
                        config=config,
                        output_root=output_root,
                        backend_name=backend_name_resolved,
                        chunk_spec=chunk_spec,
                        max_output_tokens=max_output_tokens,
                    )
                    _set_completed_chunk(manifest, chunk_metadata)
                    current_status = "running"
                    message_suffix = "saved"
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
                    current_status = "partial"
                    message_suffix = f"failed ({type(exc).__name__})"
                _update_manifest_status(manifest, chunk_plan)
                _write_manifest(manifest_path, manifest)
                progress_status = _emit_progress_update(
                    output_root=output_root,
                    backend_name=backend_name_resolved,
                    manifest=manifest,
                    chunk_plan=chunk_plan,
                    current_status=current_status,
                )
                end_inclusive = int(chunk_spec["end_exclusive"]) - 1
                progress_line = (
                    f"config {config_identifier}{_status_context_suffix(progress_status)}: "
                    f"chunk {int(chunk_spec['chunk_index']) + 1}/{progress_status['total_chunks']} "
                    f"tickets {int(chunk_spec['start'])}-{end_inclusive} {message_suffix} | "
                    f"completed {progress_status['completed_tickets']}/{progress_status['total_tickets_expected']} tickets | "
                    f"chunks {progress_status['completed_chunks']}/{progress_status['total_chunks']} | "
                    f"elapsed {progress_status['elapsed_hms']} | "
                    f"eta {progress_status['eta_hms'] or 'unknown'}"
                )
                _append_progress_line(config_progress_log_path(config_dir), progress_line)
                print(progress_line, flush=True)
                should_check_quality = (
                    bool(quality_gate_settings["enabled"])
                    and message_suffix == "saved"
                    and int(progress_status["completed_chunks"]) > 0
                    and int(progress_status["completed_chunks"]) % int(quality_gate_settings["check_after_chunks"]) == 0
                )
                if should_check_quality:
                    quality_summary = _maybe_evaluate_quality_gate(
                        config_dir=config_dir,
                        config_identifier=config_identifier,
                        settings=quality_gate_settings,
                    )
                    if quality_summary is not None and not bool(quality_summary.get("passes_quality_gate", True)):
                        manifest["run_status"] = "failed_quality_gate"
                        manifest["last_run_completed_at"] = _utc_now()
                        manifest["updated_at"] = _utc_now()
                        manifest["quality_gate_failure"] = dict(quality_summary)
                        manifest["output_files"].update(_write_quality_gate_failure_reports(config_dir, quality_summary))
                        _write_manifest(manifest_path, manifest)
                        failure_status = _emit_progress_update(
                            output_root=output_root,
                            backend_name=backend_name_resolved,
                            manifest=manifest,
                            chunk_plan=chunk_plan,
                            current_status="failed_quality_gate",
                            progress_line=(
                                f"config {config_identifier}{_status_context_suffix(progress_status)}: "
                                f"failed_quality_gate | completed {progress_status['completed_chunks']}/"
                                f"{progress_status['total_chunks']} chunks | "
                                f"elapsed {progress_status['elapsed_hms']} | "
                                f"see {quality_gate_report_json_path(config_dir)}"
                            ),
                        )
                        return {
                            "status": "failed_quality_gate",
                            "config_identifier": config_identifier,
                            "manifest_path": str(manifest_path),
                            "merged_results_path": str(manifest["output_files"].get("merged_results_path", "")),
                            "merged_failures_path": str(manifest["output_files"].get("merged_failures_path", "")),
                            "progress_log_path": str(config_progress_log_path(config_dir)),
                            "status_path": str(config_status_path(config_dir)),
                            "runtime_manifest_path": str(runtime_manifest_path(config_dir)),
                            "backend_status_summary_path": str(backend_status_summary_path(output_root, backend_name_resolved)),
                            "startup_resume_state": dict(resume_state),
                            "final_resume_state": dict(summarize_resume_state(config_dir, manifest, chunk_plan)),
                            "final_status_payload": dict(failure_status),
                            "resume_behavior_summary": resume_summary,
                            "quality_gate": dict(quality_gate_settings),
                            "quality_gate_failure_report_json": str(quality_gate_report_json_path(config_dir)),
                            "quality_gate_failure_report_md": str(quality_gate_report_md_path(config_dir)),
                        }
        except BaseException:
            _update_manifest_status(manifest, chunk_plan)
            _write_manifest(manifest_path, manifest)
            _emit_progress_update(
                output_root=output_root,
                backend_name=backend_name_resolved,
                manifest=manifest,
                chunk_plan=chunk_plan,
                current_status="interrupted",
                progress_line=(
                    f"config {config_identifier}{_status_context_suffix(startup_status)}: interrupted | "
                    f"{describe_resume_behavior(manifest, chunk_plan)}"
                ),
            )
            raise

    merged = merge_completed_chunks(manifest_path)
    manifest = _load_manifest(manifest_path) or manifest
    manifest["last_run_completed_at"] = _utc_now()
    manifest["output_files"]["merged_results_path"] = merged["merged_results_path"]
    manifest["output_files"]["merged_failures_path"] = merged["merged_failures_path"]
    _update_manifest_status(manifest, chunk_plan)
    _write_manifest(manifest_path, manifest)
    final_resume_state = summarize_resume_state(config_dir, manifest, chunk_plan)
    final_status_payload = build_config_status_payload(
        output_root=output_root,
        backend_name=backend_name_resolved,
        manifest=manifest,
        chunk_plan=chunk_plan,
        current_status=str(manifest["run_status"]),
    )
    final_status = _emit_progress_update(
        output_root=output_root,
        backend_name=backend_name_resolved,
        manifest=manifest,
        chunk_plan=chunk_plan,
        current_status=str(manifest["run_status"]),
        progress_line=(
            f"config {config_identifier}{_status_context_suffix(final_status_payload)}: "
            f"{str(manifest['run_status'])} | "
            f"completed {final_resume_state['completed_chunks']}/{final_resume_state['total_chunks']} chunks | "
            f"elapsed {final_status_payload['elapsed_hms']}"
        ),
    )
    return {
        "status": manifest["run_status"],
        "config_identifier": config_identifier,
        "manifest_path": str(manifest_path),
        "merged_results_path": merged["merged_results_path"],
        "merged_failures_path": merged["merged_failures_path"],
        "progress_log_path": str(config_progress_log_path(config_dir)),
        "status_path": str(config_status_path(config_dir)),
        "runtime_manifest_path": str(runtime_manifest_path(config_dir)),
        "backend_status_summary_path": str(backend_status_summary_path(output_root, backend_name_resolved)),
        "startup_resume_state": dict(resume_state),
        "final_resume_state": dict(final_resume_state),
        "final_status_payload": dict(final_status),
        "resume_behavior_summary": resume_summary,
        "quality_gate": dict(quality_gate_settings),
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
    max_output_tokens: Optional[int] = None,
    enable_quality_abort: bool = False,
    quality_check_after_chunks: Optional[int] = None,
    placeholder_answer_max_rate: float = PLACEHOLDER_THRESHOLD,
    empty_steps_max_rate: float = EMPTY_STEPS_THRESHOLD,
    min_raw_response_saved_rate: float = RAW_RESPONSE_THRESHOLD,
    min_generation_valid_rate: float = GENERATION_VALID_THRESHOLD,
    allow_slm_dominant_full_escalation: bool = False,
    command_line_args: Optional[Sequence[str]] = None,
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
            max_output_tokens=max_output_tokens,
            enable_quality_abort=enable_quality_abort,
            quality_check_after_chunks=quality_check_after_chunks,
            placeholder_answer_max_rate=placeholder_answer_max_rate,
            empty_steps_max_rate=empty_steps_max_rate,
            min_raw_response_saved_rate=min_raw_response_saved_rate,
            min_generation_valid_rate=min_generation_valid_rate,
            allow_slm_dominant_full_escalation=allow_slm_dominant_full_escalation,
            command_line_args=command_line_args,
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
    "runtime_manifest_path",
    "merge_completed_chunks",
    "resolve_backend_details",
    "resolve_selected_configs",
    "run_benchmark_matrix_chunked",
    "run_config_chunked",
    "summarize_resume_state",
]
