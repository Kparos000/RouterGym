"""Run the frozen benchmark matrix in resumable chunks.

Examples:
    python -m RouterGym.scripts.run_chunked_benchmark --preflight-size 100 --config-id slm_only__base_slm1__mem_none --dry-run
    python -m RouterGym.scripts.run_chunked_benchmark --backend openai_compatible --chunk-size 100
    python -m RouterGym.scripts.run_chunked_benchmark --config-ids slm_only__base_slm1__mem_none slm_only__base_slm2__mem_none --parallel-workers 2 --gpu-ids 0,1
    python -m RouterGym.scripts.run_chunked_benchmark --config-id slm_dominant__base_slm1__esc_llm1__mem_rag_bm25 --merge-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from RouterGym.benchmark_spec import PRODUCTION_CHUNK_SIZE
from RouterGym.experiments.chunked_execution import (
    DEFAULT_OUTPUT_ROOT,
    backend_status_summary_path,
    build_parallel_config_plan,
    config_progress_log_path,
    config_status_path,
    get_manifest_path,
    merge_completed_chunks,
    resolve_backend_details,
    resolve_selected_configs,
    run_benchmark_matrix_chunked,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunked/resumable production benchmark runner.")
    parser.add_argument("--config-id", type=str, default=None, help="Run only one frozen config identifier.")
    parser.add_argument(
        "--config-ids",
        nargs="+",
        default=None,
        help="Optional ordered list of frozen config identifiers.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for manifests and chunk outputs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=PRODUCTION_CHUNK_SIZE,
        help=f"Tickets per chunk (default {PRODUCTION_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional generation cap override for benchmark responses.",
    )
    parser.add_argument("--start", type=int, default=0, help="0-based dataset start index.")
    parser.add_argument("--limit", type=int, default=None, help="Optional ticket limit.")
    parser.add_argument(
        "--preflight-size",
        type=int,
        choices=[100, 500, 2000],
        default=None,
        help="Optional preflight size; overrides --limit.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf_inference", "openai_compatible", "vllm_local"],
        default=None,
        help="Optional backend override for this run.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Config-level worker count for concurrent execution (default 1).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Optional comma-separated GPU ids; one visible GPU per worker slot.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge completed chunk outputs without running new work.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create/inspect manifests and chunk plan without executing tickets.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed chunk entries and rerun all chunks.",
    )
    return parser


def _parse_gpu_ids(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []
    return [chunk.strip() for chunk in str(raw_value).split(",") if chunk.strip()]


def _resolve_parallel_workers(requested_workers: int, gpu_ids: Sequence[str]) -> int:
    if requested_workers <= 0:
        raise ValueError("parallel_workers must be > 0")
    if gpu_ids and requested_workers == 1:
        return len(gpu_ids)
    return requested_workers


def _merge_selection(
    *,
    output_root: Path,
    config_id: Optional[str],
    config_ids: Optional[Sequence[str]],
    backend_name: str,
) -> List[dict]:
    selected_ids = [selected_id for selected_id, _ in resolve_selected_configs(config_id=config_id, config_ids=config_ids)]
    merged_results = []
    for selected_id in selected_ids:
        manifest_path = get_manifest_path(output_root, backend_name, selected_id)
        merged = merge_completed_chunks(manifest_path)
        merged_results.append(
            {
                "config_identifier": selected_id,
                "manifest_path": str(manifest_path),
                **merged,
            }
        )
    return merged_results


def _single_config_command(
    *,
    selected_id: str,
    output_root: Path,
    chunk_size: int,
    ticket_start: int,
    ticket_limit: Optional[int],
    backend_name: Optional[str],
    resume: bool,
    max_output_tokens: Optional[int],
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "RouterGym.scripts.run_chunked_benchmark",
        "--config-id",
        selected_id,
        "--output-root",
        str(output_root),
        "--chunk-size",
        str(chunk_size),
        "--start",
        str(ticket_start),
    ]
    if ticket_limit is not None:
        command.extend(["--limit", str(ticket_limit)])
    if max_output_tokens is not None:
        command.extend(["--max-output-tokens", str(max_output_tokens)])
    if backend_name:
        command.extend(["--backend", backend_name])
    if not resume:
        command.append("--no-resume")
    return command


def _run_parallel_selection(
    *,
    output_root: Path,
    config_id: Optional[str],
    config_ids: Optional[Sequence[str]],
    chunk_size: int,
    ticket_start: int,
    ticket_limit: Optional[int],
    backend_name: Optional[str],
    resume: bool,
    dry_run: bool,
    parallel_workers: int,
    gpu_ids: Sequence[str],
    max_output_tokens: Optional[int],
) -> List[Dict[str, Any]]:
    resolved_backend = str(resolve_backend_details(backend_name)["backend_name"])
    effective_workers = _resolve_parallel_workers(parallel_workers, gpu_ids)
    launch_plan = build_parallel_config_plan(
        output_root=output_root,
        backend_name=resolved_backend,
        config_id=config_id,
        config_ids=config_ids,
        parallel_workers=effective_workers,
        gpu_ids=gpu_ids,
        ticket_start=ticket_start,
        ticket_limit=ticket_limit,
        chunk_size=chunk_size,
    )

    if dry_run:
        return [
            {
                "status": "parallel_dry_run",
                "backend_name": resolved_backend,
                "parallel_workers": effective_workers,
                "gpu_ids": list(gpu_ids),
                "launch_plan": launch_plan,
            }
        ]

    worker_limit = max((int(entry["worker_slot"]) for entry in launch_plan), default=-1) + 1
    pending = list(launch_plan)
    active: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    while pending or active:
        active_slots = {int(item["entry"]["worker_slot"]) for item in active}
        free_slots = [slot for slot in range(worker_limit) if slot not in active_slots]
        for free_slot in free_slots:
            match_index = next(
                (idx for idx, candidate in enumerate(pending) if int(candidate["worker_slot"]) == free_slot),
                None,
            )
            if match_index is None:
                continue

            entry = dict(pending.pop(match_index))
            config_dir = Path(str(entry["output_dir"]))
            config_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env["ROUTERGYM_WORKER_SLOT"] = str(entry["worker_slot"])
            if entry.get("gpu_id") is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(entry["gpu_id"])
                env["ROUTERGYM_ASSIGNED_GPU_ID"] = str(entry["gpu_id"])

            progress_log = config_progress_log_path(config_dir)
            status_file = config_status_path(config_dir)
            gpu_suffix = f" gpu {entry['gpu_id']}" if entry.get("gpu_id") is not None else ""
            print(
                (
                    f"launching config {entry['config_identifier']} "
                    f"[worker {entry['worker_slot']}{gpu_suffix}]"
                ),
                flush=True,
            )
            process = subprocess.Popen(
                _single_config_command(
                    selected_id=str(entry["config_identifier"]),
                    output_root=output_root,
                    chunk_size=chunk_size,
                    ticket_start=ticket_start,
                    ticket_limit=ticket_limit,
                    backend_name=backend_name,
                    resume=resume,
                    max_output_tokens=max_output_tokens,
                ),
                cwd=str(Path.cwd()),
                env=env,
            )
            active.append(
                {
                    "entry": entry,
                    "process": process,
                    "progress_log_path": progress_log,
                    "status_path": status_file,
                }
            )

        time.sleep(0.25)
        next_active: List[Dict[str, Any]] = []
        for item in active:
            process = item["process"]
            if process.poll() is None:
                next_active.append(item)
                continue

            entry = dict(item["entry"])
            results.append(
                {
                    "config_identifier": entry["config_identifier"],
                    "backend_name": resolved_backend,
                    "worker_slot": entry["worker_slot"],
                    "gpu_id": entry.get("gpu_id"),
                    "manifest_path": entry["manifest_path"],
                    "progress_log_path": str(item["progress_log_path"]),
                    "status_path": str(item["status_path"]),
                    "backend_status_summary_path": str(backend_status_summary_path(output_root, resolved_backend)),
                    "exit_code": int(process.returncode or 0),
                    "status": "completed" if int(process.returncode or 0) == 0 else "worker_failed",
                }
            )
        active = next_active

    return sorted(results, key=lambda item: str(item["config_identifier"]))


def main() -> None:
    args = build_parser().parse_args()
    effective_limit = args.preflight_size if args.preflight_size is not None else args.limit
    backend_name = str(resolve_backend_details(args.backend)["backend_name"])
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    should_parallelize = bool(gpu_ids) or args.parallel_workers > 1

    if args.merge_only:
        payload = _merge_selection(
            output_root=args.output_root,
            config_id=args.config_id,
            config_ids=args.config_ids,
            backend_name=backend_name,
        )
    elif should_parallelize:
        payload = _run_parallel_selection(
            output_root=args.output_root,
            config_id=args.config_id,
            config_ids=args.config_ids,
            chunk_size=args.chunk_size,
            ticket_start=args.start,
            ticket_limit=effective_limit,
            backend_name=args.backend,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            parallel_workers=args.parallel_workers,
            gpu_ids=gpu_ids,
            max_output_tokens=args.max_output_tokens,
        )
    else:
        payload = run_benchmark_matrix_chunked(
            output_root=args.output_root,
            config_id=args.config_id,
            config_ids=args.config_ids,
            chunk_size=args.chunk_size,
            ticket_start=args.start,
            ticket_limit=effective_limit,
            backend_name=args.backend,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            max_output_tokens=args.max_output_tokens,
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
