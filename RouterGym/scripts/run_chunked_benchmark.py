"""Run the frozen benchmark matrix in resumable chunks.

Examples:
    python -m RouterGym.scripts.run_chunked_benchmark --preflight-size 100 --config-id slm_only__base_slm1__mem_none --dry-run
    python -m RouterGym.scripts.run_chunked_benchmark --backend openai_compatible --chunk-size 5000
    python -m RouterGym.scripts.run_chunked_benchmark --config-id slm_dominant__base_slm1__esc_llm1__mem_rag_bm25 --merge-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from RouterGym.benchmark_spec import PRODUCTION_CHUNK_SIZE
from RouterGym.experiments.chunked_execution import (
    DEFAULT_OUTPUT_ROOT,
    get_frozen_config_map,
    get_manifest_path,
    merge_completed_chunks,
    resolve_backend_details,
    run_benchmark_matrix_chunked,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunked/resumable production benchmark runner.")
    parser.add_argument("--config-id", type=str, default=None, help="Run only one frozen config identifier.")
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


def _merge_selection(
    *,
    output_root: Path,
    config_id: Optional[str],
    backend_name: str,
) -> List[dict]:
    config_map = get_frozen_config_map()
    selected_ids = [config_id] if config_id else sorted(config_map)
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


def main() -> None:
    args = build_parser().parse_args()
    effective_limit = args.preflight_size if args.preflight_size is not None else args.limit
    backend_name = str(resolve_backend_details(args.backend)["backend_name"])

    if args.merge_only:
        payload = _merge_selection(
            output_root=args.output_root,
            config_id=args.config_id,
            backend_name=backend_name,
        )
    else:
        payload = run_benchmark_matrix_chunked(
            output_root=args.output_root,
            config_id=args.config_id,
            chunk_size=args.chunk_size,
            ticket_start=args.start,
            ticket_limit=effective_limit,
            backend_name=args.backend,
            resume=not args.no_resume,
            dry_run=args.dry_run,
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
