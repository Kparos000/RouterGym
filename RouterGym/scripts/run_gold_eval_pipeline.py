"""Run the gold eval pipeline: build template, autofill, audit, and freeze.

PowerShell example:
  python -m RouterGym.scripts.run_gold_eval_pipeline --start 0 --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from RouterGym.data.policy_kb import kb_loader
from RouterGym.label_space import CANONICAL_LABELS

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "gold_eval"
KB_INDEX_PATH = Path(__file__).resolve().parents[1] / "data" / "policy_kb" / "policy_kb_index.json"


def _run_module_main(main_func, argv: Sequence[str]) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = list(argv)
        main_func()
    finally:
        sys.argv = old_argv


def run_template_builder(start: int, limit: int, output_dir: Path) -> None:
    from RouterGym.scripts import build_gold_eval_template as builder

    output_path = output_dir / "gold_eval.jsonl"
    argv = [
        "build_gold_eval_template",
        "--start",
        str(start),
        "--limit",
        str(limit),
        "--output-path",
        str(output_path),
    ]
    _run_module_main(builder.main, argv)


def run_autofill(start: int, limit: int, output_dir: Path) -> None:
    from RouterGym.scripts import autofill_gold_eval as autofill

    input_path = output_dir / "gold_eval.jsonl"
    output_path = output_dir / "gold_eval_auto.jsonl"
    review_path = output_dir / "gold_eval_review_queue.jsonl"
    argv = [
        "autofill_gold_eval",
        "--start",
        str(start),
        "--limit",
        str(limit),
        "--input-path",
        str(input_path),
        "--output-path",
        str(output_path),
        "--review-queue-path",
        str(review_path),
    ]
    _run_module_main(autofill.main, argv)


def _read_jsonl(path: Path, required: bool = True) -> List[Dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def _load_kb_index() -> List[Dict[str, Any]]:
    try:
        return [dict(entry) for entry in kb_loader.load_kb_index()]
    except Exception:
        return []


def _kb_index_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _count_list(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0


def _short_summary(summary: Any) -> bool:
    text = str(summary or "").strip()
    return not text or len(text) < 8


def compute_audit(
    template_records: Sequence[Dict[str, Any]],
    auto_records: Sequence[Dict[str, Any]],
    review_records: Sequence[Dict[str, Any]],
    kb_index: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    total_records = len(auto_records)
    per_label_counts = Counter(str(r.get("topic_group", "")) for r in auto_records)

    review_reasons: Counter[str] = Counter()
    for rec in review_records:
        for reason in rec.get("review_reasons", []) or []:
            review_reasons[str(reason)] += 1

    policy_counts: Counter[str] = Counter()
    total_policy_refs = 0
    for rec in auto_records:
        policies = rec.get("gold_resolution", {}).get("kb_policies", []) or []
        if isinstance(policies, list):
            total_policy_refs += len(policies)
            for pid in policies:
                policy_counts[str(pid)] += 1

    unique_policy_ids_used = len(policy_counts)
    kb_index_size = len(kb_index)
    unique_policy_ratio = unique_policy_ids_used / kb_index_size if kb_index_size else 0.0
    top_policies = policy_counts.most_common(10)
    top_policy_share = (top_policies[0][1] / max(total_policy_refs, 1)) if top_policies else 0.0

    steps_counts = []
    acceptance_counts = []
    escalation_count = 0
    short_summary_count = 0
    for rec in auto_records:
        gold = rec.get("gold_resolution", {}) or {}
        steps_counts.append(_count_list(gold.get("steps")))
        acceptance_counts.append(_count_list(gold.get("acceptance_criteria")))
        if bool(gold.get("escalation_required", False)):
            escalation_count += 1
        if _short_summary(gold.get("summary")):
            short_summary_count += 1

    avg_steps = sum(steps_counts) / total_records if total_records else 0.0
    avg_acceptance = sum(acceptance_counts) / total_records if total_records else 0.0
    escalation_pct = (escalation_count / total_records * 100.0) if total_records else 0.0
    short_summary_pct = (short_summary_count / total_records * 100.0) if total_records else 0.0

    return {
        "template_records": len(template_records),
        "total_records": total_records,
        "per_label_counts": dict(per_label_counts),
        "review_queue_count": len(review_records),
        "review_reasons": dict(review_reasons),
        "kb_index_size": kb_index_size,
        "policy_usage": {
            "total_policy_refs": total_policy_refs,
            "unique_policy_ids_used": unique_policy_ids_used,
            "unique_policy_ratio": unique_policy_ratio,
            "top_policies": [{"policy_id": pid, "count": count} for pid, count in top_policies],
            "top_policy_share": top_policy_share,
        },
        "content_quality": {
            "avg_steps": avg_steps,
            "avg_acceptance_criteria": avg_acceptance,
            "percent_escalation_required": escalation_pct,
            "percent_short_summary": short_summary_pct,
        },
    }


def _format_audit_text(audit: Dict[str, Any]) -> str:
    lines = []
    lines.append("Gold Eval Audit Summary")
    lines.append("=" * 24)
    lines.append(f"Template records: {audit['template_records']}")
    lines.append(f"Total records: {audit['total_records']}")
    lines.append(f"Review queue count: {audit['review_queue_count']}")
    lines.append("Per-label counts:")
    for label, count in sorted(audit["per_label_counts"].items()):
        lines.append(f"  {label}: {count}")
    lines.append("Review reasons:")
    for reason, count in sorted(audit["review_reasons"].items()):
        lines.append(f"  {reason}: {count}")
    policy = audit["policy_usage"]
    lines.append("Policy usage:")
    lines.append(f"  Total policy refs: {policy['total_policy_refs']}")
    lines.append(f"  Unique policy ids: {policy['unique_policy_ids_used']}")
    lines.append(f"  Unique policy ratio: {policy['unique_policy_ratio']:.3f}")
    lines.append(f"  Top policy share: {policy['top_policy_share']:.3f}")
    lines.append("  Top policies:")
    for entry in policy["top_policies"]:
        lines.append(f"    {entry['policy_id']}: {entry['count']}")
    quality = audit["content_quality"]
    lines.append("Content quality:")
    lines.append(f"  Avg steps: {quality['avg_steps']:.2f}")
    lines.append(f"  Avg acceptance criteria: {quality['avg_acceptance_criteria']:.2f}")
    lines.append(f"  % escalation_required: {quality['percent_escalation_required']:.2f}")
    lines.append(f"  % short summary: {quality['percent_short_summary']:.2f}")
    return "\n".join(lines)


def _format_samples(
    auto_records: Sequence[Dict[str, Any]],
    sample_per_label: int,
) -> str:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    sorted_records = sorted(auto_records, key=lambda r: int(r.get("ticket_index", 0)))
    for rec in sorted_records:
        label = str(rec.get("topic_group", ""))
        if len(grouped[label]) < sample_per_label:
            grouped[label].append(rec)

    lines = []
    for label in CANONICAL_LABELS:
        samples = grouped.get(label, [])
        lines.append(f"{label} ({len(samples)} samples)")
        lines.append("-" * (len(label) + 12))
        for rec in samples:
            gold = rec.get("gold_resolution", {}) or {}
            lines.append(f"ticket_index: {rec.get('ticket_index')}")
            lines.append(f"summary: {gold.get('summary', '')}")
            lines.append(f"steps: {gold.get('steps', [])}")
            lines.append(f"acceptance_criteria: {gold.get('acceptance_criteria', [])}")
            lines.append(f"kb_policies: {gold.get('kb_policies', [])}")
            lines.append("")
        if not samples:
            lines.append("(no samples)")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_thresholds(audit: Dict[str, Any], args: argparse.Namespace) -> int:
    if args.fail_on_review_queue and audit["review_queue_count"] > args.max_review_queue:
        return 2
    if audit["policy_usage"]["unique_policy_ratio"] < args.min_unique_policy_ratio:
        return 3
    if audit["policy_usage"]["top_policy_share"] > args.max_top_policy_share:
        return 4
    if audit["content_quality"]["avg_steps"] < args.min_avg_steps:
        return 5
    if audit["content_quality"]["avg_acceptance_criteria"] < args.min_avg_acceptance_criteria:
        return 6
    return 0


def _freeze_final(
    output_dir: Path,
    final_name: str,
    start: int,
    limit: int,
    kb_index_size: int,
    kb_index_sha: str,
    thresholds: Dict[str, Any],
) -> None:
    final_path = output_dir / final_name
    source_path = output_dir / "gold_eval_auto.jsonl"
    shutil.copyfile(source_path, final_path)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "limit": limit,
        "kb_index_size": kb_index_size,
        "kb_index_sha256": kb_index_sha,
        "thresholds": thresholds,
        "git_commit_hash": _get_git_commit_hash(),
    }
    meta_path = output_dir / f"{final_name}.meta.json"
    _write_json(meta_path, meta)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run gold eval pipeline with audit and thresholds.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--final-name", type=str, default="gold_eval_final.jsonl")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-review-queue", type=int, default=0)
    parser.add_argument("--fail-on-review-queue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_unique_policy_ratio", type=float, default=0.30)
    parser.add_argument("--max_top_policy_share", type=float, default=0.20)
    parser.add_argument("--min_avg_steps", type=float, default=4.0)
    parser.add_argument("--min_avg_acceptance_criteria", type=float, default=2.0)
    parser.add_argument("--sample-per-label", type=int, default=2)
    return parser


def run_pipeline(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.audit_only:
        run_template_builder(args.start, args.limit, output_dir)
        run_autofill(args.start, args.limit, output_dir)

    template_path = output_dir / "gold_eval.jsonl"
    auto_path = output_dir / "gold_eval_auto.jsonl"
    review_path = output_dir / "gold_eval_review_queue.jsonl"
    template_records = _read_jsonl(template_path, required=True)
    auto_records = _read_jsonl(auto_path, required=True)
    review_records = _read_jsonl(review_path, required=False)
    kb_index = _load_kb_index()

    audit = compute_audit(template_records, auto_records, review_records, kb_index)
    audit_path = output_dir / "gold_eval_audit.json"
    audit_txt_path = output_dir / "gold_eval_audit.txt"
    samples_path = output_dir / "gold_eval_samples.txt"
    _write_json(audit_path, audit)
    _write_text(audit_txt_path, _format_audit_text(audit))
    _write_text(samples_path, _format_samples(auto_records, args.sample_per_label))

    exit_code = _check_thresholds(audit, args)
    if exit_code != 0:
        return exit_code

    if not args.audit_only:
        thresholds = {
            "max_review_queue": args.max_review_queue,
            "min_unique_policy_ratio": args.min_unique_policy_ratio,
            "max_top_policy_share": args.max_top_policy_share,
            "min_avg_steps": args.min_avg_steps,
            "min_avg_acceptance_criteria": args.min_avg_acceptance_criteria,
            "fail_on_review_queue": args.fail_on_review_queue,
        }
        _freeze_final(
            output_dir=output_dir,
            final_name=args.final_name,
            start=args.start,
            limit=args.limit,
            kb_index_size=audit["kb_index_size"],
            kb_index_sha=_kb_index_sha256(KB_INDEX_PATH),
            thresholds=thresholds,
        )
    return 0


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    code = run_pipeline(args)
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
