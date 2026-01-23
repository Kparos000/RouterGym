"""Tests for the gold eval pipeline runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from RouterGym.scripts import run_gold_eval_pipeline as pipeline


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    lines = [json.dumps(rec, ensure_ascii=False) for rec in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_auto_record(
    ticket_index: int,
    label: str,
    policies: List[str],
    steps_count: int = 4,
    acceptance_count: int = 2,
    summary: str = "Summary text",
) -> Dict[str, Any]:
    return {
        "ticket_index": ticket_index,
        "topic_group": label,
        "ticket_text": "Sample ticket text",
        "gold_resolution": {
            "summary": summary,
            "steps": [f"step {i}" for i in range(steps_count)],
            "escalation_required": False,
            "escalation_reason": "",
            "kb_policies": policies,
            "acceptance_criteria": [f"criterion {i}" for i in range(acceptance_count)],
        },
        "needs_human_review": False,
        "review_reasons": [],
    }


def _make_review_record() -> Dict[str, Any]:
    rec = _make_auto_record(99, "Access", ["kb-0"])
    rec["needs_human_review"] = True
    rec["review_reasons"] = ["model_invalid_output"]
    return rec


def _mock_kb_index(size: int) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"kb-{i}",
            "category": "Access",
            "title": "T",
            "content": "C",
            "escalation_notes": "",
            "path": f"p{i}",
            "tags": [],
        }
        for i in range(size)
    ]


def _make_args(output_dir: Path, extra: List[str]) -> argparse.Namespace:
    parser = pipeline.build_arg_parser()
    return parser.parse_args(["--output-dir", str(output_dir), *extra])


def test_pipeline_writes_audit_files(tmp_path: Path, monkeypatch: Any) -> None:
    out_dir = tmp_path / "gold_eval"
    out_dir.mkdir()
    _write_jsonl(out_dir / "gold_eval.jsonl", [{"ticket_index": 0}])
    _write_jsonl(
        out_dir / "gold_eval_auto.jsonl",
        [_make_auto_record(0, "Access", ["kb-0"]), _make_auto_record(1, "Hardware", ["kb-1"])],
    )
    _write_jsonl(out_dir / "gold_eval_review_queue.jsonl", [])

    monkeypatch.setattr(pipeline, "_load_kb_index", lambda: _mock_kb_index(10))
    monkeypatch.setattr(pipeline, "_kb_index_sha256", lambda _: "hash")

    args = _make_args(
        out_dir,
        ["--audit-only", "--min_unique_policy_ratio", "0.1", "--max_top_policy_share", "1.0"],
    )
    exit_code = pipeline.run_pipeline(args)
    assert exit_code == 0
    assert (out_dir / "gold_eval_audit.json").exists()
    assert (out_dir / "gold_eval_audit.txt").exists()
    assert (out_dir / "gold_eval_samples.txt").exists()


def test_pipeline_review_queue_threshold(tmp_path: Path, monkeypatch: Any) -> None:
    out_dir = tmp_path / "gold_eval"
    out_dir.mkdir()
    _write_jsonl(out_dir / "gold_eval.jsonl", [{"ticket_index": 0}])
    _write_jsonl(out_dir / "gold_eval_auto.jsonl", [_make_auto_record(0, "Access", ["kb-0"])])
    _write_jsonl(out_dir / "gold_eval_review_queue.jsonl", [_make_review_record()])

    monkeypatch.setattr(pipeline, "_load_kb_index", lambda: _mock_kb_index(10))
    args = _make_args(out_dir, ["--audit-only", "--max-review-queue", "0"])
    exit_code = pipeline.run_pipeline(args)
    assert exit_code == 2


def test_pipeline_unique_policy_ratio_threshold(tmp_path: Path, monkeypatch: Any) -> None:
    out_dir = tmp_path / "gold_eval"
    out_dir.mkdir()
    _write_jsonl(out_dir / "gold_eval.jsonl", [{"ticket_index": 0}])
    _write_jsonl(out_dir / "gold_eval_auto.jsonl", [_make_auto_record(0, "Access", ["kb-0"])])
    _write_jsonl(out_dir / "gold_eval_review_queue.jsonl", [])

    monkeypatch.setattr(pipeline, "_load_kb_index", lambda: _mock_kb_index(10))
    args = _make_args(out_dir, ["--audit-only", "--min_unique_policy_ratio", "0.5"])
    exit_code = pipeline.run_pipeline(args)
    assert exit_code == 3


def test_pipeline_top_policy_share_threshold(tmp_path: Path, monkeypatch: Any) -> None:
    out_dir = tmp_path / "gold_eval"
    out_dir.mkdir()
    _write_jsonl(out_dir / "gold_eval.jsonl", [{"ticket_index": 0}])
    records = [
        _make_auto_record(0, "Access", ["kb-0"]),
        _make_auto_record(1, "Access", ["kb-0"]),
        _make_auto_record(2, "Access", ["kb-0"]),
    ]
    _write_jsonl(out_dir / "gold_eval_auto.jsonl", records)
    _write_jsonl(out_dir / "gold_eval_review_queue.jsonl", [])

    monkeypatch.setattr(pipeline, "_load_kb_index", lambda: _mock_kb_index(10))
    args = _make_args(
        out_dir,
        ["--audit-only", "--max_top_policy_share", "0.2", "--min_unique_policy_ratio", "0.0"],
    )
    exit_code = pipeline.run_pipeline(args)
    assert exit_code == 4


def test_pipeline_avg_steps_threshold(tmp_path: Path, monkeypatch: Any) -> None:
    out_dir = tmp_path / "gold_eval"
    out_dir.mkdir()
    _write_jsonl(out_dir / "gold_eval.jsonl", [{"ticket_index": 0}])
    _write_jsonl(out_dir / "gold_eval_auto.jsonl", [_make_auto_record(0, "Access", ["kb-0"], steps_count=2)])
    _write_jsonl(out_dir / "gold_eval_review_queue.jsonl", [])

    monkeypatch.setattr(pipeline, "_load_kb_index", lambda: _mock_kb_index(10))
    args = _make_args(
        out_dir,
        ["--audit-only", "--min_avg_steps", "4", "--min_unique_policy_ratio", "0.0", "--max_top_policy_share", "1.0"],
    )
    exit_code = pipeline.run_pipeline(args)
    assert exit_code == 5


def test_pipeline_avg_acceptance_threshold(tmp_path: Path, monkeypatch: Any) -> None:
    out_dir = tmp_path / "gold_eval"
    out_dir.mkdir()
    _write_jsonl(out_dir / "gold_eval.jsonl", [{"ticket_index": 0}])
    _write_jsonl(
        out_dir / "gold_eval_auto.jsonl",
        [_make_auto_record(0, "Access", ["kb-0"], steps_count=4, acceptance_count=1)],
    )
    _write_jsonl(out_dir / "gold_eval_review_queue.jsonl", [])

    monkeypatch.setattr(pipeline, "_load_kb_index", lambda: _mock_kb_index(10))
    args = _make_args(
        out_dir,
        [
            "--audit-only",
            "--min_avg_acceptance_criteria",
            "2",
            "--min_unique_policy_ratio",
            "0.0",
            "--max_top_policy_share",
            "1.0",
        ],
    )
    exit_code = pipeline.run_pipeline(args)
    assert exit_code == 6


def test_pipeline_freezes_final(tmp_path: Path, monkeypatch: Any) -> None:
    out_dir = tmp_path / "gold_eval"
    out_dir.mkdir()
    _write_jsonl(out_dir / "gold_eval.jsonl", [{"ticket_index": 0}])
    _write_jsonl(
        out_dir / "gold_eval_auto.jsonl",
        [_make_auto_record(0, "Access", ["kb-0", "kb-1"]), _make_auto_record(1, "Hardware", ["kb-2"])],
    )
    _write_jsonl(out_dir / "gold_eval_review_queue.jsonl", [])

    monkeypatch.setattr(pipeline, "run_template_builder", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "run_autofill", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_load_kb_index", lambda: _mock_kb_index(10))
    monkeypatch.setattr(pipeline, "_kb_index_sha256", lambda _: "hash")
    monkeypatch.setattr(pipeline, "_get_git_commit_hash", lambda: "abc123")

    args = _make_args(
        out_dir,
        [
            "--final-name",
            "gold_eval_final.jsonl",
            "--min_unique_policy_ratio",
            "0.1",
            "--max_top_policy_share",
            "1.0",
            "--min_avg_steps",
            "1.0",
            "--min_avg_acceptance_criteria",
            "1.0",
        ],
    )
    exit_code = pipeline.run_pipeline(args)
    assert exit_code == 0
    assert (out_dir / "gold_eval_final.jsonl").exists()
    meta = json.loads((out_dir / "gold_eval_final.jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["start"] == args.start
    assert meta["limit"] == args.limit
