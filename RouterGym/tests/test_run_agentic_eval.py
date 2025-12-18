from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from RouterGym.contracts.json_contract import validate_agent_output
import RouterGym.experiments.run_agentic_eval as agentic


def _fake_agent_output(category: str) -> Dict[str, Any]:
    return {
        "ticket_id": "t1",
        "original_query": "text",
        "rewritten_query": "text",
        "topic_group": category,
        "model_name": "slm1",
        "router_mode": "slm_dominant",
        "classifier_label": category,
        "classifier_confidence_bucket": "high",
        "classifier_backend": "encoder_calibrated",
        "classifier_confidence": 0.9,
        "classification": {
            "label": category,
            "confidence": 0.9,
            "confidence_bucket": "high",
        },
        "memory_mode": "none",
        "kb_policy_ids": [],
        "kb_categories": [],
        "resolution_steps": [],
        "final_answer": "stub",
        "reasoning": "stub",
        "escalation_flags": {"needs_human": False, "needs_llm_escalation": False, "policy_gap": False},
        "metrics": {
            "latency_ms": 1.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
        },
    }


def test_run_agentic_eval_writes_jsonl(monkeypatch: Any, tmp_path: Path) -> None:
    # Stub dataset loader to avoid real CSV.
    fake_df = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "label": ["hardware", "access", "purchase"],
        }
    )
    monkeypatch.setattr(agentic, "load_dataset", lambda n=None: fake_df)

    # Stub pipeline to return deterministic AgentOutput.
    def _fake_pipeline(ticket_text: str, router_name: str, context_mode: str) -> Dict[str, Any]:
        return _fake_agent_output("Hardware")

    monkeypatch.setattr(agentic, "run_ticket_pipeline", _fake_pipeline)

    out_path = tmp_path / "agentic.jsonl"
    agentic.run_agentic_eval(
        ticket_limit=2,
        router_name="slm_dominant",
        context_mode="none",
        output_path=out_path,
    )

    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        # Validate core contract.
        validated = validate_agent_output(obj)
        assert validated["topic_group"] == "Hardware"
        assert "ticket_id" in obj
        assert "gold_label" in obj
