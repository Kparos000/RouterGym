"""Tests for gold eval template builder."""

from __future__ import annotations

import pandas as pd

from RouterGym.scripts import build_gold_eval_template as builder


def test_build_gold_eval_template_structure() -> None:
    df = pd.DataFrame(
        {
            "Document": [
                "Reset my password please",
                "Laptop screen is flickering",
                "Need vpn access for travel",
            ],
            "Topic_group": ["access", "Hardware", "Access"],
        }
    )
    target_counts = {"Access": 2, "Hardware": 1}

    records, counts = builder.build_gold_eval(df, target_counts, seed=7)

    assert counts == {"Access": 2, "Hardware": 1}
    assert len(records) == 3

    first = records[0]
    assert set(first.keys()) == {"ticket_index", "topic_group", "ticket_text", "gold_resolution"}
    gold = first["gold_resolution"]
    assert set(gold.keys()) == {
        "summary",
        "steps",
        "escalation_required",
        "escalation_reason",
        "kb_policies",
        "acceptance_criteria",
    }
    assert gold["summary"] == "TODO"
    assert isinstance(gold["steps"], list)
    assert isinstance(gold["kb_policies"], list)
    assert isinstance(gold["acceptance_criteria"], list)
    assert isinstance(gold["escalation_required"], bool)


def test_build_gold_eval_limit_and_start() -> None:
    df = pd.DataFrame(
        {
            "Document": [f"doc {i}" for i in range(6)],
            "Topic_group": ["Access", "Hardware", "Access", "Hardware", "Access", "Hardware"],
        }
    )
    target_counts = {"Access": 3, "Hardware": 3}
    records, counts = builder.build_gold_eval(df, target_counts, seed=1, start=2, limit=2)
    assert len(records) == 2
    assert sum(counts.values()) == 2
