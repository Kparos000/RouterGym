"""Tests for deterministic scoring against frozen gold examples."""

from __future__ import annotations

from RouterGym.evaluation import gold_scoring, metrics
from RouterGym.scripts import score_gold_outputs


def _gold_record() -> dict:
    return {
        "ticket_index": 7,
        "topic_group": "Access",
        "ticket_text": "Need VPN access",
        "review_status": "human_approved",
        "gold_resolution": {
            "summary": "Provision VPN access",
            "steps": [
                "Verify the user's identity and account status.",
                "Provision VPN access according to the policy.",
                "Ask the user to test the VPN connection.",
            ],
            "escalation_required": False,
            "escalation_reason": "",
            "kb_policies": ["access.vpn"],
            "acceptance_criteria": [
                "The user can sign in to the VPN successfully.",
                "The user confirms remote access is working.",
            ],
        },
    }


def test_gold_scoring_good_match() -> None:
    generated = {
        "ticket_id": "7",
        "resolution_steps": [
            "Confirm the user's identity and that the account is active.",
            "Grant VPN access following the VPN policy.",
            "Have the user test the VPN sign-in from their workstation.",
        ],
        "acceptance_criteria": [
            "User signs in to VPN successfully.",
            "User confirms remote access works.",
        ],
        "kb_policy_ids": ["access.vpn"],
        "escalation_flags": {
            "needs_human": False,
            "needs_llm_escalation": False,
            "policy_gap": False,
            "reasons": [],
        },
    }

    scores = gold_scoring.score_record_against_gold(generated, _gold_record()).as_dict()
    assert scores["gold_match_found"] is True
    assert scores["step_coverage_score"] is not None and scores["step_coverage_score"] > 0.60
    assert (
        scores["acceptance_criteria_alignment_score"] is not None
        and scores["acceptance_criteria_alignment_score"] > 0.60
    )
    assert scores["policy_grounding_match_score"] == 1.0
    assert (
        scores["overall_gold_quality_score"] is not None
        and scores["overall_gold_quality_score"] > 0.75
    )


def test_gold_scoring_partial_match() -> None:
    generated = {
        "ticket_id": "7",
        "resolution_steps": [
            "Verify the account.",
            "Tell the user to try again later.",
        ],
        "final_answer": "User should confirm whether VPN access is working after a retry.",
        "kb_policy_ids": [],
        "escalation_flags": {
            "needs_human": False,
            "needs_llm_escalation": False,
            "policy_gap": False,
            "reasons": [],
        },
    }

    scores = gold_scoring.score_record_against_gold(generated, _gold_record()).as_dict()
    assert scores["step_coverage_score"] is not None and 0.15 < scores["step_coverage_score"] < 0.75
    assert (
        scores["acceptance_criteria_alignment_score"] is not None
        and 0.10 < scores["acceptance_criteria_alignment_score"] < 0.75
    )
    assert scores["policy_grounding_match_score"] == 0.0
    assert (
        scores["overall_gold_quality_score"] is not None
        and 0.10 < scores["overall_gold_quality_score"] < 0.70
    )


def test_gold_scoring_bad_match() -> None:
    generated = {
        "ticket_id": "7",
        "resolution_steps": [
            "Replace the printer toner cartridge.",
            "Restart the office printer.",
            "Print a test page.",
        ],
        "acceptance_criteria": ["The printer produces a test page."],
        "kb_policy_ids": ["hardware.printer"],
        "escalation_flags": {
            "needs_human": True,
            "needs_llm_escalation": False,
            "policy_gap": True,
            "reasons": ["security_review"],
        },
    }

    scores = gold_scoring.score_record_against_gold(generated, _gold_record()).as_dict()
    assert scores["step_coverage_score"] is not None and scores["step_coverage_score"] < 0.20
    assert (
        scores["acceptance_criteria_alignment_score"] is not None
        and scores["acceptance_criteria_alignment_score"] < 0.20
    )
    assert scores["escalation_correctness_score"] == 0.0
    assert scores["policy_grounding_match_score"] == 0.0
    assert (
        scores["overall_gold_quality_score"] is not None
        and scores["overall_gold_quality_score"] < 0.20
    )


def test_score_records_against_gold_merges_fields() -> None:
    scored = score_gold_outputs.score_records_against_gold(
        records=[
            {
                "ticket_id": "7",
                "resolution_steps": ["Verify account", "Provision VPN", "Ask user to test"],
                "acceptance_criteria": ["VPN connects", "User confirms access"],
                "kb_policy_ids": ["access.vpn"],
                "escalation_flags": {
                    "needs_human": False,
                    "needs_llm_escalation": False,
                    "policy_gap": False,
                    "reasons": [],
                },
            }
        ],
        gold_records=[_gold_record()],
    )

    assert len(scored) == 1
    assert scored[0]["gold_match_found"] is True
    assert "overall_gold_quality_score" in scored[0]


def test_compute_all_metrics_includes_gold_scores() -> None:
    metrics_out = metrics.compute_all_metrics(
        {
            "output": {
                "final_answer": "VPN is working",
                "reasoning": "Used the VPN policy",
                "predicted_category": "Access",
            },
            "gold_category": "Access",
            "predicted_category": "Access",
            "kb_snippets": ["VPN access policy"],
            "kb_attached": True,
            "model_used": "slm",
            "latency_ms": 12.0,
            "prompt_text": "prompt",
            "resolution_steps": ["Verify account", "Provision VPN", "Ask user to test"],
            "acceptance_criteria": ["VPN connects", "User confirms access"],
            "kb_policy_ids": ["access.vpn"],
            "escalation_flags": {
                "needs_human": False,
                "needs_llm_escalation": False,
                "policy_gap": False,
                "reasons": [],
            },
            "gold_record": _gold_record(),
        }
    )

    assert "step_coverage_score" in metrics_out
    assert "overall_gold_quality_score" in metrics_out
