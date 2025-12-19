"""JSON contract validation."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

from RouterGym.contracts.schema_contract import AgentOutputSchema


class JSONContract:
    """Validate JSON strings and return parsed objects."""

    def validate(self, text: str) -> Tuple[bool, Any]:
        """Return (is_valid, parsed_json or None)."""
        try:
            parsed = json.loads(text)
        except Exception:
            return False, None
        if not isinstance(parsed, dict):
            return False, None
        return True, parsed


def validate_agent_output(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate an AgentOutput payload and return a normalized copy or raise ValueError."""
    schema = AgentOutputSchema()
    data = dict(payload)
    if "ticket_id" in data and data.get("ticket_id") is not None:
        data["ticket_id"] = str(data["ticket_id"])
    else:
        data["ticket_id"] = ""
    data.setdefault("rewritten_query", data.get("original_query", ""))
    data.setdefault("topic_group", data.get("category", data.get("classifier_label", "")))
    data.setdefault("model_name", data.get("model_used", ""))
    data.setdefault("router_mode", data.get("router_name", ""))
    data.setdefault("base_model_name", data.get("base_model_name", data.get("model_name", "")))
    data.setdefault("escalation_model_name", data.get("escalation_model_name"))
    data.setdefault("classifier_label", data.get("classification", {}).get("label", data.get("topic_group", "")))
    data.setdefault("classifier_confidence", data.get("classification", {}).get("confidence", 0.0))
    data.setdefault("classifier_confidence_bucket", data.get("classification", {}).get("confidence_bucket", "low"))
    data.setdefault("memory_mode", data.get("context_mode", "none"))
    data.setdefault("kb_policy_ids", [])
    data.setdefault("kb_categories", [])
    data.setdefault(
        "escalation_flags",
        {"needs_human": False, "needs_llm_escalation": False, "policy_gap": False},
    )
    if "metrics" not in data:
        data["metrics"] = {
            "latency_ms": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
        }
    ok, errors = schema.validate(data)
    if not ok:
        raise ValueError(f"AgentOutput validation failed: {errors}")
    return data


__all__ = ["JSONContract", "validate_agent_output"]
