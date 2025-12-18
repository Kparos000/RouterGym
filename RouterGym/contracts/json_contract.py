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
    data.setdefault("classifier_label", data.get("classification", {}).get("label", ""))
    data.setdefault("classifier_confidence", data.get("classification", {}).get("confidence", 0.0))
    data.setdefault("classifier_confidence_bucket", data.get("classification", {}).get("confidence_bucket", "low"))
    data.setdefault("memory_mode", data.get("context_mode"))
    data.setdefault("kb_policy_ids", [])
    data.setdefault("kb_categories", [])
    if "metrics" not in data:
        data["metrics"] = {
            "latency_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
        }
    ok, errors = schema.validate(data)
    if not ok:
        raise ValueError(f"AgentOutput validation failed: {errors}")
    return data


__all__ = ["JSONContract", "validate_agent_output"]
