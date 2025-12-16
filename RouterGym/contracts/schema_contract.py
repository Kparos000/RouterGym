"""Schema contract validation (minimal)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from RouterGym.label_space import CANONICAL_LABEL_SET, canonicalize_label

ALLOWED_CONTEXT_MODES = {"none", "rag_dense", "rag_hybrid", "transcript"}
ALLOWED_CONFIDENCE_BUCKETS = {"high", "medium", "low"}
_METRIC_FIELDS = {
    "latency_ms": (float, int, type(None)),
    "prompt_tokens": (int, type(None)),
    "completion_tokens": (int, type(None)),
    "total_tokens": (int, type(None)),
    "cost_usd": (float, int, type(None)),
}


class SchemaContract:
    """Validate the minimal required schema for agent outputs."""

    required_fields = {
        "reasoning": str,
        "final_answer": str,
        "predicted_category": str,
    }

    def validate(self, json_obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Return (is_valid, errors)."""
        errors: List[str] = []
        if not isinstance(json_obj, dict):
            return False, ["Output is not a JSON object"]
        for field, ftype in self.required_fields.items():
            if field not in json_obj:
                errors.append(f"Missing field: {field}")
                continue
            if not isinstance(json_obj[field], ftype):
                errors.append(f"Field {field} has wrong type")
                continue
            if isinstance(json_obj[field], str) and not json_obj[field].strip():
                errors.append(f"Field {field} is empty")
            if field == "predicted_category":
                try:
                    normalized = canonicalize_label(json_obj[field])
                except RuntimeError:
                    errors.append("Field predicted_category is not in the allowed label set")
                else:
                    if normalized not in CANONICAL_LABEL_SET:
                        errors.append("Field predicted_category is not in the allowed label set")
        return len(errors) == 0, errors


class AgentOutputSchema:
    """Validate structured AgentOutput payloads for a single ticket."""

    required_string_fields = {
        "original_query",
        "rewritten_query",
        "category",
        "classifier_backend",
        "router_name",
        "model_used",
        "context_mode",
        "reasoning",
    }

    def validate(self, json_obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not isinstance(json_obj, dict):
            return False, ["Output is not a JSON object"]

        for field in self.required_string_fields:
            if field not in json_obj:
                errors.append(f"Missing field: {field}")
                continue
            value = json_obj[field]
            if not isinstance(value, str):
                errors.append(f"Field {field} must be a string")
                continue
            if not value.strip():
                errors.append(f"Field {field} is empty")

        if "category" in json_obj:
            try:
                json_obj["category"] = canonicalize_label(json_obj["category"])
            except RuntimeError:
                errors.append("Field category is not in the allowed label set")

        if "classifier_confidence" not in json_obj:
            errors.append("Missing field: classifier_confidence")
        else:
            conf = json_obj["classifier_confidence"]
            if not isinstance(conf, (int, float)):
                errors.append("classifier_confidence must be a number")
            elif not (0.0 <= float(conf) <= 1.0):
                errors.append("classifier_confidence must be between 0 and 1")

        if "context_mode" in json_obj and isinstance(json_obj.get("context_mode"), str):
            if json_obj["context_mode"] not in ALLOWED_CONTEXT_MODES:
                errors.append(f"context_mode must be one of {sorted(ALLOWED_CONTEXT_MODES)}")

        if "classification" not in json_obj:
            errors.append("Missing field: classification")
        else:
            cls = json_obj["classification"]
            if not isinstance(cls, dict):
                errors.append("classification must be an object")
            else:
                if "label" not in cls or not isinstance(cls.get("label"), str):
                    errors.append("classification.label must be a string")
                else:
                    try:
                        cls["label"] = canonicalize_label(cls["label"])
                    except RuntimeError:
                        errors.append("classification.label is not in the allowed label set")
                if "confidence" not in cls or not isinstance(cls.get("confidence"), (int, float)):
                    errors.append("classification.confidence must be a number")
                else:
                    conf_val = float(cls["confidence"])
                    if not (0.0 <= conf_val <= 1.0):
                        errors.append("classification.confidence must be between 0 and 1")
                if "confidence_bucket" not in cls or not isinstance(cls.get("confidence_bucket"), str):
                    errors.append("classification.confidence_bucket must be a string")
                else:
                    if cls["confidence_bucket"] not in ALLOWED_CONFIDENCE_BUCKETS:
                        errors.append(
                            f"classification.confidence_bucket must be one of {sorted(ALLOWED_CONFIDENCE_BUCKETS)}"
                        )

        if "resolution_steps" not in json_obj:
            errors.append("Missing field: resolution_steps")
        else:
            steps = json_obj["resolution_steps"]
            if not isinstance(steps, list):
                errors.append("resolution_steps must be a list")
            elif not all(isinstance(s, str) for s in steps):
                errors.append("resolution_steps must contain strings")

        if "escalation" not in json_obj:
            errors.append("Missing field: escalation")
        else:
            esc = json_obj["escalation"]
            if not isinstance(esc, dict):
                errors.append("escalation must be an object")
            else:
                for key in ("agent_escalation", "human_escalation"):
                    if key not in esc:
                        errors.append(f"escalation missing field: {key}")
                    elif not isinstance(esc[key], bool):
                        errors.append(f"escalation field {key} must be a bool")
                if "reasons" not in esc:
                    errors.append("escalation missing field: reasons")
                elif not isinstance(esc["reasons"], list) or not all(
                    isinstance(r, str) for r in esc["reasons"]
                ):
                    errors.append("escalation.reasons must be a list of strings")

        if "metrics" not in json_obj:
            errors.append("Missing field: metrics")
        else:
            metrics = json_obj["metrics"]
            if not isinstance(metrics, dict):
                errors.append("metrics must be an object")
            else:
                for field, allowed_types in _METRIC_FIELDS.items():
                    if field not in metrics:
                        errors.append(f"metrics missing field: {field}")
                        continue
                    if not isinstance(metrics[field], allowed_types):
                        errors.append(f"metrics field {field} has wrong type")

        return len(errors) == 0, errors


__all__ = ["SchemaContract", "AgentOutputSchema", "ALLOWED_CONTEXT_MODES", "ALLOWED_CONFIDENCE_BUCKETS"]
