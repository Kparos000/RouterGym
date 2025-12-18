"""Schema contract validation (minimal)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from RouterGym.label_space import CANONICAL_LABEL_SET, canonicalize_label

ALLOWED_CONTEXT_MODES = {"none", "rag_dense", "rag_bm25", "rag_hybrid"}
ALLOWED_CONFIDENCE_BUCKETS = {"high", "medium", "low"}
_METRIC_FIELDS = {
    "latency_ms": (float, int),
    "total_input_tokens": (int, float),
    "total_output_tokens": (int, float),
    "total_cost_usd": (float, int),
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
        "topic_group",
        "model_name",
        "router_mode",
        "classifier_label",
        "classifier_confidence_bucket",
        "classifier_backend",
        "memory_mode",
        "reasoning",
        "final_answer",
    }

    def validate(self, json_obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not isinstance(json_obj, dict):
            return False, ["Output is not a JSON object"]

        if "ticket_id" in json_obj and json_obj.get("ticket_id") is not None:
            if not isinstance(json_obj["ticket_id"], str):
                errors.append("ticket_id must be a string when provided")

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

        for label_field in ("topic_group", "classifier_label"):
            if label_field in json_obj:
                try:
                    json_obj[label_field] = canonicalize_label(json_obj[label_field])
                except RuntimeError:
                    errors.append(f"Field {label_field} is not in the allowed label set")
        if "classifier_confidence_bucket" in json_obj:
            bucket = json_obj.get("classifier_confidence_bucket")
            if bucket and isinstance(bucket, str) and bucket not in ALLOWED_CONFIDENCE_BUCKETS:
                errors.append(
                    f"classifier_confidence_bucket must be one of {sorted(ALLOWED_CONFIDENCE_BUCKETS)}"
                )

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
        if "memory_mode" in json_obj and json_obj.get("memory_mode") is not None:
            mem_mode = json_obj.get("memory_mode")
            if not isinstance(mem_mode, str) or mem_mode not in ALLOWED_CONTEXT_MODES:
                errors.append(f"memory_mode must be one of {sorted(ALLOWED_CONTEXT_MODES)} or None")

        if "classification" in json_obj:
            cls = json_obj["classification"]
            if not isinstance(cls, dict):
                errors.append("classification must be an object")
            else:
                if "label" in cls:
                    try:
                        cls["label"] = canonicalize_label(cls["label"])
                    except RuntimeError:
                        errors.append("classification.label is not in the allowed label set")
                if "confidence" in cls:
                    conf_val = cls.get("confidence", 0.0)
                    if not isinstance(conf_val, (int, float)) or not (0.0 <= float(conf_val) <= 1.0):
                        errors.append("classification.confidence must be between 0 and 1")
                if "confidence_bucket" in cls:
                    if cls["confidence_bucket"] not in ALLOWED_CONFIDENCE_BUCKETS:
                        errors.append(
                            f"classification.confidence_bucket must be one of {sorted(ALLOWED_CONFIDENCE_BUCKETS)}"
                        )

        # resolution_steps
        if "resolution_steps" not in json_obj:
            errors.append("Missing field: resolution_steps")
        else:
            steps = json_obj["resolution_steps"]
            if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
                errors.append("resolution_steps must be a list of strings")

        # escalation flags
        if "escalation_flags" not in json_obj:
            errors.append("Missing field: escalation_flags")
        else:
            esc = json_obj["escalation_flags"]
            if not isinstance(esc, dict):
                errors.append("escalation_flags must be an object")
            else:
                for key in ("needs_human", "needs_llm_escalation", "policy_gap"):
                    if key not in esc:
                        errors.append(f"escalation_flags missing field: {key}")
                    elif not isinstance(esc[key], bool):
                        errors.append(f"escalation_flags field {key} must be a bool")

        if "kb_policy_ids" in json_obj:
            ids = json_obj.get("kb_policy_ids")
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                errors.append("kb_policy_ids must be a list of strings")
        if "kb_categories" in json_obj:
            cats = json_obj.get("kb_categories")
            if not isinstance(cats, list) or not all(isinstance(c, str) for c in cats):
                errors.append("kb_categories must be a list of strings")
            else:
                for cat in cats:
                    if cat and cat not in CANONICAL_LABEL_SET:
                        errors.append(f"kb_categories contains non-canonical label: {cat}")

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
