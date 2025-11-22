"""Schema contract validation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


class SchemaContract:
    """Validate the required schema for agent outputs."""

    required_fields = {
        "classification": str,
        "reasoning": str,
        "action_steps": list,
        "final_answer": str,
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
        return len(errors) == 0, errors


__all__ = ["SchemaContract"]
