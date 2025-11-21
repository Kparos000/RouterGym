"""Contract schemas and validation stubs."""

from typing import Any, Dict


def get_schema(task: str) -> Dict[str, Any]:
    """Return a placeholder JSON schema for a task."""
    return {"title": task, "type": "object", "properties": {}}


def validate_payload(schema: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    """Stub schema validation."""
    return True
