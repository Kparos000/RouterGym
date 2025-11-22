"""JSON contract validation."""

from __future__ import annotations

import json
from typing import Any, Tuple


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


__all__ = ["JSONContract"]
