"""Compatibility wrapper to policy_kb loader."""

from RouterGym.data.policy_kb.kb_loader import load_kb, retrieve  # type: ignore

__all__ = ["load_kb", "retrieve"]
