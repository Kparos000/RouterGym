"""Canonical label space and normalization utilities."""

from __future__ import annotations

import logging
from typing import Dict, List

CANONICAL_LABELS: List[str] = [
    "Access",
    "Administrative rights",
    "HR Support",
    "Hardware",
    "Internal Project",
    "Miscellaneous",
    "Purchase",
    "Storage",
]

CANONICAL_LABEL_SET = set(CANONICAL_LABELS)
_CANONICAL_LOWER_TO_LABEL: Dict[str, str] = {label.lower(): label for label in CANONICAL_LABELS}
LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}

# Map common variants and synonyms into the canonical 8-label space. All keys are lower-case.
LABEL_NORMALIZATION_MAP: Dict[str, str] = {
    # Access
    "access": "Access",
    "login": "Access",
    "log in": "Access",
    "password": "Access",
    "account": "Access",
    "mfa": "Access",
    "sso": "Access",
    "otp": "Access",
    "lockout": "Access",
    "vpn": "Access",
    # Administrative rights
    "administrative rights": "Administrative rights",
    "admin rights": "Administrative rights",
    "admin": "Administrative rights",
    "administrator": "Administrative rights",
    "admin role": "Administrative rights",
    "role change": "Administrative rights",
    "permission": "Administrative rights",
    "permissions": "Administrative rights",
    "privilege": "Administrative rights",
    "entitlement": "Administrative rights",
    "group membership": "Administrative rights",
    "security group": "Administrative rights",
    # Hardware
    "hardware": "Hardware",
    "device": "Hardware",
    "laptop": "Hardware",
    "printer": "Hardware",
    "monitor": "Hardware",
    "dock": "Hardware",
    "keyboard": "Hardware",
    "mouse": "Hardware",
    # HR Support
    "hr": "HR Support",
    "hr support": "HR Support",
    "hr_support": "HR Support",
    "human resources": "HR Support",
    "benefits": "HR Support",
    "leave": "HR Support",
    "vacation": "HR Support",
    "payroll": "HR Support",
    # Purchase
    "purchase": "Purchase",
    "buy": "Purchase",
    "order": "Purchase",
    "invoice": "Purchase",
    "billing": "Purchase",
    "subscription": "Purchase",
    "licence": "Purchase",
    "license": "Purchase",
    "renewal": "Purchase",
    "quote": "Purchase",
    "po": "Purchase",
    "procurement": "Purchase",
    "payment": "Purchase",
    "refund": "Purchase",
    # Internal Project
    "internal project": "Internal Project",
    "internal projects": "Internal Project",
    # Storage
    "storage": "Storage",
    # Miscellaneous
    "misc": "Miscellaneous",
    "miscellaneous": "Miscellaneous",
    "general": "Miscellaneous",
    "other": "Miscellaneous",
}

logger = logging.getLogger(__name__)


def canonical_label(label: str | None) -> str:
    """Map any free-form label into the canonical 8-label space."""
    text = (label or "").strip()
    if not text:
        raise RuntimeError("Empty label encountered; provide one of the canonical labels or extend the mapping.")
    normalized = " ".join(text.lower().replace("_", " ").split())
    if normalized in LABEL_NORMALIZATION_MAP:
        return LABEL_NORMALIZATION_MAP[normalized]
    if normalized in _CANONICAL_LOWER_TO_LABEL:
        return _CANONICAL_LOWER_TO_LABEL[normalized]
    for key, target in LABEL_NORMALIZATION_MAP.items():
        if key and key in normalized:
            return target
    raise RuntimeError(
        f"Unexpected label '{label}'. Extend RouterGym/label_space.py LABEL_NORMALIZATION_MAP to map it explicitly."
    )


def canonicalize_label(label: str | None) -> str:
    """Public normalization helper to ensure labels land in CANONICAL_LABELS."""
    return canonical_label(label)


__all__ = [
    "CANONICAL_LABELS",
    "CANONICAL_LABEL_SET",
    "LABEL_NORMALIZATION_MAP",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "canonical_label",
    "canonicalize_label",
]
