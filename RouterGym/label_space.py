"""Canonical label space and normalization utilities."""

from __future__ import annotations

import logging
from typing import Dict, List

CANONICAL_LABELS: List[str] = [
    "access",
    "administrative rights",
    "hardware",
    "hr support",
    "purchase",
    "miscellaneous",
]

CANONICAL_LABEL_SET = set(CANONICAL_LABELS)
LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(CANONICAL_LABELS)}
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}

# Collapse long-tail labels into the canonical set. We explicitly map
# legacy labels like "internal project" and "storage" into "miscellaneous"
# to keep a single, consistent label space across data, prompts, and classifiers.
LABEL_NORMALIZATION_MAP: Dict[str, str] = {
    "access": "access",
    "login": "access",
    "log in": "access",
    "password": "access",
    "account": "access",
    "mfa": "access",
    "sso": "access",
    "otp": "access",
    "lockout": "access",
    "administrative rights": "administrative rights",
    "admin rights": "administrative rights",
    "admin": "administrative rights",
    "administrator": "administrative rights",
    "admin role": "administrative rights",
    "role change": "administrative rights",
    "permission": "administrative rights",
    "permissions": "administrative rights",
    "privilege": "administrative rights",
    "entitlement": "administrative rights",
    "group membership": "administrative rights",
    "security group": "administrative rights",
    "hardware": "hardware",
    "device": "hardware",
    "laptop": "hardware",
    "printer": "hardware",
    "monitor": "hardware",
    "dock": "hardware",
    "keyboard": "hardware",
    "mouse": "hardware",
    "vpn": "access",
    "network": "miscellaneous",
    "software": "miscellaneous",
    "hr": "hr support",
    "hr support": "hr support",
    "hr_support": "hr support",
    "human resources": "hr support",
    "benefits": "hr support",
    "leave": "hr support",
    "vacation": "hr support",
    "payroll": "hr support",
    "purchase": "purchase",
    "buy": "purchase",
    "order": "purchase",
    "invoice": "purchase",
    "billing": "purchase",
    "subscription": "purchase",
    "licence": "purchase",
    "license": "purchase",
    "renewal": "purchase",
    "quote": "purchase",
    "po": "purchase",
    "procurement": "purchase",
    "payment": "purchase",
    "refund": "purchase",
    "misc": "miscellaneous",
    "miscellaneous": "miscellaneous",
    "general": "miscellaneous",
    "other": "miscellaneous",
    "storage": "miscellaneous",
    "internal project": "miscellaneous",
    "internal projects": "miscellaneous",
}

logger = logging.getLogger(__name__)


def canonical_label(label: str | None) -> str:
    """Map any free-form label into the canonical 6-label space."""
    text = (label or "").strip().lower()
    if not text:
        logger.warning("Empty label encountered; mapping to 'miscellaneous'.")
        return "miscellaneous"
    if text in LABEL_NORMALIZATION_MAP:
        return LABEL_NORMALIZATION_MAP[text]
    if text in CANONICAL_LABEL_SET:
        return text
    for key, target in LABEL_NORMALIZATION_MAP.items():
        if key and key in text:
            return target
    logger.warning("Unexpected label '%s'; mapping to 'miscellaneous'. Extend LABEL_NORMALIZATION_MAP if needed.", text)
    return "miscellaneous"


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
