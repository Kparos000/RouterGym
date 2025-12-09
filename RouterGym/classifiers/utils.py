"""Utilities and registry helpers for RouterGym classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Protocol, runtime_checkable

from RouterGym.label_space import CANONICAL_LABELS, canonical_label

DEFAULT_LABELS: List[str] = [label for label in CANONICAL_LABELS]


def canonical_mode(name: str | None) -> str:
    """Normalize classifier/router mode identifiers without touching label semantics."""
    return str(name or "").strip().lower() or "unknown"


def normalize_probabilities(scores: Dict[str, float], labels: Iterable[str]) -> Dict[str, float]:
    """Project arbitrary score dicts into a normalized probability distribution."""
    normalized_labels = [canonical_label(lbl) for lbl in labels]
    projected: Dict[str, float] = {lbl: 0.0 for lbl in normalized_labels}
    for key, value in scores.items():
        normalized_key = canonical_label(key)
        if normalized_key in projected:
            projected[normalized_key] = max(0.0, float(value))
    total = sum(projected.values())
    if total <= 0:
        uniform = 1.0 / max(len(projected), 1)
        return {label: uniform for label in projected}
    return {label: projected[label] / total for label in projected}


CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "access": [
        "login",
        "log in",
        "password",
        "account",
        "lockout",
        "locked",
        "sso",
        "sign-in",
        "sign in",
        "mfa",
    ],
    # Explicitly separate administrative rights from access to strengthen recall on permissions.
    "administrative rights": [
        "admin",
        "administrator",
        "permission",
        "role",
        "rights",
        "privilege",
        "entitlement",
        "group membership",
        "security group",
        "access change",
        "group change",
    ],
    "hardware": [
        "laptop",
        "computer",
        "pc",
        "desktop",
        "printer",
        "monitor",
        "screen",
        "keyboard",
        "mouse",
        "device",
        "dock",
    ],
    "hr support": [
        "leave",
        "vacation",
        "holiday",
        "benefits",
        "payroll",
        "manager",
        "hr",
        "human resources",
    ],
    "purchase": [
        "purchase",
        "buy",
        "buying",
        "order",
        "invoice",
        "billing",
        "subscription",
        "licence",
        "license",
        "renewal",
        "payment",
        "quote",
        "po",
        "procurement",
        "refund",
    ],
    "miscellaneous": ["general", "question", "enquiry", "inquiry", "other"],
}


def apply_lexical_prior(text: str, probs: Dict[str, float], alpha: float = 0.8, beta: float = 0.2) -> Dict[str, float]:
    """Blend model probabilities with a simple keyword-based prior."""
    try:
        lower = (text or "").lower()
        hits: Dict[str, float] = {}
        for label, keywords in CATEGORY_KEYWORDS.items():
            count = 0.0
            for kw in keywords:
                if kw and kw in lower:
                    count += 1.0
            hits[canonical_label(label)] = count

        total_hits = sum(hits.values())
        if total_hits <= 0:
            return probs

        epsilon = 1e-6
        prior = {lbl: (hits.get(lbl, 0.0) + epsilon) for lbl in probs}
        prior_total = sum(prior.values()) or 1.0
        prior = {lbl: val / prior_total for lbl, val in prior.items()}

        blended: Dict[str, float] = {}
        for lbl in probs:
            blended[lbl] = max(0.0, alpha * probs.get(lbl, 0.0) + beta * prior.get(lbl, 0.0))
        return normalize_probabilities(blended, probs.keys())
    except Exception:
        return probs


@dataclass(slots=True)
class ClassifierMetadata:
    name: str
    mode: str
    provider: str
    model_reference: str
    token_cost: float = 0.0
    latency_ms: float = 0.0
    description: str = ""

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "name": self.name,
            "mode": self.mode,
            "provider": self.provider,
            "model_reference": self.model_reference,
            "token_cost": self.token_cost,
            "latency_ms": self.latency_ms,
            "description": self.description,
        }


@runtime_checkable
class ClassifierProtocol(Protocol):
    metadata: ClassifierMetadata

    def predict_proba(self, text: str) -> Dict[str, float]:  # pragma: no cover - Protocol definition only
        ...

    def predict_label(self, text: str) -> str:  # pragma: no cover - Protocol definition only
        ...


ClassifierFactory = Callable[[], ClassifierProtocol]

_CLASSIFIER_REGISTRY: Dict[str, ClassifierFactory] = {}


def register_classifier(name: str, factory: ClassifierFactory) -> None:
    key = canonical_mode(name)
    _CLASSIFIER_REGISTRY[key] = factory


def get_classifier(name: str) -> ClassifierProtocol:
    key = canonical_mode(name)
    if key not in _CLASSIFIER_REGISTRY:
        available = ", ".join(sorted(_CLASSIFIER_REGISTRY)) or "none"
        raise ValueError(f"Unknown classifier '{name}'. Available options: {available}")
    classifier = _CLASSIFIER_REGISTRY[key]()
    if not isinstance(classifier, ClassifierProtocol):
        raise TypeError(f"Classifier '{name}' does not implement the required protocol")
    return classifier


def available_classifiers() -> List[str]:
    return list(_CLASSIFIER_REGISTRY.keys())


__all__ = [
    "DEFAULT_LABELS",
    "ClassifierMetadata",
    "ClassifierProtocol",
    "ClassifierFactory",
    "available_classifiers",
    "canonical_label",
    "canonical_mode",
    "get_classifier",
    "normalize_probabilities",
    "CATEGORY_KEYWORDS",
    "apply_lexical_prior",
    "register_classifier",
]
