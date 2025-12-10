"""Heuristic classifier mimicking a fine-tuned SLM."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from RouterGym.classifiers.utils import (
    ClassifierMetadata,
    ClassifierProtocol,
    DEFAULT_LABELS,
    canonical_label,
    normalize_probabilities,
    apply_lexical_prior,
    CATEGORY_KEYWORDS,
)

_KEYWORDS: Dict[str, List[str]] = {
    "access": ["login", "password", "account", "mfa", "otp", "sso", "lockout"],
    "administrative rights": [
        "admin",
        "administrator",
        "privilege",
        "permission",
        "role",
        "rights",
        "entitlement",
        "group",
        "security group",
    ],
    "hardware": ["laptop", "printer", "dock", "monitor", "battery", "device", "keyboard", "mouse", "screen"],
    "hr support": ["benefit", "leave", "vacation", "payroll", "hr", "human resources"],
    "miscellaneous": ["misc", "general", "other"],
    "purchase": [
        "purchase",
        "buy",
        "order",
        "invoice",
        "procure",
        "bill",
        "subscription",
        "license",
        "licence",
        "renew",
        "payment",
        "quote",
        "po",
    ],
}


class SLMClassifier(ClassifierProtocol):
    """Rule-backed approximation of a fine-tuned small language model."""

    def __init__(
        self,
        labels: Optional[Iterable[str]] = None,
        model_reference: str = "mistralai/Mistral-7B-Instruct-v0.3",
        token_cost: float = 0.0025,
        latency_ms: float = 55.0,
    ) -> None:
        self.labels = [canonical_label(lbl) for lbl in (labels or DEFAULT_LABELS)]
        self.metadata = ClassifierMetadata(
            name="SLM Fine-tuned",
            mode="slm_finetuned",
            provider="simulated",
            model_reference=model_reference,
            token_cost=token_cost,
            latency_ms=latency_ms,
            description="Heuristic approximation of a fine-tuned SLM classifier",
        )

    def _score_text(self, text: str) -> Dict[str, float]:
        lower = (text or "").lower()
        scores: Dict[str, float] = {label: 0.1 for label in self.labels}
        for label, keywords in _KEYWORDS.items():
            canonical = canonical_label(label)
            if canonical not in scores:
                continue
            for keyword in keywords:
                if keyword in lower:
                    scores[canonical] += 1.5
        if len(lower.split()) < 5:
            scores["miscellaneous"] = scores.get("miscellaneous", 0.1) + 0.3
        return scores

    def predict_proba(self, text: str) -> Dict[str, float]:
        scores = self._score_text(text)
        base = normalize_probabilities(scores, self.labels)
        return apply_lexical_prior(text, base)

    def predict_label(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        label = max(probabilities, key=probabilities.__getitem__)
        if label == "miscellaneous":
            # If the model falls back to miscellaneous, check lexical prior for strong concrete evidence.
            lower = (text or "").lower()
            prior_hits: Dict[str, float] = {}
            for lbl, keywords in CATEGORY_KEYWORDS.items():
                count = 0.0
                for kw in keywords:
                    if kw and kw in lower:
                        count += 1.0
                prior_hits[canonical_label(lbl)] = count
            total_hits = sum(prior_hits.values())
            if total_hits > 0:
                # Normalize and find strongest non-misc label.
                miscless = {k: v for k, v in prior_hits.items() if k != "miscellaneous"}
                if miscless:
                    top_label = max(miscless, key=lambda k: miscless[k])
                    top_score = miscless[top_label] / max(total_hits, 1e-9)
                    if top_score >= 0.75:
                        label = top_label
                        self.metadata.description = self.metadata.description + " (misc overridden by lexical prior)"
        return label


__all__ = ["SLMClassifier"]
