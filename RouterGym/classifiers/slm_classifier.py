"""Prompted small language model classifier with deterministic decoding and robust parsing."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from RouterGym.agents import generator as gen
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
    "Access": ["login", "password", "account", "mfa", "otp", "sso", "lockout"],
    "Administrative rights": [
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
    "Hardware": ["laptop", "printer", "dock", "monitor", "battery", "device", "keyboard", "mouse", "screen"],
    "HR Support": [
        "benefit",
        "benefits",
        "leave",
        "vacation",
        "payroll",
        "hr",
        "human resources",
        "salary",
        "compensation",
        "paternity",
        "maternity",
        "time off",
        "pto",
    ],
    "Miscellaneous": ["misc", "general", "other"],
    "Purchase": [
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
        "vendor",
        "supplier",
        "procure",
        "procurement",
        "billing",
    ],
    "Internal Project": ["internal project", "project work", "initiative", "internal program"],
    "Storage": ["storage", "quota", "disk", "drive", "backup", "archive"],
}


class SLMClassifier(ClassifierProtocol):
    """Prompt-driven approximation of a fine-tuned small language model."""

    def __init__(
        self,
        labels: Optional[Iterable[str]] = None,
        model_reference: str = "mistralai/Mistral-7B-Instruct-v0.3",
        token_cost: float = 0.0025,
        latency_ms: float = 55.0,
        model: Any = None,
    ) -> None:
        self.labels = [canonical_label(lbl) for lbl in (labels or DEFAULT_LABELS)]
        self.label_set = set(self.labels)
        self.model = model
        self._last_model_label: Optional[str] = None
        self.metadata = ClassifierMetadata(
            name="SLM Fine-tuned",
            mode="slm_finetuned",
            provider="simulated",
            model_reference=model_reference,
            token_cost=token_cost,
            latency_ms=latency_ms,
            description="Prompted small LM classifier (deterministic, lexical fallback)",
        )

    # === Prompted path ===
    def _build_prompt(self, text: str) -> str:
        instruction = gen.classification_instruction()
        ticket_block = f"Ticket:\n{text.strip()}"
        return "\n\n".join([instruction, ticket_block, "Answer with strict JSON only."])

    def _parse_output(self, raw: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                fragment = gen._extract_json_fragment(raw)
                if isinstance(fragment, dict):
                    return fragment
        return None

    def _predict_with_model(self, text: str) -> Optional[Tuple[Dict[str, float], str]]:
        """Call the underlying model with deterministic decoding and robust parsing.

        We try once, and if parsing fails, retry with the same prompt. If parsing still fails,
        we return None so the heuristic/lexical path can take over.
        """
        if self.model is None:
            return None
        prompt = self._build_prompt(text)
        parsed: Optional[Dict[str, Any]] = None
        for _ in range(2):
            raw = gen._call_model(self.model, prompt)
            parsed = self._parse_output(raw)
            if parsed:
                break
        if parsed is None:
            return None
        try:
            category = canonical_label(
                parsed.get("category") or parsed.get("label") or parsed.get("predicted_category") or ""
            )
        except RuntimeError:
            category = None
        if category not in self.label_set:
            # Fall back to lexical prior if the model produced an unexpected label.
            prior = apply_lexical_prior(text, {lbl: 1.0 / max(len(self.labels), 1) for lbl in self.labels})
            category = max(prior, key=lambda k: prior[k])
        base = {lbl: 1.0 / max(len(self.labels), 1) for lbl in self.labels}
        probs = apply_lexical_prior(text, base, alpha=0.6, beta=0.4)
        if category in probs:
            # Make the chosen category visibly dominant while keeping distribution normalized.
            probs[category] = min(1.0, probs[category] + 0.3)
            probs = normalize_probabilities(probs, self.labels)
        return probs, category

    def _score_text(self, text: str) -> Dict[str, float]:
        lower = (text or "").lower()
        scores: Dict[str, float] = {label: 0.05 for label in self.labels}
        for label, keywords in _KEYWORDS.items():
            canonical = canonical_label(label)
            if canonical not in scores:
                continue
            for keyword in keywords:
                if keyword in lower:
                    boost = 1.8 if canonical in {"HR Support", "Purchase"} else 1.5
                    scores[canonical] += boost
        if len(lower.split()) < 5:
            scores["Miscellaneous"] = scores.get("Miscellaneous", 0.05) + 0.2
        return scores

    def predict_proba(self, text: str) -> Dict[str, float]:
        model_result = self._predict_with_model(text)
        if model_result is not None:
            probs, lbl = model_result
            self._last_model_label = lbl
            return probs
        scores = self._score_text(text)
        base = normalize_probabilities(scores, self.labels)
        return apply_lexical_prior(text, base)

    def predict_label(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        label = self._last_model_label or max(probabilities, key=probabilities.__getitem__)
        if label == "Miscellaneous":
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
                miscless = {k: v for k, v in prior_hits.items() if k != "Miscellaneous"}
                if miscless:
                    top_label = max(miscless, key=lambda k: miscless[k])
                    top_score = miscless[top_label] / max(total_hits, 1e-9)
                    if top_score >= 0.65:
                        label = top_label
                        self.metadata.description = self.metadata.description + " (misc overridden by lexical prior)"
        return label


__all__ = ["SLMClassifier"]
