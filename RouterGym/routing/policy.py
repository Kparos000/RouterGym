"""Canonical routing policy shared across RouterGym entry points.

This module is intentionally lightweight and dependency-free so router
decisions can be reasoned about and tested in isolation. It centralizes the
thresholds and decision rules used by router wrappers and the ticket pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


ROUTING_POLICY_VERSION = "routing_policy/v1"

SLM_DOMINANT_LOW_CONFIDENCE_THRESHOLD = 0.50
SLM_DOMINANT_WEAK_RETRIEVAL_THRESHOLD = 0.10
SLM_DOMINANT_SHORT_ANSWER_WORDS = 10
LONG_TICKET_THRESHOLD = 512

RISK_LOW = 0.30
RISK_HIGH = 0.70

HARD_CATEGORIES = {"security", "benefits", "legal", "compliance"}


def get_confidence_bucket(confidence: float) -> str:
    """Map a numeric confidence into low/medium/high buckets."""
    if confidence >= 0.80:
        return "high"
    if confidence >= SLM_DOMINANT_LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _normalize_category(category: str) -> str:
    return str(category or "").strip().lower()


def _normalize_model_name(model_name: Optional[str], fallback: str) -> str:
    value = str(model_name or "").strip()
    return value or fallback


@dataclass(slots=True)
class RoutingDecision:
    """Serializable routing decision returned by the canonical policy."""

    router_mode: str
    initial_model: str
    final_model: str
    escalated: bool
    escalation_reasons: List[str] = field(default_factory=list)
    classifier_confidence: float = 0.0
    confidence_bucket: str = "low"
    retrieval_score: Optional[float] = None
    routing_policy_version: str = ROUTING_POLICY_VERSION
    router_confidence_score: float = 0.0
    router_decision_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _slm_dominant_reasons(
    *,
    text: str,
    category: str,
    classifier_confidence: float,
    retrieval_score: Optional[float],
    final_answer: str,
    resolution_steps_count: int,
    schema_valid: bool,
    force_llm: bool,
) -> List[str]:
    reasons: List[str] = []
    if force_llm:
        reasons.append("force_llm")
    if len(text) >= LONG_TICKET_THRESHOLD:
        reasons.append("long_ticket")
    if _normalize_category(category) in HARD_CATEGORIES:
        reasons.append("hard_category")
    if classifier_confidence < SLM_DOMINANT_LOW_CONFIDENCE_THRESHOLD:
        reasons.append("low_confidence")
    if retrieval_score is not None and retrieval_score < SLM_DOMINANT_WEAK_RETRIEVAL_THRESHOLD:
        reasons.append("weak_kb")
    if not schema_valid:
        reasons.append("schema_invalid")
    if not final_answer.strip():
        reasons.append("no_answer")
    if resolution_steps_count <= 0:
        reasons.append("no_steps")
    answer_lower = final_answer.lower()
    if any(
        phrase in answer_lower
        for phrase in ("as an ai", "as a language model", "i cannot", "i'm unable", "i am unable")
    ):
        reasons.append("ai_disclaimer")
    if final_answer.strip() and len(final_answer.split()) < SLM_DOMINANT_SHORT_ANSWER_WORDS:
        reasons.append("short_answer")
    return reasons


def _slm_dominant_score(
    *,
    text: str,
    category: str,
    classifier_confidence: float,
    retrieval_score: Optional[float],
) -> float:
    score = max(0.0, min(1.0, 1.0 - classifier_confidence))
    if len(text) >= LONG_TICKET_THRESHOLD:
        score += 0.10
    if _normalize_category(category) in HARD_CATEGORIES:
        score += 0.20
    if retrieval_score is not None and retrieval_score < SLM_DOMINANT_WEAK_RETRIEVAL_THRESHOLD:
        score += 0.10
    return max(0.0, min(1.0, score))


def should_escalate_heuristic(
    text: str,
    category: str = "",
    classifier_confidence: Optional[float] = None,
    retrieval_score: Optional[float] = None,
) -> Tuple[bool, str, float]:
    """Compatibility helper for router strategy tests."""
    confidence = 0.5 if classifier_confidence is None else float(classifier_confidence)
    reasons = _slm_dominant_reasons(
        text=text,
        category=category,
        classifier_confidence=confidence,
        retrieval_score=retrieval_score,
        final_answer="sufficient answer",
        resolution_steps_count=1,
        schema_valid=True,
        force_llm=False,
    )
    reasons = [
        reason for reason in reasons if reason not in {"no_answer", "no_steps", "short_answer"}
    ]
    score = _slm_dominant_score(
        text=text,
        category=category,
        classifier_confidence=confidence,
        retrieval_score=retrieval_score,
    )
    if reasons:
        return True, f"escalate: {' + '.join(reasons)}", score
    return False, "stay_on_slm: heuristic_safe", score


def risk_score(
    text: str,
    category: str = "",
    classifier_confidence: float = 0.5,
) -> Tuple[float, str]:
    """Return the centralized hybrid-specialist risk score and explanation."""
    risk = max(0.0, min(1.0, 1.0 - classifier_confidence))
    reasons: List[str] = []
    if _normalize_category(category) in HARD_CATEGORIES:
        risk += 0.20
        reasons.append("hard_category")
    if len(text) > LONG_TICKET_THRESHOLD:
        risk += 0.10
        reasons.append("long_ticket")
    risk = max(0.0, min(1.0, risk))
    if risk <= RISK_LOW:
        reasons.append("low_risk")
    elif risk >= RISK_HIGH:
        reasons.append("high_risk")
    else:
        reasons.append("moderate_risk")
    return risk, "+".join(reasons)


def build_routing_decision(
    *,
    router_mode: str,
    text: str,
    base_model_name: str,
    escalation_model_name: Optional[str] = None,
    category: str = "",
    classifier_confidence: float = 0.5,
    retrieval_score: Optional[float] = None,
    final_answer: str = "",
    resolution_steps_count: int = 0,
    schema_valid: bool = True,
    force_llm: bool = False,
) -> RoutingDecision:
    """Return the canonical routing decision for a ticket run."""
    confidence_bucket = get_confidence_bucket(classifier_confidence)
    default_base_model = "llm" if router_mode == "llm_only" else "slm"
    normalized_base = _normalize_model_name(base_model_name, default_base_model)
    normalized_escalation = _normalize_model_name(escalation_model_name, "llm")

    if router_mode == "slm_only":
        return RoutingDecision(
            router_mode=router_mode,
            initial_model=normalized_base,
            final_model=normalized_base,
            escalated=False,
            escalation_reasons=[],
            classifier_confidence=classifier_confidence,
            confidence_bucket=confidence_bucket,
            retrieval_score=retrieval_score,
            router_confidence_score=classifier_confidence,
            router_decision_reason="slm_only: always_use_slm",
        )

    if router_mode == "llm_only":
        return RoutingDecision(
            router_mode=router_mode,
            initial_model=normalized_base,
            final_model=normalized_base,
            escalated=False,
            escalation_reasons=[],
            classifier_confidence=classifier_confidence,
            confidence_bucket=confidence_bucket,
            retrieval_score=retrieval_score,
            router_confidence_score=1.0,
            router_decision_reason="llm_only: always_use_llm",
        )

    if router_mode == "slm_dominant":
        reasons = _slm_dominant_reasons(
            text=text,
            category=category,
            classifier_confidence=classifier_confidence,
            retrieval_score=retrieval_score,
            final_answer=final_answer,
            resolution_steps_count=resolution_steps_count,
            schema_valid=schema_valid,
            force_llm=force_llm,
        )
        score = _slm_dominant_score(
            text=text,
            category=category,
            classifier_confidence=classifier_confidence,
            retrieval_score=retrieval_score,
        )
        escalated = bool(reasons)
        return RoutingDecision(
            router_mode=router_mode,
            initial_model=normalized_base,
            final_model=normalized_escalation if escalated else normalized_base,
            escalated=escalated,
            escalation_reasons=reasons,
            classifier_confidence=classifier_confidence,
            confidence_bucket=confidence_bucket,
            retrieval_score=retrieval_score,
            router_confidence_score=score,
            router_decision_reason=(
                f"slm_dominant: escalate ({'+'.join(reasons)})"
                if escalated
                else "slm_dominant: stay_on_slm"
            ),
        )

    if router_mode == "hybrid_specialist":
        risk, reason = risk_score(
            text, category=category, classifier_confidence=classifier_confidence
        )
        if force_llm or risk >= RISK_HIGH:
            return RoutingDecision(
                router_mode=router_mode,
                initial_model=normalized_escalation if force_llm else normalized_base,
                final_model=normalized_escalation,
                escalated=True,
                escalation_reasons=["force_llm"] if force_llm else reason.split("+"),
                classifier_confidence=classifier_confidence,
                confidence_bucket=confidence_bucket,
                retrieval_score=retrieval_score,
                router_confidence_score=risk,
                router_decision_reason="hybrid_specialist: llm_path",
            )
        if risk <= RISK_LOW:
            return RoutingDecision(
                router_mode=router_mode,
                initial_model=normalized_base,
                final_model=normalized_base,
                escalated=False,
                escalation_reasons=reason.split("+"),
                classifier_confidence=classifier_confidence,
                confidence_bucket=confidence_bucket,
                retrieval_score=retrieval_score,
                router_confidence_score=risk,
                router_decision_reason="hybrid_specialist: slm_path",
            )
        return RoutingDecision(
            router_mode=router_mode,
            initial_model=normalized_base,
            final_model=normalized_escalation,
            escalated=True,
            escalation_reasons=reason.split("+"),
            classifier_confidence=classifier_confidence,
            confidence_bucket=confidence_bucket,
            retrieval_score=retrieval_score,
            router_confidence_score=risk,
            router_decision_reason="hybrid_specialist: mixed_path",
        )

    return RoutingDecision(
        router_mode=router_mode,
        initial_model=normalized_base,
        final_model=normalized_base,
        escalated=False,
        escalation_reasons=[],
        classifier_confidence=classifier_confidence,
        confidence_bucket=confidence_bucket,
        retrieval_score=retrieval_score,
        router_confidence_score=classifier_confidence,
        router_decision_reason=f"{router_mode}: passthrough",
    )


__all__ = [
    "HARD_CATEGORIES",
    "LONG_TICKET_THRESHOLD",
    "RISK_HIGH",
    "RISK_LOW",
    "ROUTING_POLICY_VERSION",
    "RoutingDecision",
    "SLM_DOMINANT_LOW_CONFIDENCE_THRESHOLD",
    "SLM_DOMINANT_SHORT_ANSWER_WORDS",
    "SLM_DOMINANT_WEAK_RETRIEVAL_THRESHOLD",
    "build_routing_decision",
    "get_confidence_bucket",
    "risk_score",
    "should_escalate_heuristic",
]
