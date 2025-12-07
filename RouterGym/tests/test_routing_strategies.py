"""Router strategy helper tests (heuristics and confidence-based risk)."""

from RouterGym.routing.slm_dominant import should_escalate_heuristic
from RouterGym.routing.hybrid_specialist import risk_score, RISK_LOW, RISK_HIGH


def test_slm_dominant_escalates_on_hard_long_low_conf() -> None:
    text = "security incident with very long description " + ("x" * 600)
    escalate, reason, score = should_escalate_heuristic(text, category="security", classifier_confidence=0.2)
    assert escalate is True
    assert "hard_category" in reason or "long_ticket" in reason
    assert score >= 0.8  # inverted confidence for escalation


def test_slm_dominant_stays_on_slm_for_easy_high_conf() -> None:
    text = "reset my laptop password"
    escalate, reason, score = should_escalate_heuristic(text, category="hardware", classifier_confidence=0.9)
    assert escalate is False
    assert "stay_on_slm" in reason
    assert score <= 0.9


def test_hybrid_risk_low_stays_slm() -> None:
    text = "simple hr question"
    risk, reason = risk_score(text, category="hr_support", classifier_confidence=0.95)
    assert risk <= RISK_LOW
    assert "low" in reason


def test_hybrid_risk_high_escalates() -> None:
    text = "critical security breach" + ("x" * 600)
    risk, reason = risk_score(text, category="security", classifier_confidence=0.1)
    assert risk >= RISK_HIGH
    assert "high" in reason or "hard_category" in reason
