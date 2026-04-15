"""Canonical routing-policy tests."""

from RouterGym.routing.policy import ROUTING_POLICY_VERSION, build_routing_decision


def test_llm_only_policy_always_uses_llm() -> None:
    decision = build_routing_decision(
        router_mode="llm_only",
        text="complex payroll issue",
        base_model_name="llm1",
        classifier_confidence=0.2,
    )
    assert decision.initial_model == "llm1"
    assert decision.final_model == "llm1"
    assert decision.escalated is False
    assert decision.routing_policy_version == ROUTING_POLICY_VERSION


def test_slm_only_policy_always_uses_slm() -> None:
    decision = build_routing_decision(
        router_mode="slm_only",
        text="reset my password",
        base_model_name="slm1",
        classifier_confidence=0.9,
    )
    assert decision.initial_model == "slm1"
    assert decision.final_model == "slm1"
    assert decision.escalated is False
    assert decision.escalation_reasons == []


def test_slm_dominant_policy_easy_case_stays_on_slm() -> None:
    decision = build_routing_decision(
        router_mode="slm_dominant",
        text="reset my laptop password",
        base_model_name="slm1",
        escalation_model_name="llm1",
        category="Hardware",
        classifier_confidence=0.92,
        retrieval_score=0.4,
        final_answer="Use the self-service reset flow and confirm login works again.",
        resolution_steps_count=3,
        schema_valid=True,
    )
    assert decision.final_model == "slm1"
    assert decision.escalated is False
    assert decision.escalation_reasons == []


def test_slm_dominant_policy_hard_case_escalates() -> None:
    decision = build_routing_decision(
        router_mode="slm_dominant",
        text="security breach " + ("x" * 600),
        base_model_name="slm1",
        escalation_model_name="llm1",
        category="security",
        classifier_confidence=0.2,
        retrieval_score=0.01,
        final_answer="short",
        resolution_steps_count=0,
        schema_valid=False,
    )
    assert decision.final_model == "llm1"
    assert decision.escalated is True
    assert "low_confidence" in decision.escalation_reasons
    assert "weak_kb" in decision.escalation_reasons


def test_hybrid_specialist_policy_uses_centralized_risk_logic() -> None:
    low_risk = build_routing_decision(
        router_mode="hybrid_specialist",
        text="simple hr question",
        base_model_name="slm1",
        escalation_model_name="llm1",
        category="HR Support",
        classifier_confidence=0.95,
    )
    high_risk = build_routing_decision(
        router_mode="hybrid_specialist",
        text="critical security breach" + ("x" * 600),
        base_model_name="slm1",
        escalation_model_name="llm1",
        category="security",
        classifier_confidence=0.1,
    )
    assert low_risk.final_model == "slm1"
    assert low_risk.escalated is False
    assert high_risk.final_model == "llm1"
    assert high_risk.escalated is True
