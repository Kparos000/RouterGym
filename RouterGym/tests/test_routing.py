"""Routing tests."""

from RouterGym.routing.llm_first import LLMFirstRouter
from RouterGym.routing.slm_only import SLMOnlyRouter
from RouterGym.routing.slm_dominant import SLMDominantRouter
from RouterGym.routing.hybrid_specialist import HybridSpecialistRouter


def test_llm_first_router() -> None:
    router = LLMFirstRouter()
    result = router.route({"text": "hello"})
    assert result["strategy"] == "llm_first"
    assert "target_model" in result
    assert "final_answer" in result["final_output"]


def test_slm_only_router() -> None:
    router = SLMOnlyRouter()
    result = router.route({"text": "hello"})
    assert result["strategy"] == "slm_only"
    assert result["target_model"] == "slm"
    assert result["model_used"] == "slm"
    assert result["escalated"] is False


def test_slm_dominant_router() -> None:
    router = SLMDominantRouter()
    result = router.route({"text": "hello"})
    assert result["strategy"] == "slm_dominant"
    # without models provided, fallback llm branch won't trigger
    assert result["target_model"] in {"slm", "llm"}
    assert "final_answer" in result["final_output"]


def test_hybrid_router_category_routing() -> None:
    router = HybridSpecialistRouter()
    result = router.route({"text": "hello", "category": "access"})
    assert result["strategy"] == "hybrid_specialist"
    assert result["target_model"] in {"slm", "llm"}
    assert "final_answer" in result["final_output"]


def test_routing_metadata_fields_present() -> None:
    router = SLMDominantRouter()
    result = router.route(
        {"text": "reset my password", "classifier_confidence": 0.9, "category": "Access"}
    )
    for key in (
        "router_mode",
        "initial_model",
        "final_model",
        "escalated",
        "escalation_reasons",
        "classifier_confidence",
        "confidence_bucket",
        "retrieval_score",
        "routing_policy_version",
    ):
        assert key in result
