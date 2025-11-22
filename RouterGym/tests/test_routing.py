"""Routing tests."""

from RouterGym.routing.llm_first import LLMFirstRouter
from RouterGym.routing.slm_dominant import SLMDominantRouter
from RouterGym.routing.hybrid_specialist import HybridSpecialistRouter


def test_llm_first_router() -> None:
    router = LLMFirstRouter()
    result = router.route({"text": "hello"})
    assert result["strategy"] == "llm_first"
    assert "target_model" in result
    assert "final_answer" in result["final_output"]


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
