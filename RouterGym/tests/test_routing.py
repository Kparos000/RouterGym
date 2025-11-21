"""Routing tests stubs."""

from RouterGym.routing.llm_first import LLMFirstRouter
from RouterGym.routing.slm_dominant import SLMDominantRouter
from RouterGym.routing.hybrid_specialist import HybridSpecialistRouter


def test_llm_first_router_stub() -> None:
    """Placeholder test for LLM-first router."""
    assert LLMFirstRouter().route("prompt")["strategy"] == "llm_first"


def test_slm_dominant_router_stub() -> None:
    """Placeholder test for SLM-dominant router."""
    assert SLMDominantRouter().route("prompt")["strategy"] == "slm_dominant"


def test_hybrid_router_stub() -> None:
    """Placeholder test for hybrid specialist router."""
    assert HybridSpecialistRouter().route("prompt")["strategy"] == "hybrid_specialist"
