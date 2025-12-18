"""Memory package and registry."""

import warnings
from typing import Dict, Type

from RouterGym.memory.base import MemoryRetrieval
from RouterGym.memory.none import NoneMemory
from RouterGym.memory.rag import RAGMemory, DenseRAGMemory
from RouterGym.memory.bm25 import BM25Memory
from RouterGym.memory.hybrid import HybridRAGMemory
from RouterGym.memory.salience import SalienceGatedMemory

# Public memory modes
MEMORY_MODES = ["none", "rag_dense", "rag_bm25", "rag_hybrid"]

_MEMORY_REGISTRY: Dict[str, Type] = {
    "none": NoneMemory,
    "rag_dense": DenseRAGMemory,
    "rag_bm25": BM25Memory,
    "rag_hybrid": HybridRAGMemory,
}

_LEGACY_MAP = {
    "rag": "rag_dense",
    "salience": "rag_hybrid",
}


def resolve_memory_mode(name: str) -> str:
    """Map legacy names to canonical modes."""
    if name in _LEGACY_MAP:
        canonical = _LEGACY_MAP[name]
        warnings.warn(f"Memory mode '{name}' is deprecated; using '{canonical}' instead.", RuntimeWarning)
        return canonical
    return name


def get_memory_class(name: str):
    """Return the memory class for a given mode name."""
    canonical = resolve_memory_mode(name)
    return _MEMORY_REGISTRY.get(canonical)


__all__ = [
    "MemoryRetrieval",
    "NoneMemory",
    "RAGMemory",
    "DenseRAGMemory",
    "BM25Memory",
    "HybridRAGMemory",
    "SalienceGatedMemory",
    "MEMORY_MODES",
    "get_memory_class",
    "resolve_memory_mode",
]
