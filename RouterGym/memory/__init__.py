"""Memory package and registry with lazy backend loading.

The benchmark only needs a backend class when a memory mode is requested.
Delaying imports keeps unrelated module imports from pulling dense-retrieval
dependencies into test collection or simple CLI startup.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Type

from RouterGym.memory.base import MemoryRetrieval

MEMORY_MODES = ["none", "rag_dense", "rag_bm25", "rag_hybrid"]

_LEGACY_MAP = {
    "rag": "rag_dense",
    "salience": "rag_hybrid",
}

_MEMORY_TARGETS = {
    "none": ("RouterGym.memory.none", "NoneMemory"),
    "rag_dense": ("RouterGym.memory.rag", "DenseRAGMemory"),
    "rag_bm25": ("RouterGym.memory.bm25", "BM25Memory"),
    "rag_hybrid": ("RouterGym.memory.hybrid", "HybridRAGMemory"),
    "RAGMemory": ("RouterGym.memory.rag", "RAGMemory"),
    "DenseRAGMemory": ("RouterGym.memory.rag", "DenseRAGMemory"),
    "BM25Memory": ("RouterGym.memory.bm25", "BM25Memory"),
    "HybridRAGMemory": ("RouterGym.memory.hybrid", "HybridRAGMemory"),
    "NoneMemory": ("RouterGym.memory.none", "NoneMemory"),
    "SalienceGatedMemory": ("RouterGym.memory.salience", "SalienceGatedMemory"),
}

__all__ = [
    "BM25Memory",
    "DenseRAGMemory",
    "HybridRAGMemory",
    "MEMORY_MODES",
    "MemoryRetrieval",
    "NoneMemory",
    "RAGMemory",
    "SalienceGatedMemory",
    "get_memory_class",
    "resolve_memory_mode",
]


def _load_memory_target(name: str) -> Type:
    module_name, attr_name = _MEMORY_TARGETS[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


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
    if canonical not in _MEMORY_TARGETS:
        return None
    return _load_memory_target(canonical)


def __getattr__(name: str):
    if name in _MEMORY_TARGETS:
        return _load_memory_target(name)
    raise AttributeError(name)
